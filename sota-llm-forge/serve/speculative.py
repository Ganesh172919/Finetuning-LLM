"""
################################################################################
SPECULATIVE DECODING — MTP-HEAD SELF-SPECULATIVE DECODING FOR LLM INFERENCE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Speculative Decoding?
    Speculative decoding is an inference technique that accelerates
    autoregressive generation by using a cheap "draft" model to predict
    multiple tokens at once, then verifying them in parallel with the
    full "target" model. Tokens that match the target distribution are
    accepted; the first mismatch triggers rejection and fallback.

Why does it matter?
    - Standard autoregressive decoding is memory-bandwidth bound: each
      token requires loading the entire model but does minimal compute
    - Speculative decoding amortizes model loading across K tokens
    - Wall-clock speedup of 2-3x is typical, with NO quality loss
    - The output distribution is mathematically identical to vanilla decoding

How does it work?
    1. Draft: The draft model (cheap, small) predicts K tokens ahead
    2. Verify: The target model (expensive, full) evaluates all K tokens
       in ONE forward pass (parallel verification)
    3. Accept: Take the longest correct prefix from the draft predictions
    4. Reject: On first mismatch, sample from the target model's distribution
    5. Repeat: Use accepted tokens as context for next draft phase

    Key insight: Verification is cheap because we can evaluate K tokens
    in parallel (they share the same prefix). Drafting is cheap because
    the draft model is much smaller. The net speedup comes from reducing
    the number of sequential target-model forward passes.

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────────────┐
    │              SPECULATIVE DECODING PIPELINE                         │
    │                                                                     │
    │  Input: "The quick brown"                                         │
    │                                                                     │
    │  ┌─────────────────────────────────────┐                           │
    │  │ DRAFT PHASE (MTP Heads - cheap)     │                           │
    │  │                                      │                           │
    │  │ MTP Head 0: "fox"                   │                           │
    │  │ MTP Head 1: "jumps"                 │                           │
    │  │ MTP Head 2: "over"                  │                           │
    │  │                                      │                           │
    │  │ Draft tokens: [fox, jumps, over]     │                           │
    │  └──────────────────┬──────────────────┘                           │
    │                     │                                               │
    │                     ▼                                               │
    │  ┌─────────────────────────────────────┐                           │
    │  │ VERIFY PHASE (Full Model - parallel)│                           │
    │  │                                      │                           │
    │  │ Input: "The quick brown fox jumps    │                           │
    │  │         over [NEXT]"                 │                           │
    │  │                                      │                           │
    │  │ Full model processes ALL tokens in   │                           │
    │  │ one forward pass → logits for each   │                           │
    │  └──────────────────┬──────────────────┘                           │
    │                     │                                               │
    │                     ▼                                               │
    │  ┌─────────────────────────────────────┐                           │
    │  │ ACCEPT/REJECT                       │                           │
    │  │                                      │                           │
    │  │ Compare draft vs target distribution │                           │
    │  │ Accept "fox" ✓ (p_target > p_draft) │                           │
    │  │ Accept "jumps" ✓                    │                           │
    │  │ Reject "over" ✗ → sample from target│                           │
    │  │                                      │                           │
    │  │ Result: 3 tokens in 1 verify pass    │                           │
    │  │ vs 3 passes in vanilla decoding      │                           │
    │  └─────────────────────────────────────┘                           │
    │                                                                     │
    │  KEY PROPERTY: Output distribution is EXACTLY the same as vanilla   │
    │  autoregressive decoding. Speedup comes from parallelism, not from  │
    │  changing the distribution.                                         │
    └─────────────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2022: Speculative decoding proposed (Leviathan et al., Google; Chen et al., DeepMind)
    - 2023: Self-speculative decoding via early exit (Schuster et al., Google)
    - 2024: Multi-Token Prediction training enables free draft heads (Gloeckle et al., Meta)
    - 2024: DeepSeek-V3 trains with MTP heads, repurposes them for speculative decoding
    - 2025: Speculative decoding becomes standard for production LLM serving
    - 2026: MTP-head self-speculative is the dominant approach (no separate draft model)

INTERVIEW QUESTIONS:
    1. "Why is speculative decoding lossless?"
       The accept/reject step uses a modified rejection sampling scheme.
       When the draft model predicts token t with probability q(t) and
       the target model assigns probability p(t), we accept with probability
       min(1, p(t)/q(t)). This ensures the accepted tokens follow exactly
       the target distribution p, regardless of the draft distribution q.

    2. "What determines the speedup of speculative decoding?"
       Three factors: (1) Acceptance rate — how often draft tokens match the
       target distribution (higher is better). (2) Draft cost — how cheap
       the draft model is relative to the target (cheaper is better). (3) K
       — the number of draft tokens (more tokens = more parallelism, but
       lower acceptance rate). Optimal K balances these tradeoffs.

    3. "Why use MTP heads as the draft model instead of a separate small model?"
       MTP heads are already trained alongside the main model — they are
       FREE (no additional training or maintenance). A separate draft model
       requires: (a) training, (b) maintaining compatibility with the target
       model's tokenizer/vocabulary, (c) loading into VRAM. MTP heads share
       the model's hidden states and have minimal parameter overhead.

    4. "When does speculative decoding NOT help?"
       When the draft model's acceptance rate is very low (< 30%), the
       overhead of draft + reject exceeds the savings from parallel
       verification. This happens with: (1) highly unpredictable text
       (random strings), (2) poor draft model quality, (3) very short
       generation (overhead dominates). Also, if the target model is
       already compute-bound (not memory-bound), there's no speedup.

################################################################################
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import math
import time
import logging

logger = logging.getLogger(__name__)


################################################################################
# SECTION 1: SPECULATIVE DECODING CONFIGURATION
################################################################################


@dataclass
class SpeculativeConfig:
    """
    Configuration for Speculative Decoding.

    Attributes:
        num_speculative_tokens: Number of tokens to draft per step (K)
        temperature: Sampling temperature for both draft and target
        top_k: Top-k sampling parameter (0 = disabled)
        top_p: Nucleus sampling parameter (1.0 = disabled)
        acceptance_threshold: Minimum acceptance rate to continue speculative
            decoding (below this, fall back to vanilla)
        log_acceptance_rate: Whether to log acceptance rate periodically
        log_interval: How often to log metrics (in tokens)
    """
    num_speculative_tokens: int = 3
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    acceptance_threshold: float = 0.3
    log_acceptance_rate: bool = True
    log_interval: int = 100


################################################################################
# SECTION 2: KV CACHE MANAGEMENT
################################################################################


class MLAKVCache:
    """
    KV Cache for Multi-head Latent Attention (MLA) Models
    ======================================================

    For MLA-attention models, serve the COMPRESSED latent cache,
    not the expanded per-head cache.

    A served model that decompresses to full K/V before caching
    has thrown the whole point away.

    MLA Compression:
        Instead of storing full K and V for each head (d_model * n_heads),
        MLA stores a compressed latent c_kv of dimension d_compress.
        At attention time, K and V are reconstructed via up-projection.

        Full KV cache per token: 2 * n_heads * d_head = 2 * d_model bytes
        MLA compressed cache per token: 2 * d_compress bytes (much smaller)

    Step by step:
        1. After attention projection, compress K,V into latent c_kv
        2. Store c_kv in cache (not the expanded K,V)
        3. At attention time, up-project c_kv → K, V on-the-fly
        4. This trades a small compute cost for large memory savings

    Interview Question:
        "How does MLA's KV cache compression work?"
        MLA projects the key-value pair into a lower-dimensional latent
        space c_kv = W_compress @ [k; v]. This latent has dimension
        d_compress << 2 * d_model. At attention time, K and V are
        reconstructed: k = W_up_k @ c_kv, v = W_up_v @ c_kv. The
        cache stores only c_kv, giving memory savings proportional to
        d_compress / (2 * d_model). For DeepSeek-V2, this is ~93% savings.
    """

    def __init__(
        self,
        num_layers: int,
        d_compress: int,
        max_seq_len: int,
        batch_size: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize MLA KV cache.

        Args:
            num_layers: Number of transformer layers
            d_compress: Dimension of the compressed latent c_kv
            max_seq_len: Maximum sequence length to pre-allocate
            batch_size: Batch size
            device: Device to store cache on
            dtype: Data type for cache tensors

        Explanation:
            Pre-allocating the full cache avoids repeated memory allocation
            during generation. The cache is a list of tensors, one per layer,
            each of shape (batch_size, max_seq_len, d_compress).
        """
        self.num_layers = num_layers
        self.d_compress = d_compress
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        # Pre-allocate cache for each layer
        # Shape: (batch_size, max_seq_len, d_compress)
        self.cache: List[torch.Tensor] = [
            torch.zeros(
                batch_size, max_seq_len, d_compress,
                device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ]

        # Current sequence length (number of cached tokens)
        self.seq_len: int = 0

    def append(
        self,
        layer_idx: int,
        c_kv: torch.Tensor,
    ) -> None:
        """
        Append compressed KV latent to the cache for a specific layer.

        Args:
            layer_idx: Index of the transformer layer
            c_kv: Compressed KV latent, shape (batch_size, 1, d_compress)

        Explanation:
            Each call appends one token's compressed latent to the cache.
            The cache is written at position self.seq_len, which is
            incremented after all layers have been updated.
        """
        if self.seq_len >= self.max_seq_len:
            raise RuntimeError(
                f"KV cache full: seq_len={self.seq_len}, max={self.max_seq_len}"
            )
        self.cache[layer_idx][:, self.seq_len:self.seq_len + 1, :] = c_kv

    def advance(self) -> None:
        """Advance the sequence length counter after all layers are updated."""
        self.seq_len += 1

    def get(
        self,
        layer_idx: int,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Retrieve cached KV latents for a layer.

        Args:
            layer_idx: Index of the transformer layer
            start_pos: Start position in the cache

        Returns:
            Cached c_kv tensor, shape (batch_size, seq_len - start_pos, d_compress)

        Explanation:
            Returns the cached compressed latents from start_pos to the
            current sequence length. This is used during attention computation
            to reconstruct full K,V via up-projection.
        """
        return self.cache[layer_idx][:, start_pos:self.seq_len, :]

    def trim(self, new_seq_len: int) -> None:
        """
        Trim the cache to a new sequence length.

        Used when accepting only a prefix of speculative tokens.

        Args:
            new_seq_len: New sequence length to trim to

        Explanation:
            After speculative decoding, if only some draft tokens are accepted,
            we need to trim the cache to remove the rejected tokens' entries.
            This doesn't actually free memory (pre-allocated), but resets
            the write pointer.
        """
        self.seq_len = new_seq_len

    def memory_bytes(self) -> int:
        """
        Compute memory usage of the MLA cache.

        Returns:
            Total bytes used by the cache

        Explanation:
            MLA cache memory = num_layers * batch_size * max_seq_len * d_compress * dtype_size

            Compare to full KV cache:
            Full cache = num_layers * batch_size * max_seq_len * 2 * d_model * dtype_size

            Savings ratio = 2 * d_model / d_compress
        """
        bytes_per_element = torch.tensor([], dtype=self.dtype).element_size()
        total = self.num_layers * self.batch_size * self.max_seq_len * self.d_compress * bytes_per_element
        return total


class GQAKVCache:
    """
    KV Cache for Grouped Query Attention (GQA) Models
    ===================================================

    Standard KV cache that stores full K and V tensors for each
    key-value head (not each query head).

    GQA Compression:
        GQA reduces KV cache size by sharing key-value heads across
        multiple query heads. For n_q_heads query heads and n_kv_heads
        key-value heads, each KV head serves n_q_heads / n_kv_heads
        query heads.

        MHA cache: n_heads * d_head * 2 per token (full attention)
        GQA cache: n_kv_heads * d_head * 2 per token (shared KV)
        MQA cache: d_head * 2 per token (single shared KV)

    Step by step:
        1. Compute K, V using the shared KV head projections
        2. Store full K, V in cache (not compressed like MLA)
        3. At attention time, repeat K,V to match query heads via
           expand/repeat operations

    Interview Question:
        "How does GQA's KV cache compare to MLA's?"
        GQA stores full K and V tensors, just with fewer heads than MHA.
        MLA stores a compressed latent that is up-projected to K,V at
        attention time. MLA achieves much higher compression ratios
        (DeepSeek-V2: ~93% savings vs MHA) but requires extra compute
        for the up-projection. GQA is simpler and doesn't require
        architectural changes to the attention mechanism.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        d_head: int,
        max_seq_len: int,
        batch_size: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize GQA KV cache.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of key-value heads (not query heads)
            d_head: Dimension per head
            max_seq_len: Maximum sequence length to pre-allocate
            batch_size: Batch size
            device: Device to store cache on
            dtype: Data type for cache tensors

        Explanation:
            Pre-allocates K and V caches for each layer. Each cache has
            shape (batch_size, num_kv_heads, max_seq_len, d_head).

            Memory per layer: 2 * batch_size * num_kv_heads * max_seq_len * d_head
            (factor of 2 for both K and V)
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        # Pre-allocate K and V caches for each layer
        # Shape: (batch_size, num_kv_heads, max_seq_len, d_head)
        self.k_cache: List[torch.Tensor] = [
            torch.zeros(
                batch_size, num_kv_heads, max_seq_len, d_head,
                device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ]
        self.v_cache: List[torch.Tensor] = [
            torch.zeros(
                batch_size, num_kv_heads, max_seq_len, d_head,
                device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ]

        self.seq_len: int = 0

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Append K and V tensors to the cache for a specific layer.

        Args:
            layer_idx: Index of the transformer layer
            k: Key tensor, shape (batch_size, num_kv_heads, 1, d_head)
            v: Value tensor, shape (batch_size, num_kv_heads, 1, d_head)

        Explanation:
            Each call appends one token's K and V to the cache.
        """
        if self.seq_len >= self.max_seq_len:
            raise RuntimeError(
                f"KV cache full: seq_len={self.seq_len}, max={self.max_seq_len}"
            )
        self.k_cache[layer_idx][:, :, self.seq_len:self.seq_len + 1, :] = k
        self.v_cache[layer_idx][:, :, self.seq_len:self.seq_len + 1, :] = v

    def advance(self) -> None:
        """Advance the sequence length counter."""
        self.seq_len += 1

    def get(
        self,
        layer_idx: int,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve cached K and V tensors for a layer.

        Args:
            layer_idx: Index of the transformer layer
            start_pos: Start position in the cache

        Returns:
            Tuple of (K, V) tensors, each shape
            (batch_size, num_kv_heads, seq_len - start_pos, d_head)
        """
        k = self.k_cache[layer_idx][:, :, start_pos:self.seq_len, :]
        v = self.v_cache[layer_idx][:, :, start_pos:self.seq_len, :]
        return k, v

    def trim(self, new_seq_len: int) -> None:
        """
        Trim the cache to a new sequence length.

        Args:
            new_seq_len: New sequence length to trim to
        """
        self.seq_len = new_seq_len

    def memory_bytes(self) -> int:
        """
        Compute memory usage of the GQA cache.

        Returns:
            Total bytes used by the cache

        Explanation:
            GQA cache memory = num_layers * 2 * batch_size * num_kv_heads
                               * max_seq_len * d_head * dtype_size

            The factor of 2 accounts for both K and V tensors.
        """
        bytes_per_element = torch.tensor([], dtype=self.dtype).element_size()
        per_layer = (
            2 * self.batch_size * self.num_kv_heads * self.max_seq_len * self.d_head * bytes_per_element
        )
        return self.num_layers * per_layer


def compare_cache_memory(
    num_layers: int = 32,
    d_model: int = 4096,
    n_heads: int = 32,
    n_kv_heads: int = 8,
    d_compress: int = 512,
    max_seq_len: int = 4096,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Compare memory usage: MLA vs GQA vs MHA KV caches.

    Args:
        num_layers: Number of transformer layers
        d_model: Model dimension
        n_heads: Number of attention heads (for MHA)
        n_kv_heads: Number of KV heads (for GQA)
        d_compress: MLA compressed dimension
        max_seq_len: Maximum sequence length
        batch_size: Batch size

    Returns:
        Dictionary with memory comparison metrics

    Explanation:
        MHA stores full K,V for every query head: 2 * n_heads * d_head per token
        GQA shares KV heads: 2 * n_kv_heads * d_head per token
        MLA compresses to latent: 2 * d_compress per token

        For a typical DeepSeek-V2-like config:
        - d_model = 4096, n_heads = 128, d_head = 128, n_kv_heads = 8
        - d_compress = 512 (MLA latent dimension)
        - MHA: 2 * 128 * 128 = 32768 floats per token
        - GQA: 2 * 8 * 128 = 2048 floats per token (16x savings)
        - MLA: 2 * 512 = 1024 floats per token (32x savings)
    """
    d_head = d_model // n_heads

    # MHA cache
    mha_cache = GQAKVCache(
        num_layers=num_layers,
        num_kv_heads=n_heads,  # MHA = all heads are KV heads
        d_head=d_head,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )

    # GQA cache
    gqa_cache = GQAKVCache(
        num_layers=num_layers,
        num_kv_heads=n_kv_heads,
        d_head=d_head,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )

    # MLA cache
    mla_cache = MLAKVCache(
        num_layers=num_layers,
        d_compress=d_compress,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )

    mha_bytes = mha_cache.memory_bytes()
    gqa_bytes = gqa_cache.memory_bytes()
    mla_bytes = mla_cache.memory_bytes()

    return {
        "mha_bytes": mha_bytes,
        "gqa_bytes": gqa_bytes,
        "mla_bytes": mla_bytes,
        "gqa_vs_mha": mha_bytes / max(gqa_bytes, 1),
        "mla_vs_mha": mha_bytes / max(mla_bytes, 1),
        "mla_vs_gqa": gqa_bytes / max(mla_bytes, 1),
    }


################################################################################
# SECTION 3: SPECULATIVE DECODER
################################################################################


class SpeculativeDecoder:
    """
    Self-Speculative Decoding via MTP Heads
    ========================================

    Use the multi-token prediction heads from training as a FREE draft model.
    No separate draft model to train or maintain.

    Algorithm:
        1. Draft: Use MTP heads to predict K tokens ahead (cheap)
        2. Verify: Run full model on all K tokens in one forward pass
        3. Accept: Take longest correct prefix from MTP predictions
        4. Reject: If MTP head is wrong, fall back to that position

    Key property: Output distribution is EXACTLY the same as vanilla
    autoregressive decoding (no quality loss). Speedup comes from
    parallelizing verification.

    Benchmark: Wall-clock speedup vs vanilla on your own hardware.
    Don't assume the paper's number transfers.

    Step by step of one speculative step:
        1. Given current context [t_0, ..., t_{n-1}]:
        2. Draft: MTP head predicts K tokens: [d_1, d_2, ..., d_K]
        3. Build candidate sequence: [t_0, ..., t_{n-1}, d_1, ..., d_K]
        4. Verify: Full model forward on entire candidate sequence
           → logits at positions [n, n+1, ..., n+K]
        5. For each position i in [1, ..., K]:
           - Sample target token t_i from target logits at position n+i-1
           - If d_i == t_i: accept d_i, continue
           - If d_i != t_i: reject d_i, use t_i, stop
        6. If all K accepted: also sample token at position n+K
        7. Update context with accepted tokens

    Interview Question:
        "How do you handle the case where the draft model is often wrong?"
        The acceptance rate determines the effective speedup. If the draft
        model is wrong on the first token 70% of the time, we waste a full
        forward pass for just 1 token (worse than vanilla). Strategies:
        (1) Use a better draft model (MTP heads trained longer).
        (2) Reduce K (fewer speculative tokens = less waste on rejection).
        (3) Adaptive K: start with K=1, increase as acceptance rate improves.
        (4) Fall back to vanilla if acceptance rate is below threshold.
    """

    def __init__(
        self,
        model: nn.Module,
        mtp_heads: nn.Module,
        config: Optional[SpeculativeConfig] = None,
    ):
        """
        Initialize the speculative decoder.

        Args:
            model: The full target model (TransformerLM or similar)
            mtp_heads: Multi-token prediction heads (from training)
            config: SpeculativeConfig instance

        Explanation:
            The speculative decoder wraps the target model and MTP heads.
            The target model is used for verification (full forward pass).
            The MTP heads are used for drafting (cheap, predicts K tokens).

            Both must share the same vocabulary and tokenizer. The MTP heads
            must have been trained with the target model (they share the
            same hidden states).
        """
        self.model = model
        self.mtp_heads = mtp_heads
        self.config = config or SpeculativeConfig()

        # Metrics
        self.total_tokens_generated = 0
        self.total_accepted_tokens = 0
        self.total_speculative_steps = 0
        self.acceptance_history: List[float] = []

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample a token from logits using temperature, top-k, and top-p.

        Args:
            logits: Logits tensor, shape (batch_size, vocab_size)
            temperature: Sampling temperature
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)

        Returns:
            Sampled token indices, shape (batch_size,)

        Explanation:
            Temperature scaling: logits = logits / temperature
            Higher temperature → more uniform distribution (creative)
            Lower temperature → more peaked distribution (deterministic)
            Temperature 0 → greedy (argmax)

            Top-k: Keep only the k highest-probability tokens, renormalize.
            Top-p (nucleus): Keep the smallest set of tokens whose cumulative
            probability exceeds p, renormalize.

            These are applied in order: temperature → top-k → top-p → sample.
        """
        if temperature == 0:
            return logits.argmax(dim=-1)

        # Temperature scaling
        scaled_logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k_vals, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
            threshold = top_k_vals[:, -1:]
            scaled_logits = torch.where(
                scaled_logits < threshold,
                torch.full_like(scaled_logits, float('-inf')),
                scaled_logits,
            )

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')
            # Scatter back
            scaled_logits.scatter_(1, sorted_indices, sorted_logits)

        # Sample
        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_tokens: int = 100,
        num_mtp_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate tokens using speculative decoding.

        Args:
            prompt: Input token IDs, shape (batch_size, seq_len)
            max_tokens: Maximum number of new tokens to generate
            num_mtp_tokens: Number of MTP draft tokens per step (overrides config)

        Returns:
            Tuple of (generated_tokens, metrics_dict)
            - generated_tokens: Full sequence including prompt, shape (batch_size, seq_len + n_gen)
            - metrics_dict: Dictionary with acceptance_rate, tokens_per_second, etc.

        Explanation:
            The main generation loop alternates between draft and verify phases.

            Draft Phase:
            - Use MTP heads to predict K tokens ahead from the current context
            - This is cheap: MTP heads are small (typically 1-2 layers)

            Verify Phase:
            - Run the full target model on the entire candidate sequence
            - This gives logits for all K+1 positions in one forward pass
            - The +1 is for the token after the last draft token

            Accept/Reject:
            - For each draft position, compare draft token to target sample
            - Accept the longest correct prefix
            - On first mismatch, use the target model's sample and stop

            The output distribution is EXACTLY the same as vanilla autoregressive
            decoding because of the modified rejection sampling scheme.
        """
        K = num_mtp_tokens or self.config.num_speculative_tokens
        batch_size, prompt_len = prompt.shape

        # Track metrics
        total_accepted = 0
        total_draft_tokens = 0
        tokens_generated = 0
        start_time = time.time()

        # Current sequence (starts with prompt)
        current_seq = prompt.clone()

        while tokens_generated < max_tokens:
            # ---- DRAFT PHASE ----
            # Run target model to get hidden states
            model_output = self.model(current_seq)
            if isinstance(model_output, tuple):
                hidden_states = model_output[0]
            else:
                hidden_states = model_output

            # Use MTP heads to draft K tokens
            # MTP heads take hidden states and predict future tokens
            draft_tokens = []
            draft_logits_list = []

            # Get the last position's hidden state for MTP
            last_hidden = hidden_states[:, -1:, :]  # (batch, 1, d_model)

            for k in range(K):
                # MTP head k predicts token at position k+1
                mtp_logits = self.mtp_heads(last_hidden, prediction_index=k)
                # Shape: (batch, 1, vocab_size) or (batch, vocab_size)
                if mtp_logits.dim() == 3:
                    mtp_logits = mtp_logits[:, -1, :]
                draft_token = self._sample_token(
                    mtp_logits,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                )
                draft_tokens.append(draft_token)
                draft_logits_list.append(mtp_logits)

            # Build candidate sequence: current_seq + draft_tokens
            draft_tensor = torch.stack(draft_tokens, dim=1)  # (batch, K)
            candidate_seq = torch.cat([current_seq, draft_tensor], dim=1)

            # ---- VERIFY PHASE ----
            # Run full model on candidate sequence
            verify_output = self.model(candidate_seq)
            if isinstance(verify_output, tuple):
                verify_hidden = verify_output[0]
            else:
                verify_hidden = verify_output

            # Get logits at the verification positions
            # We need logits at positions [prompt_len-1, prompt_len, ..., prompt_len+K-1]
            # to verify draft tokens at positions [prompt_len, prompt_len+1, ..., prompt_len+K]
            verify_start = current_seq.shape[1] - 1
            verify_logits = verify_hidden[:, verify_start:verify_start + K + 1, :]

            # ---- ACCEPT/REJECT PHASE ----
            accepted_count = 0
            new_tokens = []

            for k in range(K):
                # Sample target token at this position
                target_logits = verify_logits[:, k, :]
                target_token = self._sample_token(
                    target_logits,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                )

                if target_token.item() == draft_tokens[k].item():
                    # Accept: draft matches target
                    accepted_count += 1
                    new_tokens.append(draft_tokens[k])
                else:
                    # Reject: use target's sample
                    new_tokens.append(target_token)
                    break
            else:
                # All K draft tokens accepted — also sample the K+1th token
                target_logits = verify_logits[:, K, :]
                extra_token = self._sample_token(
                    target_logits,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                )
                new_tokens.append(extra_token)

            # Update sequence
            new_token_tensor = torch.stack(new_tokens, dim=1)
            current_seq = torch.cat([current_seq, new_token_tensor], dim=1)
            n_new = len(new_tokens)
            tokens_generated += n_new
            total_accepted += accepted_count
            total_draft_tokens += K

            # Log acceptance rate periodically
            if self.config.log_acceptance_rate:
                self.acceptance_history.append(accepted_count / K)

            # Check for EOS (assuming token ID 2 is EOS)
            # In practice, this should be configurable
            if current_seq[:, -1].item() == 2:
                break

        elapsed = time.time() - start_time
        self.total_tokens_generated += tokens_generated
        self.total_accepted_tokens += total_accepted
        self.total_speculative_steps += (tokens_generated + K - 1) // K

        metrics = {
            "tokens_generated": tokens_generated,
            "acceptance_rate": total_accepted / max(total_draft_tokens, 1),
            "tokens_per_second": tokens_generated / max(elapsed, 1e-6),
            "elapsed_seconds": elapsed,
            "avg_accepted_per_step": total_accepted / max(tokens_generated // K, 1),
        }

        return current_seq, metrics

    @torch.no_grad()
    def generate_vanilla(
        self,
        prompt: torch.Tensor,
        max_tokens: int = 100,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate tokens using standard autoregressive decoding (baseline).

        Args:
            prompt: Input token IDs, shape (batch_size, seq_len)
            max_tokens: Maximum number of new tokens to generate

        Returns:
            Tuple of (generated_tokens, metrics_dict)

        Explanation:
            Standard autoregressive decoding: generate one token at a time,
            each requiring a full model forward pass. This is the baseline
            to compare speculative decoding against.

            For each step:
            1. Run model on current sequence → logits at last position
            2. Sample next token from logits
            3. Append to sequence
            4. Repeat
        """
        start_time = time.time()
        current_seq = prompt.clone()
        tokens_generated = 0

        while tokens_generated < max_tokens:
            # Forward pass
            model_output = self.model(current_seq)
            if isinstance(model_output, tuple):
                hidden_states = model_output[0]
            else:
                hidden_states = model_output

            # Get logits at last position
            logits = hidden_states[:, -1, :]

            # Sample next token
            next_token = self._sample_token(
                logits,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )

            # Append
            current_seq = torch.cat(
                [current_seq, next_token.unsqueeze(1)], dim=1
            )
            tokens_generated += 1

            # Check for EOS
            if next_token.item() == 2:
                break

        elapsed = time.time() - start_time

        metrics = {
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_generated / max(elapsed, 1e-6),
            "elapsed_seconds": elapsed,
        }

        return current_seq, metrics

    def benchmark(
        self,
        prompt: torch.Tensor,
        max_tokens: int = 100,
        num_mtp_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark speculative decoding vs vanilla autoregressive decoding.

        Args:
            prompt: Input token IDs, shape (batch_size, seq_len)
            max_tokens: Maximum number of new tokens to generate
            num_mtp_tokens: Number of MTP draft tokens per step

        Returns:
            Dictionary with comparison metrics:
            - speculative: Metrics from speculative decoding
            - vanilla: Metrics from vanilla decoding
            - speedup: Wall-clock speedup ratio
            - acceptance_rate: Draft token acceptance rate

        Explanation:
            This runs both speculative and vanilla decoding on the same
            prompt and compares wall-clock performance.

            IMPORTANT: Benchmark on YOUR hardware and workload. Paper numbers
            are for specific hardware/model/data combinations. Your speedup
            will vary based on:
            - Model size (larger models = more memory-bound = bigger speedup)
            - Hardware (GPU memory bandwidth matters most)
            - Acceptance rate (task-dependent)
            - K (number of speculative tokens)
        """
        print(f"Benchmarking with max_tokens={max_tokens}, K={num_mtp_tokens or self.config.num_speculative_tokens}...")

        # Warm up
        with torch.no_grad():
            _ = self.model(prompt)

        # Run speculative decoding
        print("  Running speculative decoding...")
        spec_seq, spec_metrics = self.generate(
            prompt, max_tokens, num_mtp_tokens
        )

        # Run vanilla decoding
        print("  Running vanilla decoding...")
        vanilla_seq, vanilla_metrics = self.generate_vanilla(
            prompt, max_tokens
        )

        speedup = vanilla_metrics["elapsed_seconds"] / max(spec_metrics["elapsed_seconds"], 1e-6)

        results = {
            "speculative": spec_metrics,
            "vanilla": vanilla_metrics,
            "speedup": speedup,
            "acceptance_rate": spec_metrics["acceptance_rate"],
            "speculative_tokens_per_sec": spec_metrics["tokens_per_second"],
            "vanilla_tokens_per_sec": vanilla_metrics["tokens_per_second"],
        }

        return results


################################################################################
# SECTION 4: TESTING & DEMONSTRATION
################################################################################


def demonstrate_speculative_decoding():
    """
    Demonstrate speculative decoding with MTP heads.

    Shows:
        1. KV cache comparison: MLA vs GQA vs MHA
        2. Speculative decoding simulation (with mock model)
        3. Acceptance rate analysis
        4. Speedup estimation

    Note: This uses mock models for demonstration. In production,
    the model and MTP heads would be real trained components.
    """
    print("=" * 70)
    print("SPECULATIVE DECODING DEMONSTRATION")
    print("=" * 70)

    # ---- Part 1: KV Cache Comparison ----
    print("\n1. KV Cache Memory Comparison")
    print("-" * 50)

    cache_comparison = compare_cache_memory(
        num_layers=32,
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        d_compress=512,
        max_seq_len=4096,
        batch_size=1,
    )

    print(f"  MHA cache: {cache_comparison['mha_bytes'] / 1e6:.1f} MB")
    print(f"  GQA cache: {cache_comparison['gqa_bytes'] / 1e6:.1f} MB")
    print(f"  MLA cache: {cache_comparison['mla_bytes'] / 1e6:.1f} MB")
    print(f"\n  GQA vs MHA: {cache_comparison['gqa_vs_mha']:.1f}x savings")
    print(f"  MLA vs MHA: {cache_comparison['mla_vs_mha']:.1f}x savings")
    print(f"  MLA vs GQA: {cache_comparison['mla_vs_gqa']:.1f}x savings")

    # ---- Part 2: Cache Operations ----
    print("\n2. MLA Cache Operations")
    print("-" * 50)

    mla_cache = MLAKVCache(
        num_layers=4,
        d_compress=128,
        max_seq_len=256,
        batch_size=1,
    )

    # Simulate appending tokens
    for i in range(10):
        for layer in range(4):
            c_kv = torch.randn(1, 1, 128)
            mla_cache.append(layer, c_kv)
        mla_cache.advance()

    print(f"  Cached {mla_cache.seq_len} tokens across {mla_cache.num_layers} layers")
    print(f"  Cache shape per layer: {mla_cache.cache[0].shape}")
    print(f"  Memory: {mla_cache.memory_bytes() / 1024:.1f} KB")

    # Trim cache
    mla_cache.trim(5)
    print(f"  After trim to 5: seq_len={mla_cache.seq_len}")

    # ---- Part 3: Mock Speculative Decoding ----
    print("\n3. Speculative Decoding (Mock Model)")
    print("-" * 50)

    # Create a mock model that returns random hidden states
    class MockModel(nn.Module):
        def __init__(self, d_model=64, vocab_size=1000):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.linear = nn.Linear(d_model, d_model)
            self.lm_head = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.linear(x)
            logits = self.lm_head(x)
            return logits

    # Create mock MTP heads
    class MockMTPHeads(nn.Module):
        def __init__(self, d_model=64, vocab_size=1000, num_heads=3):
            super().__init__()
            self.heads = nn.ModuleList([
                nn.Linear(d_model, vocab_size) for _ in range(num_heads)
            ])

        def forward(self, hidden_states, prediction_index=0):
            if prediction_index < len(self.heads):
                return self.heads[prediction_index](hidden_states)
            return self.heads[-1](hidden_states)

    # Initialize
    torch.manual_seed(42)
    model = MockModel(d_model=64, vocab_size=1000)
    mtp_heads = MockMTPHeads(d_model=64, vocab_size=1000, num_heads=3)

    # Create decoder
    config = SpeculativeConfig(
        num_speculative_tokens=3,
        temperature=1.0,
        log_acceptance_rate=True,
    )
    decoder = SpeculativeDecoder(model, mtp_heads, config)

    # Generate with speculative decoding
    prompt = torch.randint(0, 1000, (1, 5))
    print(f"  Prompt shape: {prompt.shape}")
    print(f"  Prompt tokens: {prompt[0].tolist()}")

    spec_seq, spec_metrics = decoder.generate(prompt, max_tokens=20)
    print(f"\n  Speculative Decoding Results:")
    print(f"    Tokens generated: {spec_metrics['tokens_generated']}")
    print(f"    Acceptance rate: {spec_metrics['acceptance_rate']:.2%}")
    print(f"    Tokens/second: {spec_metrics['tokens_per_second']:.1f}")
    print(f"    Elapsed: {spec_metrics['elapsed_seconds']:.4f}s")

    # Generate with vanilla decoding
    vanilla_seq, vanilla_metrics = decoder.generate_vanilla(prompt, max_tokens=20)
    print(f"\n  Vanilla Decoding Results:")
    print(f"    Tokens generated: {vanilla_metrics['tokens_generated']}")
    print(f"    Tokens/second: {vanilla_metrics['tokens_per_second']:.1f}")
    print(f"    Elapsed: {vanilla_metrics['elapsed_seconds']:.4f}s")

    # ---- Part 4: Speedup Analysis ----
    print("\n4. Speedup Analysis")
    print("-" * 50)

    if vanilla_metrics['elapsed_seconds'] > 0:
        speedup = vanilla_metrics['elapsed_seconds'] / max(spec_metrics['elapsed_seconds'], 1e-6)
        print(f"  Wall-clock speedup: {speedup:.2f}x")
    else:
        print("  (Elapsed time too small to measure reliably)")

    print(f"\n  Theoretical speedup formula:")
    print(f"    speedup = K / (c_draft + K * c_verify)")
    print(f"    where K = {config.num_speculative_tokens}")
    print(f"    c_draft = cost of MTP heads (cheap)")
    print(f"    c_verify = cost of one full model pass (expensive)")
    print(f"\n  With high acceptance rate (~80%), speedup approaches K/{config.num_speculative_tokens + 1}")
    print(f"  With low acceptance rate (~30%), speedup can be < 1x (slower than vanilla)")

    # ---- Part 5: Acceptance Rate Impact ----
    print("\n5. Acceptance Rate Impact on Speedup")
    print("-" * 50)

    K = 3
    c_ratio = 0.1  # draft cost / verify cost

    print(f"  K={K} speculative tokens, draft_cost={c_ratio:.1%} of verify_cost")
    print(f"  {'Accept Rate':>15} | {'Theoretical Speedup':>20}")
    print(f"  {'-'*15} | {'-'*20}")

    for accept_rate in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # Expected tokens per step: sum_{i=0}^{K} accept_rate^i
        # But simplified: approximately 1 + K * accept_rate for high rates
        expected_accepted = sum(accept_rate ** i for i in range(K + 1))
        cost_per_step = c_ratio + K  # draft + verify
        speedup = expected_accepted / cost_per_step
        print(f"  {accept_rate:>14.0%} | {speedup:>19.2f}x")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("  - Speculative decoding is LOSSLESS (same output distribution)")
    print("  - MTP heads serve as a FREE draft model (already trained)")
    print("  - Speedup depends on acceptance rate and K")
    print("  - MLA cache is ~32x smaller than MHA cache")
    print("  - Always benchmark on YOUR hardware and workload")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_speculative_decoding()
