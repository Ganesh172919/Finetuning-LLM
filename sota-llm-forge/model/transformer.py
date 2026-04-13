"""
################################################################################
TRANSFORMER LANGUAGE MODEL — FULL MODEL ASSEMBLY
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Transformer Language Model?
    A decoder-only transformer that processes a sequence of tokens and
    predicts the next token at each position. This is the architecture
    behind GPT, Llama, DeepSeek, and essentially all modern LLMs.

    The model consists of:
    1. Token and position embeddings.
    2. N transformer blocks (each: attention + FFN/MoE with residuals).
    3. Final RMS normalization.
    4. Language model head (vocabulary projection).
    5. Optional MTP heads for multi-token prediction.

Why does it matter?
    This is the complete, end-to-end model that everything else plugs into.
    The configuration system ensures reproducibility. The modular design
    allows swapping attention mechanisms (GQA, MLA, hybrid), FFN variants
    (dense, MoE), and optional features (MTP, hyper-connections) via config
    flags without changing the core architecture.

How does it work?
    1. Embed tokens: x = embed(token_ids) — (batch, seq_len, d_model).
    2. For each transformer block:
       a. RMSNorm → Attention → Residual.
       b. RMSNorm → FFN/MoE → Residual.
    3. Final RMSNorm.
    4. LM Head: logits = x @ embed_weight^T — (batch, seq_len, vocab_size).
    5. Optional: MTP heads generate auxiliary predictions.

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌──────────────────────────────────────────────────────────────┐
    │                     TransformerLM                             │
    │                                                              │
    │  token_ids: (batch, seq_len)                                 │
    │       │                                                      │
    │       ↓                                                      │
    │  ┌──────────┐                                                │
    │  │Embedding │ (vocab_size, d_model)                          │
    │  └────┬─────┘                                                │
    │       │ x: (batch, seq_len, d_model)                         │
    │       ↓                                                      │
    │  ┌──────────────────────────────────────────┐                │
    │  │     TransformerBlock (× n_layers)         │               │
    │  │                                           │               │
    │  │  x → RMSNorm → Attention → (+residual)   │               │
    │  │  x → RMSNorm → FFN/MoE   → (+residual)   │               │
    │  │                                           │               │
    │  │  Configurable:                            │               │
    │  │    attention_type: gqa | mla | hybrid      │               │
    │  │    use_moe: true | false                   │               │
    │  │    use_hyper_connections: true | false     │               │
    │  └──────────────────────────────────────────┘                │
    │       │                                                      │
    │       ↓                                                      │
    │  ┌──────────┐                                                │
    │  │ RMSNorm  │ (final normalization)                          │
    │  └────┬─────┘                                                │
    │       ↓                                                      │
    │  ┌──────────┐                                                │
    │  │ LM Head  │ (d_model, vocab_size) — tied to embedding      │
    │  └────┬─────┘                                                │
    │       ↓                                                      │
    │  logits: (batch, seq_len, vocab_size)                        │
    │       │                                                      │
    │       ↓ (optional)                                           │
    │  ┌──────────────┐                                            │
    │  │  MTP Heads   │ (multi-token prediction)                   │
    │  └──────────────┘                                            │
    └──────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2017: Transformer architecture — "Attention Is All You Need" (Vaswani et al.)
    - 2018: GPT — First decoder-only transformer LM (OpenAI)
    - 2018: BERT — Encoder-only transformer (Google)
    - 2020: GPT-3 — Scaling to 175B parameters (OpenAI)
    - 2023: Llama 2 — Open-weight GQA transformer (Meta)
    - 2024: DeepSeek-V2 — MLA + MoE architecture (DeepSeek-AI)
    - 2025: DeepSeek-V3/V4 — Full SOTA stack (DeepSeek-AI)

INTERVIEW QUESTIONS:
    1. "Why do modern LLMs use decoder-only architecture instead of
        encoder-decoder?"
       Decoder-only is simpler (one stack, one attention mask) and has
       proven to scale well. Encoder-decoder was dominant for translation
       and summarization, but autoregressive language modeling (which
       enables generation) naturally fits the decoder-only design. The
       causal mask also provides a natural curriculum: each position
       sees only its past, making it easy to parallelize training.

    2. "How do you choose between GQA, MLA, and hybrid attention?"
       GQA is the safe default — well-tested, good quality, reasonable
       memory savings (4-8x KV cache reduction). MLA provides much better
       compression (10-20x) but is more complex to implement and optimize.
       Hybrid sparse attention enables very long contexts (128K+) but
       adds implementation complexity. Recommendation: start with GQA,
       switch to MLA for inference-heavy workloads, add hybrid for
       long-context applications.

    3. "What is the purpose of the final RMSNorm before the LM head?"
       The residual stream accumulates values across layers, potentially
       growing in magnitude. The final RMSNorm ensures the hidden states
       are well-scaled before the vocabulary projection. Without it, the
       LM head would receive inputs of varying magnitude across layers,
       making the output logits unstable. It also ensures consistency
       with the norm applied before each sublayer (pre-norm pattern).

################################################################################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

from .attention import (
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    HybridSparseAttention,
    create_causal_mask,
)
from .moe import DeepSeekMoELayer, MoEConfig
from .mtp import MultiTokenPredictionHead, MTPLoss, MTPConfig
from .residual import RMSNorm, ResidualStream, HyperConnections


################################################################################
# SECTION 1: TRANSFORMER CONFIGURATION
################################################################################


@dataclass
class TransformerConfig:
    """
    Complete configuration for the Transformer Language Model.

    Every hyperparameter is a named field. No magic numbers allowed.

    Attributes:
        vocab_size: Size of the vocabulary.
        d_model: Model (hidden) dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention query heads.
        n_kv_heads: Number of attention KV heads (for GQA).
        d_head: Dimension per attention head.
        intermediate_dim: FFN intermediate dimension (for dense layers).
        max_seq_len: Maximum sequence length.
        rope_base: RoPE base frequency.
        dropout: Dropout rate.

        attention_type: Which attention mechanism to use.
                       "gqa" — Grouped Query Attention (default, Tier A)
                       "mla" — Multi-head Latent Attention (Tier B)
                       "hybrid" — Hybrid Sparse Attention (Tier C)

        # MLA-specific
        d_latent: Latent dimension for MLA KV compression.
        d_rope: Decoupled RoPE sub-component dimension for MLA.

        # Hybrid-specific
        n_sparse_heads: Number of sparse attention heads in hybrid mode.
        top_k_blocks: Number of blocks selected per token in sparse attention.
        block_size: Block size for sparse attention.

        # MoE
        moe_config: Optional MoE configuration. If None, use dense FFN.
        moe_layers: Which layers use MoE (None = all layers).
                    Example: [2, 5, 8] means only layers 2, 5, 8 use MoE.

        # MTP
        mtp_config: Optional MTP configuration. If None, no MTP.

        # Hyper-connections
        use_hyper_connections: Whether to use hyper-connections (advanced, optional).
        n_streams: Number of parallel streams for hyper-connections.

        # RoPE scaling
        rope_scaling_factor: Optional RoPE scaling for context extension.

        # Initialization
        d_base: Base model width for muP initialization.
    """

    # Core architecture
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    d_head: int = 64
    intermediate_dim: int = 2048
    max_seq_len: int = 4096
    rope_base: float = 10000.0
    dropout: float = 0.0

    # Attention type
    attention_type: str = "gqa"  # "gqa" | "mla" | "hybrid"

    # MLA-specific
    d_latent: int = 256
    d_rope: int = 32

    # Hybrid-specific
    n_sparse_heads: int = 6
    top_k_blocks: int = 4
    block_size: int = 64

    # MoE
    moe_config: Optional[MoEConfig] = None
    moe_layers: Optional[List[int]] = None

    # MTP
    mtp_config: Optional[MTPConfig] = None

    # Hyper-connections
    use_hyper_connections: bool = False
    n_streams: int = 2

    # RoPE scaling
    rope_scaling_factor: Optional[float] = None

    # muP
    d_base: int = 128

    def __post_init__(self):
        """Validate configuration consistency."""
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        assert self.attention_type in ("gqa", "mla", "hybrid"), (
            f"attention_type must be 'gqa', 'mla', or 'hybrid', got '{self.attention_type}'"
        )
        assert self.d_model == self.n_heads * self.d_head, (
            f"d_model ({self.d_model}) must equal n_heads * d_head "
            f"({self.n_heads} * {self.d_head} = {self.n_heads * self.d_head})"
        )


################################################################################
# SECTION 2: DENSE FFN (SWIGLU)
################################################################################


class DenseFFN(nn.Module):
    """
    Dense SwiGLU Feed-Forward Network
    ==================================

    Standard FFN used when MoE is not enabled for a layer.

    Formula:
        gate = x @ W_gate          # (batch, seq, intermediate_dim)
        up   = x @ W_up            # (batch, seq, intermediate_dim)
        h    = SiLU(gate) * up     # (batch, seq, intermediate_dim)
        out  = h @ W_down          # (batch, seq, d_model)

    Interview Question:
        "Why not just use a single linear layer instead of gate + up + down?"
        The gate-up-down structure (SwiGLU) is more parameter-efficient than
        a single large FFN for the same compute budget. The gating mechanism
        controls information flow, and the two-path design (gate and up)
        allows the network to learn more complex transformations. Empirically,
        SwiGLU with 2/3 * 4 * d_model intermediate dim matches the performance
        of ReLU FFN with 4 * d_model intermediate dim.
    """

    def __init__(self, d_model: int, intermediate_dim: int, dropout: float = 0.0):
        """
        Initialize dense FFN.

        Args:
            d_model: Model dimension.
            intermediate_dim: Intermediate (hidden) dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.w_gate = nn.Linear(d_model, intermediate_dim, bias=False)
        self.w_up = nn.Linear(d_model, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Output tensor, shape (batch, seq_len, d_model).
        """
        # gate: (batch, seq_len, intermediate_dim)
        gate = self.w_gate(x)
        # up: (batch, seq_len, intermediate_dim)
        up = self.w_up(x)
        # h: (batch, seq_len, intermediate_dim) — SwiGLU
        h = F.silu(gate) * up
        # out: (batch, seq_len, d_model)
        return self.w_down(self.dropout(h))


################################################################################
# SECTION 3: TRANSFORMER BLOCK
################################################################################


class TransformerBlock(nn.Module):
    """
    Transformer Block — Attention + FFN/MoE with Residuals
    =======================================================

    A single transformer layer consisting of:
    1. RMSNorm → Attention → Residual
    2. RMSNorm → FFN/MoE → Residual

    With optional hyper-connections replacing standard residuals.

    Step by step:
        1. Normalize input: h = RMSNorm(x).
        2. Apply attention: attn_out = Attention(h).
        3. Residual: x = x + attn_out.
        4. Normalize: h = RMSNorm(x).
        5. Apply FFN or MoE: ffn_out = FFN(h) or MoE(h).
        6. Residual: x = x + ffn_out.

    WHY this matters:
        Each transformer block transforms the representation incrementally.
        The pre-norm residual pattern keeps gradients flowing smoothly.
        The choice of attention type and FFN/MoE is configured per-block
        (via the config), allowing heterogeneous architectures where
        some layers use MoE and others use dense FFN.

    Interview Question:
        "How would you make some layers use MoE and others use dense FFN?"
        Use a configuration that specifies which layer indices use MoE.
        In the TransformerBlock constructor, check if the current layer
        index is in the MoE layers list. If yes, create a DeepSeekMoELayer;
        otherwise, create a DenseFFN. This is common in practice — e.g.,
        DeepSeek-V3 uses MoE for most layers but keeps some layers dense
        for stability.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int,
    ):
        """
        Initialize a transformer block.

        Args:
            config: Full transformer configuration.
            layer_idx: Index of this block (0-based).
        """
        super().__init__()
        self.layer_idx = layer_idx

        # --- Attention ---
        if config.attention_type == "gqa":
            self.attention = GroupedQueryAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_head=config.d_head,
                dropout=config.dropout,
                rope_base=config.rope_base,
                max_seq_len=config.max_seq_len,
                rope_scaling_factor=config.rope_scaling_factor,
            )
        elif config.attention_type == "mla":
            self.attention = MultiHeadLatentAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_head=config.d_head,
                d_latent=config.d_latent,
                d_rope=config.d_rope,
                dropout=config.dropout,
                rope_base=config.rope_base,
                max_seq_len=config.max_seq_len,
                rope_scaling_factor=config.rope_scaling_factor,
            )
        elif config.attention_type == "hybrid":
            self.attention = HybridSparseAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_head=config.d_head,
                n_sparse_heads=config.n_sparse_heads,
                top_k_blocks=config.top_k_blocks,
                block_size=config.block_size,
                dropout=config.dropout,
                rope_base=config.rope_base,
                max_seq_len=config.max_seq_len,
            )
        else:
            raise ValueError(f"Unknown attention_type: {config.attention_type}")

        # --- FFN or MoE ---
        use_moe = config.moe_config is not None
        if use_moe and (config.moe_layers is None or layer_idx in config.moe_layers):
            self.ffn = DeepSeekMoELayer(config.moe_config)
            self.is_moe = True
        else:
            self.ffn = DenseFFN(
                d_model=config.d_model,
                intermediate_dim=config.intermediate_dim,
                dropout=config.dropout,
            )
            self.is_moe = False

        # --- Residual connections ---
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Forward pass through a transformer block.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            mask: Optional causal attention mask.
            kv_cache: Optional KV cache from previous steps.

        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: Updated KV cache.
            aux_loss: MoE auxiliary loss (0.0 for dense FFN).

        Explanation:
            Pre-norm residual pattern:
            1. x_normed = RMSNorm(x)
            2. attn_out = Attention(x_normed, mask, kv_cache)
            3. x = x + dropout(attn_out)
            4. x_normed = RMSNorm(x)
            5. ffn_out = FFN(x_normed) or MoE(x_normed)
            6. x = x + dropout(ffn_out)
        """
        # --- Attention ---
        # normed: (batch, seq_len, d_model)
        normed = self.norm1(x)
        # attn_out: (batch, seq_len, d_model)
        # new_kv_cache: updated cache
        attn_out, new_kv_cache = self.attention(normed, mask=mask, kv_cache=kv_cache)
        # x: (batch, seq_len, d_model) — residual
        x = x + self.attn_dropout(attn_out)

        # --- FFN / MoE ---
        # normed: (batch, seq_len, d_model)
        normed = self.norm2(x)
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if self.is_moe:
            # ffn_out: (batch, seq_len, d_model)
            # aux_loss: scalar
            ffn_out, aux_loss = self.ffn(normed)
        else:
            # ffn_out: (batch, seq_len, d_model)
            ffn_out = self.ffn(normed)

        # x: (batch, seq_len, d_model) — residual
        x = x + self.ffn_dropout(ffn_out)

        return x, new_kv_cache, aux_loss


################################################################################
# SECTION 4: TRANSFORMER LANGUAGE MODEL
################################################################################


class TransformerLM(nn.Module):
    """
    Transformer Language Model — Full Assembly
    ===========================================

    Complete decoder-only transformer for language modeling.

    Architecture:
        Embedding → N × TransformerBlock → Final RMSNorm → LM Head
        Optional: MTP heads on top

    Methods:
        forward(): Full forward pass (training).
        generate(): Autoregressive generation (inference).
        get_num_params(): Count parameters.
        get_kv_cache(): Initialize empty KV cache.

    Step by step (training):
        1. Embed tokens: x = embed(token_ids).
        2. For each block: x, cache, aux_loss = block(x, mask, cache).
        3. Final norm: x = RMSNorm(x).
        4. LM head: logits = x @ embed_weight^T.
        5. Optional: MTP predictions.
        6. Return logits, total_aux_loss, mtp_loss.

    Step by step (inference):
        1. Embed new tokens: x = embed(token_ids).
        2. For each block: x, cache = block(x, mask, cache).
        3. Final norm + LM head → logits.
        4. Sample or argmax → next token.
        5. Repeat with updated cache.

    WHY this matters:
        This is the complete model that ties everything together. The
        configuration system allows exploring different architectures
        (GQA vs MLA, dense vs MoE, with/without MTP) by changing config
        fields, not code.

    Interview Question:
        "How do you handle KV caching for efficient autoregressive generation?"
        During the prefill phase (processing the prompt), we compute K and
        V for all positions and cache them. During generation (one token at
        a time), we only compute K and V for the new token and concatenate
        with the cache. This reduces generation from O(n^2) to O(n) per
        token. The cache is a list of (K, V) tuples, one per layer.
    """

    def __init__(self, config: TransformerConfig):
        """
        Initialize the transformer language model.

        Args:
            config: TransformerConfig with all hyperparameters.
        """
        super().__init__()
        self.config = config

        # --- Token Embedding ---
        # embedding: (vocab_size, d_model)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # --- Transformer Blocks ---
        # layers: ModuleList of n_layers TransformerBlock instances
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # --- Final Normalization ---
        self.final_norm = RMSNorm(config.d_model)

        # --- LM Head (tied to embedding) ---
        # The output projection shares weights with the embedding table.
        # This is standard practice and saves vocab_size * d_model parameters.
        # lm_head: (d_model, vocab_size) — weight is self.embedding.weight
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embedding and LM head weights
        self.lm_head.weight = self.embedding.weight

        # --- MTP (optional) ---
        self.mtp_head = None
        self.mtp_loss_fn = None
        if config.mtp_config is not None:
            self.mtp_head = MultiTokenPredictionHead(config.mtp_config)
            # Tie MTP output to main embedding
            self.mtp_head.tie_weights(self.embedding.weight)
            self.mtp_loss_fn = MTPLoss(mtp_weight=config.mtp_config.mtp_loss_weight)

        # --- Hyper-connections (optional) ---
        self.use_hyper_connections = config.use_hyper_connections

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize model weights.

        Explanation:
            We use a simple initialization here. For production, use
            muPInitializer from init.py which provides width-aware
            initialization and LR multipliers.
        """
        # Initialize embedding
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Count the number of parameters in the model.

        Args:
            non_embedding: If True, subtract embedding parameters (since
                          they are tied with the LM head).

        Returns:
            Number of parameters.

        Explanation:
            The embedding table is shared with the LM head, so counting
            it twice overstates the model size. Setting non_embedding=True
            subtracts one copy.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embedding.weight.numel()
        return n_params

    def get_kv_cache(self, batch_size: int, device: torch.device) -> list:
        """
        Initialize an empty KV cache for autoregressive generation.

        Args:
            batch_size: Batch size for the cache.
            device: Torch device.

        Returns:
            List of (K, V) cache tuples, one per layer.
            Each element is None initially (no cached values).
        """
        return [None] * len(self.layers)

    def forward(
        self,
        token_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], list]:
        """
        Forward pass through the full transformer.

        Args:
            token_ids: Input token IDs, shape (batch, seq_len).
            target_ids: Target token IDs for loss computation, shape (batch, seq_len).
                       Required for training.
            kv_cache: Optional KV cache from previous steps (for generation).

        Returns:
            logits: (batch, seq_len, vocab_size) — next-token logits.
            total_aux_loss: Scalar — sum of MoE auxiliary losses across layers.
            mtp_loss: Scalar or None — MTP auxiliary loss if MTP is enabled.
            new_kv_cache: Updated KV cache (list of (K, V) tuples).

        Explanation:
            1. Embed tokens.
            2. Create causal mask.
            3. Process through each transformer block.
            4. Apply final norm and LM head.
            5. Optionally compute MTP predictions and loss.
            6. Return logits, losses, and updated cache.
        """
        batch_size, seq_len = token_ids.shape

        # --- Step 1: Embed tokens ---
        # x: (batch, seq_len, d_model)
        x = self.embedding(token_ids)

        # --- Step 2: Create causal mask ---
        # mask: (seq_len, seq_len) — True = masked
        offset = 0
        if kv_cache is not None and kv_cache[0] is not None:
            # During incremental decoding, the full sequence length includes cached tokens
            offset = kv_cache[0][0].size(2) if isinstance(kv_cache[0], tuple) else kv_cache[0].size(2)

        total_len = offset + seq_len
        mask = create_causal_mask(total_len, x.device)
        # For incremental decoding, we only need the last row of the mask
        # mask: (1, 1, seq_len, total_len) — broadcastable
        mask = mask[offset:offset + seq_len, :total_len].unsqueeze(0).unsqueeze(0)

        # --- Step 3: Process through transformer blocks ---
        new_kv_cache = []
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for i, layer in enumerate(self.layers):
            # Get cached KV for this layer
            layer_cache = kv_cache[i] if kv_cache is not None else None

            # x: (batch, seq_len, d_model)
            # new_layer_cache: updated cache for this layer
            # aux_loss: MoE aux loss for this layer
            x, new_layer_cache, aux_loss = layer(x, mask=mask, kv_cache=layer_cache)

            new_kv_cache.append(new_layer_cache)
            total_aux_loss = total_aux_loss + aux_loss

        # --- Step 4: Final norm and LM head ---
        # x: (batch, seq_len, d_model)
        x = self.final_norm(x)
        # logits: (batch, seq_len, vocab_size)
        logits = self.lm_head(x)

        # --- Step 5: MTP (optional) ---
        mtp_loss = None
        if self.mtp_head is not None and target_ids is not None:
            # MTP forward: compute auxiliary predictions and loss
            mtp_loss_raw, _ = self.mtp_head(
                h_main=x,
                target_ids=target_ids,
                input_embed_fn=lambda ids: self.embedding(ids),
                mask=mask,
            )
            mtp_loss = mtp_loss_raw

        return logits, total_aux_loss, mtp_loss, new_kv_cache

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            prompt_ids: Prompt token IDs, shape (batch, prompt_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. Lower = more deterministic.
            top_k: If set, sample from top-k tokens only.
            top_p: If set, use nucleus (top-p) sampling.

        Returns:
            Generated token IDs, shape (batch, prompt_len + max_new_tokens).

        Explanation:
            1. Process the full prompt in one forward pass (prefill).
            2. For each new token:
               a. Take the last token's logits.
               b. Apply temperature scaling.
               c. Optionally apply top-k or top-p filtering.
               d. Sample from the filtered distribution.
               e. Append the new token to the sequence.
            3. The KV cache ensures we don't recompute attention for previous tokens.
        """
        self.eval()
        device = prompt_ids.device
        batch_size = prompt_ids.shape[0]

        # Initialize with the prompt
        generated = prompt_ids.clone()  # (batch, prompt_len)
        kv_cache = self.get_kv_cache(batch_size, device)

        # Prefill: process the entire prompt
        logits, _, _, kv_cache = self.forward(generated, kv_cache=kv_cache)

        for _ in range(max_new_tokens):
            # Get logits for the last position
            # next_logits: (batch, vocab_size)
            next_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                # Keep only top-k logits, set rest to -inf
                top_k_values, _ = torch.topk(next_logits, top_k, dim=-1)
                # threshold: (batch, 1)
                threshold = top_k_values[:, -1:]
                next_logits = torch.where(
                    next_logits < threshold,
                    torch.tensor(float("-inf"), device=device),
                    next_logits,
                )

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                # cumulative_probs: (batch, vocab_size)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Mask tokens with cumulative prob above threshold
                # sorted_mask: (batch, vocab_size)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")

                # Scatter back to original indices
                next_logits.scatter_(1, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(next_logits, dim=-1)  # (batch, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Forward pass with just the new token (incremental decoding)
            logits, _, _, kv_cache = self.forward(next_token, kv_cache=kv_cache)

        return generated

    def get_mtp_loss(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute MTP loss from hidden states (for external use).

        Args:
            hidden_states: Final hidden states, shape (batch, seq_len, d_model).
            target_ids: Target token IDs, shape (batch, seq_len).

        Returns:
            MTP loss (scalar) or None if MTP is not enabled.

        Explanation:
            This method is useful when you want to compute MTP loss
            separately from the main forward pass, e.g., for logging
            or when using a custom training loop.
        """
        if self.mtp_head is None:
            return None

        mask = create_causal_mask(hidden_states.size(1), hidden_states.device)
        mask = mask.unsqueeze(0).unsqueeze(0)

        mtp_loss, _ = self.mtp_head(
            h_main=hidden_states,
            target_ids=target_ids,
            input_embed_fn=lambda ids: self.embedding(ids),
            mask=mask,
        )

        return mtp_loss


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################


def demonstrate_transformer():
    """Demonstrate the full transformer language model."""
    print("=" * 70)
    print("TRANSFORMER LANGUAGE MODEL DEMONSTRATION")
    print("=" * 70)

    device = torch.device("cpu")

    # --- Configuration: Small model for testing ---
    config = TransformerConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        intermediate_dim=128,
        max_seq_len=128,
        rope_base=10000.0,
        dropout=0.0,
        attention_type="gqa",
    )

    print(f"\nConfiguration:")
    print(f"  vocab_size:           {config.vocab_size}")
    print(f"  d_model:              {config.d_model}")
    print(f"  n_layers:             {config.n_layers}")
    print(f"  n_heads:              {config.n_heads}")
    print(f"  n_kv_heads:           {config.n_kv_heads}")
    print(f"  d_head:               {config.d_head}")
    print(f"  intermediate_dim:     {config.intermediate_dim}")
    print(f"  attention_type:       {config.attention_type}")

    # --- Create Model ---
    model = TransformerLM(config)

    # Count parameters
    total_params = model.get_num_params(non_embedding=False)
    non_embed_params = model.get_num_params(non_embedding=True)
    print(f"\nParameter counts:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Non-embedding params: {non_embed_params:,}")

    # --- Forward Pass (Training) ---
    print(f"\n--- Training Forward Pass ---")
    batch_size = 2
    seq_len = 16

    # token_ids: (batch, seq_len)
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    # target_ids: (batch, seq_len) — shifted right
    target_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    model.train()
    logits, aux_loss, mtp_loss, kv_cache = model(token_ids, target_ids=target_ids)
    print(f"  Input shape:          {token_ids.shape}")
    print(f"  Logits shape:         {logits.shape}")
    print(f"  Aux loss (MoE):       {aux_loss.item():.6f}")
    print(f"  MTP loss:             {mtp_loss}")

    # Compute loss
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, config.vocab_size),
        target_ids[:, 1:].reshape(-1),
    )
    print(f"  Cross-entropy loss:   {loss.item():.4f}")

    # --- Generation (Inference) ---
    print(f"\n--- Inference: Autoregressive Generation ---")
    model.eval()
    # prompt: (1, 5) — short prompt
    prompt = torch.randint(0, config.vocab_size, (1, 5), device=device)
    print(f"  Prompt shape:         {prompt.shape}")

    generated = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_k=50,
    )
    print(f"  Generated shape:      {generated.shape}")
    print(f"  Prompt tokens:        {prompt[0].tolist()}")
    print(f"  Generated tokens:     {generated[0, 5:].tolist()}")

    # --- Model with MoE ---
    print(f"\n--- Model with MoE ---")
    moe_config = MoEConfig(
        d_model=64,
        expert_intermediate_dim=64,
        n_shared_experts=1,
        n_routed_experts=4,
        top_k=2,
    )
    config_moe = TransformerConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        intermediate_dim=128,
        max_seq_len=128,
        attention_type="gqa",
        moe_config=moe_config,
        moe_layers=[1, 3],  # Only layers 1 and 3 use MoE
    )
    model_moe = TransformerLM(config_moe)
    total_moe_params = model_moe.get_num_params(non_embedding=False)
    print(f"  MoE model params:     {total_moe_params:,}")
    print(f"  MoE layers:           {config_moe.moe_layers}")

    # Forward pass with MoE
    logits_moe, aux_loss_moe, _, _ = model_moe(token_ids, target_ids=target_ids)
    print(f"  Logits shape:         {logits_moe.shape}")
    print(f"  MoE aux loss:         {aux_loss_moe.item():.6f}")

    # --- Model with MLA ---
    print(f"\n--- Model with MLA ---")
    config_mla = TransformerConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,  # MLA uses n_kv_heads = n_heads (no GQA)
        d_head=16,
        d_latent=32,
        d_rope=8,
        intermediate_dim=128,
        max_seq_len=128,
        attention_type="mla",
    )
    model_mla = TransformerLM(config_mla)
    total_mla_params = model_mla.get_num_params(non_embedding=False)
    print(f"  MLA model params:     {total_mla_params:,}")

    logits_mla, _, _, kv_cache_mla = model_mla(token_ids)
    print(f"  Logits shape:         {logits_mla.shape}")
    if kv_cache_mla[0] is not None:
        print(f"  KV cache c_kv shape:  {kv_cache_mla[0][0].shape}")
        print(f"  KV cache k_rope shape:{kv_cache_mla[0][1].shape}")

    # --- Layer breakdown ---
    print(f"\n--- Layer Breakdown ---")
    for i, layer in enumerate(model.layers):
        n_attn = sum(p.numel() for p in layer.attention.parameters())
        n_ffn = sum(p.numel() for p in layer.ffn.parameters())
        ffn_type = "MoE" if layer.is_moe else "Dense"
        print(f"  Layer {i}: attention={n_attn:,}, ffn={n_ffn:,} ({ffn_type})")

    print("\n" + "=" * 70)
    print("All transformer demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_transformer()
