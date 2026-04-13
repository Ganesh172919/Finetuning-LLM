"""
################################################################################
ATTENTION MECHANISMS — THREE TIERS OF ATTENTION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Attention?
    Attention is the mechanism that allows each token in a sequence to
    compute a weighted sum of all other tokens' representations. The
    weights are determined by compatibility (dot product) between queries
    and keys, scaled and softmax-normalized.

    In standard Multi-Head Attention (MHA), every head has its own Q, K, V
    projections. This is expressive but memory-hungry for the KV cache at
    inference time.

Why does it matter?
    Attention is the computational bottleneck of transformers — O(seq_len^2)
    in both compute and memory. The three tiers represent increasingly
    aggressive memory/compute optimizations:

    Tier A — GQA:  Groups of query heads share KV heads. Reduces KV cache
             by n_kv_heads / n_heads ratio. Llama 2/3, Mistral use this.

    Tier B — MLA:  Compresses KV into a low-dimensional latent vector.
             Cache stores latent c_kv instead of full K/V. DeepSeek-V2/V3.

    Tier C — Hybrid Sparse: Most layers use block-sparse/top-k attention
             (cheap). A few layers use full attention (expensive). Enables
             very long context lengths. DeepSeek-V3/V4 pattern.

How does it work?
    1. Project input x into Q, K, V using learned linear maps.
    2. Apply Rotary Position Embedding (RoPE) to encode position.
    3. Compute attention scores: softmax(Q @ K^T / sqrt(d_head)).
    4. Multiply by V to get output.
    5. Merge heads and project to output dimension.

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌──────────────────────────────────────────────────────────────┐
    │                   ATTENTION TIERS                             │
    │                                                              │
    │  Tier A: GQA                                                │
    │    Q₁ Q₂ Q₃ Q₄ ──┐                                         │
    │    Q₅ Q₆ Q₇ Q₈ ──┼── K₁ V₁  (shared KV head)              │
    │    ...             │                                         │
    │    Q₂₉ Q₃₀ Q₃₁ Q₃₂── K₈ V₈                                │
    │                                                              │
    │  Tier B: MLA                                                │
    │    x → W_compress → c_kv (latent, small)                    │
    │    c_kv → W_k_up → K,  c_kv → W_v_up → V                   │
    │    x → W_q → Q_full,  split → Q_rope (small) + Q_content    │
    │    Cache: only c_kv (not K, V)                               │
    │                                                              │
    │  Tier C: Hybrid Sparse                                       │
    │    Layer i (sparse): top-k tokens attend to each other       │
    │    Layer j (full):   all tokens attend to all tokens         │
    └──────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2017: Scaled Dot-Product Attention — "Attention Is All You Need" (Vaswani et al.)
    - 2019: Multi-Query Attention (MQA) — "Fast Transformer Decoding" (Shazeer)
    - 2021: Rotary Position Embedding — RoFormer (Su et al.)
    - 2023: Grouped Query Attention — Llama 2 (Touvron et al.)
    - 2024: Multi-head Latent Attention — DeepSeek-V2 (DeepSeek-AI)
    - 2025: Hybrid Sparse Attention (CSA/HCA) — DeepSeek-V3/V4 (DeepSeek-AI)

INTERVIEW QUESTIONS:
    1. "Why does GQA save memory without much quality loss?"
       Because adjacent query heads tend to learn similar attention patterns.
       Sharing KV heads exploits this redundancy. The ratio (e.g., 4:1) is
       a tunable knob between quality and efficiency.

    2. "How does MLA achieve better compression than GQA?"
       GQA reduces the number of KV heads but each head is still full-sized.
       MLA projects KV into a shared low-dimensional latent space (d_latent
       << n_kv_heads * d_head), then reconstructs full K/V on the fly.
       The latent captures the essential information in fewer dimensions.

    3. "What is the key challenge with applying RoPE in MLA?"
       RoPE mixes position information into Q and K in a way that depends
       on the full head dimension. If you compress K into a latent and
       decompress it, the rotary structure may not be preserved. MLA solves
       this by using a small 'decoupled' RoPE sub-component that is NOT
       compressed — it is computed and cached separately.

################################################################################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


################################################################################
# SECTION 1: UTILITY FUNCTIONS
################################################################################


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (autoregressive) attention mask.

    Args:
        seq_len: Length of the sequence.
        device: Torch device for the mask tensor.

    Returns:
        Boolean tensor of shape (seq_len, seq_len) where True means "masked"
        (i.e., position i cannot attend to position j if j > i).

    Explanation:
        This is an upper-triangular boolean matrix. Position i can only
        attend to positions <= i. This prevents information leakage from
        future tokens during training.

    Example:
        >>> mask = create_causal_mask(4, torch.device("cpu"))
        >>> mask.long()
        tensor([[0, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0]])
    """
    # mask: (seq_len, seq_len) — upper triangular, True = masked
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return mask


################################################################################
# SECTION 2: ROTARY POSITION EMBEDDING (RoPE)
################################################################################


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    =================================

    Encodes absolute position by rotating query and key vectors in 2D
    subspaces. The rotation angle depends on the position and a
    frequency that varies across dimensions.

    Formula:
        For dimension pair (2i, 2i+1):
            theta_i = base^(-2i / d_head)
            angle   = position * theta_i
            R(angle) = [[cos(angle), -sin(angle)],
                        [sin(angle),  cos(angle)]]

        q_rotated = R @ q  (applied pairwise to adjacent dimensions)

    Step by step:
        1. Precompute frequency table: theta_i = base^(-2i/d_head)
        2. For each position, compute angles = pos * theta
        3. Compute cos and sin of angles
        4. Apply rotation: split q into pairs, rotate each pair

    WHY this matters:
        RoPE encodes relative position implicitly. The dot product
        Q_pos_m @ K_pos_n depends only on (m - n), not on absolute positions.
        This allows the model to generalize to longer sequences than seen
        during training (with appropriate scaling).

    Interview Question:
        "How does RoPE encode relative position with absolute rotations?"
        The rotation for position m is R(m*theta). The dot product between
        Q rotated by m and K rotated by n involves R(m*theta)^T @ R(n*theta)
        = R((n-m)*theta), which depends only on the relative distance (n-m).
    """

    def __init__(
        self,
        d_head: int,
        base: float = 10000.0,
        max_seq_len: int = 4096,
        scaling_factor: Optional[float] = None,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        """
        Initialize RoPE.

        Args:
            d_head: Dimension of each attention head. Must be even.
            base: Base frequency for computing theta. Default 10000.
            max_seq_len: Maximum sequence length for precomputation.
            scaling_factor: YaRN/NTK-aware scaling factor for context extension.
                           If None, no scaling is applied (standard RoPE).
            beta_fast: YaRN low-frequency correction factor.
            beta_slow: YaRN high-frequency correction factor.
        """
        super().__init__()
        assert d_head % 2 == 0, f"d_head must be even, got {d_head}"

        self.d_head = d_head          # int: head dimension
        self.base = base              # float: frequency base
        self.max_seq_len = max_seq_len
        self.scaling_factor = scaling_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # Precompute inverse frequencies
        # inv_freq: (d_head // 2,) — theta_i values
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Apply NTK-aware scaling if requested
        if scaling_factor is not None:
            inv_freq_scaled = self._compute_ntk_scaled_inv_freq(inv_freq, scaling_factor)
            self.register_buffer("inv_freq_scaled", inv_freq_scaled, persistent=False)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _compute_ntk_scaled_inv_freq(
        self, inv_freq: torch.Tensor, scaling_factor: float
    ) -> torch.Tensor:
        """
        Apply NTK-aware scaling to inverse frequencies.

        Args:
            inv_freq: Original inverse frequencies, shape (d_head // 2,).
            scaling_factor: The scaling factor (> 1 for context extension).

        Returns:
            Scaled inverse frequencies, shape (d_head // 2,).

        Explanation:
            NTK-aware scaling adjusts the frequency spectrum so that the
            model can handle longer contexts without retraining. Low
            frequencies (which encode long-range dependencies) are scaled
            more aggressively than high frequencies.

            YaRN refines this with beta_fast/beta_slow to control which
            frequency bands are rescaled vs. left alone.
        """
        # wavelength: (d_head // 2,) — period of each frequency component
        wavelength = 2 * math.pi / inv_freq  # (d_head // 2,)

        # Compute YaRN interpolation mask
        # freq_mask: (d_head // 2,) — 1.0 for frequencies to scale, 0.0 for others
        low_freq_wavelength = self.max_seq_len / self.beta_slow   # scalar
        high_freq_wavelength = self.max_seq_len / self.beta_fast  # scalar

        # Smooth ramp between beta_fast and beta_slow
        # smooth: (d_head // 2,) — ramps from 0 to 1
        smooth = (wavelength - high_freq_wavelength) / (low_freq_wavelength - high_freq_wavelength)
        smooth = smooth.clamp(0.0, 1.0)

        # Scaled inverse frequencies: blend of scaled and original
        inv_freq_scaled = inv_freq / scaling_factor  # (d_head // 2,)
        # Blend: high freq (short wavelength) kept as-is, low freq scaled
        inv_freq_blended = smooth * inv_freq_scaled + (1.0 - smooth) * inv_freq  # (d_head // 2,)

        return inv_freq_blended

    def _build_cache(self, seq_len: int) -> None:
        """
        Precompute cos and sin values for all positions up to seq_len.

        Args:
            seq_len: Maximum sequence length.
        """
        # positions: (seq_len,) — [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.float32)

        # Choose which inv_freq to use
        inv_freq = self.inv_freq_scaled if self.scaling_factor is not None else self.inv_freq

        # angles: (seq_len, d_head // 2) — outer product of positions and inv_freq
        angles = torch.outer(positions, inv_freq)  # (seq_len, d_head // 2)

        # Duplicate for sin/cos application to pairs
        # angles: (seq_len, d_head) — [theta_0, theta_0, theta_1, theta_1, ...]
        angles = torch.cat([angles, angles], dim=-1)  # (seq_len, d_head)

        # cos_cache: (1, 1, seq_len, d_head) — broadcastable for attention
        self.register_buffer("cos_cache", angles.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        # sin_cache: (1, 1, seq_len, d_head)
        self.register_buffer("sin_cache", angles.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to queries and keys.

        Args:
            q: Query tensor, shape (batch, n_heads, seq_len, d_head).
            k: Key tensor, shape (batch, n_kv_heads, seq_len, d_head).
            offset: Position offset for KV cache (incremental decoding).

        Returns:
            q_rotated: (batch, n_heads, seq_len, d_head)
            k_rotated: (batch, n_kv_heads, seq_len, d_head)

        Explanation:
            The rotation is applied by splitting each vector into pairs
            of adjacent dimensions and applying a 2D rotation matrix:

            For pair (x, y) at dimensions (2i, 2i+1):
                x' = x * cos(theta) - y * sin(theta)
                y' = x * sin(theta) + y * cos(theta)

            This is equivalent to complex multiplication:
                (x + iy) * e^(i*theta) = (x + iy)(cos(theta) + i*sin(theta))
        """
        seq_len = q.size(2)

        # Slice the precomputed cache for current positions
        # cos_slice: (1, 1, seq_len, d_head)
        cos_slice = self.cos_cache[:, :, offset:offset + seq_len, :]
        # sin_slice: (1, 1, seq_len, d_head)
        sin_slice = self.sin_cache[:, :, offset:offset + seq_len, :]

        # Apply rotation using the half-space trick:
        # Split into first half and second half, then combine with sin/cos
        # q_first: (batch, n_heads, seq_len, d_head // 2) — first half of each pair
        q_first = q[..., : self.d_head // 2]
        # q_second: (batch, n_heads, seq_len, d_head // 2) — second half of each pair
        q_second = q[..., self.d_head // 2 :]

        # Alternate sign pattern: [-second, first]
        # q_rotated: (batch, n_heads, seq_len, d_head)
        q_rotated = torch.cat([-q_second, q_first], dim=-1)

        # Same for keys
        # k_first: (batch, n_kv_heads, seq_len, d_head // 2)
        k_first = k[..., : self.d_head // 2]
        # k_second: (batch, n_kv_heads, seq_len, d_head // 2)
        k_second = k[..., self.d_head // 2 :]
        # k_rotated: (batch, n_kv_heads, seq_len, d_head)
        k_rotated = torch.cat([-k_second, k_first], dim=-1)

        # Apply: q' = q * cos + rotate(q) * sin
        # q_out: (batch, n_heads, seq_len, d_head)
        q_out = q * cos_slice + q_rotated * sin_slice
        # k_out: (batch, n_kv_heads, seq_len, d_head)
        k_out = k * cos_slice + k_rotated * sin_slice

        return q_out, k_out


################################################################################
# SECTION 3: GROUPED QUERY ATTENTION (GQA) — TIER A
################################################################################


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    =============================

    Multiple query heads share a smaller number of KV heads.
    Example: 32 query heads / 8 KV heads = 4:1 sharing ratio.

    Formula:
        Q = x @ W_Q           # (batch, seq_len, n_heads, d_head)
        K = x @ W_K           # (batch, seq_len, n_kv_heads, d_head)
        V = x @ W_V           # (batch, seq_len, n_kv_heads, d_head)
        # Repeat K,V to match n_heads:
        #   every n_heads // n_kv_heads query heads share one KV head
        K_expanded = repeat(K)  # (batch, seq_len, n_heads, d_head)
        V_expanded = repeat(V)  # (batch, seq_len, n_heads, d_head)
        attn = softmax(Q @ K_expanded^T / sqrt(d_head)) @ V_expanded

    Step by step:
        1. Project input to Q (full), K (reduced), V (reduced).
        2. Apply RoPE to Q and K.
        3. Expand K,V by repeating each KV head for its group of query heads.
        4. Compute scaled dot-product attention with causal mask.
        5. Merge heads and project to output.

    WHY this matters:
        During autoregressive generation, the KV cache is the main memory
        bottleneck. GQA reduces cache size by n_kv_heads/n_heads (e.g., 4x
        reduction with 8 KV heads and 32 query heads). Quality degradation
        is minimal because adjacent query heads tend to learn similar patterns.

    Interview Question:
        "You have 32 query heads and 8 KV heads. How do you expand K and V?"
        Each KV head is shared by 32/8 = 4 query heads. We use
        torch.repeat_interleave to duplicate each KV head 4 times along
        the head dimension, so K goes from (batch, 8, seq, d) to
        (batch, 32, seq, d). Then standard attention computation proceeds.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        max_seq_len: int = 4096,
        rope_scaling_factor: Optional[float] = None,
    ):
        """
        Initialize GQA.

        Args:
            d_model: Model dimension (input/output size).
            n_heads: Number of query attention heads.
            n_kv_heads: Number of key/value heads (must divide n_heads).
            d_head: Dimension per head.
            dropout: Attention dropout rate.
            rope_base: Base frequency for RoPE.
            max_seq_len: Maximum sequence length.
            rope_scaling_factor: Optional RoPE scaling for context extension.
        """
        super().__init__()

        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        self.n_heads = n_heads              # int: number of query heads
        self.n_kv_heads = n_kv_heads        # int: number of KV heads
        self.d_head = d_head                # int: dimension per head
        self.n_rep = n_heads // n_kv_heads  # int: repeat factor for KV expansion
        self.head_dim = d_head              # int: alias for clarity
        self.dropout_rate = dropout

        # Q projection: full number of heads
        # w_q: (d_model, n_heads * d_head)
        self.w_q = nn.Linear(d_model, n_heads * d_head, bias=False)

        # K projection: reduced number of heads
        # w_k: (d_model, n_kv_heads * d_head)
        self.w_k = nn.Linear(d_model, n_kv_heads * d_head, bias=False)

        # V projection: reduced number of heads
        # w_v: (d_model, n_kv_heads * d_head)
        self.w_v = nn.Linear(d_model, n_kv_heads * d_head, bias=False)

        # Output projection
        # w_o: (n_heads * d_head, d_model)
        self.w_o = nn.Linear(n_heads * d_head, d_model, bias=False)

        # RoPE
        self.rope = RotaryPositionEmbedding(
            d_head=d_head,
            base=rope_base,
            max_seq_len=max_seq_len,
            scaling_factor=rope_scaling_factor,
        )

        self.attn_dropout = nn.Dropout(dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match the number of query heads.

        Args:
            x: Tensor with KV heads, shape (batch, n_kv_heads, seq_len, d_head).

        Returns:
            Tensor with heads expanded, shape (batch, n_heads, seq_len, d_head).

        Explanation:
            If n_heads=32 and n_kv_heads=8, each KV head is repeated 4 times.
            This is torch.repeat_interleave with repeats=self.n_rep along dim=1.
        """
        if self.n_rep == 1:
            return x  # No expansion needed when n_heads == n_kv_heads

        # Expand: (batch, n_kv_heads, seq, d) -> (batch, n_heads, seq, d)
        batch, n_kv, seq_len, d = x.shape
        # x_expanded: (batch, n_kv_heads, 1, seq_len, d_head) — insert repeat dim
        x = x[:, :, None, :, :].expand(batch, n_kv, self.n_rep, seq_len, d)
        # x: (batch, n_heads, seq_len, d_head) — flatten repeat into head dim
        return x.reshape(batch, self.n_kv_heads * self.n_rep, seq_len, d)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Grouped Query Attention.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            mask: Optional boolean attention mask, True = masked.
                  Shape broadcastable to (batch, n_heads, seq_len, kv_len).
            kv_cache: Optional cached (K, V) tensors from previous steps.
                      Each shape (batch, n_kv_heads, cached_len, d_head).

        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: Updated (K, V) cache, each (batch, n_kv_heads, total_len, d_head)

        Explanation:
            1. Project x to Q, K, V.
            2. Reshape to (batch, heads, seq_len, d_head).
            3. Apply RoPE.
            4. Concatenate with KV cache if present (incremental decoding).
            5. Expand KV heads to match query heads.
            6. Compute attention with causal mask.
            7. Project output.
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.w_q(x)  # (batch, seq_len, n_heads * d_head)
        k = self.w_k(x)  # (batch, seq_len, n_kv_heads * d_head)
        v = self.w_v(x)  # (batch, seq_len, n_kv_heads * d_head)

        # Reshape to multi-head format
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)  # (batch, seq, n_heads, d_head)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_head)  # (batch, seq, n_kv_heads, d_head)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_head)  # (batch, seq, n_kv_heads, d_head)

        # Transpose to (batch, heads, seq_len, d_head) for attention
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, d_head)
        k = k.transpose(1, 2)  # (batch, n_kv_heads, seq_len, d_head)
        v = v.transpose(1, 2)  # (batch, n_kv_heads, seq_len, d_head)

        # Apply RoPE
        offset = 0
        if kv_cache is not None:
            offset = kv_cache[0].size(2)  # cached sequence length

        q, k = self.rope(q, k, offset=offset)  # q: (batch, n_heads, seq, d_head)
                                                # k: (batch, n_kv_heads, seq, d_head)

        # KV cache: concatenate previous and current K, V
        if kv_cache is not None:
            # k_cached: (batch, n_kv_heads, cached_len, d_head)
            # k: (batch, n_kv_heads, seq_len, d_head)
            k = torch.cat([kv_cache[0], k], dim=2)  # (batch, n_kv_heads, total_len, d_head)
            v = torch.cat([kv_cache[1], v], dim=2)  # (batch, n_kv_heads, total_len, d_head)

        new_kv_cache = (k, v)

        # Expand KV heads to match query heads
        k_expanded = self._repeat_kv(k)  # (batch, n_heads, total_len, d_head)
        v_expanded = self._repeat_kv(v)  # (batch, n_heads, total_len, d_head)

        # Compute attention scores
        # scores: (batch, n_heads, seq_len, total_len)
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply causal mask
        if mask is not None:
            # mask: broadcast to (batch, n_heads, seq_len, total_len)
            scores = scores.masked_fill(mask, float("-inf"))

        # Softmax and dropout
        # attn_weights: (batch, n_heads, seq_len, total_len)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        # attn_output: (batch, n_heads, seq_len, d_head)
        attn_output = torch.matmul(attn_weights, v_expanded)

        # Merge heads: (batch, n_heads, seq_len, d_head) -> (batch, seq_len, n_heads * d_head)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.d_head)

        # Output projection
        output = self.w_o(attn_output)  # (batch, seq_len, d_model)

        return output, new_kv_cache


################################################################################
# SECTION 4: MULTI-HEAD LATENT ATTENTION (MLA) — TIER B
################################################################################


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA)
    =================================

    Compresses KV cache into a low-dimensional latent representation.
    Instead of caching full K and V tensors, we cache only the compressed
    latent c_kv (and a small decoupled RoPE component).

    Key Innovation:
        RoPE and content-based attention have conflicting requirements for
        compression. MLA resolves this by splitting Q and K into:
        - Content component: compressed through latent space (cache-friendly)
        - RoPE component: small, uncompressed, cached separately

    Cache structure:
        Standard GQA: cache K (batch, n_kv_heads, seq, d_head) and V (same)
        MLA:          cache c_kv (batch, seq, d_latent) + k_rope (batch, seq, d_rope)

        Savings: d_latent + d_rope  vs  2 * n_kv_heads * d_head

    Step by step:
        1. Compress x to c_kv latent: c_kv = W_compress @ x
        2. Reconstruct K from latent: k = W_k_up @ c_kv
        3. Reconstruct V from latent: v = W_v_up @ c_kv
        4. Compute RoPE components separately:
           q_rope = W_q_rope @ x,  k_rope = W_k_rope @ x
           Apply RoPE to (q_rope, k_rope)
        5. Attention = softmax([q_content, q_rope] @ [k_content, k_rope]^T / sqrt) @ v

    WHY this matters:
        The KV cache is the primary memory bottleneck during inference.
        MLA can achieve 5-10x cache compression vs standard MHA while
        maintaining quality. This enables much longer context lengths
        or larger batch sizes on the same hardware.

    Interview Question:
        "Why can't you just compress the KV cache with PCA or autoencoders?"
        The problem is that RoPE creates a specific structure in Q and K
        that must be preserved for positional reasoning. A naive compression
        would destroy this structure. MLA's decoupled approach avoids this
        by keeping the RoPE sub-component uncompressed while compressing
        the content component separately.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        d_latent: int,
        d_rope: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        max_seq_len: int = 4096,
        rope_scaling_factor: Optional[float] = None,
    ):
        """
        Initialize MLA.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_head: Dimension per head (content portion).
            d_latent: Dimension of the compressed KV latent.
            d_rope: Dimension of the decoupled RoPE sub-component.
            dropout: Attention dropout rate.
            rope_base: RoPE base frequency.
            max_seq_len: Maximum sequence length.
            rope_scaling_factor: Optional RoPE scaling for context extension.
        """
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_head           # content head dimension
        self.d_rope = d_rope           # RoPE sub-component dimension per head
        self.d_latent = d_latent
        self.d_total = d_head + d_rope  # total per-head dimension (content + rope)
        self.dropout_rate = dropout

        # --- KV Compression (latent) ---
        # W_compress: projects input to latent KV space
        # (d_model, d_latent)
        self.W_compress = nn.Linear(d_model, d_latent, bias=False)

        # W_k_up: reconstructs K content from latent
        # (d_latent, n_heads * d_head)
        self.W_k_up = nn.Linear(d_latent, n_heads * d_head, bias=False)

        # W_v_up: reconstructs V from latent
        # (d_latent, n_heads * d_head)
        self.W_v_up = nn.Linear(d_latent, n_heads * d_head, bias=False)

        # --- Decoupled RoPE components ---
        # W_q_rope: projects input to Q's RoPE sub-component
        # (d_model, n_heads * d_rope)
        self.W_q_rope = nn.Linear(d_model, n_heads * d_rope, bias=False)

        # W_k_rope: projects input to K's RoPE sub-component
        # (d_model, d_rope) — shared across heads
        self.W_k_rope = nn.Linear(d_model, d_rope, bias=False)

        # --- Q content projection ---
        # W_q_content: projects input to Q's content sub-component
        # (d_model, n_heads * d_head)
        self.W_q_content = nn.Linear(d_model, n_heads * d_head, bias=False)

        # --- Output projection ---
        # Input to w_o is n_heads * (d_head + d_rope) because we concatenate
        # content and rope attention outputs
        # w_o: (n_heads * d_total, d_model)
        self.w_o = nn.Linear(n_heads * self.d_total, d_model, bias=False)

        # RoPE (only for the decoupled sub-component)
        self.rope = RotaryPositionEmbedding(
            d_head=d_rope,
            base=rope_base,
            max_seq_len=max_seq_len,
            scaling_factor=rope_scaling_factor,
        )

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Multi-Head Latent Attention.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            mask: Optional boolean attention mask, True = masked.
            kv_cache: Optional cached (c_kv, k_rope) tensors.
                      c_kv: (batch, cached_len, d_latent)
                      k_rope: (batch, n_heads, cached_len, d_rope)

        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: (c_kv, k_rope) updated with current step

        Explanation:
            1. Compress input to latent c_kv for K and V reconstruction.
            2. Reconstruct K_content and V from latent.
            3. Compute Q_content and Q_rope, K_rope from input.
            4. Apply RoPE only to the rope sub-components.
            5. Concatenate content + rope for Q and K.
            6. Standard attention computation.
            7. Cache only c_kv and k_rope (compact!).
        """
        batch_size, seq_len, _ = x.shape

        # --- Step 1: Compress to latent ---
        # c_kv: (batch, seq_len, d_latent)
        c_kv = self.W_compress(x)

        # --- Step 2: Reconstruct K, V from latent ---
        # k_content: (batch, seq_len, n_heads * d_head)
        k_content = self.W_k_up(c_kv)
        # v: (batch, seq_len, n_heads * d_head)
        v = self.W_v_up(c_kv)

        # --- Step 3: Compute Q content, Q rope, K rope ---
        # q_content: (batch, seq_len, n_heads * d_head)
        q_content = self.W_q_content(x)
        # q_rope: (batch, seq_len, n_heads * d_rope)
        q_rope = self.W_q_rope(x)
        # k_rope: (batch, seq_len, d_rope)
        k_rope = self.W_k_rope(x)

        # --- Step 4: Reshape to multi-head format ---
        # q_content: (batch, n_heads, seq_len, d_head)
        q_content = q_content.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # k_content: (batch, n_heads, seq_len, d_head)
        k_content = k_content.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # v: (batch, n_heads, seq_len, d_head)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # q_rope: (batch, n_heads, seq_len, d_rope)
        q_rope = q_rope.view(batch_size, seq_len, self.n_heads, self.d_rope).transpose(1, 2)
        # k_rope: expand to all heads — (batch, 1, seq_len, d_rope) -> (batch, n_heads, seq_len, d_rope)
        k_rope = k_rope.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        # --- Step 5: Apply RoPE only to decoupled components ---
        offset = 0
        if kv_cache is not None:
            offset = kv_cache[0].size(1)  # c_kv cached length

        q_rope, k_rope = self.rope(q_rope, k_rope, offset=offset)

        # --- Step 6: KV cache (compact: only latent + rope) ---
        if kv_cache is not None:
            # c_kv_cached: (batch, cached_len, d_latent)
            # c_kv: (batch, seq_len, d_latent)
            c_kv = torch.cat([kv_cache[0], c_kv], dim=1)  # (batch, total_len, d_latent)
            # k_rope_cached: (batch, n_heads, cached_len, d_rope)
            # k_rope: (batch, n_heads, seq_len, d_rope)
            k_rope = torch.cat([kv_cache[1], k_rope], dim=2)  # (batch, n_heads, total_len, d_rope)

        new_kv_cache = (c_kv, k_rope)

        # Reconstruct full K and V from updated cache for attention
        # k_content_full: (batch, total_len, n_heads * d_head)
        k_content_full = self.W_k_up(c_kv)
        # v_full: (batch, total_len, n_heads * d_head)
        v_full = self.W_v_up(c_kv)
        # k_content_full: (batch, n_heads, total_len, d_head)
        k_content_full = k_content_full.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        # v_full: (batch, n_heads, total_len, d_head)
        v_full = v_full.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # --- Step 7: Concatenate content + rope for Q and K ---
        # q: (batch, n_heads, seq_len, d_head + d_rope)
        q = torch.cat([q_content, q_rope], dim=-1)
        # k: (batch, n_heads, total_len, d_head + d_rope)
        k = torch.cat([k_content_full, k_rope], dim=-1)

        # --- Step 8: Attention ---
        # scores: (batch, n_heads, seq_len, total_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_total)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # attn_weights: (batch, n_heads, seq_len, total_len)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # attn_output: (batch, n_heads, seq_len, d_head)
        attn_output = torch.matmul(attn_weights, v_full)

        # --- Step 9: Merge heads and project ---
        # attn_output: (batch, seq_len, n_heads * d_head)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.d_head)

        # Pad to include rope dimension in output (zero-pad for rope part)
        # This keeps the output projection dimension consistent
        # Actually, we need to restructure: the output should be n_heads * d_total
        # We use only v (which has d_head dims) so we pad zeros for rope dims
        rope_padding = torch.zeros(
            batch_size, seq_len, self.n_heads * self.d_rope, device=x.device, dtype=x.dtype
        )  # (batch, seq_len, n_heads * d_rope)
        attn_output = torch.cat([attn_output, rope_padding], dim=-1)  # (batch, seq_len, n_heads * d_total)

        # output: (batch, seq_len, d_model)
        output = self.w_o(attn_output)

        return output, new_kv_cache


################################################################################
# SECTION 5: HYBRID SPARSE ATTENTION — TIER C
################################################################################


class HybridSparseAttention(nn.Module):
    """
    Hybrid Long-Context Attention
    ==============================

    Approximation of DeepSeek-V3/V4's CSA/HCA pattern.
    Most layers use block-sparse/top-k token selection (cheap, approximate).
    A minority of layers use full attention (expensive, precise).

    NOTE: The exact CSA (Cross-head Sparse Attention) and HCA (Hybrid
    Compressed Attention) internals from DeepSeek-V4 are not fully public.
    This implementation is the best-documented open approximation based on
    published descriptions and analyses. Where details are speculative,
    they are marked with comments.

    Architecture:
        - Divide attention heads into groups.
        - Some groups use top-k sparse attention: each token attends only
          to the k most relevant tokens (selected by a learned or heuristic
          scoring function).
        - Some groups use full causal attention.
        - This layer is config-gated and OFF by default.

    WHY this matters:
        Full attention is O(n^2) in sequence length. For very long contexts
        (128K+ tokens), this becomes prohibitive. Sparse attention reduces
        this to O(n * k) where k << n. The hybrid approach ensures that at
        least some heads maintain full attention for tasks that require
        global context, while most heads handle local/sparse patterns cheaply.

    Interview Question:
        "How do you decide which tokens to attend to in sparse attention?"
        Common approaches: (1) Block-sparse: divide sequence into blocks,
        use a learned scoring function to select top-k blocks. (2) Local
        window: each token attends to its local neighborhood plus a few
        globally-selected tokens. (3) Hash-based: LSH buckets for approximate
        nearest-neighbor attention. This implementation uses approach (1)
        with a learned block scoring function.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        n_sparse_heads: int,
        top_k_blocks: int,
        block_size: int = 64,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        max_seq_len: int = 4096,
    ):
        """
        Initialize Hybrid Sparse Attention.

        Args:
            d_model: Model dimension.
            n_heads: Total number of attention heads.
            n_kv_heads: Number of KV heads (for full-attention portion).
            d_head: Dimension per head.
            n_sparse_heads: Number of heads using sparse attention.
                           The remaining (n_heads - n_sparse_heads) use full attention.
            top_k_blocks: Number of blocks each token selects for sparse attention.
            block_size: Size of each attention block for sparse selection.
            dropout: Dropout rate.
            rope_base: RoPE base frequency.
            max_seq_len: Maximum sequence length.
        """
        super().__init__()

        assert n_sparse_heads <= n_heads, (
            f"n_sparse_heads ({n_sparse_heads}) must be <= n_heads ({n_heads})"
        )

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.n_sparse_heads = n_sparse_heads
        self.n_full_heads = n_heads - n_sparse_heads
        self.top_k_blocks = top_k_blocks
        self.block_size = block_size
        self.dropout_rate = dropout

        # Full attention component (GQA-based)
        if self.n_full_heads > 0:
            self.full_attention = GroupedQueryAttention(
                d_model=d_model,
                n_heads=self.n_full_heads,
                n_kv_heads=min(n_kv_heads, self.n_full_heads),
                d_head=d_head,
                dropout=dropout,
                rope_base=rope_base,
                max_seq_len=max_seq_len,
            )

        # Sparse attention component
        if self.n_sparse_heads > 0:
            # Q, K, V projections for sparse heads
            # w_q_sparse: (d_model, n_sparse_heads * d_head)
            self.w_q_sparse = nn.Linear(d_model, n_sparse_heads * d_head, bias=False)
            # w_k_sparse: (d_model, n_kv_heads * d_head) — KV sharing like GQA
            sparse_kv_heads = min(n_kv_heads, n_sparse_heads)
            self.sparse_kv_heads = sparse_kv_heads
            self.sparse_n_rep = n_sparse_heads // sparse_kv_heads if sparse_kv_heads > 0 else 1
            # w_k_sparse: (d_model, sparse_kv_heads * d_head)
            self.w_k_sparse = nn.Linear(d_model, sparse_kv_heads * d_head, bias=False)
            # w_v_sparse: (d_model, sparse_kv_heads * d_head)
            self.w_v_sparse = nn.Linear(d_model, sparse_kv_heads * d_head, bias=False)

            # Block scoring function: learned linear projection to score blocks
            # block_score: (d_model, 1) — projects each block to a scalar score
            self.block_score = nn.Linear(d_model, 1, bias=False)

            # RoPE for sparse heads
            self.sparse_rope = RotaryPositionEmbedding(
                d_head=d_head,
                base=rope_base,
                max_seq_len=max_seq_len,
            )

            # w_o_sparse: (n_sparse_heads * d_head, d_model)
            self.w_o_sparse = nn.Linear(n_sparse_heads * d_head, d_model, bias=False)

            self.sparse_dropout = nn.Dropout(dropout)

        # Combine full + sparse outputs
        # gating: learned scalar to mix full and sparse outputs
        self.gate = nn.Parameter(torch.tensor(0.5))

    def _compute_block_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute relevance scores for each block of tokens.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Block scores, shape (batch, n_blocks) where n_blocks = ceil(seq_len / block_size).

        Explanation:
            We divide the sequence into blocks, pool each block to a single
            representation (mean pooling), then project to a scalar score.
            This is a simplified version of what DeepSeek-V3 likely uses.
        """
        batch_size, seq_len, d_model = x.shape
        n_blocks = math.ceil(seq_len / self.block_size)

        # Pad sequence to be divisible by block_size
        pad_len = n_blocks * self.block_size - seq_len
        if pad_len > 0:
            # x_padded: (batch, n_blocks * block_size, d_model)
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x

        # Reshape to blocks
        # x_blocks: (batch, n_blocks, block_size, d_model)
        x_blocks = x_padded.view(batch_size, n_blocks, self.block_size, d_model)

        # Mean pool each block
        # block_repr: (batch, n_blocks, d_model)
        block_repr = x_blocks.mean(dim=2)

        # Score each block
        # scores: (batch, n_blocks, 1) -> (batch, n_blocks)
        scores = self.block_score(block_repr).squeeze(-1)

        return scores

    def _sparse_attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute sparse attention for the sparse-head subset.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            mask: Optional causal mask.

        Returns:
            Sparse attention output, shape (batch, seq_len, n_sparse_heads * d_head).
        """
        batch_size, seq_len, _ = x.shape

        # Compute block scores for token selection
        # block_scores: (batch, n_blocks)
        block_scores = self._compute_block_scores(x)

        n_blocks = math.ceil(seq_len / self.block_size)

        # Select top-k blocks
        # top_k_indices: (batch, top_k_blocks)
        k = min(self.top_k_blocks, n_blocks)
        _, top_k_indices = torch.topk(block_scores, k, dim=-1)
        # top_k_indices: (batch, top_k_blocks) — block indices to attend to

        # Build sparse mask: True = masked (don't attend)
        # sparse_mask: (batch, seq_len, seq_len) — start with all masked
        sparse_mask = torch.ones(batch_size, seq_len, seq_len, device=x.device, dtype=torch.bool)

        # For each batch element, unmask the selected blocks
        for b in range(batch_size):
            for block_idx in range(n_blocks):
                if block_idx in top_k_indices[b]:
                    start = block_idx * self.block_size
                    end = min(start + self.block_size, seq_len)
                    # Unmask: tokens can attend to this block
                    sparse_mask[b, :, start:end] = False

        # Apply causal constraint on top of sparse mask
        if mask is not None:
            # mask: (seq_len, seq_len) — broadcast to (batch, seq_len, seq_len)
            sparse_mask = sparse_mask | mask.unsqueeze(0)

        # Project to Q, K, V for sparse heads
        q = self.w_q_sparse(x)  # (batch, seq_len, n_sparse_heads * d_head)
        k = self.w_k_sparse(x)  # (batch, seq_len, sparse_kv_heads * d_head)
        v = self.w_v_sparse(x)  # (batch, seq_len, sparse_kv_heads * d_head)

        # Reshape
        q = q.view(batch_size, seq_len, self.n_sparse_heads, self.d_head).transpose(1, 2)
        # q: (batch, n_sparse_heads, seq_len, d_head)
        k = k.view(batch_size, seq_len, self.sparse_kv_heads, self.d_head).transpose(1, 2)
        # k: (batch, sparse_kv_heads, seq_len, d_head)
        v = v.view(batch_size, seq_len, self.sparse_kv_heads, self.d_head).transpose(1, 2)
        # v: (batch, sparse_kv_heads, seq_len, d_head)

        # Apply RoPE
        q, k = self.sparse_rope(q, k)  # q: (batch, n_sparse_heads, seq, d_head)
                                        # k: (batch, sparse_kv_heads, seq, d_head)

        # Expand KV heads
        if self.sparse_n_rep > 1:
            k = k[:, :, None, :, :].expand(
                batch_size, self.sparse_kv_heads, self.sparse_n_rep, seq_len, self.d_head
            ).reshape(batch_size, self.n_sparse_heads, seq_len, self.d_head)
            v = v[:, :, None, :, :].expand(
                batch_size, self.sparse_kv_heads, self.sparse_n_rep, seq_len, self.d_head
            ).reshape(batch_size, self.n_sparse_heads, seq_len, self.d_head)

        # Compute attention with sparse mask
        # scores: (batch, n_sparse_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # sparse_mask: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
        scores = scores.masked_fill(sparse_mask.unsqueeze(1), float("-inf"))

        # attn_weights: (batch, n_sparse_heads, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.sparse_dropout(attn_weights)

        # attn_output: (batch, n_sparse_heads, seq_len, d_head)
        attn_output = torch.matmul(attn_weights, v)

        # Merge heads: (batch, seq_len, n_sparse_heads * d_head)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_sparse_heads * self.d_head)

        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Hybrid Sparse Attention.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            mask: Optional causal mask.
            kv_cache: Optional KV cache (for the full-attention portion).

        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: Updated KV cache from full-attention portion.
        """
        output_parts = []
        new_kv_cache = kv_cache

        # Full attention for a subset of heads
        if self.n_full_heads > 0:
            full_out, new_kv_cache = self.full_attention(x, mask=mask, kv_cache=kv_cache)
            # full_out: (batch, seq_len, d_model)
            output_parts.append(full_out)

        # Sparse attention for the remaining heads
        if self.n_sparse_heads > 0:
            sparse_out = self._sparse_attention(x, mask=mask)
            # Project sparse output to d_model
            sparse_out = self.w_o_sparse(sparse_out)  # (batch, seq_len, d_model)
            output_parts.append(sparse_out)

        # Combine: weighted sum of full and sparse outputs
        if len(output_parts) == 2:
            gate = torch.sigmoid(self.gate)
            output = gate * output_parts[0] + (1 - gate) * output_parts[1]
        elif len(output_parts) == 1:
            output = output_parts[0]
        else:
            raise ValueError("At least one attention type must be enabled")

        return output, new_kv_cache


################################################################################
# SECTION 6: TESTING & DEMONSTRATION
################################################################################


def demonstrate_attention():
    """Demonstrate all three tiers of attention."""
    print("=" * 70)
    print("ATTENTION MECHANISMS DEMONSTRATION")
    print("=" * 70)

    device = torch.device("cpu")
    batch_size = 2
    seq_len = 16
    d_model = 64
    n_heads = 8
    n_kv_heads = 2
    d_head = 8
    max_seq_len = 128

    # Create dummy input
    # x: (batch, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Create causal mask
    mask = create_causal_mask(seq_len, device)  # (seq_len, seq_len)

    # --- Tier A: GQA ---
    print("\n--- Tier A: Grouped Query Attention (GQA) ---")
    gqa = GroupedQueryAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        max_seq_len=max_seq_len,
    )
    gqa_output, gqa_cache = gqa(x, mask=mask)
    print(f"  Input shape:           {x.shape}")
    print(f"  Output shape:          {gqa_output.shape}")
    print(f"  KV cache K shape:      {gqa_cache[0].shape}")
    print(f"  KV cache V shape:      {gqa_cache[1].shape}")
    print(f"  KV cache memory (K):   {gqa_cache[0].numel()} elements")
    print(f"  Full MHA cache would:  {batch_size * n_heads * seq_len * d_head} elements")
    print(f"  Memory savings:        {n_heads / n_kv_heads:.1f}x reduction")

    # Test incremental decoding
    x_new = torch.randn(batch_size, 1, d_model, device=device)  # (batch, 1, d_model)
    mask_new = create_causal_mask(1, device)
    gqa_out_inc, gqa_cache_inc = gqa(x_new, mask=mask_new, kv_cache=gqa_cache)
    print(f"\n  Incremental decoding:")
    print(f"  Input shape:           {x_new.shape}")
    print(f"  Output shape:          {gqa_out_inc.shape}")
    print(f"  Updated cache K shape: {gqa_cache_inc[0].shape}")

    # --- Tier B: MLA ---
    print("\n--- Tier B: Multi-head Latent Attention (MLA) ---")
    d_latent = 32
    d_rope = 4
    mla = MultiHeadLatentAttention(
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_head,
        d_latent=d_latent,
        d_rope=d_rope,
        max_seq_len=max_seq_len,
    )
    mla_output, mla_cache = mla(x, mask=mask)
    print(f"  Input shape:           {x.shape}")
    print(f"  Output shape:          {mla_output.shape}")
    print(f"  Cache c_kv shape:      {mla_cache[0].shape}")
    print(f"  Cache k_rope shape:    {mla_cache[1].shape}")
    mla_cache_elements = mla_cache[0].numel() + mla_cache[1].numel()
    gqa_cache_elements = gqa_cache[0].numel() + gqa_cache[1].numel()
    print(f"  MLA cache elements:    {mla_cache_elements}")
    print(f"  GQA cache elements:    {gqa_cache_elements}")
    print(f"  Compression ratio:     {gqa_cache_elements / mla_cache_elements:.2f}x")

    # --- Tier C: Hybrid Sparse ---
    print("\n--- Tier C: Hybrid Sparse Attention ---")
    hybrid = HybridSparseAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        n_sparse_heads=4,
        top_k_blocks=2,
        block_size=8,
        max_seq_len=max_seq_len,
    )
    hybrid_output, hybrid_cache = hybrid(x, mask=mask)
    print(f"  Input shape:           {x.shape}")
    print(f"  Output shape:          {hybrid_output.shape}")
    if hybrid_cache is not None:
        print(f"  Cache K shape:         {hybrid_cache[0].shape}")

    # --- RoPE demonstration ---
    print("\n--- RoPE Demonstration ---")
    rope = RotaryPositionEmbedding(d_head=d_head, base=10000.0, max_seq_len=max_seq_len)
    q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device)
    k = torch.randn(batch_size, n_kv_heads, seq_len, d_head, device=device)
    q_rot, k_rot = rope(q, k)
    print(f"  Q input shape:         {q.shape}")
    print(f"  Q rotated shape:       {q_rot.shape}")
    print(f"  K input shape:         {k.shape}")
    print(f"  K rotated shape:       {k_rot.shape}")

    # Verify that RoPE preserves norm (rotation preserves vector length)
    q_norm_before = q.norm(dim=-1).mean()
    q_norm_after = q_rot.norm(dim=-1).mean()
    print(f"  Q norm before RoPE:    {q_norm_before:.6f}")
    print(f"  Q norm after RoPE:     {q_norm_after:.6f}")
    print(f"  Norm preserved:        {torch.allclose(q_norm_before, q_norm_after, atol=1e-5)}")

    # --- Causal mask demonstration ---
    print("\n--- Causal Mask Demonstration ---")
    small_mask = create_causal_mask(4, device)
    print(f"  Mask shape:            {small_mask.shape}")
    print(f"  Mask (True=masked):")
    print(f"  {small_mask.long()}")

    print("\n" + "=" * 70)
    print("All attention demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_attention()
