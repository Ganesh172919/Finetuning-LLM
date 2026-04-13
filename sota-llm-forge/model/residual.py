"""
################################################################################
RESIDUAL STREAMS — STANDARD AND HYPER-CONNECTIONS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Residual Stream?
    The residual stream is the highway through which information flows in a
    transformer. Each layer adds its output to the running stream:
        x_out = x_in + sublayer(norm(x_in))
    This additive connection allows gradients to flow directly from the
    output back to early layers, enabling training of very deep networks.

Why does it matter?
    Without residual connections, transformers deeper than ~10 layers become
    very difficult to train due to vanishing gradients. The residual stream
    creates a "gradient highway" that keeps gradient magnitudes stable across
    hundreds of layers.

    Hyper-connections extend this idea: instead of one stream, maintain k
    parallel streams with learned mixing. This allows different layers to
    "specialize" which stream they read from and write to, enabling more
    flexible information routing.

How does it work?
    Standard Residual: x = x + sublayer(norm(x))
    Hyper-Connections: k parallel streams, each layer applies a learned
    mixing matrix to combine streams before processing, then distributes
    the output back across streams.

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌──────────────────────────────────────────────────────────────┐
    │              STANDARD RESIDUAL STREAM                         │
    │                                                              │
    │  x ──────────────────────────────────────→ (+) → out         │
    │  │                                       ↑                   │
    │  ↓                                       │                   │
    │  RMSNorm → Attention/MLP ────────────────┘                   │
    │                                                              │
    │              HYPER-CONNECTIONS                                │
    │                                                              │
    │  stream_1 ──→ [Mix Matrix] ──→ Layer ──→ [Distribute] ──→ s1'│
    │  stream_2 ──→   ↓           ↓    ↓         ↓           ──→ s2'│
    │  stream_k ──→   └─── combined input ──── output split   ──→ sk'│
    └──────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2015: Residual Networks (ResNet) — "Deep Residual Learning" (He et al.)
    - 2017: Residual connections in Transformers — "Attention Is All You Need"
    - 2023: DeepNet — Pre-norm residual scaling for 1000-layer models (Microsoft)
    - 2025: Hyper-Connections — Multi-stream residual (DeepSeek-V3/V4)

INTERVIEW QUESTIONS:
    1. "Why do modern transformers use pre-norm (norm before sublayer) instead
        of post-norm (norm after sublayer)?"
       Pre-norm keeps the residual stream's magnitude stable. In post-norm,
       the residual addition happens before the norm, so each layer's
       contribution can grow unbounded. Pre-norm ensures the residual
       stream stays well-conditioned, making training more stable for
       deep models (100+ layers).

    2. "What problem do hyper-connections solve that standard residuals don't?"
       In a standard residual, every layer reads and writes to the same
       single stream. This creates a "tragedy of the commons" where layers
       must share the same representational bandwidth. Hyper-connections
       give each layer access to k parallel streams with learned routing,
       so different layers can specialize different streams for different
       types of information (e.g., syntax vs semantics).

    3. "How does the mixing matrix work in hyper-connections?"
       Each layer has a k x k mixing matrix M. The input to layer l is:
       input_i = sum_j M_ij * stream_j for each stream i.
       The output is distributed back: stream_i += sum_j D_ij * output_j.
       M and D are learned and can be different for each layer.

################################################################################
"""

import torch
import torch.nn as nn
from typing import Optional


################################################################################
# SECTION 1: RMS LAYER NORMALIZATION
################################################################################


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    =====================================

    A simpler, faster alternative to LayerNorm that doesn't subtract the mean.
    Used in Llama, DeepSeek, and most modern LLMs.

    Formula:
        RMS(x) = sqrt(mean(x^2) + eps)
        output = (x / RMS(x)) * gamma

    Step by step:
        1. Compute x^2 for each element.
        2. Take the mean across the normalized dimension.
        3. Add epsilon for numerical stability.
        4. Take the square root.
        5. Divide x by RMS(x).
        6. Multiply by learnable scale parameter gamma.

    WHY this matters:
        RMSNorm is ~10-15% faster than LayerNorm because it skips the mean
        subtraction and beta bias. Empirically, it performs just as well
        for LLMs. This matters because normalization is applied 2 * n_layers
        times per forward pass.

    Interview Question:
        "Why is RMSNorm preferred over LayerNorm in modern LLMs?"
        RMSNorm removes the mean-centering step and bias term, reducing
        computation. Since the transformer's residual connections already
        center the activations implicitly, the mean subtraction in LayerNorm
        is redundant. RMSNorm preserves the scaling normalization that
        matters for stable training.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.

        Args:
            d_model: Dimension to normalize over.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        # gamma: (d_model,) — learnable scale parameter
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor, shape (..., d_model).

        Returns:
            Normalized tensor, same shape as input.

        Explanation:
            We use the float32 trick for numerical stability: compute RMS
            in float32 even if the input is in lower precision (e.g., bfloat16).
        """
        # Compute RMS in float32 for stability
        # x_float: (..., d_model) — cast to float32
        x_float = x.float()

        # variance: (..., 1) — mean of squared values
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)

        # rms: (..., 1) — root mean square
        rms = torch.sqrt(variance + self.eps)

        # Normalize: (..., d_model)
        x_normed = x_float / rms

        # Scale and cast back to original dtype
        # output: (..., d_model)
        output = (x_normed * self.weight).to(x.dtype)

        return output


################################################################################
# SECTION 2: STANDARD RESIDUAL STREAM
################################################################################


class ResidualStream(nn.Module):
    """
    Standard Residual Connection
    =============================

    Applies: x_out = x + sublayer(norm(x))

    This is the "pre-norm" residual pattern used in GPT, Llama, DeepSeek, etc.

    Formula:
        h = RMSNorm(x)
        y = sublayer(h)
        x_out = x + y

    Step by step:
        1. Apply RMSNorm to the input.
        2. Pass through the sublayer (attention or MLP/MoE).
        3. Add the result back to the original input.

    WHY this matters:
        The skip connection creates a direct gradient path. If the sublayer
        produces near-zero output early in training, the residual still
        passes the input through unchanged. This prevents the "gradient
        highway" from being blocked.

    Interview Question:
        "What happens if you remove residual connections from a 100-layer
        transformer?"
        The model would essentially not train. Gradients would vanish
        exponentially through the stack of nonlinear transformations.
        Residual connections reduce the effective depth for gradient flow
        from O(n_layers) to O(1) — each layer only needs to learn a
        small perturbation to the running representation.
    """

    def __init__(self, sublayer: nn.Module, d_model: int, dropout: float = 0.0):
        """
        Initialize the residual stream wrapper.

        Args:
            sublayer: The module to wrap (e.g., attention or MLP).
            d_model: Model dimension (for RMSNorm).
            dropout: Dropout rate applied to sublayer output.
        """
        super().__init__()
        self.sublayer = sublayer
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            **kwargs: Additional arguments passed to the sublayer.

        Returns:
            Output tensor, shape (batch, seq_len, d_model).

        Explanation:
            Pre-norm pattern: normalize BEFORE the sublayer, add AFTER.
            This is preferred over post-norm for deep models because it
            keeps the residual stream's magnitude stable.
        """
        # normed: (batch, seq_len, d_model)
        normed = self.norm(x)

        # sublayer_output: (batch, seq_len, d_model)
        sublayer_output = self.sublayer(normed, **kwargs)

        # Apply dropout and residual
        # output: (batch, seq_len, d_model)
        output = x + self.dropout(sublayer_output)

        return output


################################################################################
# SECTION 3: HYPER-CONNECTIONS (ADVANCED, OPTIONAL)
################################################################################


class HyperConnections(nn.Module):
    """
    Multi-stream Hyper-Connections
    ===============================

    Maintains k parallel residual streams with learned mixing matrices.
    Each layer reads a learned combination of all streams, processes it,
    then distributes the output back across streams.

    NOTE: The exact manifold-constrained projection from DeepSeek-V4 is not
    fully public. This implements the openly-documented multi-stream
    hyper-connection as the closest reproducible approximation. The key
    difference from the paper is that we use standard linear mixing rather
    than manifold-constrained projections.

    Architecture:
        Input: k streams [s_1, s_2, ..., s_k]
        For each layer:
            1. Mix: input_i = sum_j M_ij * s_j  (learned mixing)
            2. Process: output_i = sublayer(norm(input_i))
            3. Distribute: s_j += sum_i D_ij * output_i  (learned distribution)

    Formula:
        M = softmax(W_mix)     # (k, k) mixing matrix, row-normalized
        D = softmax(W_dist)    # (k, k) distribution matrix, row-normalized
        mixed = M @ streams    # (k, batch, seq, d_model)
        outputs = [sublayer(norm(mixed_i)) for i in range(k)]
        streams += D @ outputs # distribute back

    Step by step:
        1. Stack k streams into a matrix.
        2. Apply learned mixing: each stream gets a weighted combination.
        3. Normalize and process each mixed stream through the sublayer.
        4. Apply learned distribution: each output is split across streams.
        5. Add to streams (residual).

    WHY this matters:
        Standard residuals force all layers to share one representational
        stream. Hyper-connections allow specialization: some streams can
        carry syntactic information, others semantic, etc. This has shown
        improvements in deep models (100+ layers) where the single-stream
        bottleneck becomes limiting.

    Interview Question:
        "How are hyper-connections different from mixture-of-experts?"
        MoE routes tokens to different expert networks. Hyper-connections
        route information across parallel residual streams within the SAME
        network. MoE operates at the FFN level; hyper-connections operate
        at the residual stream level. They are complementary and can be
        combined (as in DeepSeek-V3/V4).
    """

    def __init__(
        self,
        sublayer: nn.Module,
        d_model: int,
        n_streams: int = 2,
        dropout: float = 0.0,
    ):
        """
        Initialize Hyper-Connections.

        Args:
            sublayer: The module to wrap (e.g., attention or MLP).
            d_model: Model dimension.
            n_streams: Number of parallel residual streams (k).
            dropout: Dropout rate.
        """
        super().__init__()
        self.sublayer = sublayer
        self.n_streams = n_streams
        self.d_model = d_model

        # RMSNorm for each stream
        # norms: ModuleList of k RMSNorm instances
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_streams)])

        # Mixing matrix: how each stream reads from others
        # W_mix: (n_streams, n_streams) — raw logits
        self.W_mix = nn.Parameter(torch.randn(n_streams, n_streams) * 0.02)

        # Distribution matrix: how output is distributed back
        # W_dist: (n_streams, n_streams) — raw logits
        self.W_dist = nn.Parameter(torch.randn(n_streams, n_streams) * 0.02)

        # Stream-specific scaling (learned)
        # stream_scales: (n_streams,) — per-stream residual scaling
        self.stream_scales = nn.Parameter(torch.ones(n_streams))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        streams: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with hyper-connections.

        Args:
            streams: Stacked stream tensor, shape (n_streams, batch, seq_len, d_model).
                     Each stream is a separate residual representation.
            **kwargs: Additional arguments passed to the sublayer.

        Returns:
            Updated streams, shape (n_streams, batch, seq_len, d_model).

        Explanation:
            1. Compute mixing weights: softmax over each row of W_mix.
            2. Mix streams: weighted sum across stream dimension.
            3. Normalize each mixed stream.
            4. Process through sublayer.
            5. Compute distribution weights: softmax over each row of W_dist.
            6. Distribute output back to streams with residual addition.
        """
        n_streams, batch_size, seq_len, d_model = streams.shape

        # --- Mixing ---
        # mix_weights: (n_streams, n_streams) — row-normalized
        mix_weights = torch.softmax(self.W_mix, dim=-1)

        # mixed: (n_streams, batch, seq_len, d_model)
        # For each stream i: mixed_i = sum_j mix_weights[i,j] * streams_j
        # Einstein notation: 'ij,jbld->ibld'
        # streams: (n_streams, batch, seq_len, d_model)
        # mix_weights: (n_streams, n_streams)
        # mixed: (n_streams, batch, seq_len, d_model)
        mixed = torch.einsum("ij,jbld->ibld", mix_weights, streams)

        # --- Normalize and Process ---
        # outputs: (n_streams, batch, seq_len, d_model)
        outputs = []
        for i in range(n_streams):
            # normed_i: (batch, seq_len, d_model)
            normed_i = self.norms[i](mixed[i])
            # sublayer_out_i: (batch, seq_len, d_model)
            sublayer_out_i = self.sublayer(normed_i, **kwargs)
            outputs.append(sublayer_out_i)

        # outputs: (n_streams, batch, seq_len, d_model)
        outputs = torch.stack(outputs, dim=0)

        # --- Distribution ---
        # dist_weights: (n_streams, n_streams) — row-normalized
        dist_weights = torch.softmax(self.W_dist, dim=-1)

        # distributed: (n_streams, batch, seq_len, d_model)
        # For each stream j: distributed_j = sum_i dist_weights[i,j] * output_i
        distributed = torch.einsum("ij,ibld->jbld", dist_weights, outputs)

        # --- Residual Addition ---
        # Apply per-stream scaling and residual
        # scale: (n_streams, 1, 1, 1) for broadcasting
        scale = self.stream_scales.view(n_streams, 1, 1, 1)

        # streams_out: (n_streams, batch, seq_len, d_model)
        streams_out = streams + self.dropout(distributed) * scale

        return streams_out


################################################################################
# SECTION 4: TESTING & DEMONSTRATION
################################################################################


def demonstrate_residual():
    """Demonstrate residual connections and hyper-connections."""
    print("=" * 70)
    print("RESIDUAL STREAM DEMONSTRATION")
    print("=" * 70)

    device = torch.device("cpu")
    batch_size = 2
    seq_len = 16
    d_model = 64

    # --- Standard Residual ---
    print("\n--- Standard Residual Stream ---")
    # Simple sublayer: a linear projection
    sublayer = nn.Linear(d_model, d_model)
    residual = ResidualStream(sublayer=sublayer, d_model=d_model)

    # x: (batch, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output = residual(x)
    print(f"  Input shape:           {x.shape}")
    print(f"  Output shape:          {output.shape}")
    print(f"  Norm before:           {x.norm(dim=-1).mean():.4f}")
    print(f"  Norm after:            {output.norm(dim=-1).mean():.4f}")

    # Verify residual connection: output should be close to input
    # when sublayer is near-zero
    with torch.no_grad():
        sublayer.weight.zero_()
        sublayer.bias.zero_()
    output_zero = residual(x)
    print(f"  With zero sublayer — output == input: {torch.allclose(output_zero, x, atol=1e-6)}")

    # --- Hyper-Connections ---
    print("\n--- Hyper-Connections ---")
    n_streams = 3
    sublayer_hc = nn.Linear(d_model, d_model)
    hyper = HyperConnections(
        sublayer=sublayer_hc,
        d_model=d_model,
        n_streams=n_streams,
    )

    # streams: (n_streams, batch, seq_len, d_model)
    streams = torch.randn(n_streams, batch_size, seq_len, d_model, device=device)
    streams_out = hyper(streams)
    print(f"  Input streams shape:   {streams.shape}")
    print(f"  Output streams shape:  {streams_out.shape}")
    print(f"  Number of streams:     {n_streams}")

    # Show mixing weights
    mix_weights = torch.softmax(hyper.W_mix, dim=-1)
    print(f"  Mixing matrix (learned):")
    for i in range(n_streams):
        row = [f"{mix_weights[i, j]:.3f}" for j in range(n_streams)]
        print(f"    Stream {i} reads from: [{', '.join(row)}]")

    # Show distribution weights
    dist_weights = torch.softmax(hyper.W_dist, dim=-1)
    print(f"  Distribution matrix (learned):")
    for i in range(n_streams):
        row = [f"{dist_weights[i, j]:.3f}" for j in range(n_streams)]
        print(f"    Output {i} writes to: [{', '.join(row)}]")

    # --- RMSNorm ---
    print("\n--- RMSNorm ---")
    norm = RMSNorm(d_model)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    x_normed = norm(x)
    print(f"  Input shape:           {x.shape}")
    print(f"  Output shape:          {x_normed.shape}")
    print(f"  Input RMS:             {x.pow(2).mean(dim=-1).sqrt().mean():.4f}")
    print(f"  Output RMS:            {x_normed.pow(2).mean(dim=-1).sqrt().mean():.4f}")

    print("\n" + "=" * 70)
    print("All residual demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_residual()
