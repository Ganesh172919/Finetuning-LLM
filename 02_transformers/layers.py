"""
################################################################################
TRANSFORMER LAYERS — BUILDING BLOCKS OF MODERN AI
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Transformer Layers?
    Transformer layers are the repeated blocks that make up a transformer model.
    Each layer has two main components:
    1. Self-Attention: tokens look at each other
    2. Feed-Forward Network: process each token independently

    A transformer stacks N of these layers (e.g., 32 for LLaMA-7B).

Why do we need them?
    Each layer adds capacity to the model:
    - Layer 1-3: Basic patterns (word boundaries, syntax)
    - Layer 4-8: Syntactic relationships (subject-verb agreement)
    - Layer 9-16: Semantic understanding (meaning, context)
    - Layer 17-32: High-level reasoning (inference, planning)

    More layers = more capacity = better understanding.

How do they work?
    For each layer:
    1. Self-Attention: x = x + Attention(LayerNorm(x))
    2. Feed-Forward: x = x + FFN(LayerNorm(x))

    The residual connections (+ x) are crucial:
    - They allow gradients to flow directly through the network
    - Without them, deep networks would be untrainable

########################################

LAYERS IN THIS FILE:

1. RMSNorm (Root Mean Square Normalization)
2. FeedForward (Position-wise Feed-Forward)
3. GatedLinearUnit (SwiGLU, used in LLaMA/Mistral)
4. TransformerBlock (Complete transformer layer)

################################################################################
"""

import numpy as np
from typing import Optional
import math

################################################################################
# SECTION 1: NORMALIZATION LAYERS
################################################################################

class RMSNorm:
    """
    Root Mean Square Layer Normalization (RMSNorm)
    ================================================

    Definition: Normalize activations by their root mean square,
    without subtracting the mean (unlike LayerNorm).

    Formula:
        RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ

    Where:
        - mean(x²) = average of squared values
        - ε = small constant for numerical stability (1e-6)
        - γ = learned scale parameter (one per feature)

    WHY RMSNorm INSTEAD OF LayerNorm?
    ==================================
    LayerNorm: y = (x - mean(x)) / sqrt(var(x) + ε) * γ + β
    RMSNorm:   y = x / sqrt(mean(x²) + ε) * γ

    Differences:
    1. RMSNorm doesn't subtract mean (no centering)
    2. RMSNorm doesn't have bias term (no β)
    3. RMSNorm is 10-15% faster (fewer operations)
    4. RMSNorm works just as well in practice

    Used by:
    - LLaMA (all sizes)
    - Mistral
    - Qwen
    - DeepSeek
    - Most modern LLMs

    Why it works:
    The key insight is that re-centering (subtracting mean) isn't
    necessary for transformers. What matters is the scale of activations,
    which RMSNorm controls effectively.

    Interview Question:
        "What's the difference between LayerNorm and RMSNorm?"
        LayerNorm normalizes to zero mean and unit variance.
        RMSNorm only normalizes to unit scale (no centering).
        RMSNorm is faster and works equally well in transformers.
        Both ensure stable training by controlling activation scales.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.

        Args:
            d_model: Feature dimension
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        # Learnable scale parameter (initialized to 1)
        self.weight = np.ones(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply RMSNorm.

        Args:
            x: Input tensor [... × d_model]

        Returns:
            Normalized tensor [... × d_model]

        Steps:
            1. Compute mean of squares: ms = mean(x²)
            2. Compute RMS: rms = sqrt(ms + ε)
            3. Normalize: x_norm = x / rms
            4. Scale: output = x_norm * γ
        """
        # Step 1: Mean of squares
        # Keep last dimension for broadcasting
        ms = np.mean(x ** 2, axis=-1, keepdims=True)

        # Step 2: Root mean square
        rms = np.sqrt(ms + self.eps)

        # Step 3: Normalize and scale
        x_norm = x / rms
        return x_norm * self.weight


class LayerNorm:
    """
    Layer Normalization
    ===================

    Definition: Normalize activations to zero mean and unit variance.

    Formula:
        LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + ε) * γ + β

    This was the standard normalization before RMSNorm.

    Used by:
    - BERT
    - GPT-2
    - T5
    - Original Transformer
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        self.weight = np.ones(d_model)  # γ
        self.bias = np.zeros(d_model)    # β

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Layer Normalization.

        Args:
            x: Input [... × d_model]
        Returns:
            Normalized [... × d_model]
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


################################################################################
# SECTION 2: FEED-FORWARD NETWORKS
################################################################################

class FeedForward:
    """
    Position-wise Feed-Forward Network
    ====================================

    Definition: A two-layer MLP applied to each position independently.

    Architecture:
        FFN(x) = W₂ @ activation(W₁ @ x + b₁) + b₂

    Where:
        W₁ ∈ R^{d_model × d_ff} — expand dimension
        W₂ ∈ R^{d_ff × d_model} — project back
        d_ff = 4 × d_model (standard ratio)
        activation = ReLU (original) or GELU (GPT-2+) or SwiGLU (LLaMA)

    Why does it exist?
    The attention layer mixes information across positions.
    The FFN processes information within each position.

    Think of it as:
    - Attention: "look at other tokens" (cross-position)
    - FFN: "think about this token" (within-position)

    The FFN is where most of the model's parameters live!
    For d_model=4096:
    - Attention: 4096² × 4 = 67M params (4 heads)
    - FFN: 4096 × 16384 × 2 = 134M params

    Interview Questions:
        1. "Why is d_ff = 4 × d_model?"
           Empirically found to work well. The ratio controls
           the model's capacity. Larger ratio = more parameters.

        2. "What does the FFN learn?"
           Research suggests FFNs store factual knowledge.
           "The Eiffel Tower is in Paris" is stored in FFN weights.

        3. "Why GELU instead of ReLU?"
           GELU is smooth and has non-zero gradients everywhere,
           which helps training. ReLU can "die" (zero gradient).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.0
    ):
        """
        Initialize Feed-Forward Network.

        Args:
            d_model: Model dimension
            d_ff: Hidden dimension (default: 4 × d_model)
            activation: Activation function ("relu", "gelu", "silu")
            dropout: Dropout probability
        """
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.activation = activation
        self.dropout = dropout

        # Initialize weights
        scale1 = math.sqrt(2.0 / (d_model + self.d_ff))
        scale2 = math.sqrt(2.0 / (self.d_ff + d_model))

        self.W1 = np.random.randn(d_model, self.d_ff) * scale1
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "gelu":
            # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        elif self.activation == "silu":
            # SiLU/Swish: x * sigmoid(x)
            return x / (1 + np.exp(-x))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of FFN.

        Args:
            x: Input [batch × seq × d_model]

        Returns:
            output: [batch × seq × d_model]

        Flow:
            x → W₁ → activation → W₂ → output
        """
        # Layer 1: expand
        h = np.matmul(x, self.W1) + self.b1  # [batch × seq × d_ff]

        # Activation
        h = self._activate(h)

        # Layer 2: project back
        output = np.matmul(h, self.W2) + self.b2  # [batch × seq × d_model]

        return output


class GatedLinearUnit:
    """
    Gated Linear Unit (SwiGLU)
    ===========================

    Definition: A variant of FFN that uses gating to control information flow.

    Formula:
        SwiGLU(x) = (Swish(x @ W_gate) ⊙ (x @ W_up)) @ W_down

    Where:
        ⊙ = element-wise multiplication
        Swish(x) = x * sigmoid(x)

    Architecture:
        x → W_gate → Swish ─┐
                              ⊙ → W_down → output
        x → W_up ───────────┘

    WHY GLU?
    =========
    Standard FFN: FFN(x) = W₂ @ activation(W₁ @ x)
    GLU: GLU(x) = W_down @ (activation(x @ W_gate) ⊙ (x @ W_up))

    The gate (W_gate) controls which features pass through.
    This is more expressive than standard FFN.

    Used by:
    - LLaMA (all sizes)
    - Mistral
    - Qwen
    - DeepSeek
    - Most modern LLMs (2023+)

    Performance:
    SwiGLU consistently outperforms standard FFN with ReLU/GELU.
    To match performance, reduce d_ff by ~2/3 (e.g., 4d → 8/3 d).

    Interview Question:
        "What is SwiGLU and why is it better than ReLU FFN?"
        SwiGLU uses a gating mechanism where one branch controls
        information flow through the other. This allows more
        selective and expressive processing. The "Swish" activation
        (x * sigmoid(x)) is smooth and works well in practice.
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        self.d_model = d_model
        # LLaMA uses 2/3 * 4 * d_model for SwiGLU
        self.d_ff = d_ff or int(2 / 3 * 4 * d_model)

        scale = math.sqrt(2.0 / (d_model + self.d_ff))
        self.W_gate = np.random.randn(d_model, self.d_ff) * scale
        self.W_up = np.random.randn(d_model, self.d_ff) * scale
        self.W_down = np.random.randn(self.d_ff, d_model) * scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of SwiGLU.

        Args:
            x: Input [batch × seq × d_model]

        Returns:
            output: [batch × seq × d_model]
        """
        # Gate branch: Swish(x @ W_gate)
        gate = np.matmul(x, self.W_gate)  # [batch × seq × d_ff]
        gate = gate / (1 + np.exp(-gate))  # Swish activation

        # Up branch: x @ W_up
        up = np.matmul(x, self.W_up)  # [batch × seq × d_ff]

        # Gate controls information flow
        hidden = gate * up  # Element-wise multiplication

        # Down projection
        output = np.matmul(hidden, self.W_down)  # [batch × seq × d_model]

        return output


################################################################################
# SECTION 3: TRANSFORMER BLOCK
################################################################################

class TransformerBlock:
    """
    Transformer Block
    ==================

    Definition: The fundamental repeated unit of a transformer.

    Architecture (Pre-Norm, used by GPT-2+, LLaMA, etc.):
    ┌─────────────────────────────────────────┐
    │ Input x                                   │
    │   │                                       │
    │   ├──→ RMSNorm ──→ Attention ──→ + ──→ h │
    │   │        ↑           │          ↑       │
    │   └────────┘           └──────────┘       │
    │   │                                       │
    │   ├──→ RMSNorm ──→ FFN/SwiGLU ──→ + ──→ out
    │   │        ↑           │          ↑       │
    │   └────────┘           └──────────┘       │
    └─────────────────────────────────────────┘

    Pre-Norm vs Post-Norm:
    - Pre-Norm: x + Attention(Norm(x)) — easier to train, used by GPT-2+
    - Post-Norm: Norm(x + Attention(x)) — original Transformer, harder to train

    Pre-Norm is preferred because:
    1. Gradients flow directly through residual connections
    2. No need for learning rate warmup
    3. More stable training for deep models

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension (default: 4 × d_model)
        n_kv_heads: Number of KV heads for GQA (None = MHA)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        n_kv_heads: Optional[int] = None,
        use_swiglu: bool = True,
        dropout: float = 0.0
    ):
        self.d_model = d_model

        # Pre-attention normalization
        self.norm1 = RMSNorm(d_model)

        # Self-attention
        if n_kv_heads is not None and n_kv_heads < n_heads:
            from .attention import GroupedQueryAttention
            self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
        else:
            from .attention import MultiHeadAttention
            self.attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Pre-FFN normalization
        self.norm2 = RMSNorm(d_model)

        # Feed-forward network
        if use_swiglu:
            self.ffn = GatedLinearUnit(d_model, d_ff)
        else:
            self.ffn = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass of transformer block.

        Args:
            x: Input [batch × seq × d_model]
            mask: Optional attention mask

        Returns:
            output: [batch × seq × d_model]

        The flow:
            1. Normalize → Attention → Residual
            2. Normalize → FFN → Residual
        """
        # Sub-layer 1: Self-Attention with residual
        h = self.norm1.forward(x)
        attn_out, _ = self.attention.forward(h, mask=mask)
        x = x + attn_out  # Residual connection

        # Sub-layer 2: FFN with residual
        h = self.norm2.forward(x)
        ffn_out = self.ffn.forward(h)
        x = x + ffn_out  # Residual connection

        return x


################################################################################
# SECTION 4: TESTING & EXAMPLES
################################################################################

def demonstrate_layers():
    """Demonstrate transformer layers."""
    print("=" * 70)
    print("TRANSFORMER LAYERS DEMONSTRATION")
    print("=" * 70)

    batch_size = 2
    seq_len = 4
    d_model = 64
    n_heads = 4

    x = np.random.randn(batch_size, seq_len, d_model)

    # RMSNorm
    print("\n--- RMSNorm ---")
    norm = RMSNorm(d_model)
    x_norm = norm.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_norm.shape}")
    print(f"Input mean (should be ~0): {x.mean():.4f}")
    print(f"Output RMS (should be ~1): {np.sqrt(np.mean(x_norm**2)):.4f}")

    # FeedForward
    print("\n--- FeedForward ---")
    ffn = FeedForward(d_model, d_ff=256)
    out_ffn = ffn.forward(x)
    print(f"FFN output shape: {out_ffn.shape}")

    # SwiGLU
    print("\n--- SwiGLU ---")
    swiglu = GatedLinearUnit(d_model)
    out_swiglu = swiglu.forward(x)
    print(f"SwiGLU output shape: {out_swiglu.shape}")
    print(f"SwiGLU d_ff: {swiglu.d_ff}")

    # TransformerBlock
    print("\n--- TransformerBlock ---")
    block = TransformerBlock(d_model, n_heads)
    from .attention import create_causal_mask
    mask = create_causal_mask(seq_len)
    out_block = block.forward(x, mask=mask)
    print(f"Block output shape: {out_block.shape}")

    # Verify residual connections preserve scale
    print(f"\nInput norm: {np.linalg.norm(x[0, 0]):.4f}")
    print(f"Block output norm: {np.linalg.norm(out_block[0, 0]):.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_layers()


################################################################################
# REFERENCES
################################################################################

# [1] Vaswani, A., et al. (2017). Attention Is All You Need.
# [2] Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
# [3] Shazeer, N. (2020). GLU Variants Improve Transformer.
# [4] Su, J. (2024). RMSNorm and SwiGLU: Why Modern LLMs Use Them.

################################################################################
