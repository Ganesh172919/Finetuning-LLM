"""
################################################################################
LINEAR ATTENTION — EFFICIENT ALTERNATIVE TO STANDARD ATTENTION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Linear Attention?
    Linear attention achieves O(n) complexity instead of O(n²)
    by decomposing the attention computation.

Why does it matter?
    Standard attention: O(n²) in sequence length
    Linear attention: O(n) in sequence length

    For long sequences (100K+ tokens), this is essential.

Interview Questions:
    1. "What is linear attention?"
        Attention with O(n) complexity by decomposing the softmax.

    2. "What are the tradeoffs?"
        Faster but may lose some expressiveness.

################################################################################
"""

import numpy as np
from typing import Optional
import math

################################################################################
# SECTION 1: LINEAR ATTENTION
################################################################################

class LinearAttention:
    """
    Linear Attention
    ================

    Decomposes softmax attention into linear operations.

    Standard: softmax(QK^T)V — O(n²)
    Linear: φ(Q)(φ(K)^T V) — O(n)

    Where φ is a feature map (e.g., elu + 1).
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02

    def feature_map(self, x: np.ndarray) -> np.ndarray:
        """Apply feature map (elu + 1)."""
        return np.maximum(0, x) + 1  # Simplified

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Linear attention forward pass.

        Args:
            x: [batch × seq × d_model]

        Returns:
            output: [batch × seq × d_model]
        """
        batch, seq, d = x.shape

        Q = np.matmul(x, self.W_Q)
        K = np.matmul(x, self.W_K)
        V = np.matmul(x, self.W_V)

        # Apply feature map
        Q = self.feature_map(Q)
        K = self.feature_map(K)

        # Linear attention: φ(Q)(φ(K)^T V)
        # First compute K^T V: [batch × d × d]
        KV = np.matmul(K.transpose(0, 2, 1), V)

        # Then Q @ KV: [batch × seq × d]
        output = np.matmul(Q, KV)

        # Normalize
        normalizer = np.matmul(Q, np.sum(K, axis=1, keepdims=True).T)
        output = output / (normalizer + 1e-8)

        return output


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_linear_attention():
    """Demonstrate linear attention."""
    print("=" * 70)
    print("LINEAR ATTENTION DEMONSTRATION")
    print("=" * 70)

    attn = LinearAttention(d_model=64, n_heads=4)
    x = np.random.randn(2, 100, 64)

    output = attn.forward(x)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")

    # Complexity comparison
    seq_len = 1000
    print(f"\nStandard attention: O({seq_len}²) = {seq_len**2:,}")
    print(f"Linear attention: O({seq_len}) = {seq_len:,}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_linear_attention()


################################################################################
# REFERENCES
################################################################################

# [1] Katharopoulos, A., et al. (2020). Transformers are RNNs.

################################################################################
