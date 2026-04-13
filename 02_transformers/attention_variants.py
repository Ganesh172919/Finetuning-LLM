"""
################################################################################
ATTENTION VARIANTS — EFFICIENT ATTENTION MECHANISMS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Attention Variants?
    Different attention mechanisms that improve efficiency:
    - Sparse Attention: Attend to subset of tokens
    - Linear Attention: O(n) complexity
    - Sliding Window: Local attention
    - Dilated Attention: Skip tokens

Why Efficient Attention?
    Standard attention: O(n²) in sequence length
    For n=100K tokens: 100K² = 10B operations per head!

    Efficient attention reduces this to O(n) or O(n log n).

Interview Questions:
        Q: "What's the difference between sparse and linear attention?"
        A: Sparse: attend to selected tokens (still quadratic in worst case)
           Linear: decompose attention for O(n) complexity

################################################################################
"""

import numpy as np
from typing import Optional
import math

################################################################################
# SECTION 1: SLIDING WINDOW ATTENTION
################################################################################

class SlidingWindowAttention:
    """
    Sliding Window Attention
    =========================

    Each token attends only to nearby tokens within a window.

    Complexity: O(n × w) where w is window size

    Used by: Mistral, Longformer

    Interview Questions:
        Q: "What is sliding window attention?"
        A: Each token attends only to tokens within a fixed window.
           Reduces complexity from O(n²) to O(n×w).
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int = 256):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size

        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02
        self.W_O = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Sliding window attention forward pass.

        Args:
            x: [batch × seq × d_model]

        Returns:
            output: [batch × seq × d_model]
        """
        batch, seq, d = x.shape

        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        output = np.zeros_like(x)

        for i in range(seq):
            # Attend to tokens within window
            start = max(0, i - self.window_size // 2)
            end = min(seq, i + self.window_size // 2)

            q = Q[:, i:i+1, :]
            k = K[:, start:end, :]
            v = V[:, start:end, :]

            scores = q @ k.transpose(0, 2, 1) / math.sqrt(self.d_k)
            weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            weights = weights / np.sum(weights, axis=-1, keepdims=True)

            output[:, i:i+1, :] = weights @ v

        return output @ self.W_O


################################################################################
# SECTION 2: DILATED ATTENTION
################################################################################

class DilatedAttention:
    """
    Dilated Attention
    =================

    Attend to tokens at regular intervals (dilation).
    Captures long-range dependencies with fewer computations.

    Pattern: attend to every k-th token
    Complexity: O(n² / k)

    Interview Questions:
        Q: "What is dilated attention?"
        A: Attend to tokens at regular intervals instead of all tokens.
           Captures long-range dependencies efficiently.
    """

    def __init__(self, d_model: int, n_heads: int, dilation: int = 2):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dilation = dilation

        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Dilated attention forward pass.
        """
        batch, seq, d = x.shape

        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        # Dilated indices
        indices = np.arange(0, seq, self.dilation)

        K_dilated = K[:, indices, :]
        V_dilated = V[:, indices, :]

        scores = Q @ K_dilated.transpose(0, 2, 1) / math.sqrt(self.d_k)
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)

        return weights @ V_dilated


################################################################################
# SECTION 3: LONGformer ATTENTION
################################################################################

class LongformerAttention:
    """
    Longformer Attention
    ====================

    Combines local sliding window with global attention on selected tokens.

    Three types of attention:
    1. Sliding window: local context
    2. Global: selected tokens attend to all
    3. Dilated: long-range dependencies

    Interview Questions:
        Q: "How does Longformer handle long documents?"
        A: Uses sliding window for most tokens, global attention
           for special tokens (like CLS).
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int = 256):
        self.d_model = d_model
        self.window_size = window_size

        # Local attention
        self.local_attn = SlidingWindowAttention(d_model, n_heads, window_size)

    def forward(
        self,
        x: np.ndarray,
        global_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Longformer attention.

        Args:
            x: [batch × seq × d_model]
            global_mask: [seq] boolean mask for global tokens

        Returns:
            output: [batch × seq × d_model]
        """
        # Local sliding window attention
        return self.local_attn.forward(x)


################################################################################
# SECTION 4: TESTING
################################################################################

def demonstrate_attention_variants():
    """Demonstrate attention variants."""
    print("=" * 70)
    print("ATTENTION VARIANTS DEMONSTRATION")
    print("=" * 70)

    batch, seq, d = 1, 16, 32

    # Sliding window
    print("\n--- Sliding Window ---")
    swa = SlidingWindowAttention(d, n_heads=4, window_size=4)
    x = np.random.randn(batch, seq, d)
    out = swa.forward(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    # Dilated
    print("\n--- Dilated ---")
    da = DilatedAttention(d, n_heads=4, dilation=2)
    out = da.forward(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    # Longformer
    print("\n--- Longformer ---")
    lfa = LongformerAttention(d, n_heads=4, window_size=4)
    out = lfa.forward(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    # Complexity comparison
    print("\n--- Complexity ---")
    for n in [1000, 10000, 100000]:
        standard = n * n
        sliding = n * 256
        print(f"n={n}: Standard={standard/1e6:.1f}M, Sliding={sliding/1e6:.1f}M")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_attention_variants()


################################################################################
# REFERENCES
################################################################################

# [1] Beltagy, I., et al. (2020). Longformer: The Long-Document Transformer.
# [2] Zaheer, M., et al. (2020). Big Bird: Transformers for Longer Sequences.

################################################################################
