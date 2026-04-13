"""
################################################################################
RWKV — RECEPTANCE WEIGHTED KEY VALUE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is RWKV?
    RWKV combines the best of RNNs and transformers:
    - RNN-like: O(n) complexity, no KV cache
    - Transformer-like: parallel training

Why does it matter?
    RWKV offers:
    - Linear complexity
    - No KV cache needed
    - Efficient inference
    - Competitive quality

Interview Questions:
    1. "What is RWKV?"
        A model that combines RNN efficiency with transformer quality.

    2. "How does RWKV work?"
        Uses attention-like mechanism but with linear complexity.
        Can be trained in parallel like transformers.

################################################################################
"""

import numpy as np
from typing import Optional
import math

################################################################################
# SECTION 1: RWKV MODEL
################################################################################

class RWKV:
    """
    RWKV: Receptance Weighted Key Value
    ====================================

    Combines RNN and transformer properties.

    Key innovation: Attention-like mechanism with O(n) complexity.
    """

    def __init__(self, d_model: int, n_layers: int = 6):
        self.d_model = d_model
        self.n_layers = n_layers

        # Parameters
        self.W_R = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02

    def time_mixing(self, x: np.ndarray) -> np.ndarray:
        """
        RWKV time mixing (attention alternative).

        Args:
            x: [batch × seq × d_model]

        Returns:
            output: [batch × seq × d_model]
        """
        batch, seq, d = x.shape

        R = np.matmul(x, self.W_R)  # Receptance
        K = np.matmul(x, self.W_K)  # Key
        V = np.matmul(x, self.W_V)  # Value

        # Simplified RWKV computation
        # Real implementation uses more complex recurrence
        output = R * (K @ V.T)  # Simplified

        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return self.time_mixing(x)


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_rwkv():
    """Demonstrate RWKV concepts."""
    print("=" * 70)
    print("RWKV DEMONSTRATION")
    print("=" * 70)

    rwkv = RWKV(d_model=64, n_layers=4)
    x = np.random.randn(2, 10, 64)
    output = rwkv.forward(x)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_rwkv()


################################################################################
# REFERENCES
################################################################################

# [1] Peng, B., et al. (2023). RWKV: Reinventing RNNs for the Transformer Era.

################################################################################
