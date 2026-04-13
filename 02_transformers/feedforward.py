"""
################################################################################
FEED FORWARD NETWORK — POSITION-WISE MLP
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Feed Forward Network?
    A position-wise MLP applied to each token independently.

Architecture:
    FFN(x) = W2 @ activation(W1 @ x + b1) + b2

Why it matters:
    - Attention mixes information across positions
    - FFN processes information within each position

Interview Questions:
    Q: "What does the FFN learn in transformers?"
    A: Research suggests FFNs store factual knowledge.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: FEED FORWARD NETWORK
################################################################################

class FeedForward:
    """
    Feed Forward Network
    ====================

    Position-wise MLP.
    """

    def __init__(self, d_model: int, d_ff: int = None):
        self.d_ff = d_ff or 4 * d_model
        self.W1 = np.random.randn(d_model, self.d_ff) * 0.02
        self.W2 = np.random.randn(self.d_ff, d_model) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        h = np.maximum(0, x @ self.W1)  # ReLU
        return h @ self.W2


class SwiGLU:
    """
    SwiGLU Activation
    =================

    Used by LLaMA, Mistral.
    """

    def __init__(self, d_model: int, d_ff: int = None):
        self.d_ff = d_ff or int(2 / 3 * 4 * d_model)
        self.W_gate = np.random.randn(d_model, self.d_ff) * 0.02
        self.W_up = np.random.randn(d_model, self.d_ff) * 0.02
        self.W_down = np.random.randn(self.d_ff, d_model) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        gate = x @ self.W_gate
        gate = gate / (1 + np.exp(-gate))  # Swish
        up = x @ self.W_up
        return (gate * up) @ self.W_down


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_feedforward():
    """Demonstrate feed forward network."""
    print("=" * 70)
    print("FEED FORWARD NETWORK DEMONSTRATION")
    print("=" * 70)

    # FFN
    print("\n--- FFN ---")
    ffn = FeedForward(d_model=64)
    x = np.random.randn(4, 64)
    y = ffn.forward(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")

    # SwiGLU
    print("\n--- SwiGLU ---")
    swiglu = SwiGLU(d_model=64)
    y = swiglu.forward(x)
    print(f"Output: {y.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_feedforward()
