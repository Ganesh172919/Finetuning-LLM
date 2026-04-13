"""
################################################################################
MODEL ARCHITECTURE — BUILDING BLOCKS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Model Architecture?
    The structure and components of neural networks.

Key Components:
    - Layers (linear, attention, etc.)
    - Activations (ReLU, GELU, etc.)
    - Normalization (LayerNorm, RMSNorm)
    - Residual connections

Interview Questions:
    Q: "What makes a good model architecture?"
    A: Balance between capacity and efficiency,
       proper normalization, residual connections.

################################################################################
"""

import numpy as np
import math

################################################################################
# SECTION 1: BUILDING BLOCKS
################################################################################

class Linear:
    """Linear layer: y = x @ W + b"""

    def __init__(self, in_features: int, out_features: int):
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight + self.bias


class LayerNorm:
    """Layer normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.weight = np.ones(d_model)
        self.bias = np.zeros(d_model)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.weight * (x - mean) / np.sqrt(var + self.eps) + self.bias


class RMSNorm:
    """RMS normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.weight = np.ones(d_model)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return self.weight * x / rms


class GELU:
    """GELU activation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class SiLU:
    """SiLU/Swish activation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x / (1 + np.exp(-x))


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_architecture():
    """Demonstrate model architecture."""
    print("=" * 70)
    print("MODEL ARCHITECTURE DEMONSTRATION")
    print("=" * 70)

    # Linear
    print("\n--- Linear ---")
    linear = Linear(64, 32)
    x = np.random.randn(4, 64)
    y = linear.forward(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")

    # LayerNorm
    print("\n--- LayerNorm ---")
    ln = LayerNorm(64)
    y = ln.forward(x)
    print(f"Mean: {np.mean(y, axis=-1).round(4)}")
    print(f"Std: {np.std(y, axis=-1).round(4)}")

    # RMSNorm
    print("\n--- RMSNorm ---")
    rms = RMSNorm(64)
    y = rms.forward(x)
    print(f"RMS: {np.sqrt(np.mean(y**2, axis=-1)).round(4)}")

    # Activations
    print("\n--- Activations ---")
    gelu = GELU()
    silu = SiLU()
    x = np.array([-2, -1, 0, 1, 2])
    print(f"GELU: {gelu.forward(x).round(4)}")
    print(f"SiLU: {silu.forward(x).round(4)}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_architecture()
