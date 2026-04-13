"""
DoRA: Weight-Decomposed Low-Rank Adaptation
=============================================

DoRA decomposes the weight update into magnitude and direction components,
then applies LoRA only to the direction component.

Key Insight:
  LoRA updates weights as: W_new = W + B @ A
  DoRA decomposes weights as: W = m × (V / ||V||)
    - m = magnitude (learned scalar per output dimension)
    - V = direction (the weight matrix normalized)
  Then applies LoRA only to the direction:
    W_new = m × ((W + B @ A) / ||W + B @ A||)

Benefits:
  - Better than LoRA at same rank
  - More stable training
  - Closer to full fine-tuning behavior
  - Can use lower rank than LoRA for same quality

References:
  - Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DoRAConfig:
    """
    Configuration for DoRA adapter.

    Attributes:
        in_features: Input dimension
        out_features: Output dimension
        rank: Low-rank dimension
        alpha: Scaling factor
        dropout: Dropout probability
    """
    in_features: int = 768
    out_features: int = 768
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0


# ============================================================================
# DORA ADAPTER
# ============================================================================

class DoRAAdapter:
    """
    DoRA: Weight-Decomposed Low-Rank Adaptation.

    Decomposes weights into magnitude and direction:
        W = m × (V / ||V||)

    Then applies LoRA to direction only:
        V_new = V + B @ A
        W_new = m × (V_new / ||V_new||)

    This decomposition allows:
    - Learning magnitude changes independently
    - More stable direction updates via LoRA
    - Better approximation of full fine-tuning
    """

    def __init__(self, config: DoRAConfig):
        """
        Initialize DoRA adapter.

        Args:
            config: DoRA configuration
        """
        self.config = config
        rank = config.rank
        in_features = config.in_features
        out_features = config.out_features

        # Pre-trained weight (frozen)
        self.weight = np.random.randn(out_features, in_features) * 0.01

        # Magnitude vector (trainable)
        # Initialized to norm of pre-trained weight
        self.magnitude = np.linalg.norm(self.weight, axis=1)

        # Direction LoRA matrices
        self.A = np.random.randn(in_features, rank) * 0.01
        self.B = np.zeros((rank, out_features))

        # Scaling
        self.scaling = config.alpha / config.rank

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through DoRA.

        Args:
            x: Input [batch, in_features]

        Returns:
            DoRA output [batch, out_features]
        """
        # LoRA update for direction
        # A: [in_features, rank], B: [rank, out_features]
        # delta_V = A @ B = [in_features, out_features]
        delta_V = self.A @ self.B  # [in_features, out_features]
        V_new = self.weight + delta_V.T * self.scaling

        # Normalize direction
        V_norm = np.linalg.norm(V_new, axis=1, keepdims=True)
        V_normalized = V_new / (V_norm + 1e-8)

        # Apply magnitude
        W_new = self.magnitude[:, np.newaxis] * V_normalized

        # Forward pass
        output = x @ W_new.T

        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return self.magnitude.size + self.A.size + self.B.size


# ============================================================================
# COMPARISON
# ============================================================================

def compare_dora_lora():
    """Compare DoRA with LoRA."""
    return """
    ┌──────────────────┬─────────────┬─────────────┐
    │ Property         │ LoRA        │ DoRA        │
    ├──────────────────┼─────────────┼─────────────┤
    │ Update           │ W + B @ A   │ m × dir(W + B @ A) │
    │ Components       │ ΔW only     │ Magnitude + Direction │
    │ Stability        │ Good        │ Better      │
    │ Quality at rank 8│ ~99%        │ ~99.5%      │
    │ Min effective rank│ 8           │ 4-6         │
    │ Training         │ Simple      │ Simple      │
    └──────────────────┴─────────────┴─────────────┘
    """


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_dora():
    """
    Demonstrate DoRA adapter.

    Shows:
        1. Weight decomposition
        2. Forward pass
        3. Parameter counting
        4. Comparison with LoRA
    """
    print("=" * 70)
    print("DoRA: Weight-Decomposed Low-Rank Adaptation — Demonstration")
    print("=" * 70)

    # Configuration
    config = DoRAConfig(in_features=768, out_features=768, rank=8)

    # Create adapter
    adapter = DoRAAdapter(config)

    print(f"\nConfiguration:")
    print(f"  in_features: {config.in_features}")
    print(f"  out_features: {config.out_features}")
    print(f"  rank: {config.rank}")
    print(f"  alpha: {config.alpha}")

    # Weight decomposition
    print("\n[Weight Decomposition]")
    W = adapter.weight
    m = adapter.magnitude
    V_norm = np.linalg.norm(W, axis=1)

    print(f"  Weight shape: {W.shape}")
    print(f"  Magnitude shape: {m.shape}")
    print(f"  Magnitude range: [{m.min():.4f}, {m.max():.4f}]")

    # Parameter count
    params = adapter.count_parameters()
    lora_params = 2 * config.in_features * config.rank

    print(f"\nParameter Count:")
    print(f"  DoRA parameters: {params:,}")
    print(f"  LoRA parameters (same rank): {lora_params:,}")
    print(f"  Extra: magnitude vector ({config.out_features:,})")

    # Forward pass
    print("\n[Forward Pass]")
    batch_size = 4
    x = np.random.randn(batch_size, config.in_features)
    output = adapter.forward(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    # Comparison
    print("\n[DoRA vs LoRA Comparison]")
    print(compare_dora_lora())

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. DoRA decomposes weights into magnitude + direction")
    print("  2. LoRA updates direction, magnitude learned separately")
    print("  3. More stable training than LoRA")
    print("  4. Can use lower rank for same quality")
    print("  5. Closer to full fine-tuning behavior")
    print("=" * 70)


if __name__ == "__main__":
    demo_dora()
