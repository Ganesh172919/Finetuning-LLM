"""
################################################################################
LoRA — LOW-RANK ADAPTATION FOR IMAGE MODELS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is LoRA?
    Low-Rank Adaptation: fine-tune models efficiently by
    learning low-rank decomposition of weight updates.

LoRA: W' = W + A @ B
Where A ∈ R^{d×r}, B ∈ R^{r×d}, r << d

Benefits:
    - Much fewer parameters to train
    - Faster training
    - Less memory

################################################################################
"""

import numpy as np
from typing import Optional
import math

################################################################################
# SECTION 1: LoRA LAYER
################################################################################

class LoRALayer:
    """
    LoRA: Low-Rank Adaptation
    ===========================

    Adds trainable low-rank matrices to frozen weights.
    """

    def __init__(self, d_model: int, rank: int = 16, alpha: float = 16.0):
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.A = np.random.randn(d_model, rank) * 0.01
        self.B = np.random.randn(rank, d_model) * 0.01

    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Apply LoRA.

        Args:
            x: Input
            W: Original weight matrix

        Returns:
            Output with LoRA adaptation
        """
        # Original output
        original = np.matmul(x, W)

        # LoRA adaptation
        lora_out = np.matmul(np.matmul(x, self.A), self.B) * self.scaling

        return original + lora_out


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_lora():
    """Demonstrate LoRA."""
    print("=" * 70)
    print("LoRA DEMONSTRATION")
    print("=" * 70)

    lora = LoRALayer(d_model=64, rank=8)
    x = np.random.randn(2, 64)
    W = np.random.randn(64, 64) * 0.02

    output = lora.forward(x, W)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"LoRA params: {64*8 + 8*64} vs full {64*64}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_lora()


################################################################################
# REFERENCES
################################################################################

# [1] Hu, E.J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.

################################################################################
