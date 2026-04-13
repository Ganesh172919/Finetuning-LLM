"""
################################################################################
LoRA TRAINING — LOW-RANK ADAPTATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is LoRA?
    Low-Rank Adaptation: fine-tune models efficiently by
    learning low-rank decomposition of weight updates.

Formula:
    W' = W + (α/r) × A × B

Where:
    A ∈ R^{d×r}, B ∈ R^{r×d}, r << d

Benefits:
    - Much fewer parameters (120x reduction)
    - Faster training
    - Less memory
    - Easy to swap (multiple LoRAs for one base model)

Interview Questions:
        Q: "What is LoRA?"
        A: Low-Rank Adaptation. Instead of fine-tuning all weights,
           learn a low-rank update. Much more efficient.

        Q: "How does LoRA work?"
        A: Freeze original weights W. Learn A and B such that
           the update ΔW = A × B is low-rank.

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

    Interview Questions:
        Q: "What rank should I use for LoRA?"
        A: Typically 4-64. Higher rank = more capacity but more parameters.
           8-16 works well for most tasks.

        Q: "Can I combine multiple LoRAs?"
        A: Yes! Can merge, interpolate, or switch between them.
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.A = np.random.randn(d_model, rank) * 0.01
        self.B = np.zeros((rank, d_model))  # Initialize B to zero

        # Optional dropout
        self.dropout = dropout

    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Apply LoRA.

        Args:
            x: Input
            W: Original frozen weight

        Returns:
            output: x @ (W + scaling * A @ B)
        """
        # Original output
        original = x @ W

        # LoRA update
        lora_out = (x @ self.A @ self.B) * self.scaling

        return original + lora_out

    def merge_weights(self, W: np.ndarray) -> np.ndarray:
        """
        Merge LoRA weights into original.

        Returns merged weight for inference.
        """
        return W + self.scaling * (self.A @ self.B)


################################################################################
# SECTION 2: LoRA TRAINER
################################################################################

class LoRATrainer:
    """
    LoRA Trainer
    ============

    Trains only LoRA parameters while keeping base model frozen.

    Interview Questions:
        Q: "How do you train LoRA?"
        A: Freeze base model, only update A and B matrices.
           Much less memory and compute than full fine-tuning.
    """

    def __init__(self, lora_layer: LoRALayer, learning_rate: float = 1e-4):
        self.lora = lora_layer
        self.lr = learning_rate

    def train_step(self, x: np.ndarray, W: np.ndarray, target: np.ndarray) -> float:
        """
        Single training step.

        Args:
            x: Input
            W: Frozen weights
            target: Target output

        Returns:
            loss: Training loss
        """
        # Forward
        output = self.lora.forward(x, W)

        # Loss
        loss = np.mean((output - target) ** 2)

        # Simplified gradient update
        grad = 2 * (output - target) / len(x)
        self.lora.A -= self.lr * (x.T @ grad @ self.lora.B.T) * self.lora.scaling
        self.lora.B -= self.lr * (self.lora.A.T @ x.T @ grad) * self.lora.scaling

        return loss


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_lora():
    """Demonstrate LoRA."""
    print("=" * 70)
    print("LoRA TRAINING DEMONSTRATION")
    print("=" * 70)

    # Create LoRA layer
    d_model = 64
    rank = 8
    lora = LoRALayer(d_model, rank=rank, alpha=16.0)

    # Frozen weights
    W = np.random.randn(d_model, d_model) * 0.1

    # Forward pass
    x = np.random.randn(4, d_model)
    output = lora.forward(x, W)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"LoRA params: {d_model * rank * 2} vs full {d_model * d_model}")

    # Training
    print("\n--- Training ---")
    trainer = LoRATrainer(lora, learning_rate=1e-3)
    target = np.random.randn(4, d_model)
    loss = trainer.train_step(x, W, target)
    print(f"Training loss: {loss:.4f}")

    # Merge weights
    print("\n--- Merge ---")
    merged = lora.merge_weights(W)
    print(f"Merged shape: {merged.shape}")

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
