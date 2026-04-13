"""
################################################################################
MIXED PRECISION TRAINING — fp16 + fp32
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Mixed Precision?
    Using both fp16 and fp32 during training.
    fp16: faster, less memory
    fp32: more precise

Interview Questions:
    1. "What is mixed precision training?"
        Using fp16 for most ops, fp32 for critical ones.

################################################################################
"""

import numpy as np
from typing import List

################################################################################
# SECTION 1: MIXED PRECISION TRAINER
################################################################################

class MixedPrecisionTrainer:
    """
    Mixed Precision Trainer
    ========================

    Manages fp16/fp32 training.
    """

    def __init__(self, loss_scale: float = 1024.0):
        self.loss_scale = loss_scale

    def scale_loss(self, loss: float) -> float:
        """Scale loss to prevent fp16 underflow."""
        return loss * self.loss_scale

    def unscale_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Unscale gradients."""
        return [g / self.loss_scale for g in gradients]

    def clip_gradients(self, gradients: List[np.ndarray], max_norm: float = 1.0) -> List[np.ndarray]:
        """Clip gradients by norm."""
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            gradients = [g * scale for g in gradients]
        return gradients


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_mixed_precision():
    """Demonstrate mixed precision."""
    print("=" * 70)
    print("MIXED PRECISION DEMONSTRATION")
    print("=" * 70)

    mp = MixedPrecisionTrainer(loss_scale=1024.0)

    loss = 0.001
    scaled = mp.scale_loss(loss)
    print(f"Original loss: {loss}")
    print(f"Scaled loss: {scaled}")

    grads = [np.random.randn(10, 10) * 0.0001]
    unscaled = mp.unscale_gradients(grads)
    print(f"Unscaled grad norm: {np.linalg.norm(unscaled[0]):.6f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_mixed_precision()
