"""
################################################################################
FLOW MATCHING — STRAIGHT-LINE GENERATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Flow Matching?
    A generative model that uses straight-line interpolation
    between noise and data.

Advantages over diffusion:
    - Straighter paths → fewer steps needed
    - Simpler training
    - Better for certain tasks

Interview Questions:
    1. "What is flow matching?"
        A generative model using straight-line paths between noise and data.

################################################################################
"""

import numpy as np
from typing import Tuple

################################################################################
# SECTION 1: FLOW MATCHING
################################################################################

class FlowMatching:
    """
    Flow Matching Model
    ====================

    Uses straight-line interpolation for generation.
    """

    def __init__(self):
        pass

    def interpolate(self, x_0: np.ndarray, x_1: np.ndarray, t: float) -> np.ndarray:
        """
        Straight-line interpolation.

        x_t = (1-t) * x_0 + t * x_1

        Where x_0 is noise, x_1 is data.
        """
        return (1 - t) * x_0 + t * x_1

    def target_velocity(self, x_0: np.ndarray, x_1: np.ndarray) -> np.ndarray:
        """
        Target velocity for training.

        v = x_1 - x_0
        """
        return x_1 - x_0

    def generate(self, shape: Tuple, n_steps: int = 20) -> np.ndarray:
        """Generate sample using Euler integration."""
        x = np.random.randn(*shape)  # Start with noise

        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = step / n_steps
            # Predicted velocity (simplified)
            v = -x  # Simplified
            x = x + v * dt

        return x


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_flow_matching():
    """Demonstrate flow matching."""
    print("=" * 70)
    print("FLOW MATCHING DEMONSTRATION")
    print("=" * 70)

    fm = FlowMatching()

    x_0 = np.random.randn(3, 3)  # Noise
    x_1 = np.random.randn(3, 3)  # Data

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x_t = fm.interpolate(x_0, x_1, t)
        print(f"t={t:.2f}: x_t mean={x_t.mean():.3f}")

    v = fm.target_velocity(x_0, x_1)
    print(f"Target velocity: {v.mean():.3f}")

    sample = fm.generate((1, 3, 16, 16), n_steps=10)
    print(f"Generated: {sample.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_flow_matching()


################################################################################
# REFERENCES
################################################################################

# [1] Lipman, Y., et al. (2023). Flow Matching for Generative Modeling.

################################################################################
