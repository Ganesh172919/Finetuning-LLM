"""
################################################################################
GRADIENT ACCUMULATION — SIMULATING LARGE BATCHES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Gradient Accumulation?
    Accumulating gradients over multiple mini-batches before updating.
    Simulates larger batch sizes without extra memory.

Interview Questions:
    1. "What is gradient accumulation?"
        Accumulate gradients over N steps before updating.
        Effective batch size = mini_batch × N.

################################################################################
"""

import numpy as np
from typing import List

################################################################################
# SECTION 1: GRADIENT ACCUMULATOR
################################################################################

class GradientAccumulator:
    """
    Gradient Accumulator
    ====================

    Accumulates gradients over multiple steps.
    """

    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.accumulated = []
        self.step_count = 0

    def accumulate(self, gradients: List[np.ndarray]):
        """Accumulate gradients."""
        if not self.accumulated:
            self.accumulated = [np.zeros_like(g) for g in gradients]

        for i, grad in enumerate(gradients):
            self.accumulated[i] += grad / self.accumulation_steps

        self.step_count += 1

    def should_update(self) -> bool:
        """Check if should update weights."""
        return self.step_count >= self.accumulation_steps

    def get_gradients(self) -> List[np.ndarray]:
        """Get accumulated gradients and reset."""
        grads = self.accumulated
        self.accumulated = []
        self.step_count = 0
        return grads


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_gradient_accumulation():
    """Demonstrate gradient accumulation."""
    print("=" * 70)
    print("GRADIENT ACCUMULATION DEMONSTRATION")
    print("=" * 70)

    ga = GradientAccumulator(accumulation_steps=4)

    for i in range(4):
        grads = [np.random.randn(10, 10)]
        ga.accumulate(grads)
        print(f"Step {i+1}: should_update={ga.should_update()}")

    final = ga.get_gradients()
    print(f"Final gradient shape: {final[0].shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_gradient_accumulation()
