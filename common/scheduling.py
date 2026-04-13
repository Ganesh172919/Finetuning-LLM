"""
################################################################################
SCHEDULING — LEARNING RATE AND NOISE SCHEDULES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Schedules?
    Functions that control values over time (e.g., learning rate).

Key Schedules:
    - Cosine: Smooth decay
    - Linear: Linear decay
    - Warmup: Gradual increase

Interview Questions:
    Q: "What learning rate schedule should I use?"
    A: Warmup + cosine decay is standard for LLMs.

################################################################################
"""

import numpy as np
import math

################################################################################
# SECTION 1: SCHEDULES
################################################################################

class CosineSchedule:
    """
    Cosine Schedule
    ===============

    Smooth decay following cosine curve.
    """

    def __init__(self, max_val: float, min_val: float, warmup_steps: int, total_steps: int):
        self.max_val = max_val
        self.min_val = min_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_value(self, step: int) -> float:
        """Get value at given step."""
        if step < self.warmup_steps:
            return self.max_val * (step / self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_val + 0.5 * (self.max_val - self.min_val) * (1 + math.cos(math.pi * progress))


class LinearSchedule:
    """
    Linear Schedule
    ===============

    Linear decay.
    """

    def __init__(self, max_val: float, min_val: float, total_steps: int):
        self.max_val = max_val
        self.min_val = min_val
        self.total_steps = total_steps

    def get_value(self, step: int) -> float:
        """Get value at given step."""
        progress = min(step / self.total_steps, 1.0)
        return self.max_val + (self.min_val - self.max_val) * progress


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_scheduling():
    """Demonstrate scheduling."""
    print("=" * 70)
    print("SCHEDULING DEMONSTRATION")
    print("=" * 70)

    # Cosine
    print("\n--- Cosine Schedule ---")
    cosine = CosineSchedule(max_val=3e-4, min_val=3e-5, warmup_steps=100, total_steps=1000)
    for step in [0, 50, 100, 500, 999]:
        print(f"Step {step}: {cosine.get_value(step):.6f}")

    # Linear
    print("\n--- Linear Schedule ---")
    linear = LinearSchedule(max_val=3e-4, min_val=3e-5, total_steps=1000)
    for step in [0, 500, 999]:
        print(f"Step {step}: {linear.get_value(step):.6f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_scheduling()
