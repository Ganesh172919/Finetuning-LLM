"""
################################################################################
TRAINING CURRICULUM — ORDERING TRAINING DATA
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Training Curriculum?
    Ordering training data from easy to hard.

Benefits:
    - Faster convergence
    - Better generalization
    - More stable training

Interview Questions:
    Q: "What is curriculum learning?"
    A: Training on easier examples first, then harder ones.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: CURRICULUM
################################################################################

class CurriculumScheduler:
    """
    Curriculum Scheduler
    ====================

    Schedules training data difficulty.
    """

    def __init__(self, total_steps: int):
        self.total_steps = total_steps

    def get_difficulty(self, step: int) -> float:
        """Get current difficulty level."""
        return min(1.0, step / self.total_steps)

    def filter_data(self, data: np.ndarray, difficulty: float) -> np.ndarray:
        """Filter data by difficulty."""
        n_samples = int(len(data) * difficulty)
        return data[:n_samples]


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_curriculum():
    """Demonstrate curriculum learning."""
    print("=" * 70)
    print("TRAINING CURRICULUM DEMONSTRATION")
    print("=" * 70)

    scheduler = CurriculumScheduler(total_steps=1000)

    for step in [0, 250, 500, 750, 999]:
        difficulty = scheduler.get_difficulty(step)
        print(f"Step {step}: difficulty={difficulty:.2f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_curriculum()
