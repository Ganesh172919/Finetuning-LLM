"""
################################################################################
SCORE MATCHING — LEARNING THE SCORE FUNCTION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Score Matching?
    Learning the gradient of log probability density.

Score: s(x) = ∇_x log p(x)

Interview Questions:
    1. "What is score-based modeling?"
        Learn the score function (gradient of log density).

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: SCORE MODEL
################################################################################

class ScoreModel:
    """
    Score-Based Generative Model
    =============================

    Learns the score function for generation.
    """

    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    def score(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """
        Compute score (gradient of log density).

        For Gaussian: s(x) = -x / σ²
        """
        return -x / (sigma ** 2)

    def langevin_step(self, x: np.ndarray, sigma: float, step_size: float) -> np.ndarray:
        """One step of Langevin dynamics."""
        score = self.score(x, sigma)
        noise = np.random.randn(*x.shape)
        return x + (step_size / 2) * score + np.sqrt(step_size) * noise


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_score_matching():
    """Demonstrate score matching."""
    print("=" * 70)
    print("SCORE MATCHING DEMONSTRATION")
    print("=" * 70)

    model = ScoreModel(input_dim=10)
    x = np.random.randn(5, 10)
    score = model.score(x, sigma=1.0)
    print(f"Input: {x.shape}")
    print(f"Score: {score.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_score_matching()


################################################################################
# REFERENCES
################################################################################

# [1] Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution.

################################################################################
