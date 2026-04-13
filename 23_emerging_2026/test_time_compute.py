"""
################################################################################
TEST-TIME COMPUTE — REASONING AT INFERENCE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Test-Time Compute?
    Spending more computation during inference to improve answers.
    Instead of generating one answer, generate multiple and select the best.

Why does it matter?
    More compute = better answers:
    - Chain of thought: think step by step
    - Self-consistency: multiple attempts
    - Tree of thoughts: explore options

Interview Questions:
    1. "What is test-time compute?"
        Using more computation during inference to improve quality.

    2. "How does it differ from training compute?"
        Training: improve model weights
        Test-time: improve individual predictions

################################################################################
"""

import numpy as np
from typing import List

################################################################################
# SECTION 1: TEST-TIME COMPUTE
################################################################################

class TestTimeCompute:
    """
    Test-Time Compute Scaling
    ==========================

    Improves answers by spending more computation at inference.

    Methods:
    1. Chain of Thought: think step by step
    2. Self-consistency: multiple attempts, vote
    3. Tree of Thoughts: explore multiple paths
    4. Verification: check and correct

    Interview Question:
        "How do you scale test-time compute?"
        Generate multiple reasoning paths, evaluate them,
        and select the best answer. More paths = better answers.
    """

    def __init__(self, n_samples: int = 5):
        self.n_samples = n_samples

    def generate_and_select(
        self,
        generate_fn,
        evaluate_fn,
        prompt: str
    ) -> str:
        """
        Generate multiple candidates and select best.

        Args:
            generate_fn: Function to generate candidate
            evaluate_fn: Function to evaluate candidate
            prompt: Input prompt

        Returns:
            Best candidate
        """
        candidates = []
        scores = []

        for _ in range(self.n_samples):
            candidate = generate_fn(prompt)
            score = evaluate_fn(candidate)
            candidates.append(candidate)
            scores.append(score)

        # Select best
        best_idx = np.argmax(scores)
        return candidates[best_idx]


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_test_time_compute():
    """Demonstrate test-time compute."""
    print("=" * 70)
    print("TEST-TIME COMPUTE DEMONSTRATION")
    print("=" * 70)

    ttc = TestTimeCompute(n_samples=5)

    # Simulate generation and evaluation
    def generate(prompt):
        return f"Answer to: {prompt[:20]}..."

    def evaluate(candidate):
        return np.random.random()

    result = ttc.generate_and_select(generate, evaluate, "What is 2+2?")
    print(f"Selected: {result}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_test_time_compute()
