"""
################################################################################
PROCESS REWARD MODEL (PRM) — STEP-BY-STEP REASONING VERIFICATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Process Reward Model?
    A Process Reward Model (PRM) scores each STEP of a reasoning process,
    not just the final answer. This is crucial for training reasoning models
    because it provides dense feedback on the reasoning chain.

    Traditional Reward Model (ORM - Outcome Reward Model):
    - Only scores the FINAL answer
    - Problem: correct answer can come from wrong reasoning
    - Problem: wrong answer can come from right reasoning with one mistake

    Process Reward Model (PRM):
    - Scores EACH reasoning step
    - Provides dense feedback for learning
    - Can identify WHERE reasoning goes wrong
    - Enables "best-of-N" selection at each step

Why PRM matters:
    - DeepSeek-R1, OpenAI o1, and similar reasoning models use PRMs
    - PRMs enable test-time compute scaling (generate many paths, select best)
    - PRMs catch errors early (before they propagate)
    - PRMs provide better training signal than ORMs

Algorithm:
    1. Given a reasoning chain: [step1, step2, ..., stepN]
    2. Score each step: PRM(step_i | context, step_{i-1})
    3. Combine scores: overall = Σ score_i (or product)
    4. Use for: training (RL), inference (selection), evaluation

Interview Questions:
    Q: "What is a Process Reward Model?"
    A: A model that scores each step of a reasoning chain, not just
       the final answer. Provides dense feedback for training and
       enables step-level selection during inference.

    Q: "How does PRM differ from ORM?"
    A: ORM (Outcome Reward Model) only scores the final answer.
       PRM scores each reasoning step. PRM catches errors earlier
       and provides better training signal for reasoning tasks.

    Q: "How do you train a PRM?"
    A: Collect reasoning chains with step-level correctness labels.
       Train a classifier to predict if each step is correct.
       Use human annotation or automated verification (for math).

################################################################################
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

################################################################################
# SECTION 1: PROCESS REWARD MODEL
################################################################################

@dataclass
class PRMConfig:
    """Configuration for Process Reward Model."""
    d_model: int = 256
    n_layers: int = 4
    n_classes: int = 2  # correct/incorrect per step
    learning_rate: float = 1e-5


class ProcessRewardModel:
    """
    Process Reward Model (PRM) — Step-by-Step Reasoning Scorer

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Process Reward Model                      │
    │                                                              │
    │  Reasoning Chain: [step1, step2, ..., stepN]                │
    │       │                                                      │
    │       ▼                                                      │
    │  For each step i:                                            │
    │    - Encode step_i with context (previous steps)             │
    │    - Predict correctness probability                         │
    │    - Score: P(correct | context, step_i)                    │
    │       │                                                      │
    │       ▼                                                      │
    │  Step Scores: [s1, s2, ..., sN]                             │
    │  Overall Score: Σ s_i or Π s_i                              │
    └─────────────────────────────────────────────────────────────┘

    Interview Question:
        Q: "How does a PRM score reasoning steps?"
        A: For each step, the PRM encodes the step along with its
           context (previous steps) and predicts a correctness score.
           The score represents P(step is correct | context).
           This provides dense feedback for RL training.
    """

    def __init__(self, config: PRMConfig = None):
        self.config = config or PRMConfig()

        # Scoring network (simplified: linear layers)
        self.step_encoder = np.random.randn(512, self.config.d_model) * 0.02
        self.context_encoder = np.random.randn(self.config.d_model, self.config.d_model) * 0.02
        self.scorer = np.random.randn(self.config.d_model, self.config.n_classes) * 0.02

    def score_steps(
        self,
        steps: List[str],
        step_embeddings: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Score each step of a reasoning chain.

        Args:
            steps: List of reasoning step strings
            step_embeddings: Optional pre-computed embeddings [n_steps × d_model]

        Returns:
            Dictionary with step scores and overall score
        """
        n_steps = len(steps)

        # Generate embeddings if not provided
        if step_embeddings is None:
            step_embeddings = self._encode_steps(steps)

        # Score each step
        step_scores = []
        context = np.zeros(self.config.d_model)

        for i in range(n_steps):
            # Combine step embedding with context
            combined = step_embeddings[i] + context @ self.context_encoder

            # Predict correctness
            logits = combined @ self.scorer
            probs = self._softmax(logits)
            score = probs[1]  # P(correct)

            step_scores.append(float(score))

            # Update context (running average of correct steps)
            if score > 0.5:
                context = 0.8 * context + 0.2 * step_embeddings[i]

        # Overall score
        overall_sum = sum(step_scores)
        overall_product = 1.0
        for s in step_scores:
            overall_product *= s

        return {
            "step_scores": step_scores,
            "overall_sum": overall_sum,
            "overall_product": overall_product,
            "min_score": min(step_scores),
            "n_correct": sum(1 for s in step_scores if s > 0.5),
            "first_error_idx": next((i for i, s in enumerate(step_scores) if s <= 0.5), None),
        }

    def select_best_chain(
        self,
        chains: List[List[str]],
    ) -> Tuple[int, Dict]:
        """
        Select the best reasoning chain from multiple candidates.

        This is the key use case for PRMs in test-time compute scaling:
        generate many reasoning chains, score each with PRM, select best.

        Args:
            chains: List of reasoning chains (each is a list of steps)

        Returns:
            best_idx: Index of the best chain
            scores: Scores for all chains
        """
        chain_scores = []
        for chain in chains:
            result = self.score_steps(chain)
            chain_scores.append(result["overall_product"])

        best_idx = int(np.argmax(chain_scores))
        return best_idx, {"chain_scores": chain_scores, "best_idx": best_idx}

    def _encode_steps(self, steps: List[str]) -> np.ndarray:
        """Encode reasoning steps to embeddings."""
        embeddings = []
        for step in steps:
            # Simple hash-based embedding (in production: use transformer)
            hash_val = hash(step) % (2**31)
            np.random.seed(hash_val)
            emb = np.random.randn(self.config.d_model) * 0.1
            embeddings.append(emb)
        np.random.seed(None)
        return np.array(embeddings)

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)


################################################################################
# SECTION 2: AUTOMATED PRM TRAINING DATA
################################################################################

class PRMDataGenerator:
    """
    Generate training data for Process Reward Models.

    For math tasks, we can automatically verify each step:
    - Check if the arithmetic is correct
    - Check if the logic follows
    - Check if the final answer is correct

    For other tasks, we need human annotation or LLM-as-judge.
    """

    @staticmethod
    def generate_math_chain(
        problem: str,
        correct: bool = True,
        error_step: Optional[int] = None,
    ) -> Dict:
        """
        Generate a reasoning chain for a math problem.

        Args:
            problem: The math problem
            correct: Whether the chain should be correct
            error_step: Which step to introduce an error (if not correct)

        Returns:
            Dictionary with steps and labels
        """
        steps = []
        labels = []

        # Generate steps
        n_steps = np.random.randint(3, 7)
        for i in range(n_steps):
            if not correct and error_step is not None and i == error_step:
                steps.append(f"Step {i+1}: [ERROR] Incorrect calculation")
                labels.append(0)  # Incorrect
            else:
                steps.append(f"Step {i+1}: Correct reasoning step")
                labels.append(1)  # Correct

        return {
            "problem": problem,
            "steps": steps,
            "labels": labels,
            "overall_correct": all(labels),
        }


################################################################################
# SECTION 3: TESTING AND DEMONSTRATION
################################################################################

def demonstrate_prm():
    """Comprehensive PRM demonstration."""
    print("=" * 70)
    print("PROCESS REWARD MODEL DEMONSTRATION")
    print("=" * 70)

    prm = ProcessRewardModel()

    # === Demo 1: Score a correct chain ===
    print("\n--- Demo 1: Correct Reasoning Chain ---")
    correct_chain = [
        "Step 1: Identify the problem type — this is a linear equation",
        "Step 2: Isolate the variable: 2x + 3 = 7 → 2x = 4",
        "Step 3: Solve: x = 4/2 = 2",
        "Step 4: Verify: 2(2) + 3 = 7 ✓",
    ]
    result = prm.score_steps(correct_chain)
    print(f"Step scores: {[f'{s:.3f}' for s in result['step_scores']]}")
    print(f"Overall (product): {result['overall_product']:.3f}")
    print(f"First error: {result['first_error_idx']}")

    # === Demo 2: Score an incorrect chain ===
    print("\n--- Demo 2: Incorrect Reasoning Chain ---")
    incorrect_chain = [
        "Step 1: Identify the problem type — this is a linear equation",
        "Step 2: Isolate the variable: 2x + 3 = 7 → 2x = 10",  # Error!
        "Step 3: Solve: x = 10/2 = 5",
        "Step 4: Verify: 2(5) + 3 = 13 ≠ 7",
    ]
    result = prm.score_steps(incorrect_chain)
    print(f"Step scores: {[f'{s:.3f}' for s in result['step_scores']]}")
    print(f"Overall (product): {result['overall_product']:.3f}")
    print(f"First error at step: {result['first_error_idx']}")

    # === Demo 3: Select best chain ===
    print("\n--- Demo 3: Best-of-N Selection ---")
    chains = [correct_chain, incorrect_chain, correct_chain[:2] + ["Step 3: Random guess"]]
    best_idx, scores = prm.select_best_chain(chains)
    print(f"Chain scores: {[f'{s:.3f}' for s in scores['chain_scores']]}")
    print(f"Best chain: {best_idx}")

    # === Demo 4: PRM vs ORM comparison ===
    print("\n--- Demo 4: PRM vs ORM ---")
    print(f"{'Aspect':<25} {'ORM':<20} {'PRM':<20}")
    print("-" * 65)
    print(f"{'Scores':<25} {'Final answer only':<20} {'Each step':<20}")
    print(f"{'Feedback density':<25} {'Sparse':<20} {'Dense':<20}")
    print(f"{'Error detection':<25} {'After the fact':<20} {'Real-time':<20}")
    print(f"{'Training signal':<25} {'Weak':<20} {'Strong':<20}")
    print(f"{'Use case':<25} {'Simple tasks':<20} {'Reasoning tasks':<20}")

    print("\n" + "=" * 70)
    print("All PRM demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_prm()


################################################################################
# REFERENCES
################################################################################

# [1] Lightman, H., et al. (2023). Let's Verify Step by Step.
#     arXiv:2305.20050. (Process Reward Models)
#
# [2] Wang, P., et al. (2024). Math-Shepherd: Verify and Reinforce LLMs
#     Step-by-step without Human Annotations. arXiv:2312.08935.
#
# [3] Luo, L., et al. (2024). Improve Mathematical Reasoning in Language
#     Models by Automated Process Supervision. arXiv:2406.06592.

################################################################################
