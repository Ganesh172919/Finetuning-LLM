"""
################################################################################
REWARD MODEL — LEARNING HUMAN PREFERENCES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Reward Model?
    A model that predicts how good a response is based on human preferences.
    Used in RLHF to guide language model training.

Training:
    Given (chosen, rejected) pairs:
    Loss = -log σ(reward(chosen) - reward(rejected))

Interview Questions:
    1. "How do you train a reward model?"
        Collect human preferences, train model to predict rewards
        such that preferred responses get higher rewards.

    2. "What are the challenges of reward modeling?"
        Reward hacking, distribution shift, annotation quality.

################################################################################
"""

import numpy as np
from typing import List, Tuple

################################################################################
# SECTION 1: REWARD MODEL
################################################################################

class RewardModel:
    """
    Reward Model
    ============

    Predicts reward scores for responses.

    Architecture:
    Response → Encoder → Reward Head → Score

    Interview Question:
        "What does a reward model learn?"
        It learns to predict human preferences.
        Preferred responses get higher scores.
    """

    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        self.reward_head = np.random.randn(d_model, 1) * 0.02

    def compute_reward(self, features: np.ndarray) -> np.ndarray:
        """
        Compute reward for features.

        Args:
            features: [batch × d_model]

        Returns:
            reward: [batch × 1]
        """
        return np.matmul(features, self.reward_head)

    def training_loss(
        self,
        chosen_features: np.ndarray,
        rejected_features: np.ndarray
    ) -> float:
        """
        Compute training loss.

        Loss = -log σ(r_chosen - r_rejected)
        """
        r_chosen = self.compute_reward(chosen_features)
        r_rejected = self.compute_reward(rejected_features)

        logits = r_chosen - r_rejected
        loss = -np.mean(np.log(1 / (1 + np.exp(-logits)) + 1e-8))

        return loss

    def rank_responses(self, features_list: List[np.ndarray]) -> List[int]:
        """
        Rank multiple responses by reward.

        Args:
            features_list: List of feature arrays

        Returns:
            Indices sorted by reward (highest first)
        """
        rewards = [float(self.compute_reward(f.reshape(1, -1))) for f in features_list]
        return list(np.argsort(rewards)[::-1])


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_reward_model():
    """Demonstrate reward model."""
    print("=" * 70)
    print("REWARD MODEL DEMONSTRATION")
    print("=" * 70)

    rm = RewardModel(d_model=64)

    # Compute rewards
    chosen = np.random.randn(4, 64)
    rejected = np.random.randn(4, 64)

    loss = rm.training_loss(chosen, rejected)
    print(f"Training loss: {loss:.4f}")

    # Rank responses
    responses = [np.random.randn(64) for _ in range(5)]
    ranking = rm.rank_responses(responses)
    print(f"Ranking: {ranking}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_reward_model()


################################################################################
# REFERENCES
################################################################################

# [1] Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback.

################################################################################
