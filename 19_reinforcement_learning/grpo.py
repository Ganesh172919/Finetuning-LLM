"""
################################################################################
GRPO — GROUP RELATIVE POLICY OPTIMIZATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is GRPO?
    GRPO is DeepSeek's method for training reasoning models.
    Instead of comparing chosen vs rejected (DPO), it generates
    multiple responses and ranks them.

Algorithm:
    1. Generate N responses for each prompt
    2. Score each response (e.g., correctness)
    3. Rank responses
    4. Optimize to prefer higher-ranked responses

Why GRPO?
    - No need for explicit preference data
    - Can use automatic rewards (e.g., math correctness)
    - Works well for reasoning tasks

Interview Questions:
        Q: "What is GRPO?"
        A: Group Relative Policy Optimization. Generate multiple responses,
           rank them, optimize to prefer higher-ranked ones.

################################################################################
"""

import numpy as np
from typing import List

################################################################################
# SECTION 1: GRPO TRAINER
################################################################################

class GRPOTrainer:
    """
    GRPO Trainer
    ============

    Trains models using group-relative rewards.

    Interview Questions:
        Q: "How does GRPO differ from DPO?"
        A: DPO uses explicit preference pairs.
           GRPO generates multiple responses and ranks them.
           GRPO can use automatic rewards.
    """

    def __init__(self, beta: float = 0.1, n_samples: int = 8):
        self.beta = beta
        self.n_samples = n_samples

    def compute_loss(
        self,
        logprobs: np.ndarray,
        rewards: np.ndarray
    ) -> float:
        """
        Compute GRPO loss.

        Args:
            logprobs: [batch × n_samples] log probabilities
            rewards: [batch × n_samples] reward scores

        Returns:
            loss: GRPO loss
        """
        # Normalize rewards within group
        reward_mean = np.mean(rewards, axis=1, keepdims=True)
        reward_std = np.std(rewards, axis=1, keepdims=True) + 1e-8
        advantages = (rewards - reward_mean) / reward_std

        # Policy gradient loss
        loss = -np.mean(logprobs * advantages)

        return loss


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_grpo():
    """Demonstrate GRPO."""
    print("=" * 70)
    print("GRPO DEMONSTRATION")
    print("=" * 70)

    trainer = GRPOTrainer(beta=0.1, n_samples=8)

    logprobs = np.random.randn(4, 8)
    rewards = np.random.randn(4, 8)

    loss = trainer.compute_loss(logprobs, rewards)
    print(f"GRPO loss: {loss:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_grpo()


################################################################################
# REFERENCES
################################################################################

# [1] DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs.

################################################################################
