"""
################################################################################
RLHF — REINFORCEMENT LEARNING FROM HUMAN FEEDBACK
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is RLHF?
    RLHF aligns language models with human preferences using
    reinforcement learning. It's how ChatGPT, Claude, and other
    aligned models are trained.

Training Process:
    1. Supervised Fine-Tuning (SFT): Train on human demonstrations
    2. Reward Model Training: Train model to predict human preferences
    3. RL Optimization: Optimize policy to maximize reward

Why does it matter?
    Pre-trained LLMs just predict next tokens.
    RLHF makes them:
    - Helpful: Answer questions well
    - Harmless: Avoid dangerous content
    - Honest: Don't hallucinate

Historical Evolution:
    - 2022: InstructGPT (OpenAI)
    - 2023: ChatGPT, Claude
    - 2024: DPO (simpler alternative)
    - 2025: GRPO (DeepSeek)

Interview Questions:
    1. "What is RLHF?"
       Training LLMs with human feedback using RL.
       Humans rank outputs, we train a reward model,
       then optimize the LLM to maximize reward.

    2. "What are the steps of RLHF?"
       1. SFT on demonstrations
       2. Train reward model on preferences
       3. Optimize with PPO

    3. "What's the difference between RLHF and DPO?"
       RLHF: explicit reward model + PPO
       DPO: direct optimization from preferences

################################################################################
"""

import numpy as np
from typing import List, Dict, Tuple

################################################################################
# SECTION 1: REWARD MODEL
################################################################################

class RewardModel:
    """
    Reward Model
    ============

    Predicts how good a response is based on human preferences.

    Training:
    Given (prompt, chosen, rejected) pairs:
    Loss = -log σ(reward(chosen) - reward(rejected))

    This teaches the model that chosen responses are better.

    Interview Question:
        "How do you train a reward model?"
        Collect human preferences (A is better than B).
        Train model to predict reward such that
        reward(A) > reward(B) for preferred responses.
    """

    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        # Simplified reward head
        self.reward_head = np.random.randn(d_model, 1) * 0.02

    def compute_reward(self, features: np.ndarray) -> np.ndarray:
        """
        Compute reward for a response.

        Args:
            features: Response features [batch × d_model]

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
        Compute reward model training loss.

        Loss = -log σ(r_chosen - r_rejected)
        """
        r_chosen = self.compute_reward(chosen_features)
        r_rejected = self.compute_reward(rejected_features)

        # Bradley-Terry model
        logits = r_chosen - r_rejected
        loss = -np.mean(np.log(1 / (1 + np.exp(-logits)) + 1e-8))

        return loss


################################################################################
# SECTION 2: PPO TRAINER
################################################################################

class PPOTrainer:
    """
    PPO: Proximal Policy Optimization
    ===================================

    Optimizes the policy (LLM) to maximize reward while
    staying close to the reference policy.

    Loss = reward - β × KL(policy || reference)

    The KL penalty prevents the model from "hacking" the reward.

    Interview Question:
        "What is PPO and why use it for RLHF?"
        PPO constrains policy updates to prevent large changes.
        This stabilizes training and prevents reward hacking.
    """

    def __init__(self, beta: float = 0.1, clip_range: float = 0.2):
        self.beta = beta
        self.clip_range = clip_range

    def compute_loss(
        self,
        logprobs: np.ndarray,
        old_logprobs: np.ndarray,
        rewards: np.ndarray,
        kl_penalty: np.ndarray
    ) -> float:
        """
        Compute PPO loss.

        Args:
            logprobs: Current policy log probs
            old_logprobs: Old policy log probs
            rewards: Reward scores
            kl_penalty: KL divergence penalty

        Returns:
            PPO loss
        """
        # Probability ratio
        ratio = np.exp(logprobs - old_logprobs)

        # Clipped surrogate loss
        clipped_ratio = np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
        surrogate1 = ratio * rewards
        surrogate2 = clipped_ratio * rewards
        surrogate_loss = np.minimum(surrogate1, surrogate2)

        # Total loss
        loss = -np.mean(surrogate_loss - self.beta * kl_penalty)

        return loss


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_rlhf():
    """Demonstrate RLHF concepts."""
    print("=" * 70)
    print("RLHF DEMONSTRATION")
    print("=" * 70)

    # Reward Model
    print("\n--- Reward Model ---")
    rm = RewardModel(d_model=64)
    chosen_features = np.random.randn(4, 64)
    rejected_features = np.random.randn(4, 64)
    rm_loss = rm.training_loss(chosen_features, rejected_features)
    print(f"Reward model loss: {rm_loss:.4f}")

    # PPO
    print("\n--- PPO ---")
    ppo = PPOTrainer(beta=0.1, clip_range=0.2)
    logprobs = np.random.randn(4)
    old_logprobs = np.random.randn(4)
    rewards = np.array([1.0, 0.5, -0.5, -1.0])
    kl_penalty = np.abs(np.random.randn(4)) * 0.1
    ppo_loss = ppo.compute_loss(logprobs, old_logprobs, rewards, kl_penalty)
    print(f"PPO loss: {ppo_loss:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_rlhf()


################################################################################
# REFERENCES
################################################################################

# [1] Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback.

################################################################################
