"""
################################################################################
DPO — DIRECT PREFERENCE OPTIMIZATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is DPO?
    DPO is a method to align language models with human preferences
    WITHOUT training a separate reward model. It directly optimizes
    the policy using preference data.

Why does it matter?
    Traditional RLHF requires:
    1. Collect human preferences
    2. Train reward model
    3. Optimize policy with PPO

    DPO simplifies to:
    1. Collect human preferences
    2. Directly optimize policy

    Benefits:
    - Simpler (no reward model)
    - More stable (no RL)
    - Easier to implement
    - Often better results

How does it work?
    Given preference pairs (chosen, rejected):
    - Chosen: response humans prefer
    - Rejected: response humans don't prefer

    DPO maximizes:
    L = log σ(β × (log π(chosen) - log π(rejected)))

    Where:
    - π is the policy (language model)
    - β controls deviation from reference
    - σ is the sigmoid function

Interview Questions:
    1. "What is DPO?"
       Direct Preference Optimization: aligns models with preferences
       without a reward model. Uses preference pairs directly.

    2. "How is DPO different from RLHF?"
       RLHF: train reward model → optimize with PPO
       DPO: directly optimize with preference data
       DPO is simpler and often more stable.

    3. "When should I use DPO vs RLHF?"
       DPO: simpler, when you have preference data
       RLHF: when you need iterative refinement or online learning

################################################################################
"""

import numpy as np
from typing import List, Dict, Tuple
import math

################################################################################
# SECTION 1: DPO TRAINER
################################################################################

class DPOTrainer:
    """
    DPO Trainer
    ===========

    Trains a language model using preference data.

    Loss Function:
    L_DPO = -E[log σ(β × (log π_θ(y_w|x) - log π_θ(y_l|x)
                         - log π_ref(y_w|x) + log π_ref(y_l|x)))]

    Where:
    - y_w: chosen (winning) response
    - y_l: rejected (losing) response
    - π_θ: policy being trained
    - π_ref: reference policy (frozen)
    - β: KL penalty coefficient

    Interview Question:
        "Explain the DPO loss function."
        DPO maximizes the difference in log-probabilities between
        chosen and rejected responses, relative to a reference model.
        The β parameter controls how much we deviate from the reference.
    """

    def __init__(
        self,
        beta: float = 0.1,
        learning_rate: float = 1e-5
    ):
        self.beta = beta
        self.learning_rate = learning_rate

    def compute_loss(
        self,
        policy_chosen_logps: np.ndarray,
        policy_rejected_logps: np.ndarray,
        reference_chosen_logps: np.ndarray,
        reference_rejected_logps: np.ndarray
    ) -> float:
        """
        Compute DPO loss.

        Args:
            policy_chosen_logps: Log probs of policy for chosen responses
            policy_rejected_logps: Log probs of policy for rejected responses
            reference_chosen_logps: Log probs of reference for chosen
            reference_rejected_logps: Log probs of reference for rejected

        Returns:
            loss: DPO loss (scalar)
        """
        # Compute log ratios
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # Compute logits
        logits = self.beta * (chosen_logratios - rejected_logratios)

        # DPO loss: -log σ(logits)
        loss = -np.mean(np.log(1 / (1 + np.exp(-logits)) + 1e-8))

        return loss

    def compute_logprobs(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute log probabilities of labels given logits.

        Args:
            logits: [batch × seq × vocab]
            labels: [batch × seq]

        Returns:
            logprobs: [batch] sum of log probs
        """
        vocab_size = logits.shape[-1]

        # Flatten
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # Log softmax
        shifted = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
        log_probs = shifted[np.arange(len(labels_flat)), labels_flat] - log_sum_exp

        # Reshape and sum over sequence
        log_probs = log_probs.reshape(logits.shape[0], -1)
        return np.sum(log_probs, axis=-1)


################################################################################
# SECTION 2: PREFERENCE DATA
################################################################################

class PreferenceDataset:
    """
    Preference Dataset
    ===================

    Stores (prompt, chosen, rejected) triplets for DPO training.

    Data format:
    {
        "prompt": "What is machine learning?",
        "chosen": "Machine learning is a subset of AI...",
        "rejected": "I don't know what that is."
    }
    """

    def __init__(self):
        self.data: List[Dict[str, str]] = []

    def add(self, prompt: str, chosen: str, rejected: str):
        """Add a preference pair."""
        self.data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })

    def sample(self, batch_size: int) -> List[Dict[str, str]]:
        """Sample a batch of preference pairs."""
        indices = np.random.choice(len(self.data), size=min(batch_size, len(self.data)))
        return [self.data[i] for i in indices]


################################################################################
# SECTION 3: GRPO (GROUP RELATIVE POLICY OPTIMIZATION)
################################################################################

class GRPOTrainer:
    """
    GRPO: Group Relative Policy Optimization
    ==========================================

    Used by DeepSeek R1 for reasoning training.

    Key idea: Instead of comparing chosen vs rejected,
    compare multiple responses and rank them.

    Algorithm:
    1. Generate N responses for each prompt
    2. Score each response (e.g., correctness)
    3. Rank responses
    4. Optimize to prefer higher-ranked responses

    Benefits:
    - No need for explicit preference data
    - Can use automatic rewards (e.g., math correctness)
    - Works well for reasoning tasks

    Interview Question:
        "What is GRPO?"
        Group Relative Policy Optimization generates multiple responses,
        ranks them, and optimizes to prefer higher-ranked ones.
        Used by DeepSeek R1 for reasoning training.
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
            logprobs: [batch × n_samples] log probs
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
# SECTION 4: TESTING & EXAMPLES
################################################################################

def demonstrate_dpo():
    """Demonstrate DPO concepts."""
    print("=" * 70)
    print("DPO DEMONSTRATION")
    print("=" * 70)

    # DPO Trainer
    print("\n--- DPO Trainer ---")
    trainer = DPOTrainer(beta=0.1)

    # Preference data
    print("\n--- Preference Data ---")
    dataset = PreferenceDataset()
    dataset.add(
        prompt="What is AI?",
        chosen="AI is artificial intelligence...",
        rejected="I don't know."
    )
    print(f"Dataset size: {len(dataset.data)}")

    # DPO loss computation
    print("\n--- DPO Loss ---")
    policy_chosen = np.array([-1.0, -1.5, -2.0])
    policy_rejected = np.array([-3.0, -2.5, -4.0])
    ref_chosen = np.array([-1.2, -1.3, -1.8])
    ref_rejected = np.array([-2.8, -2.3, -3.5])

    loss = trainer.compute_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    print(f"DPO loss: {loss:.4f}")

    # GRPO
    print("\n--- GRPO ---")
    grpo = GRPOTrainer(beta=0.1, n_samples=4)
    logprobs = np.array([[-1.0, -1.5, -2.0, -3.0]])
    rewards = np.array([[0.9, 0.7, 0.5, 0.2]])
    grpo_loss = grpo.compute_loss(logprobs, rewards)
    print(f"GRPO loss: {grpo_loss:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_dpo()


################################################################################
# REFERENCES
################################################################################

# [1] Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model.
# [2] DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL.

################################################################################
