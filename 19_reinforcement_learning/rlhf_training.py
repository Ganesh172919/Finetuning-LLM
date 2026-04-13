"""
################################################################################
RLHF TRAINING — REINFORCEMENT LEARNING FROM HUMAN FEEDBACK
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is RLHF?
    RLHF aligns language models with human preferences using
    reinforcement learning. It's how ChatGPT, Claude, and other
    aligned models are trained.

Training Pipeline:
    Phase 1: Supervised Fine-Tuning (SFT)
        Train on human demonstrations
        Model learns to follow instructions

    Phase 2: Reward Model Training
        Collect human preferences (A is better than B)
        Train model to predict rewards

    Phase 3: RL Optimization (PPO)
        Generate responses
        Score with reward model
        Optimize to maximize reward while staying close to SFT model

Why RLHF?
    Pre-trained LLMs just predict next tokens.
    RLHF makes them:
    - Helpful: Answer questions well
    - Harmless: Avoid dangerous content
    - Honest: Don't hallucinate

Historical Impact:
    - 2022: InstructGPT (OpenAI) — RLHF
    - 2023: ChatGPT — RLHF at scale
    - 2024: DPO — Simpler alternative
    - 2025: GRPO — DeepSeek's approach

Interview Questions:
        Q: "Walk me through the RLHF pipeline."
        A: 1) SFT on demonstrations
           2) Train reward model on preferences
           3) Optimize with PPO using reward model
           The reward model acts as a proxy for human preferences.

        Q: "What are the challenges of RLHF?"
        A: Reward hacking (model finds shortcuts),
           distribution shift, annotation quality,
           training instability.

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

    Architecture:
        Response → Transformer Encoder → Linear → Reward Score

    Interview Questions:
        Q: "How do you collect preference data?"
        A: Show humans two responses to the same prompt.
           Ask which is better. Collect (chosen, rejected) pairs.

        Q: "What makes a good reward model?"
        A: Accurately captures human preferences, generalizes
           to new prompts, robust to adversarial inputs.
    """

    def __init__(self, d_model: int = 256):
        self.d_model = d_model

        # Reward head
        self.reward_head = np.random.randn(d_model, 1) * 0.02
        self.reward_bias = 0.0

    def compute_reward(self, features: np.ndarray) -> np.ndarray:
        """
        Compute reward for response features.

        Args:
            features: [batch × d_model]

        Returns:
            reward: [batch × 1]
        """
        return np.matmul(features, self.reward_head) + self.reward_bias

    def training_loss(
        self,
        chosen_features: np.ndarray,
        rejected_features: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute reward model training loss.

        Args:
            chosen_features: Features of chosen responses
            rejected_features: Features of rejected responses

        Returns:
            loss: Training loss
            accuracy: How often chosen > rejected
        """
        r_chosen = self.compute_reward(chosen_features)
        r_rejected = self.compute_reward(rejected_features)

        # Bradley-Terry model
        logits = r_chosen - r_rejected
        loss = -np.mean(np.log(1 / (1 + np.exp(-logits)) + 1e-8))

        accuracy = np.mean(logits > 0)

        return float(loss), float(accuracy)


################################################################################
# SECTION 2: PPO TRAINER
################################################################################

class PPOTrainer:
    """
    PPO: Proximal Policy Optimization
    ===================================

    Optimizes the policy (LLM) to maximize reward while
    staying close to the reference policy.

    Objective:
        L = E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)]

    Where:
        r_t = π(a_t|s_t) / π_old(a_t|s_t) (probability ratio)
        A_t = advantage estimate
        ε = clip range (typically 0.2)

    KL Penalty:
        Total reward = reward_model_score - β × KL(policy || reference)

    Interview Questions:
        Q: "Why PPO instead of other RL algorithms?"
        A: PPO is stable (clips updates), sample efficient,
           and works well for LLMs. Simple to implement.

        Q: "How do you prevent reward hacking?"
        A: KL penalty prevents the policy from deviating too far
           from the reference model. Also, reward model ensemble.
    """

    def __init__(
        self,
        clip_range: float = 0.2,
        kl_coeff: float = 0.1,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01
    ):
        self.clip_range = clip_range
        self.kl_coeff = kl_coeff
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

    def compute_ppo_loss(
        self,
        logprobs: np.ndarray,
        old_logprobs: np.ndarray,
        advantages: np.ndarray,
        values: np.ndarray,
        returns: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Compute PPO loss.

        Args:
            logprobs: Current policy log probs [batch]
            old_logprobs: Old policy log probs [batch]
            advantages: Advantage estimates [batch]
            values: Value predictions [batch]
            returns: Discounted returns [batch]

        Returns:
            loss: Combined PPO loss
            metrics: Training metrics
        """
        # Probability ratio
        ratio = np.exp(logprobs - old_logprobs)

        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -np.mean(np.minimum(surr1, surr2))

        # Value loss
        value_loss = np.mean((values - returns) ** 2)

        # Entropy bonus (encourages exploration)
        entropy = -np.mean(logprobs)

        # Combined loss
        loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

        metrics = {
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy': float(entropy),
            'ratio_mean': float(np.mean(ratio)),
            'ratio_max': float(np.max(ratio)),
        }

        return loss, metrics

    def compute_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> np.ndarray:
        """
        Compute GAE (Generalized Advantage Estimation).

        A_t = Σ_{l=0}^{T-t} (γλ)^l × δ_{t+l}

        Where δ_t = r_t + γ × V(s_{t+1}) - V(s_t)

        Interview Questions:
            Q: "What is GAE?"
            A: Generalized Advantage Estimation. Balances bias vs variance
               in advantage estimates. λ=0: low variance, high bias.
               λ=1: high variance, low bias.
        """
        T = len(rewards)
        advantages = np.zeros(T)
        gae = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        return advantages


################################################################################
# SECTION 3: KL DIVERGENCE
################################################################################

def compute_kl_divergence(
    policy_logprobs: np.ndarray,
    reference_logprobs: np.ndarray
) -> np.ndarray:
    """
    Compute KL divergence between policy and reference.

    KL(π || π_ref) = E_π[log π(a) - log π_ref(a)]

    This is used to prevent the policy from deviating too far
    from the reference model.

    Interview Questions:
        Q: "Why use KL divergence in RLHF?"
        A: Prevents reward hacking. Without KL penalty, the model
           might find adversarial outputs that score high on the
           reward model but are actually bad.
    """
    return policy_logprobs - reference_logprobs


################################################################################
# SECTION 4: COMPLETE RLHF PIPELINE
################################################################################

class RLHFPipeline:
    """
    Complete RLHF Training Pipeline
    ================================

    Combines all components for RLHF training.

    Steps:
    1. SFT: Train on demonstrations
    2. Reward Model: Train on preferences
    3. PPO: Optimize policy

    Interview Questions:
        Q: "How long does RLHF training take?"
        A: Usually 1-3 days on 8-64 GPUs.
           SFT: few hours to 1 day
           Reward model: few hours
           PPO: 1-2 days

        Q: "What are alternatives to RLHF?"
        A: DPO (simpler), ORPO (one-step), GRPO (group-based),
           RLAIF (AI feedback instead of human).
    """

    def __init__(self, d_model: int = 256):
        self.reward_model = RewardModel(d_model)
        self.ppo_trainer = PPOTrainer()

    def train_reward_model(
        self,
        chosen_features: np.ndarray,
        rejected_features: np.ndarray,
        n_steps: int = 100
    ) -> Dict:
        """
        Train reward model on preference data.

        Args:
            chosen_features: Features of chosen responses
            rejected_features: Features of rejected responses
            n_steps: Training steps

        Returns:
            metrics: Training metrics
        """
        losses = []
        accuracies = []

        for step in range(n_steps):
            loss, accuracy = self.reward_model.training_loss(
                chosen_features, rejected_features
            )
            losses.append(loss)
            accuracies.append(accuracy)

        return {
            'final_loss': losses[-1],
            'final_accuracy': accuracies[-1],
            'losses': losses,
        }

    def compute_rewards(
        self,
        response_features: np.ndarray,
        reference_logprobs: np.ndarray,
        policy_logprobs: np.ndarray
    ) -> np.ndarray:
        """
        Compute rewards for PPO training.

        Total reward = reward_model_score - β × KL(policy || reference)
        """
        # Reward model score
        rm_score = self.reward_model.compute_reward(response_features).squeeze()

        # KL penalty
        kl = compute_kl_divergence(policy_logprobs, reference_logprobs)
        kl_penalty = self.ppo_trainer.kl_coeff * kl

        return rm_score - kl_penalty


################################################################################
# SECTION 5: TESTING
################################################################################

def demonstrate_rlhf():
    """Demonstrate RLHF concepts."""
    print("=" * 70)
    print("RLHF TRAINING DEMONSTRATION")
    print("=" * 70)

    # Reward Model
    print("\n--- Reward Model ---")
    rm = RewardModel(d_model=64)
    chosen = np.random.randn(4, 64)
    rejected = np.random.randn(4, 64)
    loss, acc = rm.training_loss(chosen, rejected)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.2%}")

    # PPO
    print("\n--- PPO ---")
    ppo = PPOTrainer()
    logprobs = np.random.randn(8)
    old_logprobs = np.random.randn(8)
    advantages = np.random.randn(8)
    values = np.random.randn(8)
    returns = np.random.randn(8)

    loss, metrics = ppo.compute_ppo_loss(logprobs, old_logprobs, advantages, values, returns)
    print(f"PPO loss: {loss:.4f}")
    print(f"Policy loss: {metrics['policy_loss']:.4f}")
    print(f"Value loss: {metrics['value_loss']:.4f}")

    # GAE
    print("\n--- GAE ---")
    rewards = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
    values = np.array([0.8, 0.6, 0.4, 0.3, 0.2])
    advantages = ppo.compute_advantages(rewards, values)
    print(f"Rewards: {rewards}")
    print(f"Advantages: {advantages.round(3)}")

    # Complete pipeline
    print("\n--- RLHF Pipeline ---")
    pipeline = RLHFPipeline(d_model=64)
    metrics = pipeline.train_reward_model(chosen, rejected, n_steps=10)
    print(f"Final loss: {metrics['final_loss']:.4f}")
    print(f"Final accuracy: {metrics['final_accuracy']:.2%}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_rlhf()


################################################################################
# REFERENCES
################################################################################

# [1] Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback.
# [2] Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.

################################################################################
