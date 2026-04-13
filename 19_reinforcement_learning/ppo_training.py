"""
PPO (Proximal Policy Optimization) for LLM Alignment
======================================================

The workhorse algorithm for RLHF (Reinforcement Learning from Human Feedback).
PPO constrains policy updates to prevent catastrophic divergence while
maximizing the reward signal from a learned reward model.

BEGINNER LEVEL:
    Imagine training a dog. You reward good behavior and ignore bad behavior.
    But you don't want to change the dog's entire personality overnight —
    you want small, gradual improvements. PPO is like a "training leash"
    that prevents the model from changing too much at once.

INTERMEDIATE LEVEL:
    PPO is a policy gradient method that clips the objective function to
    constrain how far the new policy can deviate from the old policy.
    This prevents destructively large updates while still allowing
    meaningful learning. For LLMs, PPO maximizes reward model scores
    while staying close to the SFT (supervised fine-tuned) model.

ADVANCED LEVEL:
    PPO optimizes: L = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]
    where r(θ) = π_new/π_old is the probability ratio and A is the
    advantage estimate. The clipping prevents r from deviating too far
    from 1, ensuring stable training. For LLMs, we add a KL penalty
    to prevent reward hacking.

EXPERT LEVEL:
    In RLHF, PPO has 4 models: policy (trained), reference (frozen SFT),
    reward model (frozen), and value/critic model (trained). The value
    model estimates expected returns for advantage computation (GAE).
    Key hyperparameters: clip range ε (0.2), KL coefficient (0.1-0.2),
    value loss coefficient (0.5), entropy bonus (0.01).

INTERVIEW LEVEL:
    Q: Why PPO for RLHF instead of REINFORCE or A2C?
    A: PPO's clipping prevents catastrophic policy collapse — critical
    for LLMs where a single bad update can destroy language quality.
    REINFORCE has high variance; A2C can still make large updates.
    PPO gives stable, monotonic improvement with reasonable compute.
    Q: What prevents reward hacking?
    A: KL penalty against the reference model ensures the policy
    doesn't diverge too far from human-like responses, even if the
    reward model would score such responses highly.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PPOConfig:
    """Configuration for PPO training of LLMs.

    ATTRIBUTES:
        clip_epsilon: Clipping parameter for PPO objective (0.2 typical)
        gamma: Discount factor for future rewards (0.99 typical)
        gae_lambda: GAE lambda for bias-variance tradeoff (0.95 typical)
        value_loss_coef: Weight of value function loss (0.5 typical)
        entropy_coef: Weight of entropy bonus for exploration (0.01 typical)
        max_grad_norm: Gradient clipping threshold (1.0 typical)
        ppo_epochs: Number of optimization epochs per batch (4 typical)
        mini_batch_size: Size of mini-batches for PPO updates
        kl_penalty_coef: KL divergence penalty coefficient (0.1 typical)
        target_kl: Target KL divergence for early stopping (0.02 typical)
        learning_rate: Learning rate for policy optimizer (1e-5 typical)
        vf_learning_rate: Learning rate for value function (may differ from policy)

    DESIGN DECISIONS:
        - Clip epsilon: smaller = more conservative, larger = faster learning
        - GAE lambda: 0.95 balances bias (low lambda) vs variance (high lambda)
        - KL penalty: prevents reward hacking while allowing learning
        - Separate LR for policy and value: value function often needs higher LR

    TRADEOFFS:
        - More PPO epochs = better sample efficiency but risk overfitting
        - Larger clip = faster learning but less stable
        - Higher KL penalty = safer but slower learning
    """

    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    kl_penalty_coef: float = 0.1
    target_kl: float = 0.02
    learning_rate: float = 1e-5
    vf_learning_rate: float = 3e-5


# ============================================================================
# ADVANTAGE ESTIMATION (GAE)
# ============================================================================

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    DEFINITION:
        GAE estimates the advantage of taking an action vs the average.
        It balances bias (from bootstrapping) and variance (from Monte Carlo)
        using the lambda parameter.

        A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
        where δ_t = r_t + γ·V(s_{t+1}) - V(s_t) is the TD error

    PROBLEM:
        We need advantages to know which actions were "better than expected."
        Simple Monte Carlo (full returns) has high variance.
        Temporal difference (one-step) has high bias.
        GAE smoothly interpolates between them.

    INPUTS:
        rewards: Array of rewards for each timestep, shape (T,)
        values: Value function estimates for each timestep, shape (T+1,)
        dones: Whether each timestep is terminal, shape (T,)
        gamma: Discount factor (how much we care about future rewards)
        lam: GAE lambda (bias-variance tradeoff, 0=high bias, 1=high variance)

    OUTPUTS:
        advantages: Advantage estimates, shape (T,)
        returns: Discounted returns, shape (T,)

    EXECUTION FLOW:
        1. Compute TD errors: δ_t = r_t + γ·V(s_{t+1})·(1-done_t) - V(s_t)
        2. Compute advantages backwards: A_t = δ_t + γλ·(1-done_t)·A_{t+1}
        3. Compute returns: R_t = A_t + V(s_t)

    COMPLEXITY:
        Time: O(T) where T = number of timesteps
        Space: O(T) for advantages and returns

    EDUCATIONAL:
        When λ=0: GAE = TD error (one-step advantage, high bias)
        When λ=1: GAE = Monte Carlo advantage (full returns, high variance)
        When λ=0.95: Sweet spot — low bias, reasonable variance

        For LLMs: each "timestep" is a generated token.
        Rewards come from the reward model at the end of generation.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)

    # Compute TD errors
    # δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:T]

    # Compute advantages backwards using GAE formula
    # A_t = δ_t + γλ · (1 - done_t) · A_{t+1}
    advantage = 0.0
    for t in reversed(range(T)):
        advantage = deltas[t] + gamma * lam * (1 - dones[t]) * advantage
        advantages[t] = advantage

    # Returns = advantages + values
    returns = advantages + values[:T]

    return advantages, returns


# ============================================================================
# PPO CLIPPED OBJECTIVE
# ============================================================================

def ppo_clipped_objective(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantages: np.ndarray,
    clip_epsilon: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the PPO clipped surrogate objective.

    DEFINITION:
        L^CLIP = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]
        where r(θ) = exp(log_π_new - log_π_old) = π_new / π_old

    PROBLEM:
        Standard policy gradient (REINFORCE) can make arbitrarily large
        updates, causing catastrophic performance collapse. PPO clips
        the objective to prevent the policy ratio from deviating too far.

    INPUTS:
        log_probs_new: Log probabilities under current policy
        log_probs_old: Log probabilities under old policy (when data was collected)
        advantages: Advantage estimates (from GAE)
        clip_epsilon: Clipping parameter (default 0.2)

    OUTPUTS:
        clipped_loss: The PPO loss (to be minimized, so negative of objective)
        ratio: The probability ratio π_new/π_old (for monitoring)

    EXECUTION FLOW:
        1. Compute ratio: r(θ) = exp(log_new - log_old)
        2. Compute surrogate: surr1 = r(θ) · A
        3. Compute clipped surrogate: surr2 = clip(r, 1-ε, 1+ε) · A
        4. Take minimum: L = min(surr1, surr2)
        5. Return negative (for minimization)

    COMPLEXITY:
        Time: O(B) where B = batch size
        Space: O(B) for the ratio and surrogate arrays

    EDUCATIONAL:
        The clipping acts as a "trust region":
        - If A > 0 (good action): ratio can go up to 1+ε (encourage more)
        - If A < 0 (bad action): ratio can go down to 1-ε (discourage)
        - Beyond these bounds, the gradient is zero (no further update)

        This prevents the "policy collapse" problem where a single bad
        update causes the model to generate garbage, which then gets
        negative reward, causing an even worse update, etc.

    INTERVIEW:
        Q: What happens without clipping?
        A: Large policy updates → distribution shift → poor samples →
        even larger updates → catastrophic collapse. The clip ensures
        we only update within a "trust region" where our advantage
        estimates are still valid.
    """
    # Compute probability ratio: r(θ) = π_new / π_old = exp(log_new - log_old)
    log_ratio = log_probs_new - log_probs_old
    ratio = np.exp(log_ratio)

    # Clamp log_ratio for numerical stability
    log_ratio = np.clip(log_ratio, -20, 20)

    # Surrogate objectives
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

    # PPO objective: take minimum (pessimistic bound)
    # We minimize the negative, so the loss is -min(surr1, surr2)
    ppo_loss = -np.minimum(surr1, surr2)

    return ppo_loss, ratio


# ============================================================================
# VALUE FUNCTION LOSS
# ============================================================================

def value_loss(
    values_new: np.ndarray,
    values_old: np.ndarray,
    returns: np.ndarray,
    clip_epsilon: float = 0.2,
) -> np.ndarray:
    """Compute the clipped value function loss.

    DEFINITION:
        L^VF = max((V_new - R)², (clip(V_new, V_old-ε, V_old+ε) - R)²)

    PROBLEM:
        Like the policy, the value function can also make large updates
        that destabilize training. Clipping the value function loss
        prevents this.

    INPUTS:
        values_new: Value predictions from current value function
        values_old: Value predictions from old value function (when data was collected)
        returns: Target returns (from GAE)
        clip_epsilon: Clipping parameter

    OUTPUTS:
        Value loss (scalar or array)

    EDUCATIONAL:
        The value function estimates "how good is this state?"
        It's trained to predict the expected return.
        Clipping prevents it from changing too rapidly.
    """
    # Unclipped value loss
    loss_unclipped = (values_new - returns) ** 2

    # Clipped value loss
    values_clipped = values_old + np.clip(
        values_new - values_old, -clip_epsilon, clip_epsilon
    )
    loss_clipped = (values_clipped - returns) ** 2

    # Take maximum (pessimistic)
    return np.maximum(loss_unclipped, loss_clipped)


# ============================================================================
# ENTROPY BONUS
# ============================================================================

def entropy_bonus(log_probs: np.ndarray) -> np.ndarray:
    """Compute entropy bonus for exploration.

    DEFINITION:
        H(π) = -E[log π(a|s)] = -Σ π(a) · log π(a)

    PROBLEM:
        Without entropy bonus, the policy can quickly collapse to
        always choosing the same (highest-reward) action, missing
        potentially better alternatives. Entropy encourages exploration.

    INPUTS:
        log_probs: Log probabilities of taken actions

    OUTPUTS:
        Entropy estimate (higher = more exploration)

    EDUCATIONAL:
        For LLMs, entropy bonus prevents the model from becoming
        too repetitive or "lazy" — always generating the same
        high-reward template response instead of exploring diverse
        good responses.

        Typical coefficient: 0.01 (small but meaningful)
    """
    return -log_probs


# ============================================================================
# KL DIVERGENCE PENALTY
# ============================================================================

def kl_divergence_penalty(
    log_probs_policy: np.ndarray,
    log_probs_reference: np.ndarray,
    coef: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute KL divergence penalty against reference model.

    DEFINITION:
        KL(π || π_ref) = E_π[log π(a|s) - log π_ref(a|s)]

    PROBLEM:
        The reward model is imperfect. Without KL penalty, the policy
        can "hack" the reward model — finding adversarial outputs that
        score high but are actually bad (e.g., repetitive, incoherent).
        KL penalty keeps the policy close to the reference (SFT) model.

    INPUTS:
        log_probs_policy: Log probabilities under current policy
        log_probs_reference: Log probabilities under reference (SFT) model
        coef: KL penalty coefficient

    OUTPUTS:
        kl_penalty: The KL penalty term (to be subtracted from reward)
        kl_estimate: Raw KL divergence estimate (for monitoring)

    EDUCATIONAL:
        The reference model is the SFT (supervised fine-tuned) model
        that was trained on human demonstrations. It represents
        "human-like" behavior. By penalizing divergence from it,
        we ensure the RL-trained model stays grounded in human-like
        responses while optimizing for reward.

        If KL is too high: model is just copying reference (not learning)
        If KL is too low: model is reward hacking (diverging from humans)
        Target: KL ≈ 0.01-0.05 (small but non-zero divergence)
    """
    # KL estimate: E[log π - log π_ref]
    kl_estimate = log_probs_policy - log_probs_reference

    # Penalty: coef * KL
    kl_penalty = coef * kl_estimate

    return kl_penalty, kl_estimate


# ============================================================================
# ROLLOUT BUFFER
# ============================================================================

@dataclass
class RolloutBuffer:
    """Stores collected experience for PPO training.

    ARCHITECTURE:
        - Collects (state, action, reward, value, log_prob) tuples
        - Computes advantages and returns using GAE
        - Provides mini-batches for PPO updates

    WHY BUFFER:
        PPO is an on-policy algorithm — it needs fresh data each iteration.
        The buffer collects a batch of experience, then PPO trains on it
        for multiple epochs before collecting new data.
    """

    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    advantages: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add a single timestep to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Compute GAE advantages and returns for the entire rollout.

        EXECUTION FLOW:
            1. Stack all values and append last_value for bootstrapping
            2. Call compute_gae to get advantages and returns
            3. Normalize advantages (optional but recommended)
        """
        values = np.array(self.values + [last_value], dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        self.advantages, self.returns = compute_gae(
            rewards, values, dones, gamma, gae_lambda
        )

        # Normalize advantages (reduces variance, improves stability)
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages) + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

    def get_batches(self, mini_batch_size: int) -> List[Dict]:
        """Generate randomized mini-batches for PPO training.

        EXECUTION FLOW:
            1. Create random permutation of indices
            2. Split into mini-batches of mini_batch_size
            3. Yield each batch as a dict of arrays
        """
        n = len(self.observations)
        indices = np.random.permutation(n)

        batches = []
        for start in range(0, n, mini_batch_size):
            end = min(start + mini_batch_size, n)
            batch_idx = indices[start:end]

            batches.append({
                'observations': np.array([self.observations[i] for i in batch_idx]),
                'actions': np.array([self.actions[i] for i in batch_idx]),
                'old_log_probs': np.array([self.log_probs[i] for i in batch_idx]),
                'advantages': self.advantages[batch_idx],
                'returns': self.returns[batch_idx],
            })

        return batches

    def clear(self):
        """Clear the buffer after PPO training."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None


# ============================================================================
# PPO TRAINER
# ============================================================================

class PPOTrainer:
    """PPO trainer for LLM alignment (simplified simulation).

    ARCHITECTURE:
        - Policy model: generates responses (LLM being trained)
        - Reference model: frozen SFT model (KL anchor)
        - Reward model: scores responses (frozen)
        - Value model: estimates expected returns (trained alongside policy)

    TRAINING LOOP:
        1. Generate responses using current policy
        2. Score responses with reward model
        3. Compute advantages using GAE
        4. Update policy using PPO clipped objective
        5. Update value function
        6. Repeat

    USAGE:
        trainer = PPOTrainer(config)
        for iteration in range(num_iterations):
            batch = trainer.collect_rollouts(prompts)
            metrics = trainer.train(batch)
    """

    def __init__(self, config: PPOConfig):
        """Initialize PPO trainer.

        Args:
            config: PPO training configuration
        """
        self.config = config
        self.buffer = RolloutBuffer()

        # Training metrics history
        self.metrics_history: List[Dict] = []

    def compute_ppo_loss(
        self,
        log_probs_new: np.ndarray,
        log_probs_old: np.ndarray,
        advantages: np.ndarray,
    ) -> Tuple[float, Dict]:
        """Compute PPO loss with clipping.

        EXECUTION FLOW:
            1. Compute clipped surrogate objective
            2. Compute entropy bonus
            3. Combine: total_loss = policy_loss - entropy_coef * entropy

        OUTPUTS:
            total_loss: Combined PPO loss
            metrics: Dict with ratio, entropy, clip_fraction
        """
        ppo_loss, ratio = ppo_clipped_objective(
            log_probs_new, log_probs_old, advantages, self.config.clip_epsilon
        )

        # Entropy bonus (encourages exploration)
        entropy = entropy_bonus(log_probs_new)

        # Combined loss
        total_loss = (
            np.mean(ppo_loss)
            - self.config.entropy_coef * np.mean(entropy)
        )

        # Metrics
        clip_fraction = np.mean(np.abs(ratio - 1.0) > self.config.clip_epsilon)
        metrics = {
            'ppo_loss': float(np.mean(ppo_loss)),
            'entropy': float(np.mean(entropy)),
            'ratio_mean': float(np.mean(ratio)),
            'ratio_max': float(np.max(ratio)),
            'clip_fraction': float(clip_fraction),
        }

        return total_loss, metrics

    def compute_value_loss(
        self,
        values_new: np.ndarray,
        values_old: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """Compute value function loss.

        EXECUTION FLOW:
            1. Compute clipped value loss
            2. Scale by value_loss_coef
            3. Return mean loss
        """
        vf_loss = value_loss(values_new, values_old, returns, self.config.clip_epsilon)
        return float(np.mean(vf_loss) * self.config.value_loss_coef)

    def train_step(
        self,
        batch: Dict,
        policy_fn: Callable,
        value_fn: Callable,
        ref_policy_fn: Callable,
    ) -> Dict:
        """Perform one PPO training step on a batch.

        DEFINITION:
            Runs one forward pass through policy and value models,
            computes losses, and returns metrics.

        EXECUTION FLOW:
            1. Get new log probs and values from current models
            2. Get reference log probs for KL penalty
            3. Compute PPO policy loss
            4. Compute value loss
            5. Compute KL penalty
            6. Combine all losses
            7. Return metrics

        NOTE:
            In a real implementation, this would:
            - Run the policy model to get new log probs
            - Run the value model to get new values
            - Run the reference model for KL
            - Compute gradients and update parameters

            Here we simulate the computation for educational purposes.
        """
        obs = batch['observations']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']

        # Simulate model forward passes
        # In reality, these would be neural network forward passes
        new_log_probs = policy_fn(obs)
        new_values = value_fn(obs)
        ref_log_probs = ref_policy_fn(obs)

        # PPO policy loss
        policy_loss, policy_metrics = self.compute_ppo_loss(
            new_log_probs, old_log_probs, advantages
        )

        # Value loss
        vf_loss = self.compute_value_loss(new_values, old_log_probs, returns)

        # KL penalty
        kl_penalty, kl_estimate = kl_divergence_penalty(
            new_log_probs, ref_log_probs, self.config.kl_penalty_coef
        )

        # Total loss
        total_loss = policy_loss + vf_loss + np.mean(kl_penalty)

        # Combine all metrics
        metrics = {
            **policy_metrics,
            'value_loss': vf_loss,
            'kl_estimate': float(np.mean(kl_estimate)),
            'kl_penalty': float(np.mean(kl_penalty)),
            'total_loss': float(total_loss),
        }

        return metrics

    def train(
        self,
        num_epochs: Optional[int] = None,
        policy_fn: Optional[Callable] = None,
        value_fn: Optional[Callable] = None,
        ref_policy_fn: Optional[Callable] = None,
    ) -> Dict:
        """Run PPO training for multiple epochs on the current buffer.

        EXECUTION FLOW:
            1. Compute returns and advantages using GAE
            2. For each PPO epoch:
               a. Generate random mini-batches
               b. For each mini-batch, run train_step
               c. Check KL divergence for early stopping
            3. Clear buffer
            4. Return aggregated metrics
        """
        if num_epochs is None:
            num_epochs = self.config.ppo_epochs

        # Default functions (random for simulation)
        if policy_fn is None:
            policy_fn = lambda x: np.random.randn(len(x)) * 0.1
        if value_fn is None:
            value_fn = lambda x: np.random.randn(len(x)) * 0.5
        if ref_policy_fn is None:
            ref_policy_fn = lambda x: np.random.randn(len(x)) * 0.1

        # Compute advantages
        self.buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        all_metrics = []
        early_stop = False

        for epoch in range(num_epochs):
            batches = self.buffer.get_batches(self.config.mini_batch_size)
            epoch_metrics = []

            for batch in batches:
                metrics = self.train_step(batch, policy_fn, value_fn, ref_policy_fn)
                epoch_metrics.append(metrics)

            # Aggregate epoch metrics
            avg_metrics = {
                k: float(np.mean([m[k] for m in epoch_metrics]))
                for k in epoch_metrics[0]
            }
            avg_metrics['epoch'] = epoch
            all_metrics.append(avg_metrics)

            # Early stopping if KL is too high
            if avg_metrics['kl_estimate'] > self.config.target_kl * 1.5:
                early_stop = True
                break

        # Clear buffer
        self.buffer.clear()

        # Aggregate final metrics
        final_metrics = {
            k: float(np.mean([m[k] for m in all_metrics]))
            for k in all_metrics[0] if k != 'epoch'
        }
        final_metrics['ppo_epochs_run'] = len(all_metrics)
        final_metrics['early_stopped'] = early_stop

        self.metrics_history.append(final_metrics)
        return final_metrics


# ============================================================================
# REWARD MODEL (SIMPLIFIED)
# ============================================================================

class SimpleRewardModel:
    """Simplified reward model for demonstration.

    ARCHITECTURE:
        - Scores responses based on simple heuristics
        - In production: neural network trained on human preferences

    EDUCATIONAL:
        A real reward model is trained on (chosen, rejected) pairs:
        L = -log(σ(r(chosen) - r(rejected)))
        where r(x) is the reward score for response x.
    """

    def __init__(self, bias: float = 0.0, noise_std: float = 0.1):
        """Initialize reward model.

        Args:
            bias: Reward bias (simulates reward model preferences)
            noise_std: Noise standard deviation (simulates reward uncertainty)
        """
        self.bias = bias
        self.noise_std = noise_std

    def score(self, response_quality: np.ndarray) -> np.ndarray:
        """Score responses based on quality.

        INPUTS:
            response_quality: Array of quality scores (0-1)

        OUTPUTS:
            rewards: Noised reward scores
        """
        rewards = response_quality + self.bias + np.random.randn(*response_quality.shape) * self.noise_std
        return rewards


# ============================================================================
# RLHF PIPELINE
# ============================================================================

class RLHFPipeline:
    """Complete RLHF pipeline using PPO.

    ARCHITECTURE:
        1. SFT model → Reference model (frozen)
        2. SFT model → Policy model (trained with PPO)
        3. Reward model (frozen, scores responses)
        4. Value model (trained alongside policy)

    TRAINING LOOP:
        1. Sample prompts from dataset
        2. Generate responses using policy
        3. Score with reward model
        4. Compute advantages (GAE)
        5. PPO update (multiple epochs)
        6. Repeat

    EDUCATIONAL:
        This pipeline demonstrates the full RLHF workflow.
        In production, each component would be a large neural network.
        Here we simulate with simple functions for clarity.
    """

    def __init__(self, config: PPOConfig):
        """Initialize RLHF pipeline.

        Args:
            config: PPO training configuration
        """
        self.config = config
        self.trainer = PPOTrainer(config)
        self.reward_model = SimpleRewardModel(bias=0.3, noise_std=0.1)

        # Simulated model states
        self.policy_quality = 0.5  # Starting quality
        self.metrics_log: List[Dict] = []

    def generate_responses(self, num_samples: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Generate responses using current policy (simulated).

        DEFINITION:
            In reality, this would run the LLM to generate text.
            Here we simulate by generating quality scores.

        OUTPUTS:
            observations: Random prompts (simulated)
            response_quality: Quality of generated responses (0-1)
        """
        observations = np.random.randn(num_samples, 8)  # Simulated embeddings
        # Response quality depends on policy quality + noise
        response_quality = np.clip(
            self.policy_quality + np.random.randn(num_samples) * 0.15,
            0, 1
        )
        return observations, response_quality

    def compute_rewards(self, response_quality: np.ndarray) -> np.ndarray:
        """Score responses with reward model."""
        return self.reward_model.score(response_quality)

    def train_iteration(
        self,
        num_samples: int = 64,
        num_epochs: int = 4,
    ) -> Dict:
        """Run one full RLHF training iteration.

        EXECUTION FLOW:
            1. Generate responses with current policy
            2. Score with reward model
            3. Compute values (simulated)
            4. Add to rollout buffer
            5. Run PPO training
            6. Update policy quality (simulated)

        OUTPUTS:
            metrics: Training metrics for this iteration
        """
        # Generate responses
        observations, response_quality = self.generate_responses(num_samples)
        rewards = self.compute_rewards(response_quality)

        # Simulate value estimates and log probs
        # values has num_samples+1 elements (last one for bootstrapping)
        values = np.concatenate([
            response_quality + np.random.randn(num_samples) * 0.1,
            [np.mean(response_quality)]  # bootstrap value
        ])
        log_probs = np.random.randn(num_samples) * 0.1 - 0.5

        # Add to buffer
        for i in range(num_samples):
            self.trainer.buffer.add(
                obs=observations[i],
                action=np.array([response_quality[i]]),
                reward=float(rewards[i]),
                value=float(values[i]),
                log_prob=float(log_probs[i]),
                done=(i == num_samples - 1),
            )

        # Run PPO training
        metrics = self.trainer.train(num_epochs=num_epochs)

        # Simulate policy improvement
        # In reality, the gradient update would improve the model
        improvement = 0.01 * (np.mean(rewards) - 0.5)
        self.policy_quality = np.clip(self.policy_quality + improvement, 0, 1)

        metrics['policy_quality'] = float(self.policy_quality)
        metrics['mean_reward'] = float(np.mean(rewards))
        self.metrics_log.append(metrics)

        return metrics

    def train(
        self,
        num_iterations: int = 50,
        num_samples: int = 64,
        num_epochs: int = 4,
        verbose: bool = True,
    ) -> List[Dict]:
        """Run the full RLHF training loop.

        EXECUTION FLOW:
            For each iteration:
                1. Collect rollout data
                2. Run PPO updates
                3. Log metrics
                4. Report progress
        """
        for i in range(num_iterations):
            metrics = self.train_iteration(num_samples, num_epochs)

            if verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{num_iterations}: "
                      f"reward={metrics['mean_reward']:.3f}, "
                      f"quality={metrics['policy_quality']:.3f}, "
                      f"kl={metrics['kl_estimate']:.4f}, "
                      f"clip={metrics['clip_fraction']:.2%}")

        return self.metrics_log


# ============================================================================
# DEMO / VISUALIZATION HELPERS
# ============================================================================

def demo_ppo_training(num_iterations: int = 50) -> Dict:
    """Run a complete PPO training demo.

    EDUCATIONAL:
        Demonstrates the RLHF pipeline with simulated models.
        Shows how policy quality improves over training iterations.
    """
    config = PPOConfig(
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=4,
        kl_penalty_coef=0.1,
        target_kl=0.02,
    )

    pipeline = RLHFPipeline(config)
    metrics = pipeline.train(
        num_iterations=num_iterations,
        num_samples=64,
        verbose=True,
    )

    return {
        'config': config,
        'metrics': metrics,
        'final_quality': pipeline.policy_quality,
        'iterations': num_iterations,
    }


def print_training_summary(result: Dict):
    """Print a summary of PPO training results."""
    print("=" * 60)
    print("PPO TRAINING SUMMARY")
    print("=" * 60)
    print(f"Iterations: {result['iterations']}")
    print(f"Final Policy Quality: {result['final_quality']:.3f}")
    print(f"Clip Epsilon: {result['config'].clip_epsilon}")
    print(f"KL Penalty Coef: {result['config'].kl_penalty_coef}")
    print(f"GAE Lambda: {result['config'].gae_lambda}")
    print()

    metrics = result['metrics']
    if metrics:
        first = metrics[0]
        last = metrics[-1]
        print(f"Reward:        {first['mean_reward']:.3f} → {last['mean_reward']:.3f}")
        print(f"KL Divergence: {first['kl_estimate']:.4f} → {last['kl_estimate']:.4f}")
        print(f"Clip Fraction: {first['clip_fraction']:.2%} → {last['clip_fraction']:.2%}")
        print(f"Entropy:       {first['entropy']:.3f} → {last['entropy']:.3f}")
    print("=" * 60)
