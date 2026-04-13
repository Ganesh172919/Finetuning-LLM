"""
################################################################################
ADVANCED GRPO — PRODUCTION-GRADE GROUP RELATIVE POLICY OPTIMIZATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is GRPO?
    GRPO (Group Relative Policy Optimization) is the training algorithm behind
    DeepSeek-R1, one of the most capable reasoning models in 2025. Unlike RLHF
    which needs a separate reward model, and DPO which needs preference pairs,
    GRPO generates N responses per prompt, scores them automatically, and
    optimizes the policy to prefer higher-scoring responses.

    KEY INSIGHT: Instead of comparing "chosen vs rejected" (DPO), GRPO
    compares N generated responses relative to each other within a group.

Algorithm Deep Dive:
    1. For each prompt, generate N responses from the current policy
    2. Score each response with a reward function (e.g., math correctness)
    3. Normalize rewards within the group (mean=0, std=1)
    4. Compute advantages: how much better/worse each response is vs group average
    5. Apply clipped policy gradient with KL penalty to prevent drift
    6. Update policy to increase probability of high-advantage responses

Why GRPO over RLHF/DPO?
    RLHF: Needs expensive reward model training, reward hacking risk
    DPO:  Needs human preference pairs (expensive, noisy)
    GRPO: Uses automatic rewards (math correctness, format compliance),
          no preference data needed, scales with compute

Mathematical Foundation:
    L_GRPO = -E[ min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t ) ] + β * KL(π || π_ref)

    Where:
    - r_t = π(a|s) / π_old(a|s)  — probability ratio
    - A_t = (R_i - mean(R)) / std(R)  — group-normalized advantage
    - ε   = clipping parameter (prevents too-large updates)
    - β   = KL penalty coefficient (keeps policy close to reference)
    - KL   = divergence from reference policy (prevents reward hacking)

Interview Questions:
    Q: "What is GRPO and how does it differ from PPO?"
    A: GRPO eliminates the value function (critic) by using group-relative
       advantages. PPO needs a separate value network; GRPO just generates
       N samples and compares them. This saves ~50% memory and compute.

    Q: "Why normalize advantages within groups?"
    A: It provides a natural baseline without needing a value function.
       The group mean acts as the baseline, and std normalization keeps
       the gradient magnitude stable across different reward scales.

    Q: "How does GRPO prevent reward hacking?"
    A: Three mechanisms: (1) KL penalty keeps policy close to reference,
       (2) clipping prevents too-large policy updates, (3) relative ranking
       within groups is more robust than absolute reward magnitudes.

################################################################################
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import json
import time

################################################################################
# SECTION 1: DATA STRUCTURES
################################################################################

@dataclass
class GRPOConfig:
    """
    GRPO Training Configuration
    ============================

    All hyperparameters for GRPO training. Each parameter includes
    its purpose, typical values, and impact on training.

    Interview Question:
        Q: "What are the key hyperparameters in GRPO?"
        A: (1) n_samples: responses per prompt (8-64)
           (2) beta: KL penalty weight (0.01-0.5)
           (3) clip_epsilon: PPO-style clipping (0.1-0.3)
           (4) learning_rate: optimizer LR (1e-6 to 5e-6)
    """

    # === Core GRPO Parameters ===

    n_samples: int = 8
    # Number of responses generated per prompt.
    # More samples = better advantage estimates = higher quality training.
    # But more samples = more compute. 8 is a good balance.
    # DeepSeek-R1 used 64 samples for math reasoning.

    beta: float = 0.04
    # KL penalty coefficient. Controls how much the policy can deviate
    # from the reference policy. Higher beta = more conservative.
    # Too low: reward hacking. Too high: slow learning.
    # DeepSeek-R1 used 0.04 for reasoning tasks.

    clip_epsilon: float = 0.2
    # PPO-style clipping. Prevents too-large policy updates.
    # If ratio > 1+eps or < 1-eps, we clip it.
    # Standard PPO value, works well for GRPO too.

    learning_rate: float = 1e-6
    # Learning rate for policy updates.
    # GRPO needs smaller LR than supervised fine-tuning.
    # Typical: 1e-6 to 5e-6 for 7B models.

    # === Advantage Normalization ===

    advantage_normalization: str = "group"
    # How to normalize advantages:
    # "group": normalize within each prompt group (standard GRPO)
    # "global": normalize across entire batch
    # "none": no normalization (raw rewards)

    epsilon_advantage: float = 1e-8
    # Small constant to prevent division by zero in advantage normalization.

    # === KL Divergence Options ===

    kl_type: str = "approx"
    # Type of KL divergence to use:
    # "approx": KL ≈ log(π/π_ref) (cheap approximation)
    # "exact": exact KL using full vocabulary distribution
    # "adaptive": beta adjusts based on current KL magnitude

    adaptive_kl_target: float = 6.0
    # Target KL divergence for adaptive beta.
    # If KL > target, increase beta. If KL < target, decrease beta.

    adaptive_kl_lr: float = 0.1
    # How fast adaptive beta adjusts.

    # === Training Loop ===

    epochs_per_batch: int = 1
    # Number of gradient updates per batch of generated responses.
    # More epochs = more data efficiency but risk of overfitting.
    # Standard: 1 epoch (on-policy methods prefer fresh data).

    max_grad_norm: float = 1.0
    # Gradient clipping threshold. Prevents gradient explosions.
    # Standard value for transformer fine-tuning.

    # === Reward Shaping ===

    reward_clip: float = 10.0
    # Clip reward values to [-clip, +clip] to prevent outliers
    # from dominating the gradient.

    use_reward_scaling: bool = True
    # Scale rewards to have mean=0, std=1 across the entire batch.
    # Helps with training stability.

    # === Reference Policy ===

    ref_update_freq: int = 0
    # How often to update the reference policy.
    # 0 = never (fixed reference from SFT model)
    # >0 = update every N steps (frozen reference with periodic refresh)

    ref_ema_decay: float = 0.999
    # EMA decay for reference policy update (if ref_update_freq > 0).


class RewardType(Enum):
    """
    Types of reward functions for GRPO training.

    Different tasks need different reward signals:
    - MATH: binary correctness (1 if correct, 0 if not)
    - FORMAT: reward for following output format (e.g., <think>...</think>)
    - LENGTH: penalize overly long responses
    - COMPOSITE: weighted combination of multiple rewards
    """
    MATH = "math"              # Binary correctness reward
    FORMAT = "format"          # Format compliance reward
    LENGTH = "length"          # Length penalty
    COMPOSITE = "composite"    # Weighted combination
    CUSTOM = "custom"          # User-defined function


@dataclass
class GRPOBatch:
    """
    A batch of prompts with their generated responses and rewards.

    Structure:
        prompts:    [batch_size] strings
        responses:  [batch_size × n_samples] strings
        logprobs:   [batch_size × n_samples] float — log π(response|prompt)
        ref_logprobs: [batch_size × n_samples] float — log π_ref(response|prompt)
        rewards:    [batch_size × n_samples] float — reward scores

    This is the core data structure that flows through the GRPO pipeline.
    """
    prompts: List[str]
    responses: List[List[str]]
    logprobs: np.ndarray        # [batch × n_samples]
    ref_logprobs: np.ndarray    # [batch × n_samples]
    rewards: np.ndarray         # [batch × n_samples]


################################################################################
# SECTION 2: REWARD FUNCTIONS
################################################################################

class RewardFunction:
    """
    Base class for GRPO reward functions.

    Reward functions are the key to GRPO quality. They define what
    "good" means for your task. The reward function is what makes
    DeepSeek-R1 good at math — it rewards correct answers.

    Design Decision:
        We use a class hierarchy rather than plain functions because:
        1. Reward functions need configuration (weights, thresholds)
        2. We need to compose multiple rewards (CompositeReward)
        3. We need logging and debugging capabilities
        4. Some rewards need state (running statistics)
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self.call_count = 0
        self.total_reward = 0.0

    def __call__(self, prompt: str, response: str) -> float:
        raise NotImplementedError

    def get_stats(self) -> Dict:
        """Return reward statistics for monitoring."""
        avg = self.total_reward / max(self.call_count, 1)
        return {"name": self.name, "calls": self.call_count, "avg_reward": avg}


class MathReward(RewardFunction):
    """
    Mathematical Correctness Reward
    ================================

    Rewards responses that contain the correct mathematical answer.

    Strategy:
    1. Extract the final answer from the response (look for patterns like
       "the answer is X", "\\boxed{X}", "= X")
    2. Compare with ground truth
    3. Return 1.0 if correct, 0.0 if incorrect

    Why this works:
    - Math has objectively correct answers
    - Binary reward is simple but effective
    - DeepSeek-R1 used this for math reasoning training

    Interview Question:
        Q: "How do you design rewards for math reasoning?"
        A: Binary correctness is surprisingly effective. Extract the
           final answer using regex patterns, compare with ground truth.
           Add format bonuses (e.g., showing work) for better learning.
    """

    def __init__(self, ground_truth: Dict[str, str] = None):
        super().__init__("math")
        self.ground_truth = ground_truth or {}

    def __call__(self, prompt: str, response: str) -> float:
        self.call_count += 1

        # Extract answer from response
        extracted = self._extract_answer(response)

        # Get ground truth for this prompt
        truth = self.ground_truth.get(prompt, None)
        if truth is None:
            # If no ground truth, use a heuristic score
            score = self._heuristic_score(response)
        else:
            score = 1.0 if self._answers_match(extracted, truth) else 0.0

        self.total_reward += score
        return score

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract the final answer from a response."""
        import re

        # Try \boxed{...} first (LaTeX format)
        boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
        if boxed:
            return boxed[-1].strip()

        # Try "the answer is X"
        answer_patterns = [
            r'the answer is[:\s]+([^\n.]+)',
            r'answer[:\s]+([^\n.]+)',
            r'= ([\d\.\-\+/]+)\s*$',
        ]
        for pattern in answer_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                return matches[-1].strip()

        # Fallback: last number in the response
        numbers = re.findall(r'[\-]?\d+\.?\d*', response)
        if numbers:
            return numbers[-1]

        return None

    def _answers_match(self, extracted: Optional[str], truth: str) -> bool:
        """Check if extracted answer matches ground truth."""
        if extracted is None:
            return False

        # Normalize both answers
        def normalize(s):
            s = s.strip().rstrip('.').lower()
            # Try to parse as number
            try:
                return str(float(s))
            except ValueError:
                return s

        return normalize(extracted) == normalize(truth)

    def _heuristic_score(self, response: str) -> float:
        """Heuristic scoring when no ground truth is available."""
        score = 0.0

        # Reward showing work (has math operators)
        if any(op in response for op in ['=', '+', '-', '×', '÷', '\\frac']):
            score += 0.2

        # Reward structured reasoning
        if any(marker in response.lower() for marker in ['step', 'first', 'therefore', 'thus']):
            score += 0.2

        # Reward having a clear answer
        if self._extract_answer(response) is not None:
            score += 0.3

        # Reward conciseness (not too long)
        if len(response) < 2000:
            score += 0.1

        # Reward proper formatting
        if '\\boxed' in response or 'answer' in response.lower():
            score += 0.2

        return min(score, 1.0)


class FormatReward(RewardFunction):
    """
    Format Compliance Reward
    =========================

    Rewards responses that follow the expected output format.

    For reasoning models like DeepSeek-R1, the expected format is:
    <think>
    ... reasoning steps ...
    </think>

    <answer>
    ... final answer ...
    </answer>

    Why format matters:
    - Structured output is easier to parse
    - Separating reasoning from answer improves quality
    - Makes the model's thinking process visible and debuggable
    """

    def __init__(self, require_think: bool = True, require_answer: bool = True):
        super().__init__("format")
        self.require_think = require_think
        self.require_answer = require_answer

    def __call__(self, prompt: str, response: str) -> float:
        self.call_count += 1
        score = 0.0

        # Check for <think> tags
        if self.require_think:
            if '<think>' in response and '</think>' in response:
                score += 0.4
                # Bonus: thinking comes before answer
                think_end = response.index('</think>')
                if 'answer' not in response[:think_end].lower():
                    score += 0.1
            elif '<think>' in response:
                score += 0.2  # Partial credit

        # Check for answer tags
        if self.require_answer:
            if '<answer>' in response and '</answer>' in response:
                score += 0.4
            elif any(marker in response.lower() for marker in ['answer:', 'the answer is']):
                score += 0.2  # Partial credit for alternative formats

        # Bonus: response is well-structured
        if response.count('\n') >= 2:  # Multiple lines
            score += 0.1

        self.total_reward += score
        return score


class CompositeReward(RewardFunction):
    """
    Composite Reward Function
    ===========================

    Combines multiple reward signals with configurable weights.

    This is how production GRPO systems work — they combine:
    - Correctness reward (is the answer right?)
    - Format reward (does it follow the expected format?)
    - Length penalty (is it reasonably concise?)
    - Style reward (is it clear and well-written?)

    The weights determine the relative importance of each signal.

    Interview Question:
        Q: "How do you combine multiple reward signals?"
        A: Weighted sum with careful tuning. Start with correctness
           dominant (0.6-0.8), add format bonus (0.1-0.2), and
           length penalty (0.05-0.1). Monitor each component separately.
    """

    def __init__(self, rewards_and_weights: List[Tuple[RewardFunction, float]]):
        super().__init__("composite")
        self.rewards_and_weights = rewards_and_weights

    def __call__(self, prompt: str, response: str) -> float:
        self.call_count += 1
        total = 0.0

        for reward_fn, weight in self.rewards_and_weights:
            score = reward_fn(prompt, response)
            total += weight * score

        self.total_reward += total
        return total


################################################################################
# SECTION 3: ADVANTAGE ESTIMATION
################################################################################

class AdvantageEstimator:
    """
    Advantage Estimation for GRPO
    ===============================

    The core innovation of GRPO: computing advantages by comparing
    responses within a group, rather than using a value function.

    Advantage = how much better is this response compared to the
    average response in its group?

    A_i = (R_i - mean(R_group)) / std(R_group)

    This is simpler and more memory-efficient than PPO's value function
    approach, and works surprisingly well for reasoning tasks.

    Interview Question:
        Q: "How does GRPO compute advantages without a value function?"
        A: It generates N responses per prompt, scores them all, and
           uses the group statistics (mean, std) as a natural baseline.
           Each response's advantage is its z-score within the group.
    """

    def __init__(self, config: GRPOConfig):
        self.config = config

    def compute_advantages(self, rewards: np.ndarray) -> np.ndarray:
        """
        Compute group-normalized advantages.

        Args:
            rewards: [batch_size × n_samples] reward scores

        Returns:
            advantages: [batch_size × n_samples] normalized advantages

        Algorithm:
            For each prompt group (row):
            1. Compute mean and std of rewards in the group
            2. Normalize: advantage = (reward - mean) / (std + eps)
            3. Optionally clip to prevent extreme values

        Time Complexity: O(batch_size × n_samples)
        Space Complexity: O(batch_size × n_samples)
        """
        if self.config.advantage_normalization == "group":
            # Standard GRPO: normalize within each prompt group
            mean = np.mean(rewards, axis=1, keepdims=True)
            std = np.std(rewards, axis=1, keepdims=True)
            advantages = (rewards - mean) / (std + self.config.epsilon_advantage)

        elif self.config.advantage_normalization == "global":
            # Global normalization across entire batch
            mean = np.mean(rewards)
            std = np.std(rewards)
            advantages = (rewards - mean) / (std + self.config.epsilon_advantage)

        elif self.config.advantage_normalization == "none":
            # No normalization — use raw rewards
            advantages = rewards

        else:
            raise ValueError(f"Unknown normalization: {self.config.advantage_normalization}")

        return advantages

    def compute_gae_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> np.ndarray:
        """
        Generalized Advantage Estimation (GAE) for GRPO.

        When you have per-token rewards (not just per-response), GAE
        provides better credit assignment by considering the temporal
        structure of the reward signal.

        This is used in advanced GRPO variants where the reward model
        provides token-level feedback (e.g., process reward models).

        Args:
            rewards: [batch × n_samples × seq_len] per-token rewards
            values: [batch × n_samples × seq_len] value estimates
            gamma: discount factor
            lam: GAE lambda

        Returns:
            advantages: [batch × n_samples × seq_len] GAE advantages
        """
        # GAE: A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
        # where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(rewards.shape[-1])):
            next_values = values[..., t + 1] if t < rewards.shape[-1] - 1 else 0
            delta = rewards[..., t] + gamma * next_values - values[..., t]
            advantages[..., t] = last_gae = delta + gamma * lam * last_gae

        return advantages


################################################################################
# SECTION 4: GRPO TRAINER (PRODUCTION-GRADE)
################################################################################

class AdvancedGRPOTrainer:
    """
    Production-Grade GRPO Trainer
    ===============================

    Complete GRPO training implementation with all the bells and whistles
    needed for real reasoning model training.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    GRPO Training Loop                       │
    │                                                             │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
    │  │  Prompt   │───▶│ Generate │───▶│  Score   │              │
    │  │  Batch    │    │ N per    │    │ Rewards  │              │
    │  │          │    │ prompt   │    │          │              │
    │  └──────────┘    └──────────┘    └──────────┘              │
    │                                            │                │
    │                                            ▼                │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
    │  │  Update   │◀───│ Compute  │◀───│ Estimate │              │
    │  │  Policy   │    │  Loss    │    │Advantages│              │
    │  │          │    │          │    │          │              │
    │  └──────────┘    └──────────┘    └──────────┘              │
    │                                                             │
    │  Key Components:                                            │
    │  - Clipped policy gradient (PPO-style)                      │
    │  - KL penalty (prevents reward hacking)                     │
    │  - Advantage normalization (group-relative)                 │
    │  - Gradient clipping (training stability)                   │
    └─────────────────────────────────────────────────────────────┘

    Interview Question:
        Q: "Walk me through a GRPO training step."
        A: (1) Sample prompts from dataset
           (2) Generate N responses per prompt from current policy
           (3) Score each response with reward function
           (4) Compute group-normalized advantages
           (5) Compute log probabilities under current and reference policy
           (6) Compute clipped surrogate loss with KL penalty
           (7) Backpropagate and update policy
    """

    def __init__(
        self,
        policy_model,          # Current policy (to be updated)
        reference_model,       # Frozen reference policy (SFT checkpoint)
        reward_fn: RewardFunction,
        config: GRPOConfig = None,
        tokenizer=None,
        optimizer=None
    ):
        self.policy = policy_model
        self.reference = reference_model
        self.reward_fn = reward_fn
        self.config = config or GRPOConfig()
        self.tokenizer = tokenizer
        self.optimizer = optimizer

        # Advantage estimator
        self.advantage_estimator = AdvantageEstimator(self.config)

        # Training statistics
        self.step_count = 0
        self.training_log = []
        self.kl_history = []
        self.reward_history = []

        # Adaptive KL state
        self.current_beta = self.config.beta

    def generate_responses(
        self,
        prompts: List[str],
        n_samples: int = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> List[List[str]]:
        """
        Generate N responses for each prompt.

        This is where the "group" in GRPO comes from — we generate
        a group of responses for each prompt, then compare them.

        Args:
            prompts: List of prompt strings
            n_samples: Number of responses per prompt (default: config)
            max_new_tokens: Maximum generation length
            temperature: Sampling temperature (higher = more diverse)
            top_p: Nucleus sampling threshold

        Returns:
            responses: [len(prompts) × n_samples] generated strings

        Note: In production, this uses vLLM or similar for efficient
        batched generation. Here we show the conceptual flow.
        """
        n_samples = n_samples or self.config.n_samples
        all_responses = []

        for prompt in prompts:
            responses = []
            for _ in range(n_samples):
                # In production: self.policy.generate(prompt, ...)
                # Here: placeholder for the generation logic
                response = self._sample_response(
                    prompt, max_new_tokens, temperature, top_p
                )
                responses.append(response)
            all_responses.append(responses)

        return all_responses

    def _sample_response(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ) -> str:
        """
        Sample a single response from the policy.

        In production, this calls the model's generate() method.
        Here we show the sampling logic conceptually.
        """
        # Placeholder: in production, this calls the actual model
        # self.policy.generate(prompt, max_tokens, temperature, top_p)
        return f"[Generated response for: {prompt[:50]}...]"

    def compute_logprobs(
        self,
        model,
        prompts: List[str],
        responses: List[List[str]]
    ) -> np.ndarray:
        """
        Compute log probabilities of responses under a model.

        log π(response | prompt) = Σ_t log π(token_t | prompt, tokens_{<t})

        Args:
            model: The model to score with (policy or reference)
            prompts: List of prompts
            responses: [batch × n_samples] responses

        Returns:
            logprobs: [batch × n_samples] log probabilities
        """
        batch_size = len(prompts)
        n_samples = len(responses[0])
        logprobs = np.zeros((batch_size, n_samples))

        for i, prompt in enumerate(prompts):
            for j, response in enumerate(responses[i]):
                # In production: compute token-level logprobs and sum
                # logprobs[i, j] = model.compute_logprob(prompt, response)
                logprobs[i, j] = np.random.randn() * 0.5  # Placeholder

        return logprobs

    def compute_grpo_loss(
        self,
        logprobs: np.ndarray,       # [batch × n_samples] current policy logprobs
        ref_logprobs: np.ndarray,    # [batch × n_samples] reference policy logprobs
        advantages: np.ndarray       # [batch × n_samples] normalized advantages
    ) -> Tuple[float, Dict]:
        """
        Compute the GRPO loss function.

        L_GRPO = -E[ min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t ) ] + β * KL

        Where:
        - r_t = exp(log_π - log_π_ref) — probability ratio
        - A_t — group-normalized advantage
        - ε — clipping parameter
        - β — KL penalty weight

        This is the heart of GRPO. It's similar to PPO's clipped objective
        but uses group-relative advantages instead of GAE.

        Args:
            logprobs: log π(response|prompt) under current policy
            ref_logprobs: log π(response|prompt) under reference policy
            advantages: group-normalized advantages

        Returns:
            loss: scalar loss value
            info: dictionary of training metrics

        Time Complexity: O(batch_size × n_samples)
        Space Complexity: O(batch_size × n_samples)
        """
        # === Step 1: Compute probability ratio ===
        # r_t = π(a|s) / π_ref(a|s) = exp(log_π - log_π_ref)
        log_ratio = logprobs - ref_logprobs
        ratio = np.exp(log_ratio)

        # === Step 2: Compute clipped surrogate loss ===
        # PPO-style clipping: prevents too-large policy updates
        eps = self.config.clip_epsilon
        clipped_ratio = np.clip(ratio, 1.0 - eps, 1.0 + eps)

        # Surrogate loss: min(ratio * A, clipped_ratio * A)
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        surrogate_loss = -np.mean(np.minimum(surrogate1, surrogate2))

        # === Step 3: Compute KL divergence penalty ===
        # KL(π || π_ref) prevents the policy from diverging too far
        kl_loss = self._compute_kl_penalty(logprobs, ref_logprobs)

        # === Step 4: Combine losses ===
        total_loss = surrogate_loss + self.current_beta * kl_loss

        # === Step 5: Compute metrics for logging ===
        info = {
            "surrogate_loss": float(surrogate_loss),
            "kl_loss": float(kl_loss),
            "total_loss": float(total_loss),
            "mean_ratio": float(np.mean(ratio)),
            "max_ratio": float(np.max(ratio)),
            "min_ratio": float(np.min(ratio)),
            "mean_advantage": float(np.mean(advantages)),
            "clip_fraction": float(np.mean(np.abs(ratio - 1.0) > eps)),
            "beta": self.current_beta,
        }

        return total_loss, info

    def _compute_kl_penalty(
        self, logprobs: np.ndarray, ref_logprobs: np.ndarray
    ) -> float:
        """
        Compute KL divergence between current and reference policy.

        KL(π || π_ref) ≈ E[ log(π/π_ref) ] = E[ logprobs - ref_logprobs ]

        This is the "approximate" KL — it's cheap and works well in practice.
        The exact KL would require summing over the full vocabulary.
        """
        if self.config.kl_type == "approx":
            # Approximate KL: KL ≈ log(π/π_ref)
            kl = np.mean(logprobs - ref_logprobs)
        elif self.config.kl_type == "exact":
            # Exact KL: requires full vocabulary distribution
            # More accurate but much more expensive
            kl = np.mean(np.exp(logprobs) * (logprobs - ref_logprobs))
        else:
            kl = np.mean(logprobs - ref_logprobs)

        return float(kl)

    def _update_adaptive_beta(self, current_kl: float):
        """
        Adaptively adjust beta to maintain target KL divergence.

        If KL is too high (policy diverging), increase beta.
        If KL is too low (policy not learning), decrease beta.

        This is crucial for stable GRPO training — the right beta
        is hard to tune manually and changes during training.
        """
        if self.config.kl_type != "adaptive":
            return

        target = self.config.adaptive_kl_target
        lr = self.config.adaptive_kl_lr

        if current_kl > target * 1.5:
            # KL too high — increase penalty
            self.current_beta *= (1.0 + lr)
        elif current_kl < target / 1.5:
            # KL too low — decrease penalty
            self.current_beta *= (1.0 - lr)

        # Clamp beta to reasonable range
        self.current_beta = np.clip(self.current_beta, 0.001, 1.0)

    def training_step(
        self,
        prompts: List[str],
        ground_truth: Optional[List[str]] = None
    ) -> Dict:
        """
        Perform one GRPO training step.

        Complete pipeline:
        1. Generate N responses per prompt
        2. Score with reward function
        3. Compute logprobs under current and reference policy
        4. Compute advantages
        5. Compute loss and update policy

        Args:
            prompts: batch of prompt strings
            ground_truth: optional ground truth answers for reward computation

        Returns:
            metrics: dictionary of training metrics
        """
        step_start = time.time()

        # === Step 1: Generate responses ===
        responses = self.generate_responses(prompts)

        # === Step 2: Compute rewards ===
        rewards = np.zeros((len(prompts), self.config.n_samples))
        for i, prompt in enumerate(prompts):
            for j, response in enumerate(responses[i]):
                rewards[i, j] = self.reward_fn(prompt, response)

        # Clip rewards to prevent outliers
        rewards = np.clip(rewards, -self.config.reward_clip, self.config.reward_clip)

        # === Step 3: Compute log probabilities ===
        logprobs = self.compute_logprobs(self.policy, prompts, responses)
        ref_logprobs = self.compute_logprobs(self.reference, prompts, responses)

        # === Step 4: Compute advantages ===
        advantages = self.advantage_estimator.compute_advantages(rewards)

        # === Step 5: Compute loss and update ===
        loss, info = self.compute_grpo_loss(logprobs, ref_logprobs, advantages)

        # === Step 6: Adaptive beta update ===
        self._update_adaptive_beta(info["kl_loss"])

        # === Step 7: Log metrics ===
        step_time = time.time() - step_start
        info.update({
            "step": self.step_count,
            "mean_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "step_time": step_time,
        })

        self.training_log.append(info)
        self.reward_history.append(float(np.mean(rewards)))
        self.kl_history.append(info["kl_loss"])
        self.step_count += 1

        return info

    def get_training_summary(self) -> Dict:
        """Get summary of training progress."""
        if not self.training_log:
            return {"status": "no training steps completed"}

        recent = self.training_log[-10:]
        return {
            "total_steps": self.step_count,
            "mean_reward_recent": np.mean([s["mean_reward"] for s in recent]),
            "mean_kl_recent": np.mean([s["kl_loss"] for s in recent]),
            "mean_loss_recent": np.mean([s["total_loss"] for s in recent]),
            "current_beta": self.current_beta,
            "reward_fn_stats": self.reward_fn.get_stats(),
        }


################################################################################
# SECTION 5: DEEPSEEK-R1 STYLE TRAINING PIPELINE
################################################################################

class DeepSeekR1Pipeline:
    """
    DeepSeek-R1 Style Training Pipeline
    ======================================

    Complete pipeline for training a reasoning model using GRPO,
    following the DeepSeek-R1 methodology.

    Pipeline Stages:
    ┌─────────────────────────────────────────────────────────────┐
    │ Stage 1: Cold Start (SFT)                                   │
    │   - Fine-tune base model on reasoning examples              │
    │   - Teach the model the <think>...</think> format            │
    │   - Use curated CoT data                                    │
    ├─────────────────────────────────────────────────────────────┤
    │ Stage 2: Reasoning RL (GRPO)                                │
    │   - Train with GRPO using rule-based rewards                │
    │   - Reward: correctness of final answer                     │
    │   - This is where the "aha moment" emerges                  │
    ├─────────────────────────────────────────────────────────────┤
    │ Stage 3: Rejection Sampling + SFT                           │
    │   - Generate many responses, keep only correct ones         │
    │   - Fine-tune on the filtered high-quality data             │
    │   - Mix with general SFT data to prevent forgetting         │
    ├─────────────────────────────────────────────────────────────┤
    │ Stage 4: All-Scenario RL                                    │
    │   - GRPO with mixed rewards (helpfulness + correctness)     │
    │   - Align for both reasoning AND general conversation       │
    │   - Final polish                                            │
    └─────────────────────────────────────────────────────────────┘

    Interview Question:
        Q: "How was DeepSeek-R1 trained?"
        A: Four stages: (1) SFT cold start with CoT data,
           (2) GRPO reasoning RL with correctness rewards,
           (3) rejection sampling to create high-quality SFT data,
           (4) final RL with mixed rewards for both reasoning
           and helpfulness.
    """

    def __init__(self, base_model, tokenizer):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.stages_completed = []

    def stage1_cold_start_sft(
        self,
        cot_data: List[Dict],
        epochs: int = 3,
        lr: float = 2e-5
    ) -> Dict:
        """
        Stage 1: Cold Start SFT

        Fine-tune the base model on chain-of-thought examples.
        This teaches the model the reasoning format before RL.

        Args:
            cot_data: List of {"prompt": ..., "response": "...</think>..."}
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            metrics: Training metrics
        """
        # In production: standard SFT training loop
        # self.base_model.fit(cot_data, epochs, lr)
        self.stages_completed.append("cold_start_sft")
        return {"stage": "cold_start_sft", "status": "complete"}

    def stage2_reasoning_rl(
        self,
        prompts: List[str],
        reward_fn: RewardFunction,
        n_steps: int = 1000,
        config: GRPOConfig = None
    ) -> Dict:
        """
        Stage 2: Reasoning RL with GRPO

        This is the key stage where reasoning capability emerges.
        The model learns to reason by being rewarded for correct answers.

        The "aha moment": During this stage, the model learns to
        self-correct, verify its work, and explore multiple approaches —
        all without being explicitly taught these behaviors.

        Args:
            prompts: List of reasoning prompts (math, logic, etc.)
            reward_fn: Reward function (typically correctness-based)
            n_steps: Number of GRPO training steps
            config: GRPO configuration

        Returns:
            metrics: Training metrics including reward progression
        """
        config = config or GRPOConfig(
            n_samples=16,       # More samples for better advantage estimates
            beta=0.04,          # DeepSeek-R1's KL coefficient
            clip_epsilon=0.2,   # Standard PPO clipping
            learning_rate=1e-6, # Conservative LR for stability
        )

        trainer = AdvancedGRPOTrainer(
            policy_model=self.base_model,
            reference_model=self.base_model,  # Will be frozen copy
            reward_fn=reward_fn,
            config=config,
            tokenizer=self.tokenizer,
        )

        metrics_history = []
        for step in range(n_steps):
            # Sample a batch of prompts
            batch_prompts = prompts[step % len(prompts):step % len(prompts) + 4]
            metrics = trainer.training_step(batch_prompts)
            metrics_history.append(metrics)

            if step % 100 == 0:
                summary = trainer.get_training_summary()
                print(f"Step {step}: reward={summary['mean_reward_recent']:.3f}, "
                      f"kl={summary['mean_kl_recent']:.3f}")

        self.stages_completed.append("reasoning_rl")
        return {"stage": "reasoning_rl", "metrics": metrics_history}

    def stage3_rejection_sampling(
        self,
        prompts: List[str],
        n_samples: int = 32,
        keep_top_k: int = 1
    ) -> List[Dict]:
        """
        Stage 3: Rejection Sampling

        Generate many responses per prompt, keep only the best ones.
        This creates high-quality training data for a second SFT round.

        Key insight: The RL-trained model can generate better responses
        than the original SFT data. By filtering for correctness,
        we create a dataset of "ideal" reasoning traces.

        Args:
            prompts: List of prompts
            n_samples: Responses to generate per prompt
            keep_top_k: Number of best responses to keep

        Returns:
            filtered_data: High-quality prompt-response pairs
        """
        filtered_data = []

        for prompt in prompts:
            # Generate many responses
            responses = []
            for _ in range(n_samples):
                # response = self.base_model.generate(prompt)
                # responses.append(response)
                pass  # Placeholder

            # Score and rank
            # scored = [(r, reward_fn(prompt, r)) for r in responses]
            # scored.sort(key=lambda x: x[1], reverse=True)

            # Keep top-k
            # for response, score in scored[:keep_top_k]:
            #     filtered_data.append({"prompt": prompt, "response": response})

            pass  # Placeholder

        self.stages_completed.append("rejection_sampling")
        return filtered_data

    def stage4_all_scenario_rl(
        self,
        reasoning_prompts: List[str],
        general_prompts: List[str],
        n_steps: int = 500
    ) -> Dict:
        """
        Stage 4: All-Scenario RL

        Final RL training that balances reasoning quality with
        general helpfulness. This prevents the model from becoming
        a pure reasoning machine that can't hold a conversation.

        Uses a composite reward:
        - 70% correctness (for reasoning prompts)
        - 20% helpfulness (for general prompts)
        - 10% format compliance

        Args:
            reasoning_prompts: Math/logic reasoning prompts
            general_prompts: General conversation prompts
            n_steps: Number of training steps

        Returns:
            metrics: Training metrics
        """
        # Composite reward combining multiple signals
        math_reward = MathReward()
        format_reward = FormatReward()
        composite = CompositeReward([
            (math_reward, 0.7),
            (format_reward, 0.3),
        ])

        # Train with mixed prompts
        all_prompts = reasoning_prompts + general_prompts
        # ... training loop ...

        self.stages_completed.append("all_scenario_rl")
        return {"stage": "all_scenario_rl", "status": "complete"}

    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status."""
        return {
            "stages_completed": self.stages_completed,
            "next_stage": self._get_next_stage(),
        }

    def _get_next_stage(self) -> Optional[str]:
        """Determine the next pipeline stage."""
        all_stages = ["cold_start_sft", "reasoning_rl", "rejection_sampling", "all_scenario_rl"]
        for stage in all_stages:
            if stage not in self.stages_completed:
                return stage
        return None


################################################################################
# SECTION 6: TESTING AND DEMONSTRATION
################################################################################

def demonstrate_grpo():
    """
    Comprehensive GRPO demonstration.

    Shows all the key components working together:
    1. Reward functions
    2. Advantage estimation
    3. GRPO loss computation
    4. Training loop
    """
    print("=" * 70)
    print("ADVANCED GRPO DEMONSTRATION")
    print("=" * 70)

    # === Setup ===
    config = GRPOConfig(
        n_samples=8,
        beta=0.04,
        clip_epsilon=0.2,
        advantage_normalization="group",
    )

    # Create reward function
    math_reward = MathReward()
    format_reward = FormatReward()
    composite = CompositeReward([
        (math_reward, 0.7),
        (format_reward, 0.3),
    ])

    # === Demo 1: Advantage Estimation ===
    print("\n--- Demo 1: Advantage Estimation ---")
    estimator = AdvantageEstimator(config)

    # Simulate rewards: 4 prompts × 8 responses each
    rewards = np.array([
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],  # 3/8 correct
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 3/8 correct
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 1/8 correct
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 8/8 correct
    ])

    advantages = estimator.compute_advantages(rewards)
    print(f"Rewards shape: {rewards.shape}")
    print(f"Advantages shape: {advantages.shape}")
    print(f"Advantages for prompt 0: {advantages[0].round(3)}")
    print(f"Advantages for prompt 3 (all correct): {advantages[3].round(3)}")

    # === Demo 2: GRPO Loss ===
    print("\n--- Demo 2: GRPO Loss Computation ---")
    trainer = AdvancedGRPOTrainer(
        policy_model=None,  # Placeholder
        reference_model=None,
        reward_fn=composite,
        config=config,
    )

    logprobs = np.random.randn(4, 8) * 0.5
    ref_logprobs = np.random.randn(4, 8) * 0.5
    loss, info = trainer.compute_grpo_loss(logprobs, ref_logprobs, advantages)

    print(f"Loss: {loss:.4f}")
    print(f"Surrogate loss: {info['surrogate_loss']:.4f}")
    print(f"KL loss: {info['kl_loss']:.4f}")
    print(f"Mean ratio: {info['mean_ratio']:.4f}")
    print(f"Clip fraction: {info['clip_fraction']:.4f}")

    # === Demo 3: Reward Functions ===
    print("\n--- Demo 3: Reward Functions ---")

    # Math reward
    response_correct = "Let me solve this step by step. 2 + 2 = 4. The answer is \\boxed{4}"
    response_wrong = "The answer is 5."

    score_correct = math_reward("What is 2+2?", response_correct)
    score_wrong = math_reward("What is 2+2?", response_wrong)
    print(f"Math reward (correct): {score_correct:.2f}")
    print(f"Math reward (wrong): {score_wrong:.2f}")

    # Format reward
    response_formatted = "<think>\nLet me think about this...\n2+2=4\n</think>\n<answer>4</answer>"
    score_format = format_reward("What is 2+2?", response_formatted)
    print(f"Format reward (formatted): {score_format:.2f}")

    # === Demo 4: Training Summary ===
    print("\n--- Demo 4: Training Summary ---")
    summary = trainer.get_training_summary()
    print(f"Status: {summary}")

    print("\n" + "=" * 70)
    print("All GRPO demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_grpo()


################################################################################
# REFERENCES
################################################################################

# [1] DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability
#     in LLMs via Reinforcement Learning. arXiv:2501.12948.
#
# [2] Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical
#     Reasoning in Open Language Models. arXiv:2402.03300.
#
# [3] Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
#     arXiv:1707.06347.
#
# [4] Ouyang, L., et al. (2022). Training language models to follow instructions
#     with human feedback. arXiv:2203.02155.

################################################################################
