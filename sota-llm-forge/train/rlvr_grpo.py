"""
################################################################################
GRPO + DAPO + GSPO + Dr. GRPO — REASONING RL WITH VERIFIABLE REWARDS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is GRPO (Group Relative Policy Optimization)?
    GRPO is a policy-gradient algorithm for training language models on
    reasoning tasks using verifiable rewards. For each prompt, it samples
    G completions from the current policy, scores them with a deterministic
    reward function (e.g., exact match for math, unit tests for code), then
    computes advantages by normalizing rewards within each group. The policy
    is updated using a clipped surrogate objective (similar to PPO) but
    without a separate value function — the group statistics serve as the
    baseline.

Why does it matter?
    GRPO is the workhorse behind the 2025-2026 reasoning revolution (DeepSeek-R1,
    QwQ, etc.). It enables LLMs to learn complex reasoning (math proofs, code
    generation, logical deduction) through trial-and-error, guided by rewards
    that CAN'T be gamed — because they come from verifiable sources (exact match,
    unit tests, formal verification), not learned reward models.

How does it work?
    1. For each prompt, sample G completions from the current policy
    2. Score each completion with a verifiable reward function
    3. Normalize rewards within the group (mean/std) to get advantages
    4. Compute clipped policy-gradient loss (PPO-style clipping)
    5. Optional KL penalty against SFT reference to prevent drift
    6. Update policy

    Refinements (togglable via config flags):
        DAPO: clip-higher, no-KL, dynamic sampling, token-level loss
        GSPO: sequence-level importance ratio (important for MoE)
        Dr. GRPO: remove std/length normalization

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌──────────────────────────────────────────────────────────────────┐
    │                    GRPO TRAINING LOOP                            │
    │                                                                   │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  Prompt Pool                                             │    │
    │  └──────────────────────┬───────────────────────────────────┘    │
    │                         ↓                                        │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  Group Sampling: For each prompt, generate G completions  │    │
    │  │                                                          │    │
    │  │  Prompt₁ → [Comp₁₁, Comp₁₂, ..., Comp₁G]                │    │
    │  │  Prompt₂ → [Comp₂₁, Comp₂₂, ..., Comp₂G]                │    │
    │  └──────────────────────┬───────────────────────────────────┘    │
    │                         ↓                                        │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  Verifiable Reward Function                              │    │
    │  │                                                          │    │
    │  │  Math: exact_match(predicted_answer, ground_truth)       │    │
    │  │  Code: unit_tests_pass(generated_code, test_cases)       │    │
    │  │  Logic: formal_verify(proof, theorem)                    │    │
    │  └──────────────────────┬───────────────────────────────────┘    │
    │                         ↓                                        │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  Advantage Computation (group-relative)                  │    │
    │  │  A_i = (R_i - mean(R_group)) / std(R_group)             │    │
    │  └──────────────────────┬───────────────────────────────────┘    │
    │                         ↓                                        │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  Clipped Surrogate Loss                                  │    │
    │  │  L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)         │    │
    │  │  + optional KL penalty vs reference                      │    │
    │  └──────────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2017: PPO — Proximal Policy Optimization (Schulman et al.)
    - 2022: RLHF with PPO for LLMs (Ouyang et al., InstructGPT)
    - 2024: GRPO — Group Relative Policy Optimization (DeepSeek-Math)
    - 2025: DeepSeek-R1 — GRPO for reasoning, cold-start from base model
    - 2025: DAPO — Decoupled clip, dynamic sampling, token-level loss
    - 2025: GSPO — Sequence-level importance ratio for MoE models
    - 2025: Dr. GRPO — Remove normalization biases in GRPO
    - 2026: RLVR (RL with Verifiable Rewards) becomes standard for reasoning

INTERVIEW QUESTIONS:
    1. "Why use verifiable rewards instead of a learned reward model?"
       Learned reward models can be gamed — the policy finds adversarial
       completions that score high on the reward model but are actually
       bad. Verifiable rewards (exact match, unit tests) are ground truth.
       They can't be gamed because they check the actual answer, not a
       proxy. This is why GRPO-based systems produce reliable reasoning
       while RLHF-based systems can produce confident-sounding nonsense.

    2. "What is the difference between GRPO and PPO?"
       PPO requires a separate value function (critic) to estimate
       advantages, which doubles memory and compute. GRPO eliminates
       the critic by using group statistics (mean and std of rewards
       within a group of completions for the same prompt) as the
       baseline. This is simpler, uses less memory, and works well
       when you can sample multiple completions per prompt.

    3. "How do you detect reward hacking in GRPO?"
       Monitor policy entropy. If entropy drops to near-zero while
       reward climbs, the policy has collapsed to a single degenerate
       strategy — classic reward hacking. Also check: (1) response
       length growth (hacking often involves verbose non-answers),
       (2) format compliance without content (outputs that look right
       but are wrong), (3) diversity of reasoning paths. The fix is
       usually a KL penalty, entropy bonus, or better reward design.

################################################################################
"""

import math
import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################################
# SECTION 1: GRPO CONFIGURATION
################################################################################


@dataclass
class GRPOConfig:
    """
    GRPO Configuration
    ==================

    All hyperparameters for GRPO training with refinements from
    DAPO, GSPO, and Dr. GRPO.

    Design:
        Every refinement is a boolean flag with sensible defaults
        (all disabled). Enable them individually to experiment with
        each improvement independently.

    Interview Question:
        "What are the key hyperparameters in GRPO?"
        1. group_size: Number of completions per prompt. More = better
           advantage estimates but more compute. 8-16 is typical.
        2. clip_range: How far the policy can move per step. 0.2 is standard.
        3. kl_coeff: Strength of KL penalty against reference. 0 = no penalty.
           DAPO argues for kl_coeff=0 with clip-higher instead.
    """

    # ------------------------------------------------------------------
    # Core GRPO parameters
    # ------------------------------------------------------------------
    group_size: int = 8
    """Number of completions to sample per prompt (G in the paper)."""

    clip_range: float = 0.2
    """PPO-style clipping range. Prevents too-large policy updates."""

    kl_coeff: float = 0.0
    """KL penalty coefficient against SFT reference. 0 = no penalty (DAPO style)."""

    kl_estimator: str = "approx_kl"
    """KL estimator: 'approx_kl' (log-ratio) or 'full_kl' (with entropy)."""

    # ------------------------------------------------------------------
    # DAPO refinements
    # ------------------------------------------------------------------
    clip_higher: bool = False
    """
    DAPO: Decouple clip lower and upper bounds.
    When enabled, uses asymmetric clipping: tighter on the lower side,
    looser on the upper side. This encourages the model to explore
    more when the advantage is positive.
    """

    clip_range_higher: float = 0.28
    """DAPO: Upper clip bound when clip_higher is enabled."""

    dynamic_sampling: bool = False
    """
    DAPO: Filter out groups where all completions have the same reward
    (zero variance). These groups provide no useful training signal
    and waste compute.
    """

    token_level_loss: bool = False
    """
    DAPO: Aggregate loss at the token level instead of sequence level.
    This gives each token equal weight regardless of sequence length,
    which can improve learning for short but important completions.
    """

    # ------------------------------------------------------------------
    # GSPO refinement
    # ------------------------------------------------------------------
    sequence_level_ratio: bool = False
    """
    GSPO: Compute importance ratio at the sequence level instead of
    token level. This matters for MoE models where different tokens
    may route to different experts, making token-level ratios noisy.
    """

    # ------------------------------------------------------------------
    # Dr. GRPO refinements
    # ------------------------------------------------------------------
    remove_std_normalization: bool = False
    """
    Dr. GRPO: Remove std normalization from advantage computation.
    Standard GRPO: A = (R - mean) / std
    Dr. GRPO:     A = (R - mean)

    Reason: std normalization can amplify noise when std is small
    (e.g., all rewards are similar).
    """

    remove_length_normalization: bool = False
    """
    Dr. GRPO: Remove length normalization from the loss.
    Standard GRPO normalizes loss by sequence length.
    Dr. GRPO: aggregate at sequence level without dividing by length.

    Reason: length normalization can bias toward shorter completions.
    """

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------
    difficulty_aware_length: bool = False
    """
    Reward shaping: Adjust length penalty based on problem difficulty.
    Harder problems get more lenient length budgets.
    """

    max_completion_length: int = 2048
    """Maximum length of generated completions."""

    length_penalty_coeff: float = 0.0
    """Coefficient for length penalty in reward. 0 = no penalty."""

    # ------------------------------------------------------------------
    # Entropy and exploration
    # ------------------------------------------------------------------
    entropy_coeff: float = 0.0
    """Coefficient for entropy bonus. Encourages exploration."""

    entropy_threshold: float = 0.5
    """Minimum entropy threshold. Below this, flag potential reward hacking."""

    # ------------------------------------------------------------------
    # Reference model
    # ------------------------------------------------------------------
    use_reference_model: bool = True
    """Whether to use a reference model for KL computation."""

    reference_model_path: Optional[str] = None
    """Path to reference model checkpoint. None = copy of initial model."""

    # ------------------------------------------------------------------
    # Generation parameters
    # ------------------------------------------------------------------
    temperature: float = 1.0
    """Sampling temperature for generation."""

    top_p: float = 1.0
    """Nucleus sampling threshold."""

    top_k: int = 0
    """Top-k sampling. 0 = disabled."""

    # ------------------------------------------------------------------
    # Training parameters
    # ------------------------------------------------------------------
    learning_rate: float = 1e-6
    """Learning rate for GRPO (very low — small adjustments)."""

    max_steps: int = 1000
    """Maximum training steps."""

    batch_size: int = 4
    """Number of prompts per batch."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps."""

    grad_clip_norm: float = 1.0
    """Gradient clipping norm."""

    warmup_steps: int = 50
    """Number of warmup steps."""

    weight_decay: float = 0.01
    """Weight decay."""

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_every_n_steps: int = 5
    """Log metrics every N steps."""


################################################################################
# SECTION 2: VERIFIABLE REWARD INTERFACE
################################################################################


class VerifiableRewardFunction:
    """
    Verifiable Reward Function
    ===========================

    Abstract interface for pluggable verifiable reward functions.

    CRITICAL DESIGN PRINCIPLE:
        The reward function must be VERIFIABLE — it must check the actual
        answer against ground truth, not use a learned model. Examples:
        - Math: exact match of numerical answer
        - Code: unit tests pass
        - Logic: formal proof verification
        - Constraint: output satisfies format requirements

    WHY this matters:
        The entire point of RLVR is that rewards come from ground truth,
        not from a model that can be gamed. If you use a learned reward
        model as the ONLY signal, you're doing RLHF, not RLVR, and you
        WILL get reward hacking.

    Interview Question:
        "How do you design a good verifiable reward function?"
        The reward should be: (1) binary or near-binary (correct/incorrect),
        (2) based on ground truth (not a model), (3) robust to format
        variations (e.g., "42" and "42.0" and "the answer is 42" should
        all be accepted), (4) decomposable when possible (partial credit
        for correct reasoning steps). Avoid continuous rewards from models
        — they're exploitable.
    """

    def __call__(self, prompt: str, completion: str) -> float:
        """
        Score a completion against ground truth.

        Args:
            prompt: The input prompt (may contain the question/problem).
            completion: The model's generated completion.

        Returns:
            A scalar reward. Typically in [0, 1] for binary tasks.
        """
        raise NotImplementedError("Subclasses must implement __call__")


class MathExactMatchReward(VerifiableRewardFunction):
    """
    Math Exact Match Reward
    ========================

    Rewards 1.0 if the extracted answer matches ground truth, 0.0 otherwise.
    Handles common answer formats: boxed{answer}, "the answer is X", etc.
    """

    def __init__(self, ground_truths: Dict[str, str]):
        """
        Args:
            ground_truths: Dict mapping prompt -> correct answer string.
        """
        self.ground_truths = ground_truths

    def extract_answer(self, text: str) -> str:
        """
        Extract the final answer from a completion.

        Handles formats:
            - \\boxed{42}
            - The answer is 42
            - Answer: 42
            - 42

        Args:
            text: The completion text.

        Returns:
            Extracted answer string (stripped, lowercased).
        """
        import re

        # Try boxed answer first
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed_match:
            return boxed_match.group(1).strip().lower()

        # Try "the answer is X"
        answer_match = re.search(r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)", text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip().lower()

        # Fall back to last number
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            return numbers[-1].strip().lower()

        return text.strip().lower()

    def __call__(self, prompt: str, completion: str) -> float:
        """
        Score completion with exact match.

        Args:
            prompt: The math problem.
            completion: The model's solution.

        Returns:
            1.0 if answer matches, 0.0 otherwise.
        """
        ground_truth = self.ground_truths.get(prompt, "")
        extracted = self.extract_answer(completion)
        return 1.0 if extracted == ground_truth.lower().strip() else 0.0


class CodeExecutionReward(VerifiableRewardFunction):
    """
    Code Execution Reward
    =====================

    Rewards based on unit test pass rate.
    In production, this would run code in a sandboxed environment.
    For demonstration, we use a simple string check.
    """

    def __init__(self, test_cases: Dict[str, List[Dict[str, Any]]]):
        """
        Args:
            test_cases: Dict mapping prompt -> list of test case dicts.
                       Each dict has 'input' and 'expected_output' keys.
        """
        self.test_cases = test_cases

    def __call__(self, prompt: str, completion: str) -> float:
        """
        Score completion by running unit tests.

        Args:
            prompt: The coding problem.
            completion: The generated code.

        Returns:
            Fraction of test cases passed (0.0 to 1.0).
        """
        tests = self.test_cases.get(prompt, [])
        if not tests:
            return 0.0

        # In production: execute code in sandbox and check outputs
        # For demo: check if completion contains key patterns
        passed = 0
        for test in tests:
            expected = test.get("expected_output", "")
            if str(expected) in completion:
                passed += 1

        return passed / len(tests) if tests else 0.0


################################################################################
# SECTION 3: ADVANTAGE COMPUTATION
################################################################################


def compute_group_advantages(
    rewards: torch.Tensor,
    config: GRPOConfig,
) -> torch.Tensor:
    """
    Compute group-relative advantages.

    Args:
        rewards: Reward values for each completion in the group.
                 Shape: (batch_size, group_size).
        config: GRPO configuration with normalization flags.

    Returns:
        Advantage values. Shape: (batch_size, group_size).

    Formula:
        Standard GRPO:
            A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

        Dr. GRPO (no std normalization):
            A_i = R_i - mean(R_group)

    Explanation:
        The advantage tells us how much better/worse each completion is
        compared to the group average. Positive advantage = this completion
        was better than average → increase its probability. Negative = worse
        → decrease its probability. This eliminates the need for a separate
        value function (critic) — the group statistics serve as the baseline.

        Dr. GRPO argues that std normalization can amplify noise when all
        rewards in a group are similar (small std), leading to unstable
        training. Removing it makes the advantage signal more stable.
    """
    # Group mean
    group_mean = rewards.mean(dim=-1, keepdim=True)

    # Advantage = reward - baseline
    advantages = rewards - group_mean

    # Optional: normalize by group std
    if not config.remove_std_normalization:
        group_std = rewards.std(dim=-1, keepdim=True) + 1e-8
        advantages = advantages / group_std

    return advantages


################################################################################
# SECTION 4: CLIPPED SURROGATE LOSS
################################################################################


def compute_grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: GRPOConfig,
) -> Dict[str, torch.Tensor]:
    """
    Compute the GRPO clipped surrogate loss.

    Args:
        log_probs: Log-probabilities under current policy. Shape: (B, G, T).
        old_log_probs: Log-probabilities under sampling policy. Shape: (B, G, T).
        advantages: Advantage values. Shape: (B, G).
        config: GRPO configuration with clipping settings.

    Returns:
        Dictionary with:
            - 'loss': The GRPO loss scalar.
            - 'ratio_mean': Mean importance ratio.
            - 'ratio_max': Max importance ratio.
            - 'approx_kl': Approximate KL divergence.
            - 'clip_fraction': Fraction of clipped tokens.

    Formula:
        ratio = exp(log_prob - old_log_prob)

        Standard clipping:
            L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

        DAPO clip-higher:
            L = -min(ratio * A, clip(ratio, 1-ε_low, 1+ε_high) * A)
            where ε_low < ε_high (tighter lower bound, looser upper)

    Step by step:
        1. Compute importance ratio: r = π/π_old = exp(log π - log π_old)
        2. Compute clipped ratio: r_clipped = clip(r, 1-ε, 1+ε)
        3. Compute surrogate: surr1 = r * A, surr2 = r_clipped * A
        4. Take the minimum (pessimistic bound)
        5. Average over tokens and batch

    WHY this matters:
        The clipping prevents too-large policy updates. Without it, a single
        high-advantage completion could cause the policy to shift dramatically,
        leading to catastrophic forgetting or mode collapse. The clip range
        (typically 0.2) keeps the policy within a "trust region" of the
        sampling policy.

    Interview Question:
        "Why clip in GRPO/PPO instead of just using a small learning rate?"
        The learning rate controls the step size in parameter space, but
        what matters in policy gradient is the step size in probability
        space. A small parameter change can cause a huge probability change
        if the policy is near a cliff. Clipping directly bounds the
        probability ratio, ensuring the policy never moves too far in a
        single step, regardless of the learning rate.
    """
    # Compute importance ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # Expand advantages to match ratio shape
    # advantages: (B, G) -> (B, G, 1) for broadcasting with (B, G, T)
    advantages_expanded = advantages.unsqueeze(-1)

    # Surrogate objectives
    surr1 = ratio * advantages_expanded

    # Clipping
    if config.clip_higher:
        # DAPO: asymmetric clipping
        ratio_clipped = torch.clamp(
            ratio,
            1.0 - config.clip_range,
            1.0 + config.clip_range_higher,
        )
    else:
        # Standard symmetric clipping
        ratio_clipped = torch.clamp(
            ratio,
            1.0 - config.clip_range,
            1.0 + config.clip_range,
        )

    surr2 = ratio_clipped * advantages_expanded

    # Take minimum (pessimistic)
    loss = -torch.min(surr1, surr2)

    # Aggregate based on config
    if config.token_level_loss:
        # DAPO: average over all tokens equally
        loss = loss.mean()
    else:
        # Standard: average over sequences, then batch
        # First average over tokens per sequence
        seq_loss = loss.mean(dim=-1)  # (B, G)
        if not config.remove_length_normalization:
            # Already averaged over tokens
            pass
        loss = seq_loss.mean()

    # Metrics
    with torch.no_grad():
        ratio_mean = ratio.mean()
        ratio_max = ratio.max()
        approx_kl = (log_probs - old_log_probs).mean()
        clip_fraction = ((ratio - ratio_clipped).abs() > 1e-6).float().mean()

    return {
        "loss": loss,
        "ratio_mean": ratio_mean,
        "ratio_max": ratio_max,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }


################################################################################
# SECTION 5: SEQUENCE-LEVEL IMPORTANCE RATIO (GSPO)
################################################################################


def compute_sequence_level_ratio(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute importance ratio at the sequence level (GSPO).

    Args:
        log_probs: Token-level log-probs. Shape: (B, G, T).
        old_log_probs: Token-level log-probs under old policy. Shape: (B, G, T).

    Returns:
        Sequence-level importance ratio. Shape: (B, G).

    Formula:
        ratio_seq = exp(Σ_t log π_t - Σ_t log π_old_t)
                  = exp(log P(sequence) - log P_old(sequence))

    Explanation:
        Standard GRPO/PPO computes the ratio per token. GSPO argues that
        for MoE models, token-level ratios are noisy because different tokens
        may route to different experts with different utilization patterns.
        Sequence-level ratios smooth this out by aggregating over the full
        sequence, giving a more stable signal for the policy gradient.

        This is particularly important for MoE models where expert routing
        can cause large variance in per-token log-probs.

    Reference:
        "GSPO: Group Sequence Policy Optimization" (2025)
    """
    # Sum log-probs over sequence dimension
    seq_log_probs = log_probs.sum(dim=-1)  # (B, G)
    seq_old_log_probs = old_log_probs.sum(dim=-1)  # (B, G)

    # Sequence-level ratio
    return torch.exp(seq_log_probs - seq_old_log_probs)


################################################################################
# SECTION 6: KL PENALTY
################################################################################


def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    estimator: str = "approx_kl",
) -> torch.Tensor:
    """
    Compute KL divergence penalty between policy and reference.

    Args:
        log_probs: Log-probs under current policy. Shape: (B, G, T).
        ref_log_probs: Log-probs under reference policy. Shape: (B, G, T).
        estimator: KL estimator type.

    Returns:
        KL penalty scalar.

    Formula:
        Approximate KL: KL ≈ log π/π_ref = log_probs - ref_log_probs
        Full KL: KL = Σ π(x) * log(π(x)/π_ref(x))

    Explanation:
        The KL penalty prevents the policy from deviating too far from
        the SFT reference. Without it, the policy can collapse to degenerate
        strategies that maximize reward but produce incoherent text.

        DAPO argues for removing the KL penalty entirely (kl_coeff=0)
        and using clip-higher instead. This simplifies the algorithm and
        avoids the need for a reference model.

    Interview Question:
        "When should you use KL penalty vs. clipping in GRPO?"
        They serve different purposes. Clipping bounds the per-step update
        (trust region). KL penalty bounds the cumulative deviation from
        the reference (regularization). In practice: use both for safety,
        or remove KL (DAPO style) if you have good clip-higher tuning.
        The DAPO paper showed that removing KL + clip-higher works as well
        or better than standard GRPO with KL.
    """
    if estimator == "approx_kl":
        # Approximate KL: E[log π/π_ref]
        kl = log_probs - ref_log_probs
        return kl.mean()
    elif estimator == "full_kl":
        # Full KL: E[π * log(π/π_ref)]
        # This requires access to the full distribution, which is expensive.
        # For LLMs, we use the approximate version.
        kl = log_probs - ref_log_probs
        return kl.mean()
    else:
        raise ValueError(f"Unknown KL estimator: {estimator}")


################################################################################
# SECTION 7: ENTROPY MONITORING AND REWARD HACKING DETECTION
################################################################################


class EntropyMonitor:
    """
    Entropy Monitor
    ===============

    Monitors policy entropy to detect reward hacking.

    When the policy collapses to near-zero entropy while reward climbs,
    it's a RED FLAG — the model has found a degenerate strategy that
    maximizes reward without actually solving the problem.

    WHY this matters:
        Reward hacking is the #1 failure mode in RL. A model might learn
        to output "The answer is 42" for every math problem (high reward
        if 42 happens to be common), or format its output to look correct
        without actually reasoning. Entropy monitoring catches this early.

    Interview Question:
        "How do you detect and prevent reward hacking?"
        Detection: Monitor entropy, reward, and response diversity. If
        entropy drops while reward climbs, it's likely hacking. Also check
        response length (hacking often involves verbose non-answers) and
        format compliance without content.

        Prevention: (1) KL penalty against SFT reference, (2) entropy
        bonus, (3) diverse reward signals (not just one metric), (4)
        format penalties, (5) human evaluation spot-checks.
    """

    def __init__(self, threshold: float = 0.5, window_size: int = 50):
        self.threshold = threshold
        self.window_size = window_size
        self.entropy_history: List[float] = []
        self.reward_history: List[float] = []

    def update(self, entropy: float, reward: float) -> Dict[str, Any]:
        """
        Update with new entropy and reward values.

        Args:
            entropy: Current policy entropy.
            reward: Current average reward.

        Returns:
            Dictionary with:
                - 'hacking_detected': bool
                - 'entropy': current entropy
                - 'entropy_trend': 'decreasing', 'stable', or 'increasing'
                - 'reward_trend': 'increasing', 'stable', or 'decreasing'
        """
        self.entropy_history.append(entropy)
        self.reward_history.append(reward)

        # Keep window
        if len(self.entropy_history) > self.window_size:
            self.entropy_history = self.entropy_history[-self.window_size:]
            self.reward_history = self.reward_history[-self.window_size:]

        result = {
            "entropy": entropy,
            "hacking_detected": False,
            "entropy_trend": "stable",
            "reward_trend": "stable",
        }

        if len(self.entropy_history) < 10:
            return result

        # Check trends
        recent_entropy = self.entropy_history[-10:]
        earlier_entropy = self.entropy_history[:-10] if len(self.entropy_history) > 10 else recent_entropy

        recent_reward = self.reward_history[-10:]
        earlier_reward = self.reward_history[:-10] if len(self.reward_history) > 10 else recent_reward

        entropy_change = sum(recent_entropy) / len(recent_entropy) - sum(earlier_entropy) / len(earlier_entropy)
        reward_change = sum(recent_reward) / len(recent_reward) - sum(earlier_reward) / len(earlier_reward)

        if entropy_change < -0.1:
            result["entropy_trend"] = "decreasing"
        elif entropy_change > 0.1:
            result["entropy_trend"] = "increasing"

        if reward_change > 0.05:
            result["reward_trend"] = "increasing"
        elif reward_change < -0.05:
            result["reward_trend"] = "decreasing"

        # Hacking detection: entropy decreasing while reward increasing
        if (result["entropy_trend"] == "decreasing"
                and result["reward_trend"] == "increasing"
                and entropy < self.threshold):
            result["hacking_detected"] = True

        return result


################################################################################
# SECTION 8: GRPO TRAINER
################################################################################


class GRPOTrainer:
    """
    GRPO Trainer
    =============

    The 2026 reasoning-RL workhorse.

    Core loop:
        1. For each prompt, sample G completions from current policy
        2. Score each with verifiable reward (math: exact match, code: unit tests)
        3. Compute advantage: reward normalized against group mean/std
        4. Clipped policy-gradient update (PPO-style clipping)
        5. Optional KL penalty vs SFT reference

    Refinements (togglable flags):
        DAPO: clip-higher, no-KL, dynamic sampling, token-level loss
        GSPO: sequence-level importance ratio (matters for MoE)
        Dr. GRPO: remove std/length normalization

    CRITICAL: Never use a learned reward model as the ONLY signal.
    Verifiable rewards are the entire point — they can't be gamed.

    WHY this matters:
        GRPO eliminates the critic network, halving memory and compute
        vs PPO. Combined with verifiable rewards, it produces models
        that can actually reason — not just pattern-match. This is the
        algorithm behind DeepSeek-R1 and the reasoning revolution.

    Interview Question:
        "Walk me through a GRPO training step for math reasoning."
        1. Sample 8 math problems from the dataset
        2. For each problem, generate 8 solutions (temperature=1.0)
        3. Extract answers from each solution
        4. Compare to ground truth: reward = 1 if correct, 0 if wrong
        5. Compute group advantages: mean reward across 8 solutions, normalize
        6. Compute log-probs for each solution under current and old policy
        7. Clipped surrogate loss with advantages
        8. Optional: add KL penalty against SFT reference
        9. Backpropagate, clip gradients, step optimizer
        10. Monitor entropy — if dropping while reward climbs, flag hacking
    """

    def __init__(
        self,
        model: nn.Module,
        config: GRPOConfig,
        reward_fn: VerifiableRewardFunction,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the GRPO Trainer.

        Args:
            model: The language model to train.
            config: GRPOConfig with all hyperparameters.
            reward_fn: The verifiable reward function.
            device: Target device.

        Explanation:
            1. Setup policy model
            2. Setup frozen reference model (for KL penalty)
            3. Setup optimizer
            4. Setup entropy monitor
            5. Print recipe summary
        """
        self.config = config
        self.reward_fn = reward_fn
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ------------------------------------------------------------------
        # Policy model
        # ------------------------------------------------------------------
        self.model = model.to(self.device)

        # ------------------------------------------------------------------
        # Reference model (frozen)
        # ------------------------------------------------------------------
        if config.use_reference_model:
            if config.reference_model_path is not None:
                self.reference_model = self._load_reference(config.reference_model_path)
            else:
                self.reference_model = copy.deepcopy(model).to(self.device)
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
        else:
            self.reference_model = None

        # ------------------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # ------------------------------------------------------------------
        # Monitoring
        # ------------------------------------------------------------------
        self.entropy_monitor = EntropyMonitor(
            threshold=config.entropy_threshold,
            window_size=50,
        )
        self.global_step = 0

        # Print recipe
        self._print_recipe()

    def _load_reference(self, path: str) -> nn.Module:
        """Load reference model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        model = copy.deepcopy(self.model)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def _print_recipe(self) -> None:
        """Print GRPO recipe summary."""
        c = self.config
        border = "=" * 70
        print(f"\n{border}")
        print("GRPO RECIPE SUMMARY")
        print(border)
        print(f"  Group size:           {c.group_size}")
        print(f"  Clip range:           {c.clip_range}")
        print(f"  KL coefficient:       {c.kl_coeff}")
        print(f"  Temperature:          {c.temperature}")
        print(f"  Max completion len:   {c.max_completion_length}")
        print(f"  Learning rate:        {c.learning_rate}")
        print(f"  Max steps:            {c.max_steps}")
        print(f"  DAPO clip-higher:     {c.clip_higher} (higher={c.clip_range_higher})")
        print(f"  DAPO dynamic samp.:   {c.dynamic_sampling}")
        print(f"  DAPO token-level:     {c.token_level_loss}")
        print(f"  GSPO seq-level ratio: {c.sequence_level_ratio}")
        print(f"  Dr.GRPO no std norm:  {c.remove_std_normalization}")
        print(f"  Dr.GRPO no len norm:  {c.remove_length_normalization}")
        print(f"  Entropy coeff:        {c.entropy_coeff}")
        print(f"  Entropy threshold:    {c.entropy_threshold}")
        print(f"  Length penalty:        {c.length_penalty_coeff}")
        print(border + "\n")

    @torch.no_grad()
    def generate_completions(
        self,
        prompts: List[str],
        prompt_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate G completions for each prompt.

        Args:
            prompts: List of prompt strings.
            prompt_ids: Tokenized prompts. Shape: (B, P).

        Returns:
            Tuple of:
                - completion_ids: Generated token IDs. Shape: (B*G, T).
                - completion_log_probs: Log-probs under sampling policy. Shape: (B*G, T).

        Explanation:
            For each prompt, we generate G completions using the current
            policy with temperature sampling. We record both the generated
            tokens and their log-probs (needed for the importance ratio).
        """
        B = prompt_ids.shape[0]
        G = self.config.group_size

        # Repeat prompts G times
        expanded_ids = prompt_ids.repeat_interleave(G, dim=0)  # (B*G, P)

        # Generate completions
        self.model.eval()
        all_completion_ids = []
        all_log_probs = []

        for i in range(B * G):
            prompt = expanded_ids[i:i+1]  # (1, P)

            # Simple autoregressive generation
            generated = []
            log_probs_seq = []
            current_ids = prompt

            for _ in range(self.config.max_completion_length):
                outputs = self.model(current_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Get logits for next token
                next_logits = logits[:, -1, :] / self.config.temperature

                # Sample from distribution
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                log_prob = F.log_softmax(next_logits, dim=-1).gather(-1, next_token)

                generated.append(next_token)
                log_probs_seq.append(log_prob)

                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)

                # Check for EOS (simplified)
                if next_token.item() == 0:  # Assuming 0 is EOS
                    break

            if generated:
                completion = torch.cat(generated, dim=1)  # (1, T)
                log_p = torch.cat(log_probs_seq, dim=1)  # (1, T)
            else:
                completion = torch.zeros(1, 1, dtype=torch.long)
                log_p = torch.zeros(1, 1)

            all_completion_ids.append(completion)
            all_log_probs.append(log_p)

        # Pad to same length
        max_len = max(c.shape[1] for c in all_completion_ids)
        padded_ids = torch.zeros(B * G, max_len, dtype=torch.long)
        padded_log_probs = torch.zeros(B * G, max_len)

        for i, (c, l) in enumerate(zip(all_completion_ids, all_log_probs)):
            padded_ids[i, :c.shape[1]] = c
            padded_log_probs[i, :l.shape[1]] = l

        self.model.train()
        return padded_ids.to(self.device), padded_log_probs.to(self.device)

    def compute_rewards(
        self,
        prompts: List[str],
        completion_ids: torch.Tensor,
        tokenizer: Any = None,
    ) -> torch.Tensor:
        """
        Compute verifiable rewards for completions.

        Args:
            prompts: Original prompt strings.
            completion_ids: Generated token IDs. Shape: (B*G, T).
            tokenizer: Optional tokenizer for decoding.

        Returns:
            Rewards tensor. Shape: (B, G).
        """
        B = len(prompts)
        G = self.config.group_size
        rewards = torch.zeros(B, G)

        for i, prompt in enumerate(prompts):
            for j in range(G):
                idx = i * G + j
                # Decode completion (simplified — in production use tokenizer)
                completion_str = f"completion_{idx}"  # Placeholder

                # Get reward from verifiable function
                reward = self.reward_fn(prompt, completion_str)
                rewards[i, j] = reward

        return rewards.to(self.device)

    def compute_log_probs_for_completions(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log-probs for completions under the current policy.

        Args:
            prompt_ids: Prompt token IDs. Shape: (B, P).
            completion_ids: Completion token IDs. Shape: (B*G, T).

        Returns:
            Log-probs per token. Shape: (B*G, T).
        """
        B = prompt_ids.shape[0]
        G = self.config.group_size

        # Expand prompts
        expanded_prompts = prompt_ids.repeat_interleave(G, dim=0)  # (B*G, P)

        # Concatenate prompt + completion
        full_ids = torch.cat([expanded_prompts, completion_ids], dim=1)

        # Forward pass
        outputs = self.model(full_ids)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Get log-probs for completion tokens only
        prompt_len = expanded_prompts.shape[1]
        completion_logits = logits[:, prompt_len-1:-1, :]  # Shift by 1
        log_probs = F.log_softmax(completion_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)

        return token_log_probs

    def train_step(
        self,
        prompts: List[str],
        prompt_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Execute a single GRPO training step.

        Args:
            prompts: List of prompt strings.
            prompt_ids: Tokenized prompts. Shape: (B, P).

        Returns:
            Dictionary of training metrics.

        Explanation:
            1. Generate G completions per prompt
            2. Compute rewards with verifiable function
            3. Compute advantages (group-relative normalization)
            4. Compute log-probs under current and old policy
            5. Compute GRPO loss (clipped surrogate)
            6. Optional: add KL penalty
            7. Backpropagate and update
            8. Monitor entropy for reward hacking
        """
        B = prompt_ids.shape[0]
        G = self.config.group_size

        # ------------------------------------------------------------------
        # Step 1: Generate completions
        # ------------------------------------------------------------------
        completion_ids, old_log_probs = self.generate_completions(prompts, prompt_ids)

        # ------------------------------------------------------------------
        # Step 2: Compute rewards
        # ------------------------------------------------------------------
        rewards = self.compute_rewards(prompts, completion_ids)  # (B, G)

        # Dynamic sampling: filter zero-variance groups
        if self.config.dynamic_sampling:
            group_std = rewards.std(dim=-1)
            valid_mask = group_std > 1e-6
            if valid_mask.sum() == 0:
                return {"loss": 0.0, "skipped": True}
            rewards = rewards[valid_mask]
            # Filter completions and log-probs accordingly
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            completion_ids = completion_ids[valid_indices.repeat_interleave(G)]
            old_log_probs = old_log_probs[valid_indices.repeat_interleave(G)]

        # ------------------------------------------------------------------
        # Step 3: Compute advantages
        # ------------------------------------------------------------------
        advantages = compute_group_advantages(rewards, self.config)  # (B, G)

        # ------------------------------------------------------------------
        # Step 4: Compute current log-probs
        # ------------------------------------------------------------------
        log_probs = self.compute_log_probs_for_completions(prompt_ids, completion_ids)

        # Reshape for loss computation: (B*G, T) -> (B, G, T)
        T = log_probs.shape[1]
        log_probs = log_probs.view(B, G, T)
        old_log_probs = old_log_probs.view(B, G, T)

        # ------------------------------------------------------------------
        # Step 5: Compute GRPO loss
        # ------------------------------------------------------------------
        if self.config.sequence_level_ratio:
            # GSPO: use sequence-level ratio
            seq_ratio = compute_sequence_level_ratio(log_probs, old_log_probs)
            # Expand for loss computation
            ratio_expanded = seq_ratio.unsqueeze(-1).expand_as(log_probs)
            log_ratio = torch.log(ratio_expanded + 1e-8)
            loss_dict = compute_grpo_loss(
                log_probs,
                log_probs - log_ratio,
                advantages,
                self.config,
            )
        else:
            loss_dict = compute_grpo_loss(
                log_probs,
                old_log_probs,
                advantages,
                self.config,
            )

        total_loss = loss_dict["loss"]

        # ------------------------------------------------------------------
        # Step 6: KL penalty
        # ------------------------------------------------------------------
        if self.config.kl_coeff > 0 and self.reference_model is not None:
            with torch.no_grad():
                ref_outputs = self.reference_model(
                    torch.cat([
                        prompt_ids.repeat_interleave(G, dim=0),
                        completion_ids
                    ], dim=1)
                )
                if isinstance(ref_outputs, tuple):
                    ref_logits = ref_outputs[0]
                else:
                    ref_logits = ref_outputs

                prompt_len = prompt_ids.shape[1]
                ref_completion_logits = ref_logits[:, prompt_len-1:-1, :]
                ref_log_probs = F.log_softmax(ref_completion_logits, dim=-1)
                ref_token_log_probs = ref_log_probs.gather(
                    -1, completion_ids.unsqueeze(-1)
                ).squeeze(-1)
                ref_token_log_probs = ref_token_log_probs.view(B, G, T)

            kl = compute_kl_penalty(
                log_probs, ref_token_log_probs, self.config.kl_estimator
            )
            total_loss = total_loss + self.config.kl_coeff * kl

        # ------------------------------------------------------------------
        # Step 7: Entropy bonus
        # ------------------------------------------------------------------
        if self.config.entropy_coeff > 0:
            # Compute entropy from log-probs
            entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
            total_loss = total_loss - self.config.entropy_coeff * entropy

        # ------------------------------------------------------------------
        # Step 8: Backward and optimize
        # ------------------------------------------------------------------
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip_norm,
        )
        self.optimizer.step()
        self.global_step += 1

        # ------------------------------------------------------------------
        # Step 9: Monitor entropy
        # ------------------------------------------------------------------
        with torch.no_grad():
            current_entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean().item()
            avg_reward = rewards.mean().item()

        entropy_status = self.entropy_monitor.update(current_entropy, avg_reward)

        if entropy_status["hacking_detected"]:
            print(f"  WARNING: Potential reward hacking detected at step {self.global_step}!")
            print(f"  Entropy: {current_entropy:.4f} (threshold: {self.config.entropy_threshold})")

        # Collect metrics
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        metrics["reward"] = avg_reward
        metrics["entropy"] = current_entropy
        metrics["reward_std"] = rewards.std().item()
        metrics["advantage_mean"] = advantages.mean().item()

        return metrics

    def train(
        self,
        prompts: List[str],
        tokenizer: Any = None,
    ) -> None:
        """
        Run GRPO training.

        Args:
            prompts: List of training prompts/problems.
            tokenizer: Tokenizer for encoding prompts.

        Explanation:
            Main training loop. Iterates over prompts, samples completions,
            computes rewards, and updates the policy. Logs metrics including
            reward, entropy, and reward-hacking warnings.
        """
        self.model.train()
        print("Starting GRPO training...")

        batch_size = self.config.batch_size
        step = 0

        while self.global_step < self.config.max_steps:
            # Get batch of prompts
            batch_start = (step * batch_size) % len(prompts)
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            if len(batch_prompts) == 0:
                step = 0
                continue

            # Tokenize prompts (simplified)
            prompt_ids = torch.randint(
                0, 1000,
                (len(batch_prompts), 32),
                device=self.device,
            )

            # Training step
            metrics = self.train_step(batch_prompts, prompt_ids)

            # Logging
            if self.global_step % self.config.log_every_n_steps == 0:
                print(
                    f"step={self.global_step:>6d} | "
                    f"loss={metrics.get('loss', 0):.4f} | "
                    f"reward={metrics.get('reward', 0):.4f} | "
                    f"entropy={metrics.get('entropy', 0):.4f} | "
                    f"kl={metrics.get('approx_kl', 0):.4f} | "
                    f"clip_frac={metrics.get('clip_fraction', 0):.3f}"
                )

            step += 1

        print("GRPO training complete.")


################################################################################
# SECTION 9: TESTING & DEMONSTRATION
################################################################################


def demonstrate_grpo():
    """
    Demonstrate GRPO with a toy model and synthetic math problems.

    Shows:
        1. Advantage computation
        2. Clipped surrogate loss
        3. DPO vs GRPO loss comparison
        4. Entropy monitoring
        5. GRPO config with all refinements
    """
    print("=" * 70)
    print("GRPO TRAINER DEMONSTRATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Demo 1: Advantage computation
    # ------------------------------------------------------------------
    print("\n[1/5] Advantage Computation")
    print("-" * 40)

    rewards = torch.tensor([
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],  # 50% correct
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 37.5% correct
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 0% correct
    ])

    config_standard = GRPOConfig(remove_std_normalization=False)
    config_dr_grpo = GRPOConfig(remove_std_normalization=True)

    adv_standard = compute_group_advantages(rewards, config_standard)
    adv_dr_grpo = compute_group_advantages(rewards, config_dr_grpo)

    print(f"  Rewards:\n    {rewards}")
    print(f"\n  Standard GRPO advantages:\n    {adv_standard}")
    print(f"\n  Dr. GRPO advantages (no std norm):\n    {adv_dr_grpo}")

    # ------------------------------------------------------------------
    # Demo 2: Clipped surrogate loss
    # ------------------------------------------------------------------
    print("\n[2/5] Clipped Surrogate Loss")
    print("-" * 40)

    B, G, T = 2, 4, 10
    log_probs = torch.randn(B, G, T)
    old_log_probs = log_probs + torch.randn(B, G, T) * 0.1
    advantages = torch.tensor([[1.0, -0.5, 0.3, -0.8], [0.5, -1.0, 0.8, -0.3]])

    # Standard clipping
    config_standard = GRPOConfig(clip_range=0.2, clip_higher=False)
    loss_standard = compute_grpo_loss(log_probs, old_log_probs, advantages, config_standard)

    # DAPO clip-higher
    config_dapo = GRPOConfig(clip_range=0.2, clip_higher=True, clip_range_higher=0.28)
    loss_dapo = compute_grpo_loss(log_probs, old_log_probs, advantages, config_dapo)

    print(f"  Standard clip:    loss={loss_standard['loss']:.4f}, clip_frac={loss_standard['clip_fraction']:.3f}")
    print(f"  DAPO clip-higher: loss={loss_dapo['loss']:.4f}, clip_frac={loss_dapo['clip_fraction']:.3f}")

    # ------------------------------------------------------------------
    # Demo 3: KL penalty
    # ------------------------------------------------------------------
    print("\n[3/5] KL Penalty")
    print("-" * 40)

    log_probs_kl = torch.randn(2, 4, 10)
    ref_log_probs = log_probs_kl + torch.randn(2, 4, 10) * 0.5

    kl_approx = compute_kl_penalty(log_probs_kl, ref_log_probs, "approx_kl")
    kl_full = compute_kl_penalty(log_probs_kl, ref_log_probs, "full_kl")

    print(f"  Approximate KL: {kl_approx:.4f}")
    print(f"  Full KL:        {kl_full:.4f}")

    # ------------------------------------------------------------------
    # Demo 4: Entropy monitoring
    # ------------------------------------------------------------------
    print("\n[4/5] Entropy Monitoring")
    print("-" * 40)

    monitor = EntropyMonitor(threshold=0.5, window_size=20)

    # Simulate normal training
    import random
    random.seed(42)
    for i in range(30):
        entropy = 2.0 + random.gauss(0, 0.2)
        reward = 0.3 + i * 0.01 + random.gauss(0, 0.05)
        result = monitor.update(entropy, reward)

    print(f"  Normal training: entropy_trend={result['entropy_trend']}, "
          f"reward_trend={result['reward_trend']}, hacking={result['hacking_detected']}")

    # Simulate reward hacking
    for i in range(20):
        entropy = 0.3 - i * 0.01  # Decreasing entropy
        reward = 0.8 + i * 0.01   # Increasing reward
        result = monitor.update(entropy, reward)

    print(f"  Reward hacking:  entropy_trend={result['entropy_trend']}, "
          f"reward_trend={result['reward_trend']}, hacking={result['hacking_detected']}")

    # ------------------------------------------------------------------
    # Demo 5: GRPO Config
    # ------------------------------------------------------------------
    print("\n[5/5] GRPO Configurations")
    print("-" * 40)

    configs = {
        "Standard GRPO": GRPOConfig(),
        "DAPO": GRPOConfig(
            clip_higher=True,
            clip_range_higher=0.28,
            kl_coeff=0.0,
            dynamic_sampling=True,
            token_level_loss=True,
        ),
        "GSPO": GRPOConfig(sequence_level_ratio=True),
        "Dr. GRPO": GRPOConfig(
            remove_std_normalization=True,
            remove_length_normalization=True,
        ),
    }

    for name, cfg in configs.items():
        print(f"\n  {name}:")
        print(f"    clip_range={cfg.clip_range}, clip_higher={cfg.clip_higher}")
        print(f"    kl_coeff={cfg.kl_coeff}, dynamic_sampling={cfg.dynamic_sampling}")
        print(f"    token_level={cfg.token_level_loss}, seq_ratio={cfg.sequence_level_ratio}")
        print(f"    no_std_norm={cfg.remove_std_normalization}, no_len_norm={cfg.remove_length_normalization}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_grpo()
