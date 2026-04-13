"""
################################################################################
DIRECT PREFERENCE OPTIMIZATION (DPO) — PREFERENCE LEARNING WITHOUT RL
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Direct Preference Optimization (DPO)?
    DPO is an algorithm for training language models on human preference
    data without needing a separate reward model or RL training loop.
    Given (prompt, chosen_response, rejected_response) triples, DPO
    directly optimizes the policy to increase the probability of chosen
    responses relative to rejected ones, using a simple classification
    loss derived from the Bradley-Terry preference model.

Why does it matter?
    Traditional RLHF (Reinforcement Learning from Human Feedback) requires
    training a separate reward model, then using PPO to optimize against
    it — a complex, unstable, and expensive pipeline. DPO collapses this
    into a single supervised-style training step. It is simpler to
    implement, more stable to train, and empirically competitive with
    RLHF on benchmarks. This made preference tuning accessible to teams
    without RL infrastructure.

How does it work?
    1. Load a frozen reference model (copy of the SFT model)
    2. For each (prompt, chosen, rejected) triple:
        a. Compute log-probs of chosen/rejected under current policy
        b. Compute log-probs of chosen/rejected under frozen reference
        c. Compute DPO loss using the implicit reward formulation
    3. Backpropagate and update the policy model

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌──────────────────────────────────────────────────────────────────┐
    │                    DPO TRAINING PIPELINE                         │
    │                                                                   │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  Input: (prompt, chosen, rejected) triples               │    │
    │  └──────────────────────┬───────────────────────────────────┘    │
    │                         ↓                                        │
    │  ┌──────────────────────┐   ┌──────────────────────────────┐    │
    │  │   Policy Model (θ)   │   │   Reference Model (θ_ref)    │    │
    │  │   (trainable)        │   │   (frozen)                   │    │
    │  └──────────┬───────────┘   └──────────────┬───────────────┘    │
    │             ↓                               ↓                    │
    │  ┌────────────────────┐    ┌────────────────────────────┐       │
    │  │ log π(chosen)      │    │ log π_ref(chosen)          │       │
    │  │ log π(rejected)    │    │ log π_ref(rejected)        │       │
    │  └─────────┬──────────┘    └──────────────┬─────────────┘       │
    │            ↓                               ↓                     │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  DPO Loss                                                │    │
    │  │  L = -log σ(β * (log_ratio_chosen - log_ratio_rejected)) │    │
    │  │  where log_ratio = log π/π_ref                           │    │
    │  └──────────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2017: RLHF for language models (Ziegler et al.)
    - 2022: InstructGPT — Full RLHF pipeline (Ouyang et al.)
    - 2023: DPO — "Direct Preference Optimization" (Rafailov et al.)
    - 2023: IPO — Identity Preference Optimization (Azar et al.)
    - 2024: SimPO — Reference-free variant (Meng et al.)
    - 2024: ORPO — Combines SFT and preference in one step
    - 2025: DPO variants standard in post-training pipelines

INTERVIEW QUESTIONS:
    1. "How does DPO derive its loss from the RLHF objective?"
       DPO starts from the RLHF objective: maximize reward while staying
       close to the reference policy via KL penalty. This has a closed-form
       solution: the optimal policy is proportional to the reference policy
       times the exponentiated reward. Rearranging gives an expression for
       the implicit reward in terms of the policy and reference log-probs.
       Substituting this into the Bradley-Terry preference model yields the
       DPO loss — a simple binary cross-entropy on the log-ratio difference.

    2. "What is the role of beta in DPO?"
       Beta controls the strength of the KL penalty — how far the policy
       can deviate from the reference. High beta (e.g., 0.5) means strong
       regularization: the policy stays close to the reference, resulting
       in conservative updates. Low beta (e.g., 0.01) allows aggressive
       changes but risks reward hacking and mode collapse. Typical values
       are 0.1-0.5. Beta is analogous to the KL coefficient in PPO.

    3. "When would you choose DPO over RLHF/PPO?"
       DPO is preferred when: (1) you have good preference data, (2) you
       want simplicity and stability, (3) you lack RL infrastructure.
       PPO is preferred when: (1) you need online data collection, (2) the
       reward signal is complex (e.g., learned from human ratings), (3) you
       need fine-grained control over the KL-reward tradeoff. In practice,
       many teams start with DPO and move to PPO only if DPO isn't enough.

################################################################################
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pretrain import PretrainingConfig, Pretrainer


################################################################################
# SECTION 1: DPO CONFIGURATION
################################################################################


@dataclass
class DPOConfig(PretrainingConfig):
    """
    DPO Configuration
    =================

    Extends PretrainingConfig with DPO-specific settings.

    Formula:
        L_DPO = -log(σ(β * (log_ratio_chosen - log_ratio_rejected)))
        where log_ratio = log π(x|prompt) - log π_ref(x|prompt)

    Interview Question:
        "What happens if beta is too low in DPO?"
        If beta is too low, the policy can deviate wildly from the reference.
        This leads to: (1) mode collapse (generating only one type of response),
        (2) reward hacking (exploiting quirks in the preference data), (3)
        degenerate outputs (repetition, incoherence). The KL term is what
        keeps the policy "on distribution" — without it, optimization is
        unbounded and unstable.
    """

    # ------------------------------------------------------------------
    # DPO-specific parameters
    # ------------------------------------------------------------------
    beta: float = 0.1
    """
    KL penalty coefficient (beta in the DPO paper).

    Controls how far the policy can deviate from the reference.
    Higher beta = more conservative. Lower beta = more aggressive.
    """

    reference_model_path: Optional[str] = None
    """
    Path to the reference model checkpoint.
    If None, a copy of the initial model is used as reference.
    """

    label_smoothing: float = 0.0
    """
    Label smoothing for the DPO loss.
    0.0 = standard DPO
    0.1 = smooth labels (reduces overconfidence)
    Equivalent to the IPO (Identity Preference Optimization) objective
    when set to 0.5.
    """

    # ------------------------------------------------------------------
    # DPO learning rate (typically lower than SFT)
    # ------------------------------------------------------------------
    learning_rate: float = 5e-7
    """DPO learning rate (very low — small adjustments to SFT model)."""

    max_steps: int = 2000
    """DPO typically needs very few steps."""

    warmup_steps: int = 50
    """Short warmup for DPO."""


################################################################################
# SECTION 2: LOG-PROBABILITY COMPUTATION
################################################################################


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute per-token log-probabilities under a model.

    Args:
        model: The language model (policy or reference).
        input_ids: Token IDs of shape (batch_size, seq_len).
        labels: Target token IDs of shape (batch_size, seq_len).
        attention_mask: Optional attention mask of shape (batch_size, seq_len).

    Returns:
        Sum of log-probabilities for each sequence in the batch.
        Shape: (batch_size,)

    Explanation:
        1. Run forward pass to get logits
        2. Compute log-softmax over vocabulary dimension
        3. Gather the log-probs at the target token positions
        4. Mask out padding tokens (if attention_mask provided)
        5. Sum log-probs across sequence dimension

    Formula:
        log P(sequence) = Σ_t log P(x_t | x_{<t})

    Example:
        >>> log_probs = compute_log_probs(model, input_ids, labels)
        >>> print(log_probs.shape)
        (batch_size,)
    """
    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask)
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs

    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute log-softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log-probs at target positions
    # Shape: (batch_size, seq_len - 1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    # Mask padding tokens
    if attention_mask is not None:
        shift_mask = attention_mask[..., 1:].contiguous()
        token_log_probs = token_log_probs * shift_mask

    # Also mask positions where label is -100 (ignore_index)
    valid_mask = (shift_labels != -100).float()
    token_log_probs = token_log_probs * valid_mask

    # Sum over sequence dimension
    return token_log_probs.sum(dim=-1)


################################################################################
# SECTION 3: DPO LOSS
################################################################################


class DPOLoss(nn.Module):
    """
    DPO Loss
    ========

    The core DPO loss function.

    Formula:
        L_DPO = -E[log σ(β * (log_ratio_chosen - log_ratio_rejected))]

        where:
            log_ratio_chosen = log π(chosen) - log π_ref(chosen)
            log_ratio_rejected = log π(rejected) - log π_ref(rejected)
            σ = sigmoid function
            β = KL penalty coefficient

    Step by step:
        1. Compute log π(chosen) and log π(rejected) under policy
        2. Compute log π_ref(chosen) and log π_ref(rejected) under reference
        3. Compute log ratios: log π/π_ref for both chosen and rejected
        4. Compute difference: log_ratio_chosen - log_ratio_rejected
        5. Apply sigmoid cross-entropy loss

    WHY this matters:
        The DPO loss directly optimizes the implicit reward:
            r(x, y) = β * log(π(y|x) / π_ref(y|x))
        By maximizing the margin between chosen and rejected rewards,
        the model learns to prefer human-preferred responses without
        ever explicitly training a reward model.

    Interview Question:
        "What is the implicit reward in DPO?"
        The implicit reward is r(x, y) = β * log(π(y|x) / π_ref(y|x)).
        It measures how much more likely the current policy makes a
        response compared to the reference policy, scaled by beta.
        This is the reward that the policy implicitly optimizes — it
        increases for chosen responses and decreases for rejected ones.
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        """
        Initialize DPO loss.

        Args:
            beta: KL penalty coefficient.
            label_smoothing: Label smoothing factor (0 = standard DPO).
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_logps_chosen: torch.Tensor,
        policy_logps_rejected: torch.Tensor,
        reference_logps_chosen: torch.Tensor,
        reference_logps_rejected: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DPO loss and metrics.

        Args:
            policy_logps_chosen: Log-probs of chosen under policy. Shape: (B,).
            policy_logps_rejected: Log-probs of rejected under policy. Shape: (B,).
            reference_logps_chosen: Log-probs of chosen under reference. Shape: (B,).
            reference_logps_rejected: Log-probs of rejected under reference. Shape: (B,).

        Returns:
            Dictionary with:
                - 'loss': The DPO loss scalar.
                - 'chosen_reward': Implicit reward for chosen responses.
                - 'rejected_reward': Implicit reward for rejected responses.
                - 'reward_margin': chosen_reward - rejected_reward.
                - 'accuracy': Fraction of pairs where chosen > rejected.
                - 'log_ratio_chosen': Log-ratio for chosen.
                - 'log_ratio_rejected': Log-ratio for rejected.

        Explanation:
            The loss encourages the log-ratio difference (chosen - rejected)
            to be positive. The sigmoid converts this to a probability, and
            we minimize the negative log-likelihood. Label smoothing reduces
            overconfidence by softening the target from 1.0 to (1 - alpha).
        """
        # Compute log ratios
        log_ratio_chosen = policy_logps_chosen - reference_logps_chosen
        log_ratio_rejected = policy_logps_rejected - reference_logps_rejected

        # Compute the DPO logits
        logits = self.beta * (log_ratio_chosen - log_ratio_rejected)

        # Apply label smoothing
        if self.label_smoothing > 0:
            # Equivalent to IPO when label_smoothing = 0.5
            targets = torch.ones_like(logits) * (1.0 - self.label_smoothing)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        else:
            # Standard DPO: target is 1 (chosen should have higher reward)
            loss = -F.logsigmoid(logits).mean()

        # Compute metrics
        chosen_reward = self.beta * log_ratio_chosen.detach()
        rejected_reward = self.beta * log_ratio_rejected.detach()
        reward_margin = (chosen_reward - rejected_reward).mean()
        accuracy = (log_ratio_chosen > log_ratio_rejected).float().mean()

        return {
            "loss": loss,
            "chosen_reward": chosen_reward.mean(),
            "rejected_reward": rejected_reward.mean(),
            "reward_margin": reward_margin,
            "accuracy": accuracy,
            "log_ratio_chosen": log_ratio_chosen.mean(),
            "log_ratio_rejected": log_ratio_rejected.mean(),
        }


################################################################################
# SECTION 4: DPO TRAINER
################################################################################


class DPOTrainer:
    """
    DPO Trainer
    ============

    Trains a language model using Direct Preference Optimization.

    Given (prompt, chosen, rejected) triples, optimizes the policy to
    prefer chosen completions relative to a frozen reference policy.

    Step by step:
        1. Load reference model (frozen copy of SFT model)
        2. For each batch of (prompt, chosen, rejected):
            a. Concatenate prompt+chosen and prompt+rejected
            b. Compute log-probs under policy (with gradient)
            c. Compute log-probs under reference (no gradient)
            d. Compute DPO loss
            e. Backpropagate and update policy
        3. Log metrics: loss, rewards, margin, accuracy

    WHY this matters:
        DPO is the simplest way to incorporate human preferences into
        a language model. No reward model, no PPO, no RL infrastructure.
        Just a classification loss on preference pairs. This simplicity
        made preference tuning accessible to every team, not just those
        with RL expertise.

    Interview Question:
        "Walk me through a DPO training step."
        Given a batch of (prompt, chosen, rejected) triples:
        1. Concatenate prompt + chosen → input_chosen
        2. Concatenate prompt + rejected → input_rejected
        3. Forward pass through policy: get log P(chosen) and log P(rejected)
        4. Forward pass through frozen ref: get log P_ref(chosen) and log P_ref(rejected)
        5. Compute log-ratios: log_ratio_chosen = log P(chosen) - log P_ref(chosen)
        6. DPO loss = -log σ(β * (log_ratio_chosen - log_ratio_rejected))
        7. Backpropagate through policy only (reference is frozen)
        8. Update policy with optimizer
    """

    def __init__(
        self,
        model: nn.Module,
        config: DPOConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the DPO Trainer.

        Args:
            model: The SFT model to further align with DPO.
            config: DPOConfig with all DPO hyperparameters.
            device: Target device. Defaults to CUDA if available.

        Explanation:
            1. Create a frozen copy of the model as the reference
            2. Setup the DPO loss function
            3. Use the Pretrainer for the core training loop mechanics
        """
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ------------------------------------------------------------------
        # Policy model (trainable)
        # ------------------------------------------------------------------
        self.model = model.to(self.device)

        # ------------------------------------------------------------------
        # Reference model (frozen)
        # ------------------------------------------------------------------
        if config.reference_model_path is not None:
            # Load reference from checkpoint
            self.reference_model = self._load_reference_model(
                config.reference_model_path
            )
        else:
            # Use a copy of the current model as reference
            import copy
            self.reference_model = copy.deepcopy(model).to(self.device)

        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()

        # ------------------------------------------------------------------
        # DPO loss
        # ------------------------------------------------------------------
        self.dpo_loss = DPOLoss(
            beta=config.beta,
            label_smoothing=config.label_smoothing,
        )

        # ------------------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # ------------------------------------------------------------------
        # Track metrics
        # ------------------------------------------------------------------
        self.global_step = 0

        # Print recipe
        self._print_recipe()

    def _load_reference_model(self, path: str) -> nn.Module:
        """
        Load a reference model from a checkpoint.

        Args:
            path: Path to the reference model checkpoint.

        Returns:
            The loaded reference model.
        """
        checkpoint = torch.load(path, map_location=self.device)
        model = self.model.__class__()  # Create same architecture
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def _print_recipe(self) -> None:
        """Print DPO recipe summary."""
        c = self.config
        border = "=" * 70
        print(f"\n{border}")
        print("DPO RECIPE SUMMARY")
        print(border)
        print(f"  Model:            {c.model_name}")
        print(f"  Beta:             {c.beta}")
        print(f"  Label smoothing:  {c.label_smoothing}")
        print(f"  Learning rate:    {c.learning_rate}")
        print(f"  Max steps:        {c.max_steps}")
        print(f"  Ref model path:   {c.reference_model_path or 'copy of initial model'}")
        print(border + "\n")

    def _compute_log_probs_batch(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log-probabilities for a batch.

        Args:
            model: The model to compute log-probs under.
            input_ids: Token IDs of shape (batch_size, seq_len).
            labels: Target labels of shape (batch_size, seq_len).

        Returns:
            Sum of log-probs per sequence. Shape: (batch_size,).
        """
        return compute_log_probs(model, input_ids, labels)

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Execute a single DPO training step.

        Args:
            batch: Dictionary with:
                - 'prompt_input_ids': Prompt token IDs. Shape: (B, P).
                - 'chosen_input_ids': Prompt+chosen token IDs. Shape: (B, C1).
                - 'chosen_labels': Labels for chosen. Shape: (B, C1).
                - 'rejected_input_ids': Prompt+rejected token IDs. Shape: (B, C2).
                - 'rejected_labels': Labels for rejected. Shape: (B, C2).

        Returns:
            Dictionary of metrics (loss, rewards, margin, accuracy).

        Explanation:
            1. Forward pass through policy for chosen and rejected
            2. Forward pass through frozen reference for chosen and rejected
            3. Compute DPO loss from the four log-prob sets
            4. Backpropagate through policy
            5. Step optimizer
        """
        self.model.train()

        # Move to device
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)

        # ------------------------------------------------------------------
        # Policy log-probs (with gradient)
        # ------------------------------------------------------------------
        policy_logps_chosen = self._compute_log_probs_batch(
            self.model, chosen_ids, chosen_labels
        )
        policy_logps_rejected = self._compute_log_probs_batch(
            self.model, rejected_ids, rejected_labels
        )

        # ------------------------------------------------------------------
        # Reference log-probs (no gradient)
        # ------------------------------------------------------------------
        with torch.no_grad():
            ref_logps_chosen = self._compute_log_probs_batch(
                self.reference_model, chosen_ids, chosen_labels
            )
            ref_logps_rejected = self._compute_log_probs_batch(
                self.reference_model, rejected_ids, rejected_labels
            )

        # ------------------------------------------------------------------
        # DPO loss
        # ------------------------------------------------------------------
        loss_dict = self.dpo_loss(
            policy_logps_chosen,
            policy_logps_rejected,
            ref_logps_chosen,
            ref_logps_rejected,
        )

        # ------------------------------------------------------------------
        # Backward and optimize
        # ------------------------------------------------------------------
        self.optimizer.zero_grad()
        loss_dict["loss"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip_norm,
        )

        self.optimizer.step()
        self.global_step += 1

        # Return metrics as floats
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}

    def train(
        self,
        dataloader: Any,
    ) -> None:
        """
        Run DPO training over a dataloader.

        Args:
            dataloader: An iterable yielding batches of preference data.
                        Each batch should have keys:
                            - 'chosen_input_ids', 'chosen_labels'
                            - 'rejected_input_ids', 'rejected_labels'

        Explanation:
            Main training loop. Iterates over the dataloader, calls
            train_step for each batch, and logs metrics periodically.
        """
        self.model.train()

        print("Starting DPO training...")
        for step, batch in enumerate(dataloader):
            if self.global_step >= self.config.max_steps:
                break

            metrics = self.train_step(batch)

            # Logging
            if self.global_step % self.config.log_every_n_steps == 0:
                print(
                    f"step={self.global_step:>6d} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"chosen_r={metrics['chosen_reward']:.4f} | "
                    f"rejected_r={metrics['rejected_reward']:.4f} | "
                    f"margin={metrics['reward_margin']:.4f} | "
                    f"acc={metrics['accuracy']:.3f}"
                )

        print("DPO training complete.")


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################


def demonstrate_dpo():
    """
    Demonstrate DPO with a toy model and synthetic preference data.

    Shows:
        1. DPO loss computation
        2. Log-probability computation
        3. A brief training loop
    """
    print("=" * 70)
    print("DPO TRAINER DEMONSTRATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Demo 1: DPO Loss
    # ------------------------------------------------------------------
    print("\n[1/4] DPO Loss Computation")
    print("-" * 40)

    dpo_loss_fn = DPOLoss(beta=0.1, label_smoothing=0.0)

    # Simulate log-probs
    policy_chosen = torch.tensor([-1.0, -1.5, -0.8])
    policy_rejected = torch.tensor([-2.0, -2.5, -1.8])
    ref_chosen = torch.tensor([-1.5, -2.0, -1.3])
    ref_rejected = torch.tensor([-1.5, -2.0, -1.3])

    loss_dict = dpo_loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    print(f"  Loss:            {loss_dict['loss']:.4f}")
    print(f"  Chosen reward:   {loss_dict['chosen_reward']:.4f}")
    print(f"  Rejected reward: {loss_dict['rejected_reward']:.4f}")
    print(f"  Reward margin:   {loss_dict['reward_margin']:.4f}")
    print(f"  Accuracy:        {loss_dict['accuracy']:.3f}")

    # ------------------------------------------------------------------
    # Demo 2: Log-probability computation
    # ------------------------------------------------------------------
    print("\n[2/4] Log-Probability Computation")
    print("-" * 40)

    class ToyModel(nn.Module):
        def __init__(self, vocab_size: int = 100, d_model: int = 32):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.linear = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            return self.linear(x)

    model = ToyModel(vocab_size=100, d_model=32)
    input_ids = torch.randint(0, 100, (2, 16))
    labels = torch.randint(0, 100, (2, 16))

    log_probs = compute_log_probs(model, input_ids, labels)
    print(f"  Log-probs shape: {log_probs.shape}")
    print(f"  Log-probs:       {log_probs.tolist()}")

    # ------------------------------------------------------------------
    # Demo 3: Label smoothing comparison
    # ------------------------------------------------------------------
    print("\n[3/4] Label Smoothing Comparison")
    print("-" * 40)

    for ls in [0.0, 0.1, 0.5]:
        dpo = DPOLoss(beta=0.1, label_smoothing=ls)
        result = dpo(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        label = "DPO" if ls == 0.0 else ("Smoothed DPO" if ls == 0.1 else "IPO")
        print(f"  {label:>15s} (ls={ls}): loss={result['loss']:.4f}, acc={result['accuracy']:.3f}")

    # ------------------------------------------------------------------
    # Demo 4: DPO Config
    # ------------------------------------------------------------------
    print("\n[4/4] DPO Configuration")
    print("-" * 40)

    config = DPOConfig(
        model_name="sota-llm-dpo",
        beta=0.1,
        label_smoothing=0.0,
        learning_rate=5e-7,
        max_steps=2000,
    )
    print(f"  Beta:            {config.beta}")
    print(f"  Label smoothing: {config.label_smoothing}")
    print(f"  Learning rate:   {config.learning_rate}")
    print(f"  Max steps:       {config.max_steps}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_dpo()
