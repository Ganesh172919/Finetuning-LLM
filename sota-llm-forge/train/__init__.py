"""
################################################################################
SOTA LLM FORGE — TRAINING PACKAGE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is this package?
    The training infrastructure for a state-of-the-art large language model.
    Implements pretraining, supervised fine-tuning (SFT), direct preference
    optimization (DPO), and reinforcement learning with verifiable rewards
    (GRPO and its variants).

Why does it matter?
    Training is where the model learns. The quality of the training loop,
    the precision of the loss computation, and the stability of the
    optimization directly determine the model's capabilities. A well-
    designed training pipeline saves compute, prevents failures, and
    produces better models.

How does it work?
    The package is organized by training phase:
      - pretrain.py:  Main pretraining loop with spike detection, curriculum
      - sft.py:       Supervised fine-tuning with chat templates and loss masking
      - dpo.py:       Direct preference optimization for alignment
      - rlvr_grpo.py: GRPO + DAPO + GSPO + Dr. GRPO for reasoning RL

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌──────────────────────────────────────────────────────────────────┐
    │                    TRAINING PIPELINE                              │
    │                                                                   │
    │  Phase 1: Pretraining                                            │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  pretrain.py                                              │    │
    │  │  • Massive text corpus → base model                       │    │
    │  │  • Mixed precision (BF16/FP8)                             │    │
    │  │  • Loss spike detection + rollback                        │    │
    │  │  • Sequence length curriculum                             │    │
    │  └──────────────────────────┬───────────────────────────────┘    │
    │                             ↓                                    │
    │  Phase 2: Supervised Fine-Tuning                                 │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  sft.py                                                   │    │
    │  │  • Chat templates (system/user/assistant)                 │    │
    │  │  • Loss masking (train only on completions)               │    │
    │  │  • Think-token handling (<|think|>...</|think|>)          │    │
    │  └──────────────────────────┬───────────────────────────────┘    │
    │                             ↓                                    │
    │  Phase 3a: Preference Alignment (DPO)                            │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  dpo.py                                                   │    │
    │  │  • (prompt, chosen, rejected) triples                     │    │
    │  │  • No reward model needed                                 │    │
    │  │  • Beta-controlled KL penalty                             │    │
    │  └──────────────────────────────────────────────────────────┘    │
    │                                                                   │
    │  Phase 3b: Reasoning RL (GRPO)                                   │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  rlvr_grpo.py                                             │    │
    │  │  • Verifiable rewards (math, code)                        │    │
    │  │  • Group-relative advantages (no critic)                  │    │
    │  │  • DAPO/GSPO/Dr. GRPO refinements                        │    │
    │  │  • Reward-hacking detection                               │    │
    │  └──────────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2017: Transformer + AdamW pretraining (Vaswani et al.)
    - 2020: GPT-3 — Scaling laws for LLMs (Brown et al.)
    - 2022: InstructGPT — SFT + RLHF pipeline (Ouyang et al.)
    - 2023: DPO — Preference alignment without RL (Rafailov et al.)
    - 2024: GRPO — Group-relative policy optimization (DeepSeek-Math)
    - 2025: DeepSeek-R1 — Reasoning via GRPO with verifiable rewards
    - 2025: DAPO, GSPO, Dr. GRPO — GRPO refinements
    - 2026: RLVR becomes standard for reasoning model training

INTERVIEW QUESTIONS:
    1. "What is the standard LLM training pipeline?"
       Three phases: (1) Pretraining on trillions of tokens to learn language,
       (2) SFT on instruction-response pairs to learn to follow instructions,
       (3) Alignment via DPO or RLHF/GRPO to match human preferences or
       learn reasoning. Each phase builds on the previous one.

    2. "When should you use DPO vs GRPO?"
       DPO for general preference alignment (helpfulness, safety, style).
       GRPO for reasoning tasks where you have verifiable rewards (math,
       code, logic). DPO needs preference pairs; GRPO needs a reward
       function. Many pipelines use both: DPO for general alignment,
       then GRPO for reasoning specialization.

    3. "How do you prevent catastrophic forgetting during fine-tuning?"
       (1) Use a much lower learning rate than pretraining (10-100x lower),
       (2) Train for fewer steps (thousands, not millions),
       (3) Use KL penalty against the pretrained reference model,
       (4) Mix in some pretraining data during SFT,
       (5) Use LoRA or other parameter-efficient methods.

################################################################################
"""

from .pretrain import (
    PretrainingConfig,
    Pretrainer,
    LossSpikeDetector,
    MetricsTracker,
    get_cosine_schedule_with_warmup,
    get_wsd_schedule_with_warmup,
)

from .sft import (
    SFTConfig,
    SFTTrainer,
    SFTTokenizerWrapper,
    format_chat_message,
    format_chat_conversation,
    create_completion_mask,
    create_think_mask,
    SPECIAL_TOKENS,
)

from .dpo import (
    DPOConfig,
    DPOTrainer,
    DPOLoss,
    compute_log_probs,
)

from .rlvr_grpo import (
    GRPOConfig,
    GRPOTrainer,
    VerifiableRewardFunction,
    MathExactMatchReward,
    CodeExecutionReward,
    EntropyMonitor,
    compute_group_advantages,
    compute_grpo_loss,
    compute_sequence_level_ratio,
    compute_kl_penalty,
)

__all__ = [
    # Pretraining
    "PretrainingConfig",
    "Pretrainer",
    "LossSpikeDetector",
    "MetricsTracker",
    "get_cosine_schedule_with_warmup",
    "get_wsd_schedule_with_warmup",
    # SFT
    "SFTConfig",
    "SFTTrainer",
    "SFTTokenizerWrapper",
    "format_chat_message",
    "format_chat_conversation",
    "create_completion_mask",
    "create_think_mask",
    "SPECIAL_TOKENS",
    # DPO
    "DPOConfig",
    "DPOTrainer",
    "DPOLoss",
    "compute_log_probs",
    # GRPO
    "GRPOConfig",
    "GRPOTrainer",
    "VerifiableRewardFunction",
    "MathExactMatchReward",
    "CodeExecutionReward",
    "EntropyMonitor",
    "compute_group_advantages",
    "compute_grpo_loss",
    "compute_sequence_level_ratio",
    "compute_kl_penalty",
]
