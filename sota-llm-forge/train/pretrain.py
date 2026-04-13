"""
################################################################################
PRETRAINING LOOP — MAIN PRETRAINING INFRASTRUCTURE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is pretraining?
    Pretraining is the foundational phase of LLM training where the model
    learns to predict the next token on massive text corpora (trillions of
    tokens). This is where the model acquires language understanding, world
    knowledge, and basic reasoning capabilities. It is the most compute-
    intensive phase — a single run can cost millions of dollars and take
    weeks on thousands of GPUs.

Why does it matter?
    Pretraining determines the model's "base capabilities." Everything
    downstream — SFT, RLHF, reasoning — builds on what the model learned
    during pretraining. A well-designed pretraining loop with proper
    curriculum, spike detection, and mixed precision can save enormous
    compute costs while producing better models.

How does it work?
    1. Load tokenized data into a streaming DataLoader
    2. For each batch: forward pass -> compute loss -> backward pass
    3. Accumulate gradients over micro-batches (gradient accumulation)
    4. Clip gradients to prevent explosions
    5. Step optimizer and scheduler
    6. Monitor for loss spikes (z-score detection)
    7. Rollback automatically if spike detected
    8. Periodically save checkpoints
    9. Log metrics (loss, grad norm, throughput, LR)

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌──────────────────────────────────────────────────────────────────┐
    │                    PRETRAINING LOOP                               │
    │                                                                   │
    │  ┌─────────────┐                                                  │
    │  │  DataLoader  │──── streaming tokenized text                    │
    │  └──────┬──────┘                                                  │
    │         ↓                                                         │
    │  ┌──────────────────────────────────────────────────────────────┐ │
    │  │  Gradient Accumulation Window (K micro-batches)              │ │
    │  │  ┌──────────┐                                                │ │
    │  │  │ Micro-B1 │──forward──loss──backward──→ grad_acc[0]        │ │
    │  │  │ Micro-B2 │──forward──loss──backward──→ grad_acc[1]        │ │
    │  │  │   ...    │                                                │ │
    │  │  │ Micro-BK │──forward──loss──backward──→ grad_acc[K-1]      │ │
    │  │  └──────────┘                                                │ │
    │  └──────────────────────────┬───────────────────────────────────┘ │
    │                             ↓                                     │
    │  ┌──────────────┐  ┌───────────────┐  ┌───────────────────────┐  │
    │  │ Grad Clip    │→ │ Optimizer Step│→ │ LR Scheduler Step     │  │
    │  └──────────────┘  └───────────────┘  └───────────────────────┘  │
    │                             ↓                                     │
    │  ┌──────────────────────────────────────────────────────────────┐ │
    │  │  Monitoring & Control                                        │ │
    │  │  • Loss spike detection (z-score)                            │ │
    │  │  • Automatic rollback                                        │ │
    │  │  • Checkpointing                                             │ │
    │  │  • Logging (W&B / console)                                   │ │
    │  └──────────────────────────────────────────────────────────────┘ │
    └──────────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2017: "Attention Is All You Need" — Transformer architecture, AdamW optimizer
    - 2020: GPT-3 (175B) — First large-scale pretraining with gradient accumulation
    - 2022: Chinchilla — Compute-optimal training (Hoffmann et al.)
    - 2022: FlashAttention — IO-aware attention, enabling longer contexts
    - 2023: FP8 Training — H100 native precision, 2x throughput gains
    - 2024: Sequence length curriculum — Pretrain short, extend later
    - 2024: WSD scheduler — Warmup-Stable-Decay replaces cosine for flexibility
    - 2025: DeepSeek-V3 — 14.8T tokens, MoE, aux-loss-free routing
    - 2025: Loss spike detection and automatic rollback become standard practice

INTERVIEW QUESTIONS:
    1. "How do you handle loss spikes during pretraining?"
       Detect spikes using a z-score test against a rolling window of recent
       losses. If z-score exceeds a threshold (e.g., 5.0), roll back to the
       last checkpoint and skip the offending data. The root cause is usually
       bad data (corrupted/tokenization errors) or numerical instability.
       Some teams also reduce the learning rate temporarily after a spike.

    2. "What is gradient accumulation and why is it needed?"
       Gradient accumulation simulates a larger batch size than fits in GPU
       memory. Instead of updating weights after each micro-batch, we
       accumulate gradients over K micro-batches, average them, then update.
       Effective batch size = micro_batch_size * K * num_GPUs. This is
       critical for large models where even batch_size=1 may not fit.

    3. "Why use BF16 instead of FP16 for pretraining?"
       BF16 has the same exponent range as FP32 (8 bits for exponent) but
       fewer mantissa bits (7 vs 23). This means BF16 rarely overflows or
       underflows, making it much more stable for training than FP16, which
       has a narrower dynamic range and often requires loss scaling. The
       trade-off is slightly lower precision, but for LLM training the
       stability gains far outweigh the precision loss.

################################################################################
"""

import os
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Iterator

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


################################################################################
# SECTION 1: PRETRAINING CONFIGURATION
################################################################################


@dataclass
class PretrainingConfig:
    """
    Pretraining Configuration
    =========================

    All pretraining hyperparameters in one place. No magic numbers —
    every value is named and documented. This config is passed to
    Pretrainer at construction time and printed as a "recipe summary"
    at the start of every run.

    Design philosophy:
        - Every constant that would otherwise be a magic number lives here
        - Sensible defaults for a 7B-class model
        - Override fields for specific experiments without subclassing

    Interview Question:
        "Why use a dataclass instead of a dict for training config?"
        A dataclass gives us: (1) type hints for IDE support and validation,
        (2) default values with clear documentation, (3) immutable fields
        when frozen, (4) easy serialization to/from YAML/JSON. A dict gives
        none of these guarantees and is prone to silent typo bugs.
    """

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------
    model_name: str = "sota-llm-forge"
    """Name of the model being trained (for logging and checkpoint naming)."""

    vocab_size: int = 128256
    """Vocabulary size. Must match tokenizer."""

    d_model: int = 4096
    """Model hidden dimension."""

    num_layers: int = 32
    """Number of transformer blocks."""

    num_heads: int = 32
    """Number of attention heads."""

    # ------------------------------------------------------------------
    # Optimizer configuration
    # ------------------------------------------------------------------
    learning_rate: float = 3e-4
    """Peak learning rate. For muP, this is the base LR (transfers across widths)."""

    min_learning_rate: float = 3e-5
    """Minimum LR at end of cosine decay (typically 10% of peak)."""

    weight_decay: float = 0.1
    """AdamW weight decay coefficient."""

    beta1: float = 0.9
    """AdamW first moment decay."""

    beta2: float = 0.95
    """AdamW second moment decay. 0.95 is standard for LLMs (vs 0.999 for vision)."""

    grad_clip_norm: float = 1.0
    """Max gradient norm for clipping. Prevents gradient explosions."""

    # ------------------------------------------------------------------
    # Schedule configuration
    # ------------------------------------------------------------------
    max_steps: int = 100_000
    """Total training steps."""

    warmup_steps: int = 2000
    """Number of warmup steps (linear LR warmup)."""

    scheduler_type: str = "cosine"
    """LR scheduler type: 'cosine' or 'wsd' (warmup-stable-decay)."""

    wsd_decay_fraction: float = 0.1
    """For WSD scheduler: fraction of steps spent in decay phase."""

    # ------------------------------------------------------------------
    # Data configuration
    # ------------------------------------------------------------------
    batch_size: int = 4
    """Micro-batch size (sequences per GPU per step)."""

    gradient_accumulation_steps: int = 8
    """Number of micro-batches before optimizer step."""

    sequence_length: int = 4096
    """Default sequence length in tokens."""

    # ------------------------------------------------------------------
    # Sequence length curriculum
    # ------------------------------------------------------------------
    sequence_length_curriculum: List[Tuple[int, int]] = field(
        default_factory=lambda: [(0, 2048), (10_000, 4096), (50_000, 8192)]
    )
    """
    Sequence length curriculum: list of (step, seq_len) pairs.
    At each step threshold, the sequence length increases to the specified value.
    Example: [(0, 2048), (10000, 4096)] means start at 2048, extend to 4096 at step 10k.
    This saves compute early on and lets the model learn short-range patterns first.
    """

    # ------------------------------------------------------------------
    # Batch size ramp
    # ------------------------------------------------------------------
    batch_size_ramp_steps: int = 0
    """Number of steps to ramp batch size from 1 to target. 0 = no ramp."""

    # ------------------------------------------------------------------
    # Precision configuration
    # ------------------------------------------------------------------
    precision: str = "bf16"
    """Training precision: 'bf16', 'fp16', 'fp32', or 'fp8' (experimental)."""

    fp8_format: str = "e4m3"
    """FP8 format if precision='fp8': 'e4m3' (forward) or 'e5m2' (backward)."""

    # ------------------------------------------------------------------
    # Parallelism configuration
    # ------------------------------------------------------------------
    parallelism: Dict[str, int] = field(
        default_factory=lambda: {"dp": 1, "tp": 1, "ep": 1, "pp": 1}
    )
    """
    Parallelism strategy:
        dp: Data Parallel degree (DDP or FSDP2)
        tp: Tensor Parallel degree (split attention heads across GPUs)
        ep: Expert Parallel degree (split MoE experts across GPUs)
        pp: Pipeline Parallel degree (split layers across GPUs)
    Note: tp, ep, pp are stubs — documented but not fully wired in this file.
    """

    # ------------------------------------------------------------------
    # Activation checkpointing
    # ------------------------------------------------------------------
    activation_checkpointing: bool = True
    """Enable activation checkpointing on transformer blocks to save memory."""

    activation_checkpointing_omit_last_n: int = 0
    """Do NOT checkpoint the last N layers (they are cheap and frequent)."""

    # ------------------------------------------------------------------
    # Spike detection and rollback
    # ------------------------------------------------------------------
    spike_detection_threshold: float = 5.0
    """Z-score threshold for loss spike detection. Higher = less sensitive."""

    spike_detection_window: int = 100
    """Number of recent losses to use for rolling mean/std."""

    auto_rollback_on_spike: bool = True
    """Automatically rollback to last good checkpoint on spike detection."""

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    checkpoint_every_n_steps: int = 1000
    """Save a checkpoint every N steps."""

    checkpoint_dir: str = "./checkpoints"
    """Directory to save checkpoints."""

    keep_last_n_checkpoints: int = 3
    """Number of recent checkpoints to keep (older ones are deleted)."""

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_every_n_steps: int = 10
    """Log metrics every N steps."""

    use_wandb: bool = False
    """Whether to log to Weights & Biases."""

    wandb_project: str = "sota-llm-forge"
    """W&B project name."""

    wandb_run_name: Optional[str] = None
    """W&B run name. None = auto-generated."""

    # ------------------------------------------------------------------
    # System configuration
    # ------------------------------------------------------------------
    seed: int = 42
    """Random seed for reproducibility."""

    num_workers: int = 4
    """Number of DataLoader workers."""

    pin_memory: bool = True
    """Pin memory for faster CPU->GPU transfer."""

    compile_model: bool = False
    """Use torch.compile() for the model (requires PyTorch 2.0+)."""

    # ------------------------------------------------------------------
    # Multi-token prediction
    # ------------------------------------------------------------------
    mtp_loss_weight: float = 0.0
    """Weight for multi-token prediction loss. 0 = disabled."""

    def effective_batch_size(self) -> int:
        """
        Compute the effective (global) batch size.

        effective_batch_size = micro_batch_size * grad_accum * dp_degree

        Returns:
            The total number of sequences processed per optimizer step.
        """
        dp_degree = self.parallelism.get("dp", 1)
        return self.batch_size * self.gradient_accumulation_steps * dp_degree

    def tokens_per_step(self) -> int:
        """
        Compute the number of tokens processed per optimizer step.

        Returns:
            effective_batch_size * sequence_length
        """
        return self.effective_batch_size() * self.sequence_length


################################################################################
# SECTION 2: LEARNING RATE SCHEDULERS
################################################################################


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine annealing schedule with linear warmup.

    The learning rate follows:
        - Linear warmup from 0 to peak_lr over warmup_steps
        - Cosine decay from peak_lr to min_lr over remaining steps

    Formula:
        LR(step) = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
        where progress = (step - warmup_steps) / (max_steps - warmup_steps)

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of linear warmup steps.
        max_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of peak LR.

    Returns:
        A LambdaLR scheduler.

    Interview Question:
        "Why cosine schedule instead of linear decay?"
        Cosine decay spends more time near the peak LR (gentler initial decay),
        which allows the model to learn longer at higher LRs. Linear decay
        drops too quickly. Empirically, cosine produces slightly better final
        loss for the same number of steps.
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # Cosine decay
        progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_wsd_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    decay_fraction: float = 0.1,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Warmup-Stable-Decay (WSD) schedule.

    Three phases:
        1. Warmup: linear increase from 0 to peak_lr
        2. Stable: constant peak_lr for most of training
        3. Decay: rapid decay (cosine or linear) to min_lr

    Formula:
        if step < warmup:         LR = peak * (step / warmup)
        elif step < stable_end:   LR = peak
        else:                     LR = peak * decay_fn(progress)

    This schedule is more flexible than cosine because you can decide
    WHEN to start decaying (e.g., when data is exhausted or loss plateaus).

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of linear warmup steps.
        max_steps: Total training steps.
        decay_fraction: Fraction of total steps spent in decay phase.
        min_lr_ratio: Minimum LR as fraction of peak LR.

    Returns:
        A LambdaLR scheduler.

    Reference:
        "MiniCPM: Unveiling the Potential of Small Language Models"
        (Hu et al., 2024) — popularized WSD for LLM pretraining.
    """
    decay_steps = int(max_steps * decay_fraction)
    stable_end = max_steps - decay_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        if current_step < stable_end:
            return 1.0
        # Decay phase (cosine)
        decay_progress = (current_step - stable_end) / max(1, decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


################################################################################
# SECTION 3: LOSS SPIKE DETECTION
################################################################################


class LossSpikeDetector:
    """
    Loss Spike Detector
    ===================

    Detects anomalous loss spikes using a z-score test against a rolling
    window of recent loss values.

    Formula:
        z = (loss_current - mean(loss_window)) / std(loss_window)

    If z > threshold, the current step is flagged as a spike.

    Step by step:
        1. Maintain a rolling window of the last N loss values
        2. When a new loss arrives, compute z-score
        3. If z > threshold, return True (spike detected)
        4. Optionally trigger automatic rollback

    WHY this matters:
        Loss spikes can corrupt model weights. If not detected and handled,
        a single bad batch (corrupted data, numerical overflow) can waste
        hours of compute. Automatic detection + rollback saves both time
        and money.

    Interview Question:
        "How do you distinguish a real loss spike from normal training noise?"
        A z-score of 5+ is very unlikely under normal conditions (p < 0.000001).
        We also require the window to have enough data points (>20) to get
        reliable statistics. Some teams also check the gradient norm — a
        spike in loss usually coincides with a spike in grad norm. If both
        spike together, it is almost certainly a real anomaly, not noise.
    """

    def __init__(self, threshold: float = 5.0, window_size: int = 100):
        """
        Initialize the spike detector.

        Args:
            threshold: Z-score threshold for spike detection.
            window_size: Number of recent losses to keep in the rolling window.
        """
        self.threshold = threshold
        self.window_size = window_size
        self.loss_history: List[float] = []
        self.spike_count: int = 0

    def check(self, loss: float) -> bool:
        """
        Check if a loss value is a spike.

        Args:
            loss: The current loss value.

        Returns:
            True if the loss is detected as a spike, False otherwise.

        Explanation:
            We need at least 20 data points for reliable statistics.
            The z-score measures how many standard deviations the current
            loss is from the rolling mean. A z-score of 5 means the loss
            is 5 standard deviations away — extremely unlikely by chance.
        """
        self.loss_history.append(loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history = self.loss_history[-self.window_size:]

        # Need enough data for reliable statistics
        if len(self.loss_history) < 20:
            return False

        window = self.loss_history[:-1]  # Exclude current from stats
        mean_loss = sum(window) / len(window)
        variance = sum((x - mean_loss) ** 2 for x in window) / len(window)
        std_loss = math.sqrt(variance + 1e-8)  # Add epsilon for numerical stability

        z_score = (loss - mean_loss) / std_loss

        if z_score > self.threshold:
            self.spike_count += 1
            logger.warning(
                f"LOSS SPIKE DETECTED: loss={loss:.4f}, z_score={z_score:.2f}, "
                f"mean={mean_loss:.4f}, std={std_loss:.4f}, "
                f"spike_count={self.spike_count}"
            )
            return True
        return False

    def reset(self) -> None:
        """Reset the loss history (e.g., after rollback)."""
        self.loss_history = []


################################################################################
# SECTION 4: METRICS TRACKER
################################################################################


class MetricsTracker:
    """
    Metrics Tracker
    ===============

    Tracks and aggregates training metrics over a logging window.
    Computes running averages for loss, grad norm, throughput, etc.

    WHY this matters:
        Raw metrics are noisy. Averaging over a window gives a smoother
        picture of training progress. We also compute tokens/second to
        monitor hardware utilization and estimate time to completion.
    """

    def __init__(self, log_every_n_steps: int = 10):
        self.log_every_n_steps = log_every_n_steps
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators for the next logging window."""
        self.loss_sum: float = 0.0
        self.grad_norm_sum: float = 0.0
        self.tokens_sum: int = 0
        self.count: int = 0
        self.window_start_time: float = time.time()

    def update(
        self,
        loss: float,
        grad_norm: float,
        num_tokens: int,
    ) -> None:
        """
        Accumulate metrics from one training step.

        Args:
            loss: The loss value for this step.
            grad_norm: The gradient norm after clipping.
            num_tokens: Number of tokens processed in this step.
        """
        self.loss_sum += loss
        self.grad_norm_sum += grad_norm
        self.tokens_sum += num_tokens
        self.count += 1

    def get_and_reset(self) -> Dict[str, float]:
        """
        Compute averages and reset accumulators.

        Returns:
            Dictionary of averaged metrics:
                - loss: Average loss over the window
                - grad_norm: Average gradient norm
                - tokens_per_sec: Throughput in tokens/second
                - tokens_processed: Total tokens in window
        """
        if self.count == 0:
            return {}

        elapsed = time.time() - self.window_start_time
        metrics = {
            "loss": self.loss_sum / self.count,
            "grad_norm": self.grad_norm_sum / self.count,
            "tokens_per_sec": self.tokens_sum / max(elapsed, 1e-6),
            "tokens_processed": self.tokens_sum,
        }
        self.reset()
        return metrics


################################################################################
# SECTION 5: PRETRAINER
################################################################################


class Pretrainer:
    """
    Pretrainer
    ==========

    Full pretraining loop with:
        - BF16 default precision (FP8 optional flag)
        - Data parallel (DDP/FSDP2)
        - Tensor parallel stub (documented, not fully wired)
        - Expert parallel stub
        - Activation checkpointing on deepest blocks
        - Batch size ramp (start smaller, ramp to target)
        - Sequence length curriculum (pretrain at shorter, extend later)
        - Loss spike detection (z-score against rolling window)
        - Automatic rollback to last good checkpoint
        - Recipe summary at startup
        - Logging: loss, grad norm, expert load, LR, tokens/sec

    Formula:
        L_pretrain = -1/T * Σ_t log P(x_t | x_{<t}; θ)

    Step by step:
        1. Setup: model, optimizer, scheduler, logging
        2. For each step:
            a. Get current sequence length from curriculum
            b. Load batch of token sequences
            c. Forward pass (with autocast for mixed precision)
            d. Compute cross-entropy loss
            e. Scale loss for gradient accumulation
            f. Backward pass
            g. If accumulation window complete:
                - Clip gradients
                - Optimizer step
                - Scheduler step
                - Zero gradients
            h. Check for loss spikes
            i. Rollback if spike detected
            j. Log metrics periodically
            k. Save checkpoint periodically

    WHY this matters:
        The pretraining loop is where 99% of compute is spent. Small
        improvements (better curriculum, spike detection, mixed precision)
        compound over trillions of tokens into massive savings.

    Interview Question:
        "Walk me through the pretraining loop for a 7B model."
        Start with config: 7B params, 4096 seq_len, batch=4, grad_accum=8,
        BF16 precision, cosine schedule, LR=3e-4. Setup AdamW with
        weight_decay=0.1. Each step: load 4 sequences of 4096 tokens,
        forward pass produces logits of shape (4, 4096, 128256). Cross-entropy
        loss against shifted targets. Backward pass computes gradients.
        After 8 micro-batches, clip gradients to norm 1.0, step optimizer,
        step scheduler, zero gradients. Monitor loss — if z-score > 5,
        rollback to last checkpoint. Save checkpoint every 1000 steps.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PretrainingConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Pretrainer.

        Args:
            model: The transformer language model to pretrain.
            config: PretrainingConfig with all hyperparameters.
            device: Target device. Defaults to CUDA if available.

        Explanation:
            Sets up everything needed for training:
            1. Move model to device
            2. Wrap in DDP if distributed
            3. Setup optimizer (AdamW with weight decay)
            4. Setup LR scheduler (cosine or WSD)
            5. Setup mixed precision (autocast + GradScaler)
            6. Setup loss spike detector
            7. Setup metrics tracker
            8. Print recipe summary
        """
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.global_step = 0
        self.best_loss = float("inf")
        self.last_good_checkpoint_path: Optional[str] = None

        # ------------------------------------------------------------------
        # Model setup
        # ------------------------------------------------------------------
        self.model = model.to(self.device)

        # Activation checkpointing
        if config.activation_checkpointing:
            self._apply_activation_checkpointing()

        # DDP wrapping (if distributed)
        if dist.is_initialized() and dist.get_world_size() > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )
            self.is_main_process = dist.get_rank() == 0
        else:
            self.is_main_process = True

        # torch.compile (optional)
        if config.compile_model:
            self.model = torch.compile(self.model)

        # ------------------------------------------------------------------
        # Optimizer setup
        # ------------------------------------------------------------------
        # Separate parameters that should/should not have weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Bias, LayerNorm/RMSNorm weights, and embedding should not decay
            if "bias" in name or "norm" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
        )

        # ------------------------------------------------------------------
        # Scheduler setup
        # ------------------------------------------------------------------
        if config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                min_lr_ratio=config.min_learning_rate / config.learning_rate,
            )
        elif config.scheduler_type == "wsd":
            self.scheduler = get_wsd_schedule_with_warmup(
                self.optimizer,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                decay_fraction=config.wsd_decay_fraction,
                min_lr_ratio=config.min_learning_rate / config.learning_rate,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")

        # ------------------------------------------------------------------
        # Mixed precision setup
        # ------------------------------------------------------------------
        self.use_autocast = config.precision in ("bf16", "fp16", "fp8")
        self.autocast_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "fp8": torch.bfloat16,  # FP8 uses BF16 as the wrapper dtype
        }.get(config.precision, torch.float32)

        # GradScaler for FP16 (BF16 and FP8 do not need it)
        self.scaler = GradScaler(enabled=(config.precision == "fp16"))

        # ------------------------------------------------------------------
        # Spike detection and metrics
        # ------------------------------------------------------------------
        self.spike_detector = LossSpikeDetector(
            threshold=config.spike_detection_threshold,
            window_size=config.spike_detection_window,
        )
        self.metrics_tracker = MetricsTracker(
            log_every_n_steps=config.log_every_n_steps,
        )

        # ------------------------------------------------------------------
        # W&B setup (optional)
        # ------------------------------------------------------------------
        self.wandb_run = None
        if config.use_wandb and self.is_main_process:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=vars(config),
                )
            except ImportError:
                logger.warning("wandb not installed. Skipping W&B logging.")

        # ------------------------------------------------------------------
        # Print recipe summary
        # ------------------------------------------------------------------
        if self.is_main_process:
            self._print_recipe()

    # ------------------------------------------------------------------
    # Activation checkpointing
    # ------------------------------------------------------------------

    def _apply_activation_checkpointing(self) -> None:
        """
        Apply activation checkpointing to transformer blocks.

        Activation checkpointing trades compute for memory: instead of
        storing intermediate activations during the forward pass, we
        recompute them during the backward pass. This reduces memory
        usage by ~60% at the cost of ~30% more compute.

        We checkpoint all transformer blocks except the last N layers
        (controlled by config.activation_checkpointing_omit_last_n),
        since the last layers are smaller and more frequently accessed.

        Interview Question:
            "What is activation checkpointing and when would you use it?"
            Activation checkpointing saves GPU memory by not storing
            intermediate activations during the forward pass. Instead,
            they are recomputed during the backward pass. This roughly
            halves memory usage at the cost of ~33% more compute (one
            extra forward pass per backward). Use it when your model
            doesn't fit in GPU memory even with the smallest batch size.
        """
        from torch.utils.checkpoint import checkpoint

        model = self.model.module if hasattr(self.model, "module") else self.model

        # Find transformer blocks (assuming they have a 'layers' attribute)
        if not hasattr(model, "layers"):
            logger.info("Model does not have 'layers' attribute. Skipping activation checkpointing.")
            return

        num_layers = len(model.layers)
        omit = self.config.activation_checkpointing_omit_last_n

        for i, layer in enumerate(model.layers):
            if i < num_layers - omit:
                # Wrap the forward method with checkpointing
                original_forward = layer.forward

                def make_checkpointed_forward(fwd):
                    def checkpointed_forward(*args, **kwargs):
                        return checkpoint(fwd, *args, use_reentrant=False, **kwargs)
                    return checkpointed_forward

                layer.forward = make_checkpointed_forward(original_forward)

        logger.info(
            f"Activation checkpointing applied to {num_layers - omit}/{num_layers} layers"
        )

    # ------------------------------------------------------------------
    # Recipe summary
    # ------------------------------------------------------------------

    def _print_recipe(self) -> None:
        """
        Print a full recipe summary at startup.

        This prints every configuration parameter so that the training
        run is fully reproducible from the logs alone. This is a
        critical practice for research reproducibility.

        Interview Question:
            "Why print a recipe summary at the start of training?"
            Reproducibility. If something goes wrong (or goes right!),
            you need to know exactly what configuration was used. The
            recipe summary, combined with the random seed, should be
            sufficient to reproduce the training run exactly.
        """
        c = self.config
        border = "=" * 70
        print(f"\n{border}")
        print("PRETRAINING RECIPE SUMMARY")
        print(border)
        print(f"  Model:            {c.model_name}")
        print(f"  Vocab size:       {c.vocab_size:,}")
        print(f"  d_model:          {c.d_model:,}")
        print(f"  num_layers:       {c.num_layers}")
        print(f"  num_heads:        {c.num_heads}")
        print(f"  Precision:        {c.precision}")
        print(f"  Scheduler:        {c.scheduler_type}")
        print(f"  Learning rate:    {c.learning_rate}")
        print(f"  Min LR:           {c.min_learning_rate}")
        print(f"  Weight decay:     {c.weight_decay}")
        print(f"  Beta1/Beta2:      {c.beta1}/{c.beta2}")
        print(f"  Grad clip norm:   {c.grad_clip_norm}")
        print(f"  Max steps:        {c.max_steps:,}")
        print(f"  Warmup steps:     {c.warmup_steps:,}")
        print(f"  Batch size:       {c.batch_size}")
        print(f"  Grad accum:       {c.gradient_accumulation_steps}")
        print(f"  Effective batch:  {c.effective_batch_size()}")
        print(f"  Sequence length:  {c.sequence_length}")
        print(f"  Tokens/step:      {c.tokens_per_step():,}")
        print(f"  Parallelism:      {c.parallelism}")
        print(f"  Act. checkpoint:  {c.activation_checkpointing}")
        print(f"  Spike threshold:  {c.spike_detection_threshold}")
        print(f"  Auto rollback:    {c.auto_rollback_on_spike}")
        print(f"  Checkpoint every: {c.checkpoint_every_n_steps} steps")
        print(f"  Log every:        {c.log_every_n_steps} steps")
        print(f"  Seed:             {c.seed}")
        print(f"  Seq len curriculum: {c.sequence_length_curriculum}")
        print(border)
        print(f"  Total tokens:     {c.max_steps * c.tokens_per_step():,}")
        print(border + "\n")

    # ------------------------------------------------------------------
    # Sequence length from curriculum
    # ------------------------------------------------------------------

    def _get_current_sequence_length(self) -> int:
        """
        Get the sequence length for the current training step.

        Returns:
            The sequence length to use for this step, based on the curriculum.

        Explanation:
            We walk through the curriculum list (sorted by step) and find
            the last entry whose step threshold is <= the current step.
            This allows gradual sequence length increases during training.
        """
        curriculum = sorted(self.config.sequence_length_curriculum, key=lambda x: x[0])
        current_seq_len = self.config.sequence_length
        for step_threshold, seq_len in curriculum:
            if self.global_step >= step_threshold:
                current_seq_len = seq_len
            else:
                break
        return current_seq_len

    # ------------------------------------------------------------------
    # Checkpoint save/load
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: Optional[str] = None) -> str:
        """
        Save a distributed checkpoint.

        Args:
            tag: Optional tag for the checkpoint directory name.
                 Defaults to 'step_{global_step}'.

        Returns:
            Path to the saved checkpoint directory.

        Explanation:
            Saves:
                - Model state dict (unwrapped from DDP if needed)
                - Optimizer state dict
                - Scheduler state dict
                - Global step
                - Best loss
                - RNG states (for exact resumption)
            For distributed training, each rank saves its own shard.
        """
        if tag is None:
            tag = f"step_{self.global_step}"

        checkpoint_dir = os.path.join(self.config.checkpoint_dir, tag)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Unwrap DDP if needed
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": vars(self.config),
            "torch_rng_state": torch.random.get_rng_state(),
        }

        if torch.cuda.is_available():
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

        save_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        torch.save(checkpoint, save_path)

        if self.is_main_process:
            logger.info(f"Checkpoint saved: {save_path}")

        # Cleanup old checkpoints
        if self.is_main_process:
            self._cleanup_old_checkpoints()

        return checkpoint_dir

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint and restore training state.

        Args:
            checkpoint_path: Path to the checkpoint directory or file.

        Explanation:
            Restores model weights, optimizer state, scheduler state,
            and RNG states so that training resumes exactly where it left off.
        """
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "checkpoint.pt")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]

        # Restore RNG states
        if "torch_rng_state" in checkpoint:
            torch.random.set_rng_state(checkpoint["torch_rng_state"])
        if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

        logger.info(f"Checkpoint loaded: {checkpoint_path} (step {self.global_step})")

    def _rollback(self) -> None:
        """
        Rollback to the last good checkpoint.

        Called when a loss spike is detected. Restores the model to the
        last known-good state and resets the spike detector.

        Explanation:
            Rollback is a safety mechanism. When a loss spike occurs (usually
            due to bad data or numerical instability), continuing training
            from the corrupted state wastes compute and may permanently
            damage the model. By rolling back, we recover to a clean state
            and can continue training with minimal disruption.
        """
        if self.last_good_checkpoint_path is None:
            logger.warning("No checkpoint available for rollback. Continuing from current state.")
            return

        logger.warning(
            f"ROLLING BACK to checkpoint: {self.last_good_checkpoint_path}"
        )
        self._load_checkpoint(self.last_good_checkpoint_path)
        self.spike_detector.reset()

    def _cleanup_old_checkpoints(self) -> None:
        """
        Remove old checkpoints, keeping only the most recent N.

        Explanation:
            Checkpoints are large (multiple GB each). Keeping all of them
            would quickly fill disk space. We keep the last N checkpoints
            plus the best checkpoint.
        """
        if not os.path.exists(self.config.checkpoint_dir):
            return

        checkpoints = sorted(
            [
                d
                for d in os.listdir(self.config.checkpoint_dir)
                if os.path.isdir(os.path.join(self.config.checkpoint_dir, d))
            ]
        )

        while len(checkpoints) > self.config.keep_last_n_checkpoints:
            oldest = checkpoints.pop(0)
            oldest_path = os.path.join(self.config.checkpoint_dir, oldest)
            import shutil
            shutil.rmtree(oldest_path)
            logger.info(f"Removed old checkpoint: {oldest_path}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """
        Log training metrics to console and optionally W&B.

        Args:
            step: The current training step.
            metrics: Dictionary of metric name -> value.

        Explanation:
            Logs to console in a formatted line. If W&B is enabled,
            also logs to W&B for dashboard visualization.
        """
        if not self.is_main_process:
            return

        lr = self.optimizer.param_groups[0]["lr"]
        current_seq_len = self._get_current_sequence_length()

        # Console logging
        log_parts = [
            f"step={step:>7d}",
            f"loss={metrics['loss']:.4f}",
            f"grad_norm={metrics['grad_norm']:.4f}",
            f"lr={lr:.2e}",
            f"tok/s={metrics['tokens_per_sec']:,.0f}",
            f"seq_len={current_seq_len}",
        ]
        logger.info(" | ".join(log_parts))

        # W&B logging
        if self.wandb_run is not None:
            import wandb
            wandb.log(
                {
                    "train/loss": metrics["loss"],
                    "train/grad_norm": metrics["grad_norm"],
                    "train/learning_rate": lr,
                    "train/tokens_per_sec": metrics["tokens_per_sec"],
                    "train/tokens_processed": metrics["tokens_processed"],
                    "train/sequence_length": current_seq_len,
                    "train/step": step,
                },
                step=step,
            )

    # ------------------------------------------------------------------
    # Batch size ramp
    # ------------------------------------------------------------------

    def _get_effective_micro_batch_size(self) -> int:
        """
        Get the effective micro-batch size for the current step.

        Returns:
            The micro-batch size, ramped from 1 to config.batch_size
            over config.batch_size_ramp_steps.

        Explanation:
            Starting with a large batch size can cause instability. By
            ramping from 1 to the target over the first N steps, we give
            the model time to stabilize before hitting full batch size.
        """
        if self.config.batch_size_ramp_steps <= 0:
            return self.config.batch_size

        progress = min(1.0, self.global_step / self.config.batch_size_ramp_steps)
        return max(1, int(progress * self.config.batch_size))

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, dataloader: Iterator[Dict[str, torch.Tensor]]) -> None:
        """
        Main pretraining loop.

        Args:
            dataloader: An iterator that yields batches of tokenized text.
                        Each batch should be a dict with at least:
                            - 'input_ids': torch.Tensor of shape (B, T)
                            - 'labels': torch.Tensor of shape (B, T)
                        Or just 'input_ids' — labels will be derived by shifting.

        Explanation:
            This is the heart of pretraining. The loop:
            1. Sets random seeds for reproducibility
            2. For each step, gets a batch from the dataloader
            3. Runs forward + backward with mixed precision
            4. Accumulates gradients over micro-batches
            5. Clips gradients and steps optimizer
            6. Monitors for loss spikes
            7. Logs metrics and saves checkpoints

        Example:
            >>> model = TransformerLM(config)
            >>> pretrainer = Pretrainer(model, PretrainingConfig())
            >>> pretrainer.train(my_dataloader)
        """
        # Set seeds
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        self.model.train()

        # Zero gradients to start
        self.optimizer.zero_grad()

        logger.info("Starting pretraining...")
        start_time = time.time()

        while self.global_step < self.config.max_steps:
            # --------------------------------------------------------------
            # Get current sequence length from curriculum
            # --------------------------------------------------------------
            current_seq_len = self._get_current_sequence_length()

            # --------------------------------------------------------------
            # Gradient accumulation loop
            # --------------------------------------------------------------
            accumulated_loss = 0.0

            for micro_step in range(self.config.gradient_accumulation_steps):
                # Get batch from dataloader
                try:
                    batch = next(iter(dataloader))
                except StopIteration:
                    logger.info("Dataloader exhausted. Restarting.")
                    break

                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids[:, 1:].contiguous())

                # Truncate to current sequence length if needed
                if input_ids.shape[1] > current_seq_len:
                    input_ids = input_ids[:, :current_seq_len]
                    if labels.shape[1] > current_seq_len:
                        labels = labels[:, :current_seq_len]

                # Forward pass with mixed precision
                with autocast(
                    device_type=self.device.type,
                    dtype=self.autocast_dtype,
                    enabled=self.use_autocast,
                ):
                    outputs = self.model(input_ids)
                    # Handle both tuple and tensor returns
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                    # Compute loss
                    # Shift logits and labels for next-token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous() if labels.shape[1] > 1 else labels

                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,  # Standard ignore index for padding
                    )

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                self.scaler.scale(scaled_loss).backward()

                accumulated_loss += loss.item()

            # --------------------------------------------------------------
            # Gradient clipping and optimizer step
            # --------------------------------------------------------------
            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)

            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm,
            )
            grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

            self.global_step += 1

            # Average loss over accumulation steps
            avg_loss = accumulated_loss / self.config.gradient_accumulation_steps

            # --------------------------------------------------------------
            # Loss spike detection
            # --------------------------------------------------------------
            is_spike = self.spike_detector.check(avg_loss)

            if is_spike and self.config.auto_rollback_on_spike:
                logger.warning(f"Loss spike at step {self.global_step}. Rolling back...")
                self._rollback()
                continue  # Retry from the rolled-back state

            # Track best loss and save "good" checkpoint
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss

            # --------------------------------------------------------------
            # Metrics tracking and logging
            # --------------------------------------------------------------
            num_tokens = input_ids.numel() * self.config.gradient_accumulation_steps
            self.metrics_tracker.update(avg_loss, grad_norm_value, num_tokens)

            if self.global_step % self.config.log_every_n_steps == 0:
                metrics = self.metrics_tracker.get_and_reset()
                if metrics:
                    self._log_metrics(self.global_step, metrics)

            # --------------------------------------------------------------
            # Checkpointing
            # --------------------------------------------------------------
            if self.global_step % self.config.checkpoint_every_n_steps == 0:
                self.last_good_checkpoint_path = self._save_checkpoint()

        # End of training
        total_time = time.time() - start_time
        if self.is_main_process:
            logger.info(f"Pretraining complete. Total time: {total_time:.1f}s")
            logger.info(f"Final loss: {avg_loss:.4f}")
            self._save_checkpoint(tag="final")


################################################################################
# SECTION 6: TESTING & DEMONSTRATION
################################################################################


def demonstrate_pretrainer():
    """
    Demonstrate the Pretrainer with a toy model and synthetic data.

    This creates a tiny transformer model and runs a few pretraining steps
    to verify that the training loop, spike detection, and logging all work.
    """
    print("=" * 70)
    print("PRETRAINER DEMONSTRATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Create a toy model
    # ------------------------------------------------------------------
    class ToyTransformerLM(nn.Module):
        """Minimal transformer for demonstration."""

        def __init__(self, vocab_size: int = 1000, d_model: int = 128, num_layers: int = 2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=4,
                    dim_feedforward=d_model * 4,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = self.embed(input_ids)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.lm_head(x)

    # ------------------------------------------------------------------
    # Create toy data
    # ------------------------------------------------------------------
    class SyntheticDataLoader:
        """Infinite synthetic data for demonstration."""

        def __init__(self, vocab_size: int = 1000, seq_len: int = 64, batch_size: int = 2):
            self.vocab_size = vocab_size
            self.seq_len = seq_len
            self.batch_size = batch_size

        def __iter__(self):
            return self

        def __next__(self) -> Dict[str, torch.Tensor]:
            input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
            return {"input_ids": input_ids}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    print("\n[1/4] Creating toy model and config...")
    model = ToyTransformerLM(vocab_size=1000, d_model=128, num_layers=2)

    config = PretrainingConfig(
        model_name="toy-llm",
        vocab_size=1000,
        d_model=128,
        num_layers=2,
        num_heads=4,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        max_steps=20,
        warmup_steps=5,
        batch_size=2,
        gradient_accumulation_steps=2,
        sequence_length=64,
        sequence_length_curriculum=[(0, 32), (10, 64)],
        checkpoint_every_n_steps=10,
        log_every_n_steps=5,
        spike_detection_threshold=5.0,
        checkpoint_dir="./demo_checkpoints",
        precision="fp32",  # Use fp32 for demo (no GPU required)
    )

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Run pretraining
    # ------------------------------------------------------------------
    print("\n[2/4] Initializing Pretrainer...")
    device = torch.device("cpu")
    pretrainer = Pretrainer(model, config, device=device)

    print("\n[3/4] Running pretraining loop (20 steps)...")
    dataloader = SyntheticDataLoader(vocab_size=1000, seq_len=64, batch_size=2)
    pretrainer.train(dataloader)

    # ------------------------------------------------------------------
    # Test spike detection independently
    # ------------------------------------------------------------------
    print("\n[4/4] Testing spike detection...")
    detector = LossSpikeDetector(threshold=3.0, window_size=50)

    # Normal losses
    import random
    random.seed(42)
    for i in range(30):
        loss = 2.0 + random.gauss(0, 0.1)
        is_spike = detector.check(loss)

    # Inject a spike
    is_spike = detector.check(100.0)
    print(f"  Spike detected for loss=100.0: {is_spike}")

    is_spike = detector.check(2.1)
    print(f"  Spike detected for loss=2.1: {is_spike}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    demonstrate_pretrainer()
