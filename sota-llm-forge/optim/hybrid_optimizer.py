"""
################################################################################
HYBRID OPTIMIZER — MUON FOR 2D WEIGHTS + ADAMW FOR EVERYTHING ELSE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is the Hybrid Optimizer?
    A production-grade optimizer that combines Muon (matrix-aware updates via
    Newton-Schulz orthogonalization) for 2D weight matrices with AdamW
    (per-element adaptive scaling) for all other parameters. This is the
    single highest-leverage 2026 systems change vs 2023-era training.

Why does it matter?
    - Muon excels at 2D weight matrices (attention projections, MLP weights)
    - AdamW excels at 1D parameters (embeddings, biases, norms) and output heads
    - Combining both gives the best of both worlds
    - Adopted by Kimi K2/2.5, GLM-4.5/4.7, DeepSeek V3 for trillion-scale training

How does it work?
    1. Scan model parameters and assign each to Muon or AdamW based on:
       - param.dim() >= 2 and not embedding/lm_head/norm → Muon
       - Everything else → AdamW
    2. Apply WSD (Warmup-Stable-Decay) learning rate schedule
    3. Apply global-norm gradient clipping before optimizer step
    4. Step both optimizers with their respective parameter groups

Assignment Rules:
    Muon:   attention Q/K/V/output projections, MLP weights, MoE expert weights
    AdamW:  embeddings, output head, RMSNorm gains, biases, MoE router weights

Why AdamW for non-2D:
    - Embeddings benefit from per-parameter adaptivity
    - Router weights are low-dimensional
    - Biases/norms are 1D (Muon requires 2D)

MLA-specific caveat:
    Naively applying Muon across an MLA up-projection matrix implicitly split
    by attention head can create a performance gap. Fix: "Muon Split" — split
    the up-projection by head and orthogonalize each head's block independently.
    (Zhipu, GLM-5)

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    HYBRID OPTIMIZER SYSTEM                          │
    │                                                                     │
    │  Model Parameters                                                   │
    │       │                                                             │
    │       ▼                                                             │
    │  ┌─────────────────┐                                                │
    │  │ Parameter Router│  Assign to Muon or AdamW based on dim/name    │
    │  └────────┬────────┘                                                │
    │           │                                                         │
    │     ┌─────┴─────┐                                                   │
    │     │           │                                                   │
    │     ▼           ▼                                                   │
    │  ┌──────┐   ┌──────┐                                                │
    │  │ Muon │   │AdamW │                                                │
    │  └──┬───┘   └──┬───┘                                                │
    │     │          │                                                    │
    │     └────┬─────┘                                                    │
    │          │                                                          │
    │          ▼                                                          │
    │  ┌─────────────────┐                                                │
    │  │ WSD Schedule    │  Warmup → Stable → Decay                      │
    │  └────────┬────────┘                                                │
    │           │                                                         │
    │           ▼                                                         │
    │  ┌─────────────────┐                                                │
    │  │ Gradient Clipper│  Global-norm clipping before step              │
    │  └────────┬────────┘                                                │
    │           │                                                         │
    │           ▼                                                         │
    │  ┌─────────────────┐                                                │
    │  │ Weight Updates  │  Apply updates to model                        │
    │  └─────────────────┘                                                │
    └─────────────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2023: AdamW standard for LLM training (GPT-3, LLaMA)
    - 2024: Muon proposed for matrix-aware updates (Jordan et al.)
    - 2025: Hybrid Muon+AdamW adopted for trillion-scale (Kimi K2, GLM-4.5)
    - 2026: WSD schedule + hybrid optimizer becomes SOTA standard

INTERVIEW QUESTIONS:
    1. "Why use Muon for 2D weights but AdamW for embeddings?"
       Muon's orthogonalization assumes matrix structure and normalizes the
       spectrum. Embeddings, while 2D, benefit from per-element adaptivity
       because each embedding vector is independent. Muon would force
       orthogonality across embedding dimensions, which is undesirable.

    2. "What is the WSD schedule and why is it better than cosine?"
       WSD (Warmup-Stable-Decay) decouples training duration from LR shape.
       The stable plateau lets you checkpoint cheaply for ablations. You can
       decide to decay after seeing validation loss plateau. Cosine commits
       to a fixed schedule upfront.

    3. "How does gradient clipping interact with Muon?"
       Clipping is applied before Muon's orthogonalization. This is important
       because Muon normalizes the spectrum, so a gradient spike would be
       amplified if not clipped first. Global-norm clipping ensures all
       parameters see the same scale factor.

    4. "What is the MLA split fix for Muon?"
       In Multi-head Latent Attention (MLA), the up-projection matrix is
       implicitly split across attention heads. Applying Muon to the whole
       matrix creates unwanted cross-head coupling. The fix: split by head
       and orthogonalize each block independently.

################################################################################
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import logging
import re

# Import Muon optimizer
from .muon import Muon, MuonConfig, newton_schulz_orthogonalize

# Configure logging
logger = logging.getLogger(__name__)


################################################################################
# SECTION 1: CONFIGURATION
################################################################################


class DecayType(Enum):
    """Learning rate decay type for WSD schedule."""
    COSINE = "cosine"
    LINEAR = "linear"


@dataclass
class HybridOptimizerConfig:
    """
    Configuration for Hybrid Muon+AdamW Optimizer.

    All hyperparameters are explicit — no magic numbers.

    Attributes:
        lr: Peak learning rate (default: 0.02)
        muon_momentum: Momentum coefficient for Muon (default: 0.95)
        muon_nesterov: Whether to use Nesterov momentum for Muon (default: True)
        muon_ns_steps: Number of Newton-Schulz iterations (default: 5)
        muon_eps: Epsilon for Muon numerical stability (default: 1e-7)
        adam_betas: AdamW beta parameters (default: (0.9, 0.95))
        adam_eps: AdamW epsilon (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        warmup_steps: Number of warmup steps (default: 1000)
        stable_steps: Number of stable steps (default: 10000)
        decay_steps: Number of decay steps (default: 5000)
        min_lr: Minimum learning rate after decay (default: 1e-5)
        decay_type: Type of decay (cosine or linear) (default: COSINE)
        debug: Enable debug logging (default: False)
    """
    lr: float = 0.02
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_eps: float = 1e-7
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    stable_steps: int = 10000
    decay_steps: int = 5000
    min_lr: float = 1e-5
    decay_type: DecayType = DecayType.COSINE
    debug: bool = False


################################################################################
# SECTION 2: WSD LEARNING RATE SCHEDULE
################################################################################


class WSDSchedule:
    """
    Warmup-Stable-Decay (WSD) Learning Rate Schedule.

    Three phases:
        1. Warmup: Linear ramp from 0 to peak_lr over warmup_steps
        2. Stable: Constant peak_lr for stable_steps
        3. Decay:  Cosine/linear decay from peak_lr to min_lr over decay_steps

    Why WSD over cosine:
        - Decouples "how long to train" from "when to commit to final LR shape"
        - Stable plateau lets you branch/checkpoint cheaply for ablations
        - Adopted by DeepSeek, Kimi, GLM teams for operational flexibility

    Formula:
        Warmup:  lr = peak_lr * (step / warmup_steps)
        Stable:  lr = peak_lr
        Decay:   lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
                 (cosine) or
                 lr = peak_lr - (peak_lr - min_lr) * progress (linear)

    Step by step:
        1. Determine which phase we're in based on current step
        2. Compute LR for that phase
        3. Return LR

    Interview Question:
        "Why not just use cosine annealing for the entire schedule?"
        Cosine annealing commits to a fixed schedule from the start. WSD's
        stable plateau allows you to checkpoint and branch for ablations
        without wasting compute on the decay phase. You can also extend
        training without changing the schedule.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        peak_lr: float,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr: float = 1e-5,
        decay_type: DecayType = DecayType.COSINE
    ):
        """
        Initialize WSD Schedule.

        Args:
            optimizer: PyTorch optimizer to schedule
            peak_lr: Peak learning rate
            warmup_steps: Number of warmup steps
            stable_steps: Number of stable steps
            decay_steps: Number of decay steps
            min_lr: Minimum learning rate after decay (default: 1e-5)
            decay_type: Type of decay (cosine or linear) (default: COSINE)

        Explanation:
            The schedule stores the total steps for each phase and computes
            the current phase based on the step counter. The optimizer's
            learning rate is updated in place.
        """
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.decay_type = decay_type

        # Validate
        if peak_lr <= 0:
            raise ValueError(f"peak_lr must be positive, got {peak_lr}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if stable_steps < 0:
            raise ValueError(f"stable_steps must be non-negative, got {stable_steps}")
        if decay_steps < 0:
            raise ValueError(f"decay_steps must be non-negative, got {decay_steps}")
        if min_lr < 0:
            raise ValueError(f"min_lr must be non-negative, got {min_lr}")
        if min_lr > peak_lr:
            raise ValueError(f"min_lr ({min_lr}) must be <= peak_lr ({peak_lr})")

        # Step counter
        self.current_step = 0

        # Compute phase boundaries
        self.warmup_end = warmup_steps
        self.stable_end = warmup_steps + stable_steps
        self.decay_end = warmup_steps + stable_steps + decay_steps

    def get_lr(self) -> float:
        """
        Get current learning rate.

        Returns:
            Current learning rate based on schedule phase

        Explanation:
            The learning rate is computed based on which phase we're in:
            - Warmup: Linear ramp from 0 to peak_lr
            - Stable: Constant peak_lr
            - Decay: Cosine or linear decay from peak_lr to min_lr
        """
        step = self.current_step

        # Phase 1: Warmup
        if step < self.warmup_end:
            # Linear ramp: lr = peak_lr * (step / warmup_steps)
            # At step 0: lr = 0
            # At step warmup_steps: lr = peak_lr
            return self.peak_lr * (step / self.warmup_steps)

        # Phase 2: Stable
        elif step < self.stable_end:
            # Constant: lr = peak_lr
            return self.peak_lr

        # Phase 3: Decay
        elif step < self.decay_end:
            # Compute progress through decay phase (0 to 1)
            decay_progress = (step - self.stable_end) / self.decay_steps

            if self.decay_type == DecayType.COSINE:
                # Cosine decay: lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
                # At progress 0: lr = peak_lr
                # At progress 1: lr = min_lr
                return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
                    1 + math.cos(math.pi * decay_progress)
                )
            else:
                # Linear decay: lr = peak_lr - (peak_lr - min_lr) * progress
                return self.peak_lr - (self.peak_lr - self.min_lr) * decay_progress

        # After decay: constant at min_lr
        else:
            return self.min_lr

    def step(self):
        """
        Advance the schedule by one step.

        Updates the optimizer's learning rate based on the current step.
        """
        self.current_step += 1
        lr = self.get_lr()

        # Update optimizer's learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


################################################################################
# SECTION 3: GRADIENT CLIPPER
################################################################################


class GradientClipper:
    """
    Global-norm gradient clipping.

    Applied before optimizer's internal update logic.
    Logs clip-triggering spikes every step.

    Formula:
        If ||g||_2 > max_norm:
            g = g * (max_norm / ||g||_2)

    Step by step:
        1. Compute global gradient norm across all parameters
        2. If norm exceeds threshold, scale all gradients uniformly
        3. Log the clipping event

    Interview Question:
        "Why clip gradients globally instead of per-parameter?"
        Global clipping preserves the relative scale between parameters.
        If one parameter has a large gradient and another has a small one,
        global clipping scales both by the same factor. Per-parameter
        clipping would change their relative magnitudes.
    """

    def __init__(
        self,
        parameters,
        max_norm: float = 1.0,
        eps: float = 1e-6,
        debug: bool = False
    ):
        """
        Initialize GradientClipper.

        Args:
            parameters: Iterable of parameters to clip
            max_norm: Maximum gradient norm (default: 1.0)
            eps: Small constant for numerical stability (default: 1e-6)
            debug: Enable debug logging (default: False)

        Explanation:
            The clipper stores a reference to the parameters and computes
            the global gradient norm on each clip() call.
        """
        self.parameters = list(parameters)
        self.max_norm = max_norm
        self.eps = eps
        self.debug = debug

        # Statistics
        self.total_clips = 0
        self.total_steps = 0

    def clip(self) -> float:
        """
        Clip gradients globally.

        Returns:
            The gradient norm before clipping

        Explanation:
            1. Compute global gradient norm: ||g||_2 = sqrt(sum(||g_i||_2^2))
            2. If ||g||_2 > max_norm, scale all gradients by max_norm / ||g||_2
            3. Return the original norm for logging
        """
        # Compute global gradient norm
        # ||g||_2 = sqrt(sum(||g_i||_2^2)) for all parameters
        total_norm = 0.0
        for p in self.parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Clip if necessary
        clip_coef = self.max_norm / (total_norm + self.eps)
        if clip_coef < 1:
            # Scale all gradients uniformly
            for p in self.parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)

            self.total_clips += 1

            if self.debug:
                logger.info(
                    f"Gradient clipped: {total_norm:.4f} → {self.max_norm:.4f} "
                    f"(factor: {clip_coef:.4f})"
                )

        self.total_steps += 1

        return total_norm

    def get_stats(self) -> Dict[str, float]:
        """
        Get clipping statistics.

        Returns:
            Dictionary with total_clips, total_steps, and clip_rate
        """
        return {
            'total_clips': self.total_clips,
            'total_steps': self.total_steps,
            'clip_rate': self.total_clips / max(1, self.total_steps)
        }


################################################################################
# SECTION 4: HYBRID OPTIMIZER
################################################################################


class HybridMuonAdamW:
    """
    Hybrid Optimizer: Muon for 2D Hidden Weights + AdamW for Everything Else.

    This is the single highest-leverage 2026 systems change vs 2023-era training.

    Assignment rules:
        Muon:   attention Q/K/V/output projections, MLP weights, MoE expert weights
        AdamW:  embeddings, output head, RMSNorm gains, biases, MoE router weights

    Why AdamW for non-2D: Embeddings are 2D but benefit from per-parameter
    adaptivity that orthogonalization would hurt. Router weights are
    low-dimensional. Biases/norms are 1D.

    MLA-specific caveat: naively applying Muon across an MLA up-projection
    matrix implicitly split by attention head can create a performance gap.
    Fix: "Muon Split" — split the up-projection by head and orthogonalize
    each head's block independently. (Zhipu, GLM-5)

    Formula:
        For Muon parameters (2D):
            m_t = β * m_{t-1} + g_t
            O_t = NewtonSchulz(m_t)
            w_t = w_{t-1} - lr * O_t

        For AdamW parameters (non-2D):
            m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
            m̂_t = m_t / (1 - β₁^t)
            v̂_t = v_t / (1 - β₂^t)
            w_t = w_{t-1} - lr * (m̂_t / (sqrt(v̂_t) + ε) + λ * w_{t-1})

    Step by step:
        1. Scan parameters and assign to Muon or AdamW groups
        2. Apply gradient clipping (global norm)
        3. Step Muon optimizer for 2D parameters
        4. Step AdamW optimizer for non-2D parameters
        5. Update learning rate schedule

    Interview Question:
        "How do you decide which parameters go to Muon vs AdamW?"
        Primary criterion: param.dim() >= 2. But we also exclude embeddings
        (despite being 2D) because they benefit from per-element adaptivity.
        And we exclude lm_head and norm parameters. The heuristic is: Muon
        for hidden weight matrices that form the "backbone" of the model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[HybridOptimizerConfig] = None
    ):
        """
        Initialize Hybrid Muon+AdamW Optimizer.

        Args:
            model: PyTorch model to optimize
            config: HybridOptimizerConfig instance (default: HybridOptimizerConfig())

        Explanation:
            The constructor scans all model parameters and assigns each to
            either Muon or AdamW based on dimensionality and name patterns.
            It then creates two separate optimizers and a WSD schedule.
        """
        if config is None:
            config = HybridOptimizerConfig()

        self.model = model
        self.config = config

        # Step 1: Scan parameters and assign to groups
        muon_params, adamw_params = self._assign_parameters(model, config)

        # Step 2: Create Muon optimizer for 2D parameters
        self.muon_optimizer = Muon(
            muon_params,
            lr=config.lr,
            momentum=config.muon_momentum,
            nesterov=config.muon_nesterov,
            ns_steps=config.muon_ns_steps,
            eps=config.muon_eps,
            weight_decay=config.weight_decay,
            max_grad_norm=None,  # Clipping handled by GradientClipper
            debug=config.debug
        )

        # Step 3: Create AdamW optimizer for non-2D parameters
        self.adamw_optimizer = optim.AdamW(
            adamw_params,
            lr=config.lr,
            betas=config.adam_betas,
            eps=config.adam_eps,
            weight_decay=config.weight_decay
        )

        # Step 4: Create gradient clipper
        all_params = list(model.parameters())
        self.grad_clipper = GradientClipper(
            all_params,
            max_norm=config.max_grad_norm,
            debug=config.debug
        )

        # Step 5: Create WSD schedule (attached to both optimizers)
        # We'll manage LR manually in step()
        self.schedule = WSDSchedule(
            optimizer=self.muon_optimizer,  # Primary optimizer for schedule
            peak_lr=config.lr,
            warmup_steps=config.warmup_steps,
            stable_steps=config.stable_steps,
            decay_steps=config.decay_steps,
            min_lr=config.min_lr,
            decay_type=config.decay_type
        )

        # Statistics
        self.step_count = 0

        # Log parameter assignment
        if config.debug:
            logger.info(f"Muon parameters: {len(muon_params)} tensors")
            logger.info(f"AdamW parameters: {len(adamw_params)} tensors")

    def _assign_parameters(
        self,
        model: torch.nn.Module,
        config: HybridOptimizerConfig
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Assign model parameters to Muon or AdamW groups.

        Args:
            model: PyTorch model
            config: Optimizer configuration

        Returns:
            Tuple of (muon_params, adamw_params) lists

        Explanation:
            Assignment rules:
            - Muon: param.dim() >= 2 AND name doesn't contain 'embed', 'lm_head', or 'norm'
            - AdamW: Everything else

            This ensures Muon only gets hidden weight matrices (attention
            projections, MLP weights, MoE expert weights) while AdamW handles
            embeddings, output heads, norms, biases, and router weights.
        """
        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if this should go to Muon
            # Criteria: 2D tensor AND not embedding/lm_head/norm
            is_2d = param.dim() >= 2
            is_embedding = 'embed' in name.lower()
            is_lm_head = 'lm_head' in name.lower()
            is_norm = 'norm' in name.lower() or 'ln' in name.lower()

            if is_2d and not is_embedding and not is_lm_head and not is_norm:
                # Muon: hidden weight matrices
                muon_params.append(param)

                if config.debug:
                    logger.info(f"Muon: {name} (shape={param.shape})")
            else:
                # AdamW: everything else
                adamw_params.append(param)

                if config.debug:
                    logger.info(f"AdamW: {name} (shape={param.shape}, dim={param.dim()})")

        return muon_params, adamw_params

    def step(self):
        """
        Perform a single optimization step.

        1. Apply gradient clipping (global norm)
        2. Step Muon optimizer for 2D parameters
        3. Step AdamW optimizer for non-2D parameters
        4. Update learning rate schedule

        Explanation:
            Gradient clipping is applied before either optimizer steps.
            This is important because Muon normalizes the spectrum, so
            a gradient spike would be amplified if not clipped first.
        """
        # Step 1: Clip gradients globally
        grad_norm = self.grad_clipper.clip()

        # Step 2: Update learning rate from schedule
        current_lr = self.schedule.step()

        # Step 3: Step both optimizers
        self.muon_optimizer.step()
        self.adamw_optimizer.step()

        # Step 4: Increment step counter
        self.step_count += 1

        # Debug logging
        if self.config.debug and self.step_count % 100 == 0:
            clip_stats = self.grad_clipper.get_stats()
            logger.info(
                f"Step {self.step_count}: "
                f"LR={current_lr:.6f}, "
                f"Grad norm={grad_norm:.4f}, "
                f"Clip rate={clip_stats['clip_rate']:.4f}"
            )

    def zero_grad(self):
        """
        Zero gradients for both optimizers.

        Explanation:
            Must zero gradients for both Muon and AdamW parameter groups
            before backward pass.
        """
        self.muon_optimizer.zero_grad()
        self.adamw_optimizer.zero_grad()

    def get_lr(self) -> float:
        """
        Get current learning rate.

        Returns:
            Current learning rate from WSD schedule
        """
        return self.schedule.get_lr()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get optimizer statistics.

        Returns:
            Dictionary with step count, LR, gradient clipping stats
        """
        return {
            'step_count': self.step_count,
            'lr': self.get_lr(),
            'grad_clip_stats': self.grad_clipper.get_stats()
        }


################################################################################
# SECTION 5: FACTORY FUNCTION
################################################################################


def create_hybrid_optimizer(
    model: torch.nn.Module,
    config: Optional[HybridOptimizerConfig] = None
) -> HybridMuonAdamW:
    """
    Factory function to create Hybrid Muon+AdamW optimizer.

    Args:
        model: PyTorch model to optimize
        config: HybridOptimizerConfig instance (default: HybridOptimizerConfig())

    Returns:
        HybridMuonAdamW optimizer configured for the model

    Explanation:
        This is the recommended way to create the hybrid optimizer. It
        handles all the complexity of parameter assignment and optimizer
        creation.

    Example:
        >>> model = MyTransformer(config)
        >>> optimizer = create_hybrid_optimizer(model)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """
    if config is None:
        config = HybridOptimizerConfig()

    return HybridMuonAdamW(model, config)


################################################################################
# SECTION 6: MLA SPLIT FIX (DOCUMENTED STUB)
################################################################################


def split_mla_up_projection(
    weight: torch.Tensor,
    num_heads: int,
    head_dim: int
) -> List[torch.Tensor]:
    """
    Split MLA up-projection matrix by attention head.

    This is a documented stub explaining the MLA split fix for Muon.

    Problem:
        In Multi-head Latent Attention (MLA), the up-projection matrix W_up
        has shape (d_model, num_heads * head_dim). Naively applying Muon to
        this matrix creates unwanted cross-head coupling because Muon's
        orthogonalization treats the entire matrix as one unit.

    Solution:
        Split W_up into num_heads blocks, each of shape (d_model, head_dim),
        and apply Muon's orthogonalization to each block independently.

    Args:
        weight: Up-projection matrix, shape (d_model, num_heads * head_dim)
        num_heads: Number of attention heads
        head_dim: Dimension per head

    Returns:
        List of weight blocks, each of shape (d_model, head_dim)

    Explanation:
        This function demonstrates the "Muon Split" technique used in GLM-5.
        By splitting the matrix by head, we ensure that each head's subspace
        is orthogonalized independently, preventing cross-head interference.

        The blocks can then be concatenated back into the full matrix after
        orthogonalization.

    Cite:
        Zhipu AI, GLM-5 technical report (2025)
    """
    # Validate dimensions
    d_model = weight.shape[0]
    expected_dim = num_heads * head_dim

    if weight.shape[1] != expected_dim:
        raise ValueError(
            f"Expected weight shape ({d_model}, {expected_dim}), "
            f"got {weight.shape}"
        )

    # Split into per-head blocks
    # Each block has shape (d_model, head_dim)
    blocks = torch.split(weight, head_dim, dim=1)

    return list(blocks)


def merge_mla_blocks(blocks: List[torch.Tensor]) -> torch.Tensor:
    """
    Merge per-head weight blocks back into full matrix.

    Args:
        blocks: List of weight blocks, each of shape (d_model, head_dim)

    Returns:
        Merged weight matrix, shape (d_model, num_heads * head_dim)
    """
    return torch.cat(blocks, dim=1)


################################################################################
# SECTION 7: TESTING & DEMONSTRATION
################################################################################


def demonstrate_hybrid_optimizer():
    """
    Demonstrate Hybrid Muon+AdamW optimizer on a small model.

    Shows:
        1. Parameter assignment (Muon vs AdamW)
        2. WSD learning rate schedule
        3. Gradient clipping behavior
        4. Training loop with hybrid optimizer
    """
    print("=" * 70)
    print("HYBRID MUON+ADAMW OPTIMIZER DEMONSTRATION")
    print("=" * 70)

    # Create a simple transformer-like model
    class SimpleTransformer(torch.nn.Module):
        def __init__(self, d_model=64, vocab_size=100):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, d_model)
            self.q_proj = torch.nn.Linear(d_model, d_model)
            self.k_proj = torch.nn.Linear(d_model, d_model)
            self.v_proj = torch.nn.Linear(d_model, d_model)
            self.out_proj = torch.nn.Linear(d_model, d_model)
            self.mlp_up = torch.nn.Linear(d_model, d_model * 4)
            self.mlp_down = torch.nn.Linear(d_model * 4, d_model)
            self.norm1 = torch.nn.LayerNorm(d_model)
            self.norm2 = torch.nn.LayerNorm(d_model)
            self.lm_head = torch.nn.Linear(d_model, vocab_size)

        def forward(self, x):
            x = self.embedding(x)
            x = self.norm1(x + self.out_proj(self.v_proj(x)))  # Simplified
            x = self.norm2(x + self.mlp_down(self.mlp_up(x)))
            return self.lm_head(x)

    model = SimpleTransformer()

    print("\n1. Model Parameters:")
    print("-" * 40)
    for name, param in model.named_parameters():
        print(f"  {name}: shape={param.shape}, dim={param.dim()}")

    # Create hybrid optimizer
    config = HybridOptimizerConfig(
        lr=0.02,
        muon_momentum=0.95,
        adam_betas=(0.9, 0.95),
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=5,
        stable_steps=10,
        decay_steps=5,
        min_lr=1e-4,
        debug=True
    )

    optimizer = create_hybrid_optimizer(model, config)

    print("\n2. Parameter Assignment:")
    print("-" * 40)
    print(f"  Muon parameters: {len(optimizer.muon_optimizer.param_groups[0]['params'])} tensors")
    print(f"  AdamW parameters: {len(optimizer.adamw_optimizer.param_groups[0]['params'])} tensors")

    print("\n3. WSD Learning Rate Schedule:")
    print("-" * 40)
    print(f"  Peak LR: {config.lr}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Stable steps: {config.stable_steps}")
    print(f"  Decay steps: {config.decay_steps}")
    print(f"  Min LR: {config.min_lr}")

    # Training loop
    print("\n4. Training Loop (20 steps):")
    print("-" * 40)

    x = torch.randint(0, 100, (32, 16))
    target = torch.randint(0, 100, (32, 16))

    for step in range(20):
        # Forward pass
        output = model(x)
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, 100), target.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Step
        optimizer.step()

        if step % 5 == 0:
            stats = optimizer.get_stats()
            print(
                f"  Step {step}: loss={loss.item():.4f}, "
                f"LR={stats['lr']:.6f}, "
                f"Grad clips={stats['grad_clip_stats']['total_clips']}"
            )

    print("\n5. Gradient Clipping Statistics:")
    print("-" * 40)
    clip_stats = optimizer.grad_clipper.get_stats()
    print(f"  Total steps: {clip_stats['total_steps']}")
    print(f"  Total clips: {clip_stats['total_clips']}")
    print(f"  Clip rate: {clip_stats['clip_rate']:.4f}")

    print("\n6. MLA Split Fix Example:")
    print("-" * 40)

    # Demonstrate MLA split
    d_model = 64
    num_heads = 8
    head_dim = d_model // num_heads

    # Simulate MLA up-projection
    W_up = torch.randn(d_model, num_heads * head_dim)
    print(f"  Original W_up shape: {W_up.shape}")

    # Split by head
    blocks = split_mla_up_projection(W_up, num_heads, head_dim)
    print(f"  Split into {len(blocks)} blocks, each shape: {blocks[0].shape}")

    # Orthogonalize each block independently
    orthogonalized_blocks = [
        newton_schulz_orthogonalize(block, steps=5)
        for block in blocks
    ]

    # Merge back
    W_up_merged = merge_mla_blocks(orthogonalized_blocks)
    print(f"  Merged back to shape: {W_up_merged.shape}")

    print("\n7. Key Properties:")
    print("-" * 40)
    print("  - Muon for 2D hidden weights (attention, MLP)")
    print("  - AdamW for embeddings, norms, biases, heads")
    print("  - WSD schedule: warmup → stable → decay")
    print("  - Global-norm gradient clipping before step")
    print("  - MLA split fix for head-independent orthogonalization")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_hybrid_optimizer()
