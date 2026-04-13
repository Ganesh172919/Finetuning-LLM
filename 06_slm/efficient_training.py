"""
################################################################################
EFFICIENT TRAINING — MIXED PRECISION, GRADIENT CHECKPOINTING, WSD SCHEDULE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Efficient Training?
    Techniques to train models faster and with less memory:
    - Mixed Precision: FP16/BF16 for speed, FP32 for stability
    - Gradient Checkpointing: Trade compute for memory
    - Gradient Accumulation: Simulate large batches
    - Learning Rate Schedules: Warmup + Cosine/WSD decay

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


################################################################################
# SECTION 1: MIXED PRECISION
################################################################################

class MixedPrecision:
    """
    Mixed Precision Training — FP16/BF16 with FP32 master weights.

    Key idea: do most computation in FP16 (fast), keep master weights
    in FP32 (stable), use loss scaling to prevent underflow.

    Interview Question:
        "How does mixed precision training work?"
        (1) Store master weights in FP32, (2) Cast to FP16 for forward/backward,
        (3) Scale loss up to prevent gradient underflow, (4) Unscale gradients
        before FP32 update. Result: 2x speed, same quality.
    """

    def __init__(self, scale_factor: float = 65536.0):
        self.scale_factor = scale_factor
        self.found_overflow = False

    def scale_loss(self, loss: float) -> float:
        """Scale loss up to prevent gradient underflow."""
        return loss * self.scale_factor

    def unscale_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Unscale gradients back to normal range."""
        return gradients / self.scale_factor

    def check_overflow(self, gradients: np.ndarray) -> bool:
        """Check for Inf/NaN in gradients."""
        return bool(np.any(np.isinf(gradients)) or np.any(np.isnan(gradients)))

    def update_scale(self, overflow: bool):
        """Dynamic loss scaling: halve on overflow, double otherwise."""
        if overflow:
            self.scale_factor /= 2
            self.found_overflow = True
        else:
            self.scale_factor = min(self.scale_factor * 2, 65536.0)


################################################################################
# SECTION 2: GRADIENT CHECKPOINTING
################################################################################

class GradientCheckpointing:
    """
    Gradient Checkpointing — Trade compute for memory.

    Instead of storing all activations for backprop, recompute them.
    Memory: O(sqrt(N)) instead of O(N), compute: 33% more.

    Interview Question:
        "What is gradient checkpointing?"
        During forward pass, don't save intermediate activations.
        During backward pass, recompute them from checkpoints.
        Saves ~60% memory at cost of ~33% more compute.
    """

    def __init__(self, checkpoint_every: int = 2):
        self.checkpoint_every = checkpoint_every

    def should_checkpoint(self, layer_idx: int) -> bool:
        """Whether to checkpoint this layer."""
        return layer_idx % self.checkpoint_every == 0


################################################################################
# SECTION 3: GRADIENT ACCUMULATION
################################################################################

class GradientAccumulation:
    """
    Simulate large batches with small memory.

    Accumulate gradients over N micro-batches before updating.

    Formula: grad_total = (1/N) * sum(grad_i)

    Interview Question:
        "How does gradient accumulation work?"
        Split a large batch into N micro-batches. Compute gradients
        for each micro-batch and accumulate (sum) them. After N steps,
        divide by N and update weights. Simulates large batch training
        with small GPU memory.
    """

    def __init__(self, n_accumulation_steps: int = 4):
        self.n_steps = n_accumulation_steps
        self.accumulated_gradients = None
        self.step_count = 0

    def accumulate(self, gradients: np.ndarray) -> bool:
        """
        Accumulate gradients. Returns True when ready to update.

        Args:
            gradients: Gradients from current micro-batch

        Returns:
            True if accumulated enough steps
        """
        if self.accumulated_gradients is None:
            self.accumulated_gradients = gradients.copy()
        else:
            self.accumulated_gradients += gradients
        self.step_count += 1

        return self.step_count >= self.n_steps

    def get_gradients(self) -> np.ndarray:
        """Get averaged gradients and reset."""
        grads = self.accumulated_gradients / self.n_steps
        self.accumulated_gradients = None
        self.step_count = 0
        return grads


################################################################################
# SECTION 4: LEARNING RATE SCHEDULES
################################################################################

class LearningRateScheduler:
    """
    Learning Rate Schedules — Warmup + Decay.

    WSD (Warmup-Stable-Decay) is the 2025 SOTA schedule.

    Interview Question:
        "What learning rate schedule should I use?"
        (1) Linear warmup: 0 → lr_max over warmup_steps,
        (2) Cosine decay: lr_max → lr_min over remaining steps,
        (3) WSD: warmup → stable → fast decay (newest SOTA).
    """

    def __init__(self, lr_max: float = 3e-4, lr_min: float = 1e-5,
                 warmup_steps: int = 1000, total_steps: int = 100000):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def linear_warmup(self, step: int) -> float:
        """Linear warmup from 0 to lr_max."""
        return self.lr_max * min(step / self.warmup_steps, 1.0)

    def cosine_decay(self, step: int) -> float:
        """Cosine decay from lr_max to lr_min."""
        if step < self.warmup_steps:
            return self.linear_warmup(step)
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))

    def wsd(self, step: int, stable_ratio: float = 0.8, decay_ratio: float = 0.1) -> float:
        """
        Warmup-Stable-Decay schedule (2025 SOTA).

        Args:
            step: Current step
            stable_ratio: Fraction of steps in stable phase
            decay_ratio: Fraction of steps in decay phase
        """
        warmup_end = self.warmup_steps
        stable_end = int(self.total_steps * stable_ratio)
        decay_end = int(self.total_steps * (stable_ratio + decay_ratio))

        if step < warmup_end:
            return self.linear_warmup(step)
        elif step < stable_end:
            return self.lr_max
        elif step < decay_end:
            progress = (step - stable_end) / (decay_end - stable_end)
            return self.lr_max * (1 - progress) + self.lr_min * progress
        else:
            return self.lr_min


################################################################################
# SECTION 5: DEMONSTRATION
################################################################################

def demonstrate_efficient_training():
    """Demonstrate efficient training techniques."""
    print("=" * 70)
    print("EFFICIENT TRAINING DEMONSTRATION")
    print("=" * 70)

    # Mixed Precision
    print("\n1. MIXED PRECISION")
    print("-" * 40)
    mp = MixedPrecision()
    loss = 2.5
    scaled = mp.scale_loss(loss)
    print(f"  Original loss: {loss}")
    print(f"  Scaled loss: {scaled}")
    print(f"  Scale factor: {mp.scale_factor}")

    # Gradient Checkpointing
    print("\n2. GRADIENT CHECKPOINTING")
    print("-" * 40)
    gc = GradientCheckpointing(checkpoint_every=3)
    for i in range(10):
        cp = gc.should_checkpoint(i)
        print(f"  Layer {i}: checkpoint={cp}")

    # Gradient Accumulation
    print("\n3. GRADIENT ACCUMULATION")
    print("-" * 40)
    ga = GradientAccumulation(n_accumulation_steps=4)
    for step in range(8):
        grads = np.random.randn(10)
        ready = ga.accumulate(grads)
        if ready:
            final_grads = ga.get_gradients()
            print(f"  Step {step}: UPDATE (grad norm: {np.linalg.norm(final_grads):.4f})")
        else:
            print(f"  Step {step}: accumulating ({ga.step_count}/4)")

    # LR Schedules
    print("\n4. LEARNING RATE SCHEDULES")
    print("-" * 40)
    sched = LearningRateScheduler(lr_max=3e-4, warmup_steps=1000, total_steps=100000)
    for step in [0, 500, 1000, 50000, 80000, 95000, 100000]:
        cosine = sched.cosine_decay(step)
        wsd = sched.wsd(step)
        print(f"  Step {step:6d}: cosine={cosine:.6f}, WSD={wsd:.6f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_efficient_training()
