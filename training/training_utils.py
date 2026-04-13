"""
################################################################################
TRAINING UTILITIES — ESSENTIAL TOOLS FOR TRAINING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Training Utilities?
    Common tools and techniques used during model training:
    - Learning rate scheduling
    - Gradient clipping
    - Checkpointing
    - Logging
    - Early stopping

Interview Questions:
    Q: "What learning rate schedule should I use?"
    A: Warmup + cosine decay is standard for LLMs.
       Warmup: linearly increase from 0 to max_lr
       Decay: cosine decay from max_lr to min_lr

    Q: "Why clip gradients?"
    A: Prevents exploding gradients. Without clipping,
       large gradients can cause training instability.

################################################################################
"""

import numpy as np
from typing import Dict, Optional
import math

################################################################################
# SECTION 1: LEARNING RATE SCHEDULES
################################################################################

class CosineScheduleWithWarmup:
    """
    Cosine Learning Rate Schedule with Warmup
    ==========================================

    Standard schedule for LLM training.

    Phase 1 (Warmup): Linearly increase lr from 0 to max_lr
    Phase 2 (Decay): Cosine decay from max_lr to min_lr

    Formula:
        Warmup: lr = max_lr × (step / warmup_steps)
        Decay: lr = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × progress))

    Interview Questions:
        Q: "Why warmup?"
        A: At the start, weights are random. Large gradients with
           high lr can cause instability. Warmup gradually increases lr.
    """

    def __init__(
        self,
        max_lr: float = 3e-4,
        min_lr: float = 3e-5,
        warmup_steps: int = 2000,
        total_steps: int = 100000
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        if step < self.warmup_steps:
            # Warmup phase
            return self.max_lr * (step / self.warmup_steps)
        else:
            # Cosine decay phase
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


class LinearScheduleWithWarmup:
    """
    Linear Learning Rate Schedule with Warmup
    ===========================================

    Simpler alternative to cosine schedule.
    """

    def __init__(
        self,
        max_lr: float = 3e-4,
        warmup_steps: int = 2000,
        total_steps: int = 100000
    ):
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        if step < self.warmup_steps:
            return self.max_lr * (step / self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.max_lr * (1 - progress)


################################################################################
# SECTION 2: GRADIENT CLIPPING
################################################################################

class GradientClipper:
    """
    Gradient Clipping
    =================

    Clips gradients to prevent exploding gradients.

    Methods:
    1. Clip by norm: rescale if total norm > max_norm
    2. Clip by value: clamp each gradient value

    Interview Questions:
        Q: "Clip by norm vs clip by value?"
        A: Clip by norm preserves gradient direction.
           Clip by value changes direction but is simpler.
           Clip by norm is preferred.
    """

    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def clip_by_norm(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Clip gradients by global norm.

        Args:
            gradients: Dictionary of gradients

        Returns:
            clipped: Clipped gradients
        """
        # Compute total norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)

        # Clip if needed
        if total_norm > self.max_norm:
            clip_coef = self.max_norm / (total_norm + 1e-6)
            gradients = {k: v * clip_coef for k, v in gradients.items()}

        return gradients

    def clip_by_value(self, gradients: Dict[str, np.ndarray], clip_value: float = 1.0) -> Dict[str, np.ndarray]:
        """Clip gradients by value."""
        return {k: np.clip(v, -clip_value, clip_value) for k, v in gradients.items()}


################################################################################
# SECTION 3: EARLY STOPPING
################################################################################

class EarlyStopping:
    """
    Early Stopping
    ==============

    Stop training when validation loss stops improving.

    Interview Questions:
        Q: "How do you prevent overfitting?"
        A: Early stopping, regularization, dropout, data augmentation.
           Early stopping is the simplest approach.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def should_stop(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


################################################################################
# SECTION 4: EXPERIMENT TRACKER
################################################################################

class ExperimentTracker:
    """
    Experiment Tracker
    ==================

    Tracks training metrics for analysis.

    Interview Questions:
        Q: "What metrics should I track during training?"
        A: Loss, learning rate, gradient norm, validation metrics,
           throughput (tokens/sec), memory usage.
    """

    def __init__(self):
        self.metrics = {}

    def log(self, key: str, value: float, step: int):
        """Log a metric."""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append((step, value))

    def get_latest(self, key: str) -> Optional[float]:
        """Get latest value for a metric."""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1][1]
        return None

    def summary(self) -> Dict:
        """Get summary of all metrics."""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                vals = [v for _, v in values]
                summary[key] = {
                    'latest': vals[-1],
                    'mean': np.mean(vals),
                    'std': np.std(vals),
                    'min': np.min(vals),
                    'max': np.max(vals),
                }
        return summary


################################################################################
# SECTION 5: TESTING
################################################################################

def demonstrate_training_utils():
    """Demonstrate training utilities."""
    print("=" * 70)
    print("TRAINING UTILITIES DEMONSTRATION")
    print("=" * 70)

    # Learning rate schedule
    print("\n--- Learning Rate Schedule ---")
    schedule = CosineScheduleWithWarmup(
        max_lr=3e-4, min_lr=3e-5,
        warmup_steps=100, total_steps=1000
    )
    for step in [0, 50, 100, 500, 999]:
        lr = schedule.get_lr(step)
        print(f"Step {step}: lr={lr:.6f}")

    # Gradient clipping
    print("\n--- Gradient Clipping ---")
    clipper = GradientClipper(max_norm=1.0)
    grads = {'w1': np.random.randn(10, 10) * 10, 'w2': np.random.randn(10) * 5}
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
    print(f"Before clipping: norm={total_norm:.4f}")
    clipped = clipper.clip_by_norm(grads)
    total_norm = np.sqrt(sum(np.sum(g**2) for g in clipped.values()))
    print(f"After clipping: norm={total_norm:.4f}")

    # Early stopping
    print("\n--- Early Stopping ---")
    early_stop = EarlyStopping(patience=3, min_delta=0.01)
    losses = [1.0, 0.9, 0.85, 0.84, 0.83, 0.83, 0.83]
    for i, loss in enumerate(losses):
        should_stop = early_stop.should_stop(loss)
        print(f"Epoch {i}: loss={loss:.2f}, stop={should_stop}")

    # Experiment tracker
    print("\n--- Experiment Tracker ---")
    tracker = ExperimentTracker()
    for step in range(10):
        tracker.log('loss', 1.0 - step * 0.1, step)
        tracker.log('lr', 3e-4 * (1 - step/10), step)

    summary = tracker.summary()
    print(f"Loss: {summary['loss']}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_training_utils()
