"""
################################################################################
OPTIMIZATION — TRAINING NEURAL NETWORKS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Optimization?
    Optimization is the process of adjusting model parameters to minimize
    the loss function. It's how models learn from data.

Why does it matter?
    Without optimization, models can't learn. The choice of optimizer
    affects:
    - Training speed
    - Model quality
    - Convergence stability
    - Memory usage

Key Algorithms:
    1. SGD: Simple but slow
    2. Adam: Fast and stable (most popular)
    3. AdamW: Adam with weight decay (modern standard)

Interview Questions:
    1. "What optimizer should I use?"
       AdamW is the standard for LLM training.
       It's fast, stable, and handles weight decay well.

    2. "What's the learning rate?"
       Controls how much weights change per step.
       Too high: diverges. Too low: slow convergence.
       Use warmup + cosine decay schedule.

    3. "What is gradient clipping?"
       Limits gradient magnitude to prevent exploding gradients.
       Essential for stable LLM training.

################################################################################
"""

import numpy as np
from typing import List, Dict

################################################################################
# SECTION 1: SGD
################################################################################

class SGD:
    """
    Stochastic Gradient Descent
    ============================

    The simplest optimizer: update weights in the direction of negative gradient.

    Update rule:
        w = w - lr * gradient

    With momentum:
        v = β * v + gradient
        w = w - lr * v

    Momentum helps:
    - Accelerate in consistent directions
    - Dampen oscillations
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """Update parameters."""
        for name in params:
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(params[name])

            # Update velocity
            self.velocity[name] = self.momentum * self.velocity[name] + grads[name]

            # Update parameters
            params[name] = params[name] - self.lr * self.velocity[name]


################################################################################
# SECTION 2: ADAM
################################################################################

class Adam:
    """
    Adam Optimizer
    ==============

    Definition: Adaptive Moment Estimation.
    Combines momentum (first moment) and RMSprop (second moment).

    Update rule:
        m = β1 * m + (1 - β1) * gradient        (first moment)
        v = β2 * v + (1 - β2) * gradient²        (second moment)
        m̂ = m / (1 - β1^t)                       (bias correction)
        v̂ = v / (1 - β2^t)                       (bias correction)
        w = w - lr * m̂ / (√v̂ + ε)

    Why Adam?
    - Adaptive learning rates per parameter
    - Fast convergence
    - Works well in practice
    - Default for most deep learning

    Interview Question:
        "What is Adam and why is it popular?"
        Adam combines momentum and adaptive learning rates.
        It maintains running averages of gradients (first moment)
        and squared gradients (second moment). This gives each
        parameter its own adaptive learning rate.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m: Dict[str, np.ndarray] = {}  # First moment
        self.v: Dict[str, np.ndarray] = {}  # Second moment

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """Update parameters using Adam."""
        self.t += 1

        for name in params:
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])

            # Update moments
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grads[name] ** 2

            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Update parameters
            params[name] = params[name] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


################################################################################
# SECTION 3: ADAMW
################################################################################

class AdamW:
    """
    AdamW Optimizer
    ===============

    Adam with decoupled weight decay.
    This is the standard optimizer for LLM training.

    Difference from Adam:
    - Adam: weight decay is coupled with gradient
    - AdamW: weight decay is applied directly to weights

    Update rule:
        m = β1 * m + (1 - β1) * gradient
        v = β2 * v + (1 - β2) * gradient²
        w = w - lr * (m̂ / (√v̂ + ε) + λ * w)

    Why AdamW?
    - Better generalization than Adam
    - Proper weight decay behavior
    - Used by GPT, LLaMA, Mistral, etc.

    Interview Question:
        "What's the difference between Adam and AdamW?"
        AdamW decouples weight decay from the gradient update.
        This gives better generalization and is the standard
        for LLM training.
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0

        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """Update parameters using AdamW."""
        self.t += 1

        for name in params:
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])

            # Update moments
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grads[name] ** 2

            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Update with weight decay
            params[name] = params[name] - self.lr * (
                m_hat / (np.sqrt(v_hat) + self.epsilon) +
                self.weight_decay * params[name]
            )


################################################################################
# SECTION 4: LEARNING RATE SCHEDULES
################################################################################

class CosineSchedule:
    """
    Cosine Learning Rate Schedule
    =============================

    Learning rate follows a cosine curve:
    - Warmup: linearly increase from 0 to lr
    - Decay: cosine decay from lr to 0.1 * lr

    Formula:
        lr(t) = lr_max * 0.5 * (1 + cos(π * t / T))

    This is the standard schedule for LLM training.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 2000,
        total_steps: int = 100000,
        min_lr_ratio: float = 0.1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.lr

    def step(self):
        """Update learning rate."""
        t = self.optimizer.t

        if t < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (t / self.warmup_steps)
        else:
            # Cosine decay
            progress = (t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr * (
                self.min_lr_ratio +
                (1 - self.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
            )

        self.optimizer.lr = lr
        return lr


################################################################################
# SECTION 5: TESTING & EXAMPLES
################################################################################

def demonstrate_optimization():
    """Demonstrate optimization concepts."""
    print("=" * 70)
    print("OPTIMIZATION DEMONSTRATION")
    print("=" * 70)

    # Create dummy parameters and gradients
    params = {
        'weight': np.random.randn(10, 10),
        'bias': np.zeros(10)
    }
    grads = {
        'weight': np.random.randn(10, 10) * 0.1,
        'bias': np.ones(10) * 0.1
    }

    # SGD
    print("\n--- SGD ---")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    sgd.step(params, grads)
    print(f"Updated weight norm: {np.linalg.norm(params['weight']):.4f}")

    # Adam
    print("\n--- Adam ---")
    adam = Adam(learning_rate=1e-3)
    adam.step(params, grads)
    print(f"Updated weight norm: {np.linalg.norm(params['weight']):.4f}")

    # AdamW
    print("\n--- AdamW ---")
    adamw = AdamW(learning_rate=1e-4, weight_decay=0.01)
    adamw.step(params, grads)
    print(f"Updated weight norm: {np.linalg.norm(params['weight']):.4f}")

    # Learning rate schedule
    print("\n--- Cosine Schedule ---")
    schedule = CosineSchedule(adamw, warmup_steps=100, total_steps=1000)
    for step in [0, 50, 100, 500, 999]:
        adamw.t = step
        lr = schedule.step()
        print(f"Step {step}: lr={lr:.6f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_optimization()


################################################################################
# REFERENCES
################################################################################

# [1] Robbins, H., & Monro, S. (1951). A Stochastic Approximation Method.
# [2] Kingma, D.P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization.
# [3] Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization.

################################################################################
