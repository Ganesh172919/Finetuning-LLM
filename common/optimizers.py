"""
################################################################################
OPTIMIZERS — TRAINING ALGORITHMS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Optimizers?
    Algorithms that update model weights to minimize loss.

Key Optimizers:
    - SGD: Stochastic Gradient Descent
    - Adam: Adaptive Moment Estimation
    - AdamW: Adam with Weight Decay

Interview Questions:
    Q: "What optimizer should I use for LLMs?"
    A: AdamW is the standard. Fast, stable, handles weight decay well.

################################################################################
"""

import numpy as np
from typing import Dict

################################################################################
# SECTION 1: OPTIMIZERS
################################################################################

class AdamW:
    """
    AdamW Optimizer
    ===============

    Adam with decoupled weight decay.

    Update rule:
        m = β1 * m + (1-β1) * grad
        v = β2 * v + (1-β2) * grad²
        m̂ = m / (1-β1^t)
        v̂ = v / (1-β2^t)
        w = w - lr * (m̂ / (√v̂ + ε) + λ * w)

    Interview Questions:
        Q: "What's the difference between Adam and AdamW?"
        A: AdamW decouples weight decay from gradient update.
           Better generalization for LLMs.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """Update parameters."""
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

            # Update
            params[name] -= self.lr * (
                m_hat / (np.sqrt(v_hat) + self.eps) +
                self.weight_decay * params[name]
            )


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_optimizers():
    """Demonstrate optimizers."""
    print("=" * 70)
    print("OPTIMIZERS DEMONSTRATION")
    print("=" * 70)

    # AdamW
    print("\n--- AdamW ---")
    optimizer = AdamW(lr=1e-3, weight_decay=0.01)

    params = {'w': np.random.randn(10, 10)}
    grads = {'w': np.random.randn(10, 10) * 0.1}

    for step in range(5):
        optimizer.step(params, grads)
        print(f"Step {step+1}: norm={np.linalg.norm(params['w']):.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_optimizers()
