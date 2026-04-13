"""
GaLore: Gradient Low-Rank Projection
======================================

GaLore projects gradients into a low-rank subspace before updating weights.
This allows training with full-rank weights while using low-rank optimization.

Key Insight:
  LoRA constrains weights to be low-rank.
  GaLore constrains gradients to be low-rank (weights stay full-rank).

  This means:
  - Full model capacity preserved (full-rank weights)
  - Low memory optimizer states (low-rank gradients)
  - Best of both worlds!

How it works:
  1. Compute full gradient G
  2. Project: G_low = P @ G @ Q (using random projections)
  3. Update low-rank optimizer states
  4. Recover: G_approx = P.T @ G_low @ Q.T
  5. Update full weights with approximated gradient

Memory savings:
  - Adam: 2 × d_model × d_model states (m, v)
  - GaLore Adam: 2 × d_model × rank states
  - Savings: d_model / rank (e.g., 4096/128 = 32x)

References:
  - Zhao et al., "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection" (2024)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GaLoreConfig:
    """
    Configuration for GaLore.

    Attributes:
        rank: Low-rank dimension for gradient projection
        update_proj_interval: How often to update projection matrices
        scale: Scaling factor for projected gradients
        proj_type: Type of projection ('svd', 'random')
    """
    rank: int = 128
    update_proj_interval: int = 200
    scale: float = 1.0
    proj_type: str = "random"


# ============================================================================
# GRADIENT PROJECTION
# ============================================================================

class GradientProjector:
    """
    Projects gradients into low-rank subspace.

    Two projection methods:
    1. Random: Use random orthogonal matrices P, Q
    2. SVD: Use top-k singular vectors of gradient

    Random is faster, SVD is more accurate.
    """

    def __init__(self, rows: int, cols: int, rank: int, proj_type: str = "random"):
        """
        Initialize projector.

        Args:
            rows: Gradient rows
            cols: Gradient columns
            rank: Low-rank dimension
            proj_type: 'random' or 'svd'
        """
        self.rows = rows
        self.cols = cols
        self.rank = rank
        self.proj_type = proj_type

        # Initialize projection matrices
        if proj_type == "random":
            # Random orthogonal projections
            P, _ = np.linalg.qr(np.random.randn(rows, rank))
            Q, _ = np.linalg.qr(np.random.randn(cols, rank))
            self.P = P  # [rows, rank]
            self.Q = Q  # [cols, rank]
        else:
            self.P = None
            self.Q = None

    def project(self, G: np.ndarray) -> np.ndarray:
        """
        Project gradient to low-rank: G_low = P.T @ G @ Q

        Args:
            G: Full gradient [rows, cols]

        Returns:
            Low-rank gradient [rank, rank]
        """
        if self.proj_type == "random":
            return self.P.T @ G @ self.Q
        else:
            # SVD-based projection
            U, S, Vt = np.linalg.svd(G, full_matrices=False)
            return np.diag(S[:self.rank]) @ Vt[:self.rank]

    def unproject(self, G_low: np.ndarray) -> np.ndarray:
        """
        Recover full gradient from low-rank: G_approx = P @ G_low @ Q.T

        Args:
            G_low: Low-rank gradient [rank, rank]

        Returns:
            Approximated full gradient [rows, cols]
        """
        if self.proj_type == "random":
            return self.P @ G_low @ self.Q.T
        else:
            # SVD-based unprojection (approximate)
            U, S, Vt = np.linalg.svd(G_low, full_matrices=False)
            return U @ np.diag(S) @ Vt

    def update_projection(self, G: np.ndarray):
        """
        Update projection matrices using SVD of current gradient.

        This adapts the projection to the current gradient landscape.
        """
        if self.proj_type == "svd":
            U, S, Vt = np.linalg.svd(G, full_matrices=False)
            self.P = U[:, :self.rank]
            self.Q = Vt[:self.rank].T


# ============================================================================
# GALORE OPTIMIZER
# ============================================================================

class GaLoreAdam:
    """
    GaLore with Adam optimizer.

    Combines gradient low-rank projection with Adam optimization.
    This gives the memory savings of GaLore with the convergence of Adam.

    Memory comparison:
    - Standard Adam: 2 × d × d states (m, v) = 2d²
    - GaLore Adam: 2 × d × rank states = 2d·rank
    - Savings: d/rank (e.g., 4096/128 = 32x)
    """

    def __init__(self, params_shape: Tuple[int, ...], config: GaLoreConfig,
                 lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8):
        """
        Initialize GaLore Adam.

        Args:
            params_shape: Shape of parameter tensor
            config: GaLore configuration
            lr: Learning rate
            betas: Adam betas
            eps: Adam epsilon
        """
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.config = config

        # Create projector
        if len(params_shape) == 2:
            rows, cols = params_shape
        else:
            rows = params_shape[0]
            cols = 1

        self.projector = GradientProjector(rows, cols, config.rank, config.proj_type)

        # Low-rank optimizer states (instead of full-rank!)
        self.m = np.zeros((config.rank, config.rank))  # First moment
        self.v = np.zeros((config.rank, config.rank))  # Second moment
        self.t = 0

        # Memory comparison
        self.full_rank_states = 2 * rows * cols
        self.low_rank_states = 2 * config.rank * config.rank
        self.memory_savings = self.full_rank_states / self.low_rank_states

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Optimization step with GaLore.

        Args:
            param: Current parameters [rows, cols]
            grad: Full gradient [rows, cols]

        Returns:
            Updated parameters
        """
        self.t += 1

        # Step 1: Project gradient to low-rank
        G_low = self.projector.project(grad)

        # Step 2: Update low-rank Adam states
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * G_low
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * G_low ** 2

        # Step 3: Bias correction
        m_hat = self.m / (1 - self.betas[0] ** self.t)
        v_hat = self.v / (1 - self.betas[1] ** self.t)

        # Step 4: Compute low-rank update
        G_low_update = m_hat / (np.sqrt(v_hat) + self.eps)

        # Step 5: Unproject to full rank
        G_approx = self.projector.unproject(G_low_update)

        # Step 6: Update parameters
        param_new = param - self.lr * G_approx

        # Step 7: Update projection periodically
        if self.t % self.config.update_proj_interval == 0:
            self.projector.update_projection(grad)

        return param_new

    def get_memory_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "full_rank_states": self.full_rank_states,
            "low_rank_states": self.low_rank_states,
            "memory_savings": self.memory_savings,
            "rank": self.config.rank,
        }


# ============================================================================
# COMPARISON
# ============================================================================

def compare_galore_with_others():
    """Compare GaLore with other memory-efficient methods."""
    return """
    ┌──────────────────┬───────────┬───────────┬────────────┬──────────────┐
    │ Method           │ Quality   │ Memory    │ Speed      │ Best For     │
    ├──────────────────┼───────────┼───────────┼────────────┼──────────────┤
    │ Standard Adam    │ Best      │ High      │ Fast       │ Small models │
    │ 8-bit Adam       │ ~99%      │ Medium    │ Fast       │ Medium models│
    │ LoRA             │ ~99%      │ Low       │ Fast       │ Fine-tuning  │
    │ QLoRA            │ ~98%      │ Very Low  │ Fast       │ Low memory   │
    │ GaLore           │ ~99%      │ Low       │ Medium     │ Pre-training │
    │ GaLore + QLoRA   │ ~98%      │ Minimal   │ Medium     │ Max savings  │
    └──────────────────┴───────────┴───────────┴────────────┴──────────────┘
    """


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_galore():
    """
    Demonstrate GaLore.

    Shows:
        1. Gradient projection
        2. Memory savings
        3. Optimization step
        4. Comparison with other methods
    """
    print("=" * 70)
    print("GaLore: Gradient Low-Rank Projection — Demonstration")
    print("=" * 70)

    # Configuration
    config = GaLoreConfig(rank=64, update_proj_interval=100)

    # Create optimizer for a 4096×4096 weight matrix
    rows, cols = 4096, 4096
    optimizer = GaLoreAdam((rows, cols), config, lr=1e-3)

    print(f"\nConfiguration:")
    print(f"  Weight matrix: {rows} × {cols}")
    print(f"  Rank: {config.rank}")
    print(f"  Projection type: {config.proj_type}")

    # Memory comparison
    stats = optimizer.get_memory_stats()
    print(f"\nMemory Comparison:")
    print(f"  Full-rank Adam states: {stats['full_rank_states']:,} parameters")
    print(f"  GaLore Adam states: {stats['low_rank_states']:,} parameters")
    print(f"  Memory savings: {stats['memory_savings']:.1f}x")

    # Gradient projection demo
    print("\n[Gradient Projection]")
    G = np.random.randn(rows, cols) * 0.01
    G_low = optimizer.projector.project(G)
    G_approx = optimizer.projector.unproject(G_low)

    print(f"  Original gradient: {G.shape}")
    print(f"  Low-rank gradient: {G_low.shape}")
    print(f"  Reconstruction error: {np.mean(np.abs(G - G_approx)):.6f}")

    # Optimization step
    print("\n[Optimization Step]")
    param = np.random.randn(rows, cols) * 0.01
    grad = np.random.randn(rows, cols) * 0.01

    param_new = optimizer.step(param, grad)
    update_norm = np.linalg.norm(param_new - param)
    print(f"  Update norm: {update_norm:.6f}")

    # Comparison
    print("\n[Memory-Efficient Methods Comparison]")
    print(compare_galore_with_others())

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. GaLore projects gradients, not weights (unlike LoRA)")
    print("  2. Weights stay full-rank → full model capacity")
    print("  3. Optimizer states are low-rank → huge memory savings")
    print("  4. Best for pre-training (LoRA is for fine-tuning)")
    print("  5. Can combine with QLoRA for maximum savings")
    print("=" * 70)


if __name__ == "__main__":
    demo_galore()
