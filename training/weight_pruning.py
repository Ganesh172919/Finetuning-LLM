"""
Weight Pruning for Model Compression
======================================

Pruning removes unnecessary weights from a neural network, making it
smaller and faster while maintaining quality.

Types of Pruning:
┌─────────────────────────────────────────────────────────────┐
│ Unstructured Pruning:                                       │
│   Remove individual weights (set to 0)                      │
│   → Creates sparse matrices                                 │
│   → Needs sparse hardware for speedup                       │
│   → 90%+ sparsity possible with minimal quality loss        │
├─────────────────────────────────────────────────────────────┤
│ Structured Pruning:                                         │
│   Remove entire neurons, channels, or attention heads       │
│   → Creates smaller dense matrices                          │
│   → Works on standard hardware                              │
│   → 30-50% compression typical                              │
├─────────────────────────────────────────────────────────────┤
│ Semi-Structured (N:M Sparsity):                            │
│   Keep N out of every M weights                             │
│   → 2:4 = 50% sparsity (hardware supported on A100+)       │
│   → Good balance of compression and quality                 │
└─────────────────────────────────────────────────────────────┘

Pruning Criteria:
  - Magnitude: Remove smallest weights (simplest, works well)
  - Gradient: Remove weights with smallest gradients
  - Hessian: Remove weights with smallest second-order impact
  - Movement: Remove weights that moved least during training

Pruning Schedule:
  - One-shot: Prune once after training
  - Iterative: Prune gradually during training (better quality)
  - Gradual: Increase sparsity over training (e.g., cubic schedule)

References:
  - Han et al., "Learning both Weights and Connections" (2015)
  - Frantar & Alistarh, "SparseGPT" (2023)
  - Ashkboos et al., "Wanda" (2024)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PruningConfig:
    """
    Configuration for weight pruning.

    Attributes:
        sparsity: Target sparsity (0.0 = no pruning, 1.0 = all pruned)
        method: Pruning method (magnitude, gradient, structured, n:m)
        structured: Whether to use structured pruning
        n_m_ratio: For N:M sparsity (e.g., (2, 4) for 2:4)
        schedule: Pruning schedule (one_shot, iterative, gradual)
        iterative_steps: Number of iterative pruning steps
    """
    sparsity: float = 0.5
    method: str = "magnitude"
    structured: bool = False
    n_m_ratio: Tuple[int, int] = (2, 4)
    schedule: str = "one_shot"
    iterative_steps: int = 10


# ============================================================================
# MAGNITUDE PRUNING
# ============================================================================

def magnitude_pruning(weight: np.ndarray, sparsity: float = 0.5
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prune weights by magnitude (remove smallest).

    This is the simplest and most widely used pruning method.
    Works surprisingly well because:
    - Small weights contribute little to output
    - Large weights capture important features
    - After pruning, remaining weights can be fine-tuned

    Args:
        weight: Weight matrix
        sparsity: Fraction of weights to prune (0.0 to 1.0)

    Returns:
        pruned_weight: Weight with pruned entries set to 0
        mask: Binary mask (1 = keep, 0 = pruned)
    """
    # Compute threshold: weights below this are pruned
    abs_weight = np.abs(weight)
    threshold = np.percentile(abs_weight, sparsity * 100)

    # Create mask: 1 where |weight| > threshold
    mask = (abs_weight > threshold).astype(np.float32)

    # Apply mask
    pruned_weight = weight * mask

    return pruned_weight, mask


# ============================================================================
# STRUCTURED PRUNING
# ============================================================================

def structured_pruning(weight: np.ndarray, sparsity: float = 0.5,
                       axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prune entire rows/columns based on L2 norm.

    Structured pruning removes entire structures (neurons, channels):
    - For linear layers: remove output neurons (rows) or input features (columns)
    - For conv layers: remove filters or channels
    - Result: smaller dense matrix (not sparse)

    Benefits:
    - Works on standard hardware (no sparse support needed)
    - Simpler deployment
    - More predictable latency

    Args:
        weight: Weight matrix [out_features, in_features]
        sparsity: Fraction of structures to prune
        axis: Which axis to prune (0 = rows, 1 = columns)

    Returns:
        pruned_weight: Weight with pruned rows/columns zeroed out
        mask: Binary mask for the pruned axis
    """
    # Compute L2 norm along the other axis
    norms = np.linalg.norm(weight, axis=1-axis)

    # Determine how many to prune
    n_prune = int(len(norms) * sparsity)

    # Find indices to keep (highest norms)
    keep_indices = np.argsort(norms)[-len(norms)+n_prune:]

    # Create mask
    mask = np.zeros(weight.shape[axis])
    mask[keep_indices] = 1.0

    # Apply mask
    pruned_weight = weight.copy()
    if axis == 0:
        pruned_weight[mask == 0] = 0
    else:
        pruned_weight[:, mask == 0] = 0

    return pruned_weight, mask


# ============================================================================
# N:M SPARSITY (Semi-Structured)
# ============================================================================

def n_m_sparsity(weight: np.ndarray, n: int = 2, m: int = 4
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply N:M semi-structured sparsity.

    Keep exactly N out of every M weights (by magnitude).
    This is hardware-supported on NVIDIA A100+ GPUs.

    Common: 2:4 sparsity (50% sparse, 2x speedup on A100)

    Args:
        weight: Weight matrix
        n: Number of weights to keep per group
        m: Group size

    Returns:
        pruned_weight: Weight with N:M sparsity applied
        mask: Binary mask
    """
    W_flat = weight.reshape(-1)
    n_groups = len(W_flat) // m

    mask = np.zeros_like(W_flat)

    for g in range(n_groups):
        start = g * m
        end = start + m
        group = W_flat[start:end]

        # Keep top-n by magnitude
        top_n_indices = np.argsort(np.abs(group))[-n:]
        mask[start + top_n_indices] = 1.0

    mask = mask.reshape(weight.shape)
    pruned_weight = weight * mask

    return pruned_weight, mask


# ============================================================================
# WANDA (Weight and Activation Based Pruning)
# ============================================================================

def wanda_pruning(weight: np.ndarray, activations: np.ndarray,
                  sparsity: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wanda: Pruning by Weights and Activations.

    Key insight: A weight is important if BOTH:
    1. The weight magnitude is large
    2. The corresponding input activation is large

    Score = |weight| × ||activation||

    This is more accurate than magnitude-only pruning because:
    - A large weight with zero activation is useless
    - A small weight with large activation matters

    Args:
        weight: Weight matrix [out, in]
        activations: Input activations [batch, seq_len, in]
        sparsity: Target sparsity

    Returns:
        pruned_weight: Pruned weight matrix
        mask: Binary mask
    """
    # Compute per-input-channel activation norm
    act_norm = np.mean(np.abs(activations), axis=(0, 1))  # [in]

    # Compute importance score
    score = np.abs(weight) * act_norm[np.newaxis, :]

    # Prune by score magnitude
    threshold = np.percentile(score, sparsity * 100)
    mask = (score > threshold).astype(np.float32)

    pruned_weight = weight * mask

    return pruned_weight, mask


# ============================================================================
# PRUNING SCHEDULES
# ============================================================================

def cubic_schedule(step: int, total_steps: int, final_sparsity: float = 0.5
                   ) -> float:
    """
    Cubic sparsity schedule (from Gradual Magnitude Pruning).

    Sparsity increases as a cubic function of training progress:
    s(t) = s_final × (1 - (1 - t/T)³)

    This means:
    - Early training: low sparsity (lots of capacity)
    - Late training: rapid sparsity increase
    - Final: target sparsity reached

    Args:
        step: Current training step
        total_steps: Total training steps
        final_sparsity: Target final sparsity

    Returns:
        Current sparsity target
    """
    progress = step / total_steps
    return final_sparsity * (1 - (1 - progress) ** 3)


def linear_schedule(step: int, total_steps: int, final_sparsity: float = 0.5
                    ) -> float:
    """Linear sparsity schedule."""
    progress = step / total_steps
    return final_sparsity * progress


# ============================================================================
# PRUNING TRAINER
# ============================================================================

class PruningTrainer:
    """
    Training loop with pruning.

    Supports:
    - One-shot pruning (prune after training)
    - Iterative pruning (prune during training)
    - Gradual pruning (increase sparsity over training)
    """

    def __init__(self, config: PruningConfig):
        self.config = config
        self.step_count = 0
        self.masks = {}  # Layer name → binary mask
        self.history = []

    def apply_pruning(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply pruning to all weight matrices.

        Args:
            weights: Dictionary of layer_name → weight_matrix

        Returns:
            Pruned weights
        """
        if self.config.schedule == "one_shot":
            sparsity = self.config.sparsity
        elif self.config.schedule == "gradual":
            sparsity = cubic_schedule(
                self.step_count, 1000, self.config.sparsity
            )
        else:
            sparsity = self.config.sparsity

        pruned = {}
        for name, W in weights.items():
            if self.config.method == "magnitude":
                pruned_W, mask = magnitude_pruning(W, sparsity)
            elif self.config.method == "structured":
                pruned_W, mask = structured_pruning(W, sparsity)
            elif self.config.method == "n:m":
                n, m = self.config.n_m_ratio
                pruned_W, mask = n_m_sparsity(W, n, m)
            else:
                pruned_W, mask = magnitude_pruning(W, sparsity)

            pruned[name] = pruned_W
            self.masks[name] = mask

        self.step_count += 1
        return pruned

    def get_sparsity_stats(self) -> Dict:
        """Get current sparsity statistics."""
        if not self.masks:
            return {"status": "not_pruned"}

        total_params = 0
        pruned_params = 0

        for name, mask in self.masks.items():
            total_params += mask.size
            pruned_params += np.sum(mask == 0)

        return {
            "total_params": total_params,
            "pruned_params": pruned_params,
            "sparsity": pruned_params / total_params if total_params > 0 else 0,
            "remaining_params": total_params - pruned_params,
            "compression_ratio": total_params / max(1, total_params - pruned_params),
        }


# ============================================================================
# BENCHMARKS
# ============================================================================

def pruning_benchmarks():
    """Compare pruning methods."""
    return """
    ┌──────────────────┬──────────┬───────────┬────────────┬──────────────┐
    │ Method           │ Sparsity │ Quality   │ Speedup    │ Hardware     │
    ├──────────────────┼──────────┼───────────┼────────────┼──────────────┤
    │ Magnitude        │ 50-90%   │ 95-99%    │ 1x (sparse)│ Needs sparse │
    │ Structured       │ 30-50%   │ 90-95%    │ 1.5-2x     │ Standard     │
    │ 2:4 Sparsity     │ 50%      │ 97-99%    │ 2x         │ A100+        │
    │ Wanda            │ 50-90%   │ 96-99%    │ 1x (sparse)│ Needs sparse │
    │ SparseGPT        │ 50-90%   │ 97-99%    │ 1x (sparse)│ Needs sparse │
    └──────────────────┴──────────┴───────────┴────────────┴──────────────┘
    """


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_pruning():
    """
    Demonstrate weight pruning.

    Shows:
        1. Magnitude pruning
        2. Structured pruning
        3. N:M sparsity
        4. Sparsity visualization
        5. Quality vs compression tradeoff
    """
    print("=" * 70)
    print("Weight Pruning — Demonstration")
    print("=" * 70)

    # Create sample weight matrix
    np.random.seed(42)
    weight = np.random.randn(128, 128).astype(np.float32)

    print(f"\nOriginal Weight Matrix:")
    print(f"  Shape: {weight.shape}")
    print(f"  Total params: {weight.size:,}")
    print(f"  Non-zero: {np.count_nonzero(weight):,}")

    # Magnitude pruning
    print("\n[Magnitude Pruning @ 50% sparsity]")
    pruned, mask = magnitude_pruning(weight, sparsity=0.5)
    print(f"  Non-zero after: {np.count_nonzero(pruned):,}")
    print(f"  Sparsity: {1 - np.count_nonzero(pruned)/pruned.size:.1%}")
    print(f"  Reconstruction error: {np.mean(np.abs(weight - pruned)):.6f}")

    # Structured pruning
    print("\n[Structured Pruning @ 50% sparsity]")
    pruned_s, mask_s = structured_pruning(weight, sparsity=0.5, axis=0)
    print(f"  Remaining rows: {np.count_nonzero(mask_s):.0f}/{len(mask_s)}")
    print(f"  Reconstruction error: {np.mean(np.abs(weight - pruned_s)):.6f}")

    # N:M sparsity
    print("\n[2:4 Semi-Structured Sparsity]")
    pruned_nm, mask_nm = n_m_sparsity(weight, n=2, m=4)
    print(f"  Sparsity: {1 - np.count_nonzero(pruned_nm)/pruned_nm.size:.1%}")
    print(f"  Hardware speedup: 2x (A100+ GPU)")

    # Pruning schedule
    print("\n[Gradual Pruning Schedule]")
    for step in [0, 250, 500, 750, 1000]:
        sparsity = cubic_schedule(step, 1000, final_sparsity=0.8)
        print(f"  Step {step}: sparsity = {sparsity:.1%}")

    # Comparison
    print("\n[Pruning Methods Comparison]")
    print(pruning_benchmarks())

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. 50% pruning = 2x smaller, minimal quality loss")
    print("  2. 90% pruning = 10x smaller, some quality loss")
    print("  3. Structured pruning works on standard hardware")
    print("  4. 2:4 sparsity is hardware-supported on A100+")
    print("  5. Wanda uses activation info for better pruning")
    print("=" * 70)


if __name__ == "__main__":
    demo_pruning()
