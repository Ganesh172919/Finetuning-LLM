"""
Model Merging: TIES, DARE, and Linear Interpolation
=====================================================

Model merging combines multiple fine-tuned models into a single model
without any additional training. This is a powerful technique for:
- Combining capabilities from different fine-tuned models
- Creating multi-task models without multi-task training
- Improving robustness through ensemble-like effects

Methods:
┌─────────────────────────────────────────────────────────────┐
│ Linear Interpolation:                                       │
│   θ_merged = α·θ_A + (1-α)·θ_B                            │
│   Simple weighted average. Works surprisingly well.         │
├─────────────────────────────────────────────────────────────┤
│ TIES (Trim, Elect, Sign):                                   │
│   1. Trim: Remove small magnitude changes                   │
│   2. Elect: Resolve sign conflicts (majority vote)          │
│   3. Sign: Keep only agreed-upon directions                 │
│   Better preservation of task-specific knowledge.           │
├─────────────────────────────────────────────────────────────┤
│ DARE (Drop And REscale):                                    │
│   1. Randomly drop delta parameters (set to 0)              │
│   2. Rescale remaining to preserve variance                 │
│   3. Merge rescaled deltas                                  │
│   Reduces interference between merged models.               │
├─────────────────────────────────────────────────────────────┤
│ SLERP (Spherical Linear Interpolation):                     │
│   Interpolate along the surface of a hypersphere.           │
│   Preserves magnitude better than linear interp.            │
└─────────────────────────────────────────────────────────────┘

References:
  - Yadav et al., "TIES-Merging" (2023)
  - Yu et al., "Language Models are Super Mario" (2024)
  - Ilharco et al., "Editing Models with Task Arithmetic" (2023)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MergingConfig:
    """
    Configuration for model merging.

    Attributes:
        method: Merging method (linear, ties, dare, slerp)
        alpha: Interpolation weight (0.0 = all model B, 1.0 = all model A)
        density: Fraction of parameters to keep (for TIES/DARE)
        tie_strength: Threshold for TIES trimming
        dare_prob: Probability of dropping a parameter (for DARE)
    """
    method: str = "linear"
    alpha: float = 0.5
    density: float = 0.5
    tie_strength: float = 0.1
    dare_prob: float = 0.5


# ============================================================================
# LINEAR INTERPOLATION
# ============================================================================

def linear_merge(weights_a: Dict[str, np.ndarray],
                 weights_b: Dict[str, np.ndarray],
                 alpha: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Simple linear interpolation between two models.

    θ_merged = α·θ_A + (1-α)·θ_B

    This is the simplest merging method and often works surprisingly well.
    The key insight: models fine-tuned from the same base model stay
    in a connected region of parameter space, so interpolation is valid.

    Args:
        weights_a: First model's weights (layer_name → tensor)
        weights_b: Second model's weights
        alpha: Weight for model A (0.0 to 1.0)

    Returns:
        Merged weights
    """
    merged = {}
    for key in weights_a:
        if key in weights_b:
            merged[key] = alpha * weights_a[key] + (1 - alpha) * weights_b[key]
        else:
            merged[key] = weights_a[key]
    return merged


# ============================================================================
# TIES-MERGING
# ============================================================================

def ties_merge(weights_list: List[Dict[str, np.ndarray]],
               base_weights: Dict[str, np.ndarray],
               density: float = 0.5,
               ) -> Dict[str, np.ndarray]:
    """
    TIES-Merging: Trim, Elect, Sign.

    Algorithm:
    1. Compute deltas: Δθ_i = θ_i - θ_base
    2. TRIM: Remove small changes (keep top-k by magnitude)
    3. ELECT: For each parameter, vote on sign (+/-)
    4. SIGN: Keep only parameters that agree with majority sign

    This resolves conflicts between models and keeps only
    the most important, consistent changes.

    Args:
        weights_list: List of fine-tuned model weights
        base_weights: Base model weights (before fine-tuning)
        density: Fraction of parameters to keep after trimming

    Returns:
        Merged weights
    """
    # Step 1: Compute deltas (task vectors)
    deltas = []
    for weights in weights_list:
        delta = {}
        for key in weights:
            if key in base_weights:
                delta[key] = weights[key] - base_weights[key]
        deltas.append(delta)

    # Step 2: TRIM - keep only top-k by magnitude
    trimmed_deltas = []
    for delta in deltas:
        trimmed = {}
        for key, d in delta.items():
            # Flatten for percentile computation
            flat = np.abs(d).flatten()
            threshold = np.percentile(flat, (1 - density) * 100)
            mask = np.abs(d) >= threshold
            trimmed[key] = d * mask
        trimmed_deltas.append(trimmed)

    # Step 3: ELECT - determine sign agreement
    merged = {}
    for key in base_weights:
        if key not in trimmed_deltas[0]:
            merged[key] = base_weights[key]
            continue

        # Collect all deltas for this key
        key_deltas = [d[key] for d in trimmed_deltas if key in d]

        if not key_deltas:
            merged[key] = base_weights[key]
            continue

        # Step 4: SIGN - apply majority sign
        # Sum of signs across models
        sign_sum = np.zeros_like(key_deltas[0])
        for d in key_deltas:
            sign_sum += np.sign(d)

        # Majority sign
        majority_sign = np.sign(sign_sum)

        # Keep only deltas that agree with majority
        aligned_deltas = []
        for d in key_deltas:
            mask = np.sign(d) == majority_sign
            aligned_deltas.append(d * mask)

        # Average the aligned deltas
        merged_delta = np.mean(aligned_deltas, axis=0)

        merged[key] = base_weights[key] + merged_delta

    return merged


# ============================================================================
# DARE (Drop And REscale)
# ============================================================================

def dare_merge(weights_list: List[Dict[str, np.ndarray]],
               base_weights: Dict[str, np.ndarray],
               drop_prob: float = 0.5,
               ) -> Dict[str, np.ndarray]:
    """
    DARE: Drop And REscale merging.

    Algorithm:
    1. Compute deltas: Δθ_i = θ_i - θ_base
    2. Randomly drop each delta parameter with probability p
    3. Rescale remaining by 1/(1-p) to preserve variance
    4. Merge rescaled deltas

    Key insight: Most delta parameters are redundant. Dropping random
    ones and rescaling preserves the important information while
    reducing interference between models.

    The variance preservation:
    Var[Δθ_dropped] = (1-p)·Var[Δθ]·(1/(1-p))² = Var[Δθ]/(1-p)

    Args:
        weights_list: List of fine-tuned model weights
        base_weights: Base model weights
        drop_prob: Probability of dropping each parameter

    Returns:
        Merged weights
    """
    merged = {}
    scale = 1.0 / (1.0 - drop_prob)

    for key in base_weights:
        if key not in weights_list[0]:
            merged[key] = base_weights[key]
            continue

        # Collect deltas
        deltas = []
        for weights in weights_list:
            if key in weights:
                delta = weights[key] - base_weights[key]

                # Random dropout mask
                mask = np.random.random(delta.shape) >= drop_prob

                # Apply mask and rescale
                delta_dropped = delta * mask * scale

                deltas.append(delta_dropped)

        if deltas:
            # Average the dropped-and-rescaled deltas
            merged_delta = np.mean(deltas, axis=0)
            merged[key] = base_weights[key] + merged_delta
        else:
            merged[key] = base_weights[key]

    return merged


# ============================================================================
# SLERP (Spherical Linear Interpolation)
# ============================================================================

def slerp_merge(weights_a: Dict[str, np.ndarray],
                weights_b: Dict[str, np.ndarray],
                alpha: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Spherical Linear Interpolation (SLERP).

    Unlike linear interpolation, SLERP moves along the surface of a
    hypersphere. This preserves the magnitude of the weight vectors
    better than linear interp.

    Formula:
    slerp(a, b, t) = sin((1-t)·Ω)/sin(Ω) · a + sin(t·Ω)/sin(Ω) · b
    where Ω = arccos(a·b / (|a|·|b|))

    Best for:
    - Models that are far apart in parameter space
    - When you want to preserve weight magnitudes

    Args:
        weights_a: First model's weights
        weights_b: Second model's weights
        alpha: Interpolation weight (0.0 to 1.0)

    Returns:
        Merged weights
    """
    merged = {}

    for key in weights_a:
        if key not in weights_b:
            merged[key] = weights_a[key]
            continue

        a = weights_a[key].flatten()
        b = weights_b[key].flatten()

        # Compute angle between vectors
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-10 or norm_b < 1e-10:
            # Degenerate case: use linear interpolation
            merged[key] = alpha * weights_a[key] + (1 - alpha) * weights_b[key]
            continue

        cos_omega = np.clip(dot / (norm_a * norm_b), -1, 1)
        omega = np.arccos(cos_omega)

        if omega < 1e-6:
            # Vectors are nearly parallel: use linear interpolation
            merged[key] = alpha * weights_a[key] + (1 - alpha) * weights_b[key]
            continue

        # SLERP formula
        sin_omega = np.sin(omega)
        coeff_a = np.sin((1 - alpha) * omega) / sin_omega
        coeff_b = np.sin(alpha * omega) / sin_omega

        merged_flat = coeff_a * a + coeff_b * b
        merged[key] = merged_flat.reshape(weights_a[key].shape)

    return merged


# ============================================================================
# TASK ARITHMETIC
# ============================================================================

def task_arithmetic_add(base_weights: Dict[str, np.ndarray],
                        task_vector: Dict[str, np.ndarray],
                        scale: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Add a task vector to base model.

    θ_new = θ_base + scale · (θ_task - θ_base)

    Task vectors encode the "direction" of fine-tuning.
    Adding a task vector adds that capability to the model.

    Args:
        base_weights: Base model weights
        task_vector: Fine-tuned model weights
        scale: Scaling factor for the task vector

    Returns:
        Updated weights
    """
    result = {}
    for key in base_weights:
        if key in task_vector:
            delta = task_vector[key] - base_weights[key]
            result[key] = base_weights[key] + scale * delta
        else:
            result[key] = base_weights[key]
    return result


def task_arithmetic_negate(base_weights: Dict[str, np.ndarray],
                           task_vector: Dict[str, np.ndarray],
                           scale: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Negate a task vector (remove a capability).

    θ_new = θ_base - scale · (θ_task - θ_base)

    This can "unlearn" a task, though it's not perfect.

    Args:
        base_weights: Base model weights
        task_vector: Fine-tuned model weights to negate
        scale: Scaling factor

    Returns:
        Updated weights
    """
    result = {}
    for key in base_weights:
        if key in task_vector:
            delta = task_vector[key] - base_weights[key]
            result[key] = base_weights[key] - scale * delta
        else:
            result[key] = base_weights[key]
    return result


# ============================================================================
# MERGING COMPARISON
# ============================================================================

def compare_merging_methods():
    """Compare model merging methods."""
    return """
    ┌──────────────────┬───────────┬───────────┬────────────┬──────────────┐
    │ Method           │ Quality   │ Speed     │ Complexity │ Best For     │
    ├──────────────────┼───────────┼───────────┼────────────┼──────────────┤
    │ Linear           │ Good      │ Instant   │ Trivial    │ Similar tasks│
    │ TIES             │ Better    │ Fast      │ Medium     │ Conflicts    │
    │ DARE             │ Better    │ Fast      │ Medium     │ Many models  │
    │ SLERP            │ Good      │ Fast      │ Medium     │ Far models   │
    │ Task Arithmetic  │ Variable  │ Instant   │ Simple     │ Adding/remov │
    └──────────────────┴───────────┴───────────┴────────────┴──────────────┘
    """


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_model_merging():
    """
    Demonstrate model merging techniques.

    Shows:
        1. Linear interpolation
        2. TIES merging
        3. DARE merging
        4. Task arithmetic
    """
    print("=" * 70)
    print("Model Merging — Demonstration")
    print("=" * 70)

    # Create synthetic model weights
    np.random.seed(42)
    base = {"w1": np.random.randn(32, 32), "w2": np.random.randn(32)}

    # Fine-tuned models (base + small perturbation)
    model_a = {k: v + np.random.randn(*v.shape) * 0.1 for k, v in base.items()}
    model_b = {k: v + np.random.randn(*v.shape) * 0.1 for k, v in base.items()}

    print(f"\nModel A delta norm: {np.linalg.norm(model_a['w1'] - base['w1']):.4f}")
    print(f"Model B delta norm: {np.linalg.norm(model_b['w1'] - base['w1']):.4f}")

    # Linear merge
    print("\n[Linear Interpolation]")
    merged_linear = linear_merge(model_a, model_b, alpha=0.5)
    print(f"  Merged delta norm: {np.linalg.norm(merged_linear['w1'] - base['w1']):.4f}")

    # TIES merge
    print("\n[TIES Merging]")
    merged_ties = ties_merge([model_a, model_b], base, density=0.5)
    print(f"  Merged delta norm: {np.linalg.norm(merged_ties['w1'] - base['w1']):.4f}")

    # DARE merge
    print("\n[DARE Merging]")
    merged_dare = dare_merge([model_a, model_b], base, drop_prob=0.5)
    print(f"  Merged delta norm: {np.linalg.norm(merged_dare['w1'] - base['w1']):.4f}")

    # Task arithmetic
    print("\n[Task Arithmetic]")
    enhanced = task_arithmetic_add(base, model_a, scale=1.5)
    print(f"  Enhanced delta norm: {np.linalg.norm(enhanced['w1'] - base['w1']):.4f}")

    # Comparison
    print("\n[Merging Methods Comparison]")
    print(compare_merging_methods())

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. Linear interpolation is surprisingly effective")
    print("  2. TIES resolves conflicts between models")
    print("  3. DARE reduces interference through random dropout")
    print("  4. Task arithmetic can add/remove capabilities")
    print("  5. All methods work without additional training")
    print("=" * 70)


if __name__ == "__main__":
    demo_model_merging()
