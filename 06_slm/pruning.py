"""
################################################################################
PRUNING — REMOVING REDUNDANT WEIGHTS FOR EFFICIENCY
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Pruning?
    Pruning removes redundant weights from neural networks, making them
    smaller and faster with minimal quality loss. Like trimming a bonsai
    tree — remove the unnecessary branches, keep the essential structure.

Why does it matter?
    Large models are overparameterized:
    - 90%+ of weights can be pruned with <1% quality loss
    - Pruned models are faster (sparse operations)
    - Combined with quantization: 10-20x compression

How does it work?
    1. Magnitude Pruning: Remove smallest weights
    2. Structured Pruning: Remove entire heads/neurons
    3. SparseGPT: Optimal pruning using Hessian
    4. Iterative: Gradual pruning with fine-tuning

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


################################################################################
# SECTION 1: CONFIGURATION
################################################################################

@dataclass
class PruningConfig:
    """Pruning configuration."""
    method: str = 'magnitude'  # magnitude, structured, sparsegpt
    sparsity: float = 0.5  # Fraction of weights to remove
    granularity: str = 'unstructured'  # unstructured, row, column, head
    schedule: str = 'one_shot'  # one_shot, iterative


################################################################################
# SECTION 2: MAGNITUDE PRUNER
################################################################################

class MagnitudePruner:
    """
    Unstructured Magnitude Pruning.

    Remove weights with smallest absolute values.

    Formula: mask = (|W| > threshold), threshold = percentile(|W|, sparsity)

    Interview Question:
        "How does magnitude pruning work?"
        Sort weights by absolute value, remove the bottom sparsity%.
        This assumes small weights contribute less to the output.
        Simple but effective — 50% pruning usually has <1% quality loss.
    """

    def compute_mask(self, weights: np.ndarray, sparsity: float) -> np.ndarray:
        """
        Compute binary mask for pruning.

        Args:
            weights: Weight matrix
            sparsity: Fraction to prune (0-1)

        Returns:
            Binary mask (1=keep, 0=prune)
        """
        threshold = np.percentile(np.abs(weights), sparsity * 100)
        return (np.abs(weights) > threshold).astype(np.float32)

    def prune(self, weights: np.ndarray, sparsity: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prune weights by magnitude.

        Args:
            weights: Weight matrix
            sparsity: Fraction to prune

        Returns:
            Tuple of (pruned weights, mask)
        """
        mask = self.compute_mask(weights, sparsity)
        return weights * mask, mask


################################################################################
# SECTION 3: STRUCTURED PRUNER
################################################################################

class StructuredPruner:
    """
    Structured Pruning — Remove entire structures.

    Instead of individual weights, remove:
    - Attention heads
    - FFN neurons
    - Entire layers

    Interview Question:
        "What's the difference between structured and unstructured pruning?"
        Unstructured: remove individual weights → sparse matrix (needs special HW)
        Structured: remove entire rows/columns/heads → smaller dense matrix
        Structured is hardware-friendly but less flexible.
    """

    def compute_head_importance(self, attention_weights: np.ndarray) -> np.ndarray:
        """
        Compute importance score for each attention head.

        Args:
            attention_weights: (n_heads, seq_len, seq_len)

        Returns:
            Importance score per head
        """
        # Importance = attention entropy (lower entropy = more focused = more important)
        n_heads = attention_weights.shape[0]
        importance = np.zeros(n_heads)
        for h in range(n_heads):
            attn = attention_weights[h]
            # Entropy: -sum(p * log(p))
            attn_safe = attn + 1e-10
            entropy = -np.sum(attn_safe * np.log(attn_safe), axis=-1)
            importance[h] = -np.mean(entropy)  # Lower entropy = higher importance
        return importance

    def prune_heads(self, weights: List[np.ndarray], n_prune: int) -> Tuple[List[np.ndarray], List[int]]:
        """
        Prune least important attention heads.

        Args:
            weights: List of weight matrices per head
            n_prune: Number of heads to prune

        Returns:
            Tuple of (remaining weights, pruned indices)
        """
        # Simplified: prune heads with smallest weight norms
        norms = [np.linalg.norm(w) for w in weights]
        prune_indices = np.argsort(norms)[:n_prune]
        keep_indices = [i for i in range(len(weights)) if i not in prune_indices]
        return [weights[i] for i in keep_indices], prune_indices.tolist()


################################################################################
# SECTION 4: SparseGPT PRUNER
################################################################################

class SparseGPTPruner:
    """
    SparseGPT — Optimal pruning using Hessian information.

    Key insight: prune weights while preserving output by solving
    a local optimization problem for each weight column.

    Paper: "SparseGPT: Massive Language Models Can be Accurately
            Pruned in One-Shot" (Frantar & Alistarh, 2023)

    Interview Question:
        "How does SparseGPT work?"
        SparseGPT prunes one column at a time, minimizing the output
        error using the Hessian. After pruning weight w_ij, it adjusts
        remaining weights in the same row to compensate. This achieves
        50-60% pruning with minimal quality loss in one shot.
    """

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def prune(self, weights: np.ndarray, hessian: np.ndarray,
              sparsity: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        SparseGPT pruning.

        Args:
            weights: Weight matrix (d_out, d_in)
            hessian: Hessian matrix (d_in, d_in)
            sparsity: Target sparsity

        Returns:
            Tuple of (pruned weights, mask)
        """
        d_out, d_in = weights.shape
        mask = np.ones_like(weights)

        # Compute importance scores using Hessian diagonal
        hessian_diag = np.diag(hessian) + 1e-10
        importance = weights ** 2 / hessian_diag

        # Prune lowest importance weights
        threshold = np.percentile(np.abs(importance), sparsity * 100)
        mask = (np.abs(importance) > threshold).astype(np.float32)

        # Compensate: adjust remaining weights
        pruned = weights * mask
        # Simplified compensation (full version uses Hessian inverse)
        error = weights - pruned
        compensation = error @ hessian / (np.diag(hessian) + 1e-10)
        pruned += compensation * mask * 0.1  # Partial compensation

        return pruned, mask


################################################################################
# SECTION 5: ITERATIVE PRUNING
################################################################################

class IterativePruning:
    """
    Iterative Pruning — Gradual pruning with fine-tuning.

    Instead of pruning all at once, gradually increase sparsity
    over training. This allows the model to adapt.

    Schedule: sparsity(t) = target * (1 - (1 - t/T)^3)  (cubic)

    Interview Question:
        "Why use iterative pruning over one-shot?"
        Gradual pruning lets the model adapt to each pruning step.
        One-shot pruning can cause sudden quality drops. With iterative,
        you fine-tune between rounds, recovering lost quality.
        Typically: prune 10% → finetune → prune 10% → finetune → ...
    """

    def __init__(self, target_sparsity: float = 0.5, n_rounds: int = 5):
        self.target_sparsity = target_sparsity
        self.n_rounds = n_rounds

    def get_sparsity(self, round_idx: int) -> float:
        """
        Get sparsity for current round (cubic schedule).

        Args:
            round_idx: Current round (0 to n_rounds-1)

        Returns:
            Sparsity for this round
        """
        t = round_idx / max(self.n_rounds - 1, 1)
        return self.target_sparsity * (1 - (1 - t) ** 3)

    def prune_round(self, weights: np.ndarray, round_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one round of pruning.

        Args:
            weights: Current weights
            round_idx: Round number

        Returns:
            Tuple of (pruned weights, mask)
        """
        sparsity = self.get_sparsity(round_idx)
        pruner = MagnitudePruner()
        return pruner.prune(weights, sparsity)


################################################################################
# SECTION 6: DEMONSTRATION
################################################################################

def demonstrate_pruning():
    """Demonstrate pruning methods."""
    print("=" * 70)
    print("PRUNING DEMONSTRATION")
    print("=" * 70)

    weights = np.random.randn(64, 128) * 0.02

    # Magnitude Pruning
    print("\n1. MAGNITUDE PRUNING")
    print("-" * 40)
    pruner = MagnitudePruner()
    for sparsity in [0.3, 0.5, 0.7, 0.9]:
        pruned, mask = pruner.prune(weights, sparsity)
        actual_sparsity = 1 - mask.sum() / mask.size
        print(f"  Target: {sparsity:.0%}, Actual: {actual_sparsity:.0%}, "
              f"Non-zero: {mask.sum():.0f}/{mask.size}")

    # Structured Pruning
    print("\n2. STRUCTURED PRUNING")
    print("-" * 40)
    spruner = StructuredPruner()
    head_weights = [np.random.randn(8, 8) for _ in range(8)]
    remaining, pruned_idx = spruner.prune_heads(head_weights, 3)
    print(f"  Original heads: {len(head_weights)}")
    print(f"  Pruned indices: {pruned_idx}")
    print(f"  Remaining heads: {len(remaining)}")

    # SparseGPT
    print("\n3. SparseGPT PRUNING")
    print("-" * 40)
    sgpt = SparseGPTPruner(group_size=64)
    hessian = np.eye(128) * 0.01
    pruned, mask = sgpt.prune(weights, hessian, sparsity=0.5)
    print(f"  Sparsity: {1 - mask.sum() / mask.size:.0%}")
    print(f"  Output change: {np.linalg.norm(pruned - weights):.6f}")

    # Iterative Pruning
    print("\n4. ITERATIVE PRUNING")
    print("-" * 40)
    ip = IterativePruning(target_sparsity=0.8, n_rounds=5)
    for r in range(5):
        sparsity = ip.get_sparsity(r)
        print(f"  Round {r}: sparsity = {sparsity:.2%}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_pruning()
