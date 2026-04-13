"""
################################################################################
MODEL COMPRESSION — REDUCING MODEL SIZE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Model Compression?
    Techniques to reduce model size while maintaining quality.

Key Techniques:
    - Quantization: Reduce precision
    - Pruning: Remove unnecessary weights
    - Distillation: Train smaller model
    - Factorization: Decompose weight matrices

Interview Questions:
    Q: "How do you compress models for deployment?"
    A: Quantization (int8/int4), pruning, distillation.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: PRUNING
################################################################################

class ModelPruner:
    """
    Model Pruner
    ============

    Removes unnecessary weights from models.

    Types:
    1. Unstructured: remove individual weights
    2. Structured: remove entire neurons/layers

    Interview Questions:
        Q: "What is pruning?"
        A: Removing weights that contribute little to the output.
           Sparse models are smaller and faster.
    """

    def __init__(self, sparsity: float = 0.5):
        self.sparsity = sparsity

    def prune(self, weights: np.ndarray) -> np.ndarray:
        """
        Prune weights by magnitude.

        Args:
            weights: Weight matrix

        Returns:
            pruned: Pruned weight matrix (with zeros)
        """
        threshold = np.percentile(np.abs(weights), self.sparsity * 100)
        mask = np.abs(weights) > threshold
        return weights * mask


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_compression():
    """Demonstrate model compression."""
    print("=" * 70)
    print("MODEL COMPRESSION DEMONSTRATION")
    print("=" * 70)

    # Pruning
    print("\n--- Pruning ---")
    pruner = ModelPruner(sparsity=0.5)
    weights = np.random.randn(10, 10)
    pruned = pruner.prune(weights)
    sparsity = np.mean(pruned == 0)
    print(f"Original shape: {weights.shape}")
    print(f"Sparsity: {sparsity:.2%}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_compression()
