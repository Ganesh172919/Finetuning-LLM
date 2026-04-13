"""
################################################################################
FUSION LAYERS — COMBINING MULTIMODAL FEATURES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Fusion Layers?
    Layers that combine features from different modalities.

Types:
    1. Concatenation: Simple concatenation
    2. Addition: Element-wise addition
    3. Gating: Learned combination
    4. Cross-attention: Attend across modalities

Interview Questions:
    Q: "How do you combine vision and language features?"
    A: Concatenation (simple), cross-attention (powerful),
       or gating (adaptive).

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: FUSION LAYER
################################################################################

class FusionLayer:
    """
    Fusion Layer
    ============

    Combines features from different modalities.
    """

    def __init__(self, d_model: int, n_modalities: int = 2):
        self.d_model = d_model
        self.gate = np.random.randn(d_model * n_modalities, n_modalities) * 0.02

    def fuse(self, features: List[np.ndarray]) -> np.ndarray:
        """
        Fuse multimodal features.

        Args:
            features: List of feature arrays

        Returns:
            fused: Fused features
        """
        # Concatenation + gating
        concat = np.concatenate(features, axis=-1)
        gate_logits = concat @ self.gate
        gate_weights = np.exp(gate_logits) / np.sum(np.exp(gate_logits), axis=-1, keepdims=True)

        stacked = np.stack(features, axis=-1)
        return np.sum(stacked * gate_weights[..., np.newaxis, :], axis=-1)


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_fusion():
    """Demonstrate fusion layer."""
    print("=" * 70)
    print("FUSION LAYER DEMONSTRATION")
    print("=" * 70)

    fusion = FusionLayer(d_model=64, n_modalities=2)
    feat1 = np.random.randn(4, 64)
    feat2 = np.random.randn(4, 64)
    fused = fusion.fuse([feat1, feat2])
    print(f"Feature 1: {feat1.shape}")
    print(f"Feature 2: {feat2.shape}")
    print(f"Fused: {fused.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_fusion()
