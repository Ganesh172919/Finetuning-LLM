"""
################################################################################
MULTIMODAL FUSION — COMBINING MODALITIES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Multimodal Fusion?
    Combining information from different modalities (text, image, audio).

Types:
    1. Early fusion: combine raw inputs
    2. Late fusion: combine after processing
    3. Cross-attention: attend across modalities

Interview Questions:
    1. "How do you combine vision and language?"
        Cross-attention is most common. Text queries attend to image features.

    2. "What's the difference between early and late fusion?"
        Early: combine before processing
        Late: combine after processing

################################################################################
"""

import numpy as np
from typing import List

################################################################################
# SECTION 1: FUSION STRATEGIES
################################################################################

class MultimodalFusion:
    """
    Multimodal Fusion Strategies
    =============================

    Different ways to combine modalities.
    """

    def __init__(self, d_model: int):
        self.d_model = d_model

    def concatenate(self, features: List[np.ndarray]) -> np.ndarray:
        """Simple concatenation."""
        return np.concatenate(features, axis=-1)

    def add(self, features: List[np.ndarray]) -> np.ndarray:
        """Element-wise addition."""
        return np.sum(features, axis=0)

    def weighted_sum(
        self,
        features: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """Weighted sum of features."""
        result = np.zeros_like(features[0])
        for feat, w in zip(features, weights):
            result += w * feat
        return result


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_fusion():
    """Demonstrate fusion concepts."""
    print("=" * 70)
    print("MULTIMODAL FUSION DEMONSTRATION")
    print("=" * 70)

    fusion = MultimodalFusion(d_model=64)

    # Create features
    text_feat = np.random.randn(2, 64)
    image_feat = np.random.randn(2, 64)

    # Concatenation
    concat = fusion.concatenate([text_feat, image_feat])
    print(f"Concatenation: {concat.shape}")

    # Addition
    add = fusion.add([text_feat, image_feat])
    print(f"Addition: {add.shape}")

    # Weighted sum
    weighted = fusion.weighted_sum([text_feat, image_feat], [0.7, 0.3])
    print(f"Weighted sum: {weighted.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_fusion()
