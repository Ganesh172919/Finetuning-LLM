"""
################################################################################
IMAGE UNDERSTANDING — ANALYZING IMAGE CONTENT
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Image Understanding?
    Models that analyze and understand image content.

Key Tasks:
    - Image classification
    - Object detection
    - Semantic segmentation
    - Visual question answering

Interview Questions:
    Q: "How does image understanding work?"
    A: Extract features with CNN or ViT, then apply task-specific heads.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: IMAGE UNDERSTANDING MODEL
################################################################################

class ImageUnderstandingModel:
    """
    Image Understanding Model
    =========================

    Analyzes image content.

    Interview Questions:
        Q: "What's the difference between classification and detection?"
        A: Classification: what is in the image
           Detection: where are the objects
    """

    def __init__(self, d_model: int = 256, n_classes: int = 1000):
        self.d_model = d_model
        self.n_classes = n_classes

        # Feature extractor
        self.feature_proj = np.random.randn(768, d_model) * 0.02

        # Classification head
        self.classifier = np.random.randn(d_model, n_classes) * 0.02

    def classify(self, image_features: np.ndarray) -> np.ndarray:
        """Classify image."""
        features = image_features @ self.feature_proj
        return features @ self.classifier

    def detect(self, image_features: np.ndarray) -> np.ndarray:
        """Detect objects (simplified)."""
        return np.random.randn(10, 5)  # 10 objects, [x,y,w,h,score]


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_image_understanding():
    """Demonstrate image understanding."""
    print("=" * 70)
    print("IMAGE UNDERSTANDING DEMONSTRATION")
    print("=" * 70)

    model = ImageUnderstandingModel(d_model=64, n_classes=10)
    features = np.random.randn(768)
    logits = model.classify(features)
    print(f"Features: {features.shape}")
    print(f"Logits: {logits.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_image_understanding()
