"""
################################################################################
SigLIP — SIGMOID LOSS FOR LANGUAGE-IMAGE PRE-TRAINING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is SigLIP?
    SigLIP improves CLIP by using sigmoid loss instead of softmax.
    This allows training with smaller batch sizes and is more efficient.

CLIP uses softmax (contrastive):
    L = -log(exp(sim(i,t)/τ) / Σ exp(sim(i,tⱼ)/τ))

SigLIP uses sigmoid (binary):
    L = -Σ log σ(y_ij × sim(i,t)/τ)

Where y_ij = 1 for matching pairs, -1 for non-matching.

Why SigLIP?
    CLIP: requires large batch sizes (32K+) for enough negatives
    SigLIP: works with smaller batches (sigmoid doesn't need all negatives)

Benefits:
    - More efficient training
    - Better performance
    - Simpler implementation

Interview Questions:
        Q: "What's the difference between CLIP and SigLIP?"
        A: CLIP uses softmax (contrastive) loss, SigLIP uses sigmoid (binary).
           SigLIP is more efficient and works with smaller batches.

################################################################################
"""

import numpy as np
from typing import Tuple
import math

################################################################################
# SECTION 1: SigLIP MODEL
################################################################################

class SigLIP:
    """
    SigLIP: Sigmoid Loss for Language-Image Pre-training
    =====================================================

    Improves CLIP with sigmoid loss.

    Architecture:
        Image → Vision Encoder → Image Embedding
        Text → Text Encoder → Text Embedding

        Loss: Sigmoid contrastive loss
    """

    def __init__(self, d_model: int = 768, d_embed: int = 512):
        self.d_model = d_model
        self.d_embed = d_embed

        # Vision encoder (simplified)
        self.vision_proj = np.random.randn(d_model, d_embed) * 0.02

        # Text encoder (simplified)
        self.text_proj = np.random.randn(d_model, d_embed) * 0.02

        # Temperature
        self.logit_scale = np.log(1 / 0.07)

    def encode_image(self, image_features: np.ndarray) -> np.ndarray:
        """Encode image to embedding."""
        emb = image_features @ self.vision_proj
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    def encode_text(self, text_features: np.ndarray) -> np.ndarray:
        """Encode text to embedding."""
        emb = text_features @ self.text_proj
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    def sigmoid_loss(
        self,
        image_embeds: np.ndarray,
        text_embeds: np.ndarray
    ) -> float:
        """
        Compute SigLIP sigmoid loss.

        Unlike CLIP's softmax, SigLIP uses sigmoid:
        L = -Σ log σ(y_ij × sim(i,j) / τ)

        Where y_ij = 1 for matching, y_ij = -1 for non-matching.

        Interview Questions:
            Q: "Why is sigmoid loss better than softmax?"
            A: Softmax needs all negatives in the batch.
               Sigmoid treats each pair independently.
               More efficient, especially with small batches.
        """
        batch_size = image_embeds.shape[0]

        # Compute similarity matrix
        similarity = image_embeds @ text_embeds.T * np.exp(self.logit_scale)

        # Labels: diagonal is matching
        labels = np.eye(batch_size) * 2 - 1  # 1 for match, -1 for non-match

        # Sigmoid loss
        loss = -np.mean(np.log(1 / (1 + np.exp(-labels * similarity)) + 1e-8))

        return loss


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_siglip():
    """Demonstrate SigLIP."""
    print("=" * 70)
    print("SigLIP DEMONSTRATION")
    print("=" * 70)

    model = SigLIP(d_model=128, d_embed=64)

    # Create features
    image_features = np.random.randn(4, 128)
    text_features = np.random.randn(4, 128)

    # Encode
    image_embeds = model.encode_image(image_features)
    text_embeds = model.encode_text(text_features)

    print(f"Image embeds: {image_embeds.shape}")
    print(f"Text embeds: {text_embeds.shape}")

    # Loss
    loss = model.sigmoid_loss(image_embeds, text_embeds)
    print(f"SigLIP loss: {loss:.4f}")

    # Compare with CLIP
    print("\n--- SigLIP vs CLIP ---")
    print("SigLIP: Sigmoid loss, works with small batches")
    print("CLIP: Softmax loss, needs large batches")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_siglip()


################################################################################
# REFERENCES
################################################################################

# [1] Zhai, X., et al. (2023). Sigmoid Loss for Language Image Pre-Training.

################################################################################
