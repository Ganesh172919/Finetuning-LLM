"""
################################################################################
ALIGNMENT TRAINING — ALIGNING MODALITIES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Alignment Training?
    Training models to align features from different modalities
    in a shared embedding space.

Key Methods:
    - Contrastive learning (CLIP)
    - Matching loss
    - Cross-modal prediction

Interview Questions:
    Q: "How do you align vision and language?"
    A: Contrastive learning: pull matching pairs together,
       push non-matching pairs apart.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: ALIGNMENT TRAINER
################################################################################

class AlignmentTrainer:
    """
    Alignment Trainer
    =================

    Trains alignment between modalities.
    """

    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature

    def contrastive_loss(
        self,
        vision_embeds: np.ndarray,
        text_embeds: np.ndarray
    ) -> float:
        """
        Compute contrastive loss.

        Args:
            vision_embeds: [batch × d]
            text_embeds: [batch × d]

        Returns:
            loss: Contrastive loss
        """
        # Normalize
        vision_embeds = vision_embeds / np.linalg.norm(vision_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / np.linalg.norm(text_embeds, axis=-1, keepdims=True)

        # Similarity
        similarity = vision_embeds @ text_embeds.T / self.temperature

        # Labels: diagonal
        batch_size = vision_embeds.shape[0]
        labels = np.arange(batch_size)

        # Cross-entropy
        shifted = similarity - np.max(similarity, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
        loss = -np.mean(shifted[np.arange(batch_size), labels] - log_sum_exp)

        return loss


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_alignment():
    """Demonstrate alignment training."""
    print("=" * 70)
    print("ALIGNMENT TRAINING DEMONSTRATION")
    print("=" * 70)

    trainer = AlignmentTrainer()
    vision = np.random.randn(4, 64)
    text = np.random.randn(4, 64)
    loss = trainer.contrastive_loss(vision, text)
    print(f"Contrastive loss: {loss:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_alignment()
