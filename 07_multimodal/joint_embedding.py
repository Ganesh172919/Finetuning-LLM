"""
################################################################################
JOINT EMBEDDING — SHARED EMBEDDING SPACE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Joint Embedding?
    Mapping different modalities to a shared embedding space.

Used by:
    - CLIP: Vision and language
    - ImageBind: Multiple modalities
    - MetaCLIP: Better CLIP training

Interview Questions:
    Q: "What is a joint embedding space?"
    A: A shared space where different modalities can be compared.
       Similar concepts have similar embeddings regardless of modality.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: JOINT EMBEDDING
################################################################################

class JointEmbedding:
    """
    Joint Embedding
    ===============

    Maps multiple modalities to shared space.
    """

    def __init__(self, d_embed: int = 512):
        self.d_embed = d_embed

    def embed_vision(self, vision_features: np.ndarray) -> np.ndarray:
        """Map vision to joint space."""
        return np.random.randn(*vision_features.shape[:-1], self.d_embed) * 0.1

    def embed_text(self, text_features: np.ndarray) -> np.ndarray:
        """Map text to joint space."""
        return np.random.randn(*text_features.shape[:-1], self.d_embed) * 0.1

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute similarity in joint space."""
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        return float(np.dot(emb1.flatten(), emb2.flatten()))


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_joint_embedding():
    """Demonstrate joint embedding."""
    print("=" * 70)
    print("JOINT EMBEDDING DEMONSTRATION")
    print("=" * 70)

    model = JointEmbedding(d_embed=64)
    vision = np.random.randn(128)
    text = np.random.randn(128)

    v_emb = model.embed_vision(vision)
    t_emb = model.embed_text(text)
    sim = model.similarity(v_emb, t_emb)

    print(f"Vision embedding: {v_emb.shape}")
    print(f"Text embedding: {t_emb.shape}")
    print(f"Similarity: {sim:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_joint_embedding()
