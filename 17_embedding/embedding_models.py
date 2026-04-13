"""
################################################################################
EMBEDDING MODELS — SEMANTIC REPRESENTATIONS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Embedding Models?
    Models that convert text into dense vectors capturing semantic meaning.
    Similar texts produce similar vectors.

Key Models:
    - BERT (2018): First popular embedding model
    - Sentence-BERT (2019): Efficient sentence embeddings
    - E5 (2022): State-of-the-art embeddings
    - GTE (2023): General text embeddings
    - BGE (2023): BAAI general embeddings

Applications:
    - Semantic search
    - RAG retrieval
    - Clustering
    - Classification
    - Recommendation

Interview Questions:
        Q: "How do embedding models work?"
        A: Encode text into dense vectors using transformers.
           Train with contrastive learning on text pairs.

        Q: "What makes a good embedding?"
        A: Captures semantic meaning, invariant to paraphrasing,
           efficient to compute and compare.

################################################################################
"""

import numpy as np
from typing import List, Tuple
import math

################################################################################
# SECTION 1: SENTENCE TRANSFORMER
################################################################################

class SentenceTransformer:
    """
    Sentence Transformer
    ====================

    Produces sentence-level embeddings using mean pooling.

    Architecture:
    Token Embeddings → Transformer → Mean Pooling → Sentence Embedding

    Training:
    Contrastive learning on (anchor, positive, negative) triplets.

    Interview Questions:
        Q: "What's the difference between word and sentence embeddings?"
        A: Word: one vector per word
           Sentence: one vector per sentence (aggregated)
    """

    def __init__(self, d_model: int = 384, max_seq_len: int = 128):
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embed = np.random.randn(10000, d_model) * 0.02

        # Transformer layers (simplified)
        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02

        # Projection head
        self.projection = np.random.randn(d_model, d_model) * 0.02

    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Encode tokens to sentence embedding.

        Args:
            token_ids: [batch × seq_len]

        Returns:
            embeddings: [batch × d_model]
        """
        batch, seq_len = token_ids.shape

        # Token embedding
        x = self.token_embed[token_ids]  # [batch × seq × d_model]

        # Simplified transformer
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        scores = Q @ K.transpose(0, 2, 1) / math.sqrt(self.d_model)
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        x = weights @ V

        # Mean pooling
        embedding = np.mean(x, axis=1)  # [batch × d_model]

        # Normalize
        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)

        return embedding


################################################################################
# SECTION 2: CONTRASTIVE LOSS
################################################################################

class ContrastiveLoss:
    """
    Contrastive Loss for Embedding Training
    =========================================

    Trains embeddings to be similar for related texts,
    different for unrelated texts.

    Loss types:
    1. Triplet loss: anchor, positive, negative
    2. InfoNCE: contrastive loss with negatives
    3. Multiple negatives ranking loss

    Interview Questions:
        Q: "How do you train embedding models?"
        A: Use contrastive learning. Pull similar texts together,
           push different texts apart.
    """

    def __init__(self, temperature: float = 0.05):
        self.temperature = temperature

    def info_nce_loss(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negatives: np.ndarray
    ) -> float:
        """
        InfoNCE loss.

        Args:
            anchor: [batch × d]
            positive: [batch × d]
            negatives: [batch × n_neg × d]

        Returns:
            loss: InfoNCE loss
        """
        # Positive similarity
        pos_sim = np.sum(anchor * positive, axis=-1) / self.temperature

        # Negative similarities
        neg_sim = np.sum(anchor[:, np.newaxis, :] * negatives, axis=-1) / self.temperature

        # InfoNCE loss
        logits = np.concatenate([pos_sim[:, np.newaxis], neg_sim], axis=1)
        labels = np.zeros(len(anchor), dtype=int)

        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
        loss = -np.mean(shifted[np.arange(len(labels)), labels] - log_sum_exp)

        return loss


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_embeddings():
    """Demonstrate embedding models."""
    print("=" * 70)
    print("EMBEDDING MODELS DEMONSTRATION")
    print("=" * 70)

    # Sentence transformer
    print("\n--- Sentence Transformer ---")
    model = SentenceTransformer(d_model=128)
    token_ids = np.random.randint(0, 1000, (4, 20))
    embeddings = model.encode(token_ids)
    print(f"Input: {token_ids.shape}")
    print(f"Embeddings: {embeddings.shape}")

    # Similarity
    print("\n--- Similarity ---")
    sim = np.dot(embeddings[0], embeddings[1])
    print(f"Similarity between first two: {sim:.4f}")

    # Contrastive loss
    print("\n--- Contrastive Loss ---")
    loss_fn = ContrastiveLoss(temperature=0.05)
    anchor = np.random.randn(4, 128)
    positive = np.random.randn(4, 128)
    negatives = np.random.randn(4, 8, 128)
    loss = loss_fn.info_nce_loss(anchor, positive, negatives)
    print(f"InfoNCE loss: {loss:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_embeddings()


################################################################################
# REFERENCES
################################################################################

# [1] Reimers, N., & Gurevych, I. (2019). Sentence-BERT.
# [2] Wang, L., et al. (2022). Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5).

################################################################################
