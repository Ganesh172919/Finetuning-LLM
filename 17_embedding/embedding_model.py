"""
################################################################################
EMBEDDING MODEL — TEXT TO VECTORS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is an Embedding Model?
    A model that converts text into dense vectors.
    Similar texts produce similar vectors.

Why does it matter?
    Embeddings are the foundation of:
    - Semantic search
    - RAG systems
    - Recommendation
    - Clustering

Interview Questions:
    1. "How do embedding models work?"
        Encode text into dense vectors using transformers.
        Train with contrastive learning on text pairs.

    2. "What makes a good embedding?"
        Captures semantic meaning, invariant to paraphrasing,
        efficient to compute and compare.

################################################################################
"""

import numpy as np
from typing import List

################################################################################
# SECTION 1: EMBEDDING MODEL
################################################################################

class EmbeddingModel:
    """
    Embedding Model
    ===============

    Converts text to dense vectors.

    In production, use models like:
    - text-embedding-3-small (OpenAI)
    - e5-large-v2 (Microsoft)
    - gte-large (Alibaba)
    - bge-large (BAAI)
    """

    def __init__(self, d_model: int = 384):
        self.d_model = d_model
        # Simplified embedding weights
        self.embedding = np.random.randn(10000, d_model) * 0.02

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to vector.

        Args:
            text: Input text

        Returns:
            embedding: [d_model]
        """
        # Simplified: hash text to get consistent embedding
        hash_val = hash(text) % 10000
        return self.embedding[hash_val]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts."""
        return np.array([self.encode(t) for t in texts])


################################################################################
# SECTION 2: SENTENCE TRANSFORMER
################################################################################

class SentenceTransformer:
    """
    Sentence Transformer
    ====================

    Specialized model for sentence embeddings.
    Uses mean pooling over token embeddings.
    """

    def __init__(self, d_model: int = 384):
        self.d_model = d_model
        self.model = EmbeddingModel(d_model)

    def encode(self, sentences: List[str]) -> np.ndarray:
        """
        Encode sentences to vectors.

        Args:
            sentences: List of sentences

        Returns:
            embeddings: [n × d_model]
        """
        return self.model.encode_batch(sentences)

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Similarity score
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_embeddings():
    """Demonstrate embedding concepts."""
    print("=" * 70)
    print("EMBEDDING MODEL DEMONSTRATION")
    print("=" * 70)

    # Create model
    model = SentenceTransformer(d_model=128)

    # Encode sentences
    sentences = [
        "The cat sat on the mat",
        "A feline rested on the rug",
        "The dog played in the park"
    ]

    embeddings = model.encode(sentences)
    print(f"\nEmbeddings shape: {embeddings.shape}")

    # Compute similarities
    print("\n--- Similarities ---")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = model.similarity(embeddings[i], embeddings[j])
            print(f"'{sentences[i][:30]}...' vs '{sentences[j][:30]}...': {sim:.3f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_embeddings()
