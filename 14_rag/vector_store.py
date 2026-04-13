"""
################################################################################
VECTOR STORE — DATABASE FOR EMBEDDINGS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Vector Store?
    A database optimized for storing and searching high-dimensional vectors.
    Used in RAG to find relevant documents.

Why does it matter?
    Vector stores enable:
    - Semantic search
    - Similarity matching
    - RAG retrieval

Interview Questions:
    1. "What is a vector database?"
        A database optimized for storing and searching embeddings.

    2. "How does similarity search work?"
        Compute distance (cosine, L2) between query and all vectors.
        Return closest matches.

################################################################################
"""

import numpy as np
from typing import List, Dict, Optional

################################################################################
# SECTION 1: VECTOR STORE
################################################################################

class VectorStore:
    """
    Vector Store
    ============

    Simple in-memory vector store for demonstrations.

    In production, use:
    - Pinecone
    - Weaviate
    - ChromaDB
    - FAISS
    """

    def __init__(self, d_model: int = 384):
        self.d_model = d_model
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict] = []

    def add(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict]] = None
    ):
        """Add vectors to store."""
        for i, vec in enumerate(vectors):
            self.vectors.append(vec)
            self.metadata.append(metadata[i] if metadata else {})

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """Search for similar vectors."""
        if not self.vectors:
            return []

        # Stack all vectors
        all_vectors = np.stack(self.vectors)

        # Cosine similarity
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        vec_norms = all_vectors / (np.linalg.norm(all_vectors, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(vec_norms, query_norm)

        # Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'score': float(similarities[idx]),
                'metadata': self.metadata[idx]
            })

        return results


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_vector_store():
    """Demonstrate vector store."""
    print("=" * 70)
    print("VECTOR STORE DEMONSTRATION")
    print("=" * 70)

    store = VectorStore(d_model=64)

    # Add vectors
    vectors = np.random.randn(10, 64)
    metadata = [{'text': f'Document {i}'} for i in range(10)]
    store.add(vectors, metadata)
    print(f"Added {len(store.vectors)} vectors")

    # Search
    query = np.random.randn(64)
    results = store.search(query, top_k=3)
    print(f"\nTop-3 results:")
    for r in results:
        print(f"  Score: {r['score']:.3f} - {r['metadata']['text']}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_vector_store()
