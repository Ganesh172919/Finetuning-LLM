"""
################################################################################
RETRIEVER — DOCUMENT RETRIEVAL FOR RAG
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Retriever?
    A system that finds relevant documents for a query.
    In RAG, the retriever finds context for the LLM.

Types:
    1. Dense: Use embeddings for semantic search
    2. Sparse: Use keywords (BM25)
    3. Hybrid: Combine both

Interview Questions:
    1. "What's the difference between dense and sparse retrieval?"
       Dense: semantic similarity (embeddings)
       Sparse: keyword matching (BM25)
       Hybrid: best of both worlds

    2. "When should I use dense vs sparse?"
       Dense: semantic understanding needed
       Sparse: exact keyword matching needed
       Hybrid: most production systems

################################################################################
"""

import numpy as np
from typing import List, Dict, Tuple

################################################################################
# SECTION 1: DENSE RETRIEVER
################################################################################

class DenseRetriever:
    """
    Dense Retriever
    ===============

    Uses embeddings for semantic search.

    Process:
    1. Embed query and documents
    2. Compute cosine similarity
    3. Return top-K most similar

    Interview Question:
        "How does dense retrieval work?"
        Embed query and documents into vectors.
        Find documents with highest cosine similarity to query.
        This captures semantic meaning, not just keywords.
    """

    def __init__(self, d_model: int = 384):
        self.d_model = d_model
        self.document_embeddings = None
        self.documents = []

    def add_documents(self, documents: List[str], embeddings: np.ndarray):
        """Add documents with embeddings."""
        self.documents = documents
        self.document_embeddings = embeddings

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        if self.document_embeddings is None:
            return []

        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = self.document_embeddings / (
            np.linalg.norm(self.document_embeddings, axis=1, keepdims=True) + 1e-8
        )
        similarities = np.dot(doc_norms, query_norm)

        # Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(similarities[idx])
            })

        return results


################################################################################
# SECTION 2: SPARSE RETRIEVER (BM25)
################################################################################

class SparseRetriever:
    """
    Sparse Retriever (BM25)
    ========================

    Uses keyword matching for retrieval.

    BM25 Formula:
    score(q, d) = Σ IDF(qi) × (f(qi, d) × (k1 + 1)) / (f(qi, d) + k1 × (1 - b + b × |d|/avgdl))

    Interview Question:
        "What is BM25?"
        A keyword-based retrieval algorithm.
        It considers term frequency and document length.
        Good for exact keyword matching.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = {}
        self.avg_doc_len = 0

    def add_documents(self, documents: List[str]):
        """Add documents."""
        self.documents = documents

        # Compute document frequencies
        for doc in documents:
            words = set(doc.lower().split())
            for word in words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        # Average document length
        self.avg_doc_len = np.mean([len(d.split()) for d in documents])

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using BM25."""
        query_terms = query.lower().split()
        n_docs = len(self.documents)

        scores = []
        for doc in self.documents:
            doc_words = doc.lower().split()
            doc_len = len(doc_words)
            score = 0.0

            for term in query_terms:
                if term not in self.doc_freqs:
                    continue

                # Term frequency
                tf = doc_words.count(term)

                # IDF
                df = self.doc_freqs[term]
                idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)

                # BM25 score
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
                score += idf * tf_norm

            scores.append(score)

        # Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(scores[idx])
            })

        return results


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_retriever():
    """Demonstrate retriever concepts."""
    print("=" * 70)
    print("RETRIEVER DEMONSTRATION")
    print("=" * 70)

    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Natural language processing deals with text",
        "Computer vision processes images and videos"
    ]

    # Dense retriever
    print("\n--- Dense Retriever ---")
    dense = DenseRetriever(d_model=64)
    embeddings = np.random.randn(len(documents), 64)
    dense.add_documents(documents, embeddings)
    query_emb = np.random.randn(64)
    results = dense.search(query_emb, top_k=2)
    for r in results:
        print(f"Score: {r['score']:.3f} - {r['document'][:50]}...")

    # Sparse retriever
    print("\n--- Sparse Retriever ---")
    sparse = SparseRetriever()
    sparse.add_documents(documents)
    results = sparse.search("neural network", top_k=2)
    for r in results:
        print(f"Score: {r['score']:.3f} - {r['document'][:50]}...")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_retriever()


################################################################################
# REFERENCES
################################################################################

# [1] Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.

################################################################################
