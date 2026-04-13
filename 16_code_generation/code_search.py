"""
################################################################################
CODE SEARCH — SEMANTIC SEARCH OVER CODE REPOSITORIES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Code Search?
    Finding code snippets by natural language queries or code similarity.
    Uses contrastive learning to embed code and queries in the same space.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


################################################################################
# SECTION 1: CODE EMBEDDER
################################################################################

class CodeEmbedder:
    """
    Embed code snippets into dense vectors.

    Interview Question:
        "How does semantic code search work?"
        Embed code and queries into the same vector space using
        contrastive learning. Train with (query, positive_code, negative_code)
        triplets using InfoNCE loss. At search time, embed query, find
        nearest code embeddings.
    """

    def __init__(self, d_model: int = 384):
        self.d_model = d_model
        # Simulated embedding weights
        self.weights = np.random.randn(1000, d_model) * 0.02

    def tokenize_code(self, code: str) -> List[str]:
        """Tokenize code (split by camelCase, snake_case, etc.)."""
        tokens = []
        for word in code.replace('_', ' ').replace('(', ' ').replace(')', ' ').split():
            # Split camelCase
            current = ""
            for c in word:
                if c.isupper() and current:
                    tokens.append(current.lower())
                    current = c
                else:
                    current += c
            if current:
                tokens.append(current.lower())
        return tokens

    def embed(self, code: str) -> np.ndarray:
        """
        Embed code into a vector.

        Args:
            code: Code string

        Returns:
            Embedding vector (d_model,)
        """
        tokens = self.tokenize_code(code)
        # Simulate: hash-based embedding
        hash_val = sum(hash(t) for t in tokens)
        np.random.seed(abs(hash_val) % 2**31)
        embedding = np.random.randn(self.d_model)
        return embedding / np.linalg.norm(embedding)


################################################################################
# SECTION 2: CONTRASTIVE CODE SEARCH
################################################################################

class ContrastiveCodeSearch:
    """
    Train code search with contrastive learning.

    InfoNCE loss: L = -log(exp(sim(q,c+)/tau) / sum(exp(sim(q,c)/tau)))

    Interview Question:
        "How do you train a code search model?"
        Use contrastive learning: (1) embed query and code separately,
        (2) maximize similarity for correct (query, code) pairs,
        (3) minimize for incorrect pairs using InfoNCE loss,
        (4) use in-batch negatives for efficiency.
    """

    def __init__(self, d_model: int = 384, temperature: float = 0.07):
        self.d_model = d_model
        self.temperature = temperature
        self.code_embedder = CodeEmbedder(d_model)
        self.query_embedder = CodeEmbedder(d_model)

    def compute_similarity(self, query_emb: np.ndarray,
                           code_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and code embeddings."""
        return code_embs @ query_emb

    def info_nce_loss(self, query_emb: np.ndarray,
                      positive_emb: np.ndarray,
                      negative_embs: np.ndarray) -> float:
        """
        Compute InfoNCE loss.

        Args:
            query_emb: Query embedding
            positive_emb: Positive code embedding
            negative_embs: Negative code embeddings

        Returns:
            InfoNCE loss value
        """
        pos_sim = np.dot(query_emb, positive_emb) / self.temperature
        neg_sims = negative_embs @ query_emb / self.temperature
        all_sims = np.concatenate([[pos_sim], neg_sims])
        log_softmax = all_sims - np.log(np.sum(np.exp(all_sims)))
        return -log_softmax[0]

    def search(self, query: str, code_snippets: List[str], top_k: int = 5) -> List[Dict]:
        """
        Search for code snippets matching a query.

        Args:
            query: Natural language query
            code_snippets: Code snippets to search
            top_k: Number of results

        Returns:
            Ranked results with scores
        """
        query_emb = self.query_embedder.embed(query)
        code_embs = np.array([self.code_embedder.embed(c) for c in code_snippets])
        similarities = self.compute_similarity(query_emb, code_embs)

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [
            {'code': code_snippets[i], 'score': similarities[i], 'rank': rank + 1}
            for rank, i in enumerate(top_indices)
        ]


################################################################################
# SECTION 3: CODE CLONE DETECTOR
################################################################################

class CodeCloneDetector:
    """
    Detect code clones (similar code).

    Types:
    - Type 1: Exact copies (modulo whitespace)
    - Type 2: Renamed copies (same structure, different names)
    - Type 3: Near-miss copies (small modifications)

    Interview Question:
        "How do you detect code clones?"
        Embed code snippets, compute pairwise similarity. Type 1: exact
        match. Type 2: same AST, different names. Type 3: embedding
        similarity above threshold. Use both token-level and structural
        features for robust detection.
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.embedder = CodeEmbedder()

    def detect(self, code_snippets: List[str]) -> List[Dict]:
        """
        Detect clone pairs.

        Args:
            code_snippets: List of code snippets

        Returns:
            List of clone pairs with similarity scores
        """
        n = len(code_snippets)
        embeddings = [self.embedder.embed(c) for c in code_snippets]

        clones = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j])
                if sim >= self.threshold:
                    clones.append({
                        'idx1': i, 'idx2': j,
                        'similarity': sim,
                        'type': self._classify_clone(code_snippets[i], code_snippets[j], sim)
                    })

        return clones

    def _classify_clone(self, code1: str, code2: str, sim: float) -> str:
        """Classify clone type."""
        if code1.strip() == code2.strip():
            return "Type 1 (exact)"
        elif sim > 0.95:
            return "Type 2 (renamed)"
        else:
            return "Type 3 (near-miss)"


################################################################################
# SECTION 4: DEMONSTRATION
################################################################################

def demonstrate_code_search():
    """Demonstrate code search."""
    print("=" * 70)
    print("CODE SEARCH DEMONSTRATION")
    print("=" * 70)

    snippets = [
        "def fibonacci(n): return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)",
        "def factorial(n): return 1 if n<=1 else n*factorial(n-1)",
        "def binary_search(arr, target): pass",
        "def quicksort(arr): pass",
        "def merge_sort(arr): pass",
        "def fib(n): return n if n<2 else fib(n-1)+fib(n-2)",
    ]

    # Code Search
    print("\n1. CODE SEARCH")
    print("-" * 40)
    searcher = ContrastiveCodeSearch()
    results = searcher.search("recursive fibonacci function", snippets)
    for r in results:
        print(f"  #{r['rank']}: {r['score']:.3f} — {r['code'][:50]}...")

    # Clone Detection
    print("\n2. CLONE DETECTION")
    print("-" * 40)
    detector = CodeCloneDetector(threshold=0.7)
    clones = detector.detect(snippets)
    for c in clones:
        print(f"  {c['type']}: [{c['idx1']}] & [{c['idx2']}] ({c['similarity']:.3f})")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_code_search()
