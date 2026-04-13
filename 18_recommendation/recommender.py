"""
################################################################################
RECOMMENDER — RECOMMENDATION ENGINE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Recommender?
    A system that suggests items to users.

Interview Questions:
    1. "How do recommendation systems work?"
        Collaborative filtering, content-based, or hybrid approaches.

################################################################################
"""

import numpy as np
from typing import List, Dict

################################################################################
# SECTION 1: RECOMMENDER
################################################################################

class Recommender:
    """
    Recommendation System
    =====================

    Simple recommendation engine.
    """

    def __init__(self, n_items: int, n_features: int = 64):
        self.n_items = n_items
        self.n_features = n_features
        self.item_embeddings = np.random.randn(n_items, n_features) * 0.02

    def recommend(self, user_embedding: np.ndarray, top_k: int = 5) -> List[int]:
        """
        Recommend items for a user.

        Args:
            user_embedding: User preference vector
            top_k: Number of recommendations

        Returns:
            List of item indices
        """
        # Compute similarities
        similarities = np.dot(self.item_embeddings, user_embedding)

        # Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices.tolist()


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_recommender():
    """Demonstrate recommender."""
    print("=" * 70)
    print("RECOMMENDER DEMONSTRATION")
    print("=" * 70)

    rec = Recommender(n_items=100, n_features=32)
    user_emb = np.random.randn(32)
    recommendations = rec.recommend(user_emb, top_k=5)
    print(f"Top-5 recommendations: {recommendations}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_recommender()
