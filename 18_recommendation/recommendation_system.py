"""
################################################################################
RECOMMENDATION SYSTEMS — PERSONALIZED SUGGESTIONS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Recommendation Systems?
    Systems that suggest items to users based on preferences.

Types:
    1. Collaborative filtering: based on user behavior
    2. Content-based: based on item features
    3. Hybrid: combine both
    4. Sequential: based on user history

Interview Questions:
    Q: "How do recommendation systems work?"
    A: Collaborative filtering uses user-item interactions.
       Content-based uses item features.
       Modern systems use deep learning (two-tower, transformers).

################################################################################
"""

import numpy as np
from typing import List, Tuple

################################################################################
# SECTION 1: TWO-TOWER MODEL
################################################################################

class TwoTowerModel:
    """
    Two-Tower Model
    ===============

    Architecture for recommendation:
    - User tower: encodes user features
    - Item tower: encodes item features
    - Score: dot product of embeddings

    Interview Questions:
        Q: "What is a two-tower model?"
        A: Separate encoders for users and items.
           Score = dot(user_embedding, item_embedding).
           Efficient for retrieval (precompute item embeddings).
    """

    def __init__(self, d_user: int, d_item: int, d_embed: int = 64):
        self.user_proj = np.random.randn(d_user, d_embed) * 0.02
        self.item_proj = np.random.randn(d_item, d_embed) * 0.02

    def encode_user(self, user_features: np.ndarray) -> np.ndarray:
        """Encode user to embedding."""
        emb = user_features @ self.user_proj
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    def encode_item(self, item_features: np.ndarray) -> np.ndarray:
        """Encode item to embedding."""
        emb = item_features @ self.item_proj
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    def score(self, user_emb: np.ndarray, item_emb: np.ndarray) -> np.ndarray:
        """Compute relevance score."""
        return np.sum(user_emb * item_emb, axis=-1)


################################################################################
# SECTION 2: COLLABORATIVE FILTERING
################################################################################

class CollaborativeFiltering:
    """
    Collaborative Filtering
    =======================

    Recommend based on similar users' preferences.

    Matrix Factorization:
    R ≈ U × V^T

    Where:
    - R: user-item rating matrix
    - U: user embedding matrix
    - V: item embedding matrix

    Interview Questions:
        Q: "What's the cold start problem?"
        A: New users/items have no interactions.
           Solution: content-based features, popularity-based.
    """

    def __init__(self, n_users: int, n_items: int, d_embed: int = 32):
        self.user_embed = np.random.randn(n_users, d_embed) * 0.1
        self.item_embed = np.random.randn(n_items, d_embed) * 0.1

    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """Predict ratings for user-item pairs."""
        user_emb = self.user_embed[user_id]
        item_embs = self.item_embed[item_ids]
        return item_embs @ user_emb

    def recommend(self, user_id: int, top_k: int = 5) -> List[int]:
        """Recommend top-k items for user."""
        scores = self.item_embed @ self.user_embed[user_id]
        return np.argsort(scores)[::-1][:top_k].tolist()


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_recommendation():
    """Demonstrate recommendation systems."""
    print("=" * 70)
    print("RECOMMENDATION SYSTEM DEMONSTRATION")
    print("=" * 70)

    # Two-tower
    print("\n--- Two-Tower Model ---")
    model = TwoTowerModel(d_user=32, d_item=64, d_embed=16)
    user_feat = np.random.randn(4, 32)
    item_feat = np.random.randn(10, 64)
    user_emb = model.encode_user(user_feat)
    item_emb = model.encode_item(item_feat)
    scores = model.score(user_emb[:1], item_emb)
    print(f"User embedding: {user_emb.shape}")
    print(f"Item embedding: {item_emb.shape}")
    print(f"Scores: {scores.shape}")

    # Collaborative filtering
    print("\n--- Collaborative Filtering ---")
    cf = CollaborativeFiltering(n_users=100, n_items=1000, d_embed=32)
    recommendations = cf.recommend(user_id=0, top_k=5)
    print(f"Top-5 recommendations: {recommendations}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_recommendation()
