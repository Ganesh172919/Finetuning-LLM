"""
################################################################################
SEQUENTIAL RECOMMENDATION — PREDICTING NEXT ITEM
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Sequential Recommendation?
    Predicting the next item based on user's interaction history.

Key Models:
    - SASRec: Self-Attentive Sequential Recommendation
    - BERT4Rec: BERT for Recommendation
    - GRU4Rec: GRU-based Recommendation

Interview Questions:
    Q: "What is sequential recommendation?"
    A: Predict next item from user's interaction sequence.
       Uses transformer or RNN to model temporal patterns.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: SEQUENTIAL RECOMMENDER
################################################################################

class SequentialRecommender:
    """
    Sequential Recommender
    ======================

    Predicts next item from interaction history.

    Interview Questions:
        Q: "How does sequential recommendation differ from collaborative filtering?"
        A: CF: static user-item matrix
           Sequential: models temporal order of interactions
    """

    def __init__(self, n_items: int, d_model: int = 64):
        self.n_items = n_items
        self.d_model = d_model

        # Item embeddings
        self.item_embed = np.random.randn(n_items, d_model) * 0.02

        # Prediction head
        self.predictor = np.random.randn(d_model, n_items) * 0.02

    def predict_next(self, item_sequence: List[int], top_k: int = 5) -> List[int]:
        """
        Predict next item.

        Args:
            item_sequence: User's interaction history
            top_k: Number of recommendations

        Returns:
            recommendations: Top-k item IDs
        """
        # Encode sequence
        embeddings = self.item_embed[item_sequence]
        sequence_emb = np.mean(embeddings, axis=0)

        # Predict
        scores = sequence_emb @ self.predictor

        # Top-k
        return np.argsort(scores)[::-1][:top_k].tolist()


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_sequential():
    """Demonstrate sequential recommendation."""
    print("=" * 70)
    print("SEQUENTIAL RECOMMENDATION DEMONSTRATION")
    print("=" * 70)

    model = SequentialRecommender(n_items=100, d_model=32)
    history = [1, 5, 12, 45, 23]
    recommendations = model.predict_next(history, top_k=5)
    print(f"History: {history}")
    print(f"Recommendations: {recommendations}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_sequential()


################################################################################
# REFERENCES
################################################################################

# [1] Kang, W., & McAuley, J. (2018). Self-Attentive Sequential Recommendation.

################################################################################
