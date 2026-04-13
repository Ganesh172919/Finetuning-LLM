"""
################################################################################
LOSS FUNCTIONS — MEASURING MODEL ERROR
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Loss Functions?
    Functions that measure how wrong the model's predictions are.

Key Loss Functions:
    - Cross-Entropy: Classification
    - MSE: Regression
    - Contrastive: Similarity learning

Interview Questions:
    Q: "What loss function should I use for LLMs?"
    A: Cross-entropy loss for next token prediction.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: LOSS FUNCTIONS
################################################################################

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Cross-entropy loss.

    L = -log(P(correct token))

    Interview Questions:
        Q: "Why cross-entropy for language models?"
        A: Measures how well model predicts correct tokens.
           Equivalent to maximizing likelihood.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
    log_probs = shifted[np.arange(len(targets)), targets] - log_sum_exp
    return -np.mean(log_probs)


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error loss."""
    return np.mean((predictions - targets) ** 2)


def cosine_similarity_loss(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
    """Cosine similarity loss."""
    sim = np.sum(embeddings1 * embeddings2, axis=-1)
    norms = np.linalg.norm(embeddings1, axis=-1) * np.linalg.norm(embeddings2, axis=-1)
    return -np.mean(sim / (norms + 1e-8))


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_loss_functions():
    """Demonstrate loss functions."""
    print("=" * 70)
    print("LOSS FUNCTIONS DEMONSTRATION")
    print("=" * 70)

    # Cross-entropy
    print("\n--- Cross-Entropy ---")
    logits = np.random.randn(4, 100)
    targets = np.random.randint(0, 100, 4)
    ce_loss = cross_entropy_loss(logits, targets)
    print(f"Cross-entropy: {ce_loss:.4f}")

    # MSE
    print("\n--- MSE ---")
    preds = np.random.randn(4, 10)
    targets = np.random.randn(4, 10)
    mse = mse_loss(preds, targets)
    print(f"MSE: {mse:.4f}")

    # Cosine similarity
    print("\n--- Cosine Similarity ---")
    emb1 = np.random.randn(4, 64)
    emb2 = np.random.randn(4, 64)
    cos_loss = cosine_similarity_loss(emb1, emb2)
    print(f"Cosine loss: {cos_loss:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_loss_functions()
