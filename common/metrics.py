"""
################################################################################
METRICS — EVALUATION MEASURES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Metrics?
    Measures of model quality. Different tasks need different metrics.

Interview Questions:
    1. "What is perplexity?"
        How "surprised" the model is by text. Lower is better.

    2. "What's the difference between BLEU and ROUGE?"
        BLEU: precision-focused (translation)
        ROUGE: recall-focused (summarization)

################################################################################
"""

import numpy as np
from typing import List

################################################################################
# SECTION 1: METRICS
################################################################################

def compute_perplexity(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute perplexity.

    PPL = 2^(cross-entropy loss)
    """
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    shifted = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
    log_probs = shifted[np.arange(len(targets_flat)), targets_flat] - log_sum_exp
    ce = -np.mean(log_probs)

    return float(np.exp(ce))


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute accuracy."""
    return float(np.mean(predictions == targets))


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_metrics():
    """Demonstrate metrics."""
    print("=" * 70)
    print("METRICS DEMONSTRATION")
    print("=" * 70)

    # Perplexity
    logits = np.random.randn(2, 10, 100)
    targets = np.random.randint(0, 100, (2, 10))
    ppl = compute_perplexity(logits, targets)
    print(f"Perplexity: {ppl:.2f}")

    # Accuracy
    preds = np.array([1, 2, 3, 4, 5])
    targets = np.array([1, 2, 3, 5, 5])
    acc = compute_accuracy(preds, targets)
    print(f"Accuracy: {acc:.2%}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_metrics()
