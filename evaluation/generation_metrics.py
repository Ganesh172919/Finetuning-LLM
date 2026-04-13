"""
################################################################################
GENERATION METRICS — EVALUATING TEXT QUALITY
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Generation Metrics?
    Measures of text generation quality.

Key Metrics:
    - BLEU: N-gram overlap (translation)
    - ROUGE: Recall-oriented (summarization)
    - BERTScore: Semantic similarity
    - Perplexity: Model confidence

Interview Questions:
    Q: "How do you evaluate generated text?"
    A: BLEU for translation, ROUGE for summarization,
       BERTScore for semantic similarity.

################################################################################
"""

import numpy as np
from typing import List

################################################################################
# SECTION 1: BLEU SCORE
################################################################################

def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4
) -> float:
    """
    Compute BLEU score.

    BLEU measures n-gram overlap between prediction and reference.

    Interview Questions:
        Q: "What are the limitations of BLEU?"
        A: Only measures n-gram overlap, not semantic similarity.
    """
    # Simplified implementation
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        if len(ref_tokens) > 0:
            overlap = len(pred_tokens & ref_tokens)
            precision = overlap / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = overlap / len(ref_tokens)
            scores.append(2 * precision * recall / (precision + recall + 1e-8))

    return float(np.mean(scores)) if scores else 0.0


################################################################################
# SECTION 2: ROUGE SCORE
################################################################################

def compute_rouge(
    predictions: List[str],
    references: List[str]
) -> dict:
    """
    Compute ROUGE score.

    ROUGE measures recall of reference content.

    Interview Questions:
        Q: "When should I use ROUGE vs BLEU?"
        A: ROUGE for summarization, BLEU for translation.
    """
    rouge1_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        # ROUGE-1
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        if len(ref_set) > 0:
            overlap = len(pred_set & ref_set)
            rouge1_scores.append(overlap / len(ref_set))

        # ROUGE-L (simplified)
        # Use LCS-based recall
        common = len(pred_set & ref_set)
        if len(ref_tokens) > 0:
            rougeL_scores.append(common / len(ref_tokens))

    return {
        'rouge-1': float(np.mean(rouge1_scores)) if rouge1_scores else 0.0,
        'rouge-L': float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
    }


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_generation_metrics():
    """Demonstrate generation metrics."""
    print("=" * 70)
    print("GENERATION METRICS DEMONSTRATION")
    print("=" * 70)

    predictions = ["the cat sat on the mat"]
    references = ["the cat sat on the mat"]

    # BLEU
    bleu = compute_bleu(predictions, references)
    print(f"BLEU: {bleu:.4f}")

    # ROUGE
    rouge = compute_rouge(predictions, references)
    print(f"ROUGE-1: {rouge['rouge-1']:.4f}")
    print(f"ROUGE-L: {rouge['rouge-L']:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_generation_metrics()
