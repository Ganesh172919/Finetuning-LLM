"""
################################################################################
EVALUATION METRICS FOR AI MODELS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Evaluation Metrics?
    Metrics quantify model performance. Different tasks need different metrics:
    - Language Modeling: Perplexity, bits per character
    - Classification: Accuracy, F1, precision, recall
    - Generation: BLEU, ROUGE, BERTScore
    - Reasoning: Exact match, chain-of-thought quality

Why do they matter?
    Metrics are how we:
    - Compare models objectively
    - Track training progress
    - Select best checkpoints
    - Report results in papers

################################################################################
"""

import numpy as np
from typing import List, Dict, Optional
from collections import Counter

################################################################################
# SECTION 1: PERPLEXITY
################################################################################

def compute_perplexity(
    logits: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Compute Perplexity
    ==================

    Definition: How "surprised" the model is by the text.
    Formula: PPL = 2^H where H is cross-entropy loss.

    Lower is better:
    - PPL = 1: model is certain (impossible in practice)
    - PPL = 10: model considers ~10 tokens equally likely
    - PPL = 100: model is quite uncertain
    - PPL = 1000: model is very uncertain

    Interpretation:
    If PPL = 50, the model is as confused as if choosing
    uniformly among 50 tokens at each position.

    Args:
        logits: Model predictions [batch × seq × vocab_size]
        targets: True token IDs [batch × seq]

    Returns:
        Perplexity (scalar)

    Interview Question:
        "What's a good perplexity for an LLM?"
        Depends on the dataset and model size.
        GPT-3 (175B) achieves ~20 perplexity on WikiText-103.
        Smaller models have higher perplexity.
    """
    vocab_size = logits.shape[-1]

    # Flatten
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Cross-entropy
    shifted = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
    log_probs = shifted[np.arange(len(targets_flat)), targets_flat] - log_sum_exp
    cross_entropy = -np.mean(log_probs)

    # Perplexity
    return float(np.exp(cross_entropy))


################################################################################
# SECTION 2: ACCURACY
################################################################################

def compute_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Compute Accuracy
    ================

    Definition: Fraction of correct predictions.

    Args:
        predictions: Predicted tokens [batch × seq] or [batch]
        targets: True tokens [batch × seq] or [batch]

    Returns:
        Accuracy (0 to 1)
    """
    return float(np.mean(predictions == targets))


################################################################################
# SECTION 3: BLEU SCORE
################################################################################

def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4
) -> float:
    """
    Compute BLEU Score
    ===================

    Definition: Bilingual Evaluation Understudy.
    Measures overlap of n-grams between prediction and reference.

    Formula:
    BLEU = BP × exp(Σ wₙ × log pₙ)

    Where:
    - BP: Brevity penalty (penalizes short predictions)
    - pₙ: n-gram precision
    - wₙ: weights (typically uniform: 1/4 for n=1..4)

    Used for: Machine translation, text generation

    Range: 0 to 1 (higher is better)

    Interview Question:
        "What are the limitations of BLEU?"
        BLEU only measures n-gram overlap, not semantic similarity.
        "The cat sat on the mat" and "A feline rested on the rug"
        have low BLEU but similar meaning.
    """
    # Tokenize
    pred_tokens = [p.lower().split() for p in predictions]
    ref_tokens = [r.lower().split() for r in references]

    # Compute n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        total_matches = 0
        total_count = 0

        for pred, ref in zip(pred_tokens, ref_tokens):
            # Get n-grams
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)

            # Count matches
            for ngram, count in pred_ngrams.items():
                total_matches += min(count, ref_ngrams.get(ngram, 0))
            total_count += sum(pred_ngrams.values())

        if total_count > 0:
            precisions.append(total_matches / total_count)
        else:
            precisions.append(0.0)

    # Brevity penalty
    pred_len = sum(len(p) for p in pred_tokens)
    ref_len = sum(len(r) for r in ref_tokens)

    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / max(pred_len, 1))

    # BLEU score
    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(np.log(p) / max_n for p in precisions)
    return float(bp * np.exp(log_avg))


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from token list."""
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] += 1
    return ngrams


################################################################################
# SECTION 4: ROUGE SCORE
################################################################################

def compute_rouge(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE Score
    ====================

    Definition: Recall-Oriented Understudy for Gisting Evaluation.
    Measures how much of the reference is captured by the prediction.

    ROUGE-N: N-gram recall
    ROUGE-L: Longest Common Subsequence

    Used for: Summarization, text generation

    Interview Question:
        "When should I use ROUGE vs BLEU?"
        ROUGE for summarization (captures recall).
        BLEU for translation (captures precision).
    """
    # ROUGE-1 (unigram recall)
    rouge1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())

        if len(ref_tokens) > 0:
            overlap = len(pred_tokens & ref_tokens)
            recall = overlap / len(ref_tokens)
            rouge1_scores.append(recall)

    # ROUGE-L (LCS-based)
    rougeL_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        lcs_len = lcs_length(pred_tokens, ref_tokens)
        if len(ref_tokens) > 0:
            recall = lcs_len / len(ref_tokens)
            rougeL_scores.append(recall)

    return {
        'rouge-1': float(np.mean(rouge1_scores)) if rouge1_scores else 0.0,
        'rouge-L': float(np.mean(rougeL_scores)) if rougeL_scores else 0.0
    }


def lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


################################################################################
# SECTION 5: F1 SCORE
################################################################################

def compute_f1(
    predictions: List[str],
    references: List[str]
) -> float:
    """
    Compute F1 Score
    ================

    Definition: Harmonic mean of precision and recall.
    Formula: F1 = 2 × (precision × recall) / (precision + recall)

    Used for: Question answering, classification
    """
    precision_scores = []
    recall_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())

        if len(pred_tokens) > 0 and len(ref_tokens) > 0:
            overlap = len(pred_tokens & ref_tokens)
            precision = overlap / len(pred_tokens)
            recall = overlap / len(ref_tokens)

            precision_scores.append(precision)
            recall_scores.append(recall)

    if not precision_scores:
        return 0.0

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)

    if avg_precision + avg_recall > 0:
        return float(2 * avg_precision * avg_recall / (avg_precision + avg_recall))
    return 0.0


################################################################################
# SECTION 6: TESTING & EXAMPLES
################################################################################

def demonstrate_metrics():
    """Demonstrate evaluation metrics."""
    print("=" * 70)
    print("EVALUATION METRICS DEMONSTRATION")
    print("=" * 70)

    # Perplexity
    print("\n--- Perplexity ---")
    batch, seq, vocab = 2, 4, 10
    logits = np.random.randn(batch, seq, vocab)
    targets = np.random.randint(0, vocab, (batch, seq))
    ppl = compute_perplexity(logits, targets)
    print(f"Perplexity: {ppl:.2f}")

    # Accuracy
    print("\n--- Accuracy ---")
    predictions = np.array([1, 2, 3, 4, 5])
    targets = np.array([1, 2, 3, 5, 5])
    acc = compute_accuracy(predictions, targets)
    print(f"Accuracy: {acc:.2%}")

    # BLEU
    print("\n--- BLEU ---")
    preds = ["the cat sat on the mat"]
    refs = ["the cat sat on the mat"]
    bleu = compute_bleu(preds, refs)
    print(f"BLEU (perfect): {bleu:.4f}")

    preds2 = ["the dog ran in the park"]
    bleu2 = compute_bleu(preds2, refs)
    print(f"BLEU (different): {bleu2:.4f}")

    # ROUGE
    print("\n--- ROUGE ---")
    rouge = compute_rouge(preds, refs)
    print(f"ROUGE-1: {rouge['rouge-1']:.4f}")
    print(f"ROUGE-L: {rouge['rouge-L']:.4f}")

    # F1
    print("\n--- F1 Score ---")
    f1 = compute_f1(preds, refs)
    print(f"F1: {f1:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_metrics()


################################################################################
# REFERENCES
################################################################################

# [1] Papineni, K., et al. (2002). BLEU: a Method for Automatic Evaluation of MT.
# [2] Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
# [3] Rajpurkar, P., et al. (2016). SQuAD: 100,000+ Questions for Machine Reading.

################################################################################
