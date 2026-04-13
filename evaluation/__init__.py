"""
################################################################################
EVALUATION METRICS — MEASURING MODEL QUALITY
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Evaluation Metrics?
    Metrics measure how well a model performs. For language models,
    the key metrics are:
    - Perplexity: How "surprised" the model is by text
    - Accuracy: How often predictions are correct
    - BLEU: Quality of generated text (for translation)
    - ROUGE: Quality of summaries
    - F1: Balance of precision and recall

Why do they matter?
    Without metrics, we can't:
    - Compare different models
    - Know if training is working
    - Select hyperparameters
    - Publish research results

Interview Questions:
    1. "What is perplexity?"
       Perplexity measures how well a model predicts text.
       Lower perplexity = better model.
       PPL = 2^(cross-entropy loss)

    2. "What's the difference between BLEU and ROUGE?"
       BLEU: precision-focused (how much generated text is correct)
       ROUGE: recall-focused (how much reference text is captured)
       BLEU for translation, ROUGE for summarization.

    3. "How do you evaluate an LLM?"
       Multiple metrics: perplexity, benchmarks (MMLU, HumanEval),
       human evaluation, task-specific metrics.

################################################################################
"""

from .metrics import (
    compute_perplexity,
    compute_accuracy,
    compute_bleu,
    compute_rouge,
    compute_f1
)
