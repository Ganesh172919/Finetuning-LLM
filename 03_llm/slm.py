"""
################################################################################
SMALL LANGUAGE MODELS (SLMs) — EFFICIENT AI
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Small Language Models?
    SLMs are language models with fewer parameters (1B-7B) designed
    for efficiency and deployment on edge devices.

Why do they matter?
    Large models (70B+) are expensive:
    - Need multiple GPUs
    - High latency
    - High cost

    SLMs can:
    - Run on a single GPU
    - Run on mobile devices
    - Provide fast responses
    - Be cheaper to serve

Historical Evolution:
    - 2023: Phi-1.5 (Microsoft), Gemma (Google)
    - 2024: Phi-3, Gemma 2, Qwen2
    - 2025: Smarter SLMs with better data

Key Techniques:
    1. Better data: Quality over quantity
    2. Knowledge distillation: Learn from larger models
    3. Architecture efficiency: Better use of parameters
    4. Quantization: Lower precision

Interview Questions:
    1. "What's the difference between SLMs and LLMs?"
       SLMs: 1B-7B parameters, efficient, edge deployment
       LLMs: 70B+ parameters, powerful, cloud deployment

    2. "How do SLMs achieve good performance?"
       Better data quality, knowledge distillation,
       efficient architecture design.

    3. "When should I use SLM vs LLM?"
       SLM: latency-sensitive, cost-sensitive, edge devices
       LLM: quality-critical, complex reasoning, cloud

################################################################################
"""

import numpy as np
from typing import Optional
import math

import sys
sys.path.append('..')
from ..02_transformers.model import TransformerLM
from ..02_transformers.layers import TransformerBlock, RMSNorm

################################################################################
# SECTION 1: EFFICIENT TRANSFORMER
################################################################################

class SmallLanguageModel(TransformerLM):
    """
    Small Language Model
    ====================

    Optimized for efficiency:
    - Smaller hidden dimensions
    - Fewer layers
    - Efficient attention (GQA)
    - Better initialization

    Model Sizes:
    - Phi-1.5: 1.3B params
    - Gemma-2B: 2B params
    - Qwen2-1.5B: 1.5B params
    """

    @classmethod
    def phi_1_5(cls):
        """Microsoft Phi-1.5: 1.3B parameters"""
        return cls(
            vocab_size=51200,
            d_model=2048,
            n_layers=24,
            n_heads=32,
            n_kv_heads=32,
            max_seq_len=2048
        )

    @classmethod
    def gemma_2b(cls):
        """Google Gemma: 2B parameters"""
        return cls(
            vocab_size=256000,
            d_model=2048,
            n_layers=18,
            n_heads=8,
            n_kv_heads=4,  # GQA
            max_seq_len=8192
        )

    @classmethod
    def qwen2_1_5b(cls):
        """Alibaba Qwen2: 1.5B parameters"""
        return cls(
            vocab_size=151936,
            d_model=1536,
            n_layers=28,
            n_heads=12,
            n_kv_heads=2,  # GQA
            max_seq_len=32768
        )


################################################################################
# SECTION 2: KNOWLEDGE DISTILLATION
################################################################################

class KnowledgeDistillation:
    """
    Knowledge Distillation
    ======================

    Definition: Train a small model to mimic a larger model.

    Process:
    1. Large model (teacher) generates soft labels
    2. Small model (student) learns from soft labels
    3. Student learns the teacher's "knowledge"

    Loss:
    L = α × L_hard + (1-α) × KL(student_logits, teacher_logits)

    Benefits:
    - Student learns faster
    - Student often better than training from scratch
    - Can transfer complex reasoning patterns

    Interview Question:
        "What is knowledge distillation?"
        Training a small model to mimic a larger model's behavior.
        The small model learns from the larger model's soft predictions,
        which contain more information than hard labels.
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 2.0):
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self,
        student_logits: np.ndarray,
        teacher_logits: np.ndarray,
        hard_labels: np.ndarray
    ) -> float:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model output [batch × vocab]
            teacher_logits: Teacher model output [batch × vocab]
            hard_labels: True labels [batch]

        Returns:
            Combined loss
        """
        # Hard label loss (standard cross-entropy)
        hard_loss = self._cross_entropy(student_logits, hard_labels)

        # Soft label loss (KL divergence)
        student_probs = self._softmax(student_logits / self.temperature)
        teacher_probs = self._softmax(teacher_logits / self.temperature)

        soft_loss = -np.sum(teacher_probs * np.log(student_probs + 1e-8)) / len(student_logits)

        # Combined loss
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

    @staticmethod
    def _softmax(x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def _cross_entropy(logits, targets):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
        log_probs = shifted[np.arange(len(targets)), targets] - log_sum_exp
        return -np.mean(log_probs)


################################################################################
# SECTION 3: TESTING & EXAMPLES
################################################################################

def demonstrate_slm():
    """Demonstrate SLM concepts."""
    print("=" * 70)
    print("SMALL LANGUAGE MODEL DEMONSTRATION")
    print("=" * 70)

    # Create SLM
    print("\n--- Creating SLM ---")
    slm = SmallLanguageModel(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=256
    )

    # Forward pass
    print("\n--- Forward Pass ---")
    token_ids = np.random.randint(0, 1000, (1, 8))
    targets = np.random.randint(0, 1000, (1, 8))
    logits, loss = slm.forward(token_ids, targets)
    print(f"Input: {token_ids.shape}")
    print(f"Output: {logits.shape}")
    print(f"Loss: {loss:.4f}")

    # Knowledge distillation
    print("\n--- Knowledge Distillation ---")
    kd = KnowledgeDistillation(alpha=0.5, temperature=2.0)
    student_logits = np.random.randn(2, 100)
    teacher_logits = np.random.randn(2, 100)
    labels = np.array([1, 2])
    kd_loss = kd.compute_loss(student_logits, teacher_logits, labels)
    print(f"Distillation loss: {kd_loss:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_slm()


################################################################################
# REFERENCES
################################################################################

# [1] Li, Y., et al. (2023). Textbooks Are All You Need (Phi).
# [2] Google. (2024). Gemma: Open Models Based on Gemini.
# [3] Bai, J., et al. (2023). Qwen Technical Report.

################################################################################
