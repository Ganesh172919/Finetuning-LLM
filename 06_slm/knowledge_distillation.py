"""
################################################################################
KNOWLEDGE DISTILLATION — LEARNING FROM LARGER MODELS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Knowledge Distillation?
    Training a small model (student) to mimic a larger model (teacher).

Why Distill?
    - Large models are expensive to run
    - Small models are faster but less capable
    - Distillation transfers knowledge from large to small

How it Works:
    1. Teacher model generates soft labels (probabilities)
    2. Student model learns from soft labels
    3. Student also learns from hard labels (ground truth)

Loss = α × L_hard + (1-α) × KL(student, teacher)

Interview Questions:
        Q: "What is knowledge distillation?"
        A: Training a small model to mimic a larger model.
           The soft labels contain more information than hard labels.

################################################################################
"""

import numpy as np
from typing import Tuple

################################################################################
# SECTION 1: DISTILLATION TRAINER
################################################################################

class DistillationTrainer:
    """
    Knowledge Distillation Trainer
    ===============================

    Trains student model using teacher's soft predictions.

    Interview Questions:
        Q: "Why use soft labels instead of hard labels?"
        A: Soft labels contain more information. They show the model's
           uncertainty and relationships between classes.
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
            student_logits: Student output [batch × vocab]
            teacher_logits: Teacher output [batch × vocab]
            hard_labels: True labels [batch]

        Returns:
            loss: Combined loss
        """
        # Hard label loss
        hard_loss = self._cross_entropy(student_logits, hard_labels)

        # Soft label loss
        student_probs = self._softmax(student_logits / self.temperature)
        teacher_probs = self._softmax(teacher_logits / self.temperature)
        soft_loss = -np.sum(teacher_probs * np.log(student_probs + 1e-8)) / len(student_logits)

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
# SECTION 2: TESTING
################################################################################

def demonstrate_distillation():
    """Demonstrate knowledge distillation."""
    print("=" * 70)
    print("KNOWLEDGE DISTILLATION DEMONSTRATION")
    print("=" * 70)

    trainer = DistillationTrainer(alpha=0.5, temperature=2.0)

    # Simulate teacher and student
    batch_size = 4
    vocab_size = 100

    teacher_logits = np.random.randn(batch_size, vocab_size)
    student_logits = np.random.randn(batch_size, vocab_size)
    hard_labels = np.random.randint(0, vocab_size, batch_size)

    loss = trainer.compute_loss(student_logits, teacher_logits, hard_labels)
    print(f"Distillation loss: {loss:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_distillation()


################################################################################
# REFERENCES
################################################################################

# [1] Hinton, G., et al. (2015). Distilling the Knowledge in a Neural Network.

################################################################################
