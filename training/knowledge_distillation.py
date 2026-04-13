"""
Knowledge Distillation for LLMs
=================================

Knowledge Distillation transfers knowledge from a large "teacher" model
to a smaller "student" model. The student learns to mimic the teacher's
behavior, achieving similar quality with fewer parameters.

Key Insight:
  The teacher's output distribution contains more information than just
  the hard label. "Soft targets" reveal inter-class relationships
  (e.g., "cat" is more similar to "dog" than to "airplane").

  Student Loss = α × KL(teacher_logits/T, student_logits/T) + (1-α) × CE(student, labels)

  Where T = temperature (softens the distribution)
        α = weight for distillation loss

Distillation Types:
  1. Logit-based: Match output distributions
  2. Feature-based: Match intermediate representations
  3. Relation-based: Match relationships between samples

Applications in LLMs:
  - GPT-4 → smaller GPT-4 variants
  - Llama 70B → Llama 7B
  - Teacher-student for code generation
  - Speculative decoding (draft model is distilled)

References:
  - Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
  - Gu et al., "MiniLLM: Knowledge Distillation of Large Language Models" (2023)
  - Ko et al., "Large Language Models are Reasoning Teachers" (2024)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DistillationConfig:
    """
    Configuration for knowledge distillation.

    Attributes:
        temperature: Softens probability distribution (higher = softer)
        alpha: Weight for distillation loss (vs hard label loss)
        learning_rate: Student learning rate
        max_steps: Training steps
        warmup_steps: LR warmup steps
        batch_size: Training batch size
    """
    temperature: float = 2.0
    alpha: float = 0.7
    learning_rate: float = 1e-4
    max_steps: int = 10000
    warmup_steps: int = 500
    batch_size: int = 32


# ============================================================================
# DISTILLATION LOSS
# ============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL(p || q) = Σ p(x) log(p(x) / q(x))

    Measures how much q differs from p.
    Lower = more similar.
    """
    # Add epsilon for numerical stability
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Cross-entropy loss.

    Args:
        logits: Model output [batch, vocab_size]
        targets: Target indices [batch]
    """
    log_probs = np.log(softmax(logits) + 1e-10)
    batch_size = logits.shape[0]
    return -np.mean(log_probs[np.arange(batch_size), targets])


def distillation_loss(teacher_logits: np.ndarray,
                      student_logits: np.ndarray,
                      targets: np.ndarray,
                      temperature: float = 2.0,
                      alpha: float = 0.7) -> Tuple[float, Dict]:
    """
    Compute knowledge distillation loss.

    The loss combines:
    1. Soft loss: KL divergence between teacher and student distributions
    2. Hard loss: Cross-entropy between student and true labels

    L = α × T² × KL(softmax(t/T) || softmax(s/T)) + (1-α) × CE(s, y)

    Why T²? The gradients of KL loss scale as 1/T², so we multiply by T²
    to keep the balance between soft and hard losses stable.

    Args:
        teacher_logits: Teacher output logits [batch, vocab_size]
        student_logits: Student output logits [batch, vocab_size]
        targets: Ground truth labels [batch]
        temperature: Softening temperature
        alpha: Weight for soft loss

    Returns:
        total_loss: Combined distillation loss
        metrics: Dictionary of loss components
    """
    # ── Soft targets (teacher distribution) ───────────────────
    teacher_probs = softmax(teacher_logits / temperature)
    student_probs = softmax(student_logits / temperature)

    # ── Soft loss: KL divergence ──────────────────────────────
    # KL(teacher || student) with temperature scaling
    soft_loss = 0.0
    for i in range(len(teacher_probs)):
        soft_loss += kl_divergence(teacher_probs[i], student_probs[i])
    soft_loss = soft_loss / len(teacher_probs)
    soft_loss = soft_loss * (temperature ** 2)  # Scale by T²

    # ── Hard loss: Cross-entropy ──────────────────────────────
    hard_loss = cross_entropy(student_logits, targets)

    # ── Combined loss ─────────────────────────────────────────
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss

    metrics = {
        "total_loss": float(total_loss),
        "soft_loss": float(soft_loss),
        "hard_loss": float(hard_loss),
        "teacher_entropy": float(-np.sum(teacher_probs[0] * np.log(teacher_probs[0] + 1e-10))),
        "student_entropy": float(-np.sum(student_probs[0] * np.log(student_probs[0] + 1e-10))),
    }

    return total_loss, metrics


# ============================================================================
# TEACHER AND STUDENT MODELS
# ============================================================================

class SimpleModel:
    """
    Simple model for demonstration.

    In practice, these would be full Transformer models.
    """

    def __init__(self, d_model: int, vocab_size: int, name: str = "model"):
        self.name = name
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Simple 2-layer FFN
        self.w1 = np.random.randn(d_model, d_model * 4) * 0.01
        self.w2 = np.random.randn(d_model * 4, vocab_size) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. Returns logits."""
        hidden = np.maximum(0, x @ self.w1)  # ReLU
        logits = hidden @ self.w2
        return logits

    def count_params(self) -> int:
        """Count trainable parameters."""
        return self.d_model * self.d_model * 4 + self.d_model * 4 * self.vocab_size


# ============================================================================
# DISTILLATION TRAINER
# ============================================================================

class DistillationTrainer:
    """
    Knowledge Distillation Training Loop.

    Training procedure:
    1. Teacher generates soft targets (frozen)
    2. Student forward pass
    3. Compute distillation loss
    4. Update student weights
    5. Repeat
    """

    def __init__(self, teacher: SimpleModel, student: SimpleModel,
                 config: DistillationConfig):
        """
        Initialize distillation trainer.

        Args:
            teacher: Large teacher model (frozen)
            student: Small student model (trainable)
            config: Distillation configuration
        """
        self.teacher = teacher
        self.student = student
        self.config = config
        self.step_count = 0
        self.history = []

    def train_step(self, x: np.ndarray, targets: np.ndarray) -> Dict:
        """
        Execute one distillation training step.

        Args:
            x: Input features [batch, d_model]
            targets: Ground truth labels [batch]

        Returns:
            Dictionary of metrics
        """
        # ── Teacher forward (no gradients) ─────────────────────
        teacher_logits = self.teacher.forward(x)

        # ── Student forward ────────────────────────────────────
        student_logits = self.student.forward(x)

        # ── Compute distillation loss ──────────────────────────
        loss, metrics = distillation_loss(
            teacher_logits, student_logits, targets,
            temperature=self.config.temperature,
            alpha=self.config.alpha
        )

        self.step_count += 1
        self.history.append(metrics)

        return metrics

    def get_summary(self) -> Dict:
        """Get training summary."""
        if not self.history:
            return {"status": "not_started"}

        return {
            "steps": self.step_count,
            "avg_loss": np.mean([m["total_loss"] for m in self.history[-100:]]),
            "avg_soft_loss": np.mean([m["soft_loss"] for m in self.history[-100:]]),
            "avg_hard_loss": np.mean([m["hard_loss"] for m in self.history[-100:]]),
            "teacher_params": self.teacher.count_params(),
            "student_params": self.student.count_params(),
            "compression_ratio": self.teacher.count_params() / self.student.count_params(),
        }


# ============================================================================
# FEATURE-BASED DISTILLATION
# ============================================================================

class FeatureDistillation:
    """
    Feature-based distillation: match intermediate representations.

    Instead of just matching outputs, we also match hidden states:
    L_feature = MSE(teacher_hidden, student_hidden_proj)

    This gives the student richer supervision signals.
    """

    def __init__(self, teacher_dim: int, student_dim: int):
        """
        Initialize feature distillation.

        Args:
            teacher_dim: Teacher hidden dimension
            student_dim: Student hidden dimension
        """
        # Projection to match dimensions
        self.projection = np.random.randn(student_dim, teacher_dim) * 0.01

    def feature_loss(self, teacher_hidden: np.ndarray,
                     student_hidden: np.ndarray) -> float:
        """
        Compute feature matching loss.

        Args:
            teacher_hidden: Teacher intermediate features [batch, teacher_dim]
            student_hidden: Student intermediate features [batch, student_dim]

        Returns:
            MSE loss between projected features
        """
        # Project student features to teacher dimension
        student_projected = student_hidden @ self.projection

        # MSE loss
        return np.mean((teacher_hidden - student_projected) ** 2)


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_distillation():
    """
    Demonstrate knowledge distillation.

    Shows:
        1. Teacher-student setup
        2. Distillation loss computation
        3. Training step
        4. Compression statistics
    """
    print("=" * 70)
    print("Knowledge Distillation — Demonstration")
    print("=" * 70)

    # Create teacher and student
    d_model = 64
    vocab_size = 1000

    teacher = SimpleModel(d_model, vocab_size, "Teacher-Large")
    student = SimpleModel(d_model // 2, vocab_size, "Student-Small")

    print(f"\nModel Comparison:")
    print(f"  Teacher: {teacher.count_params():,} parameters")
    print(f"  Student: {student.count_params():,} parameters")
    print(f"  Compression: {teacher.count_params() / student.count_params():.1f}x")

    # Configuration
    config = DistillationConfig(
        temperature=2.0,
        alpha=0.7,
        learning_rate=1e-4,
    )

    print(f"\nDistillation Config:")
    print(f"  Temperature: {config.temperature}")
    print(f"  Alpha (soft loss weight): {config.alpha}")
    print(f"  Hard loss weight: {1 - config.alpha}")

    # Create trainer
    trainer = DistillationTrainer(teacher, student, config)

    # Training step
    print("\n[Training Step]")
    batch_size = 16
    x = np.random.randn(batch_size, d_model)
    targets = np.random.randint(0, vocab_size, batch_size)

    metrics = trainer.train_step(x, targets)

    print(f"  Total loss: {metrics['total_loss']:.4f}")
    print(f"  Soft loss (KL): {metrics['soft_loss']:.4f}")
    print(f"  Hard loss (CE): {metrics['hard_loss']:.4f}")
    print(f"  Teacher entropy: {metrics['teacher_entropy']:.2f}")
    print(f"  Student entropy: {metrics['student_entropy']:.2f}")

    # Temperature effect
    print("\n[Temperature Effect]")
    teacher_logits = teacher.forward(x[:1])
    for temp in [0.5, 1.0, 2.0, 5.0, 10.0]:
        probs = softmax(teacher_logits / temp)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        print(f"  T={temp:.1f}: entropy={entropy:.2f}, max_prob={np.max(probs):.3f}")

    # Distillation variants
    print("\n[Distillation Variants]")
    print("  1. Logit-based: Match output distributions (standard)")
    print("  2. Feature-based: Match intermediate representations")
    print("  3. Relation-based: Match sample relationships")
    print("  4. Self-distillation: Teacher = larger version of student")
    print("  5. Online distillation: Teacher and student train together")

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. Soft targets contain more info than hard labels")
    print("  2. Temperature controls knowledge transfer intensity")
    print("  3. Alpha balances soft vs hard loss")
    print("  4. 2-10x compression with minimal quality loss")
    print("  5. Critical for edge deployment and cost reduction")
    print("=" * 70)


if __name__ == "__main__":
    demo_distillation()
