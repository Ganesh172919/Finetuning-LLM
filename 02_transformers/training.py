"""
################################################################################
TRANSFORMER TRAINING PIPELINE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Transformer Training?
    The process of training a transformer language model on text data.
    This involves:
    1. Data preparation (tokenization, batching)
    2. Forward pass (compute predictions)
    3. Loss computation (cross-entropy)
    4. Backward pass (compute gradients)
    5. Optimizer step (update weights)
    6. Repeat

Why does it matter?
    Training is where models learn. The quality of training determines:
    - Model capability
    - Training cost
    - Time to convergence

Interview Questions:
    1. "How do you train an LLM?"
       Tokenize text, split into batches, compute loss (next token prediction),
       backpropagate, update weights. Repeat for billions of tokens.

    2. "What's the most important hyperparameter?"
       Learning rate. Use warmup + cosine decay schedule.

    3. "How long does training take?"
       LLaMA-7B: ~21 days on 2048 A100 GPUs
       GPT-4: estimated 3-4 months on 25000 A100 GPUs

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, Dict
import math

################################################################################
# SECTION 1: TRAINING CONFIGURATION
################################################################################

class TrainingConfig:
    """
    Training Configuration
    ======================

    All hyperparameters for training.
    """

    def __init__(
        self,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        grad_clip: float = 1.0,
        accumulation_steps: int = 1,
        fp16: bool = True,
        checkpoint_every: int = 1000
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.accumulation_steps = accumulation_steps
        self.fp16 = fp16
        self.checkpoint_every = checkpoint_every


################################################################################
# SECTION 2: TRAINING LOOP
################################################################################

class TransformerTrainer:
    """
    Transformer Trainer
    ===================

    Complete training loop for transformer models.

    Key features:
    - Gradient accumulation
    - Mixed precision
    - Gradient clipping
    - Learning rate scheduling
    - Checkpointing
    - Logging

    Interview Question:
        "Walk me through the training loop."
        1. Get batch of data
        2. Forward pass → logits
        3. Compute loss (cross-entropy)
        4. Backward pass → gradients
        5. Clip gradients
        6. Optimizer step → update weights
        7. Log metrics
        8. Repeat
    """

    def __init__(self, model, config: TrainingConfig):
        self.model = model
        self.config = config
        self.step = 0

        # Optimizer (simplified)
        self.learning_rate = config.learning_rate

    def train_step(self, token_ids: np.ndarray, targets: np.ndarray) -> Dict:
        """
        Single training step.

        Args:
            token_ids: Input tokens [batch × seq_len]
            targets: Target tokens [batch × seq_len]

        Returns:
            Dictionary with metrics
        """
        # Forward pass
        logits, loss = self.model.forward(token_ids, targets)

        # In real implementation:
        # 1. loss.backward() — compute gradients
        # 2. clip_grad_norm_(model.parameters(), self.config.grad_clip)
        # 3. optimizer.step() — update weights
        # 4. optimizer.zero_grad() — reset gradients

        self.step += 1

        # Compute metrics
        metrics = {
            'loss': float(loss),
            'learning_rate': self.get_lr(),
            'step': self.step
        }

        return metrics

    def get_lr(self) -> float:
        """Get current learning rate with warmup + cosine schedule."""
        if self.step < self.config.warmup_steps:
            return self.config.learning_rate * (self.step / self.config.warmup_steps)

        progress = (self.step - self.config.warmup_steps) / (
            self.config.max_steps - self.config.warmup_steps
        )
        return self.config.learning_rate * (
            0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
        )


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_training():
    """Demonstrate training pipeline."""
    print("=" * 70)
    print("TRAINING PIPELINE DEMONSTRATION")
    print("=" * 70)

    # Create config
    print("\n--- Training Config ---")
    config = TrainingConfig(
        batch_size=16,
        learning_rate=3e-4,
        warmup_steps=100,
        max_steps=1000
    )
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Warmup steps: {config.warmup_steps}")

    # Learning rate schedule
    print("\n--- Learning Rate Schedule ---")
    for step in [0, 50, 100, 500, 999]:
        if step < config.warmup_steps:
            lr = config.learning_rate * (step / config.warmup_steps)
        else:
            progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
            lr = config.learning_rate * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))
        print(f"Step {step}: lr={lr:.6f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_training()
