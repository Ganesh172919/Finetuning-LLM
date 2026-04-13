"""
################################################################################
GPT TRAINING PIPELINE — FROM DATA TO MODEL
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is GPT Training?
    Training a Generative Pre-trained Transformer on text data.
    The model learns to predict the next token given previous tokens.

Training Recipe (LLaMA-style):
    Data: 1-2 trillion tokens
    Batch size: 4M tokens
    Learning rate: 3e-4 → 1e-5 (cosine decay)
    Warmup: 2000 steps
    Weight decay: 0.1
    Optimizer: AdamW
    Precision: bf16

Training Loop:
    1. Load batch of text
    2. Tokenize
    3. Forward pass → logits
    4. Compute loss (cross-entropy)
    5. Backward pass → gradients
    6. Clip gradients
    7. Optimizer step
    8. Repeat

Interview Questions:
        Q: "How long does it take to train an LLM?"
        A: LLaMA-7B: ~21 days on 2048 A100 GPUs
           GPT-4: estimated 3-4 months on 25000 A100 GPUs
           Cost: millions to hundreds of millions of dollars

        Q: "What's the most important hyperparameter?"
        A: Learning rate. Too high: diverges. Too low: slow.
           Use warmup + cosine decay schedule.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, Dict
import math

################################################################################
# SECTION 1: GPT TRAINING CONFIG
################################################################################

class GPTTrainingConfig:
    """
    GPT Training Configuration
    ==========================

    All hyperparameters for GPT training.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        max_seq_len: int = 4096,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        min_lr: float = 3e-5,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        accumulation_steps: int = 1,
        fp16: bool = True
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.accumulation_steps = accumulation_steps
        self.fp16 = fp16


################################################################################
# SECTION 2: DATA PIPELINE
################################################################################

class TextDataPipeline:
    """
    Text Data Pipeline
    ==================

    Handles data loading and preprocessing for GPT training.

    Steps:
    1. Load text data
    2. Tokenize
    3. Create batches
    4. Shuffle and iterate
    """

    def __init__(self, vocab_size: int, max_seq_len: int, batch_size: int):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def create_batch(self, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training batch.

        Args:
            token_ids: Full token sequence

        Returns:
            input_ids: [batch × seq_len]
            target_ids: [batch × seq_len] (shifted by 1)
        """
        # Sample random positions
        positions = np.random.randint(0, len(token_ids) - self.max_seq_len, self.batch_size)

        input_ids = np.zeros((self.batch_size, self.max_seq_len), dtype=np.int64)
        target_ids = np.zeros((self.batch_size, self.max_seq_len), dtype=np.int64)

        for i, pos in enumerate(positions):
            input_ids[i] = token_ids[pos:pos + self.max_seq_len]
            target_ids[i] = token_ids[pos + 1:pos + self.max_seq_len + 1]

        return input_ids, target_ids


################################################################################
# SECTION 3: TRAINING LOOP
################################################################################

class GPTTrainer:
    """
    GPT Trainer
    ===========

    Complete training loop for GPT models.

    Key features:
    - Gradient accumulation
    - Mixed precision
    - Gradient clipping
    - Learning rate scheduling
    - Checkpointing
    - Logging

    Interview Questions:
        Q: "Walk me through the training loop."
        A: 1) Get batch of data
           2) Forward pass → logits
           3) Compute loss (cross-entropy)
           4) Backward pass → gradients
           5) Clip gradients
           6) Optimizer step → update weights
           7) Log metrics
           8) Repeat
    """

    def __init__(self, config: GPTTrainingConfig):
        self.config = config
        self.step = 0

        # Learning rate schedule
        self.lr_schedule = self._create_lr_schedule()

    def _create_lr_schedule(self):
        """Create learning rate schedule."""
        def get_lr(step):
            if step < self.config.warmup_steps:
                return self.config.learning_rate * (step / self.config.warmup_steps)
            else:
                progress = (step - self.config.warmup_steps) / (
                    self.config.max_steps - self.config.warmup_steps
                )
                return self.config.min_lr + 0.5 * (
                    self.config.learning_rate - self.config.min_lr
                ) * (1 + math.cos(math.pi * progress))
        return get_lr

    def train_step(
        self,
        input_ids: np.ndarray,
        target_ids: np.ndarray
    ) -> Dict:
        """
        Single training step.

        Args:
            input_ids: [batch × seq_len]
            target_ids: [batch × seq_len]

        Returns:
            metrics: Training metrics
        """
        # Get current learning rate
        lr = self.lr_schedule(self.step)

        # Forward pass (simplified)
        # In real implementation:
        # logits = model.forward(input_ids)
        # loss = cross_entropy(logits, target_ids)

        loss = 2.5 * math.exp(-self.step / 10000)  # Simulated loss

        # Backward pass (simplified)
        # gradients = loss.backward()
        # gradients = clip_gradients(gradients, self.config.grad_clip)
        # optimizer.step(gradients)

        self.step += 1

        return {
            'loss': loss,
            'learning_rate': lr,
            'step': self.step,
            'tokens_per_step': self.config.batch_size * self.config.max_seq_len,
        }

    def train(
        self,
        data_pipeline: TextDataPipeline,
        token_ids: np.ndarray,
        n_steps: int = 1000
    ):
        """
        Main training loop.

        Args:
            data_pipeline: Data loading pipeline
            token_ids: Training data
            n_steps: Number of training steps
        """
        for step in range(n_steps):
            # Get batch
            input_ids, target_ids = data_pipeline.create_batch(token_ids)

            # Train step
            metrics = self.train_step(input_ids, target_ids)

            # Log
            if step % 100 == 0:
                print(f"Step {metrics['step']}: loss={metrics['loss']:.4f}, lr={metrics['learning_rate']:.6f}")


################################################################################
# SECTION 4: TESTING
################################################################################

def demonstrate_gpt_training():
    """Demonstrate GPT training."""
    print("=" * 70)
    print("GPT TRAINING DEMONSTRATION")
    print("=" * 70)

    # Config
    print("\n--- Training Config ---")
    config = GPTTrainingConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        max_seq_len=64,
        batch_size=4,
        learning_rate=3e-4,
        warmup_steps=100,
        max_steps=1000
    )
    print(f"Model: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"Training: lr={config.learning_rate}, warmup={config.warmup_steps}")

    # Data pipeline
    print("\n--- Data Pipeline ---")
    pipeline = TextDataPipeline(config.vocab_size, config.max_seq_len, config.batch_size)
    token_ids = np.random.randint(0, config.vocab_size, 10000)
    input_ids, target_ids = pipeline.create_batch(token_ids)
    print(f"Input: {input_ids.shape}")
    print(f"Target: {target_ids.shape}")

    # Training
    print("\n--- Training ---")
    trainer = GPTTrainer(config)
    trainer.train(pipeline, token_ids, n_steps=500)

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_gpt_training()


################################################################################
# REFERENCES
################################################################################

# [1] Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners.
# [2] Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models.

################################################################################
