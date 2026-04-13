"""
################################################################################
DISTRIBUTED TRAINING — SCALING TO MULTIPLE GPUs
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Distributed Training?
    Training a model across multiple GPUs or machines simultaneously.
    This is essential for large models that don't fit on a single GPU.

Types of Parallelism:
    1. Data Parallel (DP): Each GPU has full model, different data
    2. Model Parallel: Split model across GPUs
    3. Pipeline Parallel: Split model into stages
    4. Tensor Parallel: Split individual operations
    5. Expert Parallel: Split MoE experts

Why it matters:
    LLaMA-70B: 140GB in fp16, needs 2+ A100 GPUs (80GB each)
    GPT-4: estimated 3.6TB, needs 50+ GPUs

    Without distributed training, we couldn't train modern LLMs.

Interview Questions:
    1. "What's the difference between data and model parallelism?"
       Data parallel: replicate model, split data
       Model parallel: split model, replicate data
       Most large training uses both (3D parallelism).

    2. "What is ZeRO?"
       ZeRO (Zero Redundancy Optimizer) reduces memory in data parallel:
       - Stage 1: Shard optimizer states
       - Stage 2: Shard gradients
       - Stage 3: Shard parameters
       This allows fitting larger models on same hardware.

    3. "How do you handle GPU failures during training?"
       Regular checkpointing (every N steps).
       Automatic restart from last checkpoint.
       Elastic training that adjusts to available GPUs.

################################################################################
"""

import numpy as np
from typing import Optional, List
import math

################################################################################
# SECTION 1: DATA PARALLEL TRAINING
################################################################################

class DataParallel:
    """
    Data Parallel Training
    ======================

    Definition: Replicate the model on each GPU, but split the data.

    Algorithm:
    1. Copy model to all GPUs
    2. Split batch across GPUs
    3. Each GPU computes gradients on its portion
    4. All-reduce: average gradients across GPUs
    5. Each GPU updates its model with averaged gradients

    Visual:
    GPU 0: Model → Batch 0 → Grad 0 ─┐
    GPU 1: Model → Batch 1 → Grad 1 ─┼→ Avg Grad → Update
    GPU 2: Model → Batch 2 → Grad 2 ─┤
    GPU 3: Model → Batch 3 → Grad 3 ─┘

    Memory: Each GPU stores full model (redundant!)
    Communication: All-reduce gradients after each step

    Interview Question:
        "What's the limitation of data parallelism?"
        Each GPU must store the full model.
        For very large models, this is impossible.
        Solution: model parallelism or ZeRO.
    """

    def __init__(self, world_size: int = 1, rank: int = 0):
        """
        Initialize data parallel trainer.

        Args:
            world_size: Total number of GPUs
            rank: This GPU's index
        """
        self.world_size = world_size
        self.rank = rank

    def split_batch(
        self,
        data: np.ndarray,
        batch_size: int
    ) -> np.ndarray:
        """
        Split batch across GPUs.

        Args:
            data: Full batch [batch_size × ...]
            batch_size: Batch size per GPU

        Returns:
            This GPU's portion of the data
        """
        per_gpu = batch_size // self.world_size
        start = self.rank * per_gpu
        end = start + per_gpu
        return data[start:end]

    def all_reduce(self, gradients: List[np.ndarray]) -> np.ndarray:
        """
        Average gradients across all GPUs.

        In real distributed training, this uses NCCL (NVIDIA Collective
        Communication Library) for efficient GPU-to-GPU communication.

        Args:
            gradients: List of gradients from each GPU

        Returns:
            Averaged gradient
        """
        # In real implementation:
        # dist.all_reduce(gradients, op=dist.ReduceOp.AVG)
        return np.mean(gradients, axis=0)


################################################################################
# SECTION 2: MIXED PRECISION TRAINING
################################################################################

class MixedPrecisionTrainer:
    """
    Mixed Precision Training
    =========================

    Definition: Use both fp16 and fp32 during training.

    Why?
    - fp16: 2x faster, 2x less memory
    - But: can overflow/underflow, less precise
    - Solution: use fp16 for most ops, fp32 for critical ones

    Algorithm:
    1. Keep master weights in fp32
    2. Cast to fp16 for forward pass
    3. Compute loss in fp32
    4. Scale loss to prevent fp16 underflow
    5. Backward pass in fp16
    6. Update fp32 master weights

    Loss Scaling:
    - fp16 smallest: ~6e-8
    - Gradients can be smaller → underflow to zero
    - Solution: multiply loss by large number (e.g., 1024)
    - Divide gradients by same number after backward

    Benefits:
    - 2x memory reduction → larger models/batches
    - 2-3x speedup on modern GPUs (Tensor Cores)
    - Same model quality as fp32

    Interview Questions:
        1. "What is mixed precision training?"
           Using fp16 for most operations but fp32 for critical ones.
           It's faster and uses less memory than pure fp32.

        2. "Why do we need loss scaling?"
           Gradients in fp16 can underflow (become zero).
           Scaling loss up prevents this.

        3. "When should I use fp32 instead of fp16?"
           For loss computation, softmax, and layer normalization.
           These operations need higher precision.
    """

    def __init__(self, loss_scale: float = 1024.0):
        self.loss_scale = loss_scale

    def scale_loss(self, loss: float) -> float:
        """Scale loss to prevent fp16 underflow."""
        return loss * self.loss_scale

    def unscale_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Unscale gradients after backward pass."""
        return [grad / self.loss_scale for grad in gradients]

    def clip_gradients(
        self,
        gradients: List[np.ndarray],
        max_norm: float = 1.0
    ) -> List[np.ndarray]:
        """
        Clip gradients to prevent exploding gradients.

        Algorithm:
        1. Compute total gradient norm
        2. If norm > max_norm, scale gradients down
        """
        total_norm = 0.0
        for grad in gradients:
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)

        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            gradients = [grad * clip_coef for grad in gradients]

        return gradients


################################################################################
# SECTION 3: GRADIENT ACCUMULATION
################################################################################

class GradientAccumulator:
    """
    Gradient Accumulation
    ======================

    Definition: Accumulate gradients over multiple mini-batches
    before updating weights. Simulates larger batch sizes.

    Why?
    - Large batch sizes need lots of GPU memory
    - Gradient accumulation: accumulate N mini-batches
    - Effective batch size = mini_batch × accumulation_steps

    Algorithm:
    for i in range(accumulation_steps):
        loss = model(batch_i) / accumulation_steps
        loss.backward()  # Gradients accumulate
    optimizer.step()
    optimizer.zero_grad()

    Example:
    - Desired batch size: 1024
    - GPU can fit: 128
    - Accumulation steps: 8
    - Effective batch: 128 × 8 = 1024

    Interview Question:
        "What is gradient accumulation?"
        Accumulating gradients over multiple mini-batches before updating.
        It simulates larger batch sizes without extra memory.
    """

    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.accumulated_gradients = []
        self.step_count = 0

    def accumulate(self, gradients: List[np.ndarray]):
        """
        Accumulate gradients from one mini-batch.

        Args:
            gradients: Gradients from current mini-batch
        """
        if not self.accumulated_gradients:
            self.accumulated_gradients = [np.zeros_like(g) for g in gradients]

        for i, grad in enumerate(gradients):
            self.accumulated_gradients[i] += grad / self.accumulation_steps

        self.step_count += 1

    def should_update(self) -> bool:
        """Check if we should update weights."""
        return self.step_count >= self.accumulation_steps

    def get_gradients(self) -> List[np.ndarray]:
        """Get accumulated gradients and reset."""
        gradients = self.accumulated_gradients
        self.accumulated_gradients = []
        self.step_count = 0
        return gradients


################################################################################
# SECTION 4: CHECKPOINTING
################################################################################

class CheckpointManager:
    """
    Checkpoint Manager
    ===================

    Definition: Save and load model state during training.

    Why it matters:
    - GPU failures are common in large-scale training
    - Checkpointing allows recovery without starting over
    - Can also use for model selection, evaluation, etc.

    What to save:
    - Model weights
    - Optimizer state (for Adam: m, v)
    - Training step
    - Learning rate
    - Random state (for reproducibility)

    Interview Question:
        "How often should you checkpoint?"
        Every 1000-5000 steps, depending on:
        - Training time per step
        - Storage capacity
        - Failure frequency
        More frequent = less wasted work, but more storage.
    """

    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = save_dir
        self.checkpoints = []

    def save(
        self,
        model_state: dict,
        optimizer_state: dict,
        step: int,
        loss: float
    ):
        """
        Save checkpoint.

        Args:
            model_state: Model weights
            optimizer_state: Optimizer state
            step: Current training step
            loss: Current loss
        """
        checkpoint = {
            'model': model_state,
            'optimizer': optimizer_state,
            'step': step,
            'loss': loss
        }
        self.checkpoints.append(checkpoint)

    def load(self, checkpoint_idx: int = -1) -> dict:
        """
        Load checkpoint.

        Args:
            checkpoint_idx: Which checkpoint to load (-1 for latest)

        Returns:
            Checkpoint data
        """
        return self.checkpoints[checkpoint_idx]


################################################################################
# SECTION 5: TESTING & EXAMPLES
################################################################################

def demonstrate_training():
    """Demonstrate training concepts."""
    print("=" * 70)
    print("TRAINING INFRASTRUCTURE DEMONSTRATION")
    print("=" * 70)

    # Data Parallel
    print("\n--- Data Parallel ---")
    dp = DataParallel(world_size=4, rank=0)
    data = np.random.randn(32, 10)  # Batch of 32
    my_data = dp.split_batch(data, batch_size=32)
    print(f"Full batch: {data.shape}")
    print(f"GPU 0 batch: {my_data.shape}")

    # Mixed Precision
    print("\n--- Mixed Precision ---")
    mp = MixedPrecisionTrainer(loss_scale=1024.0)
    loss = 0.001  # Small loss
    scaled_loss = mp.scale_loss(loss)
    print(f"Original loss: {loss}")
    print(f"Scaled loss: {scaled_loss}")

    # Gradient Accumulation
    print("\n--- Gradient Accumulation ---")
    ga = GradientAccumulator(accumulation_steps=4)
    for i in range(4):
        grads = [np.random.randn(10, 10)]
        ga.accumulate(grads)
        print(f"Step {i+1}: accumulated={ga.should_update()}")

    final_grads = ga.get_gradients()
    print(f"Final gradient shape: {final_grads[0].shape}")

    # Checkpointing
    print("\n--- Checkpointing ---")
    cm = CheckpointManager()
    cm.save(
        model_state={'weights': np.random.randn(10, 10)},
        optimizer_state={'lr': 1e-4},
        step=1000,
        loss=0.5
    )
    loaded = cm.load()
    print(f"Loaded checkpoint at step {loaded['step']}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_training()


################################################################################
# REFERENCES
################################################################################

# [1] Rajbhandari, S., et al. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.
# [2] Micikevicius, P., et al. (2018). Mixed Precision Training.
# [3] Huang, Y., et al. (2019). GPipe: Efficient Training of Giant Neural Networks.
# [4] Shoeybi, M., et al. (2020). Megatron-LM: Training Multi-Billion Parameter Language Models.

################################################################################
