"""
################################################################################
DISTRIBUTED TRAINING — SCALING TO MULTIPLE GPUs
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Distributed Training?
    Training a model across multiple GPUs or machines.
    Essential for large models that don't fit on a single GPU.

Types of Parallelism:
    1. Data Parallel (DP): Each GPU has full model, different data
    2. Tensor Parallel (TP): Split individual tensors across GPUs
    3. Pipeline Parallel (PP): Split model into stages
    4. Expert Parallel (EP): Split MoE experts across GPUs
    5. FSDP: Fully Sharded Data Parallel (ZeRO)

Why Distributed Training?
    LLaMA-70B: 140GB in fp16, needs 2+ A100-80GB
    GPT-4: estimated 3.6TB, needs 50+ GPUs

    Without distributed training, modern LLMs couldn't exist.

Interview Questions:
        Q: "What's the difference between data and model parallelism?"
        A: Data parallel: replicate model, split data
           Model parallel: split model, replicate data
           Most large training uses both (3D parallelism).

        Q: "What is ZeRO?"
        A: ZeRO (Zero Redundancy Optimizer) reduces memory in data parallel:
           Stage 1: Shard optimizer states
           Stage 2: Shard gradients
           Stage 3: Shard parameters

################################################################################
"""

import numpy as np
from typing import List, Dict, Optional

################################################################################
# SECTION 1: DATA PARALLEL
################################################################################

class DataParallel:
    """
    Data Parallel Training
    ======================

    Replicate the model on each GPU, but split the data.

    Algorithm:
    1. Copy model to all GPUs
    2. Split batch across GPUs
    3. Each GPU computes gradients on its portion
    4. All-reduce: average gradients across GPUs
    5. Each GPU updates its model with averaged gradients

    Communication:
        All-reduce: O(model_size) communication per step

    Interview Questions:
        Q: "What's the limitation of data parallelism?"
        A: Each GPU must store the full model. For very large models,
           this is impossible. Solution: model parallelism or ZeRO.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    def split_batch(self, data: np.ndarray, batch_size: int) -> np.ndarray:
        """Split batch across GPUs."""
        per_gpu = batch_size // self.world_size
        start = self.rank * per_gpu
        return data[start:start + per_gpu]

    def all_reduce(self, gradients: List[np.ndarray]) -> np.ndarray:
        """Average gradients across GPUs."""
        return np.mean(gradients, axis=0)


################################################################################
# SECTION 2: FSDP (FULLY SHARDED DATA PARALLEL)
################################################################################

class FSDP:
    """
    Fully Sharded Data Parallel (FSDP)
    ====================================

    Shards model parameters across GPUs.

    ZeRO Stages:
    Stage 1: Shard optimizer states (4x memory reduction)
    Stage 2: Shard gradients (8x memory reduction)
    Stage 3: Shard parameters (N*x memory reduction, N=GPU count)

    Memory per GPU:
    - Original: model + optimizer + gradients
    - ZeRO-1: model + optimizer/N + gradients
    - ZeRO-2: model + optimizer/N + gradients/N
    - ZeRO-3: model/N + optimizer/N + gradients/N

    Interview Questions:
        Q: "What's the difference between FSDP and DDP?"
        A: DDP replicates the full model on each GPU.
           FSDP shards the model across GPUs.
           FSDP uses less memory but more communication.
    """

    def __init__(self, world_size: int, rank: int, shard_params: bool = True):
        self.world_size = world_size
        self.rank = rank
        self.shard_params = shard_params

    def shard_parameter(self, param: np.ndarray) -> np.ndarray:
        """Shard a parameter across GPUs."""
        if not self.shard_params:
            return param

        total_size = param.size
        shard_size = total_size // self.world_size
        start = self.rank * shard_size
        return param.flat[start:start + shard_size].reshape(-1)

    def all_gather(self, shards: List[np.ndarray]) -> np.ndarray:
        """Gather shards from all GPUs."""
        return np.concatenate(shards)


################################################################################
# SECTION 3: TENSOR PARALLEL
################################################################################

class TensorParallel:
    """
    Tensor Parallel
    ===============

    Split individual tensors across GPUs.

    For a matrix multiplication Y = X @ W:
    - Column parallel: Split W into columns, each GPU computes part
    - Row parallel: Split W into rows, each GPU computes part

    Used in Megatron-LM for training large models.

    Interview Questions:
        Q: "What is tensor parallelism?"
        A: Splitting individual weight matrices across GPUs.
           Each GPU computes part of the matrix multiplication.
           Requires communication within each layer.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    def column_parallel(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Column parallel matmul.

        Split W into columns: W = [W_0 | W_1 | ... | W_{n-1}]
        Each GPU i computes: Y_i = X @ W_i
        """
        w_shard = np.array_split(w, self.world_size, axis=1)[self.rank]
        return x @ w_shard

    def row_parallel(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Row parallel matmul.

        Split W into rows: W = [W_0; W_1; ...; W_{n-1}]
        Split X into columns: X = [X_0 | X_1 | ... | X_{n-1}]
        Each GPU i computes: Y_i = X_i @ W_i
        Then all-reduce: Y = Σ Y_i
        """
        w_shard = np.array_split(w, self.world_size, axis=0)[self.rank]
        x_shard = np.array_split(x, self.world_size, axis=1)[self.rank]
        return x_shard @ w_shard


################################################################################
# SECTION 4: PIPELINE PARALLEL
################################################################################

class PipelineParallel:
    """
    Pipeline Parallel
    =================

    Split the model into stages, each on a different GPU.

    GPU 0: Layers 0-7
    GPU 1: Layers 8-15
    GPU 2: Layers 16-23
    GPU 3: Layers 24-31

    Micro-batches flow through the pipeline.

    Problem: Pipeline bubble (GPUs idle waiting for data)
    Solution: Use micro-batches to keep pipeline full

    Interview Questions:
        Q: "What is the pipeline bubble?"
        A: The time when some GPUs are idle waiting for data
           from previous stages. Minimized by using micro-batches.

        Q: "What's GPipe vs PipeDream?"
        A: GPipe: synchronous, all micro-batches before update
           PipeDream: asynchronous, update after each micro-batch
    """

    def __init__(self, n_stages: int, stage_id: int):
        self.n_stages = n_stages
        self.stage_id = stage_id

    def forward_stage(self, x: np.ndarray, layer) -> np.ndarray:
        """Forward through this pipeline stage."""
        return layer.forward(x)

    def compute_pipeline_schedule(
        self,
        n_micro_batches: int
    ) -> List[Tuple[int, str]]:
        """
        Compute pipeline schedule.

        Returns list of (micro_batch_id, 'forward'/'backward') operations.
        """
        schedule = []

        # Simple GPipe schedule
        for i in range(n_micro_batches):
            schedule.append((i, 'forward'))
        for i in range(n_micro_batches):
            schedule.append((i, 'backward'))

        return schedule


################################################################################
# SECTION 5: GRADIENT ACCUMULATION
################################################################################

class GradientAccumulator:
    """
    Gradient Accumulation
    =====================

    Accumulate gradients over multiple mini-batches before updating.
    Simulates larger batch sizes without extra memory.

    Effective batch size = mini_batch × accumulation_steps × world_size

    Interview Questions:
        Q: "Why use gradient accumulation?"
        A: When the desired batch size doesn't fit in GPU memory.
           Accumulate gradients over multiple forward/backward passes.
    """

    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self.accumulated = []
        self.step_count = 0

    def accumulate(self, gradients: List[np.ndarray]):
        """Accumulate gradients."""
        if not self.accumulated:
            self.accumulated = [np.zeros_like(g) for g in gradients]

        for i, grad in enumerate(gradients):
            self.accumulated[i] += grad / self.accumulation_steps

        self.step_count += 1

    def should_update(self) -> bool:
        """Check if should update weights."""
        return self.step_count >= self.accumulation_steps

    def get_and_reset(self) -> List[np.ndarray]:
        """Get accumulated gradients and reset."""
        grads = self.accumulated
        self.accumulated = []
        self.step_count = 0
        return grads


################################################################################
# SECTION 6: TESTING
################################################################################

def demonstrate_distributed():
    """Demonstrate distributed training concepts."""
    print("=" * 70)
    print("DISTRIBUTED TRAINING DEMONSTRATION")
    print("=" * 70)

    # Data Parallel
    print("\n--- Data Parallel ---")
    dp = DataParallel(world_size=4, rank=0)
    data = np.random.randn(32, 10)
    my_data = dp.split_batch(data, 32)
    print(f"Full batch: {data.shape}")
    print(f"GPU 0 batch: {my_data.shape}")

    # FSDP
    print("\n--- FSDP ---")
    fsdp = FSDP(world_size=4, rank=0)
    param = np.random.randn(1000)
    shard = fsdp.shard_parameter(param)
    print(f"Full param: {param.shape}")
    print(f"Shard: {shard.shape}")

    # Tensor Parallel
    print("\n--- Tensor Parallel ---")
    tp = TensorParallel(world_size=4, rank=0)
    x = np.random.randn(8, 64)
    w = np.random.randn(64, 256)
    y = tp.column_parallel(x, w)
    print(f"Input: {x.shape}")
    print(f"Weight: {w.shape}")
    print(f"Output (shard): {y.shape}")

    # Gradient Accumulation
    print("\n--- Gradient Accumulation ---")
    ga = GradientAccumulator(accumulation_steps=4)
    for i in range(4):
        grads = [np.random.randn(10, 10)]
        ga.accumulate(grads)
        print(f"Step {i+1}: should_update={ga.should_update()}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_distributed()


################################################################################
# REFERENCES
################################################################################

# [1] Rajbhandari, S., et al. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.
# [2] Shoeybi, M., et al. (2020). Megatron-LM: Training Multi-Billion Parameter Language Models.
# [3] Huang, Y., et al. (2019). GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.

################################################################################
