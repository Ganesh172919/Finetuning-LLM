"""
################################################################################
TRAINING INFRASTRUCTURE — FROM SINGLE GPU TO CLUSTERS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Training Infrastructure?
    The systems and tools needed to train large AI models efficiently.
    This includes:
    - Data loading and preprocessing
    - Distributed training across multiple GPUs
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing and fault recovery
    - Experiment tracking

Why does it matter?
    Training large models requires:
    - Millions of dollars of compute
    - Weeks or months of training
    - Efficient use of hardware
    - Fault tolerance (GPUs fail!)

    Good infrastructure can:
    - Reduce training time by 10x
    - Reduce cost by 10x
    - Enable training larger models
    - Recover from failures automatically

Training at Scale:
    ┌─────────────────────────────────────────────────┐
    │ Single GPU: 1 model, 1 GPU                      │
    │   → Small models, debugging                      │
    │                                                  │
    │ Data Parallel: Same model, multiple GPUs         │
    │   → Each GPU has full model, different data      │
    │                                                  │
    │ Model Parallel: Split model across GPUs          │
    │   → Each GPU has part of the model               │
    │                                                  │
    │ Pipeline Parallel: Split model into stages       │
    │   → Different layers on different GPUs            │
    │                                                  │
    │ Tensor Parallel: Split individual tensors         │
    │   → Split matrix multiplication across GPUs      │
    │                                                  │
    │ Expert Parallel: Split MoE experts across GPUs   │
    │   → Different experts on different GPUs          │
    └─────────────────────────────────────────────────┘

################################################################################
"""

from .distributed import DataParallel
from .distributed import CheckpointManager
