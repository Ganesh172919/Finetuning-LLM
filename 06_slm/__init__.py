"""
################################################################################
SMALL LANGUAGE MODELS — EFFICIENT AI
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Small Language Models?
    SLMs are language models with fewer parameters (1B-7B) designed for
    efficiency and deployment on edge devices. They achieve competitive
    performance through better data, distillation, and architecture design.

Why do they matter?
    Large models (70B+) need multiple GPUs and have high latency/cost.
    SLMs can run on a single GPU or mobile device, provide fast responses,
    and are cheaper to serve. They democratize AI deployment.

How do they work?
    1. Efficient Architectures — GQA, sliding window, parallel attention
    2. Knowledge Distillation — Learn from larger teacher models
    3. Quantization — INT4/INT8 weights for faster inference
    4. Pruning — Remove redundant weights
    5. Efficient Training — Mixed precision, gradient checkpointing

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ Small Language Model                                        │
    │                                                              │
    │  Input → [Efficient Attention] → [Compact FFN] → Output     │
    │                                                              │
    │  Key Features:                                               │
    │  • Grouped Query Attention (GQA) — fewer KV heads            │
    │  • Sliding Window — local attention for efficiency            │
    │  • SwiGLU/GeGLU — better activation functions                 │
    │  • RoPE — position encoding for length generalization         │
    │  • RMSNorm — faster than LayerNorm                           │
    └─────────────────────────────────────────────────────────────┘

Historical Context:
    - 2023: Phi-1.5 (1.3B), Gemma-2B, Mistral-7B
    - 2024: Phi-3 (3.8B), Gemma 2, Qwen2, Mistral NeMo
    - 2025: Phi-4-mini, Gemma 3, Qwen3 — approaching 70B quality
    - 2026: On-device SLMs, sub-1B models matching 2023 7B models

################################################################################
"""

from .slm_architectures import Phi4Mini, Gemma3Small, SLMTraining
from .quantization import AbsmaxQuantizer, ZeropointQuantizer, GPTQQuantizer, AWQQuantizer, QuantizedLinear
from .pruning import MagnitudePruner, StructuredPruner, SparseGPTPruner, IterativePruning
from .efficient_training import MixedPrecision, GradientCheckpointing, GradientAccumulation, LearningRateScheduler
from .knowledge_distillation import KnowledgeDistillation
