"""
################################################################################
SOTA LLM FORGE — MODEL PACKAGE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is this package?
    The core model architecture for a state-of-the-art large language model.
    Implements attention, mixture-of-experts, multi-token prediction,
    residual streams, and the full transformer assembly.

Why does it matter?
    This is the heart of the LLM. Every downstream capability — reasoning,
    coding, instruction following — depends on the architecture choices made
    here. The components are designed to be modular and configurable so that
    different model scales and training regimes can be explored.

How does it work?
    The package is organized by component:
      - attention.py:   Three tiers of attention (GQA, MLA, Hybrid Sparse)
      - moe.py:         DeepSeekMoE-style mixture of experts
      - mtp.py:         Multi-token prediction heads
      - residual.py:    Residual stream and hyper-connections
      - transformer.py: Full model assembly
      - init.py:        muP initialization

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────┐
    │                     TransformerLM                            │
    │  ┌──────────┐                                               │
    │  │ Embedding │                                              │
    │  └────┬─────┘                                               │
    │       ↓                                                     │
    │  ┌──────────────────────────────────────────┐               │
    │  │         TransformerBlock (× N)            │               │
    │  │  ┌──────────┐   ┌─────────┐   ┌───────┐ │               │
    │  │  │ RMSNorm  │→  │Attention│→  │ Resid │ │               │
    │  │  └──────────┘   └─────────┘   └───────┘ │               │
    │  │  ┌──────────┐   ┌─────────┐   ┌───────┐ │               │
    │  │  │ RMSNorm  │→  │MoE/MLP  │→  │ Resid │ │               │
    │  │  └──────────┘   └─────────┘   └───────┘ │               │
    │  └──────────────────────────────────────────┘               │
    │       ↓                                                     │
    │  ┌──────────┐   ┌─────────────┐                            │
    │  │ RMSNorm  │→  │  LM Head    │                            │
    │  └──────────┘   └─────────────┘                            │
    │       ↓                                                     │
    │  ┌──────────────┐                                          │
    │  │  MTP Heads   │  (optional, multi-token prediction)       │
    │  └──────────────┘                                          │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2017: "Attention Is All You Need" — Transformer architecture (Vaswani et al.)
    - 2020: GPT-3 — Scaling laws for LLMs (Brown et al.)
    - 2022: Tensor Programs V / muP — Width-independent hyperparameter transfer (Yang et al.)
    - 2023: Grouped Query Attention in Llama 2 (Touvron et al.)
    - 2024: DeepSeek-V2 — Multi-head Latent Attention + MoE (DeepSeek-AI)
    - 2024: Multi-Token Prediction for LLMs (Gloeckle et al., Meta)
    - 2025: DeepSeek-V3 — Aux-loss-free MoE, Hyper-Connections
    - 2025: DeepSeek-V4 — CSA/HCA sparse attention patterns

INTERVIEW QUESTIONS:
    1. "What is the difference between MHA, MQA, and GQA?"
       MHA has separate KV for every query head. MQA shares one KV across
       all heads. GQA is the middle ground: groups of query heads share
       one KV head. GQA retains most of MHA's quality with MQA's memory
       savings.

    2. "Why use multi-token prediction instead of standard next-token?"
       MTP provides denser training signal (D loss terms per position
       instead of 1), which improves sample efficiency. At inference time,
       the MTP heads serve as a free draft model for speculative decoding,
       accelerating generation without extra training.

    3. "How does aux-loss-free load balancing work in DeepSeekMoE?"
       Instead of adding an auxiliary loss to the training objective (which
       can hurt model quality), a per-expert bias b_i is maintained. This
       bias is added to routing scores ONLY for top-k selection, not for
       computing the actual gating weights. The bias is updated each step
       proportional to the difference between target and actual load.
       This keeps experts balanced without interfering with the LM gradient.

################################################################################
"""

from .attention import (
    RotaryPositionEmbedding,
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    HybridSparseAttention,
    create_causal_mask,
)

from .moe import (
    ExpertRouter,
    SharedExperts,
    RoutedExperts,
    DeepSeekMoELayer,
)

from .mtp import (
    MultiTokenPredictionHead,
    MTPLoss,
)

from .residual import (
    ResidualStream,
    HyperConnections,
)

from .transformer import (
    TransformerConfig,
    TransformerBlock,
    TransformerLM,
)

from .init import muPInitializer

__all__ = [
    # Attention
    "RotaryPositionEmbedding",
    "GroupedQueryAttention",
    "MultiHeadLatentAttention",
    "HybridSparseAttention",
    "create_causal_mask",
    # MoE
    "ExpertRouter",
    "SharedExperts",
    "RoutedExperts",
    "DeepSeekMoELayer",
    # MTP
    "MultiTokenPredictionHead",
    "MTPLoss",
    # Residual
    "ResidualStream",
    "HyperConnections",
    # Transformer
    "TransformerConfig",
    "TransformerBlock",
    "TransformerLM",
    # Initialization
    "muPInitializer",
]
