"""
################################################################################
LONG CONTEXT ARCHITECTURES — PROCESSING VERY LONG SEQUENCES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Long Context?
    Long context refers to models that can process sequences of 100K+ tokens.
    Standard transformers have O(N²) attention, making long sequences
    prohibitively expensive. Long context techniques solve this.

Why does it matter?
    Many real-world tasks need long context:
    - Document understanding (books, legal contracts)
    - Code analysis (entire codebases)
    - Multi-turn conversations (hundreds of turns)
    - Video understanding (hours of video)

How does it work?
    1. Ring Attention — Distribute KV across devices in a ring
    2. Context Compression — Compress KV cache to reduce memory
    3. Sliding Window — Attend only to nearby tokens
    4. Sparse Attention — Attend to a subset of tokens
    5. RoPE Scaling — Extend position encoding range

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ Long Context Model                                          │
    │                                                              │
    │  Full Sequence: [t₁ t₂ t₃ ... t₁₀₀₀₀₀]                    │
    │                                                              │
    │  Ring Attention:  Device0 [t₁..t₂₅ₖ] → Device1 → ...       │
    │  Sliding Window:  Each token attends to W neighbors          │
    │  Compression:     Keep sink + recent, compress middle        │
    │  RoPE Scaling:    Extend position range via scaling factor   │
    └─────────────────────────────────────────────────────────────┘

Historical Context:
    - 2020: Longformer (4K), BigBird (4K)
    - 2023: RoPE scaling to 100K+, YaRN
    - 2024: Ring Attention, Gemini 1M context
    - 2025: Streaming LLM, KV cache compression
    - 2026: Infinite context with compression

################################################################################
"""

from .long_context import LongContextModel, RoPEScaling
from .rope_scaling import RoPEScaling as RoPEScaler
from .ring_attention import RingAttention, BlockSparseAttention, DistributedSequenceParallelism
from .context_compression import KVCacheCompressor, AttentionSink, PromptCompression
from .sliding_window import SlidingWindowAttention, DilatedAttention, LongformerAttention
