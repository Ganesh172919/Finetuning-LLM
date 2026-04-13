"""
################################################################################
SOTA LLM FORGE — INFERENCE & SERVING
################################################################################

Quantization and speculative decoding for the SOTA LLM training stack.

Components:
    quantize.py     — FP8 / NVFP4 post-training quantization
    speculative.py  — MTP-head-driven self-speculative decoding

Serving stack:
    1. Quantize trained model (BF16 → FP8 for production)
    2. Load MTP heads as draft model for speculative decoding
    3. Serve with KV cache management (MLA: compressed latent cache)

################################################################################
"""

from .quantize import (
    QuantizationConfig,
    FP8Quantizer,
    NVFP4Quantizer,
    quantize_to_fp8,
    dequantize_from_fp8,
)

from .speculative import (
    SpeculativeConfig,
    SpeculativeDecoder,
    MLAKVCache,
    GQAKVCache,
    compare_cache_memory,
)

__all__ = [
    # Quantization
    "QuantizationConfig",
    "FP8Quantizer",
    "NVFP4Quantizer",
    "quantize_to_fp8",
    "dequantize_from_fp8",
    # Speculative Decoding
    "SpeculativeConfig",
    "SpeculativeDecoder",
    # KV Cache
    "MLAKVCache",
    "GQAKVCache",
    "compare_cache_memory",
]
