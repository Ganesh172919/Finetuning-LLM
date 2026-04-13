"""
################################################################################
TRANSFORMER VARIANTS — ALTERNATIVE ARCHITECTURES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Transformer Variants?
    Alternative architectures that address transformer limitations:
    - Quadratic attention complexity: O(n²)
    - Limited context length
    - Inference speed
    - Memory usage

Why do they matter?
    Standard transformers are powerful but expensive:
    - Attention: O(n²) in sequence length
    - KV cache: grows linearly with context
    - Inference: autoregressive = slow

    Variants aim to:
    - Reduce complexity: O(n) or O(n log n)
    - Extend context: millions of tokens
    - Speed up inference: parallel generation

########################################

VARIANTS IMPLEMENTED:
1. mamba.py — State Space Models (SSM)
2. rwkv.py — RWKV (RNN + Transformer)
3. hyena.py — Hyena (long convolutions)
4. linear_attention.py — Linear attention
5. sliding_window.py — Sliding window attention

################################################################################
"""

from .mamba import Mamba, StateSpaceModel
from .rwkv import RWKV
from .linear_attention import LinearAttention
