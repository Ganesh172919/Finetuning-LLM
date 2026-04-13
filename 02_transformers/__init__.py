"""
################################################################################
TRANSFORMER ARCHITECTURE — THE ENGINE OF MODERN AI
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Transformer?
    A transformer is a neural network architecture that processes sequences
    using self-attention mechanisms. Unlike RNNs, it processes all tokens
    in parallel, making it vastly more efficient on modern hardware.

Why does it matter?
    The transformer powers virtually ALL modern AI:
    - GPT, Claude, Gemini, LLaMA (language)
    - CLIP, SigLIP (multimodal)
    - Stable Diffusion, Flux (image generation)
    - Whisper (speech)
    - And thousands more

    The 2017 paper "Attention Is All You Need" is arguably the most
    impactful paper in AI history.

How does it work?
    1. Input tokens → Embeddings (vectors)
    2. Add positional information (RoPE, ALiBi)
    3. Multi-head self-attention (tokens look at each other)
    4. Feed-forward network (process each token)
    5. Repeat N times (layers)
    6. Output → Next token probabilities

Historical Evolution:
    - 2014: Seq2Seq with attention (Bahdanau)
    - 2017: Transformer ("Attention Is All You Need")
    - 2018: BERT, GPT-1
    - 2019: GPT-2, T5
    - 2020: GPT-3 (scaling laws)
    - 2022: ChatGPT, Chinchilla
    - 2023: LLaMA, Mistral, Claude
    - 2024: Mixtral, DeepSeek, Llama 3
    - 2025: DeepSeek R1, reasoning models
    - 2026: Hybrid architectures, state space models

########################################

MODULES IN THIS DIRECTORY:

1. attention.py — All attention mechanisms
2. embeddings.py — Token and position embeddings
3. layers.py — Core transformer layers
4. model.py — Complete transformer model
5. training.py — Training pipeline
6. inference.py — Inference and generation

################################################################################
"""

from .attention import MultiHeadAttention, FlashAttention
from .embeddings import TokenEmbedding, RoPE, ALiBi
from .layers import TransformerBlock, FeedForward, RMSNorm
from .model import TransformerLM
