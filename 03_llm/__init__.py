"""
################################################################################
LARGE LANGUAGE MODELS (LLMs) — THE FOUNDATION OF MODERN AI
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Large Language Models?
    LLMs are neural networks trained on massive text datasets to predict
    the next token. Through this simple objective, they learn:
    - Grammar and syntax
    - Factual knowledge
    - Reasoning patterns
    - Code generation
    - And much more

Why do they matter?
    LLMs are the foundation of modern AI:
    - ChatGPT, Claude, Gemini (conversational AI)
    - Code generation (GitHub Copilot, Cursor)
    - Scientific research (protein folding, drug discovery)
    - Education (tutoring, explanation)
    - Automation (agents, workflows)

Historical Evolution:
    - 2017: Transformer architecture ("Attention Is All You Need")
    - 2018: GPT-1 (117M), BERT (340M)
    - 2019: GPT-2 (1.5B)
    - 2020: GPT-3 (175B) — few-shot learning emerges
    - 2022: ChatGPT — AI goes mainstream
    - 2023: LLaMA, Mistral — open-source revolution
    - 2024: Mixtral, DeepSeek — MoE and reasoning
    - 2025: DeepSeek R1 — reasoning models
    - 2026: Hybrid architectures, efficiency improvements

########################################

MODELS IMPLEMENTED IN THIS DIRECTORY:

1. gpt.py — GPT architecture (decoder-only)
2. encoder_decoder.py — T5-style encoder-decoder
3. encoder_only.py — BERT-style encoder
4. reasoning.py — Reasoning models (CoT, ToT)
5. moe.py — Mixture of Experts
6. slm.py — Small Language Models

################################################################################
"""

from .gpt import GPT, GPTConfig
from .moe import MixtureOfExperts
from .reasoning import ReasoningModel
