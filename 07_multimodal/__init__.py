"""
################################################################################
MULTIMODAL MODELS — CONNECTING VISION, LANGUAGE, AND AUDIO
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Multimodal Models?
    Models that can process and generate multiple types of data:
    - Text (language)
    - Images (vision)
    - Audio (speech, music)
    - Video (temporal vision)

    They learn joint representations that connect these modalities.

Why do they matter?
    The real world is multimodal. Humans read, see, hear, and speak.
    Multimodal models can:
    - Answer questions about images (VQA)
    - Generate images from text (text-to-image)
    - Describe images in words (captioning)
    - Understand video content
    - Process speech and text together

Historical Evolution:
    - 2021: CLIP (contrastive learning)
    - 2022: Flamingo (visual language model)
    - 2023: LLaVA (visual instruction tuning)
    - 2024: GPT-4V, Gemini, Claude 3 (native multimodal)
    - 2025: Unified multimodal architectures
    - 2026: Omni models (any-to-any generation)

########################################

MODELS IMPLEMENTED:

1. clip.py — Contrastive Language-Image Pre-training
2. llava.py — Visual Language Model
3. cross_attention.py — Cross-attention mechanisms
4. fusion.py — Multimodal fusion strategies

################################################################################
"""

from .clip import CLIP, ContrastiveLoss
from .cross_attention import CrossAttention
from .fusion import MultimodalFusion
