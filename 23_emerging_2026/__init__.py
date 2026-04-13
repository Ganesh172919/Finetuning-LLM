"""
################################################################################
EMERGING 2026 RESEARCH ARCHITECTURES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Emerging Architectures?
    Cutting-edge research directions that may define the next generation
    of AI models. These are experimental and rapidly evolving.

Key Trends in 2026:
    1. Hybrid SSM-Attention: Combine state space models with attention
    2. Multimodal Unified: Single model for all modalities
    3. Reasoning Models: Test-time compute scaling
    4. Efficient Architectures: Lower cost, same quality
    5. World Models: Understanding physics and causality
    6. Agent Teams: Multiple AI agents collaborating (Claude 4.6)
    7. Auto-routing: System selects fast vs reasoning model (GPT-5)
    8. Sparse MoE: 128 experts, only 8 active per token (Gemini 3)

########################################

ARCHITECTURES COVERED:
1. hybrid_ssm.py — Combined SSM + Attention + Mamba-2 + MoE-Mamba
2. test_time_compute.py — Reasoning at inference
3. world_models.py — Physics and causality
4. efficient_transformers.py — Cost-effective architectures
5. neural_scaling.py — Scaling laws and predictions
6. deepseek_r1.py — DeepSeek R1 (MLA, MoE, GRPO training)
7. claude5.py — Claude 5 (model tiers, agent teams, safety)
8. gemini3.py — Gemini 3 (sparse MoE, Deep Think, Nano Banana)
9. gpt5.py — GPT-5 (auto-routing, native multimodal, safe completions)

################################################################################
"""

from .hybrid_ssm import HybridSSM, Mamba2Block, MoEMamba
from .test_time_compute import TestTimeCompute
from .deepseek_r1 import DeepSeekR1, MultiHeadLatentAttention, DeepSeekMoE, GRPOTrainer
from .claude5 import Claude5, ModelTier, AgentTeam, ExtendedThinking
from .gemini3 import Gemini3, GeminiTier, SparseMoE, DeepThink, NanoBanana
from .gpt5 import GPT5, GPTRouter, MultimodalEncoder, SafeCompletions
