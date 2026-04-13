"""
################################################################################
REINFORCEMENT LEARNING FOR LANGUAGE MODELS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is RL for LLMs?
    Reinforcement learning aligns language models with human preferences.
    After pre-training on text, RL fine-tunes models to be:
    - Helpful: Answer questions accurately
    - Harmless: Avoid dangerous content
    - Honest: Don't hallucinate or lie

Why does it matter?
    Pre-trained LLMs just predict next tokens.
    RLHF makes them:
    - Follow instructions
    - Be safe and ethical
    - Provide high-quality answers
    - Refuse harmful requests

Historical Evolution:
    - 2022: InstructGPT (OpenAI) — RLHF
    - 2023: DPO (direct preference optimization)
    - 2024: ORPO, SimPO
    - 2025: GRPO (DeepSeek), reasoning training

########################################

METHODS IMPLEMENTED:
1. RLHF: Reinforcement Learning from Human Feedback
2. DPO: Direct Preference Optimization
3. GRPO: Group Relative Policy Optimization
4. PPO: Proximal Policy Optimization
5. Reward Model: Train reward model from preferences

################################################################################
"""

from .rlhf import RLHFTrainer
from .dpo import DPOTrainer
from .reward_model import RewardModel
