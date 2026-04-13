"""
################################################################################
DIFFUSION MODELS — THEORY AND IMPLEMENTATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Diffusion Models?
    Generative models that learn to reverse a noise process.

Key Variants:
    1. DDPM: Denoising Diffusion Probabilistic Models
    2. Score-based: Learn score function
    3. Flow Matching: Straight-line interpolation
    4. Consistency: One-step generation

Interview Questions:
    1. "What is a diffusion model?"
        A model that generates data by iteratively denoising.

################################################################################
"""

from .ddpm import DDPM
from .score_matching import ScoreModel
from .flow_matching import FlowMatching
