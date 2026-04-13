"""
################################################################################
WORLD MODELS — LEARNING TO SIMULATE THE WORLD
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are World Models?
    World models are neural networks that learn to simulate how the world
    works. They predict future states given current observations and actions,
    enabling agents to "imagine" outcomes before acting.

Why do they matter?
    - Planning: Agents can plan by simulating futures internally
    - Sample efficiency: Learn from imagined experience, not just real
    - Safety: Test actions in simulation before real-world execution
    - Understanding: Models that predict must understand physics, causality

How do they work?
    1. Encode observations into latent representations
    2. Predict next states given actions (in latent space)
    3. Use predictions for planning or policy learning

Historical Evolution:
    - 1990s: Recurrent world models (Schmidhuber)
    - 2018: World Models (Ha & Schmidhuber) — VAE + MDN-RNN
    - 2022: JEPA (LeCun) — Joint Embedding Predictive Architecture
    - 2024: DIAMOND — Diffusion for world modeling
    - 2024: GameNGen — Transformer-based game simulation
    - 2024: Genie 2 — Interactive world generation

################################################################################
"""

from .jepa import JEPA, JEPAEncoder, JEPAPredictor
from .diffusion_world_model import DiffusionWorldModel, ActionConditionedDiffusion
from .transformer_world_model import TransformerWorldModel, AutoregressiveWorldModel
from .causal_world_model import CausalWorldModel, StructuralCausalModel
