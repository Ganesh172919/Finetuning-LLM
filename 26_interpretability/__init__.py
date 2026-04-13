"""
################################################################################
MECHANISTIC INTERPRETABILITY — UNDERSTANDING HOW NEURAL NETWORKS WORK
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Mechanistic Interpretability?
    The study of understanding WHAT neural networks learn internally
    and HOW they compute. Instead of treating models as black boxes,
    we reverse-engineer their internal mechanisms.

Why does it matter?
    Neural networks are powerful but opaque:
    - We don't understand WHY they make certain decisions
    - We can't predict WHEN they'll fail
    - We can't ensure they're safe and aligned
    - We can't debug them effectively

    Mechanistic interpretability enables:
    - Understanding model capabilities and failures
    - Detecting deceptive alignment
    - Building trust through transparency
    - Improving models by understanding them

Key Concepts:
    - Features: Internal representations of concepts
    - Circuits: Groups of neurons that implement algorithms
    - Superposition: Features overlapping in neuron space
    - Monosemanticity: One neuron = one concept (ideal)

Historical Evolution:
    - 2020: Zoom In (Anthropic) — Introduction to circuits
    - 2021: Mathematical Framework for Transformer Circuits
    - 2023: Towards Monosemanticity — Sparse autoencoders
    - 2024: Scaling Monosemanticity — Features in Claude

################################################################################
"""

from .feature_visualization import FeatureVisualizer, ActivationMaximization
from .circuit_analysis import CircuitTracer, InductionHead
from .probing import LinearProbe, RepresentationAnalysis
from .sparse_autoencoder import SparseAutoencoder, FeatureDiscovery
