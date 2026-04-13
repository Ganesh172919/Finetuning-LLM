"""
################################################################################
SYNTHETIC DATA GENERATION — CREATING TRAINING DATA WITH AI
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Synthetic Data Generation?
    Using AI models to create training data instead of collecting it
    from the real world. This includes:
    - Self-play: Models training against themselves
    - Synthetic pipelines: Generating textbook-quality data
    - Data augmentation: Expanding existing datasets
    - Verification: Ensuring synthetic data quality

Why does it matter?
    Real-world data is:
    - Expensive to collect and label
    - Limited in quantity and diversity
    - Privacy-sensitive (medical, financial)
    - Biased (reflects collection process)

    Synthetic data enables:
    - Unlimited training data generation
    - Controlled diversity and coverage
    - Privacy-preserving ML
    - Rapid iteration on data quality

How does it work?
    1. Use a strong model (teacher) to generate data
    2. Filter and verify generated data quality
    3. Train student model on synthetic data
    4. Iterate: student becomes new teacher

Key Approaches:
    - Phi-4: Synthetic textbook generation for SLMs
    - Nemotron: Synthetic data from NVIDIA
    - Self-Play: Models improving through self-interaction
    - Evol-Instruct: Evolving instruction complexity

################################################################################
"""

from .self_play import SelfPlayTrainer, SelfPlayConfig
from .synthetic_pipeline import SyntheticPipeline, TextbookGenerator
from .data_augmentation import InstructionAugmenter, EvolInstruct
from .verification import DataVerifier, ExecutionVerifier, ConsistencyChecker
