"""
################################################################################
ROBOTICS FOUNDATION MODELS — AI FOR PHYSICAL EMBODIED AGENTS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Robotics Foundation Models?
    AI models that control robots by mapping observations (cameras, sensors)
    to actions (joint torques, end-effector movements). These models bring
    the power of large-scale pre-training to physical manipulation.

Why does it matter?
    Traditional robotics requires hand-engineered controllers for each task.
    Foundation models enable:
    - General-purpose robot policies
    - Natural language instruction following
    - Few-shot adaptation to new tasks
    - Sim-to-real transfer

How does it work?
    1. Imitation Learning — Learn from expert demonstrations
    2. Robot Transformers — VLMs that output robot actions
    3. Sim-to-Real — Train in simulation, deploy in reality
    4. RL for Robotics — Learn from trial and error
    5. Language Conditioning — Follow natural language instructions

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ Robot Transformer (RT-2 style)                              │
    │                                                              │
    │  Camera Image → [Vision Encoder] → Visual Tokens            │
    │  "Pick up the red block" → [Tokenizer] → Language Tokens    │
    │  Joint Angles → [Encoder] → Proprioception Tokens           │
    │       ↓                                                      │
    │  [Causal Transformer] → Action Token Prediction             │
    │       ↓                                                      │
    │  Action Tokens → [Detokenizer] → Joint Commands             │
    └─────────────────────────────────────────────────────────────┘

Historical Context:
    - 2018: Behavioral Cloning, DAgger
    - 2022: RT-1 (Google), SayCan
    - 2023: RT-2 (Google), VIMA
    - 2024: OpenVLA, π0
    - 2025: Humanoid foundation models, dexterous manipulation
    - 2026: Generalist robot policies, multi-embodiment models

################################################################################
"""

from .robot_model import RobotModel
from .robotics_model import RoboticsModel
from .imitation_learning import BehavioralCloning, DAgger, InverseRL, MultiTaskImitation
from .robot_transformer import RobotTransformer, ActionTokenizer, VisionEncoder, MultiModalRobotPolicy
from .sim_to_real import DomainRandomization, SystemIdentification, SimToRealAdapter, Simulator
