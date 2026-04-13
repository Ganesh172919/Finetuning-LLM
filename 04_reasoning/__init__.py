"""
################################################################################
REASONING MODELS — THINKING BEFORE ANSWERING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Reasoning Models?
    Reasoning models are LLMs that can think step-by-step before answering.
    Instead of immediately producing an answer, they analyze the problem,
    consider different approaches, work through the solution, and verify
    their answer. This is similar to how humans think through complex problems.

Why do they matter?
    Standard LLMs often fail at multi-step math, complex logic, planning,
    and code debugging. Reasoning models excel at these tasks by "thinking
    out loud" and allocating more test-time compute to harder problems.

How do they work?
    1. Chain of Thought (CoT) — Prompt the model to reason step by step
    2. Tree of Thoughts (ToT) — Explore multiple reasoning paths like a tree
    3. Self-Consistency — Sample many paths, vote on the best answer
    4. Reasoning Tokens — Hidden thinking before answering (o1/R1 style)
    5. MCTS Reasoning — Monte Carlo tree search over reasoning steps

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ Reasoning Model                                             │
    │                                                              │
    │  Prompt ──▶ [Think Step by Step] ──▶ [Verify] ──▶ Answer    │
    │                                                              │
    │  Chain of Thought:    A → B → C → D → Answer                │
    │  Tree of Thoughts:    A → B₁/B₂ → C₁/C₂ → Best → Answer   │
    │  Self-Consistency:    Path1 ─┐                               │
    │                             Path2 ──▶ Vote ──▶ Answer       │
    │                             Path3 ─┘                         │
    │  Reasoning Tokens:    <think>...</think> → Answer            │
    └─────────────────────────────────────────────────────────────┘

Historical Context:
    - 2022: Chain of Thought prompting (Wei et al.)
    - 2023: Tree of Thoughts (Yao et al.), Self-Consistency (Wang et al.)
    - 2024: OpenAI o1, test-time compute scaling
    - 2025: DeepSeek R1, reasoning training with RL
    - 2026: Hybrid reasoning models, AlphaProof-style MCTS

################################################################################
"""

from .chain_of_thought import ZeroShotCoT, FewShotCoT, AutoCoT, SelfAskCoT
from .tree_of_thoughts import ThoughtNode, ThoughtTree, TreeOfThoughts, BreadthFirstSearch, DepthFirstSearch
from .self_consistency import SelfConsistency, MajorityVoting, WeightedVoting, AdaptiveSelfConsistency
from .reasoning_tokens import ThinkingTokenModel, BudgetForcing, ReasoningVerifier
from .mcts_reasoning import MCTSNode, MCTSReasoning, AlphaProofStyle, UCBSelector
