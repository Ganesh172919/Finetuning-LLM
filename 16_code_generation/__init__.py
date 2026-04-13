"""
################################################################################
CODE GENERATION — AI FOR SOFTWARE ENGINEERING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Code Generation?
    AI models that can write, understand, and transform code. These models
    are trained on large code corpora and can generate code from natural
    language descriptions, complete partial code, and debug errors.

Why does it matter?
    Code generation is one of the most impactful AI applications:
    - 10x developer productivity
    - Automated code review and testing
    - Code translation between languages
    - Bug detection and fixing
    - Documentation generation

How does it work?
    1. Code LLMs — Transformer models trained on code corpora
    2. Code Reward Models — Evaluate code quality via execution
    3. Code Search — Semantic search over code repositories
    4. AST Processing — Structural code understanding
    5. RL from Execution — Train with execution-based rewards

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ Code Generation Pipeline                                     │
    │                                                              │
    │  "Write a function to sort a list"                           │
    │       ↓                                                      │
    │  Code LLM → Generate N candidate solutions                  │
    │       ↓                                                      │
    │  Code Reward Model → Score each solution                    │
    │       ↓                                                      │
    │  RL Training → Prefer high-scoring solutions                │
    │       ↓                                                      │
    │  Best Solution: def sort_list(lst): return sorted(lst)      │
    └─────────────────────────────────────────────────────────────┘

Historical Context:
    - 2021: Codex (OpenAI), CodeParrot
    - 2022: Code LLaMA, AlphaCode
    - 2023: StarCoder, DeepSeek-Coder
    - 2024: AlphaCode 2, execution-based RL
    - 2025: SWE-bench, autonomous coding agents
    - 2026: Full-stack code agents, verified code generation

################################################################################
"""

from .code_llm import CodeLLM, CodeTokenizer
from .code_model import CodeModel
from .code_reward_model import ExecutionReward, ASTSimilarityReward, CodeQualityReward, CombinedCodeReward
from .code_search import CodeEmbedder, ContrastiveCodeSearch, CodeCloneDetector
from .ast_processing import ASTNode, ASTParser, ASTEncoder, CodePatternMatcher
