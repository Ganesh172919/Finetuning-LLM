"""
################################################################################
SOTA LLM FORGE — EVALUATION SUITES PACKAGE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is this package?
    The collection of benchmark suite evaluators for the SOTA LLM Forge
    project. Each evaluator implements the BenchmarkSuite protocol and
    provides standardized evaluation for a specific benchmark.

Why does it matter?
    Standardized evaluation suites ensure fair, reproducible comparison
    across models. Each suite handles its own prompt formatting, answer
    extraction, and scoring logic while conforming to the common interface.

How does it work?
    Each suite implements:
    - load_data(): Load benchmark data and format prompts
    - evaluate(): Run evaluation and return scores
    - check_decontamination(): Check for training data contamination

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────┐
    │                    Benchmark Suites                           │
    │                                                              │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
    │  │  MMLU    │  │  GPQA    │  │   AIME   │  │  HLE     │   │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
    │  │  LCB     │  │  SWE     │  │ ARC-AGI  │  │FrontierM │   │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
    │                                                              │
    │  All implement BenchmarkSuite protocol                       │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2021: MMLU established as broad knowledge benchmark
    - 2023: GPQA and SWE-bench introduced for specialized evaluation
    - 2024: LiveCodeBench and FrontierMath added for code and math
    - 2025: HLE and ARC-AGI-2/3 for frontier-level evaluation
    - 2026: Comprehensive suite covers all major capability dimensions

INTERVIEW QUESTIONS:
    1. "Which benchmarks should you include in an eval suite?"
       Cover multiple dimensions: broad knowledge (MMLU-Pro), deep
       reasoning (GPQA), math (AIME, FrontierMath), code (LiveCodeBench,
       SWE-bench), and frontier challenges (HLE, ARC-AGI). Use at least
       one benchmark per capability dimension.

    2. "How do you choose between similar benchmarks?"
       Prefer contamination-resistant benchmarks (LiveCodeBench over
       HumanEval). Prefer harder benchmarks that discriminate at your
       model's level (MMLU-Pro over MMLU for frontier models). Consider
       cost (SWE-bench is expensive).

    3. "How do you aggregate results across benchmarks?"
       Report per-benchmark scores rather than a single number. Different
       benchmarks test different capabilities and have different score
       ranges. Cost-aware metrics (tokens per correct answer) enable
       fair comparison across benchmarks.

################################################################################
"""

from .mmlu import MMLUEvaluator, MMLUConfig
from .gpqa import GPQAEvaluator, GPQAConfig
from .aime import AIMEEvaluator, AIMEConfig
from .livecodebench import LiveCodeBenchEvaluator, LiveCodeBenchConfig
from .swe_bench import SWEBenchEvaluator, SWEBenchConfig
from .hle import HLEEvaluator, HLEConfig
from .arc_agi import ARCAGIEvaluator, ARCAGIConfig
from .frontiermath import FrontierMathEvaluator, FrontierMathConfig

__all__ = [
    # MMLU
    "MMLUEvaluator",
    "MMLUConfig",
    # GPQA
    "GPQAEvaluator",
    "GPQAConfig",
    # AIME
    "AIMEEvaluator",
    "AIMEConfig",
    # LiveCodeBench
    "LiveCodeBenchEvaluator",
    "LiveCodeBenchConfig",
    # SWE-bench
    "SWEBenchEvaluator",
    "SWEBenchConfig",
    # HLE
    "HLEEvaluator",
    "HLEConfig",
    # ARC-AGI
    "ARCAGIEvaluator",
    "ARCAGIConfig",
    # FrontierMath
    "FrontierMathEvaluator",
    "FrontierMathConfig",
]
