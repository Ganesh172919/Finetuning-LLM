"""
################################################################################
SOTA LLM FORGE — EVALUATION HARNESS PACKAGE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is this package?
    The unified evaluation harness for the SOTA LLM Forge project.
    Provides a standardized framework for running models against
    multiple benchmark suites with full reproducibility, decontamination
    checking, and cost-aware metrics.

Why does it matter?
    Without a unified harness, evaluations are ad-hoc, poorly logged,
    and cannot detect contamination. This harness ensures every
    evaluation is fair, reproducible, and cost-aware, enabling
    meaningful comparison across models and checkpoints.

How does it work?
    1. Create an EvalHarness with configuration
    2. Point it at a model and list of benchmark suites
    3. Run evaluation (handles prompting, generation, scoring)
    4. Get EvalResults with accuracy, cost metrics, and decontamination

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────┐
    │                    Eval Package                              │
    │                                                              │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │                  EvalHarness                          │   │
    │  │  - run(model, suites, config) → EvalResults          │   │
    │  │  - DecontaminationChecker                             │   │
    │  │  - CostAwareMetrics                                   │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                          │                                   │
    │                          ▼                                   │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │               Benchmark Suites                        │   │
    │  │  MMLU │ GPQA │ AIME │ LCB │ SWE │ HLE │ ARC │ FM    │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2020: GPT-3 evaluation established broad benchmark evaluation
    - 2022: HELM standardized evaluation with reproducibility
    - 2024: Decontamination became essential for credible evaluation
    - 2025: Cost-aware metrics (tokens per correct) gained importance
    - 2026: This harness combines all best practices

INTERVIEW QUESTIONS:
    1. "What makes a good evaluation harness?"
       Three pillars: (1) reproducibility through detailed logging of
       prompts and parameters, (2) fairness through identical pipelines
       across models, (3) cost awareness through token-per-correct metrics.
       Add decontamination as a guardrail.

    2. "How do you handle benchmark saturation?"
       Replace saturated benchmarks with harder alternatives: MMLU-Pro
       for MMLU, AIME for GSM8K, LiveCodeBench for HumanEval. Keep
       saturated benchmarks as sanity checks but don't headline them.

    3. "What is the minimum eval suite for a new model?"
       At minimum: one broad knowledge benchmark (MMLU-Pro), one
       reasoning benchmark (GPQA or AIME), one code benchmark
       (LiveCodeBench), and one frontier benchmark (HLE or ARC-AGI).
       This covers the main capability dimensions.

################################################################################
"""

from .harness import (
    EvalHarness,
    EvalConfig,
    SamplingConfig,
    EvalResults,
    SuiteResult,
    DecontaminationStatus,
    DecontaminationChecker,
    CostAwareMetrics,
    BenchmarkSuite,
)

from .suites import (
    MMLUEvaluator,
    GPQAEvaluator,
    AIMEEvaluator,
    LiveCodeBenchEvaluator,
    SWEBenchEvaluator,
    HLEEvaluator,
    ARCAGIEvaluator,
    FrontierMathEvaluator,
)

__all__ = [
    # Core harness
    "EvalHarness",
    "EvalConfig",
    "SamplingConfig",
    "EvalResults",
    "SuiteResult",
    "DecontaminationStatus",
    "DecontaminationChecker",
    "CostAwareMetrics",
    "BenchmarkSuite",
    # Benchmark suites
    "MMLUEvaluator",
    "GPQAEvaluator",
    "AIMEEvaluator",
    "LiveCodeBenchEvaluator",
    "SWEBenchEvaluator",
    "HLEEvaluator",
    "ARCAGIEvaluator",
    "FrontierMathEvaluator",
]
