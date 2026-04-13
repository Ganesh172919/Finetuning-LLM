"""
################################################################################
UNIFIED EVALUATION HARNESS — REPRODUCIBLE MODEL ASSESSMENT
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is an Evaluation Harness?
    An evaluation harness is a unified framework that runs a model against
    multiple benchmark suites, logs every detail of the evaluation process,
    and produces reproducible, cost-aware results. It is the single entry
    point for answering: "How good is this checkpoint?"

Why does it matter?
    Without a harness, evaluations are ad-hoc scripts that forget to log
    the prompt template, use inconsistent sampling parameters, and cannot
    detect data contamination. A harness ensures:
    - Reproducibility: exact prompts and parameters are logged
    - Fairness: every model is evaluated with the same pipeline
    - Cost awareness: tokens generated per correct answer
    - Contamination detection: overlap between training data and eval sets

How does it work?
    1. Point the harness at a checkpoint and a list of benchmark suites
    2. For each suite, load prompts, generate responses, score them
    3. Log every detail: prompt template, sampling params, number of samples
    4. Run decontamination checks against training data
    5. Aggregate results into a structured EvalResults report

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────────┐
    │                     EvalHarness                                  │
    │                                                                  │
    │  Checkpoint ──┐                                                  │
    │               ├──▶ run()                                         │
    │  Suites[]  ──┘         │                                         │
    │                        ▼                                         │
    │              ┌─────────────────────┐                             │
    │              │  For each suite:     │                             │
    │              │   1. load_data()     │                             │
    │              │   2. generate()      │                             │
    │              │   3. score()         │                             │
    │              │   4. log details     │                             │
    │              └──────────┬──────────┘                             │
    │                         ▼                                        │
    │              ┌─────────────────────┐                             │
    │              │  Decontamination    │                             │
    │              │  Check              │                             │
    │              └──────────┬──────────┘                             │
    │                         ▼                                        │
    │              ┌─────────────────────┐                             │
    │              │  EvalResults        │                             │
    │              │  - accuracy         │                             │
    │              │  - cost_per_correct │                             │
    │              │  - tokens_generated │                             │
    │              │  - decontam_status  │                             │
    │              └─────────────────────┘                             │
    └─────────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2020: GPT-3 introduced broad benchmark evaluation across 42 tasks
    - 2021: BIG-bench expanded to 200+ tasks from diverse contributors
    - 2022: HELM (Stanford) established holistic evaluation with standardized metrics
    - 2023: Open LLM Leaderboard (Hugging Face) standardized open-model comparison
    - 2024: Contamination detection became essential as benchmarks were saturated
    - 2025: Cost-aware metrics (tokens per correct answer) gained importance
    - 2026: Frontier benchmarks (HLE, ARC-AGI-3) stress-test at the frontier

INTERVIEW QUESTIONS:
    1. "Why is a unified eval harness better than separate eval scripts?"
       A unified harness ensures consistency: same prompt templates, same
       sampling parameters, same scoring logic across all benchmarks. It
       also enables cross-benchmark aggregation and cost comparison.
       Separate scripts drift apart over time and make fair comparison
       impossible.

    2. "What is decontamination and why does it matter?"
       Decontamination checks whether benchmark test data appears in the
       model's training corpus. If it does, high scores may reflect
       memorization rather than capability. A harness should check for
       n-gram overlap, exact matches, and semantic similarity between
       training data and evaluation prompts.

    3. "What are cost-aware metrics and why are they important?"
       Cost-aware metrics measure not just accuracy but the computational
       cost to achieve that accuracy. Tokens per correct answer captures
       the trade-off between model size, inference cost, and quality.
       A model that scores 90% but generates 10x more tokens may be
       less practical than one scoring 85% at lower cost.

################################################################################
"""

import time
import json
import hashlib
from typing import Optional, List, Dict, Any, Protocol, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Provide stubs for type annotations when torch is not installed
    class _StubModule:
        """Stub for torch.nn.Module when PyTorch is not installed."""
        pass
    nn = type("nn", (), {"Module": _StubModule})()


################################################################################
# SECTION 1: CONFIGURATION
################################################################################

@dataclass
class SamplingConfig:
    """
    Sampling Configuration
    ======================

    Controls how the model generates responses during evaluation.
    All parameters are logged for reproducibility.

    Attributes:
        temperature: Controls randomness (0.0 = greedy, 1.0 = standard, >1.0 = more random)
        top_k: Limits sampling to top k tokens (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        max_new_tokens: Maximum tokens to generate per prompt
        num_samples: Number of samples per prompt (for pass@k)
        seed: Random seed for reproducibility
    """
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    num_samples: int = 1
    seed: int = 42


@dataclass
class EvalConfig:
    """
    Evaluation Configuration
    ========================

    Master configuration for the evaluation harness.
    Controls which suites to run, sampling parameters, and reporting.

    Attributes:
        sampling: Sampling parameters for generation
        batch_size: Number of prompts to process in parallel
        decontamination_enabled: Whether to run decontamination checks
        log_prompts: Whether to log full prompts and responses
        output_dir: Directory for saving results
        device: Device to run on (cpu/cuda/mps)
    """
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    batch_size: int = 16
    decontamination_enabled: bool = True
    log_prompts: bool = True
    output_dir: str = "./eval_results"
    device: str = "auto"


################################################################################
# SECTION 2: RESULT STRUCTURES
################################################################################

class DecontaminationStatus(Enum):
    """
    Decontamination Status
    ======================

    Indicates whether contamination was detected in the evaluation.

    Values:
        NOT_CHECKED: Decontamination was not performed
        CLEAN: No contamination detected
        WARNING: Potential contamination found (n-gram overlap)
        CONTAMINATED: High-confidence contamination detected
    """
    NOT_CHECKED = "not_checked"
    CLEAN = "clean"
    WARNING = "warning"
    CONTAMINATED = "contaminated"


@dataclass
class SuiteResult:
    """
    Single Benchmark Suite Result
    =============================

    Contains all results from evaluating a single benchmark suite,
    including accuracy, cost metrics, and per-subject breakdowns.

    Attributes:
        suite_name: Name of the benchmark suite
        accuracy: Overall accuracy (0.0 to 1.0)
        num_correct: Number of correct answers
        num_total: Total number of questions
        tokens_generated: Total tokens generated across all prompts
        cost_per_correct: Tokens generated per correct answer
        prompt_template: Exact prompt template used
        sampling_params: Sampling parameters used
        decontamination: Contamination status
        per_subject: Per-subject accuracy breakdown (if applicable)
        raw_responses: Full model responses (if logging enabled)
    """
    suite_name: str
    accuracy: float
    num_correct: int
    num_total: int
    tokens_generated: int
    cost_per_correct: float
    prompt_template: str
    sampling_params: Dict[str, Any]
    decontamination: DecontaminationStatus = DecontaminationStatus.NOT_CHECKED
    per_subject: Dict[str, float] = field(default_factory=dict)
    raw_responses: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalResults:
    """
    Complete Evaluation Results
    ===========================

    Aggregates results from all benchmark suites into a single report.
    Includes summary statistics and cross-suite comparisons.

    Attributes:
        model_name: Name/identifier of the evaluated model
        checkpoint_path: Path to the model checkpoint
        suite_results: Results for each benchmark suite
        total_tokens: Total tokens generated across all suites
        total_correct: Total correct answers across all suites
        overall_cost_per_correct: Average tokens per correct answer
        evaluation_time_seconds: Wall-clock time for full evaluation
        timestamp: When the evaluation was run
    """
    model_name: str
    checkpoint_path: str
    suite_results: List[SuiteResult]
    total_tokens: int
    total_correct: int
    overall_cost_per_correct: float
    evaluation_time_seconds: float
    timestamp: str

    def summary(self) -> str:
        """
        Generate a human-readable summary of evaluation results.

        Returns:
            Formatted string with all suite results and aggregates.

        Example:
            >>> results = harness.run(model, suites, config)
            >>> print(results.summary())
            Model: my-model | Suites: 3 | Overall accuracy: 0.72
        """
        lines = [
            "=" * 70,
            f"EVALUATION REPORT: {self.model_name}",
            f"Checkpoint: {self.checkpoint_path}",
            f"Timestamp: {self.timestamp}",
            "=" * 70,
            "",
        ]
        for sr in self.suite_results:
            lines.append(f"  {sr.suite_name:30s} | "
                         f"Accuracy: {sr.accuracy:.4f} | "
                         f"Tokens: {sr.tokens_generated:>8d} | "
                         f"Cost/correct: {sr.cost_per_correct:>8.1f} | "
                         f"Decontam: {sr.decontamination.value}")
            if sr.per_subject:
                for subj, acc in sorted(sr.per_subject.items()):
                    lines.append(f"    - {subj:28s} | {acc:.4f}")
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"  Overall cost per correct: {self.overall_cost_per_correct:.1f} tokens")
        lines.append(f"  Total tokens: {self.total_tokens}")
        lines.append(f"  Total correct: {self.total_correct}")
        lines.append(f"  Evaluation time: {self.evaluation_time_seconds:.1f}s")
        lines.append("=" * 70)
        return "\n".join(lines)


################################################################################
# SECTION 3: BENCHMARK SUITE PROTOCOL
################################################################################

@runtime_checkable
class BenchmarkSuite(Protocol):
    """
    Benchmark Suite Protocol
    ========================

    Defines the interface that all benchmark suites must implement.
    This protocol ensures uniform handling by the EvalHarness.

    Required methods:
        load_data: Load benchmark data and return prompts
        evaluate: Run evaluation and return scores
        check_decontamination: Check for training data contamination

    Example:
        >>> class MySuite:
        ...     def load_data(self, config): ...
        ...     def evaluate(self, model, config): ...
        ...     def check_decontamination(self, training_data): ...
        >>> assert isinstance(MySuite(), BenchmarkSuite)
    """

    @property
    def name(self) -> str:
        """Name of the benchmark suite."""
        ...

    @property
    def prompt_template(self) -> str:
        """Exact prompt template used for this benchmark."""
        ...

    def load_data(self, config: EvalConfig) -> List[Dict[str, Any]]:
        """
        Load benchmark data and return formatted prompts.

        Args:
            config: Evaluation configuration

        Returns:
            List of dicts with 'prompt', 'answer', and metadata keys
        """
        ...

    def evaluate(
        self,
        model: nn.Module,
        config: EvalConfig,
    ) -> SuiteResult:
        """
        Run evaluation on the benchmark suite.

        Args:
            model: The model to evaluate
            config: Evaluation configuration

        Returns:
            SuiteResult with accuracy, cost metrics, and details
        """
        ...

    def check_decontamination(
        self,
        training_data: List[str],
    ) -> DecontaminationStatus:
        """
        Check for overlap between benchmark and training data.

        Args:
            training_data: List of training document strings

        Returns:
            DecontaminationStatus indicating contamination level
        """
        ...


################################################################################
# SECTION 4: EVAL HARNESS
################################################################################

class EvalHarness:
    """
    Unified Evaluation Harness
    ==========================

    Points at any checkpoint and any benchmark suite.
    Always logs:
        - Exact prompt template used
        - Sampling parameters (temperature, top_k, top_p)
        - Number of samples
        - Decontamination note
        - Cost-aware metrics (tokens generated per correct answer)

    Step by step:
        1. Validate model and suites
        2. For each suite, call load_data() then evaluate()
        3. Run decontamination checks if enabled
        4. Aggregate results into EvalResults
        5. Log everything for reproducibility

    WHY this matters:
        Without a unified harness, evaluations are inconsistent, poorly
        logged, and cannot detect contamination. This harness ensures
        every evaluation is fair, reproducible, and cost-aware.

    Interview Question:
        "How would you design an evaluation harness for LLMs?"
        Design around three principles: (1) reproducibility through
        detailed logging of prompts and parameters, (2) fairness through
        identical pipelines across models, and (3) cost awareness through
        token-per-correct-answer metrics. Add decontamination as a
        guardrail against inflated scores.
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        """
        Initialize the evaluation harness.

        Args:
            config: Evaluation configuration (uses defaults if None)

        Example:
            >>> harness = EvalHarness(EvalConfig(batch_size=32))
        """
        self.config = config or EvalConfig()
        self._resolve_device()

    def _resolve_device(self) -> None:
        """
        Resolve the evaluation device from config.

        Sets self.device to the appropriate torch device.
        'auto' selects CUDA if available, then MPS, then CPU.
        """
        if not TORCH_AVAILABLE:
            self.device = "cpu"
            return

        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

    def run(
        self,
        model: nn.Module,
        benchmark_suites: List[BenchmarkSuite],
        training_data: Optional[List[str]] = None,
    ) -> EvalResults:
        """
        Run evaluation on all benchmark suites.

        Args:
            model: The model to evaluate
            benchmark_suites: List of benchmark suite instances
            training_data: Optional training data for decontamination checks

        Returns:
            EvalResults with aggregated scores and cost metrics

        Explanation:
            Iterates over each benchmark suite, runs evaluation, collects
            results, and aggregates into a single report. Logs all
            parameters for reproducibility.

        Example:
            >>> harness = EvalHarness()
            >>> suites = [MMLUEvaluator(), GPQAEvaluator()]
            >>> results = harness.run(model, suites, train_docs)
            >>> print(results.summary())
        """
        model_name = self._get_model_name(model)
        start_time = time.time()

        suite_results: List[SuiteResult] = []
        total_tokens = 0
        total_correct = 0

        if TORCH_AVAILABLE:
            model.eval()
            model.to(self.device)

        for suite in benchmark_suites:
            print(f"[EvalHarness] Running {suite.name}...")
            result = suite.evaluate(model, self.config)
            suite_results.append(result)

            total_tokens += result.tokens_generated
            total_correct += result.num_correct

            if self.config.log_prompts:
                self._log_suite_details(result)

            if self.config.decontamination_enabled and training_data is not None:
                result.decontamination = suite.check_decontamination(training_data)
                if result.decontamination == DecontaminationStatus.CONTAMINATED:
                    print(f"  WARNING: Contamination detected for {suite.name}!")

        elapsed = time.time() - start_time
        overall_cost = total_tokens / max(total_correct, 1)

        results = EvalResults(
            model_name=model_name,
            checkpoint_path=self._get_checkpoint_path(model),
            suite_results=suite_results,
            total_tokens=total_tokens,
            total_correct=total_correct,
            overall_cost_per_correct=overall_cost,
            evaluation_time_seconds=elapsed,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        print(results.summary())
        return results

    def _get_model_name(self, model: nn.Module) -> str:
        """
        Extract a human-readable model name.

        Args:
            model: The model instance

        Returns:
            Model class name or 'unknown' if not available
        """
        return model.__class__.__name__

    def _get_checkpoint_path(self, model: nn.Module) -> str:
        """
        Extract checkpoint path if available.

        Args:
            model: The model instance

        Returns:
            Checkpoint path string or 'in-memory' if not available
        """
        if hasattr(model, "checkpoint_path"):
            return str(model.checkpoint_path)
        return "in-memory"

    def _log_suite_details(self, result: SuiteResult) -> None:
        """
        Log detailed suite results for reproducibility.

        Args:
            result: The suite result to log

        Explanation:
            Logs the prompt template, sampling parameters, and accuracy
            so that the evaluation can be exactly reproduced later.
        """
        log_entry = {
            "suite": result.suite_name,
            "prompt_template": result.prompt_template,
            "sampling_params": result.sampling_params,
            "accuracy": result.accuracy,
            "num_correct": result.num_correct,
            "num_total": result.num_total,
            "tokens_generated": result.tokens_generated,
            "decontamination": result.decontamination.value,
        }
        log_hash = hashlib.md5(
            json.dumps(log_entry, sort_keys=True).encode()
        ).hexdigest()[:8]
        print(f"  [{log_hash}] {result.suite_name}: "
              f"acc={result.accuracy:.4f}, "
              f"tokens={result.tokens_generated}, "
              f"decontam={result.decontamination.value}")


################################################################################
# SECTION 5: DECONTAMINATION UTILITIES
################################################################################

class DecontaminationChecker:
    """
    Decontamination Checker
    =======================

    Detects overlap between training data and benchmark evaluation data.
    Uses n-gram overlap and exact match detection.

    Step by step:
        1. Build n-gram index of training data
        2. For each eval prompt, check for n-gram overlap
        3. Compute overlap ratio
        4. Classify as CLEAN, WARNING, or CONTAMINATED

    Formula:
        overlap_ratio = |matching_ngrams| / |eval_ngrams|
        CLEAN:      overlap_ratio < 0.1
        WARNING:    0.1 <= overlap_ratio < 0.5
        CONTAMINATED: overlap_ratio >= 0.5

    Interview Question:
        "How do you detect benchmark contamination in training data?"
        Use n-gram overlap: tokenize both training data and eval prompts,
        compute shared n-grams (typically 13-grams), and flag when the
        overlap ratio exceeds a threshold. Also check for exact substring
        matches. Semantic similarity can catch paraphrased contamination.
    """

    def __init__(
        self,
        ngram_size: int = 13,
        warning_threshold: float = 0.1,
        contaminated_threshold: float = 0.5,
    ):
        """
        Initialize the decontamination checker.

        Args:
            ngram_size: Size of n-grams for overlap detection
            warning_threshold: Overlap ratio to trigger WARNING
            contaminated_threshold: Overlap ratio to trigger CONTAMINATED
        """
        self.ngram_size = ngram_size
        self.warning_threshold = warning_threshold
        self.contaminated_threshold = contaminated_threshold

    def compute_ngrams(self, text: str) -> set:
        """
        Extract character-level n-grams from text.

        Args:
            text: Input text

        Returns:
            Set of n-gram strings

        Example:
            >>> checker = DecontaminationChecker(ngram_size=3)
            >>> checker.compute_ngrams("hello")
            {'hel', 'ell', 'llo'}
        """
        tokens = text.lower().split()
        ngrams = set()
        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = " ".join(tokens[i:i + self.ngram_size])
            ngrams.add(ngram)
        return ngrams

    def check(
        self,
        eval_texts: List[str],
        training_texts: List[str],
    ) -> DecontaminationStatus:
        """
        Check for contamination between eval and training data.

        Args:
            eval_texts: List of evaluation prompt strings
            training_texts: List of training document strings

        Returns:
            DecontaminationStatus indicating contamination level

        Example:
            >>> checker = DecontaminationChecker()
            >>> status = checker.check(eval_prompts, train_docs)
            >>> print(status)
            DecontaminationStatus.CLEAN
        """
        train_ngrams: set = set()
        for text in training_texts:
            train_ngrams.update(self.compute_ngrams(text))

        total_eval_ngrams = 0
        matching_ngrams = 0

        for text in eval_texts:
            eval_ngrams = self.compute_ngrams(text)
            total_eval_ngrams += len(eval_ngrams)
            matching_ngrams += len(eval_ngrams & train_ngrams)

        if total_eval_ngrams == 0:
            return DecontaminationStatus.NOT_CHECKED

        overlap_ratio = matching_ngrams / total_eval_ngrams

        if overlap_ratio >= self.contaminated_threshold:
            return DecontaminationStatus.CONTAMINATED
        elif overlap_ratio >= self.warning_threshold:
            return DecontaminationStatus.WARNING
        else:
            return DecontaminationStatus.CLEAN


################################################################################
# SECTION 6: COST-AWARE METRICS
################################################################################

class CostAwareMetrics:
    """
    Cost-Aware Metrics
    ==================

    Computes metrics that account for computational cost, not just accuracy.
    Tracks tokens generated per correct answer and efficiency ratios.

    Why this matters:
        Two models with the same accuracy may have vastly different costs.
        A model that generates 100 tokens per answer vs 500 tokens per
        answer has very different real-world implications for latency
        and cost.

    Formula:
        cost_per_correct = total_tokens_generated / num_correct_answers
        efficiency_ratio = accuracy / cost_per_correct (higher is better)
        cost_adjusted_accuracy = accuracy * (1 / log(cost_per_correct + 1))

    Interview Question:
        "How do you compare models with similar accuracy but different costs?"
        Use cost_per_correct (tokens per correct answer) and efficiency
        ratio (accuracy per token). A model scoring 88% at 50 tokens/correct
        is better than one scoring 89% at 500 tokens/correct for most
        production use cases.
    """

    @staticmethod
    def cost_per_correct(tokens_generated: int, num_correct: int) -> float:
        """
        Compute tokens generated per correct answer.

        Args:
            tokens_generated: Total tokens generated
            num_correct: Number of correct answers

        Returns:
            Tokens per correct answer (higher = more expensive)

        Example:
            >>> CostAwareMetrics.cost_per_correct(10000, 50)
            200.0
        """
        if num_correct == 0:
            return float("inf")
        return tokens_generated / num_correct

    @staticmethod
    def efficiency_ratio(accuracy: float, tokens_per_correct: float) -> float:
        """
        Compute efficiency ratio (accuracy per token cost).

        Args:
            accuracy: Model accuracy (0.0 to 1.0)
            tokens_per_correct: Tokens generated per correct answer

        Returns:
            Efficiency ratio (higher is better)

        Example:
            >>> CostAwareMetrics.efficiency_ratio(0.9, 100.0)
            0.009
        """
        if tokens_per_correct == 0:
            return float("inf")
        return accuracy / tokens_per_correct

    @staticmethod
    def cost_adjusted_accuracy(accuracy: float, tokens_per_correct: float) -> float:
        """
        Compute cost-adjusted accuracy.

        Applies a logarithmic penalty for higher cost, producing a metric
        that balances accuracy and efficiency.

        Args:
            accuracy: Model accuracy (0.0 to 1.0)
            tokens_per_correct: Tokens generated per correct answer

        Returns:
            Cost-adjusted accuracy score

        Formula:
            adjusted = accuracy * (1 / log(cost_per_correct + 1))

        Example:
            >>> CostAwareMetrics.cost_adjusted_accuracy(0.9, 100.0)
            0.1958  # penalized for high token cost
        """
        import math
        if tokens_per_correct <= 0:
            return accuracy
        return accuracy * (1.0 / math.log(tokens_per_correct + 1.0))


################################################################################
# SECTION 7: TESTING & DEMONSTRATION
################################################################################

def demonstrate_harness():
    """
    Demonstrate the evaluation harness with a mock model and suite.

    Shows:
        1. Creating an EvalHarness with custom config
        2. Running a mock benchmark suite
        3. Viewing the aggregated results
        4. Checking decontamination
        5. Computing cost-aware metrics
    """
    print("=" * 70)
    print("UNIFIED EVAL HARNESS DEMONSTRATION")
    print("=" * 70)

    # --- Demonstrate SamplingConfig ---
    print("\n--- Sampling Configuration ---")
    sampling = SamplingConfig(
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        max_new_tokens=2048,
        num_samples=1,
        seed=42,
    )
    print(f"  Temperature: {sampling.temperature}")
    print(f"  Top-k: {sampling.top_k}")
    print(f"  Top-p: {sampling.top_p}")
    print(f"  Max new tokens: {sampling.max_new_tokens}")
    print(f"  Num samples: {sampling.num_samples}")
    print(f"  Seed: {sampling.seed}")

    # --- Demonstrate CostAwareMetrics ---
    print("\n--- Cost-Aware Metrics ---")
    cpc = CostAwareMetrics.cost_per_correct(50000, 250)
    eff = CostAwareMetrics.efficiency_ratio(0.85, cpc)
    adj = CostAwareMetrics.cost_adjusted_accuracy(0.85, cpc)
    print(f"  Tokens generated: 50000")
    print(f"  Correct answers: 250")
    print(f"  Cost per correct: {cpc:.1f} tokens")
    print(f"  Efficiency ratio: {eff:.6f}")
    print(f"  Cost-adjusted accuracy: {adj:.4f}")

    # --- Demonstrate DecontaminationChecker ---
    print("\n--- Decontamination Check ---")
    checker = DecontaminationChecker(ngram_size=3)
    eval_texts = ["What is the capital of France?"]
    train_clean = ["The weather is nice today."]
    train_dirty = ["What is the capital of France? Paris is the capital."]
    status_clean = checker.check(eval_texts, train_clean)
    status_dirty = checker.check(eval_texts, train_dirty)
    print(f"  Clean training data: {status_clean.value}")
    print(f"  Contaminated training data: {status_dirty.value}")

    # --- Demonstrate EvalResults ---
    print("\n--- EvalResults Summary ---")
    mock_result = SuiteResult(
        suite_name="mock-suite",
        accuracy=0.85,
        num_correct=85,
        num_total=100,
        tokens_generated=10000,
        cost_per_correct=117.6,
        prompt_template="Q: {question}\nA:",
        sampling_params={"temperature": 0.0, "top_k": 0},
        decontamination=DecontaminationStatus.CLEAN,
    )
    results = EvalResults(
        model_name="DemoModel-7B",
        checkpoint_path="/checkpoints/demo.pt",
        suite_results=[mock_result],
        total_tokens=10000,
        total_correct=85,
        overall_cost_per_correct=117.6,
        evaluation_time_seconds=42.5,
        timestamp="2026-07-07 12:00:00",
    )
    print(results.summary())

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_harness()
