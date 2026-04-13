"""
################################################################################
LIVECODEBENCH — CODE GENERATION ON RECENT PROBLEMS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is LiveCodeBench?
    LiveCodeBench is a code generation benchmark that uses recent
    competitive programming problems. Unlike static benchmarks (HumanEval),
    LiveCodeBench continuously adds new problems, reducing contamination
    risk and providing a more current assessment of coding ability.

Why does it matter?
    Static code benchmarks like HumanEval and MBPP are saturated and
    heavily contaminated in training data. LiveCodeBench addresses both
    issues by using recent problems that are unlikely to be in training
    corpora. It tests genuine coding ability, not memorization.

How does it work?
    1. Load competitive programming problems with test cases
    2. Format as function signature + docstring → code
    3. Generate model response (code)
    4. Execute generated code against test cases
    5. Compute pass@k metric
    6. Report accuracy

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────┐
    │ LiveCodeBench Evaluation                                    │
    │                                                              │
    │  Problems ──▶ Format ──▶ Generate Code ──▶ Execute Tests    │
    │                                              ↓               │
    │                                        Pass / Fail           │
    │                                              ↓               │
    │                                        pass@k Metric         │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2021: HumanEval becomes standard code benchmark
    - 2023: HumanEval saturated; contamination concerns rise
    - 2024: LiveCodeBench introduced — continuous, contamination-resistant
    - 2025: LiveCodeBench adopted as primary code generation benchmark
    - 2026: Preferred over HumanEval for defensible evaluation numbers

INTERVIEW QUESTIONS:
    1. "Why is LiveCodeBench better than HumanEval?"
       HumanEval problems are static and widely available in training data.
       LiveCodeBench uses recent competitive programming problems that
       are continuously updated, making contamination much less likely.
       It also tests harder problems that better discriminate models.

    2. "What is pass@k and how do you compute it?"
       pass@k is the probability that at least one of k generated samples
       passes all test cases. It's computed as: pass@k = 1 - C(n-c, k)/C(n, k)
       where n is total samples and c is correct samples. This is unbiased
       when using multiple samples per problem.

    3. "How do you safely execute untrusted model-generated code?"
       Use sandboxing: Docker containers, seccomp profiles, or dedicated
       sandbox services. Set strict resource limits (CPU time, memory,
       network). Never execute untrusted code on the host system.

################################################################################
"""

import re
import time
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    class _StubModule:
        pass
    nn = type("nn", (), {"Module": _StubModule})()

import sys
sys.path.append('..')
sys.path.append('../..')
from ..harness import (
    BenchmarkSuite,
    EvalConfig,
    SuiteResult,
    DecontaminationStatus,
    DecontaminationChecker,
    CostAwareMetrics,
)


################################################################################
# SECTION 1: CONFIGURATION
################################################################################

@dataclass
class LiveCodeBenchConfig:
    """
    LiveCodeBench Configuration
    ===========================

    Controls LiveCodeBench-specific evaluation parameters.

    Attributes:
        release_version: Which release to use ('latest', 'v1', 'v2', etc.)
        num_samples: Number of samples per problem (for pass@k)
        max_new_tokens: Maximum tokens to generate per solution
        timeout_seconds: Execution timeout per test case
        prompt_format: Template for formatting problems
        language: Programming language to evaluate
    """
    release_version: str = "latest"
    num_samples: int = 20
    max_new_tokens: int = 4096
    timeout_seconds: int = 10
    prompt_format: str = (
        "You are an expert programmer. Solve the following problem.\n\n"
        "{problem_description}\n\n"
        "Function signature:\n```python\n{function_signature}\n```\n\n"
        "Write the complete function implementation:\n```python\n"
    )
    language: str = "python"


################################################################################
# SECTION 2: CODE EXECUTION ENGINE
################################################################################

class ExecutionResult(Enum):
    """Result of code execution against test cases."""
    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class TestCaseResult:
    """
    Result of a single test case execution.

    Attributes:
        test_id: Identifier of the test case
        result: Execution result (PASS/FAIL/TIMEOUT/ERROR)
        execution_time: Time taken in seconds
        error_message: Error message if execution failed
    """
    test_id: str
    result: ExecutionResult
    execution_time: float
    error_message: Optional[str] = None


class CodeExecutor:
    """
    Safe Code Executor
    ==================

    Executes model-generated code against test cases in a sandboxed
    environment. Handles timeouts, errors, and resource limits.

    Step by step:
        1. Prepare code with function implementation
        2. Add test case assertions
        3. Execute in sandboxed environment
        4. Capture output and errors
        5. Determine pass/fail status

    WHY this matters:
        Code evaluation requires actual execution — you cannot score
        code by text matching. The executor must be safe (untrusted code),
        fast (thousands of test cases), and reliable (no false positives).

    Interview Question:
        "How do you evaluate code generation safely?"
        Execute in isolated containers with resource limits. Use seccomp
        to restrict syscalls, cgroups for CPU/memory limits, and network
        namespaces to block network access. Set strict timeouts. Capture
        stdout/stderr for debugging failed tests.
    """

    def __init__(self, timeout_seconds: int = 10):
        """
        Initialize the code executor.

        Args:
            timeout_seconds: Maximum execution time per test case
        """
        self.timeout_seconds = timeout_seconds

    def execute(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
    ) -> List[TestCaseResult]:
        """
        Execute code against test cases.

        Args:
            code: The generated function implementation
            test_cases: List of test case dicts with 'input' and 'expected_output'

        Returns:
            List of TestCaseResult for each test case

        Explanation:
            In production, this would use subprocess or Docker to execute
            code in a sandbox. For this implementation, we provide the
            interface and safety patterns.

        Example:
            >>> executor = CodeExecutor(timeout_seconds=5)
            >>> results = executor.execute(
            ...     "def add(a, b): return a + b",
            ...     [{"input": "(1, 2)", "expected_output": "3"}],
            ... )
        """
        results = []
        for i, test in enumerate(test_cases):
            result = self._execute_single(code, test, f"test_{i}")
            results.append(result)
        return results

    def _execute_single(
        self,
        code: str,
        test: Dict[str, Any],
        test_id: str,
    ) -> TestCaseResult:
        """
        Execute a single test case.

        Args:
            code: Function implementation
            test: Test case dict
            test_id: Identifier for this test

        Returns:
            TestCaseResult with pass/fail status
        """
        start_time = time.time()

        # In production: execute in sandboxed subprocess
        # This shows the interface
        try:
            # Placeholder: would actually execute code here
            elapsed = time.time() - start_time
            return TestCaseResult(
                test_id=test_id,
                result=ExecutionResult.PASS,
                execution_time=elapsed,
            )
        except TimeoutError:
            return TestCaseResult(
                test_id=test_id,
                result=ExecutionResult.TIMEOUT,
                execution_time=self.timeout_seconds,
                error_message="Execution timed out",
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return TestCaseResult(
                test_id=test_id,
                result=ExecutionResult.ERROR,
                execution_time=elapsed,
                error_message=str(e),
            )


################################################################################
# SECTION 3: PASS@K COMPUTATION
################################################################################

def compute_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """
    Compute unbiased pass@k metric.

    Args:
        num_samples: Total number of samples generated
        num_correct: Number of samples that passed all tests
        k: k value for pass@k

    Returns:
        pass@k value (0.0 to 1.0)

    Formula:
        pass@k = 1 - C(n-c, k) / C(n, k)
        where n = num_samples, c = num_correct

    Explanation:
        This is the unbiased estimator from the Codex paper. It computes
        the probability that at least one of k randomly selected samples
        (without replacement) passes all test cases.

    Example:
        >>> compute_pass_at_k(20, 5, 1)
        0.25
        >>> compute_pass_at_k(20, 5, 10)
        0.941
    """
    if num_samples - num_correct < k:
        return 1.0

    # Use log-space to avoid overflow
    import math

    def log_comb(n: int, r: int) -> float:
        """Compute log of C(n, r)."""
        if r > n or r < 0:
            return float("-inf")
        return (math.lgamma(n + 1) - math.lgamma(r + 1)
                - math.lgamma(n - r + 1))

    log_prob_fail = log_comb(num_samples - num_correct, k) - log_comb(num_samples, k)
    return 1.0 - math.exp(log_prob_fail)


################################################################################
# SECTION 4: LIVECODEBENCH EVALUATOR
################################################################################

class LiveCodeBenchEvaluator:
    """
    LiveCodeBench: Code Generation on Recent Problems
    ==================================================

    Prefer this over static code benchmarks for defensible numbers.
    Less contaminated than HumanEval.
    Format: function signature + docstring → code
    Metric: pass@k (execute against test cases)

    Step by step:
        1. Load recent competitive programming problems
        2. Format as function signature + docstring
        3. Generate k code samples per problem
        4. Execute each sample against test cases
        5. Compute pass@k metric
        6. Report accuracy

    WHY this matters:
        LiveCodeBench tests genuine coding ability on problems unlikely
        to be in training data. The pass@k metric captures the model's
        ability to generate correct code, not just code that looks right.

    Interview Question:
        "How do you evaluate code generation objectively?"
        Use execution-based evaluation: generate code, run it against
        test cases, and check if output matches expected results. Use
        pass@k to account for the stochastic nature of generation.
        Prefer recent problems to avoid contamination.
    """

    # Class-level constants
    DEFAULT_NUM_SAMPLES: int = 20
    DEFAULT_K_VALUES: List[int] = field(default_factory=lambda: [1, 5, 10])

    def __init__(self, config: Optional[LiveCodeBenchConfig] = None):
        """
        Initialize LiveCodeBench evaluator.

        Args:
            config: LiveCodeBench-specific configuration
        """
        self.config = config or LiveCodeBenchConfig()
        self.executor = CodeExecutor(timeout_seconds=self.config.timeout_seconds)
        self._data: List[Dict[str, Any]] = []
        self._loaded = False

    @property
    def name(self) -> str:
        """Name of the benchmark suite."""
        return f"LiveCodeBench-{self.config.release_version}"

    @property
    def prompt_template(self) -> str:
        """Exact prompt template used."""
        return self.config.prompt_format

    def load_data(self, config: EvalConfig) -> List[Dict[str, Any]]:
        """
        Load LiveCodeBench dataset.

        Args:
            config: Evaluation configuration

        Returns:
            List of dicts with problem descriptions, signatures, and test cases
        """
        # Placeholder: in production, load from dataset
        self._data = self._load_problems()
        self._loaded = True
        return self._data

    def _load_problems(self) -> List[Dict[str, Any]]:
        """
        Load competitive programming problems.

        Returns:
            List of problem dicts
        """
        return [
            {
                "problem_id": f"lcb_{i}",
                "problem_description": f"Sample competitive programming problem {i}",
                "function_signature": "def solve(n: int, arr: List[int]) -> int:",
                "test_cases": [
                    {"input": "(5, [1,2,3,4,5])", "expected_output": "15"},
                    {"input": "(0, [])", "expected_output": "0"},
                ],
                "difficulty": "medium",
            }
            for i in range(100)
        ]

    def evaluate(
        self,
        model: nn.Module,
        config: EvalConfig,
    ) -> SuiteResult:
        """
        Run LiveCodeBench evaluation on the model.

        Args:
            model: The model to evaluate
            config: Evaluation configuration

        Returns:
            SuiteResult with pass@k and cost metrics

        Example:
            >>> evaluator = LiveCodeBenchEvaluator()
            >>> result = evaluator.evaluate(model, EvalConfig())
            >>> print(result.accuracy)  # pass@1
        """
        if not self._loaded:
            self.load_data(config)

        device = next(model.parameters()).device
        total_tokens = 0
        total_pass = 0
        total_problems = len(self._data)

        for item in self._data:
            prompt = self.config.prompt_format.format(
                problem_description=item["problem_description"],
                function_signature=item["function_signature"],
            )

            # Generate multiple samples for pass@k
            problem_pass = 0
            for _ in range(self.config.num_samples):
                response, tokens_used = self._generate_response(
                    model, prompt, config.sampling, device
                )
                total_tokens += tokens_used

                # Extract code from response
                code = self._extract_code(response, item["function_signature"])

                # Execute against test cases
                results = self.executor.execute(code, item["test_cases"])

                # Check if all tests pass
                if all(r.result == ExecutionResult.PASS for r in results):
                    problem_pass += 1

            if problem_pass > 0:
                total_pass += 1

        accuracy = total_pass / total_problems if total_problems > 0 else 0.0
        cost_per_correct = CostAwareMetrics.cost_per_correct(total_tokens, total_pass)

        return SuiteResult(
            suite_name=self.name,
            accuracy=accuracy,
            num_correct=total_pass,
            num_total=total_problems,
            tokens_generated=total_tokens,
            cost_per_correct=cost_per_correct,
            prompt_template=self.config.prompt_format,
            sampling_params={
                "temperature": config.sampling.temperature,
                "top_k": config.sampling.top_k,
                "top_p": config.sampling.top_p,
                "num_samples": self.config.num_samples,
            },
        )

    def _generate_response(
        self,
        model: nn.Module,
        prompt: str,
        sampling: Any,
        device: Any,
    ) -> Tuple[str, int]:
        """Generate a code response from the model."""
        # Placeholder: in production, use actual tokenizer and generate
        return "def solve(n, arr):\n    return sum(arr)", 50

    def _extract_code(self, response: str, function_signature: str) -> str:
        """
        Extract code from model response.

        Args:
            response: Raw model response
            function_signature: Expected function signature

        Returns:
            Extracted code string

        Explanation:
            Looks for code blocks (```python ... ```) or extracts
            the function implementation directly.
        """
        # Try to extract from code block
        code_block = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        # Fallback: use the whole response
        return response.strip()

    def check_decontamination(
        self,
        training_data: List[str],
    ) -> DecontaminationStatus:
        """
        Check for problem overlap in training data.

        Args:
            training_data: List of training document strings

        Returns:
            DecontaminationStatus indicating contamination level

        Explanation:
            Uses both n-gram overlap and problem ID matching.
            Recent problems are less likely to be contaminated.
        """
        checker = DecontaminationChecker(ngram_size=10)
        eval_texts = [item["problem_description"] for item in self._data]
        return checker.check(eval_texts, training_data)


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################

def demonstrate_livecodebench():
    """
    Demonstrate LiveCodeBench evaluation pipeline.

    Shows:
        1. Configuration
        2. Code extraction
        3. Pass@k computation
        4. Safe execution patterns
    """
    print("=" * 70)
    print("LIVECODEBENCH EVALUATION DEMONSTRATION")
    print("=" * 70)

    # --- Demonstrate Configuration ---
    print("\n--- Configuration ---")
    config = LiveCodeBenchConfig(
        release_version="latest",
        num_samples=20,
        timeout_seconds=10,
    )
    print(f"  Release: {config.release_version}")
    print(f"  Samples per problem: {config.num_samples}")
    print(f"  Timeout: {config.timeout_seconds}s")
    print(f"  Language: {config.language}")

    # --- Demonstrate Code Extraction ---
    print("\n--- Code Extraction ---")
    evaluator = LiveCodeBenchEvaluator(config)
    test_response = '''Here's my solution:

```python
def solve(n, arr):
    return sum(arr)
```

This sums all elements in the array.'''
    extracted = evaluator._extract_code(test_response, "def solve(n, arr):")
    print(f"  Extracted code:\n  {extracted}")

    # --- Demonstrate Pass@k ---
    print("\n--- Pass@k Computation ---")
    test_cases = [
        (20, 1, 1, "1 correct out of 20 samples, k=1"),
        (20, 5, 1, "5 correct out of 20 samples, k=1"),
        (20, 5, 5, "5 correct out of 20 samples, k=5"),
        (20, 5, 10, "5 correct out of 20 samples, k=10"),
        (20, 10, 1, "10 correct out of 20 samples, k=1"),
    ]
    for n, c, k, desc in test_cases:
        pass_k = compute_pass_at_k(n, c, k)
        print(f"  {desc}: pass@{k} = {pass_k:.4f}")

    # --- Demonstrate Safe Execution ---
    print("\n--- Safe Execution ---")
    executor = CodeExecutor(timeout_seconds=5)
    code = "def add(a, b): return a + b"
    tests = [
        {"input": "(1, 2)", "expected_output": "3"},
        {"input": "(0, 0)", "expected_output": "0"},
    ]
    results = executor.execute(code, tests)
    for r in results:
        print(f"  {r.test_id}: {r.result.value} ({r.execution_time:.4f}s)")

    # --- Demonstrate Evaluator ---
    print("\n--- Evaluator ---")
    print(f"  Name: {evaluator.name}")
    print(f"  Template: {evaluator.prompt_template[:80]}...")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_livecodebench()
