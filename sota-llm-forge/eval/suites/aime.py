"""
################################################################################
AIME — COMPETITION MATHEMATICS EVALUATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is AIME?
    AIME (American Invitational Mathematics Examination) is a high-school
    math competition with 15 problems per exam, each requiring an integer
    answer between 0 and 999. It sits between AMC (easier) and USAMO
    (harder) in difficulty.

Why does it matter?
    AIME problems test mathematical reasoning that goes beyond pattern
    matching. They require multi-step problem solving, algebraic
    manipulation, and creative insight. AIME has become a key benchmark
    for evaluating mathematical reasoning in frontier LLMs.

How does it work?
    1. Load AIME problems from a specific year (or multiple years)
    2. Format each as a free-response prompt
    3. Generate model response (typically with chain-of-thought)
    4. Extract the integer answer from the response
    5. Compare to ground truth (exact integer match)
    6. Report accuracy

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────┐
    │ AIME Evaluation Pipeline                                    │
    │                                                              │
    │  AIME Problems ──▶ Format ──▶ Generate (CoT) ──▶ Extract    │
    │                                              ↓               │
    │                                      Integer Match (0-999)    │
    │                                              ↓               │
    │                                        Accuracy               │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 1983: AIME first administered by MAA
    - 2023: AIME problems used to evaluate GPT-4 math capabilities
    - 2024: Frontier models reach 50-70% on AIME
    - 2025: AIME becomes standard math benchmark for LLM evaluation
    - 2026: Contamination concern — AIME problems circulate widely after release

INTERVIEW QUESTIONS:
    1. "Why use AIME over GSM8K for math evaluation?"
       GSM8K is largely saturated by frontier models (>95%). AIME problems
       are harder, require genuine mathematical reasoning, and better
       discriminate between frontier models. The integer answer format
       also makes scoring objective.

    2. "How do you handle AIME contamination?"
       AIME problems are publicly available shortly after each exam.
       Use the most recent AIME (current year) and check for n-gram
       overlap with training data. Some evaluators use AIME II (the
       harder variant) for better discrimination.

    3. "What's the best prompting strategy for AIME?"
       Chain-of-thought is essential. The model should show its work:
       define variables, set up equations, solve step by step, and
       box the final answer. This typically improves accuracy by
       15-25% over direct answering.

################################################################################
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

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
class AIMEConfig:
    """
    AIME Evaluation Configuration
    ==============================

    Controls AIME-specific evaluation parameters.

    Attributes:
        year: AIME year to evaluate (e.g., 2025, 2026)
        exam: Which exam ('I', 'II', or 'both')
        num_samples: Number of samples per problem (for pass@k)
        max_new_tokens: Maximum tokens to generate (needs room for CoT)
        prompt_format: Template for formatting problems
        require_boxed_answer: Whether to require \\boxed{} format
    """
    year: int = 2026
    exam: str = "I"
    num_samples: int = 1
    max_new_tokens: int = 4096
    prompt_format: str = (
        "Solve the following AIME problem step by step. "
        "Give your final answer as an integer between 0 and 999, "
        "enclosed in \\boxed{{}}.\n\n"
        "Problem: {problem}\n\n"
        "Solution:"
    )
    require_boxed_answer: bool = True

    @property
    def num_problems(self) -> int:
        """Number of problems per AIME exam."""
        return 15


################################################################################
# SECTION 2: ANSWER EXTRACTION
################################################################################

class AIMEAnswerExtractor:
    """
    AIME Answer Extractor
    =====================

    Extracts integer answers from model responses.
    AIME answers are integers between 0 and 999 inclusive.

    Step by step:
        1. Look for \\boxed{N} pattern (standard math format)
        2. Look for "answer is N" pattern
        3. Look for last integer in the response
        4. Validate range (0-999)

    Interview Question:
        "How do you extract structured answers from free-form text?"
        Use multiple regex patterns in priority order: (1) explicit
        formatting like \\boxed{}, (2) natural language patterns like
        "the answer is", (3) fallback to last number. Always validate
        the extracted value against expected constraints.
    """

    # Pattern for \boxed{answer}
    BOXED_PATTERN: str = r"\\boxed\{(\d{1,3})\}"

    # Pattern for "the answer is N"
    ANSWER_IS_PATTERN: str = r"[Tt]he answer is\s*(\d{1,3})"

    # Pattern for any integer at end of text
    LAST_INT_PATTERN: str = r"(\d{1,3})\s*$"

    # Valid answer range
    MIN_ANSWER: int = 0
    MAX_ANSWER: int = 999

    @classmethod
    def extract(cls, response: str) -> Optional[int]:
        """
        Extract integer answer from model response.

        Args:
            response: Raw model response text

        Returns:
            Integer answer (0-999), or None if extraction fails

        Explanation:
            Tries multiple extraction strategies in priority order.
            Returns the first valid integer found.

        Example:
            >>> AIMEAnswerExtractor.extract("...therefore \\boxed{42}")
            42
            >>> AIMEAnswerExtractor.extract("The answer is 128")
            128
        """
        # Strategy 1: \boxed{N}
        match = re.search(cls.BOXED_PATTERN, response)
        if match:
            answer = int(match.group(1))
            if cls.MIN_ANSWER <= answer <= cls.MAX_ANSWER:
                return answer

        # Strategy 2: "the answer is N"
        match = re.search(cls.ANSWER_IS_PATTERN, response)
        if match:
            answer = int(match.group(1))
            if cls.MIN_ANSWER <= answer <= cls.MAX_ANSWER:
                return answer

        # Strategy 3: Last integer in text
        match = re.search(cls.LAST_INT_PATTERN, response, re.MULTILINE)
        if match:
            answer = int(match.group(1))
            if cls.MIN_ANSWER <= answer <= cls.MAX_ANSWER:
                return answer

        return None


################################################################################
# SECTION 3: AIME EVALUATOR
################################################################################

class AIMEEvaluator:
    """
    AIME (Current Year) Evaluation
    ===============================

    Competition math problems.
    Watch for contamination — AIME problems circulate widely after release.
    Format: free-response integer answer (0-999)
    Metric: exact match accuracy

    Step by step:
        1. Load AIME problems for specified year and exam
        2. Format each as a free-response prompt
        3. Generate model response with chain-of-thought
        4. Extract integer answer from response
        5. Compare to ground truth (exact match)
        6. Report accuracy

    WHY this matters:
        AIME tests genuine mathematical problem-solving ability. Unlike
        multiple-choice benchmarks, there's no guessing advantage.
        The integer answer format makes scoring objective and unambiguous.

    Interview Question:
        "Why is AIME a good benchmark for math reasoning?"
        AIME requires multi-step reasoning, creative problem-solving,
        and exact answers (no partial credit). The 0-999 range means
        random guessing has only 0.1% chance of success. Problems are
        from actual competitions, ensuring quality and difficulty.
    """

    # Class-level constants
    DEFAULT_YEAR: int = 2026
    DEFAULT_EXAM: str = "I"
    PROBLEMS_PER_EXAM: int = 15

    def __init__(self, config: Optional[AIMEConfig] = None):
        """
        Initialize AIME evaluator.

        Args:
            config: AIME-specific configuration (uses defaults if None)
        """
        self.config = config or AIMEConfig()
        self.extractor = AIMEAnswerExtractor()
        self._data: List[Dict[str, Any]] = []
        self._loaded = False

    @property
    def name(self) -> str:
        """Name of the benchmark suite."""
        return f"AIME-{self.config.year}-{self.config.exam}"

    @property
    def prompt_template(self) -> str:
        """Exact prompt template used."""
        return self.config.prompt_format

    def load_data(self, config: EvalConfig) -> List[Dict[str, Any]]:
        """
        Load AIME dataset.

        Args:
            config: Evaluation configuration

        Returns:
            List of dicts with 'prompt', 'answer', and metadata

        Explanation:
            Loads AIME problems from the specified year and exam.
            In production, this would fetch from a dataset or API.
        """
        if self.config.exam == "both":
            self._data = (
                self._load_exam("I") + self._load_exam("II")
            )
        else:
            self._data = self._load_exam(self.config.exam)
        self._loaded = True
        return self._data

    def _load_exam(self, exam: str) -> List[Dict[str, Any]]:
        """
        Load problems for a single AIME exam.

        Args:
            exam: Exam identifier ('I' or 'II')

        Returns:
            List of problem dicts
        """
        # Placeholder: in production, load from actual dataset
        return [
            {
                "problem": f"AIME {self.config.year} {exam} Problem {i+1}: "
                           f"Sample competition math problem.",
                "answer": 42,  # Integer answer 0-999
                "problem_number": i + 1,
                "exam": exam,
                "year": self.config.year,
            }
            for i in range(self.PROBLEMS_PER_EXAM)
        ]

    def evaluate(
        self,
        model: nn.Module,
        config: EvalConfig,
    ) -> SuiteResult:
        """
        Run AIME evaluation on the model.

        Args:
            model: The model to evaluate
            config: Evaluation configuration

        Returns:
            SuiteResult with accuracy and cost metrics

        Example:
            >>> evaluator = AIMEEvaluator(AIMEConfig(year=2026))
            >>> result = evaluator.evaluate(model, EvalConfig())
            >>> print(result.accuracy)
        """
        if not self._loaded:
            self.load_data(config)

        device = next(model.parameters()).device
        num_correct = 0
        num_total = 0
        total_tokens = 0

        for item in self._data:
            prompt = self.config.prompt_format.format(problem=item["problem"])

            response, tokens_used = self._generate_response(
                model, prompt, config.sampling, device
            )
            total_tokens += tokens_used

            predicted = self.extractor.extract(response)
            is_correct = predicted == item["answer"]

            if is_correct:
                num_correct += 1
            num_total += 1

        accuracy = num_correct / num_total if num_total > 0 else 0.0
        cost_per_correct = CostAwareMetrics.cost_per_correct(total_tokens, num_correct)

        return SuiteResult(
            suite_name=self.name,
            accuracy=accuracy,
            num_correct=num_correct,
            num_total=num_total,
            tokens_generated=total_tokens,
            cost_per_correct=cost_per_correct,
            prompt_template=self.config.prompt_format,
            sampling_params={
                "temperature": config.sampling.temperature,
                "top_k": config.sampling.top_k,
                "top_p": config.sampling.top_p,
            },
        )

    def _generate_response(
        self,
        model: nn.Module,
        prompt: str,
        sampling: Any,
        device: Any,
    ) -> Tuple[str, int]:
        """Generate a response from the model."""
        # Placeholder: in production, use actual tokenizer and generate
        return "Let me solve this step by step... \\boxed{42}", 200

    def check_decontamination(
        self,
        training_data: List[str],
    ) -> DecontaminationStatus:
        """
        Check for AIME problem overlap in training data.

        Args:
            training_data: List of training document strings

        Returns:
            DecontaminationStatus indicating contamination level

        Explanation:
            AIME problems are widely circulated after each exam.
            Contamination is a serious concern. Use shorter n-grams
            (8-grams) to catch partial problem reuse.
        """
        checker = DecontaminationChecker(ngram_size=8)
        eval_texts = [item["problem"] for item in self._data]
        return checker.check(eval_texts, training_data)


################################################################################
# SECTION 4: TESTING & DEMONSTRATION
################################################################################

def demonstrate_aime():
    """
    Demonstrate AIME evaluation pipeline.

    Shows:
        1. AIME configuration
        2. Answer extraction from various formats
        3. Problem formatting
        4. Contamination checking
    """
    print("=" * 70)
    print("AIME EVALUATION DEMONSTRATION")
    print("=" * 70)

    # --- Demonstrate Configuration ---
    print("\n--- Configuration ---")
    config = AIMEConfig(year=2026, exam="I")
    print(f"  Year: {config.year}")
    print(f"  Exam: {config.exam}")
    print(f"  Problems per exam: {config.num_problems}")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Require boxed: {config.require_boxed_answer}")

    # --- Demonstrate Answer Extraction ---
    print("\n--- Answer Extraction ---")
    extractor = AIMEAnswerExtractor()
    test_cases = [
        ("...therefore \\boxed{42}", 42),
        ("The answer is 128.", 128),
        ("We get 256 at the end", 256),
        ("\\boxed{999}", 999),
        ("\\boxed{0}", 0),
        ("No number here", None),
        ("\\boxed{1000}", None),  # Out of range
    ]
    for response, expected in test_cases:
        extracted = extractor.extract(response)
        status = "OK" if extracted == expected else "FAIL"
        print(f"  [{status}] '{response[:40]}...' -> {extracted} (expected {expected})")

    # --- Demonstrate Prompt Formatting ---
    print("\n--- Prompt Formatting ---")
    prompt = config.prompt_format.format(
        problem="Find the number of positive integers n <= 1000 "
                "such that n^2 + n + 1 is divisible by 7."
    )
    print(f"  Prompt preview:")
    print(f"  {prompt[:200]}...")

    # --- Demonstrate Evaluator ---
    print("\n--- Evaluator ---")
    evaluator = AIMEEvaluator(config)
    print(f"  Name: {evaluator.name}")
    print(f"  Template: {evaluator.prompt_template[:80]}...")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_aime()
