"""
################################################################################
FRONTIERMATH — RESEARCH-LEVEL MATHEMATICS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is FrontierMath?
    FrontierMath is a benchmark of research-level mathematics problems
    organized into four tiers of increasing difficulty: undergraduate,
    graduate, research, and open problems. Even frontier models struggle
    significantly at higher tiers, making it a useful ceiling check.

Why does it matter?
    FrontierMath tests mathematical reasoning at the highest levels.
    While AIME tests competition math, FrontierMath goes further with
    problems that require graduate-level knowledge and research-level
    creativity. It provides a long runway for improvement and helps
    identify the ceiling of current mathematical reasoning capabilities.

How does it work?
    1. Load problems from specified tiers
    2. Format as free-response math prompts
    3. Generate model response (typically with chain-of-thought)
    4. Score via exact match or symbolic equivalence
    5. Report per-tier and overall accuracy

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────┐
    │ FrontierMath Evaluation                                     │
    │                                                              │
    │  Tier 1 (Undergrad) ──┐                                    │
    │  Tier 2 (Graduate)  ──┼──▶ Format ──▶ Generate ──▶ Score   │
    │  Tier 3 (Research)  ──┤                          ↓          │
    │  Tier 4 (Open)      ──┘                   Exact/Symbolic    │
    │                                              ↓               │
    │                                        Per-Tier Accuracy     │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2024: FrontierMath introduced — research-level math benchmark
    - 2024: Frontier models score <2% on Tier 3-4 problems
    - 2025: Tier 1-2 become more tractable; Tier 3-4 remain extremely hard
    - 2026: Used as ceiling check for mathematical reasoning capabilities

INTERVIEW QUESTIONS:
    1. "How does FrontierMath differ from AIME or MATH?"
       AIME and MATH test competition-level math (high school to undergrad).
       FrontierMath goes to research level with problems requiring graduate
       knowledge and creative proof techniques. Tier 4 contains open
       problems that may not have known solutions.

    2. "Why use symbolic matching instead of exact string match?"
       Mathematical expressions can be written in multiple equivalent forms.
       Symbolic matching (using a CAS like SymPy) normalizes expressions
       to canonical form, allowing correct answers written differently
       to be recognized as equivalent.

    3. "What does it mean that Tier 4 has 'open problems'?"
       Some Tier 4 problems may not have known solutions. These test
       whether the model can make meaningful progress on unsolved
       problems, not just solve known exercises. Scoring may be
       partial or use expert evaluation.

################################################################################
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

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
class FrontierMathConfig:
    """
    FrontierMath Evaluation Configuration
    ======================================

    Controls FrontierMath-specific evaluation parameters.

    Attributes:
        tiers: Which tiers to evaluate (1-4)
        max_new_tokens: Maximum tokens to generate
        prompt_format: Template for formatting problems
        scoring_method: How to score ('exact', 'symbolic', 'hybrid')
        allow_partial_credit: Whether to give partial credit for close answers
    """
    tiers: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    max_new_tokens: int = 8192
    prompt_format: str = (
        "Solve the following {tier_name}-level mathematics problem. "
        "Show your work step by step, then give your final answer.\n\n"
        "Problem: {problem}\n\n"
        "Solution:"
    )
    scoring_method: str = "symbolic"
    allow_partial_credit: bool = False

    TIER_NAMES: Dict[int, str] = field(default_factory=lambda: {
        1: "undergraduate",
        2: "graduate",
        3: "research",
        4: "open problem",
    })


################################################################################
# SECTION 2: SYMBOLIC SCORING
################################################################################

class SymbolicScorer:
    """
    Symbolic Mathematics Scorer
    ============================

    Scores mathematical answers using symbolic computation.
    Handles equivalent expressions written in different forms.

    Step by step:
        1. Parse model answer and reference as symbolic expressions
        2. Simplify both expressions
        3. Check if the difference simplifies to zero
        4. If exact match fails, try numerical evaluation

    WHY this matters:
        Mathematical answers can be written in many equivalent forms.
        "1/2" and "0.5" and "sin(pi/6)" are all the same answer.
        Symbolic scoring normalizes expressions to catch these cases.

    Interview Question:
        "How do you compare mathematical expressions for equivalence?"
        Use a computer algebra system (CAS) like SymPy. Parse both
        expressions symbolically, simplify them to canonical form, and
        check if their difference is zero. For numerical answers,
        evaluate both to high precision and compare.
    """

    @staticmethod
    def parse_expression(expr_str: str):
        """
        Parse a string into a SymPy expression.

        Args:
            expr_str: String representation of a mathematical expression

        Returns:
            SymPy expression, or None if parsing fails or SymPy unavailable

        Example:
            >>> SymbolicScorer.parse_expression("x^2 + 2*x + 1")
            x**2 + 2*x + 1
        """
        if not SYMPY_AVAILABLE:
            return None
        try:
            # Clean up common notation
            expr_str = expr_str.strip()
            expr_str = expr_str.replace("^", "**")
            expr_str = expr_str.replace("\\times", "*")
            expr_str = expr_str.replace("\\cdot", "*")

            # Parse with SymPy
            expr = sympy.sympify(expr_str)
            return expr
        except (sympy.SympifyError, TypeError, ValueError):
            return None

    @staticmethod
    def symbolic_match(model_answer: str, reference: str) -> bool:
        """
        Check symbolic equivalence of two mathematical expressions.

        Args:
            model_answer: Model's answer string
            reference: Reference answer string

        Returns:
            True if expressions are symbolically equivalent

        Example:
            >>> SymbolicScorer.symbolic_match("x^2 + 1 + x^2", "2*x^2 + 1")
            True
        """
        model_expr = SymbolicScorer.parse_expression(model_answer)
        ref_expr = SymbolicScorer.parse_expression(reference)

        if model_expr is None or ref_expr is None:
            return False

        try:
            diff = sympy.simplify(model_expr - ref_expr)
            return diff == 0
        except Exception:
            return False

    @staticmethod
    def numerical_match(
        model_answer: str,
        reference: str,
        tolerance: float = 1e-10,
    ) -> bool:
        """
        Check numerical equivalence of two expressions.

        Args:
            model_answer: Model's answer string
            reference: Reference answer string
            tolerance: Numerical tolerance for comparison

        Returns:
            True if expressions evaluate to the same number

        Example:
            >>> SymbolicScorer.numerical_match("1/3", "0.3333333333333333")
            True
        """
        model_expr = SymbolicScorer.parse_expression(model_answer)
        ref_expr = SymbolicScorer.parse_expression(reference)

        if model_expr is None or ref_expr is None:
            return False

        try:
            model_val = float(model_expr.evalf())
            ref_val = float(ref_expr.evalf())
            return abs(model_val - ref_val) < tolerance
        except Exception:
            return False

    @classmethod
    def score(
        cls,
        model_answer: str,
        reference: str,
        method: str = "symbolic",
    ) -> float:
        """
        Score a mathematical answer.

        Args:
            model_answer: Model's answer string
            reference: Reference answer string
            method: Scoring method ('exact', 'symbolic', 'hybrid')

        Returns:
            Score (0.0 or 1.0)

        Example:
            >>> SymbolicScorer.score("2*x", "x + x", method="symbolic")
            1.0
        """
        if method == "exact":
            return 1.0 if model_answer.strip() == reference.strip() else 0.0
        elif method == "symbolic":
            if cls.symbolic_match(model_answer, reference):
                return 1.0
            return 1.0 if cls.numerical_match(model_answer, reference) else 0.0
        else:  # hybrid
            if model_answer.strip() == reference.strip():
                return 1.0
            if cls.symbolic_match(model_answer, reference):
                return 1.0
            return 1.0 if cls.numerical_match(model_answer, reference) else 0.0


################################################################################
# SECTION 3: ANSWER EXTRACTION
################################################################################

class FrontierMathAnswerExtractor:
    """
    FrontierMath Answer Extractor
    ==============================

    Extracts mathematical answers from model responses.
    Looks for explicit answer markers and boxed expressions.

    Step by step:
        1. Look for \\boxed{...} pattern
        2. Look for "answer is ..." pattern
        3. Look for "therefore ..." at end of response
        4. Fall back to last expression in response
    """

    # Pattern for \boxed{answer}
    BOXED_PATTERN: str = r"\\boxed\{(.*?)\}"

    # Pattern for "the answer is ..."
    ANSWER_IS_PATTERN: str = r"[Tt]he answer is\s*(.*?)(?:\.|$)"

    # Pattern for "therefore ..."
    THEREFORE_PATTERN: str = r"[Tt]herefore[,:]\s*(.*?)(?:\.|$)"

    @classmethod
    def extract(cls, response: str) -> Optional[str]:
        """
        Extract mathematical answer from model response.

        Args:
            response: Raw model response text

        Returns:
            Extracted answer string, or None if extraction fails

        Example:
            >>> FrontierMathAnswerExtractor.extract(
            ...     "Solving... \\boxed{x^2 + 1}"
            ... )
            'x^2 + 1'
        """
        # Strategy 1: \boxed{answer}
        match = re.search(cls.BOXED_PATTERN, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Strategy 2: "the answer is ..."
        match = re.search(cls.ANSWER_IS_PATTERN, response)
        if match:
            return match.group(1).strip()

        # Strategy 3: "therefore ..."
        match = re.search(cls.THEREFORE_PATTERN, response)
        if match:
            return match.group(1).strip()

        # Strategy 4: Last line with mathematical content
        lines = response.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and any(c in line for c in "0123456789xynmz^+-*/="):
                return line

        return None


################################################################################
# SECTION 4: FRONTIERMATH EVALUATOR
################################################################################

class FrontierMathEvaluator:
    """
    FrontierMath (Tiers 1-4): Research-Level Math
    ==============================================

    Extremely hard even for frontier models at higher tiers.
    Useful mainly as a ceiling check.

    Format: math problem → free response
    Metric: exact/symbolic match
    Tiers: 1 (undergrad), 2 (grad), 3 (research), 4 (open problems)

    Step by step:
        1. Load problems from specified tiers
        2. Format as free-response math prompts
        3. Generate model response with chain-of-thought
        4. Extract mathematical answer
        5. Score via symbolic matching
        6. Report per-tier and overall accuracy

    WHY this matters:
        FrontierMath tests the absolute ceiling of mathematical reasoning.
        While AIME and MATH are largely solved by frontier models,
        FrontierMath Tier 3-4 remains extremely challenging. It helps
        identify where current models plateau in mathematical ability.

    Interview Question:
        "What is the hardest math benchmark for LLMs?"
        FrontierMath Tier 3-4. These contain research-level and open
        problems that even specialists find challenging. Frontier models
        score <5% on Tier 3 and near 0% on Tier 4. This contrasts with
        AIME (~70%) and MATH (~90%) which are largely saturated.
    """

    # Class-level constants
    TIER_PROBLEM_COUNTS: Dict[int, int] = {
        1: 100,  # Undergraduate
        2: 100,  # Graduate
        3: 50,   # Research
        4: 25,   # Open problems
    }

    EXPECTED_FRONTIER_ACCURACY: Dict[int, float] = {
        1: 0.50,
        2: 0.20,
        3: 0.05,
        4: 0.01,
    }

    def __init__(self, config: Optional[FrontierMathConfig] = None):
        """
        Initialize FrontierMath evaluator.

        Args:
            config: FrontierMath-specific configuration
        """
        self.config = config or FrontierMathConfig()
        self.scorer = SymbolicScorer()
        self.extractor = FrontierMathAnswerExtractor()
        self._data: List[Dict[str, Any]] = []
        self._loaded = False

    @property
    def name(self) -> str:
        """Name of the benchmark suite."""
        tiers_str = "-".join(str(t) for t in self.config.tiers)
        return f"FrontierMath-T{tiers_str}"

    @property
    def prompt_template(self) -> str:
        """Exact prompt template used."""
        return self.config.prompt_format

    def load_data(self, config: EvalConfig) -> List[Dict[str, Any]]:
        """
        Load FrontierMath dataset.

        Args:
            config: Evaluation configuration

        Returns:
            List of dicts with 'problem', 'answer', 'tier', and metadata
        """
        self._data = []
        for tier in self.config.tiers:
            tier_data = self._load_tier(tier)
            self._data.extend(tier_data)
        self._loaded = True
        return self._data

    def _load_tier(self, tier: int) -> List[Dict[str, Any]]:
        """
        Load problems for a specific tier.

        Args:
            tier: Tier number (1-4)

        Returns:
            List of problem dicts
        """
        # Placeholder: in production, load from dataset
        expected_count = self.TIER_PROBLEM_COUNTS.get(tier, 50)
        tier_name = self.config.TIER_NAMES.get(tier, "unknown")

        return [
            {
                "problem": f"Sample {tier_name}-level problem {i}",
                "answer": "42",  # Symbolic answer
                "tier": tier,
                "tier_name": tier_name,
                "problem_id": f"fm_t{tier}_{i}",
            }
            for i in range(min(expected_count, 10))
        ]

    def evaluate(
        self,
        model: nn.Module,
        config: EvalConfig,
    ) -> SuiteResult:
        """
        Run FrontierMath evaluation on the model.

        Args:
            model: The model to evaluate
            config: Evaluation configuration

        Returns:
            SuiteResult with per-tier accuracy and cost metrics

        Example:
            >>> evaluator = FrontierMathEvaluator()
            >>> result = evaluator.evaluate(model, EvalConfig())
            >>> print(result.per_subject)
        """
        if not self._loaded:
            self.load_data(config)

        device = next(model.parameters()).device
        num_correct = 0
        num_total = 0
        total_tokens = 0
        tier_correct: Dict[int, int] = {}
        tier_total: Dict[int, int] = {}

        for item in self._data:
            tier = item["tier"]
            tier_name = item["tier_name"]

            prompt = self.config.prompt_format.format(
                tier_name=tier_name,
                problem=item["problem"],
            )

            response, tokens_used = self._generate_response(
                model, prompt, config.sampling, device
            )
            total_tokens += tokens_used

            # Extract answer
            extracted = self.extractor.extract(response)
            if extracted is None:
                extracted = response.strip()

            # Score
            score = self.scorer.score(
                extracted,
                item["answer"],
                method=self.config.scoring_method,
            )

            if score > 0:
                num_correct += 1
            num_total += 1

            tier_total[tier] = tier_total.get(tier, 0) + 1
            if score > 0:
                tier_correct[tier] = tier_correct.get(tier, 0) + 1

        # Compute per-tier accuracy
        per_tier_acc = {}
        for tier in tier_total:
            correct = tier_correct.get(tier, 0)
            total = tier_total[tier]
            tier_name = self.config.TIER_NAMES.get(tier, f"tier_{tier}")
            per_tier_acc[tier_name] = correct / total if total > 0 else 0.0

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
            per_subject=per_tier_acc,
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
        return "\\boxed{42}", 500

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
        """
        checker = DecontaminationChecker(ngram_size=10)
        eval_texts = [item["problem"] for item in self._data]
        return checker.check(eval_texts, training_data)


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################

def demonstrate_frontiermath():
    """
    Demonstrate FrontierMath evaluation pipeline.

    Shows:
        1. Configuration and tier system
        2. Symbolic scoring
        3. Answer extraction
        4. Per-tier accuracy expectations
    """
    print("=" * 70)
    print("FRONTIERMATH EVALUATION DEMONSTRATION")
    print("=" * 70)

    # --- Demonstrate Configuration ---
    print("\n--- Configuration ---")
    config = FrontierMathConfig(tiers=[1, 2, 3, 4])
    print(f"  Tiers: {config.tiers}")
    print(f"  Scoring method: {config.scoring_method}")
    for tier, name in config.TIER_NAMES.items():
        count = FrontierMathEvaluator.TIER_PROBLEM_COUNTS.get(tier, 0)
        expected = FrontierMathEvaluator.EXPECTED_FRONTIER_ACCURACY.get(tier, 0)
        print(f"  Tier {tier} ({name}): {count} problems, "
              f"expected ~{expected:.0%} accuracy")

    # --- Demonstrate Symbolic Scoring ---
    print("\n--- Symbolic Scoring ---")
    scorer = SymbolicScorer()
    test_cases = [
        ("2*x", "x + x", True),
        ("x^2 + 1", "1 + x^2", True),
        ("1/2", "0.5", True),
        ("sqrt(4)", "2", True),
        ("sin(pi/6)", "1/2", True),
        ("x^2", "x^3", False),
        ("pi", "3.14", False),  # Not exactly equal
    ]
    for model_ans, ref, expected in test_cases:
        result = scorer.symbolic_match(model_ans, ref)
        status = "OK" if result == expected else "FAIL"
        print(f"  [{status}] '{model_ans}' == '{ref}': {result}")

    # --- Demonstrate Answer Extraction ---
    print("\n--- Answer Extraction ---")
    extractor = FrontierMathAnswerExtractor()
    test_responses = [
        ("Solving... \\boxed{x^2 + 1}", "x^2 + 1"),
        ("The answer is 42.", "42"),
        ("Therefore, we get pi^2/6", "pi^2/6"),
    ]
    for response, expected in test_responses:
        extracted = extractor.extract(response)
        status = "OK" if extracted == expected else "FAIL"
        print(f"  [{status}] Extracted: '{extracted}' (expected '{expected}')")

    # --- Demonstrate Evaluator ---
    print("\n--- Evaluator ---")
    evaluator = FrontierMathEvaluator(config)
    print(f"  Name: {evaluator.name}")
    print(f"  Template: {evaluator.prompt_template[:80]}...")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_frontiermath()
