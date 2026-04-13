"""
################################################################################
HUMANITY'S LAST EXAM (HLE) — EXPERT-LEVEL ACADEMIC QUESTIONS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is HLE?
    Humanity's Last Exam (HLE) is a benchmark of expert-level closed-ended
    academic questions across 100+ subjects. Questions are sourced from
    domain experts (professors, researchers) and designed to be at the
    frontier of human knowledge. Frontier models score ~33% against ~90%
    human-expert accuracy.

Why does it matter?
    HLE is designed to be the "last exam" — questions so hard that even
    frontier models struggle. It provides a long runway for improvement
    and tests genuine expertise across a vast range of academic disciplines.
    Low absolute scores are expected and indicate the benchmark is working.

How does it work?
    1. Load expert-sourced academic questions (100+ subjects)
    2. Format as free-response prompts
    3. Generate model response
    4. Score via exact match or semantic equivalence
    5. Report accuracy (expect low numbers)

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────┐
    │ HLE Evaluation Pipeline                                     │
    │                                                              │
    │  Expert Questions (100+ subjects) ──▶ Format ──▶ Generate   │
    │                                                  ↓           │
    │                                    Score (exact/semantic)    │
    │                                                  ↓           │
    │                                    Accuracy (expect ~33%)    │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2024: HLE introduced — expert-level academic questions
    - 2024: Frontier models score ~25-33% vs ~90% human experts
    - 2025: HLE becomes key frontier benchmark for academic knowledge
    - 2026: Remains one of the hardest benchmarks; models slowly improving

INTERVIEW QUESTIONS:
    1. "Why are low scores expected on HLE?"
       HLE is designed to test the frontier of human knowledge. Questions
       are sourced from domain experts and are intentionally very hard.
       A score of 33% means the model can answer a third of expert-level
       questions — that's actually impressive for an AI system.

    2. "How do you score free-response answers on HLE?"
       Use a combination of exact match (for numerical/symbolic answers)
       and semantic equivalence (for text answers). Semantic scoring may
       use another LLM as a judge, comparing the model's answer to the
       reference answer for equivalence.

    3. "What subjects does HLE cover?"
       HLE covers 100+ subjects across STEM, humanities, social sciences,
       arts, and professional fields. Questions range from advanced
       mathematics to obscure historical facts to specialized medical
       knowledge.

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
class HLEConfig:
    """
    HLE Evaluation Configuration
    =============================

    Controls HLE-specific evaluation parameters.

    Attributes:
        num_shots: Number of few-shot examples (0 for zero-shot)
        max_new_tokens: Maximum tokens to generate
        prompt_format: Template for formatting questions
        scoring_method: How to score responses ('exact', 'semantic', 'hybrid')
        semantic_threshold: Threshold for semantic equivalence (0.0-1.0)
    """
    num_shots: int = 0
    max_new_tokens: int = 2048
    prompt_format: str = (
        "Answer the following expert-level academic question. "
        "Provide a concise, precise answer.\n\n"
        "Subject: {subject}\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    scoring_method: str = "hybrid"
    semantic_threshold: float = 0.85


################################################################################
# SECTION 2: ANSWER SCORING
################################################################################

class HLEScorer:
    """
    HLE Answer Scorer
    =================

    Scores model responses against reference answers.
    Supports exact match, semantic equivalence, and hybrid scoring.

    Step by step:
        1. Normalize model response and reference answer
        2. Try exact match first
        3. If no exact match, try semantic equivalence
        4. Return score (0.0 or 1.0)

    Interview Question:
        "How do you evaluate free-form answers objectively?"
        Use a combination of exact match (for unambiguous answers) and
        semantic equivalence (for text answers). Exact match handles
        numbers, formulas, and single-word answers. Semantic scoring
        uses embedding similarity or an LLM judge for longer answers.
    """

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """
        Normalize answer for comparison.

        Args:
            answer: Raw answer string

        Returns:
            Normalized answer string

        Explanation:
            Strips whitespace, converts to lowercase, removes punctuation,
            and standardizes formatting for fair comparison.
        """
        answer = answer.strip().lower()
        answer = re.sub(r"[^\w\s]", "", answer)
        answer = re.sub(r"\s+", " ", answer)
        return answer

    @staticmethod
    def exact_match(model_answer: str, reference: str) -> bool:
        """
        Check exact match between model answer and reference.

        Args:
            model_answer: Model's generated answer
            reference: Reference answer

        Returns:
            True if answers match exactly after normalization

        Example:
            >>> HLEScorer.exact_match("Paris", "paris")
            True
            >>> HLEScorer.exact_match("The answer is Paris", "Paris")
            False
        """
        norm_model = HLEScorer.normalize_answer(model_answer)
        norm_ref = HLEScorer.normalize_answer(reference)
        return norm_model == norm_ref

    @staticmethod
    def semantic_match(
        model_answer: str,
        reference: str,
        threshold: float = 0.85,
    ) -> bool:
        """
        Check semantic equivalence between model answer and reference.

        Args:
            model_answer: Model's generated answer
            reference: Reference answer
            threshold: Similarity threshold for match

        Returns:
            True if answers are semantically equivalent

        Explanation:
            In production, this would use an embedding model or LLM judge.
            For this implementation, we use a simple word overlap heuristic.
        """
        model_words = set(HLEScorer.normalize_answer(model_answer).split())
        ref_words = set(HLEScorer.normalize_answer(reference).split())

        if not ref_words:
            return not model_words

        overlap = len(model_words & ref_words)
        similarity = overlap / len(ref_words)
        return similarity >= threshold

    @classmethod
    def score(
        cls,
        model_answer: str,
        reference: str,
        method: str = "hybrid",
        threshold: float = 0.85,
    ) -> float:
        """
        Score a model answer against reference.

        Args:
            model_answer: Model's generated answer
            reference: Reference answer
            method: Scoring method ('exact', 'semantic', 'hybrid')
            threshold: Semantic similarity threshold

        Returns:
            Score (0.0 or 1.0)

        Example:
            >>> HLEScorer.score("Paris", "Paris", method="exact")
            1.0
        """
        if method == "exact":
            return 1.0 if cls.exact_match(model_answer, reference) else 0.0
        elif method == "semantic":
            return 1.0 if cls.semantic_match(model_answer, reference, threshold) else 0.0
        else:  # hybrid
            if cls.exact_match(model_answer, reference):
                return 1.0
            if cls.semantic_match(model_answer, reference, threshold):
                return 1.0
            return 0.0


################################################################################
# SECTION 3: HLE EVALUATOR
################################################################################

class HLEEvaluator:
    """
    Humanity's Last Exam (HLE)
    ==========================

    Expert-level closed-ended academic questions across 100+ subjects.
    Frontier models reach ~33% against ~90% human-expert accuracy.
    Expect low absolute numbers — that's the point.

    Format: question → free response
    Metric: accuracy (exact match or semantic equivalence)

    Step by step:
        1. Load HLE questions from 100+ subjects
        2. Format as free-response prompts
        3. Generate model response
        4. Score via exact match or semantic equivalence
        5. Report accuracy (expect ~33% for frontier models)

    WHY this matters:
        HLE tests the absolute frontier of academic knowledge. Unlike
        MMLU (saturated) or GPQA (narrow science focus), HLE covers
        the full breadth of human knowledge at expert difficulty.
        It provides a long runway for model improvement.

    Interview Question:
        "What is HLE and why is it important?"
        HLE is an expert-level academic benchmark covering 100+ subjects.
        Questions are sourced from domain experts and designed to be at
        the frontier of human knowledge. It's important because it
        provides a challenging, long-runway benchmark that won't be
        saturated soon — frontier models score only ~33%.
    """

    # Class-level constants
    EXPECTED_QUESTION_COUNT: int = 3000
    FRONTIER_MODEL_ACCURACY: float = 0.33
    HUMAN_EXPERT_ACCURACY: float = 0.90

    def __init__(self, config: Optional[HLEConfig] = None):
        """
        Initialize HLE evaluator.

        Args:
            config: HLE-specific configuration
        """
        self.config = config or HLEConfig()
        self.scorer = HLEScorer()
        self._data: List[Dict[str, Any]] = []
        self._loaded = False

    @property
    def name(self) -> str:
        """Name of the benchmark suite."""
        return "HLE"

    @property
    def prompt_template(self) -> str:
        """Exact prompt template used."""
        return self.config.prompt_format

    def load_data(self, config: EvalConfig) -> List[Dict[str, Any]]:
        """
        Load HLE dataset.

        Args:
            config: Evaluation configuration

        Returns:
            List of dicts with 'question', 'answer', 'subject', and metadata
        """
        # Placeholder: in production, load from dataset
        self._data = [
            {
                "question": f"Sample expert-level question from subject {i % 50}",
                "answer": "Sample answer",
                "subject": f"subject_{i % 50}",
                "difficulty": "expert",
            }
            for i in range(min(self.EXPECTED_QUESTION_COUNT, 100))
        ]
        self._loaded = True
        return self._data

    def evaluate(
        self,
        model: nn.Module,
        config: EvalConfig,
    ) -> SuiteResult:
        """
        Run HLE evaluation on the model.

        Args:
            model: The model to evaluate
            config: Evaluation configuration

        Returns:
            SuiteResult with accuracy and cost metrics

        Example:
            >>> evaluator = HLEEvaluator()
            >>> result = evaluator.evaluate(model, EvalConfig())
            >>> print(result.accuracy)  # expect ~0.33
        """
        if not self._loaded:
            self.load_data(config)

        device = next(model.parameters()).device
        num_correct = 0
        num_total = 0
        total_tokens = 0
        per_subject_correct: Dict[str, int] = {}
        per_subject_total: Dict[str, int] = {}

        for item in self._data:
            prompt = self.config.prompt_format.format(
                subject=item["subject"],
                question=item["question"],
            )

            response, tokens_used = self._generate_response(
                model, prompt, config.sampling, device
            )
            total_tokens += tokens_used

            score = self.scorer.score(
                response,
                item["answer"],
                method=self.config.scoring_method,
                threshold=self.config.semantic_threshold,
            )

            if score > 0:
                num_correct += 1
            num_total += 1

            subject = item["subject"]
            per_subject_total[subject] = per_subject_total.get(subject, 0) + 1
            if score > 0:
                per_subject_correct[subject] = per_subject_correct.get(subject, 0) + 1

        # Compute per-subject accuracy
        per_subject_acc = {}
        for subject in per_subject_total:
            correct = per_subject_correct.get(subject, 0)
            total = per_subject_total[subject]
            per_subject_acc[subject] = correct / total if total > 0 else 0.0

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
            per_subject=per_subject_acc,
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
        return "Sample answer", 100

    def check_decontamination(
        self,
        training_data: List[str],
    ) -> DecontaminationStatus:
        """
        Check for question overlap in training data.

        Args:
            training_data: List of training document strings

        Returns:
            DecontaminationStatus indicating contamination level
        """
        checker = DecontaminationChecker(ngram_size=13)
        eval_texts = [item["question"] for item in self._data]
        return checker.check(eval_texts, training_data)


################################################################################
# SECTION 4: TESTING & DEMONSTRATION
################################################################################

def demonstrate_hle():
    """
    Demonstrate HLE evaluation pipeline.

    Shows:
        1. Configuration
        2. Answer scoring (exact and semantic)
        3. Expected score ranges
        4. Per-subject tracking
    """
    print("=" * 70)
    print("HUMANITY'S LAST EXAM (HLE) DEMONSTRATION")
    print("=" * 70)

    # --- Demonstrate Configuration ---
    print("\n--- Configuration ---")
    config = HLEConfig(
        scoring_method="hybrid",
        semantic_threshold=0.85,
    )
    print(f"  Scoring method: {config.scoring_method}")
    print(f"  Semantic threshold: {config.semantic_threshold}")
    print(f"  Expected questions: {HLEEvaluator.EXPECTED_QUESTION_COUNT}")

    # --- Demonstrate Scoring ---
    print("\n--- Answer Scoring ---")
    scorer = HLEScorer()
    test_cases = [
        ("Paris", "Paris", "exact"),
        ("paris", "Paris", "exact (case insensitive)"),
        ("The capital of France is Paris", "Paris", "semantic"),
        ("London", "Paris", "no match"),
        ("42", "42", "exact (number)"),
        ("approximately 42", "42", "semantic"),
    ]
    for model_ans, ref, desc in test_cases:
        exact = scorer.exact_match(model_ans, ref)
        semantic = scorer.semantic_match(model_ans, ref, 0.85)
        print(f"  '{model_ans}' vs '{ref}' ({desc})")
        print(f"    Exact: {exact}, Semantic: {semantic}")

    # --- Demonstrate Score Ranges ---
    print("\n--- Expected Score Ranges ---")
    print(f"  Frontier model accuracy: ~{HLEEvaluator.FRONTIER_MODEL_ACCURACY:.0%}")
    print(f"  Human expert accuracy: ~{HLEEvaluator.HUMAN_EXPERT_ACCURACY:.0%}")
    print(f"  Gap to close: {HLEEvaluator.HUMAN_EXPERT_ACCURACY - HLEEvaluator.FRONTIER_MODEL_ACCURACY:.0%}")

    # --- Demonstrate Evaluator ---
    print("\n--- Evaluator ---")
    evaluator = HLEEvaluator(config)
    print(f"  Name: {evaluator.name}")
    print(f"  Template: {evaluator.prompt_template[:80]}...")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_hle()
