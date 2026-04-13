"""
################################################################################
MMLU / MMLU-PRO — BROAD KNOWLEDGE MULTIPLE CHOICE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is MMLU?
    MMLU (Massive Multitask Language Understanding) is a benchmark of
    57 subjects spanning STEM, humanities, social sciences, and more.
    Each question is multiple choice with 4 options (A/B/C/D).
    MMLU-Pro is a harder variant with 10 options and more reasoning-
    intensive questions.

Why does it matter?
    MMLU was the gold standard for broad knowledge evaluation from 2021
    to 2024. By 2025-2026, frontier models exceed 88% on MMLU, making
    it a sanity floor rather than a headline metric. MMLU-Pro remains
    more discriminating with its harder questions and more options.

How does it work?
    1. Load questions from 57 subjects (MMLU) or 14 subjects (MMLU-Pro)
    2. Format each as: question + choices + "Answer:"
    3. Generate model response
    4. Extract predicted letter (A/B/C/D or A-J for MMLU-Pro)
    5. Compare to ground truth
    6. Report per-subject and overall accuracy

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────┐
    │ MMLU Evaluation Pipeline                                    │
    │                                                              │
    │  Dataset ──▶ Format Prompts ──▶ Generate ──▶ Extract Answer │
    │                                                  ↓           │
    │                                           Compare to GT      │
    │                                                  ↓           │
    │                                        Per-Subject Accuracy   │
    │                                                  ↓           │
    │                                        Overall Accuracy       │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2021: MMLU introduced (Hendrycks et al.) — 57 subjects, 14K questions
    - 2022: GPT-3.5 reaches ~70% on MMLU
    - 2023: GPT-4 reaches ~86% on MMLU
    - 2024: MMLU-Pro introduced — harder, 10 options, more reasoning
    - 2025: Frontier models exceed 88% on MMLU; MMLU-Pro becomes primary
    - 2026: MMLU treated as sanity check; MMLU-Pro still discriminating

INTERVIEW QUESTIONS:
    1. "Why is MMLU no longer a good headline metric?"
       Frontier models exceed 88% on MMLU, creating a compressed range
       at the top. Small differences (88% vs 90%) don't reliably indicate
       better capability. MMLU-Pro, with 10 options and harder questions,
       provides more discrimination at the frontier.

    2. "How do you evaluate multiple-choice without logit access?"
       Use letter extraction: generate text and parse the predicted answer
       letter. If logit access is available, compute P(A), P(B), P(C), P(D)
       directly and pick the highest probability option.

    3. "What are the limitations of multiple-choice benchmarks?"
       Multiple choice allows guessing (25% random for 4 options), doesn't
       test generation ability, and can be gamed by pattern matching.
       Free-response benchmarks (like AIME) better test true capability.

################################################################################
"""

import re
import json
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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
class MMLUConfig:
    """
    MMLU Evaluation Configuration
    ==============================

    Controls MMLU-specific evaluation parameters.

    Attributes:
        variant: Which MMLU variant to use ('mmlu' or 'mmlu-pro')
        num_shots: Number of few-shot examples per subject
        subjects: List of subjects to evaluate (None = all)
        choice_letters: Valid answer letters
        prompt_format: Template for formatting questions
    """
    variant: str = "mmlu"
    num_shots: int = 5
    subjects: Optional[List[str]] = None
    choice_letters: List[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    prompt_format: str = (
        "The following are multiple choice questions about {subject}.\n\n"
        "{examples}\n"
        "{question}\n"
        "{choices}\n"
        "Answer:"
    )

    @property
    def num_choices(self) -> int:
        """Number of answer choices."""
        return len(self.choice_letters)


################################################################################
# SECTION 2: MMLU SUBJECT DEFINITIONS
################################################################################

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

MMLU_PRO_SUBJECTS = [
    "biology", "business", "chemistry", "computer_science",
    "economics", "engineering", "health", "history",
    "law", "math", "other", "philosophy", "physics", "psychology",
]


################################################################################
# SECTION 3: PROMPT FORMATTER
################################################################################

class MMLUPromptFormatter:
    """
    MMLU Prompt Formatter
    =====================

    Formats MMLU questions into the standard prompt template.
    Handles few-shot example insertion and subject-specific prefixes.

    Step by step:
        1. Build subject header with subject name
        2. Insert few-shot examples (if available)
        3. Format the target question with choices
        4. Append "Answer:" for the model to complete

    Interview Question:
        "How does prompt formatting affect MMLU scores?"
        Prompt format significantly impacts scores. Using the exact format
        from the original paper (subject header + examples + question)
        typically gives best results. Mismatches between training and
        eval prompt formats can cause 2-5% accuracy drops.
    """

    @staticmethod
    def format_question(
        question: str,
        choices: List[str],
        subject: str,
        examples: List[Dict[str, Any]],
        config: MMLUConfig,
    ) -> str:
        """
        Format a single MMLU question into a prompt.

        Args:
            question: The question text
            choices: List of choice texts
            subject: Subject name (e.g., 'abstract_algebra')
            examples: List of few-shot example dicts with 'question', 'choices', 'answer'
            config: MMLU configuration

        Returns:
            Formatted prompt string ready for model input

        Example:
            >>> formatter = MMLUPromptFormatter()
            >>> prompt = formatter.format_question(
            ...     "What is 2+2?",
            ...     ["2", "3", "4", "5"],
            ...     "math",
            ...     [],
            ...     MMLUConfig(),
            ... )
            >>> "Answer:" in prompt
            True
        """
        # Format few-shot examples
        example_str = ""
        for ex in examples[:config.num_shots]:
            ex_choices = "\n".join(
                f"{config.choice_letters[i]}. {c}"
                for i, c in enumerate(ex["choices"])
            )
            example_str += f"{ex['question']}\n{ex_choices}\nAnswer: {ex['answer']}\n\n"

        # Format target question
        choice_str = "\n".join(
            f"{config.choice_letters[i]}. {c}"
            for i, c in enumerate(choices)
        )

        # Build full prompt
        prompt = config.prompt_format.format(
            subject=subject.replace("_", " "),
            examples=example_str,
            question=question,
            choices=choice_str,
        )

        return prompt

    @staticmethod
    def extract_answer(response: str, valid_letters: List[str]) -> Optional[str]:
        """
        Extract the predicted answer letter from model response.

        Args:
            response: Raw model response text
            valid_letters: List of valid answer letters

        Returns:
            Predicted answer letter, or None if extraction fails

        Explanation:
            Tries multiple extraction strategies:
            1. First character if it's a valid letter
            2. First occurrence of a valid letter in the text
            3. Pattern matching for "The answer is X"

        Example:
            >>> MMLUPromptFormatter.extract_answer("A", ["A", "B", "C", "D"])
            'A'
            >>> MMLUPromptFormatter.extract_answer("The answer is B.", ["A", "B", "C", "D"])
            'B'
        """
        response = response.strip()

        # Strategy 1: First character
        if response and response[0].upper() in valid_letters:
            return response[0].upper()

        # Strategy 2: Pattern "The answer is X"
        pattern = r"[Tt]he answer is\s*([A-J])"
        match = re.search(pattern, response)
        if match and match.group(1).upper() in valid_letters:
            return match.group(1).upper()

        # Strategy 3: First valid letter anywhere
        for char in response:
            if char.upper() in valid_letters:
                return char.upper()

        return None


################################################################################
# SECTION 4: MMLU EVALUATOR
################################################################################

class MMLUEvaluator:
    """
    MMLU / MMLU-Pro Evaluation
    ===========================

    Broad knowledge, multiple choice (57 subjects).
    2026 status: MMLU itself is saturated (>88% frontier). Treat as
    sanity floor, not headline metric. MMLU-Pro is harder, more
    discriminating.

    Format: question + 4 choices → A/B/C/D (MMLU) or A-J (MMLU-Pro)
    Metric: accuracy

    Step by step:
        1. Load MMLU dataset (all 57 subjects or subset)
        2. For each question, format with few-shot examples
        3. Generate model response
        4. Extract predicted answer letter
        5. Compare to ground truth
        6. Aggregate per-subject and overall accuracy

    WHY this matters:
        MMLU provides a broad coverage test across academic domains.
        While saturated at the frontier, it remains a useful sanity check
        for knowledge breadth. MMLU-Pro extends this with harder questions
        that better discriminate frontier models.

    Interview Question:
        "How would you evaluate an LLM's broad knowledge?"
        Use MMLU for 57-subject coverage. For frontier models, prefer
        MMLU-Pro which has 10 options and harder questions. Always report
        per-subject breakdowns to identify knowledge gaps. Use few-shot
        prompting (typically 5 shots) for consistency.
    """

    # Class-level constants
    DEFAULT_NUM_SHOTS: int = 5
    DEFAULT_MAX_NEW_TOKENS: int = 1
    ANSWER_PATTERN: str = r"^([A-J])"

    def __init__(self, config: Optional[MMLUConfig] = None):
        """
        Initialize MMLU evaluator.

        Args:
            config: MMLU-specific configuration (uses defaults if None)

        Example:
            >>> evaluator = MMLUEvaluator(MMLUConfig(variant="mmlu-pro"))
        """
        self.config = config or MMLUConfig()
        self.formatter = MMLUPromptFormatter()
        self._data: List[Dict[str, Any]] = []
        self._loaded = False

    @property
    def name(self) -> str:
        """Name of the benchmark suite."""
        return f"MMLU-{'Pro' if self.config.variant == 'mmlu-pro' else 'Standard'}"

    @property
    def prompt_template(self) -> str:
        """Exact prompt template used."""
        return self.config.prompt_format

    def load_data(self, config: EvalConfig) -> List[Dict[str, Any]]:
        """
        Load MMLU dataset and format prompts.

        Args:
            config: Evaluation configuration

        Returns:
            List of dicts with 'prompt', 'answer', 'subject', and metadata

        Explanation:
            In a real implementation, this would load from HuggingFace
            datasets or local files. For this implementation, we provide
            the loading structure and sample data format.

        Example:
            >>> evaluator = MMLUEvaluator()
            >>> data = evaluator.load_data(EvalConfig())
            >>> print(len(data))
        """
        subjects = self.config.subjects or (
            MMLU_PRO_SUBJECTS if self.config.variant == "mmlu-pro"
            else MMLU_SUBJECTS
        )

        # In production, load from dataset files
        # This shows the expected data structure
        self._data = []
        for subject in subjects:
            questions = self._load_subject(subject)
            self._data.extend(questions)

        self._loaded = True
        return self._data

    def _load_subject(self, subject: str) -> List[Dict[str, Any]]:
        """
        Load questions for a single subject.

        Args:
            subject: Subject identifier

        Returns:
            List of question dicts for the subject

        Explanation:
            Each question dict contains:
            - 'question': question text
            - 'choices': list of choice texts
            - 'answer': correct answer letter
            - 'subject': subject name
        """
        # Placeholder: in production, load from actual dataset
        # Expected format per question:
        return [
            {
                "question": f"Sample {subject} question",
                "choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
                "answer": "A",
                "subject": subject,
            }
        ]

    def evaluate(
        self,
        model: nn.Module,
        config: EvalConfig,
    ) -> SuiteResult:
        """
        Run MMLU evaluation on the model.

        Args:
            model: The model to evaluate
            config: Evaluation configuration

        Returns:
            SuiteResult with accuracy, per-subject scores, and cost metrics

        Explanation:
            For each question:
            1. Format prompt with few-shot examples
            2. Generate model response (typically 1 token for MC)
            3. Extract predicted letter
            4. Compare to ground truth

        Example:
            >>> evaluator = MMLUEvaluator()
            >>> result = evaluator.evaluate(model, EvalConfig())
            >>> print(result.accuracy)
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
            subject = item["subject"]
            prompt = self.formatter.format_question(
                question=item["question"],
                choices=item["choices"],
                subject=subject,
                examples=[],  # Would use actual examples in production
                config=self.config,
            )

            # Tokenize and generate
            # In production, use the model's tokenizer
            # Here we show the interface
            response, tokens_used = self._generate_response(
                model, prompt, config.sampling, device
            )
            total_tokens += tokens_used

            # Extract and compare answer
            predicted = self.formatter.extract_answer(
                response, self.config.choice_letters
            )

            is_correct = predicted == item["answer"]
            if is_correct:
                num_correct += 1

            num_total += 1
            per_subject_total[subject] = per_subject_total.get(subject, 0) + 1
            if is_correct:
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
        """
        Generate a response from the model.

        Args:
            model: The model to generate from
            prompt: Formatted prompt string
            sampling: Sampling configuration
            device: Device to run on

        Returns:
            Tuple of (response_text, tokens_generated)

        Explanation:
            In production, this would use the model's tokenizer and
            generate method. For MMLU, we typically generate only 1 token
            (the answer letter).
        """
        # Placeholder: in production, use actual tokenizer and generate
        # This shows the expected interface
        return "A", 1

    def check_decontamination(
        self,
        training_data: List[str],
    ) -> DecontaminationStatus:
        """
        Check for MMLU question overlap in training data.

        Args:
            training_data: List of training document strings

        Returns:
            DecontaminationStatus indicating contamination level

        Explanation:
            MMLU questions are widely available on the internet, making
            contamination a significant concern. We check for n-gram
            overlap between training data and MMLU question text.
        """
        checker = DecontaminationChecker(ngram_size=13)
        eval_texts = [item["question"] for item in self._data]
        return checker.check(eval_texts, training_data)


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################

def demonstrate_mmlu():
    """
    Demonstrate MMLU evaluation pipeline.

    Shows:
        1. MMLU configuration
        2. Prompt formatting
        3. Answer extraction
        4. Per-subject tracking
        5. Decontamination checking
    """
    print("=" * 70)
    print("MMLU / MMLU-PRO EVALUATION DEMONSTRATION")
    print("=" * 70)

    # --- Demonstrate Configuration ---
    print("\n--- Configuration ---")
    config_standard = MMLUConfig(variant="mmlu")
    config_pro = MMLUConfig(variant="mmlu-pro")
    print(f"  Standard MMLU: {len(MMLU_SUBJECTS)} subjects, "
          f"{config_standard.num_choices} choices")
    print(f"  MMLU-Pro: {len(MMLU_PRO_SUBJECTS)} subjects, "
          f"{config_pro.num_choices} choices")

    # --- Demonstrate Prompt Formatting ---
    print("\n--- Prompt Formatting ---")
    formatter = MMLUPromptFormatter()
    prompt = formatter.format_question(
        question="What is the derivative of x^2?",
        choices=["x", "2x", "x^2", "2x^2"],
        subject="calculus",
        examples=[
            {
                "question": "What is the integral of 2x?",
                "choices": ["x", "x^2", "2x^2", "x^3"],
                "answer": "B",
            }
        ],
        config=config_standard,
    )
    print(f"  Formatted prompt (first 200 chars):")
    print(f"  {prompt[:200]}...")

    # --- Demonstrate Answer Extraction ---
    print("\n--- Answer Extraction ---")
    test_responses = [
        ("A", "A"),
        ("The answer is B.", "B"),
        ("I think C is correct", "C"),
        ("Based on my analysis, D", "D"),
        ("", None),
    ]
    for response, expected in test_responses:
        extracted = formatter.extract_answer(response, ["A", "B", "C", "D"])
        status = "OK" if extracted == expected else "FAIL"
        print(f"  [{status}] '{response}' -> {extracted} (expected {expected})")

    # --- Demonstrate Evaluator ---
    print("\n--- Evaluator ---")
    evaluator = MMLUEvaluator(config_standard)
    print(f"  Name: {evaluator.name}")
    print(f"  Template: {evaluator.prompt_template[:80]}...")

    # --- Demonstrate Decontamination ---
    print("\n--- Decontamination Check ---")
    checker = DecontaminationChecker(ngram_size=5)
    sample_eval = ["What is the capital of France in geography?"]
    sample_train = ["The capital of France is Paris in geography"]
    status = checker.check(sample_eval, sample_train)
    print(f"  Status: {status.value}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_mmlu()
