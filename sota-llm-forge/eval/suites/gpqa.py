"""
################################################################################
GPQA-DIAMOND — GRADUATE-LEVEL SCIENCE REASONING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is GPQA-Diamond?
    GPQA (Graduate-Level Google-Proof Q&A) Diamond is a benchmark of
    graduate-level science questions that are difficult even for domain
    experts. The Diamond subset contains the hardest questions, verified
    by PhD-level annotators.

Why does it matter?
    GPQA-Diamond remains one of the few benchmarks that discriminates
    at the frontier level. While MMLU is saturated, GPQA-Diamond
    still challenges top models with deep scientific reasoning.
    Questions require genuine understanding, not pattern matching.

How does it work?
    1. Load graduate-level science questions (physics, chemistry, biology)
    2. Format as multiple-choice with 4 options
    3. Generate model response
    4. Extract predicted answer
    5. Compare to ground truth expert annotation

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────┐
    │ GPQA-Diamond Evaluation                                     │
    │                                                              │
    │  Expert-Verified Questions ──▶ Format ──▶ Generate ──▶ Score │
    │                                                              │
    │  Subjects: Physics, Chemistry, Biology                       │
    │  Difficulty: Graduate/PhD level                              │
    │  Format: 4-choice multiple choice                            │
    │  Metric: Accuracy                                            │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2023: GPQA introduced (Rein et al.) — expert-verified science QA
    - 2024: GPQA-Diamond subset identified as most discriminating
    - 2024: Frontier models reach 50-60% (vs ~65% expert accuracy)
    - 2025: Still discriminating at frontier; used in model cards
    - 2026: Remains a key frontier benchmark alongside HLE

INTERVIEW QUESTIONS:
    1. "What makes GPQA different from MMLU?"
       GPQA questions are verified by PhD experts to be genuinely hard.
       MMLU includes many easy questions that test recall. GPQA requires
       deep reasoning and domain expertise, making it more discriminating
       at the frontier.

    2. "How do you handle questions where even experts disagree?"
       GPQA uses a consensus-based approach: questions are kept only when
       domain experts agree on the answer. The Diamond subset has the
       highest expert agreement, ensuring ground truth reliability.

    3. "Why is GPQA still useful in 2026?"
       Unlike MMLU (saturated), GPQA-Diamond still has meaningful headroom.
       Frontier models score 50-70% while expert accuracy is ~65%.
       This means there's still room to improve and differentiate models.

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
class GPQAConfig:
    """
    GPQA Evaluation Configuration
    ==============================

    Controls GPQA-specific evaluation parameters.

    Attributes:
        subset: Which GPQA subset ('diamond', 'extended', 'main')
        num_shots: Number of few-shot examples
        choice_letters: Valid answer letters
        prompt_format: Template for formatting questions
        require_chain_of_thought: Whether to require reasoning before answer
    """
    subset: str = "diamond"
    num_shots: int = 5
    choice_letters: List[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    prompt_format: str = (
        "Answer the following graduate-level science question. "
        "Think step by step, then give your answer as a single letter.\n\n"
        "{question}\n"
        "{choices}\n\n"
        "Reasoning: {{model reasoning here}}\n"
        "Answer:"
    )
    require_chain_of_thought: bool = True


################################################################################
# SECTION 2: GPQA SUBJECT DEFINITIONS
################################################################################

GPQA_SUBJECTS = ["physics", "chemistry", "biology"]

GPQA_SUBSET_SIZES = {
    "diamond": 198,
    "extended": 546,
    "main": 448,
}


################################################################################
# SECTION 3: PROMPT FORMATTER
################################################################################

class GPQAPromptFormatter:
    """
    GPQA Prompt Formatter
    =====================

    Formats GPQA questions with optional chain-of-thought prompting.
    Graduate-level questions benefit from explicit reasoning.

    Step by step:
        1. Build question with subject context
        2. Format answer choices
        3. Optionally add chain-of-thought instruction
        4. Append "Answer:" for model completion

    Interview Question:
        "Should you use chain-of-thought for GPQA?"
        Yes. Graduate-level science questions require multi-step reasoning.
        Chain-of-thought prompting typically improves GPQA scores by
        5-10% by forcing the model to reason through the problem before
        answering.
    """

    @staticmethod
    def format_question(
        question: str,
        choices: List[str],
        config: GPQAConfig,
        examples: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Format a GPQA question into a prompt.

        Args:
            question: The question text
            choices: List of choice texts
            config: GPQA configuration
            examples: Optional few-shot examples

        Returns:
            Formatted prompt string

        Example:
            >>> formatter = GPQAPromptFormatter()
            >>> prompt = formatter.format_question(
            ...     "What is the ground state energy of a harmonic oscillator?",
            ...     ["ℏω/2", "ℏω", "3ℏω/2", "2ℏω"],
            ...     GPQAConfig(),
            ... )
        """
        # Format examples
        example_str = ""
        if examples:
            for ex in examples[:config.num_shots]:
                ex_choices = "\n".join(
                    f"{config.choice_letters[i]}. {c}"
                    for i, c in enumerate(ex["choices"])
                )
                if config.require_chain_of_thought:
                    example_str += (
                        f"{ex['question']}\n{ex_choices}\n\n"
                        f"Reasoning: {ex.get('reasoning', 'Step by step analysis.')}\n"
                        f"Answer: {ex['answer']}\n\n"
                    )
                else:
                    example_str += f"{ex['question']}\n{ex_choices}\nAnswer: {ex['answer']}\n\n"

        # Format target question
        choice_str = "\n".join(
            f"{config.choice_letters[i]}. {c}"
            for i, c in enumerate(choices)
        )

        prompt = config.prompt_format.format(
            question=question,
            choices=choice_str,
        )

        if example_str:
            prompt = example_str + prompt

        return prompt

    @staticmethod
    def extract_answer(response: str, valid_letters: List[str]) -> Optional[str]:
        """
        Extract predicted answer from model response.

        Args:
            response: Raw model response text
            valid_letters: List of valid answer letters

        Returns:
            Predicted answer letter, or None if extraction fails
        """
        response = response.strip()

        # Look for explicit "Answer: X" pattern
        pattern = r"[Aa]nswer:\s*([A-D])"
        match = re.search(pattern, response)
        if match and match.group(1).upper() in valid_letters:
            return match.group(1).upper()

        # First character
        if response and response[0].upper() in valid_letters:
            return response[0].upper()

        # First valid letter
        for char in response:
            if char.upper() in valid_letters:
                return char.upper()

        return None


################################################################################
# SECTION 4: GPQA EVALUATOR
################################################################################

class GPQAEvaluator:
    """
    GPQA-Diamond: Graduate-Level Science Reasoning
    ===============================================

    Still discriminating at frontier level.
    Format: graduate-level multiple choice (4 options)
    Metric: accuracy

    Step by step:
        1. Load GPQA-Diamond questions (198 questions)
        2. Format with chain-of-thought prompting
        3. Generate model response
        4. Extract predicted answer
        5. Compare to expert-verified ground truth
        6. Report accuracy

    WHY this matters:
        GPQA-Diamond tests deep scientific understanding that cannot be
        solved by surface-level pattern matching. It remains one of the
        few benchmarks where frontier models have not yet reached expert
        performance, making it valuable for model comparison.

    Interview Question:
        "How does GPQA test something different from MMLU?"
        MMLU tests broad knowledge recall across 57 subjects with many
        easy questions. GPQA-Diamond tests deep reasoning in hard science
        with questions verified by PhD experts to be genuinely difficult.
        A model can score 90% on MMLU through memorization but struggle
        on GPQA without true understanding.
    """

    # Class-level constants
    DEFAULT_NUM_SHOTS: int = 5
    DEFAULT_MAX_NEW_TOKENS: int = 512  # Longer for chain-of-thought

    def __init__(self, config: Optional[GPQAConfig] = None):
        """
        Initialize GPQA evaluator.

        Args:
            config: GPQA-specific configuration (uses defaults if None)
        """
        self.config = config or GPQAConfig()
        self.formatter = GPQAPromptFormatter()
        self._data: List[Dict[str, Any]] = []
        self._loaded = False

    @property
    def name(self) -> str:
        """Name of the benchmark suite."""
        return f"GPQA-{self.config.subset.capitalize()}"

    @property
    def prompt_template(self) -> str:
        """Exact prompt template used."""
        return self.config.prompt_format

    def load_data(self, config: EvalConfig) -> List[Dict[str, Any]]:
        """
        Load GPQA dataset.

        Args:
            config: Evaluation configuration

        Returns:
            List of dicts with 'prompt', 'answer', 'subject', and metadata
        """
        # In production, load from dataset files
        # Expected format: question, choices, answer, subject, reasoning
        self._data = self._load_subset(self.config.subset)
        self._loaded = True
        return self._data

    def _load_subset(self, subset: str) -> List[Dict[str, Any]]:
        """
        Load a specific GPQA subset.

        Args:
            subset: Subset name ('diamond', 'extended', 'main')

        Returns:
            List of question dicts
        """
        # Placeholder: in production, load from actual dataset
        expected_size = GPQA_SUBSET_SIZES.get(subset, 198)
        return [
            {
                "question": f"Sample GPQA {subset} question {i}",
                "choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
                "answer": "A",
                "subject": GPQA_SUBJECTS[i % len(GPQA_SUBJECTS)],
                "reasoning": "Sample reasoning chain.",
            }
            for i in range(expected_size)
        ]

    def evaluate(
        self,
        model: nn.Module,
        config: EvalConfig,
    ) -> SuiteResult:
        """
        Run GPQA evaluation on the model.

        Args:
            model: The model to evaluate
            config: Evaluation configuration

        Returns:
            SuiteResult with accuracy and cost metrics

        Example:
            >>> evaluator = GPQAEvaluator()
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
            prompt = self.formatter.format_question(
                question=item["question"],
                choices=item["choices"],
                config=self.config,
            )

            response, tokens_used = self._generate_response(
                model, prompt, config.sampling, device
            )
            total_tokens += tokens_used

            predicted = self.formatter.extract_answer(
                response, self.config.choice_letters
            )

            if predicted == item["answer"]:
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
        return "A", 100

    def check_decontamination(
        self,
        training_data: List[str],
    ) -> DecontaminationStatus:
        """
        Check for GPQA question overlap in training data.

        Args:
            training_data: List of training document strings

        Returns:
            DecontaminationStatus indicating contamination level
        """
        checker = DecontaminationChecker(ngram_size=13)
        eval_texts = [item["question"] for item in self._data]
        return checker.check(eval_texts, training_data)


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################

def demonstrate_gpqa():
    """
    Demonstrate GPQA evaluation pipeline.

    Shows:
        1. GPQA configuration and subsets
        2. Prompt formatting with chain-of-thought
        3. Answer extraction
        4. Decontamination checking
    """
    print("=" * 70)
    print("GPQA-DIAMOND EVALUATION DEMONSTRATION")
    print("=" * 70)

    # --- Demonstrate Configuration ---
    print("\n--- Configuration ---")
    config = GPQAConfig(subset="diamond", require_chain_of_thought=True)
    print(f"  Subset: {config.subset}")
    print(f"  Expected questions: {GPQA_SUBSET_SIZES[config.subset]}")
    print(f"  Chain-of-thought: {config.require_chain_of_thought}")
    print(f"  Subjects: {GPQA_SUBJECTS}")

    # --- Demonstrate Prompt Formatting ---
    print("\n--- Prompt Formatting ---")
    formatter = GPQAPromptFormatter()
    prompt = formatter.format_question(
        question="What is the Casimir effect between two parallel conducting plates?",
        choices=[
            "Attractive force proportional to 1/d^4",
            "Repulsive force proportional to 1/d^3",
            "Attractive force proportional to 1/d^4",
            "No force in vacuum",
        ],
        config=config,
        examples=[
            {
                "question": "What is the Lamb shift?",
                "choices": ["QED effect", "Nuclear force", "Gravity", "Weak force"],
                "answer": "A",
                "reasoning": "The Lamb shift is a QED effect from vacuum fluctuations.",
            }
        ],
    )
    print(f"  Prompt preview (first 300 chars):")
    print(f"  {prompt[:300]}...")

    # --- Demonstrate Answer Extraction ---
    print("\n--- Answer Extraction ---")
    test_cases = [
        ("Reasoning: The Casimir effect arises from vacuum fluctuations.\nAnswer: A", "A"),
        ("The answer is C based on quantum field theory.", "C"),
        ("D", "D"),
    ]
    for response, expected in test_cases:
        extracted = formatter.extract_answer(response, ["A", "B", "C", "D"])
        status = "OK" if extracted == expected else "FAIL"
        print(f"  [{status}] Extracted {extracted} (expected {expected})")

    # --- Demonstrate Evaluator ---
    print("\n--- Evaluator ---")
    evaluator = GPQAEvaluator(config)
    print(f"  Name: {evaluator.name}")
    print(f"  Template: {evaluator.prompt_template[:80]}...")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_gpqa()
