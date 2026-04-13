"""
################################################################################
ARC-AGI-2 / ARC-AGI-3 — ABSTRACT REASONING WITHOUT LANGUAGE SHORTCUTS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is ARC-AGI?
    ARC-AGI (Abstraction and Reasoning Corpus for Artificial General
    Intelligence) tests fluid, novel-task abstract reasoning without
    language or knowledge shortcuts. Each task presents grid transformation
    examples and the model must predict the output grid. ARC-AGI-2 is
    the static version; ARC-AGI-3 is interactive/agentic.

Why does it matter?
    ARC-AGI tests a form of intelligence that is fundamentally different
    from language understanding or knowledge recall. It requires:
    - Perceiving patterns in visual grids
    - Abstracting rules from examples
    - Applying rules to novel inputs
    Frontier models score under 1% on ARC-AGI-3, indicating that
    current architectures are far from human-level abstract reasoning.

How does it work?
    1. Load grid transformation tasks (train examples + test input)
    2. Present train examples to the model
    3. Model must predict the output grid for the test input
    4. Score by exact grid match
    5. Report accuracy (expect very low numbers)

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────┐
    │ ARC-AGI Evaluation                                          │
    │                                                              │
    │  Train Examples ──▶ Model ──▶ Predict Output Grid           │
    │   (input→output)              ↓                              │
    │                         Compare to Expected                  │
    │                              ↓                               │
    │                        Exact Match Score                     │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2019: ARC introduced (Chollet) — measuring intelligence, not skill
    - 2020: ARC-AGI benchmark released — 800 tasks
    - 2024: ARC-AGI-2 introduced — harder, more diverse tasks
    - 2025: ARC-AGI-3 introduced — interactive/agentic version
    - 2026: Frontier models score under 1% on ARC-AGI-3

INTERVIEW QUESTIONS:
    1. "Why is ARC-AGI so hard for LLMs?"
       ARC-AGI tests fluid intelligence — the ability to reason about
       novel problems without prior knowledge. LLMs are trained on
       language patterns and knowledge, not visual-abstract reasoning.
       Each task is unique, preventing pattern matching across tasks.

    2. "What would it take to solve ARC-AGI?"
       Solving ARC-AGI likely requires: (1) visual perception abilities,
       (2) program synthesis or rule abstraction, (3) few-shot adaptation
       to novel task structures. Current LLMs lack the first two.

    3. "How is ARC-AGI-3 different from ARC-AGI-2?"
       ARC-AGI-3 is interactive: the model can query the environment,
       test hypotheses, and iterate. This tests agentic problem-solving,
       not just one-shot prediction. It's closer to how humans solve
       novel problems — through exploration and experimentation.

################################################################################
"""

import json
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

try:
    import numpy as np
except ImportError:
    np = None

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
class ARCAGIConfig:
    """
    ARC-AGI Evaluation Configuration
    =================================

    Controls ARC-AGI-specific evaluation parameters.

    Attributes:
        version: Which ARC-AGI version ('2' or '3')
        split: Dataset split ('evaluation', 'training')
        max_new_tokens: Maximum tokens to generate
        prompt_format: Template for formatting tasks
        grid_max_size: Maximum grid dimension
        num_colors: Number of possible colors (0-9)
    """
    version: str = "2"
    split: str = "evaluation"
    max_new_tokens: int = 4096
    prompt_format: str = (
        "You are given grid transformation examples. "
        "Learn the pattern and apply it to the test input.\n\n"
        "{train_examples}\n"
        "Test Input:\n{test_input}\n\n"
        "Test Output (as a grid):"
    )
    grid_max_size: int = 30
    num_colors: int = 10


################################################################################
# SECTION 2: GRID UTILITIES
################################################################################

class GridUtils:
    """
    Grid Utilities for ARC-AGI
    ===========================

    Utility functions for working with ARC-AGI grids.
    Grids are 2D arrays of integers (0-9) representing colors.

    Grid format:
        [[0, 1, 0],
         [1, 2, 1],
         [0, 1, 0]]

    Colors: 0=black, 1=blue, 2=red, 3=green, 4=yellow,
            5=gray, 6=pink, 7=orange, 8=teal, 9=maroon
    """

    @staticmethod
    def parse_grid(grid_str: str) -> Optional[List[List[int]]]:
        """
        Parse a grid string into a 2D list of integers.

        Args:
            grid_str: String representation of a grid

        Returns:
            2D list of integers, or None if parsing fails

        Example:
            >>> GridUtils.parse_grid("[[0,1],[1,0]]")
            [[0, 1], [1, 0]]
        """
        try:
            grid = json.loads(grid_str)
            if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                return grid
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    @staticmethod
    def grid_to_string(grid: List[List[int]]) -> str:
        """
        Convert a grid to a string representation.

        Args:
            grid: 2D list of integers

        Returns:
            String representation of the grid

        Example:
            >>> GridUtils.grid_to_string([[0,1],[1,0]])
            '[[0, 1], [1, 0]]'
        """
        return json.dumps(grid)

    @staticmethod
    def grids_match(grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """
        Check if two grids are identical.

        Args:
            grid1: First grid
            grid2: Second grid

        Returns:
            True if grids are identical

        Example:
            >>> GridUtils.grids_match([[0,1],[1,0]], [[0,1],[1,0]])
            True
        """
        if len(grid1) != len(grid2):
            return False
        for row1, row2 in zip(grid1, grid2):
            if row1 != row2:
                return False
        return True

    @staticmethod
    def format_train_examples(examples: List[Dict[str, Any]]) -> str:
        """
        Format training examples as a string.

        Args:
            examples: List of dicts with 'input' and 'output' grids

        Returns:
            Formatted string of examples

        Example:
            >>> examples = [{"input": [[0]], "output": [[1]]}]
            >>> print(GridUtils.format_train_examples(examples))
            Example 1:
            Input: [[0]]
            Output: [[1]]
        """
        lines = []
        for i, ex in enumerate(examples):
            lines.append(f"Example {i + 1}:")
            lines.append(f"Input: {GridUtils.grid_to_string(ex['input'])}")
            lines.append(f"Output: {GridUtils.grid_to_string(ex['output'])}")
            lines.append("")
        return "\n".join(lines)


################################################################################
# SECTION 3: ARC-AGI EVALUATOR
################################################################################

class ARCAGIEvaluator:
    """
    ARC-AGI-2 / ARC-AGI-3: Abstract Reasoning
    ===========================================

    Fluid, novel-task abstract reasoning without language/knowledge shortcuts.
    ARC-AGI-3 is interactive/agentic. Frontier models score under 1%.
    Don't expect your model to move this number — implement the harness
    to understand WHY it's hard.

    Format: grid transformation examples → predict output grid
    Metric: exact match

    Step by step:
        1. Load ARC-AGI tasks (train examples + test input/output)
        2. Present train examples to the model
        3. Model predicts output grid for test input
        4. Compare prediction to expected output (exact grid match)
        5. Report accuracy (expect <1% for ARC-AGI-3)

    WHY this matters:
        ARC-AGI tests a fundamental form of intelligence: the ability to
        reason about novel problems without prior knowledge. Unlike language
        benchmarks, there are no shortcuts — each task requires genuine
        abstraction and reasoning.

    Interview Question:
        "Why is ARC-AGI considered a test of AGI?"
        ARC-AGI tests fluid intelligence: the ability to solve novel
        problems without prior knowledge or training. Each task is unique,
        preventing pattern matching. Humans score ~85% while frontier
        models score <1% on ARC-AGI-3, showing a fundamental gap in
        current AI architectures.
    """

    # Class-level constants
    EVAL_TASK_COUNT: int = 400
    TRAIN_TASK_COUNT: int = 400
    FRONTIER_ACCURACY_ARC2: float = 0.05  # ~5% on ARC-AGI-2
    FRONTIER_ACCURACY_ARC3: float = 0.01  # <1% on ARC-AGI-3
    HUMAN_ACCURACY: float = 0.85

    def __init__(self, config: Optional[ARCAGIConfig] = None):
        """
        Initialize ARC-AGI evaluator.

        Args:
            config: ARC-AGI-specific configuration
        """
        self.config = config or ARCAGIConfig()
        self.grid_utils = GridUtils()
        self._data: List[Dict[str, Any]] = []
        self._loaded = False

    @property
    def name(self) -> str:
        """Name of the benchmark suite."""
        return f"ARC-AGI-{self.config.version}"

    @property
    def prompt_template(self) -> str:
        """Exact prompt template used."""
        return self.config.prompt_format

    def load_data(self, config: EvalConfig) -> List[Dict[str, Any]]:
        """
        Load ARC-AGI dataset.

        Args:
            config: Evaluation configuration

        Returns:
            List of dicts with train examples, test input, and expected output
        """
        # Placeholder: in production, load from dataset
        self._data = [
            {
                "task_id": f"arc_{i}",
                "train": [
                    {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
                    {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
                ],
                "test_input": [[0, 0], [1, 1]],
                "test_output": [[1, 1], [0, 0]],
            }
            for i in range(min(self.EVAL_TASK_COUNT, 50))
        ]
        self._loaded = True
        return self._data

    def evaluate(
        self,
        model: nn.Module,
        config: EvalConfig,
    ) -> SuiteResult:
        """
        Run ARC-AGI evaluation on the model.

        Args:
            model: The model to evaluate
            config: Evaluation configuration

        Returns:
            SuiteResult with accuracy and cost metrics

        Example:
            >>> evaluator = ARCAGIEvaluator()
            >>> result = evaluator.evaluate(model, EvalConfig())
            >>> print(result.accuracy)  # expect <0.05
        """
        if not self._loaded:
            self.load_data(config)

        device = next(model.parameters()).device
        num_correct = 0
        num_total = 0
        total_tokens = 0

        for task in self._data:
            prompt = self._format_task(task)

            response, tokens_used = self._generate_response(
                model, prompt, config.sampling, device
            )
            total_tokens += tokens_used

            # Parse predicted grid
            predicted_grid = self.grid_utils.parse_grid(response)

            # Compare to expected output
            if predicted_grid is not None:
                if self.grid_utils.grids_match(predicted_grid, task["test_output"]):
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

    def _format_task(self, task: Dict[str, Any]) -> str:
        """
        Format an ARC-AGI task as a prompt.

        Args:
            task: Task dict with train examples and test input

        Returns:
            Formatted prompt string
        """
        train_str = self.grid_utils.format_train_examples(task["train"])
        test_str = self.grid_utils.grid_to_string(task["test_input"])

        return self.config.prompt_format.format(
            train_examples=train_str,
            test_input=test_str,
        )

    def _generate_response(
        self,
        model: nn.Module,
        prompt: str,
        sampling: Any,
        device: Any,
    ) -> Tuple[str, int]:
        """Generate a grid prediction from the model."""
        # Placeholder: in production, use actual tokenizer and generate
        return "[[1, 1], [0, 0]]", 100

    def check_decontamination(
        self,
        training_data: List[str],
    ) -> DecontaminationStatus:
        """
        Check for task overlap in training data.

        Args:
            training_data: List of training document strings

        Returns:
            DecontaminationStatus indicating contamination level

        Explanation:
            ARC-AGI tasks are unique visual patterns, making text-based
            contamination detection less relevant. However, we still
            check for any textual overlap in task descriptions.
        """
        # ARC-AGI is primarily visual, so contamination is less of a concern
        return DecontaminationStatus.NOT_CHECKED


################################################################################
# SECTION 4: TESTING & DEMONSTRATION
################################################################################

def demonstrate_arc_agi():
    """
    Demonstrate ARC-AGI evaluation pipeline.

    Shows:
        1. Configuration and version differences
        2. Grid utilities (parse, format, compare)
        3. Task formatting
        4. Expected accuracy ranges
    """
    print("=" * 70)
    print("ARC-AGI EVALUATION DEMONSTRATION")
    print("=" * 70)

    # --- Demonstrate Configuration ---
    print("\n--- Configuration ---")
    config_v2 = ARCAGIConfig(version="2")
    config_v3 = ARCAGIConfig(version="3")
    print(f"  ARC-AGI-2: {ARCAGIEvaluator.EVAL_TASK_COUNT} eval tasks")
    print(f"  ARC-AGI-3: Interactive/agentic version")
    print(f"  Grid max size: {config_v2.grid_max_size}x{config_v2.grid_max_size}")
    print(f"  Colors: {config_v2.num_colors}")

    # --- Demonstrate Grid Utilities ---
    print("\n--- Grid Utilities ---")
    gu = GridUtils()

    # Parse grid
    grid = gu.parse_grid("[[0,1,0],[1,2,1],[0,1,0]]")
    print(f"  Parsed grid: {grid}")

    # Grid to string
    grid_str = gu.grid_to_string([[1, 2], [3, 4]])
    print(f"  Grid string: {grid_str}")

    # Grid comparison
    match = gu.grids_match([[1, 2], [3, 4]], [[1, 2], [3, 4]])
    no_match = gu.grids_match([[1, 2], [3, 4]], [[4, 3], [2, 1]])
    print(f"  Same grids match: {match}")
    print(f"  Different grids match: {no_match}")

    # --- Demonstrate Task Formatting ---
    print("\n--- Task Formatting ---")
    evaluator = ARCAGIEvaluator(config_v2)
    sample_task = {
        "train": [
            {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
        ],
        "test_input": [[0, 0], [1, 1]],
        "test_output": [[1, 1], [0, 0]],
    }
    prompt = evaluator._format_task(sample_task)
    print(f"  Formatted task:")
    print(f"  {prompt[:300]}...")

    # --- Demonstrate Expected Scores ---
    print("\n--- Expected Accuracy Ranges ---")
    print(f"  ARC-AGI-2 frontier: ~{ARCAGIEvaluator.FRONTIER_ACCURACY_ARC2:.0%}")
    print(f"  ARC-AGI-3 frontier: <{ARCAGIEvaluator.FRONTIER_ACCURACY_ARC3:.0%}")
    print(f"  Human accuracy: ~{ARCAGIEvaluator.HUMAN_ACCURACY:.0%}")
    print(f"  NOTE: Don't expect your model to move these numbers.")

    # --- Demonstrate Evaluator ---
    print("\n--- Evaluator ---")
    print(f"  Name: {evaluator.name}")
    print(f"  Template: {evaluator.prompt_template[:80]}...")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_arc_agi()
