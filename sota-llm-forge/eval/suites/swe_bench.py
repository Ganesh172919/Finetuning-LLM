"""
################################################################################
SWE-BENCH VERIFIED — REAL GITHUB ISSUE RESOLUTION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is SWE-bench Verified?
    SWE-bench Verified is a benchmark where a model must resolve real
    GitHub issues by generating patches to open-source repositories.
    Each task presents an issue description and the repository state,
    and the model must produce a patch that fixes the issue (verified
    by running the repository's test suite).

Why does it matter?
    SWE-bench tests real-world software engineering ability: reading
    code, understanding issues, navigating large codebases, and
    producing correct patches. It's the most realistic code benchmark
    available, but also the most expensive to run (requires agentic,
    multi-turn interactions).

How does it work?
    1. Load issue descriptions and repository snapshots
    2. Provide the model with issue context and codebase access
    3. Model generates a patch (multi-turn, agentic)
    4. Apply the patch to the repository
    5. Run the repository's test suite
    6. Report resolved percentage

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────┐
    │ SWE-bench Evaluation                                        │
    │                                                              │
    │  Issue + Repo ──▶ Agent Loop ──▶ Generate Patch             │
    │                                      ↓                       │
    │                               Apply Patch                    │
    │                                      ↓                       │
    │                               Run Tests                      │
    │                                      ↓                       │
    │                            Pass / Fail                       │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2023: SWE-bench introduced (Jimenez et al.) — GitHub issue resolution
    - 2024: SWE-bench Verified subset created — human-validated tasks
    - 2024: Frontier agents reach 20-30% on SWE-bench Verified
    - 2025: Top agents reach 50%+ with improved scaffolding
    - 2026: SWE-bench Verified remains key agentic coding benchmark

INTERVIEW QUESTIONS:
    1. "How is SWE-bench different from HumanEval?"
       HumanEval tests isolated function generation. SWE-bench tests
       real-world issue resolution: understanding existing code, navigating
       large codebases, and producing patches that pass existing tests.
       It's multi-turn and agentic, not single-shot.

    2. "Why is SWE-bench expensive to run?"
       Each task requires: cloning a repo, understanding the codebase,
       generating multiple patch candidates, applying patches, and running
       the full test suite. This can take minutes to hours per task,
       with significant compute for large repositories.

    3. "What makes a good SWE-bench agent?"
       Effective agents: (1) explore the codebase systematically,
       (2) understand the issue deeply before coding, (3) generate
       minimal patches that fix the issue without breaking other things,
       (4) verify their patches by running relevant tests.

################################################################################
"""

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
class SWEBenchConfig:
    """
    SWE-bench Evaluation Configuration
    ===================================

    Controls SWE-bench-specific evaluation parameters.

    Attributes:
        subset: Which subset ('verified', 'full', 'lite')
        max_turns: Maximum agentic turns per task
        timeout_minutes: Timeout per task in minutes
        prompt_format: Template for formatting issues
        use_scaffolding: Whether to use agent scaffolding
        num_patch_candidates: Number of patch candidates to generate
    """
    subset: str = "verified"
    max_turns: int = 50
    timeout_minutes: int = 30
    prompt_format: str = (
        "You are a software engineer. Fix the following GitHub issue.\n\n"
        "Repository: {repo}\n"
        "Issue:\n{issue_description}\n\n"
        "Relevant files:\n{relevant_files}\n\n"
        "Generate a patch that fixes this issue."
    )
    use_scaffolding: bool = True
    num_patch_candidates: int = 1


################################################################################
# SECTION 2: PATCH GENERATION
################################################################################

class PatchResult(Enum):
    """Result of applying and testing a patch."""
    RESOLVED = "resolved"      # Patch applied, tests pass
    APPLIED_FAIL = "applied_fail"  # Patch applied, tests fail
    APPLY_ERROR = "apply_error"    # Patch could not be applied
    TIMEOUT = "timeout"            # Test execution timed out
    ERROR = "error"                # Other error


@dataclass
class TaskResult:
    """
    Result of a single SWE-bench task.

    Attributes:
        task_id: Identifier for the task
        repo: Repository name
        result: Patch result status
        patch: Generated patch (if any)
        execution_time: Time taken in seconds
        error_message: Error message if failed
    """
    task_id: str
    repo: str
    result: PatchResult
    patch: Optional[str] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None


################################################################################
# SECTION 3: AGENT SCAFFOLDING
################################################################################

class AgentScaffold:
    """
    Agent Scaffold for SWE-bench
    =============================

    Provides the scaffolding loop for agentic interaction with
    the codebase. Manages context, tool calls, and patch generation.

    Step by step:
        1. Parse issue description
        2. Explore codebase (file listing, reading, searching)
        3. Identify relevant files
        4. Generate patch candidates
        5. Apply and test patches
        6. Return best patch

    WHY this matters:
        SWE-bench requires multi-turn interaction with a codebase.
        The scaffold manages the agent loop, context window, and
        tool calls that enable the model to effectively navigate
        and modify the codebase.

    Interview Question:
        "How would you design an agent for SWE-bench?"
        Use a ReAct-style loop: observe (read files, search code),
        think (reason about the issue), act (edit files, run tests).
        Keep a scratchpad of findings. Use file-level context management
        to stay within context limits. Verify patches by running tests.
    """

    def __init__(self, config: SWEBenchConfig):
        """
        Initialize the agent scaffold.

        Args:
            config: SWE-bench configuration
        """
        self.config = config
        self.tools = self._setup_tools()

    def _setup_tools(self) -> Dict[str, Any]:
        """
        Set up available tools for the agent.

        Returns:
            Dict of tool name → tool function
        """
        return {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "search_code": self._search_code,
            "edit_file": self._edit_file,
            "run_tests": self._run_tests,
        }

    def _list_files(self, path: str) -> List[str]:
        """List files in a directory."""
        # Placeholder: in production, list actual files
        return []

    def _read_file(self, path: str) -> str:
        """Read file contents."""
        # Placeholder: in production, read actual file
        return ""

    def _search_code(self, query: str) -> List[Dict[str, str]]:
        """Search for code matching query."""
        # Placeholder: in production, use grep/ripgrep
        return []

    def _edit_file(self, path: str, old: str, new: str) -> bool:
        """Edit a file by replacing old text with new text."""
        # Placeholder: in production, make actual edit
        return True

    def _run_tests(self, test_command: str) -> Tuple[bool, str]:
        """Run tests and return (passed, output)."""
        # Placeholder: in production, run actual tests
        return True, ""

    def generate_patch(
        self,
        model: nn.Module,
        issue: Dict[str, Any],
        device: Any,
    ) -> Tuple[str, int]:
        """
        Generate a patch for an issue using the model.

        Args:
            model: The model to use for generation
            issue: Issue dict with description and repo info
            device: Device to run on

        Returns:
            Tuple of (patch_string, tokens_used)

        Explanation:
            Runs the agentic loop for up to max_turns, using the model
            to decide what actions to take at each step.
        """
        total_tokens = 0
        context = self._build_initial_context(issue)

        for turn in range(self.config.max_turns):
            # Generate next action
            response, tokens = self._model_step(model, context, device)
            total_tokens += tokens

            # Parse and execute action
            action = self._parse_action(response)
            if action["type"] == "submit_patch":
                return action["patch"], total_tokens

            # Execute tool and update context
            tool_result = self._execute_tool(action)
            context += f"\n{tool_result}"

        return "", total_tokens

    def _build_initial_context(self, issue: Dict[str, Any]) -> str:
        """Build initial context from issue description."""
        return self.config.prompt_format.format(
            repo=issue.get("repo", "unknown"),
            issue_description=issue.get("description", ""),
            relevant_files=issue.get("relevant_files", "None listed"),
        )

    def _model_step(
        self,
        model: nn.Module,
        context: str,
        device: Any,
    ) -> Tuple[str, int]:
        """Take a single model step in the agent loop."""
        # Placeholder: in production, use actual model
        return "submit_patch: diff --git a/...", 100

    def _parse_action(self, response: str) -> Dict[str, Any]:
        """Parse model response into an action."""
        # Placeholder: in production, parse actual response
        return {"type": "submit_patch", "patch": ""}

    def _execute_tool(self, action: Dict[str, Any]) -> str:
        """Execute a tool call."""
        tool_name = action.get("tool", "")
        if tool_name in self.tools:
            return str(self.tools[tool_name](**action.get("args", {})))
        return "Unknown tool"


################################################################################
# SECTION 4: SWE-BENCH EVALUATOR
################################################################################

class SWEBenchEvaluator:
    """
    SWE-bench Verified: Real GitHub Issue Resolution
    =================================================

    Expensive to run (agentic, multi-turn).
    Budget accordingly.
    Format: issue description + codebase → patch
    Metric: resolved (patch passes tests)

    Step by step:
        1. Load SWE-bench Verified tasks (500 human-verified tasks)
        2. For each task, run agent scaffold with the model
        3. Generate patch candidates
        4. Apply patch to repository
        5. Run test suite
        6. Report resolved percentage

    WHY this matters:
        SWE-bench is the most realistic coding benchmark. It tests
        the full software engineering workflow: understanding issues,
        navigating codebases, and producing correct patches. Success
        on SWE-bench indicates real-world coding capability.

    Interview Question:
        "What does SWE-bench test that other benchmarks don't?"
        SWE-bench tests real-world software engineering: reading existing
        code, understanding complex issue descriptions, navigating large
        codebases, and producing minimal patches that fix issues without
        breaking other functionality. It requires multi-turn interaction
        and tool use, unlike single-shot benchmarks.
    """

    # Class-level constants
    VERIFIED_TASK_COUNT: int = 500
    FULL_TASK_COUNT: int = 2294

    def __init__(self, config: Optional[SWEBenchConfig] = None):
        """
        Initialize SWE-bench evaluator.

        Args:
            config: SWE-bench-specific configuration
        """
        self.config = config or SWEBenchConfig()
        self.scaffold = AgentScaffold(self.config)
        self._data: List[Dict[str, Any]] = []
        self._loaded = False

    @property
    def name(self) -> str:
        """Name of the benchmark suite."""
        return f"SWE-bench-{self.config.subset.capitalize()}"

    @property
    def prompt_template(self) -> str:
        """Exact prompt template used."""
        return self.config.prompt_format

    def load_data(self, config: EvalConfig) -> List[Dict[str, Any]]:
        """
        Load SWE-bench dataset.

        Args:
            config: Evaluation configuration

        Returns:
            List of task dicts with issues and repo info
        """
        # Placeholder: in production, load from dataset
        expected_size = (
            self.VERIFIED_TASK_COUNT if self.config.subset == "verified"
            else self.FULL_TASK_COUNT
        )
        self._data = [
            {
                "task_id": f"swe_{i}",
                "repo": f"owner/repo_{i % 10}",
                "description": f"Sample GitHub issue {i}",
                "relevant_files": f"src/file_{i}.py",
                "base_commit": "abc123",
                "test_patch": "test_patch_diff",
            }
            for i in range(min(expected_size, 10))  # Limit for demo
        ]
        self._loaded = True
        return self._data

    def evaluate(
        self,
        model: nn.Module,
        config: EvalConfig,
    ) -> SuiteResult:
        """
        Run SWE-bench evaluation on the model.

        Args:
            model: The model to evaluate
            config: Evaluation configuration

        Returns:
            SuiteResult with resolved percentage and cost metrics

        Example:
            >>> evaluator = SWEBenchEvaluator()
            >>> result = evaluator.evaluate(model, EvalConfig())
            >>> print(result.accuracy)  # resolved percentage
        """
        if not self._loaded:
            self.load_data(config)

        device = next(model.parameters()).device
        num_resolved = 0
        num_total = 0
        total_tokens = 0

        for task in self._data:
            print(f"  [{self.name}] Task {task['task_id']}...")

            patch, tokens_used = self.scaffold.generate_patch(
                model, task, device
            )
            total_tokens += tokens_used

            # Apply and test patch
            result = self._apply_and_test(patch, task)

            if result == PatchResult.RESOLVED:
                num_resolved += 1
            num_total += 1

        accuracy = num_resolved / num_total if num_total > 0 else 0.0
        cost_per_correct = CostAwareMetrics.cost_per_correct(
            total_tokens, num_resolved
        )

        return SuiteResult(
            suite_name=self.name,
            accuracy=accuracy,
            num_correct=num_resolved,
            num_total=num_total,
            tokens_generated=total_tokens,
            cost_per_correct=cost_per_correct,
            prompt_template=self.config.prompt_format,
            sampling_params={
                "temperature": config.sampling.temperature,
                "top_k": config.sampling.top_k,
                "top_p": config.sampling.top_p,
                "max_turns": self.config.max_turns,
            },
        )

    def _apply_and_test(
        self,
        patch: str,
        task: Dict[str, Any],
    ) -> PatchResult:
        """
        Apply patch and run tests.

        Args:
            patch: The generated patch string
            task: Task dict with test information

        Returns:
            PatchResult indicating success or failure
        """
        if not patch:
            return PatchResult.APPLY_ERROR

        # Placeholder: in production, actually apply patch and run tests
        return PatchResult.RESOLVED

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
        """
        checker = DecontaminationChecker(ngram_size=10)
        eval_texts = [item["description"] for item in self._data]
        return checker.check(eval_texts, training_data)


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################

def demonstrate_swe_bench():
    """
    Demonstrate SWE-bench evaluation pipeline.

    Shows:
        1. Configuration
        2. Agent scaffold setup
        3. Patch generation interface
        4. Cost considerations
    """
    print("=" * 70)
    print("SWE-BENCH VERIFIED EVALUATION DEMONSTRATION")
    print("=" * 70)

    # --- Demonstrate Configuration ---
    print("\n--- Configuration ---")
    config = SWEBenchConfig(
        subset="verified",
        max_turns=50,
        timeout_minutes=30,
    )
    print(f"  Subset: {config.subset}")
    print(f"  Verified tasks: {SWEBenchEvaluator.VERIFIED_TASK_COUNT}")
    print(f"  Max turns: {config.max_turns}")
    print(f"  Timeout: {config.timeout_minutes} minutes")
    print(f"  Scaffolding: {config.use_scaffolding}")

    # --- Demonstrate Agent Scaffold ---
    print("\n--- Agent Scaffold ---")
    scaffold = AgentScaffold(config)
    print(f"  Available tools: {list(scaffold.tools.keys())}")
    print(f"  Max turns per task: {config.max_turns}")

    # --- Demonstrate Patch Results ---
    print("\n--- Patch Results ---")
    for result in PatchResult:
        print(f"  {result.value}: {result.name}")

    # --- Demonstrate Cost Awareness ---
    print("\n--- Cost Considerations ---")
    estimated_tokens_per_task = 50000  # Conservative estimate
    tasks = SWEBenchEvaluator.VERIFIED_TASK_COUNT
    total_tokens = estimated_tokens_per_task * tasks
    print(f"  Estimated tokens per task: {estimated_tokens_per_task:,}")
    print(f"  Total tasks: {tasks}")
    print(f"  Estimated total tokens: {total_tokens:,}")
    print(f"  At $0.01/1K tokens: ${total_tokens / 1000 * 0.01:.2f}")

    # --- Demonstrate Evaluator ---
    print("\n--- Evaluator ---")
    evaluator = SWEBenchEvaluator(config)
    print(f"  Name: {evaluator.name}")
    print(f"  Template: {evaluator.prompt_template[:80]}...")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_swe_bench()
