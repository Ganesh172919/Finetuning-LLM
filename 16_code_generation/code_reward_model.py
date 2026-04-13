"""
################################################################################
CODE REWARD MODELS — EVALUATING CODE QUALITY VIA EXECUTION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Code Reward Models?
    Reward models that evaluate code quality by running it, checking
    outputs, analyzing structure, and measuring correctness. Used
    to train code generation models via RL.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


################################################################################
# SECTION 1: EXECUTION REWARD
################################################################################

class ExecutionReward:
    """
    Execution-based Reward — Run code and check output.

    Gold standard for code evaluation: either it works or it doesn't.

    Interview Question:
        "How do you evaluate generated code?"
        Execution reward: run the code with test cases, check if output
        matches expected. This is the most reliable signal — either the
        code produces correct results or it doesn't. Combine with style
        and efficiency rewards for comprehensive evaluation.
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def evaluate(self, code: str, test_cases: List[Dict]) -> Dict:
        """
        Evaluate code against test cases.

        Args:
            code: Generated code string
            test_cases: List of {'input': ..., 'expected': ...}

        Returns:
            Dictionary with score, pass/fail per test
        """
        results = []
        for test in test_cases:
            # Simulate execution (in production: subprocess.run)
            passed = np.random.random() > 0.3  # 70% pass rate simulation
            results.append({
                'input': test['input'],
                'expected': test['expected'],
                'passed': passed
            })

        n_passed = sum(1 for r in results if r['passed'])
        return {
            'score': n_passed / max(len(results), 1),
            'n_passed': n_passed,
            'n_total': len(results),
            'results': results
        }


################################################################################
# SECTION 2: AST SIMILARITY REWARD
################################################################################

class ASTSimilarityReward:
    """
    AST Similarity — Compare code structure.

    Ignore variable names, focus on structure. Captures structural
    correctness even when implementation differs.

    Interview Question:
        "How do you compare code structurally?"
        Parse both codes into ASTs, compute tree edit distance.
        Similar structure = similar score. This captures correctness
        independent of variable naming or style differences.
    """

    def __init__(self):
        pass

    def compute_similarity(self, code1: str, code2: str) -> float:
        """
        Compute structural similarity between two code snippets.

        Args:
            code1: First code snippet
            code2: Second code snippet

        Returns:
            Similarity score (0 to 1)
        """
        # Simplified: compare token distributions
        tokens1 = set(code1.split())
        tokens2 = set(code2.split())
        if not tokens1 or not tokens2:
            return 0.0
        overlap = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return overlap / union


################################################################################
# SECTION 3: CODE QUALITY REWARD
################################################################################

class CodeQualityReward:
    """
    Code Quality Metrics.

    Score: cyclomatic complexity, naming, comments, length.

    Interview Question:
        "What makes good code beyond correctness?"
        (1) Low complexity — simple control flow,
        (2) Clear naming — descriptive variable/function names,
        (3) Adequate comments — explain why, not what,
        (4) Reasonable length — not too long or short,
        (5) Style consistency — follow conventions.
    """

    def evaluate(self, code: str) -> Dict:
        """
        Evaluate code quality.

        Args:
            code: Code string

        Returns:
            Dictionary with quality metrics
        """
        lines = code.split('\n')
        n_lines = len(lines)
        n_comments = sum(1 for l in lines if l.strip().startswith('#'))
        n_functions = sum(1 for l in lines if 'def ' in l)

        # Complexity estimate (simplified)
        n_branches = sum(1 for l in lines if any(k in l for k in ['if ', 'for ', 'while ']))
        complexity = n_branches + 1

        # Naming quality (simplified)
        has_descriptive = any(len(word) > 3 for word in code.split() if word.isalpha())

        score = 0.5
        if n_comments > 0:
            score += 0.1
        if complexity < 10:
            score += 0.1
        if has_descriptive:
            score += 0.1
        if 5 < n_lines < 100:
            score += 0.1

        return {
            'score': min(score, 1.0),
            'n_lines': n_lines,
            'n_comments': n_comments,
            'complexity': complexity,
            'n_functions': n_functions
        }


################################################################################
# SECTION 4: COMBINED CODE REWARD
################################################################################

class CombinedCodeReward:
    """
    Combined code reward for RL training.

    Formula: R = w1 * execution + w2 * ast_sim + w3 * quality

    Interview Question:
        "How do you combine multiple code rewards?"
        Weighted combination: execution (0.6) + AST similarity (0.2) +
        quality (0.2). Execution is most important (correctness > style).
        Normalize each reward to [0,1] before combining.
    """

    def __init__(self, w_execution: float = 0.6, w_ast: float = 0.2,
                 w_quality: float = 0.2):
        self.w_execution = w_execution
        self.w_ast = w_ast
        self.w_quality = w_quality
        self.execution = ExecutionReward()
        self.ast = ASTSimilarityReward()
        self.quality = CodeQualityReward()

    def evaluate(self, code: str, test_cases: List[Dict],
                 reference: str = "") -> Dict:
        """
        Combined evaluation.

        Args:
            code: Generated code
            test_cases: Test cases for execution
            reference: Reference solution for AST comparison

        Returns:
            Dictionary with combined score
        """
        exec_result = self.execution.evaluate(code, test_cases)
        quality_result = self.quality.evaluate(code)
        ast_score = self.ast.compute_similarity(code, reference) if reference else 0.5

        combined = (
            self.w_execution * exec_result['score'] +
            self.w_ast * ast_score +
            self.w_quality * quality_result['score']
        )

        return {
            'combined_score': combined,
            'execution_score': exec_result['score'],
            'ast_similarity': ast_score,
            'quality_score': quality_result['score'],
            'n_passed': exec_result['n_passed'],
            'n_total': exec_result['n_total']
        }


################################################################################
# SECTION 5: DEMONSTRATION
################################################################################

def demonstrate_code_reward_model():
    """Demonstrate code reward models."""
    print("=" * 70)
    print("CODE REWARD MODEL DEMONSTRATION")
    print("=" * 70)

    code = """
def fibonacci(n):
    # Calculate Fibonacci number
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    test_cases = [
        {'input': 0, 'expected': 0},
        {'input': 1, 'expected': 1},
        {'input': 10, 'expected': 55},
    ]

    # Execution Reward
    print("\n1. EXECUTION REWARD")
    print("-" * 40)
    er = ExecutionReward()
    result = er.evaluate(code, test_cases)
    print(f"  Score: {result['score']:.2%}")
    print(f"  Passed: {result['n_passed']}/{result['n_total']}")

    # AST Similarity
    print("\n2. AST SIMILARITY")
    print("-" * 40)
    ast = ASTSimilarityReward()
    ref = "def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)"
    sim = ast.compute_similarity(code, ref)
    print(f"  Similarity: {sim:.2%}")

    # Code Quality
    print("\n3. CODE QUALITY")
    print("-" * 40)
    cq = CodeQualityReward()
    result = cq.evaluate(code)
    print(f"  Score: {result['score']:.2%}")
    print(f"  Lines: {result['n_lines']}, Comments: {result['n_comments']}")
    print(f"  Complexity: {result['complexity']}")

    # Combined
    print("\n4. COMBINED REWARD")
    print("-" * 40)
    combined = CombinedCodeReward()
    result = combined.evaluate(code, test_cases, reference=ref)
    print(f"  Combined: {result['combined_score']:.3f}")
    print(f"  Execution: {result['execution_score']:.3f}")
    print(f"  AST: {result['ast_similarity']:.3f}")
    print(f"  Quality: {result['quality_score']:.3f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_code_reward_model()
