"""
################################################################################
VERIFICATION — ENSURING SYNTHETIC DATA QUALITY
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Data Verification?
    Checking that synthetic data is accurate, consistent, and useful
    for training. Without verification, synthetic data can contain
    hallucinations, contradictions, and noise that hurt training.

Why does it matter?
    Synthetic data without verification:
    - May contain factual errors (hallucinations)
    - Can be internally inconsistent
    - Might not match the target distribution
    - Could amplify biases

    Verification ensures:
    - Factual accuracy (for knowledge tasks)
    - Internal consistency (no contradictions)
    - Execution correctness (for code tasks)
    - Quality alignment (meets standards)

Key Methods:
    1. Execution-based: Run code to verify correctness
    2. Consistency: Check for contradictions
    3. Cross-reference: Compare with known facts
    4. Model-based: Use models to judge quality

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Verification Pipeline                            │
    │                                                                  │
    │  Synthetic Data ──▶ Execution Check ──▶ Pass? ──▶ Keep         │
    │                        │                                         │
    │                        ↓ Fail                                    │
    │                  Consistency Check ──▶ Pass? ──▶ Keep           │
    │                        │                                         │
    │                        ↓ Fail                                    │
    │                  Cross-Reference ──▶ Pass? ──▶ Keep             │
    │                        │                                         │
    │                        ↓ Fail                                    │
    │                  Reject                                          │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "How do you verify synthetic code?"
       Execute it! Run the code with test inputs and check outputs.
       Also check for syntax errors, edge cases, and style.

    2. "How do you verify factual claims?"
       Cross-reference with knowledge bases, use retrieval-augmented
       verification, or use multiple models for consensus.

    3. "What's the most important verification method?"
       For code: execution. For text: consistency checking.
       For reasoning: step-by-step verification.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: EXECUTION VERIFIER
################################################################################

class ExecutionVerifier:
    """
    Execution-Based Verifier
    ==========================

    Verifies code by executing it and checking outputs.

    Process:
    1. Parse the generated code
    2. Execute with test inputs
    3. Compare outputs with expected results
    4. Check for runtime errors

    This is the gold standard for code verification.

    Interview Question:
        "Is execution-based verification safe?"
        For untrusted code, no! Use sandboxing (Docker, containers)
        with resource limits (time, memory). For synthetic data
        generation, the code is typically simple and safe.
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def verify_function(
        self,
        code: str,
        test_cases: List[Dict]
    ) -> Tuple[bool, List[Dict]]:
        """
        Verify a function by running test cases.

        Args:
            code: Python code string
            test_cases: List of {'input': ..., 'expected': ...}

        Returns:
            (all_passed, results)
        """
        results = []

        try:
            # Create isolated namespace
            namespace = {}
            exec(code, namespace)

            for test in test_cases:
                try:
                    # Run the function
                    func_name = self._extract_function_name(code)
                    if func_name and func_name in namespace:
                        result = namespace[func_name](*test['input'])
                        passed = self._compare_outputs(result, test['expected'])
                        results.append({
                            'input': test['input'],
                            'expected': test['expected'],
                            'actual': result,
                            'passed': passed
                        })
                    else:
                        results.append({'error': 'Function not found', 'passed': False})
                except Exception as e:
                    results.append({'error': str(e), 'passed': False})

        except Exception as e:
            results.append({'error': f'Code execution failed: {e}', 'passed': False})

        all_passed = all(r.get('passed', False) for r in results)
        return all_passed, results

    def _extract_function_name(self, code: str) -> Optional[str]:
        """Extract function name from code."""
        for line in code.split('\n'):
            if line.strip().startswith('def '):
                return line.split('(')[0].replace('def ', '').strip()
        return None

    def _compare_outputs(self, actual, expected) -> bool:
        """Compare actual and expected outputs."""
        if isinstance(expected, (int, float)):
            return abs(actual - expected) < 1e-6
        elif isinstance(expected, list):
            return len(actual) == len(expected) and all(
                self._compare_outputs(a, e) for a, e in zip(actual, expected)
            )
        else:
            return actual == expected


################################################################################
# SECTION 2: CONSISTENCY CHECKER
################################################################################

class ConsistencyChecker:
    """
    Consistency Checker
    ====================

    Checks for internal contradictions in generated text.

    Types of consistency:
    1. Logical: No contradictory statements
    2. Temporal: Events in correct order
    3. Numerical: Numbers are consistent
    4. Entity: Same entity referenced consistently

    Method:
    - Extract claims from text
    - Check for contradictions between claims
    - Flag inconsistencies for review

    Interview Question:
        "How do you check consistency in long texts?"
        Break into claims, create an entailment graph,
        check for contradictions. For numerical claims,
        extract and verify arithmetic consistency.
    """

    def __init__(self):
        pass

    def check_logical_consistency(
        self,
        claims: List[str]
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Check for logical contradictions between claims.

        Simplified: Check for negation patterns.

        Args:
            claims: List of claim strings

        Returns:
            (is_consistent, list of contradictory pairs)
        """
        contradictions = []

        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                if self._are_contradictory(claims[i], claims[j]):
                    contradictions.append((i, j))

        return len(contradictions) == 0, contradictions

    def _are_contradictory(self, claim1: str, claim2: str) -> bool:
        """Check if two claims contradict each other (simplified)."""
        # Simple heuristic: check for negation
        negations = ["not", "no", "never", "isn't", "doesn't", "won't"]

        c1_lower = claim1.lower()
        c2_lower = claim2.lower()

        for neg in negations:
            # If one has negation and other doesn't, might be contradictory
            if neg in c1_lower and neg not in c2_lower:
                # Check if they're about the same topic
                words1 = set(c1_lower.split())
                words2 = set(c2_lower.split())
                overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                if overlap > 0.5:
                    return True

        return False

    def check_numerical_consistency(
        self,
        numbers: List[float],
        tolerance: float = 0.01
    ) -> Tuple[bool, List[str]]:
        """
        Check numerical consistency.

        Args:
            numbers: List of numbers to check
            tolerance: Allowed deviation

        Returns:
            (is_consistent, list of issues)
        """
        issues = []

        # Check for duplicates that should be same
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                if abs(numbers[i] - numbers[j]) < tolerance:
                    if numbers[i] != numbers[j]:
                        issues.append(
                            f"Numbers {numbers[i]} and {numbers[j]} are "
                            f"close but not equal"
                        )

        return len(issues) == 0, issues


################################################################################
# SECTION 3: CROSS-REFERENCE VERIFIER
################################################################################

class CrossReferenceVerifier:
    """
    Cross-Reference Verifier
    ==========================

    Verifies facts by cross-referencing with knowledge bases.

    Process:
    1. Extract factual claims from generated text
    2. Look up claims in knowledge base
    3. Flag unsupported claims

    This catches hallucinations where models generate plausible
    but factually incorrect statements.

    Interview Question:
        "How do you handle the knowledge base coverage gap?"
        Not all facts are in the knowledge base. Use confidence
        scoring: high-confidence matches are verified, low-confidence
        are flagged for human review.
    """

    def __init__(self, knowledge_base: Optional[Dict[str, str]] = None):
        self.knowledge_base = knowledge_base or {}

    def add_knowledge(self, key: str, value: str):
        """Add a fact to the knowledge base."""
        self.knowledge_base[key.lower()] = value

    def verify_claim(self, claim: str) -> Tuple[bool, float]:
        """
        Verify a claim against the knowledge base.

        Args:
            claim: Factual claim to verify

        Returns:
            (is_verified, confidence)
        """
        claim_lower = claim.lower()

        # Search for matching knowledge
        best_match = None
        best_score = 0.0

        for key, value in self.knowledge_base.items():
            # Simple word overlap scoring
            claim_words = set(claim_lower.split())
            key_words = set(key.split())

            overlap = len(claim_words & key_words) / max(len(claim_words | key_words), 1)

            if overlap > best_score:
                best_score = overlap
                best_match = value

        if best_score > 0.5:
            return True, best_score
        else:
            return False, best_score

    def verify_batch(self, claims: List[str]) -> List[Tuple[bool, float]]:
        """Verify multiple claims."""
        return [self.verify_claim(claim) for claim in claims]


################################################################################
# SECTION 4: REWARD MODEL FILTER
################################################################################

class RewardModelFilter:
    """
    Reward Model Filter
    ====================

    Uses a reward model to score and filter synthetic data.

    Process:
    1. Generate synthetic samples
    2. Score each with reward model
    3. Keep only high-scoring samples

    This is similar to RLHF but applied to data filtering.

    Interview Question:
        "How does reward model filtering compare to rule-based?"
        Reward models are more flexible and can catch subtle issues
        that rules miss. But they can also have biases. Best
        practice: combine both approaches.
    """

    def __init__(self, d_model: int = 32):
        self.d_model = d_model

        # Simple reward model
        self.reward_weights = np.random.randn(d_model) * 0.02

    def score(self, features: np.ndarray) -> float:
        """
        Score a sample using the reward model.

        Args:
            features: Feature vector of the sample

        Returns:
            Reward score (higher is better)
        """
        return float(features @ self.reward_weights)

    def filter_batch(
        self,
        samples: List[Dict],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Filter a batch of samples by reward score.

        Args:
            samples: List of sample dictionaries with 'features'
            threshold: Minimum reward score

        Returns:
            Filtered samples above threshold
        """
        filtered = []

        for sample in samples:
            if 'features' in sample:
                score = self.score(sample['features'])
                sample['reward_score'] = score
                if score >= threshold:
                    filtered.append(sample)

        return filtered


################################################################################
# SECTION 5: DATA VERIFIER (COMPLETE)
################################################################################

class DataVerifier:
    """
    Complete Data Verification Pipeline
    ======================================

    Combines all verification methods for comprehensive checking.

    Verification levels:
    1. Execution: For code (gold standard)
    2. Consistency: For logical coherence
    3. Cross-reference: For factual accuracy
    4. Reward model: For overall quality

    Interview Questions:
        1. "What's the verification pass rate for synthetic data?"
           Typically 60-80% passes all checks. The rest is filtered.
           Higher thresholds = cleaner but smaller dataset.

        2. "How do you handle verification failures?"
           Either discard the sample or attempt to fix it
           (regenerate, edit). For critical applications,
           always discard. For training, discarding is safer.

        3. "Can verification be automated completely?"
           For code: yes (execution). For text: partially.
           Some cases need human review, especially for
           nuanced factual claims.
    """

    def __init__(self):
        self.execution_verifier = ExecutionVerifier()
        self.consistency_checker = ConsistencyChecker()
        self.cross_ref_verifier = CrossReferenceVerifier()
        self.reward_filter = RewardModelFilter()

    def verify_sample(self, sample: Dict) -> Dict[str, any]:
        """
        Run all verification checks on a sample.

        Args:
            sample: Sample dictionary with content and metadata

        Returns:
            Verification results
        """
        results = {
            'execution': None,
            'consistency': None,
            'cross_reference': None,
            'reward_score': None,
            'overall_passed': False
        }

        # Execution check (if code)
        if 'code' in sample:
            passed, exec_results = self.execution_verifier.verify_function(
                sample['code'],
                sample.get('test_cases', [])
            )
            results['execution'] = passed

        # Consistency check
        if 'claims' in sample:
            consistent, contradictions = self.consistency_checker.check_logical_consistency(
                sample['claims']
            )
            results['consistency'] = consistent

        # Cross-reference check
        if 'factual_claims' in sample:
            for claim in sample['factual_claims']:
                verified, confidence = self.cross_ref_verifier.verify_claim(claim)
                if not verified and confidence > 0.3:
                    results['cross_reference'] = False
                    break
            if results['cross_reference'] is None:
                results['cross_reference'] = True

        # Overall: pass if all checks pass (or not applicable)
        checks = [v for v in results.values() if v is not None and v is not True]
        results['overall_passed'] = len(checks) == 0

        return results

    def verify_batch(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Verify a batch of samples.

        Args:
            samples: List of samples

        Returns:
            (passed_samples, failed_samples)
        """
        passed = []
        failed = []

        for sample in samples:
            results = self.verify_sample(sample)
            sample['verification'] = results

            if results['overall_passed']:
                passed.append(sample)
            else:
                failed.append(sample)

        return passed, failed


################################################################################
# SECTION 6: TESTING & DEMONSTRATION
################################################################################

def demonstrate_verification():
    """Demonstrate verification pipeline."""
    print("=" * 70)
    print("DATA VERIFICATION")
    print("=" * 70)

    # Create verifier
    verifier = DataVerifier()

    # Execution verification
    print("\n--- Execution Verification ---")
    code = """
def add(a, b):
    return a + b
"""
    test_cases = [
        {'input': (1, 2), 'expected': 3},
        {'input': (5, 5), 'expected': 10},
        {'input': (-1, 1), 'expected': 0},
    ]

    passed, results = verifier.execution_verifier.verify_function(code, test_cases)
    print(f"  Code: {code.strip()}")
    print(f"  All tests passed: {passed}")
    for r in results:
        print(f"    {r['input']} → {r['actual']} (expected {r['expected']}): {'✓' if r['passed'] else '✗'}")

    # Consistency checking
    print("\n--- Consistency Checking ---")
    consistent_claims = [
        "The sky is blue",
        "Water boils at 100°C",
        "The Earth orbits the Sun"
    ]
    inconsistent_claims = [
        "The sky is blue",
        "The sky is not blue",
        "Water boils at 100°C"
    ]

    is_cons, contradictions = verifier.consistency_checker.check_logical_consistency(consistent_claims)
    print(f"  Consistent claims: {is_cons}")

    is_cons2, contradictions2 = verifier.consistency_checker.check_logical_consistency(inconsistent_claims)
    print(f"  Inconsistent claims: {is_cons2}")
    if contradictions2:
        print(f"  Contradictions found: {contradictions2}")

    # Cross-reference verification
    print("\n--- Cross-Reference Verification ---")
    verifier.cross_ref_verifier.add_knowledge("the earth is round", "fact_1")
    verifier.cross_ref_verifier.add_knowledge("water boils at 100 degrees", "fact_2")

    claims = ["The earth is round", "The moon is made of cheese"]
    for claim in claims:
        verified, confidence = verifier.cross_ref_verifier.verify_claim(claim)
        print(f"  '{claim}': verified={verified}, confidence={confidence:.2f}")

    # Full verification pipeline
    print("\n--- Full Verification Pipeline ---")
    samples = [
        {
            'code': 'def multiply(a, b): return a * b',
            'test_cases': [{'input': (3, 4), 'expected': 12}],
            'claims': ['Multiplication is commutative']
        },
        {
            'code': 'def bad_add(a, b): return a - b',
            'test_cases': [{'input': (1, 2), 'expected': 3}],
            'claims': ['Addition increases values']
        }
    ]

    passed, failed = verifier.verify_batch(samples)
    print(f"  Samples: {len(samples)}")
    print(f"  Passed: {len(passed)}")
    print(f"  Failed: {len(failed)}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Verification is essential for synthetic data quality!")
    print("Execution-based verification is the gold standard for code.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_verification()
