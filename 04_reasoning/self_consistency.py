"""
################################################################################
SELF-CONSISTENCY — SAMPLING MULTIPLE REASONING PATHS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Self-Consistency?
    Instead of generating one reasoning path, sample MULTIPLE paths
    with temperature, then vote on the most common answer. The key
    insight: correct reasoning paths converge on the same answer,
    while errors are diverse.

Why does it matter?
    Single-path reasoning is fragile — one mistake ruins everything.
    Self-consistency is robust because:
    - Multiple paths reduce variance
    - Correct answers appear more frequently
    - Errors in different paths cancel out
    - Simple to implement, big accuracy gain

How does it work?
    1. Sample N reasoning paths with temperature > 0
    2. Extract the final answer from each path
    3. Take a majority vote on the answers
    4. Return the most common answer

Results:
    GSM8K: CoT 58% → Self-Consistency 74% (+16%)
    ARC: CoT 73% → Self-Consistency 79% (+6%)
    Just by sampling 40 paths and voting!

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ Self-Consistency                                            │
    │                                                              │
    │  Question ──▶ Sample N paths with temperature               │
    │                  ↓    ↓    ↓    ↓                            │
    │               Path1 Path2 Path3 PathN                       │
    │                ↓     ↓     ↓     ↓                           │
    │              A₁    A₂    A₃    Aₙ   (extract answers)       │
    │                ↓     ↓     ↓     ↓                           │
    │              [Majority Vote] → Final Answer                  │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2022: Self-Consistency (Wang et al., Google Brain)
    - 2023: Universal Self-Consistency (use LLM to select)
    - 2024: Applied to code generation, math, planning
    - 2025: Combined with process reward models

INTERVIEW QUESTIONS:
    1. "What is Self-Consistency and why does it work?"
       Sample N reasoning paths, vote on the final answer. Works because
       correct reasoning converges (same answer) while errors diverge
       (different wrong answers). Majority vote picks the correct one.

    2. "How many paths should you sample?"
       More paths = better accuracy but higher cost. Typical: 5-40 paths.
       Diminishing returns past ~20 paths. Use adaptive: stop early if
       one answer has overwhelming majority.

    3. "What's the difference between Self-Consistency and Ensemble?"
       Ensemble uses different MODELS. Self-Consistency uses the SAME
       model with different SAMPLING. SC is cheaper (one model) and
       leverages the fact that LLMs have internalized multiple solutions.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from collections import Counter
import math

import sys
sys.path.append('..')
from ..01_math.probability import softmax, Categorical


################################################################################
# SECTION 1: CONFIGURATION
################################################################################

@dataclass
class ConsistencyConfig:
    """
    Self-Consistency Configuration.

    Attributes:
        num_samples: Number of reasoning paths to sample
        temperature: Sampling temperature (higher = more diverse)
        voting_method: How to aggregate answers ('majority', 'weighted')
        confidence_threshold: Early stopping threshold (0.0 = disabled)
        top_p: Nucleus sampling parameter
    """
    num_samples: int = 10
    temperature: float = 0.7
    voting_method: str = 'majority'
    confidence_threshold: float = 0.0
    top_p: float = 0.9


################################################################################
# SECTION 2: REASONING PATH
################################################################################

@dataclass
class ReasoningPath:
    """
    A single reasoning path with its answer and metadata.

    Attributes:
        steps: List of reasoning steps
        answer: Extracted final answer
        confidence: Confidence score for this path
        log_probability: Log probability of the generation
    """
    steps: List[str] = field(default_factory=list)
    answer: str = ""
    confidence: float = 0.0
    log_probability: float = 0.0

    def __post_init__(self):
        if not self.steps:
            self.steps = []


################################################################################
# SECTION 3: ANSWER EXTRACTION
################################################################################

class AnswerExtractor:
    """
    Extract the final answer from a reasoning chain.

    The answer is typically at the end of the reasoning, often
    preceded by phrases like "The answer is" or "Therefore".
    """

    # Common answer patterns
    ANSWER_PATTERNS = [
        "the answer is",
        "therefore,",
        "so the answer is",
        "in conclusion,",
        "the result is",
        "we get",
        "this gives us",
        "finally,",
    ]

    def extract(self, reasoning_text: str) -> str:
        """
        Extract answer from reasoning text.

        Args:
            reasoning_text: Full reasoning chain text

        Returns:
            Extracted answer string
        """
        text_lower = reasoning_text.lower()

        # Try to find answer after known patterns
        for pattern in self.ANSWER_PATTERNS:
            idx = text_lower.rfind(pattern)
            if idx != -1:
                answer = reasoning_text[idx + len(pattern):].strip()
                # Take first sentence
                if '.' in answer:
                    answer = answer[:answer.index('.')]
                return answer.strip()

        # Fallback: take last sentence
        sentences = reasoning_text.split('.')
        if len(sentences) > 1:
            return sentences[-2].strip() if len(sentences) > 2 else sentences[-1].strip()
        return reasoning_text.strip()[-50:]  # Last 50 chars


################################################################################
# SECTION 4: MAJORITY VOTING
################################################################################

class MajorityVoting:
    """
    Simple majority vote across sampled paths.

    Count how many times each answer appears, return the most common.

    Interview Question:
        "How does majority voting work in Self-Consistency?"
        Count the frequency of each unique answer across all sampled
        reasoning paths. The answer that appears most often wins.
        Simple but effective — correct answers naturally converge.
    """

    def vote(self, answers: List[str]) -> Tuple[str, float]:
        """
        Perform majority voting.

        Args:
            answers: List of extracted answers from each path

        Returns:
            Tuple of (winning_answer, confidence)
        """
        if not answers:
            return "", 0.0

        # Normalize answers for comparison
        normalized = [a.strip().lower() for a in answers]
        counter = Counter(normalized)

        # Get most common
        most_common_answer, count = counter.most_common(1)[0]
        confidence = count / len(answers)

        # Return original (not normalized) answer
        for orig in answers:
            if orig.strip().lower() == most_common_answer:
                return orig.strip(), confidence

        return most_common_answer, confidence


################################################################################
# SECTION 5: WEIGHTED VOTING
################################################################################

class WeightedVoting:
    """
    Weighted voting using path confidence/log-probability.

    Paths with higher confidence or log-probability get more weight
    in the vote. This accounts for the fact that some reasoning
    paths are more reliable than others.

    Interview Question:
        "How does weighted voting improve over majority voting?"
        Weighted voting accounts for path quality. A path with high
        confidence and clear reasoning should count more than a path
        with uncertain reasoning. Weight = f(confidence, log_prob).
    """

    def vote(self, answers: List[str], weights: List[float]) -> Tuple[str, float]:
        """
        Perform weighted voting.

        Args:
            answers: List of extracted answers
            weights: Weight for each answer (higher = more important)

        Returns:
            Tuple of (winning_answer, weighted_confidence)
        """
        if not answers:
            return "", 0.0

        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(answers)) / len(answers)

        # Aggregate weights per unique answer
        answer_weights: Dict[str, float] = {}
        for answer, weight in zip(answers, weights):
            key = answer.strip().lower()
            answer_weights[key] = answer_weights.get(key, 0.0) + weight

        # Select answer with highest aggregate weight
        best_answer = max(answer_weights, key=answer_weights.get)
        confidence = answer_weights[best_answer]

        # Return original answer
        for orig in answers:
            if orig.strip().lower() == best_answer:
                return orig.strip(), confidence

        return best_answer, confidence


################################################################################
# SECTION 6: SELF-CONSISTENCY (MAIN CLASS)
################################################################################

class SelfConsistency:
    """
    Self-Consistency Decoding — Sample multiple paths, vote on answer.

    Paper: "Self-Consistency Improves Chain of Thought Reasoning
            in Language Models" (Wang et al., ICLR 2023)

    Key Formula:
        P(answer | question) ≈ (1/N) Σᵢ 𝟚[answer_i = answer]
        where answer_i is extracted from the i-th sampled path

    Step by step:
        1. Given a question, sample N reasoning paths (temp > 0)
        2. Extract the final answer from each path
        3. Take majority vote (or weighted vote)
        4. Return the most common answer

    Interview Question:
        "Implement Self-Consistency from scratch."
        1. Sample N paths with temperature (diversity is key)
        2. Extract answers (parse "The answer is X")
        3. Count occurrences of each unique answer
        4. Return the most frequent one
        Cost: N× the cost of single CoT, but much more accurate.
    """

    def __init__(self, config: Optional[ConsistencyConfig] = None):
        """Initialize Self-Consistency."""
        self.config = config or ConsistencyConfig()
        self.extractor = AnswerExtractor()
        self.majority_voter = MajorityVoting()
        self.weighted_voter = WeightedVoting()

    def sample_path(self, question: str, path_id: int) -> ReasoningPath:
        """
        Sample a single reasoning path.

        Args:
            question: The question to reason about
            path_id: Identifier for this path

        Returns:
            ReasoningPath with steps and answer
        """
        # Simulate sampling with temperature
        # In production: model.generate(question, temperature=self.config.temperature)
        num_steps = np.random.randint(3, 8)
        steps = [f"Step {i+1}: Reasoning step {i+1} for path {path_id}"
                 for i in range(num_steps)]
        answer = f"answer_{path_id % 3}"  # Simulate convergence

        # Simulate log probability
        log_prob = -np.random.exponential(2.0)

        return ReasoningPath(
            steps=steps,
            answer=answer,
            confidence=max(0, 1.0 + log_prob / 10),
            log_probability=log_prob
        )

    def sample_paths(self, question: str) -> List[ReasoningPath]:
        """
        Sample N reasoning paths.

        Args:
            question: The question

        Returns:
            List of N ReasoningPath objects
        """
        paths = []
        for i in range(self.config.num_samples):
            path = self.sample_path(question, i)
            paths.append(path)
        return paths

    def extract_answers(self, paths: List[ReasoningPath]) -> List[str]:
        """Extract answers from all paths."""
        return [p.answer for p in paths]

    def vote(self, paths: List[ReasoningPath]) -> Tuple[str, float]:
        """
        Vote on the final answer.

        Args:
            paths: Sampled reasoning paths

        Returns:
            Tuple of (answer, confidence)
        """
        answers = self.extract_answers(paths)

        if self.config.voting_method == 'weighted':
            weights = [p.confidence for p in paths]
            return self.weighted_voter.vote(answers, weights)
        else:
            return self.majority_voter.vote(answers)

    def run(self, question: str) -> Dict:
        """
        Full Self-Consistency pipeline.

        Args:
            question: The question to answer

        Returns:
            Dictionary with answer, confidence, paths, agreement
        """
        # Sample paths
        paths = self.sample_paths(question)

        # Extract answers
        answers = self.extract_answers(paths)

        # Vote
        answer, confidence = self.vote(paths)

        # Compute agreement metrics
        answer_counts = Counter(a.strip().lower() for a in answers)
        unique_answers = len(answer_counts)

        return {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'num_paths': len(paths),
            'unique_answers': unique_answers,
            'answer_distribution': dict(answer_counts),
            'agreement_ratio': confidence
        }


################################################################################
# SECTION 7: ADAPTIVE SELF-CONSISTENCY
################################################################################

class AdaptiveSelfConsistency:
    """
    Adaptive Self-Consistency — Stop early if confidence is high enough.

    Instead of always sampling N paths, stop when one answer has
    overwhelming majority (e.g., 80% agreement).

    Interview Question:
        "How do you make Self-Consistency more efficient?"
        Use adaptive stopping: sample paths one at a time, track
        answer distribution, stop when one answer has >threshold
        agreement. This saves compute on easy questions while
        maintaining accuracy on hard ones.
    """

    def __init__(self, min_samples: int = 3, max_samples: int = 20,
                 confidence_threshold: float = 0.8):
        """
        Args:
            min_samples: Minimum paths before checking confidence
            max_samples: Maximum paths to sample
            confidence_threshold: Stop when one answer exceeds this
        """
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.confidence_threshold = confidence_threshold
        self.extractor = AnswerExtractor()
        self.voter = MajorityVoting()

    def run(self, question: str) -> Dict:
        """
        Run adaptive self-consistency.

        Args:
            question: The question to answer

        Returns:
            Dictionary with answer, paths sampled, early stop info
        """
        answers = []
        paths = []

        for i in range(self.max_samples):
            # Sample one more path
            num_steps = np.random.randint(3, 7)
            steps = [f"Step {j+1}" for j in range(num_steps)]
            answer = f"answer_{i % 3}"
            paths.append(ReasoningPath(steps=steps, answer=answer))
            answers.append(answer)

            # Check early stopping (after min samples)
            if i + 1 >= self.min_samples:
                _, confidence = self.voter.vote(answers)
                if confidence >= self.confidence_threshold:
                    return {
                        'question': question,
                        'answer': self.voter.vote(answers)[0],
                        'confidence': confidence,
                        'paths_sampled': i + 1,
                        'early_stopped': True,
                        'max_samples': self.max_samples
                    }

        # Reached max samples
        answer, confidence = self.voter.vote(answers)
        return {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'paths_sampled': self.max_samples,
            'early_stopped': False,
            'max_samples': self.max_samples
        }


################################################################################
# SECTION 8: TESTING & DEMONSTRATION
################################################################################

def demonstrate_self_consistency():
    """
    Demonstrate Self-Consistency methods.
    """
    print("=" * 70)
    print("SELF-CONSISTENCY DEMONSTRATION")
    print("=" * 70)

    # ── Answer Extraction ──
    print("\n1. ANSWER EXTRACTION")
    print("-" * 40)
    extractor = AnswerExtractor()
    texts = [
        "Step 1: Calculate 5 + 3 = 8. Step 2: 8 * 2 = 16. The answer is 16.",
        "I need to find the sum. 10 + 20 = 30. Therefore, the result is 30.",
        "First I add 7 and 3 to get 10. So the answer is 10."
    ]
    for text in texts:
        answer = extractor.extract(text)
        print(f"  Text: ...{text[-40:]}")
        print(f"  Answer: {answer}")

    # ── Majority Voting ──
    print("\n2. MAJORITY VOTING")
    print("-" * 40)
    voter = MajorityVoting()
    answers = ["16", "16", "18", "16", "16", "20", "16"]
    winner, conf = voter.vote(answers)
    print(f"  Answers: {answers}")
    print(f"  Winner: {winner} (confidence: {conf:.2%})")

    # ── Weighted Voting ──
    print("\n3. WEIGHTED VOTING")
    print("-" * 40)
    wvoter = WeightedVoting()
    answers = ["16", "18", "16"]
    weights = [0.9, 0.3, 0.8]
    winner, conf = wvoter.vote(answers, weights)
    print(f"  Answers: {answers}")
    print(f"  Weights: {weights}")
    print(f"  Winner: {winner} (confidence: {conf:.2%})")

    # ── Self-Consistency ──
    print("\n4. SELF-CONSISTENCY")
    print("-" * 40)
    config = ConsistencyConfig(num_samples=10, temperature=0.7)
    sc = SelfConsistency(config)
    result = sc.run("What is 15 + 27?")
    print(f"  Question: {result['question']}")
    print(f"  Answer: {result['answer']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Paths sampled: {result['num_paths']}")
    print(f"  Unique answers: {result['unique_answers']}")
    print(f"  Distribution: {result['answer_distribution']}")

    # ── Adaptive Self-Consistency ──
    print("\n5. ADAPTIVE SELF-CONSISTENCY")
    print("-" * 40)
    asc = AdaptiveSelfConsistency(min_samples=3, max_samples=20, confidence_threshold=0.7)
    result = asc.run("What is 8 * 7?")
    print(f"  Answer: {result['answer']}")
    print(f"  Paths sampled: {result['paths_sampled']}")
    print(f"  Early stopped: {result['early_stopped']}")
    print(f"  Confidence: {result['confidence']:.2%}")

    # ── Cost Analysis ──
    print("\n6. COST ANALYSIS")
    print("-" * 40)
    for n in [1, 5, 10, 20, 40]:
        config = ConsistencyConfig(num_samples=n)
        sc = SelfConsistency(config)
        result = sc.run("test question")
        print(f"  N={n:2d}: confidence={result['confidence']:.2%}, "
              f"cost={n}x single CoT")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_self_consistency()
