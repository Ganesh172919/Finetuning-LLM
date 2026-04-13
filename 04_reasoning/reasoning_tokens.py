"""
################################################################################
REASONING TOKENS — HIDDEN THINKING BEFORE ANSWERING (o1/R1 STYLE)
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Reasoning Tokens?
    Reasoning tokens are hidden thinking steps that the model generates
    BEFORE producing the final answer. The user sees only the answer,
    but the model has done extensive internal reasoning.

    This is the approach behind OpenAI o1/o3 and DeepSeek R1.

Why does it matter?
    Traditional CoT shows reasoning to the user, which:
    - Can be misleading (plausible but wrong reasoning)
    - Is verbose (users want answers, not process)
    - Can't be rewarded during training (reasoning is visible)

    Hidden reasoning tokens enable:
    - Private reasoning (model thinks internally)
    - Test-time compute scaling (think longer = better)
    - Reward-based training (reward correct answers, not reasoning)
    - Budget control (limit thinking for efficiency)

How does it work?
    1. Model generates "thinking tokens" in a special channel
    2. These tokens are NOT shown to the user
    3. Model can backtrack, verify, explore in thinking
    4. When thinking is done, model generates visible answer
    5. Training uses RL to reward correct answers

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ Reasoning Token Model                                       │
    │                                                              │
    │  Prompt ──▶ <think>...reasoning...</think>              │
    │              ├── Generate thinking tokens (hidden)           │
    │              ├── Explore, backtrack, verify                  │
    │              └── When confident: close <think>               │
    │                                                              │
    │           ──▶ <answer>...response...</answer>               │
    │              └── Generate visible answer                     │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2024: OpenAI o1 — first reasoning model with hidden thinking
    - 2024: o1-mini, o3 — reasoning model family
    - 2025: DeepSeek R1 — open-source reasoning with RL training
    - 2025: Budget forcing — control thinking length
    - 2026: Hybrid models — switch between thinking and fast mode

INTERVIEW QUESTIONS:
    1. "How do reasoning tokens work in o1/R1?"
       The model generates hidden "thinking tokens" before the answer.
       These tokens are not shown to the user but allow the model to
       reason internally. Training uses RL (GRPO) to reward correct
       answers, so the model learns to think effectively.

    2. "How do you control thinking length?"
       Budget forcing: set max/min thinking tokens. Too short = poor
       reasoning. Too long = waste compute. Adaptive: stop when
       confidence exceeds threshold. DeepSeek R1 uses a "thinking
       budget" that scales with problem difficulty.

    3. "How do you train reasoning tokens?"
       Use RL (GRPO/PPO) with rule-based rewards:
       - Correctness: did the answer match ground truth?
       - Format: did the model use <think> correctly?
       - Efficiency: penalize excessive thinking
       No labeled reasoning data needed — just correct answers.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import math

import sys
sys.path.append('..')
from ..02_transformers.model import TransformerLM
from ..02_transformers.layers import TransformerBlock


################################################################################
# SECTION 1: CONFIGURATION
################################################################################

@dataclass
class ReasoningConfig:
    """
    Reasoning Token Model Configuration.

    Attributes:
        max_thinking_tokens: Maximum tokens for internal reasoning
        min_thinking_tokens: Minimum tokens before allowing answer
        thinking_temperature: Temperature for thinking (higher = more exploratory)
        answer_temperature: Temperature for answer (lower = more precise)
        budget_forcing: Whether to enforce budget limits
        verify_reasoning: Whether to check reasoning consistency
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
    """
    max_thinking_tokens: int = 100
    min_thinking_tokens: int = 10
    thinking_temperature: float = 0.8
    answer_temperature: float = 0.3
    budget_forcing: bool = True
    verify_reasoning: bool = True
    vocab_size: int = 1000
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4


################################################################################
# SECTION 2: THINKING TOKEN MODEL
################################################################################

class ThinkingTokenModel:
    """
    Model that generates hidden reasoning before answering.

    This implements the core mechanism of o1/R1-style reasoning:
    1. Generate thinking tokens (hidden from user)
    2. Generate answer tokens (visible to user)

    Key Insight:
        By training with RL on answer correctness, the model learns
        to reason effectively in its thinking channel. The thinking
        is not supervised — only the answer matters for training.

    Step by step:
        1. Encode prompt into hidden states
        2. Generate thinking tokens autoregressively
           - Use higher temperature for exploration
           - Can backtrack, verify, explore alternatives
        3. When thinking is complete, generate answer
           - Use lower temperature for precision
           - Answer is what the user sees
        4. Return only the answer (thinking is hidden)

    Interview Question:
        "How does the ThinkingTokenModel work?"
        The model has two modes: thinking and answering. In thinking mode,
        it generates tokens with high temperature to explore reasoning
        paths. These tokens are hidden. In answering mode, it generates
        the final response with low temperature for precision. Training
        uses RL to reward correct answers, so the model learns to think
        effectively.
    """

    def __init__(self, config: Optional[ReasoningConfig] = None):
        """
        Initialize the Thinking Token Model.

        Args:
            config: Model configuration
        """
        self.config = config or ReasoningConfig()
        self.thinking_history: List[str] = []
        self.answer_history: List[str] = []

        # Special tokens
        self.THINK_START = "<think>"
        self.THINK_END = "</think>"
        self.ANSWER_START = "<answer>"
        self.ANSWER_END = "</answer>"

        # Simulated vocabulary
        self.thinking_vocab = [
            "let me think", "consider", "alternatively", "wait",
            "actually", "therefore", "because", "if we",
            "the key insight", "looking at this", "breaking down",
            "step 1", "step 2", "checking", "verifying",
            "this seems right", "let me reconsider", "on second thought"
        ]

        self.answer_vocab = [
            "the answer is", "therefore", "in conclusion",
            "the result is", "we find that", "it turns out"
        ]

    def generate_thinking_token(self, context: str) -> str:
        """
        Generate a single thinking token.

        Args:
            context: Previous thinking context

        Returns:
            Next thinking token
        """
        # Simulate thinking token generation
        # In production: model.generate(context, temperature=thinking_temp)
        probs = np.random.dirichlet(np.ones(len(self.thinking_vocab)))
        probs = probs ** (1.0 / self.config.thinking_temperature)
        probs = probs / probs.sum()
        idx = np.random.choice(len(self.thinking_vocab), p=probs)
        return self.thinking_vocab[idx]

    def generate_answer_token(self, context: str) -> str:
        """
        Generate a single answer token.

        Args:
            context: Previous answer context

        Returns:
            Next answer token
        """
        probs = np.random.dirichlet(np.ones(len(self.answer_vocab)))
        probs = probs ** (1.0 / self.config.answer_temperature)
        probs = probs / probs.sum()
        idx = np.random.choice(len(self.answer_vocab), p=probs)
        return self.answer_vocab[idx]

    def think(self, prompt: str) -> List[str]:
        """
        Generate hidden reasoning chain.

        Args:
            prompt: Input prompt

        Returns:
            List of thinking tokens (hidden from user)
        """
        thinking_tokens = []
        context = prompt

        for i in range(self.config.max_thinking_tokens):
            token = self.generate_thinking_token(context)
            thinking_tokens.append(token)
            context += " " + token

            # Early stopping if confident
            if i >= self.config.min_thinking_tokens:
                if np.random.random() < 0.1:  # 10% chance of stopping
                    break

        self.thinking_history = thinking_tokens
        return thinking_tokens

    def answer(self, prompt: str, thinking: List[str]) -> str:
        """
        Generate visible answer using thinking context.

        Args:
            prompt: Input prompt
            thinking: Hidden thinking tokens

        Returns:
            Visible answer string
        """
        # Combine prompt + thinking as context
        thinking_str = " ".join(thinking)
        context = f"{prompt} {self.THINK_START}{thinking_str}{self.THINK_END}"

        # Generate answer tokens
        answer_tokens = []
        for i in range(5):  # Short answer
            token = self.generate_answer_token(context)
            answer_tokens.append(token)
            context += " " + token

        answer = " ".join(answer_tokens)
        self.answer_history = answer_tokens
        return answer

    def forward(self, prompt: str) -> Dict:
        """
        Full think → answer pipeline.

        Args:
            prompt: Input prompt

        Returns:
            Dictionary with thinking (hidden), answer (visible), metadata
        """
        # Step 1: Think (hidden)
        thinking = self.think(prompt)

        # Step 2: Answer (visible)
        visible_answer = self.answer(prompt, thinking)

        return {
            'prompt': prompt,
            'thinking_tokens': thinking,  # Hidden in production
            'num_thinking_tokens': len(thinking),
            'answer': visible_answer,  # Visible to user
            'thinking_temperature': self.config.thinking_temperature,
            'answer_temperature': self.config.answer_temperature
        }


################################################################################
# SECTION 3: BUDGET FORCING
################################################################################

class BudgetForcing:
    """
    Control thinking length via budget forcing.

    Strategies:
    1. Max budget: Truncate thinking at token limit
    2. Min budget: Force minimum thinking before allowing answer
    3. Confidence-based: Stop when reasoning confidence > threshold

    Interview Question:
        "How do you control how long a reasoning model thinks?"
        Budget forcing: (1) Set max_thinking_tokens to cap compute,
        (2) Set min_thinking_tokens to ensure minimum reasoning,
        (3) Use confidence threshold for adaptive stopping,
        (4) Scale budget with problem difficulty.
    """

    def __init__(self, min_budget: int = 10, max_budget: int = 100,
                 confidence_threshold: float = 0.9):
        """
        Args:
            min_budget: Minimum thinking tokens
            max_budget: Maximum thinking tokens
            confidence_threshold: Stop when confidence exceeds this
        """
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.confidence_threshold = confidence_threshold

    def enforce_budget(self, thinking_tokens: List[str],
                       confidence: float) -> Tuple[List[str], str]:
        """
        Enforce budget on thinking tokens.

        Args:
            thinking_tokens: Generated thinking tokens
            confidence: Current reasoning confidence

        Returns:
            Tuple of (trimmed_tokens, stop_reason)
        """
        # Max budget: truncate
        if len(thinking_tokens) > self.max_budget:
            return thinking_tokens[:self.max_budget], "max_budget"

        # Min budget: ensure minimum
        if len(thinking_tokens) < self.min_budget:
            # Pad with continuation tokens
            while len(thinking_tokens) < self.min_budget:
                thinking_tokens.append("continuing reasoning...")
            return thinking_tokens, "min_budget_padding"

        # Confidence-based: stop if confident
        if confidence >= self.confidence_threshold:
            return thinking_tokens, "confidence_threshold"

        return thinking_tokens, "natural_stop"

    def compute_confidence(self, tokens: List[str]) -> float:
        """
        Estimate confidence from thinking tokens.

        Args:
            tokens: Thinking tokens

        Returns:
            Confidence score between 0 and 1
        """
        # Simulate confidence estimation
        # In production: use model's internal confidence
        base = 0.3
        # More tokens = slightly more confident (diminishing returns)
        token_bonus = min(0.4, len(tokens) * 0.02)
        # Add noise
        noise = np.random.normal(0, 0.05)
        return np.clip(base + token_bonus + noise, 0.0, 1.0)


################################################################################
# SECTION 4: REASONING VERIFIER
################################################################################

class ReasoningVerifier:
    """
    Verify reasoning quality before returning answer.

    Checks:
    1. Logical consistency — no contradictions
    2. Step completeness — no skipped steps
    3. Answer-reasoning alignment — answer follows from reasoning

    Interview Question:
        "How do you verify reasoning quality?"
        Check: (1) Logical consistency — no contradictions in reasoning,
        (2) Completeness — all necessary steps present,
        (3) Alignment — the answer actually follows from the reasoning,
        (4) Correctness — each step is factually correct.
    """

    def check_consistency(self, tokens: List[str]) -> Dict:
        """
        Check for logical contradictions in reasoning.

        Args:
            tokens: Thinking tokens

        Returns:
            Dictionary with consistency score and issues
        """
        # Simulate consistency checking
        issues = []
        score = 1.0

        # Check for contradiction keywords
        contradiction_pairs = [
            ("yes", "no"), ("true", "false"),
            ("increase", "decrease"), ("positive", "negative")
        ]

        token_text = " ".join(tokens).lower()
        for pos, neg in contradiction_pairs:
            if pos in token_text and neg in token_text:
                issues.append(f"Potential contradiction: '{pos}' and '{neg}'")
                score -= 0.2

        return {
            'consistent': score > 0.5,
            'score': max(0, score),
            'issues': issues
        }

    def check_completeness(self, tokens: List[str]) -> Dict:
        """
        Check if reasoning has all necessary steps.

        Args:
            tokens: Thinking tokens

        Returns:
            Dictionary with completeness score
        """
        # Check for reasoning structure
        token_text = " ".join(tokens).lower()
        has_steps = any(f"step {i}" in token_text for i in range(1, 5))
        has_conclusion = any(w in token_text for w in ['therefore', 'thus', 'so', 'conclusion'])

        score = 0.5
        if has_steps:
            score += 0.25
        if has_conclusion:
            score += 0.25

        return {
            'complete': score > 0.7,
            'score': score,
            'has_steps': has_steps,
            'has_conclusion': has_conclusion
        }

    def verify(self, thinking_tokens: List[str], answer: str) -> Dict:
        """
        Full verification of reasoning and answer.

        Args:
            thinking_tokens: Hidden thinking tokens
            answer: Visible answer

        Returns:
            Dictionary with verification results
        """
        consistency = self.check_consistency(thinking_tokens)
        completeness = self.check_completeness(thinking_tokens)

        overall_score = (consistency['score'] + completeness['score']) / 2

        return {
            'consistency': consistency,
            'completeness': completeness,
            'overall_score': overall_score,
            'verified': overall_score > 0.6,
            'recommendation': 'accept' if overall_score > 0.6 else 'regenerate'
        }


################################################################################
# SECTION 5: HIDDEN CHAIN OF THOUGHT
################################################################################

class HiddenChainOfThought:
    """
    Hidden Chain of Thought — The core mechanism.

    Model generates tokens in "thinking mode" that are not shown to user.
    Only the final answer is visible.

    This is the fundamental mechanism behind o1/o3/R1:
    - Thinking tokens are generated with high temperature (exploration)
    - Answer tokens are generated with low temperature (precision)
    - Training rewards correct answers (RL), not reasoning quality

    Interview Question:
        "What is Hidden Chain of Thought?"
        The model generates reasoning tokens that are hidden from the user.
        These tokens allow the model to think through the problem internally.
        Only the final answer is shown. This enables test-time compute scaling:
        more thinking tokens = better answers on hard problems.
    """

    def __init__(self, config: Optional[ReasoningConfig] = None):
        """Initialize Hidden CoT."""
        self.config = config or ReasoningConfig()
        self.model = ThinkingTokenModel(config)
        self.budget = BudgetForcing(
            min_budget=config.min_thinking_tokens,
            max_budget=config.max_thinking_tokens
        )
        self.verifier = ReasoningVerifier()

    def forward(self, prompt: str) -> Dict:
        """
        Full hidden CoT pipeline.

        Args:
            prompt: Input prompt

        Returns:
            Dictionary with answer (visible) and metadata
        """
        # Generate thinking tokens
        thinking = self.model.think(prompt)

        # Compute confidence
        confidence = self.budget.compute_confidence(thinking)

        # Enforce budget
        thinking, stop_reason = self.budget.enforce_budget(thinking, confidence)

        # Generate answer
        answer = self.model.answer(prompt, thinking)

        # Verify reasoning
        verification = self.verifier.verify(thinking, answer)

        return {
            'prompt': prompt,
            'answer': answer,  # Visible to user
            'num_thinking_tokens': len(thinking),
            'confidence': confidence,
            'stop_reason': stop_reason,
            'verification': verification,
            'thinking': thinking  # Hidden in production
        }


################################################################################
# SECTION 6: TESTING & DEMONSTRATION
################################################################################

def demonstrate_reasoning_tokens():
    """
    Demonstrate reasoning token methods.
    """
    print("=" * 70)
    print("REASONING TOKENS DEMONSTRATION")
    print("=" * 70)

    config = ReasoningConfig(
        max_thinking_tokens=50,
        min_thinking_tokens=5,
        thinking_temperature=0.8,
        answer_temperature=0.3
    )

    # ── Thinking Token Model ──
    print("\n1. THINKING TOKEN MODEL")
    print("-" * 40)
    model = ThinkingTokenModel(config)
    result = model.forward("What is 23 * 47?")
    print(f"  Prompt: {result['prompt']}")
    print(f"  Thinking tokens: {result['num_thinking_tokens']}")
    print(f"  Thinking preview: {' '.join(result['thinking_tokens'][:5])}...")
    print(f"  Answer: {result['answer']}")

    # ── Budget Forcing ──
    print("\n2. BUDGET FORCING")
    print("-" * 40)
    budget = BudgetForcing(min_budget=5, max_budget=30)
    thinking = ["step1", "step2", "step3", "step4", "step5", "step6"]
    confidence = 0.85
    trimmed, reason = budget.enforce_budget(thinking, confidence)
    print(f"  Original tokens: {len(thinking)}")
    print(f"  Trimmed tokens: {len(trimmed)}")
    print(f"  Stop reason: {reason}")

    # ── Reasoning Verifier ──
    print("\n3. REASONING VERIFIER")
    print("-" * 40)
    verifier = ReasoningVerifier()
    thinking = ["step 1: analyze", "step 2: calculate", "therefore the answer is 1081"]
    verification = verifier.verify(thinking, "1081")
    print(f"  Consistent: {verification['consistency']['consistent']}")
    print(f"  Complete: {verification['completeness']['complete']}")
    print(f"  Overall: {verification['overall_score']:.3f}")
    print(f"  Recommendation: {verification['recommendation']}")

    # ── Hidden Chain of Thought ──
    print("\n4. HIDDEN CHAIN OF THOUGHT")
    print("-" * 40)
    hcot = HiddenChainOfThought(config)
    result = hcot.forward("Solve: 15 * 23 + 7")
    print(f"  Prompt: {result['prompt']}")
    print(f"  Answer (visible): {result['answer']}")
    print(f"  Thinking tokens (hidden): {result['num_thinking_tokens']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Stop reason: {result['stop_reason']}")
    print(f"  Verified: {result['verification']['verified']}")

    # ── Budget Comparison ──
    print("\n5. BUDGET COMPARISON")
    print("-" * 40)
    for budget_size in [5, 10, 20, 50]:
        cfg = ReasoningConfig(max_thinking_tokens=budget_size, min_thinking_tokens=2)
        model = ThinkingTokenModel(cfg)
        result = model.forward("Complex math problem")
        print(f"  Budget={budget_size:3d}: tokens={result['num_thinking_tokens']:3d}, "
              f"answer='{result['answer'][:30]}...'")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_reasoning_tokens()
