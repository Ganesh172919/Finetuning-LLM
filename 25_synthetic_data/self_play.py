"""
################################################################################
SELF-PLAY — MODELS IMPROVING THROUGH SELF-INTERACTION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Self-Play?
    A training paradigm where a model improves by playing against itself.
    The model generates data, evaluates it, and uses the evaluation
    to improve — creating a virtuous cycle of self-improvement.

Why does it matter?
    Traditional training requires:
    - Human annotators (expensive)
    - External datasets (limited)
    - Reward models (need training data)

    Self-play enables:
    - Continuous self-improvement without human data
    - Automatic curriculum (gets harder as model improves)
    - Discovery of novel strategies
    - Scalable training

How does it work?
    1. Model generates multiple responses
    2. Responses are evaluated (by model itself or rules)
    3. Best responses become training data
    4. Model is fine-tuned on its own best outputs
    5. Repeat — model improves each iteration

Key Papers:
    - SPIN (2024): Self-Play Fine-Tuning for LLMs
    - Self-Play (DeepMind, 2023): Improves reasoning
    - AlphaGo/AlphaZero: Self-play for game AI

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Self-Play Training Loop                      │
    │                                                                  │
    │  Model (iteration t) ──▶ Generate responses                     │
    │         ↓                                                        │
    │  Evaluate responses (model-as-judge or rules)                   │
    │         ↓                                                        │
    │  Select best responses ──▶ Training data                        │
    │         ↓                                                        │
    │  Fine-tune model ──▶ Model (iteration t+1)                      │
    │         │                                                        │
    │         └──────────▶ Repeat until convergence                   │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "What is self-play in the context of LLMs?"
       Self-play is when a model generates its own training data
       by producing multiple responses, evaluating them, and
       fine-tuning on the best ones. It's like a student practicing
       problems and learning from their best attempts.

    2. "How does self-play avoid degeneration?"
       Key strategies: (a) Keep a reference model to prevent drift,
       (b) Use diverse prompts, (c) Evaluate with external metrics,
       (d) Mix real and synthetic data.

    3. "What's the difference between self-play and RLHF?"
       RLHF uses human preferences to train a reward model.
       Self-play uses the model's own outputs and evaluations.
       Self-play is more scalable but can be less aligned.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: CONFIGURATION
################################################################################

@dataclass
class SelfPlayConfig:
    """
    Configuration for Self-Play Training.
    """
    # Model parameters
    vocab_size: int = 1000
    d_model: int = 128
    seq_len: int = 64

    # Self-play parameters
    num_iterations: int = 5
    responses_per_prompt: int = 4
    top_k_select: int = 1  # Keep top-K responses

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 16

    # Reference model (prevents drift)
    use_reference: bool = True
    kl_penalty: float = 0.1


################################################################################
# SECTION 2: RESPONSE GENERATOR
################################################################################

class ResponseGenerator:
    """
    Response Generator
    ===================

    Generates multiple responses for a given prompt.
    Used in self-play to create candidate outputs.

    Strategies:
    - Temperature sampling: Diverse responses
    - Top-k/top-p sampling: Controlled diversity
    - Beam search: High-quality but less diverse

    Interview Question:
        "How do you ensure diversity in self-play responses?"
        Use different sampling strategies: high temperature for
        diversity, top-k for quality. Also use different random
        seeds. The goal is to explore the response space broadly.
    """

    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Simple language model (logits from embedding)
        self.embed = np.random.randn(vocab_size, d_model) * 0.02
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.02

    def generate(
        self,
        prompt_tokens: List[int],
        max_len: int = 32,
        temperature: float = 1.0,
        num_responses: int = 1
    ) -> List[List[int]]:
        """
        Generate multiple responses for a prompt.

        Args:
            prompt_tokens: Input prompt as token list
            max_len: Maximum response length
            temperature: Sampling temperature
            num_responses: Number of responses to generate

        Returns:
            List of response token lists
        """
        responses = []

        for _ in range(num_responses):
            tokens = list(prompt_tokens)

            for _ in range(max_len):
                # Get embeddings for current sequence
                emb = self.embed[tokens[-1:]]

                # Predict next token logits
                logits = emb @ self.output_proj

                # Apply temperature
                logits = logits / temperature

                # Sample from distribution
                probs = self._softmax(logits[0])
                next_token = np.random.choice(self.vocab_size, p=probs)

                tokens.append(next_token)

                # Stop at end token (simplified: token 0)
                if next_token == 0:
                    break

            responses.append(tokens[len(prompt_tokens):])

        return responses

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)


################################################################################
# SECTION 3: RESPONSE EVALUATOR
################################################################################

class ResponseEvaluator:
    """
    Response Evaluator (Model-as-Judge)
    =====================================

    Evaluates response quality for self-play.
    Can use:
    - Rule-based scoring (length, format, keywords)
    - Model-as-judge (model evaluates its own outputs)
    - External metrics (execution, consistency)

    Interview Question:
        "Can a model reliably evaluate its own outputs?"
        Partially. Models can catch obvious errors but may miss
        subtle issues. Best practice: combine model evaluation
        with rule-based checks and external validation.
    """

    def __init__(self, d_model: int):
        self.d_model = d_model

        # Scoring network
        self.score_net_w = np.random.randn(d_model, 1) * 0.02
        self.score_net_b = np.zeros(1)

    def evaluate(
        self,
        response_tokens: List[int],
        embed_matrix: np.ndarray
    ) -> float:
        """
        Evaluate a response's quality.

        Args:
            response_tokens: Response as token list
            embed_matrix: Token embedding matrix

        Returns:
            Quality score (higher is better)
        """
        if len(response_tokens) == 0:
            return 0.0

        # Embed response tokens
        embeddings = embed_matrix[response_tokens]

        # Mean pooling
        pooled = np.mean(embeddings, axis=0)

        # Score
        score = pooled @ self.score_net_w + self.score_net_b

        # Add heuristic bonuses
        length_bonus = min(len(response_tokens) / 20.0, 1.0)  # Prefer longer (up to a point)
        diversity_bonus = len(set(response_tokens)) / max(len(response_tokens), 1)

        return float(score[0]) + 0.3 * length_bonus + 0.2 * diversity_bonus

    def rank_responses(
        self,
        responses: List[List[int]],
        embed_matrix: np.ndarray
    ) -> List[Tuple[int, float]]:
        """
        Rank responses by quality.

        Args:
            responses: List of response token lists
            embed_matrix: Token embedding matrix

        Returns:
            List of (index, score) sorted by score descending
        """
        scores = []
        for i, resp in enumerate(responses):
            score = self.evaluate(resp, embed_matrix)
            scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


################################################################################
# SECTION 4: SELF-PLAY TRAINER
################################################################################

class SelfPlayTrainer:
    """
    Self-Play Trainer
    ==================

    Implements the self-play training loop for LLMs.

    Process:
    1. For each iteration:
       a. Generate responses for a set of prompts
       b. Evaluate responses
       c. Select best responses as training data
       d. Fine-tune model on selected data
    2. Reference model prevents drift from original capabilities

    Key insight: The model is both the generator AND the teacher.
    This creates a virtuous cycle of improvement.

    Interview Questions:
        1. "What prevents self-play from converging to degenerate outputs?"
           The reference model KL penalty keeps the model close to its
           original distribution. Also, diverse prompts and multi-metric
           evaluation prevent gaming a single metric.

        2. "How many self-play iterations are typical?"
           Usually 3-5 iterations. After that, returns diminish.
           Each iteration should show measurable improvement on
           held-out benchmarks.

        3. "What's the relationship between self-play and DPO?"
           DPO uses pairs of (preferred, rejected) responses.
           Self-play generates these pairs automatically, making
           it a scalable alternative to human annotation.
    """

    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.generator = ResponseGenerator(config.vocab_size, config.d_model)
        self.evaluator = ResponseEvaluator(config.d_model)

        # Reference model (copy of initial model, never updated)
        self.reference_embed = self.generator.embed.copy()

        # Training history
        self.history = []

    def select_best_responses(
        self,
        prompt: List[int],
        responses: List[List[int]],
        top_k: int = 1
    ) -> List[List[int]]:
        """
        Select the best responses for training.

        Args:
            prompt: Original prompt
            responses: Generated responses
            top_k: Number of best responses to keep

        Returns:
            Selected responses
        """
        rankings = self.evaluator.rank_responses(
            responses, self.generator.embed
        )

        selected = []
        for idx, score in rankings[:top_k]:
            selected.append(responses[idx])

        return selected

    def compute_kl_penalty(self, tokens: List[int]) -> float:
        """
        Compute KL divergence between current and reference model.

        This prevents the model from drifting too far from its
        original capabilities.

        KL(P_ref || P_current) = Σ P_ref(x) log(P_ref(x) / P_current(x))

        Args:
            tokens: Token sequence

        Returns:
            KL divergence penalty
        """
        if not self.config.use_reference:
            return 0.0

        kl = 0.0
        for t in tokens:
            # Current model distribution
            current_logits = self.generator.embed[t] @ self.generator.output_proj
            current_probs = self._softmax(current_logits)

            # Reference model distribution
            ref_logits = self.reference_embed[t] @ self.generator.output_proj
            ref_probs = self._softmax(ref_logits)

            # KL divergence
            kl += np.sum(ref_probs * np.log(ref_probs / (current_probs + 1e-8) + 1e-8))

        return kl / max(len(tokens), 1)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def train_iteration(
        self,
        prompts: List[List[int]],
        iteration: int
    ) -> Dict[str, float]:
        """
        Run one self-play iteration.

        Args:
            prompts: List of prompt token lists
            iteration: Current iteration number

        Returns:
            Dictionary of metrics
        """
        all_selected = []
        total_score = 0.0

        for prompt in prompts:
            # Generate responses
            responses = self.generator.generate(
                prompt,
                max_len=self.config.seq_len,
                temperature=0.8 + iteration * 0.1,  # Increase diversity over time
                num_responses=self.config.responses_per_prompt
            )

            # Select best
            selected = self.select_best_responses(
                prompt, responses, self.config.top_k_select
            )

            all_selected.extend([(prompt, s) for s in selected])

            # Track scores
            rankings = self.evaluator.rank_responses(responses, self.generator.embed)
            total_score += rankings[0][1]  # Best score

        avg_score = total_score / len(prompts)

        return {
            'iteration': iteration,
            'avg_score': avg_score,
            'num_training_pairs': len(all_selected),
            'prompts_processed': len(prompts)
        }

    def train(self, prompts: List[List[int]]) -> List[Dict]:
        """
        Run full self-play training.

        Args:
            prompts: List of prompt token lists

        Returns:
            Training history
        """
        print(f"Starting self-play training for {self.config.num_iterations} iterations")

        for iteration in range(self.config.num_iterations):
            metrics = self.train_iteration(prompts, iteration)
            self.history.append(metrics)

            print(f"  Iteration {iteration}: "
                  f"avg_score={metrics['avg_score']:.3f}, "
                  f"training_pairs={metrics['num_training_pairs']}")

        return self.history


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################

def demonstrate_self_play():
    """Demonstrate self-play training."""
    print("=" * 70)
    print("SELF-PLAY TRAINING")
    print("=" * 70)

    # Configuration
    config = SelfPlayConfig(
        vocab_size=50,
        d_model=32,
        seq_len=16,
        num_iterations=3,
        responses_per_prompt=4,
        top_k_select=2
    )

    print(f"\nConfiguration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Model dim: {config.d_model}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Responses per prompt: {config.responses_per_prompt}")

    # Create trainer
    trainer = SelfPlayTrainer(config)

    # Create dummy prompts
    prompts = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ]

    # Generate responses
    print("\n--- Generating responses for prompt [1,2,3,4] ---")
    responses = trainer.generator.generate(
        prompts[0], max_len=8, temperature=1.0, num_responses=4
    )
    for i, resp in enumerate(responses):
        print(f"  Response {i}: {resp}")

    # Evaluate responses
    print("\n--- Evaluating responses ---")
    rankings = trainer.evaluator.rank_responses(responses, trainer.generator.embed)
    for idx, score in rankings:
        print(f"  Response {idx}: score={score:.3f}")

    # Run self-play training
    print("\n--- Self-Play Training ---")
    history = trainer.train(prompts)

    print("\n--- Training History ---")
    for metrics in history:
        print(f"  Iteration {metrics['iteration']}: "
              f"score={metrics['avg_score']:.3f}")

    # KL penalty demonstration
    print("\n--- KL Penalty ---")
    test_tokens = [1, 2, 3, 4, 5]
    kl = trainer.compute_kl_penalty(test_tokens)
    print(f"  KL divergence from reference: {kl:.4f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Self-play enables models to improve without human data!")
    print("The model generates, evaluates, and learns from its own outputs.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_self_play()
