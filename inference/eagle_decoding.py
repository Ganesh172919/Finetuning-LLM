"""
################################################################################
EAGLE — EXTRAPOLATION ALGORITHM FOR GREATER LANGUAGE-MODEL EFFICIENCY
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is EAGLE?
    EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)
    is an advanced speculative decoding method from Li et al. (2024).
    Unlike standard speculative decoding that uses a separate draft model,
    EAGLE uses the target model's own features to predict future tokens.

Key Innovation:
    Instead of running a full draft model, EAGLE trains a lightweight
    "draft head" that takes the target model's hidden states as input
    and predicts future token distributions. This is:
    - Faster than running a separate model (shared features)
    - More accurate than simple Medusa heads (richer input)
    - Memory efficient (small head, not full model)

How EAGLE works:
    1. Run target model to get hidden state h_t
    2. Draft head predicts distribution for token at t+1 using h_t
    3. Draft head predicts distribution for token at t+2 using h_t and draft_1
    4. Continue for K steps
    5. Verify all K tokens with target model in ONE forward pass
    6. Accept matching tokens, reject and resample from first mismatch

Why EAGLE is better:
    - 2.5-3.5x speedup (vs 2-3x for standard speculative decoding)
    - Higher acceptance rate (draft head is more accurate)
    - No separate model needed (just a small head)
    - Can be fine-tuned on top of existing model

Interview Questions:
    Q: "How does EAGLE differ from standard speculative decoding?"
    A: EAGLE uses the target model's hidden states to predict future tokens
       via a lightweight draft head, instead of running a separate draft model.
       This gives higher accuracy (better acceptance rate) and lower overhead.

    Q: "Why is EAGLE faster than Medusa?"
    A: Medusa heads only see the current hidden state. EAGLE's draft head
       sees the current hidden state AND the previously drafted tokens,
       giving it more context for accurate prediction.

################################################################################
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

################################################################################
# SECTION 1: EAGLE DRAFT HEAD
################################################################################

class EagleDraftHead:
    """
    EAGLE Draft Head — Feature-based token prediction.

    Instead of running a full model, this lightweight head predicts
    future tokens using the target model's hidden states.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    EAGLE Draft Head                          │
    │                                                              │
    │  Hidden state h_t (from target model)                        │
    │       │                                                      │
    │       ├──▶ Feature Projection                                │
    │       │         │                                            │
    │       │         ▼                                            │
    │       ├──▶ + Previous token embedding                        │
    │       │         │                                            │
    │       │         ▼                                            │
    │       └──▶ Transformer Layer (1-2 layers)                    │
    │                 │                                            │
    │                 ▼                                            │
    │            Token Distribution                                │
    └─────────────────────────────────────────────────────────────┘

    The key insight: by conditioning on BOTH the hidden state and
    previously drafted tokens, the head can make more accurate
    predictions than Medusa (which only uses hidden state).
    """

    def __init__(self, d_model: int = 256, vocab_size: int = 32000, n_layers: int = 2):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        # Feature projection
        self.feature_proj = np.random.randn(d_model, d_model) * 0.02

        # Token embedding (for conditioning on previous tokens)
        self.token_embed = np.random.randn(vocab_size, d_model) * 0.02

        # Transformer-like layers
        self.layer_weights = [np.random.randn(d_model, d_model) * 0.02 for _ in range(n_layers)]
        self.layer_norms = [np.ones(d_model) for _ in range(n_layers)]

        # Output projection
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.02

    def predict_next(
        self,
        hidden_state: np.ndarray,
        prev_token_id: int,
    ) -> np.ndarray:
        """
        Predict distribution for next token.

        Args:
            hidden_state: [d_model] from target model
            prev_token_id: ID of previously generated token

        Returns:
            probs: [vocab_size] probability distribution
        """
        # Project hidden state
        h = hidden_state @ self.feature_proj

        # Add previous token embedding
        h = h + self.token_embed[prev_token_id]

        # Process through layers
        for i in range(self.n_layers):
            h = self._layer_norm(h + np.tanh(h @ self.layer_weights[i]), self.layer_norms[i])

        # Output logits
        logits = h @ self.output_proj

        # Softmax
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        return probs

    def draft_sequence(
        self,
        hidden_state: np.ndarray,
        start_token: int,
        K: int = 5,
        temperature: float = 0.7,
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        Draft K tokens autoregressively using the draft head.

        Args:
            hidden_state: [d_model] from target model
            start_token: Token to start drafting from
            K: Number of tokens to draft
            temperature: Sampling temperature

        Returns:
            tokens: K drafted token IDs
            probs: K probability distributions
        """
        tokens = []
        probs = []
        current_token = start_token

        for _ in range(K):
            # Predict next token distribution
            prob = self.predict_next(hidden_state, current_token)

            # Apply temperature
            if temperature != 1.0:
                log_prob = np.log(prob + 1e-8) / temperature
                prob = np.exp(log_prob) / np.sum(np.exp(log_prob))

            # Sample
            token = int(np.random.choice(self.vocab_size, p=prob))

            tokens.append(token)
            probs.append(prob)
            current_token = token

        return tokens, probs

    def _layer_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x)
        var = np.var(x)
        return (x - mean) / np.sqrt(var + eps) * weight


################################################################################
# SECTION 2: EAGLE DECODING ENGINE
################################################################################

class EagleDecodingEngine:
    """
    EAGLE Decoding Engine — Complete speculative decoding with feature-based draft.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    EAGLE Pipeline                            │
    │                                                              │
    │  1. Target Model → hidden_state h_t                          │
    │  2. Draft Head → K candidate tokens using h_t               │
    │  3. Target Model → verify all K tokens (one forward pass)   │
    │  4. Accept/reject → accepted tokens + 1 resampled           │
    │                                                              │
    │  Speedup: 2.5-3.5x (higher than standard speculative)       │
    │  Quality: Exact (same distribution as target model)          │
    └─────────────────────────────────────────────────────────────┘

    Interview Question:
        Q: "Walk me through EAGLE's decoding process."
        A: (1) Run target model to get hidden state
           (2) Draft head predicts K future tokens using hidden state
           (3) Verify all K tokens with target model in one pass
           (4) Accept matching tokens, resample from first mismatch
           (5) Net: 2.5-3.5 tokens per target model forward pass
    """

    def __init__(
        self,
        d_model: int = 256,
        vocab_size: int = 32000,
        K: int = 5,
        temperature: float = 0.7,
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.K = K
        self.temperature = temperature

        # Draft head
        self.draft_head = EagleDraftHead(d_model, vocab_size)

        # Statistics
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.total_steps = 0

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 100,
    ) -> List[int]:
        """
        Generate tokens using EAGLE speculative decoding.

        Args:
            prompt_ids: Starting token IDs
            max_new_tokens: Maximum tokens to generate

        Returns:
            generated: All token IDs (prompt + generated)
        """
        generated = list(prompt_ids)
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # Step 1: Get hidden state from target model
            hidden_state = self._get_hidden_state(generated)

            # Step 2: Draft K tokens using draft head
            last_token = generated[-1]
            draft_tokens, draft_probs = self.draft_head.draft_sequence(
                hidden_state, last_token, self.K, self.temperature
            )

            # Step 3: Verify with target model
            target_probs = self._verify_tokens(generated, draft_tokens)

            # Step 4: Accept/reject
            accepted, resampled = self._accept_reject(
                draft_tokens, draft_probs, target_probs
            )

            generated.extend(accepted)
            if resampled is not None:
                generated.append(resampled)

            tokens_generated += len(accepted) + (1 if resampled is not None else 0)
            self.total_steps += 1

        return generated

    def _get_hidden_state(self, context: List[int]) -> np.ndarray:
        """Get hidden state from target model (simulated)."""
        # In production: target_model.get_hidden_state(context)
        seed = hash(tuple(context[-5:])) % (2**31)
        np.random.seed(seed)
        h = np.random.randn(self.d_model) * 0.1
        np.random.seed(None)
        return h

    def _verify_tokens(self, context: List[int], draft_tokens: List[int]) -> List[np.ndarray]:
        """Verify all K tokens with target model in ONE forward pass."""
        # In production: one forward pass over [context + draft_tokens]
        target_probs = []
        for i in range(len(draft_tokens)):
            seed = hash(tuple(context[-3:] + draft_tokens[:i+1])) % (2**31)
            np.random.seed(seed)
            prob = np.random.dirichlet(np.ones(self.vocab_size) * 0.1)
            # Boost a few likely tokens
            for _ in range(3):
                idx = np.random.randint(0, self.vocab_size)
                prob[idx] += np.random.random() * 5
            prob = prob / np.sum(prob)
            target_probs.append(prob)
            np.random.seed(None)
        return target_probs

    def _accept_reject(
        self,
        draft_tokens: List[int],
        draft_probs: List[np.ndarray],
        target_probs: List[np.ndarray],
    ) -> Tuple[List[int], Optional[int]]:
        """Accept/reject with proper rejection sampling."""
        accepted = []

        for i, (token, dp, tp) in enumerate(zip(draft_tokens, draft_probs, target_probs)):
            p_target = tp[token]
            p_draft = dp[token]

            if np.random.random() < min(1.0, p_target / (p_draft + 1e-8)):
                accepted.append(token)
                self.accepted_tokens += 1
            else:
                # Resample from adjusted distribution
                adjusted = np.maximum(0, tp - dp)
                total = np.sum(adjusted)
                if total < 1e-10:
                    adjusted = np.ones(self.vocab_size) / self.vocab_size
                else:
                    adjusted = adjusted / total
                resampled = int(np.random.choice(self.vocab_size, p=adjusted))
                self.total_tokens += len(accepted) + 1
                return accepted, resampled

        # All accepted
        resampled = int(np.random.choice(self.vocab_size, p=target_probs[-1]))
        self.total_tokens += len(accepted) + 1
        return accepted, resampled

    def get_stats(self) -> Dict:
        """Get EAGLE decoding statistics."""
        total = self.total_tokens
        return {
            "total_tokens": total,
            "accepted": self.accepted_tokens,
            "acceptance_rate": self.accepted_tokens / max(total, 1),
            "steps": self.total_steps,
            "tokens_per_step": total / max(self.total_steps, 1),
            "speedup": total / max(self.total_steps, 1),
        }


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_eagle():
    """Demonstrate EAGLE decoding."""
    print("=" * 70)
    print("EAGLE DECODING DEMONSTRATION")
    print("=" * 70)

    # === Demo 1: Draft Head ===
    print("\n--- Demo 1: EAGLE Draft Head ---")
    head = EagleDraftHead(d_model=64, vocab_size=1000, n_layers=2)
    hidden = np.random.randn(64) * 0.1
    probs = head.predict_next(hidden, prev_token_id=42)
    print(f"Hidden state shape: {hidden.shape}")
    print(f"Output probs shape: {probs.shape}")
    print(f"Top-5 tokens: {np.argsort(probs)[-5:][::-1].tolist()}")

    # Draft sequence
    tokens, probs = head.draft_sequence(hidden, start_token=42, K=5)
    print(f"Drafted tokens: {tokens}")

    # === Demo 2: EAGLE Engine ===
    print("\n--- Demo 2: EAGLE Decoding Engine ---")
    engine = EagleDecodingEngine(d_model=64, vocab_size=1000, K=5)
    prompt = [1, 2, 3, 4, 5]
    generated = engine.generate(prompt, max_new_tokens=20)
    stats = engine.get_stats()
    print(f"Generated {len(generated) - len(prompt)} tokens")
    print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"Speedup: {stats['speedup']:.2f}x")

    # === Demo 3: Comparison ===
    print("\n--- Demo 3: Method Comparison ---")
    print(f"{'Method':<25} {'Speedup':<10} {'Overhead':<10}")
    print("-" * 45)
    print(f"{'Standard Decoding':<25} {'1.0x':<10} {'None':<10}")
    print(f"{'Speculative (draft model)':<25} {'2-3x':<10} {'Full model':<10}")
    print(f"{'Medusa (heads)':<25} {'2-3x':<10} {'Small heads':<10}")
    print(f"{'EAGLE (feature head)':<25} {'2.5-3.5x':<10} {'Small head':<10}")

    print("\n" + "=" * 70)
    print("All EAGLE demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_eagle()


################################################################################
# REFERENCES
################################################################################

# [1] Li, Y., et al. (2024). EAGLE: Speculative Sampling Requires Rethinking
#     Feature Uncertainty. arXiv:2401.15077.
#
# [2] Li, Y., et al. (2024). EAGLE-2: Faster Inference of Language Models
#     with Dynamic Draft Trees. arXiv:2406.16858.

################################################################################
