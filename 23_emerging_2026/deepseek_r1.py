"""
################################################################################
DEEPSEEK R1 — REASONING MODEL WITH GRPO TRAINING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is DeepSeek R1?
    A reasoning-focused LLM released January 2025 by DeepSeek (China).
    It matches GPT-4o and OpenAI o1 performance at a fraction of the cost.

Why is it important?
    - Trained for ~$6 million vs $100 million for GPT-4
    - Uses GRPO (Group Relative Policy Optimization) — no critic model needed
    - Open-source under MIT License
    - Surpassed ChatGPT as #1 on US iOS App Store (Jan 27, 2025)

Key Innovations:
    1. Multi-head Latent Attention (MLA): Compresses KV cache with low-rank
    2. Mixture of Experts (MoE): 671B total params, only 37B active per token
    3. GRPO Training: Rule-based rewards, no human feedback needed
    4. Multi-token prediction for faster decoding

Architecture (DeepSeek-V3 base):
    - 671 billion total parameters
    - 37 billion active per token
    - 61 transformer layers
    - 128K context length
    - Custom 8-bit floating point (FP8) training

Interview Questions:
    1. "What makes DeepSeek R1 different?"
        Uses GRPO RL training with rule-based rewards (no human feedback).
        Multi-head Latent Attention compresses KV cache for efficiency.
        MoE with shared + routed experts for sparse activation.

    2. "How does MLA work?"
        Instead of storing full K,V heads, project to low-rank space.
        Saves memory while preserving attention quality.

    3. "What is GRPO?"
        Group Relative Policy Optimization — generates multiple responses,
        ranks them with rule-based rewards, updates policy. No critic model.

################################################################################
"""

import numpy as np
from typing import Optional, List, Tuple

################################################################################
# SECTION 1: MULTI-HEAD LATENT ATTENTION (MLA)
################################################################################

class MultiHeadLatentAttention:
    """
    Multi-head Latent Attention (MLA)
    =================================

    DeepSeek's key innovation for efficient attention.

    Standard MHA: stores full K,V for each head → large KV cache
    MLA: compresses K,V to low-rank latent space → smaller cache

    How it works:
        1. Project input to low-rank latent: c_kv = W_kv @ x  (d_model → d_latent)
        2. Decompress for attention: K = W_k @ c_kv, V = W_v @ c_kv
        3. Cache only c_kv (much smaller than full K,V)

    Memory savings:
        Standard: cache_size = 2 * n_heads * d_head * seq_len
        MLA: cache_size = d_latent * seq_len  (d_latent << 2 * n_heads * d_head)

    Interview Question:
        "How does MLA reduce memory?"
        By projecting K,V to a low-rank latent space and caching only
        the compressed representation. At attention time, decompress.
    """

    def __init__(self, d_model: int = 4096, n_heads: int = 32,
                 d_latent: int = 512):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_latent: Latent space dimension (much smaller than d_model)
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_latent = d_latent

        # Low-rank projection matrices
        self.W_kv = np.random.randn(d_model, d_latent) * 0.02
        self.W_k = np.random.randn(d_latent, d_model) * 0.02
        self.W_v = np.random.randn(d_latent, d_model) * 0.02
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

    def compress_kv(self, x: np.ndarray) -> np.ndarray:
        """
        Compress input to low-rank latent space.

        Instead of storing full K,V (d_model each), store only
        the compressed c_kv (d_latent).

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            c_kv: Compressed latent [batch, seq_len, d_latent]
        """
        return x @ self.W_kv

    def decompress_kv(self, c_kv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompress latent back to K,V for attention.

        Args:
            c_kv: Compressed latent [batch, seq_len, d_latent]

        Returns:
            K, V: Key and Value tensors [batch, seq_len, d_model]
        """
        K = c_kv @ self.W_k
        V = c_kv @ self.W_v
        return K, V

    def forward(self, x: np.ndarray, cached_kv: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MLA forward pass.

        Args:
            x: Input [batch, seq_len, d_model]
            cached_kv: Previously cached latent (for generation)

        Returns:
            output: Attention output [batch, seq_len, d_model]
            c_kv: Latent to cache for next step
        """
        batch, seq_len, _ = x.shape

        # 1. Compute queries
        Q = x @ self.W_q

        # 2. Get or compute compressed KV
        if cached_kv is not None:
            c_kv = cached_kv
        else:
            c_kv = self.compress_kv(x)

        # 3. Decompress K,V
        K, V = self.decompress_kv(c_kv)

        # 4. Reshape for multi-head attention
        Q = Q.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch, -1, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch, -1, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        # 5. Scaled dot-product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)
        attn = self._softmax(scores)
        context = attn @ V

        # 6. Reshape and project output
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        output = context @ self.W_o

        return output, c_kv

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)


################################################################################
# SECTION 2: MIXTURE OF EXPERTS (MOE) WITH SHARED EXPERTS
################################################################################

class DeepSeekMoE:
    """
    DeepSeek Mixture of Experts
    ============================

    DeepSeek's MoE has two types of experts:
    1. Shared experts: Always activated for every token
    2. Routed experts: Selected per token by a router

    Architecture:
        token → [shared_expert_1, shared_expert_2, ...] → shared_output
        token → router → [routed_expert_i, routed_expert_j, ...] → routed_output
        final = shared_output + routed_output

    Why shared experts?
        - Capture common patterns (syntax, basic semantics)
        - Routed experts specialize (math, code, etc.)
        - Better than pure routing (some knowledge always available)

    Interview Question:
        "How does DeepSeek MoE differ from standard MoE?"
        DeepSeek adds shared experts that are ALWAYS active,
        alongside routed experts selected per token. This ensures
        common knowledge is always available while allowing specialization.
    """

    def __init__(self, d_model: int = 4096, n_shared: int = 2,
                 n_routed: int = 64, top_k: int = 6):
        """
        Args:
            d_model: Model dimension
            n_shared: Number of shared (always-active) experts
            n_routed: Number of routed (selectable) experts
            top_k: Number of routed experts to activate per token
        """
        self.d_model = d_model
        self.n_shared = n_shared
        self.n_routed = n_routed
        self.top_k = top_k

        # Shared experts (always active)
        self.shared_experts = [
            self._create_expert() for _ in range(n_shared)
        ]

        # Routed experts
        self.routed_experts = [
            self._create_expert() for _ in range(n_routed)
        ]

        # Router: projects token to expert selection scores
        self.router_weight = np.random.randn(d_model, n_routed) * 0.02

    def _create_expert(self) -> dict:
        """Create a single expert (2-layer MLP)."""
        d_ff = self.d_model * 4
        return {
            'W1': np.random.randn(self.d_model, d_ff) * 0.02,
            'W2': np.random.randn(d_ff, self.d_model) * 0.02,
        }

    def _expert_forward(self, x: np.ndarray, expert: dict) -> np.ndarray:
        """Forward through a single expert (SwiGLU-style MLP)."""
        hidden = x @ expert['W1']
        hidden = hidden * (1 / (1 + np.exp(-hidden)))  # SiLU activation
        return hidden @ expert['W2']

    def route(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Route tokens to experts.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            indices: Selected expert indices [batch, seq_len, top_k]
            weights: Expert weights [batch, seq_len, top_k]
        """
        # Compute routing scores
        scores = x @ self.router_weight  # [batch, seq_len, n_routed]

        # Top-k selection
        indices = np.argsort(scores, axis=-1)[:, :, -self.top_k:]

        # Softmax over selected experts
        batch, seq_len, _ = x.shape
        selected_scores = np.take_along_axis(scores, indices, axis=-1)
        weights = self._softmax(selected_scores)

        return indices, weights

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        DeepSeek MoE forward pass.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            output: MoE output [batch, seq_len, d_model]
        """
        batch, seq_len, d = x.shape

        # 1. Shared experts (always active)
        shared_output = np.zeros_like(x)
        for expert in self.shared_experts:
            shared_output += self._expert_forward(x, expert)
        shared_output /= self.n_shared  # Average

        # 2. Routed experts
        indices, weights = self.route(x)
        routed_output = np.zeros_like(x)

        for i in range(self.top_k):
            expert_idx = indices[:, :, i]  # [batch, seq_len]
            expert_weight = weights[:, :, i:i+1]  # [batch, seq_len, 1]

            # Process through selected expert
            expert_out = self._expert_forward(x, self.routed_experts[i])
            routed_output += expert_weight * expert_out

        # 3. Combine shared + routed
        return shared_output + routed_output

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)


################################################################################
# SECTION 3: GRPO TRAINING
################################################################################

class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO)
    ===========================================

    DeepSeek R1's training algorithm. Key insight: you don't need a
    separate critic/value model. Just compare multiple outputs.

    How GRPO works:
        1. For each prompt, generate G outputs from current policy
        2. Score each output with rule-based rewards
        3. Compute advantages: how much better/worse than group average
        4. Update policy using clipped objective (like PPO but simpler)

    Advantages over PPO:
        - No critic model needed (saves ~50% compute)
        - Simpler implementation
        - Rule-based rewards (no human feedback needed)

    DeepSeek R1 Reward Types:
        1. Accuracy reward: Is the answer correct?
        2. Format reward: Does it follow the thinking format?
        3. Language reward: Consistent language use

    Interview Question:
        "How does GRPO differ from PPO?"
        GRPO eliminates the critic model by using group-relative
        advantages. Generate multiple outputs, rank them, and use
        relative ranking as advantage signal. Much simpler and cheaper.
    """

    def __init__(self, policy_model, group_size: int = 8,
                 clip_range: float = 0.2, learning_rate: float = 1e-6):
        """
        Args:
            policy_model: The language model to train
            group_size: Number of outputs to generate per prompt
            clip_range: PPO-style clipping range
            learning_rate: Policy update learning rate
        """
        self.policy = policy_model
        self.group_size = group_size
        self.clip_range = clip_range
        self.lr = learning_rate

    def compute_rewards(self, outputs: List[str], correct_answer: str
                        ) -> np.ndarray:
        """
        Compute rule-based rewards for each output.

        No human feedback needed! Just check:
        1. Is the final answer correct?
        2. Does it follow the format (  reasoning  answer)?

        Args:
            outputs: Generated responses
            correct_answer: Ground truth answer

        Returns:
            rewards: Reward for each output [group_size]
        """
        rewards = []
        for output in outputs:
            reward = 0.0

            # Accuracy reward: extract answer and check
            extracted = self._extract_answer(output)
            if extracted == correct_answer:
                reward += 1.0

            # Format reward: has thinking tags
            if '<thinking>' in output and '</thinking>' in output:
                reward += 0.1

            # Length penalty: prefer concise answers
            if len(output) < 1000:
                reward += 0.05

            rewards.append(reward)

        return np.array(rewards)

    def compute_advantages(self, rewards: np.ndarray) -> np.ndarray:
        """
        Compute group-relative advantages.

        Instead of a learned value function, use the group statistics:
        advantage_i = (reward_i - mean(rewards)) / std(rewards)

        This is the key GRPO insight!

        Args:
            rewards: Rewards for each output [group_size]

        Returns:
            advantages: Normalized advantages [group_size]
        """
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8  # Avoid division by zero
        return (rewards - mean_reward) / std_reward

    def compute_loss(self, log_probs: np.ndarray, old_log_probs: np.ndarray,
                     advantages: np.ndarray) -> float:
        """
        Compute clipped policy loss (similar to PPO).

        Loss = -min(ratio * advantages, clip(ratio) * advantages)

        Where ratio = exp(log_prob - old_log_prob)

        Args:
            log_probs: Current policy log probabilities
            old_log_probs: Old policy log probabilities
            advantages: Computed advantages

        Returns:
            loss: Scalar loss value
        """
        # Probability ratio
        ratio = np.exp(log_probs - old_log_probs)

        # Clipped surrogate
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - self.clip_range,
                        1 + self.clip_range) * advantages

        return -np.mean(np.minimum(surr1, surr2))

    def train_step(self, prompt: str, correct_answer: str) -> dict:
        """
        One GRPO training step.

        Args:
            prompt: Input prompt
            correct_answer: Ground truth answer

        Returns:
            metrics: Training metrics
        """
        # 1. Generate multiple outputs
        outputs = []
        old_log_probs = []
        for _ in range(self.group_size):
            output, log_prob = self.policy.generate(prompt)
            outputs.append(output)
            old_log_probs.append(log_prob)

        # 2. Compute rewards
        rewards = self.compute_rewards(outputs, correct_answer)

        # 3. Compute advantages (group-relative!)
        advantages = self.compute_advantages(rewards)

        # 4. Compute loss and update
        new_log_probs = [self.policy.log_prob(prompt, o) for o in outputs]
        loss = self.compute_loss(
            np.array(new_log_probs),
            np.array(old_log_probs),
            advantages
        )

        return {
            'loss': loss,
            'mean_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'advantages_mean': np.mean(advantages),
        }

    def _extract_answer(self, output: str) -> str:
        """Extract final answer from output."""
        # Look for answer after thinking tags
        if '</thinking>' in output:
            answer = output.split('</thinking>')[-1].strip()
            return answer
        return output.strip()


################################################################################
# SECTION 4: MULTI-TOKEN PREDICTION
################################################################################

class MultiTokenPredictor:
    """
    Multi-Token Prediction (MTP)
    =============================

    DeepSeek V3/R1 predicts multiple future tokens simultaneously.

    Standard LLM: predict next token only
    MTP: predict next 2-4 tokens at once

    Benefits:
        - Faster inference (fewer forward passes)
        - Better representations (forces model to plan ahead)
        - Training signal: auxiliary loss on future tokens

    Trade-off:
        - Slightly less accurate per token
        - But much faster overall

    Architecture:
        Hidden state → head_1 → token_1
        Hidden state → head_2 → token_2
        Hidden state → head_3 → token_3

    Interview Question:
        "What is multi-token prediction?"
        Instead of predicting only the next token, predict multiple
        future tokens simultaneously. Improves inference speed and
        forces the model to develop better internal representations.
    """

    def __init__(self, d_model: int, vocab_size: int, n_future: int = 3):
        """
        Args:
            d_model: Model dimension
            vocab_size: Vocabulary size
            n_future: Number of future tokens to predict
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_future = n_future

        # Separate prediction heads for each future position
        self.heads = [
            np.random.randn(d_model, vocab_size) * 0.02
            for _ in range(n_future)
        ]

    def forward(self, hidden_states: np.ndarray) -> List[np.ndarray]:
        """
        Predict multiple future tokens.

        Args:
            hidden_states: Model hidden states [batch, seq_len, d_model]

        Returns:
            predictions: List of logits for each future position
        """
        predictions = []
        for head in self.heads:
            logits = hidden_states @ head  # [batch, seq_len, vocab_size]
            predictions.append(logits)
        return predictions

    def compute_loss(self, predictions: List[np.ndarray],
                     targets: List[np.ndarray]) -> float:
        """
        Compute multi-token prediction loss.

        Loss = sum of cross-entropy losses for each future position.

        Args:
            predictions: List of predicted logits
            targets: List of target token IDs

        Returns:
            total_loss: Combined loss
        """
        total_loss = 0.0
        for pred, target in zip(predictions, targets):
            # Cross-entropy loss
            probs = self._softmax(pred)
            log_probs = np.log(probs + 1e-8)
            loss = -np.mean(log_probs[np.arange(len(target)), target])
            total_loss += loss

        return total_loss / self.n_future

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)


################################################################################
# SECTION 5: COMPLETE DEEPSEEK R1 MODEL
################################################################################

class DeepSeekR1:
    """
    DeepSeek R1 Complete Model
    ===========================

    Combines all DeepSeek innovations:
    1. Multi-head Latent Attention (MLA)
    2. Mixture of Experts (MoE) with shared experts
    3. Multi-token prediction
    4. GRPO training

    Architecture (V3 base):
        - 671B total parameters
        - 37B active per token
        - 61 transformer layers
        - 128K context length
        - FP8 training

    Interview Question:
        "Describe DeepSeek R1's architecture."
        Built on DeepSeek-V3 with MLA for efficient attention,
        MoE with shared+routed experts for sparse activation,
        multi-token prediction for speed, and trained with GRPO
        using rule-based rewards (no human feedback needed).
    """

    def __init__(self, d_model: int = 4096, n_layers: int = 61,
                 n_heads: int = 32, n_shared_experts: int = 2,
                 n_routed_experts: int = 64, vocab_size: int = 128000):
        self.d_model = d_model
        self.n_layers = n_layers

        # MLA attention layers
        self.attention_layers = [
            MultiHeadLatentAttention(d_model, n_heads)
            for _ in range(n_layers)
        ]

        # MoE layers (with shared experts)
        self.moe_layers = [
            DeepSeekMoE(d_model, n_shared_experts, n_routed_experts)
            for _ in range(n_layers)
        ]

        # Multi-token prediction
        self.mtp = MultiTokenPredictor(d_model, vocab_size)

        # Embedding
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02

    def forward(self, input_ids: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through DeepSeek R1.

        Args:
            input_ids: Input token IDs [batch, seq_len]

        Returns:
            hidden: Final hidden states
            mtp_logits: Multi-token predictions
        """
        # Embedding
        x = self.embedding[input_ids]

        # Transformer layers
        for i in range(self.n_layers):
            # MLA attention
            attn_out, _ = self.attention_layers[i].forward(x)
            x = x + attn_out  # Residual

            # MoE FFN
            moe_out = self.moe_layers[i].forward(x)
            x = x + moe_out  # Residual

        # Multi-token prediction
        mtp_logits = self.mtp.forward(x)

        return x, mtp_logits

    def generate(self, prompt_ids: np.ndarray, max_tokens: int = 100
                 ) -> np.ndarray:
        """
        Generate text with multi-token prediction.

        Uses MTP to predict multiple tokens per step, then verifies.

        Args:
            prompt_ids: Prompt token IDs
            max_tokens: Maximum tokens to generate

        Returns:
            generated: Generated token IDs
        """
        generated = prompt_ids.copy()

        for _ in range(max_tokens // self.n_future):
            hidden, mtp_logits = self.forward(generated)

            # Get predictions from each head
            new_tokens = []
            for logits in mtp_logits:
                next_token = np.argmax(logits[:, -1, :], axis=-1)
                new_tokens.append(next_token)

            # Append predicted tokens
            generated = np.concatenate([
                generated,
                np.array(new_tokens).reshape(1, -1)
            ], axis=1)

        return generated


################################################################################
# DEMO
################################################################################

if __name__ == "__main__":
    print("=" * 70)
    print("DEEPSEEK R1 ARCHITECTURE DEMO")
    print("=" * 70)

    # MLA demo
    print("\n1. Multi-head Latent Attention (MLA)")
    print("-" * 40)
    mla = MultiHeadLatentAttention(d_model=256, n_heads=8, d_latent=32)

    x = np.random.randn(1, 10, 256)
    output, c_kv = mla.forward(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Cached KV shape: {c_kv.shape}")
    print(f"   Memory saved: {2 * 256 * 10} -> {32 * 10} = "
          f"{(2 * 256 * 10) / (32 * 10):.1f}x compression")

    # MoE demo
    print("\n2. Mixture of Experts (MoE)")
    print("-" * 40)
    moe = DeepSeekMoE(d_model=256, n_shared=2, n_routed=16, top_k=4)

    x = np.random.randn(1, 10, 256)
    output = moe.forward(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Shared experts: {moe.n_shared} (always active)")
    print(f"   Routed experts: {moe.n_routed} (top-{moe.top_k} selected)")

    # GRPO demo
    print("\n3. GRPO Training")
    print("-" * 40)
    print("   Group size: 8 outputs per prompt")
    print("   No critic model needed!")
    print("   Rule-based rewards: accuracy + format")

    # MTP demo
    print("\n4. Multi-Token Prediction")
    print("-" * 40)
    mtp = MultiTokenPredictor(d_model=256, vocab_size=1000, n_future=3)

    hidden = np.random.randn(1, 10, 256)
    predictions = mtp.forward(hidden)

    print(f"   Predicting {mtp.n_future} future tokens simultaneously")
    print(f"   Each prediction shape: {predictions[0].shape}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. MLA compresses KV cache for efficient attention")
    print("2. MoE with shared experts ensures common knowledge")
    print("3. GRPO trains without human feedback")
    print("4. Multi-token prediction speeds up inference")
    print("5. Total: 671B params, only 37B active per token")
