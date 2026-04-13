"""
################################################################################
TRANSFORMER LANGUAGE MODEL — THE COMPLETE ARCHITECTURE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Transformer Language Model?
    A language model that predicts the next token given previous tokens.
    It's the architecture behind GPT, Claude, LLaMA, and all modern LLMs.

    Input:  "The cat sat on the"
    Output: "mat" (predicted next token)

Architecture Overview:
    ┌─────────────────────────────────────────────┐
    │ Input Tokens: [The, cat, sat, on, the]      │
    │                                               │
    │ Token Embedding + Position Embedding          │
    │        ↓                                      │
    │ ┌─────────────────────────────────────────┐  │
    │ │ Transformer Block × N                    │  │
    │ │   ├── RMSNorm → Self-Attention → +      │  │
    │ │   └── RMSNorm → FFN/SwiGLU → +         │  │
    │ └─────────────────────────────────────────┘  │
    │        ↓                                      │
    │ Final RMSNorm                                 │
    │        ↓                                      │
    │ Output Head (LM Head)                         │
    │        ↓                                      │
    │ Logits: [vocab_size] for each position        │
    │        ↓                                      │
    │ Softmax → Probability Distribution            │
    │        ↓                                      │
    │ Sample/Argmax → Next Token                    │
    └─────────────────────────────────────────────┘

Training:
    Given: dataset of text
    Goal: maximize P(correct_next_token | previous_tokens)
    Loss: cross-entropy between predicted and actual next token
    Method: backpropagation + gradient descent

Inference:
    1. Start with prompt tokens
    2. Forward pass → predict next token
    3. Append prediction to input
    4. Repeat until done or max length

########################################

MODELS IMPLEMENTED HERE:
    1. GPT-style (decoder-only, causal)
    2. BERT-style (encoder-only, bidirectional)
    3. T5-style (encoder-decoder)

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List
import math

from .embeddings import TokenEmbedding, RotaryPositionEmbedding
from .layers import TransformerBlock, RMSNorm
from .attention import create_causal_mask


################################################################################
# SECTION 1: GPT-STYLE DECODER-ONLY MODEL
################################################################################

class TransformerLM:
    """
    Transformer Language Model (GPT-style)
    ========================================

    Definition: A decoder-only transformer that predicts the next token.

    This is the architecture used by:
    - GPT-2, GPT-3, GPT-4 (OpenAI)
    - Claude (Anthropic)
    - LLaMA, LLaMA 2, LLaMA 3 (Meta)
    - Mistral, Mixtral (Mistral AI)
    - Qwen, Qwen 2 (Alibaba)
    - DeepSeek, DeepSeek V2, DeepSeek R1 (DeepSeek)
    - Gemma (Google)
    - Phi (Microsoft)

    Why decoder-only?
    1. Simple: one architecture, one task (predict next token)
    2. Scales well: more parameters → better performance
    3. Flexible: can do many tasks via prompting
    4. Efficient: parallel training, autoregressive inference

    Scaling Laws (Kaplan et al., 2020):
    - Performance scales as power law with:
      1. Number of parameters (N)
      2. Dataset size (D)
      3. Compute budget (C)
    - Optimal: scale all three together

    Common Model Sizes:
    ┌──────────────┬────────────┬──────────┬───────────┬──────────┐
    │ Model        │ Parameters │ Layers   │ d_model   │ Heads    │
    ├──────────────┼────────────┼──────────┼───────────┼──────────┤
    │ GPT-2 Small  │ 117M       │ 12       │ 768       │ 12       │
    │ GPT-2 Large  │ 774M       │ 36       │ 1280      │ 20       │
    │ LLaMA-7B     │ 6.7B       │ 32       │ 4096      │ 32       │
    │ LLaMA-13B    │ 13B        │ 40       │ 5120      │ 40       │
    │ LLaMA-70B    │ 70B        │ 80       │ 8192      │ 64       │
    │ Mistral-7B   │ 7.3B       │ 32       │ 4096      │ 32       │
    │ GPT-4 (est.) │ 1.8T       │ 120      │ 12288     │ 96       │
    └──────────────┴────────────┴──────────┴───────────┴──────────┘

    Interview Questions:
        1. "Why decoder-only instead of encoder-decoder?"
           Decoder-only is simpler and scales better. With enough
           parameters, it can do everything encoder-decoder can.
           Most SOTA models are decoder-only.

        2. "How does GPT generate text?"
           Autoregressively: predict next token, append, repeat.
           Each new token is conditioned on all previous tokens.

        3. "What determines model quality?"
           Three factors: model size, data quality, training compute.
           Scaling laws show predictable improvements with scale.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        tie_weights: bool = True
    ):
        """
        Initialize Transformer Language Model.

        Args:
            vocab_size: Vocabulary size (e.g., 32000 for LLaMA)
            d_model: Model dimension (e.g., 4096)
            n_layers: Number of transformer blocks
            n_heads: Number of attention heads
            n_kv_heads: Number of KV heads for GQA (None = MHA)
            d_ff: Feed-forward dimension (None = auto)
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            tie_weights: Whether to share input/output embeddings
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # Token embedding: convert token IDs to vectors
        self.token_embedding = TokenEmbedding(vocab_size, d_model)

        # Position embedding: RoPE
        self.rope = RotaryPositionEmbedding(d_model, max_seq_len=max_seq_len)

        # Transformer blocks
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                n_kv_heads=n_kv_heads,
                use_swiglu=True,  # Modern standard
                dropout=dropout
            ))

        # Final normalization
        self.norm = RMSNorm(d_model)

        # Output head (language model head)
        # Often tied to input embedding weights to save parameters
        if tie_weights:
            self.output_weight = self.token_embedding.weight  # Shared
        else:
            scale = math.sqrt(2.0 / (d_model + vocab_size))
            self.output_weight = np.random.randn(vocab_size, d_model) * scale

        # Precompute causal mask
        self.causal_mask = create_causal_mask(max_seq_len)

        # Count parameters
        self._count_parameters()

    def _count_parameters(self):
        """Count total model parameters."""
        # Token embedding
        params = self.vocab_size * self.d_model

        # Transformer blocks
        for layer in self.layers:
            # Attention: Q, K, V, O projections
            params += 4 * self.d_model * self.d_model
            # FFN: gate, up, down (SwiGLU)
            d_ff = layer.ffn.d_ff if hasattr(layer.ffn, 'd_ff') else 4 * self.d_model
            params += 3 * self.d_model * d_ff
            # Norms
            params += 2 * self.d_model

        # Final norm
        params += self.d_model

        # Output head (if not tied)
        if not hasattr(self, 'output_weight') or self.output_weight is not self.token_embedding.weight:
            params += self.vocab_size * self.d_model

        self.n_params = params
        print(f"Model parameters: {params:,} ({params/1e9:.2f}B)")

    def forward(
        self,
        token_ids: np.ndarray,
        targets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Forward pass of the language model.

        Args:
            token_ids: Input token IDs [batch × seq_len]
            targets: Target token IDs [batch × seq_len] (for loss computation)

        Returns:
            logits: Output logits [batch × seq_len × vocab_size]
            loss: Cross-entropy loss (if targets provided)

        Flow:
            token_ids → embedding → rope → N transformer blocks → norm → logits
        """
        batch_size, seq_len = token_ids.shape

        # Step 1: Token embedding
        # Convert token IDs to dense vectors
        h = self.token_embedding.forward(token_ids)  # [batch × seq × d_model]

        # Step 2: Apply RoPE
        # This encodes position information
        position_ids = np.arange(seq_len)[np.newaxis, :]
        # Note: RoPE is applied inside attention, not here
        # We'll apply it when computing Q and K

        # Step 3: Apply causal mask
        mask = self.causal_mask[:seq_len, :seq_len]

        # Step 4: Apply transformer blocks
        for layer in self.layers:
            h = layer.forward(h, mask=mask)

        # Step 5: Final normalization
        h = self.norm.forward(h)

        # Step 6: Output projection (language model head)
        # logits = h @ W_output^T
        logits = np.matmul(h, self.output_weight.T)  # [batch × seq × vocab_size]

        # Step 7: Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = self._compute_loss(logits, targets)

        return logits, loss

    def _compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        Args:
            logits: [batch × seq × vocab_size]
            targets: [batch × seq]

        Returns:
            Average cross-entropy loss
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for easier computation
        logits_flat = logits.reshape(-1, vocab_size)  # [batch*seq × vocab]
        targets_flat = targets.reshape(-1)  # [batch*seq]

        # Numerically stable softmax
        shifted = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
        log_probs = shifted[np.arange(len(targets_flat)), targets_flat] - log_sum_exp

        # Negative log likelihood
        loss = -np.mean(log_probs)
        return loss

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate text autoregressively.

        Args:
            prompt_ids: Starting token IDs [1 × prompt_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-K sampling parameter
            top_p: Top-P sampling parameter
            eos_token_id: End-of-sequence token ID

        Returns:
            generated_ids: Complete sequence [1 × total_len]

        Algorithm:
            1. Start with prompt
            2. Forward pass → get logits for last position
            3. Apply temperature scaling
            4. Apply top-k/top-p filtering
            5. Sample from distribution
            6. Append to sequence
            7. Repeat until EOS or max length

        This is how ChatGPT, Claude, and all LLMs generate text!
        """
        generated = prompt_ids.copy()

        for _ in range(max_new_tokens):
            # Forward pass (only need last position's logits)
            logits, _ = self.forward(generated)

            # Get logits for last position
            next_logits = logits[0, -1, :]  # [vocab_size]

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-K filtering
            if top_k > 0:
                top_k_indices = np.argsort(next_logits)[-top_k:]
                filtered = np.full_like(next_logits, -np.inf)
                filtered[top_k_indices] = next_logits[top_k_indices]
                next_logits = filtered

            # Top-P filtering
            if top_p < 1.0:
                sorted_indices = np.argsort(next_logits)[::-1]
                sorted_logits = next_logits[sorted_indices]
                sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
                sorted_probs = sorted_probs / np.sum(sorted_probs)

                cumulative = np.cumsum(sorted_probs)
                nucleus_size = np.searchsorted(cumulative, top_p) + 1
                nucleus_indices = sorted_indices[:nucleus_size]

                filtered = np.full_like(next_logits, -np.inf)
                filtered[nucleus_indices] = next_logits[nucleus_indices]
                next_logits = filtered

            # Softmax
            probs = np.exp(next_logits - np.max(next_logits))
            probs = probs / np.sum(probs)

            # Sample
            next_token = np.random.choice(len(probs), p=probs)

            # Append
            next_token_array = np.array([[next_token]])
            generated = np.concatenate([generated, next_token_array], axis=1)

            # Check for EOS
            if eos_token_id is not None and next_token == eos_token_id:
                break

            # Truncate if too long (for memory)
            if generated.shape[1] > self.max_seq_len:
                generated = generated[:, -self.max_seq_len:]

        return generated


################################################################################
# SECTION 2: KV CACHE FOR EFFICIENT INFERENCE
################################################################################

class KVCache:
    """
    Key-Value Cache for Efficient Autoregressive Inference
    ======================================================

    Definition: Cache the K and V tensors from previous tokens
    so we don't recompute them for each new token.

    THE PROBLEM:
    Without caching, generating token N requires:
    1. Forward pass through ALL N tokens
    2. This is O(N²) per token, O(N³) total!

    THE SOLUTION:
    Cache K and V from previous tokens.
    For new token, only compute Q, K, V for that token.
    Then use cached K, V for attention.

    This reduces per-token cost from O(N) to O(1)!

    Architecture:
    ┌─────────────────────────────────────────────┐
    │ Token 1: Compute K1, V1 → Cache             │
    │ Token 2: Compute K2, V2 → Cache             │
    │ Token 3: Compute K3, V3 → Cache             │
    │ ...                                          │
    │ Token N: Compute Kn, Vn                      │
    │   Attention(Qn, [K1...Kn], [V1...Vn])        │
    │   → Use all cached K, V                      │
    └─────────────────────────────────────────────┘

    Memory: O(batch × layers × heads × seq_len × d_k)
    For LLaMA-7B with 4K context:
    32 layers × 32 heads × 4096 × 128 × 2 bytes = 1GB

    This is why GQA helps: fewer KV heads = smaller cache!

    Interview Question:
        "What is KV cache and why is it important?"
        KV cache stores computed key and value tensors from previous tokens.
        Without it, each new token would require recomputing attention
        over ALL previous tokens. With KV cache, we only compute Q, K, V
        for the new token and reuse cached K, V for attention.
        This makes generation O(1) per token instead of O(N).
    """

    def __init__(self, n_layers: int, n_kv_heads: int, d_k: int, max_seq_len: int):
        """
        Initialize KV cache.

        Args:
            n_layers: Number of transformer layers
            n_kv_heads: Number of KV heads (for GQA)
            d_k: Dimension per head
            max_seq_len: Maximum sequence length
        """
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Cache for each layer
        # Shape: [batch × n_kv_heads × max_seq × d_k]
        self.k_cache = [None] * n_layers
        self.v_cache = [None] * n_layers
        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        new_k: np.ndarray,
        new_v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update cache with new K, V and return full cached K, V.

        Args:
            layer_idx: Which transformer layer
            new_k: New key tensor [batch × n_kv_heads × 1 × d_k]
            new_v: New value tensor [batch × n_kv_heads × 1 × d_k]

        Returns:
            full_k: All cached keys [batch × n_kv_heads × seq_len × d_k]
            full_v: All cached values [batch × n_kv_heads × seq_len × d_k]
        """
        if self.k_cache[layer_idx] is None:
            # First token: initialize cache
            self.k_cache[layer_idx] = new_k
            self.v_cache[layer_idx] = new_v
        else:
            # Append new K, V
            self.k_cache[layer_idx] = np.concatenate(
                [self.k_cache[layer_idx], new_k], axis=2
            )
            self.v_cache[layer_idx] = np.concatenate(
                [self.v_cache[layer_idx], new_v], axis=2
            )

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def clear(self):
        """Clear the cache."""
        self.k_cache = [None] * self.n_layers
        self.v_cache = [None] * self.n_layers
        self.seq_len = 0


################################################################################
# SECTION 3: TRAINING PIPELINE
################################################################################

class Trainer:
    """
    Training Pipeline for Transformer Language Model
    ================================================

    Training a language model involves:
    1. Data preparation: tokenization, batching
    2. Forward pass: compute predictions
    3. Loss computation: cross-entropy
    4. Backward pass: compute gradients
    5. Optimizer step: update weights
    6. Repeat until convergence

    Key Concepts:
    - Learning rate: How much to update weights
    - Batch size: How many examples per update
    - Gradient accumulation: Simulate large batches
    - Mixed precision: Use fp16 for speed
    - Gradient clipping: Prevent exploding gradients

    Training Recipe (LLaMA-7B):
    - Tokens: 1-2 trillion
    - Batch size: 4M tokens
    - Learning rate: 3e-4 → 1e-5 (cosine schedule)
    - Warmup: 2000 steps
    - Hardware: 2048 A100 GPUs
    - Time: ~21 days

    Interview Questions:
        1. "How long does it take to train an LLM?"
           LLaMA-7B: ~21 days on 2048 A100s
           GPT-4: estimated 3-4 months on 25000 A100s
           Cost: millions to hundreds of millions of dollars

        2. "What's the most important hyperparameter?"
           Learning rate. Too high: divergence. Too low: slow convergence.
           Use cosine schedule with warmup.

        3. "How do you know when training is done?"
           Monitor validation loss. When it stops decreasing (or starts
           increasing = overfitting), training is done.
    """

    def __init__(
        self,
        model: TransformerLM,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        grad_clip: float = 1.0
    ):
        self.model = model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.step = 0

    def get_lr(self) -> float:
        """
        Learning rate schedule (cosine with warmup).

        Warmup: linearly increase from 0 to lr
        Then: cosine decay from lr to 0.1 * lr

        This is the standard schedule for LLM training.
        """
        if self.step < self.warmup_steps:
            # Linear warmup
            return self.lr * (self.step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    def train_step(
        self,
        token_ids: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Single training step.

        Args:
            token_ids: Input tokens [batch × seq_len]
            targets: Target tokens [batch × seq_len]

        Returns:
            loss: Training loss for this step

        In practice, this would use:
        - PyTorch autograd for backpropagation
        - AdamW optimizer
        - Mixed precision (fp16)
        - Gradient accumulation
        """
        # Forward pass
        logits, loss = self.model.forward(token_ids, targets)

        # In real implementation:
        # 1. loss.backward() — compute gradients
        # 2. clip_grad_norm_(model.parameters(), grad_clip)
        # 3. optimizer.step() — update weights
        # 4. optimizer.zero_grad() — reset gradients

        self.step += 1
        return loss


################################################################################
# SECTION 4: TESTING & EXAMPLES
################################################################################

def demonstrate_model():
    """Demonstrate the complete transformer model."""
    print("=" * 70)
    print("TRANSFORMER LANGUAGE MODEL DEMONSTRATION")
    print("=" * 70)

    # Create a small model for demonstration
    print("\n--- Creating Model ---")
    model = TransformerLM(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,  # GQA
        max_seq_len=256
    )

    # Forward pass
    print("\n--- Forward Pass ---")
    batch_size = 2
    seq_len = 8
    token_ids = np.random.randint(0, 1000, (batch_size, seq_len))
    targets = np.random.randint(0, 1000, (batch_size, seq_len))

    logits, loss = model.forward(token_ids, targets)
    print(f"Input shape: {token_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss:.4f}")

    # Generation
    print("\n--- Text Generation ---")
    prompt = np.array([[1, 5, 42]])  # Some prompt tokens
    generated = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )
    print(f"Prompt: {prompt[0].tolist()}")
    print(f"Generated: {generated[0].tolist()}")

    # KV Cache
    print("\n--- KV Cache ---")
    cache = KVCache(
        n_layers=4,
        n_kv_heads=2,
        d_k=32,  # d_model / n_heads
        max_seq_len=256
    )
    print(f"KV Cache created for {cache.n_layers} layers")

    # Training
    print("\n--- Training Pipeline ---")
    trainer = Trainer(model, learning_rate=3e-4, max_steps=1000)
    print(f"Initial LR: {trainer.get_lr():.6f}")
    trainer.step = 500
    print(f"Mid-training LR: {trainer.get_lr():.6f}")
    trainer.step = 1000
    print(f"Final LR: {trainer.get_lr():.6f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_model()


################################################################################
# REFERENCES
################################################################################

# [1] Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners.
# [2] Brown, T., et al. (2020). Language Models are Few-Shot Learners.
# [3] Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models.
# [4] Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models.
# [5] Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models.

################################################################################
