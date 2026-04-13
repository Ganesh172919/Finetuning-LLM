"""
RWKV-v6: Reinvention of RNN for the Transformer Era (Extended Implementation)
==============================================================================

RWKV (Receptance Weighted Key Value) is a novel architecture that combines
the training parallelism of Transformers with the inference efficiency of RNNs.

RWKV-v6 introduces several key improvements:
- Data-dependent linear attention with improved expressivity
- LoRA-based time-mixing for better parameter efficiency
- Enhanced token-shift mechanism
- Improved training stability

Architecture Overview:
┌─────────────────────────────────────────────────────────────┐
│                    RWKV-v6 Block                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Token Shift │───▶│ Time Mixing │───▶│Channel Mix  │     │
│  │  (Attention) │    │  (WKV)      │    │  (FFN)      │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │              │
│         ▼                  ▼                  ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  LayerNorm   │    │  LayerNorm   │    │  LayerNorm   │    │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘

Key Innovation — WKV Mechanism:
  Unlike standard attention O(n²), RWKV computes:
  wkv_t = (Σ_{i=1}^{t-1} e^{-(t-1-i)w + k_i} v_i + e^{u+k_t} v_t)
          / (Σ_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t})

  This can be computed recurrently: O(n) time, O(1) per step
  This can be computed in parallel: O(n) time with scan

References:
  - RWKV: Reinventing RNNs for the Transformer Era (Peng et al., 2023)
  - RWKV-v6: Eagle and Finch (Peng et al., 2024)
  - Linear Transformers with Learnable Kernel Functions (Katharopoulos et al., 2020)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RWKVv6Config:
    """
    Configuration for RWKV-v6 model.

    Attributes:
        vocab_size: Size of vocabulary
        d_model: Model dimension (embedding size)
        n_layers: Number of RWKV blocks
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        layer_norm_eps: LayerNorm epsilon for numerical stability
        context_length: Maximum sequence length
        time_shift_size: Token shift window size (typically 1 or 2)
        beta: LoRA beta for time mixing
        lwkv: Number of LoRA ranks for WKV
        tiny_att_dim: Dimension for tiny attention (v6 innovation)
        tiny_att_layer: Layer index to start using tiny attention
    """
    vocab_size: int = 50277
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    context_length: int = 2048
    time_shift_size: int = 1
    beta: float = 0.5
    lwkv: int = 32
    tiny_att_dim: int = 0  # 0 = disabled
    tiny_att_layer: int = -1

    def head_size(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """
    Layer Normalization.

    Normalizes input across the last dimension:
        y = (x - mean) / sqrt(var + eps) * weight + bias

    Args:
        x: Input tensor [..., d_model]
        weight: Scale parameter [d_model]
        bias: Shift parameter [d_model]
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor same shape as x
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * weight + bias


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation with numerical stability."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def token_shift(x: np.ndarray, prev_x: np.ndarray,
                shift_ratio: float = 0.5) -> np.ndarray:
    """
    Token Shift — RWKV's mechanism for mixing adjacent tokens.

    Instead of explicit attention, RWKV mixes each token with the previous
    token using a simple interpolation:
        shifted = shift_ratio * x_prev + (1 - shift_ratio) * x_current

    This gives the model a "memory" of the previous token without
    requiring quadratic attention.

    Args:
        x: Current token [..., d_model]
        prev_x: Previous token [..., d_model]
        shift_ratio: Mixing ratio (0.5 = equal mix)

    Returns:
        Shifted token representation
    """
    return shift_ratio * prev_x + (1 - shift_ratio) * x


# ============================================================================
# WKV (Weighted Key-Value) MECHANISM
# ============================================================================

def wkv_forward_serial(
    time_decay: np.ndarray,   # [n_heads, head_size] — per-channel decay
    time_first: np.ndarray,   # [n_heads, head_size] — bonus for current token
    k: np.ndarray,            # [T, n_heads, head_size] — keys
    v: np.ndarray,            # [T, n_heads, head_size] — values
) -> np.ndarray:
    """
    WKV computation in serial (RNN-style) — for understanding.

    The WKV formula for each time step t:
        numerator_t = Σ_{i=1}^{t-1} e^{-(t-1-i)w + k_i} v_i + e^{u+k_t} v_t
        denominator_t = Σ_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t}

    Where:
        w = time_decay (learnable per-channel decay rate)
        u = time_first (bonus weight for current token)
        k = key projections
        v = value projections

    This is O(T) per step, O(T²) total — but can be parallelized with scan.

    Args:
        time_decay: Decay rates [n_heads, head_size]
        time_first: Current token bonus [n_heads, head_size]
        k: Key tensor [T, n_heads, head_size]
        v: Value tensor [T, n_heads, head_size]

    Returns:
        WKV output [T, n_heads, head_size]
    """
    T, n_heads, head_size = k.shape
    wkv = np.zeros_like(k)

    # Running state: numerator and denominator
    num = np.zeros((n_heads, head_size))  # Σ e^{...} v_i
    den = np.zeros((n_heads, head_size))  # Σ e^{...}

    for t in range(T):
        # Current token: e^{u + k_t} * v_t
        current_num = np.exp(time_first + k[t]) * v[t]
        current_den = np.exp(time_first + k[t])

        # WKV_t = (accumulated + current_num) / (accumulated + current_den)
        wkv[t] = (num + current_num) / (den + current_den)

        # Update running state with decay
        # Multiply accumulated by e^{-w} (decay), then add current
        num = num * np.exp(-time_decay) + np.exp(k[t]) * v[t]
        den = den * np.exp(-time_decay) + np.exp(k[t])

    return wkv


def wkv_forward_parallel(
    time_decay: np.ndarray,
    time_first: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    WKV computation in parallel (scan-style) — for training.

    Uses parallel prefix scan to compute all WKV values simultaneously.
    This is the key insight that makes RWKV trainable like a Transformer.

    The trick: we can reformulate the recurrence as a linear recurrence
    that admits parallel scan computation.

    State at time t: s_t = e^{-w} * s_{t-1} + e^{k_t} * v_t

    This is an associative operation when reformulated as:
    (a2, b2) ∘ (a1, b1) = (a2 * a1, a2 * b1 + b2)

    Args:
        time_decay: Decay rates [n_heads, head_size]
        time_first: Current token bonus [n_heads, head_size]
        k: Key tensor [T, n_heads, head_size]
        v: Value tensor [T, n_heads, head_size]

    Returns:
        WKV output [T, n_heads, head_size]
    """
    T, n_heads, head_size = k.shape

    # For simplicity, use serial implementation
    # (Production code would use parallel scan)
    return wkv_forward_serial(time_decay, time_first, k, v)


# ============================================================================
# RWKV-v6 TIME MIXING (ATTENTION LAYER)
# ============================================================================

class RWKVTimeMixingv6:
    """
    RWKV-v6 Time Mixing Layer — the core attention mechanism.

    This implements the data-dependent linear attention of RWKV-v6,
    which improves over v5 by using LoRA-based parameterization
    for better expressivity with fewer parameters.

    Forward pass:
        1. Token shift: mix current and previous tokens
        2. Compute R, K, V projections (with LoRA in v6)
        3. Apply WKV mechanism
        4. Output projection with gating

    Parameters (v6 style with LoRA):
        x_r, x_k, x_v, x_w: Token shift ratios (learnable)
        r_proj, k_proj, v_proj: LoRA projection pairs
        w_proj: Time decay LoRA projection
        o_proj: Output projection
        ln_x: LayerNorm for stabilization

    Complexity:
        Time: O(T * d) — linear in sequence length
        Space: O(T * d) — for storing activations
    """

    def __init__(self, config: RWKVv6Config, layer_idx: int):
        """
        Initialize time mixing layer.

        Args:
            config: Model configuration
            layer_idx: Layer index (affects initialization)
        """
        self.config = config
        self.layer_idx = layer_idx
        d_model = config.d_model
        n_heads = config.n_heads
        head_size = config.head_size()

        # ── Token shift parameters ──────────────────────────────
        # These control how much to mix with previous token
        # Initialized differently per layer for diversity
        ratio = 1.0 - layer_idx / config.n_layers
        self.x_r = np.ones(d_model) * ratio
        self.x_k = np.ones(d_model) * ratio
        self.x_v = np.ones(d_model) * ratio
        self.x_w = np.ones(d_model) * ratio

        # ── LoRA projections for R, K, V (v6 innovation) ────────
        # Instead of direct projection, v6 uses low-rank decomposition
        # This reduces parameters while maintaining expressivity
        lora_rank = config.lwkv

        # R (Receptance) — controls how much to attend to WKV
        self.r_proj_lo = np.random.randn(d_model, lora_rank) * 0.01
        self.r_proj_hi = np.random.randn(lora_rank, d_model) * 0.01

        # K (Key) — used in WKV scoring
        self.k_proj_lo = np.random.randn(d_model, lora_rank) * 0.01
        self.k_proj_hi = np.random.randn(lora_rank, d_model) * 0.01

        # V (Value) — information to retrieve
        self.v_proj_lo = np.random.randn(d_model, lora_rank) * 0.01
        self.v_proj_hi = np.random.randn(lora_rank, d_model) * 0.01

        # W (Time decay) — learned per-channel decay rates
        self.w_proj_lo = np.random.randn(d_model, lora_rank) * 0.01
        self.w_proj_hi = np.random.randn(lora_rank, d_model) * 0.01

        # ── Time decay base ──────────────────────────────────────
        # Initialized to allow learning both short and long-range dependencies
        # Using log-space for numerical stability
        self.time_decay = np.random.randn(n_heads, head_size) * 0.5 - 5.0

        # ── Time first (bonus for current token) ─────────────────
        # This gives extra weight to the current token in WKV
        self.time_first = np.random.randn(n_heads, head_size) * 0.1

        # ── Output projection ────────────────────────────────────
        self.o_proj = np.random.randn(d_model, d_model) * 0.01

        # ── LayerNorm for stabilization ──────────────────────────
        self.ln_w = np.ones(d_model)
        self.ln_b = np.zeros(d_model)

    def forward(self, x: np.ndarray, state: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of RWKV-v6 time mixing.

        Args:
            x: Input tensor [T, d_model]
            state: Optional RNN state from previous chunk

        Returns:
            output: Transformed tensor [T, d_model]
            new_state: Updated RNN state
        """
        T, d_model = x.shape
        n_heads = self.config.n_heads
        head_size = self.config.head_size()

        # ── Step 1: Token Shift ──────────────────────────────────
        # Mix each token with its predecessor
        # This is RWKV's simple alternative to positional encoding
        if state is not None:
            prev_x = state  # Use stored state for continuation
        else:
            prev_x = np.zeros_like(x)

        # Shift with learnable mixing ratios
        x_shifted_r = token_shift(x, np.roll(x, 1, axis=0), 0.5)
        x_shifted_k = token_shift(x, np.roll(x, 1, axis=0), 0.5)
        x_shifted_v = token_shift(x, np.roll(x, 1, axis=0), 0.5)
        x_shifted_w = token_shift(x, np.roll(x, 1, axis=0), 0.5)

        # Apply learnable shift ratios
        xr = x * (1 - self.x_r) + x_shifted_r * self.x_r
        xk = x * (1 - self.x_k) + x_shifted_k * self.x_k
        xv = x * (1 - self.x_v) + x_shifted_v * self.x_v
        xw = x * (1 - self.x_w) + x_shifted_w * self.x_w

        # ── Step 2: LoRA Projections (v6 innovation) ─────────────
        # R, K, V projections through low-rank bottleneck
        r = sigmoid(xr @ self.r_proj_lo @ self.r_proj_hi)  # Receptance [0,1]
        k = xk @ self.k_proj_lo @ self.k_proj_hi           # Key
        v = xv @ self.v_proj_lo @ self.v_proj_hi           # Value
        w = np.tanh(xw @ self.w_proj_lo @ self.w_proj_hi)  # Time decay modulation

        # Reshape for multi-head: [T, d_model] -> [T, n_heads, head_size]
        r = r.reshape(T, n_heads, head_size)
        k = k.reshape(T, n_heads, head_size)
        v = v.reshape(T, n_heads, head_size)
        w = w.reshape(T, n_heads, head_size)

        # Modulate time decay with input-dependent component
        time_decay = self.time_decay + w  # [T, n_heads, head_size]

        # ── Step 3: WKV Computation ──────────────────────────────
        # This is the core of RWKV — data-dependent linear attention
        wkv = wkv_forward_parallel(
            self.time_decay,  # Base decay rates
            self.time_first,  # Current token bonus
            k, v
        )

        # ── Step 4: Receptance Gating ────────────────────────────
        # R controls how much of the WKV output to keep
        # This is analogous to the forget gate in LSTMs
        output = r * wkv  # [T, n_heads, head_size]

        # Reshape back: [T, n_heads, head_size] -> [T, d_model]
        output = output.reshape(T, d_model)

        # ── Step 5: Output Projection ────────────────────────────
        output = layer_norm(output, self.ln_w, self.ln_b)
        output = output @ self.o_proj

        # Save last token as new state
        new_state = x[-1:]  # [1, d_model]

        return output, new_state


# ============================================================================
# RWKV-v6 CHANNEL MIXING (FFN LAYER)
# ============================================================================

class RWKVChannelMixingv6:
    """
    RWKV-v6 Channel Mixing Layer — replaces standard FFN.

    Instead of a standard 2-layer MLP, RWKV uses a gated mechanism:
        output = r * (k ** 2 @ v_proj)

    The squaring of k ensures non-negative activations, providing
    a form of sparsity and improved training dynamics.

    Key difference from standard FFN:
        Standard: output = activation(W1 @ x) @ W2
        RWKV:     output = sigmoid(W_r @ x_shifted) * (W_k @ x_shifted)² @ W_v

    Parameters:
        x_r, x_k: Token shift ratios
        r_proj: Receptance (gate) projection
        k_proj: Key projection (squared activation)
        v_proj: Value projection
    """

    def __init__(self, config: RWKVv6Config, layer_idx: int):
        d_model = config.d_model
        d_ff = config.d_ff

        # Token shift
        ratio = 1.0 - layer_idx / config.n_layers
        self.x_r = np.ones(d_model) * ratio
        self.x_k = np.ones(d_model) * ratio

        # Projections
        self.r_proj = np.random.randn(d_model, d_ff) * 0.01
        self.k_proj = np.random.randn(d_model, d_ff) * 0.01
        self.v_proj = np.random.randn(d_ff, d_model) * 0.01

    def forward(self, x: np.ndarray, state: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of channel mixing.

        Args:
            x: Input [T, d_model]
            state: Optional previous token state

        Returns:
            output: Transformed [T, d_model]
            new_state: Updated state
        """
        # Token shift
        if state is not None:
            prev_x = state
        else:
            prev_x = np.zeros_like(x)

        x_shifted = token_shift(x, np.roll(x, 1, axis=0), 0.5)

        # Apply learnable shift
        xr = x * (1 - self.x_r) + x_shifted * self.x_r
        xk = x * (1 - self.x_k) + x_shifted * self.x_k

        # Channel mixing with squared activation
        r = sigmoid(xr @ self.r_proj)    # Gate [T, d_ff]
        k = (xk @ self.k_proj) ** 2      # Squared activation [T, d_ff]
        output = (r * k) @ self.v_proj   # Gated output [T, d_model]

        new_state = x[-1:]
        return output, new_state


# ============================================================================
# RWKV-v6 BLOCK
# ============================================================================

class RWKVBlockv6:
    """
    Single RWKV-v6 Transformer block.

    Combines time mixing (attention) and channel mixing (FFN)
    with residual connections and layer normalization.

    Structure:
        x = x + TimeMixing(LayerNorm(x))
        x = x + ChannelMixing(LayerNorm(x))

    This mirrors the Transformer block structure but uses
    linear-complexity mechanisms instead of quadratic attention.
    """

    def __init__(self, config: RWKVv6Config, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx

        # Layer norms
        self.ln1_w = np.ones(config.d_model)
        self.ln1_b = np.zeros(config.d_model)
        self.ln2_w = np.ones(config.d_model)
        self.ln2_b = np.zeros(config.d_model)

        # Time mixing (attention replacement)
        self.time_mixing = RWKVTimeMixingv6(config, layer_idx)

        # Channel mixing (FFN replacement)
        self.channel_mixing = RWKVChannelMixingv6(config, layer_idx)

    def forward(self, x: np.ndarray,
                state: Optional[Tuple[np.ndarray, np.ndarray]] = None
                ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass through one RWKV block.

        Args:
            x: Input [T, d_model]
            state: Tuple of (time_mix_state, channel_mix_state)

        Returns:
            output: Transformed [T, d_model]
            new_state: Updated states
        """
        tm_state = state[0] if state is not None else None
        cm_state = state[1] if state is not None else None

        # Time mixing with residual
        ln1_out = layer_norm(x, self.ln1_w, self.ln1_b)
        tm_out, new_tm_state = self.time_mixing.forward(ln1_out, tm_state)
        x = x + tm_out

        # Channel mixing with residual
        ln2_out = layer_norm(x, self.ln2_w, self.ln2_b)
        cm_out, new_cm_state = self.channel_mixing.forward(ln2_out, cm_state)
        x = x + cm_out

        return x, (new_tm_state, new_cm_state)


# ============================================================================
# RWKV-v6 MODEL
# ============================================================================

class RWKVv6Model:
    """
    Complete RWKV-v6 Language Model.

    Architecture:
        Token Embedding → N × RWKVBlock → LayerNorm → Output Head

    Key Properties:
        - Training: O(T * d²) — parallel, like Transformer
        - Inference: O(d²) per token — recurrent, like RNN
        - Memory: O(d) per token at inference — constant KV cache

    This makes RWKV ideal for:
        - Long context processing (no quadratic blowup)
        - Efficient inference (constant memory per step)
        - Streaming applications (process one token at a time)
    """

    def __init__(self, config: RWKVv6Config):
        self.config = config

        # Token embedding
        self.embedding = np.random.randn(config.vocab_size, config.d_model) * 0.01

        # Stack of RWKV blocks
        self.blocks = [RWKVBlockv6(config, i) for i in range(config.n_layers)]

        # Final layer norm
        self.ln_out_w = np.ones(config.d_model)
        self.ln_out_b = np.zeros(config.d_model)

        # Language model head (usually tied to embedding)
        self.lm_head = self.embedding.T  # Weight tying

    def forward(self, input_ids: np.ndarray,
                states: Optional[List] = None
                ) -> Tuple[np.ndarray, List]:
        """
        Forward pass through the complete model.

        Args:
            input_ids: Token indices [T]
            states: Optional list of states per layer

        Returns:
            logits: Output logits [T, vocab_size]
            new_states: Updated states per layer
        """
        # Token embedding
        x = self.embedding[input_ids]  # [T, d_model]

        # Process through all blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            state = states[i] if states is not None else None
            x, new_state = block.forward(x, state)
            new_states.append(new_state)

        # Final layer norm + output head
        x = layer_norm(x, self.ln_out_w, self.ln_out_b)
        logits = x @ self.lm_head  # [T, vocab_size]

        return logits, new_states

    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 50
                 ) -> np.ndarray:
        """
        Autoregressive text generation.

        Because RWKV is recurrent, each step is O(1) — no KV cache growth!

        Args:
            input_ids: Initial tokens [T]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Generated token sequence
        """
        generated = list(input_ids)
        states = None

        for _ in range(max_new_tokens):
            # Forward pass (only need last token's output)
            logits, states = self.forward(
                np.array([generated[-1]]),
                states
            )

            # Apply temperature
            logits = logits[-1] / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_idx = np.argsort(logits)[-top_k:]
                mask = np.full_like(logits, -np.inf)
                mask[top_k_idx] = logits[top_k_idx]
                logits = mask

            # Sample
            probs = softmax(logits)
            next_token = np.random.choice(len(probs), p=probs)
            generated.append(next_token)

        return np.array(generated)


# ============================================================================
# RWKV-v6 TRAINING
# ============================================================================

class RWKv6Trainer:
    """
    Training loop for RWKV-v6.

    RWKV can be trained in parallel (like Transformer) because the
    WKV computation can be expressed as a parallel scan operation.

    Training techniques:
        - Gradient clipping for stability
        - Learning rate warmup + cosine decay
        - Mixed precision (when using frameworks)
        - Gradient accumulation for large batches
    """

    def __init__(self, model: RWKVv6Model, learning_rate: float = 1e-4):
        self.model = model
        self.lr = learning_rate
        self.step_count = 0

    def compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Cross-entropy loss for language modeling.

        Args:
            logits: Model output [T, vocab_size]
            targets: Target token indices [T]

        Returns:
            Scalar loss value
        """
        # Numerically stable log-softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        log_probs = logits - logits_max - np.log(
            np.sum(np.exp(logits - logits_max), axis=-1, keepdims=True)
        )

        # Gather target log-probs
        T = len(targets)
        target_log_probs = log_probs[np.arange(T), targets]

        return -np.mean(target_log_probs)

    def train_step(self, input_ids: np.ndarray, targets: np.ndarray) -> float:
        """
        Single training step (forward + loss).

        Note: Full backpropagation requires a deep learning framework.
              This demonstrates the forward pass and loss computation.

        Args:
            input_ids: Input tokens [T]
            targets: Target tokens [T]

        Returns:
            Loss value
        """
        logits, _ = self.model.forward(input_ids)
        loss = self.compute_loss(logits, targets)
        self.step_count += 1
        return loss


# ============================================================================
# COMPARISON: RWKV vs Transformer vs Mamba
# ============================================================================

def compare_architectures():
    """
    Compare key properties of RWKV, Transformer, and Mamba.

    Returns comparison table as string.
    """
    comparison = """
    ┌──────────────────┬─────────────┬─────────────┬─────────────┐
    │ Property         │ Transformer │ RWKV-v6     │ Mamba-2     │
    ├──────────────────┼─────────────┼─────────────┼─────────────┤
    │ Training Time    │ O(T²d)      │ O(Td)       │ O(Td)       │
    │ Inference/step   │ O(Td)       │ O(d)        │ O(d)        │
    │ Memory (infer)   │ O(Td)       │ O(d)        │ O(d)        │
    │ Long Context     │ Quadratic   │ Linear      │ Linear      │
    │ Parallelizable   │ Yes         │ Yes (scan)  │ Yes (scan)  │
    │ Attention Type   │ Full        │ Linear WKV  │ Selective   │
    │ Positional Enc   │ RoPE/ALiBi  │ Token Shift │ Implicit    │
    │ Gating           │ Softmax     │ Sigmoid(R)  │ SSM gates   │
    │ Key Innovation   │ QKV Attn    │ WKV + Recur │ Selective   │
    │ Production Use   │ Ubiquitous  │ Growing     │ Growing     │
    └──────────────────┴─────────────┴─────────────┴─────────────┘
    """
    return comparison


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_rwkv_v6():
    """
    Demonstrate RWKV-v6 on a simple task.

    Shows:
        1. Model initialization
        2. Forward pass
        3. Generation
        4. Architecture comparison
    """
    print("=" * 70)
    print("RWKV-v6: Reinvention of RNN for the Transformer Era")
    print("=" * 70)

    # Configuration
    config = RWKVv6Config(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        context_length=128
    )

    print(f"\nModel Config:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")
    print(f"  Head size: {config.head_size()}")

    # Create model
    model = RWKVv6Model(config)

    # Count parameters
    param_count = (
        config.vocab_size * config.d_model +  # Embedding
        config.n_layers * (
            4 * config.d_model * config.lwkv +  # LoRA projections
            4 * config.lwkv * config.d_model +
            config.d_model * config.d_model +   # Output proj
            config.d_model * config.d_ff * 2 +  # Channel mixing
            config.d_model * 4                   # LayerNorms
        )
    )
    print(f"\n  Estimated parameters: {param_count:,}")

    # Forward pass
    print("\n[Forward Pass]")
    input_ids = np.array([65, 66, 67, 68])  # ABCD
    logits, states = model.forward(input_ids)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  States: {len(states)} layers, each with 2 states")

    # Generation
    print("\n[Generation]")
    prompt = np.array([65, 66, 67])  # ABC
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"  Prompt: {prompt}")
    print(f"  Generated: {generated}")

    # Architecture comparison
    print("\n[Architecture Comparison]")
    print(compare_architectures())

    # Training demo
    print("\n[Training Demo]")
    trainer = RWKv6Trainer(model, learning_rate=1e-4)
    targets = np.array([66, 67, 68, 69])  # BCDE
    loss = trainer.train_step(input_ids, targets)
    print(f"  Loss: {loss:.4f}")
    print(f"  Perplexity: {np.exp(loss):.2f}")

    print("\n" + "=" * 70)
    print("RWKV-v6 Key Insights:")
    print("  1. Combines Transformer parallelism with RNN efficiency")
    print("  2. O(n) training, O(1) inference per token")
    print("  3. No KV cache growth — ideal for long contexts")
    print("  4. Token shift replaces positional encoding")
    print("  5. WKV mechanism is the core innovation")
    print("=" * 70)


if __name__ == "__main__":
    demo_rwkv_v6()
