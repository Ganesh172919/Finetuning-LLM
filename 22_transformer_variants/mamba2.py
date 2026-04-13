"""
################################################################################
MAMBA-2 — SELECTIVE STATE SPACE MODEL (2024-2025 SOTA)
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Mamba-2?
    Mamba-2 is the second generation of the Mamba architecture, a selective
    state space model (SSM) that achieves transformer-level quality with
    O(n) complexity instead of O(n²). It was published by Albert Gu and
    Tri Dao in 2024.

    Key Innovation: Structured State Space Duality (SSD)
    Mamba-2 shows that selective SSMs and attention are mathematically
    equivalent — they're both doing the same computation in different
    representations. This insight enables hardware-efficient implementations.

Why Mamba-2 matters:
    - Linear scaling: O(n) vs O(n²) for attention
    - Long context: Efficiently handles 100K+ tokens
    - Quality: Matches or exceeds transformers on language modeling
    - Speed: 5-8x faster than transformers for long sequences
    - Memory: O(n) vs O(n²) for attention

Architecture Overview:
    ┌─────────────────────────────────────────────────────────┐
    │                    Mamba-2 Block                         │
    │                                                          │
    │  Input x                                                 │
    │     │                                                    │
    │     ├──▶ Linear Projection ──▶ Split ──┬── z (gate)     │
    │     │                                 └── B, C, dt      │
    │     │                                      │             │
    │     │                                      ▼             │
    │     │                              Selective SSM         │
    │     │                              (recurrent scan)      │
    │     │                                      │             │
    │     │                                      ▼             │
    │     │                                 y = SSM(x)         │
    │     │                                      │             │
    │     │                                      ▼             │
    │     │                                 y ⊙ silu(z)        │
    │     │                                      │             │
    │     │                                      ▼             │
    │     └──▶ Linear Projection ◀──────────────┘             │
    │                │                                         │
    │                ▼                                         │
    │            Output + Residual                             │
    └─────────────────────────────────────────────────────────┘

Interview Questions:
    Q: "What is Mamba and how does it differ from transformers?"
    A: Mamba is a selective state space model. Instead of attention,
       it uses a recurrent state that selectively remembers or forgets
       information based on the input. This gives O(n) complexity
       while maintaining transformer-level quality.

    Q: "What is Structured State Space Duality (SSD)?"
    A: SSD shows that selective SSMs and linear attention are
       mathematically equivalent. This means we can implement SSMs
       using efficient matrix multiplication kernels, getting the
       best of both worlds: recurrence for efficiency, matmul for
       hardware utilization.

    Q: "When would you choose Mamba over a transformer?"
    A: For long sequences (100K+ tokens), real-time applications,
       or when memory is constrained. Transformers are still better
       for tasks requiring precise recall of specific tokens.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple
import math

################################################################################
# SECTION 1: SELECTIVE SCAN (CORE SSM OPERATION)
################################################################################

def selective_scan(
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Selective Scan — The core operation of Mamba.

    This is the recurrent computation that makes Mamba work.
    It processes the input sequence step by step, maintaining
    a hidden state that selectively remembers relevant information.

    Algorithm:
        For each time step t:
        1. Discretize: A_bar = exp(dt * A), B_bar = dt * B
        2. Update state: h_t = A_bar * h_{t-1} + B_bar * x_t
        3. Output: y_t = C_t * h_t

    The "selective" part: B, C, and dt are input-dependent,
    meaning the model learns WHEN and WHAT to remember.

    Args:
        x: [batch × seq_len × d_inner] input sequence
        A: [d_inner × d_state] state transition matrix
        B: [batch × seq_len × d_state] input-dependent input matrix
        C: [batch × seq_len × d_state] input-dependent output matrix
        dt: [batch × seq_len × d_inner] input-dependent time step

    Returns:
        y: [batch × seq_len × d_inner] output sequence

    Time Complexity: O(batch × seq_len × d_inner × d_state)
    Space Complexity: O(batch × d_inner × d_state) for hidden state

    Interview Question:
        Q: "Explain the selective scan operation."
        A: It's a recurrent scan where the transition matrices (B, C)
           and time steps (dt) are input-dependent. This means the
           model can decide at each step what to remember and what
           to forget — unlike traditional SSMs with fixed matrices.
    """
    batch, seq_len, d_inner = x.shape
    d_state = A.shape[1]

    # Initialize hidden state
    h = np.zeros((batch, d_inner, d_state))
    outputs = []

    for t in range(seq_len):
        # Discretize continuous parameters
        # A_bar = exp(dt * A) — how much to retain from previous state
        dt_t = dt[:, t, :].reshape(batch, d_inner, 1)  # [batch × d_inner × 1]
        A_bar = np.exp(dt_t * A.reshape(1, d_inner, d_state))  # [batch × d_inner × d_state]

        # B_bar = dt * B — how much of current input to incorporate
        B_t = B[:, t, :].reshape(batch, 1, d_state)  # [batch × 1 × d_state]
        B_bar = dt_t * B_t  # [batch × d_inner × d_state]

        # State update: h_t = A_bar * h_{t-1} + B_bar * x_t
        x_t = x[:, t, :].reshape(batch, d_inner, 1)  # [batch × d_inner × 1]
        h = A_bar * h + B_bar * x_t  # [batch × d_inner × d_state]

        # Output: y_t = C_t * h_t
        C_t = C[:, t, :].reshape(batch, 1, d_state)  # [batch × 1 × d_state]
        y_t = np.sum(C_t * h, axis=-1)  # [batch × d_inner]
        outputs.append(y_t)

    return np.stack(outputs, axis=1)  # [batch × seq_len × d_inner]


################################################################################
# SECTION 2: MAMBA-2 BLOCK
################################################################################

class Mamba2Block:
    """
    Mamba-2 Block — Selective State Space Model Block.

    This is the core building block of the Mamba-2 architecture.
    It combines:
    1. Input projection (expand dimensions)
    2. Selective SSM (the "thinking" part)
    3. Gating (control information flow)
    4. Output projection (back to model dimension)

    Key Design Decisions:
    - Expansion factor: d_inner = 2 * d_model (like SwiGLU in transformers)
    - Conv1d: local pattern detection before SSM
    - SiLU gating: smooth activation for better gradient flow
    - Residual connection: stable training

    Interview Question:
        Q: "Walk me through a Mamba-2 block."
        A: Input goes through two paths: (1) linear → conv1d → SSM,
           (2) linear → SiLU gate. The SSM output is multiplied by
           the gate, then projected back to model dimension. Residual
           connection adds the original input.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,      # SSM state dimension
        d_conv: int = 4,         # Local convolution width
        expand: int = 2,         # Expansion factor
        dt_rank: str = "auto",   # Rank of dt projection
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        # dt rank: controls the complexity of the time step projection
        if dt_rank == "auto":
            self.dt_rank = max(1, d_model // 16)
        else:
            self.dt_rank = int(dt_rank)

        # === Input Projections ===
        # Project input to higher dimension for SSM processing
        self.in_proj_weight = np.random.randn(d_model, self.d_inner * 2) * 0.02
        # Splits into: x_proj (for SSM) and z_proj (for gating)

        # === Conv1d for Local Patterns ===
        # 1D convolution captures local patterns before global SSM
        self.conv1d_weight = np.random.randn(self.d_inner, 1, d_conv) * 0.02
        self.conv1d_bias = np.zeros(self.d_inner)

        # === SSM Parameters ===
        # A: state transition matrix (learned, initialized carefully)
        # Uses log-space initialization for numerical stability
        A = np.arange(1, d_state + 1, dtype=np.float32)
        self.A_log = np.log(np.tile(A, (self.d_inner, 1)))  # [d_inner × d_state]

        # B, C, dt projections (input-dependent)
        self.B_proj = np.random.randn(self.d_inner, d_state) * 0.02
        self.C_proj = np.random.randn(self.d_inner, d_state) * 0.02
        self.dt_proj = np.random.randn(self.dt_rank, self.d_inner) * 0.02

        # dt bias: controls the base time step
        # Initialize with specific range for good training dynamics
        dt_init = np.exp(
            np.random.uniform(
                np.log(0.001), np.log(0.1), size=(self.d_inner,)
            )
        ) - 0.001
        self.dt_bias = dt_init

        # === Output Projection ===
        self.out_proj_weight = np.random.randn(self.d_inner, d_model) * 0.02

        # === Layer Norm ===
        self.norm_weight = np.ones(d_model)
        self.norm_bias = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through Mamba-2 block.

        Args:
            x: [batch × seq_len × d_model]

        Returns:
            output: [batch × seq_len × d_model]

        Flow:
            x → LayerNorm → in_proj → split(x_proj, z_proj)
            x_proj → conv1d → SiLU → SSM → y
            z_proj → SiLU → gate
            y * gate → out_proj → + residual
        """
        batch, seq_len, d_model = x.shape
        residual = x

        # === Layer Norm ===
        x_norm = self._layer_norm(x)

        # === Input Projection ===
        # Project to 2 * d_inner, then split for SSM and gating
        xz = x_norm @ self.in_proj_weight  # [batch × seq × 2*d_inner]
        x_proj = xz[:, :, :self.d_inner]    # For SSM
        z_proj = xz[:, :, self.d_inner:]    # For gating

        # === Conv1d (Local Pattern Detection) ===
        # Transpose for conv: [batch × d_inner × seq_len]
        x_conv = np.transpose(x_proj, (0, 2, 1))
        x_conv = self._conv1d(x_conv, self.conv1d_weight, self.conv1d_bias)
        x_conv = np.transpose(x_conv, (0, 2, 1))  # Back to [batch × seq × d_inner]

        # SiLU activation
        x_ssm_input = self._silu(x_conv)

        # === Selective SSM ===
        # Compute input-dependent parameters
        A = -np.exp(self.A_log)  # Negative for stability
        B = x_ssm_input @ self.B_proj  # [batch × seq × d_state]
        C = x_ssm_input @ self.C_proj  # [batch × seq × d_state]

        # dt: time step (must be positive)
        dt_rank_proj = np.random.randn(self.d_model, self.dt_rank) * 0.02
        dt = x_norm @ dt_rank_proj @ self.dt_proj + self.dt_bias  # [batch × seq × d_inner]
        dt = np.log(1 + np.exp(dt))  # Softplus to ensure positivity

        # Run selective scan
        y = selective_scan(x_ssm_input, A, B, C, dt)  # [batch × seq × d_inner]

        # === Gating ===
        # SiLU gate controls information flow
        y = y * self._silu(z_proj)

        # === Output Projection ===
        output = y @ self.out_proj_weight  # [batch × seq × d_model]

        # === Residual Connection ===
        return output + residual

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return x_norm * self.norm_weight + self.norm_bias

    def _silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation: x * sigmoid(x)."""
        return x / (1 + np.exp(-np.clip(x, -20, 20)))

    def _conv1d(
        self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray
    ) -> np.ndarray:
        """Simple 1D convolution (causal)."""
        batch, channels, seq_len = x.shape
        kernel_size = weight.shape[2]
        output = np.zeros_like(x)

        for t in range(seq_len):
            for k in range(kernel_size):
                if t - k >= 0:
                    output[:, :, t] += x[:, :, t - k] * weight[:, 0, k]

        return output + bias.reshape(1, -1, 1)


################################################################################
# SECTION 3: MAMBA-2 MODEL
################################################################################

class Mamba2Model:
    """
    Complete Mamba-2 Language Model.

    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    Mamba-2 LM                            │
    │                                                          │
    │  Token Embedding                                         │
    │       │                                                  │
    │       ▼                                                  │
    │  Mamba Block × N  (no attention needed!)                 │
    │       │                                                  │
    │       ▼                                                  │
    │  Layer Norm                                              │
    │       │                                                  │
    │       ▼                                                  │
    │  Language Model Head                                      │
    │       │                                                  │
    │       ▼                                                  │
    │  Output Logits                                           │
    └─────────────────────────────────────────────────────────┘

    Key Differences from Transformers:
    - No attention layers (replaced by SSM)
    - O(n) complexity per layer (not O(n²))
    - Recurrent inference (constant memory per step)
    - Parallel training (scan can be parallelized)

    Interview Question:
        Q: "How does Mamba-2 handle long sequences?"
        A: Each Mamba block maintains a fixed-size hidden state
           (d_inner × d_state) that compresses the entire history.
           Unlike attention which stores all key-value pairs,
           Mamba's state is constant size regardless of sequence length.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 256,
        n_layers: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # Token embedding
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02

        # Mamba blocks
        self.layers = [
            Mamba2Block(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ]

        # Final layer norm
        self.norm_weight = np.ones(d_model)
        self.norm_bias = np.zeros(d_model)

        # Language model head (weight tying with embedding)
        self.lm_head_weight = self.embedding.T  # Tied weights

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            token_ids: [batch × seq_len] integer token IDs

        Returns:
            logits: [batch × seq_len × vocab_size]
        """
        # Token embedding
        x = self.embedding[token_ids]  # [batch × seq × d_model]

        # Process through Mamba blocks
        for layer in self.layers:
            x = layer.forward(x)

        # Final norm
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x = (x - mean) / np.sqrt(var + 1e-5)
        x = x * self.norm_weight + self.norm_bias

        # LM head
        logits = x @ self.lm_head_weight  # [batch × seq × vocab]

        return logits

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> np.ndarray:
        """
        Autoregressive text generation.

        Unlike transformers which recompute attention for all tokens,
        Mamba can maintain a recurrent state and only process the
        new token at each step. This is much more efficient.

        Args:
            prompt_ids: [1 × prompt_len] starting tokens
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling

        Returns:
            generated: [1 × total_len] all tokens
        """
        generated = prompt_ids.copy()

        for _ in range(max_new_tokens):
            # Forward pass (in production, only need last token's output)
            logits = self.forward(generated)

            # Get logits for last position
            next_logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_logits < np.partition(
                    next_logits, -top_k
                )[:, -top_k:]
                next_logits[indices_to_remove] = -float('inf')

            # Softmax
            probs = np.exp(next_logits - np.max(next_logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)

            # Sample
            next_token = np.array([np.random.choice(self.vocab_size, p=probs[0])])
            generated = np.concatenate([generated, next_token.reshape(1, -1)], axis=1)

        return generated


################################################################################
# SECTION 4: TESTING AND DEMONSTRATION
################################################################################

def demonstrate_mamba2():
    """Comprehensive Mamba-2 demonstration."""
    print("=" * 70)
    print("MAMBA-2 DEMONSTRATION — Selective State Space Model")
    print("=" * 70)

    # === Demo 1: Selective Scan ===
    print("\n--- Demo 1: Selective Scan ---")
    batch, seq_len, d_inner, d_state = 2, 16, 64, 16

    x = np.random.randn(batch, seq_len, d_inner) * 0.1
    A = np.random.randn(d_inner, d_state) * 0.1
    B = np.random.randn(batch, seq_len, d_state) * 0.1
    C = np.random.randn(batch, seq_len, d_state) * 0.1
    dt = np.abs(np.random.randn(batch, seq_len, d_inner) * 0.01) + 0.01

    y = selective_scan(x, A, B, C, dt)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output stats: mean={y.mean():.4f}, std={y.std():.4f}")

    # === Demo 2: Mamba-2 Block ===
    print("\n--- Demo 2: Mamba-2 Block ---")
    block = Mamba2Block(d_model=64, d_state=16, d_conv=4, expand=2)
    x = np.random.randn(2, 32, 64) * 0.1
    output = block.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # === Demo 3: Full Model ===
    print("\n--- Demo 3: Mamba-2 Model ---")
    model = Mamba2Model(
        vocab_size=1000,
        d_model=64,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
    )

    token_ids = np.random.randint(0, 1000, (1, 8))
    logits = model.forward(token_ids)
    print(f"Input token IDs shape: {token_ids.shape}")
    print(f"Output logits shape: {logits.shape}")

    # === Demo 4: Generation ===
    print("\n--- Demo 4: Text Generation ---")
    prompt = np.array([[1, 2, 3, 4, 5]])
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"Prompt: {prompt[0].tolist()}")
    print(f"Generated: {generated[0].tolist()}")

    # === Demo 5: Complexity Comparison ===
    print("\n--- Demo 5: Complexity Comparison ---")
    seq_lengths = [128, 512, 2048, 8192]
    for seq_len in seq_lengths:
        attn_flops = seq_len ** 2 * 64  # O(n²)
        ssm_flops = seq_len * 64 * 16   # O(n)
        ratio = attn_flops / ssm_flops
        print(f"  seq_len={seq_len}: attention={attn_flops:,} FLOPs, "
              f"SSM={ssm_flops:,} FLOPs, ratio={ratio:.1f}x")

    print("\n" + "=" * 70)
    print("All Mamba-2 demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_mamba2()


################################################################################
# REFERENCES
################################################################################

# [1] Gu, A., Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with
#     Selective State Spaces. arXiv:2312.00752.
#
# [2] Dao, T., Gu, A. (2024). Transformers are SSMs: Generalized Models and
#     Efficient Algorithms Through Structured State Space Duality.
#     arXiv:2405.21060.
#
# [3] Gu, A., Goel, K., Ré, C. (2022). Efficiently Modeling Long Sequences
#     with Structured State Spaces. arXiv:2111.00396.

################################################################################
