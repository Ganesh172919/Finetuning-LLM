"""
################################################################################
MAMBA — STATE SPACE MODELS AS TRANSFORMER ALTERNATIVES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Mamba?
    Mamba is a state space model (SSM) that achieves transformer-level
    performance with linear complexity. Unlike transformers which have
    O(n²) attention, Mamba processes sequences in O(n).

Why does it matter?
    Transformers are powerful but expensive:
    - Attention: O(n²) in sequence length
    - KV cache: grows with context
    - Long sequences: very expensive

    Mamba offers:
    - Linear complexity: O(n)
    - No KV cache needed
    - Long context: efficient
    - Competitive quality

How does it work?
    Mamba uses selective state spaces:
    1. Input sequence → state transitions
    2. States capture relevant information
    3. Output depends on current state

    Key innovation: Selective scan
    - Different inputs get different state transitions
    - This allows the model to "choose" what to remember

Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Input x                                          │
    │        ↓                                          │
    │ Linear projection → expanded                     │
    │        ↓                                          │
    │ Conv1D (local context)                           │
    │        ↓                                          │
    │ SiLU activation                                  │
    │        ↓                                          │
    │ Selective SSM (global context)                   │
    │        ↓                                          │
    │ Multiply with gate                               │
    │        ↓                                          │
    │ Linear projection → output                       │
    └─────────────────────────────────────────────────┘

Historical Evolution:
    - 2021: S4 (Structured State Spaces)
    - 2022: S4D, H3
    - 2023: Mamba (Gu & Dao)
    - 2024: Mamba-2, hybrid architectures
    - 2025: Mamba in production models

Interview Questions:
        1. "What is Mamba?"
           A state space model that processes sequences in O(n) time.
           It's an alternative to transformers for long sequences.

        2. "When should I use Mamba vs Transformer?"
           Mamba: long sequences, streaming, low latency
           Transformers: when attention patterns are complex
           Hybrid: best of both worlds

        3. "How does selective scan work?"
           Instead of fixed state transitions, the model learns
           input-dependent transitions. This allows selective
           memory: remember important inputs, forget others.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple
import math

################################################################################
# SECTION 1: STATE SPACE MODEL
################################################################################

class StateSpaceModel:
    """
    State Space Model (SSM)
    ========================

    Definition: A model that maps input sequences to output sequences
    through a latent state.

    State equations:
        h(t) = A * h(t-1) + B * x(t)    (state update)
        y(t) = C * h(t) + D * x(t)      (output)

    Where:
        h: hidden state
        x: input
        y: output
        A, B, C, D: learned matrices

    Key property: Linear time complexity O(n)
    For each input, we update state and compute output.

    Discretization:
        Ā = exp(Δ * A)
        B̄ = (Δ * A)^(-1) * (exp(Δ * A) - I) * Δ * B

    This converts continuous-time SSM to discrete-time.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        """
        Initialize SSM.

        Args:
            d_model: Model dimension
            d_state: State dimension (usually 16-64)
        """
        self.d_model = d_model
        self.d_state = d_state

        # SSM parameters
        # A: state transition matrix
        self.A = np.random.randn(d_model, d_state) * 0.01
        # B: input matrix
        self.B = np.random.randn(d_model, d_state) * 0.01
        # C: output matrix
        self.C = np.random.randn(d_model, d_state) * 0.01
        # D: skip connection
        self.D = np.ones(d_model)

        # Discretization parameter
        self.dt = np.ones(d_model) * 0.1

    def discretize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize continuous SSM to discrete.

        Returns:
            A_bar: Discretized A
            B_bar: Discretized B
        """
        # Simplified discretization
        A_bar = np.exp(np.expand_dims(self.dt, -1) * self.A)
        B_bar = np.expand_dims(self.dt, -1) * self.B
        return A_bar, B_bar

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Process sequence through SSM.

        Args:
            x: Input [batch × seq_len × d_model]

        Returns:
            output: [batch × seq_len × d_model]
        """
        batch, seq_len, d_model = x.shape
        A_bar, B_bar = self.discretize()

        # Initialize state
        h = np.zeros(batch, self.d_state)

        # Process sequence
        outputs = []
        for t in range(seq_len):
            # State update: h = A_bar * h + B_bar * x
            h = np.expand_dims(A_bar, 0) * np.expand_dims(h, 1)
            h = h + np.expand_dims(B_bar, 0) * np.expand_dims(x[:, t, :], -1)
            h = np.sum(h, axis=1)  # Sum over d_model

            # Output: y = C * h + D * x
            y = np.sum(np.expand_dims(self.C, 0) * np.expand_dims(h, 1), axis=-1)
            y = y + self.D * x[:, t, :]

            outputs.append(y)

        return np.stack(outputs, axis=1)


################################################################################
# SECTION 2: SELECTIVE SSM (MAMBA)
################################################################################

class SelectiveSSM:
    """
    Selective State Space Model (Mamba)
    =====================================

    Key innovation: input-dependent state transitions.

    Standard SSM: A, B, C are fixed
    Selective SSM: B, C, Δ depend on input

    This allows:
    - Selective memory: remember important inputs
    - Context-dependent processing
    - Better performance on complex tasks

    Interview Question:
        "What makes Mamba different from S4?"
        Mamba uses selective state spaces where B, C, and the
        step size Δ are input-dependent. This allows the model
        to selectively remember or forget information.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv

        # Input-dependent parameters
        self.B_proj = np.random.randn(d_model, d_state) * 0.01
        self.C_proj = np.random.randn(d_model, d_state) * 0.01
        self.dt_proj = np.random.randn(d_model) * 0.01

        # Fixed A parameter
        self.A = np.random.randn(d_model, d_state) * 0.01

        # Conv1D for local context
        self.conv_weight = np.random.randn(d_model, 1, d_conv) * 0.01

        # Output projection
        self.D = np.ones(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Selective scan.

        Args:
            x: Input [batch × seq_len × d_model]

        Returns:
            output: [batch × seq_len × d_model]
        """
        batch, seq_len, d_model = x.shape

        # Compute input-dependent parameters
        B = np.matmul(x, self.B_proj)  # [batch × seq × d_state]
        C = np.matmul(x, self.C_proj)  # [batch × seq × d_state]
        dt = np.abs(np.matmul(x, self.dt_proj))  # [batch × seq]

        # Initialize state
        h = np.zeros(batch, self.d_state)

        # Selective scan
        outputs = []
        for t in range(seq_len):
            # Discretize with input-dependent dt
            A_bar = np.exp(np.expand_dims(dt[:, t], -1) * self.A)
            B_bar = np.expand_dims(dt[:, t], -1) * B[:, t, :]

            # State update
            h = A_bar * h + B_bar * x[:, t, :]

            # Output
            y = np.sum(C[:, t, :] * h, axis=-1, keepdims=True)
            y = y + self.D * x[:, t, :]

            outputs.append(y)

        return np.stack(outputs, axis=1)


################################################################################
# SECTION 3: MAMBA BLOCK
################################################################################

class MambaBlock:
    """
    Mamba Block
    ===========

    Complete Mamba block with:
    1. Linear projection (expand)
    2. Conv1D (local context)
    3. Selective SSM (global context)
    4. Gate mechanism
    5. Output projection

    Architecture:
        x → proj → conv → silu → SSM ─┐
        x → proj → silu ─────────────── ⊙ → proj → output
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        self.d_model = d_model
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = np.random.randn(d_model, self.d_inner * 2) * 0.02

        # Conv1D
        self.conv1d = np.random.randn(self.d_inner, 1, 4) * 0.02

        # SSM
        self.ssm = SelectiveSSM(self.d_inner, d_state)

        # Output projection
        self.out_proj = np.random.randn(self.d_inner, d_model) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Mamba block forward pass.

        Args:
            x: [batch × seq_len × d_model]

        Returns:
            output: [batch × seq_len × d_model]
        """
        # Input projection (split for gate)
        xz = np.matmul(x, self.in_proj)
        x_proj, z = np.split(xz, 2, axis=-1)

        # Conv1D (simplified)
        x_proj = np.maximum(0, x_proj)  # SiLU

        # SSM
        y = self.ssm.forward(x_proj)

        # Gate
        y = y * (1 / (1 + np.exp(-z)))  # SiLU gate

        # Output projection
        output = np.matmul(y, self.out_proj)

        return output


################################################################################
# SECTION 4: TESTING & EXAMPLES
################################################################################

def demonstrate_mamba():
    """Demonstrate Mamba concepts."""
    print("=" * 70)
    print("MAMBA DEMONSTRATION")
    print("=" * 70)

    # State Space Model
    print("\n--- State Space Model ---")
    ssm = StateSpaceModel(d_model=64, d_state=16)
    x = np.random.randn(2, 10, 64)  # batch=2, seq=10, dim=64
    y = ssm.forward(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")

    # Selective SSM
    print("\n--- Selective SSM ---")
    selective_ssm = SelectiveSSM(d_model=64, d_state=16)
    y = selective_ssm.forward(x)
    print(f"Output: {y.shape}")

    # Mamba Block
    print("\n--- Mamba Block ---")
    mamba = MambaBlock(d_model=64, d_state=16)
    y = mamba.forward(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")

    # Complexity comparison
    print("\n--- Complexity Comparison ---")
    seq_len = 1000
    print(f"Transformer: O({seq_len}²) = {seq_len**2:,} operations")
    print(f"Mamba: O({seq_len}) = {seq_len:,} operations")
    print(f"Speedup: {seq_len**2 / seq_len:.0f}x")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_mamba()


################################################################################
# REFERENCES
################################################################################

# [1] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
# [2] Gu, A., et al. (2022). Efficiently Modeling Long Sequences with Structured State Spaces.
# [3] Gu, A., & Dao, T. (2024). Mamba-2: Efficient Sequence Modeling with Linear Attention.

################################################################################
