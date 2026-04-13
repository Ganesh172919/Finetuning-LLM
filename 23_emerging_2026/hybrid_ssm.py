"""
################################################################################
HYBRID SSM-ATTENTION — COMBINING BEST OF BOTH WORLDS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Hybrid SSM-Attention?
    Combining state space models (SSM) with attention mechanisms.
    SSM for efficiency, attention for expressiveness.

Why does it matter?
    - SSM: O(n) complexity, efficient for long sequences
    - Attention: O(n²) but more expressive
    - Hybrid: balance efficiency and quality

Historical Evolution:
    - 2024: Jamba (AI21), Zamba (Zyphra)
    - 2025: Hybrid models in production
    - 2026: Standard architecture for long context

Interview Questions:
    1. "What is hybrid SSM-attention?"
        Combines SSM layers (efficient) with attention layers (expressive).
        Alternates between the two for best of both worlds.

    2. "When should I use hybrid models?"
        When you need long context (100K+) with good quality.

################################################################################
"""

import numpy as np
from typing import Optional
import math

import sys
import importlib

# Dynamic import for numbered directories (Python doesn't allow numeric module names)
def _import_from_numbered_dir(module_path: str):
    """Import from directories with numeric prefixes."""
    parts = module_path.split('.')
    if len(parts) >= 2 and parts[-2][0].isdigit():
        # Handle numeric directory imports
        dir_name = parts[-2]
        module_name = parts[-1]
        full_path = f"{dir_name}.{module_name}"
        return importlib.import_module(full_path)
    return importlib.import_module(module_path)

# Import with fallback for standalone usage
try:
    mamba_module = _import_from_numbered_dir('22_transformer_variants.mamba')
    MambaBlock = mamba_module.MambaBlock
except (ImportError, ModuleNotFoundError):
    # Fallback: create a minimal MambaBlock for standalone testing
    class MambaBlock:
        def __init__(self, d_model): self.d_model = d_model
        def forward(self, x): return x  # Identity for testing

try:
    attention_module = _import_from_numbered_dir('02_transformers.attention')
    MultiHeadAttention = attention_module.MultiHeadAttention
except (ImportError, ModuleNotFoundError):
    class MultiHeadAttention:
        def __init__(self, d_model, n_heads): self.d_model = d_model; self.n_heads = n_heads
        def forward(self, x): return x, None  # Identity for testing

try:
    layers_module = _import_from_numbered_dir('02_transformers.layers')
    TransformerBlock = layers_module.TransformerBlock
    RMSNorm = layers_module.RMSNorm
except (ImportError, ModuleNotFoundError):
    class TransformerBlock:
        def __init__(self, d_model, n_heads): pass
        def forward(self, x): return x
    class RMSNorm:
        def __init__(self, d_model): pass
        def forward(self, x): return x

################################################################################
# SECTION 1: HYBRID BLOCK
################################################################################

class HybridBlock:
    """
    Hybrid SSM-Attention Block
    ==========================

    Alternates between SSM and attention layers.

    Architecture:
        x → SSM Layer → Attention Layer → output

    Benefits:
    - SSM handles local patterns efficiently
    - Attention handles global dependencies
    - Best of both worlds
    """

    def __init__(self, d_model: int, n_heads: int = 8, use_attention: bool = True):
        self.d_model = d_model
        self.use_attention = use_attention

        # SSM component
        self.ssm = MambaBlock(d_model)

        # Attention component (optional)
        if use_attention:
            self.attention = MultiHeadAttention(d_model, n_heads)
            self.norm = RMSNorm(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: [batch × seq × d_model]

        Returns:
            output: [batch × seq × d_model]
        """
        # SSM layer
        h = self.ssm.forward(x)

        # Attention layer (optional)
        if self.use_attention:
            h_norm = self.norm.forward(h)
            attn_out, _ = self.attention.forward(h_norm)
            h = h + attn_out

        return h


################################################################################
# SECTION 2: HYBRID MODEL
################################################################################

class HybridSSM:
    """
    Hybrid SSM-Attention Model
    ===========================

    Complete model alternating SSM and attention layers.

    Interview Question:
        "How does a hybrid SSM model work?"
        Alternates between SSM layers for efficiency and
        attention layers for expressiveness. Some layers
        use SSM, others use attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 8,
        attention_every_n: int = 4
    ):
        self.layers = []
        for i in range(n_layers):
            use_attention = (i % attention_every_n == 0)
            self.layers.append(HybridBlock(d_model, n_heads, use_attention))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through hybrid model."""
        for layer in self.layers:
            x = layer.forward(x)
        return x


################################################################################
# SECTION 3: MAMBA-2 (STRUCTURED STATE SPACE DUALITY)
################################################################################

class Mamba2Block:
    """
    Mamba-2 — Structured State Space Duality (2024)
    ================================================

    Mamba-2 reveals a deep connection between SSMs and attention:
    they are not competing paradigms but can be unified through
    Structured State Space Duality (SSD).

    Key Innovation:
        - Mamba-1: Selection mechanism (time-varying SSM)
        - Mamba-2: Structured matrix framework (2-8x faster)

    How SSD works:
        1. SSMs can be expressed as structured matrix operations
        2. Attention is also a structured matrix operation
        3. Both are special cases of a general framework
        4. This enables hardware-optimized implementations

    Performance:
        - 2-8x faster training than Mamba-1
        - Same modeling quality
        - Bridges gap between SSMs and attention

    Interview Question:
        "What is Mamba-2's key insight?"
        Mamba-2 shows that SSMs and attention are not competing paradigms
        but can be unified through Structured State Space Duality (SSD).
        SSMs can be expressed as structured matrix operations, just like
        attention. This enables 2-8x faster training.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
        """
        self.d_model = d_model
        self.d_state = d_state

        # Structured state space parameters
        self.A = np.random.randn(d_state, d_state) * 0.01
        self.B = np.random.randn(d_state, d_model) * 0.01
        self.C = np.random.randn(d_model, d_state) * 0.01
        self.D = np.random.randn(d_model) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Mamba-2 forward pass using structured matrix operations.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seq_len, d = x.shape

        # Initialize state
        h = np.zeros((batch, self.d_state))

        outputs = []
        for t in range(seq_len):
            # State update: h_t = A @ h_{t-1} + B @ x_t
            h = h @ self.A.T + x[:, t, :] @ self.B.T

            # Output: y_t = C @ h_t + D * x_t
            y = h @ self.C.T + x[:, t, :] * self.D

            outputs.append(y)

        return np.stack(outputs, axis=1)


class MoEMamba:
    """
    MoE-Mamba — Mixture of Experts + Mamba
    ========================================

    Integrates Mamba with Mixture of Experts (MoE) layers.
    Achieves comparable performance with 2.2x fewer training steps.

    Architecture:
        x → Mamba Layer → MoE Layer → output

    Benefits:
        - Mamba: Efficient sequence modeling
        - MoE: Sparse scaling for capacity
        - Combined: Better efficiency + quality

    Interview Question:
        "What is MoE-Mamba?"
        Combines Mamba (SSM) with Mixture of Experts layers.
        Achieves comparable performance to pure Mamba with
        2.2x fewer training steps by adding sparse capacity.
    """

    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2):
        """
        Args:
            d_model: Model dimension
            n_experts: Number of experts
            top_k: Experts to activate per token
        """
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        # Mamba layer
        self.mamba = Mamba2Block(d_model)

        # MoE components
        self.experts = [
            {'W1': np.random.randn(d_model, d_model * 4) * 0.02,
             'W2': np.random.randn(d_model * 4, d_model) * 0.02}
            for _ in range(n_experts)
        ]
        self.router = np.random.randn(d_model, n_experts) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        MoE-Mamba forward pass.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
        """
        # Mamba layer
        h = self.mamba.forward(x)

        # MoE layer
        scores = h @ self.router
        indices = np.argsort(scores, axis=-1)[:, :, -self.top_k:]
        weights = self._softmax(np.take_along_axis(scores, indices, axis=-1))

        output = np.zeros_like(h)
        for i in range(self.top_k):
            expert = self.experts[i]
            hidden = h @ expert['W1']
            hidden = hidden * (1 / (1 + np.exp(-hidden)))  # SiLU
            expert_out = hidden @ expert['W2']
            output += weights[:, :, i:i+1] * expert_out

        return output

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)


################################################################################
# SECTION 4: TESTING
################################################################################

def demonstrate_hybrid():
    """Demonstrate hybrid SSM-attention."""
    print("=" * 70)
    print("HYBRID SSM-ATTENTION DEMONSTRATION")
    print("=" * 70)

    model = HybridSSM(d_model=64, n_layers=8, n_heads=4, attention_every_n=4)
    x = np.random.randn(2, 50, 64)
    output = model.forward(x)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")

    # Count attention vs SSM layers
    attn_count = sum(1 for l in model.layers if l.use_attention)
    ssm_count = len(model.layers) - attn_count
    print(f"Attention layers: {attn_count}")
    print(f"SSM layers: {ssm_count}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_hybrid()


################################################################################
# REFERENCES
################################################################################

# [1] Lieber, O., et al. (2024). Jamba: A Hybrid Transformer-Mamba Language Model.

################################################################################
