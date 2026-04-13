"""
################################################################################
SMALL LANGUAGE MODEL ARCHITECTURES — PHI-4, GEMMA 3, QWEN3
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are SLM Architectures?
    Efficient transformer architectures optimized for 1B-7B parameter models.
    These models achieve competitive performance through architectural
    innovations: Grouped Query Attention, sliding window, efficient FFN.

Why do they matter?
    SLMs democratize AI deployment:
    - Run on mobile devices and laptops
    - 10-100x cheaper inference than 70B models
    - Privacy-preserving (local execution)
    - Sub-second latency

How do they work?
    1. Grouped Query Attention (GQA) — fewer KV heads than Q heads
    2. Sliding Window Attention — local attention for efficiency
    3. SwiGLU/GeGLU — better activation functions
    4. RoPE — rotary position embeddings
    5. RMSNorm — faster normalization

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ SLM Block (Phi-4/Gemma 3 style)                            │
    │                                                              │
    │  Input → RMSNorm → GQA → + Input (residual)                │
    │       → RMSNorm → SwiGLU FFN → + Residual                  │
    │       → Output                                              │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2023: Phi-1.5 (1.3B), Mistral-7B, Gemma-2B
    - 2024: Phi-3 (3.8B), Gemma 2, Qwen2
    - 2025: Phi-4-mini (3.8B), Gemma 3, Qwen3
    - 2026: Sub-1B models matching 2023 7B quality

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

import sys
sys.path.append('..')
from ..02_transformers.layers import TransformerBlock, RMSNorm
from ..02_transformers.embeddings import RoPE


################################################################################
# SECTION 1: CONFIGURATIONS
################################################################################

@dataclass
class Phi4MiniConfig:
    """
    Phi-4-mini Configuration (3.8B parameters).

    Key features: GQA (3:1 ratio), parallel attention+FFN, RoPE.
    """
    vocab_size: int = 51200
    d_model: int = 3072
    n_layers: int = 32
    n_heads: int = 24
    n_kv_heads: int = 8  # GQA: 3:1 ratio
    d_ff: int = 8192
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    dropout: float = 0.0


@dataclass
class Gemma3Config:
    """
    Gemma 3 Configuration (2B parameters).

    Key features: sliding window + global attention interleaved, GeGLU.
    """
    vocab_size: int = 256128
    d_model: int = 2048
    n_layers: int = 18
    n_heads: int = 8
    n_kv_heads: int = 4  # GQA 2:1
    d_ff: int = 16384
    sliding_window: int = 4096
    max_seq_len: int = 131072
    layer_types: str = "LSLSLSLSLSLSLSLSLS"  # L=local, S=sliding


################################################################################
# SECTION 2: GROUPED QUERY ATTENTION
################################################################################

class GroupedQueryAttention:
    """
    Grouped Query Attention (GQA) — fewer KV heads than Q heads.

    Instead of N heads for Q, K, V (MHA) or 1 head for K,V (MQA),
    use G groups for K,V. This balances quality and efficiency.

    Formula:
        Q = [h1, h2, ..., h_n]      (n query heads)
        K = [g1, g2, ..., g_g]      (g key heads, g < n)
        V = [g1, g2, ..., g_g]      (g value heads)
        Each KV head is shared by n/g query heads

    Example: n=24, g=8 → each KV head shared by 3 Q heads

    Interview Question:
        "What is Grouped Query Attention?"
        GQA uses fewer KV heads than Q heads. With n=24 Q heads and
        g=8 KV heads, each KV head is shared by 3 Q heads. This reduces
        KV cache size by 3x (faster inference) while maintaining quality
        close to full multi-head attention.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # Repetition factor

        # Weight matrices
        scale = 1.0 / math.sqrt(self.head_dim)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, n_kv_heads * self.head_dim) * scale
        self.W_v = np.random.randn(d_model, n_kv_heads * self.head_dim) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def repeat_kv(self, x: np.ndarray) -> np.ndarray:
        """
        Repeat KV heads to match Q heads.

        Args:
            x: (batch, seq_len, n_kv_heads, head_dim)
        Returns:
            (batch, seq_len, n_heads, head_dim)
        """
        if self.n_rep == 1:
            return x
        # Repeat each KV head n_rep times
        return np.repeat(x, self.n_rep, axis=2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        GQA forward pass.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = x @ self.W_q  # (batch, seq, d_model)
        k = x @ self.W_k  # (batch, seq, n_kv_heads * head_dim)
        v = x @ self.W_v

        # Reshape to heads
        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Repeat K,V to match Q heads
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        # Scaled dot-product attention
        scores = np.einsum('bshd,bthd->bhst', q, k) / math.sqrt(self.head_dim)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)

        out = np.einsum('bhst,bthd->bshd', weights, v)
        out = out.reshape(batch, seq_len, self.d_model)
        return out @ self.W_o


################################################################################
# SECTION 3: PHI-4 MINI
################################################################################

class Phi4Mini:
    """
    Phi-4-mini Architecture (3.8B parameters).

    Key innovations:
    - Grouped Query Attention (24 Q heads, 8 KV heads)
    - Parallel attention + FFN (like GPT-J)
    - RoPE with theta=10000
    - Long context (4K with RoPE scaling)

    Paper: "Phi-4 Technical Report" (Microsoft, 2025)

    Interview Question:
        "What makes Phi-4-mini efficient?"
        (1) GQA reduces KV cache by 3x, (2) parallel attention+FFN
        reduces latency, (3) high-quality training data compensates
        for smaller size, (4) RoPE enables long context.
    """

    def __init__(self, config: Optional[Phi4MiniConfig] = None):
        self.config = config or Phi4MiniConfig()
        self.attention = GroupedQueryAttention(
            self.config.d_model, self.config.n_heads, self.config.n_kv_heads
        )
        self.norm = RMSNorm(self.config.d_model)
        # FFN weights
        d = self.config.d_model
        self.W_gate = np.random.randn(d, self.config.d_ff) * 0.02
        self.W_up = np.random.randn(d, self.config.d_ff) * 0.02
        self.W_down = np.random.randn(self.config.d_ff, d) * 0.02

    def swiglu(self, x: np.ndarray) -> np.ndarray:
        """SwiGLU activation: SiLU(x @ W_gate) * (x @ W_up)."""
        gate = x @ self.W_gate
        silu = gate * (1.0 / (1.0 + np.exp(-gate)))  # SiLU
        up = x @ self.W_up
        return (silu * up) @ self.W_down

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with parallel attention + FFN.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        normed = self.norm(x)
        # Parallel attention + FFN (GPT-J style)
        attn_out = self.attention.forward(normed)
        ffn_out = self.swiglu(normed)
        return x + attn_out + ffn_out  # Double residual

    @classmethod
    def from_preset(cls) -> 'Phi4Mini':
        """Create Phi-4-mini with default config."""
        return cls(Phi4MiniConfig())


################################################################################
# SECTION 4: GEMMA 3 SMALL
################################################################################

class Gemma3Small:
    """
    Gemma 3 Architecture (2B parameters).

    Key innovations:
    - Sliding window + global attention interleaved
    - GeGLU activation
    - Logit soft-capping
    - Multi-query attention variant

    Paper: "Gemma 3 Technical Report" (Google, 2025)

    Interview Question:
        "How does Gemma 3 handle long context efficiently?"
        Gemma 3 interleaves local (sliding window) and global attention
        layers. Local layers attend to W=4096 nearby tokens (O(N*W)).
        Global layers attend to all tokens but use GQA. This gives
        effective long-range attention at lower cost than full MHA.
    """

    def __init__(self, config: Optional[Gemma3Config] = None):
        self.config = config or Gemma3Config()
        self.layers = []
        for i in range(self.config.n_layers):
            is_global = self.config.layer_types[i] == 'L' if i < len(self.config.layer_types) else True
            self.layers.append({
                'type': 'global' if is_global else 'sliding',
                'window': self.config.sliding_window if not is_global else None
            })

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with interleaved attention types."""
        for i, layer in enumerate(self.layers):
            if layer['type'] == 'sliding':
                # Sliding window attention (local)
                pass  # Simplified
            else:
                # Global attention with GQA
                pass  # Simplified
        return x

    @classmethod
    def from_preset(cls) -> 'Gemma3Small':
        """Create Gemma 3 2B with default config."""
        return cls(Gemma3Config())


################################################################################
# SECTION 5: SLM TRAINING
################################################################################

class SLMTraining:
    """
    Training utilities for Small Language Models.

    Key techniques:
    - Knowledge distillation from larger models
    - Data curriculum (easy → hard)
    - Learning rate schedule (warmup + cosine decay)

    Interview Question:
        "How do you train an SLM effectively?"
        (1) High-quality data > more data (Phi approach),
        (2) Knowledge distillation from teacher model,
        (3) Curriculum learning: easy examples first,
        (4) Cosine LR schedule with warmup.
    """

    def __init__(self, d_model: int = 2048, lr: float = 3e-4,
                 warmup_steps: int = 1000, max_steps: int = 100000):
        self.d_model = d_model
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step: int) -> float:
        """
        Cosine learning rate schedule with warmup.

        Args:
            step: Current training step

        Returns:
            Learning rate for this step
        """
        if step < self.warmup_steps:
            return self.lr * step / self.warmup_steps
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


################################################################################
# SECTION 6: DEMONSTRATION
################################################################################

def demonstrate_slm_architectures():
    """Demonstrate SLM architectures."""
    print("=" * 70)
    print("SLM ARCHITECTURES DEMONSTRATION")
    print("=" * 70)

    # Phi-4-mini
    print("\n1. PHI-4 MINI (3.8B)")
    print("-" * 40)
    phi = Phi4Mini.from_preset()
    x = np.random.randn(1, 10, phi.config.d_model)
    out = phi.forward(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Heads: {phi.config.n_heads} Q, {phi.config.n_kv_heads} KV")
    print(f"  GQA ratio: {phi.config.n_heads // phi.config.n_kv_heads}:1")

    # Gemma 3
    print("\n2. GEMMA 3 (2B)")
    print("-" * 40)
    gemma = Gemma3Small.from_preset()
    print(f"  Layers: {gemma.config.n_layers}")
    print(f"  Sliding window: {gemma.config.sliding_window}")
    print(f"  Layer types: {gemma.config.layer_types[:10]}...")
    print(f"  Max context: {gemma.config.max_seq_len}")

    # Learning rate schedule
    print("\n3. LEARNING RATE SCHEDULE")
    print("-" * 40)
    trainer = SLMTraining()
    for step in [0, 500, 1000, 50000, 100000]:
        lr = trainer.get_lr(step)
        print(f"  Step {step:6d}: LR = {lr:.6f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_slm_architectures()
