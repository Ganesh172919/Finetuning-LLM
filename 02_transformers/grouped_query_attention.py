"""
Grouped-Query Attention (GQA)
==============================

Grouped-Query Attention is a variant of multi-head attention that reduces
the KV cache size by sharing key and value heads across multiple query heads.

Motivation:
  Standard Multi-Head Attention (MHA) has n_heads Q, K, V projections.
  Multi-Query Attention (MQA) shares single K, V across all heads.
  GQA is the middle ground: groups of heads share K, V.

  ┌─────────────────────────────────────────────────────────────┐
  │ MHA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8                             │
  │       K1 K2 K3 K4 K5 K6 K7 K8   (8 KV heads)              │
  │       V1 V2 V3 V4 V5 V6 V7 V8                             │
  ├─────────────────────────────────────────────────────────────┤
  │ GQA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8                             │
  │       K1 K1 K1 K1 K2 K2 K2 K2   (2 KV heads, 4 groups)    │
  │       V1 V1 V1 V1 V2 V2 V2 V2                             │
  ├─────────────────────────────────────────────────────────────┤
  │ MQA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8                             │
  │       K1 K1 K1 K1 K1 K1 K1 K1   (1 KV head)               │
  │       V1 V1 V1 V1 V1 V1 V1 V1                             │
  └─────────────────────────────────────────────────────────────┘

Benefits:
  - KV cache: n_kv_heads × head_size × seq_len (vs n_heads for MHA)
  - Llama 2 70B: 8 KV heads instead of 64 → 8x smaller KV cache
  - Minimal quality loss compared to MHA
  - Better than MQA (which can lose quality)

Production Usage:
  - Llama 2 70B: GQA with 8 KV heads
  - Mistral 7B: GQA with 8 KV heads
  - Gemma 2: GQA with variable groups
  - Qwen 2: GQA

References:
  - Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
    Models from Multi-Head Checkpoints" (2023)
  - Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GQAConfig:
    """
    Configuration for Grouped-Query Attention.

    Attributes:
        d_model: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads (must divide n_heads)
        d_k: Dimension per head (computed as d_model // n_heads)
        dropout: Dropout probability
    """
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    dropout: float = 0.0

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

    @property
    def d_k(self) -> int:
        """Dimension per head."""
        return self.d_model // self.n_heads

    @property
    def n_groups(self) -> int:
        """Number of query heads per KV head."""
        return self.n_heads // self.n_kv_heads

    @property
    def kv_dim(self) -> int:
        """Total KV dimension."""
        return self.n_kv_heads * self.d_k


# ============================================================================
# GROUPED-QUERY ATTENTION
# ============================================================================

class GroupedQueryAttention:
    """
    Grouped-Query Attention implementation.

    Key insight: Instead of having n_heads independent K, V projections,
    we have n_kv_heads K, V projections that are shared across groups
    of query heads.

    Memory savings:
        MHA KV cache: 2 × n_heads × d_k × seq_len
        GQA KV cache: 2 × n_kv_heads × d_k × seq_len
        Savings: n_heads / n_kv_heads (e.g., 32/8 = 4x)

    Quality tradeoff:
        - MQA (n_kv_heads=1): Can lose quality, especially on complex tasks
        - GQA (n_kv_heads=8): Minimal quality loss, significant memory savings
        - MHA (n_kv_heads=n_heads): Full quality, full memory

    Training:
        Can convert MHA checkpoint to GQA by averaging K, V heads within groups.
        This is the "uptraining" approach from the GQA paper.
    """

    def __init__(self, config: GQAConfig):
        """
        Initialize GQA layer.

        Args:
            config: GQA configuration
        """
        self.config = config
        d_model = config.d_model
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        d_k = config.d_k

        # Query projection: d_model → n_heads × d_k
        # (same as MHA — every query head is independent)
        self.W_q = np.random.randn(d_model, n_heads * d_k) * 0.01

        # Key projection: d_model → n_kv_heads × d_k
        # (fewer heads than queries!)
        self.W_k = np.random.randn(d_model, n_kv_heads * d_k) * 0.01

        # Value projection: d_model → n_kv_heads × d_k
        # (fewer heads than queries!)
        self.W_v = np.random.randn(d_model, n_kv_heads * d_k) * 0.01

        # Output projection: n_heads × d_k → d_model
        self.W_o = np.random.randn(n_heads * d_k, d_model) * 0.01

    def _repeat_kv(self, x: np.ndarray) -> np.ndarray:
        """
        Repeat KV heads to match query heads.

        If n_heads=32, n_kv_heads=8, then each KV head is repeated 4 times.

        Args:
            x: KV tensor [batch, seq_len, n_kv_heads, d_k]

        Returns:
            Repeated KV tensor [batch, seq_len, n_heads, d_k]
        """
        batch_size, seq_len, n_kv_heads, d_k = x.shape
        n_groups = self.config.n_groups

        # Repeat each KV head n_groups times
        # [batch, seq_len, n_kv_heads, d_k] → [batch, seq_len, n_kv_heads, 1, d_k]
        x = x[:, :, :, np.newaxis, :]

        # → [batch, seq_len, n_kv_heads, n_groups, d_k]
        x = np.repeat(x, n_groups, axis=3)

        # → [batch, seq_len, n_heads, d_k]
        x = x.reshape(batch_size, seq_len, -1, d_k)

        return x

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of GQA.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional causal mask [seq_len, seq_len]

        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            weights: Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        config = self.config
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        d_k = config.d_k

        # ── Step 1: Project Q, K, V ────────────────────────────
        # Q: [batch, seq_len, n_heads × d_k]
        Q = x @ self.W_q

        # K: [batch, seq_len, n_kv_heads × d_k]  (fewer heads!)
        K = x @ self.W_k

        # V: [batch, seq_len, n_kv_heads × d_k]  (fewer heads!)
        V = x @ self.W_v

        # ── Step 2: Reshape to multi-head ───────────────────────
        Q = Q.reshape(batch_size, seq_len, n_heads, d_k)
        K = K.reshape(batch_size, seq_len, n_kv_heads, d_k)
        V = V.reshape(batch_size, seq_len, n_kv_heads, d_k)

        # ── Step 3: Repeat K, V to match Q heads ────────────────
        # This is the key GQA operation!
        K = self._repeat_kv(K)  # [batch, seq_len, n_heads, d_k]
        V = self._repeat_kv(V)  # [batch, seq_len, n_heads, d_k]

        # ── Step 4: Transpose for attention computation ──────────
        Q = Q.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_len, d_k]
        K = K.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_len, d_k]
        V = V.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_len, d_k]

        # ── Step 5: Scaled dot-product attention ─────────────────
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores - scores_max)
        weights = weights / np.sum(weights, axis=-1, keepdims=True)

        # Apply attention to values
        output = weights @ V  # [batch, n_heads, seq_len, d_k]

        # ── Step 6: Reshape and project output ──────────────────
        output = output.transpose(0, 2, 1, 3)  # [batch, seq_len, n_heads, d_k]
        output = output.reshape(batch_size, seq_len, -1)  # [batch, seq_len, n_heads*d_k]
        output = output @ self.W_o  # [batch, seq_len, d_model]

        return output, weights


# ============================================================================
# COMPARISON: MHA vs GQA vs MQA
# ============================================================================

def compare_attention_variants():
    """
    Compare MHA, GQA, and MQA variants.

    Returns comparison as formatted string.
    """
    comparison = """
    ┌──────────────────┬─────────────┬─────────────┬─────────────┐
    │ Property         │ MHA         │ GQA         │ MQA         │
    ├──────────────────┼─────────────┼─────────────┼─────────────┤
    │ Query heads      │ 32          │ 32          │ 32          │
    │ KV heads         │ 32          │ 8           │ 1           │
    │ KV cache size    │ 100%        │ 25%         │ 3.1%        │
    │ Quality          │ Best        │ Near-best   │ Good        │
    │ Inference speed  │ Baseline    │ 1.5-2x      │ 2-3x        │
    │ Used in          │ GPT-4, etc  │ Llama 2 70B │ PaLM, etc   │
    │ Training         │ Standard    │ Uptrain     │ From scratch│
    └──────────────────┴─────────────┴─────────────┴─────────────┘
    """
    return comparison


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_gqa():
    """
    Demonstrate Grouped-Query Attention.

    Shows:
        1. Configuration comparison
        2. Forward pass
        3. KV cache savings
        4. Quality vs efficiency tradeoff
    """
    print("=" * 70)
    print("Grouped-Query Attention (GQA) — Demonstration")
    print("=" * 70)

    # Configuration
    config = GQAConfig(d_model=256, n_heads=16, n_kv_heads=4)

    print(f"\nConfiguration:")
    print(f"  d_model: {config.d_model}")
    print(f"  Query heads: {config.n_heads}")
    print(f"  KV heads: {config.n_kv_heads}")
    print(f"  Groups: {config.n_groups}")
    print(f"  Head dim: {config.d_k}")
    print(f"  KV dim: {config.kv_dim}")

    # Create GQA layer
    gqa = GroupedQueryAttention(config)

    # Count parameters
    q_params = config.d_model * config.n_heads * config.d_k
    k_params = config.d_model * config.n_kv_heads * config.d_k
    v_params = config.d_model * config.n_kv_heads * config.d_k
    o_params = config.n_heads * config.d_k * config.d_model
    total_params = q_params + k_params + v_params + o_params

    # MHA parameters for comparison
    mha_params = 4 * config.d_model * config.n_heads * config.d_k

    print(f"\nParameter Count:")
    print(f"  Q projection: {q_params:,}")
    print(f"  K projection: {k_params:,}")
    print(f"  V projection: {v_params:,}")
    print(f"  O projection: {o_params:,}")
    print(f"  GQA total: {total_params:,}")
    print(f"  MHA total: {mha_params:,}")
    print(f"  Savings: {1 - total_params/mha_params:.1%}")

    # Forward pass
    print("\n[Forward Pass]")
    batch_size = 2
    seq_len = 16
    x = np.random.randn(batch_size, seq_len, config.d_model)

    output, weights = gqa.forward(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Weights shape: {weights.shape}")

    # KV cache comparison
    print("\n[KV Cache Comparison]")
    seq_len = 4096
    dtype_size = 2  # fp16

    mha_kv = 2 * config.n_heads * config.d_k * seq_len * dtype_size
    gqa_kv = 2 * config.n_kv_heads * config.d_k * seq_len * dtype_size

    print(f"  Sequence length: {seq_len}")
    print(f"  MHA KV cache: {mha_kv:,} bytes ({mha_kv/1e6:.1f} MB)")
    print(f"  GQA KV cache: {gqa_kv:,} bytes ({gqa_kv/1e6:.1f} MB)")
    print(f"  Reduction: {mha_kv/gqa_kv:.1f}x smaller")

    # Comparison table
    print("\n[Attention Variants Comparison]")
    print(compare_attention_variants())

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. GQA = MHA quality with MQA-like efficiency")
    print("  2. KV cache reduced by n_heads/n_kv_heads factor")
    print("  3. Can uptrain from MHA checkpoint (not from scratch)")
    print("  4. Standard in modern LLMs (Llama 2, Mistral, Gemma)")
    print("  5. Sweet spot: 8 KV heads for 32 query heads")
    print("=" * 70)


if __name__ == "__main__":
    demo_gqa()
