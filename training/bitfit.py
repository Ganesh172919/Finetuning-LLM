"""
BitFit: Bias-Only Fine-Tuning
===============================

BitFit freezes all model parameters except bias terms.
This is the most extreme parameter-efficient method:
only ~0.1% of parameters are trainable.

Key Insight:
  Bias terms capture task-specific adjustments to activation distributions.
  Fine-tuning only biases is surprisingly effective for many tasks.

What BitFit trains:
  - Attention bias terms (Q, K, V, O projections)
  - FFN bias terms
  - LayerNorm bias terms

What BitFit freezes:
  - All weight matrices
  - Embeddings
  - LayerNorm scale parameters

Benefits:
  - ~0.1% trainable parameters
  - Very fast training
  - Minimal storage per task
  - Good for domain adaptation

References:
  - Zaken et al., "BitFit: Simple Parameter-efficient Fine-tuning
    for Transformer-based Masked Language-models" (2022)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BitFitConfig:
    """
    Configuration for BitFit.

    Attributes:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        train_layernorm: Whether to train LayerNorm bias
        train_embedding: Whether to train embedding bias
    """
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    train_layernorm: bool = True
    train_embedding: bool = False


# ============================================================================
# BIAS-ONLY ADAPTER
# ============================================================================

class BitFitAdapter:
    """
    BitFit: Fine-tune only bias terms.

    In a Transformer, each linear layer has:
      output = input @ weight + bias

    BitFit freezes the weight and only trains the bias.
    This is equivalent to learning a constant shift for each neuron.

    For a d_model → d_model linear layer:
    - Full fine-tuning: d_model × d_model parameters
    - BitFit: d_model parameters (just the bias)

    Ratio: d_model / (d_model × d_model) = 1/d_model ≈ 0.13% for d=768
    """

    def __init__(self, config: BitFitConfig):
        """
        Initialize BitFit adapter.

        Args:
            config: BitFit configuration
        """
        self.config = config
        d_model = config.d_model
        d_ff = config.d_ff

        # Attention bias terms
        self.q_bias = np.zeros(d_model)
        self.k_bias = np.zeros(d_model)
        self.v_bias = np.zeros(d_model)
        self.o_bias = np.zeros(d_model)

        # FFN bias terms
        self.ff1_bias = np.zeros(d_ff)
        self.ff2_bias = np.zeros(d_model)

        # LayerNorm bias terms
        if config.train_layernorm:
            self.ln1_bias = np.zeros(d_model)
            self.ln2_bias = np.zeros(d_model)

    def apply_attention_bias(self, Q: np.ndarray, K: np.ndarray,
                             V: np.ndarray, O: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply bias terms to attention projections.

        Args:
            Q, K, V, O: Attention tensors [batch, seq_len, d_model]

        Returns:
            Biased Q, K, V, O
        """
        return (
            Q + self.q_bias,
            K + self.k_bias,
            V + self.v_bias,
            O + self.o_bias,
        )

    def apply_ffn_bias(self, ff1: np.ndarray, ff2: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply bias terms to FFN.

        Args:
            ff1: FFN intermediate [batch, seq_len, d_ff]
            ff2: FFN output [batch, seq_len, d_model]

        Returns:
            Biased ff1, ff2
        """
        return ff1 + self.ff1_bias, ff2 + self.ff2_bias

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        count = (
            self.q_bias.size +
            self.k_bias.size +
            self.v_bias.size +
            self.o_bias.size +
            self.ff1_bias.size +
            self.ff2_bias.size
        )
        if self.config.train_layernorm:
            count += self.ln1_bias.size + self.ln2_bias.size
        return count

    def get_trainable_ratio(self, model_params: int) -> float:
        """Get ratio of trainable to total parameters."""
        return self.count_parameters() / model_params


# ============================================================================
# BIAS ANALYSIS
# ============================================================================

def analyze_bias_impact():
    """
    Analyze why bias terms are so effective.

    Key findings from the BitFit paper:
    1. Attention biases capture task-specific attention patterns
    2. FFN biases adjust activation thresholds
    3. LayerNorm biases shift activation centers
    4. Most effective for smaller models and simpler tasks
    5. Less effective for tasks requiring new knowledge
    """
    return """
    ┌─────────────────────────────────────────────────────────────┐
    │                    Bias Impact Analysis                     │
    ├─────────────────────────────────────────────────────────────┤
    │ Q, K biases: Adjust attention scoring offsets               │
    │ V bias: Shift attention output distributions                │
    │ O bias: Adjust multi-head combination                      │
    │ FF1 bias: Shift activation thresholds (ReLU/GELU)          │
    │ FF2 bias: Adjust final output distribution                 │
    │ LN bias: Shift normalization centers                       │
    └─────────────────────────────────────────────────────────────┘

    Why it works:
    - Pre-trained weights capture general language knowledge
    - Biases capture task-specific "adjustments"
    - Like adding a constant offset to each neuron's activation
    - Small but effective for many downstream tasks
    """


# ============================================================================
# COMPARISON
# ============================================================================

def compare_bitfit_with_others():
    """Compare BitFit with other PEFT methods."""
    return """
    ┌──────────────────┬───────────┬───────────┬────────────┬──────────────┐
    │ Method           │ Params    │ Quality   │ Storage    │ Best For     │
    ├──────────────────┼───────────┼───────────┼────────────┼──────────────┤
    │ Full FT          │ 100%      │ Best      │ Large      │ Max quality  │
    │ LoRA             │ 0.1-1%    │ ~99%      │ Small      │ General      │
    │ QLoRA            │ 0.1-1%    │ ~98%      │ Tiny       │ Low memory   │
    │ Prefix Tuning    │ 0.01-0.1% │ ~95%      │ Tiny       │ Multi-task   │
    │ IA3              │ <0.01%    │ ~94%      │ Tiny       │ Few-shot     │
    │ BitFit           │ ~0.1%     │ ~93%      │ Tiny       │ Simple tasks │
    └──────────────────┴───────────┴───────────┴────────────┴──────────────┘
    """


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_bitfit():
    """
    Demonstrate BitFit.

    Shows:
        1. Parameter counting
        2. Bias operations
        3. Comparison with other methods
    """
    print("=" * 70)
    print("BitFit: Bias-Only Fine-Tuning — Demonstration")
    print("=" * 70)

    # Configuration
    config = BitFitConfig(d_model=768, d_ff=3072, n_heads=12)

    # Create adapter
    adapter = BitFitAdapter(config)

    print(f"\nConfiguration:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  Train LayerNorm: {config.train_layernorm}")

    # Parameter count
    params = adapter.count_parameters()
    model_params = 125_000_000  # Approximate

    print(f"\nParameter Count:")
    print(f"  BitFit parameters: {params:,}")
    print(f"  Model parameters: {model_params:,}")
    print(f"  Trainable ratio: {adapter.get_trainable_ratio(model_params):.4%}")

    # Bias operations
    print("\n[Bias Operations]")
    batch_size, seq_len = 2, 10
    Q = np.random.randn(batch_size, seq_len, config.d_model)
    K = np.random.randn(batch_size, seq_len, config.d_model)
    V = np.random.randn(batch_size, seq_len, config.d_model)
    O = np.random.randn(batch_size, seq_len, config.d_model)

    Q_b, K_b, V_b, O_b = adapter.apply_attention_bias(Q, K, V, O)
    print(f"  Q shape: {Q.shape} → Q+bias shape: {Q_b.shape}")
    print(f"  Bias effect: mean shift = {np.mean(Q_b - Q):.4f}")

    # Analysis
    print("\n[Bias Impact Analysis]")
    print(analyze_bias_impact())

    # Comparison
    print("\n[PEFT Methods Comparison]")
    print(compare_bitfit_with_others())

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. BitFit trains only ~0.1% of parameters (bias terms)")
    print("  2. Surprisingly effective for many NLP tasks")
    print("  3. Like adding a constant offset to each neuron")
    print("  4. Best for simple tasks and domain adaptation")
    print("  5. Can store hundreds of tasks as separate biases")
    print("=" * 70)


if __name__ == "__main__":
    demo_bitfit()
