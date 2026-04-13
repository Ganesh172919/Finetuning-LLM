"""
IA3: Infused Adapter by Inhibiting and Amplifying Inner Activations
====================================================================

IA3 is an extremely parameter-efficient adapter method that learns
rescaling vectors for the model's activations.

Key Insight:
  Instead of adding new parameters (like LoRA or Prefix Tuning),
  IA3 learns to rescale existing activations:
    h_new = l_k ⊙ K, l_v ⊙ V, l_ff ⊙ FF(x)

  Where l_k, l_v, l_ff are learnable rescaling vectors.

Benefits:
  - Extremely few parameters (typically < 0.01% of model)
  - No additional inference latency (just element-wise multiply)
  - Works well with few-shot learning
  - Can be composed with other adapters

References:
  - Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better
    and Cheaper than In-Context Learning" (2022)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IA3Config:
    """
    Configuration for IA3 adapter.

    Attributes:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        n_heads: Number of attention heads
        head_size: Dimension per head
    """
    d_model: int = 768
    d_ff: int = 3072
    n_heads: int = 12
    head_size: int = 64


# ============================================================================
# IA3 ADAPTER
# ============================================================================

class IA3Adapter:
    """
    IA3: Infused Adapter by Inhibiting and Amplifying.

    Instead of adding new layers, IA3 learns to rescale:
    1. Key vectors in attention: K_new = l_k ⊙ K
    2. Value vectors in attention: V_new = l_v ⊙ V
    3. FFN activations: FF_new = l_ff ⊙ FF(x)

    This is the most parameter-efficient PEFT method:
    - LoRA: 0.1-1% of model parameters
    - Prefix Tuning: 0.01-0.1%
    - IA3: <0.01%

    The rescaling vectors are initialized to ones (no effect at init).
    """

    def __init__(self, config: IA3Config):
        """
        Initialize IA3 adapter.

        Args:
            config: IA3 configuration
        """
        self.config = config

        # Rescaling vectors (initialized to 1.0 = no effect)
        self.l_k = np.ones(config.d_model)  # For key vectors
        self.l_v = np.ones(config.d_model)  # For value vectors
        self.l_ff = np.ones(config.d_ff)    # For FFN activations

    def rescale_key(self, K: np.ndarray) -> np.ndarray:
        """
        Rescale key vectors.

        Args:
            K: Key tensor [batch, seq_len, d_model]

        Returns:
            Rescaled keys
        """
        return K * self.l_k

    def rescale_value(self, V: np.ndarray) -> np.ndarray:
        """
        Rescale value vectors.

        Args:
            V: Value tensor [batch, seq_len, d_model]

        Returns:
            Rescaled values
        """
        return V * self.l_v

    def rescale_ffn(self, ffn_output: np.ndarray) -> np.ndarray:
        """
        Rescale FFN activations.

        Args:
            ffn_output: FFN output [batch, seq_len, d_ff]

        Returns:
            Rescaled FFN output
        """
        return ffn_output * self.l_ff

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return self.l_k.size + self.l_v.size + self.l_ff.size


# ============================================================================
# IA3 TRAINING
# ============================================================================

class IA3Trainer:
    """
    Training loop for IA3.

    Only the rescaling vectors are trained. The base model is frozen.
    """

    def __init__(self, adapter: IA3Adapter, learning_rate: float = 1e-3):
        self.adapter = adapter
        self.lr = learning_rate
        self.step_count = 0

    def train_step(self, loss_fn, model_fn, batch) -> float:
        """
        Single training step.

        Args:
            loss_fn: Loss function
            model_fn: Model forward function
            batch: Training batch

        Returns:
            Loss value
        """
        # Forward pass with IA3 rescaling
        output = model_fn(batch, self.adapter)
        loss = loss_fn(output, batch["labels"])

        # Gradient update (simplified)
        # In practice, use automatic differentiation
        self.step_count += 1

        return loss


# ============================================================================
# COMPARISON
# ============================================================================

def compare_ia3_with_others():
    """Compare IA3 with other PEFT methods."""
    return """
    ┌──────────────────┬───────────┬───────────┬────────────┬──────────────┐
    │ Method           │ Params    │ Latency   │ Quality    │ Best For     │
    ├──────────────────┼───────────┼───────────┼────────────┼──────────────┤
    │ Full FT          │ 100%      │ 1x        │ Best       │ Max quality  │
    │ LoRA             │ 0.1-1%    │ 1x        │ ~99%       │ General      │
    │ QLoRA            │ 0.1-1%    │ 1x        │ ~98%       │ Low memory   │
    │ Prefix Tuning    │ 0.01-0.1% │ 1x        │ ~95%       │ Multi-task   │
    │ Prompt Tuning    │ 0.001%    │ 1x        │ ~93%       │ Huge models  │
    │ IA3              │ <0.01%    │ 1x        │ ~94%       │ Few-shot     │
    └──────────────────┴───────────┴───────────┴────────────┴──────────────┘
    """


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_ia3():
    """
    Demonstrate IA3 adapter.

    Shows:
        1. IA3 configuration
        2. Rescaling operations
        3. Parameter counting
        4. Comparison with other methods
    """
    print("=" * 70)
    print("IA3: Infused Adapter — Demonstration")
    print("=" * 70)

    # Configuration
    config = IA3Config(d_model=768, d_ff=3072, n_heads=12)

    # Create adapter
    adapter = IA3Adapter(config)

    print(f"\nConfiguration:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  Heads: {config.n_heads}")

    # Parameter count
    params = adapter.count_parameters()
    model_params = 125_000_000  # Approximate

    print(f"\nParameter Count:")
    print(f"  IA3 parameters: {params:,}")
    print(f"  Model parameters: {model_params:,}")
    print(f"  IA3 ratio: {params/model_params:.4%}")

    # Rescaling demo
    print("\n[Rescaling Demo]")
    K = np.random.randn(2, 10, config.d_model)
    V = np.random.randn(2, 10, config.d_model)
    FF = np.random.randn(2, 10, config.d_ff)

    K_scaled = adapter.rescale_key(K)
    V_scaled = adapter.rescale_value(V)
    FF_scaled = adapter.rescale_ffn(FF)

    print(f"  K shape: {K.shape} → {K_scaled.shape}")
    print(f"  V shape: {V.shape} → {V_scaled.shape}")
    print(f"  FF shape: {FF.shape} → {FF_scaled.shape}")

    # Comparison
    print("\n[PEFT Methods Comparison]")
    print(compare_ia3_with_others())

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. IA3 learns rescaling vectors, not new weights")
    print("  2. <0.01% parameters — most efficient PEFT method")
    print("  3. No additional inference latency")
    print("  4. Works well for few-shot learning")
    print("  5. Can be composed with LoRA or other adapters")
    print("=" * 70)


if __name__ == "__main__":
    demo_ia3()
