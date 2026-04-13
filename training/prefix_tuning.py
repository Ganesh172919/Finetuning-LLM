"""
Prefix Tuning: Learning Soft Prompts
======================================

Prefix Tuning prepends learnable "virtual tokens" to the input sequence.
These soft prompts steer the frozen model to perform specific tasks.

Key Insight:
  Instead of fine-tuning the model, we learn a continuous "prefix" that
  guides the model's behavior. The prefix exists in the model's embedding
  space but doesn't correspond to any real tokens.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                    Prefix Tuning                            │
  │                                                              │
  │  [P₁][P₂][P₃][P₄] [Token₁][Token₂][Token₃] ...          │
  │   ↑  ↑  ↑  ↑                                              │
  │   Learnable prefix (virtual tokens)                        │
  │   (not real words, just optimized vectors)                 │
  │                                                              │
  │  These prefix vectors are prepended to the Key and Value   │
  │  in every attention layer, steering the model's behavior.  │
  └─────────────────────────────────────────────────────────────┘

Comparison with Other Methods:
  - Full fine-tuning: Update all parameters (expensive)
  - LoRA: Learn low-rank weight updates
  - Prefix Tuning: Learn input embeddings (even cheaper!)
  - Prompt Tuning: Learn a single embedding vector

Benefits:
  - Extremely parameter-efficient (only prefix embeddings)
  - No model architecture changes needed
  - Can store many tasks as separate prefixes
  - Fast task switching (just swap the prefix)

References:
  - Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts" (2021)
  - Lester et al., "The Power of Scale for Parameter-Efficient Prompt Tuning" (2021)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PrefixTuningConfig:
    """
    Configuration for Prefix Tuning.

    Attributes:
        d_model: Model dimension
        n_layers: Number of attention layers
        n_heads: Number of attention heads
        prefix_length: Number of virtual prefix tokens
        prefix_hidden: Hidden dimension for prefix MLP
        dropout: Dropout probability
    """
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    prefix_length: int = 20
    prefix_hidden: int = 512
    dropout: float = 0.1


# ============================================================================
# PREFIX TUNING
# ============================================================================

class PrefixTuning:
    """
    Prefix Tuning implementation.

    The prefix is parameterized as:
    P = MLP(P_raw) where P_raw is a learnable embedding

    The MLP reparameterization helps optimization:
    - Without MLP: direct optimization of prefix embeddings is unstable
    - With MLP: smoother optimization landscape

    At each attention layer, the prefix provides additional K, V:
    - K_prefix: [n_layers, prefix_length, d_model]
    - V_prefix: [n_layers, prefix_length, d_model]

    These are concatenated with the original K, V from the input.
    """

    def __init__(self, config: PrefixTuningConfig):
        """
        Initialize Prefix Tuning.

        Args:
            config: Prefix Tuning configuration
        """
        self.config = config
        d_model = config.d_model
        n_layers = config.n_layers
        prefix_length = config.prefix_length
        prefix_hidden = config.prefix_hidden

        # Learnable prefix embeddings (raw)
        # These are the "virtual tokens" we optimize
        self.prefix_raw = np.random.randn(n_layers * 2, prefix_length, d_model) * 0.01

        # MLP to reparameterize prefix
        # Raw prefix → hidden → K and V for each layer
        self.mlp_1 = np.random.randn(d_model, prefix_hidden) * 0.01
        self.mlp_2 = np.random.randn(prefix_hidden, d_model * 2) * 0.01

        # Activation
        self.activation = lambda x: np.tanh(x)

    def get_prefix(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prefix K, V for all layers.

        Args:
            batch_size: Batch size

        Returns:
            prefix_k: Prefix keys [n_layers, batch_size, prefix_length, d_model]
            prefix_v: Prefix values [n_layers, batch_size, prefix_length, d_model]
        """
        config = self.config

        # Apply MLP reparameterization
        # [n_layers*2, prefix_length, d_model] → [n_layers*2, prefix_length, d_model]
        h = self.prefix_raw @ self.mlp_1  # [n_layers*2, prefix_length, prefix_hidden]
        h = self.activation(h)
        prefix_kv = h @ self.mlp_2  # [n_layers*2, prefix_length, d_model*2]

        # Split into K and V
        prefix_k = prefix_kv[:, :, :config.d_model]  # [n_layers*2, prefix_length, d_model]
        prefix_v = prefix_kv[:, :, config.d_model:]  # [n_layers*2, prefix_length, d_model]

        # Reshape to [n_layers, 2, prefix_length, d_model]
        prefix_k = prefix_k.reshape(config.n_layers, 2, config.prefix_length, config.d_model)
        prefix_v = prefix_v.reshape(config.n_layers, 2, config.prefix_length, config.d_model)

        # Take K and V for each layer
        prefix_k = prefix_k[:, 0]  # [n_layers, prefix_length, d_model]
        prefix_v = prefix_v[:, 0]  # [n_layers, prefix_length, d_model]

        # Expand batch dimension
        prefix_k = np.broadcast_to(
            prefix_k[:, np.newaxis, :, :],
            (config.n_layers, batch_size, config.prefix_length, config.d_model)
        )
        prefix_v = np.broadcast_to(
            prefix_v[:, np.newaxis, :, :],
            (config.n_layers, batch_size, config.prefix_length, config.d_model)
        )

        return prefix_k, prefix_v

    def count_parameters(self) -> int:
        """Count trainable parameters (only prefix, not the frozen model)."""
        return self.prefix_raw.size + self.mlp_1.size + self.mlp_2.size


# ============================================================================
# PROMPT TUNING (Simplified)
# ============================================================================

class PromptTuning:
    """
    Prompt Tuning: Even simpler than Prefix Tuning.

    Instead of per-layer prefixes, learn a single set of soft prompt
    embeddings prepended to the input.

    This is the simplest parameter-efficient method:
    - Only learn: [prefix_length, d_model] parameters
    - No MLP reparameterization
    - Works well with large models (11B+)
    """

    def __init__(self, d_model: int, prefix_length: int = 20):
        """
        Initialize Prompt Tuning.

        Args:
            d_model: Model dimension
            prefix_length: Number of soft prompt tokens
        """
        self.d_model = d_model
        self.prefix_length = prefix_length

        # Learnable prompt embeddings
        self.prompt_embeddings = np.random.randn(prefix_length, d_model) * 0.01

    def get_prompt(self, batch_size: int) -> np.ndarray:
        """
        Get prompt embeddings for a batch.

        Args:
            batch_size: Batch size

        Returns:
            prompt: [batch_size, prefix_length, d_model]
        """
        return np.broadcast_to(
            self.prompt_embeddings[np.newaxis, :, :],
            (batch_size, self.prefix_length, self.d_model)
        )

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return self.prompt_embeddings.size


# ============================================================================
# ADAPTER COMPARISON
# ============================================================================

def compare_peft_methods():
    """Compare Parameter-Efficient Fine-Tuning methods."""
    return """
    ┌──────────────────┬───────────┬───────────┬────────────┬──────────────┐
    │ Method           │ Params    │ Quality   │ Speed      │ Best For     │
    ├──────────────────┼───────────┼───────────┼────────────┼──────────────┤
    │ Full FT          │ 100%      │ Best      │ Slow       │ Max quality  │
    │ LoRA             │ 0.1-1%    │ ~99%      │ Fast       │ General PEFT │
    │ QLoRA            │ 0.1-1%    │ ~98%      │ Fast       │ Low memory   │
    │ Prefix Tuning    │ 0.01-0.1% │ ~95%      │ Very fast  │ Multi-task   │
    │ Prompt Tuning    │ 0.001%    │ ~93%      │ Very fast  │ Huge models  │
    │ Adapter Fusion   │ 0.1-1%    │ ~97%      │ Fast       │ Multi-task   │
    └──────────────────┴───────────┴───────────┴────────────┴──────────────┘
    """


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_prefix_tuning():
    """
    Demonstrate Prefix Tuning.

    Shows:
        1. Prefix Tuning configuration
        2. Prefix generation
        3. Parameter counting
        4. Comparison with other methods
    """
    print("=" * 70)
    print("Prefix Tuning — Demonstration")
    print("=" * 70)

    # Configuration
    config = PrefixTuningConfig(
        d_model=768,
        n_layers=12,
        n_heads=12,
        prefix_length=20,
        prefix_hidden=512,
    )

    # Create prefix tuning
    prefix = PrefixTuning(config)

    print(f"\nConfiguration:")
    print(f"  d_model: {config.d_model}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Prefix length: {config.prefix_length}")
    print(f"  Prefix hidden: {config.prefix_hidden}")

    # Parameter count
    prefix_params = prefix.count_parameters()
    model_params = 125_000_000  # Approximate for a small model

    print(f"\nParameter Count:")
    print(f"  Prefix parameters: {prefix_params:,}")
    print(f"  Model parameters: {model_params:,}")
    print(f"  Prefix ratio: {prefix_params/model_params:.4%}")

    # Get prefix
    batch_size = 4
    prefix_k, prefix_v = prefix.get_prefix(batch_size)

    print(f"\nPrefix Shapes:")
    print(f"  prefix_k: {prefix_k.shape}")
    print(f"  prefix_v: {prefix_v.shape}")

    # Prompt Tuning comparison
    print("\n[Prompt Tuning (simpler variant)]")
    prompt = PromptTuning(d_model=768, prefix_length=20)
    prompt_emb = prompt.get_prompt(batch_size=4)
    print(f"  Prompt shape: {prompt_emb.shape}")
    print(f"  Parameters: {prompt.count_parameters():,}")

    # Comparison
    print("\n[PEFT Methods Comparison]")
    print(compare_peft_methods())

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. Prefix tuning adds virtual tokens to every attention layer")
    print("  2. MLP reparameterization helps optimization")
    print("  3. Prompt tuning is even simpler (single embedding)")
    print("  4. Works best with very large models (10B+)")
    print("  5. Can store hundreds of tasks as separate prefixes")
    print("=" * 70)


if __name__ == "__main__":
    demo_prefix_tuning()
