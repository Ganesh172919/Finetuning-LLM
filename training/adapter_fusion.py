"""
Adapter Fusion: Combining Multiple Adapters
=============================================

Adapter Fusion combines multiple task-specific adapters into a single model
without interference. Unlike model merging (which combines full models),
Adapter Fusion works with lightweight adapter modules.

Key Insight:
  Instead of training one adapter for all tasks, train separate adapters
  per task, then fuse them with learned attention weights.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                    Adapter Fusion                            │
  │                                                              │
  │  Input → [Adapter₁] [Adapter₂] ... [Adapterₙ]              │
  │              ↓           ↓              ↓                    │
  │           h₁          h₂            hₙ                      │
  │              ↘           ↓              ↙                    │
  │                   Attention Layer                            │
  │                      (learned)                               │
  │                        ↓                                     │
  │                   Fused Output                               │
  └─────────────────────────────────────────────────────────────┘

Benefits:
  - No interference between tasks
  - Can add new tasks without retraining
  - Learned attention weights find optimal combination
  - Works with LoRA, prefix tuning, or any adapter method

References:
  - Pfeiffer et al., "AdapterHub: A Framework for Adapting Transformers" (2020)
  - Pfeiffer et al., "AdapterFusion" (2021)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AdapterFusionConfig:
    """
    Configuration for Adapter Fusion.

    Attributes:
        d_model: Model dimension
        n_adapters: Number of task-specific adapters
        adapter_dim: Bottleneck dimension for each adapter
        fusion_type: How to combine adapters (attention, average, learned)
    """
    d_model: int = 768
    n_adapters: int = 3
    adapter_dim: int = 64
    fusion_type: str = "attention"


# ============================================================================
# TASK-SPECIFIC ADAPTER
# ============================================================================

class TaskAdapter:
    """
    A single task-specific adapter module.

    Architecture: d_model → adapter_dim → d_model (with residual)

    This is a bottleneck adapter similar to LoRA but with
    explicit up/down projections and activation function.
    """

    def __init__(self, d_model: int, adapter_dim: int, task_name: str = ""):
        """
        Initialize task adapter.

        Args:
            d_model: Model dimension
            adapter_dim: Bottleneck dimension
            task_name: Name of the task
        """
        self.task_name = task_name
        self.d_model = d_model
        self.adapter_dim = adapter_dim

        # Down projection: d_model → adapter_dim
        self.down_proj = np.random.randn(d_model, adapter_dim) * 0.01

        # Up projection: adapter_dim → d_model
        self.up_proj = np.random.randn(adapter_dim, d_model) * 0.01

        # Activation (ReLU)
        self.activation = lambda x: np.maximum(0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through adapter.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            Adapter output [batch, seq_len, d_model]
        """
        # Down project
        h = x @ self.down_proj  # [batch, seq_len, adapter_dim]

        # Activation
        h = self.activation(h)

        # Up project
        output = h @ self.up_proj  # [batch, seq_len, d_model]

        # Residual connection
        return x + output


# ============================================================================
# ADAPTER FUSION
# ============================================================================

class AdapterFusion:
    """
    Adapter Fusion layer.

    Combines multiple task-specific adapters using learned attention weights.
    The attention mechanism learns which adapters are most useful for each input.

    Fusion Methods:
    1. Attention: Learned query-key attention over adapters
    2. Average: Simple average of adapter outputs
    3. Learned: Input-dependent weighted combination

    The attention-based fusion is the most expressive:
    - Query: input representation
    - Keys: adapter-specific learned vectors
    - Values: adapter outputs
    """

    def __init__(self, config: AdapterFusionConfig):
        """
        Initialize Adapter Fusion.

        Args:
            config: Fusion configuration
        """
        self.config = config
        d_model = config.d_model
        n_adapters = config.n_adapters

        # Create task-specific adapters
        self.adapters = [
            TaskAdapter(d_model, config.adapter_dim, f"task_{i}")
            for i in range(n_adapters)
        ]

        # Fusion attention parameters
        if config.fusion_type == "attention":
            # Query projection (from input)
            self.W_q = np.random.randn(d_model, d_model) * 0.01

            # Key projections (one per adapter)
            self.W_k = np.random.randn(n_adapters, d_model) * 0.01

        # Learned weights (for non-attention fusion)
        self.fusion_weights = np.ones(n_adapters) / n_adapters

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through Adapter Fusion.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            output: Fused adapter output [batch, seq_len, d_model]
            attention_weights: Attention weights over adapters [batch, seq_len, n_adapters]
        """
        # Get outputs from all adapters
        adapter_outputs = []
        for adapter in self.adapters:
            adapter_out = adapter.forward(x)
            adapter_outputs.append(adapter_out)

        # Stack: [batch, seq_len, n_adapters, d_model]
        stacked = np.stack(adapter_outputs, axis=2)

        if self.config.fusion_type == "attention":
            # Attention-based fusion
            # Query from input
            Q = x @ self.W_q  # [batch, seq_len, d_model]

            # Compute attention scores
            scores = np.zeros((*x.shape[:2], self.config.n_adapters))
            for i, adapter in enumerate(self.adapters):
                # Key for this adapter
                k = self.W_k[i]  # [d_model]
                # Score = dot product of query and key
                scores[:, :, i] = np.sum(Q * k, axis=-1)

            # Softmax over adapters
            scores_max = np.max(scores, axis=-1, keepdims=True)
            attn_weights = np.exp(scores - scores_max)
            attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

            # Weighted combination
            output = np.sum(stacked * attn_weights[:, :, :, np.newaxis], axis=2)

        elif self.config.fusion_type == "average":
            # Simple average
            output = np.mean(stacked, axis=2)
            attn_weights = np.ones((*x.shape[:2], self.config.n_adapters)) / self.config.n_adapters

        else:
            # Learned weights
            output = np.sum(stacked * self.fusion_weights[np.newaxis, np.newaxis, :, np.newaxis], axis=2)
            attn_weights = np.broadcast_to(self.fusion_weights, (*x.shape[:2], self.config.n_adapters))

        return output, attn_weights

    def get_adapter_usage(self) -> Dict[str, float]:
        """Get average adapter usage statistics."""
        # Run a dummy forward pass to get attention weights
        x = np.random.randn(1, 10, self.config.d_model)
        _, attn_weights = self.forward(x)

        usage = {}
        for i, adapter in enumerate(self.adapters):
            usage[adapter.task_name] = float(np.mean(attn_weights[:, :, i]))

        return usage


# ============================================================================
# MULTI-TASK ADAPTER MANAGER
# ============================================================================

class MultiTaskAdapterManager:
    """
    Manages multiple task-specific adapters and their fusion.

    Workflow:
    1. Train separate adapters for each task
    2. Freeze adapter weights
    3. Train fusion attention layer
    4. At inference: route inputs through fused adapters
    """

    def __init__(self, config: AdapterFusionConfig):
        self.config = config
        self.fusion = AdapterFusion(config)
        self.task_names = [f"task_{i}" for i in range(config.n_adapters)]

    def set_task_names(self, names: List[str]):
        """Set human-readable task names."""
        self.task_names = names
        for i, name in enumerate(names):
            self.fusion.adapters[i].task_name = name

    def forward(self, x: np.ndarray, task_id: Optional[int] = None
                ) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass with optional task routing.

        Args:
            x: Input [batch, seq_len, d_model]
            task_id: If specified, use only this adapter

        Returns:
            output: Model output
            metadata: Usage statistics
        """
        if task_id is not None:
            # Single adapter mode
            output = self.fusion.adapters[task_id].forward(x)
            attn_weights = np.zeros((*x.shape[:2], self.config.n_adapters))
            attn_weights[:, :, task_id] = 1.0
        else:
            # Fusion mode
            output, attn_weights = self.fusion.forward(x)

        metadata = {
            "adapter_usage": {
                self.task_names[i]: float(np.mean(attn_weights[:, :, i]))
                for i in range(self.config.n_adapters)
            },
            "fusion_type": self.config.fusion_type,
        }

        return output, metadata


# ============================================================================
# COMPARISON
# ============================================================================

def compare_adapter_methods():
    """Compare adapter fusion approaches."""
    return """
    ┌──────────────────┬───────────┬───────────┬────────────┬──────────────┐
    │ Method           │ Quality   │ Params    │ Flexibility│ Best For     │
    ├──────────────────┼───────────┼───────────┼────────────┼──────────────┤
    │ Single Adapter   │ Good      │ Low       │ Low        │ Single task  │
    │ Sequential       │ Medium    │ Low       │ Medium     │ Task chain   │
    │ Average Fusion   │ Medium    │ Low       │ High       │ Simple merge │
    │ Attention Fusion │ Best      │ Medium    │ High       │ Multi-task   │
    │ MoE-style        │ Good      │ Medium    │ High       │ Scale        │
    └──────────────────┴───────────┴───────────┴────────────┴──────────────┘
    """


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_adapter_fusion():
    """
    Demonstrate Adapter Fusion.

    Shows:
        1. Single adapter forward pass
        2. Multi-adapter fusion
        3. Attention weights visualization
        4. Task routing
    """
    print("=" * 70)
    print("Adapter Fusion — Demonstration")
    print("=" * 70)

    # Configuration
    config = AdapterFusionConfig(
        d_model=64,
        n_adapters=3,
        adapter_dim=16,
        fusion_type="attention",
    )

    # Create manager
    manager = MultiTaskAdapterManager(config)
    manager.set_task_names(["Sentiment", "NER", "QA"])

    print(f"\nConfiguration:")
    print(f"  d_model: {config.d_model}")
    print(f"  Adapters: {config.n_adapters}")
    print(f"  Adapter dim: {config.adapter_dim}")
    print(f"  Fusion type: {config.fusion_type}")

    # Count parameters
    params_per_adapter = config.d_model * config.adapter_dim * 2
    total_adapter_params = params_per_adapter * config.n_adapters
    fusion_params = config.d_model * config.d_model + config.n_adapters * config.d_model

    print(f"\nParameter Count:")
    print(f"  Per adapter: {params_per_adapter:,}")
    print(f"  Total adapters: {total_adapter_params:,}")
    print(f"  Fusion layer: {fusion_params:,}")
    print(f"  Total: {total_adapter_params + fusion_params:,}")

    # Forward pass
    print("\n[Fusion Forward Pass]")
    x = np.random.randn(2, 10, config.d_model)
    output, metadata = manager.forward(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Adapter usage:")
    for task, usage in metadata["adapter_usage"].items():
        print(f"    {task}: {usage:.3f}")

    # Single task mode
    print("\n[Single Task Mode]")
    output_single, metadata_single = manager.forward(x, task_id=0)
    print(f"  Using adapter: {manager.task_names[0]}")
    print(f"  Output shape: {output_single.shape}")

    # Comparison
    print("\n[Adapter Methods Comparison]")
    print(compare_adapter_methods())

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. Each adapter handles a specific task")
    print("  2. Fusion learns optimal adapter combination")
    print("  3. Can add new tasks without retraining existing adapters")
    print("  4. Attention-based fusion is most expressive")
    print("  5. Works with LoRA, prefix tuning, or any adapter method")
    print("=" * 70)


if __name__ == "__main__":
    demo_adapter_fusion()
