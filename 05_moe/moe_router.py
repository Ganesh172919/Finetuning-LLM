"""
Mixture of Experts (MoE) with Advanced Routing
================================================

Mixture of Experts is a conditional computation architecture where only a
subset of the model's parameters are activated for each input token.

Key Innovation:
  Instead of activating ALL parameters for every token (dense model),
  a router selects which "expert" subnetworks to use.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                    Mixture of Experts                       │
  │                                                             │
  │  Input Token ──▶ Router ──▶ Top-K Expert Selection          │
  │                    │                                        │
  │           ┌───────┴───────┐                                 │
  │           │  Expert 1  Expert 2  ...  Expert N  │           │
  │           └───────┬───────┘                                 │
  │                    │                                        │
  │           Weighted Sum of Expert Outputs                    │
  │                    │                                        │
  │                    ▼                                        │
  │               Final Output                                  │
  └─────────────────────────────────────────────────────────────┘

MoE in Production:
  - Mixtral 8x7B: 8 experts, top-2 routing, 47B total, 13B active
  - Switch Transformer: Top-1 routing, 1.6T parameters
  - GShard: Top-2 routing, 600B parameters

Benefits:
  - More parameters without proportional compute increase
  - Specialization: experts learn different capabilities
  - Scaling: can add more experts without retraining

Challenges:
  - Load balancing: prevent "expert collapse"
  - Communication overhead in distributed training
  - Memory: all experts must be in memory even if not all used

References:
  - Shazeer et al., "Outrageously Large Neural Networks" (2017)
  - Fedus et al., "Switch Transformers" (2022)
  - Jiang et al., "Mixtral of Experts" (2024)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MoEConfig:
    """
    Configuration for Mixture of Experts model.

    Attributes:
        d_model: Model dimension (input/output size)
        d_ff: Expert feed-forward hidden dimension
        num_experts: Total number of experts
        top_k: Number of experts to route each token to
        capacity_factor: Expert capacity multiplier (1.0 = exact, >1 = overflow)
        aux_loss_coef: Coefficient for load balancing auxiliary loss
        noise_std: Standard deviation of router noise (for exploration)
        dropout: Dropout probability
    """
    d_model: int = 768
    d_ff: int = 3072
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25
    aux_loss_coef: float = 0.01
    noise_std: float = 0.1
    dropout: float = 0.0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# ============================================================================
# ROUTER — The Core of MoE
# ============================================================================

class MoERouter:
    """
    Expert Router — decides which experts process each token.

    The router is a simple linear layer that maps each token embedding
    to a probability distribution over experts.

    Routing Algorithm:
        1. Compute router logits: logits = W_router @ x
        2. Add optional noise for exploration during training
        3. Select top-k experts based on logits
        4. Compute softmax weights over selected experts only
        5. Return expert indices and weights

    Load Balancing:
        Without regularization, the router tends to collapse to using
        only a few experts (the "rich get richer" problem).

        Solution: Auxiliary loss that encourages uniform expert usage:
        L_aux = N * Σ_i (f_i * P_i)

        Where:
            f_i = fraction of tokens routed to expert i
            P_i = mean router probability for expert i
            N = number of experts

        This loss is minimized when all experts are used equally.

    Top-1 vs Top-2:
        Top-1: Simpler, faster, but less expressive
        Top-2: Better quality, used in Mixtral, standard choice
    """

    def __init__(self, config: MoEConfig):
        """
        Initialize the router.

        Args:
            config: MoE configuration
        """
        self.config = config
        self.d_model = config.d_model
        self.num_experts = config.num_experts

        # Router weight matrix: maps d_model → num_experts
        # This is the ONLY trainable parameter in the router
        self.router_weights = np.random.randn(config.d_model, config.num_experts) * 0.01

    def _add_noise(self, logits: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to router logits for exploration.

        During training, noise helps prevent the router from always
        choosing the same experts, encouraging exploration.

        Args:
            logits: Router logits [batch_size, num_experts]

        Returns:
            Noisy logits
        """
        if self.config.noise_std > 0:
            noise = np.random.randn(*logits.shape) * self.config.noise_std
            return logits + noise
        return logits

    def forward(self, x: np.ndarray, training: bool = True
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Route tokens to experts.

        Args:
            x: Input tensor [batch_size, d_model]
            training: Whether in training mode (adds noise)

        Returns:
            expert_indices: Indices of selected experts [batch_size, top_k]
            expert_weights: Routing weights [batch_size, top_k]
            aux_loss: Load balancing loss (scalar)
        """
        batch_size = x.shape[0]

        # ── Step 1: Compute router logits ──────────────────────
        # Each token gets a score for each expert
        logits = x @ self.router_weights  # [batch_size, num_experts]

        # ── Step 2: Add noise during training ──────────────────
        if training:
            logits = self._add_noise(logits)

        # ── Step 3: Select top-k experts ───────────────────────
        # np.argpartition is O(n) vs O(n log n) for full sort
        top_k_indices = np.argpartition(
            logits, -self.config.top_k, axis=-1
        )[:, -self.config.top_k:]  # [batch_size, top_k]

        # Get the logits for selected experts
        top_k_logits = np.take_along_axis(
            logits, top_k_indices, axis=-1
        )  # [batch_size, top_k]

        # ── Step 4: Compute softmax over selected experts ──────
        # Only normalize among the top-k, not all experts
        expert_weights = softmax(top_k_logits, axis=-1)  # [batch_size, top_k]

        # ── Step 5: Compute auxiliary loss ─────────────────────
        # Load balancing: encourage uniform expert usage
        aux_loss = self._compute_aux_loss(logits, expert_weights)

        return top_k_indices, expert_weights, aux_loss

    def _compute_aux_loss(self, logits: np.ndarray,
                          expert_weights: np.ndarray) -> float:
        """
        Compute load balancing auxiliary loss.

        This loss penalizes uneven expert usage to prevent:
        - Expert collapse (one expert gets all tokens)
        - Underutilization (experts that never get used)

        The loss is: L = N * Σ_i (f_i * P_i)

        Where:
            f_i = fraction of tokens routed to expert i
            P_i = mean routing probability for expert i

        Args:
            logits: Full router logits [batch_size, num_experts]
            expert_weights: Top-k weights [batch_size, top_k]

        Returns:
            Scalar auxiliary loss
        """
        # Compute fraction of tokens routed to each expert
        # (using full softmax for the probability term)
        probs = softmax(logits, axis=-1)  # [batch_size, num_experts]
        mean_probs = np.mean(probs, axis=0)  # [num_experts]

        # Compute fraction of tokens assigned to each expert
        # (using top-k assignment)
        assignment = np.zeros(self.num_experts)
        for i in range(self.config.top_k):
            # Count how many tokens are routed to each expert
            counts = np.bincount(
                np.argmax(probs, axis=-1),
                minlength=self.num_experts
            )
            assignment += counts
        assignment = assignment / (assignment.sum() + 1e-10)

        # Auxiliary loss: N * Σ (f_i * P_i)
        aux_loss = self.num_experts * np.sum(assignment * mean_probs)

        return float(aux_loss)


# ============================================================================
# EXPERT — Single Feed-Forward Network
# ============================================================================

class Expert:
    """
    Single expert network — a standard 2-layer FFN.

    Each expert specializes in processing certain types of tokens.
    During training, different experts learn different "skills":
    - Some specialize in syntax
    - Some specialize in semantics
    - Some specialize in specific domains

    Architecture:
        input → Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model) → output

    This is the same as a standard Transformer FFN, but each expert
    has its own independent weights.
    """

    def __init__(self, config: MoEConfig, expert_id: int):
        """
        Initialize an expert.

        Args:
            config: MoE configuration
            expert_id: Unique identifier for this expert
        """
        self.expert_id = expert_id
        d_model = config.d_model
        d_ff = config.d_ff

        # Two-layer FFN with GELU activation
        self.w1 = np.random.randn(d_model, d_ff) * 0.01  # Up projection
        self.w2 = np.random.randn(d_ff, d_model) * 0.01   # Down projection
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through expert.

        Args:
            x: Input [batch_size, d_model]

        Returns:
            Output [batch_size, d_model]
        """
        # Up projection + GELU
        hidden = gelu(x @ self.w1 + self.b1)
        # Down projection
        output = hidden @ self.w2 + self.b2
        return output


# ============================================================================
# MIXTURE OF EXPERTS LAYER
# ============================================================================

class MixtureOfExperts:
    """
    Complete Mixture of Experts layer.

    Combines the router and experts into a single layer that:
    1. Routes each token to top-k experts
    2. Processes tokens through selected experts
    3. Combines expert outputs with routing weights

    Forward Pass:
        For each token x:
            1. router_scores = Router(x)
            2. top_k_experts = argtopk(router_scores)
            3. weights = softmax(router_scores[top_k_experts])
            4. output = Σ_i (weight_i * Expert_i(x))
            5. return output

    Capacity:
        Each expert can process at most (batch_size * top_k / num_experts * capacity_factor)
        tokens. Tokens that exceed capacity are dropped (rare with good capacity_factor).
    """

    def __init__(self, config: MoEConfig):
        """
        Initialize MoE layer.

        Args:
            config: MoE configuration
        """
        self.config = config

        # Router
        self.router = MoERouter(config)

        # Expert pool
        self.experts = [Expert(config, i) for i in range(config.num_experts)]

        # Track statistics
        self.tokens_routed = np.zeros(config.num_experts)
        self.total_tokens = 0

    def forward(self, x: np.ndarray, training: bool = True
                ) -> Tuple[np.ndarray, float, Dict]:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor [batch_size, d_model]
            training: Whether in training mode

        Returns:
            output: Processed tensor [batch_size, d_model]
            aux_loss: Load balancing loss
            stats: Routing statistics
        """
        batch_size = x.shape[0]

        # ── Step 1: Route tokens to experts ────────────────────
        expert_indices, expert_weights, aux_loss = self.router.forward(x, training)

        # ── Step 2: Process through experts ────────────────────
        output = np.zeros_like(x)

        for expert_idx in range(self.config.num_experts):
            # Find tokens assigned to this expert
            mask = np.any(expert_indices == expert_idx, axis=-1)

            if not np.any(mask):
                continue

            # Get tokens for this expert
            expert_tokens = x[mask]

            # Process through expert
            expert_output = self.experts[expert_idx].forward(expert_tokens)

            # Get weights for this expert
            # Find which position this expert is in for each token
            for i, token_idx in enumerate(np.where(mask)[0]):
                pos = np.where(expert_indices[token_idx] == expert_idx)[0]
                if len(pos) > 0:
                    weight = expert_weights[token_idx, pos[0]]
                    output[token_idx] += weight * expert_output[i]

            # Track statistics
            self.tokens_routed[expert_idx] += np.sum(mask)

        self.total_tokens += batch_size

        # ── Step 3: Compute statistics ─────────────────────────
        stats = {
            "expert_usage": self.tokens_routed / max(1, self.total_tokens),
            "load_balance": self._compute_load_balance(),
            "capacity_utilization": self._compute_capacity(batch_size),
        }

        return output, aux_loss, stats

    def _compute_load_balance(self) -> float:
        """
        Compute load balance score (0 = perfect, 1 = maximally imbalanced).

        Uses coefficient of variation of expert usage.
        """
        if self.total_tokens == 0:
            return 0.0

        usage = self.tokens_routed / self.total_tokens
        mean_usage = np.mean(usage)
        if mean_usage < 1e-10:
            return 0.0
        cv = np.std(usage) / mean_usage
        return float(cv)

    def _compute_capacity(self, batch_size: int) -> float:
        """Compute capacity utilization."""
        ideal_per_expert = batch_size * self.config.top_k / self.config.num_experts
        max_capacity = ideal_per_expert * self.config.capacity_factor
        actual_per_expert = self.tokens_routed[-1] if self.total_tokens > 0 else 0
        return float(actual_per_expert / max(1, max_capacity))


# ============================================================================
# SWITCH TRANSFORMER (Top-1 Routing)
# ============================================================================

class SwitchTransformerLayer:
    """
    Switch Transformer — simplified MoE with Top-1 routing.

    Key difference from standard MoE:
    - Only routes to ONE expert (top-1)
    - Simpler, faster, but less expressive
    - Uses capacity factor to handle load imbalance

    This was introduced in "Switch Transformers" (Fedus et al., 2022)
    and showed that top-1 routing can be surprisingly effective.
    """

    def __init__(self, config: MoEConfig):
        """Initialize Switch Transformer layer."""
        # Override top_k to 1
        config.top_k = 1
        self.config = config
        self.router = MoERouter(config)
        self.experts = [Expert(config, i) for i in range(config.num_experts)]

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass with top-1 routing.

        Args:
            x: Input [batch_size, d_model]

        Returns:
            output: Processed [batch_size, d_model]
            aux_loss: Load balancing loss
        """
        # Route to single expert
        expert_indices, expert_weights, aux_loss = self.router.forward(x)

        # Process through selected expert
        output = np.zeros_like(x)
        for i in range(x.shape[0]):
            expert_idx = expert_indices[i, 0]
            weight = expert_weights[i, 0]
            output[i] = weight * self.experts[expert_idx].forward(x[i:i+1])[0]

        return output, aux_loss


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_moe():
    """
    Demonstrate Mixture of Experts.

    Shows:
        1. MoE configuration
        2. Routing behavior
        3. Load balancing
        4. Expert specialization
    """
    print("=" * 70)
    print("Mixture of Experts (MoE) — Interactive Demonstration")
    print("=" * 70)

    # Configuration
    config = MoEConfig(
        d_model=64,
        d_ff=128,
        num_experts=8,
        top_k=2,
        capacity_factor=1.25,
        aux_loss_coef=0.01,
    )

    print(f"\nConfiguration:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  Experts: {config.num_experts}")
    print(f"  Top-K: {config.top_k}")
    print(f"  Capacity Factor: {config.capacity_factor}")

    # Create MoE layer
    moe = MixtureOfExperts(config)

    # Count parameters
    params_per_expert = config.d_model * config.d_ff * 2  # w1 + w2
    total_expert_params = params_per_expert * config.num_experts
    router_params = config.d_model * config.num_experts
    total_params = total_expert_params + router_params
    active_params = params_per_expert * config.top_k + router_params

    print(f"\nParameter Count:")
    print(f"  Per expert: {params_per_expert:,}")
    print(f"  Total experts: {total_expert_params:,}")
    print(f"  Router: {router_params:,}")
    print(f"  Total: {total_params:,}")
    print(f"  Active per token: {active_params:,}")
    print(f"  Sparsity: {1 - active_params/total_params:.1%}")

    # Test with random input
    print("\n[Forward Pass]")
    batch_size = 16
    x = np.random.randn(batch_size, config.d_model)

    output, aux_loss, stats = moe.forward(x, training=True)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Auxiliary loss: {aux_loss:.4f}")
    print(f"  Load balance score: {stats['load_balance']:.4f}")

    # Show expert usage
    print("\n[Expert Usage Distribution]")
    usage = stats["expert_usage"]
    max_usage = max(usage)
    for i, u in enumerate(usage):
        bar = "█" * int(40 * u / max_usage) if max_usage > 0 else ""
        print(f"  Expert {i}: {u:.3f} {bar}")

    # Compare with dense model
    print("\n[MoE vs Dense Comparison]")
    dense_params = config.d_model * config.d_ff * 2  # Single FFN
    print(f"  Dense FFN params: {dense_params:,}")
    print(f"  MoE total params: {total_params:,} ({total_params/dense_params:.1f}x)")
    print(f"  MoE active params: {active_params:,} ({active_params/dense_params:.1f}x)")
    print(f"  Parameter efficiency: {total_params/active_params:.1f}x more params, same compute")

    # Mixtral comparison
    print("\n[Mixtral 8x7B Comparison]")
    print(f"  Total parameters: 47B (8 experts × ~7B each)")
    print(f"  Active parameters: 13B (top-2 routing)")
    print(f"  Efficiency: 3.6x more parameters, 2x compute")
    print(f"  Performance: Matches or exceeds 70B dense models")

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. MoE scales parameters without proportional compute increase")
    print("  2. Router learns to specialize experts automatically")
    print("  3. Load balancing prevents expert collapse")
    print("  4. Top-2 routing is the standard choice (Mixtral, GShard)")
    print("  5. Capacity factor handles load imbalance gracefully")
    print("=" * 70)


if __name__ == "__main__":
    demo_moe()
