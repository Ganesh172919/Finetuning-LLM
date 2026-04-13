"""
################################################################################
MIXTURE OF EXPERTS (MoE) — SCALING WITHOUT PROPORTIONAL COST
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Mixture of Experts?
    MoE is a technique where only a SUBSET of model parameters are
    activated for each input token. Instead of using all parameters,
    a "router" selects which "experts" to use.

    This allows:
    - Much larger total parameter count
    - Same computational cost as a smaller dense model
    - Better performance per FLOP

Why does it matter?
    Dense models: 70B params → 70B FLOPs per token
    MoE models: 175B params → 30B FLOPs per token
    → More parameters, less compute!

    This is how:
    - Mixtral 8x7B: 47B total, 13B active per token
    - DeepSeek-V2: 236B total, 21B active per token
    - GPT-4 (rumored): 1.8T total, ~200B active per token

How does it work?
    ┌─────────────────────────────────────────────┐
    │ Input token x                                │
    │        ↓                                      │
    │ Router: compute gating scores                │
    │   G(x) = softmax(W_g @ x)                   │
    │        ↓                                      │
    │ Select top-K experts (e.g., K=2)             │
    │        ↓                                      │
    │ Compute expert outputs (only selected)       │
    │   y = Σ G(x)_i * Expert_i(x)                │
    │        ↓                                      │
    │ Output                                        │
    └─────────────────────────────────────────────┘

Historical Evolution:
    - 2017: Shazeer et al. "Outrageously Large Neural Networks"
    - 2021: Switch Transformer (Google)
    - 2022: ST-MoE (Google)
    - 2023: Mixtral 8x7B (Mistral AI)
    - 2024: DeepSeek-V2, DeepSeek-MoE
    - 2025: DeepSeek-R1 (MoE + reasoning)

########################################

KEY CONCEPTS:

1. Router/Gating: Decides which experts to use
2. Experts: Individual feed-forward networks
3. Load Balancing: Ensure experts are used equally
4. Auxiliary Loss: Encourage balanced expert usage
5. Capacity Factor: How many tokens each expert handles

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List
import math

import sys
sys.path.append('..')
from ..02_transformers.layers import RMSNorm, FeedForward, GatedLinearUnit, TransformerBlock


################################################################################
# SECTION 1: ROUTER / GATING NETWORK
################################################################################

class TopKRouter:
    """
    Top-K Router for Mixture of Experts
    =====================================

    Definition: Routes each token to the top-K most relevant experts.

    Algorithm:
    1. Compute gating scores: G(x) = W_g @ x
    2. Select top-K experts by score
    3. Normalize scores for selected experts (softmax over top-K)
    4. Return expert indices and weights

    Why Top-K?
    - K=1: Fastest, but no expert mixing (Switch Transformer)
    - K=2: Best balance of quality and efficiency (Mixtral)
    - K=4: More mixing, higher compute cost

    Load Balancing Problem:
    Without regularization, router might send all tokens to one expert.
    Solution: Add auxiliary loss to encourage uniform routing.

    Interview Questions:
        1. "How does the router decide which expert to use?"
           The router is a learned linear layer that outputs logits
           for each expert. Top-K experts are selected by highest logits.

        2. "What happens if all tokens go to one expert?"
           This is called "expert collapse" — other experts don't learn.
           Load balancing loss prevents this.

        3. "Why is MoE more efficient than dense models?"
           Only a subset of parameters are used per token,
           so you get more parameters (capacity) for less compute.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01
    ):
        """
        Initialize router.

        Args:
            d_model: Model dimension
            num_experts: Total number of experts
            top_k: Number of experts to select per token
            aux_loss_weight: Weight for load balancing loss
        """
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight

        # Router weights
        self.W_gate = np.random.randn(d_model, num_experts) * 0.02

    def forward(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Route tokens to experts.

        Args:
            x: Input [batch × seq × d_model]

        Returns:
            expert_indices: [batch × seq × top_k] — which experts to use
            expert_weights: [batch × seq × top_k] — how much to weight each
            aux_loss: Load balancing loss (scalar)

        Example:
            x shape: [2, 4, 768]
            num_experts: 8, top_k: 2
            → expert_indices: [2, 4, 2] (2 experts per token)
            → expert_weights: [2, 4, 2] (weights sum to 1)
        """
        batch, seq, d = x.shape

        # Compute gating logits
        logits = np.matmul(x, self.W_gate)  # [batch × seq × num_experts]

        # Select top-K experts
        # Get indices of top-K experts
        top_k_indices = np.argsort(logits, axis=-1)[..., -self.top_k:]  # [batch × seq × top_k]

        # Get scores for selected experts
        top_k_scores = np.take_along_axis(logits, top_k_indices, axis=-1)

        # Normalize scores (softmax over selected experts)
        top_k_weights = np.exp(top_k_scores - np.max(top_k_scores, axis=-1, keepdims=True))
        top_k_weights = top_k_weights / np.sum(top_k_weights, axis=-1, keepdims=True)

        # Compute auxiliary loss for load balancing
        aux_loss = self._compute_aux_loss(logits, top_k_indices)

        return top_k_indices, top_k_weights, aux_loss

    def _compute_aux_loss(
        self,
        logits: np.ndarray,
        expert_indices: np.ndarray
    ) -> float:
        """
        Compute auxiliary loss for load balancing.

        The loss encourages uniform expert usage:
        - f_i: fraction of tokens routed to expert i
        - P_i: average routing probability for expert i
        - Loss = N * Σ(f_i * P_i)

        This penalizes experts that both:
        1. Receive many tokens (high f_i)
        2. Have high routing probability (high P_i)
        """
        batch, seq, num_experts = logits.shape

        # Fraction of tokens routed to each expert
        # (how many tokens selected this expert)
        one_hot = np.zeros((batch, seq, num_experts))
        for b in range(batch):
            for s in range(seq):
                for k in range(self.top_k):
                    one_hot[b, s, expert_indices[b, s, k]] = 1

        f = np.mean(one_hot, axis=(0, 1))  # [num_experts]

        # Average routing probability for each expert
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        P = np.mean(probs, axis=(0, 1))  # [num_experts]

        # Auxiliary loss
        aux_loss = num_experts * np.sum(f * P)
        return aux_loss * self.aux_loss_weight


################################################################################
# SECTION 2: EXPERT NETWORK
################################################################################

class Expert:
    """
    Expert Network (Feed-Forward)
    =============================

    Definition: An individual feed-forward network in the MoE layer.

    Each expert specializes in different types of tokens:
    - Some experts handle code
    - Some handle math
    - Some handle natural language
    - Some handle specific languages

    Architecture: Same as standard FFN
    - SwiGLU: x → W_gate → Swish → (⊙ x → W_up) → W_down

    In Mixtral-8x7B:
    - 8 experts, each with 14B parameters
    - 2 experts activated per token
    - Active parameters: ~14B (2 × 7B equivalent)
    - Total parameters: ~47B

    Interview Question:
        "Do different experts specialize in different things?"
        Yes! Research shows experts develop specializations:
        - Some handle syntax/grammar
        - Some handle specific domains (code, math)
        - Some handle specific languages
        - The specialization emerges naturally during training.
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        self.d_model = d_model
        self.d_ff = d_ff or int(2 / 3 * 4 * d_model)

        # SwiGLU architecture
        scale = math.sqrt(2.0 / (d_model + self.d_ff))
        self.W_gate = np.random.randn(d_model, self.d_ff) * scale
        self.W_up = np.random.randn(d_model, self.d_ff) * scale
        self.W_down = np.random.randn(self.d_ff, d_model) * scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Expert forward pass.

        Args:
            x: Input [batch × seq × d_model]
        Returns:
            output: [batch × seq × d_model]
        """
        # SwiGLU
        gate = np.matmul(x, self.W_gate)
        gate = gate * (1 / (1 + np.exp(-gate)))  # Swish

        up = np.matmul(x, self.W_up)
        hidden = gate * up
        output = np.matmul(hidden, self.W_down)

        return output


################################################################################
# SECTION 3: MIXTURE OF EXPERTS LAYER
################################################################################

class MixtureOfExperts:
    """
    Mixture of Experts Layer
    =========================

    Definition: A layer that routes tokens to different expert networks.

    Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Input x [batch × seq × d_model]                  │
    │   │                                               │
    │   ├──→ Router: compute expert scores             │
    │   │        ↓                                      │
    │   │    Select top-K experts                       │
    │   │                                               │
    │   ├──→ Expert 1 ──┐                               │
    │   ├──→ Expert 2 ──┤                               │
    │   ├──→ Expert 3 ──┤  (only top-K are computed)   │
    │   ├──→ Expert 4 ──┤                               │
    │   ├──→ Expert 5 ──┤                               │
    │   ├──→ Expert 6 ──┤                               │
    │   ├──→ Expert 7 ──┤                               │
    │   ├──→ Expert 8 ──┘                               │
    │   │                                               │
    │   └──→ Weighted sum of selected experts           │
    │          ↓                                        │
    │ Output                                            │
    └─────────────────────────────────────────────────┘

    Used by:
    - Mixtral 8x7B (8 experts, 2 active)
    - DeepSeek-V2 (160 experts, 6 active)
    - Switch Transformer (1 expert)
    - GPT-4 (rumored: 16 experts)

    Args:
        d_model: Model dimension
        num_experts: Total number of experts
        top_k: Number of experts to activate per token
        d_ff: Expert hidden dimension
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        aux_loss_weight: float = 0.01
    ):
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = TopKRouter(d_model, num_experts, top_k, aux_loss_weight)

        # Experts
        self.experts = [Expert(d_model, d_ff) for _ in range(num_experts)]

    def forward(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Forward pass of MoE layer.

        Args:
            x: Input [batch × seq × d_model]

        Returns:
            output: [batch × seq × d_model]
            aux_loss: Load balancing loss

        Algorithm:
            1. Route tokens to experts
            2. Compute expert outputs (only for selected experts)
            3. Weighted sum of expert outputs
        """
        batch, seq, d = x.shape

        # Step 1: Route tokens to experts
        expert_indices, expert_weights, aux_loss = self.router.forward(x)
        # expert_indices: [batch × seq × top_k]
        # expert_weights: [batch × seq × top_k]

        # Step 2: Compute expert outputs
        # For efficiency, we should only compute selected experts.
        # Here we compute all for simplicity.
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert.forward(x))
        # expert_outputs: list of [batch × seq × d_model]

        # Stack expert outputs
        expert_outputs = np.stack(expert_outputs, axis=2)
        # expert_outputs: [batch × seq × num_experts × d_model]

        # Step 3: Select and weight expert outputs
        # Gather selected expert outputs
        batch_idx = np.arange(batch)[:, None, None]
        seq_idx = np.arange(seq)[None, :, None]

        selected_outputs = expert_outputs[batch_idx, seq_idx, expert_indices]
        # selected_outputs: [batch × seq × top_k × d_model]

        # Weighted sum
        output = np.sum(
            selected_outputs * expert_weights[..., np.newaxis],
            axis=2
        )
        # output: [batch × seq × d_model]

        return output, aux_loss


################################################################################
# SECTION 4: MoE TRANSFORMER BLOCK
################################################################################

class MoETransformerBlock:
    """
    MoE Transformer Block
    ======================

    Definition: A transformer block that replaces the FFN with MoE.

    Architecture:
        x → RMSNorm → Self-Attention → + → h
        h → RMSNorm → MoE-FFN → + → output

    This is how Mixtral, DeepSeek-V2, etc. are built:
    - Attention layers are dense (shared across all tokens)
    - FFN layers are MoE (different experts per token)

    Interview Question:
        "Why put MoE in FFN but not attention?"
        Attention is already efficient (O(n²d) for context mixing).
        FFN is where most parameters live (4d² per layer).
        MoE in FFN gives the most parameter scaling benefit.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        n_kv_heads: Optional[int] = None
    ):
        from ..02_transformers.attention import MultiHeadAttention, GroupedQueryAttention

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Dense attention
        if n_kv_heads is not None and n_kv_heads < n_heads:
            self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        else:
            self.attention = MultiHeadAttention(d_model, n_heads)

        # MoE FFN
        self.moe = MixtureOfExperts(d_model, num_experts, top_k)

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Forward pass.

        Returns:
            output: [batch × seq × d_model]
            aux_loss: Load balancing loss
        """
        # Self-attention with residual
        h = self.norm1.forward(x)
        attn_out, _ = self.attention.forward(h, mask=mask)
        x = x + attn_out

        # MoE FFN with residual
        h = self.norm2.forward(x)
        moe_out, aux_loss = self.moe.forward(h)
        x = x + moe_out

        return x, aux_loss


################################################################################
# SECTION 5: TESTING & EXAMPLES
################################################################################

def demonstrate_moe():
    """Demonstrate Mixture of Experts."""
    print("=" * 70)
    print("MIXTURE OF EXPERTS DEMONSTRATION")
    print("=" * 70)

    batch_size = 2
    seq_len = 4
    d_model = 64
    n_heads = 4
    num_experts = 8
    top_k = 2

    # Router
    print("\n--- Router ---")
    router = TopKRouter(d_model, num_experts, top_k)
    x = np.random.randn(batch_size, seq_len, d_model)
    indices, weights, aux_loss = router.forward(x)
    print(f"Expert indices shape: {indices.shape}")
    print(f"Expert weights shape: {weights.shape}")
    print(f"Expert indices (batch 0):\n{indices[0]}")
    print(f"Expert weights (batch 0):\n{weights[0].round(3)}")
    print(f"Auxiliary loss: {aux_loss:.4f}")

    # MoE Layer
    print("\n--- MoE Layer ---")
    moe = MixtureOfExperts(d_model, num_experts, top_k)
    output, loss = moe.forward(x)
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {loss:.4f}")

    # MoE Transformer Block
    print("\n--- MoE Transformer Block ---")
    from ..02_transformers.attention import create_causal_mask
    mask = create_causal_mask(seq_len)
    block = MoETransformerBlock(d_model, n_heads, num_experts, top_k)
    output, loss = block.forward(x, mask)
    print(f"Block output shape: {output.shape}")

    # Show expert specialization concept
    print("\n--- Expert Specialization Concept ---")
    print("In a trained MoE model:")
    print("- Expert 0 might specialize in code")
    print("- Expert 1 might specialize in math")
    print("- Expert 2 might specialize in natural language")
    print("- Expert 3 might specialize in specific languages")
    print("- etc.")
    print("This specialization emerges naturally during training!")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_moe()


################################################################################
# REFERENCES
################################################################################

# [1] Shazeer, N., et al. (2017). Outrageously Large Neural Networks.
# [2] Fedus, W., et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models.
# [3] Jiang, A., et al. (2024). Mixtral of Experts.
# [4] DeepSeek-AI. (2024). DeepSeek-V2: A Strong, Economical, and Efficient MoE LLM.
# [5] Lepikhin, D., et al. (2021). GShard: Scaling Giant Models with Conditional Computation.

################################################################################
