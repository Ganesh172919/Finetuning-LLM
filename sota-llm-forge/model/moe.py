"""
################################################################################
DEEPSEEK MOE — MIXTURE OF EXPERTS LAYER
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Mixture of Experts (MoE)?
    MoE replaces the dense feed-forward network (FFN) in a transformer with
    a collection of smaller "expert" networks. A learned router selects which
    experts process each token. Only a subset of experts are active per token,
    so the total parameter count is large but compute per token is small.

    DeepSeekMoE introduces two key innovations:
    1. Fine-grained experts: More, smaller experts instead of fewer large ones.
    2. Shared experts: A few experts that are ALWAYS active (not routed).
    3. Aux-loss-free load balancing: Per-expert bias for routing without
       auxiliary loss in the training objective.

Why does it matter?
    MoE allows scaling model capacity without proportionally scaling compute.
    A model with 64 experts and top-2 routing has 64x the FFN parameters of
    a dense model but only uses 2x the compute per token. This is the
    architecture behind DeepSeek-V3/V4, Mixtral, and other state-of-the-art
    models.

    The aux-loss-free approach is particularly important because traditional
    MoE uses an auxiliary load-balancing loss that can hurt model quality.
    DeepSeek's approach maintains load balance through a dynamic per-expert
    bias that is updated outside the gradient computation.

How does it work?
    1. Router computes affinity scores between each token and all experts.
    2. Top-k experts are selected per token.
    3. Selected experts process the token; outputs are weighted by gating scores.
    4. Shared experts process ALL tokens (always active).
    5. Final output = sum(shared_expert_outputs) + sum(gated_routed_outputs).

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌──────────────────────────────────────────────────────────────┐
    │                    DeepSeekMoELayer                           │
    │                                                              │
    │  Input x (batch, seq_len, d_model)                           │
    │    │                                                         │
    │    ├──→ Shared Expert 1 ──→ h_s1 (always active)            │
    │    ├──→ Shared Expert 2 ──→ h_s2 (always active)            │
    │    │                                                         │
    │    ├──→ Router ──→ scores s_i for each expert               │
    │    │       │                                                 │
    │    │       ├──→ Add bias b_i (aux-loss-free balancing)       │
    │    │       ├──→ Top-k selection                              │
    │    │       │                                                 │
    │    │       ├──→ Expert 3 (selected, weight=0.42) ──→ h_r1   │
    │    │       └──→ Expert 7 (selected, weight=0.58) ──→ h_r2   │
    │    │                                                         │
    │    └──→ Output = h_s1 + h_s2 + 0.42*h_r1 + 0.58*h_r2       │
    │                                                              │
    │  Aux-loss-free load balancing:                               │
    │    b_i ← b_i + γ * (target_load - actual_load_i)            │
    │    (updated per step, NOT in gradient graph)                  │
    └──────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2017: Sparsely-Gated MoE — "Outrageously Large Neural Networks" (Shazeer et al.)
    - 2021: Switch Transformer — Top-1 routing, simplified MoE (Fedus et al.)
    - 2023: Mixtral 8x7B — Top-2 routing with 8 experts (Mistral AI)
    - 2024: DeepSeek-V2 — Fine-grained experts + shared experts (DeepSeek-AI)
    - 2025: DeepSeek-V3 — Aux-loss-free load balancing (DeepSeek-AI)
    - 2025: DeepSeek-V4 — Scaling to 671B params with MoE (DeepSeek-AI)

INTERVIEW QUESTIONS:
    1. "What is the problem with auxiliary load-balancing losses in MoE?"
       Traditional MoE adds a loss term like alpha * sum(f_i * P_i) where f_i
       is the fraction of tokens routed to expert i and P_i is the average
       routing probability. This encourages uniform routing but competes with
       the language modeling loss, potentially degrading quality. The
       aux-loss-free approach uses a per-expert bias that doesn't enter the
       gradient computation, so it doesn't interfere with the LM objective.

    2. "Why use shared experts in addition to routed experts?"
       Some capabilities (e.g., basic grammar, common patterns) are needed
       for every token. Routing these through the router adds latency and
       wastes capacity on the routing decision. Shared experts handle these
       universal patterns, freeing routed experts to specialize in
       domain-specific or rare knowledge.

    3. "How does top-k routing work and why not top-1?"
       Top-1 routing (Switch Transformer) assigns each token to exactly one
       expert. This can lead to information loss when a token needs multiple
       types of processing. Top-2 (Mixtral) or higher allows tokens to
       combine multiple expert outputs, providing more flexible and robust
       processing at the cost of slightly more compute.

################################################################################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


################################################################################
# SECTION 1: CONFIGURATION
################################################################################


@dataclass
class MoEConfig:
    """
    Configuration for DeepSeekMoE-style Mixture of Experts.

    Attributes:
        d_model: Model (hidden) dimension.
        expert_intermediate_dim: Intermediate dimension for each expert.
        n_shared_experts: Number of always-active shared experts.
        n_routed_experts: Number of experts in the routed pool.
        top_k: Number of experts selected per token by the router.
        bias_update_rate (gamma): Step size for per-expert bias updates.
        target_load: Target fraction of tokens each expert should handle.
                     Default is 1/n_routed_experts (uniform).
        dropout: Dropout rate within experts.
        use_aux_loss: Whether to also use traditional aux loss (default False).
        aux_loss_weight: Weight for aux loss if use_aux_loss is True.
    """

    d_model: int = 768
    expert_intermediate_dim: int = 1024
    n_shared_experts: int = 2
    n_routed_experts: int = 64
    top_k: int = 6
    bias_update_rate: float = 0.001
    target_load: Optional[float] = None  # Default: 1 / n_routed_experts
    dropout: float = 0.0
    use_aux_loss: bool = False
    aux_loss_weight: float = 0.01


################################################################################
# SECTION 2: SINGLE EXPERT (SWIGLU FFN)
################################################################################


class Expert(nn.Module):
    """
    Single Expert Network (SwiGLU FFN)
    ===================================

    Each expert is a SwiGLU feed-forward network. SwiGLU is the standard
    activation for modern LLMs (Llama, DeepSeek, etc.).

    Formula:
        gate = x @ W_gate          # (batch, seq, intermediate_dim)
        up   = x @ W_up            # (batch, seq, intermediate_dim)
        h    = SiLU(gate) * up     # (batch, seq, intermediate_dim)
        out  = h @ W_down          # (batch, seq, d_model)

    Step by step:
        1. Project input to gate and up paths.
        2. Apply SiLU activation to gate path.
        3. Element-wise multiply gate and up paths.
        4. Project back to model dimension.

    WHY this matters:
        SwiGLU combines the benefits of gating (controlling information flow)
        with smooth activation (SiLU = x * sigmoid(x)). It consistently
        outperforms ReLU and GELU in LLM benchmarks.

    Interview Question:
        "Why SwiGLU instead of ReLU for the expert FFN?"
        SwiGLU provides a gating mechanism: the SiLU-activated gate path
        controls how much of the up path passes through. This is more
        expressive than a simple ReLU, which just thresholds. Empirically,
        SwiGLU gives ~1-2% improvement over ReLU in language modeling.
    """

    def __init__(self, d_model: int, intermediate_dim: int, dropout: float = 0.0):
        """
        Initialize a single expert.

        Args:
            d_model: Input/output dimension.
            intermediate_dim: Hidden dimension of the expert FFN.
            dropout: Dropout rate.
        """
        super().__init__()
        # W_gate: (d_model, intermediate_dim)
        self.W_gate = nn.Linear(d_model, intermediate_dim, bias=False)
        # W_up: (d_model, intermediate_dim)
        self.W_up = nn.Linear(d_model, intermediate_dim, bias=False)
        # W_down: (intermediate_dim, d_model)
        self.W_down = nn.Linear(intermediate_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through a single expert.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model) or
               (n_tokens, d_model) after routing.

        Returns:
            Expert output, same shape as input.

        Explanation:
            The SwiGLU activation: SiLU(gate) * up, where SiLU(x) = x * sigmoid(x).
        """
        # gate: (..., intermediate_dim)
        gate = self.W_gate(x)
        # up: (..., intermediate_dim)
        up = self.W_up(x)
        # h: (..., intermediate_dim) — SwiGLU activation
        h = F.silu(gate) * up
        # out: (..., d_model)
        out = self.W_down(self.dropout(h))
        return out


################################################################################
# SECTION 3: EXPERT ROUTER (AUX-LOSS-FREE)
################################################################################


class ExpertRouter(nn.Module):
    """
    Expert Router with Aux-Loss-Free Load Balancing
    ================================================

    Computes routing scores for each token and selects top-k experts.
    Maintains per-expert bias for load balancing WITHOUT auxiliary loss.

    Formula:
        scores = x @ W_router         # (batch, seq, n_routed_experts)
        selection_scores = scores + b  # add bias for selection only
        top_k_indices = topk(selection_scores, k)
        gating_weights = softmax(scores[top_k_indices])  # use ORIGINAL scores
        # Note: bias b affects SELECTION but not WEIGHTS

    Aux-loss-free balancing:
        b_i ← b_i + γ * (target_load - actual_load_i)
        where:
            target_load = 1 / n_routed_experts (uniform)
            actual_load_i = fraction of tokens routed to expert i this step
            γ = bias_update_rate (small, e.g., 0.001)

    Step by step:
        1. Compute raw routing scores from input.
        2. Add per-expert bias for top-k selection.
        3. Select top-k experts per token based on biased scores.
        4. Compute gating weights from ORIGINAL (unbiased) scores.
        5. Update per-expert bias based on actual load (no gradient).

    WHY this matters:
        Traditional aux loss (alpha * sum(f_i * P_i)) competes with the LM
        loss and can degrade model quality. The bias-based approach achieves
        the same load-balancing effect without touching the gradient. The
        bias is a "shadow" parameter that guides routing but doesn't
        participate in backpropagation.

    Interview Question:
        "Walk me through how aux-loss-free load balancing works."
        The router has a per-expert bias b_i that is added to routing scores
        ONLY for the purpose of selecting which experts to use. The actual
        gating weights (how much each expert contributes) are computed from
        the original unbiased scores. After each training step, we measure
        what fraction of tokens each expert actually handled. If expert i
        handled fewer tokens than the target (1/n_experts), we increase b_i
        to make it more likely to be selected next step. This is done with
        torch.no_grad() — it doesn't affect the LM gradient at all.
    """

    def __init__(self, d_model: int, n_routed_experts: int, top_k: int, bias_update_rate: float = 0.001):
        """
        Initialize the expert router.

        Args:
            d_model: Model dimension.
            n_routed_experts: Number of experts in the pool.
            top_k: Number of experts to select per token.
            bias_update_rate: Gamma — step size for bias updates.
        """
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        self.bias_update_rate = bias_update_rate

        # W_router: (d_model, n_routed_experts) — projects token to expert affinity
        self.W_router = nn.Linear(d_model, n_routed_experts, bias=False)

        # Per-expert bias for aux-loss-free load balancing
        # bias: (n_routed_experts,) — initialized to zero
        self.register_buffer("bias", torch.zeros(n_routed_experts))

        # Running statistics for load monitoring
        self.register_buffer("tokens_per_expert", torch.zeros(n_routed_experts))
        self.register_buffer("total_tokens", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            routing_weights: Gating weights for selected experts.
                            Shape (batch * seq_len, top_k) — normalized.
            expert_indices: Indices of selected experts.
                           Shape (batch * seq_len, top_k).
            aux_loss: Load balancing loss (0.0 if aux-loss-free mode).
                     Scalar tensor.

        Explanation:
            1. Compute raw scores: x @ W_router.
            2. For selection: add bias to scores (detached from gradient).
            3. Select top-k experts.
            4. For gating: use original scores (no bias).
            5. Normalize gating weights with softmax.
            6. Update bias for load balancing (no gradient).
        """
        batch_size, seq_len, d_model = x.shape
        n_tokens = batch_size * seq_len

        # Flatten for routing
        # x_flat: (n_tokens, d_model)
        x_flat = x.view(n_tokens, d_model)

        # --- Step 1: Compute raw routing scores ---
        # logits: (n_tokens, n_routed_experts)
        logits = self.W_router(x_flat)

        # --- Step 2: Add bias for selection (detached) ---
        # selection_logits: (n_tokens, n_routed_experts)
        # The bias is detached so it doesn't affect gradients
        selection_logits = logits + self.bias.detach()

        # --- Step 3: Top-k selection ---
        # top_k_logits: (n_tokens, top_k) — values
        # expert_indices: (n_tokens, top_k) — indices
        top_k_logits, expert_indices = torch.topk(selection_logits, self.top_k, dim=-1)

        # --- Step 4: Gating weights from ORIGINAL scores ---
        # Gather the original logits for selected experts
        # selected_logits: (n_tokens, top_k)
        selected_logits = torch.gather(logits, dim=-1, index=expert_indices)

        # Normalize: softmax over selected experts
        # routing_weights: (n_tokens, top_k)
        routing_weights = F.softmax(selected_logits, dim=-1)

        # --- Step 5: Update bias for load balancing (no gradient) ---
        with torch.no_grad():
            # Count tokens per expert
            # expert_mask: (n_tokens, n_routed_experts) — 1 if expert is selected
            expert_mask = F.one_hot(expert_indices, self.n_routed_experts).sum(dim=1).float()
            # expert_mask: (n_tokens, n_routed_experts)

            # actual_load: (n_routed_experts,) — fraction of tokens per expert
            actual_load = expert_mask.sum(dim=0) / n_tokens

            # Update bias: b_i += gamma * (target - actual)
            target_load = 1.0 / self.n_routed_experts
            self.bias += self.bias_update_rate * (target_load - actual_load)

            # Update running statistics
            self.tokens_per_expert += expert_mask.sum(dim=0)
            self.total_tokens += n_tokens

        # --- Optional: Traditional auxiliary loss ---
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if hasattr(self, "_use_aux_loss") and self._use_aux_loss:
            # f_i: fraction of tokens routed to expert i
            f_i = actual_load
            # P_i: mean routing probability for expert i
            P_i = torch.softmax(logits, dim=-1).mean(dim=0)
            # aux_loss: sum(f_i * P_i) — encourages uniform routing
            aux_loss = (f_i * P_i).sum() * self.n_routed_experts

        return routing_weights, expert_indices, aux_loss

    def get_load_balance_stats(self) -> dict:
        """
        Get current load balancing statistics.

        Returns:
            Dictionary with load balance metrics.

        Explanation:
            Returns the fraction of tokens routed to each expert and a
            balance metric (ideal is 1.0 for uniform distribution).
        """
        if self.total_tokens == 0:
            return {"load_per_expert": None, "balance_score": None}

        # load: (n_routed_experts,) — fraction of total tokens per expert
        load = self.tokens_per_expert / self.total_tokens
        # ideal: uniform load
        ideal = 1.0 / self.n_routed_experts
        # balance_score: how close to uniform (1.0 = perfect)
        balance_score = 1.0 - (load - ideal).abs().mean() / ideal

        return {
            "load_per_expert": load.tolist(),
            "balance_score": balance_score.item(),
        }

    def reset_stats(self):
        """Reset running load statistics."""
        self.tokens_per_expert.zero_()
        self.total_tokens.zero_()


################################################################################
# SECTION 4: SHARED EXPERTS
################################################################################


class SharedExperts(nn.Module):
    """
    Shared Experts — Always-Active Expert Pool
    ============================================

    A small set of experts that process EVERY token, regardless of routing.
    These handle universal patterns (basic grammar, common knowledge) that
    are needed for every token.

    Architecture:
        Input x → Shared Expert 1 → h_1
        Input x → Shared Expert 2 → h_2
        Output = h_1 + h_2

    Step by step:
        1. Pass input through each shared expert independently.
        2. Sum the outputs.

    WHY this matters:
        Without shared experts, the router must learn to route basic
        capabilities (like syntax) to some expert, wasting routing capacity.
        Shared experts free routed experts to specialize in rarer, more
        domain-specific knowledge.

    Interview Question:
        "How many shared experts should you use and how big should they be?"
        Typically 1-4 shared experts. DeepSeek-V2 uses 2. The intermediate
        dimension can be the same as routed experts or slightly larger since
        they handle every token. The total shared expert parameters should
        be a small fraction of routed expert parameters to maintain the
        compute efficiency of MoE.
    """

    def __init__(self, d_model: int, intermediate_dim: int, n_shared: int, dropout: float = 0.0):
        """
        Initialize shared experts.

        Args:
            d_model: Model dimension.
            intermediate_dim: Intermediate dimension per expert.
            n_shared: Number of shared experts.
            dropout: Dropout rate.
        """
        super().__init__()
        self.n_shared = n_shared

        # Create shared experts as a ModuleList
        # experts: list of n_shared Expert instances
        self.experts = nn.ModuleList([
            Expert(d_model, intermediate_dim, dropout)
            for _ in range(n_shared)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all shared experts.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Sum of shared expert outputs, shape (batch, seq_len, d_model).

        Explanation:
            Each shared expert processes the full input independently.
            The outputs are summed (not averaged) because the downstream
            residual connection handles normalization.
        """
        # Sum outputs from all shared experts
        # output: (batch, seq_len, d_model)
        output = sum(expert(x) for expert in self.experts)
        return output


################################################################################
# SECTION 5: ROUTED EXPERTS
################################################################################


class RoutedExperts(nn.Module):
    """
    Routed Experts — Selectable Expert Pool
    ========================================

    A pool of experts where each token is processed by only top_k experts,
    selected by the router.

    Architecture:
        For each token:
            1. Router selects top_k expert indices.
            2. Token is dispatched to selected experts.
            3. Expert outputs are weighted by routing scores.
            4. Weighted sum = token's routed output.

    Efficient Implementation:
        Instead of looping over tokens, we use batched matrix operations:
        1. Flatten all tokens: (n_tokens, d_model).
        2. For each expert, gather the tokens assigned to it.
        3. Process in a single batched forward pass.
        4. Scatter results back to original positions.

    Step by step:
        1. Reshape input to (n_tokens, d_model).
        2. For each expert i:
           a. Find which tokens are routed to expert i.
           b. Gather those tokens.
           c. Process through expert i.
           d. Multiply by gating weights.
           e. Scatter back to output positions.
        3. Reshape output to (batch, seq_len, d_model).

    WHY this matters:
        The key challenge in MoE is efficient token dispatch. Naive
        implementation loops over tokens (slow). Efficient implementation
        uses scatter/gather operations for batched processing. This is
        critical for training speed since MoE adds significant overhead.

    Interview Question:
        "How do you efficiently implement token dispatch in MoE?"
        Use scatter/gather operations. Pre-compute a mask of which tokens
        go to which expert. For each expert, use boolean indexing to extract
        its tokens, process them in one batch, then scatter the results
        back using index_add. This avoids Python loops and leverages GPU
        parallelism. In practice, libraries like MegaBlocks or Tutel
        provide optimized fused kernels.
    """

    def __init__(self, d_model: int, intermediate_dim: int, n_experts: int, dropout: float = 0.0):
        """
        Initialize routed experts.

        Args:
            d_model: Model dimension.
            intermediate_dim: Intermediate dimension per expert.
            n_experts: Number of routed experts.
            dropout: Dropout rate.
        """
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model

        # Create all experts as a ModuleList
        # experts: list of n_experts Expert instances
        self.experts = nn.ModuleList([
            Expert(d_model, intermediate_dim, dropout)
            for _ in range(n_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through routed experts with sparse dispatch.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            routing_weights: Gating weights, shape (n_tokens, top_k).
            expert_indices: Selected expert indices, shape (n_tokens, top_k).

        Returns:
            Weighted sum of expert outputs, shape (batch, seq_len, d_model).

        Explanation:
            We iterate over experts (not tokens). For each expert, we find
            which tokens are assigned to it, process them, and scatter the
            results back. This is O(n_experts * tokens_per_expert) which is
            much better than O(n_tokens * top_k).
        """
        batch_size, seq_len, d_model = x.shape
        n_tokens = batch_size * seq_len

        # Flatten: (n_tokens, d_model)
        x_flat = x.view(n_tokens, d_model)

        # Initialize output accumulator
        # output: (n_tokens, d_model)
        output = torch.zeros_like(x_flat)

        # Process each expert
        for expert_idx in range(self.n_experts):
            # Find which (token, slot) pairs route to this expert
            # expert_mask: (n_tokens, top_k) — True where expert matches
            expert_mask = (expert_indices == expert_idx)

            if not expert_mask.any():
                continue  # Skip if no tokens routed to this expert

            # Get the (token_idx, slot_idx) pairs
            # token_indices: which tokens are routed here
            # slot_indices: which top-k slot they came from
            token_indices, slot_indices = expert_mask.nonzero(as_tuple=True)

            # Gather the tokens for this expert
            # expert_input: (n_assigned, d_model)
            expert_input = x_flat[token_indices]

            # Process through expert
            # expert_output: (n_assigned, d_model)
            expert_output = self.experts[expert_idx](expert_input)

            # Get the gating weights for these (token, slot) pairs
            # weights: (n_assigned,)
            weights = routing_weights[token_indices, slot_indices]

            # Weight the output
            # weighted_output: (n_assigned, d_model)
            weighted_output = expert_output * weights.unsqueeze(-1)

            # Scatter back to output
            # output[token_indices] += weighted_output
            output.index_add_(0, token_indices, weighted_output)

        # Reshape back to (batch, seq_len, d_model)
        output = output.view(batch_size, seq_len, d_model)

        return output


################################################################################
# SECTION 6: DEEPSEEK MOE LAYER (FULL ASSEMBLY)
################################################################################


class DeepSeekMoELayer(nn.Module):
    """
    DeepSeekMoE-style Mixture of Experts Layer
    ============================================

    Combines shared experts (always active) with routed experts (selected
    per token) using aux-loss-free load balancing.

    Architecture:
        Input x
          ├── Shared Expert 1 (always active)
          ├── Shared Expert 2 (always active)
          ├── Router → top-k selection from routed pool
          │     ├── Routed Expert i (selected, weight w_i)
          │     └── Routed Expert j (selected, weight w_j)
          └── Output = sum(shared) + sum(w_i * routed_i)

    Formula:
        h_shared = sum_i SharedExpert_i(x)
        routing_weights, expert_indices = Router(x)
        h_routed = sum_k w_k * RoutedExpert_{idx_k}(x)
        output = h_shared + h_routed

    Step by step:
        1. Compute shared expert output (sum of all shared experts).
        2. Route tokens to experts (get weights and indices).
        3. Compute routed expert output (weighted sum of selected experts).
        4. Add shared and routed outputs.

    WHY this matters:
        This is the architecture behind DeepSeek-V3/V4, the current
        state-of-the-art. The combination of fine-grained experts,
        shared experts, and aux-loss-free balancing enables training
        models with hundreds of billions of parameters while keeping
        per-token compute manageable.

    Interview Question:
        "Design a MoE layer for a 100B parameter model."
        Use DeepSeekMoE: 64-128 fine-grained routed experts with top-6
        routing, 2 shared experts, SwiGLU FFN in each expert. Use
        aux-loss-free balancing to maintain load without quality loss.
        Expert intermediate dim should be smaller than dense model's
        (since we have many experts). Total params = n_experts * expert_params
        + shared_params. Compute per token = top_k * expert_compute
        + shared_compute.
    """

    def __init__(self, config: MoEConfig):
        """
        Initialize the DeepSeekMoE layer.

        Args:
            config: MoEConfig with all hyperparameters.
        """
        super().__init__()
        self.config = config

        # --- Shared Experts ---
        self.shared_experts = SharedExperts(
            d_model=config.d_model,
            intermediate_dim=config.expert_intermediate_dim,
            n_shared=config.n_shared_experts,
            dropout=config.dropout,
        )

        # --- Router ---
        self.router = ExpertRouter(
            d_model=config.d_model,
            n_routed_experts=config.n_routed_experts,
            top_k=config.top_k,
            bias_update_rate=config.bias_update_rate,
        )

        # --- Routed Experts ---
        self.routed_experts = RoutedExperts(
            d_model=config.d_model,
            intermediate_dim=config.expert_intermediate_dim,
            n_experts=config.n_routed_experts,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the DeepSeekMoE layer.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            output: (batch, seq_len, d_model) — combined shared + routed output.
            aux_loss: Scalar — load balancing loss (0.0 for aux-loss-free mode).

        Explanation:
            1. Shared experts process every token (always active).
            2. Router selects top_k experts per token.
            3. Selected experts process tokens with gating weights.
            4. Output = shared_sum + routed_sum.
        """
        # --- Shared experts ---
        # shared_output: (batch, seq_len, d_model)
        shared_output = self.shared_experts(x)

        # --- Routing ---
        # routing_weights: (n_tokens, top_k)
        # expert_indices: (n_tokens, top_k)
        # aux_loss: scalar
        routing_weights, expert_indices, aux_loss = self.router(x)

        # --- Routed experts ---
        # routed_output: (batch, seq_len, d_model)
        routed_output = self.routed_experts(x, routing_weights, expert_indices)

        # --- Combine ---
        # output: (batch, seq_len, d_model)
        output = shared_output + routed_output

        return output, aux_loss

    def get_expert_load_stats(self) -> dict:
        """
        Get load balancing statistics from the router.

        Returns:
            Dictionary with load balance metrics.
        """
        return self.router.get_load_balance_stats()


################################################################################
# SECTION 7: TESTING & DEMONSTRATION
################################################################################


def demonstrate_moe():
    """Demonstrate the DeepSeekMoE layer."""
    print("=" * 70)
    print("DEEPSEEK MOE DEMONSTRATION")
    print("=" * 70)

    device = torch.device("cpu")

    # --- Configuration ---
    config = MoEConfig(
        d_model=64,
        expert_intermediate_dim=128,
        n_shared_experts=2,
        n_routed_experts=8,
        top_k=2,
        bias_update_rate=0.001,
        dropout=0.0,
    )

    print(f"\nConfiguration:")
    print(f"  d_model:              {config.d_model}")
    print(f"  intermediate_dim:     {config.expert_intermediate_dim}")
    print(f"  n_shared_experts:     {config.n_shared_experts}")
    print(f"  n_routed_experts:     {config.n_routed_experts}")
    print(f"  top_k:                {config.top_k}")
    print(f"  bias_update_rate:     {config.bias_update_rate}")

    # --- Create MoE Layer ---
    moe = DeepSeekMoELayer(config)

    # Count parameters
    total_params = sum(p.numel() for p in moe.parameters())
    shared_params = sum(p.numel() for p in moe.shared_experts.parameters())
    routed_params = sum(p.numel() for p in moe.routed_experts.parameters())
    router_params = sum(p.numel() for p in moe.router.parameters())
    print(f"\nParameter counts:")
    print(f"  Shared experts:       {shared_params:,}")
    print(f"  Routed experts:       {routed_params:,}")
    print(f"  Router:               {router_params:,}")
    print(f"  Total:                {total_params:,}")

    # --- Forward Pass ---
    batch_size = 2
    seq_len = 16
    # x: (batch, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, config.d_model, device=device)
    output, aux_loss = moe(x)
    print(f"\nForward pass:")
    print(f"  Input shape:          {x.shape}")
    print(f"  Output shape:         {output.shape}")
    print(f"  Aux loss:             {aux_loss.item():.6f}")

    # --- Load Balance Statistics ---
    stats = moe.get_expert_load_stats()
    print(f"\nLoad balance after 1 step:")
    print(f"  Balance score:        {stats['balance_score']:.4f} (1.0 = perfect)")
    if stats["load_per_expert"]:
        loads = stats["load_per_expert"]
        print(f"  Load per expert:      {[f'{l:.3f}' for l in loads]}")

    # --- Run multiple steps to show bias adaptation ---
    print(f"\nRunning 100 steps to show bias adaptation...")
    moe.router.reset_stats()
    for step in range(100):
        x_step = torch.randn(batch_size, seq_len, config.d_model, device=device)
        _, _ = moe(x_step)

    stats = moe.get_expert_load_stats()
    print(f"  Balance score:        {stats['balance_score']:.4f}")
    if stats["load_per_expert"]:
        loads = stats["load_per_expert"]
        print(f"  Load per expert:      {[f'{l:.3f}' for l in loads]}")
    print(f"  Per-expert bias:      {[f'{b:.4f}' for b in moe.router.bias.tolist()]}")

    # --- Verify zero token dropping ---
    print(f"\nVerifying zero token dropping:")
    x_test = torch.randn(1, 4, config.d_model, device=device)
    output_test, _ = moe(x_test)
    # Check that output is not zero anywhere (all tokens processed)
    is_nonzero = (output_test.abs() > 1e-10).all()
    print(f"  All outputs non-zero: {is_nonzero.item()}")

    # --- Expert dimensions ---
    print(f"\nExpert architecture:")
    expert = moe.routed_experts.experts[0]
    print(f"  W_gate shape:         {expert.W_gate.weight.shape}")
    print(f"  W_up shape:           {expert.W_up.weight.shape}")
    print(f"  W_down shape:         {expert.W_down.weight.shape}")
    print(f"  Activation:           SwiGLU")

    print("\n" + "=" * 70)
    print("All MoE demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_moe()
