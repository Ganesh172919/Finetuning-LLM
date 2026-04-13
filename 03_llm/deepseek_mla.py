"""
################################################################################
DEEPSEEK MULTI-HEAD LATENT ATTENTION (MLA)
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Multi-head Latent Attention?
    MLA is DeepSeek's innovative attention mechanism that compresses
    the KV cache into a low-dimensional latent space, dramatically
    reducing memory usage while maintaining quality.

Key Innovation:
    Instead of storing full K and V tensors (which are large),
    MLA stores a compressed latent vector and reconstructs K, V on the fly.

    Standard MHA: Store K [n_kv_heads × d_k], V [n_kv_heads × d_v] per token
    MLA: Store latent c [d_latent] per token, where d_latent << n_kv_heads × d_k

Memory Savings:
    For DeepSeek-V2:
    - Standard KV cache: 32768 tokens × 128 heads × 128 dim × 2 = 1GB
    - MLA KV cache: 32768 tokens × 512 latent dim = 16MB (64x smaller!)

How it works:
    1. Compress: c = W_compress @ [K; V]  (or directly from input)
    2. Store: Only store c (small latent vector)
    3. Reconstruct: K = W_k @ c, V = W_v @ c (on the fly)
    4. Attention: Standard attention with reconstructed K, V

This is similar to LoRA's low-rank idea applied to KV cache.

Interview Questions:
        Q: "What is DeepSeek's MLA?"
        A: Multi-head Latent Attention compresses KV cache into a
           low-dimensional latent space. Instead of storing full K and V,
           store a compressed vector and reconstruct on the fly.

        Q: "How does MLA compare to GQA?"
        A: GQA reduces KV heads. MLA compresses the representation itself.
           MLA achieves more compression with less quality loss.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple
import math

################################################################################
# SECTION 1: MULTI-HEAD LATENT ATTENTION
################################################################################

class MultiHeadLatentAttention:
    """
    Multi-head Latent Attention (MLA)
    ==================================

    Compresses KV cache into latent space for efficient inference.

    Architecture:
        Input x
          ├──→ W_Q → Q (query, full size)
          └──→ W_compress → c_latent (compressed KV representation)
                              ├──→ W_K_reconstruct → K
                              └──→ W_V_reconstruct → V

        Attention: softmax(Q @ K^T / sqrt(d_k)) @ V

    Key Insight:
        The KV representation has redundancy. By learning a compressed
        latent space, we can store much less information per token
        while reconstructing high-quality K and V.

    Interview Questions:
        Q: "Why does MLA work?"
        A: KV representations have redundancy. The information in
           K and V can be compressed into a lower-dimensional space.
           MLA learns this compression end-to-end.

        Q: "What's the computational overhead of MLA?"
        A: Small overhead for compression/decompression during training.
           During inference, the memory savings far outweigh the cost.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_latent: int,
        d_rope: int = 64
    ):
        """
        Initialize MLA.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_latent: Latent dimension (compressed KV size)
            d_rope: RoPE dimension (not compressed)
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_latent = d_latent
        self.d_rope = d_rope

        # Query projection
        self.W_Q = np.random.randn(d_model, n_heads * self.d_k) * 0.02

        # KV compression: project input to latent space
        self.W_compress = np.random.randn(d_model, d_latent) * 0.02

        # KV decompression: reconstruct K, V from latent
        self.W_K = np.random.randn(d_latent, n_heads * self.d_k) * 0.02
        self.W_V = np.random.randn(d_latent, n_heads * self.d_k) * 0.02

        # RoPE projections (not compressed)
        self.W_Q_rope = np.random.randn(d_model, n_heads * d_rope) * 0.02
        self.W_K_rope = np.random.randn(d_model, d_rope) * 0.02

        # Output projection
        self.W_O = np.random.randn(n_heads * self.d_k, d_model) * 0.02

    def forward(
        self,
        x: np.ndarray,
        kv_cache: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of MLA.

        Args:
            x: Input [batch × seq × d_model]
            kv_cache: Cached latent [batch × cached_seq × d_latent]

        Returns:
            output: [batch × seq × d_model]
            new_kv_cache: Updated latent cache
        """
        batch, seq, _ = x.shape

        # 1. Compute query
        Q = np.matmul(x, self.W_Q)  # [batch × seq × n_heads × d_k]

        # 2. Compress to latent space
        c_latent = np.matmul(x, self.W_compress)  # [batch × seq × d_latent]

        # 3. Update KV cache
        if kv_cache is not None:
            c_full = np.concatenate([kv_cache, c_latent], axis=1)
        else:
            c_full = c_latent

        # 4. Reconstruct K, V from latent
        K = np.matmul(c_full, self.W_K)  # [batch × full_seq × n_heads × d_k]
        V = np.matmul(c_full, self.W_V)

        # 5. Reshape for attention
        Q = Q.reshape(batch, seq, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # 6. Attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        out = np.matmul(weights, V)

        # 7. Merge heads and project
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, -1)
        output = np.matmul(out, self.W_O)

        return output, c_full


################################################################################
# SECTION 2: DEEPSEEK MoE LAYER
################################################################################

class DeepSeekMoE:
    """
    DeepSeek Mixture of Experts
    ============================

    DeepSeek uses a unique MoE design with:
    - Shared experts: always active, capture common knowledge
    - Routed experts: activated per token, capture specialized knowledge

    Architecture:
        Input x
          ├──→ Shared Expert 1 (always active)
          ├──→ Shared Expert 2 (always active)
          └──→ Router → Top-K experts from N routed experts

        output = Σ shared_expert(x) + Σ routed_weight_i * expert_i(x)

    This ensures basic capability (shared) while allowing specialization (routed).

    Interview Question:
        Q: "How does DeepSeek's MoE differ from Mixtral?"
        A: DeepSeek has shared experts that are always active,
           providing a baseline capability. Mixtral only has routed experts.
    """

    def __init__(
        self,
        d_model: int,
        n_shared_experts: int = 2,
        n_routed_experts: int = 64,
        top_k: int = 6,
        d_ff: int = None
    ):
        self.d_model = d_model
        self.n_shared = n_shared_experts
        self.n_routed = n_routed_experts
        self.top_k = top_k
        self.d_ff = d_ff or int(2 / 3 * 4 * d_model)

        # Shared experts (always active)
        self.shared_experts = []
        for _ in range(n_shared_experts):
            self.shared_experts.append({
                'W_gate': np.random.randn(d_model, self.d_ff) * 0.02,
                'W_up': np.random.randn(d_model, self.d_ff) * 0.02,
                'W_down': np.random.randn(self.d_ff, d_model) * 0.02,
            })

        # Routed experts
        self.routed_experts = []
        for _ in range(n_routed_experts):
            self.routed_experts.append({
                'W_gate': np.random.randn(d_model, self.d_ff) * 0.02,
                'W_up': np.random.randn(d_model, self.d_ff) * 0.02,
                'W_down': np.random.randn(self.d_ff, d_model) * 0.02,
            })

        # Router
        self.W_router = np.random.randn(d_model, n_routed_experts) * 0.02

    def _expert_forward(self, x: np.ndarray, expert: dict) -> np.ndarray:
        """Forward through a single expert (SwiGLU)."""
        gate = np.matmul(x, expert['W_gate'])
        gate = gate / (1 + np.exp(-gate))  # Swish
        up = np.matmul(x, expert['W_up'])
        return np.matmul(gate * up, expert['W_down'])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: [batch × seq × d_model]

        Returns:
            output: [batch × seq × d_model]
        """
        batch, seq, d = x.shape

        # 1. Shared experts (always active)
        shared_out = np.zeros_like(x)
        for expert in self.shared_experts:
            shared_out = shared_out + self._expert_forward(x, expert)

        # 2. Router: compute scores for each expert
        router_logits = np.matmul(x, self.W_router)  # [batch × seq × n_routed]

        # 3. Select top-K experts
        top_k_indices = np.argsort(router_logits, axis=-1)[..., -self.top_k:]
        top_k_scores = np.take_along_axis(router_logits, top_k_indices, axis=-1)
        top_k_weights = np.exp(top_k_scores - np.max(top_k_scores, axis=-1, keepdims=True))
        top_k_weights = top_k_weights / np.sum(top_k_weights, axis=-1, keepdims=True)

        # 4. Routed experts
        routed_out = np.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[..., k]  # [batch × seq]
            weight = top_k_weights[..., k:k+1]  # [batch × seq × 1]

            # Process through selected expert
            # (Simplified: process all tokens through each selected expert)
            for i in range(self.n_routed):
                mask = (expert_idx == i)
                if np.any(mask):
                    expert_out = self._expert_forward(x, self.routed_experts[i])
                    routed_out = routed_out + weight * expert_out * mask[..., np.newaxis]

        # 5. Combine shared and routed
        return shared_out + routed_out


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_deepseek():
    """Demonstrate DeepSeek components."""
    print("=" * 70)
    print("DEEPSEEK MLA DEMONSTRATION")
    print("=" * 70)

    # MLA
    print("\n--- Multi-head Latent Attention ---")
    mla = MultiHeadLatentAttention(d_model=128, n_heads=8, d_latent=32)
    x = np.random.randn(1, 4, 128)
    out, cache = mla.forward(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"KV Cache: {cache.shape}")
    print(f"Cache reduction: {x.shape[-1]} → {cache.shape[-1]} = {x.shape[-1]/cache.shape[-1]:.1f}x")

    # DeepSeek MoE
    print("\n--- DeepSeek MoE ---")
    moe = DeepSeekMoE(d_model=64, n_shared_experts=2, n_routed_experts=8, top_k=2)
    x = np.random.randn(1, 4, 64)
    out = moe.forward(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_deepseek()


################################################################################
# REFERENCES
################################################################################

# [1] DeepSeek-AI. (2024). DeepSeek-V2: A Strong, Economical, and Efficient MoE LLM.
# [2] DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs.

################################################################################
