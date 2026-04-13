"""
################################################################################
ATTENTION MECHANISMS — THE HEART OF TRANSFORMERS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Attention?
    Attention is a mechanism that allows each element in a sequence to
    look at every other element and decide how much to "attend" to them.

    It's like reading a sentence: when you read "it" in "The cat sat on
    the mat because it was tired", your brain attends to "cat" to
    understand what "it" refers to.

Why does it matter?
    Before attention, RNNs processed sequences one token at a time,
    losing information over long distances. Attention allows:
    1. Direct connections between any two tokens
    2. Parallel processing of entire sequences
    3. Learning which tokens are related

How does it work?
    For each token, we compute three vectors:
    - Query (Q): "What am I looking for?"
    - Key (K): "What do I contain?"
    - Value (V): "What information do I provide?"

    Then: Attention(Q,K,V) = softmax(QK^T/√d)V

    This computes weighted sums of values, where weights are
    determined by query-key similarity.

########################################

ATTENTION TYPES IN THIS FILE:

1. Scaled Dot-Product Attention (base)
2. Multi-Head Attention (standard)
3. Multi-Query Attention (MQA, faster inference)
4. Grouped-Query Attention (GQA, LLaMA/Mistral)
5. Flash Attention (memory-efficient)
6. Causal Attention (autoregressive)
7. Cross Attention (multimodal)

################################################################################
"""

import numpy as np
from typing import Optional, Tuple
import math

################################################################################
# SECTION 1: SCALED DOT-PRODUCT ATTENTION
################################################################################

class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention
    =============================

    Definition: The fundamental attention mechanism.

    Formula:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Step by step:
        1. scores = Q @ K^T          — How similar are queries to keys?
        2. scaled = scores / √d_k    — Prevent large values
        3. masked = scaled + mask    — Optional: prevent attending to future
        4. weights = softmax(masked) — Convert to probabilities
        5. output = weights @ V      — Weighted sum of values

    WHY SCALING BY √d_k?
    =====================
    Without scaling, dot products grow with dimension d_k.
    If Q and K have entries ~ N(0,1), then Q·K has variance d_k.
    Large values → softmax saturates → gradients vanish.
    Dividing by √d_k keeps variance ≈ 1.

    Example:
        d_k = 64
        Q·K without scaling: variance ≈ 64
        Q·K with scaling: variance ≈ 1
        → Softmax works properly with scaling!

    Interview Questions:
        1. "Why do we need attention?"
           To capture long-range dependencies. RNNs lose information
           over distance; attention has direct connections.

        2. "What's the complexity of attention?"
           O(n²d) where n = sequence length, d = dimension.
           This is why long context is expensive.

        3. "Why Q, K, V instead of one matrix?"
           Different roles: Q asks questions, K provides answers,
           V provides information. This asymmetry is powerful.
    """

    def __init__(self, d_k: int):
        self.d_k = d_k
        self.scale = math.sqrt(d_k)

    def forward(
        self,
        Q: np.ndarray,  # [batch × seq_len × d_k]
        K: np.ndarray,  # [batch × seq_len × d_k]
        V: np.ndarray,  # [batch × seq_len × d_v]
        mask: Optional[np.ndarray] = None  # [batch × seq_len × seq_len]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of scaled dot-product attention.

        Returns:
            output: Weighted values [batch × seq_len × d_v]
            weights: Attention weights [batch × seq_len × seq_len]
        """
        # Step 1: Compute attention scores
        # Q @ K^T: how much should each query attend to each key?
        scores = np.matmul(Q, K.transpose(0, 2, 1))  # [batch × seq × seq]

        # Step 2: Scale to prevent large dot products
        scores = scores / self.scale

        # Step 3: Apply mask (for causal/decoder attention)
        if mask is not None:
            scores = scores + mask  # mask has -inf for masked positions

        # Step 4: Softmax to get attention weights
        # Numerically stable softmax
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(shifted)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Step 5: Weighted sum of values
        output = np.matmul(weights, V)  # [batch × seq × d_v]

        return output, weights


################################################################################
# SECTION 2: MULTI-HEAD ATTENTION
################################################################################

class MultiHeadAttention:
    """
    Multi-Head Attention
    =====================

    Definition: Run multiple attention operations in parallel,
    each with different learned projections.

    Formula:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
        where head_i = Attention(Q @ W_Qi, K @ W_Ki, V @ W_Vi)

    WHY MULTIPLE HEADS?
    ===================
    Single attention can only capture one type of relationship.
    Multiple heads can attend to:
    - Head 1: syntactic relationships (subject-verb)
    - Head 2: semantic relationships (pronouns-antecedents)
    - Head 3: positional relationships (nearby tokens)
    - Head 4: rare/important tokens

    Each head learns a different "view" of the sequence.

    Architecture:
    ┌─────────────────────────────────────────┐
    │ Input: X [batch × seq × d_model]        │
    │                                           │
    │ Split into h heads:                       │
    │   X → Q = X @ W_Q → [batch × seq × d_k] │
    │   X → K = X @ W_K → [batch × seq × d_k] │
    │   X → V = X @ W_V → [batch × seq × d_v] │
    │                                           │
    │ For each head i:                          │
    │   head_i = Attention(Q_i, K_i, V_i)      │
    │                                           │
    │ Concatenate all heads:                    │
    │   concat = Concat(head_1, ..., head_h)   │
    │                                           │
    │ Output projection:                        │
    │   output = concat @ W_O                   │
    └─────────────────────────────────────────┘

    Args:
        d_model: Model dimension (e.g., 768, 1024, 4096)
        n_heads: Number of attention heads (e.g., 8, 12, 32)
        dropout: Dropout probability (for training)

    Example:
        d_model = 512, n_heads = 8
        d_k = d_v = 512 / 8 = 64
        Each head operates on 64-dimensional vectors.

    Interview Questions:
        1. "How many attention heads should I use?"
           Typically d_model / 64. More heads = more diverse attention
           patterns, but each head has lower dimension.

        2. "What if I use only 1 head?"
           It still works, but you lose the ability to capture
           multiple types of relationships simultaneously.

        3. "Do all heads learn different things?"
           Research shows heads specialize: some attend locally,
           some attend globally, some attend to specific patterns.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        """
        Initialize multi-head attention.

        Args:
            d_model: Total model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear projections
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.dropout = dropout

        # Linear projections for Q, K, V, and output
        # These are the learned parameters
        scale = math.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

        if bias:
            self.b_Q = np.zeros(d_model)
            self.b_K = np.zeros(d_model)
            self.b_V = np.zeros(d_model)
            self.b_O = np.zeros(d_model)
        else:
            self.b_Q = self.b_K = self.b_V = self.b_O = None

        self.attention = ScaledDotProductAttention(self.d_k)

    def _linear(self, x: np.ndarray, W: np.ndarray, b: Optional[np.ndarray]) -> np.ndarray:
        """Apply linear transformation: y = x @ W + b"""
        result = np.matmul(x, W)
        if b is not None:
            result = result + b
        return result

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split the last dimension into (n_heads, d_k).

        Input: [batch × seq × d_model]
        Output: [batch × n_heads × seq × d_k]

        This allows each head to process independently.
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        x = x.transpose(0, 2, 1, 3)  # [batch × heads × seq × d_k]
        return x

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Merge heads back into single tensor.

        Input: [batch × n_heads × seq × d_k]
        Output: [batch × seq × d_model]
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)  # [batch × seq × heads × d_k]
        x = x.reshape(batch_size, seq_len, self.d_model)
        return x

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_weights: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor [batch × seq × d_model]
            mask: Optional attention mask [batch × seq × seq]
            return_weights: Whether to return attention weights

        Returns:
            output: [batch × seq × d_model]
            weights: [batch × n_heads × seq × seq] (if return_weights)
        """
        # Step 1: Linear projections
        Q = self._linear(x, self.W_Q, self.b_Q)  # [batch × seq × d_model]
        K = self._linear(x, self.W_K, self.b_K)
        V = self._linear(x, self.W_V, self.b_V)

        # Step 2: Split into multiple heads
        Q = self._split_heads(Q)  # [batch × n_heads × seq × d_k]
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Step 3: Apply attention to each head
        # We process all heads in parallel using batched matmul
        batch_size = Q.shape[0]

        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))  # [batch × heads × seq × seq]
        scores = scores / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask[:, np.newaxis, :, :]  # Broadcast mask

        # Softmax
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(shifted)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Apply attention to values
        attn_output = np.matmul(weights, V)  # [batch × heads × seq × d_k]

        # Step 4: Merge heads
        output = self._merge_heads(attn_output)  # [batch × seq × d_model]

        # Step 5: Output projection
        output = self._linear(output, self.W_O, self.b_O)

        if return_weights:
            return output, weights
        return output, None


################################################################################
# SECTION 3: GROUPED-QUERY ATTENTION (GQA)
################################################################################

class GroupedQueryAttention:
    """
    Grouped-Query Attention (GQA)
    ==============================

    Definition: A middle ground between Multi-Head Attention (MHA) and
    Multi-Query Attention (MQA). Groups of query heads share one KV head.

    Why GQA?
    =========
    Problem: MHA has h KV heads → large KV cache during inference
    Solution: MQA has 1 KV head → fast inference but quality loss
    GQA: g KV heads → balance between quality and speed

    Architecture:
        MHA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8    (8 heads)
              K1 K2 K3 K4 K5 K6 K7 K8    (8 KV heads)

        MQA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8    (8 heads)
              K1                          (1 KV head)

        GQA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8    (8 heads)
              K1 K1 K1 K1 K2 K2 K2 K2    (2 KV heads)

    Used by:
    - LLaMA 2/3 (8 KV heads for 70B model)
    - Mistral (8 KV heads)
    - Gemma

    Benefits:
    - 4x smaller KV cache than MHA (with 2 KV heads)
    - Almost same quality as MHA
    - Much faster inference than MHA

    Args:
        d_model: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key-value heads (must divide n_heads)

    Interview Question:
        "What's the difference between MHA, MQA, and GQA?"
        MHA: Each head has its own Q, K, V. Best quality, largest cache.
        MQA: All heads share one K, V. Smallest cache, slight quality loss.
        GQA: Groups of heads share K, V. Middle ground.
        Most modern models use GQA.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0
    ):
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # How many Q heads share each KV head
        self.d_k = d_model // n_heads

        # Q has n_heads projections, K/V have n_kv_heads projections
        scale = math.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, n_heads * self.d_k) * scale
        self.W_K = np.random.randn(d_model, n_kv_heads * self.d_k) * scale
        self.W_V = np.random.randn(d_model, n_kv_heads * self.d_k) * scale
        self.W_O = np.random.randn(n_heads * self.d_k, d_model) * scale

    def _repeat_kv(self, x: np.ndarray) -> np.ndarray:
        """
        Repeat KV heads to match query heads.

        If n_heads=8, n_kv_heads=2, then each KV head is repeated 4 times.

        Input: [batch × n_kv_heads × seq × d_k]
        Output: [batch × n_heads × seq × d_k]
        """
        if self.n_rep == 1:
            return x
        batch, n_kv, seq, d_k = x.shape
        x = np.repeat(x, self.n_rep, axis=1)
        return x

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass of grouped-query attention.

        Args:
            x: Input [batch × seq × d_model]
            mask: Optional attention mask

        Returns:
            output: [batch × seq × d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = np.matmul(x, self.W_Q).reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = np.matmul(x, self.W_K).reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = np.matmul(x, self.W_V).reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)

        # Transpose to [batch × heads × seq × d_k]
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Repeat KV heads to match query heads
        K = self._repeat_kv(K)
        V = self._repeat_kv(V)

        # Compute attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask[:, np.newaxis, :, :]

        # Softmax
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)

        # Weighted sum
        output = np.matmul(weights, V)  # [batch × heads × seq × d_k]

        # Merge heads
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Output projection
        output = np.matmul(output, self.W_O)

        return output


################################################################################
# SECTION 4: CAUSAL MASK
################################################################################

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create Causal (Autoregressive) Mask
    =====================================

    Definition: A mask that prevents tokens from attending to future tokens.

    Why it matters:
    In autoregressive generation (GPT, Claude), each token can only
    attend to previous tokens, not future ones. This mask ensures that.

    Visual:
        Token:    0  1  2  3  4
        Token 0: [✓  ✗  ✗  ✗  ✗]  — can only see itself
        Token 1: [✓  ✓  ✗  ✗  ✗]  — can see 0 and 1
        Token 2: [✓  ✓  ✓  ✗  ✗]  — can see 0, 1, 2
        Token 3: [✓  ✓  ✓  ✓  ✗]  — can see 0, 1, 2, 3
        Token 4: [✓  ✓  ✓  ✓  ✓]  — can see everything

    The mask uses -inf for positions to mask (will become 0 after softmax).

    Example:
        mask = create_causal_mask(4)
        # [[  0, -inf, -inf, -inf],
        #  [  0,    0, -inf, -inf],
        #  [  0,    0,    0, -inf],
        #  [  0,    0,    0,    0]]
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = np.where(mask == 1, -np.inf, 0.0)
    return mask


################################################################################
# SECTION 5: FLASH ATTENTION (CONCEPTUAL)
################################################################################

class FlashAttention:
    """
    Flash Attention
    ===============

    Definition: An IO-aware attention algorithm that is both faster AND
    more memory-efficient than standard attention.

    THE PROBLEM:
    ============
    Standard attention computes:
    1. S = Q @ K^T  — [n × n] matrix (must materialize!)
    2. P = softmax(S)  — [n × n] matrix
    3. O = P @ V  — [n × d] matrix

    For sequence length n=4096:
    - S and P are 4096×4096 = 16M elements
    - With batch and heads: 16M × 32 × 8 = 4B elements
    - At fp16: 8GB just for attention!

    THE SOLUTION (Flash Attention):
    ===============================
    Instead of materializing the full n×n matrix:
    1. Process attention in blocks/tiles
    2. Keep only running statistics (not full matrix)
    3. Use GPU SRAM (fast) instead of HBM (slow)
    4. Recompute in backward pass instead of storing

    Results:
    - 2-4x faster than standard attention
    - Memory: O(n) instead of O(n²)
    - Exact computation (not approximate!)

    Key Insight:
    The algorithm is equivalent to standard attention mathematically,
    but uses a clever tiling strategy to avoid materializing the
    full attention matrix.

    This was a breakthrough because:
    1. Longer context became practical
    2. Larger batch sizes became possible
    3. Training costs decreased significantly

    Used by: GPT-4, Claude, LLaMA, Mistral, and virtually all
    modern language models.

    Interview Question:
        "How does Flash Attention work?"
        It tiles the attention computation into blocks that fit in
        GPU SRAM. Instead of materializing the full n×n matrix,
        it computes attention block-by-block using online softmax
        (maintaining running max and sum). This reduces memory
        from O(n²) to O(n) while being faster due to better
        memory access patterns.

    Reference:
        Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient
        Exact Attention with IO-Awareness.
    """

    def __init__(self, d_model: int, n_heads: int, block_size: int = 256):
        """
        Initialize Flash Attention.

        Note: This is a simplified conceptual implementation.
        Real Flash Attention uses CUDA kernels.

        Args:
            d_model: Model dimension
            n_heads: Number of heads
            block_size: Tile size for processing
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.block_size = block_size

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Conceptual Flash Attention forward pass.

        Real implementation uses:
        1. Tiling into blocks of size block_size
        2. Online softmax (running max and sum)
        3. CUDA kernel for GPU SRAM access

        This simplified version shows the mathematical equivalence.
        """
        batch, heads, seq_len, d_k = Q.shape

        # Standard attention (simplified — real Flash uses tiling)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores + mask[:, np.newaxis, :, :]

        # Online softmax (simplified)
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)

        output = np.matmul(weights, V)
        return output


################################################################################
# SECTION 6: TESTING & EXAMPLES
################################################################################

def demonstrate_attention():
    """Demonstrate attention mechanisms."""
    print("=" * 70)
    print("ATTENTION MECHANISMS DEMONSTRATION")
    print("=" * 70)

    # Setup
    batch_size = 2
    seq_len = 4
    d_model = 16
    n_heads = 4

    # Random input
    x = np.random.randn(batch_size, seq_len, d_model)

    # Causal mask
    mask = create_causal_mask(seq_len)
    print(f"\nCausal mask:\n{mask}")

    # Multi-Head Attention
    print("\n--- Multi-Head Attention ---")
    mha = MultiHeadAttention(d_model, n_heads)
    output, weights = mha.forward(x, mask=mask, return_weights=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights (head 0, batch 0):\n{weights[0, 0].round(3)}")

    # Grouped-Query Attention
    print("\n--- Grouped-Query Attention ---")
    gqa = GroupedQueryAttention(d_model, n_heads=4, n_kv_heads=2)
    output_gqa = gqa.forward(x, mask=mask)
    print(f"Output shape: {output_gqa.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_attention()


################################################################################
# REFERENCES
################################################################################

# [1] Vaswani, A., et al. (2017). Attention Is All You Need.
# [2] Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need.
# [3] Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models.
# [4] Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention.
# [5] Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism.

################################################################################
