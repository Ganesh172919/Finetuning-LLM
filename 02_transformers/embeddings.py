"""
################################################################################
EMBEDDINGS — CONVERTING TOKENS TO VECTORS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Embeddings?
    Embeddings convert discrete tokens (words, subwords) into continuous
    vectors that neural networks can process.

    The cat sat → [0.2, -0.5, 0.8] [0.1, 0.3, -0.2] [0.7, -0.1, 0.4]

Why do we need them?
    Neural networks work with numbers, not words.
    Embeddings encode semantic meaning into vectors:
    - Similar words → similar vectors
    - king - man + woman ≈ queen (famous example)

How do they work?
    1. Token Embedding: Lookup table mapping token IDs to vectors
    2. Position Embedding: Add information about position in sequence
    3. The combination tells the model WHAT each token is and WHERE it is

Position Embedding Evolution:
    - 2017: Sinusoidal (original Transformer)
    - 2018: Learned position embeddings (BERT, GPT-2)
    - 2021: RoPE (Rotary Position Embeddings) — Su et al.
    - 2022: ALiBi (Attention with Linear Biases) — Press et al.
    - 2024+: RoPE dominates (LLaMA, Mistral, GPT-4)

########################################
"""

import numpy as np
from typing import Optional
import math

################################################################################
# SECTION 1: TOKEN EMBEDDING
################################################################################

class TokenEmbedding:
    """
    Token Embedding Layer
    =====================

    Definition: A lookup table that maps token IDs to dense vectors.

    How it works:
    - Vocabulary size V (e.g., 32000 for LLaMA, 100000 for GPT-4)
    - Embedding dimension d (e.g., 768, 1024, 4096)
    - Weight matrix W ∈ R^{V × d}
    - For token i: embedding = W[i]

    This is equivalent to one-hot encoding @ W, but much faster.

    Example:
        vocab_size = 5, embed_dim = 3
        W = [[0.1, 0.2, 0.3],   # embedding for "the"
             [0.4, 0.5, 0.6],   # embedding for "cat"
             [0.7, 0.8, 0.9],   # embedding for "sat"
             [1.0, 1.1, 1.2],   # embedding for "on"
             [1.3, 1.4, 1.5]]   # embedding for "mat"

        Token "cat" (id=1) → [0.4, 0.5, 0.6]

    Why learn embeddings?
    - Random initialization: vectors start random
    - Training: backpropagation updates vectors
    - Result: similar tokens get similar vectors
    - "king" and "queen" end up close in embedding space

    Interview Questions:
        1. "What's the difference between word2vec and transformer embeddings?"
           Word2vec: fixed embeddings (one vector per word)
           Transformer: contextual embeddings (vector depends on context)
           "bank" has different embeddings in "river bank" vs "bank account"

        2. "How large should the embedding dimension be?"
           Tradeoff: larger = more expressive, more parameters, slower
           Common: 768 (BERT-base), 1024 (GPT-2), 4096 (LLaMA-7B)
           Scaling law: d ≈ 0.02 × (parameters)^0.5

        3. "Should input and output embeddings be shared?"
           Sharing reduces parameters and often improves performance.
           GPT-2, LLaMA share them. BERT does not.
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize token embedding.

        Args:
            vocab_size: Number of tokens in vocabulary
            d_model: Embedding dimension
        """
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Xavier initialization: scale by sqrt(2/(fan_in + fan_out))
        # This keeps gradients well-behaved during training
        scale = math.sqrt(2.0 / (vocab_size + d_model))
        self.weight = np.random.randn(vocab_size, d_model) * scale

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Look up embeddings for token IDs.

        Args:
            token_ids: Integer token IDs [batch × seq_len]

        Returns:
            embeddings: [batch × seq_len × d_model]

        This is just indexing: for each token ID, grab the corresponding row.
        """
        return self.weight[token_ids]

    def backward(self, grad_output: np.ndarray, token_ids: np.ndarray) -> None:
        """
        Backward pass: accumulate gradients for used embeddings.

        Only the embeddings that were actually looked up receive gradients.
        This is efficient — we don't compute gradients for unused tokens.
        """
        # Gradient for embedding weight
        self.grad_weight = np.zeros_like(self.weight)
        for i in range(token_ids.shape[0]):
            for j in range(token_ids.shape[1]):
                self.grad_weight[token_ids[i, j]] += grad_output[i, j]


################################################################################
# SECTION 2: ROTARY POSITION EMBEDDINGS (RoPE)
################################################################################

class RotaryPositionEmbedding:
    """
    Rotary Position Embeddings (RoPE)
    ===================================

    Definition: Encode position by ROTATING vectors in 2D subspaces.
    Each pair of dimensions is rotated by an angle proportional to position.

    WHY RoPE?
    =========
    Problem: How to encode position so the model knows token order?

    Previous approaches:
    - Sinusoidal: Add fixed sin/cos patterns (Transformer, 2017)
    - Learned: Learn position embeddings (BERT, GPT-2)
    - Relative: Encode relative distances (T5)

    RoPE's insight: Instead of ADDING position info, ROTATE the vectors.

    Key property:
    When computing attention between positions m and n:
    q_m · k_n depends on (m - n), i.e., the RELATIVE position!

    This means:
    1. The model naturally captures relative positions
    2. No explicit relative position encoding needed
    3. Extrapolates to longer sequences (with NTK-aware scaling)

    How it works:
    =============
    For a vector [x1, x2, x3, x4, x5, x6, ...]:
    - Pair (x1, x2): rotate by angle θ₁ × position
    - Pair (x3, x4): rotate by angle θ₂ × position
    - Pair (x5, x6): rotate by angle θ₃ × position

    Where θ_i = 1 / (10000^(2i/d))

    The rotation preserves the dot product property:
    R(pos_m)q · R(pos_n)k = q · R(pos_n - pos_m)k

    This is beautiful because the dot product naturally depends
    on relative position!

    Formula:
        RoPE(x, pos) = R(pos) @ x
        where R(pos) is a block-diagonal rotation matrix

    For 2D case:
        R(θ) = [[cos(θ), -sin(θ)],
                [sin(θ),  cos(θ)]]

    Visual:
        Position 0: no rotation
        Position 1: rotate by θ
        Position 2: rotate by 2θ
        ...

    Used by:
    - LLaMA (all sizes)
    - Mistral
    - Qwen
    - DeepSeek
    - Most modern LLMs

    Interview Questions:
        1. "Why is RoPE better than learned position embeddings?"
           RoPE naturally encodes relative positions and extrapolates
           to longer sequences. Learned embeddings are fixed to training length.

        2. "How does RoPE help with long context?"
           With NTK-aware interpolation, RoPE can extend to sequences
           longer than seen during training (e.g., 4K → 128K tokens).

        3. "What's the computational cost of RoPE?"
           Very cheap: just element-wise rotations. No additional parameters.
    """

    def __init__(self, d_model: int, base: float = 10000.0, max_seq_len: int = 8192):
        """
        Initialize RoPE.

        Args:
            d_model: Model dimension (must be even)
            base: Base for frequency computation (10000 is standard)
            max_seq_len: Maximum sequence length for precomputation
        """
        assert d_model % 2 == 0, "d_model must be even for RoPE"

        self.d_model = d_model
        self.base = base
        self.max_seq_len = max_seq_len

        # Compute frequency for each dimension pair
        # θ_i = 1 / (base^(2i/d))
        # Lower dimensions → higher frequency (rotate fast)
        # Higher dimensions → lower frequency (rotate slow)
        inv_freq = 1.0 / (base ** (np.arange(0, d_model, 2) / d_model))
        self.inv_freq = inv_freq  # [d_model/2]

        # Precompute sin and cos for all positions
        self._precompute(max_seq_len)

    def _precompute(self, seq_len: int):
        """
        Precompute sin and cos values for efficiency.

        Instead of computing sin/cos every time, we precompute
        for all positions up to max_seq_len.
        """
        positions = np.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]

        # angles[pos, i] = pos * θ_i
        angles = np.outer(positions, self.inv_freq)  # [seq_len × d_model/2]

        # Cache sin and cos
        self.cos_cache = np.cos(angles)  # [seq_len × d_model/2]
        self.sin_cache = np.sin(angles)  # [seq_len × d_model/2]

    def forward(self, x: np.ndarray, position_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply rotary position embeddings.

        Args:
            x: Input tensor [batch × seq_len × d_model]
            position_ids: Optional position indices [batch × seq_len]

        Returns:
            Rotated tensor [batch × seq_len × d_model]

        How rotation works:
        For each pair (x_2i, x_2i+1):
            x'_2i   = x_2i * cos(θ) - x_2i+1 * sin(θ)
            x'_2i+1 = x_2i * sin(θ) + x_2i+1 * cos(θ)
        """
        batch, seq_len, d = x.shape

        # Get cos and sin for these positions
        if position_ids is not None:
            cos = self.cos_cache[position_ids]  # [batch × seq × d/2]
            sin = self.sin_cache[position_ids]
        else:
            cos = self.cos_cache[:seq_len]  # [seq × d/2]
            sin = self.sin_cache[:seq_len]

        # Reshape for broadcasting
        cos = cos[np.newaxis, :, :]  # [1 × seq × d/2]
        sin = sin[np.newaxis, :, :]

        # Split x into pairs
        x1 = x[:, :, 0::2]  # Even indices: [batch × seq × d/2]
        x2 = x[:, :, 1::2]  # Odd indices: [batch × seq × d/2]

        # Apply rotation
        # x'_even = x_even * cos - x_odd * sin
        # x'_odd  = x_even * sin + x_odd * cos
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        # Interleave back
        result = np.empty_like(x)
        result[:, :, 0::2] = x1_rot
        result[:, :, 1::2] = x2_rot

        return result


################################################################################
# SECTION 3: ALiBi (Attention with Linear Biases)
################################################################################

class ALiBi:
    """
    ALiBi: Attention with Linear Biases
    =====================================

    Definition: Add a linear bias to attention scores based on
    the distance between query and key positions.

    Formula:
        attention_score(q_i, k_j) = q_i · k_j - m * |i - j|

    Where m is a head-specific slope.

    WHY ALiBi?
    ===========
    RoPE requires precomputation and can struggle with extrapolation.
    ALiBi is simpler: just subtract a penalty proportional to distance.

    Properties:
    1. No learned parameters (just fixed slopes)
    2. Extrapolates to longer sequences naturally
    3. Each head has different slope → different attention patterns
    4. Nearby tokens get less penalty → attend more

    Slope assignment:
        For h heads, slopes are: 2^(-8/h), 2^(-16/h), ..., 2^(-8)
        This gives a geometric progression of attention ranges.

    Visual (4 heads):
        Position:  0  1  2  3  4  5
        Head 1:   [0 -1 -2 -3 -4 -5]  (sharp decay)
        Head 2:   [0 -0.5 -1 -1.5 -2 -2.5]  (moderate)
        Head 3:   [0 -0.25 -0.5 -0.75 -1 -1.25]  (gentle)
        Head 4:   [0 -0.125 -0.25 -0.375 -0.5 -0.625]  (very gentle)

    Used by:
    - BLOOM (176B)
    - MPT
    - Some code generation models

    Interview Question:
        "What's the difference between RoPE and ALiBi?"
        RoPE rotates vectors → dot product captures relative position.
        ALiBi adds linear bias → penalizes distant tokens directly.
        RoPE is more popular now due to better extrapolation with scaling.
        ALiBi is simpler but may not capture position as precisely.
    """

    def __init__(self, n_heads: int, max_seq_len: int = 8192):
        """
        Initialize ALiBi.

        Args:
            n_heads: Number of attention heads
            max_seq_len: Maximum sequence length
        """
        self.n_heads = n_heads

        # Compute slopes for each head
        # Slopes: 2^(-8/h), 2^(-16/h), ..., 2^(-8)
        slopes = 2 ** (-8 * np.arange(1, n_heads + 1) / n_heads)
        self.slopes = slopes  # [n_heads]

        # Precompute bias matrix
        self._precompute(max_seq_len)

    def _precompute(self, seq_len: int):
        """
        Precompute the linear bias matrix.

        bias[i][j] = -slope * |i - j|
        """
        positions = np.arange(seq_len)
        # Distance matrix: |i - j|
        distance = np.abs(positions[:, np.newaxis] - positions[np.newaxis, :])  # [seq × seq]

        # bias[head, i, j] = -slope[head] * distance[i, j]
        self.bias = -self.slopes[:, np.newaxis, np.newaxis] * distance[np.newaxis, :, :]
        # Shape: [n_heads × seq × seq]

    def get_bias(self, seq_len: int) -> np.ndarray:
        """
        Get ALiBi bias for given sequence length.

        Returns: [n_heads × seq_len × seq_len]
        """
        return self.bias[:, :seq_len, :seq_len]


################################################################################
# SECTION 4: TESTING & EXAMPLES
################################################################################

def demonstrate_embeddings():
    """Demonstrate embedding mechanisms."""
    print("=" * 70)
    print("EMBEDDINGS DEMONSTRATION")
    print("=" * 70)

    # Token Embedding
    print("\n--- Token Embedding ---")
    vocab_size = 1000
    d_model = 64
    embedding = TokenEmbedding(vocab_size, d_model)

    token_ids = np.array([[1, 5, 42, 100]])
    embeddings = embedding.forward(token_ids)
    print(f"Token IDs: {token_ids}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First token embedding (first 5 dims): {embeddings[0, 0, :5].round(3)}")

    # RoPE
    print("\n--- Rotary Position Embeddings ---")
    rope = RotaryPositionEmbedding(d_model=64, max_seq_len=128)

    x = np.random.randn(1, 4, 64)
    x_rotated = rope.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Rotated shape: {x_rotated.shape}")
    print(f"Input[0,0,:5]: {x[0, 0, :5].round(3)}")
    print(f"Rotated[0,0,:5]: {x_rotated[0, 0, :5].round(3)}")

    # Verify rotation preserves norm
    input_norm = np.linalg.norm(x[0, 0])
    output_norm = np.linalg.norm(x_rotated[0, 0])
    print(f"Input norm: {input_norm:.4f}")
    print(f"Output norm: {output_norm:.4f}")
    print(f"Norm preserved: {abs(input_norm - output_norm) < 1e-6}")

    # ALiBi
    print("\n--- ALiBi ---")
    alibi = ALiBi(n_heads=8, max_seq_len=128)
    bias = alibi.get_bias(seq_len=4)
    print(f"Bias shape: {bias.shape}")
    print(f"Bias for head 0:\n{bias[0]}")
    print(f"Bias for head 7:\n{bias[7]}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_embeddings()


################################################################################
# REFERENCES
################################################################################

# [1] Vaswani, A., et al. (2017). Attention Is All You Need.
# [2] Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Embedding.
# [3] Press, O., et al. (2022). Train Short, Test Long: Attention with Linear Biases.
# [4] Chen, S., et al. (2023). Extending Context Window of Large Language Models via RoPE.
# [5] bloc97. (2023). NTK-Aware Scaled RoPE.

################################################################################
