"""
################################################################################
FLASH ATTENTION — IO-AWARE ATTENTION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Flash Attention?
    Flash Attention computes exact attention while being 2-4x faster
    and using O(n) memory instead of O(n²). The key insight is to
    optimize for GPU memory hierarchy (SRAM vs HBM).

The Problem:
    Standard attention materializes the full n×n attention matrix.
    For sequence length n=4096:
    - Attention matrix: 4096×4096 = 16M elements
    - With batch=32, heads=32: 16M × 32 × 32 = 16B elements
    - At fp16: 32GB just for attention!

The Solution:
    Process attention in tiles that fit in GPU SRAM (fast memory).
    Use online softmax to compute attention without materializing
    the full matrix.

GPU Memory Hierarchy:
    SRAM (fast): ~20 MB, ~20 TB/s
    HBM (slow): ~80 GB, ~2 TB/s

    Standard attention: reads/writes to HBM many times
    Flash attention: reads/writes to HBM fewer times

Key Algorithm: Online Softmax
    Instead of computing softmax over the entire row at once:
    1. Process tiles of the row
    2. Maintain running max and sum
    3. Update output incrementally

    This is mathematically equivalent to standard attention!

Interview Questions:
        Q: "How does Flash Attention work?"
        A: It tiles the attention computation into blocks that fit in
           GPU SRAM. Instead of materializing the full n×n matrix,
           it computes attention block-by-block using online softmax.
           This reduces memory from O(n²) to O(n).

        Q: "Is Flash Attention exact or approximate?"
        A: EXACT! It computes the same result as standard attention.
           The optimization is purely in memory access patterns.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple
import math

################################################################################
# SECTION 1: ONLINE SOFTMAX
################################################################################

class OnlineSoftmax:
    """
    Online Softmax
    ==============

    Computes softmax incrementally, processing one element at a time.

    Standard softmax: need all elements to compute max and sum
    Online softmax: maintain running max and sum, update incrementally

    Algorithm:
    For each new element x_i:
        m_new = max(m_old, x_i)
        l_new = l_old * exp(m_old - m_new) + exp(x_i - m_new)
        m = m_new
        l = l_new

    Final: softmax(x_i) = exp(x_i - m) / l

    This is the key insight behind Flash Attention!
    """

    @staticmethod
    def online_softmax_max(x: np.ndarray) -> Tuple[float, float]:
        """
        Compute max and sum of exp(x) online.

        Args:
            x: 1D array

        Returns:
            max_val: Maximum value
            sum_exp: Sum of exp(x - max_val)
        """
        max_val = -np.inf
        sum_exp = 0.0

        for val in x:
            if val > max_val:
                sum_exp = sum_exp * math.exp(max_val - val) + 1.0
                max_val = val
            else:
                sum_exp += math.exp(val - max_val)

        return max_val, sum_exp


################################################################################
# SECTION 2: FLASH ATTENTION FORWARD
################################################################################

def flash_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size: int = 64
) -> np.ndarray:
    """
    Flash Attention forward pass (simplified).

    Instead of computing the full attention matrix:
    S = Q @ K^T  [n × n]

    We process in blocks:
    For each block of Q:
        For each block of K:
            Compute partial attention
            Update output using online softmax

    Args:
        Q: [n × d]
        K: [n × d]
        V: [n × d]
        block_size: Size of blocks for tiling

    Returns:
        output: [n × d]

    Interview Questions:
        Q: "Why is Flash Attention faster?"
        A: It reduces HBM reads/writes. Standard attention writes
           the full n×n matrix to HBM, then reads it back for softmax.
           Flash Attention computes everything in SRAM.
    """
    n, d = Q.shape

    # Initialize output and auxiliary variables
    O = np.zeros((n, d))
    l = np.zeros(n)  # sum of exp
    m = np.full(n, -np.inf)  # running max

    # Process in blocks
    for j in range(0, n, block_size):
        # Load K, V block
        j_end = min(j + block_size, n)
        K_block = K[j:j_end]
        V_block = V[j:j_end]

        for i in range(0, n, block_size):
            # Load Q block
            i_end = min(i + block_size, n)
            Q_block = Q[i:i_end]

            # Compute attention scores for this block
            S_block = Q_block @ K_block.T / math.sqrt(d)  # [block × block]

            # Online softmax update
            m_block = np.max(S_block, axis=-1)  # [block]
            m_new = np.maximum(m[i:i_end], m_block)

            # Rescale previous output
            scale_old = np.exp(m[i:i_end] - m_new)
            scale_new = np.exp(m_block - m_new)

            # Update sum
            l_new = l[i:i_end] * scale_old + np.sum(np.exp(S_block - m_new[:, np.newaxis]), axis=-1)

            # Update output
            O[i:i_end] = (O[i:i_end] * l[i:i_end, np.newaxis] * scale_old[:, np.newaxis] +
                          np.exp(S_block - m_new[:, np.newaxis]) @ V_block) / l_new[:, np.newaxis]

            # Update running values
            m[i:i_end] = m_new
            l[i:i_end] = l_new

    return O


################################################################################
# SECTION 3: FLASH ATTENTION 2 — IMPROVED TILING
################################################################################

def flash_attention_2_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size: int = 64
) -> np.ndarray:
    """
    Flash Attention 2 — Improved tiling strategy.

    Key improvements over Flash Attention 1:
    1. Better parallelism: loop over Q in outer loop (better GPU utilization)
    2. Reduced non-matmul FLOPs: fewer rescaling operations
    3. Work partitioning: split across warps for better occupancy

    Speedup: 2x over Flash Attention 1 on A100 GPUs

    Interview Question:
        Q: "What improvements does Flash Attention 2 make?"
        A: (1) Outer loop over Q instead of K — better GPU parallelism
           (2) Fewer rescaling operations — reduces non-matmul FLOPs
           (3) Better work partitioning across GPU warps
           Result: 2x speedup over Flash Attention 1.
    """
    n, d = Q.shape

    O = np.zeros((n, d))
    l = np.zeros(n)
    m = np.full(n, -np.inf)

    # Flash Attention 2: outer loop over Q (better parallelism)
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        Q_block = Q[i:i_end]

        # Initialize block-local variables
        O_block = np.zeros((i_end - i, d))
        l_block = np.zeros(i_end - i)
        m_block = np.full(i_end - i, -np.inf)

        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            K_block = K[j:j_end]
            V_block = V[j:j_end]

            # Compute scores
            S = Q_block @ K_block.T / math.sqrt(d)

            # Causal mask (if needed)
            if i < j:  # Q comes before K
                continue  # Skip (causal)

            # Online softmax
            m_new = np.maximum(m_block, np.max(S, axis=-1))
            scale = np.exp(m_block - m_new)
            exp_S = np.exp(S - m_new[:, np.newaxis])

            # Update
            l_new = l_block * scale + np.sum(exp_S, axis=-1)
            O_block = (O_block * l_block[:, np.newaxis] * scale[:, np.newaxis] +
                       exp_S @ V_block) / l_new[:, np.newaxis]

            m_block = m_new
            l_block = l_new

        O[i:i_end] = O_block

    return O


################################################################################
# SECTION 4: FLASH ATTENTION WITH CAUSAL MASK
################################################################################

def flash_attention_causal(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size: int = 64
) -> np.ndarray:
    """
    Flash Attention with causal mask.

    For autoregressive models, each position can only attend to
    previous positions. This is implemented by masking future positions.

    Interview Questions:
        Q: "How does Flash Attention handle causal masking?"
        A: When processing a block, check if any positions should be
           masked (future tokens). Set those attention scores to -inf.
    """
    n, d = Q.shape

    O = np.zeros((n, d))
    l = np.zeros(n)
    m = np.full(n, -np.inf)

    for j in range(0, n, block_size):
        j_end = min(j + block_size, n)
        K_block = K[j:j_end]
        V_block = V[j:j_end]

        for i in range(0, n, block_size):
            i_end = min(i + block_size, n)
            Q_block = Q[i:i_end]

            # Compute scores
            S_block = Q_block @ K_block.T / math.sqrt(d)

            # Apply causal mask
            for ii in range(i_end - i):
                for jj in range(j_end - j):
                    if j + jj > i + ii:  # Future position
                        S_block[ii, jj] = -np.inf

            # Online softmax (handle -inf)
            m_block = np.max(S_block, axis=-1)
            m_block = np.where(m_block == -np.inf, 0, m_block)

            m_new = np.maximum(m[i:i_end], m_block)

            # Rest of the computation...
            scale_old = np.exp(m[i:i_end] - m_new)
            l_new = l[i:i_end] * scale_old + np.sum(np.exp(S_block - m_new[:, np.newaxis]), axis=-1)

            O[i:i_end] = (O[i:i_end] * l[i:i_end, np.newaxis] * scale_old[:, np.newaxis] +
                          np.exp(S_block - m_new[:, np.newaxis]) @ V_block) / l_new[:, np.newaxis]

            m[i:i_end] = m_new
            l[i:i_end] = l_new

    return O


################################################################################
# SECTION 4: TESTING
################################################################################

def demonstrate_flash_attention():
    """Demonstrate Flash Attention."""
    print("=" * 70)
    print("FLASH ATTENTION DEMONSTRATION")
    print("=" * 70)

    # Online softmax
    print("\n--- Online Softmax ---")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    max_val, sum_exp = OnlineSoftmax.online_softmax_max(x)
    print(f"Input: {x}")
    print(f"Max: {max_val}")
    print(f"Sum exp: {sum_exp}")

    # Compare with standard softmax
    standard = np.exp(x - max_val) / sum_exp
    print(f"Softmax: {standard.round(4)}")

    # Flash attention
    print("\n--- Flash Attention ---")
    n, d = 16, 8
    Q = np.random.randn(n, d)
    K = np.random.randn(n, d)
    V = np.random.randn(n, d)

    # Standard attention
    scores = Q @ K.T / math.sqrt(d)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    standard_out = weights @ V

    # Flash attention
    flash_out = flash_attention_forward(Q, K, V, block_size=4)

    # Compare
    error = np.mean(np.abs(standard_out - flash_out))
    print(f"Standard output shape: {standard_out.shape}")
    print(f"Flash output shape: {flash_out.shape}")
    print(f"Mean absolute error: {error:.8f}")

    # Causal flash attention
    print("\n--- Causal Flash Attention ---")
    causal_out = flash_attention_causal(Q, K, V, block_size=4)
    print(f"Causal output shape: {causal_out.shape}")

    # Complexity comparison
    print("\n--- Complexity Comparison ---")
    for n in [128, 512, 2048, 8192]:
        standard_mem = n * n * 2  # bytes for fp16
        flash_mem = n * 64 * 2  # block size 64
        print(f"n={n}: Standard={standard_mem/1024:.0f}KB, Flash={flash_mem/1024:.0f}KB")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_flash_attention()


################################################################################
# REFERENCES
################################################################################

# [1] Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
# [2] Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.

################################################################################
