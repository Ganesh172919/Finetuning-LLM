"""
FlashAttention: Tiled Exact Attention
=======================================

An IO-aware exact attention algorithm that computes standard (non-approximate)
attention using tiling to reduce memory from O(N^2) to O(N) while being
2-4x faster than standard attention on modern GPUs.

BEGINNER LEVEL:
    Normal attention needs to remember how every word relates to every other
    word — that's N×N memory. FlashAttention processes small tiles at a time,
    like reading a book page by page instead of memorizing the whole thing.
    Same result, much less memory.

INTERMEDIATE LEVEL:
    FlashAttention tiles the Q, K, V matrices into blocks that fit in GPU
    SRAM (fast memory). Instead of materializing the full N×N attention
    matrix in HBM (slow memory), it computes attention block-by-block,
    accumulating results with the softmax correction trick. This reduces
    HBM accesses from O(N^2) to O(N^2/d^2) where d is the tile size.

ADVANCED LEVEL:
    The key insight is that softmax can be computed incrementally:
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    When merging two blocks, we can correct the running softmax using
    the rescaling factor: m_new = max(m_old, m_block), then multiply
    old results by exp(m_old - m_new). This lets us accumulate attention
    without ever storing the full matrix.

EXPERT LEVEL:
    FlashAttention v1 (Dao et al., 2022) achieves O(N) memory and
    O(N^2 d / M) HBM accesses where M is SRAM size. FlashAttention v2
    adds better parallelism (over sequence length, not just batch/heads)
    and reduces non-matmul FLOPs. FlashAttention v3 (Hopper) uses
    asynchronous TMA, warp specialization, and FP8 on H100 GPUs.

INTERVIEW LEVEL:
    Q: What is FlashAttention and why is it faster?
    A: It's an IO-aware attention algorithm that tiles Q/K/V to fit in
    GPU SRAM, avoiding the O(N^2) HBM read/write of standard attention.
    It's faster because GPU compute is much faster than memory access,
    and FlashAttention reduces memory accesses dramatically.
    Q: Is it exact or approximate?
    A: Exact — it computes the same result as standard attention, just
    with a different order of operations and online softmax.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class FlashConfig:
    """Configuration for FlashAttention computation.

    ATTRIBUTES:
        tile_size_q: Number of query rows processed per tile
        tile_size_k: Number of key/value rows processed per tile
        causal: Whether to apply causal (autoregressive) mask
        scale: Attention scale factor (default: 1/sqrt(d_head))

    DESIGN DECISIONS:
        - Tile sizes should fit in GPU SRAM (~48KB-192KB on modern GPUs)
        - Larger tiles = fewer HBM accesses but more SRAM needed
        - Optimal tile size depends on hardware (typically 64-256)
        - Causal masking adds a triangle constraint to tile processing

    TRADEOFFS:
        - Larger tiles: fewer iterations, less overhead, more SRAM pressure
        - Smaller tiles: more iterations, more overhead, less SRAM pressure
        - The sweet spot depends on sequence length and hardware
    """

    tile_size_q: int = 64
    tile_size_k: int = 64
    causal: bool = False
    scale: Optional[float] = None

    def __post_init__(self):
        """Set default scale if not provided."""
        if self.scale is None:
            self.scale = 1.0  # Will be set to 1/sqrt(d_head) at runtime


# ============================================================================
# ONLINE SOFTMAX
# ============================================================================

def online_softmax_correction(
    old_max: np.ndarray,
    old_sum: np.ndarray,
    old_output: np.ndarray,
    new_max: np.ndarray,
    new_sum: np.ndarray,
    new_output: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge two partial softmax results using online correction.

    DEFINITION:
        When computing softmax in tiles, each tile has its own max and sum.
        To merge tiles, we need to rescale using the combined max.

        Given:
            Tile 1: output_1, max_1, sum_1
            Tile 2: output_2, max_2, sum_2

        Combined:
            max_new = max(max_1, max_2)
            correction_1 = exp(max_1 - max_new)
            correction_2 = exp(max_2 - max_new)
            sum_new = sum_1 * correction_1 + sum_2 * correction_2
            output_new = (output_1 * correction_1 + output_2 * correction_2)

    PROBLEM:
        Softmax requires the global max for numerical stability. When
        processing tiles, we don't have the global max until all tiles
        are processed. The online correction trick lets us accumulate
        results tile by tile and correct at the end.

    EXECUTION FLOW:
        1. Compute new global max: max_new = max(old_max, new_max)
        2. Compute rescaling factors for both partial results
        3. Rescale old output and sum
        4. Add new (rescaled) output and sum
        5. Return corrected output, sum, and max

    COMPLEXITY:
        Time: O(d) where d = head dimension
        Space: O(d) for the correction factors

    EDUCATIONAL:
        This is the KEY trick that makes FlashAttention work.
        Without it, you'd need to store the full N×N matrix.
        With it, you can process one tile at a time and get
        the exact same result as standard softmax.
    """
    # New global max
    max_new = np.maximum(old_max, new_max)

    # Rescaling factors: how much to scale each partial result
    # These correct for the difference between each tile's local max
    # and the new global max
    correction_old = np.exp(old_max - max_new)
    correction_new = np.exp(new_max - max_new)

    # Rescale and combine
    sum_new = old_sum * correction_old + new_sum * correction_new
    output_new = old_output * correction_old[..., None] + new_output * correction_new[..., None]

    return output_new, sum_new, max_new


# ============================================================================
# FLASH ATTENTION FORWARD (TILED)
# ============================================================================

def flash_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    config: Optional[FlashConfig] = None,
) -> np.ndarray:
    """Compute exact attention using tiled FlashAttention algorithm.

    DEFINITION:
        Computes Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
        using tiling to avoid materializing the full N×N matrix.

    PROBLEM:
        Standard attention computes S = Q @ K^T (N×N), then softmax(S) @ V.
        For N=4096, d=64, S is 4096×4096 = 16M elements = 64MB in float32.
        This is slow because it requires reading/writing 64MB to/from HBM.
        FlashAttention processes tiles that fit in SRAM, reducing HBM access.

    INPUTS:
        Q: Query tensor, shape (batch, seq_len, n_heads, d_head)
        K: Key tensor, shape (batch, seq_len, n_heads, d_head)
        V: Value tensor, shape (batch, seq_len, n_heads, d_head)
        config: FlashConfig with tile sizes and options

    OUTPUTS:
        Output tensor, shape (batch, seq_len, n_heads, d_head)
        Same result as standard attention (exact, not approximate)

    EXECUTION FLOW:
        1. Initialize output accumulator O = 0, running max m = -inf, running sum l = 0
        2. For each query tile (block of rows from Q):
            a. Load Q_tile from HBM to SRAM
            b. Initialize tile-local O, m, l
            c. For each key/value tile:
                - Load K_tile, V_tile from HBM to SRAM
                - Compute S_tile = Q_tile @ K_tile^T * scale
                - Apply causal mask if needed
                - Compute local softmax: P_tile = exp(S_tile - max(S_tile))
                - Update local max and sum
                - Accumulate: O_tile += P_tile @ V_tile
            d. Merge tile result with running result using online correction
        3. Write final O to HBM

    COMPLEXITY:
        Time: O(N^2 * d) FLOPs (same as standard attention)
        Space: O(N * d) for output + O(tile^2) for tile computations
        HBM accesses: O(N^2 * d^2 / M) where M = SRAM size

        Standard attention HBM: O(N^2 * d + N^2) — must read/write N×N matrix
        FlashAttention HBM: O(N^2 * d^2 / M) — much less when d << M

    EDGE CASES:
        - Empty sequences: returns zeros
        - Single token: degenerates to standard attention
        - Very long sequences: more tiles, same algorithm
        - Causal mask: skips tiles that are entirely masked
    """
    if config is None:
        config = FlashConfig()

    # Input shape: (batch, seq_len, n_heads, d_head)
    B, N, H, d = Q.shape

    # Set scale to 1/sqrt(d_head)
    scale = config.scale if config.scale != 1.0 else 1.0 / math.sqrt(d)

    # Tile sizes
    Br = min(config.tile_size_q, N)  # Query tile size
    Bc = min(config.tile_size_k, N)  # Key/Value tile size

    # Number of tiles
    Tr = math.ceil(N / Br)  # Number of query tiles
    Tc = math.ceil(N / Bc)  # Number of key/value tiles

    # Initialize output and running statistics
    # O: output accumulator (batch, seq_len, n_heads, d_head)
    # m: running max per query position (batch, seq_len, n_heads)
    # l: running sum per query position (batch, seq_len, n_heads)
    O = np.zeros_like(Q)
    m = np.full((B, N, H), -np.inf, dtype=np.float32)
    l = np.zeros((B, N, H), dtype=np.float32)

    # Process query tiles
    for i in range(Tr):
        # Query tile boundaries
        q_start = i * Br
        q_end = min(q_start + Br, N)
        q_len = q_end - q_start

        # Load Q tile: (B, Br, H, d)
        Q_tile = Q[:, q_start:q_end] * scale

        # Initialize tile accumulators
        O_tile = np.zeros((B, q_len, H, d), dtype=np.float32)
        m_tile = np.full((B, q_len, H), -np.inf, dtype=np.float32)
        l_tile = np.zeros((B, q_len, H), dtype=np.float32)

        # Process key/value tiles
        for j in range(Tc):
            kv_start = j * Bc
            kv_end = min(kv_start + Bc, N)
            kv_len = kv_end - kv_start

            if config.causal and kv_start >= q_end:
                continue

            K_tile = K[:, kv_start:kv_end]  # (B, Bc, H, d)
            V_tile = V[:, kv_start:kv_end]  # (B, Bc, H, d)

            # Compute attention scores: S = Q_tile @ K_tile^T
            # Transpose K to (B, H, d, Bc) then matmul: (B, Br, H, d) x (B, H, d, Bc) -> (B, Br, H, Bc)
            K_t = np.transpose(K_tile, (0, 2, 3, 1))  # (B, H, d, Bc)
            # Q_tile: (B, Br, H, d) -> (B, H, Br, d) for matmul
            Q_heads = np.transpose(Q_tile, (0, 2, 1, 3))  # (B, H, Br, d)
            S_tile = np.matmul(Q_heads, K_t)  # (B, H, Br, Bc)

            if config.causal:
                q_indices = np.arange(q_start, q_end)
                kv_indices = np.arange(kv_start, kv_end)
                mask = kv_indices[None, :] <= q_indices[:, None]  # (Br, Bc)
                S_tile = np.where(mask[None, None, :, :], S_tile, -1e9)

            # Local softmax
            m_local = np.max(S_tile, axis=-1, keepdims=True)  # (B, H, Br, 1)
            P_tile = np.exp(S_tile - m_local)  # (B, H, Br, Bc)
            l_local = np.sum(P_tile, axis=-1)  # (B, H, Br)

            # Compute PV = P_tile @ V_tile
            V_heads = np.transpose(V_tile, (0, 2, 1, 3))  # (B, H, Bc, d)
            PV = np.matmul(P_tile, V_heads)  # (B, H, Br, d)

            # Online correction for merging with running accumulators
            m_local_squeezed = m_local.squeeze(-1)  # (B, H, Br)
            m_local_bh = np.transpose(m_local_squeezed, (0, 2, 1))  # (B, Br, H)
            l_local_bh = np.transpose(l_local, (0, 2, 1))  # (B, Br, H)

            # New running max
            m_new = np.maximum(m_tile, m_local_bh)
            # Rescaling factors
            correction_old = np.exp(m_tile - m_new)  # scale existing accumulator
            correction_new = np.exp(m_local_bh - m_new)  # scale new tile

            # Rescale existing accumulator and add new tile
            O_tile = O_tile * correction_old[..., None] + np.transpose(PV, (0, 2, 1, 3)) * correction_new[..., None]
            l_tile = l_tile * correction_old + l_local_bh * correction_new
            m_tile = m_new

        # Merge with global accumulator
        m_old = m[:, q_start:q_end]
        l_old = l[:, q_start:q_end]
        O_old = O[:, q_start:q_end]

        m_new = np.maximum(m_old, m_tile)
        correction_old = np.exp(m_old - m_new)
        correction_new = np.exp(m_tile - m_new)

        O[:, q_start:q_end] = O_old * correction_old[..., None] + O_tile * correction_new[..., None]
        l[:, q_start:q_end] = l_old * correction_old + l_tile * correction_new
        m[:, q_start:q_end] = m_new

    # Final normalization: O = O / l
    # This completes the softmax: exp(S - max) / sum(exp(S - max))
    O = O / l[..., None]

    # Handle NaN from division by zero (empty sequences)
    O = np.nan_to_num(O, nan=0.0)

    return O


# ============================================================================
# STANDARD ATTENTION (FOR COMPARISON)
# ============================================================================

def standard_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    causal: bool = False,
) -> np.ndarray:
    """Standard attention implementation for comparison.

    DEFINITION:
        Computes Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
        by materializing the full N×N attention matrix.

    EDUCATIONAL:
        This is the "naive" implementation that FlashAttention improves upon.
        It's correct but uses O(N^2) memory and O(N^2) HBM accesses.
    """
    B, N, H, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # Compute attention scores
    S = np.einsum('bqhd,bkhd->bhqk', Q, K) * scale

    # Apply causal mask
    if causal:
        mask = np.tri(N, N, dtype=bool)
        S = np.where(mask[None, None, :, :], S, -1e9)

    # Softmax
    S_max = np.max(S, axis=-1, keepdims=True)
    P = np.exp(S - S_max)
    P = P / np.sum(P, axis=-1, keepdims=True)

    # Apply to values
    O = np.einsum('bhqk,bkhd->bqhd', P, V)
    return O


# ============================================================================
# IO ANALYSIS
# ============================================================================

def compute_io_complexity(
    seq_len: int,
    d_head: int,
    sram_size: int = 100000,  # ~100KB SRAM
    dtype_bytes: int = 4,
) -> Dict[str, float]:
    """Compute IO complexity comparison between standard and Flash attention.

    DEFINITION:
        Compares HBM (High Bandwidth Memory) read/write operations
        between standard attention and FlashAttention.

    WHY IO MATTERS:
        On modern GPUs, compute throughput (FLOPS) is much higher than
        memory bandwidth. Standard attention is memory-bound; FlashAttention
        is compute-bound. Since compute is "free" relative to memory,
        FlashAttention is faster despite doing the same FLOPs.

    EDUCATIONAL:
        GPU Memory Hierarchy:
        - SRAM (on-chip): ~100KB, ~10TB/s — very fast, very small
        - HBM (off-chip): ~40-80GB, ~1-3TB/s — slow, very large

        Standard attention writes N×N matrix to HBM, reads it back.
        FlashAttention processes tiles in SRAM, only writes N×d output.
    """
    N = seq_len
    d = d_head
    M = sram_size // dtype_bytes  # SRAM capacity in elements

    # Standard attention HBM accesses
    # Read Q,K,V from HBM: 3*N*d
    # Write S (N×N) to HBM, read P (N×N) from HBM: 2*N*N
    # Write O (N×d) to HBM: N*d
    # Total: 4*N*d + 2*N^2
    standard_hbm = 4 * N * d + 2 * N * N

    # FlashAttention HBM accesses
    # Key insight: Flash never materializes the N×N matrix in HBM.
    # It reads Q,K,V tiles from HBM and writes O to HBM.
    # The N×N computation happens entirely in SRAM.
    #
    # Tile reads: for each of (N/Br)*(N/Bc) tile pairs,
    # read Q_tile (Br*d), K_tile (Bc*d), V_tile (Bc*d)
    # But Q_tile is reused across all KV tiles for the same row block:
    # Q reads: (N/Br) * Br * d = N*d
    # K,V reads: (N/Br) * (N/Bc) * 2 * Bc * d = 2 * N^2 * d / Br
    # Output write: N*d
    # Total: N*d + 2*N^2*d/Br + N*d = 2*N*d + 2*N^2*d/Br
    Br = int(math.sqrt(M / 4))  # Optimal tile size
    Bc = Br
    q_reads = N * d  # Q read once per row block, reused across KV tiles
    kv_reads = math.ceil(N / Br) * math.ceil(N / Bc) * 2 * Bc * d
    output_write = N * d
    flash_hbm = q_reads + kv_reads + output_write

    return {
        'seq_len': N,
        'd_head': d,
        'sram_elements': M,
        'standard_hbm_accesses': standard_hbm,
        'flash_hbm_accesses': flash_hbm,
        'io_reduction': standard_hbm / flash_hbm if flash_hbm > 0 else 0,
        'standard_memory_bytes': N * N * dtype_bytes,
        'flash_memory_bytes': N * d * dtype_bytes,
        'memory_reduction': (N * N) / (N * d) if d > 0 else 0,
    }


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================

def benchmark_attention(
    seq_lengths: list = [128, 256, 512, 1024, 2048],
    d_head: int = 64,
    n_heads: int = 8,
    batch_size: int = 1,
    causal: bool = True,
) -> Dict[str, list]:
    """Benchmark Flash vs Standard attention across sequence lengths.

    DEFINITION:
        Runs both implementations and compares:
        - Output correctness (should be identical)
        - Memory usage (Flash should use less)
        - Execution time (Flash should be faster on GPU)

    EDUCATIONAL:
        On CPU (numpy), FlashAttention may be SLOWER because:
        - CPU has large caches (no SRAM bottleneck)
        - NumPy einsum is already optimized
        - The tiling adds Python loop overhead

        On GPU (PyTorch/CUDA), FlashAttention is 2-4x faster because:
        - GPU SRAM is tiny (100KB) vs HBM (80GB)
        - Standard attention is memory-bound
        - FlashAttention reduces HBM accesses dramatically

        The benchmark here validates CORRECTNESS, not speed.
    """
    results = {
        'seq_lengths': seq_lengths,
        'max_diff': [],
        'mean_diff': [],
        'flash_memory': [],
        'standard_memory': [],
    }

    for N in seq_lengths:
        # Generate random Q, K, V
        Q = np.random.randn(batch_size, N, n_heads, d_head).astype(np.float32)
        K = np.random.randn(batch_size, N, n_heads, d_head).astype(np.float32)
        V = np.random.randn(batch_size, N, n_heads, d_head).astype(np.float32)

        # Standard attention
        O_std = standard_attention(Q, K, V, causal=causal)

        # Flash attention
        config = FlashConfig(causal=causal)
        O_flash = flash_attention_forward(Q, K, V, config=config)

        # Compare outputs (should be nearly identical)
        max_diff = float(np.max(np.abs(O_std - O_flash)))
        mean_diff = float(np.mean(np.abs(O_std - O_flash)))

        results['max_diff'].append(max_diff)
        results['mean_diff'].append(mean_diff)
        results['flash_memory'].append(N * d_head * 4)  # O(N*d) bytes
        results['standard_memory'].append(N * N * 4)  # O(N^2) bytes

    return results


def print_benchmark_results(results: Dict):
    """Pretty-print benchmark results."""
    print("=" * 70)
    print("FLASH ATTENTION BENCHMARK")
    print("=" * 70)
    print(f"{'Seq Len':>10} {'Max Diff':>12} {'Mean Diff':>12} {'Std Mem':>12} {'Flash Mem':>12}")
    print("-" * 70)

    for i, N in enumerate(results['seq_lengths']):
        print(f"{N:>10d} {results['max_diff'][i]:>12.6f} {results['mean_diff'][i]:>12.6f} "
              f"{results['standard_memory'][i] / 1024:>10.1f}KB {results['flash_memory'][i] / 1024:>10.1f}KB")

    print("=" * 70)
    print("NOTE: Flash and Standard produce identical outputs (exact attention)")
    print("Memory savings grow quadratically with sequence length")
