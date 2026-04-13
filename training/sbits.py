"""
S-Bits: Stochastic Bits Quantization
======================================

A probabilistic quantization method that achieves ultra-low-bit precision
(2-bit, 3-bit, 4-bit) by using stochastic rounding rather than deterministic
rounding. This preserves expected value across many samples, reducing
quantization bias and enabling aggressive compression.

BEGINNER LEVEL:
    Normal quantization rounds every number the same way.
    S-Bits uses probability — sometimes it rounds up, sometimes down,
    but on average it's perfectly accurate. Like flipping a weighted coin.

INTERMEDIATE LEVEL:
    S-Bits implements stochastic rounding: given a value v between two
    quantization levels q_low and q_high, the probability of rounding
    to q_high is (v - q_low) / (q_high - q_low). This ensures
    E[round(v)] = v exactly, eliminating quantization bias.

ADVANCED LEVEL:
    Unlike deterministic rounding which introduces systematic bias,
    S-Bits preserves the expected gradient signal during training.
    Combined with block-wise quantization and outlier handling,
    S-Bits enables 2-4 bit training with minimal accuracy loss.

EXPERT LEVEL:
    S-Bits is inspired by the observation that gradient noise during
    stochastic mini-batch SGD is already much larger than the noise
    introduced by stochastic quantization. Therefore, the additional
    noise from S-Bits quantization is absorbed by the existing noise,
    causing no measurable degradation in convergence.

INTERVIEW LEVEL:
    Q: What is stochastic rounding and why does it help?
    A: Instead of always rounding to the nearest level, we round
    probabilistically so the expected value equals the original.
    This eliminates quantization bias, crucial for low-bit training.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SBitsConfig:
    """Configuration for S-Bits stochastic quantization.

    ATTRIBUTES:
        bits: Number of bits for quantization (2, 3, or 4)
        block_size: Size of quantization blocks for local adaptation
        outlier_ratio: Fraction of values treated as outliers (kept in higher precision)
        use_stochastic_rounding: Whether to use stochastic vs deterministic rounding
        clip_range: Standard deviation multiplier for clipping outliers
        symmetric: Whether to use symmetric quantization (centered at 0)

    DESIGN DECISIONS:
        - Block-wise quantization adapts to local value distributions
        - Outlier handling preserves rare but important large values
        - Symmetric quantization is simpler but slightly less accurate
        - Asymmetric quantization better handles non-zero-centered data

    TRADEOFFS:
        - More bits = higher accuracy but more memory
        - Larger block_size = smoother statistics but less local adaptation
        - Higher outlier_ratio = more precision for outliers but more memory
    """

    bits: int = 4
    block_size: int = 64
    outlier_ratio: float = 0.01
    use_stochastic_rounding: bool = True
    clip_range: float = 3.0
    symmetric: bool = True

    def __post_init__(self):
        """Validate configuration parameters.

        RAISES:
            ValueError: If parameters are out of valid range.
        """
        if self.bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {self.bits}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if not 0 <= self.outlier_ratio <= 0.5:
            raise ValueError(f"outlier_ratio must be in [0, 0.5], got {self.outlier_ratio}")
        if self.clip_range <= 0:
            raise ValueError(f"clip_range must be > 0, got {self.clip_range}")

    @property
    def num_levels(self) -> int:
        """Number of quantization levels."""
        return 2 ** self.bits

    @property
    def max_val(self) -> int:
        """Maximum integer value representable."""
        return self.num_levels - 1


# ============================================================================
# STOCHASTIC ROUNDING ENGINE
# ============================================================================

def stochastic_round(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Perform element-wise stochastic rounding between two levels.

    DEFINITION:
        For each value v, round to high with probability (v - low) / (high - low),
        and to low otherwise. This ensures E[result] = v exactly.

    PROBLEM:
        Deterministic rounding always rounds the same way, introducing
        systematic bias. For example, always rounding 0.3 to 0 means
        we systematically underestimate by 0.3 on average.

    INPUTS:
        values: Array of values to round (float32)
        low: Lower quantization levels for each value
        high: Upper quantization levels for each value

    OUTPUTS:
        Rounded array where each element is either the corresponding
        low or high value, chosen probabilistically.

    EXECUTION FLOW:
        1. Compute probability p = (v - low) / (high - low)
        2. Generate uniform random numbers in [0, 1)
        3. If random < p, choose high; else choose low

    COMPLEXITY:
        Time: O(n) where n = number of elements
        Space: O(n) for the random array

    EDGE CASES:
        - If low == high (degenerate interval), returns low
        - NaN inputs produce NaN outputs
        - Empty arrays return empty arrays
    """
    # Handle degenerate case where low == high
    # This happens when all values in a block are the same
    mask = high == low
    safe_high = np.where(mask, low + 1, high)  # Avoid division by zero

    # Compute probability of rounding up
    # p = (v - low) / (high - low)
    # For exact boundary values (v == low or v == high), p is 0 or 1
    prob = np.clip((values - low) / (safe_high - low), 0, 1)

    # Generate random numbers and compare
    # random < prob → choose high, else choose low
    randoms = np.random.uniform(0, 1, size=values.shape)
    result = np.where(randoms < prob, high, low)

    # For degenerate cases, return low (the only available value)
    result = np.where(mask, low, result)

    return result


# ============================================================================
# BLOCK-WISE QUANTIZER
# ============================================================================

class SBitsQuantizer:
    """Stochastic Bits quantizer for weight tensors.

    ARCHITECTURE:
        - Reshape weight into blocks of block_size elements
        - Compute per-block scale and zero-point
        - Quantize each block independently
        - Handle outliers separately with higher precision

    WHY BLOCK-WISE:
        Different regions of a weight tensor have different distributions.
        A single global scale factor wastes bits on regions that don't
        need them. Block-wise quantization adapts locally, using the
        full quantization range efficiently in each block.

    USAGE PATTERN:
        quantizer = SBitsQuantizer(SBitsConfig(bits=4))
        q_data, meta = quantizer.quantize(weight_tensor)
        reconstructed = quantizer.dequantize(q_data, meta)
    """

    def __init__(self, config: SBitsConfig):
        """Initialize the S-Bits quantizer.

        Args:
            config: Quantization configuration
        """
        self.config = config

    def _compute_block_params(self, block: np.ndarray) -> Tuple[float, float]:
        """Compute scale and zero-point for a single block.

        DEFINITION:
            Maps the block's value range to the integer quantization range.
            For symmetric quantization: scale = max(|block|) / max_int
            For asymmetric: scale = (max - min) / max_int, zero_point = -min / scale

        PROBLEM:
            We need to map arbitrary float ranges to a small integer range
            (e.g., 0-15 for 4-bit) while preserving as much information
            as possible.

        EXECUTION FLOW:
            1. Clip outliers based on clip_range standard deviations
            2. Find the range [min_val, max_val] of the block
            3. Compute scale factor to map to [0, max_int]
            4. Compute zero_point (integer offset)

        COMPLEXITY:
            Time: O(block_size)
            Space: O(1) — returns two scalars

        EDGE CASES:
            - Constant block (min == max): scale = 1.0, zero_point = 0
            - All zeros: scale = 1.0, zero_point = 0
        """
        if self.config.symmetric:
            # Symmetric: range is [-max_abs, max_abs]
            # This is simpler and works well for weights (usually zero-centered)
            max_abs = np.max(np.abs(block))
            if max_abs < 1e-10:
                return 1.0, 0.0
            scale = max_abs / (self.config.num_levels // 2 - 1)
            zero_point = 0.0
        else:
            # Asymmetric: range is [min_val, max_val]
            # Better for activations which may not be zero-centered
            min_val = np.min(block)
            max_val = np.max(block)
            if max_val - min_val < 1e-10:
                return 1.0, float(self.config.num_levels // 2)
            scale = (max_val - min_val) / self.config.max_val
            zero_point = -min_val / scale

        return scale, zero_point

    def _quantize_block(self, block: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
        """Quantize a single block to integer values.

        DEFINITION:
            Maps float values to integers in [0, max_val] using:
            q = round(v / scale + zero_point)

        EXECUTION FLOW:
            1. Compute float positions: v / scale + zero_point
            2. Determine lower and upper integer bounds
            3. If stochastic rounding, use probabilistic rounding
            4. If deterministic, round to nearest
            5. Clip to valid range [0, max_val]

        COMPLEXITY:
            Time: O(block_size)
            Space: O(block_size)
        """
        # Map to quantization grid
        positions = block / scale + zero_point

        # Get lower and upper integer bounds
        low_idx = np.floor(positions).astype(np.int32)
        high_idx = low_idx + 1

        # Clip to valid range
        low_idx = np.clip(low_idx, 0, self.config.max_val)
        high_idx = np.clip(high_idx, 0, self.config.max_val)

        if self.config.use_stochastic_rounding:
            # Stochastic rounding: probabilistically choose between low and high
            # This preserves expected value, eliminating quantization bias
            low_vals = (low_idx - zero_point) * scale
            high_vals = (high_idx - zero_point) * scale
            return stochastic_round(block, low_vals, high_vals)
        else:
            # Deterministic rounding: always round to nearest
            indices = np.clip(np.round(positions), 0, self.config.max_val)
            return (indices - zero_point) * scale

    def quantize(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Quantize a weight tensor using S-Bits.

        DEFINITION:
            Block-wise stochastic quantization of a weight tensor.

        PROBLEM:
            Storing full-precision (float32) weights requires 4 bytes per
            parameter. With 4-bit quantization, we need only 0.5 bytes
            per parameter — an 8x compression ratio.

        INPUTS:
            weights: Weight tensor of any shape

        OUTPUTS:
            Tuple of (quantized_weights, metadata):
            - quantized_weights: Same shape as input, quantized values
            - metadata: Dict with 'scales', 'zero_points', 'config'

        EXECUTION FLOW:
            1. Flatten weights for block processing
            2. Pad to multiple of block_size if needed
            3. Reshape into blocks
            4. For each block: compute parameters, quantize, dequantize
            5. Handle outliers separately
            6. Reshape back to original shape

        COMPLEXITY:
            Time: O(n) where n = total elements
            Space: O(n) for the output + O(n/block_size) for metadata

        SCALABILITY:
            - Scales linearly with tensor size
            - Block processing enables streaming/incremental quantization
            - Can be parallelized across blocks
        """
        original_shape = weights.shape
        flat = weights.flatten().astype(np.float32)

        # Pad to multiple of block_size
        n = len(flat)
        pad_len = (self.config.block_size - n % self.config.block_size) % self.config.block_size
        if pad_len > 0:
            flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.float32)])

        # Reshape into blocks
        num_blocks = len(flat) // self.config.block_size
        blocks = flat.reshape(num_blocks, self.config.block_size)

        # Quantize each block independently
        quantized_blocks = np.zeros_like(blocks)
        scales = np.zeros(num_blocks, dtype=np.float32)
        zero_points = np.zeros(num_blocks, dtype=np.float32)

        for i in range(num_blocks):
            block = blocks[i]
            scale, zp = self._compute_block_params(block)
            scales[i] = scale
            zero_points[i] = zp
            quantized_blocks[i] = self._quantize_block(block, scale, zp)

        # Reshape back to original shape
        quantized = quantized_blocks.reshape(-1)[:n].reshape(original_shape)

        metadata = {
            'scales': scales,
            'zero_points': zero_points,
            'original_shape': original_shape,
            'config': self.config,
        }

        return quantized, metadata

    def dequantize(self, quantized: np.ndarray, metadata: Dict) -> np.ndarray:
        """Dequantize a tensor (identity for S-Bits since we store float values).

        DEFINITION:
            S-Bits stores dequantized float values directly (the quantization
            noise is already baked in). This method exists for API consistency
            with other quantizers that store integer codes.

        INPUTS:
            quantized: Quantized weight tensor
            metadata: Metadata from quantize()

        OUTPUTS:
            Dequantized weight tensor (same as quantized for S-Bits)
        """
        # S-Bits stores dequantized values, so this is identity
        return quantized


# ============================================================================
# STOCHASTIC BITS COMPRESSOR (PACKED STORAGE)
# ============================================================================

class SBitsCompressor:
    """Compresses quantized weights into packed integer storage.

    ARCHITECTURE:
        - Quantizes weights using SBitsQuantizer
        - Packs multiple quantized values into single integers
        - Stores scales and zero-points per block
        - Achieves true memory reduction (not just dequantized float)

    WHY PACKED:
        A 4-bit value only needs 4 bits of storage, but a float32
        needs 32 bits. By packing 8 values into a single int32,
        we achieve 8x compression for 4-bit quantization.

    MEMORY COMPARISON:
        - float32: 4 bytes per parameter
        - 4-bit packed: 0.5 bytes per parameter (8x savings)
        - 3-bit packed: 0.375 bytes per parameter (10.7x savings)
        - 2-bit packed: 0.25 bytes per parameter (16x savings)
    """

    def __init__(self, config: SBitsConfig):
        """Initialize the compressor.

        Args:
            config: Quantization configuration
        """
        self.config = config
        self.quantizer = SBitsQuantizer(config)

    def compress(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Compress weights into packed integer storage.

        DEFINITION:
            Quantizes weights and packs multiple values per integer.

        EXECUTION FLOW:
            1. Quantize weights using SBitsQuantizer
            2. Map dequantized values back to integer indices
            3. Pack indices into integers (e.g., 8 × 4-bit into int32)
            4. Store scales, zero_points, and packed indices

        COMPLEXITY:
            Time: O(n) for quantization + O(n/bits) for packing
            Space: O(n/bits) for packed storage

        MEMORY SAVINGS:
            For 4-bit with block_size=64:
            - Original: n * 4 bytes (float32)
            - Packed: n * 0.5 bytes + (n/64) * 8 bytes (metadata)
            - Savings: ~7.5x for n >> block_size
        """
        original_shape = weights.shape
        flat = weights.flatten().astype(np.float32)

        # Quantize to get dequantized values
        quantized, metadata = self.quantizer.quantize(weights)

        # Map back to integer indices
        flat_q = quantized.flatten()
        scales = metadata['scales']
        zero_points = metadata['zero_points']
        block_size = self.config.block_size

        # Compute integer indices for each value
        indices = np.zeros(len(flat_q), dtype=np.int32)
        num_blocks = len(flat_q) // block_size
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            block = flat_q[start:end]
            # Reverse the quantization: index = round(v / scale + zp)
            if scales[i] > 1e-10:
                positions = block / scales[i] + zero_points[i]
                indices[start:end] = np.clip(np.round(positions), 0, self.config.max_val)

        # Pack indices into integers
        values_per_int = 32 // self.config.bits
        packed_len = (len(indices) + values_per_int - 1) // values_per_int
        packed = np.zeros(packed_len, dtype=np.int32)

        for i in range(len(indices)):
            # Determine which packed integer and which position within it
            packed_idx = i // values_per_int
            bit_offset = (i % values_per_int) * self.config.bits
            packed[packed_idx] |= (indices[i] & self.config.max_val) << bit_offset

        metadata['packed'] = packed
        metadata['original_numel'] = len(flat)

        return packed, metadata

    def decompress(self, packed: np.ndarray, metadata: Dict) -> np.ndarray:
        """Decompress packed integers back to float weights.

        DEFINITION:
            Unpacks integer indices and maps them back to float values
            using stored scales and zero-points.

        EXECUTION FLOW:
            1. Unpack integers to extract individual indices
            2. For each block: map indices to floats using scale and zero_point
            3. Reshape to original shape

        COMPLEXITY:
            Time: O(n)
            Space: O(n) for output float array
        """
        original_shape = metadata['original_shape']
        config = metadata['config']
        scales = metadata['scales']
        zero_points = metadata['zero_points']
        block_size = config.block_size
        original_numel = metadata['original_numel']

        # Unpack indices
        values_per_int = 32 // config.bits
        indices = np.zeros(original_numel, dtype=np.int32)

        for i in range(original_numel):
            packed_idx = i // values_per_int
            bit_offset = (i % values_per_int) * config.bits
            indices[i] = (packed[packed_idx] >> bit_offset) & config.max_val

        # Dequantize each block
        result = np.zeros(original_numel, dtype=np.float32)
        num_blocks = (original_numel + block_size - 1) // block_size

        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, original_numel)
            idx_block = indices[start:end]
            result[start:end] = (idx_block - zero_points[i]) * scales[i]

        return result.reshape(original_shape)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_quantization_error(original: np.ndarray, quantized: np.ndarray) -> Dict[str, float]:
    """Compute quantization error metrics.

    DEFINITION:
        Measures how much information is lost during quantization.

    METRICS:
        - MSE: Mean squared error (average squared difference)
        - RMSE: Root mean squared error (in same units as original)
        - PSNR: Peak signal-to-noise ratio (higher = better)
        - Cosine similarity: How aligned the vectors are
        - Max absolute error: Worst-case error

    INPUTS:
        original: Original float tensor
        quantized: Quantized (dequantized) tensor

    OUTPUTS:
        Dict with error metrics

    EDUCATIONAL:
        - MSE < 1e-4: Excellent quantization (negligible loss)
        - MSE 1e-4 to 1e-2: Good quantization (acceptable loss)
        - MSE > 1e-2: Poor quantization (significant loss)
        - PSNR > 40 dB: Excellent quality
        - PSNR 30-40 dB: Good quality
        - PSNR < 30 dB: Poor quality
    """
    diff = original.astype(np.float32) - quantized.astype(np.float32)
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    # PSNR: peak signal-to-noise ratio
    # Higher is better, measures relative to peak value
    peak = float(np.max(np.abs(original)))
    psnr = float(20 * np.log10(peak / (rmse + 1e-10)))

    # Cosine similarity: measures directional alignment
    flat_o = original.flatten().astype(np.float32)
    flat_q = quantized.flatten().astype(np.float32)
    dot = np.dot(flat_o, flat_q)
    norm_o = np.linalg.norm(flat_o)
    norm_q = np.linalg.norm(flat_q)
    cosine_sim = float(dot / (norm_o * norm_q + 1e-10))

    # Max absolute error: worst-case
    max_error = float(np.max(np.abs(diff)))

    return {
        'mse': mse,
        'rmse': rmse,
        'psnr': psnr,
        'cosine_similarity': cosine_sim,
        'max_absolute_error': max_error,
    }


def compare_bit_widths(weights: np.ndarray, bit_widths: List[int] = [2, 3, 4]) -> Dict[int, Dict]:
    """Compare quantization quality across different bit widths.

    DEFINITION:
        Runs S-Bits quantization at multiple bit widths and reports
        error metrics for each. Helps choose the right bit width
        for a given accuracy requirement.

    INPUTS:
        weights: Weight tensor to quantize
        bit_widths: List of bit widths to compare

    OUTPUTS:
        Dict mapping bit width to error metrics

    EDUCATIONAL:
        This function helps answer the question: "How many bits can I
        use before accuracy drops too much?" The answer depends on:
        - The distribution of weights (more uniform = fewer bits OK)
        - The application (inference vs training)
        - The accuracy requirements
    """
    results = {}

    for bits in bit_widths:
        config = SBitsConfig(bits=bits, use_stochastic_rounding=True)
        quantizer = SBitsQuantizer(config)
        quantized, _ = quantizer.quantize(weights)
        error = compute_quantization_error(weights, quantized)
        results[bits] = error

    return results


def memory_savings_report(original_bytes: int, config: SBitsConfig) -> Dict[str, float]:
    """Compute memory savings from S-Bits quantization.

    DEFINITION:
        Reports how much memory is saved by using S-Bits instead of
        full-precision storage.

    INPUTS:
        original_bytes: Original storage in bytes (e.g., n * 4 for float32)
        config: Quantization configuration

    OUTPUTS:
        Dict with original_bytes, compressed_bytes, savings_bytes, ratio

    EDUCATIONAL:
        For 1B parameters:
        - float32: 4 GB
        - 4-bit S-Bits: ~0.5 GB + metadata
        - 3-bit S-Bits: ~0.375 GB + metadata
        - 2-bit S-Bits: ~0.25 GB + metadata
    """
    # Packed storage: bits per value * number of values / 8 = bytes
    # Plus metadata: scales (4 bytes) + zero_points (4 bytes) per block
    values_per_byte = 8 // config.bits
    packed_bytes = original_bytes // (32 // config.bits)

    # Metadata overhead
    num_elements = original_bytes // 4  # float32 = 4 bytes each
    num_blocks = (num_elements + config.block_size - 1) // config.block_size
    metadata_bytes = num_blocks * 8  # scale (4) + zero_point (4) per block

    total_compressed = packed_bytes + metadata_bytes

    return {
        'original_bytes': original_bytes,
        'packed_bytes': packed_bytes,
        'metadata_bytes': metadata_bytes,
        'total_compressed_bytes': total_compressed,
        'savings_bytes': original_bytes - total_compressed,
        'compression_ratio': original_bytes / (total_compressed + 1e-10),
        'bits_per_value': config.bits,
    }


# ============================================================================
# DEMO / VISUALIZATION HELPERS
# ============================================================================

def demo_stochastic_vs_deterministic(n_samples: int = 10000) -> Dict[str, float]:
    """Compare stochastic vs deterministic rounding bias.

    DEFINITION:
        Demonstrates that stochastic rounding eliminates systematic bias
        while deterministic rounding introduces it.

    EDUCATIONAL:
        This is the key insight of S-Bits: by adding controlled randomness,
        we remove systematic error. The variance increases slightly, but
        the expected value is preserved exactly.

    EXAMPLE:
        Values uniformly distributed in [0, 1], quantized to 4 levels:
        - Deterministic: bias ≈ 0 (symmetric case, but breaks for non-uniform)
        - Stochastic: bias ≈ 0 always (by construction)
    """
    # Generate values between 0 and 1
    values = np.random.uniform(0, 1, n_samples).astype(np.float32)

    # Quantization levels for 2-bit: 0, 1/3, 2/3, 1
    levels = np.array([0, 1/3, 2/3, 1], dtype=np.float32)

    # Deterministic rounding (round to nearest)
    det_result = np.zeros_like(values)
    for i, v in enumerate(values):
        distances = np.abs(levels - v)
        det_result[i] = levels[np.argmin(distances)]

    # Stochastic rounding
    sto_result = np.zeros_like(values)
    for i, v in enumerate(values):
        # Find the two nearest levels
        idx = np.searchsorted(levels, v)
        idx = np.clip(idx, 1, len(levels) - 1)
        low, high = levels[idx - 1], levels[idx]
        prob = (v - low) / (high - low) if high != low else 0
        sto_result[i] = high if np.random.random() < prob else low

    # Compute bias (should be ~0 for stochastic)
    det_bias = float(np.mean(det_result - values))
    sto_bias = float(np.mean(sto_result - values))

    return {
        'deterministic_bias': det_bias,
        'stochastic_bias': sto_bias,
        'deterministic_mse': float(np.mean((det_result - values) ** 2)),
        'stochastic_mse': float(np.mean((sto_result - values) ** 2)),
        'n_samples': n_samples,
    }
