"""
Model Quantization: INT8, INT4, GPTQ, AWQ
============================================

Quantization reduces model precision from FP32/FP16 to lower bit widths,
dramatically reducing memory usage and improving inference speed.

Precision Levels:
┌─────────────────────────────────────────────────────────────┐
│ FP32: 32 bits per weight → 4 bytes → 7B model = 28 GB     │
│ FP16: 16 bits per weight → 2 bytes → 7B model = 14 GB     │
│ INT8:  8 bits per weight → 1 byte  → 7B model = 7 GB      │
│ INT4:  4 bits per weight → 0.5 byte → 7B model = 3.5 GB   │
│ NF4:   4 bits (normal)  → 0.5 byte → 7B model = 3.5 GB   │
└─────────────────────────────────────────────────────────────┘

Quantization Types:
  1. Post-Training Quantization (PTQ): Quantize after training
  2. Quantization-Aware Training (QAT): Simulate quantization during training
  3. Dynamic Quantization: Quantize activations at runtime

Methods:
  - Round-to-Nearest (RTN): Simple rounding
  - GPTQ: Second-order information based (OBQ family)
  - AWQ: Activation-aware weight quantization
  - GGUF: llama.cpp format with multiple quant levels

References:
  - Dettmers et al., "LLM.int8()" (2022)
  - Frantar et al., "GPTQ" (2022)
  - Lin et al., "AWQ" (2023)
  - Dettmers et al., "QLoRA" (2023)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class QuantizationConfig:
    """
    Configuration for quantization.

    Attributes:
        bits: Target bit width (4, 8, 16)
        method: Quantization method (rt, gptq, awq, nf4)
        group_size: Group size for per-group quantization
        sym: Whether to use symmetric quantization
        per_channel: Whether to quantize per-channel
    """
    bits: int = 8
    method: str = "rt"
    group_size: int = 128
    sym: bool = True
    per_channel: bool = True


# ============================================================================
# BASIC QUANTIZATION
# ============================================================================

def quantize_tensor(x: np.ndarray, bits: int = 8, sym: bool = True
                    ) -> Tuple[np.ndarray, float, int]:
    """
    Quantize a tensor to lower precision.

    Symmetric quantization:
        x_q = round(x / scale)
        scale = max(|x|) / (2^(bits-1) - 1)

    Asymmetric quantization:
        x_q = round((x - zero_point) / scale)
        scale = (max(x) - min(x)) / (2^bits - 1)
        zero_point = round(-min(x) / scale)

    Args:
        x: Input tensor (any shape)
        bits: Target bit width
        sym: Symmetric or asymmetric

    Returns:
        x_q: Quantized tensor (int8 or int4 packed)
        scale: Quantization scale
        zero_point: Zero point (0 for symmetric)
    """
    if sym:
        # Symmetric: range is [-2^(b-1), 2^(b-1)-1]
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1

        # Scale = max_abs / qmax
        max_abs = np.max(np.abs(x))
        scale = max_abs / qmax if max_abs > 0 else 1.0

        # Quantize
        x_q = np.clip(np.round(x / scale), qmin, qmax).astype(np.int8)

        return x_q, float(scale), 0
    else:
        # Asymmetric: range is [0, 2^b - 1]
        qmin = 0
        qmax = 2 ** bits - 1

        # Scale and zero point
        x_min, x_max = np.min(x), np.max(x)
        scale = (x_max - x_min) / (qmax - qmin) if x_max != x_min else 1.0
        zero_point = int(np.clip(np.round(-x_min / scale), qmin, qmax))

        # Quantize
        x_q = np.clip(np.round(x / scale) + zero_point, qmin, qmax).astype(np.uint8)

        return x_q, float(scale), zero_point


def dequantize_tensor(x_q: np.ndarray, scale: float, zero_point: int = 0
                      ) -> np.ndarray:
    """
    Dequantize tensor back to floating point.

    Args:
        x_q: Quantized tensor
        scale: Quantization scale
        zero_point: Zero point

    Returns:
        Dequantized tensor
    """
    return (x_q.astype(np.float32) - zero_point) * scale


# ============================================================================
# INT8 QUANTIZATION (LLM.int8())
# ============================================================================

class INT8Quantizer:
    """
    INT8 quantization for LLM inference.

    Key insight from LLM.int8():
    - Most weights can be quantized to INT8
    - Outlier features (>6σ) should stay in FP16
    - Mixed-precision decomposition handles this automatically

    Memory savings: 2x over FP16
    Quality: Minimal loss (<1% on most benchmarks)
    Speed: 1.5-2x on modern GPUs with INT8 tensor cores
    """

    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig(bits=8, method="rt")

    def quantize_weight(self, weight: np.ndarray) -> Dict:
        """
        Quantize weight matrix to INT8.

        Args:
            weight: Weight matrix [out_features, in_features]

        Returns:
            Dictionary with quantized weight and metadata
        """
        # Per-channel quantization (one scale per output channel)
        if self.config.per_channel:
            scales = np.zeros(weight.shape[0])
            for i in range(weight.shape[0]):
                max_abs = np.max(np.abs(weight[i]))
                scales[i] = max_abs / 127.0 if max_abs > 0 else 1.0
            weight_q = np.clip(np.round(weight / scales[:, np.newaxis]), -128, 127).astype(np.int8)
        else:
            weight_q, scale, _ = quantize_tensor(weight, bits=8, sym=True)
            scales = scale

        return {
            "weight": weight_q,
            "scales": scales,
            "bits": 8,
            "method": "int8",
        }

    def dequantize_weight(self, quantized: Dict) -> np.ndarray:
        """Dequantize weight back to float."""
        if isinstance(quantized["scales"], np.ndarray) and quantized["scales"].ndim > 0:
            return quantized["weight"].astype(np.float32) * quantized["scales"][:, np.newaxis]
        return quantized["weight"].astype(np.float32) * quantized["scales"]


# ============================================================================
# INT4 QUANTIZATION (GPTQ-style)
# ============================================================================

class INT4Quantizer:
    """
    INT4 quantization using GPTQ-style methods.

    GPTQ uses second-order information (Hessian) to find optimal quantization:
    1. Compute Hessian H = 2 * X * X^T
    2. For each column: find quantized value that minimizes reconstruction error
    3. Update remaining columns to compensate for quantization error

    This achieves much better quality than simple rounding at 4-bit precision.

    Memory savings: 4x over FP16, 8x over FP32
    Quality: ~1-3% degradation on most benchmarks
    Speed: 2-3x on INT4-capable hardware
    """

    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig(bits=4, method="gptq", group_size=128)

    def quantize_weight_gptq(self, weight: np.ndarray, hessian: Optional[np.ndarray] = None
                              ) -> Dict:
        """
        Quantize weight using GPTQ algorithm.

        Simplified version of the GPTQ algorithm:
        1. For each group of columns:
           a. Compute optimal quantization for each column
           b. Update remaining columns to compensate error

        Args:
            weight: Weight matrix [out_features, in_features]
            hessian: Hessian matrix (if None, use identity)

        Returns:
            Quantized weight and metadata
        """
        W = weight.copy()
        n_rows, n_cols = W.shape
        group_size = self.config.group_size

        # Quantization levels for INT4
        n_levels = 2 ** self.config.bits
        qmin = -(n_levels // 2)
        qmax = n_levels // 2 - 1

        # Storage for quantized weights and scales
        W_q = np.zeros_like(W, dtype=np.int8)
        scales = np.zeros((n_rows, n_cols // group_size))

        # Process each group
        for g in range(n_cols // group_size):
            start = g * group_size
            end = start + group_size

            # Get group of columns
            W_group = W[:, start:end]

            # Compute scale for this group (per-row)
            for i in range(n_rows):
                max_abs = np.max(np.abs(W_group[i]))
                scale = max_abs / qmax if max_abs > 0 else 1.0
                scales[i, g] = scale

                # Quantize
                W_q[i, start:end] = np.clip(
                    np.round(W_group[i] / scale), qmin, qmax
                ).astype(np.int8)

                # Compute quantization error
                error = W_group[i] - W_q[i, start:end].astype(np.float32) * scale

                # Update remaining columns to compensate (GPTQ key insight)
                if hessian is not None and end < n_cols:
                    # Simple compensation: distribute error to next group
                    h_diag = np.diag(hessian[start:end, start:end])
                    h_diag = np.maximum(h_diag, 1e-6)
                    compensation = error @ hessian[start:end, end:end+group_size] / h_diag.mean()
                    W[:, end:end+group_size] += compensation.reshape(-1, 1) if compensation.ndim == 1 else compensation

        return {
            "weight": W_q,
            "scales": scales,
            "bits": 4,
            "method": "gptq",
            "group_size": group_size,
        }

    def dequantize_weight(self, quantized: Dict) -> np.ndarray:
        """Dequantize GPTQ weight."""
        W_q = quantized["weight"]
        scales = quantized["scales"]
        group_size = quantized["group_size"]
        n_rows, n_cols = W_q.shape

        W = np.zeros_like(W_q, dtype=np.float32)
        for g in range(n_cols // group_size):
            start = g * group_size
            end = start + group_size
            W[:, start:end] = W_q[:, start:end].astype(np.float32) * scales[:, g:g+1]

        return W


# ============================================================================
# NF4 QUANTIZATION (QLoRA)
# ============================================================================

class NF4Quantizer:
    """
    NormalFloat4 (NF4) quantization from QLoRA.

    Key insight: Neural network weights are approximately normally distributed.
    NF4 places quantization levels according to the normal distribution,
    with more levels near zero (where most weights are).

    Standard INT4: Uniform levels → wastes precision where weights are sparse
    NF4: Non-uniform levels → optimal for normal distributions

    NF4 levels (for standard normal):
    [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0,
     0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]

    Double Quantization: Quantize the quantization constants too!
    - First: weights → NF4 (4 bits)
    - Second: scales → FP8 (8 bits)
    - Effective: ~3.5 bits per weight
    """

    # NF4 quantization levels (precomputed for standard normal)
    NF4_LEVELS = np.array([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ])

    def __init__(self):
        pass

    def quantize_weight_nf4(self, weight: np.ndarray, group_size: int = 64
                             ) -> Dict:
        """
        Quantize weight to NF4 format.

        Args:
            weight: Weight matrix
            group_size: Number of weights sharing a scale

        Returns:
            Quantized weight and metadata
        """
        n_rows, n_cols = weight.shape

        # Reshape to groups
        n_groups = (n_rows * n_cols) // group_size
        W_flat = weight.reshape(-1)[:n_groups * group_size].reshape(n_groups, group_size)

        # Compute absmax scale per group
        absmax = np.max(np.abs(W_flat), axis=1, keepdims=True)
        absmax = np.maximum(absmax, 1e-8)

        # Normalize to [-1, 1]
        W_norm = W_flat / absmax

        # Find nearest NF4 level for each weight
        W_q = np.zeros_like(W_norm, dtype=np.int8)
        for i, level in enumerate(self.NF4_LEVELS):
            if i == 0:
                mask = W_norm <= (self.NF4_LEVELS[0] + self.NF4_LEVELS[1]) / 2
            elif i == len(self.NF4_LEVELS) - 1:
                mask = W_norm > (self.NF4_LEVELS[-2] + self.NF4_LEVELS[-1]) / 2
            else:
                lower = (self.NF4_LEVELS[i-1] + self.NF4_LEVELS[i]) / 2
                upper = (self.NF4_LEVELS[i] + self.NF4_LEVELS[i+1]) / 2
                mask = (W_norm > lower) & (W_norm <= upper)
            W_q[mask] = i

        return {
            "weight": W_q.reshape(-1)[:n_rows * n_cols].reshape(n_rows, n_cols // (n_cols // group_size) if n_cols > group_size else n_cols),
            "scales": absmax.squeeze(),
            "bits": 4,
            "method": "nf4",
            "group_size": group_size,
        }

    def dequantize_weight(self, quantized: Dict) -> np.ndarray:
        """Dequantize NF4 weight."""
        W_q = quantized["weight"]
        scales = quantized["scales"]

        # Map indices back to NF4 levels
        W_dq = self.NF4_LEVELS[W_q.astype(int)]

        # Rescale
        if scales.ndim == 1:
            W_dq = W_dq * scales[:, np.newaxis]
        else:
            W_dq = W_dq * scales

        return W_dq


# ============================================================================
# AWQ (Activation-Aware Weight Quantization)
# ============================================================================

class AWQQuantizer:
    """
    AWQ: Activation-Aware Weight Quantization.

    Key insight: Not all weights are equally important. Weights that correspond
    to large activations should be quantized less aggressively.

    Algorithm:
    1. Run calibration data through the model
    2. Measure activation magnitudes per channel
    3. Scale important channels before quantization
    4. Quantize with standard INT4
    5. Compensate scaling in the next layer

    Benefits:
    - Better quality than GPTQ at same bit width
    - Faster quantization (no Hessian computation)
    - Works well with group quantization
    """

    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig(bits=4, method="awq")

    def compute_activation_scales(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute per-channel activation scales.

        Args:
            activations: Activation tensor [batch, seq_len, channels]

        Returns:
            Per-channel importance scores
        """
        # Average absolute activation per channel
        return np.mean(np.abs(activations), axis=(0, 1))

    def quantize_weight_awq(self, weight: np.ndarray,
                            activation_scales: np.ndarray,
                            alpha: float = 0.5) -> Dict:
        """
        Quantize weight with activation-aware scaling.

        Args:
            weight: Weight matrix [out, in]
            activation_scales: Per-input-channel importance [in]
            alpha: Scaling strength (0=no scaling, 1=full scaling)

        Returns:
            Quantized weight and metadata
        """
        # Compute scaling factors
        # Important channels get scaled up before quantization
        scales = np.power(activation_scales, alpha)
        scales = scales / np.mean(scales)  # Normalize

        # Apply scaling
        W_scaled = weight * scales[np.newaxis, :]

        # Quantize with standard INT4
        quantizer = INT4Quantizer(self.config)
        result = quantizer.quantize_weight_gptq(W_scaled)

        # Store scales for dequantization
        result["awq_scales"] = scales
        result["method"] = "awq"

        return result


# ============================================================================
# QUANTIZATION BENCHMARKS
# ============================================================================

def benchmark_quantization():
    """
    Compare quantization methods.

    Returns comparison table.
    """
    comparison = """
    ┌──────────────┬──────┬───────────┬─────────┬────────────┬──────────┐
    │ Method       │ Bits │ Memory    │ Speed   │ Quality    │ Hardware │
    ├──────────────┼──────┼───────────┼─────────┼────────────┼──────────┤
    │ FP32         │ 32   │ 1x        │ 1x      │ Best       │ Any      │
    │ FP16/BF16    │ 16   │ 0.5x      │ 1x      │ Best       │ GPU      │
    │ INT8 (RTN)   │ 8    │ 0.25x     │ 1.5x    │ ~99%       │ GPU/CPU  │
    │ INT8 (GPTQ)  │ 8    │ 0.25x     │ 1.5x    │ ~99.5%     │ GPU      │
    │ INT4 (RTN)   │ 4    │ 0.125x    │ 2x      │ ~95%       │ GPU      │
    │ INT4 (GPTQ)  │ 4    │ 0.125x    │ 2x      │ ~97%       │ GPU      │
    │ INT4 (AWQ)   │ 4    │ 0.125x    │ 2x      │ ~98%       │ GPU      │
    │ NF4 (QLoRA)  │ 4    │ 0.125x    │ 2x      │ ~97%       │ GPU      │
    │ INT4 (GGUF)  │ 4    │ 0.125x    │ 1.5x    │ ~96%       │ CPU      │
    └──────────────┴──────┴───────────┴─────────┴────────────┴──────────┘
    """
    return comparison


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_quantization():
    """
    Demonstrate quantization methods.

    Shows:
        1. INT8 quantization
        2. INT4 quantization
        3. NF4 quantization
        4. Error analysis
        5. Memory savings
    """
    print("=" * 70)
    print("Model Quantization — Demonstration")
    print("=" * 70)

    # Create sample weight matrix
    np.random.seed(42)
    weight = np.random.randn(256, 256).astype(np.float32)

    print(f"\nOriginal Weight Matrix:")
    print(f"  Shape: {weight.shape}")
    print(f"  Dtype: {weight.dtype}")
    print(f"  Memory: {weight.nbytes:,} bytes ({weight.nbytes/1024:.1f} KB)")
    print(f"  Range: [{weight.min():.3f}, {weight.max():.3f}]")

    # INT8 quantization
    print("\n[INT8 Quantization]")
    int8_q, int8_scale, _ = quantize_tensor(weight, bits=8, sym=True)
    int8_dq = dequantize_tensor(int8_q, int8_scale)
    int8_error = np.mean(np.abs(weight - int8_dq))
    print(f"  Quantized dtype: {int8_q.dtype}")
    print(f"  Memory: {int8_q.nbytes:,} bytes ({int8_q.nbytes/1024:.1f} KB)")
    print(f"  Compression: {weight.nbytes / int8_q.nbytes:.1f}x")
    print(f"  Mean error: {int8_error:.6f}")
    print(f"  Relative error: {int8_error / np.std(weight):.4%}")

    # INT4 quantization
    print("\n[INT4 Quantization]")
    int4_q, int4_scale, _ = quantize_tensor(weight, bits=4, sym=True)
    int4_dq = dequantize_tensor(int4_q, int4_scale)
    int4_error = np.mean(np.abs(weight - int4_dq))
    print(f"  Compression: {weight.nbytes / int4_q.nbytes:.1f}x")
    print(f"  Mean error: {int4_error:.6f}")
    print(f"  Relative error: {int4_error / np.std(weight):.4%}")

    # NF4 quantization
    print("\n[NF4 Quantization]")
    nf4 = NF4Quantizer()
    nf4_result = nf4.quantize_weight_nf4(weight, group_size=64)
    print(f"  Method: {nf4_result['method']}")
    print(f"  Group size: {nf4_result['group_size']}")
    print(f"  Levels: {len(nf4.NF4_LEVELS)}")

    # Memory comparison for LLM
    print("\n[Memory Comparison: Llama 2 7B]")
    params = 7e9
    for dtype, bits in [("FP32", 32), ("FP16", 16), ("INT8", 8), ("INT4", 4)]:
        memory_gb = params * bits / 8 / 1e9
        print(f"  {dtype}: {memory_gb:.1f} GB")

    # Comparison table
    print("\n[Quantization Methods Comparison]")
    print(benchmark_quantization())

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. INT8: 2x memory, ~1% quality loss, safe choice")
    print("  2. INT4: 4x memory, 2-3% quality loss, best for edge")
    print("  3. NF4: Optimal for normal distributions (QLoRA)")
    print("  4. AWQ: Best INT4 quality (activation-aware)")
    print("  5. GPTQ: Fast quantization with Hessian info")
    print("=" * 70)


if __name__ == "__main__":
    demo_quantization()
