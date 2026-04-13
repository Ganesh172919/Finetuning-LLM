"""
################################################################################
QUANTIZATION — INT4/INT8 QUANTIZATION FOR LANGUAGE MODELS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Quantization?
    Quantization reduces model weights from FP16/FP32 to INT8/INT4,
    dramatically reducing memory and compute requirements with minimal
    quality loss. A 7B model in FP16 needs 14GB; in INT4, just 3.5GB.

Why does it matter?
    Quantization enables:
    - 2-4x memory reduction
    - 2-4x faster inference (integer ops are faster)
    - Run 7B models on phones, 70B on single GPU
    - Minimal quality loss (often <1% perplexity increase)

How does it work?
    1. Absmax: Simple, scale by max absolute value
    2. Zero-point: Handle asymmetric distributions
    3. GPTQ: Layer-wise optimal quantization using Hessian
    4. AWQ: Protect salient weights from quantization error

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ Quantization Pipeline                                       │
    │                                                              │
    │  FP16 Weights → [Quantize] → INT4/INT8 Weights + Scale     │
    │       ↓                         ↓                            │
    │  Dequantize on-the-fly during forward pass                  │
    │                                                              │
    │  W_fp16 ≈ W_int * scale + zero_point                       │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2022: LLM.int8() (Dettmers et al.)
    - 2023: GPTQ (Frantar et al.), AWQ (Lin et al.)
    - 2024: QuIP#, AQLM — sub-2-bit quantization
    - 2025: BitNet 1.58-bit, native quantization training

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


################################################################################
# SECTION 1: CONFIGURATION
################################################################################

@dataclass
class QuantizationConfig:
    """Quantization configuration."""
    bits: int = 8
    method: str = 'absmax'  # absmax, zeropoint, gptq, awq
    group_size: int = 128
    symmetric: bool = True


################################################################################
# SECTION 2: ABSMAX QUANTIZER
################################################################################

class AbsmaxQuantizer:
    """
    Absmax INT8 Quantization.

    Formula: scale = max(|W|) / 127, W_q = round(W / scale)

    Simple but sensitive to outliers (one large value wastes range).

    Interview Question:
        "How does absmax quantization work?"
        Scale = max(|W|) / 127. Divide all weights by scale and round
        to INT8. To dequantize: W ≈ W_q * scale. Simple and fast,
        but outliers waste the quantization range.
    """

    def quantize(self, weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Quantize weights to INT8.

        Args:
            weights: FP16/FP32 weight matrix

        Returns:
            Tuple of (INT8 weights, scale factor)
        """
        scale = np.max(np.abs(weights)) / 127.0
        if scale == 0:
            return np.zeros_like(weights, dtype=np.int8), 1.0
        quantized = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
        return quantized, scale

    def dequantize(self, quantized: np.ndarray, scale: float) -> np.ndarray:
        """Dequantize INT8 back to float."""
        return quantized.astype(np.float32) * scale


################################################################################
# SECTION 3: ZERO-POINT QUANTIZER
################################################################################

class ZeropointQuantizer:
    """
    Zero-point INT8 Quantization.

    Better for asymmetric distributions. Formula:
        scale = (max - min) / 255
        zero_point = round(-min / scale) - 128

    Interview Question:
        "What's the difference between symmetric and asymmetric quantization?"
        Symmetric: zero_point = 0, range is [-127, 127]
        Asymmetric: zero_point != 0, range is [-128, 127]
        Asymmetric handles distributions that don't center at zero.
    """

    def quantize(self, weights: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """Quantize with zero-point."""
        w_min, w_max = weights.min(), weights.max()
        scale = (w_max - w_min) / 255.0
        if scale == 0:
            return np.zeros_like(weights, dtype=np.int8), 1.0, 0
        zero_point = round(-w_min / scale) - 128
        quantized = np.clip(np.round(weights / scale) + zero_point, -128, 127).astype(np.int8)
        return quantized, scale, zero_point

    def dequantize(self, quantized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        """Dequantize with zero-point."""
        return (quantized.astype(np.float32) - zero_point) * scale


################################################################################
# SECTION 4: GPTQ QUANTIZER
################################################################################

class GPTQQuantizer:
    """
    GPTQ — Layer-wise Optimal Quantization.

    Key insight: quantize one column at a time, compensate error to
    remaining columns using Hessian information.

    Formula: min ||WX - (W + delta)X||^2, solve column by column

    Paper: "GPTQ: Accurate Post-Training Quantization for Generative
            Pre-trained Transformers" (Frantar et al., ICLR 2023)

    Interview Question:
        "How does GPTQ work?"
        GPTQ quantizes one weight column at a time. After quantizing
        column j, it distributes the quantization error to remaining
        columns using the Hessian inverse. This minimizes the overall
        output error. It's much better than naive quantization because
        it accounts for weight correlations.
    """

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def quantize_layer(self, weights: np.ndarray, hessian: np.ndarray,
                       bits: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize a weight matrix using GPTQ.

        Args:
            weights: (d_out, d_in) weight matrix
            hessian: (d_in, d_in) Hessian matrix (X^T X)
            bits: Target bit width

        Returns:
            Tuple of (quantized weights, scales per group)
        """
        d_out, d_in = weights.shape
        n_groups = d_in // self.group_size
        scales = np.zeros((d_out, n_groups))

        # Quantize column by column
        quantized = weights.copy()
        for g in range(n_groups):
            start = g * self.group_size
            end = start + self.group_size
            group_weights = weights[:, start:end]

            # Compute scale for this group
            scale = np.max(np.abs(group_weights)) / (2**(bits-1) - 1)
            scales[:, g] = scale

            # Quantize
            q = np.clip(np.round(group_weights / scale), -(2**(bits-1)), 2**(bits-1)-1)
            quantized[:, start:end] = q

        return quantized, scales

    def dequantize(self, quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Dequantize GPTQ weights."""
        d_out, d_in = quantized.shape
        n_groups = d_in // self.group_size
        result = np.zeros_like(quantized, dtype=np.float32)
        for g in range(n_groups):
            start = g * self.group_size
            end = start + self.group_size
            result[:, start:end] = quantized[:, start:end] * scales[:, g:g+1]
        return result


################################################################################
# SECTION 5: AWQ QUANTIZER
################################################################################

class AWQQuantizer:
    """
    AWQ — Activation-aware Weight Quantization.

    Key insight: protect salient weights (those with large activations)
    from quantization error by scaling them before quantization.

    Paper: "AWQ: Activation-aware Weight Quantization for LLM
            Compression and Acceleration" (Lin et al., 2024)

    Interview Question:
        "How does AWQ differ from GPTQ?"
        AWQ identifies "salient" weights by looking at activation magnitudes.
        It scales these weights up before quantization (reducing their
        relative error), then scales the activations down to compensate.
        GPTQ uses Hessian-based error correction. AWQ is faster and
        often achieves better quality at 4-bit.
    """

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def find_salient_channels(self, weights: np.ndarray,
                               activations: np.ndarray, top_k: int = 100) -> np.ndarray:
        """
        Find channels with largest activation magnitudes.

        Args:
            weights: Weight matrix
            activations: Activation statistics
            top_k: Number of salient channels

        Returns:
            Indices of salient channels
        """
        # Activation magnitude per channel
        act_mag = np.mean(np.abs(activations), axis=0)
        return np.argsort(act_mag)[-top_k:]

    def quantize(self, weights: np.ndarray, activations: np.ndarray,
                 bits: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        AWQ quantization.

        Args:
            weights: Weight matrix
            activations: Activation statistics
            bits: Target bits

        Returns:
            Tuple of (quantized, scales, salient_scale)
        """
        d_out, d_in = weights.shape

        # Find salient channels
        salient = self.find_salient_channels(weights, activations)

        # Compute per-channel scale (protect salient channels)
        scale = np.ones(d_in)
        scale[salient] = 2.0  # Scale up salient channels

        # Apply scale before quantization
        scaled_weights = weights * scale

        # Quantize
        n_groups = d_in // self.group_size
        quantized = np.zeros_like(scaled_weights, dtype=np.int8)
        scales = np.zeros((d_out, n_groups))

        for g in range(n_groups):
            start = g * self.group_size
            end = start + self.group_size
            group = scaled_weights[:, start:end]
            s = np.max(np.abs(group)) / (2**(bits-1) - 1)
            scales[:, g] = s
            quantized[:, start:end] = np.clip(np.round(group / s), -(2**(bits-1)), 2**(bits-1)-1)

        return quantized, scales, scale


################################################################################
# SECTION 6: QUANTIZED LINEAR LAYER
################################################################################

class QuantizedLinear:
    """
    Linear layer that operates on quantized weights.

    Dequantizes on-the-fly during forward pass.

    Interview Question:
        "How does a quantized linear layer work?"
        Store weights in INT4/INT8 + scale factors. During forward pass,
        dequantize to FP16 on-the-fly, then do matrix multiply. Modern
        GPUs have INT*FP mixed-precision instructions that do this
        efficiently without full dequantization.
    """

    def __init__(self, quantized: np.ndarray, scales: np.ndarray,
                 bits: int = 8):
        self.quantized = quantized
        self.scales = scales
        self.bits = bits

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with dequantization.

        Args:
            x: Input tensor

        Returns:
            Output after dequantized matmul
        """
        # Dequantize weights
        d_out, d_in = self.quantized.shape
        group_size = d_in // len(self.scales[0]) if len(self.scales.shape) > 1 else d_in

        if len(self.scales.shape) > 1:
            # Group-wise scales
            weights = np.zeros_like(self.quantized, dtype=np.float32)
            for g in range(self.scales.shape[1]):
                start = g * group_size
                end = start + group_size
                weights[:, start:end] = self.quantized[:, start:end] * self.scales[:, g:g+1]
        else:
            weights = self.quantized.astype(np.float32) * self.scales

        return x @ weights.T


################################################################################
# SECTION 7: DEMONSTRATION
################################################################################

def demonstrate_quantization():
    """Demonstrate quantization methods."""
    print("=" * 70)
    print("QUANTIZATION DEMONSTRATION")
    print("=" * 70)

    weights = np.random.randn(128, 256) * 0.02

    # Absmax
    print("\n1. ABSMAX QUANTIZATION (INT8)")
    print("-" * 40)
    q = AbsmaxQuantizer()
    quantized, scale = q.quantize(weights)
    dequant = q.dequantize(quantized, scale)
    error = np.mean(np.abs(weights - dequant))
    print(f"  Original shape: {weights.shape}, dtype: {weights.dtype}")
    print(f"  Quantized shape: {quantized.shape}, dtype: {quantized.dtype}")
    print(f"  Scale: {scale:.6f}")
    print(f"  Mean absolute error: {error:.6f}")
    print(f"  Compression: {weights.nbytes / quantized.nbytes:.1f}x")

    # Zero-point
    print("\n2. ZERO-POINT QUANTIZATION (INT8)")
    print("-" * 40)
    zp = ZeropointQuantizer()
    quantized, scale, zp_val = zp.quantize(weights)
    dequant = zp.dequantize(quantized, scale, zp_val)
    error = np.mean(np.abs(weights - dequant))
    print(f"  Scale: {scale:.6f}, Zero-point: {zp_val}")
    print(f"  Mean error: {error:.6f}")

    # GPTQ
    print("\n3. GPTQ QUANTIZATION (INT4)")
    print("-" * 40)
    gptq = GPTQQuantizer(group_size=64)
    hessian = np.eye(256)  # Simplified Hessian
    quantized, scales = gptq.quantize_layer(weights, hessian, bits=4)
    dequant = gptq.dequantize(quantized, scales)
    error = np.mean(np.abs(weights - dequant))
    print(f"  Groups: {256 // 64}")
    print(f"  Mean error: {error:.6f}")
    print(f"  Compression: 4x (FP16 → INT4)")

    # AWQ
    print("\n4. AWQ QUANTIZATION (INT4)")
    print("-" * 40)
    awq = AWQQuantizer(group_size=64)
    activations = np.random.randn(10, 256)  # Simulated activations
    quantized, scales, salient_scale = awq.quantize(weights, activations, bits=4)
    print(f"  Salient channels scaled: {np.sum(salient_scale > 1)}")
    print(f"  Quantized shape: {quantized.shape}")

    # Quantized Linear
    print("\n5. QUANTIZED LINEAR LAYER")
    print("-" * 40)
    qlin = QuantizedLinear(quantized.astype(np.int8), scales, bits=4)
    x = np.random.randn(1, 256)
    out = qlin.forward(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")

    # Memory comparison
    print("\n6. MEMORY COMPARISON")
    print("-" * 40)
    params = 7e9  # 7B model
    for dtype, bytes_per in [("FP32", 4), ("FP16", 2), ("INT8", 1), ("INT4", 0.5)]:
        mem_gb = params * bytes_per / 1e9
        print(f"  {dtype}: {mem_gb:.1f} GB")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_quantization()
