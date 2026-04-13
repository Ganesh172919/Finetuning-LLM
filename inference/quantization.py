"""
################################################################################
QUANTIZATION — COMPRESSING MODELS FOR EFFICIENCY
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Quantization?
    Quantization reduces the precision of model weights and activations
    from floating point (fp32/fp16) to lower precision (int8/int4).

Why Quantize?
    - fp32: 4 bytes per parameter
    - fp16: 2 bytes per parameter
    - int8: 1 byte per parameter (4x smaller than fp32)
    - int4: 0.5 bytes per parameter (8x smaller than fp32!)

    LLaMA-70B:
    - fp32: 280GB (impossible on single GPU)
    - fp16: 140GB (needs 2 A100-80GB)
    - int8: 70GB (fits on 1 A100-80GB)
    - int4: 35GB (fits on consumer GPU!)

How Quantization Works:
    1. Find the range of values (min, max)
    2. Map floating point range to integer range
    3. Store integers + scale factor + zero point

    Formula:
    x_quant = round(x / scale) + zero_point
    x_dequant = (x_quant - zero_point) * scale

Types of Quantization:
    1. Post-Training Quantization (PTQ): Quantize after training
    2. Quantization-Aware Training (QAT): Train with quantization
    3. Dynamic Quantization: Quantize activations at runtime

Interview Questions:
        Q: "What is quantization and why is it important?"
        A: Reducing model precision from fp32 to int8/int4.
           It's important because it reduces memory and compute
           requirements, enabling deployment on smaller hardware.

        Q: "What's the tradeoff of quantization?"
        A: Smaller model and faster inference, but potential quality loss.
           The key is to quantize intelligently to minimize quality impact.

################################################################################
"""

import numpy as np
from typing import Tuple, Optional

################################################################################
# SECTION 1: LINEAR QUANTIZATION
################################################################################

class LinearQuantizer:
    """
    Linear (Affine) Quantization
    =============================

    Maps floating point values to integers using a linear transformation.

    Formula:
        scale = (max_val - min_val) / (2^n_bits - 1)
        zero_point = round(-min_val / scale)
        x_quant = clamp(round(x / scale) + zero_point, 0, 2^n_bits - 1)
        x_dequant = (x_quant - zero_point) * scale

    Symmetric vs Asymmetric:
        Symmetric: zero_point = 0, range is [-max, max]
        Asymmetric: zero_point is learned, range is [min, max]

    Interview Questions:
        Q: "What's the difference between symmetric and asymmetric quantization?"
        A: Symmetric maps values symmetrically around zero (simpler).
           Asymmetric can represent the actual range (more accurate
           for activations that aren't centered at zero).
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = True):
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.qmin = -(2 ** (n_bits - 1)) if symmetric else 0
        self.qmax = 2 ** (n_bits - 1) - 1 if symmetric else 2 ** n_bits - 1

    def compute_params(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Compute quantization parameters (scale and zero_point).

        Args:
            x: Tensor to quantize

        Returns:
            scale: Scaling factor
            zero_point: Zero point offset
        """
        if self.symmetric:
            abs_max = np.max(np.abs(x))
            scale = abs_max / self.qmax
            zero_point = 0
        else:
            min_val = np.min(x)
            max_val = np.max(x)
            scale = (max_val - min_val) / (self.qmax - self.qmin)
            zero_point = self.qmin - np.round(min_val / scale)

        return scale, zero_point

    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Quantize tensor.

        Args:
            x: Float tensor

        Returns:
            x_quant: Quantized integer tensor
            scale: Scaling factor
            zero_point: Zero point
        """
        scale, zero_point = self.compute_params(x)

        # Quantize
        x_quant = np.clip(
            np.round(x / scale + zero_point),
            self.qmin, self.qmax
        ).astype(np.int8 if self.n_bits == 8 else np.int32)

        return x_quant, scale, zero_point

    def dequantize(self, x_quant: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
        """
        Dequantize tensor.

        Args:
            x_quant: Quantized integer tensor
            scale: Scaling factor
            zero_point: Zero point

        Returns:
            x: Float tensor (approximate)
        """
        return (x_quant.astype(np.float32) - zero_point) * scale


################################################################################
# SECTION 2: INT8 QUANTIZATION
################################################################################

class Int8Quantizer:
    """
    INT8 Quantization
    =================

    The most common quantization for production inference.

    Weight-Only INT8:
        Weights: int8
        Activations: fp16
        Computation: mixed precision

    Weight-Activation INT8:
        Weights: int8
        Activations: int8
        Computation: int8

    Interview Questions:
        Q: "When should I use weight-only vs weight-activation quantization?"
        A: Weight-only: simpler, less quality loss, good for memory-bound models.
           Weight-activation: more speedup, but harder to implement well.
    """

    def __init__(self):
        self.quantizer = LinearQuantizer(n_bits=8, symmetric=True)

    def quantize_weights(self, weight: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Quantize weight matrix.

        Args:
            weight: Float weight matrix

        Returns:
            weight_int8: Quantized weights
            scale: Scaling factor
        """
        w_quant, scale, zero_point = self.quantizer.quantize(weight)
        return w_quant, scale

    def quantize_activation(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Quantize activation tensor."""
        return self.quantize_weights(x)  # Same process

    def matmul_int8(
        self,
        a_int8: np.ndarray,
        a_scale: float,
        b_int8: np.ndarray,
        b_scale: float
    ) -> np.ndarray:
        """
        Matrix multiplication with INT8 tensors.

        Args:
            a_int8: Quantized matrix A
            a_scale: Scale for A
            b_int8: Quantized matrix B
            b_scale: Scale for B

        Returns:
            result: fp32 result
        """
        # Integer matmul (would use optimized kernel in practice)
        result_int32 = np.matmul(a_int8.astype(np.int32), b_int8.astype(np.int32))

        # Rescale
        return result_int32.astype(np.float32) * a_scale * b_scale


################################################################################
# SECTION 3: INT4 QUANTIZATION
################################################################################

class Int4Quantizer:
    """
    INT4 Quantization
    =================

    Aggressive quantization: 4 bits per weight.

    Techniques:
    1. Group Quantization: Quantize groups of values together
    2. Symmetric: Simpler but less accurate
    3. Asymmetric: More accurate but needs zero point

    Group Quantization:
        Instead of one scale per tensor, use one scale per group.
        Group size 128: 128 values share one scale.
        This captures local variations better.

    Interview Questions:
        Q: "How does INT4 quantization work?"
        A: Group weights (e.g., 128 per group), quantize each group
           separately with its own scale. This preserves local
           variations while using only 4 bits per weight.

        Q: "What's GPTQ?"
        A: A post-training quantization method that uses second-order
           information (Hessian) to minimize quantization error.
    """

    def __init__(self, group_size: int = 128):
        self.group_size = group_size
        self.quantizer = LinearQuantizer(n_bits=4, symmetric=True)

    def quantize(self, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize weight matrix with group quantization.

        Args:
            weight: [out_features × in_features]

        Returns:
            weight_int4: Quantized weights (packed)
            scales: Scale factors per group
        """
        out_features, in_features = weight.shape
        n_groups = in_features // self.group_size

        # Reshape to groups
        weight_grouped = weight.reshape(out_features, n_groups, self.group_size)

        # Quantize each group
        scales = np.zeros((out_features, n_groups))
        weight_int4 = np.zeros_like(weight_grouped, dtype=np.int8)

        for i in range(out_features):
            for j in range(n_groups):
                group = weight_grouped[i, j]
                w_q, scale, _ = self.quantizer.quantize(group)
                weight_int4[i, j] = w_q
                scales[i, j] = scale

        return weight_int4, scales

    def dequantize(self, weight_int4: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Dequantize weight matrix."""
        out_features, n_groups, group_size = weight_int4.shape

        weight = np.zeros_like(weight_int4, dtype=np.float32)
        for i in range(out_features):
            for j in range(n_groups):
                weight[i, j] = weight_int4[i, j].astype(np.float32) * scales[i, j]

        return weight.reshape(out_features, -1)


################################################################################
# SECTION 4: QUANTIZED LINEAR LAYER
################################################################################

class QuantizedLinear:
    """
    Quantized Linear Layer
    ======================

    A linear layer with quantized weights for efficient inference.

    Original: y = x @ W + b (fp32 or fp16)
    Quantized: y = x @ dequant(W_int4) + b (mixed precision)

    Interview Questions:
        Q: "How do you use quantized models?"
        A: Load quantized weights, dequantize on the fly during inference.
           The dequantization overhead is small compared to the memory savings.
    """

    def __init__(self, weight: np.ndarray, bias: Optional[np.ndarray] = None, n_bits: int = 4):
        self.n_bits = n_bits

        if n_bits == 4:
            self.quantizer = Int4Quantizer()
            self.weight_q, self.scales = self.quantizer.quantize(weight)
        elif n_bits == 8:
            self.quantizer = Int8Quantizer()
            self.weight_q, self.scales = self.quantizer.quantize_weights(weight)
        else:
            raise ValueError(f"Unsupported n_bits: {n_bits}")

        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with dequantization.

        Args:
            x: Input tensor

        Returns:
            output: y = x @ W + b
        """
        # Dequantize weights
        if self.n_bits == 4:
            weight = self.quantizer.dequantize(self.weight_q, self.scales)
        else:
            weight = self.weight_q.astype(np.float32) * self.scales

        # Linear transformation
        output = np.matmul(x, weight.T)

        if self.bias is not None:
            output = output + self.bias

        return output


################################################################################
# SECTION 5: TESTING
################################################################################

def demonstrate_quantization():
    """Demonstrate quantization concepts."""
    print("=" * 70)
    print("QUANTIZATION DEMONSTRATION")
    print("=" * 70)

    # Original weights
    weight = np.random.randn(64, 64)

    # INT8 quantization
    print("\n--- INT8 Quantization ---")
    int8_q = Int8Quantizer()
    w_int8, scale = int8_q.quantize_weights(weight)
    w_recon = w_int8.astype(np.float32) * scale
    error = np.mean(np.abs(weight - w_recon))
    print(f"Original shape: {weight.shape}")
    print(f"Quantized dtype: {w_int8.dtype}")
    print(f"Mean absolute error: {error:.6f}")
    print(f"Compression: 4 bytes → 1 byte (4x)")

    # INT4 quantization
    print("\n--- INT4 Quantization ---")
    int4_q = Int4Quantizer(group_size=32)
    w_int4, scales = int4_q.quantize(weight)
    w_recon = int4_q.dequantize(w_int4, scales)
    error = np.mean(np.abs(weight - w_recon))
    print(f"Quantized shape: {w_int4.shape}")
    print(f"Mean absolute error: {error:.6f}")
    print(f"Compression: 4 bytes → 0.5 bytes (8x)")

    # Quantized linear layer
    print("\n--- Quantized Linear Layer ---")
    qlayer = QuantizedLinear(weight, n_bits=4)
    x = np.random.randn(1, 64)
    out = qlayer.forward(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")

    # Size comparison
    print("\n--- Size Comparison ---")
    n_params = 70e9  # LLaMA-70B
    for dtype, bytes_per_param in [("fp32", 4), ("fp16", 2), ("int8", 1), ("int4", 0.5)]:
        size_gb = n_params * bytes_per_param / 1e9
        print(f"{dtype}: {size_gb:.0f} GB")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_quantization()


################################################################################
# REFERENCES
################################################################################

# [1] Jacob, B., et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.
# [2] Frantar, E., et al. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.
# [3] Dettmers, T., et al. (2023). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.
# [4] Lin, J., et al. (2024). AWQ: Activation-aware Weight Quantization.

################################################################################
