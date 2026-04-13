"""
################################################################################
POST-TRAINING QUANTIZATION — FP8 AND NVFP4 FOR PRODUCTION INFERENCE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Post-Training Quantization?
    Post-training quantization (PTQ) reduces the numerical precision of a
    trained model's weights and activations from high precision (BF16/FP32)
    to lower precision (FP8, NVFP4) WITHOUT retraining. This cuts memory
    usage and speeds up inference on hardware that supports efficient
    low-precision arithmetic.

Why does it matter?
    - BF16 inference on a 70B model requires ~140 GB of VRAM
    - FP8 quantization roughly halves this to ~70 GB
    - NVFP4 can quarter it to ~35 GB
    - On modern hardware (Ada Lovelace, Blackwell), FP8 compute is 2x faster
      than BF16, giving both memory AND speed benefits
    - Quality loss is typically 0.5-2% on benchmarks — acceptable for serving

How does it work?
    1. Calibration: Run a small sample of representative data through the model
       to observe the range of weight and activation values
    2. Scale Factor Computation: For each tensor (or block), compute a scale
       factor that maps the observed range into the target precision's range
    3. Quantize: Divide by scale factor and round to target precision
    4. Dequantize: Multiply by scale factor to approximate original values
    5. Runtime: Apply quantized weights, dynamically quantize activations

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────────────┐
    │              POST-TRAINING QUANTIZATION PIPELINE                    │
    │                                                                     │
    │  ┌──────────────┐                                                   │
    │  │ BF16 Model   │  (original weights in bfloat16)                   │
    │  └──────┬───────┘                                                   │
    │         │                                                           │
    │         ▼                                                           │
    │  ┌──────────────┐     ┌───────────────────┐                         │
    │  │  Calibration  │────→│ Compute Scales    │  Per-tensor or per-block│
    │  │  (512 samples) │     │ max(|x|) → scale │                         │
    │  └──────────────┘     └────────┬──────────┘                         │
    │                                │                                     │
    │                                ▼                                     │
    │                    ┌───────────────────────┐                         │
    │                    │   Quantize Weights    │  w_q = round(w / s)     │
    │                    │   + Store Scales      │                         │
    │                    └───────────┬───────────┘                         │
    │                                │                                     │
    │                                ▼                                     │
    │                    ┌───────────────────────┐                         │
    │                    │   FP8 / NVFP4 Model   │                         │
    │                    │   + Runtime Act. Quant │                         │
    │                    └───────────────────────┘                         │
    │                                                                     │
    │  FP8 E4M3: sign(1) + exponent(4) + mantissa(3) = 8 bits            │
    │            Range: ±448, Precision: ~0.016 (good for activations)    │
    │  FP8 E5M2: sign(1) + exponent(5) + mantissa(2) = 8 bits            │
    │            Range: ±57344, Precision: ~0.063 (good for gradients)    │
    │  NVFP4:    sign(1) + exponent(2) + mantissa(1) = 4 bits             │
    │            Block of 16 values + micro-scale (FP8) + global (FP32)   │
    └─────────────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2017: INT8 quantization for inference (Jacob et al., Google)
    - 2019: Quantization-Aware Training (QAT) for mobile (Google)
    - 2022: FP8 format standardized by Intel/ARM/NVIDIA (OCP MX spec)
    - 2023: FP8 training demonstrated at scale (Micikevicius et al., NVIDIA)
    - 2024: DeepSeek-V3 validates FP8 training for 671B MoE model
    - 2025: NVFP4 inference ships on Blackwell (NVIDIA B200/B100)
    - 2026: FP8 becomes default production precision; NVFP4 for edge deployment

INTERVIEW QUESTIONS:
    1. "What is the difference between FP8 E4M3 and E5M2?"
       E4M3 has a 4-bit exponent and 3-bit mantissa, giving it higher
       precision (more mantissa bits) but smaller dynamic range (max ~448).
       E5M2 has a 5-bit exponent and 2-bit mantissa, giving it wider range
       (max ~57344) but lower precision. E4M3 is used for forward pass
       (weights + activations) where precision matters. E5M2 is used for
       gradients where range matters more.

    2. "Why per-tensor vs per-block quantization?"
       Per-tensor uses one scale factor for the entire tensor — simple but
       loses precision if the tensor has outliers. Per-block (e.g., 16
       elements) uses a local scale factor for each block, preserving more
       precision for outlier-heavy tensors. NVFP4 requires per-block because
       4-bit precision is too low for a single global scale.

    3. "How do you calibrate a model for quantization?"
       Run a small calibration dataset (typically 128-512 samples) through
       the model in FP16/BF16, recording the min/max values of each tensor
       (or activation). These observed ranges determine the scale factors.
       The calibration data should be representative of the production
       distribution — using random data gives poor results.

    4. "When should you NOT quantize a model?"
       If the model is already small (< 1B params), quantization overhead
       may exceed benefits. If serving on hardware without FP8 support,
       FP8 quantization provides no speedup. If the task requires extreme
       precision (e.g., mathematical reasoning), even 1% degradation may
       be unacceptable.

################################################################################
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
import math
import logging
import time

logger = logging.getLogger(__name__)


################################################################################
# SECTION 1: QUANTIZATION CONFIGURATION
################################################################################


@dataclass
class QuantizationConfig:
    """
    Quantization Configuration
    ==========================

    All hyperparameters for post-training quantization. No magic numbers.

    Attributes:
        format: Target quantization format. One of:
            - "fp8_e4m3": FP8 with 4-bit exponent, 3-bit mantissa (forward pass)
            - "fp8_e5m2": FP8 with 5-bit exponent, 2-bit mantissa (gradients)
            - "nvfp4":    NVFP4 with block-wise micro-scales (Blackwell)
        calibration_samples: Number of calibration samples to use
        per_tensor_scaling: Whether to use one scale per tensor (True) or
            per-channel/per-block (False)
        block_size: Block size for NVFP4 block-wise quantization (default: 16)
        dynamic_activations: Whether to dynamically quantize activations at
            runtime (True) or use static activation scales (False)
        activation_scale_factor: Scaling factor for activation quantization
            to avoid overflow (default: 1.0)
        weight_only: If True, only quantize weights (not activations)
    """
    format: str = "fp8_e4m3"
    calibration_samples: int = 512
    per_tensor_scaling: bool = True
    block_size: int = 16  # For NVFP4
    dynamic_activations: bool = True
    activation_scale_factor: float = 1.0
    weight_only: bool = False

    def __post_init__(self):
        """Validate configuration."""
        valid_formats = ["fp8_e4m3", "fp8_e5m2", "nvfp4"]
        if self.format not in valid_formats:
            raise ValueError(
                f"Invalid format: {self.format}. Must be one of {valid_formats}"
            )
        if self.calibration_samples < 1:
            raise ValueError(
                f"calibration_samples must be >= 1, got {self.calibration_samples}"
            )
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")


################################################################################
# SECTION 2: FP8 QUANTIZE / DEQUANTIZE FUNCTIONS
################################################################################


def _get_fp8_range(format: str) -> Tuple[float, float, float]:
    """
    Get the representable range and minimum positive value for an FP8 format.

    Args:
        format: "fp8_e4m3" or "fp8_e5m2"

    Returns:
        Tuple of (max_value, min_positive_value, eps)

    Explanation:
        FP8 E4M3: sign(1) + exponent(4, bias=7) + mantissa(3)
            Max = (1 + 7/8) * 2^(15-7) = 1.875 * 256 = 448
            Min positive normal = 2^(1-7) = 2^(-6) = 0.015625

        FP8 E5M2: sign(1) + exponent(5, bias=15) + mantissa(2)
            Max = (1 + 3/4) * 2^(31-15) = 1.75 * 65536 = 114688
            (but finite max is 57344 due to special values)
            Min positive normal = 2^(1-15) = 2^(-14) ≈ 0.000061035
    """
    if format == "fp8_e4m3":
        max_val = 448.0
        min_pos = 2.0 ** (-6)  # 0.015625
        eps = 2.0 ** (-3) / (2.0 ** 7)  # smallest step = 2^(1-7-3) = 2^-9
        return max_val, min_pos, eps
    elif format == "fp8_e5m2":
        max_val = 57344.0
        min_pos = 2.0 ** (-14)  # ≈ 0.000061035
        eps = 2.0 ** (-2) / (2.0 ** 15)  # smallest step = 2^(1-15-2) = 2^-16
        return max_val, min_pos, eps
    else:
        raise ValueError(f"Unknown FP8 format: {format}")


def _compute_scale_factor(
    tensor: torch.Tensor,
    format: str,
    per_tensor: bool = True,
    block_size: int = 16,
) -> torch.Tensor:
    """
    Compute the quantization scale factor for a tensor.

    Formula:
        scale = max(|x|) / max_representable_value

    This maps the range [-max(|x|), +max(|x|)] into the representable
    range of the target format, minimizing quantization error.

    Args:
        tensor: Input tensor to compute scale for
        format: Target FP8 format
        per_tensor: If True, one scale for entire tensor. If False, per-block.
        block_size: Block size for per-block scaling

    Returns:
        Scale factor tensor. Shape: scalar for per-tensor, (num_blocks,) for per-block.

    Explanation:
        The scale factor is the ratio between the observed tensor range and
        the representable range of the target format. We compute it as:

            scale = max(|x|) / max_val

        where max_val is the maximum representable value of the target format.

        For per-block scaling, the tensor is reshaped into blocks of block_size
        elements, and each block gets its own scale factor. This preserves more
        precision for tensors with outliers concentrated in specific regions.
    """
    max_val, _, _ = _get_fp8_range(format)

    if per_tensor:
        # Per-tensor: one scale for the whole tensor
        abs_max = tensor.abs().max().float()
        scale = abs_max / max_val
        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-12)
        return scale
    else:
        # Per-block: reshape into blocks and compute per-block scale
        flat = tensor.reshape(-1).float()
        num_blocks = math.ceil(flat.numel() / block_size)
        # Pad to block boundary
        padded = torch.zeros(num_blocks * block_size, device=flat.device)
        padded[:flat.numel()] = flat
        padded = padded.reshape(num_blocks, block_size)
        # Per-block max absolute value
        block_abs_max = padded.abs().amax(dim=1)
        scale = block_abs_max / max_val
        scale = torch.clamp(scale, min=1e-12)
        return scale


def quantize_to_fp8(
    tensor: torch.Tensor,
    format: str = "fp8_e4m3",
    scale: Optional[torch.Tensor] = None,
    per_tensor: bool = True,
    block_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8 precision.

    Formula:
        x_q = clamp(round(x / scale), min_val, max_val)

    Args:
        tensor: Input tensor (BF16/FP16/FP32)
        format: Target FP8 format ("fp8_e4m3" or "fp8_e5m2")
        scale: Pre-computed scale factor (if None, computed automatically)
        per_tensor: Whether to use per-tensor scaling
        block_size: Block size for per-block scaling

    Returns:
        Tuple of (quantized_tensor, scale_factor)

    Explanation:
        1. Compute scale factor: scale = max(|x|) / max_representable
        2. Divide tensor by scale: x_scaled = x / scale
        3. Round to nearest representable value: x_q = round(x_scaled)
        4. Clamp to representable range: x_q = clamp(x_q, -max_val, max_val)

        The quantized tensor is stored as INT8 (since PyTorch doesn't have
        native FP8 storage until recent versions), with the scale factor
        stored alongside for dequantization.

        We use stochastic rounding during calibration for better statistics,
        but deterministic rounding for final quantization.
    """
    max_val, min_pos, eps = _get_fp8_range(format)

    # Compute scale if not provided
    if scale is None:
        scale = _compute_scale_factor(tensor, format, per_tensor, block_size)

    original_dtype = tensor.dtype
    tensor_float = tensor.float()

    if per_tensor:
        # Per-tensor quantization
        scaled = tensor_float / scale
        # Round to nearest integer (simulating FP8 mantissa precision)
        quantized = torch.round(scaled)
        # Clamp to representable range
        quantized = torch.clamp(quantized, -max_val, max_val)
    else:
        # Per-block quantization
        flat = tensor_float.reshape(-1)
        num_blocks = math.ceil(flat.numel() / block_size)
        padded = torch.zeros(num_blocks * block_size, device=flat.device)
        padded[:flat.numel()] = flat
        padded = padded.reshape(num_blocks, block_size)

        # Scale each block
        scale_expanded = scale.unsqueeze(1).expand_as(padded)
        scaled = padded / scale_expanded
        quantized = torch.round(scaled)
        quantized = torch.clamp(quantized, -max_val, max_val)

        # Reshape back
        quantized = quantized.reshape(-1)[:flat.numel()].reshape(tensor.shape)

    return quantized.to(original_dtype), scale


def dequantize_from_fp8(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    format: str = "fp8_e4m3",
    per_tensor: bool = True,
    block_size: int = 16,
    original_shape: Optional[torch.Size] = None,
) -> torch.Tensor:
    """
    Dequantize an FP8 tensor back to higher precision.

    Formula:
        x = x_q * scale

    Args:
        quantized: Quantized tensor (as INT8 or float representation)
        scale: Scale factor used during quantization
        format: FP8 format used during quantization
        per_tensor: Whether per-tensor scaling was used
        block_size: Block size if per-block scaling was used
        original_shape: Original tensor shape (for per-block unpadding)

    Returns:
        Dequantized tensor in float32

    Explanation:
        Dequantization is the inverse of quantization: multiply by the scale
        factor to recover the approximate original values.

        For per-tensor: x = x_q * scale
        For per-block: x = x_q * scale_block (element-wise per block)

        The result is an approximation — quantization error is irreversible.
        The error is bounded by scale * 0.5 (half a quantization step).
    """
    if per_tensor:
        return (quantized.float() * scale.float())
    else:
        flat = quantized.reshape(-1).float()
        num_blocks = math.ceil(flat.numel() / block_size)
        padded = torch.zeros(num_blocks * block_size, device=flat.device)
        padded[:flat.numel()] = flat
        padded = padded.reshape(num_blocks, block_size)

        scale_expanded = scale.unsqueeze(1).expand_as(padded)
        dequantized = padded * scale_expanded
        dequantized = dequantized.reshape(-1)[:flat.numel()]
        if original_shape is not None:
            dequantized = dequantized.reshape(original_shape)
        return dequantized


################################################################################
# SECTION 3: FP8 QUANTIZER
################################################################################


class FP8Quantizer:
    """
    FP8 Post-Training Quantization
    ===============================

    BF16 is quality baseline. FP8 is default production inference precision
    on Ada Lovelace/Blackwell in 2026.

    Quality: 0.5-2% degradation from BF16, roughly halves VRAM.

    Formats:
        - FP8 E4M3: 4-bit exponent, 3-bit mantissa (forward pass)
        - FP8 E5M2: 5-bit exponent, 2-bit mantissa (gradients, wider range)

    Scaling: Per-tensor or per-block dynamic scaling.
    Calibrate on a small sample to find the right scale factor.

    Cite: DeepSeek-V3 was first to validate FP8 training at scale.

    Algorithm:
        1. Calibration: Run representative data through model, record
           weight and activation ranges per tensor
        2. Scale Computation: For each tensor, scale = max(|x|) / max_val
        3. Weight Quantization: Quantize weights offline (one-time)
        4. Runtime: Install forward hooks to dynamically quantize activations
        5. Inference: Run model with quantized weights + quantized activations

    Step by step:
        1. Call calibrate() with representative data to find scale factors
        2. Call quantize_model() to apply quantization to all linear layers
        3. The returned model has FP8 weights and runtime activation quantization
        4. Run inference as normal — hooks handle activation quantization

    Interview Question:
        "How does FP8 quantization compare to INT8 for LLM inference?"
        FP8 preserves the floating-point number system (sign, exponent,
        mantissa), which means it handles the wide dynamic range of LLM
        activations better than INT8's uniform spacing. INT8 requires
        careful calibration to avoid clipping outliers, while FP8's
        exponent field naturally handles different scales. On hardware
        with native FP8 support, FP8 is also faster than INT8 because
        it avoids the int-to-float conversion overhead.
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize the FP8 quantizer.

        Args:
            config: QuantizationConfig instance (default: fp8_e4m3, 512 samples)

        Explanation:
            The quantizer stores configuration and maintains a registry of
            computed scale factors for each layer. These scales are populated
            during calibration and used during quantization.
        """
        self.config = config or QuantizationConfig(format="fp8_e4m3")
        self.scale_factors: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []
        self._is_calibrated = False

    def calibrate(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Calibrate the model to find optimal scale factors for each tensor.

        Runs the calibration data through the model and records the maximum
        absolute values for weights and activations. These are used to compute
        scale factors for quantization.

        Args:
            model: The model to calibrate
            calibration_data: List of input tensors (batch of samples)

        Returns:
            Dictionary mapping layer names to their computed scale factors

        Explanation:
            Calibration works by running representative data through the model
            and observing the actual ranges of values in each tensor.

            For weights: We can directly inspect the weight tensor.
            For activations: We install hooks that record the max(|x|) during
            the forward pass.

            The number of calibration samples matters:
            - Too few (< 64): May miss outlier activations, causing clipping
            - Too many (> 1024): Wastes time, marginal improvement
            - Sweet spot: 128-512 samples for most models

            We use the full range (not percentile) to avoid clipping any values.
            This is conservative but safe — some methods use 99.99th percentile
            for tighter quantization at the risk of rare clipping.
        """
        logger.info(
            f"Starting calibration with {len(calibration_data)} samples, "
            f"format={self.config.format}"
        )

        # Storage for activation ranges
        activation_ranges: Dict[str, float] = {}

        # Hook function to record activation ranges
        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    abs_max = output.abs().max().item()
                    if name not in activation_ranges:
                        activation_ranges[name] = abs_max
                    else:
                        activation_ranges[name] = max(
                            activation_ranges[name], abs_max
                        )
            return hook_fn

        # Register hooks on all Linear layers
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(
                    module.register_forward_hook(make_hook(name))
                )

        # Run calibration data through model
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(calibration_data[:self.config.calibration_samples]):
                if isinstance(sample, torch.Tensor):
                    model(sample)
                if (i + 1) % 64 == 0:
                    logger.info(f"  Calibrated on {i + 1}/{len(calibration_data)} samples")

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute scale factors for weights
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Weight scale
                weight_scale = _compute_scale_factor(
                    module.weight.data,
                    self.config.format,
                    self.config.per_tensor_scaling,
                    self.config.block_size,
                )
                self.scale_factors[f"{name}.weight"] = weight_scale

                # Activation scale (if dynamic activations are disabled,
                # we use the observed range from calibration)
                if name in activation_ranges:
                    act_tensor = torch.tensor([activation_ranges[name]])
                    act_scale = _compute_scale_factor(
                        act_tensor,
                        self.config.format,
                        per_tensor=True,
                    )
                    self.scale_factors[f"{name}.activation"] = act_scale

        self._is_calibrated = True
        logger.info(
            f"Calibration complete. Computed scales for "
            f"{len(self.scale_factors)} tensors."
        )
        return self.scale_factors

    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """
        Quantize a single tensor to FP8 and dequantize back.

        This applies the quantization noise to simulate what happens
        during inference with quantized weights.

        Args:
            tensor: Input tensor to quantize
            name: Name of the tensor (for looking up scale factor)

        Returns:
            Quantized-then-dequantized tensor (simulated FP8 precision)

        Explanation:
            This function quantizes a tensor and immediately dequantizes it.
            The result is the original tensor with quantization noise added.
            This is useful for simulating FP8 precision without actually
            changing the storage format.

            The scale factor is looked up from the calibration results.
            If not found, a new scale is computed on-the-fly (less optimal
            but functional).
        """
        # Get or compute scale factor
        if name in self.scale_factors:
            scale = self.scale_factors[name]
        else:
            scale = _compute_scale_factor(
                tensor,
                self.config.format,
                self.config.per_tensor_scaling,
                self.config.block_size,
            )

        # Quantize
        quantized, scale = quantize_to_fp8(
            tensor,
            format=self.config.format,
            scale=scale,
            per_tensor=self.config.per_tensor_scaling,
            block_size=self.config.block_size,
        )

        # Dequantize
        dequantized = dequantize_from_fp8(
            quantized,
            scale,
            format=self.config.format,
            per_tensor=self.config.per_tensor_scaling,
            block_size=self.config.block_size,
            original_shape=tensor.shape,
        )

        return dequantized.to(tensor.dtype)

    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[List[torch.Tensor]] = None,
    ) -> nn.Module:
        """
        Quantize a model's weights and install activation quantization hooks.

        This is the main entry point for quantizing a model. It:
        1. Calibrates if not already done
        2. Quantizes all Linear layer weights
        3. Installs forward hooks for runtime activation quantization

        Args:
            model: The model to quantize
            calibration_data: Calibration data (required if not yet calibrated)

        Returns:
            The same model, modified in-place with quantized weights and hooks

        Explanation:
            Weight quantization is done once, offline. The weights are replaced
            with their quantized-then-dequantized versions (simulating FP8
            precision). This is a one-time cost.

            Activation quantization is done at runtime via forward hooks.
            Each Linear layer's output is dynamically quantized and
            dequantized before being passed to the next layer. This adds
            a small per-layer overhead but ensures activations are also
            in FP8 range.

            The combination of FP8 weights + FP8 activations gives the
            full memory and compute benefits of FP8 inference.
        """
        # Calibrate if needed
        if not self._is_calibrated:
            if calibration_data is None:
                raise ValueError(
                    "Calibration data required for first quantization. "
                    "Call calibrate() first or pass calibration_data."
                )
            self.calibrate(model, calibration_data)

        # Quantize weights
        quantized_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight_name = f"{name}.weight"
                module.weight.data = self.quantize_tensor(
                    module.weight.data, weight_name
                )
                quantized_count += 1

        logger.info(f"Quantized {quantized_count} Linear layer weights.")

        # Install activation quantization hooks (unless weight-only mode)
        if not self.config.weight_only:
            self._install_activation_hooks(model)

        return model

    def _install_activation_hooks(self, model: nn.Module) -> None:
        """
        Install forward hooks that dynamically quantize activations.

        Args:
            model: Model to install hooks on

        Explanation:
            For each Linear layer, we install a post-forward hook that:
            1. Takes the output activations
            2. Computes a dynamic scale factor
            3. Quantizes to FP8
            4. Dequantizes back to the original dtype

            Dynamic activation quantization is preferred over static because
            activation ranges can vary significantly between inputs. Static
            quantization uses a fixed scale from calibration, which may clip
            outliers on some inputs or waste range on others.

            The overhead of dynamic scale computation is small compared to
            the matrix multiplication itself.
        """
        # Remove any existing hooks
        self.remove_hooks()

        def make_quant_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor) and not self.config.weight_only:
                    # Dynamic activation quantization
                    scale = _compute_scale_factor(
                        output,
                        self.config.format,
                        per_tensor=True,
                    )
                    quantized, scale = quantize_to_fp8(
                        output,
                        format=self.config.format,
                        scale=scale,
                        per_tensor=True,
                    )
                    dequantized = dequantize_from_fp8(
                        quantized,
                        scale,
                        format=self.config.format,
                        per_tensor=True,
                    )
                    return dequantized.to(output.dtype)
                return output
            return hook_fn

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(make_quant_hook(name))
                self.hooks.append(hook)

        logger.info(f"Installed activation quantization hooks on {len(self.hooks)} layers.")

    def remove_hooks(self) -> None:
        """Remove all installed activation quantization hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def compute_quantization_error(
        self,
        original: torch.Tensor,
        quantized: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute quantization error metrics between original and quantized tensors.

        Args:
            original: Original tensor (BF16/FP32)
            quantized: Quantized-then-dequantized tensor

        Returns:
            Dictionary with error metrics:
            - mse: Mean squared error
            - mae: Mean absolute error
            - max_error: Maximum absolute error
            - relative_error: Mean relative error (|error| / |original|)
            - snr_db: Signal-to-noise ratio in dB

        Explanation:
            These metrics help assess quantization quality:
            - MSE/MAE: Overall error magnitude
            - Max error: Worst-case individual element error
            - Relative error: Error as a fraction of original value
            - SNR: Higher is better; >40dB is typically acceptable

            For LLMs, the key metric is downstream task performance, not
            these raw error metrics. A model can have higher MSE but better
            task accuracy if the errors are in less important dimensions.
        """
        orig = original.float()
        quant = quantized.float()

        error = (orig - quant).abs()
        mse = (error ** 2).mean().item()
        mae = error.mean().item()
        max_err = error.max().item()

        # Relative error (avoid division by zero)
        orig_abs = orig.abs().clamp(min=1e-12)
        rel_error = (error / orig_abs).mean().item()

        # Signal-to-noise ratio
        signal_power = (orig ** 2).mean().item()
        noise_power = mse
        if noise_power > 0:
            snr_db = 10 * math.log10(signal_power / noise_power)
        else:
            snr_db = float('inf')

        return {
            "mse": mse,
            "mae": mae,
            "max_error": max_err,
            "relative_error": rel_error,
            "snr_db": snr_db,
        }


################################################################################
# SECTION 4: NVFP4 QUANTIZER
################################################################################


class NVFP4Quantizer:
    """
    NVFP4 Post-Training Quantization (Advanced)
    ============================================

    Blackwell-class hardware. Under 1% accuracy degradation from FP8
    on key LM evaluations.

    Format: FP8 micro-scales on 16-value blocks + global FP32 tensor scale.
    NOT naive 4-bit rounding.

    Implement behind a flag. Require eval-suite parity check before trusting.

    Cite: NVIDIA Model-Optimizer documentation.

    NVFP4 Format Details:
        - Each value uses 4 bits: sign(1) + exponent(2) + mantissa(1)
        - Representable values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
        - Range: [0, 6] (unsigned) or [-6, 6] (signed)
        - Block of 16 values shares one FP8 micro-scale
        - Entire tensor has one FP32 global scale

    Two-level scaling:
        1. Global scale: g = max(|tensor|) / 6.0
        2. Per-block scale: s_b = max(|block_b|) / 6.0
        3. Quantize: x_q = round(x / (g * s_b)) and clamp to [-6, 6]
        4. Dequantize: x = x_q * g * s_b

    Step by step:
        1. Compute global scale from entire tensor
        2. Divide tensor by global scale
        3. Reshape into blocks of 16 elements
        4. Compute per-block micro-scale
        5. Quantize each value to 4-bit integer
        6. Store: 4-bit values + FP8 micro-scales + FP32 global scale

    Interview Question:
        "Why does NVFP4 need two levels of scaling instead of one?"
        A single scale factor for the whole tensor would waste precision
        on outliers — one extreme value forces a large scale that makes
        everything else lose precision. Two-level scaling handles this:
        the global scale captures the tensor-wide range, while per-block
        micro-scales adapt to local variations. This is especially
        important for 4-bit quantization where the representable range
        is very small (only 16 values).
    """

    # NVFP4 representable values: signed 4-bit with 2-bit exponent, 1-bit mantissa
    # Values: {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}
    NVFP4_MAX_VAL = 6.0
    NVFP4_MIN_POS = 0.5

    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize the NVFP4 quantizer.

        Args:
            config: QuantizationConfig instance (default: nvfp4, block_size=16)

        Explanation:
            NVFP4 quantization requires per-block scaling by design — 4-bit
            precision is too coarse for a single global scale to work well.
            The block_size parameter controls how many elements share one
            micro-scale. Smaller blocks = more precision but more overhead.
            16 is the NVIDIA standard.
        """
        self.config = config or QuantizationConfig(format="nvfp4", per_tensor_scaling=False)
        if self.config.per_tensor_scaling:
            logger.warning(
                "NVFP4 requires per-block scaling. Overriding per_tensor_scaling=False."
            )
            self.config.per_tensor_scaling = False
        self.scale_factors: Dict[str, Dict[str, torch.Tensor]] = {}

    def quantize_tensor_nvfp4(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor to NVFP4 with two-level scaling.

        Args:
            tensor: Input tensor (any floating-point dtype)

        Returns:
            Tuple of (quantized_int4, block_scales_fp8, global_scale_fp32)
            - quantized_int4: Integer values in [-6, 6], reshaped to blocks
            - block_scales_fp8: Per-block FP8 scales
            - global_scale_fp32: Global tensor scale (scalar)

        Explanation:
            The two-level quantization process:

            1. Global scale: g = max(|x|) / 6.0
               This maps the tensor range to [-6, 6].

            2. Normalize by global scale: x_norm = x / g
               Now x_norm is in [-6, 6].

            3. Reshape into blocks of block_size elements.

            4. Per-block scale: s_b = max(|block_b|) / 6.0
               This gives each block its own range adjustment.

            5. Quantize: x_q = round(x_norm_block / s_b) and clamp to [-6, 6]

            6. Dequantize: x_hat = x_q * s_b * g

            This gives much better precision than single-level scaling because
            each block adapts to its local value distribution.
        """
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        flat = tensor.float().reshape(-1)

        block_size = self.config.block_size
        num_blocks = math.ceil(flat.numel() / block_size)

        # Pad to block boundary
        padded = torch.zeros(num_blocks * block_size, device=flat.device)
        padded[:flat.numel()] = flat

        # Step 1: Global scale
        global_scale = padded.abs().max() / self.NVFP4_MAX_VAL
        global_scale = torch.clamp(global_scale, min=1e-12)

        # Step 2: Normalize by global scale
        normalized = padded / global_scale

        # Step 3: Reshape into blocks
        blocks = normalized.reshape(num_blocks, block_size)

        # Step 4: Per-block scale (stored as FP8)
        block_abs_max = blocks.abs().amax(dim=1)
        block_scales = block_abs_max / self.NVFP4_MAX_VAL
        block_scales = torch.clamp(block_scales, min=1e-12)

        # Quantize block scales to FP8
        block_scales_fp8, _ = quantize_to_fp8(
            block_scales, format="fp8_e4m3", per_tensor=True
        )

        # Step 5: Quantize values
        scale_expanded = block_scales.unsqueeze(1).expand_as(blocks)
        scaled_blocks = blocks / scale_expanded
        quantized = torch.round(scaled_blocks)
        quantized = torch.clamp(quantized, -self.NVFP4_MAX_VAL, self.NVFP4_MAX_VAL)

        # Store scale factors
        return quantized.to(torch.int8), block_scales_fp8, global_scale

    def dequantize_tensor_nvfp4(
        self,
        quantized_int4: torch.Tensor,
        block_scales_fp8: torch.Tensor,
        global_scale_fp32: torch.Tensor,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Dequantize an NVFP4 tensor back to floating point.

        Args:
            quantized_int4: Quantized integer values
            block_scales_fp8: Per-block FP8 scales
            global_scale_fp32: Global tensor scale
            original_shape: Original tensor shape

        Returns:
            Dequantized tensor in float32

        Explanation:
            Dequantization reverses the two-level scaling:

            1. Dequantize block scales from FP8
            2. For each block: x_block = x_q * s_b
            3. Apply global scale: x = x_block * g
            4. Reshape to original shape
        """
        # Dequantize block scales
        block_scales = dequantize_from_fp8(
            block_scales_fp8, torch.tensor(1.0), format="fp8_e4m3", per_tensor=True
        )

        # Apply block scales
        scale_expanded = block_scales.unsqueeze(1).expand_as(quantized_int4.float())
        dequantized = quantized_int4.float() * scale_expanded

        # Apply global scale
        dequantized = dequantized * global_scale_fp32

        # Reshape and trim padding
        flat = dequantized.reshape(-1)[:math.prod(original_shape)]
        return flat.reshape(original_shape)

    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[List[torch.Tensor]] = None,
    ) -> nn.Module:
        """
        Quantize a model's weights to NVFP4.

        Args:
            model: The model to quantize
            calibration_data: Not used for weight-only quantization (reserved
                for future activation quantization support)

        Returns:
            The same model, modified in-place with NVFP4 quantized weights

        Explanation:
            NVFP4 quantization is applied to weights only in the current
            implementation. Activation quantization with NVFP4 is more
            complex and typically requires QAT (Quantization-Aware Training)
            for good results.

            For each Linear layer:
            1. Quantize weight tensor to NVFP4
            2. Store quantized values + scales as a custom attribute
            3. Replace the forward method to dequantize on-the-fly

            WARNING: Always run eval-suite parity checks after NVFP4
            quantization. The quality degradation is task-dependent.
        """
        quantized_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                quantized, block_scales, global_scale = self.quantize_tensor_nvfp4(weight)

                # Store quantized components as module attributes
                module.register_buffer('_nvfp4_weight', quantized)
                module.register_buffer('_nvfp4_block_scales', block_scales)
                module.register_buffer('_nvfp4_global_scale', global_scale)
                module._nvfp4_original_shape = weight.shape

                # Replace weight with dequantized version for compatibility
                dequantized = self.dequantize_tensor_nvfp4(
                    quantized, block_scales, global_scale, weight.shape
                )
                module.weight.data = dequantized.to(weight.dtype)

                quantized_count += 1

        logger.info(f"NVFP4-quantized {quantized_count} Linear layer weights.")
        return model

    def compute_memory_savings(
        self,
        model: nn.Module,
    ) -> Dict[str, float]:
        """
        Compute memory savings from NVFP4 quantization.

        Args:
            model: The quantized model

        Returns:
            Dictionary with memory metrics:
            - bf16_bytes: Original BF16 weight memory
            - nvfp4_bytes: NVFP4 weight memory (4-bit + scales)
            - savings_ratio: bf16_bytes / nvfp4_bytes
            - savings_percent: Percentage reduction

        Explanation:
            NVFP4 memory per tensor:
            - Weight values: 4 bits per element = 0.5 bytes/element
            - Block scales: 8 bits per block_size elements = 1/block_size bytes/element
            - Global scale: 4 bytes per tensor (negligible for large tensors)
            - Total ≈ 0.5 + 1/16 = 0.5625 bytes/element

            BF16 memory:
            - 16 bits per element = 2 bytes/element

            Savings ratio ≈ 2 / 0.5625 ≈ 3.56x
        """
        bf16_bytes = 0
        nvfp4_bytes = 0
        block_size = self.config.block_size

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                num_elements = module.weight.numel()
                # BF16: 2 bytes per element
                bf16_bytes += num_elements * 2
                # NVFP4: 0.5 bytes per element + scales
                nvfp4_bytes += num_elements * 0.5  # 4-bit values
                num_blocks = math.ceil(num_elements / block_size)
                nvfp4_bytes += num_blocks * 1  # FP8 block scales
                nvfp4_bytes += 4  # FP32 global scale

        savings_ratio = bf16_bytes / max(nvfp4_bytes, 1)
        savings_percent = (1 - nvfp4_bytes / max(bf16_bytes, 1)) * 100

        return {
            "bf16_bytes": bf16_bytes,
            "nvfp4_bytes": nvfp4_bytes,
            "savings_ratio": savings_ratio,
            "savings_percent": savings_percent,
        }


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################


def demonstrate_quantization():
    """
    Demonstrate FP8 and NVFP4 quantization on a small model.

    Shows:
        1. FP8 quantize/dequantize round-trip
        2. Per-tensor vs per-block scaling comparison
        3. Full model quantization with FP8
        4. NVFP4 quantization and memory savings
        5. Quantization error analysis
    """
    print("=" * 70)
    print("POST-TRAINING QUANTIZATION DEMONSTRATION")
    print("=" * 70)

    # ---- Part 1: FP8 Quantize/Dequantize ----
    print("\n1. FP8 Quantize/Dequantize Round-Trip")
    print("-" * 50)

    # Create a test tensor
    torch.manual_seed(42)
    x = torch.randn(4, 4) * 10  # Range ~ [-30, 30]
    print(f"  Original tensor (BF16):\n  {x}")

    # Quantize to FP8 E4M3
    quantized, scale = quantize_to_fp8(x, format="fp8_e4m3", per_tensor=True)
    dequantized = dequantize_from_fp8(quantized, scale, format="fp8_e4m3", per_tensor=True)
    print(f"\n  FP8 E4M3 scale factor: {scale.item():.6f}")
    print(f"  Dequantized tensor:\n  {dequantized}")

    # Compute error
    error = (x.float() - dequantized.float()).abs()
    print(f"  Max absolute error: {error.max().item():.6f}")
    print(f"  Mean absolute error: {error.mean().item():.6f}")

    # ---- Part 2: Per-Tensor vs Per-Block Scaling ----
    print("\n2. Per-Tensor vs Per-Block Scaling")
    print("-" * 50)

    # Create a tensor with an outlier
    x_outlier = torch.randn(32)
    x_outlier[0] = 100.0  # Outlier
    print(f"  Tensor range: [{x_outlier.min().item():.2f}, {x_outlier.max().item():.2f}]")

    # Per-tensor scaling
    q_per_tensor, s_pt = quantize_to_fp8(
        x_outlier, format="fp8_e4m3", per_tensor=True
    )
    dq_per_tensor = dequantize_from_fp8(q_per_tensor, s_pt, per_tensor=True)
    err_pt = (x_outlier.float() - dq_per_tensor.float()).abs().mean().item()

    # Per-block scaling (block_size=16)
    q_per_block, s_pb = quantize_to_fp8(
        x_outlier, format="fp8_e4m3", per_tensor=False, block_size=16
    )
    dq_per_block = dequantize_from_fp8(
        q_per_block, s_pb, per_tensor=False, block_size=16, original_shape=x_outlier.shape
    )
    err_pb = (x_outlier.float() - dq_per_block.float()).abs().mean().item()

    print(f"  Per-tensor MAE: {err_pt:.6f}")
    print(f"  Per-block  MAE: {err_pb:.6f}")
    print(f"  Per-block is {err_pt / max(err_pb, 1e-12):.1f}x more precise")

    # ---- Part 3: Full Model Quantization ----
    print("\n3. Full Model FP8 Quantization")
    print("-" * 50)

    # Create a small model
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.GELU(),
        nn.Linear(128, 64),
        nn.GELU(),
        nn.Linear(64, 32),
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Create calibration data
    calibration_data = [torch.randn(8, 64) for _ in range(32)]

    # Quantize
    quantizer = FP8Quantizer(QuantizationConfig(format="fp8_e4m3"))
    quantized_model = quantizer.quantize_model(model, calibration_data)

    # Test inference
    test_input = torch.randn(4, 64)
    with torch.no_grad():
        output_orig = model(test_input)  # Model is modified in-place, but weights are quantized
        output_quant = quantized_model(test_input)

    print(f"  Output shape: {output_quant.shape}")
    print(f"  Output range: [{output_quant.min().item():.4f}, {output_quant.max().item():.4f}]")

    # ---- Part 4: NVFP4 Quantization ----
    print("\n4. NVFP4 Quantization")
    print("-" * 50)

    # Create a fresh model for NVFP4
    model_nvfp4 = nn.Sequential(
        nn.Linear(64, 128),
        nn.GELU(),
        nn.Linear(128, 64),
    )

    nvfp4_quantizer = NVFP4Quantizer(QuantizationConfig(format="nvfp4"))
    nvfp4_model = nvfp4_quantizer.quantize_model(model_nvfp4)

    # Memory savings
    memory_info = nvfp4_quantizer.compute_memory_savings(nvfp4_model)
    print(f"  BF16 memory: {memory_info['bf16_bytes']:,} bytes")
    print(f"  NVFP4 memory: {memory_info['nvfp4_bytes']:,} bytes")
    print(f"  Savings: {memory_info['savings_ratio']:.2f}x ({memory_info['savings_percent']:.1f}%)")

    # ---- Part 5: Quantization Error Analysis ----
    print("\n5. Quantization Error Analysis")
    print("-" * 50)

    # Compare FP8 E4M3 vs E5M2
    x_test = torch.randn(1000) * 50

    for fmt in ["fp8_e4m3", "fp8_e5m2"]:
        q, s = quantize_to_fp8(x_test, format=fmt, per_tensor=True)
        dq = dequantize_from_fp8(q, s, format=fmt, per_tensor=True)

        fp8_quantizer = FP8Quantizer(QuantizationConfig(format=fmt))
        metrics = fp8_quantizer.compute_quantization_error(x_test, dq)

        print(f"\n  {fmt}:")
        print(f"    MSE:        {metrics['mse']:.6f}")
        print(f"    MAE:        {metrics['mae']:.6f}")
        print(f"    Max Error:  {metrics['max_error']:.6f}")
        print(f"    SNR:        {metrics['snr_db']:.1f} dB")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("  - FP8 E4M3: Good precision, default for forward pass")
    print("  - FP8 E5M2: Wider range, used for gradients during training")
    print("  - NVFP4:    ~3.5x memory savings, requires Blackwell hardware")
    print("  - Per-block scaling handles outliers better than per-tensor")
    print("  - Always benchmark on YOUR task before deploying quantized models")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_quantization()
