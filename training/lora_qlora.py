"""
################################################################################
LoRA & QLoRA — PARAMETER-EFFICIENT FINE-TUNING (2023-2025 SOTA)
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is LoRA?
    LoRA (Low-Rank Adaptation) is a method for fine-tuning large models
    by adding small trainable matrices to frozen weights. Instead of
    updating all parameters (billions), LoRA only trains ~0.1% of them.

    Original weight: W ∈ R^{d×d} (frozen)
    LoRA update: ΔW = A × B where A ∈ R^{d×r}, B ∈ R^{r×d}, r << d
    Final: W' = W + ΔW

What is QLoRA?
    QLoRA = Quantized LoRA. It combines:
    1. 4-bit quantization of the base model (saves memory)
    2. LoRA adapters in 16-bit (trainable)
    3. Double quantization (quantize the quantization constants)
    4. Paged optimizers (handle memory spikes)

    Result: Fine-tune a 65B model on a single 48GB GPU!

Why LoRA/QLoRA?
    - Full fine-tuning: needs GPU memory for all parameters + gradients + optimizer states
    - LoRA: needs memory only for small adapters (0.1% of parameters)
    - QLoRA: further reduces by quantizing the frozen base model

    Example (7B model):
    - Full FT: ~56GB GPU memory
    - LoRA: ~16GB GPU memory
    - QLoRA: ~6GB GPU memory (!)

Interview Questions:
    Q: "What is LoRA and why does it work?"
    A: LoRA adds low-rank matrices (A×B) to frozen weights. It works
       because fine-tuning mostly updates a low-dimensional subspace.
       Instead of updating W (d×d), we update A (d×r) and B (r×d)
       where r << d. This reduces trainable parameters by 1000x+.

    Q: "How does QLoRA improve on LoRA?"
    A: QLoRA quantizes the base model to 4-bit, reducing memory by 4x.
       It uses double quantization (quantize the quantization constants)
       and paged optimizers for memory spikes. Result: fine-tune 65B
       on a single 48GB GPU.

    Q: "When would you use LoRA vs full fine-tuning?"
    A: LoRA when: limited GPU memory, quick iteration, multiple task
       adapters. Full FT when: maximum quality needed, sufficient
       compute, domain-specific pre-training.

################################################################################
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

################################################################################
# SECTION 1: LoRA LAYER
################################################################################

@dataclass
class LoRAConfig:
    """LoRA configuration."""
    rank: int = 8              # Rank of low-rank matrices (r)
    alpha: float = 16.0        # Scaling factor (alpha/r)
    dropout: float = 0.05      # Dropout on LoRA layers
    target_modules: List[str] = None  # Which modules to apply LoRA to

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class LoRALayer:
    """
    LoRA (Low-Rank Adaptation) Layer
    ==================================

    Adds a low-rank update to a frozen weight matrix.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    LoRA Layer                                │
    │                                                              │
    │  Input x                                                     │
    │     │                                                        │
    │     ├──▶ Frozen W (no gradient) ──▶ Wx                      │
    │     │                                                        │
    │     ├──▶ LoRA A (trainable) ──▶ Ax                          │
    │     │         │                                              │
    │     │         ▼                                              │
    │     └──▶ LoRA B (trainable) ──▶ BAx                        │
    │                                                              │
    │  Output = Wx + (alpha/r) * BAx                              │
    └─────────────────────────────────────────────────────────────┘

    Key insight: The gradient only flows through A and B,
    not through the frozen W. This is why LoRA is memory-efficient.

    Interview Question:
        Q: "Why does LoRA use low-rank matrices?"
        A: The weight updates during fine-tuning tend to be low-rank
           (most of the change happens in a small subspace). LoRA
           exploits this by learning only the important directions
           of change, reducing parameters by 1000x+.
    """

    def __init__(self, d_in: int, d_out: int, config: LoRAConfig = None):
        self.config = config or LoRAConfig()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = self.config.rank

        # Frozen weight (pretrained)
        self.W = np.random.randn(d_in, d_out) * 0.02  # Frozen

        # LoRA matrices (trainable)
        self.A = np.random.randn(d_in, self.rank) * 0.01  # Low-rank
        self.B = np.zeros((self.rank, d_out))               # Init to zero

        # Scaling factor
        self.scaling = self.config.alpha / self.rank

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: output = Wx + (alpha/r) * B @ A @ x

        Args:
            x: [batch × d_in] input

        Returns:
            output: [batch × d_out]
        """
        # Frozen path
        frozen_out = x @ self.W

        # LoRA path
        lora_out = x @ self.A @ self.B
        lora_out = lora_out * self.scaling

        return frozen_out + lora_out

    def get_trainable_params(self) -> Dict[str, np.ndarray]:
        """Get only trainable parameters (A and B)."""
        return {"A": self.A, "B": self.B}

    def merge_weights(self) -> np.ndarray:
        """
        Merge LoRA weights into the frozen weight.

        W_merged = W + (alpha/r) * B @ A

        This eliminates the LoRA overhead at inference time.
        """
        return self.W + self.scaling * (self.A @ self.B)


################################################################################
# SECTION 2: QLoRA (QUANTIZED LoRA)
################################################################################

class QLoRALayer:
    """
    QLoRA — Quantized LoRA
    ========================

    Combines 4-bit quantization with LoRA for extreme memory efficiency.

    Memory savings:
    - Original FP32: 4 bytes per parameter
    - Original FP16: 2 bytes per parameter
    - QLoRA 4-bit: 0.5 bytes per parameter + small LoRA matrices

    For a 7B model:
    - FP32: 28GB
    - FP16: 14GB
    - QLoRA: ~3.5GB base + ~0.1GB LoRA = ~3.6GB total!

    Interview Question:
        Q: "How does QLoRA achieve such memory savings?"
        A: Three techniques: (1) 4-bit NormalFloat quantization —
           more efficient than regular 4-bit for normal distributions,
           (2) Double quantization — quantize the quantization constants,
           (3) Paged optimizers — use CPU memory for optimizer states.
    """

    def __init__(self, d_in: int, d_out: int, config: LoRAConfig = None):
        self.config = config or LoRAConfig()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = self.config.rank

        # Quantized frozen weight (4-bit simulation)
        # In production: use bitsandbytes or similar
        self.W_quantized = np.random.randn(d_in, d_out) * 0.02
        self.quantization_scale = 1.0  # Simulated

        # LoRA matrices in FP16 (trainable)
        self.A = np.random.randn(d_in, self.rank) * 0.01
        self.B = np.zeros((self.rank, self.d_out))

        self.scaling = self.config.alpha / self.rank

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with quantized base + LoRA.

        Same as LoRA but the base weight is quantized.
        """
        # Dequantize and compute (simulated)
        frozen_out = x @ self.W_quantized

        # LoRA path (in FP16)
        lora_out = x @ self.A @ self.B * self.scaling

        return frozen_out + lora_out

    def get_memory_usage(self) -> Dict:
        """Calculate memory usage."""
        base_params = self.d_in * self.d_out
        lora_params = (self.d_in + self.d_out) * self.rank

        return {
            "base_params_fp32": base_params * 4,  # bytes
            "base_params_4bit": base_params * 0.5,
            "lora_params_fp16": lora_params * 2,
            "total_qlora": base_params * 0.5 + lora_params * 2,
            "savings_ratio": (base_params * 4) / (base_params * 0.5 + lora_params * 2),
        }


################################################################################
# SECTION 3: LoRA MODEL WRAPPER
################################################################################

class LoRAModel:
    """
    Apply LoRA to a complete model.

    Replaces target modules with LoRA versions while keeping
    all other parameters frozen.

    Interview Question:
        Q: "Which layers should you apply LoRA to?"
        A: Typically attention projections (Q, K, V, O) give the best
           results. Some papers also apply to MLP layers. The rank
           matters more than which layers — rank 8-16 is usually enough.
    """

    def __init__(self, base_model, config: LoRAConfig = None):
        self.base_model = base_model
        self.config = config or LoRAConfig()
        self.lora_layers = {}

    def add_lora(self, layer_name: str, d_in: int, d_out: int):
        """Add LoRA adapter to a specific layer."""
        self.lora_layers[layer_name] = LoRALayer(d_in, d_out, self.config)

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        total = 0
        for layer in self.lora_layers.values():
            total += layer.A.size + layer.B.size
        return total

    def merge_and_save(self) -> Dict[str, np.ndarray]:
        """Merge all LoRA weights and return merged model."""
        merged = {}
        for name, layer in self.lora_layers.items():
            merged[name] = layer.merge_weights()
        return merged


################################################################################
# SECTION 4: TESTING AND DEMONSTRATION
################################################################################

def demonstrate_lora():
    """Comprehensive LoRA/QLoRA demonstration."""
    print("=" * 70)
    print("LoRA & QLoRA DEMONSTRATION")
    print("=" * 70)

    # === Demo 1: LoRA Layer ===
    print("\n--- Demo 1: LoRA Layer ---")
    config = LoRAConfig(rank=8, alpha=16)
    lora = LoRALayer(d_in=256, d_out=256, config=config)

    x = np.random.randn(4, 256)
    output = lora.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable params: {lora.A.size + lora.B.size}")
    print(f"Total params: {lora.W.size}")
    print(f"LoRA ratio: {(lora.A.size + lora.B.size) / lora.W.size:.2%}")

    # === Demo 2: QLoRA Memory ===
    print("\n--- Demo 2: QLoRA Memory Savings ---")
    qlora = QLoRALayer(d_in=4096, d_out=4096)
    mem = qlora.get_memory_usage()
    print(f"Base FP32: {mem['base_params_fp32'] / 1e6:.1f} MB")
    print(f"Base 4-bit: {mem['base_params_4bit'] / 1e6:.1f} MB")
    print(f"LoRA FP16: {mem['lora_params_fp16'] / 1e6:.1f} MB")
    print(f"Total QLoRA: {mem['total_qlora'] / 1e6:.1f} MB")
    print(f"Savings: {mem['savings_ratio']:.1f}x")

    # === Demo 3: Model Size Comparison ===
    print("\n--- Demo 3: Model Size Comparison ---")
    model_sizes = [7, 13, 33, 65, 70]
    print(f"{'Model':<10} {'FP32':<10} {'FP16':<10} {'LoRA':<10} {'QLoRA':<10}")
    print("-" * 50)
    for size in model_sizes:
        params = size * 1e9
        fp32 = params * 4 / 1e9
        fp16 = params * 2 / 1e9
        lora = (params * 0.001 * 2 + params * 4 * 0.001) / 1e9  # 0.1% params
        qlora = (params * 0.5 + params * 0.001 * 2) / 1e9
        print(f"{size}B{'':<7} {fp32:<10.0f} {fp16:<10.0f} {lora:<10.1f} {qlora:<10.1f}")

    # === Demo 4: Rank Selection ===
    print("\n--- Demo 4: Rank Selection Guide ---")
    print(f"{'Rank':<8} {'Params':<12} {'Quality':<12} {'Use Case':<20}")
    print("-" * 52)
    print(f"{'1':<8} {'Minimal':<12} {'Basic':<12} {'Quick prototyping':<20}")
    print(f"{'4':<8} {'Small':<12} {'Good':<12} {'Most tasks':<20}")
    print(f"{'8':<8} {'Medium':<12} {'Better':<12} {'Complex tasks':<20}")
    print(f"{'16':<8} {'Large':<12} {'Best':<12} {'Maximum quality':<20}")
    print(f"{'64':<8} {'Very Large':<12} {'Near FT':<12} {'When LoRA not enough':<20}")

    print("\n" + "=" * 70)
    print("All LoRA/QLoRA demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_lora()


################################################################################
# REFERENCES
################################################################################

# [1] Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language
#     Models. arXiv:2106.09685.
#
# [2] Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized
#     Language Models. arXiv:2305.14314.
#
# [3] Mangrulkar, S., et al. (2022). PEFT: State-of-the-art Parameter-
#     Efficient Fine-Tuning methods. https://github.com/huggingface/peft.

################################################################################
