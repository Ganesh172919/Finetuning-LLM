"""
################################################################################
MU-P INITIALIZATION — MAXIMAL UPDATE PARAMETRIZATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is muP (Maximal Update Parametrization)?
    muP is a set of rules for initializing neural network parameters and
    setting per-layer learning rate multipliers such that hyperparameters
    tuned on a small "base" model transfer optimally to a much larger model.

    The key insight: as you scale model width (d_model), the optimal
    learning rate and initialization variance for each layer change in a
    predictable way. muP prescribes these scaling rules so you can tune
    hyperparameters on a cheap nano/small model and apply them to a
    large model with confidence.

Why does it matter?
    Hyperparameter tuning is the most expensive part of LLM training.
    Without muP, every new model size requires a full hyperparameter sweep.
    With muP, you sweep on a small model (orders of magnitude cheaper) and
    transfer the optimal hyperparameters to the large model using a
    documented scaling rule.

    This is especially important for MoE models where the "effective width"
    includes the number of experts, and for the SOTA LLM Forge project
    where we define tiers (nano → small-MoE → large-MoE) that should
    share tuned hyperparameters.

How does it work?
    1. Define a "base width" d_base (the width of the model you tune on).
    2. For the actual model with width d_model, compute the width ratio
       alpha = d_model / d_base.
    3. Scale initialization variance and learning rates per layer type:
       - Embedding: init std = 1, LR multiplier = 1
       - Hidden layers (attention QKV, FFN up): init std = 1/sqrt(fan_in),
         LR multiplier = 1/alpha
       - Output layers (attention out, FFN down, LM head): init std =
         1/sqrt(fan_out), LR multiplier = 1/alpha^2
       - LayerNorm/RMSNorm: standard init, LR = 1
    4. The output layer (LM head) uses a special "zero init" for the bias
       and a small init for the weight.

########################################

ARCHITECTURE DIAGRAM (ASCII art):
    ┌──────────────────────────────────────────────────────────────┐
    │                    muP INITIALIZATION                         │
    │                                                              │
    │  Base model (width d_base):  sweep hyperparams               │
    │       ↓                                                      │
    │  Width ratio: alpha = d_model / d_base                       │
    │       ↓                                                      │
    │  For each parameter:                                         │
    │    Embedding:      std = 1,           LR_mult = 1            │
    │    Hidden (QKV/Up): std = 1/√fan_in, LR_mult = 1/α          │
    │    Output (Out/Dn): std = 1/√fan_out, LR_mult = 1/α²        │
    │    Norm:           std = 1,           LR_mult = 1            │
    │    LM Head bias:   std = 0,           LR_mult = 1            │
    │                                                              │
    │  Learning rate schedule:                                     │
    │    param_lr = base_lr * layer_multiplier                     │
    └──────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2020: "Tensor Programs I-IV" — Theoretical foundations (Yang et al.)
    - 2022: "Tensor Programs V: Tuning Large Neural Networks via
             Zero-Shot Hyperparameter Transfer" — muP paper (Yang et al.)
    - 2023: Cerebras-GPT — First large model trained with muP (Cerebras)
    - 2024: Practical adoption in research labs and open-source projects

INTERVIEW QUESTIONS:
    1. "How does muP differ from standard PyTorch initialization (e.g., Xavier)?"
       Xavier/He initialization optimizes for a single model's forward/backward
       signal propagation. muP optimizes for hyperparameter TRANSFER across
       model sizes. Xavier gives the same init regardless of width; muP
       explicitly depends on the width ratio alpha. The goal is different:
       muP wants the optimal learning rate to be width-independent.

    2. "What is the 'feature learning' regime and why does muP target it?"
       In the infinite-width limit, neural networks behave like linear models
       (kernel regime). muP ensures that as width increases, the network
       stays in the "feature learning" regime where hidden representations
       are actively updated. This is achieved by scaling the learning rate
       inversely with width for hidden layers, keeping the update magnitude
       constant regardless of width.

    3. "Can you use muP with Adam optimizer?"
       Yes, muP is compatible with Adam and most common optimizers. The
       learning rate multipliers are applied to the per-parameter learning
       rate in Adam. The key is that the base learning rate (tuned on the
       small model) is shared across all layers, and the multipliers handle
       the per-layer scaling.

################################################################################
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional
from dataclasses import dataclass


################################################################################
# SECTION 1: MU-P CONFIGURATION
################################################################################


@dataclass
class muPConfig:
    """
    Configuration for muP initialization.

    Attributes:
        d_base: Width of the base model used for hyperparameter tuning.
        d_model: Width of the actual model to initialize.
        base_lr: Base learning rate (tuned on base model).
        output_mult: Multiplier for output layer LR (default 1.0 —
                     some implementations use different values).
        embedding_mult: Multiplier for embedding LR (default 1.0).
        query_embedding_mult: Special multiplier for Q projection in attention.
                              Default: 1/d_model (very conservative).
    """

    d_base: int = 128
    d_model: int = 768
    base_lr: float = 3e-4
    output_mult: float = 1.0
    embedding_mult: float = 1.0
    query_embedding_mult: Optional[float] = None  # Default: 1/d_model


################################################################################
# SECTION 2: MU-P INITIALIZER
################################################################################


class muPInitializer:
    """
    Maximal Update Parametrization (muP) Initializer
    =================================================

    Applies muP initialization and computes per-layer learning rate
    multipliers for a transformer model.

    Cite: Yang et al. 2022, "Tensor Programs V: Tuning Large Neural
          Networks via Zero-Shot Hyperparameter Transfer"

    Initialization Rules:
        Let alpha = d_model / d_base (width ratio).

        Layer Type        | Init Std              | LR Multiplier
        ------------------|-----------------------|---------------
        Embedding         | 1                     | 1
        Attention QKV     | 1 / sqrt(fan_in)      | 1 / alpha
        Attention Output  | 1 / sqrt(fan_out)     | 1 / alpha^2
        FFN Up/Gate       | 1 / sqrt(fan_in)      | 1 / alpha
        FFN Down          | 1 / sqrt(fan_out)     | 1 / alpha^2
        LM Head           | 1 / sqrt(fan_out)     | 1 / alpha^2
        LM Head Bias      | 0                     | 1
        RMSNorm           | 1 (standard)          | 1

    Step by step:
        1. Compute alpha = d_model / d_base.
        2. Walk through all model parameters.
        3. Classify each parameter by its layer type (hidden vs output).
        4. Apply the appropriate initialization variance.
        5. Return a dictionary mapping parameter names to LR multipliers.

    WHY this matters:
        Without muP, hyperparameter tuning must be repeated for every model
        size. With muP, you tune once on a small model and transfer. This
        can save 10-100x in compute for hyperparameter search.

    Interview Question:
        "How do you determine whether a linear layer is 'hidden' or 'output'
        for muP purposes?"
        A "hidden" layer is one whose output feeds into another layer
        (e.g., attention QKV projections, FFN up/gate). An "output" layer
        is one whose output contributes directly to the final result or
        feeds into a normalization/residual (e.g., attention output
        projection, FFN down, LM head). The distinction matters because
        they have different scaling rules.
    """

    def __init__(self, config: muPConfig):
        """
        Initialize the muP initializer.

        Args:
            config: muPConfig with base and target model widths.
        """
        self.config = config
        self.d_base = config.d_base
        self.d_model = config.d_model
        self.alpha = d_model / d_base  # Width ratio
        self.base_lr = config.base_lr

    def _classify_parameter(self, name: str, param: nn.Parameter) -> str:
        """
        Classify a parameter as 'hidden', 'output', 'embedding', 'norm', or 'bias'.

        Args:
            name: Full parameter name (e.g., "layers.0.attention.w_q.weight").
            param: The parameter tensor.

        Returns:
            Classification string.

        Explanation:
            We use naming conventions to classify:
            - Names containing 'embedding' → 'embedding'
            - Names containing 'norm' → 'norm'
            - Names ending in '.bias' → 'bias'
            - Hidden layers: w_q, w_k, w_v (attention input projections),
              W_gate, W_up (FFN input projections), W_compress, W_q_content,
              W_q_rope, W_k_rope (MLA components)
            - Output layers: w_o (attention output), W_down (FFN output),
              lm_head (language model head), W_k_up, W_v_up (MLA up-projections)
        """
        name_lower = name.lower()

        # Embedding
        if "embedding" in name_lower or "wte" in name_lower:
            return "embedding"

        # Normalization
        if "norm" in name_lower or "ln" in name_lower:
            return "norm"

        # Bias terms
        if name_lower.endswith(".bias"):
            return "bias"

        # Output layers: projections that go back to d_model
        output_keywords = [
            "w_o", "w_down", "lm_head", "W_down",
            "W_k_up", "W_v_up",  # MLA up-projections (reconstruct from latent)
        ]
        for keyword in output_keywords:
            if keyword in name:
                return "output"

        # Hidden layers: everything else that's a weight matrix
        return "hidden"

    def init_parameter(self, name: str, param: nn.Parameter) -> None:
        """
        Initialize a single parameter according to muP rules.

        Args:
            name: Parameter name.
            param: The parameter tensor.

        Explanation:
            Uses Kaiming-style initialization with muP-scaled variance.
            For hidden layers: std = gain / sqrt(fan_in)
            For output layers: std = gain / sqrt(fan_out)
            Where gain accounts for the activation function (default 1.0).
        """
        if param.dim() < 2:
            # 1D parameters (biases, norms) — standard init
            return

        category = self._classify_parameter(name, param)

        if category == "embedding":
            # Embedding: standard normal, scaled by 1
            nn.init.normal_(param, mean=0.0, std=1.0)

        elif category == "norm":
            # Norm: leave as-is (typically ones for weight, zeros for bias)
            pass

        elif category == "hidden":
            # Hidden layers: Kaiming-style with muP scaling
            fan_in = param.size(1)
            # std: 1 / sqrt(fan_in) — standard Kaiming init
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(param, mean=0.0, std=std)

        elif category == "output":
            # Output layers: Kaiming-style with fan_out
            fan_out = param.size(0)
            # std: 1 / sqrt(fan_out)
            std = 1.0 / math.sqrt(fan_out)
            nn.init.normal_(param, mean=0.0, std=std)

        elif category == "bias":
            # Biases: zero init
            nn.init.zeros_(param)

    def get_lr_multipliers(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute per-parameter learning rate multipliers.

        Args:
            model: The model to compute multipliers for.

        Returns:
            Dictionary mapping parameter names to LR multipliers.

        Explanation:
            The LR multiplier determines how the base learning rate is
            scaled for each parameter:
            - Hidden layers: 1/alpha (slower learning for wider models)
            - Output layers: 1/alpha^2 (even slower — output magnitude
              scales quadratically with width)
            - Embedding: 1 (width-independent)
            - Norm: 1 (width-independent)
        """
        alpha = self.alpha
        multipliers = {}

        for name, param in model.named_parameters():
            category = self._classify_parameter(name, param)

            if category == "hidden":
                # Hidden: LR scales as 1/alpha
                multipliers[name] = 1.0 / alpha

            elif category == "output":
                # Output: LR scales as 1/alpha^2
                multipliers[name] = 1.0 / (alpha ** 2)

            elif category == "embedding":
                # Embedding: width-independent
                multipliers[name] = self.config.embedding_mult

            elif category == "norm":
                # Norm: width-independent
                multipliers[name] = 1.0

            elif category == "bias":
                # Bias: width-independent
                multipliers[name] = 1.0

            else:
                # Default: no scaling
                multipliers[name] = 1.0

        return multipliers

    def apply_initialization(self, model: nn.Module) -> None:
        """
        Apply muP initialization to all parameters in the model.

        Args:
            model: The model to initialize.

        Explanation:
            Iterates through all parameters and applies the appropriate
            muP initialization. Call this AFTER model construction and
            BEFORE training begins.
        """
        for name, param in model.named_parameters():
            self.init_parameter(name, param)

    def get_param_groups(
        self,
        model: nn.Module,
        weight_decay: float = 0.1,
    ) -> list:
        """
        Create optimizer parameter groups with muP-scaled learning rates.

        Args:
            model: The model.
            weight_decay: Weight decay factor.

        Returns:
            List of parameter group dictionaries for use with optimizers.

        Explanation:
            Returns parameter groups compatible with torch.optim.AdamW.
            Each group has:
            - params: list of parameters
            - lr: base_lr * multiplier
            - weight_decay: 0.0 for biases/norms, weight_decay for others

            Usage:
                initializer = muPInitializer(config)
                param_groups = initializer.get_param_groups(model)
                optimizer = torch.optim.AdamW(param_groups)
        """
        multipliers = self.get_lr_multipliers(model)

        # Group by LR multiplier for efficiency
        groups = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            lr_mult = multipliers.get(name, 1.0)
            lr = self.base_lr * lr_mult

            # Determine weight decay
            category = self._classify_parameter(name, param)
            wd = 0.0 if category in ("bias", "norm") else weight_decay

            # Group key: (lr, weight_decay)
            key = (lr, wd)
            if key not in groups:
                groups[key] = {"params": [], "lr": lr, "weight_decay": wd}
            groups[key]["params"].append(param)

        return list(groups.values())


################################################################################
# SECTION 3: TESTING & DEMONSTRATION
################################################################################


def demonstrate_init():
    """Demonstrate muP initialization and LR multipliers."""
    print("=" * 70)
    print("MU-P INITIALIZATION DEMONSTRATION")
    print("=" * 70)

    # --- Configuration ---
    d_base = 128
    d_model = 512
    config = muPConfig(d_base=d_base, d_model=d_model, base_lr=3e-4)

    initializer = muPInitializer(config)

    print(f"\nConfiguration:")
    print(f"  Base width (d_base):  {d_base}")
    print(f"  Model width (d_model): {d_model}")
    print(f"  Width ratio (alpha):  {initializer.alpha:.2f}")
    print(f"  Base LR:              {config.base_lr}")

    # --- Create a simple model to demonstrate ---
    class SimpleModel(nn.Module):
        """Simple transformer-like model for demonstration."""
        def __init__(self, d_model: int):
            super().__init__()
            self.embedding = nn.Embedding(1000, d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.w_q = nn.Linear(d_model, d_model, bias=False)  # Hidden layer
            self.w_k = nn.Linear(d_model, d_model, bias=False)  # Hidden layer
            self.w_v = nn.Linear(d_model, d_model, bias=False)  # Hidden layer
            self.w_o = nn.Linear(d_model, d_model, bias=False)  # Output layer
            self.norm2 = nn.LayerNorm(d_model)
            self.w_up = nn.Linear(d_model, d_model * 4, bias=False)  # Hidden layer
            self.w_down = nn.Linear(d_model * 4, d_model, bias=False)  # Output layer
            self.lm_head = nn.Linear(d_model, 1000, bias=True)  # Output layer

        def forward(self, x):
            return self.lm_head(self.w_down(F.gelu(self.w_up(self.norm2(self.w_o(self.w_v(self.w_k(self.w_q(self.norm1(self.embedding(x)))))))))))

    model = SimpleModel(d_model)

    # --- Apply muP initialization ---
    print(f"\nApplying muP initialization...")
    initializer.apply_initialization(model)

    # --- Show initialization stats ---
    print(f"\nInitialization statistics:")
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            category = initializer._classify_parameter(name, param)
            print(f"  {name:30s} | {str(list(param.shape)):20s} | "
                  f"mean={param.mean().item():+.4f} | std={param.std().item():.4f} | "
                  f"type={category}")

    # --- Show LR multipliers ---
    multipliers = initializer.get_lr_multipliers(model)
    print(f"\nLearning rate multipliers (alpha={initializer.alpha:.2f}):")
    for name, mult in sorted(multipliers.items()):
        actual_lr = config.base_lr * mult
        print(f"  {name:30s} | mult={mult:.6f} | lr={actual_lr:.2e}")

    # --- Show optimizer parameter groups ---
    param_groups = initializer.get_param_groups(model, weight_decay=0.1)
    print(f"\nOptimizer parameter groups:")
    for i, group in enumerate(param_groups):
        n_params = sum(p.numel() for p in group["params"])
        print(f"  Group {i}: lr={group['lr']:.2e}, "
              f"weight_decay={group['weight_decay']:.1f}, "
              f"n_params={n_params:,}")

    # --- Show scaling behavior ---
    print(f"\nScaling behavior (how multipliers change with width):")
    print(f"  {'d_model':>10s} | {'alpha':>8s} | {'hidden_lr':>12s} | {'output_lr':>12s}")
    print(f"  {'-'*10} | {'-'*8} | {'-'*12} | {'-'*12}")
    for dm in [64, 128, 256, 512, 1024, 2048, 4096]:
        a = dm / d_base
        hidden_lr = config.base_lr / a
        output_lr = config.base_lr / (a ** 2)
        print(f"  {dm:>10d} | {a:>8.2f} | {hidden_lr:>12.2e} | {output_lr:>12.2e}")

    print("\n" + "=" * 70)
    print("All muP initialization demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_init()
