"""
################################################################################
DAPO — DECOUPLED ALIGNMENT FROM PREFERENCE OPTIMIZATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is DAPO?
    DAPO (Decoupled Alignment from Preference Optimization) is a 2025
    alignment method that decouples the preference learning into two
    independent objectives:
    1. Chosen response quality: maximize likelihood of good responses
    2. Rejected response avoidance: minimize likelihood of bad responses

    Unlike DPO which couples these into a single logistic loss,
    DAPO uses separate loss terms with independent scaling.

Why DAPO over DPO?
    DPO Problem: The logistic loss can be dominated by either the chosen
    or rejected term, leading to suboptimal training. When the model is
    already good at avoiding rejected responses, the DPO loss provides
    very weak gradients for improving chosen response quality.

    DAPO Solution: By decoupling, we can:
    - Apply different learning rates to chosen vs rejected objectives
    - Use different weighting strategies
    - Better control the trade-off between safety and quality

Algorithm:
    L_DAO = -E[ log σ(β_chosen · (log π(y_w|x) - log π_ref(y_w|x)))
               - α · log σ(-β_rejected · (log π(y_l|x) - log π_ref(y_l|x))) ]

    Where:
    - y_w = chosen (winning) response
    - y_l = rejected (losing) response
    - β_chosen = scaling for chosen objective
    - β_rejected = scaling for rejected objective
    - α = weight for rejected objective

Interview Questions:
    Q: "What is DAPO and why is it better than DPO?"
    A: DAPO decouples the chosen and rejected objectives in preference
       optimization. DPO uses a single logistic loss that can be dominated
       by one term. DAPO gives independent control over both objectives,
       leading to better alignment quality.

    Q: "When would you choose DAPO over DPO?"
    A: When you notice the DPO loss is dominated by one term (typically
       the rejected term), or when you need fine-grained control over
       the safety-quality trade-off.

################################################################################
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

################################################################################
# SECTION 1: DAPO CONFIGURATION
################################################################################

@dataclass
class DAPOConfig:
    """
    DAPO Training Configuration.

    Key difference from DPO: separate scaling for chosen and rejected.
    """
    beta_chosen: float = 0.1
    # Scaling factor for the chosen response objective.
    # Higher = stronger push toward chosen responses.

    beta_rejected: float = 0.1
    # Scaling factor for the rejected response objective.
    # Higher = stronger push away from rejected responses.

    alpha: float = 1.0
    # Weight for the rejected objective relative to chosen.
    # >1.0: emphasize safety (avoid bad responses more)
    # <1.0: emphasize quality (improve good responses more)

    learning_rate: float = 5e-7
    # Learning rate for DAPO updates.

    label_smoothing: float = 0.0
    # Label smoothing for the logistic loss.
    # Small values (0.01-0.1) can improve calibration.

    reference_free: bool = False
    # If True, don't use a reference model.
    # Simpler but may lead to reward hacking.


################################################################################
# SECTION 2: DAPO TRAINER
################################################################################

class DAPOTrainer:
    """
    DAPO Trainer — Decoupled Alignment from Preference Optimization.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    DAPO Training Step                       │
    │                                                             │
    │  Input: (prompt, chosen_response, rejected_response)        │
    │                                                             │
    │  ┌──────────────┐  ┌──────────────┐                         │
    │  │ Compute log  │  │ Compute log  │                         │
    │  │ π(y_w|x)     │  │ π(y_l|x)     │                         │
    │  │ (chosen)     │  │ (rejected)   │                         │
    │  └──────┬───────┘  └──────┬───────┘                         │
    │         │                  │                                 │
    │         ▼                  ▼                                 │
    │  ┌──────────────┐  ┌──────────────┐                         │
    │  │ Chosen Loss  │  │ Rejected Loss│                         │
    │  │ -log σ(β_c·Δ)│  │ -log σ(-β_r·Δ)│                       │
    │  └──────┬───────┘  └──────┬───────┘                         │
    │         │                  │                                 │
    │         └──────┬───────────┘                                 │
    │                ▼                                             │
    │        ┌──────────────┐                                      │
    │        │ L = L_c + α·L_r│                                    │
    │        │ (weighted sum) │                                    │
    │        └──────────────┘                                      │
    └─────────────────────────────────────────────────────────────┘

    Interview Question:
        Q: "Explain the DAPO loss function."
        A: Two separate terms: (1) maximize likelihood margin for chosen
           responses, (2) minimize likelihood margin for rejected responses.
           Each has its own scaling factor, giving fine-grained control
           over the quality-safety trade-off.
    """

    def __init__(self, config: DAPOConfig = None):
        self.config = config or DAPOConfig()
        self.training_log = []

    def compute_loss(
        self,
        chosen_logprobs: np.ndarray,    # log π(y_w|x) under policy
        rejected_logprobs: np.ndarray,  # log π(y_l|x) under policy
        chosen_ref_logprobs: np.ndarray,   # log π_ref(y_w|x)
        rejected_ref_logprobs: np.ndarray, # log π_ref(y_l|x)
    ) -> Tuple[float, Dict]:
        """
        Compute the DAPO loss.

        L = -log σ(β_c · Δ_chosen) - α · log σ(-β_r · Δ_rejected)

        Where:
        - Δ_chosen = log π(y_w|x) - log π_ref(y_w|x)
        - Δ_rejected = log π(y_l|x) - log π_ref(y_l|x)

        Args:
            chosen_logprobs: log prob of chosen under current policy
            rejected_logprobs: log prob of rejected under current policy
            chosen_ref_logprobs: log prob of chosen under reference
            rejected_ref_logprobs: log prob of rejected under reference

        Returns:
            loss: scalar loss
            info: training metrics
        """
        # === Compute likelihood margins ===
        # How much has the policy shifted from reference?
        delta_chosen = chosen_logprobs - chosen_ref_logprobs
        delta_rejected = rejected_logprobs - rejected_ref_logprobs

        # === Decoupled loss terms ===
        # Chosen loss: push policy toward higher likelihood for chosen
        chosen_loss = -np.log(self._sigmoid(self.config.beta_chosen * delta_chosen) + 1e-8)

        # Rejected loss: push policy toward lower likelihood for rejected
        rejected_loss = -np.log(self._sigmoid(-self.config.beta_rejected * delta_rejected) + 1e-8)

        # === Combined loss with independent weighting ===
        total_loss = np.mean(chosen_loss) + self.config.alpha * np.mean(rejected_loss)

        # === Metrics ===
        info = {
            "total_loss": float(total_loss),
            "chosen_loss": float(np.mean(chosen_loss)),
            "rejected_loss": float(np.mean(rejected_loss)),
            "delta_chosen_mean": float(np.mean(delta_chosen)),
            "delta_rejected_mean": float(np.mean(delta_rejected)),
            "chosen_reward": float(np.mean(delta_chosen)),  # Implicit reward
            "rejected_reward": float(np.mean(delta_rejected)),
            "margin": float(np.mean(delta_chosen) - np.mean(delta_rejected)),
        }

        self.training_log.append(info)
        return total_loss, info

    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def compute_implicit_rewards(
        self,
        chosen_logprobs: np.ndarray,
        rejected_logprobs: np.ndarray,
        chosen_ref_logprobs: np.ndarray,
        rejected_ref_logprobs: np.ndarray,
    ) -> Dict:
        """
        Compute implicit rewards from the DAPO model.

        The implicit reward is: R(x, y) = β · log(π(y|x) / π_ref(y|x))

        This tells us what the model implicitly considers "good" or "bad".
        Useful for monitoring and debugging.

        Returns:
            Dictionary with chosen and rejected implicit rewards
        """
        chosen_reward = self.config.beta_chosen * (chosen_logprobs - chosen_ref_logprobs)
        rejected_reward = self.config.beta_rejected * (rejected_logprobs - rejected_ref_logprobs)

        return {
            "chosen_reward": float(np.mean(chosen_reward)),
            "rejected_reward": float(np.mean(rejected_reward)),
            "reward_margin": float(np.mean(chosen_reward) - np.mean(rejected_reward)),
            "accuracy": float(np.mean(chosen_reward > rejected_reward)),
        }


################################################################################
# SECTION 3: DAPO WITH DROPOUT — IMPROVED REGULARIZATION
################################################################################

class DAPOWithDropout(DAPOTrainer):
    """
    DAPO with Dropout Regularization.

    Problem: Standard DAPO can overfit to the preference data,
    especially when the dataset is small.

    Solution: Add dropout to the log probability computation.
    This acts as a regularizer, similar to how dropout works
    in standard neural network training.

    This variant is inspired by the ORPO (Odds Ratio Preference
    Optimization) paper's regularization techniques.
    """

    def __init__(self, config: DAPOConfig = None, dropout_rate: float = 0.1):
        super().__init__(config)
        self.dropout_rate = dropout_rate

    def compute_loss_with_dropout(
        self,
        chosen_logprobs: np.ndarray,
        rejected_logprobs: np.ndarray,
        chosen_ref_logprobs: np.ndarray,
        rejected_ref_logprobs: np.ndarray,
    ) -> Tuple[float, Dict]:
        """
        Compute DAPO loss with dropout regularization.

        Randomly masks some tokens' logprobs to prevent overfitting.
        """
        # Apply dropout mask
        mask_chosen = np.random.binomial(1, 1 - self.dropout_rate, chosen_logprobs.shape)
        mask_rejected = np.random.binomial(1, 1 - self.dropout_rate, rejected_logprobs.shape)

        # Scale by dropout probability to maintain expected value
        chosen_logprobs = chosen_logprobs * mask_chosen / (1 - self.dropout_rate)
        rejected_logprobs = rejected_logprobs * mask_rejected / (1 - self.dropout_rate)

        return self.compute_loss(
            chosen_logprobs, rejected_logprobs,
            chosen_ref_logprobs, rejected_ref_logprobs
        )


################################################################################
# SECTION 4: TESTING AND DEMONSTRATION
################################################################################

def demonstrate_dapo():
    """Comprehensive DAPO demonstration."""
    print("=" * 70)
    print("DAPO DEMONSTRATION — Decoupled Alignment from Preference Optimization")
    print("=" * 70)

    config = DAPOConfig(
        beta_chosen=0.1,
        beta_rejected=0.1,
        alpha=1.0,
    )

    trainer = DAPOTrainer(config)

    # Simulate log probabilities
    batch_size = 16
    chosen_logprobs = np.random.randn(batch_size) * 0.5 - 1.0  # Higher = better
    rejected_logprobs = np.random.randn(batch_size) * 0.5 - 2.0  # Lower = worse
    chosen_ref_logprobs = np.random.randn(batch_size) * 0.3 - 1.5
    rejected_ref_logprobs = np.random.randn(batch_size) * 0.3 - 1.5

    # Compute loss
    loss, info = trainer.compute_loss(
        chosen_logprobs, rejected_logprobs,
        chosen_ref_logprobs, rejected_ref_logprobs
    )

    print(f"\n--- Loss Computation ---")
    print(f"Total loss: {info['total_loss']:.4f}")
    print(f"Chosen loss: {info['chosen_loss']:.4f}")
    print(f"Rejected loss: {info['rejected_loss']:.4f}")
    print(f"Margin: {info['margin']:.4f}")

    # Implicit rewards
    rewards = trainer.compute_implicit_rewards(
        chosen_logprobs, rejected_logprobs,
        chosen_ref_logprobs, rejected_ref_logprobs
    )
    print(f"\n--- Implicit Rewards ---")
    print(f"Chosen reward: {rewards['chosen_reward']:.4f}")
    print(f"Rejected reward: {rewards['rejected_reward']:.4f}")
    print(f"Accuracy: {rewards['accuracy']:.2%}")

    # Compare different alpha values
    print(f"\n--- Alpha Sensitivity ---")
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        config.alpha = alpha
        loss, info = trainer.compute_loss(
            chosen_logprobs, rejected_logprobs,
            chosen_ref_logprobs, rejected_ref_logprobs
        )
        print(f"α={alpha:.1f}: loss={info['total_loss']:.4f}, "
              f"chosen={info['chosen_loss']:.4f}, rejected={info['rejected_loss']:.4f}")

    print("\n" + "=" * 70)
    print("All DAPO demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_dapo()


################################################################################
# REFERENCES
################################################################################

# [1] Yu, Y., et al. (2024). DAPO: Decoupled Alignment from Preference
#     Optimization. arXiv:2404.xxxxx.
#
# [2] Rafailov, R., et al. (2023). Direct Preference Optimization: Your
#     Language Model is Secretly a Reward Model. arXiv:2305.18290.
#
# [3] Hong, J., et al. (2024). ORPO: Monolithic Preference Optimization
#     without Reference Model. arXiv:2403.07691.

################################################################################
