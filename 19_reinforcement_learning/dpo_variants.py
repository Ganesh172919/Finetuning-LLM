"""
################################################################################
DPO VARIANTS — IPO, KTO, ORPO — MODERN ALIGNMENT METHODS (2024-2025 SOTA)
################################################################################

########################################
CONCEPT EXPLANATION
########################################

This file implements three important DPO variants that address limitations
of the original DPO (Direct Preference Optimization):

1. IPO (Identity Preference Optimization)
   Problem: DPO can overfit to preference data due to the logistic loss.
   Solution: Use a squared loss that's more robust to overfitting.

2. KTO (Kahneman-Tversky Optimization)
   Problem: DPO needs preference PAIRS (chosen + rejected). Getting pairs
   is expensive — you often just have "this is good" or "this is bad".
   Solution: KTO works with INDIVIDUAL feedback (thumbs up/down), not pairs.
   Based on Prospect Theory from behavioral economics.

3. ORPO (Odds Ratio Preference Optimization)
   Problem: DPO needs a separate reference model (2x memory).
   Solution: ORPO adds preference learning directly to the SFT loss,
   eliminating the reference model entirely.

Comparison Table:
┌──────────┬──────────────┬───────────────┬──────────────┬──────────────┐
│ Method   │ Input Type   │ Ref Model?    │ Loss Type    │ Best For     │
├──────────┼──────────────┼───────────────┼──────────────┼──────────────┤
│ DPO      │ Pairs        │ Yes           │ Logistic     │ General      │
│ IPO      │ Pairs        │ Yes           │ Squared      │ Robustness   │
│ KTO      │ Individual   │ Yes           │ Prospect     │ Limited data │
│ ORPO     │ Pairs        │ No            │ Odds Ratio   │ Memory-limited│
└──────────┴──────────────┴───────────────┴──────────────┴──────────────┘

Interview Questions:
    Q: "What are the main DPO variants and when would you use each?"
    A: IPO uses squared loss for robustness, KTO works with individual
       feedback (no pairs needed), ORPO eliminates the reference model.
       Use IPO when overfitting, KTO when you lack pairs, ORPO when
       memory-constrained.

    Q: "How does KTO work without preference pairs?"
    A: It's based on Prospect Theory — losses loom larger than gains.
       For "good" responses, maximize log π/π_ref. For "bad" responses,
       minimize it. The asymmetry (losses > gains) handles the lack of
       explicit comparison.

################################################################################
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

################################################################################
# SECTION 1: IPO — IDENTITY PREFERENCE OPTIMIZATION
################################################################################

@dataclass
class IPOConfig:
    """IPO Configuration."""
    beta: float = 0.1          # Scaling factor
    learning_rate: float = 5e-7
    tau: float = 0.1           # IPO regularization parameter


class IPOTrainer:
    """
    IPO — Identity Preference Optimization
    ========================================

    IPO addresses DPO's tendency to overfit to preference data.

    Problem with DPO:
    The logistic loss in DPO can push the model to always prefer the
    chosen response and always reject the alternative, even when the
    preference is marginal. This leads to overconfidence.

    IPO Solution:
    Use a squared loss instead of logistic:
    L_IPO = (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x)) - τ)²

    The τ parameter controls the target margin — how much better the
    chosen response should be. This prevents the model from becoming
    overconfident.

    Interview Question:
        Q: "How does IPO improve on DPO?"
        A: IPO uses a squared regression loss instead of DPO's logistic loss.
           This prevents the model from becoming overconfident about marginal
           preferences. The τ parameter explicitly controls the target margin
           between chosen and rejected responses.
    """

    def __init__(self, config: IPOConfig = None):
        self.config = config or IPOConfig()
        self.training_log = []

    def compute_loss(
        self,
        chosen_logprobs: np.ndarray,
        rejected_logprobs: np.ndarray,
        chosen_ref_logprobs: np.ndarray,
        rejected_ref_logprobs: np.ndarray,
    ) -> Tuple[float, Dict]:
        """
        Compute IPO loss.

        L_IPO = (log_ratio_chosen - log_ratio_rejected - τ)²

        Where:
        - log_ratio_chosen = log(π(y_w|x) / π_ref(y_w|x))
        - log_ratio_rejected = log(π(y_l|x) / π_ref(y_l|x))
        - τ = target margin (how much better chosen should be)
        """
        # Log ratios (how much policy has shifted from reference)
        log_ratio_chosen = chosen_logprobs - chosen_ref_logprobs
        log_ratio_rejected = rejected_logprobs - rejected_ref_logprobs

        # IPO loss: squared difference from target margin
        diff = log_ratio_chosen - log_ratio_rejected
        loss = np.mean((diff - self.config.tau) ** 2)

        info = {
            "loss": float(loss),
            "mean_diff": float(np.mean(diff)),
            "target_margin": self.config.tau,
            "implicit_accuracy": float(np.mean(diff > 0)),
        }

        self.training_log.append(info)
        return loss, info


################################################################################
# SECTION 2: KTO — KAHNEMAN-TVERSKY OPTIMIZATION
################################################################################

@dataclass
class KTOConfig:
    """KTO Configuration."""
    beta: float = 0.1
    learning_rate: float = 5e-7
    loss_weight: float = 1.0       # Weight for loss aversion
    desirable_weight: float = 1.0  # Weight for desirable examples
    undesirable_weight: float = 1.0  # Weight for undesirable examples


class KTOTrainer:
    """
    KTO — Kahneman-Tversky Optimization
    ======================================

    KTO is based on Prospect Theory from behavioral economics.
    Key insight: Humans feel losses more strongly than gains
    (loss aversion). KTO applies this to LLM alignment.

    Why KTO?
    - DPO needs preference PAIRS (chosen vs rejected)
    - In practice, you often just have individual feedback:
      "This response is good" (👍) or "This response is bad" (👎)
    - Getting pairs is expensive; getting individual labels is cheap

    How KTO works:
    - For DESIRABLE responses (👍): maximize log(π/π_ref)
    - For UNDESIRABLE responses (👎): minimize log(π/π_ref)
    - Apply loss aversion: undesirable losses are weighted more heavily

    Mathematical Formulation:
    L_KTO = E_desirable[-σ(β · log(π/π_ref))] +
            λ · E_undesirable[-σ(-β · log(π/π_ref))]

    Where λ > 1 implements loss aversion (undesirable errors hurt more).

    Interview Question:
        Q: "How does KTO differ from DPO?"
        A: KTO works with individual feedback (good/bad labels) instead
           of preference pairs. It's based on Prospect Theory — losses
           loom larger than gains. This makes it practical when you have
           thumbs up/down data but not explicit comparisons.

        Q: "When would you choose KTO over DPO?"
        A: When you have individual quality labels but not preference pairs.
           This is common in production — users give thumbs up/down but
           rarely compare two responses side by side.
    """

    def __init__(self, config: KTOConfig = None):
        self.config = config or KTOConfig()
        self.training_log = []

    def compute_loss(
        self,
        logprobs: np.ndarray,         # π(response|x)
        ref_logprobs: np.ndarray,     # π_ref(response|x)
        is_desirable: np.ndarray,     # 1 if good, 0 if bad
    ) -> Tuple[float, Dict]:
        """
        Compute KTO loss.

        For desirable responses: push π higher than π_ref
        For undesirable responses: push π lower than π_ref
        Loss aversion: undesirable errors weighted more heavily

        Args:
            logprobs: [batch] log prob under current policy
            ref_logprobs: [batch] log prob under reference policy
            is_desirable: [batch] 1 if response is good, 0 if bad

        Returns:
            loss: scalar KTO loss
            info: training metrics
        """
        # Log ratio: how much policy has shifted from reference
        log_ratio = logprobs - ref_logprobs

        # Desirable loss: want log_ratio > 0 (policy prefers this response)
        desirable_mask = is_desirable.astype(bool)
        desirable_loss = -np.log(
            self._sigmoid(self.config.beta * log_ratio[desirable_mask]) + 1e-8
        )

        # Undesirable loss: want log_ratio < 0 (policy avoids this response)
        undesirable_mask = ~desirable_mask
        undesirable_loss = -np.log(
            self._sigmoid(-self.config.beta * log_ratio[undesirable_mask]) + 1e-8
        )

        # Weighted combination with loss aversion
        n_desirable = max(np.sum(desirable_mask), 1)
        n_undesirable = max(np.sum(undesirable_mask), 1)

        total_loss = (
            self.config.desirable_weight * np.sum(desirable_loss) / n_desirable +
            self.config.undesirable_weight * self.config.loss_weight * np.sum(undesirable_loss) / n_undesirable
        )

        info = {
            "loss": float(total_loss),
            "desirable_loss": float(np.mean(desirable_loss)) if len(desirable_loss) > 0 else 0,
            "undesirable_loss": float(np.mean(undesirable_loss)) if len(undesirable_loss) > 0 else 0,
            "n_desirable": int(np.sum(desirable_mask)),
            "n_undesirable": int(np.sum(undesirable_mask)),
            "mean_log_ratio": float(np.mean(log_ratio)),
            "desirable_accuracy": float(np.mean(log_ratio[desirable_mask] > 0)) if np.any(desirable_mask) else 0,
            "undesirable_accuracy": float(np.mean(log_ratio[undesirable_mask] < 0)) if np.any(undesirable_mask) else 0,
        }

        self.training_log.append(info)
        return total_loss, info

    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


################################################################################
# SECTION 3: ORPO — ODDS RATIO PREFERENCE OPTIMIZATION
################################################################################

@dataclass
class ORPOConfig:
    """ORPO Configuration."""
    lambda_orpo: float = 0.1     # Weight for ORPO loss term
    learning_rate: float = 5e-7


class ORPOTrainer:
    """
    ORPO — Odds Ratio Preference Optimization
    ============================================

    ORPO eliminates the reference model entirely!

    Problem with DPO/KTO/IPO:
    - All need a reference model (π_ref)
    - Reference model = 2x memory consumption
    - Reference model must be kept in sync

    ORPO Solution:
    - Combine SFT loss with preference loss in one objective
    - Use odds ratio instead of log probability ratio
    - No reference model needed!

    Mathematical Formulation:
    L_ORPO = L_SFT(y_w) + λ · L_OR

    Where:
    - L_SFT(y_w) = -log π(y_w|x)  (standard SFT on chosen responses)
    - L_OR = -log σ(log(odds(y_w)) - log(odds(y_l)))
    - odds(y) = π(y|x) / (1 - π(y|x))

    The odds ratio measures how much MORE likely the chosen response
    is compared to the rejected, relative to the probability of NOT
    generating them.

    Interview Question:
        Q: "How does ORPO eliminate the reference model?"
        A: ORPO combines SFT and preference learning into one loss.
           Instead of comparing π to π_ref, it uses odds ratios
           (π/(1-π)) to measure preference. This means you only need
           one model, saving 50% memory.

        Q: "What are the tradeoffs of ORPO vs DPO?"
        A: Pros: 50% less memory, simpler training, no reference sync.
           Cons: SFT and preference learning may interfere; less stable
           on some tasks. Best for memory-constrained settings.
    """

    def __init__(self, config: ORPOConfig = None):
        self.config = config or ORPOConfig()
        self.training_log = []

    def compute_loss(
        self,
        chosen_logprobs: np.ndarray,   # log π(y_w|x)
        rejected_logprobs: np.ndarray, # log π(y_l|x)
    ) -> Tuple[float, Dict]:
        """
        Compute ORPO loss (no reference model needed!).

        L_ORPO = L_SFT + λ · L_OR

        Args:
            chosen_logprobs: [batch] log prob of chosen responses
            rejected_logprobs: [batch] log prob of rejected responses

        Returns:
            loss: scalar ORPO loss
            info: training metrics
        """
        # === SFT Loss (on chosen responses) ===
        # Maximize likelihood of chosen responses
        sft_loss = -np.mean(chosen_logprobs)

        # === Odds Ratio Loss ===
        # Convert log probs to odds: odds = p / (1-p)
        # In log space: log_odds = log_p - log(1 - exp(log_p))
        chosen_odds = self._log_odds(chosen_logprobs)
        rejected_odds = self._log_odds(rejected_logprobs)

        # Odds ratio loss: want chosen odds > rejected odds
        or_loss = -np.log(self._sigmoid(chosen_odds - rejected_odds) + 1e-8)
        or_loss = np.mean(or_loss)

        # === Combined Loss ===
        total_loss = sft_loss + self.config.lambda_orpo * or_loss

        info = {
            "loss": float(total_loss),
            "sft_loss": float(sft_loss),
            "or_loss": float(or_loss),
            "mean_chosen_logprob": float(np.mean(chosen_logprobs)),
            "mean_rejected_logprob": float(np.mean(rejected_logprobs)),
            "accuracy": float(np.mean(chosen_logprobs > rejected_logprobs)),
            "mean_odds_ratio": float(np.mean(chosen_odds - rejected_odds)),
        }

        self.training_log.append(info)
        return total_loss, info

    def _log_odds(self, log_probs: np.ndarray) -> np.ndarray:
        """
        Compute log odds from log probabilities.

        log_odds = log(p / (1-p)) = log_p - log(1 - exp(log_p))

        Numerically stable implementation.
        """
        # Clamp for numerical stability
        log_probs = np.clip(log_probs, -20, 0)
        # log(1 - exp(log_p)) = log(1 - p)
        log_one_minus_p = np.log(1 - np.exp(log_probs) + 1e-8)
        return log_probs - log_one_minus_p

    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


################################################################################
# SECTION 4: UNIFIED COMPARISON
################################################################################

class AlignmentMethodComparison:
    """
    Compare all alignment methods side by side.

    Useful for understanding the tradeoffs between methods
    and choosing the right one for your use case.
    """

    @staticmethod
    def compare_all(
        chosen_logprobs: np.ndarray,
        rejected_logprobs: np.ndarray,
        chosen_ref_logprobs: np.ndarray,
        rejected_ref_logprobs: np.ndarray,
    ) -> Dict:
        """
        Run all methods on the same data and compare.

        Returns:
            Dictionary with results from each method
        """
        results = {}

        # DPO (from existing implementation)
        from .dapo import DAPOConfig, DAPOTrainer
        dapo = DAPOTrainer(DAPOConfig())
        _, dapo_info = dapo.compute_loss(
            chosen_logprobs, rejected_logprobs,
            chosen_ref_logprobs, rejected_ref_logprobs
        )
        results["DAPO"] = dapo_info

        # IPO
        ipo = IPOTrainer(IPOConfig())
        _, ipo_info = ipo.compute_loss(
            chosen_logprobs, rejected_logprobs,
            chosen_ref_logprobs, rejected_ref_logprobs
        )
        results["IPO"] = ipo_info

        # KTO (needs individual labels)
        kto = KTOTrainer(KTOConfig())
        all_logprobs = np.concatenate([chosen_logprobs, rejected_logprobs])
        all_ref_logprobs = np.concatenate([chosen_ref_logprobs, rejected_ref_logprobs])
        is_desirable = np.concatenate([
            np.ones(len(chosen_logprobs)),
            np.zeros(len(rejected_logprobs))
        ])
        _, kto_info = kto.compute_loss(all_logprobs, all_ref_logprobs, is_desirable)
        results["KTO"] = kto_info

        # ORPO (no reference model)
        orpo = ORPOTrainer(ORPOConfig())
        _, orpo_info = orpo.compute_loss(chosen_logprobs, rejected_logprobs)
        results["ORPO"] = orpo_info

        return results


################################################################################
# SECTION 5: TESTING AND DEMONSTRATION
################################################################################

def demonstrate_dpo_variants():
    """Comprehensive demonstration of all DPO variants."""
    print("=" * 70)
    print("DPO VARIANTS DEMONSTRATION — IPO, KTO, ORPO")
    print("=" * 70)

    # Generate synthetic data
    batch_size = 32
    chosen_logprobs = np.random.randn(batch_size) * 0.5 - 1.0
    rejected_logprobs = np.random.randn(batch_size) * 0.5 - 2.0
    chosen_ref_logprobs = np.random.randn(batch_size) * 0.3 - 1.5
    rejected_ref_logprobs = np.random.randn(batch_size) * 0.3 - 1.5

    # === Demo 1: IPO ===
    print("\n--- Demo 1: IPO (Identity Preference Optimization) ---")
    ipo = IPOTrainer(IPOConfig(tau=0.1))
    loss, info = ipo.compute_loss(
        chosen_logprobs, rejected_logprobs,
        chosen_ref_logprobs, rejected_ref_logprobs
    )
    print(f"Loss: {info['loss']:.4f}")
    print(f"Mean diff: {info['mean_diff']:.4f}")
    print(f"Accuracy: {info['implicit_accuracy']:.2%}")

    # === Demo 2: KTO ===
    print("\n--- Demo 2: KTO (Kahneman-Tversky Optimization) ---")
    kto = KTOTrainer(KTOConfig(loss_weight=2.0))
    all_logprobs = np.concatenate([chosen_logprobs, rejected_logprobs])
    all_ref_logprobs = np.concatenate([chosen_ref_logprobs, rejected_ref_logprobs])
    is_desirable = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])
    loss, info = kto.compute_loss(all_logprobs, all_ref_logprobs, is_desirable)
    print(f"Loss: {info['loss']:.4f}")
    print(f"Desirable accuracy: {info['desirable_accuracy']:.2%}")
    print(f"Undesirable accuracy: {info['undesirable_accuracy']:.2%}")
    print(f"Desirable count: {info['n_desirable']}, Undesirable: {info['n_undesirable']}")

    # === Demo 3: ORPO ===
    print("\n--- Demo 3: ORPO (Odds Ratio Preference Optimization) ---")
    orpo = ORPOTrainer(ORPOConfig(lambda_orpo=0.1))
    loss, info = orpo.compute_loss(chosen_logprobs, rejected_logprobs)
    print(f"Loss: {info['loss']:.4f}")
    print(f"SFT loss: {info['sft_loss']:.4f}")
    print(f"OR loss: {info['or_loss']:.4f}")
    print(f"Accuracy: {info['accuracy']:.2%}")
    print(f"Mean odds ratio: {info['mean_odds_ratio']:.4f}")

    # === Demo 4: Comparison ===
    print("\n--- Demo 4: Method Comparison ---")
    print(f"{'Method':<10} {'Loss':<10} {'Accuracy':<10}")
    print("-" * 30)

    # IPO
    _, ipo_info = ipo.compute_loss(chosen_logprobs, rejected_logprobs, chosen_ref_logprobs, rejected_ref_logprobs)
    print(f"{'IPO':<10} {ipo_info['loss']:<10.4f} {ipo_info['implicit_accuracy']:<10.2%}")

    # KTO
    _, kto_info = kto.compute_loss(all_logprobs, all_ref_logprobs, is_desirable)
    kto_acc = (kto_info['desirable_accuracy'] + kto_info['undesirable_accuracy']) / 2
    print(f"{'KTO':<10} {kto_info['loss']:<10.4f} {kto_acc:<10.2%}")

    # ORPO
    _, orpo_info = orpo.compute_loss(chosen_logprobs, rejected_logprobs)
    print(f"{'ORPO':<10} {orpo_info['loss']:<10.4f} {orpo_info['accuracy']:<10.2%}")

    print("\n" + "=" * 70)
    print("All DPO variant demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_dpo_variants()


################################################################################
# REFERENCES
################################################################################

# [1] Azar, M.G., et al. (2023). A General Theoretical Paradigm to Understand
#     Learning from Human Feedback. arXiv:2310.12036. (IPO)
#
# [2] Ethayarajh, K., et al. (2024). KTO: Model Alignment as Prospect
#     Theoretic Optimization. arXiv:2402.01306. (KTO)
#
# [3] Hong, J., et al. (2024). ORPO: Monolithic Preference Optimization
#     without Reference Model. arXiv:2403.07691. (ORPO)
#
# [4] Rafailov, R., et al. (2023). Direct Preference Optimization: Your
#     Language Model is Secretly a Reward Model. arXiv:2305.18290. (DPO)

################################################################################
