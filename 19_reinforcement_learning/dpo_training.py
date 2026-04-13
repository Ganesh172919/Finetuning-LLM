"""
################################################################################
DPO TRAINING — DIRECT PREFERENCE OPTIMIZATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is DPO?
    DPO aligns language models with human preferences WITHOUT
    training a separate reward model. It directly optimizes the
    policy using preference data.

Why DPO instead of RLHF?
    Traditional RLHF:
    1. Collect human preferences
    2. Train reward model
    3. Optimize policy with PPO (complex, unstable)

    DPO:
    1. Collect human preferences
    2. Directly optimize policy (simple, stable)

    DPO is simpler, more stable, and often works better.

Mathematical Derivation:
    RLHF objective:
    max E[reward(x, y)] - β * KL(π || π_ref)

    This can be solved analytically:
    π*(y|x) = π_ref(y|x) * exp(reward(x,y)/β) / Z(x)

    Rearranging:
    reward(x,y) = β * log(π*(y|x) / π_ref(y|x)) + β * log Z(x)

    Substituting into Bradley-Terry model:
    P(y_w > y_l) = σ(reward(x,y_w) - reward(x,y_l))
                  = σ(β * log(π(y_w|x)/π_ref(y_w|x)) - β * log(π(y_l|x)/π_ref(y_l|x)))

    This is the DPO loss!

Interview Questions:
        Q: "What is DPO?"
        A: Direct Preference Optimization. It aligns models with preferences
           without a reward model. Uses preference pairs directly.

        Q: "How does DPO derive from RLHF?"
        A: DPO shows that the RLHF objective has an analytical solution.
           Substituting this into the preference model gives a simple
           loss function that can be optimized directly.

################################################################################
"""

import numpy as np
from typing import List, Dict, Tuple

################################################################################
# SECTION 1: DPO TRAINER
################################################################################

class DPOTrainer:
    """
    DPO Trainer
    ===========

    Trains a language model using preference data.

    Loss Function:
    L_DPO = -E[log σ(β × (log π_θ(y_w|x) - log π_θ(y_l|x)
                         - log π_ref(y_w|x) + log π_ref(y_l|x)))]

    Where:
    - y_w: chosen (winning) response
    - y_l: rejected (losing) response
    - π_θ: policy being trained
    - π_ref: reference policy (frozen)
    - β: KL penalty coefficient

    Interview Questions:
        Q: "What does β control in DPO?"
        A: How much the policy can deviate from the reference.
           Large β: more conservative (closer to reference)
           Small β: more aggressive (further from reference)
           Typical values: 0.1 to 0.5

        Q: "What happens if β is too large?"
        A: The policy barely changes from the reference model.
           No alignment happens.

        Q: "What happens if β is too small?"
        A: The policy can diverge too much, potentially losing
           general capabilities.
    """

    def __init__(
        self,
        beta: float = 0.1,
        learning_rate: float = 5e-7,
        label_smoothing: float = 0.0
    ):
        """
        Initialize DPO trainer.

        Args:
            beta: KL penalty coefficient
            learning_rate: Learning rate
            label_smoothing: Label smoothing for robustness
        """
        self.beta = beta
        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing

    def compute_logprobs(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute log probabilities of labels given logits.

        Args:
            logits: [batch × seq × vocab]
            labels: [batch × seq]

        Returns:
            logprobs: [batch] sum of log probs over sequence
        """
        vocab_size = logits.shape[-1]

        # Log softmax
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
        log_probs = shifted - log_sum_exp[..., np.newaxis]

        # Gather log probs for labels
        batch, seq = labels.shape
        log_probs_selected = log_probs[np.arange(batch)[:, np.newaxis],
                                        np.arange(seq)[np.newaxis],
                                        labels]

        # Sum over sequence (masked by non-padding tokens)
        return np.sum(log_probs_selected, axis=-1)

    def compute_loss(
        self,
        policy_chosen_logps: np.ndarray,
        policy_rejected_logps: np.ndarray,
        reference_chosen_logps: np.ndarray,
        reference_rejected_logps: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Compute DPO loss.

        Args:
            policy_chosen_logps: Log probs of policy for chosen responses
            policy_rejected_logps: Log probs of policy for rejected responses
            reference_chosen_logps: Log probs of reference for chosen
            reference_rejected_logps: Log probs of reference for rejected

        Returns:
            loss: DPO loss
            metrics: Dictionary of metrics
        """
        # Compute log ratios
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # Compute logits
        logits = self.beta * (chosen_logratios - rejected_logratios)

        # DPO loss with optional label smoothing
        if self.label_smoothing > 0:
            # Conservative DPO
            loss = (
                self.label_smoothing * (-np.log(1 + np.exp(-logits))) +
                (1 - self.label_smoothing) * (-np.log(1 / (1 + np.exp(-logits))))
            )
        else:
            loss = -np.log(1 / (1 + np.exp(-logits)) + 1e-8)

        loss = np.mean(loss)

        # Metrics
        metrics = {
            'loss': float(loss),
            'chosen_logratios': float(np.mean(chosen_logratios)),
            'rejected_logratios': float(np.mean(rejected_logratios)),
            'logits': float(np.mean(logits)),
            'accuracy': float(np.mean(logits > 0)),
            'margin': float(np.mean(chosen_logratios - rejected_logratios)),
        }

        return loss, metrics

    def train_step(
        self,
        policy_chosen_logps: np.ndarray,
        policy_rejected_logps: np.ndarray,
        reference_chosen_logps: np.ndarray,
        reference_rejected_logps: np.ndarray
    ) -> Dict:
        """
        Single training step.

        Args:
            Various log probabilities

        Returns:
            metrics: Training metrics
        """
        # Compute loss
        loss, metrics = self.compute_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )

        # In real implementation:
        # 1. loss.backward()
        # 2. clip_grad_norm_()
        # 3. optimizer.step()

        return metrics


################################################################################
# SECTION 2: SIMPO (SIMPLIFIED DPO)
################################################################################

class SimPOTrainer:
    """
    SimPO: Simple Preference Optimization
    ======================================

    A simplified version of DPO that doesn't need a reference model.

    Key insight: Use average log probability instead of sum.
    This normalizes for response length.

    Loss = -log σ(β_avg × (avg_logp_chosen - avg_logp_rejected - γ))

    Where:
    - β_avg: scaling factor
    - γ: margin (target reward margin)

    Interview Questions:
        Q: "What's the difference between DPO and SimPO?"
        A: SimPO doesn't need a reference model, uses average log
           probability (length-normalized), and adds a target margin.
           Simpler and often works as well as DPO.
    """

    def __init__(self, beta: float = 2.0, gamma: float = 0.5):
        self.beta = beta
        self.gamma = gamma

    def compute_loss(
        self,
        policy_chosen_logps: np.ndarray,
        policy_rejected_logps: np.ndarray,
        chosen_lengths: np.ndarray,
        rejected_lengths: np.ndarray
    ) -> float:
        """
        Compute SimPO loss.

        Args:
            policy_chosen_logps: Sum log probs for chosen [batch]
            policy_rejected_logps: Sum log probs for rejected [batch]
            chosen_lengths: Length of chosen responses [batch]
            rejected_lengths: Length of rejected responses [batch]

        Returns:
            loss: SimPO loss
        """
        # Average log probability (length-normalized)
        avg_chosen = policy_chosen_logps / chosen_lengths
        avg_rejected = policy_rejected_logps / rejected_lengths

        # SimPO loss
        logits = self.beta * (avg_chosen - avg_rejected) - self.gamma
        loss = -np.mean(np.log(1 / (1 + np.exp(-logits)) + 1e-8))

        return loss


################################################################################
# SECTION 3: ORPO (ODDS RATIO PREFERENCE OPTIMIZATION)
################################################################################

class ORPOTrainer:
    """
    ORPO: Odds Ratio Preference Optimization
    ==========================================

    Combines SFT and preference optimization in one step.

    Loss = L_SFT + λ × L_OR

    Where:
    - L_SFT: Supervised fine-tuning loss on chosen responses
    - L_OR: Odds ratio loss for preferences

    Interview Questions:
        Q: "What's special about ORPO?"
        A: It combines SFT and preference optimization in one step,
           eliminating the need for a reference model or separate SFT phase.
    """

    def __init__(self, lambda_or: float = 0.1):
        self.lambda_or = lambda_or

    def compute_loss(
        self,
        sft_loss: float,
        chosen_logps: np.ndarray,
        rejected_logps: np.ndarray
    ) -> float:
        """
        Compute ORPO loss.

        Args:
            sft_loss: Supervised fine-tuning loss
            chosen_logps: Log probs for chosen [batch]
            rejected_logps: Log probs for rejected [batch]

        Returns:
            loss: ORPO loss
        """
        # Odds ratio
        log_odds = chosen_logps - rejected_logps
        or_loss = -np.mean(np.log(1 / (1 + np.exp(-log_odds)) + 1e-8))

        return sft_loss + self.lambda_or * or_loss


################################################################################
# SECTION 4: TESTING
################################################################################

def demonstrate_dpo():
    """Demonstrate DPO concepts."""
    print("=" * 70)
    print("DPO TRAINING DEMONSTRATION")
    print("=" * 70)

    # DPO Trainer
    print("\n--- DPO Trainer ---")
    trainer = DPOTrainer(beta=0.1)

    # Simulate log probabilities
    batch_size = 4
    policy_chosen = np.random.randn(batch_size) * 0.5
    policy_rejected = np.random.randn(batch_size) * 0.5 - 1.0
    ref_chosen = np.random.randn(batch_size) * 0.3
    ref_rejected = np.random.randn(batch_size) * 0.3 - 0.5

    # Compute loss
    loss, metrics = trainer.compute_loss(
        policy_chosen, policy_rejected,
        ref_chosen, ref_rejected
    )
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Margin: {metrics['margin']:.4f}")

    # SimPO
    print("\n--- SimPO ---")
    simpo = SimPOTrainer(beta=2.0, gamma=0.5)
    chosen_logps = np.random.randn(batch_size)
    rejected_logps = np.random.randn(batch_size) - 1.0
    chosen_lens = np.array([20, 25, 30, 22])
    rejected_lens = np.array([15, 20, 25, 18])
    simpo_loss = simpo.compute_loss(chosen_logps, rejected_logps, chosen_lens, rejected_lens)
    print(f"SimPO loss: {simpo_loss:.4f}")

    # ORPO
    print("\n--- ORPO ---")
    orpo = ORPOTrainer(lambda_or=0.1)
    sft_loss = 2.5
    orpo_loss = orpo.compute_loss(sft_loss, chosen_logps, rejected_logps)
    print(f"ORPO loss: {orpo_loss:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_dpo()


################################################################################
# REFERENCES
################################################################################

# [1] Rafailov, R., et al. (2023). Direct Preference Optimization.
# [2] Meng, Y., et al. (2024). SimPO: Simple Preference Optimization.
# [3] Hong, J., et al. (2024). ORPO: Monolithic Preference Optimization without Reference Model.

################################################################################
