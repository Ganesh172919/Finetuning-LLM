"""
DPO (Direct Preference Optimization) Training Pipeline
========================================================

DPO is a simplified alternative to RLHF that directly optimizes a language
model using preference data, without requiring a separate reward model.

Key Insight:
  The RLHF objective (maximize reward while staying close to reference policy)
  has a closed-form solution. We can optimize it directly with a simple loss.

  DPO Loss:
  L_DPO = -E[log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

  Where:
    π = policy model (being trained)
    π_ref = reference model (frozen)
    y_w = preferred response
    y_l = rejected response
    β = temperature parameter

Why DPO over RLHF?
  - No reward model needed (simpler pipeline)
  - No PPO instability (more stable training)
  - Same performance as RLHF on many benchmarks
  - Easier to implement and debug

Pipeline Overview:
┌─────────────────────────────────────────────────────────────┐
│                    DPO Training Pipeline                     │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Preference│───▶│   DPO    │───▶│  Trained │              │
│  │   Data    │    │  Trainer │    │  Policy  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │               │               │                     │
│       ▼               ▼               ▼                     │
│  (chosen,         Loss = -log σ      Better                 │
│   rejected)       (margin)           responses              │
└─────────────────────────────────────────────────────────────┘

References:
  - Rafailov et al., "Direct Preference Optimization" (2023)
  - Ethayarajh et al., "KTO" (2024)
  - Azar et al., "IPO" (2023)

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DPOConfig:
    """
    Configuration for DPO training.

    Attributes:
        beta: Temperature parameter. Higher β = stay closer to reference.
              Typical values: 0.1 to 0.5
        learning_rate: Learning rate for policy updates
        max_length: Maximum sequence length
        batch_size: Training batch size
        gradient_accumulation_steps: Accumulate gradients over N steps
        warmup_steps: LR warmup steps
        max_steps: Total training steps
        logging_steps: Log metrics every N steps
        save_steps: Save checkpoint every N steps
        reference_model_sync: How often to sync reference model (0 = never)
    """
    beta: float = 0.1
    learning_rate: float = 5e-7
    max_length: int = 2048
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_steps: int = 1000
    logging_steps: int = 10
    save_steps: int = 500
    reference_model_sync: int = 0


# ============================================================================
# PREFERENCE DATA
# ============================================================================

@dataclass
class PreferenceExample:
    """
    A single preference pair for DPO training.

    Contains:
    - prompt: The input question/instruction
    - chosen: The preferred response (human ranked higher)
    - rejected: The non-preferred response (human ranked lower)

    Example:
        prompt: "Explain quantum computing"
        chosen: "Quantum computing uses qubits that can be 0 and 1 simultaneously..."
        rejected: "Quantum computing is when computers use quantum stuff..."
    """
    prompt: str
    chosen: str
    rejected: str
    metadata: Dict = field(default_factory=dict)


class PreferenceDataset:
    """
    Dataset of preference pairs for DPO training.

    Data format:
        [
            {"prompt": "...", "chosen": "...", "rejected": "..."},
            ...
        ]

    Quality considerations:
        - Chosen responses should be consistently better
        - Rejected responses should be plausible but flawed
        - Diverse prompts covering different tasks
        - Consistent annotation guidelines
    """

    def __init__(self, examples: List[PreferenceExample]):
        """
        Initialize dataset.

        Args:
            examples: List of preference pairs
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def shuffle(self):
        """Shuffle dataset in-place."""
        np.random.shuffle(self.examples)

    @staticmethod
    def from_dicts(data: List[Dict]) -> 'PreferenceDataset':
        """Create dataset from list of dictionaries."""
        examples = [
            PreferenceExample(
                prompt=d["prompt"],
                chosen=d["chosen"],
                rejected=d["rejected"],
            )
            for d in data
        ]
        return PreferenceDataset(examples)

    def split(self, val_ratio: float = 0.1) -> Tuple['PreferenceDataset', 'PreferenceDataset']:
        """Split into train and validation sets."""
        n = len(self.examples)
        indices = np.random.permutation(n)
        split_idx = int(n * (1 - val_ratio))

        train = PreferenceDataset([self.examples[i] for i in indices[:split_idx]])
        val = PreferenceDataset([self.examples[i] for i in indices[split_idx:]])

        return train, val


# ============================================================================
# LOG-PROBABILITY COMPUTATION
# ============================================================================

def compute_logprobs(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute log probabilities of labels given logits.

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]

    Returns:
        Log probabilities per token [batch_size, seq_len]
    """
    # Numerically stable log-softmax
    logits_max = np.max(logits, axis=-1, keepdims=True)
    log_probs = logits - logits_max - np.log(
        np.sum(np.exp(logits - logits_max), axis=-1, keepdims=True)
    )

    # Gather log probs for target tokens
    batch_size, seq_len = labels.shape
    batch_idx = np.arange(batch_size)[:, None]
    seq_idx = np.arange(seq_len)[None, :]
    token_log_probs = log_probs[batch_idx, seq_idx, labels]

    return token_log_probs


def sequence_logprob(token_log_probs: np.ndarray,
                     attention_mask: np.ndarray) -> np.ndarray:
    """
    Compute sequence-level log probability (sum over tokens).

    Args:
        token_log_probs: Per-token log probs [batch_size, seq_len]
        attention_mask: Mask for valid tokens [batch_size, seq_len]

    Returns:
        Sequence log probs [batch_size]
    """
    return np.sum(token_log_probs * attention_mask, axis=-1)


# ============================================================================
# DPO LOSS FUNCTIONS
# ============================================================================

def dpo_loss(policy_chosen_logps: np.ndarray,
             policy_rejected_logps: np.ndarray,
             reference_chosen_logps: np.ndarray,
             reference_rejected_logps: np.ndarray,
             beta: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """
    Compute DPO loss.

    The DPO objective:
        L = -E[log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

    This is equivalent to:
        L = -E[log σ(β · (chosen_logratios - rejected_logratios))]

    Where:
        logratios = policy_logps - reference_logps

    The intuition:
        - If chosen_logratio > rejected_logratio: model is doing well, small loss
        - If chosen_logratio < rejected_logratio: model prefers bad answers, large loss
        - β controls how much we penalize deviation from reference

    Args:
        policy_chosen_logps: Policy log probs for chosen [batch_size]
        policy_rejected_logps: Policy log probs for rejected [batch_size]
        reference_chosen_logps: Reference log probs for chosen [batch_size]
        reference_rejected_logps: Reference log probs for rejected [batch_size]
        beta: Temperature parameter

    Returns:
        loss: Scalar DPO loss
        metrics: Dictionary of training metrics
    """
    # Compute log ratios (how much policy differs from reference)
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    # Compute the DPO margin
    logits = beta * (chosen_logratios - rejected_logratios)

    # DPO loss: negative log sigmoid of the margin
    # Using log-sigmoid for numerical stability: log σ(x) = -log(1 + e^{-x})
    loss = -np.mean(np.log(1 + np.exp(-logits)))

    # Compute metrics
    metrics = {
        "loss": float(loss),
        "chosen_logratios": float(np.mean(chosen_logratios)),
        "rejected_logratios": float(np.mean(rejected_logratios)),
        "margin": float(np.mean(chosen_logratios - rejected_logratios)),
        "accuracy": float(np.mean((chosen_logratios - rejected_logratios) > 0)),
        "chosen_rewards": float(np.mean(beta * chosen_logratios)),
        "rejected_rewards": float(np.mean(beta * rejected_logratios)),
    }

    return loss, metrics


def ipo_loss(policy_chosen_logps: np.ndarray,
             policy_rejected_logps: np.ndarray,
             reference_chosen_logps: np.ndarray,
             reference_rejected_logps: np.ndarray,
             beta: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """
    Identity Preference Optimization (IPO) loss.

    IPO addresses DPO's tendency to overfit by using a squared loss
    instead of log-sigmoid:

    L = E[(log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x)) - 1/(2β))²]

    Args:
        Same as dpo_loss

    Returns:
        loss: Scalar IPO loss
        metrics: Dictionary of training metrics
    """
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    # IPO uses squared loss on the margin
    target = 1.0 / (2 * beta)
    loss = np.mean((chosen_logratios - rejected_logratios - target) ** 2)

    metrics = {
        "loss": float(loss),
        "margin": float(np.mean(chosen_logratios - rejected_logratios)),
        "accuracy": float(np.mean((chosen_logratios - rejected_logratios) > target)),
    }

    return loss, metrics


def kto_loss(policy_logps: np.ndarray,
             reference_logps: np.ndarray,
             is_desirable: np.ndarray,
             beta: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """
    KTO (Kahneman-Tversky Optimization) loss.

    KTO only needs binary feedback (good/bad) instead of pairwise preferences.
    Based on prospect theory: losses hurt more than gains help.

    Args:
        policy_logps: Policy log probs [batch_size]
        reference_logps: Reference log probs [batch_size]
        is_desirable: Boolean mask [batch_size]
        beta: Temperature parameter

    Returns:
        loss: Scalar KTO loss
        metrics: Dictionary of training metrics
    """
    logratios = policy_logps - reference_logps

    # Prospect theory: losses weighted more than gains
    desirable_loss = -np.mean(np.log(1 + np.exp(-beta * logratios[is_desirable])))
    undesirable_loss = -np.mean(np.log(1 + np.exp(beta * logratios[~is_desirable])))

    loss = desirable_loss + undesirable_loss

    metrics = {
        "loss": float(loss),
        "desirable_loss": float(desirable_loss),
        "undesirable_loss": float(undesirable_loss),
    }

    return loss, metrics


# ============================================================================
# DPO TRAINER
# ============================================================================

class DPOTrainer:
    """
    DPO Training Pipeline.

    Handles the complete DPO training loop:
    1. Load preference data
    2. Compute reference model log probs (offline or synced)
    3. Train policy with DPO loss
    4. Log metrics and save checkpoints
    5. Evaluate on held-out data

    Training loop:
        for batch in dataloader:
            # Get log probs from policy (trainable)
            policy_logps = policy(batch)

            # Get log probs from reference (frozen)
            with torch.no_grad():
                ref_logps = reference(batch)

            # Compute DPO loss
            loss, metrics = dpo_loss(policy_logps, ref_logps)

            # Update policy
            loss.backward()
            optimizer.step()
    """

    def __init__(self, config: DPOConfig):
        """
        Initialize DPO trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.step_count = 0
        self.best_loss = float('inf')

        # Training history
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_margin": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def compute_reference_logprobs(self, dataset: PreferenceDataset,
                                    reference_model_fn) -> Dict[str, np.ndarray]:
        """
        Pre-compute reference model log probabilities.

        This is done once before training starts (offline) or periodically
        during training (online sync).

        Args:
            dataset: Preference dataset
            reference_model_fn: Function that computes log probs

        Returns:
            Dictionary with chosen and rejected reference log probs
        """
        chosen_logps = []
        rejected_logps = []

        for example in dataset:
            chosen_lp = reference_model_fn(example.prompt, example.chosen)
            rejected_lp = reference_model_fn(example.prompt, example.rejected)
            chosen_logps.append(chosen_lp)
            rejected_logps.append(rejected_lp)

        return {
            "chosen": np.array(chosen_logps),
            "rejected": np.array(rejected_logps),
        }

    def train_step(self, policy_chosen_logps: np.ndarray,
                   policy_rejected_logps: np.ndarray,
                   reference_chosen_logps: np.ndarray,
                   reference_rejected_logps: np.ndarray,
                   loss_type: str = "dpo") -> Dict:
        """
        Execute one DPO training step.

        Args:
            policy_chosen_logps: Policy log probs for chosen responses
            policy_rejected_logps: Policy log probs for rejected responses
            reference_chosen_logps: Reference log probs for chosen responses
            reference_rejected_logps: Reference log probs for rejected responses
            loss_type: Loss function type ("dpo", "ipo", "kto")

        Returns:
            Dictionary with loss and metrics
        """
        if loss_type == "dpo":
            loss, metrics = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                beta=self.config.beta
            )
        elif loss_type == "ipo":
            loss, metrics = ipo_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                beta=self.config.beta
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.step_count += 1

        # Track history
        self.history["train_loss"].append(metrics["loss"])
        self.history["train_accuracy"].append(metrics.get("accuracy", 0))
        self.history["train_margin"].append(metrics.get("margin", 0))

        return metrics

    def evaluate(self, policy_chosen_logps: np.ndarray,
                 policy_rejected_logps: np.ndarray,
                 reference_chosen_logps: np.ndarray,
                 reference_rejected_logps: np.ndarray) -> Dict:
        """
        Evaluate on validation data.

        Args:
            Same as train_step

        Returns:
            Evaluation metrics
        """
        loss, metrics = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps,
            beta=self.config.beta
        )

        self.history["val_loss"].append(metrics["loss"])
        self.history["val_accuracy"].append(metrics.get("accuracy", 0))

        return metrics

    def get_training_summary(self) -> Dict:
        """Get summary of training progress."""
        if not self.history["train_loss"]:
            return {"status": "not_started"}

        return {
            "steps": self.step_count,
            "train_loss": self.history["train_loss"][-1],
            "train_accuracy": self.history["train_accuracy"][-1],
            "train_margin": self.history["train_margin"][-1],
            "best_loss": min(self.history["train_loss"]),
            "loss_trend": (
                "decreasing"
                if len(self.history["train_loss"]) > 1 and
                   self.history["train_loss"][-1] < self.history["train_loss"][-2]
                else "increasing"
            ),
        }


# ============================================================================
# VARIANTS COMPARISON
# ============================================================================

def compare_dpo_variants():
    """
    Compare DPO variants: DPO, IPO, KTO.

    Returns comparison as formatted string.
    """
    comparison = """
    ┌──────────────────┬─────────────┬─────────────┬─────────────┐
    │ Property         │ DPO         │ IPO         │ KTO         │
    ├──────────────────┼─────────────┼─────────────┼─────────────┤
    │ Data format      │ Pairwise    │ Pairwise    │ Binary      │
    │ Loss function    │ Log-sigmoid │ Squared     │ Prospect    │
    │ Reward model     │ No          │ No          │ No          │
    │ Reference model  │ Yes         │ Yes         │ Yes         │
    │ β parameter      │ Temperature │ Regularize  │ Temperature │
    │ Overfitting risk │ Medium      │ Low         │ Low         │
    │ Implementation   │ Simple      │ Simple      │ Simple      │
    │ Paper            │ Rafailov+   │ Azar+       │ Ethayarajh+ │
    │ Year             │ 2023        │ 2023        │ 2024        │
    └──────────────────┴─────────────┴─────────────┴─────────────┘
    """
    return comparison


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_dpo_pipeline():
    """
    Demonstrate DPO training pipeline.

    Shows:
        1. Dataset creation
        2. Loss computation
        3. Training step
        4. Metrics tracking
        5. Variant comparison
    """
    print("=" * 70)
    print("DPO Training Pipeline — Demonstration")
    print("=" * 70)

    # Create synthetic preference data
    print("\n[1. Creating Preference Dataset]")
    data = [
        {
            "prompt": "Explain quantum computing",
            "chosen": "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information in fundamentally new ways. Qubits can exist in superposition of 0 and 1 simultaneously, enabling parallel computation.",
            "rejected": "Quantum computing is when computers use quantum stuff to be faster. It's really complicated but basically qubits are better than regular bits."
        },
        {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a subset of artificial intelligence where systems learn patterns from data without being explicitly programmed. The three main paradigms are supervised learning, unsupervised learning, and reinforcement learning.",
            "rejected": "Machine learning is when computers learn things. You give them data and they figure stuff out automatically."
        },
        {
            "prompt": "How does the internet work?",
            "chosen": "The internet is a global network of interconnected computers using standardized protocols (TCP/IP). Data is broken into packets, routed through multiple nodes, and reassembled at the destination. DNS translates domain names to IP addresses.",
            "rejected": "The internet works by sending information through wires and wireless signals. Websites are stored on servers and you access them through your browser."
        },
    ]

    dataset = PreferenceDataset.from_dicts(data)
    train_data, val_data = dataset.split(val_ratio=0.33)
    print(f"  Total examples: {len(dataset)}")
    print(f"  Training: {len(train_data)}")
    print(f"  Validation: {len(val_data)}")

    # Initialize trainer
    print("\n[2. Initializing DPO Trainer]")
    config = DPOConfig(
        beta=0.1,
        learning_rate=5e-7,
        batch_size=4,
        max_steps=1000,
    )
    trainer = DPOTrainer(config)
    print(f"  Beta: {config.beta}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max steps: {config.max_steps}")

    # Simulate training step
    print("\n[3. Simulating Training Step]")
    batch_size = 4
    policy_chosen = np.random.randn(batch_size) * 2 - 1
    policy_rejected = np.random.randn(batch_size) * 2 - 2
    ref_chosen = np.random.randn(batch_size) * 2 - 1
    ref_rejected = np.random.randn(batch_size) * 2 - 2

    metrics = trainer.train_step(
        policy_chosen, policy_rejected,
        ref_chosen, ref_rejected,
        loss_type="dpo"
    )

    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Margin: {metrics['margin']:.4f}")
    print(f"  Chosen rewards: {metrics['chosen_rewards']:.4f}")
    print(f"  Rejected rewards: {metrics['rejected_rewards']:.4f}")

    # Compare variants
    print("\n[4. DPO Variants Comparison]")
    print(compare_dpo_variants())

    # Training summary
    print("\n[5. Training Summary]")
    summary = trainer.get_training_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. DPO is simpler than RLHF — no reward model needed")
    print("  2. Loss = -log σ(β × (chosen_ratio - rejected_ratio))")
    print("  3. β controls how much we stay close to reference")
    print("  4. IPO uses squared loss for better stability")
    print("  5. KTO only needs binary feedback (good/bad)")
    print("=" * 70)


if __name__ == "__main__":
    demo_dpo_pipeline()
