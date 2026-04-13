"""
################################################################################
CHECKPOINTING — SAVING AND LOADING MODEL STATE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Checkpointing?
    Saving model state during training for recovery or deployment.

What to Save:
    - Model weights
    - Optimizer state
    - Training step
    - Learning rate
    - Random state

Interview Questions:
    Q: "How often should you checkpoint?"
    A: Every 1000-5000 steps, depending on training time.

################################################################################
"""

import numpy as np
from typing import Dict

################################################################################
# SECTION 1: CHECKPOINT MANAGER
################################################################################

class CheckpointManager:
    """
    Checkpoint Manager
    ===================

    Manages model checkpoints.
    """

    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = save_dir
        self.checkpoints = []

    def save(
        self,
        model_state: Dict,
        optimizer_state: Dict,
        step: int,
        loss: float
    ):
        """Save checkpoint."""
        checkpoint = {
            'model': model_state,
            'optimizer': optimizer_state,
            'step': step,
            'loss': loss,
        }
        self.checkpoints.append(checkpoint)

    def load(self, checkpoint_idx: int = -1) -> Dict:
        """Load checkpoint."""
        return self.checkpoints[checkpoint_idx]

    def get_best(self) -> Dict:
        """Get checkpoint with lowest loss."""
        return min(self.checkpoints, key=lambda x: x['loss'])


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_checkpointing():
    """Demonstrate checkpointing."""
    print("=" * 70)
    print("CHECKPOINTING DEMONSTRATION")
    print("=" * 70)

    manager = CheckpointManager()

    # Save checkpoints
    for step in range(0, 1000, 100):
        loss = 2.5 * np.exp(-step / 500)
        manager.save(
            model_state={'weights': np.random.randn(10, 10)},
            optimizer_state={'lr': 3e-4},
            step=step,
            loss=loss
        )

    # Load best
    best = manager.get_best()
    print(f"Best checkpoint: step={best['step']}, loss={best['loss']:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_checkpointing()
