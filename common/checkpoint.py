"""
################################################################################
CHECKPOINTING — SAVING AND LOADING MODELS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Checkpointing?
    Saving model state during training for recovery or deployment.

Why does it matter?
    - GPU failures are common in large-scale training
    - Checkpointing allows recovery without starting over
    - Can also use for model selection, evaluation

Interview Questions:
    1. "What do you save in a checkpoint?"
        Model weights, optimizer state, training step, learning rate.

    2. "How often should you checkpoint?"
        Every 1000-5000 steps, depending on training time.

################################################################################
"""

import numpy as np
from typing import Dict, Optional

################################################################################
# SECTION 1: CHECKPOINT MANAGER
################################################################################

def save_checkpoint(
    model_state: Dict,
    optimizer_state: Dict,
    step: int,
    path: str
):
    """
    Save checkpoint to file.

    Args:
        model_state: Model weights
        optimizer_state: Optimizer state
        step: Current training step
        path: Save path
    """
    checkpoint = {
        'model': model_state,
        'optimizer': optimizer_state,
        'step': step
    }
    # In real implementation: save to file
    print(f"Saved checkpoint at step {step}")


def load_checkpoint(path: str) -> Dict:
    """
    Load checkpoint from file.

    Args:
        path: Checkpoint path

    Returns:
        Checkpoint data
    """
    # In real implementation: load from file
    return {
        'model': {},
        'optimizer': {},
        'step': 0
    }


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_checkpointing():
    """Demonstrate checkpointing."""
    print("=" * 70)
    print("CHECKPOINTING DEMONSTRATION")
    print("=" * 70)

    # Save
    model_state = {'weights': np.random.randn(10, 10)}
    optimizer_state = {'lr': 1e-4, 'momentum': 0.9}
    save_checkpoint(model_state, optimizer_state, step=1000, path="ckpt.pt")

    # Load
    loaded = load_checkpoint("ckpt.pt")
    print(f"Loaded checkpoint at step {loaded['step']}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_checkpointing()
