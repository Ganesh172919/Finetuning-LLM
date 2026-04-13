"""
################################################################################
FAULT RECOVERY — HANDLING FAILURES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Fault Recovery?
    Handling failures during training.

Key Strategies:
    - Checkpointing
    - Automatic restart
    - Elastic training

Interview Questions:
    Q: "How do you handle GPU failures during training?"
    A: Regular checkpointing, automatic restart, elastic training.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: FAULT RECOVERY
################################################################################

class FaultRecoveryManager:
    """
    Fault Recovery Manager
    ======================

    Manages fault recovery during training.
    """

    def __init__(self, checkpoint_interval: int = 1000):
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints = []

    def save_checkpoint(self, state: Dict, step: int):
        """Save checkpoint."""
        self.checkpoints.append({'state': state, 'step': step})

    def load_latest_checkpoint(self) -> Optional[Dict]:
        """Load latest checkpoint."""
        if self.checkpoints:
            return self.checkpoints[-1]
        return None

    def should_checkpoint(self, step: int) -> bool:
        """Check if should save checkpoint."""
        return step % self.checkpoint_interval == 0


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_fault_recovery():
    """Demonstrate fault recovery."""
    print("=" * 70)
    print("FAULT RECOVERY DEMONSTRATION")
    print("=" * 70)

    manager = FaultRecoveryManager(checkpoint_interval=100)

    # Save checkpoints
    for step in range(0, 500, 100):
        manager.save_checkpoint({'weights': np.random.randn(10, 10)}, step)

    # Load latest
    latest = manager.load_latest_checkpoint()
    print(f"Latest checkpoint: step {latest['step']}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_fault_recovery()
