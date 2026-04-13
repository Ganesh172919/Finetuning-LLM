"""
################################################################################
WANDB — EXPERIMENT TRACKING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is WandB?
    Weights & Biases - an experiment tracking platform.

Features:
    - Metric logging
    - Hyperparameter tracking
    - Model versioning
    - Collaboration

Interview Questions:
    Q: "What is WandB?"
    A: An experiment tracking platform for ML.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: WANDB LOGGER
################################################################################

class WandBLogger:
    """
    WandB Logger
    ============

    Logs experiments to WandB.
    """

    def __init__(self, project: str = "sota-ai"):
        self.project = project
        self.metrics = {}

    def log(self, metrics: Dict):
        """Log metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def finish(self):
        """Finish run."""
        print(f"Run finished with {len(self.metrics)} metrics")


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_wandb():
    """Demonstrate WandB."""
    print("=" * 70)
    print("WANDB DEMONSTRATION")
    print("=" * 70)

    logger = WandBLogger(project="sota-ai")
    for step in range(10):
        logger.log({'loss': 2.5 * np.exp(-step / 5), 'lr': 3e-4})
    logger.finish()

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_wandb()
