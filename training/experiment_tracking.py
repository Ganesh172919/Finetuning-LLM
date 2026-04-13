"""
################################################################################
EXPERIMENT TRACKING — LOGGING TRAINING RUNS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Experiment Tracking?
    Logging and comparing training experiments.

Key Tools:
    - Weights & Biases (wandb)
    - MLflow
    - TensorBoard

Interview Questions:
    Q: "How do you track experiments?"
    A: Log hyperparameters, metrics, and artifacts.
       Use tools like wandb or MLflow.

################################################################################
"""

import numpy as np
from typing import Dict, List

################################################################################
# SECTION 1: EXPERIMENT TRACKER
################################################################################

class ExperimentTracker:
    """
    Experiment Tracker
    ===================

    Tracks training experiments.
    """

    def __init__(self, project_name: str = "sota-ai"):
        self.project_name = project_name
        self.runs: List[Dict] = []
        self.current_run = None

    def start_run(self, config: Dict):
        """Start a new experiment run."""
        self.current_run = {
            'config': config,
            'metrics': {},
        }

    def log_metric(self, key: str, value: float, step: int):
        """Log a metric."""
        if self.current_run:
            if key not in self.current_run['metrics']:
                self.current_run['metrics'][key] = []
            self.current_run['metrics'][key].append((step, value))

    def end_run(self):
        """End current run."""
        if self.current_run:
            self.runs.append(self.current_run)
            self.current_run = None

    def compare_runs(self) -> Dict:
        """Compare all runs."""
        comparison = {}
        for i, run in enumerate(self.runs):
            if 'loss' in run['metrics']:
                losses = [v for _, v in run['metrics']['loss']]
                comparison[f'run_{i}'] = {
                    'final_loss': losses[-1] if losses else None,
                    'best_loss': min(losses) if losses else None,
                }
        return comparison


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_experiment_tracking():
    """Demonstrate experiment tracking."""
    print("=" * 70)
    print("EXPERIMENT TRACKING DEMONSTRATION")
    print("=" * 70)

    tracker = ExperimentTracker()

    # Run experiment
    tracker.start_run({'lr': 3e-4, 'batch_size': 32})
    for step in range(100):
        loss = 2.5 * np.exp(-step / 50)
        tracker.log_metric('loss', loss, step)
    tracker.end_run()

    # Compare
    comparison = tracker.compare_runs()
    print(f"Runs: {comparison}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_experiment_tracking()
