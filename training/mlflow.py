"""
################################################################################
MLFLOW — ML LIFECYCLE MANAGEMENT
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is MLflow?
    An open-source ML lifecycle management platform.

Features:
    - Experiment tracking
    - Model registry
    - Model serving
    - Project management

Interview Questions:
    Q: "What is MLflow?"
    A: An open-source platform for ML lifecycle management.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: MLFLOW TRACKER
################################################################################

class MLflowTracker:
    """
    MLflow Tracker
    ==============

    Tracks ML experiments.
    """

    def __init__(self, experiment_name: str = "sota-ai"):
        self.experiment_name = experiment_name
        self.runs = []

    def start_run(self):
        """Start a new run."""
        self.current_run = {}

    def log_param(self, key: str, value):
        """Log a parameter."""
        if 'params' not in self.current_run:
            self.current_run['params'] = {}
        self.current_run['params'][key] = value

    def log_metric(self, key: str, value: float):
        """Log a metric."""
        if 'metrics' not in self.current_run:
            self.current_run['metrics'] = {}
        self.current_run['metrics'][key] = value

    def end_run(self):
        """End current run."""
        self.runs.append(self.current_run)


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_mlflow():
    """Demonstrate MLflow."""
    print("=" * 70)
    print("MLFLOW DEMONSTRATION")
    print("=" * 70)

    tracker = MLflowTracker()
    tracker.start_run()
    tracker.log_param("lr", 3e-4)
    tracker.log_metric("loss", 2.5)
    tracker.end_run()

    print(f"Runs: {len(tracker.runs)}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_mlflow()
