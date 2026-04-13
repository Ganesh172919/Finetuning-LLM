"""
################################################################################
TRAINING MONITORING — TRACKING TRAINING PROGRESS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Training Monitoring?
    Tracking and visualizing training progress.

Key Metrics:
    - Loss
    - Learning rate
    - Gradient norm
    - Throughput
    - Memory usage

Interview Questions:
    Q: "What should I monitor during training?"
    A: Loss, learning rate, gradient norm, validation metrics.

################################################################################
"""

import numpy as np
from typing import Dict, List

################################################################################
# SECTION 1: TRAINING MONITOR
################################################################################

class TrainingMonitor:
    """
    Training Monitor
    ================

    Tracks training metrics.
    """

    def __init__(self):
        self.metrics: Dict[str, List] = {}

    def log(self, key: str, value: float):
        """Log a metric."""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)

    def get_summary(self) -> Dict:
        """Get summary of all metrics."""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'latest': values[-1],
                    'mean': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }
        return summary

    def check_anomalies(self) -> List[str]:
        """Check for training anomalies."""
        anomalies = []

        if 'loss' in self.metrics and len(self.metrics['loss']) > 10:
            recent = self.metrics['loss'][-10:]
            if any(np.isnan(recent)):
                anomalies.append("NaN in loss")
            if any(np.isinf(recent)):
                anomalies.append("Inf in loss")

        if 'grad_norm' in self.metrics and self.metrics['grad_norm']:
            if self.metrics['grad_norm'][-1] > 100:
                anomalies.append("Exploding gradients")

        return anomalies


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_monitoring():
    """Demonstrate training monitoring."""
    print("=" * 70)
    print("TRAINING MONITORING DEMONSTRATION")
    print("=" * 70)

    monitor = TrainingMonitor()

    # Log some metrics
    for step in range(100):
        monitor.log('loss', 2.5 * np.exp(-step / 50) + np.random.randn() * 0.1)
        monitor.log('lr', 3e-4 * (1 - step / 100))

    # Summary
    summary = monitor.get_summary()
    print(f"Loss: {summary['loss']}")
    print(f"Learning rate: {summary['lr']}")

    # Check anomalies
    anomalies = monitor.check_anomalies()
    print(f"Anomalies: {anomalies}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_monitoring()
