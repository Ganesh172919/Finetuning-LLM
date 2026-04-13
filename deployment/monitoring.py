"""
################################################################################
MONITORING — TRACKING SYSTEM HEALTH
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Monitoring?
    Tracking system health and performance.

Key Metrics:
    - Latency
    - Throughput
    - Error rate
    - GPU utilization

Interview Questions:
    Q: "What should you monitor in production?"
    A: Latency, throughput, errors, GPU utilization, cost.

################################################################################
"""

import numpy as np
from typing import Dict, List

################################################################################
# SECTION 1: MONITOR
################################################################################

class SystemMonitor:
    """
    System Monitor
    ==============

    Tracks system health metrics.
    """

    def __init__(self):
        self.metrics: Dict[str, List] = {}

    def log(self, key: str, value: float):
        """Log a metric."""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)

    def get_stats(self) -> Dict:
        """Get statistics for all metrics."""
        stats = {}
        for key, values in self.metrics.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                }
        return stats

    def check_health(self) -> Dict:
        """Check system health."""
        health = {'status': 'healthy', 'issues': []}

        if 'latency' in self.metrics and self.metrics['latency']:
            if np.mean(self.metrics['latency'][-10:]) > 1.0:
                health['issues'].append('High latency')
                health['status'] = 'degraded'

        return health


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_monitoring():
    """Demonstrate monitoring."""
    print("=" * 70)
    print("MONITORING DEMONSTRATION")
    print("=" * 70)

    monitor = SystemMonitor()

    # Simulate metrics
    for _ in range(100):
        monitor.log('latency', np.random.exponential(0.1))
        monitor.log('throughput', np.random.uniform(50, 150))

    stats = monitor.get_stats()
    print(f"Latency stats: {stats['latency']}")
    print(f"Throughput stats: {stats['throughput']}")

    health = monitor.check_health()
    print(f"Health: {health}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_monitoring()
