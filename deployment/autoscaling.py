"""
################################################################################
AUTOSCALING — AUTOMATIC RESOURCE MANAGEMENT
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Autoscaling?
    Automatically adjusting resources based on demand.

Key Metrics:
    - Request rate
    - Latency
    - GPU utilization

Interview Questions:
    Q: "How do you autoscale model serving?"
    A: Monitor request rate and latency,
       scale up when overloaded, scale down when idle.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: AUTOSCALER
################################################################################

class Autoscaler:
    """
    Autoscaler
    ==========

    Manages automatic scaling of resources.
    """

    def __init__(self, min_replicas: int = 1, max_replicas: int = 10):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = min_replicas

    def should_scale_up(self, request_rate: float, latency: float) -> bool:
        """Check if should scale up."""
        return request_rate > 100 or latency > 1.0

    def should_scale_down(self, request_rate: float, latency: float) -> bool:
        """Check if should scale down."""
        return request_rate < 10 and latency < 0.1

    def update(self, request_rate: float, latency: float):
        """Update replica count."""
        if self.should_scale_up(request_rate, latency):
            self.current_replicas = min(self.current_replicas + 1, self.max_replicas)
        elif self.should_scale_down(request_rate, latency):
            self.current_replicas = max(self.current_replicas - 1, self.min_replicas)


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_autoscaling():
    """Demonstrate autoscaling."""
    print("=" * 70)
    print("AUTOSCALING DEMONSTRATION")
    print("=" * 70)

    scaler = Autoscaler(min_replicas=1, max_replicas=5)

    # Simulate traffic
    for i in range(10):
        request_rate = np.random.uniform(0, 200)
        latency = np.random.uniform(0.05, 2.0)
        scaler.update(request_rate, latency)
        print(f"Step {i}: rate={request_rate:.0f}, latency={latency:.2f}, replicas={scaler.current_replicas}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_autoscaling()
