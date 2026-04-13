"""
################################################################################
COST OPTIMIZATION — REDUCING AI COSTS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Cost Optimization?
    Reducing the cost of AI systems.

Key Strategies:
    - Use smaller models when possible
    - Quantization
    - Caching
    - Autoscaling
    - Spot instances

Interview Questions:
    Q: "How do you reduce AI costs?"
    A: Use smaller models, quantize, cache, autoscale.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: COST OPTIMIZER
################################################################################

class CostOptimizer:
    """
    Cost Optimizer
    ==============

    Optimizes AI system costs.
    """

    def __init__(self, cost_per_gpu_hour: float = 3.0):
        self.cost_per_gpu_hour = cost_per_gpu_hour

    def estimate_cost(
        self,
        n_gpus: int,
        hours: float,
        utilization: float = 0.8
    ) -> float:
        """
        Estimate cost.

        Args:
            n_gpus: Number of GPUs
            hours: Hours of usage
            utilization: GPU utilization

        Returns:
            cost: Estimated cost in dollars
        """
        return n_gpus * hours * utilization * self.cost_per_gpu_hour

    def optimize(self, strategy: str):
        """Apply optimization strategy."""
        pass


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_cost_optimization():
    """Demonstrate cost optimization."""
    print("=" * 70)
    print("COST OPTIMIZATION DEMONSTRATION")
    print("=" * 70)

    optimizer = CostOptimizer(cost_per_gpu_hour=3.0)

    # Estimate costs
    cost = optimizer.estimate_cost(n_gpus=8, hours=24, utilization=0.8)
    print(f"8 GPUs × 24 hours: ${cost:.2f}")

    cost = optimizer.estimate_cost(n_gpus=4, hours=24, utilization=0.6)
    print(f"4 GPUs × 24 hours (60% util): ${cost:.2f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_cost_optimization()
