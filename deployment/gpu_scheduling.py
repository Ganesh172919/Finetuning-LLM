"""
################################################################################
GPU SCHEDULING — MANAGING GPU RESOURCES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is GPU Scheduling?
    Managing GPU resources for efficient utilization.

Key Concepts:
    - GPU allocation
    - Memory management
    - Multi-tenancy

Interview Questions:
    Q: "How do you manage GPU resources?"
    A: Use GPU scheduling, memory management, and multi-tenancy.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: GPU SCHEDULER
################################################################################

class GPUScheduler:
    """
    GPU Scheduler
    =============

    Manages GPU resource allocation.
    """

    def __init__(self, n_gpus: int = 8):
        self.n_gpus = n_gpus
        self.available = list(range(n_gpus))
        self.allocated = {}

    def allocate(self, job_id: str, n_gpus: int = 1) -> List[int]:
        """Allocate GPUs for a job."""
        if len(self.available) < n_gpus:
            return []

        gpus = self.available[:n_gpus]
        self.available = self.available[n_gpus:]
        self.allocated[job_id] = gpus
        return gpus

    def release(self, job_id: str):
        """Release GPUs from a job."""
        if job_id in self.allocated:
            self.available.extend(self.allocated[job_id])
            del self.allocated[job_id]


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_gpu_scheduling():
    """Demonstrate GPU scheduling."""
    print("=" * 70)
    print("GPU SCHEDULING DEMONSTRATION")
    print("=" * 70)

    scheduler = GPUScheduler(n_gpus=8)

    # Allocate
    gpus = scheduler.allocate("job1", n_gpus=2)
    print(f"Allocated for job1: {gpus}")
    print(f"Available: {scheduler.available}")

    # Release
    scheduler.release("job1")
    print(f"After release: {scheduler.available}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_gpu_scheduling()
