"""
################################################################################
CONTINUOUS BATCHING — EFFICIENT REQUEST HANDLING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Continuous Batching?
    Dynamically managing batches as requests complete.

Traditional: wait for all requests in batch to finish
Continuous: new requests join as others complete

Benefits:
    - 2-3x throughput improvement
    - Better GPU utilization
    - Lower latency

Interview Questions:
    1. "What is continuous batching?"
        A batching strategy where requests join and leave dynamically.

################################################################################
"""

import numpy as np
from typing import List, Dict
from collections import deque

################################################################################
# SECTION 1: CONTINUOUS BATCHER
################################################################################

class ContinuousBatcher:
    """
    Continuous Batcher
    ==================

    Dynamically manages batches for efficient GPU utilization.
    """

    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        self.active: List[Dict] = []
        self.queue = deque()

    def add_request(self, request: Dict):
        """Add a new request."""
        if len(self.active) < self.max_batch_size:
            self.active.append(request)
        else:
            self.queue.append(request)

    def remove_completed(self):
        """Remove completed requests and add from queue."""
        self.active = [r for r in self.active if not r.get('done', False)]
        while len(self.active) < self.max_batch_size and self.queue:
            self.active.append(self.queue.popleft())

    def get_batch(self) -> List[Dict]:
        """Get current batch."""
        return self.active


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_batching():
    """Demonstrate continuous batching."""
    print("=" * 70)
    print("CONTINUOUS BATCHING DEMONSTRATION")
    print("=" * 70)

    batcher = ContinuousBatcher(max_batch_size=4)

    # Add requests
    for i in range(6):
        batcher.add_request({'id': i, 'done': False})
        print(f"Added request {i}, batch size: {len(batcher.active)}")

    # Complete some
    batcher.active[0]['done'] = True
    batcher.remove_completed()
    print(f"\nAfter completing 1, batch size: {len(batcher.active)}")
    print(f"Queue size: {len(batcher.queue)}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_batching()
