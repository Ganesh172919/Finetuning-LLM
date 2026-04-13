"""
################################################################################
CACHING — STORING FREQUENT RESULTS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Caching?
    Storing frequently accessed results for faster retrieval.

Applications:
    - KV cache for attention
    - Response caching for common queries
    - Embedding caching

Interview Questions:
    Q: "How do you optimize LLM inference with caching?"
    A: KV cache for attention, response caching for common prompts.

################################################################################
"""

import numpy as np
from typing import Dict, Optional

################################################################################
# SECTION 1: CACHE
################################################################################

class LRUCache:
    """
    LRU Cache
    =========

    Least Recently Used cache.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache: Dict[str, np.ndarray] = {}
        self.order: List[str] = []

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get value from cache."""
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: np.ndarray):
        """Put value in cache."""
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]

        self.cache[key] = value
        self.order.append(key)


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_caching():
    """Demonstrate caching."""
    print("=" * 70)
    print("CACHING DEMONSTRATION")
    print("=" * 70)

    cache = LRUCache(capacity=3)

    cache.put("a", np.array([1, 2, 3]))
    cache.put("b", np.array([4, 5, 6]))
    cache.put("c", np.array([7, 8, 9]))

    print(f"Get a: {cache.get('a')}")
    print(f"Get d: {cache.get('d')}")

    cache.put("d", np.array([10, 11, 12]))
    print(f"Get b (should be None): {cache.get('b')}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_caching()
