"""
################################################################################
DATA LOADING — LOADING TRAINING DATA
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Data Loading?
    Loading and preprocessing training data.

Key Considerations:
    - Efficient loading (avoid I/O bottleneck)
    - Proper batching
    - Shuffling
    - Preprocessing

Interview Questions:
    Q: "How do you handle large datasets?"
    A: Use streaming, lazy loading, and efficient formats.

################################################################################
"""

import numpy as np
from typing import List, Tuple

################################################################################
# SECTION 1: DATA LOADER
################################################################################

class DataLoader:
    """
    Data Loader
    ===========

    Loads and batches training data.
    """

    def __init__(self, data: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self) -> np.ndarray:
        if len(self.indices) == 0:
            raise StopIteration

        batch_indices = self.indices[:self.batch_size]
        self.indices = self.indices[self.batch_size:]
        return self.data[batch_indices]


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_data_loading():
    """Demonstrate data loading."""
    print("=" * 70)
    print("DATA LOADING DEMONSTRATION")
    print("=" * 70)

    data = np.random.randn(100, 10)
    loader = DataLoader(data, batch_size=16)

    for i, batch in enumerate(loader):
        if i >= 3:
            break
        print(f"Batch {i}: {batch.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_data_loading()
