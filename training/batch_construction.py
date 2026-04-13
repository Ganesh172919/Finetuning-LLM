"""
################################################################################
BATCH CONSTRUCTION — CREATING TRAINING BATCHES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Batch Construction?
    Creating training batches from data.

Methods:
    - Fixed length: pad to same length
    - Packing: pack multiple sequences
    - Dynamic batching: group similar lengths

Interview Questions:
    Q: "How do you handle variable length sequences?"
    A: Padding, packing, or dynamic batching.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: BATCH CONSTRUCTOR
################################################################################

class BatchConstructor:
    """
    Batch Constructor
    =================

    Creates training batches.
    """

    def __init__(self, max_seq_len: int, batch_size: int):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def create_batches(self, token_ids: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create training batches."""
        batches = []
        n_tokens = len(token_ids)

        for i in range(0, n_tokens - self.max_seq_len, self.max_seq_len * self.batch_size):
            batch_inputs = []
            batch_targets = []

            for j in range(self.batch_size):
                start = i + j * self.max_seq_len
                if start + self.max_seq_len + 1 > n_tokens:
                    break

                batch_inputs.append(token_ids[start:start + self.max_seq_len])
                batch_targets.append(token_ids[start + 1:start + self.max_seq_len + 1])

            if len(batch_inputs) == self.batch_size:
                batches.append((np.array(batch_inputs), np.array(batch_targets)))

        return batches


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_batch_construction():
    """Demonstrate batch construction."""
    print("=" * 70)
    print("BATCH CONSTRUCTION DEMONSTRATION")
    print("=" * 70)

    constructor = BatchConstructor(max_seq_len=8, batch_size=2)
    token_ids = np.arange(50)
    batches = constructor.create_batches(token_ids)
    print(f"Number of batches: {len(batches)}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_batch_construction()
