"""
################################################################################
DATA PIPELINE — PREPARING DATA FOR TRAINING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Data Pipeline?
    The system for loading, processing, and batching data for training.

Steps:
    1. Data collection
    2. Data cleaning
    3. Deduplication
    4. Tokenization
    5. Batching
    6. Shuffling

Interview Questions:
    Q: "How do you prepare data for LLM training?"
    A: Collect text, clean it, deduplicate, tokenize,
       create batches with proper sequence lengths.

################################################################################
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import Counter

################################################################################
# SECTION 1: DATA CLEANER
################################################################################

class DataCleaner:
    """
    Data Cleaner
    ============

    Cleans raw text data for training.

    Common cleaning steps:
    1. Remove HTML tags
    2. Normalize whitespace
    3. Remove special characters
    4. Fix encoding issues
    5. Remove duplicates
    """

    def clean(self, text: str) -> str:
        """Clean a single text."""
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove control characters
        text = ''.join(c for c in text if c.isprintable() or c in '\n\t')

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts."""
        return [self.clean(t) for t in texts]


################################################################################
# SECTION 2: DEDUPLICATOR
################################################################################

class Deduplicator:
    """
    Data Deduplicator
    =================

    Removes duplicate or near-duplicate texts.

    Methods:
    1. Exact dedup: hash-based
    2. MinHash: approximate dedup for large datasets
    3. SimHash: near-duplicate detection

    Interview Questions:
        Q: "Why is deduplication important?"
        A: Duplicates cause overfitting to repeated content.
           Models memorize instead of learning patterns.
    """

    def __init__(self):
        self.seen = set()

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate."""
        hash_val = hash(text)
        if hash_val in self.seen:
            return True
        self.seen.add(hash_val)
        return False

    def deduplicate(self, texts: List[str]) -> List[str]:
        """Remove duplicates from text list."""
        return [t for t in texts if not self.is_duplicate(t)]


################################################################################
# SECTION 3: BATCH CONSTRUCTOR
################################################################################

class BatchConstructor:
    """
    Batch Constructor
    =================

    Creates training batches from tokenized data.

    Strategies:
    1. Fixed length: pad/truncate to same length
    2. Packing: pack multiple short sequences
    3. Dynamic batching: group similar lengths

    Interview Questions:
        Q: "How do you handle variable length sequences?"
        A: Padding (wasteful), packing (efficient), or dynamic batching.
    """

    def __init__(self, max_seq_len: int, batch_size: int):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def create_batches(
        self,
        token_ids: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create training batches.

        Args:
            token_ids: Full token sequence

        Returns:
            List of (input, target) batches
        """
        batches = []
        n_tokens = len(token_ids)

        for i in range(0, n_tokens - self.max_seq_len * 2, self.max_seq_len * self.batch_size):
            batch_inputs = []
            batch_targets = []

            for j in range(self.batch_size):
                start = i + j * self.max_seq_len
                if start + self.max_seq_len + 1 > n_tokens:
                    break

                batch_inputs.append(token_ids[start:start + self.max_seq_len])
                batch_targets.append(token_ids[start + 1:start + self.max_seq_len + 1])

            if len(batch_inputs) == self.batch_size:
                batches.append((
                    np.array(batch_inputs),
                    np.array(batch_targets)
                ))

        return batches


################################################################################
# SECTION 4: TESTING
################################################################################

def demonstrate_data_pipeline():
    """Demonstrate data pipeline."""
    print("=" * 70)
    print("DATA PIPELINE DEMONSTRATION")
    print("=" * 70)

    # Cleaner
    print("\n--- Data Cleaner ---")
    cleaner = DataCleaner()
    dirty_text = "  Hello   world!  \x00  "
    clean_text = cleaner.clean(dirty_text)
    print(f"Dirty: '{dirty_text}'")
    print(f"Clean: '{clean_text}'")

    # Deduplicator
    print("\n--- Deduplicator ---")
    dedup = Deduplicator()
    texts = ["hello", "world", "hello", "foo", "world"]
    unique = dedup.deduplicate(texts)
    print(f"Original: {texts}")
    print(f"Unique: {unique}")

    # Batch constructor
    print("\n--- Batch Constructor ---")
    constructor = BatchConstructor(max_seq_len=8, batch_size=2)
    token_ids = np.arange(50)
    batches = constructor.create_batches(token_ids)
    print(f"Number of batches: {len(batches)}")
    if batches:
        print(f"Batch shape: {batches[0][0].shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_data_pipeline()
