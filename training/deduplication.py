"""
################################################################################
DATA DEDUPLICATION — REMOVING DUPLICATES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Data Deduplication?
    Removing duplicate or near-duplicate data.

Methods:
    - Exact dedup: hash-based
    - MinHash: approximate dedup
    - SimHash: near-duplicate detection

Interview Questions:
    Q: "Why is deduplication important?"
    A: Duplicates cause overfitting and waste compute.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: DEDUPLICATOR
################################################################################

class Deduplicator:
    """
    Data Deduplicator
    =================

    Removes duplicate data.
    """

    def __init__(self):
        self.seen = set()

    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate."""
        hash_val = hash(text)
        if hash_val in self.seen:
            return True
        self.seen.add(hash_val)
        return False

    def deduplicate(self, texts: List[str]) -> List[str]:
        """Remove duplicates."""
        return [t for t in texts if not self.is_duplicate(t)]


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_deduplication():
    """Demonstrate deduplication."""
    print("=" * 70)
    print("DATA DEDUPLICATION DEMONSTRATION")
    print("=" * 70)

    dedup = Deduplicator()
    texts = ["hello", "world", "hello", "foo", "world"]
    unique = dedup.deduplicate(texts)
    print(f"Original: {len(texts)}")
    print(f"Unique: {len(unique)}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_deduplication()
