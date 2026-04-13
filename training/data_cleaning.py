"""
################################################################################
DATA CLEANING — PREPARING TRAINING DATA
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Data Cleaning?
    Removing noise and errors from training data.

Key Steps:
    - Remove duplicates
    - Fix encoding
    - Remove low-quality data
    - Normalize text

Interview Questions:
    Q: "How do you clean training data?"
    A: Remove duplicates, fix encoding, filter low-quality.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: DATA CLEANER
################################################################################

class DataCleaner:
    """
    Data Cleaner
    ============

    Cleans training data.
    """

    def __init__(self, min_length: int = 10, max_length: int = 10000):
        self.min_length = min_length
        self.max_length = max_length

    def clean(self, texts: List[str]) -> List[str]:
        """Clean text data."""
        cleaned = []
        for text in texts:
            # Remove extra whitespace
            text = ' '.join(text.split())

            # Filter by length
            if self.min_length <= len(text) <= self.max_length:
                cleaned.append(text)

        return cleaned


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_data_cleaning():
    """Demonstrate data cleaning."""
    print("=" * 70)
    print("DATA CLEANING DEMONSTRATION")
    print("=" * 70)

    cleaner = DataCleaner(min_length=5, max_length=100)
    texts = ["Hello world", "x", "This is a longer text that should be kept"]
    cleaned = cleaner.clean(texts)
    print(f"Original: {len(texts)}")
    print(f"Cleaned: {len(cleaned)}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_data_cleaning()
