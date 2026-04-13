"""
################################################################################
VOCABULARY CONSTRUCTION — BUILDING TOKEN VOCABULARIES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Vocabulary Construction?
    Building the set of tokens for tokenization.

Methods:
    - BPE: Byte Pair Encoding
    - WordPiece: Used by BERT
    - SentencePiece: Used by LLaMA

Interview Questions:
    Q: "How do you build a vocabulary?"
    A: Use BPE or SentencePiece on training corpus.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: VOCABULARY BUILDER
################################################################################

class VocabularyBuilder:
    """
    Vocabulary Builder
    ==================

    Builds token vocabulary from text.
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}

    def build(self, texts: List[str]):
        """Build vocabulary from texts."""
        # Count characters
        char_counts = {}
        for text in texts:
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1

        # Add most common
        sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])
        for i, (char, _) in enumerate(sorted_chars[:self.vocab_size]):
            self.token_to_id[char] = i
            self.id_to_token[i] = char


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_vocabulary():
    """Demonstrate vocabulary construction."""
    print("=" * 70)
    print("VOCABULARY CONSTRUCTION DEMONSTRATION")
    print("=" * 70)

    builder = VocabularyBuilder(vocab_size=100)
    texts = ["hello world", "foo bar", "test"]
    builder.build(texts)
    print(f"Vocabulary size: {len(builder.token_to_id)}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_vocabulary()
