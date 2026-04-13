"""
################################################################################
TOKENIZATION — CONVERTING TEXT TO NUMBERS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Tokenization?
    Tokenization converts text into a sequence of integers (token IDs)
    that models can process. Each integer represents a token (word, subword,
    or character).

Why does it matter?
    Neural networks work with numbers, not text.
    The choice of tokenizer affects:
    - Vocabulary size
    - Token sequence length
    - Model performance
    - Language coverage

Types of Tokenization:
    1. Character-level: Each character is a token
    2. Word-level: Each word is a token
    3. Subword-level: Words are split into subwords (BPE, SentencePiece)
    4. Byte-level: Each byte is a token

Modern LLMs use subword tokenization:
    - GPT-2/3/4: BPE (50,257 tokens)
    - LLaMA: SentencePiece (32,000 tokens)
    - Mistral: SentencePiece (32,000 tokens)
    - BERT: WordPiece (30,522 tokens)

Interview Questions:
    1. "What tokenization do modern LLMs use?"
       Most use subword tokenization (BPE or SentencePiece).
       This balances vocabulary size with sequence length.

    2. "Why not word-level tokenization?"
       Vocabulary would be huge (millions of words).
       Rare words and misspellings would be unknown tokens.

    3. "Why not character-level?"
       Sequences would be very long (each char = 1 token).
       Models would need many layers to learn word patterns.

################################################################################
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
import re

################################################################################
# SECTION 1: SIMPLE TOKENIZER
################################################################################

class SimpleTokenizer:
    """
    Simple Tokenizer (for demonstration)
    ======================================

    A basic tokenizer that splits text into words and maps them to IDs.
    Not production quality, but useful for understanding concepts.
    """

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        # Initialize special tokens
        self._add_token(self.pad_token)
        self._add_token(self.unk_token)
        self._add_token(self.bos_token)
        self._add_token(self.eos_token)

    def _add_token(self, token: str) -> int:
        """Add token to vocabulary."""
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    def train(self, texts: List[str]):
        """
        Build vocabulary from text corpus.

        Algorithm:
        1. Count word frequencies
        2. Keep top (vocab_size - special_tokens) words
        3. Add to vocabulary
        """
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)

        # Keep top words
        most_common = word_counts.most_common(self.vocab_size - 4)
        for word, _ in most_common:
            self._add_token(word)

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        tokens = [self.token_to_id.get(self.bos_token)]
        for word in text.lower().split():
            token_id = self.token_to_id.get(word, self.token_to_id[self.unk_token])
            tokens.append(token_id)
        tokens.append(self.token_to_id.get(self.eos_token))
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        tokens = []
        for idx in token_ids:
            token = self.id_to_token.get(idx, self.unk_token)
            if token in [self.pad_token, self.bos_token, self.eos_token]:
                continue
            tokens.append(token)
        return " ".join(tokens)


################################################################################
# SECTION 2: BYTE PAIR ENCODING (BPE)
################################################################################

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer
    ====================================

    Definition: A subword tokenization algorithm that iteratively merges
    the most frequent pair of adjacent tokens.

    Algorithm:
    1. Start with character-level tokens
    2. Find most frequent adjacent pair
    3. Merge that pair into a new token
    4. Repeat until desired vocabulary size

    Example:
        "aaabdaaabac"
        Step 1: Most frequent pair: "aa" → merge to "Z"
        Step 2: "ZabdZabac"
        Step 3: Most frequent pair: "Za" → merge to "Y"
        Step 4: "YbdYbac"
        ...

    Used by:
    - GPT-2, GPT-3, GPT-4 (OpenAI)
    - LLaMA (with SentencePiece)
    - Most modern LLMs

    Interview Questions:
        1. "How does BPE work?"
           Start with characters, iteratively merge most frequent pairs.
           Common words become single tokens, rare words are split.

        2. "What's the advantage of BPE?"
           Handles any text (including rare words and misspellings).
           Common words are efficient (1 token), rare words are decomposed.
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def train(self, texts: List[str]):
        """
        Train BPE on text corpus.

        Algorithm:
        1. Initialize vocabulary with characters
        2. Count frequency of each adjacent pair
        3. Merge most frequent pair
        4. Repeat until vocab_size reached
        """
        # Start with character-level
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            for word in words:
                # Represent word as tuple of characters
                word_freqs[tuple(word)] += 1

        # Initialize vocabulary with characters
        vocab = set()
        for word in word_freqs:
            for char in word:
                vocab.add(char)

        # Merge iterations
        while len(vocab) < self.vocab_size:
            # Count pair frequencies
            pair_counts = Counter()
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Get most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            self.merges.append(best_pair)

            # Create new token
            new_token = best_pair[0] + best_pair[1]
            vocab.add(new_token)

            # Update word representations
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

        # Build token mappings
        for i, token in enumerate(sorted(vocab)):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def encode(self, text: str) -> List[int]:
        """
        Encode text using learned BPE merges.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        tokens = []
        for word in text.split():
            # Start with characters
            word_tokens = list(word)

            # Apply merges
            for merge in self.merges:
                new_word = []
                i = 0
                while i < len(word_tokens):
                    if (i < len(word_tokens) - 1 and
                        word_tokens[i] == merge[0] and
                        word_tokens[i + 1] == merge[1]):
                        new_word.append(merge[0] + merge[1])
                        i += 2
                    else:
                        new_word.append(word_tokens[i])
                        i += 1
                word_tokens = new_word

            # Convert to IDs
            for token in word_tokens:
                tokens.append(self.token_to_id.get(token, 0))

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.id_to_token.get(idx, "?") for idx in token_ids]
        return "".join(tokens)


################################################################################
# SECTION 3: TESTING & EXAMPLES
################################################################################

def demonstrate_tokenization():
    """Demonstrate tokenization concepts."""
    print("=" * 70)
    print("TOKENIZATION DEMONSTRATION")
    print("=" * 70)

    # Simple tokenizer
    print("\n--- Simple Tokenizer ---")
    tokenizer = SimpleTokenizer(vocab_size=100)
    texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a cat and a dog"
    ]
    tokenizer.train(texts)

    text = "the cat sat"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"Text: '{text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")

    # BPE tokenizer
    print("\n--- BPE Tokenizer ---")
    bpe = BPETokenizer(vocab_size=50)
    bpe.train(texts)

    text = "the cat"
    encoded = bpe.encode(text)
    print(f"Text: '{text}'")
    print(f"Encoded: {encoded}")
    print(f"Merges learned: {len(bpe.merges)}")

    # Show vocabulary
    print("\n--- Vocabulary ---")
    print(f"Simple tokenizer vocab size: {len(tokenizer.token_to_id)}")
    print(f"BPE vocab size: {len(bpe.token_to_id)}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_tokenization()


################################################################################
# REFERENCES
################################################################################

# [1] Sennrich, R., et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
# [2] Kudo, T., & Richardson, J. (2018). SentencePiece.
# [3] Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2).

################################################################################
