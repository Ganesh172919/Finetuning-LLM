"""
################################################################################
BPE TOKENIZER — BYTE PAIR ENCODING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is BPE?
    Byte Pair Encoding is a subword tokenization algorithm that
    iteratively merges the most frequent pair of adjacent tokens.

Why BPE?
    - Word-level: vocabulary too large, can't handle rare words
    - Character-level: sequences too long, can't learn word patterns
    - Subword: balance between vocabulary size and sequence length

Algorithm:
    1. Start with character-level tokens
    2. Count frequency of each adjacent pair
    3. Merge the most frequent pair into a new token
    4. Repeat until desired vocabulary size

Example:
    Training corpus: "aaabdaaabac"
    Step 1: Count pairs: "aa"=4, "ab"=2, "bd"=1, ...
    Step 2: Merge "aa" → "Z"
    Step 3: "ZabdZabac"
    Step 4: Count pairs: "Za"=2, "ab"=2, ...
    Step 5: Merge "Za" → "Y"
    Step 6: "YbdYbac"
    ...

Used by:
    - GPT-2, GPT-3, GPT-4 (OpenAI)
    - LLaMA (with SentencePiece)
    - Mistral
    - Most modern LLMs

Interview Questions:
        Q: "How does BPE work?"
        A: Start with characters, iteratively merge most frequent pairs.
           Common words become single tokens, rare words are split.

        Q: "What's the advantage of BPE?"
        A: Handles any text (including rare words and misspellings).
           Common words are efficient (1 token), rare words are decomposed.

################################################################################
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

################################################################################
# SECTION 1: BPE TOKENIZER
################################################################################

class BPETokenizer:
    """
    BPE Tokenizer
    =============

    Learns subword vocabulary from text corpus using Byte Pair Encoding.

    Training:
    1. Initialize with characters
    2. Iteratively merge most frequent pairs
    3. Stop when vocabulary reaches desired size

    Encoding:
    1. Split text into characters
    2. Apply learned merges in order
    3. Map tokens to IDs

    Decoding:
    1. Map IDs to tokens
    2. Concatenate tokens
    3. Remove special characters

    Interview Questions:
        Q: "How do you choose vocabulary size?"
        A: Tradeoff: larger vocab = shorter sequences but more parameters.
           Typical: 32K-100K tokens.

        Q: "What's the difference between BPE and WordPiece?"
        A: BPE merges most frequent pair.
           WordPiece merges pair that maximizes likelihood.
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def train(self, texts: List[str]):
        """
        Train BPE on text corpus.

        Args:
            texts: List of training texts
        """
        # Count word frequencies
        word_freqs = Counter()
        for text in texts:
            for word in text.split():
                word_freqs[word] += 1

        # Initialize vocabulary with characters
        vocab = set()
        word_splits = {}
        for word, freq in word_freqs.items():
            chars = list(word)
            word_splits[word] = chars
            for char in chars:
                vocab.add(char)

        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        for token in special_tokens:
            vocab.add(token)

        # Merge iterations
        while len(vocab) < self.vocab_size:
            # Count pair frequencies
            pair_counts = Counter()
            for word, freq in word_freqs.items():
                chars = word_splits[word]
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Get most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            self.merges.append(best_pair)

            # Create new token
            new_token = best_pair[0] + best_pair[1]
            vocab.add(new_token)

            # Update word splits
            for word in word_splits:
                chars = word_splits[word]
                new_chars = []
                i = 0
                while i < len(chars):
                    if (i < len(chars) - 1 and
                        chars[i] == best_pair[0] and
                        chars[i + 1] == best_pair[1]):
                        new_chars.append(new_token)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                word_splits[word] = new_chars

        # Build token mappings
        for i, token in enumerate(sorted(vocab)):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            token_ids: List of token IDs
        """
        tokens = []

        # Add BOS token
        if '<BOS>' in self.token_to_id:
            tokens.append(self.token_to_id['<BOS>'])

        # Encode each word
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
                tokens.append(self.token_to_id.get(token, self.token_to_id['<UNK>']))

            # Add space token (if word is not last)
            if 'Ġ' in self.token_to_id:
                tokens.append(self.token_to_id['Ġ'])

        # Add EOS token
        if '<EOS>' in self.token_to_id:
            tokens.append(self.token_to_id['<EOS>'])

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs

        Returns:
            text: Decoded text
        """
        tokens = []
        for idx in token_ids:
            token = self.id_to_token.get(idx, '<UNK>')
            if token in ['<PAD>', '<BOS>', '<EOS>']:
                continue
            tokens.append(token)

        # Join and clean
        text = ''.join(tokens)
        text = text.replace('Ġ', ' ').strip()

        return text


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_bpe():
    """Demonstrate BPE tokenizer."""
    print("=" * 70)
    print("BPE TOKENIZER DEMONSTRATION")
    print("=" * 70)

    # Training data
    texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a cat and a dog played together",
        "the quick brown fox jumped over the lazy dog",
    ]

    # Train tokenizer
    print("\n--- Training ---")
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(texts)
    print(f"Vocabulary size: {len(tokenizer.token_to_id)}")
    print(f"Number of merges: {len(tokenizer.merges)}")
    print(f"First 10 merges: {tokenizer.merges[:10]}")

    # Encode
    print("\n--- Encoding ---")
    text = "the cat sat"
    encoded = tokenizer.encode(text)
    print(f"Text: '{text}'")
    print(f"Encoded: {encoded}")
    print(f"Tokens: {[tokenizer.id_to_token[id] for id in encoded]}")

    # Decode
    print("\n--- Decoding ---")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_bpe()


################################################################################
# REFERENCES
################################################################################

# [1] Sennrich, R., et al. (2016). Neural Machine Translation of Rare Words with Subword Units.
# [2] Kudo, T., & Richardson, J. (2018). SentencePiece.

################################################################################
