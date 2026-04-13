"""
################################################################################
BYTE-LEVEL BPE TOKENIZER — TRAINING & INFERENCE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Byte-Level BPE Tokenizer?
    A tokenizer that compresses text by iteratively merging the most frequent
    adjacent byte pairs into new tokens. "Byte-level" means the atomic unit
    is a single byte (0-255), not a Unicode character. This guarantees that
    every possible byte sequence — text, code, binary data, emoji, any
    language — can be encoded without ever producing an "unknown" token.

Why does it matter?
    Tokenization is the bridge between raw text and model inputs. Poor
    tokenization causes:
      - Inflated sequence lengths (wasting compute)
      - Inability to represent certain inputs (UNK tokens)
      - Broken arithmetic (e.g., "1234" as one token confuses the model)
    Byte-level BPE solves all three: it handles any input, compresses common
    patterns into short tokens, and with digit-splitting preserves numerical
    structure.

How does it work?
    TRAINING:
      1. Start with a base vocabulary of 256 byte-level tokens (one per byte).
      2. Pre-tokenize the corpus using regex patterns (split into words,
         numbers, punctuation, whitespace chunks).
      3. Convert each chunk to its byte sequence.
      4. Count frequency of every adjacent byte pair across all chunks.
      5. Merge the most frequent pair into a new token. Add it to the vocab.
      6. Repeat steps 3-5 until the desired vocabulary size is reached.

    ENCODING:
      1. Apply the same regex pre-tokenization to split input into chunks.
      2. Convert each chunk to bytes.
      3. Apply learned merges in priority order (earliest-learned = highest
         priority, greedily matching longest sequences first).
      4. Map each resulting token to its integer ID.

    DECODING:
      1. Map each token ID back to its byte sequence.
      2. Concatenate all byte sequences.
      3. Decode bytes as UTF-8.

########################################

ARCHITECTURE DIAGRAM (ASCII art):

    ┌──────────────────────────────────────────────────────────────────┐
    │                    BYTE-LEVEL BPE TOKENIZER                      │
    │                                                                  │
    │  ENCODING PIPELINE:                                              │
    │                                                                  │
    │  "Hello, world! 123"                                             │
    │       │                                                          │
    │       ▼                                                          │
    │  ┌─────────────────────┐                                         │
    │  │ Regex Pre-Tokenizer │  Split into chunks by pattern           │
    │  │  (GPT-4-style)      │  ["Hello", ",", " world", "!", " 123"]  │
    │  └──────────┬──────────┘                                         │
    │             ▼                                                    │
    │  ┌─────────────────────┐                                         │
    │  │ Digit Splitting     │  "123" → ["1", "2", "3"]               │
    │  └──────────┬──────────┘                                         │
    │             ▼                                                    │
    │  ┌─────────────────────┐                                         │
    │  │ Byte Conversion     │  Each chunk → byte sequence             │
    │  └──────────┬──────────┘                                         │
    │             ▼                                                    │
    │  ┌─────────────────────┐                                         │
    │  │ BPE Merge Engine    │  Apply merges in priority order         │
    │  │ (greedy, in-order)  │  [72, 101] → [token_id_for_He]         │
    │  └──────────┬──────────┘                                         │
    │             ▼                                                    │
    │  ┌─────────────────────┐                                         │
    │  │ Token ID Mapping    │  Map tokens to integer IDs              │
    │  └──────────┬──────────┘                                         │
    │             ▼                                                    │
    │  [20912, 11, 995, 0, 16, 17, 18]                                │
    │                                                                  │
    │  DECODING: IDs → bytes → UTF-8 string                           │
    └──────────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2015: Sennrich et al. introduce BPE for neural machine translation
             ("Neural Machine Translation of Rare Words with Subword Units")
    - 2018: Radford et al. apply BPE at byte level in GPT-1
    - 2019: GPT-2 uses byte-level BPE with 50,257 tokens
    - 2020: SentencePiece (Kudo & Richardson) popularizes Unigram LM tokenizer
    - 2022: OpenAI's tiktoken provides fast Rust-based byte-level BPE
    - 2023: GPT-4 uses cl100k_base with 100,256 tokens
    - 2024: Most frontier LLMs (Claude, Gemini, Llama 3) use byte-level BPE
             with vocab sizes in the 100k-200k range

INTERVIEW QUESTIONS:
    1. "Why byte-level BPE instead of character-level BPE?"
       Character-level requires a predefined character set and produces UNK
       for out-of-vocabulary characters. Byte-level starts with just 256
       atomic tokens (one per byte), so ANY byte sequence can be represented.
       This handles all languages, code, binary data, and emoji without UNK.

    2. "What is the tradeoff in choosing vocabulary size?"
       Larger vocab (200k) → shorter sequences (fewer tokens per text) but
       bigger embedding tables and softmax layers (more parameters, more
       memory). Smaller vocab (50k) → longer sequences (more tokens) but
       smaller model components. The sweet spot for modern LLMs is 100k-150k.
       Empirically, 128k offers a good balance for multilingual + code models.

    3. "Why split digits individually instead of merging them?"
       If "1234" is a single token, the model cannot generalize arithmetic
       to "1235" or "5678" — each number is an atomic lookup. Splitting into
       ["1","2","3","4"] lets the model learn positional number patterns.
       GPT-2/3 era showed this materially improves arithmetic capability.

    4. "How does the regex pre-tokenizer prevent bad merges?"
       Without pre-tokenization, BPE might merge characters across word
       boundaries (e.g., "e the" → a single token), creating meaningless
       tokens that waste vocabulary slots. The regex splits text into
       linguistically meaningful chunks BEFORE byte-pair merging, ensuring
       merges only occur within words, numbers, or punctuation groups.

################################################################################
"""

import re
import json
import os
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field


################################################################################
# SECTION 1: REGEX PRE-TOKENIZATION PATTERNS
################################################################################

# GPT-4-style regex patterns for splitting text into chunks before BPE.
# Each pattern handles a specific linguistic category.
# The order matters: earlier patterns have priority.
#
# Why regex pre-tokenization?
#   Without it, BPE would merge bytes across word/number boundaries,
#   creating meaningless tokens. Pre-tokenization constrains merges
#   to occur only within linguistically coherent chunks.

# NOTE: Python's `re` module does not support Unicode property escapes
# like \p{L} or \p{N}. We use equivalent character classes:
#   \p{L}  →  [a-zA-Z] for ASCII letters (sufficient for pre-tokenization)
#   \p{N}  →  [0-9] for digits
#
# For production use, the `regex` package supports \p{L} and \p{N},
# but we avoid external dependencies per CLAUDE.md conventions.

GPT4_SPLIT_PATTERN = "|".join([
    # Pattern 1: Contractions — "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"
    # These are common English contractions that should stay as one chunk.
    # (?i) makes it case-insensitive so "Don'T" is handled correctly.
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",

    # Pattern 2: Letter sequences — words like "Hello", "world"
    # [^\r\n\p{L}\p{N}]? optionally matches a leading non-alphanumeric char
    # (like a quote or bracket) that is NOT a newline, then \p{L}+ matches
    # one or more letters. In Python's re module, \p{L} is not supported,
    # so we use [a-zA-Z] as the ASCII equivalent.
    r"[^\r\na-zA-Z0-9]?[a-zA-Z]+",

    # Pattern 3: Numbers split into 1-3 digit chunks
    # \p{N}{1,3} matches 1 to 3 consecutive digits. This is CRITICAL:
    # it ensures "123456" becomes ["123","456"] rather than one token,
    # and with digit-level merging, each digit stays separate.
    # This materially improves arithmetic capability (GPT-2/3 lesson).
    # We use [0-9] as the ASCII equivalent of \p{N}.
    r"[0-9]{1,3}",

    # Pattern 4: Non-letter/number sequences followed by optional newlines
    # Matches punctuation, symbols, etc. that are NOT letters/numbers,
    # optionally ending with \r\n. This handles things like "!!!\n".
    # Uses [a-zA-Z0-9] instead of \p{L}\p{N}.
    r"[^\sa-zA-Z0-9]+[\r\n]*",

    # Pattern 5: Whitespace followed by newlines
    # Matches whitespace that contains at least one newline.
    # Separating this from pattern 7 preserves newline structure.
    r"\s*[\r\n]+",

    # Pattern 6: Trailing whitespace (before a non-whitespace char)
    # \s+(?!\S) matches whitespace only when followed by non-whitespace.
    # This separates trailing spaces from the next word.
    r"\s+(?!\S)",

    # Pattern 7: Other whitespace
    # Catches any remaining whitespace (spaces, tabs, etc.).
    r"\s+",
])

# Compile once for performance
_PRE_TOKENIZE_RE = re.compile(GPT4_SPLIT_PATTERN)

# Simplified pattern for digit splitting
_DIGIT_RE = re.compile(r"[0-9]")


################################################################################
# SECTION 2: SPECIAL TOKEN DEFINITIONS
################################################################################

# Special tokens are reserved tokens with specific semantic meaning.
# They are placed AFTER the byte-level vocab (IDs 0-255) and any learned
# BPE tokens, so they never conflict with byte encodings.
#
# Why these specific tokens?
#   - BOS/EOS/PAD: Standard sequence delimiters
#   - System/User/Assistant: Chat role markers (ChatML format)
#   - Tool call/result: Function calling support
#   - Think/Think-end: Chain-of-thought reasoning markers

SPECIAL_TOKENS = [
    "<|bos|>",       # Beginning of sequence
    "<|eos|>",       # End of sequence
    "<|pad|>",       # Padding token
    "<|system|>",    # System message role
    "<|user|>",      # User message role
    "<|assistant|>", # Assistant message role
    "<|tool_call|>", # Tool/function call marker
    "<|tool_result|>", # Tool/function result marker
    "<|think|>",     # Chain-of-thought reasoning start
    "</think|>",     # Chain-of-thought reasoning end
]

# Mapping from special token string to its ID offset.
# Actual IDs = base_vocab_size + learned_tokens + offset.
# This is computed dynamically after training.
SPECIAL_TOKEN_OFFSETS = {token: i for i, token in enumerate(SPECIAL_TOKENS)}


################################################################################
# SECTION 3: BYTE-LEVEL BPE TOKENIZER
################################################################################

class ByteLevelBPETokenizer:
    """
    Byte-Level BPE Tokenizer
    =========================

    A tokenizer that compresses text by iteratively merging the most frequent
    adjacent byte pairs. Uses byte-level fallback (all 256 bytes are always
    representable) and regex pre-tokenization (GPT-4-style patterns).

    Vocabulary structure:
        - IDs 0-255: Byte-level tokens (one per byte value)
        - IDs 256 to 256+N-1: Learned BPE merge tokens
        - IDs 256+N to 256+N+9: Special tokens

    Formula:
        V_final = 256 (bytes) + N (merges) + len(SPECIAL_TOKENS)

    Step by step:
        1. Start with 256 byte tokens as base vocabulary.
        2. Pre-tokenize corpus with regex to get word-level chunks.
        3. Convert each chunk to a sequence of byte tokens.
        4. Find the most frequent adjacent pair across all chunks.
        5. Merge that pair into a new token. Record the merge rule.
        6. Repeat until vocabulary reaches target size.

    WHY byte-level (not character-level)?
        Characters require a predefined alphabet. Bytes do not. Any Unicode
        text is just a sequence of bytes in UTF-8 encoding, so byte-level
        BPE can represent ANY input — English, Chinese, emoji, code, even
        raw binary data — without ever producing an "unknown" token.

    Interview Question:
        "Why not just use character-level tokenization?"
        Character-level maps each Unicode character to a token. This fails
        for rare characters (UNK), requires a fixed character set, and
        produces very long sequences for languages with large character sets
        (Chinese has 50,000+ characters). Byte-level avoids all these issues
        by working at the byte level — there are only 256 possible bytes.
    """

    def __init__(self, vocab_size: int = 128_000):
        """
        Initialize the Byte-Level BPE Tokenizer.

        Args:
            vocab_size: Target vocabulary size. Must be >= 266 (256 bytes +
                        10 special tokens). Recommended range: 100k-200k.
                        Default 128k, which balances sequence compression
                        against embedding table size.

        Returns:
            None (constructor).

        Explanation:
            Initializes empty vocab and merge structures. The vocabulary
            is populated by calling `train()` or `load()`.

            Vocab size tradeoff:
                - Larger vocab (200k): Shorter sequences, fewer tokens per text,
                  but bigger embedding tables (more parameters, more memory).
                - Smaller vocab (50k): Longer sequences, more tokens per text,
                  but smaller model components.
                - 128k is a sweet spot for multilingual + code models.

        Example:
            >>> tokenizer = ByteLevelBPETokenizer(vocab_size=128_000)
            >>> print(tokenizer.vocab_size)
            128000
        """
        # Validate vocab size: must fit at least 256 bytes + 10 special tokens
        min_vocab = 256 + len(SPECIAL_TOKENS)
        if vocab_size < min_vocab:
            raise ValueError(
                f"vocab_size must be >= {min_vocab} (256 bytes + "
                f"{len(SPECIAL_TOKENS)} special tokens), got {vocab_size}"
            )

        self.target_vocab_size = vocab_size

        # Core data structures:
        # vocab: token_bytes (tuple of ints) → token_id
        # inverse_vocab: token_id → token_bytes (tuple of ints)
        # merges: ordered list of (token_a_bytes, token_b_bytes) → merged_bytes
        self.vocab: Dict[Tuple[int, ...], int] = {}
        self.inverse_vocab: Dict[int, Tuple[int, ...]] = {}
        self.merges: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

        # Special token mappings (populated after training)
        self.special_token_to_id: Dict[str, int] = {}
        self.id_to_special_token: Dict[int, str] = {}

        # Initialize base vocabulary with all 256 byte values
        self._init_base_vocab()

    def _init_base_vocab(self):
        """
        Initialize base vocabulary with all 256 single-byte tokens.

        Args:
            None.

        Returns:
            None.

        Explanation:
            Every byte value 0-255 gets its own token. This is the foundation
            of byte-level BPE — it guarantees that ANY byte sequence can be
            represented. Byte value N maps to token ID N.

            This is why byte-level BPE never needs UNK tokens: even if a
            character is not in the learned vocabulary, its UTF-8 bytes are
            always representable.

        Example:
            >>> tokenizer = ByteLevelBPETokenizer()
            >>> # Byte 65 ('A') should map to token ID 65
            >>> assert tokenizer.vocab[(65,)] == 65
        """
        for byte_val in range(256):
            token_bytes = (byte_val,)
            self.vocab[token_bytes] = byte_val
            self.inverse_vocab[byte_val] = token_bytes

    def _register_special_tokens(self):
        """
        Register special tokens in the vocabulary after learned tokens.

        Args:
            None.

        Returns:
            None.

        Explanation:
            Special tokens are added AFTER all byte tokens and learned BPE
            tokens. Their IDs are sequential starting from the next available
            ID. This ensures they never conflict with byte encodings.

            Special tokens are treated as atomic units during encoding — they
            are never split or merged.

        Example:
            >>> tokenizer = ByteLevelBPETokenizer()
            >>> tokenizer._register_special_tokens()
            >>> bos_id = tokenizer.special_token_to_id["<|bos|>"]
        """
        next_id = len(self.vocab)
        for token in SPECIAL_TOKENS:
            self.special_token_to_id[token] = next_id
            self.id_to_special_token[next_id] = token
            next_id += 1

    ################################################################################
    # SECTION 3A: PRE-TOKENIZATION
    ################################################################################

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Split text into chunks using GPT-4-style regex patterns.

        Args:
            text: Raw input text.

        Returns:
            List of text chunks, each of which will be independently
            byte-encoded and BPE-merged.

        Explanation:
            Pre-tokenization is CRITICAL for BPE quality. Without it, BPE
            would merge bytes across word boundaries (e.g., "e the" might
            become a single token), wasting vocabulary slots on meaningless
            cross-boundary patterns.

            The regex patterns split text into:
              - Contractions ("'s", "'re", etc.)
              - Letter sequences (words)
              - Number sequences (1-3 digit chunks)
              - Punctuation groups
              - Whitespace groups (separated by newlines)

            This ensures merges only occur within linguistically coherent
            chunks, producing a much higher-quality vocabulary.

        Example:
            >>> tokenizer = ByteLevelBPETokenizer()
            >>> chunks = tokenizer._pre_tokenize("Hello, world! 123")
            >>> print(chunks)
            ['Hello', ',', ' world', '!', ' ', '1', '2', '3']
        """
        # Apply the compiled regex to split text into chunks
        chunks = _PRE_TOKENIZE_RE.findall(text)
        return chunks

    def _split_digits(self, chunk: str) -> List[str]:
        """
        Split digit sequences into individual digits.

        Args:
            chunk: A text chunk that may contain digit sequences.

        Returns:
            List of sub-chunks with digits split into individual characters.

        Explanation:
            This is a CRITICAL design choice. If "1234" is a single token,
            the model cannot generalize arithmetic to "1235" or "5678".
            By splitting into ["1", "2", "3", "4"], each digit becomes a
            separate token (or merges with adjacent context), allowing the
            model to learn positional number patterns.

            This was a key lesson from the GPT-2/3 era: models with merged
            number tokens performed poorly on arithmetic tasks.

        Example:
            >>> tokenizer = ByteLevelBPETokenizer()
            >>> result = tokenizer._split_digits("abc123def")
            >>> print(result)
            ['abc', '1', '2', '3', 'def']
        """
        # Split the chunk into runs of digits and non-digits
        parts = _DIGIT_RE.split(chunk)
        digits = _DIGIT_RE.findall(chunk)

        # Interleave: for each digit, put it as a separate element
        result = []
        for i, part in enumerate(parts):
            if part:
                result.append(part)
            if i < len(digits):
                result.append(digits[i])
        return result

    def _tokenize_chunk(self, chunk: str) -> List[Tuple[int, ...]]:
        """
        Convert a text chunk to a list of byte-tuple tokens.

        Args:
            chunk: A pre-tokenized text chunk.

        Returns:
            List of byte tuples, each representing a token before BPE merging.
            Initially, each tuple is a single byte.

        Explanation:
            Converts the chunk to UTF-8 bytes, then represents each byte
            as a single-element tuple. These are the atomic units that BPE
            will merge.

            UTF-8 encoding is used because:
              - It's the standard encoding for text on the web
              - ASCII characters (0-127) map to single bytes
              - Non-ASCII characters use 2-4 bytes, but their bytes are
                always in the base vocabulary (0-255)

        Example:
            >>> tokenizer = ByteLevelBPETokenizer()
            >>> tokens = tokenizer._tokenize_chunk("Hi")
            >>> print(tokens)
            [(72,), (105,)]
        """
        byte_values = chunk.encode("utf-8")
        return [(b,) for b in byte_values]

    ################################################################################
    # SECTION 3B: BPE TRAINING
    ################################################################################

    def train(self, texts: List[str], vocab_size: Optional[int] = None):
        """
        Train the BPE tokenizer on a corpus of texts.

        Args:
            texts: List of training texts (strings).
            vocab_size: Target vocabulary size. If None, uses self.target_vocab_size.

        Returns:
            None (modifies self.vocab and self.merges in place).

        Explanation:
            BPE TRAINING ALGORITHM:

            1. Pre-tokenize all texts using regex patterns.
            2. Split digit sequences into individual digits.
            3. Convert each chunk to byte-level tokens.
            4. Count frequency of every adjacent byte pair across all chunks.
            5. Find the most frequent pair.
            6. Merge that pair everywhere it occurs, creating a new token.
            7. Record the merge rule.
            8. Repeat steps 4-7 until vocabulary reaches target size.

            WHY merge most frequent first?
                The most frequent pair represents the most common bigram in the
                corpus. Merging it maximizes compression (fewest tokens needed
                to represent the corpus). This greedy strategy produces a
                vocabulary where common words/phrases become single tokens.

            WHY not merge across newlines?
                Newlines are structural signals in code and text. Merging a
                token that spans a newline (e.g., "end\n" + "def ") would
                create a token that only appears at that specific structural
                boundary, wasting a vocabulary slot. The pre-tokenizer ensures
                newlines are separate chunks.

        Example:
            >>> tokenizer = ByteLevelBPETokenizer(vocab_size=300)
            >>> tokenizer.train(["hello world", "hello there"], vocab_size=300)
            >>> print(len(tokenizer.merges))  # Should be 300 - 256 - 10 = 34
        """
        if vocab_size is not None:
            self.target_vocab_size = vocab_size

        # Calculate how many merges we need
        # Total vocab = 256 (bytes) + N (merges) + len(SPECIAL_TOKENS)
        num_merges = self.target_vocab_size - 256 - len(SPECIAL_TOKENS)
        if num_merges <= 0:
            raise ValueError(
                f"vocab_size ({self.target_vocab_size}) too small. "
                f"Need at least {256 + len(SPECIAL_TOKENS) + 1}."
            )

        print(f"Training BPE tokenizer with {num_merges} merges...")
        print(f"Target vocab size: {self.target_vocab_size}")

        # Step 1: Pre-tokenize and convert to byte-level tokens
        print("Step 1: Pre-tokenizing corpus...")
        all_chunks_tokens: List[List[Tuple[int, ...]]] = []
        for text in texts:
            chunks = self._pre_tokenize(text)
            for chunk in chunks:
                # Split digits into individual characters
                sub_chunks = self._split_digits(chunk)
                for sub_chunk in sub_chunks:
                    tokens = self._tokenize_chunk(sub_chunk)
                    if tokens:  # Skip empty chunks
                        all_chunks_tokens.append(tokens)

        print(f"  Found {len(all_chunks_tokens)} chunks to process.")

        # Step 2: Iteratively merge most frequent pairs
        print("Step 2: Learning BPE merges...")
        for merge_idx in range(num_merges):
            # Count all adjacent pairs across all chunks
            pair_counts: Counter = Counter()
            for tokens in all_chunks_tokens:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] += 1

            if not pair_counts:
                print(f"  No more pairs to merge at step {merge_idx}. Stopping early.")
                break

            # Find the most frequent pair
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            token_a, token_b = most_frequent_pair

            # Create new merged token
            merged_bytes = token_a + token_b

            # Check if merged token already exists (shouldn't, but safety)
            if merged_bytes in self.vocab:
                continue

            # Assign new token ID
            new_id = len(self.vocab)
            self.vocab[merged_bytes] = new_id
            self.inverse_vocab[new_id] = merged_bytes
            self.merges.append((token_a, token_b))

            # Apply this merge to all chunks
            new_chunks_tokens = []
            for tokens in all_chunks_tokens:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1
                            and tokens[i] == token_a
                            and tokens[i + 1] == token_b):
                        # Merge this pair
                        new_tokens.append(merged_bytes)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_chunks_tokens.append(new_tokens)
            all_chunks_tokens = new_chunks_tokens

            # Progress reporting
            if (merge_idx + 1) % 100 == 0 or merge_idx == 0:
                print(f"  Merge {merge_idx + 1}/{num_merges}: "
                      f"{token_a} + {token_b} -> {merged_bytes} "
                      f"(freq={pair_counts[most_frequent_pair]})")

        # Step 3: Register special tokens
        self._register_special_tokens()

        print(f"Training complete. Vocabulary size: {len(self.vocab)}")
        print(f"  Byte tokens: 256")
        print(f"  Learned merges: {len(self.merges)}")
        print(f"  Special tokens: {len(SPECIAL_TOKENS)}")

    ################################################################################
    # SECTION 3C: ENCODING
    ################################################################################

    def encode(self, text: str, allowed_special: str = "none") -> List[int]:
        """
        Encode text into a sequence of token IDs.

        Args:
            text: Input text to encode.
            allowed_special: How to handle special tokens in the text.
                - "none": Treat special token strings as regular text.
                - "all": Recognize and encode all special tokens.
                - "auto": Recognize only special tokens that appear in text.

        Returns:
            List of integer token IDs.

        Explanation:
            ENCODING ALGORITHM:

            1. If allowed_special != "none", find and handle special tokens first.
            2. Apply regex pre-tokenization to split text into chunks.
            3. Split digit sequences into individual digits.
            4. Convert each chunk to byte-level tokens.
            5. Apply learned BPE merges in priority order:
               - For each merge rule (in order learned), scan the token sequence
                 and merge every occurrence of that pair.
               - This is greedy: once merged, a token participates in later merges.
            6. Map each resulting token to its integer ID.

            WHY apply merges in order?
                The order of merges encodes priority. The first merge learned
                (e.g., merging "t" + "h" → "th") has highest priority. If we
                applied merges out of order, we'd get different tokenizations
                for the same text, breaking consistency.

        Example:
            >>> tokenizer = ByteLevelBPETokenizer()
            >>> tokenizer.train(["hello world hello"])
            >>> ids = tokenizer.encode("hello world")
            >>> print(ids)  # Should be a list of integer IDs
        """
        if not text:
            return []

        # Handle special tokens if allowed
        if allowed_special != "none":
            # Find special token positions in text
            special_positions = []
            for token in SPECIAL_TOKENS:
                if allowed_special == "all" or (
                    allowed_special == "auto" and token in text
                ):
                    start = 0
                    while True:
                        idx = text.find(token, start)
                        if idx == -1:
                            break
                        special_positions.append((idx, idx + len(token), token))
                        start = idx + 1

            # Sort by position and encode segments between special tokens
            special_positions.sort()
            if special_positions:
                result_ids = []
                last_end = 0
                for start, end, token in special_positions:
                    # Encode text before this special token
                    if start > last_end:
                        segment = text[last_end:start]
                        result_ids.extend(self._encode_segment(segment))
                    # Encode the special token itself
                    result_ids.append(self.special_token_to_id[token])
                    last_end = end
                # Encode remaining text after last special token
                if last_end < len(text):
                    result_ids.extend(self._encode_segment(text[last_end:]))
                return result_ids

        return self._encode_segment(text)

    def _encode_segment(self, text: str) -> List[int]:
        """
        Encode a text segment (without special tokens) into token IDs.

        Args:
            text: A text segment to encode. Must not contain special tokens.

        Returns:
            List of integer token IDs.

        Explanation:
            This is the core encoding logic:
            1. Pre-tokenize with regex.
            2. Split digits.
            3. Convert to byte tokens.
            4. Apply BPE merges greedily in order.
            5. Map to IDs.

        Example:
            >>> tokenizer = ByteLevelBPETokenizer()
            >>> ids = tokenizer._encode_segment("hello")
        """
        if not text:
            return []

        # Step 1: Pre-tokenize
        chunks = self._pre_tokenize(text)

        all_ids = []
        for chunk in chunks:
            # Step 2: Split digits
            sub_chunks = self._split_digits(chunk)

            for sub_chunk in sub_chunks:
                # Step 3: Convert to byte-level tokens
                tokens = self._tokenize_chunk(sub_chunk)
                if not tokens:
                    continue

                # Step 4: Apply BPE merges in order
                for token_a, token_b in self.merges:
                    merged_bytes = token_a + token_b
                    new_tokens = []
                    i = 0
                    while i < len(tokens):
                        if (i < len(tokens) - 1
                                and tokens[i] == token_a
                                and tokens[i + 1] == token_b):
                            new_tokens.append(merged_bytes)
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    tokens = new_tokens

                # Step 5: Map to IDs
                for token in tokens:
                    all_ids.append(self.vocab[token])

        return all_ids

    ################################################################################
    # SECTION 3D: DECODING
    ################################################################################

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs back to text.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded text string.

        Explanation:
            DECODING ALGORITHM:

            1. For each token ID, look up its byte sequence.
            2. Handle special tokens by inserting their string representation.
            3. Concatenate all byte sequences.
            4. Decode the concatenated bytes as UTF-8.

            Error handling:
                - Unknown IDs are skipped with a warning.
                - Invalid UTF-8 sequences are replaced with the Unicode
                  replacement character (U+FFFD).

        Example:
            >>> tokenizer = ByteLevelBPETokenizer()
            >>> tokenizer.train(["hello world"])
            >>> ids = tokenizer.encode("hello")
            >>> text = tokenizer.decode(ids)
            >>> print(text)
            hello
        """
        byte_chunks = []
        for token_id in ids:
            # Check if it's a special token
            if token_id in self.id_to_special_token:
                byte_chunks.append(
                    self.id_to_special_token[token_id].encode("utf-8")
                )
            elif token_id in self.inverse_vocab:
                byte_chunks.append(bytes(self.inverse_vocab[token_id]))
            else:
                # Unknown token ID — skip with a warning
                # WHY skip instead of raising? Robustness: a single bad ID
                # shouldn't crash the entire decode operation.
                import sys
                print(
                    f"Warning: Unknown token ID {token_id}, skipping.",
                    file=sys.stderr,
                )

        # Concatenate all byte sequences and decode as UTF-8
        raw_bytes = b"".join(byte_chunks)
        return raw_bytes.decode("utf-8", errors="replace")

    ################################################################################
    # SECTION 3E: SAVE & LOAD
    ################################################################################

    def save(self, path: str):
        """
        Save the trained tokenizer to disk.

        Args:
            path: Directory path to save tokenizer files. Created if it
                  does not exist.

        Returns:
            None.

        Explanation:
            Saves three files:
              - merges.json: Ordered list of BPE merge rules.
              - vocab.json: Complete vocabulary mapping (bytes → ID).
              - special_tokens.json: Special token mappings.

            The merge ORDER is critical — it encodes priority. Loading
            merges in a different order would produce different tokenizations.

        Example:
            >>> tokenizer = ByteLevelBPETokenizer()
            >>> tokenizer.train(["hello world"])
            >>> tokenizer.save("./my_tokenizer")
        """
        os.makedirs(path, exist_ok=True)

        # Save merges (order matters!)
        merges_data = [
            [list(token_a), list(token_b)]
            for token_a, token_b in self.merges
        ]
        with open(os.path.join(path, "merges.json"), "w") as f:
            json.dump(merges_data, f, indent=2)

        # Save vocabulary
        vocab_data = {
            str(list(k)): v for k, v in self.vocab.items()
        }
        with open(os.path.join(path, "vocab.json"), "w") as f:
            json.dump(vocab_data, f, indent=2)

        # Save special token mappings
        with open(os.path.join(path, "special_tokens.json"), "w") as f:
            json.dump(self.special_token_to_id, f, indent=2)

        # Save metadata
        metadata = {
            "target_vocab_size": self.target_vocab_size,
            "actual_vocab_size": len(self.vocab),
            "num_merges": len(self.merges),
            "num_special_tokens": len(SPECIAL_TOKENS),
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """
        Load a trained tokenizer from disk.

        Args:
            path: Directory path containing tokenizer files.

        Returns:
            None (modifies self in place).

        Explanation:
            Loads merges, vocabulary, and special token mappings from disk.
            The merge order is preserved exactly, ensuring identical
            tokenization behavior to the saved model.

        Example:
            >>> tokenizer = ByteLevelBPETokenizer()
            >>> tokenizer.load("./my_tokenizer")
            >>> ids = tokenizer.encode("hello world")
        """
        # Load metadata
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        self.target_vocab_size = metadata["target_vocab_size"]

        # Load merges
        with open(os.path.join(path, "merges.json"), "r") as f:
            merges_data = json.load(f)
        self.merges = [
            (tuple(pair[0]), tuple(pair[1])) for pair in merges_data
        ]

        # Rebuild vocabulary from base + merges
        self.vocab = {}
        self.inverse_vocab = {}
        self._init_base_vocab()

        # Add merged tokens in order
        for token_a, token_b in self.merges:
            merged_bytes = token_a + token_b
            if merged_bytes not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[merged_bytes] = new_id
                self.inverse_vocab[new_id] = merged_bytes

        # Load special token mappings
        with open(os.path.join(path, "special_tokens.json"), "r") as f:
            self.special_token_to_id = json.load(f)
        self.id_to_special_token = {
            v: k for k, v in self.special_token_to_id.items()
        }

        print(f"Tokenizer loaded from {path}")
        print(f"  Vocabulary size: {len(self.vocab)}")
        print(f"  Merges: {len(self.merges)}")

    @property
    def vocab_size(self) -> int:
        """
        Get the current vocabulary size.

        Args:
            None.

        Returns:
            Integer vocabulary size (bytes + merges + special tokens).
        """
        return len(self.vocab) + len(SPECIAL_TOKENS)


################################################################################
# SECTION 4: CHAT TEMPLATE
################################################################################

class ChatTemplate:
    """
    Chat Template
    =============

    Formats a list of chat messages into a single string using the tokenizer's
    special tokens. Follows the ChatML format used by GPT-4 and many other
    instruction-tuned models.

    Format:
        <|system|>
        {system_message}
        <|user|>
        {user_message}
        <|assistant|>
        {assistant_message}
        ...

    Step by step:
        1. For each message, determine its role (system/user/assistant).
        2. Wrap the message content with the appropriate role tokens.
        3. Add generation prompt token (<|assistant|>) at the end if requested.
        4. Optionally add BOS/EOS tokens.

    WHY this format?
        The ChatML format is widely adopted because:
          - It's human-readable (easy to debug).
          - It clearly delineates roles (no ambiguity about who said what).
          - It supports system prompts (for behavior steering).
          - It's extensible (tool calls, chain-of-thought, etc.).

    Interview Question:
        "Why do we need chat templates?"
        Without standardized formatting, each model would use ad-hoc delimiters
        (e.g., "User:", "Assistant:"), leading to ambiguity (what if the user
        types "Assistant:"?), inconsistent training, and inability to swap models.
        Chat templates ensure every message is clearly attributed to a role.
    """

    def __init__(self, tokenizer: ByteLevelBPETokenizer):
        """
        Initialize the chat template.

        Args:
            tokenizer: The ByteLevelBPETokenizer instance to use for encoding.

        Returns:
            None.

        Explanation:
            Stores a reference to the tokenizer for encoding the formatted
            chat string into token IDs.
        """
        self.tokenizer = tokenizer

    def format(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> str:
        """
        Format a list of chat messages into a single string.

        Args:
            messages: List of message dicts, each with "role" and "content" keys.
                Example: [{"role": "user", "content": "Hello!"}]
            add_generation_prompt: If True, append <|assistant|> at the end
                to prompt the model to generate an assistant response.
            add_bos: If True, prepend <|bos|> token.
            add_eos: If True, append <|eos|> token (typically only for training).

        Returns:
            Formatted chat string ready for encoding.

        Explanation:
            Each message is formatted as:
                <|{role}|>
                {content}

            Valid roles: "system", "user", "assistant".
            Tool-related roles: "tool_call", "tool_result".

            The generation prompt (<|assistant|>) signals to the model that
            it should generate a response. Without it, the model might not
            know it's supposed to respond.

        Example:
            >>> tokenizer = ByteLevelBPETokenizer()
            >>> template = ChatTemplate(tokenizer)
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Hi!"},
            ... ]
            >>> formatted = template.format(messages)
            >>> print(formatted)
            <|bos|>
            <|system|>
            You are helpful.
            <|user|>
            Hi!
            <|assistant|>
        """
        parts = []

        # Add BOS token if requested
        if add_bos:
            parts.append("<|bos|>")

        # Format each message
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Map role name to special token
            role_token = f"<|{role}|>"
            if role_token not in SPECIAL_TOKENS:
                raise ValueError(
                    f"Unknown role: {role}. Valid roles: "
                    "system, user, assistant, tool_call, tool_result"
                )

            parts.append(role_token)
            parts.append(content)

        # Add generation prompt if requested
        if add_generation_prompt:
            parts.append("<|assistant|>")

        # Add EOS token if requested
        if add_eos:
            parts.append("<|eos|>")

        # Join with newlines for readability
        return "\n".join(parts)

    def encode_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Format and encode a chat into token IDs.

        Args:
            messages: List of message dicts with "role" and "content".
            add_generation_prompt: If True, append <|assistant|> prompt.
            add_bos: If True, prepend <|bos|> token.
            add_eos: If True, append <|eos|> token.

        Returns:
            List of token IDs representing the formatted chat.

        Explanation:
            Combines formatting and encoding in one step. Uses
            allowed_special="all" to ensure role tokens are recognized
            as special tokens rather than regular text.

        Example:
            >>> tokenizer = ByteLevelBPETokenizer()
            >>> template = ChatTemplate(tokenizer)
            >>> ids = template.encode_chat([
            ...     {"role": "user", "content": "Hello!"}
            ... ])
        """
        formatted = self.format(
            messages,
            add_generation_prompt=add_generation_prompt,
            add_bos=add_bos,
            add_eos=add_eos,
        )
        return self.tokenizer.encode(formatted, allowed_special="all")


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################

def demonstrate_tokenizer():
    """
    Demonstrate the Byte-Level BPE Tokenizer's capabilities.

    Args:
        None.

    Returns:
        None.

    Explanation:
        Runs through key tokenizer features:
        1. Training on a sample corpus
        2. Encoding and decoding text
        3. Digit splitting behavior
        4. Byte-level fallback
        5. Chat template formatting
        6. Save and load
    """
    print("=" * 70)
    print("BYTE-LEVEL BPE TOKENIZER DEMONSTRATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Demo 1: Training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEMO 1: Training on sample corpus")
    print("=" * 70)

    # Sample corpus covering various patterns
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world! This is a test of the BPE tokenizer.",
        "Python is a great programming language for AI.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "1234567890 9876543210 111222333444555",
        "The year 2024 saw massive advances in AI technology.",
        "Hello! How are you? I'm doing great, thanks!",
        "Don't worry, we'll figure it out. We've done it before.",
        "Temperature: 98.6°F, Humidity: 45%, Pressure: 1013.25 hPa",
        "Machine learning models require large datasets for training.",
        "The transformer architecture revolutionized NLP in 2017.",
        "Byte-pair encoding was introduced by Sennrich et al. in 2015.",
    ]

    # Use a small vocab size for demonstration
    tokenizer = ByteLevelBPETokenizer(vocab_size=300)
    tokenizer.train(corpus, vocab_size=300)

    # -------------------------------------------------------------------------
    # Demo 2: Encoding and Decoding
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEMO 2: Encoding and Decoding")
    print("=" * 70)

    test_texts = [
        "Hello, world!",
        "The quick brown fox",
        "123 + 456 = 579",
        "def main(): print('hi')",
    ]

    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"\n  Original:  {text!r}")
        print(f"  Token IDs: {ids}")
        print(f"  Decoded:   {decoded!r}")
        print(f"  Roundtrip: {'OK' if decoded == text else 'MISMATCH'}")

    # -------------------------------------------------------------------------
    # Demo 3: Digit Splitting
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEMO 3: Digit Splitting Behavior")
    print("=" * 70)

    number_texts = ["123", "4567", "2024", "3.14159"]
    for text in number_texts:
        ids = tokenizer.encode(text)
        tokens = [tokenizer.inverse_vocab.get(i, f"<special:{i}>") for i in ids]
        print(f"\n  Input:  {text!r}")
        print(f"  Tokens: {ids}")
        # Show that digits are split (token count >= number of digits for pure numbers)
        digit_count = sum(1 for c in text if c.isdigit())
        print(f"  Digits in input: {digit_count}, Token count: {len(ids)}")

    # -------------------------------------------------------------------------
    # Demo 4: Byte-level Fallback
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEMO 4: Byte-Level Fallback (No UNK Tokens)")
    print("=" * 70)

    # Test with various character types
    # NOTE: We use repr() for display to avoid console encoding issues on Windows
    fallback_texts = [
        ("English text", "English"),
        ("Chinese text (zhongwen)", "Chinese"),       # Description for display
        ("Japanese text (nihongo)", "Japanese"),       # Description for display
        ("Hello emoji!", "Emoji"),                     # Description for display
        ("\x00\x01\x02\x03", "Control chars"),
        ("cafe resume naive", "Accented (ASCII)"),
    ]

    # Also test with actual Unicode but display safely
    unicode_test_cases = [
        "中文文本",         # Chinese
        "日本語テキスト",   # Japanese
        "café résumé naïve", # Accented characters
    ]

    print("\n  ASCII-compatible tests:")
    for text, description in fallback_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"\n  [{description}] Input:  {text!r}")
        print(f"  [{description}] IDs:    {ids}")
        print(f"  [{description}] Roundtrip: {'OK' if decoded == text else 'MISMATCH'}")

    print("\n  Unicode tests (roundtrip verified internally):")
    for text in unicode_test_cases:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        ok = decoded == text
        # Use ascii() to avoid console encoding errors on Windows
        print(f"\n  Input (ascii):  {ascii(text)}")
        print(f"  IDs:            {ids}")
        print(f"  Roundtrip:      {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print(f"  Decoded (ascii): {ascii(decoded)}")

    # -------------------------------------------------------------------------
    # Demo 5: Chat Template
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEMO 5: Chat Template")
    print("=" * 70)

    template = ChatTemplate(tokenizer)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is BPE tokenization?"},
        {"role": "assistant", "content": "BPE stands for Byte-Pair Encoding. It's a subword tokenization algorithm that iteratively merges the most frequent adjacent byte pairs."},
        {"role": "user", "content": "How does it handle unknown words?"},
    ]

    formatted = template.format(messages)
    print("\nFormatted chat:")
    print(formatted)

    chat_ids = template.encode_chat(messages)
    print(f"\nEncoded as {len(chat_ids)} token IDs:")
    print(chat_ids[:20], "..." if len(chat_ids) > 20 else "")

    # -------------------------------------------------------------------------
    # Demo 6: Save and Load
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEMO 6: Save and Load")
    print("=" * 70)

    save_path = "./demo_tokenizer"
    tokenizer.save(save_path)

    # Load into a new instance
    new_tokenizer = ByteLevelBPETokenizer()
    new_tokenizer.load(save_path)

    # Verify consistency
    test_text = "Hello, world! 123"
    original_ids = tokenizer.encode(test_text)
    loaded_ids = new_tokenizer.encode(test_text)
    print(f"\n  Original IDs:  {original_ids}")
    print(f"  Loaded IDs:    {loaded_ids}")
    print(f"  Consistent: {original_ids == loaded_ids}")

    # Clean up
    import shutil
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print(f"\n  Cleaned up {save_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Learned merges:  {len(tokenizer.merges)}")
    print(f"  Special tokens:  {len(SPECIAL_TOKENS)}")
    print(f"  Byte tokens:     256 (guaranteed coverage)")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_tokenizer()
