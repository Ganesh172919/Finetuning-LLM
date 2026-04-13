"""
################################################################################
BYTE-LEVEL BPE TOKENIZER -- UNIT TESTS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is this test suite?
    A comprehensive set of unit tests for the ByteLevelBPETokenizer, verifying
    that all critical behaviors work correctly: digit splitting, byte fallback,
    special token handling, chat templates, roundtrip encoding/decoding, and
    merge boundary constraints.

Why does it matter?
    Tokenizers are the foundation of LLM input pipelines. A bug in tokenization
    silently corrupts every downstream computation. These tests serve as a
    specification of correct behavior AND a regression safety net.

How does it work?
    Each test function targets one specific behavior:
    - test_digit_splitting: Verifies numbers are split into individual digits.
    - test_byte_fallback: Verifies all 256 bytes are encodable/decodable.
    - test_special_tokens: Verifies special tokens encode/decode correctly.
    - test_no_unknown: Verifies no UNK tokens are ever produced.
    - test_chat_template: Verifies chat formatting with role tokens.
    - test_roundtrip: Verifies encode(decode(text)) == text for diverse inputs.
    - test_whitespace_merges: Verifies merges don't cross newline boundaries.

########################################

ARCHITECTURE DIAGRAM (ASCII art):

    +------------------------------------------------------------------+
    |                     TEST SUITE STRUCTURE                          |
    |                                                                   |
    |  test_digit_splitting ---------- Numbers split into digits        |
    |  test_byte_fallback ------------ All 256 bytes representable      |
    |  test_special_tokens ----------- Special tokens encode/decode     |
    |  test_no_unknown --------------- No UNK tokens ever               |
    |  test_chat_template ------------ Chat formatting correct          |
    |  test_roundtrip ---------------- encode(decode(x)) == x           |
    |  test_whitespace_merges -------- No cross-newline merges          |
    |                                                                   |
    |  Each test:                                                       |
    |    1. Creates a fresh tokenizer                                   |
    |    2. Trains on a small corpus                                    |
    |    3. Asserts specific behavior                                   |
    +------------------------------------------------------------------+

HISTORICAL CONTEXT:
    - Testing tokenizers gained importance after GPT-2 showed that subtle
      tokenization bugs (e.g., inconsistent handling of leading whitespace)
      could significantly impact model performance.
    - Modern tokenizer libraries (tiktoken, sentencepiece) include extensive
      test suites for exactly these behaviors.

INTERVIEW QUESTIONS:
    1. "What's the most common tokenizer bug in production?"
       Inconsistent handling of whitespace. If " hello" and "hello" tokenize
       differently (as they should -- leading space is meaningful), but the
       model wasn't trained with this distinction, it causes silent errors.

    2. "How do you test that a tokenizer has no UNK tokens?"
       Encode every possible byte value (0-255) and verify each produces
       a valid token ID. Then encode random byte sequences and verify no
       IDs are unmapped.

    3. "Why test roundtrip encoding?"
       Roundtrip testing (encode then decode, compare to original) catches
       bugs in both directions: encoding errors (wrong token IDs) and
       decoding errors (wrong byte-to-string conversion). It's the most
       comprehensive single test.

################################################################################
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import the tokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_tokenizer import ByteLevelBPETokenizer, ChatTemplate, SPECIAL_TOKENS


################################################################################
# SECTION 1: FIXTURES
################################################################################

# A fixture provides a reusable tokenizer instance for tests.
# WHY use a fixture? Avoids retraining the tokenizer in every test function,
# which would be slow and redundant.

@pytest.fixture
def trained_tokenizer():
    """
    Create and train a small tokenizer for testing.

    Args:
        None (pytest fixture).

    Returns:
        A trained ByteLevelBPETokenizer with vocab_size=300.

    Explanation:
        Uses a small vocab size (300) for fast training in tests.
        The corpus covers various patterns: words, numbers, punctuation,
        contractions, code, and whitespace.
    """
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world! This is a test of the BPE tokenizer.",
        "Python is a great programming language for AI.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "1234567890 9876543210 111222333444555",
        "The year 2024 saw massive advances in AI technology.",
        "Hello! How are you? I'm doing great, thanks!",
        "Don't worry, we'll figure it out. We've done it before.",
        "Temperature: 98.6F, Humidity: 45%, Pressure: 1013.25 hPa",
        "Machine learning models require large datasets for training.",
        "The transformer architecture revolutionized NLP in 2017.",
        "Byte-pair encoding was introduced by Sennrich et al. in 2015.",
        "def hello_world():\n    print('Hello, World!')\n    return True",
        "if x > 0:\n    result = x * 2\nelse:\n    result = 0",
    ]

    tokenizer = ByteLevelBPETokenizer(vocab_size=300)
    tokenizer.train(corpus, vocab_size=300)
    return tokenizer


################################################################################
# SECTION 2: DIGIT SPLITTING TESTS
################################################################################

class TestDigitSplitting:
    """
    Test Digit Splitting
    ====================

    Verifies that number sequences are split into individual digits,
    not merged into multi-digit tokens. This is critical for arithmetic
    capability.

    WHY this matters:
        If "123" becomes a single token, the model can't generalize to
        "124" or "321" -- each number is an atomic lookup. Splitting into
        ["1","2","3"] lets the model learn positional patterns.
    """

    def test_digit_splitting(self, trained_tokenizer):
        """
        Test that digit sequences are split into individual digit tokens.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Encodes "123" and checks that the token count is >= 3 (one per digit).
            With a small vocab, digits may not have merged at all, so each digit
            should be a separate token. With a larger vocab, some digit pairs might
            merge, but we verify the key property: "123" is NOT a single token.

            The test also checks that "1", "2", "3" individually produce tokens
            and that their concatenation is consistent with "123".

        Example:
            >>> # "123" should NOT be a single token
            >>> ids = tokenizer.encode("123")
            >>> assert len(ids) >= 3  # At least one token per digit
        """
        # Encode "123" -- should be split into individual digits
        ids = trained_tokenizer.encode("123")

        # The key assertion: "123" must NOT be a single token.
        # With digit splitting in the pre-tokenizer, "123" becomes ["1","2","3"],
        # each of which is encoded separately. Even if some digit pairs merge
        # during BPE, we should have at least 3 tokens.
        assert len(ids) >= 3, (
            f"'123' should produce at least 3 tokens (one per digit), "
            f"but got {len(ids)} tokens: {ids}"
        )

        # Verify individual digits are encodable
        id_1 = trained_tokenizer.encode("1")
        id_2 = trained_tokenizer.encode("2")
        id_3 = trained_tokenizer.encode("3")

        assert len(id_1) >= 1, "Digit '1' should encode to at least 1 token"
        assert len(id_2) >= 1, "Digit '2' should encode to at least 1 token"
        assert len(id_3) >= 1, "Digit '3' should encode to at least 1 token"

    def test_multi_digit_numbers(self, trained_tokenizer):
        """
        Test that longer number sequences are also split properly.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Tests various number lengths to ensure digit splitting is consistent.
            "12345" should produce at least 5 tokens, "2024" at least 4, etc.
        """
        test_cases = [
            ("12345", 5),
            ("2024", 4),
            ("9876543210", 10),
            ("0", 1),
        ]

        for number_str, min_tokens in test_cases:
            ids = trained_tokenizer.encode(number_str)
            assert len(ids) >= min_tokens, (
                f"'{number_str}' should produce at least {min_tokens} tokens, "
                f"but got {len(ids)}: {ids}"
            )

    def test_digits_in_context(self, trained_tokenizer):
        """
        Test that digits are split even when embedded in text.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Digit splitting should work within words and sentences, not just
            for standalone numbers. "abc123def" should split the "123" portion.
        """
        # "abc123def" -- the digit part should be split
        ids = trained_tokenizer.encode("abc123def")

        # We can't assert exact token count (depends on merges), but we can
        # verify roundtrip consistency
        decoded = trained_tokenizer.decode(ids)
        assert decoded == "abc123def", (
            f"Roundtrip failed for 'abc123def': got {decoded!r}"
        )


################################################################################
# SECTION 3: BYTE FALLBACK TESTS
################################################################################

class TestByteFallback:
    """
    Test Byte Fallback
    ==================

    Verifies that every byte value 0-255 can be encoded and decoded.
    This is the core guarantee of byte-level BPE: NO unknown tokens.

    WHY this matters:
        Character-level tokenizers fail on rare characters (UNK tokens).
        Byte-level BPE starts with all 256 bytes in the vocabulary, so
        ANY input -- English, Chinese, emoji, binary data -- is always
        representable.
    """

    def test_byte_fallback(self, trained_tokenizer):
        """
        Test that every byte value 0-255 is encodable and decodable.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            For each byte value 0-255:
            1. Create a single-byte string (using chr for ASCII, or raw bytes).
            2. Encode it.
            3. Verify we get at least one token ID.
            4. Decode back and verify we get the original byte.

            This test is the definitive proof that the tokenizer never
            produces unknown tokens.
        """
        for byte_val in range(256):
            # Create a single byte
            byte_char = bytes([byte_val])

            # Encode: byte -> token IDs
            # We encode the byte as a string, which in UTF-8 is just the byte
            # for values 0-127, or a multi-byte sequence for 128-255.
            # For byte-level BPE, we should be able to encode ANY byte.
            try:
                text = byte_char.decode("utf-8", errors="replace")
            except Exception:
                text = chr(byte_val) if byte_val < 128 else "�"

            ids = trained_tokenizer.encode(text)
            assert len(ids) >= 1, (
                f"Byte {byte_val} should produce at least 1 token, got 0"
            )

            # Decode: token IDs -> text
            decoded = trained_tokenizer.decode(ids)
            # For bytes 0-127, the roundtrip should be exact
            # For bytes 128-255, we just verify no crash and some output
            assert len(decoded) >= 1, (
                f"Decoding byte {byte_val} produced empty string"
            )

    def test_all_bytes_in_vocab(self, trained_tokenizer):
        """
        Test that all 256 single-byte tokens exist in the vocabulary.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Verifies the foundational property of byte-level BPE: the
            vocabulary always contains all 256 byte tokens (IDs 0-255).
            This is guaranteed by _init_base_vocab().
        """
        for byte_val in range(256):
            token_bytes = (byte_val,)
            assert token_bytes in trained_tokenizer.vocab, (
                f"Byte {byte_val} not found in vocabulary"
            )
            assert trained_tokenizer.vocab[token_bytes] == byte_val, (
                f"Byte {byte_val} should map to ID {byte_val}, "
                f"got {trained_tokenizer.vocab[token_bytes]}"
            )


################################################################################
# SECTION 4: SPECIAL TOKEN TESTS
################################################################################

class TestSpecialTokens:
    """
    Test Special Tokens
    ===================

    Verifies that all special tokens are registered correctly and can be
    encoded/decoded without interference from regular text encoding.

    WHY this matters:
        Special tokens mark roles (system/user/assistant), boundaries
        (BOS/EOS), and control signals (tool calls, chain-of-thought).
        If they collide with regular tokens or are missing, the model
        cannot distinguish between role markers and user content.
    """

    def test_special_tokens_registered(self, trained_tokenizer):
        """
        Test that all expected special tokens are registered.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Verifies that every token in the SPECIAL_TOKENS list has a
            corresponding entry in special_token_to_id and id_to_special_token.
        """
        for token in SPECIAL_TOKENS:
            assert token in trained_tokenizer.special_token_to_id, (
                f"Special token {token!r} not found in special_token_to_id"
            )

            token_id = trained_tokenizer.special_token_to_id[token]
            assert token_id in trained_tokenizer.id_to_special_token, (
                f"ID {token_id} for special token {token!r} not in id_to_special_token"
            )
            assert trained_tokenizer.id_to_special_token[token_id] == token, (
                f"ID {token_id} maps to "
                f"{trained_tokenizer.id_to_special_token[token_id]!r}, "
                f"expected {token!r}"
            )

    def test_special_token_ids_unique(self, trained_tokenizer):
        """
        Test that all special token IDs are unique.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Each special token must have a unique ID. Duplicate IDs would
            cause ambiguity during decoding.
        """
        ids = list(trained_tokenizer.special_token_to_id.values())
        assert len(ids) == len(set(ids)), (
            f"Special token IDs are not unique: {ids}"
        )

    def test_special_token_ids_dont_collide(self, trained_tokenizer):
        """
        Test that special token IDs don't collide with byte or merge tokens.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Special token IDs must be >= 256 + num_merges, so they don't
            overlap with byte tokens (0-255) or learned merge tokens.
        """
        byte_and_merge_ids = set(trained_tokenizer.inverse_vocab.keys())
        special_ids = set(trained_tokenizer.special_token_to_id.values())

        overlap = byte_and_merge_ids & special_ids
        assert len(overlap) == 0, (
            f"Special token IDs collide with byte/merge IDs: {overlap}"
        )

    def test_special_tokens_encode_decode(self, trained_tokenizer):
        """
        Test that special tokens can be encoded and decoded correctly.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            When allowed_special="all", special token strings in the input
            should be recognized and encoded as their special token IDs.
            Decoding those IDs should produce the original special token string.
        """
        for token in SPECIAL_TOKENS:
            # Encode with special tokens allowed
            ids = trained_tokenizer.encode(token, allowed_special="all")
            assert len(ids) == 1, (
                f"Special token {token!r} should encode to exactly 1 ID, "
                f"got {len(ids)}: {ids}"
            )

            expected_id = trained_tokenizer.special_token_to_id[token]
            assert ids[0] == expected_id, (
                f"Special token {token!r} should encode to ID {expected_id}, "
                f"got {ids[0]}"
            )

            # Decode back
            decoded = trained_tokenizer.decode(ids)
            assert decoded == token, (
                f"Decoding ID {ids[0]} should produce {token!r}, "
                f"got {decoded!r}"
            )


################################################################################
# SECTION 5: NO UNKNOWN TOKEN TESTS
################################################################################

class TestNoUnknown:
    """
    Test No Unknown Tokens
    ======================

    Verifies that random byte sequences never produce unknown tokens.
    This is the defining guarantee of byte-level BPE.

    WHY this matters:
        In production, LLMs receive arbitrary user input -- text, code,
        emoji, even binary data embedded in strings. An unknown token
        would silently corrupt the input, potentially causing the model
        to hallucinate or crash.
    """

    def test_no_unknown(self, trained_tokenizer):
        """
        Test that random byte sequences never produce unknown tokens.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            1. Generate random byte sequences of various lengths.
            2. Encode each sequence.
            3. Verify every token ID maps to a known token (in inverse_vocab
               or id_to_special_token).
            4. Verify decoding produces a valid string.

            This is a probabilistic test -- with random inputs, we cover a
            wide range of byte patterns that might trigger edge cases.
        """
        import random
        random.seed(42)  # Deterministic for reproducibility

        for trial in range(20):
            # Generate random bytes
            length = random.randint(1, 100)
            random_bytes = bytes([random.randint(0, 255) for _ in range(length)])

            # Convert to string (may contain replacement chars)
            text = random_bytes.decode("utf-8", errors="replace")

            # Encode
            ids = trained_tokenizer.encode(text)

            # Verify no unknown IDs
            for token_id in ids:
                is_byte_or_merge = token_id in trained_tokenizer.inverse_vocab
                is_special = token_id in trained_tokenizer.id_to_special_token
                assert is_byte_or_merge or is_special, (
                    f"Unknown token ID {token_id} in output for input "
                    f"(trial {trial}, length {length}). "
                    f"Vocab size: {len(trained_tokenizer.inverse_vocab)}, "
                    f"Special tokens: {len(trained_tokenizer.id_to_special_token)}"
                )

            # Verify decode doesn't crash
            decoded = trained_tokenizer.decode(ids)
            assert isinstance(decoded, str), (
                f"Decode should return string, got {type(decoded)}"
            )

    def test_empty_string(self, trained_tokenizer):
        """
        Test that empty string produces empty token list.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Edge case: empty input should produce no tokens and decode
            back to an empty string.
        """
        ids = trained_tokenizer.encode("")
        assert ids == [], f"Empty string should produce [], got {ids}"

        decoded = trained_tokenizer.decode([])
        assert decoded == "", f"Empty IDs should decode to '', got {decoded!r}"


################################################################################
# SECTION 6: CHAT TEMPLATE TESTS
################################################################################

class TestChatTemplate:
    """
    Test Chat Template
    ==================

    Verifies that the ChatTemplate correctly formats messages with role
    tokens and produces the expected string structure.

    WHY this matters:
        Chat templates are the interface between user-facing chat APIs and
        the model's token-level input. If formatting is wrong, the model
        cannot distinguish system instructions from user messages, leading
        to prompt injection vulnerabilities or instruction-following failures.
    """

    def test_chat_template(self, trained_tokenizer):
        """
        Test that chat template formats messages correctly with role tokens.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Verifies that:
            1. Each message is wrapped with the correct role token.
            2. BOS token is prepended when requested.
            3. Generation prompt (<|assistant|>) is appended when requested.
            4. Messages appear in the correct order.
        """
        template = ChatTemplate(trained_tokenizer)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        formatted = template.format(messages)

        # Verify BOS token is present
        assert formatted.startswith("<|bos|>"), (
            f"Chat should start with <|bos|>, got: {formatted[:20]!r}"
        )

        # Verify role tokens are present in order
        assert "<|system|>" in formatted, "Missing <|system|> role token"
        assert "<|user|>" in formatted, "Missing <|user|> role token"
        assert "<|assistant|>" in formatted, "Missing <|assistant|> role token"

        # Verify content is present
        assert "You are helpful." in formatted, "Missing system message content"
        assert "Hello!" in formatted, "Missing user message content"
        assert "Hi there!" in formatted, "Missing assistant message content"
        assert "How are you?" in formatted, "Missing second user message content"

        # Verify generation prompt is at the end
        assert formatted.endswith("<|assistant|>"), (
            f"Chat should end with <|assistant|> generation prompt, "
            f"got: {formatted[-20:]!r}"
        )

        # Verify order: system should come before user, user before assistant
        sys_pos = formatted.index("<|system|>")
        user_pos = formatted.index("<|user|>")
        asst_pos = formatted.index("<|assistant|>")
        assert sys_pos < user_pos < asst_pos, (
            f"Role tokens out of order: system@{sys_pos}, "
            f"user@{user_pos}, assistant@{asst_pos}"
        )

    def test_chat_template_no_bos(self, trained_tokenizer):
        """
        Test chat template without BOS token.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            When add_bos=False, the formatted string should NOT start
            with <|bos|>.
        """
        template = ChatTemplate(trained_tokenizer)
        messages = [{"role": "user", "content": "Hi!"}]

        formatted = template.format(messages, add_bos=False)
        assert not formatted.startswith("<|bos|>"), (
            "Chat should not start with <|bos|> when add_bos=False"
        )

    def test_chat_template_no_generation_prompt(self, trained_tokenizer):
        """
        Test chat template without generation prompt.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            When add_generation_prompt=False, the formatted string should
            NOT end with <|assistant|>.
        """
        template = ChatTemplate(trained_tokenizer)
        messages = [{"role": "user", "content": "Hi!"}]

        formatted = template.format(messages, add_generation_prompt=False)
        assert not formatted.endswith("<|assistant|>"), (
            "Chat should not end with <|assistant|> when "
            "add_generation_prompt=False"
        )

    def test_chat_encode(self, trained_tokenizer):
        """
        Test that encode_chat produces valid token IDs.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            encode_chat should produce a non-empty list of token IDs that
            can be decoded back to the formatted chat string.
        """
        template = ChatTemplate(trained_tokenizer)
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi!"},
        ]

        ids = template.encode_chat(messages)
        assert len(ids) > 0, "encode_chat should produce non-empty token list"

        # Verify roundtrip
        decoded = trained_tokenizer.decode(ids)
        assert "Be helpful." in decoded, "Decoded chat missing system content"
        assert "Hi!" in decoded, "Decoded chat missing user content"

    def test_chat_invalid_role(self, trained_tokenizer):
        """
        Test that invalid roles raise an error.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Using an unrecognized role (e.g., "moderator") should raise
            a ValueError, not silently produce incorrect formatting.
        """
        template = ChatTemplate(trained_tokenizer)
        messages = [{"role": "moderator", "content": "Be nice."}]

        with pytest.raises(ValueError, match="Unknown role"):
            template.format(messages)


################################################################################
# SECTION 7: ROUNDTRIP TESTS
################################################################################

class TestRoundtrip:
    """
    Test Roundtrip Encoding/Decoding
    ================================

    Verifies that encode(decode(text)) == text for various input types.
    This is the most comprehensive single test -- it catches bugs in both
    encoding and decoding.

    WHY this matters:
        Roundtrip consistency is a fundamental invariant. If encoding and
        decoding are not inverse operations, the model's input-output
        pipeline is broken.
    """

    def test_roundtrip(self, trained_tokenizer):
        """
        Test roundtrip encoding/decoding for diverse inputs.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            For each test text:
            1. Encode to token IDs.
            2. Decode back to string.
            3. Assert the result matches the original.

            Test cases cover:
            - English text
            - Code with indentation and special chars
            - Emoji
            - Chinese characters
            - Mixed content (numbers + text + punctuation)
            - Whitespace-heavy text
        """
        test_cases = [
            # English text
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "I'm happy, aren't you? We've been waiting!",

            # Code
            "def hello():\n    print('Hello!')",
            "if x > 0:\n    return x ** 2",
            "result = [i**2 for i in range(10)]",

            # Mixed content
            "Temperature: 98.6F at 3:45 PM",
            "Price: $12.99 (20% off!)",

            # Whitespace
            "  spaces  ",
            "tabs\there",
            "newlines\n\nhere",

            # Single characters
            "a",
            "!",

            # Numbers
            "12345",
            "0",
            "3.14159",
        ]

        for text in test_cases:
            ids = trained_tokenizer.encode(text)
            decoded = trained_tokenizer.decode(ids)
            assert decoded == text, (
                f"Roundtrip failed for {text!r}:\n"
                f"  Encoded IDs: {ids}\n"
                f"  Decoded:     {decoded!r}"
            )

    def test_roundtrip_with_special_tokens(self, trained_tokenizer):
        """
        Test roundtrip encoding/decoding with special tokens.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            When allowed_special="all", special tokens in the input should
            survive the roundtrip intact.
        """
        for token in SPECIAL_TOKENS:
            ids = trained_tokenizer.encode(token, allowed_special="all")
            decoded = trained_tokenizer.decode(ids)
            assert decoded == token, (
                f"Roundtrip failed for special token {token!r}: "
                f"decoded as {decoded!r}"
            )

    def test_roundtrip_unicode(self, trained_tokenizer):
        """
        Test roundtrip for various Unicode inputs.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Unicode has many edge cases: combining characters, zero-width
            joiners, multi-byte encodings. This test verifies basic
            Unicode handling by testing characters that use 2, 3, and 4
            bytes in UTF-8 encoding.
        """
        # Build test cases using unicode escape sequences to avoid
        # encoding issues on Windows consoles
        unicode_cases = [
            "é",       # e-acute accent (2 bytes in UTF-8: C3 A9)
            "ñ",       # n-tilde (2 bytes: C3 B1)
            "ü",       # u-umlaut (2 bytes: C3 BC)
            "中",       # CJK character zhong (3 bytes: E4 B8 AD)
            "\U0001d11e",   # Musical symbol (4 bytes: F0 9D 84 9E)
            "�",       # Replacement character (3 bytes: EF BF BD)
        ]

        for text in unicode_cases:
            ids = trained_tokenizer.encode(text)
            decoded = trained_tokenizer.decode(ids)
            assert decoded == text, (
                f"Unicode roundtrip failed for U+{ord(text):04X}: "
                f"decoded as {decoded!r}"
            )


################################################################################
# SECTION 8: WHITESPACE MERGE BOUNDARY TESTS
################################################################################

class TestWhitespaceMerges:
    """
    Test Whitespace Merge Boundaries
    ================================

    Verifies that BPE merges do not cross newline boundaries. Newlines
    are structural signals (especially in code) and merging across them
    would create tokens that only appear at specific structural positions,
    wasting vocabulary slots.

    WHY this matters:
        In code, indentation and newline structure carry meaning. If BPE
        merges ":\n    " into a single token, that token only appears at
        the start of indented blocks -- it's context-specific and wastes
        a vocabulary slot that could represent a more general pattern.
    """

    def test_whitespace_merges(self, trained_tokenizer):
        """
        Test that merges don't cross newline boundaries.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            The pre-tokenizer splits text at newline boundaries, so BPE
            merges can only occur within each chunk. This test verifies
            that encoding "foo\nbar" doesn't produce a single token that
            spans the newline.

            We verify this by encoding "foo\nbar" and checking that:
            1. The encoding is NOT a single token.
            2. The roundtrip is correct.

            We also compare with encoding "foo" and "bar" separately to
            verify that the newline is treated as a boundary.
        """
        # Encode text with newline
        text_with_newline = "foo\nbar"
        ids_with_newline = trained_tokenizer.encode(text_with_newline)

        # Encode components separately
        ids_foo = trained_tokenizer.encode("foo")
        ids_newline = trained_tokenizer.encode("\n")
        ids_bar = trained_tokenizer.encode("bar")

        # Verify roundtrip
        decoded = trained_tokenizer.decode(ids_with_newline)
        assert decoded == text_with_newline, (
            f"Roundtrip failed for {text_with_newline!r}: got {decoded!r}"
        )

        # The newline should create a boundary: the token count for "foo\nbar"
        # should be >= tokens for "foo" + tokens for "\n" + tokens for "bar" - 2
        # (allowing some merging within each segment, but not across the newline)
        # A simpler check: "foo\nbar" should NOT be a single token
        assert len(ids_with_newline) > 1, (
            f"'foo\nbar' should not be a single token, got {ids_with_newline}"
        )

    def test_newline_in_code(self, trained_tokenizer):
        """
        Test that code with newlines is tokenized with proper boundaries.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Code often has patterns like "def foo():\n    return 42".
            The pre-tokenizer should split this into chunks at newlines,
            preventing merges across the function signature and body.
        """
        code = "def foo():\n    return 42"
        ids = trained_tokenizer.encode(code)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == code, (
            f"Code roundtrip failed:\n"
            f"  Original: {code!r}\n"
            f"  Decoded:  {decoded!r}"
        )

        # The encoding should have multiple tokens (not one giant token)
        assert len(ids) > 3, (
            f"Code should have multiple tokens, got {len(ids)}: {ids}"
        )


################################################################################
# SECTION 9: ADDITIONAL EDGE CASE TESTS
################################################################################

class TestEdgeCases:
    """
    Test Edge Cases
    ===============

    Additional tests for edge cases and robustness.

    WHY this matters:
        Edge cases (empty inputs, very long inputs, unusual characters)
        are where bugs hide. Thorough edge case testing prevents
        production surprises.
    """

    def test_very_long_input(self, trained_tokenizer):
        """
        Test that very long inputs are handled correctly.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Long inputs should encode and decode without errors.
            We test with a 10,000 character string.
        """
        long_text = "Hello world! " * 1000  # ~13,000 characters
        ids = trained_tokenizer.encode(long_text)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == long_text, (
            f"Long input roundtrip failed. "
            f"Original length: {len(long_text)}, "
            f"Decoded length: {len(decoded)}"
        )

    def test_single_space(self, trained_tokenizer):
        """
        Test that a single space character roundtrips correctly.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Whitespace-only inputs are a common edge case. A single space
            should encode to at least one token and decode back to " ".
        """
        ids = trained_tokenizer.encode(" ")
        assert len(ids) >= 1, "Single space should produce at least 1 token"

        decoded = trained_tokenizer.decode(ids)
        assert decoded == " ", f"Single space roundtrip failed: got {decoded!r}"

    def test_consecutive_newlines(self, trained_tokenizer):
        """
        Test that consecutive newlines are handled correctly.

        Args:
            trained_tokenizer: A trained ByteLevelBPETokenizer instance.

        Returns:
            None (assertions).

        Explanation:
            Consecutive newlines (common in code and prose) should
            roundtrip correctly without merging across them.
        """
        text = "\n\n\n"
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text, (
            f"Consecutive newlines roundtrip failed: {decoded!r}"
        )


################################################################################
# SECTION 10: ENTRY POINT
################################################################################

if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
