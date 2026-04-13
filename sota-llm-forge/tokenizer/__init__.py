"""
################################################################################
SOTA LLM FORGE — TOKENIZER
################################################################################

Byte-level BPE tokenizer for the SOTA LLM training stack.

Implements tiktoken-style byte-level BPE with:
    - Digit splitting (individual digits, not merged numbers)
    - Byte-level fallback (every byte representable, no UNK tokens)
    - Regex pre-tokenization (GPT-4 style patterns)
    - Special tokens for chat templates and reasoning blocks

See also:
    train_tokenizer.py — Training and encode/decode
    test_tokenizer.py  — Unit tests

################################################################################
"""

from .train_tokenizer import ByteLevelBPETokenizer, ChatTemplate

__all__ = ["ByteLevelBPETokenizer", "ChatTemplate"]
