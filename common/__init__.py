"""
################################################################################
COMMON UTILITIES — SHARED ACROSS ALL MODULES
################################################################################

This module provides utilities used across the entire SOTA AI codebase.

Modules:
    1. tokenizer.py — Tokenization utilities
    2. data.py — Data loading and processing
    3. checkpoint.py — Model checkpointing
    4. logging.py — Training logging
    5. metrics.py — Evaluation metrics

################################################################################
"""

from .tokenizer import SimpleTokenizer, BPETokenizer
from .checkpoint import save_checkpoint, load_checkpoint
from .metrics import compute_perplexity, compute_accuracy
