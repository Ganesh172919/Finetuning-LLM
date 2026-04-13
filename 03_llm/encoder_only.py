"""
################################################################################
ENCODER-ONLY MODELS — BERT-STYLE ARCHITECTURES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Encoder-Only Models?
    Models that only use the encoder part of the transformer.
    They process input bidirectionally (seeing all tokens at once).

    Key difference from decoder-only (GPT):
    - Encoder: bidirectional (sees all context)
    - Decoder: causal (only sees past tokens)

Why do they matter?
    Encoder models excel at:
    - Understanding tasks (classification, NER)
    - Sentence embeddings
    - Semantic search
    - Question answering

Historical Evolution:
    - 2018: BERT (Google)
    - 2019: RoBERTa, ALBERT, DistilBERT
    - 2020: DeBERTa
    - 2022: ModernBERT

Interview Questions:
    1. "What's the difference between BERT and GPT?"
       BERT: encoder-only, bidirectional, understanding tasks
       GPT: decoder-only, causal, generation tasks

    2. "When should I use encoder vs decoder models?"
       Encoder: classification, embeddings, understanding
       Decoder: generation, chat, reasoning

    3. "Why is BERT bidirectional?"
       For understanding tasks, you need to see the full context.
       "bank" in "river bank" vs "bank account" needs both sides.

################################################################################
"""

import numpy as np
from typing import Optional, List
import math

import sys
sys.path.append('..')
from ..02_transformers.attention import MultiHeadAttention
from ..02_transformers.layers import TransformerBlock, RMSNorm, FeedForward

################################################################################
# SECTION 1: BERT MODEL
################################################################################

class BERT:
    """
    BERT: Bidirectional Encoder Representations from Transformers
    ==============================================================

    Definition: An encoder-only transformer that processes text bidirectionally.

    Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Input: [CLS] token1 token2 ... [SEP]            │
    │        ↓                                          │
    │ Token Embedding + Position Embedding             │
    │        ↓                                          │
    │ Transformer Encoder × N (bidirectional)          │
    │        ↓                                          │
    │ Output: contextual embeddings for each token     │
    └─────────────────────────────────────────────────┘

    Pre-training tasks:
    1. Masked Language Modeling (MLM): predict masked tokens
    2. Next Sentence Prediction (NSP): predict if sentences follow

    Interview Question:
        "How is BERT pre-trained?"
        Two tasks:
        1. MLM: randomly mask 15% of tokens, predict them
        2. NSP: predict if sentence B follows sentence A
        This learns bidirectional context.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_seq_len: int = 512
    ):
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.position_embedding = np.random.randn(max_seq_len, d_model) * 0.02
        self.segment_embedding = np.random.randn(2, d_model) * 0.02

        # Encoder layers
        self.layers = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        token_ids: np.ndarray,
        segment_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Encode input tokens.

        Args:
            token_ids: [batch × seq_len]
            segment_ids: [batch × seq_len] (0 for sentence A, 1 for B)

        Returns:
            embeddings: [batch × seq_len × d_model]
        """
        batch, seq_len = token_ids.shape

        # Embeddings
        x = self.token_embedding[token_ids]
        x = x + self.position_embedding[:seq_len]

        if segment_ids is not None:
            x = x + self.segment_embedding[segment_ids]

        # Encoder (bidirectional - no causal mask)
        for layer in self.layers:
            x = layer.forward(x, mask=None)  # No mask = bidirectional

        x = self.norm.forward(x)
        return x


################################################################################
# SECTION 2: MASKED LANGUAGE MODELING
################################################################################

class MaskedLM:
    """
    Masked Language Modeling (MLM)
    ==============================

    Definition: Predict randomly masked tokens.

    Algorithm:
    1. Randomly mask 15% of tokens
    2. Of masked tokens:
       - 80% replace with [MASK]
       - 10% replace with random token
       - 10% keep original
    3. Predict original tokens

    Interview Question:
        "Why does BERT use MLM instead of next token prediction?"
        MLM allows bidirectional context (seeing both sides).
        Next token prediction only sees left context.
        For understanding tasks, bidirectional is better.
    """

    def __init__(self, d_model: int, vocab_size: int):
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Prediction head
        self.head = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, encoder_output: np.ndarray) -> np.ndarray:
        """
        Predict masked tokens.

        Args:
            encoder_output: [batch × seq × d_model]

        Returns:
            logits: [batch × seq × vocab_size]
        """
        return np.matmul(encoder_output, self.head)


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_encoder_only():
    """Demonstrate encoder-only models."""
    print("=" * 70)
    print("ENCODER-ONLY MODEL DEMONSTRATION")
    print("=" * 70)

    # BERT
    print("\n--- BERT ---")
    bert = BERT(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        max_seq_len=64
    )

    token_ids = np.random.randint(0, 1000, (2, 10))
    output = bert.forward(token_ids)
    print(f"Input: {token_ids.shape}")
    print(f"Output: {output.shape}")

    # MLM
    print("\n--- Masked LM ---")
    mlm = MaskedLM(d_model=128, vocab_size=1000)
    logits = mlm.forward(output)
    print(f"MLM logits: {logits.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_encoder_only()


################################################################################
# REFERENCES
################################################################################

# [1] Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.

################################################################################
