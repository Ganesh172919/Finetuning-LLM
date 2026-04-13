"""
################################################################################
SLIDING WINDOW ATTENTION — EFFICIENT LONG-CONTEXT ATTENTION PATTERNS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Sliding Window Attention?
    Each token attends only to W nearby tokens instead of all N tokens.
    This reduces O(N²) to O(N*W), enabling much longer sequences.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


################################################################################
# SECTION 1: SLIDING WINDOW ATTENTION
################################################################################

class SlidingWindowAttention:
    """
    Mistral-style Sliding Window Attention.

    Each token attends only to W previous tokens. With L layers,
    the effective receptive field is L*W tokens.

    Paper: "Mistral 7B" (Jiang et al., 2023)

    Interview Question:
        "How does sliding window attention work?"
        Each token attends to W=4096 previous tokens. With L=32 layers,
        effective context = 32*4096 = 131K tokens. This gives O(N*W)
        compute instead of O(N²), much more efficient for long sequences.
    """

    def __init__(self, window_size: int = 4096, causal: bool = True):
        self.window_size = window_size
        self.causal = causal

    def create_mask(self, seq_len: int) -> np.ndarray:
        """
        Create sliding window attention mask.

        Args:
            seq_len: Sequence length

        Returns:
            Attention mask (seq_len, seq_len)
        """
        mask = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1) if self.causal else max(0, i - self.window_size // 2)
            end = i + 1 if self.causal else min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1.0
        return mask

    def effective_receptive_field(self, n_layers: int) -> int:
        """Effective receptive field after L layers."""
        return n_layers * self.window_size


################################################################################
# SECTION 2: DILATED ATTENTION
################################################################################

class DilatedAttention:
    """
    Dilated Attention — Attend at regular intervals.

    Like dilated convolutions, attend to every D-th token.
    This gives O(N/D) attention per token with large receptive field.

    Interview Question:
        "What is dilated attention?"
        Attend to tokens at intervals of D. Token i attends to
        i, i-D, i-2D, i-3D, etc. This gives O(N/D) attention
        cost with receptive field of N. Good for capturing long-range
        dependencies efficiently.
    """

    def __init__(self, dilation: int = 4, causal: bool = True):
        self.dilation = dilation
        self.causal = causal

    def create_mask(self, seq_len: int) -> np.ndarray:
        """Create dilated attention mask."""
        mask = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            if self.causal:
                positions = range(0, i + 1, self.dilation)
            else:
                positions = range(max(0, i % self.dilation), seq_len, self.dilation)
            for j in positions:
                mask[i, j] = 1.0
        return mask


################################################################################
# SECTION 3: LONGFORMER ATTENTION
################################################################################

class LongformerAttention:
    """
    Longformer-style Hybrid Attention.

    Combines: local sliding window + global attention tokens + dilated.

    Paper: "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)

    Interview Question:
        "How does Longformer handle long documents?"
        Three attention types: (1) Local sliding window for most tokens,
        (2) Global attention for special tokens ([CLS], question tokens),
        (3) Dilated attention for very long range. This gives O(N) compute
        while maintaining global connectivity through special tokens.
    """

    def __init__(self, window_size: int = 512, global_tokens: List[int] = None):
        self.window_size = window_size
        self.global_tokens = global_tokens or [0]  # [CLS] is global

    def create_mask(self, seq_len: int) -> np.ndarray:
        """Create Longformer mask with local + global patterns."""
        mask = np.zeros((seq_len, seq_len))

        # Local sliding window
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1.0

        # Global tokens attend to all
        for g in self.global_tokens:
            if g < seq_len:
                mask[g, :] = 1.0
                mask[:, g] = 1.0

        return mask


################################################################################
# SECTION 4: HIERARCHICAL ATTENTION
################################################################################

class HierarchicalAttention:
    """
    Multi-Resolution Attention.

    Attend at multiple levels: token → sentence → document.

    Interview Question:
        "What is hierarchical attention?"
        Process at multiple granularities: (1) Token-level: local attention
        within sentences, (2) Sentence-level: attend between sentence
        representations, (3) Document-level: attend between documents.
        This captures both local and global dependencies efficiently.
    """

    def __init__(self, sentence_len: int = 64):
        self.sentence_len = sentence_len

    def pool_to_sentences(self, tokens: np.ndarray) -> np.ndarray:
        """Pool tokens into sentence representations."""
        seq_len = tokens.shape[0]
        n_sentences = math.ceil(seq_len / self.sentence_len)
        sentences = np.zeros((n_sentences, tokens.shape[1]))
        for i in range(n_sentences):
            start = i * self.sentence_len
            end = min(start + self.sentence_len, seq_len)
            sentences[i] = tokens[start:end].mean(axis=0)
        return sentences


################################################################################
# SECTION 5: DEMONSTRATION
################################################################################

def demonstrate_sliding_window():
    """Demonstrate sliding window attention patterns."""
    print("=" * 70)
    print("SLIDING WINDOW ATTENTION DEMONSTRATION")
    print("=" * 70)

    # Sliding Window
    print("\n1. SLIDING WINDOW ATTENTION")
    print("-" * 40)
    swa = SlidingWindowAttention(window_size=256)
    mask = swa.create_mask(1024)
    density = mask.sum() / mask.size
    print(f"  Window: 256, Seq: 1024")
    print(f"  Density: {density:.2%}")
    print(f"  Receptive field (32 layers): {swa.effective_receptive_field(32)}")

    # Dilated
    print("\n2. DILATED ATTENTION")
    print("-" * 40)
    da = DilatedAttention(dilation=4)
    mask = da.create_mask(64)
    density = mask.sum() / mask.size
    print(f"  Dilation: 4, Seq: 64")
    print(f"  Density: {density:.2%}")

    # Longformer
    print("\n3. LONGFORMER ATTENTION")
    print("-" * 40)
    lf = LongformerAttention(window_size=128, global_tokens=[0])
    mask = lf.create_mask(512)
    density = mask.sum() / mask.size
    print(f"  Window: 128, Global tokens: 1")
    print(f"  Density: {density:.2%}")

    # Hierarchical
    print("\n4. HIERARCHICAL ATTENTION")
    print("-" * 40)
    ha = HierarchicalAttention(sentence_len=32)
    tokens = np.random.randn(256, 128)
    sentences = ha.pool_to_sentences(tokens)
    print(f"  Tokens: {tokens.shape}")
    print(f"  Sentences: {sentences.shape}")
    print(f"  Compression: {tokens.shape[0] / sentences.shape[0]:.0f}x")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_sliding_window()
