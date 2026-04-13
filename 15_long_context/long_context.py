"""
################################################################################
LONG CONTEXT MODELS — EXTENDING SEQUENCE LENGTH
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Long Context Models?
    Models that can process very long sequences (100K+ tokens).

Key Techniques:
    1. RoPE Scaling: Extend position embeddings
    2. Ring Attention: Distribute across GPUs
    3. Sparse Attention: Attend to subset of tokens
    4. Sliding Window: Local attention

Historical Progression:
    - 2017: 512 tokens (original Transformer)
    - 2020: 2048 tokens (GPT-3)
    - 2023: 8K-32K tokens (GPT-4, Claude)
    - 2024: 128K-1M tokens (Claude 3, Gemini)
    - 2025: 10M+ tokens (research)

Interview Questions:
        Q: "How do you extend context length?"
        A: RoPE scaling (NTK-aware), ring attention, sparse attention.

        Q: "What's the challenge with long context?"
        A: O(n²) attention complexity, KV cache memory,
           attention sink phenomenon.

################################################################################
"""

import numpy as np
import math

################################################################################
# SECTION 1: RoPE SCALING
################################################################################

class RoPEScaling:
    """
    RoPE Scaling for Long Context
    ===============================

    Extends RoPE to longer sequences.

    Methods:
    1. Linear scaling: divide position by scale factor
    2. NTK-aware: adjust frequency base
    3. YaRN: yet another RoPE extension

    Interview Questions:
        Q: "How does NTK-aware RoPE scaling work?"
        A: Instead of scaling positions, adjust the frequency base.
           This preserves local position information while extending range.
    """

    def __init__(self, d_model: int, base: float = 10000.0, scale: float = 1.0):
        self.d_model = d_model
        self.base = base
        self.scale = scale

        # Adjusted frequencies
        inv_freq = 1.0 / (base ** (np.arange(0, d_model, 2) / d_model))
        self.inv_freq = inv_freq / scale

    def get_angles(self, position: int) -> np.ndarray:
        """Get rotation angles for position."""
        return position * self.inv_freq


################################################################################
# SECTION 2: SLIDING WINDOW WITH ATTENTION SINK
################################################################################

class SlidingWindowWithSink:
    """
    Sliding Window with Attention Sink
    ====================================

    Combines sliding window with attention to initial tokens.

    Problem: Models rely heavily on initial tokens ("attention sink")
    Solution: Always attend to initial tokens + sliding window

    Pattern:
    [Initial tokens] + [Sliding window of recent tokens]

    Interview Questions:
        Q: "What is the attention sink phenomenon?"
        A: Models assign high attention to initial tokens regardless
           of content. These tokens act as a "sink" for attention.
    """

    def __init__(self, window_size: int = 4096, n_sink_tokens: int = 4):
        self.window_size = window_size
        self.n_sink_tokens = n_sink_tokens

    def get_attention_mask(self, seq_len: int) -> np.ndarray:
        """
        Get attention mask.

        Args:
            seq_len: Sequence length

        Returns:
            mask: [seq_len × seq_len] boolean mask
        """
        mask = np.zeros((seq_len, seq_len), dtype=bool)

        for i in range(seq_len):
            # Always attend to initial tokens
            mask[i, :self.n_sink_tokens] = True

            # Attend to sliding window
            start = max(self.n_sink_tokens, i - self.window_size)
            mask[i, start:i + 1] = True

        return mask


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_long_context():
    """Demonstrate long context techniques."""
    print("=" * 70)
    print("LONG CONTEXT DEMONSTRATION")
    print("=" * 70)

    # RoPE scaling
    print("\n--- RoPE Scaling ---")
    rope = RoPEScaling(d_model=64, scale=2.0)
    angles = rope.get_angles(1000)
    print(f"Scale: {rope.scale}")
    print(f"Angles shape: {angles.shape}")

    # Sliding window with sink
    print("\n--- Sliding Window with Sink ---")
    swa = SlidingWindowWithSink(window_size=16, n_sink_tokens=4)
    mask = swa.get_attention_mask(32)
    print(f"Mask shape: {mask.shape}")
    print(f"Active attention pairs: {np.sum(mask)}")
    print(f"Sparsity: {1 - np.sum(mask) / (32*32):.2%}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_long_context()


################################################################################
# REFERENCES
################################################################################

# [1] Chen, S., et al. (2023). Extending Context Window of Large Language Models via RoPE.
# [2] Xiao, G., et al. (2023). Efficient Streaming Language Models with Attention Sinks.

################################################################################
