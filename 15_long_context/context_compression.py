"""
################################################################################
CONTEXT COMPRESSION — KV CACHE COMPRESSION AND ATTENTION SINKS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Context Compression?
    Techniques to reduce the memory and compute cost of long contexts
    by compressing the KV cache or the input prompt itself.

Why does it matter?
    100K context with 70B model = 40GB KV cache. Compression reduces this
    to manageable sizes while preserving important information.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


################################################################################
# SECTION 1: KV CACHE COMPRESSION
################################################################################

class KVCacheCompressor:
    """
    Compress KV cache to reduce memory.

    Methods:
    - H2O (Heavy Hitter Oracle): Keep tokens with highest attention
    - StreamingLLM: Keep sink tokens + recent window
    - SnapKV: Cluster-based compression

    Interview Question:
        "How do you compress the KV cache?"
        (1) H2O: keep tokens that receive the most attention (heavy hitters),
        (2) StreamingLLM: keep first K tokens (attention sinks) + last W tokens,
        (3) SnapKV: cluster similar KV pairs and keep cluster centers.
    """

    def __init__(self, keep_ratio: float = 0.5, method: str = 'h2o'):
        self.keep_ratio = keep_ratio
        self.method = method

    def h2o_compress(self, kv_cache: np.ndarray,
                     attention_scores: np.ndarray) -> np.ndarray:
        """
        H2O: Keep tokens with highest cumulative attention.

        Args:
            kv_cache: (seq_len, d_model)
            attention_scores: (seq_len,) cumulative attention per token

        Returns:
            Compressed KV cache
        """
        n_keep = max(1, int(len(kv_cache) * self.keep_ratio))
        top_indices = np.argsort(attention_scores)[-n_keep:]
        top_indices = np.sort(top_indices)  # Keep sorted order
        return kv_cache[top_indices]


################################################################################
# SECTION 2: ATTENTION SINK
################################################################################

class AttentionSink:
    """
    Streaming LLM — Keep attention sink tokens + sliding window.

    Key Insight: First few tokens receive disproportionately high attention
    regardless of content. These are "attention sinks" — keep them always.

    Paper: "Efficient Streaming Language Models with Attention Sinks"
           (Xiao et al., ICLR 2024)

    Interview Question:
        "What are attention sinks in streaming LLMs?"
        The first few tokens get very high attention weights even though
        they're not semantically important. This is an artifact of
        softmax. StreamingLLM keeps these "sink" tokens plus a sliding
        window of recent tokens, enabling infinite-length generation.
    """

    def __init__(self, n_sink_tokens: int = 4, window_size: int = 1024):
        self.n_sink_tokens = n_sink_tokens
        self.window_size = window_size

    def compress(self, kv_cache: np.ndarray) -> np.ndarray:
        """
        Keep sink tokens + recent window.

        Args:
            kv_cache: (seq_len, d_model)

        Returns:
            Compressed KV cache
        """
        seq_len = len(kv_cache)
        if seq_len <= self.n_sink_tokens + self.window_size:
            return kv_cache

        # Keep first K (sinks) + last W (recent)
        sinks = kv_cache[:self.n_sink_tokens]
        recent = kv_cache[-self.window_size:]
        return np.concatenate([sinks, recent], axis=0)

    def get_attention_distribution(self, attention_weights: np.ndarray) -> Dict:
        """
        Analyze attention distribution to find sinks.

        Args:
            attention_weights: (n_heads, seq_len, seq_len)

        Returns:
            Dictionary with sink analysis
        """
        # Average attention to each position
        avg_attn = attention_weights.mean(axis=(0, 1))
        sink_attn = avg_attn[:self.n_sink_tokens].sum()
        total_attn = avg_attn.sum()

        return {
            'sink_attention_ratio': sink_attn / total_attn,
            'sink_tokens': self.n_sink_tokens,
            'top_attended_positions': np.argsort(avg_attn)[-10:].tolist()
        }


################################################################################
# SECTION 3: PROMPT COMPRESSION
################################################################################

class PromptCompression:
    """
    Compress the input prompt itself.

    Methods:
    - LLMLingua: Remove low-entropy (uninformative) tokens
    - Selective Context: Keep only informative sentences
    - Gist Tokens: Compress into learned summary tokens

    Interview Question:
        "How does prompt compression work?"
        (1) LLMLingua: compute per-token entropy, remove low-entropy tokens
        (keep only surprising/informative tokens), (2) Selective Context:
        score sentences by information density, keep top-K, (3) Gist Tokens:
        train a model to compress prompts into K summary tokens.
    """

    def __init__(self, compression_ratio: float = 0.5):
        self.compression_ratio = compression_ratio

    def entropy_based_compression(self, token_probs: np.ndarray,
                                   tokens: List[str]) -> List[str]:
        """
        LLMLingua-style: keep high-entropy tokens.

        Args:
            token_probs: Probability of each token
            tokens: Token strings

        Returns:
            Compressed token list
        """
        # Compute entropy
        entropy = -token_probs * np.log(token_probs + 1e-10)
        n_keep = max(1, int(len(tokens) * self.compression_ratio))
        top_indices = np.argsort(entropy)[-n_keep:]
        top_indices = np.sort(top_indices)
        return [tokens[i] for i in top_indices]

    def sentence_compression(self, sentences: List[str],
                              scores: List[float]) -> List[str]:
        """
        Keep most informative sentences.

        Args:
            sentences: List of sentences
            scores: Importance score per sentence

        Returns:
            Compressed sentences
        """
        n_keep = max(1, int(len(sentences) * self.compression_ratio))
        top_indices = np.argsort(scores)[-n_keep:]
        top_indices = np.sort(top_indices)
        return [sentences[i] for i in top_indices]


################################################################################
# SECTION 4: DEMONSTRATION
################################################################################

def demonstrate_context_compression():
    """Demonstrate context compression."""
    print("=" * 70)
    print("CONTEXT COMPRESSION DEMONSTRATION")
    print("=" * 70)

    # KV Cache Compression
    print("\n1. KV CACHE COMPRESSION")
    print("-" * 40)
    kv = np.random.randn(1024, 256)
    scores = np.random.rand(1024)
    compressor = KVCacheCompressor(keep_ratio=0.5)
    compressed = compressor.h2o_compress(kv, scores)
    print(f"  Original: {kv.shape}")
    print(f"  Compressed: {compressed.shape}")
    print(f"  Compression ratio: {kv.shape[0] / compressed.shape[0]:.1f}x")

    # Attention Sink
    print("\n2. ATTENTION SINK (STREAMING LLM)")
    print("-" * 40)
    sink = AttentionSink(n_sink_tokens=4, window_size=256)
    kv_cache = np.random.randn(2048, 256)
    compressed = sink.compress(kv_cache)
    print(f"  Original: {kv_cache.shape}")
    print(f"  Compressed: {compressed.shape}")
    print(f"  Kept: {4} sink + {256} window = {compressed.shape[0]}")

    # Prompt Compression
    print("\n3. PROMPT COMPRESSION")
    print("-" * 40)
    pc = PromptCompression(compression_ratio=0.5)
    tokens = ["The", "cat", "sat", "on", "the", "mat", "and", "looked", "out", "the", "window"]
    probs = np.random.dirichlet(np.ones(len(tokens)))
    compressed = pc.entropy_based_compression(probs, tokens)
    print(f"  Original: {tokens}")
    print(f"  Compressed: {compressed}")
    print(f"  Ratio: {len(compressed)}/{len(tokens)}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_context_compression()
