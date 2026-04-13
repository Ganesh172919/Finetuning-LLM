"""
################################################################################
TRANSFORMER INFERENCE — TEXT GENERATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Transformer Inference?
    Using a trained transformer to generate text. This involves:
    1. Tokenize input
    2. Forward pass through model
    3. Sample next token
    4. Repeat until done

Why does it matter?
    Inference is how users interact with models. Key concerns:
    - Latency: How fast is the response?
    - Throughput: How many requests per second?
    - Cost: How much GPU time?

Interview Questions:
    1. "How does text generation work?"
       Autoregressive: predict one token at a time.
       Each new token is conditioned on all previous tokens.

    2. "How do you speed up inference?"
       KV cache, batching, quantization, speculative decoding.

    3. "What's the difference between latency and throughput?"
       Latency: time for one request
       Throughput: requests per second

################################################################################
"""

import numpy as np
from typing import Optional, List
import math

################################################################################
# SECTION 1: SAMPLING STRATEGIES
################################################################################

def greedy_decode(logits: np.ndarray) -> int:
    """Greedy: always pick most likely token."""
    return int(np.argmax(logits))


def temperature_sample(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Temperature sampling."""
    scaled = logits / temperature
    probs = np.exp(scaled - np.max(scaled))
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


def top_k_sample(logits: np.ndarray, k: int = 50, temperature: float = 1.0) -> int:
    """Top-K sampling."""
    scaled = logits / temperature
    top_indices = np.argsort(scaled)[-k:]
    filtered = np.full_like(scaled, -np.inf)
    filtered[top_indices] = scaled[top_indices]
    probs = np.exp(filtered - np.max(filtered))
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


def top_p_sample(logits: np.ndarray, p: float = 0.9, temperature: float = 1.0) -> int:
    """Top-P (nucleus) sampling."""
    scaled = logits / temperature
    sorted_indices = np.argsort(scaled)[::-1]
    sorted_probs = np.exp(scaled[sorted_indices] - np.max(scaled))
    sorted_probs = sorted_probs / np.sum(sorted_probs)

    cumulative = np.cumsum(sorted_probs)
    nucleus_size = np.searchsorted(cumulative, p) + 1
    nucleus_indices = sorted_indices[:nucleus_size]
    nucleus_probs = sorted_probs[:nucleus_size]
    nucleus_probs = nucleus_probs / np.sum(nucleus_probs)

    chosen = np.random.choice(nucleus_size, p=nucleus_probs)
    return int(nucleus_indices[chosen])


################################################################################
# SECTION 2: KV CACHE
################################################################################

class SimpleKVCache:
    """
    Simple KV Cache for efficient inference.

    Stores key and value tensors from previous tokens
    to avoid recomputation.
    """

    def __init__(self):
        self.cache = {}

    def update(self, layer_idx: int, key: np.ndarray, value: np.ndarray):
        """Update cache for a layer."""
        if layer_idx not in self.cache:
            self.cache[layer_idx] = {'key': key, 'value': value}
        else:
            self.cache[layer_idx]['key'] = np.concatenate(
                [self.cache[layer_idx]['key'], key], axis=-2
            )
            self.cache[layer_idx]['value'] = np.concatenate(
                [self.cache[layer_idx]['value'], value], axis=-2
            )

    def get(self, layer_idx: int):
        """Get cached key and value."""
        if layer_idx in self.cache:
            return self.cache[layer_idx]['key'], self.cache[layer_idx]['value']
        return None, None

    def clear(self):
        """Clear the cache."""
        self.cache = {}


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_inference():
    """Demonstrate inference concepts."""
    print("=" * 70)
    print("INFERENCE DEMONSTRATION")
    print("=" * 70)

    # Sampling strategies
    print("\n--- Sampling Strategies ---")
    logits = np.array([2.0, 1.5, 1.0, 0.5, 0.1, -1.0])

    print(f"Greedy: {greedy_decode(logits)}")
    print(f"Temperature (0.5): {temperature_sample(logits, 0.5)}")
    print(f"Temperature (2.0): {temperature_sample(logits, 2.0)}")
    print(f"Top-K (3): {top_k_sample(logits, k=3)}")
    print(f"Top-P (0.8): {top_p_sample(logits, p=0.8)}")

    # KV Cache
    print("\n--- KV Cache ---")
    cache = SimpleKVCache()
    cache.update(0, np.random.randn(1, 4, 64), np.random.randn(1, 4, 64))
    cache.update(0, np.random.randn(1, 1, 64), np.random.randn(1, 1, 64))
    k, v = cache.get(0)
    print(f"Cached key shape: {k.shape}")
    print(f"Cached value shape: {v.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_inference()
