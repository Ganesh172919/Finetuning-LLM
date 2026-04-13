"""
################################################################################
INFERENCE ENGINE — PRODUCTION TEXT GENERATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is an Inference Engine?
    A system that takes prompts and generates text responses.
    It handles:
    - Request queuing
    - Batching for efficiency
    - KV cache management
    - Sampling strategies
    - Response streaming

Why does it matter?
    Inference is how users interact with models.
    Key metrics:
    - Time to First Token (TTFT): latency for first response
    - Tokens per Second: generation speed
    - Throughput: requests per second
    - Cost per Token: GPU cost efficiency

Interview Questions:
    1. "How do you optimize LLM inference?"
       KV cache, batching, quantization, speculative decoding,
       continuous batching, paged attention.

    2. "What's the difference between latency and throughput?"
       Latency: time for one request (user experience)
       Throughput: requests per second (system capacity)
       Often a tradeoff between them.

    3. "What is speculative decoding?"
       Use a small, fast model to draft tokens.
       Verify with large model in parallel.
       If accepted, get multiple tokens per large model call.

################################################################################
"""

import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass
from collections import deque
import time

################################################################################
# SECTION 1: SAMPLING PARAMETERS
################################################################################

@dataclass
class SamplingParams:
    """
    Parameters for text generation sampling.

    Attributes:
        temperature: Controls randomness (0=greedy, 1=normal, >1=more random)
        top_k: Only sample from top K tokens
        top_p: Nucleus sampling threshold
        max_tokens: Maximum tokens to generate
        stop: Stop sequences
        repetition_penalty: Penalize repeated tokens
    """
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    max_tokens: int = 100
    stop: Optional[List[str]] = None
    repetition_penalty: float = 1.0


################################################################################
# SECTION 2: INFERENCE ENGINE
################################################################################

class InferenceEngine:
    """
    Inference Engine for Language Models
    =====================================

    Definition: Manages model loading, request processing, and response generation.

    Architecture:
    ┌─────────────────────────────────────────────┐
    │ InferenceEngine                               │
    │   ├── Model (loaded in memory)               │
    │   ├── KV Cache                               │
    │   ├── Request Queue                          │
    │   └── Batch Scheduler                        │
    └─────────────────────────────────────────────┘

    Key Features:
    1. Request queuing and batching
    2. KV cache management
    3. Streaming responses
    4. Multiple sampling strategies

    Interview Questions:
        1. "How does continuous batching work?"
           Instead of waiting for all requests to finish,
           new requests join the batch as others complete.
           This maximizes GPU utilization.

        2. "What is PagedAttention?"
           Allocates KV cache memory in pages (like OS virtual memory).
           Reduces memory fragmentation and waste.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.request_queue = deque()
        self.active_requests = []

    def generate(
        self,
        prompt: str,
        params: SamplingParams = SamplingParams()
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text
            params: Sampling parameters

        Returns:
            Generated text
        """
        # Tokenize
        token_ids = self.tokenizer.encode(prompt)

        # Generate
        generated_ids = self._generate_tokens(token_ids, params)

        # Decode
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text

    def _generate_tokens(
        self,
        prompt_ids: List[int],
        params: SamplingParams
    ) -> List[int]:
        """
        Generate token IDs autoregressively.

        This is a simplified version. Real implementations use:
        - KV cache for efficiency
        - Batching for throughput
        - Streaming for latency
        """
        generated = list(prompt_ids)

        for _ in range(params.max_tokens):
            # Forward pass
            logits = self._forward(generated)

            # Apply sampling
            next_token = self._sample(logits[-1], params)
            generated.append(next_token)

            # Check stop conditions
            if self._should_stop(generated, params):
                break

        return generated[len(prompt_ids):]

    def _forward(self, token_ids: List[int]) -> np.ndarray:
        """Run forward pass (simplified)."""
        # In real implementation: use KV cache
        return np.random.randn(len(token_ids), 32000)  # Placeholder

    def _sample(self, logits: np.ndarray, params: SamplingParams) -> int:
        """
        Sample next token from logits.

        Applies temperature, top-k, and top-p filtering.
        """
        # Temperature
        if params.temperature != 1.0:
            logits = logits / params.temperature

        # Top-K
        if params.top_k > 0:
            top_k_indices = np.argsort(logits)[-params.top_k:]
            filtered = np.full_like(logits, -np.inf)
            filtered[top_k_indices] = logits[top_k_indices]
            logits = filtered

        # Top-P
        if params.top_p < 1.0:
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
            sorted_probs = sorted_probs / np.sum(sorted_probs)

            cumulative = np.cumsum(sorted_probs)
            nucleus_size = np.searchsorted(cumulative, params.top_p) + 1
            nucleus_indices = sorted_indices[:nucleus_size]

            filtered = np.full_like(logits, -np.inf)
            filtered[nucleus_indices] = logits[nucleus_indices]
            logits = filtered

        # Softmax and sample
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs)

    def _should_stop(self, token_ids: List[int], params: SamplingParams) -> bool:
        """Check if generation should stop."""
        # Check max tokens
        if len(token_ids) >= params.max_tokens:
            return True

        # Check stop sequences
        if params.stop:
            text = self.tokenizer.decode(token_ids)
            for stop_seq in params.stop:
                if stop_seq in text:
                    return True

        return False


################################################################################
# SECTION 3: CONTINUOUS BATCHING
################################################################################

class ContinuousBatcher:
    """
    Continuous Batching
    ====================

    Definition: Dynamically manage batches as requests complete.

    Traditional batching:
    - Wait for all requests in batch to finish
    - New requests wait until batch is done
    - Wastes GPU time on finished requests

    Continuous batching:
    - New requests join batch immediately
    - Finished requests leave batch
    - GPU always has work to do

    Visual:
    Traditional: [req1, req2, req3] → wait → [req4, req5, req6]
    Continuous:  [req1, req2, req3] → req1 done, req4 joins → [req2, req3, req4]

    Benefits:
    - 2-3x throughput improvement
    - Better GPU utilization
    - Lower latency for new requests

    Used by: vLLM, TensorRT-LLM, TGI

    Interview Question:
        "What is continuous batching?"
        A batching strategy where requests can join and leave
        the batch at any time. It maximizes GPU utilization
        by always keeping the GPU busy.
    """

    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        self.active_batch = []
        self.waiting_queue = deque()

    def add_request(self, request: dict):
        """Add a new request to the queue."""
        if len(self.active_batch) < self.max_batch_size:
            self.active_batch.append(request)
        else:
            self.waiting_queue.append(request)

    def remove_completed(self):
        """Remove completed requests and add new ones."""
        # Remove completed
        self.active_batch = [r for r in self.active_batch if not r.get('done', False)]

        # Add from queue
        while len(self.active_batch) < self.max_batch_size and self.waiting_queue:
            self.active_batch.append(self.waiting_queue.popleft())

    def get_batch(self) -> List[dict]:
        """Get current active batch."""
        return self.active_batch


################################################################################
# SECTION 4: TESTING & EXAMPLES
################################################################################

def demonstrate_inference():
    """Demonstrate inference concepts."""
    print("=" * 70)
    print("INFERENCE ENGINE DEMONSTRATION")
    print("=" * 70)

    # Sampling parameters
    print("\n--- Sampling Parameters ---")
    params = SamplingParams(
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        max_tokens=100
    )
    print(f"Temperature: {params.temperature}")
    print(f"Top-K: {params.top_k}")
    print(f"Top-P: {params.top_p}")

    # Continuous batcher
    print("\n--- Continuous Batching ---")
    batcher = ContinuousBatcher(max_batch_size=4)

    # Simulate requests
    for i in range(6):
        batcher.add_request({'id': i, 'prompt': f'Request {i}', 'done': False})
        print(f"Added request {i}, batch size: {len(batcher.active_batch)}")

    # Complete some requests
    batcher.active_batch[0]['done'] = True
    batcher.active_batch[1]['done'] = True
    batcher.remove_completed()
    print(f"After completing 2, batch size: {len(batcher.active_batch)}")
    print(f"Queue size: {len(batcher.waiting_queue)}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_inference()


################################################################################
# REFERENCES
################################################################################

# [1] Kwon, W., et al. (2023). Efficient Memory Management for LLM Serving with PagedAttention.
# [2] Yu, G., et al. (2022). Orca: A Distributed Serving System for Transformer-Based Generative Models.
# [3] Leviathan, Y., et al. (2023). Fast Inference from Transformers via Speculative Decoding.

################################################################################
