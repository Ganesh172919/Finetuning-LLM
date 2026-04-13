"""
################################################################################
MODEL SERVING — PRODUCTION INFERENCE SYSTEMS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Model Serving?
    Systems for serving model predictions to users in production.

Key Technologies:
    - vLLM: Efficient LLM serving with PagedAttention
    - TensorRT-LLM: NVIDIA optimized serving
    - Triton: General model serving
    - Ray Serve: Scalable serving

Key Metrics:
    - TTFT (Time to First Token): Latency for first response
    - TPS (Tokens per Second): Generation speed
    - Throughput: Requests per second
    - Cost: GPU cost per request

Interview Questions:
        Q: "How do you optimize LLM inference?"
        A: KV cache, batching, quantization, speculative decoding,
           continuous batching, paged attention.

        Q: "What is vLLM?"
        A: A serving system that uses PagedAttention for efficient
           KV cache management. 2-4x throughput improvement.

################################################################################
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque
import time

################################################################################
# SECTION 1: REQUEST QUEUE
################################################################################

class RequestQueue:
    """
    Request Queue
    =============

    Queues incoming requests for batch processing.
    """

    def __init__(self, max_size: int = 1000):
        self.queue = deque(maxlen=max_size)

    def enqueue(self, request: Dict):
        """Add request to queue."""
        self.queue.append(request)

    def dequeue(self) -> Optional[Dict]:
        """Remove and return next request."""
        if self.queue:
            return self.queue.popleft()
        return None

    def size(self) -> int:
        """Get queue size."""
        return len(self.queue)


################################################################################
# SECTION 2: BATCH SCHEDULER
################################################################################

class BatchScheduler:
    """
    Batch Scheduler
    ===============

    Groups requests into batches for efficient processing.

    Strategies:
    1. Fixed batch: wait for batch_size requests
    2. Dynamic batch: process when batch is full or timeout
    3. Continuous batch: add/remove requests dynamically

    Interview Questions:
        Q: "What is continuous batching?"
        A: Requests join and leave the batch dynamically.
           Maximizes GPU utilization by always keeping GPU busy.
    """

    def __init__(self, max_batch_size: int = 8, timeout_ms: float = 100):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.pending: List[Dict] = []

    def add_request(self, request: Dict):
        """Add request to pending list."""
        self.pending.append(request)

    def get_batch(self) -> List[Dict]:
        """Get next batch of requests."""
        if len(self.pending) >= self.max_batch_size:
            batch = self.pending[:self.max_batch_size]
            self.pending = self.pending[self.max_batch_size:]
            return batch
        return []

    def get_batch_with_timeout(self) -> List[Dict]:
        """Get batch, even if not full, after timeout."""
        if self.pending:
            batch = self.pending[:self.max_batch_size]
            self.pending = self.pending[self.max_batch_size:]
            return batch
        return []


################################################################################
# SECTION 3: MODEL SERVER
################################################################################

class ModelServer:
    """
    Model Server
    ============

    Serves model predictions with batching and streaming.

    Features:
    - Request queuing
    - Batch scheduling
    - Streaming responses
    - Health checks
    - Metrics tracking

    Interview Questions:
        Q: "How do you handle high traffic?"
        A: Load balancing, autoscaling, request queuing,
           batch processing, caching.
    """

    def __init__(self, model, max_batch_size: int = 8):
        self.model = model
        self.queue = RequestQueue()
        self.scheduler = BatchScheduler(max_batch_size)
        self.request_count = 0
        self.total_tokens = 0

    def predict(self, prompt: str, max_tokens: int = 100) -> Dict:
        """
        Generate response for a prompt.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate

        Returns:
            response: Generated text and metadata
        """
        self.request_count += 1

        # Simulate generation
        generated = f"Response to: {prompt[:50]}..."
        tokens_generated = len(generated.split())
        self.total_tokens += tokens_generated

        return {
            'text': generated,
            'tokens_generated': tokens_generated,
            'request_id': self.request_count,
        }

    def health_check(self) -> Dict:
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'requests_served': self.request_count,
            'total_tokens': self.total_tokens,
            'queue_size': self.queue.size(),
        }

    def get_metrics(self) -> Dict:
        """Get server metrics."""
        return {
            'requests': self.request_count,
            'tokens': self.total_tokens,
            'avg_tokens_per_request': self.total_tokens / max(1, self.request_count),
        }


################################################################################
# SECTION 4: TESTING
################################################################################

def demonstrate_serving():
    """Demonstrate model serving."""
    print("=" * 70)
    print("MODEL SERVING DEMONSTRATION")
    print("=" * 70)

    # Create server
    server = ModelServer(model=None, max_batch_size=4)

    # Process requests
    print("\n--- Processing Requests ---")
    prompts = [
        "What is machine learning?",
        "Explain transformers",
        "How does attention work?",
    ]

    for prompt in prompts:
        result = server.predict(prompt)
        print(f"Request {result['request_id']}: {result['tokens_generated']} tokens")

    # Health check
    print("\n--- Health Check ---")
    health = server.health_check()
    print(f"Status: {health['status']}")
    print(f"Requests: {health['requests_served']}")

    # Metrics
    print("\n--- Metrics ---")
    metrics = server.get_metrics()
    print(f"Total tokens: {metrics['tokens']}")
    print(f"Avg tokens/request: {metrics['avg_tokens_per_request']:.1f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_serving()


################################################################################
# REFERENCES
################################################################################

# [1] Kwon, W., et al. (2023). Efficient Memory Management for LLM Serving with PagedAttention.

################################################################################
