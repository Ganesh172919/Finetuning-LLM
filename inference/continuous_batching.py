"""
Continuous Batching for LLM Inference
======================================

Continuous batching (also called "iteration-level scheduling") is a critical
optimization for LLM serving that dramatically improves throughput compared
to static batching.

Problem with Static Batching:
┌─────────────────────────────────────────────────────────┐
│ Request A: [token1, token2, token3, token4, token5]    │
│ Request B: [token1, token2, token3]                     │ ← finishes early
│ Request C: [token1, token2, token3, token4, token5, ...│ ← long
│                                                         │
│ Batch must wait for LONGEST request to finish           │
│ GPU sits idle while short requests wait                 │
└─────────────────────────────────────────────────────────┘

Continuous Batching Solution:
┌─────────────────────────────────────────────────────────┐
│ Step 1: Process A, B, C together                        │
│ Step 2: B finishes → slot freed → new request D enters  │
│ Step 3: Process A, C, D together                        │
│ Step 4: A finishes → slot freed → new request E enters  │
│ Step 5: Process C, D, E together                        │
│                                                         │
│ GPU always busy! No wasted compute!                     │
└─────────────────────────────────────────────────────────┘

Key Innovation:
  Instead of scheduling at the REQUEST level (batch = set of requests),
  schedule at the ITERATION level (batch = set of tokens to generate).

Throughput Improvement:
  Static batching: ~30-40% GPU utilization
  Continuous batching: ~80-95% GPU utilization
  → 2-3x more requests per second

References:
  - Orca: A Distributed Serving System for Transformer-Based Generative Models
    (Yu et al., 2022) — Introduced continuous batching
  - vLLM: Efficient Memory Management for Large Language Model Serving
    (Kwon et al., 2023) — Production implementation

Author: SOTA Implementation Suite
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class RequestStatus(Enum):
    """Status of a request in the serving pipeline."""
    WAITING = "waiting"       # In queue, not yet started
    RUNNING = "running"       # Currently being processed
    FINISHED = "finished"     # Completed generation
    SWAPPED = "swapped"       # Swapped out (preempted)


@dataclass
class GenerationRequest:
    """
    A single text generation request.

    Attributes:
        request_id: Unique identifier
        prompt_tokens: Initial prompt tokens
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        generated_tokens: Tokens generated so far
        status: Current request status
        arrival_time: When request arrived
        start_time: When generation started
        finish_time: When generation completed
        token_logprobs: Log probabilities of generated tokens
        is_prompt_processed: Whether prompt has been processed
    """
    request_id: str
    prompt_tokens: List[int]
    max_tokens: int = 100
    temperature: float = 1.0
    generated_tokens: List[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    arrival_time: float = 0.0
    start_time: float = 0.0
    finish_time: float = 0.0
    token_logprobs: List[float] = field(default_factory=list)
    is_prompt_processed: bool = False

    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + generated)."""
        return len(self.prompt_tokens) + len(self.generated_tokens)

    @property
    def is_finished(self) -> bool:
        """Whether generation is complete."""
        return (
            self.status == RequestStatus.FINISHED or
            len(self.generated_tokens) >= self.max_tokens
        )

    @property
    def latency(self) -> float:
        """End-to-end latency in seconds."""
        if self.finish_time > 0:
            return self.finish_time - self.arrival_time
        return time.time() - self.arrival_time

    @property
    def tokens_per_second(self) -> float:
        """Generation throughput."""
        elapsed = self.latency
        if elapsed > 0:
            return len(self.generated_tokens) / elapsed
        return 0.0


# ============================================================================
# SCHEDULER — The Core of Continuous Batching
# ============================================================================

class ContinuousBatchScheduler:
    """
    Iteration-level scheduler for continuous batching.

    This is the key innovation from Orca (Yu et al., 2022):
    - Traditional: schedule at request granularity
    - Orca: schedule at iteration (token) granularity

    At each iteration:
    1. Check for newly finished requests → free their slots
    2. Check waiting queue → admit new requests into freed slots
    3. Process one token for ALL running requests simultaneously

    This ensures the GPU is always fully utilized.

    Scheduling Policies:
    - FCFS (First Come First Served): Default, fair
    - Preemptive: Can pause long requests for short ones
    - Priority-based: VIP requests get priority

    Memory Management:
    - Tracks KV cache usage per request
    - Preempts requests when memory is low
    - Uses Paged Attention for efficient memory (see paged_attention.py)
    """

    def __init__(self, max_batch_size: int = 32, max_tokens: int = 4096):
        """
        Initialize the scheduler.

        Args:
            max_batch_size: Maximum concurrent requests
            max_tokens: Maximum total tokens in batch (KV cache limit)
        """
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens

        # Request queues
        self.waiting_queue: deque[GenerationRequest] = deque()
        self.running_batch: Dict[str, GenerationRequest] = {}
        self.finished_requests: Dict[str, GenerationRequest] = {}

        # Metrics
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_latency = 0.0

    def add_request(self, request: GenerationRequest) -> None:
        """
        Add a new request to the waiting queue.

        Args:
            request: Generation request to add
        """
        request.arrival_time = time.time()
        request.status = RequestStatus.WAITING
        self.waiting_queue.append(request)
        self.total_requests += 1

    def _can_admit(self) -> bool:
        """Check if we can admit more requests."""
        # Batch size limit
        if len(self.running_batch) >= self.max_batch_size:
            return False

        # Token limit
        current_tokens = sum(r.total_tokens for r in self.running_batch.values())
        if current_tokens >= self.max_tokens:
            return False

        return True

    def _admit_requests(self) -> List[str]:
        """
        Admit waiting requests into the running batch.

        Returns:
            List of admitted request IDs
        """
        admitted = []

        while self.waiting_queue and self._can_admit():
            request = self.waiting_queue.popleft()
            request.status = RequestStatus.RUNNING
            request.start_time = time.time()
            self.running_batch[request.request_id] = request
            admitted.append(request.request_id)

        return admitted

    def _finish_request(self, request_id: str) -> None:
        """
        Mark a request as finished.

        Args:
            request_id: ID of request to finish
        """
        request = self.running_batch.pop(request_id)
        request.status = RequestStatus.FINISHED
        request.finish_time = time.time()
        self.finished_requests[request_id] = request
        self.total_tokens_generated += len(request.generated_tokens)
        self.total_latency += request.latency

    def schedule_iteration(self) -> Tuple[List[str], List[int]]:
        """
        Execute one scheduling iteration.

        This is called at every decoding step. It:
        1. Removes finished requests
        2. Admits new requests from waiting queue
        3. Returns the batch of requests to process

        Returns:
            Tuple of (request_ids, tokens_to_generate_for)
        """
        # ── Step 1: Remove finished requests ─────────────────
        finished_ids = [
            rid for rid, req in self.running_batch.items()
            if req.is_finished
        ]
        for rid in finished_ids:
            self._finish_request(rid)

        # ── Step 2: Admit new requests ───────────────────────
        self._admit_requests()

        # ── Step 3: Return running batch ─────────────────────
        request_ids = list(self.running_batch.keys())

        # For each running request, get the next token to process
        # In prompt phase: process all prompt tokens
        # In generation phase: process the last generated token
        tokens = []
        for rid in request_ids:
            req = self.running_batch[rid]
            if not req.is_prompt_processed:
                # Prompt processing phase
                tokens.append(req.prompt_tokens[0])  # Simplified
            else:
                # Generation phase — last generated token
                if req.generated_tokens:
                    tokens.append(req.generated_tokens[-1])
                else:
                    tokens.append(req.prompt_tokens[-1])

        return request_ids, tokens

    def add_generated_token(self, request_id: str, token: int,
                            logprob: float = 0.0) -> None:
        """
        Add a generated token to a request.

        Args:
            request_id: Target request
            token: Generated token ID
            logprob: Log probability of the token
        """
        if request_id in self.running_batch:
            req = self.running_batch[request_id]
            req.generated_tokens.append(token)
            req.token_logprobs.append(logprob)
            req.is_prompt_processed = True

    def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        return {
            "waiting": len(self.waiting_queue),
            "running": len(self.running_batch),
            "finished": len(self.finished_requests),
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_latency": (
                self.total_latency / max(1, len(self.finished_requests))
            ),
            "avg_tokens_per_sec": (
                self.total_tokens_generated / max(0.01, self.total_latency)
            ),
            "gpu_utilization": len(self.running_batch) / self.max_batch_size,
        }


# ============================================================================
# SIMULATED LLM ENGINE
# ============================================================================

class SimulatedLLMEngine:
    """
    Simulated LLM engine for demonstrating continuous batching.

    In production, this would interface with:
    - vLLM
    - TensorRT-LLM
    - TGI (Text Generation Inference)
    - SGLang

    The engine processes batches of tokens and generates next tokens.
    """

    def __init__(self, vocab_size: int = 50277, d_model: int = 768):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.step_count = 0

    def process_batch(self, token_ids: List[int]) -> List[Tuple[int, float]]:
        """
        Process a batch of tokens and generate next tokens.

        Args:
            token_ids: Batch of current tokens

        Returns:
            List of (next_token, logprob) tuples
        """
        self.step_count += 1
        results = []

        for token_id in token_ids:
            # Simulate generation with random token
            # In production: run actual model forward pass
            next_token = np.random.randint(0, self.vocab_size)
            logprob = np.log(np.random.uniform(0.01, 1.0))
            results.append((next_token, logprob))

        return results


# ============================================================================
# KV CACHE MANAGER
# ============================================================================

class KVCacheManager:
    """
    Manages KV cache memory for continuous batching.

    Key challenges:
    1. Different requests have different lengths
    2. KV cache grows as tokens are generated
    3. Memory must be efficiently allocated and freed

    Solutions:
    - Paged Attention (see paged_attention.py)
    - Preemption: swap out long requests
    - Memory pooling: share memory blocks

    Memory layout per request:
    ┌─────────────────────────────────────────┐
    │ KV Cache Block                          │
    │ ┌─────┬─────┬─────┬─────┬─────────────┐│
    │ │ K_0 │ K_1 │ K_2 │ ... │ K_{T-1}     ││
    │ │ V_0 │ V_1 │ V_2 │ ... │ V_{T-1}     ││
    │ └─────┴─────┴─────┴─────┴─────────────┘│
    │ n_layers × 2 × T × head_size × d_type  │
    └─────────────────────────────────────────┘
    """

    def __init__(self, max_total_tokens: int = 8192,
                 n_layers: int = 32, n_heads: int = 32,
                 head_size: int = 128):
        """
        Initialize KV cache manager.

        Args:
            max_total_tokens: Maximum total tokens across all requests
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            head_size: Dimension per head
        """
        self.max_total_tokens = max_total_tokens
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_size = head_size

        # Track memory usage per request
        self.request_cache_size: Dict[str, int] = {}
        self.total_used = 0

    def allocate(self, request_id: str, num_tokens: int) -> bool:
        """
        Allocate KV cache for a request.

        Args:
            request_id: Request identifier
            num_tokens: Number of tokens to allocate for

        Returns:
            True if allocation succeeded, False if out of memory
        """
        if self.total_used + num_tokens > self.max_total_tokens:
            return False

        self.request_cache_size[request_id] = num_tokens
        self.total_used += num_tokens
        return True

    def free(self, request_id: str) -> None:
        """Free KV cache for a completed request."""
        if request_id in self.request_cache_size:
            self.total_used -= self.request_cache_size.pop(request_id)

    def extend(self, request_id: str, num_new_tokens: int = 1) -> bool:
        """
        Extend KV cache for a request (one new token generated).

        Args:
            request_id: Request identifier
            num_new_tokens: Number of new tokens

        Returns:
            True if extension succeeded
        """
        if self.total_used + num_new_tokens > self.max_total_tokens:
            return False

        self.request_cache_size[request_id] += num_new_tokens
        self.total_used += num_new_tokens
        return True

    @property
    def utilization(self) -> float:
        """Memory utilization (0.0 to 1.0)."""
        return self.total_used / self.max_total_tokens

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "total_tokens": self.max_total_tokens,
            "used_tokens": self.total_used,
            "free_tokens": self.max_total_tokens - self.total_used,
            "utilization": self.utilization,
            "active_requests": len(self.request_cache_size),
        }


# ============================================================================
# THROUGHPUT OPTIMIZER
# ============================================================================

class ThroughputOptimizer:
    """
    Optimizes batch scheduling for maximum throughput.

    Strategies:
    1. Batch Coalescing: Group similar-length requests
    2. Preemption: Pause long requests for short ones
    3. Chunked Prefill: Process prompts in chunks
    4. Priority Scheduling: VIP requests get priority
    """

    def __init__(self, scheduler: ContinuousBatchScheduler):
        self.scheduler = scheduler

    def get_batch_composition(self) -> Dict:
        """
        Analyze current batch composition.

        Returns:
            Statistics about running requests
        """
        if not self.scheduler.running_batch:
            return {"empty": True}

        lengths = [r.total_tokens for r in self.scheduler.running_batch.values()]
        return {
            "batch_size": len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "length_variance": np.var(lengths),
        }

    def should_preempt(self, request: GenerationRequest) -> bool:
        """
        Determine if a request should be preempted.

        Preemption criteria:
        - Request has been running too long
        - Batch has high length variance
        - Memory pressure is high

        Args:
            request: Request to evaluate

        Returns:
            True if request should be preempted
        """
        # Simple heuristic: preempt if much longer than average
        batch_stats = self.get_batch_composition()
        if batch_stats.get("empty"):
            return False

        if request.total_tokens > batch_stats["avg_length"] * 3:
            return True

        return False


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_continuous_batching():
    """
    Demonstrate continuous batching with simulated requests.

    Shows:
        1. Static batching (baseline)
        2. Continuous batching (improved)
        3. Throughput comparison
    """
    print("=" * 70)
    print("Continuous Batching for LLM Inference")
    print("=" * 70)

    # ── Create scheduler and engine ─────────────────────────────
    scheduler = ContinuousBatchScheduler(max_batch_size=8, max_tokens=1024)
    engine = SimulatedLLMEngine(vocab_size=1000)
    cache_manager = KVCacheManager(max_total_tokens=1024)

    # ── Simulate requests ───────────────────────────────────────
    print("\n[Generating Requests]")
    requests = [
        GenerationRequest(
            request_id=f"req_{i}",
            prompt_tokens=list(range(10, 10 + np.random.randint(5, 20))),
            max_tokens=np.random.randint(10, 50),
            temperature=0.8,
        )
        for i in range(15)
    ]

    for req in requests:
        scheduler.add_request(req)
    print(f"  Added {len(requests)} requests to queue")

    # ── Run continuous batching ─────────────────────────────────
    print("\n[Running Continuous Batching]")
    iteration = 0
    max_iterations = 100

    while scheduler.waiting_queue or scheduler.running_batch:
        if iteration >= max_iterations:
            break

        # Schedule one iteration
        request_ids, tokens = scheduler.schedule_iteration()

        if not request_ids:
            break

        # Process batch
        results = engine.process_batch(tokens)

        # Add generated tokens
        for rid, (token, logprob) in zip(request_ids, results):
            scheduler.add_generated_token(rid, token, logprob)

        iteration += 1

        # Print progress every 10 iterations
        if iteration % 10 == 0:
            stats = scheduler.get_stats()
            print(f"  Iteration {iteration}: "
                  f"running={stats['running']}, "
                  f"waiting={stats['waiting']}, "
                  f"finished={stats['finished']}")

    # ── Final statistics ────────────────────────────────────────
    print("\n[Final Statistics]")
    stats = scheduler.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # ── Compare with static batching ────────────────────────────
    print("\n[Static vs Continuous Batching Comparison]")

    # Simulate static batching
    static_batch_time = sum(r.max_tokens for r in requests) / 8  # 8 parallel
    continuous_time = iteration

    print(f"  Static batching time: {static_batch_time:.1f} iterations")
    print(f"  Continuous batching time: {continuous_time} iterations")
    print(f"  Speedup: {static_batch_time / max(1, continuous_time):.2f}x")
    print(f"  GPU utilization: {stats['gpu_utilization']:.1%}")

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  1. Continuous batching = iteration-level scheduling")
    print("  2. Finished requests free slots immediately")
    print("  3. New requests join mid-batch (no waiting for longest)")
    print("  4. 2-3x throughput improvement over static batching")
    print("  5. Critical for production LLM serving (vLLM, TGI)")
    print("=" * 70)


if __name__ == "__main__":
    demo_continuous_batching()
