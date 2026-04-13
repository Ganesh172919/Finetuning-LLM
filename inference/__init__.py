"""
################################################################################
INFERENCE & MODEL SERVING — PRODUCTION DEPLOYMENT
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Inference?
    Inference is using a trained model to make predictions.
    For LLMs, this means generating text from prompts.

Why does it matter?
    Training is expensive, but inference is what users see.
    Key challenges:
    - Latency: Users want fast responses
    - Throughput: Handle many users simultaneously
    - Cost: GPU time is expensive
    - Quality: Maintain model quality

Inference Optimization:
    1. KV Cache: Don't recompute past tokens
    2. Batching: Process multiple requests together
    3. Quantization: Use lower precision (int8, int4)
    4. Speculative Decoding: Use small model to draft
    5. Continuous Batching: Dynamic batch management
    6. PagedAttention: Efficient memory management

Production Architecture:
    ┌─────────────────────────────────────────────┐
    │ User Request                                 │
    │   ↓                                          │
    │ Load Balancer                                │
    │   ↓                                          │
    │ API Server (vLLM, TensorRT-LLM)             │
    │   ↓                                          │
    │ Request Queue                                │
    │   ↓                                          │
    │ Batch Scheduler                              │
    │   ↓                                          │
    │ GPU Inference Engine                         │
    │   ↓                                          │
    │ Response Streaming                           │
    └─────────────────────────────────────────────┘

################################################################################
"""

from .engine import InferenceEngine, SamplingParams
from .batching import ContinuousBatcher
