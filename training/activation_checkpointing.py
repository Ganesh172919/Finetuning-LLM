"""
Activation Checkpointing (Gradient Checkpointing)
===================================================

A memory optimization technique that trades compute for memory by
selectively discarding intermediate activations during the forward pass
and recomputing them during the backward pass.

BEGINNER LEVEL:
    When training a neural network, the system remembers every calculation
    along the way (activations). For big models, this eats huge amounts
    of memory. Activation checkpointing says: "Don't remember everything,
    just remember a few checkpoints. We'll redo the math when needed."
    Like taking notes at key points instead of recording the whole lecture.

INTERMEDIATE LEVEL:
    During backpropagation, we need intermediate activations to compute
    gradients. Normally these are all stored during the forward pass,
    requiring O(L) memory where L is the number of layers. With
    checkpointing, we store only every √L activations and recompute
    the rest, reducing memory to O(√L) at the cost of ~33% more compute.

ADVANCED LEVEL:
    The key insight is that backpropagation traverses the computation
    graph in reverse order. By storing activations at strategic
    "checkpoint" layers, we can recompute all intermediate activations
    between two checkpoints during the backward pass. The optimal
    checkpoint placement follows a recursive bisection strategy.

EXPERT LEVEL:
    PyTorch's torch.utils.checkpoint implements this with a clever
    trick: it detaches tensors from the computation graph during
    the forward pass and rebuilds the graph during backward. This
    works seamlessly with autograd. The recomputation cost is
    typically 20-35% additional FLOPs, but enables 2-4x larger models.

INTERVIEW LEVEL:
    Q: How does gradient checkpointing reduce memory?
    A: Instead of storing all L layers' activations (O(L) memory),
    we store √L checkpoints and recompute the rest during backward.
    This reduces memory to O(√L) at the cost of ~33% more compute.
    Q: When would you use it?
    A: When training models that don't fit in GPU memory, before
    switching to model parallelism (which has communication overhead).
"""

import numpy as np
from typing import List, Callable, Tuple, Optional, Dict
from dataclasses import dataclass
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CheckpointConfig:
    """Configuration for activation checkpointing.

    ATTRIBUTES:
        num_checkpoints: Number of checkpoint segments
            - None: auto-detect based on num_layers (sqrt(num_layers))
            - int: explicit number of checkpoints
        use_reentrant: Whether to use reentrant autograd (PyTorch-specific)
        preserve_rng_state: Whether to preserve RNG state for reproducibility

    TRADEOFFS:
        More checkpoints = less recomputation but more memory
        Fewer checkpoints = more recomputation but less memory
        Optimal: sqrt(num_layers) checkpoints balances both
    """

    num_checkpoints: Optional[int] = None
    use_reentrant: bool = False
    preserve_rng_state: bool = True


# ============================================================================
# ACTIVATION STORAGE ANALYZER
# ============================================================================

class ActivationAnalyzer:
    """Analyzes memory usage of activations in a neural network.

    ARCHITECTURE:
        - Estimates activation memory per layer
        - Computes optimal checkpoint placement
        - Reports memory savings

    WHY ANALYZE FIRST:
        Before applying checkpointing, we need to understand where
        memory is spent. Different layers have different activation
        sizes (attention layers are much larger than FFN layers).
    """

    @staticmethod
    def estimate_layer_activation_memory(
        batch_size: int,
        seq_len: int,
        d_model: int,
        layer_type: str = 'transformer',
        dtype_bytes: int = 4,  # float32 = 4, float16 = 2
    ) -> int:
        """Estimate activation memory for a single layer.

        DEFINITION:
            Computes how many bytes of activations a single layer
            stores during the forward pass.

        EXECUTION FLOW:
            1. Compute input activation size
            2. Add attention-specific activations (Q, K, V, attention weights)
            3. Add FFN activations
            4. Add normalization activations

        COMPLEXITY:
            Time: O(1)
            Space: O(1)

        EDUCATIONAL:
            For a transformer layer:
            - Input: B × T × d (batch × seq_len × d_model)
            - Q, K, V: 3 × B × T × d
            - Attention weights: B × h × T × T (the big one!)
            - FFN intermediate: B × T × 4d
            - Output: B × T × d
            Total ≈ B × T × (9d + h×T) × dtype_bytes
        """
        B, T, d = batch_size, seq_len, d_model

        if layer_type == 'transformer':
            # Input activation
            input_act = B * T * d

            # Q, K, V projections
            qkv = 3 * B * T * d

            # Attention weights: B × heads × T × T
            # This is often the dominant term for long sequences!
            num_heads = max(1, d // 64)
            attn_weights = B * num_heads * T * T

            # Softmax output (same size as attention weights)
            softmax_out = attn_weights

            # FFN: first linear expands to 4d, second projects back
            ffn_intermediate = B * T * 4 * d
            ffn_output = B * T * d

            # LayerNorm activations
            layernorm = 2 * B * T * d

            total = input_act + qkv + attn_weights + softmax_out + ffn_intermediate + ffn_output + layernorm

        elif layer_type == 'mamba':
            # Mamba has much smaller activations (no attention matrix!)
            input_act = B * T * d
            conv_state = B * d * 4  # Conv1d state
            ssm_state = B * d * d  # SSM state
            gate_act = B * T * d
            total = input_act + conv_state + ssm_state + gate_act

        elif layer_type == 'rwkv':
            # RWKV: RNN-style, no attention matrix
            input_act = B * T * d
            time_state = B * d * d  # WKV state
            channel_state = B * d * 4 * d  # FFN state
            total = input_act + time_state + channel_state

        else:
            # Generic: assume d-sized activations
            total = B * T * d * 5

        return total * dtype_bytes

    @staticmethod
    def estimate_total_memory(
        num_layers: int,
        batch_size: int,
        seq_len: int,
        d_model: int,
        layer_type: str = 'transformer',
        dtype_bytes: int = 4,
    ) -> Dict[str, float]:
        """Estimate total activation memory for the full model.

        DEFINITION:
            Sums activation memory across all layers and reports
            in human-readable units.

        OUTPUTS:
            Dict with:
            - per_layer_bytes: bytes per layer
            - total_bytes: total without checkpointing
            - with_checkpoint_bytes: total with optimal checkpointing
            - savings_bytes: memory saved
            - savings_ratio: ratio of saved to original
        """
        per_layer = ActivationAnalyzer.estimate_layer_activation_memory(
            batch_size, seq_len, d_model, layer_type, dtype_bytes
        )

        total = per_layer * num_layers

        # With optimal checkpointing (sqrt(n) checkpoints)
        # Memory = checkpoints × full_memory + (n/checkpoints) × per_layer
        # Optimal checkpoints = sqrt(n)
        optimal_ckpt = max(1, int(math.sqrt(num_layers)))
        with_ckpt = optimal_ckpt * per_layer + (num_layers // optimal_ckpt) * per_layer

        return {
            'per_layer_bytes': per_layer,
            'total_bytes': total,
            'total_mb': total / (1024 * 1024),
            'with_checkpoint_bytes': with_ckpt,
            'with_checkpoint_mb': with_ckpt / (1024 * 1024),
            'savings_bytes': total - with_ckpt,
            'savings_mb': (total - with_ckpt) / (1024 * 1024),
            'savings_ratio': (total - with_ckpt) / total if total > 0 else 0,
            'optimal_checkpoints': optimal_ckpt,
            'recomputation_overhead': 0.33,  # ~33% more FLOPs
        }


# ============================================================================
# CHECKPOINT SCHEDULER
# ============================================================================

class CheckpointScheduler:
    """Determines optimal checkpoint placement in a network.

    ARCHITECTURE:
        - Analyzes layer memory profiles
        - Places checkpoints to minimize peak memory
        - Supports uniform and non-uniform placement

    STRATEGIES:
        1. Uniform: evenly space checkpoints every n/checkpoints layers
        2. Recursive: bisect the network recursively (optimal for uniform layers)
        3. Adaptive: place more checkpoints where memory usage is highest

    WHY SCHEDULING MATTERS:
        Naive checkpointing (every other layer) gives ~2x savings.
        Optimal checkpointing (recursive bisection) gives O(sqrt(n)) memory.
        For 100 layers: naive = 50 stored, optimal = 10 stored.
    """

    @staticmethod
    def uniform_placement(num_layers: int, num_checkpoints: int) -> List[int]:
        """Place checkpoints uniformly across layers.

        DEFINITION:
            Stores activations at evenly spaced layers.

        EXAMPLE:
            num_layers=12, num_checkpoints=4
            → checkpoints at layers [0, 3, 6, 9]

        COMPLEXITY:
            Time: O(1)
            Space: O(num_checkpoints)
        """
        if num_checkpoints <= 0:
            return []
        if num_checkpoints >= num_layers:
            return list(range(num_layers))

        interval = num_layers / num_checkpoints
        return [int(i * interval) for i in range(num_checkpoints)]

    @staticmethod
    def recursive_bisection(num_layers: int, num_checkpoints: Optional[int] = None) -> List[int]:
        """Place checkpoints using recursive bisection (optimal strategy).

        DEFINITION:
            Recursively bisects the network to find optimal checkpoint
            positions. This minimizes peak memory usage for uniform layers.

        ALGORITHM:
            The optimal strategy for n layers with k checkpoints is:
            1. Place first checkpoint at n/k position
            2. Recursively solve for left (n/k layers, k/2 checkpoints)
            3. Recursively solve for right (n-n/k layers, k/2 checkpoints)

        For simplicity, we use the analytical solution:
            checkpoints = [i * sqrt(n) for i in range(sqrt(n))]

        EDUCATIONAL:
            This is the strategy from Griewank & Walther (2000).
            It achieves O(sqrt(n)) memory with O(n) recomputation.
            The key insight: storing sqrt(n) equally-spaced checkpoints
            means each segment has sqrt(n) layers, so the maximum
            recomputation per backward step is sqrt(n) layers.
        """
        if num_checkpoints is None:
            num_checkpoints = max(1, int(math.sqrt(num_layers)))

        if num_checkpoints >= num_layers:
            return list(range(num_layers))

        # Analytical solution: equally spaced at sqrt(n) intervals
        interval = num_layers / num_checkpoints
        checkpoints = []
        for i in range(num_checkpoints):
            pos = int(i * interval)
            if pos not in checkpoints:
                checkpoints.append(pos)

        return sorted(checkpoints)

    @staticmethod
    def adaptive_placement(
        num_layers: int,
        layer_memory_profile: List[int],
        budget_bytes: int,
    ) -> List[int]:
        """Place checkpoints adaptively based on memory profile.

        DEFINITION:
            Places more checkpoints where memory usage is highest.
            Greedy: always checkpoint the layer that reduces peak memory most.

        EXECUTION FLOW:
            1. Start with no checkpoints (peak = sum of all layers)
            2. For each possible checkpoint, compute resulting peak memory
            3. Place checkpoint at the position that reduces peak most
            4. Repeat until budget is met

        COMPLEXITY:
            Time: O(n × k) where k = number of checkpoints placed
            Space: O(n)
        """
        if not layer_memory_profile:
            return []

        checkpoints = []
        current_peak = sum(layer_memory_profile)

        while current_peak > budget_bytes and len(checkpoints) < num_layers:
            best_reduction = 0
            best_pos = 0

            for pos in range(num_layers):
                if pos in checkpoints:
                    continue

                # Simulate placing checkpoint here
                test_checkpoints = sorted(checkpoints + [pos])
                peak = CheckpointScheduler._compute_peak_memory(
                    layer_memory_profile, test_checkpoints
                )
                reduction = current_peak - peak

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_pos = pos

            if best_reduction > 0:
                checkpoints.append(best_pos)
                current_peak -= best_reduction
            else:
                break

        return sorted(checkpoints)

    @staticmethod
    def _compute_peak_memory(
        layer_memory: List[int],
        checkpoints: List[int],
    ) -> int:
        """Compute peak memory given checkpoint positions.

        DEFINITION:
            Simulates the backward pass and computes the maximum
            memory usage at any point.

        EXECUTION FLOW:
            1. For each segment between checkpoints:
               - Memory = sum of activations in segment
            2. Peak = max(segment memories)

        EDUCATIONAL:
            With checkpoints at [0, 5, 10], the backward pass for
            layers 5-10 needs to store at most 5 layers' activations
            at a time (recomputed from checkpoint at layer 5).
        """
        if not checkpoints:
            return sum(layer_memory)

        max_segment = 0
        sorted_ckpts = sorted(checkpoints + [len(layer_memory)])

        prev = 0
        for ckpt in sorted_ckpts:
            segment_mem = sum(layer_memory[prev:ckpt])
            max_segment = max(max_segment, segment_mem)
            prev = ckpt

        return max_segment


# ============================================================================
# SIMULATED CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Manages activation checkpointing for training.

    ARCHITECTURE:
        - Tracks which layers are checkpointed
        - Simulates forward/backward with checkpointing
        - Reports memory usage and recomputation cost

    USAGE PATTERN:
        manager = CheckpointManager(config, num_layers=24)
        manager.configure(batch_size=32, seq_len=2048, d_model=1024)

        # During training (simulated)
        for step in range(num_steps):
            loss = manager.forward_backward_step()
    """

    def __init__(self, config: CheckpointConfig, num_layers: int):
        """Initialize the checkpoint manager.

        Args:
            config: Checkpointing configuration
            num_layers: Number of layers in the model
        """
        self.config = config
        self.num_layers = num_layers

        # Auto-determine number of checkpoints
        if config.num_checkpoints is None:
            self.num_checkpoints = max(1, int(math.sqrt(num_layers)))
        else:
            self.num_checkpoints = config.num_checkpoints

        # Compute checkpoint positions
        self.checkpoint_layers = CheckpointScheduler.recursive_bisection(
            num_layers, self.num_checkpoints
        )

        # Track memory usage
        self.memory_log: List[Dict] = []

    def configure(
        self,
        batch_size: int = 32,
        seq_len: int = 2048,
        d_model: int = 1024,
        layer_type: str = 'transformer',
        dtype_bytes: int = 4,
    ):
        """Configure activation memory estimates.

        Args:
            batch_size: Training batch size
            seq_len: Sequence length
            d_model: Model dimension
            layer_type: Type of layers (transformer/mamba/rwkv)
            dtype_bytes: Bytes per element (4 for float32, 2 for float16)
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.layer_type = layer_type
        self.dtype_bytes = dtype_bytes

        # Compute memory profile
        self.memory_estimate = ActivationAnalyzer.estimate_total_memory(
            num_layers=self.num_layers,
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            layer_type=layer_type,
            dtype_bytes=dtype_bytes,
        )

        # Per-layer memory (for peak computation)
        self.per_layer_bytes = ActivationAnalyzer.estimate_layer_activation_memory(
            batch_size, seq_len, d_model, layer_type, dtype_bytes
        )

    def get_memory_report(self) -> Dict:
        """Generate comprehensive memory report.

        DEFINITION:
            Reports memory usage with and without checkpointing,
            including recomputation cost.

        OUTPUTS:
            Dict with:
            - without_checkpoint: memory without optimization
            - with_checkpoint: memory with checkpointing
            - savings: memory saved
            - recomputation_cost: additional FLOPs
            - checkpoint_positions: which layers are checkpointed
        """
        total_no_ckpt = self.per_layer_bytes * self.num_layers
        total_with_ckpt = self.memory_estimate['with_checkpoint_bytes']

        # Compute recomputation: each segment between checkpoints
        # requires recomputing (segment_size) layers
        segments = []
        sorted_ckpts = sorted(self.checkpoint_layers + [self.num_layers])
        prev = 0
        total_recompute_layers = 0
        for ckpt in sorted_ckpts:
            seg_size = ckpt - prev
            segments.append(seg_size)
            # During backward, we recompute (seg_size - 1) layers per segment
            total_recompute_layers += max(0, seg_size - 1)
            prev = ckpt

        recompute_ratio = total_recompute_layers / self.num_layers if self.num_layers > 0 else 0

        return {
            'num_layers': self.num_layers,
            'num_checkpoints': self.num_checkpoints,
            'checkpoint_positions': self.checkpoint_layers,
            'segment_sizes': segments,
            'without_checkpoint_mb': total_no_ckpt / (1024 * 1024),
            'with_checkpoint_mb': total_with_ckpt / (1024 * 1024),
            'savings_mb': (total_no_ckpt - total_with_ckpt) / (1024 * 1024),
            'savings_ratio': (total_no_ckpt - total_with_ckpt) / total_no_ckpt if total_no_ckpt > 0 else 0,
            'recompute_layers': total_recompute_layers,
            'recompute_overhead': recompute_ratio,
            'peak_memory_mb': self.memory_estimate['with_checkpoint_mb'],
            'layer_type': self.layer_type,
            'batch_size': self.batch_size,
            'seq_len': self.seq_len,
            'd_model': self.d_model,
        }

    def simulate_forward_backward(self, num_steps: int = 10) -> List[Dict]:
        """Simulate training steps with checkpointing.

        DEFINITION:
            Simulates forward and backward passes, tracking memory
            usage at each step.

        EXECUTION FLOW:
            For each step:
                1. Forward pass: store only checkpoint activations
                2. Compute loss
                3. Backward pass: recompute activations from checkpoints
                4. Free recomputed activations
                5. Log peak memory

        EDUCATIONAL:
            The memory profile during backward is NOT uniform:
            - At the last layer: need to store 1 segment of activations
            - At checkpoint boundaries: memory drops (freed segment)
            - Peak is when processing the largest segment
        """
        log = []
        sorted_ckpts = sorted(self.checkpoint_layers + [self.num_layers])

        for step in range(num_steps):
            # Simulate forward: memory = sum of checkpoint activations
            forward_mem = len(self.checkpoint_layers) * self.per_layer_bytes

            # Simulate backward: peak = max segment size + checkpoints
            max_segment = 0
            prev = 0
            for ckpt in sorted_ckpts:
                seg_size = ckpt - prev
                seg_mem = seg_size * self.per_layer_bytes
                max_segment = max(max_segment, seg_mem)
                prev = ckpt

            peak_mem = forward_mem + max_segment

            # Add some noise to simulate real-world variance
            noise = np.random.normal(1.0, 0.02)
            peak_mem = int(peak_mem * noise)

            log.append({
                'step': step,
                'forward_memory_mb': forward_mem / (1024 * 1024),
                'peak_memory_mb': peak_mem / (1024 * 1024),
                'recomputation_cost': self.memory_estimate['recomputation_overhead'],
            })

        self.memory_log = log
        return log


# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

def compare_strategies(
    num_layers: int,
    batch_size: int,
    seq_len: int,
    d_model: int,
    layer_type: str = 'transformer',
) -> Dict[str, Dict]:
    """Compare different checkpointing strategies.

    DEFINITION:
        Runs all checkpoint strategies and compares memory usage,
        recomputation cost, and tradeoffs.

    STRATEGIES COMPARED:
        1. No checkpointing (baseline)
        2. Uniform checkpointing
        3. Recursive bisection (optimal)
        4. Every-other-layer (simple)

    EDUCATIONAL:
        This comparison shows why checkpointing strategy matters:
        - No checkpointing: lowest compute, highest memory
        - Every-other: 2x memory reduction, 50% more compute
        - Recursive bisection: sqrt(n) memory, ~33% more compute
        - The optimal strategy depends on your memory budget
    """
    results = {}

    # 1. No checkpointing
    total_mem = ActivationAnalyzer.estimate_layer_activation_memory(
        batch_size, seq_len, d_model, layer_type
    ) * num_layers
    results['none'] = {
        'strategy': 'No Checkpointing',
        'memory_mb': total_mem / (1024 * 1024),
        'checkpoints': 0,
        'recompute_overhead': 0.0,
    }

    # 2. Uniform checkpointing
    for n_ckpt in [2, 4, int(math.sqrt(num_layers))]:
        ckpts = CheckpointScheduler.uniform_placement(num_layers, n_ckpt)
        scheduler = CheckpointScheduler._compute_peak_memory(
            [ActivationAnalyzer.estimate_layer_activation_memory(
                batch_size, seq_len, d_model, layer_type
            )] * num_layers,
            ckpts
        )
        results[f'uniform_{n_ckpt}'] = {
            'strategy': f'Uniform ({n_ckpt} ckpts)',
            'memory_mb': scheduler / (1024 * 1024),
            'checkpoints': n_ckpt,
            'recompute_overhead': (num_layers - n_ckpt) / num_layers,
        }

    # 3. Recursive bisection (optimal)
    opt_ckpt = max(1, int(math.sqrt(num_layers)))
    ckpts = CheckpointScheduler.recursive_bisection(num_layers)
    peak = CheckpointScheduler._compute_peak_memory(
        [ActivationAnalyzer.estimate_layer_activation_memory(
            batch_size, seq_len, d_model, layer_type
        )] * num_layers,
        ckpts
    )
    results['recursive'] = {
        'strategy': f'Recursive Bisection ({len(ckpts)} ckpts)',
        'memory_mb': peak / (1024 * 1024),
        'checkpoints': len(ckpts),
        'recompute_overhead': 0.33,
    }

    # 4. Every other layer
    ckpts = list(range(0, num_layers, 2))
    peak = CheckpointScheduler._compute_peak_memory(
        [ActivationAnalyzer.estimate_layer_activation_memory(
            batch_size, seq_len, d_model, layer_type
        )] * num_layers,
        ckpts
    )
    results['every_other'] = {
        'strategy': 'Every Other Layer',
        'memory_mb': peak / (1024 * 1024),
        'checkpoints': len(ckpts),
        'recompute_overhead': 0.5,
    }

    return results


def print_memory_report(report: Dict):
    """Pretty-print a memory report.

    EDUCATIONAL:
        Visual representation of memory savings helps understand
        the impact of checkpointing at a glance.
    """
    print("=" * 60)
    print("ACTIVATION CHECKPOINTING MEMORY REPORT")
    print("=" * 60)
    print(f"Model: {report['num_layers']}L, {report['d_model']}d, {report['layer_type']}")
    print(f"Batch: {report['batch_size']}, SeqLen: {report['seq_len']}")
    print(f"Checkpoints: {report['num_checkpoints']} at layers {report['checkpoint_positions']}")
    print()
    print(f"Without checkpointing: {report['without_checkpoint_mb']:.1f} MB")
    print(f"With checkpointing:    {report['with_checkpoint_mb']:.1f} MB")
    print(f"Memory saved:          {report['savings_mb']:.1f} MB ({report['savings_ratio']:.1%})")
    print(f"Recomputation cost:    {report['recompute_overhead']:.1%} more FLOPs")
    print()
    print("Segments between checkpoints:")
    for i, seg_size in enumerate(report['segment_sizes']):
        bar = "#" * min(40, seg_size)
        print(f"  Segment {i}: {seg_size:3d} layers [{bar}]")
    print("=" * 60)
