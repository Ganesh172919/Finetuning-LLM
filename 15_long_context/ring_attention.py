"""
################################################################################
RING ATTENTION — DISTRIBUTED SEQUENCE PARALLELISM FOR LONG CONTEXT
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Ring Attention?
    Ring Attention distributes a long sequence across multiple devices,
    each holding a block of KV. Devices pass KV blocks in a ring while
    computing partial attention, enabling arbitrarily long sequences.

Why does it matter?
    Standard attention needs O(N²) memory — 1M tokens = 1TB KV cache.
    Ring Attention distributes this: 1M tokens on 64 devices = 16K per device.

How does it work?
    1. Split sequence into blocks, one per device
    2. Each device computes attention with its local KV block
    3. Pass KV block to next device in ring
    4. Repeat until all blocks have been processed
    5. Combine partial results

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


################################################################################
# SECTION 1: CONFIGURATION
################################################################################

@dataclass
class RingAttentionConfig:
    """Ring Attention configuration."""
    block_size: int = 1024
    num_devices: int = 4
    causal: bool = True
    d_model: int = 256
    n_heads: int = 8


################################################################################
# SECTION 2: BLOCK SPARSE ATTENTION
################################################################################

class BlockSparseAttention:
    """
    Block-Sparse Attention — Attend only between relevant blocks.

    Patterns:
    - Local: attend to neighboring blocks
    - Global: special tokens attend to all
    - Strided: attend every Nth block

    Interview Question:
        "What is block-sparse attention?"
        Divide sequence into blocks, compute attention only between
        relevant block pairs. Local blocks get full attention, distant
        blocks get sparse attention. Reduces O(N²) to O(N*W) where W
        is the local window size.
    """

    def __init__(self, block_size: int = 256, local_blocks: int = 3):
        self.block_size = block_size
        self.local_blocks = local_blocks

    def create_mask(self, seq_len: int) -> np.ndarray:
        """
        Create block-sparse attention mask.

        Args:
            seq_len: Sequence length

        Returns:
            Attention mask (seq_len, seq_len)
        """
        n_blocks = math.ceil(seq_len / self.block_size)
        mask = np.zeros((n_blocks, n_blocks))

        for i in range(n_blocks):
            # Local blocks
            for j in range(max(0, i - self.local_blocks),
                          min(n_blocks, i + self.local_blocks + 1)):
                mask[i, j] = 1.0

        # Expand to full mask
        full_mask = np.zeros((seq_len, seq_len))
        for i in range(n_blocks):
            for j in range(n_blocks):
                if mask[i, j]:
                    i_start = i * self.block_size
                    i_end = min((i + 1) * self.block_size, seq_len)
                    j_start = j * self.block_size
                    j_end = min((j + 1) * self.block_size, seq_len)
                    full_mask[i_start:i_end, j_start:j_end] = 1.0

        return full_mask


################################################################################
# SECTION 3: RING COMMUNICATOR
################################################################################

class RingCommunicator:
    """
    Simulate ring communication between devices.

    Each device passes its KV block to the next device in the ring.
    Communication overlaps with computation.

    Interview Question:
        "How does ring communication work in Ring Attention?"
        Devices are arranged in a ring. Each holds a KV block. In each
        step: (1) compute attention with current KV block, (2) send KV
        block to next device, (3) receive KV block from previous device.
        Communication overlaps with computation, hiding latency.
    """

    def __init__(self, num_devices: int):
        self.num_devices = num_devices

    def get_next_device(self, device_id: int) -> int:
        """Get next device in ring."""
        return (device_id + 1) % self.num_devices

    def get_prev_device(self, device_id: int) -> int:
        """Get previous device in ring."""
        return (device_id - 1) % self.num_devices


################################################################################
# SECTION 4: RING ATTENTION
################################################################################

class RingAttention:
    """
    Ring Attention — Distributed long-context attention.

    Paper: "Ring Attention with Blockwise Transformers for
            Near-Infinite Context" (Liu et al., ICLR 2024)

    Key Insight:
        Each device computes attention with one KV block at a time.
        As KV blocks rotate through the ring, each device eventually
        sees all KV blocks. Communication overlaps with computation.

    Interview Question:
        "How does Ring Attention enable infinite context?"
        Split the sequence across devices. Each device holds Q and one
        KV block. Compute attention, then pass KV to next device. After
        N rotations, each device has attended to all KV blocks. Memory
        per device = O(N/P) where P = number of devices.
    """

    def __init__(self, config: Optional[RingAttentionConfig] = None):
        self.config = config or RingAttentionConfig()
        self.communicator = RingCommunicator(self.config.num_devices)

    def split_sequence(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Split sequence across devices.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            List of blocks, one per device
        """
        seq_len = x.shape[1]
        block_size = math.ceil(seq_len / self.config.num_devices)
        blocks = []
        for i in range(self.config.num_devices):
            start = i * block_size
            end = min(start + block_size, seq_len)
            blocks.append(x[:, start:end, :])
        return blocks

    def compute_local_attention(self, q: np.ndarray, kv: np.ndarray) -> np.ndarray:
        """
        Compute attention between Q and one KV block.

        Args:
            q: Query block (batch, q_len, d_model)
            kv: Key-Value block (batch, kv_len, d_model)

        Returns:
            Attention output (batch, q_len, d_model)
        """
        head_dim = self.config.d_model // self.config.n_heads
        # Simplified: random attention output
        return np.random.randn(*q.shape) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Ring Attention forward pass.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        # Split Q across devices
        q_blocks = self.split_sequence(x)
        # Split KV across devices (same as Q for self-attention)
        kv_blocks = self.split_sequence(x)

        outputs = []
        for device_id in range(self.config.num_devices):
            q = q_blocks[device_id]
            device_output = np.zeros_like(q)

            # Rotate KV blocks through this device
            current_kv_idx = device_id
            for step in range(self.config.num_devices):
                kv = kv_blocks[current_kv_idx]
                partial = self.compute_local_attention(q, kv)
                device_output += partial
                current_kv_idx = self.communicator.get_next_device(current_kv_idx)

            outputs.append(device_output)

        # Concatenate outputs
        return np.concatenate(outputs, axis=1)


################################################################################
# SECTION 5: DISTRIBUTED SEQUENCE PARALLELISM
################################################################################

class DistributedSequenceParallelism:
    """
    Higher-level abstraction for sequence parallelism.

    Interview Question:
        "What is sequence parallelism?"
        Split the sequence dimension across devices. Each device processes
        a sub-sequence. Attention requires communication between devices
        (all-reduce or ring). This enables processing sequences longer
        than fit in a single device's memory.
    """

    def __init__(self, num_devices: int, block_size: int = 1024):
        self.num_devices = num_devices
        self.block_size = block_size
        self.ring = RingAttention(RingAttentionConfig(
            block_size=block_size, num_devices=num_devices
        ))

    def memory_per_device(self, seq_len: int, d_model: int) -> float:
        """
        Compute memory per device.

        Args:
            seq_len: Total sequence length
            d_model: Model dimension

        Returns:
            Memory in MB per device
        """
        block_len = math.ceil(seq_len / self.num_devices)
        # KV cache: 2 * seq_len * d_model * 4 bytes (FP32)
        kv_bytes = 2 * block_len * d_model * 4
        return kv_bytes / 1e6


################################################################################
# SECTION 6: DEMONSTRATION
################################################################################

def demonstrate_ring_attention():
    """Demonstrate Ring Attention."""
    print("=" * 70)
    print("RING ATTENTION DEMONSTRATION")
    print("=" * 70)

    # Block Sparse Attention
    print("\n1. BLOCK SPARSE ATTENTION")
    print("-" * 40)
    bsa = BlockSparseAttention(block_size=64, local_blocks=2)
    mask = bsa.create_mask(256)
    density = mask.sum() / mask.size
    print(f"  Sequence: 256, Block size: 64")
    print(f"  Mask density: {density:.2%}")
    print(f"  Full attention density: 100%")

    # Ring Attention
    print("\n2. RING ATTENTION")
    print("-" * 40)
    config = RingAttentionConfig(block_size=256, num_devices=4, d_model=128)
    ring = RingAttention(config)
    x = np.random.randn(1, 1024, 128)
    output = ring.forward(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Devices: {config.num_devices}")
    print(f"  Block size: {x.shape[1] // config.num_devices}")

    # Memory Analysis
    print("\n3. MEMORY ANALYSIS")
    print("-" * 40)
    dsp = DistributedSequenceParallelism(num_devices=8)
    for seq_len in [4096, 16384, 65536, 262144, 1048576]:
        mem = dsp.memory_per_device(seq_len, 4096)
        print(f"  Seq {seq_len:8d}: {mem:8.1f} MB/device (8 devices)")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_ring_attention()
