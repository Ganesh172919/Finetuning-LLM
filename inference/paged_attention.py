"""
################################################################################
PAGED ATTENTION — EFFICIENT KV CACHE MANAGEMENT (2023-2025 SOTA)
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Paged Attention?
    Paged Attention is a memory management technique for the KV cache
    during LLM inference. Instead of allocating contiguous memory for
    the full KV cache, it uses a paging system (like virtual memory
    in operating systems) to manage memory in fixed-size blocks.

The Problem:
    During LLM inference, the KV cache grows with sequence length.
    Traditional implementation allocates contiguous memory for the
    maximum possible sequence length, leading to:
    - Memory waste (most sequences are shorter than max)
    - Internal fragmentation (allocated but unused memory)
    - External fragmentation (scattered free blocks)
    - Memory limit on concurrent requests

The Solution:
    Paged Attention borrows from OS virtual memory:
    - Fixed-size blocks (pages) for KV cache
    - Non-contiguous allocation (pages can be anywhere)
    - Block table maps logical → physical blocks
    - Only allocate blocks as needed (no waste)

Benefits:
    - Near-zero memory waste (<4% vs 60-80% for contiguous)
    - 2-4x more concurrent requests
    - Enables longer sequences
    - Supports copy-on-write for beam search

Interview Questions:
    Q: "What is Paged Attention?"
    A: A memory management technique for KV cache that uses fixed-size
       blocks instead of contiguous allocation. Like virtual memory in
       OSes, it maps logical blocks to physical blocks, eliminating
       fragmentation and enabling 2-4x more concurrent requests.

    Q: "How does Paged Attention improve throughput?"
    A: By eliminating memory fragmentation, it can serve 2-4x more
       concurrent requests on the same GPU. Traditional allocation
       wastes 60-80% of KV cache memory; Paged Attention wastes <4%.

################################################################################
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

################################################################################
# SECTION 1: BLOCK TABLE
################################################################################

@dataclass
class BlockTableEntry:
    """Maps a logical block to a physical block."""
    logical_block: int
    physical_block: int
    ref_count: int = 1  # For copy-on-write


class BlockTable:
    """
    Block Table — Maps logical to physical KV cache blocks.

    Like a page table in an OS, this maps the logical blocks
    (what the model sees) to physical blocks (where data lives
    in GPU memory).

    Interview Question:
        Q: "What is the block table in Paged Attention?"
        A: It's analogous to a page table in virtual memory. It maps
           logical block indices (sequential) to physical block indices
           (can be non-contiguous). This enables efficient memory
           management without requiring contiguous allocation.
    """

    def __init__(self):
        self.entries: Dict[int, BlockTableEntry] = {}

    def map_block(self, logical: int, physical: int) -> None:
        """Map a logical block to a physical block."""
        self.entries[logical] = BlockTableEntry(
            logical_block=logical,
            physical_block=physical,
        )

    def get_physical(self, logical: int) -> Optional[int]:
        """Get physical block for a logical block."""
        entry = self.entries.get(logical)
        return entry.physical_block if entry else None

    def increment_ref(self, logical: int) -> None:
        """Increment reference count (for copy-on-write)."""
        if logical in self.entries:
            self.entries[logical].ref_count += 1

    def decrement_ref(self, logical: int) -> int:
        """Decrement reference count. Returns new count."""
        if logical in self.entries:
            self.entries[logical].ref_count -= 1
            return self.entries[logical].ref_count
        return 0


################################################################################
# SECTION 2: PHYSICAL BLOCK POOL
################################################################################

class PhysicalBlockPool:
    """
    Pool of physical memory blocks for KV cache.

    Manages allocation and deallocation of fixed-size blocks.
    Similar to a physical memory manager in an OS.

    Interview Question:
        Q: "How does the physical block pool work?"
        A: It maintains a free list of available blocks. When a new
           block is needed, it pops from the free list. When a block
           is no longer referenced, it's returned to the free list.
           This ensures efficient memory reuse.
    """

    def __init__(self, num_blocks: int, block_size: int, d_model: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.d_model = d_model

        # Physical storage: [num_blocks × block_size × d_model]
        # In production: this is GPU memory
        self.k_cache = np.zeros((num_blocks, block_size, d_model), dtype=np.float16)
        self.v_cache = np.zeros((num_blocks, block_size, d_model), dtype=np.float16)

        # Free list
        self.free_blocks = list(range(num_blocks))

        # Statistics
        self.allocated_count = 0
        self.peak_usage = 0

    def allocate(self) -> Optional[int]:
        """Allocate a physical block. Returns block index or None."""
        if not self.free_blocks:
            return None

        block = self.free_blocks.pop(0)
        self.allocated_count += 1
        self.peak_usage = max(self.peak_usage, self.allocated_count)
        return block

    def free(self, block: int) -> None:
        """Free a physical block."""
        self.free_blocks.append(block)
        self.allocated_count -= 1

    def get_usage(self) -> Dict:
        """Get memory usage statistics."""
        return {
            "total_blocks": self.num_blocks,
            "allocated": self.allocated_count,
            "free": len(self.free_blocks),
            "utilization": self.allocated_count / self.num_blocks,
            "peak_usage": self.peak_usage,
            "memory_mb": self.num_blocks * self.block_size * self.d_model * 2 * 2 / 1e6,
        }


################################################################################
# SECTION 3: PAGED ATTENTION ENGINE
################################################################################

class PagedAttentionEngine:
    """
    Paged Attention Engine — Complete KV cache management.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Paged Attention Engine                    │
    │                                                              │
    │  Request → Logical Blocks → Block Table → Physical Blocks   │
    │                                                              │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
    │  │ Logical  │    │  Block   │    │ Physical │              │
    │  │ Blocks   │───▶│  Table   │───▶│  Blocks  │              │
    │  │ (seq)    │    │ (mapping)│    │ (GPU mem)│              │
    │  └──────────┘    └──────────┘    └──────────┘              │
    │                                                              │
    │  Benefits:                                                   │
    │  - <4% memory waste (vs 60-80% contiguous)                 │
    │  - 2-4x more concurrent requests                            │
    │  - Copy-on-write for beam search                            │
    └─────────────────────────────────────────────────────────────┘

    Interview Question:
        Q: "Walk me through Paged Attention's inference flow."
        A: (1) Request arrives, allocate logical blocks as needed
           (2) For each new token, check if logical block exists
           (3) If not, allocate physical block and add to block table
           (4) Store KV in physical block via block table mapping
           (5) At generation end, free all blocks for the request
    """

    def __init__(self, num_blocks: int = 1024, block_size: int = 16, d_model: int = 64):
        self.block_size = block_size
        self.d_model = d_model

        # Physical block pool
        self.pool = PhysicalBlockPool(num_blocks, block_size, d_model)

        # Per-request block tables
        self.request_tables: Dict[str, BlockTable] = {}

        # Per-request sequence lengths
        self.request_lengths: Dict[str, int] = {}

    def create_request(self, request_id: str) -> None:
        """Initialize a new request."""
        self.request_tables[request_id] = BlockTable()
        self.request_lengths[request_id] = 0

    def append_kv(
        self,
        request_id: str,
        k: np.ndarray,
        v: np.ndarray,
    ) -> bool:
        """
        Append a new KV pair to the request's cache.

        Args:
            request_id: Request identifier
            k: [d_model] key vector
            v: [d_model] value vector

        Returns:
            success: True if appended, False if out of memory
        """
        table = self.request_tables[request_id]
        seq_len = self.request_lengths[request_id]

        # Which logical block?
        logical_block = seq_len // self.block_size
        offset = seq_len % self.block_size

        # Allocate physical block if needed
        if table.get_physical(logical_block) is None:
            physical = self.pool.allocate()
            if physical is None:
                return False  # Out of memory!
            table.map_block(logical_block, physical)

        # Get physical block
        physical = table.get_physical(logical_block)

        # Store KV
        self.pool.k_cache[physical, offset] = k.astype(np.float16)
        self.pool.v_cache[physical, offset] = v.astype(np.float16)

        self.request_lengths[request_id] += 1
        return True

    def get_kv(
        self,
        request_id: str,
        position: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get KV at a specific position."""
        table = self.request_tables[request_id]
        logical_block = position // self.block_size
        offset = position % self.block_size

        physical = table.get_physical(logical_block)
        if physical is None:
            return None, None

        return self.pool.k_cache[physical, offset], self.pool.v_cache[physical, offset]

    def free_request(self, request_id: str) -> None:
        """Free all blocks for a request."""
        if request_id not in self.request_tables:
            return

        table = self.request_tables[request_id]
        for entry in table.entries.values():
            entry.ref_count -= 1
            if entry.ref_count <= 0:
                self.pool.free(entry.physical_block)

        del self.request_tables[request_id]
        del self.request_lengths[request_id]

    def get_stats(self) -> Dict:
        """Get overall statistics."""
        return {
            "active_requests": len(self.request_tables),
            "pool_usage": self.pool.get_usage(),
            "avg_seq_length": (
                np.mean(list(self.request_lengths.values()))
                if self.request_lengths else 0
            ),
        }


################################################################################
# SECTION 4: TESTING AND DEMONSTRATION
################################################################################

def demonstrate_paged_attention():
    """Comprehensive Paged Attention demonstration."""
    print("=" * 70)
    print("PAGED ATTENTION DEMONSTRATION")
    print("=" * 70)

    engine = PagedAttentionEngine(num_blocks=64, block_size=16, d_model=64)

    # === Demo 1: Single Request ===
    print("\n--- Demo 1: Single Request ---")
    engine.create_request("req_1")
    for i in range(50):
        k = np.random.randn(64) * 0.1
        v = np.random.randn(64) * 0.1
        success = engine.append_kv("req_1", k, v)
        if not success:
            print(f"Out of memory at position {i}")
            break

    stats = engine.get_stats()
    print(f"Sequence length: {engine.request_lengths['req_1']}")
    print(f"Blocks allocated: {stats['pool_usage']['allocated']}")
    print(f"Utilization: {stats['pool_usage']['utilization']:.1%}")

    # === Demo 2: Multiple Requests ===
    print("\n--- Demo 2: Multiple Concurrent Requests ---")
    for req_id in ["req_2", "req_3", "req_4"]:
        engine.create_request(req_id)
        for i in range(30):
            k = np.random.randn(64) * 0.1
            v = np.random.randn(64) * 0.1
            engine.append_kv(req_id, k, v)

    stats = engine.get_stats()
    print(f"Active requests: {stats['active_requests']}")
    print(f"Total blocks used: {stats['pool_usage']['allocated']}")
    print(f"Peak usage: {stats['pool_usage']['peak_usage']}")

    # === Demo 3: Memory Efficiency ===
    print("\n--- Demo 3: Memory Efficiency Comparison ---")
    print(f"{'Allocation':<25} {'Waste':<15} {'Max Requests':<15}")
    print("-" * 55)
    print(f"{'Contiguous (traditional)':<25} {'60-80%':<15} {'~100':<15}")
    print(f"{'Paged Attention':<25} {'<4%':<15} {'~400':<15}")

    # === Demo 4: Block Table ===
    print("\n--- Demo 4: Block Table ---")
    table = engine.request_tables["req_1"]
    print(f"Mapped blocks: {len(table.entries)}")
    for logical, entry in list(table.entries.items())[:5]:
        print(f"  Logical {logical} -> Physical {entry.physical_block} (refs: {entry.ref_count})")

    # === Demo 5: Cleanup ===
    print("\n--- Demo 5: Cleanup ---")
    for req_id in ["req_1", "req_2", "req_3", "req_4"]:
        engine.free_request(req_id)
    stats = engine.get_stats()
    print(f"Active requests after cleanup: {stats['active_requests']}")
    print(f"Free blocks: {stats['pool_usage']['free']}")

    print("\n" + "=" * 70)
    print("All Paged Attention demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_paged_attention()


################################################################################
# REFERENCES
################################################################################

# [1] Kwon, W., et al. (2023). Efficient Memory Management for Large Language
#     Model Serving with PagedAttention. arXiv:2309.06180. (vLLM)
#
# [2] Zheng, L., et al. (2023). S-LoRA: Serving Thousands of Concurrent
#     LoRA Adapters. arXiv:2311.03285.

################################################################################
