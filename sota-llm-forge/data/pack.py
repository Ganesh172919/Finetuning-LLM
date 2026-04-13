"""
################################################################################
SEQUENCE PACKING — Document-Boundary-Aware Packing for Efficient Training
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Sequence Packing?
    Sequence packing concatenates multiple documents into a single
    fixed-length training sequence, maximizing GPU utilization by
    eliminating padding waste. However, attention must NOT leak across
    document boundaries — each document should only attend to itself.

Why does it matter?
    Without packing, shorter documents are padded to max_seq_len,
    wasting 30-70% of compute on padding tokens. Packing eliminates
    this waste by filling sequences to capacity. But naive packing
    without boundary-aware attention masks causes the model to learn
    spurious cross-document correlations, degrading quality.

    Common bug: forgetting the attention mask → cross-document
    contamination. We test this explicitly.

How does it work?
    1. Tokenize all documents
    2. Sort by length (for efficient bin-packing)
    3. Greedily pack into sequences of max_seq_len tokens
    4. Build attention mask: tokens can attend within their document,
       not across packed documents
    5. Track document boundaries for loss masking
    6. Only compute loss on non-padding tokens

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Document A (100 tokens)  Document B (150 tokens)                 │
    │ Document C (80 tokens)   Document D (200 tokens)                 │
    │ Document E (120 tokens)                                           │
    │        ↓                                                          │
    │ ┌──────────────────────────────────────────────────────────────┐ │
    │ │ Bin-Packer (First-Fit Decreasing)                            │ │
    │ │   Bin 1: [A(100) + B(150) + C(80)] = 330 tokens             │ │
    │ │   Bin 2: [D(200) + E(120)] = 320 tokens                      │ │
    │ └──────────────────────────────────────────────────────────────┘ │
    │        ↓                                                          │
    │ ┌──────────────────────────────────────────────────────────────┐ │
    │ │ Attention Mask Builder                                        │ │
    │ │   Block-diagonal: each document only attends to itself       │ │
    │ │   ┌─────────────────────────────────┐                        │ │
    │ │   │ A: 11110000...                  │  A attends to A only   │ │
    │ │   │ B: 0000111111000...             │  B attends to B only   │ │
    │ │   │ C: 00000000001111000            │  C attends to C only   │ │
    │ │   │ Padding: 000000000000000...     │  Padding: masked       │ │
    │ │   └─────────────────────────────────┘                        │ │
    │ └──────────────────────────────────────────────────────────────┘ │
    │        ↓                                                          │
    │ Packed Sequence: [A tokens][B tokens][C tokens][PAD tokens]       │
    │ Attention Mask:  Block-diagonal (document-aware)                   │
    │ Labels:          Shifted input_ids (loss masked at boundaries)     │
    └─────────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2019: T5 introduces packing with "packing" input feature
    - 2021: GPT-3 uses document-padding (no packing) — massive waste
    - 2022: Packing becomes standard in Hugging Face Transformers
    - 2023: Multiple papers identify cross-document attention leakage
      as a quality issue in packed training
    - 2024: Document-boundary-aware attention masks become standard
    - 2025: Efficient packing algorithms (bin-packing) reduce padding
      to <5% across the full corpus
    - 2026: Packing is table stakes; boundary-aware masking is mandatory

INTERVIEW QUESTIONS:
    1. "What happens if you pack without document-boundary attention masks?"
       Answer: The model learns spurious correlations across document
       boundaries. For example, if document A discusses climate change and
       document B discusses cooking, the model learns that "climate" is
       related to "cooking" because they appear in the same sequence.
       This cross-contamination degrades downstream performance,
       especially on tasks requiring precise context understanding.

    2. "Why use first-fit decreasing for bin-packing?"
       Answer: First-fit decreasing (FFD) sorts documents by length in
       descending order, then greedily places each document in the first
       bin with space. This produces near-optimal packing (within 11/9
       of optimal for bin-packing). The key insight: large documents
       first leaves flexible small documents to fill gaps, minimizing
       wasted space.

    3. "How do you handle the last token in a packed sequence for
       next-token prediction?"
       Answer: The labels are the input_ids shifted right by one position.
       At document boundaries, the last token of document A should NOT
       predict the first token of document B. We mask the loss at these
       boundaries by setting the label to -100 (ignored by cross-entropy
       loss). This ensures each document's loss is computed independently.

################################################################################
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


################################################################################
# SECTION 1: CONFIGURATION
################################################################################


@dataclass
class PackingConfig:
    """
    Configuration for sequence packing.
    ====================================

    All parameters are explicit config fields. No magic numbers.
    """

    max_seq_len: int = 2048  # Maximum sequence length
    pad_token_id: int = 0  # Token ID used for padding
    eos_token_id: int = 2  # End-of-sequence token ID
    ignore_index: int = -100  # Label value to ignore in loss computation
    min_fill_ratio: float = 0.9  # Minimum fill ratio before closing a bin
    sort_by_length: bool = True  # Sort documents by length before packing


################################################################################
# SECTION 2: DOCUMENT REPRESENTATION
################################################################################


@dataclass
class DocumentTokens:
    """
    Represents a tokenized document with metadata.
    ==============================================

    Attributes:
        token_ids: List of token IDs
        doc_id: Unique document identifier
        metadata: Optional metadata dict
    """

    token_ids: List[int]
    doc_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Return document length in tokens."""
        return len(self.token_ids)


@dataclass
class PackedSequence:
    """
    Represents a packed sequence ready for training.
    ================================================

    Attributes:
        input_ids: Token IDs for input (shape: [seq_len])
        attention_mask: Document-boundary-aware attention mask (shape: [seq_len, seq_len])
        labels: Target token IDs with boundary masking (shape: [seq_len])
        doc_boundaries: List of (start, end) tuples for each document
        doc_ids: List of document IDs in this packed sequence
    """

    input_ids: np.ndarray
    attention_mask: np.ndarray
    labels: np.ndarray
    doc_boundaries: List[Tuple[int, int]]
    doc_ids: List[int]


################################################################################
# SECTION 3: BIN-PACKING ALGORITHM
################################################################################


class BinPacker:
    """
    Bin Packer
    ==========

    Greedy bin-packing algorithm for packing documents into fixed-length
    sequences. Implements First-Fit Decreasing (FFD) heuristic.

    Algorithm:
        1. Sort documents by length in descending order
        2. For each document, find the first bin with enough remaining space
        3. If no bin fits, create a new bin
        4. Continue until all documents are packed

    WHY this matters:
        Efficient packing reduces padding waste from 30-70% to <5%,
        directly translating to faster training and lower cost.

    Interview Question:
        "What is the approximation ratio of first-fit decreasing?"
        Answer: FFD produces at most 11/9 * OPT + 6/9 bins, where OPT
        is the optimal number of bins. In practice, for sequences of
        varying lengths typical in NLP, FFD achieves near-optimal packing
        with minimal wasted space.
    """

    def __init__(self, config: Optional[PackingConfig] = None):
        """
        Initialize the bin packer.

        Args:
            config: Packing configuration
        """
        self.config = config or PackingConfig()
        self.max_bin_size = self.config.max_seq_len
        self.min_fill = int(self.max_bin_size * self.config.min_fill_ratio)

    def pack(
        self,
        documents: List[DocumentTokens],
    ) -> List[List[DocumentTokens]]:
        """
        Pack documents into bins using First-Fit Decreasing.

        Args:
            documents: List of tokenized documents

        Returns:
            List of bins, where each bin is a list of documents

        Explanation:
            1. Sort documents by length (descending) for FFD
            2. For each document, try to fit in existing bins
            3. If no bin has space, create new bin
            4. Each bin's total length <= max_seq_len
        """
        if not documents:
            return []

        # Filter out documents longer than max_bin_size
        valid_docs = [d for d in documents if d.length <= self.max_bin_size]
        oversized = [d for d in documents if d.length > self.max_bin_size]

        if oversized:
            logger.warning(
                f"Discarded {len(oversized)} documents longer than {self.max_bin_size} tokens"
            )

        # Sort by length descending (FFD heuristic)
        if self.config.sort_by_length:
            valid_docs.sort(key=lambda d: d.length, reverse=True)

        # First-fit decreasing bin-packing
        bins: List[List[DocumentTokens]] = []
        bin_remaining: List[int] = []

        for doc in valid_docs:
            placed = False
            for i, remaining in enumerate(bin_remaining):
                if doc.length <= remaining:
                    bins[i].append(doc)
                    bin_remaining[i] -= doc.length
                    placed = True
                    break

            if not placed:
                bins.append([doc])
                bin_remaining.append(self.max_bin_size - doc.length)

        # Log packing statistics
        total_tokens = sum(d.length for d in valid_docs)
        total_bins = len(bins)
        avg_fill = total_tokens / (total_bins * self.max_bin_size) if total_bins > 0 else 0
        logger.info(
            f"Bin packing: {len(valid_docs)} docs -> {total_bins} bins "
            f"(avg fill: {avg_fill:.1%})"
        )

        return bins


################################################################################
# SECTION 4: ATTENTION MASK BUILDER
################################################################################


def build_document_boundary_mask(
    doc_boundaries: List[Tuple[int, int]],
    seq_len: int,
) -> np.ndarray:
    """
    Build a document-boundary-aware attention mask.

    Args:
        doc_boundaries: List of (start, end) tuples for each document
        seq_len: Total sequence length

    Returns:
        Boolean attention mask of shape (seq_len, seq_len)
        True = can attend, False = masked (cannot attend)

    Explanation:
        Creates a block-diagonal attention mask where each document
        can only attend to tokens within itself. Padding tokens
        (outside any document boundary) cannot attend to anything
        and nothing can attend to them.

        For document with boundaries [start, end):
        - mask[start:end, start:end] = True (self-attention within doc)
        - All other entries = False (no cross-document attention)

    Example:
        >>> boundaries = [(0, 5), (5, 10)]
        >>> mask = build_document_boundary_mask(boundaries, 12)
        >>> mask[0, 3]  # Within doc 0 — can attend
        True
        >>> mask[0, 6]  # Cross-document — cannot attend
        False
        >>> mask[10, 0]  # Padding — cannot attend
        False
    """
    mask = np.zeros((seq_len, seq_len), dtype=bool)

    for start, end in doc_boundaries:
        # Ensure boundaries are within sequence length
        start = max(0, min(start, seq_len))
        end = max(0, min(end, seq_len))

        # Document can attend to itself
        mask[start:end, start:end] = True

    return mask


def build_causal_document_mask(
    doc_boundaries: List[Tuple[int, int]],
    seq_len: int,
) -> np.ndarray:
    """
    Build a causal (autoregressive) document-boundary-aware attention mask.

    Args:
        doc_boundaries: List of (start, end) tuples for each document
        seq_len: Total sequence length

    Returns:
        Boolean attention mask of shape (seq_len, seq_len)
        True = can attend, False = masked

    Explanation:
        Combines causal masking (each token only attends to previous tokens)
        with document boundary masking (no cross-document attention).

        For token at position i in document [start, end):
        - Can attend to positions [start, i] (causal within document)
        - Cannot attend to positions < start (other documents)
        - Cannot attend to positions > i (future tokens)

    Example:
        >>> boundaries = [(0, 5), (5, 10)]
        >>> mask = build_causal_document_mask(boundaries, 10)
        >>> mask[3, 1]  # Causal within doc 0 — can attend
        True
        >>> mask[3, 4]  # Future within doc 0 — cannot attend
        False
        >>> mask[6, 3]  # Cross-document — cannot attend
        False
    """
    mask = np.zeros((seq_len, seq_len), dtype=bool)

    for start, end in doc_boundaries:
        start = max(0, min(start, seq_len))
        end = max(0, min(end, seq_len))

        # Causal mask within document: token i attends to [start, i]
        for i in range(start, end):
            mask[i, start : i + 1] = True

    return mask


################################################################################
# SECTION 5: LABEL BUILDER
################################################################################


def build_labels(
    input_ids: np.ndarray,
    doc_boundaries: List[Tuple[int, int]],
    ignore_index: int = -100,
) -> np.ndarray:
    """
    Build labels for next-token prediction with boundary masking.

    Args:
        input_ids: Token IDs of shape (seq_len,)
        doc_boundaries: List of (start, end) tuples for each document
        ignore_index: Label value to ignore in loss computation

    Returns:
        Labels array of shape (seq_len,)

    Explanation:
        Labels are input_ids shifted right by one position (next-token
        prediction). At document boundaries, we set the label to
        ignore_index so the model doesn't try to predict the first
        token of the next document from the last token of the previous
        document.

        For document [start, end):
        - labels[start:end-1] = input_ids[start+1:end] (standard shift)
        - labels[end-1] = ignore_index (last token has no next token)
        - All padding positions = ignore_index
    """
    seq_len = len(input_ids)
    labels = np.full(seq_len, ignore_index, dtype=np.int64)

    for start, end in doc_boundaries:
        start = max(0, min(start, seq_len))
        end = max(0, min(end, seq_len))

        if end - start < 2:
            # Document too short for next-token prediction
            continue

        # Shift: label[i] = input_ids[i+1] for i in [start, end-2]
        labels[start : end - 1] = input_ids[start + 1 : end]
        # Last token has no next token -> ignore
        labels[end - 1] = ignore_index

    return labels


################################################################################
# SECTION 6: SEQUENCE PACKER (ORCHESTRATOR)
################################################################################


class SequencePacker:
    """
    Sequence Packer
    ===============

    Orchestrates document packing with boundary-aware attention masking.

    Pipeline:
        1. Tokenize all documents
        2. Greedily pack into sequences of max_seq_len tokens
        3. Build attention mask: tokens can attend within their document,
           not across packed documents
        4. Track document boundaries for loss masking
        5. Build labels with boundary masking

    WHY this matters:
        Packing is essential for training efficiency. Without it, GPU
        utilization drops to 30-70% due to padding. With proper
        boundary-aware masking, packing achieves near-100% utilization
        without quality loss.

    Interview Question:
        "How do you verify that packing doesn't leak information across
        document boundaries?"
        Answer: We test explicitly: create a packed sequence with known
        documents, run a forward pass, then verify that the gradient
        of tokens in document A with respect to tokens in document B
        is zero. We also unit test the attention mask: it must be
        block-diagonal (not full) when multiple documents are packed.
    """

    def __init__(self, config: Optional[PackingConfig] = None):
        """
        Initialize the sequence packer.

        Args:
            config: Packing configuration
        """
        self.config = config or PackingConfig()
        self.bin_packer = BinPacker(self.config)

        self.stats = {
            "total_documents": 0,
            "total_packed_sequences": 0,
            "total_tokens": 0,
            "padding_tokens": 0,
            "fill_ratio": 0.0,
        }

    def pack(
        self,
        documents: List[DocumentTokens],
        use_causal_mask: bool = True,
    ) -> List[PackedSequence]:
        """
        Pack documents into fixed-length training sequences.

        Args:
            documents: List of tokenized documents
            use_causal_mask: If True, use causal (autoregressive) attention mask.
                           If True, use bidirectional within-document attention.

        Returns:
            List of PackedSequence objects ready for training

        Explanation:
            1. Bin-pack documents into groups that fit in max_seq_len
            2. For each group, concatenate tokens and track boundaries
            3. Build document-boundary-aware attention mask
            4. Build labels with boundary masking
            5. Pad to max_seq_len if needed
        """
        self.stats["total_documents"] = len(documents)

        # Step 1: Bin-pack
        bins = self.bin_packer.pack(documents)
        self.stats["total_packed_sequences"] = len(bins)

        packed_sequences = []

        for bin_docs in bins:
            # Step 2: Concatenate tokens and track boundaries
            all_token_ids = []
            doc_boundaries = []
            doc_ids = []
            current_pos = 0

            for doc in bin_docs:
                start = current_pos
                all_token_ids.extend(doc.token_ids)
                current_pos += doc.length
                end = current_pos
                doc_boundaries.append((start, end))
                doc_ids.append(doc.doc_id)

            # Pad to max_seq_len
            actual_len = len(all_token_ids)
            padding_len = self.config.max_seq_len - actual_len
            all_token_ids.extend([self.config.pad_token_id] * padding_len)

            self.stats["total_tokens"] += actual_len
            self.stats["padding_tokens"] += padding_len

            # Step 3: Build attention mask
            if use_causal_mask:
                attention_mask = build_causal_document_mask(
                    doc_boundaries, self.config.max_seq_len
                )
            else:
                attention_mask = build_document_boundary_mask(
                    doc_boundaries, self.config.max_seq_len
                )

            # Step 4: Build labels
            input_ids = np.array(all_token_ids, dtype=np.int64)
            labels = build_labels(
                input_ids, doc_boundaries, self.config.ignore_index
            )

            packed_sequences.append(PackedSequence(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                doc_boundaries=doc_boundaries,
                doc_ids=doc_ids,
            ))

        # Compute fill ratio
        total_capacity = len(bins) * self.config.max_seq_len
        self.stats["fill_ratio"] = (
            self.stats["total_tokens"] / total_capacity if total_capacity > 0 else 0.0
        )

        logger.info(
            f"Packed {self.stats['total_documents']} docs into "
            f"{self.stats['total_packed_sequences']} sequences "
            f"(fill ratio: {self.stats['fill_ratio']:.1%})"
        )

        return packed_sequences

    def unpack(self, packed: PackedSequence) -> List[Dict[str, Any]]:
        """
        Unpack a packed sequence back into individual documents.

        Args:
            packed: A PackedSequence object

        Returns:
            List of dicts with keys: doc_id, input_ids, labels

        Explanation:
            Useful for debugging and verification. Extracts each
            document's tokens and labels from the packed representation.
        """
        documents = []
        for i, (start, end) in enumerate(packed.doc_boundaries):
            doc_input_ids = packed.input_ids[start:end]
            doc_labels = packed.labels[start:end]
            documents.append({
                "doc_id": packed.doc_ids[i],
                "input_ids": doc_input_ids,
                "labels": doc_labels,
            })
        return documents

    def verify_no_leakage(
        self,
        packed: PackedSequence,
        doc_a_idx: int = 0,
        doc_b_idx: int = 1,
    ) -> bool:
        """
        Verify that attention doesn't leak across document boundaries.

        Args:
            packed: A PackedSequence object
            doc_a_idx: Index of first document to check
            doc_b_idx: Index of second document to check

        Returns:
            True if no leakage detected, False otherwise

        Explanation:
            Checks that tokens in document A have zero attention to
            tokens in document B (and vice versa). This is the critical
            invariant for correct packing.
        """
        if doc_a_idx >= len(packed.doc_boundaries) or doc_b_idx >= len(packed.doc_boundaries):
            logger.warning("Document index out of range")
            return True

        start_a, end_a = packed.doc_boundaries[doc_a_idx]
        start_b, end_b = packed.doc_boundaries[doc_b_idx]

        # Check: no token in A can attend to any token in B
        mask_ab = packed.attention_mask[start_a:end_a, start_b:end_b]
        if mask_ab.any():
            logger.error(
                f"LEAKAGE DETECTED: Document {doc_a_idx} can attend to Document {doc_b_idx}"
            )
            return False

        # Check: no token in B can attend to any token in A
        mask_ba = packed.attention_mask[start_b:end_b, start_a:end_a]
        if mask_ba.any():
            logger.error(
                f"LEAKAGE DETECTED: Document {doc_b_idx} can attend to Document {doc_a_idx}"
            )
            return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Return packing statistics."""
        return dict(self.stats)


################################################################################
# SECTION 7: TESTING & DEMONSTRATION
################################################################################


def demonstrate_packer():
    """Demonstrate the sequence packing pipeline."""
    print("=" * 70)
    print("SEQUENCE PACKING DEMONSTRATION")
    print("=" * 70)

    # Create sample documents
    documents = [
        DocumentTokens(
            token_ids=list(range(10, 60)),  # 50 tokens
            doc_id=0,
            metadata={"domain": "web", "title": "AI Overview"},
        ),
        DocumentTokens(
            token_ids=list(range(100, 220)),  # 120 tokens
            doc_id=1,
            metadata={"domain": "code", "title": "Python Tutorial"},
        ),
        DocumentTokens(
            token_ids=list(range(200, 240)),  # 40 tokens
            doc_id=2,
            metadata={"domain": "math", "title": "Calculus Basics"},
        ),
        DocumentTokens(
            token_ids=list(range(300, 380)),  # 80 tokens
            doc_id=3,
            metadata={"domain": "web", "title": "History of Science"},
        ),
        DocumentTokens(
            token_ids=list(range(400, 435)),  # 35 tokens
            doc_id=4,
            metadata={"domain": "code", "title": "Data Structures"},
        ),
        DocumentTokens(
            token_ids=list(range(500, 560)),  # 60 tokens
            doc_id=5,
            metadata={"domain": "math", "title": "Linear Algebra"},
        ),
    ]

    total_tokens = sum(d.length for d in documents)
    print(f"\nInput: {len(documents)} documents, {total_tokens} total tokens")
    for doc in documents:
        print(f"  Doc {doc.doc_id}: {doc.length} tokens ({doc.metadata['title']})")

    # Pack with different sequence lengths
    for max_seq_len in [128, 256]:
        print(f"\n{'=' * 60}")
        print(f"Packing with max_seq_len = {max_seq_len}")
        print(f"{'=' * 60}")

        config = PackingConfig(max_seq_len=max_seq_len)
        packer = SequencePacker(config)

        packed = packer.pack(documents, use_causal_mask=True)

        # Print results
        stats = packer.get_stats()
        print(f"\nPacking Statistics:")
        print(f"  Total documents:    {stats['total_documents']}")
        print(f"  Packed sequences:   {stats['total_packed_sequences']}")
        print(f"  Total tokens:       {stats['total_tokens']}")
        print(f"  Padding tokens:     {stats['padding_tokens']}")
        print(f"  Fill ratio:         {stats['fill_ratio']:.1%}")

        # Print packed sequence details
        for i, seq in enumerate(packed):
            print(f"\n  Sequence {i}:")
            print(f"    Documents: {seq.doc_ids}")
            print(f"    Boundaries: {seq.doc_boundaries}")
            print(f"    Input shape: {seq.input_ids.shape}")
            print(f"    Mask shape: {seq.attention_mask.shape}")
            print(f"    Labels shape: {seq.labels.shape}")

            # Count non-ignored labels
            valid_labels = (seq.labels != config.ignore_index).sum()
            print(f"    Valid labels: {valid_labels}")

            # Verify no leakage
            if len(seq.doc_ids) >= 2:
                no_leak = packer.verify_no_leakage(seq, 0, 1)
                print(f"    No cross-doc leakage: {no_leak}")

        # Demonstrate unpacking
        print(f"\n  Unpacking first sequence:")
        unpacked = packer.unpack(packed[0])
        for doc in unpacked:
            print(f"    Doc {doc['doc_id']}: {len(doc['input_ids'])} tokens")

    # Demonstrate attention mask visualization
    print(f"\n{'=' * 60}")
    print("ATTENTION MASK VISUALIZATION (small example)")
    print(f"{'=' * 60}")

    small_docs = [
        DocumentTokens(token_ids=[1, 2, 3], doc_id=0),
        DocumentTokens(token_ids=[4, 5], doc_id=1),
    ]

    config = PackingConfig(max_seq_len=10, pad_token_id=0)
    packer = SequencePacker(config)
    packed = packer.pack(small_docs, use_causal_mask=True)

    seq = packed[0]
    print(f"\nInput IDs: {seq.input_ids}")
    print(f"Labels:    {seq.labels}")
    print(f"Boundaries: {seq.doc_boundaries}")
    print(f"\nAttention Mask (causal, boundary-aware):")
    print(f"  Doc 0: tokens [0,1,2], Doc 1: tokens [3,4], Pad: tokens [5-9]")

    # Show mask as block diagram
    mask = seq.attention_mask[:7, :7]  # Show first 7x7 for readability
    for i in range(mask.shape[0]):
        row = "".join(["1" if mask[i, j] else "0" for j in range(mask.shape[1])])
        boundary_marker = ""
        for start, end in seq.doc_boundaries:
            if i == start:
                boundary_marker = f"  <- Doc {seq.doc_ids[seq.doc_boundaries.index((start, end))]} start"
            elif i == end - 1:
                boundary_marker = f"  <- Doc {seq.doc_ids[seq.doc_boundaries.index((start, end))]} end"
        print(f"  [{i}] {row}{boundary_marker}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_packer()
