"""
################################################################################
SOTA LLM FORGE — DATA PIPELINE
################################################################################

Data curation, synthetic generation, and sequence packing for the SOTA LLM
training stack.

Components:
    curate.py    — Deduplication (exact + MinHash/LSH), quality filtering,
                   decontamination against eval benchmarks
    synthetic.py — Multi-teacher synthetic data generation with diversity checks
    pack.py      — Sequence packing with document-boundary attention masking

Key design principle (2026 lesson):
    Curation discipline, not generation volume, separates useful synthetic
    pretraining data from collapse risk.

################################################################################
"""

from .curate import (
    DataCurator,
    ExactDeduplicator,
    MinHashDeduplicator,
    QualityFilter,
    Decontaminator,
    DomainMixture,
    CurationConfig,
)
from .synthetic import (
    SyntheticDataGenerator,
    MultiTeacherGenerator,
    DiversityChecker,
    RealDataSeeder,
    QualityScorer,
    SyntheticDataConfig,
)
from .pack import (
    SequencePacker,
    BinPacker,
    DocumentTokens,
    PackedSequence,
    PackingConfig,
    build_document_boundary_mask,
    build_causal_document_mask,
    build_labels,
)

__all__ = [
    # Curate
    "DataCurator",
    "ExactDeduplicator",
    "MinHashDeduplicator",
    "QualityFilter",
    "Decontaminator",
    "DomainMixture",
    "CurationConfig",
    # Synthetic
    "SyntheticDataGenerator",
    "MultiTeacherGenerator",
    "DiversityChecker",
    "RealDataSeeder",
    "QualityScorer",
    "SyntheticDataConfig",
    # Pack
    "SequencePacker",
    "BinPacker",
    "DocumentTokens",
    "PackedSequence",
    "PackingConfig",
    "build_document_boundary_mask",
    "build_causal_document_mask",
    "build_labels",
]
