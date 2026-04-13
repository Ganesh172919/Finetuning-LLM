"""
################################################################################
SOTA LLM FORGE — OPTIMIZER STACK
################################################################################

Hybrid Muon + AdamW optimizer for the SOTA LLM training stack.

The single highest-leverage 2026 systems change vs 2023-era training.

Components:
    muon.py            — Newton-Schulz orthogonalized optimizer for 2D weights
    hybrid_optimizer.py — Muon (2D hidden) + AdamW (rest) + WSD schedule

Assignment rules:
    Muon:   attention Q/K/V/output projections, MLP weights, MoE expert weights
    AdamW:  embeddings, output head, RMSNorm gains, biases, MoE router weights

Why hybrid:
    Muon normalizes the *spectrum* of the weight update (matrix-aware).
    AdamW adapts per-element. Hybrid captures both benefits.

Cite: Jordan et al. 2024 "Muon"; scaled to trillion-param models (Kimi K2, GLM-4.5)

################################################################################
"""

from .muon import Muon, MuonConfig, newton_schulz_orthogonalize, create_muon_optimizer
from .hybrid_optimizer import (
    HybridMuonAdamW,
    HybridOptimizerConfig,
    WSDSchedule,
    GradientClipper,
    DecayType,
    create_hybrid_optimizer,
    split_mla_up_projection,
    merge_mla_blocks,
)

__all__ = [
    # Core optimizers
    "Muon",
    "HybridMuonAdamW",
    # Configuration
    "MuonConfig",
    "HybridOptimizerConfig",
    # Functions
    "newton_schulz_orthogonalize",
    "create_muon_optimizer",
    "create_hybrid_optimizer",
    # Schedule and clipping
    "WSDSchedule",
    "GradientClipper",
    "DecayType",
    # MLA utilities
    "split_mla_up_projection",
    "merge_mla_blocks",
]
