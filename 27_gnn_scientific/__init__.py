"""
################################################################################
GRAPH NEURAL NETWORKS & SCIENTIFIC AI — LEARNING ON GRAPHS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Graph Neural Networks?
    Neural networks that operate on graph-structured data — data where
    entities (nodes) are connected by relationships (edges).

    Many real-world systems are naturally graphs:
    - Molecules: atoms (nodes) connected by bonds (edges)
    - Social networks: people connected by friendships
    - Knowledge bases: entities connected by relations
    - Physical systems: particles connected by forces

Why does it matter?
    Traditional neural networks assume grid-like data (images, sequences).
    GNNs handle arbitrary graph structures, enabling:
    - Drug discovery: Predict molecular properties
    - Material science: Design new materials
    - Physics simulation: Model particle interactions
    - Recommendation: Model user-item interactions

Key Architectures:
    1. GCN (Graph Convolutional Network): Aggregate neighbor info
    2. GAT (Graph Attention Network): Weighted neighbor aggregation
    3. GraphSAGE: Sample and aggregate neighbors
    4. MPNN (Message Passing Neural Network): General framework

Key Applications:
    - AlphaFold 2/3: Protein structure prediction
    - GNoME: Novel material discovery (Google DeepMind)
    - DiffDock: Molecular docking
    - MACE: Fast molecular dynamics

################################################################################
"""

from .gnn_basics import GCN, GAT, GraphSAGE, MessagePassing
from .molecular_ai import MolecularGNN, PropertyPrediction, MolecularGeneration
from .geometric_deep_learning import SE3Equivariant, EquivariantLayer
from .crystal_generation import CrystalGNN, MaterialDiscovery
