"""
################################################################################
CRYSTAL GENERATION — MATERIAL DISCOVERY WITH AI
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Crystal Generation?
    Using AI to discover and design new crystalline materials with
    desired properties. Crystals have periodic atomic structures
    that determine their electronic, mechanical, and thermal properties.

Why does it matter?
    New materials enable:
    - Better batteries (higher energy density)
    - Efficient solar cells (better light absorption)
    - Stronger structures (lighter, more durable)
    - Superconductors (zero resistance)

    Traditional material discovery:
    - Trial and error (slow, expensive)
    - Limited by human intuition
    - 200,000 known materials, infinite possibilities

    AI-accelerated discovery:
    - Predict properties before synthesis
    - Generate novel crystal structures
    - Explore vast chemical space efficiently

Key Innovations:
    - GNoME (Google DeepMind, 2023): Discovered 2.2M new materials
    - MACE: Fast molecular dynamics
    - Crystal Diffusion Variational Autoencoder (CDVAE)

Crystal Structure:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Crystal Structure                                │
    │                                                                  │
    │  Unit Cell: Repeating unit of the crystal                      │
    │  - Lattice vectors: a, b, c (define cell shape)                │
    │  - Atomic positions: (x, y, z) within the cell                 │
    │  - Atom types: Which elements are at each position             │
    │                                                                  │
    │  Properties depend on:                                         │
    │  - Composition (which elements)                                │
    │  - Structure (how atoms are arranged)                          │
    │  - Bonding (how atoms connect)                                 │
    │                                                                  │
    │  Example: NaCl (table salt)                                    │
    │  - Na and Cl atoms in alternating positions                    │
    │  - Cubic unit cell                                             │
    │  - Ionic bonding                                               │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "How do you represent crystals for neural networks?"
       As periodic graphs: atoms are nodes, bonds are edges,
       with periodic boundary conditions. Also include lattice
       vectors to capture the periodicity.

    2. "What is GNoME?"
       Google DeepMind's system that discovered 2.2 million new
       crystal structures, of which 380,000 are stable. This
       expanded known stable materials by an order of magnitude.

    3. "How do you generate valid crystal structures?"
       (a) Generate lattice vectors and atomic positions,
       (b) Ensure physical validity (no overlapping atoms),
       (c) Verify stability with energy calculations.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: CRYSTAL STRUCTURE
################################################################################

class CrystalStructure:
    """
    Crystal Structure Representation
    ==================================

    Represents a crystalline material with:
    - Lattice vectors (unit cell shape)
    - Atomic positions (within the unit cell)
    - Atom types (elements)

    The lattice vectors define the periodicity:
    - a, b, c: three vectors that tile space
    - α, β, γ: angles between vectors

    Interview Question:
        "What are lattice vectors?"
        Three vectors that define the unit cell of a crystal.
        The entire crystal is built by repeating this unit cell
        in all three dimensions. They determine the crystal system
        (cubic, hexagonal, etc.).
    """

    # Common elements for crystals
    ELEMENTS = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
        'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'
    ]

    def __init__(self, num_atoms: int, lattice: np.ndarray):
        """
        Args:
            num_atoms: Number of atoms in unit cell
            lattice: Lattice vectors (3, 3) — rows are a, b, c
        """
        self.num_atoms = num_atoms
        self.lattice = lattice  # (3, 3)

        # Fractional coordinates (0 to 1)
        self.frac_coords = np.random.rand(num_atoms, 3)

        # Atom types (integer indices)
        self.atom_types = np.random.randint(0, len(self.ELEMENTS), num_atoms)

    @property
    def cart_coords(self) -> np.ndarray:
        """Convert fractional to Cartesian coordinates."""
        return self.frac_coords @ self.lattice

    def compute_distances(self) -> np.ndarray:
        """
        Compute pairwise distances between atoms.

        Uses periodic boundary conditions.
        """
        distances = np.zeros((self.num_atoms, self.num_atoms))

        for i in range(self.num_atoms):
            for j in range(i + 1, self.num_atoms):
                # Fractional difference
                diff = self.frac_coords[i] - self.frac_coords[j]

                # Apply periodic boundary conditions
                diff = diff - np.round(diff)

                # Convert to Cartesian
                diff_cart = diff @ self.lattice

                # Distance
                d = np.linalg.norm(diff_cart)
                distances[i, j] = d
                distances[j, i] = d

        return distances

    def to_graph(self, cutoff: float = 5.0) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Convert crystal to graph representation.

        Atoms within cutoff distance are connected.

        Args:
            cutoff: Maximum bond distance (Angstroms)

        Returns:
            (atom_features, adjacency_list)
        """
        distances = self.compute_distances()

        # Build adjacency
        neighbors = [[] for _ in range(self.num_atoms)]
        for i in range(self.num_atoms):
            for j in range(self.num_atoms):
                if i != j and distances[i, j] < cutoff:
                    neighbors[i].append(j)

        # Atom features (one-hot element type)
        features = np.zeros((self.num_atoms, len(self.ELEMENTS)))
        for i in range(self.num_atoms):
            features[i, self.atom_types[i]] = 1.0

        return features, neighbors


################################################################################
# SECTION 2: CRYSTAL GNN
################################################################################

class CrystalGNNLayer:
    """
    Crystal GNN Layer
    ==================

    Message passing layer for crystal structures.

    Handles periodic boundary conditions by using fractional coordinates
    and minimum image convention for distances.
    """

    def __init__(self, atom_dim: int, hidden_dim: int):
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim

        # Message network
        self.msg_W = np.random.randn(atom_dim * 2 + 1, hidden_dim) * 0.02

        # Update network
        self.update_W = np.random.randn(atom_dim + hidden_dim, atom_dim) * 0.02

    def forward(
        self,
        features: np.ndarray,
        neighbors: List[List[int]],
        distances: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass through crystal GNN layer.

        Args:
            features: Atom features (num_atoms, atom_dim)
            neighbors: Adjacency list
            distances: Pairwise distances

        Returns:
            Updated atom features
        """
        num_atoms = features.shape[0]
        new_features = np.zeros_like(features)

        for i in range(num_atoms):
            messages = []

            for j in neighbors[i]:
                # Distance feature
                dist = distances[i, j]

                # Message
                msg_input = np.concatenate([features[i], features[j], [dist]])
                msg = np.tanh(self.msg_W @ msg_input)
                messages.append(msg)

            # Aggregate
            if messages:
                agg = np.mean(messages, axis=0)
            else:
                agg = np.zeros(self.hidden_dim)

            # Update
            update_input = np.concatenate([features[i], agg])
            new_features[i] = np.tanh(self.update_W @ update_input)

        return new_features


class CrystalGNN:
    """
    Crystal GNN for Property Prediction
    ======================================

    Predicts material properties from crystal structure.

    Architecture:
    1. Multiple crystal GNN layers
    2. Readout: aggregate atom features
    3. MLP: predict property

    Interview Questions:
        1. "What properties can crystal GNNs predict?"
           Formation energy, band gap, bulk modulus, shear modulus,
           thermal conductivity, and more. Any property that depends
           on crystal structure.

        2. "How accurate are these predictions?"
           For formation energy: MAE ~ 0.05 eV/atom (very accurate).
           For band gap: MAE ~ 0.3 eV (good but not perfect).
    """

    def __init__(self, atom_dim: int, hidden_dim: int, num_layers: int = 3):
        self.atom_dim = atom_dim
        self.num_layers = num_layers

        # GNN layers
        self.layers = [
            CrystalGNNLayer(atom_dim, hidden_dim)
            for _ in range(num_layers)
        ]

        # Readout
        self.readout_W = np.random.randn(atom_dim, hidden_dim) * 0.02
        self.predict_W = np.random.randn(hidden_dim, 1) * 0.02

    def forward(
        self,
        features: np.ndarray,
        neighbors: List[List[int]],
        distances: np.ndarray
    ) -> float:
        """
        Predict crystal property.

        Args:
            features: Atom features
            neighbors: Adjacency list
            distances: Pairwise distances

        Returns:
            Predicted property value
        """
        h = features

        for layer in self.layers:
            h = layer.forward(h, neighbors, distances)

        # Readout
        mol_repr = np.mean(h, axis=0)
        hidden = np.tanh(self.readout_W @ mol_repr)
        prediction = float(hidden @ self.predict_W)

        return prediction


################################################################################
# SECTION 3: MATERIAL DISCOVERY
################################################################################

class MaterialDiscovery:
    """
    Material Discovery System
    ===========================

    Uses GNNs to discover new materials with desired properties.

    Workflow:
    1. Generate candidate crystal structures
    2. Predict properties with GNN
    3. Filter by property criteria
    4. Verify stability
    5. Suggest synthesis targets

    Based on GNoME (Google DeepMind, 2023):
    - Discovered 2.2 million new crystal structures
    - 380,000 predicted to be stable
    - Expanded known materials by 10x

    Interview Questions:
        1. "How does GNoME work?"
           Uses GNNs to predict formation energy and stability.
           Generates candidates through structural modifications
           of known materials, then filters by predicted stability.

        2. "What makes a material 'stable'?"
           A material is stable if it has the lowest energy for its
           composition. This means it won't spontaneously decompose
           into other materials. Stability is measured by formation
           energy and energy above convex hull.
    """

    def __init__(self, atom_dim: int = 30, hidden_dim: int = 64):
        self.gnn = CrystalGNN(atom_dim, hidden_dim, num_layers=3)

    def generate_random_crystal(
        self,
        num_atoms: int = 5,
        lattice_param: float = 5.0
    ) -> CrystalStructure:
        """Generate a random crystal structure."""
        # Random lattice (cubic with small perturbation)
        lattice = np.eye(3) * lattice_param
        lattice += np.random.randn(3, 3) * 0.1

        return CrystalStructure(num_atoms, lattice)

    def predict_formation_energy(self, crystal: CrystalStructure) -> float:
        """Predict formation energy (eV/atom)."""
        features, neighbors = crystal.to_graph(cutoff=5.0)
        distances = crystal.compute_distances()
        return self.gnn.forward(features, neighbors, distances)

    def screen_materials(
        self,
        num_candidates: int = 100,
        energy_threshold: float = -0.5
    ) -> List[Tuple[CrystalStructure, float]]:
        """
        Screen candidate materials.

        Args:
            num_candidates: Number of candidates to generate
            energy_threshold: Maximum formation energy for stability

        Returns:
            List of (crystal, energy) for stable candidates
        """
        stable = []

        for _ in range(num_candidates):
            crystal = self.generate_random_crystal()
            energy = self.predict_formation_energy(crystal)

            if energy < energy_threshold:
                stable.append((crystal, energy))

        return stable


################################################################################
# SECTION 4: TESTING & DEMONSTRATION
################################################################################

def demonstrate_crystal_generation():
    """Demonstrate crystal generation and material discovery."""
    print("=" * 70)
    print("CRYSTAL GENERATION & MATERIAL DISCOVERY")
    print("=" * 70)

    # Create a crystal structure
    print("\n--- Creating Crystal Structure ---")
    lattice = np.eye(3) * 5.0  # 5 Angstrom cubic cell
    crystal = CrystalStructure(num_atoms=4, lattice=lattice)

    print(f"  Atoms: {crystal.num_atoms}")
    print(f"  Lattice:\n{crystal.lattice}")
    print(f"  Atom types: {[CrystalStructure.ELEMENTS[t] for t in crystal.atom_types]}")

    # Compute distances
    print("\n--- Pairwise Distances ---")
    distances = crystal.compute_distances()
    print(f"  Distance matrix shape: {distances.shape}")
    print(f"  Min distance: {distances[distances > 0].min():.2f} Å")

    # Convert to graph
    print("\n--- Graph Representation ---")
    features, neighbors = crystal.to_graph(cutoff=4.0)
    print(f"  Feature shape: {features.shape}")
    print(f"  Avg neighbors: {np.mean([len(n) for n in neighbors]):.1f}")

    # Crystal GNN
    print("\n--- Crystal GNN ---")
    gnn = CrystalGNN(atom_dim=30, hidden_dim=32, num_layers=2)

    # Predict property
    energy = gnn.forward(features, neighbors, distances)
    print(f"  Predicted energy: {energy:.4f}")

    # Material discovery
    print("\n--- Material Discovery ---")
    discovery = MaterialDiscovery(atom_dim=30, hidden_dim=32)

    # Screen materials
    stable = discovery.screen_materials(num_candidates=50, energy_threshold=0.0)
    print(f"  Generated 50 candidates")
    print(f"  Stable materials found: {len(stable)}")

    if stable:
        print("\n  Top 3 most stable:")
        sorted_stable = sorted(stable, key=lambda x: x[1])
        for i, (crystal, energy) in enumerate(sorted_stable[:3]):
            elements = [CrystalStructure.ELEMENTS[t] for t in crystal.atom_types]
            print(f"    {i+1}. Energy: {energy:.4f}, Atoms: {elements}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: GNNs can predict crystal properties and discover new materials!")
    print("GNoME discovered 2.2M new materials using this approach.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_crystal_generation()
