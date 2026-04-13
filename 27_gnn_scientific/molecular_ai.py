"""
################################################################################
MOLECULAR AI — DRUG DISCOVERY & MOLECULAR UNDERSTANDING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Molecular AI?
    Using AI (especially GNNs) to understand, predict, and design molecules.
    This includes:
    - Property prediction: Will this molecule dissolve in water?
    - Molecular generation: Design new drug molecules
    - Molecular docking: How does a drug bind to a protein?
    - Retrosynthesis: How to synthesize a molecule?

Why does it matter?
    Drug discovery is:
    - Extremely expensive ($2.6B per drug on average)
    - Very slow (10-15 years from discovery to market)
    - High failure rate (90% of drugs fail in trials)

    AI can accelerate this by:
    - Predicting properties before synthesis
    - Generating novel drug candidates
    - Understanding protein-drug interactions
    - Optimizing molecular structures

Key Innovations:
    - AlphaFold 2/3: Protein structure prediction
    - DiffDock: Molecular docking with diffusion
    - MolGAN: Molecular graph generation
    - SchNet: 3D molecular representations

Molecular Representation:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Molecular Graph                                  │
    │                                                                  │
    │  Atoms = Nodes (features: element, charge, hybridization)      │
    │  Bonds = Edges (features: single/double/triple, conjugated)    │
    │                                                                  │
    │      H   H                                                      │
    │       \ /                                                       │
    │    H - C - H   →  C is node, C-H bonds are edges               │
    │       / \                                                       │
    │      H   H                                                      │
    │                                                                  │
    │  GNN processes this graph to predict properties                 │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "How do you represent molecules for neural networks?"
       As graphs: atoms are nodes (with features like element type),
       bonds are edges (with features like bond type). 3D coordinates
       can also be used for geometric models.

    2. "What is AlphaFold?"
       A system that predicts protein 3D structure from amino acid
       sequence. It uses attention and structure modules to predict
       atomic coordinates with near-experimental accuracy.

    3. "How does AI help in drug discovery?"
       (a) Virtual screening: predict binding affinity,
       (b) De novo design: generate novel molecules,
       (c) ADMET prediction: predict drug properties,
       (d) Lead optimization: improve drug candidates.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: MOLECULAR GRAPH
################################################################################

class MolecularGraph:
    """
    Molecular Graph Representation
    ================================

    Represents a molecule as a graph where:
    - Nodes = atoms (with features: element, charge, etc.)
    - Edges = bonds (with features: bond type, conjugation, etc.)

    Atom features typically include:
    - Element type (one-hot encoded)
    - Formal charge
    - Hybridization (sp, sp2, sp3)
    - Aromaticity
    - Number of hydrogens

    Bond features:
    - Bond type (single, double, triple, aromatic)
    - Conjugated
    - In ring
    """

    # Element encodings (simplified)
    ELEMENTS = {
        'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4,
        'Cl': 5, 'Br': 6, 'I': 7, 'P': 8, 'Si': 9
    }

    def __init__(self, num_atoms: int, atom_dim: int = 10, bond_dim: int = 4):
        self.num_atoms = num_atoms
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim

        # Atom features
        self.atom_features = np.zeros((num_atoms, atom_dim))

        # Adjacency with bond features
        self.neighbors: List[List[int]] = [[] for _ in range(num_atoms)]
        self.bond_features: Dict[Tuple[int, int], np.ndarray] = {}

    def set_atom(self, atom_idx: int, element: str, **kwargs):
        """Set atom features."""
        if element in self.ELEMENTS:
            self.atom_features[atom_idx, self.ELEMENTS[element]] = 1.0

    def add_bond(self, atom_i: int, atom_j: int, bond_type: str = "single"):
        """Add a bond between atoms."""
        self.neighbors[atom_i].append(atom_j)
        self.neighbors[atom_j].append(atom_i)

        # Bond type encoding
        type_map = {"single": 0, "double": 1, "triple": 2, "aromatic": 3}
        bond_feat = np.zeros(self.bond_dim)
        if bond_type in type_map:
            bond_feat[type_map[bond_type]] = 1.0

        self.bond_features[(atom_i, atom_j)] = bond_feat
        self.bond_features[(atom_j, atom_i)] = bond_feat


################################################################################
# SECTION 2: MOLECULAR GNN
################################################################################

class MolecularGNNLayer:
    """
    Molecular GNN Layer
    =====================

    Message passing layer for molecular graphs.

    Unlike generic GNNs, molecular GNNs:
    - Use bond features in messages
    - Handle different atom types
    - Preserve chemical constraints

    Message function:
        m_ij = f(h_i, h_j, e_ij)

    Where e_ij is the bond feature between atoms i and j.
    """

    def __init__(self, atom_dim: int, bond_dim: int, hidden_dim: int):
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim

        # Message network
        self.msg_W = np.random.randn(atom_dim * 2 + bond_dim, hidden_dim) * 0.02

        # Update network
        self.update_W = np.random.randn(atom_dim + hidden_dim, atom_dim) * 0.02

    def forward(self, mol: MolecularGraph) -> np.ndarray:
        """
        Forward pass through molecular GNN layer.

        Args:
            mol: Molecular graph

        Returns:
            Updated atom features
        """
        num_atoms = mol.num_atoms
        new_features = np.zeros_like(mol.atom_features)

        for i in range(num_atoms):
            messages = []

            for j in mol.neighbors[i]:
                # Get bond feature
                bond = mol.bond_features.get((i, j), np.zeros(self.bond_dim))

                # Message: concatenate atom features and bond
                msg_input = np.concatenate([
                    mol.atom_features[i],
                    mol.atom_features[j],
                    bond
                ])

                # Message function
                msg = np.tanh(self.msg_W @ msg_input)
                messages.append(msg)

            # Aggregate messages
            if messages:
                agg_msg = np.mean(messages, axis=0)
            else:
                agg_msg = np.zeros(self.hidden_dim)

            # Update
            update_input = np.concatenate([mol.atom_features[i], agg_msg])
            new_features[i] = np.tanh(self.update_W @ update_input)

        return new_features


class MolecularGNN:
    """
    Molecular GNN for Property Prediction
    ========================================

    Predicts molecular properties from graph structure.

    Architecture:
    1. Multiple message passing layers
    2. Readout: aggregate atom features to molecular representation
    3. MLP: predict property from molecular representation

    Interview Questions:
        1. "How do you go from atom features to molecular property?"
           Use a READOUT function: aggregate all atom features
           (mean, sum, or attention) to get a molecular-level vector.
           Then use an MLP to predict the property.

        2. "What properties can GNNs predict?"
           Solubility, toxicity, binding affinity, drug-likeness,
           metabolic stability, and more. Any molecular property
           that depends on structure.
    """

    def __init__(self, atom_dim: int, bond_dim: int, hidden_dim: int, num_layers: int = 3):
        self.atom_dim = atom_dim
        self.num_layers = num_layers

        # Message passing layers
        self.layers = [
            MolecularGNNLayer(atom_dim, bond_dim, hidden_dim)
            for _ in range(num_layers)
        ]

        # Readout MLP
        self.readout_W = np.random.randn(atom_dim, hidden_dim) * 0.02
        self.predict_W = np.random.randn(hidden_dim, 1) * 0.02

    def forward(self, mol: MolecularGraph) -> float:
        """
        Predict molecular property.

        Args:
            mol: Molecular graph

        Returns:
            Predicted property value
        """
        # Message passing
        h = mol.atom_features
        for layer in self.layers:
            h = layer.forward(mol)

        # Readout: mean pooling over atoms
        mol_repr = np.mean(h, axis=0)

        # Predict
        hidden = np.tanh(self.readout_W @ mol_repr)
        prediction = float(hidden @ self.predict_W)

        return prediction

    def train_step(
        self,
        mol: MolecularGraph,
        target: float,
        learning_rate: float = 0.01
    ) -> float:
        """
        One training step.

        Args:
            mol: Molecular graph
            target: Target property value
            learning_rate: Learning rate

        Returns:
            Loss value
        """
        # Forward
        prediction = self.forward(mol)
        loss = (prediction - target) ** 2

        # Simplified gradient update (just update final layer)
        grad = 2 * (prediction - target)
        self.predict_W -= learning_rate * grad * np.tanh(self.readout_W @ np.mean(mol.atom_features, axis=0))

        return loss


################################################################################
# SECTION 3: PROPERTY PREDICTION
################################################################################

class PropertyPrediction:
    """
    Molecular Property Prediction
    ================================

    Predicts various molecular properties using GNNs.

    Common properties:
    - Solubility (logP)
    - Toxicity (LD50)
    - Drug-likeness (QED)
    - Binding affinity (pIC50)

    Interview Question:
        "How accurate are GNN property predictions?"
        For simple properties (solubility): R² ~ 0.8-0.9
        For complex properties (binding): R² ~ 0.5-0.7
        Accuracy depends on training data quality and quantity.
    """

    def __init__(self, atom_dim: int = 10, bond_dim: int = 4):
        self.gnn = MolecularGNN(atom_dim, bond_dim, hidden_dim=32, num_layers=3)

    def predict_solubility(self, mol: MolecularGraph) -> float:
        """Predict aqueous solubility (logP)."""
        return self.gnn.forward(mol)

    def predict_toxicity(self, mol: MolecularGraph) -> float:
        """Predict toxicity score."""
        return self.gnn.forward(mol)


################################################################################
# SECTION 4: MOLECULAR GENERATION
################################################################################

class MolecularGeneration:
    """
    Molecular Generation
    =====================

    Generates novel molecular structures with desired properties.

    Approaches:
    1. SMILES-based: Generate SMILES strings (text)
    2. Graph-based: Generate molecular graphs directly
    3. 3D-based: Generate 3D conformations

    Interview Question:
        "How do you generate valid molecules?"
        (a) Use validity checks during generation,
        (b) Train with validity as a reward signal,
        (c) Use fragment-based generation (combine valid fragments).
    """

    def __init__(self, atom_dim: int = 10, bond_dim: int = 4):
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim

    def generate_random_molecule(self, num_atoms: int = 5) -> MolecularGraph:
        """Generate a random molecular graph."""
        mol = MolecularGraph(num_atoms, self.atom_dim, self.bond_dim)

        # Set random atom types
        elements = list(MolecularGraph.ELEMENTS.keys())
        for i in range(num_atoms):
            element = elements[np.random.randint(len(elements))]
            mol.set_atom(i, element)

        # Add random bonds (ensure connected)
        for i in range(num_atoms - 1):
            mol.add_bond(i, i + 1, "single")

        # Add some extra bonds
        for _ in range(num_atoms // 2):
            i, j = np.random.randint(0, num_atoms, 2)
            if i != j and j not in mol.neighbors[i]:
                bond_type = np.random.choice(["single", "double"])
                mol.add_bond(i, j, bond_type)

        return mol


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################

def demonstrate_molecular_ai():
    """Demonstrate molecular AI capabilities."""
    print("=" * 70)
    print("MOLECULAR AI")
    print("=" * 70)

    # Create a molecule (simplified methane: CH4)
    print("\n--- Creating Molecular Graph (Methane) ---")
    methane = MolecularGraph(5, atom_dim=10, bond_dim=4)
    methane.set_atom(0, 'C')
    for i in range(1, 5):
        methane.set_atom(i, 'H')
        methane.add_bond(0, i, 'single')

    print(f"  Atoms: {methane.num_atoms}")
    print(f"  Carbon neighbors: {methane.neighbors[0]}")

    # Molecular GNN
    print("\n--- Molecular GNN ---")
    gnn = MolecularGNN(atom_dim=10, bond_dim=4, hidden_dim=16, num_layers=2)

    # Forward pass
    property_value = gnn.forward(methane)
    print(f"  Predicted property: {property_value:.4f}")

    # Train step
    print("\n--- Training ---")
    for i in range(10):
        loss = gnn.train_step(methane, target=0.5, learning_rate=0.01)
        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}: loss={loss:.4f}")

    # Generate random molecules
    print("\n--- Molecular Generation ---")
    gen = MolecularGeneration()
    for i in range(3):
        mol = gen.generate_random_molecule(num_atoms=4)
        prop = gnn.forward(mol)
        print(f"  Molecule {i+1}: {mol.num_atoms} atoms, property={prop:.4f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: GNNs process molecular graphs to predict properties!")
    print("Message passing aggregates atom and bond information.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_molecular_ai()
