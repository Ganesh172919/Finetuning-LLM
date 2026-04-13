"""
################################################################################
GEOMETRIC DEEP LEARNING — EQUIVARIANT NEURAL NETWORKS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Geometric Deep Learning?
    Neural networks that respect symmetries of the data — particularly
    rotations, translations, and reflections. For 3D data (molecules,
    point clouds), the network's output should transform predictably
    when the input is rotated or translated.

Why does it matter?
    Standard neural networks don't understand 3D geometry:
    - Rotating a molecule changes its representation
    - The model must re-learn the same thing for each orientation
    - This wastes capacity and hurts generalization

    Equivariant networks handle this automatically:
    - Rotate input → output rotates accordingly
    - Translate input → output translates accordingly
    - The network LEARNS geometry-aware features

Key Concepts:
    - Equivariance: f(Rx) = Rf(x) (output transforms with input)
    - Invariance: f(Rx) = f(x) (output doesn't change)
    - SE(3): Special Euclidean group (rotations + translations)

Equivariance Examples:
    - Predict forces on atoms: forces should rotate with the molecule
    - Predict energy: energy should NOT change with rotation (invariant)

Architecture (SE(3)-Equivariant Layer):
    ┌─────────────────────────────────────────────────────────────────┐
    │                SE(3)-Equivariant Layer                          │
    │                                                                  │
    │  Input: Atom positions x_i, features h_i                       │
    │                                                                  │
    │  1. Compute relative positions: x_ij = x_j - x_i              │
    │  2. Compute distances: d_ij = ||x_ij||                         │
    │  3. Message: m_ij = f(h_i, h_j, d_ij) × x_ij                 │
    │  4. Update: h_i' = h_i + Σ_j m_ij                             │
    │                                                                  │
    │  Key: Messages are VECTORS (not scalars), so they rotate       │
    │  when the input rotates. This ensures equivariance.            │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "What is equivariance?"
       A function f is equivariant to transformation T if f(Tx) = Tf(x).
       For rotations R: the output rotates when the input rotates.
       This is crucial for 3D data where orientation is arbitrary.

    2. "Why not just use data augmentation?"
       Augmentation helps but doesn't guarantee exact equivariance.
       Equivariant networks have it built-in by construction,
       which is more efficient and guarantees the property.

    3. "What's the difference between equivariance and invariance?"
       Equivariance: output transforms with input (forces on atoms).
       Invariance: output is unchanged (total energy).
       Both are useful for different predictions.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: ROTATION AND TRANSLATION
################################################################################

def rotation_matrix_x(angle: float) -> np.ndarray:
    """Rotation matrix around x-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rotation_matrix_y(angle: float) -> np.ndarray:
    """Rotation matrix around y-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotation_matrix_z(angle: float) -> np.ndarray:
    """Rotation matrix around z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def random_rotation() -> np.ndarray:
    """Generate a random 3D rotation matrix."""
    # Random Euler angles
    a1, a2, a3 = np.random.uniform(0, 2*np.pi, 3)
    return rotation_matrix_z(a1) @ rotation_matrix_y(a2) @ rotation_matrix_z(a3)


################################################################################
# SECTION 2: EQUIVARIANT LAYER
################################################################################

class EquivariantLayer:
    """
    SE(3)-Equivariant Layer
    =========================

    Processes 3D point clouds with equivariance to rotations and translations.

    Key idea: use RELATIVE positions (which are translation-invariant)
    and compute messages as VECTORS (which are rotation-equivariant).

    The message from atom j to atom i:
        m_ij = f_scalar(h_i, h_j, d_ij) × (x_j - x_i)

    Where:
        f_scalar is an MLP that outputs a scalar weight
        (x_j - x_i) is the relative position vector

    This ensures:
    - Translation invariance: x_j - x_i doesn't change with translation
    - Rotation equivariance: the vector rotates with the molecule

    Interview Question:
        "How do you make a neural network equivariant?"
        Use relative positions (translation-invariant) and
        vector-valued messages (rotation-equivariant). The scalar
        part (distances, features) is invariant; the vector part
        (directions) transforms with rotation.
    """

    def __init__(self, feature_dim: int, hidden_dim: int):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Scalar network (invariant)
        self.scalar_W1 = np.random.randn(feature_dim * 2 + 1, hidden_dim) * 0.02
        self.scalar_W2 = np.random.randn(hidden_dim, 1) * 0.02

        # Feature update
        self.update_W = np.random.randn(feature_dim + hidden_dim, feature_dim) * 0.02

    def forward(
        self,
        positions: np.ndarray,
        features: np.ndarray,
        neighbors: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through equivariant layer.

        Args:
            positions: Atom positions (num_atoms, 3)
            features: Atom features (num_atoms, feature_dim)
            neighbors: Adjacency list

        Returns:
            (updated_positions, updated_features)
        """
        num_atoms = positions.shape[0]
        new_features = np.zeros_like(features)
        position_updates = np.zeros_like(positions)

        for i in range(num_atoms):
            messages_scalar = []
            messages_vector = []

            for j in neighbors[i]:
                # Relative position (translation-invariant)
                rel_pos = positions[j] - positions[i]

                # Distance (rotation-invariant)
                dist = np.linalg.norm(rel_pos)

                # Scalar message weight
                scalar_input = np.concatenate([features[i], features[j], [dist]])
                h = np.tanh(self.scalar_W1 @ scalar_input)
                weight = float(np.tanh(self.scalar_W2 @ h))

                # Vector message (rotation-equivariant)
                vector_msg = weight * rel_pos

                messages_scalar.append(h)
                messages_vector.append(vector_msg)

            # Aggregate
            if messages_scalar:
                agg_scalar = np.mean(messages_scalar, axis=0)
                agg_vector = np.mean(messages_vector, axis=0)
            else:
                agg_scalar = np.zeros(self.hidden_dim)
                agg_vector = np.zeros(3)

            # Update features
            update_input = np.concatenate([features[i], agg_scalar])
            new_features[i] = np.tanh(self.update_W @ update_input)

            # Position update (equivariant)
            position_updates[i] = agg_vector

        new_positions = positions + position_updates

        return new_positions, new_features


################################################################################
# SECTION 3: SE3 EQUIVARIANT NETWORK
################################################################################

class SE3Equivariant:
    """
    SE(3)-Equivariant Network
    ===========================

    Multi-layer equivariant network for 3D molecular data.

    Architecture:
    1. Multiple equivariant layers
    2. Invariant readout for property prediction
    3. Equivariant readout for force prediction

    Use cases:
    - Energy prediction (invariant output)
    - Force prediction (equivariant output)
    - Molecular property prediction

    Interview Questions:
        1. "How does SE(3) equivariance help in molecular modeling?"
           Molecules exist in 3D space and their properties shouldn't
           depend on orientation. SE(3)-equivariant networks ensure
           this by construction, leading to better generalization.

        2. "What's the computational cost of equivariant networks?"
           Higher than standard GNNs due to vector operations.
           But the improved sample efficiency often compensates.
           Modern implementations (MACE, NequIP) are quite efficient.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 3):
        self.num_layers = num_layers
        self.layers = [
            EquivariantLayer(feature_dim, hidden_dim)
            for _ in range(num_layers)
        ]

        # Invariant readout (for energy prediction)
        self.readout_W = np.random.randn(feature_dim, 1) * 0.02

    def forward(
        self,
        positions: np.ndarray,
        features: np.ndarray,
        neighbors: List[List[int]]
    ) -> Tuple[float, np.ndarray]:
        """
        Forward pass.

        Args:
            positions: Atom positions (num_atoms, 3)
            features: Atom features (num_atoms, feature_dim)
            neighbors: Adjacency list

        Returns:
            (invariant_prediction, equivariant_output)
        """
        pos = positions.copy()
        feat = features.copy()

        # Message passing layers
        for layer in self.layers:
            pos, feat = layer.forward(pos, feat, neighbors)

        # Invariant readout: mean pool features
        mol_repr = np.mean(feat, axis=0)
        invariant_pred = float(mol_repr @ self.readout_W)

        # Equivariant output: per-atom predictions
        equivariant_out = pos - positions  # Position changes

        return invariant_pred, equivariant_out

    def predict_energy(
        self,
        positions: np.ndarray,
        features: np.ndarray,
        neighbors: List[List[int]]
    ) -> float:
        """Predict molecular energy (invariant)."""
        energy, _ = self.forward(positions, features, neighbors)
        return energy

    def predict_forces(
        self,
        positions: np.ndarray,
        features: np.ndarray,
        neighbors: List[List[int]]
    ) -> np.ndarray:
        """Predict atomic forces (equivariant)."""
        _, forces = self.forward(positions, features, neighbors)
        return -forces  # Forces are negative gradient of energy


################################################################################
# SECTION 4: TESTING & DEMONSTRATION
################################################################################

def demonstrate_geometric_deep_learning():
    """Demonstrate geometric deep learning."""
    print("=" * 70)
    print("GEOMETRIC DEEP LEARNING (SE(3)-EQUIVARIANT)")
    print("=" * 70)

    # Create a simple molecule (water: H2O)
    num_atoms = 3
    feature_dim = 8

    # Positions: O at origin, H atoms at typical bond distance
    positions = np.array([
        [0.0, 0.0, 0.0],      # O
        [0.96, 0.0, 0.0],     # H1
        [-0.24, 0.93, 0.0]    # H2
    ])

    # Features (simplified)
    features = np.random.randn(num_atoms, feature_dim) * 0.1
    features[0, 0] = 1.0  # Mark oxygen
    features[1, 1] = 1.0  # Mark hydrogen
    features[2, 1] = 1.0  # Mark hydrogen

    # Neighbors
    neighbors = [[1, 2], [0], [0]]  # O connected to both H

    print(f"\nMolecule: {num_atoms} atoms")
    print(f"Positions:\n{positions}")

    # Create equivariant network
    model = SE3Equivariant(feature_dim, hidden_dim=16, num_layers=2)

    # Predict energy
    print("\n--- Energy Prediction ---")
    energy = model.predict_energy(positions, features, neighbors)
    print(f"  Energy: {energy:.4f}")

    # Predict forces
    print("\n--- Force Prediction ---")
    forces = model.predict_forces(positions, features, neighbors)
    print(f"  Forces shape: {forces.shape}")
    print(f"  Forces:\n{forces}")

    # Test equivariance
    print("\n--- Equivariance Test ---")
    # Rotate the molecule
    R = rotation_matrix_z(np.pi / 4)  # 45 degree rotation
    rotated_positions = positions @ R.T

    # Predict on rotated molecule
    energy_rot = model.predict_energy(rotated_positions, features, neighbors)
    forces_rot = model.predict_forces(rotated_positions, features, neighbors)

    print(f"  Original energy: {energy:.4f}")
    print(f"  Rotated energy: {energy_rot:.4f}")
    print(f"  Energy difference: {abs(energy - energy_rot):.6f} (should be ~0)")

    # Forces should rotate with the molecule
    forces_rotated_expected = forces @ R.T
    force_diff = np.linalg.norm(forces_rot - forces_rotated_expected)
    print(f"  Force equivariance error: {force_diff:.6f} (should be ~0)")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Equivariant networks respect 3D symmetries!")
    print("Energy is invariant, forces are equivariant to rotations.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_geometric_deep_learning()
