"""
################################################################################
SPARSE AUTOENCODER — DISCOVERING MONOSEMANTIC FEATURES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Sparse Autoencoder (SAE)?
    A neural network that learns to decompose model activations into
    sparse, interpretable features. Each feature corresponds to a
    single, understandable concept.

Why does it matter?
    Neural networks suffer from SUPERPOSITION — multiple features
    are encoded in the same neurons. This makes interpretation difficult.

    SAEs solve this by:
    1. Projecting activations into a higher-dimensional space
    2. Enforcing sparsity (only a few features are active)
    3. Each feature becomes interpretable (monosemantic)

    This is Anthropic's key insight: "Towards Monosemanticity" (2023)

How does it work?
    1. Encode: activation → high-dimensional sparse features
    2. Decode: sparse features → reconstructed activation
    3. Train to minimize reconstruction error + sparsity penalty
    4. Each feature in the high-dimensional space becomes interpretable

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Sparse Autoencoder                               │
    │                                                                  │
    │  Model Activation (d_model) ──▶ Encoder ──▶ Features (d_sae)   │
    │                                              ↓                   │
    │                                      Sparsity Penalty (L1)      │
    │                                              ↓                   │
    │  Reconstructed (d_model) ◀── Decoder ◀── Features (d_sae)      │
    │                                                                  │
    │  Loss = ||activation - reconstructed||² + λ × ||features||₁    │
    │                                                                  │
    │  d_sae >> d_model (expansion factor typically 4-32x)           │
    └─────────────────────────────────────────────────────────────────┘

Key Results (Anthropic, 2024):
    - Found interpretable features in Claude:
      - "Golden Gate Bridge" feature
      - "Code error" feature
      - "Deception" feature
    - Features are monosemantic (one concept per feature)
    - Can be used to modify model behavior

Interview Questions:
    1. "What is a sparse autoencoder?"
       A network that decomposes model activations into sparse,
       interpretable features. It maps to a higher-dimensional
       space where each dimension corresponds to one concept.

    2. "Why does expansion help with interpretability?"
       In the original space, features are superimposed (mixed).
       Expansion allows each feature to have its own dimension.
       Sparsity ensures only a few features are active at once.

    3. "How do you verify features are interpretable?"
       (a) Look at what maximally activates each feature,
       (b) Check if the feature consistently responds to one concept,
       (c) Intervene on the feature and observe behavior changes.

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
class SAEConfig:
    """
    Configuration for Sparse Autoencoder.
    """
    # Dimensions
    d_model: int = 256       # Model activation dimension
    d_sae: int = 1024        # SAE feature dimension (expanded)
    expansion_factor: int = 4  # d_sae = d_model * expansion_factor

    # Training
    learning_rate: float = 1e-3
    sparsity_coeff: float = 0.1  # L1 penalty weight
    num_epochs: int = 100
    batch_size: int = 64

    # Sparsity
    target_sparsity: float = 0.05  # Target fraction of active features


################################################################################
# SECTION 2: SPARSE AUTOENCODER
################################################################################

class SparseAutoencoder:
    """
    Sparse Autoencoder for Feature Discovery
    ===========================================

    Learns to decompose model activations into sparse, interpretable features.

    The key insight: by expanding to a higher-dimensional space and
    enforcing sparsity, each dimension can learn to represent a single,
    monosemantic concept.

    Training loss:
        L = ||x - x̂||² + λ × ||z||₁

    Where:
        x = original activation
        x̂ = reconstructed activation
        z = sparse features
        λ = sparsity coefficient

    Interview Questions:
        1. "Why L1 penalty for sparsity?"
           L1 penalty (||z||₁) pushes most features to exactly zero.
           This is different from L2 which just makes features small.
           L1 gives true sparsity — most features are inactive.

        2. "How do you choose the expansion factor?"
           Empirically, 4-32x works well. Larger expansion captures
           more features but is more expensive. The sweet spot depends
           on the model and layer being analyzed.

        3. "What's the relationship between SAE features and neurons?"
           SAE features are linear combinations of neurons. Each
           feature may span multiple neurons, but represents one
           concept. This is "decomposing" superposition.
    """

    def __init__(self, config: SAEConfig):
        self.config = config

        # Encoder: d_model → d_sae
        self.W_enc = np.random.randn(config.d_model, config.d_sae) * 0.02
        self.b_enc = np.zeros(config.d_sae)

        # Decoder: d_sae → d_model
        self.W_dec = np.random.randn(config.d_sae, config.d_model) * 0.02
        self.b_dec = np.zeros(config.d_model)

        # Initialize decoder weights to have unit norm
        norms = np.linalg.norm(self.W_dec, axis=1, keepdims=True)
        self.W_dec = self.W_dec / (norms + 1e-8)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode activations to sparse features.

        Args:
            x: Model activations (batch, d_model)

        Returns:
            Sparse features (batch, d_sae)
        """
        # Linear projection
        z = x @ self.W_enc + self.b_enc

        # ReLU activation (ensure non-negative features)
        z = np.maximum(0, z)

        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode sparse features to reconstructed activations.

        Args:
            z: Sparse features (batch, d_sae)

        Returns:
            Reconstructed activations (batch, d_model)
        """
        return z @ self.W_dec + self.b_dec

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full forward pass: encode then decode.

        Args:
            x: Model activations

        Returns:
            (reconstructed, sparse_features)
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def compute_loss(
        self,
        x: np.ndarray,
        x_hat: np.ndarray,
        z: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute training loss.

        Loss = reconstruction_loss + sparsity_loss

        Reconstruction: ||x - x̂||²
        Sparsity: λ × ||z||₁

        Args:
            x: Original activations
            x_hat: Reconstructed activations
            z: Sparse features

        Returns:
            (total_loss, reconstruction_loss, sparsity_loss)
        """
        # Reconstruction loss (MSE)
        recon_loss = np.mean((x - x_hat) ** 2)

        # Sparsity loss (L1)
        sparsity_loss = self.config.sparsity_coeff * np.mean(np.abs(z))

        total_loss = recon_loss + sparsity_loss

        return total_loss, recon_loss, sparsity_loss

    def train_step(self, x: np.ndarray) -> Dict[str, float]:
        """
        One training step.

        Args:
            x: Batch of activations

        Returns:
            Dictionary of losses
        """
        batch_size = x.shape[0]

        # Forward pass
        x_hat, z = self.forward(x)

        # Compute loss
        total_loss, recon_loss, sparsity_loss = self.compute_loss(x, x_hat, z)

        # Backward pass (simplified)
        # Gradient of reconstruction loss w.r.t. x_hat
        grad_x_hat = 2 * (x_hat - x) / batch_size

        # Gradient through decoder
        grad_z = grad_x_hat @ self.W_dec.T

        # Gradient of L1 penalty
        grad_z += self.config.sparsity_coeff * np.sign(z) / batch_size

        # Gradient through ReLU
        grad_z = grad_z * (z > 0)

        # Update decoder
        self.W_dec -= self.learning_rate * (z.T @ grad_x_hat)
        self.b_dec -= self.learning_rate * np.mean(grad_x_hat, axis=0)

        # Update encoder
        self.W_enc -= self.learning_rate * (x.T @ grad_z)
        self.b_enc -= self.learning_rate * np.mean(grad_z, axis=0)

        # Normalize decoder weights (weight normalization)
        norms = np.linalg.norm(self.W_dec, axis=1, keepdims=True)
        self.W_dec = self.W_dec / (norms + 1e-8)

        return {
            'total_loss': float(total_loss),
            'recon_loss': float(recon_loss),
            'sparsity_loss': float(sparsity_loss),
            'mean_sparsity': float(np.mean(np.abs(z)))
        }

    @property
    def learning_rate(self):
        return self.config.learning_rate


################################################################################
# SECTION 3: FEATURE DISCOVERY
################################################################################

class FeatureDiscovery:
    """
    Feature Discovery using Sparse Autoencoder
    =============================================

    Discovers and analyzes interpretable features learned by the SAE.

    Methods:
    1. Max Activation Analysis: What maximally activates each feature?
    2. Feature Ablation: What happens when we remove a feature?
    3. Feature Steering: Modify behavior by manipulating features

    Interview Questions:
        1. "How do you discover what a feature represents?"
           Look at the examples that maximally activate the feature.
           If the top-10 examples all involve "Golden Gate Bridge",
           that's probably what the feature represents.

        2. "Can you edit model behavior with SAE features?"
           Yes! This is "feature steering" — activate a feature
           to induce its behavior, or ablate it to remove it.
           Anthropic showed this can modify Claude's behavior.

        3. "What's the difference between features and neurons?"
           Neurons are the network's computational units.
           Features are interpretable concepts. Due to superposition,
           one neuron may encode multiple features, and one feature
           may span multiple neurons. SAEs disentangle them.
    """

    def __init__(self, sae: SparseAutoencoder):
        self.sae = sae

    def find_max_activating_examples(
        self,
        activations: np.ndarray,
        feature_idx: int,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find examples that maximally activate a feature.

        Args:
            activations: Dataset of activations (n_samples, d_model)
            feature_idx: Index of feature to analyze
            top_k: Number of top examples

        Returns:
            (top_activations, feature_values)
        """
        # Encode all activations
        features = self.sae.encode(activations)

        # Get feature values
        feature_values = features[:, feature_idx]

        # Find top-k
        top_indices = np.argsort(feature_values)[::-1][:top_k]

        return activations[top_indices], feature_values[top_indices]

    def compute_feature_statistics(
        self,
        activations: np.ndarray
    ) -> Dict[int, Dict]:
        """
        Compute statistics for all features.

        Args:
            activations: Dataset of activations

        Returns:
            Dictionary mapping feature index to statistics
        """
        features = self.sae.encode(activations)

        stats = {}
        for i in range(self.sae.config.d_sae):
            values = features[:, i]
            stats[i] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'sparsity': float(np.mean(values > 0)),
                'max': float(np.max(values)),
                'active_count': int(np.sum(values > 0))
            }

        return stats

    def ablate_feature(
        self,
        activations: np.ndarray,
        feature_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ablate a feature and observe the effect.

        Args:
            activations: Original activations
            feature_idx: Feature to ablate

        Returns:
            (original_reconstruction, ablated_reconstruction)
        """
        # Original reconstruction
        x_hat_orig, z_orig = self.sae.forward(activations)

        # Ablate: set feature to zero
        z_ablated = z_orig.copy()
        z_ablated[:, feature_idx] = 0

        # Reconstruct without the feature
        x_hat_ablated = self.sae.decode(z_ablated)

        return x_hat_orig, x_hat_ablated

    def steer_feature(
        self,
        activations: np.ndarray,
        feature_idx: int,
        scale: float = 2.0
    ) -> np.ndarray:
        """
        Steer model behavior by scaling a feature.

        Args:
            activations: Original activations
            feature_idx: Feature to steer
            scale: Scale factor (>1 = amplify, <1 = suppress)

        Returns:
            Steered activations
        """
        # Encode
        features = self.sae.encode(activations)

        # Scale the feature
        features[:, feature_idx] *= scale

        # Decode
        return self.sae.decode(features)


################################################################################
# SECTION 4: TESTING & DEMONSTRATION
################################################################################

def demonstrate_sparse_autoencoder():
    """Demonstrate sparse autoencoder for feature discovery."""
    print("=" * 70)
    print("SPARSE AUTOENCODER FOR FEATURE DISCOVERY")
    print("=" * 70)

    # Configuration
    config = SAEConfig(
        d_model=32,
        d_sae=128,
        expansion_factor=4,
        learning_rate=0.01,
        sparsity_coeff=0.1,
        num_epochs=50
    )

    print(f"\nConfiguration:")
    print(f"  Model dim: {config.d_model}")
    print(f"  SAE dim: {config.d_sae}")
    print(f"  Expansion: {config.expansion_factor}x")
    print(f"  Sparsity coeff: {config.sparsity_coeff}")

    # Create SAE
    sae = SparseAutoencoder(config)

    # Generate synthetic activations (with structure)
    print("\n--- Generating Synthetic Activations ---")
    n_samples = 500
    activations = np.random.randn(n_samples, config.d_model)

    # Add some structure (features that can be discovered)
    for i in range(5):
        # Feature i responds to dimension i
        activations[:, i] += np.random.randn(n_samples) * 2

    print(f"  Samples: {n_samples}")
    print(f"  Shape: {activations.shape}")

    # Train SAE
    print("\n--- Training Sparse Autoencoder ---")
    for epoch in range(config.num_epochs):
        # Mini-batch training
        indices = np.random.permutation(n_samples)[:config.batch_size]
        batch = activations[indices]
        losses = sae.train_step(batch)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: "
                  f"total={losses['total_loss']:.4f}, "
                  f"recon={losses['recon_loss']:.4f}, "
                  f"sparse={losses['sparsity_loss']:.4f}")

    # Feature discovery
    print("\n--- Feature Discovery ---")
    discovery = FeatureDiscovery(sae)

    # Compute feature statistics
    stats = discovery.compute_feature_statistics(activations)

    # Find most active features
    active_features = sorted(
        stats.items(),
        key=lambda x: x[1]['active_count'],
        reverse=True
    )

    print("Top 5 most active features:")
    for feat_idx, feat_stats in active_features[:5]:
        print(f"  Feature {feat_idx}: "
              f"active_count={feat_stats['active_count']}, "
              f"sparsity={feat_stats['sparsity']:.3f}")

    # Max activating examples
    print("\n--- Max Activating Examples ---")
    top_feat = active_features[0][0]
    top_acts, top_vals = discovery.find_max_activating_examples(
        activations, top_feat, top_k=3
    )
    print(f"Feature {top_feat}:")
    print(f"  Top activation values: {top_vals}")

    # Feature ablation
    print("\n--- Feature Ablation ---")
    orig, ablated = discovery.ablate_feature(activations[:10], top_feat)
    diff = np.mean(np.abs(orig - ablated))
    print(f"  Mean change from ablating feature {top_feat}: {diff:.4f}")

    # Feature steering
    print("\n--- Feature Steering ---")
    steered = discovery.steer_feature(activations[:10], top_feat, scale=3.0)
    change = np.mean(np.abs(steered - activations[:10]))
    print(f"  Mean change from steering feature {top_feat}: {change:.4f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: SAEs decompose superposition into interpretable features!")
    print("Each feature corresponds to one concept (monosemanticity).")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_sparse_autoencoder()
