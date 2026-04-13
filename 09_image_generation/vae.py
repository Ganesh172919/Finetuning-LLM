"""
################################################################################
VARIATIONAL AUTOENCODER (VAE) — LEARNING LATENT REPRESENTATIONS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a VAE?
    A Variational Autoencoder learns to compress images into a small
    latent space and reconstruct them. It's used in Stable Diffusion
    to work in a smaller, more efficient space.

Why does it matter?
    Working directly with pixels is expensive:
    - 512×512 image = 786,432 values
    - Latent 64×64 = 12,288 values (64x reduction!)

    Stable Diffusion works in latent space:
    1. Encode image to latent (VAE encoder)
    2. Diffusion in latent space
    3. Decode latent to image (VAE decoder)

Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Image (512×512×3)                                │
    │        ↓                                          │
    │ VAE Encoder                                       │
    │        ↓                                          │
    │ Latent (64×64×4)                                 │
    │        ↓                                          │
    │ [Diffusion happens here]                         │
    │        ↓                                          │
    │ VAE Decoder                                       │
    │        ↓                                          │
    │ Reconstructed Image (512×512×3)                  │
    └─────────────────────────────────────────────────┘

Interview Questions:
        1. "What is a VAE?"
           A model that learns to compress data into a latent space
           and reconstruct it. The latent space is regularized to be
           smooth and continuous.

        2. "Why use VAE in diffusion models?"
           To work in a smaller latent space, which is much more
           efficient than working with pixels directly.

        3. "What's the difference between AE and VAE?"
           AE: deterministic encoding
           VAE: probabilistic encoding (learns distribution)
           VAE enables sampling and interpolation.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple
import math

################################################################################
# SECTION 1: VAE ENCODER
################################################################################

class VAEEncoder:
    """
    VAE Encoder
    ============

    Definition: Compresses images into latent representations.

    Architecture: Convolutional layers → mean and variance

    Output: μ and σ of the latent distribution
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 4):
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Conv layers (simplified)
        scale = math.sqrt(2.0 / (in_channels + latent_dim))
        self.conv = np.random.randn(latent_dim * 2, in_channels, 3, 3) * scale

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode image to latent distribution parameters.

        Args:
            x: Image [batch × channels × height × width]

        Returns:
            mean: Latent mean [batch × latent_dim × h/8 × w/8]
            logvar: Latent log variance [batch × latent_dim × h/8 × w/8]
        """
        # Simplified encoding
        batch, channels, height, width = x.shape

        # Downsample (simplified - real uses strided convolutions)
        h = height // 8
        w = width // 8

        # Output mean and log variance
        mean = np.random.randn(batch, self.latent_dim, h, w) * 0.1
        logvar = np.random.randn(batch, self.latent_dim, h, w) * 0.01

        return mean, logvar

    @staticmethod
    def reparameterize(mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick.

        Instead of sampling from N(μ, σ), sample from N(0, 1) and transform:
        z = μ + σ * ε, where ε ~ N(0, 1)

        This allows backpropagation through the sampling.
        """
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mean.shape)
        return mean + std * eps


################################################################################
# SECTION 2: VAE DECODER
################################################################################

class VAEDecoder:
    """
    VAE Decoder
    ===========

    Definition: Reconstructs images from latent representations.

    Architecture: Transposed convolutions (upsampling)
    """

    def __init__(self, latent_dim: int = 4, out_channels: int = 3):
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        # Transposed conv layers (simplified)
        scale = math.sqrt(2.0 / (latent_dim + out_channels))
        self.conv = np.random.randn(out_channels, latent_dim, 3, 3) * scale

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent to image.

        Args:
            z: Latent [batch × latent_dim × h × w]

        Returns:
            image: [batch × channels × height × width]
        """
        # Upsample (simplified - real uses transposed convolutions)
        batch, latent_dim, h, w = z.shape
        height = h * 8
        width = w * 8

        # Output image
        image = np.random.randn(batch, self.out_channels, height, width) * 0.1

        return image


################################################################################
# SECTION 3: COMPLETE VAE
################################################################################

class VariationalAutoencoder:
    """
    Variational Autoencoder
    ========================

    Complete VAE with encoder and decoder.

    Training Loss:
        L = Reconstruction Loss + KL Divergence

        Reconstruction Loss: ||x - x̄||² (MSE)
        KL Divergence: KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

    Interview Question:
        "What is the VAE loss function?"
        Two components:
        1. Reconstruction loss: how well can we reconstruct the input?
        2. KL divergence: how close is the latent distribution to N(0,1)?
        The balance between these determines the quality of the latent space.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 4):
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, in_channels)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full forward pass: encode → sample → decode.

        Args:
            x: Input image [batch × channels × height × width]

        Returns:
            reconstruction: Reconstructed image
            mean: Latent mean
            logvar: Latent log variance
        """
        # Encode
        mean, logvar = self.encoder.forward(x)

        # Sample
        z = VAEEncoder.reparameterize(mean, logvar)

        # Decode
        reconstruction = self.decoder.forward(z)

        return reconstruction, mean, logvar

    def compute_loss(self, x: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute VAE loss.

        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        recon, mean, logvar = self.forward(x)

        # Reconstruction loss (MSE)
        recon_loss = np.mean((x - recon) ** 2)

        # KL divergence
        kl_loss = -0.5 * np.mean(1 + logvar - mean ** 2 - np.exp(logvar))

        # Total loss
        total_loss = recon_loss + kl_loss

        return total_loss, recon_loss, kl_loss

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode to latent space."""
        mean, logvar = self.encoder.forward(x)
        return VAEEncoder.reparameterize(mean, logvar)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode from latent space."""
        return self.decoder.forward(z)


################################################################################
# SECTION 4: TESTING & EXAMPLES
################################################################################

def demonstrate_vae():
    """Demonstrate VAE concepts."""
    print("=" * 70)
    print("VARIATIONAL AUTOENCODER DEMONSTRATION")
    print("=" * 70)

    # Create VAE
    print("\n--- Creating VAE ---")
    vae = VariationalAutoencoder(in_channels=3, latent_dim=4)

    # Forward pass
    print("\n--- Forward Pass ---")
    x = np.random.randn(1, 3, 64, 64)  # Small image
    recon, mean, logvar = vae.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent mean shape: {mean.shape}")
    print(f"Reconstruction shape: {recon.shape}")

    # Loss computation
    print("\n--- Loss Computation ---")
    total, recon, kl = vae.compute_loss(x)
    print(f"Total loss: {total:.4f}")
    print(f"Reconstruction loss: {recon:.4f}")
    print(f"KL divergence: {kl:.4f}")

    # Encode/Decode
    print("\n--- Encode/Decode ---")
    z = vae.encode(x)
    print(f"Latent shape: {z.shape}")
    x_recon = vae.decode(z)
    print(f"Decoded shape: {x_recon.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_vae()


################################################################################
# REFERENCES
################################################################################

# [1] Kingma, D.P., & Welling, M. (2014). Auto-Encoding Variational Bayes.
# [2] Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models.

################################################################################
