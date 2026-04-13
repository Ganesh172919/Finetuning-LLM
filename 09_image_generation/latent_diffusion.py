"""
################################################################################
LATENT DIFFUSION — STABLE DIFFUSION ARCHITECTURE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Latent Diffusion?
    Latent diffusion performs diffusion in a compressed latent space
    instead of pixel space. This is much more efficient.

    Architecture:
    1. VAE encodes image to latent space
    2. Diffusion happens in latent space
    3. VAE decodes latent back to image

Why does it matter?
    Pixel space: 512×512×3 = 786,432 dimensions
    Latent space: 64×64×4 = 16,384 dimensions (48x smaller!)

    This enables:
    - Higher resolution generation
    - Faster training
    - Lower memory usage

Historical Evolution:
    - 2022: Latent Diffusion Models (Stable Diffusion)
    - 2023: SDXL, ControlNet
    - 2024: Flux, SD3
    - 2025: Faster architectures

Interview Questions:
    1. "What is latent diffusion?"
       Diffusion in a compressed latent space instead of pixels.
       Uses VAE to encode/decode between pixel and latent space.

    2. "Why is latent diffusion better than pixel diffusion?"
       Much more efficient (48x fewer dimensions).
       Enables higher resolution and faster training.

    3. "What is Stable Diffusion?"
       An open-source latent diffusion model for image generation.
       Uses CLIP text encoder, U-Net denoiser, and VAE decoder.

################################################################################
"""

import numpy as np
from typing import Optional
import math

import sys
sys.path.append('..')
from .diffusion import DiffusionModel, NoiseScheduler
from .vae import VariationalAutoencoder
from .unet import UNet

################################################################################
# SECTION 1: LATENT DIFFUSION MODEL
################################################################################

class LatentDiffusionModel:
    """
    Latent Diffusion Model
    ======================

    Combines VAE and diffusion for efficient image generation.

    Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Training:                                         │
    │ Image → VAE Encoder → Latent → Add Noise → UNet │
    │                                                  │
    │ Generation:                                       │
    │ Noise → UNet Denoise → Latent → VAE Decoder → Image│
    └─────────────────────────────────────────────────┘

    Interview Question:
        "Walk me through Stable Diffusion generation."
        1. Encode text with CLIP
        2. Start with random noise in latent space
        3. Iteratively denoise with U-Net (conditioned on text)
        4. Decode latent to image with VAE
    """

    def __init__(
        self,
        image_size: int = 512,
        latent_dim: int = 4,
        d_model: int = 320,
        n_layers: int = 4
    ):
        # VAE for encoding/decoding
        self.vae = VariationalAutoencoder(in_channels=3, latent_dim=latent_dim)

        # U-Net for denoising
        self.unet = UNet(
            in_channels=latent_dim,
            out_channels=latent_dim,
            model_channels=d_model,
            channel_mult=[1, 2, 4, 4],
            attention_resolutions=[2, 4]
        )

        # Noise scheduler
        self.scheduler = NoiseScheduler(num_timesteps=1000)

    def training_loss(self, image: np.ndarray) -> float:
        """
        Compute training loss.

        1. Encode image to latent
        2. Add noise to latent
        3. Predict noise with U-Net
        4. Compute MSE loss
        """
        # Encode to latent
        latent = self.vae.encode(image)

        # Sample timestep
        t = np.random.randint(0, 1000)

        # Add noise
        noise = np.random.randn(*latent.shape)
        noisy_latent, _ = self.scheduler.add_noise(latent, t, noise)

        # Predict noise
        predicted_noise = self.unet.forward(noisy_latent, t)

        # MSE loss
        loss = np.mean((noise - predicted_noise) ** 2)
        return loss

    @staticmethod
    def generate(
        model: 'LatentDiffusionModel',
        prompt_embedding: Optional[np.ndarray] = None,
        n_steps: int = 50
    ) -> np.ndarray:
        """
        Generate image from noise.

        Args:
            model: Trained model
            prompt_embedding: Text conditioning
            n_steps: Denoising steps

        Returns:
            Generated image
        """
        # Start with noise in latent space
        latent = np.random.randn(1, 4, 64, 64)

        # Denoise
        timesteps = np.linspace(999, 0, n_steps, dtype=int)
        for t in timesteps:
            predicted_noise = model.unet.forward(latent, t)
            # Simplified denoising step
            latent = latent - 0.01 * predicted_noise

        # Decode to image
        image = model.vae.decode(latent)
        return image


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_latent_diffusion():
    """Demonstrate latent diffusion."""
    print("=" * 70)
    print("LATENT DIFFUSION DEMONSTRATION")
    print("=" * 70)

    # Create model
    print("\n--- Creating Model ---")
    model = LatentDiffusionModel(
        image_size=64,
        latent_dim=4,
        d_model=32,
        n_layers=2
    )

    # Training loss
    print("\n--- Training Loss ---")
    image = np.random.randn(1, 3, 64, 64)
    loss = model.training_loss(image)
    print(f"Loss: {loss:.4f}")

    # Generation
    print("\n--- Generation ---")
    generated = LatentDiffusionModel.generate(model, n_steps=10)
    print(f"Generated shape: {generated.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_latent_diffusion()


################################################################################
# REFERENCES
################################################################################

# [1] Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models.

################################################################################
