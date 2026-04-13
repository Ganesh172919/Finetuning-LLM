"""
################################################################################
IMAGE GENERATION MODELS — FROM NOISE TO IMAGES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Image Generation Models?
    Models that create images from text descriptions, other images, or noise.
    The current SOTA uses diffusion models that gradually denoise random
    noise into coherent images.

Why do they matter?
    Image generation powers:
    - DALL-E, Midjourney, Stable Diffusion (text-to-image)
    - Image editing and inpainting
    - Video generation (frame by frame)
    - Design and creative tools
    - Data augmentation
    - Medical imaging

Historical Evolution:
    - 2014: GANs (Goodfellow et al.)
    - 2015: DCGAN
    - 2018: StyleGAN
    - 2020: DDPM (diffusion models)
    - 2021: DALL-E, Stable Diffusion
    - 2022: Stable Diffusion open-source
    - 2023: SDXL, Flux
    - 2024: Sora (video), consistency models
    - 2025: Rectified flow, faster generation
    - 2026: Real-time generation, multimodal integration

########################################

MODELS IMPLEMENTED:

1. diffusion.py — Core diffusion model (DDPM)
2. latent_diffusion.py — Latent diffusion (Stable Diffusion)
3. unet.py — U-Net architecture for diffusion
4. vae.py — Variational Autoencoder
5. controlnet.py — Controllable generation
6. lora.py — Low-Rank Adaptation for fine-tuning
7. flux.py — Flux architecture
8. consistency.py — Consistency models

################################################################################
"""

from .diffusion import DiffusionModel, NoiseScheduler
from .latent_diffusion import LatentDiffusionModel
from .unet import UNet
from .vae import VariationalAutoencoder
