"""
################################################################################
TEXT-TO-IMAGE GENERATION PIPELINE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Text-to-Image?
    Generating images from text descriptions using AI models.

Key Components:
    1. Text Encoder: Convert text to embeddings (CLIP)
    2. Diffusion Model: Generate image from noise (U-Net)
    3. VAE Decoder: Convert latent to pixels

Pipeline:
    "A cat sitting on a mat" → CLIP Text Encoder → Text Embedding
    Random Noise → U-Net (conditioned on text) → Denoised Latent
    Latent → VAE Decoder → Image

Models:
    - Stable Diffusion (2022)
    - DALL-E 2 (2022)
    - Midjourney (2022)
    - SDXL (2023)
    - Flux (2024)

Interview Questions:
        Q: "How does text-to-image generation work?"
        A: 1) Encode text with CLIP
           2) Start with random noise
           3) Iteratively denoise using U-Net conditioned on text
           4) Decode latent to image with VAE

        Q: "What is classifier-free guidance?"
        A: Run the model twice: once with text conditioning,
           once without. The final prediction is:
           output = uncond + guidance_scale * (cond - uncond)
           Higher guidance scale = more adherence to text.

################################################################################
"""

import numpy as np
from typing import Optional
import math

################################################################################
# SECTION 1: TEXT ENCODER
################################################################################

class TextEncoder:
    """
    Text Encoder for Text-to-Image
    ===============================

    Encodes text prompts into embeddings using CLIP.

    CLIP maps text and images to the same embedding space,
    so text embeddings can guide image generation.
    """

    def __init__(self, d_text: int = 768, max_seq_len: int = 77):
        self.d_text = d_text
        self.max_seq_len = max_seq_len

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to embedding.

        Args:
            text: Text prompt

        Returns:
            embedding: [1 × d_text]
        """
        # Simplified encoding
        return np.random.randn(1, self.d_text) * 0.1


################################################################################
# SECTION 2: CLASSIFIER-FREE GUIDANCE
################################################################################

class ClassifierFreeGuidance:
    """
    Classifier-Free Guidance
    ========================

    Balances between following the text prompt and generating
    high-quality images.

    Formula:
        output = unconditional + scale × (conditional - unconditional)

    Higher scale: more faithful to text, but may reduce quality
    Lower scale: higher quality, but may ignore text

    Typical values: 7.0 - 15.0

    Interview Questions:
        Q: "What is classifier-free guidance?"
        A: A technique to control how closely the generated image
           follows the text prompt. Run model with and without
           text conditioning, then interpolate.
    """

    def __init__(self, scale: float = 7.5):
        self.scale = scale

    def apply(
        self,
        unconditional: np.ndarray,
        conditional: np.ndarray
    ) -> np.ndarray:
        """
        Apply classifier-free guidance.

        Args:
            unconditional: Model output without text
            conditional: Model output with text

        Returns:
            guided: Guided output
        """
        return unconditional + self.scale * (conditional - unconditional)


################################################################################
# SECTION 3: NOISE SCHEDULER
################################################################################

class DDIMScheduler:
    """
    DDIM Scheduler
    ==============

    Denoising Diffusion Implicit Models scheduler.
    Allows fewer denoising steps than DDPM.

    DDPM: stochastic (adds noise at each step)
    DDIM: deterministic (no added noise)

    With DDIM, can use 20-50 steps instead of 1000.
    """

    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps

        # Noise schedule
        self.betas = np.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)

    def step(
        self,
        model_output: np.ndarray,
        timestep: int,
        sample: np.ndarray,
        eta: float = 0.0
    ) -> np.ndarray:
        """
        Denoising step.

        Args:
            model_output: Predicted noise
            timestep: Current timestep
            sample: Current noisy sample
            eta: Stochasticity (0=deterministic, 1=DDPM)

        Returns:
            prev_sample: Less noisy sample
        """
        alpha_bar_t = self.alpha_bar[timestep]
        alpha_bar_prev = self.alpha_bar[timestep - 1] if timestep > 0 else 1.0

        # Predicted x_0
        pred_x0 = (sample - np.sqrt(1 - alpha_bar_t) * model_output) / np.sqrt(alpha_bar_t)

        # Direction pointing to x_t
        dir_xt = np.sqrt(1 - alpha_bar_prev - eta**2 * (1 - alpha_bar_prev)) * model_output

        # Add noise
        noise = np.random.randn(*sample.shape) if eta > 0 and timestep > 0 else 0

        prev_sample = np.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + eta * np.sqrt(1 - alpha_bar_prev) * noise

        return prev_sample


################################################################################
# SECTION 4: TEXT-TO-IMAGE PIPELINE
################################################################################

class TextToImagePipeline:
    """
    Text-to-Image Pipeline
    ======================

    Complete pipeline for generating images from text.

    Steps:
    1. Encode text with CLIP
    2. Initialize with random noise
    3. Iteratively denoise with U-Net
    4. Decode latent to image with VAE

    Interview Questions:
        Q: "How many denoising steps do you need?"
        A: Typically 20-50 steps with DDIM scheduler.
           More steps = higher quality but slower.

        Q: "What's the difference between SD and SDXL?"
        A: SDXL has larger U-Net, more parameters, higher base
           resolution (1024 vs 512), and better quality.
    """

    def __init__(
        self,
        d_text: int = 768,
        d_latent: int = 4,
        latent_size: int = 64,
        n_steps: int = 30,
        guidance_scale: float = 7.5
    ):
        self.d_text = d_text
        self.d_latent = d_latent
        self.latent_size = latent_size
        self.n_steps = n_steps

        # Components
        self.text_encoder = TextEncoder(d_text)
        self.guidance = ClassifierFreeGuidance(guidance_scale)
        self.scheduler = DDIMScheduler(num_timesteps=1000)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate image from text prompt.

        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid
            seed: Random seed for reproducibility

        Returns:
            image: Generated image [height × width × channels]
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. Encode text
        text_embedding = self.text_encoder.encode(prompt)
        uncond_embedding = self.text_encoder.encode(negative_prompt)

        # 2. Initialize with random noise
        latent = np.random.randn(1, self.d_latent, self.latent_size, self.latent_size)

        # 3. Denoising loop
        timesteps = np.linspace(999, 0, self.n_steps, dtype=int)

        for t in timesteps:
            # Predict noise (conditional)
            noise_cond = self._unet_forward(latent, t, text_embedding)

            # Predict noise (unconditional)
            noise_uncond = self._unet_forward(latent, t, uncond_embedding)

            # Apply classifier-free guidance
            noise_pred = self.guidance.apply(noise_uncond, noise_cond)

            # Denoise step
            latent = self.scheduler.step(noise_pred, t, latent)

        # 4. Decode to image
        image = self._decode_latent(latent)

        return image

    def _unet_forward(
        self,
        latent: np.ndarray,
        timestep: int,
        text_embedding: np.ndarray
    ) -> np.ndarray:
        """U-Net forward pass (simplified)."""
        return np.random.randn(*latent.shape) * 0.1

    def _decode_latent(self, latent: np.ndarray) -> np.ndarray:
        """VAE decoder (simplified)."""
        # Upsample latent to image
        image = np.random.randn(512, 512, 3)
        return (image * 0.5 + 0.5).clip(0, 1)


################################################################################
# SECTION 5: TESTING
################################################################################

def demonstrate_text_to_image():
    """Demonstrate text-to-image pipeline."""
    print("=" * 70)
    print("TEXT-TO-IMAGE DEMONSTRATION")
    print("=" * 70)

    # Create pipeline
    pipeline = TextToImagePipeline(
        d_text=128,
        d_latent=4,
        latent_size=16,
        n_steps=10,
        guidance_scale=7.5
    )

    # Generate
    print("\n--- Generating Image ---")
    prompt = "A cat sitting on a mat in a sunny garden"
    image = pipeline.generate(prompt, seed=42)
    print(f"Prompt: {prompt}")
    print(f"Image shape: {image.shape}")
    print(f"Image range: [{image.min():.2f}, {image.max():.2f}]")

    # Guidance scale comparison
    print("\n--- Guidance Scale ---")
    for scale in [1.0, 5.0, 7.5, 15.0]:
        print(f"Scale {scale}: {'more creative' if scale < 7.5 else 'more faithful'}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_text_to_image()


################################################################################
# REFERENCES
################################################################################

# [1] Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models.
# [2] Song, J., et al. (2021). Denoising Diffusion Implicit Models.

################################################################################
