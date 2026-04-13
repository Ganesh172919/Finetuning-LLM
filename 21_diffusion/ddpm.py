"""
################################################################################
DDPM — DENOISING DIFFUSION PROBABILISTIC MODELS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is DDPM?
    The foundational diffusion model that learns to reverse
    a gradual noise addition process.

Training:
    1. Take clean image x₀
    2. Sample timestep t
    3. Add noise: xₜ = √(ᾱₜ)x₀ + √(1-ᾱₜ)ε
    4. Train to predict ε

Interview Questions:
    1. "How does DDPM work?"
        Learn to predict noise added to images.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple

################################################################################
# SECTION 1: DDPM
################################################################################

class DDPM:
    """
    DDPM: Denoising Diffusion Probabilistic Model
    ================================================

    Learns to generate images by reversing noise.
    """

    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps

        # Noise schedule
        self.betas = np.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)

    def add_noise(self, x_0: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise to clean image."""
        noise = np.random.randn(*x_0.shape)
        sqrt_alpha = np.sqrt(self.alpha_bar[t])
        sqrt_one_minus_alpha = np.sqrt(1 - self.alpha_bar[t])
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def sample(self, shape: Tuple, n_steps: int = 50) -> np.ndarray:
        """Generate sample by denoising."""
        x_t = np.random.randn(*shape)

        timesteps = np.linspace(self.num_timesteps - 1, 0, n_steps, dtype=int)
        for t in timesteps:
            # Simplified denoising
            predicted_noise = x_t * 0.1
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bar[t]
            x_t = (1 / np.sqrt(alpha)) * (x_t - (1 - alpha) / np.sqrt(1 - alpha_bar) * predicted_noise)

        return x_t


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_ddpm():
    """Demonstrate DDPM."""
    print("=" * 70)
    print("DDPM DEMONSTRATION")
    print("=" * 70)

    ddpm = DDPM(num_timesteps=1000)

    # Add noise
    x_0 = np.random.randn(1, 3, 32, 32)
    x_t, noise = ddpm.add_noise(x_0, t=500)
    print(f"Clean: {x_0.shape}")
    print(f"Noisy: {x_t.shape}")

    # Sample
    sample = ddpm.sample((1, 3, 32, 32), n_steps=20)
    print(f"Sample: {sample.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_ddpm()


################################################################################
# REFERENCES
################################################################################

# [1] Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models.

################################################################################
