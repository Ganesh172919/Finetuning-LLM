"""
################################################################################
FLUX — RECTIFIED FLOW TRANSFORMERS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Flux?
    Flux is a modern image generation architecture using rectified flow.
    It's the architecture behind Black Forest Labs' image models.

Key Innovation:
    Instead of diffusion (noising/denoising), uses rectified flow
    (straight-line interpolation between noise and image).

Interview Questions:
    1. "What is Flux?"
        A transformer-based image generation model using rectified flow.

    2. "How is Flux different from Stable Diffusion?"
        Flux uses transformers instead of U-Net, and rectified flow
        instead of diffusion.

################################################################################
"""

import numpy as np
from typing import Optional
import math

################################################################################
# SECTION 1: RECTIFIED FLOW
################################################################################

class RectifiedFlow:
    """
    Rectified Flow
    ==============

    Instead of diffusion noise schedule, uses straight-line
    interpolation between noise and clean image.

    Formula:
    x_t = (1-t) * noise + t * x_0

    The model learns to predict the velocity (dx/dt).
    """

    def __init__(self):
        pass

    def interpolate(self, x_0: np.ndarray, noise: np.ndarray, t: float) -> np.ndarray:
        """
        Interpolate between noise and clean image.

        Args:
            x_0: Clean image
            noise: Random noise
            t: Time (0=noise, 1=clean)

        Returns:
            Interpolated sample
        """
        return (1 - t) * noise + t * x_0

    def velocity(self, x_0: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """
        Compute velocity (target for model).

        velocity = x_0 - noise
        """
        return x_0 - noise


################################################################################
# SECTION 2: FLUX MODEL
################################################################################

class FluxModel:
    """
    Flux Architecture
    =================

    Transformer-based image generation with rectified flow.

    Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Noise + Time → Patchify → Transformer Blocks    │
    │        ↓                                          │
    │ Text Conditioning (cross-attention)              │
    │        ↓                                          │
    │ Predict Velocity                                  │
    │        ↓                                          │
    │ Unpatchify → Image                               │
    └─────────────────────────────────────────────────┘
    """

    def __init__(self, d_model: int = 64, n_layers: int = 4):
        self.d_model = d_model
        self.n_layers = n_layers
        self.flow = RectifiedFlow()

    def generate(
        self,
        prompt_embedding: Optional[np.ndarray] = None,
        n_steps: int = 20
    ) -> np.ndarray:
        """
        Generate image using rectified flow.

        Args:
            prompt_embedding: Text conditioning
            n_steps: Number of steps

        Returns:
            Generated image
        """
        # Start with noise
        x = np.random.randn(1, 3, 64, 64)

        # Euler steps
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = step / n_steps
            # Predict velocity (simplified)
            velocity = -x  # Simplified
            x = x + velocity * dt

        return x


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_flux():
    """Demonstrate Flux concepts."""
    print("=" * 70)
    print("FLUX DEMONSTRATION")
    print("=" * 70)

    # Rectified flow
    print("\n--- Rectified Flow ---")
    flow = RectifiedFlow()
    x_0 = np.random.randn(3, 3)  # Clean
    noise = np.random.randn(3, 3)  # Noise

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x_t = flow.interpolate(x_0, noise, t)
        print(f"t={t:.2f}: x_t mean={x_t.mean():.3f}")

    velocity = flow.velocity(x_0, noise)
    print(f"Velocity: {velocity.mean():.3f}")

    # Flux model
    print("\n--- Flux Model ---")
    model = FluxModel(d_model=32, n_layers=2)
    image = model.generate(n_steps=10)
    print(f"Generated shape: {image.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_flux()


################################################################################
# REFERENCES
################################################################################

# [1] Esser, P., et al. (2024). Scaling Rectified Flow Transformers for Image Synthesis.

################################################################################
