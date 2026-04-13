"""
################################################################################
INPAINTING — FILLING IN MISSING PARTS OF IMAGES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Inpainting?
    Filling in masked or missing regions of an image.

Applications:
    - Object removal
    - Image restoration
    - Photo editing

Architecture:
    Image + Mask → Diffusion Model → Completed Image

Interview Questions:
    Q: "How does inpainting work?"
    A: Condition the diffusion model on the unmasked regions
       and generate content for the masked regions.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: INPAINTING MODEL
################################################################################

class InpaintingModel:
    """
    Inpainting Model
    ================

    Fills in masked regions of images.

    Interview Questions:
        Q: "What's the difference between inpainting and outpainting?"
        A: Inpainting: fill in missing interior regions
           Outpainting: extend the image beyond its borders
    """

    def __init__(self, d_model: int = 64):
        self.d_model = d_model

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        n_steps: int = 20
    ) -> np.ndarray:
        """
        Inpaint masked region.

        Args:
            image: Input image
            mask: Binary mask (1=masked, 0=keep)
            n_steps: Denoising steps

        Returns:
            completed: Completed image
        """
        # Start with noise in masked region
        completed = image.copy()
        noise = np.random.randn(*image.shape)
        completed = completed * (1 - mask) + noise * mask

        # Denoise masked region
        for _ in range(n_steps):
            # Simplified denoising
            completed = completed * 0.95 + image * 0.05 * mask

        return completed


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_inpainting():
    """Demonstrate inpainting."""
    print("=" * 70)
    print("INPAINTING DEMONSTRATION")
    print("=" * 70)

    model = InpaintingModel()

    # Create image and mask
    image = np.random.randn(64, 64, 3)
    mask = np.zeros((64, 64, 3))
    mask[20:40, 20:40] = 1  # Mask center region

    completed = model.inpaint(image, mask, n_steps=10)
    print(f"Image: {image.shape}")
    print(f"Mask: {mask.shape}")
    print(f"Completed: {completed.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_inpainting()
