"""
################################################################################
IMAGE UPSCALING — INCREASING IMAGE RESOLUTION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Image Upscaling?
    Increasing the resolution of an image while preserving quality.

Key Models:
    - Real-ESRGAN: Real-world super resolution
    - SwinIR: Swin Transformer for restoration
    - ESRGAN: Enhanced super resolution

Interview Questions:
    Q: "How does AI upscaling work?"
    A: Learn to predict high-frequency details from low-resolution input.
       Uses perceptual loss and adversarial training.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: UPSCALE MODEL
################################################################################

class UpscaleModel:
    """
    Image Upscale Model
    ===================

    Increases image resolution.

    Interview Questions:
        Q: "What's the difference between traditional and AI upscaling?"
        A: Traditional: interpolation (bicubic, bilinear)
           AI: learned to predict details
    """

    def __init__(self, scale_factor: int = 4):
        self.scale_factor = scale_factor

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale image.

        Args:
            image: [h × w × c]

        Returns:
            upscaled: [h*scale × w*scale × c]
        """
        h, w, c = image.shape
        new_h, new_w = h * self.scale_factor, w * self.scale_factor

        # Simplified: nearest neighbor upsampling
        upscaled = np.repeat(np.repeat(image, self.scale_factor, axis=0), self.scale_factor, axis=1)

        return upscaled


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_upscaling():
    """Demonstrate image upscaling."""
    print("=" * 70)
    print("IMAGE UPSCALING DEMONSTRATION")
    print("=" * 70)

    model = UpscaleModel(scale_factor=4)
    image = np.random.randn(16, 16, 3)
    upscaled = model.upscale(image)
    print(f"Input: {image.shape}")
    print(f"Upscaled: {upscaled.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_upscaling()
