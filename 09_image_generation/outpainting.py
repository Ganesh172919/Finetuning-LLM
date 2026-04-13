"""
################################################################################
OUTPAINTING — EXTENDING IMAGE BOUNDARIES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Outpainting?
    Extending an image beyond its original boundaries.

Interview Questions:
    Q: "What's the difference between inpainting and outpainting?"
    A: Inpainting: fill in missing interior regions
       Outpainting: extend the image beyond its borders

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: OUTPAINTING MODEL
################################################################################

class OutpaintingModel:
    """
    Outpainting Model
    =================

    Extends images beyond their boundaries.
    """

    def __init__(self, d_model: int = 64):
        self.d_model = d_model

    def extend(
        self,
        image: np.ndarray,
        direction: str = "right",
        pixels: int = 64
    ) -> np.ndarray:
        """
        Extend image.

        Args:
            image: Input image [h × w × c]
            direction: Direction to extend
            pixels: Number of pixels to add

        Returns:
            extended: Extended image
        """
        h, w, c = image.shape

        if direction == "right":
            extension = np.random.randn(h, pixels, c) * 0.1
            return np.concatenate([image, extension], axis=1)
        elif direction == "left":
            extension = np.random.randn(h, pixels, c) * 0.1
            return np.concatenate([extension, image], axis=1)
        elif direction == "bottom":
            extension = np.random.randn(pixels, w, c) * 0.1
            return np.concatenate([image, extension], axis=0)
        else:
            extension = np.random.randn(pixels, w, c) * 0.1
            return np.concatenate([extension, image], axis=0)


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_outpainting():
    """Demonstrate outpainting."""
    print("=" * 70)
    print("OUTPAINTING DEMONSTRATION")
    print("=" * 70)

    model = OutpaintingModel()
    image = np.random.randn(32, 32, 3)
    extended = model.extend(image, direction="right", pixels=16)
    print(f"Input: {image.shape}")
    print(f"Extended: {extended.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_outpainting()
