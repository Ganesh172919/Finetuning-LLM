"""
################################################################################
IMAGE EDITING — AI-POWERED IMAGE MODIFICATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Image Editing?
    Using AI to modify images based on text instructions.

Key Models:
    - InstructPix2Pix: Edit images with text instructions
    - Imagic: Text-based image editing
    - Prompt-to-Prompt: Attention-based editing

Interview Questions:
    Q: "How does text-based image editing work?"
    A: Condition the diffusion model on both the original image
       and the editing instruction.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: IMAGE EDITOR
################################################################################

class ImageEditor:
    """
    Image Editor
    ============

    Edits images based on text instructions.

    Interview Questions:
        Q: "What's the difference between editing and generation?"
        A: Editing preserves most of the original image.
           Generation creates from scratch.
    """

    def __init__(self, d_model: int = 64):
        self.d_model = d_model

    def edit(
        self,
        image: np.ndarray,
        instruction: str,
        strength: float = 0.5
    ) -> np.ndarray:
        """
        Edit image based on instruction.

        Args:
            image: Original image
            instruction: Text instruction
            strength: How much to change (0=none, 1=full)

        Returns:
            edited: Edited image
        """
        # Simplified editing
        noise = np.random.randn(*image.shape)
        edited = image * (1 - strength) + noise * strength
        return edited


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_editing():
    """Demonstrate image editing."""
    print("=" * 70)
    print("IMAGE EDITING DEMONSTRATION")
    print("=" * 70)

    editor = ImageEditor()
    image = np.random.randn(64, 64, 3)
    edited = editor.edit(image, "Make it blue", strength=0.3)
    print(f"Original: {image.shape}")
    print(f"Edited: {edited.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_editing()
