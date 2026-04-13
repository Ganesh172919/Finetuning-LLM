"""
################################################################################
IMAGE CAPTIONING — GENERATING TEXT FROM IMAGES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Image Captioning?
    Generating text descriptions of images.

Architecture:
    Image → Vision Encoder → Decoder → Caption

Key Models:
    - BLIP (2022): Bootstrapping Language-Image Pre-training
    - BLIP-2 (2023): More efficient
    - CoCa (2022): Contrastive Captioners

Interview Questions:
    Q: "How does image captioning work?"
    A: Encode image with vision encoder, decode text with
       transformer decoder using cross-attention.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: IMAGE CAPTIONING MODEL
################################################################################

class ImageCaptioningModel:
    """
    Image Captioning Model
    ======================

    Generates text descriptions of images.

    Interview Questions:
        Q: "What's the difference between captioning and VQA?"
        A: Captioning: describe the image
           VQA: answer specific questions about the image
    """

    def __init__(self, d_vision: int = 256, d_language: int = 256, vocab_size: int = 1000):
        self.d_vision = d_vision
        self.d_language = d_language

        # Vision encoder
        self.vision_proj = np.random.randn(d_vision, d_language) * 0.02

        # Language decoder
        self.token_embed = np.random.randn(vocab_size, d_language) * 0.02
        self.output_head = np.random.randn(d_language, vocab_size) * 0.02

    def caption(self, image_features: np.ndarray, max_len: int = 20) -> np.ndarray:
        """
        Generate caption for image.

        Args:
            image_features: Vision features

        Returns:
            token_ids: Generated caption tokens
        """
        # Project image features
        hidden = image_features @ self.vision_proj

        tokens = []
        for _ in range(max_len):
            logits = hidden @ self.output_head
            token = np.argmax(logits)
            tokens.append(token)

            # Update hidden (simplified)
            hidden = hidden + self.token_embed[token] * 0.1

        return np.array(tokens)


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_captioning():
    """Demonstrate image captioning."""
    print("=" * 70)
    print("IMAGE CAPTIONING DEMONSTRATION")
    print("=" * 70)

    model = ImageCaptioningModel(d_vision=64, d_language=64, vocab_size=100)
    image_features = np.random.randn(64)
    caption = model.caption(image_features, max_len=10)
    print(f"Image features: {image_features.shape}")
    print(f"Caption tokens: {caption.tolist()}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_captioning()


################################################################################
# REFERENCES
################################################################################

# [1] Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training.

################################################################################
