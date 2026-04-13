"""
################################################################################
LLaVA — VISUAL INSTRUCTION TUNING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is LLaVA?
    LLaVA (Large Language and Vision Assistant) connects a vision
    encoder with a language model to understand images and text.

Key Innovation:
    Simple but effective: use a linear projection to map vision
    features into the language model's embedding space.

Architecture:
    Image → Vision Encoder (CLIP ViT) → Linear Projection → LLM
    Text → Token Embedding → LLM

    The LLM then generates text conditioned on both image and text.

Training:
    Stage 1: Pre-train projection layer on image-text pairs
    Stage 2: Fine-tune entire model on visual instruction data

Interview Questions:
        Q: "How does LLaVA work?"
        A: Encode image with CLIP ViT, project to LLM space with
           linear layer, then generate text with LLM.

        Q: "Why use a linear projection instead of cross-attention?"
        A: Simpler, fewer parameters, works surprisingly well.
           The LLM can learn to attend to image features through
           its own attention mechanism.

################################################################################
"""

import numpy as np
from typing import Optional
import math

################################################################################
# SECTION 1: CLIP VISION ENCODER
################################################################################

class CLIPVisionEncoder:
    """
    CLIP Vision Encoder
    ===================

    Encodes images into feature vectors using CLIP's ViT.
    """

    def __init__(self, d_vision: int = 768, n_patches: int = 196):
        self.d_vision = d_vision
        self.n_patches = n_patches

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image to features.

        Args:
            image: [batch × channels × height × width]

        Returns:
            features: [batch × n_patches × d_vision]
        """
        batch = image.shape[0]
        return np.random.randn(batch, self.n_patches, self.d_vision) * 0.1


################################################################################
# SECTION 2: MULTIMODAL PROJECTOR
################################################################################

class MultimodalProjector:
    """
    Multimodal Projector
    ====================

    Maps vision features to language model embedding space.

    LLaVA uses a simple linear projection:
        projected = W @ vision_features + b

    More advanced versions use MLP:
        projected = W2 @ GELU(W1 @ vision_features + b1) + b2
    """

    def __init__(self, d_vision: int = 768, d_language: int = 4096):
        self.d_vision = d_vision
        self.d_language = d_language

        # Linear projection
        self.weight = np.random.randn(d_vision, d_language) * math.sqrt(2.0 / d_vision)
        self.bias = np.zeros(d_language)

    def forward(self, vision_features: np.ndarray) -> np.ndarray:
        """
        Project vision features to language space.

        Args:
            vision_features: [batch × n_patches × d_vision]

        Returns:
            projected: [batch × n_patches × d_language]
        """
        return np.matmul(vision_features, self.weight) + self.bias


################################################################################
# SECTION 3: LLaVA MODEL
################################################################################

class LLaVAModel:
    """
    LLaVA: Large Language and Vision Assistant
    ===========================================

    Combines vision encoder and language model.

    Architecture:
        Image → CLIP ViT → Projector → Image Tokens
        Text → Tokenizer → Text Tokens

        Combined: [Image Tokens; Text Tokens] → LLM → Response

    Training:
        Stage 1: Pre-train projector on image-text pairs
        Stage 2: Fine-tune on visual instruction data

    Interview Questions:
        Q: "How does LLaVA handle multiple images?"
        A: Each image is encoded separately and projected.
           All image tokens are concatenated with text tokens.

        Q: "What's the maximum image resolution?"
        A: Depends on the vision encoder. CLIP ViT typically
           handles 224×224 or 336×336.
    """

    def __init__(
        self,
        d_vision: int = 768,
        d_language: int = 4096,
        vocab_size: int = 32000,
        max_seq_len: int = 4096
    ):
        self.d_vision = d_vision
        self.d_language = d_language

        # Vision encoder
        self.vision_encoder = CLIPVisionEncoder(d_vision)

        # Projector
        self.projector = MultimodalProjector(d_vision, d_language)

        # Language model (simplified)
        self.token_embed = np.random.randn(vocab_size, d_language) * 0.02
        self.output_head = np.random.randn(d_language, vocab_size) * 0.02

    def forward(
        self,
        image: Optional[np.ndarray] = None,
        text_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            image: [batch × channels × height × width]
            text_ids: [batch × seq_len]

        Returns:
            logits: [batch × total_seq_len × vocab_size]
        """
        batch = text_ids.shape[0] if text_ids is not None else image.shape[0]

        # Encode image
        if image is not None:
            vision_features = self.vision_encoder.encode(image)
            image_tokens = self.projector.forward(vision_features)
        else:
            image_tokens = np.zeros((batch, 0, self.d_language))

        # Encode text
        if text_ids is not None:
            text_tokens = self.token_embed[text_ids]
        else:
            text_tokens = np.zeros((batch, 0, self.d_language))

        # Concatenate image and text tokens
        combined = np.concatenate([image_tokens, text_tokens], axis=1)

        # Generate logits (simplified)
        logits = np.matmul(combined, self.output_head)

        return logits

    def chat(
        self,
        image: np.ndarray,
        question: str
    ) -> str:
        """
        Chat with the model.

        Args:
            image: Input image
            question: Question about the image

        Returns:
            answer: Model's answer
        """
        # Simplified chat interface
        return f"Answer about the image: {question[:50]}..."


################################################################################
# SECTION 4: TESTING
################################################################################

def demonstrate_llava():
    """Demonstrate LLaVA model."""
    print("=" * 70)
    print("LLaVA DEMONSTRATION")
    print("=" * 70)

    # Create model
    model = LLaVAModel(d_vision=128, d_language=256, vocab_size=1000)

    # Image only
    print("\n--- Image Encoding ---")
    image = np.random.randn(1, 3, 224, 224)
    vision_features = model.vision_encoder.encode(image)
    projected = model.projector.forward(vision_features)
    print(f"Vision features: {vision_features.shape}")
    print(f"Projected: {projected.shape}")

    # Combined
    print("\n--- Combined Forward ---")
    text_ids = np.random.randint(0, 1000, (1, 10))
    logits = model.forward(image=image, text_ids=text_ids)
    print(f"Logits: {logits.shape}")

    # Chat
    print("\n--- Chat ---")
    answer = model.chat(image, "What is in this image?")
    print(f"Answer: {answer}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_llava()


################################################################################
# REFERENCES
################################################################################

# [1] Liu, H., et al. (2023). Visual Instruction Tuning.

################################################################################
