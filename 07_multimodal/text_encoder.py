"""
################################################################################
TEXT ENCODER FOR MULTIMODAL MODELS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Text Encoder?
    Converts text to embeddings for use in multimodal models.

Key Models:
    - CLIP Text Encoder
    - T5 Text Encoder
    - BERT Text Encoder

Interview Questions:
    Q: "Why use a separate text encoder for multimodal models?"
    A: To map text to the same embedding space as images.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: TEXT ENCODER
################################################################################

class MultimodalTextEncoder:
    """
    Text Encoder for Multimodal Models
    ====================================

    Encodes text for use in vision-language models.
    """

    def __init__(self, vocab_size: int = 49408, d_model: int = 512):
        self.d_model = d_model
        self.token_embed = np.random.randn(vocab_size, d_model) * 0.02

    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Encode text tokens.

        Args:
            token_ids: [batch × seq_len]

        Returns:
            embeddings: [batch × d_model]
        """
        x = self.token_embed[token_ids]
        return np.mean(x, axis=1)


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_text_encoder():
    """Demonstrate text encoder."""
    print("=" * 70)
    print("TEXT ENCODER DEMONSTRATION")
    print("=" * 70)

    encoder = MultimodalTextEncoder(vocab_size=1000, d_model=64)
    token_ids = np.random.randint(0, 1000, (2, 10))
    embeddings = encoder.encode(token_ids)
    print(f"Input: {token_ids.shape}")
    print(f"Embeddings: {embeddings.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_text_encoder()
