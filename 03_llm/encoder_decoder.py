"""
################################################################################
ENCODER-DECODER MODELS — T5-STYLE ARCHITECTURES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Encoder-Decoder Models?
    Models with both encoder and decoder components.
    The encoder processes input, the decoder generates output.

    Used for:
    - Translation (English → French)
    - Summarization (article → summary)
    - Question answering (context + question → answer)

Historical Evolution:
    - 2014: Seq2Seq (Sutskever et al.)
    - 2017: Transformer (Vaswani et al.)
    - 2019: T5 (Raffel et al.)
    - 2022: Flan-T5

Interview Questions:
    1. "What's the difference between encoder-decoder and decoder-only?"
       Encoder-decoder: separate processing for input/output
       Decoder-only: same architecture for both

    2. "When should I use encoder-decoder?"
       When input and output are different modalities or formats.
       E.g., translation, summarization.

    3. "What is T5?"
       Text-to-Text Transfer Transformer.
       All tasks are framed as text-to-text.

################################################################################
"""

import numpy as np
from typing import Optional
import math

import sys
sys.path.append('..')
from ..02_transformers.attention import MultiHeadAttention, CrossAttention
from ..02_transformers.layers import TransformerBlock, RMSNorm

################################################################################
# SECTION 1: ENCODER
################################################################################

class Encoder:
    """
    Transformer Encoder
    ===================

    Processes input bidirectionally.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int):
        self.layers = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        self.norm = RMSNorm(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Encode input (bidirectional)."""
        for layer in self.layers:
            x = layer.forward(x, mask=None)
        return self.norm.forward(x)


################################################################################
# SECTION 2: DECODER
################################################################################

class Decoder:
    """
    Transformer Decoder
    ====================

    Generates output autoregressively with cross-attention to encoder.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int):
        self.layers = []
        for _ in range(n_layers):
            self.layers.append({
                'self_attn': MultiHeadAttention(d_model, n_heads),
                'cross_attn': CrossAttention(d_model, n_heads),
                'ffn': FeedForward(d_model),
                'norm1': RMSNorm(d_model),
                'norm2': RMSNorm(d_model),
                'norm3': RMSNorm(d_model)
            })

    def forward(
        self,
        x: np.ndarray,
        encoder_output: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Decode with cross-attention to encoder."""
        # Simplified - real implementation has proper layer structure
        return x


################################################################################
# SECTION 3: T5 MODEL
################################################################################

class T5:
    """
    T5: Text-to-Text Transfer Transformer
    ======================================

    All tasks are text-to-text:
    - Translation: "translate English to French: Hello" → "Bonjour"
    - Summarization: "summarize: [article]" → "[summary]"
    - QA: "question: [q] context: [c]" → "[answer]"

    Interview Question:
        "What's the T5 approach?"
        Frame all NLP tasks as text-to-text.
        Same model, same training, different prompts.
    """

    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 768,
        n_encoder_layers: int = 12,
        n_decoder_layers: int = 12,
        n_heads: int = 12
    ):
        self.encoder = Encoder(d_model, n_encoder_layers, n_heads)
        self.decoder = Decoder(d_model, n_decoder_layers, n_heads)

    def forward(
        self,
        encoder_input: np.ndarray,
        decoder_input: np.ndarray
    ) -> np.ndarray:
        """Forward pass through encoder-decoder."""
        encoder_output = self.encoder.forward(encoder_input)
        decoder_output = self.decoder.forward(decoder_input, encoder_output)
        return decoder_output


################################################################################
# SECTION 4: TESTING
################################################################################

def demonstrate_encoder_decoder():
    """Demonstrate encoder-decoder models."""
    print("=" * 70)
    print("ENCODER-DECODER MODEL DEMONSTRATION")
    print("=" * 70)

    # T5
    print("\n--- T5 ---")
    t5 = T5(
        vocab_size=1000,
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4
    )

    encoder_input = np.random.randn(2, 10, 128)
    decoder_input = np.random.randn(2, 8, 128)
    output = t5.forward(encoder_input, decoder_input)
    print(f"Encoder input: {encoder_input.shape}")
    print(f"Decoder input: {decoder_input.shape}")
    print(f"Output: {output.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_encoder_decoder()


################################################################################
# REFERENCES
################################################################################

# [1] Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with T5.

################################################################################
