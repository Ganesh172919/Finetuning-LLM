"""
################################################################################
MUSIC GENERATION — AI-POWERED MUSIC CREATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Music Generation?
    AI models that generate music from text descriptions or other inputs.

Key Models:
    - MusicGen (Meta): Text-to-music
    - AudioLM: Audio generation
    - Bark: Speech and music

Architecture:
    Text → Text Encoder → Music Decoder → Audio Waveform

Interview Questions:
    Q: "How does music generation work?"
    A: Similar to text generation but for audio tokens.
       Encode music as discrete tokens, then autoregressively generate.

################################################################################
"""

import numpy as np
from typing import Optional

################################################################################
# SECTION 1: MUSIC ENCODER
################################################################################

class MusicEncoder:
    """
    Music Encoder
    =============

    Encodes audio waveform to discrete tokens.

    Uses neural audio codec (like EnCodec) to compress
    audio into discrete tokens.
    """

    def __init__(self, n_tokens: int = 1024, sample_rate: int = 24000):
        self.n_tokens = n_tokens
        self.sample_rate = sample_rate

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        """Encode waveform to tokens."""
        # Simplified: quantize to n_tokens
        tokens = np.random.randint(0, self.n_tokens, len(waveform) // 1000)
        return tokens

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """Decode tokens to waveform."""
        # Simplified: random waveform
        return np.random.randn(len(tokens) * 1000) * 0.1


################################################################################
# SECTION 2: MUSIC GENERATOR
################################################################################

class MusicGenerator:
    """
    Music Generator
    ===============

    Generates music from text descriptions.

    Pipeline:
    1. Encode text prompt
    2. Generate music tokens autoregressively
    3. Decode tokens to waveform

    Interview Questions:
        Q: "How do you generate music with AI?"
        A: Tokenize music, train transformer to predict next token,
           generate autoregressively, decode to audio.
    """

    def __init__(self, vocab_size: int = 1024, d_model: int = 256):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Text encoder
        self.text_proj = np.random.randn(128, d_model) * 0.02

        # Token embeddings
        self.token_embed = np.random.randn(vocab_size, d_model) * 0.02

        # Output head
        self.output_head = np.random.randn(d_model, vocab_size) * 0.02

    def generate(
        self,
        text_embedding: np.ndarray,
        max_tokens: int = 100,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Generate music tokens.

        Args:
            text_embedding: Text conditioning
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            tokens: Generated music tokens
        """
        tokens = []
        hidden = text_embedding @ self.text_proj

        for _ in range(max_tokens):
            logits = hidden @ self.output_head
            probs = np.exp(logits / temperature - np.max(logits / temperature))
            probs = probs / np.sum(probs)

            token = np.random.choice(self.vocab_size, p=probs)
            tokens.append(token)

            # Update hidden state (simplified)
            hidden = hidden + self.token_embed[token] * 0.1

        return np.array(tokens)


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_music():
    """Demonstrate music generation."""
    print("=" * 70)
    print("MUSIC GENERATION DEMONSTRATION")
    print("=" * 70)

    # Encoder
    print("\n--- Music Encoder ---")
    encoder = MusicEncoder(n_tokens=512)
    waveform = np.random.randn(24000)  # 1 second
    tokens = encoder.encode(waveform)
    decoded = encoder.decode(tokens)
    print(f"Waveform: {waveform.shape}")
    print(f"Tokens: {tokens.shape}")
    print(f"Decoded: {decoded.shape}")

    # Generator
    print("\n--- Music Generator ---")
    generator = MusicGenerator(vocab_size=512, d_model=64)
    text_emb = np.random.randn(128)
    music_tokens = generator.generate(text_emb, max_tokens=50)
    print(f"Generated tokens: {music_tokens.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_music()


################################################################################
# REFERENCES
################################################################################

# [1] Copet, J., et al. (2023). Simple and Controllable Music Generation.

################################################################################
