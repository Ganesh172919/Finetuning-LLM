"""
################################################################################
AUDIO ENCODER — ENCODING AUDIO FOR MULTIMODAL MODELS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is an Audio Encoder?
    Converts audio to embeddings for use in multimodal models.

Key Models:
    - Whisper Encoder
    - HuBERT
    - Wav2Vec

Interview Questions:
    Q: "How do you encode audio for AI models?"
    A: Convert to mel spectrogram, then encode with transformer.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: AUDIO ENCODER
################################################################################

class AudioEncoder:
    """
    Audio Encoder
    =============

    Encodes audio for multimodal models.
    """

    def __init__(self, n_mels: int = 80, d_model: int = 512):
        self.n_mels = n_mels
        self.d_model = d_model

    def encode(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """
        Encode mel spectrogram.

        Args:
            mel_spectrogram: [n_mels × n_frames]

        Returns:
            embedding: [d_model]
        """
        # Simplified encoding
        return np.mean(mel_spectrogram, axis=1)[:self.d_model] if mel_spectrogram.shape[0] >= self.d_model else np.zeros(self.d_model)


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_audio_encoder():
    """Demonstrate audio encoder."""
    print("=" * 70)
    print("AUDIO ENCODER DEMONSTRATION")
    print("=" * 70)

    encoder = AudioEncoder(n_mels=80, d_model=64)
    mel = np.random.randn(80, 100)
    embedding = encoder.encode(mel)
    print(f"Mel: {mel.shape}")
    print(f"Embedding: {embedding.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_audio_encoder()
