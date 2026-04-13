"""
################################################################################
WHISPER — AUTOMATIC SPEECH RECOGNITION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Whisper?
    Whisper is a speech recognition model by OpenAI that converts
    audio to text. It's trained on 680,000 hours of multilingual data.

Why does it matter?
    Whisper powers:
    - Transcription services
    - Voice assistants
    - Subtitles and captions
    - Accessibility tools
    - Meeting transcription

Architecture:
    Audio waveform → Mel spectrogram → Encoder → Decoder → Text

    ┌─────────────────────────────────────────────────┐
    │ Audio: "Hello world" (waveform)                  │
    │        ↓                                          │
    │ Mel Spectrogram (visual representation)          │
    │        ↓                                          │
    │ Transformer Encoder (audio understanding)        │
    │        ↓                                          │
    │ Transformer Decoder (text generation)            │
    │        ↓                                          │
    │ Text: "Hello world"                              │
    └─────────────────────────────────────────────────┘

Historical Evolution:
    - 2022: Whisper (OpenAI)
    - 2023: Whisper v2, faster-whisper
    - 2024: Whisper v3, distil-whisper
    - 2025: Real-time transcription

Interview Questions:
    1. "How does Whisper work?"
       Converts audio to mel spectrogram, encodes with transformer,
       decodes to text using cross-attention.

    2. "What's a mel spectrogram?"
       A visual representation of audio frequencies over time,
       scaled to match human hearing (mel scale).

    3. "How is Whisper different from traditional ASR?"
       Traditional: acoustic model + language model + decoder
       Whisper: single end-to-end model, trained on massive data

################################################################################
"""

import numpy as np
from typing import Optional, List
import math

import sys
sys.path.append('..')
from ..02_transformers.attention import MultiHeadAttention
from ..02_transformers.layers import TransformerBlock, RMSNorm

################################################################################
# SECTION 1: AUDIO PROCESSING
################################################################################

class AudioProcessor:
    """
    Audio Processing Pipeline
    =========================

    Converts raw audio to model inputs.

    Steps:
    1. Resample to 16kHz
    2. Compute mel spectrogram
    3. Normalize
    """

    def __init__(self, sample_rate: int = 16000, n_mels: int = 80):
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def waveform_to_mel(self, waveform: np.ndarray) -> np.ndarray:
        """
        Convert waveform to mel spectrogram.

        Args:
            waveform: Audio samples [samples]

        Returns:
            mel: Mel spectrogram [n_mels × time]
        """
        # Simplified - real implementation uses FFT + mel filterbank
        n_frames = len(waveform) // 160  # 10ms frames
        mel = np.random.randn(self.n_mels, n_frames) * 0.1
        return mel


################################################################################
# SECTION 2: WHISPER ENCODER
################################################################################

class WhisperEncoder:
    """
    Whisper Encoder
    ===============

    Processes mel spectrogram into audio features.

    Architecture:
    - Convolutional layers (downsampling)
    - Transformer encoder blocks
    - Output: audio features

    The encoder captures:
    - Phonemes (basic sounds)
    - Words and phrases
    - Speaker characteristics
    - Audio context
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8
    ):
        self.d_model = d_model

        # Conv layers for downsampling
        self.conv1 = np.random.randn(d_model, n_mels, 3, 3) * 0.02
        self.conv2 = np.random.randn(d_model, d_model, 3, 3) * 0.02

        # Transformer encoder
        self.layers = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        self.norm = RMSNorm(d_model)

    def forward(self, mel: np.ndarray) -> np.ndarray:
        """
        Encode mel spectrogram.

        Args:
            mel: [batch × n_mels × time]

        Returns:
            features: [batch × time/4 × d_model]
        """
        batch, n_mels, time = mel.shape

        # Simplified encoding
        # Real: Conv layers downsample time dimension
        x = np.random.randn(batch, time // 4, self.d_model)

        # Transformer
        for layer in self.layers:
            x = layer.forward(x)

        x = self.norm.forward(x)
        return x


################################################################################
# SECTION 3: WHISPER DECODER
################################################################################

class WhisperDecoder:
    """
    Whisper Decoder
    ===============

    Generates text from audio features.

    Architecture:
    - Token embedding
    - Transformer decoder with cross-attention
    - Output: text tokens

    Cross-attention:
    The decoder attends to encoder outputs to generate text.
    This is how audio information flows to text generation.
    """

    def __init__(
        self,
        vocab_size: int = 51865,  # Whisper vocabulary
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 448
    ):
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.position_embedding = np.random.randn(max_seq_len, d_model) * 0.02

        # Transformer decoder blocks with cross-attention
        self.layers = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        self.norm = RMSNorm(d_model)

        # Output projection
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.02

    def forward(
        self,
        token_ids: np.ndarray,
        encoder_output: np.ndarray
    ) -> np.ndarray:
        """
        Decode text from audio features.

        Args:
            token_ids: [batch × seq_len]
            encoder_output: [batch × audio_len × d_model]

        Returns:
            logits: [batch × seq_len × vocab_size]
        """
        batch, seq_len = token_ids.shape

        # Token + position embedding
        x = self.token_embedding[token_ids]
        x = x + self.position_embedding[:seq_len]

        # Transformer decoder (with cross-attention to encoder)
        for layer in self.layers:
            x = layer.forward(x)

        x = self.norm.forward(x)

        # Project to vocabulary
        logits = np.matmul(x, self.output_proj)

        return logits


################################################################################
# SECTION 4: WHISPER MODEL
################################################################################

class WhisperModel:
    """
    Whisper: Automatic Speech Recognition
    =======================================

    Complete Whisper model with encoder-decoder architecture.

    Model sizes:
    - Tiny: 39M params, 1GB VRAM
    - Base: 74M params, 1GB VRAM
    - Small: 244M params, 2GB VRAM
    - Medium: 769M params, 5GB VRAM
    - Large: 1550M params, 10GB VRAM

    Interview Question:
        "What languages does Whisper support?"
        Whisper supports 99 languages. It's trained on multilingual
        data and can transcribe and translate to English.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        n_heads: int = 8,
        vocab_size: int = 51865
    ):
        self.audio_processor = AudioProcessor(n_mels=n_mels)
        self.encoder = WhisperEncoder(n_mels, d_model, n_encoder_layers, n_heads)
        self.decoder = WhisperDecoder(vocab_size, d_model, n_decoder_layers, n_heads)

    def transcribe(self, waveform: np.ndarray) -> str:
        """
        Transcribe audio to text.

        Args:
            waveform: Audio samples

        Returns:
            text: Transcribed text
        """
        # Process audio
        mel = self.audio_processor.waveform_to_mel(waveform)

        # Encode audio
        encoder_output = self.encoder.forward(mel)

        # Decode text (simplified - real uses autoregressive generation)
        prompt_ids = np.array([[50258]])  # <|startoftranscript|>
        logits = self.decoder.forward(prompt_ids, encoder_output)

        # In real implementation: autoregressive generation
        return "Transcribed text would appear here"


################################################################################
# SECTION 5: TESTING & EXAMPLES
################################################################################

def demonstrate_whisper():
    """Demonstrate Whisper concepts."""
    print("=" * 70)
    print("WHISPER DEMONSTRATION")
    print("=" * 70)

    # Audio processing
    print("\n--- Audio Processing ---")
    processor = AudioProcessor(sample_rate=16000, n_mels=80)
    waveform = np.random.randn(16000)  # 1 second of audio
    mel = processor.waveform_to_mel(waveform)
    print(f"Waveform: {waveform.shape}")
    print(f"Mel spectrogram: {mel.shape}")

    # Whisper model
    print("\n--- Whisper Model ---")
    whisper = WhisperModel(
        n_mels=80,
        d_model=256,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4
    )

    # Transcribe
    text = whisper.transcribe(waveform)
    print(f"Transcribed: {text}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_whisper()


################################################################################
# REFERENCES
################################################################################

# [1] Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision.
# [2] Gandhe, A., et al. (2024). Distil-Whisper: Robust Knowledge Distillation.

################################################################################
