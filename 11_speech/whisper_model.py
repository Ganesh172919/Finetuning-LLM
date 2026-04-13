"""
################################################################################
WHISPER — AUTOMATIC SPEECH RECOGNITION MODEL
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Whisper?
    Whisper is OpenAI's speech recognition model that transcribes
    audio to text. It's trained on 680,000 hours of multilingual data.

Key Innovation:
    Instead of training on curated datasets, Whisper is trained on
    massive weakly-supervised data from the internet. This makes it
    robust to accents, background noise, and technical language.

Architecture:
    Audio → Mel Spectrogram → Encoder → Decoder → Text

    Encoder:
    - Convolutional layers (downsample)
    - Transformer encoder blocks
    - Processes audio features

    Decoder:
    - Token embedding
    - Transformer decoder with cross-attention
    - Generates text tokens

Historical Evolution:
    - 2022: Whisper v1 (OpenAI)
    - 2023: Whisper v2, faster-whisper
    - 2024: Whisper v3, distil-whisper
    - 2025: Real-time speech-to-speech

Interview Questions:
        Q: "How does Whisper work?"
        A: Converts audio to mel spectrogram, encodes with transformer,
           decodes to text with cross-attention. Trained on massive data.

        Q: "What's a mel spectrogram?"
        A: A visual representation of audio frequencies over time,
           scaled to match human hearing (mel scale).

################################################################################
"""

import numpy as np
from typing import Optional, List
import math

################################################################################
# SECTION 1: MEL SPECTROGRAM
################################################################################

class MelSpectrogram:
    """
    Mel Spectrogram
    ===============

    Converts audio waveform to a visual representation.

    Steps:
    1. Short-Time Fourier Transform (STFT)
    2. Compute power spectrum
    3. Apply mel filterbank
    4. Take log

    Mel Scale:
        Human hearing is logarithmic. The mel scale maps
        frequencies to match human perception.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def forward(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram.

        Args:
            waveform: Audio samples [samples]

        Returns:
            mel: Mel spectrogram [n_mels × n_frames]
        """
        # Number of frames
        n_frames = 1 + len(waveform) // self.hop_length

        # Simplified: return random mel spectrogram
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
    1. Two convolution layers (downsample time dimension)
    2. Sinusoidal position embeddings
    3. Transformer encoder blocks
    4. Layer normalization

    The convolutions downsample the time dimension by 4x,
    reducing the sequence length for the transformer.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8
    ):
        self.d_model = d_model

        # Convolutional layers
        self.conv1_weight = np.random.randn(d_model, n_mels, 3) * 0.02
        self.conv2_weight = np.random.randn(d_model, d_model, 3) * 0.02

        # Transformer encoder
        self.layers = []
        for _ in range(n_layers):
            self.layers.append({
                'norm1_weight': np.ones(d_model),
                'norm1_bias': np.zeros(d_model),
                'W_Q': np.random.randn(d_model, d_model) * 0.02,
                'W_K': np.random.randn(d_model, d_model) * 0.02,
                'W_V': np.random.randn(d_model, d_model) * 0.02,
                'W_O': np.random.randn(d_model, d_model) * 0.02,
                'norm2_weight': np.ones(d_model),
                'norm2_bias': np.zeros(d_model),
                'ffn1': np.random.randn(d_model, d_model * 4) * 0.02,
                'ffn2': np.random.randn(d_model * 4, d_model) * 0.02,
            })

        self.norm_weight = np.ones(d_model)
        self.norm_bias = np.zeros(d_model)

    def forward(self, mel: np.ndarray) -> np.ndarray:
        """
        Encode mel spectrogram.

        Args:
            mel: [batch × n_mels × n_frames]

        Returns:
            features: [batch × n_frames/4 × d_model]
        """
        batch, n_mels, n_frames = mel.shape

        # Simplified encoding (real implementation uses proper convolutions)
        seq_len = n_frames // 4
        x = np.random.randn(batch, seq_len, self.d_model) * 0.1

        # Transformer encoder
        for layer in self.layers:
            # Self-attention
            residual = x
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            x = layer['norm1_weight'] * (x - mean) / np.sqrt(var + 1e-6) + layer['norm1_bias']

            Q = np.matmul(x, layer['W_Q'])
            K = np.matmul(x, layer['W_K'])
            V = np.matmul(x, layer['W_V'])

            scores = np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(self.d_model)
            weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            weights = weights / np.sum(weights, axis=-1, keepdims=True)
            attn_out = np.matmul(weights, V)
            attn_out = np.matmul(attn_out, layer['W_O'])

            x = residual + attn_out

            # FFN
            residual = x
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            x = layer['norm2_weight'] * (x - mean) / np.sqrt(var + 1e-6) + layer['norm2_bias']

            x = np.maximum(0, np.matmul(x, layer['ffn1']))  # GELU
            x = np.matmul(x, layer['ffn2'])
            x = residual + x

        # Final norm
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x = self.norm_weight * (x - mean) / np.sqrt(var + 1e-6) + self.norm_bias

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
    1. Token embedding
    2. Sinusoidal position embeddings
    3. Transformer decoder blocks (with cross-attention to encoder)
    4. Layer normalization
    5. Linear projection to vocabulary

    Cross-attention allows the decoder to attend to encoder outputs,
    which represent the audio content.
    """

    def __init__(
        self,
        vocab_size: int = 51865,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 448
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding
        self.token_embed = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, d_model) * 0.02

        # Transformer decoder layers
        self.layers = []
        for _ in range(n_layers):
            self.layers.append({
                'self_attn': {
                    'W_Q': np.random.randn(d_model, d_model) * 0.02,
                    'W_K': np.random.randn(d_model, d_model) * 0.02,
                    'W_V': np.random.randn(d_model, d_model) * 0.02,
                    'W_O': np.random.randn(d_model, d_model) * 0.02,
                },
                'cross_attn': {
                    'W_Q': np.random.randn(d_model, d_model) * 0.02,
                    'W_K': np.random.randn(d_model, d_model) * 0.02,
                    'W_V': np.random.randn(d_model, d_model) * 0.02,
                    'W_O': np.random.randn(d_model, d_model) * 0.02,
                },
                'ffn1': np.random.randn(d_model, d_model * 4) * 0.02,
                'ffn2': np.random.randn(d_model * 4, d_model) * 0.02,
            })

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

        # Embedding
        x = self.token_embed[token_ids] + self.pos_embed[:seq_len]

        # Transformer decoder (simplified)
        for layer in self.layers:
            # Self-attention (causal)
            # Cross-attention to encoder
            # FFN
            pass

        # Output projection
        logits = np.matmul(x, self.output_proj)

        return logits


################################################################################
# SECTION 4: WHISPER MODEL
################################################################################

class WhisperModel:
    """
    Complete Whisper Model
    ======================

    Combines encoder and decoder for speech recognition.

    Model Sizes:
        Whisper-tiny:   39M params, 1GB VRAM
        Whisper-base:   74M params, 1GB VRAM
        Whisper-small:  244M params, 2GB VRAM
        Whisper-medium: 769M params, 5GB VRAM
        Whisper-large:  1550M params, 10GB VRAM

    Interview Questions:
        Q: "How does Whisper handle different languages?"
        A: Trained on multilingual data. Can transcribe 99 languages
           and translate to English.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        n_heads: int = 4,
        vocab_size: int = 51865
    ):
        self.mel_spec = MelSpectrogram(n_mels=n_mels)
        self.encoder = WhisperEncoder(n_mels, d_model, n_encoder_layers, n_heads)
        self.decoder = WhisperDecoder(vocab_size, d_model, n_decoder_layers, n_heads)

    def transcribe(self, waveform: np.ndarray) -> str:
        """
        Transcribe audio to text.

        Args:
            waveform: Audio samples at 16kHz

        Returns:
            text: Transcribed text
        """
        # 1. Compute mel spectrogram
        mel = self.mel_spec.forward(waveform)

        # 2. Encode audio
        encoder_output = self.encoder.forward(mel)

        # 3. Decode text (simplified)
        prompt = np.array([[50258]])  # <|startoftranscript|>
        logits = self.decoder.forward(prompt, encoder_output)

        return "Transcribed text"


################################################################################
# SECTION 5: TESTING
################################################################################

def demonstrate_whisper():
    """Demonstrate Whisper model."""
    print("=" * 70)
    print("WHISPER MODEL DEMONSTRATION")
    print("=" * 70)

    # Mel spectrogram
    print("\n--- Mel Spectrogram ---")
    mel_spec = MelSpectrogram(sample_rate=16000, n_mels=80)
    waveform = np.random.randn(16000)  # 1 second
    mel = mel_spec.forward(waveform)
    print(f"Waveform: {waveform.shape}")
    print(f"Mel: {mel.shape}")

    # Encoder
    print("\n--- Encoder ---")
    encoder = WhisperEncoder(n_mels=80, d_model=128, n_layers=2, n_heads=4)
    features = encoder.forward(mel)
    print(f"Encoder output: {features.shape}")

    # Full model
    print("\n--- Whisper Model ---")
    model = WhisperModel(d_model=128, n_encoder_layers=2, n_decoder_layers=2, n_heads=4)
    text = model.transcribe(waveform)
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

################################################################################
