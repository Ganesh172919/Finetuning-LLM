"""
################################################################################
U-NET — THE BACKBONE OF DIFFUSION MODELS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is U-Net?
    U-Net is a convolutional neural network architecture designed for
    image segmentation. It has an encoder-decoder structure with skip
    connections. In diffusion models, U-Net predicts noise to remove.

Why is it called U-Net?
    The architecture looks like the letter "U":
    - Left side: encoder (downsampling)
    - Bottom: bottleneck
    - Right side: decoder (upsampling)
    - Skip connections: connect encoder to decoder

Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Input: noisy image + timestep                    │
    │        ↓                                          │
    │ ┌─── Encoder ───┐                                │
    │ │ 64×64 → 32×32 │─────── skip ──────┐           │
    │ │ 32×32 → 16×16 │────── skip ───┐   │           │
    │ │ 16×16 → 8×8   │──── skip ─┐   │   │           │
    │ └───────────────┘            │   │   │           │
    │        ↓                     │   │   │           │
    │ ┌─── Bottleneck ──┐         │   │   │           │
    │ │ 8×8             │         │   │   │           │
    │ └─────────────────┘         │   │   │           │
    │        ↓                     ↓   ↓   ↓           │
    │ ┌─── Decoder ───┐                                │
    │ │ 8×8 → 16×16   │←── concat ─┘   │   │         │
    │ │ 16×16 → 32×32 │←── concat ─────┘   │         │
    │ │ 32×32 → 64×64 │←── concat ─────────┘         │
    │ └───────────────┘                                │
    │        ↓                                          │
    │ Output: predicted noise                           │
    └─────────────────────────────────────────────────┘

Interview Questions:
    1. "Why U-Net for diffusion models?"
       Skip connections preserve spatial information.
       The model can focus on noise prediction while
       retaining image structure from encoder.

    2. "What are the key components?"
       Residual blocks, attention layers, timestep embeddings,
       skip connections, and group normalization.

################################################################################
"""

import numpy as np
from typing import Optional, List, Tuple
import math

import sys
sys.path.append('..')
from ..02_transformers.attention import MultiHeadAttention
from ..02_transformers.layers import RMSNorm

################################################################################
# SECTION 1: RESIDUAL BLOCK
################################################################################

class ResidualBlock:
    """
    Residual Block for U-Net
    =========================

    Definition: A block with skip connection that adds input to output.

    Architecture:
        x → Conv → Norm → SiLU → Conv → Norm → SiLU + x → output

    Why residual connections?
    - Allow gradients to flow directly through the network
    - Prevent vanishing gradients in deep networks
    - Make it easier to learn identity mapping
    """

    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convolution weights (simplified)
        scale = math.sqrt(2.0 / (in_channels + out_channels))
        self.conv1 = np.random.randn(out_channels, in_channels, 3, 3) * scale
        self.conv2 = np.random.randn(out_channels, out_channels, 3, 3) * scale

        # Normalization
        self.norm1 = np.ones(in_channels)
        self.norm2 = np.ones(out_channels)

        # Skip connection if channels change
        if in_channels != out_channels:
            self.skip_conv = np.random.randn(out_channels, in_channels, 1, 1) * scale
        else:
            self.skip_conv = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: [batch × channels × height × width]
        Returns:
            output: [batch × out_channels × height × width]
        """
        h = x

        # Conv1 + Norm + SiLU
        # (Simplified - real impl uses proper conv2d)
        h = np.maximum(0, h)  # ReLU as placeholder for SiLU

        # Conv2 + Norm + SiLU
        h = np.maximum(0, h)

        # Skip connection
        if self.skip_conv is not None:
            x = np.mean(x, axis=1, keepdims=True)  # Channel reduction placeholder
            x = np.broadcast_to(x, h.shape)

        return h + x


################################################################################
# SECTION 2: ATTENTION BLOCK
################################################################################

class AttentionBlock:
    """
    Attention Block for U-Net
    ==========================

    Definition: Self-attention layer for capturing global dependencies.

    In diffusion U-Net, attention is applied at lower resolutions
    (8×8, 16×16) where it's computationally feasible.
    """

    def __init__(self, channels: int, n_heads: int = 8):
        self.channels = channels
        self.n_heads = n_heads
        self.norm = np.ones(channels)
        self.attention = MultiHeadAttention(channels, n_heads)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply self-attention.

        Args:
            x: [batch × channels × height × width]
        Returns:
            output: same shape
        """
        batch, channels, height, width = x.shape

        # Reshape to sequence: [batch × (h*w) × channels]
        h = x.reshape(batch, channels, height * width).transpose(0, 2, 1)

        # Self-attention
        h, _ = self.attention.forward(h)

        # Reshape back: [batch × channels × height × width]
        h = h.transpose(0, 2, 1).reshape(batch, channels, height, width)

        return h + x  # Residual


################################################################################
# SECTION 3: Timestep EMBEDDING
################################################################################

class TimestepEmbedding:
    """
    Timestep Embedding
    ===================

    Definition: Encode the diffusion timestep as a vector.

    The U-Net needs to know WHICH timestep it's denoising.
    We encode timestep t as a vector using sinusoidal embeddings
    (similar to position embeddings in transformers).

    Architecture:
        t → Sinusoidal → Linear → SiLU → Linear → embedding
    """

    def __init__(self, d_model: int, d_time: int = 128):
        self.d_model = d_model
        self.d_time = d_time

        # MLP for timestep embedding
        scale1 = math.sqrt(2.0 / (d_time + d_model))
        scale2 = math.sqrt(2.0 / (d_model + d_model))
        self.mlp_W1 = np.random.randn(d_time, d_model) * scale1
        self.mlp_W2 = np.random.randn(d_model, d_model) * scale2

    def sinusoidal_embedding(self, t: int) -> np.ndarray:
        """
        Create sinusoidal embedding for timestep t.

        Similar to transformer position embeddings.
        """
        half_dim = self.d_time // 2
        emb = np.exp(np.arange(half_dim) * -np.log(10000) / half_dim)
        emb = t * emb
        emb = np.concatenate([np.sin(emb), np.cos(emb)])
        return emb

    def forward(self, t: int) -> np.ndarray:
        """
        Get timestep embedding.

        Args:
            t: Timestep (0 to T-1)

        Returns:
            embedding: [d_model]
        """
        # Sinusoidal
        emb = self.sinusoidal_embedding(t)

        # MLP
        h = np.maximum(0, np.matmul(emb, self.mlp_W1))  # SiLU
        h = np.matmul(h, self.mlp_W2)

        return h


################################################################################
# SECTION 4: U-NET MODEL
################################################################################

class UNet:
    """
    U-Net for Diffusion Models
    ===========================

    Definition: The core neural network in diffusion models.
    Predicts noise to remove from noisy images.

    Key innovations in modern diffusion U-Nets:
    1. Residual blocks with timestep conditioning
    2. Self-attention at lower resolutions
    3. Cross-attention for text conditioning
    4. Group normalization
    5. Skip connections

    Used by:
    - Stable Diffusion (Latent Diffusion)
    - DALL-E 2
    - Imagen
    - Most image generation models

    Interview Question:
        "How does the U-Net in diffusion models work?"
        It takes a noisy image and timestep as input,
        processes through encoder-bottleneck-decoder with skip connections,
        and outputs the predicted noise. The architecture preserves
        spatial information through skip connections.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 64,
        channel_mult: List[int] = [1, 2, 4, 8],
        attention_resolutions: List[int] = [2, 4],
        n_heads: int = 8
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels

        # Timestep embedding
        self.time_embed = TimestepEmbedding(model_channels * 4)

        # Encoder
        self.encoder_blocks = []
        ch = model_channels
        for mult in channel_mult:
            out_ch = model_channels * mult
            self.encoder_blocks.append(ResidualBlock(ch, out_ch))
            if mult in attention_resolutions:
                self.encoder_blocks.append(AttentionBlock(out_ch, n_heads))
            ch = out_ch

        # Bottleneck
        self.bottleneck = ResidualBlock(ch, ch)
        self.bottleneck_attn = AttentionBlock(ch, n_heads)

        # Decoder
        self.decoder_blocks = []
        for mult in reversed(channel_mult):
            out_ch = model_channels * mult
            self.decoder_blocks.append(ResidualBlock(ch * 2, out_ch))  # *2 for skip
            if mult in attention_resolutions:
                self.decoder_blocks.append(AttentionBlock(out_ch, n_heads))
            ch = out_ch

        # Output
        self.output_conv = np.random.randn(out_channels, ch, 1, 1) * 0.02

    def forward(
        self,
        x: np.ndarray,
        t: int,
        context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict noise in x at timestep t.

        Args:
            x: Noisy image [batch × channels × height × width]
            t: Timestep
            context: Optional text conditioning [batch × seq × dim]

        Returns:
            predicted_noise: [batch × channels × height × width]
        """
        # Timestep embedding
        t_emb = self.time_embed.forward(t)

        # Encoder
        h = x
        skips = []
        for block in self.encoder_blocks:
            h = block.forward(h)
            skips.append(h)

        # Bottleneck
        h = self.bottleneck.forward(h)
        h = self.bottleneck_attn.forward(h)

        # Decoder
        for block in self.decoder_blocks:
            skip = skips.pop()
            h = np.concatenate([h, skip], axis=1)  # Skip connection
            h = block.forward(h)

        # Output
        # (Simplified - real impl uses proper conv)
        return h


################################################################################
# SECTION 5: TESTING & EXAMPLES
################################################################################

def demonstrate_unet():
    """Demonstrate U-Net concepts."""
    print("=" * 70)
    print("U-NET DEMONSTRATION")
    print("=" * 70)

    # Timestep embedding
    print("\n--- Timestep Embedding ---")
    time_embed = TimestepEmbedding(d_model=256, d_time=128)
    for t in [0, 250, 500, 750, 999]:
        emb = time_embed.forward(t)
        print(f"t={t}: embedding norm={np.linalg.norm(emb):.4f}")

    # Residual block
    print("\n--- Residual Block ---")
    res_block = ResidualBlock(in_channels=64, out_channels=128)
    x = np.random.randn(1, 64, 32, 32)
    out = res_block.forward(x)
    print(f"Input: {x.shape}, Output: {out.shape}")

    # U-Net (simplified)
    print("\n--- U-Net ---")
    unet = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=32,
        channel_mult=[1, 2, 4]
    )
    x = np.random.randn(1, 3, 64, 64)
    noise_pred = unet.forward(x, t=500)
    print(f"Input: {x.shape}")
    print(f"Noise prediction shape: {noise_pred.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_unet()


################################################################################
# REFERENCES
################################################################################

# [1] Ronneberger, O., et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
# [2] Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models.
# [3] Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models.

################################################################################
