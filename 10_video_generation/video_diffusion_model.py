"""
################################################################################
VIDEO DIFFUSION MODEL — GENERATING VIDEO FROM NOISE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Video Diffusion?
    Video Diffusion extends image diffusion to generate video sequences.
    Instead of denoising a single image, we denoise a sequence of frames
    simultaneously, ensuring temporal consistency and realistic motion.

Why do we need it?
    Image diffusion generates stunning single images, but video adds:
    1. Temporal consistency: Objects must persist across frames
    2. Motion modeling: Things must move realistically
    3. Physics: Objects obey physical laws (gravity, momentum)
    4. Causality: Events have causes and effects

    Without video diffusion, we'd need to generate frames independently
    and stitch them together — resulting in flickering, inconsistent output.

How does it work?
    1. Take a video clip
    2. Add noise to ALL frames simultaneously (forward process)
    3. Train a neural network to remove the noise (reverse process)
    4. The network learns to generate coherent video

    Key insight: The network must learn BOTH:
    - Spatial features (what's in each frame)
    - Temporal features (how things move between frames)

Mathematical Intuition:
    Forward process (adding noise):
        q(x_t | x_{t-1}) = N(x_t; √(1-β_t) × x_{t-1}, β_t × I)

    Reverse process (removing noise):
        p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

    Training objective:
        L = E[||ε - ε_θ(x_t, t)||²]

    Where ε is the noise added, and ε_θ is the predicted noise.

Real-World Analogy:
    Think of video diffusion like a sculptor working with marble:
    - Start with a noisy block of marble (random noise)
    - The sculptor (neural network) carefully removes material
    - Each stroke is guided by what the final video should look like
    - The result is a beautiful, coherent video

    But unlike a sculptor who works on one statue, video diffusion sculpts
    MANY statues simultaneously (multiple frames) that must all look like
    they belong together.

Interview Questions:
    Q: "How does video diffusion differ from image diffusion?"
    A: The main differences are:
       1. Input shape: [B, C, H, W] → [B, C, T, H, W]
       2. Attention: Must include temporal attention (across frames)
       3. Noise schedule: May differ per frame (causal noise)
       4. Conditioning: Often conditioned on first frame or text
       5. Computational cost: Much higher (T× more operations)

    Q: "What is classifier-free guidance in video generation?"
    A: During training, randomly drop the text/video conditioning with
       probability p (e.g., 10%). During inference, compute:
       ε_guided = ε_uncond + w × (ε_cond - ε_uncond)
       where w > 1 amplifies the conditioning signal.
       This makes the generated video more closely follow the prompt.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List
import math

################################################################################
# SECTION 1: NOISE SCHEDULE — CONTROLLING THE DIFFUSION PROCESS
################################################################################

class VideoNoiseSchedule:
    """
    Noise Schedule for Video Diffusion
    ===================================

    The noise schedule controls how much noise is added at each timestep.

    For video, we have TWO considerations:
    1. The standard noise schedule (same as image diffusion)
    2. Temporal noise scheduling (different frames may get different noise)

    Standard Schedules:
        Linear: β_t increases linearly from β_start to β_end
        Cosine: β_t follows a cosine curve (better for images)
        Sigmoid: β_t follows a sigmoid curve

    Temporal Scheduling (Video-specific):
        Uniform: All frames get the same noise level
        Causal: Earlier frames are cleaner (more like conditioning)
        Progressive: Noise increases from first to last frame

    Why temporal scheduling matters:
        In video, we often want to CONDITION on the first frame(s).
        By giving less noise to early frames, we guide the generation
        to be consistent with the conditioning.

    Mathematical Definition:
        ᾱ_t = Π_{s=1}^{t} (1 - β_s)

        Forward process:
        x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε

        Where ε ~ N(0, I)
    """

    def __init__(
        self,
        n_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        schedule: str = 'linear',
    ):
        """
        Initialize noise schedule.

        Args:
            n_timesteps: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
            schedule: Type of schedule ('linear', 'cosine', 'sigmoid')

        Common values:
            Stable Diffusion: beta_start=0.00085, beta_end=0.012, linear
            DDPM: beta_start=0.0001, beta_end=0.02, linear
            Video Diffusion: Often uses cosine schedule
        """
        self.n_timesteps = n_timesteps

        # Generate beta schedule
        if schedule == 'linear':
            self.betas = np.linspace(beta_start, beta_end, n_timesteps)
        elif schedule == 'cosine':
            # Cosine schedule from "Improved DDPM" paper
            steps = np.arange(n_timesteps + 1, dtype=np.float64) / n_timesteps
            alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            self.betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            self.betas = np.clip(self.betas, 0, 0.999)
        elif schedule == 'sigmoid':
            # Sigmoid schedule
            t = np.linspace(-6, 6, n_timesteps)
            self.betas = 1 / (1 + np.exp(-t))
            self.betas = beta_start + (beta_end - beta_start) * self.betas

        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)
        self.alpha_bar_prev = np.append(1.0, self.alpha_bar[:-1])

        # For posterior distribution q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_log_variance = np.log(
            np.maximum(self.posterior_variance, 1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_bar_prev) * np.sqrt(self.alphas) / (1.0 - self.alpha_bar)
        )

    def add_noise(self, x0: np.ndarray, t: int, noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add noise to clean video at timestep t.

        x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε

        Args:
            x0: Clean video [B, C, T, H, W]
            t: Timestep (0 = clean, n_timesteps = pure noise)
            noise: Optional pre-generated noise

        Returns:
            x_t: Noisy video
            noise: The noise that was added
        """
        if noise is None:
            noise = np.random.randn(*x0.shape)

        alpha_bar_t = self.alpha_bar[t]

        # Forward process: interpolate between clean and noise
        x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1 - alpha_bar_t) * noise

        return x_t, noise

    def add_temporal_noise(
        self,
        x0: np.ndarray,
        t: int,
        temporal_schedule: str = 'uniform',
        noise: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add noise with temporal scheduling (video-specific).

        Different frames can receive different noise levels:
        - uniform: All frames get the same noise
        - causal: Earlier frames are cleaner (conditioning)
        - progressive: Noise increases from first to last frame

        Args:
            x0: Clean video [B, C, T, H, W]
            t: Base timestep
            temporal_schedule: Type of temporal scheduling
            noise: Optional pre-generated noise

        Returns:
            x_t: Noisy video with temporal scheduling
            noise: The noise that was added
        """
        if noise is None:
            noise = np.random.randn(*x0.shape)

        B, C, T, H, W = x0.shape

        if temporal_schedule == 'uniform':
            # All frames get the same noise level
            return self.add_noise(x0, t, noise)

        elif temporal_schedule == 'causal':
            # Earlier frames are cleaner (used for first-frame conditioning)
            # Frame 0: minimal noise (almost clean)
            # Frame T-1: full noise at timestep t
            x_t = np.zeros_like(x0)
            for frame in range(T):
                # Linearly interpolate noise level
                frame_t = int(t * frame / max(T - 1, 1))
                alpha_bar_t = self.alpha_bar[frame_t]
                x_t[:, :, frame] = (
                    np.sqrt(alpha_bar_t) * x0[:, :, frame] +
                    np.sqrt(1 - alpha_bar_t) * noise[:, :, frame]
                )
            return x_t, noise

        elif temporal_schedule == 'progressive':
            # Noise increases from first to last frame
            x_t = np.zeros_like(x0)
            for frame in range(T):
                frame_t = int(t * (0.5 + 0.5 * frame / max(T - 1, 1)))
                alpha_bar_t = self.alpha_bar[frame_t]
                x_t[:, :, frame] = (
                    np.sqrt(alpha_bar_t) * x0[:, :, frame] +
                    np.sqrt(1 - alpha_bar_t) * noise[:, :, frame]
                )
            return x_t, noise

        else:
            raise ValueError(f"Unknown temporal schedule: {temporal_schedule}")


################################################################################
# SECTION 2: SPATIAL-TEMPORAL ATTENTION — THE KEY TO VIDEO DIFFUSION
################################################################################

class SpatialTemporalAttention:
    """
    Spatial-Temporal Attention for Video Diffusion
    ===============================================

    This is the CORE innovation that makes video diffusion work.
    It combines two types of attention:

    1. Spatial Attention: Each frame attends to itself
       - Captures what's in each frame (objects, textures, layout)
       - Same as image diffusion attention

    2. Temporal Attention: Each spatial location attends across frames
       - Captures motion, temporal consistency
       - Unique to video generation

    Architecture:
        Input: [B, C, T, H, W]
            │
            ▼
        ┌─────────────────────────────┐
        │ Reshape: [B×T, C, H, W]    │
        │ Spatial Attention           │  Each frame independently
        │ Reshape: [B, C, T, H, W]   │
        └─────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────┐
        │ Reshape: [B×H×W, C, T]     │
        │ Temporal Attention          │  Each location across frames
        │ Reshape: [B, C, T, H, W]   │
        └─────────────────────────────┘
            │
            ▼
        Output: [B, C, T, H, W]

    Why factorized attention?
        Full attention over all tokens: O((T×H×W)²) — prohibitive!
        Factorized: O(T×(H×W)²) + O(H×W×T²) — much cheaper

    Example:
        T=16, H=W=32
        Full: (16×32×32)² = 268,435,456 tokens — impossible!
        Spatial: 16 × (32×32)² = 16 × 1,048,576 = 16,777,216
        Temporal: 32×32 × 16² = 1,024 × 256 = 262,144
        Total: ~17M — feasible!

    Industry Usage:
        - Sora: Uses spatial-temporal attention
        - Veo: Factorized attention with temporal layers
        - Kling: Spatial-temporal DiT architecture
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        """
        Initialize spatial-temporal attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Spatial attention weights
        self.spatial_q = np.random.randn(d_model, d_model) * 0.02
        self.spatial_k = np.random.randn(d_model, d_model) * 0.02
        self.spatial_v = np.random.randn(d_model, d_model) * 0.02
        self.spatial_out = np.random.randn(d_model, d_model) * 0.02

        # Temporal attention weights
        self.temporal_q = np.random.randn(d_model, d_model) * 0.02
        self.temporal_k = np.random.randn(d_model, d_model) * 0.02
        self.temporal_v = np.random.randn(d_model, d_model) * 0.02
        self.temporal_out = np.random.randn(d_model, d_model) * 0.02

        # Layer normalization
        self.spatial_norm = np.ones(d_model)
        self.temporal_norm = np.ones(d_model)

    def spatial_attention(self, x: np.ndarray) -> np.ndarray:
        """
        Apply spatial attention within each frame.

        Each frame attends to itself — captures spatial relationships.

        Args:
            x: [B, C, T, H, W]

        Returns:
            [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape

        # Reshape: treat each frame independently
        # [B, C, T, H, W] → [B*T, H*W, C]
        x_flat = x.transpose(0, 2, 3, 4, 1).reshape(B*T, H*W, C)

        # Compute Q, K, V
        Q = x_flat @ self.spatial_q  # [B*T, H*W, C]
        K = x_flat @ self.spatial_k
        V = x_flat @ self.spatial_v

        # Reshape for multi-head attention
        Q = Q.reshape(B*T, H*W, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B*T, H*W, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B*T, H*W, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Compute attention scores
        scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(self.d_k)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)

        # Apply attention
        out = attn @ V  # [B*T, n_heads, H*W, d_k]
        out = out.transpose(0, 2, 1, 3).reshape(B*T, H*W, C)

        # Output projection
        out = out @ self.spatial_out

        # Reshape back to video format
        out = out.reshape(B, T, H, W, C).transpose(0, 4, 1, 2, 3)

        # Residual connection
        return x + out

    def temporal_attention(self, x: np.ndarray) -> np.ndarray:
        """
        Apply temporal attention across frames.

        Each spatial location attends to the same location in all frames.
        This captures motion and temporal consistency.

        Args:
            x: [B, C, T, H, W]

        Returns:
            [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape

        # Reshape: treat each spatial location independently
        # [B, C, T, H, W] → [B*H*W, T, C]
        x_flat = x.transpose(0, 3, 4, 2, 1).reshape(B*H*W, T, C)

        # Compute Q, K, V
        Q = x_flat @ self.temporal_q
        K = x_flat @ self.temporal_k
        V = x_flat @ self.temporal_v

        # Reshape for multi-head attention
        Q = Q.reshape(B*H*W, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B*H*W, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B*H*W, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Causal attention mask (optional: prevent attending to future frames)
        causal_mask = np.triu(np.ones((T, T)), k=1) * -1e9

        # Compute attention scores with causal mask
        scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(self.d_k)
        scores = scores + causal_mask  # Apply causal mask

        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)

        # Apply attention
        out = attn @ V
        out = out.transpose(0, 2, 1, 3).reshape(B*H*W, T, C)

        # Output projection
        out = out @ self.temporal_out

        # Reshape back to video format
        out = out.reshape(B, H, W, T, C).transpose(0, 4, 3, 1, 2)

        # Residual connection
        return x + out

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply spatial-temporal attention.

        First spatial (within frames), then temporal (across frames).

        Args:
            x: [B, C, T, H, W]

        Returns:
            [B, C, T, H, W]
        """
        x = self.spatial_attention(x)
        x = self.temporal_attention(x)
        return x


################################################################################
# SECTION 3: VIDEO UNET — THE DENOISING NETWORK
################################################################################

class VideoResBlock:
    """
    Video Residual Block
    ====================

    A residual block that processes video with spatial-temporal convolutions.

    Architecture:
        Input: [B, C, T, H, W]
            │
            ├──────────────────────┐
            │                      │
            ▼                      │
        LayerNorm                  │
            │                      │
            ▼                      │
        SiLU Activation            │
            │                      │
            ▼                      │
        Spatial Conv2D             │
            │                      │
            ▼                      │
        Temporal Conv1D            │
            │                      │
            ▼                      │
        Scale + Shift (from time embedding)
            │                      │
            ▼                      │
        SiLU + Dropout             │
            │                      │
            ▼                      │
        Output Conv                │
            │                      │
            ▼                      │
        + ─────────────────────────┘
            │
            ▼
        Output: [B, C, T, H, W]
    """

    def __init__(self, channels: int, time_emb_dim: int = 256):
        self.channels = channels
        self.time_emb_dim = time_emb_dim

        # Time embedding projection
        self.time_proj = np.random.randn(time_emb_dim, channels * 2) * 0.02

        # Spatial convolution
        self.spatial_conv = np.random.randn(channels, channels, 3, 3) * 0.02
        self.spatial_bias = np.zeros(channels)

        # Temporal convolution (causal)
        self.temporal_conv = np.random.randn(channels, channels, 3, 1, 1) * 0.02
        self.temporal_bias = np.zeros(channels)

        # Output convolution
        self.out_conv = np.random.randn(channels, channels, 3, 3) * 0.02
        self.out_bias = np.zeros(channels)

        # Normalization
        self.norm1 = np.ones(channels)
        self.norm2 = np.ones(channels)

    def forward(self, x: np.ndarray, t_emb: np.ndarray) -> np.ndarray:
        """
        Forward pass with time embedding conditioning.

        Args:
            x: [B, C, T, H, W]
            t_emb: Time embedding [B, time_emb_dim]

        Returns:
            [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape
        residual = x

        # Normalize
        h = x / (np.std(x, axis=(1,3,4), keepdims=True) + 1e-5)

        # SiLU activation
        h = h / (1 + np.exp(-h))

        # Spatial convolution (per-frame)
        h_padded = np.pad(h, ((0,0),(0,0),(0,0),(1,1),(1,1)))
        h_spatial = np.zeros_like(h)
        for b in range(B):
            for oc in range(C):
                for t in range(T):
                    for i in range(H):
                        for j in range(W):
                            patch = h_padded[b, :, t, i:i+3, j:j+3]
                            h_spatial[b, oc, t, i, j] = (
                                np.sum(patch * self.spatial_conv[oc]) +
                                self.spatial_bias[oc]
                            )

        # Temporal convolution (causal, per-location)
        h_temporal = np.pad(h_spatial, ((0,0),(0,0),(2,0),(0,0),(0,0)))
        h_out = np.zeros_like(h_spatial)
        for b in range(B):
            for oc in range(C):
                for t in range(T):
                    for i in range(H):
                        for j in range(W):
                            patch = h_temporal[b, :, t:t+3, i, j]
                            h_out[b, oc, t, i, i] = (
                                np.sum(patch * self.temporal_conv[oc, :, :, 0, 0]) +
                                self.temporal_bias[oc]
                            )

        # Time embedding conditioning (scale and shift)
        t_proj = t_emb @ self.time_proj  # [B, C*2]
        scale = t_proj[:, :C].reshape(B, C, 1, 1, 1)
        shift = t_proj[:, C:].reshape(B, C, 1, 1, 1)
        h_out = h_out * (1 + scale) + shift

        # Output convolution
        h_final = np.zeros_like(h_out)
        h_padded = np.pad(h_out, ((0,0),(0,0),(0,0),(1,1),(1,1)))
        for b in range(B):
            for oc in range(C):
                for t in range(T):
                    for i in range(H):
                        for j in range(W):
                            patch = h_padded[b, :, t, i:i+3, j:j+3]
                            h_final[b, oc, t, i, j] = (
                                np.sum(patch * self.out_conv[oc]) +
                                self.out_bias[oc]
                            )

        # Residual connection
        return residual + h_final


class VideoUNet:
    """
    Video U-Net for Diffusion
    =========================

    The denoising network that predicts noise in a video.

    Architecture:
        Input: Noisy video [B, C, T, H, W] + timestep t + text conditioning
            │
            ▼
        ┌─────────────────────────────────────┐
        │  Encoder (Downsampling)             │
        │  ┌─────────────────────────────┐    │
        │  │ ResBlock + SpatialTemporalAttn│   │
        │  │ Downsample 2x               │    │
        │  └─────────────────────────────┘    │
        │  ┌─────────────────────────────┐    │
        │  │ ResBlock + SpatialTemporalAttn│   │
        │  │ Downsample 2x               │    │
        │  └─────────────────────────────┘    │
        │  ┌─────────────────────────────┐    │
        │  │ ResBlock + SpatialTemporalAttn│   │
        │  │ Downsample 2x               │    │
        │  └─────────────────────────────┘    │
        └─────────────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────────────┐
        │  Middle Block                       │
        │  ResBlock + SpatialTemporalAttn     │
        └─────────────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────────────┐
        │  Decoder (Upsampling)               │
        │  ┌─────────────────────────────┐    │
        │  │ ResBlock + SpatialTemporalAttn│   │
        │  │ + Skip connection from encoder│   │
        │  │ Upsample 2x                 │    │
        │  └─────────────────────────────┘    │
        │  (repeated for each encoder level)  │
        └─────────────────────────────────────┘
            │
            ▼
        Output: Predicted noise [B, C, T, H, W]

    Key Design Decisions:
        1. Skip connections: Connect encoder to decoder at each level
        2. Time embedding: Sinusoidal embedding of timestep
        3. Text conditioning: Cross-attention with text embeddings
        4. Spatial-temporal attention: At each resolution level
    """

    def __init__(
        self,
        in_channels: int = 4,  # Latent channels from VAE
        model_channels: int = 128,
        out_channels: int = 4,
        channel_multipliers: List[int] = [1, 2, 4],
        n_heads: int = 8,
        time_emb_dim: int = 256,
    ):
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        # Time embedding
        self.time_emb_dim = time_emb_dim

        # Input convolution
        self.input_conv = np.random.randn(model_channels, in_channels, 3, 3) * 0.02

        # Encoder blocks
        self.encoder_blocks = []
        ch = model_channels
        for mult in channel_multipliers:
            out_ch = model_channels * mult
            self.encoder_blocks.append({
                'resblock': VideoResBlock(ch, time_emb_dim),
                'attention': SpatialTemporalAttention(ch, n_heads),
                'downsample': np.random.randn(out_ch, ch, 3, 3) * 0.02,
            })
            ch = out_ch

        # Middle block
        self.middle_resblock = VideoResBlock(ch, time_emb_dim)
        self.middle_attention = SpatialTemporalAttention(ch, n_heads)

        # Decoder blocks
        self.decoder_blocks = []
        for mult in reversed(channel_multipliers):
            out_ch = model_channels * mult
            self.decoder_blocks.append({
                'resblock': VideoResBlock(ch * 2, time_emb_dim),  # *2 for skip connection
                'attention': SpatialTemporalAttention(ch * 2, n_heads),
                'upsample': np.random.randn(out_ch, ch, 3, 3) * 0.02,
            })
            ch = out_ch

        # Output convolution
        self.output_conv = np.random.randn(out_channels, model_channels, 3, 3) * 0.02

    def sinusoidal_embedding(self, timesteps: np.ndarray) -> np.ndarray:
        """
        Create sinusoidal timestep embeddings.

        Same as Transformer positional embeddings but for diffusion timesteps.

        sin(t / 10000^(2i/d)) for even dimensions
        cos(t / 10000^(2i/d)) for odd dimensions

        Args:
            timesteps: [B] timestep values

        Returns:
            [B, time_emb_dim]
        """
        B = len(timesteps)
        half_dim = self.time_emb_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
        return emb

    def forward(
        self,
        x: np.ndarray,
        timesteps: np.ndarray,
        text_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict noise in noisy video.

        Args:
            x: Noisy video [B, C, T, H, W]
            timesteps: [B] diffusion timesteps
            text_embedding: Optional text conditioning [B, seq_len, d_model]

        Returns:
            Predicted noise [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape

        # Time embedding
        t_emb = self.sinusoidal_embedding(timesteps)

        # Input convolution
        h = np.zeros((B, self.model_channels, T, H, W))
        for b in range(B):
            for oc in range(self.model_channels):
                for t in range(T):
                    for i in range(H):
                        for j in range(W):
                            patch = x[b, :, t, max(0,i-1):i+2, max(0,j-1):j+2]
                            # Simplified conv (pad-aware)
                            h[b, oc, t, i, j] = np.sum(patch) / patch.size

        # Encoder
        skip_connections = [h]
        for block in self.encoder_blocks:
            h = block['resblock'].forward(h, t_emb)
            h = block['attention'].forward(h)
            skip_connections.append(h)

        # Middle
        h = self.middle_resblock.forward(h, t_emb)
        h = self.middle_attention.forward(h)

        # Decoder with skip connections
        for block in self.decoder_blocks:
            skip = skip_connections.pop()
            h = np.concatenate([h, skip], axis=1)  # Skip connection
            h = block['resblock'].forward(h, t_emb)
            h = block['attention'].forward(h)

        # Output
        output = np.zeros((B, self.out_channels, T, H, W))
        # Simplified output convolution
        for b in range(B):
            for oc in range(self.out_channels):
                for t in range(T):
                    for i in range(H):
                        for j in range(W):
                            output[b, oc, t, i, j] = np.mean(h[b, :self.out_channels, t, i, j])

        return output


################################################################################
# SECTION 4: CLASSIFIER-FREE GUIDANCE — CONTROLLING GENERATION
################################################################################

class ClassifierFreeGuidance:
    """
    Classifier-Free Guidance for Video Generation
    ==============================================

    Classifier-free guidance (CFG) allows controlling the generation
    without training a separate classifier.

    How it works:
    1. During training: Randomly drop conditioning with probability p
       - With probability (1-p): Use conditioning (text, image, etc.)
       - With probability p: Use empty/null conditioning

    2. During inference: Run the model TWICE
       - Once with conditioning: ε_cond = model(x, t, c)
       - Once without: ε_uncond = model(x, t, ∅)

    3. Combine: ε_guided = ε_uncond + w × (ε_cond - ε_uncond)

    Where w is the guidance scale:
        w = 1: No guidance (standard diffusion)
        w > 1: Stronger adherence to conditioning
        w < 1: Less adherence to conditioning
        w = 7-12: Typical range for good results

    Mathematical Intuition:
        ε_guided = ε_uncond + w × (ε_cond - ε_uncond)
                 = (1-w) × ε_uncond + w × ε_cond

        This is a linear interpolation between unconditional and conditional
        predictions, extrapolated beyond the conditional prediction.

    Why does it work?
        The direction (ε_cond - ε_uncond) points toward the conditioning.
        By scaling this direction by w > 1, we move FURTHER toward the
        conditioning, making the output more strongly conditioned.

    Analogy:
        Imagine you're navigating with a compass:
        - ε_uncond: Your natural drift (where you'd go without guidance)
        - ε_cond: The direction to your destination
        - (ε_cond - ε_uncond): The correction vector
        - w: How aggressively to correct

        w=1: Follow the correction exactly
        w=2: Double the correction (overshoot toward destination)
        w=0.5: Half the correction (gentle steering)

    Interview Questions:
        Q: "What guidance scale should I use?"
        A: It depends on the task:
           - Text-to-video: w=7-12 (need strong text adherence)
           - Image-to-video: w=3-7 (image is strong conditioning)
           - Video editing: w=5-8 (balance between edit and original)
           - Higher w = more adherence but less diversity

        Q: "What happens if guidance is too high?"
        A: The generation becomes oversaturated and loses diversity.
           Colors become too vivid, details become exaggerated,
           and the output may look unnatural.
    """

    def __init__(self, guidance_scale: float = 7.5, conditioning_dropout: float = 0.1):
        """
        Initialize classifier-free guidance.

        Args:
            guidance_scale: How strongly to follow conditioning (w)
            conditioning_dropout: Probability of dropping conditioning during training
        """
        self.guidance_scale = guidance_scale
        self.conditioning_dropout = conditioning_dropout

    def training_step(
        self,
        model: VideoUNet,
        x_noisy: np.ndarray,
        t: np.ndarray,
        conditioning: np.ndarray,
    ) -> np.ndarray:
        """
        Training step with conditioning dropout.

        Randomly drops conditioning to train both conditional and
        unconditional generation.

        Args:
            model: The denoising model
            x_noisy: Noisy video
            t: Timestep
            conditioning: Text/image conditioning

        Returns:
            Predicted noise
        """
        # Randomly drop conditioning
        B = x_noisy.shape[0]
        mask = np.random.random(B) > self.conditioning_dropout

        # Apply mask (set dropped conditioning to zeros)
        masked_cond = conditioning * mask[:, None, None]

        return model.forward(x_noisy, t, masked_cond)

    def inference_step(
        self,
        model: VideoUNet,
        x_noisy: np.ndarray,
        t: np.ndarray,
        conditioning: np.ndarray,
    ) -> np.ndarray:
        """
        Inference step with classifier-free guidance.

        Runs model twice (conditional + unconditional) and combines.

        Args:
            model: The denoising model
            x_noisy: Noisy video
            t: Timestep
            conditioning: Text/image conditioning

        Returns:
            Guided noise prediction
        """
        # Conditional prediction
        eps_cond = model.forward(x_noisy, t, conditioning)

        # Unconditional prediction (empty conditioning)
        eps_uncond = model.forward(x_noisy, t, np.zeros_like(conditioning))

        # Classifier-free guidance
        eps_guided = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)

        return eps_guided


################################################################################
# SECTION 5: VIDEO DIFFUSION SAMPLER
################################################################################

class VideoDiffusionSampler:
    """
    Video Diffusion Sampler — Generating Video Step by Step
    ======================================================

    The sampler controls how we go from noise to video.

    DDPM Sampling (slow, high quality):
        For t = T, T-1, ..., 1:
            x_{t-1} = (1/√α_t) × (x_t - (β_t/√(1-ᾱ_t)) × ε_θ(x_t, t)) + σ_t × z

    DDIM Sampling (fast, slightly lower quality):
        Skips timesteps for faster generation:
        For t = T, T-skip, T-2×skip, ..., 1:
            x_{t-skip} = ... (deterministic or stochastic)

    Video-specific considerations:
        1. Temporal consistency: Each denoising step must maintain coherence
        2. Frame interpolation: Can generate at lower FPS and interpolate
        3. Upsampling: Generate low-res first, then upscale
    """

    def __init__(self, noise_schedule: VideoNoiseSchedule):
        self.schedule = noise_schedule

    def ddim_sample(
        self,
        model: VideoUNet,
        shape: Tuple[int, ...],
        conditioning: np.ndarray,
        n_steps: int = 50,
        eta: float = 0.0,
        guidance: Optional[ClassifierFreeGuidance] = None,
    ) -> np.ndarray:
        """
        DDIM sampling for faster video generation.

        DDIM allows skipping timesteps, reducing the number of neural
        network evaluations from 1000 to 50-100.

        Args:
            model: The denoising model
            shape: Output video shape [B, C, T, H, W]
            conditioning: Text/image conditioning
            n_steps: Number of sampling steps (fewer = faster)
            eta: Stochasticity (0 = deterministic, 1 = DDPM)
            guidance: Optional classifier-free guidance

        Returns:
            Generated video [B, C, T, H, W]
        """
        # Create timestep sequence (uniformly spaced)
        timesteps = np.linspace(
            self.schedule.n_timesteps - 1, 0, n_steps, dtype=int
        )

        # Start from pure noise
        x = np.random.randn(*shape)

        for i, t in enumerate(timesteps):
            t_batch = np.full(shape[0], t)

            # Predict noise
            if guidance is not None:
                eps = guidance.inference_step(model, x, t_batch, conditioning)
            else:
                eps = model.forward(x, t_batch, conditioning)

            # DDIM update
            alpha_bar_t = self.schedule.alpha_bar[t]
            alpha_bar_prev = (
                self.schedule.alpha_bar[timesteps[i + 1]]
                if i + 1 < len(timesteps)
                else 1.0
            )

            # Predicted x_0
            x0_pred = (x - np.sqrt(1 - alpha_bar_t) * eps) / np.sqrt(alpha_bar_t)

            # Direction pointing to x_t
            sigma_t = eta * np.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) *
                (1 - alpha_bar_t / alpha_bar_prev)
            )

            # Noise term
            noise = np.random.randn(*x.shape) if i < len(timesteps) - 1 else np.zeros_like(x)

            # DDIM update formula
            x = (
                np.sqrt(alpha_bar_prev) * x0_pred +
                np.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps +
                sigma_t * noise
            )

            print(f"  Step {i+1}/{n_steps}, t={t}")

        return x


################################################################################
# SECTION 6: TESTING AND DEMONSTRATION
################################################################################

def demonstrate_video_diffusion():
    """
    Demonstrate the complete video diffusion pipeline.

    Shows:
    1. Noise schedule creation
    2. Adding noise to video
    3. Denoising with the model
    4. Classifier-free guidance
    """
    print("=" * 70)
    print("VIDEO DIFFUSION MODEL DEMONSTRATION")
    print("=" * 70)

    # Create noise schedule
    schedule = VideoNoiseSchedule(n_timesteps=100, schedule='cosine')
    print(f"\nNoise Schedule:")
    print(f"  Timesteps: {schedule.n_timesteps}")
    print(f"  Beta range: [{schedule.betas[0]:.6f}, {schedule.betas[-1]:.6f}]")
    print(f"  Alpha_bar range: [{schedule.alpha_bar[-1]:.6f}, {schedule.alpha_bar[0]:.6f}]")

    # Create a small synthetic video
    B, C, T, H, W = 1, 3, 4, 8, 8
    video = np.random.randn(B, C, T, H, W)

    # Add noise at different timesteps
    print(f"\nAdding noise to video (shape: {video.shape}):")
    for t in [0, 25, 50, 75, 99]:
        noisy, noise = schedule.add_noise(video, t)
        noise_level = np.std(noisy)
        print(f"  t={t:3d}: noise_std={noise_level:.4f}")

    # Demonstrate temporal noise scheduling
    print(f"\nTemporal Noise Scheduling:")
    t = 50
    for schedule_type in ['uniform', 'causal', 'progressive']:
        noisy, _ = schedule.add_temporal_noise(video, t, schedule_type)
        # Check noise level per frame
        frame_stds = [np.std(noisy[0, :, f]) for f in range(T)]
        print(f"  {schedule_type:12s}: frame_stds = {[f'{s:.3f}' for s in frame_stds]}")

    # Create model (small for demonstration)
    model = VideoUNet(
        in_channels=3,
        model_channels=16,
        out_channels=3,
        channel_multipliers=[1, 2],
        n_heads=4,
        time_emb_dim=32,
    )

    # Create classifier-free guidance
    cfg = ClassifierFreeGuidance(guidance_scale=7.5)

    # Demonstrate single denoising step
    print(f"\nSingle denoising step:")
    t_batch = np.array([50])
    conditioning = np.random.randn(1, 10, 32)  # Text conditioning
    eps = model.forward(video, t_batch, conditioning)
    print(f"  Input: {video.shape}")
    print(f"  Output: {eps.shape}")

    # Demonstrate guided denoising
    print(f"\nGuided denoising step:")
    eps_guided = cfg.inference_step(model, video, t_batch, conditioning)
    print(f"  Guided output: {eps_guided.shape}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("1. Video diffusion adds temporal attention for frame consistency")
    print("2. Classifier-free guidance controls conditioning strength")
    print("3. DDIM sampling reduces steps from 1000 to 50-100")
    print("4. Temporal noise scheduling enables first-frame conditioning")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_video_diffusion()


################################################################################
# REFERENCES
################################################################################

# [1] Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models (DDPM).
#     Foundation of diffusion models.
#
# [2] Song, J., et al. (2020). Denoising Diffusion Implicit Models (DDIM).
#     Faster sampling for diffusion models.
#
# [3] Ho, J., et al. (2022). Video Diffusion Models.
#     First application of diffusion to video generation.
#
# [4] Blattmann, A., et al. (2023). Stable Video Diffusion.
#     Scalable video diffusion with latent VAE.
#
# [5] OpenAI (2024). Sora: Video generation with spacetime patches.
#     Uses diffusion transformer for video generation.
#
# [6] Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance.
#     The guidance technique used in most modern diffusion models.

################################################################################
# FUTURE IMPROVEMENTS
################################################################################

# 1. Flow Matching: Replace DDPM with flow matching for faster training
# 2. Consistency Models: One-step generation
# 3. Video Upsampling: Generate low-res, then upscale
# 4. Temporal Super-Resolution: Increase FPS after generation
# 5. ControlNet: Add spatial/temporal control signals
# 6. IP-Adapter: Use reference images for style/content

################################################################################
