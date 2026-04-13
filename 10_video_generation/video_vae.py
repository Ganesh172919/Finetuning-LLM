"""
################################################################################
VIDEO VAE — LEARNING LATENT REPRESENTATIONS OF VIDEO
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Video VAE?
    A Video Variational Autoencoder learns to compress video into a compact
    latent representation and reconstruct it back. Unlike image VAEs that
    handle single frames, Video VAEs must capture both spatial AND temporal
    correlations across frames.

Why do we need it?
    Raw video is extremely high-dimensional:
    - 1 second of 720p video at 30fps = 30 × 1280 × 720 × 3 = ~83 million values
    - 10 seconds = ~830 million values

    Processing this directly in pixel space is:
    1. Computationally prohibitive (quadratic attention in diffusion)
    2. Memory-intensive (can't fit in GPU memory)
    3. Redundant (adjacent frames share 95%+ of information)

    A Video VAE compresses this to a latent space that is:
    - 4-16x smaller spatially (like image VAE)
    - 4-8x smaller temporally (unique to video)
    - Total: 64-128x compression while preserving visual quality

What problem does it solve?
    The "curse of dimensionality" in video generation:
    - Without VAE: Diffusion operates on ~830M values per 10s clip
    - With VAE: Diffusion operates on ~6-13M values per 10s clip
    This makes video generation feasible on modern GPUs.

Historical Evolution:
    2013: VAE introduced (Kingma & Welling)
    2015: Convolutional VAE
    2016: Video Prediction with VAE (Srivastava et al.)
    2020: NVAE — Neural Video VAE
    2022: Latent Video Diffusion (Video LDM)
    2023: Stable Video Diffusion uses Video VAE
    2024: Sora uses spacetime patches (implicit VAE)
    2025: Hunyuan Video, Wan use 3D Causal VAE
    2026: State-of-the-art uses factorized spatial-temporal VAE

Real-World Analogy:
    Think of a Video VAE like a video codec (H.264/H.265):
    - Encoder: Takes raw video → compressed stream
    - Decoder: Takes compressed stream → reconstructed video
    - The "latent space" is like the compressed bitstream
    - Just as H.265 achieves 100:1 compression, VAE achieves 64-128x

    But unlike traditional codecs that use hand-crafted rules (motion vectors,
    DCT transforms), a VAE LEARNS the optimal compression from data.

Interview Question:
    Q: "Why can't we just use an image VAE frame-by-frame for video?"
    A: Image VAE treats each frame independently. This causes:
       1. Temporal inconsistency: latent codes for adjacent frames may be
          completely different even though the frames are nearly identical
       2. Lost temporal information: motion, object trajectories, causality
       3. Flickering: decoded frames may flicker because the VAE doesn't
          enforce temporal smoothness

       A Video VAE adds temporal layers (3D convolutions, temporal attention)
       that explicitly model relationships between frames.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List
import math

################################################################################
# SECTION 1: 3D CONVOLUTION — THE BUILDING BLOCK
################################################################################

class Conv3D:
    """
    3D Convolution for Video Processing
    ====================================

    A 3D convolution slides a 3D kernel across spatial AND temporal dimensions.

    Image Convolution (2D):
        Kernel: [height × width]
        Slides over: [H × W]
        Captures: spatial features (edges, textures)

    Video Convolution (3D):
        Kernel: [depth × height × width]  (depth = temporal)
        Slides over: [T × H × W]
        Captures: spatiotemporal features (motion, temporal edges)

    Mathematical Definition:
        Output[t, i, j] = Σ_{dt, di, dj} Kernel[dt, di, dj] × Input[t+dt, i+di, j+dj] + bias

    Why 3D Convolution?
        Adjacent video frames are highly correlated. A 3D kernel can learn:
        - Spatial patterns (what's in each frame)
        - Temporal patterns (how things move between frames)
        - Spatiotemporal patterns (how spatial features evolve over time)

    Architecture Choice:
        Video VAEs typically use FACTORIZED convolutions:
        - 2D spatial conv within each frame (efficient)
        - 1D temporal conv across frames (captures motion)
        This is much more parameter-efficient than full 3D conv.

    Industry Usage:
        - CogVideo: Uses 3D causal convolution
        - Hunyuan Video: Factorized spatial-temporal convolution
        - Wan: 3D convolution with causal temporal padding
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (1, 1, 1),
    ):
        """
        Initialize 3D convolution layer.

        Args:
            in_channels: Number of input channels (e.g., 3 for RGB)
            out_channels: Number of output channels (feature maps)
            kernel_size: (temporal, height, width) of the kernel
            stride: (temporal, height, width) stride
            padding: (temporal, height, width) padding

        Note on kernel_size:
            (3, 3, 3) means the kernel looks at 3 consecutive frames
            and a 3×3 spatial region in each frame.

            A (1, 3, 3) kernel is equivalent to a 2D convolution applied
            independently to each frame — no temporal modeling.

            A (3, 1, 1) kernel only models temporal evolution at a single
            spatial location — useful for factorized architectures.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights using He initialization
        # He init is designed for ReLU activations: Var(W) = 2 / fan_in
        fan_in = in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
        self.weight = np.random.randn(
            out_channels, in_channels, *kernel_size
        ) * np.sqrt(2.0 / fan_in)
        self.bias = np.zeros(out_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply 3D convolution to video tensor.

        Args:
            x: Input video tensor [batch, channels, time, height, width]

        Returns:
            Output feature map [batch, out_channels, time', height', width']

        Computational Complexity:
            O(B × C_out × T' × H' × W' × C_in × kt × kh × kw)

            For a typical video VAE layer:
            B=1, C_out=128, T'=16, H'=64, W'=64, C_in=128, k=(3,3,3)
            ≈ 128 × 16 × 64 × 64 × 128 × 27 ≈ 34.4 billion operations

            This is why factorized convolutions are preferred!
        """
        batch, channels, time, height, width = x.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        pt, ph, pw = self.padding

        # Apply padding
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (pt, pt), (ph, ph), (pw, pw)),
            mode='constant'
        )

        # Calculate output dimensions
        out_t = (time + 2 * pt - kt) // st + 1
        out_h = (height + 2 * ph - kh) // sh + 1
        out_w = (width + 2 * pw - kw) // sw + 1

        # Initialize output
        output = np.zeros((batch, self.out_channels, out_t, out_h, out_w))

        # Perform convolution (simplified loop for clarity)
        # In production, use im2col + GEMM or specialized CUDA kernels
        for b in range(batch):
            for oc in range(self.out_channels):
                for t in range(out_t):
                    for i in range(out_h):
                        for j in range(out_w):
                            # Extract the 3D patch
                            t_start = t * st
                            i_start = i * sh
                            j_start = j * sw

                            patch = x_padded[
                                b, :,
                                t_start:t_start+kt,
                                i_start:i_start+kh,
                                j_start:j_start+kw
                            ]

                            # Element-wise multiply and sum
                            output[b, oc, t, i, j] = (
                                np.sum(patch * self.weight[oc]) + self.bias[oc]
                            )

        return output


class CausalConv3D(Conv3D):
    """
    Causal 3D Convolution
    =====================

    A causal convolution only looks at current and PAST frames, never future.
    This is essential for:
    1. Autoregressive generation (can't see future)
    2. Streaming applications (process frames as they arrive)
    3. Physical consistency (causality — effects follow causes)

    How it works:
        Regular padding: [pad_left, pad_right] on temporal axis
        Causal padding:  [pad_left, 0] on temporal axis (no future padding)

    Example:
        kernel_size = (3, 3, 3), padding = (1, 1, 1)
        Regular: looks at frames [t-1, t, t+1]
        Causal: looks at frames [t-2, t-1, t]  (only past + present)

    Industry Usage:
        - CogVideoX: Uses causal 3D convolution
        - Hunyuan Video: Causal temporal convolution
        - Wan: Causal VAE for temporal consistency
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
    ):
        # For causal convolution, we only pad the LEFT side temporally
        # This ensures the output at time t only depends on inputs at time ≤ t
        temporal_pad = kernel_size[0] - 1  # Full left padding, no right padding
        spatial_pad = kernel_size[1] // 2  # Standard spatial padding

        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding=(temporal_pad, spatial_pad, spatial_pad)
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply causal 3D convolution.

        The key difference from regular Conv3D:
        We pad ONLY on the left (past) side of the temporal dimension.
        This ensures temporal causality.

        Args:
            x: [batch, channels, time, height, width]

        Returns:
            [batch, out_channels, time', height, width]
        """
        batch, channels, time, height, width = x.shape
        kt, kh, kw = self.kernel_size
        pt, ph, pw = self.padding

        # Causal padding: only pad LEFT side temporally
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (pt, 0), (ph, ph), (pw, pw)),
            mode='constant'
        )

        # Standard convolution on causally-padded input
        out_t = time  # Causal conv preserves temporal dimension
        out_h = (height + 2 * ph - kh) // self.stride[1] + 1
        out_w = (width + 2 * pw - kw) // self.stride[2] + 1

        output = np.zeros((batch, self.out_channels, out_t, out_h, out_w))

        for b in range(batch):
            for oc in range(self.out_channels):
                for t in range(out_t):
                    for i in range(out_h):
                        for j in range(out_w):
                            t_start = t * self.stride[0]
                            i_start = i * self.stride[1]
                            j_start = j * self.stride[2]

                            patch = x_padded[
                                b, :,
                                t_start:t_start+kt,
                                i_start:i_start+kh,
                                j_start:j_start+kw
                            ]

                            output[b, oc, t, i, j] = (
                                np.sum(patch * self.weight[oc]) + self.bias[oc]
                            )

        return output


################################################################################
# SECTION 2: FACTORIZED SPATIAL-TEMPORAL BLOCKS
################################################################################

class SpatialTemporalBlock:
    """
    Factorized Spatial-Temporal Processing Block
    =============================================

    Instead of full 3D convolution (expensive), we FACTORIZE into:
    1. Spatial block: 2D convolution within each frame
    2. Temporal block: 1D convolution across frames at each spatial location

    This is the key insight behind modern Video VAEs!

    Full 3D Conv:
        Kernel: [T × H × W] parameters
        Complexity: O(T × H × W) per output element

    Factorized:
        Spatial kernel: [1 × H × W] parameters
        Temporal kernel: [T × 1 × 1] parameters
        Total: H×W + T parameters (much less than T×H×W)

    Example:
        Full 3D: 3 × 3 × 3 = 27 parameters per channel pair
        Factorized: 1 × 3 × 3 + 3 × 1 × 1 = 9 + 3 = 12 parameters per channel pair
        Savings: 56% fewer parameters, similar quality!

    Architecture Diagram:
        Input: [B, C, T, H, W]
            │
            ▼
        ┌───────────────────────┐
        │  Spatial Conv2D       │  Process each frame independently
        │  [1, 3, 3] kernel     │  Captures edges, textures, objects
        └───────────────────────┘
            │
            ▼
        ┌───────────────────────┐
        │  Temporal Conv1D      │  Process each spatial location across time
        │  [3, 1, 1] kernel     │  Captures motion, temporal evolution
        └───────────────────────┘
            │
            ▼
        Output: [B, C', T, H, W]

    Industry Usage:
        - Video LDM: Factorized spatial-temporal layers
        - Stable Video Diffusion: Factorized architecture
        - CogVideo: Factorized 3D blocks
    """

    def __init__(self, channels: int, spatial_kernel: int = 3, temporal_kernel: int = 3):
        """
        Initialize factorized spatial-temporal block.

        Args:
            channels: Number of feature channels
            spatial_kernel: Size of spatial convolution kernel
            temporal_kernel: Size of temporal convolution kernel
        """
        self.channels = channels

        # Spatial convolution: processes each frame independently
        # Shape: [C_out, C_in, 1, spatial_kernel, spatial_kernel]
        # The '1' in temporal dimension means "apply per-frame"
        spatial_fan_in = channels * spatial_kernel * spatial_kernel
        self.spatial_weight = np.random.randn(
            channels, channels, 1, spatial_kernel, spatial_kernel
        ) * np.sqrt(2.0 / spatial_fan_in)
        self.spatial_bias = np.zeros(channels)

        # Temporal convolution: processes each spatial location across time
        # Shape: [C_out, C_in, temporal_kernel, 1, 1]
        # The '1's in spatial dimensions mean "apply per-location"
        temporal_fan_in = channels * temporal_kernel
        self.temporal_weight = np.random.randn(
            channels, channels, temporal_kernel, 1, 1
        ) * np.sqrt(2.0 / temporal_fan_in)
        self.temporal_bias = np.zeros(channels)

        # Normalization layers (simplified)
        self.spatial_norm_weight = np.ones(channels)
        self.spatial_norm_bias = np.zeros(channels)
        self.temporal_norm_weight = np.ones(channels)
        self.temporal_norm_bias = np.zeros(channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply factorized spatial-temporal processing.

        Args:
            x: [batch, channels, time, height, width]

        Returns:
            [batch, channels, time, height, width]
        """
        batch, C, T, H, W = x.shape
        sk = self.spatial_weight.shape[-1]
        tk = self.temporal_weight.shape[2]
        spad = sk // 2
        tpad = tk // 2

        # Step 1: Spatial convolution (per-frame)
        x_spatial = np.pad(x, ((0,0),(0,0),(0,0),(spad,spad),(spad,spad)))
        spatial_out = np.zeros_like(x)

        for b in range(batch):
            for oc in range(C):
                for t in range(T):
                    for i in range(H):
                        for j in range(W):
                            patch = x_spatial[b, :, t, i:i+sk, j:j+sk]
                            spatial_out[b, oc, t, i, j] = (
                                np.sum(patch * self.spatial_weight[oc, :, 0]) +
                                self.spatial_bias[oc]
                            )

        # Apply spatial normalization (simplified LayerNorm)
        mean = spatial_out.mean(axis=(1,3,4), keepdims=True)
        std = spatial_out.std(axis=(1,3,4), keepdims=True) + 1e-5
        spatial_out = (spatial_out - mean) / std
        spatial_out = spatial_out * self.spatial_norm_weight[None, :, None, None, None] + \
                      self.spatial_norm_bias[None, :, None, None, None]

        # Apply activation (SiLU/Swish: x * sigmoid(x))
        spatial_out = spatial_out / (1 + np.exp(-spatial_out))

        # Step 2: Temporal convolution (per-location)
        x_temporal = np.pad(spatial_out, ((0,0),(0,0),(tpad,0),(0,0),(0,0)))  # Causal padding
        temporal_out = np.zeros_like(spatial_out)

        for b in range(batch):
            for oc in range(C):
                for t in range(T):
                    for i in range(H):
                        for j in range(W):
                            patch = x_temporal[b, :, t:t+tk, i, j]
                            temporal_out[b, oc, t, i, j] = (
                                np.sum(patch * self.temporal_weight[oc, :, :, 0, 0]) +
                                self.temporal_bias[oc]
                            )

        # Apply temporal normalization
        mean = temporal_out.mean(axis=(1,3,4), keepdims=True)
        std = temporal_out.std(axis=(1,3,4), keepdims=True) + 1e-5
        temporal_out = (temporal_out - mean) / std
        temporal_out = temporal_out * self.temporal_norm_weight[None, :, None, None, None] + \
                       self.temporal_norm_bias[None, :, None, None, None]

        # Apply activation
        temporal_out = temporal_out / (1 + np.exp(-temporal_out))

        # Residual connection
        return x + temporal_out


################################################################################
# SECTION 3: VIDEO VAE ENCODER
################################################################################

class VideoVAEEncoder:
    """
    Video VAE Encoder — Compressing Video to Latent Space
    =====================================================

    The encoder takes a video clip and compresses it into a compact latent
    representation. This is the first half of the Video VAE.

    Architecture Overview:
        Input Video: [B, 3, T, H, W]  (e.g., 16 frames of 256×256 RGB)
            │
            ▼
        ┌─────────────────────────────┐
        │  Initial Conv3D             │  3 → 128 channels
        │  Downsample spatially 2x    │  256×256 → 128×128
        └─────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────┐
        │  Spatial-Temporal Block 1   │  128 channels
        │  + Downsample 2x            │  128×128 → 64×64
        └─────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────┐
        │  Spatial-Temporal Block 2   │  256 channels
        │  + Downsample 2x            │  64×64 → 32×32
        └─────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────┐
        │  Spatial-Temporal Block 3   │  512 channels
        │  + Temporal downsample 2x   │  T → T/2
        └─────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────┐
        │  Output Conv                │  → 2 × latent_channels
        │  (mean + log_variance)      │  For reparameterization trick
        └─────────────────────────────┘
            │
            ▼
        Latent: [B, latent_dim, T/2, H/8, W/8]

    Compression Ratios (typical):
        Spatial: 8x (3 downsamples of 2x each)
        Temporal: 2x (1 temporal downsample)
        Channel: 3 → 16 latent channels
        Total: 8 × 8 × 2 × (3/16) ≈ 24x compression

    Interview Questions:
        Q: "Why downsample temporally in a Video VAE?"
        A: Video has massive temporal redundancy. Adjacent frames share 95%+
           of their content. Temporal downsampling removes this redundancy.
           The latent space only needs to encode the CHANGES between frames.

        Q: "Why use mean + log_variance instead of just mean?"
        A: This enables the reparameterization trick for training with
           backpropagation. During training, we sample: z = mean + std * noise
           During inference, we just use the mean.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: List[int] = [1, 2, 4],
        spatial_downsample: int = 3,  # 2^3 = 8x spatial
        temporal_downsample: int = 1,  # 2^1 = 2x temporal
    ):
        """
        Initialize Video VAE Encoder.

        Args:
            in_channels: Input video channels (3 for RGB)
            latent_channels: Dimension of latent space
            base_channels: Base number of feature channels
            channel_multipliers: Channel multiplier for each level
            spatial_downsample: Number of spatial downsampling steps
            temporal_downsample: Number of temporal downsampling steps
        """
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels

        # Build encoder layers
        self.layers = []

        # Initial convolution: map input to feature space
        # This is like the "stem" in ResNet — low-level feature extraction
        self.initial_conv = Conv3D(
            in_channels, base_channels,
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )

        # Downsampling blocks
        current_channels = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult

            # Spatial-temporal processing block
            self.layers.append(
                SpatialTemporalBlock(current_channels)
            )

            # Downsampling convolution
            # Spatial downsample: stride (1, 2, 2) halves H and W
            # Temporal downsample (if needed): stride (2, 1, 1) halves T
            if i < spatial_downsample:
                spatial_stride = (1, 2, 2)
            else:
                spatial_stride = (1, 1, 1)

            if i >= len(channel_multipliers) - temporal_downsample:
                temporal_stride = (2, 1, 1)
            else:
                temporal_stride = (1, 1, 1)

            self.layers.append(
                Conv3D(current_channels, out_channels,
                       kernel_size=(3, 3, 3),
                       stride=(max(spatial_stride[0], temporal_stride[0]),
                               spatial_stride[1], spatial_stride[2]),
                       padding=(1, 1, 1))
            )

            current_channels = out_channels

        # Output: mean and log_variance for reparameterization
        self.output_conv = Conv3D(
            current_channels, latent_channels * 2,
            kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)
        )

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode video to latent distribution parameters.

        Args:
            x: Input video [batch, channels, time, height, width]

        Returns:
            mean: Mean of latent distribution [B, latent_dim, T', H', W']
            log_var: Log variance of latent distribution

        The reparameterization trick:
            During training: z = mean + exp(0.5 * log_var) * epsilon
            During inference: z = mean (deterministic)

            This allows gradients to flow through the sampling operation!
        """
        # Initial feature extraction
        h = self.initial_conv.forward(x)

        # Process through encoder layers
        for layer in self.layers:
            h = layer.forward(h)

        # Split into mean and log_variance
        output = self.output_conv.forward(h)
        mean = output[:, :self.latent_channels]
        log_var = output[:, self.latent_channels:]

        return mean, log_var

    def reparameterize(self, mean: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        """
        Sample from latent distribution using reparameterization trick.

        z = mean + std * epsilon, where epsilon ~ N(0, 1)

        This is differentiable! The randomness comes from epsilon,
        not from the parameters, so gradients flow through mean and std.

        Args:
            mean: Mean of latent distribution
            log_var: Log variance

        Returns:
            Sampled latent code
        """
        std = np.exp(0.5 * log_var)
        epsilon = np.random.randn(*mean.shape)
        return mean + std * epsilon


################################################################################
# SECTION 4: VIDEO VAE DECODER
################################################################################

class VideoVAEDecoder:
    """
    Video VAE Decoder — Reconstructing Video from Latents
    =====================================================

    The decoder takes a latent representation and reconstructs the video.
    This mirrors the encoder architecture but in reverse.

    Architecture:
        Latent: [B, latent_dim, T/2, H/8, W/8]
            │
            ▼
        ┌─────────────────────────────┐
        │  Initial Conv               │  latent_dim → 512 channels
        └─────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────┐
        │  Spatial-Temporal Block 1   │  512 channels
        │  + Upsample 2x             │  32×32 → 64×64
        └─────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────┐
        │  Spatial-Temporal Block 2   │  256 channels
        │  + Upsample 2x             │  64×64 → 128×128
        └─────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────┐
        │  Spatial-Temporal Block 3   │  128 channels
        │  + Upsample 2x             │  128×128 → 256×256
        └─────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────┐
        │  Temporal Upsample 2x      │  T/2 → T
        └─────────────────────────────┘
            │
            ▼
        ┌─────────────────────────────┐
        │  Output Conv               │  128 → 3 channels (RGB)
        └─────────────────────────────┘
            │
            ▼
        Reconstructed Video: [B, 3, T, H, W]

    Key Design Decisions:
        1. Symmetric with encoder (mirrors its architecture)
        2. Uses nearest-neighbor upsampling + convolution (not transposed conv)
           - Transposed conv can cause checkerboard artifacts
           - Nearest-neighbor + conv is smoother
        3. Skip connections from encoder (optional but improves quality)
    """

    def __init__(
        self,
        latent_channels: int = 16,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: List[int] = [4, 2, 1],
        spatial_upsample: int = 3,
        temporal_upsample: int = 1,
    ):
        self.latent_channels = latent_channels
        self.out_channels = out_channels

        current_channels = base_channels * channel_multipliers[0]

        # Initial projection from latent space
        self.initial_conv = Conv3D(
            latent_channels, current_channels,
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )

        # Upsampling blocks
        self.layers = []
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult

            # Spatial-temporal processing
            self.layers.append(SpatialTemporalBlock(current_channels))

            # Upsampling (implemented as nearest-neighbor + conv)
            if i < spatial_upsample:
                self.layers.append(('upsample_spatial', 2))

            if i < temporal_upsample:
                self.layers.append(('upsample_temporal', 2))

            # Channel reduction convolution
            self.layers.append(
                Conv3D(current_channels, out_ch,
                       kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            )
            current_channels = out_ch

        # Final output convolution
        self.output_conv = Conv3D(
            current_channels, out_channels,
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent representation to video.

        Args:
            z: Latent code [batch, latent_channels, T', H', W']

        Returns:
            Reconstructed video [batch, 3, T, H, W]
        """
        h = self.initial_conv.forward(z)

        for layer in self.layers:
            if isinstance(layer, tuple):
                # Upsampling operation
                op, scale = layer
                if op == 'upsample_spatial':
                    # Nearest-neighbor upsampling
                    h = np.repeat(h, scale, axis=3)
                    h = np.repeat(h, scale, axis=4)
                elif op == 'upsample_temporal':
                    h = np.repeat(h, scale, axis=2)
            else:
                h = layer.forward(h)

        return self.output_conv.forward(h)


################################################################################
# SECTION 5: COMPLETE VIDEO VAE
################################################################################

class VideoVAE:
    """
    Complete Video Variational Autoencoder
    =======================================

    Combines encoder and decoder with training loss.

    Training Objective:
        L = L_reconstruction + β × L_KL

        Where:
        - L_reconstruction: How well can we reconstruct the input video?
          Typically MSE or perceptual loss (LPIPS)
        - L_KL: KL divergence between latent distribution and N(0,1)
          This regularizes the latent space to be smooth and continuous
        - β: Weight for KL term (β-VAE trick for disentanglement)

    Why β matters:
        - β too small: Latent space is irregular, can't interpolate
        - β too large: Reconstruction quality suffers (posterior collapse)
        - β = 1: Standard VAE
        - β < 1: Better reconstruction, worse latent structure
        - β > 1: Better latent structure, worse reconstruction (β-VAE)

    Latent Space Properties:
        1. Continuous: Can interpolate between any two videos
        2. Smooth: Small changes in latent → small changes in video
        3. Disentangled: Different dimensions control different aspects
        4. Compact: Only encodes meaningful information

    Interview Questions:
        Q: "What is the KL divergence loss doing in a VAE?"
        A: It prevents the encoder from just memorizing inputs. Without it,
           the encoder could map each input to a unique point, and the decoder
           would just memorize the mapping. KL forces the latent distribution
           to be close to N(0,1), which:
           1. Makes the latent space smooth and continuous
           2. Enables sampling (generate new videos by sampling z ~ N(0,1))
           3. Acts as a regularizer

        Q: "How do you prevent posterior collapse?"
        A: Posterior collapse occurs when the decoder ignores the latent code.
           Solutions:
           1. KL annealing: Start with β=0, gradually increase to 1
           2. Free bits: Set minimum KL per dimension
           3. Use a powerful decoder (but risk auto-encoding)
           4. Use perceptual loss instead of pixel MSE
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        kl_weight: float = 1e-6,  # Small weight for KL (common in video VAEs)
    ):
        self.encoder = VideoVAEEncoder(in_channels, latent_channels, base_channels)
        self.decoder = VideoVAEDecoder(latent_channels, in_channels, base_channels)
        self.kl_weight = kl_weight

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode video to latent distribution parameters."""
        return self.encoder.encode(x)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent code to video."""
        return self.decoder.decode(z)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full forward pass: encode → sample → decode.

        Returns:
            reconstruction: Reconstructed video
            mean: Latent mean
            log_var: Latent log variance
        """
        mean, log_var = self.encode(x)
        z = self.encoder.reparameterize(mean, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mean, log_var

    def compute_loss(self, x: np.ndarray) -> dict:
        """
        Compute VAE training loss.

        L = L_recon + β × L_KL

        L_recon = ||x - x_hat||²  (MSE reconstruction)
        L_KL = -0.5 × Σ(1 + log(σ²) - μ² - σ²)

        Returns dict with total loss and components.
        """
        reconstruction, mean, log_var = self.forward(x)

        # Reconstruction loss (MSE)
        recon_loss = np.mean((x - reconstruction) ** 2)

        # KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0,1)
        # = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * np.mean(1 + log_var - mean**2 - np.exp(log_var))

        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
        }


################################################################################
# SECTION 6: TESTING AND DEMONSTRATION
################################################################################

def demonstrate_video_vae():
    """
    Demonstrate Video VAE with a small example.

    This shows:
    1. How video is encoded to latent space
    2. How latent codes are sampled
    3. How video is reconstructed
    4. The loss computation
    """
    print("=" * 70)
    print("VIDEO VAE DEMONSTRATION")
    print("=" * 70)

    # Create a small Video VAE
    vae = VideoVAE(
        in_channels=3,
        latent_channels=4,
        base_channels=16,
        kl_weight=1e-6
    )

    # Create a small synthetic video
    # In practice, this would be real video data
    batch_size = 1
    n_frames = 4
    height = 16
    width = 16

    # Random "video" (in practice, real frames)
    video = np.random.randn(batch_size, 3, n_frames, height, width)

    print(f"\nInput video shape: {video.shape}")
    print(f"  Batch: {batch_size}")
    print(f"  Frames: {n_frames}")
    print(f"  Resolution: {height}×{width}")
    print(f"  Channels: 3 (RGB)")

    # Encode
    mean, log_var = vae.encode(video)
    print(f"\nLatent mean shape: {mean.shape}")
    print(f"Latent log_var shape: {log_var.shape}")

    # Sample latent code
    z = vae.encoder.reparameterize(mean, log_var)
    print(f"Sampled latent shape: {z.shape}")

    # Decode
    reconstruction = vae.decode(z)
    print(f"Reconstruction shape: {reconstruction.shape}")

    # Compute loss
    loss_dict = vae.compute_loss(video)
    print(f"\nLosses:")
    print(f"  Total loss: {loss_dict['total_loss']:.4f}")
    print(f"  Reconstruction loss: {loss_dict['reconstruction_loss']:.4f}")
    print(f"  KL loss: {loss_dict['kl_loss']:.4f}")

    # Show compression ratio
    video_size = video.size
    latent_size = z.size
    compression = video_size / latent_size
    print(f"\nCompression:")
    print(f"  Video elements: {video_size}")
    print(f"  Latent elements: {latent_size}")
    print(f"  Compression ratio: {compression:.1f}x")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: The latent space is", compression:.1f, "x smaller!")
    print("This makes diffusion in latent space", compression:.1f, "x more efficient.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_video_vae()


################################################################################
# REFERENCES
################################################################################

# [1] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
#     The original VAE paper — introduces reparameterization trick.
#
# [2] Ho, J., et al. (2022). Video Diffusion Models.
#     First paper to apply diffusion to video generation.
#
# [3] Blattmann, A., et al. (2023). Stable Video Diffusion: Scaling Latent
#     Video Diffusion Models to Large Datasets.
#     Uses a Video VAE for latent video diffusion.
#
# [4] Yang, L., et al. (2024). CogVideoX: Text-to-Video Diffusion Models
#     With An Expert Transformer.
#     Uses causal 3D VAE for temporal consistency.
#
# [5] Tencent (2025). Hunyuan Video: A Systematic Framework For Large Video
#     Generative Models.
#     State-of-the-art video VAE with factorized spatial-temporal layers.
#
# [6] Alibaba (2025). Wan: Large-Scale Video Generation Model.
#     Uses 3D causal VAE with advanced temporal modeling.

################################################################################
# FUTURE IMPROVEMENTS
################################################################################

# 1. Perceptual Loss: Replace MSE with LPIPS for better visual quality
# 2. Adversarial Training: Add discriminator for sharper reconstructions
# 3. Temporal Attention: Add attention layers between spatial and temporal blocks
# 4. Progressive Training: Start with low resolution, gradually increase
# 5. Mixed Precision: Use FP16 for faster training
# 6. Memory Optimization: Gradient checkpointing for long videos

################################################################################
