"""
################################################################################
VISION ENCODERS — FROM CNNs TO VISION TRANSFORMERS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Vision Encoders?
    Vision encoders convert raw images into meaningful feature vectors
    that can be used by downstream tasks (classification, detection,
    segmentation, or as input to multimodal models).

Historical Evolution:
    1998: LeNet (LeCun) — First CNN for digit recognition
    2012: AlexNet (Krizhevsky) — Deep learning revolution
    2014: VGGNet — Deeper networks with 3×3 convolutions
    2014: GoogLeNet — Inception modules
    2015: ResNet (He) — Skip connections, 152 layers
    2019: EfficientNet — Compound scaling
    2020: ViT (Dosovitskiy) — Pure transformer for vision
    2021: Swin Transformer — Hierarchical vision transformer
    2022: ConvNeXt — Modernized CNN
    2023: DINOv2 — Self-supervised vision transformers
    2024: SigLIP — Better CLIP training
    2025: Vision encoders in multimodal LLMs

Why Vision Encoders Matter:
    Every multimodal system needs a way to "see" images.
    The vision encoder is the "eyes" of AI systems like:
    - GPT-4V, Claude 3, Gemini (vision-language models)
    - CLIP, SigLIP (contrastive learning)
    - Stable Diffusion (image generation conditioning)

################################################################################
"""

import numpy as np
from typing import Optional, List, Tuple
import math

################################################################################
# SECTION 1: CONVOLUTIONAL NEURAL NETWORK (CNN)
################################################################################

########################################
CONVOLUTIONAL NEURAL NETWORK
########################################

Definition:
    A CNN applies learnable filters (kernels) across an image to detect
    features like edges, textures, and patterns.

Why CNNs?
    Images have spatial structure. A pixel at (10, 10) is related to
    pixels at (9, 10), (11, 10), (10, 9), (10, 11). CNNs exploit
    this locality through small kernels that slide across the image.

Key Properties:
    1. Local connectivity: each neuron sees only a small region
    2. Weight sharing: same filter applied everywhere
    3. Translation equivariance: pattern detected regardless of position

Interview Questions:
    Q: "What is a convolution in CNNs?"
    A: A convolution slides a small filter across the input, computing
       dot products at each position. This detects local patterns.

    Q: "Why use 3×3 kernels instead of larger ones?"
    A: Two 3×3 layers have the same receptive field as one 5×5 layer,
       but with fewer parameters and more non-linearity.

################################################################################
"""

class Conv2D:
    """
    2D Convolutional Layer
    ======================

    Slides a kernel across the input to produce feature maps.

    Mathematical Operation:
        output[i,j] = Σ_m Σ_n input[i+m, j+n] * kernel[m, n] + bias

    For a kernel of size (kH, kW):
        output_height = (input_height - kH + 2*padding) / stride + 1
        output_width = (input_width - kW + 2*padding) / stride + 1

    Visual Example (3×3 kernel on 5×5 input):
        Input:              Kernel:         Output:
        [1 2 3 4 5]        [1 0 1]        [8  12 16]
        [6 7 8 9 0]    *   [0 1 0]    =   [18 22 10]
        [1 2 3 4 5]        [1 0 1]        [8  12 16]
        [6 7 8 9 0]
        [1 2 3 4 5]

    Parameters:
        in_channels: Number of input channels (e.g., 3 for RGB)
        out_channels: Number of output channels (filters)
        kernel_size: Size of the convolution kernel
        stride: Step size for sliding the kernel
        padding: Zero-padding around the input

    Interview Questions:
        Q: "What is the difference between convolution and cross-correlation?"
        A: Convolution flips the kernel before sliding. In practice,
           deep learning frameworks use cross-correlation (no flip),
           but call it convolution.

        Q: "How do you compute the output size?"
        A: out_size = (in_size - kernel + 2*padding) / stride + 1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization: scale = sqrt(2 / fan_in)
        # This keeps variance stable through layers
        fan_in = in_channels * kernel_size * kernel_size
        scale = math.sqrt(2.0 / fan_in)

        # Kernel shape: [out_channels × in_channels × kH × kW]
        self.weight = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * scale

        self.bias = np.zeros(out_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of 2D convolution.

        Args:
            x: Input tensor [batch × in_channels × height × width]

        Returns:
            output: [batch × out_channels × out_height × out_width]

        Algorithm:
            1. Add padding if needed
            2. For each output position:
               - Extract the patch from input
               - Compute dot product with each kernel
               - Add bias
            3. Return output feature map
        """
        batch, _, in_h, in_w = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        # Output dimensions
        out_h = (in_h - k + 2 * p) // s + 1
        out_w = (in_w - k + 2 * p) // s + 1

        # Add padding
        if p > 0:
            x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        # Initialize output
        output = np.zeros((batch, self.out_channels, out_h, out_w))

        # Compute convolution
        # This is a naive implementation — real frameworks use im2col or FFT
        for i in range(out_h):
            for j in range(out_w):
                # Extract patch: [batch × in_channels × kH × kW]
                h_start = i * s
                w_start = j * s
                patch = x[:, :, h_start:h_start + k, w_start:w_start + k]

                # Compute output for all filters
                # patch: [batch × in_c × kH × kW]
                # weight: [out_c × in_c × kH × kW]
                # output[:, :, i, j] = sum over in_c, kH, kW
                for oc in range(self.out_channels):
                    output[:, oc, i, j] = np.sum(
                        patch * self.weight[oc], axis=(1, 2, 3)
                    ) + self.bias[oc]

        return output


class MaxPool2D:
    """
    Max Pooling Layer
    =================

    Reduces spatial dimensions by taking the maximum value in each window.

    Purpose:
        1. Reduce computation (fewer parameters in subsequent layers)
        2. Add translation invariance (small shifts don't change output)
        3. Increase receptive field

    Mathematical Operation:
        output[i,j] = max(input[i*s:i*s+k, j*s:j*s+k])

    Visual Example (2×2 pool, stride 2):
        Input:          Output:
        [1 3 2 4]      [3 4]
        [5 6 7 8]  →   [6 9]
        [1 2 3 4]
        [5 6 9 1]

    Interview Question:
        Q: "Why max pooling instead of average pooling?"
        A: Max pooling captures the strongest activation, which is often
           the most important feature. Average pooling smooths everything,
           which can dilute important signals.
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of max pooling.

        Args:
            x: [batch × channels × height × width]

        Returns:
            output: [batch × channels × height/k × width/k]
        """
        batch, channels, in_h, in_w = x.shape
        k = self.kernel_size
        s = self.stride

        out_h = (in_h - k) // s + 1
        out_w = (in_w - k) // s + 1

        output = np.zeros((batch, channels, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, :, i*s:i*s+k, j*s:j*s+k]
                output[:, :, i, j] = np.max(patch, axis=(2, 3))

        return output


################################################################################
# SECTION 2: RESNET (RESIDUAL NETWORK)
################################################################################

########################################
RESNET (RESIDUAL NETWORK)
########################################

Definition:
    ResNet introduces skip connections that allow gradients to flow
    directly through the network, enabling training of very deep networks.

Key Innovation: Residual Connection
    Instead of learning H(x), learn F(x) = H(x) - x
    Then H(x) = F(x) + x

    This is easier because:
    - If the identity is optimal, F(x) = 0 is easy to learn
    - Gradients flow directly through the skip connection
    - No vanishing gradient problem

Historical Impact:
    - Won ImageNet 2015 (3.57% top-5 error)
    - Enabled training of 100+ layer networks
    - Skip connections became standard in all architectures
    - Influenced transformer design (residual connections)

Architecture:
    Input → Conv → BN → ReLU → Conv → BN → (+Input) → ReLU
                               ↑
                           Skip Connection

Interview Questions:
    Q: "Why do skip connections help?"
    A: They provide a gradient highway. During backpropagation,
       gradients can flow directly through the skip connection,
       avoiding vanishing gradients in deep networks.

    Q: "What's the difference between ResNet-50 and ResNet-152?"
    A: Depth. ResNet-50 has 50 layers, ResNet-152 has 152.
       Deeper networks can learn more complex features but are
       harder to train and slower.

################################################################################
"""

class BatchNorm2D:
    """
    Batch Normalization
    ===================

    Normalizes activations across the batch dimension.

    Formula:
        y = γ * (x - μ_B) / sqrt(σ_B² + ε) + β

    Where:
        μ_B = mean across batch
        σ_B² = variance across batch
        γ, β = learnable scale and shift

    Benefits:
        1. Faster training (higher learning rates)
        2. Regularization effect
        3. Less sensitivity to initialization

    Interview Question:
        Q: "Why does batch normalization help?"
        A: It normalizes activations, preventing internal covariate shift.
           This allows higher learning rates and faster convergence.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        self.num_features = num_features
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass of batch normalization.

        Args:
            x: [batch × channels × height × width]
            training: Whether in training mode

        Returns:
            Normalized tensor
        """
        if training:
            # Compute batch statistics
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)

            # Update running statistics
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean.squeeze()
            self.running_var = 0.9 * self.running_var + 0.1 * var.squeeze()
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)


class ResidualBlock:
    """
    Residual Block
    ==============

    The fundamental building block of ResNet.

    Architecture:
        x → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+x) → ReLU

    The skip connection adds the input directly to the output.

    For ResNet-50 and above, uses bottleneck design:
        x → Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN → (+x) → ReLU

    The 1×1 convolutions reduce and then restore dimensions,
    making the 3×3 convolution operate on fewer channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_bottleneck: bool = False
    ):
        self.use_bottleneck = use_bottleneck
        self.stride = stride

        if use_bottleneck:
            # Bottleneck: reduce → process → restore
            mid_channels = out_channels // 4
            self.conv1 = Conv2D(in_channels, mid_channels, kernel_size=1)
            self.bn1 = BatchNorm2D(mid_channels)
            self.conv2 = Conv2D(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
            self.bn2 = BatchNorm2D(mid_channels)
            self.conv3 = Conv2D(mid_channels, out_channels, kernel_size=1)
            self.bn3 = BatchNorm2D(out_channels)
        else:
            # Basic block: two 3×3 convolutions
            self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = BatchNorm2D(out_channels)
            self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = BatchNorm2D(out_channels)

        # Skip connection (if dimensions change)
        if stride != 1 or in_channels != out_channels:
            self.skip_conv = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride)
            self.skip_bn = BatchNorm2D(out_channels)
        else:
            self.skip_conv = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of residual block.

        Args:
            x: [batch × in_channels × height × width]

        Returns:
            output: [batch × out_channels × height/stride × width/stride]
        """
        identity = x

        if self.use_bottleneck:
            out = np.maximum(0, self.bn1.forward(self.conv1.forward(x)))  # ReLU
            out = np.maximum(0, self.bn2.forward(self.conv2.forward(out)))
            out = self.bn3.forward(self.conv3.forward(out))
        else:
            out = np.maximum(0, self.bn1.forward(self.conv1.forward(x)))
            out = self.bn2.forward(self.conv2.forward(out))

        # Skip connection
        if self.skip_conv is not None:
            identity = self.skip_bn.forward(self.skip_conv.forward(x))

        # Residual addition + ReLU
        return np.maximum(0, out + identity)


class ResNet:
    """
    ResNet: Deep Residual Learning
    ===============================

    Complete ResNet architecture for image classification.

    Architecture (ResNet-50):
        Input (224×224×3)
        → Conv 7×7, 64, stride 2 (112×112×64)
        → MaxPool 3×3, stride 2 (56×56×64)
        → Stage 1: 3 bottleneck blocks (56×56×256)
        → Stage 2: 4 bottleneck blocks (28×28×512)
        → Stage 3: 6 bottleneck blocks (14×14×1024)
        → Stage 4: 3 bottleneck blocks (7×7×2048)
        → Global Average Pool (1×1×2048)
        → FC (1000 classes)

    Model Variants:
        ResNet-18:  [2, 2, 2, 2] basic blocks, 11M params
        ResNet-34:  [3, 4, 6, 3] basic blocks, 21M params
        ResNet-50:  [3, 4, 6, 3] bottleneck blocks, 25M params
        ResNet-101: [3, 4, 23, 3] bottleneck blocks, 44M params
        ResNet-152: [3, 8, 36, 3] bottleneck blocks, 60M params

    Interview Questions:
        Q: "How does ResNet handle the vanishing gradient problem?"
        A: Skip connections provide a direct path for gradients.
           Even if the learned residual F(x) has small gradients,
           the gradient through the skip connection (identity) is 1.

        Q: "When should I use ResNet vs ViT?"
        A: ResNet for smaller datasets or when compute is limited.
           ViT for large datasets where attention can learn global patterns.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        layers: List[int] = [3, 4, 6, 3],
        use_bottleneck: bool = True
    ):
        self.num_classes = num_classes

        # Initial convolution
        self.conv1 = Conv2D(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm2D(64)
        self.maxpool = MaxPool2D(kernel_size=3, stride=2)

        # Residual stages
        if use_bottleneck:
            channels = [256, 512, 1024, 2048]
        else:
            channels = [64, 128, 256, 512]

        self.stages = []
        in_channels = 64
        for i, (n_blocks, out_channels) in enumerate(zip(layers, channels)):
            blocks = []
            for j in range(n_blocks):
                stride = 2 if j == 0 and i > 0 else 1
                blocks.append(ResidualBlock(
                    in_channels if j == 0 else out_channels,
                    out_channels,
                    stride=stride,
                    use_bottleneck=use_bottleneck
                ))
            self.stages.append(blocks)
            in_channels = out_channels

        # Classification head
        self.fc_weight = np.random.randn(channels[-1], num_classes) * 0.01
        self.fc_bias = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of ResNet.

        Args:
            x: [batch × 3 × 224 × 224]

        Returns:
            logits: [batch × num_classes]
        """
        # Initial layers
        x = np.maximum(0, self.bn1.forward(self.conv1.forward(x)))  # ReLU
        x = self.maxpool.forward(x)

        # Residual stages
        for stage in self.stages:
            for block in stage:
                x = block.forward(x)

        # Global average pooling
        x = np.mean(x, axis=(2, 3))  # [batch × channels]

        # Classification
        logits = np.matmul(x, self.fc_weight) + self.fc_bias

        return logits

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """
        Extract features (before classification head).

        Useful for transfer learning or as input to other models.
        """
        x = np.maximum(0, self.bn1.forward(self.conv1.forward(x)))
        x = self.maxpool.forward(x)

        for stage in self.stages:
            for block in stage:
                x = block.forward(x)

        return np.mean(x, axis=(2, 3))


################################################################################
# SECTION 3: VISION TRANSFORMER (ViT)
################################################################################

########################################
VISION TRANSFORMER (ViT)
########################################

Definition:
    ViT applies the transformer architecture to images by splitting
    them into fixed-size patches and treating each patch as a token.

Key Insight:
    An image is worth 16×16 words!
    Split 224×224 image into 14×14 = 196 patches of 16×16 pixels.
    Process these patches like tokens in a language model.

Architecture:
    Image (224×224) → 196 patches (16×16 each)
    → Linear projection → 196 patch embeddings
    + CLS token → 197 tokens
    + Position embeddings
    → Transformer encoder (12 layers)
    → CLS token output → Classification

Advantages over CNNs:
    1. Global attention from layer 1
    2. More flexible architecture
    3. Scales better with data
    4. Unified architecture with NLP

Disadvantages:
    1. Needs more data than CNNs
    2. More compute for same accuracy on small datasets
    3. No built-in inductive bias for locality

Interview Questions:
        Q: "How does ViT process images?"
        A: Split into patches, linearly project each patch to an embedding,
           add position embeddings, then process with standard transformer
           encoder. Use CLS token for classification.

        Q: "Why does ViT need more data than CNNs?"
        A: CNNs have built-in inductive biases (locality, translation
           equivariance). ViT must learn these from data, so it needs
           more examples to generalize well.

################################################################################
"""

class PatchEmbedding:
    """
    Patch Embedding
    ===============

    Splits an image into patches and projects each to an embedding vector.

    For 224×224 image with 16×16 patches:
    - Number of patches: (224/16)² = 14² = 196
    - Each patch: 16×16×3 = 768 values
    - Project to d_model dimensions

    This is equivalent to a convolution with kernel_size=patch_size, stride=patch_size.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_patches = (image_size // patch_size) ** 2

        # Projection weights
        patch_dim = patch_size * patch_size * in_channels
        scale = math.sqrt(2.0 / (patch_dim + d_model))
        self.projection = np.random.randn(patch_dim, d_model) * scale
        self.bias = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Convert image to patch embeddings.

        Args:
            x: [batch × channels × height × width]

        Returns:
            patches: [batch × n_patches × d_model]
        """
        batch, channels, height, width = x.shape
        p = self.patch_size

        # Reshape to extract patches
        # [batch × C × H × W] → [batch × C × H/p × p × W/p × p]
        x = x.reshape(batch, channels, height // p, p, width // p, p)
        # → [batch × H/p × W/p × C × p × p]
        x = x.transpose(0, 2, 4, 1, 3, 5)
        # → [batch × n_patches × patch_dim]
        x = x.reshape(batch, self.n_patches, -1)

        # Linear projection
        return np.matmul(x, self.projection) + self.bias


class VisionTransformer:
    """
    Vision Transformer (ViT)
    =========================

    Complete ViT implementation for image classification.

    Architecture:
        1. Patch Embedding: split image into patches
        2. CLS Token: prepend learnable classification token
        3. Position Embedding: add learnable position information
        4. Transformer Encoder: N layers of self-attention + FFN
        5. Classification Head: project CLS token to classes

    Model Variants:
        ViT-Small:  d_model=384, heads=6, layers=12, 22M params
        ViT-Base:   d_model=768, heads=12, layers=12, 86M params
        ViT-Large:  d_model=1024, heads=16, layers=24, 307M params
        ViT-Huge:   d_model=1280, heads=16, layers=32, 632M params

    Training Recipe:
        - Pre-training: ImageNet-21k (14M images), 224×224
        - Fine-tuning: ImageNet-1k (1.2M images), 384×384
        - Optimizer: AdamW, lr=1e-3, weight_decay=0.1
        - Augmentation: RandAugment, Mixup, CutMix

    Interview Questions:
        Q: "What is the CLS token in ViT?"
        A: A learnable token prepended to the sequence. After transformer
           processing, its representation is used for classification.
           It aggregates information from all patches through attention.

        Q: "How does ViT handle different image sizes?"
        A: Position embeddings are interpolated for new sizes.
           Patch count changes, but the model can adapt.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        num_classes: int = 1000,
        dropout: float = 0.1
    ):
        self.d_model = d_model
        self.n_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model)

        # CLS token (learnable)
        self.cls_token = np.random.randn(1, 1, d_model) * 0.02

        # Position embeddings (learnable)
        self.pos_embed = np.random.randn(1, self.n_patches + 1, d_model) * 0.02

        # Transformer encoder layers
        self.layers = []
        for _ in range(n_layers):
            self.layers.append({
                'norm1': LayerNorm(d_model),
                'attn': MultiHeadAttention(d_model, n_heads),
                'norm2': LayerNorm(d_model),
                'ffn1_weight': np.random.randn(d_model, d_ff) * math.sqrt(2.0 / d_model),
                'ffn2_weight': np.random.randn(d_ff, d_model) * math.sqrt(2.0 / d_ff),
            })

        # Final norm
        self.norm = LayerNorm(d_model)

        # Classification head
        self.head_weight = np.random.randn(d_model, num_classes) * 0.01
        self.head_bias = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of ViT.

        Args:
            x: [batch × channels × height × width]

        Returns:
            logits: [batch × num_classes]
        """
        batch = x.shape[0]

        # 1. Patch embedding
        patches = self.patch_embed.forward(x)  # [batch × n_patches × d_model]

        # 2. Prepend CLS token
        cls = np.broadcast_to(self.cls_token, (batch, 1, self.d_model))
        x = np.concatenate([cls, patches], axis=1)  # [batch × n_patches+1 × d_model]

        # 3. Add position embeddings
        x = x + self.pos_embed

        # 4. Transformer encoder
        for layer in self.layers:
            # Self-attention with residual
            h = layer['norm1'].forward(x)
            attn_out, _ = layer['attn'].forward(h)
            x = x + attn_out

            # FFN with residual
            h = layer['norm2'].forward(x)
            h = np.maximum(0, np.matmul(h, layer['ffn1_weight']))  # GELU approx
            h = np.matmul(h, layer['ffn2_weight'])
            x = x + h

        # 5. Extract CLS token and classify
        x = self.norm.forward(x[:, 0])  # CLS token
        logits = np.matmul(x, self.head_weight) + self.head_bias

        return logits


class LayerNorm:
    """Layer normalization for transformer models."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.weight = np.ones(d_model)
        self.bias = np.zeros(d_model)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.weight * (x - mean) / np.sqrt(var + self.eps) + self.bias


class MultiHeadAttention:
    """Simplified multi-head attention for ViT."""
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02
        self.W_O = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch, seq, _ = x.shape

        Q = np.matmul(x, self.W_Q).reshape(batch, seq, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.matmul(x, self.W_K).reshape(batch, seq, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.matmul(x, self.W_V).reshape(batch, seq, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)

        out = np.matmul(weights, V)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)
        out = np.matmul(out, self.W_O)

        return out, weights


################################################################################
# SECTION 4: TESTING & EXAMPLES
################################################################################

def demonstrate_vision_encoders():
    """Demonstrate vision encoder concepts."""
    print("=" * 70)
    print("VISION ENCODERS DEMONSTRATION")
    print("=" * 70)

    # Conv2D
    print("\n--- Conv2D ---")
    conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    x = np.random.randn(1, 3, 8, 8)
    out = conv.forward(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    # MaxPool
    print("\n--- MaxPool2D ---")
    pool = MaxPool2D(kernel_size=2, stride=2)
    out = pool.forward(out)
    print(f"After pooling: {out.shape}")

    # ResNet features
    print("\n--- ResNet (small) ---")
    resnet = ResNet(num_classes=10, layers=[2, 2, 2, 2], use_bottleneck=False)
    x = np.random.randn(1, 3, 32, 32)
    features = resnet.extract_features(x)
    print(f"Feature shape: {features.shape}")

    # ViT
    print("\n--- Vision Transformer ---")
    vit = VisionTransformer(
        image_size=32, patch_size=4, d_model=64,
        n_layers=2, n_heads=4, num_classes=10
    )
    x = np.random.randn(1, 3, 32, 32)
    logits = vit.forward(x)
    print(f"Input: {x.shape} → Logits: {logits.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_vision_encoders()


################################################################################
# REFERENCES
################################################################################

# [1] He, K., et al. (2016). Deep Residual Learning for Image Recognition.
# [2] Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words.
# [3] Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer.
# [4] Liu, Z., et al. (2022). A ConvNet for the 2020s (ConvNeXt).
# [5] Oquab, M., et al. (2024). DINOv2: Learning Robust Visual Features.

################################################################################
