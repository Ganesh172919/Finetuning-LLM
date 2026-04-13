"""
################################################################################
VISION MAMBA (Vim) — STATE SPACE MODELS FOR VISION (2024 SOTA)
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Vision Mamba?
    Vision Mamba (Vim) applies the Mamba state space model architecture
    to computer vision tasks. Instead of using Vision Transformers (ViT)
    with O(n²) attention, Vim processes image patches with O(n) SSMs.

Why Vision Mamba?
    - ViT: O(n²) attention for n patches → expensive for high-res images
    - Vim: O(n) SSM → efficient for high-resolution and long sequences
    - Competitive accuracy with ViT on ImageNet
    - Much faster inference for high-resolution images

Architecture:
    Image → Split into Patches → Linear Embedding → Mamba Blocks → Classification

    Unlike ViT which uses attention between all patches, Vim uses
    bidirectional SSM to capture both local and global patterns.

Key Innovations:
    1. Bidirectional scanning: process patches left-to-right AND right-to-left
    2. Position embedding: learn spatial relationships
    3. [CLS] token: global image representation
    4. Bidirectional fusion: combine forward and backward SSM outputs

Interview Questions:
    Q: "How does Vision Mamba differ from ViT?"
    A: ViT uses O(n²) attention between all patches. Vim uses O(n) state
       space models with bidirectional scanning. This makes Vim more
       efficient for high-resolution images while maintaining accuracy.

    Q: "Why bidirectional scanning for vision?"
    A: Images don't have a natural left-to-right order like text.
       Bidirectional scanning captures context from both directions,
       similar to how BiLSTM works for NLP. This is crucial for
       understanding spatial relationships in images.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List

################################################################################
# SECTION 1: PATCH EMBEDDING
################################################################################

class PatchEmbedding:
    """
    Convert image to patch embeddings.

    Splits an image into non-overlapping patches and projects each
    patch to a vector representation.

    For a 224×224 image with 16×16 patches:
    - Number of patches: (224/16)² = 196 patches
    - Each patch: 16×16×3 = 768 pixels
    - Embedded dimension: d_model (e.g., 192)

    This is identical to ViT's patch embedding — the difference
    is what comes AFTER the embedding (SSM vs attention).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 192,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model

        # Number of patches
        self.n_patches = (image_size // patch_size) ** 2

        # Projection weight: flatten patch → project to d_model
        patch_dim = patch_size * patch_size * in_channels
        self.projection = np.random.randn(patch_dim, d_model) * 0.02
        self.bias = np.zeros(d_model)

    def forward(self, images: np.ndarray) -> np.ndarray:
        """
        Convert images to patch embeddings.

        Args:
            images: [batch × channels × height × width]

        Returns:
            patches: [batch × n_patches × d_model]
        """
        batch, channels, height, width = images.shape

        # Reshape to patches
        # [batch × C × H × W] → [batch × n_patches × patch_dim]
        patches = self._extract_patches(images)

        # Project to d_model
        # [batch × n_patches × patch_dim] @ [patch_dim × d_model]
        embeddings = patches @ self.projection + self.bias

        return embeddings

    def _extract_patches(self, images: np.ndarray) -> np.ndarray:
        """Extract non-overlapping patches from images."""
        batch, channels, height, width = images.shape
        ps = self.patch_size

        # Reshape into patches
        # [batch × C × H × W] → [batch × C × H/ps × ps × W/ps × ps]
        patches = images.reshape(batch, channels, height // ps, ps, width // ps, ps)
        # → [batch × H/ps × W/ps × ps × ps × C]
        patches = patches.transpose(0, 2, 4, 3, 5, 1)
        # → [batch × n_patches × patch_dim]
        patches = patches.reshape(batch, -1, ps * ps * channels)

        return patches


################################################################################
# SECTION 2: BIDIRECTIONAL MAMBA BLOCK
################################################################################

class BidirectionalMambaBlock:
    """
    Bidirectional Mamba Block for Vision.

    Unlike text (which is sequential), images need context from
    both directions. This block processes the sequence twice:
    1. Forward: left-to-right, top-to-bottom
    2. Backward: right-to-left, bottom-to-top

    The outputs are fused to create a representation that captures
    context from all directions.

    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │            Bidirectional Mamba Block                     │
    │                                                          │
    │  Input x                                                 │
    │     │                                                    │
    │     ├────▶ Forward SSM ──▶ y_fwd                        │
    │     │                                                    │
    │     └──▶ Backward SSM ──▶ y_bwd (reversed)              │
    │                         │                                │
    │                         ▼                                │
    │                    y_fwd + y_bwd                         │
    │                         │                                │
    │                         ▼                                │
    │                    Linear + Gate                         │
    │                         │                                │
    │                         ▼                                │
    │                    Output + Residual                     │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, d_model: int = 192, d_state: int = 16, expand: int = 2):
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Forward SSM parameters
        self.A_fwd = -np.exp(np.tile(np.arange(1, d_state + 1), (self.d_inner, 1)).astype(float))
        self.B_fwd_proj = np.random.randn(self.d_inner, d_state) * 0.02
        self.C_fwd_proj = np.random.randn(self.d_inner, d_state) * 0.02

        # Backward SSM parameters (separate from forward)
        self.A_bwd = -np.exp(np.tile(np.arange(1, d_state + 1), (self.d_inner, 1)).astype(float))
        self.B_bwd_proj = np.random.randn(self.d_inner, d_state) * 0.02
        self.C_bwd_proj = np.random.randn(self.d_inner, d_state) * 0.02

        # Input projection
        self.in_proj = np.random.randn(d_model, self.d_inner * 2) * 0.02

        # Output projection
        self.out_proj = np.random.randn(self.d_inner, d_model) * 0.02

        # Layer norm
        self.norm_weight = np.ones(d_model)
        self.norm_bias = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with bidirectional scanning.

        Args:
            x: [batch × seq_len × d_model]

        Returns:
            output: [batch × seq_len × d_model]
        """
        residual = x
        x_norm = self._layer_norm(x)

        # Project input
        xz = x_norm @ self.in_proj
        x_proj = xz[:, :, :self.d_inner]
        z = self._silu(xz[:, :, self.d_inner:])

        # Forward SSM
        y_fwd = self._ssm_forward(x_proj, self.A_fwd, self.B_fwd_proj, self.C_fwd_proj)

        # Backward SSM (reverse input, then reverse output)
        x_reversed = x_proj[:, ::-1, :]
        y_bwd = self._ssm_forward(x_reversed, self.A_bwd, self.B_bwd_proj, self.C_bwd_proj)
        y_bwd = y_bwd[:, ::-1, :]  # Reverse back

        # Fuse forward and backward
        y = y_fwd + y_bwd

        # Gate and project
        y = y * z
        output = y @ self.out_proj

        return output + residual

    def _ssm_forward(
        self, x: np.ndarray, A: np.ndarray, B_proj: np.ndarray, C_proj: np.ndarray
    ) -> np.ndarray:
        """Run SSM in one direction."""
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Compute input-dependent B and C
        B = x @ B_proj  # [batch × seq × d_state]
        C = x @ C_proj  # [batch × seq × d_state]

        # Discretize
        dt = 0.01  # Fixed time step for simplicity
        A_bar = np.exp(dt * A)  # [d_inner × d_state]

        # Recurrent scan
        h = np.zeros((batch, d_inner, d_state))
        outputs = []

        for t in range(seq_len):
            B_t = B[:, t, :].reshape(batch, 1, d_state)
            x_t = x[:, t, :].reshape(batch, d_inner, 1)
            h = A_bar * h + dt * B_t * x_t
            C_t = C[:, t, :].reshape(batch, 1, d_state)
            y_t = np.sum(C_t * h, axis=-1)
            outputs.append(y_t)

        return np.stack(outputs, axis=1)

    def _layer_norm(self, x, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * self.norm_weight + self.norm_bias

    def _silu(self, x):
        return x / (1 + np.exp(-np.clip(x, -20, 20)))


################################################################################
# SECTION 3: VISION MAMBA MODEL
################################################################################

class VisionMamba:
    """
    Vision Mamba (Vim) — Complete Image Classification Model.

    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    Vision Mamba                          │
    │                                                          │
    │  Image [B × C × H × W]                                  │
    │     │                                                    │
    │     ▼                                                    │
    │  Patch Embedding → [B × N × D]                          │
    │     │                                                    │
    │     ├── + [CLS] token                                    │
    │     ├── + Position embedding                             │
    │     │                                                    │
    │     ▼                                                    │
    │  Bidirectional Mamba Block × L                           │
    │     │                                                    │
    │     ▼                                                    │
    │  Layer Norm → [CLS] token → Classification Head          │
    │                                                          │
    └─────────────────────────────────────────────────────────┘

    Interview Question:
        Q: "Walk me through Vision Mamba's architecture."
        A: Image is split into patches, embedded, then processed by
           bidirectional Mamba blocks. The [CLS] token aggregates
           global information. For classification, we take the [CLS]
           token's final representation and project to class logits.
           Key advantage: O(n) complexity vs ViT's O(n²).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 192,
        n_layers: int = 12,
        d_state: int = 16,
        n_classes: int = 1000,
    ):
        self.d_model = d_model
        self.n_classes = n_classes

        # Patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        n_patches = self.patch_embed.n_patches

        # [CLS] token
        self.cls_token = np.random.randn(1, 1, d_model) * 0.02

        # Position embedding (n_patches + 1 for CLS)
        self.pos_embed = np.random.randn(1, n_patches + 1, d_model) * 0.02

        # Mamba blocks
        self.blocks = [
            BidirectionalMambaBlock(d_model, d_state)
            for _ in range(n_layers)
        ]

        # Classification head
        self.norm_weight = np.ones(d_model)
        self.norm_bias = np.zeros(d_model)
        self.head_weight = np.random.randn(d_model, n_classes) * 0.02
        self.head_bias = np.zeros(n_classes)

    def forward(self, images: np.ndarray) -> np.ndarray:
        """
        Classify images.

        Args:
            images: [batch × channels × height × width]

        Returns:
            logits: [batch × n_classes]
        """
        batch = images.shape[0]

        # Patch embedding
        x = self.patch_embed.forward(images)  # [batch × n_patches × d_model]

        # Prepend [CLS] token
        cls_tokens = np.tile(self.cls_token, (batch, 1, 1))
        x = np.concatenate([cls_tokens, x], axis=1)  # [batch × (n_patches+1) × d_model]

        # Add position embedding
        x = x + self.pos_embed

        # Process through Mamba blocks
        for block in self.blocks:
            x = block.forward(x)

        # Extract [CLS] token and classify
        cls_output = x[:, 0, :]  # [batch × d_model]

        # Layer norm
        mean = np.mean(cls_output, axis=-1, keepdims=True)
        var = np.var(cls_output, axis=-1, keepdims=True)
        cls_output = (cls_output - mean) / np.sqrt(var + 1e-5)
        cls_output = cls_output * self.norm_weight + self.norm_bias

        # Classification
        logits = cls_output @ self.head_weight + self.head_bias

        return logits

    def get_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features (before classification head).

        Useful for transfer learning, clustering, visualization.
        """
        batch = images.shape[0]
        x = self.patch_embed.forward(images)
        cls_tokens = np.tile(self.cls_token, (batch, 1, 1))
        x = np.concatenate([cls_tokens, x], axis=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block.forward(x)

        return x[:, 0, :]


################################################################################
# SECTION 4: TESTING AND DEMONSTRATION
################################################################################

def demonstrate_vision_mamba():
    """Demonstrate Vision Mamba."""
    print("=" * 70)
    print("VISION MAMBA DEMONSTRATION")
    print("=" * 70)

    # === Demo 1: Patch Embedding ===
    print("\n--- Demo 1: Patch Embedding ---")
    patch_embed = PatchEmbedding(image_size=64, patch_size=16, in_channels=3, d_model=48)
    images = np.random.randn(2, 3, 64, 64)
    patches = patch_embed.forward(images)
    print(f"Image shape: {images.shape}")
    print(f"Patches shape: {patches.shape}")
    print(f"Number of patches: {patch_embed.n_patches}")

    # === Demo 2: Bidirectional Mamba Block ===
    print("\n--- Demo 2: Bidirectional Mamba Block ---")
    block = BidirectionalMambaBlock(d_model=48, d_state=8, expand=2)
    x = np.random.randn(2, 16, 48) * 0.1
    output = block.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # === Demo 3: Full Vision Mamba ===
    print("\n--- Demo 3: Vision Mamba Model ---")
    model = VisionMamba(
        image_size=64, patch_size=16, in_channels=3,
        d_model=48, n_layers=4, d_state=8, n_classes=10
    )
    images = np.random.randn(2, 3, 64, 64)
    logits = model.forward(images)
    print(f"Images shape: {images.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions: {np.argmax(logits, axis=1)}")

    # === Demo 4: Feature Extraction ===
    print("\n--- Demo 4: Feature Extraction ---")
    features = model.get_features(images)
    print(f"Features shape: {features.shape}")

    # === Demo 5: Complexity Comparison ===
    print("\n--- Demo 5: ViT vs Vim Complexity ---")
    for n_patches in [196, 784, 3136]:  # 224, 448, 896 images
        attn_flops = n_patches ** 2
        ssm_flops = n_patches * 64
        print(f"  {n_patches} patches: ViT={attn_flops:,} FLOPs, "
              f"Vim={ssm_flops:,} FLOPs, ratio={attn_flops/ssm_flops:.1f}x")

    print("\n" + "=" * 70)
    print("All Vision Mamba demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_vision_mamba()


################################################################################
# REFERENCES
################################################################################

# [1] Zhu, L., et al. (2024). Vision Mamba: Efficient Visual Representation
#     Learning with Bidirectional State Space Model. arXiv:2401.09417.
#
# [2] Liu, Y., et al. (2024). VMamba: Visual State Space Model.
#     arXiv:2401.10166.
#
# [3] Gu, A., Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with
#     Selective State Spaces. arXiv:2312.00752.

################################################################################
