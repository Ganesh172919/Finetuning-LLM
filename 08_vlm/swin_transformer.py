"""
################################################################################
SWIN TRANSFORMER — HIERARCHICAL VISION TRANSFORMER
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Swin Transformer?
    Swin (Shifted Window) Transformer computes attention within local
    windows instead of globally, making it efficient for high-resolution
    images. It creates a hierarchical representation like CNNs.

Key Innovation: Shifted Windows
    Instead of global attention (O(n²)), use local windows:
    1. Partition image into non-overlapping windows
    2. Compute self-attention within each window
    3. Shift windows between layers for cross-window connections

Why Swin?
    ViT's global attention is O(n²) in number of patches.
    For 224×224 image with 4×4 patches: 56×56 = 3136 patches
    Global attention: 3136² ≈ 10M operations per head!

    Swin with 7×7 windows: 49² = 2401 operations per window
    Much more efficient!

Architecture:
    Stage 1: 56×56 patches, 4×4 window, d=96
    Stage 2: 28×28 patches, 7×7 window, d=192
    Stage 3: 14×14 patches, 7×7 window, d=384
    Stage 4: 7×7 patches, 7×7 window, d=768

    Between stages: Patch Merging (downsample)
    Within stages: Window Attention + Shifted Window Attention

Interview Questions:
        Q: "What's the difference between ViT and Swin?"
        A: ViT uses global attention (all patches attend to all).
           Swin uses local window attention (patches attend to neighbors).
           Swin is more efficient and creates hierarchical features.

        Q: "Why shift windows between layers?"
        A: Without shifting, windows are isolated — no information flow
           between them. Shifting creates cross-window connections.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple
import math

################################################################################
# SECTION 1: WINDOW ATTENTION
################################################################################

class WindowAttention:
    """
    Window-based Self-Attention
    ============================

    Computes attention within non-overlapping windows.

    For a feature map of size (H, W) with window size (M, M):
    - Number of windows: (H/M) × (W/M)
    - Each window has M² tokens
    - Attention within each window: O(M⁴) per window
    - Total: O(H × W × M²) — linear in image size!

    Visual:
        Feature map (8×8) with 4×4 windows:
        ┌───┬───┐
        │ W1│ W2│   W1 attends only to tokens in W1
        ├───┼───┤   W2 attends only to tokens in W2
        │ W3│ W4│   etc.
        └───┴───┘
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size

        # Attention projections
        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02
        self.W_O = np.random.randn(d_model, d_model) * 0.02

        # Relative position bias
        # Each position in the window can attend to all others
        # We learn a bias for each relative position
        n_tokens = window_size * window_size
        self.relative_bias = np.zeros((n_heads, n_tokens, n_tokens))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Window attention forward pass.

        Args:
            x: [batch × n_windows × window_tokens × d_model]

        Returns:
            output: same shape
        """
        batch, n_windows, n_tokens, d = x.shape

        # Project to Q, K, V
        Q = np.matmul(x, self.W_Q)
        K = np.matmul(x, self.W_K)
        V = np.matmul(x, self.W_V)

        # Reshape for multi-head
        Q = Q.reshape(batch, n_windows, n_tokens, self.n_heads, self.d_k).transpose(0, 1, 3, 2, 4)
        K = K.reshape(batch, n_windows, n_tokens, self.n_heads, self.d_k).transpose(0, 1, 3, 2, 4)
        V = V.reshape(batch, n_windows, n_tokens, self.n_heads, self.d_k).transpose(0, 1, 3, 2, 4)

        # Attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 2, 4, 3)) / math.sqrt(self.d_k)

        # Add relative position bias
        scores = scores + self.relative_bias[np.newaxis, :, :, :]

        # Softmax
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)

        # Weighted sum
        out = np.matmul(weights, V)
        out = out.transpose(0, 1, 3, 2, 4).reshape(batch, n_windows, n_tokens, d)

        return np.matmul(out, self.W_O)


################################################################################
# SECTION 2: SHIFTED WINDOW
################################################################################

def window_partition(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    Partition feature map into non-overlapping windows.

    Args:
        x: [batch × H × W × C]
        window_size: M

    Returns:
        windows: [batch × n_windows × M² × C]
    """
    batch, H, W, C = x.shape
    M = window_size

    # Reshape to windows
    x = x.reshape(batch, H // M, M, W // M, M, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # [batch × H/M × W/M × M × M × C]
    x = x.reshape(batch, -1, M * M, C)  # [batch × n_windows × M² × C]

    return x


def window_reverse(windows: np.ndarray, window_size: int, H: int, W: int) -> np.ndarray:
    """
    Reverse window partition.

    Args:
        windows: [batch × n_windows × M² × C]
        window_size: M
        H, W: Original feature map size

    Returns:
        x: [batch × H × W × C]
    """
    batch = windows.shape[0]
    M = window_size
    C = windows.shape[-1]

    x = windows.reshape(batch, H // M, W // M, M, M, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    x = x.reshape(batch, H, W, C)

    return x


################################################################################
# SECTION 3: SWIN TRANSFORMER BLOCK
################################################################################

class SwinTransformerBlock:
    """
    Swin Transformer Block
    ======================

    Two consecutive blocks:
    1. Window Attention (W-MSA)
    2. Shifted Window Attention (SW-MSA)

    Each block:
        x → LayerNorm → Window Attention → (+x) → LayerNorm → FFN → (+x)

    Interview Question:
        Q: "How does the shifted window work?"
        A: Before computing attention, shift the feature map by
           (M/2, M/2) pixels. This creates different window boundaries,
           enabling cross-window information flow.
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int = 7, shift: bool = False):
        self.d_model = d_model
        self.window_size = window_size
        self.shift = shift

        # Layer norms
        self.norm1_weight = np.ones(d_model)
        self.norm1_bias = np.zeros(d_model)
        self.norm2_weight = np.ones(d_model)
        self.norm2_bias = np.zeros(d_model)

        # Window attention
        self.attn = WindowAttention(d_model, n_heads, window_size)

        # FFN
        d_ff = d_model * 4
        self.ffn1 = np.random.randn(d_model, d_ff) * math.sqrt(2.0 / d_model)
        self.ffn2 = np.random.randn(d_ff, d_model) * math.sqrt(2.0 / d_ff)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: [batch × H × W × d_model]

        Returns:
            output: [batch × H × W × d_model]
        """
        batch, H, W, d = x.shape
        M = self.window_size
        residual = x

        # LayerNorm
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x = self.norm1_weight * (x - mean) / np.sqrt(var + 1e-6) + self.norm1_bias

        # Shift (for SW-MSA)
        if self.shift:
            shift_size = M // 2
            x = np.roll(x, shift_size, axis=1)
            x = np.roll(x, shift_size, axis=2)

        # Window partition
        windows = window_partition(x, M)  # [batch × n_windows × M² × d]

        # Window attention
        windows = self.attn.forward(windows)

        # Reverse windows
        x = window_reverse(windows, M, H, W)

        # Unshift
        if self.shift:
            x = np.roll(x, -shift_size, axis=1)
            x = np.roll(x, -shift_size, axis=2)

        # Residual
        x = x + residual

        # FFN with residual
        residual = x
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x = self.norm2_weight * (x - mean) / np.sqrt(var + 1e-6) + self.norm2_bias

        x = np.maximum(0, np.matmul(x, self.ffn1))  # GELU
        x = np.matmul(x, self.ffn2)

        return x + residual


################################################################################
# SECTION 4: TESTING
################################################################################

def demonstrate_swin():
    """Demonstrate Swin Transformer."""
    print("=" * 70)
    print("SWIN TRANSFORMER DEMONSTRATION")
    print("=" * 70)

    # Window attention
    print("\n--- Window Attention ---")
    d_model = 64
    window_size = 4
    batch = 1
    n_windows = 4

    attn = WindowAttention(d_model, n_heads=4, window_size=window_size)
    x = np.random.randn(batch, n_windows, window_size * window_size, d_model)
    out = attn.forward(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    # Swin block
    print("\n--- Swin Block ---")
    block = SwinTransformerBlock(d_model, n_heads=4, window_size=window_size, shift=False)
    x = np.random.randn(batch, 8, 8, d_model)
    out = block.forward(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    # With shifted window
    print("\n--- Shifted Window ---")
    block_shift = SwinTransformerBlock(d_model, n_heads=4, window_size=window_size, shift=True)
    out = block_shift.forward(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_swin()


################################################################################
# REFERENCES
################################################################################

# [1] Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.

################################################################################
