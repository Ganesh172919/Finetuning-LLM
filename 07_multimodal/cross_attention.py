"""
################################################################################
CROSS-ATTENTION — CONNECTING DIFFERENT MODALITIES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Cross-Attention?
    Cross-attention allows one modality (e.g., text) to attend to
    another modality (e.g., images). Unlike self-attention where
    Q, K, V come from the same input, cross-attention has:
    - Q from one modality (text)
    - K, V from another modality (image)

Why does it matter?
    Cross-attention is how multimodal models connect vision and language:
    - Image captioning: text attends to image features
    - Visual QA: question attends to image regions
    - Text-to-image: image generation attends to text

Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Text: "A cat sitting on a mat"                   │
    │        ↓                                          │
    │ Text Encoder → Q (queries)                       │
    │                                                  │
    │ Image: [cat photo]                               │
    │        ↓                                          │
    │ Image Encoder → K, V (keys, values)              │
    │                                                  │
    │ Cross-Attention:                                 │
    │   Attention(Q_text, K_image, V_image)            │
    │        ↓                                          │
    │ Output: text features that "understand" the image│
    └─────────────────────────────────────────────────┘

Interview Questions:
    1. "What's the difference between self and cross attention?"
       Self: Q, K, V from same input (intra-modality)
       Cross: Q from one input, K/V from another (inter-modality)

    2. "Where is cross-attention used?"
       Multimodal models (LLaVA, Flamingo), image generation
       (Stable Diffusion), machine translation.

    3. "How does cross-attention help multimodal AI?"
       It allows one modality to "query" another. Text can
       ask "what's in this image?" and get visual information.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple
import math

################################################################################
# SECTION 1: CROSS-ATTENTION
################################################################################

class CrossAttention:
    """
    Cross-Attention Mechanism
    ==========================

    Definition: Attention where queries come from one source
    and keys/values come from another source.

    Formula:
        CrossAttention(Q, K, V) = softmax(QK^T / √d_k) V

    Where:
        Q = W_Q @ X_query    (from text)
        K = W_K @ X_context  (from image)
        V = W_V @ X_context  (from image)

    Used in:
    - Flamingo: text attends to images
    - LLaVA: language attends to vision
    - Stable Diffusion: generation attends to text
    - Machine translation: target attends to source
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q projection (from query modality)
        self.W_Q = np.random.randn(d_model, d_model) * 0.02

        # K, V projections (from context modality)
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02

        # Output projection
        self.W_O = np.random.randn(d_model, d_model) * 0.02

    def forward(
        self,
        query: np.ndarray,
        context: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Cross-attention forward pass.

        Args:
            query: Query input [batch × seq_q × d_model] (e.g., text)
            context: Context input [batch × seq_c × d_model] (e.g., image)
            mask: Optional attention mask

        Returns:
            output: [batch × seq_q × d_model]
        """
        batch, seq_q, d = query.shape
        seq_c = context.shape[1]

        # Project Q from query, K/V from context
        Q = np.matmul(query, self.W_Q)  # [batch × seq_q × d_model]
        K = np.matmul(context, self.W_K)  # [batch × seq_c × d_model]
        V = np.matmul(context, self.W_V)  # [batch × seq_c × d_model]

        # Reshape for multi-head
        Q = Q.reshape(batch, seq_q, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_c, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_c, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask

        # Softmax
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)

        # Weighted sum
        output = np.matmul(weights, V)  # [batch × heads × seq_q × d_k]

        # Merge heads
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_q, d)

        # Output projection
        output = np.matmul(output, self.W_O)

        return output


################################################################################
# SECTION 2: MULTIMODAL FUSION
################################################################################

class MultimodalFusion:
    """
    Multimodal Fusion Strategies
    =============================

    Different ways to combine multiple modalities:

    1. Concatenation: Simply concatenate features
    2. Addition: Add features together
    3. Cross-Attention: Attend across modalities
    4. Gating: Learn which modality to trust more

    Interview Question:
        "How do you combine vision and language features?"
        Common approaches:
        - Concatenation + MLP (simple)
        - Cross-attention (powerful)
        - Gating (adaptive)
        Cross-attention is most common in modern VLMs.
    """

    def __init__(self, d_model: int, n_modalities: int = 2):
        self.d_model = d_model
        self.n_modalities = n_modalities

        # Gating mechanism
        self.gate = np.random.randn(d_model * n_modalities, n_modalities) * 0.02

    def concatenate(self, features: List[np.ndarray]) -> np.ndarray:
        """Simple concatenation."""
        return np.concatenate(features, axis=-1)

    def add(self, features: List[np.ndarray]) -> np.ndarray:
        """Element-wise addition."""
        return np.sum(features, axis=0)

    def gate_fusion(self, features: List[np.ndarray]) -> np.ndarray:
        """
        Gated fusion: learn which modality to trust more.

        Args:
            features: List of feature arrays

        Returns:
            Fused features
        """
        # Concatenate all features
        concat = np.concatenate(features, axis=-1)

        # Compute gates
        gate_logits = np.matmul(concat, self.gate)
        gate_weights = np.exp(gate_logits) / np.sum(np.exp(gate_logits), axis=-1, keepdims=True)

        # Weighted sum
        stacked = np.stack(features, axis=-1)
        fused = np.sum(stacked * gate_weights[..., np.newaxis, :], axis=-1)

        return fused


################################################################################
# SECTION 3: TESTING & EXAMPLES
################################################################################

def demonstrate_cross_attention():
    """Demonstrate cross-attention."""
    print("=" * 70)
    print("CROSS-ATTENTION DEMONSTRATION")
    print("=" * 70)

    batch_size = 2
    seq_text = 10  # Text sequence length
    seq_image = 16  # Image sequence length (e.g., patches)
    d_model = 64

    # Create inputs
    text_features = np.random.randn(batch_size, seq_text, d_model)
    image_features = np.random.randn(batch_size, seq_image, d_model)

    # Cross-attention
    print("\n--- Cross-Attention ---")
    cross_attn = CrossAttention(d_model, n_heads=4)
    output = cross_attn.forward(text_features, image_features)
    print(f"Text features: {text_features.shape}")
    print(f"Image features: {image_features.shape}")
    print(f"Output: {output.shape}")

    # Multimodal fusion
    print("\n--- Multimodal Fusion ---")
    fusion = MultimodalFusion(d_model, n_modalities=2)

    # Concatenation
    concat = fusion.concatenate([text_features[:, 0, :], image_features[:, 0, :]])
    print(f"Concatenation: {concat.shape}")

    # Addition
    add = fusion.add([text_features[:, 0, :], image_features[:, 0, :]])
    print(f"Addition: {add.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_cross_attention()


################################################################################
# REFERENCES
################################################################################

# [1] Vaswani, A., et al. (2017). Attention Is All You Need.
# [2] Alayrac, J.-B., et al. (2022). Flamingo: a Visual Language Model for Few-Shot Learning.
# [3] Liu, H., et al. (2023). Visual Instruction Tuning (LLaVA).

################################################################################
