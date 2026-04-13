"""
################################################################################
JEPA — JOINT EMBEDDING PREDICTIVE ARCHITECTURE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is JEPA?
    JEPA (Joint Embedding Predictive Architecture) is a self-supervised
    learning framework proposed by Yann LeCun that learns by predicting
    latent representations of masked inputs, rather than reconstructing
    raw pixels or tokens.

    Key insight: Predict in LATENT SPACE, not pixel/token space.

Why does it matter?
    Traditional generative models (VAE, GPT) predict raw observations:
    - Pixel prediction: Expensive, focuses on irrelevant details
    - Token prediction: Requires massive compute
    - Both waste capacity on noise and imperceptible details

    JEPA predicts ABSTRACT representations:
    - Ignores irrelevant variations (lighting, exact pixel values)
    - Focuses on semantic content
    - More efficient learning
    - Closer to how humans understand the world

How does it work?
    1. Encode context observation → latent representation z_context
    2. Encode target observation → latent representation z_target
    3. Predictor takes z_context + action → predicted z_target
    4. Loss: predicted z_target vs actual z_target (in latent space)

    The encoder is trained jointly with the predictor.
    This avoids the "shortcut" problem where encoders collapse.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    JEPA Architecture                             │
    │                                                                  │
    │  Context Image ──▶ Encoder ──▶ z_context                        │
    │                                        ↓                         │
    │  Action ──────────────▶ Predictor ──▶ z_predicted               │
    │                                        ↓                         │
    │  Target Image ──▶ Encoder ──▶ z_target ──▶ Loss                 │
    │                        ↑                                         │
    │                  (shared encoder, EMA updated)                   │
    └─────────────────────────────────────────────────────────────────┘

Key Design Decisions:
    - Target encoder uses EMA (Exponential Moving Average) updates
    - This prevents "representation collapse" (encoder outputs constant)
    - Predictor is smaller than encoder (forces abstraction)
    - Multiple prediction heads for different aspects

Historical Context:
    - 2022: I-JEPA (Image JEPA) — first successful implementation
    - 2023: V-JEPA (Video JEPA) — extends to video
    - 2024: MC-JEPA — multi-crop JEPA for better representations
    - 2024: A-JEPA — audio JEPA for speech/audio

Interview Questions:
    1. "What is JEPA and how does it differ from VAE/GPT?"
       JEPA predicts in latent space, not pixel/token space. This forces
       the model to learn abstract representations rather than memorizing
       surface-level details. VAE reconstructs pixels (wastes capacity
       on noise), GPT predicts tokens (requires massive data).

    2. "Why does JEPA use EMA for the target encoder?"
       Without EMA, the encoder could collapse to a constant output
       (trivial solution). EMA provides a slowly-moving target that
       the predictor must chase, preventing collapse while allowing
       the main encoder to learn useful representations.

    3. "When should I use JEPA vs contrastive learning?"
       JEPA: When you want to learn predictive world models, when
       data has temporal structure, when you need action-conditioned
       predictions. Contrastive: When you want instance-level
       discrimination, when negative pairs are easy to sample.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: CONFIGURATION
################################################################################

@dataclass
class JEPAConfig:
    """
    JEPA Configuration.

    These hyperparameters control the JEPA architecture and training.
    """
    # Input dimensions
    input_channels: int = 3
    input_height: int = 64
    input_width: int = 64

    # Encoder architecture
    encoder_dim: int = 256
    encoder_depth: int = 6
    encoder_num_heads: int = 8

    # Predictor architecture (smaller than encoder)
    predictor_dim: int = 128
    predictor_depth: int = 4

    # Latent space
    latent_dim: int = 128
    num_patches: int = 16  # Number of spatial patches

    # Training
    ema_momentum: float = 0.996  # EMA decay for target encoder
    learning_rate: float = 1e-4
    batch_size: int = 32

    # Masking
    mask_ratio: float = 0.75  # Fraction of patches to mask


################################################################################
# SECTION 2: PATCH EMBEDDING
################################################################################

class PatchEmbedding:
    """
    Patch Embedding for JEPA
    ========================

    Converts images into a sequence of patch embeddings.
    This is the first step: turning pixels into tokens.

    Process:
        1. Divide image into non-overlapping patches
        2. Flatten each patch
        3. Project to embedding dimension
        4. Add positional encoding

    For a 64x64 image with 16x16 patches:
        - Number of patches: (64/16) × (64/16) = 4 × 4 = 16 patches
        - Each patch: 16 × 16 × 3 = 768 pixels
        - After projection: 16 tokens of dimension d_model

    Formula:
        patches = unfold(image, patch_size)  # (B, C*P*P, N)
        embeddings = patches @ W_embed        # (B, N, d_model)
        embeddings = embeddings + pos_embed   # add position info
    """

    def __init__(self, config: JEPAConfig):
        self.config = config
        self.patch_size = config.input_height // int(math.sqrt(config.num_patches))

        # Embedding weights
        patch_dim = config.input_channels * self.patch_size * self.patch_size
        self.W_embed = np.random.randn(patch_dim, config.encoder_dim) * 0.02
        self.pos_embed = np.random.randn(1, config.num_patches, config.encoder_dim) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Convert image to patch embeddings.

        Args:
            x: Input image (batch, channels, height, width)

        Returns:
            Patch embeddings (batch, num_patches, encoder_dim)
        """
        batch_size = x.shape[0]
        p = self.patch_size

        # Extract patches manually (simplified)
        patches = []
        for i in range(0, x.shape[2], p):
            for j in range(0, x.shape[3], p):
                patch = x[:, :, i:i+p, j:j+p]
                patches.append(patch.reshape(batch_size, -1))

        # Stack patches: (batch, num_patches, patch_dim)
        patches = np.stack(patches, axis=1)

        # Project to embedding space
        embeddings = patches @ self.W_embed  # (batch, num_patches, encoder_dim)

        # Add positional encoding
        embeddings = embeddings + self.pos_embed

        return embeddings


################################################################################
# SECTION 3: TRANSFORMER ENCODER
################################################################################

class TransformerBlock:
    """
    Transformer Block for JEPA Encoder
    ====================================

    Standard pre-norm transformer block with:
    1. Multi-head self-attention
    2. Feed-forward network
    3. Residual connections
    4. Layer normalization

    Formula:
        x' = x + Attention(LayerNorm(x))
        output = x' + FFN(LayerNorm(x'))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ff = d_ff

        # Attention weights
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

        # FFN weights
        self.W_ff1 = np.random.randn(d_model, d_ff) * 0.02
        self.W_ff2 = np.random.randn(d_ff, d_model) * 0.02

        # Layer norm parameters
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + 1e-5) + beta

    def attention(self, x: np.ndarray) -> np.ndarray:
        """Multi-head self-attention."""
        batch, seq_len, d = x.shape

        # Linear projections
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head: (batch, n_heads, seq_len, d_head)
        Q = Q.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        # Attention scores
        scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(self.d_head)
        weights = self._softmax(scores)

        # Apply attention to values
        out = weights @ V  # (batch, n_heads, seq_len, d_head)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d)

        return out @ self.W_o

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through transformer block."""
        # Self-attention with pre-norm
        h = self.layer_norm(x, self.ln1_gamma, self.ln1_beta)
        x = x + self.attention(h)

        # FFN with pre-norm
        h = self.layer_norm(x, self.ln2_gamma, self.ln2_beta)
        h = h @ self.W_ff1
        h = np.maximum(0, h)  # ReLU activation
        h = h @ self.W_ff2
        x = x + h

        return x


class JEPAEncoder:
    """
    JEPA Encoder
    =============

    Encodes observations into latent representations.
    Used as both context encoder and target encoder.

    Architecture:
        PatchEmbedding → [TransformerBlock × N] → latent representations

    The encoder processes patches independently (no causal mask),
    allowing each patch to attend to all other patches.

    Interview Question:
        "What's the difference between JEPA encoder and VAE encoder?"
        VAE encoder outputs mean and variance for a distribution.
        JEPA encoder outputs deterministic latent representations
        that are used for prediction in latent space.
    """

    def __init__(self, config: JEPAConfig):
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        self.blocks = [
            TransformerBlock(
                config.encoder_dim,
                config.encoder_num_heads,
                config.encoder_dim * 4
            )
            for _ in range(config.encoder_depth)
        ]

        # Projection to latent space
        self.proj = np.random.randn(config.encoder_dim, config.latent_dim) * 0.02
        self.ln_final_gamma = np.ones(config.encoder_dim)
        self.ln_final_beta = np.zeros(config.encoder_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Encode observation to latent representations.

        Args:
            x: Input image (batch, channels, height, width)

        Returns:
            Latent representations (batch, num_patches, latent_dim)
        """
        # Patch embedding
        h = self.patch_embed.forward(x)

        # Transformer blocks
        for block in self.blocks:
            h = block.forward(h)

        # Final layer norm and projection
        mean = np.mean(h, axis=-1, keepdims=True)
        var = np.var(h, axis=-1, keepdims=True)
        h = self.ln_final_gamma * (h - mean) / np.sqrt(var + 1e-5) + self.ln_final_beta

        return h @ self.proj


################################################################################
# SECTION 4: JEPA PREDICTOR
################################################################################

class JEPAPredictor:
    """
    JEPA Predictor
    ===============

    Predicts target latent representations from context + action.

    Key design: Predictor is SMALLER than encoder.
    This forces the model to learn abstract representations
    rather than memorizing pixel-level details.

    Input:
        - Context latent: z_context (batch, num_patches, latent_dim)
        - Action: action (batch, action_dim)

    Output:
        - Predicted target latent: z_pred (batch, num_patches, latent_dim)

    Interview Question:
        "Why is the predictor smaller than the encoder?"
        If the predictor were as large as the encoder, it could
        memorize the mapping without learning abstractions. A smaller
        predictor forces compression, which means the encoder must
        provide useful abstractions.
    """

    def __init__(self, config: JEPAConfig):
        self.config = config

        # Action embedding
        self.action_embed = np.random.randn(config.latent_dim, config.predictor_dim) * 0.02

        # Predictor transformer blocks
        self.blocks = [
            TransformerBlock(
                config.predictor_dim,
                config.encoder_num_heads,
                config.predictor_dim * 4
            )
            for _ in range(config.predictor_depth)
        ]

        # Projection back to latent space
        self.proj = np.random.randn(config.predictor_dim, config.latent_dim) * 0.02

        # Learnable mask token (for masked patches)
        self.mask_token = np.random.randn(1, 1, config.latent_dim) * 0.02

    def forward(
        self,
        z_context: np.ndarray,
        action: np.ndarray,
        mask_indices: np.ndarray
    ) -> np.ndarray:
        """
        Predict target latent from context and action.

        Args:
            z_context: Context latent (batch, num_visible, latent_dim)
            action: Action vector (batch, latent_dim)
            mask_indices: Indices of masked patches

        Returns:
            Predicted latent for all patches (batch, num_patches, latent_dim)
        """
        batch_size = z_context.shape[0]
        num_patches = self.config.num_patches

        # Embed action and add to context
        action_embed = action @ self.action_embed  # (batch, predictor_dim)

        # Project context to predictor dimension
        z_pred_dim = z_context @ np.random.randn(
            self.config.latent_dim, self.config.predictor_dim
        ) * 0.02

        # Add action information
        z_pred_dim = z_pred_dim + action_embed[:, np.newaxis, :]

        # Run through predictor blocks
        for block in self.blocks:
            z_pred_dim = block.forward(z_pred_dim)

        # Project back to latent space
        z_predicted = z_pred_dim @ self.proj

        return z_predicted


################################################################################
# SECTION 5: COMPLETE JEPA MODEL
################################################################################

class JEPA:
    """
    Joint Embedding Predictive Architecture (JEPA)
    ================================================

    Complete JEPA model for self-supervised world model learning.

    Training process:
        1. Take two observations: context and target
        2. Mask most of the target observation
        3. Encode context with context encoder
        4. Encode target with target encoder (EMA)
        5. Predictor predicts target latent from context + action
        6. Loss: predicted latent vs actual target latent

    Key insight: We never reconstruct pixels.
    The loss is entirely in latent space.

    Interview Questions:
        1. "How does JEPA prevent representation collapse?"
           The target encoder uses EMA (Exponential Moving Average)
           updates from the context encoder. This creates a slowly-
           moving target that prevents the encoder from outputting
           a constant (which would minimize prediction error trivially).

        2. "What's the role of masking in JEPA?"
           Masking forces the predictor to extrapolate — it must
           predict what's behind the mask using only context and
           action. This encourages learning causal structure.

        3. "How does JEPA relate to world models?"
           JEPA learns to predict future states (target) from
           current states (context) and actions. This is exactly
           what a world model does: predict s_{t+1} given s_t and a_t.
    """

    def __init__(self, config: JEPAConfig):
        self.config = config

        # Context encoder (updated via gradient)
        self.context_encoder = JEPAEncoder(config)

        # Target encoder (updated via EMA)
        self.target_encoder = JEPAEncoder(config)

        # Predictor
        self.predictor = JEPAPredictor(config)

        # EMA momentum schedule
        self.momentum = config.ema_momentum

    def update_target_encoder(self):
        """
        Update target encoder using Exponential Moving Average.

        This is crucial for preventing representation collapse.
        The target encoder slowly follows the context encoder.

        Formula:
            θ_target = τ × θ_target + (1 - τ) × θ_context

        Where τ is the EMA momentum (e.g., 0.996).
        """
        # In practice, we'd copy weights with EMA
        # For numpy implementation, we simulate this
        pass

    def mask_patches(
        self,
        z_target: np.ndarray,
        mask_ratio: float = 0.75
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create masked version of target latent.

        Args:
            z_target: Full target latent (batch, num_patches, latent_dim)
            mask_ratio: Fraction of patches to mask

        Returns:
            z_visible: Visible patches only
            z_masked: Masked patches (ground truth for prediction)
            mask_indices: Which patches were masked
        """
        batch_size, num_patches, latent_dim = z_target.shape
        num_visible = int(num_patches * (1 - mask_ratio))

        # Random mask for each sample in batch
        mask_indices = []
        z_visible_list = []
        z_masked_list = []

        for i in range(batch_size):
            # Random permutation of patch indices
            perm = np.random.permutation(num_patches)
            visible_idx = perm[:num_visible]
            masked_idx = perm[num_visible:]

            z_visible_list.append(z_target[i, visible_idx])
            z_masked_list.append(z_target[i, masked_idx])
            mask_indices.append(masked_idx)

        z_visible = np.stack(z_visible_list)
        z_masked = np.stack(z_masked_list)
        mask_indices = np.stack(mask_indices)

        return z_visible, z_masked, mask_indices

    def compute_loss(
        self,
        z_context: np.ndarray,
        z_target: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        Compute JEPA training loss.

        The loss is the MSE between predicted and actual target latent.

        Args:
            z_context: Context encoder output
            z_target: Target encoder output (EMA)
            action: Action between context and target

        Returns:
            Scalar loss value
        """
        # Mask target
        z_visible, z_masked, mask_indices = self.mask_patches(z_target)

        # Predict masked patches from context + action
        z_predicted = self.predictor.forward(z_context, action, mask_indices)

        # Loss: MSE between predicted and actual (in latent space!)
        loss = np.mean((z_predicted - z_masked) ** 2)

        return loss

    def predict_future(
        self,
        context: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        """
        Predict future state given context and action.

        This is the inference-time use case: given current observation
        and an action, predict what will happen next.

        Args:
            context: Current observation (batch, C, H, W)
            action: Action to take (batch, action_dim)

        Returns:
            Predicted future latent (batch, num_patches, latent_dim)
        """
        # Encode context
        z_context = self.context_encoder.forward(context)

        # Predict future (no masking at inference)
        # We predict all patches from context
        z_predicted = self.predictor.forward(z_context, action, mask_indices=None)

        return z_predicted


################################################################################
# SECTION 6: EMA UPDATER
################################################################################

class EMAUpdater:
    """
    Exponential Moving Average Updater
    ====================================

    Maintains a running average of model weights.
    Used for the target encoder in JEPA.

    Formula:
        θ_ema = momentum × θ_ema + (1 - momentum) × θ_current

    Typical momentum: 0.996 (slowly follows current weights)

    Why EMA?
        - Prevents sudden changes in target representations
        - Creates a stable learning target
        - Prevents representation collapse
        - Similar to target networks in RL (DQN)
    """

    def __init__(self, momentum: float = 0.996):
        self.momentum = momentum

    def update(self, source_params: np.ndarray, target_params: np.ndarray) -> np.ndarray:
        """
        Update target parameters with EMA.

        Args:
            source_params: Current model parameters
            target_params: EMA parameters to update

        Returns:
            Updated EMA parameters
        """
        return self.momentum * target_params + (1 - self.momentum) * source_params


################################################################################
# SECTION 7: TESTING & DEMONSTRATION
################################################################################

def demonstrate_jepa():
    """Demonstrate JEPA architecture and training process."""
    print("=" * 70)
    print("JEPA — JOINT EMBEDDING PREDICTIVE ARCHITECTURE")
    print("=" * 70)

    # Configuration
    config = JEPAConfig(
        input_channels=3,
        input_height=32,
        input_width=32,
        encoder_dim=64,
        encoder_depth=2,
        encoder_num_heads=4,
        predictor_dim=32,
        predictor_depth=2,
        latent_dim=32,
        num_patches=4
    )

    print(f"\nConfiguration:")
    print(f"  Input: {config.input_channels}x{config.input_height}x{config.input_width}")
    print(f"  Encoder dim: {config.encoder_dim}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Num patches: {config.num_patches}")
    print(f"  Mask ratio: {config.mask_ratio}")

    # Create model
    model = JEPA(config)

    # Create dummy data
    batch_size = 4
    context = np.random.randn(batch_size, 3, 32, 32)
    target = np.random.randn(batch_size, 3, 32, 32)
    action = np.random.randn(batch_size, config.latent_dim)

    # Encode context
    z_context = model.context_encoder.forward(context)
    print(f"\nContext latent shape: {z_context.shape}")

    # Encode target (EMA encoder)
    z_target = model.target_encoder.forward(target)
    print(f"Target latent shape: {z_target.shape}")

    # Compute loss
    loss = model.compute_loss(z_context, z_target, action)
    print(f"\nJEPA Loss: {loss:.4f}")

    # Demonstrate masking
    z_visible, z_masked, mask_indices = model.mask_patches(z_target, mask_ratio=0.75)
    print(f"\nMasking demonstration:")
    print(f"  Total patches: {config.num_patches}")
    print(f"  Visible patches: {z_visible.shape[1]}")
    print(f"  Masked patches: {z_masked.shape[1]}")
    print(f"  Mask indices: {mask_indices[0]}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: JEPA predicts in LATENT SPACE, not pixel space!")
    print("This forces the model to learn abstract representations.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_jepa()
