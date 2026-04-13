"""
################################################################################
CLIP — CONTRASTIVE LANGUAGE-IMAGE PRE-TRAINING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is CLIP?
    CLIP learns to connect images and text by training on millions of
    (image, text) pairs from the internet. It learns that an image of
    a dog and the word "dog" should have similar representations.

Why does it matter?
    CLIP is the foundation of:
    - Text-to-image generation (Stable Diffusion uses CLIP text encoder)
    - Image search (find images by text description)
    - Zero-shot classification (classify without training)
    - Vision-Language Models (LLaVA, GPT-4V)

How does it work?
    1. Image Encoder: converts images to vectors
    2. Text Encoder: converts text to vectors
    3. Contrastive Loss: pulls matching pairs together, pushes non-matching apart

    Training data: 400 million (image, text) pairs from the internet

    The key insight: by learning to match images with their captions,
    CLIP learns a shared embedding space where vision and language meet.

Architecture:
    ┌─────────────────────────────────────────────────┐
    │ Image: [dog photo]                               │
    │   ↓                                              │
    │ Vision Encoder (ViT) → image_embedding          │
    │                                                  │
    │ Text: "a photo of a dog"                         │
    │   ↓                                              │
    │ Text Encoder (Transformer) → text_embedding     │
    │                                                  │
    │ Contrastive Loss:                                │
    │ - Maximize similarity of matching pairs          │
    │ - Minimize similarity of non-matching pairs      │
    └─────────────────────────────────────────────────┘

Interview Questions:
    1. "What is CLIP and how does it work?"
       CLIP learns to match images with their text descriptions.
       It uses contrastive learning to align vision and language
       in a shared embedding space.

    2. "Why is CLIP important for text-to-image generation?"
       Stable Diffusion uses CLIP's text encoder to understand prompts.
       The text embedding guides the image generation process.

    3. "What's zero-shot classification?"
       Classify images without any training examples.
       Compare image embedding with text embeddings of class names.
       The closest text embedding determines the class.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List
import math

import sys
sys.path.append('..')
from ..02_transformers.attention import MultiHeadAttention
from ..02_transformers.layers import RMSNorm, FeedForward, TransformerBlock


################################################################################
# SECTION 1: VISION ENCODER (ViT-based)
################################################################################

class VisionEncoder:
    """
    Vision Transformer (ViT) Encoder
    ==================================

    Definition: A transformer that processes images by splitting them
    into patches and treating each patch as a token.

    How it works:
    1. Split image into patches (e.g., 16×16 pixels)
    2. Flatten each patch into a vector
    3. Add position embeddings
    4. Process with transformer blocks
    5. Output: one vector per patch + CLS token

    Architecture:
    Image (224×224) → 196 patches (16×16 each)
    Each patch → linear projection → 768-dim vector
    + CLS token → 197 tokens
    → Transformer (12 layers) → 197 × 768 embeddings
    → CLS token is the image representation

    Interview Question:
        "How does ViT process images?"
        It splits the image into patches, linearly projects them,
        and processes them like tokens in a transformer.
        The CLS token aggregates information from all patches.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model

        # Number of patches
        self.n_patches = (image_size // patch_size) ** 2

        # Patch embedding: project patch to d_model dimensions
        patch_dim = patch_size * patch_size * in_channels
        scale = math.sqrt(2.0 / (patch_dim + d_model))
        self.patch_embedding = np.random.randn(patch_dim, d_model) * scale

        # CLS token (learnable)
        self.cls_token = np.random.randn(1, 1, d_model) * 0.02

        # Position embeddings (learnable)
        self.position_embedding = np.random.randn(self.n_patches + 1, d_model) * 0.02

        # Transformer blocks
        self.layers = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        self.norm = RMSNorm(d_model)

    def patchify(self, image: np.ndarray) -> np.ndarray:
        """
        Split image into patches.

        Args:
            image: [batch × channels × height × width]

        Returns:
            patches: [batch × n_patches × patch_dim]
        """
        batch, channels, height, width = image.shape
        p = self.patch_size

        # Reshape to extract patches
        patches = image.reshape(batch, channels, height // p, p, width // p, p)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)  # [batch × h/p × w/p × c × p × p]
        patches = patches.reshape(batch, self.n_patches, -1)  # [batch × n_patches × patch_dim]

        return patches

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image to feature vectors.

        Args:
            image: [batch × channels × height × width]

        Returns:
            features: [batch × n_patches+1 × d_model]
        """
        batch = image.shape[0]

        # Step 1: Extract patches
        patches = self.patchify(image)  # [batch × n_patches × patch_dim]

        # Step 2: Project patches to d_model
        x = np.matmul(patches, self.patch_embedding)  # [batch × n_patches × d_model]

        # Step 3: Prepend CLS token
        cls = np.broadcast_to(self.cls_token, (batch, 1, self.d_model))
        x = np.concatenate([cls, x], axis=1)  # [batch × n_patches+1 × d_model]

        # Step 4: Add position embeddings
        x = x + self.position_embedding

        # Step 5: Transformer processing
        for layer in self.layers:
            x = layer.forward(x)

        x = self.norm.forward(x)

        return x


################################################################################
# SECTION 2: TEXT ENCODER
################################################################################

class TextEncoder:
    """
    Text Encoder (Transformer)
    ===========================

    Definition: A transformer that encodes text into vectors.

    Architecture: Standard transformer encoder
    - Token embedding
    - Position embedding (learned)
    - Transformer blocks
    - CLS token or mean pooling

    For CLIP, the text encoder outputs a single vector
    representing the entire text.
    """

    def __init__(
        self,
        vocab_size: int = 49408,  # CLIP vocabulary size
        max_seq_len: int = 77,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8
    ):
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        scale = math.sqrt(2.0 / (vocab_size + d_model))
        self.token_embedding = np.random.randn(vocab_size, d_model) * scale

        # Position embedding (learned)
        self.position_embedding = np.random.randn(max_seq_len, d_model) * 0.02

        # Transformer blocks
        self.layers = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        self.norm = RMSNorm(d_model)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Encode text tokens to feature vectors.

        Args:
            token_ids: [batch × seq_len]

        Returns:
            features: [batch × d_model] (EOS token representation)
        """
        batch, seq_len = token_ids.shape

        # Token + position embedding
        x = self.token_embedding[token_ids] + self.position_embedding[:seq_len]

        # Transformer processing
        for layer in self.layers:
            x = layer.forward(x)

        x = self.norm.forward(x)

        # Take the EOS token (last token) as text representation
        # In CLIP, the EOS token aggregates text information
        text_features = x[:, -1, :]  # [batch × d_model]

        return text_features


################################################################################
# SECTION 3: CLIP MODEL
################################################################################

class CLIP:
    """
    CLIP: Contrastive Language-Image Pre-training
    ===============================================

    Definition: A model that learns to connect images and text
    through contrastive learning.

    Training:
    Given a batch of N (image, text) pairs:
    - Compute image embeddings: I₁, I₂, ..., I_N
    - Compute text embeddings: T₁, T₂, ..., T_N
    - Contrastive loss:
      - For each image Iᵢ, the matching text Tᵢ should have high similarity
      - All non-matching texts should have low similarity
      - Same for each text with all images

    The loss is symmetric:
    L = (L_image_to_text + L_text_to_image) / 2

    Similarity:
    sim(I, T) = I · T / (||I|| × ||T||)  (cosine similarity)

    Temperature:
    A learnable temperature parameter τ scales the logits:
    logits = sim(I, T) / τ

    Interview Questions:
        1. "How does CLIP training work?"
           CLIP uses contrastive learning on (image, text) pairs.
           It maximizes similarity for matching pairs and minimizes
           for non-matching pairs.

        2. "What can you do with CLIP?"
           Zero-shot classification, image search, text-to-image guidance,
           image captioning, and more.

        3. "What are the limitations of CLIP?"
           Cannot generate images, struggles with counting and spatial reasoning,
           biased toward common internet patterns.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        vision_d_model: int = 768,
        vision_layers: int = 12,
        vision_heads: int = 12,
        text_vocab_size: int = 49408,
        text_max_seq_len: int = 77,
        text_d_model: int = 512,
        text_layers: int = 12,
        text_heads: int = 8,
        embed_dim: int = 512
    ):
        self.embed_dim = embed_dim

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            d_model=vision_d_model,
            n_layers=vision_layers,
            n_heads=vision_heads
        )

        # Vision projection: vision_d_model → embed_dim
        self.vision_projection = np.random.randn(vision_d_model, embed_dim) * 0.02

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=text_vocab_size,
            max_seq_len=text_max_seq_len,
            d_model=text_d_model,
            n_layers=text_layers,
            n_heads=text_heads
        )

        # Text projection: text_d_model → embed_dim
        self.text_projection = np.random.randn(text_d_model, embed_dim) * 0.02

        # Learnable temperature
        self.logit_scale = np.log(1 / 0.07)  # Initial temperature: 0.07

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image to CLIP embedding space.

        Args:
            image: [batch × channels × height × width]

        Returns:
            image_embedding: [batch × embed_dim]
        """
        # Get vision features (CLS token)
        vision_features = self.vision_encoder.forward(image)
        cls_token = vision_features[:, 0, :]  # [batch × vision_d_model]

        # Project to shared embedding space
        image_embedding = np.matmul(cls_token, self.vision_projection)

        # Normalize
        norms = np.linalg.norm(image_embedding, axis=-1, keepdims=True)
        image_embedding = image_embedding / (norms + 1e-8)

        return image_embedding

    def encode_text(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Encode text to CLIP embedding space.

        Args:
            token_ids: [batch × seq_len]

        Returns:
            text_embedding: [batch × embed_dim]
        """
        # Get text features (EOS token)
        text_features = self.text_encoder.forward(token_ids)

        # Project to shared embedding space
        text_embedding = np.matmul(text_features, self.text_projection)

        # Normalize
        norms = np.linalg.norm(text_embedding, axis=-1, keepdims=True)
        text_embedding = text_embedding / (norms + 1e-8)

        return text_embedding

    def forward(
        self,
        images: np.ndarray,
        token_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass: compute contrastive loss.

        Args:
            images: [batch × channels × height × width]
            token_ids: [batch × seq_len]

        Returns:
            logits_per_image: [batch × batch] similarity scores
            logits_per_text: [batch × batch] similarity scores
            loss: Contrastive loss
        """
        # Encode images and text
        image_embeddings = self.encode_image(images)  # [batch × embed_dim]
        text_embeddings = self.encode_text(token_ids)  # [batch × embed_dim]

        # Compute similarity
        # sim(i, j) = image_i · text_j
        similarity = np.matmul(image_embeddings, text_embeddings.T)  # [batch × batch]

        # Scale by temperature
        logit_scale = np.exp(self.logit_scale)
        logits_per_image = similarity * logit_scale
        logits_per_text = similarity.T * logit_scale

        # Contrastive loss
        batch_size = images.shape[0]
        labels = np.arange(batch_size)

        loss_image = self._cross_entropy(logits_per_image, labels)
        loss_text = self._cross_entropy(logits_per_text, labels)
        loss = (loss_image + loss_text) / 2

        return logits_per_image, logits_per_text, loss

    @staticmethod
    def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
        log_probs = shifted[np.arange(len(targets)), targets] - log_sum_exp
        return -np.mean(log_probs)


################################################################################
# SECTION 4: CONTRASTIVE LOSS
################################################################################

class ContrastiveLoss:
    """
    Contrastive Loss (InfoNCE)
    ===========================

    Definition: Pull matching pairs together, push non-matching apart.

    Formula:
    L = -log(exp(sim(i,t)/τ) / Σ_j exp(sim(i,tⱼ)/τ))

    For a batch of N pairs:
    - N matching pairs (positive)
    - N²-N non-matching pairs (negative)

    Temperature τ:
    - Controls how "sharp" the distribution is
    - Small τ: very peaked (hard negatives matter more)
    - Large τ: smoother (all negatives matter equally)
    - Learnable in CLIP (initialized to 0.07)

    Interview Question:
        "What is contrastive learning?"
        Learning by comparing: pull similar things together,
        push different things apart. CLIP uses it to align
        images and text in a shared embedding space.
    """

    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature

    def compute_loss(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray
    ) -> float:
        """
        Compute contrastive loss between two sets of embeddings.

        Args:
            embeddings_a: [batch × dim] (e.g., image embeddings)
            embeddings_b: [batch × dim] (e.g., text embeddings)

        Returns:
            loss: Contrastive loss
        """
        # Normalize embeddings
        norms_a = np.linalg.norm(embeddings_a, axis=-1, keepdims=True)
        norms_b = np.linalg.norm(embeddings_b, axis=-1, keepdims=True)
        embeddings_a = embeddings_a / (norms_a + 1e-8)
        embeddings_b = embeddings_b / (norms_b + 1e-8)

        # Compute similarity matrix
        similarity = np.matmul(embeddings_a, embeddings_b.T) / self.temperature

        # Labels: diagonal is matching
        batch_size = embeddings_a.shape[0]
        labels = np.arange(batch_size)

        # Cross-entropy loss
        loss_a = self._cross_entropy(similarity, labels)
        loss_b = self._cross_entropy(similarity.T, labels)

        return (loss_a + loss_b) / 2

    @staticmethod
    def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
        log_probs = shifted[np.arange(len(targets)), targets] - log_sum_exp
        return -np.mean(log_probs)


################################################################################
# SECTION 5: TESTING & EXAMPLES
################################################################################

def demonstrate_clip():
    """Demonstrate CLIP model."""
    print("=" * 70)
    print("CLIP DEMONSTRATION")
    print("=" * 70)

    # Create small CLIP model
    print("\n--- Creating CLIP Model ---")
    clip = CLIP(
        image_size=64,  # Small for demo
        patch_size=8,
        vision_d_model=64,
        vision_layers=2,
        vision_heads=4,
        text_vocab_size=1000,
        text_max_seq_len=32,
        text_d_model=64,
        text_layers=2,
        text_heads=4,
        embed_dim=64
    )

    # Create dummy data
    batch_size = 4
    images = np.random.randn(batch_size, 3, 64, 64)
    token_ids = np.random.randint(0, 1000, (batch_size, 16))

    # Forward pass
    print("\n--- Forward Pass ---")
    logits_image, logits_text, loss = clip.forward(images, token_ids)
    print(f"Image logits shape: {logits_image.shape}")
    print(f"Text logits shape: {logits_text.shape}")
    print(f"Contrastive loss: {loss:.4f}")

    # Encode separately
    print("\n--- Encoding ---")
    image_emb = clip.encode_image(images)
    text_emb = clip.encode_text(token_ids)
    print(f"Image embedding shape: {image_emb.shape}")
    print(f"Text embedding shape: {text_emb.shape}")

    # Similarity
    print("\n--- Similarity ---")
    similarity = np.matmul(image_emb, text_emb.T)
    print(f"Similarity matrix:\n{similarity.round(3)}")

    # Contrastive loss
    print("\n--- Contrastive Loss ---")
    contrastive = ContrastiveLoss(temperature=0.07)
    loss = contrastive.compute_loss(image_emb, text_emb)
    print(f"Contrastive loss: {loss:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_clip()


################################################################################
# REFERENCES
################################################################################

# [1] Radford, A., et al. (2021). Learning Transferable Visual Models From NL Supervision.
# [2] Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words.
# [3] Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training.
# [4] Liu, H., et al. (2023). Visual Instruction Tuning (LLaVA).

################################################################################
