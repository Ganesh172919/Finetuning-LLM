"""
################################################################################
VIDEO UNDERSTANDING — ANALYZING VIDEO CONTENT
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Video Understanding?
    Models that analyze and understand video content.

Key Tasks:
    - Video classification
    - Action recognition
    - Video captioning
    - Video question answering

Architecture:
    Video Frames → Frame Encoder → Temporal Aggregation → Output

Interview Questions:
    Q: "How do you process video with AI?"
    A: Encode each frame with vision encoder,
       then aggregate temporally with attention or pooling.

################################################################################
"""

import numpy as np
from typing import List

################################################################################
# SECTION 1: FRAME ENCODER
################################################################################

class FrameEncoder:
    """
    Frame Encoder
    =============

    Encodes individual video frames.
    """

    def __init__(self, d_model: int = 256):
        self.d_model = d_model

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Encode a single frame."""
        return np.random.randn(self.d_model) * 0.1

    def encode_batch(self, frames: np.ndarray) -> np.ndarray:
        """Encode batch of frames."""
        batch = frames.shape[0]
        return np.random.randn(batch, self.d_model) * 0.1


################################################################################
# SECTION 2: TEMPORAL AGGREGATION
################################################################################

class TemporalAggregator:
    """
    Temporal Aggregation
    ====================

    Aggregates frame features across time.

    Methods:
    1. Mean pooling: simple average
    2. Attention: learn to weight frames
    3. Transformer: model temporal relationships
    """

    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        self.attn_weight = np.random.randn(d_model, 1) * 0.02

    def aggregate(self, frame_features: np.ndarray) -> np.ndarray:
        """
        Aggregate frame features.

        Args:
            frame_features: [n_frames × d_model]

        Returns:
            video_feature: [d_model]
        """
        # Attention-weighted pooling
        scores = frame_features @ self.attn_weight
        weights = np.exp(scores - np.max(scores))
        weights = weights / np.sum(weights)

        return np.sum(frame_features * weights, axis=0)


################################################################################
# SECTION 3: VIDEO MODEL
################################################################################

class VideoUnderstandingModel:
    """
    Video Understanding Model
    ==========================

    Analyzes video content.

    Interview Questions:
        Q: "How does video understanding differ from image understanding?"
        A: Need to model temporal relationships between frames.
           Motion, actions, and events unfold over time.
    """

    def __init__(self, d_model: int = 256, n_classes: int = 400):
        self.frame_encoder = FrameEncoder(d_model)
        self.temporal_agg = TemporalAggregator(d_model)
        self.classifier = np.random.randn(d_model, n_classes) * 0.02

    def classify(self, frames: np.ndarray) -> np.ndarray:
        """
        Classify video.

        Args:
            frames: [n_frames × channels × height × width]

        Returns:
            logits: [n_classes]
        """
        frame_features = self.frame_encoder.encode_batch(frames)
        video_feature = self.temporal_agg.aggregate(frame_features)
        return video_feature @ self.classifier


################################################################################
# SECTION 4: TESTING
################################################################################

def demonstrate_video():
    """Demonstrate video understanding."""
    print("=" * 70)
    print("VIDEO UNDERSTANDING DEMONSTRATION")
    print("=" * 70)

    model = VideoUnderstandingModel(d_model=64, n_classes=10)

    # Simulate video
    frames = np.random.randn(16, 3, 64, 64)
    logits = model.classify(frames)
    print(f"Frames: {frames.shape}")
    print(f"Logits: {logits.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_video()
