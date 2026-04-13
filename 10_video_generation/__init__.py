"""
################################################################################
VIDEO GENERATION MODELS — CREATING MOVING IMAGES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Video Generation Models?
    Models that generate video sequences from text, images, or other videos.
    They extend image generation to the temporal dimension.

Why does it matter?
    Video generation powers:
    - Content creation (movies, ads)
    - Simulation (training robots, autonomous driving)
    - Education (animated explanations)
    - Entertainment (games, virtual worlds)

Historical Evolution:
    - 2022: Make-A-Video (Meta)
    - 2023: Runway Gen-1, Pika
    - 2024: Sora (OpenAI), Veo (Google)
    - 2025: Kling, Seedance, Wan
    - 2026: Real-time video generation

Key Challenges:
    - Temporal consistency (frames must be coherent)
    - Motion modeling (objects move realistically)
    - Long videos (many frames = lots of compute)
    - Quality (each frame must be high quality)

########################################

APPROACHES:
1. Image-to-Video: Generate frames from image + text
2. Text-to-Video: Generate frames from text only
3. Video Diffusion: Extend image diffusion to video
4. Autoregressive: Generate frames one at a time
5. Hybrid: Combine multiple approaches

################################################################################
"""

from .video_diffusion import VideoDiffusionModel
from .temporal_attention import TemporalAttention
