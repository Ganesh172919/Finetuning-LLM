"""
################################################################################
SPEECH MODELS — UNDERSTANDING AND GENERATING SPEECH
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Speech Models?
    Models that understand (speech-to-text) or generate (text-to-speech) speech.
    They process audio waveforms or spectrograms.

Why does it matter?
    Speech AI powers:
    - Voice assistants (Siri, Alexa, Google Assistant)
    - Transcription (Whisper)
    - Accessibility (screen readers)
    - Content creation (podcasts, audiobooks)
    - Translation (speech-to-speech)

Historical Evolution:
    - 2012: Deep speech (Baidu)
    - 2015: Attention-based ASR
    - 2020: wav2vec (self-supervised)
    - 2022: Whisper (OpenAI)
    - 2023: Bark, Tortoise TTS
    - 2024: Voice cloning, zero-shot TTS
    - 2025: Real-time speech-to-speech

########################################

MODELS IMPLEMENTED:
1. whisper.py — Speech recognition (ASR)
2. tts.py — Text-to-speech synthesis
3. voice_clone.py — Voice cloning

################################################################################
"""

from .whisper import WhisperModel
from .tts import TextToSpeech
