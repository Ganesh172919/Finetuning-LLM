# SOTA AI — State-of-the-Art AI from Scratch

## A Complete AI University & Production Research Lab

This repository is a comprehensive, educational implementation of state-of-the-art AI models,
built from first principles. Every file teaches. Every class explains. Every function documents
not just *what* it does, but *why* it exists, *how* it works, and *when* to use it.

---

## Who This Is For

- AI engineers transitioning toward research
- ML practitioners wanting deep understanding
- Students learning modern AI architectures
- Researchers exploring new model families
- Engineers building production AI systems

**Assumed knowledge:** Basic Python, high school math. Everything else is explained from scratch.

---

## Repository Structure

```
sota/
├── 01_math/                    # Mathematical foundations
├── 02_transformers/            # Core transformer architecture
├── 03_llm/                     # Large Language Models
├── 04_reasoning/               # Reasoning: CoT, ToT, Self-Consistency, MCTS, Reasoning Tokens
├── 05_moe/                     # Mixture of Experts
├── 06_slm/                     # Small Language Models: Phi4, Gemma3, Quantization, Pruning
├── 07_multimodal/              # Multimodal models
├── 08_vlm/                     # Vision Language Models
├── 09_image_generation/        # Image generation (diffusion, etc.)
├── 10_video_generation/        # Video generation models
├── 11_speech/                  # Speech models
├── 12_audio_generation/        # Audio generation
├── 13_agentic_ai/              # Agentic AI systems
├── 14_rag/                     # RAG: Basic, Self-RAG, CRAG, Graph RAG, Multimodal RAG
├── 15_long_context/            # Long Context: Ring Attention, Compression, Sliding Window
├── 16_code_generation/         # Code Gen: Reward Models, Code Search, AST Processing
├── 17_embedding/               # Embedding models
├── 18_recommendation/          # Recommendation systems
├── 19_reinforcement_learning/  # RL for language models
├── 20_robotics/                # Robotics: Imitation Learning, Robot Transformer, Sim-to-Real
├── 21_diffusion/               # Diffusion models
├── 22_transformer_variants/    # Transformer architecture variants
├── 23_emerging_2026/           # 2026 research architectures
├── common/                     # Shared utilities
├── training/                   # Training infrastructure
├── inference/                  # Inference & serving
├── evaluation/                 # Evaluation metrics & benchmarks
├── deployment/                 # Production deployment
└── docs/                       # Extended documentation
```

---

## Learning Path

### Phase 1: Foundations
1. **01_math/** — Linear algebra, probability, optimization
2. **02_transformers/** — The architecture behind modern AI

### Phase 2: Language Models
3. **03_llm/** — GPT, decoder-only, encoder-decoder
4. **04_reasoning/** — Chain of thought, tree of thoughts
5. **05_moe/** — Mixture of Experts
6. **06_slm/Small Language Models
7. **15_long_context/** — Long context architectures
8. **16_code_generation/** — Code generation models
9. **17_embedding/** — Embedding models

### Phase 3: Multimodal
10. **07_multimodal/** — CLIP, SigLIP, Flamingo
11. **08_vlm/** — Vision Language Models
12. **09_image_generation/** — Diffusion, Stable Diffusion, Flux
13. **10_video_generation/** — Video generation
14. **11_speech/** — Speech models
15. **12_audio_generation/** — Audio generation

### Phase 4: Advanced Systems
16. **13_agentic_ai/** — Agentic systems
17. **14_rag/** — Retrieval-Augmented Generation
18. **18_recommendation/** — Recommendation systems
19. **19_reinforcement_learning/** — RLHF, DPO, PPO
20. **20_robotics/** — Robotics foundation models

### Phase 5: Research Frontier
21. **21_diffusion/** — Diffusion model theory
22. **22_transformer_variants/** — Alternative architectures
23. **23_emerging_2026/** — 2026 research directions
    - **DeepSeek R1**: MLA, MoE with shared experts, GRPO training
    - **Claude 5**: Model tiers, agent teams, 1M context
    - **Gemini 3**: Sparse MoE (128 experts), Deep Think, Nano Banana
    - **GPT-5**: Auto-routing, native multimodal, safe completions
    - **Mamba-2**: Structured State Space Duality (2-8x faster)
    - **MoE-Mamba**: Mixture of Experts + Mamba (2.2x fewer steps)

---

## Design Principles

1. **Teach First** — Every file is a lesson, not just code
2. **From Scratch** — No black boxes; implement everything
3. **Production Ready** — Real engineering, not toy examples
4. **Mathematical Rigor** — Every equation explained intuitively
5. **Research Current** — Covering 2024–2026 SOTA

---

## How to Use This Repository

### For Learning
Read files top-to-bottom. Each module builds on previous ones.
Start with `01_math/` and progress sequentially.

### For Reference
Jump to any module. Each file is self-contained with cross-references.

### For Production
See `deployment/` and `inference/` for production-ready systems.

---

## Models Implemented

| Category | Models | Directory |
|----------|--------|-----------|
| LLMs | GPT, LLaMA, Mistral, DeepSeek | `03_llm/` |
| Reasoning | CoT, ToT, Self-Consistency, MCTS, Reasoning Tokens | `04_reasoning/` |
| MoE | Switch, Mixtral, DeepSeek MoE | `05_moe/` |
| SLM | Phi4, Gemma3, Quantization, Pruning | `06_slm/` |
| Multimodal | CLIP, SigLIP, Flamingo | `07_multimodal/` |
| VLM | LLaVA, QwenVL, InternVL | `08_vlm/` |
| Image Gen | SD, SDXL, Flux, Consistency | `09_image_generation/` |
| Video Gen | Sora-style, Video Diffusion | `10_video_generation/` |
| Speech | Whisper, StyleTTS | `11_speech/` |
| Audio | AudioLM, MusicGen | `12_audio_generation/` |
| Agents | ReAct, Tool-use, Planning, MCTS, Multi-Agent RL | `13_agentic_ai/` |
| RAG | Dense, Sparse, Self-RAG, CRAG, Graph RAG, Multimodal | `14_rag/` |
| Long Context | RoPE, Ring Attention, Compression, Sliding Window | `15_long_context/` |
| Code | CodeLLaMA, Reward Models, Code Search, AST | `16_code_generation/` |
| Embedding | BERT, E5, GTE | `17_embedding/` |
| RecSys | Two-tower, Sequential | `18_recommendation/` |
| RL | RLHF, DPO, GRPO, PPO | `19_reinforcement_learning/` |
| Robotics | RT-2, Imitation Learning, Sim-to-Real | `20_robotics/` |
| Diffusion | DDPM, Score, Flow Matching | `21_diffusion/` |
| Variants | Mamba, RWKV, Hyena | `22_transformer_variants/` |
| 2026 | Emerging architectures | `23_emerging_2026/` |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Start with math foundations
cd 01_math/
python linear_algebra.py

# Train a small transformer
cd 02_transformers/
python train_mini_gpt.py
```

---

## References

Key papers and resources are linked in each module's README.
See `docs/references.md` for the complete bibliography.

---

## License

MIT License — Use freely for learning and production.

---

*Built as a combination of OpenAI research docs, DeepMind educational material,
Anthropic engineering documentation, Stanford/MIT AI courses, and production AI codebases.*
