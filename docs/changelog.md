# SOTA AI — Changelog

## Version 1.1.0 (2026-07-04)

### 2026 Model Updates

#### DeepSeek R1 (`23_emerging_2026/deepseek_r1.py`)
- Multi-head Latent Attention (MLA) — low-rank KV compression
- Mixture of Experts with shared experts (671B total, 37B active)
- GRPO training — no critic model needed, rule-based rewards
- Multi-token prediction for faster inference

#### Claude 5 (`23_emerging_2026/claude5.py`)
- 5 model tiers: Haiku 4.5, Sonnet 5, Opus 4.8, Fable 5, Mythos 5
- Agent teams — 16 agents collaborated to write a C compiler
- Extended thinking for complex reasoning
- Safety system with domain restrictions (Fable 5)
- 1M token context window (Sonnet 5)

#### Gemini 3 (`23_emerging_2026/gemini3.py`)
- Sparse Mixture of Experts (128 experts, 8 active per token)
- Deep Think — math olympiad gold medal
- Nano Banana — viral image generation (10M+ users)
- 2M context window, 64K output tokens
- Agent-first design (Gemini 3.5)

#### GPT-5 (`23_emerging_2026/gpt5.py`)
- Auto-routing system (fast vs reasoning model)
- Native multimodal training (text + images from scratch)
- Safe completions instead of refusals
- Agentic capabilities (browser, code execution)

#### Hybrid SSM Updates (`23_emerging_2026/hybrid_ssm.py`)
- Mamba-2 — Structured State Space Duality (2-8x faster)
- MoE-Mamba — Mixture of Experts + Mamba (2.2x fewer steps)

---

## Version 1.0.0 (2026-06-13)

### Initial Release

#### Mathematics (`01_math/`)
- Linear algebra (vectors, matrices, tensors)
- Probability (distributions, entropy, KL divergence)
- Optimization (SGD, Adam, AdamW)
- Calculus (derivatives, backpropagation)

#### Transformers (`02_transformers/`)
- Attention mechanisms (multi-head, GQA, Flash)
- Embeddings (RoPE, ALiBi)
- Layers (RMSNorm, SwiGLU, TransformerBlock)
- Complete transformer model
- KV cache
- Training pipeline

#### Language Models (`03_llm/`)
- GPT architecture
- Mixture of Experts
- Reasoning models (CoT, ToT)
- Small Language Models
- Encoder-only (BERT)
- Encoder-decoder (T5)

#### Image Generation (`09_image_generation/`)
- Diffusion models (DDPM)
- Latent diffusion
- U-Net architecture
- VAE
- ControlNet
- LoRA
- Flux
- Consistency models

#### Multimodal (`07_multimodal/`)
- CLIP
- Cross-attention
- Multimodal fusion

#### Vision Language Models (`08_vlm/`)
- VLM architecture

#### Video Generation (`10_video_generation/`)
- Video diffusion
- Temporal attention

#### Speech (`11_speech/`)
- Whisper
- Text-to-speech

#### Audio Generation (`12_audio_generation/`)
- Audio model

#### Agentic AI (`13_agentic_ai/`)
- ReAct agent
- Plan-and-execute
- Tools

#### RAG (`14_rag/`)
- RAG pipeline
- Dense/sparse retrieval
- Vector store

#### Long Context (`15_long_context/`)
- RoPE scaling

#### Code Generation (`16_code_generation/`)
- Code model

#### Embeddings (`17_embedding/`)
- Embedding model

#### Recommendation (`18_recommendation/`)
- Recommender

#### Reinforcement Learning (`19_reinforcement_learning/`)
- RLHF
- DPO
- GRPO
- Reward model

#### Robotics (`20_robotics/`)
- Robot model

#### Diffusion Theory (`21_diffusion/`)
- DDPM
- Score matching
- Flow matching

#### Transformer Variants (`22_transformer_variants/`)
- Mamba (SSM)
- RWKV
- Linear attention

#### Emerging 2026 (`23_emerging_2026/`)
- Hybrid SSM-attention
- Test-time compute

#### Infrastructure
- Training (distributed, mixed precision, gradient accumulation)
- Inference (engine, batching)
- Evaluation (metrics, benchmarks)
- Deployment (serving)
- Common (tokenizer, checkpoint, metrics)

#### Documentation
- README
- Project index
- Learning path
- Interview prep
- Cheat sheets
- Glossary
- Paper reading guide
- Research directions
- Tools and frameworks
- Best practices
- Troubleshooting
- FAQ
- Contributing
- License
- Changelog

---

*First comprehensive release covering all major AI architectures.*
