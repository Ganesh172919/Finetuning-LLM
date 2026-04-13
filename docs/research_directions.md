# SOTA AI — Research Directions (2025-2026)

## Latest Model Releases (2025-2026)

### GPT-5 (OpenAI — August 2025)
- **Architecture**: System of fast model + reasoning model + auto-router
- **Key Feature**: Automatic model selection based on task complexity
- **Training**: Trained from scratch on text + images (native multimodal)
- **Safety**: "Safe completions" instead of refusals
- **Agentic**: Can autonomously browse web, execute code

### Claude 5 (Anthropic — June 2026)
- **Model Tiers**: Haiku 4.5, Sonnet 5, Opus 4.8, Fable 5, Mythos 5
- **Key Feature**: Sonnet 5 has 1M token context window
- **Innovation**: Agent teams (16 agents wrote a C compiler in Rust)
- **Safety**: Multiple levels, Fable 5 has domain restrictions
- **Controversy**: Claude 4.7 had excessive refusals (April 2026)

### Gemini 3 (Google — November 2025+)
- **Architecture**: Sparse Mixture of Experts (128 experts, 8 active)
- **Key Feature**: 2M context window, 64K output tokens
- **Deep Think**: Math olympiad gold medal (December 2025)
- **Nano Banana**: Viral image generation (10M+ users)
- **Agent Focus**: Gemini 3.5 designed for agents, not chatbots

### DeepSeek R1 (DeepSeek — January 2025)
- **Architecture**: MLA + MoE (671B total, 37B active)
- **Training**: GRPO with rule-based rewards (no human feedback)
- **Cost**: ~$6 million vs $100 million for GPT-4
- **Impact**: Surpassed ChatGPT as #1 on US iOS App Store
- **Open Source**: MIT License

---

## Hot Research Areas

### 1. Reasoning Models
- Test-time compute scaling
- Chain of thought training
- Self-improvement
- DeepSeek R1 approach (GRPO)
- Gemini Deep Think (math olympiad)

### 2. Hybrid Architectures
- SSM + Attention combinations
- Mamba-2 (Structured State Space Duality)
- MoE-Mamba (2.2x fewer training steps)
- Linear attention improvements
- Long context efficiency

### 3. Multimodal Unified Models
- Any-to-any generation
- Unified architectures
- Cross-modal reasoning
- Native multimodal training (GPT-5)

### 4. Efficiency
- Quantization (int4, int8, FP8)
- Distillation
- Pruning
- Efficient attention
- Sparse MoE (Gemini 3)

### 5. Alignment
- DPO variants
- Constitutional AI
- Scalable oversight
- Safe completions (GPT-5)
- Agent teams (Claude 4.6)

### 6. Agents
- Tool use training
- Planning capabilities
- Multi-agent systems
- Agent teams (Claude 4.6)
- Agent-first design (Gemini 3.5)

### 7. World Models
- Physics understanding
- Causal reasoning
- Video prediction

### 8. Image Generation
- Nano Banana (Google, viral 3D figurines)
- Photorealistic generation
- Text rendering improvements

---

## Key Trends (2026)

1. **Auto-routing**: GPT-5's system automatically selects fast vs reasoning model
2. **Agent Teams**: Multiple AI agents collaborating (Claude 4.6)
3. **Sparse MoE**: Gemini 3's 128 experts with only 8 active per token
4. **1M+ Context**: Sonnet 5 and Gemini 3 support million-token contexts
5. **Open Source**: DeepSeek R1 matches GPT-4 at fraction of cost
6. **Safety Levels**: Multiple tiers (Fable 5, Mythos 5) for different risk levels
7. **Native Multimodal**: GPT-5 trained from scratch on text + images
8. **Agent-First**: Gemini 3.5 designed for agents, not chatbots

---

*Stay current with arXiv, conferences (NeurIPS, ICML, ICLR), and industry blogs.*
