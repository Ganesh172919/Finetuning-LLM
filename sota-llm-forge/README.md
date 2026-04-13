# SOTA LLM Forge

## A Decoder-Only Transformer LLM Training + Alignment + Inference Stack

Built from first principles, implementing the architectural and systems patterns used by frontier open-weight labs as of mid-2026 (DeepSeek, Moonshot/Kimi, Zhipu/GLM, Qwen, Meta).

---

## Mission

Build a training stack that is:
- **Correct** — validated against reference behavior (gradient checks, shape tests, loss-curve sanity)
- **Explained** — every non-obvious design choice gets a comment citing why, not just what
- **Modular** — attention, MoE routing, optimizer, and RL post-training each swappable independently
- **Honest about scale** — every major file documents what would change at 100B+ params

---

## Repository Structure

```
sota-llm-forge/
├── configs/                  # YAML configs per scale tier
│   ├── nano.yaml             # ~30-60M params, dense, 1 GPU
│   ├── small_moe.yaml        # ~1-3B total, MLA+MoE, 8 GPUs
│   └── frontier_pattern.yaml # 100B-1.6T (documentation only)
├── tokenizer/                # Byte-level BPE tokenizer
│   ├── train_tokenizer.py    # Training + encode/decode
│   └── test_tokenizer.py     # Unit tests
├── model/                    # Model architecture
│   ├── attention.py          # GQA, MLA, hybrid sparse-attention
│   ├── moe.py                # DeepSeekMoE-style routed + shared experts
│   ├── mtp.py                # Multi-token prediction modules
│   ├── residual.py           # Standard residual + hyper-connections
│   ├── transformer.py        # Full block assembly
│   └── init.py               # muP-style initialization
├── optim/                    # Optimizer stack
│   ├── muon.py               # Newton-Schulz orthogonalized optimizer
│   └── hybrid_optimizer.py   # Muon (2D) + AdamW (rest) + WSD schedule
├── data/                     # Data pipeline
│   ├── curate.py             # Dedup, quality filters, decontamination
│   ├── synthetic.py          # Multi-teacher synthetic generation
│   └── pack.py               # Sequence packing with doc-boundary masking
├── train/                    # Training loops
│   ├── pretrain.py           # Main pretraining loop, 4D parallel
│   ├── sft.py                # Supervised fine-tuning
│   ├── dpo.py                # Direct Preference Optimization
│   └── rlvr_grpo.py          # GRPO/DAPO/GSPO reasoning RL
├── eval/                     # Evaluation harness
│   ├── harness.py            # Unified eval runner
│   └── suites/               # One file per benchmark family
├── serve/                    # Inference & serving
│   ├── quantize.py           # FP8 / NVFP4 post-training quantization
│   └── speculative.py        # MTP-head-driven self-speculative decoding
├── tests/                    # Integration tests
└── README.md                 # This file
```

---

## Scale Tiers

| Tier | Total Params | Active Params | Layers | Attention | MoE | Hardware | Purpose |
|------|-------------|---------------|--------|-----------|-----|----------|---------|
| **Nano** | ~30-60M | all active | 6-12 | GQA only | none | 1 consumer GPU | Fast iteration, debugging, muP search |
| **Small-MoE** | ~1-3B | ~300-600M | 16-24 | MLA | fine-grained routed + shared | 8×GPU node, days | Reference SOTA implementation |
| **Frontier** | 100B-1.6T | 10-50B | dozens | hybrid MLA + sparse | hundreds of experts | thousands of GPUs | Documentation only |

---

## Key Design Decisions

### Optimizer: Muon + AdamW Hybrid (Section 7)
**Decision**: Muon for all 2D hidden weight matrices, AdamW for everything else.
**Why**: Muon normalizes the *spectrum* of the weight update (matrix-aware), while AdamW adapts per-element. Empirically ~2× fewer steps to a given loss on hidden weights. (Jordan et al. 2024, "Muon")
**Tradeoff**: Muon doesn't apply to 1D params (biases, norms) or embeddings. Hybrid approach captures both benefits.

### Attention: MLA over GQA (Section 6.2)
**Decision**: Implement both GQA (baseline) and MLA (production). MLA is the default for Small-MoE tier.
**Why**: MLA compresses KV cache into a low-dimensional latent, achieving 10-60× memory savings vs full MHA with minimal quality loss. (DeepSeek-V3, arXiv:2412.19437)
**Tradeoff**: MLA adds complexity (compression/decompression). GQA is simpler and still better than MHA. Use GQA for nano tier, MLA for production.

### MoE: Aux-Loss-Free Balancing (Section 6.3)
**Decision**: Per-expert bias term for load balancing, NOT auxiliary loss.
**Why**: Auxiliary loss distorts the primary language-modeling gradient. DeepSeek-V3 showed you can balance load without touching the LM gradient by maintaining a per-expert bias that's updated with a fixed step size. (DeepSeek-V3 technical report)
**Tradeoff**: Slightly more complex router implementation. Worth it for cleaner training signal.

### RL: GRPO over PPO (Section 12.3)
**Decision**: GRPO as the base RL algorithm with DAPO/GSPO/Dr.GRPO refinements as flags.
**Why**: GRPO eliminates the need for a separate value/critic network by using group-relative advantages. Cheaper and simpler than PPO while maintaining quality. (DeepSeek-R1)
**Tradeoff**: Group sampling adds overhead per prompt. Offset by not needing a critic model.

### Schedule: WSD over Cosine (Section 7)
**Decision**: Warmup-Stable-Decay learning rate schedule.
**Why**: Decouples "how long to train" from "when to commit to final LR shape." Stable plateau enables cheap checkpoint-forking for ablations. (DeepSeek, Kimi, GLM teams)
**Tradeoff**: Requires specifying stable/decay phase boundaries. More config knobs than cosine.

### Tokenizer: Byte-Level BPE (Section 5)
**Decision**: Byte-level BPE with digit splitting, 128k default vocab.
**Why**: Byte-level fallback ensures every input is representable (no UNK tokens). Digit splitting materially improves arithmetic. (GPT-2/3-era lesson)
**Tradeoff**: Larger vocab = shorter sequences but bigger embedding tables. 128k is the sweet spot for nano/small tiers.

---

## Build Order (Phased Roadmap)

1. **Tokenizer** — Byte-level BPE with digit splitting
2. **Nano-tier backbone** — RMSNorm + SwiGLU + RoPE + GQA
3. **MLA** — Verify KV-cache win at nano tier
4. **MoE + aux-loss-free balancing** — At small enough scale to inspect
5. **MTP** — Training signal density + speculative decoding
6. **Muon + hybrid optimizer** — A/B against AdamW
7. **Scale to Small-MoE** — Bring up parallelism as needed
8. **Data pipeline** — Dedup, quality, decontamination
9. **Full pretraining** — With context extension
10. **Post-training** — SFT → DPO → RLVR/GRPO
11. **Eval harness** — MMLU-Pro, GPQA, code benchmarks
12. **Inference** — FP8 quantization + speculative decoding
13. **Safety pass** — Harmlessness DPO + red-team prompts
14. **Frontier config** — Documentation-only annotated config

---

## What Would Change at Frontier Scale

Every major file includes a "at frontier scale" comment block. Key themes:

- **Parallelism**: Nano needs none. Small-MoE needs expert parallel. Frontier needs full 4D (DP+TP+EP+PP) across thousands of GPUs.
- **Precision**: BF16 for nano/small. FP8 mixed precision for frontier (DeepSeek-V3 validated this at 671B params).
- **MoE**: Nano has no MoE. Small-MoE has ~64-128 experts. Frontier has hundreds of experts with hybrid sparse attention for long context.
- **Context**: 4k-8k base for pretraining. YaRN/NTK-aware scaling to 32k-128k for mid-training. Frontier targets 1M+ tokens.
- **Data**: Frontier requires petabyte-scale curation with rigorous decontamination. Our pipeline implements the same techniques at smaller scale.
- **Optimizer**: Muon scales to trillion-parameter models (Kimi K2/2.5, GLM-4.5/4.7). Same algorithm, more engineering.

---

## Running Tests

```bash
cd sota-llm-forge
python -m pytest tests/ -v
python -m pytest tokenizer/test_tokenizer.py -v
python -m pytest model/ -v
```

---

## References

- DeepSeek-V3 Technical Report (arXiv:2412.19437)
- DeepSeek-V4 Technical Documentation (April 2026)
- "Muon is Scalable for LLM Training" (Jordan et al., 2024)
- GLM-5 / GLM-5.2 model cards (Zhipu/Z.ai)
- Kimi K2 / K2.6 technical materials (Moonshot AI)
- DAPO, GSPO, Dr. GRPO papers
- YaRN / NTK-aware RoPE scaling papers
- NVIDIA NVFP4 / Model-Optimizer documentation

---

*Built as an educational implementation of 2026 frontier LLM training patterns. Every file teaches. Every decision is documented.*
