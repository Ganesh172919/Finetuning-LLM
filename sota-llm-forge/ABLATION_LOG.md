# Ablation Log — SOTA LLM Forge

Track every architecture/optimizer variant tried and its effect.
This turns training runs into research artifacts, not one-off script executions.

## Format

| Date | Run ID | Config | Variant | Metric | Result | Notes |
|------|--------|--------|---------|--------|--------|-------|

## Experiments

| Date | Run ID | Config | Variant | Metric | Result | Notes |
|------|--------|--------|---------|--------|--------|-------|
| 2026-07-07 | — | nano | baseline | — | — | Initial setup, no runs yet |

---

## Key Decisions Log

| Date | Decision | Rationale | Section |
|------|----------|-----------|---------|
| 2026-07-07 | Muon+AdamW hybrid over pure AdamW | ~2× training efficiency on 2D weights (Jordan et al. 2024) | Prompt §7 |
| 2026-07-07 | MLA over GQA for Small-MoE | 10-60× KV cache savings (DeepSeek-V3) | Prompt §6.2 |
| 2026-07-07 | Aux-loss-free over aux-loss MoE | Cleaner LM gradient (DeepSeek-V3) | Prompt §6.3 |
| 2026-07-07 | GRPO over PPO for RLVR | No critic network needed (DeepSeek-R1) | Prompt §12.3 |
| 2026-07-07 | WSD over cosine schedule | Decouples training duration from LR shape | Prompt §7 |
| 2026-07-07 | Byte-level BPE with digit splitting | No UNK tokens, better arithmetic | Prompt §5 |
