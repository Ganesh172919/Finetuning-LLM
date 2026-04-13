# SOTA AI — API Reference

## Core Modules

### `sota.math`
- `Vector` — Mathematical vector
- `Matrix` — Mathematical matrix
- `Tensor` — Multi-dimensional array

### `sota.transformers`
- `MultiHeadAttention` — Attention mechanism
- `TransformerBlock` — Transformer layer
- `TransformerLM` — Language model

### `sota.llm`
- `GPT` — GPT model
- `MixtureOfExperts` — MoE layer
- `ReasoningModel` — Reasoning model

### `sota.image`
- `DiffusionModel` — Diffusion model
- `LatentDiffusionModel` — Latent diffusion
- `UNet` — U-Net architecture

### `sota.multimodal`
- `CLIP` — CLIP model
- `CrossAttention` — Cross-attention

---

## Usage Examples

```python
from sota.transformers import TransformerLM

model = TransformerLM(vocab_size=32000, d_model=4096, n_layers=32)
output = model.forward(token_ids)
```

---

*See individual module files for detailed API documentation.*
