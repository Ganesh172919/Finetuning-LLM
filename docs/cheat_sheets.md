# SOTA AI — Cheat Sheets

## Key Formulas

### Attention
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

### Cross-Entropy Loss
```
L = -Σ y_true * log(y_pred)
```

### KL Divergence
```
KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
```

### RoPE
```
RoPE(x, pos) = R(pos) @ x
where R(pos) is a rotation matrix
```

### LoRA
```
W' = W + (α/r) * A @ B
where A ∈ R^{d×r}, B ∈ R^{r×d}
```

### Diffusion
```
x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
```

---

## Architecture Cheat Sheet

### Transformer Block
```
x → RMSNorm → Attention → + → h
h → RMSNorm → FFN → + → output
```

### GPT (Decoder-Only)
```
Tokens → Embedding → N × TransformerBlock → Norm → LMHead → Logits
```

### Stable Diffusion
```
Noise → U-Net (conditioned on text) → Latent → VAE Decoder → Image
```

---

## Model Sizes

| Model | Params | Layers | d_model | Heads |
|-------|--------|--------|---------|-------|
| GPT-2 | 117M | 12 | 768 | 12 |
| LLaMA-7B | 6.7B | 32 | 4096 | 32 |
| LLaMA-70B | 70B | 80 | 8192 | 64 |
| Mistral-7B | 7.3B | 32 | 4096 | 32 |

---

## Training Hyperparameters

| Parameter | Typical Value |
|-----------|---------------|
| Learning rate | 1e-4 to 3e-4 |
| Batch size | 1M-4M tokens |
| Warmup steps | 2000 |
| Weight decay | 0.1 |
| Max gradient norm | 1.0 |

---

*Quick reference for common patterns and values.*
