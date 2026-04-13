# SOTA AI — Examples

## Quick Start

### 1. Math Foundations
```python
from sota.math.linear_algebra import Vector, Matrix

v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])
print(v1.dot(v2))  # 32
```

### 2. Transformer
```python
from sota.transformers.model import TransformerLM

model = TransformerLM(vocab_size=1000, d_model=128, n_layers=4)
logits, loss = model.forward(token_ids, targets)
```

### 3. Image Generation
```python
from sota.image.diffusion import DiffusionModel

model = DiffusionModel(input_dim=64)
sample = DiffusionModel.sample(model, shape=(1, 64))
```

### 4. RAG
```python
from sota.rag.pipeline import RAGPipeline

rag = RAGPipeline(embedder, vector_store)
result = rag.query("What is AI?")
```

---

## More Examples

See individual module files for detailed examples.

---

*Start with these examples and explore further.*
