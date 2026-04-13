# SOTA AI — Deployment Guide

## Local Deployment

### CPU
```python
from sota import model
model = model.load('small')
output = model.generate('Hello')
```

### GPU
```python
import torch
model = model.load('small', device='cuda')
```

---

## Cloud Deployment

### AWS
- Use SageMaker for managed deployment
- Use EC2 with GPUs for custom setups

### GCP
- Use Vertex AI for managed deployment
- Use Compute Engine with GPUs

### Azure
- Use Azure ML for managed deployment

---

## Production Deployment

### vLLM
```bash
vllm serve model_name --tensor-parallel-size 4
```

### Docker
```bash
docker build -t sota-ai .
docker run -p 8000:8000 sota-ai
```

---

## Scaling

### Horizontal Scaling
- Use load balancer
- Deploy multiple instances

### Vertical Scaling
- Use larger GPUs
- Use more GPUs

---

*Choose deployment strategy based on your needs.*
