# SOTA AI — Troubleshooting

## Common Training Issues

### 1. Loss Not Decreasing
**Possible causes:**
- Learning rate too high or too low
- Bad data
- Bug in model

**Solutions:**
- Try different learning rates
- Check data quality
- Debug with small batch

### 2. Gradient Explosion
**Symptoms:**
- Loss becomes NaN
- Gradients very large

**Solutions:**
- Gradient clipping
- Lower learning rate
- Check for bugs

### 3. Out of Memory
**Solutions:**
- Reduce batch size
- Use gradient accumulation
- Use mixed precision
- Use model parallelism

### 4. Slow Training
**Solutions:**
- Use Flash Attention
- Optimize data loading
- Use multiple GPUs
- Profile bottlenecks

---

## Common Inference Issues

### 1. Slow Generation
**Solutions:**
- Use KV cache
- Batch requests
- Quantize model

### 2. Poor Quality
**Solutions:**
- Adjust temperature
- Use top-p sampling
- Check prompt engineering

### 3. High Latency
**Solutions:**
- Optimize model size
- Use speculative decoding
- Cache common requests

---

*Most issues have well-known solutions. Check the documentation first.*
