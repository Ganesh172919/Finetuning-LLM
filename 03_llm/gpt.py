"""
################################################################################
GPT — GENERATIVE PRE-TRAINED TRANSFORMER
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is GPT?
    GPT (Generative Pre-trained Transformer) is a decoder-only transformer
    trained to predict the next token. It's the architecture behind:
    - GPT-1, GPT-2, GPT-3, GPT-4 (OpenAI)
    - Claude (Anthropic)
    - LLaMA (Meta)
    - Mistral (Mistral AI)
    - And most modern LLMs

Why is it called "Generative Pre-trained"?
    - Generative: It generates text autoregressively
    - Pre-trained: It's trained on large corpora before fine-tuning
    - Transformer: The underlying architecture

How does it work?
    1. Pre-training: Learn language from massive text data
    2. Fine-tuning: Adapt to specific tasks (optional)
    3. Inference: Generate text token by token

The key insight: by learning to predict the next token,
the model learns grammar, facts, reasoning, and more.

########################################

GPT ARCHITECTURE:
    ┌─────────────────────────────────────┐
    │ Input: "The cat sat on the"          │
    │                                       │
    │ Token Embedding + RoPE               │
    │        ↓                              │
    │ Transformer Block × N                 │
    │   ├── RMSNorm → Causal Attention     │
    │   └── RMSNorm → SwiGLU FFN          │
    │        ↓                              │
    │ Final RMSNorm                         │
    │        ↓                              │
    │ LM Head → logits                      │
    │        ↓                              │
    │ Softmax → "mat" (predicted)           │
    └─────────────────────────────────────┘

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math

import sys
sys.path.append('..')
from ..02_transformers.model import TransformerLM, KVCache
from ..02_transformers.attention import create_causal_mask
from ..01_math.probability import temperature_scaled_softmax, top_k_sampling, top_p_sampling


################################################################################
# SECTION 1: GPT CONFIGURATION
################################################################################

@dataclass
class GPTConfig:
    """
    GPT Model Configuration.

    This defines all hyperparameters for a GPT model.
    Common configurations are provided as class methods.
    """
    vocab_size: int = 32000
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None  # None = MHA, < n_heads = GQA
    d_ff: Optional[int] = None  # None = auto
    max_seq_len: int = 8192
    dropout: float = 0.0
    tie_weights: bool = True
    norm_eps: float = 1e-6
    rope_base: float = 10000.0

    @classmethod
    def gpt2_small(cls):
        """GPT-2 Small: 117M parameters"""
        return cls(
            vocab_size=50257, d_model=768, n_layers=12,
            n_heads=12, max_seq_len=1024
        )

    @classmethod
    def gpt2_large(cls):
        """GPT-2 Large: 774M parameters"""
        return cls(
            vocab_size=50257, d_model=1280, n_layers=36,
            n_heads=20, max_seq_len=1024
        )

    @classmethod
    def llama_7b(cls):
        """LLaMA-7B: 6.7B parameters"""
        return cls(
            vocab_size=32000, d_model=4096, n_layers=32,
            n_heads=32, n_kv_heads=32, max_seq_len=4096,
            rope_base=10000.0
        )

    @classmethod
    def llama_13b(cls):
        """LLaMA-13B: 13B parameters"""
        return cls(
            vocab_size=32000, d_model=5120, n_layers=40,
            n_heads=40, n_kv_heads=40, max_seq_len=4096
        )

    @classmethod
    def llama_70b(cls):
        """LLaMA-70B: 70B parameters with GQA"""
        return cls(
            vocab_size=32000, d_model=8192, n_layers=80,
            n_heads=64, n_kv_heads=8,  # GQA: 8 KV heads
            max_seq_len=4096
        )

    @classmethod
    def mistral_7b(cls):
        """Mistral-7B: 7.3B parameters with GQA and sliding window"""
        return cls(
            vocab_size=32000, d_model=4096, n_layers=32,
            n_heads=32, n_kv_heads=8,  # GQA
            max_seq_len=32768  # Sliding window attention
        )

    @classmethod
    def deepseek_v2(cls):
        """DeepSeek-V2: 236B parameters with MLA"""
        return cls(
            vocab_size=100000, d_model=5120, n_layers=60,
            n_heads=128, n_kv_heads=16,  # GQA
            max_seq_len=16384
        )


################################################################################
# SECTION 2: GPT MODEL
################################################################################

class GPT(TransformerLM):
    """
    GPT: Generative Pre-trained Transformer
    =========================================

    Extends the base TransformerLM with:
    1. Configuration-based initialization
    2. Advanced generation strategies
    3. Training utilities
    4. Model loading/saving

    This is the model behind ChatGPT, Claude, LLaMA, etc.

    Key Innovations in Modern GPTs:
    1. RoPE instead of learned position embeddings
    2. RMSNorm instead of LayerNorm
    3. SwiGLU instead of ReLU FFN
    4. GQA instead of MHA (for larger models)
    5. Flash Attention for efficiency
    6. Longer context (4K → 128K+)

    Interview Questions:
        1. "What makes modern LLMs different from GPT-2?"
           Architecture: RoPE, RMSNorm, SwiGLU, GQA
           Scale: Billions of parameters
           Data: Trillions of tokens
           Alignment: RLHF/DPO for safety

        2. "How does GPT learn?"
           Through next-token prediction on massive text.
           The model learns grammar, facts, reasoning patterns,
           and even code by minimizing prediction error.

        3. "Why is scaling so effective?"
           Scaling laws show predictable improvement with scale.
           More parameters capture more patterns.
           More data provides more patterns to learn.
    """

    def __init__(self, config: GPTConfig):
        """
        Initialize GPT from configuration.

        Args:
            config: GPTConfig with all hyperparameters
        """
        self.config = config
        super().__init__(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            tie_weights=config.tie_weights
        )

    @classmethod
    def from_config(cls, config_name: str) -> 'GPT':
        """Create GPT from named configuration."""
        configs = {
            'gpt2-small': GPTConfig.gpt2_small(),
            'gpt2-large': GPTConfig.gpt2_large(),
            'llama-7b': GPTConfig.llama_7b(),
            'llama-13b': GPTConfig.llama_13b(),
            'llama-70b': GPTConfig.llama_70b(),
            'mistral-7b': GPTConfig.mistral_7b(),
            'deepseek-v2': GPTConfig.deepseek_v2(),
        }
        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}. Choose from {list(configs.keys())}")
        return cls(configs[config_name])


################################################################################
# SECTION 3: ADVANCED GENERATION
################################################################################

class TextGenerator:
    """
    Advanced Text Generation for GPT Models
    =========================================

    Supports multiple generation strategies:
    1. Greedy: Always pick most likely token
    2. Temperature: Control randomness
    3. Top-K: Sample from K most likely
    4. Top-P (Nucleus): Sample from smallest set exceeding P
    5. Beam Search: Keep top-B candidates
    6. Speculative Decoding: Use small model to draft

    Interview Questions:
        1. "What's the difference between greedy and sampling?"
           Greedy always picks the most likely token (deterministic).
           Sampling picks according to probability (stochastic).
           Sampling produces more diverse and natural text.

        2. "When should I use beam search?"
           Beam search is good for tasks with a single correct answer
           (translation, summarization). For creative generation,
           sampling is better.

        3. "What's speculative decoding?"
           Use a small, fast model to draft tokens.
           Then verify with the large model in parallel.
           If accepted, you get multiple tokens for one large model call.
    """

    def __init__(self, model: GPT):
        self.model = model
        self.kv_cache = None

    def greedy(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 100,
        eos_token_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Greedy Generation: always pick most likely token.

        Pros: Deterministic, fast
        Cons: Can be repetitive, misses diverse completions
        """
        generated = prompt_ids.copy()

        for _ in range(max_new_tokens):
            logits, _ = self.model.forward(generated)
            next_logits = logits[0, -1, :]
            next_token = np.argmax(next_logits)

            generated = np.concatenate([
                generated,
                np.array([[next_token]])
            ], axis=1)

            if eos_token_id is not None and next_token == eos_token_id:
                break

        return generated

    def sample(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Sampling Generation with temperature, top-k, top-p.

        Pros: Diverse, natural text
        Cons: Can be incoherent with high temperature
        """
        generated = prompt_ids.copy()

        for _ in range(max_new_tokens):
            logits, _ = self.model.forward(generated)
            next_logits = logits[0, -1, :]

            # Temperature scaling
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-K filtering
            if top_k > 0:
                top_k_indices = np.argsort(next_logits)[-top_k:]
                filtered = np.full_like(next_logits, -np.inf)
                filtered[top_k_indices] = next_logits[top_k_indices]
                next_logits = filtered

            # Top-P filtering
            if top_p < 1.0:
                sorted_indices = np.argsort(next_logits)[::-1]
                sorted_logits = next_logits[sorted_indices]
                sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
                sorted_probs = sorted_probs / np.sum(sorted_probs)

                cumulative = np.cumsum(sorted_probs)
                nucleus_size = np.searchsorted(cumulative, top_p) + 1
                nucleus_indices = sorted_indices[:nucleus_size]

                filtered = np.full_like(next_logits, -np.inf)
                filtered[nucleus_indices] = next_logits[nucleus_indices]
                next_logits = filtered

            # Softmax and sample
            probs = np.exp(next_logits - np.max(next_logits))
            probs = probs / np.sum(probs)
            next_token = np.random.choice(len(probs), p=probs)

            generated = np.concatenate([
                generated,
                np.array([[next_token]])
            ], axis=1)

            if eos_token_id is not None and next_token == eos_token_id:
                break

        return generated

    def beam_search(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 100,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        eos_token_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Beam Search: keep top-B candidates at each step.

        Algorithm:
        1. Start with B copies of prompt
        2. For each step:
           a. Expand each beam with all possible next tokens
           b. Keep top-B based on cumulative log probability
        3. Return best beam

        Pros: Higher quality output
        Cons: Slower, less diverse

        Used for: Translation, summarization
        """
        batch_size = 1
        beams = [(prompt_ids.copy(), 0.0)]  # (sequence, score)

        for _ in range(max_new_tokens):
            candidates = []

            for seq, score in beams:
                logits, _ = self.model.forward(seq)
                next_logits = logits[0, -1, :]
                log_probs = next_logits - np.log(np.sum(np.exp(next_logits)))

                # Get top-k tokens
                top_indices = np.argsort(log_probs)[-num_beams:]

                for idx in top_indices:
                    new_seq = np.concatenate([
                        seq,
                        np.array([[idx]])
                    ], axis=1)
                    new_score = score + log_probs[idx]
                    candidates.append((new_seq, new_score))

            # Keep top-B candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:num_beams]

            # Check for EOS
            if eos_token_id is not None:
                beams = [
                    (seq, score) for seq, score in beams
                    if seq[0, -1] != eos_token_id
                ]
                if not beams:
                    break

        # Return best beam
        return beams[0][0]


################################################################################
# SECTION 4: TESTING & EXAMPLES
################################################################################

def demonstrate_gpt():
    """Demonstrate GPT model."""
    print("=" * 70)
    print("GPT MODEL DEMONSTRATION")
    print("=" * 70)

    # Create small model for demonstration
    print("\n--- Creating GPT Model ---")
    config = GPTConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=256
    )
    model = GPT(config)

    # Forward pass
    print("\n--- Forward Pass ---")
    token_ids = np.random.randint(0, 1000, (1, 8))
    targets = np.random.randint(0, 1000, (1, 8))
    logits, loss = model.forward(token_ids, targets)
    print(f"Input: {token_ids[0].tolist()}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss:.4f}")

    # Generation
    print("\n--- Text Generation ---")
    generator = TextGenerator(model)
    prompt = np.array([[1, 5, 42]])

    # Greedy
    greedy_output = generator.greedy(prompt, max_new_tokens=5)
    print(f"Greedy: {greedy_output[0].tolist()}")

    # Sampling
    sample_output = generator.sample(
        prompt, max_new_tokens=5, temperature=0.8
    )
    print(f"Sample: {sample_output[0].tolist()}")

    # Beam search
    beam_output = generator.beam_search(
        prompt, max_new_tokens=5, num_beams=3
    )
    print(f"Beam: {beam_output[0].tolist()}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_gpt()


################################################################################
# REFERENCES
################################################################################

# [1] Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training.
# [2] Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners.
# [3] Brown, T., et al. (2020). Language Models are Few-Shot Learners.
# [4] Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models.
# [5] Jiang, A., et al. (2023). Mistral 7B.

################################################################################
