"""
################################################################################
TRANSFORMER WORLD MODEL — AUTOREGRESSIVE WORLD SIMULATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Transformer World Model?
    A world model that uses transformer architecture to autoregressively
    predict future states. Each state is tokenized, and the model
    predicts the next state token by token — just like GPT predicts
    the next word.

Why does it matter?
    Transformers excel at sequence modeling:
    - Long-range dependencies (attention mechanism)
    - Parallel training (unlike RNNs)
    - Scales well with data and compute
    - Proven architecture (GPT, BERT, etc.)

    Applying transformers to world models enables:
    - Interactive environment simulation (GameNGen)
    - Real-time game generation
    - Physics-aware predictions
    - Long-horizon planning

How does it work?
    1. Encode each state into tokens (patch-based or latent)
    2. Interleave state tokens with action tokens
    3. Train with next-token prediction (standard transformer loss)
    4. At inference: predict next state token by token

Architecture (GameNGen style):
    ┌─────────────────────────────────────────────────────────────────┐
    │           Transformer World Model (Autoregressive)              │
    │                                                                  │
    │  State_0 tokens → Action_0 → State_1 tokens → Action_1 → ...   │
    │                                                                  │
    │  Each "State_i" is a sequence of visual tokens                  │
    │  The model predicts the next state given history + action       │
    │                                                                  │
    │  Training: Standard next-token prediction                       │
    │  Inference: Autoregressive generation                           │
    └─────────────────────────────────────────────────────────────────┘

Key Papers:
    - GameNGen (Google, 2024): Real-time game simulation with diffusion
    - Genie 2 (DeepMind, 2024): Interactive world generation
    - IRIS (Vincent-Lancrin et al., 2023): Transformer world model for Atari

Interview Questions:
    1. "How does a transformer world model differ from JEPA?"
       JEPA predicts in latent space (abstract). Transformer world models
       predict token sequences (explicit). JEPA is more efficient;
       transformers are more flexible and can generate detailed outputs.

    2. "Can transformers model physics accurately?"
       Yes, with enough data. The attention mechanism can learn
       physical relationships (gravity, collisions, etc.) from
       observations. GameNGen generates playable game frames in real-time.

    3. "What are the limitations of transformer world models?"
       Computation: O(n²) in sequence length. For high-res video,
       this becomes expensive. Solutions: patching, latent space,
       or hybrid approaches (transformer + diffusion).

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: CONFIGURATION
################################################################################

@dataclass
class TransformerWorldConfig:
    """
    Configuration for Transformer World Model.
    """
    # State representation
    state_vocab_size: int = 512  # Discrete state tokens
    state_seq_len: int = 16     # Tokens per state
    action_vocab_size: int = 64  # Discrete action tokens

    # Transformer architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024

    # Context
    context_len: int = 10  # Number of past states to condition on
    dropout: float = 0.1


################################################################################
# SECTION 2: TOKEN EMBEDDING
################################################################################

class TokenEmbedding:
    """
    Token Embedding for World Model
    =================================

    Converts discrete tokens (state or action) into continuous embeddings.

    For states:
        Each state is represented as a sequence of discrete tokens
        (like image patches quantized to vocabulary).

    For actions:
        Actions are single tokens from a discrete vocabulary.

    The embedding adds:
        - Token embedding: What the token represents
        - Position embedding: Where in the sequence
        - Type embedding: State vs action token
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512):
        self.d_model = d_model
        self.token_embed = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_embed = np.random.randn(max_len, d_model) * 0.02

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """
        Embed tokens.

        Args:
            tokens: Token indices (batch, seq_len)

        Returns:
            Embeddings (batch, seq_len, d_model)
        """
        batch_size, seq_len = tokens.shape

        # Look up token embeddings
        token_emb = self.token_embed[tokens]  # (batch, seq_len, d_model)

        # Add positional encoding
        pos_emb = self.pos_embed[:seq_len]
        token_emb = token_emb + pos_emb

        # Scale by sqrt(d_model) as in original transformer
        return token_emb * math.sqrt(self.d_model)


################################################################################
# SECTION 3: CAUSAL TRANSFORMER
################################################################################

class CausalTransformerBlock:
    """
    Causal Transformer Block
    ==========================

    Standard transformer block with CAUSAL attention mask.
    Each token can only attend to previous tokens (not future).

    This is essential for autoregressive generation:
    - At training time: parallel processing with causal mask
    - At inference time: generate one token at a time

    Formula:
        x' = x + CausalAttention(LayerNorm(x))
        output = x' + FFN(LayerNorm(x'))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Attention weights
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

        # FFN weights
        self.W_ff1 = np.random.randn(d_model, d_ff) * 0.02
        self.W_ff2 = np.random.randn(d_ff, d_model) * 0.02

        # Layer norm
        self.ln1_g = np.ones(d_model)
        self.ln1_b = np.zeros(d_model)
        self.ln2_g = np.ones(d_model)
        self.ln2_b = np.zeros(d_model)

    def layer_norm(self, x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / np.sqrt(var + 1e-5) + b

    def causal_attention(self, x: np.ndarray) -> np.ndarray:
        """Multi-head attention with causal mask."""
        batch, seq_len, d = x.shape

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head
        Q = Q.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        # Attention scores
        scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(self.d_head)

        # Causal mask: prevent attending to future tokens
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + mask

        weights = self._softmax(scores)
        out = weights @ V

        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d)
        return out @ self.W_o

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        h = self.layer_norm(x, self.ln1_g, self.ln1_b)
        x = x + self.causal_attention(h)

        h = self.layer_norm(x, self.ln2_g, self.ln2_b)
        h = np.maximum(0, h @ self.W_ff1)  # ReLU
        h = h @ self.W_ff2
        x = x + h

        return x


################################################################################
# SECTION 4: AUTOREGRESSIVE WORLD MODEL
################################################################################

class AutoregressiveWorldModel:
    """
    Autoregressive Transformer World Model
    ========================================

    Predicts future states autoregressively, token by token.

    Input sequence format:
        [S_0, A_0, S_1, A_1, S_2, A_2, ...]

    Where:
        S_i = state tokens (sequence of discrete tokens)
        A_i = action token

    Training:
        Standard next-token prediction on the interleaved sequence.
        The model learns to predict the next state/action token
        given all previous tokens.

    Inference:
        Given initial state and actions, generate future states
        token by token.

    Interview Question:
        "How does the model know when a state ends and action begins?"
        Special separator tokens or type embeddings distinguish
        state tokens from action tokens. The model learns this
        distinction during training.
    """

    def __init__(self, config: TransformerWorldConfig):
        self.config = config

        # Embeddings
        self.state_embed = TokenEmbedding(
            config.state_vocab_size, config.d_model
        )
        self.action_embed = TokenEmbedding(
            config.action_vocab_size, config.d_model
        )

        # Special tokens
        self.sep_embed = np.random.randn(1, 1, config.d_model) * 0.02

        # Transformer blocks
        self.blocks = [
            CausalTransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.n_layers)
        ]

        # Output heads
        self.state_head = np.random.randn(config.d_model, config.state_vocab_size) * 0.02
        self.action_head = np.random.randn(config.d_model, config.action_vocab_size) * 0.02

    def build_sequence(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Build interleaved sequence of states and actions.

        Args:
            states: List of state token arrays
            actions: List of action tokens

        Returns:
            Interleaved sequence embeddings
        """
        parts = []
        for i in range(len(states)):
            # State tokens
            s_emb = self.state_embed.forward(states[i])
            parts.append(s_emb)

            if i < len(actions):
                # Action token
                a_emb = self.action_embed.forward(actions[i:i+1])
                parts.append(a_emb)

        return np.concatenate(parts, axis=1)

    def forward(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the model.

        Args:
            sequence: Input sequence embeddings (batch, seq_len, d_model)

        Returns:
            state_logits: Logits for next state token
            action_logits: Logits for next action token
        """
        h = sequence

        # Transformer blocks
        for block in self.blocks:
            h = block.forward(h)

        # Output projections
        state_logits = h @ self.state_head
        action_logits = h @ self.action_head

        return state_logits, action_logits

    def generate_next_state(
        self,
        history_states: List[np.ndarray],
        history_actions: List[np.ndarray],
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Generate next state tokens autoregressively.

        Given history of states and actions, generate the next state
        token by token.

        Args:
            history_states: Past state token sequences
            history_actions: Past action tokens
            temperature: Sampling temperature

        Returns:
            next_state: Generated state tokens
        """
        seq_len = self.config.state_seq_len
        generated = []

        # Build prefix from history
        prefix = self.build_sequence(history_states, history_actions)

        for i in range(seq_len):
            # Current input
            if generated:
                current_state = np.array([generated])
                current_emb = self.state_embed.forward(current_state)
                full_seq = np.concatenate([prefix, current_emb], axis=1)
            else:
                full_seq = prefix

            # Forward pass
            state_logits, _ = self.forward(full_seq)

            # Get logits for next position
            next_logits = state_logits[0, -1] / temperature

            # Sample from distribution
            probs = self._softmax(next_logits)
            token = np.random.choice(len(probs), p=probs)
            generated.append(token)

        return np.array(generated)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def compute_loss(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray]
    ) -> float:
        """
        Compute training loss (next-token prediction).

        Args:
            states: List of state token sequences
            actions: List of action tokens

        Returns:
            Cross-entropy loss
        """
        # Build full sequence
        sequence = self.build_sequence(states, actions)

        # Forward pass
        state_logits, action_logits = self.forward(sequence)

        # For simplicity, compute loss on predicting next state
        # In practice, you'd properly mask and compute loss
        target = states[-1]  # Target is the last state
        logits = state_logits[0, -len(target):]

        # Cross-entropy loss
        probs = self._softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(len(target)), target] + 1e-8))

        return loss


################################################################################
# SECTION 5: TRANSFORMER WORLD MODEL (COMPLETE)
################################################################################

class TransformerWorldModel:
    """
    Complete Transformer World Model
    ==================================

    Combines autoregressive transformer with world model capabilities.

    Use cases:
        - Game simulation (GameNGen style)
        - Interactive environment generation
        - Physics simulation
        - Planning with look-ahead

    Interview Questions:
        1. "How does this scale to high-resolution video?"
           Use a VQ-VAE to compress frames to discrete tokens,
           then model the token sequence with the transformer.
           This is the approach used by GameNGen.

        2. "Can this model do zero-shot generalization?"
           To some extent. If trained on diverse environments,
           it can generalize to new scenarios. But it's limited
           by the training distribution.

        3. "How do you handle continuous actions?"
           Discretize them. Or use a hybrid approach: discrete
           tokens for states, continuous embeddings for actions.
    """

    def __init__(self, config: Optional[TransformerWorldConfig] = None):
        if config is None:
            config = TransformerWorldConfig()
        self.config = config
        self.model = AutoregressiveWorldModel(config)

    def simulate(
        self,
        initial_state: np.ndarray,
        actions: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Simulate future states given initial state and actions.

        Args:
            initial_state: Starting state tokens
            actions: Sequence of action tokens

        Returns:
            List of predicted state token sequences
        """
        states = [initial_state]
        history_actions = []

        for action in actions:
            next_state = self.model.generate_next_state(
                states, history_actions + [action]
            )
            states.append(next_state)
            history_actions.append(action)

        return states[1:]  # Return generated states (not initial)


################################################################################
# SECTION 6: TESTING & DEMONSTRATION
################################################################################

def demonstrate_transformer_world_model():
    """Demonstrate transformer world model."""
    print("=" * 70)
    print("TRANSFORMER WORLD MODEL")
    print("=" * 70)

    # Configuration
    config = TransformerWorldConfig(
        state_vocab_size=256,
        state_seq_len=8,
        action_vocab_size=16,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        context_len=5
    )

    print(f"\nConfiguration:")
    print(f"  State vocab: {config.state_vocab_size}")
    print(f"  State seq len: {config.state_seq_len}")
    print(f"  Action vocab: {config.action_vocab_size}")
    print(f"  Model dim: {config.d_model}")
    print(f"  Layers: {config.n_layers}")

    # Create model
    model = TransformerWorldModel(config)

    # Dummy data
    state = np.random.randint(0, config.state_vocab_size, (1, config.state_seq_len))
    action = np.array([[3]])  # Single action token

    # Build sequence
    sequence = model.model.build_sequence([state], [action])
    print(f"\nSequence shape: {sequence.shape}")

    # Forward pass
    state_logits, action_logits = model.model.forward(sequence)
    print(f"State logits shape: {state_logits.shape}")
    print(f"Action logits shape: {action_logits.shape}")

    # Generate next state
    next_state = model.model.generate_next_state(
        [state], [action], temperature=0.8
    )
    print(f"\nGenerated next state: {next_state}")

    # Simulate trajectory
    actions = [np.array([i % config.action_vocab_size]) for i in range(3)]
    trajectory = model.simulate(state, actions)
    print(f"\nSimulated {len(trajectory)} future states")
    for i, s in enumerate(trajectory):
        print(f"  State {i+1}: {s[:5]}...")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Transformer world models generate states autoregressively!")
    print("Like GPT predicts words, this predicts future states.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_transformer_world_model()
