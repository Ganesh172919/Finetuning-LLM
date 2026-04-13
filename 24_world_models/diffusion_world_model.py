"""
################################################################################
DIFFUSION WORLD MODEL — PREDICTING FUTURES WITH DIFFUSION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Diffusion World Model?
    A world model that uses diffusion processes to predict future states.
    Instead of predicting a single deterministic future, it models the
    distribution of possible futures — capturing uncertainty in the world.

Why does it matter?
    Traditional world models predict one future (deterministic):
    - Misses stochastic aspects of the real world
    - Can't capture multiple possible outcomes
    - Blurry predictions when future is uncertain

    Diffusion world models predict DISTRIBUTIONS:
    - Model uncertainty explicitly
    - Generate diverse possible futures
    - Sharper predictions through iterative denoising
    - Better for planning under uncertainty

How does it work?
    1. Encode current state and action
    2. Add noise to target future state (forward diffusion)
    3. Learn to denoise given context (reverse diffusion)
    4. At inference: start from noise, iteratively denoise to generate future

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Diffusion World Model                            │
    │                                                                  │
    │  Current State ──▶ State Encoder ──▶ z_current                  │
    │                                              ↓                   │
    │  Action ──────────────────────────▶ Conditioning                 │
    │                                              ↓                   │
    │  Noisy Future ──▶ Denoiser ──▶ Predicted Noise                  │
    │         ↑                    (conditioned on z_current + action) │
    │         │                                                        │
    │    Iterative Denoising (T steps)                                 │
    │         │                                                        │
    │    Start: Pure Noise ──▶ End: Predicted Future State             │
    └─────────────────────────────────────────────────────────────────┘

Key Innovation (DIAMOND, 2024):
    - Uses diffusion for world model prediction
    - Action-conditioned denoising
    - Handles stochastic environments naturally
    - Better than deterministic world models for planning

Interview Questions:
    1. "Why use diffusion for world models?"
       Real-world dynamics are stochastic — multiple futures are possible.
       Diffusion naturally models this distribution, generating diverse
       plausible futures rather than a single blurry average.

    2. "How does action conditioning work in diffusion world models?"
       The denoiser network takes both the noisy future state AND the
       action as input. This conditions the denoising process on what
       action was taken, so the generated future reflects that action.

    3. "What's the advantage over deterministic world models?"
       Deterministic models produce blurry predictions when the future
       is uncertain (averaging over possibilities). Diffusion models
       can generate sharp, specific futures by sampling from the
       learned distribution.

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
class DiffusionWorldConfig:
    """
    Configuration for Diffusion World Model.
    """
    # State dimensions
    state_dim: int = 64
    action_dim: int = 4

    # Diffusion parameters
    num_diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 4

    # Conditioning
    condition_dim: int = 128


################################################################################
# SECTION 2: NOISE SCHEDULE
################################################################################

class NoiseSchedule:
    """
    Noise Schedule for Diffusion
    =============================

    Controls how much noise is added at each timestep.

    Linear schedule (DDPM default):
        β_t = β_start + (β_end - β_start) × t / T

    Cosine schedule (improved):
        ᾱ_t = cos²(π/2 × t/T)
        β_t = 1 - ᾱ_t/ᾱ_{t-1}

    The noise schedule determines how quickly information is destroyed.
    Too fast: model can't learn to denoise
    Too slow: too many steps needed

    Formula:
        q(x_t | x_0) = N(x_t; √ᾱ_t × x_0, (1-ᾱ_t) × I)

    Where ᾱ_t = ∏_{s=1}^{t} (1-β_s)
    """

    def __init__(self, config: DiffusionWorldConfig):
        self.config = config
        self.num_steps = config.num_diffusion_steps

        # Linear beta schedule
        self.betas = np.linspace(config.beta_start, config.beta_end, self.num_steps)

        # Alpha values
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)

        # Pre-compute useful quantities
        self.sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - self.alpha_cumprod)

    def add_noise(self, x_0: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add noise to clean data at timestep t.

        This is the FORWARD diffusion process:
            q(x_t | x_0) = N(√ᾱ_t × x_0, (1-ᾱ_t) × I)

        Args:
            x_0: Clean data (batch, state_dim)
            t: Timestep (0 = clean, T = pure noise)

        Returns:
            x_t: Noisy version of x_0
            noise: The noise that was added
        """
        noise = np.random.randn(*x_0.shape)

        sqrt_alpha = self.sqrt_alpha_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t]

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

        return x_t, noise

    def get_variance(self, t: int) -> float:
        """Get the variance at timestep t."""
        return 1.0 - self.alpha_cumprod[t]


################################################################################
# SECTION 3: DENOISER NETWORK
################################################################################

class DenoiserNetwork:
    """
    Denoiser Network for World Model
    ==================================

    Predicts the noise added to a state, given the noisy state,
    timestep, and conditioning (current state + action).

    Architecture:
        Input: [noisy_state, timestep_embed, condition]
        → MLP with residual connections
        → Predicted noise

    The denoiser learns:
        ε_θ(x_t, t, c) ≈ ε (the noise that was added)

    Training loss:
        L = E[||ε - ε_θ(x_t, t, c)||²]

    Where:
        x_t = noisy future state
        t = diffusion timestep
        c = conditioning (current state + action)
        ε = actual noise added
    """

    def __init__(self, config: DiffusionWorldConfig):
        self.config = config
        self.state_dim = config.state_dim
        self.hidden_dim = config.hidden_dim
        self.condition_dim = config.condition_dim

        # Timestep embedding
        self.time_embed_dim = 64
        self.time_mlp_w1 = np.random.randn(1, self.time_embed_dim) * 0.02
        self.time_mlp_b1 = np.zeros(self.time_embed_dim)
        self.time_mlp_w2 = np.random.randn(self.time_embed_dim, self.time_embed_dim) * 0.02
        self.time_mlp_b2 = np.zeros(self.time_embed_dim)

        # Main network layers
        input_dim = config.state_dim + self.time_embed_dim + config.condition_dim
        self.layers = []
        dims = [input_dim] + [config.hidden_dim] * config.num_layers + [config.state_dim]

        for i in range(len(dims) - 1):
            self.layers.append({
                'W': np.random.randn(dims[i], dims[i+1]) * 0.02,
                'b': np.zeros(dims[i+1])
            })

    def sinusoidal_embedding(self, t: int) -> np.ndarray:
        """
        Create sinusoidal embedding for timestep.

        Similar to positional encoding in transformers:
            PE(t, 2i) = sin(t / 10000^(2i/d))
            PE(t, 2i+1) = cos(t / 10000^(2i/d))

        This gives each timestep a unique embedding.
        """
        half_dim = self.time_embed_dim // 2
        emb = np.zeros(self.time_embed_dim)

        for i in range(half_dim):
            freq = np.exp(-np.log(10000.0) * i / half_dim)
            emb[2*i] = np.sin(t * freq)
            emb[2*i+1] = np.cos(t * freq)

        return emb

    def forward(
        self,
        x_t: np.ndarray,
        t: int,
        condition: np.ndarray
    ) -> np.ndarray:
        """
        Predict noise given noisy state and conditioning.

        Args:
            x_t: Noisy state (batch, state_dim)
            t: Diffusion timestep
            condition: Conditioning vector (batch, condition_dim)

        Returns:
            Predicted noise (batch, state_dim)
        """
        batch_size = x_t.shape[0]

        # Timestep embedding
        t_embed = self.sinusoidal_embedding(t)
        t_embed = np.tanh(t_embed @ self.time_mlp_w1 + self.time_mlp_b1)
        t_embed = t_embed @ self.time_mlp_w2 + self.time_mlp_b2
        t_embed = np.tile(t_embed, (batch_size, 1))

        # Concatenate inputs
        h = np.concatenate([x_t, t_embed, condition], axis=-1)

        # Forward through layers with ReLU
        for i, layer in enumerate(self.layers[:-1]):
            h = h @ layer['W'] + layer['b']
            h = np.maximum(0, h)  # ReLU

        # Final layer (no activation)
        final = self.layers[-1]
        output = h @ final['W'] + final['b']

        return output


################################################################################
# SECTION 4: STATE ENCODER
################################################################################

class StateEncoder:
    """
    State Encoder
    ==============

    Encodes the current observation into a conditioning vector.
    This conditioning guides the denoiser to predict appropriate futures.

    Input: Current state (raw observation)
    Output: Conditioning vector (latent representation)
    """

    def __init__(self, state_dim: int, condition_dim: int):
        self.W1 = np.random.randn(state_dim, condition_dim) * 0.02
        self.b1 = np.zeros(condition_dim)
        self.W2 = np.random.randn(condition_dim, condition_dim) * 0.02
        self.b2 = np.zeros(condition_dim)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Encode state to conditioning vector."""
        h = np.tanh(state @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


################################################################################
# SECTION 5: ACTION-CONDITIONED DIFFUSION
################################################################################

class ActionConditionedDiffusion:
    """
    Action-Conditioned Diffusion for World Modeling
    =================================================

    Generates future states conditioned on actions.

    Training:
        1. Sample clean future state x_0
        2. Sample random timestep t
        3. Add noise: x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε
        4. Condition on current state and action
        5. Train denoiser to predict ε

    Inference:
        1. Start from pure noise x_T
        2. Iteratively denoise for T steps
        3. Each step conditioned on current state + action
        4. Final output: predicted future state

    Interview Question:
        "How do you generate diverse futures?"
        Run the reverse diffusion multiple times with different
        random noise seeds. Each produces a different but plausible
        future state, capturing the distribution of possibilities.
    """

    def __init__(self, config: DiffusionWorldConfig):
        self.config = config
        self.noise_schedule = NoiseSchedule(config)
        self.state_encoder = StateEncoder(config.state_dim, config.condition_dim)
        self.denoiser = DenoiserNetwork(config)

    def create_condition(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Create conditioning vector from current state and action.

        Args:
            current_state: Current observation (batch, state_dim)
            action: Action taken (batch, action_dim)

        Returns:
            Conditioning vector (batch, condition_dim)
        """
        # Encode state
        state_cond = self.state_encoder.forward(current_state)

        # Project action to same dimension
        action_proj = action @ np.random.randn(
            self.config.action_dim, self.config.condition_dim
        ) * 0.02

        # Combine (additive conditioning)
        return state_cond + action_proj

    def training_loss(
        self,
        current_state: np.ndarray,
        future_state: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        Compute training loss for diffusion world model.

        The loss is the MSE between predicted and actual noise:
            L = E[||ε - ε_θ(x_t, t, c)||²]

        Args:
            current_state: Current observation
            future_state: Future observation (ground truth)
            action: Action taken

        Returns:
            Scalar loss value
        """
        batch_size = current_state.shape[0]

        # Sample random timestep
        t = np.random.randint(0, self.config.num_diffusion_steps)

        # Add noise to future state
        x_t, noise = self.noise_schedule.add_noise(future_state, t)

        # Create conditioning
        condition = self.create_condition(current_state, action)

        # Predict noise
        predicted_noise = self.denoiser.forward(x_t, t, condition)

        # MSE loss
        loss = np.mean((noise - predicted_noise) ** 2)

        return loss

    def generate_future(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
        num_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate future state by reverse diffusion.

        Start from noise and iteratively denoise.

        Args:
            current_state: Current observation (batch, state_dim)
            action: Action to take (batch, action_dim)
            num_steps: Number of denoising steps (default: all)

        Returns:
            Predicted future state (batch, state_dim)
        """
        if num_steps is None:
            num_steps = self.config.num_diffusion_steps

        batch_size = current_state.shape[0]

        # Start from pure noise
        x = np.random.randn(batch_size, self.config.state_dim)

        # Create conditioning
        condition = self.create_condition(current_state, action)

        # Iterative denoising (simplified DDPM sampling)
        for t in reversed(range(num_steps)):
            # Predict noise
            predicted_noise = self.denoiser.forward(x, t, condition)

            # Denoise step
            alpha = self.noise_schedule.alphas[t]
            alpha_cumprod = self.noise_schedule.alpha_cumprod[t]

            # DDPM update rule
            x = (1 / np.sqrt(alpha)) * (
                x - (1 - alpha) / np.sqrt(1 - alpha_cumprod) * predicted_noise
            )

            # Add noise (except at last step)
            if t > 0:
                noise = np.random.randn(*x.shape)
                beta = self.noise_schedule.betas[t]
                x = x + np.sqrt(beta) * noise

        return x


################################################################################
# SECTION 6: DIFFUSION WORLD MODEL (COMPLETE)
################################################################################

class DiffusionWorldModel:
    """
    Complete Diffusion World Model
    ================================

    Combines all components for world modeling with diffusion.

    Use cases:
        - Robot planning: Imagine outcomes of actions
        - Game AI: Simulate game states
        - Autonomous driving: Predict traffic scenarios
        - Scientific simulation: Model physical systems

    Interview Questions:
        1. "How many denoising steps do you need?"
           Typically 50-100 for good quality. Fewer steps (10-20)
           with DDIM sampling can be faster. Trade-off: more steps
           = better quality but slower inference.

        2. "How does this compare to GAN-based world models?"
           Diffusion: more stable training, better mode coverage,
           but slower inference. GANs: faster inference, but can
           suffer from mode collapse and training instability.

        3. "Can you do planning with this model?"
           Yes! Generate multiple futures for each candidate action,
           evaluate them with a reward model, pick the best action.
           This is "model-predictive control" with diffusion.
    """

    def __init__(self, config: Optional[DiffusionWorldConfig] = None):
        if config is None:
            config = DiffusionWorldConfig()
        self.config = config
        self.diffusion = ActionConditionedDiffusion(config)

    def imagine(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
        num_futures: int = 5
    ) -> np.ndarray:
        """
        Imagine multiple possible futures.

        Generate diverse futures by sampling different noise seeds.

        Args:
            current_state: Current observation
            action: Action to take
            num_futures: Number of futures to generate

        Returns:
            Array of possible futures (num_futures, batch, state_dim)
        """
        futures = []
        for _ in range(num_futures):
            future = self.diffusion.generate_future(current_state, action)
            futures.append(future)

        return np.stack(futures)

    def plan(
        self,
        current_state: np.ndarray,
        candidate_actions: np.ndarray,
        reward_fn=None,
        num_futures: int = 5
    ) -> Tuple[int, np.ndarray]:
        """
        Plan by imagining futures for each candidate action.

        1. For each candidate action, generate multiple futures
        2. Evaluate each future with reward function
        3. Pick the action with highest expected reward

        Args:
            current_state: Current observation
            candidate_actions: Array of possible actions (n_actions, action_dim)
            reward_fn: Function to evaluate future states
            num_futures: Futures per action

        Returns:
            best_action_idx: Index of best action
            best_future: The best predicted future
        """
        n_actions = candidate_actions.shape[0]
        best_action_idx = 0
        best_reward = -float('inf')
        best_future = None

        for i in range(n_actions):
            action = candidate_actions[i:i+1]

            # Generate multiple futures
            futures = self.imagine(current_state, action, num_futures)

            # Evaluate futures
            if reward_fn is not None:
                rewards = [reward_fn(f) for f in futures]
                avg_reward = np.mean(rewards)
            else:
                # Default: prefer futures close to a goal
                avg_reward = -np.mean([np.linalg.norm(f) for f in futures])

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_action_idx = i
                best_future = futures[0]

        return best_action_idx, best_future


################################################################################
# SECTION 7: TESTING & DEMONSTRATION
################################################################################

def demonstrate_diffusion_world_model():
    """Demonstrate diffusion world model."""
    print("=" * 70)
    print("DIFFUSION WORLD MODEL")
    print("=" * 70)

    # Configuration
    config = DiffusionWorldConfig(
        state_dim=16,
        action_dim=4,
        num_diffusion_steps=50,
        hidden_dim=64,
        num_layers=3,
        condition_dim=32
    )

    print(f"\nConfiguration:")
    print(f"  State dim: {config.state_dim}")
    print(f"  Action dim: {config.action_dim}")
    print(f"  Diffusion steps: {config.num_diffusion_steps}")
    print(f"  Hidden dim: {config.hidden_dim}")

    # Create model
    model = DiffusionWorldModel(config)

    # Dummy data
    batch_size = 4
    current_state = np.random.randn(batch_size, config.state_dim)
    future_state = np.random.randn(batch_size, config.state_dim)
    action = np.random.randn(batch_size, config.action_dim)

    # Training loss
    loss = model.diffusion.training_loss(current_state, future_state, action)
    print(f"\nTraining loss: {loss:.4f}")

    # Generate future
    predicted_future = model.diffusion.generate_future(current_state, action, num_steps=10)
    print(f"Predicted future shape: {predicted_future.shape}")

    # Imagine multiple futures
    futures = model.imagine(current_state[:1], action[:1], num_futures=3)
    print(f"Multiple futures shape: {futures.shape}")

    # Planning
    candidate_actions = np.random.randn(5, config.action_dim)
    best_idx, best_future = model.plan(
        current_state[:1],
        candidate_actions,
        num_futures=2
    )
    print(f"\nPlanning result:")
    print(f"  Best action index: {best_idx}")
    print(f"  Best future shape: {best_future.shape}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Diffusion models capture DISTRIBUTIONS of futures!")
    print("This enables planning under uncertainty.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_diffusion_world_model()
