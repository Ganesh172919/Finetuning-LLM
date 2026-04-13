"""
################################################################################
DIFFUSION MODELS — THE MATHEMATICS OF IMAGE GENERATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Diffusion Models?
    Diffusion models learn to generate data by reversing a gradual
    noise addition process. Think of it like:
    1. Forward: Gradually add noise to an image until it's pure noise
    2. Reverse: Learn to gradually remove noise to create an image

    Like un-shredding a document: you learn to reassemble pieces.

Why do they matter?
    Diffusion models are the foundation of:
    - Stable Diffusion (open-source)
    - DALL-E 3 (OpenAI)
    - Midjourney
    - Imagen (Google)
    - Flux (Black Forest Labs)

    They produce higher quality images than GANs and are more stable.

How do they work?
    Forward Process (adding noise):
    x₀ → x₁ → x₂ → ... → xₜ (pure noise)
    Each step adds a small amount of Gaussian noise.

    Reverse Process (removing noise):
    xₜ → xₜ₋₁ → ... → x₁ → x₀ (clean image)
    A neural network learns to predict and remove the noise.

    Training:
    1. Take a clean image x₀
    2. Add noise: xₜ = √(αₜ) * x₀ + √(1-αₜ) * ε
    3. Train network to predict ε from xₜ
    4. Loss: ||ε - ε_θ(xₜ, t)||²

    Generation:
    1. Start with pure noise xₜ ~ N(0, I)
    2. Iteratively denoise: xₜ → xₜ₋₁ → ... → x₀
    3. Result: generated image!

Mathematical Intuition:
    Forward: q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)
    Reverse: p(xₜ₋₁|xₜ) = N(xₜ₋₁; μ_θ(xₜ,t), Σ_θ(xₜ,t))

    The model learns μ_θ (the mean of the reverse distribution).

########################################

KEY PAPERS:
    [1] Ho et al. (2020). Denoising Diffusion Probabilistic Models (DDPM)
    [2] Song et al. (2021). Score-Based Generative Modeling
    [3] Rombach et al. (2022). High-Resolution Image Synthesis with LDM
    [4] Esser et al. (2024). Scaling Rectified Flow Transformers (Flux)

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List
import math

################################################################################
# SECTION 1: NOISE SCHEDULER
################################################################################

class NoiseScheduler:
    """
    Noise Scheduler for Diffusion Models
    ======================================

    Definition: Controls how much noise is added at each timestep.

    The scheduler defines:
    - βₜ: noise schedule (how much noise per step)
    - αₜ = 1 - βₜ
    - ᾱₜ = Πᵢ₌₁ᵗ αᵢ (cumulative product)

    Common Schedules:
    1. Linear: β increases linearly from β_start to β_end
    2. Cosine: β follows a cosine curve
    3. Scaled Linear: β increases with a power law

    Why it matters:
    - Too much noise early: lose signal too fast
    - Too much noise late: model can't learn
    - Good schedule: smooth transition from signal to noise

    Visual:
    Timestep:  0                    T
    Noise:     [low ─────────── high]
    Image:     [clean ──────── noise]

    Interview Question:
        "What noise schedule should I use?"
        Cosine schedule works best for most applications.
        It provides a smoother transition than linear.
        For faster generation, use fewer steps with adjusted schedule.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "cosine"
    ):
        """
        Initialize noise scheduler.

        Args:
            num_timesteps: Number of diffusion steps (T)
            beta_start: Starting noise level
            beta_end: Ending noise level
            schedule: "linear" or "cosine"
        """
        self.num_timesteps = num_timesteps

        if schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            # Cosine schedule from "Improved DDPM" paper
            steps = np.arange(num_timesteps + 1) / num_timesteps
            alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            self.betas = np.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Compute alpha values
        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)  # ᾱₜ

        # Precompute coefficients for forward process
        self.sqrt_alpha_bar = np.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = np.sqrt(1.0 - self.alpha_bar)

    def add_noise(
        self,
        x_0: np.ndarray,
        t: int,
        noise: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add noise to clean image at timestep t.

        Formula: xₜ = √(ᾱₜ) * x₀ + √(1-ᾱₜ) * ε

        Args:
            x_0: Clean image [...]
            t: Timestep (0 to T-1)
            noise: Optional pre-generated noise (same shape as x_0)

        Returns:
            x_t: Noisy image at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = np.random.randn(*x_0.shape)

        sqrt_alpha = self.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_bar[t]

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

        return x_t, noise

    def get_variance(self, t: int) -> float:
        """
        Get variance of noise at timestep t.

        Var(xₜ) = 1 - ᾱₜ

        At t=0: variance ≈ 0 (almost clean)
        At t=T: variance ≈ 1 (almost pure noise)
        """
        return 1.0 - self.alpha_bar[t]


################################################################################
# SECTION 2: DENOISING NETWORK (SIMPLIFIED)
################################################################################

class SimpleDenoiser:
    """
    Simple Denoising Network
    =========================

    Definition: A neural network that predicts noise in an image.

    Input: noisy image xₜ, timestep t
    Output: predicted noise ε_θ(xₜ, t)

    Architecture (simplified):
    - MLP or small CNN
    - Takes flattened image + timestep embedding
    - Outputs predicted noise of same shape

    In practice, this is a U-Net (see unet.py).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Simple MLP layers
        scale1 = math.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = math.sqrt(2.0 / (hidden_dim + input_dim))

        self.W1 = np.random.randn(input_dim + 1, hidden_dim) * scale1
        self.W2 = np.random.randn(hidden_dim, input_dim) * scale2

    def forward(self, x_t: np.ndarray, t: int) -> np.ndarray:
        """
        Predict noise in x_t at timestep t.

        Args:
            x_t: Noisy image (flattened) [batch × input_dim]
            t: Timestep

        Returns:
            predicted_noise: [batch × input_dim]
        """
        batch_size = x_t.shape[0]

        # Encode timestep
        t_encoded = np.full((batch_size, 1), t / 1000.0)

        # Concatenate image and timestep
        x_input = np.concatenate([x_t, t_encoded], axis=-1)

        # MLP
        h = np.maximum(0, np.matmul(x_input, self.W1))  # ReLU
        predicted_noise = np.matmul(h, self.W2)

        return predicted_noise


################################################################################
# SECTION 3: DIFFUSION MODEL
################################################################################

class DiffusionModel:
    """
    DDPM: Denoising Diffusion Probabilistic Model
    ================================================

    Definition: A generative model that learns to reverse a noise process.

    Training Algorithm:
    1. Sample clean image x₀ from dataset
    2. Sample random timestep t ~ Uniform(0, T)
    3. Sample noise ε ~ N(0, I)
    4. Create noisy image: xₜ = √(ᾱₜ)x₀ + √(1-ᾱₜ)ε
    5. Train model to predict ε from xₜ
    6. Loss: ||ε - ε_θ(xₜ, t)||²

    Sampling Algorithm (DDPM):
    1. Start with xₜ ~ N(0, I)
    2. For t = T, T-1, ..., 1:
       a. Predict noise: ε_θ(xₜ, t)
       b. Compute xₜ₋₁ using formula
    3. Return x₀

    Sampling Formula:
    xₜ₋₁ = (1/√αₜ) * (xₜ - (1-αₜ)/√(1-ᾱₜ) * ε_θ(xₜ, t)) + σₜz

    Where z ~ N(0, I) and σₜ is the noise schedule.

    Interview Questions:
        1. "How do diffusion models generate images?"
           Start with random noise, then iteratively denoise it.
           Each step removes a small amount of noise.
           After many steps, a clean image emerges.

        2. "Why are diffusion models better than GANs?"
           More stable training, better mode coverage, easier to scale.
           GANs can suffer from mode collapse and training instability.

        3. "How many denoising steps do you need?"
           Typically 20-1000 steps. More steps = higher quality but slower.
           Modern schedulers can generate good images in 20-50 steps.
    """

    def __init__(
        self,
        input_dim: int,
        num_timesteps: int = 1000,
        schedule: str = "cosine"
    ):
        """
        Initialize diffusion model.

        Args:
            input_dim: Dimension of input (flattened image)
            num_timesteps: Number of diffusion steps
            schedule: Noise schedule type
        """
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps

        # Noise scheduler
        self.scheduler = NoiseScheduler(num_timesteps, schedule=schedule)

        # Denoising network
        self.denoiser = SimpleDenoiser(input_dim)

    def training_loss(
        self,
        x_0: np.ndarray,
        t: Optional[int] = None
    ) -> float:
        """
        Compute training loss.

        The loss is the MSE between actual and predicted noise.

        Args:
            x_0: Clean images [batch × input_dim]
            t: Optional timestep (random if not provided)

        Returns:
            loss: MSE loss
        """
        batch_size = x_0.shape[0]

        # Sample random timestep
        if t is None:
            t = np.random.randint(0, self.num_timesteps)

        # Sample noise
        noise = np.random.randn(*x_0.shape)

        # Add noise to get x_t
        x_t, _ = self.scheduler.add_noise(x_0, t, noise)

        # Predict noise
        predicted_noise = self.denoiser.forward(x_t, t)

        # MSE loss
        loss = np.mean((noise - predicted_noise) ** 2)

        return loss

    @staticmethod
    def sample(
        model: 'DiffusionModel',
        shape: Tuple[int, ...],
        num_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate images by iterative denoising.

        Args:
            model: Trained diffusion model
            shape: Shape of images to generate [batch × input_dim]
            num_steps: Number of denoising steps (default: all)

        Returns:
            x_0: Generated images [batch × input_dim]

        Algorithm:
            1. Start with pure noise
            2. Iteratively denoise
            3. Return clean image
        """
        if num_steps is None:
            num_steps = model.num_timesteps

        # Start with pure noise
        x_t = np.random.randn(*shape)

        # Denoise iteratively
        timesteps = np.linspace(
            model.num_timesteps - 1, 0, num_steps, dtype=int
        )

        for t in timesteps:
            # Predict noise
            predicted_noise = model.denoiser.forward(x_t, t)

            # Get scheduler values
            alpha = model.scheduler.alphas[t]
            alpha_bar = model.scheduler.alpha_bar[t]
            beta = model.scheduler.betas[t]

            # Compute x_{t-1}
            # x_{t-1} = (1/√αₜ) * (xₜ - (1-αₜ)/√(1-ᾱₜ) * ε_θ) + σₜz
            coeff1 = 1.0 / np.sqrt(alpha)
            coeff2 = (1 - alpha) / np.sqrt(1 - alpha_bar)

            mean = coeff1 * (x_t - coeff2 * predicted_noise)

            # Add noise (except at last step)
            if t > 0:
                noise = np.random.randn(*x_t.shape)
                x_t = mean + np.sqrt(beta) * noise
            else:
                x_t = mean

        return x_t


################################################################################
# SECTION 4: SCORE-BASED GENERATIVE MODELING
################################################################################

class ScoreModel:
    """
    Score-Based Generative Modeling
    ================================

    Definition: Instead of predicting noise, predict the SCORE
    (gradient of log probability density).

    Score: s(x) = ∇_x log p(x)

    This is equivalent to noise prediction:
    ε_θ(xₜ, t) ≈ -√(1-ᾱₜ) * s_θ(xₜ, t)

    The score tells you which direction to move to increase probability.

    Visual:
    High density region ←─────── s(x) points toward here
    Low density region

    Interview Question:
        "What is score-based modeling?"
        It learns the gradient of the log probability density (score).
        The score points toward regions of higher probability.
        Sampling follows the score through Langevin dynamics.
    """

    def __init__(self, input_dim: int, sigma_min: float = 0.01, sigma_max: float = 1.0):
        self.input_dim = input_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def score(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """
        Compute score (gradient of log density).

        In practice, this is learned by a neural network.
        """
        # Placeholder: Gaussian score
        return -x / (sigma ** 2)

    def langevin_step(
        self,
        x: np.ndarray,
        sigma: float,
        step_size: float
    ) -> np.ndarray:
        """
        One step of Langevin dynamics.

        x_{t+1} = x_t + (ε/2) * s(x_t) + √ε * z

        This follows the score to higher probability regions.
        """
        score = self.score(x, sigma)
        noise = np.random.randn(*x.shape)

        x_new = x + (step_size / 2) * score + np.sqrt(step_size) * noise
        return x_new


################################################################################
# SECTION 5: TESTING & EXAMPLES
################################################################################

def demonstrate_diffusion():
    """Demonstrate diffusion model concepts."""
    print("=" * 70)
    print("DIFFUSION MODEL DEMONSTRATION")
    print("=" * 70)

    # Noise Scheduler
    print("\n--- Noise Scheduler ---")
    scheduler = NoiseScheduler(num_timesteps=1000)
    print(f"Number of timesteps: {scheduler.num_timesteps}")
    print(f"Beta range: [{scheduler.betas[0]:.6f}, {scheduler.betas[-1]:.6f}]")
    print(f"Alpha_bar at t=0: {scheduler.alpha_bar[0]:.4f}")
    print(f"Alpha_bar at t=500: {scheduler.alpha_bar[500]:.4f}")
    print(f"Alpha_bar at t=999: {scheduler.alpha_bar[999]:.4f}")

    # Add noise demonstration
    print("\n--- Adding Noise ---")
    x_0 = np.random.randn(1, 64)  # "Clean" image
    for t in [0, 250, 500, 750, 999]:
        x_t, noise = scheduler.add_noise(x_0, t)
        signal_to_noise = np.var(x_0) / np.var(noise)
        print(f"t={t}: signal_var={np.var(x_t):.4f}, SNR={signal_to_noise:.4f}")

    # Diffusion Model
    print("\n--- Diffusion Model ---")
    model = DiffusionModel(input_dim=64, num_timesteps=1000)

    # Training loss
    loss = model.training_loss(x_0)
    print(f"Training loss: {loss:.4f}")

    # Sampling
    print("\n--- Sampling ---")
    generated = DiffusionModel.sample(model, shape=(1, 64), num_steps=50)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated mean: {generated.mean():.4f}")
    print(f"Generated std: {generated.std():.4f}")

    # Score model
    print("\n--- Score Model ---")
    score_model = ScoreModel(input_dim=64)
    x = np.random.randn(1, 64)
    score = score_model.score(x, sigma=1.0)
    print(f"Score shape: {score.shape}")
    print(f"Score points toward origin: {np.allclose(score, -x, atol=0.1)}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_diffusion()


################################################################################
# REFERENCES
################################################################################

# [1] Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models.
# [2] Song, Y., et al. (2021). Score-Based Generative Modeling through SDEs.
# [3] Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models.
# [4] Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models.
# [5] Esser, P., et al. (2024). Scaling Rectified Flow Transformers for Image Synthesis.

################################################################################
