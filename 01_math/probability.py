"""
################################################################################
PROBABILITY & INFORMATION THEORY — THE FOUNDATION OF AI LEARNING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Probability?
    Probability measures the likelihood of events. In AI, it's how models
    express uncertainty — every prediction is a probability distribution.

Why do we need it?
    1. Loss functions: Cross-entropy measures prediction quality
    2. Sampling: Generate text by sampling from distributions
    3. Regularization: KL divergence prevents overfitting
    4. Bayesian methods: Prior knowledge + data = posterior
    5. Information theory: Measuring information content

What problem does it solve?
    - How confident is the model? → Probability distributions
    - How wrong is the prediction? → Cross-entropy loss
    - How much information is in a signal? → Entropy
    - How similar are two distributions? → KL divergence

Mathematical Intuition:
    A neural network outputs logits (raw scores).
    Softmax converts them to probabilities.
    Cross-entropy measures how far from the true distribution.
    Backpropagation adjusts weights to minimize this distance.

Real-world Analogy:
    Think of a weather forecast:
    - 70% chance of rain, 30% sunshine (probability distribution)
    - If it rains, the forecast was good (low cross-entropy)
    - If it's sunny, the forecast was wrong (high cross-entropy)
    - A confident forecast (95%/5%) is more extreme than uncertain (55%/45%)
    - KL divergence measures how different these distributions are

########################################
"""

import numpy as np
from typing import List, Optional, Tuple
import math

################################################################################
# SECTION 1: PROBABILITY DISTRIBUTIONS
################################################################################

class Distribution:
    """
    Base class for probability distributions.

    ########################################
    DISTRIBUTIONS IN AI
    ########################################

    1. Categorical: Output of language models (next token prediction)
    2. Gaussian: VAE latent spaces, diffusion models
    3. Bernoulli: Binary classification outputs
    4. Multinomial: Multi-class sampling

    ########################################
    """

    def sample(self, n: int = 1) -> np.ndarray:
        raise NotImplementedError

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def prob(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_prob(x))

    def entropy(self) -> float:
        raise NotImplementedError


class Categorical(Distribution):
    """
    Categorical Distribution
    =========================

    Definition: Distribution over a finite set of categories.
    This is THE output distribution of language models.

    When GPT predicts the next token, it outputs a categorical
    distribution over the entire vocabulary (50,000+ tokens).

    Formula:
        P(X = k) = p_k, where Σ p_k = 1

    Example:
        Vocabulary: ["the", "cat", "sat", "on"]
        Probabilities: [0.5, 0.3, 0.15, 0.05]
        → Model thinks "the" is most likely next token

    Interview Question:
        "How does a language model generate text?"
        Answer: It outputs a categorical distribution over the vocabulary,
        then samples from it. Temperature controls how "peaked" the
        distribution is (low temp = confident, high temp = random).
    """

    def __init__(self, logits: np.ndarray):
        """
        Initialize from logits (raw scores before softmax).

        Args:
            logits: Raw scores for each category [num_categories]

        Why logits instead of probabilities?
        - Logits are unbounded (can be any real number)
        - Probabilities must sum to 1 (constrained)
        - Numerical stability: working in log space prevents underflow
        """
        self.logits = logits
        # Numerically stable softmax
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        self.probs = exp_logits / np.sum(exp_logits)
        self.log_probs = np.log(self.probs + 1e-10)  # Add epsilon for stability

    def sample(self, n: int = 1) -> np.ndarray:
        """
        Sample from the distribution.

        This is how language models generate text:
        1. Compute logits for each token in vocabulary
        2. Convert to probabilities via softmax
        3. Sample from the categorical distribution
        4. The sampled token becomes the next input

        Temperature controls randomness:
        - temp → 0: always pick most likely (greedy)
        - temp = 1: sample according to probabilities
        - temp → ∞: uniform random (maximum randomness)
        """
        return np.random.choice(len(self.probs), size=n, p=self.probs)

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Log probability of given outcomes.

        Used in loss computation:
        loss = -log(P(correct_token))

        This is cross-entropy loss!
        """
        return self.log_probs[x]

    def entropy(self) -> float:
        """
        Entropy of the distribution.

        Measures uncertainty:
        - H = 0: certain (one category has prob 1)
        - H = log(n): uniform (all categories equally likely)

        In language modeling:
        - Low entropy: model is confident about next token
        - High entropy: model is uncertain
        """
        return -np.sum(self.probs * self.log_probs)


class Gaussian(Distribution):
    """
    Gaussian (Normal) Distribution
    ===============================

    Definition: The "bell curve" distribution.
    Formula: P(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))

    Why it's everywhere in AI:
    1. VAE: Latent space is Gaussian
    2. Diffusion models: Noise is Gaussian
    3. Weight initialization: He/Xavier use Gaussian
    4. Central limit theorem: averages converge to Gaussian
    5. Maximum entropy: Gaussian has max entropy for given mean/variance

    Key Properties:
    - 68% of data within 1 std of mean
    - 95% within 2 std
    - 99.7% within 3 std

    Example:
        Image generation: Start with Gaussian noise, denoise into image.
        VAE: Encode image to Gaussian, sample, decode back.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
        self.var = std ** 2

    def sample(self, n: int = 1) -> np.ndarray:
        """
        Sample from Gaussian distribution.

        In diffusion models:
        noise = Gaussian(0, 1).sample(image_shape)
        noisy_image = sqrt(alpha) * image + sqrt(1-alpha) * noise
        """
        return np.random.normal(self.mean, self.std, n)

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Log probability density.

        Used in VAE loss:
        KL(q(z|x) || p(z)) where both are Gaussian
        """
        return -0.5 * np.log(2 * np.pi * self.var) - (x - self.mean) ** 2 / (2 * self.var)

    def entropy(self) -> float:
        """
        Entropy of Gaussian: 0.5 * log(2πeσ²)

        Larger std → more entropy → more uncertainty
        """
        return 0.5 * np.log(2 * np.pi * np.e * self.var)


################################################################################
# SECTION 2: LOSS FUNCTIONS
################################################################################

def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Cross-Entropy Loss
    ===================

    Definition: Measures how bad predictions are compared to true labels.
    Formula: H(p, q) = -Σ p(x) * log(q(x))

    For classification with one-hot targets:
    H = -log(q(target_class))

    This is THE loss function for language models:
    - Model predicts P(next_token) for all tokens
    - Cross-entropy measures how far from the true token
    - Training minimizes this loss

    WHY IT WORKS:
    =============
    Cross-entropy measures the "surprise" of seeing the true token
    given the model's predictions.

    If model assigns high prob to correct token → low loss (good!)
    If model assigns low prob to correct token → high loss (bad!)

    Mathematical derivation:
    - Maximum likelihood: maximize Σ log P(correct_token)
    - Negative log likelihood: minimize -Σ log P(correct_token)
    - This IS cross-entropy loss!

    Args:
        logits: Raw model outputs [batch_size × vocab_size]
        targets: True token indices [batch_size]

    Returns:
        Average cross-entropy loss over the batch

    Example:
        logits = [[2.0, 1.0, 0.1]]  # Model's raw scores
        targets = [0]                 # True token is index 0

        After softmax: [0.659, 0.242, 0.099]
        Loss = -log(0.659) = 0.417

    Interview Questions:
        1. "Why cross-entropy instead of MSE for classification?"
           Cross-entropy has stronger gradients when predictions are wrong,
           leading to faster learning. MSE gradients vanish near extremes.

        2. "What's the relationship between cross-entropy and KL divergence?"
           H(p,q) = H(p) + KL(p||q)
           Since H(p) is constant during training, minimizing cross-entropy
           is equivalent to minimizing KL divergence.

        3. "How does label smoothing affect cross-entropy?"
           Instead of one-hot [0,0,1,0], use [0.033, 0.033, 0.9, 0.033]
           This prevents overconfident predictions and improves generalization.
    """
    # Numerically stable: subtract max
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
    log_probs = shifted[np.arange(len(targets)), targets] - log_sum_exp
    return -np.mean(log_probs)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL Divergence
    ==============

    Definition: Measures how different two distributions are.
    Formula: KL(P||Q) = Σ P(x) * log(P(x)/Q(x))

    Properties:
    - KL(P||Q) ≥ 0 (always non-negative)
    - KL(P||Q) = 0 iff P = Q
    - NOT symmetric: KL(P||Q) ≠ KL(Q||P)

    Why it matters in AI:
    1. VAE: KL(q(z|x) || p(z)) regularizes latent space
    2. Knowledge distillation: KL(student || teacher)
    3. RLHF: KL(policy || reference) prevents reward hacking
    4. Policy optimization: TRPO/PPO use KL constraints

    Example:
        In RLHF:
        - We want to optimize reward
        - But we don't want to deviate too much from the reference model
        - KL penalty: reward - β * KL(policy || reference)
        - This prevents the model from "cheating" the reward model

    Interview Question:
        "Why is KL divergence not symmetric?"
        KL(P||Q) penalizes where P has mass but Q doesn't (mode-seeking).
        KL(Q||P) penalizes where Q has mass but P doesn't (mode-covering).
        In practice, forward KL (P||Q) is often preferred.
    """
    # Add epsilon to avoid log(0)
    p = p + 1e-10
    q = q + 1e-10
    return np.sum(p * np.log(p / q))


def entropy(probs: np.ndarray) -> float:
    """
    Shannon Entropy
    ===============

    Definition: Measures the average "surprise" or information content.
    Formula: H(X) = -Σ P(x) * log(P(x))

    Units: bits (log base 2) or nats (natural log)

    Interpretation:
    - H = 0: certain event (no surprise)
    - H = log(n): maximum uncertainty (uniform distribution)

    Why it matters in AI:
    1. Decision trees: split on features with highest information gain
    2. Language modeling: perplexity = 2^H (how "confused" the model is)
    3. Exploration in RL: entropy bonus encourages exploration
    4. Regularization: maximize entropy of predictions

    Example:
        Fair coin: H = -0.5*log(0.5) - 0.5*log(0.5) = 1 bit
        Loaded coin (90/10): H = -0.9*log(0.9) - 0.1*log(0.1) = 0.47 bits
        Certain outcome: H = 0

    In Language Modeling:
        Perplexity = 2^H
        - Perplexity 1: model is certain (impossible in practice)
        - Perplexity 10: model considers 10 tokens equally likely
        - Perplexity 100: model is quite uncertain
    """
    probs = probs + 1e-10  # Avoid log(0)
    return -np.sum(probs * np.log(probs))


################################################################################
# SECTION 3: SAMPLING STRATEGIES
################################################################################

def temperature_scaled_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Temperature-Scaled Softmax
    ============================

    Definition: Adjust the "sharpness" of a probability distribution.

    Formula: P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)

    Temperature effects:
    - T → 0: Greedy (always pick most likely token)
    - T = 1: Normal sampling
    - T > 1: More random (exploration)
    - T → ∞: Uniform random

    Why it matters:
    - Creative writing: higher temperature (more diverse)
    - Factual answers: lower temperature (more deterministic)
    - Code generation: lower temperature (more precise)

    Example:
        logits = [2.0, 1.0, 0.1]
        T=0.5: [0.84, 0.14, 0.02] (confident)
        T=1.0: [0.66, 0.24, 0.10] (normal)
        T=2.0: [0.50, 0.30, 0.20] (uncertain)

    Interview Question:
        "How does temperature affect text generation?"
        Answer: Temperature scales logits before softmax.
        Low temp makes the distribution peakier (more deterministic),
        high temp flattens it (more random/creative).
    """
    scaled_logits = logits / temperature
    # Numerically stable softmax
    shifted = scaled_logits - np.max(scaled_logits)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits)


def top_k_sampling(logits: np.ndarray, k: int = 50, temperature: float = 1.0) -> int:
    """
    Top-K Sampling
    ==============

    Definition: Only sample from the K most likely tokens.

    Algorithm:
    1. Sort tokens by probability
    2. Keep only top K
    3. Renormalize probabilities
    4. Sample from the truncated distribution

    Why it matters:
    - Prevents sampling very unlikely tokens
    - Balances quality and diversity
    - Used in early GPT models

    Example:
        logits = [2.0, 1.5, 1.0, 0.5, 0.1, -1.0]
        k=3: Only consider first 3 tokens
        → More focused, less random

    Tradeoffs:
    - Small k: Less diverse, more coherent
    - Large k: More diverse, potentially less coherent
    - k=vocab_size: Same as normal sampling
    """
    scaled = logits / temperature
    # Get top-k indices
    top_k_indices = np.argsort(scaled)[-k:]
    # Zero out everything else
    filtered = np.full_like(scaled, -np.inf)
    filtered[top_k_indices] = scaled[top_k_indices]
    # Softmax and sample
    probs = temperature_scaled_softmax(filtered, temperature=1.0)
    return np.random.choice(len(probs), p=probs)


def top_p_sampling(logits: np.ndarray, p: float = 0.9, temperature: float = 1.0) -> int:
    """
    Top-P (Nucleus) Sampling
    =========================

    Definition: Sample from the smallest set of tokens whose
    cumulative probability exceeds P.

    Algorithm:
    1. Sort tokens by probability (descending)
    2. Accumulate probabilities until sum ≥ P
    3. Sample from this "nucleus"

    Why it's better than Top-K:
    - Adapts to the distribution's shape
    - When model is confident: nucleus is small (focused)
    - When model is uncertain: nucleus is large (diverse)
    - Used in GPT-3, ChatGPT, Claude

    Example:
        probs = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]
        p=0.8: nucleus = [0.5, 0.3] (first two tokens)
        p=0.9: nucleus = [0.5, 0.3, 0.1] (first three tokens)

    Interview Question:
        "What's the difference between Top-K and Top-P?"
        Top-K uses a fixed number of tokens.
        Top-P adapts to the distribution's confidence.
        Top-P is generally preferred because it's adaptive.
    """
    scaled = logits / temperature
    probs = temperature_scaled_softmax(scaled, temperature=1.0)

    # Sort by probability (descending)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # Find nucleus
    cumulative = np.cumsum(sorted_probs)
    nucleus_size = np.searchsorted(cumulative, p) + 1
    nucleus_indices = sorted_indices[:nucleus_size]
    nucleus_probs = sorted_probs[:nucleus_size]

    # Renormalize
    nucleus_probs = nucleus_probs / np.sum(nucleus_probs)

    # Sample from nucleus
    chosen_idx = np.random.choice(nucleus_size, p=nucleus_probs)
    return nucleus_indices[chosen_idx]


################################################################################
# SECTION 4: STANDALONE FUNCTIONS
################################################################################

def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax Function
    =================

    Converts raw logits into a probability distribution.

    Formula: softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

    Numerically stable version:
        softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))

    Args:
        logits: Raw scores, any shape
        axis: Axis along which to compute softmax (default -1)

    Returns:
        Probability distribution that sums to 1 along the given axis

    Example:
        >>> softmax(np.array([2.0, 1.0, 0.1]))
        array([0.659, 0.242, 0.099])
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


################################################################################
# SECTION 5: TESTING & EXAMPLES
################################################################################

def demonstrate_probability():
    """Demonstrate key probability concepts."""
    print("=" * 70)
    print("PROBABILITY & INFORMATION THEORY DEMONSTRATION")
    print("=" * 70)

    # Categorical distribution (language model output)
    print("\n--- Language Model Output Distribution ---")
    vocab = ["the", "cat", "sat", "on", "mat"]
    logits = np.array([2.0, 1.5, 1.0, 0.5, 0.1])
    dist = Categorical(logits)

    print(f"Vocabulary: {vocab}")
    print(f"Logits: {logits}")
    print(f"Probabilities: {dist.probs.round(3)}")
    print(f"Entropy: {dist.entropy():.3f}")
    print(f"Sample: {vocab[dist.sample()[0]]}")

    # Temperature effects
    print("\n--- Temperature Effects ---")
    for temp in [0.5, 1.0, 2.0]:
        probs = temperature_scaled_softmax(logits, temp)
        print(f"T={temp}: {probs.round(3)}")

    # Cross-entropy loss
    print("\n--- Cross-Entropy Loss ---")
    logits_batch = np.array([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
    targets_batch = np.array([0, 1])
    loss = cross_entropy(logits_batch, targets_batch)
    print(f"Loss: {loss:.4f}")

    # KL Divergence
    print("\n--- KL Divergence ---")
    p = np.array([0.5, 0.3, 0.2])
    q1 = np.array([0.4, 0.35, 0.25])  # Close to p
    q2 = np.array([0.1, 0.1, 0.8])    # Far from p
    print(f"KL(p||q1) = {kl_divergence(p, q1):.4f} (close)")
    print(f"KL(p||q2) = {kl_divergence(p, q2):.4f} (far)")

    # Sampling strategies
    print("\n--- Sampling Strategies ---")
    print(f"Top-K (k=3): {top_k_sampling(logits, k=3)}")
    print(f"Top-P (p=0.8): {top_p_sampling(logits, p=0.8)}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_probability()


################################################################################
# REFERENCES
################################################################################

# [1] Shannon, C. E. (1948). A Mathematical Theory of Communication.
# [2] Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory.
# [3] Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
# [4] Holtzman, A., et al. (2019). The Curious Case of Neural Text Degeneration.
# [5] Fan, A., et al. (2018). Hierarchical Neural Story Generation.

################################################################################
