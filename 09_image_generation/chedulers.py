"""
################################################################################
NOISE SCHEDULERS — CONTROLLING DIFFUSION PROCESS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Noise Schedulers?
    Control how much noise is added at each timestep in diffusion.

Types:
    - Linear: linear schedule
    - Cosine: cosine schedule
    - Scaled linear: power law

Interview Questions:
    Q: "What noise schedule should I use?"
    A: Cosine schedule works best for most applications.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: NOISE SCHEDULERS
################################################################################

class LinearScheduler:
    """
    Linear Noise Scheduler
    ======================

    Linearly increases noise from beta_start to beta_end.
    """

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.betas = np.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)


class CosineScheduler:
    """
    Cosine Noise Scheduler
    ======================

    Uses cosine curve for smoother noise schedule.
    """

    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
        steps = np.arange(num_timesteps + 1) / num_timesteps
        alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        self.betas = np.clip(betas, 0, 0.999)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_schedulers():
    """Demonstrate noise schedulers."""
    print("=" * 70)
    print("NOISE SCHEDULERS DEMONSTRATION")
    print("=" * 70)

    # Linear
    print("\n--- Linear Scheduler ---")
    linear = LinearScheduler(num_timesteps=1000)
    print(f"Beta range: [{linear.betas[0]:.6f}, {linear.betas[-1]:.6f}]")
    print(f"Alpha_bar at t=0: {linear.alpha_bar[0]:.4f}")
    print(f"Alpha_bar at t=500: {linear.alpha_bar[500]:.4f}")
    print(f"Alpha_bar at t=999: {linear.alpha_bar[999]:.4f}")

    # Cosine
    print("\n--- Cosine Scheduler ---")
    cosine = CosineScheduler(num_timesteps=1000)
    print(f"Beta range: [{cosine.betas[0]:.6f}, {cosine.betas[-1]:.6f}]")
    print(f"Alpha_bar at t=0: {cosine.alpha_bar[0]:.4f}")
    print(f"Alpha_bar at t=500: {cosine.alpha_bar[500]:.4f}")
    print(f"Alpha_bar at t=999: {cosine.alpha_bar[999]:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_schedulers()
