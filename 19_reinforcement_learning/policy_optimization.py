"""
################################################################################
POLICY OPTIMIZATION — RL FOR LANGUAGE MODELS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Policy Optimization?
    Optimizing language model outputs using reinforcement learning.

Key Methods:
    - PPO: Proximal Policy Optimization
    - DPO: Direct Preference Optimization
    - GRPO: Group Relative Policy Optimization

Interview Questions:
    Q: "Why use RL for language models?"
    A: To align with human preferences. RL optimizes for
       reward signals that capture helpfulness, harmlessness, honesty.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: POLICY OPTIMIZER
################################################################################

class PolicyOptimizer:
    """
    Policy Optimizer
    ================

    Optimizes policy using policy gradient methods.

    Interview Questions:
        Q: "What is policy gradient?"
        A: Update policy in direction that increases expected reward.
           ∇J = E[∇log π(a|s) × A(s,a)]
    """

    def __init__(self, learning_rate: float = 1e-4):
        self.learning_rate = learning_rate

    def policy_gradient(
        self,
        logprobs: np.ndarray,
        advantages: np.ndarray
    ) -> float:
        """
        Compute policy gradient loss.

        Args:
            logprobs: Log probabilities of actions
            advantages: Advantage estimates

        Returns:
            loss: Policy gradient loss
        """
        return -np.mean(logprobs * advantages)


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_policy_optimization():
    """Demonstrate policy optimization."""
    print("=" * 70)
    print("POLICY OPTIMIZATION DEMONSTRATION")
    print("=" * 70)

    optimizer = PolicyOptimizer()
    logprobs = np.random.randn(8)
    advantages = np.random.randn(8)
    loss = optimizer.policy_gradient(logprobs, advantages)
    print(f"Policy gradient loss: {loss:.4f}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_policy_optimization()
