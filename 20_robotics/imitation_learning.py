"""
################################################################################
IMITATION LEARNING — LEARNING FROM EXPERT DEMONSTRATIONS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Imitation Learning?
    Learning robot policies from expert demonstrations instead of
    hand-engineered controllers. The robot watches an expert perform
    a task and learns to imitate their behavior.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


################################################################################
# SECTION 1: BEHAVIORAL CLONING
################################################################################

class BehavioralCloning:
    """
    Behavioral Cloning — Supervised imitation from demonstrations.

    Train a policy to predict expert actions from states.

    Formula: L = E[||pi(s) - a_expert||^2]

    Interview Question:
        "What is behavioral cloning?"
        Supervised learning from expert demonstrations: given state s,
        predict the expert's action a*. Simple but suffers from
        distribution shift — errors compound during execution because
        the policy sees states it wasn't trained on.
    """

    def __init__(self, state_dim: int = 10, action_dim: int = 5,
                 lr: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        # Simple linear policy
        self.weights = np.random.randn(state_dim, action_dim) * 0.01
        self.bias = np.zeros(action_dim)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action from state."""
        return state @ self.weights + self.bias

    def train_step(self, state: np.ndarray, expert_action: np.ndarray) -> float:
        """
        One training step.

        Args:
            state: State vector
            expert_action: Expert's action

        Returns:
            Loss value
        """
        predicted = self.predict(state)
        error = predicted - expert_action
        loss = np.mean(error ** 2)

        # Gradient update
        grad = 2 * error / len(error)
        self.weights -= self.lr * np.outer(state, grad)
        self.bias -= self.lr * grad

        return loss


################################################################################
# SECTION 2: DAGGER
################################################################################

class DAgger:
    """
    DAgger — Dataset Aggregation.

    Solves distribution shift by collecting demonstrations on the
    policy's own trajectories.

    Paper: "A Reduction of Imitation Learning and Structured Prediction
            to No-Regret Online Learning" (Ross et al., 2011)

    Step by step:
        1. Train initial policy from expert data
        2. Roll out learned policy, collect states
        3. Ask expert for actions on those states
        4. Aggregate old + new data, retrain
        5. Iterate

    Interview Question:
        "What is DAgger and why use it?"
        DAgger solves distribution shift in behavioral cloning. Instead
        of only training on expert states, it rolls out the learned
        policy, asks the expert what to do at those states, and adds
        this to the training data. This exposes the policy to the
        states it will actually visit.
    """

    def __init__(self, state_dim: int = 10, action_dim: int = 5,
                 n_iterations: int = 5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_iterations = n_iterations
        self.policy = BehavioralCloning(state_dim, action_dim)

    def expert_action(self, state: np.ndarray) -> np.ndarray:
        """Simulate expert providing action for a state."""
        # Simulated expert: optimal linear policy
        return np.random.randn(self.action_dim)

    def collect_states(self, n_states: int = 100) -> np.ndarray:
        """Roll out policy and collect visited states."""
        states = []
        state = np.random.randn(self.state_dim)
        for _ in range(n_states):
            action = self.policy.predict(state)
            # Simulate next state
            state = state + np.random.randn(self.state_dim) * 0.1
            states.append(state.copy())
        return np.array(states)

    def train(self, initial_states: np.ndarray,
              initial_actions: np.ndarray) -> List[float]:
        """
        Full DAgger training loop.

        Args:
            initial_states: Initial expert states
            initial_actions: Initial expert actions

        Returns:
            List of losses per iteration
        """
        all_states = list(initial_states)
        all_actions = list(initial_actions)
        losses = []

        for iteration in range(self.n_iterations):
            # Train on all collected data
            loss = 0
            for s, a in zip(all_states, all_actions):
                loss += self.policy.train_step(s, a)
            losses.append(loss / len(all_states))

            # Collect new states from current policy
            new_states = self.collect_states(50)

            # Get expert actions for new states
            for s in new_states:
                a = self.expert_action(s)
                all_states.append(s)
                all_actions.append(a)

        return losses


################################################################################
# SECTION 3: INVERSE RL
################################################################################

class InverseRL:
    """
    Inverse Reinforcement Learning — Learn reward from demonstrations.

    Instead of learning a policy directly, learn the reward function
    that explains the expert's behavior.

    Interview Question:
        "What is Inverse RL?"
        IRL learns the reward function from demonstrations. The idea:
        the expert is optimizing some unknown reward. By observing their
        behavior, we infer what they're optimizing for. Then we can
        optimize the learned reward with standard RL.
    """

    def __init__(self, state_dim: int = 10, n_features: int = 20):
        self.state_dim = state_dim
        self.n_features = n_features
        self.reward_weights = np.random.randn(n_features) * 0.01

    def features(self, state: np.ndarray) -> np.ndarray:
        """Extract features from state."""
        # Simulate feature extraction
        return np.random.randn(self.n_features)

    def reward(self, state: np.ndarray) -> float:
        """Compute reward for a state."""
        phi = self.features(state)
        return np.dot(self.reward_weights, phi)

    def train(self, expert_trajectories: List[List[np.ndarray]]) -> float:
        """
        Train IRL on expert trajectories.

        Args:
            expert_trajectories: List of state trajectories

        Returns:
            Training loss
        """
        # Simplified max-entropy IRL
        expert_features = np.zeros(self.n_features)
        for traj in expert_trajectories:
            for state in traj:
                expert_features += self.features(state)
        expert_features /= sum(len(t) for t in expert_trajectories)

        # Gradient step to match feature expectations
        self.reward_weights += 0.01 * expert_features
        return np.linalg.norm(expert_features)


################################################################################
# SECTION 4: MULTI-TASK IMITATION
################################################################################

class MultiTaskImitation:
    """
    Multi-Task Imitation Learning.

    Learn a shared policy across multiple tasks, conditioned on task ID.

    Interview Question:
        "How do you do multi-task imitation?"
        Shared backbone with task-specific heads. Input: state + task_id.
        The backbone learns shared features, heads learn task-specific
        actions. Benefits: data efficiency, transfer to new tasks.
    """

    def __init__(self, state_dim: int = 10, action_dim: int = 5,
                 n_tasks: int = 5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_tasks = n_tasks
        # Shared weights
        self.shared_weights = np.random.randn(state_dim, 64) * 0.01
        # Task-specific heads
        self.task_heads = [np.random.randn(64, action_dim) * 0.01
                          for _ in range(n_tasks)]

    def predict(self, state: np.ndarray, task_id: int) -> np.ndarray:
        """Predict action for a specific task."""
        shared = state @ self.shared_weights
        return shared @ self.task_heads[task_id]


################################################################################
# SECTION 5: DEMONSTRATION
################################################################################

def demonstrate_imitation_learning():
    """Demonstrate imitation learning."""
    print("=" * 70)
    print("IMITATION LEARNING DEMONSTRATION")
    print("=" * 70)

    # Behavioral Cloning
    print("\n1. BEHAVIORAL CLONING")
    print("-" * 40)
    bc = BehavioralCloning(state_dim=10, action_dim=5)
    state = np.random.randn(10)
    action = bc.predict(state)
    loss = bc.train_step(state, np.random.randn(5))
    print(f"  State dim: 10, Action dim: 5")
    print(f"  Predicted action: {action[:3]}...")
    print(f"  Training loss: {loss:.4f}")

    # DAgger
    print("\n2. DAGGER")
    print("-" * 40)
    dagger = DAgger(state_dim=10, action_dim=5, n_iterations=3)
    init_states = np.random.randn(50, 10)
    init_actions = np.random.randn(50, 5)
    losses = dagger.train(init_states, init_actions)
    print(f"  Iterations: {len(losses)}")
    for i, l in enumerate(losses):
        print(f"  Iteration {i}: loss = {l:.4f}")

    # Inverse RL
    print("\n3. INVERSE RL")
    print("-" * 40)
    irl = InverseRL(state_dim=10, n_features=20)
    trajectories = [[np.random.randn(10) for _ in range(20)] for _ in range(5)]
    loss = irl.train(trajectories)
    reward = irl.reward(np.random.randn(10))
    print(f"  Trajectories: {len(trajectories)}")
    print(f"  Learned reward: {reward:.4f}")

    # Multi-Task
    print("\n4. MULTI-TASK IMITATION")
    print("-" * 40)
    mt = MultiTaskImitation(state_dim=10, action_dim=5, n_tasks=3)
    state = np.random.randn(10)
    for task in range(3):
        action = mt.predict(state, task)
        print(f"  Task {task}: action = {action[:3]}...")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_imitation_learning()
