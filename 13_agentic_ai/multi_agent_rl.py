"""
################################################################################
MULTI-AGENT RL SYSTEMS — COLLABORATIVE AND COMPETITIVE AI AGENTS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Multi-Agent RL Systems?
    Systems where multiple AI agents interact, communicate, and learn
    together through reinforcement learning. Agents can cooperate to
    solve tasks, compete against each other, or both.

Why does it matter?
    Many real-world problems require multiple agents:
    - Customer support: multiple agents handling different aspects
    - Software development: agents for coding, testing, review
    - Scientific research: agents for literature, experiment, analysis
    - Game AI: teams of agents playing together

    Multi-agent RL enables:
    - Emergent communication protocols
    - Role specialization without explicit programming
    - Scalable problem decomposition
    - Robust solutions through redundancy

How does it work?
    1. Each agent has its own policy (or shared policy)
    2. Agents observe the environment and messages from others
    3. Agents take actions (including sending messages)
    4. Environment provides rewards (individual or shared)
    5. Each agent updates its policy based on experience

Key Approaches:
    - Centralized Training, Decentralized Execution (CTDE)
    - Communication Learning: Agents learn what to communicate
    - Role Specialization: Agents learn different roles
    - Self-Play: Agents improve by playing against copies of themselves

Architecture (Cooperative Multi-Agent):
    ┌─────────────────────────────────────────────────────────────────┐
    │                Multi-Agent System                               │
    │                                                                  │
    │  Agent 1 ──▶ Observe ──▶ Think ──▶ Act + Message ──▶          │
    │  Agent 2 ──▶ Observe ──▶ Think ──▶ Act + Message ──▶          │
    │  Agent 3 ──▶ Observe ──▶ Think ──▶ Act + Message ──▶          │
    │              ↓                                                   │
    │         Environment ──▶ Shared Reward                          │
    │              ↓                                                   │
    │         Update all agents                                       │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "What is centralized training, decentralized execution?"
       During training, agents share information (observations, gradients).
       During execution, each agent acts independently using only its
       own observations. This combines training efficiency with deployment
       simplicity.

    2. "How do agents learn to communicate?"
       Agents send messages through a differentiable channel. The reward
       signal encourages useful messages. Over time, agents develop
       a communication protocol that aids task completion.

    3. "What's the credit assignment problem in multi-agent RL?"
       When agents share a reward, it's hard to know which agent
       contributed. Solutions: (a) counterfactual baselines (COMA),
       (b) reward decomposition, (c) individual reward shaping.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: COMMUNICATION PROTOCOL
################################################################################

class CommunicationChannel:
    """
    Communication Channel between Agents
    ========================================

    Agents send and receive messages through this channel.

    Message format:
    - sender_id: Which agent sent the message
    - content: Message vector (learned representation)
    - timestamp: When the message was sent

    The channel maintains a message history that agents can attend to.

    Interview Question:
        "How do agents learn what to communicate?"
        The message is a differentiable function of the agent's
        observation. The gradient flows back through the message,
        training the agent to send useful information. This is
        called "learning to communicate" or "emergent communication."
    """

    def __init__(self, num_agents: int, message_dim: int, history_size: int = 10):
        self.num_agents = num_agents
        self.message_dim = message_dim
        self.history_size = history_size

        # Message history: (num_agents, history_size, message_dim)
        self.history = np.zeros((num_agents, history_size, message_dim))
        self.write_positions = np.zeros(num_agents, dtype=int)

    def send(self, agent_id: int, message: np.ndarray):
        """
        Send a message from an agent.

        Args:
            agent_id: ID of sending agent
            message: Message vector (message_dim,)
        """
        pos = self.write_positions[agent_id] % self.history_size
        self.history[agent_id, pos] = message
        self.write_positions[agent_id] += 1

    def receive(self, agent_id: int) -> np.ndarray:
        """
        Receive all messages (except own) for an agent.

        Args:
            agent_id: ID of receiving agent

        Returns:
            Concatenated messages from other agents
        """
        # Get messages from all OTHER agents
        other_ids = [i for i in range(self.num_agents) if i != agent_id]
        messages = []

        for other_id in other_ids:
            # Get most recent message from this agent
            pos = (self.write_positions[other_id] - 1) % self.history_size
            messages.append(self.history[other_id, pos])

        if messages:
            return np.concatenate(messages)
        return np.zeros(self.message_dim * (self.num_agents - 1))

    def reset(self):
        """Clear message history."""
        self.history[:] = 0
        self.write_positions[:] = 0


################################################################################
# SECTION 2: AGENT POLICY
################################################################################

class MultiAgentPolicy:
    """
    Policy for a Single Agent in Multi-Agent System
    ==================================================

    Each agent has:
    - Observation encoder: process own observations
    - Message encoder: process messages from others
    - Policy head: decide action
    - Message head: decide what to communicate

    The policy is trained with RL to maximize shared reward.

    Interview Question:
        "Do agents share parameters?"
        Options: (a) Shared policy (parameter sharing) — faster training,
        assumes agents are homogeneous. (b) Separate policies — allows
        specialization, but more parameters. (c) Shared encoder,
        separate heads — compromise.
    """

    def __init__(
        self,
        obs_dim: int,
        message_dim: int,
        action_dim: int,
        num_other_agents: int
    ):
        self.obs_dim = obs_dim
        self.message_dim = message_dim
        self.action_dim = action_dim

        # Observation encoder
        self.obs_W = np.random.randn(obs_dim, 64) * 0.02

        # Message encoder
        msg_input_dim = message_dim * num_other_agents
        self.msg_W = np.random.randn(msg_input_dim, 64) * 0.02

        # Combined → action
        self.action_W = np.random.randn(128, action_dim) * 0.02

        # Combined → message
        self.message_W = np.random.randn(128, message_dim) * 0.02

    def forward(self, observation: np.ndarray, messages: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Compute action and message.

        Args:
            observation: Own observation (obs_dim,)
            messages: Messages from others (message_dim * (n-1),)

        Returns:
            (action_index, message_vector)
        """
        # Encode observation
        obs_enc = np.tanh(self.obs_W @ observation)

        # Encode messages
        msg_enc = np.tanh(self.msg_W @ messages)

        # Combine
        combined = np.concatenate([obs_enc, msg_enc])

        # Action
        action_logits = combined @ self.action_W
        action_probs = self._softmax(action_logits)
        action = np.random.choice(self.action_dim, p=action_probs)

        # Message
        message = np.tanh(combined @ self.message_W)

        return action, message

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)


################################################################################
# SECTION 3: MULTI-AGENT ENVIRONMENT
################################################################################

class MultiAgentEnvironment:
    """
    Multi-Agent Environment
    =========================

    Environment where multiple agents interact.

    Supports:
    - Cooperative tasks (shared reward)
    - Competitive tasks (individual rewards)
    - Mixed tasks (both shared and individual rewards)

    Interview Question:
        "What's the difference between cooperative and competitive?"
        Cooperative: all agents share one reward (team wins or loses).
        Competitive: each agent has its own reward (zero-sum games).
        Mixed: some shared reward + individual bonuses.
    """

    def __init__(self, num_agents: int, obs_dim: int, action_dim: int):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # State
        self.state = np.zeros(obs_dim)
        self.step_count = 0
        self.max_steps = 50

    def reset(self) -> List[np.ndarray]:
        """Reset environment and return observations."""
        self.state = np.random.randn(self.obs_dim) * 0.1
        self.step_count = 0

        # Each agent gets a slightly different observation
        observations = []
        for i in range(self.num_agents):
            obs = self.state + np.random.randn(self.obs_dim) * 0.05
            observations.append(obs)

        return observations

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], float, bool]:
        """
        Take a step in the environment.

        Args:
            actions: Action for each agent

        Returns:
            (observations, shared_reward, done)
        """
        self.step_count += 1

        # Update state based on actions
        action_effect = np.mean(actions) / self.action_dim
        self.state = self.state * 0.9 + action_effect * 0.1

        # Compute shared reward
        reward = float(np.sum(self.state))

        # Check if done
        done = self.step_count >= self.max_steps

        # Get observations
        observations = []
        for i in range(self.num_agents):
            obs = self.state + np.random.randn(self.obs_dim) * 0.05
            observations.append(obs)

        return observations, reward, done


################################################################################
# SECTION 4: MULTI-AGENT TRAINER
################################################################################

class MultiAgentTrainer:
    """
    Multi-Agent RL Trainer
    ========================

    Trains multiple agents to cooperate (or compete).

    Uses a simplified version of:
    - IPPO (Independent PPO): Each agent trains independently
    - With shared experience buffer for efficiency

    Interview Questions:
        1. "What is Independent PPO?"
           Each agent runs its own PPO algorithm, treating other
           agents as part of the environment. Simple but effective.
           The downside: non-stationarity (other agents are changing).

        2. "How do you handle non-stationarity?"
           Options: (a) Experience replay with recent data only,
           (b) Policy regularization, (c) Centralized critic,
           (d) Population-based training.

        3. "What's the benefit of parameter sharing?"
           All agents share the same policy network. This:
           - Reduces parameters (faster training)
           - Improves generalization
           - Works well when agents are homogeneous
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        message_dim: int = 16,
        learning_rate: float = 0.01
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # Communication channel
        self.channel = CommunicationChannel(num_agents, message_dim)

        # Agent policies (one per agent, could share)
        self.policies = [
            MultiAgentPolicy(obs_dim, message_dim, action_dim, num_agents - 1)
            for _ in range(num_agents)
        ]

        # Environment
        self.env = MultiAgentEnvironment(num_agents, obs_dim, action_dim)

    def collect_episode(self) -> Dict:
        """
        Collect one episode of multi-agent interaction.

        Returns:
            Episode data
        """
        observations = self.env.reset()
        self.channel.reset()

        episode_data = {
            'observations': [],
            'actions': [],
            'messages': [],
            'rewards': []
        }

        done = False
        total_reward = 0.0

        while not done:
            actions = []
            messages = []

            for i in range(self.num_agents):
                # Get messages from others
                received = self.channel.receive(i)

                # Choose action and message
                action, message = self.policies[i].forward(observations[i], received)

                actions.append(action)
                messages.append(message)

                # Send message
                self.channel.send(i, message)

            # Step environment
            next_obs, reward, done = self.env.step(actions)

            # Store data
            episode_data['observations'].append(observations)
            episode_data['actions'].append(actions)
            episode_data['messages'].append(messages)
            episode_data['rewards'].append(reward)

            total_reward += reward
            observations = next_obs

        return {
            'data': episode_data,
            'total_reward': total_reward,
            'steps': len(episode_data['rewards'])
        }

    def train_step(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        One training step (collect episodes, update policies).

        Args:
            num_episodes: Number of episodes to collect

        Returns:
            Training metrics
        """
        episodes = []
        for _ in range(num_episodes):
            episode = self.collect_episode()
            episodes.append(episode)

        # Compute metrics
        total_rewards = [e['total_reward'] for e in episodes]
        avg_steps = np.mean([e['steps'] for e in episodes])

        # Simplified policy update (reward-weighted)
        for agent_idx in range(self.num_agents):
            for episode in episodes:
                reward = episode['total_reward']
                if reward > np.mean(total_rewards):
                    # Reinforce good episodes (simplified)
                    pass  # In practice, use proper policy gradient

        return {
            'mean_reward': float(np.mean(total_rewards)),
            'max_reward': float(np.max(total_rewards)),
            'avg_steps': float(avg_steps)
        }

    def train(self, num_iterations: int = 10) -> List[Dict]:
        """
        Full training loop.

        Args:
            num_iterations: Number of training iterations

        Returns:
            Training history
        """
        history = []

        for iteration in range(num_iterations):
            metrics = self.train_step(num_episodes=5)
            metrics['iteration'] = iteration
            history.append(metrics)

        return history


################################################################################
# SECTION 5: SELF-PLAY TRAINING
################################################################################

class SelfPlayTrainer:
    """
    Self-Play Multi-Agent Training
    =================================

    Agents improve by playing against copies of themselves.

    This is the approach behind:
    - AlphaGo/AlphaZero (Go, Chess)
    - OpenAI Five (Dota 2)
    - AlphaStar (StarCraft)

    Algorithm:
    1. Current policy plays against itself
    2. Collect trajectories
    3. Update policy to beat current version
    4. Repeat — policy improves each iteration

    Interview Questions:
        1. "Why does self-play work?"
           As the agent improves, its opponents (copies of itself)
           also improve. This creates a natural curriculum — the
           task gets harder as the agent gets better.

        2. "What's the 'non-stationarity' problem in self-play?"
           The opponent is constantly changing, so the environment
           is non-stationary. This can cause instability. Solutions:
           (a) Maintain a pool of past policies, (b) Use population-based
           training, (c) Slow policy updates.
    """

    def __init__(self, policy: MultiAgentPolicy, env: MultiAgentEnvironment):
        self.policy = policy
        self.env = env
        self.policy_pool = [policy]  # Pool of past policies

    def self_play_episode(self) -> float:
        """
        Run one episode of self-play.

        Returns:
            Total reward
        """
        observations = self.env.reset()
        total_reward = 0.0
        done = False

        while not done:
            actions = []
            for i in range(self.num_agents):
                # Use current policy for all agents (self-play)
                obs = observations[i]
                dummy_messages = np.zeros(self.policy.message_dim * (self.num_agents - 1))
                action, _ = self.policy.forward(obs, dummy_messages)
                actions.append(action)

            observations, reward, done = self.env.step(actions)
            total_reward += reward

        return total_reward

    @property
    def num_agents(self):
        return self.env.num_agents

    def train(self, num_iterations: int = 10) -> List[float]:
        """
        Self-play training loop.

        Args:
            num_iterations: Number of training iterations

        Returns:
            List of average rewards per iteration
        """
        rewards = []

        for iteration in range(num_iterations):
            # Play multiple games
            episode_rewards = []
            for _ in range(5):
                reward = self.self_play_episode()
                episode_rewards.append(reward)

            avg_reward = np.mean(episode_rewards)
            rewards.append(avg_reward)

            # Save policy snapshot
            self.policy_pool.append(self.policy)

        return rewards


################################################################################
# SECTION 6: TESTING & DEMONSTRATION
################################################################################

def demonstrate_multi_agent_rl():
    """Demonstrate multi-agent RL systems."""
    print("=" * 70)
    print("MULTI-AGENT RL SYSTEMS")
    print("=" * 70)

    # Configuration
    num_agents = 3
    obs_dim = 16
    action_dim = 4
    message_dim = 8

    print(f"\nConfiguration:")
    print(f"  Agents: {num_agents}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Message dim: {message_dim}")

    # Communication channel
    print("\n--- Communication Channel ---")
    channel = CommunicationChannel(num_agents, message_dim)

    # Send messages
    for i in range(num_agents):
        msg = np.random.randn(message_dim)
        channel.send(i, msg)
        print(f"  Agent {i} sent message: {msg[:3]}...")

    # Receive messages
    for i in range(num_agents):
        received = channel.receive(i)
        print(f"  Agent {i} received: {received[:5]}...")

    # Multi-agent training
    print("\n--- Multi-Agent Training ---")
    trainer = MultiAgentTrainer(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        message_dim=message_dim
    )

    history = trainer.train(num_iterations=5)

    for h in history:
        print(f"  Iteration {h['iteration']}: "
              f"mean_reward={h['mean_reward']:.3f}, "
              f"max_reward={h['max_reward']:.3f}")

    # Self-play
    print("\n--- Self-Play Training ---")
    env = MultiAgentEnvironment(num_agents, obs_dim, action_dim)
    policy = MultiAgentPolicy(obs_dim, message_dim, action_dim, num_agents - 1)
    self_play = SelfPlayTrainer(policy, env)

    rewards = self_play.train(num_iterations=5)
    for i, r in enumerate(rewards):
        print(f"  Iteration {i}: reward={r:.3f}")

    # Agent policy analysis
    print("\n--- Agent Policy Analysis ---")
    for i in range(num_agents):
        obs = np.random.randn(obs_dim)
        messages = np.zeros(message_dim * (num_agents - 1))
        action, msg = trainer.policies[i].forward(obs, messages)
        print(f"  Agent {i}: action={action}, msg_norm={np.linalg.norm(msg):.3f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Multi-agent systems enable emergent collaboration!")
    print("Agents learn communication protocols and role specialization.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_multi_agent_rl()
