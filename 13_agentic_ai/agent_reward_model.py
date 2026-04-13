"""
################################################################################
AGENT REWARD MODELING — TRAINING REWARD MODELS FOR AGENT TASKS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Agent Reward Modeling?
    Training reward models specifically for agent tasks — evaluating
    not just final answers, but the quality of each step in an
    agent's reasoning and tool-use trajectory.

Why does it matter?
    Standard reward models evaluate single responses. Agent tasks
    require evaluating multi-step trajectories:

    - Did the agent use the right tools?
    - Was each reasoning step sound?
    - Was the final answer correct?
    - Was the process efficient?

    Agent reward models provide:
    - Step-level feedback (process rewards)
    - Trajectory-level evaluation (outcome rewards)
    - Tool-use quality assessment
    - Efficiency scoring

How does it work?
    1. Collect agent trajectories (thoughts, tool calls, answers)
    2. Label trajectories with quality scores
    3. Train reward model to predict quality from trajectory
    4. Use reward model to guide RL training

Key Types:
    - Outcome Reward Model (ORM): Score final answer only
    - Process Reward Model (PRM): Score each reasoning step
    - Tool Reward Model (TRM): Score tool usage quality

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Agent Reward Model                               │
    │                                                                  │
    │  Trajectory: [Step1, Step2, ..., StepN, Answer]                │
    │       ↓                                                          │
    │  Step Encoder: encode each step                                │
    │       ↓                                                          │
    │  Trajectory Encoder: aggregate step representations            │
    │       ↓                                                          │
    │  Reward Head: predict quality score                            │
    │       ↓                                                          │
    │  Reward: scalar quality score                                  │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "What's the difference between ORM and PRM?"
       ORM (Outcome Reward Model) only scores the final answer.
       PRM (Process Reward Model) scores each reasoning step.
       PRM is better for debugging and credit assignment.

    2. "How do you collect training data for agent rewards?"
       (a) Human annotation of trajectories,
       (b) Automatic rewards (correctness, format),
       (c) LLM-as-judge for step quality,
       (d) Comparison data (trajectory A > B).

    3. "Why is reward modeling hard for agents?"
       Agents have long trajectories with many steps. Credit
       assignment is difficult — which step caused success/failure?
       Also, the same action can be good or bad depending on context.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: TRAJECTORY REPRESENTATION
################################################################################

@dataclass
class AgentStep:
    """
    Single step in an agent trajectory.

    Attributes:
        step_type: 'thought', 'tool_call', 'observation', 'answer'
        content: Text content of the step
        tool_name: Tool used (if tool_call)
        success: Whether the step succeeded
        embedding: Encoded representation
    """
    step_type: str
    content: str
    tool_name: Optional[str] = None
    success: bool = True
    embedding: Optional[np.ndarray] = None


class AgentTrajectory:
    """
    Complete agent trajectory.

    A trajectory is a sequence of steps from problem to solution:
    [Thought → Tool Call → Observation → ... → Answer]

    Interview Question:
        "What information should be in an agent trajectory?"
        Everything the agent did: thoughts (reasoning), tool calls
        (actions), observations (tool results), and the final answer.
        This enables both outcome and process evaluation.
    """

    def __init__(self):
        self.steps: List[AgentStep] = []
        self.final_answer: str = ""
        self.total_reward: float = 0.0

    def add_step(self, step: AgentStep):
        """Add a step to the trajectory."""
        self.steps.append(step)

    def to_feature_vector(self, embed_dim: int = 32) -> np.ndarray:
        """
        Convert trajectory to feature vector for reward model.

        Features:
        - Mean step embedding
        - Number of steps
        - Tool success rate
        - Step type distribution
        """
        if not self.steps:
            return np.zeros(embed_dim)

        # Step embeddings
        embeddings = [s.embedding for s in self.steps if s.embedding is not None]
        if embeddings:
            mean_embed = np.mean(embeddings, axis=0)
        else:
            mean_embed = np.zeros(embed_dim)

        # Statistics
        num_steps = len(self.steps)
        tool_calls = [s for s in self.steps if s.step_type == 'tool_call']
        success_rate = np.mean([s.success for s in tool_calls]) if tool_calls else 1.0

        # Step type distribution
        type_counts = {}
        for s in self.steps:
            type_counts[s.step_type] = type_counts.get(s.step_type, 0) + 1

        features = np.concatenate([
            mean_embed,
            [num_steps / 10.0],  # Normalize
            [success_rate],
            [type_counts.get('thought', 0) / max(num_steps, 1)],
            [type_counts.get('tool_call', 0) / max(num_steps, 1)],
        ])

        return features


################################################################################
# SECTION 2: OUTCOME REWARD MODEL
################################################################################

class OutcomeRewardModel:
    """
    Outcome Reward Model (ORM)
    ===========================

    Scores trajectories based on final outcome only.

    Training data: (trajectory, correct/incorrect) pairs
    Output: probability of correct outcome

    Simple but effective for tasks with clear success criteria.

    Interview Questions:
        1. "When should I use ORM vs PRM?"
           ORM: when the final answer is all that matters (math,
           factual QA). PRM: when process matters (reasoning tasks,
           debugging, multi-step planning).

        2. "What are the limitations of ORM?"
           Credit assignment: ORM can't tell which steps were good
           or bad. A correct answer could come from a flawed process.
           Also, ORM provides no feedback for intermediate steps.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Reward network
        self.W1 = np.random.randn(feature_dim, hidden_dim) * 0.02
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.02
        self.b2 = np.zeros(1)

    def predict(self, trajectory: AgentTrajectory) -> float:
        """
        Predict outcome reward for a trajectory.

        Args:
            trajectory: Agent trajectory

        Returns:
            Predicted reward (0 to 1)
        """
        features = trajectory.to_feature_vector(self.feature_dim)

        # Forward pass
        h = np.tanh(self.W1 @ features + self.b1)
        logit = float(h @ self.W2 + self.b2)

        # Sigmoid
        reward = 1.0 / (1.0 + np.exp(-logit))

        return reward

    def train_step(
        self,
        trajectory: AgentTrajectory,
        label: float,  # 1.0 = correct, 0.0 = incorrect
        learning_rate: float = 0.01
    ) -> float:
        """
        One training step.

        Args:
            trajectory: Training trajectory
            label: Ground truth label (0 or 1)
            learning_rate: Learning rate

        Returns:
            Loss value
        """
        features = trajectory.to_feature_vector(self.feature_dim)

        # Forward
        h = np.tanh(self.W1 @ features + self.b1)
        logit = float(h @ self.W2 + self.b2)
        pred = 1.0 / (1.0 + np.exp(-logit))

        # Binary cross-entropy loss
        loss = -(label * np.log(pred + 1e-8) + (1 - label) * np.log(1 - pred + 1e-8))

        # Backward (simplified)
        grad_logit = pred - label
        grad_h = grad_logit * self.W2.flatten()

        # Update
        self.W2 -= learning_rate * grad_logit * h.reshape(-1, 1)
        self.b2 -= learning_rate * grad_logit

        tanh_grad = 1 - h ** 2
        self.W1 -= learning_rate * np.outer(features, tanh_grad * grad_h)
        self.b1 -= learning_rate * tanh_grad * grad_h

        return float(loss)


################################################################################
# SECTION 3: PROCESS REWARD MODEL
################################################################################

class ProcessRewardModel:
    """
    Process Reward Model (PRM)
    ============================

    Scores each step in the trajectory, not just the final outcome.

    This provides:
    - Step-level feedback for debugging
    - Better credit assignment
    - Intermediate rewards for RL training

    Training data: (step, correct/incorrect) labels for each step.

    Interview Questions:
        1. "How do you train a PRM?"
           Label each step as correct/incorrect (human or automated).
           Train a classifier on step features. At inference, score
           each step and aggregate for trajectory quality.

        2. "What's the advantage of PRM over ORM?"
           PRM can identify WHERE in the trajectory things went wrong.
           This enables: (a) better RL training with step-level rewards,
           (b) debugging agent failures, (c) credit assignment.

        3. "Is PRM always better than ORM?"
           Not always. PRM requires step-level labels (more expensive).
           For simple tasks, ORM is sufficient. For complex reasoning,
           PRM significantly improves performance.
    """

    def __init__(self, step_dim: int, hidden_dim: int = 32):
        self.step_dim = step_dim
        self.hidden_dim = hidden_dim

        # Step scoring network
        self.step_W1 = np.random.randn(step_dim, hidden_dim) * 0.02
        self.step_W2 = np.random.randn(hidden_dim, 1) * 0.02

        # Trajectory aggregation
        self.agg_W = np.random.randn(hidden_dim, 1) * 0.02

    def score_step(self, step: AgentStep) -> float:
        """
        Score a single step.

        Args:
            step: Agent step

        Returns:
            Step quality score (0 to 1)
        """
        if step.embedding is None:
            return 0.5  # Unknown

        h = np.tanh(self.step_W1 @ step.embedding)
        logit = float(h @ self.step_W2)

        return 1.0 / (1.0 + np.exp(-logit))

    def score_trajectory(self, trajectory: AgentTrajectory) -> Tuple[List[float], float]:
        """
        Score all steps and aggregate.

        Args:
            trajectory: Agent trajectory

        Returns:
            (step_scores, overall_score)
        """
        step_scores = []
        step_embeddings = []

        for step in trajectory.steps:
            score = self.score_step(step)
            step_scores.append(score)

            if step.embedding is not None:
                h = np.tanh(self.step_W1 @ step.embedding)
                step_embeddings.append(h)

        # Aggregate: mean of step representations
        if step_embeddings:
            mean_embed = np.mean(step_embeddings, axis=0)
            overall_logit = float(mean_embed @ self.agg_W)
            overall_score = 1.0 / (1.0 + np.exp(-overall_logit))
        else:
            overall_score = 0.5

        return step_scores, overall_score

    def train_step(
        self,
        trajectory: AgentTrajectory,
        step_labels: List[float],
        learning_rate: float = 0.01
    ) -> float:
        """
        Train on a trajectory with step-level labels.

        Args:
            trajectory: Agent trajectory
            step_labels: Correctness label for each step
            learning_rate: Learning rate

        Returns:
            Average loss
        """
        total_loss = 0.0

        for step, label in zip(trajectory.steps, step_labels):
            if step.embedding is None:
                continue

            score = self.score_step(step)

            # Binary cross-entropy
            loss = -(label * np.log(score + 1e-8) + (1 - label) * np.log(1 - score + 1e-8))
            total_loss += loss

            # Simplified gradient update
            grad = score - label
            h = np.tanh(self.step_W1 @ step.embedding)
            self.step_W2 -= learning_rate * grad * h.reshape(-1, 1)
            self.step_W1 -= learning_rate * np.outer(
                step.embedding,
                (1 - h ** 2).flatten() * float(grad * self.step_W2.flatten())
            )

        return total_loss / max(len(step_labels), 1)


################################################################################
# SECTION 4: TOOL-USE REWARD MODEL
################################################################################

class ToolUseRewardModel:
    """
    Tool-Use Reward Model
    =======================

    Specialized reward model for evaluating tool usage quality.

    Criteria:
    1. Relevance: Was the tool appropriate for the task?
    2. Correctness: Were the arguments correct?
    3. Efficiency: Were unnecessary tool calls avoided?
    4. Success: Did the tool call succeed?

    Interview Questions:
        1. "How do you evaluate tool usage quality?"
           Score each tool call on: (a) relevance to the current
           reasoning step, (b) correctness of arguments, (c) whether
           the result was useful, (d) efficiency (fewer calls = better).

        2. "What rewards work for tool-use RL?"
           Combine: (a) outcome reward (final answer correct),
           (b) tool relevance reward (right tool chosen),
           (c) efficiency reward (penalize excessive calls),
           (d) format reward (correct tool-call syntax).
    """

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim

        # Tool relevance scorer
        self.relevance_W = np.random.randn(feature_dim, 1) * 0.02

        # Efficiency penalty coefficient
        self.efficiency_coeff = 0.01

    def score_tool_call(
        self,
        tool_call_embedding: np.ndarray,
        context_embedding: np.ndarray
    ) -> Dict[str, float]:
        """
        Score a single tool call.

        Args:
            tool_call_embedding: Encoded tool call
            context_embedding: Encoded context (reasoning so far)

        Returns:
            Dictionary of scores
        """
        # Relevance: how well does tool match context?
        relevance = float(np.dot(tool_call_embedding, context_embedding))

        # Correctness (simplified)
        correctness = 1.0 if np.linalg.norm(tool_call_embedding) > 0.1 else 0.0

        return {
            'relevance': relevance,
            'correctness': correctness,
            'combined': 0.5 * relevance + 0.5 * correctness
        }

    def score_trajectory(self, trajectory: AgentTrajectory) -> float:
        """
        Score tool usage in a trajectory.

        Args:
            trajectory: Agent trajectory

        Returns:
            Tool-use quality score
        """
        tool_calls = [s for s in trajectory.steps if s.step_type == 'tool_call']

        if not tool_calls:
            return 1.0  # No tools needed = fine

        # Success rate
        success_rate = np.mean([s.success for s in tool_calls])

        # Efficiency (penalize too many calls)
        efficiency = max(0, 1.0 - self.efficiency_coeff * len(tool_calls))

        return 0.7 * success_rate + 0.3 * efficiency


################################################################################
# SECTION 5: COMBINED AGENT REWARD MODEL
################################################################################

class AgentRewardModel:
    """
    Combined Agent Reward Model
    ==============================

    Combines outcome, process, and tool-use rewards into a single
    quality score for agent trajectories.

    Final reward = w1 * ORM + w2 * PRM + w3 * TRM

    Interview Questions:
        1. "How do you combine different reward signals?"
           Weighted sum with tunable weights. Outcome reward is
           usually most important. Process and tool rewards provide
           shaping signals that guide learning.

        2. "Can reward models be gamed?"
           Yes — agents can learn to satisfy the reward model without
           actually solving the task (reward hacking). Solutions:
           (a) Ensemble of reward models, (b) KL penalty to reference,
           (c) Regular human evaluation.
    """

    def __init__(self, feature_dim: int):
        self.orm = OutcomeRewardModel(feature_dim)
        self.prm = ProcessRewardModel(feature_dim)
        self.trm = ToolUseRewardModel(feature_dim)

        # Weights
        self.w_outcome = 0.5
        self.w_process = 0.3
        self.w_tool = 0.2

    def score(self, trajectory: AgentTrajectory) -> Dict[str, float]:
        """
        Score a trajectory with all reward models.

        Args:
            trajectory: Agent trajectory

        Returns:
            Dictionary of reward scores
        """
        # Outcome reward
        outcome_score = self.orm.predict(trajectory)

        # Process reward
        step_scores, process_score = self.prm.score_trajectory(trajectory)

        # Tool-use reward
        tool_score = self.trm.score_trajectory(trajectory)

        # Combined
        combined = (
            self.w_outcome * outcome_score +
            self.w_process * process_score +
            self.w_tool * tool_score
        )

        return {
            'outcome': outcome_score,
            'process': process_score,
            'tool_use': tool_score,
            'combined': combined,
            'step_scores': step_scores
        }

    def train(
        self,
        trajectories: List[AgentTrajectory],
        labels: List[float],
        num_epochs: int = 10
    ):
        """
        Train all reward models.

        Args:
            trajectories: Training trajectories
            labels: Quality labels (0 to 1)
            num_epochs: Training epochs
        """
        for epoch in range(num_epochs):
            total_loss = 0.0

            for traj, label in zip(trajectories, labels):
                # Train ORM
                loss = self.orm.train_step(traj, label)
                total_loss += loss

            avg_loss = total_loss / len(trajectories)

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}")


################################################################################
# SECTION 6: TESTING & DEMONSTRATION
################################################################################

def demonstrate_agent_reward_model():
    """Demonstrate agent reward modeling."""
    print("=" * 70)
    print("AGENT REWARD MODELING")
    print("=" * 70)

    feature_dim = 32

    # Create reward model
    reward_model = AgentRewardModel(feature_dim)

    # Create sample trajectory
    print("\n--- Sample Trajectory ---")
    trajectory = AgentTrajectory()

    steps = [
        AgentStep("thought", "I need to search for Python information"),
        AgentStep("tool_call", "search('Python')", tool_name="search", success=True),
        AgentStep("observation", "Python is a programming language"),
        AgentStep("thought", "I have enough information to answer"),
        AgentStep("answer", "Python is a high-level programming language"),
    ]

    for step in steps:
        step.embedding = np.random.randn(feature_dim)
        trajectory.add_step(step)

    print(f"  Steps: {len(trajectory.steps)}")
    print(f"  Types: {[s.step_type for s in trajectory.steps]}")

    # Score trajectory
    print("\n--- Scoring Trajectory ---")
    scores = reward_model.score(trajectory)

    print(f"  Outcome score: {scores['outcome']:.3f}")
    print(f"  Process score: {scores['process']:.3f}")
    print(f"  Tool-use score: {scores['tool_use']:.3f}")
    print(f"  Combined score: {scores['combined']:.3f}")

    print(f"\n  Step scores:")
    for i, (step, score) in enumerate(zip(trajectory.steps, scores['step_scores'])):
        print(f"    Step {i} ({step.step_type}): {score:.3f}")

    # Training
    print("\n--- Training Reward Model ---")
    trajectories = [trajectory]
    labels = [0.8]  # Good trajectory

    reward_model.train(trajectories, labels, num_epochs=10)

    # Re-score after training
    print("\n--- Re-scoring After Training ---")
    scores_after = reward_model.score(trajectory)
    print(f"  Combined score: {scores_after['combined']:.3f}")

    # Tool-use scoring
    print("\n--- Tool-Use Reward Model ---")
    tool_score = reward_model.trm.score_trajectory(trajectory)
    print(f"  Tool-use quality: {tool_score:.3f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Agent reward models score trajectories at multiple levels!")
    print("PRM (process) + ORM (outcome) + TRM (tool use) = comprehensive evaluation.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_agent_reward_model()
