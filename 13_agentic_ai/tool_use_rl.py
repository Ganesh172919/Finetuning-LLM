"""
################################################################################
TOOL-USE RL TRAINING — TRAINING AGENTS TO USE TOOLS VIA REINFORCEMENT LEARNING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Tool-Use RL Training?
    Training language model agents to correctly select and use external tools
    (APIs, calculators, search engines, code interpreters) through reinforcement
    learning. Instead of supervised fine-tuning on tool-use examples, the agent
    learns by trying tools and receiving rewards for successful task completion.

Why does it matter?
    Supervised tool-use training requires:
    - Large datasets of correct tool usage (expensive to collect)
    - Human annotation of which tool to use when
    - Fixed tool schemas (can't adapt to new tools)

    RL-based tool-use training enables:
    - Learning from trial and error (no labeled data needed)
    - Discovering novel tool-use strategies
    - Adapting to new tools without retraining
    - Optimizing for task completion, not just tool-call format

How does it work?
    1. Agent generates a response that may include tool calls
    2. Tools are executed, results returned to agent
    3. Agent continues reasoning with tool results
    4. Final response is evaluated (correctness, completeness)
    5. RL algorithm (GRPO/PPO) updates policy based on reward

Key Innovation (DeepSeek-R1 style):
    - Use GRPO (no critic model needed)
    - Rule-based rewards: correct answer, proper format, tool usage
    - Group-relative advantages: compare N attempts per prompt
    - Tool calls are part of the generation, not separate

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Tool-Use RL Training Loop                       │
    │                                                                  │
    │  Prompt ──▶ Agent generates response + tool calls              │
    │                  ↓                                               │
    │          Execute tools, get results                            │
    │                  ↓                                               │
    │          Agent generates final answer                          │
    │                  ↓                                               │
    │          Compute reward (correctness, tool efficiency)         │
    │                  ↓                                               │
    │          GRPO update: prefer high-reward trajectories          │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "How do you train an agent to use tools with RL?"
       Generate multiple tool-use trajectories per prompt, score them
       by task completion, use GRPO to update the policy toward
       higher-scoring trajectories. No labeled tool-use data needed.

    2. "What rewards work for tool-use training?"
       (a) Outcome reward: did the final answer match?,
       (b) Format reward: were tool calls syntactically correct?,
       (c) Efficiency reward: fewer tool calls is better,
       (d) Step reward: each tool call that produces useful info.

    3. "How does this differ from Toolformer?"
       Toolformer uses self-supervised learning (mask tool outputs,
       predict them). RL-based training optimizes for task completion,
       which can discover non-obvious tool-use strategies.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass, field
import math

################################################################################
# SECTION 1: TOOL ENVIRONMENT
################################################################################

@dataclass
class ToolCall:
    """
    Represents a single tool call.

    Attributes:
        tool_name: Name of the tool to call
        arguments: Arguments to pass to the tool
        result: Result from the tool execution
        success: Whether the call succeeded
        cost: Computational cost of the call
    """
    tool_name: str
    arguments: Dict
    result: Optional[str] = None
    success: bool = False
    cost: float = 1.0


class ToolEnvironment:
    """
    Tool Execution Environment
    ============================

    Simulates tool execution for agent training.

    Tools available:
    - search: Query a knowledge base
    - calculator: Perform arithmetic
    - code_runner: Execute code snippets
    - retriever: Retrieve relevant documents

    The environment tracks:
    - Tool calls made
    - Results returned
    - Total cost (efficiency metric)

    Interview Question:
        "How do you simulate tools for training?"
        Options: (a) Use actual tools (expensive but realistic),
        (b) Use a tool simulator (cheaper, may miss edge cases),
        (c) Use a language model as tool oracle (flexible).
    """

    def __init__(self):
        # Tool registry
        self.tools = {
            'search': self._search,
            'calculator': self._calculator,
            'code_runner': self._code_runner,
            'retriever': self._retriever,
        }

        # Knowledge base for search
        self.knowledge_base = {
            'python': 'Python is a high-level programming language.',
            'transformer': 'Transformers use self-attention mechanisms.',
            'grpo': 'GRPO is Group Relative Policy Optimization.',
            'rlhf': 'RLHF is Reinforcement Learning from Human Feedback.',
        }

        # Execution history
        self.history: List[ToolCall] = []
        self.total_cost = 0.0

    def reset(self):
        """Reset environment for new episode."""
        self.history = []
        self.total_cost = 0.0

    def execute(self, tool_name: str, arguments: Dict) -> ToolCall:
        """
        Execute a tool call.

        Args:
            tool_name: Name of tool to call
            arguments: Tool arguments

        Returns:
            ToolCall with result
        """
        call = ToolCall(tool_name=tool_name, arguments=arguments)

        if tool_name in self.tools:
            try:
                result = self.tools[tool_name](**arguments)
                call.result = str(result)
                call.success = True
            except Exception as e:
                call.result = f"Error: {e}"
                call.success = False
        else:
            call.result = f"Unknown tool: {tool_name}"
            call.success = False

        self.history.append(call)
        self.total_cost += call.cost

        return call

    def _search(self, query: str) -> str:
        """Simulate search tool."""
        query_lower = query.lower()
        results = []
        for key, value in self.knowledge_base.items():
            if key in query_lower:
                results.append(value)
        return '; '.join(results) if results else 'No results found.'

    def _calculator(self, expression: str) -> str:
        """Simulate calculator tool."""
        try:
            # Safely evaluate arithmetic
            allowed = set('0123456789+-*/.() ')
            if all(c in allowed for c in expression):
                return str(eval(expression))
            return 'Invalid expression'
        except:
            return 'Calculation error'

    def _code_runner(self, code: str) -> str:
        """Simulate code execution."""
        # Simplified: just return success
        return f"Code executed: {len(code)} chars"

    def _retriever(self, query: str, top_k: int = 3) -> str:
        """Simulate document retrieval."""
        return f"Retrieved {top_k} documents for: {query}"


################################################################################
# SECTION 2: TOOL-USE POLICY
################################################################################

class ToolUsePolicy:
    """
    Tool-Use Policy Network
    =========================

    A policy that decides:
    1. Whether to use a tool or answer directly
    2. Which tool to use
    3. What arguments to pass

    The policy is trained with RL to maximize task completion.

    Architecture:
        Input: prompt + conversation history
        → Transformer encoder
        → Tool selection head (which tool)
        → Argument generation head (what args)
        → Answer head (if answering directly)

    Interview Question:
        "How does the policy decide when to use a tool?"
        The policy learns a threshold: if the model is confident
        in its answer, it answers directly. If uncertain, it calls
        a tool. This is learned from RL rewards.
    """

    def __init__(self, vocab_size: int, d_model: int, num_tools: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_tools = num_tools

        # Embedding
        self.embed = np.random.randn(vocab_size, d_model) * 0.02

        # Tool selection head
        self.tool_head_W = np.random.randn(d_model, num_tools) * 0.02
        self.tool_head_b = np.zeros(num_tools)

        # Answer head
        self.answer_head_W = np.random.randn(d_model, vocab_size) * 0.02

        # Value head (for MCTS/search)
        self.value_head_W = np.random.randn(d_model, 1) * 0.02

    def select_tool(self, state_embedding: np.ndarray) -> Tuple[int, float]:
        """
        Select a tool based on current state.

        Args:
            state_embedding: Current state representation (d_model,)

        Returns:
            (tool_index, confidence)
        """
        logits = state_embedding @ self.tool_head_W + self.tool_head_b
        probs = self._softmax(logits)

        # Sample tool
        tool_idx = np.random.choice(self.num_tools, p=probs)

        return tool_idx, float(probs[tool_idx])

    def generate_answer(self, state_embedding: np.ndarray) -> np.ndarray:
        """Generate answer logits."""
        return state_embedding @ self.answer_head_W

    def estimate_value(self, state_embedding: np.ndarray) -> float:
        """Estimate state value (for search/planning)."""
        return float(state_embedding @ self.value_head_W)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)


################################################################################
# SECTION 3: GRPO TRAINER FOR TOOL USE
################################################################################

class GRPOToolTrainer:
    """
    GRPO Trainer for Tool-Use Agents
    ===================================

    Trains tool-use policy using Group Relative Policy Optimization.

    Key idea: Generate N tool-use trajectories per prompt,
    score them by task completion, update policy to prefer
    high-scoring trajectories.

    Algorithm:
    1. For each prompt, generate N trajectories with tool calls
    2. Execute tools, get final answers
    3. Score each trajectory (reward)
    4. Normalize rewards within group (mean=0, std=1)
    5. Update policy: increase prob of high-reward trajectories

    Interview Questions:
        1. "Why GRPO for tool-use training?"
           GRPO doesn't need a critic model (saves 50% memory).
           Tool-use rewards are often binary (correct/incorrect),
           making group-relative comparison natural.

        2. "How do you handle tool failures?"
           Tool failures get low rewards, teaching the agent to
           avoid that tool/argument combination. The agent learns
           robust tool-use strategies.

        3. "What's the reward structure?"
           Multi-component: (a) Final answer correctness (main),
           (b) Tool call format (bonus), (c) Efficiency (fewer
           calls = higher reward), (d) Tool relevance.
    """

    def __init__(
        self,
        policy: ToolUsePolicy,
        env: ToolEnvironment,
        num_samples: int = 4,
        learning_rate: float = 0.01,
        kl_coeff: float = 0.1
    ):
        self.policy = policy
        self.env = env
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.kl_coeff = kl_coeff

        # Reference policy (for KL penalty)
        self.ref_tool_W = policy.tool_head_W.copy()
        self.ref_tool_b = policy.tool_head_b.copy()

        # Training history
        self.history = []

    def compute_reward(
        self,
        trajectory: List[ToolCall],
        final_answer: str,
        ground_truth: str
    ) -> float:
        """
        Compute reward for a tool-use trajectory.

        Reward components:
        1. Answer correctness: 1.0 if correct, 0.0 if wrong
        2. Format bonus: 0.1 if tool calls are well-formatted
        3. Efficiency: -0.01 per tool call (penalize excessive calls)
        4. Tool success: 0.05 per successful tool call

        Args:
            trajectory: List of tool calls made
            final_answer: Agent's final answer
            ground_truth: Expected answer

        Returns:
            Scalar reward
        """
        reward = 0.0

        # Answer correctness (main reward)
        if self._answers_match(final_answer, ground_truth):
            reward += 1.0

        # Format bonus
        if all(call.success for call in trajectory):
            reward += 0.1

        # Efficiency penalty
        reward -= 0.01 * len(trajectory)

        # Tool success bonus
        for call in trajectory:
            if call.success and call.result:
                reward += 0.05

        return reward

    def _answers_match(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        # Simplified comparison
        return ground_truth.lower().strip() in predicted.lower().strip()

    def generate_trajectory(
        self,
        prompt: str,
        ground_truth: str,
        max_steps: int = 5
    ) -> Tuple[List[ToolCall], str, float]:
        """
        Generate a single tool-use trajectory.

        Args:
            prompt: Input prompt
            ground_truth: Expected answer
            max_steps: Maximum tool-call steps

        Returns:
            (trajectory, final_answer, reward)
        """
        self.env.reset()
        trajectory = []

        # Simulate agent decision-making
        state = np.random.randn(self.policy.d_model)  # Simplified state

        for step in range(max_steps):
            # Decide: use tool or answer?
            tool_idx, confidence = self.policy.select_tool(state)

            # If confident enough, answer directly
            if confidence > 0.7 or step == max_steps - 1:
                answer_logits = self.policy.generate_answer(state)
                # Simulate answer
                final_answer = f"Answer based on {len(trajectory)} tool calls"
                break
            else:
                # Use tool
                tool_names = list(self.env.tools.keys())
                tool_name = tool_names[tool_idx % len(tool_names)]

                # Generate arguments (simplified)
                args = {'query': prompt[:50]}

                # Execute tool
                call = self.env.execute(tool_name, args)
                trajectory.append(call)

                # Update state with tool result
                state = state * 0.9 + np.random.randn(self.policy.d_model) * 0.1
        else:
            final_answer = f"Max steps reached with {len(trajectory)} tool calls"

        # Compute reward
        reward = self.compute_reward(trajectory, final_answer, ground_truth)

        return trajectory, final_answer, reward

    def train_step(self, prompt: str, ground_truth: str) -> Dict[str, float]:
        """
        One GRPO training step.

        Args:
            prompt: Training prompt
            ground_truth: Expected answer

        Returns:
            Training metrics
        """
        # Generate N trajectories (group)
        trajectories = []
        rewards = []

        for _ in range(self.num_samples):
            traj, answer, reward = self.generate_trajectory(prompt, ground_truth)
            trajectories.append(traj)
            rewards.append(reward)

        rewards = np.array(rewards)

        # Group-relative normalization
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8
        advantages = (rewards - mean_reward) / std_reward

        # Policy gradient update (simplified)
        # Increase probability of high-advantage trajectories
        for i, (traj, advantage) in enumerate(zip(trajectories, advantages)):
            if len(traj) > 0 and advantage > 0:
                # Reinforce successful tool-use patterns
                for call in traj:
                    if call.success:
                        # Update tool selection toward successful tools
                        tool_idx = list(self.env.tools.keys()).index(call.tool_name)
                        self.policy.tool_head_W[:, tool_idx] += self.learning_rate * advantage * 0.01

        # KL penalty
        kl = np.mean((self.policy.tool_head_W - self.ref_tool_W) ** 2)

        metrics = {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'max_reward': float(np.max(rewards)),
            'kl_divergence': float(kl),
            'avg_trajectory_length': float(np.mean([len(t) for t in trajectories]))
        }

        self.history.append(metrics)
        return metrics

    def train(self, dataset: List[Tuple[str, str]], num_epochs: int = 3) -> List[Dict]:
        """
        Train on a dataset of (prompt, ground_truth) pairs.

        Args:
            dataset: List of (prompt, ground_truth) tuples
            num_epochs: Number of training epochs

        Returns:
            Training history
        """
        all_metrics = []

        for epoch in range(num_epochs):
            epoch_metrics = []

            for prompt, ground_truth in dataset:
                metrics = self.train_step(prompt, ground_truth)
                epoch_metrics.append(metrics)

            # Aggregate epoch metrics
            avg_metrics = {
                'epoch': epoch,
                'mean_reward': np.mean([m['mean_reward'] for m in epoch_metrics]),
                'max_reward': np.max([m['max_reward'] for m in epoch_metrics]),
                'avg_traj_len': np.mean([m['avg_trajectory_length'] for m in epoch_metrics])
            }
            all_metrics.append(avg_metrics)

        return all_metrics


################################################################################
# SECTION 4: TOOL-AUGMENTED REASONING
################################################################################

class ToolAugmentedReasoning:
    """
    Tool-Augmented Reasoning with RL
    ===================================

    Combines chain-of-thought reasoning with tool use,
    trained end-to-end with RL.

    The agent learns to:
    1. Reason about what information is needed
    2. Decide which tool can provide it
    3. Use tool results to continue reasoning
    4. Synthesize a final answer

    This is more powerful than either pure reasoning or pure tool use.

    Interview Questions:
        1. "How do you combine reasoning and tool use?"
           Interleave reasoning steps with tool calls. The agent
           thinks about what it needs, calls a tool, observes the
           result, and continues reasoning. RL optimizes the whole
           chain for final answer quality.

        2. "What's the advantage over ReAct?"
           ReAct uses a fixed reasoning-acting pattern. RL-trained
           agents can learn more flexible patterns — sometimes
           multiple tool calls before reasoning, sometimes reasoning
           first to decide which tool to use.
    """

    def __init__(self, policy: ToolUsePolicy, env: ToolEnvironment):
        self.policy = policy
        self.env = env

    def reason_and_act(
        self,
        prompt: str,
        max_reasoning_steps: int = 10
    ) -> Dict:
        """
        Perform tool-augmented reasoning.

        Args:
            prompt: Input question/task
            max_reasoning_steps: Maximum reasoning steps

        Returns:
            Dictionary with reasoning trace and answer
        """
        trace = []
        state = np.random.randn(self.policy.d_model)

        for step in range(max_reasoning_steps):
            # Decide action
            tool_idx, confidence = self.policy.select_tool(state)

            if confidence > 0.8:
                # Answer directly
                trace.append({
                    'step': step,
                    'action': 'answer',
                    'confidence': confidence
                })
                break
            else:
                # Use tool
                tool_names = list(self.env.tools.keys())
                tool_name = tool_names[tool_idx % len(tool_names)]

                call = self.env.execute(tool_name, {'query': prompt[:50]})

                trace.append({
                    'step': step,
                    'action': 'tool_call',
                    'tool': tool_name,
                    'success': call.success,
                    'confidence': confidence
                })

                # Update state
                state = state * 0.9 + np.random.randn(self.policy.d_model) * 0.1

        return {
            'trace': trace,
            'num_steps': len(trace),
            'tools_used': [t['tool'] for t in trace if t['action'] == 'tool_call']
        }


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################

def demonstrate_tool_use_rl():
    """Demonstrate tool-use RL training."""
    print("=" * 70)
    print("TOOL-USE RL TRAINING")
    print("=" * 70)

    # Configuration
    vocab_size = 100
    d_model = 32
    num_tools = 4

    print(f"\nConfiguration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model dim: {d_model}")
    print(f"  Num tools: {num_tools}")

    # Create components
    policy = ToolUsePolicy(vocab_size, d_model, num_tools)
    env = ToolEnvironment()
    trainer = GRPOToolTrainer(policy, env, num_samples=4, learning_rate=0.01)

    # Training dataset
    dataset = [
        ("What is Python?", "Python is a programming language"),
        ("Calculate 2+3", "5"),
        ("What is a transformer?", "Transformers use attention"),
    ]

    # Tool environment demo
    print("\n--- Tool Environment ---")
    env.reset()
    for tool_name in ['search', 'calculator', 'retriever']:
        if tool_name == 'calculator':
            call = env.execute(tool_name, {'expression': '2+3'})
        else:
            call = env.execute(tool_name, {'query': 'test'})
        print(f"  {tool_name}: success={call.success}, result={call.result[:50]}")

    # Training
    print("\n--- GRPO Training ---")
    metrics = trainer.train(dataset, num_epochs=3)

    for m in metrics:
        print(f"  Epoch {m['epoch']}: "
              f"mean_reward={m['mean_reward']:.3f}, "
              f"max_reward={m['max_reward']:.3f}, "
              f"avg_traj_len={m['avg_traj_len']:.1f}")

    # Tool-augmented reasoning
    print("\n--- Tool-Augmented Reasoning ---")
    reasoning = ToolAugmentedReasoning(policy, env)

    result = reasoning.reason_and_act("What is Python?")
    print(f"  Steps: {result['num_steps']}")
    print(f"  Tools used: {result['tools_used']}")
    for step in result['trace']:
        print(f"    Step {step['step']}: {step['action']} "
              f"(confidence={step['confidence']:.2f})")

    # Policy analysis
    print("\n--- Policy Analysis ---")
    state = np.random.randn(d_model)
    tool_idx, confidence = policy.select_tool(state)
    tool_names = list(env.tools.keys())
    print(f"  Selected tool: {tool_names[tool_idx % len(tool_names)]}")
    print(f"  Confidence: {confidence:.3f}")

    value = policy.estimate_value(state)
    print(f"  State value: {value:.3f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: RL trains agents to use tools for task completion!")
    print("GRPO compares N trajectories per prompt, no critic model needed.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_tool_use_rl()
