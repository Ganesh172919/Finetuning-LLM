"""
################################################################################
SEARCH-AUGMENTED AGENTS — MCTS AND RL-GUIDED SEARCH FOR AGENTS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Search-Augmented Agents?
    Agents that use search algorithms (MCTS, beam search, Tree of Thoughts)
    to plan ahead before acting. Instead of greedy single-step decisions,
    the agent explores multiple possible futures and picks the best path.

Why does it matter?
    Greedy agents (like basic ReAct) make one decision at a time:
    - Can't backtrack on mistakes
    - Miss globally optimal strategies
    - No planning ahead

    Search-augmented agents:
    - Explore multiple action sequences
    - Evaluate futures with a value function
    - Pick the best overall plan
    - Handle complex, multi-step tasks

How does it work?
    1. Current state: agent's situation
    2. Expand: generate possible next actions
    3. Evaluate: score each action (heuristic or learned value)
    4. Select: pick the most promising action
    5. Repeat: plan multiple steps ahead

Key Approaches:
    - MCTS (Monte Carlo Tree Search): Used in AlphaGo/AlphaZero
    - Beam Search: Keep top-K promising paths
    - Tree of Thoughts: LLM-specific tree search
    - RL-guided search: Learn value function to guide search

Architecture (MCTS for Agents):
    ┌─────────────────────────────────────────────────────────────────┐
    │                MCTS Agent Planning                             │
    │                                                                  │
    │  State ──▶ Expand children (possible actions)                  │
    │              ↓                                                   │
    │         Select (UCB1: balance exploration/exploitation)        │
    │              ↓                                                   │
    │         Simulate (rollout: estimate value)                     │
    │              ↓                                                   │
    │         Backpropagate (update parent values)                   │
    │              ↓                                                   │
    │         Repeat K times ──▶ Best action                         │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "How does MCTS work for agents?"
       MCTS builds a search tree by: (1) selecting promising nodes,
       (2) expanding them with new actions, (3) simulating outcomes,
       (4) backpropagating results. After K iterations, pick the
       action with the highest average value.

    2. "When should I use search vs greedy?"
       Search: when decisions are irreversible, when the task requires
       planning, when you have compute budget. Greedy: when decisions
       are low-stakes, when speed matters more than optimality.

    3. "What's the relationship to AlphaZero?"
       AlphaZero uses MCTS with a learned value function and policy
       network. Agents can use the same approach: LLM as policy,
       learned value function to evaluate states.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass, field
import math

################################################################################
# SECTION 1: SEARCH TREE NODE
################################################################################

@dataclass
class SearchNode:
    """
    Node in the search tree.

    Each node represents a state in the agent's planning process.
    """
    state: np.ndarray  # State representation
    action: Optional[str] = None  # Action that led to this state
    parent: Optional['SearchNode'] = None
    children: List['SearchNode'] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    prior_prob: float = 0.0  # Prior from policy network

    @property
    def avg_value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb1_score(self, exploration_coeff: float = 1.41) -> float:
        """
        UCB1 score for node selection.

        UCB1 = avg_value + c * sqrt(ln(parent_visits) / visits)

        Balances exploitation (high value) with exploration (few visits).

        Interview Question:
            "What is UCB1?"
            Upper Confidence Bound 1: a formula that balances
            exploiting known good actions with exploring uncertain ones.
            The exploration term decreases as a node is visited more.
        """
        if self.visit_count == 0:
            return float('inf')  # Unvisited nodes are always explored

        if self.parent is None:
            return self.avg_value

        exploitation = self.avg_value
        exploration = exploration_coeff * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )

        return exploitation + exploration


################################################################################
# SECTION 2: MONTE CARLO TREE SEARCH
################################################################################

class MCTS:
    """
    Monte Carlo Tree Search for Agent Planning
    =============================================

    Builds a search tree to find the best action sequence.

    Algorithm:
    1. Selection: Traverse tree using UCB1 to find promising leaf
    2. Expansion: Add child nodes for possible actions
    3. Simulation: Rollout from leaf to estimate value
    4. Backpropagation: Update values up the tree

    After K iterations, pick the action with highest visit count
    (most robust) or highest average value.

    Interview Questions:
        1. "Why MCTS over exhaustive search?"
           MCTS focuses compute on promising branches. Exhaustive
           search is exponential in depth. MCTS gives good results
           with much less computation.

        2. "How does the simulation/rollout work?"
           From the leaf node, take random (or policy-guided) actions
           until a terminal state. The outcome estimates the node's
           value. Better: use a learned value function instead of
           random rollouts.

        3. "What's the exploration coefficient?"
           Controls how much MCTS explores vs exploits. Higher = more
           exploration (try new things). Lower = more exploitation
           (focus on known good paths). Typical: sqrt(2).
    """

    def __init__(
        self,
        action_space: List[str],
        value_fn: Callable,
        policy_fn: Optional[Callable] = None,
        num_simulations: int = 100,
        exploration_coeff: float = 1.41
    ):
        self.action_space = action_space
        self.value_fn = value_fn  # Estimates state value
        self.policy_fn = policy_fn  # Prior probabilities for actions
        self.num_simulations = num_simulations
        self.exploration_coeff = exploration_coeff

    def search(self, root_state: np.ndarray) -> str:
        """
        Run MCTS from root state.

        Args:
            root_state: Current state representation

        Returns:
            Best action
        """
        # Create root node
        root = SearchNode(state=root_state)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root

            # Selection: traverse tree using UCB1
            while node.children and not self._is_terminal(node):
                node = self._select_child(node)

            # Expansion: add children if not terminal
            if not self._is_terminal(node) and node.visit_count > 0:
                node = self._expand(node)

            # Simulation: estimate value
            value = self._simulate(node)

            # Backpropagation: update values up the tree
            self._backpropagate(node, value)

        # Select best action (highest visit count)
        best_child = max(root.children, key=lambda c: c.visit_count)
        return best_child.action

    def _select_child(self, node: SearchNode) -> SearchNode:
        """Select child with highest UCB1 score."""
        return max(node.children, key=lambda c: c.ucb1_score(self.exploration_coeff))

    def _expand(self, node: SearchNode) -> SearchNode:
        """Expand node by adding children for each action."""
        # Get prior probabilities if available
        if self.policy_fn:
            priors = self.policy_fn(node.state)
        else:
            priors = np.ones(len(self.action_space)) / len(self.action_space)

        for i, action in enumerate(self.action_space):
            child_state = self._transition(node.state, action)
            child = SearchNode(
                state=child_state,
                action=action,
                parent=node,
                prior_prob=float(priors[i])
            )
            node.children.append(child)

        # Return a random child for simulation
        return node.children[np.random.randint(len(node.children))]

    def _simulate(self, node: SearchNode) -> float:
        """Simulate/rollout from node to estimate value."""
        return self.value_fn(node.state)

    def _backpropagate(self, node: SearchNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    def _is_terminal(self, node: SearchNode) -> bool:
        """Check if node is terminal (simplified)."""
        return False

    def _transition(self, state: np.ndarray, action: str) -> np.ndarray:
        """Compute next state given action (simplified)."""
        # Random transition for demonstration
        return state + np.random.randn(*state.shape) * 0.1


################################################################################
# SECTION 3: BEAM SEARCH AGENT
################################################################################

class BeamSearchAgent:
    """
    Beam Search Agent
    ==================

    Maintains K (beam width) most promising action sequences.

    Unlike MCTS which builds a tree, beam search maintains
    a flat list of K best partial solutions.

    Algorithm:
    1. Start with K initial candidates
    2. For each step, expand each candidate with all actions
    3. Keep top-K candidates by cumulative score
    4. After max steps, return best candidate

    Interview Questions:
        1. "Beam search vs MCTS — when to use which?"
           Beam search: simpler, good for sequence generation,
           fixed width. MCTS: better for planning, adapts depth
           to promising branches, handles uncertainty better.

        2. "How do you score beam search candidates?"
           Options: (a) Cumulative log-probability,
           (b) Normalized by length, (c) Learned value function,
           (d) Task-specific heuristic.
    """

    def __init__(
        self,
        action_space: List[str],
        scoring_fn: Callable,
        beam_width: int = 5,
        max_depth: int = 10
    ):
        self.action_space = action_space
        self.scoring_fn = scoring_fn
        self.beam_width = beam_width
        self.max_depth = max_depth

    def search(self, initial_state: np.ndarray) -> List[str]:
        """
        Run beam search.

        Args:
            initial_state: Starting state

        Returns:
            Best action sequence
        """
        # Initialize beam with empty sequences
        beam = [(initial_state, [], 0.0)]  # (state, actions, score)

        for depth in range(self.max_depth):
            candidates = []

            for state, actions, score in beam:
                # Expand with all actions
                for action in self.action_space:
                    new_state = state + np.random.randn(*state.shape) * 0.1
                    action_score = self.scoring_fn(new_state, action)
                    new_score = score + action_score

                    candidates.append((new_state, actions + [action], new_score))

            # Keep top-K
            candidates.sort(key=lambda x: x[2], reverse=True)
            beam = candidates[:self.beam_width]

        # Return best sequence
        best = beam[0]
        return best[1]


################################################################################
# SECTION 4: TREE OF THOUGHTS
################################################################################

class TreeOfThoughts:
    """
    Tree of Thoughts (ToT)
    ========================

    LLM-specific tree search where:
    - Nodes are "thoughts" (reasoning steps)
    - Children are alternative thoughts
    - Value function evaluates thought quality

    Unlike ReAct (linear), ToT explores multiple reasoning paths
    and backtracks when a path seems unpromising.

    Interview Questions:
        1. "What is Tree of Thoughts?"
           A framework for LLM problem-solving that explores multiple
           reasoning paths as a tree. At each step, generate several
           possible thoughts, evaluate them, and continue from the
           most promising ones.

        2. "How does ToT differ from ReAct?"
           ReAct is linear: think → act → observe → think...
           ToT is a tree: think A → [thought A1, A2, A3]
           ToT can backtrack and explore alternatives.

        3. "When is ToT useful?"
           Problems requiring search: puzzles, math proofs,
           strategic planning, creative writing with constraints.
    """

    def __init__(
        self,
        thought_generator: Callable,
        thought_evaluator: Callable,
        num_thoughts: int = 3,
        num_simulations: int = 10
    ):
        self.thought_generator = thought_generator  # Generate thought candidates
        self.thought_evaluator = thought_evaluator  # Evaluate thought quality
        self.num_thoughts = num_thoughts
        self.num_simulations = num_simulations

    def search(self, problem: str, max_depth: int = 5) -> List[str]:
        """
        Search for solution using Tree of Thoughts.

        Args:
            problem: Problem description
            max_depth: Maximum reasoning depth

        Returns:
            Best thought sequence
        """
        # Root thought
        root = SearchNode(state=np.random.randn(32))

        best_path = []
        best_value = -float('inf')

        # BFS search
        frontier = [(root, [])]

        for depth in range(max_depth):
            next_frontier = []

            for node, path in frontier:
                # Generate thought candidates
                thoughts = [f"Thought {i}" for i in range(self.num_thoughts)]

                for thought in thoughts:
                    # Evaluate thought
                    value = self.thought_evaluator(thought, problem)

                    # Create child node
                    child = SearchNode(
                        state=node.state + np.random.randn(*node.state.shape) * 0.1,
                        action=thought,
                        parent=node
                    )
                    child.total_value = value
                    child.visit_count = 1

                    new_path = path + [thought]

                    if value > best_value:
                        best_value = value
                        best_path = new_path

                    next_frontier.append((child, new_path))

            # Keep top-K for next level
            next_frontier.sort(
                key=lambda x: x[0].total_value, reverse=True
            )
            frontier = next_frontier[:self.num_thoughts]

        return best_path


################################################################################
# SECTION 5: RL-GUIDED SEARCH
################################################################################

class RLGuidedSearch:
    """
    RL-Guided Search
    =================

    Uses a learned value function to guide search.

    Unlike heuristic search, the value function is trained
    from experience to predict which states lead to success.

    This combines the planning ability of search with the
    learning ability of RL.

    Interview Question:
        "How do you train a value function for search?"
        Collect trajectories of agent behavior, label them with
        final outcomes (success/failure), train a value function
        to predict outcome from intermediate states. Use this
        to guide future search.
    """

    def __init__(self, state_dim: int, action_space: List[str]):
        self.state_dim = state_dim
        self.action_space = action_space

        # Value network (simplified)
        self.value_W = np.random.randn(state_dim, 1) * 0.02

        # Policy network (simplified)
        self.policy_W = np.random.randn(state_dim, len(action_space)) * 0.02

    def value(self, state: np.ndarray) -> float:
        """Estimate state value."""
        return float(state @ self.value_W)

    def policy(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities."""
        logits = state @ self.policy_W
        e_x = np.exp(logits - np.max(logits))
        return e_x / np.sum(e_x)

    def search(self, state: np.ndarray, num_simulations: int = 50) -> str:
        """
        Search using value function and policy.

        Args:
            state: Current state
            num_simulations: Number of search iterations

        Returns:
            Best action
        """
        action_values = np.zeros(len(self.action_space))
        action_counts = np.zeros(len(self.action_space))

        for _ in range(num_simulations):
            # Sample action from policy
            probs = self.policy(state)
            action_idx = np.random.choice(len(self.action_space), p=probs)

            # Simulate outcome
            next_state = state + np.random.randn(*state.shape) * 0.1
            value = self.value(next_state)

            # Update action values
            action_counts[action_idx] += 1
            action_values[action_idx] += (value - action_values[action_idx]) / action_counts[action_idx]

        # Return best action
        best_idx = np.argmax(action_values)
        return self.action_space[best_idx]

    def train_value(
        self,
        states: np.ndarray,
        outcomes: np.ndarray,
        learning_rate: float = 0.01,
        num_epochs: int = 100
    ):
        """
        Train value function from experience.

        Args:
            states: State samples (n, state_dim)
            outcomes: Final outcomes (n,)
            learning_rate: Learning rate
            num_epochs: Training epochs
        """
        for epoch in range(num_epochs):
            predictions = states @ self.value_W
            loss = np.mean((predictions.flatten() - outcomes) ** 2)

            # Gradient update
            grad = 2 * (predictions.flatten() - outcomes) @ states / len(states)
            self.value_W -= learning_rate * grad.reshape(-1, 1)

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}: loss={loss:.4f}")


################################################################################
# SECTION 6: TESTING & DEMONSTRATION
################################################################################

def demonstrate_search_agents():
    """Demonstrate search-augmented agents."""
    print("=" * 70)
    print("SEARCH-AUGMENTED AGENTS")
    print("=" * 70)

    state_dim = 16
    action_space = ["search", "calculate", "answer", "think", "verify"]

    print(f"\nConfiguration:")
    print(f"  State dim: {state_dim}")
    print(f"  Actions: {action_space}")

    # MCTS
    print("\n--- Monte Carlo Tree Search ---")
    value_fn = lambda s: float(np.tanh(np.sum(s)))
    mcts = MCTS(
        action_space=action_space,
        value_fn=value_fn,
        num_simulations=50,
        exploration_coeff=1.41
    )

    root_state = np.random.randn(state_dim)
    best_action = mcts.search(root_state)
    print(f"  Best action: {best_action}")

    # Beam Search
    print("\n--- Beam Search ---")
    scoring_fn = lambda s, a: float(np.sum(s)) + hash(a) % 10 / 10.0
    beam = BeamSearchAgent(
        action_space=action_space,
        scoring_fn=scoring_fn,
        beam_width=3,
        max_depth=5
    )

    best_sequence = beam.search(root_state)
    print(f"  Best sequence: {best_sequence}")

    # Tree of Thoughts
    print("\n--- Tree of Thoughts ---")
    thought_gen = lambda problem: [f"Approach {i}" for i in range(3)]
    thought_eval = lambda thought, problem: np.random.random()

    tot = TreeOfThoughts(
        thought_generator=thought_gen,
        thought_evaluator=thought_eval,
        num_thoughts=3,
        num_simulations=5
    )

    best_path = tot.search("Solve math problem", max_depth=3)
    print(f"  Best thought path: {best_path}")

    # RL-Guided Search
    print("\n--- RL-Guided Search ---")
    rl_search = RLGuidedSearch(state_dim, action_space)

    # Train value function
    print("  Training value function...")
    states = np.random.randn(100, state_dim)
    outcomes = np.random.randn(100)
    rl_search.train_value(states, outcomes, num_epochs=60)

    # Search with learned value
    best_action = rl_search.search(root_state, num_simulations=30)
    print(f"  Best action: {best_action}")

    # Compare policies
    print("\n--- Policy Comparison ---")
    probs = rl_search.policy(root_state)
    for action, prob in zip(action_space, probs):
        bar = "█" * int(prob * 30)
        print(f"  {action}: {prob:.3f} {bar}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Search agents plan ahead by exploring multiple futures!")
    print("MCTS + learned value function = powerful planning agent.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_search_agents()
