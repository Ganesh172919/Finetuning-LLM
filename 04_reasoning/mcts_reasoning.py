"""
################################################################################
MCTS REASONING — MONTE CARLO TREE SEARCH FOR MATHEMATICAL REASONING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is MCTS Reasoning?
    Monte Carlo Tree Search (MCTS) applied to reasoning and mathematical
    proofs. Instead of exploring all paths (ToT) or following one path
    (CoT), MCTS uses intelligent exploration guided by value estimates.

    This is the approach behind AlphaProof (Google DeepMind, 2024).

Why does it matter?
    MCTS combines the best of both worlds:
    - Exploration: tries many different reasoning paths (like ToT)
    - Exploitation: focuses on the most promising paths (like CoT)
    - Principled: UCB1 provides theoretical guarantees
    - Scalable: can run for arbitrary compute budgets

How does it work?
    Four phases repeated many times:
    1. Selection: traverse tree using UCB1 (balance explore/exploit)
    2. Expansion: add a new reasoning step to the tree
    3. Simulation: estimate value by random rollouts
    4. Backpropagation: update statistics back to root

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ MCTS for Reasoning                                          │
    │                                                              │
    │  1. SELECT: UCB1 = Q + c * sqrt(ln(N_parent) / N_child)    │
    │     ↓                                                        │
    │  2. EXPAND: Generate new reasoning step                     │
    │     ↓                                                        │
    │  3. SIMULATE: Estimate value (rollout or value network)     │
    │     ↓                                                        │
    │  4. BACKPROP: Update Q values up the tree                   │
    │                                                              │
    │  Repeat N times → Best path = highest Q leaf                │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2006: MCTS for Go (Coulom, Kocsis & Szepesvári)
    - 2016: AlphaGo — MCTS + neural networks for Go
    - 2017: AlphaZero — MCTS for chess, shogi, Go
    - 2024: AlphaProof — MCTS for mathematical reasoning
    - 2025: MCTS + LLMs for reasoning, planning, code generation

INTERVIEW QUESTIONS:
    1. "How does MCTS work for reasoning?"
       MCTS builds a search tree over reasoning steps. Each node is a
       partial proof/solution. UCB1 selects which node to expand next,
       balancing exploitation (high-value paths) and exploration
       (less-visited paths). Simulations estimate node values.

    2. "What is UCB1 and why use it?"
       UCB1 = Q(v) + c * sqrt(ln(N_parent) / N_child). First term
       exploits high-value nodes, second term explores rarely-visited
       nodes. c controls the balance. It's theoretically optimal for
       multi-armed bandits and works well for tree search.

    3. "How does AlphaProof use MCTS?"
       AlphaProof uses MCTS to search for mathematical proofs.
       Each node is a partial proof step. The value function estimates
       how likely a partial proof leads to a complete proof. Trained
       on synthetic proof data.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass, field
import math

import sys
sys.path.append('..')
from ..01_math.probability import softmax


################################################################################
# SECTION 1: MCTS NODE
################################################################################

class MCTSNode:
    """
    A node in the MCTS reasoning tree.

    Each node represents a partial reasoning state (proof step, etc.).

    Attributes:
        state: The reasoning state at this node
        parent: Parent node (None for root)
        children: Child nodes (next reasoning steps)
        value_sum: Sum of all simulation values through this node
        visit_count: Number of times this node has been visited
        prior_probability: Prior probability from policy network
        is_terminal: Whether this is a complete solution
    """

    def __init__(self, state: str, parent: Optional['MCTSNode'] = None,
                 prior: float = 0.0, is_terminal: bool = False):
        self.state = state
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.value_sum = 0.0
        self.visit_count = 0
        self.prior_probability = prior
        self.is_terminal = is_terminal
        self.is_fully_expanded = False

    @property
    def q_value(self) -> float:
        """Average value (Q) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb1(self, c: float = 1.414) -> float:
        """
        UCB1 score for this node.

        Formula: UCB1 = Q + c * sqrt(ln(N_parent) / N)

        Args:
            c: Exploration constant (default sqrt(2) ≈ 1.414)

        Returns:
            UCB1 score
        """
        if self.visit_count == 0:
            return float('inf')  # Unvisited nodes have infinite UCB

        if self.parent is None:
            return self.q_value

        exploration = c * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return self.q_value + exploration

    def puct(self, c_puct: float = 1.0) -> float:
        """
        PUCT score (AlphaZero-style).

        Formula: PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N)

        Args:
            c_puct: Exploration constant

        Returns:
            PUCT score
        """
        if self.parent is None:
            return self.q_value

        exploration = c_puct * self.prior_probability * (
            math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        )
        return self.q_value + exploration

    def add_child(self, child_state: str, prior: float = 0.0,
                  is_terminal: bool = False) -> 'MCTSNode':
        """Add a child node."""
        child = MCTSNode(child_state, parent=self, prior=prior,
                        is_terminal=is_terminal)
        self.children.append(child)
        return child

    def best_child(self, c: float = 1.414) -> 'MCTSNode':
        """Select child with highest UCB1 score."""
        return max(self.children, key=lambda n: n.ucb1(c))

    def best_child_puct(self, c_puct: float = 1.0) -> 'MCTSNode':
        """Select child with highest PUCT score."""
        return max(self.children, key=lambda n: n.puct(c_puct))

    def get_path(self) -> List[str]:
        """Get path from root to this node."""
        path = []
        node = self
        while node is not None:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))

    def __repr__(self) -> str:
        return (f"MCTSNode(Q={self.q_value:.3f}, N={self.visit_count}, "
                f"children={len(self.children)})")


################################################################################
# SECTION 2: MCTS CONFIGURATION
################################################################################

@dataclass
class MCTSConfig:
    """
    MCTS Configuration.

    Attributes:
        num_simulations: Number of MCTS simulations to run
        c_puct: Exploration constant for UCB1/PUCT
        temperature: Temperature for action selection after search
        max_depth: Maximum depth of the tree
        use_value_network: Use neural value function vs rollout
    """
    num_simulations: int = 100
    c_puct: float = 1.414
    temperature: float = 1.0
    max_depth: int = 20
    use_value_network: bool = False


################################################################################
# SECTION 3: UCB SELECTOR
################################################################################

class UCBSelector:
    """
    UCB1 Selection for MCTS.

    Balances exploitation (high Q) with exploration (low N).

    Formula: UCB1 = Q + c * sqrt(ln(N_parent) / N)

    Interview Question:
        "Explain UCB1 in MCTS."
        UCB1 has two terms: exploitation (Q value, prefer high-reward
        nodes) and exploration (sqrt(ln(N_parent)/N), prefer less-visited
        nodes). The constant c balances them. c=sqrt(2) is theoretically
        optimal for multi-armed bandits.
    """

    def __init__(self, c: float = 1.414):
        self.c = c

    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Select the best child using UCB1.

        Args:
            node: Parent node

        Returns:
            Child with highest UCB1 score
        """
        if not node.children:
            return node
        return max(node.children, key=lambda n: n.ucb1(self.c))


################################################################################
# SECTION 4: VALUE ESTIMATOR
################################################################################

class ValueEstimator:
    """
    Estimate the value of a reasoning state.

    Methods:
    1. Rollout: Random simulation to terminal state
    2. Neural: Learned value function
    3. Heuristic: Rule-based evaluation

    Interview Question:
        "How do you estimate value in MCTS for reasoning?"
        Three approaches: (1) Rollout — randomly continue reasoning
        and see if it reaches a solution, (2) Value network — train a
        model to predict solution probability from partial state,
        (3) Heuristic — rule-based scoring (step quality, completeness).
    """

    def __init__(self):
        self.rollout_depth = 5

    def rollout(self, node: MCTSNode) -> float:
        """
        Estimate value by random simulation.

        Args:
            node: Starting node for rollout

        Returns:
            Estimated value (0 to 1)
        """
        # Simulate random continuation
        depth = 0
        current = node
        while depth < self.rollout_depth:
            # Simulate a random next step
            if np.random.random() < 0.1:  # 10% chance of "solving"
                return 1.0
            depth += 1
        return np.random.uniform(0, 0.5)

    def heuristic(self, node: MCTSNode) -> float:
        """
        Estimate value using heuristics.

        Args:
            node: Node to evaluate

        Returns:
            Estimated value (0 to 1)
        """
        score = 0.5
        # Reward depth (more reasoning = better)
        depth = len(node.get_path())
        score += min(0.2, depth * 0.02)
        # Add noise
        score += np.random.normal(0, 0.05)
        return np.clip(score, 0, 1)

    def estimate(self, node: MCTSNode, method: str = 'heuristic') -> float:
        """
        Estimate value using specified method.

        Args:
            node: Node to evaluate
            method: 'rollout' or 'heuristic'

        Returns:
            Estimated value
        """
        if method == 'rollout':
            return self.rollout(node)
        return self.heuristic(node)


################################################################################
# SECTION 5: PROOF VERIFIER
################################################################################

class ProofVerifier:
    """
    Verify if a reasoning chain constitutes a valid proof.

    Checks:
    1. Each step follows from previous steps
    2. No logical gaps
    3. Conclusion is reached
    4. All premises are used

    Interview Question:
        "How do you verify mathematical proofs with MCTS?"
        Each node in the MCTS tree is a partial proof. The verifier
        checks if a proof is complete by: (1) verifying each step
        logically follows, (2) checking no gaps exist, (3) confirming
        the conclusion is reached. Terminal nodes are verified proofs.
    """

    def is_valid_step(self, step: str, previous_steps: List[str]) -> bool:
        """
        Check if a proof step is valid.

        Args:
            step: Current proof step
            previous_steps: All previous steps

        Returns:
            True if step is valid
        """
        # Simplified validation
        if not step:
            return False
        # Step should reference previous context
        return len(step) > 5

    def is_complete_proof(self, steps: List[str]) -> bool:
        """
        Check if steps form a complete proof.

        Args:
            steps: All proof steps

        Returns:
            True if proof is complete
        """
        if len(steps) < 2:
            return False
        # Check for conclusion
        last_step = steps[-1].lower()
        has_conclusion = any(w in last_step for w in
                           ['therefore', 'thus', 'q.e.d', 'proved', 'hence'])
        return has_conclusion

    def verify(self, path: List[str]) -> Dict:
        """
        Verify a complete reasoning path.

        Args:
            path: List of reasoning steps

        Returns:
            Dictionary with verification results
        """
        valid_steps = sum(1 for i, step in enumerate(path)
                         if self.is_valid_step(step, path[:i]))
        complete = self.is_complete_proof(path)

        return {
            'valid_steps': valid_steps,
            'total_steps': len(path),
            'step_validity_rate': valid_steps / max(len(path), 1),
            'is_complete': complete,
            'is_valid_proof': complete and valid_steps == len(path)
        }


################################################################################
# SECTION 6: MCTS REASONING (MAIN CLASS)
################################################################################

class MCTSReasoning:
    """
    MCTS for Reasoning — Complete implementation.

    Runs MCTS to find the best reasoning path by:
    1. Building a search tree over reasoning steps
    2. Using UCB1 to balance exploration/exploitation
    3. Estimating values via rollouts or heuristics
    4. Returning the highest-value path

    Paper inspiration: "AlphaProof" (Google DeepMind, 2024)

    Step by step:
        1. Create root node from problem
        2. For each simulation:
           a. SELECT: traverse tree using UCB1
           b. EXPAND: add new reasoning step
           c. SIMULATE: estimate value
           d. BACKPROP: update values up tree
        3. Return path with highest visit count or value

    Interview Question:
        "How would you use MCTS for mathematical reasoning?"
        Build a search tree where each node is a partial proof step.
        Use UCB1 to select which step to expand next. Estimate values
        via rollouts (random completion) or learned value function.
        After many simulations, follow the most-visited path for the
        highest-confidence proof.
    """

    def __init__(self, config: Optional[MCTSConfig] = None):
        """Initialize MCTS Reasoning."""
        self.config = config or MCTSConfig()
        self.selector = UCBSelector(self.config.c_puct)
        self.estimator = ValueEstimator()
        self.verifier = ProofVerifier()

    def generate_next_steps(self, node: MCTSNode, k: int = 3) -> List[str]:
        """
        Generate K possible next reasoning steps.

        Args:
            node: Current node
            k: Number of steps to generate

        Returns:
            List of possible next step strings
        """
        # Simulate step generation (in production: LLM call)
        steps = []
        for i in range(k):
            steps.append(f"Reasoning step from '{node.state[:20]}...' option {i+1}")
        return steps

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select leaf node using UCB1."""
        current = node
        while current.children:
            current = self.selector.select(current)
        return current

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding children."""
        if node.is_terminal:
            return node

        next_steps = self.generate_next_steps(node)
        for step in next_steps:
            is_terminal = np.random.random() < 0.1  # 10% terminal
            node.add_child(step, prior=1.0/len(next_steps), is_terminal=is_terminal)

        node.is_fully_expanded = True
        # Return first child for simulation
        return node.children[0] if node.children else node

    def simulate(self, node: MCTSNode) -> float:
        """Estimate value via rollout or heuristic."""
        if self.config.use_value_network:
            return self.estimator.estimate(node, 'heuristic')
        return self.estimator.rollout(node)

    def backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = current.parent

    def search(self, problem: str) -> Dict:
        """
        Run MCTS search.

        Args:
            problem: The problem to solve

        Returns:
            Dictionary with best path, value, statistics
        """
        root = MCTSNode(f"Problem: {problem}")

        for sim in range(self.config.num_simulations):
            # 1. Select
            leaf = self.select(root)

            # 2. Expand
            if not leaf.is_terminal:
                child = self.expand(leaf)
            else:
                child = leaf

            # 3. Simulate
            value = self.simulate(child)

            # 4. Backpropagate
            self.backpropagate(child, value)

        # Extract best path (most visited child at each level)
        best_path = []
        current = root
        while current.children:
            best_child = max(current.children, key=lambda n: n.visit_count)
            best_path.append(best_child.state)
            current = best_child

        return {
            'problem': problem,
            'best_path': best_path,
            'root_visits': root.visit_count,
            'root_value': root.q_value,
            'tree_depth': max(len(n.get_path()) for n in self._get_all_nodes(root)),
            'num_simulations': self.config.num_simulations
        }

    def _get_all_nodes(self, root: MCTSNode) -> List[MCTSNode]:
        """Get all nodes in the tree."""
        nodes = [root]
        for child in root.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes


################################################################################
# SECTION 7: ALPHAPROOF STYLE
################################################################################

class AlphaProofStyle:
    """
    AlphaProof-style MCTS for mathematical reasoning.

    Combines MCTS with:
    - Policy network: suggests promising proof steps
    - Value network: estimates proof completion probability
    - Proof verifier: checks if proof is valid

    Paper: "AlphaProof" (Google DeepMind, 2024)
    Achievement: Solved 4/6 IMO 2024 problems

    Interview Question:
        "How does AlphaProof work?"
        AlphaProof uses MCTS to search for mathematical proofs. The
        policy network suggests candidate proof steps (expansion),
        the value network estimates how close we are to a complete
        proof (evaluation), and the verifier checks if a proof is
        valid (terminal check). MCTS balances exploring new paths
        with exploiting promising ones.
    """

    def __init__(self, config: Optional[MCTSConfig] = None):
        """Initialize AlphaProof-style MCTS."""
        self.config = config or MCTSConfig()
        self.mcts = MCTSReasoning(config)
        self.verifier = ProofVerifier()

    def policy_prior(self, state: str, next_steps: List[str]) -> List[float]:
        """
        Policy network: estimate prior probability for each next step.

        Args:
            state: Current reasoning state
            next_steps: Candidate next steps

        Returns:
            Prior probabilities for each step
        """
        # Simulate policy network (in production: neural network)
        priors = np.random.dirichlet(np.ones(len(next_steps)))
        return priors.tolist()

    def value_estimate(self, state: str) -> float:
        """
        Value network: estimate proof completion probability.

        Args:
            state: Current reasoning state

        Returns:
            Estimated probability of completing proof
        """
        # Simulate value network
        return np.random.uniform(0, 1)

    def search(self, problem: str) -> Dict:
        """
        Run AlphaProof-style search.

        Args:
            problem: Mathematical problem to prove

        Returns:
            Dictionary with proof attempt, verification, statistics
        """
        # Run MCTS
        result = self.mcts.search(problem)

        # Verify the proof
        verification = self.verifier.verify(result['best_path'])

        result['verification'] = verification
        result['is_proved'] = verification['is_valid_proof']

        return result


################################################################################
# SECTION 8: TESTING & DEMONSTRATION
################################################################################

def demonstrate_mcts_reasoning():
    """
    Demonstrate MCTS reasoning methods.
    """
    print("=" * 70)
    print("MCTS REASONING DEMONSTRATION")
    print("=" * 70)

    # ── MCTSNode ──
    print("\n1. MCTS NODE")
    print("-" * 40)
    root = MCTSNode("Start: Solve x^2 = 4")
    child1 = root.add_child("Step 1: Take square root", prior=0.5)
    child2 = root.add_child("Step 1: Factor as (x-2)(x+2)=0", prior=0.5)
    child1.visit_count = 10
    child1.value_sum = 7.0
    child2.visit_count = 5
    child2.value_sum = 4.0
    print(f"  Root: {root}")
    print(f"  Child 1 (square root): Q={child1.q_value:.3f}, UCB1={child1.ucb1():.3f}")
    print(f"  Child 2 (factoring): Q={child2.q_value:.3f}, UCB1={child2.ucb1():.3f}")

    # ── UCB Selector ──
    print("\n2. UCB SELECTOR")
    print("-" * 40)
    selector = UCBSelector(c=1.414)
    best = selector.select(root)
    print(f"  Selected: {best.state}")

    # ── Value Estimator ──
    print("\n3. VALUE ESTIMATOR")
    print("-" * 40)
    estimator = ValueEstimator()
    node = MCTSNode("Partial proof step")
    rollout_val = estimator.estimate(node, 'rollout')
    heuristic_val = estimator.estimate(node, 'heuristic')
    print(f"  Rollout value: {rollout_val:.3f}")
    print(f"  Heuristic value: {heuristic_val:.3f}")

    # ── MCTS Search ──
    print("\n4. MCTS SEARCH")
    print("-" * 40)
    config = MCTSConfig(num_simulations=50, c_puct=1.414)
    mcts = MCTSReasoning(config)
    result = mcts.search("Prove that sqrt(2) is irrational")
    print(f"  Problem: {result['problem']}")
    print(f"  Best path length: {len(result['best_path'])}")
    print(f"  Root visits: {result['root_visits']}")
    print(f"  Root value: {result['root_value']:.3f}")
    print(f"  Tree depth: {result['tree_depth']}")

    # ── AlphaProof Style ──
    print("\n5. ALPHAPROOF STYLE")
    print("-" * 40)
    alpha = AlphaProofStyle(config)
    result = alpha.search("Prove: For all n, n^2 + n is even")
    print(f"  Problem: {result['problem']}")
    print(f"  Proof steps: {len(result['best_path'])}")
    print(f"  Is proved: {result['is_proved']}")
    print(f"  Verification: {result['verification']}")

    # ── Comparison ──
    print("\n6. MCTS vs BFS vs DFS")
    print("-" * 40)
    print("  MCTS: Intelligent exploration via UCB1, scalable to any budget")
    print("  BFS:  Breadth-first, keeps top-B, good for shallow solutions")
    print("  DFS:  Depth-first, backtracks, memory-efficient")
    print("  MCTS is best for: deep reasoning, proofs, planning")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_mcts_reasoning()
