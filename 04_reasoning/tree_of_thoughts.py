"""
################################################################################
TREE OF THOUGHTS — EXPLORING MULTIPLE REASONING PATHS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Tree of Thoughts (ToT)?
    ToT extends Chain of Thought by exploring MULTIPLE reasoning paths
    simultaneously, like a tree. Instead of following one path, we:
    1. Generate multiple possible next thoughts at each step
    2. Evaluate each thought's promise
    3. Explore the most promising paths
    4. Backtrack if a path is unpromising

Why does it matter?
    CoT follows a single path — if it goes wrong, the whole chain fails.
    ToT explores alternatives and can recover from mistakes:
    - Crosswords: CoT 15.6% → ToT 78%
    - Game of 24: CoT 4% → ToT 74%
    - Creative writing: Significantly higher quality

How does it work?
    Think of it like chess: consider multiple moves, evaluate each,
    explore the best ones deeper. For reasoning:
    1. At each step, generate K possible next thoughts
    2. Evaluate each thought (is it promising?)
    3. Use BFS or DFS to explore the tree
    4. Backtrack when a path seems wrong

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ Tree of Thoughts                                            │
    │                                                              │
    │  Question ──▶ [Generate K thoughts]                         │
    │                    ↓                                         │
    │              [Evaluate each] ──▶ Score 1-10                 │
    │                    ↓                                         │
    │              [Select top-B] ──▶ Expand                      │
    │                    ↓                                         │
    │              [Repeat until depth D or solved]               │
    │                    ↓                                         │
    │              [Return best path]                             │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2023: Tree of Thoughts paper (Yao et al., Princeton/Google DeepMind)
    - 2023: Graph of Thoughts (Besta et al.)
    - 2024: ToT combined with MCTS for reasoning
    - 2025: AlphaProof uses tree search for math proofs

INTERVIEW QUESTIONS:
    1. "How does Tree of Thoughts differ from Chain of Thought?"
       CoT follows a single linear reasoning path. ToT explores multiple
       paths in parallel, evaluates them, and backtracks from bad paths.
       ToT is like BFS/DFS search over reasoning space.

    2. "When should you use ToT over CoT?"
       Use ToT when: (a) the problem has multiple valid approaches,
       (b) early mistakes compound, (c) you can evaluate intermediate
       steps, (d) the task benefits from exploration (puzzles, planning).

    3. "What's the computational cost of ToT?"
       ToT generates K thoughts per step × D depth = K*D evaluations.
       With BFS keeping top-B: B*K evaluations per level.
       Much more expensive than CoT but much more accurate.

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
# SECTION 1: THOUGHT NODE
################################################################################

class ThoughtNode:
    """
    A node in the Tree of Thoughts.

    Represents a single reasoning step or thought in the search tree.

    Attributes:
        thought: The text of this reasoning step
        value: Estimated quality of this thought (0-1)
        parent: Parent node (None for root)
        children: List of child nodes
        depth: Depth in the tree (0 for root)
        visit_count: Number of times this node has been visited
        cumulative_value: Sum of values along the path to this node
    """

    def __init__(self, thought: str, value: float = 0.0,
                 parent: Optional['ThoughtNode'] = None, depth: int = 0):
        self.thought = thought
        self.value = value
        self.parent = parent
        self.children: List['ThoughtNode'] = []
        self.depth = depth
        self.visit_count = 0
        self.cumulative_value = 0.0
        self.is_terminal = False
        self.is_pruned = False

    def add_child(self, child: 'ThoughtNode'):
        """Add a child node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def get_path(self) -> List[str]:
        """
        Get the reasoning path from root to this node.

        Returns:
            List of thought strings from root to this node
        """
        path = []
        node = self
        while node is not None:
            path.append(node.thought)
            node = node.parent
        return list(reversed(path))

    def get_path_value(self) -> float:
        """
        Get the average value along the path from root to this node.

        Returns:
            Average value of thoughts along the path
        """
        values = []
        node = self
        while node is not None:
            values.append(node.value)
            node = node.parent
        return np.mean(values) if values else 0.0

    def __repr__(self) -> str:
        return f"ThoughtNode(depth={self.depth}, value={self.value:.2f}, children={len(self.children)})"


################################################################################
# SECTION 2: THOUGHT TREE
################################################################################

class ThoughtTree:
    """
    The Tree of Thoughts structure.

    Manages the tree and provides operations for search algorithms.

    Interview Question:
        "How do you manage the tree in ToT?"
        The tree stores all explored thoughts. Each node has a value
        estimate, parent pointer, and children. Search algorithms
        (BFS/DFS) traverse this tree. We can prune unpromising branches
        and extract the best path from root to leaf.
    """

    def __init__(self, root_thought: str = "Root"):
        """Initialize with root thought."""
        self.root = ThoughtNode(root_thought, value=0.5, depth=0)
        self.all_nodes: List[ThoughtNode] = [self.root]
        self.best_leaf: Optional[ThoughtNode] = None

    def add_node(self, parent: ThoughtNode, thought: str, value: float) -> ThoughtNode:
        """
        Add a new thought node as a child of parent.

        Args:
            parent: Parent node
            thought: New thought text
            value: Estimated value of this thought

        Returns:
            The newly created node
        """
        child = ThoughtNode(thought, value, parent, parent.depth + 1)
        parent.add_child(child)
        self.all_nodes.append(child)

        # Update best leaf
        if self.best_leaf is None or value > self.best_leaf.value:
            self.best_leaf = child

        return child

    def get_leaves(self) -> List[ThoughtNode]:
        """Get all leaf nodes (nodes without children)."""
        return [n for n in self.all_nodes if not n.children and not n.is_pruned]

    def get_best_path(self) -> List[str]:
        """
        Get the best reasoning path (highest leaf value).

        Returns:
            List of thought strings from root to best leaf
        """
        if self.best_leaf is None:
            return [self.root.thought]
        return self.best_leaf.get_path()

    def prune_node(self, node: ThoughtNode):
        """Mark a node as pruned (don't explore further)."""
        node.is_pruned = True

    def get_stats(self) -> Dict:
        """Get tree statistics."""
        active = [n for n in self.all_nodes if not n.is_pruned]
        return {
            'total_nodes': len(self.all_nodes),
            'active_nodes': len(active),
            'pruned_nodes': len(self.all_nodes) - len(active),
            'max_depth': max(n.depth for n in active) if active else 0,
            'best_value': self.best_leaf.value if self.best_leaf else 0.0
        }


################################################################################
# SECTION 3: THOUGHT EVALUATOR
################################################################################

class ThoughtEvaluator:
    """
    Evaluate the quality of thoughts (reasoning steps).

    In a real system, this uses an LLM to score thoughts.
    Here we simulate evaluation with heuristics.

    Interview Question:
        "How do you evaluate intermediate reasoning steps?"
        Options: (1) LLM self-evaluation — ask the model to rate 1-10,
        (2) Value function — train a model to predict step quality,
        (3) Heuristic — rule-based scoring (length, keywords, consistency),
        (4) Outcome — simulate forward and check if it leads to solution.
    """

    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: Minimum value to continue exploring a path
        """
        self.threshold = threshold

    def evaluate(self, thought: str, context: str = "") -> float:
        """
        Evaluate a thought's quality.

        Args:
            thought: The thought text to evaluate
            context: Previous reasoning context

        Returns:
            Quality score between 0 and 1
        """
        # Heuristic evaluation (in production: LLM call)
        score = 0.5

        # Reward specificity
        if any(c.isdigit() for c in thought):
            score += 0.1
        # Reward causal reasoning
        if any(w in thought.lower() for w in ['because', 'therefore', 'since', 'so', 'thus']):
            score += 0.15
        # Reward detail
        if len(thought.split()) > 10:
            score += 0.1
        # Penalize very short thoughts
        if len(thought.split()) < 3:
            score -= 0.2

        # Add some noise to simulate LLM variability
        score += np.random.normal(0, 0.05)
        return np.clip(score, 0.0, 1.0)

    def should_explore(self, value: float) -> bool:
        """Whether a thought is worth exploring further."""
        return value >= self.threshold


################################################################################
# SECTION 4: BFS SEARCH
################################################################################

class BreadthFirstSearch:
    """
    Breadth-First Search for Tree of Thoughts.

    Explore all thoughts at the current depth before going deeper.
    Keep only the top-B most promising thoughts at each level.

    Step by step:
        1. Start with root thought
        2. Generate K candidate next thoughts
        3. Evaluate all K candidates
        4. Keep top-B (beam width) candidates
        5. Expand each of the B candidates
        6. Repeat until depth D or solved

    Interview Question:
        "How does BFS work in Tree of Thoughts?"
        BFS explores all thoughts at one depth before going deeper.
        At each level, generate K candidates, evaluate them, keep top-B
        (beam width). This ensures we don't miss good paths at shallow
        depths but limits exploration to B paths at each level.
    """

    def __init__(self, beam_width: int = 3, max_depth: int = 5):
        """
        Args:
            beam_width: Number of paths to keep at each level (B)
            max_depth: Maximum search depth
        """
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.evaluator = ThoughtEvaluator()
        self.total_evaluations = 0

    def generate_thoughts(self, node: ThoughtNode, k: int = 3) -> List[str]:
        """
        Generate K candidate next thoughts from a node.

        Args:
            node: Current node
            k: Number of candidates to generate

        Returns:
            List of candidate thought strings
        """
        # Simulate thought generation (in production: LLM call)
        thoughts = []
        for i in range(k):
            thoughts.append(f"Thought {i+1} from depth {node.depth}")
        return thoughts

    def search(self, initial_thought: str, target: Optional[str] = None) -> ThoughtTree:
        """
        Run BFS search.

        Args:
            initial_thought: Starting thought
            target: Optional target to find

        Returns:
            ThoughtTree with search results
        """
        tree = ThoughtTree(initial_thought)
        current_level = [tree.root]

        for depth in range(self.max_depth):
            candidates = []

            for node in current_level:
                if node.is_pruned:
                    continue

                # Generate K thoughts
                thoughts = self.generate_thoughts(node)

                # Evaluate each
                for thought in thoughts:
                    value = self.evaluator.evaluate(thought)
                    self.total_evaluations += 1
                    child = tree.add_node(node, thought, value)
                    candidates.append(child)

            if not candidates:
                break

            # Keep top-B candidates
            candidates.sort(key=lambda n: n.value, reverse=True)
            current_level = candidates[:self.beam_width]

            # Prune the rest
            for node in candidates[self.beam_width:]:
                tree.prune_node(node)

        return tree


################################################################################
# SECTION 5: DFS SEARCH
################################################################################

class DepthFirstSearch:
    """
    Depth-First Search for Tree of Thoughts.

    Explore the deepest path first, backtrack when unpromising.

    Step by step:
        1. Start at root
        2. Generate next thought, evaluate it
        3. If promising, go deeper
        4. If unpromising (below threshold), backtrack
        5. Continue until solution found or depth limit

    Interview Question:
        "When would you use DFS vs BFS for ToT?"
        DFS: when solutions are deep, backtracking is cheap, memory is limited.
        BFS: when solutions are shallow, you want completeness, parallelism.
        DFS finds deep solutions faster; BFS guarantees finding shallowest.
    """

    def __init__(self, max_depth: int = 10, backtrack_threshold: float = 0.3):
        """
        Args:
            max_depth: Maximum search depth
            backtrack_threshold: Value below which we backtrack
        """
        self.max_depth = max_depth
        self.backtrack_threshold = backtrack_threshold
        self.evaluator = ThoughtEvaluator(threshold=backtrack_threshold)
        self.total_evaluations = 0
        self.best_path: Optional[ThoughtNode] = None

    def dfs(self, node: ThoughtNode, tree: ThoughtTree) -> bool:
        """
        Recursive DFS from a node.

        Args:
            node: Current node
            tree: The thought tree

        Returns:
            True if solution found, False otherwise
        """
        # Check if we've reached max depth
        if node.depth >= self.max_depth:
            return False

        # Generate next thought
        thought = f"DFS thought at depth {node.depth + 1}"
        value = self.evaluator.evaluate(thought)
        self.total_evaluations += 1

        # Check if we should explore this path
        if not self.evaluator.should_explore(value):
            return False  # Backtrack

        child = tree.add_node(node, thought, value)

        # Update best path
        if self.best_path is None or value > self.best_path.value:
            self.best_path = child

        # Go deeper
        return self.dfs(child, tree)

    def search(self, initial_thought: str) -> ThoughtTree:
        """
        Run DFS search.

        Args:
            initial_thought: Starting thought

        Returns:
            ThoughtTree with search results
        """
        tree = ThoughtTree(initial_thought)
        self.dfs(tree.root, tree)
        return tree


################################################################################
# SECTION 6: TREE OF THOUGHTS (MAIN CLASS)
################################################################################

class TreeOfThoughts:
    """
    Tree of Thoughts — Complete implementation.

    Combines tree structure, evaluation, and search strategies
    for multi-path reasoning.

    Paper: "Tree of Thoughts: Deliberate Problem Solving with
            Large Language Models" (Yao et al., NeurIPS 2023)

    Interview Question:
        "Explain the Tree of Thoughts approach."
        ToT explores multiple reasoning paths like a tree. At each step,
        generate K possible next thoughts, evaluate each one, and keep
        the most promising. Use BFS for breadth or DFS for depth.
        Backtrack from unpromising paths. This enables deliberate
        problem solving rather than linear chain-of-thought.
    """

    def __init__(self, search_strategy: str = 'bfs', beam_width: int = 3,
                 max_depth: int = 5, backtrack_threshold: float = 0.3):
        """
        Args:
            search_strategy: 'bfs' or 'dfs'
            beam_width: Beam width for BFS
            max_depth: Maximum search depth
            backtrack_threshold: Threshold for DFS backtracking
        """
        self.search_strategy = search_strategy
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.backtrack_threshold = backtrack_threshold

    def solve(self, problem: str) -> Dict:
        """
        Solve a problem using Tree of Thoughts.

        Args:
            problem: The problem to solve

        Returns:
            Dictionary with solution path, tree stats, evaluations
        """
        if self.search_strategy == 'bfs':
            searcher = BreadthFirstSearch(self.beam_width, self.max_depth)
        else:
            searcher = DepthFirstSearch(self.max_depth, self.backtrack_threshold)

        tree = searcher.search(f"Problem: {problem}")
        stats = tree.get_stats()

        return {
            'problem': problem,
            'strategy': self.search_strategy,
            'best_path': tree.get_best_path(),
            'stats': stats,
            'total_evaluations': searcher.total_evaluations
        }


################################################################################
# SECTION 7: TESTING & DEMONSTRATION
################################################################################

def demonstrate_tree_of_thoughts():
    """
    Demonstrate Tree of Thoughts with concrete examples.
    """
    print("=" * 70)
    print("TREE OF THOUGHTS DEMONSTRATION")
    print("=" * 70)

    # ── ThoughtNode ──
    print("\n1. THOUGHT NODE")
    print("-" * 40)
    root = ThoughtNode("I need to solve this math problem", value=0.5)
    child1 = ThoughtNode("Let me try multiplication", value=0.7, parent=root, depth=1)
    child2 = ThoughtNode("Let me try addition", value=0.4, parent=root, depth=1)
    root.add_child(child1)
    root.add_child(child2)
    print(f"Root: {root}")
    print(f"Child 1 path: {child1.get_path()}")
    print(f"Child 1 path value: {child1.get_path_value():.3f}")

    # ── ThoughtTree ──
    print("\n2. THOUGHT TREE")
    print("-" * 40)
    tree = ThoughtTree("Start problem solving")
    n1 = tree.add_node(tree.root, "Step 1: Analyze", 0.6)
    n2 = tree.add_node(tree.root, "Step 1: Decompose", 0.8)
    n3 = tree.add_node(n2, "Step 2: Solve sub-problem A", 0.7)
    n4 = tree.add_node(n2, "Step 2: Solve sub-problem B", 0.9)
    print(f"Best path: {tree.get_best_path()}")
    print(f"Stats: {tree.get_stats()}")

    # ── BFS Search ──
    print("\n3. BFS SEARCH")
    print("-" * 40)
    bfs = BreadthFirstSearch(beam_width=2, max_depth=3)
    tree = bfs.search("Solve: What is 15 * 23?")
    print(f"Total evaluations: {bfs.total_evaluations}")
    print(f"Best path: {tree.get_best_path()}")
    print(f"Tree stats: {tree.get_stats()}")

    # ── DFS Search ──
    print("\n4. DFS SEARCH")
    print("-" * 40)
    dfs = DepthFirstSearch(max_depth=5, backtrack_threshold=0.3)
    tree = dfs.search("Solve: What is 15 * 23?")
    print(f"Total evaluations: {dfs.total_evaluations}")
    print(f"Best path: {tree.get_best_path()}")
    print(f"Tree stats: {tree.get_stats()}")

    # ── Complete ToT ──
    print("\n5. COMPLETE TREE OF THOUGHTS")
    print("-" * 40)
    for strategy in ['bfs', 'dfs']:
        tot = TreeOfThoughts(search_strategy=strategy, beam_width=3, max_depth=4)
        result = tot.solve("Arrange numbers to make 24: 4 5 6 7")
        print(f"\nStrategy: {strategy}")
        print(f"  Evaluations: {result['total_evaluations']}")
        print(f"  Path length: {len(result['best_path'])}")
        print(f"  Stats: {result['stats']}")

    # ── Comparison ──
    print("\n6. BFS vs DFS COMPARISON")
    print("-" * 40)
    print("BFS: Explores breadth-first, keeps top-B at each level")
    print("  Good for: shallow solutions, parallel exploration")
    print("  Cost: B * K evaluations per level")
    print("DFS: Explores depth-first, backtracks when stuck")
    print("  Good for: deep solutions, memory-efficient")
    print("  Cost: proportional to path length")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_tree_of_thoughts()
