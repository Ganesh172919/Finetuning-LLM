"""
################################################################################
REASONING MODELS — THE NEXT FRONTIER OF AI
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Reasoning Models?
    Reasoning models are LLMs that can think step-by-step before answering.
    Instead of immediately producing an answer, they:
    1. Analyze the problem
    2. Consider different approaches
    3. Work through the solution
    4. Verify their answer

    This is similar to how humans think through complex problems.

Why do they matter?
    Standard LLMs often fail at:
    - Multi-step math problems
    - Complex logic puzzles
    - Planning and scheduling
    - Code debugging
    - Scientific reasoning

    Reasoning models excel at these tasks by "thinking out loud."

Historical Evolution:
    - 2022: Chain of Thought (CoT) prompting
    - 2023: Tree of Thoughts (ToT), Self-Consistency
    - 2024: OpenAI o1, test-time compute scaling
    - 2025: DeepSeek R1, reasoning training
    - 2026: Hybrid reasoning models

Key Insight:
    You can improve reasoning by:
    1. Better prompting (CoT, ToT)
    2. Better training (RLHF for reasoning)
    3. More test-time compute (thinking longer)

########################################

REASONING APPROACHES:

1. Chain of Thought (CoT): Think step by step
2. Self-Consistency: Multiple attempts, vote on answer
3. Tree of Thoughts: Explore multiple reasoning paths
4. Reasoning Tokens: Special tokens for thinking
5. Test-Time Compute: Spend more compute at inference
6. Reasoning Training: Train models to reason

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
import math
from dataclasses import dataclass

import sys
sys.path.append('..')
from ..01_math.probability import Categorical


################################################################################
# SECTION 1: CHAIN OF THOUGHT (CoT)
################################################################################

class ChainOfThought:
    """
    Chain of Thought Prompting
    ===========================

    Definition: Instead of asking for the answer directly, ask the model
    to think step by step.

    Standard Prompt:
        Q: Roger has 5 tennis balls. He buys 2 more cans of 3. How many does he have?
        A: 11

    CoT Prompt:
        Q: Roger has 5 tennis balls. He buys 2 more cans of 3. How many does he have?
        A: Let me think step by step.
           1. Roger starts with 5 balls
           2. He buys 2 cans of 3 balls each = 2 × 3 = 6 balls
           3. Total = 5 + 6 = 11 balls
           The answer is 11.

    Why it works:
    1. Breaks complex problems into simpler steps
    2. Allows the model to "show its work"
    3. Each step can be verified
    4. Reduces errors from "jumping to conclusions"

    Research Results:
    - GSM8K (math): 17% → 58% accuracy with CoT
    - MultiArith: 52% → 88% accuracy with CoT
    - Large models benefit more than small models

    Interview Questions:
        1. "What is Chain of Thought prompting?"
           A technique where you ask the model to explain its reasoning
           step by step before giving the final answer.

        2. "When should I use CoT?"
           For complex reasoning tasks: math, logic, planning.
           Not needed for simple factual questions.

        3. "Does CoT work for all models?"
           Works best for large models (>100B parameters).
           Smaller models may produce incorrect reasoning chains.
    """

    @staticmethod
    def create_prompt(question: str) -> str:
        """
        Create a Chain of Thought prompt.

        The key is to ask the model to think step by step.
        """
        return f"""Question: {question}

Let me think through this step by step:

1. First, I need to understand what the question is asking.
2. Then, I'll identify the relevant information.
3. Next, I'll work through the solution.
4. Finally, I'll verify my answer.

Step-by-step reasoning:"""

    @staticmethod
    def create_few_shot_prompt(question: str, examples: List[Dict[str, str]]) -> str:
        """
        Create a few-shot CoT prompt with examples.

        Examples should include:
        - question: The problem
        - reasoning: Step-by-step reasoning
        - answer: The final answer
        """
        prompt = "I'll solve each problem by thinking step by step.\n\n"

        for ex in examples:
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Reasoning: {ex['reasoning']}\n"
            prompt += f"Answer: {ex['answer']}\n\n"

        prompt += f"Question: {question}\n"
        prompt += "Reasoning:"
        return prompt


################################################################################
# SECTION 2: SELF-CONSISTENCY
################################################################################

class SelfConsistency:
    """
    Self-Consistency Decoding
    ===========================

    Definition: Generate multiple reasoning paths and take a majority vote
    on the final answer.

    Algorithm:
    1. Sample N different completions (with temperature > 0)
    2. Extract the answer from each completion
    3. Take the majority vote as the final answer

    Why it works:
    - Different reasoning paths may lead to the same correct answer
    - Errors are often random, but correct answers are consistent
    - Majority voting filters out incorrect reasoning

    Example:
        Question: What is 15 × 17?

        Path 1: 15 × 17 = 15 × (10 + 7) = 150 + 105 = 255 ✓
        Path 2: 15 × 17 = 15 × 20 - 15 × 3 = 300 - 45 = 255 ✓
        Path 3: 15 × 17 = 17 × 10 + 17 × 5 = 170 + 85 = 255 ✓
        Path 4: 15 × 17 = 15 × 15 + 15 × 2 = 225 + 30 = 255 ✓
        Path 5: 15 × 17 = 15 × 17 = 245 ✗ (error)

        Majority vote: 255 (4/5 paths agree)

    Research Results:
    - GSM8K: 58% → 74% with self-consistency
    - Works even better with CoT prompting

    Interview Question:
        "What is self-consistency and when should I use it?"
        It's a technique where you sample multiple reasoning paths
        and take a majority vote on the answer. Use it when you need
        higher accuracy and can afford more compute.
    """

    def __init__(self, n_paths: int = 5, temperature: float = 0.7):
        self.n_paths = n_paths
        self.temperature = temperature

    def generate_paths(
        self,
        model,
        prompt: str,
        max_new_tokens: int = 200
    ) -> List[str]:
        """
        Generate multiple reasoning paths.

        In practice, this would call the model n_paths times
        with different random seeds or temperatures.
        """
        paths = []
        for _ in range(self.n_paths):
            # In real implementation:
            # path = model.generate(prompt, temperature=self.temperature)
            paths.append(f"Reasoning path {len(paths) + 1}")
        return paths

    def extract_answer(self, reasoning: str) -> Optional[str]:
        """
        Extract the final answer from a reasoning path.

        Looks for patterns like:
        - "The answer is X"
        - "Therefore, X"
        - "= X"
        - "Answer: X"
        """
        import re

        patterns = [
            r'(?:the answer is|answer:|therefore|thus|hence|so)[:\s]*(\d+)',
            r'=\s*(\d+)\s*$',
            r'(\d+)\s*(?:is the answer|\.?\s*$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, reasoning.lower())
            if match:
                return match.group(1)

        return None

    def vote(self, answers: List[str]) -> str:
        """
        Take majority vote on answers.

        Returns the most common answer.
        """
        from collections import Counter
        counter = Counter(answers)
        return counter.most_common(1)[0][0]


################################################################################
# SECTION 3: TREE OF THOUGHTS (ToT)
################################################################################

class TreeOfThoughts:
    """
    Tree of Thoughts (ToT)
    ======================

    Definition: Instead of a single reasoning chain, explore a TREE of
    possible reasoning paths and select the best one.

    Algorithm:
    1. Generate multiple initial thoughts
    2. Evaluate each thought (how promising is it?)
    3. Expand the most promising thoughts
    4. Repeat until solution found or budget exhausted
    5. Select the best final thought

    Visual:
                         Problem
                        /   |   \
                    Thought1 Thought2 Thought3
                    /   \      |       /   \
                  T1a   T1b   T2a    T3a   T3b
                  |     |      |      |      |
                [eval] [eval] [eval] [eval] [eval]
                  ↓     ↓      ↓      ↓      ↓
                 Best path selected

    Compared to CoT:
    - CoT: single path (linear)
    - ToT: multiple paths (tree)

    When to use:
    - Problems with multiple possible approaches
    - When early mistakes can be recovered from
    - When you can evaluate intermediate states

    Interview Question:
        "What's the difference between CoT and ToT?"
        CoT follows a single reasoning chain.
        ToT explores multiple chains and selects the best one.
        ToT is better for problems where you might take a wrong turn
        and need to backtrack.
    """

    def __init__(
        self,
        n_branches: int = 3,
        max_depth: int = 5,
        exploration_weight: float = 1.0
    ):
        self.n_branches = n_branches
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight

    @dataclass
    class ThoughtNode:
        """A node in the thought tree."""
        thought: str
        score: float
        depth: int
        parent: Optional['TreeOfThoughts.ThoughtNode'] = None
        children: List['TreeOfThoughts.ThoughtNode'] = None

        def __post_init__(self):
            if self.children is None:
                self.children = []

    def generate_thoughts(
        self,
        current_thought: str,
        n: int = 3
    ) -> List[str]:
        """
        Generate next thoughts from current thought.

        In practice, this would use the LLM to generate continuations.
        """
        # Placeholder: generate variations
        thoughts = []
        for i in range(n):
            thoughts.append(f"{current_thought} → step {i+1}")
        return thoughts

    def evaluate_thought(self, thought: str) -> float:
        """
        Evaluate how promising a thought is.

        Returns a score between 0 and 1.
        In practice, this would use the LLM to evaluate.
        """
        # Placeholder: random score
        return np.random.random()

    def solve(self, problem: str) -> Tuple[str, List[str]]:
        """
        Solve a problem using Tree of Thoughts.

        Returns:
            solution: The best solution found
            path: The reasoning path taken
        """
        # Initialize root
        root = self.ThoughtNode(
            thought=problem,
            score=0.0,
            depth=0
        )

        # Best solution found
        best_solution = None
        best_score = -1

        # BFS with pruning
        frontier = [root]

        for depth in range(self.max_depth):
            if not frontier:
                break

            # Expand all nodes at current depth
            next_frontier = []
            for node in frontier:
                # Generate next thoughts
                thoughts = self.generate_thoughts(
                    node.thought, self.n_branches
                )

                for thought in thoughts:
                    # Evaluate thought
                    score = self.evaluate_thought(thought)

                    # Create child node
                    child = self.ThoughtNode(
                        thought=thought,
                        score=score,
                        depth=depth + 1,
                        parent=node
                    )
                    node.children.append(child)

                    # Update best solution
                    if score > best_score:
                        best_score = score
                        best_solution = child

                    next_frontier.append(child)

            # Keep only best nodes (pruning)
            next_frontier.sort(key=lambda x: x.score, reverse=True)
            frontier = next_frontier[:self.n_branches * 2]

        # Extract path
        path = []
        node = best_solution
        while node:
            path.append(node.thought)
            node = node.parent
        path.reverse()

        return best_solution.thought, path


################################################################################
# SECTION 4: REASONING TOKENS
################################################################################

class ReasoningTokens:
    """
    Reasoning Tokens (Test-Time Compute)
    ======================================

    Definition: Special tokens that represent "thinking" steps.
    The model generates these tokens before producing the answer.

    How it works:
    1. Model generates <thinking> tokens
    2. These tokens are not shown to the user
    3. Model uses them to reason about the problem
    4. After <thinking>, model produces the answer

    This is how OpenAI o1 and DeepSeek R1 work:
    - The model "thinks" for many tokens
    - The thinking is hidden from the user
    - Only the final answer is shown

    Benefits:
    - More compute = better answers (scaling test-time compute)
    - Can think through complex problems
    - Can verify and correct mistakes

    Training:
    - Train with RLHF to produce good reasoning chains
    - Reward correct final answers
    - Penalize unnecessary thinking

    Interview Questions:
        1. "What are reasoning tokens?"
           Special tokens that represent internal thinking.
           The model generates them before producing the answer.
           They allow the model to "think out loud" before answering.

        2. "How does test-time compute scaling work?"
           By allowing the model to generate more thinking tokens,
           it can solve harder problems. More thinking = better answers,
           but slower inference.

        3. "How do you train reasoning models?"
           Use RLHF with rewards for correct answers.
           The model learns to produce useful reasoning chains
           that lead to correct answers.
    """

    # Special tokens for reasoning
    THINK_START = "<thinking>"
    THINK_END = "</thinking>"
    STEP = "<step>"
    VERIFY = "<verify>"
    ANSWER = "<answer>"

    @staticmethod
    def create_reasoning_prompt(question: str) -> str:
        """
        Create a prompt that encourages reasoning.
        """
        return f"""Question: {question}

{ReasoningTokens.THINK_START}
Let me think through this carefully.

{ReasoningTokens.STEP} 1: Understand the problem
{ReasoningTokens.STEP} 2: Identify relevant information
{ReasoningTokens.STEP} 3: Work through the solution
{ReasoningTokens.STEP} 4: Verify the answer
{ReasoningTokens.VERIFY} Let me double-check...

{ReasoningTokens.THINK_END}

{ReasoningTokens.ANSWER}"""

    @staticmethod
    def extract_answer(generated_text: str) -> Optional[str]:
        """
        Extract the answer from generated text.

        Looks for content after <answer> tag.
        """
        import re
        match = re.search(
            f'{ReasoningTokens.ANSWER}\\s*(.*?)(?:$|\\n)',
            generated_text
        )
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def extract_thinking(generated_text: str) -> Optional[str]:
        """
        Extract the thinking/reasoning from generated text.

        Looks for content between <thinking> and </thinking> tags.
        """
        import re
        match = re.search(
            f'{ReasoningTokens.THINK_START}(.*?){ReasoningTokens.THINK_END}',
            generated_text,
            re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return None


################################################################################
# SECTION 5: REASONING TRAINING (RLHF for Reasoning)
################################################################################

class ReasoningTrainer:
    """
    Training Reasoning Models with RLHF
    =====================================

    Definition: Train models to produce better reasoning chains
    using reinforcement learning from human feedback.

    Training Process:
    1. Collect reasoning examples (human-written)
    2. Train reward model to score reasoning quality
    3. Use RL (PPO) to optimize model for high reward
    4. Iterate: generate → evaluate → improve

    Reward Components:
    1. Correctness: Is the final answer correct?
    2. Reasoning quality: Is the reasoning logical?
    3. Efficiency: Is the reasoning concise?
    4. Verification: Did the model check its work?

    DeepSeek R1 Approach:
    - Start with base LLM
    - Train on reasoning examples
    - Use RL to improve reasoning
    - Result: Model that can "think" before answering

    Interview Question:
        "How do you train a reasoning model?"
        1. Collect high-quality reasoning examples
        2. Train a reward model to score reasoning
        3. Use RL (PPO/GRPO) to optimize the model
        4. The model learns to produce reasoning that leads to correct answers
    """

    def __init__(
        self,
        reward_model_weight: float = 1.0,
        kl_penalty_weight: float = 0.1,
        correctness_weight: float = 2.0,
        efficiency_weight: float = 0.5
    ):
        self.reward_model_weight = reward_model_weight
        self.kl_penalty_weight = kl_penalty_weight
        self.correctness_weight = correctness_weight
        self.efficiency_weight = efficiency_weight

    def compute_reward(
        self,
        reasoning: str,
        answer: str,
        correct_answer: str
    ) -> float:
        """
        Compute reward for a reasoning chain.

        Args:
            reasoning: The model's reasoning
            answer: The model's answer
            correct_answer: The correct answer

        Returns:
            reward: Combined reward score
        """
        reward = 0.0

        # Correctness reward
        is_correct = answer.strip().lower() == correct_answer.strip().lower()
        if is_correct:
            reward += self.correctness_weight

        # Efficiency reward (penalize overly long reasoning)
        reasoning_length = len(reasoning.split())
        if reasoning_length < 50:
            reward += self.efficiency_weight
        elif reasoning_length > 200:
            reward -= self.efficiency_weight * 0.5

        # Check for verification step
        if any(word in reasoning.lower() for word in ['verify', 'check', 'confirm']):
            reward += 0.2

        return reward


################################################################################
# SECTION 6: TESTING & EXAMPLES
################################################################################

def demonstrate_reasoning():
    """Demonstrate reasoning techniques."""
    print("=" * 70)
    print("REASONING MODELS DEMONSTRATION")
    print("=" * 70)

    # Chain of Thought
    print("\n--- Chain of Thought ---")
    cot = ChainOfThought()
    question = "Roger has 5 tennis balls. He buys 2 more cans of 3. How many does he have?"
    prompt = cot.create_prompt(question)
    print(f"Prompt:\n{prompt[:200]}...")

    # Self-Consistency
    print("\n--- Self-Consistency ---")
    sc = SelfConsistency(n_paths=5)
    answers = ["255", "255", "245", "255", "255"]
    majority = sc.vote(answers)
    print(f"Answers: {answers}")
    print(f"Majority vote: {majority}")

    # Tree of Thoughts
    print("\n--- Tree of Thoughts ---")
    tot = TreeOfThoughts(n_branches=3, max_depth=3)
    solution, path = tot.solve("What is 15 × 17?")
    print(f"Solution: {solution}")
    print(f"Path length: {len(path)}")

    # Reasoning Tokens
    print("\n--- Reasoning Tokens ---")
    prompt = ReasoningTokens.create_reasoning_prompt("What is 2+2?")
    print(f"Prompt:\n{prompt[:300]}...")

    # Reasoning Trainer
    print("\n--- Reasoning Training ---")
    trainer = ReasoningTrainer()
    reward = trainer.compute_reward(
        reasoning="Step 1: 5 balls. Step 2: 2×3=6. Step 3: 5+6=11. Verify: 5+6=11 ✓",
        answer="11",
        correct_answer="11"
    )
    print(f"Reward for correct reasoning: {reward}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_reasoning()


################################################################################
# REFERENCES
################################################################################

# [1] Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in LLMs.
# [2] Wang, X., et al. (2023). Self-Consistency Improves CoT Reasoning in LLMs.
# [3] Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with LLMs.
# [4] OpenAI. (2024). Learning to Reason with LLMs (o1 blog post).
# [5] DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs.

################################################################################
