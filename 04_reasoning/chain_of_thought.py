"""
################################################################################
CHAIN OF THOUGHT REASONING — THINKING STEP BY STEP
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Chain of Thought (CoT)?
    Chain of Thought is a prompting technique where we ask the model to
    reason step by step before giving a final answer. Instead of jumping
    directly to the answer, the model shows its reasoning process, which
    dramatically improves accuracy on complex tasks.

Why does it matter?
    Standard prompting fails on multi-step problems:
    - Math: "If I have 5 apples and buy 3 more, then give away 2..." → often wrong
    - Logic: "All cats are animals. Fluffy is a cat. Is Fluffy an animal?" → sometimes fails
    - Planning: Multi-step tasks require decomposition

    CoT prompting improves accuracy:
    - GSM8K (math): 17.7% → 58.1% with CoT (Wei et al., 2022)
    - MultiArith: 52.4% → 88.7% with CoT
    - Larger models benefit more from CoT

How does it work?
    1. Zero-Shot CoT: Append "Let's think step by step" to the prompt
    2. Few-Shot CoT: Provide examples with reasoning chains
    3. Auto-CoT: Automatically generate diverse reasoning exemplars
    4. Self-Ask: Decompose questions into sub-questions
    5. Multimodal CoT: Generate visual rationale + textual answer

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ Chain of Thought Pipeline                                   │
    │                                                              │
    │  Question ──▶ [Add CoT Trigger] ──▶ [Generate Reasoning]    │
    │                                          ↓                   │
    │                                    [Extract Steps]           │
    │                                          ↓                   │
    │                                    [Verify Steps]            │
    │                                          ↓                   │
    │                                    [Extract Answer]          │
    └─────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2022: Chain of Thought Prompting (Wei et al., Google Brain)
    - 2022: Self-Consistency (Wang et al.) — sample multiple CoT paths
    - 2022: Zero-Shot CoT (Kojima et al.) — "Let's think step by step"
    - 2023: Auto-CoT (Zhang et al.) — automatic exemplar generation
    - 2023: Tree of Thoughts (Yao et al.) — branching reasoning
    - 2024: Multimodal CoT — vision + text reasoning
    - 2025: Reasoning training (DeepSeek R1) — train CoT into model

INTERVIEW QUESTIONS:
    1. "What is Chain of Thought and why does it work?"
       CoT is a prompting technique where the model reasons step by step.
       It works because: (a) decomposes complex problems into simpler steps,
       (b) allows intermediate computation, (c) provides interpretable
       reasoning, (d) reduces errors from jumping to conclusions.

    2. "What's the difference between Zero-Shot and Few-Shot CoT?"
       Zero-Shot CoT appends "Let's think step by step" without examples.
       Few-Shot CoT provides example Q→reasoning→answer pairs.
       Few-Shot is generally more accurate but requires exemplar selection.

    3. "When does CoT NOT help?"
       CoT helps least on: simple tasks (overhead not worth it),
       tasks requiring world knowledge (reasoning chain may be wrong),
       small models (<10B parameters often can't CoT well).

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
import math
from dataclasses import dataclass, field

import sys
sys.path.append('..')
from ..01_math.probability import softmax


################################################################################
# SECTION 1: CONFIGURATION
################################################################################

@dataclass
class CoTConfig:
    """
    Chain of Thought Configuration
    ===============================

    Controls the behavior of CoT reasoning methods.

    Attributes:
        temperature: Sampling temperature for generation (higher = more diverse)
        max_reasoning_steps: Maximum number of reasoning steps to generate
        num_exemplars: Number of few-shot examples to use
        trigger_phrase: Zero-shot CoT trigger (e.g., "Let's think step by step")
        verify_steps: Whether to verify each reasoning step
        extract_answer: Whether to extract final answer from reasoning
    """
    temperature: float = 0.7
    max_reasoning_steps: int = 10
    num_exemplars: int = 4
    trigger_phrase: str = "Let's think step by step."
    verify_steps: bool = True
    extract_answer: bool = True


################################################################################
# SECTION 2: ZERO-SHOT CHAIN OF THOUGHT
################################################################################

class ZeroShotCoT:
    """
    Zero-Shot Chain of Thought
    ===========================

    The simplest CoT method: append a trigger phrase to the prompt
    to elicit step-by-step reasoning without any examples.

    Paper: "Large Language Models are Zero-Shot Reasoners"
           (Kojima et al., NeurIPS 2022)

    Key Finding:
        Simply adding "Let's think step by step" to a prompt improves
        math reasoning from 17.7% to 78.7% on some benchmarks.

    How it works:
        1. Take the original question
        2. Append trigger phrase: "Let's think step by step."
        3. Generate reasoning chain (model produces steps)
        4. Extract final answer from reasoning

    Step by step:
        Original: "Q: What is 23 * 47? A:"
        CoT:      "Q: What is 23 * 47? A: Let's think step by step."
        Model:    "23 * 47 = 23 * (50 - 3) = 23*50 - 23*3 = 1150 - 69 = 1081.
                   The answer is 1081."

    Interview Question:
        "How does Zero-Shot CoT work?"
        Simply append "Let's think step by step" to the prompt. This triggers
        the model to generate intermediate reasoning steps before the answer.
        It works because models have seen step-by-step reasoning in training
        data, and the trigger activates that pattern.
    """

    def __init__(self, config: Optional[CoTConfig] = None):
        """
        Initialize Zero-Shot CoT.

        Args:
            config: CoT configuration (uses defaults if None)
        """
        self.config = config or CoTConfig()
        # Simulated token probabilities for demonstration
        self.step_tokens = [
            "First", "Next", "Then", "After that", "Finally",
            "Step 1", "Step 2", "Step 3", "Therefore", "So"
        ]
        self.answer_tokens = [
            "The answer is", "Therefore,", "So the answer is",
            "In conclusion,", "The result is"
        ]

    def add_trigger(self, question: str) -> str:
        """
        Add CoT trigger phrase to question.

        Args:
            question: Original question

        Returns:
            Question with trigger appended

        Example:
            >>> cot = ZeroShotCoT()
            >>> cot.add_trigger("What is 2+2?")
            'What is 2+2? Let's think step by step.'
        """
        return f"{question} {self.config.trigger_phrase}"

    def generate_reasoning(self, question: str) -> List[str]:
        """
        Generate reasoning steps for a question.

        In a real system, this calls an LLM. Here we simulate the
        process of generating step-by-step reasoning.

        Args:
            question: The question to reason about

        Returns:
            List of reasoning steps (strings)
        """
        # Simulate generating reasoning steps
        # In production, this would be: model.generate(cot_prompt)
        steps = []
        for i in range(self.config.max_reasoning_steps):
            # Simulate step generation with temperature sampling
            probs = np.random.dirichlet(np.ones(len(self.step_tokens)))
            probs = probs ** (1.0 / self.config.temperature)
            probs = probs / probs.sum()
            step_idx = np.random.choice(len(self.step_tokens), p=probs)
            step = self.step_tokens[step_idx]
            steps.append(f"Step {i+1}: {step}...")
        return steps

    def extract_answer(self, reasoning: List[str]) -> str:
        """
        Extract final answer from reasoning chain.

        The answer is typically in the last step or after an
        answer trigger like "The answer is".

        Args:
            reasoning: List of reasoning steps

        Returns:
            Extracted answer string
        """
        if not reasoning:
            return "No reasoning generated"
        # In production, parse the last step for answer pattern
        last_step = reasoning[-1]
        return last_step.replace(f"Step {len(reasoning)}: ", "")

    def reason(self, question: str) -> Dict:
        """
        Full Zero-Shot CoT pipeline.

        Args:
            question: The question to answer

        Returns:
            Dictionary with question, trigger, reasoning, answer

        Example:
            >>> cot = ZeroShotCoT()
            >>> result = cot.reason("What is 15 * 23?")
            >>> print(result['trigger'])
            Let's think step by step.
        """
        # Step 1: Add trigger
        cot_prompt = self.add_trigger(question)

        # Step 2: Generate reasoning
        reasoning = self.generate_reasoning(cot_prompt)

        # Step 3: Extract answer
        answer = self.extract_answer(reasoning)

        return {
            'question': question,
            'trigger': self.config.trigger_phrase,
            'cot_prompt': cot_prompt,
            'reasoning': reasoning,
            'answer': answer
        }


################################################################################
# SECTION 3: FEW-SHOT CHAIN OF THOUGHT
################################################################################

class FewShotCoT:
    """
    Few-Shhot Chain of Thought
    ===========================

    Provide example Q→reasoning→answer pairs before asking the new question.
    The model learns the reasoning pattern from examples.

    Paper: "Chain-of-Thought Prompting Elicits Reasoning in Large
            Language Models" (Wei et al., NeurIPS 2022)

    Key Finding:
        Few-Shot CoT on PaLM 540B achieves 73.5% on GSM8K (vs 17.7% standard).
        Quality of exemplars matters enormously.

    Step by step:
        1. Select N exemplar questions with known reasoning chains
        2. Format as: Q: ... A: Step 1: ... Step 2: ... The answer is ...
        3. Append the new question
        4. Model generates reasoning following the exemplar pattern

    Interview Question:
        "How do you select good CoT exemplars?"
        (a) Diversity: cover different reasoning patterns
        (b) Similarity: exemplars should be similar to test question
        (c) Correctness: exemplar reasoning must be correct
        (d) Complexity: include both simple and complex examples
    """

    def __init__(self, config: Optional[CoTConfig] = None):
        """Initialize Few-Shot CoT."""
        self.config = config or CoTConfig()
        self.exemplars: List[Dict] = []

    def add_exemplar(self, question: str, reasoning: List[str], answer: str):
        """
        Add a CoT exemplar (example with reasoning chain).

        Args:
            question: Example question
            reasoning: Step-by-step reasoning (list of strings)
            answer: Correct answer
        """
        self.exemplars.append({
            'question': question,
            'reasoning': reasoning,
            'answer': answer
        })

    def format_exemplar(self, exemplar: Dict) -> str:
        """
        Format a single exemplar as a CoT prompt.

        Args:
            exemplar: Dictionary with question, reasoning, answer

        Returns:
            Formatted exemplar string
        """
        reasoning_str = " ".join(exemplar['reasoning'])
        return (
            f"Q: {exemplar['question']}\n"
            f"A: {reasoning_str} "
            f"The answer is {exemplar['answer']}."
        )

    def select_exemplars(self, question: str, n: Optional[int] = None) -> List[Dict]:
        """
        Select the best exemplars for a given question.

        Uses similarity-based selection: choose exemplars whose
        questions are most similar to the target question.

        Args:
            question: Target question
            n: Number of exemplars to select (default from config)

        Returns:
            Selected exemplars
        """
        n = n or self.config.num_exemplars
        if len(self.exemplars) <= n:
            return self.exemplars

        # Simple similarity: shared word count (in production, use embeddings)
        q_words = set(question.lower().split())
        scores = []
        for ex in self.exemplars:
            ex_words = set(ex['question'].lower().split())
            overlap = len(q_words & ex_words)
            scores.append(overlap)

        # Select top-n by similarity
        indices = np.argsort(scores)[-n:]
        return [self.exemplars[i] for i in indices]

    def build_prompt(self, question: str) -> str:
        """
        Build the full Few-Shot CoT prompt.

        Args:
            question: Question to answer

        Returns:
            Complete prompt with exemplars + question
        """
        selected = self.select_exemplars(question)
        parts = [self.format_exemplar(ex) for ex in selected]
        parts.append(f"Q: {question}\nA:")
        return "\n\n".join(parts)

    def reason(self, question: str) -> Dict:
        """
        Full Few-Shot CoT pipeline.

        Args:
            question: The question to answer

        Returns:
            Dictionary with prompt, selected exemplars, reasoning, answer
        """
        prompt = self.build_prompt(question)

        # Simulate generation (in production: model.generate(prompt))
        selected = self.select_exemplars(question)
        # Generate reasoning following exemplar pattern
        reasoning = [f"Following the pattern from examples..."]
        answer = "simulated_answer"

        return {
            'question': question,
            'prompt': prompt,
            'num_exemplars': len(selected),
            'reasoning': reasoning,
            'answer': answer
        }


################################################################################
# SECTION 4: AUTO CHAIN OF THOUGHT
################################################################################

class AutoCoT:
    """
    Auto Chain of Thought (Auto-CoT)
    ==================================

    Automatically generate CoT exemplars instead of manually crafting them.
    This eliminates the need for human-written reasoning chains.

    Paper: "Automatic Chain of Thought Prompting in Large Language Models"
           (Zhang et al., ICLR 2023)

    Key Insight:
        Manual CoT exemplars are expensive to create and may introduce bias.
        Auto-CoT generates diverse exemplars automatically.

    How it works:
        1. Cluster questions by semantic similarity
        2. Select a representative question from each cluster
        3. Generate reasoning chain for each representative
        4. Use generated chains as exemplars

    Step by step:
        1. Given a question bank, embed all questions
        2. Cluster into K groups (K = number of desired exemplars)
        3. Pick the question closest to each cluster center
        4. Generate CoT reasoning for each selected question
        5. Use these as Few-Shot CoT exemplars

    Interview Question:
        "What is Auto-CoT and why is it useful?"
        Auto-CoT automatically generates CoT exemplars by clustering
        questions and generating reasoning for cluster representatives.
        It's useful because: (a) no manual exemplar crafting needed,
        (b) produces diverse exemplars, (c) reduces human bias in
        exemplar selection.
    """

    def __init__(self, config: Optional[CoTConfig] = None):
        """Initialize Auto-CoT."""
        self.config = config or CoTConfig()

    def cluster_questions(self, questions: List[str], n_clusters: int) -> List[List[int]]:
        """
        Cluster questions by similarity.

        Simple clustering based on word overlap. In production,
        use embedding-based clustering (K-Means on sentence embeddings).

        Args:
            questions: List of questions to cluster
            n_clusters: Number of clusters

        Returns:
            List of clusters, each containing question indices
        """
        n = len(questions)
        if n <= n_clusters:
            return [[i] for i in range(n)]

        # Build similarity matrix (word overlap)
        word_sets = [set(q.lower().split()) for q in questions]
        similarity = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    overlap = len(word_sets[i] & word_sets[j])
                    union = len(word_sets[i] | word_sets[j])
                    similarity[i, j] = overlap / max(union, 1)

        # Simple K-Means-like clustering
        # Initialize centroids as most central points
        centrality = similarity.sum(axis=1)
        centroid_indices = np.argsort(centrality)[-n_clusters:]

        # Assign each question to nearest centroid
        clusters = [[] for _ in range(n_clusters)]
        for i in range(n):
            best_cluster = np.argmax([similarity[i, c] for c in centroid_indices])
            clusters[best_cluster].append(i)

        return clusters

    def select_representatives(self, questions: List[str],
                                clusters: List[List[int]]) -> List[int]:
        """
        Select the most representative question from each cluster.

        The representative is the question closest to the cluster center.

        Args:
            questions: All questions
            clusters: Cluster assignments

        Returns:
            Indices of representative questions
        """
        representatives = []
        for cluster in clusters:
            if not cluster:
                continue
            # Pick the question with highest average similarity to cluster
            best_idx = cluster[0]
            best_score = -1
            for idx in cluster:
                words_i = set(questions[idx].lower().split())
                score = sum(
                    len(words_i & set(questions[j].lower().split()))
                    for j in cluster if j != idx
                )
                if score > best_score:
                    best_score = score
                    best_idx = idx
            representatives.append(best_idx)
        return representatives

    def generate_cot_for_question(self, question: str) -> Dict:
        """
        Generate a CoT reasoning chain for a question.

        In production, this calls an LLM with Zero-Shot CoT.
        Here we simulate the generation.

        Args:
            question: Question to generate reasoning for

        Returns:
            Dictionary with question, reasoning, answer
        """
        # Simulate CoT generation
        reasoning = [
            f"Let me analyze: {question}",
            "Breaking this down into sub-problems...",
            "Solving each part step by step...",
            "Combining the results..."
        ]
        answer = "generated_answer"
        return {
            'question': question,
            'reasoning': reasoning,
            'answer': answer
        }

    def auto_generate_exemplars(self, questions: List[str]) -> List[Dict]:
        """
        Full Auto-CoT pipeline: cluster, select, generate.

        Args:
            questions: Question bank to generate exemplars from

        Returns:
            List of generated exemplars (question, reasoning, answer)
        """
        # Step 1: Cluster questions
        n_clusters = min(self.config.num_exemplars, len(questions))
        clusters = self.cluster_questions(questions, n_clusters)

        # Step 2: Select representatives
        rep_indices = self.select_representatives(questions, clusters)

        # Step 3: Generate CoT for each representative
        exemplars = []
        for idx in rep_indices:
            exemplar = self.generate_cot_for_question(questions[idx])
            exemplars.append(exemplar)

        return exemplars


################################################################################
# SECTION 5: SELF-ASK CHAIN OF THOUGHT
################################################################################

class SelfAskCoT:
    """
    Self-Ask Chain of Thought
    ===========================

    The model asks itself sub-questions, answers each one, then
    composes the final answer from sub-answers.

    Paper: "Measuring and Narrowing the Compositionality Gap in
            Language Models" (Press et al., 2022)

    Key Insight:
        Complex questions require composition of multiple facts.
        Self-Ask decomposes the question into simpler sub-questions
        that can be answered independently.

    How it works:
        1. Given a complex question
        2. Model generates sub-questions
        3. Model answers each sub-question
        4. Model composes final answer from sub-answers

    Example:
        Q: "Who was the president of the US when Apollo 11 landed?"
        Sub-Q1: "When did Apollo 11 land?" → A: July 20, 1969
        Sub-Q2: "Who was president in July 1969?" → A: Richard Nixon
        Final: Richard Nixon

    Interview Question:
        "How does Self-Ask improve reasoning?"
        Self-Ask decomposes complex questions into simpler sub-questions.
        Each sub-question is easier to answer correctly. The model then
        composes the final answer from sub-answers. This is especially
        helpful for multi-hop reasoning where multiple facts must be
        combined.
    """

    def __init__(self, config: Optional[CoTConfig] = None):
        """Initialize Self-Ask CoT."""
        self.config = config or CoTConfig()

    def decompose_question(self, question: str) -> List[str]:
        """
        Decompose a complex question into sub-questions.

        In production, this uses an LLM to generate sub-questions.
        Here we simulate the decomposition process.

        Args:
            question: Complex question to decompose

        Returns:
            List of sub-questions

        Example:
            >>> sa = SelfAskCoT()
            >>> subs = sa.decompose_question(
            ...     "Who was president when Apollo 11 landed?")
            >>> print(len(subs))
            2
        """
        # Simulate decomposition
        # In production: model.generate(f"Decpose into sub-questions: {question}")
        sub_questions = []

        # Simple heuristic decomposition
        if "when" in question.lower() or "who" in question.lower():
            sub_questions.append(f"What is the key fact needed to answer: {question}?")
            sub_questions.append(f"Based on that fact, {question}")

        if not sub_questions:
            sub_questions = [question]  # Can't decompose further

        return sub_questions

    def answer_sub_question(self, sub_question: str) -> str:
        """
        Answer a single sub-question.

        Args:
            sub_question: The sub-question to answer

        Returns:
            Answer to the sub-question
        """
        # Simulate answering
        # In production: model.generate(sub_question)
        return f"Answer to: {sub_question}"

    def compose_answer(self, question: str, sub_qa_pairs: List[Tuple[str, str]]) -> str:
        """
        Compose final answer from sub-question/answer pairs.

        Args:
            question: Original question
            sub_qa_pairs: List of (sub_question, sub_answer) tuples

        Returns:
            Final composed answer
        """
        # In production: model.generate(context_with_sub_answers + question)
        if not sub_qa_pairs:
            return "Cannot answer without sub-questions"

        # Use the last sub-answer as the final answer
        return sub_qa_pairs[-1][1]

    def reason(self, question: str) -> Dict:
        """
        Full Self-Ask CoT pipeline.

        Args:
            question: Complex question to answer

        Returns:
            Dictionary with sub-questions, sub-answers, final answer
        """
        # Step 1: Decompose into sub-questions
        sub_questions = self.decompose_question(question)

        # Step 2: Answer each sub-question
        sub_qa_pairs = []
        for sq in sub_questions:
            answer = self.answer_sub_question(sq)
            sub_qa_pairs.append((sq, answer))

        # Step 3: Compose final answer
        final_answer = self.compose_answer(question, sub_qa_pairs)

        return {
            'question': question,
            'sub_questions': sub_questions,
            'sub_qa_pairs': sub_qa_pairs,
            'answer': final_answer
        }


################################################################################
# SECTION 6: MULTIMODAL CHAIN OF THOUGHT
################################################################################

class MultimodalCoT:
    """
    Multimodal Chain of Thought
    =============================

    Combine vision and text for reasoning. First generate a visual
    rationale (describe what's in the image), then reason about it.

    Paper: "Multimodal Chain-of-Thought Reasoning in Language Models"
           (Zhang et al., 2023)

    Two-Stage Process:
        Stage 1: Rationale Generation — extract visual features, describe image
        Stage 2: Answer Inference — use rationale + question to produce answer

    Key Insight:
        Visual information provides grounding for reasoning.
        By first describing what we see (rationale), we create
        a textual representation that the language model can reason over.

    Interview Question:
        "How does Multimodal CoT work?"
        Two stages: (1) Generate a visual rationale by describing the image
        content relevant to the question, (2) Use the rationale + question
        to infer the answer. This bridges the vision-language gap and
        enables reasoning over visual information.
    """

    def __init__(self, config: Optional[CoTConfig] = None):
        """Initialize Multimodal CoT."""
        self.config = config or CoTConfig()

    def extract_visual_features(self, image_description: str) -> Dict:
        """
        Extract relevant visual features from image.

        In production, this uses a vision encoder (ViT, CLIP).
        Here we simulate feature extraction.

        Args:
            image_description: Text description of the image

        Returns:
            Dictionary of visual features
        """
        # Simulate visual feature extraction
        features = {
            'objects': image_description.split()[:5],  # First 5 words as "objects"
            'attributes': ['color', 'shape', 'size'],
            'spatial': ['left', 'right', 'above', 'below']
        }
        return features

    def generate_rationale(self, image_description: str, question: str) -> str:
        """
        Stage 1: Generate visual rationale.

        Describe what's in the image that's relevant to the question.

        Args:
            image_description: Description of the image
            question: The question being asked

        Returns:
            Visual rationale text
        """
        # In production: vision_model.generate(image, question)
        features = self.extract_visual_features(image_description)
        rationale = (
            f"Looking at the image, I can see: {image_description}. "
            f"The relevant objects are: {', '.join(features['objects'])}. "
            f"This relates to the question because..."
        )
        return rationale

    def infer_answer(self, rationale: str, question: str) -> str:
        """
        Stage 2: Infer answer from rationale + question.

        Args:
            rationale: Visual rationale from Stage 1
            question: Original question

        Returns:
            Final answer
        """
        # In production: language_model.generate(rationale + question)
        return f"Based on the visual analysis: answer to '{question}'"

    def reason(self, image_description: str, question: str) -> Dict:
        """
        Full Multimodal CoT pipeline.

        Args:
            image_description: Description of the image
            question: Question about the image

        Returns:
            Dictionary with rationale and answer
        """
        # Stage 1: Generate visual rationale
        rationale = self.generate_rationale(image_description, question)

        # Stage 2: Infer answer
        answer = self.infer_answer(rationale, question)

        return {
            'image_description': image_description,
            'question': question,
            'rationale': rationale,
            'answer': answer
        }


################################################################################
# SECTION 7: COT EVALUATION
################################################################################

class CoTEvaluator:
    """
    Chain of Thought Evaluator
    ============================

    Evaluate the quality of CoT reasoning chains.

    Metrics:
        1. Step Correctness: Are individual steps logically valid?
        2. Chain Coherence: Do steps follow from each other?
        3. Answer Correctness: Does the final answer match ground truth?
        4. Reasoning Faithfulness: Does the reasoning actually lead to the answer?
    """

    def evaluate_step(self, step: str) -> float:
        """
        Evaluate a single reasoning step.

        Args:
            step: A reasoning step

        Returns:
            Quality score between 0 and 1
        """
        # Simple heuristics for step quality
        score = 0.5
        if any(kw in step.lower() for kw in ['because', 'therefore', 'since', 'so']):
            score += 0.2  # Causal reasoning
        if any(c.isdigit() for c in step):
            score += 0.1  # Contains numbers (math grounding)
        if len(step.split()) > 5:
            score += 0.1  # Sufficiently detailed
        return min(score, 1.0)

    def evaluate_chain(self, reasoning: List[str]) -> Dict:
        """
        Evaluate an entire reasoning chain.

        Args:
            reasoning: List of reasoning steps

        Returns:
            Dictionary with step scores and aggregate metrics
        """
        step_scores = [self.evaluate_step(step) for step in reasoning]

        return {
            'step_scores': step_scores,
            'mean_score': np.mean(step_scores) if step_scores else 0.0,
            'min_score': np.min(step_scores) if step_scores else 0.0,
            'chain_length': len(reasoning),
            'coherence': np.mean([
                abs(step_scores[i] - step_scores[i-1])
                for i in range(1, len(step_scores))
            ]) if len(step_scores) > 1 else 1.0
        }

    def compute_reward(self, reasoning: List[str], correct_answer: str,
                       predicted_answer: str) -> float:
        """
        Compute reward for RL training of CoT models.

        Args:
            reasoning: Generated reasoning chain
            correct_answer: Ground truth answer
            predicted_answer: Model's predicted answer

        Returns:
            Reward value (higher is better)
        """
        # Answer correctness reward
        answer_reward = 1.0 if predicted_answer.strip().lower() == correct_answer.strip().lower() else 0.0

        # Reasoning quality reward
        chain_eval = self.evaluate_chain(reasoning)
        reasoning_reward = chain_eval['mean_score']

        # Combined reward (answer correctness weighted more)
        return 0.7 * answer_reward + 0.3 * reasoning_reward


################################################################################
# SECTION 8: TESTING & DEMONSTRATION
################################################################################

def demonstrate_chain_of_thought():
    """
    Demonstrate Chain of Thought reasoning methods.

    Shows all CoT variants with concrete examples.
    """
    print("=" * 70)
    print("CHAIN OF THOUGHT REASONING DEMONSTRATION")
    print("=" * 70)

    # ── Zero-Shot CoT ──
    print("\n1. ZERO-SHOT CHAIN OF THOUGHT")
    print("-" * 40)
    cot = ZeroShotCoT()
    question = "What is 23 * 47?"
    result = cot.reason(question)
    print(f"Question: {result['question']}")
    print(f"Trigger:  {result['trigger']}")
    print(f"CoT Prompt: {result['cot_prompt']}")
    print(f"Reasoning steps: {len(result['reasoning'])}")
    for step in result['reasoning'][:3]:
        print(f"  {step}")
    print(f"Answer: {result['answer']}")

    # ── Few-Shot CoT ──
    print("\n2. FEW-SHOT CHAIN OF THOUGHT")
    print("-" * 40)
    few_shot = FewShotCoT()
    few_shot.add_exemplar(
        "Roger has 5 tennis balls. He buys 2 cans of 3. How many?",
        ["Roger starts with 5 balls.", "2 cans × 3 balls = 6 balls.", "5 + 6 = 11."],
        "11"
    )
    few_shot.add_exemplar(
        "A bakery makes 3 batches of 12 cookies. They sell 20. How many left?",
        ["3 batches × 12 = 36 cookies.", "36 - 20 = 16 cookies remaining."],
        "16"
    )
    result = few_shot.reason("John has 8 apples. He buys 4 more. How many?")
    print(f"Question: {result['question']}")
    print(f"Exemplars used: {result['num_exemplars']}")
    print(f"Answer: {result['answer']}")

    # ── Auto-CoT ──
    print("\n3. AUTO CHAIN OF THOUGHT")
    print("-" * 40)
    auto_cot = AutoCoT()
    questions = [
        "What is 5 + 3?",
        "What is 12 - 4?",
        "Who invented the telephone?",
        "Who created Python?",
        "What is 7 * 8?",
    ]
    exemplars = auto_cot.auto_generate_exemplars(questions)
    print(f"Generated {len(exemplars)} exemplars from {len(questions)} questions")
    for ex in exemplars:
        print(f"  Q: {ex['question']}")

    # ── Self-Ask ──
    print("\n4. SELF-ASK CHAIN OF THOUGHT")
    print("-" * 40)
    self_ask = SelfAskCoT()
    question = "Who was the US president when Apollo 11 landed?"
    result = self_ask.reason(question)
    print(f"Question: {result['question']}")
    print(f"Sub-questions:")
    for sq, sa in result['sub_qa_pairs']:
        print(f"  Q: {sq}")
        print(f"  A: {sa}")
    print(f"Final Answer: {result['answer']}")

    # ── Multimodal CoT ──
    print("\n5. MULTIMODAL CHAIN OF THOUGHT")
    print("-" * 40)
    mm_cot = MultimodalCoT()
    result = mm_cot.reason(
        "A photo of a cat sitting on a red mat",
        "What color is the mat?"
    )
    print(f"Image: {result['image_description']}")
    print(f"Question: {result['question']}")
    print(f"Rationale: {result['rationale'][:100]}...")
    print(f"Answer: {result['answer']}")

    # ── CoT Evaluation ──
    print("\n6. COT EVALUATION")
    print("-" * 40)
    evaluator = CoTEvaluator()
    reasoning = [
        "First, I need to calculate 23 * 47.",
        "I can break this down: 23 * 47 = 23 * (50 - 3).",
        "23 * 50 = 1150.",
        "23 * 3 = 69.",
        "Therefore, 1150 - 69 = 1081."
    ]
    eval_result = evaluator.evaluate_chain(reasoning)
    print(f"Chain length: {eval_result['chain_length']}")
    print(f"Mean step score: {eval_result['mean_score']:.3f}")
    print(f"Min step score: {eval_result['min_score']:.3f}")

    reward = evaluator.compute_reward(reasoning, "1081", "1081")
    print(f"Reward (correct answer): {reward:.3f}")

    reward_wrong = evaluator.compute_reward(reasoning, "1081", "1082")
    print(f"Reward (wrong answer): {reward_wrong:.3f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_chain_of_thought()
