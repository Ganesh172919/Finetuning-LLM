"""
################################################################################
CONSTITUTIONAL AI (CAI) — SELF-ALIGNMENT WITH PRINCIPLES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Constitutional AI?
    Constitutional AI (CAI) is a method for aligning AI models using
    a set of principles (a "constitution") rather than human feedback.
    The model critiques and revises its own outputs according to these
    principles, reducing the need for human annotation.

Why CAI matters:
    - Reduces reliance on human feedback (expensive, slow, inconsistent)
    - Scales alignment to more principles without more humans
    - Makes alignment principles explicit and auditable
    - Can be applied to safety, helpfulness, and other dimensions

How CAI works (two phases):
    Phase 1: Supervised Self-Improvement
    - Generate response to prompt
    - Critique response according to constitution
    - Revise response based on critique
    - Fine-tune on revised responses

    Phase 2: RL from AI Feedback (RLAIF)
    - Generate pairs of responses
    - Have AI judge which is better according to constitution
    - Train reward model on AI preferences
    - Use RLHF with AI-generated preferences

Interview Questions:
    Q: "What is Constitutional AI?"
    A: A method that uses explicit principles (a constitution) to align
       AI models. The model critiques and revises its own outputs
       according to these principles, reducing human annotation needs.

    Q: "How does CAI differ from RLHF?"
    A: RLHF uses human preferences; CAI uses AI-generated preferences
       based on explicit principles. CAI is more scalable and auditable,
       but may miss nuances that humans catch.

    Q: "What goes into a constitution?"
    A: Principles like "be helpful", "be harmless", "be honest",
       "don't help with illegal activities", "cite sources", etc.
       Each principle is used for critique and revision.

################################################################################
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

################################################################################
# SECTION 1: CONSTITUTION
################################################################################

@dataclass
class Principle:
    """A single constitutional principle."""
    name: str
    description: str
    critique_prompt: str
    revision_prompt: str
    weight: float = 1.0


class Constitution:
    """
    Collection of principles for AI alignment.

    A constitution defines what "good" behavior looks like for the AI.
    Each principle provides:
    - A description of the desired behavior
    - A prompt for critiquing violations
    - A prompt for revising to fix violations

    Interview Question:
        Q: "How do you design a constitution for CAI?"
        A: Start with high-level goals (helpful, harmless, honest),
           then create specific, testable principles for each.
           Each principle needs a critique prompt (what's wrong)
           and a revision prompt (how to fix it).
    """

    def __init__(self):
        self.principles: List[Principle] = []
        self._load_default_principles()

    def _load_default_principles(self):
        """Load default alignment principles."""
        self.principles = [
            Principle(
                name="helpfulness",
                description="The response should be helpful and address the user's question",
                critique_prompt="Is this response helpful? Does it address what the user asked?",
                revision_prompt="Make this response more helpful by directly addressing the question",
                weight=1.0,
            ),
            Principle(
                name="harmlessness",
                description="The response should not cause harm or help with harmful activities",
                critique_prompt="Could this response cause harm? Does it help with dangerous activities?",
                revision_prompt="Remove any harmful content while keeping the helpful parts",
                weight=1.5,  # Higher weight for safety
            ),
            Principle(
                name="honesty",
                description="The response should be truthful and not mislead",
                critique_prompt="Is this response truthful? Does it contain false claims?",
                revision_prompt="Correct any false claims and add appropriate uncertainty",
                weight=1.2,
            ),
            Principle(
                name="specificity",
                description="The response should be specific and detailed",
                critique_prompt="Is this response specific enough? Is it too vague?",
                revision_prompt="Add more specific details and examples",
                weight=0.8,
            ),
            Principle(
                name="clarity",
                description="The response should be clear and well-organized",
                critique_prompt="Is this response clear? Is it well-organized?",
                revision_prompt="Improve clarity and organization",
                weight=0.7,
            ),
        ]

    def add_principle(self, principle: Principle):
        """Add a new principle to the constitution."""
        self.principles.append(principle)

    def get_principles(self) -> List[Principle]:
        """Get all principles."""
        return self.principles


################################################################################
# SECTION 2: CONSTITUTIONAL AI ENGINE
################################################################################

class ConstitutionalAI:
    """
    Constitutional AI Engine — Self-alignment with principles.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                Constitutional AI Pipeline                    │
    │                                                              │
    │  Phase 1: Self-Improvement                                   │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
    │  │ Generate │───▶│  Critique │───▶│  Revise  │              │
    │  │ Response │    │ (by       │    │ (by      │              │
    │  │          │    │  principles)│   │  principles)│           │
    │  └──────────┘    └──────────┘    └──────────┘              │
    │       │                              │                      │
    │       └──────────────────────────────┘                      │
    │                    │                                        │
    │                    ▼                                        │
    │            Fine-tune on revised responses                   │
    │                                                              │
    │  Phase 2: RLAIF (RL from AI Feedback)                      │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
    │  │ Generate │───▶│  AI Judge │───▶│  Train   │              │
    │  │ Pairs    │    │ (by       │    │  Reward  │              │
    │  │          │    │  principles)│   │  Model   │              │
    │  └──────────┘    └──────────┘    └──────────┘              │
    │                                        │                    │
    │                                        ▼                    │
    │                                    RLHF Training            │
    └─────────────────────────────────────────────────────────────┘

    Interview Question:
        Q: "Walk me through the CAI pipeline."
        A: Two phases: (1) Self-improvement — generate, critique against
           constitution, revise, fine-tune on revisions. (2) RLAIF —
           generate response pairs, AI judges which is better per
           constitution, train reward model, do RLHF with AI feedback.
    """

    def __init__(self, constitution: Constitution = None):
        self.constitution = constitution or Constitution()
        self.critique_history = []
        self.revision_history = []

    def critique(
        self,
        prompt: str,
        response: str,
        principle: Principle,
    ) -> Dict:
        """
        Critique a response according to a principle.

        Args:
            prompt: Original user prompt
            response: AI-generated response
            principle: Principle to check against

        Returns:
            Critique with score and explanation
        """
        # Simulate critique (in production: use LLM)
        score = self._evaluate_response(response, principle)
        explanation = self._generate_critique(response, principle, score)

        critique = {
            "principle": principle.name,
            "score": score,
            "explanation": explanation,
            "needs_revision": score < 0.7,
        }

        self.critique_history.append(critique)
        return critique

    def revise(
        self,
        prompt: str,
        response: str,
        critiques: List[Dict],
    ) -> str:
        """
        Revise a response based on critiques.

        Args:
            prompt: Original user prompt
            response: Original response
            critiques: List of critiques from critique()

        Returns:
            revised: Improved response
        """
        if not any(c["needs_revision"] for c in critiques):
            return response

        # Simulate revision (in production: use LLM with revision prompts)
        revised = response
        for critique in critiques:
            if critique["needs_revision"]:
                principle = next(
                    p for p in self.constitution.principles
                    if p.name == critique["principle"]
                )
                revised = self._apply_revision(revised, principle, critique)

        self.revision_history.append({
            "original": response,
            "revised": revised,
            "critiques": critiques,
        })

        return revised

    def self_improve(
        self,
        prompt: str,
        response: str,
        n_iterations: int = 3,
    ) -> Tuple[str, List[Dict]]:
        """
        Iteratively improve a response using constitutional principles.

        Args:
            prompt: Original user prompt
            response: Initial response
            n_iterations: Number of critique-revise cycles

        Returns:
            final_response: Best response after iterations
            history: List of critiques from each iteration
        """
        current = response
        history = []

        for i in range(n_iterations):
            # Critique against all principles
            critiques = []
            for principle in self.constitution.principles:
                critique = self.critique(prompt, current, principle)
                critiques.append(critique)

            history.append({"iteration": i, "critiques": critiques})

            # Check if all principles satisfied
            if all(c["score"] >= 0.7 for c in critiques):
                break

            # Revise
            current = self.revise(prompt, current, critiques)

        return current, history

    def judge_pair(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> Dict:
        """
        Judge which of two responses is better according to the constitution.

        This is the RLAIF step — instead of human judges, we use
        the constitution to determine which response is preferred.

        Args:
            prompt: Original prompt
            response_a: First response
            response_b: Second response

        Returns:
            Judgment with winner and reasoning
        """
        scores_a = []
        scores_b = []
        reasons = []

        for principle in self.constitution.principles:
            score_a = self._evaluate_response(response_a, principle)
            score_b = self._evaluate_response(response_b, principle)

            scores_a.append(score_a * principle.weight)
            scores_b.append(score_b * principle.weight)

            if score_a > score_b:
                reasons.append(f"A is better on {principle.name}")
            elif score_b > score_a:
                reasons.append(f"B is better on {principle.name}")

        total_a = sum(scores_a)
        total_b = sum(scores_b)

        winner = "A" if total_a > total_b else "B"

        return {
            "winner": winner,
            "score_a": total_a,
            "score_b": total_b,
            "reasons": reasons,
            "confidence": abs(total_a - total_b) / (total_a + total_b + 1e-8),
        }

    def _evaluate_response(self, response: str, principle: Principle) -> float:
        """Evaluate response against a principle (simulated)."""
        # In production: use LLM to evaluate
        base_score = 0.5 + np.random.random() * 0.4

        # Heuristic bonuses
        if principle.name == "helpfulness" and len(response) > 100:
            base_score += 0.1
        if principle.name == "harmlessness" and "sorry" not in response.lower():
            base_score += 0.05
        if principle.name == "specificity" and any(c.isdigit() for c in response):
            base_score += 0.05

        return min(1.0, base_score)

    def _generate_critique(self, response: str, principle: Principle, score: float) -> str:
        """Generate critique explanation (simulated)."""
        if score >= 0.7:
            return f"Response satisfies the {principle.name} principle (score: {score:.2f})"
        else:
            return f"Response violates the {principle.name} principle (score: {score:.2f}). {principle.critique_prompt}"

    def _apply_revision(self, response: str, principle: Principle, critique: Dict) -> str:
        """Apply revision based on critique (simulated)."""
        # In production: use LLM with revision prompt
        return response + f"\n[Revised for {principle.name}]"


################################################################################
# SECTION 3: TESTING AND DEMONSTRATION
################################################################################

def demonstrate_cai():
    """Comprehensive Constitutional AI demonstration."""
    print("=" * 70)
    print("CONSTITUTIONAL AI DEMONSTRATION")
    print("=" * 70)

    cai = ConstitutionalAI()

    # === Demo 1: Critique ===
    print("\n--- Demo 1: Critique ---")
    prompt = "How do I make a bomb?"
    response = "I can't help with that request as it could cause harm."

    for principle in cai.constitution.principles:
        critique = cai.critique(prompt, response, principle)
        print(f"  {principle.name}: {critique['score']:.2f} {'✓' if critique['score'] >= 0.7 else '✗'}")

    # === Demo 2: Self-Improvement ===
    print("\n--- Demo 2: Self-Improvement ---")
    prompt2 = "What is machine learning?"
    response2 = "ML is when computers learn stuff."

    improved, history = cai.self_improve(prompt2, response2, n_iterations=3)
    print(f"Original: {response2}")
    print(f"Improved: {improved}")
    print(f"Iterations: {len(history)}")

    # === Demo 3: Judge Pair ===
    print("\n--- Demo 3: RLAIF — Judge Pair ---")
    prompt3 = "Explain quantum computing"
    response_a = "Quantum computing uses qubits that can be 0 and 1 at the same time."
    response_b = "Quantum computing is a type of computing."

    judgment = cai.judge_pair(prompt3, response_a, response_b)
    print(f"Winner: Response {judgment['winner']}")
    print(f"Score A: {judgment['score_a']:.2f}, Score B: {judgment['score_b']:.2f}")
    print(f"Confidence: {judgment['confidence']:.2%}")
    print(f"Reasons: {judgment['reasons']}")

    # === Demo 4: Constitution Summary ===
    print("\n--- Demo 4: Constitution ---")
    for p in cai.constitution.principles:
        print(f"  {p.name}: {p.description} (weight: {p.weight})")

    print("\n" + "=" * 70)
    print("All Constitutional AI demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_cai()


################################################################################
# REFERENCES
################################################################################

# [1] Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback.
#     arXiv:2212.08073.
#
# [2] Sun, Z., et al. (2024). Principle-Driven Self-Alignment of Language Models
#     from Scratch with Minimal Human Supervision. arXiv:2305.03047.

################################################################################
