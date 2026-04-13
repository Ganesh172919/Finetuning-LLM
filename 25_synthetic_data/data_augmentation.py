"""
################################################################################
DATA AUGMENTATION — EXPANDING TRAINING DATA WITH TRANSFORMATIONS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Data Augmentation?
    Creating new training examples by transforming existing ones.
    For text/NLP, this includes paraphrasing, back-translation,
    instruction evolution, and more.

Why does it matter?
    - Increases dataset size without collecting new data
    - Improves model robustness to variations
    - Reduces overfitting
    - Enables better generalization

Key Techniques:
    1. Paraphrasing: Reword while preserving meaning
    2. Back-translation: Translate to another language and back
    3. Evol-Instruct: Evolve instructions to be more complex
    4. Template-based: Fill templates with different values

Evol-Instruct (WizardLM):
    Evolves simple instructions into complex ones:
    - Add constraints
    - Deepen (add reasoning steps)
    - Concretize (add specifics)
    - Increase reasoning requirements

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Data Augmentation Pipeline                       │
    │                                                                  │
    │  Original Data ──▶ Paraphrase ──▶ Augmented v1                 │
    │                 ──▶ Back-Translate ──▶ Augmented v2             │
    │                 ──▶ Evol-Instruct ──▶ Augmented v3              │
    │                 ──▶ Template Fill ──▶ Augmented v4              │
    │                                                                  │
    │  Combine all augmented versions ──▶ Expanded Dataset            │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "What's the best data augmentation for NLP?"
       Depends on the task. For classification: paraphrasing and
       back-translation. For instruction tuning: Evol-Instruct.
       For code: mutation-based augmentation.

    2. "How does Evol-Instruct work?"
       Start with simple instructions, then "evolve" them by adding
       constraints, depth, or specificity. This creates a curriculum
       from easy to hard, improving model capabilities.

    3. "Can augmentation cause label noise?"
       Yes, if not done carefully. Paraphrasing might change meaning.
       Always verify augmented samples or use conservative transforms.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: PARAPHRASING
################################################################################

class Paraphraser:
    """
    Text Paraphraser
    =================

    Generates paraphrased versions of text while preserving meaning.

    Techniques:
    - Synonym replacement
    - Sentence restructuring
    - Active/passive voice conversion
    - Formality adjustment

    Interview Question:
        "How do you ensure paraphrases preserve meaning?"
        Use semantic similarity metrics (embedding cosine similarity)
        to verify. Also use NLI models to check entailment.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Embeddings for semantic similarity
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.02

        # Synonym groups (simplified)
        self.synonym_groups = self._create_synonym_groups()

    def _create_synonym_groups(self) -> List[List[int]]:
        """Create groups of synonymous tokens."""
        groups = []
        for i in range(0, self.vocab_size, 10):
            group = list(range(i, min(i + 3, self.vocab_size)))
            groups.append(group)
        return groups

    def synonym_replacement(
        self,
        tokens: List[int],
        replacement_rate: float = 0.1
    ) -> List[int]:
        """
        Replace some tokens with synonyms.

        Args:
            tokens: Input token sequence
            replacement_rate: Fraction of tokens to replace

        Returns:
            Augmented token sequence
        """
        result = list(tokens)
        num_replace = max(1, int(len(tokens) * replacement_rate))

        # Randomly select positions to replace
        positions = np.random.choice(len(tokens), min(num_replace, len(tokens)), replace=False)

        for pos in positions:
            token = result[pos]
            # Find synonym group
            for group in self.synonym_groups:
                if token in group:
                    # Replace with random synonym from group
                    synonym = np.random.choice([t for t in group if t != token])
                    result[pos] = synonym
                    break

        return result

    def random_insertion(
        self,
        tokens: List[int],
        num_insert: int = 1
    ) -> List[int]:
        """Insert random related tokens."""
        result = list(tokens)
        for _ in range(num_insert):
            pos = np.random.randint(0, len(result))
            new_token = np.random.randint(0, self.vocab_size)
            result.insert(pos, new_token)
        return result

    def random_deletion(
        self,
        tokens: List[int],
        delete_rate: float = 0.1
    ) -> List[int]:
        """Randomly delete some tokens."""
        if len(tokens) <= 2:
            return tokens

        num_delete = max(1, int(len(tokens) * delete_rate))
        result = list(tokens)

        for _ in range(num_delete):
            if len(result) > 2:
                pos = np.random.randint(0, len(result))
                result.pop(pos)

        return result


################################################################################
# SECTION 2: BACK TRANSLATION
################################################################################

class BackTranslator:
    """
    Back-Translation Augmentation
    ================================

    Translate text to another "language" and back to create paraphrases.

    Process:
    1. Encode original text
    2. "Translate" to intermediate representation
    3. "Translate" back to original language
    4. Result is a paraphrase

    This is a powerful augmentation because it naturally
    restructures sentences while preserving meaning.

    Interview Question:
        "Why does back-translation work for augmentation?"
        Translation requires understanding meaning, not just
        surface form. The intermediate representation captures
        semantics, and re-generation creates natural variation.
    """

    def __init__(self, vocab_size: int, latent_dim: int = 32):
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        # "Encoder" (source → latent)
        self.enc_embed = np.random.randn(vocab_size, latent_dim) * 0.02

        # "Decoder" (latent → source)
        self.dec_proj = np.random.randn(latent_dim, vocab_size) * 0.02

    def encode(self, tokens: List[int]) -> np.ndarray:
        """Encode tokens to latent representation."""
        if not tokens:
            return np.zeros(self.latent_dim)
        embeddings = self.enc_embed[tokens]
        return np.mean(embeddings, axis=0)

    def decode(self, latent: np.ndarray, length: int) -> List[int]:
        """Decode latent representation to tokens."""
        tokens = []
        for _ in range(length):
            logits = latent @ self.dec_proj
            probs = self._softmax(logits)
            token = np.random.choice(self.vocab_size, p=probs)
            tokens.append(token)
        return tokens

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def back_translate(
        self,
        tokens: List[int],
        noise_scale: float = 0.1
    ) -> List[int]:
        """
        Back-translate tokens to create a paraphrase.

        Args:
            tokens: Input token sequence
            noise_scale: Amount of noise in latent space

        Returns:
            Paraphrased token sequence
        """
        # Encode to latent
        latent = self.encode(tokens)

        # Add small noise (simulates translation variation)
        noise = np.random.randn(self.latent_dim) * noise_scale
        latent_noisy = latent + noise

        # Decode back
        return self.decode(latent_noisy, len(tokens))


################################################################################
# SECTION 3: EVOL-INSTRUCT
################################################################################

class EvolInstruct:
    """
    Evol-Instruct — Evolving Instructions
    ========================================

    Evolves simple instructions into more complex ones.

    Evolution strategies:
    1. Add constraints: "Write a poem" → "Write a sonnet about AI"
    2. Deepen: "Explain X" → "Explain X, then compare with Y"
    3. Concretize: "Write code" → "Write Python code using numpy"
    4. Reasoning: "Solve X" → "Solve X step by step, explaining why"

    Based on WizardLM (2023):
    - Start with simple instructions
    - Apply evolution operators
    - Create instruction complexity curriculum
    - Train on evolved instructions for better capabilities

    Interview Questions:
        1. "How does Evol-Instruct improve model capabilities?"
           By creating a curriculum from simple to complex instructions.
           Models learn to handle harder tasks by progressively
           training on evolved, more challenging examples.

        2. "What evolution strategies work best?"
           Deepening (adding reasoning steps) is most effective for
           reasoning tasks. Concretizing helps with specificity.
           Adding constraints improves instruction following.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

        # Constraint tokens (simplified)
        self.constraint_tokens = list(range(min(50, vocab_size)))
        self.reasoning_tokens = list(range(50, min(100, vocab_size)))
        self.specific_tokens = list(range(100, min(150, vocab_size)))

    def add_constraints(
        self,
        tokens: List[int],
        num_constraints: int = 2
    ) -> List[int]:
        """
        Add constraints to an instruction.

        Example: "Write a poem" → "Write a poem [about AI] [in sonnet form]"
        """
        result = list(tokens)
        for _ in range(num_constraints):
            constraint = np.random.choice(self.constraint_tokens)
            result.append(constraint)
        return result

    def deepen(
        self,
        tokens: List[int],
        num_steps: int = 1
    ) -> List[int]:
        """
        Deepen the instruction by adding reasoning steps.

        Example: "Explain X" → "Explain X, then compare with Y and Z"
        """
        result = list(tokens)
        for _ in range(num_steps):
            # Add reasoning tokens
            reasoning = np.random.choice(self.reasoning_tokens, size=3).tolist()
            result.extend(reasoning)
        return result

    def concretize(
        self,
        tokens: List[int],
        specificity: int = 2
    ) -> List[int]:
        """
        Make the instruction more specific.

        Example: "Write code" → "Write Python code using numpy for matrix multiplication"
        """
        result = list(tokens)
        specific = np.random.choice(self.specific_tokens, size=specificity).tolist()
        result.extend(specific)
        return result

    def evolve(
        self,
        tokens: List[int],
        strategy: str = "random"
    ) -> List[int]:
        """
        Apply evolution strategy to instruction.

        Args:
            tokens: Original instruction tokens
            strategy: Evolution strategy

        Returns:
            Evolved instruction tokens
        """
        if strategy == "random":
            strategy = np.random.choice(["constrain", "deepen", "concretize"])

        if strategy == "constrain":
            return self.add_constraints(tokens)
        elif strategy == "deepen":
            return self.deepen(tokens)
        elif strategy == "concretize":
            return self.concretize(tokens)
        else:
            return tokens

    def create_curriculum(
        self,
        tokens: List[int],
        num_levels: int = 3
    ) -> List[List[int]]:
        """
        Create a curriculum from simple to complex.

        Args:
            tokens: Original instruction
            num_levels: Number of complexity levels

        Returns:
            List of evolved instructions at increasing complexity
        """
        curriculum = [tokens]

        for level in range(num_levels):
            evolved = self.evolve(
                curriculum[-1],
                strategy=["constrain", "deepen", "concretize"][level % 3]
            )
            curriculum.append(evolved)

        return curriculum


################################################################################
# SECTION 4: INSTRUCTION AUGMENTER (COMPLETE)
################################################################################

class InstructionAugmenter:
    """
    Complete Instruction Augmentation Pipeline
    =============================================

    Combines all augmentation techniques for instruction tuning data.

    Use cases:
    - Expanding instruction tuning datasets
    - Creating diverse training examples
    - Building complexity curricula
    - Improving model robustness

    Interview Questions:
        1. "How much augmentation is too much?"
           Diminishing returns after 3-5x augmentation. Quality
           degrades if augmented samples are too noisy. Monitor
           validation performance to find the sweet spot.

        2. "Should you augment validation data too?"
           No. Validation should reflect real-world distribution.
           Only augment training data.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.paraphraser = Paraphraser(vocab_size)
        self.back_translator = BackTranslator(vocab_size)
        self.evol_instruct = EvolInstruct(vocab_size)

    def augment(
        self,
        tokens: List[int],
        num_augmented: int = 4
    ) -> List[List[int]]:
        """
        Generate multiple augmented versions.

        Args:
            tokens: Original instruction tokens
            num_augmented: Number of augmented versions

        Returns:
            List of augmented token sequences
        """
        augmented = []

        # Paraphrasing
        augmented.append(self.paraphraser.synonym_replacement(tokens))

        # Back-translation
        augmented.append(self.back_translator.back_translate(tokens))

        # Evol-Instruct
        augmented.append(self.evol_instruct.evolve(tokens, "constrain"))
        augmented.append(self.evol_instruct.evolve(tokens, "deepen"))

        return augmented[:num_augmented]


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################

def demonstrate_data_augmentation():
    """Demonstrate data augmentation techniques."""
    print("=" * 70)
    print("DATA AUGMENTATION")
    print("=" * 70)

    vocab_size = 100

    # Create augmenter
    augmenter = InstructionAugmenter(vocab_size)

    # Original instruction (simplified as token list)
    original = [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"\nOriginal tokens: {original}")

    # Paraphrasing
    print("\n--- Paraphrasing ---")
    paraphraser = Paraphraser(vocab_size)
    for i in range(3):
        paraphrased = paraphraser.synonym_replacement(original, replacement_rate=0.2)
        print(f"  Version {i+1}: {paraphrased}")

    # Back-translation
    print("\n--- Back-Translation ---")
    back_translator = BackTranslator(vocab_size)
    for i in range(3):
        back_translated = back_translator.back_translate(original, noise_scale=0.2)
        print(f"  Version {i+1}: {back_translated}")

    # Evol-Instruct
    print("\n--- Evol-Instruct ---")
    evol = EvolInstruct(vocab_size)
    curriculum = evol.create_curriculum(original, num_levels=3)
    for i, evolved in enumerate(curriculum):
        print(f"  Level {i}: {evolved}")

    # Combined augmentation
    print("\n--- Combined Augmentation ---")
    augmented = augmenter.augment(original, num_augmented=4)
    for i, aug in enumerate(augmented):
        print(f"  Augmented {i+1}: {aug}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Multiple augmentation strategies create diverse training data!")
    print("Evol-Instruct progressively increases instruction complexity.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_data_augmentation()
