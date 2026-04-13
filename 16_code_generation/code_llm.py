"""
################################################################################
CODE GENERATION MODEL — AI FOR PROGRAMMING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Code Generation Model?
    A language model specialized for code understanding and generation.

Key Models:
    - CodeLLaMA (Meta): LLaMA fine-tuned for code
    - StarCoder (BigCode): Open-source code model
    - DeepSeek-Coder: Strong coding model
    - GitHub Copilot: Production code assistant

Capabilities:
    - Code completion
    - Code generation from description
    - Bug detection
    - Code explanation
    - Test generation

Interview Questions:
    Q: "How do code models work?"
    A: Same as language models, but trained on code data.
       Learn programming patterns, syntax, and semantics.

################################################################################
"""

import numpy as np
from typing import Optional, List

################################################################################
# SECTION 1: CODE MODEL
################################################################################

class CodeModel:
    """
    Code Generation Model
    =====================

    Specialized language model for code.

    Training:
    - Pre-trained on code repositories
    - Fine-tuned on instruction-following data

    Interview Questions:
        Q: "What makes code models different from general LLMs?"
        A: Trained on code data, understand programming patterns,
           can generate syntactically correct code.
    """

    def __init__(self, vocab_size: int = 32000, d_model: int = 256):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token embeddings
        self.token_embed = np.random.randn(vocab_size, d_model) * 0.02

        # Simplified transformer
        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02
        self.output_head = np.random.randn(d_model, vocab_size) * 0.02

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 100,
        temperature: float = 0.2
    ) -> np.ndarray:
        """
        Generate code tokens.

        Args:
            prompt_ids: Prompt token IDs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Generated token IDs
        """
        generated = list(prompt_ids[0])

        for _ in range(max_tokens):
            # Simplified forward pass
            x = self.token_embed[generated[-1]]
            logits = x @ self.output_head

            # Sample
            probs = np.exp(logits / temperature - np.max(logits / temperature))
            probs = probs / np.sum(probs)
            token = np.random.choice(self.vocab_size, p=probs)

            generated.append(token)

        return np.array([generated])

    def complete_function(self, function_signature: str) -> str:
        """
        Complete a function from its signature.

        Args:
            function_signature: Function definition

        Returns:
            Completed function
        """
        return f"{function_signature}\n    # Implementation\n    pass"

    def generate_tests(self, function_code: str) -> str:
        """
        Generate tests for a function.

        Args:
            function_code: Function to test

        Returns:
            Test code
        """
        return "def test_function():\n    assert True"


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_code_model():
    """Demonstrate code model."""
    print("=" * 70)
    print("CODE MODEL DEMONSTRATION")
    print("=" * 70)

    model = CodeModel(vocab_size=1000, d_model=64)

    # Generate
    print("\n--- Code Generation ---")
    prompt = np.array([[1, 2, 3]])
    generated = model.generate(prompt, max_tokens=20, temperature=0.2)
    print(f"Prompt: {prompt[0].tolist()}")
    print(f"Generated: {generated[0].tolist()}")

    # Complete function
    print("\n--- Function Completion ---")
    completed = model.complete_function("def add(a, b):")
    print(f"Completed:\n{completed}")

    # Generate tests
    print("\n--- Test Generation ---")
    tests = model.generate_tests("def add(a, b): return a + b")
    print(f"Tests:\n{tests}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_code_model()


################################################################################
# REFERENCES
################################################################################

# [1] Rozière, B., et al. (2024). Code Llama: Open Foundation Models for Code.

################################################################################
