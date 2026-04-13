"""
################################################################################
CIRCUIT ANALYSIS — REVERSE-ENGINEERING NEURAL NETWORK ALGORITHMS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Circuit Analysis?
    The study of identifying and understanding CIRCUITS in neural networks —
    groups of neurons that work together to implement specific algorithms.

    A "circuit" is like a sub-program in the network:
    - Input detection neurons
    - Processing neurons
    - Output neurons
    - Connections between them

Why does it matter?
    Neural networks aren't just collections of independent neurons.
    They implement ALGORITHMS through connected circuits:

    - Induction heads: Implement in-context learning
    - Successor heads: Track entity states
    - Name movers: Move information between positions

    Understanding circuits reveals:
    - HOW the model implements specific capabilities
    - WHERE to intervene to change behavior
    - WHAT the model has learned algorithmically

Key Discovery: Induction Heads (Anthropic, 2022)
    Induction heads implement the algorithm:
    1. Find previous occurrence of current token
    2. Look at what came AFTER that occurrence
    3. Predict that token will come next

    Example: "A B C ... A B" → predict "C"

    This is a CIRCUIT: specific attention heads working together.

Architecture (Induction Head):
    ┌─────────────────────────────────────────────────────────────────┐
    │                Induction Head Circuit                           │
    │                                                                  │
    │  Previous Token Head (Layer 1):                                │
    │    "A at position 3" → copies info to position 3               │
    │                                                                  │
    │  Induction Head (Layer 2):                                     │
    │    "Current token B at position 5" →                           │
    │    looks back, finds "B at position 2" →                       │
    │    sees "C at position 3" → predicts C                         │
    │                                                                  │
    │  Algorithm: "If I see B, and last time B was followed by C,    │
    │             predict C"                                          │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "What is an induction head?"
       A circuit in transformers that implements in-context learning.
       It finds previous occurrences of the current token and predicts
       what followed it last time. This enables few-shot learning.

    2. "How do you find circuits in neural networks?"
       Methods: (a) Activation patching — replace activations and
       observe effects, (b) Causal intervention — ablate components,
       (c) Automated circuit discovery — search for subgraphs.

    3. "Why are circuits important for safety?"
       If we understand the circuits for harmful behavior, we can
       precisely remove them without affecting other capabilities.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: ATTENTION HEAD
################################################################################

class AttentionHead:
    """
    Single Attention Head
    ======================

    Implements one attention head with interpretable components.

    Each head has:
    - Q, K, V projections
    - Attention pattern
    - Output

    In circuit analysis, we study:
    - What the head attends to (attention pattern)
    - What information it moves (output)
    - How it combines with other heads
    """

    def __init__(self, d_model: int, d_head: int):
        self.d_model = d_model
        self.d_head = d_head

        # Projection matrices
        self.W_q = np.random.randn(d_model, d_head) * 0.02
        self.W_k = np.random.randn(d_model, d_head) * 0.02
        self.W_v = np.random.randn(d_model, d_head) * 0.02
        self.W_o = np.random.randn(d_head, d_model) * 0.02

    def forward(
        self,
        x: np.ndarray,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass through attention head.

        Args:
            x: Input (batch, seq_len, d_model)
            return_attention: Whether to return attention weights

        Returns:
            (output, attention_weights or None)
        """
        batch, seq_len, _ = x.shape

        # Compute Q, K, V
        Q = x @ self.W_q  # (batch, seq_len, d_head)
        K = x @ self.W_k
        V = x @ self.W_v

        # Attention scores
        scores = Q @ K.transpose(0, 2, 1) / math.sqrt(self.d_head)

        # Attention weights
        weights = self._softmax(scores)

        # Output
        out = weights @ V  # (batch, seq_len, d_head)
        out = out @ self.W_o  # (batch, seq_len, d_model)

        if return_attention:
            return out, weights
        return out, None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


################################################################################
# SECTION 2: INDUCTION HEAD
################################################################################

class InductionHead:
    """
    Induction Head Circuit
    ========================

    Implements the induction algorithm:
    "If I see token B now, and last time B was followed by C, predict C"

    Components:
    1. Previous Token Head (Layer 1): Copies previous token info
    2. Induction Head (Layer 2): Uses that info for prediction

    This is a CIRCUIT because it requires specific coordination
    between multiple attention heads across layers.

    Interview Question:
        "How do induction heads enable few-shot learning?"
        When you give a model examples like "A→B, C→D, E→",
        induction heads pattern-match: "E was last seen followed by...
        look at previous context... F!" This enables in-context
        learning without weight updates.
    """

    def __init__(self, d_model: int, d_head: int):
        self.d_model = d_model
        self.d_head = d_head

        # Previous token head (Layer 1)
        self.prev_token_head = AttentionHead(d_model, d_head)

        # Induction head (Layer 2)
        self.induction_head = AttentionHead(d_model, d_head)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass through induction circuit.

        Args:
            x: Input embeddings (batch, seq_len, d_model)

        Returns:
            (output, analysis_dict)
        """
        batch, seq_len, _ = x.shape

        # Layer 1: Previous token head
        # This copies information from the previous position
        prev_out, prev_attn = self.prev_token_head.forward(x, return_attention=True)

        # Layer 2: Induction head
        # Takes output of previous token head and performs induction
        combined = x + prev_out  # Residual connection
        ind_out, ind_attn = self.induction_head.forward(combined, return_attention=True)

        analysis = {
            'prev_token_attention': prev_attn,
            'induction_attention': ind_attn,
            'prev_token_output': prev_out,
            'induction_output': ind_out
        }

        return ind_out, analysis

    def detect_induction_pattern(
        self,
        tokens: List[int]
    ) -> List[Tuple[int, int, int]]:
        """
        Detect induction patterns in a sequence.

        Induction pattern: token A appears at position i,
        then appears again at position j, and the token
        at position i+1 is the prediction for position j+1.

        Args:
            tokens: Token sequence

        Returns:
            List of (prev_pos, current_pos, predicted_token)
        """
        patterns = []

        for j in range(1, len(tokens)):
            current_token = tokens[j]

            # Find previous occurrence
            for i in range(j):
                if tokens[i] == current_token and i + 1 < len(tokens):
                    predicted = tokens[i + 1]
                    patterns.append((i, j, predicted))

        return patterns


################################################################################
# SECTION 3: CIRCUIT TRACER
################################################################################

class CircuitTracer:
    """
    Circuit Tracer
    ===============

    Traces information flow through the network to identify circuits.

    Methods:
    1. Activation Patching: Replace activations, observe effects
    2. Causal Intervention: Ablate components, measure impact
    3. Path Analysis: Track information through layers

    Interview Question:
        "What is activation patching?"
        Replace a component's activation with its value from a
        different input. If this changes the output, that component
        is important for the task. It's like a controlled experiment
        for neural networks.
    """

    def __init__(self, num_layers: int, num_heads: int):
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Component importance scores
        self.importance = np.zeros((num_layers, num_heads))

    def activation_patching(
        self,
        clean_activations: np.ndarray,
        corrupted_activations: np.ndarray,
        layer: int,
        head: int
    ) -> float:
        """
        Perform activation patching for one component.

        Replace the activation of (layer, head) with the corrupted
        version and measure the effect on the output.

        Args:
            clean_activations: Clean run activations
            corrupted_activations: Corrupted run activations
            layer: Layer to patch
            head: Head to patch

        Returns:
            Importance score (effect on output)
        """
        # Create patched activations
        patched = clean_activations.copy()
        patched[:, :, layer, head] = corrupted_activations[:, :, layer, head]

        # Measure effect (simplified)
        effect = np.mean(np.abs(patched - clean_activations))

        return float(effect)

    def trace_circuit(
        self,
        clean_activations: np.ndarray,
        corrupted_activations: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """
        Trace the full circuit by patching each component.

        Args:
            clean_activations: Activations from clean input
            corrupted_activations: Activations from corrupted input

        Returns:
            List of (layer, head, importance) sorted by importance
        """
        components = []

        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                importance = self.activation_patching(
                    clean_activations,
                    corrupted_activations,
                    layer,
                    head
                )
                components.append((layer, head, importance))
                self.importance[layer, head] = importance

        # Sort by importance
        components.sort(key=lambda x: x[2], reverse=True)

        return components

    def find_critical_path(
        self,
        components: List[Tuple[int, int, float]],
        threshold: float = 0.1
    ) -> List[Tuple[int, int]]:
        """
        Find the critical path (most important components).

        Args:
            components: Component importance list
            threshold: Minimum importance to include

        Returns:
            List of (layer, head) for critical components
        """
        return [(l, h) for l, h, imp in components if imp > threshold]


################################################################################
# SECTION 4: CIRCUIT ANALYSIS (COMPLETE)
################################################################################

class CircuitAnalyzer:
    """
    Complete Circuit Analysis Toolkit
    ===================================

    Combines all circuit analysis methods.

    Workflow:
    1. Run model on clean input → get activations
    2. Run model on corrupted input → get activations
    3. Trace circuit by activation patching
    4. Identify critical components
    5. Analyze what each component does

    Interview Questions:
        1. "How do you verify a discovered circuit?"
           (a) Ablate the circuit and check if capability is lost,
           (b) Check if the circuit is minimal (no redundant parts),
           (c) Verify it works on different inputs.

        2. "Can circuits be edited?"
           Yes! You can modify specific components to change behavior.
           This is "model editing" — precise interventions to fix
           specific issues without retraining.

        3. "What's the relationship between circuits and features?"
           Circuits are built from features. Features are what neurons
           detect; circuits are how neurons work together to implement
           algorithms.
    """

    def __init__(self, d_model: int, num_layers: int, num_heads: int):
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Create attention heads
        d_head = d_model // num_heads
        self.heads = [
            [AttentionHead(d_model, d_head) for _ in range(num_heads)]
            for _ in range(num_layers)
        ]

        # Circuit tracer
        self.tracer = CircuitTracer(num_layers, num_heads)

        # Induction head detector
        self.induction = InductionHead(d_model, d_head)

    def analyze(
        self,
        clean_input: np.ndarray,
        corrupted_input: np.ndarray
    ) -> Dict:
        """
        Run complete circuit analysis.

        Args:
            clean_input: Clean input
            corrupted_input: Corrupted input

        Returns:
            Analysis results
        """
        # Get activations (simplified)
        clean_acts = self._get_activations(clean_input)
        corrupt_acts = self._get_activations(corrupted_input)

        # Trace circuit
        components = self.tracer.trace_circuit(clean_acts, corrupt_acts)

        # Find critical path
        critical = self.tracer.find_critical_path(components)

        return {
            'components': components,
            'critical_path': critical,
            'importance_matrix': self.tracer.importance
        }

    def _get_activations(self, x: np.ndarray) -> np.ndarray:
        """Get activations for all layers and heads."""
        batch, seq_len, _ = x.shape
        activations = np.zeros((batch, seq_len, self.num_layers, self.num_heads))

        h = x
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                out, _ = self.heads[layer][head].forward(h)
                activations[:, :, layer, head] = np.mean(out, axis=-1)
            h = h + out  # Residual

        return activations


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################

def demonstrate_circuit_analysis():
    """Demonstrate circuit analysis techniques."""
    print("=" * 70)
    print("CIRCUIT ANALYSIS")
    print("=" * 70)

    # Create analyzer
    d_model = 32
    num_layers = 3
    num_heads = 4
    analyzer = CircuitAnalyzer(d_model, num_layers, num_heads)

    # Create inputs
    batch_size = 2
    seq_len = 10
    clean_input = np.random.randn(batch_size, seq_len, d_model)
    corrupted_input = clean_input + np.random.randn(*clean_input.shape) * 0.5

    # Run analysis
    print("\n--- Circuit Analysis ---")
    results = analyzer.analyze(clean_input, corrupted_input)

    print(f"Total components: {len(results['components'])}")
    print(f"Critical path components: {len(results['critical_path'])}")

    print("\nTop 5 most important components:")
    for layer, head, importance in results['components'][:5]:
        print(f"  Layer {layer}, Head {head}: importance={importance:.4f}")

    # Induction head detection
    print("\n--- Induction Head Detection ---")
    tokens = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    patterns = analyzer.induction.detect_induction_pattern(tokens)

    print(f"Sequence: {tokens}")
    print(f"Induction patterns found: {len(patterns)}")
    for prev_pos, curr_pos, predicted in patterns[:5]:
        print(f"  Position {prev_pos}→{curr_pos}: predict token {predicted}")

    # Importance heatmap
    print("\n--- Importance Heatmap ---")
    importance = results['importance_matrix']
    for layer in range(num_layers):
        vals = [f"{importance[layer, h]:.3f}" for h in range(num_heads)]
        print(f"  Layer {layer}: [{', '.join(vals)}]")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Circuits are algorithms implemented by neurons!")
    print("Induction heads enable in-context learning through pattern matching.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_circuit_analysis()
