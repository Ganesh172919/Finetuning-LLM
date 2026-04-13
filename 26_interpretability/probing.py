"""
################################################################################
PROBING — UNDERSTANDING INTERNAL REPRESENTATIONS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Probing?
    Training simple classifiers on top of neural network representations
    to understand WHAT information is encoded at each layer.

    A "probe" is a small model (usually linear) that tries to predict
    some property from the network's internal activations.

    If the probe succeeds, the information IS in the representation.
    If it fails, the information is NOT there (or not easily accessible).

Why does it matter?
    We don't know what information neural networks encode internally.
    Probing reveals:
    - Does the model know part-of-speech at layer 3?
    - Does it encode sentiment at layer 6?
    - Where does factual knowledge reside?

    This helps us understand:
    - How information flows through layers
    - Where specific capabilities emerge
    - What the model has learned

Key Methods:
    1. Linear Probing: Simple linear classifier
    2. Representation Analysis: Compare representations across layers
    3. Concept Activation Vectors: Directions in activation space

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Linear Probing                                  │
    │                                                                  │
    │  Input ──▶ Model ──▶ Layer k activations ──▶ Linear Probe      │
    │                                                      ↓          │
    │                                              Predicted Property │
    │                                                      ↓          │
    │                                              Compare with Truth │
    │                                                                  │
    │  If accuracy is high → property is encoded at layer k          │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "What is a linear probe?"
       A simple linear classifier trained on a model's internal
       representations to predict some property. If it works,
       the property is linearly decodable from the representation.

    2. "Why linear and not a complex probe?"
       Complex probes can learn the property themselves, not just
       extract it. Linear probes tell us what's ALREADY in the
       representation, not what could be learned from it.

    3. "Where do facts live in a transformer?"
       Research shows factual knowledge is primarily stored in the
       middle-to-late MLP layers. Early layers handle syntax;
       later layers handle semantics and facts.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: LINEAR PROBE
################################################################################

class LinearProbe:
    """
    Linear Probe
    =============

    Trains a linear classifier on internal representations.

    The probe answers: "Is property X encoded at layer Y?"

    Training:
        1. Run model on inputs, collect activations at layer Y
        2. Train linear classifier to predict property X from activations
        3. If accuracy is high → property X is encoded at layer Y

    Formula:
        prediction = softmax(W @ activation + b)
        loss = cross_entropy(prediction, truth)

    Interview Questions:
        1. "How do you choose which layer to probe?"
           Probe all layers! This reveals how information evolves
           across the network. Plot accuracy vs layer to see
           where each property emerges.

        2. "Can probing be misleading?"
           Yes. High probe accuracy doesn't mean the model USES
           that information. The information might be present but
           not connected to the model's behavior. Use causal
           interventions to verify.
    """

    def __init__(self, activation_dim: int, num_classes: int, learning_rate: float = 0.01):
        self.activation_dim = activation_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Linear classifier weights
        self.W = np.random.randn(activation_dim, num_classes) * 0.02
        self.b = np.zeros(num_classes)

    def forward(self, activations: np.ndarray) -> np.ndarray:
        """
        Forward pass through probe.

        Args:
            activations: Model activations (batch, activation_dim)

        Returns:
            Class probabilities (batch, num_classes)
        """
        logits = activations @ self.W + self.b
        return self._softmax(logits)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def compute_loss(
        self,
        activations: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute cross-entropy loss.

        Args:
            activations: Model activations
            labels: True labels (integer class indices)

        Returns:
            Scalar loss
        """
        probs = self.forward(activations)
        batch_size = activations.shape[0]

        # Cross-entropy loss
        log_probs = np.log(probs[np.arange(batch_size), labels] + 1e-8)
        return -np.mean(log_probs)

    def train_step(
        self,
        activations: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        One training step for the probe.

        Args:
            activations: Model activations
            labels: True labels

        Returns:
            Loss value
        """
        batch_size = activations.shape[0]

        # Forward
        probs = self.forward(activations)
        loss = self.compute_loss(activations, labels)

        # Backward (simplified gradient)
        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), labels] -= 1
        grad_logits /= batch_size

        # Update weights
        self.W -= self.learning_rate * (activations.T @ grad_logits)
        self.b -= self.learning_rate * np.mean(grad_logits, axis=0)

        return loss

    def predict(self, activations: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.forward(activations)
        return np.argmax(probs, axis=-1)

    def accuracy(self, activations: np.ndarray, labels: np.ndarray) -> float:
        """Compute prediction accuracy."""
        predictions = self.predict(activations)
        return float(np.mean(predictions == labels))


################################################################################
# SECTION 2: CONCEPT ACTIVATION VECTOR
################################################################################

class ConceptActivationVector:
    """
    Concept Activation Vector (CAV)
    =================================

    Finds directions in activation space that correspond to concepts.

    Method:
    1. Collect examples with concept (e.g., "happy") and without (e.g., "sad")
    2. Train linear classifier to separate them
    3. The decision boundary direction is the CAV

    The CAV points in the direction of the concept in activation space.

    Interview Question:
        "What is a Concept Activation Vector?"
        A direction in the neural network's activation space that
        corresponds to a human-understandable concept. For example,
        a "positive sentiment" CAV points in the direction that
        activations move when sentiment becomes more positive.
    """

    def __init__(self, activation_dim: int):
        self.activation_dim = activation_dim
        self.cav = None

    def fit(
        self,
        positive_activations: np.ndarray,
        negative_activations: np.ndarray
    ) -> np.ndarray:
        """
        Compute CAV from positive and negative examples.

        Args:
            positive_activations: Activations with concept
            negative_activations: Activations without concept

        Returns:
            Concept Activation Vector
        """
        # Compute mean of each class
        pos_mean = np.mean(positive_activations, axis=0)
        neg_mean = np.mean(negative_activations, axis=0)

        # CAV is the direction from negative to positive
        self.cav = pos_mean - neg_mean

        # Normalize
        self.cav = self.cav / (np.linalg.norm(self.cav) + 1e-8)

        return self.cav

    def score(self, activation: np.ndarray) -> float:
        """
        Score an activation along the CAV direction.

        Positive score = concept is present
        Negative score = concept is absent

        Args:
            activation: Single activation vector

        Returns:
            Concept score
        """
        if self.cav is None:
            raise ValueError("CAV not fitted yet")

        return float(np.dot(activation, self.cav))

    def score_batch(self, activations: np.ndarray) -> np.ndarray:
        """Score a batch of activations."""
        return activations @ self.cav


################################################################################
# SECTION 3: REPRESENTATION ANALYSIS
################################################################################

class RepresentationAnalysis:
    """
    Representation Analysis
    ========================

    Analyzes how representations change across layers.

    Methods:
    1. Centered Kernel Alignment (CKA): Compare representations
    2. Representational Similarity Analysis (RSA)
    3. Linear interpolation between layers

    Interview Question:
        "How do representations evolve across layers?"
        Early layers capture syntax (word order, POS tags).
        Middle layers capture semantics (meaning, relations).
        Late layers capture task-specific information (predictions).
        This progression is consistent across many models.
    """

    def __init__(self):
        pass

    def compute_cka(
        self,
        representations_1: np.ndarray,
        representations_2: np.ndarray
    ) -> float:
        """
        Compute Centered Kernel Alignment between two representations.

        CKA measures similarity between representations:
        - CKA = 1: Identical representations (up to rotation)
        - CKA = 0: Completely different

        Formula:
            CKA(K1, K2) = ||K1 @ K2||_F / (||K1||_F * ||K2||_F)

        Where K1, K2 are centered Gram matrices.

        Args:
            representations_1: First representation (n_samples, d1)
            representations_2: Second representation (n_samples, d2)

        Returns:
            CKA score (0 to 1)
        """
        # Center the representations
        r1 = representations_1 - np.mean(representations_1, axis=0)
        r2 = representations_2 - np.mean(representations_2, axis=0)

        # Compute Gram matrices
        K1 = r1 @ r1.T
        K2 = r2 @ r2.T

        # Center the Gram matrices
        n = K1.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K1_centered = H @ K1 @ H
        K2_centered = H @ K2 @ H

        # Compute CKA
        numerator = np.sum(K1_centered * K2_centered)
        denominator = np.sqrt(np.sum(K1_centered ** 2) * np.sum(K2_centered ** 2))

        if denominator < 1e-10:
            return 0.0

        return float(numerator / denominator)

    def analyze_layer_progression(
        self,
        layer_representations: List[np.ndarray]
    ) -> Dict:
        """
        Analyze how representations change across layers.

        Args:
            layer_representations: List of activations at each layer

        Returns:
            Analysis results
        """
        num_layers = len(layer_representations)

        # Compute CKA between all pairs of layers
        cka_matrix = np.zeros((num_layers, num_layers))
        for i in range(num_layers):
            for j in range(num_layers):
                cka_matrix[i, j] = self.compute_cka(
                    layer_representations[i],
                    layer_representations[j]
                )

        # Compute representation norms
        norms = [np.linalg.norm(r) for r in layer_representations]

        return {
            'cka_matrix': cka_matrix,
            'norms': norms,
            'num_layers': num_layers
        }


################################################################################
# SECTION 4: PROBING ANALYSIS (COMPLETE)
################################################################################

class ProbingAnalysis:
    """
    Complete Probing Analysis
    ===========================

    Runs probing experiments across all layers to understand
    what information is encoded where.

    Workflow:
    1. Collect activations at each layer
    2. Train linear probes for each property
    3. Compare accuracy across layers
    4. Identify where each property emerges

    Interview Questions:
        1. "What have probing experiments revealed?"
           (a) Syntax is in early layers, semantics in middle,
           (b) Factual knowledge is in middle-to-late MLP layers,
           (c) Task-specific info emerges in later layers.

        2. "How do you choose what to probe for?"
           Choose properties relevant to your question:
           - For capability: probe for task-specific features
           - For safety: probe for harmful content representations
           - For understanding: probe for linguistic properties
    """

    def __init__(self, d_model: int, num_layers: int):
        self.d_model = d_model
        self.num_layers = num_layers

        # Probes for each layer
        self.probes = {}  # property_name -> list of probes per layer

    def train_probe(
        self,
        property_name: str,
        layer_activations: List[np.ndarray],
        labels: np.ndarray,
        num_classes: int,
        num_epochs: int = 50
    ) -> Dict[int, float]:
        """
        Train probes for a property across all layers.

        Args:
            property_name: Name of property to probe
            layer_activations: Activations at each layer
            labels: Ground truth labels
            num_classes: Number of classes
            num_epochs: Training epochs

        Returns:
            Dictionary mapping layer to accuracy
        """
        accuracies = {}

        for layer_idx, activations in enumerate(layer_activations):
            # Create probe
            probe = LinearProbe(self.d_model, num_classes)

            # Train
            for epoch in range(num_epochs):
                loss = probe.train_step(activations, labels)

            # Evaluate
            acc = probe.accuracy(activations, labels)
            accuracies[layer_idx] = acc

            # Store probe
            if property_name not in self.probes:
                self.probes[property_name] = {}
            self.probes[property_name][layer_idx] = probe

        return accuracies

    def find_emergence_layer(
        self,
        accuracies: Dict[int, float],
        threshold: float = 0.7
    ) -> Optional[int]:
        """
        Find the layer where a property first emerges.

        Args:
            accuracies: Layer accuracies
            threshold: Accuracy threshold for emergence

        Returns:
            Layer index or None
        """
        for layer in sorted(accuracies.keys()):
            if accuracies[layer] >= threshold:
                return layer
        return None


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################

def demonstrate_probing():
    """Demonstrate probing techniques."""
    print("=" * 70)
    print("PROBING ANALYSIS")
    print("=" * 70)

    d_model = 32
    num_layers = 5
    num_samples = 100
    num_classes = 3

    # Simulate layer activations
    print("\n--- Simulating Layer Activations ---")
    layer_activations = []
    for layer in range(num_layers):
        # Later layers have more structured representations
        structure = (layer + 1) / num_layers
        acts = np.random.randn(num_samples, d_model) * (1 - structure)
        # Add class-specific signal (stronger in later layers)
        labels = np.random.randint(0, num_classes, num_samples)
        for i in range(num_samples):
            acts[i, labels[i]] += structure * 3.0
        layer_activations.append(acts)

    labels = np.random.randint(0, num_classes, num_samples)

    # Linear probing
    print("\n--- Linear Probing Across Layers ---")
    analysis = ProbingAnalysis(d_model, num_layers)
    accuracies = analysis.train_probe(
        "test_property",
        layer_activations,
        labels,
        num_classes,
        num_epochs=100
    )

    for layer, acc in sorted(accuracies.items()):
        bar = "█" * int(acc * 30)
        print(f"  Layer {layer}: {acc:.3f} {bar}")

    # Find emergence layer
    emergence = analysis.find_emergence_layer(accuracies, threshold=0.5)
    print(f"\nProperty emerges at layer: {emergence}")

    # Concept Activation Vector
    print("\n--- Concept Activation Vector ---")
    cav = ConceptActivationVector(d_model)

    # Create positive and negative examples
    pos_acts = np.random.randn(50, d_model) + 1.0
    neg_acts = np.random.randn(50, d_model) - 1.0

    cav_vector = cav.fit(pos_acts, neg_acts)
    print(f"  CAV norm: {np.linalg.norm(cav_vector):.3f}")

    # Score some activations
    test_act = np.random.randn(d_model)
    score = cav.score(test_act)
    print(f"  Test activation score: {score:.3f}")

    # Representation analysis
    print("\n--- Representation Analysis ---")
    rep_analysis = RepresentationAnalysis()

    cka = rep_analysis.compute_cka(layer_activations[0], layer_activations[-1])
    print(f"  CKA between layer 0 and layer {num_layers-1}: {cka:.3f}")

    progression = rep_analysis.analyze_layer_progression(layer_activations)

    print("\n  CKA Matrix (layer similarity):")
    for i in range(num_layers):
        vals = [f"{progression['cka_matrix'][i, j]:.2f}" for j in range(num_layers)]
        print(f"    Layer {i}: [{', '.join(vals)}]")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Probing reveals WHAT information is encoded WHERE!")
    print("Different properties emerge at different layers.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_probing()
