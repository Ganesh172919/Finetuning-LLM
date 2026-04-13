"""
################################################################################
FEATURE VISUALIZATION — SEEING WHAT NEURAL NETWORKS LEARN
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Feature Visualization?
    Techniques to visualize and understand what individual neurons
    or layers in a neural network have learned to detect.

Why does it matter?
    Without visualization, we don't know:
    - What concepts a neuron represents
    - Whether features are meaningful or spurious
    - How information flows through the network

    Feature visualization reveals:
    - Neurons that detect specific patterns
    - Hierarchical feature learning (edges → textures → objects)
    - How the model represents concepts internally

Key Methods:
    1. Activation Maximization: Find input that maximizes neuron
    2. Feature Inversion: Reconstruct input from activations
    3. Activation Atlases: Visualize activation space
    4. Saliency Maps: Which input parts matter most

Architecture (Activation Maximization):
    ┌─────────────────────────────────────────────────────────────────┐
    │                Activation Maximization                          │
    │                                                                  │
    │  Random Input ──▶ Forward Pass ──▶ Get Neuron Activation       │
    │       ↑                                    ↓                    │
    │       │                              Compute Gradient           │
    │       │                                    ↓                    │
    │       └────────── Update Input to Maximize Activation           │
    │                                                                  │
    │  Repeat until convergence ──▶ Visualize Optimal Input          │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "What does a neuron in a neural network detect?"
       Each neuron responds to specific patterns. Early layers detect
       edges and textures. Deeper layers detect objects and concepts.
       The optimal input for a neuron shows what it's "looking for."

    2. "What is activation maximization?"
       Optimization: find the input that maximizes a neuron's activation.
       This reveals the neuron's "preferred" input pattern.

    3. "How does this relate to adversarial examples?"
       Adversarial examples exploit the same gradient-based optimization
       to find inputs that cause misclassification. Feature visualization
       uses it to understand (not attack) the model.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: ACTIVATION MAXIMIZATION
################################################################################

class ActivationMaximization:
    """
    Activation Maximization
    ========================

    Finds the input that maximizes a specific neuron's activation.

    Algorithm:
    1. Start with random input
    2. Forward pass: compute neuron activation
    3. Backward pass: compute gradient of activation w.r.t. input
    4. Update input in direction of gradient
    5. Repeat until convergence

    This reveals what pattern the neuron is "looking for."

    Formula:
        x* = argmax_x f_neuron(x)
        x_{t+1} = x_t + α × ∇_x f_neuron(x_t)

    Interview Question:
        "Why do we need regularization for activation maximization?"
        Without regularization, the optimized input looks like noise.
        Regularization (blur, clipping, prior) encourages natural-
        looking inputs that reveal interpretable features.
    """

    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 0.01,
        num_steps: int = 100
    ):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.num_steps = num_steps

    def maximize(
        self,
        model_weights: np.ndarray,
        target_neuron: int,
        regularization: float = 0.01
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Find input that maximizes target neuron activation.

        Args:
            model_weights: Weight matrix (input_dim, hidden_dim)
            target_neuron: Index of neuron to visualize
            regularization: L2 regularization strength

        Returns:
            (optimal_input, activation_history)
        """
        # Initialize with random input
        x = np.random.randn(self.input_dim) * 0.1

        activation_history = []

        for step in range(self.num_steps):
            # Forward pass: compute activation
            activation = x @ model_weights[:, target_neuron]

            # Record activation
            activation_history.append(float(activation))

            # Gradient of activation w.r.t. input
            gradient = model_weights[:, target_neuron]

            # L2 regularization gradient
            reg_gradient = regularization * x

            # Update input
            x = x + self.learning_rate * (gradient - reg_gradient)

            # Clip to reasonable range
            x = np.clip(x, -3.0, 3.0)

        return x, activation_history

    def visualize_features(
        self,
        model_weights: np.ndarray,
        num_neurons: int = 10
    ) -> Dict[int, np.ndarray]:
        """
        Visualize features for multiple neurons.

        Args:
            model_weights: Weight matrix
            num_neurons: Number of neurons to visualize

        Returns:
            Dictionary mapping neuron index to optimal input
        """
        features = {}

        for neuron in range(min(num_neurons, model_weights.shape[1])):
            optimal_input, _ = self.maximize(model_weights, neuron)
            features[neuron] = optimal_input

        return features


################################################################################
# SECTION 2: SALIENCY MAP
################################################################################

class SaliencyMap:
    """
    Saliency Map
    =============

    Shows which parts of the input are most important for a prediction.

    Method:
    1. Forward pass: compute prediction
    2. Backward pass: compute gradient of prediction w.r.t. input
    3. Gradient magnitude = importance of each input feature

    High gradient = input feature is important for the prediction.

    Interview Question:
        "What's the difference between saliency maps and attention?"
        Attention shows where the model LOOKS (attention weights).
        Saliency shows where the model's prediction CHANGES (gradients).
        They can differ — the model might attend to something
        but not use it for the final prediction.
    """

    def __init__(self):
        pass

    def compute_saliency(
        self,
        model_weights: np.ndarray,
        input_features: np.ndarray,
        output_class: int
    ) -> np.ndarray:
        """
        Compute saliency map for a prediction.

        Args:
            model_weights: Weight matrix (input_dim, output_dim)
            input_features: Input vector
            output_class: Output class to explain

        Returns:
            Saliency map (importance of each input feature)
        """
        # Gradient of output w.r.t. input
        # For linear model: gradient = weights[:, output_class]
        gradient = model_weights[:, output_class]

        # Saliency = absolute gradient
        saliency = np.abs(gradient * input_features)

        return saliency

    def top_features(
        self,
        saliency: np.ndarray,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Get top-k most important features.

        Args:
            saliency: Saliency map
            k: Number of features to return

        Returns:
            List of (feature_index, importance) sorted by importance
        """
        indices = np.argsort(saliency)[::-1][:k]
        return [(int(i), float(saliency[i])) for i in indices]


################################################################################
# SECTION 3: FEATURE VISUALIZER (COMPLETE)
################################################################################

class FeatureVisualizer:
    """
    Feature Visualizer
    ===================

    Complete toolkit for understanding what neural networks learn.

    Capabilities:
    1. Activation maximization: What does each neuron detect?
    2. Saliency maps: Which inputs matter for predictions?
    3. Feature analysis: How are features distributed?

    Use cases:
    - Debugging: Find neurons that learned wrong features
    - Safety: Detect neurons for sensitive concepts
    - Research: Understand how models represent information

    Interview Questions:
        1. "How do you use feature visualization in practice?"
           Visualize neurons in each layer to understand what
           the model has learned. Look for interpretable features.
           Flag neurons that respond to sensitive concepts.

        2. "What are the limitations of feature visualization?"
           (a) Optimized inputs may not look natural,
           (b) Single neuron may not capture full concept,
           (c) Superposition makes interpretation harder.

        3. "How does this relate to safety?"
           Feature visualization can detect neurons that respond
           to harmful content, deceptive behavior, or other
           safety-relevant concepts.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize model weights (simplified single layer)
        self.weights = np.random.randn(input_dim, hidden_dim) * 0.02

        # Components
        self.activation_max = ActivationMaximization(input_dim)
        self.saliency = SaliencyMap()

    def analyze_layer(self, layer_name: str = "layer_0") -> Dict:
        """
        Analyze all neurons in a layer.

        Returns:
            Analysis results including feature visualizations
        """
        print(f"Analyzing {layer_name}...")

        # Visualize features for each neuron
        features = self.activation_max.visualize_features(
            self.weights, num_neurons=self.hidden_dim
        )

        # Compute feature statistics
        feature_norms = {k: np.linalg.norm(v) for k, v in features.items()}

        # Find similar neurons (feature clustering)
        similar_pairs = self._find_similar_neurons(features)

        return {
            'layer': layer_name,
            'features': features,
            'feature_norms': feature_norms,
            'similar_pairs': similar_pairs
        }

    def _find_similar_neurons(
        self,
        features: Dict[int, np.ndarray],
        threshold: float = 0.8
    ) -> List[Tuple[int, int, float]]:
        """Find neurons with similar feature visualizations."""
        similar = []
        neuron_ids = list(features.keys())

        for i in range(len(neuron_ids)):
            for j in range(i + 1, len(neuron_ids)):
                f1 = features[neuron_ids[i]]
                f2 = features[neuron_ids[j]]

                # Cosine similarity
                cos_sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)

                if cos_sim > threshold:
                    similar.append((neuron_ids[i], neuron_ids[j], float(cos_sim)))

        return similar

    def explain_prediction(
        self,
        input_features: np.ndarray,
        output_class: int
    ) -> Dict:
        """
        Explain a specific prediction.

        Args:
            input_features: Input that was classified
            output_class: Predicted class

        Returns:
            Explanation including saliency and important features
        """
        # Compute saliency
        saliency_map = self.saliency.compute_saliency(
            self.weights, input_features, output_class
        )

        # Get top features
        top_k = self.saliency.top_features(saliency_map, k=5)

        # Compute prediction
        prediction = input_features @ self.weights[:, output_class]

        return {
            'output_class': output_class,
            'prediction_value': float(prediction),
            'saliency_map': saliency_map,
            'top_features': top_k
        }


################################################################################
# SECTION 4: TESTING & DEMONSTRATION
################################################################################

def demonstrate_feature_visualization():
    """Demonstrate feature visualization techniques."""
    print("=" * 70)
    print("FEATURE VISUALIZATION")
    print("=" * 70)

    # Create visualizer
    input_dim = 20
    hidden_dim = 10
    visualizer = FeatureVisualizer(input_dim, hidden_dim)

    # Make some neurons more interpretable
    # Neuron 0: responds to first 5 features
    visualizer.weights[:5, 0] = 1.0
    visualizer.weights[5:, 0] = 0.0

    # Neuron 1: responds to features 5-10
    visualizer.weights[:5, 1] = 0.0
    visualizer.weights[5:10, 1] = 1.0
    visualizer.weights[10:, 1] = 0.0

    # Activation maximization
    print("\n--- Activation Maximization ---")
    for neuron in range(3):
        optimal_input, history = visualizer.activation_max.maximize(
            visualizer.weights, neuron, num_steps=50
        )
        print(f"  Neuron {neuron}: final_activation={history[-1]:.3f}")
        print(f"    Top features: {np.argsort(np.abs(optimal_input))[-5:].tolist()}")

    # Saliency maps
    print("\n--- Saliency Maps ---")
    input_features = np.random.randn(input_dim)
    explanation = visualizer.explain_prediction(input_features, output_class=0)

    print(f"  Prediction: {explanation['prediction_value']:.3f}")
    print(f"  Top features: {explanation['top_features']}")

    # Layer analysis
    print("\n--- Layer Analysis ---")
    analysis = visualizer.analyze_layer("layer_0")

    print(f"  Number of neurons: {len(analysis['features'])}")
    print(f"  Similar neuron pairs: {len(analysis['similar_pairs'])}")

    if analysis['similar_pairs']:
        for n1, n2, sim in analysis['similar_pairs'][:3]:
            print(f"    Neuron {n1} ≈ Neuron {n2}: similarity={sim:.3f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Feature visualization reveals what neurons detect!")
    print("Activation maximization finds the 'preferred input' for each neuron.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_feature_visualization()
