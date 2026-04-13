"""
################################################################################
CAUSAL WORLD MODEL — LEARNING CAUSE AND EFFECT
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Causal World Model?
    A world model that understands CAUSAL relationships — not just
    correlations, but what actually causes what. This enables:
    - Interventional reasoning: "What if I do X?"
    - Counterfactual thinking: "What would have happened if I did Y?"
    - Causal discovery: Learning the causal graph from data

Why does it matter?
    Standard ML models learn correlations, not causation:
    - "Ice cream sales correlate with drowning" (both caused by summer)
    - A standard model might suggest: "Reduce ice cream → less drowning"
    - A causal model knows: summer → ice cream AND summer → drowning

    Causal models enable:
    - Robust predictions under distribution shift
    - Effective interventions and planning
    - Understanding WHY something happens
    - Generalization to new environments

How does it work?
    1. Learn a Structural Causal Model (SCM) from observations
    2. SCMs define: X_i = f_i(Parents(X_i), noise_i)
    3. Use the SCM for:
       - Prediction: Given current state, predict next state
       - Intervention: Change a variable, propagate effects
       - Counterfactual: What if past was different?

Causal Hierarchy (Pearl's Ladder):
    Level 1: Association — P(Y|X) [seeing]
    Level 2: Intervention — P(Y|do(X)) [doing]
    Level 3: Counterfactual — P(Y_x|X', Y') [imagining]

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Causal World Model                               │
    │                                                                  │
    │  Variables: X1, X2, X3, X4                                      │
    │                                                                  │
    │  Causal Graph:                                                   │
    │       X1 ──▶ X2 ──▶ X4                                          │
    │       │                    (X1 causes X2, X2 causes X4)         │
    │       └──▶ X3              (X1 also causes X3)                  │
    │                                                                  │
    │  Structural Equations:                                           │
    │       X2 = f2(X1, ε2)                                           │
    │       X3 = f3(X1, ε3)                                           │
    │       X4 = f4(X2, ε4)                                           │
    │                                                                  │
    │  The model learns these f_i functions from data.                │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "What's the difference between correlation and causation?"
       Correlation: X and Y co-occur. Causation: X causes Y.
       Example: Umbrella use correlates with rain, but umbrellas
       don't cause rain. Rain causes umbrella use.

    2. "How do you learn causal structure from data?"
       Methods: (a) Randomized controlled trials (gold standard),
       (b) Instrumental variables, (c) Granger causality for time
       series, (d) Structure learning algorithms (PC, FCI, GES).

    3. "Why are causal models better for planning?"
       Planning requires predicting effects of ACTIONS (interventions).
       Correlation-based models predict effects of OBSERVATIONS.
       Only causal models correctly predict: P(Y|do(X)) vs P(Y|X).

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass, field
import math

################################################################################
# SECTION 1: CAUSAL GRAPH
################################################################################

class CausalGraph:
    """
    Causal Graph (Directed Acyclic Graph)
    =======================================

    Represents causal relationships between variables.

    A DAG where:
    - Nodes are variables
    - Edges are causal relationships (X → Y means X causes Y)

    Key properties:
    - Parents(X): Variables that directly cause X
    - Children(X): Variables directly caused by X
    - Ancestors(X): All upstream causes
    - Descendants(X): All downstream effects

    Interview Question:
        "Can cycles exist in causal graphs?"
        In standard SCMs, no — the graph must be a DAG.
        But in time-series, you can have feedback loops:
        X_t → Y_{t+1} → X_{t+2}. This is acyclic across time.
    """

    def __init__(self, num_variables: int, variable_names: Optional[List[str]] = None):
        self.num_variables = num_variables
        self.names = variable_names or [f"X{i}" for i in range(num_variables)]

        # Adjacency list: parents[i] = set of parent indices
        self.parents: List[Set[int]] = [set() for _ in range(num_variables)]
        self.children: List[Set[int]] = [set() for _ in range(num_variables)]

    def add_edge(self, cause: int, effect: int):
        """
        Add a causal edge: cause → effect.

        Args:
            cause: Index of causing variable
            effect: Index of affected variable
        """
        self.parents[effect].add(cause)
        self.children[cause].add(effect)

    def get_parents(self, node: int) -> Set[int]:
        """Get direct causes of a variable."""
        return self.parents[node]

    def get_children(self, node: int) -> Set[int]:
        """Get direct effects of a variable."""
        return self.children[node]

    def get_ancestors(self, node: int) -> Set[int]:
        """Get all upstream causes (transitive closure)."""
        ancestors = set()
        frontier = list(self.parents[node])
        while frontier:
            n = frontier.pop()
            if n not in ancestors:
                ancestors.add(n)
                frontier.extend(self.parents[n])
        return ancestors

    def get_descendants(self, node: int) -> Set[int]:
        """Get all downstream effects."""
        descendants = set()
        frontier = list(self.children[node])
        while frontier:
            n = frontier.pop()
            if n not in descendants:
                descendants.add(n)
                frontier.extend(self.children[n])
        return descendants

    def topological_order(self) -> List[int]:
        """
        Get topological ordering of variables.

        Variables are ordered so that causes come before effects.
        This is the order in which we compute values in an SCM.

        Algorithm: Kahn's algorithm (BFS-based topological sort)
        """
        in_degree = [len(self.parents[i]) for i in range(self.num_variables)]
        queue = [i for i in range(self.num_variables) if in_degree[i] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return order

    def is_valid_dag(self) -> bool:
        """Check if the graph is a valid DAG (no cycles)."""
        try:
            order = self.topological_order()
            return len(order) == self.num_variables
        except:
            return False

    def visualize(self) -> str:
        """Create ASCII visualization of the causal graph."""
        lines = ["Causal Graph:"]
        lines.append("=" * 50)

        for i in range(self.num_variables):
            name = self.names[i]
            parents = [self.names[p] for p in self.parents[i]]
            if parents:
                lines.append(f"  {name} ← {', '.join(parents)}")
            else:
                lines.append(f"  {name} (root cause)")

        lines.append("")
        lines.append("Edges:")
        for i in range(self.num_variables):
            for child in self.children[i]:
                lines.append(f"  {self.names[i]} → {self.names[child]}")

        return "\n".join(lines)


################################################################################
# SECTION 2: STRUCTURAL CAUSAL MODEL
################################################################################

class StructuralCausalModel:
    """
    Structural Causal Model (SCM)
    ===============================

    An SCM defines how each variable is generated from its causes.

    Components:
    - Graph: Causal relationships (DAG)
    - Functions: X_i = f_i(Parents(X_i), ε_i)
    - Noise: Exogenous variables ε_i ~ P(ε)

    The SCM fully specifies the data generating process.

    Example (3 variables):
        X1 = ε1                           (root cause)
        X2 = 2*X1 + ε2                   (X1 causes X2)
        X3 = 0.5*X1 + 0.3*X2 + ε3       (X1 and X2 cause X3)

    Interview Questions:
        1. "What is an SCM?"
           A Structural Causal Model defines variables, their causal
           relationships (as a DAG), and the functional form of each
           relationship. It's a complete specification of how data
           is generated.

        2. "What are exogenous variables?"
           Noise terms (ε) that capture unobserved factors. They're
           assumed independent of each other. The observed variables
           are called endogenous.

        3. "How does an SCM enable counterfactuals?"
           Fix the noise terms (from observation), then change
           a variable and propagate through the structural equations.
           This gives the counterfactual outcome.
    """

    def __init__(self, graph: CausalGraph):
        self.graph = graph
        self.functions = {}  # node_idx -> function
        self.noise_distributions = {}  # node_idx -> (mean, std)

    def set_function(self, node: int, func, noise_std: float = 1.0):
        """
        Set the structural equation for a variable.

        Args:
            node: Variable index
            func: Function(parents_values) -> value
            noise_std: Standard deviation of noise term
        """
        self.functions[node] = func
        self.noise_distributions[node] = (0.0, noise_std)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample from the SCM.

        Process:
        1. Get topological order
        2. For each variable (in order):
           a. Sample noise ε_i
           b. Compute X_i = f_i(Parents(X_i), ε_i)

        Args:
            n_samples: Number of samples to generate

        Returns:
            Data array (n_samples, num_variables)
        """
        order = self.graph.topological_order()
        data = np.zeros((n_samples, self.graph.num_variables))

        # Sample all noise terms
        noise = {}
        for node in range(self.graph.num_variables):
            mean, std = self.noise_distributions.get(node, (0, 1))
            noise[node] = np.random.randn(n_samples) * std + mean

        # Compute values in topological order
        for node in order:
            if node in self.functions:
                parent_values = {p: data[:, p] for p in self.graph.parents[node]}
                data[:, node] = self.functions[node](parent_values, noise[node])
            else:
                # Default: just noise (root cause)
                data[:, node] = noise[node]

        return data

    def intervene(self, node: int, value: float, n_samples: int = 1) -> np.ndarray:
        """
        Perform intervention: do(X_i = value).

        This is the "do-operator" from Pearl's framework.
        Unlike conditioning (observing), intervention:
        1. Removes all incoming edges to the intervened variable
        2. Sets it to the specified value
        3. Propagates effects downstream

        Interview Question:
            "What's the difference between P(Y|X) and P(Y|do(X))?"
            P(Y|X): Observe X, update belief about Y (conditioning)
            P(Y|do(X)): Set X to a value, compute Y (intervention)
            Example: P(lung cancer | smoking) ≠ P(lung cancer | do(smoking))
            because confounders (genetics) affect both.

        Args:
            node: Variable to intervene on
            value: Value to set
            n_samples: Number of samples

        Returns:
            Data after intervention
        """
        order = self.graph.topological_order()
        data = np.zeros((n_samples, self.graph.num_variables))

        # Sample noise (but not for intervened node)
        noise = {}
        for n in range(self.graph.num_variables):
            mean, std = self.noise_distributions.get(n, (0, 1))
            noise[n] = np.random.randn(n_samples) * std + mean

        for n in order:
            if n == node:
                # Intervention: set to fixed value
                data[:, n] = value
            elif n in self.functions:
                # Normal structural equation
                parent_values = {p: data[:, p] for p in self.graph.parents[n]}
                data[:, n] = self.functions[n](parent_values, noise[n])
            else:
                data[:, n] = noise[n]

        return data

    def counterfactual(
        self,
        observed: np.ndarray,
        intervention_node: int,
        intervention_value: float
    ) -> np.ndarray:
        """
        Compute counterfactual: "What would have happened if X was different?"

        Steps (Abduction-Action-Prediction):
        1. Abduction: Given observation, infer noise terms
        2. Action: Apply intervention
        3. Prediction: Propagate with fixed noise

        This answers: Given what I observed, what WOULD HAVE happened
        if I had done something different?

        Args:
            observed: Observed data point (1, num_variables)
            intervention_node: Variable to change
            intervention_value: New value

        Returns:
            Counterfactual outcome
        """
        # Step 1: Abduction — infer noise from observation
        inferred_noise = {}
        for node in range(self.graph.num_variables):
            if node in self.functions:
                parent_values = {p: observed[0, p] for p in self.graph.parents[node]}
                # noise = observed - f(parents)
                inferred_noise[node] = observed[0, node] - self.functions[node](
                    parent_values, np.zeros(1)
                )[0]
            else:
                inferred_noise[node] = observed[0, node]

        # Step 2 & 3: Action and Prediction
        order = self.graph.topological_order()
        result = np.zeros_like(observed)

        for node in order:
            if node == intervention_node:
                result[0, node] = intervention_value
            elif node in self.functions:
                parent_values = {p: result[0, p] for p in self.graph.parents[node]}
                result[0, node] = self.functions[node](
                    parent_values, np.array([inferred_noise[node]])
                )[0]
            else:
                result[0, node] = inferred_noise[node]

        return result


################################################################################
# SECTION 3: CAUSAL DISCOVERY
################################################################################

class CausalDiscovery:
    """
    Causal Discovery Algorithms
    ==============================

    Learn causal structure from observational data.

    Methods:
    1. Granger Causality: Time-series based
    2. PC Algorithm: Constraint-based
    3. NOTEARS: Continuous optimization for DAG learning

    Interview Question:
        "Can you learn causation from observational data alone?"
        Not definitively. Observational data can narrow down
        possible causal structures, but cannot uniquely determine
        them (Markov equivalence class). Interventional data
        (experiments) is needed for unique identification.
    """

    @staticmethod
    def granger_causality(
        data: np.ndarray,
        max_lag: int = 1
    ) -> np.ndarray:
        """
        Granger Causality test for time-series data.

        X Granger-causes Y if past values of X help predict Y
        beyond what past values of Y alone can predict.

        Test:
        1. Restricted model: Y_t = f(Y_{t-1}, ..., Y_{t-lag})
        2. Unrestricted model: Y_t = f(Y_{t-1}, ..., Y_{t-lag}, X_{t-1}, ..., X_{t-lag})
        3. If unrestricted is significantly better → X Granger-causes Y

        Args:
            data: Time series data (time_steps, num_variables)
            max_lag: Maximum lag to consider

        Returns:
            Causality matrix (num_vars, num_vars) — entry (i,j) = 1 if i→j
        """
        n_vars = data.shape[1]
        causality = np.zeros((n_vars, n_vars))

        for cause in range(n_vars):
            for effect in range(n_vars):
                if cause == effect:
                    continue

                # Build lagged features
                T = data.shape[0] - max_lag

                # Restricted model: only past values of effect
                X_restricted = np.zeros((T, max_lag))
                for lag in range(1, max_lag + 1):
                    X_restricted[:, lag-1] = data[max_lag-lag:T+max_lag-lag, effect]

                # Unrestricted model: add past values of cause
                X_unrestricted = np.zeros((T, 2 * max_lag))
                for lag in range(1, max_lag + 1):
                    X_unrestricted[:, lag-1] = data[max_lag-lag:T+max_lag-lag, effect]
                    X_unrestricted[:, max_lag+lag-1] = data[max_lag-lag:T+max_lag-lag, cause]

                y = data[max_lag:T+max_lag, effect]

                # Fit models (simplified with least squares)
                try:
                    # Restricted
                    beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
                    resid_r = y - X_restricted @ beta_r
                    rss_r = np.sum(resid_r ** 2)

                    # Unrestricted
                    beta_u = np.linalg.lstsq(X_unrestricted, y, rcond=None)[0]
                    resid_u = y - X_unrestricted @ beta_u
                    rss_u = np.sum(resid_u ** 2)

                    # F-test (simplified)
                    n = len(y)
                    f_stat = ((rss_r - rss_u) / max_lag) / (rss_u / (n - 2*max_lag - 1))

                    # If F-statistic is large, X Granger-causes Y
                    if f_stat > 3.84:  # Approximate critical value
                        causality[cause, effect] = 1.0
                except:
                    pass

        return causality


################################################################################
# SECTION 4: CAUSAL WORLD MODEL (COMPLETE)
################################################################################

class CausalWorldModel:
    """
    Causal World Model
    ====================

    Combines causal reasoning with world modeling.

    Capabilities:
    1. Predict future states using causal structure
    2. Perform interventions (what if I do X?)
    3. Compute counterfactuals (what would have happened?)
    4. Discover causal structure from data

    Use cases:
    - Robotics: Understand cause-effect in physical world
    - Healthcare: Predict treatment effects
    - Economics: Model policy interventions
    - Science: Discover causal mechanisms

    Interview Questions:
        1. "How does a causal world model differ from a standard one?"
           Standard models learn P(Y|X) — correlations.
           Causal models learn P(Y|do(X)) — effects of actions.
           This is crucial for planning and intervention.

        2. "When is causality important in AI?"
           When the model will be used for decision-making (actions),
           when the distribution may shift, when you need to explain
           WHY something happens, when confounders exist.

        3. "What's the biggest challenge in causal AI?"
           Identifying causation from observational data alone.
           Often requires domain knowledge or experiments.
    """

    def __init__(self, num_variables: int, variable_names: Optional[List[str]] = None):
        self.graph = CausalGraph(num_variables, variable_names)
        self.scm = StructuralCausalModel(self.graph)
        self.data = None

    def add_causal_relation(
        self,
        cause: int,
        effect: int,
        strength: float = 1.0,
        noise_std: float = 0.1
    ):
        """
        Add a causal relationship.

        Args:
            cause: Causing variable index
            effect: Affected variable index
            strength: Causal strength (coefficient)
            noise_std: Noise in the relationship
        """
        self.graph.add_edge(cause, effect)

        # Define structural equation
        old_func = self.scm.functions.get(effect)

        def make_func(cause_idx, strength_val, old_fn):
            def func(parents, noise):
                contribution = strength_val * parents[cause_idx]
                if old_fn is not None:
                    return old_fn(parents, noise) + contribution
                return contribution + noise
            return func

        self.scm.functions[effect] = make_func(cause, strength, old_func)
        self.scm.noise_distributions[effect] = (0, noise_std)

    def generate_data(self, n_samples: int = 1000) -> np.ndarray:
        """Generate observational data from the SCM."""
        self.data = self.scm.sample(n_samples)
        return self.data

    def predict_intervention(
        self,
        variable: int,
        value: float,
        n_samples: int = 100
    ) -> np.ndarray:
        """
        Predict outcome of intervention: do(X = value).

        Args:
            variable: Variable to intervene on
            value: Intervention value
            n_samples: Number of samples

        Returns:
            Predicted outcomes after intervention
        """
        return self.scm.intervene(variable, value, n_samples)

    def compute_counterfactual(
        self,
        observation: np.ndarray,
        variable: int,
        value: float
    ) -> np.ndarray:
        """
        Compute counterfactual for an observation.

        Args:
            observation: Observed data point
            variable: Variable to change
            value: New value

        Returns:
            Counterfactual outcome
        """
        return self.scm.counterfactual(observation, variable, value)

    def discover_structure(self, data: np.ndarray) -> np.ndarray:
        """
        Discover causal structure from data using Granger causality.

        Args:
            data: Observational data (time_steps, num_variables)

        Returns:
            Adjacency matrix of discovered causal graph
        """
        return CausalDiscovery.granger_causality(data)


################################################################################
# SECTION 5: TESTING & DEMONSTRATION
################################################################################

def demonstrate_causal_world_model():
    """Demonstrate causal world model capabilities."""
    print("=" * 70)
    print("CAUSAL WORLD MODEL")
    print("=" * 70)

    # Create model with 4 variables
    model = CausalWorldModel(
        num_variables=4,
        variable_names=["Temperature", "IceCream", "Sunscreen", "Drowning"]
    )

    # Define causal structure:
    # Temperature → IceCream
    # Temperature → Sunscreen
    # Temperature → Drowning
    # (IceCream does NOT cause Drowning — it's a confounder!)
    model.add_causal_relation(0, 1, strength=0.8, noise_std=0.1)  # Temp → IceCream
    model.add_causal_relation(0, 2, strength=0.6, noise_std=0.1)  # Temp → Sunscreen
    model.add_causal_relation(0, 3, strength=0.5, noise_std=0.2)  # Temp → Drowning

    # Visualize graph
    print("\n" + model.graph.visualize())

    # Generate observational data
    print("\nGenerating observational data...")
    data = model.generate_data(1000)
    print(f"Data shape: {data.shape}")
    print(f"Correlation(IceCream, Drowning): {np.corrcoef(data[:, 1], data[:, 3])[0,1]:.3f}")

    # Intervention: do(Temperature = 0.5)
    print("\n--- Intervention: do(Temperature = 0.5) ---")
    interv_data = model.predict_intervention(0, value=0.5, n_samples=100)
    print(f"After intervention, mean Drowning: {interv_data[:, 3].mean():.3f}")

    # Intervention: do(IceCream = 0) — does reducing ice cream reduce drowning?
    print("\n--- Intervention: do(IceCream = 0) ---")
    interv_data2 = model.predict_intervention(1, value=0.0, n_samples=100)
    print(f"After intervention, mean Drowning: {interv_data2[:, 3].mean():.3f}")
    print("(Should be SAME as observational — ice cream doesn't cause drowning!)")

    # Counterfactual
    print("\n--- Counterfactual ---")
    obs = data[0:1]  # Single observation
    print(f"Observed Temperature: {obs[0, 0]:.3f}")
    print(f"Observed Drowning: {obs[0, 3]:.3f}")

    cf = model.compute_counterfactual(obs, variable=0, value=2.0)
    print(f"Counterfactual (if Temp was 2.0): Drowning = {cf[0, 3]:.3f}")

    # Causal discovery
    print("\n--- Causal Discovery (Granger Causality) ---")
    causality = model.discover_structure(data)
    print("Discovered causal matrix:")
    print(causality.astype(int))

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Correlation ≠ Causation!")
    print("Ice cream correlates with drowning, but doesn't cause it.")
    print("Only causal models can correctly predict intervention effects.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_causal_world_model()
