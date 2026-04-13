"""
################################################################################
WORLD MODELS — UNDERSTANDING PHYSICS AND CAUSALITY
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are World Models?
    Models that understand how the world works:
    - Physics (gravity, collisions)
    - Causality (cause and effect)
    - Object permanence
    - Spatial relationships

Key Models:
    - Genie (Google): World model from video
    - SORA (OpenAI): World simulator
    - UniSim: Universal simulator

Interview Questions:
    Q: "What is a world model?"
    A: A model that understands how the world changes given actions.
       Enables planning and simulation without real-world interaction.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: WORLD MODEL
################################################################################

class WorldModel:
    """
    World Model
    ===========

    Predicts future states given current state and action.

    s_{t+1} = f(s_t, a_t)

    Interview Questions:
        Q: "How do you train a world model?"
        A: On video data. Predict next frame given current frame and action.
    """

    def __init__(self, state_dim: int = 64, action_dim: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Dynamics model
        self.dynamics = np.random.randn(state_dim + action_dim, state_dim) * 0.02

    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict next state."""
        combined = np.concatenate([state, action])
        return combined @ self.dynamics

    def simulate(self, initial_state: np.ndarray, actions: List[np.ndarray]) -> List[np.ndarray]:
        """
        Simulate trajectory.

        Args:
            initial_state: Starting state
            actions: Sequence of actions

        Returns:
            states: Predicted states
        """
        states = [initial_state]
        state = initial_state

        for action in actions:
            state = self.predict_next_state(state, action)
            states.append(state)

        return states


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_world_model():
    """Demonstrate world model."""
    print("=" * 70)
    print("WORLD MODEL DEMONSTRATION")
    print("=" * 70)

    model = WorldModel(state_dim=16, action_dim=4)

    state = np.random.randn(16)
    actions = [np.random.randn(4) for _ in range(5)]

    states = model.simulate(state, actions)
    print(f"Initial state: {state.shape}")
    print(f"Number of states: {len(states)}")
    print(f"Final state: {states[-1].shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_world_model()


################################################################################
# REFERENCES
################################################################################

# [1] Bruce, J., et al. (2024). Genie: Generative Interactive Environments.

################################################################################
