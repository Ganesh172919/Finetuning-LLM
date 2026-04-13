"""
################################################################################
SIM-TO-REAL TRANSFER — TRAIN IN SIMULATION, DEPLOY IN REALITY
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Sim-to-Real Transfer?
    Training robot policies in simulation and deploying them on real robots.
    The "reality gap" between simulation and real world is bridged by
    domain randomization and adaptation techniques.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


################################################################################
# SECTION 1: DOMAIN RANDOMIZATION
################################################################################

class DomainRandomization:
    """
    Domain Randomization — Randomize simulation parameters.

    By training across diverse simulation conditions, the policy
    learns to generalize to the real world.

    Paper: "Domain Randomization for Transferring Deep Neural Networks
            from Simulation to the Real World" (Tobin et al., 2017)

    Interview Question:
        "How does domain randomization work?"
        Randomize simulation parameters (friction, mass, lighting, etc.)
        during training. The policy learns to be robust to these variations,
        so it generalizes to the real world where parameters are different
        from any specific simulation setting.
    """

    def __init__(self):
        self.param_ranges = {
            'friction': (0.5, 1.5),
            'mass': (0.8, 1.2),
            'damping': (0.9, 1.1),
            'lighting': (0.7, 1.3),
            'texture': (0.0, 1.0),
        }

    def randomize(self) -> Dict[str, float]:
        """
        Sample randomized parameters.

        Returns:
            Dictionary of randomized parameter values
        """
        params = {}
        for name, (lo, hi) in self.param_ranges.items():
            params[name] = np.random.uniform(lo, hi)
        return params

    def create_randomized_env(self) -> Dict:
        """Create a randomized environment configuration."""
        params = self.randomize()
        return {
            'physics': {
                'friction': params['friction'],
                'mass': params['mass'],
                'damping': params['damping'],
            },
            'visual': {
                'lighting': params['lighting'],
                'texture': params['texture'],
            }
        }


################################################################################
# SECTION 2: SYSTEM IDENTIFICATION
################################################################################

class SystemIdentification:
    """
    System Identification — Estimate real-world parameters.

    Collect real-world data and optimize simulation parameters to match.

    Interview Question:
        "What is system identification for robotics?"
        Collect trajectories from the real robot. Then optimize simulation
        parameters (friction, mass, etc.) so that the simulated robot
        behavior matches the real behavior. This reduces the reality gap
        by making the simulation more realistic.
    """

    def __init__(self, n_params: int = 5):
        self.n_params = n_params
        self.params = np.ones(n_params)  # Initial guess

    def simulate(self, params: np.ndarray, n_steps: int = 100) -> np.ndarray:
        """Simulate with given parameters."""
        # Simplified simulation
        trajectory = np.cumsum(np.random.randn(n_steps) * params[0])
        return trajectory

    def real_trajectory(self, n_steps: int = 100) -> np.ndarray:
        """Get real-world trajectory (simulated)."""
        return np.cumsum(np.random.randn(n_steps) * 1.1)

    def optimize(self, real_data: np.ndarray, n_iterations: int = 50) -> np.ndarray:
        """
        Optimize simulation parameters to match real data.

        Args:
            real_data: Real-world trajectory
            n_iterations: Optimization iterations

        Returns:
            Optimized parameters
        """
        lr = 0.01
        for i in range(n_iterations):
            sim_data = self.simulate(self.params, len(real_data))
            error = np.mean((sim_data - real_data) ** 2)
            # Simple gradient step
            grad = 2 * np.mean((sim_data - real_data) * sim_data)
            self.params -= lr * grad
            self.params = np.clip(self.params, 0.5, 2.0)
        return self.params


################################################################################
# SECTION 3: SIM-TO-REAL ADAPTER
################################################################################

class SimToRealAdapter:
    """
    Adapt policy from simulation to real world.

    Interview Question:
        "How do you adapt a sim-trained policy to the real world?"
        (1) Domain randomization: train across diverse sim conditions,
        (2) Fine-tune on small real dataset,
        (3) Feature alignment: match sim/real feature distributions,
        (4) Progressive adaptation: gradually shift from sim to real.
    """

    def __init__(self, state_dim: int = 10):
        self.state_dim = state_dim
        # Feature alignment weights
        self.alignment = np.eye(state_dim)

    def align_features(self, sim_features: np.ndarray) -> np.ndarray:
        """Align simulation features to real-world distribution."""
        return sim_features @ self.alignment

    def fine_tune(self, real_states: np.ndarray, real_actions: np.ndarray,
                  n_steps: int = 100) -> float:
        """
        Fine-tune policy on real-world data.

        Args:
            real_states: Real-world states
            real_actions: Real-world actions
            n_steps: Training steps

        Returns:
            Final loss
        """
        lr = 0.001
        loss = 0
        for step in range(n_steps):
            idx = np.random.randint(len(real_states))
            pred = self.align_features(real_states[idx])
            error = pred - real_actions[idx] if step < len(real_actions) else 0
            loss = np.mean(error ** 2) if isinstance(error, np.ndarray) else 0
        return loss


################################################################################
# SECTION 4: SIMULATOR
################################################################################

class Simulator:
    """
    Simple 2D physics simulator for demonstration.

    Interview Question:
        "Why train in simulation?"
        Simulation provides: (1) unlimited data, (2) safe exploration,
        (3) perfect ground truth, (4) parallel environments. The
        challenge is the "reality gap" — simulation isn't perfect.
        Domain randomization and system identification bridge this gap.
    """

    def __init__(self, dt: float = 0.01):
        self.dt = dt
        self.gravity = 9.81

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Simulate one step.

        Args:
            state: [x, y, vx, vy]
            action: [fx, fy]

        Returns:
            Next state
        """
        x, y, vx, vy = state
        fx, fy = action

        # Simple physics
        ax = fx
        ay = fy - self.gravity

        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt
        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt

        # Floor collision
        if y_new < 0:
            y_new = 0
            vy_new = -vy_new * 0.5  # Bounce with damping

        return np.array([x_new, y_new, vx_new, vy_new])

    def simulate_trajectory(self, initial_state: np.ndarray,
                            actions: List[np.ndarray]) -> np.ndarray:
        """Simulate a full trajectory."""
        states = [initial_state]
        state = initial_state.copy()
        for action in actions:
            state = self.step(state, action)
            states.append(state.copy())
        return np.array(states)


################################################################################
# SECTION 5: DEMONSTRATION
################################################################################

def demonstrate_sim_to_real():
    """Demonstrate sim-to-real transfer."""
    print("=" * 70)
    print("SIM-TO-REAL TRANSFER DEMONSTRATION")
    print("=" * 70)

    # Domain Randomization
    print("\n1. DOMAIN RANDOMIZATION")
    print("-" * 40)
    dr = DomainRandomization()
    for i in range(3):
        env = dr.create_randomized_env()
        print(f"  Env {i}: friction={env['physics']['friction']:.2f}, "
              f"mass={env['physics']['mass']:.2f}")

    # System Identification
    print("\n2. SYSTEM IDENTIFICATION")
    print("-" * 40)
    si = SystemIdentification(n_params=3)
    real_data = si.real_trajectory(100)
    optimized = si.optimize(real_data, n_iterations=20)
    print(f"  Initial params: {np.ones(3)}")
    print(f"  Optimized params: {optimized}")

    # Simulator
    print("\n3. PHYSICS SIMULATOR")
    print("-" * 40)
    sim = Simulator(dt=0.01)
    state = np.array([0.0, 1.0, 1.0, 0.0])  # x, y, vx, vy
    actions = [np.array([0.0, 5.0]) for _ in range(50)]
    trajectory = sim.simulate_trajectory(state, actions)
    print(f"  Initial: {state}")
    print(f"  Final: {trajectory[-1]}")
    print(f"  Trajectory: {trajectory.shape}")

    # Adaptation
    print("\n4. SIM-TO-REAL ADAPTER")
    print("-" * 40)
    adapter = SimToRealAdapter(state_dim=4)
    real_states = np.random.randn(20, 4)
    real_actions = np.random.randn(20, 4)
    loss = adapter.fine_tune(real_states, real_actions, n_steps=50)
    print(f"  Fine-tuning loss: {loss:.4f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_sim_to_real()
