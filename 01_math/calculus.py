"""
################################################################################
CALCULUS — DERIVATIVES AND BACKPROPAGATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Calculus in AI?
    Calculus provides the tools for computing gradients.
    Gradients tell us how to adjust weights to reduce loss.

Key Concepts:
    1. Derivative: rate of change
    2. Gradient: vector of partial derivatives
    3. Chain rule: derivative of composed functions
    4. Backpropagation: efficient gradient computation

Interview Questions:
    1. "What is backpropagation?"
        Efficient computation of gradients using chain rule.

    2. "Why do we need gradients?"
        Gradients tell us how to adjust weights to reduce loss.

################################################################################
"""

import numpy as np
from typing import Callable

################################################################################
# SECTION 1: DERIVATIVES
################################################################################

def numerical_derivative(f: Callable, x: float, h: float = 1e-5) -> float:
    """
    Compute numerical derivative.

    f'(x) ≈ (f(x+h) - f(x-h)) / 2h
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def gradient(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Compute gradient (numerical).

    For each dimension, compute partial derivative.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


################################################################################
# SECTION 2: BACKPROPAGATION
################################################################################

class Backpropagation:
    """
    Backpropagation
    ===============

    Efficient gradient computation using chain rule.

    Forward pass: compute outputs
    Backward pass: compute gradients

    Interview Question:
        "How does backpropagation work?"
        Apply chain rule from output to input.
        Each layer computes local gradient and passes it backward.
    """

    @staticmethod
    def chain_rule(
        df_dg: np.ndarray,
        dg_dx: np.ndarray
    ) -> np.ndarray:
        """
        Chain rule: df/dx = df/dg * dg/dx

        Args:
            df_dg: Gradient of f w.r.t. g
            dg_dx: Gradient of g w.r.t. x

        Returns:
            df/dx: Gradient of f w.r.t. x
        """
        return df_dg * dg_dx


################################################################################
# SECTION 3: TESTING
################################################################################

def demonstrate_calculus():
    """Demonstrate calculus concepts."""
    print("=" * 70)
    print("CALCULUS DEMONSTRATION")
    print("=" * 70)

    # Numerical derivative
    print("\n--- Numerical Derivative ---")
    f = lambda x: x ** 2
    x = 3.0
    deriv = numerical_derivative(f, x)
    print(f"f(x) = x², f'({x}) = {deriv:.4f}")

    # Gradient
    print("\n--- Gradient ---")
    f_vec = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0, 3.0])
    grad = gradient(f_vec, x)
    print(f"f(x) = Σx², ∇f({x}) = {grad}")

    # Chain rule
    print("\n--- Chain Rule ---")
    df_dg = np.array([2.0])
    dg_dx = np.array([3.0])
    df_dx = Backpropagation.chain_rule(df_dg, dg_dx)
    print(f"df/dg = {df_dg}, dg/dx = {dg_dx}")
    print(f"df/dx = {df_dx}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_calculus()
