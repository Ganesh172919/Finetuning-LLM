"""
################################################################################
TESTING UTILITIES — VERIFYING MODEL CORRECTNESS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Testing Utilities?
    Tools for verifying model implementations.

Key Tests:
    - Gradient checking
    - Shape checking
    - Numerical stability

Interview Questions:
    Q: "How do you verify a model implementation?"
    A: Gradient checking, shape tests, numerical tests.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: TESTING UTILITIES
################################################################################

def check_gradient(func, x: np.ndarray, grad: np.ndarray, eps: float = 1e-5) -> float:
    """
    Check gradient using numerical differentiation.

    Args:
        func: Function to differentiate
        x: Input
        grad: Analytical gradient
        eps: Small number for numerical differentiation

    Returns:
        relative_error: Relative error between analytical and numerical
    """
    numerical_grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy().flat
        x_plus[i] += eps
        x_plus = x_plus.reshape(x.shape)

        x_minus = x.copy().flat
        x_minus[i] -= eps
        x_minus = x_minus.reshape(x.shape)

        numerical_grad.flat[i] = (func(x_plus) - func(x_minus)) / (2 * eps)

    relative_error = np.linalg.norm(grad - numerical_grad) / (np.linalg.norm(grad) + np.linalg.norm(numerical_grad))
    return relative_error


def check_shape(actual: tuple, expected: tuple, name: str = ""):
    """Check tensor shape."""
    assert actual == expected, f"{name}: expected {expected}, got {actual}"


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_testing():
    """Demonstrate testing utilities."""
    print("=" * 70)
    print("TESTING UTILITIES DEMONSTRATION")
    print("=" * 70)

    # Gradient check
    print("\n--- Gradient Check ---")
    func = lambda x: np.sum(x ** 2)
    grad_func = lambda x: 2 * x
    x = np.random.randn(5)
    grad = grad_func(x)
    error = check_gradient(func, x, grad)
    print(f"Gradient error: {error:.8f}")

    # Shape check
    print("\n--- Shape Check ---")
    try:
        check_shape((4, 64), (4, 32), "test")
    except AssertionError as e:
        print(f"Caught: {e}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_testing()
