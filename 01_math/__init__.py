"""
################################################################################
MATH FOUNDATIONS FOR AI
################################################################################

This module covers the mathematical foundations required to understand,
implement, and innovate in modern AI systems.

WHAT YOU'LL LEARN:
    - Linear Algebra: Vectors, matrices, tensors, decompositions
    - Probability: Distributions, Bayes theorem, information theory
    - Optimization: Gradient descent, convexity, convergence
    - Calculus: Derivatives, chain rule, backpropagation

WHY MATH MATTERS FOR AI:
    Every AI model is fundamentally a mathematical function.
    Understanding the math means:
    1. You can debug models by reasoning about gradients
    2. You can design new architectures from first principles
    3. You can read and implement research papers
    4. You can optimize training and inference
    5. You can explain models to stakeholders

HOW TO USE THIS MODULE:
    Start with linear_algebra.py — it's the foundation for everything.
    Then probability.py — needed for loss functions and sampling.
    Then optimization.py — needed for training.
    Then calculus.py — needed for backpropagation.

################################################################################
"""

from .linear_algebra import Vector, Matrix, Tensor
from .probability import Distribution, Gaussian, Categorical
from .optimization import GradientDescent, Adam
from .calculus import gradient, jacobian, hessian
