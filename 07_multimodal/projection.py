"""
################################################################################
PROJECTION LAYERS — MAPPING BETWEEN MODALITIES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Projection Layers?
    Linear layers that map features from one space to another.

In multimodal models:
    Vision features → Projection → Language space
    Audio features → Projection → Language space

Interview Questions:
    Q: "Why do we need projection layers?"
    A: Different modalities have different feature spaces.
       Projections map them to a shared space.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: PROJECTION LAYER
################################################################################

class ProjectionLayer:
    """
    Projection Layer
    ================

    Maps features between different spaces.
    """

    def __init__(self, input_dim: int, output_dim: int):
        self.weight = np.random.randn(input_dim, output_dim) * 0.02
        self.bias = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project features."""
        return x @ self.weight + self.bias


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_projection():
    """Demonstrate projection layer."""
    print("=" * 70)
    print("PROJECTION LAYER DEMONSTRATION")
    print("=" * 70)

    proj = ProjectionLayer(128, 64)
    x = np.random.randn(4, 128)
    y = proj.forward(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_projection()
