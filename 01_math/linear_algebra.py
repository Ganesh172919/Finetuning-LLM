"""
################################################################################
LINEAR ALGEBRA — THE LANGUAGE OF AI
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Linear Algebra?
    Linear algebra is the branch of mathematics concerning linear equations,
    linear maps, and their representations in vector spaces and matrices.
    It is THE mathematical language of machine learning and AI.

Why do we need it?
    Every neural network operation is a linear algebra operation:
    - Embedding lookup: matrix indexing
    - Linear layers: matrix multiplication
    - Attention: dot products and softmax
    - Convolutions: sliding matrix multiplications
    - Backpropagation: chain rule through matrices

What problem does it solve?
    Linear algebra provides efficient ways to:
    1. Represent data (vectors, matrices, tensors)
    2. Transform data (matrix multiplication)
    3. Find patterns (eigendecomposition, SVD)
    4. Optimize functions (gradients, Hessians)

How does it work?
    Data is represented as multidimensional arrays (tensors).
    Operations on these arrays transform the data.
    The key insight: complex transformations can be decomposed
    into sequences of simple linear operations.

Mathematical Intuition:
    A matrix is a function that transforms space.
    Multiplying a vector by a matrix rotates, scales, and shears it.
    Neural networks learn which transformations to apply.

Real-world Analogy:
    Think of a matrix as a recipe:
    - Each row is an ingredient
    - Each column is a measurement
    - Matrix multiplication combines recipes
    - The result is a new dish (transformed data)

########################################

HISTORICAL EVOLUTION:
    - 2000 BC: Babylonians solve linear equations
    - 1683: Leibniz develops determinants
    - 1850: Sylvester coins "matrix"
    - 1858: Cayley develops matrix algebra
    - 1900s: Hilbert spaces formalized
    - 1960s: Linear algebra in computer science
    - 2010s: GPU-accelerated linear algebra for deep learning
    - 2020s: Flash attention, efficient transformers

KEY PAPERS:
    - Strassen (1969): Fast matrix multiplication
    - Golub & Van Loan (1996): Matrix Computations
    - Vaswani et al. (2017): Attention is All You Need
    - Dao et al. (2022): FlashAttention

################################################################################
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import math

################################################################################
# SECTION 1: VECTORS
################################################################################

########################################
WHAT IS A VECTOR?
########################################

A vector is a list of numbers that represents:
1. A point in space (position)
2. A direction and magnitude (velocity, force)
3. An embedding (word representation in AI)

In AI, vectors represent everything:
- Word embeddings: [0.2, -0.5, 0.8, ...] (768 or 1024 dimensions)
- Image features: [0.1, 0.3, -0.2, ...]
- Audio spectrograms: [0.5, 0.7, 0.2, ...]

########################################


class Vector:
    """
    A mathematical vector with educational implementations of all operations.

    ########################################
    VECTOR OPERATIONS IN AI
    ########################################

    1. Dot Product → Attention scores, similarity
    2. Norm → Normalization, regularization
    3. Addition → Residual connections
    4. Scalar Multiplication → Scaling
    5. Cosine Similarity → Semantic similarity
    6. Projection → Dimensionality reduction

    ########################################
    """

    def __init__(self, data: Union[List[float], np.ndarray]):
        """
        Initialize a vector from a list of numbers or numpy array.

        Args:
            data: The vector components

        Example:
            v = Vector([1.0, 2.0, 3.0])
            print(v)  # [1.0, 2.0, 3.0]
        """
        self.data = np.array(data, dtype=np.float64)
        self.shape = self.data.shape

    def __repr__(self) -> str:
        return f"Vector({self.data.tolist()})"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]

    def __add__(self, other: 'Vector') -> 'Vector':
        """
        Vector Addition
        ================

        Definition: Add corresponding components.
        Formula: (a + b)[i] = a[i] + b[i]

        Why it matters in AI:
        - Residual connections: output = x + f(x)
        - Combining embeddings: sentence = word1 + word2 + word3
        - Gradient accumulation: gradients from multiple batches

        Example:
            a = [1, 2, 3], b = [4, 5, 6]
            a + b = [5, 7, 9]

        Visual:
            a: →→→
            b:   →→→
            a+b: →→→→→→

        """
        if len(self) != len(other):
            raise ValueError(f"Vector dimensions must match: {len(self)} vs {len(other)}")
        return Vector(self.data + other.data)

    def __sub__(self, other: 'Vector') -> 'Vector':
        """
        Vector Subtraction
        ===================

        Definition: Subtract corresponding components.
        Formula: (a - b)[i] = a[i] - b[i]

        Why it matters in AI:
        - Computing differences between embeddings
        - Gradient direction: θ_new = θ_old - lr * gradient

        Example:
            a = [5, 7, 9], b = [1, 2, 3]
            a - b = [4, 5, 6]
        """
        if len(self) != len(other):
            raise ValueError(f"Vector dimensions must match: {len(self)} vs {len(other)}")
        return Vector(self.data - other.data)

    def __mul__(self, scalar: float) -> 'Vector':
        """
        Scalar Multiplication
        ======================

        Definition: Multiply every component by a scalar.
        Formula: (c * v)[i] = c * v[i]

        Why it matters in AI:
        - Learning rate scaling: gradient * lr
        - Temperature scaling: logits / temperature
        - Normalization: v / ||v||

        Example:
            v = [1, 2, 3], c = 2
            c * v = [2, 4, 6]
        """
        return Vector(self.data * scalar)

    def dot(self, other: 'Vector') -> float:
        """
        Dot Product
        ===========

        Definition: Sum of element-wise products.
        Formula: a · b = Σ(a[i] * b[i])

        Why it's the most important operation in AI:
        1. Attention: Q·K gives attention scores
        2. Similarity: cosine similarity uses dot product
        3. Linear layers: y = W·x + b
        4. Classification: logits = W·h

        Mathematical Properties:
        - Commutative: a·b = b·a
        - Distributive: a·(b+c) = a·b + a·c
        - Geometric meaning: |a||b|cos(θ)

        When a·b = 0: vectors are perpendicular (orthogonal)
        When a·b > 0: vectors point in similar directions
        When a·b < 0: vectors point in opposite directions

        Example:
            Q = [1, 0, 1], K = [1, 1, 0]
            Q·K = 1*1 + 0*1 + 1*0 = 1

        Real-world Example:
            In transformer attention:
            score = Q · K / sqrt(d_k)
            This tells us how much each token should attend to others.

        """
        if len(self) != len(other):
            raise ValueError(f"Vector dimensions must match: {len(self)} vs {len(other)}")
        return float(np.dot(self.data, other.data))

    def norm(self, p: int = 2) -> float:
        """
        Vector Norm (Magnitude)
        ========================

        Definition: The "length" of a vector.
        Formula: ||v||_p = (Σ|v[i]|^p)^(1/p)

        Common norms:
        - L1 norm (p=1): Σ|v[i]| — used for sparsity (Lasso regression)
        - L2 norm (p=2): sqrt(Σv[i]²) — most common (Ridge regression)
        - L-infinity: max|v[i]| — used for robustness

        Why it matters in AI:
        - Gradient clipping: prevent exploding gradients
        - Normalization: LayerNorm, RMSNorm
        - Regularization: weight decay penalizes large norms
        - Similarity: cosine similarity = (a·b)/(||a||·||b||)

        Example:
            v = [3, 4]
            L2 norm = sqrt(9 + 16) = sqrt(25) = 5

        """
        if p == 1:
            return float(np.sum(np.abs(self.data)))
        elif p == 2:
            return float(np.sqrt(np.sum(self.data ** 2)))
        else:
            return float(np.sum(np.abs(self.data) ** p) ** (1.0 / p))

    def normalize(self, p: int = 2) -> 'Vector':
        """
        Normalize Vector (Unit Vector)
        ===============================

        Definition: Scale vector to have unit length.
        Formula: v_norm = v / ||v||

        Why it matters in AI:
        - Cosine similarity requires normalized vectors
        - Stable training: prevents activation explosion
        - Weight normalization technique

        Example:
            v = [3, 4], ||v|| = 5
            v_norm = [0.6, 0.8]
        """
        n = self.norm(p)
        if n == 0:
            return Vector(np.zeros_like(self.data))
        return Vector(self.data / n)

    def cosine_similarity(self, other: 'Vector') -> float:
        """
        Cosine Similarity
        ==================

        Definition: Measures the angle between two vectors.
        Formula: cos(θ) = (a·b) / (||a|| * ||b||)

        Range: [-1, 1]
        - 1: vectors point in same direction (identical meaning)
        - 0: vectors are perpendicular (unrelated)
        - -1: vectors point in opposite directions (opposite meaning)

        Why it matters in AI:
        - Semantic search: find similar documents
        - Recommendation: find similar items
        - Contrastive learning: CLIP, SimCLR
        - Attention: scaled dot-product attention

        Real-world Example:
            Word embeddings for "king" and "queen" have high cosine similarity.
            Word embeddings for "king" and "car" have low cosine similarity.

        Interview Question:
            "Why cosine similarity instead of Euclidean distance?"
            Answer: Cosine similarity is invariant to vector magnitude.
            Two documents with same words but different lengths have
            high cosine similarity but large Euclidean distance.

        """
        dot_product = self.dot(other)
        norm_product = self.norm() * other.norm()
        if norm_product == 0:
            return 0.0
        return dot_product / norm_product

    def outer_product(self, other: 'Vector') -> 'Matrix':
        """
        Outer Product
        ==============

        Definition: Creates a matrix from two vectors.
        Formula: (a ⊗ b)[i][j] = a[i] * b[j]

        Why it matters in AI:
        - Attention: outer product of Q and K creates attention matrix
        - Low-rank approximation: LoRA uses outer products
        - Feature interactions: capturing pairwise relationships

        Example:
            a = [1, 2], b = [3, 4]
            a ⊗ b = [[3, 4],
                      [6, 8]]
        """
        result = np.outer(self.data, other.data)
        return Matrix(result)


################################################################################
# SECTION 2: MATRICES
################################################################################

########################################
WHAT IS A MATRIX?
########################################

A matrix is a 2D array of numbers that represents:
1. A linear transformation (rotation, scaling, shearing)
2. A system of linear equations
3. A weight layer in a neural network
4. An attention pattern
5. A covariance structure

In AI, matrices are EVERYWHERE:
- Linear layer weights: [output_dim × input_dim]
- Attention weights: [seq_len × seq_len]
- Embedding matrices: [vocab_size × embed_dim]
- Convolution kernels: [out_channels × in_channels × kH × kW]

########################################


class Matrix:
    """
    A mathematical matrix with educational implementations.

    ########################################
    MATRIX OPERATIONS IN AI
    ########################################

    1. Matrix Multiplication → Linear layers, attention
    2. Transpose → Backpropagation, attention
    3. Inverse → Solving linear systems
    4. Determinant → Checking invertibility
    5. Eigenvalues → PCA, understanding transformations
    6. SVD → Low-rank approximation, LoRA

    ########################################
    """

    def __init__(self, data: Union[List[List[float]], np.ndarray]):
        """
        Initialize a matrix from 2D list or numpy array.

        Args:
            data: 2D array of numbers

        Example:
            M = Matrix([[1, 2], [3, 4]])
            print(M.shape)  # (2, 2)
        """
        self.data = np.array(data, dtype=np.float64)
        self.shape = self.data.shape  # (rows, cols)

    def __repr__(self) -> str:
        return f"Matrix(shape={self.shape})"

    def __getitem__(self, idx):
        return self.data[idx]

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        """
        Matrix Multiplication
        ======================

        Definition: Multiply two matrices.
        Formula: (AB)[i][j] = Σ_k A[i][k] * B[k][j]

        WHY THIS IS THE MOST IMPORTANT OPERATION IN AI:
        ================================================
        Every neural network layer is essentially: y = Wx + b

        When you call:
            output = self.linear(input)

        It's doing: output = weight_matrix @ input + bias

        The entire training process learns what values should
        be in these weight matrices.

        Computational Complexity: O(n³) for n×n matrices
        This is why GPUs exist — they parallelize this.

        Example:
            A = [[1, 2],    B = [[5, 6],
                 [3, 4]]         [7, 8]]

            AB = [[1*5+2*7, 1*6+2*8],   = [[19, 22],
                  [3*5+4*7, 3*6+4*8]]       [43, 50]]

        Visual Intuition:
            Row i of A "selects" which linear combination
            Column j of B "selects" which features to combine.

        Real-world Example:
            In a transformer:
            - Q = X @ W_Q  (query projection)
            - K = X @ W_K  (key projection)
            - V = X @ W_V  (value projection)
            - Attention = softmax(Q @ K^T / sqrt(d)) @ V

        Interview Question:
            "Why is matrix multiplication O(n³)?"
            Answer: For each of n² output elements, we compute
            a dot product of length n, giving n² × n = n³ operations.

        """
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Matrix dimensions incompatible for multiplication: "
                f"{self.shape} @ {other.shape}"
            )
        result = self.data @ other.data
        return Matrix(result)

    def transpose(self) -> 'Matrix':
        """
        Matrix Transpose
        =================

        Definition: Swap rows and columns.
        Formula: (A^T)[i][j] = A[j][i]

        Why it matters in AI:
        - Backpropagation: gradient of y=Wx is dL/dW = (dL/dy) @ x^T
        - Attention: Q @ K^T computes query-key similarity
        - Self-attention: each token attends to all others

        Example:
            A = [[1, 2, 3],
                 [4, 5, 6]]

            A^T = [[1, 4],
                   [2, 5],
                   [3, 6]]

        Properties:
        - (A^T)^T = A
        - (AB)^T = B^T A^T
        - (A + B)^T = A^T + B^T
        """
        return Matrix(self.data.T)

    def inverse(self) -> 'Matrix':
        """
        Matrix Inverse
        ===============

        Definition: A matrix that "undoes" the transformation.
        Formula: A * A^(-1) = I (identity matrix)

        Why it matters in AI:
        - Solving linear systems: Ax = b → x = A^(-1)b
        - Understanding transformations: what input gives desired output?
        - Covariance matrices: precision = covariance^(-1)

        Note: Not all matrices are invertible!
        A matrix is invertible iff its determinant ≠ 0.
        In practice, we rarely compute explicit inverses —
        we use more efficient methods (LU decomposition, etc.).

        Example:
            A = [[2, 1],
                 [5, 3]]
            A^(-1) = [[3, -1],
                       [-5, 2]]

        """
        det = np.linalg.det(self.data)
        if abs(det) < 1e-10:
            raise ValueError("Matrix is singular (not invertible)")
        return Matrix(np.linalg.inv(self.data))

    def determinant(self) -> float:
        """
        Determinant
        ===========

        Definition: A scalar that measures how a matrix scales space.
        Formula for 2x2: det([[a,b],[c,d]]) = ad - bc

        Interpretation:
        - |det| > 1: matrix expands space
        - |det| < 1: matrix compresses space
        - |det| = 0: matrix collapses space (not invertible)
        - det < 0: matrix flips orientation

        Why it matters in AI:
        - Check if transformation is invertible
        - Change of variables in probability
        - Jacobian determinant in normalizing flows

        Example:
            A = [[2, 1],
                 [5, 3]]
            det = 2*3 - 1*5 = 6 - 5 = 1
            (This preserves volume!)
        """
        return float(np.linalg.det(self.data))

    def eigenvalues(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigendecomposition
        ===================

        Definition: Find special vectors that only get scaled by the matrix.
        Formula: A @ v = λ * v

        Where:
        - v is an eigenvector (direction that doesn't change)
        - λ is an eigenvalue (how much it gets scaled)

        Why it matters in AI:
        - PCA: eigenvectors of covariance matrix = principal components
        - Understanding training dynamics: Hessian eigenvalues
        - Spectral normalization: controls Lipschitz constant
        - Stability: eigenvalues of recurrent weight matrix

        Example:
            A = [[2, 1],
                 [1, 2]]
            Eigenvalues: λ₁ = 3, λ₂ = 1
            Eigenvectors: v₁ = [1, 1], v₂ = [1, -1]

            A @ [1, 1] = [3, 3] = 3 * [1, 1]  ✓
            A @ [1, -1] = [1, -1] = 1 * [1, -1]  ✓

        Interview Question:
            "What do eigenvalues tell us about a neural network?"
            Answer: Large eigenvalues in the Hessian indicate
            directions of high curvature — the loss surface is steep.
            This relates to training stability and learning rate selection.
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.data)
        return eigenvalues, eigenvectors

    def svd(self) -> Tuple['Matrix', np.ndarray, 'Matrix']:
        """
        Singular Value Decomposition (SVD)
        =====================================

        Definition: Decompose any matrix into three simpler matrices.
        Formula: A = U @ Σ @ V^T

        Where:
        - U: left singular vectors (orthogonal)
        - Σ: singular values (diagonal, non-negative)
        - V^T: right singular vectors (orthogonal)

        Why it's CRITICAL for modern AI:
        1. LoRA: Low-Rank Adaptation uses SVD-like decomposition
        2. Model compression: keep only top-k singular values
        3. PCA: principal components from SVD of data matrix
        4. Pseudoinverse: A^+ = V @ Σ^(-1) @ U^T
        5. Understanding attention patterns

        Example:
            A = [[1, 2],
                 [3, 4],
                 [5, 6]]

            A = U @ diag(s) @ V^T

            We can approximate A by keeping only top-k singular values.
            This is how LoRA works! Instead of training full A,
            we train A ≈ U_k @ diag(s_k) @ V_k^T with small k.

        Real-world Example:
            LoRA for LLM fine-tuning:
            - Original weight: W ∈ R^{4096 × 4096} (16M params)
            - LoRA: W ≈ W₀ + A @ B where A ∈ R^{4096 × 16}, B ∈ R^{16 × 4096}
            - Trainable params: 131K (120x reduction!)
        """
        U, s, Vt = np.linalg.svd(self.data, full_matrices=False)
        return Matrix(U), s, Matrix(Vt)

    def frobenius_norm(self) -> float:
        """
        Frobenius Norm
        ==============

        Definition: The "magnitude" of a matrix.
        Formula: ||A||_F = sqrt(Σ A[i][j]²)

        Why it matters in AI:
        - Regularization: penalize large weights
        - Measuring approximation quality
        - Gradient clipping
        """
        return float(np.sqrt(np.sum(self.data ** 2)))


################################################################################
# SECTION 3: TENSORS
################################################################################

########################################
WHAT IS A TENSOR?
########################################

A tensor is a generalization of vectors and matrices to N dimensions:
- 0D tensor: scalar (a single number)
- 1D tensor: vector (a list of numbers)
- 2D tensor: matrix (a grid of numbers)
- 3D tensor: e.g., a color image [channels × height × width]
- 4D tensor: e.g., a batch of images [batch × channels × height × width]
- 5D tensor: e.g., a batch of videos [batch × channels × frames × height × width]

In PyTorch/TensorFlow, EVERYTHING is a tensor.

########################################


class Tensor:
    """
    A multi-dimensional array with educational implementations.

    This is a simplified version — real frameworks (PyTorch, JAX)
    use highly optimized C++/CUDA implementations.

    ########################################
    TENSORS IN MODERN AI
    ########################################

    Model weights: [out_features × in_features]
    Batch of tokens: [batch_size × seq_len]
    Image batch: [batch_size × channels × height × width]
    Video batch: [batch_size × channels × frames × height × width]
    Attention scores: [batch_size × heads × seq_len × seq_len]

    ########################################
    """

    def __init__(self, data: Union[List, np.ndarray]):
        """
        Initialize a tensor from nested lists or numpy array.

        Args:
            data: N-dimensional array of numbers

        Example:
            t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
            print(t.shape)  # (2, 2, 2)
        """
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.ndim = len(self.shape)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype=float32)"

    def reshape(self, *new_shape) -> 'Tensor':
        """
        Reshape Tensor
        ===============

        Definition: Change the shape without changing the data.

        Why it matters in AI:
        - Attention: reshape for multi-head attention
        - Convolutions: reshape for different operations
        - Flattening: reshape for fully connected layers

        Example:
            t = Tensor([1, 2, 3, 4, 5, 6])  # shape: (6,)
            t.reshape(2, 3)  # shape: (2, 3)
            # [[1, 2, 3],
            #  [4, 5, 6]]
        """
        return Tensor(self.data.reshape(new_shape))

    def transpose(self, *axes) -> 'Tensor':
        """
        Transpose Tensor
        =================

        Definition: Reorder the dimensions of a tensor.

        Why it matters in AI:
        - Attention: transpose for multi-head computation
        - Image processing: channels-first vs channels-last
        - Batch operations: rearrange dimensions

        Example:
            t = Tensor(...)  # shape: (batch, seq, embed)
            t.transpose(0, 2, 1)  # shape: (batch, embed, seq)
        """
        return Tensor(np.transpose(self.data, axes))

    def mean(self, axis: Optional[int] = None) -> Union['Tensor', float]:
        """
        Mean
        ====

        Definition: Average of all elements (or along an axis).

        Why it matters in AI:
        - LayerNorm: mean over features
        - Batch statistics: mean over batch dimension
        - Loss averaging: mean over batch
        """
        result = np.mean(self.data, axis=axis)
        if isinstance(result, np.ndarray):
            return Tensor(result)
        return float(result)

    def std(self, axis: Optional[int] = None) -> Union['Tensor', float]:
        """
        Standard Deviation
        ===================

        Definition: Measure of spread around the mean.

        Why it matters in AI:
        - LayerNorm: normalize by std
        - Initialization: Xavier/He initialization uses std
        - Monitoring: track activation statistics
        """
        result = np.std(self.data, axis=axis)
        if isinstance(result, np.ndarray):
            return Tensor(result)
        return float(result)

    def softmax(self, axis: int = -1) -> 'Tensor':
        """
        Softmax
        =======

        Definition: Convert logits to probabilities.
        Formula: softmax(x_i) = exp(x_i) / Σ exp(x_j)

        Why it's CRITICAL for AI:
        - Output layer: convert logits to class probabilities
        - Attention: compute attention weights
        - Temperature sampling: control randomness

        Properties:
        - Output sums to 1 (valid probability distribution)
        - Preserves order (larger input → larger output)
        - Differentiable (can backpropagate through it)

        Numerical Stability:
        Subtract max before exp to prevent overflow:
        softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))

        Example:
            logits = [2.0, 1.0, 0.1]
            softmax = [0.659, 0.242, 0.099]
            Sum = 1.0

        Interview Question:
            "Why subtract the max in softmax?"
            Answer: Without it, exp(large_number) overflows.
            Subtracting max ensures all exponents are ≤ 0,
            so exp values are in [0, 1].
        """
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp_data = np.exp(shifted)
        result = exp_data / np.sum(exp_data, axis=axis, keepdims=True)
        return Tensor(result)


################################################################################
# SECTION 4: COMMON AI OPERATIONS
################################################################################

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    temperature: Optional[float] = None
) -> Tensor:
    """
    Scaled Dot-Product Attention
    =============================

    Definition: The core attention mechanism from "Attention Is All You Need".

    Formula:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Where:
        Q = Query matrix [batch × seq_len × d_k]
        K = Key matrix [batch × seq_len × d_k]
        V = Value matrix [batch × seq_len × d_v]
        d_k = dimension of keys (for scaling)

    WHY THIS IS THE MOST IMPORTANT EQUATION IN MODERN AI:
    =====================================================

    This single equation powers:
    - GPT, Claude, Gemini, LLaMA, Mistral (language models)
    - CLIP, SigLIP (multimodal models)
    - Stable Diffusion, Flux (image generation)
    - Whisper, Bark (speech models)
    - And virtually every SOTA model since 2017

    Step-by-step:
    1. Compute attention scores: scores = Q @ K^T
       (How much should each token attend to every other token?)

    2. Scale by sqrt(d_k): scaled = scores / sqrt(d_k)
       (Prevent dot products from getting too large)

    3. Apply mask (optional): masked = scaled + mask
       (Prevent attending to future tokens in decoder)

    4. Softmax: weights = softmax(masked)
       (Convert to probabilities that sum to 1)

    5. Weighted sum: output = weights @ V
       (Combine values according to attention weights)

    Args:
        Q: Query tensor [batch × seq_len × d_k]
        K: Key tensor [batch × seq_len × d_k]
        V: Value tensor [batch × seq_len × d_v]
        mask: Optional mask tensor [batch × seq_len × seq_len]
              Use -inf for positions to mask (e.g., future tokens)
        temperature: Optional scaling factor (default: sqrt(d_k))

    Returns:
        output: Attention output [batch × seq_len × d_v]
        weights: Attention weights [batch × seq_len × seq_len]

    Example:
        Q = [[1, 0], [0, 1]]  # Two queries
        K = [[1, 0], [0, 1]]  # Two keys
        V = [[1, 2], [3, 4]]  # Two values

        scores = Q @ K^T = [[1, 0], [0, 1]]
        weights = softmax(scores) = [[0.73, 0.27], [0.27, 0.73]]
        output = weights @ V = [[1.54, 2.54], [2.46, 3.46]]

    Interview Questions:
        1. "Why scale by sqrt(d_k)?"
           Without scaling, dot products grow with dimension,
           pushing softmax into regions with tiny gradients.
           Scaling keeps variance ≈ 1 regardless of d_k.

        2. "What's the computational complexity?"
           O(n² × d) where n = sequence length, d = dimension.
           This is why long context is expensive!

        3. "Why Q, K, V instead of just one matrix?"
           Different projections serve different roles:
           - Q: "what am I looking for?"
           - K: "what do I contain?"
           - V: "what information do I provide?"
    """
    d_k = Q.shape[-1]

    if temperature is None:
        temperature = math.sqrt(d_k)

    # Step 1: Compute attention scores
    # Q @ K^T tells us how similar each query is to each key
    scores = Q.data @ K.data.transpose(0, 2, 1)  # [batch × seq × seq]

    # Step 2: Scale to prevent large dot products
    scores = scores / temperature

    # Step 3: Apply mask (for causal attention)
    if mask is not None:
        scores = scores + mask.data

    # Step 4: Convert to probabilities
    weights = Tensor(scores).softmax(axis=-1)

    # Step 5: Weighted sum of values
    output = weights.data @ V.data  # [batch × seq × d_v]

    return Tensor(output)


################################################################################
# SECTION 5: TESTING & EXAMPLES
################################################################################

def demonstrate_linear_algebra():
    """
    Demonstrate key linear algebra concepts with examples.
    """
    print("=" * 70)
    print("LINEAR ALGEBRA DEMONSTRATION")
    print("=" * 70)

    # Vector operations
    print("\n--- Vector Operations ---")
    v1 = Vector([1.0, 2.0, 3.0])
    v2 = Vector([4.0, 5.0, 6.0])

    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"v1 · v2 = {v1.dot(v2)}")
    print(f"||v1|| = {v1.norm():.4f}")
    print(f"cosine(v1, v2) = {v1.cosine_similarity(v2):.4f}")

    # Matrix operations
    print("\n--- Matrix Operations ---")
    A = Matrix([[1.0, 2.0], [3.0, 4.0]])
    B = Matrix([[5.0, 6.0], [7.0, 8.0]])

    print(f"A = {A.data.tolist()}")
    print(f"B = {B.data.tolist()}")
    print(f"A @ B = {(A @ B).data.tolist()}")
    print(f"A^T = {A.transpose().data.tolist()}")
    print(f"det(A) = {A.determinant()}")

    # Attention demonstration
    print("\n--- Attention Demonstration ---")
    batch_size, seq_len, d_model = 1, 4, 8

    Q = Tensor(np.random.randn(batch_size, seq_len, d_model))
    K = Tensor(np.random.randn(batch_size, seq_len, d_model))
    V = Tensor(np.random.randn(batch_size, seq_len, d_model))

    output = scaled_dot_product_attention(Q, K, V)
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    print(f"Output shape: {output.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_linear_algebra()


################################################################################
# REFERENCES
################################################################################

# [1] Strassen, V. (1969). Gaussian Elimination is not Optimal.
# [2] Golub, G. H., & Van Loan, C. F. (1996). Matrix Computations.
# [3] Vaswani, A., et al. (2017). Attention Is All You Need.
# [4] Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact
#     Attention with IO-Awareness.
# [5] Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
# [6] Goodfellow, I., et al. (2016). Deep Learning (Chapter 2: Linear Algebra).

################################################################################
