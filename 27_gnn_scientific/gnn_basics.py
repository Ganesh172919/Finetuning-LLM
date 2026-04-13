"""
################################################################################
GNN BASICS — GRAPH NEURAL NETWORK FUNDAMENTALS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Graph Neural Networks?
    Neural networks that learn from graph-structured data by aggregating
    information from neighbors. Each node updates its representation
    based on its neighbors' features.

Core Idea: Message Passing
    1. Each node collects "messages" from its neighbors
    2. Messages are aggregated (sum, mean, max)
    3. Node updates its representation using aggregated messages
    4. Repeat for multiple layers (multi-hop reasoning)

    This is like a neighborhood gossip network:
    - Each person (node) talks to their friends (neighbors)
    - They share information (messages)
    - Each person updates their understanding
    - After several rounds, everyone knows about their extended neighborhood

Key Architectures:
    1. GCN: Average neighbor features + transform
    2. GAT: Weighted average with learned attention
    3. GraphSAGE: Sample neighbors + aggregate
    4. MPNN: General message passing framework

Architecture (Message Passing):
    ┌─────────────────────────────────────────────────────────────────┐
    │                Message Passing Neural Network                   │
    │                                                                  │
    │  Node i ──▶ Collect messages from neighbors j                  │
    │                                                                  │
    │  Message: m_ij = f(h_i, h_j, e_ij)                            │
    │  Aggregate: m_i = AGG({m_ij for j in neighbors(i)})           │
    │  Update: h_i' = g(h_i, m_i)                                    │
    │                                                                  │
    │  After K layers: each node has K-hop neighborhood info          │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "What is message passing in GNNs?"
       Each node collects information from its neighbors, aggregates
       it, and updates its representation. After K layers, each node
       has information from its K-hop neighborhood.

    2. "How do GNNs handle different graph sizes?"
       GNNs are permutation-equivariant — the output doesn't depend
       on node ordering. They process each node independently using
       local neighborhoods, so they handle variable graph sizes.

    3. "What's the over-smoothing problem?"
       After many GNN layers, all node representations converge to
       the same value (over-smoothing). Solutions: residual connections,
       jumping knowledge, or limiting depth.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass
import math

################################################################################
# SECTION 1: GRAPH DATA STRUCTURE
################################################################################

class Graph:
    """
    Graph Data Structure
    =====================

    Stores nodes, edges, and features for GNN processing.

    Components:
    - Node features: h_v for each node v
    - Edge features: e_{uv} for each edge (u, v)
    - Adjacency: which nodes are connected

    Interview Question:
        "How do you represent a graph for neural networks?"
        Common representations: (a) Adjacency matrix A,
        (b) Edge list, (c) Adjacency list. For GNNs, we typically
        use edge lists with feature matrices.
    """

    def __init__(self, num_nodes: int, node_dim: int, edge_dim: int = 0):
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Node features
        self.node_features = np.random.randn(num_nodes, node_dim) * 0.02

        # Adjacency list: neighbors[i] = set of neighbor indices
        self.neighbors: List[Set[int]] = [set() for _ in range(num_nodes)]

        # Edge features (optional)
        self.edge_features: Dict[Tuple[int, int], np.ndarray] = {}

    def add_edge(self, u: int, v: int, feature: Optional[np.ndarray] = None):
        """Add an edge between nodes u and v."""
        self.neighbors[u].add(v)
        self.neighbors[v].add(u)

        if feature is not None:
            self.edge_features[(u, v)] = feature
            self.edge_features[(v, u)] = feature

    def get_neighbors(self, node: int) -> List[int]:
        """Get neighbor indices for a node."""
        return list(self.neighbors[node])

    def get_edge_feature(self, u: int, v: int) -> Optional[np.ndarray]:
        """Get edge feature between u and v."""
        return self.edge_features.get((u, v))


################################################################################
# SECTION 2: GRAPH CONVOLUTIONAL NETWORK (GCN)
################################################################################

class GCNLayer:
    """
    Graph Convolutional Network Layer
    ====================================

    The simplest GNN: aggregate neighbor features with mean.

    Formula:
        h_i' = σ(W × mean({h_i} ∪ {h_j for j in neighbors(i)}))

    This is like "convolution" on graphs — each node's new feature
    is a weighted average of itself and its neighbors.

    Interview Questions:
        1. "How does GCN work?"
           Each node computes a weighted average of its features and
           its neighbors' features, then applies a linear transformation
           and activation. This propagates information across the graph.

        2. "What's the difference between GCN and standard CNN?"
           CNNs operate on grids (fixed neighbors). GCNs operate on
           graphs (variable neighbors). GCNs use the adjacency structure
           to define the "convolution" operation.
    """

    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Weight matrix
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.bias = np.zeros(out_dim)

    def forward(self, node_features: np.ndarray, graph: Graph) -> np.ndarray:
        """
        Forward pass through GCN layer.

        Args:
            node_features: Node feature matrix (num_nodes, in_dim)
            graph: Graph structure

        Returns:
            Updated node features (num_nodes, out_dim)
        """
        num_nodes = graph.num_nodes
        aggregated = np.zeros_like(node_features)

        # Aggregate neighbor features (with self-loop)
        for i in range(num_nodes):
            neighbors = graph.get_neighbors(i)
            all_nodes = [i] + neighbors  # Include self

            # Mean of self and neighbors
            aggregated[i] = np.mean(node_features[all_nodes], axis=0)

        # Linear transformation
        output = aggregated @ self.W + self.bias

        # ReLU activation
        output = np.maximum(0, output)

        return output


class GCN:
    """
    Graph Convolutional Network (Multi-layer)
    ===========================================

    Stacks multiple GCN layers for multi-hop reasoning.

    After K layers, each node has information from its K-hop neighborhood.

    Interview Question:
        "How many GCN layers should I use?"
        Typically 2-3 layers. More layers cause over-smoothing
        (all nodes become similar). Use residual connections
        or jumping knowledge for deeper networks.
    """

    def __init__(self, layer_dims: List[int]):
        """
        Args:
            layer_dims: [input_dim, hidden_dim, ..., output_dim]
        """
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(GCNLayer(layer_dims[i], layer_dims[i+1]))

    def forward(self, graph: Graph) -> np.ndarray:
        """
        Forward pass through multi-layer GCN.

        Args:
            graph: Input graph

        Returns:
            Final node embeddings
        """
        h = graph.node_features

        for layer in self.layers:
            h = layer.forward(h, graph)

        return h


################################################################################
# SECTION 3: GRAPH ATTENTION NETWORK (GAT)
################################################################################

class GATLayer:
    """
    Graph Attention Network Layer
    ================================

    Uses attention to weight neighbor contributions.

    Instead of averaging all neighbors equally (like GCN),
    GAT learns which neighbors are more important.

    Formula:
        e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k exp(e_ik)
        h_i' = σ(Σ_j α_ij × Wh_j)

    Interview Questions:
        1. "How does GAT differ from GCN?"
           GCN averages all neighbors equally. GAT uses attention
           to weight neighbors differently — some neighbors are
           more informative than others.

        2. "What does the attention score mean?"
           It measures how relevant neighbor j is to node i.
           High attention = strong influence on the updated feature.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 1):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        # Weight matrices for each head
        self.W = np.random.randn(in_dim, out_dim) * 0.02
        self.a = np.random.randn(2 * self.head_dim, 1) * 0.02
        self.bias = np.zeros(out_dim)

    def forward(self, node_features: np.ndarray, graph: Graph) -> np.ndarray:
        """
        Forward pass through GAT layer.

        Args:
            node_features: Node features (num_nodes, in_dim)
            graph: Graph structure

        Returns:
            Updated node features (num_nodes, out_dim)
        """
        num_nodes = graph.num_nodes

        # Linear transformation
        h = node_features @ self.W  # (num_nodes, out_dim)

        # Reshape for multi-head
        h_heads = h.reshape(num_nodes, self.num_heads, self.head_dim)

        # Compute attention for each head
        output_heads = []
        for head in range(self.num_heads):
            h_head = h_heads[:, head, :]  # (num_nodes, head_dim)

            # Compute attention scores
            attention = np.zeros((num_nodes, num_nodes))

            for i in range(num_nodes):
                for j in graph.get_neighbors(i):
                    # Concatenate features and compute attention
                    concat = np.concatenate([h_head[i], h_head[j]])
                    score = float(np.tanh(concat @ self.a))
                    attention[i, j] = score

            # Softmax over neighbors
            for i in range(num_nodes):
                neighbors = graph.get_neighbors(i)
                if neighbors:
                    scores = attention[i, neighbors]
                    scores = np.exp(scores - np.max(scores))
                    attention[i, neighbors] = scores / np.sum(scores)

            # Aggregate with attention weights
            h_new = np.zeros_like(h_head)
            for i in range(num_nodes):
                for j in graph.get_neighbors(i):
                    h_new[i] += attention[i, j] * h_head[j]

            output_heads.append(h_new)

        # Concatenate heads
        output = np.concatenate(output_heads, axis=-1)

        return np.maximum(0, output + self.bias)  # ReLU


class GAT:
    """
    Graph Attention Network (Multi-layer)
    =======================================

    Stacks multiple GAT layers.
    """

    def __init__(self, layer_dims: List[int], num_heads: int = 4):
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(GATLayer(layer_dims[i], layer_dims[i+1], num_heads))

    def forward(self, graph: Graph) -> np.ndarray:
        h = graph.node_features
        for layer in self.layers:
            h = layer.forward(h, graph)
        return h


################################################################################
# SECTION 4: GRAPHSAGE
################################################################################

class GraphSAGELayer:
    """
    GraphSAGE Layer
    =================

    Samples and aggregates neighbors, then updates node features.

    Key innovation: SAMPLE neighbors (don't use all of them).
    This enables scaling to large graphs.

    Formula:
        h_neighbors = AGG({h_j for j in sampled_neighbors(i)})
        h_i' = σ(W × CONCAT(h_i, h_neighbors))

    Interview Question:
        "Why sample neighbors in GraphSAGE?"
        For large graphs, a node may have thousands of neighbors.
        Computing over all is expensive. Sampling (e.g., 10-25 neighbors)
        makes it tractable while preserving quality.
    """

    def __init__(self, in_dim: int, out_dim: int, sample_size: int = 10):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sample_size = sample_size

        # Weight for concatenated features
        self.W = np.random.randn(2 * in_dim, out_dim) * 0.02
        self.bias = np.zeros(out_dim)

    def forward(self, node_features: np.ndarray, graph: Graph) -> np.ndarray:
        """Forward pass through GraphSAGE layer."""
        num_nodes = graph.num_nodes
        aggregated = np.zeros_like(node_features)

        # Sample and aggregate neighbors
        for i in range(num_nodes):
            neighbors = graph.get_neighbors(i)
            if len(neighbors) > self.sample_size:
                # Sample subset of neighbors
                sampled = np.random.choice(
                    neighbors, self.sample_size, replace=False
                ).tolist()
            else:
                sampled = neighbors

            if sampled:
                aggregated[i] = np.mean(node_features[sampled], axis=0)

        # Concatenate self and neighbor features
        combined = np.concatenate([node_features, aggregated], axis=-1)

        # Transform
        output = combined @ self.W + self.bias

        return np.maximum(0, output)  # ReLU


class GraphSAGE:
    """
    GraphSAGE (Multi-layer)
    ========================

    Stacks multiple GraphSAGE layers.
    """

    def __init__(self, layer_dims: List[int], sample_size: int = 10):
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(GraphSAGELayer(layer_dims[i], layer_dims[i+1], sample_size))

    def forward(self, graph: Graph) -> np.ndarray:
        h = graph.node_features
        for layer in self.layers:
            h = layer.forward(h, graph)
        return h


################################################################################
# SECTION 5: MESSAGE PASSING (GENERAL FRAMEWORK)
################################################################################

class MessagePassing:
    """
    Message Passing Neural Network (MPNN)
    ========================================

    General framework that encompasses GCN, GAT, GraphSAGE, and more.

    The framework has three functions:
    1. Message: What information to send from j to i
    2. Aggregate: How to combine messages from all neighbors
    3. Update: How to update node state using aggregated messages

    This is the "universal" GNN framework.

    Interview Question:
        "What is the message passing framework?"
        A general GNN framework where: (a) nodes send messages to
        neighbors, (b) messages are aggregated, (c) nodes update
        their state. GCN, GAT, GraphSAGE are all special cases.
    """

    def __init__(self, message_dim: int, hidden_dim: int):
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim

        # Message function
        self.msg_W = np.random.randn(hidden_dim * 2, message_dim) * 0.02

        # Update function
        self.update_W = np.random.randn(hidden_dim + message_dim, hidden_dim) * 0.02

    def message(self, h_i: np.ndarray, h_j: np.ndarray) -> np.ndarray:
        """Compute message from j to i."""
        concat = np.concatenate([h_i, h_j])
        return np.tanh(self.msg_W @ concat)

    def aggregate(self, messages: List[np.ndarray]) -> np.ndarray:
        """Aggregate messages from neighbors."""
        if not messages:
            return np.zeros(self.message_dim)
        return np.mean(messages, axis=0)

    def update(self, h_i: np.ndarray, m_i: np.ndarray) -> np.ndarray:
        """Update node state."""
        concat = np.concatenate([h_i, m_i])
        return np.tanh(self.update_W @ concat)

    def forward(self, node_features: np.ndarray, graph: Graph) -> np.ndarray:
        """Full message passing step."""
        num_nodes = graph.num_nodes
        new_features = np.zeros_like(node_features)

        for i in range(num_nodes):
            # Collect messages
            messages = []
            for j in graph.get_neighbors(i):
                msg = self.message(node_features[i], node_features[j])
                messages.append(msg)

            # Aggregate
            m_i = self.aggregate(messages)

            # Update
            new_features[i] = self.update(node_features[i], m_i)

        return new_features


################################################################################
# SECTION 6: TESTING & DEMONSTRATION
################################################################################

def demonstrate_gnn_basics():
    """Demonstrate GNN basics."""
    print("=" * 70)
    print("GRAPH NEURAL NETWORK BASICS")
    print("=" * 70)

    # Create a simple graph
    num_nodes = 6
    node_dim = 8
    graph = Graph(num_nodes, node_dim)

    # Add edges (cycle + some extra)
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,3), (1,4)]
    for u, v in edges:
        graph.add_edge(u, v)

    print(f"\nGraph: {num_nodes} nodes, {len(edges)} edges")
    print(f"Node feature dim: {node_dim}")

    # GCN
    print("\n--- GCN ---")
    gcn = GCN([node_dim, 16, 8])
    gcn_out = gcn.forward(graph)
    print(f"  Output shape: {gcn_out.shape}")
    print(f"  Sample node 0 features: {gcn_out[0][:4]}")

    # GAT
    print("\n--- GAT ---")
    gat = GAT([node_dim, 16, 8], num_heads=2)
    gat_out = gat.forward(graph)
    print(f"  Output shape: {gat_out.shape}")
    print(f"  Sample node 0 features: {gat_out[0][:4]}")

    # GraphSAGE
    print("\n--- GraphSAGE ---")
    sage = GraphSAGE([node_dim, 16, 8], sample_size=3)
    sage_out = sage.forward(graph)
    print(f"  Output shape: {sage_out.shape}")
    print(f"  Sample node 0 features: {sage_out[0][:4]}")

    # Message Passing
    print("\n--- Message Passing ---")
    mp = MessagePassing(message_dim=8, hidden_dim=node_dim)
    mp_out = mp.forward(graph.node_features, graph)
    print(f"  Output shape: {mp_out.shape}")
    print(f"  Sample node 0 features: {mp_out[0][:4]}")

    # Compare representations
    print("\n--- Comparing Representations ---")
    # All nodes should have different representations
    for name, out in [("GCN", gcn_out), ("GAT", gat_out), ("SAGE", sage_out)]:
        unique = len(set(tuple(np.round(out[i], 3)) for i in range(num_nodes)))
        print(f"  {name}: {unique}/{num_nodes} unique node representations")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: GNNs aggregate neighbor information for each node!")
    print("After K layers, each node knows about its K-hop neighborhood.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_gnn_basics()
