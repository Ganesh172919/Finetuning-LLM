"""
################################################################################
AST PROCESSING — ABSTRACT SYNTAX TREE FOR CODE UNDERSTANDING
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is AST Processing?
    Using Abstract Syntax Trees to understand code structure. ASTs
    represent code as a tree of nodes (functions, if/else, loops),
    enabling structural analysis, pattern matching, and transformation.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import math


################################################################################
# SECTION 1: AST NODE
################################################################################

class ASTNode:
    """
    A node in the Abstract Syntax Tree.

    Attributes:
        node_type: Type of node (FunctionDef, If, For, BinOp, etc.)
        value: Node value (variable name, operator, literal)
        children: Child nodes
        depth: Depth in tree
        line_number: Source line number
    """

    def __init__(self, node_type: str, value: str = "",
                 children: List['ASTNode'] = None, line_number: int = 0):
        self.node_type = node_type
        self.value = value
        self.children = children or []
        self.depth = 0
        self.line_number = line_number

    def add_child(self, child: 'ASTNode'):
        """Add a child node."""
        child.depth = self.depth + 1
        self.children.append(child)

    def preorder(self) -> List['ASTNode']:
        """Pre-order traversal."""
        result = [self]
        for child in self.children:
            result.extend(child.preorder())
        return result

    def postorder(self) -> List['ASTNode']:
        """Post-order traversal."""
        result = []
        for child in self.children:
            result.extend(child.postorder())
        result.append(self)
        return result

    def __repr__(self) -> str:
        return f"ASTNode({self.node_type}, '{self.value}', children={len(self.children)})"


################################################################################
# SECTION 2: AST ENCODER
################################################################################

class ASTEncoder:
    """
    Encode AST for neural processing.

    Interview Question:
        "How do you encode an AST for a neural network?"
        (1) Traverse tree (pre/post/level order),
        (2) Embed node types and values,
        (3) Use tree-structured attention or GNN,
        (4) Pool to get fixed-size representation.
    """

    def __init__(self, d_model: int = 128):
        self.d_model = d_model

    def encode_node(self, node: ASTNode) -> np.ndarray:
        """
        Encode a single AST node.

        Args:
            node: AST node

        Returns:
            Node embedding (d_model,)
        """
        # Simulate: hash-based encoding
        hash_val = hash(node.node_type + node.value)
        np.random.seed(abs(hash_val) % 2**31)
        return np.random.randn(self.d_model)

    def encode_tree(self, root: ASTNode) -> np.ndarray:
        """
        Encode entire AST into a fixed-size vector.

        Args:
            root: Root AST node

        Returns:
            Tree embedding (d_model,)
        """
        nodes = root.preorder()
        embeddings = np.array([self.encode_node(n) for n in nodes])
        return embeddings.mean(axis=0)


################################################################################
# SECTION 3: CODE PATTERN MATCHER
################################################################################

class CodePatternMatcher:
    """
    Match patterns in code ASTs.

    Interview Question:
        "How do you find code patterns using ASTs?"
        Define patterns as AST templates. Traverse the code AST and
        check if any subtree matches a pattern template. Useful for
        finding anti-patterns, refactoring opportunities, and security
        vulnerabilities.
    """

    def find_pattern(self, root: ASTNode, pattern_type: str) -> List[ASTNode]:
        """
        Find all nodes matching a pattern type.

        Args:
            root: Root of AST
            pattern_type: Node type to find

        Returns:
            List of matching nodes
        """
        return [n for n in root.preorder() if n.node_type == pattern_type]

    def count_nested_depth(self, root: ASTNode, node_type: str) -> int:
        """
        Count maximum nesting depth of a node type.

        Args:
            root: Root of AST
            node_type: Node type to measure

        Returns:
            Maximum nesting depth
        """
        max_depth = 0
        for node in root.preorder():
            if node.node_type == node_type:
                max_depth = max(max_depth, node.depth)
        return max_depth


################################################################################
# SECTION 4: CODE EXTRACTOR
################################################################################

class CodeExtractor:
    """
    Extract information from code ASTs.

    Interview Question:
        "What information can you extract from an AST?"
        (1) Function signatures (name, params, return type),
        (2) Dependencies (what functions does X call?),
        (3) Complexity metrics (cyclomatic, cognitive),
        (4) Variable scope and usage patterns.
    """

    def extract_functions(self, root: ASTNode) -> List[Dict]:
        """
        Extract function definitions.

        Args:
            root: Root AST node

        Returns:
            List of function info dictionaries
        """
        functions = []
        for node in root.preorder():
            if node.node_type == 'FunctionDef':
                func_info = {
                    'name': node.value,
                    'line': node.line_number,
                    'n_children': len(node.children),
                }
                functions.append(func_info)
        return functions

    def compute_complexity(self, root: ASTNode) -> int:
        """
        Compute cyclomatic complexity.

        Complexity = 1 + number of branches (if, for, while, and, or)

        Args:
            root: Root AST node

        Returns:
            Cyclomatic complexity
        """
        branch_types = {'If', 'For', 'While', 'And', 'Or', 'Except'}
        n_branches = sum(1 for n in root.preorder() if n.node_type in branch_types)
        return 1 + n_branches


################################################################################
# SECTION 5: DEMONSTRATION
################################################################################

def demonstrate_ast_processing():
    """Demonstrate AST processing."""
    print("=" * 70)
    print("AST PROCESSING DEMONSTRATION")
    print("=" * 70)

    # Build sample AST
    root = ASTNode("Module")
    func = ASTNode("FunctionDef", "fibonacci")
    root.add_child(func)
    if_node = ASTNode("If")
    func.add_child(if_node)
    if_node.add_child(ASTNode("Compare", "<="))
    if_node.add_child(ASTNode("Return", "n"))
    else_node = ASTNode("Else")
    func.add_child(else_node)
    else_node.add_child(ASTNode("Return", "fib(n-1)+fib(n-2)"))

    # AST Node
    print("\n1. AST NODE")
    print("-" * 40)
    print(f"  Root: {root}")
    print(f"  Preorder: {[n.node_type for n in root.preorder()]}")
    print(f"  Postorder: {[n.node_type for n in root.postorder()]}")

    # AST Encoder
    print("\n2. AST ENCODING")
    print("-" * 40)
    encoder = ASTEncoder(d_model=64)
    tree_emb = encoder.encode_tree(root)
    print(f"  Tree embedding shape: {tree_emb.shape}")
    print(f"  Tree embedding norm: {np.linalg.norm(tree_emb):.3f}")

    # Pattern Matching
    print("\n3. PATTERN MATCHING")
    print("-" * 40)
    matcher = CodePatternMatcher()
    ifs = matcher.find_pattern(root, "If")
    returns = matcher.find_pattern(root, "Return")
    print(f"  If nodes: {len(ifs)}")
    print(f"  Return nodes: {len(returns)}")

    # Code Extraction
    print("\n4. CODE EXTRACTION")
    print("-" * 40)
    extractor = CodeExtractor()
    functions = extractor.extract_functions(root)
    complexity = extractor.compute_complexity(root)
    print(f"  Functions: {functions}")
    print(f"  Cyclomatic complexity: {complexity}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_ast_processing()
