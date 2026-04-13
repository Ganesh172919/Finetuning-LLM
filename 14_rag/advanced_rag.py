"""
################################################################################
ADVANCED RAG — SELF-RAG, CRAG, ADAPTIVE RAG
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Advanced RAG?
    Next-generation RAG techniques that add self-reflection, correction,
    and adaptive retrieval to the basic RAG pipeline.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


################################################################################
# SECTION 1: SELF-RAG
################################################################################

class SelfRAG:
    """
    Self-RAG — Self-Reflective RAG.

    The model decides WHEN to retrieve and self-evaluates output quality.

    Paper: "Self-RAG: Learning to Retrieve, Generate, and Critique
            through Self-Reflection" (Asai et al., ICLR 2024)

    Reflection tokens:
    - [Retrieve]: should I retrieve? (yes/no)
    - [Relevant]: is doc relevant? (relevant/irrelevant/ambiguous)
    - [Supported]: is output supported? (fully/partially/no)
    - [Utility]: answer quality (1-5)

    Interview Question:
        "How does Self-RAG differ from standard RAG?"
        Self-RAG adds self-reflection: (1) decides IF retrieval is needed,
        (2) evaluates if retrieved docs are relevant, (3) checks if the
        answer is supported by evidence, (4) scores answer quality.
        This makes RAG more selective and accurate.
    """

    def __init__(self):
        self.retrieve_threshold = 0.5

    def should_retrieve(self, query: str) -> bool:
        """Decide if query needs retrieval."""
        # Simulate: complex queries need retrieval
        return len(query.split()) > 5 or '?' in query

    def assess_relevance(self, query: str, document: str) -> str:
        """Assess document relevance."""
        # Simulate relevance check
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        overlap = len(query_words & doc_words)
        if overlap > 2:
            return "relevant"
        elif overlap > 0:
            return "ambiguous"
        return "irrelevant"

    def assess_support(self, answer: str, evidence: str) -> str:
        """Check if answer is supported by evidence."""
        answer_words = set(answer.lower().split())
        evidence_words = set(evidence.lower().split())
        overlap = len(answer_words & evidence_words)
        if overlap > len(answer_words) * 0.5:
            return "fully_supported"
        elif overlap > 0:
            return "partially_supported"
        return "not_supported"

    def generate(self, query: str, documents: List[str]) -> Dict:
        """
        Self-RAG generation pipeline.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            Dictionary with answer and reflection scores
        """
        should_retrieve = self.should_retrieve(query)

        if should_retrieve:
            relevance_scores = [self.assess_relevance(query, doc) for doc in documents]
            relevant_docs = [d for d, r in zip(documents, relevance_scores) if r == "relevant"]
        else:
            relevant_docs = []

        # Generate answer
        answer = f"Answer to: {query}"
        evidence = " ".join(relevant_docs) if relevant_docs else ""

        support = self.assess_support(answer, evidence) if evidence else "no_evidence"

        return {
            'query': query,
            'answer': answer,
            'retrieved': should_retrieve,
            'n_relevant': len(relevant_docs),
            'support': support,
        }


################################################################################
# SECTION 2: CRAG
################################################################################

class CRAG:
    """
    CRAG — Corrective RAG.

    Assess retrieval quality and correct if wrong.

    Paper: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)

    Steps:
    1. Retrieve documents
    2. Assess quality (correct/ambiguous/incorrect)
    3. If correct: use directly
    4. If ambiguous: decompose query, re-retrieve
    5. If incorrect: fall back to web search / LLM knowledge

    Interview Question:
        "What is Corrective RAG?"
        CRAG adds a quality check after retrieval. If retrieved docs are
        incorrect or ambiguous, it corrects by: (1) decomposing the query
        for better retrieval, (2) falling back to web search, (3) refining
        the knowledge extract. This prevents bad retrievals from hurting
        generation quality.
    """

    def assess_retrieval(self, query: str, documents: List[str]) -> str:
        """Assess retrieval quality."""
        if not documents:
            return "incorrect"
        # Simulate quality assessment
        return np.random.choice(["correct", "ambiguous", "incorrect"],
                               p=[0.6, 0.3, 0.1])

    def refine_knowledge(self, documents: List[str]) -> str:
        """Extract key information from documents."""
        return " ".join(documents[:2])  # Take first 2 docs

    def web_search_fallback(self, query: str) -> str:
        """Fall back to web search."""
        return f"Web search result for: {query}"

    def generate(self, query: str, documents: List[str]) -> Dict:
        """
        CRAG generation pipeline.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            Dictionary with answer and correction info
        """
        quality = self.assess_retrieval(query, documents)

        if quality == "correct":
            context = self.refine_knowledge(documents)
            correction = "none"
        elif quality == "ambiguous":
            context = self.refine_knowledge(documents)
            correction = "refined"
        else:
            context = self.web_search_fallback(query)
            correction = "web_search"

        answer = f"Based on {correction}: answer to '{query}'"

        return {
            'query': query,
            'answer': answer,
            'retrieval_quality': quality,
            'correction': correction,
        }


################################################################################
# SECTION 3: ADAPTIVE RAG
################################################################################

class AdaptiveRAG:
    """
    Adaptive RAG — Route queries to appropriate strategy.

    Interview Question:
        "How do you decide which RAG strategy to use?"
        Classify query complexity: (1) Simple → direct LLM (no retrieval),
        (2) Moderate → single-hop RAG, (3) Complex → multi-hop RAG with
        query decomposition. Route based on query features like length,
        question type, and ambiguity.
    """

    def classify_query(self, query: str) -> str:
        """Classify query complexity."""
        n_words = len(query.split())
        if n_words < 5:
            return "simple"
        elif n_words < 15:
            return "moderate"
        return "complex"

    def route(self, query: str, documents: List[str]) -> Dict:
        """
        Route query to appropriate RAG strategy.

        Args:
            query: User query
            documents: Available documents

        Returns:
            Dictionary with strategy and answer
        """
        complexity = self.classify_query(query)

        if complexity == "simple":
            strategy = "direct_llm"
            answer = f"Direct answer: {query}"
        elif complexity == "moderate":
            strategy = "single_hop_rag"
            answer = f"Single-hop RAG answer: {query}"
        else:
            strategy = "multi_hop_rag"
            answer = f"Multi-hop RAG answer: {query}"

        return {
            'query': query,
            'complexity': complexity,
            'strategy': strategy,
            'answer': answer,
        }


################################################################################
# SECTION 4: QUERY DECOMPOSER
################################################################################

class QueryDecomposer:
    """
    Decompose complex queries into sub-questions.

    Interview Question:
        "How do you handle complex queries in RAG?"
        Decompose into sub-questions: (1) Sequential: step-by-step,
        each depends on previous answer, (2) Parallel: independent
        sub-questions answered simultaneously. Then compose final
        answer from all sub-answers.
    """

    def decompose(self, query: str) -> List[str]:
        """
        Decompose query into sub-questions.

        Args:
            query: Complex query

        Returns:
            List of sub-questions
        """
        # Simulate decomposition
        words = query.split()
        mid = len(words) // 2
        return [
            " ".join(words[:mid]) + "?",
            " ".join(words[mid:]) + "?"
        ]

    def compose(self, sub_answers: List[str]) -> str:
        """Compose final answer from sub-answers."""
        return " ".join(sub_answers)


################################################################################
# SECTION 5: MULTI-HOP RAG
################################################################################

class MultiHopRAG:
    """
    Multi-Hop RAG — Iterative retrieval with reasoning.

    Interview Question:
        "What is multi-hop RAG?"
        Iterative retrieval: (1) initial retrieval, (2) reason over docs,
        (3) identify information gaps, (4) generate new query, (5) retrieve
        again, (6) compose final answer. Each "hop" fills in missing info.
    """

    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops

    def identify_gaps(self, query: str, current_context: str) -> List[str]:
        """Identify information gaps."""
        return [f"Missing info for: {query}"]

    def generate_sub_query(self, gap: str) -> str:
        """Generate retrieval query for a gap."""
        return f"Search: {gap}"

    def retrieve(self, query: str) -> List[str]:
        """Simulate retrieval."""
        return [f"Document about {query}"]

    def run(self, query: str) -> Dict:
        """
        Multi-hop RAG pipeline.

        Args:
            query: Complex query

        Returns:
            Dictionary with hops and final answer
        """
        context = ""
        hops = []

        for hop in range(self.max_hops):
            # Retrieve
            docs = self.retrieve(query if hop == 0 else sub_query)
            context += " ".join(docs)

            # Check gaps
            gaps = self.identify_gaps(query, context)
            if not gaps:
                break

            # Generate sub-query for next hop
            sub_query = self.generate_sub_query(gaps[0])
            hops.append({'hop': hop, 'query': sub_query, 'docs': len(docs)})

        return {
            'query': query,
            'n_hops': len(hops),
            'hops': hops,
            'answer': f"Multi-hop answer: {query}",
        }


################################################################################
# SECTION 6: DEMONSTRATION
################################################################################

def demonstrate_advanced_rag():
    """Demonstrate advanced RAG techniques."""
    print("=" * 70)
    print("ADVANCED RAG DEMONSTRATION")
    print("=" * 70)

    docs = ["Document about Python programming", "Document about machine learning",
            "Document about cooking recipes"]

    # Self-RAG
    print("\n1. SELF-RAG")
    print("-" * 40)
    self_rag = SelfRAG()
    result = self_rag.generate("What is Python?", docs)
    print(f"  Retrieved: {result['retrieved']}")
    print(f"  Relevant docs: {result['n_relevant']}")
    print(f"  Support: {result['support']}")

    # CRAG
    print("\n2. CRAG")
    print("-" * 40)
    crag = CRAG()
    result = crag.generate("Explain machine learning", docs)
    print(f"  Quality: {result['retrieval_quality']}")
    print(f"  Correction: {result['correction']}")

    # Adaptive RAG
    print("\n3. ADAPTIVE RAG")
    print("-" * 40)
    adaptive = AdaptiveRAG()
    for q in ["Hi", "What is Python?", "How does transformer attention work with multi-head?"]:
        result = adaptive.route(q, docs)
        print(f"  '{q[:30]}...' → {result['strategy']}")

    # Multi-Hop
    print("\n4. MULTI-HOP RAG")
    print("-" * 40)
    mhop = MultiHopRAG(max_hops=3)
    result = mhop.run("Compare Python and ML frameworks")
    print(f"  Hops: {result['n_hops']}")
    print(f"  Answer: {result['answer']}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_advanced_rag()
