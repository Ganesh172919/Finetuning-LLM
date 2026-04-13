"""
################################################################################
GRAPH RAG — KNOWLEDGE GRAPH + COMMUNITY SUMMARIZATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Graph RAG?
    Using knowledge graphs for retrieval-augmented generation. Build a
    graph of entities and relations, detect communities, and use
    community summaries to answer global/summary questions.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass, field
from collections import defaultdict
import math


################################################################################
# SECTION 1: KNOWLEDGE GRAPH
################################################################################

class KnowledgeGraph:
    """
    Knowledge Graph — Store entities and relations.

    Interview Question:
        "What is Graph RAG?"
        Graph RAG builds a knowledge graph from documents (entities +
        relations). For queries: (1) find relevant entities, (2) extract
        subgraph, (3) detect communities, (4) summarize communities,
        (5) use summaries as context for LLM. Excels at global questions
        that vector RAG struggles with.
    """

    def __init__(self):
        self.entities: Dict[str, Dict] = {}
        self.relations: List[Dict] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)

    def add_entity(self, name: str, entity_type: str = "entity",
                   embedding: np.ndarray = None):
        """Add an entity to the graph."""
        self.entities[name] = {
            'type': entity_type,
            'embedding': embedding or np.random.randn(64)
        }

    def add_relation(self, source: str, target: str, rel_type: str):
        """Add a relation between entities."""
        self.relations.append({
            'source': source, 'target': target, 'type': rel_type
        })
        self.adjacency[source].append(target)
        self.adjacency[target].append(source)

    def get_neighbors(self, entity: str, max_hops: int = 1) -> Set[str]:
        """Get entities within max_hops of entity."""
        visited = {entity}
        frontier = {entity}
        for _ in range(max_hops):
            next_frontier = set()
            for e in frontier:
                for neighbor in self.adjacency.get(e, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        return visited

    def get_subgraph(self, entities: Set[str]) -> Dict:
        """Extract subgraph containing given entities."""
        sub_relations = [r for r in self.relations
                        if r['source'] in entities and r['target'] in entities]
        return {'entities': entities, 'relations': sub_relations}


################################################################################
# SECTION 2: ENTITY EXTRACTOR
################################################################################

class EntityExtractor:
    """
    Extract entities from text.

    Interview Question:
        "How do you extract entities for Graph RAG?"
        Use NER to find entities, link to existing graph entities,
        extract relations (subject-relation-object triples). For LLM-based
        extraction: prompt the model to list entities and relations.
    """

    def extract(self, text: str) -> List[Dict]:
        """
        Extract entities from text.

        Args:
            text: Input text

        Returns:
            List of entity dictionaries
        """
        # Simulate entity extraction
        words = text.split()
        entities = []
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                entities.append({
                    'name': word,
                    'type': 'entity',
                    'position': i
                })
        return entities

    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relations between entities."""
        relations = []
        for i in range(len(entities) - 1):
            relations.append({
                'source': entities[i]['name'],
                'target': entities[i+1]['name'],
                'type': 'related_to'
            })
        return relations


################################################################################
# SECTION 3: COMMUNITY SUMMARIZER
################################################################################

class CommunitySummarizer:
    """
    Detect and summarize communities in the knowledge graph.

    Interview Question:
        "How does community summarization work in Graph RAG?"
        (1) Detect communities using Louvain or similar algorithm,
        (2) For each community, generate a summary of key entities
        and their relationships, (3) Use community summaries as context
        for global queries ("summarize the corpus").
    """

    def detect_communities(self, kg: KnowledgeGraph) -> List[Set[str]]:
        """
        Detect communities (simplified).

        Args:
            kg: Knowledge graph

        Returns:
            List of community entity sets
        """
        # Simplified: use connected components
        visited = set()
        communities = []

        for entity in kg.entities:
            if entity not in visited:
                component = kg.get_neighbors(entity, max_hops=2)
                communities.append(component)
                visited.update(component)

        return communities

    def summarize_community(self, kg: KnowledgeGraph,
                           community: Set[str]) -> str:
        """
        Generate summary for a community.

        Args:
            kg: Knowledge graph
            community: Set of entity names

        Returns:
            Community summary string
        """
        entities = list(community)[:10]
        relations = [r for r in kg.relations
                    if r['source'] in community and r['target'] in community]

        summary = f"Community with {len(community)} entities: {', '.join(entities)}. "
        summary += f"{len(relations)} relationships."
        return summary


################################################################################
# SECTION 4: GRAPH RAG PIPELINE
################################################################################

class GraphRAGPipeline:
    """
    Full Graph RAG Pipeline.

    Two modes:
    - Global queries: use community summaries
    - Local queries: use entity neighborhood

    Interview Question:
        "How does Graph RAG handle different query types?"
        Global queries ("summarize the corpus") → community summaries.
        Local queries ("what is entity X?") → entity neighborhood.
        Hybrid: combine vector search (for relevant passages) with
        graph search (for structured knowledge).
    """

    def __init__(self):
        self.kg = KnowledgeGraph()
        self.extractor = EntityExtractor()
        self.summarizer = CommunitySummarizer()

    def index_document(self, text: str):
        """Index a document into the knowledge graph."""
        entities = self.extractor.extract(text)
        relations = self.extractor.extract_relations(text, entities)

        for e in entities:
            self.kg.add_entity(e['name'], e['type'])
        for r in relations:
            self.kg.add_relation(r['source'], r['target'], r['type'])

    def query_global(self, question: str) -> Dict:
        """Answer global/summary questions using communities."""
        communities = self.summarizer.detect_communities(self.kg)
        summaries = [self.summarizer.summarize_community(self.kg, c)
                    for c in communities]

        return {
            'question': question,
            'strategy': 'global',
            'n_communities': len(communities),
            'summaries': summaries,
            'answer': f"Based on {len(communities)} communities: summary"
        }

    def query_local(self, entity: str, question: str) -> Dict:
        """Answer entity-specific questions."""
        neighbors = self.kg.get_neighbors(entity, max_hops=2)
        subgraph = self.kg.get_subgraph(neighbors)

        return {
            'question': question,
            'strategy': 'local',
            'entity': entity,
            'neighborhood_size': len(neighbors),
            'answer': f"About {entity}: information from {len(neighbors)} related entities"
        }


################################################################################
# SECTION 5: DEMONSTRATION
################################################################################

def demonstrate_graph_rag():
    """Demonstrate Graph RAG."""
    print("=" * 70)
    print("GRAPH RAG DEMONSTRATION")
    print("=" * 70)

    # Knowledge Graph
    print("\n1. KNOWLEDGE GRAPH")
    print("-" * 40)
    kg = KnowledgeGraph()
    kg.add_entity("Python", "language")
    kg.add_entity("JavaScript", "language")
    kg.add_entity("React", "framework")
    kg.add_entity("Django", "framework")
    kg.add_relation("React", "JavaScript", "built_with")
    kg.add_relation("Django", "Python", "built_with")
    print(f"  Entities: {len(kg.entities)}")
    print(f"  Relations: {len(kg.relations)}")
    print(f"  Python neighbors: {kg.get_neighbors('Python')}")

    # Entity Extraction
    print("\n2. ENTITY EXTRACTION")
    print("-" * 40)
    extractor = EntityExtractor()
    text = "Python is a programming language used by Django framework"
    entities = extractor.extract(text)
    print(f"  Text: '{text}'")
    print(f"  Entities: {[e['name'] for e in entities]}")

    # Community Detection
    print("\n3. COMMUNITY DETECTION")
    print("-" * 40)
    summarizer = CommunitySummarizer()
    communities = summarizer.detect_communities(kg)
    print(f"  Communities: {len(communities)}")
    for i, c in enumerate(communities):
        summary = summarizer.summarize_community(kg, c)
        print(f"  Community {i}: {summary[:60]}...")

    # Full Pipeline
    print("\n4. GRAPH RAG PIPELINE")
    print("-" * 40)
    pipeline = GraphRAGPipeline()
    pipeline.index_document("Python is used by Django for web development")
    pipeline.index_document("JavaScript powers React for frontend")

    global_result = pipeline.query_global("Summarize all technologies")
    print(f"  Global: {global_result['n_communities']} communities")

    local_result = pipeline.query_local("Python", "What is Python used for?")
    print(f"  Local: {local_result['neighborhood_size']} neighbors")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_graph_rag()
