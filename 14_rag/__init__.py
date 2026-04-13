"""
################################################################################
RETRIEVAL-AUGMENTED GENERATION (RAG) — GROUNDING AI IN KNOWLEDGE
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is RAG?
    RAG combines language models with external knowledge retrieval.
    Instead of relying only on training data, the model retrieves relevant
    documents from a knowledge base and generates answers using both the
    query and retrieved documents.

Why does it matter?
    RAG solves key LLM limitations:
    - Knowledge cutoff: LLMs don't know recent events
    - Hallucination: LLMs can make up facts
    - Domain expertise: LLMs may lack specialized knowledge
    - Privacy: Keep sensitive data in your knowledge base

How does it work?
    1. Basic RAG — Query → Retrieve → Generate
    2. Advanced RAG — Self-reflection, correction, adaptive retrieval
    3. Graph RAG — Knowledge graph construction + community summarization
    4. Multimodal RAG — Images, tables, and text retrieval
    5. Agentic RAG — Multi-step retrieval with reasoning

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ Advanced RAG Pipeline                                       │
    │                                                              │
    │  Query → [Classify] → Route to strategy                    │
    │    ├── Simple → Direct LLM answer                          │
    │    ├── Moderate → Single-hop RAG                           │
    │    └── Complex → Multi-hop RAG + query decomposition       │
    │                                                              │
    │  Self-RAG: Generate → Reflect → Correct → Answer           │
    │  CRAG: Retrieve → Assess → Correct if wrong → Answer       │
    │  Graph RAG: Entities → Subgraph → Communities → Answer     │
    └─────────────────────────────────────────────────────────────┘

Historical Context:
    - 2020: RAG paper (Lewis et al.)
    - 2022: Vector databases (Pinecone, Weaviate)
    - 2023: LangChain, LlamaIndex
    - 2024: Self-RAG, CRAG, Advanced RAG
    - 2025: Graph RAG (Microsoft), Agentic RAG
    - 2026: Multimodal RAG, verified retrieval

################################################################################
"""

from .retriever import DenseRetriever, SparseRetriever
from .vector_store import VectorStore
from .pipeline import RAGPipeline
from .advanced_rag import SelfRAG, CRAG, AdaptiveRAG, QueryDecomposer, MultiHopRAG
from .graph_rag import KnowledgeGraph, EntityExtractor, GraphRetriever, CommunitySummarizer, GraphRAGPipeline
from .multimodal_rag import DocumentProcessor, ImageRetriever, TableRetriever, MultimodalRAGPipeline
