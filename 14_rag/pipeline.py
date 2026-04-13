"""
################################################################################
RAG PIPELINE — RETRIEVAL-AUGMENTED GENERATION
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is RAG?
    RAG combines language models with external knowledge retrieval.
    Instead of relying only on training data, the model:
    1. Retrieves relevant documents from a knowledge base
    2. Generates answers using both the query and retrieved documents

Why does it matter?
    RAG solves key LLM limitations:
    - Knowledge cutoff: LLMs don't know recent events
    - Hallucination: LLMs can make up facts
    - Domain expertise: LLMs may lack specialized knowledge
    - Privacy: Keep sensitive data in your knowledge base

Pipeline:
    Query → Embed → Search Vector DB → Retrieve Docs → Generate Answer

Interview Questions:
    1. "What is RAG?"
       Retrieval-Augmented Generation: combines retrieval with generation.
       The model retrieves relevant documents and uses them to generate
       more accurate and grounded answers.

    2. "When should I use RAG?"
       When you need:
       - Up-to-date information
       - Domain-specific knowledge
       - Reduced hallucination
       - Citable answers

    3. "What are the components of RAG?"
       - Embedding model: converts text to vectors
       - Vector database: stores and searches embeddings
       - Retriever: finds relevant documents
       - Generator: LLM that produces answers

################################################################################
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import math

################################################################################
# SECTION 1: TEXT EMBEDDER
################################################################################

class TextEmbedder:
    """
    Text Embedding Model
    =====================

    Converts text to dense vectors for semantic search.

    In production, use models like:
    - text-embedding-3-small (OpenAI)
    - e5-large-v2 (Microsoft)
    - gte-large (Alibaba)
    - bge-large (BAAI)
    """

    def __init__(self, d_model: int = 384):
        self.d_model = d_model
        # Simplified embedding weights
        self.embedding = np.random.randn(10000, d_model) * 0.02

    def embed(self, text: str) -> np.ndarray:
        """
        Embed text to vector.

        Args:
            text: Input text

        Returns:
            embedding: [d_model]
        """
        # Simplified: hash text to get consistent embedding
        hash_val = hash(text) % 10000
        return self.embedding[hash_val]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        return np.array([self.embed(t) for t in texts])


################################################################################
# SECTION 2: VECTOR STORE
################################################################################

class VectorStore:
    """
    Vector Database
    ===============

    Stores document embeddings and enables similarity search.

    Operations:
    1. Add documents with embeddings
    2. Search for similar documents
    3. Return top-K results

    In production, use:
    - Pinecone
    - Weaviate
    - ChromaDB
    - FAISS
    """

    def __init__(self, d_model: int = 384):
        self.d_model = d_model
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def add(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add documents to the store.

        Args:
            documents: List of text documents
            embeddings: Document embeddings [n × d_model]
            metadata: Optional metadata for each document
        """
        for i, doc in enumerate(documents):
            self.documents.append({
                'text': doc,
                'metadata': metadata[i] if metadata else {},
                'embedding': embeddings[i]
            })

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector [d_model]
            top_k: Number of results to return

        Returns:
            List of documents with scores
        """
        if self.embeddings is None:
            return []

        # Compute cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)

        similarities = np.dot(doc_norms, query_norm)

        # Get top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx]['text'],
                'metadata': self.documents[idx]['metadata'],
                'score': float(similarities[idx])
            })

        return results


################################################################################
# SECTION 3: RAG PIPELINE
################################################################################

class RAGPipeline:
    """
    RAG Pipeline
    =============

    Complete RAG system combining retrieval and generation.

    Steps:
    1. Embed the query
    2. Search for relevant documents
    3. Construct prompt with retrieved context
    4. Generate answer using LLM

    Architecture:
    ┌─────────────────────────────────────────────────┐
    │ User Query: "What is our refund policy?"         │
    │        ↓                                          │
    │ Embed Query → [0.2, -0.5, 0.8, ...]            │
    │        ↓                                          │
    │ Vector Search → Top-5 documents                  │
    │        ↓                                          │
    │ Construct Prompt:                                 │
    │   "Context: [doc1, doc2, ...]                    │
    │    Question: What is our refund policy?"          │
    │        ↓                                          │
    │ LLM Generation → "Our refund policy allows..."   │
    └─────────────────────────────────────────────────┘

    Interview Questions:
        1. "How do you evaluate RAG quality?"
           Metrics: retrieval accuracy, answer quality, faithfulness.
           Use RAGAS framework for comprehensive evaluation.

        2. "What's the chunk size for RAG?"
           Typically 256-512 tokens. Smaller chunks = more precise retrieval.
           Larger chunks = more context for generation.

        3. "How do you handle multi-hop reasoning?"
           Use iterative retrieval: retrieve → generate partial answer →
           retrieve again → generate final answer.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    def add_documents(self, documents: List[str]):
        """
        Add documents to the knowledge base.

        Documents are chunked and embedded.
        """
        # Chunk documents
        chunks = []
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)

        # Embed chunks
        embeddings = self.embedder.embed_batch(chunks)

        # Add to vector store
        self.vector_store.add(chunks, embeddings)

    def _chunk_document(self, document: str) -> List[str]:
        """
        Split document into chunks.

        Args:
            document: Full document text

        Returns:
            List of text chunks
        """
        words = document.split()
        chunks = []

        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            i += self.chunk_size - self.chunk_overlap

        return chunks

    def query(self, question: str) -> Dict:
        """
        Answer a question using RAG.

        Args:
            question: User question

        Returns:
            Dict with answer and sources
        """
        # Step 1: Embed query
        query_embedding = self.embedder.embed(question)

        # Step 2: Retrieve relevant documents
        results = self.vector_store.search(query_embedding, self.top_k)

        # Step 3: Construct prompt
        context = "\n\n".join([r['text'] for r in results])
        prompt = self._construct_prompt(question, context)

        # Step 4: Generate answer (simplified)
        answer = self._generate_answer(prompt)

        return {
            'answer': answer,
            'sources': results,
            'prompt': prompt
        }

    def _construct_prompt(self, question: str, context: str) -> str:
        """Construct RAG prompt."""
        return f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""

    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using LLM (simplified)."""
        # In production: call LLM API
        return "Generated answer based on retrieved context"


################################################################################
# SECTION 4: TESTING & EXAMPLES
################################################################################

def demonstrate_rag():
    """Demonstrate RAG concepts."""
    print("=" * 70)
    print("RAG PIPELINE DEMONSTRATION")
    print("=" * 70)

    # Create components
    print("\n--- Setting up RAG ---")
    embedder = TextEmbedder(d_model=384)
    vector_store = VectorStore(d_model=384)

    # Create RAG pipeline
    rag = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=256,
        top_k=3
    )

    # Add documents
    print("\n--- Adding Documents ---")
    documents = [
        "Our refund policy allows returns within 30 days of purchase. Items must be unused.",
        "Shipping is free for orders over $50. Standard delivery takes 3-5 business days.",
        "Customer support is available Monday-Friday, 9am-5pm EST.",
    ]
    rag.add_documents(documents)
    print(f"Added {len(documents)} documents")

    # Query
    print("\n--- Querying ---")
    result = rag.query("What is the refund policy?")
    print(f"Question: What is the refund policy?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")

    # Show retrieval
    print("\n--- Retrieved Documents ---")
    for i, source in enumerate(result['sources']):
        print(f"{i+1}. Score: {source['score']:.3f}")
        print(f"   Text: {source['text'][:100]}...")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_rag()


################################################################################
# REFERENCES
################################################################################

# [1] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
# [2] Gao, Y., et al. (2024). RAGAS: Automated Evaluation of RAG.

################################################################################
