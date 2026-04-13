"""
################################################################################
MULTIMODAL RAG — RETRIEVAL OVER IMAGES, TABLES, AND DOCUMENTS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Multimodal RAG?
    RAG that processes and retrieves from multiple modalities:
    text, images, tables, and structured documents. Real-world
    documents contain all these formats.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math


################################################################################
# SECTION 1: DOCUMENT PROCESSOR
################################################################################

class DocumentProcessor:
    """
    Extract content from documents (text, tables, images).

    Interview Question:
        "How does Multimodal RAG work?"
        Process documents to extract text, tables, and images separately.
        Embed each modality. At query time, retrieve the most relevant
        content regardless of format. Combine retrieved content as
        context for the LLM.
    """

    def extract_text(self, document: str) -> List[str]:
        """Split document into text chunks."""
        sentences = document.split('. ')
        return [s.strip() for s in sentences if s.strip()]

    def extract_tables(self, document: str) -> List[Dict]:
        """Extract tables from document."""
        # Simulate table extraction
        return [{'headers': ['Name', 'Value'], 'rows': [['A', '1'], ['B', '2']]}]

    def extract_images(self, document: str) -> List[Dict]:
        """Extract image descriptions."""
        return [{'description': 'Diagram showing architecture', 'type': 'diagram'}]


################################################################################
# SECTION 2: IMAGE RETRIEVER
################################################################################

class ImageRetriever:
    """
    Retrieve images by visual similarity.

    Interview Question:
        "How do you do image retrieval in RAG?"
        (1) Embed images using a vision encoder (ViT, CLIP),
        (2) Embed text queries in the same space,
        (3) Compute cosine similarity between query and image embeddings,
        (4) Return top-K most similar images.
    """

    def __init__(self, d_model: int = 256):
        self.d_model = d_model

    def embed_image(self, image_description: str) -> np.ndarray:
        """Embed image to vector."""
        return np.random.randn(self.d_model)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text query."""
        return np.random.randn(self.d_model)

    def search(self, query: str, image_descriptions: List[str],
               top_k: int = 3) -> List[Dict]:
        """Search for relevant images."""
        query_emb = self.embed_text(query)
        results = []
        for desc in image_descriptions:
            img_emb = self.embed_image(desc)
            score = np.dot(query_emb, img_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(img_emb) + 1e-8)
            results.append({'description': desc, 'score': score})

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]


################################################################################
# SECTION 3: TABLE RETRIEVER
################################################################################

class TableRetriever:
    """
    Retrieve and query tables.

    Interview Question:
        "How do you handle tables in RAG?"
        (1) Serialize table to text (headers + rows),
        (2) Embed serialized text,
        (3) For queries: find tables with relevant schema,
        (4) For row retrieval: find rows matching query conditions.
    """

    def embed_table(self, table: Dict) -> np.ndarray:
        """Embed table as vector."""
        text = f"Headers: {table['headers']}. Rows: {len(table['rows'])}"
        return np.random.randn(64)

    def search(self, query: str, tables: List[Dict], top_k: int = 3) -> List[Dict]:
        """Search for relevant tables."""
        query_emb = np.random.randn(64)
        results = []
        for table in tables:
            table_emb = self.embed_table(table)
            score = np.dot(query_emb, table_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(table_emb) + 1e-8)
            results.append({'table': table, 'score': score})

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]


################################################################################
# SECTION 4: MULTIMODAL RAG PIPELINE
################################################################################

class MultimodalRAGPipeline:
    """
    Full Multimodal RAG Pipeline.

    Interview Question:
        "How do you build a multimodal RAG system?"
        (1) Process documents to extract text, tables, images,
        (2) Embed each modality separately,
        (3) At query time, retrieve from all modalities,
        (4) Rank by relevance across modalities,
        (5) Format retrieved content for LLM context,
        (6) Generate answer with multimodal context.
    """

    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.image_retriever = ImageRetriever()
        self.table_retriever = TableRetriever()

    def index(self, document: str) -> Dict:
        """Index a document."""
        text_chunks = self.doc_processor.extract_text(document)
        tables = self.doc_processor.extract_tables(document)
        images = self.doc_processor.extract_images(document)

        return {
            'text_chunks': text_chunks,
            'tables': tables,
            'images': images
        }

    def retrieve(self, query: str, index: Dict) -> Dict:
        """Retrieve relevant content across modalities."""
        text_results = index['text_chunks'][:3]  # Simplified
        image_results = self.image_retriever.search(query, [i['description'] for i in index['images']])
        table_results = self.table_retriever.search(query, index['tables'])

        return {
            'query': query,
            'text': text_results,
            'images': image_results,
            'tables': table_results,
        }

    def format_context(self, retrieved: Dict) -> str:
        """Format retrieved content as LLM context."""
        parts = []
        if retrieved['text']:
            parts.append("Text: " + " ".join(retrieved['text'][:2]))
        if retrieved['images']:
            parts.append("Images: " + retrieved['images'][0]['description'])
        if retrieved['tables']:
            t = retrieved['tables'][0]['table']
            parts.append(f"Table: {t['headers']}")
        return "\n".join(parts)


################################################################################
# SECTION 5: DEMONSTRATION
################################################################################

def demonstrate_multimodal_rag():
    """Demonstrate Multimodal RAG."""
    print("=" * 70)
    print("MULTIMODAL RAG DEMONSTRATION")
    print("=" * 70)

    document = """
    The system architecture shows a transformer model.
    Performance metrics: accuracy 95%, latency 10ms.
    The diagram illustrates the data flow pipeline.
    """

    # Document Processing
    print("\n1. DOCUMENT PROCESSING")
    print("-" * 40)
    processor = DocumentProcessor()
    text = processor.extract_text(document)
    tables = processor.extract_tables(document)
    images = processor.extract_images(document)
    print(f"  Text chunks: {len(text)}")
    print(f"  Tables: {len(tables)}")
    print(f"  Images: {len(images)}")

    # Image Retrieval
    print("\n2. IMAGE RETRIEVAL")
    print("-" * 40)
    ir = ImageRetriever()
    results = ir.search("architecture diagram", [i['description'] for i in images])
    for r in results:
        print(f"  {r['description']} (score: {r['score']:.3f})")

    # Full Pipeline
    print("\n3. MULTIMODAL RAG PIPELINE")
    print("-" * 40)
    pipeline = MultimodalRAGPipeline()
    index = pipeline.index(document)
    retrieved = pipeline.retrieve("What is the accuracy?", index)
    context = pipeline.format_context(retrieved)
    print(f"  Retrieved text: {len(retrieved['text'])} chunks")
    print(f"  Retrieved images: {len(retrieved['images'])}")
    print(f"  Context preview: {context[:100]}...")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_multimodal_rag()
