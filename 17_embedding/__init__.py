"""
################################################################################
EMBEDDING MODELS — SEMANTIC REPRESENTATIONS
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What are Embedding Models?
    Models that convert text into dense vectors capturing semantic meaning.
    Similar texts have similar embeddings.

Why do they matter?
    Embeddings power:
    - Semantic search
    - RAG retrieval
    - Clustering
    - Classification
    - Recommendation

Historical Evolution:
    - 2013: Word2Vec
    - 2018: BERT embeddings
    - 2022: Sentence-BERT, E5
    - 2023: GTE, BGE
    - 2024: Modern embedding models

Interview Questions:
    1. "What are text embeddings?"
        Dense vectors that capture semantic meaning.
        Similar texts have similar vectors.

    2. "How do you create good embeddings?"
        Use specialized models (E5, GTE, BGE).
        Train with contrastive learning on text pairs.

    3. "What's the difference between word and sentence embeddings?"
        Word: one vector per word
        Sentence: one vector per sentence/document

################################################################################
"""

from .embedding_model import EmbeddingModel, SentenceTransformer
