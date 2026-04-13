"""
################################################################################
DATA CURATION PIPELINE — Deduplication, Quality Filtering, Decontamination
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Data Curation?
    Data curation is the systematic process of cleaning raw text corpora
    to produce high-quality training data for language models. It removes
    duplicates, filters low-quality documents, and eliminates benchmark
    contamination that would invalidate evaluation results.

Why does it matter?
    Model quality is bounded by data quality. A model trained on noisy,
    duplicated, or contaminated data will:
    - Memorize duplicated patterns instead of learning general knowledge
    - Learn noise from low-quality documents (spam, boilerplate, gibberish)
    - Produce artificially inflated benchmark scores if eval data leaks in
    The 2024-2026 era proved that data curation is the highest-leverage
    investment in LLM training — more impactful than architecture changes.

How does it work?
    1. Exact Deduplication: Hash documents with SHA-256, remove byte-identical
       copies. Fast, deterministic, catches copy-pasted content.
    2. Near-Deduplication: MinHash + LSH estimates Jaccard similarity between
       documents. Removes near-copies that differ by minor edits.
    3. Quality Filtering: Score each document by heuristic rules (length,
       repetitiveness, special character ratio) and/or perplexity under a
       reference model. Reject documents below threshold.
    4. Decontamination: Build n-gram index of eval benchmarks, check each
       training document for overlap, remove contaminated documents.
    5. Domain Mixture: Balance remaining documents across domains (web, code,
       math, synthetic, curated) according to configured ratios.

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Raw Corpus (billions of documents)                               │
    │        ↓                                                          │
    │ ┌──────────────────────────────────────────────────────────────┐ │
    │ │ Exact Dedup (SHA-256 hash) — removes byte-identical docs    │ │
    │ └──────────────────────────────────────────────────────────────┘ │
    │        ↓                                                          │
    │ ┌──────────────────────────────────────────────────────────────┐ │
    │ │ Near-Dedup (MinHash + LSH) — removes near-copies            │ │
    │ └──────────────────────────────────────────────────────────────┘ │
    │        ↓                                                          │
    │ ┌──────────────────────────────────────────────────────────────┐ │
    │ │ Quality Filter (heuristic + perplexity) — removes noise      │ │
    │ └──────────────────────────────────────────────────────────────┘ │
    │        ↓                                                          │
    │ ┌──────────────────────────────────────────────────────────────┐ │
    │ │ Decontamination (n-gram overlap) — removes eval leakage      │ │
    │ └──────────────────────────────────────────────────────────────┘ │
    │        ↓                                                          │
    │ ┌──────────────────────────────────────────────────────────────┐ │
    │ │ Domain Mixer — balances web/code/math/synthetic/curated      │ │
    │ └──────────────────────────────────────────────────────────────┘ │
    │        ↓                                                          │
    │ Clean Corpus (ready for tokenization and training)               │
    └─────────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 1997: Broder introduces MinHash for web page deduplication at AltaVista
    - 2007: Charikar simhash — another near-dedup approach for web-scale
    - 2020: GPT-3 paper emphasizes data quality over quantity
    - 2023: Phi-1.5/2 shows synthetic + curated data beats raw web scale
    - 2024: FineWeb demonstrates 15T tokens of curated web data
    - 2025: Decontamination becomes standard practice; benchmark leakage
      identified as primary source of inflated scores
    - 2026: Multi-stage curation pipelines are table stakes for any
      competitive LLM training run

INTERVIEW QUESTIONS:
    1. "Why use MinHash/LSH instead of comparing all document pairs?"
       Answer: Comparing all N documents pairwise is O(N^2), infeasible for
       billions of documents. MinHash compresses each document into a compact
       signature (128-256 integers), and LSH bands these signatures so only
       candidate pairs (likely similar) get compared. This reduces the problem
       from O(N^2) to roughly O(N).

    2. "What is the risk of not decontaminating training data?"
       Answer: If benchmark questions or answers appear in training data,
       the model memorizes them, producing artificially high scores. This
       makes evaluation meaningless — you're measuring memorization, not
       capability. Decontamination removes documents with n-gram overlap
       to known benchmarks, ensuring eval scores reflect true generalization.

    3. "How do you set the Jaccard similarity threshold for near-dedup?"
       Answer: Common thresholds are 0.7-0.8. Lower values (0.7) catch more
       near-duplicates but risk removing documents that are legitimately
       similar (e.g., news articles about the same event). Higher values (0.8)
       are more conservative. The right choice depends on downstream task:
       pretraining can tolerate aggressive dedup, while fine-tuning data
       should be more conservative.

################################################################################
"""

import hashlib
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


################################################################################
# SECTION 1: CONFIGURATION
################################################################################


@dataclass
class CurationConfig:
    """
    Configuration for the data curation pipeline.
    ============================

    All thresholds and parameters are explicit config fields.
    No magic numbers — every constant is named and documented.
    """

    # Exact deduplication
    hash_algorithm: str = "sha256"  # Hash function for exact dedup

    # Near-deduplication (MinHash/LSH)
    num_hash_functions: int = 128  # Number of MinHash hash functions
    num_bands: int = 16  # Number of LSH bands
    jaccard_threshold: float = 0.8  # Jaccard similarity threshold
    shingle_size: int = 5  # N-gram shingle size (words)

    # Quality filtering — heuristic
    min_word_count: int = 50  # Minimum words per document
    max_word_count: int = 100000  # Maximum words per document
    min_avg_sentence_length: int = 5  # Minimum average words per sentence
    max_avg_sentence_length: int = 100  # Maximum average words per sentence
    max_special_char_ratio: float = 0.3  # Max ratio of non-alphanumeric chars
    max_repetition_ratio: float = 0.5  # Max ratio of repeated n-grams
    repetition_ngram_size: int = 4  # N-gram size for repetition check

    # Quality filtering — perplexity
    min_perplexity: float = 10.0  # Below this = too repetitive/templated
    max_perplexity: float = 10000.0  # Above this = low quality/gibberish
    perplexity_batch_size: int = 32  # Batch size for perplexity computation

    # Decontamination
    decontam_ngram_size: int = 13  # N-gram size for contamination check
    decontam_overlap_threshold: float = 0.7  # Overlap ratio threshold
    benchmark_names: List[str] = field(default_factory=lambda: [
        "mmlu", "gpqa", "aime", "humaneval", "gsm8k",
        "hle", "arc_agi", "frontiermath",
    ])

    # Domain mixture
    domain_ratios: Dict[str, float] = field(default_factory=lambda: {
        "web": 0.50,
        "code": 0.15,
        "math": 0.15,
        "synthetic": 0.10,
        "curated": 0.10,
    })
    mixture_tolerance: float = 0.05  # Allowable drift from target ratios


################################################################################
# SECTION 2: EXACT DEDUPLICATION
################################################################################


class ExactDeduplicator:
    """
    Exact Deduplicator
    ==================

    Hash-based exact deduplication using SHA-256.
    Removes byte-identical documents from the corpus.

    Algorithm:
        1. For each document, compute SHA-256 hash
        2. Store hash in a set
        3. If hash already seen, mark document as duplicate
        4. Return deduplicated corpus

    Formula:
        hash(doc) = SHA-256(doc.encode('utf-8'))
        is_duplicate = hash(doc) in seen_hashes

    WHY this matters:
        Exact duplicates waste training compute and cause the model to
        overfit to repeated content. A document appearing 100x in the
        corpus effectively has 100x the learning weight.

    Interview Question:
        "Why SHA-256 instead of a faster hash like MD5?"
        Answer: SHA-256 provides collision resistance — the probability of
        two different documents producing the same hash is ~2^-256, negligible
        for any corpus size. MD5 has known collision vulnerabilities. For
        deduplication, we need confidence that identical hashes mean identical
        content. The speed difference is negligible compared to I/O.
    """

    def __init__(self, config: Optional[CurationConfig] = None):
        """
        Initialize the exact deduplicator.

        Args:
            config: Curation configuration. Uses defaults if None.
        """
        self.config = config or CurationConfig()
        self.seen_hashes: Set[str] = set()
        self.stats = {
            "total_documents": 0,
            "duplicates_found": 0,
            "unique_documents": 0,
        }

    def compute_hash(self, document: str) -> str:
        """
        Compute SHA-256 hash of a document.

        Args:
            document: Raw text document

        Returns:
            Hexadecimal hash string

        Explanation:
            Encodes the document as UTF-8 bytes, then computes SHA-256.
            The resulting 256-bit hash is represented as 64 hex characters.
        """
        return hashlib.sha256(document.encode("utf-8")).hexdigest()

    def deduplicate(self, documents: List[str]) -> List[str]:
        """
        Remove exact duplicates from a list of documents.

        Args:
            documents: List of raw text documents

        Returns:
            List of unique documents (order preserved)

        Explanation:
            Iterates through documents, computing SHA-256 for each.
            Documents with previously seen hashes are skipped.
            First occurrence is always kept (preserves corpus order).

        Example:
            >>> dedup = ExactDeduplicator()
            >>> docs = ["hello world", "foo bar", "hello world"]
            >>> unique = dedup.deduplicate(docs)
            >>> len(unique)
            2
        """
        self.stats["total_documents"] = len(documents)
        unique_documents = []

        for doc in documents:
            doc_hash = self.compute_hash(doc)
            if doc_hash not in self.seen_hashes:
                self.seen_hashes.add(doc_hash)
                unique_documents.append(doc)
            else:
                self.stats["duplicates_found"] += 1

        self.stats["unique_documents"] = len(unique_documents)
        dup_rate = self.stats["duplicates_found"] / max(self.stats["total_documents"], 1)
        logger.info(
            f"Exact dedup: {self.stats['duplicates_found']}/{self.stats['total_documents']} "
            f"duplicates removed ({dup_rate:.2%})"
        )
        return unique_documents

    def get_stats(self) -> Dict[str, Any]:
        """Return deduplication statistics."""
        return dict(self.stats)


################################################################################
# SECTION 3: NEAR-DEDUPLICATION (MINHASH + LSH)
################################################################################


class MinHashDeduplicator:
    """
    MinHash Deduplicator
    ====================

    MinHash + Locality-Sensitive Hashing for near-duplicate detection.

    Algorithm:
        1. Generate n-gram shingles for each document
        2. Compute MinHash signature (num_hash_functions hash functions)
        3. LSH banding: split signature into bands, hash each band
        4. Candidate pairs in same bucket → Jaccard similarity check
        5. Remove if Jaccard > threshold

    Formula:
        Jaccard(A, B) = |A ∩ B| / |A ∪ B|
        MinHash(h, S) = min({h(x) : x ∈ S})
        Signature(S) = [MinHash(h_1, S), ..., MinHash(h_n, S)]
        LSH: split signature into b bands of r rows each (b * r = n)

    WHY this matters:
        Near-duplicates (e.g., same news article with minor edits) waste
        training compute and bias the model. MinHash+LSH finds these
        efficiently without O(N^2) pairwise comparison.

    Cite: Broder 1997 "On the Resemblance and Containment of Documents"
    Cite: Leskovec, Rajaraman, Ullman "Mining of Massive Datasets" Ch. 3

    Interview Question:
        "Explain the probability that a candidate pair is found by LSH."
        Answer: For documents with Jaccard similarity s, using b bands of
        r rows each, the probability they share at least one band is:
        P(candidate) = 1 - (1 - s^r)^b. This creates an S-curve: documents
        with similarity above the threshold are very likely to be found,
        while dissimilar documents are very unlikely. The threshold is
        approximately (1/b)^(1/r).
    """

    def __init__(self, config: Optional[CurationConfig] = None):
        """
        Initialize the MinHash deduplicator.

        Args:
            config: Curation configuration. Uses defaults if None.
        """
        self.config = config or CurationConfig()
        self.num_hashes = self.config.num_hash_functions
        self.num_bands = self.config.num_bands
        self.rows_per_band = self.num_hashes // self.num_bands
        self.threshold = self.config.jaccard_threshold
        self.shingle_size = self.config.shingle_size

        # Generate random hash function parameters: h(x) = (a*x + b) % p % max_val
        # Using a large prime and random coefficients
        self._prime = (1 << 31) - 1  # Mersenne prime that fits in int32
        self._max_val = (1 << 31) - 1
        rng = np.random.RandomState(42)  # Deterministic for reproducibility
        self._coefficients_a = rng.randint(1, self._prime, size=self.num_hashes).astype(np.int64)
        self._coefficients_b = rng.randint(0, self._prime, size=self.num_hashes).astype(np.int64)

        # LSH buckets: band_index -> bucket_hash -> list of doc_indices
        self.lsh_buckets: Dict[int, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

        self.stats = {
            "total_documents": 0,
            "candidate_pairs": 0,
            "duplicates_found": 0,
            "unique_documents": 0,
        }

    def _get_shingles(self, document: str) -> Set[int]:
        """
        Generate word-level n-gram shingles and hash them to integers.

        Args:
            document: Raw text document

        Returns:
            Set of hashed shingle values

        Explanation:
            Tokenizes document into words, creates sliding window of
            shingle_size words, hashes each shingle to an integer.
            The set of hashed shingles represents the document for
            MinHash computation.
        """
        words = document.lower().split()
        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i : i + self.shingle_size])
            shingle_hash = int(hashlib.md5(shingle.encode()).hexdigest(), 16) % self._max_val
            shingles.add(shingle_hash)
        return shingles

    def _compute_minhash_signature(self, shingles: Set[int]) -> np.ndarray:
        """
        Compute MinHash signature for a set of shingles.

        Args:
            shingles: Set of hashed shingle values

        Returns:
            MinHash signature array of shape (num_hash_functions,)

        Explanation:
            For each hash function h_i, compute min(h_i(x) for x in shingles).
            This produces a compact signature that preserves Jaccard similarity:
            Pr[signature_A[i] == signature_B[i]] = Jaccard(A, B).
        """
        signature = np.full(self.num_hashes, self._max_val, dtype=np.uint64)

        for shingle in shingles:
            for i in range(self.num_hashes):
                hash_val = (
                    (self._coefficients_a[i] * shingle + self._coefficients_b[i])
                    % self._prime
                    % self._max_val
                )
                if hash_val < signature[i]:
                    signature[i] = hash_val

        return signature

    def _get_lsh_candidates(self, signatures: List[np.ndarray]) -> Set[Tuple[int, int]]:
        """
        Use LSH banding to find candidate duplicate pairs.

        Args:
            signatures: List of MinHash signatures

        Returns:
            Set of (i, j) pairs where i < j are candidate duplicates

        Explanation:
            Split each signature into b bands of r rows. For each band,
            hash the band values to a bucket. Documents landing in the same
            bucket for any band are candidates. This is the "any match"
            strategy — a single matching band is sufficient.
        """
        # Build buckets
        self.lsh_buckets.clear()
        for doc_idx, sig in enumerate(signatures):
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band = tuple(sig[start:end])
                band_hash = hash(band)
                self.lsh_buckets[band_idx][band_hash].append(doc_idx)

        # Extract candidate pairs
        candidates = set()
        for band_idx in range(self.num_bands):
            for bucket_docs in self.lsh_buckets[band_idx].values():
                if len(bucket_docs) > 1:
                    for i in range(len(bucket_docs)):
                        for j in range(i + 1, len(bucket_docs)):
                            candidates.add((bucket_docs[i], bucket_docs[j]))

        return candidates

    def _compute_jaccard_similarity(self, shingles_a: Set[int], shingles_b: Set[int]) -> float:
        """
        Compute exact Jaccard similarity between two shingle sets.

        Args:
            shingles_a: Shingle set for document A
            shingles_b: Shingle set for document B

        Returns:
            Jaccard similarity in [0, 1]

        Formula:
            J(A, B) = |A ∩ B| / |A ∪ B|
        """
        if not shingles_a and not shingles_b:
            return 1.0
        intersection = len(shingles_a & shingles_b)
        union = len(shingles_a | shingles_b)
        return intersection / union if union > 0 else 0.0

    def deduplicate(self, documents: List[str]) -> List[str]:
        """
        Remove near-duplicate documents using MinHash + LSH.

        Args:
            documents: List of raw text documents

        Returns:
            List of documents with near-duplicates removed

        Explanation:
            1. Compute shingles and MinHash signatures for all documents
            2. Use LSH banding to find candidate pairs (fast)
            3. For candidates, compute exact Jaccard similarity
            4. Mark documents as duplicates if Jaccard > threshold
            5. Keep first occurrence, remove later duplicates

        Example:
            >>> dedup = MinHashDeduplicator()
            >>> docs = ["the cat sat on the mat", "the cat sat on a mat", "completely different text"]
            >>> unique = dedup.deduplicate(docs)
        """
        self.stats["total_documents"] = len(documents)
        logger.info(f"MinHash dedup: Processing {len(documents)} documents...")

        # Step 1: Compute shingles and signatures
        all_shingles = [self._get_shingles(doc) for doc in documents]
        all_signatures = [self._compute_minhash_signature(s) for s in all_shingles]

        # Step 2: LSH candidate generation
        candidates = self._get_lsh_candidates(all_signatures)
        self.stats["candidate_pairs"] = len(candidates)
        logger.info(f"MinHash dedup: {len(candidates)} candidate pairs found")

        # Step 3: Verify candidates with exact Jaccard
        duplicates = set()
        for i, j in candidates:
            if i in duplicates or j in duplicates:
                continue  # Skip if already marked
            jaccard = self._compute_jaccard_similarity(all_shingles[i], all_shingles[j])
            if jaccard >= self.threshold:
                duplicates.add(j)  # Remove later document

        self.stats["duplicates_found"] = len(duplicates)
        self.stats["unique_documents"] = len(documents) - len(duplicates)

        # Step 4: Build result
        unique_documents = [
            doc for idx, doc in enumerate(documents) if idx not in duplicates
        ]

        dup_rate = len(duplicates) / max(len(documents), 1)
        logger.info(
            f"MinHash dedup: {len(duplicates)} near-duplicates removed ({dup_rate:.2%})"
        )
        return unique_documents

    def get_stats(self) -> Dict[str, Any]:
        """Return deduplication statistics."""
        return dict(self.stats)


################################################################################
# SECTION 4: QUALITY FILTERING
################################################################################


class QualityFilter:
    """
    Quality Filter
    ==============

    Document quality scoring and filtering using two approaches:
    1. Heuristic: word count, sentence length, special chars, repetition
    2. Perplexity-based: score documents by perplexity under a reference model

    WHY this matters:
        Raw web data contains massive amounts of low-quality content: spam,
        boilerplate, auto-generated text, gibberish, and excessively
        repetitive content. Training on this noise wastes compute and
        degrades model quality. Quality filtering is the highest-ROI
        step in the curation pipeline.

    Interview Question:
        "Why filter very LOW perplexity documents?"
        Answer: Very low perplexity indicates the document is highly
        predictable — often boilerplate, templates, or repetitive content
        (e.g., terms of service, cookie notices, navigation menus). These
        documents teach the model to produce generic, repetitive text.
        The sweet spot is moderate perplexity: coherent but information-rich.
    """

    def __init__(self, config: Optional[CurationConfig] = None):
        """
        Initialize the quality filter.

        Args:
            config: Curation configuration. Uses defaults if None.
        """
        self.config = config or CurationConfig()
        self.stats = {
            "total_documents": 0,
            "filtered_by_word_count": 0,
            "filtered_by_sentence_length": 0,
            "filtered_by_special_chars": 0,
            "filtered_by_repetition": 0,
            "filtered_by_perplexity": 0,
            "passed_documents": 0,
        }

    def _count_words(self, document: str) -> int:
        """Count words in a document."""
        return len(document.split())

    def _count_sentences(self, document: str) -> int:
        """Count sentences in a document using punctuation heuristics."""
        sentences = re.split(r'[.!?]+', document)
        return max(len([s for s in sentences if s.strip()]), 1)

    def _compute_avg_sentence_length(self, document: str) -> float:
        """Compute average words per sentence."""
        word_count = self._count_words(document)
        sentence_count = self._count_sentences(document)
        return word_count / sentence_count

    def _compute_special_char_ratio(self, document: str) -> float:
        """
        Compute ratio of non-alphanumeric, non-whitespace characters.

        Explanation:
            High ratio of special characters often indicates code fragments,
            markup, or auto-generated content. Normal prose has 5-15% special
            characters (punctuation). Ratios above 30% are suspicious.
        """
        if not document:
            return 0.0
        total = len(document)
        alphanumeric = sum(1 for c in document if c.isalnum() or c.isspace())
        return 1.0 - (alphanumeric / total)

    def _compute_repetition_ratio(self, document: str) -> float:
        """
        Compute ratio of repeated n-grams.

        Explanation:
            Documents with high repetition ratio are often auto-generated,
            scraped boilerplate, or low-quality content. We compute the
            ratio of n-grams that appear more than once to total n-grams.
        """
        words = document.lower().split()
        n = self.config.repetition_ngram_size
        if len(words) < n:
            return 0.0

        ngram_counts: Dict[str, int] = defaultdict(int)
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            ngram_counts[ngram] += 1

        total_ngrams = len(words) - n + 1
        repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)
        return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0.0

    def heuristic_score(self, document: str) -> Tuple[bool, List[str]]:
        """
        Apply heuristic quality checks to a document.

        Args:
            document: Raw text document

        Returns:
            Tuple of (passes_filter, list_of_failure_reasons)

        Explanation:
            Checks multiple heuristic criteria. Returns True if the document
            passes ALL checks. Failure reasons are logged for analysis.
        """
        reasons = []

        # Word count check
        word_count = self._count_words(document)
        if word_count < self.config.min_word_count:
            reasons.append(f"too_short ({word_count} < {self.config.min_word_count})")
        if word_count > self.config.max_word_count:
            reasons.append(f"too_long ({word_count} > {self.config.max_word_count})")

        # Sentence length check
        avg_sent_len = self._compute_avg_sentence_length(document)
        if avg_sent_len < self.config.min_avg_sentence_length:
            reasons.append(
                f"avg_sentence_too_short ({avg_sent_len:.1f} < {self.config.min_avg_sentence_length})"
            )
        if avg_sent_len > self.config.max_avg_sentence_length:
            reasons.append(
                f"avg_sentence_too_long ({avg_sent_len:.1f} > {self.config.max_avg_sentence_length})"
            )

        # Special character ratio
        special_ratio = self._compute_special_char_ratio(document)
        if special_ratio > self.config.max_special_char_ratio:
            reasons.append(
                f"too_many_special_chars ({special_ratio:.2f} > {self.config.max_special_char_ratio})"
            )

        # Repetition ratio
        rep_ratio = self._compute_repetition_ratio(document)
        if rep_ratio > self.config.max_repetition_ratio:
            reasons.append(
                f"too_repetitive ({rep_ratio:.2f} > {self.config.max_repetition_ratio})"
            )

        return (len(reasons) == 0, reasons)

    def perplexity_score(self, log_probs: np.ndarray) -> float:
        """
        Compute perplexity from log probabilities.

        Args:
            log_probs: Array of log probabilities per token

        Returns:
            Perplexity value

        Formula:
            perplexity = exp(-1/N * sum(log_probs))

        Explanation:
            Perplexity measures how "surprised" the model is by the text.
            Low perplexity = predictable (could be templated/repetitive).
            High perplexity = surprising (could be noisy/gibberish).
            Moderate perplexity = coherent, information-rich text.
        """
        if len(log_probs) == 0:
            return float("inf")
        avg_log_prob = np.mean(log_probs)
        return float(np.exp(-avg_log_prob))

    def filter_by_perplexity(
        self,
        perplexities: List[float],
    ) -> List[bool]:
        """
        Filter documents based on perplexity scores.

        Args:
            perplexities: List of perplexity scores per document

        Returns:
            List of booleans (True = keep, False = filter out)

        Explanation:
            Removes documents with perplexity too low (repetitive/templated)
            or too high (noisy/gibberish). The "Goldilocks zone" of moderate
            perplexity contains the most useful training documents.
        """
        results = []
        for ppl in perplexities:
            passes = self.config.min_perplexity <= ppl <= self.config.max_perplexity
            results.append(passes)
        return results

    def filter_documents(
        self,
        documents: List[str],
        perplexities: Optional[List[float]] = None,
    ) -> List[str]:
        """
        Apply full quality filter pipeline to documents.

        Args:
            documents: List of raw text documents
            perplexities: Optional precomputed perplexity scores

        Returns:
            List of documents that pass all quality checks

        Explanation:
            Applies heuristic filters first (fast), then perplexity filter
            if scores provided. Each filtered document's reason is logged.
        """
        self.stats["total_documents"] = len(documents)
        passed_documents = []

        for idx, doc in enumerate(documents):
            # Heuristic checks
            passes, reasons = self.heuristic_score(doc)
            if not passes:
                for reason in reasons:
                    if "too_short" in reason or "too_long" in reason:
                        self.stats["filtered_by_word_count"] += 1
                    elif "sentence" in reason:
                        self.stats["filtered_by_sentence_length"] += 1
                    elif "special_chars" in reason:
                        self.stats["filtered_by_special_chars"] += 1
                    elif "repetitive" in reason:
                        self.stats["filtered_by_repetition"] += 1
                continue

            # Perplexity check
            if perplexities is not None and idx < len(perplexities):
                if not (self.config.min_perplexity <= perplexities[idx] <= self.config.max_perplexity):
                    self.stats["filtered_by_perplexity"] += 1
                    continue

            passed_documents.append(doc)

        self.stats["passed_documents"] = len(passed_documents)
        filter_rate = 1.0 - (len(passed_documents) / max(len(documents), 1))
        logger.info(
            f"Quality filter: {len(passed_documents)}/{len(documents)} passed ({filter_rate:.2%} filtered)"
        )

        # Log per-category filter rates
        total = max(len(documents), 1)
        for category in [
            "filtered_by_word_count",
            "filtered_by_sentence_length",
            "filtered_by_special_chars",
            "filtered_by_repetition",
            "filtered_by_perplexity",
        ]:
            count = self.stats[category]
            if count > 0:
                logger.info(f"  {category}: {count} ({count / total:.2%})")

        return passed_documents

    def get_stats(self) -> Dict[str, Any]:
        """Return filtering statistics."""
        return dict(self.stats)


################################################################################
# SECTION 5: DECONTAMINATION
################################################################################


class Decontaminator:
    """
    Decontaminator
    ==============

    Remove benchmark contamination from training data.

    Algorithm:
        1. Build n-gram index of all eval benchmark questions/answers
        2. For each training document, compute n-gram overlap
        3. Remove documents with overlap > threshold
        4. Log overlap rate per benchmark

    Formula:
        overlap(doc, benchmark) = |ngrams(doc) ∩ ngrams(benchmark)| / |ngrams(benchmark)|
        is_contaminated = overlap > threshold

    CRITICAL: This step is not optional. An uncontaminated eval number
    is the only kind worth reporting.

    WHY this matters:
        Benchmark contamination is the silent killer of evaluation integrity.
        If even a small fraction of training data contains benchmark questions,
        the model may memorize answers, producing scores that don't reflect
        true capability. This has been documented in multiple papers and is
        a primary concern in the 2025-2026 LLM evaluation landscape.

    Interview Question:
        "How do you handle partial matches in decontamination?"
        Answer: We use n-gram overlap (typically 13-grams) rather than exact
        match. This catches cases where benchmark content is reformatted,
        slightly reworded, or embedded in longer documents. The overlap ratio
        is computed against the benchmark's n-grams, so even partial embedding
        is detected. The threshold (e.g., 0.7) balances catching contamination
        vs. avoiding false positives from common phrases.
    """

    def __init__(self, config: Optional[CurationConfig] = None):
        """
        Initialize the decontaminator.

        Args:
            config: Curation configuration. Uses defaults if None.
        """
        self.config = config or CurationConfig()
        self.ngram_size = self.config.decontam_ngram_size
        self.threshold = self.config.decontam_overlap_threshold

        # benchmark_name -> set of n-grams
        self.benchmark_ngrams: Dict[str, Set[str]] = {}

        self.stats = {
            "total_documents": 0,
            "contaminated_documents": 0,
            "overlap_by_benchmark": defaultdict(int),
        }

    def _extract_ngrams(self, text: str, n: int) -> Set[str]:
        """
        Extract word-level n-grams from text.

        Args:
            text: Input text
            n: N-gram size

        Returns:
            Set of n-gram strings
        """
        words = text.lower().split()
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            ngrams.add(ngram)
        return ngrams

    def register_benchmark(self, benchmark_name: str, benchmark_texts: List[str]) -> None:
        """
        Register a benchmark's content for contamination checking.

        Args:
            benchmark_name: Name of the benchmark (e.g., "mmlu", "gsm8k")
            benchmark_texts: List of benchmark questions/answers as text

        Explanation:
            Extracts and stores n-grams from all benchmark content.
            This index is used to check training documents for overlap.
        """
        all_ngrams: Set[str] = set()
        for text in benchmark_texts:
            ngrams = self._extract_ngrams(text, self.ngram_size)
            all_ngrams.update(ngrams)

        self.benchmark_ngrams[benchmark_name] = all_ngrams
        logger.info(
            f"Decontam: Registered {benchmark_name} with {len(all_ngrams)} unique {self.ngram_size}-grams"
        )

    def check_contamination(self, document: str) -> Tuple[bool, Dict[str, float]]:
        """
        Check if a document is contaminated by any registered benchmark.

        Args:
            document: Raw text document

        Returns:
            Tuple of (is_contaminated, overlap_ratios_per_benchmark)

        Explanation:
            Extracts n-grams from the document and computes overlap with
            each benchmark's n-gram set. Returns True if any overlap exceeds
            the threshold.
        """
        doc_ngrams = self._extract_ngrams(document, self.ngram_size)
        if not doc_ngrams:
            return False, {}

        overlap_ratios = {}
        is_contaminated = False

        for bench_name, bench_ngrams in self.benchmark_ngrams.items():
            if not bench_ngrams:
                continue
            intersection = len(doc_ngrams & bench_ngrams)
            ratio = intersection / len(bench_ngrams)
            overlap_ratios[bench_name] = ratio

            if ratio >= self.threshold:
                is_contaminated = True

        return is_contaminated, overlap_ratios

    def decontaminate(self, documents: List[str]) -> List[str]:
        """
        Remove contaminated documents from the corpus.

        Args:
            documents: List of raw text documents

        Returns:
            List of decontaminated documents

        Explanation:
            Iterates through documents, checking each against all registered
            benchmarks. Contaminated documents are removed and their overlap
            statistics are logged.
        """
        self.stats["total_documents"] = len(documents)
        clean_documents = []

        for doc in documents:
            is_contaminated, overlap_ratios = self.check_contamination(doc)
            if is_contaminated:
                self.stats["contaminated_documents"] += 1
                for bench_name, ratio in overlap_ratios.items():
                    if ratio >= self.threshold:
                        self.stats["overlap_by_benchmark"][bench_name] += 1
            else:
                clean_documents.append(doc)

        contam_rate = self.stats["contaminated_documents"] / max(len(documents), 1)
        logger.info(
            f"Decontam: {self.stats['contaminated_documents']}/{len(documents)} "
            f"contaminated ({contam_rate:.2%})"
        )

        for bench_name, count in self.stats["overlap_by_benchmark"].items():
            logger.info(f"  {bench_name}: {count} documents contaminated")

        return clean_documents

    def get_stats(self) -> Dict[str, Any]:
        """Return decontamination statistics."""
        return dict(self.stats)


################################################################################
# SECTION 6: DOMAIN MIXTURE
################################################################################


class DomainMixture:
    """
    Domain Mixture
    ==============

    Explicit mixture ratio management across data domains (web, code,
    math, synthetic, curated).

    WHY this matters:
        Domain ratios directly influence model capabilities. Too much web
        data produces fluent but shallow models. Too much code overfits
        to programming patterns. The mixture must be deliberate, logged,
        and auditable.

    Interview Question:
        "How do you handle domain drift during training?"
        Answer: We track realized token counts per domain at each epoch
        boundary and compare to target ratios. If drift exceeds tolerance
        (e.g., 5%), we log a warning and can adjust sampling weights for
        the next epoch. This is visible in logs, making the mixture
        auditable and reproducible.
    """

    def __init__(self, config: Optional[CurationConfig] = None):
        """
        Initialize the domain mixer.

        Args:
            config: Curation configuration. Uses defaults if None.
        """
        self.config = config or CurationConfig()
        self.target_ratios = self.config.domain_ratios
        self.tolerance = self.config.mixture_tolerance

        # Track realized counts
        self.domain_counts: Dict[str, int] = defaultdict(int)
        self.domain_token_counts: Dict[str, int] = defaultdict(int)

        self.stats = {
            "total_documents": 0,
            "total_tokens": 0,
            "domain_ratios_realized": {},
            "drift_warnings": [],
        }

    def register_domain(self, domain_name: str, documents: List[str], token_counts: List[int]) -> None:
        """
        Register documents for a specific domain.

        Args:
            domain_name: Name of the domain (e.g., "web", "code")
            documents: List of documents in this domain
            token_counts: Token count for each document

        Explanation:
            Stores documents and their token counts for later sampling.
            The domain must be in the configured domain_ratios.
        """
        if domain_name not in self.target_ratios:
            logger.warning(f"Domain '{domain_name}' not in configured ratios. Adding with ratio 0.")
            self.target_ratios[domain_name] = 0.0

        self.domain_counts[domain_name] += len(documents)
        self.domain_token_counts[domain_name] += sum(token_counts)
        logger.info(
            f"Domain mixer: Registered {len(documents)} docs "
            f"({sum(token_counts)} tokens) for '{domain_name}'"
        )

    def compute_realized_ratios(self) -> Dict[str, float]:
        """
        Compute the realized domain ratios based on token counts.

        Returns:
            Dictionary mapping domain names to realized ratios
        """
        total_tokens = sum(self.domain_token_counts.values())
        if total_tokens == 0:
            return {}

        realized = {}
        for domain, tokens in self.domain_token_counts.items():
            realized[domain] = tokens / total_tokens

        self.stats["domain_ratios_realized"] = realized
        self.stats["total_tokens"] = total_tokens
        return realized

    def check_drift(self) -> List[str]:
        """
        Check if realized ratios drift from targets beyond tolerance.

        Returns:
            List of warning messages for domains that drifted

        Explanation:
            Compares realized ratios to target ratios. Logs warnings for
            any domain where |realized - target| > tolerance.
        """
        realized = self.compute_realized_ratios()
        warnings = []

        for domain, target in self.target_ratios.items():
            actual = realized.get(domain, 0.0)
            drift = abs(actual - target)
            if drift > self.tolerance:
                warning = (
                    f"Domain '{domain}' drift: target={target:.2%}, "
                    f"realized={actual:.2%}, drift={drift:.2%}"
                )
                warnings.append(warning)
                logger.warning(warning)

        self.stats["drift_warnings"] = warnings
        return warnings

    def sample_batch(
        self,
        batch_size: int,
        domain_documents: Dict[str, List[str]],
    ) -> List[str]:
        """
        Sample a batch of documents according to target domain ratios.

        Args:
            batch_size: Number of documents in the batch
            domain_documents: Dict mapping domain names to document lists

        Returns:
            List of sampled documents

        Explanation:
            Samples documents from each domain proportional to target ratios.
            If a domain is exhausted, redistributes its share to other domains.
        """
        batch = []
        remaining = batch_size

        for domain, ratio in self.target_ratios.items():
            if domain not in domain_documents:
                continue

            docs = domain_documents[domain]
            n_samples = min(int(batch_size * ratio), len(docs), remaining)

            if n_samples > 0:
                indices = np.random.choice(len(docs), size=n_samples, replace=False)
                batch.extend([docs[i] for i in indices])
                remaining -= n_samples

        # Fill remaining with any available documents
        if remaining > 0:
            all_docs = []
            for docs in domain_documents.values():
                all_docs.extend(docs)
            if all_docs:
                indices = np.random.choice(len(all_docs), size=remaining, replace=True)
                batch.extend([all_docs[i] for i in indices])

        return batch

    def get_stats(self) -> Dict[str, Any]:
        """Return domain mixture statistics."""
        self.compute_realized_ratios()
        return dict(self.stats)


################################################################################
# SECTION 7: DATA CURATOR (ORCHESTRATOR)
################################################################################


class DataCurator:
    """
    Data Curator
    ============

    Orchestrates the full data curation pipeline:
    exact dedup -> near-dedup -> quality filter -> decontamination -> domain mix.

    Pipeline:
        Raw corpus
          -> Exact dedup (hash-based)
          -> Near-dedup (MinHash/LSH)
          -> Quality filter (heuristic + perplexity)
          -> Decontamination (n-gram overlap vs eval benchmarks)
          -> Domain mixer (balance across domains)
          -> Clean corpus

    WHY this matters:
        The curator ensures every training document has been vetted through
        multiple quality gates. The pipeline is deterministic, auditable,
        and each stage has independent statistics for debugging.

    Interview Question:
        "Why run exact dedup before near-dedup?"
        Answer: Exact dedup is O(N) — just hash and compare. Near-dedup
        with MinHash+LSH is more expensive. Running exact dedup first
        reduces the corpus size, making near-dedup faster. Additionally,
        exact duplicates would pollute MinHash signatures and LSH buckets,
        potentially causing false negatives for near-duplicate detection.
    """

    def __init__(self, config: Optional[CurationConfig] = None):
        """
        Initialize the data curator.

        Args:
            config: Curation configuration. Uses defaults if None.
        """
        self.config = config or CurationConfig()

        self.exact_dedup = ExactDeduplicator(self.config)
        self.minhash_dedup = MinHashDeduplicator(self.config)
        self.quality_filter = QualityFilter(self.config)
        self.decontaminator = Decontaminator(self.config)
        self.domain_mixer = DomainMixture(self.config)

        self.pipeline_stats: List[Dict[str, Any]] = []

    def register_benchmarks(self, benchmark_data: Dict[str, List[str]]) -> None:
        """
        Register benchmark data for decontamination.

        Args:
            benchmark_data: Dict mapping benchmark names to lists of text
        """
        for name, texts in benchmark_data.items():
            self.decontaminator.register_benchmark(name, texts)

    def curate(
        self,
        documents: List[str],
        perplexities: Optional[List[float]] = None,
        enable_near_dedup: bool = True,
    ) -> List[str]:
        """
        Run the full curation pipeline.

        Args:
            documents: List of raw text documents
            perplexities: Optional precomputed perplexity scores
            enable_near_dedup: Whether to run near-deduplication (can be slow)

        Returns:
            List of curated, clean documents

        Explanation:
            Runs each stage sequentially, logging statistics at each step.
            Each stage reduces the corpus size, and the final result is
            a high-quality subset suitable for training.
        """
        logger.info(f"Starting curation pipeline with {len(documents)} documents")

        # Stage 1: Exact deduplication
        logger.info("=" * 60)
        logger.info("STAGE 1: Exact Deduplication")
        documents = self.exact_dedup.deduplicate(documents)
        self.pipeline_stats.append({"stage": "exact_dedup", **self.exact_dedup.get_stats()})

        # Stage 2: Near-deduplication
        if enable_near_dedup:
            logger.info("=" * 60)
            logger.info("STAGE 2: Near-Deduplication (MinHash/LSH)")
            documents = self.minhash_dedup.deduplicate(documents)
            self.pipeline_stats.append({"stage": "near_dedup", **self.minhash_dedup.get_stats()})

        # Stage 3: Quality filtering
        logger.info("=" * 60)
        logger.info("STAGE 3: Quality Filtering")
        documents = self.quality_filter.filter_documents(documents, perplexities)
        self.pipeline_stats.append({"stage": "quality_filter", **self.quality_filter.get_stats()})

        # Stage 4: Decontamination
        logger.info("=" * 60)
        logger.info("STAGE 4: Decontamination")
        documents = self.decontaminator.decontaminate(documents)
        self.pipeline_stats.append({"stage": "decontamination", **self.decontaminator.get_stats()})

        logger.info("=" * 60)
        logger.info(f"Curation complete: {len(documents)} clean documents")

        return documents

    def get_pipeline_stats(self) -> List[Dict[str, Any]]:
        """Return statistics for each pipeline stage."""
        return list(self.pipeline_stats)


################################################################################
# SECTION 8: TESTING & DEMONSTRATION
################################################################################


def demonstrate_curator():
    """Demonstrate the data curation pipeline."""
    print("=" * 70)
    print("DATA CURATION PIPELINE DEMONSTRATION")
    print("=" * 70)

    # Create sample documents
    documents = [
        # Good quality documents
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow parallel processing of sequences. Unlike recurrent neural networks, transformers can capture long-range dependencies without sequential computation.",

        "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations. A quantum bit or qubit can exist in multiple states simultaneously, enabling quantum computers to explore many solutions in parallel.",

        # Near-duplicate (minor edit)
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow parallel processing of sequences. Unlike recurrent neural networks, transformers can capture long-range dependencies without sequential processing.",

        # Exact duplicate
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow parallel processing of sequences. Unlike recurrent neural networks, transformers can capture long-range dependencies without sequential computation.",

        # Low quality (too short)
        "Hello world",

        # Low quality (too repetitive)
        "Buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now",

        # Code document
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# This is a simple recursive implementation\n# Time complexity: O(2^n)\n# Space complexity: O(n)",

        # Math document
        "The quadratic formula states that for any quadratic equation ax^2 + bx + c = 0, the solutions are given by x = (-b ± sqrt(b^2 - 4ac)) / (2a). The discriminant b^2 - 4ac determines the nature of the roots.",

        # Potentially contaminated document (if benchmark registered)
        "What is the capital of France? The capital of France is Paris. Paris is located in the north-central part of the country on the Seine River.",
    ]

    print(f"\nInitial corpus: {len(documents)} documents\n")

    # Initialize curator
    config = CurationConfig(
        min_word_count=10,  # Lower threshold for demo
        jaccard_threshold=0.7,
    )
    curator = DataCurator(config)

    # Run curation pipeline
    clean_documents = curator.curate(documents, enable_near_dedup=True)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Input documents:  {len(documents)}")
    print(f"Output documents: {len(clean_documents)}")
    print(f"Removed:          {len(documents) - len(clean_documents)}")

    # Print pipeline statistics
    print("\nPipeline Statistics:")
    for stage_stats in curator.get_pipeline_stats():
        stage = stage_stats.pop("stage")
        print(f"\n  {stage}:")
        for key, value in stage_stats.items():
            if key != "overlap_by_benchmark":
                print(f"    {key}: {value}")

    # Demonstrate domain mixture
    print("\n" + "=" * 60)
    print("DOMAIN MIXTURE DEMONSTRATION")
    print("=" * 60)

    domain_mixer = DomainMixture(config)
    domain_mixer.register_domain("web", clean_documents[:3], [100, 120, 90])
    domain_mixer.register_domain("code", [clean_documents[3]], [80])
    domain_mixer.register_domain("math", [clean_documents[4]], [60])

    warnings = domain_mixer.check_drift()
    if warnings:
        print("\nDrift warnings:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\nNo significant domain drift detected.")

    print("\nRealized ratios:")
    for domain, ratio in domain_mixer.compute_realized_ratios().items():
        target = config.domain_ratios.get(domain, 0.0)
        print(f"  {domain}: target={target:.2%}, realized={ratio:.2%}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_curator()
