"""
################################################################################
SYNTHETIC DATA GENERATION — Multi-Teacher Generation with Diversity Checks
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Synthetic Data Generation?
    Synthetic data generation creates training examples using existing
    language models (teachers) rather than collecting data from the real
    world. This is used to augment scarce domains (math reasoning, code,
    safety examples) and to scale training data beyond what web scraping
    provides.

Why does it matter?
    High-quality real data is finite. Web data is noisy and unevenly
    distributed across domains. Synthetic generation can fill gaps:
    - Math reasoning: generate step-by-step solutions
    - Code: generate diverse programming challenges and solutions
    - Safety: generate adversarial prompts and safe responses
    - Knowledge: generate Q&A pairs from reference texts

    However, 2024-2026 research showed that naive synthetic generation
    causes model collapse — models trained on their own output degrade
    over generations. The key mitigations are:
    1. Mix multiple teacher models (avoid single-teacher style bias)
    2. Always seed with real data (never purely from model outputs)
    3. Diversity check: detect and reject narrow/biased batches

How does it work?
    1. Real Data Seeder: Select diverse prompts from real data
    2. Multi-Teacher Generation: Each prompt is processed by multiple
       teacher models to produce diverse outputs
    3. Quality Scoring: Each generated example is scored for quality
    4. Diversity Check: Embedding-space clustering detects if a batch
       is too narrow in topic or style
    5. Mixing: Synthetic data is mixed with real data at configured ratios

ARCHITECTURE DIAGRAM (ASCII art):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Real Data Corpus                                                 │
    │        ↓                                                          │
    │ ┌──────────────────────────────────────────────────────────────┐ │
    │ │ Real Data Seeder — selects diverse prompts                   │ │
    │ └──────────────────────────────────────────────────────────────┘ │
    │        ↓                                                          │
    │ ┌──────────────────────────────────────────────────────────────┐ │
    │ │ Multi-Teacher Generator — generates outputs from N teachers   │ │
    │ └──────────────────────────────────────────────────────────────┘ │
    │        ↓                                                          │
    │ ┌──────────────────────────────────────────────────────────────┐ │
    │ │ Quality Scorer — filters low-quality generations              │ │
    │ └──────────────────────────────────────────────────────────────┘ │
    │        ↓                                                          │
    │ ┌──────────────────────────────────────────────────────────────┐ │
    │ │ Diversity Checker — embedding clustering, rejects narrow      │ │
    │ └──────────────────────────────────────────────────────────────┘ │
    │        ↓                                                          │
    │ Synthetic Data (mixed with real data for training)               │
    └─────────────────────────────────────────────────────────────────┘

HISTORICAL CONTEXT:
    - 2022: Self-Instruct (Wang et al.) — bootstrap instructions from LLMs
    - 2023: Phi-1 — "Textbooks Are All You Need" — synthetic textbooks
    - 2023: Gunasekar et al. show synthetic code data can train strong coders
    - 2024: Model collapse paper (Shumailov et al.) — risks of self-training
    - 2024: LIMA — "Less Is More" — 1000 curated examples beat millions
    - 2025: Multi-teacher distillation becomes standard practice
    - 2026: Diversity-aware synthetic generation is essential for avoiding
      collapse; single-teacher pipelines are considered risky

INTERVIEW QUESTIONS:
    1. "What is model collapse and how do you prevent it?"
       Answer: Model collapse occurs when a model is trained on its own
       output (or output from a similar model). Over generations, the
       distribution narrows — rare patterns are lost, diversity decreases,
       and the model converges to a degenerate distribution. Prevention:
       (a) always mix real data, (b) use multiple diverse teachers,
       (c) check diversity of generated batches, (d) limit self-training
       rounds.

    2. "How do you measure diversity in synthetic data?"
       Answer: Embed documents into a vector space (using a sentence
       transformer), then compute clustering metrics: number of clusters,
       average inter-cluster distance, or coverage of the embedding space.
       If a batch clusters tightly (low inter-cluster distance), it lacks
       diversity and should be regenerated with different prompts or
       temperature settings.

    3. "Why use multiple teacher models instead of one?"
       Answer: Each model has a distinct "style" — vocabulary, reasoning
       patterns, explanation depth. A single teacher produces data that
       mirrors its biases, which the student model then amplifies. Multiple
       teachers provide stylistic and perspective diversity, producing a
       student that generalizes better. Additionally, if one teacher has a
       systematic error, other teachers provide correct examples.

################################################################################
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


################################################################################
# SECTION 1: CONFIGURATION
################################################################################


@dataclass
class SyntheticDataConfig:
    """
    Configuration for synthetic data generation.
    ============================================

    All parameters are explicit config fields. No magic numbers.
    """

    # Teacher models
    teacher_models: List[str] = field(default_factory=lambda: [
        "teacher_a",  # Identifier for teacher model A
        "teacher_b",  # Identifier for teacher model B
        "teacher_c",  # Identifier for teacher model C
    ])

    # Generation parameters
    generation_temperature: float = 0.8  # Sampling temperature
    generation_top_p: float = 0.95  # Nucleus sampling threshold
    max_generation_length: int = 2048  # Maximum tokens per generation
    num_generations_per_prompt: int = 3  # How many outputs per prompt

    # Quality scoring
    min_quality_score: float = 0.5  # Minimum quality score to keep
    quality_score_weights: Dict[str, float] = field(default_factory=lambda: {
        "length": 0.2,
        "coherence": 0.3,
        "relevance": 0.3,
        "diversity": 0.2,
    })

    # Diversity checking
    diversity_threshold: float = 0.3  # Min average inter-cluster distance
    min_clusters: int = 5  # Minimum number of clusters in a batch
    embedding_dim: int = 384  # Dimension of embedding vectors
    batch_diversity_check_size: int = 100  # Check diversity every N examples

    # Real data seeding
    seed_ratio: float = 0.3  # Fraction of prompts from real data
    min_seed_examples: int = 10  # Minimum real data seeds per batch

    # Mixing
    max_synthetic_ratio: float = 0.5  # Max fraction of synthetic in training
    teacher_weight_decay: float = 0.95  # Weight decay for teacher contributions


################################################################################
# SECTION 2: REAL DATA SEEDER
################################################################################


class RealDataSeeder:
    """
    Real Data Seeder
    ================

    Seeds generation prompts from real data slices to ensure synthetic
    generation is grounded in real-world content.

    WHY this matters:
        Generating synthetic data from scratch (without real data seeds)
        produces content that drifts from real-world distribution. Real
        data seeds ensure the generated content covers topics, styles,
        and complexity levels that actually appear in real corpora.

    Interview Question:
        "How do you select seeds from real data?"
        Answer: We use diversity-aware sampling: embed real documents,
        cluster them, and sample proportionally from each cluster. This
        ensures the seeds cover the full topic distribution, not just
        the most common topics. We also stratify by domain (web, code,
        math) to ensure coverage across data types.
    """

    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        """
        Initialize the real data seeder.

        Args:
            config: Synthetic data configuration
        """
        self.config = config or SyntheticDataConfig()
        self.seed_pool: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_seed_pool(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        Build the seed pool from real data.

        Args:
            documents: List of real documents
            embeddings: Precomputed embeddings for documents (N x D)

        Explanation:
            Stores documents and their embeddings for diversity-aware
            sampling. If embeddings not provided, uses random projections
            as a simple approximation.
        """
        self.seed_pool = documents

        if embeddings is not None:
            self.embeddings = embeddings
        else:
            # Simple random projection as embedding approximation
            rng = np.random.RandomState(42)
            random_matrix = rng.randn(len(documents[0]) if documents else 100, self.config.embedding_dim)
            # Use character-level features as simple embedding
            char_features = np.zeros((len(documents), 256))
            for i, doc in enumerate(documents):
                for char in doc[:1000]:
                    char_features[i, ord(char) % 256] += 1
            # Normalize
            norms = np.linalg.norm(char_features, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            char_features = char_features / norms
            self.embeddings = char_features

        logger.info(f"Seed pool built with {len(documents)} documents")

    def sample_seeds(
        self,
        n_seeds: int,
        diversity_aware: bool = True,
    ) -> List[str]:
        """
        Sample seed prompts from the real data pool.

        Args:
            n_seeds: Number of seeds to sample
            diversity_aware: Whether to use diversity-aware sampling

        Returns:
            List of seed documents

        Explanation:
            If diversity_aware is True, uses clustering to ensure sampled
            seeds cover diverse topics. Otherwise, samples uniformly.
        """
        if not self.seed_pool:
            logger.warning("Seed pool is empty!")
            return []

        n_seeds = min(n_seeds, len(self.seed_pool))

        if not diversity_aware or self.embeddings is None:
            indices = np.random.choice(len(self.seed_pool), size=n_seeds, replace=False)
            return [self.seed_pool[i] for i in indices]

        # Diversity-aware sampling via k-means-like clustering
        embeddings = self.embeddings
        n_clusters = min(n_seeds, len(embeddings))

        # Simple k-means clustering
        centroids = self._kmeans_cluster(embeddings, n_clusters)

        # Sample one from each cluster (closest to centroid)
        selected = []
        used = set()
        for centroid in centroids:
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            sorted_indices = np.argsort(distances)
            for idx in sorted_indices:
                if idx not in used:
                    selected.append(idx)
                    used.add(idx)
                    break

        return [self.seed_pool[i] for i in selected[:n_seeds]]

    def _kmeans_cluster(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Simple k-means clustering.

        Args:
            embeddings: Embedding matrix (N x D)
            n_clusters: Number of clusters

        Returns:
            Centroid matrix (K x D)
        """
        n_samples = len(embeddings)
        # Initialize centroids randomly
        indices = np.random.choice(n_samples, size=n_clusters, replace=False)
        centroids = embeddings[indices].copy()

        for _ in range(10):  # 10 iterations
            # Assign points to nearest centroid
            distances = np.linalg.norm(
                embeddings[:, np.newaxis] - centroids[np.newaxis], axis=2
            )
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            for k in range(n_clusters):
                mask = assignments == k
                if mask.any():
                    centroids[k] = embeddings[mask].mean(axis=0)

        return centroids


################################################################################
# SECTION 3: MULTI-TEACHER GENERATOR
################################################################################


class MultiTeacherGenerator:
    """
    Multi-Teacher Generator
    =======================

    Generates synthetic data using multiple teacher models to avoid
    single-teacher style bias.

    WHY this matters:
        A single teacher produces data that mirrors its biases — vocabulary
        preferences, reasoning patterns, explanation depth. The student
        model then amplifies these biases. Multiple teachers provide
        stylistic and perspective diversity, producing a student that
        generalizes better.

    Interview Question:
        "How do you combine outputs from multiple teachers?"
        Answer: Each teacher processes the same prompt independently. We
        score each output for quality and keep the best. Alternatively,
        we can weight teachers by their historical quality scores and
        sample proportionally. The key is that teachers should be
        architecturally diverse (e.g., different families, different
        training data) to maximize output diversity.
    """

    def __init__(
        self,
        config: Optional[SyntheticDataConfig] = None,
        teacher_callables: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize the multi-teacher generator.

        Args:
            config: Synthetic data configuration
            teacher_callables: Dict mapping teacher names to generation functions.
                             Each function takes (prompt, **kwargs) -> str.
                             If None, uses placeholder functions for demonstration.
        """
        self.config = config or SyntheticDataConfig()
        self.teacher_weights: Dict[str, float] = {
            model: 1.0 for model in self.config.teacher_models
        }

        # Use provided callables or placeholder
        if teacher_callables is not None:
            self.teacher_callables = teacher_callables
        else:
            # Placeholder for demonstration
            self.teacher_callables = {
                model: self._placeholder_generate for model in self.config.teacher_models
            }

        self.stats = {
            "total_prompts": 0,
            "total_generations": 0,
            "generations_per_teacher": defaultdict(int),
        }

    def _placeholder_generate(self, prompt: str, **kwargs) -> str:
        """
        Placeholder generation function for demonstration.

        In production, this would be replaced by actual model API calls.
        """
        temperature = kwargs.get("temperature", self.config.generation_temperature)
        # Simulate generation by echoing prompt with variation
        return f"[Generated by teacher at temp={temperature:.2f}] Response to: {prompt[:50]}..."

    def generate_single(
        self,
        prompt: str,
        teacher_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate a single response using one teacher.

        Args:
            prompt: Input prompt
            teacher_name: Specific teacher to use (random if None)
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if teacher_name is None:
            # Select teacher proportionally to weights
            teachers = list(self.teacher_weights.keys())
            weights = np.array([self.teacher_weights[t] for t in teachers])
            weights = weights / weights.sum()
            teacher_name = np.random.choice(teachers, p=weights)

        generate_fn = self.teacher_callables[teacher_name]
        output = generate_fn(
            prompt,
            temperature=kwargs.get("temperature", self.config.generation_temperature),
            top_p=kwargs.get("top_p", self.config.generation_top_p),
            max_length=kwargs.get("max_length", self.config.max_generation_length),
        )

        self.stats["total_generations"] += 1
        self.stats["generations_per_teacher"][teacher_name] += 1
        return output

    def generate_batch(
        self,
        prompts: List[str],
        num_per_prompt: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple responses for each prompt using multiple teachers.

        Args:
            prompts: List of input prompts
            num_per_prompt: Number of generations per prompt (default from config)

        Returns:
            List of dicts with keys: prompt, teacher, output, quality_score

        Explanation:
            For each prompt, generates num_per_prompt outputs using different
            teachers. This ensures each prompt produces diverse outputs from
            multiple perspectives.
        """
        if num_per_prompt is None:
            num_per_prompt = self.config.num_generations_per_prompt

        results = []
        self.stats["total_prompts"] += len(prompts)

        for prompt in prompts:
            teachers = list(self.teacher_weights.keys())
            for i in range(num_per_prompt):
                # Cycle through teachers
                teacher = teachers[i % len(teachers)]
                output = self.generate_single(prompt, teacher_name=teacher)
                results.append({
                    "prompt": prompt,
                    "teacher": teacher,
                    "output": output,
                    "quality_score": None,  # To be filled by quality scorer
                })

        logger.info(
            f"Generated {len(results)} outputs from {len(prompts)} prompts "
            f"using {len(teachers)} teachers"
        )
        return results

    def update_teacher_weight(self, teacher_name: str, quality_score: float) -> None:
        """
        Update a teacher's weight based on output quality.

        Args:
            teacher_name: Name of the teacher
            quality_score: Average quality score of recent outputs [0, 1]

        Explanation:
            Teachers that produce higher quality outputs get higher weights,
            meaning they're selected more often for future generations.
            Uses exponential moving average for smooth updates.
        """
        if teacher_name not in self.teacher_weights:
            logger.warning(f"Unknown teacher: {teacher_name}")
            return

        decay = self.config.teacher_weight_decay
        self.teacher_weights[teacher_name] = (
            decay * self.teacher_weights[teacher_name] + (1 - decay) * quality_score
        )

        # Normalize weights
        total = sum(self.teacher_weights.values())
        for t in self.teacher_weights:
            self.teacher_weights[t] /= total

    def get_stats(self) -> Dict[str, Any]:
        """Return generation statistics."""
        return dict(self.stats)


################################################################################
# SECTION 4: DIVERSITY CHECKER
################################################################################


class DiversityChecker:
    """
    Diversity Checker
    =================

    Embedding-space clustering to detect style/topic collapse in
    synthetic data batches.

    WHY this matters:
        Synthetic data can converge to narrow topics or styles, especially
        when generated by a single teacher or from similar prompts. This
        "mode collapse" produces training data that teaches the model only
        a subset of language patterns. The diversity checker detects this
        early, before the data enters the training pipeline.

    Interview Question:
        "How does synthetic data diversity affect model training?"
        Answer: Low-diversity synthetic data creates a feedback loop:
        the model learns a narrow distribution, generates narrow data
        when used as a teacher, which trains an even narrower model.
        This is the core mechanism of model collapse. The diversity
        checker breaks this loop by rejecting batches that don't cover
        sufficient breadth in embedding space.
    """

    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        """
        Initialize the diversity checker.

        Args:
            config: Synthetic data configuration
        """
        self.config = config or SyntheticDataConfig()
        self.stats = {
            "batches_checked": 0,
            "batches_rejected": 0,
            "average_diversity_score": 0.0,
        }

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts.

        Args:
            texts: List of text documents

        Returns:
            Embedding matrix (N x D)

        Explanation:
            In production, this would use a sentence transformer (e.g.,
            all-MiniLM-L6-v2). For demonstration, we use a simple
            character-frequency-based embedding.
        """
        n = len(texts)
        dim = self.config.embedding_dim

        # Simple character-frequency embedding for demonstration
        embeddings = np.zeros((n, 256))
        for i, text in enumerate(texts):
            for char in text.lower()[:5000]:
                if ord(char) < 256:
                    embeddings[i, ord(char)] += 1

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        return embeddings

    def compute_diversity_score(self, embeddings: np.ndarray) -> Tuple[float, int]:
        """
        Compute diversity score for a batch of embeddings.

        Args:
            embeddings: Embedding matrix (N x D)

        Returns:
            Tuple of (diversity_score, num_clusters)

        Explanation:
            1. Cluster embeddings using k-means
            2. Compute average inter-cluster distance
            3. Higher distance = more diverse
            4. Also checks number of clusters (min_clusters threshold)
        """
        n = len(embeddings)
        if n < 2:
            return 0.0, 0

        # Determine number of clusters (sqrt(N) heuristic, capped)
        n_clusters = min(int(np.sqrt(n)), self.config.min_clusters * 2, n)
        n_clusters = max(n_clusters, 2)

        # Simple k-means
        centroids = self._kmeans(embeddings, n_clusters)

        # Compute inter-cluster distances
        if n_clusters < 2:
            return 0.0, n_clusters

        distances = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                distances.append(dist)

        avg_distance = np.mean(distances) if distances else 0.0

        # Count effective clusters (with >1 member)
        assignments = self._assign_clusters(embeddings, centroids)
        effective_clusters = sum(
            1 for k in range(n_clusters) if (assignments == k).sum() > 1
        )

        return float(avg_distance), effective_clusters

    def _kmeans(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Simple k-means implementation."""
        n = len(embeddings)
        indices = np.random.choice(n, size=min(n_clusters, n), replace=False)
        centroids = embeddings[indices].copy()

        for _ in range(10):
            assignments = self._assign_clusters(embeddings, centroids)
            for k in range(n_clusters):
                mask = assignments == k
                if mask.any():
                    centroids[k] = embeddings[mask].mean(axis=0)

        return centroids

    def _assign_clusters(self, embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each embedding to nearest centroid."""
        distances = np.linalg.norm(
            embeddings[:, np.newaxis] - centroids[np.newaxis], axis=2
        )
        return np.argmin(distances, axis=1)

    def check_batch(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a batch of generated texts has sufficient diversity.

        Args:
            texts: List of generated texts
            embeddings: Optional precomputed embeddings

        Returns:
            Tuple of (passes_diversity_check, stats_dict)

        Explanation:
            Computes diversity score and checks against thresholds.
            Returns True if the batch passes both the diversity score
            threshold and the minimum clusters threshold.
        """
        self.stats["batches_checked"] += 1

        if embeddings is None:
            embeddings = self.compute_embeddings(texts)

        diversity_score, num_clusters = self.compute_diversity_score(embeddings)

        passes = (
            diversity_score >= self.config.diversity_threshold
            and num_clusters >= self.config.min_clusters
        )

        if not passes:
            self.stats["batches_rejected"] += 1

        # Update running average
        n = self.stats["batches_checked"]
        old_avg = self.stats["average_diversity_score"]
        self.stats["average_diversity_score"] = old_avg + (diversity_score - old_avg) / n

        batch_stats = {
            "diversity_score": diversity_score,
            "num_clusters": num_clusters,
            "passes": passes,
            "threshold": self.config.diversity_threshold,
            "min_clusters": self.config.min_clusters,
        }

        if not passes:
            logger.warning(
                f"Batch failed diversity check: score={diversity_score:.3f} "
                f"(threshold={self.config.diversity_threshold}), "
                f"clusters={num_clusters} (min={self.config.min_clusters})"
            )

        return passes, batch_stats

    def get_stats(self) -> Dict[str, Any]:
        """Return diversity checking statistics."""
        return dict(self.stats)


################################################################################
# SECTION 5: QUALITY SCORER
################################################################################


class QualityScorer:
    """
    Quality Scorer
    ==============

    Scores synthetic data quality along multiple dimensions:
    length, coherence, relevance, diversity.

    WHY this matters:
        Not all synthetic data is equally useful. Some generations are
        too short, incoherent, off-topic, or redundant. Quality scoring
        filters out low-value examples before they enter training.

    Interview Question:
        "How do you score synthetic data quality without human labels?"
        Answer: We use automated heuristics: (1) length score penalizes
        too-short/too-long outputs, (2) coherence score checks for
        logical flow (sentence-level similarity), (3) relevance score
        measures semantic similarity to the prompt, (4) diversity score
        checks if the output adds new information. These heuristics are
        imperfect but scalable — human labels are reserved for final
        validation.
    """

    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        """
        Initialize the quality scorer.

        Args:
            config: Synthetic data configuration
        """
        self.config = config or SyntheticDataConfig()
        self.weights = self.config.quality_score_weights

    def score_length(self, text: str) -> float:
        """
        Score based on text length.

        Args:
            text: Generated text

        Returns:
            Score in [0, 1]

        Explanation:
            Penalizes texts that are too short (<50 words) or too long
            (>2000 words). Optimal length is 100-500 words.
        """
        word_count = len(text.split())
        if word_count < 50:
            return word_count / 50.0
        if word_count > 2000:
            return max(0.0, 1.0 - (word_count - 2000) / 2000.0)
        return 1.0

    def score_coherence(self, text: str) -> float:
        """
        Score text coherence based on sentence-level similarity.

        Args:
            text: Generated text

        Returns:
            Score in [0, 1]

        Explanation:
            Splits text into sentences, computes pairwise similarity
            between consecutive sentences. Moderate similarity indicates
            coherent flow; very low similarity indicates incoherence.
        """
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
        if len(sentences) < 2:
            return 0.5  # Too few sentences to assess

        # Simple character-overlap similarity between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            chars_a = set(sentences[i].lower())
            chars_b = set(sentences[i + 1].lower())
            if chars_a and chars_b:
                sim = len(chars_a & chars_b) / len(chars_a | chars_b)
                similarities.append(sim)

        if not similarities:
            return 0.5

        avg_sim = np.mean(similarities)
        # Moderate similarity is best (0.2-0.6)
        if 0.2 <= avg_sim <= 0.6:
            return 1.0
        elif avg_sim < 0.2:
            return avg_sim / 0.2
        else:
            return max(0.0, 1.0 - (avg_sim - 0.6) / 0.4)

    def score_relevance(self, prompt: str, output: str) -> float:
        """
        Score relevance of output to prompt.

        Args:
            prompt: Input prompt
            output: Generated output

        Returns:
            Score in [0, 1]

        Explanation:
            Measures word overlap between prompt and output. Higher
            overlap indicates the output is on-topic.
        """
        prompt_words = set(prompt.lower().split())
        output_words = set(output.lower().split())

        if not prompt_words:
            return 0.5

        overlap = len(prompt_words & output_words)
        return min(1.0, overlap / len(prompt_words))

    def score(self, prompt: str, output: str) -> float:
        """
        Compute overall quality score.

        Args:
            prompt: Input prompt
            output: Generated output

        Returns:
            Weighted quality score in [0, 1]
        """
        scores = {
            "length": self.score_length(output),
            "coherence": self.score_coherence(output),
            "relevance": self.score_relevance(prompt, output),
            "diversity": 0.7,  # Placeholder — would use embedding diversity
        }

        weighted_score = sum(
            scores[dim] * weight for dim, weight in self.weights.items()
        )
        return weighted_score

    def filter_by_quality(
        self,
        generations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Filter generations by quality score.

        Args:
            generations: List of generation dicts with 'prompt' and 'output'

        Returns:
            Filtered list with quality_score field filled
        """
        scored = []
        for gen in generations:
            quality = self.score(gen["prompt"], gen["output"])
            gen["quality_score"] = quality
            if quality >= self.config.min_quality_score:
                scored.append(gen)

        logger.info(
            f"Quality filter: {len(scored)}/{len(generations)} passed "
            f"(threshold={self.config.min_quality_score})"
        )
        return scored


################################################################################
# SECTION 6: SYNTHETIC DATA GENERATOR (ORCHESTRATOR)
################################################################################


class SyntheticDataGenerator:
    """
    Synthetic Data Generator
    ========================

    Orchestrates multi-teacher synthetic data generation with quality
    filtering and diversity checking.

    Pipeline:
        1. Seed prompts from real data
        2. Generate outputs from multiple teachers
        3. Score quality of each generation
        4. Check batch diversity
        5. Mix synthetic with real data

    WHY this matters:
        Naive synthetic generation causes model collapse. This generator
        implements the 2026 best practices:
        - Multiple diverse teachers
        - Real data seeding
        - Quality filtering
        - Diversity checking
        - Controlled mixing ratios

    Interview Question:
        "What is the maximum safe ratio of synthetic to real data?"
        Answer: Research suggests synthetic data should not exceed 50% of
        training data, with 20-30% being a safer range. The exact ratio
        depends on: (1) diversity of teachers, (2) quality of the synthetic
        data, (3) how well the diversity checker works. The key principle:
        synthetic data should augment, not replace, real data. We always
        track the ratio and log warnings if it drifts above threshold.
    """

    def __init__(
        self,
        config: Optional[SyntheticDataConfig] = None,
        teacher_callables: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize the synthetic data generator.

        Args:
            config: Synthetic data configuration
            teacher_callables: Dict mapping teacher names to generation functions
        """
        self.config = config or SyntheticDataConfig()

        self.seeder = RealDataSeeder(self.config)
        self.generator = MultiTeacherGenerator(self.config, teacher_callables)
        self.quality_scorer = QualityScorer(self.config)
        self.diversity_checker = DiversityChecker(self.config)

        self.stats = {
            "total_generated": 0,
            "total_kept": 0,
            "batches_processed": 0,
            "batches_rejected_diversity": 0,
        }

    def generate_dataset(
        self,
        real_documents: List[str],
        num_prompts: int = 100,
        max_retries: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate a complete synthetic dataset.

        Args:
            real_documents: Real data for seeding
            num_prompts: Number of prompts to generate from
            max_retries: Max retries for diversity-failed batches

        Returns:
            List of high-quality, diverse synthetic examples

        Explanation:
            Full pipeline: seed from real data, generate from multiple
            teachers, filter by quality, check diversity, retry if needed.
        """
        # Step 1: Build seed pool
        self.seeder.build_seed_pool(real_documents)

        # Step 2: Sample seeds
        n_seeds = max(self.config.min_seed_examples, int(num_prompts * self.config.seed_ratio))
        seeds = self.seeder.sample_seeds(n_seeds, diversity_aware=True)

        # Step 3: Generate with multiple teachers
        logger.info(f"Generating from {len(seeds)} seed prompts...")
        all_generations = self.generator.generate_batch(seeds)
        self.stats["total_generated"] = len(all_generations)

        # Step 4: Quality filter
        quality_generations = self.quality_scorer.filter_by_quality(all_generations)

        # Step 5: Diversity check with retries
        final_generations = []
        batch_size = self.config.batch_diversity_check_size

        for i in range(0, len(quality_generations), batch_size):
            batch = quality_generations[i : i + batch_size]
            outputs = [g["output"] for g in batch]

            for retry in range(max_retries):
                passes, div_stats = self.diversity_checker.check_batch(outputs)
                if passes:
                    final_generations.extend(batch)
                    break
                else:
                    logger.info(f"Diversity retry {retry + 1}/{max_retries}")
                    # Re-generate batch with higher temperature
                    batch_prompts = list(set(g["prompt"] for g in batch))
                    new_gens = self.generator.generate_batch(
                        batch_prompts,
                        num_per_prompt=self.config.num_generations_per_prompt,
                    )
                    new_gens = self.quality_scorer.filter_by_quality(new_gens)
                    outputs = [g["output"] for g in new_gens]
                    batch = new_gens
            else:
                self.stats["batches_rejected_diversity"] += 1

            self.stats["batches_processed"] += 1

        self.stats["total_kept"] = len(final_generations)

        logger.info(
            f"Synthetic generation complete: {self.stats['total_kept']}/{self.stats['total_generated']} "
            f"kept ({self.stats['batches_rejected_diversity']} batches rejected for low diversity)"
        )

        return final_generations

    def get_stats(self) -> Dict[str, Any]:
        """Return generation statistics."""
        return {
            **self.stats,
            "generator": self.generator.get_stats(),
            "diversity": self.diversity_checker.get_stats(),
        }


################################################################################
# SECTION 7: TESTING & DEMONSTRATION
################################################################################


def demonstrate_synthetic():
    """Demonstrate the synthetic data generation pipeline."""
    print("=" * 70)
    print("SYNTHETIC DATA GENERATION DEMONSTRATION")
    print("=" * 70)

    # Real data seeds
    real_documents = [
        "The transformer architecture uses self-attention to process sequences in parallel.",
        "Gradient descent optimizes neural network parameters by following the loss gradient.",
        "Python is a high-level programming language known for its readability.",
        "The quadratic formula solves second-degree polynomial equations.",
        "Neural networks learn representations through backpropagation.",
        "Convolutional neural networks are designed for image processing tasks.",
        "Reinforcement learning trains agents through reward signals.",
        "Transfer learning reuses pretrained models for new tasks.",
        "Attention mechanisms allow models to focus on relevant input parts.",
        "Embedding layers convert discrete tokens to continuous vectors.",
        "Batch normalization stabilizes training by normalizing layer inputs.",
        "Dropout prevents overfitting by randomly zeroing activations.",
        "The softmax function converts logits to probability distributions.",
        "Cross-entropy loss measures the difference between predicted and true distributions.",
        "Learning rate schedules adjust the step size during training.",
    ]

    # Configure
    config = SyntheticDataConfig(
        teacher_models=["teacher_alpha", "teacher_beta", "teacher_gamma"],
        num_generations_per_prompt=2,
        min_quality_score=0.3,
        diversity_threshold=0.1,  # Lower for demo with small data
        min_clusters=2,
        batch_diversity_check_size=10,
    )

    # Custom teacher functions for demonstration
    def teacher_alpha(prompt: str, **kwargs) -> str:
        return f"[Alpha] Detailed explanation of: {prompt}. This concept is fundamental to modern AI systems."

    def teacher_beta(prompt: str, **kwargs) -> str:
        return f"[Beta] Quick summary: {prompt}. Key points are important to understand."

    def teacher_gamma(prompt: str, **kwargs) -> str:
        return f"[Gamma] In-depth analysis: {prompt}. Let me break this down step by step."

    teacher_callables = {
        "teacher_alpha": teacher_alpha,
        "teacher_beta": teacher_beta,
        "teacher_gamma": teacher_gamma,
    }

    # Generate
    generator = SyntheticDataGenerator(config, teacher_callables)
    synthetic_data = generator.generate_dataset(
        real_documents=real_documents,
        num_prompts=10,
    )

    # Print results
    print("\n" + "=" * 60)
    print("GENERATED SYNTHETIC DATA")
    print("=" * 60)

    for i, example in enumerate(synthetic_data[:5]):
        print(f"\n--- Example {i + 1} ---")
        print(f"  Teacher: {example['teacher']}")
        print(f"  Quality: {example['quality_score']:.3f}")
        print(f"  Prompt:  {example['prompt'][:80]}...")
        print(f"  Output:  {example['output'][:80]}...")

    # Print statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    stats = generator.get_stats()
    print(f"  Total generated:    {stats['total_generated']}")
    print(f"  Total kept:         {stats['total_kept']}")
    print(f"  Batches processed:  {stats['batches_processed']}")
    print(f"  Batches rejected:   {stats['batches_rejected_diversity']}")

    print("\n  Teacher weights:")
    for teacher, weight in generator.generator.teacher_weights.items():
        print(f"    {teacher}: {weight:.3f}")

    print("\n  Diversity stats:")
    div_stats = stats["diversity"]
    print(f"    Batches checked:  {div_stats['batches_checked']}")
    print(f"    Batches rejected: {div_stats['batches_rejected']}")
    print(f"    Avg diversity:    {div_stats['average_diversity_score']:.3f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_synthetic()
