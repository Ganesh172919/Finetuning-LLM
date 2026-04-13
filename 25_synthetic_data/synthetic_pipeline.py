"""
################################################################################
SYNTHETIC PIPELINE — GENERATING HIGH-QUALITY TRAINING DATA
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is a Synthetic Pipeline?
    A multi-stage process for generating high-quality training data.
    Inspired by Microsoft's Phi-4 and NVIDIA's Nemotron approaches,
    which use large models to create textbook-quality synthetic data
    for training smaller, more efficient models.

Why does it matter?
    Training data quality is often more important than quantity:
    - Phi-4 used synthetic data to train a 14B model that rivals GPT-4
    - Carefully curated synthetic data > massive unfiltered data
    - Quality filters remove noise and hallucinations
    - Multi-stage pipelines ensure diversity and coverage

How does it work?
    1. Topic Selection: Choose diverse topics to cover
    2. Content Generation: Large model generates content
    3. Quality Filtering: Remove low-quality outputs
    4. Diversity Sampling: Ensure broad coverage
    5. Verification: Check factual accuracy

Key Innovation (Phi-4):
    - "Textbook-quality" synthetic data
    - Multi-stage generation with verification
    - Knowledge distillation from larger models
    - Careful filtering for quality

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Synthetic Data Pipeline                          │
    │                                                                  │
    │  Topics ──▶ Generator ──▶ Raw Content                           │
    │                              ↓                                   │
    │              Quality Filter ──▶ Filtered Content                │
    │                              ↓                                   │
    │              Diversity Sampler ──▶ Diverse Content              │
    │                              ↓                                   │
    │              Verifier ──▶ Verified Training Data                │
    └─────────────────────────────────────────────────────────────────┘

Interview Questions:
    1. "How does synthetic data avoid model collapse?"
       Model collapse occurs when training on model-generated data
       causes quality to degrade over generations. Solutions:
       (a) Mix real and synthetic data, (b) Use quality filters,
       (c) Diverse generation strategies, (d) External verification.

    2. "What made Phi-4's synthetic data approach successful?"
       Key factors: (a) Textbook-quality generation prompts,
       (b) Multi-stage filtering, (c) Knowledge distillation
       from larger models, (d) Careful coverage of topics.

    3. "How do you measure synthetic data quality?"
       Metrics: (a) Perplexity on held-out real data,
       (b) Downstream task performance, (c) Diversity metrics,
       (d) Factual accuracy via verification.

################################################################################
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass, field
import math

################################################################################
# SECTION 1: CONFIGURATION
################################################################################

@dataclass
class PipelineConfig:
    """
    Configuration for Synthetic Data Pipeline.
    """
    # Generation
    vocab_size: int = 1000
    max_seq_len: int = 128
    num_topics: int = 100

    # Quality filtering
    min_length: int = 20
    max_length: int = 200
    quality_threshold: float = 0.5

    # Diversity
    diversity_bins: int = 10
    samples_per_bin: int = 10

    # Output
    total_samples: int = 1000


################################################################################
# SECTION 2: TOPIC GENERATOR
################################################################################

class TopicGenerator:
    """
    Topic Generator
    ================

    Generates diverse topics for synthetic data creation.

    Coverage strategy:
    - Broad coverage: Many different topics
    - Deep coverage: Multiple aspects per topic
    - Balanced: Equal representation across categories

    Categories (for AI training):
    - Science, Technology, History, Mathematics
    - Language, Reasoning, Code, Common sense
    - Domain-specific (medical, legal, financial)
    """

    def __init__(self, num_topics: int, vocab_size: int):
        self.num_topics = num_topics
        self.vocab_size = vocab_size

        # Topic categories
        self.categories = [
            "science", "technology", "history", "mathematics",
            "language", "reasoning", "code", "common_sense"
        ]

    def generate_topics(self) -> List[Dict[str, any]]:
        """
        Generate diverse topics.

        Returns:
            List of topic dictionaries with category and prompt seeds
        """
        topics = []
        topics_per_category = self.num_topics // len(self.categories)

        for category in self.categories:
            for i in range(topics_per_category):
                topic = {
                    'category': category,
                    'id': len(topics),
                    'seed_tokens': self._generate_seed(category, i),
                    'difficulty': np.random.choice(['easy', 'medium', 'hard'])
                }
                topics.append(topic)

        return topics

    def _generate_seed(self, category: str, index: int) -> List[int]:
        """Generate seed tokens for a topic."""
        # Simplified: use category hash and index as seeds
        base = hash(category) % self.vocab_size
        return [(base + index * 7 + i) % self.vocab_size for i in range(4)]


################################################################################
# SECTION 3: CONTENT GENERATOR
################################################################################

class ContentGenerator:
    """
    Content Generator
    ==================

    Generates content for each topic using a language model.

    Inspired by Phi-4's approach:
    - Use a large model to generate high-quality content
    - Multiple prompts per topic for diversity
    - Control quality through prompt engineering

    Generation strategies:
    - Textbook style: Structured, educational
    - Question-answer: Interactive format
    - Tutorial: Step-by-step instructions
    - Explanation: Concept-focused
    """

    def __init__(self, vocab_size: int, d_model: int = 64):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Simple generator model
        self.embed = np.random.randn(vocab_size, d_model) * 0.02
        self.proj = np.random.randn(d_model, vocab_size) * 0.02

    def generate(
        self,
        seed_tokens: List[int],
        max_len: int = 64,
        style: str = "textbook"
    ) -> List[int]:
        """
        Generate content from seed tokens.

        Args:
            seed_tokens: Starting tokens
            max_len: Maximum generation length
            style: Generation style (textbook, qa, tutorial)

        Returns:
            Generated token sequence
        """
        tokens = list(seed_tokens)

        # Adjust temperature based on style
        temp_map = {"textbook": 0.7, "qa": 0.9, "tutorial": 0.8, "explanation": 0.75}
        temperature = temp_map.get(style, 0.8)

        for _ in range(max_len):
            # Get last token embedding
            emb = self.embed[tokens[-1]]

            # Predict next token
            logits = emb @ self.proj / temperature

            # Add style-specific bias
            if style == "textbook":
                logits[0] += 0.1  # Prefer structured tokens
            elif style == "qa":
                logits[1] += 0.1  # Prefer question tokens

            probs = self._softmax(logits)
            next_token = np.random.choice(self.vocab_size, p=probs)

            tokens.append(next_token)

            # Stop condition (simplified)
            if next_token == 0:
                break

        return tokens[len(seed_tokens):]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def generate_batch(
        self,
        topics: List[Dict],
        samples_per_topic: int = 3
    ) -> List[Dict]:
        """
        Generate content for multiple topics.

        Args:
            topics: List of topic dictionaries
            samples_per_topic: Samples per topic

        Returns:
            List of generated content dictionaries
        """
        all_content = []

        for topic in topics:
            for i in range(samples_per_topic):
                style = np.random.choice(["textbook", "qa", "tutorial", "explanation"])
                content = self.generate(
                    topic['seed_tokens'],
                    max_len=64,
                    style=style
                )

                all_content.append({
                    'topic_id': topic['id'],
                    'category': topic['category'],
                    'style': style,
                    'tokens': content,
                    'difficulty': topic['difficulty']
                })

        return all_content


################################################################################
# SECTION 4: QUALITY FILTER
################################################################################

class QualityFilter:
    """
    Quality Filter
    ===============

    Filters generated content for quality.

    Quality criteria:
    - Length: Not too short, not too long
    - Diversity: Sufficient token variety
    - Coherence: Smooth transitions between tokens
    - Completeness: Has proper ending

    Interview Question:
        "How do you filter synthetic data quality?"
        Multiple criteria: (a) Length-based filtering removes
        too short/long samples, (b) Perplexity-based filtering
        removes low-probability sequences, (c) Diversity metrics
        ensure variety, (d) Rule-based checks for format.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def filter_length(self, content: List[Dict]) -> List[Dict]:
        """Filter by sequence length."""
        filtered = []
        for item in content:
            length = len(item['tokens'])
            if self.config.min_length <= length <= self.config.max_length:
                filtered.append(item)
        return filtered

    def filter_diversity(self, content: List[Dict]) -> List[Dict]:
        """Filter for token diversity."""
        filtered = []
        for item in content:
            tokens = item['tokens']
            if len(tokens) == 0:
                continue

            unique_ratio = len(set(tokens)) / len(tokens)
            if unique_ratio > 0.2:  # At least 20% unique tokens
                item['diversity_score'] = unique_ratio
                filtered.append(item)

        return filtered

    def filter_coherence(self, content: List[Dict]) -> List[Dict]:
        """Filter for coherence (simplified)."""
        filtered = []
        for item in content:
            tokens = item['tokens']
            if len(tokens) < 3:
                continue

            # Check for excessive repetition
            repeats = 0
            for i in range(2, len(tokens)):
                if tokens[i] == tokens[i-1] == tokens[i-2]:
                    repeats += 1

            repeat_ratio = repeats / max(len(tokens) - 2, 1)
            if repeat_ratio < 0.3:  # Less than 30% repetitive
                item['coherence_score'] = 1.0 - repeat_ratio
                filtered.append(item)

        return filtered

    def apply_all(self, content: List[Dict]) -> List[Dict]:
        """Apply all quality filters."""
        filtered = self.filter_length(content)
        filtered = self.filter_diversity(filtered)
        filtered = self.filter_coherence(filtered)

        return filtered


################################################################################
# SECTION 5: TEXTBOOK GENERATOR
################################################################################

class TextbookGenerator:
    """
    Textbook-Quality Content Generator
    =====================================

    Generates textbook-quality synthetic data, inspired by Phi-4.

    Key principles:
    - Structured content with clear sections
    - Educational explanations
    - Progressive difficulty
    - Comprehensive topic coverage

    Interview Question:
        "What makes synthetic data 'textbook-quality'?"
        (a) Clear structure with sections and headings,
        (b) Accurate information, (c) Appropriate difficulty level,
        (d) Complete explanations, (e) Examples and illustrations.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.generator = ContentGenerator(vocab_size)

    def generate_textbook_chapter(
        self,
        topic: Dict,
        num_sections: int = 3
    ) -> List[Dict]:
        """
        Generate a textbook chapter on a topic.

        Args:
            topic: Topic dictionary
            num_sections: Number of sections

        Returns:
            List of section content dictionaries
        """
        sections = []

        for i in range(num_sections):
            # Generate section with increasing difficulty
            difficulty = ['easy', 'medium', 'hard'][i % 3]

            content = self.generator.generate(
                topic['seed_tokens'] + [i],  # Add section index
                max_len=48,
                style="textbook"
            )

            sections.append({
                'topic_id': topic['id'],
                'section': i,
                'difficulty': difficulty,
                'tokens': content,
                'style': 'textbook'
            })

        return sections


################################################################################
# SECTION 6: SYNTHETIC PIPELINE (COMPLETE)
################################################################################

class SyntheticPipeline:
    """
    Complete Synthetic Data Pipeline
    ==================================

    End-to-end pipeline for generating high-quality synthetic training data.

    Stages:
    1. Topic generation: Diverse topic coverage
    2. Content generation: Multiple samples per topic
    3. Quality filtering: Remove low-quality content
    4. Diversity sampling: Ensure broad coverage
    5. Final selection: Choose best samples

    Based on Phi-4 and Nemotron approaches.

    Interview Questions:
        1. "How do you ensure synthetic data covers all topics?"
           Use a structured topic generator that systematically
           covers categories. Track coverage metrics and fill gaps.

        2. "What's the typical synthetic data volume?"
           Depends on the task. Phi-4 used ~15B tokens of synthetic
           data. For smaller models, 100M-1B tokens is common.

        3. "How do you prevent hallucinations in synthetic data?"
           (a) Use grounded generation (cite sources),
           (b) Verification against known facts,
           (c) Execution-based verification for code,
           (d) Cross-reference with real data.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        if config is None:
            config = PipelineConfig()
        self.config = config

        self.topic_gen = TopicGenerator(config.num_topics, config.vocab_size)
        self.content_gen = ContentGenerator(config.vocab_size)
        self.quality_filter = QualityFilter(config)
        self.textbook_gen = TextbookGenerator(config.vocab_size)

    def run(self) -> List[Dict]:
        """
        Run the complete synthetic data pipeline.

        Returns:
            List of verified, high-quality training samples
        """
        print("Stage 1: Generating topics...")
        topics = self.topic_gen.generate_topics()
        print(f"  Generated {len(topics)} topics")

        print("Stage 2: Generating content...")
        content = self.content_gen.generate_batch(topics, samples_per_topic=3)
        print(f"  Generated {len(content)} samples")

        print("Stage 3: Quality filtering...")
        filtered = self.quality_filter.apply_all(content)
        print(f"  After filtering: {len(filtered)} samples")

        print("Stage 4: Diversity sampling...")
        diverse = self._diversity_sample(filtered)
        print(f"  After diversity sampling: {len(diverse)} samples")

        print("Stage 5: Final selection...")
        selected = self._final_selection(diverse)
        print(f"  Final dataset: {len(selected)} samples")

        return selected

    def _diversity_sample(self, content: List[Dict]) -> List[Dict]:
        """Sample for diversity across categories."""
        # Group by category
        by_category = {}
        for item in content:
            cat = item['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)

        # Sample equally from each category
        samples = []
        per_category = max(1, self.config.total_samples // len(by_category))

        for cat, items in by_category.items():
            n = min(per_category, len(items))
            indices = np.random.choice(len(items), n, replace=False)
            for idx in indices:
                samples.append(items[idx])

        return samples

    def _final_selection(self, content: List[Dict]) -> List[Dict]:
        """Final selection of best samples."""
        # Sort by quality scores
        for item in content:
            quality = item.get('diversity_score', 0.5) * item.get('coherence_score', 0.5)
            item['quality_score'] = quality

        content.sort(key=lambda x: x['quality_score'], reverse=True)

        return content[:self.config.total_samples]


################################################################################
# SECTION 7: TESTING & DEMONSTRATION
################################################################################

def demonstrate_synthetic_pipeline():
    """Demonstrate synthetic data pipeline."""
    print("=" * 70)
    print("SYNTHETIC DATA PIPELINE")
    print("=" * 70)

    # Configuration
    config = PipelineConfig(
        vocab_size=100,
        max_seq_len=32,
        num_topics=20,
        min_length=5,
        max_length=50,
        total_samples=30
    )

    print(f"\nConfiguration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Num topics: {config.num_topics}")
    print(f"  Target samples: {config.total_samples}")

    # Create pipeline
    pipeline = SyntheticPipeline(config)

    # Run pipeline
    print("\n--- Running Pipeline ---")
    dataset = pipeline.run()

    # Analyze results
    print("\n--- Dataset Analysis ---")
    categories = {}
    for item in dataset:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1

    print("Category distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Quality scores
    scores = [item.get('quality_score', 0) for item in dataset]
    print(f"\nQuality scores:")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Min: {np.min(scores):.3f}")
    print(f"  Max: {np.max(scores):.3f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Quality > Quantity for synthetic data!")
    print("Careful filtering and diversity sampling create better datasets.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_synthetic_pipeline()
