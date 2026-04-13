# NeuralForge Academy — Iteration Log

## Iteration 1 — Completed

### Focus
- Project foundation
- Hero landing page
- Navigation system
- Track/module structure

### Files Created
| File | Description |
|------|-------------|
| `index.html` | Full application shell with Three.js hero, sidebar nav, track sections |

### Implementations

**Hero Landing Page:**
- Three.js neural galaxy with 10,000 particles
- 4 semantic clusters: LLM (violet), RAG (cyan), Fine-Tuning (emerald), Optimization (amber)
- Custom GLSL vertex/fragment shaders
- Central knowledge singularity with pulsing glow
- Mouse parallax camera control
- GSAP entrance animations for title, subtitle, CTA, stats

**Navigation System:**
- Fixed sidebar with all 6 tracks and 47 modules
- Expandable/collapsible track sections
- Fixed top bar with breadcrumb and XP badge
- Module cards with difficulty badges (INIT/GRAD/EXPERT/FRONTIER)

**Track Coverage:**
- Track 0: Foundation (5 modules)
- Track 1: LLMs (9 modules)
- Track 2: RAG (11 modules)
- Track 3: Fine-Tuning (11 modules)
- Track 4: Optimization (9 modules)
- Track 5: Research (7 modules)

### Next Iteration Plan
1. Add interactive 3D embedding space (Module 1.2)
2. Create Transformer factory scene (Module 1.3)
3. Implement code playground with Pyodide
4. Add loss landscape 3D visualization (Module 1.6)
5. Create RAG pipeline 3D flow (Module 2.2)

---

## Iteration 2 — Completed

### Focus
- Module 1.3: Transformer Factory 3D Scene
- Module 1.2: Word Embeddings 3D Visualization
- Module page template system

### Files Created
| File | Description |
|------|-------------|
| `modules/transformer-factory.html` | Full Transformer module with 3D factory, 8 stations, attention sim |
| `modules/embedding-space.html` | Embedding space 3D visualization with analogy engine |

### Implementations

**Transformer Factory (Module 1.3):**
- 3D factory scene with 8 interactive stations
- Tokenization → Embedding → Position → Attention → FFN → LayerNorm → Residual → Output
- Animated data particles flowing through conveyor belt
- Station highlighting on click
- 5 explanation levels (child → researcher)
- KaTeX math formulas (attention, multi-head, FFN)
- Interactive attention heatmap simulation
- From-scratch NumPy code implementation
- Common mistakes section
- Module quiz

**Embedding Space (Module 1.2):**
- 3D semantic space with 200 words in 4 clusters
- Animals (emerald), Countries (cyan), Emotions (rose), Actions (amber)
- Word-to-word connection lines
- Analogy visualization (king - man + woman = queen)
- Cosine similarity calculator
- PCA/t-SNE/UMAP projection toggle
- Semantic Compass tool (find nearest neighbors)
- KaTeX cosine similarity formula

### Module Template Sections
1. Module Header (badge, title, subtitle, meta)
2. Hook Scene (3D canvas)
3. Conceptual Foundation (5 levels)
4. Mathematical Deep Dive (KaTeX)
5. Interactive Simulation
6. Code Implementation
7. Common Mistakes
8. Module Quiz
9. Navigation Footer

### Next Iteration Plan
1. Module 1.4: Attention Mechanisms deep dive
2. Module 1.6: Loss Landscape 3D visualization
3. Code playground with Pyodide integration
4. Module 2.1: Why LLMs Hallucinate
5. Add quiz feedback system

---

## Iteration 3 — Completed

### Focus
- Module 1.6: Loss Landscape 3D visualization
- Module 2.1: Why LLMs Hallucinate
- Optimization simulation

### Files Created
| File | Description |
|------|-------------|
| `modules/loss-landscape.html` | 3D loss landscape with optimizer navigation |
| `modules/hallucination.html` | Why LLMs hallucinate + RAG motivation |

### Implementations

**Loss Landscape (Module 1.6):**
- 3D terrain generated from loss function
- Interactive optimizer sphere navigating terrain
- SGD, Momentum, Adam, AdamW comparison
- Adjustable learning rate slider
- Trail visualization showing optimizer path
- Color-coded terrain (red = high loss, green = low loss)
- Randomize landscape button
- Gradient descent mathematics

**Why LLMs Hallucinate (Module 2.1):**
- 3D brain sphere with question particles
- Interactive hallucination demo (click questions)
- Color-coded responses (green = correct, red = hallucinated)
- Parametric vs Non-Parametric memory comparison
- Hallucination types (factual, fabricated sources, logical errors)
- Mitigation strategies (RAG, CoT, Self-Consistency, Fine-Tuning)
- Training data cutoff visualization

### Module Coverage
- Track 1: Modules 1.2, 1.3, 1.6 (3 modules)
- Track 2: Module 2.1 (1 module)
- Total: 4 module pages

### Next Iteration Plan
1. Module 1.4: Attention Mechanisms deep dive
2. Module 2.2: The RAG Pipeline
3. Code playground with Pyodide
4. Module 3.4: LoRA Mathematics
5. Add quiz feedback system

---

## Iteration 4 — Completed

### Focus
- Module 2.2: The Complete RAG Pipeline
- Chunking strategies simulator
- Retrieval demo

### Files Created
| File | Description |
|------|-------------|
| `modules/rag-pipeline.html` | Full RAG pipeline with 3D visualization |

### Implementations

**RAG Pipeline (Module 2.2):**
- 3D pipeline factory with 5 stations (Load, Chunk, Embed, Index, Retrieve)
- Animated data particles flowing through conveyor belt
- Ingestion and Query pipeline visualization
- Interactive chunking strategy simulator (Fixed, Sentence, Semantic)
- Adjustable chunk size and overlap sliders
- Chunk metrics (count, avg size, overlap)
- Embedding space visualization
- Interactive retrieval demo with top-k control
- RAG from scratch code implementation
- RAG evaluation metrics (Faithfulness, Precision, Recall, Relevance)

### Module Coverage
- Track 1: Modules 1.2, 1.3, 1.6 (3 modules)
- Track 2: Modules 2.1, 2.2 (2 modules)
- Total: 5 module pages

### Next Iteration Plan
1. Module 1.4: Attention Mechanisms deep dive
2. Module 2.5: Vector Databases (HNSW visualization)
3. Module 3.4: LoRA Mathematics
4. Code playground with Pyodide
5. Add quiz feedback system

---

## Iteration 5 — Completed

### Focus
- Module 2.5: Vector Databases & HNSW
- ANN index comparison
- Vector database comparison

### Files Created
| File | Description |
|------|-------------|
| `modules/vector-databases.html` | HNSW 3D visualization + vector DB comparison |

### Implementations

**Vector Databases (Module 2.5):**
- 3D HNSW graph visualization with 4 layers
- Layer structure: Entry Point → Sparse → Medium → Dense
- Search animation (greedy search through layers)
- Add vector button (dynamic graph updates)
- Adjustable number of vectors and search K
- ANN index comparison table (Flat, IVF, HNSW, PQ)
- Vector database comparison (Chroma, Pinecone, Weaviate, Qdrant, FAISS)
- FAISS implementation code

### Module Coverage
- Track 1: Modules 1.2, 1.3, 1.6 (3 modules)
- Track 2: Modules 2.1, 2.2, 2.5 (3 modules)
- Total: 6 module pages

### Next Iteration Plan
1. Module 1.4: Attention Mechanisms deep dive
2. Module 3.4: LoRA Mathematics
3. Module 2.8: Advanced RAG (HyDE, Self-RAG)
4. Code playground with Pyodide
5. Add quiz feedback system

---

## Iteration 6 — Completed

### Focus
- Module 1.4: Attention Mechanisms

### Files Created
| File | Description |
|------|-------------|
| `modules/attention-mechanisms.html` | Interactive attention visualization with 4 difficulty levels |

### Implementations

**Module 1.4: Attention Mechanisms:**
- 4-level explanations (Beginner/Intermediate/Advanced/Expert)
- Hero canvas with animated particle network
- Interactive self-attention heatmap visualization
- 3 attention heads: Syntactic, Semantic, Positional
- Sentence selector with 3 example sentences
- Animated attention weight growth
- Q·K·V pipeline step-through visualization
- 8-stage pipeline: Input → Q → K → V → Scores → Softmax → Weighted Sum → Output
- Code playground with self-attention implementation
- 3-question quiz with feedback
- Navigation to adjacent modules

### Module Coverage
- Track 1: Modules 1.2, 1.3, 1.4, 1.6 (4 modules)
- Track 2: Modules 2.1, 2.2, 2.5 (3 modules)
- Total: 7 module pages

### Next Iteration Plan
1. Module 3.4: LoRA Mathematics
2. Module 2.8: Advanced RAG (HyDE, Self-RAG)
3. Module 5.1: Distributed Training
4. Code playground with Pyodide
5. Quiz feedback system with scoring

---

---

## Iteration 7 — Completed

### Focus
- Module 3.4: LoRA Mathematics
- Navigation system upgrade

### Files Created
| File | Description |
|------|-------------|
| `modules/lora-mathematics.html` | Interactive LoRA decomposition with 4 difficulty levels |

### Implementations

**Module 3.4: LoRA Mathematics:**
- 4-level explanations (Beginner/Intermediate/Advanced/Expert)
- Hero canvas with low-rank flow particle animation
- Interactive matrix decomposition (W = W₀ + B·A) with animated step-through
- Rank selection explorer with quality/memory tradeoff curves
- Layer-wise LoRA application visualizer (Q/K/V, FFN, Attention)
- Code playground with LoRA implementation
- PEFT method comparison table (LoRA, QLoRA, DoRA, GaLore, Prefix, Adapter)
- 5-question quiz with feedback
- Stats bars: params, compression ratio, memory saved

**Navigation:**
- loadModule() now routes to actual module pages (was a stub)
- Module map for 8 modules

### Module Coverage
- Track 1: Modules 1.2, 1.3, 1.4, 1.6 (4 modules)
- Track 2: Modules 2.1, 2.2, 2.5 (3 modules)
- Track 3: Module 3.4 (1 module)
- Total: 8 module pages

### Next Iteration Plan
1. Module 3.5: QLoRA & Quantization
2. Module 3.6: RLHF Pipeline
3. Module 2.3: Document Chunking Strategies
4. Module 1.5: Positional Encoding
5. Quiz scoring and progress tracking

*Last updated: 2026-06-14*
