# LSA vs LDA Pipeline Flow Comparison

## Visual Pipeline Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING (SHARED)                          │
│  Raw Text → Tokenization (jieba) → Stopword Removal → Normalized Text   │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │        LSA PIPELINE           │   │        LDA PIPELINE           │
    └───────────────────────────────┘   └───────────────────────────────┘
                    │                                 │
                    ▼                                 ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │   TfidfVectorizer             │   │   CountVectorizer             │
    │                               │   │                               │
    │  Input: Processed text        │   │  Input: Processed text        │
    │  Output: TF-IDF matrix        │   │  Output: Count matrix         │
    │                               │   │                               │
    │  部落: 0.234 (weighted)        │   │  部落: 5 (raw count)          │
    │  警方: 0.456 (rare = high)     │   │  警方: 8 (raw count)          │
    │  文化: 0.089 (common = low)    │   │  文化: 3 (raw count)          │
    │                               │   │                               │
    │  Type: Real-valued [0, 1]     │   │  Type: Integer [0, ∞)         │
    │  Size: (n_docs, n_features)   │   │  Size: (n_docs, n_features)   │
    └───────────────────────────────┘   └───────────────────────────────┘
                    │                                 │
                    ▼                                 ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │   TruncatedSVD                │   │  LatentDirichletAllocation    │
    │                               │   │                               │
    │  Algorithm: Randomized SVD    │   │  Algorithm: Variational Bayes │
    │  Iterations: One-shot         │   │  Iterations: Until convergence│
    │  Deterministic: Yes ✅        │   │  Deterministic: No (random) ⚠️│
    │                               │   │                               │
    │  Decomposition:               │   │  Optimization:                │
    │    X ≈ U Σ V^T                │   │    Maximize log P(w|θ, φ)     │
    │                               │   │                               │
    │  n_components: 10             │   │  n_topics: 10                 │
    │  random_state: 42             │   │  random_state: 42             │
    └───────────────────────────────┘   └───────────────────────────────┘
                    │                                 │
                    ▼                                 ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │   OUTPUT: Matrices            │   │   OUTPUT: Distributions       │
    │                               │   │                               │
    │  doc_topic_matrix:            │   │  doc_topic_matrix:            │
    │    Shape: (13, 10)            │   │    Shape: (13, 10)            │
    │    Values: Real [-∞, +∞]      │   │    Values: Prob [0, 1]        │
    │    Example: [-0.23, 0.45, ..]│   │    Example: [0.23, 0.45, ..]  │
    │    Constraint: None           │   │    Constraint: Σ = 1.0        │
    │                               │   │                               │
    │  topic_term_matrix:           │   │  topic_term_matrix:           │
    │    Shape: (10, 254)           │   │    Shape: (10, 254)           │
    │    Values: Real (loadings)    │   │    Values: Prob (P(w|topic))  │
    │    Can be negative: Yes       │   │    Can be negative: No        │
    └───────────────────────────────┘   └───────────────────────────────┘
                    │                                 │
                    ▼                                 ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │   INTERPRETATION              │   │   INTERPRETATION              │
    │                               │   │                               │
    │  Explained Variance:          │   │  Perplexity: 319.25           │
    │    Topic 0: 1.22%             │   │  Log-likelihood: -5247.04     │
    │    Topic 1: 13.08% ⭐         │   │                               │
    │    Total: 84.14%              │   │  Topic Entropy per doc        │
    │                               │   │  Dominant topic probability   │
    │  Interpretation:              │   │                               │
    │    "Topic 1 explains 13% of   │   │  Interpretation:              │
    │     variance in corpus"       │   │    "Doc 1 is 60% Topic 0,     │
    │                               │   │     30% Topic 1, 10% Topic 2" │
    └───────────────────────────────┘   └───────────────────────────────┘
                    │                                 │
                    └────────────────┬────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │     TOPIC EXTRACTION            │
                    │  Extract top terms per topic    │
                    │  n_top_words: 15                │
                    └─────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
    ┌───────────────────────────────┐   ┌───────────────────────────────┐
    │  LSA TOPICS (2006-2015)       │   │  LDA TOPICS (2006-2015)       │
    │                               │   │                               │
    │  Topic 0 (1.22% var):         │   │  Topic 0 (23% prev):          │
    │    部落, 泰雅族, 文化         │   │    部落, 泰雅族, 庫斯         │
    │    → Background context       │   │    → General tribal context   │
    │                               │   │                               │
    │  Topic 1 (13.08% var): ⭐     │   │  Topic 1 (7.7% prev):         │
    │    警方, 現場, 打獵           │   │    母親, 老人, 百歲, 人瑞    │
    │    → Most distinctive         │   │    → Elderly stories          │
    │    → Safety incidents         │   │                               │
    │                               │   │  Topic 2 (7.7% prev):         │
    │  Topic 2 (11.89% var):        │   │    族人, 生活, 縣長           │
    │    鎮西堡, 小米, 傳統         │   │    → Daily life & officials   │
    │    → Traditional practices    │   │                               │
    └───────────────────────────────┘   └───────────────────────────────┘
```

---

## Key Decision Points in Pipeline

### 1. Vectorization Choice ⭐ MOST CRITICAL

```python
# LSA: Emphasize distinctive terms
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.8,
    use_idf=True,      # ← KEY: Use IDF weighting
    sublinear_tf=True
)
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# LDA: Preserve raw frequencies
count_vectorizer = CountVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.8
    # No IDF weighting! ← KEY DIFFERENCE
)
count_matrix = count_vectorizer.fit_transform(texts)
```

**Why it matters**:
- TF-IDF: Term "警方" gets high weight (rare term)
- Count: Term "警方" gets count of 8 (just frequency)
- Result: LSA finds "警方" topics more distinctive

---

### 2. Algorithm Choice

```python
# LSA: Closed-form solution
lsa_model = TruncatedSVD(
    n_components=10,
    algorithm='randomized',  # Fast approximation
    n_iter=100,             # Iterations for approximation
    random_state=42         # Deterministic randomized algo
)
doc_topics = lsa_model.fit_transform(tfidf_matrix)  # One call

# LDA: Iterative optimization
lda_model = LatentDirichletAllocation(
    n_components=10,
    learning_method='online',  # Stochastic variational inference
    max_iter=100,             # Maximum iterations
    random_state=42           # Still has randomness
)
doc_topics = lda_model.fit_transform(count_matrix)  # Iterates internally
```

---

### 3. Output Constraints

```python
# LSA: No constraints
doc_topics[0]  # Can be: [-0.23, 0.45, -0.12, 0.78, ...]
               # Negative values are meaningful!

# LDA: Probability constraints
doc_topics[0]  # Must be: [0.23, 0.45, 0.12, 0.08, 0.12]
               # All positive, sum to 1.0
```

---

## Parallel Processing in Your Project

```
Time Period Split
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
    2006-2015         2016-2025         Future periods
        │                  │
        ├─────────┐        ├─────────┐
        ▼         ▼        ▼         ▼
       LSA       LDA      LSA       LDA
        │         │        │         │
        └────┬────┘        └────┬────┘
             ▼                  ▼
     Comparison 1         Comparison 2
             │                  │
             └────────┬─────────┘
                      ▼
              Cross-Period Analysis
```

**Your Implementation**:
- 4 parallel pipelines run simultaneously
- LSA and LDA for each time period
- Results compared for validation

---

## Performance Characteristics

### Speed Comparison (Approximate)

```
Small Corpus (13-25 documents, 254-297 terms):

┌────────────────────────────────────────────────┐
│ LSA:  ████▌ 0.05s (vectorization + SVD)       │
│ LDA:  █████████████████████▌ 2.5s (iterations)│
└────────────────────────────────────────────────┘

Large Corpus (10,000 documents, 5,000 terms):

┌────────────────────────────────────────────────┐
│ LSA:  ████████████▌ 3.2s                       │
│ LDA:  ████████████████████████████████▌ 95s    │
└────────────────────────────────────────────────┘
```

**Your Case**: Both methods are fast enough (< 3s each)

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                       INPUT DATA                             │
│  data/03_primary/jianshi_news_raw.csv                       │
│  • 25 news articles about Jianshi Township                  │
│  • Split into 2006-2015 (13 docs) and 2016-2025 (12 docs)  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE                     │
│  src/lsa_markus/pipelines/data_engineering/                 │
│  • Chinese text segmentation (jieba)                        │
│  • Stopword removal (custom list)                           │
│  • Normalization                                            │
│  Output: preprocessed_2006_2015, preprocessed_2016_2025     │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   FEATURE EXTRACTION     │  │   FEATURE EXTRACTION     │
│   (TF-IDF Pipeline)      │  │   (LDA Pipeline)         │
│                          │  │                          │
│  src/.../tfidf_analysis/ │  │  src/.../lda_analysis/   │
│  • TfidfVectorizer       │  │  • CountVectorizer       │
│  • max_features: 1000    │  │  • max_features: 1000    │
│                          │  │                          │
│  Output:                 │  │  Output:                 │
│  tfidf_matrix_2006_2015  │  │  doc_term_matrix_*       │
│  tfidf_matrix_2016_2025  │  │                          │
└──────────────────────────┘  └──────────────────────────┘
              │                           │
              ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   LSA ANALYSIS           │  │   LDA ANALYSIS           │
│                          │  │                          │
│  src/.../lsa_analysis/   │  │  src/.../lda_analysis/   │
│  • perform_lsa()         │  │  • perform_lda()         │
│  • TruncatedSVD          │  │  • LatentDirichlet       │
│  • n_components: 10      │  │    Allocation            │
│                          │  │  • n_topics: 10          │
│  Output:                 │  │                          │
│  lsa_results_2006_2015   │  │  Output:                 │
│  lsa_results_2016_2025   │  │  lda_results_2006_2015   │
│                          │  │  lda_results_2016_2025   │
└──────────────────────────┘  └──────────────────────────┘
              │                           │
              └──────────┬────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   COMPARISON & ANALYSIS                      │
│  • compare_time_periods() for LSA                           │
│  • compare_lda_topics() for LDA                             │
│  • Cross-method validation                                  │
│                                                             │
│  Output:                                                    │
│  data/08_reporting/lsa_topic_comparison.csv                 │
│  data/08_reporting/lda_topic_comparison.csv                 │
│  data/08_reporting/lsa_vs_lda_comparison.csv                │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    VISUALIZATION                             │
│  • visualize_pkl.py                                         │
│  • compare_lsa_lda.py                                       │
│  • analyze_topic_selection.py                               │
│                                                             │
│  Output:                                                    │
│  data/08_reporting/*.png (multiple visualizations)          │
└─────────────────────────────────────────────────────────────┘
```

---

## Command Reference

```bash
# Run full pipeline (both LSA and LDA)
kedro run

# Run only LSA pipeline
kedro run --pipeline lsa_analysis

# Run only LDA pipeline
kedro run --pipeline lda_analysis

# Compare LSA vs LDA results
python src/compare_lsa_lda.py

# Analyze topic selection methodology
python src/analyze_topic_selection.py

# Interactive exploration
kedro jupyter notebook
# Then open: notebooks/01_visualize_results.ipynb
```

---

## File Structure

```
lsa-markus/
├── data/
│   ├── 03_primary/
│   │   └── jianshi_news_raw.csv        # Input data
│   ├── 05_model_input/
│   │   ├── preprocessed_2006_2015.pkl
│   │   ├── preprocessed_2016_2025.pkl
│   │   ├── tfidf_matrix_2006_2015.pkl  # LSA input
│   │   └── doc_term_matrix_*.pkl       # LDA input
│   ├── 07_model_output/
│   │   ├── lsa_results_2006_2015.pkl   # LSA results
│   │   ├── lsa_results_2016~2025.pkl
│   │   ├── lda_results_2006_2015.pkl   # LDA results
│   │   └── lda_results_2016~2025.pkl
│   └── 08_reporting/
│       ├── lsa_vs_lda_comparison.csv
│       └── *.png                       # Visualizations
│
├── src/lsa_markus/pipelines/
│   ├── lsa_analysis/
│   │   ├── nodes.py                    # LSA implementation
│   │   └── pipeline.py                 # LSA pipeline def
│   └── lda_analysis/
│       ├── nodes.py                    # LDA implementation
│       └── pipeline.py                 # LDA pipeline def
│
├── src/
│   ├── compare_lsa_lda.py              # Comparison tool
│   ├── analyze_topic_selection.py      # Topic analysis
│   └── visualize_pkl.py                # Visualization
│
└── notes/
    ├── LSA_vs_LDA_COMPARISON.md        # Full comparison
    ├── LSA_LDA_QUICK_SUMMARY.md        # Quick reference
    └── PIPELINE_FLOW_COMPARISON.md     # This file
```

---

**Created**: 2025-12-08
**Purpose**: Visual guide to understanding LSA vs LDA pipeline differences
