# LSA vs LDA: Comprehensive Comparison

## Executive Summary

This document compares LSA (Latent Semantic Analysis) and LDA (Latent Dirichlet Allocation) as implemented in your project, analyzing their algorithmic differences, pipeline structures, and practical applications.

---

## 1. Algorithmic Foundations

### LSA (Latent Semantic Analysis)

**Mathematical Foundation**: Matrix Factorization via Singular Value Decomposition (SVD)

```
X ≈ U Σ V^T

Where:
- X: Document-term matrix (TF-IDF weighted)
- U: Document-topic matrix (documents in topic space)
- Σ: Diagonal matrix of singular values (topic importance)
- V^T: Topic-term matrix (topics as term combinations)
```

**Key Characteristics**:
- ✅ **Deterministic**: Same results every run (no randomness)
- ✅ **Linear algebra-based**: Fast, closed-form solution
- ✅ **Global structure**: Captures overall corpus patterns
- ⚠️  **Bidirectional weights**: Topics can have negative values
- ✅ **Fast computation**: Efficient truncated SVD algorithms

**Theoretical Basis**: Dimensionality reduction through identifying principal components that explain maximum variance in the term-document space.

---

### LDA (Latent Dirichlet Allocation)

**Mathematical Foundation**: Probabilistic Generative Model

```
Assumptions:
1. Each document is a mixture of topics ~ Dirichlet(α)
2. Each topic is a distribution over words ~ Dirichlet(β)
3. Each word is drawn from a topic-specific distribution

Generative Process:
For each document d:
  1. Draw topic distribution θ_d ~ Dirichlet(α)
  For each word position n:
    2. Choose topic z_n ~ Multinomial(θ_d)
    3. Choose word w_n ~ Multinomial(φ_z_n)
```

**Key Characteristics**:
- ⚠️  **Probabilistic**: Results vary slightly between runs
- ✅ **Bayesian inference**: Based on statistical modeling
- ✅ **Generative model**: Models how documents are created
- ✅ **All-positive probabilities**: Easier interpretation
- ⚠️  **Iterative optimization**: Requires convergence

**Inference Method (Your Implementation)**:
- ✅ **Variational Bayes (Stochastic Variational Inference)**
- Uses `learning_method='online'` in scikit-learn's `LatentDirichletAllocation`
- Processes mini-batches (batch_size=128) for scalability
- NOT Gibbs Sampling (MCMC method)
- NOT traditional EM (though conceptually similar)

**Theoretical Basis**: Bayesian hierarchical model assuming documents are generated from latent topic mixtures.

---

## 2. Pipeline Comparison

### LSA Pipeline

```
1. Text Preprocessing
   ├─ Tokenization (jieba for Chinese)
   ├─ Stopword removal
   └─ Normalization
        ↓
2. TF-IDF Vectorization ⭐ CRITICAL STEP
   ├─ Term Frequency (TF): How often term appears
   ├─ Inverse Document Frequency (IDF): Downweight common terms
   ├─ Formula: TF-IDF(t,d) = TF(t,d) × log(N / DF(t))
   └─ Output: Real-valued, normalized matrix
        ↓
3. Truncated SVD
   ├─ Input: TF-IDF matrix (real values)
   ├─ Decomposition: X = U Σ V^T
   ├─ Keep top k singular values
   └─ Output: Low-dimensional topic representations
        ↓
4. Interpretation
   ├─ Explained variance per topic
   ├─ Topic loadings (can be negative)
   └─ Document similarity in topic space
```

**Code Structure** ([src/lsa_markus/pipelines/lsa_analysis/nodes.py](../src/lsa_markus/pipelines/lsa_analysis/nodes.py:18-46)):
```python
def perform_lsa(tfidf_matrix, parameters):
    lsa_model = TruncatedSVD(
        n_components=10,
        algorithm='randomized',
        n_iter=100,
        random_state=42
    )
    doc_topic_matrix = lsa_model.fit_transform(tfidf_matrix)
    topic_term_matrix = lsa_model.components_
    return lsa_model, doc_topic_matrix, topic_term_matrix
```

---

### LDA Pipeline

```
1. Text Preprocessing
   ├─ Tokenization (jieba for Chinese)
   ├─ Stopword removal
   └─ Normalization (same as LSA)
        ↓
2. Count Vectorization ⭐ CRITICAL DIFFERENCE
   ├─ Raw word counts (no TF-IDF weighting)
   ├─ Integer counts: 0, 1, 2, 3, ...
   ├─ Preserves frequency information
   └─ Output: Integer count matrix
        ↓
3. LDA Model Fitting ⭐ USES VARIATIONAL BAYES
   ├─ Input: Document-term count matrix
   ├─ Inference Method: Stochastic Variational Inference (SVI)
   ├─ Mini-batch processing (batch_size=128)
   ├─ Iterative optimization until convergence (max_iter=100)
   └─ Output: Probability distributions
        ↓
4. Interpretation
   ├─ Perplexity (model fit quality)
   ├─ Log-likelihood
   ├─ Topic probabilities (sum to 1)
   └─ Topic entropy (diversity measure)
```

**Code Structure** ([src/lsa_markus/pipelines/lda_analysis/nodes.py](../src/lsa_markus/pipelines/lda_analysis/nodes.py:50-81)):
```python
def perform_lda(doc_term_matrix, parameters):
    lda_model = LatentDirichletAllocation(
        n_components=10,
        max_iter=100,
        learning_method='online',
        random_state=42,
        n_jobs=-1
    )
    doc_topic_matrix = lda_model.fit_transform(doc_term_matrix)
    topic_term_matrix = lda_model.components_
    return lda_model, doc_topic_matrix, topic_term_matrix
```

**LDA Inference Method Details**:

Your implementation uses **Stochastic Variational Inference (SVI)**, NOT Gibbs Sampling or traditional EM.

**How it works**:
```
For each iteration (up to max_iter=100):
  1. Sample a mini-batch of documents (batch_size=128)
  2. E-step (local): Compute variational parameters for batch documents
     - Estimate document-topic distributions (θ)
  3. M-step (global): Update topic-word distributions (φ)
     - Using stochastic gradient with learning rate decay
  4. Check convergence (perplexity or log-likelihood)
```

**Comparison of LDA Inference Methods**:

| Method | Speed | Accuracy | Your Choice |
|--------|-------|----------|-------------|
| **Online (SVI)** | ⚡⚡⚡ Fast | ✓ Good | ✅ YES (learning_method='online') |
| **Batch (VB-EM)** | ⚡⚡ Medium | ✓✓ Better | ❌ Not used |
| **Gibbs Sampling** | ⚡ Slow | ✓✓✓ Best | ❌ Not used |

**For your small corpus (13-25 docs)**: Batch mode might give slightly better results, but online mode works fine and is faster.

---

## 3. Critical Input Difference

### LSA Input: TF-IDF Matrix

**Example Values**:
```
              部落    文化    警方    泰雅族
Document 1:  0.234   0.187   0.000   0.156
Document 2:  0.123   0.089   0.456   0.067
Document 3:  0.345   0.234   0.000   0.289
```

**Characteristics**:
- Real-valued, typically between 0 and 1
- **Emphasizes distinctive terms** (high IDF score)
- **De-emphasizes common terms** (low IDF score)
- Normalized by document length

**Effect**: Focuses on terms that differentiate documents from each other.

---

### LDA Input: Count Matrix

**Example Values**:
```
              部落   文化   警方   泰雅族
Document 1:    5      3      0      2
Document 2:    2      1      8      1
Document 3:    7      4      0      5
```

**Characteristics**:
- Integer values (0, 1, 2, 3, ...)
- **All terms weighted equally** by default
- **Preserves raw frequency** information
- No normalization by IDF

**Effect**: Treats all term frequencies as equally informative about topic mixtures.

---

## 4. Output Comparison: Your Results

### Metrics Summary

| Period    | Method | N Topics | Total Variance | Primary Metric         | Weight Type               |
|-----------|--------|----------|----------------|------------------------|---------------------------|
| 2006-2015 | LSA    | 10       | 84.14%         | Variance: 13.08% (T1)  | Real (-∞ to +∞)           |
| 2006-2015 | LDA    | 10       | N/A            | Perplexity: 319.25     | Probability (0 to 1)      |
| 2016-2025 | LSA    | 10       | 88.70%         | Variance: 13.09% (T1)  | Real (-∞ to +∞)           |
| 2016-2025 | LDA    | 10       | N/A            | Perplexity: 407.33     | Probability (0 to 1)      |

### Topic Discovery Comparison (2006-2015)

**LSA Topics**:
```
Topic 0 (1.22% variance): 部落, 泰雅族, 原住民, 庫斯, 傳統, 文化
  → Background: Indigenous cultural context

Topic 1 (13.08% variance): 警方, 現場, 樹林, 打獵, 發現
  → Distinctive: Safety incidents involving police

Topic 2 (11.89% variance): 傳統, 鎮西堡, 小米, 部落, 文化, 領域
  → Distinctive: Traditional tribal practices & tourism
```

**LDA Topics**:
```
Topic 0 (prevalence: 0.230): 部落, 泰雅族, 庫斯, 依倚, 傳統, 司馬
  → General tribal and cultural references

Topic 1 (prevalence: 0.077): 泰雅, 一名, 家人, 母親, 老人, 百歲, 人瑞
  → Human interest stories about elderly community members

Topic 2 (prevalence: 0.077): 族人, 生活, 上山, 民族, 縣長, 邱鏡淳
  → Daily life and government officials
```

**Key Observation**:
- **LSA** emphasizes **discriminative patterns** (what makes documents different)
- **LDA** discovers **content themes** (what documents are about)

---

## 5. Interpretation Differences

### LSA: Explained Variance

```python
# From your results
lsa_results['explained_variance']
# [0.0122, 0.1308, 0.1189, ...]

Interpretation:
- Topic 0: Explains 1.22% of variance (background/common)
- Topic 1: Explains 13.08% of variance (most distinctive)
- Total: 84.14% of corpus variance explained
```

**Meaning**: How much of the document differences each topic captures.

---

### LDA: Probability Distributions

```python
# From your results
doc_topic_matrix[0]  # Document 0
# [0.230, 0.077, 0.077, 0.066, 0.091, ...]
# Sums to 1.0

topic_term_matrix[0]  # Topic 0
# Probability distribution over all terms
# Each value is P(word | topic)
```

**Meaning**: How topics are mixed in documents, and how words are distributed in topics.

---

## 6. When to Use Which Method

### Use LSA When:

✅ **Speed is critical**
- Deterministic, single-pass algorithm
- No iterative convergence needed

✅ **Finding distinctive patterns**
- You want to know what separates documents
- Example: "What makes incident reports different from cultural news?"

✅ **Semantic similarity search**
- Document clustering
- Information retrieval
- Recommendation systems

✅ **Small to medium corpora**
- Your case: 13-25 documents ✓
- Works well with < 10,000 documents

✅ **Variance-based interpretation**
- You need explained variance metrics
- Understanding dimensionality reduction

### Use LDA When:

✅ **Interpretability is critical**
- All-positive probabilities
- "Document X is 60% politics, 30% culture, 10% sports"

✅ **Discovering content themes**
- You want to know what documents are about
- Example: "What topics exist in this news corpus?"

✅ **Modeling document generation**
- Understanding how content is composed
- Topic evolution over time

✅ **Medium to large corpora**
- Works best with > 1,000 documents
- Scales well with data

✅ **Probabilistic reasoning**
- Need uncertainty quantification
- Topic coherence metrics

---

## 7. Your Project: Why Use Both?

### Complementary Insights

**LSA Answers**: "What patterns discriminate documents?"
```
→ Topic 1 (13.08% variance) = Safety incidents
→ Separates incident reports from other news
→ High variance = strong discriminative power
```

**LDA Answers**: "What themes comprise documents?"
```
→ Topic 1 (7.7% prevalence) = Elderly community stories
→ Present in specific subset of documents
→ Probabilistic mixture = content composition
```

### Cross-Validation Strategy

1. **Run both methods** on same corpus ✓ (already done)
2. **Compare discovered topics** - look for consistency
3. **Validate interpretations** - do both methods find similar themes?
4. **Report complementary findings** in research paper

### Your Findings

| Aspect | LSA | LDA |
|--------|-----|-----|
| **Background Context** | Topic 0 (部落, 泰雅族, 文化) | Topic 0 (部落, 泰雅族, 庫斯) |
| **Distinctive Pattern** | Safety incidents (13% var) | Elderly stories (7.7% prev) |
| **Validation** | ✅ Both find tribal/cultural background | ✅ Consistent baseline |

---

## 8. Implementation Details

### Parameters Used

**LSA** ([conf/base/parameters.yml](../conf/base/parameters.yml:20-25)):
```yaml
lsa:
  n_components: 10
  algorithm: "randomized"
  n_iter: 100
  random_state: 42
```

**LDA** ([conf/base/parameters.yml](../conf/base/parameters.yml:27-36)):
```yaml
lda:
  n_topics: 10
  max_iter: 100
  learning_method: "online"
  random_state: 42
  n_jobs: -1
  doc_topic_prior: null  # alpha (auto)
  topic_word_prior: null  # beta (auto)
```

---

## 9. Practical Recommendations

### For Your Research Paper

**DO Report**:
```
"We employed both LSA and LDA to provide complementary perspectives:

- LSA (explained variance analysis) identified distinctive patterns
  through variance-based decomposition, with Topic 1 capturing
  safety incidents (13.08% variance explained).

- LDA (probabilistic topic modeling) discovered content themes
  through Bayesian inference, revealing document compositions
  as probability distributions over topics.

Both methods consistently identified indigenous cultural context
(部落, 泰雅族, 文化) as the baseline theme across all documents,
validating our corpus focus on Jianshi Township."
```

**DON'T Report**:
- "LSA is better than LDA" or vice versa
- Results from only one method without justification
- Conflicting interpretations without explanation

### For Future Analysis

1. **Optimal topic number**: Test k=5, 10, 15, 20 for both methods
2. **Topic coherence**: Calculate coherence scores for LDA topics
3. **Temporal evolution**: Track topic changes across finer time periods
4. **Validation**: Human evaluation of topic interpretability

---

## 10. Tools and Visualizations

### Available Scripts

1. **Compare LSA vs LDA**:
   ```bash
   python src/compare_lsa_lda.py
   ```
   - Comprehensive algorithmic comparison
   - Side-by-side visualizations
   - Output: [data/08_reporting/lsa_vs_lda_comparison_*.png](../data/08_reporting/)

2. **View Topic Terms**:
   ```bash
   python src/view_topic_term_matrix.py
   ```
   - Explore both LSA and LDA topic-term matrices

3. **Analyze Results**:
   ```bash
   python src/visualize_pkl.py
   ```
   - Publication-ready visualizations

### Generated Files

- **Comparison CSV**: [data/08_reporting/lsa_vs_lda_comparison.csv](../data/08_reporting/lsa_vs_lda_comparison.csv)
- **Visualizations**:
  - [lsa_vs_lda_comparison_2006_2015.png](../data/08_reporting/lsa_vs_lda_comparison_2006_2015.png)
  - [lsa_vs_lda_comparison_2016_2025.png](../data/08_reporting/lsa_vs_lda_comparison_2016_2025.png)

---

## Summary Table

| Aspect | LSA | LDA |
|--------|-----|-----|
| **Math Foundation** | SVD (Linear Algebra) | Bayesian Generative Model |
| **Input** | TF-IDF Matrix | Count Matrix |
| **Speed** | Fast (deterministic) | Slower (iterative) |
| **Output Weights** | Real (can be negative) | Probabilities (0-1) |
| **Interpretation** | Variance explained | Topic mixtures |
| **Best For** | Discriminative patterns | Content themes |
| **Your Use Case** | Finding distinctive topics | Understanding compositions |
| **Consistency** | ✅ Same every run | ⚠️ Slight variation |
| **Scalability** | Small-medium corpora | Medium-large corpora |

---

## Conclusion

Both LSA and LDA are valuable for your research:

- **LSA**: Efficiently identifies what makes documents different (variance-based)
- **LDA**: Interprets what documents contain (probability-based)
- **Together**: Provide robust, cross-validated topic analysis

Your implementation successfully uses both methods to analyze temporal changes in Jianshi Township news coverage, with consistent findings across approaches validating your interpretations.

---

**Generated**: 2025-12-08
**Analysis Tool**: [src/compare_lsa_lda.py](../src/compare_lsa_lda.py)
**Project**: LSA-Markus Topic Modeling Pipeline
