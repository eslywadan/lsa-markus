# LSA vs LDA: Quick Summary

## One-Sentence Difference

**LSA** finds what makes documents **different** (discriminative patterns via variance).
**LDA** finds what documents **contain** (content themes via probability distributions).

---

## Key Differences at a Glance

| Aspect | LSA | LDA |
|--------|-----|-----|
| **Input** | TF-IDF (weighted, emphasizes rare terms) | Counts (raw frequency) |
| **Algorithm** | SVD (deterministic, one-shot) | Variational Bayes (iterative) |
| **Inference** | Truncated SVD (linear algebra) | Stochastic Variational Inference (SVI) |
| **Output** | Real weights (can be negative) | Probabilities (0 to 1, sum to 1) |
| **Metric** | Explained variance | Perplexity, log-likelihood |
| **Speed** | ⚡ Fast | 🐢 Slower |
| **Interpretation** | "Topic 1 explains 13% of variance" | "Doc is 60% topic 1, 30% topic 2" |
| **Purpose** | Semantic similarity, dimensionality reduction | Topic modeling, content discovery |

---

## Your Results (2006-2015)

### LSA Found:
- **Topic 0** (1.22% variance): Background context (部落, 泰雅族, 文化)
- **Topic 1** (13.08% variance): **Most distinctive** - Safety incidents (警方, 現場, 打獵)
- **Topic 2** (11.89% variance): Traditional practices (鎮西堡, 小米, 傳統)

**Insight**: Topic 1 has high variance → strongly separates documents

### LDA Found:
- **Topic 0** (23% prevalence): General tribal context (部落, 泰雅族, 庫斯)
- **Topic 1** (7.7% prevalence): Elderly community stories (母親, 老人, 百歲, 人瑞)
- **Topic 2** (7.7% prevalence): Daily life & officials (族人, 生活, 縣長)

**Insight**: Topic 0 appears most frequently → common across documents

---

## When to Use Which?

### Use LSA for:
- ✅ Finding **distinctive patterns**
- ✅ Document **similarity** search
- ✅ Fast, **deterministic** results
- ✅ **Variance-based** interpretation
- ✅ Small corpora (< 10,000 docs) ← **Your case**

### Use LDA for:
- ✅ Discovering **content themes**
- ✅ **Probabilistic** interpretation
- ✅ "Document is X% topic A, Y% topic B"
- ✅ Topic **coherence** metrics
- ✅ Larger corpora (> 1,000 docs)

---

## Why Use Both? (Your Project)

**Cross-Validation**: Both methods found indigenous cultural context as baseline ✅

**Complementary Views**:
- LSA: "What separates incident reports from cultural news?" → Topic 1 (13% var)
- LDA: "What themes exist in the corpus?" → Multiple topic mixtures

**Research Paper**: Report both for robust, validated findings

---

## Critical Implementation Detail

### The Input Matrix Makes ALL the Difference

**LSA uses TF-IDF**:
```
TF-IDF(警方, Doc2) = 0.456  ← High (rare term, appears in Doc2)
TF-IDF(部落, Doc2) = 0.089  ← Low (common term, appears everywhere)
```
→ Emphasizes distinctive terms like "警方" (police)

**LDA uses Raw Counts**:
```
Count(警方, Doc2) = 8 times
Count(部落, Doc2) = 2 times
```
→ Treats all terms equally, preserves frequency

**Result**: LSA identifies "警方" topics as distinctive, LDA sees them as less prevalent.

---

## LDA Inference Method (Your Implementation)

**Question**: Does LDA use EM or Gibbs Sampling?

**Answer**: Your LDA uses **Stochastic Variational Inference (SVI)** ✅

```python
# From your code (src/lsa_markus/pipelines/lda_analysis/nodes.py)
lda_model = LatentDirichletAllocation(
    learning_method='online',  # ← SVI, NOT EM or Gibbs Sampling
    batch_size=128,
    max_iter=100
)
```

**Comparison of LDA Inference Methods**:

| Method | Used? | Speed | Accuracy | Notes |
|--------|-------|-------|----------|-------|
| **Online (SVI)** | ✅ YES | ⚡⚡⚡ | ✓ Good | Your choice - processes mini-batches |
| **Batch (VB-EM)** | ❌ No | ⚡⚡ | ✓✓ Better | Alternative: `learning_method='batch'` |
| **Gibbs Sampling** | ❌ No | ⚡ | ✓✓✓ Best | Not available in scikit-learn |

**For your corpus (13-25 docs)**: Online mode works fine; batch mode might be marginally better.

---

## Quick Comparison Command

```bash
python src/compare_lsa_lda.py
```

**Output**:
- `data/08_reporting/lsa_vs_lda_comparison.csv`
- `data/08_reporting/lsa_vs_lda_comparison_2006_2015.png`
- `data/08_reporting/lsa_vs_lda_comparison_2016_2025.png`

---

## Bottom Line

For your research analyzing Jianshi Township news:

1. **LSA** efficiently found the most distinctive pattern (safety incidents, 13% variance)
2. **LDA** discovered interpretable content themes (elderly stories, daily life)
3. **Both** validated the common background (indigenous cultural context)

→ Using both methods strengthens your analysis ✅

---

**Full Details**: See [LSA_vs_LDA_COMPARISON.md](LSA_vs_LDA_COMPARISON.md)
