# LSA Topic Analysis - Quick Reference Guide

## 📊 Your Validated Findings

### ✅ Pattern Confirmed in Both Periods

**2006-2015:**
- **Background Context (Topic 0)**: Explained variance 1.22%, Prevalence 0.44, Coverage 13/13 docs
  - Top terms: 部落, 泰雅族, 原住民, 庫斯, 傳統, 泰雅, 文化
  - Interpretation: Common indigenous cultural context shared across all documents

- **Most Distinctive (Topic 1)**: Explained variance 13.08%, Prevalence 0.26, Coverage 6/13 docs
  - Top terms: 警方, 現場, 樹林, 打獵
  - Interpretation: Safety incidents - separates incident reports from other news

**2016-2025:**
- **Background Context (Topic 0)**: Explained variance 1.82%, Prevalence 0.46, Coverage 12/12 docs
  - Top terms: 部落, 公所, 表示, 縣府, 遊客, 控溪, 安全
  - Interpretation: Government administration and tourism management baseline

- **Most Distinctive (Topic 1)**: Explained variance 13.09%, Prevalence 0.28, Coverage 8/12 docs
  - Top terms: 控溪 部落, 撤離, 溢流, 山頭
  - Interpretation: Emergency evacuations at Konsi tribe

---

## 📝 How to Report in Your Research Paper

### Option 1: Methodology Section

```
Topic Selection Methodology

We identified background and distinctive topics through multi-criteria analysis:
(1) explained variance ratio, indicating the discriminative power of each topic;
(2) average prevalence across documents, measuring how commonly the topic appears;
and (3) semantic coherence of top terms.

Topic 0 emerged as background context in both periods, characterized by low
explained variance (2006-2015: 1.22%; 2016-2025: 1.82%) and high prevalence
(2006-2015: 0.44; 2016-2025: 0.46), appearing consistently across all documents.

Topic 1 represented the primary distinctive pattern in both periods, with high
explained variance (2006-2015: 13.08%; 2016-2025: 13.09%), effectively separating
documents containing specialized themes from the general corpus.
```

### Option 2: Results Section

```
Period 1 (2006-2015):

Topic 0 (explained variance: 1.22%, prevalence: 0.44) represents the common
background context shared across all 13 documents, focusing on indigenous tribal
culture (部落, 泰雅族, 原住民, 文化). This topic's low variance but high
prevalence indicates it captures baseline themes present throughout the period.

Topic 1 (explained variance: 13.08%) captures the most distinctive pattern,
primarily related to safety incidents involving police and hunting activities
(警方, 現場, 打獵). This topic effectively separates incident reports (6/13
documents) from other news types, demonstrating strong discriminative power.

Period 2 (2016-2025):

Topic 0 (explained variance: 1.82%, prevalence: 0.46) represents the common
background context, with emphasis shifting toward government administration and
tourism management (部落, 公所, 縣府, 遊客). The consistent low variance and
high prevalence across all 12 documents indicates stable baseline themes.

Topic 1 (explained variance: 13.09%) captures emergency evacuation events at
Konsi tribe (控溪 部落, 撤離, 溢流), appearing strongly in 8/12 documents.
This topic's high explained variance demonstrates its effectiveness in
differentiating crisis-related coverage from routine reporting.
```

---

## 🔍 Key Concepts

### Explained Variance
- **What it measures**: How much variation in the data each topic captures
- **High variance (Topic 1: ~13%)**: Strong discriminative power, separates documents effectively
- **Low variance (Topic 0: ~1-2%)**: Captures consistent patterns across documents

### Prevalence
- **What it measures**: Average absolute weight of a topic across all documents
- **High prevalence (Topic 0: 0.44-0.46)**: Topic appears commonly/ubiquitously
- **Moderate prevalence (Topic 1: 0.26-0.28)**: Topic appears in subset of documents

### Document Coverage
- **Background topic (13/13, 12/12)**: Present in ALL documents
- **Distinctive topic (6/13, 8/12)**: Present in specialized subset

---

## 🛠️ Analysis Tools Available

### 1. View Topic-Term Matrix
```bash
python src/view_topic_term_matrix.py
```
- Interactive viewer for exploring topic-term weights
- Shows top terms for any topic
- Can export to CSV

### 2. Verify Topic Selection
```bash
python src/analyze_topic_selection.py
```
- Comprehensive validation of topic selection methodology
- Generates verification visualizations
- Creates comparison tables across periods

### 3. Visualize Results
```bash
python src/visualize_pkl.py
```
- Creates publication-ready visualizations
- Shows explained variance, prevalence, comparisons

### 4. Jupyter Notebook
```bash
kedro jupyter notebook
```
- Interactive exploration
- Access: http://localhost:8888/tree?token=cdc590761f8955379cf26b13466d95f49bd621bdc1a5d1d1
- Notebook: `notebooks/01_visualize_results.ipynb`

---

## 📁 Generated Files

### Analysis Results
- `data/08_reporting/topic_selection_comparison.csv` - Cross-period comparison table
- `data/08_reporting/topic_selection_verification_2006_2015.png` - Verification charts for 2006-2015
- `data/08_reporting/topic_selection_verification_2016~2025.png` - Verification charts for 2016-2025
- `data/08_reporting/lsa_visualization.png` - Overall LSA visualizations
- `data/08_reporting/comparison_visualization.png` - Period comparisons

### Exported Data
- `data/08_reporting/topic_term_matrix_2006_2015.csv` - Full topic-term matrix
- `data/08_reporting/lsa_topic_comparison.csv` - Topic comparison data
- `data/08_reporting/topic_term_matrix_heatmap.png` - Heatmap of top terms

---

## ✅ Best Practices (From The Analysis)

### The Topic Selection Methodology is VALID because:

1. ✅ **Clear domain context**: All articles about Jianshi Township
2. ✅ **Shared background**: Indigenous culture and local government appear consistently
3. ✅ **Distinctive events**: Emergency incidents and specific tribal events separate documents
4. ✅ **Empirically verified**: Pattern confirmed in both time periods
5. ✅ **Semantic coherence**: Terms make intuitive sense for their roles

### When Reporting, DO:
- ✅ Mention multi-criteria selection (variance + prevalence + semantic coherence)
- ✅ Report specific metrics (variance %, prevalence values, coverage)
- ✅ Explain what each topic represents semantically
- ✅ Note that pattern was verified empirically

### When Reporting, DON'T:
- ❌ Claim it's a universal rule
- ❌ Use variance alone without checking prevalence
- ❌ Assume without verification

---

## 🎯 Quick Decision Flowchart

```
Need to select topics for interpretation?
    │
    ├─→ Check LOWEST variance topic:
    │   • Is prevalence > 0.3? ✓
    │   • Appears in most/all docs? ✓
    │   • Terms represent common context? ✓
    │   └─→ YES to all = Background context
    │
    └─→ Check HIGHEST variance topic:
        • Explained variance high (>10%)? ✓
        • Appears in subset of docs? ✓
        • Terms represent specific events? ✓
        └─→ YES to all = Distinctive pattern
```

---

## 📚 Additional Documentation

- **Methodology Details**: [notes/topic_selection_best_practices.md](topic_selection_best_practices.md)
- **Project README**: [../README.md](../README.md)

---

**Last Updated**: 2025-12-08
**Validated Periods**: 2006-2015, 2016-2025
**Validation Tool**: `src/analyze_topic_selection.py`

---

## 🆕 LSA vs LDA Comparison

### Quick Links:
- **Full Comparison**: [LSA_vs_LDA_COMPARISON.md](LSA_vs_LDA_COMPARISON.md) - Comprehensive algorithmic & pipeline comparison
- **Quick Summary**: [LSA_LDA_QUICK_SUMMARY.md](LSA_LDA_QUICK_SUMMARY.md) - One-page comparison
- **Pipeline Flow**: [PIPELINE_FLOW_COMPARISON.md](PIPELINE_FLOW_COMPARISON.md) - Visual pipeline diagrams
- **LDA Inference Method**: [LDA_INFERENCE_METHOD.md](LDA_INFERENCE_METHOD.md) - Detailed explanation of SVI vs EM vs Gibbs Sampling

### One-Sentence Difference:
**LSA** finds what makes documents **different** (discriminative patterns via variance).
**LDA** finds what documents **contain** (content themes via probabilities).

### Key Algorithmic Difference:
**LSA**: Uses Truncated SVD on TF-IDF matrix (deterministic, linear algebra)
**LDA**: Uses Stochastic Variational Inference on count matrix (probabilistic, Bayesian)

**Your LDA Implementation**: Uses `learning_method='online'` → Stochastic Variational Inference (SVI), NOT EM or Gibbs Sampling

### Run Comparison:
```bash
python src/compare_lsa_lda.py
```

### Your Results:
Both methods confirmed indigenous cultural context (部落, 泰雅族, 文化) as baseline, validating your corpus focus ✅
