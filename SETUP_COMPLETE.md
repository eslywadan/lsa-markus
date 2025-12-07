# LSA-Markus Pipeline Setup Complete ✅

## Summary

A complete Kedro pipeline has been successfully set up for performing Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA) on the Jianshi Township news corpus.

## What Was Created

### 1. Project Structure
```
lsa-markus/
├── conf/base/
│   ├── catalog.yml           ✅ Data catalog with all datasets
│   └── parameters.yml        ✅ Pipeline parameters (LSA & LDA)
│
├── src/lsa_markus/pipelines/
│   ├── data_processing/      ✅ Text preprocessing & TF-IDF
│   ├── lsa_analysis/         ✅ LSA implementation (SVD-based)
│   └── lda_analysis/         ✅ LDA implementation (probabilistic)
│
├── data/
│   ├── 01_raw/               ✅ Corpus JSON files ready
│   ├── 02_intermediate/      ✅ For preprocessed data
│   ├── 03_primary/           ✅ For TF-IDF matrices
│   ├── 06_models/            ✅ For trained models
│   ├── 07_model_output/      ✅ For analysis results
│   └── 08_reporting/         ✅ For final reports
│
├── corpus/                   ✅ Original source files
│   ├── 2006-2015.json
│   └── 2016~2025.json
│
├── requirements.txt          ✅ All dependencies listed
├── README_PIPELINE.md        ✅ Comprehensive documentation
└── pyproject.toml            ✅ Kedro configuration
```

### 2. Pipelines Implemented

#### A. Data Processing Pipeline (`data_processing`)
- **Nodes:**
  - `preprocess_corpus_2006_2015` - Clean and tokenize 2006-2015 corpus
  - `preprocess_corpus_2016_2025` - Clean and tokenize 2016-2025 corpus
  - `combine_corpora` - Merge for temporal comparison
  - `create_tfidf_2006_2015` - Generate TF-IDF matrix for period 1
  - `create_tfidf_2016_2025` - Generate TF-IDF matrix for period 2

- **Features:**
  - Chinese text tokenization with jieba
  - Customizable stopword filtering
  - TF-IDF vectorization
  - Support for unigrams and bigrams

#### B. LSA Analysis Pipeline (`lsa`)
- **Nodes:**
  - `perform_lsa_2006_2015` - Apply Truncated SVD to period 1
  - `perform_lsa_2016_2025` - Apply Truncated SVD to period 2
  - `create_lsa_results_2006_2015` - Generate comprehensive results
  - `create_lsa_results_2016_2025` - Generate comprehensive results
  - `compare_lsa_topics` - Cross-temporal topic comparison

- **Three Key Steps:**
  1. **Term-Document Matrix**: TF-IDF weighted representation
  2. **SVD**: Dimensionality reduction to reveal latent concepts
  3. **Semantic Space**: Cosine similarity for semantic relationships

#### C. LDA Analysis Pipeline (`lda`)
- **Nodes:**
  - `create_doc_term_matrix_2006_2015` - Build count matrix for period 1
  - `create_doc_term_matrix_2016_2025` - Build count matrix for period 2
  - `perform_lda_2006_2015` - Train LDA model on period 1
  - `perform_lda_2016_2025` - Train LDA model on period 2
  - `create_lda_results_2006_2015` - Generate comprehensive results
  - `create_lda_results_2016_2025` - Generate comprehensive results
  - `compare_lda_topics` - Cross-temporal topic comparison

- **Probabilistic Approach:**
  - Documents as mixtures of topics
  - Topics as distributions over words
  - Perplexity and log-likelihood metrics

### 3. Configuration Files

#### parameters.yml
```yaml
# Text preprocessing
preprocessing:
  remove_stopwords: true
  min_word_length: 2
  max_word_length: 10

# TF-IDF settings
tfidf:
  max_features: 1000
  min_df: 2
  max_df: 0.8
  ngram_range: [1, 2]

# LSA settings
lsa:
  n_components: 10
  algorithm: "randomized"

# LDA settings
lda:
  n_topics: 10
  max_iter: 100
  learning_method: "online"

# Topic interpretation
topic_interpretation:
  n_top_words: 15
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- kedro==1.1.1
- kedro-datasets>=9.0.0
- pandas, numpy, scipy
- scikit-learn
- jieba (Chinese tokenization)
- matplotlib, seaborn

### 2. Verify Setup
```bash
kedro info
```

### 3. Run the Pipeline

**Full pipeline (all steps):**
```bash
kedro run
```

**Individual pipelines:**
```bash
# Preprocessing only
kedro run --pipeline=data_processing

# LSA only (requires preprocessing first)
kedro run --pipeline=lsa

# LDA only (requires preprocessing first)
kedro run --pipeline=lda
```

## Expected Outputs

### After Running the Pipeline:

1. **Preprocessing Results** (`data/02_intermediate/`)
   - `preprocessed_2006_2015.pkl`
   - `preprocessed_2016~2025.pkl`
   - `combined_corpus.pkl`

2. **TF-IDF Matrices** (`data/03_primary/`)
   - `tfidf_matrix_2006_2015.pkl`
   - `tfidf_matrix_2016~2025.pkl`
   - `tfidf_vectorizer.pkl`

3. **LSA Results** (`data/07_model_output/`)
   - `lsa_results_2006_2015.pkl` - Topic terms, document distributions, explained variance
   - `lsa_results_2016_2025.pkl` - Same for second period

4. **LDA Results** (`data/07_model_output/`)
   - `lda_results_2006_2015.pkl` - Topic terms, probabilities, perplexity
   - `lda_results_2016_2025.pkl` - Same for second period

5. **Comparison Reports** (`data/08_reporting/`)
   - `lsa_topic_comparison.csv` - Similar topics across periods
   - `lda_topic_comparison.csv` - Topic prevalence evolution
   - `lsa_topic_terms_*.csv` - Top terms per topic
   - `lda_topic_terms_*.csv` - Top terms with probabilities

## Accessing Results

### Python Example:
```python
import pickle
import pandas as pd

# Load LSA results
with open('data/07_model_output/lsa_results_2006_2015.pkl', 'rb') as f:
    lsa_results = pickle.load(f)

# View topics
print(lsa_results['topic_terms'])

# View explained variance
print(f"Variance explained: {lsa_results['total_variance_explained']:.2%}")

# View document-topic distributions
print(lsa_results['doc_topics'].head())

# Load comparison report
comparison = pd.read_csv('data/08_reporting/lsa_topic_comparison.csv')
print(comparison)
```

## Next Steps

1. **Run the Pipeline:**
   ```bash
   kedro run
   ```

2. **Explore Results:**
   - Check topic terms in reporting directory
   - Analyze temporal evolution patterns
   - Identify dominant themes for each period

3. **Fine-tune Parameters:**
   - Adjust number of topics (`n_components`, `n_topics`)
   - Modify TF-IDF settings (`max_features`, `min_df`)
   - Customize stopwords

4. **Visualize:**
   ```bash
   kedro viz  # Interactive pipeline visualization
   ```

5. **Add Custom Analysis:**
   - Create new nodes for sentiment analysis
   - Add word cloud visualizations
   - Implement topic evolution tracking

## Documentation

- **Full Documentation**: `README_PIPELINE.md`
- **Research Goals**: `notes/research_goal.md`
- **Data Conversion Script**: `src/convert_to_json.py`

## Troubleshooting

### If pipeline fails:
1. Check data exists in `data/01_raw/`
2. Verify all dependencies installed: `pip install -r requirements.txt`
3. Check configuration: `kedro catalog list`
4. Run verbose: `kedro run --verbose`

### Common Issues:
- **Missing jieba**: `pip install jieba`
- **Memory errors**: Reduce `max_features` in parameters
- **Empty topics**: Adjust `min_df` and check stopwords

## Research Alignment

This pipeline directly addresses your research objectives:

1. ✅ **Objective 1.2.1**: LSA/LDA automatically identify word co-occurrence patterns and construct semantic space
2. ✅ **Objective 1.2.2**: Extract main topics with top terms and analyze topic content
3. ✅ **Objective 1.2.3**: Built-in temporal comparison (2006-2015 vs 2016-2025)
4. ✅ **Objective 1.2.4**: Comparison reports ready for policy analysis and communication strategy recommendations

## Project Status

🎉 **Setup Complete and Ready to Run!**

All pipeline components are implemented, configured, and ready for execution. The corpus data is staged in `data/01_raw/` and the pipeline will automatically process everything when you run `kedro run`.
