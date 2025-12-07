# LSA-Markus: Topic Modeling Pipeline

This Kedro project implements Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA) for analyzing news articles about Jianshi Township (尖石鄉) across two time periods: 2006-2015 and 2016-2025.

## Research Objectives

1. Apply topic modeling techniques to automatically identify complex co-occurrence relationships between words
2. Extract main topics of media attention regarding Jianshi Township
3. Compare topic evolution across time periods (2006-2015 vs. 2016-2025)
4. Analyze consistency and differences between media representation and local government development priorities

## Project Structure

```
lsa-markus/
├── conf/
│   └── base/
│       ├── catalog.yml          # Data catalog definitions
│       └── parameters.yml       # Pipeline parameters
├── data/
│   ├── 01_raw/                  # Raw corpus JSON files
│   ├── 02_intermediate/         # Preprocessed data
│   ├── 03_primary/              # TF-IDF matrices
│   ├── 06_models/               # Trained models
│   ├── 07_model_output/         # Model results
│   └── 08_reporting/            # Final reports and visualizations
├── src/
│   └── lsa_markus/
│       └── pipelines/
│           ├── data_processing/ # Text preprocessing and TF-IDF
│           ├── lsa_analysis/    # LSA implementation
│           └── lda_analysis/    # LDA implementation
└── corpus/                      # Original corpus files
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify the setup:
```bash
kedro info
```

## Data Preparation

The corpus data should be in JSON format with the following structure:
```json
[
  {
    "news_date": "YYYY/MM/DD",
    "news_title": "Article title",
    "news_content": "Article content..."
  }
]
```

Raw data files:
- `data/01_raw/2006-2015.json` - News from 2006-2015
- `data/01_raw/2016~2025.json` - News from 2016-2025

## Running the Pipeline

### Run the complete pipeline:
```bash
kedro run
```

### Run specific pipelines:

1. **Data Processing Only:**
```bash
kedro run --pipeline=data_processing
```

2. **LSA Analysis:**
```bash
kedro run --pipeline=lsa
```

3. **LDA Analysis:**
```bash
kedro run --pipeline=lda
```

### Run specific nodes:
```bash
kedro run --node=preprocess_corpus_2006_2015
```

## Pipeline Components

### 1. Data Processing Pipeline

**Purpose:** Preprocess Chinese text and create TF-IDF matrices

**Steps:**
1. Text preprocessing with jieba tokenization
2. Stopword removal
3. TF-IDF matrix creation
4. Corpus combination for comparison

**Key Parameters** (`conf/base/parameters.yml`):
- `preprocessing.remove_stopwords`: Enable/disable stopword removal
- `tfidf.max_features`: Maximum number of features (default: 1000)
- `tfidf.min_df`: Minimum document frequency (default: 2)
- `tfidf.max_df`: Maximum document frequency (default: 0.8)

### 2. LSA Analysis Pipeline

**Purpose:** Discover latent semantic dimensions through SVD

**Three Key Steps:**
1. **Term-Document Matrix:** Converts text to TF-IDF weighted matrix
2. **Singular Value Decomposition (SVD):** Reduces dimensionality to reveal latent concepts
3. **Semantic Space Construction:** Measures similarity in concept space using cosine similarity

**Key Parameters:**
- `lsa.n_components`: Number of topics/dimensions (default: 10)
- `lsa.algorithm`: 'randomized' or 'arpack'
- `topic_interpretation.n_top_words`: Top words per topic (default: 15)

**Outputs:**
- `lsa_results_2006_2015.pkl` - Complete LSA results for first period
- `lsa_results_2016_2025.pkl` - Complete LSA results for second period
- `lsa_topic_comparison.csv` - Topic comparison across periods
- `lsa_topic_terms_*.csv` - Top terms for each topic

### 3. LDA Analysis Pipeline

**Purpose:** Probabilistic topic modeling

**Model Assumptions:**
1. Each document is a mixture of topics
2. Each topic is a distribution over words

**Key Parameters:**
- `lda.n_topics`: Number of topics (default: 10)
- `lda.max_iter`: Maximum iterations (default: 100)
- `lda.learning_method`: 'online' or 'batch'

**Outputs:**
- `lda_results_2006_2015.pkl` - Complete LDA results for first period
- `lda_results_2016_2025.pkl` - Complete LDA results for second period
- `lda_topic_comparison.csv` - Topic comparison across periods
- `lda_topic_terms_*.csv` - Top terms for each topic

## Visualizing the Pipeline

```bash
kedro viz
```

This opens an interactive visualization showing:
- Pipeline structure
- Data flow
- Node dependencies

## Configuration

### Adjusting Parameters

Edit `conf/base/parameters.yml` to customize:

```yaml
# Example: Change number of topics
lsa:
  n_components: 15  # Increase from 10 to 15

lda:
  n_topics: 15
```

### Adding Custom Stopwords

Create a stopwords file and reference it:

```yaml
preprocessing:
  use_custom_stopwords: true
  custom_stopwords_path: "conf/base/chinese_stopwords.txt"
```

## Interpreting Results

### LSA Results

Access LSA results programmatically:

```python
import pickle

# Load results
with open('data/07_model_output/lsa_results_2006_2015.pkl', 'rb') as f:
    results = pickle.load(f)

# View topic terms
print(results['topic_terms'])

# View explained variance
print(f"Total variance explained: {results['total_variance_explained']:.2%}")

# View document-topic distributions
print(results['doc_topics'].head())
```

### LDA Results

```python
# Load LDA results
with open('data/07_model_output/lda_results_2006_2015.pkl', 'rb') as f:
    results = pickle.load(f)

# View topic terms with probabilities
print(results['topic_terms'])

# View model quality metrics
print(f"Perplexity: {results['perplexity']}")
print(f"Log Likelihood: {results['log_likelihood']}")
```

## Comparison Analysis

The pipeline automatically generates comparison reports:

1. **LSA Topic Comparison** (`lsa_topic_comparison.csv`):
   - Shows similar topics across time periods
   - Cosine similarity scores
   - Top terms for aligned topics

2. **LDA Topic Comparison** (`lda_topic_comparison.csv`):
   - Topic prevalence in each period
   - Evolution of topic importance
   - Top terms for each period's topics

## Troubleshooting

### Memory Issues

If processing large corpora:
1. Reduce `max_features` in parameters
2. Increase `min_df` to filter rare terms
3. Use `learning_method: 'online'` for LDA

### Empty Results

If topics appear empty:
1. Check stopwords aren't too aggressive
2. Verify corpus has sufficient documents
3. Adjust `min_df` and `max_df` thresholds

### jieba Installation Issues

```bash
pip install --upgrade jieba
```

## Next Steps

1. **Fine-tune Parameters:** Experiment with different numbers of topics
2. **Add Visualizations:** Create word clouds, topic networks
3. **Sentiment Analysis:** Add emotion detection to topics
4. **Time Series Analysis:** Track topic prevalence over time
5. **Export for Reporting:** Generate visualizations for stakeholders

## Contact & Support

For issues or questions about this pipeline, please refer to the research documentation in `notes/research_goal.md`.
