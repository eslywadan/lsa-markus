# Cosine Similarity is Part of the LSA Process ✅

Cosine similarity is a standard component of the complete LSA methodology.e functions.

## The Complete LSA Process (Theory vs this Implementation)
## Standard LSA Workflow (Theoretical)

```
Step 1: TF-IDF Vectorization
   ↓
Step 2: Singular Value Decomposition (SVD)
   → Document-Topic Matrix (U × Σ)
   → Topic-Term Matrix (V^T)
   ↓
Step 3: Semantic Space Representation ← You are here
   → Documents represented as vectors in reduced space
   ↓
Step 4: Cosine Similarity Calculation ⭐ THE MISSING STEP
   → Calculate similarity between document vectors
   → Used for: document retrieval, clustering, comparison
```
Current Implementation
Pipeline Structure
```
From pipeline.py:21-85:
Node 1: perform_lsa() 
   ↓ Outputs: lsa_model, doc_topic_matrix, topic_term_matrix
   
Node 2: create_lsa_results()
   ↓ Calls: calculate_document_similarities()
   ↓ Outputs: doc_topics DataFrame with dominant topics
   
Node 3: compare_time_periods()
   ↓ Outputs: Topic comparison across periods
```
What calculate_document_similarities() Actually Does
From nodes.py:83-108:
```python
def calculate_document_similarities(
    doc_topic_matrix: np.ndarray,
    preprocessed_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate document similarities in semantic space.
    """
    # Create DataFrame with document-topic distributions
    topic_columns = [f'topic_{i}' for i in range(doc_topic_matrix.shape[1])]
    doc_topics_df = pd.DataFrame(doc_topic_matrix, columns=topic_columns)
    
    # Add document metadata
    doc_topics_df['news_date'] = preprocessed_df['news_date'].values
    doc_topics_df['news_title'] = preprocessed_df['news_title'].values
    
    # Calculate dominant topic (argmax)
    doc_topics_df['dominant_topic'] = doc_topic_matrix.argmax(axis=1)
    
    return doc_topics_df
```
Note: Despite the function name "calculate_document_similarities", it does NOT actually calculate cosine similarity. It only:
Creates a DataFrame with topic distributions
Finds the dominant topic (argmax)
Where Cosine Similarity SHOULD Be Used (But Isn't)

## Typical LSA Applications Using Cosine Similarity
### 1. Document-to-Document Similarity

```python
from scipy.spatial.distance import cosine
```

 After SVD, in the reduced semantic space:
doc_vectors = doc_topic_matrix  # Shape: (n_docs, n_components)

#### Calculate similarity between doc i and doc j
similarity_ij = 1 - cosine(doc_vectors[i], doc_vectors[j])
#### Range: [0, 1] where 1 = identical, 0 = orthogonal
Use case: Find similar news articles, detect duplicates, recommend related content
### 2. Query-to-Document Similarity (Information Retrieval)
#### Transform a new query into the semantic space
query_vector = lsa_model.transform(tfidf_vectorizer.transform([query_text]))

#### Find most similar documents
similarities = [1 - cosine(query_vector[0], doc_vectors[i]) 
                for i in range(len(doc_vectors))]
most_similar_docs = np.argsort(similarities)[-5:]  # Top 5
Use case: Search engine, document retrieval
### 3. Topic-to-Topic Similarity
topic_vectors = topic_term_matrix  # Shape: (n_components, n_features)

#### Similarity between topics

topic_sim = 1 - cosine(topic_vectors[0], topic_vectors[1])
Use case: Identify related topics, topic clustering
Why Cosine Similarity for LSA?
Mathematical Justification
After SVD, documents are represented as vectors in a k-dimensional semantic space:
Document d → Vector: [w₁, w₂, ..., wₖ] in ℝᵏ
Cosine similarity measures the angle between vectors:
cos(θ) = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product
- ||A|| = Euclidean norm (magnitude)
- θ = angle between vectors
Advantages:
✅ Magnitude-independent: Focuses on direction, not length
✅ Works with negative values: LSA weights can be negative
✅ Normalized: Range [−1, 1], where 1 = same direction
✅ Efficient: Single dot product + norms

## Why not Euclidean distance?
Euclidean distance is sensitive to vector magnitude
In LSA, magnitude doesn't reflect semantic importance
Cosine focuses on the pattern of topic weights, not their scale
Your Implementation Analysis
What You Have ✅
SVD transformation → Reduced semantic space
Document-topic matrix → Vector representations
Dominant topic assignment → Simple classification
Topic prevalence → Average weights
What's Missing ❌
No pairwise document similarity matrix
No similarity-based document retrieval
No similarity-based clustering
Imported cosine function is unused
When You Would Add Cosine Similarity
### Scenario 1: Document Similarity Analysis
If you wanted to find which news articles are semantically similar:
```python
def calculate_pairwise_similarities(doc_topic_matrix: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between all document pairs."""
    from scipy.spatial.distance import cosine
    
    n_docs = doc_topic_matrix.shape[0]
    similarity_matrix = np.zeros((n_docs, n_docs))
    
    for i in range(n_docs):
        for j in range(i, n_docs):
            sim = 1 - cosine(doc_topic_matrix[i], doc_topic_matrix[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetric
    
    return similarity_matrix
```

### Scenario 2: Document Retrieval
If you wanted to search for articles similar to a given article:

```python
def find_similar_documents(
    target_doc_idx: int,
    doc_topic_matrix: np.ndarray,
    top_k: int = 5
) -> list:
    """Find top-k most similar documents to target."""
    from scipy.spatial.distance import cosine
    
    target_vector = doc_topic_matrix[target_doc_idx]
    
    similarities = []
    for i in range(len(doc_topic_matrix)):
        if i != target_doc_idx:
            sim = 1 - cosine(target_vector, doc_topic_matrix[i])
            similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]
```
## Summary
Aspect	Status in Your Pipeline
Cosine similarity imported?	✅ YES (nodes.py:14)
Cosine similarity used?	❌ NO (imported but not called)
Is it part of complete LSA?	✅ YES (standard for similarity analysis)
Where would it go?	After SVD, in calculate_document_similarities()
Your current approach	Only calculates dominant topic (argmax)
Standard LSA approach	Calculate pairwise cosine similarities
The Complete LSA Process (With Cosine Similarity)

```
┌────────────────────────────────────────--─┐
│  1. TF-IDF Vectorization                  │
│     Input: Raw text                       │
│     Output: TF-IDF matrix (n_docs × vocab)│
└─────────────────────────────────────--────┘
                    ↓
┌─────────────────────────────────────--────┐
│  2. Truncated SVD                         │  ← YOUR IMPLEMENTATION ✅
│     Input: TF-IDF matrix                  │
│     Output: doc_topic_matrix (n_docs × k) │
│             topic_term_matrix (k × vocab) │
└────────────────────────────────────────--─┘
                    ↓
┌─────────────────────────────────────────┐
│  3. Semantic Space Representation       │  ← YOU ARE HERE
│     Documents as k-dimensional vectors  │
└─────────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────--──┐
│  4. Cosine Similarity Calculation         │  ← MISSING IN YOUR CODE ❌
│     Between document vectors              │     (imported but not used)
│     similarity(i,j) = 1 - cosine(v_i, v_j)│
└────────────────────────────────────────--─┘
                    ↓
┌─────────────────────────────────────────┐
│  5. Applications                        │
│     - Document retrieval                │
│     - Similarity search                 │
│     - Clustering                        │
│     - Recommendation                    │
└─────────────────────────────────────────┘
```




## Added cosine similarity calculations to LSA pipeline
Here's what was implemented:
### What Was Added

1. Two Cosine Similarity Scenarios

- Scenario 1: Pairwise Document Similarity Matrix
Function: calculate_pairwise_similarities() in nodes.py:83-123
Output: Full similarity matrix (n_docs × n_docs)
Shows: Cosine similarity between every pair of documents
Use case: Finding all document relationships, clustering, heatmap visualization
- Scenario 2: Top-K Similar Documents
Function: find_similar_documents() in nodes.py:126-170
Output: For each document, find the top-5 most similar documents
Shows: Target document → most similar documents with similarity scores
Use case: Document recommendation, finding related articles
Generated Output Files (in data/08_reporting/)

```
2006-2015 Period:
lsa_pairwise_similarities_2006_2015.csv (3.8 KB)
13×13 similarity matrix for 13 documents
Each cell shows cosine similarity between two documents
lsa_similar_documents_2006_2015.csv (8.8 KB)
65 rows (13 documents × 5 similar docs each)
Columns: target_idx, target_date, target_title, similar_idx, similar_date, similar_title, similarity
```
```
2016-2025 Period:
lsa_pairwise_similarities_2016_2025.csv (3.6 KB)
12×12 similarity matrix for 12 documents
lsa_similar_documents_2016_2025.csv (11 KB)
60 rows (12 documents × 5 similar docs each)
Example Output: Similar Documents
From lsa_similar_documents_2006_2015.csv:
target_title,similar_title,similarity
```
```
新竹縣油羅溪尖石公、尖石母，露全貌,泰雅族的遷徙故事，司馬庫斯鮮讀,0.377
新竹縣油羅溪尖石公、尖石母，露全貌,疑會車熄火到退嚕…巴士墜谷，13死10傷,0.347
黑暗部落成最賺錢部落，司馬庫斯創造觀光奇蹟,司馬庫斯護林，青壯守夜,0.336
```
Interpretation:
Article about "尖石公、尖石母" is most similar (0.377) to the article about "泰雅族的遷徙故事"

Articles about 司馬庫斯 are semantically related (similarity 0.336)

### Pipeline Changes
```
Updated Files:
nodes.py
Added calculate_pairwise_similarities() (lines 83-123)
Added find_similar_documents() (lines 126-170)
Updated create_lsa_results() to call these functions (lines 233-237)
Added extract_pairwise_similarities() (lines 259-269)
Added extract_similar_documents() (lines 272-282)
pipeline.py
Added 4 new nodes to extract and save similarity data (lines 88-115)
Updated imports (lines 11-12)
catalog.yml
Added 4 new output datasets (lines 128-143):
lsa_pairwise_similarities_2006_2015
lsa_pairwise_similarities_2016_2025
lsa_similar_documents_2006_2015
lsa_similar_documents_2016_2025
parameters.yml
Added similarity.top_k: 5 parameter (lines 54-56)
```
Pipeline Flow (Updated)
```
┌─────────────────────────────────────────┐
│  1. TF-IDF Vectorization                │
│     Input: Preprocessed text            │
│     Output: TF-IDF matrix               │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  2. Truncated SVD (LSA)                 │
│     Input: TF-IDF matrix                │
│     Output: doc_topic_matrix (n×k)      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  3. Create LSA Results                  │
│     → Extract topic terms               │
│     → Calculate document topics         │
│     → **NEW: Cosine Similarities** ⭐   │
└─────────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────---┐
│  4. Export Similarity Data ⭐ NEW          │
│     → Pairwise similarity matrix          │
│     → Top-k similar documents             │
└────────────────────────────────────────--─┘
```
How to Use
Run the complete LSA pipeline:
kedro run --pipeline=lsa
Access the similarity data:
import pandas as pd

## Load pairwise similarities
pairwise = pd.read_csv('data/08_reporting/lsa_pairwise_similarities_2006_2015.csv', index_col=0)

## Load similar documents
similar_docs = pd.read_csv('data/08_reporting/lsa_similar_documents_2006_2015.csv')

## Find documents similar to a specific article
```
target_article = "司馬庫斯創造觀光奇蹟"
matches = similar_docs[similar_docs['target_title'].str.contains(target_article)]
print(matches[['similar_title', 'similarity']].head())
```
Key Implementation Details
Cosine Similarity Formula Used:
```
from scipy.spatial.distance import cosine


# Cosine similarity = 1 - cosine distance
similarity = 1 - cosine(vector_i, vector_j)

# Range: [-1, 1] where:
#   1.0 = identical documents
#   0.0 = orthogonal (no similarity)
#  -1.0 = opposite direction (rare in LSA)
```
### Configuration:
```
Default top_k: 5 (configurable in parameters.yml)
Input: Document vectors in LSA semantic space (after SVD)
Output format: CSV files for easy analysis
Success Metrics
✅ 2006-2015 period: 13 documents → 13×13 matrix + 65 similarity pairs
✅ 2016-2025 period: 12 documents → 12×12 matrix + 60 similarity pairs
✅ Pipeline execution: 0.1 seconds (very fast!)
✅ All outputs saved to data/08_reporting/
```
