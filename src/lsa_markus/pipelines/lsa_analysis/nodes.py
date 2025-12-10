"""
LSA (Latent Semantic Analysis) nodes.

Implements the three key steps of LSA:
1. Build term-document matrix (TF-IDF)
2. Perform Singular Value Decomposition (SVD)
3. Construct semantic space and calculate similarities
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD


def perform_lsa(
    tfidf_matrix: pd.DataFrame,
    parameters: Dict
) -> Tuple[TruncatedSVD, np.ndarray, np.ndarray]:
    """
    Perform Latent Semantic Analysis using Truncated SVD.

    Args:
        tfidf_matrix: TF-IDF matrix as DataFrame
        parameters: LSA parameters (n_components, algorithm, etc.)

    Returns:
        Tuple of (fitted LSA model, document-topic matrix, topic-term matrix)
    """
    # Initialize LSA model (TruncatedSVD)
    lsa_model = TruncatedSVD(
        n_components=parameters.get('n_components', 10),
        algorithm=parameters.get('algorithm', 'randomized'),
        n_iter=parameters.get('n_iter', 100),
        random_state=parameters.get('random_state', 42)
    )

    # Fit and transform: get document-topic matrix
    doc_topic_matrix = lsa_model.fit_transform(tfidf_matrix)

    # Get topic-term matrix (components)
    topic_term_matrix = lsa_model.components_

    return lsa_model, doc_topic_matrix, topic_term_matrix


def extract_topic_terms(
    lsa_model: TruncatedSVD,
    feature_names: list,
    n_top_words: int = 15
) -> pd.DataFrame:
    """
    Extract top terms for each LSA topic.

    Args:
        lsa_model: Fitted TruncatedSVD model
        feature_names: List of feature names (vocabulary)
        n_top_words: Number of top words to extract per topic

    Returns:
        DataFrame with topics and their top terms
    """
    topics_data = []

    for topic_idx, topic in enumerate(lsa_model.components_):
        # Get top term indices (both positive and negative weights)
        top_indices = topic.argsort()[-n_top_words:][::-1]

        top_terms = [feature_names[i] for i in top_indices]
        top_weights = [topic[i] for i in top_indices]

        topics_data.append({
            'topic_id': topic_idx,
            'top_terms': ', '.join(top_terms),
            'term_weights': top_weights
        })

    return pd.DataFrame(topics_data)


def calculate_pairwise_similarities(
    doc_topic_matrix: np.ndarray,
    preprocessed_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate cosine similarity between all document pairs.

    Args:
        doc_topic_matrix: Document representations in topic space (n_docs × k)
        preprocessed_df: Original preprocessed DataFrame

    Returns:
        DataFrame with pairwise document similarities
    """
    n_docs = doc_topic_matrix.shape[0]
    similarity_matrix = np.zeros((n_docs, n_docs))

    # Calculate pairwise cosine similarities
    for i in range(n_docs):
        for j in range(i, n_docs):
            if i == j:
                sim = 1.0  # Document is identical to itself
            else:
                sim = 1 - cosine(doc_topic_matrix[i], doc_topic_matrix[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetric matrix

    # Create DataFrame with document titles as index/columns
    titles = preprocessed_df['news_title'].values
    dates = preprocessed_df['news_date'].values

    # Create labels with date and title
    labels = [f"{date} - {title[:50]}" for date, title in zip(dates, titles)]

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=labels,
        columns=labels
    )

    return similarity_df


def find_similar_documents(
    doc_topic_matrix: np.ndarray,
    preprocessed_df: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Find top-k most similar documents for each document.

    Args:
        doc_topic_matrix: Document representations in topic space
        preprocessed_df: Original preprocessed DataFrame
        top_k: Number of similar documents to find for each document

    Returns:
        DataFrame with each document and its most similar documents
    """
    n_docs = doc_topic_matrix.shape[0]
    titles = preprocessed_df['news_title'].values
    dates = preprocessed_df['news_date'].values

    results = []

    for target_idx in range(n_docs):
        target_vector = doc_topic_matrix[target_idx]

        # Calculate similarities to all other documents
        similarities = []
        for i in range(n_docs):
            if i != target_idx:
                sim = 1 - cosine(target_vector, doc_topic_matrix[i])
                similarities.append({
                    'target_idx': target_idx,
                    'target_date': dates[target_idx],
                    'target_title': titles[target_idx],
                    'similar_idx': i,
                    'similar_date': dates[i],
                    'similar_title': titles[i],
                    'similarity': sim
                })

        # Sort by similarity (descending) and take top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        results.extend(similarities[:top_k])

    return pd.DataFrame(results)


def calculate_document_similarities(
    doc_topic_matrix: np.ndarray,
    preprocessed_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate document similarities in semantic space.

    Args:
        doc_topic_matrix: Document representations in topic space
        preprocessed_df: Original preprocessed DataFrame

    Returns:
        DataFrame with document info and topic distributions
    """
    # Create DataFrame with document-topic distributions
    topic_columns = [f'topic_{i}' for i in range(doc_topic_matrix.shape[1])]
    doc_topics_df = pd.DataFrame(doc_topic_matrix, columns=topic_columns)

    # Add original document information
    doc_topics_df['news_date'] = preprocessed_df['news_date'].values
    doc_topics_df['news_title'] = preprocessed_df['news_title'].values

    # Calculate dominant topic for each document
    doc_topics_df['dominant_topic'] = doc_topic_matrix.argmax(axis=1)

    return doc_topics_df


def create_lsa_results(
    lsa_model: TruncatedSVD,
    doc_topic_matrix: np.ndarray,
    topic_term_matrix: np.ndarray,
    tfidf_matrix: pd.DataFrame,
    preprocessed_df: pd.DataFrame,
    n_top_words: int = 15,
    similarity_top_k: int = 5
) -> Dict:
    """
    Create comprehensive LSA results dictionary.

    Args:
        lsa_model: Fitted LSA model
        doc_topic_matrix: Document-topic matrix
        topic_term_matrix: Topic-term matrix
        tfidf_matrix: Original TF-IDF matrix
        preprocessed_df: Preprocessed corpus DataFrame
        n_top_words: Number of top words per topic
        similarity_top_k: Number of similar documents to find for each document

    Returns:
        Dictionary containing all LSA results including cosine similarities
    """
    feature_names = tfidf_matrix.columns.tolist()

    # Extract topic terms
    topic_terms_df = extract_topic_terms(lsa_model, feature_names, n_top_words)

    # Calculate document topic distributions
    doc_topics_df = calculate_document_similarities(doc_topic_matrix, preprocessed_df)

    # Calculate pairwise document similarities (Scenario 1)
    pairwise_similarities_df = calculate_pairwise_similarities(doc_topic_matrix, preprocessed_df)

    # Find similar documents for each document (Scenario 2)
    similar_docs_df = find_similar_documents(doc_topic_matrix, preprocessed_df, similarity_top_k)

    # Calculate explained variance
    explained_variance = lsa_model.explained_variance_ratio_

    results = {
        'model': lsa_model,
        'doc_topic_matrix': doc_topic_matrix,
        'topic_term_matrix': topic_term_matrix,
        'topic_terms': topic_terms_df,
        'doc_topics': doc_topics_df,
        'explained_variance': explained_variance,
        'total_variance_explained': explained_variance.sum(),
        'feature_names': feature_names,
        # New: Cosine similarity results
        'pairwise_similarities': pairwise_similarities_df,
        'similar_documents': similar_docs_df
    }

    return results


def extract_pairwise_similarities(lsa_results: Dict) -> pd.DataFrame:
    """
    Extract pairwise similarities DataFrame from LSA results.

    Args:
        lsa_results: Dictionary containing LSA results

    Returns:
        DataFrame with pairwise document similarities
    """
    return lsa_results['pairwise_similarities']


def extract_similar_documents(lsa_results: Dict) -> pd.DataFrame:
    """
    Extract similar documents DataFrame from LSA results.

    Args:
        lsa_results: Dictionary containing LSA results

    Returns:
        DataFrame with top-k similar documents for each document
    """
    return lsa_results['similar_documents']


def compare_time_periods(
    lsa_results_2006_2015: Dict,
    lsa_results_2016_2025: Dict,
    similarity_threshold: float = 0.7
) -> pd.DataFrame:
    """
    Compare topics between two time periods.

    Since the two periods may have different vocabularies, we compare
    topics based on their top terms rather than full vectors.

    Args:
        lsa_results_2006_2015: LSA results for 2006-2015
        lsa_results_2016_2025: LSA results for 2016-2025
        similarity_threshold: Threshold for topic similarity

    Returns:
        DataFrame with topic comparison results
    """
    topics_2006 = lsa_results_2006_2015['topic_terms']
    topics_2016 = lsa_results_2016_2025['topic_terms']

    # Get document topic distributions for prevalence analysis
    doc_topics_2006 = lsa_results_2006_2015['doc_topics']
    doc_topics_2016 = lsa_results_2016_2025['doc_topics']

    comparisons = []

    # Compare based on topic content and prevalence
    for i, row_2006 in topics_2006.iterrows():
        topic_2006_prevalence = doc_topics_2006[f'topic_{i}'].mean()

        comparisons.append({
            'period': '2006-2015',
            'topic_id': i,
            'top_terms': row_2006['top_terms'],
            'avg_prevalence': topic_2006_prevalence,
            'explained_variance': lsa_results_2006_2015['explained_variance'][i]
        })

    for i, row_2016 in topics_2016.iterrows():
        topic_2016_prevalence = doc_topics_2016[f'topic_{i}'].mean()

        comparisons.append({
            'period': '2016-2025',
            'topic_id': i,
            'top_terms': row_2016['top_terms'],
            'avg_prevalence': topic_2016_prevalence,
            'explained_variance': lsa_results_2016_2025['explained_variance'][i]
        })

    comparison_df = pd.DataFrame(comparisons)

    return comparison_df
