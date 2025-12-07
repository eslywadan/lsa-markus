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
    n_top_words: int = 15
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

    Returns:
        Dictionary containing all LSA results
    """
    feature_names = tfidf_matrix.columns.tolist()

    # Extract topic terms
    topic_terms_df = extract_topic_terms(lsa_model, feature_names, n_top_words)

    # Calculate document similarities
    doc_topics_df = calculate_document_similarities(doc_topic_matrix, preprocessed_df)

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
        'feature_names': feature_names
    }

    return results


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
