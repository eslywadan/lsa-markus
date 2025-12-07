"""
LDA (Latent Dirichlet Allocation) nodes.

Implements probabilistic topic modeling using LDA, which assumes:
1. Each document is a mixture of topics
2. Each topic is a distribution over words
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def create_document_term_matrix(
    preprocessed_df: pd.DataFrame,
    parameters: Dict
) -> Tuple[pd.DataFrame, CountVectorizer]:
    """
    Create document-term matrix for LDA (uses raw counts, not TF-IDF).

    Args:
        preprocessed_df: DataFrame with 'processed_text' column
        parameters: Vectorization parameters

    Returns:
        Tuple of (document-term matrix as DataFrame, fitted vectorizer)
    """
    vectorizer = CountVectorizer(
        max_features=parameters.get('max_features', 1000),
        min_df=parameters.get('min_df', 2),
        max_df=parameters.get('max_df', 0.8),
        ngram_range=tuple(parameters.get('ngram_range', [1, 2]))
    )

    # Fit and transform
    doc_term_matrix = vectorizer.fit_transform(preprocessed_df['processed_text'])

    # Convert to DataFrame
    doc_term_df = pd.DataFrame(
        doc_term_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    return doc_term_df, vectorizer


def perform_lda(
    doc_term_matrix: pd.DataFrame,
    parameters: Dict
) -> Tuple[LatentDirichletAllocation, np.ndarray, np.ndarray]:
    """
    Perform Latent Dirichlet Allocation topic modeling.

    Args:
        doc_term_matrix: Document-term matrix as DataFrame
        parameters: LDA parameters

    Returns:
        Tuple of (fitted LDA model, document-topic matrix, topic-term matrix)
    """
    lda_model = LatentDirichletAllocation(
        n_components=parameters.get('n_topics', 10),
        max_iter=parameters.get('max_iter', 100),
        learning_method=parameters.get('learning_method', 'online'),
        random_state=parameters.get('random_state', 42),
        n_jobs=parameters.get('n_jobs', -1),
        batch_size=parameters.get('batch_size', 128),
        doc_topic_prior=parameters.get('doc_topic_prior'),
        topic_word_prior=parameters.get('topic_word_prior')
    )

    # Fit and transform: get document-topic distribution
    doc_topic_matrix = lda_model.fit_transform(doc_term_matrix)

    # Get topic-term distribution
    topic_term_matrix = lda_model.components_

    return lda_model, doc_topic_matrix, topic_term_matrix


def extract_lda_topic_terms(
    lda_model: LatentDirichletAllocation,
    feature_names: list,
    n_top_words: int = 15
) -> pd.DataFrame:
    """
    Extract top terms for each LDA topic.

    Args:
        lda_model: Fitted LDA model
        feature_names: List of feature names (vocabulary)
        n_top_words: Number of top words to extract per topic

    Returns:
        DataFrame with topics and their top terms
    """
    topics_data = []

    for topic_idx, topic in enumerate(lda_model.components_):
        # Get top term indices (highest probabilities)
        top_indices = topic.argsort()[-n_top_words:][::-1]

        top_terms = [feature_names[i] for i in top_indices]
        top_probs = [topic[i] for i in top_indices]

        # Normalize probabilities to sum to 1 for interpretation
        normalized_probs = top_probs / np.sum(top_probs)

        topics_data.append({
            'topic_id': topic_idx,
            'top_terms': ', '.join(top_terms),
            'term_probabilities': normalized_probs.tolist()
        })

    return pd.DataFrame(topics_data)


def analyze_document_topics(
    doc_topic_matrix: np.ndarray,
    preprocessed_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze document-topic distributions.

    Args:
        doc_topic_matrix: Document-topic probability matrix
        preprocessed_df: Original preprocessed DataFrame

    Returns:
        DataFrame with document info and topic distributions
    """
    # Create DataFrame with document-topic probabilities
    topic_columns = [f'topic_{i}' for i in range(doc_topic_matrix.shape[1])]
    doc_topics_df = pd.DataFrame(doc_topic_matrix, columns=topic_columns)

    # Add original document information
    doc_topics_df['news_date'] = preprocessed_df['news_date'].values
    doc_topics_df['news_title'] = preprocessed_df['news_title'].values

    # Calculate dominant topic (highest probability)
    doc_topics_df['dominant_topic'] = doc_topic_matrix.argmax(axis=1)
    doc_topics_df['dominant_topic_prob'] = doc_topic_matrix.max(axis=1)

    # Calculate topic diversity (entropy)
    doc_topics_df['topic_entropy'] = -np.sum(
        doc_topic_matrix * np.log(doc_topic_matrix + 1e-10), axis=1
    )

    return doc_topics_df


def create_lda_results(
    lda_model: LatentDirichletAllocation,
    doc_topic_matrix: np.ndarray,
    topic_term_matrix: np.ndarray,
    doc_term_matrix: pd.DataFrame,
    preprocessed_df: pd.DataFrame,
    n_top_words: int = 15
) -> Dict:
    """
    Create comprehensive LDA results dictionary.

    Args:
        lda_model: Fitted LDA model
        doc_topic_matrix: Document-topic matrix
        topic_term_matrix: Topic-term matrix
        doc_term_matrix: Original document-term matrix
        preprocessed_df: Preprocessed corpus DataFrame
        n_top_words: Number of top words per topic

    Returns:
        Dictionary containing all LDA results
    """
    feature_names = doc_term_matrix.columns.tolist()

    # Extract topic terms
    topic_terms_df = extract_lda_topic_terms(lda_model, feature_names, n_top_words)

    # Analyze document topics
    doc_topics_df = analyze_document_topics(doc_topic_matrix, preprocessed_df)

    # Calculate perplexity and log-likelihood
    perplexity = lda_model.perplexity(doc_term_matrix)
    log_likelihood = lda_model.score(doc_term_matrix)

    results = {
        'model': lda_model,
        'doc_topic_matrix': doc_topic_matrix,
        'topic_term_matrix': topic_term_matrix,
        'topic_terms': topic_terms_df,
        'doc_topics': doc_topics_df,
        'perplexity': perplexity,
        'log_likelihood': log_likelihood,
        'feature_names': feature_names
    }

    return results


def compare_lda_topics(
    lda_results_2006_2015: Dict,
    lda_results_2016_2025: Dict
) -> pd.DataFrame:
    """
    Compare LDA topics between two time periods.

    Args:
        lda_results_2006_2015: LDA results for 2006-2015
        lda_results_2016_2025: LDA results for 2016-2025

    Returns:
        DataFrame with topic comparison
    """
    topics_2006 = lda_results_2006_2015['topic_terms']
    topics_2016 = lda_results_2016_2025['topic_terms']

    # Calculate topic prevalence for each period
    doc_topics_2006 = lda_results_2006_2015['doc_topics']
    doc_topics_2016 = lda_results_2016_2025['doc_topics']

    comparison_data = []

    for i in range(len(topics_2006)):
        topic_2006_prevalence = doc_topics_2006[f'topic_{i}'].mean()

        comparison_data.append({
            'period': '2006-2015',
            'topic_id': i,
            'top_terms': topics_2006.iloc[i]['top_terms'],
            'avg_prevalence': topic_2006_prevalence
        })

    for i in range(len(topics_2016)):
        topic_2016_prevalence = doc_topics_2016[f'topic_{i}'].mean()

        comparison_data.append({
            'period': '2016-2025',
            'topic_id': i,
            'top_terms': topics_2016.iloc[i]['top_terms'],
            'avg_prevalence': topic_2016_prevalence
        })

    comparison_df = pd.DataFrame(comparison_data)

    return comparison_df
