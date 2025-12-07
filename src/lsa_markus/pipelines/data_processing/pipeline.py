"""
Data processing pipeline definition.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess_corpus,
    create_tfidf_matrix,
    combine_corpora,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the data processing pipeline.

    Returns:
        A Pipeline object containing all data processing nodes
    """
    return pipeline(
        [
            # Preprocess 2006-2015 corpus
            node(
                func=preprocess_corpus,
                inputs=["corpus_2006_2015", "params:preprocessing"],
                outputs="preprocessed_2006_2015",
                name="preprocess_corpus_2006_2015",
            ),
            # Preprocess 2016-2025 corpus
            node(
                func=preprocess_corpus,
                inputs=["corpus_2016_2025", "params:preprocessing"],
                outputs="preprocessed_2016_2025",
                name="preprocess_corpus_2016_2025",
            ),
            # Combine corpora
            node(
                func=combine_corpora,
                inputs=["preprocessed_2006_2015", "preprocessed_2016_2025"],
                outputs="combined_corpus",
                name="combine_corpora",
            ),
            # Create TF-IDF matrix for 2006-2015
            node(
                func=create_tfidf_matrix,
                inputs=["preprocessed_2006_2015", "params:tfidf"],
                outputs=["tfidf_matrix_2006_2015", "tfidf_vectorizer"],
                name="create_tfidf_2006_2015",
            ),
            # Create TF-IDF matrix for 2016-2025
            node(
                func=create_tfidf_matrix,
                inputs=["preprocessed_2016_2025", "params:tfidf"],
                outputs=["tfidf_matrix_2016_2025", "tfidf_vectorizer_2016_2025"],
                name="create_tfidf_2016_2025",
            ),
        ]
    )
