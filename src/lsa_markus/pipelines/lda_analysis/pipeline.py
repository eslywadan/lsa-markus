"""
LDA analysis pipeline definition.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_document_term_matrix,
    perform_lda,
    create_lda_results,
    compare_lda_topics,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the LDA analysis pipeline.

    Returns:
        A Pipeline object containing all LDA analysis nodes
    """
    return pipeline(
        [
            # Create document-term matrix for 2006-2015
            node(
                func=create_document_term_matrix,
                inputs=["preprocessed_2006_2015", "params:tfidf"],
                outputs=["doc_term_matrix_2006_2015", "count_vectorizer_2006_2015"],
                name="create_doc_term_matrix_2006_2015",
            ),
            # Create document-term matrix for 2016-2025
            node(
                func=create_document_term_matrix,
                inputs=["preprocessed_2016_2025", "params:tfidf"],
                outputs=["doc_term_matrix_2016_2025", "count_vectorizer_2016_2025"],
                name="create_doc_term_matrix_2016_2025",
            ),
            # Perform LDA on 2006-2015 corpus
            node(
                func=perform_lda,
                inputs=["doc_term_matrix_2006_2015", "params:lda"],
                outputs=[
                    "lda_model_2006_2015",
                    "lda_doc_topic_matrix_2006_2015",
                    "lda_topic_term_matrix_2006_2015"
                ],
                name="perform_lda_2006_2015",
            ),
            # Perform LDA on 2016-2025 corpus
            node(
                func=perform_lda,
                inputs=["doc_term_matrix_2016_2025", "params:lda"],
                outputs=[
                    "lda_model_2016_2025",
                    "lda_doc_topic_matrix_2016_2025",
                    "lda_topic_term_matrix_2016_2025"
                ],
                name="perform_lda_2016_2025",
            ),
            # Create comprehensive results for 2006-2015
            node(
                func=create_lda_results,
                inputs=[
                    "lda_model_2006_2015",
                    "lda_doc_topic_matrix_2006_2015",
                    "lda_topic_term_matrix_2006_2015",
                    "doc_term_matrix_2006_2015",
                    "preprocessed_2006_2015",
                    "params:topic_interpretation.n_top_words"
                ],
                outputs="lda_results_2006_2015",
                name="create_lda_results_2006_2015",
            ),
            # Create comprehensive results for 2016-2025
            node(
                func=create_lda_results,
                inputs=[
                    "lda_model_2016_2025",
                    "lda_doc_topic_matrix_2016_2025",
                    "lda_topic_term_matrix_2016_2025",
                    "doc_term_matrix_2016_2025",
                    "preprocessed_2016_2025",
                    "params:topic_interpretation.n_top_words"
                ],
                outputs="lda_results_2016_2025",
                name="create_lda_results_2016_2025",
            ),
            # Compare topics across time periods
            node(
                func=compare_lda_topics,
                inputs=[
                    "lda_results_2006_2015",
                    "lda_results_2016_2025"
                ],
                outputs="lda_topic_comparison",
                name="compare_lda_topics",
            ),
        ]
    )
