"""
LSA analysis pipeline definition.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    perform_lsa,
    create_lsa_results,
    compare_time_periods,
    extract_pairwise_similarities,
    extract_similar_documents,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the LSA analysis pipeline.

    Returns:
        A Pipeline object containing all LSA analysis nodes
    """
    return pipeline(
        [
            # Perform LSA on 2006-2015 corpus
            node(
                func=perform_lsa,
                inputs=["tfidf_matrix_2006_2015", "params:lsa"],
                outputs=[
                    "lsa_model_2006_2015",
                    "doc_topic_matrix_2006_2015",
                    "topic_term_matrix_2006_2015"
                ],
                name="perform_lsa_2006_2015",
            ),
            # Perform LSA on 2016-2025 corpus
            node(
                func=perform_lsa,
                inputs=["tfidf_matrix_2016_2025", "params:lsa"],
                outputs=[
                    "lsa_model_2016_2025",
                    "doc_topic_matrix_2016_2025",
                    "topic_term_matrix_2016_2025"
                ],
                name="perform_lsa_2016_2025",
            ),
            # Create comprehensive results for 2006-2015
            node(
                func=create_lsa_results,
                inputs=[
                    "lsa_model_2006_2015",
                    "doc_topic_matrix_2006_2015",
                    "topic_term_matrix_2006_2015",
                    "tfidf_matrix_2006_2015",
                    "preprocessed_2006_2015",
                    "params:topic_interpretation.n_top_words",
                    "params:similarity.top_k"
                ],
                outputs="lsa_results_2006_2015",
                name="create_lsa_results_2006_2015",
            ),
            # Create comprehensive results for 2016-2025
            node(
                func=create_lsa_results,
                inputs=[
                    "lsa_model_2016_2025",
                    "doc_topic_matrix_2016_2025",
                    "topic_term_matrix_2016_2025",
                    "tfidf_matrix_2016_2025",
                    "preprocessed_2016_2025",
                    "params:topic_interpretation.n_top_words",
                    "params:similarity.top_k"
                ],
                outputs="lsa_results_2016_2025",
                name="create_lsa_results_2016_2025",
            ),
            # Compare topics across time periods
            node(
                func=compare_time_periods,
                inputs=[
                    "lsa_results_2006_2015",
                    "lsa_results_2016_2025",
                    "params:comparison.similarity_threshold"
                ],
                outputs="lsa_topic_comparison",
                name="compare_lsa_topics",
            ),
            # Extract and save pairwise similarities for 2006-2015
            node(
                func=extract_pairwise_similarities,
                inputs="lsa_results_2006_2015",
                outputs="lsa_pairwise_similarities_2006_2015",
                name="extract_pairwise_similarities_2006_2015",
            ),
            # Extract and save similar documents for 2006-2015
            node(
                func=extract_similar_documents,
                inputs="lsa_results_2006_2015",
                outputs="lsa_similar_documents_2006_2015",
                name="extract_similar_documents_2006_2015",
            ),
            # Extract and save pairwise similarities for 2016-2025
            node(
                func=extract_pairwise_similarities,
                inputs="lsa_results_2016_2025",
                outputs="lsa_pairwise_similarities_2016_2025",
                name="extract_pairwise_similarities_2016_2025",
            ),
            # Extract and save similar documents for 2016-2025
            node(
                func=extract_similar_documents,
                inputs="lsa_results_2016_2025",
                outputs="lsa_similar_documents_2016_2025",
                name="extract_similar_documents_2016_2025",
            ),
        ]
    )
