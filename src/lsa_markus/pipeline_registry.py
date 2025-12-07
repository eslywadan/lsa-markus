"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

from lsa_markus.pipelines import data_processing, lsa_analysis, lda_analysis


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = data_processing.create_pipeline()
    lsa_pipeline = lsa_analysis.create_pipeline()
    lda_pipeline = lda_analysis.create_pipeline()

    return {
        "data_processing": data_processing_pipeline,
        "lsa": lsa_pipeline,
        "lda": lda_pipeline,
        "__default__": data_processing_pipeline + lsa_pipeline + lda_pipeline,
    }
