"""
Data processing nodes for text preprocessing and TF-IDF transformation.
"""

import re
from typing import Dict, List, Tuple

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_stopwords(stopwords_path: str = None) -> set:
    """
    Load Chinese stopwords from file.

    Args:
        stopwords_path: Path to stopwords file

    Returns:
        Set of stopwords
    """
    default_stopwords = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
        '都', '一', '一個', '上', '也', '很', '到', '說', '要', '去',
        '你', '會', '着', '沒有', '看', '好', '自己', '這', '那', '他'
    }

    if stopwords_path:
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                custom_stopwords = set(line.strip() for line in f if line.strip())
            return default_stopwords.union(custom_stopwords)
        except FileNotFoundError:
            print(f"Stopwords file not found at {stopwords_path}, using default")
            return default_stopwords

    return default_stopwords


def preprocess_text(
    text: str,
    stopwords: set,
    min_word_length: int = 2,
    max_word_length: int = 10
) -> str:
    """
    Preprocess Chinese text using jieba tokenization.

    Args:
        text: Input text string
        stopwords: Set of stopwords to remove
        min_word_length: Minimum word length to keep
        max_word_length: Maximum word length to keep

    Returns:
        Preprocessed text string
    """
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove special characters but keep Chinese characters
    text = re.sub(r'[^\u4e00-\u9fff\s]', '', text)

    # Tokenize using jieba
    words = jieba.cut(text)

    # Filter words
    filtered_words = [
        word.strip() for word in words
        if (
            word.strip() and
            word not in stopwords and
            min_word_length <= len(word) <= max_word_length
        )
    ]

    return ' '.join(filtered_words)


def preprocess_corpus(
    corpus_df: pd.DataFrame,
    parameters: Dict
) -> pd.DataFrame:
    """
    Preprocess news corpus dataframe.

    Args:
        corpus_df: DataFrame with 'news_date', 'news_title', 'news_content'
        parameters: Preprocessing parameters

    Returns:
        DataFrame with added 'processed_text' column
    """
    # Load stopwords
    stopwords = load_stopwords(
        parameters.get('custom_stopwords_path') if parameters.get('use_custom_stopwords') else None
    )

    # Combine title and content
    corpus_df['combined_text'] = corpus_df['news_title'] + ' ' + corpus_df['news_content']

    # Preprocess each document
    corpus_df['processed_text'] = corpus_df['combined_text'].apply(
        lambda x: preprocess_text(
            x,
            stopwords,
            parameters.get('min_word_length', 2),
            parameters.get('max_word_length', 10)
        )
    )

    # Filter out empty documents
    corpus_df = corpus_df[corpus_df['processed_text'].str.len() > 0].reset_index(drop=True)

    return corpus_df


def create_tfidf_matrix(
    preprocessed_df: pd.DataFrame,
    parameters: Dict
) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Create TF-IDF matrix from preprocessed text.

    Args:
        preprocessed_df: DataFrame with 'processed_text' column
        parameters: TF-IDF parameters

    Returns:
        Tuple of (TF-IDF matrix as DataFrame, fitted vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=parameters.get('max_features', 1000),
        min_df=parameters.get('min_df', 2),
        max_df=parameters.get('max_df', 0.8),
        ngram_range=tuple(parameters.get('ngram_range', [1, 2])),
        use_idf=parameters.get('use_idf', True),
        sublinear_tf=parameters.get('sublinear_tf', True)
    )

    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(preprocessed_df['processed_text'])

    # Convert to DataFrame for easier handling
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    return tfidf_df, vectorizer


def combine_corpora(
    corpus_2006_2015: pd.DataFrame,
    corpus_2016_2025: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine two time period corpora for comparison.

    Args:
        corpus_2006_2015: First period corpus
        corpus_2016_2025: Second period corpus

    Returns:
        Combined DataFrame with 'time_period' column
    """
    corpus_2006_2015['time_period'] = '2006-2015'
    corpus_2016_2025['time_period'] = '2016-2025'

    combined = pd.concat([corpus_2006_2015, corpus_2016_2025], ignore_index=True)

    return combined
