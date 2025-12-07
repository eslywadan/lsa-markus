#!/usr/bin/env python3
"""
Interactive viewer for topic-term matrices from LSA/LDA results.

Usage:
    python src/view_topic_term_matrix.py
"""

import pickle
import pandas as pd
import numpy as np


def load_lsa_topic_term_matrix(period='2006_2015'):
    """Load LSA topic-term matrix for a specific period."""
    filepath = f'data/07_model_output/lsa_results_{period}.pkl'

    with open(filepath, 'rb') as f:
        results = pickle.load(f)

    topic_term_matrix = results['topic_term_matrix']
    feature_names = results['feature_names']

    df = pd.DataFrame(
        topic_term_matrix,
        index=[f'topic_{i}' for i in range(topic_term_matrix.shape[0])],
        columns=feature_names
    )

    return df, results


def load_lda_topic_term_matrix(period='2006_2015'):
    """Load LDA topic-term matrix for a specific period."""
    filepath = f'data/07_model_output/lda_results_{period}.pkl'

    with open(filepath, 'rb') as f:
        results = pickle.load(f)

    topic_term_matrix = results['topic_term_matrix']
    feature_names = results['feature_names']

    df = pd.DataFrame(
        topic_term_matrix,
        index=[f'topic_{i}' for i in range(topic_term_matrix.shape[0])],
        columns=feature_names
    )

    return df, results


def show_topic_summary(df, topic_id, n_terms=20):
    """Show top terms for a specific topic."""
    topic_name = f'topic_{topic_id}'

    if topic_name not in df.index:
        print(f"❌ Topic {topic_id} not found. Available topics: 0-{len(df)-1}")
        return

    print(f"\nTopic {topic_id} - Top {n_terms} terms by absolute weight")
    print("=" * 70)

    topic_weights = df.loc[topic_name].sort_values(key=abs, ascending=False).head(n_terms)

    for i, (term, weight) in enumerate(topic_weights.items(), 1):
        print(f"{i:3d}. {term:20s} : {weight:8.5f}")


def show_term_across_topics(df, term):
    """Show how a specific term is weighted across all topics."""
    if term not in df.columns:
        print(f"❌ Term '{term}' not found in vocabulary")
        print(f"Available terms: {len(df.columns)}")
        return

    print(f"\nTerm '{term}' across all topics")
    print("=" * 70)

    weights = df[term].sort_values(key=abs, ascending=False)

    for topic, weight in weights.items():
        print(f"{topic:10s} : {weight:8.5f}")


def export_matrix(df, output_path):
    """Export matrix to CSV or Excel."""
    if output_path.endswith('.xlsx'):
        df.to_excel(output_path)
    else:
        df.to_csv(output_path)

    print(f"✅ Exported to: {output_path}")


def interactive_viewer():
    """Interactive command-line viewer."""
    print("\n" + "=" * 70)
    print("TOPIC-TERM MATRIX VIEWER")
    print("=" * 70)

    # Load data
    print("\n1. Select period:")
    print("   [1] 2006-2015")
    print("   [2] 2016-2025")
    period_choice = input("\nEnter choice (1 or 2): ").strip()

    period = '2006_2015' if period_choice == '1' else '2016~2025'

    print("\n2. Select model:")
    print("   [1] LSA")
    print("   [2] LDA")
    model_choice = input("\nEnter choice (1 or 2): ").strip()

    if model_choice == '1':
        df, results = load_lsa_topic_term_matrix(period)
        model_name = 'LSA'
    else:
        df, results = load_lda_topic_term_matrix(period)
        model_name = 'LDA'

    print(f"\n✅ Loaded {model_name} results for {period}")
    print(f"   Matrix shape: {df.shape[0]} topics × {df.shape[1]} terms")

    # Interactive loop
    while True:
        print("\n" + "=" * 70)
        print("OPTIONS:")
        print("  [1] Show topic summary")
        print("  [2] Show term across topics")
        print("  [3] Export to CSV")
        print("  [4] Show matrix statistics")
        print("  [q] Quit")

        choice = input("\nEnter choice: ").strip().lower()

        if choice == 'q':
            break
        elif choice == '1':
            topic_id = int(input("Enter topic ID (0-9): "))
            n_terms = int(input("Number of terms to show (default 20): ") or 20)
            show_topic_summary(df, topic_id, n_terms)
        elif choice == '2':
            term = input("Enter term (Chinese): ").strip()
            show_term_across_topics(df, term)
        elif choice == '3':
            output_path = f'data/08_reporting/{model_name.lower()}_topic_term_matrix_{period}.csv'
            export_matrix(df, output_path)
        elif choice == '4':
            print("\nMatrix Statistics:")
            print(df.describe())
        else:
            print("❌ Invalid choice")


def main():
    """Main function - quick example."""
    print("\n🔍 LSA Topic-Term Matrix Viewer\n")

    # Example 1: Load LSA 2006-2015
    print("Example: Loading LSA 2006-2015 topic-term matrix...")
    df_lsa, results_lsa = load_lsa_topic_term_matrix('2006_2015')

    print(f"\nMatrix shape: {df_lsa.shape}")
    print(f"  - {df_lsa.shape[0]} topics")
    print(f"  - {df_lsa.shape[1]} terms")

    # Show Topic 1 summary
    show_topic_summary(df_lsa, topic_id=1, n_terms=15)

    # Show term '警方' across topics
    show_term_across_topics(df_lsa, '警方')

    print("\n" + "=" * 70)
    print("\nFor interactive mode, uncomment the line below:")
    print("# interactive_viewer()")
    print("\nOr use directly in Python:")
    print("  from view_topic_term_matrix import load_lsa_topic_term_matrix")
    print("  df, results = load_lsa_topic_term_matrix('2006_2015')")
    print("  print(df.loc['topic_1'].sort_values(key=abs, ascending=False).head(20))")


if __name__ == '__main__':
    main()
    # Uncomment for interactive mode:
    # interactive_viewer()
