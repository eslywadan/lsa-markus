#!/usr/bin/env python3
"""
Standalone script to visualize .pkl results from LSA/LDA analysis.

Usage:
    python src/visualize_pkl.py
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def load_pkl(filepath):
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def visualize_lsa_results():
    """Visualize LSA results from pickle files"""

    # Load LSA results
    lsa_2006 = load_pkl('data/07_model_output/lsa_results_2006_2015.pkl')
    lsa_2016 = load_pkl('data/07_model_output/lsa_results_2016~2025.pkl')

    print("=" * 60)
    print("LSA RESULTS SUMMARY")
    print("=" * 60)

    # Summary statistics
    print(f"\n2006-2015:")
    print(f"  Total Variance Explained: {lsa_2006['total_variance_explained']:.2%}")
    print(f"  Number of Topics: {len(lsa_2006['topic_terms'])}")
    print(f"  Number of Documents: {len(lsa_2006['doc_topics'])}")

    print(f"\n2016-2025:")
    print(f"  Total Variance Explained: {lsa_2016['total_variance_explained']:.2%}")
    print(f"  Number of Topics: {len(lsa_2016['topic_terms'])}")
    print(f"  Number of Documents: {len(lsa_2016['doc_topics'])}")

    # Display topic terms
    print("\n" + "=" * 60)
    print("TOP TOPICS (2006-2015)")
    print("=" * 60)
    print(lsa_2006['topic_terms'].to_string())

    print("\n" + "=" * 60)
    print("TOP TOPICS (2016-2025)")
    print("=" * 60)
    print(lsa_2016['topic_terms'].to_string())

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Explained Variance (2006-2015)
    axes[0, 0].bar(range(len(lsa_2006['explained_variance'])),
                   lsa_2006['explained_variance'])
    axes[0, 0].set_xlabel('Topic Number')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('LSA Explained Variance (2006-2015)')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Explained Variance (2016-2025)
    axes[0, 1].bar(range(len(lsa_2016['explained_variance'])),
                   lsa_2016['explained_variance'], color='coral')
    axes[0, 1].set_xlabel('Topic Number')
    axes[0, 1].set_ylabel('Explained Variance Ratio')
    axes[0, 1].set_title('LSA Explained Variance (2016-2025)')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Topic prevalence (2006-2015)
    doc_topics_2006 = lsa_2006['doc_topics'].filter(regex='^topic_')
    topic_prev_2006 = doc_topics_2006.mean()
    axes[1, 0].bar(range(len(topic_prev_2006)), topic_prev_2006)
    axes[1, 0].set_xlabel('Topic Number')
    axes[1, 0].set_ylabel('Average Prevalence')
    axes[1, 0].set_title('Topic Prevalence (2006-2015)')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Topic prevalence (2016-2025)
    doc_topics_2016 = lsa_2016['doc_topics'].filter(regex='^topic_')
    topic_prev_2016 = doc_topics_2016.mean()
    axes[1, 1].bar(range(len(topic_prev_2016)), topic_prev_2016, color='coral')
    axes[1, 1].set_xlabel('Topic Number')
    axes[1, 1].set_ylabel('Average Prevalence')
    axes[1, 1].set_title('Topic Prevalence (2016-2025)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = 'data/08_reporting/lsa_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ LSA visualization saved to: {output_path}")
    plt.show()


def visualize_lda_results():
    """Visualize LDA results from pickle files"""

    # Load LDA results
    lda_2006 = load_pkl('data/07_model_output/lda_results_2006_2015.pkl')
    lda_2016 = load_pkl('data/07_model_output/lda_results_2016~2025.pkl')

    print("\n" + "=" * 60)
    print("LDA RESULTS SUMMARY")
    print("=" * 60)

    # Summary statistics
    print(f"\n2006-2015:")
    print(f"  Perplexity: {lda_2006['perplexity']:.2f}")
    print(f"  Log Likelihood: {lda_2006['log_likelihood']:.2f}")
    print(f"  Number of Topics: {len(lda_2006['topic_terms'])}")
    print(f"  Number of Documents: {len(lda_2006['doc_topics'])}")

    print(f"\n2016-2025:")
    print(f"  Perplexity: {lda_2016['perplexity']:.2f}")
    print(f"  Log Likelihood: {lda_2016['log_likelihood']:.2f}")
    print(f"  Number of Topics: {len(lda_2016['topic_terms'])}")
    print(f"  Number of Documents: {len(lda_2016['doc_topics'])}")

    # Display topic terms
    print("\n" + "=" * 60)
    print("TOP TOPICS (2006-2015)")
    print("=" * 60)
    print(lda_2006['topic_terms'].to_string())

    print("\n" + "=" * 60)
    print("TOP TOPICS (2016-2025)")
    print("=" * 60)
    print(lda_2016['topic_terms'].to_string())

    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Topic prevalence comparison
    doc_topics_2006 = lda_2006['doc_topics'].filter(regex='^topic_')
    doc_topics_2016 = lda_2016['doc_topics'].filter(regex='^topic_')

    topic_prev_2006 = doc_topics_2006.mean()
    topic_prev_2016 = doc_topics_2016.mean()

    x = np.arange(len(topic_prev_2006))
    width = 0.35

    axes[0].bar(x - width/2, topic_prev_2006, width, label='2006-2015', alpha=0.8)
    axes[0].bar(x + width/2, topic_prev_2016, width, label='2016-2025', alpha=0.8)
    axes[0].set_xlabel('Topic')
    axes[0].set_ylabel('Average Prevalence')
    axes[0].set_title('LDA Topic Prevalence Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'T{i}' for i in range(len(topic_prev_2006))])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Topic entropy distribution
    entropy_2006 = lda_2006['doc_topics']['topic_entropy']
    entropy_2016 = lda_2016['doc_topics']['topic_entropy']

    axes[1].hist(entropy_2006, bins=20, alpha=0.6, label='2006-2015')
    axes[1].hist(entropy_2016, bins=20, alpha=0.6, label='2016-2025')
    axes[1].set_xlabel('Topic Entropy')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Document Topic Diversity Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = 'data/08_reporting/lda_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ LDA visualization saved to: {output_path}")
    plt.show()


def visualize_comparison():
    """Visualize comparison reports"""

    # Load comparison CSVs
    lsa_comp = pd.read_csv('data/08_reporting/lsa_topic_comparison.csv')
    lda_comp = pd.read_csv('data/08_reporting/lda_topic_comparison.csv')

    print("\n" + "=" * 60)
    print("TEMPORAL COMPARISON")
    print("=" * 60)

    print("\n--- LSA Topic Comparison ---")
    print(lsa_comp.to_string())

    print("\n--- LDA Topic Comparison ---")
    print(lda_comp.to_string())

    # Visualize LSA comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Filter by period
    lsa_2006 = lsa_comp[lsa_comp['period'] == '2006-2015'].sort_values('avg_prevalence', ascending=False)
    lsa_2016 = lsa_comp[lsa_comp['period'] == '2016-2025'].sort_values('avg_prevalence', ascending=False)

    # 2006-2015
    axes[0].barh(range(len(lsa_2006)), lsa_2006['avg_prevalence'])
    axes[0].set_yticks(range(len(lsa_2006)))
    axes[0].set_yticklabels([f"Topic {i}" for i in lsa_2006['topic_id']])
    axes[0].set_xlabel('Average Prevalence')
    axes[0].set_title('LSA Topics by Prevalence (2006-2015)')
    axes[0].invert_yaxis()

    # 2016-2025
    axes[1].barh(range(len(lsa_2016)), lsa_2016['avg_prevalence'], color='coral')
    axes[1].set_yticks(range(len(lsa_2016)))
    axes[1].set_yticklabels([f"Topic {i}" for i in lsa_2016['topic_id']])
    axes[1].set_xlabel('Average Prevalence')
    axes[1].set_title('LSA Topics by Prevalence (2016-2025)')
    axes[1].invert_yaxis()

    plt.tight_layout()

    # Save figure
    output_path = 'data/08_reporting/comparison_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison visualization saved to: {output_path}")
    plt.show()


def main():
    """Main function to run all visualizations"""
    print("\n🎨 Starting visualization of .pkl results...\n")

    try:
        visualize_lsa_results()
        visualize_lda_results()
        visualize_comparison()

        print("\n" + "=" * 60)
        print("✅ ALL VISUALIZATIONS COMPLETED!")
        print("=" * 60)
        print("\nOutput files:")
        print("  - data/08_reporting/lsa_visualization.png")
        print("  - data/08_reporting/lda_visualization.png")
        print("  - data/08_reporting/comparison_visualization.png")
        print("\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have run the pipeline first: kedro run")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
