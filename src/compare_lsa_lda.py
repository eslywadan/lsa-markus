#!/usr/bin/env python3
"""
Comprehensive comparison of LSA vs LDA pipelines and results.

Usage:
    python src/compare_lsa_lda.py
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


def load_results(method='lsa', period='2006_2015'):
    """Load LSA or LDA results."""
    filepath = f'../data/07_model_output/{method}_results_{period}.pkl'
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def compare_algorithms():
    """Compare LSA and LDA algorithms, pipelines, and results."""

    print("\n" + "=" * 80)
    print("LSA vs LDA COMPREHENSIVE COMPARISON")
    print("=" * 80)

    # Load results for both methods and periods
    lsa_2006 = load_results('lsa', '2006_2015')
    lsa_2016 = load_results('lsa', '2016~2025')
    lda_2006 = load_results('lda', '2006_2015')
    lda_2016 = load_results('lda', '2016~2025')

    # ========================================================================
    # PART 1: ALGORITHMIC DIFFERENCES
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 1: ALGORITHMIC FOUNDATIONS")
    print("=" * 80)

    print("\n📐 LSA (Latent Semantic Analysis)")
    print("-" * 80)
    print("Mathematical Foundation:")
    print("  • Matrix Factorization: Singular Value Decomposition (SVD)")
    print("  • Formula: X ≈ U Σ V^T")
    print("    - X: Document-term matrix (TF-IDF weighted)")
    print("    - U: Document-topic matrix")
    print("    - Σ: Diagonal matrix of singular values (topic importance)")
    print("    - V^T: Topic-term matrix")
    print("\nKey Characteristics:")
    print("  • Deterministic: Same results every run")
    print("  • Linear algebra-based")
    print("  • Captures global corpus structure")
    print("  • Topics can have negative weights (bidirectional)")
    print("  • Fast computation via truncated SVD")

    print("\n🎲 LDA (Latent Dirichlet Allocation)")
    print("-" * 80)
    print("Mathematical Foundation:")
    print("  • Probabilistic Generative Model")
    print("  • Assumes:")
    print("    - Each document is a mixture of topics (Dirichlet distribution)")
    print("    - Each topic is a distribution over words (Dirichlet distribution)")
    print("  • Uses raw word counts (not TF-IDF)")
    print("\nKey Characteristics:")
    print("  • Probabilistic: Results vary slightly between runs")
    print("  • Bayesian inference-based")
    print("  • Models document generation process")
    print("  • All weights are positive probabilities (interpretable)")
    print("  • Requires iterative optimization")
    print("\n⭐ Inference Method (Your Implementation):")
    print("  • Stochastic Variational Inference (SVI)")
    print("  • Uses learning_method='online' (NOT EM or Gibbs Sampling)")
    print("  • Processes mini-batches for scalability")
    print("  • Alternative: learning_method='batch' for traditional Variational EM")

    # ========================================================================
    # PART 2: PIPELINE DIFFERENCES
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("PART 2: PIPELINE STRUCTURE COMPARISON")
    print("=" * 80)

    print("\n🔄 LSA Pipeline")
    print("-" * 80)
    print("1. Preprocessing")
    print("   ├─ Tokenization & stopword removal")
    print("   └─ Text normalization")
    print("\n2. TF-IDF Vectorization (CRITICAL DIFFERENCE)")
    print("   ├─ Term Frequency (TF): Weighted by frequency")
    print("   ├─ Inverse Document Frequency (IDF): Downweight common terms")
    print("   └─ Output: TF-IDF matrix (real-valued, normalized)")
    print("\n3. Truncated SVD")
    print("   ├─ Input: TF-IDF matrix")
    print("   ├─ Decomposition: X = U Σ V^T")
    print("   └─ Output: Document-topic & topic-term matrices")
    print("\n4. Interpretation")
    print("   ├─ Explained variance per topic")
    print("   └─ Can have negative loadings")

    print("\n🔄 LDA Pipeline")
    print("-" * 80)
    print("1. Preprocessing")
    print("   ├─ Tokenization & stopword removal")
    print("   └─ Text normalization (same as LSA)")
    print("\n2. Count Vectorization (CRITICAL DIFFERENCE)")
    print("   ├─ Raw word counts (no TF-IDF)")
    print("   ├─ Integer counts preserve frequency information")
    print("   └─ Output: Document-term matrix (integer counts)")
    print("\n3. LDA Model Fitting")
    print("   ├─ Input: Document-term count matrix")
    print("   ├─ Iterative optimization (Variational Bayes or Gibbs Sampling)")
    print("   └─ Output: Document-topic & topic-term distributions")
    print("\n4. Interpretation")
    print("   ├─ Perplexity & log-likelihood metrics")
    print("   ├─ All probabilities sum to 1")
    print("   └─ Topic entropy (diversity measure)")

    # ========================================================================
    # PART 3: INPUT MATRIX DIFFERENCES
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("PART 3: INPUT MATRIX COMPARISON")
    print("=" * 80)

    print("\n📊 LSA Input: TF-IDF Matrix")
    print("-" * 80)
    print("Sample values (continuous, weighted):")
    print("  Document 1, Term '部落': 0.234")
    print("  Document 1, Term '文化': 0.187")
    print("  Document 2, Term '警方': 0.456")
    print("\nCharacteristics:")
    print("  • Values between 0 and 1 (typically)")
    print("  • Emphasizes distinctive terms (high IDF)")
    print("  • De-emphasizes common terms (low IDF)")

    print("\n📊 LDA Input: Count Matrix")
    print("-" * 80)
    print("Sample values (integer counts):")
    print("  Document 1, Term '部落': 5 times")
    print("  Document 1, Term '文化': 3 times")
    print("  Document 2, Term '警方': 8 times")
    print("\nCharacteristics:")
    print("  • Integer values (0, 1, 2, ...)")
    print("  • All terms weighted equally by default")
    print("  • Preserves raw frequency information")

    # ========================================================================
    # PART 4: OUTPUT COMPARISON
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("PART 4: OUTPUT METRICS COMPARISON")
    print("=" * 80)

    # Create comparison table
    comparison_data = []

    for period_name, lsa_res, lda_res in [
        ('2006-2015', lsa_2006, lda_2006),
        ('2016-2025', lsa_2016, lda_2016)
    ]:
        comparison_data.append({
            'Period': period_name,
            'Method': 'LSA',
            'N Topics': len(lsa_res['topic_terms']),
            'N Documents': len(lsa_res['doc_topics']),
            'N Terms': len(lsa_res['feature_names']),
            'Metric 1': f"Total Var: {lsa_res['total_variance_explained']:.2%}",
            'Metric 2': f"Top Topic Var: {lsa_res['explained_variance'][0]:.2%}",
            'Weight Type': 'Real-valued (can be negative)',
            'Interpretation': 'Variance explained'
        })

        comparison_data.append({
            'Period': period_name,
            'Method': 'LDA',
            'N Topics': len(lda_res['topic_terms']),
            'N Documents': len(lda_res['doc_topics']),
            'N Terms': len(lda_res['feature_names']),
            'Metric 1': f"Perplexity: {lda_res['perplexity']:.2f}",
            'Metric 2': f"Log-likelihood: {lda_res['log_likelihood']:.2f}",
            'Weight Type': 'Probabilities (0 to 1)',
            'Interpretation': 'Probability distributions'
        })

    df_comparison = pd.DataFrame(comparison_data)
    print("\n📋 Results Summary")
    print("-" * 80)
    print(df_comparison.to_string(index=False))

    # ========================================================================
    # PART 5: TOPIC COMPARISON
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("PART 5: DISCOVERED TOPICS COMPARISON (2006-2015)")
    print("=" * 80)

    print("\n🔍 LSA Topics (2006-2015)")
    print("-" * 80)
    for i in range(min(3, len(lsa_2006['topic_terms']))):
        terms = lsa_2006['topic_terms'].iloc[i]['top_terms']
        var = lsa_2006['explained_variance'][i] * 100
        print(f"Topic {i} ({var:.2f}% variance): {terms[:80]}...")

    print("\n🔍 LDA Topics (2006-2015)")
    print("-" * 80)
    doc_topics = lda_2006['doc_topics'].filter(regex='^topic_')
    for i in range(min(3, len(lda_2006['topic_terms']))):
        terms = lda_2006['topic_terms'].iloc[i]['top_terms']
        prevalence = doc_topics[f'topic_{i}'].mean()
        print(f"Topic {i} (prevalence: {prevalence:.3f}): {terms[:80]}...")

    # ========================================================================
    # PART 6: WHEN TO USE WHICH
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("PART 6: DECISION GUIDE - WHEN TO USE WHICH METHOD")
    print("=" * 80)

    print("\n✅ Use LSA When:")
    print("-" * 80)
    print("  1. You want FAST, deterministic results")
    print("  2. You're interested in semantic similarity and dimensionality reduction")
    print("  3. You want to capture global corpus structure")
    print("  4. You need explained variance metrics for interpretation")
    print("  5. You're working with smaller corpora (< 10,000 documents)")
    print("  6. You want to emphasize distinctive terms (TF-IDF weighting)")
    print("\n  Example Use Cases:")
    print("    • Document similarity search")
    print("    • Information retrieval")
    print("    • Semantic search engines")
    print("    • Exploratory analysis of corpus structure")

    print("\n✅ Use LDA When:")
    print("-" * 80)
    print("  1. You want INTERPRETABLE probability distributions")
    print("  2. You need topics that are mixtures of words")
    print("  3. You want to model the generative process of documents")
    print("  4. You need all-positive weights for easier interpretation")
    print("  5. You're working with larger corpora (> 1,000 documents)")
    print("  6. You want to preserve raw frequency information")
    print("\n  Example Use Cases:")
    print("    • Topic modeling for content understanding")
    print("    • Trend analysis over time")
    print("    • Document categorization")
    print("    • Discovering themes in text collections")

    print("\n🤝 Your Project - Using Both is IDEAL:")
    print("-" * 80)
    print("  • LSA: Efficient for understanding variance and finding distinctive patterns")
    print("  • LDA: Better for interpreting topic mixtures and content themes")
    print("  • Cross-validation: Compare results to validate findings")
    print("  • Complementary insights: LSA shows 'what separates', LDA shows 'what comprises'")

    # ========================================================================
    # PART 7: VISUALIZATION
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("PART 7: VISUAL COMPARISON")
    print("=" * 80)

    create_comparison_visualizations(lsa_2006, lda_2006, '2006-2015')
    create_comparison_visualizations(lsa_2016, lda_2016, '2016-2025')

    # Export comparison table
    output_csv = '../data/08_reporting/lsa_vs_lda_comparison.csv'
    df_comparison.to_csv(output_csv, index=False)
    print(f"\n✅ Comparison table exported to: {output_csv}")

    print("\n" + "=" * 80)
    print("✅ COMPARISON ANALYSIS COMPLETE!")
    print("=" * 80)


def create_comparison_visualizations(lsa_results, lda_results, period_name):
    """Create side-by-side visualizations comparing LSA and LDA."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'LSA vs LDA Comparison - {period_name}', fontsize=16, fontweight='bold')

    # Get data
    lsa_explained_var = lsa_results['explained_variance']
    lsa_doc_topics = lsa_results['doc_topics'].filter(regex='^topic_')
    lda_doc_topics = lda_results['doc_topics'].filter(regex='^topic_')

    # Handle potentially different number of topics
    n_lsa_topics = len(lsa_explained_var)
    n_lda_topics = len(lda_doc_topics.columns)
    lsa_topic_ids = range(n_lsa_topics)
    lda_topic_ids = range(n_lda_topics)

    # Row 1: LSA
    # 1. LSA Explained Variance
    axes[0, 0].bar(lsa_topic_ids, lsa_explained_var * 100, color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('Topic ID')
    axes[0, 0].set_ylabel('Explained Variance (%)')
    axes[0, 0].set_title('LSA: Explained Variance')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. LSA Topic Prevalence
    lsa_prevalence = lsa_doc_topics.abs().mean()
    axes[0, 1].bar(lsa_topic_ids, lsa_prevalence, color='steelblue', alpha=0.7)
    axes[0, 1].set_xlabel('Topic ID')
    axes[0, 1].set_ylabel('Average |Weight|')
    axes[0, 1].set_title('LSA: Topic Prevalence')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. LSA Weight Distribution
    axes[0, 2].hist(lsa_doc_topics.values.flatten(), bins=50, color='steelblue', alpha=0.7)
    axes[0, 2].set_xlabel('Topic Weight')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('LSA: Weight Distribution (can be negative)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[0, 2].legend()

    # Row 2: LDA
    # 4. LDA Topic Prevalence
    lda_prevalence = lda_doc_topics.mean()
    axes[1, 0].bar(lda_topic_ids, lda_prevalence, color='coral', alpha=0.7)
    axes[1, 0].set_xlabel('Topic ID')
    axes[1, 0].set_ylabel('Average Probability')
    axes[1, 0].set_title('LDA: Topic Prevalence (Probabilities)')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. LDA Topic Entropy
    if 'topic_entropy' in lda_results['doc_topics'].columns:
        entropy = lda_results['doc_topics']['topic_entropy']
        axes[1, 1].hist(entropy, bins=20, color='coral', alpha=0.7)
        axes[1, 1].set_xlabel('Topic Entropy')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('LDA: Document Topic Diversity')
        axes[1, 1].grid(True, alpha=0.3)

    # 6. LDA Probability Distribution
    axes[1, 2].hist(lda_doc_topics.values.flatten(), bins=50, color='coral', alpha=0.7)
    axes[1, 2].set_xlabel('Topic Probability')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('LDA: Probability Distribution (all positive)')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[1, 2].legend()

    plt.tight_layout()

    # Save
    period_clean = period_name.replace('-', '_').replace('~', '_')
    output_path = f'../data/08_reporting/lsa_vs_lda_comparison_{period_clean}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")

    plt.close()


def main():
    """Main comparison workflow."""
    print("\n🔬 Starting LSA vs LDA Comparison Analysis...\n")

    try:
        compare_algorithms()

        print("\n📁 Generated Files:")
        print("  - data/08_reporting/lsa_vs_lda_comparison.csv")
        print("  - data/08_reporting/lsa_vs_lda_comparison_2006_2015.png")
        print("  - data/08_reporting/lsa_vs_lda_comparison_2016_2025.png")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
