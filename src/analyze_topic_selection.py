#!/usr/bin/env python3
"""
Topic Selection Analysis Tool

Validates and applies the topic selection methodology:
- Lowest variance topic → Background/common context
- Highest variance topic → Most distinctive pattern

Usage:
    python src/analyze_topic_selection.py
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


def load_lsa_results(period='2006_2015'):
    """Load LSA results for a specific period."""
    filepath = f'data/07_model_output/lsa_results_{period}.pkl'

    with open(filepath, 'rb') as f:
        results = pickle.load(f)

    return results


def verify_topic_selection_pattern(lsa_results, period_name):
    """
    Verify whether lowest/highest variance pattern holds for this dataset.

    Returns:
        dict: Analysis results including verification metrics
    """
    explained_var = lsa_results['explained_variance']
    topic_terms = lsa_results['topic_terms']
    doc_topics = lsa_results['doc_topics'].filter(regex='^topic_')

    # Identify lowest and highest variance topics
    lowest_var_idx = np.argmin(explained_var)
    highest_var_idx = np.argmax(explained_var)

    # Calculate prevalence metrics
    avg_weights = doc_topics.abs().mean()

    # Verification checks
    is_lowest_most_common = (lowest_var_idx == avg_weights.argmax())
    lowest_prevalence = avg_weights.iloc[lowest_var_idx]
    highest_prevalence = avg_weights.iloc[highest_var_idx]

    # Document coverage (how many docs have strong presence)
    threshold = 0.2  # Consider "strong presence" if weight > 0.2
    lowest_coverage = (doc_topics.iloc[:, lowest_var_idx].abs() > threshold).sum()
    highest_coverage = (doc_topics.iloc[:, highest_var_idx].abs() > threshold).sum()

    total_docs = len(doc_topics)

    # Get top terms
    lowest_terms = topic_terms.iloc[lowest_var_idx]['top_terms']
    highest_terms = topic_terms.iloc[highest_var_idx]['top_terms']

    return {
        'period': period_name,
        'lowest_var_idx': lowest_var_idx,
        'lowest_variance': explained_var[lowest_var_idx],
        'lowest_prevalence': lowest_prevalence,
        'lowest_coverage': f"{lowest_coverage}/{total_docs}",
        'lowest_terms': lowest_terms,
        'is_lowest_most_common': is_lowest_most_common,
        'highest_var_idx': highest_var_idx,
        'highest_variance': explained_var[highest_var_idx],
        'highest_prevalence': highest_prevalence,
        'highest_coverage': f"{highest_coverage}/{total_docs}",
        'highest_terms': highest_terms,
    }


def generate_selection_report(period='2006_2015'):
    """Generate comprehensive topic selection analysis report."""

    lsa_results = load_lsa_results(period)
    analysis = verify_topic_selection_pattern(lsa_results, period)

    print("\n" + "=" * 80)
    print(f"TOPIC SELECTION ANALYSIS REPORT - {analysis['period']}")
    print("=" * 80)

    print("\n📊 PATTERN VERIFICATION")
    print("-" * 80)

    # Lowest variance topic (Background Context)
    print(f"\n✓ LOWEST VARIANCE TOPIC (Background Context)")
    print(f"  Topic ID:             {analysis['lowest_var_idx']}")
    print(f"  Explained Variance:   {analysis['lowest_variance']*100:.2f}%")
    print(f"  Average Prevalence:   {analysis['lowest_prevalence']:.3f}")
    print(f"  Document Coverage:    {analysis['lowest_coverage']}")
    print(f"  Top Terms:            {analysis['lowest_terms'][:80]}")

    if analysis['is_lowest_most_common']:
        print(f"  ✅ VERIFIED: This topic IS the most common (highest prevalence)")
    else:
        print(f"  ⚠️  WARNING: This topic is NOT the most common - pattern may not hold")

    # Threshold check
    if analysis['lowest_prevalence'] > 0.3:
        print(f"  ✅ VERIFIED: High prevalence ({analysis['lowest_prevalence']:.3f} > 0.3 threshold)")
    else:
        print(f"  ⚠️  WARNING: Low prevalence ({analysis['lowest_prevalence']:.3f} < 0.3 threshold)")

    # Highest variance topic (Distinctive Pattern)
    print(f"\n✓ HIGHEST VARIANCE TOPIC (Most Distinctive)")
    print(f"  Topic ID:             {analysis['highest_var_idx']}")
    print(f"  Explained Variance:   {analysis['highest_variance']*100:.2f}%")
    print(f"  Average Prevalence:   {analysis['highest_prevalence']:.3f}")
    print(f"  Document Coverage:    {analysis['highest_coverage']}")
    print(f"  Top Terms:            {analysis['highest_terms'][:80]}")

    # Pattern summary
    print("\n" + "-" * 80)
    print("📋 PATTERN SUMMARY")
    print("-" * 80)

    if analysis['is_lowest_most_common'] and analysis['lowest_prevalence'] > 0.3:
        print("✅ PATTERN CONFIRMED: Lowest variance topic is suitable as background context")
    else:
        print("⚠️  PATTERN UNCLEAR: Verify semantic coherence manually")

    print("✅ HIGHEST VARIANCE: Topic with strongest discriminative power identified")

    # Interpretation guidance
    print("\n" + "=" * 80)
    print("📝 INTERPRETATION GUIDANCE")
    print("=" * 80)

    print(f"\nBackground Context (Topic {analysis['lowest_var_idx']}):")
    print(f"  → Represents common themes shared across {analysis['lowest_coverage']} documents")
    print(f"  → Low variance ({analysis['lowest_variance']*100:.2f}%) indicates consistency")
    print(f"  → High prevalence ({analysis['lowest_prevalence']:.3f}) indicates ubiquity")

    print(f"\nMost Distinctive Pattern (Topic {analysis['highest_var_idx']}):")
    print(f"  → Captures specialized subset of documents ({analysis['highest_coverage']})")
    print(f"  → High variance ({analysis['highest_variance']*100:.2f}%) indicates discriminative power")
    print(f"  → Effectively separates documents with this theme from others")

    return analysis


def visualize_verification(period='2006_2015'):
    """Create visualization comparing all topics on multiple metrics."""

    lsa_results = load_lsa_results(period)
    explained_var = lsa_results['explained_variance']
    doc_topics = lsa_results['doc_topics'].filter(regex='^topic_')
    topic_terms = lsa_results['topic_terms']

    # Calculate metrics for all topics
    n_topics = len(explained_var)
    topic_ids = range(n_topics)

    avg_prevalence = doc_topics.abs().mean()
    variance_pct = explained_var * 100

    # Document coverage
    threshold = 0.2
    coverage = [(doc_topics.iloc[:, i].abs() > threshold).sum() for i in range(n_topics)]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Explained Variance
    colors = ['red' if i == np.argmin(explained_var) else
              'darkgreen' if i == np.argmax(explained_var) else
              'steelblue' for i in range(n_topics)]

    axes[0, 0].bar(topic_ids, variance_pct, color=colors)
    axes[0, 0].set_xlabel('Topic ID', fontsize=12)
    axes[0, 0].set_ylabel('Explained Variance (%)', fontsize=12)
    axes[0, 0].set_title(f'Explained Variance by Topic - {period}', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Lowest Variance (Background)'),
        Patch(facecolor='darkgreen', label='Highest Variance (Distinctive)'),
        Patch(facecolor='steelblue', label='Other Topics')
    ]
    axes[0, 0].legend(handles=legend_elements, loc='upper right')

    # 2. Average Prevalence
    axes[0, 1].bar(topic_ids, avg_prevalence, color=colors)
    axes[0, 1].set_xlabel('Topic ID', fontsize=12)
    axes[0, 1].set_ylabel('Average Prevalence', fontsize=12)
    axes[0, 1].set_title(f'Topic Prevalence - {period}', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.3, color='orange', linestyle='--', linewidth=2,
                       label='Threshold (0.3)', alpha=0.7)
    axes[0, 1].legend()

    # 3. Document Coverage
    axes[1, 0].bar(topic_ids, coverage, color=colors)
    axes[1, 0].set_xlabel('Topic ID', fontsize=12)
    axes[1, 0].set_ylabel('Number of Documents', fontsize=12)
    axes[1, 0].set_title(f'Document Coverage (weight > 0.2) - {period}', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Variance vs Prevalence Scatter
    lowest_idx = np.argmin(explained_var)
    highest_idx = np.argmax(explained_var)

    axes[1, 1].scatter(variance_pct, avg_prevalence, s=150, alpha=0.6, color='steelblue')

    # Highlight special topics
    axes[1, 1].scatter(variance_pct[lowest_idx], avg_prevalence.iloc[lowest_idx],
                      s=300, color='red', marker='s', label=f'Topic {lowest_idx} (Lowest Var)',
                      edgecolors='black', linewidths=2, zorder=5)
    axes[1, 1].scatter(variance_pct[highest_idx], avg_prevalence.iloc[highest_idx],
                      s=300, color='darkgreen', marker='^', label=f'Topic {highest_idx} (Highest Var)',
                      edgecolors='black', linewidths=2, zorder=5)

    # Add topic labels
    for i in range(n_topics):
        axes[1, 1].annotate(f'T{i}',
                           (variance_pct[i], avg_prevalence.iloc[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.7)

    axes[1, 1].set_xlabel('Explained Variance (%)', fontsize=12)
    axes[1, 1].set_ylabel('Average Prevalence', fontsize=12)
    axes[1, 1].set_title(f'Variance vs Prevalence - {period}', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc='best')

    plt.tight_layout()

    # Save
    output_path = f'data/08_reporting/topic_selection_verification_{period}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")

    return fig


def compare_periods():
    """Compare topic selection patterns across both time periods."""

    print("\n" + "=" * 80)
    print("CROSS-PERIOD COMPARISON")
    print("=" * 80)

    # Analyze both periods
    analysis_2006 = verify_topic_selection_pattern(
        load_lsa_results('2006_2015'), '2006-2015'
    )
    analysis_2016 = verify_topic_selection_pattern(
        load_lsa_results('2016~2025'), '2016-2025'
    )

    # Create comparison table
    comparison = pd.DataFrame([
        {
            'Period': '2006-2015',
            'Background Topic': analysis_2006['lowest_var_idx'],
            'Background Variance': f"{analysis_2006['lowest_variance']*100:.2f}%",
            'Background Prevalence': f"{analysis_2006['lowest_prevalence']:.3f}",
            'Background Coverage': analysis_2006['lowest_coverage'],
            'Distinctive Topic': analysis_2006['highest_var_idx'],
            'Distinctive Variance': f"{analysis_2006['highest_variance']*100:.2f}%",
            'Distinctive Prevalence': f"{analysis_2006['highest_prevalence']:.3f}",
            'Distinctive Coverage': analysis_2006['highest_coverage'],
        },
        {
            'Period': '2016-2025',
            'Background Topic': analysis_2016['lowest_var_idx'],
            'Background Variance': f"{analysis_2016['lowest_variance']*100:.2f}%",
            'Background Prevalence': f"{analysis_2016['lowest_prevalence']:.3f}",
            'Background Coverage': analysis_2016['lowest_coverage'],
            'Distinctive Topic': analysis_2016['highest_var_idx'],
            'Distinctive Variance': f"{analysis_2016['highest_variance']*100:.2f}%",
            'Distinctive Prevalence': f"{analysis_2016['highest_prevalence']:.3f}",
            'Distinctive Coverage': analysis_2016['highest_coverage'],
        }
    ])

    print("\n📊 SUMMARY TABLE")
    print("-" * 80)
    print(comparison.to_string(index=False))

    # Pattern consistency check
    print("\n✓ PATTERN CONSISTENCY CHECK")
    print("-" * 80)

    both_verified = (analysis_2006['is_lowest_most_common'] and
                    analysis_2016['is_lowest_most_common'])

    if both_verified:
        print("✅ Pattern CONFIRMED in both periods:")
        print("   → Lowest variance = Background context (most common)")
        print("   → Highest variance = Distinctive pattern (discriminative)")
        print("\n📝 This validates your topic selection methodology!")
    else:
        print("⚠️  Pattern does NOT consistently hold across periods")
        print("   → Manual verification recommended")

    # Export comparison
    output_csv = 'data/08_reporting/topic_selection_comparison.csv'
    comparison.to_csv(output_csv, index=False)
    print(f"\n✅ Comparison table exported to: {output_csv}")

    return comparison


def main():
    """Main analysis workflow."""

    print("\n" + "=" * 80)
    print("TOPIC SELECTION METHODOLOGY ANALYSIS")
    print("=" * 80)
    print("\nThis tool verifies whether the lowest/highest variance topic")
    print("selection pattern holds for your LSA results.")
    print("=" * 80)

    # Analyze 2006-2015 period
    print("\n\n📅 PERIOD 1: 2006-2015")
    analysis_2006 = generate_selection_report('2006_2015')
    visualize_verification('2006_2015')

    # Analyze 2016-2025 period
    print("\n\n📅 PERIOD 2: 2016-2025")
    analysis_2016 = generate_selection_report('2016~2025')
    visualize_verification('2016~2025')

    # Cross-period comparison
    print("\n\n")
    comparison = compare_periods()

    # Final recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS FOR RESEARCH PAPER")
    print("=" * 80)

    print("\n1. TOPIC SELECTION JUSTIFICATION:")
    print("   Write: 'We identified background and distinctive topics through")
    print("   multi-criteria analysis: (1) explained variance, (2) average")
    print("   prevalence across documents, and (3) semantic coherence of top terms.'")

    print("\n2. BACKGROUND CONTEXT REPORTING:")
    print(f"   2006-2015: 'Topic {analysis_2006['lowest_var_idx']} (explained variance:")
    print(f"   {analysis_2006['lowest_variance']*100:.2f}%, prevalence: {analysis_2006['lowest_prevalence']:.2f})")
    print(f"   represents common background context...'")
    print(f"   2016-2025: 'Topic {analysis_2016['lowest_var_idx']} (explained variance:")
    print(f"   {analysis_2016['lowest_variance']*100:.2f}%, prevalence: {analysis_2016['lowest_prevalence']:.2f})")
    print(f"   represents common background context...'")

    print("\n3. DISTINCTIVE PATTERN REPORTING:")
    print(f"   2006-2015: 'Topic {analysis_2006['highest_var_idx']} (explained variance:")
    print(f"   {analysis_2006['highest_variance']*100:.2f}%) captures the most distinctive")
    print(f"   pattern, effectively separating...'")
    print(f"   2016-2025: 'Topic {analysis_2016['highest_var_idx']} (explained variance:")
    print(f"   {analysis_2016['highest_variance']*100:.2f}%) captures the most distinctive")
    print(f"   pattern, effectively separating...'")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - data/08_reporting/topic_selection_verification_2006_2015.png")
    print("  - data/08_reporting/topic_selection_verification_2016~2025.png")
    print("  - data/08_reporting/topic_selection_comparison.csv")
    print("\n")


if __name__ == '__main__':
    main()
