# Topic Selection Best Practices for LSA/LDA Interpretation

## Your Observation

**Hypothesis**: Can we always use:
- **Lowest variance topic** → Background/common context
- **Highest variance topic** → Most distinctive/discriminative pattern

## Testing on Your Data

### 2006-2015 Period
- ✅ **Lowest variance (Topic 0, 1.22%)**: IS the most common (部落, 泰雅族, 原住民, 文化)
  - Avg weight: 0.44 (highest across all topics)
  - Present in 13/13 documents strongly
  - **Interpretation**: Common indigenous cultural context

- ✅ **Highest variance (Topic 1, 13.08%)**: Most distinctive pattern (警方, 現場, 樹林, 打獵)
  - Avg weight: 0.26
  - Strong in 6/13 documents only
  - **Interpretation**: Safety incidents - separates incident reports from other news

### 2016-2025 Period
- ✅ **Lowest variance (Topic 0, 1.82%)**: IS the most common (部落, 公所, 表示, 縣府, 遊客)
  - Avg weight: 0.46 (highest)
  - Present in 12/12 documents strongly
  - **Interpretation**: Government administration and tourism management baseline

- ✅ **Highest variance (Topic 1, 13.09%)**: Most distinctive (控溪 部落, 撤離, 溢流)
  - Avg weight: 0.28
  - Strong in 8/12 documents
  - **Interpretation**: Emergency evacuations at Konsi tribe

---

## Answer: Is This a Normal Practice?

### ⚠️ **PARTIALLY - With Important Caveats**

## When This Pattern Holds

### ✅ Works Well When:

1. **Your corpus has clear background context**
   - All documents share a common domain/theme
   - Example: All news about indigenous townships → tribal/cultural terms appear everywhere

2. **Your corpus has specialized subsets**
   - Some documents are very different from others
   - Example: Some articles about emergencies, others about culture

3. **Small to medium corpus size**
   - Pattern is clearer with fewer documents
   - Your case: 13 and 12 documents

4. **Domain-specific corpus**
   - Not general news, but focused topic
   - Your case: News specifically about Jianshi Township

### ❌ May NOT Work When:

1. **Diverse corpus with no common background**
   - Example: Random Wikipedia articles
   - Lowest variance topic might just be noise or artifact

2. **Very large corpus**
   - Background might be split across multiple low-variance topics
   - Highest variance might capture corpus size artifacts

3. **Highly balanced corpus**
   - All topics equally important
   - No clear "background" vs "distinctive" separation

4. **Preprocessing issues**
   - Poor stopword removal → lowest variance = common words (the, and, of)
   - Not meaningful background context

---

## Best Practice Recommendations

### 1. **Check, Don't Assume**

```python
# Always verify the pattern in your data
explained_var = lsa_results['explained_variance']
doc_topics = lsa_results['doc_topics'].filter(regex='^topic_')

# Lowest variance topic
lowest_idx = explained_var.argmin()
lowest_prevalence = doc_topics[f'topic_{lowest_idx}'].abs().mean()

# Check if it's actually common
is_background = lowest_prevalence > doc_topics.abs().mean().mean()
print(f"Lowest variance IS background: {is_background}")
```

### 2. **Use Multiple Indicators**

Don't rely on variance alone. Consider:

| Indicator | What It Tells You |
|-----------|------------------|
| **Explained Variance** | Importance for differentiating documents |
| **Average Prevalence** | How commonly the topic appears |
| **Topic Entropy** | How focused vs spread out the topic is |
| **Document Coverage** | In how many documents it appears strongly |

### 3. **Semantic Verification**

**ALWAYS** check the actual terms:

```python
# Does the lowest-variance topic actually represent background?
lowest_terms = lsa_results['topic_terms'].iloc[lowest_idx]['top_terms']
print(f"Lowest variance terms: {lowest_terms}")

# Ask yourself:
# - Are these domain-general terms?
# - Do they appear across most documents?
# - Do they represent shared context?
```

### 4. **Recommended Selection Strategy**

Instead of automatically using lowest/highest, use this approach:

#### For Background Context:
1. Find topics with **high average prevalence** (appears in many docs)
2. Among those, pick ones with **low variance** (consistent across docs)
3. **Verify** the terms make semantic sense as "background"

#### For Distinctive Patterns:
1. Find topics with **high explained variance**
2. Among those, pick ones with **moderate-to-high prevalence** (not just outliers)
3. **Verify** the terms represent meaningful themes, not noise

---

## Your Specific Case: Best Practice

### ✅ For Your Data - This IS a Good Practice

**Why it works for you:**

1. **Clear domain**: All articles about Jianshi Township
2. **Shared context**: Indigenous culture, local government (Topic 0)
3. **Distinctive events**: Emergencies, tourism issues (Topic 1)
4. **Small corpus**: Pattern is stable and interpretable

### Recommended Interpretation Framework

```python
# For temporal comparison research
def interpret_period(lsa_results, period_name):
    explained_var = lsa_results['explained_variance']
    topic_terms = lsa_results['topic_terms']
    doc_topics = lsa_results['doc_topics'].filter(regex='^topic_')

    # Background context (lowest variance + high prevalence)
    lowest_idx = explained_var.argmin()
    if doc_topics[f'topic_{lowest_idx}'].abs().mean() > 0.3:  # Threshold
        background_topic = lowest_idx
        print(f"Background context: Topic {background_topic}")
        print(f"  Terms: {topic_terms.iloc[background_topic]['top_terms'][:80]}")

    # Most distinctive pattern (highest variance)
    highest_idx = explained_var.argmax()
    distinctive_topic = highest_idx
    print(f"\nMost distinctive: Topic {distinctive_topic}")
    print(f"  Terms: {topic_terms.iloc[distinctive_topic]['top_terms'][:80]}")

    # Secondary distinctive patterns (next 2-3 highest variance topics)
    top_indices = explained_var.argsort()[-4:-1][::-1]
    print(f"\nSecondary patterns:")
    for idx in top_indices:
        print(f"  Topic {idx} ({explained_var[idx]*100:.1f}%): {topic_terms.iloc[idx]['top_terms'][:60]}")

    return background_topic, distinctive_topic
```

---

## Research Paper Guidance

### When Writing Your Results

**DO:**
- ✅ "Topic 0 (lowest explained variance, 1.22%) represents the common background context shared across documents (indigenous tribal culture), appearing with high consistency (avg weight: 0.44) across all documents."

- ✅ "Topic 1 (highest explained variance, 13.08%) captures the most distinctive pattern (emergency incidents), effectively separating incident reports from other news types."

**DON'T:**
- ❌ "The lowest variance topic always represents background context."
- ❌ "We selected topics based solely on variance ranking."

**BETTER:**
- ✅ "We identified background and distinctive topics through multi-criteria analysis: (1) explained variance, (2) average prevalence across documents, and (3) semantic coherence of top terms. Topic 0 emerged as background context (low variance 1.22%, high prevalence 0.44), while Topic 1 represented the primary distinctive pattern (high variance 13.08%)."

---

## Conclusion

### Your Observation is VALID for your specific case!

**Pattern confirmed**:
- Both periods: Lowest variance = Background context ✅
- Both periods: Highest variance = Most distinctive ✅

**But it's not a universal rule - it's a heuristic that works when**:
1. Corpus has clear shared background
2. Specialized subsets exist
3. You verify it empirically

**Best practice**:
- Use this as a **starting hypothesis**
- Always **verify with your data**
- Check **semantic coherence** of selected topics
- Report **multiple metrics**, not just variance

**For your research**: This is a defensible and interpretable approach! Just make sure to document that you verified the pattern rather than assumed it.
