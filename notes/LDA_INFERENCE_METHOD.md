# LDA Inference Method Explained

## Quick Answer

**Q: Does your LDA use EM, Gibbs Sampling, or something else?**

**A: Stochastic Variational Inference (SVI)** ✅

---

## Your Implementation

From [src/lsa_markus/pipelines/lda_analysis/nodes.py](../src/lsa_markus/pipelines/lda_analysis/nodes.py:64-73):

```python
lda_model = LatentDirichletAllocation(
    n_components=parameters.get('n_topics', 10),
    max_iter=parameters.get('max_iter', 100),
    learning_method=parameters.get('learning_method', 'online'),  # ← KEY
    random_state=parameters.get('random_state', 42),
    n_jobs=parameters.get('n_jobs', -1),
    batch_size=parameters.get('batch_size', 128),
    doc_topic_prior=parameters.get('doc_topic_prior'),
    topic_word_prior=parameters.get('topic_word_prior')
)
```

**Configuration** ([conf/base/parameters.yml](../conf/base/parameters.yml:27-36)):
```yaml
lda:
  n_topics: 10
  max_iter: 100
  learning_method: "online"  # ← Uses Stochastic Variational Inference
  random_state: 42
  batch_size: 128
```

---

## What is Stochastic Variational Inference (SVI)?

### Algorithm Overview

SVI is a scalable variant of **Variational Bayes** that approximates the posterior distribution using optimization instead of sampling.

**How it works**:

```
For each iteration (up to max_iter=100):
  1. Sample a random mini-batch of documents (batch_size=128)

  2. E-step (Local Update):
     - For documents in this batch:
       - Compute variational parameters γ (document-topic distributions)
       - Estimate θ_d (topic proportions for document d)

  3. M-step (Global Update):
     - Update λ (topic-word distributions φ)
     - Use stochastic gradient with learning rate decay
     - Formula: λ_new = (1-ρ)λ_old + ρ·λ_batch
       where ρ is the learning rate

  4. Check Convergence:
     - Monitor perplexity or log-likelihood
     - Stop if converged or max_iter reached
```

### Key Characteristics

✅ **Faster than traditional methods**
- Processes documents in mini-batches (128 docs at a time)
- Doesn't require full corpus in memory
- Suitable for online learning (streaming data)

✅ **Deterministic with fixed seed**
- Your `random_state=42` ensures reproducibility
- Same results across runs (given same seed)

✅ **Variational approximation**
- Uses mean-field approximation (factorization assumption)
- Faster but less accurate than MCMC methods

---

## Comparison: LDA Inference Methods

| Method | Type | Speed | Accuracy | Scalability | Your Use |
|--------|------|-------|----------|-------------|----------|
| **Online (SVI)** | Variational | ⚡⚡⚡ Fast | ✓ Good | ✅ Excellent | ✅ **YES** |
| **Batch (VB-EM)** | Variational | ⚡⚡ Medium | ✓✓ Better | ✓ Good | ❌ No |
| **Collapsed Gibbs** | MCMC Sampling | ⚡ Slow | ✓✓✓ Best | ✓ Moderate | ❌ Not available |
| **Variational Gibbs** | Hybrid | ⚡⚡ Medium | ✓✓ Better | ✓✓ Good | ❌ Not available |

### Method Details

#### 1. Online Variational Bayes (SVI) ✅ YOUR CHOICE

**Scikit-learn**: `learning_method='online'`

**Pros**:
- Fast convergence with mini-batches
- Memory efficient (doesn't load all docs)
- Suitable for large corpora
- Good for streaming/online learning

**Cons**:
- Slightly less accurate than batch methods
- Sensitive to learning rate schedule
- May need more iterations for small corpora

**Best For**: Large corpora (>1,000 docs), streaming data, quick experiments

---

#### 2. Batch Variational Bayes (VB-EM)

**Scikit-learn**: `learning_method='batch'`

**Pros**:
- More accurate than online mode
- Better for small corpora
- Stable convergence

**Cons**:
- Slower (processes all docs each iteration)
- Requires full corpus in memory
- Not suitable for large datasets

**Best For**: Small corpora (<1,000 docs), when accuracy > speed

**To Use Batch Mode**:
```yaml
# conf/base/parameters.yml
lda:
  learning_method: "batch"  # Change from "online"
```

---

#### 3. Collapsed Gibbs Sampling (MCMC)

**Not available in scikit-learn**. Use `gensim` instead:

```python
from gensim.models import LdaModel
lda = LdaModel(corpus, num_topics=10, passes=100)
```

**Pros**:
- Most accurate (MCMC sampling)
- No mean-field approximation
- Better topic quality

**Cons**:
- Very slow (samples each word)
- Requires many iterations (100-1000 passes)
- Not deterministic (needs burn-in)

**Best For**: Research papers requiring highest quality, small corpora

---

## Mathematical Background

### Variational Bayes vs MCMC

**Goal**: Approximate the posterior distribution P(θ, φ | data)

**Variational Bayes (Your Method)**:
```
1. Choose a simpler distribution q(θ, φ) to approximate p(θ, φ | data)
2. Minimize KL divergence: KL(q || p)
3. Use coordinate ascent optimization (ELBO maximization)
4. Result: Fast approximation, deterministic
```

**MCMC (Gibbs Sampling)**:
```
1. Sample from the true posterior p(θ, φ | data) directly
2. Use Markov Chain to explore the distribution
3. Requires burn-in period, multiple chains
4. Result: Accurate samples, but slow and stochastic
```

### The Evidence Lower Bound (ELBO)

SVI maximizes the ELBO (Evidence Lower Bound):

```
ELBO = E_q[log p(w, z, θ, φ)] - E_q[log q(z, θ, φ)]
     = Data likelihood - KL divergence

Where:
- w = observed words
- z = latent topic assignments
- θ = document-topic distributions
- φ = topic-word distributions
- q = variational approximation
```

**Stochastic Gradient**:
```
Natural gradient on ELBO using mini-batches:
∇_λ ELBO ≈ (N/batch_size) × ∇_λ ELBO_batch

Where:
- λ = global variational parameters (topic-word distributions)
- N = total number of documents
- Scaling factor accounts for batch sampling
```

---

## Why Online Mode for Your Corpus?

**Your corpus**: 13-25 documents (very small)

### Current Setup (Online Mode)
✅ Fast execution
✅ Good enough for exploration
✅ Reproducible (random_state=42)

### Potential Improvement (Batch Mode)
✅ More accurate for small corpus
✅ Better convergence
✅ Still fast (only 13-25 docs)

### Recommendation

For your **small corpus**, consider switching to **batch mode** for marginally better results:

```yaml
# conf/base/parameters.yml
lda:
  learning_method: "batch"  # Change from "online"
  max_iter: 100  # May need fewer iterations
  batch_size: null  # Not used in batch mode
```

**Expected improvement**: ~1-3% better perplexity, more stable topic assignments

---

## Verification

To verify the inference method is working correctly:

```python
import pickle

# Load LDA results
with open('data/07_model_output/lda_results_2006_2015.pkl', 'rb') as f:
    lda = pickle.load(f)

# Check model attributes
print(f"Learning method: {lda['lda_model'].learning_method}")
print(f"Max iterations: {lda['lda_model'].max_iter}")
print(f"Batch size: {lda['lda_model'].batch_size}")
print(f"Final perplexity: {lda['perplexity']:.2f}")
print(f"Log-likelihood: {lda['log_likelihood']:.2f}")
```

---

## References

### Theoretical Papers

1. **Original LDA Paper**:
   - Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993-1022.

2. **Stochastic Variational Inference**:
   - Hoffman, M., Bach, F. R., & Blei, D. M. (2010). Online Learning for Latent Dirichlet Allocation. *NIPS 2010*.
   - Paper: https://papers.nips.cc/paper/2010/hash/71f6278d140af599e06ad9bf1ba03cb0-Abstract.html

3. **Variational Inference Tutorial**:
   - Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians. *Journal of the American Statistical Association*, 112(518), 859-877.

### Implementation Documentation

- **Scikit-learn LDA**: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
- **Algorithm Details**: https://scikit-learn.org/stable/modules/decomposition.html#latentdirichletallocation

---

## Summary

| Aspect | Your Implementation |
|--------|---------------------|
| **Inference Method** | Stochastic Variational Inference (SVI) |
| **Implementation** | scikit-learn `LatentDirichletAllocation` |
| **Learning Method** | `'online'` (mini-batch processing) |
| **Batch Size** | 128 documents per iteration |
| **Max Iterations** | 100 |
| **Random Seed** | 42 (for reproducibility) |
| **Alternative** | `'batch'` mode (recommended for your small corpus) |
| **NOT Using** | EM (traditional), Gibbs Sampling, Collapsed Gibbs |

---

**Created**: 2025-12-09
**Related Docs**:
- [LSA_vs_LDA_COMPARISON.md](LSA_vs_LDA_COMPARISON.md) - Full algorithmic comparison
- [LSA_LDA_QUICK_SUMMARY.md](LSA_LDA_QUICK_SUMMARY.md) - One-page summary
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Project quick reference
