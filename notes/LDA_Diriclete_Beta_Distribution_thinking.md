# Dirichlet Distribution and Beta DIstribution with LDA

## Dirichlet Distribution

The **Dirichlet distribution** is a multivariate continuous probability distribution over probability vectors. If you have a K-dimensional Dirichlet distribution with parameter vector **α** = (α₁, α₂, ..., αₖ), it generates probability vectors **θ** = (θ₁, θ₂, ..., θₖ) where:

- Each θᵢ ∈ [0, 1]
- Σθᵢ = 1 (they sum to 1)

The probability density function is:

**p(θ|α) = (1/B(α)) ∏ᵢ θᵢ^(αᵢ-1)**

where B(α) is the multivariate Beta function (normalization constant).

### Key Properties

- **Conjugate prior** for the multinomial/categorical distribution
- The α parameters control the concentration:
  - α < 1: sparse distributions (mass concentrated on few components)
  - α = 1: uniform over the simplex
  - α > 1: dense distributions (mass spread across components)

## Why Dirichlet is Used in LDA

**Latent Dirichlet Allocation** uses the Dirichlet distribution for two key reasons:

### 1. **Conjugacy with Multinomial Distribution**

LDA models two processes:
- **Document-topic distribution**: each document has a mixture of topics (multinomial)
- **Topic-word distribution**: each topic has a distribution over words (multinomial)

The Dirichlet is the conjugate prior for the multinomial, meaning:
- Prior: Dirichlet(α)
- Likelihood: Multinomial
- Posterior: also Dirichlet

This makes inference (especially Gibbs sampling) mathematically tractable and efficient.

### 2. **Modeling Sparsity**

In LDA:
- **θ ~ Dir(α)**: document-topic proportions
- **φ ~ Dir(β)**: topic-word distributions

By setting α and β < 1, we encourage:
- Documents to focus on a few topics (not all topics equally)
- Topics to use a subset of vocabulary (not all words equally)

This reflects real-world text: documents usually cover a few themes, and topics are characterized by specific vocabularies.

### 3. **Interpretability**

The Dirichlet's concentration parameters let you encode prior beliefs:
- Small α → documents are about few topics
- Small β → topics are specific (few characteristic words)

This makes LDA's assumptions explicit and tunable.

---

**In summary**: The Dirichlet distribution is perfect for LDA because it generates probability vectors (needed for topic/word distributions), it's mathematically convenient as a conjugate prior, and it naturally encourages the sparse, interpretable topic structures we want to discover in text.
## Beta Distribution

The **Beta distribution** is a continuous probability distribution defined on the interval [0, 1]. It's parameterized by two positive shape parameters α and β:

**p(θ|α, β) = (1/B(α,β)) θ^(α-1) (1-θ)^(β-1)**

where B(α,β) is the Beta function (normalization constant).

## Relationship: Beta as Special Case of Dirichlet

**The Beta distribution IS the Dirichlet distribution for K=2.**

- **Beta(α, β)** generates a single probability θ ∈ [0, 1]
- This implicitly defines two probabilities: (θ, 1-θ) that sum to 1
- **Dirichlet(α₁, α₂, ..., αₖ)** generalizes this to K dimensions

So understanding Beta helps build intuition for Dirichlet!

## Key Characteristics of Beta (Relevant to LDA)

### 1. **Shape Behavior Based on Parameters**

The α and β parameters control the distribution shape:

| Parameters | Shape | Meaning |
|------------|-------|---------|
| α < 1, β < 1 | U-shaped | Prefers extreme values (close to 0 or 1) |
| α = 1, β = 1 | Uniform | All values equally likely |
| α > 1, β > 1 | Bell-shaped | Concentrated around the mean |
| α = β | Symmetric | Balanced around 0.5 |
| α ≠ β | Skewed | Favors one side |

### 2. **Sparsity Control**

**When α, β < 1**: The distribution has mass concentrated near 0 and 1
- This creates **sparsity** - values tend to be extreme
- In LDA context: encourages "on/off" behavior

**When α, β > 1**: The distribution concentrates around the mean
- This creates **density** - values avoid extremes
- In LDA context: encourages even mixing

### 3. **Mean and Concentration**

- **Mean**: μ = α/(α+β)
- **Concentration**: α + β (sum controls how peaked the distribution is)
  - Small sum → high variance (diffuse)
  - Large sum → low variance (concentrated)

## How This Relates to Dirichlet in LDA

### **Beta Intuition → Dirichlet Understanding**

In a 2-topic LDA model:
- **Document-topic distribution**: θ ~ Beta(α₁, α₂)
  - θ = probability of topic 1
  - 1-θ = probability of topic 2

If α₁ = α₂ = 0.1 (< 1):
- Documents will strongly prefer ONE topic
- Few documents will mix both topics equally
- Creates **sparse topic assignments**

### **Generalization to K Topics**

For K topics, **θ ~ Dirichlet(α, α, ..., α)**:

- **α < 1**: Each document uses few topics (sparse)
  - Like K-dimensional version of U-shaped Beta
  - Most θᵢ values near 0, few near 1
  
- **α = 1**: Uniform over simplex
  - Like flat Beta(1,1)
  - All topic mixtures equally likely

- **α > 1**: Documents use many topics (dense)
  - Like bell-shaped Beta
  - Topic proportions more balanced

### **Example: Visualizing α Effects**

For 3 topics (visualized on a simplex):

```
α = 0.1 (sparse):        α = 1 (uniform):        α = 10 (dense):
     ●                        ○○○                      ●
    / \                      /   \                    ●●●
   /   \                    /  ○  \                  ●●●●●
  ●-----●                  ○-------○                ●●●●●●●
(corners)                 (everywhere)              (center)
```

## Beta Properties Critical for LDA

### 1. **Conjugacy**
- Beta is conjugate prior to Binomial
- Dirichlet is conjugate prior to Multinomial
- **Result**: Closed-form posterior updates in inference

### 2. **Exchangeability**
- Beta doesn't care about order of observations
- Dirichlet preserves this for K dimensions
- **Result**: Bag-of-words assumption in LDA works

### 3. **Concentration Interpretation**

Think of α parameters as **pseudo-counts**:
- Beta(α, β) = "start with α successes, β failures"
- Dirichlet(α₁, ..., αₖ) = "start with αᵢ observations of topic i"

**In LDA**:
- Small α values (e.g., 0.1) = weak prior, data dominates
- Large α values (e.g., 50) = strong prior, forces smoothing

### 4. **Sparsity-Density Trade-off**

The most important property for LDA:

**α < 1** (Beta U-shaped → Dirichlet corner-seeking):
- Documents → few topics each
- Topics → few characteristic words
- **Creates interpretable, focused topics**

**α > 1** (Beta bell-shaped → Dirichlet center-seeking):
- Documents → many topics mixed
- Topics → many words mixed
- **Creates diffuse, harder-to-interpret topics**

## Practical Example

Suppose you're modeling news articles with 20 topics:

**α = 0.01** (very sparse):
- A sports article: 95% sports topic, 5% other
- Topics are highly specific: "basketball" vs "football" separate

**α = 5** (dense):
- A sports article: 30% sports, 20% politics, 15% business...
- Topics blur together: "sports-business-politics" mixed

---

**Key Insight**: The Beta distribution's ability to create U-shaped (sparse) or bell-shaped (dense) distributions based on its parameters is exactly what makes Dirichlet powerful in LDA. By setting α < 1, LDA naturally discovers the sparse, interpretable topic structures that exist in real text, because the Beta/Dirichlet prior expects and encourages this sparsity.


## LDA Parameters used in this project
From `conf/base/parameters.yml`:

```yaml
lda:
  doc_topic_prior: null  # alpha (α)
  topic_word_prior: null  # beta (η)
  ```

### Actual Values Used
Since both are set to null, scikit-learn automatically calculates default values:

Parameter|	Config Value|	Actual Value|	Formula
---------|--------------|---------------|--------------
alpha (α)|	null	    |   0.1	        |   1/n_topics = 1/10
beta (η) |	null	    |   0.1	        |   1/n_topics = 1/10

### What These Values Mean
#### Alpha (α = 0.1) - Document-Topic Prior
- α = 0.1 < 1  → Sparse topic distributions (fewer topics per document)

    -  Effect: Encourages each document to focus on a small number of topics rather than being evenly distributed across all 10 topics. 

    - Example:If α = 0.1: Document might be 70% Topic 1, 20% Topic 3, 10% Topic 5 ✅
    - If α = 10: Document would be ~10% for each of the 10 topics (uniform)
    
#### Beta (η = 0.1) - Topic-Word Prior
- η = 0.1 < 1  → Sparse word distributions (fewer words per topic)
    - Effect: Encourages each topic to use a focused vocabulary rather than all words equally. Example:
    - If η = 0.1: Topic might focus on {部落, 泰雅族, 文化} with high probability ✅
    - If η = 10: Topic would have nearly uniform probability across all vocabulary
Verification
You can verify these values were actually used by checking the fitted model:
import pickle

# Load LDA results

```python
with open('data/07_model_output/lda_results_2006_2015.pkl', 'rb') as f:
    lda_data = pickle.load(f)


# Check actual parameter values
lda_model = lda_data['lda_model']
print(f"Alpha (doc_topic_prior): {lda_model.doc_topic_prior_}")
print(f"Beta (topic_word_prior): {lda_model.topic_word_prior_}")
```

Expected output: Both should be approximately 0.1 (or arrays of 0.1 repeated for each topic/word).
How to Customize
If you want different sparsity levels, modify the configuration:
# conf/base/parameters.yml
lda:
  doc_topic_prior: 0.01  # More sparse (documents focus on fewer topics)
  topic_word_prior: 0.01  # More sparse (topics use fewer words)
Rules of thumb:
α < 1: Sparse topic distributions (recommended for focused documents)
α = 1: Neutral prior (Dirichlet uniform)
α > 1: Dense topic distributions (documents use many topics)
For your small corpus (13-25 documents), the default α = β = 0.1 is appropriate