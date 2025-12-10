# History of approaches

## Summary Timeline
- 1983: TF-IDF (Salton & McGill)
  ↓ Problem: No dimensionality reduction, no semantics
  
- 1990: LSI/SVD (Deerwester et al.)
  ↓ Problem: No probabilistic model
  
- 1999: pLSI (Hofmann)
  ↓ Problem: Overfitting, no model for new docs
  
- 2003: LDA (Blei, Ng, Jordan) ✅
  → Fixed all previous shortcomings
  → Proper Bayesian generative model
  → Foundation for modern topic modeling
This progression shows the field moving from:
. Heuristics (TF-IDF)
. → Linear algebra (LSI)
. → Probabilistic models (pLSI)
. → Fully Bayesian frameworks (LDA)



### Development History from IR to LDA: Summary from Blei et al. (2003)
Based on the introduction section of the seminal LDA paper, here's the historical development:
#### 1. Information Retrieval (IR) & TF-IDF Era
Foundation: Traditional IR methodology (Baeza-Yates & Ribeiro-Neto, 1999)
Goal: Find short descriptions of document collections for efficient processing while preserving statistical relationships
Method: TF-IDF scheme (Salton & McGill, 1983)
How TF-IDF Works:
Choose a basic vocabulary of "words" or "terms"
For each document, count occurrences of each word (term frequency)
Compare to inverse document frequency (IDF) - measures word occurrence across entire corpus (log scale)
Result: Term-by-document matrix X with TF-IDF values
Strengths:
✅ Identifies discriminative word sets for documents ✅ Successfully deployed in modern Internet search engines
Shortcomings:
❌ Small reduction in description length ❌ Reveals little inter- or intra-document statistical structure ❌ Limited ability to capture semantic relationships
#### 2. Latent Semantic Indexing (LSI) / SVD Era
Innovation: Deerwester et al. (1990)
Method: Singular Value Decomposition (SVD) of the TF-IDF matrix X
How LSI Works:
X ≈ U Σ V^T

Where SVD identifies a linear subspace in tf-idf feature space
that captures most variance in the collection
Advances:
✅ Significant compression in large collections ✅ Derived features (linear combinations of tf-idf) capture linguistic notions:
Synonymy: Different words, same meaning
Polysemy: Same word, different meanings
Limitations:
❌ No probabilistic foundation - just linear algebra ❌ Unclear how to recover generative model from data ❌ Not a proper generative model for new documents
#### 3. Probabilistic LSI (pLSI) / Aspect Model
Innovation: Hofmann (1999)
Breakthrough: First probabilistic approach to topic modeling
How pLSI Works:
p(d, wₙ) = p(d) Σ_z p(wₙ|z)p(z|d)

- Each word is a sample from a mixture model
- Mixture components = multinomial random variables = "topics"
- Different words in document can come from different topics
Key Concept:
Document represented as mixing proportions for topics
This distribution = "reduced description" of document
Advances Over LSI:
✅ Proper probabilistic model
✅ Multiple topics per document (not just single cluster) 
✅ Generative semantics for words
Critical Shortcomings (Why LDA was needed):
Problem 1: No document-level probabilistic model
Each document = list of numbers (mixing proportions)
No generative model for these numbers
Not well-defined for new documents outside training set
Problem 2: Linear parameter growth
Parameters = kV + kM
where M = number of documents

→ Parameters grow with corpus size!
Consequences: ❌ Serious overfitting (Table 1 in paper shows perplexity explosion) ❌ Cannot assign probability to new documents properly ❌ Requires "folding-in" heuristic for test documents (unfair advantage)

#### 4. Latent Dirichlet Allocation (LDA) - The Solution
Innovation: Blei, Ng, Jordan (2003)
Foundation: Exchangeability + de Finetti's theorem
Theoretical Insight:
Exchangeability assumption:
Words in document are exchangeable (order doesn't matter)
Documents in corpus are exchangeable
de Finetti's Representation Theorem states:
Any infinitely exchangeable sequence must have representation as a mixture distribution
How LDA Works:
Three-level hierarchical Bayesian model:

1. Corpus level: α, β (parameters)
2. Document level: θ ~ Dirichlet(α) (topic proportions)
3. Word level: 
   - Choose topic: zₙ ~ Multinomial(θ)
   - Choose word: wₙ ~ p(w|zₙ, β)
Key Formula:
p(θ, z, w|α, β) = p(θ|α) ∏ₙ p(zₙ|θ)p(wₙ|zₙ, β)

#### 5. Comparison: Evolution of Capabilities

Aspect |	TF-IDF|	LSI|	pLSI|	LDA
-------| ---------| ---| -------| ------
Foundation|	Count ratios|	Linear algebra (SVD)|	Probabilistic mixture|	Bayesian generative
Model Type|	None	    |Deterministic.         |	Probabilistic.       |	Fully Bayesian
Multiple Topics/Doc|	❌|	❌	|✅	|✅
Generative Model|	❌|	❌	|Partial	|✅ Complete
New Documents	|✅	|✅	|❌ (heuristic)|	✅ Natural
Parameters|	Fixed|	O(kV)|O(kV + kM)|	O(k + kV)
Overfitting| N/A|	Minimal|	Severe|	Minimal
Semantic Capture|	❌	|✅|	✅|	✅


-------------------------------------------------------------------------------------
##### Persepective of LSA and LDA difference on the `Multiple Topics/Doc` aspective
Why LSI/LSA Gets ❌ for Multiple Topics per Document
The Core Issue: LSI's Document Representation
LSI/LSA represents each document as a single point in topic space:
After SVD decomposition:
Document d → Single vector in k-dimensional topic space

Document representation = [weight₁, weight₂, ..., weightₖ]

This is a FIXED point, not a mixture
Example to Illustrate:
Imagine a document about "police investigating hunting accident in indigenous community": LSI/LSA Representation:
Document vector in 3-topic space:
[0.65, 0.30, 0.05]

Topic 1 (Safety/Police):    0.65
Topic 2 (Indigenous Culture): 0.30
Topic 3 (Government):         0.05
What this means:
Document is projected onto these topics
The weights show correlation with topic directions
BUT: This is NOT saying "65% about police, 30% about culture"
It's saying "document aligns most strongly with police topic direction"
LSI vs LDA: The Key Difference
LSI/LSA (❌ No True Mixture):
1. Perform SVD: X ≈ U Σ V^T
2. Document d → Row in U matrix
3. Result: Single fixed vector [w₁, w₂, ..., wₖ]

Interpretation:
- Weights can be NEGATIVE
- Weights don't sum to 1
- NOT a probability distribution
- NOT a mixture of topics

→ Document is a LINEAR COMBINATION in latent space
→ NOT a mixture of multiple discrete topics
LDA (✅ True Mixture):
1. Each document has topic distribution θ ~ Dirichlet(α)
2. θ = [p₁, p₂, ..., pₖ] where Σpᵢ = 1, all pᵢ ≥ 0
3. Each WORD drawn from a topic: zₙ ~ Multinomial(θ)
4. Different words → different topics

Interpretation:
- "This document is 65% topic 1, 30% topic 2, 5% topic 3"
- Each WORD has its own topic assignment
- True generative mixture model

→ Document IS a mixture of multiple topics
The Critical Distinction (From Blei's Paper)
What Blei Says About LSI (Page 2):
"LSI uses a singular value decomposition of the X matrix to identify a linear subspace in the space of tf-idf features..."
Key phrase: "linear subspace" - not "mixture of topics"
What Blei Says About Mixture of Unigrams (Page 8):
"Under this mixture model, each document is generated by first choosing a topic z and then generating N words independently..."
This is still ❌ because:
Document = ONE topic (choose z once)
All words from that single topic
NOT multiple topics per document
What Blei Says About pLSI (Page 8-9):
"The pLSI model attempts to relax the simplifying assumption made in the mixture of unigrams model that each document is generated from only one topic... However, it is important to note that d is a dummy index..."
pLSI gets ✅ because:
p(z|d) = mixing proportions for document d
Different words can come from different topics
BUT: Overfitting issues (parameters grow with corpus)
What Blei Says About LDA (Page 8):
"In contrast, the LDA model allows documents to exhibit multiple topics to different degrees."
LDA gets ✅ with no caveats:
Proper generative model
Each word independently chooses a topic
θ ~ Dirichlet(α) = proper probability distribution
Mathematical Proof: Why LSI ≠ Multiple Topics
LSI's SVD:
X ≈ U Σ V^T

Document d representation = U[d, :] = [u_d1, u_d2, ..., u_dk]

Properties:
1. Values can be NEGATIVE ❌
2. Values don't sum to 1 ❌
3. No probabilistic interpretation ❌
4. Just projection coefficients in latent space
LDA's Generative Model:
For document d:
1. Draw θ_d ~ Dirichlet(α)
   θ_d = [θ₁, θ₂, ..., θₖ] where Σθᵢ = 1, all θᵢ ≥ 0 ✅

2. For each word n in document d:
   a. Draw topic: z_dn ~ Multinomial(θ_d) ✅
   b. Draw word: w_dn ~ Multinomial(β_{z_dn}) ✅

Result: Each word has its own topic!
→ TRUE mixture of multiple topics ✅
Visual Comparison (From Blei's Paper - Figure 4)
Geometric Interpretation (Page 9):
Mixture of Unigrams:
Places each document at one corner of topic simplex
❌ No mixture
pLSI:
Places documents as empirical distribution on topic simplex
✅ Mixture (but only for training docs)
LDA:
Places smooth distribution over topic simplex
✅ Proper mixture with Dirichlet prior
LSI: (Not shown in figure because it doesn't operate on simplex)
Documents in Euclidean space, not probability simplex
❌ Not a mixture at all
Updated Comparison Table (Clarified)
Aspect	TF-IDF	LSI/LSA	pLSI	LDA
Multiple Topics/Doc	❌	❌	✅	✅
Interpretation	Counts	Projection in latent space	Mixture (training only)	True generative mixture
Weights	Real+	Real (can be negative)	Probabilities	Probabilities
Sum to 1?	❌	❌	✅	✅
Generative?	❌	❌	Partial	✅ Full
Practical Example: Your Data
Imagine analyzing a news article about "police investigating accident in indigenous area":
LSI/LSA Output:
doc_vector = [-0.45, 0.82, -0.12, 0.33, ...]
#              ^^^^^  ^^^^^
#           Negative! Not probabilities!

Interpretation:
- Strong negative correlation with topic 0
- Strong positive correlation with topic 1
- This is PROJECTION, not mixture
LDA Output:
doc_topics = [0.05, 0.65, 0.25, 0.03, 0.02]
#             All positive, sum to 1.0 ✅

Interpretation:
- 65% about safety/police (topic 1)
- 25% about indigenous culture (topic 2)  
- 5% about general news (topic 0)
- etc.

Word-level assignments:
- "police" → topic 1 (safety)
- "部落" → topic 2 (indigenous)
- "發現" → topic 0 (general news)
Summary: Why LSI Gets ❌
LSI/LSA does NOT model multiple topics per document because:
❌ No probabilistic mixture - just linear projection
❌ Weights can be negative - not probabilities
❌ No generative model - can't say "word came from topic X"
❌ Not interpretable as mixture - weights ≠ topic proportions
What LSI DOES: ✅ Projects documents into lower-dimensional semantic space ✅ Captures variance in corpus ✅ Enables similarity comparisons ✅ Reveals latent semantic structure But it does NOT: ❌ Model documents as mixtures of topics ❌ Assign topics to individual words ❌ Provide probability distributions over topics
Final Answer
The ❌ is correct because: LSI = Dimensionality reduction via linear algebra
Documents → vectors in latent space
Good for similarity, clustering, retrieval
NOT a mixture model
LDA = Generative probabilistic mixture model
Documents → probability distributions over topics
Each word → drawn from a specific topic
TRUE mixture of multiple topics
Your project uses both correctly:
LSA/LSI: For variance analysis and distinctive pattern finding
LDA: For probabilistic topic mixture modeling
Both are valuable, but they answer different questions!
```
-----------------------------------------------------------------------------------------------


#### 6. LDA's Improvements Over pLSI
Fixed pLSI Problem 1: Proper Generative Model
✅ Topic mixture weights θ are random variables drawn from Dirichlet(α)
Not fixed parameters tied to training documents
Easy to generate new documents with new topic mixtures
Fixed pLSI Problem 2: Constant Parameters
✅ Parameters = k + kV (doesn't grow with M)
k parameters for Dirichlet α
kV parameters for topic-word distributions β
No overfitting issues
Additional Advantages:
✅ Modular - can embed in larger models ✅ Extensible - can add time, metadata, continuous data ✅ Interpretable - Dirichlet provides smooth distribution on topic simplex
7. Next Milestones (Discussed in Paper)
Inference Methods:
Variational Bayes (presented in paper) - fast approximation
Expectation Propagation (Minka & Lafferty, 2002) - higher accuracy
Gibbs Sampling (Griffiths & Steyvers, 2002) - MCMC approach
Future Extensions (mentioned):
✅ Continuous data (Gaussian LDA) ✅ Dirichlet mixtures for richer topic structure ✅ Temporal models (partial exchangeability) ✅ Conditional models (supervised LDA) ✅ N-gram extensions beyond bag-of-words

###  Historical Context of LSI and LSA (Actually same thing)
1990 - Deerwester et al. chose "LSI":
Emphasized Indexing application
Target audience: Information retrieval researchers
Goal: Better document search/retrieval
Later - Others called it "LSA":
Emphasized Analysis of meaning
Used in cognitive science, psychology
Goal: Understanding semantic relationships
Same Math, Different Perspectives:
Aspect	LSI (IR View)	LSA (NLP View)
Goal	Find relevant documents	Understand word meanings
Input	Document collection	Text corpus
Output	Document similarity scores	Semantic space
Metric	Retrieval precision/recall	Semantic coherence
Math	SVD on TF-IDF	SVD on TF-IDF
In Summary:
✅ LSI = LSA (mathematically identical)
Blei's paper uses "LSI" because it's comparing to IR methods
Your project uses "LSA" which is perfectly fine
Both refer to SVD-based dimensionality reduction on text
