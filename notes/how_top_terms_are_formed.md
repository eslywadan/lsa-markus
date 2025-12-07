# How top terms are formed

1. LSA Top Terms Selection
Code: lsa_analysis/nodes.py:67-72
top_indices = topic.argsort()[-n_top_words:][::-1]
How it works:
Each topic is a vector of weights for all 254 words in your vocabulary
Weights represent how strongly each word loads on this semantic dimension
Can be positive or negative (indicating direction in semantic space)
Top 15 words = 15 highest absolute weights
Example from your data (LSA Topic 1):
警方 (police) = +0.2342 → strongly associated with this dimension
人瑞 (centenarian) = -0.1697 → negatively associated (inverse relationship)
This means: documents about police incidents tend NOT to mention centenarians
Interpretation:
Positive weights: words that co-occur together
Negative weights: words that distinguish this topic from others
Magnitude: how important the word is to this dimension
2. LDA Top Terms Selection
Code: lda_analysis/nodes.py:102-107
top_indices = topic.argsort()[-n_top_words:][::-1]
top_probs = [topic[i] for i in top_indices]
How it works:
Each topic is a probability distribution over all 254 words
Probabilities represent: "If I'm writing about this topic, how likely am I to use this word?"
All probabilities are positive and sum to ~89-90
Top 15 words = 15 highest probabilities
Example from your data (LDA Topic 3):
祖靈 (ancestral spirits) = 5.10 (5.69% of this topic)
突然 (suddenly) = 5.10 (5.69% of this topic)
意外 (accident) = 1.10 (1.23% of this topic)
Top 15 words together = 40.76% of the topic
Interpretation:
If a document is about this topic, these are the most likely words to appear
Probabilities show relative importance within the topic
More interpretable than LSA weights
3. Key Differences
Aspect	LSA	LDA
Values	Weights (-∞ to +∞)	Probabilities (0 to 1)
Meaning	Correlation with semantic dimension	Likelihood of word in topic
Negative values	Yes (inverse relationship)	No (always positive)
Normalization	No	Yes (sums to ~1)
Interpretability	Mathematical/abstract	Intuitive/probabilistic
4. Configuration
Both use n_top_words: 15 from parameters.yml:40:
topic_interpretation:
  n_top_words: 15  # Number of top words to display per topic
You can change this to show more or fewer terms per topic.
5. Visual Comparison
I've created a visualization showing the difference:
data/08_reporting/topic_term_selection_explained.png
The chart shows:
LSA (left): Weights span positive and negative values
LDA (right): All probabilities are positive
Summary
Top terms are selected by sorting all words in the vocabulary by their importance score (weight for LSA, probability for LDA) and taking the top N (default: 15). The key insight: These words co-occur frequently in the same documents, which is how both LSA and LDA discover that they belong together in a "topic."