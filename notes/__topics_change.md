Required Changes
We need to modify the parameters file to change n_components (LSA) and n_topics (LDA) from 10 to 4:
Read parameters.yml
Now I'll make the changes:
Edit parameters.yml
⎿
Modified
Perfect! Now let me show you a summary of the changes before running the pipeline:
Changes Made to conf/base/parameters.yml:
Before:
lsa:
  n_components: 10  # Number of topics/latent dimensions

lda:
  n_topics: 10  # Number of topics
After:
lsa:
  n_components: 4  # Number of topics/latent dimensions

lda:
  n_topics: 4  # Number of topics
What This Means:
LSA: Will now extract 4 latent topics instead of 10 (via TruncatedSVD)
LDA: Will now discover 4 probabilistic topics instead of 10
Alpha (LDA): Will auto-adjust to 1/4 = 0.25 (instead of 0.1)
Beta (LDA): Will auto-adjust to 1/4 = 0.25 (instead of 0.1)
Similarity matrices: Will be based on 4-dimensional semantic space
Now let me run the full pipeline:
Bash
⎿
Run full pipeline with 4 topics
IN
/Library/Frameworks/Python.framework/Versions/3.10/bin/kedro run
