# Summary: Kedro Viz Fixed and Running
## Issue: The MatplotlibWriter dataset type in the catalog was causing kedro-viz to fail during initialization. Fix Applied:
 - Commented out the unused MatplotlibWriter datasets in conf/base/catalog.yml
Installed kedro-viz properly
Started the viz server with correct syntax
Kedro Viz is now accessible at:
`http://0.0.0.0:4141/`
or
`http://localhost:4141/`
What You Can Do with Kedro Viz:
View Pipeline Structure - See all nodes and connections visually
Explore Data Flow - Track how data flows through the pipeline
Check Node Status - See which nodes have run successfully
Inspect Datasets - View what datasets are produced and consumed
Compare Pipelines - Switch between data_processing, lsa, and lda pipelines
To Stop Kedro Viz:
When you're done, you can stop it with:

```bash
lsof -ti:4141 | xargs kill
```

Jupyter Notebook Successfully Started
Access URLs:
http://127.0.0.1:8888/tree?token=cdc590761f8955379cf26b13466d95f49bd621bdc1a5d1d1
or
http://localhost:8888/tree?token=cdc590761f8955379cf26b13466d95f49bd621bdc1a5d1d1
Token: cdc590761f8955379cf26b13466d95f49bd621bdc1a5d1d1
What's Special About Kedro Jupyter?
When you start Jupyter through kedro jupyter notebook, it automatically:
Pre-loads Kedro Context - The context variable is available in your notebooks
Access to Catalog - Use context.catalog to load datasets
Custom Kernel - Uses kedro_lsa_markus kernel with project dependencies
Project Environment - All your project code is importable
Example Usage in Jupyter Notebook
Create a new notebook and try this:
# Kedro automatically provides these variables
%reload_kedro

# Access the data catalog
catalog = context.catalog

# Load LSA results
lsa_results = catalog.load("lsa_results_2006_2015")
lda_results = catalog.load("lda_results_2006_2015")

# View topic terms
print(lsa_results['topic_terms'])
print(f"Total variance explained: {lsa_results['total_variance_explained']:.2%}")

# Load comparison reports
import pandas as pd
lsa_comparison = catalog.load("lsa_topic_comparison")
lda_comparison = catalog.load("lda_topic_comparison")

print(lsa_comparison)
Useful Kedro Magic Commands in Jupyter
%reload_kedro              # Reload the Kedro session
%load_ext kedro.ipython    # Load Kedro IPython extension
Currently Running Services
Service	Port	URL
Kedro Viz	4141	http://localhost:4141/
Jupyter Notebook	8888	http://localhost:8888/ (with token)
To Stop Services
Stop Jupyter:
lsof -ti:8888 | xargs kill
Stop Kedro Viz:
lsof -ti:4141 | xargs kill
Notebooks Directory
Your notebooks are saved in:
/Users/qs.chou/projects/lsa-markus/notebooks/
You can now explore your LSA and LDA results interactively in Jupyter! 📊
