# embs_basics

## Getting started
```sh
pip install -r requirements.in
```

## scripts
- cluster_prompts.py - takes file of prompts, does clusters using kmeans clustering and then writes a json file with text closest to center of cluster as key and all texts in cluster as values.
- semantic_search.py - Sets up a faiss index across the texts which can then be used for basic semantic search. Recommend running as a jupytext notebook to avoid rebuilding the index each search.