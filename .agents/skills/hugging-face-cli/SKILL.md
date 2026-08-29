---
name: hugging-face-cli
description: >-
  Manage models, datasets, repositories, spaces, inference endpoints, and cloud buckets using the Hugging Face Hub CLI (`hf`).
  Use when downloading weights, uploading checkpoints/datasets, managing HF auth tokens, or querying repo metadata.
---

# Hugging Face CLI (`hf`)

The official Hugging Face Hub CLI provides high-performance operations for Hub resources.

## Key Commands

- `hf auth login` / `hf auth whoami` / `hf auth list`: Manage authentication tokens.
- `hf download REPO_ID [--local-dir PATH --include PATTERN]`: Download models, weights, or datasets from Hugging Face.
- `hf upload REPO_ID LOCAL_PATH [--type model|dataset|space --private]`: Upload files/folders in single or resumable commits.
- `hf upload-large-folder REPO_ID LOCAL_PATH`: Resumable parallel upload for large checkpoints/datasets.
- `hf cache list` / `hf cache prune` / `hf cache rm`: Inspect and clean local HF cache.
- `hf datasets list [--search QUERY]` / `hf datasets info DATASET_ID`: Browse Hub datasets and query parquet metadata.
- `hf endpoints deploy / list / pause / resume`: Manage dedicated GPU Inference Endpoints.
