---
name: hugging-face-datasets
description: >-
  Work with Hugging Face Datasets: streaming, processing, tokenization, parquet caching, DuckDB/SQL queries, and Hub publication.
  Use when loading large benchmarks, preparing multimodal vision/point annotations, or curating training datasets.
---

# Hugging Face Datasets

High-performance data loading, curation, and streaming.

## Core Operations

1. **Loading & Streaming**:
   ```python
   from datasets import load_dataset
   # In-memory dataset
   ds = load_dataset("dataset_name", split="train")
   # Streaming large out-of-memory dataset
   stream_ds = load_dataset("dataset_name", split="train", streaming=True)
   ```

2. **Transformations & Batching**:
   - `ds.map(fn, batched=True, num_proc=4)`: Fast multi-process feature extraction and tokenization.
   - `ds.filter(predicate)`: Filter bad samples, corrupted images, or empty labels.
   - `ds.to_parquet("local.parquet")`: Serialize to compact Arrow/Parquet format.

3. **SQL & DuckDB Exploration**:
   - Query remote dataset parquet files without full download:
     `hf datasets sql "SELECT count(*), avg(num_points) FROM 'hf://datasets/org/name/*.parquet'"`
