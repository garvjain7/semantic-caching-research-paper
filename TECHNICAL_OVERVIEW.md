# Technical Overview

## Research question

How does a static semantic cache's similarity threshold affect the correctness of reused answers and the number of model calls avoided, under a controlled mix of paraphrases, typed hard negatives, and unrelated queries?

## System

Each cache entry is an anchor question paired with a response generated once by `llama3.2:3b` at temperature 0. For each evaluation query, a matcher retrieves one nearest anchor and stores the score. A hit serves the anchor's cached response; a miss uses the stored fallback response. The cache does not insert on a miss during the experiment.

The same query/configuration retrieval and `phi4-mini` verdict are reused at every threshold. The runner also stores one `llama3.2:3b` fallback response and latency per query. This makes the threshold sweep a replay over persisted records rather than repeated model inference.

## Evaluation data

The final corpus has 100 anchors, 50 disjoint dev anchors, 1,000 held-out questions, and 700 unique evaluation queries: 100 each of P1–P4, 50 each of E/N/T/S, and 100 unrelated. NQ-Open is pinned to revision `5dd9790a83002ad084ddeb7c420dc716852c6f28`; sampling uses seed 42. See [`corpus_spec.md`](corpus_spec.md) and `semantic-cache-research/data/processed/final_corpus_manifest.json`.

## Matchers and threshold grid

1. all-MiniLM-L6-v2 cosine similarity.
2. BGE-base-en-v1.5 cosine similarity.
3. MiniLM top-5 retrieval followed by `cross-encoder/quora-distilroberta-base` reranking.
4. Python `difflib` sequence ratio.

Dense search uses a FAISS flat inner-product index over normalized embeddings. The tested threshold grid is 0.50 through 0.95 in steps of 0.05.

## Metrics and current result

The primary correctness metric is the fraction of cache hits whose stored response receives a `false_hit` verdict from Phi-4-mini. Label-based query false-hit rate is a separate measure; false-miss rate, call savings, latency savings, subtype rates, judge/label disagreement, and Wilson 95% intervals are also reported.

No matcher reaches a judged false-hit rate of 5% anywhere on the grid. The lowest observed judged false-hit rate is 35.8% for MiniLM plus reranking at τ=0.90, with 36.3% call savings. This is an evaluation result, not a deployment recommendation. Full tables are in `semantic-cache-research/results/threshold_sweep/`.

## Secondary work

The 90-row audit is a limited, stratified sample from the 663-query pre-recovery corpus. Its additional sensitivity files are supplementary and do not replace the 700-query sweep. The cache-scale configuration exists, but the scale experiment has not been run.

## Code map

- `semantic-cache-research/src/cache.py`, `match_strategies.py`, `embedding.py`: cache decision and matcher implementations.
- `semantic-cache-research/src/experiment_runner.py`: retrieval, fallback, and threshold replay stages.
- `semantic-cache-research/scripts/`: corpus, cache, audit, and main experiment entry points.
- `semantic-cache-research/analysis/`: metrics, plots, generated tables, and supplemental audit analysis.
- [`PROJECT_STATE.md`](PROJECT_STATE.md): current project state and reproduction instructions.
