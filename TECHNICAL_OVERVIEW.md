# Semantic Caching for RAG/LLM Pipelines — Technical Overview

## Problem

LLM/RAG systems repeatedly re-answer semantically identical or near-identical queries at full generation cost. Semantic caching addresses this by embedding incoming queries and returning a stored answer when a sufficiently similar query has already been answered — instead of re-invoking the LLM.

The unsolved part: tightening the similarity threshold to reduce **false hits** (returning a cached answer that is actually wrong for the new query) also reduces the **cost savings** the cache exists to provide. No existing system quantifies this tradeoff under controlled conditions. Prior work (GPTCache, MeanCache) either leaves the threshold-vs-false-positive relationship as unaddressed future work, or measures it against naturally-occurring query traffic — where you don't control how semantically close a "duplicate" pair actually is, and can't isolate exactly where the cache starts making mistakes.

## What we're building

A semantic caching layer, instrumented as a research testbed rather than a production service, to measure that tradeoff directly.

**Pipeline:**
```
query → embed → nearest-neighbor lookup (vector index) → similarity ≥ τ ?
                                                          ├─ yes → return cached answer (hit)
                                                          └─ no  → call LLM, cache new (query, answer) pair (miss)
```

**Components:**
- **Embedding model:** sentence-transformer (`all-MiniLM-L6-v2` as baseline — matches GPTCache's default, giving a fair comparison point).
- **Vector index:** FAISS / in-process cosine similarity (deliberately not Redis in the critical path, to avoid network-latency confounds in the correctness measurements).
- **Cached answers:** the dataset's own gold answers, not LLM-generated ones — this removes generation-quality as a second confound on top of retrieval correctness.
- **LLM fallback:** a fixed local model via Ollama (temperature 0), used only to measure real cost/latency avoided on a hit — not used to judge correctness.
- **Instrumentation:** every request logged in full (query, embedding, retrieved match, similarity score, threshold, hit/miss decision, ground-truth label, latency, tokens).

## The actual research contribution

Not the cache itself — the cache is a fairly standard architecture. The contribution is a **controlled experimental workload with engineered ground truth**, built to isolate the tradeoff:

- **Paraphrase group** — reworded queries with the *same* correct answer.
- **Hard-negative group** — queries that are lexically/topically close but have a *different* correct answer (built via entity-swap templates, not free-form generation, so the ground-truth answer change is guaranteed, not assumed).
- **Unrelated group** — control baseline.

Because ground truth is engineered into the dataset itself, "was this a false hit" is a lookup against a known label, not a post-hoc judgment call.

## Experiments (isolated variables, one at a time)

| # | Variable | Held fixed | Output |
|---|---|---|---|
| A | Similarity threshold τ, swept 0.70–0.99 | everything else | False-hit rate vs. cost-savings curve — the headline result |
| B | Embedding model (compare 2) | threshold sweep, dataset | Does embedding quality shift the tradeoff curve? |
| C | Cache scale (e.g. 200 / 2k / 10k background entries) | threshold, embedding model | Does false-hit rate rise as the corpus grows (collision risk)? |

Each metric is reported with a confidence interval, not a bare percentage.

## Stack

Python, sentence-transformers, FAISS, Ollama (local LLM inference) — no paid API dependency, fully reproducible without external cost.

## Status

4-week execution plan (build → dataset construction → experiments → analysis), scoped so A+B is the non-negotiable minimum result if the timeline compresses, C is dropped first if needed.