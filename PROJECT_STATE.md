# Project State: Semantic Caching Research

_Updated 2026-10-30. This file is the human-readable current-state record. For exact parameters, `semantic-cache-research/configs/corpus.yaml` is authoritative; manifests and saved result files are authoritative for achieved counts and measurements._

## Project objective

Measure how a static semantic cache trades judged response correctness against model-call savings as its similarity threshold changes. The controlled workload is built from NQ-Open questions and contains paraphrases at four generated difficulty levels, four typed hard-negative classes, and unrelated questions. The study compares four matching configurations using one fixed cache and a replayable threshold sweep.

## Current status

- Repository branch: `develop`.
- Main corpus: 100 anchors, 50 disjoint development anchors, 1,000 held-out questions, and 700 evaluation queries. The manifest records 100 each for P1–P4, 50 each for E/N/T/S, and 100 unrelated queries.
- Dataset: NQ-Open `google-research-datasets/nq_open`, revision `5dd9790a83002ad084ddeb7c420dc716852c6f28`, splits `train` and `validation`; seed 42.
- Cache entries: 100 Llama 3.2:3b responses at temperature 0.
- Main run: 2,800 retrieval records, 2,800 Phi-4-mini verdicts, 700 fallback records, and ten thresholds for each of four configurations.
- Figures 1–3 and Tables 1–3 are generated under `semantic-cache-research/paper/generated/`.
- Supplemental sensitivity outputs for the 90-row audit are saved under `semantic-cache-research/results/judge_robustness/`. They are secondary to the 700-query threshold results.
- The cache-scale study is not run. Its configuration exists, but its runner/output are not part of the completed main pipeline.

## Method in use

### Corpus construction

1. Load the pinned NQ-Open revision, preserve its answer lists, filter malformed/ineligible questions, and greedily deduplicate by token-set Jaccard at 0.8, retaining the first occurrence.
2. Sample anchors, disjoint dev questions, a held-out pool, and unrelated questions with seed 42. Each unrelated question has maximum token Jaccard at most 0.2 to any anchor.
3. Generate P1–P4 paraphrases with `gemma3:4b`; validate and retry failures. The initial run missed 37 paraphrases; the retry recovered all 37, leaving 100 of each level.
4. Generate up to 50 E/N/T/S hard negatives per type. NOT_APPLICABLE is permitted; the achieved corpus has 50 for each type and no NOT_APPLICABLE rows in the final quotas.
5. Store token Jaccard and normalized character edit distance to the anchor (unrelated items use their designated maximum-overlap anchor).

### Cache experiment

- The cache is static during evaluation: it contains the 100 anchor responses and does not insert on a miss.
- Configurations: MiniLM cosine, BGE cosine, MiniLM top-5 with cross-encoder reranking, and lexical `difflib` ratio.
- Dense search uses normalized embeddings and exact FAISS flat inner product.
- The nearest neighbor is stored once per query/configuration, independently of threshold.
- `llama3.2:3b` is called once per evaluation query for a stored fallback response and latency. `phi4-mini` supplies one stored verdict per query/configuration.
- The sweep replays stored records over τ=0.50, 0.55, …, 0.95; it does not make model calls per threshold.
- Metrics include judged false-hit rate, label-based query false-hit rate, false-miss rate, call savings, subtype breakdowns, Wilson 95% intervals, and judge/label disagreement.

## Main measured findings

No tested configuration reached judged false-hit rate (JFH) ≤5% at any threshold. The lowest observed JFH for each configuration was:

| Configuration | τ at minimum JFH | Minimum JFH (Wilson 95% CI) | Call savings | Label false-hit rate | False-miss rate |
|---|---:|---:|---:|---:|---:|
| MiniLM cosine | 0.75 | 40.3% (35.7–45.1%) | 58.9% | 17.4% | 27.5% |
| BGE cosine | 0.95 | 38.1% (20.8–59.1%) | 3.0% | 1.4% | 97.3% |
| MiniLM + cross-encoder | 0.90 | 35.8% (30.2–41.9%) | 36.3% | 5.6% | 46.3% |
| Lexical `difflib` | 0.60 | 45.9% (40.1–51.9%) | 38.6% | 13.9% | 56.8% |

At the shared τ=0.75, judged false-hit rates were 40.3% (MiniLM), 42.7% (BGE), 37.8% (MiniLM reranking), and 56.8% (lexical). Across the tested methods, no threshold provided both low judged false-hit rates and useful savings. The reranked MiniLM configuration had the lowest observed JFH, but that minimum was still 35.8%; it is not a safe operating point.

## Audit scope

The 90-row diagnostic has ten items per stratum and predates recovery of 37 paraphrases; it represents the earlier 663-query version, not the final corpus, and is not a population estimate. Its per-stratum label comparisons are documented in `data/processed/verification_report.md`; main threshold metrics use all 700 queries and saved Phi-4-mini verdicts.

## Key artifacts

- Corpus specification: `corpus_spec.md`
- Machine-readable configuration: `semantic-cache-research/configs/corpus.yaml`
- Sampling manifest: `semantic-cache-research/data/processed/sampling_manifest.json`
- Generation manifest: `semantic-cache-research/data/processed/generation_manifest.json`
- Final corpus manifest and hashes: `semantic-cache-research/data/processed/final_corpus_manifest.json`
- Evaluation queries: `semantic-cache-research/data/processed/test_queries.jsonl`
- Primary records and metrics: `semantic-cache-research/results/threshold_sweep/`
- Audit analysis: `semantic-cache-research/results/judge_robustness/`
- Generated figures/tables: `semantic-cache-research/paper/generated/`
- Manuscript draft: `research_paper.md`

## Reproduction

Run from `semantic-cache-research/`:

```powershell
python scripts/run_experiments.py
python -m analysis.judge_robustness
python -m analysis.paper_outputs
```

The experiment runner is idempotent for persisted query/configuration records but may call Ollama for any missing records. The two analysis commands replay stored files. Rebuilding the corpus or cache is a separate, model-backed operation; preserve the current manifests and outputs before doing so.

## Remaining work and cautions

1. Review the filled manuscript and verify bibliography entries marked `check` in `claim_sheet.md` before submission.
2. The secondary cache-scale experiment is configured for 200/2,000/10,000 entries but is not implemented in the main runner and has no results.
3. Do not claim a threshold meeting a 5% JFH target: none did. Do not generalize the stratified 90-row diagnostic to the final corpus.
4. Generated text and local model outputs are preserved, but installed Ollama model builds are not pinned as immutable digests in the corpus manifest.
