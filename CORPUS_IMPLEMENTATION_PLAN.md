# Corpus Implementation Record and Handoff

This file records the implemented corpus workflow and its current evidence. It replaces the original pre-implementation plan. For current overall status and main experimental findings, see [`PROJECT_STATE.md`](PROJECT_STATE.md). The authoritative numeric configuration is [`semantic-cache-research/configs/corpus.yaml`](semantic-cache-research/configs/corpus.yaml).

## Implemented corpus

- Source: NQ-Open (`google-research-datasets/nq_open`), revision `5dd9790a83002ad084ddeb7c420dc716852c6f28`, `train` and `validation` splits.
- Sampling seed: 42. Achieved 100 anchors, 50 disjoint dev anchors, 1,000 held-out questions, and 100 unrelated questions.
- Candidate filtering retained 91,461 of 91,535 rows; greedy token-Jaccard deduplication at threshold 0.8 retained 84,100.
- Final query set: 700 unique rows, comprising P1–P4 (100 each), E/N/T/S (50 each), and 100 unrelated.
- All 37 initially missing paraphrases were recovered in the retry workflow; final per-level failures are zero.
- The static cache contains 100 Llama 3.2:3b anchor responses at temperature 0. Model outputs and prompts are preserved in generated artifacts/manifests.

The sampling, generation, and final corpus manifests record exact counts, dataset revision, seed, and SHA-256 hashes. See `semantic-cache-research/data/processed/`.

## Implemented evaluation

The main runner computes four configurations for every query: MiniLM cosine, BGE cosine, MiniLM top-5 with cross-encoder reranking, and lexical `difflib` ratio. It stores the nearest neighbour and score, one fallback response/latency per query, and one Phi-4-mini verdict per query/configuration. Metrics are replayed at ten thresholds from 0.50 through 0.95; Wilson 95% intervals and subtype metrics are included.

Completed outputs are in `semantic-cache-research/results/threshold_sweep/`; the paper-output script writes Figures 1–3 and Tables 1–3 under `semantic-cache-research/paper/generated/`.

## Audit scope

The 90-row stratified sample was selected before the 37 paraphrase recoveries, from a 663-query version. It remains a useful diagnostic, but does not include all queries in the final 700-row set. The recorded ratings and detailed consistency analysis are in `semantic-cache-research/data/processed/` and `semantic-cache-research/results/judge_robustness/`. Treat the audit as supplementary; the main threshold analysis uses the full saved query/result set.

The audit's per-stratum constructed-label comparison is recorded in `data/processed/verification_report.md`. It is a small diagnostic and did not alter the final corpus; the primary evaluation uses all 700 queries.

## Re-running

From `semantic-cache-research/`:

```powershell
python scripts/run_experiments.py
python -m analysis.judge_robustness
python -m analysis.paper_outputs
```

`run_experiments.py` is the model-backed step and reuses saved records where possible. The latter two commands read saved results. Corpus generation and cache construction are separate operations and should not be run casually over the frozen artifacts.

## Remaining

- Review the completed Results/Discussion sections in `research_paper.md` and verify bibliography entries before submission.
- The configured cache-scale experiment has no implemented run or results. Do not treat it as complete.
- Verify bibliography entries marked `check` in `claim_sheet.md` before submission.
