# Semantic Caching Research Project

This is an undergraduate research project on the correctness–cost trade-off in a static semantic cache for question-answering queries. The experiment controls query variation, reuses one fixed cache, and replays stored retrieval and judge results over a threshold grid.

## Current status

- The active branch is `develop`.
- The main corpus contains 700 unique evaluation queries: 100 each of P1–P4, 50 each of hard-negative types E/N/T/S, and 100 unrelated queries.
- The corpus, cache responses, retrieval results, fallback responses, judge verdicts, threshold metrics, and paper figures/tables have been generated.
- The primary threshold sweep covers four matchers and thresholds 0.50–0.95. No tested configuration reaches the paper draft's planned judged-false-hit target of 5%; treat the results as a trade-off study, not a validated deployment setting.
- The 90-row stratified audit is a limited diagnostic sampled before the corpus grew from 663 to 700 queries. It is supplementary and does not replace the full-corpus metrics.
- The cache-scale experiment is configured but has not been run.

## Start here

1. [`PROJECT_STATE.md`](PROJECT_STATE.md) — current, authoritative project context, completed work, results, limitations, and next steps.
2. [`corpus_spec.md`](corpus_spec.md) — corpus construction rules and achieved audit/corpus facts.
3. [`semantic-cache-research/configs/corpus.yaml`](semantic-cache-research/configs/corpus.yaml) — authoritative machine-readable corpus and model configuration.
4. [`research_paper.md`](research_paper.md) — paper draft. Its abstract is the author-provided text; the Results section records the measured sweep and audit findings.
5. [`semantic-cache-research/results/threshold_sweep/`](semantic-cache-research/results/threshold_sweep/) — primary experiment records and metrics.

## Repository layout

- `semantic-cache-research/src/` — cache, matchers, model wrappers, and corpus-generation modules.
- `semantic-cache-research/scripts/` — corpus, cache, audit, and experiment entry points.
- `semantic-cache-research/configs/` — corpus and experiment settings.
- `semantic-cache-research/data/processed/` — frozen corpus, manifests, audit sample, and generation provenance.
- `semantic-cache-research/results/threshold_sweep/` — primary run.
- `semantic-cache-research/results/judge_robustness/` — supplemental audit sensitivity analysis.
- `semantic-cache-research/results/cache_scale/` — reserved for an unrun secondary experiment.
- `semantic-cache-research/analysis/` — metric, plotting, paper-output, and supplemental-analysis code.

## Reproducing outputs

Run commands from `semantic-cache-research/`. The main experiment uses local Ollama models and cached sentence-transformer models; missing model outputs may trigger inference. Existing result files are reused by query/configuration ID.

```powershell
python scripts/run_experiments.py
python -m analysis.judge_robustness
python -m analysis.paper_outputs
```

The last two commands read stored outputs. `run_experiments.py` is the model-backed step. Read [`semantic-cache-research/data/processed/README.md`](semantic-cache-research/data/processed/README.md) before rebuilding the corpus, since that step regenerates data and can make model calls.
