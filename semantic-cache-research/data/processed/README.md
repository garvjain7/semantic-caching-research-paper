# Processed data

This directory holds regenerable pipeline outputs. Build them in this order
from `semantic-cache-research/`:

1. `python scripts/build_corpus.py` writes sampled anchors, dev anchors,
   held-out questions, and unrelated queries.
2. `python scripts/generate_queries.py` writes `test_queries.jsonl` and a
   generation manifest. Generation uses the configured Gemma 3:4b model.
3. `python scripts/build_cache.py` writes one Llama 3.2:3b response per anchor.
   It also writes `final_corpus_manifest.json` with record counts and SHA-256
   hashes for the sampled and generated corpus files.
4. `python scripts/run_experiments.py` stores retrievals, judge verdicts,
   fallback responses, and threshold metrics under `results/threshold_sweep/`.
5. `python -m analysis.judge_robustness` runs a supplemental sensitivity
   analysis of the saved 90-row audit. It writes only to
   `results/judge_robustness/`; the threshold replay there is descriptive and
   is not a full-corpus estimate.
6. `python -m analysis.paper_outputs` writes Figures 1–3 and Tables 1–3
   from the stored metrics and the AI robustness outputs.

Generated files are preserved in this working copy. The audit sample was
drawn from the 663-query pre-recovery corpus; the final corpus contains 700
queries. See `verification_report.md` for scope and corpus diagnostics.
