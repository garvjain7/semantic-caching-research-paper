# Publication results

These figures and tables are generated from the persisted 700-query threshold sweep. They do not require model inference.

From `semantic-cache-research/`, regenerate them with:

```powershell
python -m analysis.paper_outputs
```

The root manuscript `research_paper.md` embeds the publication figures 5–10 and reports the primary result tables. CSVs provide the complete threshold and subtype detail, including counts and Wilson 95% intervals:

- `table_1_corpus_audit.csv`: corpus counts, stratum sizes and lexical covariates.
- `table_2_threshold_metrics.csv`: all four configurations over the ten thresholds.
- `table_3_subtype_metrics.csv`: query subtype metrics over the full sweep.
- `table_4_best_observed_operating_points.csv`: minimum observed JFH per configuration.
- `table_5_shared_threshold_comparison.csv`: common operating point at τ=0.75.
- `table_6_hard_negative_breakdown.csv`: E/N/T/S results at τ=0.75.
- `table_7_subtype_judge_disagreement.csv`: judge–label disagreement by subtype at τ=0.75.
- `table_8_distance_bin_metrics.csv`: false-hit counts and intervals by lexical covariate bin at τ=0.75.
- `table_9_overall_judge_label_disagreement.csv`: overall disagreement rates at τ=0.75.

Label FHR is false hits divided by cache hits. Query false-hit incidence is false hits divided by all evaluation queries. A paraphrase is label-correct only when the retrieved nearest neighbour is its own source anchor. Judge-based JFH is the primary measure and is separately calculated from persisted verdicts.
