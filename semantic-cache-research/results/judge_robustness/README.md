# Supplementary Audit Sensitivity Files

These files contain optional sensitivity analyses for the fixed 90-row audit.
They are separate from the primary 700-query threshold sweep and are not used
to determine its results. The audit predates recovery of 37 paraphrases, so its
threshold replay is descriptive only.

Files include row-level audit assessments, agreement summaries, threshold
replays, and provenance in `summary.json`. Regenerate from the repository root
with `python -m analysis.judge_robustness`; outputs are written to this folder.
