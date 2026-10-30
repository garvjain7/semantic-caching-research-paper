# Dataset Verification Record

This record summarizes the frozen 700-query corpus. Counts and hashes in
`final_corpus_manifest.json`, `sampling_manifest.json`, and
`generation_manifest.json` are the detailed provenance sources.

## Corpus counts

| Artifact | Achieved |
|---|---:|
| Anchors / static cache entries | 100 |
| Disjoint development anchors | 50 |
| Held-out pool | 1,000 |
| P1, P2, P3, P4 | 100 each |
| E, N, T, S hard negatives | 50 each |
| Unrelated queries | 100 |
| Evaluation queries | 700 |
| Final missing paraphrases | 0 |

All 37 initially missing paraphrases were recovered in the recorded retry
workflow. The hard-negative quotas were met; final NOT_APPLICABLE counts are
zero. Historical invalid generation attempts are recorded in the generation
manifest and are not final-corpus omissions.

## Measured overlap covariates

Means are computed against the corresponding anchor. Unrelated queries use
their designated maximum-overlap anchor reference.

| Stratum | n | Mean token Jaccard | Mean normalized edit distance |
|---|---:|---:|---:|
| P1 | 100 | 0.4175 | 0.4472 |
| P2 | 100 | 0.3517 | 0.5549 |
| P3 | 100 | 0.3432 | 0.5600 |
| P4 | 100 | 0.2814 | 0.6211 |
| E | 50 | 0.3102 | 0.5714 |
| N | 50 | 0.7371 | 0.1852 |
| T | 50 | 0.3079 | 0.5870 |
| S | 50 | 0.2791 | 0.6181 |
| Unrelated | 100 | 0.1543 | 0.6957 |

## Audit scope

The 90-item stratified diagnostic was drawn from the 663-query corpus before
recovery of the 37 paraphrases. It is not a sample from the final 700-query
corpus and does not establish population-wide label accuracy. Against the
constructed subtype labels, the audit assessment matched 9/10 P1,
10/10 P2, 7/10 P3, 8/10 P4, 5/10 E, 9/10 N, 5/10 T, 8/10 S, and 10/10 unrelated
items. This is a diagnostic only; the corpus was retained unchanged.
