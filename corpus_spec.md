
# Corpus Specification: Semi-Synthetic Semantic-Cache Evaluation Corpus

This document describes the implemented corpus and its construction rules. Exact configuration values live in `semantic-cache-research/configs/corpus.yaml`; achieved counts and hashes live in `semantic-cache-research/data/processed/final_corpus_manifest.json` and the sampling/generation manifests. The frozen corpus contains 700 evaluation queries.

The selected source is NQ-Open, a derived Natural Questions dataset, not the full Natural Questions release. Its dataset card describes records with a `question` string and an `answer` list of possible answer strings. Do not expect full-release annotation fields or Wikipedia page content. The card reports CC BY-SA 3.0; verify the applicable attribution/share-alike requirements before redistributing derived data. The selected repository revision is `5dd9790a83002ad084ddeb7c420dc716852c6f28`, recorded in `semantic-cache-research/configs/corpus.yaml`; use that immutable revision for all loads.

## Parameters

| Parameter | Configured / achieved |
|---|---:|
| Anchor pool (cache entries) | 100 / 100 |
| Dev anchors, disjoint from evaluated anchors | 50 / 50 |
| Held-out question pool | 1,000 / 1,000 |
| Paraphrases per anchor (P1–P4) | 4 / 400 total |
| Hard negatives per type (E/N/T/S) | maximum 50 / 200 total |
| Unrelated evaluation queries | 100 / 100 |
| Evaluation queries total | 700 / 700 |
| Stratified audit sample | 10 × 9 strata / 90 |
| Seed | 42 |

Models: `gemma3:4b` generates variants; `llama3.2:3b` produces static-cache and fallback responses at temperature 0; `phi4-mini` judges matched responses. The corpus generator and judge are from different model families. Model IDs and settings are in `configs/corpus.yaml`.

## Pipeline

**0. Setup.** Fix the seed. Record model names, versions and prompt hashes for every generation step.

**1. Extract.** Load only the `question` text and `answer` list from the selected NQ-Open release. Preserve the answer list as supplied; do not silently select one answer or convert the list into a single canonical answer. The full Natural Questions release has a different schema and is not the selected input. Do not carry unrelated dataset fields into the processed corpus.

**2. Filter candidates.** Keep questions with a non-empty answer list, 5 to 20 tokens long, and well-formed. Treat each NQ-Open answer as a possible reference answer. Drop near-duplicates with token-Jaccard of 0.8 or higher. Use lexical dedupe only, so no embedding model influences the corpus.

Tokenization is `unicode_word_tokens_lowercase_v1`: lowercase the text, then extract matches of the Unicode regex `[^\W_]+` (Unicode word characters except underscore). Punctuation and underscores are not tokens. Use this same tokenizer for question length, token-Jaccard, and overlap covariates.

The question validity checks are mechanical: reject empty/whitespace-only text; Unicode Cc control characters; text without an alphabetic character; URLs (scheme/`www.` URLs or domain-like names with a two-or-more-letter TLD); and detected HTML/Markdown markup (HTML tags, Markdown links, backticks, strong/underline/strikethrough markers, and heading/list prefixes). Do not require capitalization or a question mark. Reject a question when any one normalized token occurs in strictly more than half of its tokens. A source answer list is eligible if it contains at least one non-whitespace answer; preserve the original list unchanged.

Report counts for each rejection rule and the number of unique rejected records. Rule counts are non-exclusive: one record may be counted under multiple rules. After eligibility filtering, deduplicate greedily in configured split order (`train`, then `validation`) and source row order. Retain the first record unless its token-set Jaccard with any previously retained record is at least the configured threshold (default 0.8). Compare only against retained records; a record dropped as a duplicate does not become a representative for later comparisons. The implementation may use a prefix index to reduce candidate comparisons, but must exactly match this greedy Jaccard rule.

**3. Split.** With seed 42, randomly sample 100 anchors, 50 disjoint dev anchors, a held-out pool of at least 1,000 questions, and 100 unrelated questions. The unrelated items must have maximum token-Jaccard <= 0.2 against every anchor.

**4. Cache entries.** For each anchor, generate one response with Llama 3.2:3b (Ollama) at temperature 0. Store response text, latency, token counts, model, and prompt hash. The cache holds all 100 anchors with their responses. A paraphrase's intended correct match is its own anchor.

**5. Paraphrases.** The prompt carries 5 to 8 real NQ-Open questions from the dev anchors as style examples, and G must match the anchor's casing, punctuation and register (except at P3, where register is the change). This keeps generated queries from reading like model output. G receives the anchor and its answer list as possible reference answers, and must keep the information need and answerability unchanged. The paraphrase inherits the anchor's source answer list as reference metadata; the generator must not rewrite that list. Output is JSON only, with no explanations and no quotation marks around the query, one paraphrase per level:

- P1: light edit (synonym swap, small word-order change)
- P2: syntactic restructure
- P3: register change (casual or keyword-style)
- P4: heavy rewording with low word overlap

Automated checks: differs from the anchor, preserves numbers and years from the anchor, token-length ratio between 0.5 and 2. Up to 3 retries, then drop. The levels are the intended difficulty, and the actual difficulty is measured using stored overlap covariates.

**6. Hard negatives.** One generation prompt per type, with the same style examples and style-matching rule as step 5. Each must edit only the target element, keep the sentence frame, read as a natural question, and have a different correct answer from the anchor's. G may return NOT_APPLICABLE.

- E: entity substitution
- N: negation or polarity flip
- T: numeric or temporal change
- S: scope narrowing or broadening

G also returns a one-line statement of how the answer should differ. It is stored for the audit only and is never used by the cache. This statement is not a verified answer. Do not label a generated answer claim as `gold_answers` without independent verification. Iterate anchors in seeded random order until each type's quota of at most 50 is filled or the pool is exhausted, and report the achieved count.

**7. Unrelated.** Sample 100 unrelated questions and retain each NQ-Open answer list. Keep a question only if its maximum token-Jaccard against every anchor is 0.2 or less. Store its nearest-overlap anchor as the covariate reference. These are real queries with no correct cache entry, so any hit on them is a false hit.

**8. Covariates.** For every test query record token-Jaccard and character-level normalized Levenshtein distance against its anchor (or the explicitly marked maximum-overlap anchor for unrelated queries).

**9. Stratified audit diagnostic.** Stratify into 9 strata: P1, P2, P3, P4, E, N, T, S, unrelated. Draw 10 per stratum. Use the MiniLM cosine configuration for the cached response/neighbor shown in each rating file. Raters do not receive the intended subtype label and record:

- (a) same information need, yes or no
- (b) for hard negatives, whether the correct answer differs from the anchor's
- (c) whether the anchor's cached response is acceptable for this query, yes or no

Compare assessments with constructed subtype labels by stratum as a limited diagnostic. It does not establish population-level accuracy. In this sample, P3, P4, E, T, and S fall below 90%; they remain in the evaluation corpus. The audit was selected from the 663-query pre-recovery set and does not include the 37 recovered paraphrases. Primary threshold metrics use all 700 queries.

**10. Freeze.** Hash the final corpus. Write `data/processed/verification_report.md`. Prompts are frozen after tuning on the 50 dev anchors.

## Row schema

`query_id, anchor_id, question, role, subtype, should_hit, gold_answers, jaccard_to_anchor, edit_dist_to_anchor, generator_model, prompt_hash, seed, audit_status`

For anchors and unrelated items, `gold_answers` is the source NQ-Open answer list. Paraphrases inherit the anchor's source answer list. For hard negatives, populate `gold_answers` only after independent verification; keep any unverified generated answer claim separate from gold data. The answer-difference statement remains audit-only.

## Evaluation efficiency (keep out of the sweep loop)

- The nearest neighbour of a test query does not change with the threshold. Judge each (configuration, query) pair once and reuse the verdict across all thresholds.
- Call the fallback LLM once per test query, store its latency, and replay it when a threshold produces a miss. Do not re-call Ollama per threshold.

## What the corpus is

Semi-synthetic. Anchors and unrelated queries are real questions from NQ-Open, a dataset derived from Google Search queries. Paraphrases and hard negatives are generated from the anchors. The 90-row stratified diagnostic predates the final 37 paraphrase recoveries; do not treat it as a representative estimate of the final corpus.
