# Corpus Implementation Plan and Handoff

This document is a context handoff for the corpus work. It records the requested direction, decisions already made, current repository evidence, proposed implementation sequence, and unresolved questions. A future contributor or AI assistant should use it to continue the work without silently inventing methodology.

## Status

- Configuration implementation is complete: `semantic-cache-research/configs/corpus.yaml` records the corpus settings, and `semantic-cache-research/src/corpus_config.py` loads and validates the declared schema and invariants. The source is pinned to `google-research-datasets/nq_open` revision `5dd9790a83002ad084ddeb7c420dc716852c6f28`; the seed and model IDs remain unresolved.
- The user confirmed `nq_open`; `corpus_spec.md` now describes the `question` plus answer-list schema, preserves multiple answers, and distinguishes unverified generated negative answer claims from verified `gold_answers`. The config validator checks the source contract and requires an immutable 40-character revision.
- A source adapter in `semantic-cache-research/src/dataset_generation/nq_open_source.py` normalizes source records without collapsing answer lists and loads requested splits only at a caller-supplied commit SHA. It has fixture-based tests; no dataset download has been performed.
- The user specified tokenization and mechanical candidate checks. `text_processing.py` implements the shared lowercase Unicode tokenizer; `candidate_filter.py` applies question/answer eligibility rules and produces unique totals plus overlapping rejection counts per rule.
- The user selected the first source occurrence in train-then-validation order as the deduplication survivor. `lexical_dedup.py` implements that greedy rule with token-set Jaccard and a prefix index; exhaustive small-set tests compare it against a brute-force implementation.
- The same module exposes configuration-only readiness checks for sampling, cache responses, paraphrase generation, hard-negative generation, and judge evaluation. They report missing settings but do not claim source files, model installations, or services are available.
- Twenty-three focused `unittest` cases cover config validation, stage readiness, the NQ-Open adapter, mechanical candidate filtering, and lexical deduplication, including comparison against brute force across small token sets.
- No data loading, corpus generation, audit, or experiment-runner behavior has been implemented yet.
- `corpus_spec.md` exists at the repository root and is untracked in the observed worktree. Do not assume it has been committed.
- `research_paper.md` was already modified in the observed worktree. Do not overwrite or revert that change as part of this task.
- The project is not a blank repository. It has a partial code skeleton under `semantic-cache-research/`, including dataset-generation stubs and an experiment-runner stub.
- The selected `nq_open` dataset is not present in the observed `semantic-cache-research/data/raw/` directory; that directory contains only a README.
- Do not start expensive generation or claim the corpus is ready until the unresolved items below are settled and required data/models are available.

## User decisions already received

Treat the following as settled unless the user explicitly changes them:

1. **Use an existing natural-language QA dataset and augment it.** The user explicitly selected the exact Hugging Face dataset `google-research-datasets/nq_open` at `https://huggingface.co/datasets/google-research-datasets/nq_open`. Do not substitute the full NQ release or another dataset without approval.
2. **Use Ollama-generated responses for cache entries**, as specified in `corpus_spec.md`. This supersedes the older design that used dataset gold answers as the cached response.
3. **Generate E/N/T/S hard negatives with an LLM**, as specified in `corpus_spec.md`. This supersedes the earlier template/entity-swap-only decision.
4. **Store correct short answers wherever available**, including hard-negative and unrelated rows. Do not mislabel an unverified generated answer as a verified gold answer.
5. **Do not guess model IDs.** Defer generator G, fallback response model, judge J, and rater selection. Require explicit model IDs in config before the relevant generation/evaluation steps run.
6. **Keep the proposed corpus counts and audit design**, but make counts configurable and report achieved counts if targets are not reached.
7. **Adapt to the NQ-Open answer-list schema.** Preserve each record's `question` string and `answer` list; do not collapse a list to one answer. The dataset card reports CC BY-SA 3.0. Exact dataset revision remains to be pinned.

These choices are evidence of the user's intended direction, not proof that every detail in the specification is fully defined.

## Source documents and repository context

Read these before making implementation changes:

- `corpus_spec.md` — corpus methodology and NQ-Open-specific question/answer-list schema.
- `PROJECT_STATE.md` — older project decisions and broad architecture. Its last-update/status statements may be stale relative to the present repository.
- `TECHNICAL_OVERVIEW.md` — older overview; conflicts with the new corpus spec on cache answers and hard-negative construction.
- `semantic-cache-research/src/dataset_generation/paraphrase_gen.py`
- `semantic-cache-research/src/dataset_generation/hard_negative_gen.py`
- `semantic-cache-research/src/dataset_generation/unrelated_sampler.py`
- `semantic-cache-research/src/experiment_runner.py`
- `semantic-cache-research/src/llm_fallback.py`
- `semantic-cache-research/data/raw/README.md`
- `semantic-cache-research/data/processed/README.md`
- `semantic-cache-research/data/processed/verification_report.md`
- `semantic-cache-research/requirements.txt`

The three generator modules and the experiment runner contain `NotImplementedError` stubs. Their docstrings still reflect older assumptions in places; reconcile them when implementation begins. Do not rely on an old docstring over the user's newer choices.

## Proposed corpus pipeline

The sequence below follows `corpus_spec.md`; implementation details that are not defined there remain open rather than being silently chosen.

### 0. Configuration, provenance, and source data

- Add `semantic-cache-research/configs/corpus.yaml`.
- Put counts, fixed seed, filter limits, retry limits, audit sample size/threshold, and required model identifiers there.
- Require explicit model IDs and versions for every enabled model-dependent step; fail clearly if a required value is absent. Do not choose convenient defaults for G, fallback, or J.
- Record prompt text or a stable prompt artifact, prompt hashes, model names/versions, seed, source dataset/version, and relevant generation settings.
- Pin an exact NQ-Open dataset revision before sampling. Its dataset card reports CC BY-SA 3.0; verify attribution/share-alike requirements and institutional policy before redistributing derived data.
- Keep downloaded source data unchanged in `data/raw/`; write derived artifacts under `data/processed/`.
- Checkpoint costly work so an interrupted run can resume without duplicating completed generations. Checkpoint semantics and idempotency must be tested.

### 1. Extract and filter

- Extract only the NQ-Open `question` string and `answer` list; do not expect full-release NQ annotation fields or carry unrelated fields into derived data.
- Retain candidates with a non-null short answer, 5–20 tokens, and acceptable question form.
- Drop near-duplicates at token-Jaccard >= 0.8 using lexical comparison only. No embedding model may participate in corpus filtering or deduplication.
- Record input, retained, and rejected counts, including reasons for rejection.
- The specification does not define tokenization or the exact well-formedness test; resolve these before coding the filter (see open decisions).

### 2. Disjoint sampling

Using the configured fixed seed, sample:

- 600 anchors,
- 50 disjoint dev anchors, used only for prompt tuning and never evaluated,
- a held-out pool of at least 1,000 questions for unrelated queries.

Prove and test disjointness by stable source-item identity, not just by comparing text. Report achieved counts if the filtered data cannot support the requested sizes. Do not reuse dev items in the evaluated pool.

### 3. Anchor cache responses

- Generate one fallback response per anchor using the explicitly configured Ollama model at temperature 0.
- Store response text, latency, token counts, model/version, and provenance.
- In keeping with the user's decision, this generated response is the answer served by the cache. Keep it distinct from the NQ-Open answer list, which is source reference data, not a replacement cache response.
- Decide and document how missing usage/token metadata is represented. Do not silently substitute fabricated values.

### 4. Paraphrases

- Generate four items per anchor at intended levels P1–P4, using 5–8 real NQ-Open questions from the dev split as style examples. Pass the source answer list as reference context; paraphrase rows inherit the anchor's list without model rewriting.
- Keep information need and the gold short answer unchanged. Match the anchor's register, casing, and punctuation except for the intended P3 register change.
- Require JSON-only output and validate its exact shape.
- Apply the spec's automatic checks: distinct from anchor, preserve numbers and years in the anchor, and length ratio in [0.5, 2]. Retry up to the configured/spec limit of three; then drop and report the item.
- Store intended level as a label, not as a claim that measured semantic difficulty matches that level.

### 5. Hard negatives

- Generate E, N, T, and S hard negatives with the LLM, using the same dev style examples and style matching as for paraphrases.
- Each item should change only the targeted element, preserve the sentence frame as appropriate, read naturally, and have a different correct answer from its anchor.
- Accept `NOT_APPLICABLE`; iterate anchors in seeded random order until each type's quota is met or the pool is exhausted. Report per-type achieved counts and failures.
- Store the generator's answer-difference statement for audit only. Never expose it to the cache or use it as if it were verified gold.
- A generated proposed answer, if stored, must be explicitly distinguished from `gold_answers`. Populate `gold_answers` for hard negatives only after independent verification; the exact verification policy remains unresolved below.

### 6. Unrelated queries

- Sample from the held-out NQ-Open pool and preserve each row's source answer list.
- Keep a question only if its maximum token-Jaccard against every anchor is <= 0.2.
- Target 600, report the achieved count, and do not turn an unrelated query into an anchor-associated item.
- For each unrelated query, store its available NQ-Open answer list. Its anchor relationship and edit-distance reference need a defined representation before schema implementation.

### 7. Covariates, integrity, and final corpus

- For each evaluated query, record token-Jaccard and normalized edit distance against the appropriate anchor, according to a documented rule.
- Validate schema, uniqueness, split disjointness, foreign-key/anchor references, per-type quotas, answer provenance, and expected audit strata before freezing.
- Write a reproducible manifest with input/source identity, configuration, achieved counts, and hashes.
- Hash the final corpus and write `data/processed/verification_report.md`.
- Never report target counts as achieved counts unless verified from output records.

### 8. Human audit and judge comparison

- Create a blinded audit sample of 30 items from each of P1, P2, P3, P4, E, N, T, S, and unrelated (270 target items total).
- The rater should not see the intended subtype.
- Capture the specified labels: same information need; for negatives, whether the correct answer differs; and whether the anchor's cached response is acceptable for the query.
- Report per-stratum accuracy against a precisely defined intended-label rubric, with Wilson 95% intervals.
- Regenerate strata below 90% with a tightened prompt and re-audit, preserving the original and regenerated records and reporting the procedure.
- Compare judge and human acceptability labels on the same sample using raw agreement and Cohen's kappa.
- If only one human rater is available, state the single-rater limitation. If a second is available, record the double-labeled subset and inter-rater agreement.
- Rater identity/availability and the exact blinding workflow are not settled; do not fabricate human audit results or present an uncompleted audit as completed.

## Evaluation requirements that affect corpus integration

The corpus spec also constrains the later experiment runner:

- The nearest neighbor of a query is independent of threshold. For each `(query, nearest neighbor)` and each `(embedding, match strategy)`, run the judge once and reuse its verdict over thresholds.
- Call the fallback LLM once per test query, store the response/latency/tokens, and replay it for threshold misses. Do not call Ollama again per threshold.
- Keep corpus construction out of the threshold-sweep loop.
- The current `experiment_runner.py` is only a stub and has older broad-label assumptions. Update its input handling to preserve P1–P4, E/N/T/S, and unrelated subtypes rather than collapsing away distinctions needed for analysis.

## Unresolved decisions — do not guess

Resolve these with the user or a documented methodological decision before implementing the affected behavior.

1. **Hard-negative answer verification:** The user selected “store correct short answers wherever available” but did not provide a value when asked how to verify hard-negative answers. A generator's answer-difference statement is not evidence that the answer actually differs. Choose one:
   - store a generator-proposed answer in a separate, explicitly unverified field and verify through a defined independent process;
   - independently verify every negative answer before calling it gold (potentially reducing achieved counts); or
   - do not assign a gold answer to unverified negatives, and state the resulting limitation.
   The specified 30-per-stratum audit alone does not verify all generated negatives.
2. **Unrelated query anchor fields:** Define `anchor_id` and edit-distance semantics for unrelated rows. They have no single correct anchor; the spec requires maximum Jaccard against every anchor but does not say which anchor to use for `edit_dist_to_anchor`.
3. **Tokenization:** Define the tokenizer used for token counts, token-Jaccard, length ratio, and number/year preservation. It must be deterministic and documented.
4. **Normalized edit distance:** Define the edit unit (characters or tokens), normalization denominator, and behavior for empty inputs.
5. **“Well-formed” filtering:** Define objective acceptance/rejection criteria or a documented review process.
6. **Audit accuracy rubric:** Define how the rater's answers map to “accuracy against the intended label” for every stratum. The listed questions alone do not fully specify this mapping.
7. **Judge behavior:** Define judge input, output schema, acceptance criterion, and whether the judge sees the NQ gold answer, generated negative answer claim, or only query and cached response. Do not expose audit-only statements to the cache; decide explicitly whether they are available to the judge.
8. **Source revision and license:** Pin the exact NQ-Open repository revision. Its dataset card reports CC BY-SA 3.0; verify applicable redistribution requirements and whether processed/generated records can be published.
9. **Model and rater IDs:** The user explicitly deferred model selection. Require explicit config values at runtime; do not select models or claim a second rater is available.
10. **NQ-Open answer-list edge cases:** Specify how empty lists, duplicate answer strings, and other malformed source records are handled. Preserve multiple answers rather than inventing a canonical answer.

## Count and effort reality

The stated targets imply up to 2,400 paraphrases, up to 1,200 hard negatives, 600 unrelated items, 600 anchor responses, plus auditing 270 items. There can also be fallback responses for evaluated queries during the experiment stage. LLM generation, human review, and verification are real work; using an existing dataset avoids building the base corpus, but does not make augmentation and audit negligible.

Keep counts configurable as the user requested. Do not silently lower targets, silently skip audits, or claim an underfilled stratum passed its threshold. If time or model capacity prevents reaching targets, preserve the configured targets and report actual achieved counts and the resulting methodological limitation.

## Implementation order and acceptance checks

1. Resolve the open decisions that affect schema and validity, especially hard-negative answer verification, unrelated-row references, tokenization, and the audit rubric.
2. Reconcile older conflicting documentation with `corpus_spec.md` and the user decisions; preserve unrelated edits in the dirty worktree.
3. Add and validate `configs/corpus.yaml`; fail early on missing required model IDs or source data.
4. NQ-Open loading, candidate eligibility checks, and lexical deduplication are implemented. Next, implement and test deterministic disjoint splits as a separate slice; the random seed remains unset and must be selected before sampling runs.
5. Implement generation interfaces with structured-output validation, bounded retries, provenance, checkpoint/resume, and tests using deterministic fake model responses.
6. Implement row assembly, covariates, integrity checks, audit export/import, statistics, report generation, and corpus hash.
7. Integrate the corpus schema into the experiment runner and implement per-pair judge reuse and per-query fallback replay as a separate evaluation concern.
8. Validate determinism, split disjointness, quota/accounting behavior, schema, provenance, retry exhaustion, resume behavior, and that no generation/audit claim exceeds the evidence actually produced.

## Handoff instructions for the next AI/contributor

- Start by reading this file and the source documents listed above.
- Do not make up answers to the unresolved decisions. Ask the user if a decision is needed to proceed.
- Do not treat generated text or model confidence as verified ground truth.
- Do not replace NQ-Open or change the user's selected counts/methods without explicit approval.
- Do not overwrite existing user changes, especially the already-modified `research_paper.md`.
- Implement in small, testable stages; report what is implemented, what is blocked, actual achieved counts, and which verification evidence exists.
- If a required decision is unresolved, continue only on independent work that cannot bake in that decision; otherwise stop and ask.
