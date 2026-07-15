# PROJECT_STATE.md
_Last updated: 2026-07-15_

---

# Project Vision

**What we are building:**
A research system studying semantic caching for RAG/LLM pipelines — specifically, a caching layer that measures the tradeoff between cost savings (LLM calls avoided) and correctness (false-hit rate) as a function of similarity threshold, embedding model, and cache scale, using a controlled synthetic paraphrase/hard-negative workload.

**Why we are building it:**
Technical seminar (guide-supervised) at Poornima Institute of Engineering and Technology, Dept. of Computer Engineering. Deliverable: a research paper with an empirical, reproducible result — not a survey, not a pure engineering build.

**Current phase:**
Pre-implementation. Topic finalized, literature reviewed, system design and experimental plan finalized. Build not yet started.

---

# High-Level Architecture

**Current architecture (planned, not yet built):**
- Embedding layer: sentence-transformer model, starting with `all-MiniLM-L6-v2` (matches GPTCache's default embedding function, giving a fair baseline comparison point).
- Vector index: FAISS (or plain cosine similarity over numpy) for primary correctness experiments — deliberately NOT Redis in the critical path, to avoid network/infra variance as a confound.
- Cache logic: embed incoming query → retrieve nearest cached entry + similarity score → threshold τ decision → hit (serve cached answer) or miss (query LLM fallback, cache new entry).
- Cached answers = dataset gold answers directly (NOT LLM-generated), to eliminate LLM-generation-quality as a confound on correctness measurement.
- LLM fallback: Ollama, fixed local model (e.g. Llama 3.1 8B or Qwen2.5 7B), temperature 0, used only to measure real cost/latency avoided on a hit — not used to determine correctness.
- Instrumentation: full per-request logging (query, embedding, retrieved entry, similarity score, threshold, hit/miss label, ground-truth label, latency, tokens) — logged once at fine granularity, metrics recomputed from log rather than rerunning experiments.
- Optional secondary phase: Redis-backed version for a realistic deployed-latency measurement, only if time remains after core experiments.

**Major design decisions:** see Decisions Log.

**Technologies:** Python, sentence-transformers, FAISS, Ollama (local LLM), (optional later: Redis/RedisVL).

---

# Current State

- Research topic finalized: **Semantic Caching for RAG/LLM Pipelines — Correctness vs. Cost Tradeoffs**.
- Literature reviewed and gap verified against actual paper methodologies (not assumed).
- System architecture and experimental design finalized on paper; no code written yet.
- Timeline: 4 weeks (build + experiment + analysis), then writing phase. Tighter than the original 5–6 week estimate — flagged as a real constraint.
- Awaiting: Topic Finalization Form submission to guide (Research Area: Distributed Systems and AI Infrastructure); Priority ranking to be filled by guide.

---

# Completed

- Topic selection process across 4 candidate topics (semantic caching, filtered ANN search, distributed rate limiting, OT/CRDT consistency).
- Literature scan across all 4 candidates.
- Final topic selection: Semantic Caching (Topic 1), with Filtered ANN Search as 2nd choice, Distributed Rate Limiting as 3rd, OT/CRDT deprioritized (literature saturated/dated).
- Corrected/verified gap analysis after checking actual GPTCache and MeanCache evaluation methodologies (see Assumptions → Verified Facts).
- Full system architecture and dataset construction methodology designed.
- 4-week execution plan with priority-ordered experiment scope drafted.
- Guide Q&A prep completed for all three shortlisted topics (semantic caching, filtered ANN, rate limiting).

---

# In Progress

- Nothing in active build yet. Next actionable step is Week 1 of the execution plan (infra build + full paper reading).

---

# Next Tasks (priority order)

1. Submit Topic Finalization Form to guide; get priority/approval.
2. Week 1: Build embedding + FAISS cache-check pipeline + logging. Set up Ollama fallback. Read GPTCache and MeanCache papers in full for accurate related-work citations.
3. Week 2: Select base QA dataset (verify licensing/format — likely TriviaQA-style, short factual answers). Build paraphrase generator (LLM-based) and hard-negative generator (template/entity-swap based, for guaranteed ground-truth answer change). Generate dataset, measure actual embedding similarity distribution empirically, manually verify a sample (50–100 pairs/group).
4. Week 3: Run primary threshold sweep (τ 0.70–0.99). Run secondary B (embedding model comparison: MiniLM vs. a stronger model e.g. BGE-base). Run secondary C (cache scale effect: 200 vs 2,000 vs 10,000 filler entries) only if on schedule.
5. Week 4: Compute confidence intervals (bootstrap/binomial) on hit/false-hit rates. Produce final plots (headline: false-hit-rate vs. cost-savings curve). Draft methodology + results first, then related work, then limitations (flag synthetic-corpus limitation explicitly).

**Scope-cut rule if Week 2 slips:** drop experiment C first. A (threshold sweep) + B (embedding model) is the non-negotiable minimum publishable core.

---

# Blockers

- None currently — pre-implementation stage.

---

# Decisions Log

**Decision:** Final research topic is Semantic Caching for RAG/LLM Pipelines.
**Alternatives considered:** Filtered ANN Search Under Multi-Tenant Constraints; Distributed Rate Limiting Under Network Partitions; OT vs. CRDT consistency for real-time collaborative systems.
**Why chosen:** User's explicit selection after reviewing guide-question prep for all three shortlisted topics.
**Date:** 2026-07-15

**Decision:** OT vs. CRDT topic deprioritized to last priority.
**Alternatives considered:** Keeping as equal 4th option.
**Why chosen:** Literature is largely a saturated, decade-old debate (2019-era "real differences" framing); genuinely new material (2025 hybrid CRDT-OT paper, 2025 survey) is thin compared to other three topics; high risk of appearing as a rehash.
**Date:** 2026-07-15

**Decision:** Research topics must NOT be derived by extending/evaluating the user's existing projects (ComplySense, CodeAlive, Viewora/ScanVista).
**Alternatives considered:** Anchoring topics to existing project architecture/schema for feasibility (original approach, flagged as wrong by user).
**Why chosen:** User clarified that listed technology areas represent skill background, not literal research scope. Projects may serve as evidence of capability to execute, never as the source/justification of the topic itself.
**Date:** 2026-07-15 (correction made mid-conversation)

**Decision:** Cached answers in the experimental system will be the dataset's gold answers directly, not LLM-generated answers.
**Alternatives considered:** Using an LLM to generate the answer to be cached.
**Why chosen:** Eliminates LLM-generation-quality as a second confound stacked on top of retrieval/caching correctness; keeps the false-hit measurement attributable purely to the caching mechanism.
**Date:** 2026-07-15

**Decision:** Vector index for primary experiments will be FAISS/in-process, not Redis.
**Alternatives considered:** Redis (RedisVL) as the vector store from the start, matching production-realistic deployment.
**Why chosen:** Redis adds network/infra latency variance as an unwanted confound in the correctness-focused experiments. Redis-backed version deferred to an optional secondary "realistic deployed latency" measurement if time allows.
**Date:** 2026-07-15

**Decision:** Hard-negative pairs (same topic, different gold answer) will be constructed via template/entity-swap methods, not free-form LLM generation.
**Alternatives considered:** LLM-generated hard negatives (same method as paraphrase group).
**Why chosen:** Entity-swap gives a guaranteed, verifiable ground-truth answer change; free-form LLM generation risks accidentally preserving the answer, undermining the ground-truth labeling the whole methodology depends on.
**Date:** 2026-07-15

**Decision:** Gap framing corrected mid-conversation — from "no existing paper sweeps threshold vs. false-hit rate" to the narrower, verified claim: MeanCache does sweep threshold and report precision/F1 (optimal τ=0.83 MPNet, F1 0.89; τ=0.78 Albert, F1 0.88), but on naturally-occurring query duplicates, not a controlled graded-distance corpus with engineered hard negatives, and without a joint cost-savings-vs-false-hit-rate curve as the primary output.
**Alternatives considered:** Leaving the original, broader (and less accurate) gap claim uncorrected.
**Why chosen:** Hard rule requires verified facts over assumed ones; correction made immediately upon verifying MeanCache's actual methodology via primary sources.
**Date:** 2026-07-15

---

# Important Discussions

- User rejected an early framing where all four candidate research topics were anchored to the user's own existing projects (ComplySense, CodeAlive) as evidence of feasibility. Corrected: topics must be real, independently-existing research problems in the stated technology domains; projects are supporting evidence only, never the topic's origin or justification.
- Scope-clarification: research area on the finalization form should read "Distributed Systems and AI Infrastructure."
- Guide-facing Q&A prepared for all three shortlisted topics — includes anticipated "this already exists, what's new" pushback and rehearsed, evidence-grounded answers for each.
- User set a standing hard rule (stored in memory, applies to this project): no sugarcoating, no fabricated/assumed citations or methodology claims, verify before stating, flag uncertainty explicitly, no comparisons to other students/researchers, don't restate questions unnecessarily, respond in the register of a systems researcher — accuracy and critical analysis over encouragement.
- User established the PROJECT_STATE.md convention as the project's single source of truth, functioning as an engineering notebook (not a TODO list), to preserve context across long conversations and context-window limits.

---

# Assumptions

**Verified facts (checked against primary sources this session):**
- GPTCache's default `similarity_threshold` config is 0.8 (range 0–1), and its own paper (NLP-OSS 2023 / ACL Anthology) explicitly names threshold-vs-false-positive-rate characterization as unresolved future work, not something it systematically measured.
- MeanCache (arXiv:2403.02694) sweeps threshold τ and reports precision/recall/F1/accuracy; optimal τ=0.83 with MPNet embeddings (F1 0.89, precision 0.92, accuracy 0.90); optimal τ=0.78 with Albert (F1 0.88, precision 0.86, accuracy 0.88).
- MeanCache's evaluation corpus is real-world: 27,000 queries from 20 ChatGPT users (professors, developers, grad students), not a controlled synthetic paraphrase/hard-negative corpus.
- The commonly cited "~31% of LLM queries are redundant" statistic traces to the MeanCache paper's motivation section.

**Current assumptions (NOT yet verified — must be checked before implementation):**
- Exact choice of base QA dataset (TriviaQA vs. alternative) — licensing and short-answer format need direct verification before committing.
- Assumed feasibility of running Llama 3.1 8B / Qwen2.5 7B via Ollama locally within the user's available hardware — not yet confirmed.
- Assumed sample sizes (500+ pairs/group) will be sufficient for statistically meaningful confidence intervals — not yet validated against actual variance once real data is collected.
- Second embedding model for comparison (tentatively BGE-base) not yet finalized.

---

# Future Ideas

- Domain-dependence experiment (open-domain vs. tighter-answer-boundary questions, e.g. numerical/date-sensitive queries) — explicitly out of scope for the 4-week window; noted as future work in the eventual paper rather than attempted now.
- Multi-turn/conversational context-aware caching extension (as in ContextCache) — not in current scope.
- Redis-backed deployed-latency measurement as a secondary system-realism result, only if time remains after core experiments.

---

# Questions Needing Research

- Final base QA dataset selection and license verification.
- Confirming local compute is sufficient for the chosen Ollama model at required experiment volume within the timeline.
- Statistical method for confidence intervals on hit/false-hit rates given final sample sizes (bootstrap vs. binomial CI) — decide once real data volume is known.

---

# Risks

- **Dataset construction (Week 2) is the single largest timeline risk.** Generating and validating a controlled graded-distance corpus with reliable ground-truth labels is the hardest, least predictable part of the plan.
- 4-week timeline is tighter than the original 5–6 week estimate for the full research cycle; writing phase is not yet included in the 4 weeks.
- Risk that LLM-based paraphrase generation does not reliably produce the intended similarity bucket — mitigated by empirical post-hoc measurement and iteration, but this could consume more of Week 2 than planned.
- Risk that sample sizes per bucket are too small to produce defensible confidence intervals — won't be known until real data is generated.

---

# Session Summary

**Session: 2026-07-15 — Topic finalization and research design**

**What changed:**
- Research topic finalized (Semantic Caching for RAG/LLM Pipelines).
- Full system architecture, dataset methodology, experiment priority order, and 4-week execution plan designed.
- Gap framing corrected after verifying MeanCache/GPTCache methodology directly against primary sources.
- PROJECT_STATE.md convention established as the project's ongoing source of truth.

**What was learned:**
- MeanCache already performs a threshold-vs-F1 sweep on real-world data — the actual gap is narrower and more specific (controlled graded-distance corpus + engineered hard negatives + joint cost-savings/false-hit-rate curve) than initially framed.
- GPTCache's own paper names the threshold/false-positive tradeoff as open future work, which is a citable, defensible motivation hook.

**What remains:**
- No code has been written yet. Dataset has not been built. Base QA dataset not yet finalized.

**Next recommended task:**
Begin Week 1: build the embedding + FAISS cache-check + logging pipeline, and complete full (not skimmed) reads of the GPTCache and MeanCache papers for the related-work section.