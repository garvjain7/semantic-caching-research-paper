# Claim sheet for the research paper draft

Every cited claim, its reference number, and what was read. 'read' = the abstract or page was opened in this work. 'check' = bibliographic details came from a search listing or from memory and must be confirmed before submission.

| Ref | Key claim in the paper | What was read | Status |
|---|---|---|---|
| [1] | RAG definition | Title/venue confirmed by search; pages from memory | check pages |
| [2] | 2–10× faster on a hit; response time not affected by network fluctuation; speed/cost focus | Abstract (ACL Anthology page) | read |
| [3] | Demonstration paper; context-aware multi-turn caching; ~10× lower latency; PVLDB 18(12) 5391–5394 | arXiv abstract this session; PVLDB details from earlier session search | read |
| [4] | Static thresholds give no formal guarantee; first verified semantic cache with user-defined error-rate guarantees; meets bounds, beats baselines; ICLR 2026 | arXiv abstract; ICLR venue from the ICLR 2026 poster listing (arXiv page lists no venue) | read |
| [5] | Federated similarity model, context chains; ~17% F-score, ~20% precision, 83% storage, 11% faster decisions vs GPTCache; IPDPS 2025 | Abstract (arXiv PDF); page states acceptance at IPDPS 2025 | read |
| [6] | Relative to GPTCache: avg +63% hit ratio, +77% token savings; analysis of real chat data | Abstract (arXiv) | read |
| [7] | Async LLM judge on grey-zone; promotion to dynamic cache; 8.2→19.4% and 2.2→8.6%; oracle judge from benchmark equivalence classes; real-judge check 99/100; benchmarks of Chatbot Arena and search queries | arXiv PDF | read |
| [8] | Sentence encoder used for MiniLM model | Title/venue confirmed by search listing; pages from memory | check pages |
| [9] | Contrastive sentence-embedding training | Not searched; from memory | check |
| [10] | Dense retrieval matches questions and passages by vector similarity | Not searched; from memory | check |
| [11] | Exact and approximate vector search library | Title confirmed by search; IEEE Trans. Big Data vol/pages from memory | check vol/pages |
| [12] | Graph-based approximate nearest-neighbour index | DOI listing confirmed; vol/pages from memory | check vol/pages |
| [13] | Up to 68.8% fewer API calls; hit rates 61.6–68.8%; positive hit rates >97% | Abstract (arXiv) | read |
| [14] | Fine-tuned compact encoder; medical precision 78→92%; ~35K synthetic pairs → 87% vs 85% for text-embedding-3-large | Abstract (arXiv) | read |
| [15] | Semantic forest, attention-based context-change detection, cross-session KV sharing; −59.1% KV memory; TTFT −78.0%/−71.7%; +39.9% F1 (Qwen-2.5-1.5B, CoQA) | NeurIPS 2025 poster page | read |
| [16] | Three strategies; cascade; up to 98% cost reduction; +4% accuracy at equal cost | arXiv abstract | read |
| [17] | KV states of retrieved documents; TTFT up to 4×, throughput up to 2.1× vs vLLM+Faiss | arXiv abstract | read |
| [18] | Attention-state reuse; TTFT ~8× GPU, ~60× CPU | arXiv abstract | read |
| [19] | 108,463 pairs; high lexical overlap; models trained on earlier data perform poorly | ACL Anthology page | read |
| [20] | Authors perturb test instances so the label usually changes; 10 datasets; performance up to 25% lower | arXiv PDF | read |
| [21] | Matrix of capabilities and test types; failures found in commercial and SOTA models | arXiv PDF | read |
| [22] | LLMs: insensitivity to negation, can't capture lexical semantics of negation, fail to reason under negation | arXiv PDF | read |
| [23] | Strong judges >80% agreement, same as human-human; position, verbosity, self-enhancement biases | NeurIPS abstract page | read |
| [24] | Spearman 0.514 with GPT-4 on summarization; possible bias toward LLM text | ACL Anthology page | read |
| [25] | Self-preference linked to self-recognition | arXiv abstract | read |
| [26] | Wilson interval | Not searched; standard citation from memory | check |
| [27] | Wilson interval behaves well for small n or p near 0/1 | Title/journal confirmed by search; claim is the paper's well-known conclusion | check |
| [28] | NQ-Open introduced for open-domain QA; ACL 2019 pp. 6086–6096 | ACL Anthology page (bibliographic only) | read |
| [29] | Natural Questions: real anonymized queries to Google Search; TACL 7:453–466 | Search listing; volume/pages from memory | check pages |
| [30] | Prefix filter keeps only leading elements under a fixed ordering | Paper PDF first page; venue ICDE 2006 and year NOT printed on page | check venue |
| [31] | Duplicates are common in datasets and affect results; ACL 2022 pp. 8424–8445 | ACL Anthology page | read |
| [32] | MiniLM architecture | Title/venue confirmed by search listing; pages from memory | check pages |
| [33] | BGE authors report SOTA on MTEB for English models (as stated in the abstract) | arXiv abstract | read |
| [34] | MTEB benchmark | Title/venue confirmed by search listing; pages from memory | check pages |
| [35] | BERT re-ranking of retrieved candidates | Not searched; from memory | check |
| [36] | 3.8B parameters | arXiv PDF | read |
| [37] | Family sizes 1B/4B/12B/27B | arXiv PDF | read |
| [38] | Llama 3 family; fallback model is Llama 3.2 3B, which this report does not describe | Search listing only; author line given as the Llama Team | check authors; family only |

## Statements about our own pipeline (check against the final configuration before submission)

- Corpus sizes: 100 anchors, 50 development anchors, held-out pool of 1,000, 100 unrelated queries, at most 50 hard negatives per type, 90-item audit (9 strata × 10), sampled before 37 paraphrases were recovered.
- Unrelated queries: token-level Jaccard at most 0.2 with every anchor. Seed 42. Threshold grid 0.50–0.95 in steps of 0.05.
- Models: gemma3:4b (generator), llama3.2:3b (fallback and stored responses, temperature 0), phi4-mini (judge, temperature 0), all-MiniLM-L6-v2 and bge-base-en-v1.5, `cross-encoder/quora-distilroberta-base`, FAISS flat inner product, difflib ratio.
- Hard-negative checks (invalid output, equals anchor or a paraphrase after normalization, length ratio beyond 2, up to 3 regenerations, not-applicable allowed) come from the corpus specification. The 90% audit criterion is reported diagnostically; P3, P4, E, T, and S are below it in the saved sample and remain in the evaluated corpus.
- Mechanical filter rules and the tokenizer match the repository code that was read.
- The cross-encoder's training on duplicate-question pairs and its 0–1 score come from the model card, which was not opened in this work.
- The cache is static (no insertion on a miss). This is a design choice and the replay argument depends on it.
- Judge: Phi-4-mini, verdict-only output, judged once per (configuration, query) pair. The separate 90-row audit is a limited diagnostic.
- Statements of design reasoning (why a miss costs one call, why a false hit is worse than a miss, why replay is exact) are our own arguments, not cited claims.

## Reported experiment-specific findings
- The 90-item stratified audit is a limited diagnostic from the 663-query pre-recovery corpus, not a population estimate. Main sweep findings use the final 700-query corpus.
- Main sweep result sections have been populated from saved metrics. Audit threshold replay remains supplementary and descriptive.
- No 'first' or 'no one has done' statement about our work; gaps are limited to 'in the material we reviewed'.
