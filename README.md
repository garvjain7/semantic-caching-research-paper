# Semantic Caching for RAG/LLM Pipelines — Correctness vs. Cost Tradeoffs

Undergraduate research seminar project, Department of Computer Engineering, Poornima Institute of Engineering and Technology.

## Overview

LLM and RAG systems repeatedly answer semantically identical or near-identical queries, and semantic caching is the standard fix — but existing systems don't systematically characterize *when a cache hit is actually wrong*. This project builds a semantic caching layer and studies the tradeoff between cost savings (LLM calls avoided) and correctness (false-hit rate) across similarity threshold, embedding model choice, and cache scale, using a controlled synthetic paraphrase / hard-negative workload with engineered ground truth.

**Research area:** Distributed Systems and AI Infrastructure

**Status:** Pre-implementation — topic, system design, dataset methodology, and 4-week execution plan finalized. Build not yet started.

## Why this gap exists

Prior work on semantic caching (GPTCache, MeanCache, RAGCache, ContextCache, and related systems) reports hit rate, cost savings, or threshold-tuned precision/F1 — but each evaluates on naturally-occurring query traffic, not a corpus with controlled, graded semantic distance and engineered same-answer/different-answer ground truth. GPTCache's own paper names the threshold-vs-false-positive tradeoff as open future work rather than something it systematically measured. This project targets that gap directly, with a joint cost-savings-vs-false-hit-rate curve as the primary result rather than a single-metric report.

Full reasoning, verified citations, and the corrected gap analysis are in [`PROJECT_STATE.md`](./PROJECT_STATE.md).

## Repository Structure

```
.
├── README.md          # this file
├── PROJECT_STATE.md    # single source of truth — living engineering notebook
```

As implementation begins, this structure will expand to include the caching pipeline, dataset generation scripts, experiment logs, analysis notebooks, and the paper draft. This section will be kept up to date as those are added.

## PROJECT_STATE.md

[`PROJECT_STATE.md`](./PROJECT_STATE.md) is the authoritative, continuously updated record of this project — architecture decisions (with alternatives considered and reasoning), current status, task priority, open risks, verified facts vs. unverified assumptions, and a running decisions log. Start there for full context, especially before making any implementation choice not already decided in that document.

## Timeline

4-week build/experiment/analysis window, followed by writing. See `PROJECT_STATE.md` → Next Tasks for the current week-by-week plan and scope-cut rules if the timeline slips.

## Guide

Topic finalized and submitted for guide review. Priority to be confirmed by guide.