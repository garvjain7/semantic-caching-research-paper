# Figure Artifacts

The manuscript's method diagrams are Figures 1–4. Figure 1 is the landscape four-stage pipeline in `../generated/figures/figure_1_end_to_end_v3.svg` (source) and `../generated/figures/figure_1_end_to_end_v3.png` (manuscript image); Figures 2–4 are Mermaid diagrams in the root `research_paper.md`. The publication results are Figures 5–10 in `../generated/figures/`: threshold JFH with Wilson intervals, model-call savings, the trade-off frontier, hard-negative subtype rates, lexical covariate bins, and judge–label disagreement. Earlier numbered PNGs are exploratory or superseded; the active Figure 1 reference uses the unique `figure_1_end_to_end_v3.png` filename to avoid stale-preview caching.

Regenerate from the repository root with:

```powershell
python -m analysis.paper_outputs
```

The publication figures are reproducible from the stored results and are not hand-edited source figures. See `../generated/README.md` for the accompanying CSV tables and definitions.

\n