# Figure Artifacts

The active generated figures are `../generated/figures/figure_1.png` through `figure_3.png`. They are produced by `analysis/paper_outputs.py` from saved threshold metrics; this `figures/` directory is not the canonical output location.

Regenerate from the repository root with:

```powershell
python -m analysis.paper_outputs
```

The generated files are reproducible from the stored results and are not hand-edited source figures.
