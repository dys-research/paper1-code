# Figures from generated results

The supplied notebook contained stored images, personal paths and several alternative layouts of the same forecast figure. This directory retains the spline figure and the eight-panel forecast layout as ordinary scripts. Generated figures are excluded from source control.

```bash
python -m scripts.figures.splines
python -m scripts.figures.forecasts
```

`splines` requires `python -m scripts.robustness.inference` and reads its complete-observation curves. `forecasts` requires the overall-text and overall-text-plus-structure forecasts for citations and semantic reuse. It recomputes paired intervals from the saved predictions, without refitting models. It plots annual errors, empirical distributions of individual loss reductions, and comparisons for patents with all three roles available. Subgroup comparisons are exploratory.

Outputs are SVG, PDF and PNG files under `work/figures`, together with the forecast figure's numerical source tables. Sample sizes and test years are read from the data rather than hardcoded. Use the same encoder, data snapshot, fitting design and bootstrap settings as the corresponding tables when generating publication figures.
