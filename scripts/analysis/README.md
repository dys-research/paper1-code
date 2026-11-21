# Explanatory models and mature-label forecasts

These scripts compute the revised study's explanatory specifications and forecast comparisons from a measured patent panel. They do not require the author's historical acceptance receipts, runtime directories, fitted models, or local drive layout.

## Inputs

By default the panel is `work/measurement/data/panel_model_ready.parquet`. Set `inputs.panel` to use another panel with the same schema. `common.py` defines the 23 traditional features, novelty columns and outcome names. Patent identifiers must be unique. The panel must include exact grant dates, fifth-anniversary outcome-window ends, label-availability dates, subfields and assignee-cluster keys. Unknown assignees must have individual singleton keys.

Overall-text comparisons use `work/measurement/encoding/full_frozen/vectors.npy` and its `task_metadata.parquet`, selecting `kind == overall_tac` by patent identifier. Alternatively configure both `inputs.overall_vectors` and `inputs.overall_vector_ids`. The latter is a Parquet/CSV `patent_id` column or a UTF-8 file with one patent identifier per row. Vectors must have 768 columns. Identifier-based alignment is mandatory; row counts alone are insufficient.

## Commands

Run from the repository root:

```bash
python -m scripts.analysis.explanatory
python -m scripts.analysis.forecast
python -m scripts.analysis.summarize
```

The default forecast evaluates all six feature sets on all four outcomes for 2015–2020. Options support a smaller run without changing the feature definitions:

```bash
python -m scripts.analysis.forecast --features common,structure --targets top5 --learner tree
python -m scripts.analysis.summarize --pairs structure:common
```

The tree command reproduces the feature comparison underlying the manuscript's separate nonlinear screening result. The CLI also allows all six feature sets with the same tree learner; those additional comparisons are new evaluations, not archived paper results.

Each forecast command writes summary inputs for that selected run; to preserve multiple selections, use distinct configured work directories or retain their generated outputs before running another selection. Job caches include input and code fingerprints and reject incompatible reuse. Use `--overwrite` to explicitly recompute a changed design.

## Specifications retained

`explanatory.py` retains same-sample M1–M6 PPML/OLS, single-role additions, complete-case models, assignee-cluster covariance and core correlation/VIF diagnostics. As in the supplied implementation, explanatory novelty scores are standardized within grant-year × subfield cells before joint-score residualization; the residual is standardized within those cells again. Forecasting residualization instead uses raw novelty in the fitting partition. These operations are intentionally documented separately.

`forecast.py` retains the mature-label rule: initial training through test year minus eight, validation in years minus seven/minus six, and final fitting through minus six. Label windows and availability are checked against 1 January of the test year. Imputation, residualization, scaling, PCA and hyperparameter selection are restricted to the appropriate fitting data.

The six primary feature sets are:

| Name | Information added to the common baseline |
| --- | --- |
| `common` | None: 23 traditional features, grant year, subfield and overall novelty |
| `roles` | P/M/E novelty and availability |
| `text_fields` | Title, abstract and first-independent-claim novelty and missingness |
| `structure` | Role novelty, residual joint novelty, anchoring, coverage, availability and P–M × anchoring |
| `overall` | Fitting-only PCA of the overall-text embedding, 16 components by default |
| `overall_structure` | Both overall-text PCA and the complete structured module |

An optional `lexical` extension adds the three local unigram/bigram/trigram first-appearance shares. Generate them with `python -m scripts.robustness.lexical`, then use `python -m scripts.analysis.forecast --features common,lexical,structure`. The default input is `work/robustness/data/local_lexical_features.parquet`; `inputs.lexical_features` can override it. This benchmark is inspired by Arts et al. and uses the local A61B history; it is not a replication of their full corpus measure.

The GLM learners and penalty grid match the supplied workflow. PCA and tree random seeds retain the supplied value. The tree grid retains 7/15 leaves, 100 iterations, learning rate 0.1 and L2 regularization 1.0. The 5% screening budget uses a fixed floor of the annual test population, with patent identifiers resolving score ties.

## Outputs and manuscript mapping

| Generated output under `work/analysis` | Purpose |
| --- | --- |
| `results/explanatory_coefficients.csv` | Main regression tables and single-role comparisons |
| `explanatory/*/parameters_covariance.npz` | Marginal effects, joint tests and uncertainty calculations |
| `quality/explanatory_core_correlations.csv` and `explanatory_core_vif.csv` | Measurement dependence diagnostics |
| `results/forecast_metrics_by_year.csv` | Annual performance, Table C.2/C.3-type comparisons |
| `results/forecast_metrics_summary.csv` | Equal-year metrics, Tables 6 and C.1-type comparisons |
| `results/forecast_predictions.parquet` | Paired held-out predictions for inference and figures |
| `results/paired_cluster_intervals.csv` | Primary-metric differences with paired cluster intervals |
| `results/paired_holm_four_tasks.csv` | The same paired contrasts with centered bootstrap p-values and four-task Holm adjustment |
| `metadata/forecast_design.json` | Exact source/code fingerprints and forecast design |

The source archive did not include the patent data or historical model outputs. Running these scripts computes results; the repository does not assert that all manuscript numbers have been independently reproduced. Exact numerical agreement also depends on the original data snapshot, extraction outputs, encoder checkpoint and software versions. The current design is a retrospective evaluation of grant-disclosure information with modern measurement tools.

## Four-task prediction inference

`summarize` retains the existing means, differences and paired cluster intervals.
It additionally calculates the source workflow's two-sided centered-null
bootstrap sensitivity p-value: `(1 + count(abs(draw - point) >= abs(point))) /
(replicates + 1)`. This conditions on the fitted predictions and has finite
resampling resolution. Holm adjustment is applied separately to each learner
and candidate/baseline pair across citations, semantic reuse, new-field citations
and top-5% identification. This is distinct from the BH adjustment for subfield
regression follow-ups.

Run forecasts for all four endpoints before requesting this family. A subset
run still writes its intervals and raw p-values, but leaves
`p_Holm_four_endpoints` missing and marks `holm_status` as
`incomplete_four_task_family`; it does not present a smaller correction as the
paper's four-task test. Keep all four tree outcomes when reproducing Section 6.3:

```bash
python -m scripts.analysis.forecast --features common,structure --learner tree
python -m scripts.analysis.summarize --pairs structure:common
```

`paired_cluster_intervals.csv` keeps its original fields and gains
`centered_bootstrap_p`, `p_Holm_four_endpoints` and explicit correction-family
status columns. `paired_holm_four_tasks.csv` exposes the same enriched table.
With the archived study predictions and 500 replicates, the top-5% tree contrast
has adjusted p approximately 0.023952 (reported as 0.024). This is a reference
check, not a constant used by the computation.
