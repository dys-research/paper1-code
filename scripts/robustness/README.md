# Supplementary analyses

These commands reuse the main estimators and dated measurement outputs. They write only to `work/robustness` (or the configured work root). They do not require historical acceptance files, private experiment folders, a paid model endpoint, or a GPU.

Run from the repository root after the measurement and main analysis workflows:

```bash
python -m scripts.robustness.prepare
python -m scripts.robustness.explanatory
python -m scripts.robustness.roles
python -m scripts.robustness.inference
python -m scripts.robustness.calibration
```

`prepare` creates `data/local_panel.parquet`, source/model/span-length summaries, P/E overlap summaries and K=5/20 novelty variants. If a supplementary input is unavailable, it records the skipped block in `results/preparation.json`; it does not invent substitute observations. `explanatory` likewise records unavailable optional specifications. Estimator failures are errors, not silently skipped results.

## Inputs

| Input | Default location under the work root | Purpose |
| --- | --- | --- |
| Measured panel | `measurement/data/panel_model_ready.parquet` | Main outcomes, controls, roles and anchoring; override with `inputs.panel` |
| Role tasks and vectors | `measurement/encoding/full_frozen/task_metadata.parquet`, `vectors.npy` | Within-patent role overlap |
| Cached neighbours | `measurement/encoding/full_frozen/novelty/<kind>/metrics.parquet`, `nearest_neighbors.npz` | K=5/20 without repeating historical search |
| Extracted role records | `measurement/data/role_inputs_primary.parquet` | Optional confidence metadata; override with `inputs.role_inputs` |
| Citation scope and edges | `measurement/data/citation_scope_cpc_sets.parquet`, `study_citation_edges.parquet` | CPC-composition controls |
| Citation flags | `measurement/data/five_year_citation_flags.parquet` | Exclusion of known internal citations |
| Main explanatory estimates | `analysis/explanatory/<target>_main_M*/parameters_covariance.npz` | Joint tests and observed-support marginal associations |
| Held-out forecasts | `analysis/results/forecast_predictions.parquet` | Probability calibration and fixed budgets; use `calibration --predictions PATH` for another prediction file |

Neighbour kinds are `overall_tac`, `P`, `M`, `E`, `PM_config` and `ME_config`. Task metadata require `patent_id`, `kind` and `embedding_row`; vector row indices must refer to the same deduplicated text array. Role confidence is not a calibrated probability of correctness. The source-model restriction uses `robustness.reference_extractor_model` when configured, with the supplied study's reference extractor as its default.

## Retained specifications

- Exclusion of known same-assignee and same-family citations; continuation, divisional and CIP restrictions.
- Span length, evidence source and extraction-model controls; available confidence restrictions.
- Fixed-history K=5/20 comparisons and alternative reuse thresholds.
- CPC-composition controls and repeat-assignee fixed effects.
- Fixed-sample anchoring availability/coverage/value/interaction ablation.
- P/E role drops and merging; nonidentical-P/E and source-model restrictions.
- Matched-domain grouped-binomial models conditional on observed uptake.
- Joint block Wald tests, subfield marginal associations with Benjamini–Hochberg adjustment, observed-support anchoring associations and complete-case restricted cubic splines.
- Held-out calibration and annual fixed 1%, 5% and 10% screening budgets, with ties broken by patent identifier.

`inference` requires the main M1–M6 models. Its spline output is a conditional association on the linear-predictor scale; it is not a causal effect. Calibration intervals condition on the stored forecasts and use primary-assignee clusters.

## Local lexical baseline

```bash
python -m scripts.robustness.lexical
python -m scripts.analysis.forecast --features common,lexical
```

The lexical command reads the measured full history and `data/data/a61b_text_base_1995_2020.parquet` under the work root; override the latter with `inputs.text_panel`. It writes `robustness/data/local_lexical_features.parquet`. Each feature is the share of distinct unigrams, bigrams or trigrams that did not occur at a strictly earlier date in the available A61B corpus. Same-day first users remain tied. This is a local diagnostic inspired by Arts et al.; its corpus and preprocessing differ from the original published measure.

## Scope of the cleaned package

One-off transfer packaging, historical GPU-return acceptance, duplicated validation summaries, dated report generators and repair scripts were removed. The retained measurement workflow directly implements the corrected definitions instead of replaying those repairs. The P–M pairing and conditional-mechanism computations are retained below without their historical GPU-transfer packaging. Other legacy supplementary GPU experiments depending on unprovided payload manifests and cached results, including small-annotation representation comparisons, are not presented as independently reproduced here. Human annotations and expert adjudication files were not contained in the uploaded source archive.

This cleanup preserves the source estimators and the distinction between explanatory standardization and fitting-partition forecasting transformations. It does not certify the manuscript's numerical results; reproducing those results requires the original data releases and extraction outputs.

## Paired-text diagnostics corresponding to Table B.1

The scientific pairing functions from the supplied supplementary GPU workflow are retained in `paired_text.py`, with an exact blocked NumPy implementation that also works on CPU. Full-corpus CPU search can be slow; block sizes control memory use. No scientific dataset has been recomputed during repository cleanup.

```bash
python -m scripts.robustness.paired_text prepare
python -m scripts.robustness.paired_text conditional
python -m scripts.robustness.paired_text encode --model-dir models/PatentSBERTa
python -m scripts.robustness.paired_text shuffled-novelty
python -m scripts.robustness.paired_text fit
```

- `prepare` keeps usable P/M spans, groups patents by grant year and subfield, permutes each group with the supplied seed, and cyclically shifts donors to remove self-pairs. Single-patent groups are excluded. It writes shuffled texts and donor mappings; it makes no model or API call.
- `conditional` finds the ten most similar historical P passages in the same subfield and preceding ten calendar years, requiring valid M vectors in the historical pool. It then computes one minus the mean cosine between the focal M and those historical M vectors. Fewer than ten eligible neighbours remains missing.
- `encode` uses a local copy of the **same checkpoint** as the main embeddings, CLS pooling, 512-token truncation and L2 normalization. `local_files_only=True` prevents downloads or hosted API calls. Install the encoder requirements for this optional action. `inputs.encoder_model` can replace `--model-dir`.
- `shuffled-novelty` compares encoded shuffled P–M texts with historical shuffled P–M texts under the same date/subfield rule. It verifies its own encoding receipt. Externally encoded arrays require `inputs.shuffled_vectors` and `inputs.shuffled_vector_ids`; the latter is CSV or Parquet with one unique `patent_id` per vector row. External encoding must use exactly the prepared shuffled texts and main checkpoint; row count alone cannot establish semantic provenance.
- `fit` uses the common complete observed/shuffled sample, keeps observed role scores, controls the donor M score, and changes the P–M residual and its anchoring interaction for the null comparison. It also fits conditional-mechanism specifications with and without the added conditional score. Outputs are `data/paired_text_metrics.parquet`, `results/paired_text_metadata.json` and separately named coefficient folders.

The single shuffle is a descriptive diagnostic. Donors can be granted later within the same year; those cases are counted explicitly. This diagnostic is not used in forecasting, supplies no permutation-test p-value, and does not establish that a linear residual uniquely measures theoretical combinatorial novelty. The code preserves the supplied study's seed and pairing rule; it does not create repeated permutations.

The original conditional-mechanism regressions also restricted the sample to patents with nonmissing overall novelty under the original, strict-ten-year and five-year-history variants. The cleaned conditional estimator itself is unchanged, and `fit` continues to default to available conditional and original-overall observations. An explicit `--conditional-sample PATH` remains supported. For the paper's common sample, use the generator and manuscript mode documented below. Sample rules and counts are written in result metadata.

## Additional manuscript diagnostics

These are optional additions. They do not change the original `robustness` stage
or replace the primary semantic-reuse outcome. `python run.py diagnostics --dry-run`
lists their separate workflow entries; run individual entries after providing
their required inputs.

### Kelly-inspired forward/backward similarity

After measurement steps 33 and 34 have prepared the title-plus-abstract metadata
and vectors, run:

```bash
python run.py diagnostics --step kelly -- --device cpu
```

`--device cuda` uses the optional encoder/PyTorch environment. The calculation
reads the existing semantic-reuse vector files without modifying them, uses
bounded blocks and saves a resumable checkpoint in `work/robustness/kelly`.
Changed inputs, code or block settings require a new work directory. There is
no model API call or new encoding.

For each focal patent, sum positive cosines and count valid comparison texts
in the forward interval `(grant, fifth anniversary]` and backward interval
`[grant minus five years, grant)`. Same-day grants are excluded. Divide each sum
by its window's text count, then divide the forward mean by the backward mean
and take `log1p`. Negative cosines contribute zero to the sum but their valid
texts remain in the count. Invalid vectors are excluded explicitly. Missing
query vectors, incomplete observation windows and zero backward similarity
produce a missing ratio. Historical completeness uses the earliest candidate
date, matching the supplied study; document the actual upstream corpus coverage.

Existing study aggregates can be summarized without repeating the large search:

```bash
python run.py diagnostics --step kelly -- --aggregates PATH_TO_AGGREGATES.parquet
```

The aggregate table requires unique `patent_id`, `grant_date`, exact
`label_window_end`, Boolean `complete_backward5y`, `fsum`, `bsum`, `fn` and `bn`.
An optional Boolean `query_valid` records invalid focal vectors; the original
aggregate generator accepted only finite nonzero vectors. Both vector and
aggregate routes write `kelly/kelly_patent_scores.parquet` and
`kelly/kelly_summary.json` under the robustness work directory. The supplied
study's reference check is 134,532 valid ratios and a median log-one-plus mean
ratio of about 0.696078. The program derives these values from inputs.

This is an embedding-based retrospective uptake diagnostic corresponding to
Appendix B and Response R3.6.1. It uses future information and differs from the
original Kelly et al. TF-IDF measure; it is not a forecast input.

### Original conditional-mechanism common sample

First calculate the conditional score with `paired_text conditional`. Supply the
historical overall-novelty variants as a CSV/Parquet table with unique
`patent_id`, `strict10_overall_novelty_raw` and `window5_overall_novelty_raw`.
Every focal patent must have a row; unavailable scores are explicitly missing.
These are the original stricter ten-year comparison (excluding known family
and shared-owner histories) and five-year history comparison. Their generation
or archived input must use the same source data and encoder as the main panel;
they are not inferred from the main score or filled with zero.

```bash
python run.py diagnostics --step conditional_sample -- --history-variants PATH_TO_HISTORY_VARIANTS.parquet
python -m scripts.robustness.paired_text fit --conditional-rule manuscript
```

Alternatively configure `inputs.history_novelty_variants`. The generator reads
the measured panel plus `work/robustness/data/paired_text_metrics.parquet` and
selects patents with nonmissing conditional-M novelty and all three overall
novelty variants. `--metrics PATH` can read the original novelty-wide table
instead of the newly generated metrics. It writes a unique patent-ID CSV and
provenance JSON at `paired_text/input/conditional_manuscript_sample.*`, under
the robustness work directory. IDs, input hashes and inclusion counts make the
selection inspectable; no target sample size is hardcoded.

`fit --conditional-rule manuscript` requires this generated CSV, or the file
specified by `inputs.conditional_sample` / `--conditional-sample`. Every fixed
sample ID must have the necessary conditional and original-overall scores;
unexpectedly unavailable or missing IDs raise an error rather than silently
shrinking the sample. The added score, estimators and shuffled-pair analysis are
unchanged. The archived study has 128,026 common-sample patents versus 128,027
under the retained available-observation default. These counts are validation
references, not selection rules. The corresponding comparison is Table B.1.

### Excluding internal links from histories and subsequent text reuse

The optional commands below add the original internal-link sensitivity branches
without changing the main novelty or semantic-reuse measures:

```bash
python -m scripts.robustness.internal_exclusions history
python -m scripts.robustness.internal_exclusions reuse
python -m scripts.robustness.internal_exclusions fit
```

Run measurement and its family reconstruction first. The commands read the
date-aware domestic family cache without CIP and the existing assignee-link
portfolio. `--family-cache PATH`, `--assignee-links PATH` and
`--measurement-root PATH` can select compatible existing inputs. Assignee links
require `patent_id` and either `entity_id` or `assignee_id`; all observed owners
are retained. Provide the same ownership scope as the intended comparison.
An unobserved owner is kept as unknown, not presumed to match another patent.

`history` compares the same six primary representations within subfield. The
`strict10` branch uses the preceding ten calendar years and excludes patents in
the focal patent's family at its grant date or sharing any observed assignee.
The `window5` branch uses the preceding five years without internal-link
exclusion. Each score requires ten valid neighbors. Results are stored in
`data/history_novelty_variants.parquet` under the robustness work directory;
the conditional-model sample generator automatically accepts this output.
Field-level results can be resumed; changed inputs require a new work directory.

`reuse` reads the existing title-plus-abstract query/candidate vectors and
metadata from `measurement/semantic_reuse/all_future_text`, optionally overridden
with `--source-job PATH`. It counts all subsequent texts with cosine at least
0.8 in the exact five-year window, independently of citation links. Family
membership is evaluated at the subsequent candidate's grant date. Separate
counts exclude known same-family matches, any-shared-assignee matches, or their
union. Unknown owners remain eligible, and the number of matches with known
ownership on both sides is recorded. Each alternative is transformed by
`log1p`; results are saved in `data/internal_excluded_reuse.parquet`.

Both searches use bounded blocks and support `--device cpu` or `--device cuda`.
`--query-rows` and `--candidate-rows` control memory; `--patent-id ID` selects
queries for a separate verification run. `history --kind P` selects a single
representation for such a check. Full comparison fitting requires all six
representations and all focal records.

`fit` estimates the original, strict-history and five-year-history models on
their common observed-overall sample, and the three self-excluded semantic-reuse
alternatives with the unchanged estimators. Existing compatible results can be
read using `--history-metrics PATH` and `--reuse-results PATH`; `--branch history`
or `--branch reuse` runs one family of comparisons. Outputs use separate
`internal_history_*` and `internal_future_*` names. Main model outputs are not
overwritten.
