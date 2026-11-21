# Forecasting technological impact from structured patent text

Research code for **Forecasting Technological Impact at Patent Grant Structured
Problem Mechanism Effect Novelty and Out of Time Evidence**. The focal study
examines U.S. A61B utility grants from 2005 through 2020, with 1995–2004 grants
providing early historical text references.

The workflow extracts source-grounded problem, mechanism and effect evidence,
constructs dated novelty and subsequent-impact measures, and evaluates
explanatory associations and mature-label out-of-time forecasts. This repository
contains source code and synthetic tests. Data, extraction responses, model
weights, fitted objects and generated results are supplied or generated locally.

## Setup

Use Python 3.12 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate the environment using `.venv/bin/activate` on macOS/Linux or
`.venv\Scripts\Activate.ps1` in Windows PowerShell, then run:

```bash
python -m pip install -r requirements.txt -c constraints-tested.txt
python -c "from pathlib import Path; Path('config.toml').write_bytes(Path('config.example.toml').read_bytes())"
python run.py list
python run.py check
python -m unittest discover -s tests -v
```

Fill in the input locations in the private `config.toml`. Relative paths resolve
from that file's directory. Set `TFSC_CONFIG` or pass `--config` to the runner to
use a configuration in another location. Generated files default to `work/`.
Both private configuration and generated data are excluded from Git.

Install `requirements-encoder.txt` when generating embeddings. CUDA execution
also requires a PyTorch installation compatible with the available GPU. The
tested constraints cover the data/analysis environment; the encoder environment
must be recorded separately when it is used.

## Workflow

| Stage | Command | Input contract |
| --- | --- | --- |
| Prepare sample, text and controls | `python run.py data` | [Data preparation](scripts/data/README.md) |
| Prepare and validate extraction | `python run.py extract --until-step run_extraction` | [Extraction](scripts/extraction/README.md) |
| Compute dated measures and outcomes | `python run.py measurement` | [Measurement](scripts/measurement/README.md) |
| Fit explanatory models and forecasts | `python run.py analysis` | [Analysis](scripts/analysis/README.md) |
| Run supplementary checks | `python run.py robustness` | [Robustness](scripts/robustness/README.md) |

`extract` validates input without sending API requests by default. To perform a
new extraction, configure the model, complete chat-completions URL and credential
environment-variable name, then run:

```bash
python run.py extract --step run_extraction -- --send
python run.py extract --step prepare_roles
```

Existing extraction responses can be imported through `inputs.role_outputs`.
Primary role embeddings use a literal contiguous span from its declared source
field. Normalized paraphrases are retained only as metadata.

Inspect a stage's commands before execution, or resume a selected step:

```bash
python run.py measurement --dry-run
python run.py measurement --step 23_run_temporal_novelty -- --engine numpy
python run.py analysis --step forecast -- --features common,structure --targets top5 --learner tree
```

Set an immutable `encoder.revision` for either a remote or local checkpoint.
For a remote checkpoint, run the encoding step with `--allow-download`. For an
existing local checkpoint, also configure `inputs.encoder_model`.
Whole-corpus title-and-abstract vectors for semantic
reuse have their own explicit [input contract](scripts/measurement/REUSE.md).

## Scientific definitions

- Novelty compares a focal patent with earlier patents in the same mapped A61B
  subfield over a fixed ten-calendar-year window; ten eligible neighbors are
  required for the main score.
- Citation and subsequent-text outcomes use the interval after grant through
  its fifth anniversary, including the anniversary. New-field events share no
  CPC subclass with the focal patent's complete recorded portfolio.
- Fitting for test year `t` uses initial training through `t-8`, validation in
  `t-7` and `t-6`, and refitting through `t-6`. Preprocessing and PCA use the
  corresponding fitting data.
- The six principal feature combinations share the same traditional baseline
  and held-out patents. Screening uses a fixed annual 5% budget. Model
  comparisons use paired assignee-cluster resampling of held-out predictions.

The explanatory and forecasting residualization orders are documented in the
analysis guide. Internal legacy column names containing `config` are retained
for schema compatibility; their reported construct is residual joint-text
novelty.

Publication plots can be regenerated from saved analysis outputs using the
[figure scripts](scripts/figures/README.md). The main manuscript's pairing
diagnostics have a separate optional workflow in the robustness guide.

### Additional manuscript diagnostics

The existing data, extraction, measurement, analysis and robustness stages keep
their execution order. The optional `diagnostics` stage adds the Kelly-inspired
forward/backward calculation and a generator for the original conditional-model
common sample. See [the robustness guide](scripts/robustness/README.md#additional-manuscript-diagnostics)
for input contracts and commands. The prediction summary also writes centered
bootstrap p-values and four-task Holm adjustments, described in the analysis
guide; its original metrics and confidence intervals are retained.

## Reproducibility and data

This repository provides research code, configuration templates and tests for
the study workflow. Patent data, extraction responses and model weights are
prepared locally. The required input tables and their schemas are described
in the [data preparation guide](scripts/data/README.md).

The workflow accepts normalized PatentsView Parquet tables. Users should prepare
these tables from the relevant bulk data release before running the preparation
stage. Alternatively, setting `inputs.panel` allows the analysis to begin from
a compatible measured patent panel.

Reproducing the manuscript results requires the corresponding data release,
extraction responses, encoder checkpoint and analysis settings. Record the source
release, observation boundary, model versions and software environment for each
run. Validation coverage, including software tests and targeted numerical
comparisons with archived study outputs, is documented in
[`docs/VALIDATION.md`](docs/VALIDATION.md).

The analyses retrospectively evaluate information disclosed at patent grant
using current CPC assignments and modern language models. The measurement
definitions and temporal evaluation rules are documented in the
[measurement](scripts/measurement/README.md) and
[analysis](scripts/analysis/README.md) guides.

## Human-validation materials

To protect annotator privacy, human-validation materials will be made available
after formal publication upon reasonable request, in a form that safeguards
participant privacy.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
