# Code for structured patent novelty analysis

This repository contains the analysis code for a study of element-level novelty, knowledge anchoring, and the future impact of U.S. A61B utility patents granted from 2005 to 2019.

The repository contains code only. It does not contain patent records, claim text, LLM responses, embedding arrays, intermediate panels, fitted-model objects, figures, or result tables.

## Data source

Patent records and the supporting citation, CPC, text, inventor, and assignee tables are obtained from the United States Patent and Trademark Office PatentsView platform: https://patentsview.org/.

Users may download the required PatentsView release themselves and comply with its terms of use. The release version and access date used for a rerun should be recorded in the run log.

## Setup

Create an environment and install the dependencies:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Copy `config.example.yaml` to `config.yaml`, then replace only this placeholder with the local directory containing the downloaded PatentsView files:

```yaml
paths:
  patentsview_dir: "REPLACE_WITH_PATH_TO_PATENTSVIEW"
```

`config.example.yaml` is a public configuration template.

To place the configuration elsewhere, set the `PME_CONFIG` environment variable to its file path.

For the P/M/E extraction step, set the environment variable named in `llm.api_key_env` before running the script. The default name is `ARK_API_KEY`. Do not place an API key in `config.yaml`, a notebook, or a source file.

## Workflow

The scripts are numbered in their intended execution order.

1. `code/01_build_llm_input.py` constructs the patent universe and LLM input records from PatentsView.
2. `code/02_extract_pme_async.py` performs asynchronous P/M/E extraction. Its embedded prompt is the extraction prompt used by the workflow; model endpoint and model identifier are set in `config.yaml`.
3. `code/03_prepare_embedding_inputs.py` creates embedding inputs and a transfer manifest.
4. `code/04_embed_patentsberta.py` creates PatentSBERTa embeddings. If this runs on a separate GPU machine, place the returned arrays and metadata under `work/embedding_output/`.
5. `code/05_build_knowledge_increment.py` constructs novelty, anchoring, outcomes, controls, the modelling panel, and the base models.
6. `code/06_run_reduced_control_models.py` runs reduced-control robustness models.
7. `code/07_run_explanatory_models.py` runs explanatory models; `code/08_diagnose_cross_field_diffusion.py` reconstructs and audits the strict cross-field outcome.
8. `code/09_cache_independent_claim_counts.py` and `code/10_build_prediction_panel.py` prepare the grant-time prediction panel.
9. `code/11_run_rolling_prediction.py`, `code/12_make_prediction_figures.py`, and `code/13_year_block_bootstrap.py` run and summarize rolling out-of-time evaluation.
10. `code/14_run_extended_outcomes.py`, `code/15_run_temporal_holdout.py`, and `code/16_run_temporal_robustness.py` run supplementary robustness analyses.
11. `code/17_create_manuscript_figure_notebook.py` creates a notebook for manuscript figures after the required input panels and model outputs have been generated.

## Validation

Before a full run, check that the copied configuration points to the expected PatentsView release and run:

```bash
python -m compileall -q .
```

## Security and privacy

The public code uses relative project paths and a local, ignored configuration file. It contains no personal path, API key, access token, or raw LLM response. Do not commit `config.yaml`, `.env`, downloaded data, generated embeddings, or model outputs.

## License

The code is released under the MIT License. Patent records and other third-party materials remain subject to the terms set by their respective providers.

For questions regarding the paper, please contact the author(s) of this study.
