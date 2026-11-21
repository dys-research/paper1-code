# Measurement pipeline

This directory builds traditional grant-time features, dated outcomes, role
representations, novelty and observed knowledge anchoring. It retains the
revised study definitions and removes earlier run-specific extraction batches,
GPU transfer packages, repair scripts, acceptance reports and legacy local-path
loaders. Raw data, extraction responses, model weights, vectors and result
panels are not included in the repository.

## Prerequisites

Run the data-preparation and extraction-validation steps first. They produce:

- `work/data/data/a61b_text_base_1995_2020.parquet`
- `work/data/data/candidate_control_availability.parquet`
- `work/extraction/role_inputs_primary.parquet`

Here and below, `work` means the configured `paths.work_dir`. An optional
`inputs.role_inputs` overrides the canonical extraction Parquet. Parsed JSONL
responses use `inputs.role_outputs` and must pass through extraction validation
first; they are never ingested directly by measurement. It
checks every proposed span again against the current source text; generated
`normalized_text` is never embedded. Recorded missing/input-changed extraction
states remain unavailable. The exporter saves the current input fingerprint,
which this step checks again to detect source edits after export.

The normalized `inputs.patentsview_root` additionally needs these files:

| File | Required columns |
| --- | --- |
| `02_core/patent_dates.parquet` | `patent_id`, `patent_date` |
| `02_core/patent_cpc.parquet` | `patent_id`, `cpc_group`, `cpc_subclass`, `cpc_class`, `cpc_section` |
| `02_core/patent_assignee.parquet` | `patent_id`, `assignee_id` |
| `02_core/patent_inventor.parquet` | `patent_id`, `inventor_id` |
| `02_core/citation.parquet` | `patent_id` (citing), `citation_patent_id` (cited) |
| `02_core/patent_application.parquet` | `patent_id`, `application_id` |
| `01_stage_parquet/us_rel_doc.parquet` | `patent_id`, `related_doc_type`, `related_doc_kind`, `related_doc_number`, `published_country` |

The core dates/citation tables must represent the intended utility-grant scope.
The repository does not infer a compatible upstream normalization from arbitrary
PatentsView TSV schemas. The data-preparation README describes its additional
text, classification and control inputs. Record source release identity and
coverage before treating zero observed events as a complete five-year label.

## Execution order

Use `python run.py list` to see the declared sequence. These ranges prepare the
non-neural features and role validation:

```bash
python run.py measurement --from-step 01_build_base_panel --until-step 11_assemble_panel_and_quality
```

The sequence is:

1. `01_build_base_panel`: source text lengths, CPC breadth, entity links and base controls.
2. `02_build_entity_histories`: strictly earlier grant-based entity histories.
3. `03_build_citation_outcomes`: exact five-anniversary citations and complete-CPC new-field counts.
4. `04_prepare_role_inputs`: revalidate source-contiguous P/M/E evidence.
5. `05_build_temporal_families`: date-truncated domestic continuity components, with and without CIP links.
6. `06_build_self_citations_and_impact`: internal-link exclusions and strictly earlier assignee citation histories.
7. `11_assemble_panel_and_quality`: combine covariates, labels and role availability.

Next, set an immutable `encoder.revision` and prepare role/overall-text jobs:

```bash
python run.py measurement --step 22_prepare_encoding_manifest
python run.py measurement --step 18_encode_frozen_texts -- --device cpu --allow-download
python run.py measurement --step 23_run_temporal_novelty -- --engine numpy --device cpu --resume
python run.py measurement --step 32_prepare_anchoring_inputs
```

`--allow-download` permits downloading the configured public encoder checkpoint;
it does not call a paid LLM. For offline use set `inputs.encoder_model` to that
checkpoint's local directory and omit the flag. PyTorch and SentenceTransformers
are required for encoding. The encoder checks the fixed CLS pooling, 512-token
cap and 768-dimensional representation. A CUDA run requires explicit
`--device cuda`; failures do not silently select a different bulk-compute device.

The encoding job is `work/measurement/encoding/full_frozen`. It deduplicates
texts, saves text/configuration hashes, records truncation lengths and resumes
completed blocks. Do not reuse a work directory after changing model, source
texts or encoding settings. Patent IDs and representation-to-vector rows are in
`task_metadata.parquet`; vectors are `vectors.npy`. Representation kinds are
`P`, `M`, `E`, `PM_config`, `ME_config`, `overall_ta`, `overall_tac`, `title`,
`abstract`, and `claim_1`. Internal `*_config` names are retained for schema
compatibility; the analyses call the adjusted scores residual joint-text novelty.

The novelty formula is one minus the mean of the ten highest eligible cosines.
History must be in the same mapped A61B subfield and within
`[grant minus10calendar years, grant)`. Same-day and future grants are excluded.
Fewer than ten valid historical neighbors produces a missing primary score.
The runner retains up to20neighbors for sensitivity checks. Residualization is
performed later within the corresponding explanatory or prediction fitting
sample, not during measurement.

Prepare and calculate subsequent text reuse using the separately configured
all-patent title-and-abstract index/vectors; [REUSE.md](REUSE.md) specifies the
input contract and provenance limits:

```bash
python run.py measurement --from-step 33_prepare_future_reuse_metadata --until-step 34_pack_reuse_vectors
python run.py measurement --step 35_run_future_text_reuse -- --device cpu
python run.py measurement --step 36_compute_anchoring_from_vectors
python run.py measurement --step 37_assemble_final_panel
```

No historical local acceptance JSON, original machine directory or GPU return
archive is needed. The final output is
`work/measurement/data/panel_model_ready.parquet`, with a generated schema and
measurement dictionary in `variable_dictionary_final.csv`. Raw missing scores
remain missing. Analysis-specific imputation and transformations occur later.

## Lightweight validation

```bash
python run.py measurement --step 19_test_temporal_metrics
python run.py measurement --step 35_run_future_text_reuse -- --self-test --device cpu
```

These checks use synthetic vectors to verify date boundaries, missing-history
behavior, deterministic ties, fitting-only residualization, exact CPC bits and
semantic-reuse partitions. They do not reproduce the empirical manuscript
results or independently validate neural extraction quality.
