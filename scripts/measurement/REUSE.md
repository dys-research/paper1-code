# Subsequent semantic reuse

These scripts count similar subsequent patent texts independently of citation
links. They do not download, generate or validate a historical semantic corpus.

## Required inputs

Set `inputs.semantic_index` and `inputs.semantic_vectors` in the local project
configuration. Both focal patents and subsequent candidates must use the same
encoder checkpoint, CLS pooling, text template and truncation policy. For the
study, the semantic-reuse text is title plus whitespace plus abstract. Record
the upstream data release, encoder revision, input template and source coverage
in the run's provenance. The scripts validate IDs, vector rows and grant dates;
these checks do not establish encoder equivalence or its training cutoff.

The Parquet index requires:

| Column | Type | Meaning |
| --- | --- | --- |
| `patent_id` | numeric ID or numeric string | Unique utility-patent identifier |
| `embedding_row` | integer | Unique zero-based row in the vector array |
| `patent_date` | date | Grant date matching the normalized patent-date table |
| `emb_valid` | Boolean | Upstream vector-validity flag; false rows are excluded |

Vectors must be a floating-point NumPy `.npy` array with 768 columns. Vector
rows may be normalized or unnormalized; calculation normalizes valid rows in
float32. Nonfinite and zero vectors are excluded. Every focal patent needs an
index record; an invalid focal vector produces a missing label rather than a
zero count.

The normalized input root also needs `02_core/patent_dates.parquet` and
`02_core/patent_cpc.parquet`. The latter contains `patent_id` and
`cpc_subclass`. The base-panel preparation must already have produced
`data/patent_panel_base.parquet` under the measurement work directory.

## Execution order

Run from the repository root after configuring inputs:

```bash
python run.py measurement --step 33_prepare_future_reuse_metadata
python run.py measurement --step 34_pack_reuse_vectors
python run.py measurement --step 35_run_future_text_reuse -- --device cpu
```

The scripts can also be invoked directly with the repository root on
`PYTHONPATH`. The computation supports `--device cuda`, `--query-rows` and
`--candidate-rows`. Its synthetic check is:

```bash
python run.py measurement --step 35_run_future_text_reuse -- --self-test --device cpu
```

Outputs are stored in `semantic_reuse/all_future_text` under the measurement
work directory. Metadata construction uses complete CPC subclass sets encoded
as exact signed-int64 words with 63 bits per word. No floating-point CPC masks
are accepted.

For each focal patent, eligible subsequent grants occur strictly after its
grant date and on or before its fifth anniversary, and cannot exceed the
configured observation boundary. Cosine thresholds are fixed at 0.7, 0.8 and
0.9. Each threshold produces total, shared-subclass, new-field and unknown-CPC
counts; the three CPC partitions sum to the total. The primary outcome is
`log1p(reuse_count_80_all)`. Immature windows and invalid query vectors remain
missing. Zero denotes no qualifying match in a mature window with a valid
query.

The source arrays are opened read-only. Packing and computation use resumable
working files. Checkpoints reject changed configuration/metadata or changed
array size and modification time; large vector arrays are not fully hashed.
Use a new work directory after replacing upstream arrays or changing methods.
