# Data preparation

These scripts use the normalized Parquet tables consumed by the supplied research
code. They do not download PatentsView or reconstruct that normalization from raw
bulk files. Supply a consistent release and record its release date in the root
configuration. The data themselves are not included.

Run from the repository root using `python run.py data` (or the equivalent ordered
scripts below). The runner supplies the repository import path.

1. `build_sample.py`: select non-withdrawn U.S. A61B utility grants from 1995–2020,
   retaining 1995–2004 as historical references and 2005–2020 as focal patents.
2. `build_texts.py`: choose the earliest non-cancelled independent claim and join
   title and abstract.
3. `build_controls.py`: collect source control fields. Subsequent measurement
   scripts rebuild date-truncated histories and domestic family variables.

## Required normalized input tables

Configure `inputs.patentsview_root`; table paths below are relative to that root.

| Table | Required columns |
| --- | --- |
| `02_core/patent_dates.parquet` | `patent_id, patent_date, filing_date, grant_year, filing_year` |
| `02_core/patent_cpc.parquet` | `patent_id, cpc_sequence, cpc_group, cpc_subclass`; downstream also uses `cpc_class,cpc_section` |
| `02_core/patent_primary_field.parquet` | `patent_id, cpc_subclass, cpc_group` |
| `01_stage_parquet/patent.parquet` | `patent_id, patent_type, withdrawn` |
| `02_core/patent_text.parquet` | `patent_id, patent_title, patent_abstract` |
| `02_core/patent_assignee.parquet` | `patent_id, assignee_id, assignee_sequence` |
| `02_core/patent_application.parquet` | `patent_id, application_id` |
| `03_metrics/abstract_npl_metrics.parquet` | `patent_id, non_patent_reference_count, doi_like_reference_count, title_word_count, abstract_word_count` |
| `03_metrics/assignee_family_value_metrics.parquet` | `patent_id, grant_lag_years, num_claims, related_us_document_count, foreign_priority_count, pct_document_count, assignee_prior_patents_mean` |

Dates, patent text, primary classification, applications, and metric tables must
have at most one row per patent. CPC and entity-link tables may contain multiple
rows per patent. Additional tables for measurement are listed in its README.
Source history fields above are retained for compatibility/audit, not substituted
for the downstream date-truncated histories.

Configure `inputs.claims_root` as either a claim file or a directory containing
Parquet, TSV, TSV.GZ, or ZIP files whose TSV members have these columns:
`patent_id, claim_sequence, claim_number, claim_text, dependent`. Supply all
1995–2020 relevant claims. An empty/null dependency denotes an independent claim;
cancelled claims are excluded, and ties with conflicting text raise an error.
Claim files must not contain unrelated ZIP/GZ archives.

## Output contract

Files are written below `work/data/data/` by default:

- `a61b_candidates_1995_2020.parquet`: unique string `patent_id`, grant/filing dates
  and years, first A61B group by recorded CPC sequence (code order resolves ties),
  focal/history flags, and exact fifth-anniversary label end.
- `selected_independent_claims_1995_2020.parquet`: selected claim and its source
  basename, sequence, number, dependency and selection rule.
- `a61b_text_base_1995_2020.parquet`: candidate columns plus `patent_title`,
  `patent_abstract`, `independent_claim_text`, claim provenance and text coverage
  flags. `full_input_available` requires all three fields.
- `candidate_control_availability.parquet`: source counts, primary assignee,
  singleton fallback cluster IDs and application ID consumed by measurement.

Preparation records input basenames and sizes under `work/data/checks/`. No old
processed panel, proprietary local directory, legacy embedding cache or earlier
LLM run is required. A changed data snapshot can legitimately change sample size;
these scripts do not hardcode the manuscript's observed row counts as eligibility
criteria.
