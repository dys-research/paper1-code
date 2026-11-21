# P/M/E extraction

This folder replaces the original local-script loader and provider-specific batch
wrappers. `prompts/pme.txt` preserves the original extraction system prompt. The
client accepts an OpenAI-compatible chat-completions endpoint. It does not embed
normalized paraphrases: primary embeddings use only literal source-valid spans.

Set `extraction.model`, `extraction.endpoint`, and `extraction.api_key_env` in the
local configuration. `extraction.concurrency` defaults to 4 and `extraction.temperature`
defaults to 0, matching deterministic sampling in the original workflow.
Store the API key only in the named environment variable (default `TFSC_LLM_API_KEY`).
Do not put a key in source code or configuration files. The endpoint is the full
chat-completions URL. Use `--disable-thinking` only for a provider that supports
that extra field.

Run from the repository root. The runner sets the source import path:

```bash
python run.py extract --step prepare_inputs
python run.py extract --step run_extraction -- --limit 10
# The previous command validates inputs and sends zero requests.
python run.py extract --step run_extraction -- --limit 10 --send
python run.py extract --step run_extraction -- --send
python run.py extract --step prepare_roles
```

`prepare_inputs.py` reads `work/data/data/a61b_text_base_1995_2020.parquet`,
skips incomplete three-field inputs, and writes `work/extraction/input.jsonl`.
Only patent identifier, grant year, field label and source text are sent; outcome
labels, citations and other future information are not included. Text sources
use whitespace-only canonicalization. Stable request IDs include their input hash.

The `run_extraction` step defaults to validation without network requests. `--send`
starts requests. Successful outputs are appended to `parsed.jsonl`; restarting
skips completed request IDs. A saved batch manifest prevents silent reuse when
the input, model, prompt or generation settings change. API failures are retried;
invalid individual spans remain missing rather than causing valid roles to be
discarded. Errors record only status/type and identifiers, never headers or API
response bodies. Do not edit a running batch's input file. `--limit` applies to
remaining, not already completed, records.

`prepare_roles.py` exports three rows per text-base patent to
`work/extraction/role_inputs_primary.parquet`. It records absence, no output,
invalid span, input mismatch and incomplete input separately. `embedding_text`
is populated only for a present, literal span in its declared source field.
The output includes source positions, confidence and extraction model metadata.
It stores the current text-source fingerprint in
`current_input_sources_sha256`, separately from the extraction's original
`input_sources_sha256`, so downstream steps can detect source changes after
export. `input_hash_verified` reports whether both fingerprints were available
and matched. The script also writes a JSON coverage summary.

The role-export step requires existing parsed outputs. A dry-run does not create
them. Run the sending step explicitly before exporting, or import existing
parsed JSONL without calling a model:

```bash
python run.py extract --step prepare_roles -- --parsed data/parsed.jsonl
```

Alternatively configure `inputs.role_outputs`. Nested `elements` objects and original flat
`P_present`/`P_evidence_span`-style records are supported. Import batches must
contain at most one record per patent. Full-input hash mismatches are rejected
for primary measurement. Legacy records without a hash require the explicit
`--allow-missing-input-hash` option and are flagged as unverified; exact source
matching still applies. This option cannot establish that the original model
saw the same full input.

Raw texts, responses, credentials and generated Parquet files belong in ignored
local working directories, not in the public repository. A fresh extraction may
differ from the study's historical mixed-model outputs. The supplied source code
does not include those outputs or human annotations.
