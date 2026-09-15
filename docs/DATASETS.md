# Dataset contribution workflow

The first export is `nls-index-cards`, split `test`, hosted in the private Hub
dataset `small-models-for-glam/glam-extraction-benchmark`. Each row contains `id`,
an embedded `image`, `target_schema` and `expected_output`. Schema and output are
JSON strings; a `provenance` JSON string preserves source IDs and review metadata.
Config identity is in the manifest and must accompany a run, separately from item IDs.

## Export and validate NLS

The exporter downloads a pinned NLS source snapshot. These commands perform no
inference and no upload. The output directory must not already exist.

```bash
uv run glam_bench/export_nls.py --output ./exports/nls-v1
uv run glam_bench/validate_dataset.py ./exports/nls-v1 --config nls-index-cards
```

The export includes a Hub dataset card declaring the config and split, a Parquet
file, a manifest, the original source schema/card and a validation report.
The manifest identifies the institution, source revision/split, converter version,
scoring policy, ordered-ID hash and data/schema checksums.

The generic local loader supports the four-column contract and config manifest;
it contains no NLS import. Admission checks cover IDs, images, schema support and
gold conformance. They do not establish human label correctness. Supported scoring
schemas currently use object/array/scalar types, enums, local nonrecursive `$ref`
and nullable `anyOf[value, null]`. Other constraints are refused pending explicit
support. Full JSON Schema gold validation and JSON-object parseability are different
checks; the old result field `schema_valid` still means the latter.

## Nullable schema and historical results

The original NLS schema permits null for optional values. The legacy loader removed
those branches: 92 of 98 gold records then fail strict validation (198 null-value
errors), although every record validates against NLS's original schema.

The new target schema retains nullability and adds the existing exact-match field
annotations. Only the scorer receives the old simplified representation. Historical
predictions must remain labelled as generated with the legacy schema; this migration
does not establish what models would produce when given the new schema.

The initial migration was checked against all 98 images/gold records and all 12 cached
runs: per-card scores and denominators were unchanged. This was an implementation
check, not an extra gate for every dataset contribution.

## Version identities

- **Harness `0.0.1`:** code version, declared in `glam_bench/version.py` and recorded
  on newly attempted prediction rows. `harness.py --version` prints it. Old rows
  keep their original provenance, including on resume; missing versions remain unknown.
- **Data contract `1`:** row/manifest format, independent of collection content.
- **Dataset revision:** immutable Hub commit plus config/split, identifying evaluated data.
- **Scorer version:** changes when scoring behaviour changes; the existing version is
  `2026-08-26a`. This export does not change that scoring behaviour.

`0.0.1` is a working code version; no release tag is implied. The future Hub benchmark
task ID is distinct from config. Result exports should preserve dataset/config/revision,
metric and scoring version, raw prediction sources and exact run settings. Registration
through `eval.yaml` and model `.eval_results` files remains a later integration.

## Run and validate a config

The default dataset is `small-models-for-glam/glam-extraction-benchmark`, with config
`nls-index-cards`. The loader resolves `--dataset-revision` to a commit, validates the
export, and sends only the image, target schema and task instructions to the model.
Gold and provenance are used locally for evaluation. Choose models and serving costs
before running inference; neither export nor Space build runs models.

```bash
uv run glam_bench/harness.py --models YOUR-REGISTRY-LABEL --config nls-index-cards \
  --dataset-revision COMMIT_SHA --max-tokens 1600
uv run glam_bench/validate_submission.py results/nls-index-cards/YOUR-REGISTRY-LABEL.json
```

Results record the dataset commit, config and split. Resume uses those alongside
the existing model/run settings. The OpenAI-compatible adapter
uses vLLM's `structured_outputs.json` for guided JSON. Provider support for JSON
Schema features varies; adapter failures remain visible in results. The specialist
Space adapter cannot accept task instructions and is unsupported for config runs.

## Inference recipes and result storage

[Optional inference recipes](../examples/inference/README.md) hold Jobs launch code,
API scripts and records of how our runs were made. They live outside the harness;
any inference setup can supply the [submission format](SUBMISSION.md).

Our new run outputs live in the private bucket
`small-models-for-glam/glam-extraction-results`, grouped by config and run. Uploads
and downloads belong to the recipes, not scoring or validation. Both optional API and GPU Jobs recipes checkpoint predictions to the bucket;
the commands here use local result files.
Existing historical results remain in GitHub. The Space contains derived scores.

## Build and deploy the private Space

Upload the validated export as a private dataset, then use the resulting immutable
Hub commit. Download that commit into a fresh local directory and validate it before
building, so the local bytes correspond to the revision displayed on the page.

```bash
hf repos create small-models-for-glam/glam-extraction-benchmark --type dataset --private
hf upload small-models-for-glam/glam-extraction-benchmark ./exports/nls-v1 . --repo-type dataset
hf download small-models-for-glam/glam-extraction-benchmark --repo-type dataset \
  --revision COMMIT_SHA --local-dir ./exports/pinned
uv run glam_bench/validate_dataset.py ./exports/pinned --config nls-index-cards
uv run glam_bench/build_space.py ./exports/pinned --config nls-index-cards \
  --revision COMMIT_SHA --results ./results/nls --output ./exports/space \
  --legacy-results
hf repos create small-models-for-glam/glam-extraction-benchmark --type space --sdk static --private
hf upload small-models-for-glam/glam-extraction-benchmark ./exports/space . --repo-type space
```

Repository creation is a one-time step. Keep existing repositories private when
updating them. The builder requires a new output directory and includes `scores.json`,
the manifest, an example image/gold record and original inference provenance. It has
no server, secrets or inference calls. For future config-native runs, use
`--results ./results/<config>` and omit `--legacy-results`. Historical files keep
their original inference revision and harness provenance.

## Add another collection

1. Create a config named for the collection/task, such as `museum-accession-cards`.
   Keep institution and source dataset identity in its manifest.
2. Convert a pinned, permitted source to the same four required columns. Preserve
   source IDs, label/review provenance and image checksums in the optional provenance
   column. Store source schema/card alongside the export.
3. Supply a manifest following the NLS example: identity, licence, source revision,
   row/schema/data hashes, task instructions, label production, reporting slices and
   supported scoring policy. Define exact-match fields explicitly in the schema.
4. Add the config/split to the Hub card's `configs` list; validate and inspect gold
   against representative images before uploading a new private dataset revision.
5. Select models and inference settings, run with `--config`, validate submissions,
   then build that config's page from its pinned data and predictions.

The single-config builder renders one selected config. The collection builder below
links independent config pages through a dataset selector. Do not average scores across collections
without an explicit aggregation policy. HF benchmark registration remains separate:
[registering a benchmark](https://huggingface.co/docs/hub/main/en/eval-results#registering-a-benchmark).

## Harvard botany headers

`harvard-botany-headers` adds 30 scored cards from Harvard University Botany Libraries.
The nullable fields are `taxon_name`, `taxon_authority`, and `taxon_correction`.
Original printed headings are human-reviewed; both handwritten replacement strings
are human-confirmed. Correction absence was visually audited, with per-field review
provenance rather than claiming every null was individually human-reviewed.

Five unsupported cards from a separate random schema audit are retained in the
Hub `review` split with `schema_applicable: false`, a reason and no extraction gold.
The manifest points only to `test`, so the existing loader excludes the review split
without changing scoring or inference. Null field values mean absent, not unreadable.

The converter uses a reviewed annotation JSON, the corresponding source JPEGs, a
schema-audit directory (`findings.json`, `metadata/`, `images/`) and the pinned source
card. It writes a new config directory; it does not infer, upload, or update the Hub
card's config list. Image checksums must match the annotation provenance.

```bash
uv run glam_bench/export_harvard.py --review reviewed.json --images ./source-images \
  --audit ./schema-audit --source-card ./source-card.md --output ./exports/harvard-v1
uv run glam_bench/validate_dataset.py ./exports/harvard-v1 --config harvard-botany-headers
```

The review JSON declares `source_dataset`, `source_revision`, `target_schema` and
`items`. Each item supplies its stable `id`, source row/page/item URL, image checksum,
`expected_output`, `printed_fields_reviewed_by_human`, `correction_reviewed_by_human`,
`schema_applicable`, `scoring_eligible` and `review_status`. Only scoring-eligible
items enter `test`; pending labels remain drafts in `review`. Audit-only unsupported
cards have no expected output. Keep both splits declared explicitly in the Hub card.

Run new submissions with `--config harvard-botany-headers` and the new dataset commit.
NLS submissions keep their original config/revision; adding this config does not
require rerunning NLS. A leaderboard for Harvard still requires its own model runs.

## Build a Space with multiple configs

List the pinned local dataset snapshots and completed result folders in `runs.json`:

```json
[
  {"root": "./exports/nls-pinned", "config": "nls-index-cards",
   "revision": "NLS_COMMIT_SHA", "results": "./results/nls-completed"},
  {"root": "./exports/harvard-pinned", "config": "harvard-botany-headers",
   "revision": "HARVARD_COMMIT_SHA", "results": "./results/harvard-completed"}
]
```

Replace the revision placeholders with the full commits used for inference. Then:

```bash
uv run glam_bench/build_collection_space.py runs.json --output ./exports/space-both
```

The first config is the landing page; others have their own directories, scores,
examples and provenance. Each result must match its selected config and revision.
The selector also offers **Overall**: the equal-weight mean of each dataset’s mean
per-document F1 (currently 50% NLS and 50% Harvard). Models must have complete
results on every selected config to enter Overall; others remain on their dataset
pages. `overall-scores.json` records weights, component scores and source revisions.
Individual dataset metrics are unchanged; the overall score is not a pooled field-level F1.
Use only completed result files; retain interrupted checkpoints in the results bucket.
Upload the resulting directory to the existing private Space as before.
