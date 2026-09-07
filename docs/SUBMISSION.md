# Submission contract

A file in `results/` is a submission: a model's raw predictions for every item in one gold snapshot. The board rescores every submission from those predictions, so submitters never score their own runs. `glam_bench/harness.py` is the reference producer; any tool or script may produce the same shape.

## Rules

1. `model.label` MUST be the board name. One file per label, named `<label>.json`.
2. `dataset.inference_revision` MUST be the 40-character commit sha of the snapshot read. A branch name is refused; the validator prints the sha to use.
3. `items` MUST hold one row per `_sample_id` in that snapshot, each with `id` and `prediction` as a string: the model's raw output, JSON as text, unparsed.
4. `prediction` MAY be `null` only with `inference_status: "transport_error"` on that row.
5. Nothing else is required.

## Optional keys

| key | meaning | if present |
|---|---|---|
| `format_version` | `2` | must be `2` |
| `model.id` | Hub id, or free text | must be a string |
| `model.params` | shown beside the label | — |
| `dataset.id` | the Hub dataset | must match the board's dataset |
| `produced_by` | `{"producer", "attested", "contact"}` | `producer` is `harness`, `paratext`, `space` or `other` |
| `items[].provenance` | `served_by`, `endpoint`, `timestamp`, anything else you can state | — |

Absent `produced_by` means self-reported. `null` counts as absent everywhere.

## What the board does

It rescores every file from its predictions and refuses any file that breaks a rule, listing every problem. Every file on one board must come from one snapshot: `dataset.id`, `inference_revision`, `item_ids_sha256` and `item_count` are compared across the files that carry them. A file boards as **attested** only when `produced_by.producer` is `harness`, `attested` is `true`, and every row's `provenance` has `served_by`, `endpoint` and `timestamp`; everything else boards as **self-reported** (`†`). No check can tell which model ran, so the tier says who vouches.

## Fields the harness writes

Checked if present, never required.

| key | value |
|---|---|
| `dataset.item_count` | integer, equals the row count |
| `dataset.item_ids_sha256` | `sha256("\n".join(ids))`, ids in file order |
| `generation` | `max_tokens`, `temperature`, `guided`, `request_options` |
| `run.complete` | `false` marks an unfinished run; refused unless `--allow-incomplete` |
| `run.transport_errors` | integer, equals the number of `transport_error` rows |
| `run.model_revision_moved` | `true` when rows record two `model_revision_hub_main` values |

## Minimal example

```json
{"model": {"label": "my-extractor", "id": "acme/my-extractor", "params": "7B"},
 "dataset": {"id": "NationalLibraryOfScotland/index-cards-eval",
             "inference_revision": "9f1c0d4b2e7a5c83916d4fbb2c0e7a51d8b3c460"},
 "items": [{"id": "Allan-W.-Anderson-D.__0221", "prediction": "{\"image_type\": \"typed\"}"}]}
```

A real submission carries every item in the snapshot. Snapshot sha: `HfApi().dataset_info("NationalLibraryOfScotland/index-cards-eval").sha`; pass it as `revision` when loading the dataset.

## Validate

```bash
uv run glam_bench/validate_submission.py results/nls/my-extractor.json
```

Exit 0 prints a summary. Exit 1 lists every problem. Ids are checked against the snapshot in `dataset.inference_revision`; `--dataset-revision main` checks another. Only the NLS gold has a loader; another dataset gets the shape checks and `ids not checked`. Put the file in `results/nls/` for the NLS gold set, `results/` otherwise. The board build runs the same checks.
