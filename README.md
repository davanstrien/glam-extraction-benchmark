# GLAM extraction benchmark

A small benchmark for structured extraction from galleries, libraries, archives
and museums. Models receive a document image and a JSON Schema; predictions are
scored against reviewed reference data.

[Leaderboard](https://huggingface.co/spaces/small-models-for-glam/glam-extraction-benchmark) ·
[Dataset](https://huggingface.co/datasets/small-models-for-glam/glam-extraction-benchmark) ·
[Raw predictions](https://huggingface.co/buckets/small-models-for-glam/glam-extraction-results)

**Experimental preview:** ten models evaluated on both datasets below.
The linked assets currently require access.

| Config | Scored cards | Task |
|---|---:|---|
| `nls-index-cards` | 98 | Manuscript-catalogue metadata and identifiers |
| `harvard-botany-headers` | 30 | Printed taxon name, authority and handwritten taxon corrections |

The leaderboard shows per-dataset scores and **Overall F1: 50% NLS + 50% Harvard**.
Size filters use total parameters; columns are sortable and filtered views can be
shared by URL. These small evaluation sets do not establish a general model ranking.

## Quick start

Use Python 3.11+ and [UV](https://docs.astral.sh/uv/), from this repo's root.
The optional reference harness is version **0.0.1**. Choose a model from
[`glam_bench/models.py`](glam_bench/models.py); this example needs your own
OpenAI-compatible endpoint serving `lift-9B`; this registry entry uses `HF_TOKEN` for authentication.

```bash
uv run glam_bench/harness.py --models lift-9B --config nls-index-cards \
  --dataset-revision ecc9c02582f933ca89b73e6751e4ed76888cb24d \
  --base-url lift-9B=https://your-endpoint.example/v1 \
  --max-tokens 1600 --limit 2
```

Inspect `results/nls-index-cards/lift-9B.limit2.json`, then omit `--limit 2`
to run all 98 cards. Add `--resume` to continue an interrupted full run.
Any inference tool can produce the [submission format](docs/SUBMISSION.md);
HF Jobs and our bucket are optional.

## Workflows and design

- [Design and scoring](docs/DESIGN.md): metrics, annotation provenance and limitations.
- [Add a dataset or build the Space](docs/DATASETS.md): config format, validation and export.
- [Submit predictions](docs/SUBMISSION.md): result format and validation commands.
- [Inference recipes](examples/inference/README.md): how the current API and GPU runs were made.

## Licence

Code and documentation: [MIT](LICENSE). Dataset content retains its source rights
and attribution: NLS is CC0; the Harvard source reports public-domain status.
See [sources and review provenance](docs/DESIGN.md#sources-and-review).
