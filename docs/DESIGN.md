# Design and scoring

The benchmark separates the dataset, inference and scoring. A collection contributes
images, a target JSON Schema, reviewed outputs and provenance. Any inference tool
can produce predictions in the shared [submission format](SUBMISSION.md). The
benchmark validates those files and recomputes their scores.

The reference harness and [API/Jobs recipes](../examples/inference/README.md) are
optional producers. Dataset validation and leaderboard builds perform no inference.

## Scores

**Dataset F1** is the mean of per-card extraction F1 scores, displayed on a 0–100
scale. Schema annotations select normalized exact matching for identifiers;
plain text receives partial credit using normalized character-sequence similarity
(`SequenceMatcher`). NLS excludes `notes` from scoring. The
[scorer](../glam_bench/scorer.py) and [board aggregation](../glam_bench/board.py)
implement the field matching and summary metrics.

**Overall F1** is the arithmetic mean of dataset F1 scores: currently 50% NLS and
50% Harvard, despite their different card counts. Only models evaluated on every
selected config enter Overall. This is not a pooled field-level F1.
`overall-scores.json` preserves component scores, weights and source revisions.

Dataset pages also report identifiers wrong, invented fields and fields left blank.
These distinguish incorrect populated values from values the model omitted. Lower
is better; a dash means the identifier metric does not apply. Harvard has no fields
marked as identifiers.

Parameter filters use total parameters, including all experts for MoE models.
The plot highlights the observed size/quality Pareto frontier. Parameter count does
not measure runtime memory, latency or cost. Model names and points link to Hub
repositories; dataset and filter choices are recorded in the URL.

## Sources and review

### NLS manuscript-catalogue cards

The `nls-index-cards` config contains 98 cards from
[NationalLibraryOfScotland/index-cards-eval](https://huggingface.co/datasets/NationalLibraryOfScotland/index-cards-eval),
under CC0. Labels were drafted by Qwen3.6-35B-A3B and reviewed by NLS cataloguers:
66 accepted as drafted and 32 corrected. Original review metadata is preserved.
The config retains the nullable source schema and marks identifiers for normalized
exact matching.

### Harvard botany headers

The `harvard-botany-headers` config contains 30 curated cards from
[biglam/index-cards-harvard-botany-metropolitan-flora](https://huggingface.co/datasets/biglam/index-cards-harvard-botany-metropolitan-flora).
The source reports public-domain status; Harvard Botany Libraries attribution is
preserved. All 30 printed headings and both handwritten replacement strings were
human-reviewed. Absence of a correction was assistant-audited, with per-field review
provenance retained rather than claiming every null was individually human-reviewed.

Extract the original printed taxon name and authority even when corrected. Record
an explicit handwritten replacement separately in `taxon_correction`. Do not expand
abbreviations or modernise taxonomy. Body handwriting is outside this task.
Null means absent, not unknown or illegible.

Five unsupported cards are retained in an unscored `review` split: four have
handwritten-only headings, and one has multiple heading interventions. Their
provenance records why they are excluded; they have no extraction gold. The manifest
selects only `test` for evaluation. See [the export workflow](DATASETS.md).

## Reproducibility

Runs identify dataset commit, config and split, alongside model and inference
settings. The working harness version is `0.0.1`; it is separate from the dataset
revision and scorer version. Adding a config does not invalidate runs on an unchanged
config and revision. Keep predictions made with different input schemas distinct;
rescoring old predictions does not measure the effect of a new prompt or schema.

Completed predictions and checkpoints live in the results bucket, grouped by config
and run ID. The [run notes](../examples/inference/README.md) record exact revisions,
settings and output locations. The Space contains derived scores, source metadata
and examples; it is a static display with no inference calls or runtime secrets.
Validation establishes file shape and dataset identity, not which weights a provider
actually served.

The 12 historical NLS runs in `results/nls/` and `site/index.html` belong to the earlier
board. They are separate from the current ten-model, two-config Space. The original
private 11-item POC is not part of Overall. Legacy loaders and board scripts remain
for inspecting those historical artifacts.

## Limits and future work

These are small evaluation sets. Harvard is a curated sample, not a representative
handwriting test. NLS labels began as model drafts before cataloguer review; this
benchmark does not measure any advantage that might give related models. Nearby
scores should not be read as a definitive model ranking.

More reviewed collections and document types would broaden coverage. Future
[Hub benchmark registration](https://huggingface.co/docs/hub/main/en/eval-results#registering-a-benchmark)
is possible but not implemented; the current dataset-config workflow does not
require it.
