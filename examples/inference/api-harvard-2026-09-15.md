# Harvard API batch: 2026-09-15

Dataset `small-models-for-glam/glam-extraction-benchmark`, config
`harvard-botany-headers`, split `test`, **30 cards**, pinned at
`48e92ea1112d3fc2a32467e875d64e10129c4d46`. The separate five-card review split is
not scored. This run does not alter or rerun the NLS config.

Only the public-source image, generic target schema and task instructions are
sent to the API; ground truth and review metadata remain local. The three fields
are printed taxon name, printed authority and an explicit handwritten heading
replacement. The model receives the config's instructions unchanged.

Same API lineup and generation settings as the NLS batch:

| Model | Provider | Max output tokens | Extra settings | Budget |
|---|---|---:|---|---:|
| Qwen3.5-9B | DeepInfra | 1,600 | `chat_template_kwargs.enable_thinking=false` | $0.15 |
| Qwen3.8-27B | Novita | 1,600 | `chat_template_kwargs.enable_thinking=false` | $0.45 |
| Qwen3-VL-235B | DeepInfra | 1,600 | none | $0.20 |
| GLM-5.3-Flash | DeepInfra | 4,096 | `reasoning_effort=low` | $0.15 |

Temperature 0 throughout, unconstrained JSON generation. All four two-card smoke
tests returned complete JSON without truncation before continuing. Smoke outputs
are retained when resuming, so successful cards are not inferred twice. GLM
reported zero reasoning tokens on both smoke cards.

Run from the checkout (set the model and its budget using the table):

```bash
uv run examples/inference/api_batch.py --model Qwen3.5-9B \
  --config harvard-botany-headers \
  --revision 48e92ea1112d3fc2a32467e875d64e10129c4d46 \
  --run-id 2026-09-15-v1 --limit 2 --budget-usd 0.15
uv run examples/inference/api_batch.py --model Qwen3.5-9B \
  --config harvard-botany-headers \
  --revision 48e92ea1112d3fc2a32467e875d64e10129c4d46 \
  --run-id 2026-09-15-v1 --resume --budget-usd 0.15
```

The actual commands used `/private/tmp/glam-contract-env/bin/python` from the
existing UV-created environment in place of `uv run`, and
`HF_HUB_DISABLE_PROGRESS_BARS=1`. Exact dependency versions and helper commit are
recorded in each submission. The executed recipe SHA-256 is
`b49d7283aa6619edacf6c6b4c2af5d69ac9452a6c9f4781334f4122ec142a424`, also recorded
per prediction. Its exact source is archived in the private bucket under
`harvard-botany-headers/2026-09-15-v1/recipes/api_batch.py`.

Local outputs: `results/harvard-botany-headers/2026-09-15-v1/`. Canonical outputs
and checkpoints: private bucket `small-models-for-glam/glam-extraction-results`,
prefix `harvard-botany-headers/2026-09-15-v1/`. Each card checkpoints locally;
bucket uploads occur every ten cards and at each pause/completion. Complete files
are `<model-label>.json`; partials must not be published as completed submissions.

The recipe remains optional, outside the harness. This update adds dataset,
config and revision CLI arguments while retaining NLS defaults. Submission JSON
is the integration point. API weight revisions are not attested; each row records
its requested route, response model, response ID, usage and observed Hub revision.

All four complete files passed the independent submission validator against the
pinned config. Each final file was verified in the private bucket. No Space
changes are made by this recipe.

A shared nonperfect prediction on `mhg795390874cardfile3:0217` illustrates why
literal transcription matters: the reviewed printed heading is **Utriculata
cornuta**, while Qwen3.5-9B, Qwen3.8-27B and GLM output **Utricularia cornuta**.
The pinned label is retained; the model outputs appear to normalize the unusual
printed name. This is a transcription evaluation, not taxonomic correction.

## Completed results

| Model | Cards | Mean content F1 | Input tokens | Output tokens | Estimated USD |
|---|---:|---:|---:|---:|---:|
| GLM-5.3-Flash | 30 | 0.998650 | 64,842 | 849 | $0.01015080 |
| Qwen3-VL-235B | 30 | 0.980000 | 51,135 | 1,136 | $0.01122668 |
| Qwen3.5-9B | 30 | 0.966307 | 39,384 | 1,255 | $0.00412665 |
| Qwen3.8-27B | 30 | 0.981107 | 51,345 | 1,329 | $0.02555190 |

Total estimated API cost: **$0.05105603**. All 120
predictions parsed as JSON under the scorer; none were truncated. No transport
errors or retries occurred. These are results on a curated 30-card POC, not an
estimate of performance over the complete Harvard collection.
