# glam-extraction-benchmark

Working harness version: **0.0.1** (`uv run glam_bench/harness.py --version`).
The default dataset is `small-models-for-glam/glam-extraction-benchmark`, config
`nls-index-cards`, split `test`. See [Dataset workflow](docs/DATASETS.md) for export,
validation, adding collections and building the private static Space.

Minimal benchmark for **structured extraction from GLAM documents** — index cards + registration
forms. Given a card/form image **and a target JSON schema**, score how well a model extracts the
fielded data, across models.

> **Status: private preview.** The NLS config contains 98 manuscript-catalogue index cards
> from the public, CC0 `NationalLibraryOfScotland/index-cards-eval` dataset. Labels were
> drafted by Qwen3.6-35B-A3B and checked by NLS cataloguers: 66 accepted, 32 corrected.
> The preview reuses 12 historical runs, explicitly labelled as using the earlier schema.
> New runs preserve the source schema's nullability. The original 11-item POC dataset
> is private and has illustrative silver labels.

## Why

"There is no best model — it depends on your **cards**, your **budget**, and whether you can send
data to an **API** at all." A leaderboard makes that concrete for GLAM institutions, and pulls
AI model-builders toward GLAM data. The harness already shows a 4–8B model you can self-host landing
within a few points of a 235B frontier API on real archival cards.

## Layout

**Canonical schema = standard JSON Schema.** It's the lingua franca (OpenAI/Gemini structured
outputs, vLLM structured outputs, Pydantic, outlines), so each model's *solver* converts from it to its
own dialect (NuExtract template, `structured_outputs.json`, a prompt, or a Pydantic model). A `{"x-match":"exact"}`
annotation marks verbatim/identifier fields for the scorer. Items carry `target_schema` (JSON Schema)
and `expected_output`; adapters derive their own schema dialect.

| file | role |
|---|---|
| `glam_bench/schema.py` | **schema is first-class** — JSON Schema is canonical; converter to the NuExtract template, `to_guided_json` (vLLM/lift), and `field_match_type` (for the scorer) |
| `glam_bench/scorer.py` | model-agnostic **typed-KIE scorer** over JSON Schema — content-F1 + per-field-type breakdown (exact / free-text), abstention (false-populate rate), schema validity |
| `glam_bench/harness.py` | run selected registry models over a dataset; saves predictions + scores. Adapters: schema-native Spaces, general VLMs via the **HF Inference Providers router**, and **any OpenAI-compatible `/v1` endpoint** (`kind:"openai"` → a Jobs vLLM serve, local vLLM/TGI/Ollama, OpenAI, OpenRouter) |
| `glam_bench/result_io.py` | the result *file*: the **submission contract** (`validate_submission`), envelope shape, atomic writes, the three file states, the resume contract, and the publication gate every consumer discovers its input through |
| `glam_bench/validate_submission.py` | check one submission against that contract before sending it — the same checks the board runs |
| `glam_bench/models.py` | model registry (labels, kinds, ids — **no endpoints**), typed-core item list, POC dataset id |
| `glam_bench/dataset_contract.py` | generic config loading, schema/gold validation and evaluation identity |
| `glam_bench/build_space.py` | static Space build from a pinned config and cached predictions; performs no inference |
| `glam_bench/nls.py` | legacy NLS gold loader: dereferences the paratext schema, overlays `x-match:"exact"` on identifiers, pins a commit |
| `glam_bench/rescore.py` | rescore cached predictions against the current gold + scorer; prints drift vs the stored scores |
| `glam_bench/board.py` | the **two-axis board** (identifiers wrong / invented fields vs fields left blank), micro-averaged — the numbers on the page |
| `glam_bench/site_nls.py` | builds `site/index.html` — the published board — from those numbers |
| `glam_bench/leaderboard.py` | older single-F1 aggregator over any results dir; superseded by `board.py` for the NLS board |
| `results/` | per-model outputs. `results/nls/*.json` is the published board and is tracked in git; everything else is ignored |

## Submit results

A file in `results/` is a submission: a `model.label`, the commit sha of the gold snapshot
(`dataset.inference_revision`), and one row per item with the model's raw output as a string.
Config-based submissions also identify config and split. The board rescores every submission from those predictions; the harness
below is the reference producer, but any tool or script may produce the same shape.

Contract, optional keys and example: **[`docs/SUBMISSION.md`](docs/SUBMISSION.md)**.

```bash
uv run glam_bench/validate_submission.py results/nls-index-cards/my-extractor.json
```

Exit 0 prints a summary; exit 1 lists every problem. New config results go in `results/<config>/`, named `<model.label>.json`. Historical NLS
runs remain in `results/nls/`. Generated results are ignored until deliberately submitted.

Harness-written files board as attested. Everything else boards as self-reported, marked `†`.
The validator checks shape, ids and snapshot; it cannot check which model ran.

## Run a model

How the optional reference runner produces a submission using an OpenAI-compatible
endpoint (`lift-9B` here). `HF_TOKEN` reads the private benchmark dataset; endpoint
authentication depends on your hosting setup.

### 1. Choose an inference endpoint

The reference runner accepts an OpenAI-compatible endpoint via `--base-url`, or
uses HF Inference Providers for a router model. Hosting is your choice.
[Optional inference recipes](examples/inference/README.md) record our setup;
the [HF Jobs example](examples/inference/hf-jobs.md) is one way to serve a model.
Jobs and bucket access are not required to produce or validate a submission.

### 2. Smoke-test two items

```bash
uv run glam_bench/harness.py --models lift-9B --config nls-index-cards --max-tokens 1600 \
  --base-url lift-9B=https://your-endpoint.example/v1 --limit 2
```

Read the two predictions before paying for 98. `--limit` writes `lift-9B.limit2.json`, never the
canonical name.

### 3. Run it

```bash
uv run glam_bench/harness.py --models lift-9B --config nls-index-cards --max-tokens 1600 \
  --base-url lift-9B=https://your-endpoint.example/v1
```

- `--models` is required: registry labels, or `all` for every enabled entry.
- `--base-url LABEL=URL` is required for every `kind:"openai"` model (`models.py` holds no
  endpoints). On a router model it runs the model through the OpenAI-compatible adapter instead.
- `$GLAM_BASE_URL_<LABEL>` is the fallback (label uppercased, non-alphanumerics → `_`).
- `--provider LABEL=NAME` pins the Inference Providers provider. Left off, the router chooses and
  never says.
- `--no-thinking` sets `chat_template_kwargs.enable_thinking=false` for reasoning models.
- `--max-tokens 1600` for `nls-index-cards`; the 900 default truncates the entry lists.

Everything is resolved before the dataset loads, and the run prints how it will reach every model
before the first paid request.

### 4. What lands on disk

The default NLS config writes to `results/nls-index-cards/`; other configs use
`results/<config>/`. Explicit `--dataset nls` retains the historical `results/nls/` path.

| file | means |
|---|---|
| `<LABEL>.json.partial` | in progress; rewritten after every row, `complete` false |
| `<LABEL>.json` | finished; every item has a row, `complete` true |
| `<LABEL>.limit<N>.json` | a `--limit` smoke test |

The partial is promoted when every item has a row, then deleted. An interrupt at row 97 of 98
costs one row.

### 5. Recover a run

```bash
uv run glam_bench/harness.py --models lift-9B --config nls-index-cards --max-tokens 1600 \
  --base-url lift-9B=https://your-endpoint.example/v1 --resume
```

`--resume` continues the `.partial` if there is one, otherwise the `.json`. It refuses a file from
a different dataset snapshot, config, item slice, model, or generation setting (`--max-tokens`,
`--no-thinking`). The transport is not part of that identity, so `--resume` with `--base-url`
moves the failed rows onto a pinned serve; each row records which transport answered.

`--select` chooses which rows to redo: `errors` (default: transport failures, scorer crashes,
missing rows), `unparseable` (those plus rows the scorer could not parse), `all`.

### 6. Rescore and build the board

For the config-based private Space, use the pinned build in [Dataset workflow](docs/DATASETS.md).
The following commands retain the historical NLS loader and page:

```bash
uv run glam_bench/rescore.py           # drift vs the stored scores; --write to persist
uv run glam_bench/board.py             # the two-axis board
uv run glam_bench/site_nls.py          # writes site/index.html
```

Stored predictions can be rescored after a scorer or gold change. A changed input schema
or prompt requires a new run to measure its effect on model output. `board.py`
is what the page quotes: micro-averaged rates, `notes` field excluded. `rescore.py --write`
rewrites atomically, records `scoring.gold_revision`, and never drops a row (a prediction with no
gold row is kept, zeroed, and marked `scorer_error: "no gold for id"`).

### Result file format

Format version 2. The envelope says what was asked; every row says how the answer was obtained.
Historical NLS example with placeholders and rows omitted, so not itself a valid submission; the
complete recorded run is `results/nls/lift-9B.json`.

```json
{"format_version": 2,
 "model":   {"label": "lift-9B", "kind": "openai", "id": "datalab-to/lift", "params": "9B", "cost": "self-host",
             "guided": true},
 "dataset": {"id": "NationalLibraryOfScotland/index-cards-eval", "alias": "nls",
             "requested_revision": "main", "inference_revision": "<40-char sha>", "split": "train",
             "item_count": 98, "item_ids_sha256": "<sha256 of the id list>"},
 "generation": {"max_tokens": 1600, "temperature": 0, "guided": true, "request_options": {}},
 "scoring": {"scorer_version": "2026-08-26a"},
 "run":     {"started_at": "...Z", "finished_at": "...Z", "harness_git_sha": "...",
             "complete": true, "transport_errors": 0},
 "produced_by": {"producer": "harness", "attested": true},
 "items": [{"id": "Allan-W.-Anderson-D.__0221", "prediction": "{\"image_type\": ...}",
            "inference_status": "ok", "attempts": 1, "content_f1": 0.7414, "schema_valid": 1.0,
            "provenance": {"served_by": "openai-compatible", "endpoint": "https://your-endpoint.example/v1",
                           "endpoint_attested": true, "model_revision_hub_main": "<sha>", "timestamp": "...Z",
                           "latency_s": 2.625, "retry_wait_s": 0.0, "max_tokens": 1600, "temperature": 0,
                           "request_options": {}, "thinking_disabled": false,
                           "finish_reason": "stop", "harness_git_sha": "..."}}]}
```

- `model` carries identity only; `base_url` and `api_key_env` are never written. Where a row ran
  is `provenance.endpoint`, credentials and query strings stripped.
- `endpoint_attested` is false for `router:auto`: the router never says which provider served.
- `model_revision_hub_main` is what the model repo's `main` was during the run, not proof of the
  weights the endpoint loaded.
- On a resume, `run.started_at` and `run.harness_git_sha` stay as first written; `run.resumed_at`
  lists later sessions and each row's `provenance.harness_git_sha` says what produced it.

### Dataset snapshots

`--dataset-revision` (branch, tag or sha) is resolved once to a commit and pinned for the run,
so a run cannot straddle two versions of a dataset that is still accepting rows. The file
records `requested_revision` and `inference_revision`; `alias` records the CLI shorthand.

The inference revision and the scoring revision are separate: `rescore.py`, `board.py` and
`site_nls.py` load the gold at `main`, which may be newer than the commit the predictions were made
against. `rescore.py --write` records it as `scoring.gold_revision`.

### What gets retried

An adapter call that fails is retried up to `--max-attempts` times (default 3) with doubling
backoff from `--retry-backoff` (default 2s, capped at 30s, plus jitter), only when the failure is
transient.

- **Transient**: 408, 429, 500, 502, 503, 504, 529, and connection or timeout failures with no
  status.
- **Permanent**: other 4xx (authentication, invalid request, not found), gated or missing repos,
  URLs this client cannot speak. Never retried.
- **Unclassified**: anything else, including exceptions raised by the harness itself. Never
  retried; the row's `error` says what happened.

SDK-level retries are switched off, so `attempts` on a row is the number of requests actually
sent, across every run and resume.

An unparseable answer is a finding, not a failure: `inference_status: "ok"`, `schema_valid: 0.0`,
never retried. Only `--select unparseable` reruns it. A transport failure writes
`inference_status: "transport_error"`, `prediction: null`, zero scores, and `failure_class`.

### Publication rules

Every consumer finds its input through `result_io.discover_result_files`:

- `.partial` files are never seen; `*.limit<N>.json` smoke tests are skipped with a line saying so.
- A file that breaks the submission contract (`docs/SUBMISSION.md`) is refused, every problem
  listed. `--allow-incomplete` never covers this.
- A file whose run did not finish, or whose rows contain transport errors, is refused unless
  `--allow-incomplete` is passed (one warning per file).
- One `model.label` per file; `dataset.id` must match the gold the consumer loaded; every file on
  one board must share the same snapshot (`inference_revision`, `item_ids_sha256`, `item_count`).
- A file with no `dataset` block is legacy: it gets the id check and nothing else. Ends when the
  June–August files are rerun.

### Interrupting a run

Ctrl-c is safe: the checkpoint is written by atomic rename, so a reader sees either the previous
complete file or the new one, never a splice, and an interrupted resume is always a superset of
the file it resumed from. Then cancel the Job; billing stops with it, not with the harness:

```bash
hf jobs cancel <namespace>/<job-id>
```

## Data

- `NationalLibraryOfScotland/index-cards-eval` (public, CC0) — 98 manuscript-catalogue index cards,
  every label reviewed by NLS cataloguers: 66 `verified` (accepted as drafted) and 32 `corrected`
  (edited by a reviewer). The drafts came from Qwen3.6-35B-A3B, so the two subsets are reported
  separately; whether that gives its relatives an advantage is not something this split measures.
  The new config is the default; `--dataset nls` explicitly uses the historical loader.
- `davanstrien/glam-extraction-bench-poc` (private) — the original POC. 11 items: `image ·
  target_schema · silver_gold · item_quality · uncertain_fields · …`. Typed core (forms ×2, BPL ×2,
  Rubenstein, Peabody, Parisian) is good/usable; handwritten + multi-card items are deferred.

Discover more servable VLMs to add to the registry:

```bash
curl "https://huggingface.co/api/models?inference_provider=all&pipeline_tag=image-text-to-text"
```

## Roadmap

- [ ] More collections beyond NLS (LoC, BPL, Smithsonian); more document types beyond cards + forms
- [ ] Human-verify the POC silver gold → gold (2nd/3rd review)
- [ ] Confirm the 5 `other`-tagged source licenses before publishing the private POC dataset
- [ ] Inspect-AI scorer wrapper and a custom `evaluation_framework` registration (needs HF allow-list) for an HF "verified" path
- [ ] Cost/latency columns pulled from the router; per-card-type sub-leaderboards; a constrained-decoding track

## Caveats

On the POC set, silver gold is NuExtract-seeded → NuExtract is advantaged, and scores are
**illustrative** until human gold. On the NLS set, the drafts came from Qwen3.6-35B-A3B and every label was
reviewed by NLS cataloguers; the 66 `verified` and 32 `corrected` rows are reported separately
because of that drafting step, and any advantage to the drafting model's relatives is unverified. Some models tagged multimodal fail to serve images via
their provider — the harness records this rather than hiding it.
