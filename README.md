# glam-extraction-benchmark

Minimal benchmark for **structured extraction from GLAM documents** — index cards + registration
forms. Given a card/form image **and a target JSON schema**, score how well a model extracts the
fielded data, across models.

> **Status: v1 on institution-verified gold.** The published board runs on
> `NationalLibraryOfScotland/index-cards-eval` (public, CC0): 98 manuscript-catalogue index cards.
> Labels were drafted by Qwen3.6-35B-A3B and checked by NLS cataloguers: 66 accepted as drafted,
> 32 corrected. The original POC set (`--dataset` default) is 11
> items with *silver* labels (NuExtract-3 zero-shot + one LLM reviewer agent) in a public HF
> dataset; its scores stay illustrative.

## Why

"There is no best model — it depends on your **cards**, your **budget**, and whether you can send
data to an **API** at all." A leaderboard makes that concrete for GLAM institutions, and pulls
AI model-builders toward GLAM data. The harness already shows a 4–8B model you can self-host landing
within a few points of a 235B frontier API on real archival cards.

## Layout

**Canonical schema = standard JSON Schema.** It's the lingua franca (OpenAI/Gemini structured
outputs, vLLM `guided_json`, Pydantic, outlines), so each model's *solver* converts from it to its
own dialect (NuExtract template, `guided_json`, a prompt, or a Pydantic model). A `{"x-match":"exact"}`
annotation marks verbatim/identifier fields for the scorer. Items carry `target_schema` (JSON Schema)
plus a derived `target_schema_nuextract`.

| file | role |
|---|---|
| `glam_bench/schema.py` | **schema is first-class** — JSON Schema is canonical; converter to the NuExtract template, `to_guided_json` (vLLM/lift), and `field_match_type` (for the scorer) |
| `glam_bench/scorer.py` | model-agnostic **typed-KIE scorer** over JSON Schema — content-F1 + per-field-type breakdown (exact / free-text), abstention (false-populate rate), schema validity |
| `glam_bench/harness.py` | run selected registry models over a dataset; saves predictions + scores. Adapters: schema-native Spaces, general VLMs via the **HF Inference Providers router**, and **any OpenAI-compatible `/v1` endpoint** (`kind:"openai"` → a Jobs vLLM serve, local vLLM/TGI/Ollama, OpenAI, OpenRouter) |
| `glam_bench/result_io.py` | the result *file*: the **submission contract** (`validate_submission`), envelope shape, atomic writes, the three file states, the resume contract, and the publication gate every consumer discovers its input through |
| `glam_bench/validate_submission.py` | check one submission against that contract before sending it — the same checks the board runs |
| `glam_bench/models.py` | model registry (labels, kinds, ids — **no endpoints**), typed-core item list, POC dataset id |
| `glam_bench/nls.py` | NLS gold loader: dereferences the paratext schema, overlays `x-match:"exact"` on identifiers, pins a commit |
| `glam_bench/rescore.py` | rescore cached predictions against the current gold + scorer; prints drift vs the stored scores |
| `glam_bench/board.py` | the **two-axis board** (identifiers wrong / invented fields vs fields left blank), micro-averaged — the numbers on the page |
| `glam_bench/site_nls.py` | builds `site/index.html` — the published board — from those numbers |
| `glam_bench/leaderboard.py` | older single-F1 aggregator over any results dir; superseded by `board.py` for the NLS board |
| `results/` | per-model outputs. `results/nls/*.json` is the published board and is tracked in git; everything else is ignored |

## Submit results

A file in `results/` is a submission: a `model.label`, the commit sha of the gold snapshot
(`dataset.inference_revision`), and one row per item with the model's raw output as a string.
Nothing else is required. The board rescores every submission from those predictions; the harness
below is the reference producer, but any tool or script may produce the same shape.

Contract, optional keys and example: **[`docs/SUBMISSION.md`](docs/SUBMISSION.md)**.

```bash
uv run glam_bench/validate_submission.py results/nls/my-extractor.json
```

Exit 0 prints a summary; exit 1 lists every problem. The file goes in `results/nls/` for the NLS
gold set (`results/` otherwise), named `<model.label>.json`.

Harness-written files board as attested. Everything else boards as self-reported, marked `†`.
The validator checks shape, ids and snapshot; it cannot check which model ran.

## Run a model

How the harness produces a submission, end to end, for a model not on Inference Providers
(`lift-9B` here; `GLM-OCR` is in the registry but disabled, see `models.py` for why).
`HF_TOKEN` is needed throughout: it reads the dataset and is the API key a Jobs serve expects.

### 1. Serve it on Jobs

```bash
hf jobs run --detach --expose 8000 --flavor a100-large -s HF_TOKEN \
  vllm/vllm-openai vllm serve datalab-to/lift --max-model-len 32768
```

Flavor by model size: up to ~8B `a10g-large`, 9–30B `a100-large`, larger see `hf jobs hardware`.
Keep `--max-model-len` at 16384–32768 when sending images; the 8192 default truncates. The job
is ready when `hf jobs logs -f <namespace>/<job-id>` shows `Application startup complete` (logs
are empty while it is still scheduling; `hf jobs inspect` shows the state). The job exposes
`https://<job-id>--8000.hf.jobs`; the OpenAI base URL is that + `/v1`. It bills per minute until
cancelled.

### 2. Smoke-test two items

```bash
uv run glam_bench/harness.py --models lift-9B --dataset nls --max-tokens 1600 \
  --base-url lift-9B=https://<job-id>--8000.hf.jobs/v1 --limit 2
```

Read the two predictions before paying for 98. `--limit` writes `lift-9B.limit2.json`, never the
canonical name.

### 3. Run it

```bash
uv run glam_bench/harness.py --models lift-9B --dataset nls --max-tokens 1600 \
  --base-url lift-9B=https://<job-id>--8000.hf.jobs/v1
```

- `--models` is required: registry labels, or `all` for every enabled entry.
- `--base-url LABEL=URL` is required for every `kind:"openai"` model (`models.py` holds no
  endpoints). On a router model it runs the model through the OpenAI-compatible adapter instead.
- `$GLAM_BASE_URL_<LABEL>` is the fallback (label uppercased, non-alphanumerics → `_`).
- `--provider LABEL=NAME` pins the Inference Providers provider. Left off, the router chooses and
  never says.
- `--no-thinking` sets `chat_template_kwargs.enable_thinking=false` for reasoning models.
- `--max-tokens 1600` for `nls`; the 900 default truncates the entry lists.

Everything is resolved before the dataset loads, and the run prints how it will reach every model
before the first paid request.

### 4. What lands on disk

`--dataset nls` writes to `results/nls/`, everything else to `results/`.

| file | means |
|---|---|
| `<LABEL>.json.partial` | in progress; rewritten after every row, `complete` false |
| `<LABEL>.json` | finished; every item has a row, `complete` true |
| `<LABEL>.limit<N>.json` | a `--limit` smoke test |

The partial is promoted when every item has a row, then deleted. An interrupt at row 97 of 98
costs one row.

### 5. Recover a run

```bash
uv run glam_bench/harness.py --models lift-9B --dataset nls --max-tokens 1600 \
  --base-url lift-9B=https://<job-id>--8000.hf.jobs/v1 --resume
```

`--resume` continues the `.partial` if there is one, otherwise the `.json`. It refuses a file from
a different dataset snapshot, item slice, model, or generation setting (`--max-tokens`,
`--no-thinking`). The transport is not part of that identity, so `--resume` with `--base-url`
moves the failed rows onto a pinned serve; each row records which transport answered.

`--select` chooses which rows to redo: `errors` (default: transport failures, scorer crashes,
missing rows), `unparseable` (those plus rows the scorer could not parse), `all`.

### 6. Rescore and build the board

```bash
uv run glam_bench/rescore.py           # drift vs the stored scores; --write to persist
uv run glam_bench/board.py             # the two-axis board
uv run glam_bench/site_nls.py          # writes site/index.html
```

Predictions are stored, so a scorer, schema or gold change is a rescore, not a rerun. `board.py`
is what the page quotes: micro-averaged rates, `notes` field excluded. `rescore.py --write`
rewrites atomically, records `scoring.gold_revision`, and never drops a row (a prediction with no
gold row is kept, zeroed, and marked `scorer_error: "no gold for id"`).

### Result file format

Format version 2. The envelope says what was asked; every row says how the answer was obtained.
Abbreviated example with placeholders and rows omitted, so not itself a valid submission; the
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
            "provenance": {"served_by": "openai-compatible", "endpoint": "https://<job-id>--8000.hf.jobs/v1",
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
  Run it with `--dataset nls`.
- `davanstrien/glam-extraction-bench-poc` (public) — the original POC. 11 items: `image ·
  target_schema · silver_gold · item_quality · uncertain_fields · …`. Typed core (forms ×2, BPL ×2,
  Rubenstein, Peabody, Parisian) is good/usable; handwritten + multi-card items are deferred.

Discover more servable VLMs to add to the registry:

```bash
curl "https://huggingface.co/api/models?inference_provider=all&pipeline_tag=image-text-to-text"
```

## Roadmap

- [ ] More collections beyond NLS (LoC, BPL, Smithsonian); more document types beyond cards + forms
- [ ] Human-verify the POC silver gold → gold (2nd/3rd review)
- [ ] Confirm the 5 `other`-tagged source licenses in the (already public) POC dataset
- [ ] Inspect-AI scorer wrapper and a custom `evaluation_framework` registration (needs HF allow-list) for an HF "verified" path
- [ ] Cost/latency columns pulled from the router; per-card-type sub-leaderboards; a constrained-decoding track

## Caveats

On the POC set, silver gold is NuExtract-seeded → NuExtract is advantaged, and scores are
**illustrative** until human gold. On the NLS set, the drafts came from Qwen3.6-35B-A3B and every label was
reviewed by NLS cataloguers; the 66 `verified` and 32 `corrected` rows are reported separately
because of that drafting step, and any advantage to the drafting model's relatives is unverified. Some models tagged multimodal fail to serve images via
their provider — the harness records this rather than hiding it.
