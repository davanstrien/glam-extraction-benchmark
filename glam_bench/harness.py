# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=5.0,<6",
#   "gradio_client>=2.6,<3",
#   "huggingface_hub>=1.29,<2",
#   "openai>=3.6,<4",
#   "pillow>=12.3,<13",
#   "jsonschema>=4,<5",
# ]
# ///
# Bounded to the verified major, not pinned. Both ends matter: openai 3.6's `max_retries` default
# (switched off in `run_openai`), huggingface_hub 1.29's `chat_completion(extra_body=...)`, and the
# exception class names `classify_failure` matches on.
"""Run models over the GLAM extraction benchmark dataset and score them.

Each model gets the same image and target schema. Predictions are stored and rescored from disk.
A run checkpoints to `<LABEL>.json.partial` after every row and promotes to `<LABEL>.json` when
every item has a row; `--resume` continues one. Result files are format-2 submissions (README,
"Result file format"). Provider SDKs are imported inside the adapters, so this module imports
with the standard library only.

Usage:
    uv run glam_bench/harness.py --models all          # every ENABLED registry model
    uv run glam_bench/harness.py --models NuExtract-3,Qwen3.5-9B
    uv run glam_bench/harness.py --models Qwen3.8-27B --resume       # retry failed rows only
    uv run glam_bench/harness.py --models Qwen3.8-27B --max-attempts 5   # flaky endpoint
    uv run glam_bench/harness.py --models lift-9B \
        --base-url lift-9B=https://<job-id>--8000.hf.jobs/v1
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import result_io
from models import MODELS, TYPED_CORE
from version import __version__
from dataset_contract import BENCHMARK_ID, evaluation_identity, inference_items, load_hub_config
from scorer import SCORER_VERSION, score
from schema import field_notes, jsonschema_to_nuextract, to_guided_json

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
FORMAT_VERSION = 2
SPLIT = "train"
MAX_TOKENS = 900
TEMPERATURE = 0

# Persisted into every result file. Deliberately excludes `base_url` and `api_key_env`: a result
# file is a published artifact, and an endpoint URL is a run detail (and a credential-shaped one)
# rather than model identity. Where the model ran is recorded per row, in `provenance.endpoint`.
MODEL_PUBLIC_KEYS = ("label", "kind", "id", "params", "cost", "guided", "provider")

# Kinds whose `spec["id"]` names a Hub *model* repo. A Space kind's id names the Space, so a
# model_info lookup there would resolve a different repo (or nothing) and read as an attestation.
MODEL_ID_KINDS = ("router_vlm", "openai")

# What a row scores when there is nothing to score: a transport failure, or a scorer crash.
# Consumers treat every other score key as 0 when missing (see leaderboard.aggregate).
ZERO_SCORES = {"content_f1": 0.0, "schema_valid": 0.0}

# `target_schema` is canonical JSON Schema; each adapter converts to its model's dialect.
VLM_PROMPT = (
    "Extract this archival card/form into JSON matching EXACTLY the given JSON Schema (same keys "
    "and structure). A string with \"x-match\":\"exact\" must be copied verbatim; \"format\":"
    "\"date\" -> ISO YYYY-MM-DD; an enum value must be one of those listed; arrays are repeatable. "
    "Output ONLY JSON. Omit or null any field not present on the document.\nJSON SCHEMA:\n{schema}"
)


@dataclass
class AdapterResponse:
    """What an adapter can say about one call, beyond the content itself."""

    content: str | None
    served_by: str
    endpoint: str
    finish_reason: str | None = None
    endpoint_attested: bool = True
    request_options: dict = field(default_factory=dict)


def public_endpoint(base_url: str) -> str:
    """A base URL with userinfo, query string and fragment removed."""
    parts = urlsplit(base_url)
    netloc = parts.hostname or ""
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def adapter_name(spec: dict) -> str:
    """Which adapter actually runs this spec."""
    return spec.get("adapter") or spec["kind"]


def describe_endpoint(spec: dict) -> tuple[str, str, bool]:
    """-> (served_by, endpoint, endpoint_attested) for a resolved spec."""
    kind = adapter_name(spec)
    if kind == "router_vlm":
        provider = spec.get("provider") or "auto"
        return "hf-inference-providers", f"router:{provider}", provider != "auto"
    if kind == "openai":
        return "openai-compatible", public_endpoint(spec["base_url"]), True
    if kind in ("nuextract_space", "specialist_space"):
        return "hf-space", f"space:{spec['id']}", True
    if kind == "fake":
        return "fake-adapter", f"fake:{spec['label']}", False
    return "unknown", f"{kind}:{spec.get('id')}", False


def request_options_for(spec: dict) -> dict:
    """The extra request body this spec asks every call to carry, `{}` when it asks for none."""
    options = spec.get("request_options")
    if not isinstance(options, dict):
        return {}
    return options


def thinking_disabled(request_options: dict) -> bool:
    """Whether the request asked the chat template to turn reasoning off. This is what was ASKED FOR,
    not proof it was honoured: the vLLM-standard `chat_template_kwargs.enable_thinking` is applied
    by the serving template, and a router that forwards the body to a third-party provider cannot
    promise the provider read it.
    """
    kwargs = request_options.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        return False
    return kwargs.get("enable_thinking") is False


def endpoint_response(spec: dict, content, finish_reason: str | None) -> AdapterResponse:
    """An AdapterResponse with the endpoint and request-option fields filled in from the spec."""
    served_by, endpoint, endpoint_attested = describe_endpoint(spec)
    return AdapterResponse(content=content, served_by=served_by, endpoint=endpoint,
                           finish_reason=finish_reason, endpoint_attested=endpoint_attested,
                           request_options=request_options_for(spec))


def first_choice(response, spec: dict):
    """The one choice an OpenAI-shaped response is expected to carry."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise EmptyResponseError(
            f"{spec['label']}: the endpoint returned a response with no choices — nothing to "
            "read. The call itself succeeded, so this is the answer's shape, not the transport: "
            "a filtered completion, or a request the server accepted and could not fulfil.")
    return choices[0]


def _png_bytes(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def _temp_jpeg(img) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(_png_bytes(img))
        return f.name


def run_nuextract_space(img, schema, spec) -> AdapterResponse:
    # gradio_client 2.6 has no retry of its own (only queue polling), so nothing to disable here
    # or in run_specialist_space: `run_item` is the only retry loop.
    from gradio_client import Client, handle_file

    # solver: canonical JSON Schema -> NuExtract template dialect; template is pure structure,
    # so field descriptions ride in the instruction (parity with schema-in-prompt models)
    from dataset_contract import scoring_schema

    schema_obj = scoring_schema(json.loads(schema))
    template = json.dumps(jsonschema_to_nuextract(schema_obj), ensure_ascii=False)
    instruction = "Extract the information present; copy identifiers verbatim; leave blanks out."
    if spec.get("task_instructions"):
        instruction += " " + spec["task_instructions"]
    notes = field_notes(schema_obj)
    if notes:
        instruction += " Field notes:\n" + "\n".join(f"- {n}" for n in notes)
    path = _temp_jpeg(img)
    o = Client(spec["id"]).predict(
        instruction_val=instruction,
        template_val=template, context_text_val="", context_image_val=handle_file(path),
        temperature_val=0.0, reasoning_val=False, api_name="/on_extract_click")
    content = o[1]["value"] if isinstance(o[1], dict) else o[1]
    # A Space exposes no finish reason, so truncation there stays invisible.
    return endpoint_response(spec, content, None)


def run_specialist_space(img, schema, spec) -> AdapterResponse:
    if spec.get("task_instructions"):
        raise ValueError("specialist_space cannot accept config task instructions; use a supported adapter")
    from gradio_client import Client, handle_file

    path = _temp_jpeg(img)
    o = Client(spec["id"]).predict(handle_file(path), schema, api_name="/extract")
    if isinstance(o, dict):
        content = o.get("raw") or json.dumps(o.get("parsed", {}))
    else:
        content = str(o)
    return endpoint_response(spec, content, None)


def _vlm_messages(img, schema, task_instructions=""):
    b64 = base64.b64encode(_png_bytes(img)).decode()
    return [{"role": "user", "content": [
        {"type": "text", "text": (task_instructions + "\n" if task_instructions else "") + VLM_PROMPT.format(schema=schema)},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]


def run_router_vlm(img, schema, spec) -> AdapterResponse:
    """A VLM via the HF Inference Providers router (provider auto-selected or pinned)."""
    from huggingface_hub import InferenceClient

    # No retry to disable: huggingface_hub 1.29 does not retry on the inference path (its
    # `http_backoff` helper is used only for file and repo transfers), and InferenceClient takes
    # no max_retries. `run_item` is the only retry loop here.
    client = InferenceClient(provider=spec.get("provider") or "auto")
    # `chat_completion` DOES take extra_body in huggingface_hub 1.29 (checked against the installed
    # signature), so a router spec can carry request_options like any other. Sent only when there
    # is something to send, so an ordinary router call is byte-for-byte what it was before.
    extra = {}
    options = request_options_for(spec)
    if options:
        extra["extra_body"] = options
    r = client.chat_completion(messages=_vlm_messages(img, schema, spec.get("task_instructions", "")), model=spec["id"],
                               max_tokens=MAX_TOKENS, temperature=TEMPERATURE, **extra)
    choice = first_choice(r, spec)
    return endpoint_response(spec, choice.message.content, getattr(choice, "finish_reason", None))


def openai_extra_body(spec, schema) -> dict:
    """Everything `run_openai` sends as `extra_body`: the spec's request options, plus guided
    decoding when the spec asks for it. `structured_outputs` is the member NOT echoed into
    `provenance.request_options`: it is derived per item from that item's `target_schema` rather
    than chosen for the run, and the fact that guided decoding was on is already published in
    `model.guided`.
    """
    body = dict(request_options_for(spec))
    if spec.get("guided"):  # current vLLM structured-output request
        body["structured_outputs"] = {"json": to_guided_json(json.loads(schema))}
    return body


def run_openai(img, schema, spec) -> AdapterResponse:
    """ANY OpenAI-compatible /v1 endpoint: local vLLM/TGI/Ollama, OpenAI, OpenRouter, a provider
    direct. Resolved spec: {"kind": "openai", "id": <model>, "base_url": <.../v1>, "api_key_env":
    <ENV>}.
    """
    from openai import OpenAI

    # max_retries=0 turns off the SDK's own retrying (openai 3.6 defaults to 2, so one call
    # could become three). `run_item` owns retry, and a row that says "3 attempts" has to mean
    # three requests were sent — not nine.
    client = OpenAI(base_url=spec["base_url"],
                    api_key=os.environ.get(spec.get("api_key_env", "OPENAI_API_KEY"), "EMPTY"),
                    max_retries=0)
    extra = {}
    body = openai_extra_body(spec, schema)
    if body:
        extra["extra_body"] = body
    r = client.chat.completions.create(model=spec["id"], messages=_vlm_messages(img, schema, spec.get("task_instructions", "")),
                                       max_tokens=MAX_TOKENS, temperature=TEMPERATURE, **extra)
    choice = first_choice(r, spec)
    return endpoint_response(spec, choice.message.content, getattr(choice, "finish_reason", None))


def run_fake(_img, _schema, spec) -> AdapterResponse:
    """Test-only adapter."""
    if spec.get("raises"):
        raise RuntimeError(spec["raises"])
    return endpoint_response(spec, spec.get("canned"), spec.get("finish_reason"))


ADAPTERS = {"nuextract_space": run_nuextract_space, "specialist_space": run_specialist_space,
            "router_vlm": run_router_vlm, "openai": run_openai, "fake": run_fake}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def harness_git_sha() -> str | None:
    """This repo's HEAD, or None when git cannot answer (no git, no checkout, a tarball)."""
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def looks_like_hub_repo_id(value) -> bool:
    if not isinstance(value, str) or value.count("/") != 1:
        return False
    owner, name = value.split("/")
    if not owner or not name:
        return False
    return not any(ch.isspace() for ch in value)


def resolve_model_revision(spec: dict) -> str | None:
    """The model repo's Hub `main` commit, read once per model at run time. NOT an attestation of the
    weights the endpoint loaded — a router, a Space or a self-served vLLM can be serving anything,
    and none of them will say. Hence the key name `model_revision_hub_main`.
    """
    if spec.get("kind") not in MODEL_ID_KINDS:
        return None
    if not looks_like_hub_repo_id(spec.get("id")):
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(spec["id"]).sha
    except Exception:  # noqa: BLE001 — no provenance is better than wrong provenance
        return None


# --- failure classification -------------------------------------------------------------------
# Every provider SDK is imported lazily inside its adapter, so classification cannot use
# `isinstance` against SDK exception classes — they may not be importable at all.

# Retried. 529 is several routers' spelling of 503. 409 is deliberately absent: on an inference
# call it is not a lock conflict, so it falls through to permanent with the other 4xx.
TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504, 529})

# Not retried. Any status outside both sets (2xx, 3xx, a 5xx not in the retry list) is
# `unclassified`, not permanent.
PERMANENT_STATUS = range(400, 500)

# Retried; no HTTP status. Matched by class name across the MRO: the SDK classes cannot be
# imported here, and openai 3.6 raises `httpx2.*` while huggingface_hub 1.29 and gradio_client 2.6
# raise `httpx.*`, so no single isinstance check spans them.
TRANSIENT_EXCEPTION_NAMES = frozenset({
    # openai
    "APIConnectionError", "APITimeoutError",
    # huggingface_hub
    "InferenceTimeoutError", "OverloadedError",
    # gradio_client
    "QueueError", "TooManyRequestsError",
    # httpx / httpx2. httpx.ConnectError is NOT an OSError and httpx.ReadTimeout is NOT a builtin
    # TimeoutError, so the isinstance fallback below never sees them.
    "TransportError", "TimeoutException", "NetworkError", "ProxyError",
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
    "ReadError", "WriteError", "RemoteProtocolError",
    # requests (shadows the builtin names)
    "ConnectionError", "Timeout", "ChunkedEncodingError",
})

# Not retried. Checked BEFORE the transient names and the builtins: huggingface_hub's
# OfflineModeIsEnabled is a builtin ConnectionError, and every HfHubHTTPError is an OSError.
PERMANENT_EXCEPTION_NAMES = frozenset({
    # huggingface_hub
    "BadRequestError", "RepositoryNotFoundError", "GatedRepoError", "DisabledRepoError",
    "RevisionNotFoundError", "EntryNotFoundError", "OfflineModeIsEnabled", "HFValidationError",
    # gradio_client. AppError may wrap something transient inside the Space; treated as permanent.
    "AuthenticationError", "ValidationError", "AppError", "InvalidAPIEndpointError",
    "SerializationSetupError",
    # httpx / requests: a URL or protocol this client will never be able to speak
    "LocalProtocolError", "UnsupportedProtocol", "InvalidURL", "MissingSchema", "InvalidSchema",
})


def http_status(exc) -> int | None:
    """The HTTP status an exception carries, or None when it carries none."""
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(exc, "response", None)
    attached = getattr(response, "status_code", None)
    if isinstance(attached, int):
        return attached
    return None


def exception_names(exc) -> set[str]:
    """Every class name on the exception's MRO, so a subclass matches the rule for its base."""
    return {cls.__name__ for cls in type(exc).__mro__}


class EmptyResponseError(Exception):
    """An OpenAI-shaped endpoint answered HTTP 200 and sent no completion."""


def classify_failure(exc) -> str:
    """-> "transient" (worth calling again), "permanent" (never call again), or "unclassified". A
    status code is the most specific thing an exception can say, so it decides alone and returns:
    a 404 must not be rescued further down by the fact that HfHubHTTPError subclasses OSError.
    """
    status = http_status(exc)
    if status is not None:
        if status in TRANSIENT_STATUS_CODES:
            return "transient"
        if status in PERMANENT_STATUS:
            return "permanent"
        return "unclassified"

    names = exception_names(exc)
    if names & PERMANENT_EXCEPTION_NAMES:
        return "permanent"
    if names & TRANSIENT_EXCEPTION_NAMES:
        return "transient"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "transient"
    return "unclassified"


# Failure classes the retry loop stops on immediately. An unretried transient failure costs one
# rerun with `--resume`; a retried one of these spends the quota three times over on a call that
# was never going to answer differently.
NEVER_RETRIED = frozenset({"permanent", "unclassified"})


# --- retry policy -----------------------------------------------------------------------------

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_S = 2.0
MAX_RETRY_WAIT_S = 30.0
RETRY_JITTER = 0.1

# Indirected so tests can replace it and run instantly: `monkeypatch.setattr(harness, "sleep", ...)`.
sleep = time.sleep


@dataclass
class RetryPolicy:
    """How many times one item may be called, and how long to wait between calls."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    backoff_s: float = DEFAULT_RETRY_BACKOFF_S
    max_wait_s: float = MAX_RETRY_WAIT_S

    def wait_after(self, failed_attempt: int) -> float:
        """Seconds to sleep after attempt `failed_attempt` failed: 2s, 4s, 8s ... capped."""
        step = self.backoff_s * (2 ** (failed_attempt - 1))
        capped = min(step, self.max_wait_s)
        return capped + random.uniform(0.0, capped * RETRY_JITTER)


def build_retry_policy(max_attempts: int, backoff_s: float) -> RetryPolicy:
    """Validate the retry switches before anything is loaded or any endpoint is touched."""
    if max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1 (1 means one call, no retry)")
    if backoff_s < 0:
        raise SystemExit("--retry-backoff cannot be negative")
    return RetryPolicy(max_attempts=max_attempts, backoff_s=backoff_s)


@dataclass
class AttemptLog:
    """What the retry loop can say about the calls it made for one item."""

    attempts: int
    latency_s: float
    retry_wait_s: float = 0.0


def build_provenance(response: AdapterResponse, model_revision, log: AttemptLog, git_sha) -> dict:
    """The per-row record of how this answer was obtained."""
    options = dict(response.request_options)
    return {
        "served_by": response.served_by,
        "endpoint": response.endpoint,
        "endpoint_attested": response.endpoint_attested,
        "model_revision_hub_main": model_revision,
        "timestamp": utc_now(),
        "latency_s": round(log.latency_s, 3),
        "retry_wait_s": round(log.retry_wait_s, 3),
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "request_options": options,
        "thinking_disabled": thinking_disabled(options),
        "finish_reason": response.finish_reason,
        "harness_git_sha": git_sha,
        "harness_version": __version__,
    }


def transport_error_row(item, spec, exc, log: AttemptLog, failure_class,
                        model_revision, git_sha) -> dict:
    """The adapter raised: nothing came back, so there is no model result to judge."""
    response = endpoint_response(spec, None, None)
    row = {"id": item["id"], "prediction": None, "inference_status": "transport_error",
           "attempts": log.attempts, "failure_class": failure_class,
           "error": f"{type(exc).__name__}: {exc}"[:200]}
    row.update(ZERO_SCORES)
    row["provenance"] = build_provenance(response, model_revision, log, git_sha)
    return row


def ok_row(item, response: AdapterResponse, log: AttemptLog, model_revision, git_sha) -> dict:
    """The model answered."""
    content = "" if response.content is None else response.content
    row = {"id": item["id"], "prediction": content,
           "inference_status": "ok", "attempts": log.attempts}
    try:
        scores = score(item["gold"], content, item.get("scoring_schema", item["target_schema"]),
                       exclude=item.get("scoring_exclude", ()))
    except Exception as exc:  # noqa: BLE001
        # A scorer crash is not an inference failure. Keep the prediction and the "ok" status,
        # and record the crash under its own key so it cannot be read as a transport error.
        row.update(ZERO_SCORES)
        row["scorer_error"] = f"{type(exc).__name__}: {exc}"[:200]
    else:
        row.update(scores)
    row["provenance"] = build_provenance(response, model_revision, log, git_sha)
    return row


def run_item(item, spec, model_revision, git_sha, retry: RetryPolicy) -> dict:
    """Call the adapter for one item, retrying only what is worth retrying. KeyboardInterrupt is a
    BaseException and is deliberately not caught, so ctrl-c still stops the run at once.
    """
    adapter = ADAPTERS[adapter_name(spec)]
    retry_wait_s = 0.0
    for attempt in range(1, retry.max_attempts + 1):
        started = time.monotonic()
        try:
            response = adapter(item["image"], item["target_schema"], spec)
        except Exception as exc:  # noqa: BLE001 — anything raised out of an adapter is transport
            log = AttemptLog(attempts=attempt, latency_s=time.monotonic() - started,
                             retry_wait_s=retry_wait_s)
            failure_class = classify_failure(exc)
            if failure_class in NEVER_RETRIED or attempt == retry.max_attempts:
                return transport_error_row(item, spec, exc, log, failure_class,
                                           model_revision, git_sha)
            wait_s = retry.wait_after(attempt)
            print(f"  {spec['label']} {item['id']} attempt {attempt}/{retry.max_attempts}: "
                  f"{type(exc).__name__} — retrying in {wait_s:.1f}s", flush=True)
            sleep(wait_s)
            retry_wait_s += wait_s
        else:
            log = AttemptLog(attempts=attempt, latency_s=time.monotonic() - started,
                             retry_wait_s=retry_wait_s)
            return ok_row(item, response, log, model_revision, git_sha)
    raise AssertionError("unreachable: every path through the retry loop returns a row")


def public_model_block(spec: dict) -> dict:
    return {k: spec[k] for k in MODEL_PUBLIC_KEYS if k in spec}


def generation_block(spec: dict) -> dict:
    """What every call in this model's run asks for, beyond the image and the schema. It is in
    `result_io.RESUME_IDENTITY`, so a resume that changed any of it is refused: rows answered
    under a 900-token budget with reasoning on are not comparable with rows answered under 1600
    with it off, and one file holding both would publish a single score for two different
    questions.
    """
    return {"max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
            "guided": bool(spec.get("guided")),
            "request_options": dict(request_options_for(spec))}


def build_dataset_block(repo_id, requested, requested_revision, inference_revision, items) -> dict:
    """`id` is always a resolvable Hub repo id."""
    return {"id": repo_id, "alias": requested if requested != repo_id else None,
            "requested_revision": requested_revision,
            "inference_revision": inference_revision, "split": SPLIT,
            "item_count": len(items),
            "item_ids_sha256": result_io.item_ids_sha256(i["id"] for i in items)}


def resolve_dataset_revision(dataset: str, revision: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(dataset, revision=revision).sha


def load_items(dataset: str, limit: int | None,
               revision: str = "main") -> tuple[list[dict], str, str]:
    """-> ([{id, image, target_schema, gold}], Hub repo id, resolved commit SHA). 'nls' = the NLS
    institution-gold set.
    """
    if dataset == "nls":
        from nls import NLS_DATASET, load_nls

        rows, sha = load_nls(limit, revision)
        return rows, NLS_DATASET, sha

    from datasets import load_dataset

    sha = resolve_dataset_revision(dataset, revision)
    ds = load_dataset(dataset, split=SPLIT, revision=sha)
    items = []
    for r in ds:
        if r["id"] not in TYPED_CORE:
            continue
        items.append({"id": r["id"], "image": r["image"],
                      "target_schema": r["target_schema"], "gold": r["silver_gold"]})
    # Checked before `--limit`, so a smoke test cannot pass on a slice the full run would refuse.
    result_io.refuse_duplicate_ids([i["id"] for i in items], f"{dataset} split {SPLIT}")
    if limit:
        items = items[:limit]
    return items, dataset, sha


# --- model selection --------------------------------------------------------------------------
# `--models` is required: the registry mixes models of very different cost, so running the whole
# set is asked for by name.

ALL_MODELS = "all"


def known_labels() -> list[str]:
    return [spec["label"] for spec in MODELS]


def is_enabled(spec: dict) -> bool:
    """A registry entry runs unless it says `enabled: False`."""
    return spec.get("enabled", True) is not False


def label_menu() -> str:
    """The known labels, disabled ones marked, for an error message."""
    described = []
    for spec in MODELS:
        suffix = "" if is_enabled(spec) else " (disabled)"
        described.append(spec["label"] + suffix)
    return ", ".join(described)


def select_specs(models_arg: str | None) -> list[dict]:
    """Registry entries for --models, in registry order."""
    if models_arg is None:
        runnable = [spec["label"] for spec in MODELS if is_enabled(spec)]
        raise SystemExit(
            "--models is required. Name the models to run, or 'all' for every enabled one:\n"
            f"  --models {ALL_MODELS}\n"
            f"  --models {','.join(runnable[:2])}\n"
            f"known labels: {label_menu()}")

    if models_arg.strip() == ALL_MODELS:
        return [spec for spec in MODELS if is_enabled(spec)]

    wanted = [label.strip() for label in models_arg.split(",") if label.strip()]
    known = set(known_labels())
    unknown = [label for label in wanted if label not in known]
    if unknown:
        raise SystemExit(f"unknown --models label(s): {', '.join(unknown)}\n"
                         f"known labels: {label_menu()}")

    by_label = {spec["label"]: spec for spec in MODELS}
    disabled = [label for label in wanted if not is_enabled(by_label[label])]
    if disabled:
        raise SystemExit(
            f"--models label(s) disabled in the registry: {', '.join(disabled)}\n"
            "Each carries its reason in glam_bench/models.py: never run in v1, or run and dropped "
            "from the board. To run one\nanyway, remove its `enabled: False` there — deliberately, "
            "in a commit.")

    return [spec for spec in MODELS if spec["label"] in set(wanted)]


# --- endpoint resolution ----------------------------------------------------------------------
# Where a model is served is a run detail, not model identity: a Jobs vLLM serve gets a fresh URL
# every time it is started. `--base-url` also works on a `router_vlm` spec: the spec then runs
# through the `openai` adapter with the endpoint attested. Only the adapter changes; `model.kind`
# stays the registry kind, so the transport is not part of the resume identity
# (result_io.RESUME_IDENTITY) and `--resume` can move router-failed rows onto a pinned serve.

ENDPOINT_ENV_PREFIX = "GLAM_BASE_URL_"

# http is allowed only for a loopback address, where there is no network to sniff. Everything
# else must be https: these calls carry an HF token in an Authorization header.
PLAINTEXT_HOSTS = frozenset({"localhost", "127.0.0.1"})

# Kinds an endpoint override may be applied to. A Space is served by Gradio, not an OpenAI API,
# so pointing one at a /v1 URL is a mistake worth refusing rather than ignoring.
ENDPOINT_OVERRIDABLE_KINDS = frozenset({"openai", "router_vlm"})

# Kinds that CANNOT run without one.
ENDPOINT_REQUIRED_KINDS = frozenset({"openai"})

# Kinds whose adapter sends `request_options`. A Space adapter has nowhere to put them.
REQUEST_OPTION_KINDS = frozenset({"openai", "router_vlm"})

# vLLM's standard way to turn a reasoning chat template off. Applied by the serving template, so
# it works on a serve you control; a router forwards it to the provider and cannot promise more.
NO_THINKING_REQUEST_OPTIONS = {"chat_template_kwargs": {"enable_thinking": False}}

# The second line of every "no endpoint" refusal: the thing to do about it.
SERVE_IT_FIRST = ("Serve it first — README, \"Run a model\" — and use the URL that job exposes.")


def endpoint_env_var(label: str) -> str:
    """The environment variable that can carry `label`'s endpoint."""
    normalised = []
    for ch in label:
        keep = ch.isascii() and ch.isalnum()
        normalised.append(ch.upper() if keep else "_")
    return ENDPOINT_ENV_PREFIX + "".join(normalised)


def label_value_pairs(values, switch: str) -> dict:
    """Repeated `LABEL=VALUE` switches into {label: value}."""
    pairs = {}
    for raw in values or []:
        label, separator, value = raw.partition("=")
        if not separator or not label.strip() or not value.strip():
            raise SystemExit(f"{switch} expects LABEL=VALUE, got {raw!r}")
        label = label.strip()
        if label in pairs:
            raise SystemExit(f"{switch} given twice for {label!r}: {pairs[label]!r} and "
                             f"{value.strip()!r}")
        pairs[label] = value.strip()
    return pairs


def endpoint_overrides_from_cli(values) -> dict:
    """{label: url} from repeated `--base-url LABEL=URL`."""
    return label_value_pairs(values, "--base-url")


def provider_overrides_from_cli(values) -> dict:
    """{label: provider} from repeated `--provider LABEL=NAME`."""
    return label_value_pairs(values, "--provider")


def endpoint_from_env(label: str, environ) -> str | None:
    """`label`'s endpoint from the environment, or None."""
    value = environ.get(endpoint_env_var(label), "").strip()
    return value or None


def validate_base_url(url: str, source: str) -> str:
    """-> the URL, normalised, or stop the run. `public_endpoint` strips credentials on the way into
    a result file, but a URL carrying a token was still going to be SENT from somewhere, and one
    that ends anywhere but `/v1` is a typo that would otherwise surface as a 404 on every single
    item.
    """
    parts = urlsplit(url)
    host = parts.hostname
    path = parts.path.rstrip("/")

    def refuse(reason: str):
        return SystemExit(f"{source}: {reason}\n  got: {url!r}\n"
                          "  expected an OpenAI-compatible base URL, e.g. "
                          "https://<job-id>--8000.hf.jobs/v1")

    if not host:
        raise refuse("no host in the URL")
    if parts.username or parts.password:
        raise refuse("the URL carries credentials (user:password@); pass the key via an env var "
                     "instead")
    if parts.query or parts.fragment:
        raise refuse("the URL carries a query string or fragment; a base URL takes neither")
    if parts.scheme == "http" and host not in PLAINTEXT_HOSTS:
        raise refuse(f"http:// is allowed only for {' and '.join(sorted(PLAINTEXT_HOSTS))} — "
                     "these calls carry a token")
    if parts.scheme not in ("http", "https"):
        raise refuse(f"scheme {parts.scheme!r} is not http or https")
    if not path.endswith("/v1"):
        raise refuse("the base URL must end with /v1 (the OpenAI API root, not a specific route)")

    netloc = host if parts.port is None else f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, path, "", ""))


def resolve_endpoint(spec: dict, overrides: dict, environ) -> tuple[str | None, str]:
    """-> (validated endpoint or None, where it came from)."""
    label = spec["label"]
    from_cli = overrides.get(label)
    if from_cli is not None:
        return validate_base_url(from_cli, f"--base-url {label}"), f"--base-url {label}"

    from_env = endpoint_from_env(label, environ)
    variable = endpoint_env_var(label)
    if from_env is not None:
        return validate_base_url(from_env, f"${variable}"), f"${variable}"

    return None, f"${variable}"


def missing_endpoint_line(label: str, variable: str) -> str:
    return f"{label} needs an endpoint: pass --base-url {label}=https://.../v1 or set {variable}"


def check_endpoints_present(specs, overrides: dict, environ) -> None:
    """Name EVERY selected model that has no endpoint, in one message."""
    missing = []
    for spec in specs:
        if spec["kind"] not in ENDPOINT_REQUIRED_KINDS:
            continue
        endpoint, variable = resolve_endpoint(spec, overrides, environ)
        if endpoint is None:
            missing.append(missing_endpoint_line(spec["label"], variable))
    if missing:
        raise SystemExit("\n".join(missing) + "\n" + SERVE_IT_FIRST)


def apply_endpoint(resolved: dict, overrides: dict, environ) -> None:
    """Put the run's endpoint on a resolved spec, and decide which adapter will use it."""
    label, kind = resolved["label"], resolved["kind"]
    endpoint, variable = resolve_endpoint(resolved, overrides, environ)

    if endpoint is None:
        if kind in ENDPOINT_REQUIRED_KINDS:
            # Ordinarily unreachable: check_endpoints_present has already refused the whole run.
            # Kept so `resolve_spec` cannot hand an endpoint-less openai spec to an adapter when
            # it is called on its own.
            raise SystemExit(missing_endpoint_line(label, variable) + "\n" + SERVE_IT_FIRST)
        resolved["adapter"] = kind
        return

    if kind not in ENDPOINT_OVERRIDABLE_KINDS:
        raise SystemExit(f"{label} is a {kind!r} model and cannot be pointed at an "
                         f"OpenAI-compatible endpoint; drop the --base-url for it.")

    resolved["base_url"] = endpoint
    resolved["adapter"] = "openai"
    # A Jobs serve takes the HF token as the api key; run_openai's OPENAI_API_KEY default would
    # send "EMPTY".
    resolved.setdefault("api_key_env", "HF_TOKEN")


def apply_provider(resolved: dict, providers: dict) -> None:
    """Pin a router provider for this run, so the row can name who answered."""
    label = resolved["label"]
    provider = providers.get(label)
    if provider is None:
        return
    if resolved["kind"] != "router_vlm":
        raise SystemExit(f"--provider {label}={provider}: only a router_vlm model has a provider "
                         f"({label} is {resolved['kind']!r}).")
    if resolved.get("adapter") == "openai":
        raise SystemExit(f"{label} was given both --base-url and --provider. A pinned endpoint "
                         "does not go through the router, so there is no provider to choose.")
    resolved["provider"] = provider


def with_thinking_disabled(options: dict) -> dict:
    """`options` plus the vLLM-standard switch for turning a reasoning chat template off. Merged
    rather than replaced, so a spec that already sets other `chat_template_kwargs` keeps them.
    """
    merged = dict(options)
    kwargs = dict(merged.get("chat_template_kwargs") or {})
    kwargs.update(NO_THINKING_REQUEST_OPTIONS["chat_template_kwargs"])
    merged["chat_template_kwargs"] = kwargs
    return merged


def apply_request_options(resolved: dict, no_thinking: bool) -> list[str]:
    """Settle the extra request body for this spec; -> labels this run could not apply it to."""
    options = dict(request_options_for(resolved))
    sendable = adapter_name(resolved) in REQUEST_OPTION_KINDS
    skipped = []

    if options and not sendable:
        raise SystemExit(f"{resolved['label']} is a {resolved['kind']!r} model: its adapter cannot "
                         "send request_options, so they would be recorded but never sent.")
    if no_thinking and sendable:
        options = with_thinking_disabled(options)
    elif no_thinking:
        skipped.append(resolved["label"])

    resolved["request_options"] = options
    return skipped


def resolve_spec(spec: dict, overrides: dict, providers: dict, no_thinking: bool,
                 environ) -> tuple[dict, list[str]]:
    """One registry entry plus this run's switches -> the spec the adapters actually see."""
    resolved = dict(spec)
    apply_endpoint(resolved, overrides, environ)
    apply_provider(resolved, providers)
    skipped = apply_request_options(resolved, no_thinking)
    return resolved, skipped


def check_override_labels(overrides: dict, specs, switch: str) -> None:
    """Every LABEL= given must be one of the models actually running."""
    running = {spec["label"] for spec in specs}
    stray = [label for label in overrides if label not in running]
    if stray:
        raise SystemExit(f"{switch} names model(s) this run is not running: {', '.join(stray)}\n"
                         f"running: {', '.join(sorted(running)) or '(none)'}")


def resolve_run_specs(args, environ=None) -> list[dict]:
    """Select the models and resolve everything a run needs from the CLI and the environment."""
    environ = os.environ if environ is None else environ
    overrides = endpoint_overrides_from_cli(args.base_url)
    providers = provider_overrides_from_cli(args.provider)

    specs = select_specs(args.models)
    check_override_labels(overrides, specs, "--base-url")
    check_override_labels(providers, specs, "--provider")
    check_endpoints_present(specs, overrides, environ)

    resolved = []
    skipped = []
    for spec in specs:
        one, could_not = resolve_spec(spec, overrides, providers, args.no_thinking, environ)
        resolved.append(one)
        skipped.extend(could_not)

    if skipped:
        print(f"note: --no-thinking does not apply to {', '.join(skipped)} (no request body to "
              "put it in); those rows record request_options {} and thinking_disabled false")
    return resolved


def print_run_plan(specs) -> None:
    """Say how every model will be reached, before the first request is paid for."""
    print("running:")
    for spec in specs:
        _served_by, endpoint, attested = describe_endpoint(spec)
        marks = []
        if not attested:
            marks.append("unattested")
        if thinking_disabled(request_options_for(spec)):
            marks.append("thinking off")
        suffix = f"  [{', '.join(marks)}]" if marks else ""
        print(f"  {spec['label']:24s} -> {endpoint}{suffix}")


@dataclass
class RunOptions:
    """The CLI switches that change how a run is written, rather than what it runs."""

    outdir: Path
    limit: int | None = None
    resume: bool = False
    select: str = "errors"


def recorded_model_revision(rows) -> str | None:
    """The `model_revision_hub_main` the kept rows agree on, or None when they do not say."""
    seen = set()
    for row in rows:
        if row.get("inference_status") != "ok":
            continue
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            continue
        revision = provenance.get("model_revision_hub_main")
        if isinstance(revision, str) and revision:
            seen.add(revision)
    if len(seen) == 1:
        return seen.pop()
    return None


def warn_if_model_revision_moved(existing_rows: dict, model_revision) -> bool:
    """Say when the model repo's main has moved since the rows being kept were produced. Refusing
    would break the one resume this harness is built to support — moving the rows that failed on
    the router onto a serve you pinned — and would do it for a fact the harness cannot act on
    anyway: `model_revision_hub_main` is what `main` was at lookup time, not proof of the weights
    any endpoint loaded (see `resolve_model_revision`).
    """
    was = recorded_model_revision(existing_rows.values())
    if was is None or model_revision is None or was == model_revision:
        return False
    print(f"  warning: model repo main moved since this run started ({was[:8]} → "
          f"{model_revision[:8]}); new rows record the new value, kept rows the old — "
          "per-row provenance is the truth", flush=True)
    return True


@dataclass
class RunPlan:
    """Everything one model's run needs that does not change from row to row."""

    model_block: dict
    dataset_block: dict
    generation_block: dict
    git_sha: str | None
    run_git_sha: str | None
    final_path: Path
    partial_path: Path
    started_at: str
    resumed_at: list[str]
    # Settled after the plan is built, once the model revision has been looked up: see
    # `warn_if_model_revision_moved`, which needs both the kept rows and a fresh lookup.
    model_revision_moved: bool = False


# What a scored row that does not say which scorer scored it counts as. Sorts after every real
# version, which are dates, so it reads last in `scorer_versions`.
UNKNOWN_SCORER_VERSION = "unknown"


def was_scored(row) -> bool:
    """Whether this row carries scores that some scorer produced."""
    if row.get("inference_status") == "transport_error":
        return False
    if row.get("scorer_error"):
        return False
    return "content_f1" in row


def scorer_versions_in(rows) -> list[str]:
    """Every scorer version the rows were scored by, sorted, each named once."""
    versions = set()
    for row in rows:
        version = row.get("scorer_version")
        if isinstance(version, str) and version:
            versions.add(version)
        elif was_scored(row):
            versions.add(UNKNOWN_SCORER_VERSION)
    return sorted(versions)


def scoring_block(rows) -> dict:
    """What scored these rows."""
    versions = scorer_versions_in(rows)
    if len(versions) > 1:
        return {"scorer_version": "mixed", "scorer_versions": versions}
    if len(versions) == 1:
        return {"scorer_version": versions[0]}
    return {"scorer_version": SCORER_VERSION}


def envelope_for(plan: RunPlan, rows, complete: bool) -> dict:
    """This run's file."""
    run_block = result_io.build_run_block(plan.started_at, utc_now(), plan.run_git_sha,
                                          complete, rows, plan.resumed_at,
                                          plan.model_revision_moved)
    return result_io.build_envelope(FORMAT_VERSION, plan.model_block, plan.dataset_block,
                                    plan.generation_block, scoring_block(rows),
                                    run_block, result_io.HARNESS_PRODUCED_BY, rows)


def load_resume_state(spec, current_blocks: dict, plan_paths, options: RunOptions) -> dict:
    """Read the file a `--resume` continues, after proving it describes this same run. -> {"rows":
    {item id: row}, "started_at": str|None, "resumed_at": [...], "run_git_sha": str|None,
    "model_block": dict|None}; empty when there is nothing to resume.
    """
    final_path, partial_path = plan_paths
    empty = {"rows": {}, "started_at": None, "resumed_at": [], "run_git_sha": None,
             "model_block": None}
    if not options.resume:
        return empty

    source = result_io.find_resume_source(final_path, partial_path)
    if source is None:
        print(f"  resume: no existing file for {spec['label']}; running every item")
        return empty

    existing = result_io.load_result_file(source)
    result_io.validate_resume_target(existing, current_blocks, source)
    warning = result_io.scorer_version_warning(existing, SCORER_VERSION)
    if warning:
        print(f"  {warning}")

    rows = result_io.rows_by_id(existing)
    resumed_at = list(result_io.field_at(existing, "run", "resumed_at") or [])
    resumed_at.append(utc_now())
    existing_model = existing.get("model")
    print(f"  resume: {source.name} ({len(rows)} existing rows, --select {options.select})")
    return {"rows": rows, "started_at": result_io.field_at(existing, "run", "started_at"),
            "resumed_at": resumed_at,
            "run_git_sha": result_io.field_at(existing, "run", "harness_git_sha"),
            "model_block": existing_model if isinstance(existing_model, dict) else None}


def plan_run(spec, dataset_block, git_sha, options: RunOptions) -> tuple[RunPlan, dict]:
    """-> (plan, {item id: existing row})."""
    model_block = public_model_block(spec)
    generation = generation_block(spec)
    final_path = result_io.result_path(options.outdir, spec["label"], options.limit)
    partial_path = result_io.partial_path(final_path)
    state = load_resume_state(spec, {"model": model_block, "dataset": dataset_block,
                                     "generation": generation},
                              (final_path, partial_path), options)
    plan = RunPlan(model_block=state["model_block"] or model_block, dataset_block=dataset_block,
                   generation_block=generation,
                   git_sha=git_sha, run_git_sha=state["run_git_sha"] or git_sha,
                   final_path=final_path, partial_path=partial_path,
                   started_at=state["started_at"] or utc_now(),
                   resumed_at=state["resumed_at"])
    return plan, state["rows"]


def carried_rows(items, existing_rows: dict) -> dict:
    """The existing rows that belong to this run's items, keyed by id."""
    carried = {}
    for item in items:
        row = existing_rows.get(item["id"])
        if row is not None:
            carried[item["id"]] = row
    return carried


def total_attempts(previous, row: dict) -> int:
    """Every call ever made for one row: what earlier runs spent, plus what run_item just spent.
    """
    return result_io.previous_attempts(previous) + row["attempts"]


def ordered_rows(items, completed: dict) -> list[dict]:
    """Rows in dataset order, skipping items that have no row yet."""
    rows = []
    for item in items:
        row = completed.get(item["id"])
        if row is not None:
            rows.append(row)
    return rows


def count_by_failure_class(rows) -> dict:
    """{failure_class: how many transport-error rows carry it}."""
    counts = {}
    for row in rows:
        if row.get("inference_status") != "transport_error":
            continue
        failure_class = row.get("failure_class")
        if not isinstance(failure_class, str):
            continue
        counts[failure_class] = counts.get(failure_class, 0) + 1
    return counts


# What each failure class means in the end-of-model summary, in the order it is reported.
FAILURE_CLASS_NOTES = (
    ("permanent", "auth, bad request, not found"),
    ("unclassified", "an exception the harness does not recognise — likely a harness or "
                     "response-shape bug, see error"),
    ("transient", "ran out of attempts"),
)


def failure_class_summary(rows) -> str:
    """`1 permanent (...); 2 transient (...)` for the classes present, or "" for none of them."""
    counts = count_by_failure_class(rows)
    described = []
    for failure_class, note in FAILURE_CLASS_NOTES:
        count = counts.get(failure_class, 0)
        if count:
            described.append(f"{count} {failure_class} ({note})")
    return "; ".join(described)


def print_row(label: str, row: dict, kept: bool) -> None:
    note = "(kept)" if kept else (row.get("error") or row.get("scorer_error") or "")
    print(f"  {label:16s} {row['id']:34s} f1={row.get('content_f1', 0):.2f} "
          f"valid={row.get('schema_valid', 0):.0f} {note}", flush=True)


def run_model(spec, items, dataset_block, git_sha, options: RunOptions,
              retry: RetryPolicy) -> Path:
    """Run one model over `items` and leave exactly one finished file behind."""
    plan, existing_rows = plan_run(spec, dataset_block, git_sha, options)
    model_revision = resolve_model_revision(spec)
    plan.model_revision_moved = warn_if_model_revision_moved(existing_rows, model_revision)
    completed = carried_rows(items, existing_rows)

    for item in items:
        previous = existing_rows.get(item["id"])
        redo = previous is None or result_io.needs_redo(previous, options.select)
        if redo:
            row = run_item(item, spec, model_revision, git_sha, retry)
            row["attempts"] = total_attempts(previous, row)
            completed[item["id"]] = row
        else:
            row = previous
        print_row(spec["label"], row, kept=not redo)
        result_io.atomic_write_json(
            plan.partial_path, envelope_for(plan, ordered_rows(items, completed), complete=False))

    rows = ordered_rows(items, completed)
    result_io.atomic_write_json(plan.final_path, envelope_for(plan, rows, complete=True))
    plan.partial_path.unlink(missing_ok=True)

    transport_errors = result_io.count_transport_errors(rows)
    if transport_errors:
        summary = failure_class_summary(rows)
        note = f": {summary}" if summary else ""
        print(f"  {spec['label']}: {transport_errors} transport error(s){note} — rerun with "
              f"--resume to retry those rows; only the transient ones are worth rerunning as they "
              f"stand", flush=True)
    return plan.final_path


def positive_limit(value: str) -> int:
    """Argparse type for `--limit`: a count of items to run, so 0 and below are not a smaller run.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError(
            "--limit must be a positive integer; use no --limit for the full run")
    return number


def prepare_dataset(args):
    config = args.config or ("nls-index-cards" if args.dataset == BENCHMARK_ID else None)
    if config:
        rows, manifest, revision = load_hub_config(args.dataset, config, args.dataset_revision)
        items = inference_items(rows, manifest, args.limit)
        block = build_dataset_block(args.dataset, args.dataset, args.dataset_revision, revision, items)
        block.update(evaluation_identity(manifest))
        block["task_instructions"] = manifest["task_instructions"]
        return items, block, config
    items, repo_id, revision = load_items(args.dataset, args.limit, args.dataset_revision)
    block = build_dataset_block(repo_id, args.dataset, args.dataset_revision, revision, items)
    return items, block, "nls" if args.dataset == "nls" else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", action="version", version=f"glam-extraction-benchmark {__version__}")
    ap.add_argument("--models", default=None,
                    help="REQUIRED. Comma-separated registry labels, or 'all' for every enabled "
                         "entry. A bare run lists the labels and stops rather than calling every "
                         "model in the registry")
    ap.add_argument("--base-url", action="append", metavar="LABEL=URL",
                    help="where to reach a model, repeatable. Required for every kind:'openai' "
                         "model (the registry holds no endpoints) and allowed on a router model, "
                         "which then runs through the OpenAI-compatible adapter against your own "
                         "serve. Falls back to $GLAM_BASE_URL_<LABEL> (uppercased, non-alphanumeric "
                         "-> _). Must be https (or http on localhost) and end with /v1. The key "
                         "comes from the entry's api_key_env, defaulting to HF_TOKEN")
    ap.add_argument("--provider", action="append", metavar="LABEL=NAME",
                    help="pin the Inference Providers provider for a router model, repeatable. "
                         "Recorded as router:<name> and attested. Left off, the router chooses and "
                         "never says who answered, which is recorded as router:auto, unattested")
    ap.add_argument("--no-thinking", action="store_true",
                    help="ask the chat template to turn reasoning off for every selected model, "
                         "the vLLM-standard way (chat_template_kwargs.enable_thinking=false). "
                         "Recorded per row in provenance.request_options / thinking_disabled. On a "
                         "serve you pinned this is applied by the template; through the router it "
                         "is forwarded to a provider that may or may not honour it")
    ap.add_argument("--dataset", default=BENCHMARK_ID, help='benchmark Hub dataset ID, or "nls" for legacy NLS runs')
    ap.add_argument("--config", default=None, help="benchmark dataset config (default: nls-index-cards for the GLAM repo)")
    ap.add_argument("--dataset-revision", default="main",
                    help="branch/tag/sha; resolved once to a commit and pinned for the whole run")
    ap.add_argument("--limit", type=positive_limit, default=None,
                    help="first N items (smoke tests), N >= 1; writes <LABEL>.limit<N>.json, never the "
                         "canonical <LABEL>.json, so a smoke test cannot be published or "
                         "resumed into as if it were a full run")
    ap.add_argument("--max-tokens", type=int, default=900, help="use ~1600 for nls (long entry lists)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing <LABEL>.json.partial (preferred) or <LABEL>.json "
                         "instead of starting over; refuses a file from a different dataset "
                         "snapshot, item slice or model")
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                    help="how many times one item may be called when the failure looks "
                         "transient (a timeout, a dropped connection, 429, 5xx). Authentication, "
                         "invalid-request and not-found failures are never retried, nor is an "
                         "exception the harness does not recognise, and neither is an unparseable "
                         "answer — that is a model result (default: 3)")
    ap.add_argument("--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF_S,
                    help="seconds to wait after the first failed attempt, doubling each time and "
                         f"capped at {MAX_RETRY_WAIT_S:.0f}s, plus jitter (default: 2)")
    ap.add_argument("--select", choices=result_io.SELECT_MODES, default=None,
                    help="with --resume, which existing rows to infer again (default: errors). "
                         "errors = transport failures, scorer crashes and missing rows; "
                         "unparseable = those plus rows the scorer could not parse; "
                         "all = every row. Ignored without --resume")
    args = ap.parse_args(argv)

    global MAX_TOKENS
    MAX_TOKENS = args.max_tokens

    if args.select is not None and not args.resume:
        print("note: --select only affects a --resume run; every item is being run anyway")
    select = args.select or "errors"

    # Everything the run depends on is settled before the first byte is downloaded: the retry
    # switches, which models, and where each of them is served. A run that is going to fail on
    # configuration fails here, in under a second, having called nobody.
    retry = build_retry_policy(args.max_attempts, args.retry_backoff)
    specs = resolve_run_specs(args)
    print_run_plan(specs)

    items, dataset_block, result_subdir = prepare_dataset(args)
    git_sha = harness_git_sha()

    outdir = RESULTS / result_subdir if result_subdir else RESULTS
    outdir.mkdir(parents=True, exist_ok=True)
    options = RunOptions(outdir=outdir, limit=args.limit, resume=args.resume, select=select)
    for spec in specs:
        if dataset_block.get("config"):
            spec = {**spec, "task_instructions": dataset_block["task_instructions"]}
        path = run_model(spec, items, dataset_block, git_sha, options, retry)
        print(f"  -> wrote {path.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
