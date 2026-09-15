# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=5,<6", "huggingface_hub>=1.31,<2", "httpx>=0.28,<1", "pillow>=12,<13", "jsonschema>=4,<5"]
# ///
"""Optional HF API producer: pinned benchmark config -> submission JSON + bucket checkpoints.

Run from a checkout with `uv run examples/inference/api_batch.py --help`.
The benchmark does not import this recipe or require its providers/storage.
"""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "glam_bench"))
import harness as h  # noqa: E402
from dataset_contract import inference_items, load_hub_config  # noqa: E402
from result_io import atomic_write_json, validate_resume_target, validate_submission  # noqa: E402

DATASET = "small-models-for-glam/glam-extraction-benchmark"
REVISION = "ecc9c02582f933ca89b73e6751e4ed76888cb24d"
CONFIG = "nls-index-cards"
BUCKET = "small-models-for-glam/glam-extraction-results"
RUN_ID = "2026-09-15-nullable-v1"
ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
# USD per million tokens, read from the HF router catalogue on 2026-09-15.
MODELS = {
    "Qwen3.5-9B": ("Qwen/Qwen3.5-9B", "deepinfra", "9.65B", .10, .15, 1600, NO_THINKING),
    "Qwen3.8-27B": ("Qwen/Qwen3.8-27B", "novita", "27.78B", .42, 3., 1600, NO_THINKING),
    "Qwen3-VL-235B": ("Qwen/Qwen3-VL-235B-A22B-Instruct", "deepinfra", "235.67B", .20, .88, 1600, {}),
    "GLM-5.3-Flash": ("zai-org/GLM-5.3-Flash", "deepinfra", "321.32B", .15, .50, 4096, {"reasoning_effort": "low"}),
}


def main():  # noqa: C901 — linear, optional one-off batch recipe
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--limit", type=int, help="Pause after this many rows; resume retains them")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--provider", choices=["deepinfra", "novita"], help="Override the route; preserved per row")
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    model_id, provider, params, price_in, price_out, max_tokens, options = MODELS[args.model]
    provider = args.provider or provider
    recipe_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    h.MAX_TOKENS = max_tokens
    api = HfApi()
    assert api.bucket_info(BUCKET).private, "Expected private output bucket"
    rows, manifest, revision = load_hub_config(args.dataset, args.config, args.revision)
    items = inference_items(rows, manifest)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    spec = {"label": args.model, "id": model_id, "kind": "router_vlm", "provider": provider,
            "params": params, "request_options": options}
    folder = ROOT / "results" / args.config / args.run_id
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / f"{args.model}.json"
    partial = folder / f"{args.model}.json.partial"
    identity = {"id": args.dataset, "inference_revision": revision, "config": args.config,
                "split": manifest["split"], "item_count": len(items),
                "item_ids_sha256": h.result_io.item_ids_sha256(i["id"] for i in items)}
    doc = {"format_version": 2, "model": h.public_model_block(spec), "dataset": identity,
           "generation": h.generation_block(spec),
           "scoring": {"scorer_version": h.SCORER_VERSION},
           "run": {"started_at": h.utc_now(), "harness_git_sha": git_sha, "complete": False,
                   "transport_errors": 0, "recipe": "examples/inference/api_batch.py",
                   "dependencies": {p: importlib.metadata.version(p) for p in ["datasets", "huggingface_hub", "httpx", "pillow", "jsonschema"]}},
           "produced_by": {"producer": "other", "attested": False}, "items": [],
           "billing": {"input_usd_per_million": price_in, "output_usd_per_million": price_out,
                       "pricing_date": "2026-09-15", "estimated_usd": 0., "budget_usd": args.budget_usd}}
    previous = partial if partial.exists() else final
    if previous.exists():
        if not args.resume:
            raise SystemExit("Output exists; use --resume or a different --run-id")
        old = json.loads(previous.read_text())
        validate_resume_target(old, doc, previous)
        doc = old
        if doc["model"].get("provider") != provider:
            doc["model"]["provider"] = "mixed; see items[].provenance.endpoint"
    doc["run"]["active_recipe_sha256"] = recipe_sha
    done = {r["id"] for r in doc["items"]}
    model_revision = api.model_info(model_id).sha

    def checkpoint(complete=False):
        doc["run"].update(complete=complete, finished_at=h.utc_now())
        actual = sorted({r["provenance"]["endpoint"].removeprefix("router:") for r in doc["items"]})
        if actual:
            doc["model"]["provider"] = actual[0] if len(actual) == 1 else "mixed; see items[].provenance.endpoint"
        path = final if complete else partial
        atomic_write_json(path, doc)
        api.batch_bucket_files(BUCKET, add=[(path, f"{args.config}/{args.run_id}/{path.name}")])
        if complete:
            partial.unlink(missing_ok=True)

    with httpx.Client(headers={"Authorization": f"Bearer {get_token()}"}, timeout=240) as client:
        for item in items:
            if item["id"] in done:
                continue
            if args.limit and len(doc["items"]) >= args.limit:
                break
            # Reserve 64k input + the configured output cap before every request.
            reserve = (64000 * price_in + max_tokens * price_out) / 1e6
            if doc["billing"]["estimated_usd"] + reserve > args.budget_usd:
                checkpoint()
                raise SystemExit("Budget pause: inspect recorded usage before resuming")
            body = {"model": f"{model_id}:{provider}", "messages": h._vlm_messages(
                item["image"], item["target_schema"], manifest["task_instructions"]),
                "max_tokens": max_tokens, "temperature": 0, **options}
            started = time.monotonic()
            response = client.post(ENDPOINT, json=body)
            if response.status_code != 200:
                checkpoint()
                raise SystemExit(f"Provider HTTP {response.status_code}: {response.text[:500]}")
            result = response.json()
            usage = result.get("usage")
            if not usage or "prompt_tokens" not in usage or "completion_tokens" not in usage:
                # Preserve the response before stopping; do not run unmetered requests.
                atomic_write_json(folder / f"{args.model}.unmetered-response.json", result)
                checkpoint()
                raise SystemExit("Provider omitted token usage; stopped for inspection")
            cost = (usage["prompt_tokens"] * price_in + usage["completion_tokens"] * price_out) / 1e6
            doc["billing"]["estimated_usd"] += cost
            choice = result["choices"][0]
            answer = h.AdapterResponse(choice["message"].get("content"), "hf-inference-providers",
                                       f"router:{provider}", choice.get("finish_reason"), True, options)
            row = h.ok_row(item, answer, h.AttemptLog(1, time.monotonic()-started), model_revision, git_sha)
            row["provenance"].update(usage=usage, estimated_cost_usd=cost,
                                     response_model=result.get("model"), response_id=result.get("id"),
                                     recipe_sha256=recipe_sha)
            reasoning = choice["message"].get("reasoning_content")
            if reasoning is not None:
                row["reasoning_content"] = reasoning
            doc["items"].append(row)
            atomic_write_json(partial, doc)
            print(f"{args.model} {len(doc['items'])}/{len(items)} finish={choice.get('finish_reason')} "
                  f"F1={row.get('content_f1', 0):.3f} usage={usage} cost=${doc['billing']['estimated_usd']:.5f}", flush=True)
            if len(doc["items"]) % 10 == 0:
                checkpoint()
            if choice.get("finish_reason") == "length" and len(doc["items"]) <= 2:
                checkpoint()
                raise SystemExit("Smoke output truncated; inspect generation settings")
    complete = len(doc["items"]) == len(items)
    if complete:
        problems = validate_submission(doc, [i["id"] for i in items])
        if problems:
            raise SystemExit(str(problems))
    checkpoint(complete)
    print(f"{'COMPLETE' if complete else 'PAUSED'} {args.model}: {len(doc['items'])} rows", flush=True)


if __name__ == "__main__":
    main()
