"""Optional, self-contained HF Jobs inference recipe; not imported by the harness.

Local: uv run --with huggingface_hub==1.31.0 jobs_batch.py launch --model LABEL
A job runs vLLM, scores/checkpoints each card, uploads outputs, then exits.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import shlex
import sys
import tarfile
import time

BUCKET = "small-models-for-glam/glam-extraction-results"
NAMESPACE = os.environ.get("GLAM_JOBS_NAMESPACE", "small-models-for-glam")
RUN_ID = "2026-09-15-nullable-v1"
REVISION = "ecc9c02582f933ca89b73e6751e4ed76888cb24d"
DATASET = "small-models-for-glam/glam-extraction-benchmark"
IMAGE = "vllm/vllm-openai:v0.29.0"
HARDWARE_USD_PER_HOUR = {"l40sx1": 1.80, "l4x1": 0.80, "a10g-small": 1.00, "a100-large": 2.50}
MODELS = {
    "Qwen3.5-2B": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", "2.274B"),
    "Qwen3.5-4B": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "4.660B"),
    "NuExtract-3": ("numind/NuExtract3", "c99dc8f5641b866aa0192b6ea78f84bf9f3535f1", "4.539B"),
    "Granite-Vision-4.1-4B": (
        "ibm-granite/granite-vision-4.1-4b",
        "37d591f06319e8f1638b5adcf58bdf50e0f84f7a",
        "3.997B",
    ),
    "gemma-4-E4B": ("google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2", "7.996B"),
    "Qwen3-VL-8B": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "8.767B"),
}


def upload(api, path, prefix):
    api.batch_bucket_files(BUCKET, add=[(str(path), f"{prefix}/{path.name}")])


def wait_for_server(server):
    import httpx

    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited with {server.returncode}")
        try:
            if httpx.get("http://127.0.0.1:8000/health", timeout=3).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(5)
    else:
        raise TimeoutError("vLLM did not become ready within 15 minutes")


def request_for(item, spec, h):
    from dataset_contract import scoring_schema
    from schema import field_notes, jsonschema_to_nuextract

    schema = item["target_schema"]
    extra = dict(spec["request_options"])
    messages = h._vlm_messages(item["image"], schema, spec["task_instructions"])
    prompt_recipe = "harness-vlm"
    if spec["label"] == "NuExtract-3":
        projected = scoring_schema(json.loads(schema))
        instruction = spec["task_instructions"] + " Return null or [] for absent information."
        notes = field_notes(projected)
        if notes:
            instruction += " Field notes:\n" + "\n".join(notes)
        extra = {
            "chat_template_kwargs": {
                "template": json.dumps(jsonschema_to_nuextract(projected)),
                "instructions": instruction,
                "enable_thinking": False,
            }
        }
        messages[0]["content"] = [messages[0]["content"][1]]
        prompt_recipe = "numind-official-template"
    elif spec["label"] == "Granite-Vision-4.1-4B":
        messages[0]["content"][0]["text"] = (
            spec["task_instructions"]
            + "\nExtract structured data from this document.\nReturn a JSON object matching this schema:\n"
            + schema
            + "\nReturn null for fields you cannot find.\nReturn ONLY valid JSON.\nReturn an instance of the JSON with extracted values, not the schema itself."
        )
        messages[0]["content"].reverse()
        prompt_recipe = "granite-official-kvp"
    return messages, extra, prompt_recipe


def check_smoke(choice, row):
    # Low extraction scores are valid results; detect broken serving before the full run.
    if choice.finish_reason == "length":
        raise RuntimeError("Smoke response hit max_tokens; stop before full batch")
    if not row.get("schema_valid"):
        raise RuntimeError("Smoke response is not parseable JSON; stop before full batch")


def worker(args):
    from huggingface_hub import HfApi
    from openai import OpenAI

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "glam_bench"))
    import harness as h
    import result_io

    api = HfApi()
    prefix = f"{args.config}/{args.run_id}"
    outdir = root / "outputs"
    outdir.mkdir(exist_ok=True)
    receipt_path = outdir / f"{args.model}.launch.json"
    api.download_bucket_files(
        BUCKET, [(f"{prefix}/{receipt_path.name}", receipt_path)], raise_on_missing_files=True
    )
    job_id = json.loads(receipt_path.read_text())["job_id"]
    model_id, revision, params = MODELS[args.model]
    command = [
        "vllm",
        "serve",
        model_id,
        "--revision",
        revision,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.85",
        "--generation-config",
        "vllm",
        "--limit-mm-per-prompt",
        '{"image":1,"video":0}',
        "--enforce-eager",
    ]
    if args.model == "NuExtract-3":
        command += ["--chat-template-content-format", "openai"]
    options = (
        {} if args.model == "Granite-Vision-4.1-4B" else {"chat_template_kwargs": {"enable_thinking": False}}
    )
    spec = {
        "label": args.model,
        "id": model_id,
        "kind": "openai",
        "params": params,
        "base_url": "http://127.0.0.1:8000/v1",
        "request_options": options,
        "cost": f"HF Jobs {args.flavor} ${HARDWARE_USD_PER_HOUR[args.flavor]:.2f}/hour",
    }
    meta = {
        "run_id": args.run_id,
        "dataset": args.dataset,
        "config": args.config,
        "dataset_revision": args.dataset_revision,
        "model": model_id,
        "model_revision": revision,
        "image": IMAGE,
        "hardware": args.flavor,
        "hourly_usd": HARDWARE_USD_PER_HOUR[args.flavor],
        "job_id": job_id,
        "namespace": NAMESPACE,
        "server_command": command,
        "started_at": h.utc_now(),
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["vllm", "torch", "transformers", "huggingface_hub", "datasets", "openai"]
        },
    }
    metadata_path = outdir / f"{args.model}.job.json"
    result_io.atomic_write_json(metadata_path, meta)
    upload(api, metadata_path, prefix)
    server = subprocess.Popen(command)
    try:
        wait_for_server(server)
        h.MAX_TOKENS = 1600
        ds_args = argparse.Namespace(
            config=args.config, dataset=args.dataset, dataset_revision=args.dataset_revision, limit=args.limit
        )
        items, block, _ = h.prepare_dataset(ds_args)
        spec["task_instructions"] = block["task_instructions"]
        plan, _ = h.plan_run(
            spec, block, os.environ.get("HARNESS_GIT_SHA"), h.RunOptions(outdir=outdir, limit=args.limit)
        )
        client = OpenAI(base_url=spec["base_url"], api_key="unused", max_retries=0, timeout=180)
        rows = []
        for item in items:
            messages, extra, prompt_recipe = request_for(item, spec, h)
            started = time.monotonic()
            response = client.chat.completions.create(
                model=model_id, messages=messages, max_tokens=1600, temperature=0, extra_body=extra
            )
            choice = response.choices[0]
            adapted = h.AdapterResponse(
                content=choice.message.content,
                served_by="hf-jobs-vllm",
                endpoint=f"hf-jobs:{NAMESPACE}/{job_id}",
                finish_reason=choice.finish_reason,
                request_options=extra,
            )
            row = h.ok_row(
                item,
                adapted,
                h.AttemptLog(attempts=1, latency_s=time.monotonic() - started, retry_wait_s=0),
                revision,
                os.environ.get("HARNESS_GIT_SHA"),
            )
            row["provenance"].update(
                {
                    "loaded_model_revision": revision,
                    "prompt_recipe": prompt_recipe,
                    "hardware": args.flavor,
                    "usage": response.usage.model_dump() if response.usage else None,
                }
            )
            rows.append(row)
            payload = h.envelope_for(plan, rows, complete=False)
            result_io.atomic_write_json(plan.partial_path, payload)
            upload(api, plan.partial_path, prefix)
            h.print_row(args.model, row, kept=False)
            if len(rows) <= 3:
                check_smoke(choice, row)
            if len(rows) == 3:
                print("SMOKE_OK: three responses completed without truncation", flush=True)
        result_io.atomic_write_json(plan.final_path, h.envelope_for(plan, rows, complete=True))
        upload(api, plan.final_path, prefix)
        meta["completed_at"] = h.utc_now()
        meta["item_count"] = len(rows)
        meta["status"] = "completed"
    except BaseException as exc:
        meta["status"] = "error"
        meta["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        raise
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        meta["ended_at"] = h.utc_now()
        result_io.atomic_write_json(metadata_path, meta)
        upload(api, metadata_path, prefix)


def launch(args):
    from huggingface_hub import HfApi, get_token

    api = HfApi()
    root = Path(__file__).resolve().parents[2]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ["harness", "result_io", "dataset_contract", "schema", "scorer", "models", "version"]:
            path = root / "glam_bench" / f"{name}.py"
            archive.add(path, arcname=str(path.relative_to(root)))
        archive.add(__file__, arcname="examples/inference/jobs_batch.py")
    source = f"{args.config}/{args.run_id}/recipes/jobs-source.tar.gz"
    if not args.reuse_source:
        api.batch_bucket_files(BUCKET, add=[(buffer.getvalue(), source)])
    bootstrap = (
        "from huggingface_hub import HfApi; import tarfile,os; "
        f"HfApi().download_bucket_files({BUCKET!r},[({source!r},'/tmp/recipe.tar.gz')],raise_on_missing_files=True); "
        "os.makedirs('/tmp/glam',exist_ok=True); "
        "tarfile.open('/tmp/recipe.tar.gz').extractall('/tmp/glam',filter='data')"
    )
    # Credentials travel only as a Jobs secret; no private GitHub credentials are needed.
    command = [
        "bash",
        "-lc",
        "uv pip install --system 'huggingface_hub==1.31.0' 'datasets>=5,<6' 'jsonschema>=4,<5' && python3 -c "
        + shlex.quote(bootstrap)
        + ' && python3 /tmp/glam/examples/inference/jobs_batch.py worker --model "$MODEL_LABEL" --run-id "$RUN_ID"'
        + (f" --limit {args.limit}" if args.limit else ""),
    ]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    job_label = args.model.replace(".", "-")
    job = api.run_job(
        image=IMAGE,
        command=command,
        namespace=NAMESPACE,
        flavor=args.flavor,
        timeout="60m",
        name=f"glam-{args.config}-{job_label.lower()}",
        labels={"benchmark": "glam", "config": args.config, "run": args.run_id, "model": job_label},
        env={
            "MODEL_LABEL": args.model,
            "RUN_ID": args.run_id,
            "HARNESS_GIT_SHA": sha,
            "PYTHONUNBUFFERED": "1",
            "GLAM_JOBS_NAMESPACE": NAMESPACE,
            "JOB_FLAVOR": args.flavor,
            "BENCHMARK_CONFIG": args.config,
            "BENCHMARK_DATASET": args.dataset,
            "BENCHMARK_REVISION": args.dataset_revision,
        },
        secrets={"HF_TOKEN": get_token()},
    )
    record = {
        "job_id": job.id,
        "namespace": NAMESPACE,
        "model": args.model,
        "image": IMAGE,
        "timeout_minutes": 60,
        "hardware": args.flavor,
        "maximum_compute_usd": HARDWARE_USD_PER_HOUR[args.flavor],
        "run_id": args.run_id,
        "dataset": args.dataset,
        "config": args.config,
        "dataset_revision": args.dataset_revision,
        "url": f"https://huggingface.co/jobs/{NAMESPACE}/{job.id}",
    }
    api.batch_bucket_files(
        BUCKET,
        add=[
            (json.dumps(record, indent=2).encode(), f"{args.config}/{args.run_id}/{args.model}.launch.json")
        ],
    )
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "worker"])
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--config", default=os.environ.get("BENCHMARK_CONFIG", "nls-index-cards"))
    parser.add_argument("--dataset", default=os.environ.get("BENCHMARK_DATASET", DATASET))
    parser.add_argument("--dataset-revision", default=os.environ.get("BENCHMARK_REVISION", REVISION))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--reuse-source",
        action="store_true",
        help="Launch from the runtime bundle already staged for this run ID",
    )
    parser.add_argument(
        "--flavor", choices=HARDWARE_USD_PER_HOUR, default=os.environ.get("JOB_FLAVOR", "l40sx1")
    )
    args = parser.parse_args()
    (launch if args.action == "launch" else worker)(args)


if __name__ == "__main__":
    main()
