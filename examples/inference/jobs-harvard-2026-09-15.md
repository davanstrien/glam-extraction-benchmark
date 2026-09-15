# Harvard Jobs run — 2026-09-15

Six models completed the 30 scored Harvard botany cards. Every completed
submission passed the benchmark validator. The five unscored review cards were
excluded. No outputs were retried, discarded, or reused from NLS.

| Model | Hardware | F1 | Truncated / 30 | Running seconds | Est. compute |
|---|---|---:|---:|---:|---:|
| Qwen3.5-2B | l4x1 | 0.8162 | 0 | 250 | $0.07 |
| Qwen3.5-4B | l40sx1 | 0.9741 | 0 | 247 | $0.15 |
| NuExtract-3 | l40sx1 | 0.9401 | 0 | 238 | $0.12 |
| Granite-Vision-4.1-4B | l40sx1 | 0.9095 | 0 | 137 | $0.09 |
| gemma-4-E4B | a100-large | 0.8807 | 0 | 259 | $0.21 |
| Qwen3-VL-8B | l40sx1 | 0.9167 | 0 | 187 | $0.12 |

Estimated compute total: **$0.76**. This uses reported running seconds,
rounded up to minutes, and the checked hardware rates; it is an estimate rather
than an invoice. Six inference Jobs completed and exited. The initial Gemma L40S
request was cancelled after waiting for hardware without starting; its receipt
and initial runtime bundle are retained. Gemma then ran on an A100 80 GB at
$2.50/hour. No duplicate inference ran. No transport errors or token
limit truncations occurred across the 180 predictions.

## Reproduce the generation

The [optional recipe](jobs-batch.md) accepts another config without changing the
harness. The first model used:

```bash
GLAM_JOBS_NAMESPACE=davanstrien uv run --with huggingface_hub==1.31.0 \
  examples/inference/jobs_batch.py launch --model Qwen3.5-2B --flavor l4x1 \
  --config harvard-botany-headers \
  --dataset-revision 48e92ea1112d3fc2a32467e875d64e10129c4d46 \
  --run-id 2026-09-15-v1
```

The other models used the same config, revision, and run ID. Four used
`--reuse-source --flavor l40sx1`; Gemma used `--flavor a100-large`, staging the
updated runtime bundle with that hardware option. At most three Jobs were
active or queued together; each had a 60-minute timeout. The first three 2B responses were inspected before the
remaining models were launched. False correction values were retained and scored.

Model revisions, prompt adapters, and generation settings were unchanged from
the [NLS batch](jobs-2026-09-15.md): temperature 0, max_tokens 1600, thinking off
where supported. NuExtract used its native extraction template; Granite used its
documented KVP prompt; the other four used the reference VLM prompt.

Runtime: vLLM 0.29.0, torch 2.13.0+cu130, Transformers 5.16.1,
huggingface_hub 1.31.0, datasets 5.0.1, and openai 3.10.0.
Docker image: `vllm/vllm-openai:v0.29.0`. Exact loaded model revisions,
request options, prompt recipes, token usage, and hardware are recorded in the
outputs and job metadata.

## Artifacts and validation

Private bucket: `small-models-for-glam/glam-extraction-results`.
Batch prefix: `harvard-botany-headers/2026-09-15-v1/`. Each table label has a
completed `.json`, retained `.json.partial`, `.launch.json`, and `.job.json`.
The Gemma runtime source is retained under `recipes/jobs-source.tar.gz`;
the other five used `recipes/jobs-source-initial.tar.gz`.

Dataset: `small-models-for-glam/glam-extraction-benchmark`, config
`harvard-botany-headers`, split `test`, revision
`48e92ea1112d3fc2a32467e875d64e10129c4d46`. Existing NLS inputs and results
remain pinned to their earlier revision.

All six completed JSON files passed `glam_bench/validate_submission.py` with
30 items. Ruff and Python compilation passed for the generalized recipe.

| Model | Job |
|---|---|
| Qwen3.5-2B | [6aa9287c5527934177ee47ba](https://huggingface.co/jobs/davanstrien/6aa9287c5527934177ee47ba) |
| Qwen3.5-4B | [6aa92a11f76d6a098a70d387](https://huggingface.co/jobs/davanstrien/6aa92a11f76d6a098a70d387) |
| NuExtract-3 | [6aa92a24f76d6a098a70d38b](https://huggingface.co/jobs/davanstrien/6aa92a24f76d6a098a70d38b) |
| Granite-Vision-4.1-4B | [6aa92a495527934177ee4829](https://huggingface.co/jobs/davanstrien/6aa92a495527934177ee4829) |
| gemma-4-E4B | [6aa92ffdf76d6a098a70d41e](https://huggingface.co/jobs/davanstrien/6aa92ffdf76d6a098a70d41e) |
| Qwen3-VL-8B | [6aa92ca55527934177ee489f](https://huggingface.co/jobs/davanstrien/6aa92ca55527934177ee489f) |
| Cancelled Gemma hardware queue | [6aa92c715527934177ee4890](https://huggingface.co/jobs/davanstrien/6aa92c715527934177ee4890) |
