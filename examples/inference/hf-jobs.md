# Optional HF Jobs serving example

This is the existing `lift-9B` serving example, moved out of the main benchmark
walkthrough. It is not the launch configuration for the proposed new model batch.
Hardware, serving versions and model options need checking for each selected model.

## Start a server

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

## Use the endpoint

Pass the resulting `/v1` URL to the reference runner's `--base-url LABEL=URL`
option, as shown in the [runner walkthrough](../../README.md#2-smoke-test-two-items).
The same option works with another OpenAI-compatible server. Stop the serving Job
when inference is finished; starting a server alone does not save benchmark outputs.

For a batch Job that also executes inference, the recipe must persist local result
files before the Job exits. The planned bucket upload belongs in that recipe,
outside `glam_bench/`. See [output storage](README.md#our-run-outputs).
