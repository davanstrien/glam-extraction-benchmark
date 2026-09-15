# Optional NLS Jobs batch

`jobs_batch.py` is an example prediction producer, separate from the benchmark
harness. It starts a local vLLM server inside an HF Job, calls it, writes the
submission format using harness helpers, and uploads each checkpoint. The server
is not exposed publicly and exits with the worker. Other participants can produce
the same submission JSON however they prefer.

## Launch

From the repository root, with an HF token that can run Jobs and write to the results bucket:

```bash
GLAM_JOBS_NAMESPACE=davanstrien uv run --with huggingface_hub==1.31.0 \
  examples/inference/jobs_batch.py launch --model Qwen3.5-2B
```

Start with this model, check its logs and first three predictions, then launch
other labels from `--help`. Each job has a 60-minute timeout on `l40sx1` ($1.80/hour
at the 2026-09-15 price). Six such jobs allocate at most $10.80 before retries; the
Jobs batch allocation is $15. Check the remaining budget before retrying. The
namespace is explicit. Output storage stays
in the organization bucket regardless of billing namespace.

The worker checks its first three responses for parse failures and truncation
before continuing through the remaining cards. Low extraction scores themselves
do not stop it. Generation uses 1,600 output tokens, temperature zero, and thinking
off where supported. NuExtract receives its native extraction template and field
instructions; Granite receives its model-card KVP prompt; other models receive the
reference VLM prompt. Per-row provenance records the prompt recipe and actual
request options. These are prompt adapters, not changes to scoring.

## Inputs and outputs

- Dataset: `small-models-for-glam/glam-extraction-benchmark`
- Revision: `ecc9c02582f933ca89b73e6751e4ed76888cb24d`
- Config/split: `nls-index-cards` / `test`, 98 cards
- Model revisions: fixed in `jobs_batch.py`, passed to vLLM's `--revision`
- Serving image: `vllm/vllm-openai:v0.29.0`
- Authorized billing namespace: `davanstrien`
- Private output bucket: `small-models-for-glam/glam-extraction-results`
- Batch prefix: `nls-index-cards/2026-09-15-nullable-v1/`

Each model writes `<label>.json.partial` after every card, `<label>.json` on
completion, `<label>.launch.json` with its Job ID and scheduled cost bound, and
`<label>.job.json` with actual library versions and timestamps. Per-card usage
comes from vLLM. A timeout or failed request may leave only a partial file; never
publish that as a completed run.

For execution, the launcher stages the seven required Python modules (`harness`,
`result_io`, `dataset_contract`, `schema`, `scorer`, `models`, `version`) plus this
recipe under `recipes/jobs-source.tar.gz`. No data, results, environment files,
credentials, git history or private notes are included. HF credentials are passed
as a Jobs secret. The token needs write access to the private bucket and access to run Jobs in the selected namespace.

Download completed JSON and pass it to `glam_bench/validate_submission.py` before
building the Space. An added dataset config does not change these pinned NLS
predictions. Use another run ID for future reruns to retain earlier outputs.

## Status on 2026-09-15

No model runs have completed with this recipe yet.

Ruff, Python compilation, CLI help, all six prompt builders against the pinned
local export, and malformed/truncated smoke-stop checks passed. GPU serving and
end-to-end bucket checkpointing remain unverified; these notes do not claim a
successful model run.

Model-specific sources:

- [NuExtract template and vision API usage](https://huggingface.co/numind/NuExtract3)
- [Granite KVP prompt and native vLLM support](https://huggingface.co/ibm-granite/granite-vision-4.1-4b)
- [Gemma image and thinking configuration](https://huggingface.co/google/gemma-4-E4B-it)
