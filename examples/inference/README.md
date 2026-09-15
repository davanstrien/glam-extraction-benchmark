# Optional inference recipes

This directory records how we generate benchmark predictions. It is separate from
`glam_bench/`: scoring and submission validation do not import these recipes or
require HF Jobs, a particular serving engine, or our results bucket.

You can use a local model, an API, HF Jobs or your own scripts. The shared interface
is the [submission JSON format](../../docs/SUBMISSION.md), which the benchmark
validates and scores. The existing harness is a convenient reference producer.

## What belongs here

Keep launch scripts, serving configurations, dependency versions and short run
notes alongside the recipe they describe. Scripts can declare their own dependencies
(for example with UV script metadata), without adding Jobs tooling to the harness.
Record the command, model revision when known, hardware or API provider, generation
settings and output location for completed runs. Actual recipes and run notes will
be added as the selected batch is executed; this directory does not claim those
runs have happened.

- [HF Jobs serving example](hf-jobs.md): the existing vLLM launch example.
- Future API and batch Jobs scripts belong here too; neither becomes a required
  benchmark workflow.

## Our run outputs

The private bucket `small-models-for-glam/glam-extraction-results` is the planned
home for this project's predictions and checkpoints, for example:

```text
nls-index-cards/<run-id>/<model-label>.json
```

Use a distinct run directory for each batch so reruns keep their earlier outputs.
The bucket exists, but automated uploads are not implemented yet. The harness
currently writes local `results/<config>/` files; validation and Space builds read
local files too. Recipes will handle upload/download around those existing tools.
The Space publishes derived scores. Other submitters can deliver the same JSON
without using our bucket.
