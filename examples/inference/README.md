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
settings and output location for completed runs. Run notes distinguish completed
predictions from prepared or interrupted runs.

- [HF Jobs serving example](hf-jobs.md): the existing vLLM launch example.
- [NLS API runs](api-2026-09-15.md) and [Harvard API runs](api-harvard-2026-09-15.md):
  four complete models per config, using [the optional producer](api_batch.py).
- [Batch Jobs recipe](jobs-batch.md), [NLS GPU runs](jobs-2026-09-15.md) and
  [Harvard GPU runs](jobs-harvard-2026-09-15.md): six complete models per config,
  using [the optional worker and launcher](jobs_batch.py).

## Our run outputs

The private bucket `small-models-for-glam/glam-extraction-results` is the
home for this project's predictions and checkpoints, for example:

```text
nls-index-cards/<run-id>/<model-label>.json
```

Use a distinct run directory for each batch so reruns keep their earlier outputs.
The API recipe uploads checkpoints and completed files to this bucket. The Jobs
recipe also uploads checkpoints and completed results. The harness
writes local `results/<config>/` files; validation and Space builds read local
files too. Recipes handle storage around those existing tools.
The Space publishes derived scores. Other submitters can deliver the same JSON
without using our bucket.
