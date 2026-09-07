"""Model registry: how to run each model + display metadata (params, cost).

`kind` selects the adapter in harness.py: `nuextract_space`, `specialist_space` (Gradio Spaces),
`router_vlm` (HF Inference Providers), `openai` (any OpenAI-compatible /v1 endpoint).

No endpoint URLs live here. A `kind:"openai"` entry names the model and the env var holding the
key; the URL comes from `--base-url LABEL=URL` or `GLAM_BASE_URL_<LABEL>` at run time:

    uv run glam_bench/harness.py --models lift-9B \
      --base-url lift-9B=https://<job-id>--8000.hf.jobs/v1

`enabled: False` keeps an entry documented but out of `--models all`; naming it explicitly is
refused too.
"""

MODELS = [
    {"label": "NuExtract-3", "kind": "nuextract_space",
     "id": "numind/NuExtract3", "params": "4B", "cost": "self-host"},
    # Never run in v1: no result file exists for it, so it stays out of `--models all`.
    {"label": "Qwen3-VL-8B", "kind": "router_vlm", "enabled": False,
     "id": "Qwen/Qwen3-VL-8B-Instruct", "provider": None, "params": "8B", "cost": "$0.08/M"},
    {"label": "gemma-3-4b", "kind": "router_vlm",
     "id": "google/gemma-3-4b-it", "provider": None, "params": "4B", "cost": "$0.05/M"},
    # Never run in v1: no result file exists for it, so it stays out of `--models all`.
    {"label": "Qwen2.5-VL-72B", "kind": "router_vlm", "enabled": False,
     "id": "Qwen/Qwen2.5-VL-72B-Instruct", "provider": None, "params": "72B", "cost": "$1.01/M"},
    {"label": "Llama-4-Scout", "kind": "router_vlm",
     "id": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "provider": None,
     "params": "109B (17B act)", "cost": "$0.09/M"},
    {"label": "Qwen3-VL-235B", "kind": "router_vlm",
     "id": "Qwen/Qwen3-VL-235B-A22B-Instruct", "provider": None,
     "params": "235B (22B act)", "cost": "$0.30/M"},
    {"label": "gemma-4-31B", "kind": "router_vlm",
     "id": "google/gemma-4-31B-it", "provider": None, "params": "31B", "cost": "router"},
    {"label": "Qwen3.6-27B", "kind": "router_vlm",  # same family as NLS's labelling model (Qwen3.6-35B-A3B)
     "id": "Qwen/Qwen3.6-27B", "provider": None, "params": "27B", "cost": "router"},
    {"label": "Qwen3.5-9B", "kind": "router_vlm",
     "id": "Qwen/Qwen3.5-9B", "provider": None, "params": "9B", "cost": "router"},
    {"label": "gemma-4-26B-A4B", "kind": "router_vlm",
     "id": "google/gemma-4-26B-A4B-it", "provider": None, "params": "26B (4B act)", "cost": "router"},
    # Disabled: router providers ignore enable_thinking=false -> 32/98 truncated (2026-09-02); needs a Jobs serve
    {"label": "GLM-4.6V-Flash", "kind": "router_vlm", "enabled": False,
     "id": "zai-org/GLM-4.6V-Flash", "provider": None, "params": "flash", "cost": "router"},
    # Disabled: vLLM engine init failed on a100-large (2026-09-02); retry with a newer vllm image
    {"label": "Qwen3.6-35B-A3B", "kind": "router_vlm", "enabled": False,  # = NLS's labelling model (GGUF Q8 of this)
     "id": "Qwen/Qwen3.6-35B-A3B", "provider": None, "params": "35B (3B act)", "cost": "router"},
    # Added 2026-09-02 from the trending <=32B image-text-to-text list (warm on Inference Providers).
    {"label": "Qwen3.8-27B", "kind": "router_vlm",
     "id": "Qwen/Qwen3.8-27B", "provider": None, "params": "27.8B", "cost": "router"},
    # Disabled: thinking not switchable off on the router -> 97/98 truncated (2026-09-02)
    {"label": "Muse-Glimmer-30B", "kind": "router_vlm", "enabled": False,
     "id": "meta-models/Muse-Glimmer-30B", "provider": None, "params": "29.8B", "cost": "router"},
    # Disabled: router auto has no provider for it (featherless-only) (2026-09-02)
    {"label": "Qwen3.5-4B", "kind": "router_vlm", "enabled": False,
     "id": "Qwen/Qwen3.5-4B", "provider": None, "params": "4.7B", "cost": "router"},
    # Disabled: router auto has no provider for it (featherless-only) (2026-09-02)
    {"label": "Qwen3.5-2B", "kind": "router_vlm", "enabled": False,
     "id": "Qwen/Qwen3.5-2B", "provider": None, "params": "2.3B", "cost": "router"},
    # Disabled: router auto has no provider for it (featherless-only) (2026-09-02)
    {"label": "Qwen3-VL-4B", "kind": "router_vlm", "enabled": False,
     "id": "Qwen/Qwen3-VL-4B-Instruct", "provider": None, "params": "4.4B", "cost": "router"},
    {"label": "Apertus-v1.5-8B", "kind": "router_vlm",
     "id": "swiss-ai/Apertus-v1.5-8B", "provider": None, "params": "8.9B", "cost": "router"},
    # LFM2.5-VL-Extract (LiquidAI extraction specialist) — serve via Jobs (vLLM >=0.23, greedy),
    # then pass the exposed URL with --base-url. See README, "Run a model".
    {"label": "LFM2.5-VL-1.6B-Extract", "kind": "openai", "id": "LiquidAI/LFM2.5-VL-1.6B-Extract",
     "api_key_env": "HF_TOKEN", "params": "1.6B", "cost": "self-host"},
    # GLM-OCR (Zhipu, OCR/extraction specialist) — serve via Jobs + vLLM, same path as the two above.
    # Disabled: dumps OCR text, 53/98 truncated (2026-09-02); needs its schema/KIE mode, not the chat prompt
    {"label": "GLM-OCR", "kind": "openai", "enabled": False, "id": "zai-org/GLM-OCR",
     "api_key_env": "HF_TOKEN", "params": "1.3B", "cost": "self-host"},
    # Fine-tuned specialist — available, off by default (out-of-domain on forms/bibliographic):
    # {"label": "index-card-extractor", "kind": "specialist_space",
    #  "id": "small-models-for-glam/index-card-extractor", "params": "4B", "cost": "self-host"},

    # Any OpenAI-compatible /v1 endpoint (kind="openai") — local vLLM, OpenAI, OpenRouter, ... The
    # endpoint is NOT part of the entry; pass it at run time (see the module docstring): {"label":
    # "local-vllm", "kind": "openai", "id": "Qwen/Qwen2.5-VL-7B-Instruct", "api_key_env":
    # "VLLM_KEY", "params": "7B", "cost": "self-host"}, --base-url
    # local-vllm=http://localhost:8000/v1 {"label": "gpt-4o", "kind": "openai", "id": "gpt-4o",
    # "api_key_env": "OPENAI_API_KEY", "params": "?", "cost": "api"}, --base-url
    # gpt-4o=https://api.openai.com/v1

    # lift (Datalab, 9B, schema-constrained) — not on Inference Providers; serve it via Jobs
    # first: hf jobs run --detach --expose 8000 --flavor a100-large -s HF_TOKEN \ vllm/vllm-openai
    # vllm serve datalab-to/lift --max-model-len 32768 then pass the exposed-port URL + /v1 with
    # --base-url, and use guided decoding:
    {"label": "lift-9B", "kind": "openai", "id": "datalab-to/lift", "guided": True,
     "api_key_env": "HF_TOKEN", "params": "9B", "cost": "self-host"},
]

# The 7 typed-core POC items (clean, single, typed/printed). Handwritten + multi-card items
# in the dataset are deferred to v2.
TYPED_CORE = [
    "roots-copyright-form-loc-A783281", "roots-copyright-form-loc-A783281-p2",
    "bpl_clean_a", "bpl_clean_b", "rubenstein", "newspaper_peabody", "bio_parisian",
]

DATASET = "davanstrien/glam-extraction-bench-poc"  # POC (silver labels)
