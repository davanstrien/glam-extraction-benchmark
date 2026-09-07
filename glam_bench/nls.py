"""NLS index-cards-eval loader: first institution-verified gold set for the benchmark.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import result_io

NLS_DATASET = "NationalLibraryOfScotland/index-cards-eval"

# Fields whose values are identifiers -> exact match. Paths into the canonical schema.
EXACT_OVERLAY = [
    ("entries", "items", "ms_no"),
    ("entries", "items", "folios", "items"),
]


def _canonicalize(node, defs):
    """Dereference $refs and collapse anyOf[X, null] -> X (Pydantic Optional)."""
    if isinstance(node, list):
        return [_canonicalize(n, defs) for n in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        target = defs[node["$ref"].split("/")[-1]]
        merged = {**copy.deepcopy(target), **{k: v for k, v in node.items() if k != "$ref"}}
        return _canonicalize(merged, defs)
    if "anyOf" in node:
        non_null = [o for o in node["anyOf"] if o.get("type") != "null"]
        if len(non_null) == 1:
            merged = {**copy.deepcopy(non_null[0]), **{k: v for k, v in node.items() if k != "anyOf"}}
            return _canonicalize(merged, defs)
    return {k: _canonicalize(v, defs) for k, v in node.items() if k != "$defs"}


def _apply_exact_overlay(schema):
    for path in EXACT_OVERLAY:
        node = schema
        for step in path:
            node = node.get("properties", {}).get(step) if step != "items" else node.get("items", {})
            if not node:
                break
        else:
            if node.get("type") == "string":
                node["x-match"] = "exact"


def resolve_revision(revision: str = "main") -> str:
    """A branch/tag/sha -> the immutable commit SHA it points at."""
    from huggingface_hub import HfApi

    return HfApi().dataset_info(NLS_DATASET, revision=revision).sha


def _schema_at(revision: str) -> tuple[str, list[str]]:
    """Schema + gold field names read at an already-resolved commit SHA."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(NLS_DATASET, "schema.json", repo_type="dataset", revision=revision)
    raw = json.loads(Path(path).read_text())
    schema = _canonicalize(raw, raw.get("$defs", {}))
    _apply_exact_overlay(schema)
    return json.dumps(schema, ensure_ascii=False), list(schema["properties"].keys())



def load_nls(limit: int | None = None, revision: str = "main") -> tuple[list[dict], str]:
    """-> ([{id, image, target_schema (canonical JSON Schema str), gold (dict)}], commit SHA)"""
    from datasets import load_dataset

    sha = resolve_revision(revision)
    schema_str, fields = _schema_at(sha)
    ds = load_dataset(NLS_DATASET, split="train", revision=sha)
    rows = []
    for r in ds:
        gold = {f: r[f] for f in fields}
        rows.append({"id": r["_sample_id"], "image": r["image"],
                     "target_schema": schema_str, "gold": gold})
    # Checked over the whole split, before `--limit`: a duplicate past the limit would let a
    # two-item smoke test pass and only fail the full run, after it had been paid for.
    result_io.refuse_duplicate_ids([r["id"] for r in rows], f"{NLS_DATASET} train split")
    if limit:
        rows = rows[:limit]
    return rows, sha


def load_gold(revision: str = "main") -> tuple[str, dict[str, dict], str]:
    """Labels only, no pixels -> (schema str, {sample_id: {...}}, resolved commit SHA)."""
    from huggingface_hub import hf_hub_download

    sha = resolve_revision(revision)
    schema_str, fields = _schema_at(sha)
    meta_path = hf_hub_download(NLS_DATASET, "metadata.jsonl", repo_type="dataset", revision=sha)
    meta = Path(meta_path).read_text()
    gold = {}
    sample_ids = []
    for line in meta.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sample_ids.append(r["_sample_id"])
        gold[r["_sample_id"]] = {"gold": {f: r[f] for f in fields},
                                 "label_status": r["_label_status"], "verdict": r["_verdict"],
                                 "image_type": r["image_type"]}
    # `gold` is keyed by id, so a repeated id would silently overwrite the first record and the
    # dict would come back one row short of the export it was built from. The ids are collected
    # in file order for that reason: only the list can still see the duplicate.
    result_io.refuse_duplicate_ids(sample_ids, f"{NLS_DATASET} metadata.jsonl")
    return schema_str, gold, sha
