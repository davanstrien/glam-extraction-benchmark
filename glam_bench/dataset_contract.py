"""Local benchmark data contract. No institution-specific loading or model calls.

The target schema retains JSON Schema nullability. The existing scorer receives a
separate projection; never send that projection to a model as the task schema.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import tempfile
from pathlib import Path

CONTRACT_VERSION = "1"
REQUIRED_COLUMNS = {"id", "image", "target_schema", "expected_output"}
BENCHMARK_ID = "small-models-for-glam/glam-extraction-benchmark"


def require_config_name(config):
    if not isinstance(config, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", config):
        raise ValueError("config must be a simple name containing letters, numbers, hyphens or underscores")


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes())


def local_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"file must be inside the dataset directory: {relative!r}")
    return path


def _resolve_ref(node, root, visited):
    ref = node["$ref"]
    if not isinstance(ref, str) or not ref.startswith("#/") or ref in visited:
        raise ValueError(f"unsupported external or recursive schema reference: {ref!r}")
    target = root
    for token in ref[2:].split("/"):
        target = target[token.replace("~1", "/").replace("~0", "~")]
    # Siblings to $ref have intersection semantics, not override semantics.
    siblings = {k: v for k, v in node.items() if k != "$ref"}
    if any(k not in {"title", "description", "default", "x-match"} for k in siblings):
        raise ValueError("only annotations are supported alongside $ref")
    return {**copy.deepcopy(target), **siblings}, visited | {ref}


def _nonnull(node):
    if "anyOf" not in node:
        return node
    branches = node["anyOf"]
    nonnull = [branch for branch in branches if branch != {"type": "null"}]
    if len(branches) != 2 or len(nonnull) != 1:
        raise ValueError("the scorer supports only anyOf[value, null]")
    siblings = {k: v for k, v in node.items() if k != "anyOf"}
    if any(k not in {"title", "description", "default", "x-match"} for k in siblings):
        raise ValueError("only annotations are supported alongside nullable anyOf")
    return {**nonnull[0], **siblings}


def scoring_schema(schema):
    """Project a supported nullable schema into the existing scorer's representation."""
    allowed = {"$schema", "$defs", "$ref", "type", "properties", "items", "anyOf",
               "enum", "required", "default", "title", "description", "x-match"}

    def visit(node, visited=frozenset()):
        if "$ref" in node:
            node, visited = _resolve_ref(node, schema, visited)
        node = _nonnull(node)
        unknown = set(node) - allowed
        if unknown:
            raise ValueError(f"unsupported scoring schema keywords: {sorted(unknown)}")
        kind = node.get("type")
        if kind not in {"object", "array", "string", "integer", "number", "boolean"}:
            raise ValueError(f"unsupported scoring schema type: {kind!r}")
        if node.get("x-match") not in {None, "exact"}:
            raise ValueError("x-match must be exact when provided")
        out = {k: copy.deepcopy(v) for k, v in node.items() if k not in {"$defs", "$schema"}}
        if kind == "object":
            out["properties"] = {k: visit(v, visited) for k, v in node.get("properties", {}).items()}
        if kind == "array":
            out["items"] = visit(node["items"], visited)
        return out

    return visit(schema)


def read_config(root, config):
    """Read a config from the local export, using embedded image bytes without decoding."""
    from datasets import Dataset, Image
    import pyarrow.parquet as pq

    require_config_name(config)
    manifest_path = local_file(root, f"{config}/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["config"] != config or manifest["contract_version"] != CONTRACT_VERSION:
        raise ValueError("config identity or contract version mismatch")
    dataset = Dataset(pq.read_table(local_file(root, manifest["data_file"])))
    return dataset.cast_column("image", Image(decode=False)), manifest


def load_hub_config(repo_id, config, revision):
    """Resolve once, validate the pinned export, and return in-memory rows plus identity."""
    from huggingface_hub import HfApi, snapshot_download

    require_config_name(config)
    sha = HfApi().dataset_info(repo_id, revision=revision).sha
    with tempfile.TemporaryDirectory(prefix="glam-config-") as folder:
        snapshot_download(repo_id, repo_type="dataset", revision=sha, local_dir=folder,
                          allow_patterns=[f"{config}/*"])
        report = validate_config(folder, config)
        if not report["valid"]:
            raise ValueError(json.dumps(report))
        dataset, manifest = read_config(folder, config)
        if manifest["benchmark_id"] != repo_id:
            raise ValueError("manifest benchmark ID differs from requested repository")
        rows = list(dataset)
    return rows, manifest, sha


def inference_items(rows, manifest, limit=None):
    """Keep target schema for models; project it separately for the scorer."""
    from PIL import Image

    items = []
    for row in rows[:limit]:
        with Image.open(io.BytesIO(row["image"]["bytes"])) as image:
            pixels = image.copy()
        items.append({"id": row["id"], "image": pixels, "target_schema": row["target_schema"],
                      "gold": json.loads(row["expected_output"]),
                      "scoring_schema": scoring_schema(json.loads(row["target_schema"])),
                      "scoring_exclude": tuple(manifest["scoring"]["exclude"])})
    return items


def evaluation_identity(manifest):
    """The config within a pinned Hub snapshot."""
    return {"config": manifest["config"], "split": manifest["split"]}


def _check_row(row, index, seen, expected_schema_hash):
    from jsonschema import Draft202012Validator
    from PIL import Image

    errors = []
    item_id = row.get("id")
    if not isinstance(item_id, str) or not item_id or item_id in seen:
        errors.append(f"row {index}: empty or duplicate id {item_id!r}")
    seen.add(item_id)
    schema = json.loads(row["target_schema"])
    gold = json.loads(row["expected_output"])
    scoring_schema(schema)  # rejects unsupported schemas before any references are resolved
    Draft202012Validator.check_schema(schema)
    for error in Draft202012Validator(schema).iter_errors(gold):
        errors.append(f"{item_id}: {error.json_path}: {error.message}")
    if sha256(row["target_schema"].encode()) != expected_schema_hash:
        errors.append(f"{item_id}: target schema differs from manifest")
    data = row["image"]["bytes"]
    if not data:
        errors.append(f"{item_id}: image bytes must be embedded")
    else:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        if "provenance" in row:
            provenance = json.loads(row["provenance"])
            if provenance.get("image_sha256") != sha256(data):
                errors.append(f"{item_id}: image checksum differs from provenance")
    return errors


def validate_config(root, config):
    """Strict local admission checks. Does not certify human label accuracy."""
    from jsonschema.exceptions import SchemaError

    rows, manifest = read_config(root, config)
    errors = []
    missing = REQUIRED_COLUMNS - set(rows.column_names)
    if missing:
        return {"valid": False, "errors": [f"missing columns: {sorted(missing)}"]}
    if len(rows) != manifest["item_count"]:
        errors.append("item count differs from manifest")
    if file_hash(local_file(root, manifest["data_file"])) != manifest["data_sha256"]:
        errors.append("Parquet checksum differs from manifest")
    seen = set()
    for index, row in enumerate(rows):
        try:
            errors.extend(_check_row(row, index, seen, manifest["target_schema_sha256"]))
        except (ValueError, KeyError, TypeError, OSError, SchemaError) as exc:
            errors.append(f"row {index}: {type(exc).__name__}: {exc}")
    ids = list(rows["id"])
    if sha256(json_text(ids).encode()) != manifest["item_ids_sha256"]:
        errors.append("ordered item IDs differ from manifest")
    return {"valid": not errors, "config": config, "split": manifest["split"],
            "rows": len(rows), "errors": errors}
