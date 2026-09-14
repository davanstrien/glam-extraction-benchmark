"""Admission checks and schema projection for collections with unrelated fields."""
import io
import json

import pytest
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

from dataset_contract import json_text, scoring_schema, sha256, validate_config
from scorer import score


def local_dataset(tmp_path, *, gold=None, schema=None, ids=("one",)):
    schema = schema or {"type": "object", "properties": {
        "title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "accession": {"type": "string", "x-match": "exact"}}}
    gold = gold if gold is not None else {"title": None, "accession": "AB-12"}
    stream = io.BytesIO()
    PILImage.new("RGB", (4, 4), "white").save(stream, format="PNG")
    rows = [{"id": item_id, "image": {"bytes": stream.getvalue(), "path": "example.png"},
             "target_schema": json_text(schema), "expected_output": json_text(gold)} for item_id in ids]
    config = tmp_path / "museum-objects"
    config.mkdir()
    features = Features({"id": Value("string"), "image": Image(), "target_schema": Value("string"),
                         "expected_output": Value("string")})
    data = config / "test.parquet"
    Dataset.from_list(rows, features=features).to_parquet(data)
    manifest = {"config": config.name, "split": "test", "contract_version": "1", "item_count": len(rows),
                "benchmark_id": "small-models-for-glam/glam-extraction-benchmark",
                "task_instructions": "Extract this museum object.",
                "scoring": {"id": "typed-kie", "version": "2026-08-26a",
                            "schema_projection": "nullable-to-value-v1", "exclude": []},
                "data_file": "museum-objects/test.parquet", "data_sha256": sha256(data.read_bytes()),
                "target_schema_sha256": sha256(json_text(schema).encode()),
                "item_ids_sha256": sha256(json_text(list(ids)).encode())}
    (config / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_unrelated_collection_and_nullable_gold(tmp_path):
    root = local_dataset(tmp_path)
    assert validate_config(root, "museum-objects")["valid"]


def test_gold_type_error_is_not_treated_as_valid_json(tmp_path):
    root = local_dataset(tmp_path, gold={"accession": 42})
    report = validate_config(root, "museum-objects")
    assert not report["valid"]
    assert any("$.accession" in error for error in report["errors"])


def test_duplicate_ids_are_refused(tmp_path):
    root = local_dataset(tmp_path, ids=("same", "same"))
    assert any("duplicate id" in error for error in validate_config(root, "museum-objects")["errors"])


def test_invalid_schema_is_reported(tmp_path):
    root = local_dataset(tmp_path, schema={"type": "object", "required": "not-an-array"})
    report = validate_config(root, "museum-objects")
    assert not report["valid"]
    assert any("SchemaError" in error for error in report["errors"])


def test_corrupted_parquet_hash_is_refused(tmp_path):
    root = local_dataset(tmp_path)
    path = root / "museum-objects/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["data_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    assert any("Parquet checksum" in error for error in validate_config(root, "museum-objects")["errors"])


def test_nullable_projection_preserves_matching_annotations_and_input():
    schema = {"$defs": {"record": {"type": "object", "properties": {
        "reference": {"anyOf": [{"type": "string"}, {"type": "null"}], "x-match": "exact"}}}},
        "type": "object", "properties": {"records": {"type": "array", "items": {"$ref": "#/$defs/record"}}}}
    before = json_text(schema)
    projected = scoring_schema(schema)
    assert json_text(schema) == before
    assert projected["properties"]["records"]["items"]["properties"]["reference"]["x-match"] == "exact"
    gold = {"records": [{"reference": "AB-12"}]}
    prediction = {"records": [{"reference": "AB-13"}]}
    assert score(gold, prediction, projected)["n_ident_wrong"] == 1


@pytest.mark.parametrize("schema", [
    {"$ref": "https://example.com/schema.json"},
    {"type": "object", "properties": {"child": {"$ref": "#/properties/child"}}},
    {"anyOf": [{"type": "string", "pattern": "[A-Z]+"}, {"type": "null"}]},
    {"oneOf": [{"type": "string"}, {"type": "integer"}]},
])
def test_unsupported_schema_never_silently_scores(schema):
    with pytest.raises(ValueError):
        scoring_schema(schema)
