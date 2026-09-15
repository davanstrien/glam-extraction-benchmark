"""The same config works for inference preparation, validation and a static page."""
import json
from types import SimpleNamespace

import pytest

import harness
import validate_submission
from build_space import build
from dataset_contract import BENCHMARK_ID, read_config
from test_dataset_contract import local_dataset


def collection(tmp_path):
    root = local_dataset(tmp_path)
    path = root / "museum-objects/manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(title="Museum objects", institution={"name": "Example museum"},
                    source={"repo_id": "museum/objects", "revision": "b" * 40},
                    license="cc0-1.0", label_production={"draft_model": "Example model", "review": "Museum staff"})
    path.write_text(json.dumps(manifest))
    rows, manifest = read_config(root, "museum-objects")
    return root, list(rows), manifest


def test_config_preparation_preserves_nullable_schema_and_validates(tmp_path, monkeypatch):
    root, rows, manifest = collection(tmp_path)
    loader = lambda *_args: (rows, manifest, "a" * 40)
    monkeypatch.setattr(harness, "load_hub_config", loader)
    monkeypatch.setattr(validate_submission, "load_hub_config", loader)
    items, identity, subdir = harness.prepare_dataset(SimpleNamespace(
        dataset=BENCHMARK_ID, config="museum-objects", dataset_revision="main", limit=None))
    assert subdir == "museum-objects"
    assert identity["config"] == "museum-objects" and identity["split"] == "test"
    assert identity["inference_revision"] == "a" * 40
    assert "anyOf" in json.loads(items[0]["target_schema"])["properties"]["title"]
    assert items[0]["scoring_schema"]["properties"]["title"]["type"] == "string"
    assert validate_submission.gold_for({"dataset": identity}, None) == ({"one"}, "a" * 40, None)


def test_static_build_scores_another_collection_without_inference(tmp_path):
    root, rows, manifest = collection(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    document = {"model": {"id": "org/example", "label": "example", "params": "1B"},
                "dataset": {"id": BENCHMARK_ID, "config": "museum-objects", "split": "test",
                            "inference_revision": "a" * 40},
                "items": [{"id": "one", "prediction": rows[0]["expected_output"]}]}
    (results / "example.json").write_text(json.dumps(document))
    artifact = build(root, "museum-objects", "a" * 40, results, tmp_path / "space")
    assert artifact["rows"][0]["content_f1"] == 1
    assert artifact["benchmark"]["config"] == "museum-objects"
    assert not artifact["legacy_protocol"]
    assert (tmp_path / "space/example.jpg").is_file()
    assert "Museum objects" in (tmp_path / "space/index.html").read_text()
    document["dataset"]["config"] = "another-collection"
    (results / "example.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="config differs"):
        build(root, "museum-objects", "a" * 40, results, tmp_path / "wrong-space")


def test_collection_pages_keep_config_identity_and_relative_navigation(tmp_path):
    from build_collection_space import build_collection

    root, rows, _manifest = collection(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    document = {"model": {"id": "org/example", "label": "example", "params": "1B"},
                "dataset": {"id": BENCHMARK_ID, "config": "museum-objects", "split": "test",
                            "inference_revision": "a" * 40},
                "items": [{"id": "one", "prediction": rows[0]["expected_output"]}]}
    (results / "example.json").write_text(json.dumps(document))
    import shutil
    second = root / "botany-headers"
    shutil.copytree(root / "museum-objects", second)
    manifest = json.loads((second / "manifest.json").read_text())
    manifest.update(config="botany-headers", data_file="botany-headers/test.parquet",
                    label_production={"draft": "Visual assistant draft",
                                      "printed_fields": "Human reviewed",
                                      "corrections": "Replacement strings human reviewed"})
    (second / "manifest.json").write_text(json.dumps(manifest))
    second_results = tmp_path / "second-results"
    second_results.mkdir()
    document["dataset"].update(config="botany-headers", inference_revision="c" * 40)
    (second_results / "example.json").write_text(json.dumps(document))
    output = tmp_path / "space"
    artifacts = build_collection([
        {"root": root, "config": "museum-objects", "revision": "a" * 40, "results": results},
        {"root": root, "config": "botany-headers", "revision": "c" * 40, "results": second_results},
    ], output)
    assert [x["benchmark"]["revision"] for x in artifacts] == ["a" * 40, "c" * 40]
    main = (output / "index.html").read_text()
    nested = (output / "botany-headers/index.html").read_text()
    assert '<option data-config="botany-headers" value="botany-headers/index.html">botany-headers</option>' in main
    assert '<option data-config="museum-objects" value="../index.html">museum-objects</option>' in nested
    assert '<option data-config="botany-headers" value="../botany-headers/index.html" selected>' in nested
    assert "Replacement strings human reviewed" in nested
    assert (output / "botany-headers/example.jpg").is_file()
    assert len(json.loads((output / "configs.json").read_text())) == 2

    overall = json.loads((output / "overall-scores.json").read_text())
    assert overall["aggregation"]["dataset_weights"] == {"museum-objects": .5, "botany-headers": .5}
    assert overall["rows"][0]["content_f1"] == 1
    assert '<option data-config="overall" value="overall.html">Overall</option>' in main
    assert '<option data-config="overall" value="../overall.html">Overall</option>' in nested
    assert 'data-f1-column="1"' in (output / "overall.html").read_text()
