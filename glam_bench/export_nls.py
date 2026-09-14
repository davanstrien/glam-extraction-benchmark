# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=5,<6", "huggingface_hub>=1.29,<2", "jsonschema>=4,<5", "Pillow>=12,<13"]
# ///
"""Export the pinned NLS source into the benchmark contract. No inference or uploads.

uv run glam_bench/export_nls.py --output /path/to/new-export
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path, PurePosixPath

from dataset_contract import file_hash, json_text, sha256, validate_config
from scorer import SCORER_VERSION
from version import __version__

SOURCE_ID = "NationalLibraryOfScotland/index-cards-eval"
SOURCE_REVISION = "2a81070549d8493c2c538744a9dbbc1dc72cb146"
BENCHMARK_ID = "small-models-for-glam/glam-extraction-benchmark"
CONFIG = "nls-index-cards"
CONVERTER_VERSION = "1"


def target_schema(source_schema):
    """Retain nullability and add only benchmark matching annotations."""
    schema = copy.deepcopy(source_schema)
    entry = schema["$defs"]["ManuscriptEntry"]["properties"]
    entry["ms_no"]["x-match"] = "exact"
    entry["folios"]["items"]["x-match"] = "exact"
    return schema


def make_rows(source, raw_schema):
    fields = list(raw_schema["properties"])
    schema_text = json_text(target_schema(raw_schema))
    records = [json.loads(line) for line in (source / "metadata.jsonl").read_text().splitlines() if line]
    rows = []
    for record in records:
        relative = PurePosixPath(record["file_name"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source image name must be relative to the pinned snapshot")
        # Hub snapshots use symlinks to their content-addressed blob cache.
        image_path = source / relative
        image_bytes = image_path.read_bytes()
        provenance = {key: value for key, value in record.items() if key.startswith("_")}
        provenance.update({"source_file": record["file_name"], "source_id": record["_sample_id"],
                           "source_dataset": SOURCE_ID, "source_revision": SOURCE_REVISION,
                           "image_sha256": sha256(image_bytes)})
        rows.append({"id": record["_sample_id"],
                     "image": {"bytes": image_bytes, "path": image_path.name},
                     "target_schema": schema_text,
                     "expected_output": json_text({key: record[key] for key in fields}),
                     "provenance": json_text(provenance)})
    return rows


def make_manifest(rows, data_path, source):
    return {
        "contract_version": "1", "benchmark_id": BENCHMARK_ID, "config": CONFIG, "split": "test",
        "title": "National Library of Scotland manuscript index cards",
        "institution": {"name": "National Library of Scotland", "url": "https://www.nls.uk/"},
        "source": {"repo_id": SOURCE_ID, "revision": SOURCE_REVISION, "config": "default",
                   "split": "train", "schema_sha256": file_hash(source / "schema.json"),
                   "metadata_sha256": file_hash(source / "metadata.jsonl")},
        "license": "cc0-1.0", "converter": {"name": "nls", "version": CONVERTER_VERSION},
        "harness_version": __version__, "item_count": len(rows),
        "item_ids_sha256": sha256(json_text([row["id"] for row in rows]).encode()),
        "target_schema_sha256": sha256(rows[0]["target_schema"].encode()),
        "data_file": f"{CONFIG}/test.parquet", "data_sha256": file_hash(data_path),
        "scoring": {"id": "typed-kie", "version": SCORER_VERSION, "exclude": ["notes"],
                    "schema_projection": "nullable-to-value-v1",
                    "exact_match": "NFKC, casefold, punctuation and whitespace normalization",
                    "null_policy": "absent values; missing/null booleans use declared defaults",
                    "array_alignment": "best-pair-first for objects; set credit for scalar arrays"},
        "label_production": {"draft_model": "Qwen3.6-35B-A3B-GGUF-Q8_0",
                             "review": "NLS cataloguers", "status_field": "provenance._label_status"},
        "reporting_slices": {"review_status": ["verified", "corrected"]},
        "task_instructions": "Extract the fields described by target_schema from the supplied image. "
                             "Return a JSON object. Use null or omit optional fields when absent.",
        "protocol_note": "This export preserves source nullability. Historical runs used a schema "
                         "that removed nullable branches; they are legacy-protocol results, not "
                         "new evaluations of this export. Exact prompt/solver settings belong to each run.",
        "registration": {"status": "not_registered", "proposed_task_id": CONFIG},
    }


def dataset_card():
    return f"""---
pretty_name: GLAM extraction benchmark
license: cc0-1.0
task_categories:
  - image-to-text
configs:
  - config_name: {CONFIG}
    data_files:
      - split: test
        path: {CONFIG}/test.parquet
---
# GLAM extraction benchmark

Structured extraction from cultural-heritage documents. The first configuration is
`{CONFIG}`: 98 manuscript catalogue cards from the National Library of Scotland.

## Source and credits

Derived from [{SOURCE_ID}](https://huggingface.co/datasets/{SOURCE_ID}/tree/{SOURCE_REVISION}),
revision `{SOURCE_REVISION}` (CC0). Images and checked outputs are preserved.
NLS cataloguers reviewed the model-drafted labels: 66 accepted as drafted, 32 corrected.
Drafting model and original review metadata are preserved in `provenance`.
The original source remains maintained by the National Library of Scotland.

## Format

`id`, `image`, `target_schema`, `expected_output` are the common benchmark columns.
Schema and expected output are JSON strings. `provenance` is a JSON string containing
source IDs, revision, image checksum and the source's review metadata.
Only image, target schema and declared task instructions are model inputs.

`{CONFIG}/manifest.json` records the source-to-test split mapping and scoring policy.
`{CONFIG}/source-schema.json` preserves the original schema. The target schema retains
its nullable fields and adds `x-match: exact` annotations to manuscript numbers and folios.
The matching rule normalizes text; it is not byte-for-byte equality. The `notes` field
is preserved in gold but excluded from the extraction score, matching the original board.

## Scope

One institution, 98 cards, model-drafted then human-reviewed gold. This is a small evaluation
set, not representative of every collection. Keep results for different configs separate.
Historical benchmark predictions used a simplified schema without nullable branches;
rescoring those predictions does not constitute rerunning models with this schema.
This dataset is not yet registered as a Hub benchmark. Future task IDs will be separate
from dataset configs; result records must identify config, data revision and scorer version.
"""


def export(source, output):
    from datasets import Dataset, Features, Image, Value

    if output.exists():
        raise ValueError("output must be a new directory; existing exports are never overwritten")
    raw_schema = json.loads((source / "schema.json").read_text())
    rows = make_rows(source, raw_schema)
    features = Features({"id": Value("string"), "image": Image(), "target_schema": Value("string"),
                         "expected_output": Value("string"), "provenance": Value("string")})
    config_dir = output / CONFIG
    config_dir.mkdir(parents=True)
    data_path = config_dir / "test.parquet"
    Dataset.from_list(rows, features=features).to_parquet(data_path)
    manifest = make_manifest(rows, data_path, source)
    (config_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (config_dir / "source-schema.json").write_bytes((source / "schema.json").read_bytes())
    (config_dir / "source-card.md").write_bytes((source / "README.md").read_bytes())
    (output / "README.md").write_text(dataset_card())
    report = validate_config(output, CONFIG)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(snapshot_download(SOURCE_ID, repo_type="dataset", revision=SOURCE_REVISION))
    report = export(source, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
