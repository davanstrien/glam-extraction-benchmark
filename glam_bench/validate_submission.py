# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub", "datasets>=5,<6", "jsonschema>=4,<5", "Pillow>=12,<13"]
# ///
"""Check one submission against the contract before it is submitted or boarded.

Usage:
    uv run glam_bench/validate_submission.py results/nls/my-extractor.json
    uv run glam_bench/validate_submission.py results/nls/my-extractor.json --dataset-revision main
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import result_io
from nls import NLS_DATASET, load_gold
from dataset_contract import BENCHMARK_ID, evaluation_identity, load_hub_config


def read_submission(path: Path) -> dict:
    """The parsed file, or SystemExit with a readable message."""
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(document, dict):
        raise SystemExit(f"cannot read {path}: the top level is a "
                         f"{type(document).__name__}, not a JSON object")
    return document


def snapshot_to_check(document: dict, requested: str | None) -> str:
    """Which gold snapshot the ids are checked against."""
    if requested:
        return requested
    revision = result_io.field_at(document, "dataset", "inference_revision")
    if isinstance(revision, str) and revision.strip():
        return revision
    return "main"


def gold_for(document: dict, requested: str | None) -> tuple[set | None, str, str | None]:
    """-> (gold ids or None, the snapshot the check ran against, a note to print or None)."""
    dataset = result_io.field_at(document, "dataset", "id") or NLS_DATASET
    config = result_io.field_at(document, "dataset", "config")
    if config or dataset == BENCHMARK_ID:
        if not config:
            raise SystemExit("benchmark submissions must name dataset.config")
        rows, manifest, resolved = load_hub_config(dataset, config, snapshot_to_check(document, requested))
        for key, expected in evaluation_identity(manifest).items():
            if result_io.field_at(document, "dataset", key) != expected:
                raise SystemExit(f"dataset.{key} differs from the selected benchmark config")
        return {row["id"] for row in rows}, resolved, None
    if dataset == NLS_DATASET:
        _schema, gold, resolved = load_gold(snapshot_to_check(document, requested))
        return set(gold), resolved, None
    return (None, snapshot_to_check(document, requested),
            f"ids not checked: no gold loader for {dataset}")


def summarise(document: dict, revision: str) -> str:
    """The one line an accepted submission prints."""
    label = result_io.field_at(document, "model", "label")
    dataset = result_io.field_at(document, "dataset", "id") or NLS_DATASET
    rows = document.get("items") or []
    tier = "attested" if result_io.is_attested(document) else "self-reported"
    producer = result_io.produced_by(document).get("producer")
    return (f"OK: {len(rows)} items · model {label!r} · {dataset} @ {revision[:12]} · "
            f"produced by {producer} ({tier})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="check one result file against the submission contract (docs/SUBMISSION.md)")
    ap.add_argument("file", help="the submission to check, e.g. results/nls/my-extractor.json")
    ap.add_argument("--dataset-revision", default=None,
                    help="check the item ids against this gold snapshot instead of the one the "
                         "file names in dataset.inference_revision")
    args = ap.parse_args(argv)

    path = Path(args.file)
    document = read_submission(path)

    gold_ids, resolved, note = gold_for(document, args.dataset_revision)
    if note:
        print(note)

    problems = result_io.validate_submission(document, gold_ids)
    if problems:
        print(f"{path}: {len(problems)} problem(s) against {resolved[:12]}:")
        for problem in problems:
            print(f"  - {problem}")
        if any(problem.startswith("dataset.inference_revision") for problem in problems):
            print(f"hint: the snapshot checked against resolves to {resolved}")
        print("Contract: docs/SUBMISSION.md")
        return 1

    print(summarise(document, resolved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
