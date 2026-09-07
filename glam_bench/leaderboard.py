# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Aggregate results/*.json into a leaderboard table (content-F1 + per-field-type breakdown).

Usage: uv run glam_bench/leaderboard.py [results_subdir] [--allow-incomplete]   # e.g. `nls`
"""
import argparse
import json
from pathlib import Path

import result_io

RESULTS = Path(__file__).resolve().parent.parent / "results"

KEYS = ["content_f1", "precision", "recall", "exact_fields_f1", "fuzzy_fields_f1",
        "false_populate_rate", "schema_valid"]


def aggregate(per_item):
    """Mean over items; error rows may lack some keys — treat missing as 0."""
    return {k: round(sum(d.get(k, 0.0) for d in per_item) / len(per_item), 4) for k in KEYS}


def main(argv=None):
    ap = argparse.ArgumentParser(description="aggregate a results dir into a leaderboard table")
    ap.add_argument("results_subdir", nargs="?", help="subdirectory of results/, e.g. `nls`")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="aggregate unfinished files and files with transport errors too")
    args = ap.parse_args(argv)

    outdir = RESULTS / args.results_subdir if args.results_subdir else RESULTS
    rows = []
    for f in result_io.discover_result_files(outdir, args.allow_incomplete):
        d = json.loads(f.read_text())
        agg = aggregate(d["items"])
        agg["errors"] = sum(1 for i in d["items"] if i.get("error"))
        rows.append({"label": d["model"]["label"], "params": d["model"].get("params", "?"),
                     "cost": d["model"].get("cost", "?"), **agg})
    rows.sort(key=lambda r: -r.get("content_f1", 0))

    cols = ["content_f1", "exact_fields_f1", "fuzzy_fields_f1", "false_populate_rate", "schema_valid"]
    head = "| # | model | params | " + " | ".join(c.replace("_", " ") for c in cols) + " | errors | cost |"
    sep = "|---|---|---|" + "|".join(["---:"] * (len(cols) + 1)) + "|---|"
    print(head); print(sep)
    for i, r in enumerate(rows, 1):
        cells = " | ".join(f"{r.get(c, 0):.3f}" if c != "schema_valid" else f"{r.get(c,0)*100:.0f}%" for c in cols)
        print(f"| {i} | {r['label']} | {r['params']} | {cells} | {r.get('errors', 0)} | {r['cost']} |")


if __name__ == "__main__":
    main()
