# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub"]
# ///
"""Rescore cached predictions in results/nls/*.json against the gold labels + current scorer.

Usage:
    uv run glam_bench/rescore.py            # rescore, print board + drift vs stored scores
    uv run glam_bench/rescore.py --write    # also write the new scores back into results/nls/
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import result_io
from nls import NLS_DATASET, load_gold
from scorer import SCORER_VERSION, score

RESULTS = Path(__file__).resolve().parent.parent / "results" / "nls"
KEYS = ["content_f1", "precision", "recall", "exact_fields_f1", "fuzzy_fields_f1",
        "false_populate_rate", "fields_left_blank_rate", "identifiers_wrong_rate",
        "identifier_clean", "schema_valid"]

# A row whose id is not in the gold set. `discover_result_files` refuses such files before they
# reach `main`; kept so an imported call gets a marked row rather than a KeyError.
NO_GOLD = "no gold for id"
UNSCORED = {key: 0.0 for key in KEYS}


def mean(items, key):
    return sum(i.get(key, 0.0) for i in items) / len(items) if items else 0.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rewritten(loaded: dict, rescored: list[dict], gold_revision: str) -> dict:
    """The loaded file with new item scores, a refreshed `scoring` block, and nothing else touched.
    `scorer_versions` — the list a mixed file carries beside `scorer_version: "mixed"` — is
    dropped rather than carried over.
    """
    envelope = {k: v for k, v in loaded.items() if k != "items"}
    scoring = {k: v for k, v in loaded.get("scoring", {}).items() if k != "scorer_versions"}
    envelope["scoring"] = {**scoring, "scorer_version": SCORER_VERSION,
                           "gold_revision": gold_revision, "rescored_at": utc_now()}
    envelope["items"] = rescored
    return envelope


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="persist rescored items back to results/nls/")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="rescore unfinished files and files with transport errors too")
    args = ap.parse_args(argv)

    schema, gold, gold_revision = load_gold()
    print(f"gold: {len(gold)} rows "
          f"({sum(1 for g in gold.values() if g['label_status'] == 'verified')} verified / "
          f"{sum(1 for g in gold.values() if g['label_status'] == 'corrected')} corrected)\n")

    rows = []
    for f in result_io.discover_result_files(RESULTS, args.allow_incomplete,
                                             gold_ids=set(gold),
                                             expected_dataset_id=NLS_DATASET):
        d = json.loads(f.read_text())
        rescored, scored = [], []
        for item in d["items"]:
            g = gold.get(item["id"])
            if g is None:
                rescored.append({**item, **UNSCORED, "scorer_error": NO_GOLD})
                continue
            sc = score(g["gold"], item["prediction"], schema) if item.get("prediction") else \
                {"content_f1": 0.0, "schema_valid": 0.0}
            row = {**item, **sc, "label_status": g["label_status"]}
            rescored.append(row)
            scored.append(row)
        ver = [i for i in scored if i["label_status"] == "verified"]
        cor = [i for i in scored if i["label_status"] == "corrected"]
        rows.append({
            "label": d["model"]["label"], "params": d["model"].get("params", "?"),
            "tier": "attested" if result_io.is_attested(d) else "self-reported",
            "n": len(scored), "missing": len(rescored) - len(scored),
            "errors": sum(1 for i in d["items"] if i.get("error")),
            "stored_f1": mean(d["items"], "content_f1"),
            "verified_f1": mean(ver, "content_f1"), "corrected_f1": mean(cor, "content_f1"),
            # the means are over the rows that could be scored; an unscored row is not a zero
            **{k: mean(scored, k) for k in KEYS},
        })
        if args.write:
            result_io.atomic_write_json(f, rewritten(d, rescored, gold_revision))

    rows.sort(key=lambda r: -r["content_f1"])
    cols = ["content_f1", "fields_left_blank_rate", "identifiers_wrong_rate",
            "false_populate_rate", "schema_valid", "verified_f1", "corrected_f1"]
    print("| # | model | params | tier | " + " | ".join(c.replace("_", " ") for c in cols)
          + " | errors | drift |")
    print("|---|---|---|---|" + "|".join(["---:"] * (len(cols) + 2)) + "|")
    for i, r in enumerate(rows, 1):
        cells = " | ".join(f"{r[c]*100:.0f}%" if c == "schema_valid" else f"{r[c]:.3f}" for c in cols)
        drift = r["content_f1"] - r["stored_f1"]
        print(f"| {i} | {r['label']} | {r['params']} | {r['tier']} | {cells} | {r['errors']} "
              f"| {drift:+.4f} |")
    print("\nRates above are the mean of the per-card rates, over every field. The published board "
          "\nmicro-averages them and drops `notes` — quote board.py, not this, for the page."
          "\n`tier` says how the file was made, not how good the model is: attested = produced by "
          "\nthis harness with per-row provenance, self-reported = anything else (docs/SUBMISSION.md).")
    unscored = sum(r["missing"] for r in rows)
    if unscored:
        print(f"\nWARNING: {unscored} prediction(s) with no matching gold row, kept in the file "
              f"unscored (scorer_error: {NO_GOLD!r}) and left out of the rates above: "
              + ", ".join(f"{r['label']}={r['missing']}" for r in rows if r["missing"]))
    if args.write:
        print(f"\nwrote {len(rows)} rescored files to {RESULTS}")


if __name__ == "__main__":
    main()
