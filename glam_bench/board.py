# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub"]
# ///
"""Two-axis board aggregation: WORKLOAD (what's left to fix) vs RISK (what it would introduce).

Usage: uv run glam_bench/board.py     # print the two-axis board
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import result_io
from nls import NLS_DATASET, load_gold
from scorer import score

RESULTS = Path(__file__).resolve().parent.parent / "results" / "nls"

# Reviewer commentary about the labelling, not content of the card. See module docstring.
EXCLUDE = ("notes",)

BASELINE_LABEL = "constants baseline"


def _mode(values):
    """Most common value, counting absence as a value; ties broken by first occurrence."""
    keyed = [json.dumps(v, sort_keys=True, ensure_ascii=False) for v in values]
    counts = Counter(keyed)
    best = max(counts, key=lambda k: (counts[k], -keyed.index(k)))
    return json.loads(best)


def constants_baseline(gold: dict) -> dict:
    """A "reads nothing" model: one fixed record, emitted for all 98 cards."""
    records = [g["gold"] for g in gold.values()]
    fields = [f for f in records[0] if f != "entries"]
    rec = {f: _mode([r.get(f) for r in records]) for f in fields}

    entries = [e for r in records for e in (r.get("entries") or [])]
    n = _mode([len(r.get("entries") or []) for r in records])
    if entries and n:
        modal = {f: _mode([e.get(f) for e in entries]) for f in entries[0]}
        rec["entries"] = [modal] * n
    else:
        rec["entries"] = []
    return {k: v for k, v in rec.items() if v not in (None, "", [], {})}


def _micro(items: list[dict], num: str, den: str) -> float:
    d = sum(i.get(den, 0) for i in items)
    return sum(i.get(num, 0) for i in items) / d if d else 0.0


def _mean(items: list[dict], key: str) -> float:
    return sum(i.get(key, 0.0) for i in items) / len(items) if items else 0.0


def summarise(label: str, params: str, items: list[dict], attested: bool = False) -> dict:
    """items = per-card score() dicts (each carrying `label_status`) -> one board row."""
    ver = [i for i in items if i.get("label_status") == "verified"]
    cor = [i for i in items if i.get("label_status") == "corrected"]
    return {
        "label": label, "params": params, "n": len(items), "attested": attested,
        # the two axes
        "fields_left_blank": _micro(items, "n_blank", "n_gold_fields"),
        "identifiers_wrong": _micro(items, "n_ident_wrong", "n_ident_filled"),
        "invented_fields": _micro(items, "n_invented", "n_gold_absent"),
        # headline + the detail behind it
        "content_f1": _mean(items, "content_f1"),
        "precision": _mean(items, "precision"), "recall": _mean(items, "recall"),
        "exact_fields_f1": _mean(items, "exact_fields_f1"),
        "fuzzy_fields_f1": _mean(items, "fuzzy_fields_f1"),
        "schema_valid": _mean(items, "schema_valid"),
        "identifier_clean_cards": sum(int(i.get("identifier_clean", 0.0)) for i in items),
        "verified_f1": _mean(ver, "content_f1"), "corrected_f1": _mean(cor, "content_f1"),
        "n_verified": len(ver), "n_corrected": len(cor),
        "n_ident_filled": sum(i.get("n_ident_filled", 0) for i in items),
    }


def board_rows(include_baseline: bool = True, allow_incomplete: bool = False) -> list[dict]:
    """Rescore every publishable prediction set from disk against the current gold + scorer."""
    schema, gold, _revision = load_gold()
    rows = []
    # `results/nls/` holds submissions over the NLS gold and nothing else. Named rather than
    # assumed: id sets alone cannot tell this collection's `card-1` from another's.
    for f in result_io.discover_result_files(RESULTS, allow_incomplete, gold_ids=set(gold),
                                             expected_dataset_id=NLS_DATASET):
        d = json.loads(f.read_text())
        items = []
        for item in d["items"]:
            g = gold.get(item["id"])
            if g is None:
                continue
            sc = score(g["gold"], item["prediction"], schema, exclude=EXCLUDE) if item.get("prediction") \
                else score(g["gold"], "", schema, exclude=EXCLUDE)
            items.append({**sc, "label_status": g["label_status"]})
        # The file the row came from, for anything that needs a per-row fact the summary does
        # not carry (the page reads finish reasons from it). Never rebuilt from the label: the
        # gate accepts a file whose name differs from `model.label`.
        rows.append({**summarise(d["model"]["label"], d["model"].get("params", "?"), items,
                                 attested=result_io.is_attested(d)), "result_path": f})

    if include_baseline:
        const = json.dumps(constants_baseline(gold), ensure_ascii=False)
        items = [{**score(g["gold"], const, schema, exclude=EXCLUDE), "label_status": g["label_status"]}
                 for g in gold.values()]
        # Not a submission: this row is computed here, from the gold, every time the board builds.
        rows.append({**summarise(BASELINE_LABEL, "—", items, attested=True), "baseline": True})
    return rows


def risk_sort_key(r: dict) -> tuple:
    """Ascending identifiers-wrong — but a model that never wrote an identifier has a 0% that means
    "nothing to be wrong", not "nothing wrong".
    """
    return (0 if r["n_ident_filled"] else 1, r["identifiers_wrong"], r["fields_left_blank"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="print the two-axis board")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="board unfinished files and files with transport errors too")
    args = ap.parse_args(argv)

    rows = board_rows(allow_incomplete=args.allow_incomplete)
    real = [r for r in rows if not r.get("baseline")]
    base = [r for r in rows if r.get("baseline")]
    real.sort(key=risk_sort_key)

    cols = [("identifiers wrong", "identifiers_wrong"), ("invented fields", "invented_fields"),
            ("fields left blank", "fields_left_blank"), ("fields right (F1)", "content_f1")]
    # How many cards there are is a fact about the gold that was loaded, not a number to keep in
    # step by hand. `n` is the count each row was actually scored over.
    cards = max((r["n"] for r in rows), default=0)
    print("| model | params | " + " | ".join(c for c, _ in cols)
          + f" | ident-clean /{cards} | valid |")
    print("|---|---|" + "|".join(["---:"] * (len(cols) + 2)) + "|")
    for r in real + base:
        cells = " | ".join(f"{r[k]*100:.1f}%" if k != "content_f1" else f"{r[k]*100:.1f}"
                           for _, k in cols)
        mark = "" if r["attested"] else " †"
        print(f"| {r['label']}{mark} | {r['params']} | {cells} | {r['identifier_clean_cards']} "
              f"| {r['schema_valid']*100:.0f}% |")
    if any(not r["attested"] for r in real):
        print("\n† self-reported: the file was not produced by this harness and carries no per-row "
              "provenance. The predictions in it were rescored here like every other row.")


if __name__ == "__main__":
    main()


def f1_halfwidth_points(results_dir, n_boot: int = 2000, seed: int = 0) -> float:
    """Largest 95% bootstrap half-width, in points, of any model's mean per-card content F1.

    Resamples cards with replacement; deterministic for a given results directory. This is the
    per-card mean, not the micro-average the board quotes, so it is a guide to how far a score can
    move with a different draw of cards, not an interval on the printed number. Zero when no file
    has two or more scored cards.
    """
    import random

    rng = random.Random(seed)
    widest = 0.0
    # Plain glob, not discover_result_files: this is a guide figure over whatever is on the board's
    # disk, and the contract check has already run (or refused the file) by the time the page builds.
    files = [p for p in sorted(Path(results_dir).glob("*.json")) if ".limit" not in p.name]
    for path in files:
        items = json.loads(path.read_text())["items"]
        vals = [float(i.get("content_f1") or 0.0) for i in items]
        if len(vals) < 2:
            continue
        means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n_boot))
        lo, hi = means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
        widest = max(widest, (hi - lo) * 50)
    return widest
