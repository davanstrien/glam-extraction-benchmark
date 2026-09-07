"""Typed key-information-extraction (KIE) scorer for the GLAM extraction benchmark.

Model-agnostic: scores a predicted JSON object against a gold JSON object, using the canonical
**JSON Schema** to choose a per-field match rule (see schema.field_match_type):
  - enum / format:date / integer / number / boolean / x-match:exact  -> "exact" (normalised exact)
  - plain string                                                      -> "fuzzy" (SequenceMatcher ratio, partial)

Returns an overall content-F1 PLUS a breakdown (exact-type vs free-text fields, abstention
false-populate rate, schema validity) so the leaderboard shows *where* a model wins/loses, and
a split of present-field errors into blank (workload) vs wrong (risk) — see `score`.
"""
from __future__ import annotations
import json
import re
import unicodedata
from difflib import SequenceMatcher

from schema import field_kind, field_match_type

# Bump whenever a change here would move a number already scored from cached predictions, and say
# what moved. Every score() result carries the version, so a board can state which scorer made it.
#   2026-08-26a  explicit null takes the boolean default (was: scored blank); repeated entries
#                aligned best-pair-first (was: gold order, greedy); NFKC normalisation (was: NFKD,
#                which split accented letters).
#   2026-08-05   the two-axis split (blank / wrong / invented) and the `notes` exclusion.
SCORER_VERSION = "2026-08-26a"


def _norm(s) -> str:
    # NFKC, not NFKD: compatibility forms still fold (ligatures, width variants) but an accented
    # letter stays one codepoint. Under NFKD the combining mark is not `\w`, so the punctuation
    # strip turned "Müller" into "mu ller" and no accented string could ever match exactly.
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def _sim(a, b) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _credit(g, p, exact: bool) -> float:
    return 1.0 if _norm(g) == _norm(p) else (0.0 if exact else _sim(g, p))


def _obj_sim(a, b) -> float:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0
    keys = set(a) & set(b)
    return sum(_sim(a[k], b[k]) for k in keys if a.get(k) and b.get(k)) / (len(keys) or 1)


def _set_credit(gold, pred, exact: bool) -> float:
    """Set-F1 over the elements of a scalar array (e.g. subject_headings, notes)."""
    g = [x for x in (gold or []) if x not in (None, "", {})]
    p = [x for x in (pred or []) if x not in (None, "", {})]
    if not g or not p:
        return 0.0
    rec = sum(max((_credit(gi, pj, exact) for pj in p), default=0.0) for gi in g) / len(g)
    prec = sum(max((_credit(pj, gi, exact) for gi in g), default=0.0) for pj in p) / len(p)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _align(gold: list, pred: list) -> dict:
    """gold index -> predicted index for arrays of objects (e.g. `entries`).

    Best pair first, over every (gold, predicted) combination, so a gold entry the model skipped
    cannot claim the prediction that belongs to its neighbour. A pair with no similarity never
    matches: that gold entry scores blank, the stray predicted entry scores invented.
    """
    pairs = []
    for i, gi in enumerate(gold):
        for j, pj in enumerate(pred):
            pairs.append((_obj_sim(gi, pj), i, j))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))  # highest similarity first; ties keep order
    match = {}
    for sim, i, j in pairs:
        if sim > 0 and i not in match and j not in match.values():
            match[i] = j
    return match


MISSING = object()  # key absent from the object, as distinct from an explicit JSON null


def _walk(gold, pred, node, rows: list, pred_defaults: bool = True, path: str = "") -> None:
    """Traverse JSON Schema; rows = (gold_present, pred_present, credit, ftype, kind, path, g, p).

    Everything after `ftype` is a reporting-only annotation — nothing in the matching path reads
    it. `path` is dotted with array position (`entries[1].ms_no`), so a caller can regroup rows
    back into the record they came from; the values are carried so a display can show exactly
    what each row scored, rather than re-deriving the entry alignment and drifting from it.

    One function per schema shape below; this one only dispatches.
    """
    node_type = node.get("type") if isinstance(node, dict) else None
    if node_type == "object":
        _walk_object(gold, pred, node, rows, pred_defaults, path)
    elif node_type == "array":
        items = node.get("items", {})
        if isinstance(items, dict) and items.get("type") == "object":
            _walk_array_of_objects(gold, pred, items, rows, pred_defaults, path)
        else:
            _walk_scalar_array(gold, pred, items, rows, path)
    else:
        _walk_leaf(gold, pred, node, rows, pred_defaults, path)


def _walk_object(gold, pred, node, rows, pred_defaults, path):
    for key, sub in node.get("properties", {}).items():
        g = gold.get(key, MISSING) if isinstance(gold, dict) else MISSING
        p = pred.get(key, MISSING) if isinstance(pred, dict) else MISSING
        _walk(g, p, sub, rows, pred_defaults, f"{path}.{key}" if path else key)


def _walk_array_of_objects(gold, pred, items, rows, pred_defaults, path):
    """Repeated records (e.g. `entries`): pair gold with predicted objects, then walk each pair."""
    g = gold if isinstance(gold, list) else []
    p = pred if isinstance(pred, list) else []
    match = _align(g, p)
    for i, gi in enumerate(g):
        _walk(gi, p[match[i]] if i in match else {}, items, rows, pred_defaults, f"{path}[{i}]")
    extra = len(g)  # unmatched predicted objects = hallucinated; numbered after the gold ones
    for j, pj in enumerate(p):
        if j in match.values():
            continue
        _walk({}, pj, items, rows, pred_defaults, f"{path}[{extra}]")
        extra += 1


def _walk_scalar_array(gold, pred, items, rows, path):
    """An array of scalars is one field, scored as a set."""
    ft, kind = field_match_type(items), field_kind(items)
    gold = None if gold is MISSING else gold
    pred = None if pred is MISSING else pred
    gp = isinstance(gold, list) and len(gold) > 0
    pp = isinstance(pred, list) and len(pred) > 0
    credit = _set_credit(gold, pred, ft == "exact") if gp and pp else 0.0
    rows.append((gp, pp, credit, ft, kind, path, gold, pred))


def _walk_leaf(gold, pred, node, rows, pred_defaults, path):
    ft, kind = field_match_type(node), field_kind(node)
    # JSON Schema: an ABSENT key means the declared default. The prompt tells a model to "omit
    # or null" a field not on the document, so an explicit null is the same statement as an
    # absent key and takes the default too — otherwise a model that writes `null` is scored
    # blank where a model that skips the key is scored right, for the same reading of the card.
    if isinstance(node, dict) and node.get("type") == "boolean" and isinstance(node.get("default"), bool):
        if gold is MISSING:
            gold = node["default"]
        if pred in (MISSING, None) and pred_defaults:
            pred = node["default"]
    gold = None if gold is MISSING else gold
    pred = None if pred is MISSING else pred
    g = None if gold in (None, "", {}, []) else gold
    p = None if pred in (None, "", {}, []) else pred
    credit = _credit(g, p, ft == "exact") if (g is not None and p is not None) else 0.0
    rows.append((g is not None, p is not None, credit, ft, kind, path, g, p))


def _prf(rows):
    gp = [r for r in rows if r[0]]
    pp = [r for r in rows if r[1]]
    prec = sum(r[2] for r in pp) / len(pp) if pp else 0.0
    rec = sum(r[2] for r in gp) / len(gp) if gp else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def score_rows(gold, pred, schema, exclude: tuple[str, ...] = ()) -> tuple[list, float]:
    """The scored rows behind `score`, plus schema validity — for callers that want to show their
    working (which field scored what) rather than only the aggregate."""
    schema = json.loads(schema) if isinstance(schema, str) else schema
    gold = json.loads(gold) if isinstance(gold, str) else gold
    if isinstance(pred, str):
        try:
            pred = json.loads(re.search(r"\{.*\}", pred, re.S).group()); valid = 1.0
        except Exception:
            pred, valid = {}, 0.0
    else:
        valid = 1.0 if isinstance(pred, dict) else 0.0
        pred = pred if isinstance(pred, dict) else {}

    rows: list = []
    _walk(gold or {}, pred or {}, schema or {}, rows, pred_defaults=bool(valid))
    return [r for r in rows if r[5] not in exclude], valid


def score(gold, pred, schema, exclude: tuple[str, ...] = ()) -> dict:
    """gold/pred/schema may be dicts or JSON strings. `schema` is canonical JSON Schema.

    `exclude` drops field paths (e.g. `("notes",)`) from *reporting* — matching is untouched.

    On top of content-F1 this splits present-field errors into the two things a cataloguer cares
    about separately. F1 blends them: a field left blank and a field filled with the wrong value
    cost the same, but one is work you can see outstanding and the other is a silent error that
    looks like data. So:

      blank    gold has a value, the model wrote nothing  -> WORKLOAD (visible gap)
      wrong    the model wrote something different        -> RISK (silent error)
      invented the card is blank, the model wrote a value -> RISK (silent error)

    Counts (`n_*`) are returned alongside the per-card rates so a board can micro-average
    (sum numerators / sum denominators) rather than average per-card rates over wildly different
    denominators — a card carrying one manuscript number would otherwise weigh as much as one
    carrying six.
    """
    rows, valid = score_rows(gold, pred, schema, exclude)
    prec, rec, f1 = _prf(rows)

    gold_rows = [r for r in rows if r[0]]
    filled = [r for r in gold_rows if r[1]]
    gold_absent = [r for r in rows if not r[0]]
    ident_gold = [r for r in gold_rows if r[4] == "identifier"]
    ident_filled = [r for r in ident_gold if r[1]]
    n_ident_wrong = sum(1 for r in ident_filled if r[2] < 1.0)
    return {
        "content_f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "exact_fields_f1": round(_prf([r for r in rows if r[3] == "exact"])[2], 4),
        "fuzzy_fields_f1": round(_prf([r for r in rows if r[3] == "fuzzy"])[2], 4),
        "false_populate_rate": round(sum(1 for r in gold_absent if r[1]) / len(gold_absent), 4) if gold_absent else 0.0,
        "fields_left_blank_rate": round(1 - len(filled) / len(gold_rows), 4) if gold_rows else 0.0,
        "identifiers_wrong_rate": round(n_ident_wrong / len(ident_filled), 4) if ident_filled else 0.0,
        # vacuously clean on the cards that carry no identifier at all — same for every model
        "identifier_clean": 1.0 if all(r[2] == 1.0 for r in ident_gold) else 0.0,
        "schema_valid": valid,
        "scorer_version": SCORER_VERSION,
        "n_gold_fields": len(gold_rows),
        "n_filled": len(filled),
        "n_blank": len(gold_rows) - len(filled),
        "n_gold_absent": len(gold_absent),
        "n_invented": sum(1 for r in gold_absent if r[1]),
        "n_ident_gold": len(ident_gold),
        "n_ident_filled": len(ident_filled),
        "n_ident_wrong": n_ident_wrong,
    }

