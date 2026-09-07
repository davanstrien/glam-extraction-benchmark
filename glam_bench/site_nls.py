# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub"]
# ///
"""Build site/index.html — the two-axis board on the NLS institution-verified gold set.

    WORKLOAD  what is left to fix   — fields the model left blank. Visible.
    RISK      what it would introduce — identifiers it got wrong, fields it invented. Silent.

Usage: uv run glam_bench/site_nls.py [--allow-incomplete]
"""
import argparse
import html
import json
import statistics
from pathlib import Path

from board import EXCLUDE, board_rows, f1_halfwidth_points, risk_sort_key
from nls import NLS_DATASET, load_gold
from scorer import score_rows

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "nls"
OUT = ROOT / "site" / "index.html"

HF_IDS = {
    "lift-9B": "datalab-to/lift", "NuExtract-3": "numind/NuExtract3",
    "LFM2.5-VL-1.6B-Extract": "LiquidAI/LFM2.5-VL-1.6B-Extract",
    "Qwen3-VL-235B": "Qwen/Qwen3-VL-235B-A22B-Instruct", "gemma-4-31B": "google/gemma-4-31B-it",
    "Llama-4-Scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "gemma-3-4b": "google/gemma-3-4b-it",
    "Qwen3.6-27B": "Qwen/Qwen3.6-27B", "gemma-4-26B-A4B": "google/gemma-4-26B-A4B-it",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B", "Qwen3.8-27B": "Qwen/Qwen3.8-27B",
    "Apertus-v1.5-8B": "swiss-ai/Apertus-v1.5-8B",
}

# The one card shown in full below the board. Chosen because it demonstrates both axes at once:
# the model leaves a field blank (workload) AND copies a whole entry off the card *behind* this
# one, visible as show-through in the image (risk). build() re-checks that at build time.
WORKED = ("gemma-4-31B", "Allan-W.-Anderson-D.__0221", "worked-card.jpg")

# Row-level footnotes, appended to that row's schema-validity flag. For a model whose outputs
# fail to parse, "how often" is on the board but "why" is not, and the two have very different
# consequences for someone deciding whether to try it.
ROW_NOTES = {
    "LFM2.5-VL-1.6B-Extract":
        "On this collection the model tends to return the schema: "
        "field names and type keywords such as \"exact\" instead of values. "
        "Even parsed rows contain no manuscript numbers.",
}

COLS = [  # (header, key, css class, hover)
    ("Identifiers wrong", "identifiers_wrong", "risk",
     ("Manuscript numbers and folio references, scored with no half marks. "
      "Wrong identifiers look valid in the catalogue. "
      "Out of identifiers present on the card that the model filled in.")),
    ("Invented fields", "invented_fields", "risk",
     ("Fields filled in where the card is blank: the hardest errors to catch on review, "
      "with nothing on the card to check against.")),
    ("Fields left blank", "fields_left_blank", "work",
     ("Fields with values on the card but returned empty: visible work for a reviewer.")),
    ("Fields right", "content_f1", "score",
     ("Overall content-F1 across every field, including part marks. "
      "It combines risk and workload.")),
]


def truncated(path) -> tuple[int, int]:
    """(rows whose generation stopped at the token limit, rows in the file) for one result file.

    Read from the file so the page never types a count: a model's truncation rate is part of why
    its outputs fail to parse, and it changes with every rerun. Takes the path the board discovered
    rather than rebuilding it from the label, because the gate accepts files named otherwise.
    """
    if not path or not Path(path).exists():
        return 0, 0
    items = json.loads(Path(path).read_text())["items"]
    cut = sum(1 for i in items if (i.get("provenance") or {}).get("finish_reason") == "length")
    return cut, len(items)


def fmt(v, key):
    return f"{v * 100:.1f}" if key == "content_f1" else f"{v * 100:.1f}%"


def subtitle(n_cards: int, n_models: int) -> str:
    """The header line under the title."""
    cards = "1 index card" if n_cards == 1 else f"{n_cards} index cards"
    models = "1 model" if n_models == 1 else f"{n_models} models"
    return f"{cards} · {models}"


def count_label_status(gold: dict, status: str) -> int:
    """How many gold records a reviewer left at `status` — "verified" or "corrected"."""
    return sum(1 for record in gold.values() if record.get("label_status") == status)


def _spread(pts, key, lo, hi, gap):
    """Push overlapping labels apart along y, staying inside [lo, hi]."""
    col = sorted(pts, key=lambda p: p[key])
    for p in col:                      # clamp first: clamping after the push re-opens the gap
        p[key] = max(p[key], lo)
    for i in range(1, len(col)):
        col[i][key] = max(col[i][key], col[i - 1][key] + gap)
    if col and col[-1][key] > hi:
        col[-1][key] = hi
        for i in range(len(col) - 2, -1, -1):
            col[i][key] = min(col[i][key], col[i + 1][key] - gap)


def scatter(rows, base) -> str:
    """workload (x) vs risk (y)."""
    W, H, L, R, T, B = 900, 520, 62, 26, 26, 48
    pw, ph = W - L - R, H - T - B

    def up(v):  # round up to the next 10%
        return max(10, -(-v // 10) * 10)

    # y is meaningless for a model that never wrote an identifier: its 0% says "never attempted",
    # not "never wrong". Such a row must not set the scale, the medians, or the "better" corner.
    real = [r for r in rows if r["n_ident_filled"]] or rows
    ys = sorted(r["identifiers_wrong"] * 100 for r in real)
    inlier = 3 * statistics.median(ys)
    xmax = up(max(r["fields_left_blank"] * 100 for r in rows))
    ymax = up(max([y for y in ys if y <= inlier], default=ys[-1]))

    def X(v): return L + min(v, xmax) / xmax * pw
    def Y(v): return T + (1 - min(v, ymax) / ymax) * ph

    xmed = statistics.median(r["fields_left_blank"] * 100 for r in rows)
    ymed = statistics.median(r["identifiers_wrong"] * 100 for r in real)

    pts = [{"x": r["fields_left_blank"] * 100, "y": r["identifiers_wrong"] * 100,
            "label": r["label"], "hollow": not r["n_ident_filled"]} for r in rows]
    pts.append({"x": base["fields_left_blank"] * 100, "y": base["identifiers_wrong"] * 100,
                "label": "constants baseline", "hollow": False, "base": True})
    for p in pts:
        if p["y"] > ymax:
            p["off"] = True
            p["label"] += f" · {p['y']:.0f}% ↑"

    for p in pts:
        p["px"], p["py"] = X(p["x"]), Y(p["y"])
        p["left"] = p["px"] > L + pw * 0.7
        p["ly"] = p["py"]
    # off-scale points all land on the top edge; stack them so neither marker nor label collides
    for i, p in enumerate(sorted([p for p in pts if p.get("off")], key=lambda p: p["px"])):
        p["py"] = p["ly"] = T + i * 20
    for side in (False, True):
        _spread([p for p in pts if p["left"] is side], "ly", T + 6, T + ph - 4, 15)

    svg = [(f'<svg class="scat" viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="Fields left blank against identifiers wrong, {len(pts) - 1} models and the '
            f'constants baseline">')]
    # Both axes are "lower is better", so the good corner is the origin. Shade the quadrant below
    # both medians and name it, rather than making the reader infer direction from the axis labels.
    svg.append(f'<rect class="best" x="{L}" y="{Y(ymed):.1f}" width="{X(xmed) - L:.1f}" '
               f'height="{T + ph - Y(ymed):.1f}"/>')
    svg.append(f'<text class="bestl" x="{L + 7}" y="{T + ph - 8}">better</text>')
    svg.append(f'<line class="ax" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>'
               f'<line class="ax" x1="{L}" y1="{T}" x2="{L}" y2="{T + ph}"/>')
    svg.append(f'<line class="med" x1="{X(xmed):.1f}" y1="{T}" x2="{X(xmed):.1f}" y2="{T + ph}"/>'
               f'<line class="med" x1="{L}" y1="{Y(ymed):.1f}" x2="{L + pw}" y2="{Y(ymed):.1f}"/>'
               f'<text class="quad" x="{X(xmed) + 5:.1f}" y="{T + ph - 6:.1f}">median</text>'
               f'<text class="quad" x="{L + pw:.0f}" y="{Y(ymed) - 5:.1f}" text-anchor="end">median</text>')
    for v in (0, xmax / 2, xmax):
        svg.append(f'<text class="tick" x="{X(v):.1f}" y="{T + ph + 17}" text-anchor="middle">{v:.0f}%</text>')
    for v in (0, ymax / 2, ymax):
        svg.append(f'<text class="tick" x="{L - 9}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.0f}%</text>')
    svg.append(f'<text class="axl" x="{L + pw / 2:.0f}" y="{H - 9}" text-anchor="middle">'
               f'fields left blank &nbsp;→&nbsp; more for the reviewer to finish</text>')
    svg.append(f'<text class="axl" transform="translate(17,{T + ph / 2:.0f}) rotate(-90)" '
               f'text-anchor="middle">identifiers wrong &nbsp;→&nbsp; more silent errors</text>')
    # Markers first, labels after, so a label is never buried under someone else's dot. Where two
    # models still land on top of each other the hover carries both, and the label's white halo
    # keeps it readable over whatever it crosses.
    for p in pts:
        cls = "dot base" if p.get("base") else ("dot hollow" if p["hollow"] else "dot")
        ident = "no identifiers written" if p["hollow"] else f"{p['y']:.1f}% identifiers wrong"
        tip = html.escape(f"{p['label']} — {p['x']:.1f}% fields left blank, {ident}")
        if p.get("off"):  # above the top of the scale — triangle, true value in the label
            x, y = p["px"], p["py"]
            svg.append(f'<path class="{cls}" d="M{x - 4.6:.1f} {y + 4} L{x:.1f} {y - 4.4} '
                       f'L{x + 4.6:.1f} {y + 4} Z"><title>{tip}</title></path>')
        else:
            svg.append(f'<circle class="{cls}" cx="{p["px"]:.1f}" cy="{p["py"]:.1f}" r="4">'
                       f'<title>{tip}</title></circle>')
    for p in pts:
        anc, dx = ("end", -10) if p["left"] else ("start", 10)
        if abs(p["ly"] - p["py"]) > 2:
            svg.append(f'<line class="lead" x1="{p["px"] + dx * 0.35:.1f}" y1="{p["py"]:.1f}" '
                       f'x2="{p["px"] + dx * 0.85:.1f}" y2="{p["ly"]:.1f}"/>')
        svg.append(f'<text class="pl{" base" if p.get("base") else ""}" x="{p["px"] + dx:.1f}" '
                   f'y="{p["ly"] + 4:.1f}" text-anchor="{anc}">{html.escape(p["label"])}</text>')
    svg.append("</svg>")

    # A key, written from what is actually on the plot: a shape nobody had to draw is a shape
    # nobody has to have explained.
    key = []
    if any(p["hollow"] for p in pts):
        key.append("A hollow point wrote no identifiers to get wrong. Read its workload figure instead.")
    if any(p.get("off") for p in pts):
        key.append("A triangle exceeds the scale; its label gives the value.")
    key.append("The grey point is the constants baseline: it reads nothing and gives every card the most "
               "common value for each field. Its score is the floor. Hover for exact figures.")
    return "\n".join(svg) + f'\n<p class="cap">{" ".join(key)}</p>'


def value(v) -> str:
    if v is None or v == [] or v == "":
        return '<span class="nil">—</span>'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return html.escape(", ".join(str(x) for x in v))
    return html.escape(str(v))


def worked_panel(schema, gold) -> tuple[str, dict]:
    """The board's numbers, on one card, field by field."""
    label, card_id, img = WORKED
    d = json.loads((RESULTS / f"{label}.json").read_text())
    item = next(i for i in d["items"] if i["id"] == card_id)
    g = gold[card_id]
    rows, _ = score_rows(g["gold"], item["prediction"], schema, exclude=EXCLUDE)

    seen = {"blank": 0, "invented": 0, "wrong": 0, "right": 0}
    out = []
    for gp, pp, credit, _ft, kind, path, gv, pv in rows:
        if gp and not pp:
            tag, cls = "left blank", "t-work"; seen["blank"] += 1
        elif pp and not gp:
            tag, cls = "invented", "t-risk"; seen["invented"] += 1
        elif not gp and not pp:
            tag, cls = "", "t-none"
        elif credit == 1.0:
            tag, cls = "right", "t-ok"; seen["right"] += 1
        elif kind == "text":
            tag, cls = f"part marks · {credit:.2f}", "t-part"
        else:
            tag, cls = "wrong", "t-risk"; seen["wrong"] += 1
        out.append(f'<tr><td class="fp">{html.escape(path)}</td><td>{value(gv)}</td>'
                   f'<td>{value(pv)}</td><td class="{cls}">{tag}</td></tr>')
    out.append(f'<tr class="exc"><td class="fp">notes</td><td>{value(g["gold"].get("notes"))}</td>'
               f'<td><span class="nil">—</span></td><td class="t-none">not scored</td></tr>')

    panel = f"""<details class="worked"><summary>One card, field by field</summary>
  <p class="wlead">{html.escape(HF_IDS.get(label, label))} on card
  <span class="mono">{html.escape(card_id)}</span> — a label National Library of Scotland
  cataloguers accepted unchanged. The model catalogued a second, fainter line below the main
  entry: show-through from the card behind.</p>
  <div class="wgrid">
    <figure><img src="{img}" alt="Manuscript index card reading ALLEN (George), Publisher, and UNWIN, letter of (1930), 9221, f. 102 — with fainter show-through text from the card behind visible below it">
      <figcaption>Card detail; models received the full image.
      NationalLibraryOfScotland/index-cards-eval · CC0</figcaption></figure>
    <table class="wt"><thead><tr>
      <th>field</th><th>human-checked record</th><th>model output</th><th>scored as</th>
    </tr></thead><tbody>
{chr(10).join(out)}
    </tbody></table>
  </div>
  <p class="wnote"><b class="w">{seen["blank"]} left blank</b> is workload: the model omitted the card's epithet,
  leaving a visibly empty field. <b class="r">{seen["invented"]} invented</b> and <b class="r">{seen["wrong"]} wrong</b>
  are risk: entry 1 copies a whole manuscript entry — number, folios, description — from the card
  behind, without marking it as doubtful. <span class="mono">102</span> for <span class="mono">f. 102</span>
  has the right number but the wrong form. Identifiers get no half marks, so it scores wrong.
  <span class="mono">notes</span> is the reviewer's labelling commentary. No model can produce it,
  so it is excluded from every number on this page.</p>
</details>"""
    return panel, seen


def build(allow_incomplete: bool = False):
    schema, gold, _revision = load_gold()
    # Every sentence on the page that states how many cards there are takes these numbers, for the
    # same reason `subtitle` counts rather than states one: the gold set is still accepting rows,
    # so a typed count is true for one build and silently wrong after the next.
    n_cards = len(gold)
    n_verified = count_label_status(gold, "verified")
    n_corrected = count_label_status(gold, "corrected")
    rows = board_rows(allow_incomplete=allow_incomplete)
    base = next(r for r in rows if r.get("baseline"))
    rows = [r for r in rows if not r.get("baseline")]
    rows.sort(key=risk_sort_key)
    halfwidth = f1_halfwidth_points(RESULTS)

    body = []
    for r in rows:
        # How the file was made, not how good the model is. The predictions were rescored here
        # either way; the dagger says nobody but the submitter can vouch for how they were obtained.
        tier = "" if r["attested"] else (
            ' <span class="tier" title="Self-reported: this result file was submitted rather than '
            'produced by the benchmark harness, and carries no per-row record of which endpoint '
            'answered. Its predictions were rescored here like every other row.">†</span>')
        cut, n_out = truncated(r.get("result_path"))
        note = f" {cut} of the {n_out} outputs hit the token limit." if cut else ""
        note += (" " + ROW_NOTES[r["label"]]) if r["label"] in ROW_NOTES else ""
        flag = "" if r["schema_valid"] >= 1.0 else (
            f'<span class="flag" title="On {round((1 - r["schema_valid"]) * r["n"])} of the '
            f'{r["n"]} cards the output was not usable JSON and scored zero.'
            f'{html.escape(note)}">{r["schema_valid"] * 100:.0f}% parsed</span>')
        cells = []
        for _h, k, cls, _t in COLS:
            if k == "identifiers_wrong" and not r["n_ident_filled"]:
                cells.append('<td class="r risk na" title="This model returned no '
                             f'manuscript number or folio reference on any of the {r["n"]} cards, leaving '
                             'no identifiers to get wrong. Read its workload figure instead.">n/a</td>')
            else:
                cells.append(f'<td class="r {cls}">{fmt(r[k], k)}</td>')
        # Sort keys only; rounded so the page rebuilds byte-identically across Python builds
        # (summation order differs in the last digit, and the visible cells are formatted anyway).
        body.append(f"""<tr data-risk="{round(r['identifiers_wrong'], 6) if r['n_ident_filled'] else 9}"
    data-f1="{round(-r['content_f1'], 6)}">
  <td class="model">{HF_IDS.get(r['label'], r['label'])}{tier}
    <span class="sz">{r['params']}</span>{flag}</td>
{chr(10).join('  ' + c for c in cells)}
</tr>""")

    base_cells = "".join(f'<td class="r {cls}">{fmt(base[k], k)}</td>' for _h, k, cls, _t in COLS)
    body.append(f"""<tr class="base" data-risk="99" data-f1="99">
  <td class="model">constants baseline
    <span class="sz" title="A model that reads nothing. It gives every card the same record: the most common value
for each field across the {n_cards} gold records, counting absence as a value. Its score is the
floor: what the collection alone supplies without reading.">reads nothing</span></td>
  {base_cells}
</tr>""")

    det = []
    for r in rows + [base]:
        det.append(f"""<tr{' class="base"' if r is base else ''}>
  <td class="model">{HF_IDS.get(r['label'], r['label'])}</td>
  <td class="r">{r['exact_fields_f1'] * 100:.1f}</td><td class="r">{r['fuzzy_fields_f1'] * 100:.1f}</td>
  <td class="r">{r['precision'] * 100:.1f}</td><td class="r">{r['recall'] * 100:.1f}</td>
  <td class="r">{r['verified_f1'] * 100:.1f}</td><td class="r">{r['corrected_f1'] * 100:.1f}</td>
  <td class="r">{r['identifier_clean_cards']}</td><td class="r">{r['schema_valid'] * 100:.0f}%</td>
</tr>""")
    dhead = "".join(f"<th class='r'>{h}</th>" for h in
                    ["identifiers &amp; flags F1", "free-text F1", "precision", "recall",
                     f"verified ({n_verified})", f"corrected ({n_corrected})",
                     "identifier-clean cards", "parsed"])

    panel, seen = worked_panel(schema, gold)
    if not (seen["blank"] and (seen["invented"] or seen["wrong"])):
        raise SystemExit(f"worked card {WORKED[1]} no longer shows both axes: {seen} — pick another")

    # One footnote, and only when a self-reported row is actually on the board.
    tier_note = "" if all(r["attested"] for r in rows) else (
        " <b>†</b> is a <b>self-reported</b> row: that result file was submitted rather than "
        "produced by the harness here, so nothing outside the file itself records which endpoint "
        "answered. Its predictions were rescored on this page like every other row.")

    head = "".join(f'<th class="r {cls}" title="{html.escape(t)}">{h}</th>' for h, _k, cls, t in COLS)
    style = (ROOT / "site" / "base.css").read_text()

    html_out = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLAM document extraction — National Library of Scotland index cards</title>
<style>{style}
  td.r{{font-variant-numeric:tabular-nums;}}
  td.score{{font-weight:600;}}
  td.na{{color:var(--muted);cursor:help;}}
  .flag,.tier{{color:var(--muted);cursor:help;}}
  .flag{{font-size:11.5px;margin-left:8px;font-family:system-ui,sans-serif;}}
  tr.grp th{{border-bottom:none;padding-bottom:3px;text-transform:none;letter-spacing:0;
    font-weight:400;}}
  tr.base td{{color:var(--muted);}}
  .tldr{{font-size:15px;line-height:1.65;margin:0 0 10px;max-width:74ch;}}
  .src{{font-size:13.5px;line-height:1.65;margin:0 0 18px;max-width:74ch;}}
  .intro{{font-size:14px;line-height:1.7;margin:0 0 24px;max-width:74ch;}}
  .intro p{{margin:0 0 12px;}} .intro p:last-child{{margin-bottom:0;}}
  .scat{{display:block;width:100%;height:auto;margin:30px 0 0;}}
  .scat .ax{{stroke:#d7dbe0;stroke-width:1;}}
  .scat .med{{stroke:#eceef2;stroke-width:1;stroke-dasharray:3 4;}}
  .scat .lead{{stroke:#d7dbe0;stroke-width:1;}}
  .scat .dot{{fill:var(--ink);}} .scat .dot.hollow{{fill:#fff;stroke:var(--ink);stroke-width:1.4;}}
  .scat .dot.base{{fill:#b9bec6;}}
  .scat text{{font-family:system-ui,-apple-system,sans-serif;fill:var(--muted);}}
  .scat .pl{{font-size:11.5px;fill:var(--ink);paint-order:stroke;stroke:#fff;
    stroke-width:3px;stroke-linejoin:round;}} .scat .pl.base{{fill:#a2a8b1;}}
  .scat circle,.scat path{{cursor:help;}}
  .scat .tick{{font-size:10.5px;}} .scat .axl{{font-size:11px;}}
  .scat .quad{{font-size:10.5px;fill:#b9bec6;}}
  .scat .best{{fill:#f4f6f4;}}
  .cap{{font-size:12px;color:var(--muted);margin:10px 0 0;max-width:78ch;line-height:1.6;}}
  details{{margin:30px 0 0;}}
  summary{{font-size:13px;color:var(--muted);cursor:pointer;padding:4px 0;}}
  details table{{margin-top:14px;font-size:13px;}}
  details.brk th:first-child{{width:230px;}}
  .worked .wlead{{font-size:14px;line-height:1.7;margin:14px 0 16px;max-width:78ch;}}
  .wgrid figure{{margin:0 0 18px;max-width:660px;}}
  .wgrid img{{width:100%;border:1px solid var(--line);display:block;}}
  .wgrid figcaption{{font-size:11px;color:var(--muted);margin-top:6px;line-height:1.5;}}
  table.wt{{font-size:12.5px;}} table.wt td{{vertical-align:top;}}
  table.wt .fp{{font-family:ui-monospace,Menlo,monospace;color:var(--muted);white-space:nowrap;}}
  table.wt tr.exc td{{color:var(--muted);}}
  .nil,.t-none{{color:#c3c8d0;}}
  .t-ok{{color:#1f7a4d;}} .t-part{{color:var(--muted);}}
  .t-work,.wnote b.w{{color:#3a5a78;font-weight:600;}}
  .t-risk,.wnote b.r{{color:#9a3b2f;font-weight:600;}}
  .wnote{{font-size:12.5px;line-height:1.7;margin:16px 0 0;max-width:82ch;}}
  .scat .bestl{{font-size:10.5px;fill:#9aa39a;}}
  .sortr{{font-size:13px;color:var(--muted);margin:0 0 16px;}}
  .sortr button{{font:inherit;background:none;border:none;padding:0 0 1px;margin-left:12px;
    color:var(--muted);cursor:pointer;border-bottom:1px solid transparent;}}
  .sortr button.on{{color:var(--ink);border-bottom-color:var(--ink);}}
  th.score{{width:96px;}} th.risk,th.work{{width:118px;}}
</style></head><body><div class="wrap">
  <h1>GLAM document extraction — National Library of Scotland index cards</h1>
  <p class="tldr">Each model must return a completed record from a card image and JSON schema.</p>
  <p class="sub">{subtitle(len(gold), len(rows))}</p>
  <p class="src">Cards from the <a href="https://www.nls.uk/">National Library of Scotland</a>'s
  manuscript catalogue, <a class="mono" href="https://huggingface.co/datasets/{NLS_DATASET}">{NLS_DATASET}</a>
  (CC0). Labels drafted by Qwen3.6-35B-A3B, checked by NLS cataloguers: {n_verified} accepted as
  drafted, {n_corrected} corrected.</p>
  <div class="intro">
  <p><b>Risk</b> is a wrong identifier or an invented field: silent, because nothing in the record
  flags it. <b>Workload</b> is a visible blank field someone must fill.
  <b>Fields right</b> is overall F1; resampling the {n_cards}
  cards moves a score by up to ±{halfwidth:.0f} points, so neighbouring rows are not separated
  here.</p>
  </div>
  <div class="sortr">Sort by<button class="on" data-k="risk">identifiers wrong</button><button
    data-k="f1">overall score</button></div>
  <table><thead>
    <tr class="grp"><th></th><th class="r risk" colspan="2">Risk · silent errors</th>
      <th class="r work">Workload · visible gaps</th><th></th></tr>
    <tr><th>Model</th>{head}</tr>
  </thead><tbody id="tb">
{chr(10).join(body)}
  </tbody></table>
{scatter(rows, base)}
{panel}

  <details class="brk"><summary>Breakdown: field types, precision, recall and verified/corrected labels</summary>
  <table><thead><tr><th>Model</th>{dhead}</tr></thead><tbody>
{chr(10).join(det)}
  </tbody></table>
  <p class="cap">Identifiers &amp; flags use exact matching (manuscript numbers, folios,
  fixed categories and true/false values); free-text uses character closeness. <b>verified</b> ({n_verified} cards): labels reviewers accepted as drafted;
  <b>corrected</b> ({n_corrected}): labels reviewers edited. The drafts came from Qwen3.6-35B-A3B,
  so the subsets are reported separately. The split does not establish whether its relatives benefit.
  <b>Identifiers wrong</b> is counted over the identifiers the card has that the model filled in;
  one added where the card has none counts as invented, one left blank as workload. <b>Identifier-clean cards</b>: every manuscript number and folio
  reference in the gold record was returned exactly right. Cards with no identifiers count as
  clean for every model. <b>Parsed</b>: share of the {n_cards} outputs usable as
  JSON; the rest scored zero.</p></details>

  <p class="foot">Gold set: <a class="mono"
  href="https://huggingface.co/datasets/{NLS_DATASET}">{NLS_DATASET}</a> (CC0) — index cards from the
  National Library of Scotland's manuscript catalogue; labels drafted by Qwen3.6-35B-A3B and checked
  by NLS cataloguers ({n_verified} accepted as drafted, {n_corrected} corrected). Every model received the same card image and JSON Schema; the same scorer evaluated all outputs. The three main-table rate columns are micro-averaged over all {n_cards} cards. Fields right
  and the breakdown are per-card means, except identifier-clean cards, which is a count;
  verified/corrected F1 uses the corresponding subset.{tier_note} Code and result files: <a href="https://github.com/davanstrien/glam-extraction-benchmark">github.com/davanstrien/glam-extraction-benchmark</a></p>
</div>
<script>
const tb=document.getElementById('tb'),bs=document.querySelectorAll('.sortr button');
bs.forEach(b=>b.onclick=()=>{{bs.forEach(x=>x.classList.remove('on'));b.classList.add('on');
const k=b.dataset.k;[...tb.rows].sort((a,z)=>a.dataset[k]-z.dataset[k]).forEach(r=>tb.append(r));
tb.append(tb.querySelector('tr.base'));}});
</script>
</body></html>"""
    OUT.write_text(html_out)
    print(f"wrote {OUT} ({len(rows)} models + baseline)")
    for r in rows:
        print(f"  {r['label']:24s} risk {r['identifiers_wrong']*100:5.1f}%  invented "
              f"{r['invented_fields']*100:5.1f}%  blank {r['fields_left_blank']*100:5.1f}%  "
              f"F1 {r['content_f1']*100:5.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="build site/index.html")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="build the page from unfinished files and files with transport errors too")
    args = ap.parse_args(argv)
    build(allow_incomplete=args.allow_incomplete)


if __name__ == "__main__":
    main()
