# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=5,<6", "jsonschema>=4,<5", "Pillow>=12,<13"]
# ///
"""Build a static benchmark Space from a validated local config and raw predictions.

No inference or upload. Use --legacy-results to label predictions made with the earlier source schema.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path

from board import risk_sort_key, summarise
from dataset_contract import evaluation_identity, read_config, scoring_schema, validate_config
from result_io import discover_result_files, is_attested
from scorer import SCORER_VERSION, score
from version import __version__

METRICS = {
    "identifiers_wrong": {"label": "Identifiers wrong", "direction": "lower", "scale": [0, 1]},
    "invented_fields": {"label": "Invented fields", "direction": "lower", "scale": [0, 1]},
    "fields_left_blank": {"label": "Fields left blank", "direction": "lower", "scale": [0, 1]},
    "content_f1": {"label": "Fields right (F1)", "direction": "higher", "scale": [0, 1]},
}


def check_identity(document, manifest, revision, path, legacy):
    if legacy:
        if document["dataset"]["inference_revision"] != manifest["source"]["revision"]:
            raise ValueError("legacy result does not match source revision")
        return
    expected = {"id": manifest["benchmark_id"], "inference_revision": revision, **evaluation_identity(manifest)}
    for key, value in expected.items():
        if document["dataset"].get(key) != value:
            raise ValueError(f"{path.name}: dataset.{key} differs from selected config")


def evaluated_rows(dataset, manifest, revision, results, legacy):
    gold = {row["id"]: row for row in dataset}
    expected_id = manifest["source"]["repo_id"] if legacy else manifest["benchmark_id"]
    paths = discover_result_files(results, gold_ids=set(gold), expected_dataset_id=expected_id)
    if not paths:
        raise ValueError("no complete prediction files found")
    summaries, evidence = [], []
    for path in paths:
        document = json.loads(path.read_text())
        check_identity(document, manifest, revision, path, legacy)
        scores = []
        for item in document["items"]:
            row = gold[item["id"]]
            scored = score(json.loads(row["expected_output"]), item.get("prediction") or "",
                           scoring_schema(json.loads(row["target_schema"])),
                           exclude=tuple(manifest["scoring"]["exclude"]))
            metadata = json.loads(row.get("provenance") or "{}")
            scores.append({**scored, "label_status": metadata.get("_label_status")})
        summary = summarise(document["model"]["label"], document["model"].get("params", "?"), scores,
                            attested=is_attested(document))
        summary["model_id"] = document["model"].get("id")
        summaries.append(summary)
        evidence.append({"file": path.name, "model": document["model"],
                         "inference_dataset": document["dataset"], "legacy_protocol": bool(legacy)})
    return sorted(summaries, key=risk_sort_key), evidence


def parameter_billions(value):
    """Read total size from labels such as '235B (22B act)' or '500M'."""
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([BM])\b", str(value), re.IGNORECASE)
    if not match:
        return None
    return float(match[1]) / (1000 if match[2].upper() == "M" else 1)


def table_body(rows):
    parts = []
    for row in rows:
        cells = []
        for key in METRICS:
            value = row[key]
            display = f"{value * 100:.1f}" + ("" if key == "content_f1" else "%")
            if key == "identifiers_wrong" and not row["n_ident_filled"]:
                display, value = "—", ""
            cells.append(f'<td class="r" data-value="{value}">{display}</td>')
        params = parameter_billions(row["params"])
        size = html.escape(str(row["params"])) if params is not None else "size unknown"
        label = html.escape(row["label"])
        model_id = row.get("model_id") or ""
        name = label
        if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id):
            name = (f'<a href="https://huggingface.co/{html.escape(model_id)}" '
                    f'target="_blank" rel="noopener noreferrer" title="{html.escape(model_id)}">{label}</a>')
        parts.append(f'<tr data-params="{params if params is not None else ""}">'
                     f'<td data-value="{label}">{name} <small>{size}</small></td>'
                     + "".join(cells) + "</tr>")
    return "\n".join(parts)


def page(manifest, revision, rows, legacy):
    esc = html.escape
    dataset_url = f'https://huggingface.co/datasets/{manifest["benchmark_id"]}/tree/{revision}'
    source_url = f'https://huggingface.co/datasets/{manifest["source"]["repo_id"]}/tree/{manifest["source"]["revision"]}'
    warning = ("Historical results: these models received the earlier schema without nullable fields. "
               "Their cached predictions are rescored against the new export. "
               "Models have not yet been rerun with its nullable schema." if legacy else
               "Predictions were generated for this benchmark config and revision.")
    heading = "".join(
        f'<th class="r" scope="col"><button type="button" data-column="{index}" '
        f'data-direction="{"desc" if meta["direction"] == "higher" else "asc"}">'
        f'{meta["label"]} <span aria-hidden="true">↕</span></button></th>'
        for index, meta in enumerate(METRICS.values(), start=1))
    size_buttons = '<button type="button" data-max-params="all" aria-pressed="true">All</button>'
    for limit in (1, 3, 6, 12, 32, 128, 500):
        available = any((size := parameter_billions(row["params"])) is not None and size < limit for row in rows)
        disabled = "" if available else ' disabled title="No models in this size range"'
        size_buttons += (f'<button type="button" data-max-params="{limit}" aria-pressed="false"'
                         f'{disabled}>&lt;{limit}B</button>')
    table_script = (Path(__file__).resolve().parent.parent / "site/benchmark-table.js").read_text()
    plot_script = (Path(__file__).resolve().parent.parent / "site/benchmark-plot.js").read_text()
    css = (Path(__file__).resolve().parent.parent / "site/base.css").read_text()
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GLAM extraction benchmark</title>
<style>{css}
.wrap{{max-width:1080px}}.notice{{border-left:3px solid #8b6c36;padding:8px 16px;background:#faf8f2}}
small{{color:#666}}button,select{{font:inherit;margin:0 12px 14px 0}}.scroll{{overflow-x:auto}}
th button{{margin:0;padding:6px 0;border:0;background:transparent;color:inherit;
font:inherit;letter-spacing:inherit;text-transform:inherit;cursor:pointer;text-align:inherit}}
th button:hover{{color:var(--ink)}}th button:focus-visible{{outline:2px solid #3566a3;outline-offset:4px}}
.size-bar{{display:flex;align-items:center;flex-wrap:wrap;gap:10px 16px;padding:14px 0;
border-top:1px solid var(--line);margin-top:20px}}
.size-options{{display:flex;align-items:center;gap:6px;overflow-x:auto;max-width:100%}}
.size-options button{{margin:0;border:0;border-radius:14px;padding:4px 10px;background:transparent;
color:var(--muted);font-size:14px;white-space:nowrap;cursor:pointer}}
.size-options button[aria-pressed="true"]{{background:var(--ink);color:#fff}}
.size-options button:hover:not(:disabled):not([aria-pressed="true"]){{background:#f1f2f4;color:var(--ink)}}
.size-options button:disabled{{color:#9ca3af;cursor:default}}
.size-options button:focus-visible{{outline:2px solid #3566a3;outline-offset:2px}}
.size-label,#model-count{{font-size:13px;color:var(--muted);white-space:nowrap}}
#model-count{{margin-left:auto}}
.filter-note{{font-size:13px;color:var(--muted);margin:0 0 18px}}[hidden]{{display:none!important}}
.plot-section{{margin-top:30px}}.plot-section h2{{font-size:18px;margin-bottom:6px}}
.plot-section p{{font-size:13px;color:var(--muted)}}#size-plot{{width:100%;height:auto;display:block}}
.plot-grid{{stroke:var(--line);stroke-width:1}}.plot-tick,.plot-axis{{font-size:11px;fill:var(--muted)}}
.plot-frontier{{fill:none;stroke:#286c66;stroke-width:2;stroke-dasharray:5 4;opacity:.7}}
.plot-point{{fill:#a3aab4;stroke:white;stroke-width:1.5}}.frontier-point{{fill:#286c66}}
#size-plot a:hover .plot-point,#size-plot a:focus .plot-point{{stroke:var(--ink);stroke-width:3}}
.plot-label{{font-size:11px;fill:var(--ink);pointer-events:none}}
details{{margin-top:24px}}img{{max-width:100%;max-height:440px}}pre{{white-space:pre-wrap;font-size:12px}}
</style></head><body><main class="wrap">
<h1>GLAM extraction benchmark</h1><p>{esc(manifest["title"])}</p>
<label for="dataset">Dataset </label><select id="dataset"><option>{esc(manifest["config"])}</option></select>
<p class="sub">{manifest["item_count"]} documents · {len(rows)} models · {esc(manifest["institution"]["name"])}</p>
<p>Images and checked labels: <a href="{source_url}">{esc(manifest["source"]["repo_id"])}</a>
({esc(manifest["license"])}). Labels drafted by {esc(manifest["label_production"]["draft_model"])}
and reviewed by {esc(manifest["label_production"]["review"])}.</p>
<p class="notice">{warning}</p>
<p><b>Risk</b> is a wrong identifier or an invented field. <b>Workload</b> is a blank field that
someone must fill. F1 combines extraction precision and recall. This is a small evaluation set;
neighbouring scores should not be read as a definitive ranking.</p>
<div class="size-bar"><span class="size-label" id="size-label">Parameter size</span>
<div class="size-options" role="group" aria-labelledby="size-label">{size_buttons}</div>
<span id="model-count" role="status">{len(rows)} of {len(rows)} models</span></div>
<p class="filter-note">Uses total parameters, including for mixture-of-experts models.
Unknown sizes appear under All. Click a column heading to sort; click again to reverse.</p>
<div class="scroll"><table><thead><tr><th scope="col"><button type="button" data-column="0"
data-direction="asc">Model <span aria-hidden="true">↕</span></button></th>{heading}</tr></thead>
<tbody id="results">{table_body(rows)}</tbody></table></div>
<p id="empty-results" hidden>No models match this size limit.</p>
<p class="foot">Rates are micro-averaged over documents; F1 is the mean per-document score.
An em dash means no gold identifiers were filled, not zero errors. Excluded fields:
{esc(', '.join(manifest['scoring']['exclude']) or 'none')}.</p>
<section class="plot-section" aria-labelledby="plot-title"><h2 id="plot-title">Size vs. extraction quality</h2>
<p>Upper-left is better: smaller models, higher F1. Teal points and the dashed line show the
observed Pareto frontier: no other model uses as few parameters and scores as highly.
Small score differences may be noise.</p>
<svg id="size-plot" viewBox="0 0 760 360" role="group" aria-label="Model size versus extraction F1"></svg>
<p id="plot-note"></p></section>
<details><summary>Inspect an example document and its checked output</summary>
<img src="example.jpg" alt="Example document from the selected evaluation dataset"><pre id="gold"></pre></details>
<details><summary>Reproducibility and downloads</summary><p>Config: <code>{esc(manifest['config'])}</code>;
split: <code>{esc(manifest['split'])}</code>; <a href="{dataset_url}">dataset revision {revision[:12]}</a>.
Harness {__version__}; scorer {SCORER_VERSION}. Raw prediction files retain their original inference provenance.</p>
<p><a href="scores.json">Scores and run provenance</a> · <a href="manifest.json">Dataset manifest</a> ·
<a href="https://github.com/davanstrien/glam-extraction-benchmark">Code and raw predictions</a></p></details>
</main><script>{plot_script}
{table_script}
fetch('example.json').then(r=>r.json()).then(x=>document.getElementById('gold').textContent=JSON.stringify(x,null,2));
</script></body></html>'''


def build(root, config, revision, results, output, legacy=False):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("dataset revision must be an immutable Hub commit")
    if output.exists():
        raise ValueError("output must be a new directory")
    report = validate_config(root, config)
    if not report["valid"]:
        raise ValueError(json.dumps(report))
    dataset, manifest = read_config(root, config)
    rows, evidence = evaluated_rows(dataset, manifest, revision, results, legacy)
    output.mkdir(parents=True)
    (output / "index.html").write_text(page(manifest, revision, rows, bool(legacy)))
    from PIL import Image
    import io
    with Image.open(io.BytesIO(dataset[0]["image"]["bytes"])) as image:
        image.convert("RGB").save(output / "example.jpg")
    (output / "example.json").write_text(dataset[0]["expected_output"])
    shutil.copy2(root / config / "manifest.json", output / "manifest.json")
    artifact = {"benchmark": {"id": manifest["benchmark_id"], "revision": revision, **evaluation_identity(manifest)},
                "harness_version": __version__, "scorer_version": SCORER_VERSION, "metrics": METRICS,
                "rows": rows, "prediction_files": evidence, "legacy_protocol": bool(legacy)}
    (output / "scores.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    (output / "README.md").write_text('---\ntitle: GLAM extraction benchmark\nsdk: static\napp_file: index.html\n'
                                     'datasets:\n  - ' + manifest["benchmark_id"] + '\n---\n'
                                     '# GLAM extraction benchmark\n\nPrivate preview. '
                                     'The page states the protocol used for each result set.\n')
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--legacy-results", action="store_true", help="label runs from the earlier source schema")
    args = parser.parse_args()
    artifact = build(args.root, args.config, args.revision, args.results, args.output, args.legacy_results)
    print(f"Built {len(artifact['rows'])} models; legacy protocol: {artifact['legacy_protocol']}")


if __name__ == "__main__":
    main()
