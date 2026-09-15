"""Render an overall F1 page while retaining separate config leaderboards."""
import html
from pathlib import Path

from build_space import load_styles, parameter_billions, table_body


def overall_page(aggregate, navigation):
    esc = html.escape
    rows = aggregate['rows']
    configs = aggregate['aggregation']['configs']
    weighting = aggregate['aggregation']['weighting']
    explanation = (f'Overall F1 is the average of the dataset F1 scores. '
                   f'Each dataset has equal weight ({100 / len(configs):g}% each).' if weighting == 'datasets' else
                   'Every document contributes equally: dataset F1 scores are weighted by '
                   'their number of evaluated documents.')
    options = ''.join(f'<option data-config="{esc(x["config"], quote=True)}" value="{esc(x["url"], quote=True)}"'
                      + (' selected' if x['config'] == 'overall' else '')
                      + f'>{esc(x.get("label", x["config"]))}</option>' for x in navigation)
    sizes = '<button type="button" data-max-params="all" aria-pressed="true">All</button>'
    for limit in (1, 3, 6, 12, 32, 128, 500):
        available = any((n := parameter_billions(row['params'])) is not None and n < limit for row in rows)
        sizes += (f'<button type="button" data-max-params="{limit}" aria-pressed="false"'
                  + ('' if available else ' disabled') + f'>&lt;{limit}B</button>')
    omitted = len(aggregate.get('excluded_models', []))
    coverage = f'{omitted} model(s) awaiting complete dataset coverage are omitted.' if omitted else ''
    site = Path(__file__).resolve().parent.parent / 'site'
    table_script = (site / 'benchmark-table.js').read_text()
    plot_script = (site / 'benchmark-plot.js').read_text()
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GLAM extraction benchmark · Overall</title>
<style>{load_styles()}</style></head><body><main class="wrap">
<h1>GLAM extraction benchmark</h1><p>Overall extraction quality</p>
<label for="dataset">Dataset </label><select id="dataset">{options}</select>
<p class="sub">{len(configs)} datasets · {len(rows)} models with complete results</p>
<p>{explanation} Only models evaluated on every selected dataset are included. {coverage}</p>
<p>Choose a dataset above to inspect its results and methodology. These are small evaluation sets;
nearby scores should not be read as a definitive ranking.</p>
<div class="size-bar"><span class="size-label" id="size-label">Parameter size</span>
<div class="size-options" role="group" aria-labelledby="size-label">{sizes}</div>
<span id="model-count" role="status">{len(rows)} of {len(rows)} models</span></div>
<p class="filter-note">Uses total parameters, including for mixture-of-experts models.
Click a column heading to sort; click again to reverse.</p>
<div class="scroll"><table><thead><tr>
<th scope="col"><button type="button" data-column="0" data-direction="asc">Model <span aria-hidden="true">↕</span></button></th>
<th class="r" scope="col"><button type="button" data-column="1" data-direction="desc">Overall F1 <span aria-hidden="true">↕</span></button></th>
</tr></thead><tbody id="results" data-f1-column="1" data-default-column="1">
{table_body(rows, {'content_f1': {}})}</tbody></table></div>
<p id="empty-results" hidden>No models match this size limit.</p>
<p class="foot">Scores are shown on a 0–100 scale. Dataset weights, component scores and pinned
revisions are recorded in <a target="_blank" rel="noopener noreferrer" href="overall-scores.json">the score download</a>.</p>
<section class="plot-section" aria-labelledby="plot-title"><h2 id="plot-title">Size vs. overall extraction quality</h2>
<p>Upper-left is better: smaller models, higher overall F1. Teal points show the observed Pareto frontier.</p>
<svg id="size-plot" data-f1-column="1" viewBox="0 0 760 360" role="group" aria-label="Model size versus overall extraction F1"></svg>
<p id="plot-note"></p></section>
<details><summary>Sources and reproducibility</summary>
<p><a target="_blank" rel="noopener noreferrer" href="overall-scores.json">Overall scores and component provenance</a> ·
<a target="_blank" rel="noopener noreferrer" href="configs.json">Dataset configurations and revisions</a> ·
<a target="_blank" rel="noopener noreferrer" href="https://github.com/davanstrien/glam-extraction-benchmark">Code</a> ·
<a target="_blank" rel="noopener noreferrer" href="https://huggingface.co/buckets/small-models-for-glam/glam-extraction-results">Raw predictions</a></p>
<p>The overall view combines the selected dataset results; it does not create a new dataset or change individual scores.</p>
</details></main><script>{plot_script}\n{table_script}
</script></body></html>'''
