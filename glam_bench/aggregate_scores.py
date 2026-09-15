"""Summarize completed, validated per-config score artifacts without rerunning models.

The default gives each dataset equal weight. ``cards`` weights its mean by the
number of cards, equivalent to a mean over cards (not a pooled field-level F1).
Only models present in every selected config receive an aggregate score.
"""
from copy import deepcopy


def aggregate_scores(artifacts, weighting="datasets"):
    """Return aggregate rows and the per-config evidence used to calculate them.

Inputs are score dictionaries returned by ``build_space.build``; their runs have
already passed publication validation. Models are joined by Hub ID, not label.
"""
    if weighting not in {"datasets", "cards"}:
        raise ValueError("weighting must be 'datasets' or 'cards'")
    artifacts = list(artifacts)
    if not artifacts:
        raise ValueError("select at least one dataset config")
    configs = [artifact["benchmark"]["config"] for artifact in artifacts]
    if len(configs) != len(set(configs)):
        raise ValueError("select each dataset config only once")
    indexed = [_model_rows(artifact) for artifact in artifacts]
    all_models = set.union(*(set(rows) for rows in indexed))
    shared_models = set.intersection(*(set(rows) for rows in indexed))
    result_rows = []
    for model_id in shared_models:
        source_rows = [rows[model_id] for rows in indexed]
        weights = [row["n"] if weighting == "cards" else 1 for row in source_rows]
        first = source_rows[0]
        result_rows.append({
            "model_id": model_id, "label": first["label"], "params": first["params"],
            "content_f1": sum(row["content_f1"] * weight
                              for row, weight in zip(source_rows, weights, strict=True)) / sum(weights),
            "n": sum(row["n"] for row in source_rows),
            "config_scores": {config: row["content_f1"]
                              for config, row in zip(configs, source_rows, strict=True)},
            "config_provenance": {
                config: {"n": row["n"], "prediction_files": deepcopy([
                    item for item in artifact.get("prediction_files", [])
                    if item["model"].get("id") == model_id])}
                for config, row, artifact in zip(configs, source_rows, artifacts, strict=True)},
        })
    result_rows.sort(key=lambda row: (-row["content_f1"], row["model_id"]))
    return {
        "rows": result_rows,
        "aggregation": {
            "weighting": weighting,
            "method_label": ("Mean F1 across datasets" if weighting == "datasets"
                             else "Mean F1 weighted by card count"),
            "configs": configs,
            "coverage": "all selected configs",
            "dataset_weights": ({config: 1 / len(configs) for config in configs}
                                if weighting == "datasets" else None),
            "sources": {config: deepcopy({key: artifact[key] for key in (
                "benchmark", "harness_version", "scorer_version", "legacy_protocol")
                if key in artifact}) for config, artifact in zip(configs, artifacts, strict=True)},
        },
        "excluded_models": sorted(all_models - shared_models),
    }


def _model_rows(artifact):
    rows = {row["model_id"]: row for row in artifact["rows"]}
    if len(rows) != len(artifact["rows"]):
        raise ValueError("a config has multiple results for the same model_id")
    return rows
