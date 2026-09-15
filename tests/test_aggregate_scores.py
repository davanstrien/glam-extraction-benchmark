"""Aggregate rankings depend on explicit weighting and complete config coverage."""
from copy import deepcopy

import pytest

from aggregate_scores import aggregate_scores


def test_unequal_datasets_change_weighted_ranking_and_exclude_missing_models():
    nls = {"benchmark": {"config": "nls", "revision": "a" * 40},
           "scorer_version": "test-v1", "rows": [
               {"model_id": "org/a", "label": "A", "params": "2B", "n": 98, "content_f1": .2},
               {"model_id": "org/b", "label": "B", "params": "4B", "n": 98, "content_f1": .5},
               {"model_id": "org/missing", "label": "Missing", "params": "8B", "n": 98, "content_f1": 1.},
           ], "prediction_files": [{"file": "A.json", "model": {"id": "org/a"},
                                    "inference_dataset": {"inference_revision": "a" * 40}}]}
    harvard = {"benchmark": {"config": "harvard", "revision": "b" * 40}, "rows": [
        {"model_id": "org/a", "label": "A renamed", "params": "2B", "n": 30, "content_f1": 1.},
        {"model_id": "org/b", "label": "B", "params": "4B", "n": 30, "content_f1": .5},
    ]}
    inputs = [nls, harvard]
    original = deepcopy(inputs)
    macro = aggregate_scores(inputs)
    weighted = aggregate_scores(inputs, weighting="cards")

    # Equal datasets reward A's strong smaller dataset; card weighting flips that ranking.
    assert [row["model_id"] for row in macro["rows"]] == ["org/a", "org/b"]
    assert macro["rows"][0]["content_f1"] == pytest.approx(.6)
    assert [row["model_id"] for row in weighted["rows"]] == ["org/b", "org/a"]
    assert weighted["rows"][1]["content_f1"] == pytest.approx(.3875)
    assert macro["excluded_models"] == weighted["excluded_models"] == ["org/missing"]
    assert macro["rows"][0]["config_scores"] == {"nls": .2, "harvard": 1.}
    assert macro["rows"][0]["n"] == 128
    assert macro["aggregation"]["sources"]["nls"]["benchmark"]["revision"] == "a" * 40
    assert macro["rows"][0]["config_provenance"]["nls"]["prediction_files"] == nls["prediction_files"]
    assert inputs == original
