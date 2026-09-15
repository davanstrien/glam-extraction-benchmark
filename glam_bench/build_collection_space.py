# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=5,<6", "jsonschema>=4,<5", "Pillow>=12,<13"]
# ///
"""Build linked config pages from a JSON list of pinned datasets and result folders."""
import argparse
import json
from pathlib import Path

from aggregate_scores import aggregate_scores
from build_overall_page import overall_page
from build_space import build
from dataset_contract import require_config_name


def build_collection(runs, output):
    if not runs:
        raise ValueError("at least one config run is required")
    names = [run["config"] for run in runs]
    for name in names:
        require_config_name(name)
    if len(set(names)) != len(names):
        raise ValueError("each config must appear once")
    artifacts = []
    for index, run in enumerate(runs):
        prefix = "" if index == 0 else "../"
        navigation = [{"config": other["config"],
                       "url": prefix + ("index.html" if i == 0 else other["config"] + "/index.html")}
                      for i, other in enumerate(runs)]
        navigation.insert(0, {"config": "overall", "label": "Overall", "url": prefix + "overall.html"})
        destination = output if index == 0 else output / run["config"]
        artifacts.append(build(Path(run["root"]), run["config"], run["revision"],
                               Path(run["results"]), destination,
                               legacy=run.get("legacy", False), navigation=navigation))
    (output / "configs.json").write_text(json.dumps(
        [artifact["benchmark"] for artifact in artifacts], indent=2) + "\n")
    aggregate = aggregate_scores(artifacts)
    navigation = [{"config": "overall", "label": "Overall", "url": "overall.html"}] + [
        {"config": run["config"], "url": "index.html" if i == 0 else run["config"] + "/index.html"}
        for i, run in enumerate(runs)]
    (output / "overall-scores.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    (output / "overall.html").write_text(overall_page(aggregate, navigation))
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="JSON list: root, config, revision, results for each config")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = build_collection(json.loads(args.runs.read_text()), args.output)
    print(f"Built {len(artifacts)} separate config leaderboards")
