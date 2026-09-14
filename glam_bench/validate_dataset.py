# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=5,<6", "jsonschema>=4,<5", "Pillow>=12,<13"]
# ///
"""Validate a local benchmark export: uv run glam_bench/validate_dataset.py PATH --config NAME."""
import argparse
import json

from dataset_contract import validate_config
from version import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--config", required=True)
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()
    try:
        report = validate_config(args.root, args.config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
