"""Result files on disk: the envelope, atomic writes, and the resume contract.

    <LABEL>.json.partial    a run in progress; rewritten in full after every row, `complete` false
    <LABEL>.json            a finished run, `complete` true
    <LABEL>.limit<N>.json   a `--limit` smoke test, kept out of the canonical name on purpose
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

PARTIAL_SUFFIX = ".partial"

# The name `result_path` gives a `--limit` run, as a pattern. Both the writer and the reader go
# through this constant so a change to one cannot leave the other matching the old shape; the
# pair is pinned by test_a_limit_file_is_recognised_from_the_name_result_path_gives_it.
LIMIT_NAME = re.compile(r"\.limit\d+\.json$")

# What must be identical between an existing result file and the run resuming into it. The
# `generation` four are nested under one block in the envelope. Deliberately NOT here:
# `scoring.scorer_version` (scores are recomputed by rescore.py, so drift is a warning) and the
# endpoint/provider (moving router-failed rows onto a pinned serve is a supported resume).
RESUME_IDENTITY = (
    ("dataset", "id"),
    ("dataset", "inference_revision"),
    ("dataset", "item_ids_sha256"),
    ("dataset", "config"),
    ("dataset", "split"),
    ("model", "id"),
    ("model", "kind"),
    ("generation", "max_tokens"),
    ("generation", "temperature"),
    ("generation", "guided"),
    ("generation", "request_options"),
)

SELECT_MODES = ("errors", "unparseable", "all")


# --- paths ------------------------------------------------------------------------------------

def result_path(outdir: Path, label: str, limit: int | None = None) -> Path:
    """The final result file for one model."""
    if limit is None:
        return outdir / f"{label}.json"
    return outdir / f"{label}.limit{limit}.json"      # must keep matching LIMIT_NAME


def partial_path(final: Path) -> Path:
    """The in-progress checkpoint beside a final result file."""
    return final.with_name(final.name + PARTIAL_SUFFIX)


# --- the envelope -----------------------------------------------------------------------------

def item_ids_sha256(ids) -> str:
    """The fingerprint of an item slice: its ids, in order, newline-joined."""
    joined = "\n".join(str(item_id) for item_id in ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def duplicate_ids(ids) -> list[str]:
    """The ids that appear more than once, each named once, in the order they first repeat."""
    seen = set()
    repeated = []
    for item_id in ids:
        if item_id in seen and item_id not in repeated:
            repeated.append(item_id)
        seen.add(item_id)
    return repeated


def refuse_duplicate_ids(ids, source: str) -> None:
    """Stop the run when a slice names the same item twice."""
    repeated = duplicate_ids(ids)
    if not repeated:
        return
    raise SystemExit(
        f"{source}: {len(repeated)} duplicate item id(s): {_named(repeated)}\n"
        "An id names one item, and everything downstream keys rows by it, so the answers to two "
        "items sharing an id would collapse into one row. Fix the dataset before running over it.")


def count_transport_errors(rows) -> int:
    return sum(1 for row in rows if row.get("inference_status") == "transport_error")


def build_run_block(started_at: str, finished_at: str, git_sha, complete: bool,
                    rows, resumed_at, model_revision_moved: bool = False) -> dict:
    """The `run` block. `model_revision_moved` appears only when it is true, like `resumed_at`: it
    marks a file whose rows record two different `model_revision_hub_main` values because the
    repo's main moved between sessions.
    """
    block = {
        "started_at": started_at,
        "finished_at": finished_at,
        "harness_git_sha": git_sha,
        "complete": complete,
        "transport_errors": count_transport_errors(rows),
    }
    if resumed_at:
        block["resumed_at"] = list(resumed_at)
    if model_revision_moved:
        block["model_revision_moved"] = True
    return block


def build_envelope(format_version: int, model_block: dict, dataset_block: dict,
                   generation_block: dict, scoring_block: dict, run_block: dict,
                   produced_by_block: dict, rows) -> dict:
    """The whole result file, in its one canonical key order. `generation` sits with the first group
    because it is part of the question, which is also why it is in RESUME_IDENTITY.
    """
    return {
        "format_version": format_version,
        "model": model_block,
        "dataset": dataset_block,
        "generation": generation_block,
        "scoring": scoring_block,
        "run": run_block,
        "produced_by": dict(produced_by_block),
        "items": list(rows),
    }


# --- atomic write -----------------------------------------------------------------------------

def atomic_write_json(path: Path, payload: dict) -> None:
    """Write `payload` to `path` so that `path` is never a half-written file. The temp file is
    created in the *same directory* as the target, which keeps it on the same filesystem and makes
    `os.replace` an atomic rename. A reader (or a KeyboardInterrupt landing mid-write) therefore
    sees either the previous complete file or the new one, never a splice of the two.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp",
        delete=False)
    temp_name = handle.name
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


# --- reading and resuming ---------------------------------------------------------------------

def load_result_file(path: Path) -> dict:
    """Parse an existing result file, or stop the run with a readable message."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {type(exc).__name__}: {exc}") from exc


def find_resume_source(final: Path, partial: Path) -> Path | None:
    """Which file a `--resume` should continue from, or None when there is nothing to resume."""
    if partial.exists():
        return partial
    if final.exists():
        return final
    return None


def field_at(document: dict, block: str, key: str):
    section = document.get(block)
    if not isinstance(section, dict):
        return None
    return section.get(key)


def resume_mismatches(existing: dict, current_blocks: dict) -> list[str]:
    """The RESUME_IDENTITY fields that differ between an existing file and this run. `current_blocks`
    is {block name: block} for every block RESUME_IDENTITY names, so the two sides of the
    comparison have the same shape and adding a field to the identity is a one-line change here
    and none at all in this function.
    """
    differences = []
    for block, key in RESUME_IDENTITY:
        was = field_at(existing, block, key)
        now = field_at(current_blocks, block, key)
        if was != now:
            differences.append(f"  {block}.{key}: file has {was!r}, this run has {now!r}")
    return differences


def validate_resume_target(existing: dict, current_blocks: dict, source: Path) -> None:
    """Refuse to resume a file that describes a different run."""
    if not isinstance(existing.get("items"), list):
        raise SystemExit(f"cannot resume {source}: no 'items' list — this is not a result file.")

    differences = resume_mismatches(existing, current_blocks)
    if differences:
        raise SystemExit(
            f"cannot resume {source}: it describes a different run.\n"
            + "\n".join(differences)
            + "\nResume needs the same dataset snapshot, the same item slice in the same order,"
            "\nthe same model, and the same generation settings. Move or delete that file to run"
            "\nthis configuration fresh.")


def scorer_version_warning(existing: dict, scorer_version: str) -> str | None:
    """A message when kept rows were scored by a different scorer than this run uses."""
    was = field_at(existing, "scoring", "scorer_version")
    if was is None or was == scorer_version:
        return None
    return (f"warning: kept rows were scored by scorer {was}, this run scores with "
            f"{scorer_version}. Scores in this file will be mixed until you rerun rescore.py.")


def rows_by_id(existing: dict) -> dict:
    """{item id: row} for an existing result file."""
    indexed = {}
    for row in existing.get("items", []):
        if isinstance(row, dict) and "id" in row:
            indexed[row["id"]] = row
    return indexed


def previous_attempts(row) -> int:
    """How many attempts an existing row already records; 0 when there is no such row."""
    if not row:
        return 0
    attempts = row.get("attempts", 1)
    if isinstance(attempts, int) and attempts > 0:
        return attempts
    return 1


def needs_redo(row: dict, select: str) -> bool:
    """Whether an existing row must be inferred again under `--select <select>`. errors transport
    failures, scorer crashes, and anything not recorded as a clean "ok" unparseable the above,
    plus rows the model answered but the scorer could not parse all everything
    """
    if select == "all":
        return True
    if row.get("inference_status") != "ok":
        return True
    if row.get("scorer_error"):
        return True
    if select == "unparseable" and float(row.get("schema_valid", 0.0)) == 0.0:
        return True
    return False


# --- the submission contract ----------------------------------------------------------------------
# A file in results/ is a *submission*; `harness.py` is one producer of submissions — the
# reference one. Contract in prose: docs/SUBMISSION.md.

SUBMISSION_FORMAT_VERSION = 2

PRODUCERS = ("harness", "paratext", "space", "other")

# What a file with no `produced_by` is read as: self-reported.
UNKNOWN_PRODUCED_BY = {"producer": "other", "attested": False}

# What the harness writes into its own files.
HARNESS_PRODUCED_BY = {"producer": "harness", "attested": True}

# (block, key, why it is needed). The two strings a submission must carry; each non-empty.
REQUIRED_STRINGS = (
    ("model", "label", "the name this row gets on the board"),
    ("dataset", "inference_revision", "the commit sha of the gold snapshot that was read"),
)

# (block, key, what it is). Absent or null is fine; present means a non-empty string.
OPTIONAL_STRINGS = (
    ("model", "id", "a Hub id, or free text for a model that is not on the Hub"),
    ("dataset", "id", "the Hub dataset the images and schema came from"),
)

COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

# `dataset.item_ids_sha256`, as `item_ids_sha256` produces it.
ITEM_IDS_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# How many ids a problem line names before it stops listing and says how many are left.
IDS_SHOWN = 5


def produced_by(document: dict) -> dict:
    """Who made this file, defaulting an absent or malformed block to an unknown producer."""
    block = document.get("produced_by")
    if not isinstance(block, dict):
        return dict(UNKNOWN_PRODUCED_BY)
    return block


# What a row's `provenance` must actually say for the board to read the file as attested: where
# the answer came from (who served it, and at what endpoint) and when.
ATTESTED_PROVENANCE_KEYS = ("served_by", "endpoint", "timestamp")


def provenance_backs_the_claim(row) -> bool:
    """Whether one row's provenance says where its answer came from, and when."""
    if not isinstance(row, dict):
        return False
    block = row.get("provenance")
    if not isinstance(block, dict):
        return False
    for key in ATTESTED_PROVENANCE_KEYS:
        value = block.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


# Producers whose files the board may show as attested.
ATTESTING_PRODUCERS = frozenset({"harness"})


def is_attested(document: dict) -> bool:
    """Whether the board may show this submission as attested rather than self-reported."""
    block = produced_by(document)
    if block.get("producer") not in ATTESTING_PRODUCERS:
        return False
    if block.get("attested") is not True:
        return False
    rows = document.get("items")
    if not isinstance(rows, list) or not rows:
        return False
    return all(provenance_backs_the_claim(row) for row in rows)


def _named(ids) -> str:
    """A few ids, in order, and a count of the ones not shown."""
    listed = sorted(ids)
    shown = ", ".join(listed[:IDS_SHOWN])
    if len(listed) > IDS_SHOWN:
        return f"{shown}, … and {len(listed) - IDS_SHOWN} more"
    return shown


def check_format_version(document: dict) -> list[str]:
    """Absent means current. Present means it must be current."""
    version = document.get("format_version")
    if version is None:
        return []
    if version != SUBMISSION_FORMAT_VERSION:
        return [f"format_version: {version!r} — this board reads format_version "
                f"{SUBMISSION_FORMAT_VERSION}"]
    return []


def check_required_fields(document: dict) -> list[str]:
    """The identifying strings. A missing block is reported once, as its required key."""
    problems = []
    for block, key, why in REQUIRED_STRINGS:
        section = document.get(block)
        value = section.get(key) if isinstance(section, dict) else None
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{block}.{key}: missing — {why}")
    for block, key, why in OPTIONAL_STRINGS:
        section = document.get(block)
        value = section.get(key) if isinstance(section, dict) else None
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{block}.{key}: {value!r} is not a string — leave it out, or give "
                            f"{why}")
    return problems


def check_inference_revision(document: dict) -> list[str]:
    """The snapshot must be a commit, not a branch: `main` names a moving target."""
    revision = field_at(document, "dataset", "inference_revision")
    if not isinstance(revision, str) or not revision.strip():
        return []                         # check_required_fields already said so
    if COMMIT_SHA.match(revision):
        return []
    return [f"dataset.inference_revision: {revision!r} is not a 40-character commit sha — "
            "resolve it once with HfApi().dataset_info(<dataset>).sha, or take the value the "
            "harness prints"]


def check_produced_by(document: dict) -> list[str]:
    """An absent block is fine (self-reported); a nonsense one is a problem."""
    block = document.get("produced_by")
    if not isinstance(block, dict):
        return []
    producer = block.get("producer")
    if producer not in PRODUCERS:
        return [f"produced_by.producer: {producer!r} is not one of "
                + ", ".join(PRODUCERS)]
    return []


def check_dataset_id(document: dict, expected_dataset_id) -> list[str]:
    """The file must name the dataset the consumer actually loaded gold for."""
    if expected_dataset_id is None:
        return []
    named = field_at(document, "dataset", "id")
    if not isinstance(named, str) or not named.strip():
        return []                         # optional: absent means the board's own dataset
    if named == expected_dataset_id:
        return []
    return [f"dataset.id: {named!r} is not the dataset this board scores "
            f"({expected_dataset_id})"]


def check_transport_error_count(document: dict) -> list[str]:
    """`run.transport_errors` must agree with the rows it is counting. A file that records nothing is
    not held to anything: `run.transport_errors` is not required, and a `null` is how a producer
    says it has none to report.
    """
    recorded = field_at(document, "run", "transport_errors")
    if recorded is None:
        return []
    if not isinstance(recorded, int) or isinstance(recorded, bool):
        return [f"run.transport_errors: {recorded!r} is not an integer — leave it out, or give "
                "the number of rows that carry inference_status transport_error"]
    counted = count_transport_errors(document.get("items") or [])
    if recorded == counted:
        return []
    return [f"run.transport_errors: {recorded} recorded, but {counted} rows carry "
            f"inference_status transport_error"]


def is_unfinished(document: dict) -> bool:
    """`run.complete` is present and not true: a checkpoint promoted by hand. No `run` block, or
    no `complete` key, is not a claim either way."""
    run = document.get("run")
    if not isinstance(run, dict) or "complete" not in run:
        return False
    return run["complete"] is not True


def is_legacy(document: dict) -> bool:
    """A file with no `dataset` block: the June–August results, written before the envelope."""
    return not isinstance(document.get("dataset"), dict)


def check_item_rows(document: dict) -> list[str]:
    """The shape of each row: an id, and a prediction that is a string or an explained null."""
    rows = document.get("items")
    if not isinstance(rows, list):
        return ["items: missing — a submission must carry one row per item in the snapshot"]
    if not rows:
        return ["items: empty — a submission must carry one row per item in the snapshot"]

    problems = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"items[{index}]: not an object")
            continue
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            problems.append(f"items[{index}]: no id — every row names the item it answers")
            continue
        problems.extend(check_prediction(item_id, row))
    return problems


def check_prediction(item_id: str, row: dict) -> list[str]:
    """One row's prediction. A string is an answer; a null is a row nobody could obtain."""
    if "prediction" not in row:
        return [f"items[{item_id}]: prediction is missing — give the model's raw output as a "
                "string, or null with inference_status \"transport_error\""]
    prediction = row["prediction"]
    if isinstance(prediction, str):
        return []
    if prediction is None:
        if row.get("inference_status") == "transport_error":
            return []
        return [f"items[{item_id}]: prediction is null but inference_status is "
                f"{row.get('inference_status')!r} — a row with no answer must say "
                "inference_status \"transport_error\""]
    return [f"items[{item_id}]: prediction is a {type(prediction).__name__}, not a string — send "
            "the model's raw output as text (JSON as a JSON string), unparsed"]


def row_ids(document: dict) -> list[str]:
    """The item ids the file holds, in file order."""
    rows = document.get("items")
    if not isinstance(rows, list):
        return []
    return [row["id"] for row in rows
            if isinstance(row, dict) and isinstance(row.get("id"), str)]


def check_slice_matches_its_fingerprint(document: dict) -> list[str]:
    """The `dataset` block describes a slice; the rows must still be that slice. A JSON `null` reads
    as absent rather than as a wrong type.
    """
    ids = row_ids(document)
    problems = []

    count = field_at(document, "dataset", "item_count")
    if count is not None:
        if not isinstance(count, int) or isinstance(count, bool):
            problems.append(f"dataset.item_count: {count!r} is not an integer — leave it out, or "
                            "give the number of rows the file holds")
        elif count != len(ids):
            problems.append(f"dataset.item_count: {count}, but the file holds {len(ids)} rows — "
                            "rows have been added or removed since this run finished")

    fingerprint = field_at(document, "dataset", "item_ids_sha256")
    if fingerprint is not None:
        if not isinstance(fingerprint, str) or not ITEM_IDS_SHA256.match(fingerprint):
            problems.append(f"dataset.item_ids_sha256: {fingerprint!r} is not a 64-character "
                            "sha256 hex digest — leave it out, or see docs/SUBMISSION.md for the "
                            "one-line recipe")
        elif fingerprint != item_ids_sha256(ids):
            problems.append("dataset.item_ids_sha256: does not match the ids in the file — the "
                            "rows are not the slice this run was made over, or they are in "
                            "another order")
    return problems


def check_item_ids(document: dict, gold_ids) -> list[str]:
    """The rows must answer exactly the snapshot: every id, each once, and nothing else."""
    if gold_ids is None:
        return []
    rows = document.get("items")
    if not isinstance(rows, list):
        return []                         # check_item_rows already said so

    seen, duplicates = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        item_id = row.get("id")
        if not isinstance(item_id, str):
            continue
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)

    wanted = set(gold_ids)
    problems = []
    if duplicates:
        problems.append(f"items: {len(duplicates)} duplicate id(s): {_named(duplicates)}")
    missing = wanted - seen
    if missing:
        problems.append(f"items: {len(missing)} id(s) in the snapshot have no row: "
                        f"{_named(missing)}")
    extra = seen - wanted
    if extra:
        problems.append(f"items: {len(extra)} id(s) that are not in the snapshot: "
                        f"{_named(extra)}")
    return problems


def contract_problems(document: dict, gold_ids=None, expected_dataset_id=None) -> list[str]:
    """Every way `document` fails the contract."""
    problems = []
    problems.extend(check_format_version(document))
    problems.extend(check_required_fields(document))
    problems.extend(check_inference_revision(document))
    problems.extend(check_dataset_id(document, expected_dataset_id))
    problems.extend(check_produced_by(document))
    problems.extend(check_item_rows(document))
    problems.extend(check_slice_matches_its_fingerprint(document))
    problems.extend(check_transport_error_count(document))
    problems.extend(check_item_ids(document, gold_ids))
    return problems


def validate_submission(document: dict, gold_ids=None) -> list[str]:
    """Every way `document` fails the submission contract, in one list."""
    return contract_problems(document, gold_ids)


# --- what a consumer may publish ----------------------------------------------------------------

def is_limit_file(path: Path) -> bool:
    """Whether `path` is a `--limit` smoke test, judged by the name `result_path` gives one."""
    return LIMIT_NAME.search(path.name) is not None


def transport_errors_in(document: dict) -> int:
    """How many rows a transport failure zeroed, counted from the rows."""
    return count_transport_errors(document.get("items") or [])


def publication_problems(document: dict, gold_ids=None, expected_dataset_id=None) -> list[str]:
    """The contract failures a consumer refuses a file for, whatever switches it was given."""
    if is_legacy(document):
        return check_item_ids(document, gold_ids)
    return contract_problems(document, gold_ids, expected_dataset_id)


def publication_status(document: dict, gold_ids=None, expected_dataset_id=None) -> str:
    """Whether this result file is fit to publish, and if not, what is wrong with it. invalid breaks
    the submission contract — see `publication_problems` and docs/SUBMISSION.md.
    """
    if publication_problems(document, gold_ids, expected_dataset_id):
        return "invalid"
    if is_legacy(document):
        return "legacy"
    if is_unfinished(document):
        return "incomplete"
    if transport_errors_in(document) > 0:
        return "transport_errors"
    return "ok"


def explain_status(document: dict, status: str, gold_ids=None, expected_dataset_id=None) -> str:
    """Why `publication_status` returned something other than a pass."""
    if status == "incomplete":
        return "run.complete is not true — this run never finished every item"
    if status == "transport_errors":
        return (f"run.transport_errors is {transport_errors_in(document)} — "
                "those rows failed and are scored zero")
    if status == "invalid":
        problems = publication_problems(document, gold_ids, expected_dataset_id)
        listed = "\n".join(f"      - {problem}" for problem in problems)
        return f"fails the submission contract (docs/SUBMISSION.md):\n{listed}"
    return status


# What every file on one board must agree about. A board is one question asked of several models,
# so two files made over different snapshots of a dataset that is still accepting rows are not two
# answers to one question, however valid each is on its own.
SNAPSHOT_KEYS = ("id", "inference_revision", "item_ids_sha256", "item_count", "config", "split")


def describe_snapshot(document: dict) -> str:
    """One line naming the slice a file was made over, for a refusal message."""
    dataset = field_at(document, "dataset", "id")
    revision = field_at(document, "dataset", "inference_revision")
    count = field_at(document, "dataset", "item_count")
    fingerprint = field_at(document, "dataset", "item_ids_sha256")
    short = revision[:12] if isinstance(revision, str) else revision
    ids = f" · ids {fingerprint[:12]}" if isinstance(fingerprint, str) else ""
    return f"{dataset} @ {short} · {count} items{ids}"


def snapshot_fields_that_differ(published) -> list[str]:
    """Which of SNAPSHOT_KEYS the given (path, document) pairs disagree about."""
    differing = []
    for key in SNAPSHOT_KEYS:
        values = set()
        for _path, document in published:
            value = field_at(document, "dataset", key)
            if value is not None:
                values.add(value)
        if len(values) > 1:
            differing.append(f"dataset.{key}")
    return differing


def files_by_label(published) -> dict:
    """{model.label: [file name, ...]} across the files a build is about to publish."""
    grouped = {}
    for path, document in published:
        label = field_at(document, "model", "label")
        if not isinstance(label, str) or not label.strip():
            continue                      # check_required_fields already said so
        grouped.setdefault(label, []).append(path.name)
    return grouped


def refuse_duplicate_labels(published) -> None:
    """Stop the build when two files claim the same `model.label`."""
    collisions = []
    for label, names in sorted(files_by_label(published).items()):
        if len(names) > 1:
            collisions.append(f"  {len(names)} files claim model.label {label!r}: "
                              + ", ".join(sorted(names)))
    if not collisions:
        return
    raise SystemExit(
        "refusing to publish: one label is one row on the board, and these files share one:\n"
        + "\n".join(collisions)
        + "\nRename the label of the one you are keeping, or move the other aside.")


def warn_about_filename_label_mismatches(published) -> None:
    """Say when a file is not named after the label it carries."""
    for path, document in published:
        label = field_at(document, "model", "label")
        if not isinstance(label, str) or not label.strip():
            continue
        if path.name != f"{label}.json":
            print(f"warning: {path.name} carries model.label {label!r}, so it boards under that "
                  f"name rather than its own (the convention is {label}.json)")


def refuse_mixed_snapshots(published, results_dir) -> None:
    """Stop the build when the files on this board answer different snapshots."""
    legacy = [path for path, document in published if is_legacy(document)]
    for path in legacy:
        print(f"warning: legacy file {path.name}: snapshot unknown, not checked against the others")

    comparable = [(path, document) for path, document in published
                  if not is_legacy(document)]
    differing = snapshot_fields_that_differ(comparable)
    if not differing:
        return

    lines = "\n".join(f"  {path.name}: {describe_snapshot(document)}"
                      for path, document in comparable)
    raise SystemExit(
        f"refusing to publish {len(comparable)} result file(s) in {results_dir}: they were not "
        f"made over the same dataset snapshot ({', '.join(differing)} differ).\n{lines}\n"
        "A board is one question asked of every model. Rerun the odd ones out against the "
        "snapshot the rest used, or move them aside.")


def discover_result_files(results_dir: Path, allow_incomplete: bool = False,
                          gold_ids=None, expected_dataset_id=None) -> list[Path]:
    """Every result file in `results_dir` that a consumer may aggregate, in name order."""
    publishable, refused = [], []
    for path in sorted(Path(results_dir).glob("*.json")):
        if is_limit_file(path):
            print(f"skipping smoke-test file {path.name}")
            continue
        document = load_result_file(path)
        status = publication_status(document, gold_ids, expected_dataset_id)
        if status in ("ok", "legacy"):
            publishable.append((path, document))
        elif allow_incomplete and status != "invalid":
            print(f"warning: publishing {path.name} anyway: "
                  f"{explain_status(document, status, gold_ids, expected_dataset_id)}")
            publishable.append((path, document))
        else:
            refused.append((path,
                            explain_status(document, status, gold_ids, expected_dataset_id)))

    if refused:
        lines = "\n".join(f"  {path.name}: {reason}" for path, reason in refused)
        raise SystemExit(
            f"refusing to publish {len(refused)} result file(s) in {results_dir}:\n{lines}\n"
            "Finish them with `harness.py --resume`, move them aside, or pass --allow-incomplete "
            "to publish them as they are. A file that fails the submission contract is refused "
            "either way: fix it, checking with `uv run glam_bench/validate_submission.py <file>`.")

    # Every file is fit to publish on its own. What is left is about the SET: are they all answers
    # to the same question, and does each one claim a row nobody else claims?
    refuse_duplicate_labels(publishable)
    refuse_mixed_snapshots(publishable, results_dir)
    warn_about_filename_label_mismatches(publishable)
    return [path for path, _document in publishable]
