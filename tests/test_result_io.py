"""Unit tests for the result file layer: naming, atomic writes, and the resume contract.

The end-to-end behaviour (checkpoint, promote, --resume, --select) is covered in
test_harness.py; this file pins the pieces those rules are built from.
"""
import json
import os
import pathlib

import pytest

import result_io

MODEL = {"label": "M", "kind": "fake", "id": "fake/model"}
DATASET = {"id": "org/set", "inference_revision": "d" * 40, "item_ids_sha256": "abc",
           "item_count": 2}
GENERATION = {"max_tokens": 900, "temperature": 0, "guided": False, "request_options": {}}

# The three blocks RESUME_IDENTITY names, as `validate_resume_target` takes them.
BLOCKS = {"model": MODEL, "dataset": DATASET, "generation": GENERATION}


def existing_file(**overrides):
    """A minimal result file that a run over MODEL/DATASET is allowed to resume."""
    document = {"model": dict(MODEL), "dataset": dict(DATASET),
                "generation": dict(GENERATION),
                "scoring": {"scorer_version": "2026-08-26a"},
                "run": {"started_at": "2026-08-27T12:00:00Z", "complete": False},
                "items": []}
    document.update(overrides)
    return document


# --- file naming --------------------------------------------------------------------------------

def test_a_full_run_uses_the_canonical_name(tmp_path):
    assert result_io.result_path(tmp_path, "GLM-4.6V-Flash") == tmp_path / "GLM-4.6V-Flash.json"


def test_a_limited_run_gets_its_own_name(tmp_path):
    path = result_io.result_path(tmp_path, "GLM-4.6V-Flash", limit=2)
    assert path == tmp_path / "GLM-4.6V-Flash.limit2.json"
    assert path != result_io.result_path(tmp_path, "GLM-4.6V-Flash")


def test_the_partial_sits_beside_the_final_file_and_is_not_a_json_file(tmp_path):
    final = result_io.result_path(tmp_path, "M")
    partial = result_io.partial_path(final)
    assert partial == tmp_path / "M.json.partial"
    partial.write_text("{}")
    # every consumer globs *.json; the checkpoint must not match
    assert list(tmp_path.glob("*.json")) == []


# --- duplicate ids ------------------------------------------------------------------------------

def test_duplicate_ids_names_each_repeated_id_once():
    assert result_io.duplicate_ids(["a", "b", "c"]) == []
    assert result_io.duplicate_ids(["a", "b", "a", "b", "a"]) == ["a", "b"]


def test_a_slice_with_no_repeats_passes_silently():
    assert result_io.refuse_duplicate_ids(["a", "b"], "org/set") is None


def test_a_repeated_id_stops_the_run_and_says_which_one():
    with pytest.raises(SystemExit) as refusal:
        result_io.refuse_duplicate_ids(["a", "b", "a"], "org/set split train")
    message = str(refusal.value)
    assert "org/set split train" in message
    assert "1 duplicate item id(s): a" in message


# --- atomic write -------------------------------------------------------------------------------

def test_atomic_write_uses_a_temp_file_in_the_target_directory(tmp_path, monkeypatch):
    renames = []
    real_replace = os.replace

    def spy_replace(src, dst):
        renames.append((pathlib.Path(src), pathlib.Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(result_io.os, "replace", spy_replace)
    target = tmp_path / "M.json"
    result_io.atomic_write_json(target, {"items": [1, 2]})

    source, destination = renames[0]
    # same directory -> same filesystem -> the rename is atomic
    assert source.parent == target.parent
    assert destination == target
    assert json.loads(target.read_text()) == {"items": [1, 2]}
    assert [p.name for p in tmp_path.iterdir()] == ["M.json"]   # no temp file left behind


def test_a_failed_atomic_write_leaves_neither_a_temp_file_nor_a_target(tmp_path, monkeypatch):
    def failing_replace(_src, _dst):
        raise OSError("disk full")

    monkeypatch.setattr(result_io.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        result_io.atomic_write_json(tmp_path / "M.json", {"items": []})
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_replaces_an_existing_file_whole(tmp_path):
    target = tmp_path / "M.json"
    result_io.atomic_write_json(target, {"items": [1]})
    result_io.atomic_write_json(target, {"items": [1, 2]})
    assert json.loads(target.read_text()) == {"items": [1, 2]}
    assert [p.name for p in tmp_path.iterdir()] == ["M.json"]


# --- choosing what to resume ----------------------------------------------------------------------

def test_the_partial_wins_when_both_files_exist(tmp_path):
    final = tmp_path / "M.json"
    partial = tmp_path / "M.json.partial"
    final.write_text("{}")
    partial.write_text("{}")
    # promotion deletes the partial, so a partial next to a final one is the later, interrupted run
    assert result_io.find_resume_source(final, partial) == partial


def test_the_final_file_is_resumed_when_there_is_no_partial(tmp_path):
    final = tmp_path / "M.json"
    final.write_text("{}")
    assert result_io.find_resume_source(final, tmp_path / "M.json.partial") == final


def test_nothing_to_resume_is_not_an_error(tmp_path):
    assert result_io.find_resume_source(tmp_path / "M.json", tmp_path / "M.json.partial") is None


# --- the resume contract --------------------------------------------------------------------------

def test_an_identical_run_may_be_resumed(tmp_path):
    assert result_io.validate_resume_target(existing_file(), BLOCKS,
                                            tmp_path / "M.json") is None


@pytest.mark.parametrize("block,key,value", [
    ("dataset", "id", "org/other"),
    ("dataset", "inference_revision", "e" * 40),
    ("dataset", "item_ids_sha256", "def"),
    ("model", "id", "fake/other"),
    ("model", "kind", "router_vlm"),
    ("generation", "max_tokens", 1600),
    ("generation", "temperature", 0.7),
    ("generation", "guided", True),
    ("generation", "request_options", {"chat_template_kwargs": {"enable_thinking": False}}),
])
def test_every_identity_field_stops_a_resume(tmp_path, block, key, value):
    document = existing_file()
    document[block][key] = value
    with pytest.raises(SystemExit) as excinfo:
        result_io.validate_resume_target(document, BLOCKS, tmp_path / "M.json")
    message = str(excinfo.value)
    assert f"{block}.{key}" in message
    assert repr(value) in message                 # says what the file has
    assert "M.json" in message                    # and which file to move or delete


def test_a_resume_refusal_lists_every_mismatch_at_once(tmp_path):
    document = existing_file()
    document["dataset"]["inference_revision"] = "e" * 40
    document["model"]["id"] = "fake/other"
    with pytest.raises(SystemExit) as excinfo:
        result_io.validate_resume_target(document, BLOCKS, tmp_path / "M.json")
    assert "dataset.inference_revision" in str(excinfo.value)
    assert "model.id" in str(excinfo.value)


def test_a_file_without_items_is_not_a_resume_target(tmp_path):
    with pytest.raises(SystemExit, match="not a result file"):
        result_io.validate_resume_target({"model": MODEL, "dataset": DATASET}, BLOCKS,
                                         tmp_path / "M.json")


def test_a_missing_block_reads_as_a_mismatch_rather_than_crashing(tmp_path):
    document = existing_file()
    del document["dataset"]
    with pytest.raises(SystemExit, match="dataset.id"):
        result_io.validate_resume_target(document, BLOCKS, tmp_path / "M.json")


def test_a_drifted_scorer_version_warns_instead_of_refusing():
    document = existing_file()
    warning = result_io.scorer_version_warning(document, "2026-09-01b")
    assert warning is not None
    assert "2026-08-26a" in warning and "2026-09-01b" in warning
    assert result_io.scorer_version_warning(document, "2026-08-26a") is None


def test_unreadable_json_stops_the_run_with_the_path(tmp_path):
    broken = tmp_path / "M.json"
    broken.write_text("{ not json")
    with pytest.raises(SystemExit, match="M.json"):
        result_io.load_result_file(broken)


# --- selecting rows to redo -------------------------------------------------------------------------

OK_ROW = {"id": "card-1", "prediction": "{}", "inference_status": "ok",
          "schema_valid": 1.0, "attempts": 1}
UNPARSEABLE_ROW = {"id": "card-2", "prediction": "sorry, I cannot read this card",
                   "inference_status": "ok", "schema_valid": 0.0, "attempts": 1}
TRANSPORT_ROW = {"id": "card-3", "prediction": None,
                 "inference_status": "transport_error", "attempts": 3}
SCORER_CRASH_ROW = {"id": "card-4", "prediction": "{}", "inference_status": "ok",
                    "schema_valid": 0.0,
                    "scorer_error": "ZeroDivisionError: scorer bug", "attempts": 1}


@pytest.mark.parametrize("select,expected", [
    ("errors", [False, False, True, True]),
    ("unparseable", [False, True, True, True]),
    ("all", [True, True, True, True]),
])
def test_selection_modes(select, expected):
    rows = [OK_ROW, UNPARSEABLE_ROW, TRANSPORT_ROW, SCORER_CRASH_ROW]
    assert [result_io.needs_redo(row, select) for row in rows] == expected


def test_a_row_with_an_unknown_status_is_always_redone():
    assert result_io.needs_redo({"id": "x", "inference_status": "who knows"}, "errors") is True


def test_previous_attempts_counts_a_missing_row_as_none():
    assert result_io.previous_attempts(None) == 0
    assert result_io.previous_attempts(TRANSPORT_ROW) == 3
    # a row from an older format that never recorded attempts still counts as one attempt made
    assert result_io.previous_attempts({"id": "x"}) == 1


def test_rows_are_indexed_by_id_and_junk_is_dropped():
    indexed = result_io.rows_by_id({"items": [OK_ROW, {"no": "id"}, "not a row"]})
    assert indexed == {"card-1": OK_ROW}


def test_transport_errors_are_counted_for_the_run_block():
    assert result_io.count_transport_errors([OK_ROW, TRANSPORT_ROW, UNPARSEABLE_ROW]) == 1


# --- the envelope ---------------------------------------------------------------------------------

def test_the_envelope_key_order_is_fixed():
    envelope = result_io.build_envelope(2, MODEL, DATASET, GENERATION, {"scorer_version": "v"},
                                        {"complete": True}, result_io.HARNESS_PRODUCED_BY,
                                        [OK_ROW])
    assert list(envelope) == ["format_version", "model", "dataset", "generation", "scoring",
                              "run", "produced_by", "items"]
    # a copy: two runs in one process must not share (or edit) one produced_by dict
    assert envelope["produced_by"] == result_io.HARNESS_PRODUCED_BY
    assert envelope["produced_by"] is not result_io.HARNESS_PRODUCED_BY


def test_resumed_at_is_absent_until_a_run_is_actually_resumed():
    fresh = result_io.build_run_block("t0", "t1", "sha", True, [OK_ROW], [])
    assert "resumed_at" not in fresh
    resumed = result_io.build_run_block("t0", "t1", "sha", True, [OK_ROW], ["t2"])
    assert resumed["resumed_at"] == ["t2"]


# --- discovery and the publication gate -----------------------------------------------------------

def dataset_for(rows):
    """A dataset block that actually describes `rows`: the same count, the same id fingerprint."""
    ids = [row["id"] for row in rows]
    return {**DATASET, "item_count": len(ids), "item_ids_sha256": result_io.item_ids_sha256(ids)}


def failed_file(**run_overrides):
    """A finished file that carries a real transport failure and records it honestly.

    The recorded count and the rows have to agree: a run block claiming a failure the rows do not
    show is a contract problem in its own right (test_a_recorded_failure_count_that_the_rows_do_
    not_support_is_refused), which would refuse the file before the gate ever weighed the failure.
    """
    rows_in = [OK_ROW, TRANSPORT_ROW]
    run = {"started_at": "t0", "finished_at": "t1", "complete": True, "transport_errors": 1}
    run.update(run_overrides)
    return {"format_version": 2, "model": dict(MODEL), "dataset": dataset_for(rows_in),
            "scoring": {"scorer_version": "v"}, "run": run, "items": rows_in}


def published_file(**run_overrides):
    """A finished result file, of the shape a consumer is meant to aggregate."""
    run = {"started_at": "t0", "finished_at": "t1", "complete": True, "transport_errors": 0}
    run.update(run_overrides)
    return {"format_version": 2, "model": dict(MODEL), "dataset": dataset_for([OK_ROW]),
            "scoring": {"scorer_version": "v"}, "run": run, "items": [OK_ROW]}


def write_result(directory, name, document):
    path = directory / name
    path.write_text(json.dumps(document))
    return path


def test_a_limit_file_is_recognised_from_the_name_result_path_gives_it(tmp_path):
    """The writer and the reader must agree, or a smoke test lands on the board."""
    assert result_io.is_limit_file(result_io.result_path(tmp_path, "GLM-OCR", limit=2))
    assert not result_io.is_limit_file(result_io.result_path(tmp_path, "GLM-OCR"))
    assert not result_io.is_limit_file(tmp_path / "no-limit-here.json")


def test_discovery_skips_smoke_tests_and_checkpoints(tmp_path, capsys):
    write_result(tmp_path, "M.json", published_file())
    write_result(tmp_path, "M.limit2.json", published_file())
    (tmp_path / "M.json.partial").write_text(json.dumps(published_file(complete=False)))

    found = result_io.discover_result_files(tmp_path)

    assert [p.name for p in found] == ["M.json"]
    assert "skipping smoke-test file M.limit2.json" in capsys.readouterr().out


def test_an_old_format_file_with_no_run_block_still_publishes(tmp_path):
    write_result(tmp_path, "old.json", {"model": dict(MODEL), "items": [OK_ROW]})
    assert result_io.publication_status({"model": dict(MODEL), "items": [OK_ROW]}) == "legacy"
    assert [p.name for p in result_io.discover_result_files(tmp_path)] == ["old.json"]


def test_an_unfinished_run_is_refused_by_name(tmp_path):
    write_result(tmp_path, "good.json", published_file())
    write_result(tmp_path, "unfinished.json", published_file(complete=False))

    with pytest.raises(SystemExit) as refusal:
        result_io.discover_result_files(tmp_path)

    message = str(refusal.value)
    assert "unfinished.json" in message and "run.complete is not true" in message
    assert "good.json" not in message          # only the files at fault are named
    assert "--allow-incomplete" in message     # and the message says how to override


def test_transport_errors_are_refused_whether_recorded_or_recounted(tmp_path):
    recorded = failed_file()
    assert result_io.publication_status(recorded) == "transport_errors"

    recounted = failed_file()
    del recounted["run"]["transport_errors"]   # written before the run block counted them
    assert result_io.publication_status(recounted) == "transport_errors"

    write_result(tmp_path, "failed.json", recorded)
    with pytest.raises(SystemExit) as refusal:
        result_io.discover_result_files(tmp_path)
    assert "failed.json: run.transport_errors is 1" in str(refusal.value)


def test_a_recorded_failure_count_that_the_rows_do_not_support_is_refused(tmp_path):
    """The number decides whether the file is published, so it is checked against the rows.

    A file recording 0 while carrying failures would otherwise reach the board with those zeroed
    rows dragging its scores down and nothing on the page saying so.
    """
    lying = failed_file(transport_errors=0)
    assert result_io.validate_submission(lying) == [
        "run.transport_errors: 0 recorded, but 1 rows carry inference_status transport_error"]
    assert result_io.publication_status(lying) == "invalid"

    write_result(tmp_path, "liar.json", lying)
    with pytest.raises(SystemExit) as refusal:
        # not even with the override: this is a contract failure, not an unfinished run
        result_io.discover_result_files(tmp_path, allow_incomplete=True)
    assert "0 recorded, but 1 rows carry" in str(refusal.value)


@pytest.mark.parametrize("value", ["0", 1.0, True, [], {"n": 0}])
def test_a_failure_count_of_the_wrong_type_is_a_problem(value):
    """Same principle as the slice fields: a wrong-typed value must not be the way to switch off
    the check that belongs to it."""
    document = published_file(transport_errors=value)
    problems = [p for p in result_io.validate_submission(document)
                if p.startswith("run.transport_errors")]
    assert len(problems) == 1
    assert f"run.transport_errors: {value!r} is not an integer" in problems[0]


def test_a_failure_count_written_out_null_reads_as_one_that_was_left_out():
    document = published_file(transport_errors=None)
    assert result_io.validate_submission(document) == []


def test_the_gate_acts_on_the_recount_not_the_claim():
    """The other direction: a file over-recording is caught by the same comparison."""
    overstated = published_file(transport_errors=3)
    assert result_io.transport_errors_in(overstated) == 0
    assert result_io.validate_submission(overstated) == [
        "run.transport_errors: 3 recorded, but 0 rows carry inference_status transport_error"]


def test_allow_incomplete_publishes_them_and_says_so(tmp_path, capsys):
    # both over the same slice, or the board refuses the pair before it weighs either of them —
    # and under different labels, since one label is one row
    unfinished = failed_file(complete=False, transport_errors=0)
    unfinished["items"] = [OK_ROW, {**TRANSPORT_ROW, "prediction": "{}", "inference_status": "ok"}]
    unfinished["model"] = {**MODEL, "label": "unfinished"}
    write_result(tmp_path, "failed.json", {**failed_file(), "model": {**MODEL, "label": "failed"}})
    write_result(tmp_path, "unfinished.json", unfinished)

    found = result_io.discover_result_files(tmp_path, allow_incomplete=True)

    assert [p.name for p in found] == ["failed.json", "unfinished.json"]
    warnings = capsys.readouterr().out
    assert "warning: publishing failed.json anyway: run.transport_errors is 1" in warnings
    assert "warning: publishing unfinished.json anyway: run.complete is not true" in warnings
