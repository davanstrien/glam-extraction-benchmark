"""The submission contract: what a result file must be, whoever produced it.

A file in results/ is a submission and the harness is one producer of them. These tests pin the
checks that make that safe — the shape rules, the id check against a gold snapshot, the gate that
keeps a bad submission off the board, and the attested/self-reported tier the board shows because
no check can tell you which model actually ran. Contract in prose: docs/SUBMISSION.md.
"""
import json

import pytest

import board
import result_io
import validate_submission as cli
from nls import NLS_DATASET

GOLD_IDS = {"card-1", "card-2"}
SHA = "d" * 40


def submission(**overrides):
    """A submission that passes every check, for a two-item snapshot."""
    document = {
        "format_version": 2,
        "model": {"label": "my-extractor", "id": "acme/my-extractor", "params": "7B"},
        "dataset": {"id": NLS_DATASET, "inference_revision": SHA},
        "run": {"complete": True},
        "produced_by": {"producer": "other", "attested": False},
        "items": [{"id": "card-1", "prediction": "{}"},
                  {"id": "card-2", "prediction": "{}"}],
    }
    document.update(overrides)
    return document


def rows(*pairs):
    return [{"id": item_id, "prediction": prediction} for item_id, prediction in pairs]


def working(**overrides) -> dict:
    """A provenance block that actually says where an answer came from, and when."""
    block = {"served_by": "openai-compatible", "endpoint": "https://abc--8000.hf.jobs/v1",
             "timestamp": "2026-08-27T12:00:00Z"}
    block.update(overrides)
    return block


def attested_rows(*, provenance=None):
    """The two-item slice, every row carrying working (or the provenance given)."""
    return [{"id": item_id, "prediction": "{}",
             "provenance": working() if provenance is None else provenance}
            for item_id in ("card-1", "card-2")]


# --- a good submission --------------------------------------------------------------------------

def test_a_complete_submission_has_no_problems():
    assert result_io.validate_submission(submission(), GOLD_IDS) == []


def test_without_a_snapshot_the_id_check_is_skipped_and_the_rest_still_runs():
    """A consumer with no gold loaded can still be told a file is the wrong shape."""
    off_snapshot = submission(items=rows(("who-is-this", "{}")))
    assert result_io.validate_submission(off_snapshot, gold_ids=None) == []

    broken = submission(items=rows(("who-is-this", 42)))
    assert result_io.validate_submission(broken, gold_ids=None) == [
        "items[who-is-this]: prediction is a int, not a string — send the model's raw output as "
        "text (JSON as a JSON string), unparsed"]


# --- the ids must answer exactly the snapshot ----------------------------------------------------

def test_a_missing_id_is_named():
    problems = result_io.validate_submission(submission(items=rows(("card-1", "{}"))), GOLD_IDS)
    assert problems == ["items: 1 id(s) in the snapshot have no row: card-2"]


def test_an_extra_id_is_named():
    document = submission(items=rows(("card-1", "{}"), ("card-2", "{}"), ("card-99", "{}")))
    problems = result_io.validate_submission(document, GOLD_IDS)
    assert problems == ["items: 1 id(s) that are not in the snapshot: card-99"]


def test_a_duplicate_id_is_a_problem_even_though_every_id_is_present():
    document = submission(items=rows(("card-1", "{}"), ("card-1", "{}"), ("card-2", "{}")))
    problems = result_io.validate_submission(document, GOLD_IDS)
    assert problems == ["items: 1 duplicate id(s): card-1"]


def test_a_submission_against_another_snapshot_fails_on_the_ids_it_answers():
    """The revision and the ids are one claim: these rows are not that snapshot's rows."""
    problems = result_io.validate_submission(submission(), {"other-1", "other-2"})
    assert problems == [
        "items: 2 id(s) in the snapshot have no row: other-1, other-2",
        "items: 2 id(s) that are not in the snapshot: card-1, card-2"]


def test_long_id_lists_are_truncated_with_a_count():
    wanted = {f"card-{n}" for n in range(9)}
    problems = result_io.validate_submission(submission(), wanted)
    assert "… and 2 more" in problems[0]


# --- predictions ---------------------------------------------------------------------------------

def test_a_parsed_prediction_is_refused():
    """The scorer judges the model's own output, including output that is not valid JSON."""
    document = submission(items=[{"id": "card-1", "prediction": {"already": "parsed"}},
                                 {"id": "card-2", "prediction": "{}"}])
    assert result_io.validate_submission(document, GOLD_IDS) == [
        "items[card-1]: prediction is a dict, not a string — send the model's raw output as text "
        "(JSON as a JSON string), unparsed"]


def test_a_null_prediction_must_say_why():
    document = submission(items=[{"id": "card-1", "prediction": None},
                                 {"id": "card-2", "prediction": "{}"}])
    assert result_io.validate_submission(document, GOLD_IDS) == [
        "items[card-1]: prediction is null but inference_status is None — a row with no answer "
        "must say inference_status \"transport_error\""]


def test_a_null_prediction_with_a_transport_error_is_allowed():
    document = submission(items=[
        {"id": "card-1", "prediction": None, "inference_status": "transport_error"},
        {"id": "card-2", "prediction": "{}"}])
    assert result_io.validate_submission(document, GOLD_IDS) == []


def test_a_row_with_no_prediction_at_all_is_refused():
    document = submission(items=[{"id": "card-1"}, {"id": "card-2", "prediction": "{}"}])
    assert result_io.validate_submission(document, GOLD_IDS) == [
        "items[card-1]: prediction is missing — give the model's raw output as a string, or null "
        "with inference_status \"transport_error\""]


# --- the rest of the envelope ---------------------------------------------------------------------

def test_run_complete_is_not_a_contract_rule():
    """The rows answering the whole snapshot is what proves completeness. `run.complete: false`
    is the gate's business (a checkpoint promoted by hand), not the contract's."""
    assert result_io.validate_submission(submission(run={"complete": False}), GOLD_IDS) == []
    document = submission()
    del document["run"]
    assert result_io.validate_submission(document, GOLD_IDS) == []
    assert result_io.publication_status(document, GOLD_IDS) == "ok"


def test_a_branch_name_is_not_a_snapshot():
    document = submission(dataset={"id": "org/set", "inference_revision": "main"})
    problems = result_io.validate_submission(document, GOLD_IDS)
    assert problems == ["dataset.inference_revision: 'main' is not a 40-character commit sha — "
                        "resolve it once with HfApi().dataset_info(<dataset>).sha, or take the "
                        "value the harness prints"]


def test_the_two_required_strings_are_each_named_when_missing():
    document = submission(model={}, dataset={})
    problems = result_io.validate_submission(document, GOLD_IDS)
    assert problems == [
        "model.label: missing — the name this row gets on the board",
        "dataset.inference_revision: missing — the commit sha of the gold snapshot that was read"]


def test_a_missing_block_is_reported_once_as_its_required_key():
    document = submission()
    del document["dataset"]
    problems = result_io.validate_submission(document, GOLD_IDS)
    assert problems == ["dataset.inference_revision: missing — the commit sha of the gold "
                        "snapshot that was read"]


def test_model_id_and_dataset_id_are_optional_but_typed():
    minimal = submission(model={"label": "my-extractor"}, dataset={"inference_revision": SHA})
    assert result_io.validate_submission(minimal, GOLD_IDS) == []

    typed = submission(model={"label": "my-extractor", "id": 7},
                       dataset={"inference_revision": SHA, "id": ""})
    assert result_io.validate_submission(typed, GOLD_IDS) == [
        "model.id: 7 is not a string — leave it out, or give a Hub id, or free text for a model "
        "that is not on the Hub",
        "dataset.id: '' is not a string — leave it out, or give the Hub dataset the images and "
        "schema came from"]


def test_format_version_is_optional_but_must_be_current_when_given():
    absent = submission()
    del absent["format_version"]
    assert result_io.validate_submission(absent, GOLD_IDS) == []
    assert result_io.publication_status(absent, GOLD_IDS) == "ok"
    assert result_io.validate_submission(submission(format_version=1), GOLD_IDS) == [
        "format_version: 1 — this board reads format_version 2"]


def test_every_problem_is_reported_in_one_pass():
    """One run tells a submitter everything to fix, rather than one thing at a time."""
    document = submission(model={}, items=[{"id": "card-1", "prediction": 7}])
    problems = result_io.validate_submission(document, GOLD_IDS)
    assert len(problems) == 3
    assert any("model.label" in p for p in problems)
    assert any("prediction is a int" in p for p in problems)
    assert any("have no row: card-2" in p for p in problems)


# --- who made the file --------------------------------------------------------------------------

def test_a_submission_that_does_not_say_who_made_it_is_valid_and_self_reported():
    document = submission()
    del document["produced_by"]

    assert result_io.validate_submission(document, GOLD_IDS) == []
    assert result_io.produced_by(document) == {"producer": "other", "attested": False}
    assert result_io.is_attested(document) is False


def test_an_unrecognised_producer_is_a_problem():
    document = submission(produced_by={"producer": "vibes", "attested": False})
    assert result_io.validate_submission(document, GOLD_IDS) == [
        "produced_by.producer: 'vibes' is not one of harness, paratext, space, other"]


def test_attestation_is_only_honoured_when_every_row_shows_its_working():
    """A claim of attestation with no per-row provenance behind it reads as self-reported."""
    claimed = submission(produced_by={"producer": "harness", "attested": True})
    assert result_io.is_attested(claimed) is False

    backed = submission(produced_by=dict(result_io.HARNESS_PRODUCED_BY), items=attested_rows())
    assert result_io.is_attested(backed) is True

    half = submission(produced_by=dict(result_io.HARNESS_PRODUCED_BY),
                      items=[{"id": "card-1", "prediction": "{}", "provenance": working()},
                             {"id": "card-2", "prediction": "{}"}])
    assert result_io.is_attested(half) is False


@pytest.mark.parametrize("provenance, because", [
    ({}, "an empty block is the self-promotion this check exists to stop"),
    (working(served_by=None), "a key present and null records nothing"),
    (working(endpoint=""), "a key present and blank records nothing"),
    (working(timestamp="   "), "whitespace is not a timestamp"),
    ({"served_by": "x", "endpoint": "y"}, "no timestamp: when is half the claim"),
    ({"endpoint": "y", "timestamp": "t"}, "no served_by"),
    ({"served_by": "x", "timestamp": "t"}, "no endpoint: where is the other half"),
    (working(endpoint=42), "a number is not an endpoint"),
])
def test_a_provenance_block_that_says_nothing_is_not_working(provenance, because):
    """`provenance: {}` on every row used to earn the attested marker outright."""
    document = submission(produced_by=dict(result_io.HARNESS_PRODUCED_BY),
                          items=attested_rows(provenance=provenance))
    assert result_io.is_attested(document) is False, because


def test_extra_provenance_fields_do_not_stop_a_file_being_attested():
    """The three are a floor, not a schema — the harness writes ten more."""
    document = submission(produced_by=dict(result_io.HARNESS_PRODUCED_BY),
                          items=attested_rows(provenance=working(latency_s=1.2, finish_reason=None)))
    assert result_io.is_attested(document) is True


@pytest.mark.parametrize("producer", ["other", "paratext", "space"])
def test_only_the_reference_producer_boards_as_attested(producer):
    """The tier means "the harness ran this". A third-party file that wrote three plausible
    strings onto every row would otherwise award itself a marker that reads as a verification
    the project performed — and no file can prove what ran, which is why the tier exists."""
    document = submission(produced_by={"producer": producer, "attested": True},
                          items=attested_rows())
    assert result_io.is_attested(document) is False
    # and it is a perfectly valid submission, boarded and rescored like any other
    assert result_io.validate_submission(document, GOLD_IDS) == []


def test_the_allow_list_is_the_one_place_a_producer_is_admitted(monkeypatch):
    """Adding paratext later is a one-line, deliberate change, not a rewrite of the rule."""
    document = submission(produced_by={"producer": "paratext", "attested": True},
                          items=attested_rows())
    monkeypatch.setattr(result_io, "ATTESTING_PRODUCERS", frozenset({"harness", "paratext"}))
    assert result_io.is_attested(document) is True


def test_a_producer_that_does_not_claim_attestation_is_never_attested():
    with_provenance = submission(
        produced_by={"producer": "paratext", "contact": "someone@example.org"},
        items=attested_rows())
    assert result_io.is_attested(with_provenance) is False


def test_a_legacy_file_is_not_attested():
    assert result_io.is_attested({"model": {"label": "old"}, "items": [{"id": "card-1"}]}) is False


# --- the publication gate --------------------------------------------------------------------------

def write(directory, name, document):
    (directory / name).write_text(json.dumps(document))


def test_an_invalid_submission_is_refused_and_every_problem_is_listed(tmp_path):
    write(tmp_path, "broken.json", submission(items=[{"id": "card-1", "prediction": None}]))

    with pytest.raises(SystemExit) as refusal:
        result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS)

    message = str(refusal.value)
    assert "broken.json: fails the submission contract" in message
    assert "prediction is null but inference_status is None" in message
    assert "have no row: card-2" in message
    assert "validate_submission.py" in message


def test_allow_incomplete_does_not_wave_through_a_contract_failure(tmp_path):
    """An unfinished run can still be read honestly; rows that answer another snapshot cannot."""
    write(tmp_path, "broken.json", submission(items=rows(("card-1", "{}"))))

    with pytest.raises(SystemExit) as refusal:
        result_io.discover_result_files(tmp_path, allow_incomplete=True, gold_ids=GOLD_IDS)
    assert "broken.json" in str(refusal.value)


def test_an_unfinished_file_is_still_held_to_the_contract(tmp_path):
    """The fail-open path: status used to be decided before the contract was ever checked, so a
    `complete: false` file went through `--allow-incomplete` without one check running on it."""
    duplicated = submission(run={"complete": False},
                            items=rows(("card-1", "{}"), ("card-2", "{}"), ("card-1", "{}")))
    write(tmp_path, "broken.json", duplicated)

    assert result_io.publication_status(duplicated, GOLD_IDS) == "invalid"
    with pytest.raises(SystemExit) as refusal:
        result_io.discover_result_files(tmp_path, allow_incomplete=True, gold_ids=GOLD_IDS)

    message = str(refusal.value)
    assert "items: 1 duplicate id(s): card-1" in message
    assert "--allow-incomplete" in message      # says the switch it just declined to honour


def test_completeness_is_the_one_failure_allow_incomplete_still_waives(tmp_path):
    """A file whose only problem is that it never finished stays overridable."""
    unfinished = submission(run={"complete": False})
    write(tmp_path, "unfinished.json", unfinished)

    assert result_io.validate_submission(unfinished, GOLD_IDS) == []
    assert result_io.publication_status(unfinished, GOLD_IDS) == "incomplete"

    found = result_io.discover_result_files(tmp_path, allow_incomplete=True, gold_ids=GOLD_IDS)
    assert [p.name for p in found] == ["unfinished.json"]


def test_a_valid_submission_from_any_producer_is_published(tmp_path):
    write(tmp_path, "my-extractor.json", submission())
    found = result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS)
    assert [p.name for p in found] == ["my-extractor.json"]


def test_a_pre_envelope_file_is_legacy_and_still_passes(tmp_path):
    """The June–August files predate all of this; the gate is on data, not on file age.

    It is on data, though: the rows still have to answer the whole snapshot.
    """
    old = {"model": {"label": "old", "params": "9B"},
           "items": [{"id": "card-1", "content_f1": 0.7}, {"id": "card-2", "content_f1": 0.6}]}
    write(tmp_path, "old.json", old)

    assert result_io.publication_status(old, GOLD_IDS) == "legacy"
    assert [p.name for p in result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS)] \
        == ["old.json"]


@pytest.mark.parametrize("items, expected", [
    ([{"id": "card-1"}], "items: 1 id(s) in the snapshot have no row: card-2"),
    ([{"id": "card-1"}, {"id": "card-2"}, {"id": "card-9"}],
     "items: 1 id(s) that are not in the snapshot: card-9"),
    ([{"id": "card-1"}, {"id": "card-2"}, {"id": "card-1"}],
     "items: 1 duplicate id(s): card-1"),
])
def test_a_legacy_file_gets_the_id_check_like_everything_else(tmp_path, items, expected):
    """A legacy file (no `dataset` block) used to bypass every check, which made it the way to
    put any set of rows at all onto the board — a half-finished slice scored as whole."""
    old = {"model": {"label": "old", "params": "9B"}, "items": items}
    write(tmp_path, "old.json", old)

    assert result_io.publication_status(old, GOLD_IDS) == "invalid"
    with pytest.raises(SystemExit) as refusal:
        result_io.discover_result_files(tmp_path, allow_incomplete=True, gold_ids=GOLD_IDS)
    assert expected in str(refusal.value)


def test_a_legacy_file_is_still_held_to_nothing_else():
    """The ids and the dataset it names, and no more: a legacy file has no envelope to hold to
    a shape, so a wrong-typed prediction and an unfinished run block go unremarked."""
    old = {"model": {"label": "old"}, "run": {"complete": False},
           "items": [{"id": "card-1", "prediction": 42}, {"id": "card-2"}]}
    assert result_io.publication_problems(old, GOLD_IDS) == []
    assert result_io.publication_status(old, GOLD_IDS) == "legacy"


def test_a_file_with_a_dataset_block_is_not_legacy_and_is_held_to_what_it_named(tmp_path):
    """Legacy means no `dataset` block. A file that carries one gets the whole contract, so
    naming another dataset is refused whatever else it lacks."""
    old = {"model": {"label": "old"},
           "dataset": {"id": "acme/other-cards"},
           "items": [{"id": "card-1"}, {"id": "card-2"}]}
    write(tmp_path, "old.json", old)
    assert result_io.is_legacy(old) is False

    with pytest.raises(SystemExit) as refusal:
        result_io.discover_result_files(tmp_path, allow_incomplete=True, gold_ids=GOLD_IDS,
                                        expected_dataset_id=NLS_DATASET)

    assert (f"dataset.id: 'acme/other-cards' is not the dataset this board scores "
            f"({NLS_DATASET})") in str(refusal.value)


def test_a_legacy_file_with_no_dataset_block_is_not_held_to_one(tmp_path):
    """The June–August files were exactly this shape: rows, a label, and nothing else."""
    old = {"model": {"label": "old"}, "items": [{"id": "card-1"}, {"id": "card-2"}]}
    write(tmp_path, "old.json", old)

    found = result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS,
                                            expected_dataset_id=NLS_DATASET)
    assert [p.name for p in found] == ["old.json"]


def test_an_unfinished_format_2_file_still_reports_as_unfinished():
    """A file that is only unfinished says so, rather than being reported as a contract failure
    — which would be true and much less useful. Everything else the contract checks is refused
    first, and refused whatever switches the build was given."""
    unfinished = submission(run={"complete": False})
    assert result_io.publication_status(unfinished, GOLD_IDS) == "incomplete"
    assert "run.complete is not true" in result_io.explain_status(unfinished, "incomplete")


# --- the board tier -------------------------------------------------------------------------------

def fake_gold():
    schema = json.dumps({"type": "object", "properties": {"heading": {"type": "string"}}})
    return (schema,
            {"card-1": {"gold": {"heading": "Allan, T."}, "label_status": "verified"},
             "card-2": {"gold": {"heading": "Brown, R."}, "label_status": "corrected"}},
            SHA)


def board_over(monkeypatch, tmp_path, document, name="M.json"):
    write(tmp_path, name, document)
    monkeypatch.setattr(board, "RESULTS", tmp_path)
    monkeypatch.setattr(board, "load_gold", fake_gold)
    return board.board_rows(include_baseline=False)


def test_a_self_reported_file_boards_as_self_reported(monkeypatch, tmp_path):
    rows_out = board_over(monkeypatch, tmp_path, submission())
    assert rows_out[0]["label"] == "my-extractor"
    assert rows_out[0]["attested"] is False


def test_a_harness_file_boards_as_attested(monkeypatch, tmp_path):
    harness_file = submission(produced_by=dict(result_io.HARNESS_PRODUCED_BY),
                              items=attested_rows())
    rows_out = board_over(monkeypatch, tmp_path, harness_file)
    assert rows_out[0]["attested"] is True


def test_the_board_marks_self_reported_rows_and_footnotes_the_mark(monkeypatch, tmp_path, capsys):
    board_over(monkeypatch, tmp_path, submission())
    monkeypatch.setattr(board, "RESULTS", tmp_path)
    monkeypatch.setattr(board, "load_gold", fake_gold)
    board.main([])
    printed = capsys.readouterr().out
    assert "| my-extractor † |" in printed
    assert "† self-reported" in printed


# --- the standalone validator -----------------------------------------------------------------------

def install_fake_gold(monkeypatch):
    """-> the list of revisions load_gold was asked for."""
    asked = []

    def fake_load_gold(revision="main"):
        asked.append(revision)
        return ("{}", {"card-1": {}, "card-2": {}}, SHA)

    monkeypatch.setattr(cli, "load_gold", fake_load_gold)
    return asked


def test_the_cli_accepts_a_good_submission(monkeypatch, tmp_path, capsys):
    asked = install_fake_gold(monkeypatch)
    path = tmp_path / "my-extractor.json"
    path.write_text(json.dumps(submission()))

    assert cli.main([str(path)]) == 0

    printed = capsys.readouterr().out
    assert printed.startswith("OK: 2 items")
    assert "self-reported" in printed
    # by default the ids are checked against the snapshot the file itself names
    assert asked == [SHA]


def test_the_cli_lists_every_problem_and_exits_1(monkeypatch, tmp_path, capsys):
    install_fake_gold(monkeypatch)
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(submission(model={}, items=rows(("card-1", "{}")))))

    assert cli.main([str(path)]) == 1

    printed = capsys.readouterr().out
    assert "2 problem(s)" in printed
    assert "model.label: missing" in printed
    assert "have no row: card-2" in printed
    assert "docs/SUBMISSION.md" in printed


def test_the_cli_accepts_the_minimal_shape_without_comment(monkeypatch, tmp_path, capsys):
    install_fake_gold(monkeypatch)
    document = {"model": {"label": "anon"}, "dataset": {"inference_revision": SHA},
                "items": rows(("card-1", "{}"), ("card-2", "{}"))}
    path = tmp_path / "anon.json"
    path.write_text(json.dumps(document))

    assert cli.main([str(path)]) == 0
    printed = capsys.readouterr().out
    assert "warning" not in printed
    assert printed.startswith("OK: 2 items")
    assert "self-reported" in printed


def test_the_cli_prints_the_resolved_sha_when_the_revision_is_a_branch(monkeypatch, tmp_path,
                                                                        capsys):
    install_fake_gold(monkeypatch)
    path = tmp_path / "branchy.json"
    path.write_text(json.dumps(submission(dataset={"inference_revision": "main"})))

    assert cli.main([str(path)]) == 1
    printed = capsys.readouterr().out
    assert "dataset.inference_revision: 'main' is not a 40-character commit sha" in printed
    assert f"hint: the snapshot checked against resolves to {SHA}" in printed


def test_the_cli_can_be_pointed_at_another_snapshot(monkeypatch, tmp_path):
    asked = install_fake_gold(monkeypatch)
    path = tmp_path / "my-extractor.json"
    path.write_text(json.dumps(submission()))

    assert cli.main([str(path), "--dataset-revision", "main"]) == 0
    assert asked == ["main"]


def test_the_cli_checks_the_ids_of_an_nls_submission(monkeypatch, tmp_path, capsys):
    """The dataset the file names decides which gold, if any, the ids are checked against."""
    asked = install_fake_gold(monkeypatch)
    path = tmp_path / "my-extractor.json"
    path.write_text(json.dumps(submission(items=rows(("card-1", "{}")))))   # card-2 is missing

    assert cli.main([str(path)]) == 1
    printed = capsys.readouterr().out
    assert "items: 1 id(s) in the snapshot have no row: card-2" in printed
    assert "ids not checked" not in printed
    assert asked == [SHA]


def test_the_cli_runs_the_shape_checks_on_a_submission_over_another_dataset(monkeypatch, tmp_path,
                                                                            capsys):
    """`harness.py --dataset <hub id>` writes these. Loading NLS gold for one would report every
    row as an id that is not in the snapshot — a good file refused, for the wrong reason."""
    asked = install_fake_gold(monkeypatch)
    path = tmp_path / "my-extractor.json"
    path.write_text(json.dumps(submission(
        dataset={"id": "acme/other-cards", "inference_revision": SHA},
        items=rows(("who-is-this", "{}")))))

    assert cli.main([str(path)]) == 0
    printed = capsys.readouterr().out
    assert "ids not checked: no gold loader for acme/other-cards" in printed
    assert "OK: 1 items" in printed
    assert asked == []                    # and no gold was downloaded to not use


def test_the_shape_checks_still_bite_on_another_dataset(monkeypatch, tmp_path, capsys):
    """No id check is not no checks."""
    install_fake_gold(monkeypatch)
    path = tmp_path / "my-extractor.json"
    path.write_text(json.dumps(submission(
        dataset={"id": "acme/other-cards", "inference_revision": SHA},
        items=[{"id": "card-1", "prediction": {"already": "parsed"}}])))

    assert cli.main([str(path)]) == 1
    printed = capsys.readouterr().out
    assert "ids not checked: no gold loader for acme/other-cards" in printed
    assert "prediction is a dict, not a string" in printed


def test_the_cli_says_so_when_the_file_is_not_a_submission_at_all(monkeypatch, tmp_path):
    install_fake_gold(monkeypatch)
    path = tmp_path / "notes.json"
    path.write_text("[1, 2, 3]")

    with pytest.raises(SystemExit) as refusal:
        cli.main([str(path)])
    assert "the top level is a list, not a JSON object" in str(refusal.value)


# --- one board, one snapshot ------------------------------------------------------------------------

def fingerprinted(document: dict) -> dict:
    """The same submission plus the optional slice fields the harness writes."""
    ids = [row["id"] for row in document["items"]]
    document["dataset"] = {**document["dataset"], "item_count": len(ids),
                           "item_ids_sha256": result_io.item_ids_sha256(ids)}
    return document


def test_the_slice_fields_are_checked_only_when_the_file_carries_them():
    """A producer who never wrote them is not held to them; one who did is."""
    assert result_io.validate_submission(submission(), GOLD_IDS) == []
    assert result_io.validate_submission(fingerprinted(submission()), GOLD_IDS) == []


def test_a_file_that_lost_rows_after_it_finished_is_refused(tmp_path):
    """The failure a shrinking rescore used to produce: fewer rows, still `complete: true`."""
    shrunk = fingerprinted(submission())
    shrunk["items"] = shrunk["items"][:1]
    write(tmp_path, "shrunk.json", shrunk)

    problems = result_io.validate_submission(shrunk, GOLD_IDS)
    assert problems == [
        "dataset.item_count: 2, but the file holds 1 rows — rows have been added or removed "
        "since this run finished",
        "dataset.item_ids_sha256: does not match the ids in the file — the rows are not the "
        "slice this run was made over, or they are in another order",
        "items: 1 id(s) in the snapshot have no row: card-2"]

    with pytest.raises(SystemExit) as refusal:
        result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS)
    assert "dataset.item_count: 2, but the file holds 1 rows" in str(refusal.value)


@pytest.mark.parametrize("value, expected", [
    ("2", "dataset.item_count: '2' is not an integer"),
    (2.0, "dataset.item_count: 2.0 is not an integer"),
    (True, "dataset.item_count: True is not an integer"),
])
def test_an_item_count_of_the_wrong_type_is_a_problem_not_a_skipped_check(value, expected):
    """A field of the wrong type used to turn off the check it belongs to — the one check that
    catches a file edited after it was written."""
    document = fingerprinted(submission())
    document["dataset"]["item_count"] = value
    problems = [p for p in result_io.validate_submission(document, GOLD_IDS)
                if p.startswith("dataset.item_count")]
    assert len(problems) == 1
    assert problems[0].startswith(expected)


@pytest.mark.parametrize("value", ["", "not-a-hash", "abc", "d" * 40, "D" * 64, 12345])
def test_a_fingerprint_that_is_not_a_sha256_is_a_problem(value):
    """40 hex is a commit sha, not an id fingerprint; upper case is not what the recipe emits."""
    document = fingerprinted(submission())
    document["dataset"]["item_ids_sha256"] = value
    problems = [p for p in result_io.validate_submission(document, GOLD_IDS)
                if p.startswith("dataset.item_ids_sha256")]
    assert len(problems) == 1
    assert "is not a 64-character sha256 hex digest" in problems[0]


@pytest.mark.parametrize("field", ["item_count", "item_ids_sha256"])
def test_a_slice_field_written_out_null_reads_as_one_that_was_left_out(field):
    """`null` is how a producer says "no value", not a wrong-typed one."""
    document = fingerprinted(submission())
    document["dataset"][field] = None
    assert result_io.validate_submission(document, GOLD_IDS) == []


def test_reordering_the_rows_breaks_the_fingerprint_and_nothing_else():
    reordered = fingerprinted(submission())
    reordered["items"] = list(reversed(reordered["items"]))
    assert result_io.validate_submission(reordered, GOLD_IDS) == [
        "dataset.item_ids_sha256: does not match the ids in the file — the rows are not the "
        "slice this run was made over, or they are in another order"]


def test_two_files_made_over_different_snapshots_do_not_share_a_board(tmp_path):
    write(tmp_path, "a.json", fingerprinted(submission()))
    other = submission(model={"label": "other", "id": "acme/other"},
                       dataset={"id": NLS_DATASET, "inference_revision": "e" * 40})
    write(tmp_path, "b.json", fingerprinted(other))

    with pytest.raises(SystemExit) as refusal:
        result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS)

    message = str(refusal.value)
    assert "not made over the same dataset snapshot (dataset.inference_revision differ)" in message
    assert f"a.json: {NLS_DATASET} @ dddddddddddd · 2 items" in message
    assert f"b.json: {NLS_DATASET} @ eeeeeeeeeeee · 2 items" in message


def test_a_file_over_another_dataset_is_refused_by_a_consumer_that_says_which_one_it_loaded(
        tmp_path):
    """Id sets cannot catch this: two collections can both have a `card-1`. The consumer knows
    which gold it read and nothing in the file does, so the consumer is what says."""
    write(tmp_path, "elsewhere.json", submission(
        dataset={"id": "acme/other-cards", "inference_revision": SHA}))

    with pytest.raises(SystemExit) as refusal:
        result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS,
                                        expected_dataset_id=NLS_DATASET)

    assert (f"dataset.id: 'acme/other-cards' is not the dataset this board scores "
            f"({NLS_DATASET})") in str(refusal.value)


def test_a_consumer_that_names_no_dataset_takes_whatever_it_finds(tmp_path):
    """leaderboard.py aggregates any results directory and has no gold to be wrong about."""
    write(tmp_path, "elsewhere.json", submission(
        dataset={"id": "acme/other-cards", "inference_revision": SHA}))

    found = result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS)
    assert [p.name for p in found] == ["elsewhere.json"]


def test_a_file_that_names_no_dataset_is_taken_as_the_board_s_own():
    document = submission(dataset={"inference_revision": SHA})
    problems = result_io.contract_problems(document, GOLD_IDS, expected_dataset_id=NLS_DATASET)
    assert problems == []


def test_files_from_one_snapshot_are_published(tmp_path):
    write(tmp_path, "a.json", fingerprinted(submission()))
    write(tmp_path, "b.json", fingerprinted(submission(model={"label": "other",
                                                              "id": "acme/other"})))
    found = result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS)
    assert [p.name for p in found] == ["a.json", "b.json"]


def test_a_differing_item_count_alone_stops_the_board(tmp_path):
    """Two files over the same commit that disagree about how many items it has."""
    write(tmp_path, "a.json", fingerprinted(submission()))
    short = submission(model={"label": "other", "id": "acme/other"},
                       items=[{"id": "card-1", "prediction": "{}"}])
    write(tmp_path, "b.json", fingerprinted(short))

    with pytest.raises(SystemExit) as refusal:
        # gold_ids=None so the per-file id check cannot be what refuses it
        result_io.discover_result_files(tmp_path, gold_ids=None)
    assert "dataset.item_ids_sha256, dataset.item_count differ" in str(refusal.value)


def test_two_files_claiming_one_label_are_refused(tmp_path):
    """One label is one row. Two files carrying it means one silently wins, and which one depends
    on the order the directory happened to list — a stale rerun quietly replacing a good run."""
    write(tmp_path, "a.json", fingerprinted(submission()))
    write(tmp_path, "b.json", fingerprinted(submission()))

    with pytest.raises(SystemExit) as refusal:
        result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS)

    message = str(refusal.value)
    assert "2 files claim model.label 'my-extractor': a.json, b.json" in message
    assert "one label is one row on the board" in message


def test_files_with_different_labels_share_a_board(tmp_path):
    write(tmp_path, "a.json", fingerprinted(submission()))
    write(tmp_path, "b.json", fingerprinted(submission(
        model={"label": "other", "id": "acme/other"})))
    assert len(result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS)) == 2


def test_a_file_not_named_after_its_label_is_a_warning_not_a_refusal(tmp_path, capsys):
    """Renaming somebody's submission is not the gate's business; saying so is."""
    write(tmp_path, "whatever.json", fingerprinted(submission()))

    found = result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS)

    assert [p.name for p in found] == ["whatever.json"]
    assert ("warning: whatever.json carries model.label 'my-extractor', so it boards under that "
            "name rather than its own (the convention is my-extractor.json)") in \
        capsys.readouterr().out


def test_a_file_named_after_its_label_says_nothing(tmp_path, capsys):
    write(tmp_path, "my-extractor.json", fingerprinted(submission()))
    result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS)
    assert capsys.readouterr().out == ""


def test_a_legacy_file_is_named_as_unchecked_rather_than_assumed_to_match(tmp_path, capsys):
    write(tmp_path, "modern.json", fingerprinted(submission()))
    write(tmp_path, "old.json",
          {"model": {"label": "old"}, "items": [{"id": "card-1"}, {"id": "card-2"}]})

    found = result_io.discover_result_files(tmp_path, gold_ids=GOLD_IDS)

    assert [p.name for p in found] == ["modern.json", "old.json"]
    assert "warning: legacy file old.json: snapshot unknown, not checked against the others" \
        in capsys.readouterr().out


# --- counts on the page are counted ------------------------------------------------------------------

def three_card_gold():
    schema = json.dumps({"type": "object", "properties": {"heading": {"type": "string"}}})
    return (schema,
            {f"card-{n}": {"gold": {"heading": "Allan, T."}, "label_status": "verified"}
             for n in (1, 2, 3)},
            SHA)


def test_the_board_counts_the_cards_it_scored_instead_of_saying_98(monkeypatch, tmp_path, capsys):
    three = [{"id": f"card-{n}", "prediction": "{}"} for n in (1, 2, 3)]
    for name, label in (("a.json", "model-a"), ("b.json", "model-b")):
        write(tmp_path, name, fingerprinted(submission(
            model={"label": label, "id": f"acme/{label}"}, items=list(three))))

    monkeypatch.setattr(board, "RESULTS", tmp_path)
    monkeypatch.setattr(board, "load_gold", three_card_gold)
    board.main([])

    printed = capsys.readouterr().out
    assert "ident-clean /3" in printed
    assert "98" not in printed


def test_the_page_subtitle_counts_its_cards_and_models():
    import site_nls

    assert site_nls.subtitle(3, 2) == "3 index cards · 2 models"
    assert site_nls.subtitle(1, 1) == "1 index card · 1 model"
