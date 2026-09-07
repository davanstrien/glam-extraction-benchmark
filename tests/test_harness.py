"""Harness contract tests. No network, no inference, no provider SDKs required.

Two seams make that possible and are themselves under test here:
  - every provider SDK is imported inside the adapter that uses it, so `import harness` needs
    none of them (test_import_needs_no_provider_sdk proves it in a subprocess);
  - `main(argv)` and a "fake" adapter kind, so a whole run can be driven from canned content.
"""
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import textwrap
import types

import pytest

import board
import harness
import rescore
import result_io
import site_nls

GLAM_BENCH = pathlib.Path(__file__).resolve().parent.parent / "glam_bench"
NLS_REPO = "NationalLibraryOfScotland/index-cards-eval"

SCHEMA = {"type": "object", "properties": {"heading": {"type": "string"},
                                           "ms_no": {"type": "string", "x-match": "exact"}}}
SCHEMA_STR = json.dumps(SCHEMA)
GOLD = {"heading": "Allan, T.", "ms_no": "MS.123"}
PERFECT = json.dumps(GOLD)

ITEMS = [
    {"id": "card-1", "image": "PIXELS", "target_schema": SCHEMA_STR, "gold": GOLD},
    {"id": "card-2", "image": "PIXELS", "target_schema": SCHEMA_STR, "gold": GOLD},
]

PROVENANCE_KEYS = {"served_by", "endpoint", "endpoint_attested", "model_revision_hub_main",
                   "timestamp", "latency_s", "retry_wait_s", "max_tokens", "temperature",
                   "request_options", "thinking_disabled", "finish_reason", "harness_git_sha"}


def fake_spec(label, **extra):
    spec = {"label": label, "kind": "fake", "id": "fake/model", "params": "0B", "cost": "free"}
    spec.update(extra)
    return spec


def install_fake_run(monkeypatch, tmp_path, spec, items=ITEMS, sha="d" * 40):
    """Point the harness at tmp_path and one fake model; -> the list of load_items calls."""
    monkeypatch.setattr(harness, "MODELS", [spec])
    monkeypatch.setattr(harness, "RESULTS", tmp_path)
    calls = []

    def fake_load_items(dataset, limit, revision):
        calls.append((dataset, limit, revision))
        rows = list(items)
        if limit is not None:
            rows = rows[:limit]
        return rows, NLS_REPO, sha

    monkeypatch.setattr(harness, "load_items", fake_load_items)
    return calls


def run_harness(monkeypatch, tmp_path, spec, argv=("--dataset", "nls"), items=ITEMS):
    """Drive main() end to end over canned items, writing into tmp_path."""
    install_fake_run(monkeypatch, tmp_path, spec, items)
    harness.main(["--models", spec["label"], *argv])
    return json.loads((tmp_path / "nls" / f"{spec['label']}.json").read_text())


# --- the test seam ----------------------------------------------------------------------------

def test_import_needs_no_provider_sdk():
    """harness must import with datasets/gradio_client/huggingface_hub/openai unavailable."""
    program = textwrap.dedent("""
        import sys
        for blocked in ("datasets", "gradio_client", "huggingface_hub", "openai"):
            sys.modules[blocked] = None      # makes `import blocked` raise ImportError
        sys.path.insert(0, sys.argv[1])
        import harness
        assert set(harness.ADAPTERS) >= {"router_vlm", "openai", "fake"}
        print("imported")
    """)
    done = subprocess.run([sys.executable, "-c", program, str(GLAM_BENCH)],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert "imported" in done.stdout


def refuse_to_load(monkeypatch):
    """Make any dataset load an immediate failure.

    Every configuration error must be caught before this point: a download is the first thing that
    costs time, and inference is the first thing that costs money.
    """
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("the dataset must not be loaded: the run should have failed earlier")

    monkeypatch.setattr(harness, "load_items", must_not_run)


def test_unknown_model_label_errors_before_any_dataset_load(monkeypatch):
    monkeypatch.setattr(harness, "MODELS", [fake_spec("real-label")])
    refuse_to_load(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--models", "typo-label", "--dataset", "nls"])
    assert "typo-label" in str(excinfo.value)
    assert "real-label" in str(excinfo.value)


# --- the result contract ----------------------------------------------------------------------

def test_envelope_carries_dataset_scoring_and_run_provenance(monkeypatch, tmp_path):
    spec = fake_spec("fake-ok", canned=PERFECT, finish_reason="stop",
                     base_url="https://user:secret@host/v1?key=abc", api_key_env="HF_TOKEN")
    out = run_harness(monkeypatch, tmp_path, spec)

    assert out["format_version"] == 2
    assert list(out) == ["format_version", "model", "dataset", "generation", "scoring", "run",
                         "produced_by", "items"]

    # the harness is the reference producer of a submission, and says so in the file
    assert out["produced_by"] == {"producer": "harness", "attested": True}

    # base_url / api_key_env are run details, not model identity: they stay out of the artifact.
    assert out["model"] == {"label": "fake-ok", "kind": "fake", "id": "fake/model",
                            "params": "0B", "cost": "free"}

    # `id` is a resolvable Hub repo id; the CLI shorthand is recorded separately.
    expected_hash = hashlib.sha256("card-1\ncard-2".encode()).hexdigest()
    assert out["dataset"] == {"id": NLS_REPO, "alias": "nls", "requested_revision": "main",
                              "inference_revision": "d" * 40, "split": "train",
                              "item_count": 2, "item_ids_sha256": expected_hash}

    # what was asked of the model, once for the whole run, and the same in every row's provenance
    assert out["generation"] == {"max_tokens": 900, "temperature": 0, "guided": False,
                                 "request_options": {}}

    assert out["scoring"] == {"scorer_version": harness.SCORER_VERSION}
    assert out["run"]["complete"] is True
    assert out["run"]["started_at"].endswith("Z")
    assert out["run"]["finished_at"].endswith("Z")
    # `resumed_at` appears only once a run has actually been resumed.
    assert set(out["run"]) == {"started_at", "finished_at", "harness_git_sha", "complete",
                               "transport_errors"}
    assert out["run"]["transport_errors"] == 0

    for item in out["items"]:
        assert item["inference_status"] == "ok"
        assert item["attempts"] == 1
        assert item["prediction"] == PERFECT
        assert item["schema_valid"] == 1.0
        assert item["content_f1"] == 1.0
        assert "error" not in item
        assert set(item["provenance"]) == PROVENANCE_KEYS
        assert item["provenance"]["finish_reason"] == "stop"
        assert item["provenance"]["max_tokens"] == 900
        assert item["provenance"]["retry_wait_s"] == 0.0
        assert item["provenance"]["temperature"] == 0
        assert item["provenance"]["served_by"] == "fake-adapter"
        # nothing was asked for beyond the defaults, and the row says so rather than staying silent
        assert item["provenance"]["request_options"] == {}
        assert item["provenance"]["thinking_disabled"] is False


def test_the_harness_writes_a_submission_that_passes_its_own_validator(monkeypatch, tmp_path):
    """The reference producer has to satisfy the contract it publishes (docs/SUBMISSION.md).

    Both shapes a run can write are checked: a clean answer, and a row no endpoint answered.
    """
    gold_ids = {item["id"] for item in ITEMS}

    good = run_harness(monkeypatch, tmp_path, fake_spec("fake-ok", canned=PERFECT))
    assert result_io.validate_submission(good, gold_ids) == []
    assert result_io.is_attested(good) is True

    # a transport failure is a null prediction the file explains, which the contract allows
    failed = run_harness(monkeypatch, tmp_path, fake_spec("fake-down", raises="gateway timed out"))
    assert result_io.validate_submission(failed, gold_ids) == []
    assert result_io.is_attested(failed) is True
    # ...and is still kept off the board by the publication gate, on its own rule
    assert result_io.publication_status(failed, gold_ids) == "transport_errors"


def test_dataset_alias_is_null_when_the_cli_already_named_the_repo():
    block = harness.build_dataset_block("org/set", "org/set", "main", "a" * 40, ITEMS)
    assert block["id"] == "org/set"
    assert block["alias"] is None


def test_max_tokens_flag_reaches_the_row_provenance(monkeypatch, tmp_path):
    spec = fake_spec("fake-tokens", canned=PERFECT)
    out = run_harness(monkeypatch, tmp_path, spec, argv=("--dataset", "nls", "--max-tokens", "1600"))
    assert out["items"][0]["provenance"]["max_tokens"] == 1600


def test_adapter_exception_is_a_transport_error(monkeypatch, tmp_path):
    spec = fake_spec("fake-down", raises="gateway timed out")
    out = run_harness(monkeypatch, tmp_path, spec)

    row = out["items"][0]
    assert row["inference_status"] == "transport_error"
    assert row["prediction"] is None
    assert row["error"].startswith("RuntimeError: gateway timed out")
    # A bare RuntimeError names nothing the classifier recognises. Not retried, and not filed
    # under "the endpoint said no" either.
    assert row["failure_class"] == "unclassified"
    assert row["attempts"] == 1
    assert row["content_f1"] == 0.0
    assert row["schema_valid"] == 0.0
    # A failed call still records where it was sent and when.
    assert set(row["provenance"]) == PROVENANCE_KEYS
    assert row["provenance"]["endpoint"] == "fake:fake-down"


def test_unparseable_response_is_a_finding_not_a_failure(monkeypatch, tmp_path):
    spec = fake_spec("fake-babble", canned="I am sorry, I cannot read this card.")
    out = run_harness(monkeypatch, tmp_path, spec)

    row = out["items"][0]
    assert row["inference_status"] == "ok"
    assert row["prediction"] == "I am sorry, I cannot read this card."
    assert row["schema_valid"] == 0.0
    assert "error" not in row
    assert "scorer_error" not in row


def test_an_empty_answer_is_a_blank_prediction_not_a_null_one(monkeypatch, tmp_path):
    """A successful call whose content is null — a reasoning model that spent its whole budget
    thinking, a filtered completion — used to write `prediction: null` with status "ok", which
    the submission contract refuses outright. It is a blank answer, and scores as one."""
    spec = fake_spec("fake-empty")            # run_fake returns spec.get("canned"), i.e. None
    out = run_harness(monkeypatch, tmp_path, spec)

    row = out["items"][0]
    assert row["prediction"] == ""
    assert row["inference_status"] == "ok"
    assert row["schema_valid"] == 0.0
    assert row["content_f1"] == 0.0
    assert "error" not in row and "scorer_error" not in row
    # and the file it produces is one its own validator accepts
    assert result_io.validate_submission(out, {item["id"] for item in ITEMS}) == []


def test_scorer_crash_is_not_reported_as_a_transport_error(monkeypatch, tmp_path):
    def exploding_score(*_args, **_kwargs):
        raise ZeroDivisionError("scorer bug")

    monkeypatch.setattr(harness, "score", exploding_score)
    spec = fake_spec("fake-scorer-bug", canned=PERFECT)
    out = run_harness(monkeypatch, tmp_path, spec)

    row = out["items"][0]
    assert row["inference_status"] == "ok"
    assert row["prediction"] == PERFECT
    assert row["scorer_error"].startswith("ZeroDivisionError: scorer bug")
    assert "error" not in row
    assert row["content_f1"] == 0.0


# --- endpoint description ---------------------------------------------------------------------

def test_router_auto_is_recorded_as_unattested():
    served_by, endpoint, attested = harness.describe_endpoint(
        {"kind": "router_vlm", "id": "org/model", "provider": None})
    assert (served_by, endpoint, attested) == ("hf-inference-providers", "router:auto", False)


def test_pinned_router_provider_is_attested():
    _served_by, endpoint, attested = harness.describe_endpoint(
        {"kind": "router_vlm", "id": "org/model", "provider": "together"})
    assert (endpoint, attested) == ("router:together", True)


def test_openai_endpoint_drops_credentials_and_query():
    _served_by, endpoint, _attested = harness.describe_endpoint(
        {"kind": "openai", "id": "m", "base_url": "https://bob:hf_secret@abc--8000.hf.jobs/v1?k=1"})
    assert endpoint == "https://abc--8000.hf.jobs/v1"


def test_space_endpoint_names_the_space():
    served_by, endpoint, _attested = harness.describe_endpoint(
        {"kind": "nuextract_space", "id": "numind/NuExtract3"})
    assert (served_by, endpoint) == ("hf-space", "space:numind/NuExtract3")


def test_model_revision_is_not_looked_up_for_space_kinds():
    # A Space id is not a model repo id; a model_info lookup there would resolve another repo.
    assert harness.resolve_model_revision({"kind": "nuextract_space", "id": "numind/NuExtract3"}) is None
    assert harness.resolve_model_revision({"kind": "openai", "id": "gpt-4o"}) is None


# --- rescore round-trip -----------------------------------------------------------------------

def _scored_item(**extra):
    item = {"id": "card-1", "prediction": PERFECT, "content_f1": 1.0, "schema_valid": 1.0}
    item.update(extra)
    return item


def _fake_gold():
    return (SCHEMA_STR,
            {"card-1": {"gold": GOLD, "label_status": "verified",
                        "verdict": "ok", "image_type": "typed"}},
            "d" * 40)


def test_rescore_write_preserves_new_and_old_top_level_metadata(monkeypatch, tmp_path):
    results = tmp_path / "nls"
    results.mkdir()
    provenance = {k: None for k in PROVENANCE_KEYS}
    new_format = {
        "format_version": 2,
        "model": {"label": "new", "kind": "fake", "id": "fake/model", "params": "0B"},
        "dataset": {"id": NLS_REPO, "requested_revision": "main", "inference_revision": "d" * 40,
                    "split": "train", "item_count": 1,
                    "item_ids_sha256": result_io.item_ids_sha256(["card-1"])},
        "scoring": {"scorer_version": "1999-01-01-stale"},
        "run": {"started_at": "2026-08-27T12:00:00Z", "finished_at": "2026-08-27T12:01:00Z",
                "harness_git_sha": "c" * 40, "complete": True},
        "items": [_scored_item(inference_status="ok", attempts=1, provenance=provenance)],
    }
    old_format = {"model": {"label": "old", "params": "9B"}, "items": [_scored_item()]}
    (results / "new.json").write_text(json.dumps(new_format))
    (results / "old.json").write_text(json.dumps(old_format))

    monkeypatch.setattr(rescore, "RESULTS", results)
    monkeypatch.setattr(rescore, "load_gold", _fake_gold)
    rescore.main(["--write"])

    after_new = json.loads((results / "new.json").read_text())
    assert list(after_new) == ["format_version", "model", "dataset", "scoring", "run", "items"]
    # untouched: a rescore does not change what was inferred, or by which harness
    for block in ("format_version", "model", "dataset", "run"):
        assert after_new[block] == new_format[block]
    # refreshed: these numbers came out of the scorer running now, so the file must say so
    assert after_new["scoring"]["scorer_version"] == rescore.SCORER_VERSION
    assert after_new["scoring"]["scorer_version"] != "1999-01-01-stale"
    assert after_new["scoring"]["rescored_at"].endswith("Z")
    row = after_new["items"][0]
    assert row["inference_status"] == "ok"
    assert row["attempts"] == 1
    assert row["provenance"] == provenance
    assert row["label_status"] == "verified"

    after_old = json.loads((results / "old.json").read_text())
    assert list(after_old) == ["model", "scoring", "items"]   # gains only the scoring block
    assert after_old["model"] == old_format["model"]
    assert after_old["scoring"]["scorer_version"] == rescore.SCORER_VERSION


def _rescorable(results, extra_rows=()):
    """A finished, self-describing format-2 file in `results`.

    The dataset block is kept true to the rows, because that is what a rescore must not quietly
    break: item_count and the id fingerprint describe the slice, and a file that drops a row
    while claiming to be complete fails its own contract.
    """
    rows = [_scored_item(inference_status="ok", attempts=1), *extra_rows]
    ids = [row["id"] for row in rows]
    document = {
        "format_version": 2,
        "model": {"label": "new", "kind": "fake", "id": "fake/model", "params": "0B"},
        "dataset": {"id": NLS_REPO, "inference_revision": "d" * 40, "split": "train",
                    "item_count": len(ids),
                    "item_ids_sha256": result_io.item_ids_sha256(ids)},
        "scoring": {"scorer_version": "1999-01-01-stale"},
        "run": {"started_at": "t0", "finished_at": "t1", "complete": True,
                "transport_errors": 0},
        "items": rows,
    }
    (results / "new.json").write_text(json.dumps(document))
    return document


def _legacy_rescorable(results, extra_rows=()):
    """A pre-envelope file — the only kind that reaches rescore holding a row with no gold.

    A format-2 file cannot: discovery checks its ids against the gold first and refuses it
    (test_a_format_2_file_with_a_stray_row_never_reaches_the_rescorer). Legacy files carry no
    ids to check, and the June–August results were exactly these.
    """
    rows = [_scored_item(), *extra_rows]
    (results / "old.json").write_text(json.dumps({"model": {"label": "old", "params": "9B"},
                                                  "items": rows}))


def hand_the_files_straight_to_the_rescorer(monkeypatch):
    """Replace discovery with a plain glob, so a test can reach the rescorer past the gate.

    The gate refuses a file holding a row that is not in the gold — a legacy one included, since
    the id check runs on those too. That is the right behaviour and has its own tests; it also
    means the only way to exercise what the rescorer does with such a row is to hand it the file.
    The guard is kept rather than deleted because rescore.py is importable, and a row with no
    gold is a KeyError away from a crash without it.
    """
    def every_json(results_dir, *_args, **_kwargs):
        return sorted(pathlib.Path(results_dir).glob("*.json"))

    monkeypatch.setattr(result_io, "discover_result_files", every_json)


def test_rescore_keeps_a_row_it_cannot_score_rather_than_shrinking_the_file(monkeypatch, tmp_path,
                                                                           capsys):
    """Dropping the row loses a prediction nobody can get back without paying for it again, and
    on a file that counts its own rows it turns a finished run into an invalid one."""
    results = tmp_path / "nls"
    results.mkdir()
    _legacy_rescorable(results,
                       extra_rows=[_scored_item(id="card-9", content_f1=0.9, schema_valid=1.0)])

    hand_the_files_straight_to_the_rescorer(monkeypatch)
    monkeypatch.setattr(rescore, "RESULTS", results)
    monkeypatch.setattr(rescore, "load_gold", _fake_gold)     # gold has card-1 only
    rescore.main(["--write"])

    after = json.loads((results / "old.json").read_text())
    assert [row["id"] for row in after["items"]] == ["card-1", "card-9"]

    unscored = after["items"][1]
    assert unscored["scorer_error"] == "no gold for id"
    assert unscored["prediction"] == PERFECT    # the expensive part of the row is kept
    assert unscored["content_f1"] == 0.0        # not the 0.9 it used to carry
    assert unscored["schema_valid"] == 0.0
    assert "label_status" not in unscored       # there is no gold row to have a status

    printed = capsys.readouterr().out
    assert "1 prediction(s) with no matching gold row, kept in the file unscored" in printed


def test_rescore_leaves_an_unscorable_row_out_of_the_rates(monkeypatch, tmp_path, capsys):
    """An unscored row is not a zero: averaging it in would report a drop nobody measured."""
    results = tmp_path / "nls"
    results.mkdir()
    _legacy_rescorable(results, extra_rows=[_scored_item(id="card-9")])

    hand_the_files_straight_to_the_rescorer(monkeypatch)
    monkeypatch.setattr(rescore, "RESULTS", results)
    monkeypatch.setattr(rescore, "load_gold", _fake_gold)
    rescore.main([])

    # one scorable row, scored perfectly: the table says 1.000, not 0.500
    assert "| 1.000 |" in capsys.readouterr().out


@pytest.mark.parametrize("make_file", [_rescorable, _legacy_rescorable])
def test_a_file_with_a_stray_row_never_reaches_the_rescorer(monkeypatch, tmp_path, make_file):
    """The other half of the guarantee: such a file cannot be shrunk because it cannot be read.

    Its ids are checked against the gold before anything is scored — a legacy file's too, since
    the alternative made "no dataset block" the way to board any set of rows at all. So the
    case rescore handles defensively above is one nothing reaches it through.
    """
    results = tmp_path / "nls"
    results.mkdir()
    make_file(results, extra_rows=[_scored_item(id="card-9")])

    monkeypatch.setattr(rescore, "RESULTS", results)
    monkeypatch.setattr(rescore, "load_gold", _fake_gold)

    with pytest.raises(SystemExit) as refusal:
        rescore.main(["--write"])
    assert "items: 1 id(s) that are not in the snapshot: card-9" in str(refusal.value)


def test_rescore_records_which_gold_it_scored_against(monkeypatch, tmp_path):
    """The scorer version says what produced the numbers; the gold revision says from what."""
    results = tmp_path / "nls"
    results.mkdir()
    _rescorable(results)

    monkeypatch.setattr(rescore, "RESULTS", results)
    monkeypatch.setattr(rescore, "load_gold", _fake_gold)
    rescore.main(["--write"])

    after = json.loads((results / "new.json").read_text())
    assert after["scoring"]["gold_revision"] == "d" * 40
    assert after["scoring"]["scorer_version"] == rescore.SCORER_VERSION
    assert after["scoring"]["rescored_at"].endswith("Z")
    # and a rescore leaves a valid submission valid
    assert result_io.validate_submission(after, {"card-1"}) == []


def test_rescore_write_goes_through_the_atomic_writer(monkeypatch, tmp_path):
    """A rescore rewrites a published artifact in place; a truncated one is unrecoverable."""
    results = tmp_path / "nls"
    results.mkdir()
    _rescorable(results)

    written = []
    real_write = result_io.atomic_write_json

    def spy(path, payload):
        written.append(path)
        real_write(path, payload)

    monkeypatch.setattr(result_io, "atomic_write_json", spy)
    monkeypatch.setattr(rescore, "RESULTS", results)
    monkeypatch.setattr(rescore, "load_gold", _fake_gold)
    rescore.main(["--write"])

    assert [path.name for path in written] == ["new.json"]
    assert [path.name for path in results.iterdir()] == ["new.json"]   # no temp file left behind


# --- dataset revision pinning -----------------------------------------------------------------

def _fake_hub(tmp_path, calls, sample_ids=("card-1",)):
    """A stand-in Hub holding one gold set, whose `_sample_id`s the caller chooses.

    Both artifacts a loader reads — `metadata.jsonl` and the `train` split — are built from the
    same `sample_ids`, so a repeated id is present in whichever of them the loader happens to use.
    """
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(SCHEMA))
    meta_file = tmp_path / "metadata.jsonl"
    meta_file.write_text("".join(
        json.dumps({"_sample_id": sample_id, "_label_status": "verified",
                    "_verdict": "ok", "image_type": "typed", **GOLD}) + "\n"
        for sample_id in sample_ids))

    class FakeApi:
        def dataset_info(self, repo_id, revision=None):
            calls.append(("dataset_info", repo_id, revision))
            return types.SimpleNamespace(sha="e" * 40)

    def fake_download(_repo_id, filename, *, revision, **_ignored):
        calls.append(("hf_hub_download", filename, revision))
        return str(schema_file if filename == "schema.json" else meta_file)

    def fake_load_dataset(_repo_id, *, split, revision):
        calls.append(("load_dataset", split, revision))
        return [{"_sample_id": sample_id, "image": "PIXELS", **GOLD} for sample_id in sample_ids]

    return (types.SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_download),
            types.SimpleNamespace(load_dataset=fake_load_dataset))


@pytest.mark.parametrize("loader", ["load_nls", "load_gold"])
def test_nls_loaders_resolve_one_revision_and_use_it_everywhere(monkeypatch, tmp_path, loader):
    import nls

    calls = []
    fake_hub, fake_datasets = _fake_hub(tmp_path, calls)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    result = getattr(nls, loader)()
    assert result[-1] == "e" * 40                      # the resolved SHA is returned to the caller

    assert calls[0] == ("dataset_info", nls.NLS_DATASET, "main")
    assert sum(1 for c in calls if c[0] == "dataset_info") == 1     # resolved exactly once
    used = {c[-1] for c in calls[1:]}
    assert used == {"e" * 40}, f"every read must be pinned to the resolved SHA, got {calls}"


def test_nls_loader_forwards_a_requested_revision(monkeypatch, tmp_path):
    import nls

    calls = []
    fake_hub, fake_datasets = _fake_hub(tmp_path, calls)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    nls.load_nls(revision="v1.0")
    assert calls[0] == ("dataset_info", nls.NLS_DATASET, "v1.0")


# --- duplicate item ids ---------------------------------------------------------------------

@pytest.mark.parametrize("loader", ["load_nls", "load_gold"])
def test_a_repeated_id_in_the_gold_stops_the_load(monkeypatch, tmp_path, loader):
    """Two items under one id collapse into one row downstream, after the inference is paid for.

    `load_gold` keys its result by id and would come back a row short; `load_nls` keeps a list,
    which then fails the resume index, the carried rows and the file's own `item_count`.
    """
    import nls

    calls = []
    fake_hub, fake_datasets = _fake_hub(tmp_path, calls,
                                        sample_ids=("card-1", "card-2", "card-1"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(SystemExit) as refusal:
        getattr(nls, loader)()

    message = str(refusal.value)
    assert "1 duplicate item id(s): card-1" in message
    assert "card-2" not in message                 # only the id that repeats is named
    assert nls.NLS_DATASET in message


def test_a_repeated_id_in_a_hub_dataset_stops_the_load(monkeypatch):
    """The other loader path: any Hub dataset, not the NLS gold set."""
    rows = [{"id": "card-1", "image": "PIXELS", "target_schema": SCHEMA_STR, "silver_gold": GOLD},
            {"id": "card-1", "image": "PIXELS", "target_schema": SCHEMA_STR, "silver_gold": GOLD}]
    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=lambda *_a, **_kw: rows))
    monkeypatch.setattr(harness, "resolve_dataset_revision", lambda *_a: "d" * 40)
    monkeypatch.setattr(harness, "TYPED_CORE", {"card-1"})

    with pytest.raises(SystemExit) as refusal:
        harness.load_items("org/set", limit=None)

    assert "org/set split train" in str(refusal.value)
    assert "card-1" in str(refusal.value)


def test_a_duplicate_past_the_limit_is_still_refused(monkeypatch):
    """A `--limit 2` smoke test must not pass a dataset the full run will refuse."""
    rows = [{"id": f"card-{n}", "image": "PIXELS", "target_schema": SCHEMA_STR,
             "silver_gold": GOLD} for n in (1, 2, 1)]
    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=lambda *_a, **_kw: rows))
    monkeypatch.setattr(harness, "resolve_dataset_revision", lambda *_a: "d" * 40)
    monkeypatch.setattr(harness, "TYPED_CORE", {"card-1", "card-2"})

    with pytest.raises(SystemExit, match="card-1"):
        harness.load_items("org/set", limit=2)


# --- checkpoint, promotion, resume --------------------------------------------------------------

def items_for(ids):
    """Canned items whose `image` carries the id, so a scripted adapter can answer per item."""
    return [{"id": i, "image": f"PIXELS:{i}", "target_schema": SCHEMA_STR, "gold": GOLD}
            for i in ids]


class ScriptedAdapter:
    """A fake adapter that answers per item and records every call it received.

    `answers` maps an item id to the content to return, or to an exception to raise. Anything not
    listed gets `default`. A LIST is a script for that item, consumed one entry per call, so
    `{"card-1": [ReadTimeout(), ReadTimeout(), PERFECT]}` fails twice and then answers; the last
    entry is reused if the item is called again. Per item rather than a shared counter, so a test
    reads as "this card does that" no matter what the other cards are doing.

    There is no network here by construction: the adapter is the whole model.
    """

    def __init__(self, answers, default=PERFECT):
        self.answers = answers
        self.default = default
        self.calls = []
        self.script_position = {}

    def next_answer(self, item_id):
        answer = self.answers.get(item_id, self.default)
        if not isinstance(answer, list):
            return answer
        index = self.script_position.get(item_id, 0)
        self.script_position[item_id] = min(index + 1, len(answer) - 1)
        return answer[index]

    def __call__(self, image, _schema, spec):
        item_id = image.split(":", 1)[1]
        self.calls.append(item_id)
        answer = self.next_answer(item_id)
        if isinstance(answer, BaseException):
            raise answer
        return harness.endpoint_response(spec, answer, "stop")


def install_scripted_adapter(monkeypatch, answers, default=PERFECT):
    adapter = ScriptedAdapter(answers, default)
    monkeypatch.setitem(harness.ADAPTERS, "fake", adapter)
    return adapter


def read_result(tmp_path, label, name=None):
    return json.loads((tmp_path / "nls" / (name or f"{label}.json")).read_text())


def test_cli_defaults_reach_the_loader(monkeypatch, tmp_path):
    spec = fake_spec("fake-defaults", canned=PERFECT)
    calls = install_fake_run(monkeypatch, tmp_path, spec)
    harness.main(["--models", spec["label"], "--dataset", "nls"])
    assert calls == [("nls", None, "main")]


def test_an_interrupted_run_leaves_a_partial_and_no_result_file(monkeypatch, tmp_path):
    """A KeyboardInterrupt is not an adapter failure: it must not be caught and turned into a
    transport-error row, and the rows already paid for must survive it."""
    items = items_for(["card-1", "card-2", "card-3"])
    spec = fake_spec("fake-interrupted")
    install_scripted_adapter(monkeypatch, {"card-3": KeyboardInterrupt("ctrl-c")})
    install_fake_run(monkeypatch, tmp_path, spec, items)

    with pytest.raises(KeyboardInterrupt):
        harness.main(["--models", spec["label"], "--dataset", "nls"])

    outdir = tmp_path / "nls"
    assert not (outdir / "fake-interrupted.json").exists()
    checkpoint = json.loads((outdir / "fake-interrupted.json.partial").read_text())
    assert [row["id"] for row in checkpoint["items"]] == ["card-1", "card-2"]
    assert checkpoint["run"]["complete"] is False
    # board.py, rescore.py, leaderboard.py and site_nls.py all read `*.json`: a checkpoint is
    # invisible to every one of them until the run promotes it.
    assert sorted(p.name for p in outdir.glob("*.json")) == []


def test_a_completed_run_promotes_the_partial_and_leaves_nothing_else(monkeypatch, tmp_path):
    spec = fake_spec("fake-clean", canned=PERFECT)
    out = run_harness(monkeypatch, tmp_path, spec)
    assert out["run"]["complete"] is True
    # no `.partial`, and no temp file from the atomic writes either
    assert sorted(p.name for p in (tmp_path / "nls").iterdir()) == ["fake-clean.json"]


def test_resume_keeps_ok_rows_and_redoes_transport_errors(monkeypatch, tmp_path):
    items = items_for(["card-1", "card-2"])
    spec = fake_spec("fake-resume")
    install_scripted_adapter(monkeypatch, {"card-2": RuntimeError("gateway timed out")})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    harness.main(["--models", spec["label"], "--dataset", "nls"])

    before = read_result(tmp_path, "fake-resume")
    assert [row["inference_status"] for row in before["items"]] == ["ok", "transport_error"]
    assert before["run"]["transport_errors"] == 1

    second = install_scripted_adapter(monkeypatch, {})
    harness.main(["--models", spec["label"], "--dataset", "nls", "--resume"])
    after = read_result(tmp_path, "fake-resume")

    assert second.calls == ["card-2"]              # the good row was not inferred (or paid for) again
    assert after["items"][0] == before["items"][0]  # kept whole: scores and provenance alike
    redone = after["items"][1]
    assert redone["inference_status"] == "ok"
    assert redone["prediction"] == PERFECT
    assert redone["attempts"] == 2
    assert after["run"]["started_at"] == before["run"]["started_at"]
    assert len(after["run"]["resumed_at"]) == 1
    assert after["run"]["resumed_at"][0].endswith("Z")
    assert after["run"]["transport_errors"] == 0
    assert after["run"]["complete"] is True
    assert not (tmp_path / "nls" / "fake-resume.json.partial").exists()


def test_resume_continues_a_partial_left_by_an_interrupted_run(monkeypatch, tmp_path):
    items = items_for(["card-1", "card-2", "card-3"])
    spec = fake_spec("fake-continue")
    install_scripted_adapter(monkeypatch, {"card-3": KeyboardInterrupt("ctrl-c")})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    with pytest.raises(KeyboardInterrupt):
        harness.main(["--models", spec["label"], "--dataset", "nls"])

    second = install_scripted_adapter(monkeypatch, {})
    harness.main(["--models", spec["label"], "--dataset", "nls", "--resume"])

    assert second.calls == ["card-3"]              # only the row the interrupt cost
    after = read_result(tmp_path, "fake-continue")
    assert [row["id"] for row in after["items"]] == ["card-1", "card-2", "card-3"]
    assert after["run"]["complete"] is True
    assert not (tmp_path / "nls" / "fake-continue.json.partial").exists()


def test_resume_refuses_a_different_dataset_revision(monkeypatch, tmp_path):
    items = items_for(["card-1", "card-2"])
    spec = fake_spec("fake-revision")
    adapter = install_scripted_adapter(monkeypatch, {})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    harness.main(["--models", spec["label"], "--dataset", "nls"])
    adapter.calls.clear()

    install_fake_run(monkeypatch, tmp_path, spec, items, sha="e" * 40)
    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--models", spec["label"], "--dataset", "nls", "--resume"])
    assert "dataset.inference_revision" in str(excinfo.value)
    assert adapter.calls == []                     # refused before a single call was made


def test_resume_refuses_a_reordered_slice(monkeypatch, tmp_path):
    """Same items, different order: the ordered id hash differs, so the rows are not comparable."""
    items = items_for(["card-1", "card-2"])
    spec = fake_spec("fake-reordered")
    adapter = install_scripted_adapter(monkeypatch, {})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    harness.main(["--models", spec["label"], "--dataset", "nls"])
    adapter.calls.clear()

    install_fake_run(monkeypatch, tmp_path, spec, list(reversed(items)))
    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--models", spec["label"], "--dataset", "nls", "--resume"])
    assert "dataset.item_ids_sha256" in str(excinfo.value)
    assert adapter.calls == []


def test_resume_refuses_a_different_model_behind_the_same_label(monkeypatch, tmp_path):
    items = items_for(["card-1"])
    adapter = install_scripted_adapter(monkeypatch, {})
    install_fake_run(monkeypatch, tmp_path, fake_spec("swapped"), items)
    harness.main(["--models", "swapped", "--dataset", "nls"])
    adapter.calls.clear()

    install_fake_run(monkeypatch, tmp_path, fake_spec("swapped", id="fake/other"), items)
    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--models", "swapped", "--dataset", "nls", "--resume"])
    assert "model.id" in str(excinfo.value)
    assert adapter.calls == []


def test_select_unparseable_also_redoes_rows_the_scorer_could_not_parse(monkeypatch, tmp_path):
    items = items_for(["card-1", "card-2"])
    spec = fake_spec("fake-unparseable")
    install_scripted_adapter(monkeypatch, {"card-1": "I am sorry, I cannot read this card."})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    harness.main(["--models", spec["label"], "--dataset", "nls"])
    assert read_result(tmp_path, spec["label"])["items"][0]["schema_valid"] == 0.0

    # the default selection leaves it alone: an unparseable answer is a model result, not a failure
    default_select = install_scripted_adapter(monkeypatch, {})
    harness.main(["--models", spec["label"], "--dataset", "nls", "--resume"])
    assert default_select.calls == []

    targeted = install_scripted_adapter(monkeypatch, {})
    harness.main(["--models", spec["label"], "--dataset", "nls",
                  "--resume", "--select", "unparseable"])
    assert targeted.calls == ["card-1"]
    after = read_result(tmp_path, spec["label"])
    assert after["items"][0]["schema_valid"] == 1.0
    assert after["items"][0]["attempts"] == 2
    assert after["items"][1]["attempts"] == 1      # the row that was fine kept its own history


def test_select_all_redoes_every_row_and_keeps_the_original_start_time(monkeypatch, tmp_path):
    items = items_for(["card-1", "card-2"])
    spec = fake_spec("fake-select-all")
    install_scripted_adapter(monkeypatch, {})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    harness.main(["--models", spec["label"], "--dataset", "nls"])
    before = read_result(tmp_path, spec["label"])

    second = install_scripted_adapter(monkeypatch, {})
    harness.main(["--models", spec["label"], "--dataset", "nls", "--resume", "--select", "all"])
    after = read_result(tmp_path, spec["label"])

    assert second.calls == ["card-1", "card-2"]
    assert [row["attempts"] for row in after["items"]] == [2, 2]
    assert after["run"]["started_at"] == before["run"]["started_at"]
    assert len(after["run"]["resumed_at"]) == 1


def test_limit_writes_a_noncanonical_file(monkeypatch, tmp_path):
    items = items_for(["card-1", "card-2", "card-3"])
    spec = fake_spec("fake-smoke", canned=PERFECT)
    install_fake_run(monkeypatch, tmp_path, spec, items)
    harness.main(["--models", spec["label"], "--dataset", "nls", "--limit", "2"])

    outdir = tmp_path / "nls"
    assert sorted(p.name for p in outdir.iterdir()) == ["fake-smoke.limit2.json"]
    smoke = read_result(tmp_path, spec["label"], name="fake-smoke.limit2.json")
    assert smoke["dataset"]["item_count"] == 2
    assert [row["id"] for row in smoke["items"]] == ["card-1", "card-2"]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_limit_below_one_is_refused(monkeypatch, capsys, value):
    """`--limit 0` is falsy everywhere it is read, so it would run the whole slice and file the
    result under a smoke-test name nothing publishes from — a full run, paid for and discarded."""
    registry(monkeypatch, fake_spec("fake-guard"))
    refuse_to_load(monkeypatch)

    with pytest.raises(SystemExit) as refusal:
        harness.main(["--models", "fake-guard", "--dataset", "nls", "--limit", value])

    assert refusal.value.code == 2                # argparse usage error, not a clean exit
    assert ("--limit must be a positive integer; use no --limit for the full run"
            in capsys.readouterr().err)


def test_a_limit_of_one_is_allowed():
    assert harness.positive_limit("1") == 1


def test_an_interrupted_resume_still_holds_the_rows_the_file_already_had(monkeypatch, tmp_path):
    """The riskiest sequence: a finished file, a `--select all` resume that dies partway through.

    The checkpoint it leaves must be a superset of the file it read — not only the rows redone
    so far — or the next resume would infer (and pay for) rows that were already answered.
    """
    items = items_for(["card-1", "card-2", "card-3"])
    spec = fake_spec("fake-superset")
    install_scripted_adapter(monkeypatch, {})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    harness.main(["--models", spec["label"], "--dataset", "nls"])

    install_scripted_adapter(monkeypatch, {"card-2": KeyboardInterrupt("ctrl-c")})
    with pytest.raises(KeyboardInterrupt):
        harness.main(["--models", spec["label"], "--dataset", "nls",
                      "--resume", "--select", "all"])

    outdir = tmp_path / "nls"
    checkpoint = json.loads((outdir / "fake-superset.json.partial").read_text())
    assert [row["id"] for row in checkpoint["items"]] == ["card-1", "card-2", "card-3"]
    assert [row["attempts"] for row in checkpoint["items"]] == [2, 1, 1]
    # and the finished file it was reading is untouched, still complete
    assert read_result(tmp_path, spec["label"])["run"]["complete"] is True

    third = install_scripted_adapter(monkeypatch, {})
    harness.main(["--models", spec["label"], "--dataset", "nls", "--resume", "--select", "all"])
    assert third.calls == ["card-1", "card-2", "card-3"]     # --select all means all, again
    assert len(read_result(tmp_path, spec["label"])["run"]["resumed_at"]) == 2


# --- failure classification ---------------------------------------------------------------------

class FakeResponse:
    """The shape every SDK attaches to an HTTP error: an object carrying a status code."""

    def __init__(self, status_code):
        self.status_code = status_code


def sdk_error(name, *, status_code=None, response_status=None, bases=(Exception,)):
    """An exception shaped like a real SDK's, without importing the SDK.

    `classify_failure` reads status codes, class names and two builtins — never SDK classes, which
    may not be installed at all (see test_import_needs_no_provider_sdk). A stand-in with the same
    name, the same bases and the same attributes is therefore the same input to it. The real
    classes are checked separately, wherever they happen to be installed.
    """
    exception = type(name, bases, {})(name)
    if status_code is not None:
        exception.status_code = status_code
    if response_status is not None:
        exception.response = FakeResponse(response_status)
    return exception


@pytest.mark.parametrize("exc, expected", [
    # openai carries the status on the exception itself
    (sdk_error("RateLimitError", status_code=429), "transient"),
    (sdk_error("InternalServerError", status_code=500), "transient"),
    (sdk_error("InternalServerError", status_code=503), "transient"),
    (sdk_error("InternalServerError", status_code=529), "transient"),
    (sdk_error("APIStatusError", status_code=408), "transient"),
    (sdk_error("BadRequestError", status_code=400), "permanent"),
    (sdk_error("AuthenticationError", status_code=401), "permanent"),
    (sdk_error("PermissionDeniedError", status_code=403), "permanent"),
    (sdk_error("NotFoundError", status_code=404), "permanent"),
    (sdk_error("UnprocessableEntityError", status_code=422), "permanent"),
    # a 4xx nothing here has a name for is still the server saying no to this request
    (sdk_error("APIStatusError", status_code=418), "permanent"),
    (sdk_error("APIStatusError", status_code=451), "permanent"),
    # anything outside 4xx and the retry list: no rule covers it, and "the endpoint refused" is
    # the wrong thing to record about a 200 that arrived as an exception
    (sdk_error("APIStatusError", status_code=200), "unclassified"),
    (sdk_error("APIStatusError", status_code=204), "unclassified"),
    (sdk_error("APIStatusError", status_code=302), "unclassified"),
    (sdk_error("HfHubHTTPError", response_status=507, bases=(OSError,)), "unclassified"),
    (sdk_error("APITimeoutError"), "transient"),
    (sdk_error("APIConnectionError"), "transient"),
    # huggingface_hub keeps it on an attached response, and one class covers 429 and every 5xx
    (sdk_error("HfHubHTTPError", response_status=429, bases=(OSError,)), "transient"),
    (sdk_error("HfHubHTTPError", response_status=504, bases=(OSError,)), "transient"),
    (sdk_error("HfHubHTTPError", response_status=401, bases=(OSError,)), "permanent"),
    (sdk_error("RepositoryNotFoundError", response_status=404, bases=(OSError,)), "permanent"),
    (sdk_error("InferenceTimeoutError", bases=(TimeoutError,)), "transient"),
    # a ConnectionError subclass that means "you told me not to use the network"
    (sdk_error("OfflineModeIsEnabled", bases=(ConnectionError,)), "permanent"),
    # gradio_client says it in the class name and nowhere else
    (sdk_error("QueueError"), "transient"),
    (sdk_error("TooManyRequestsError"), "transient"),
    (sdk_error("AppError", bases=(ValueError,)), "permanent"),
    (sdk_error("ValidationError", bases=(ValueError,)), "permanent"),
    # httpx transport errors: not an OSError, not a builtin TimeoutError — only the name says so
    (sdk_error("ReadTimeout"), "transient"),
    (sdk_error("ConnectError"), "transient"),
    (sdk_error("RemoteProtocolError"), "transient"),
    (sdk_error("ConnectionError", bases=(OSError,)), "transient"),      # requests shadows the builtin
    (TimeoutError("timed out"), "transient"),
    (ConnectionError("connection reset by peer"), "transient"),
    # nothing recognisable: a bug in the harness is not worth three calls, and is not the
    # endpoint saying no either
    (RuntimeError("gateway timed out"), "unclassified"),
    (ValueError("could not fetch config"), "unclassified"),
    (harness.EmptyResponseError("no choices"), "unclassified"),
])
def test_classify_failure(exc, expected):
    assert harness.classify_failure(exc) == expected


def test_a_status_code_decides_on_its_own(monkeypatch):
    """A 404 must not be rescued by a later rule, and 5xx must not be sunk by one.

    HfHubHTTPError subclasses OSError and huggingface_hub's BadRequestError also subclasses
    ValueError, so an implementation that looked at names or builtins first would get both wrong.
    """
    monkeypatch.setattr(harness, "PERMANENT_EXCEPTION_NAMES", frozenset({"ReadTimeout"}))
    monkeypatch.setattr(harness, "TRANSIENT_EXCEPTION_NAMES", frozenset({"NotFoundError"}))
    assert harness.classify_failure(sdk_error("NotFoundError", response_status=404)) == "permanent"
    assert harness.classify_failure(sdk_error("ReadTimeout", response_status=503)) == "transient"
    # and a status outside both sets stops there too, rather than falling through to the names
    assert harness.classify_failure(sdk_error("ReadTimeout", response_status=200)) == "unclassified"


def test_classification_reads_the_real_openai_exceptions():
    """Skipped where the SDK is absent (CI runs the lean env), run wherever it is installed."""
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")

    def status_error(cls, status):
        request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
        return cls("boom", response=httpx.Response(status, request=request), body=None)

    assert harness.classify_failure(status_error(openai.InternalServerError, 503)) == "transient"
    assert harness.classify_failure(status_error(openai.RateLimitError, 429)) == "transient"
    assert harness.classify_failure(status_error(openai.APIStatusError, 408)) == "transient"
    assert harness.classify_failure(status_error(openai.AuthenticationError, 401)) == "permanent"
    assert harness.classify_failure(status_error(openai.NotFoundError, 404)) == "permanent"
    assert harness.classify_failure(status_error(openai.BadRequestError, 400)) == "permanent"

    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    assert harness.classify_failure(openai.APITimeoutError(request=request)) == "transient"
    assert harness.classify_failure(
        openai.APIConnectionError(request=request)) == "transient"


def test_classification_reads_the_real_huggingface_hub_exceptions():
    """Skipped where the SDK is absent (CI runs the lean env), run wherever it is installed."""
    errors = pytest.importorskip("huggingface_hub.errors")
    httpx = pytest.importorskip("httpx")

    def hub_error(cls, status):
        request = httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions")
        return cls("boom", response=httpx.Response(status, request=request))

    # chat_completion turns 429 and every 5xx into a bare HfHubHTTPError: only the status tells
    # them apart from a 404, and the class is an OSError either way.
    assert harness.classify_failure(hub_error(errors.HfHubHTTPError, 429)) == "transient"
    assert harness.classify_failure(hub_error(errors.HfHubHTTPError, 503)) == "transient"
    assert harness.classify_failure(hub_error(errors.HfHubHTTPError, 500)) == "transient"
    assert harness.classify_failure(hub_error(errors.HfHubHTTPError, 401)) == "permanent"
    assert harness.classify_failure(hub_error(errors.RepositoryNotFoundError, 404)) == "permanent"
    assert harness.classify_failure(hub_error(errors.BadRequestError, 400)) == "permanent"
    assert harness.classify_failure(errors.InferenceTimeoutError("slow")) == "transient"


def test_classification_reads_the_real_httpx_transport_errors():
    httpx = pytest.importorskip("httpx")

    request = httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions")
    assert harness.classify_failure(httpx.ReadTimeout("slow", request=request)) == "transient"
    assert harness.classify_failure(httpx.ConnectError("refused", request=request)) == "transient"
    # the reason the name check exists: neither of those is a builtin the harness could catch
    assert not isinstance(httpx.ConnectError("refused", request=request), OSError)
    assert not isinstance(httpx.ReadTimeout("slow", request=request), TimeoutError)


# --- backoff --------------------------------------------------------------------------------------

def test_backoff_doubles_then_stops_growing():
    policy = harness.RetryPolicy(max_attempts=10, backoff_s=2.0, max_wait_s=30.0)
    waits = [policy.wait_after(attempt) for attempt in range(1, 7)]

    # 2, 4, 8, 16, then capped at 30 — each with up to RETRY_JITTER on top, never below the step
    for wait, step in zip(waits, [2.0, 4.0, 8.0, 16.0, 30.0, 30.0], strict=True):
        assert step <= wait <= step * (1 + harness.RETRY_JITTER)


def test_max_attempts_below_one_is_refused(monkeypatch):
    spec = fake_spec("fake-guard", canned=PERFECT)
    install_fake_run(monkeypatch, tmp_path=pathlib.Path("/nonexistent"), spec=spec)
    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--models", spec["label"], "--dataset", "nls", "--max-attempts", "0"])
    assert "--max-attempts" in str(excinfo.value)


# --- bounded retry ----------------------------------------------------------------------------

def record_sleeps(monkeypatch):
    """Replace the retry sleep with a recorder; -> the durations it was asked to sleep for."""
    slept = []
    monkeypatch.setattr(harness, "sleep", slept.append)
    return slept


def transient():
    """A stand-in for the httpx read timeout the router raises when it stops answering."""
    return sdk_error("ReadTimeout")


def permanent_401():
    """A stand-in for the router's 401: a wrong token does not get better by asking again."""
    return sdk_error("HfHubHTTPError", response_status=401, bases=(OSError,))


def test_a_transient_failure_is_retried_until_the_call_succeeds(monkeypatch, tmp_path, capsys):
    items = items_for(["card-1"])
    spec = fake_spec("fake-flaky")
    adapter = install_scripted_adapter(
        monkeypatch, {"card-1": [transient(), transient(), PERFECT]})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    slept = record_sleeps(monkeypatch)

    harness.main(["--models", spec["label"], "--dataset", "nls"])

    row = read_result(tmp_path, spec["label"])["items"][0]
    assert adapter.calls == ["card-1", "card-1", "card-1"]
    assert row["inference_status"] == "ok"
    assert row["attempts"] == 3
    assert row["prediction"] == PERFECT
    assert "failure_class" not in row          # nothing failed in the end
    assert "error" not in row
    # the finish reason comes from the attempt that answered, not from the ones that did not
    assert row["provenance"]["finish_reason"] == "stop"

    assert len(slept) == 2                     # two failures, two waits, then the answer
    assert 2.0 <= slept[0] <= 2.2
    assert 4.0 <= slept[1] <= 4.4
    assert row["provenance"]["retry_wait_s"] == round(sum(slept), 3)

    printed = capsys.readouterr().out
    assert "fake-flaky card-1 attempt 1/3: ReadTimeout — retrying in 2." in printed
    assert "fake-flaky card-1 attempt 2/3: ReadTimeout — retrying in 4." in printed


def test_a_permanent_failure_is_never_retried(monkeypatch, tmp_path):
    items = items_for(["card-1"])
    spec = fake_spec("fake-401")
    adapter = install_scripted_adapter(monkeypatch, {"card-1": permanent_401()})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    slept = record_sleeps(monkeypatch)

    harness.main(["--models", spec["label"], "--dataset", "nls"])

    row = read_result(tmp_path, spec["label"])["items"][0]
    assert adapter.calls == ["card-1"]         # one call, no second chance
    assert slept == []
    assert row["inference_status"] == "transport_error"
    assert row["failure_class"] == "permanent"
    assert row["attempts"] == 1
    assert row["prediction"] is None
    assert row["provenance"]["retry_wait_s"] == 0.0


def test_a_transient_failure_gives_up_after_max_attempts(monkeypatch, tmp_path):
    items = items_for(["card-1"])
    spec = fake_spec("fake-down-hard")
    adapter = install_scripted_adapter(monkeypatch, {"card-1": transient()})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    slept = record_sleeps(monkeypatch)

    harness.main(["--models", spec["label"], "--dataset", "nls"])

    row = read_result(tmp_path, spec["label"])["items"][0]
    assert adapter.calls == ["card-1"] * 3
    assert row["inference_status"] == "transport_error"
    assert row["failure_class"] == "transient"
    assert row["attempts"] == 3
    assert row["error"].startswith("ReadTimeout: ")
    assert len(slept) == 2                     # the last failure is not followed by a wait
    assert row["provenance"]["retry_wait_s"] == round(sum(slept), 3)


def test_max_attempts_one_means_one_call(monkeypatch, tmp_path):
    items = items_for(["card-1"])
    spec = fake_spec("fake-no-retry")
    adapter = install_scripted_adapter(monkeypatch, {"card-1": transient()})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    slept = record_sleeps(monkeypatch)

    harness.main(["--models", spec["label"], "--dataset", "nls", "--max-attempts", "1"])

    row = read_result(tmp_path, spec["label"])["items"][0]
    assert adapter.calls == ["card-1"]
    assert slept == []
    assert row["attempts"] == 1
    assert row["failure_class"] == "transient"     # worth retrying, but no attempts were allowed


def test_an_unparseable_answer_is_never_retried(monkeypatch, tmp_path):
    """The model answered; it just answered badly. That is a finding to publish, not a failure
    to spend three requests on."""
    items = items_for(["card-1"])
    spec = fake_spec("fake-babbler")
    adapter = install_scripted_adapter(monkeypatch, {"card-1": "I cannot read this card."})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    slept = record_sleeps(monkeypatch)

    harness.main(["--models", spec["label"], "--dataset", "nls"])

    row = read_result(tmp_path, spec["label"])["items"][0]
    assert adapter.calls == ["card-1"]
    assert slept == []
    assert row["inference_status"] == "ok"
    assert row["attempts"] == 1
    assert row["schema_valid"] == 0.0


def test_a_resume_adds_this_runs_attempts_to_what_earlier_runs_spent(monkeypatch, tmp_path):
    """`attempts` is every call ever made for a row, not the calls the latest run made."""
    items = items_for(["card-1"])
    spec = fake_spec("fake-two-sessions")
    install_scripted_adapter(monkeypatch, {"card-1": transient()})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    record_sleeps(monkeypatch)
    harness.main(["--models", spec["label"], "--dataset", "nls"])

    first = read_result(tmp_path, spec["label"])["items"][0]
    assert first["attempts"] == 3 and first["failure_class"] == "transient"

    second = install_scripted_adapter(monkeypatch, {"card-1": [transient(), PERFECT]})
    harness.main(["--models", spec["label"], "--dataset", "nls", "--resume"])

    row = read_result(tmp_path, spec["label"])["items"][0]
    assert second.calls == ["card-1", "card-1"]
    assert row["inference_status"] == "ok"
    assert row["attempts"] == 5                # 3 spent before, 2 spent now
    assert "failure_class" not in row          # the row is no longer a failure


def test_an_unclassified_failure_is_never_retried_either(monkeypatch, tmp_path):
    """Not worth three calls, and not the endpoint's fault: an unrecognised exception is a bug."""
    items = items_for(["card-1"])
    spec = fake_spec("fake-weird")
    adapter = install_scripted_adapter(monkeypatch, {"card-1": ValueError("could not fetch config")})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    slept = record_sleeps(monkeypatch)

    harness.main(["--models", spec["label"], "--dataset", "nls"])

    row = read_result(tmp_path, spec["label"])["items"][0]
    assert adapter.calls == ["card-1"]
    assert slept == []
    assert row["failure_class"] == "unclassified"
    assert row["error"].startswith("ValueError: could not fetch config")


def test_the_run_summary_tells_the_three_failure_classes_apart(monkeypatch, tmp_path, capsys):
    """`--resume` is the right advice for a 504, the wrong advice for a 401, and no advice at all
    for an exception nothing here recognises — which is a bug to read, not a call to repeat."""
    items = items_for(["card-1", "card-2", "card-3"])
    spec = fake_spec("fake-mixed")
    install_scripted_adapter(monkeypatch, {"card-1": transient(), "card-2": permanent_401(),
                                           "card-3": ValueError("could not fetch config")})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    record_sleeps(monkeypatch)

    harness.main(["--models", spec["label"], "--dataset", "nls"])

    printed = capsys.readouterr().out
    assert "fake-mixed: 3 transport error(s): " in printed
    assert "1 permanent (auth, bad request, not found)" in printed
    assert ("1 unclassified (an exception the harness does not recognise — likely a harness or "
            "response-shape bug, see error)") in printed
    assert "1 transient (ran out of attempts)" in printed


def test_a_class_with_no_rows_is_left_out_of_the_summary(monkeypatch, tmp_path, capsys):
    items = items_for(["card-1"])
    spec = fake_spec("fake-401-only")
    install_scripted_adapter(monkeypatch, {"card-1": permanent_401()})
    install_fake_run(monkeypatch, tmp_path, spec, items)

    harness.main(["--models", spec["label"], "--dataset", "nls"])

    printed = capsys.readouterr().out
    assert "fake-401-only: 1 transport error(s): 1 permanent" in printed
    assert "unclassified" not in printed
    assert "ran out of attempts" not in printed


def test_a_row_written_before_classification_existed_counts_under_no_class():
    """A legacy transport-error row is unknown, not known to be anything."""
    rows = [{"inference_status": "transport_error"},
            {"inference_status": "transport_error", "failure_class": "permanent"},
            {"inference_status": "ok", "failure_class": "permanent"}]
    assert harness.count_by_failure_class(rows) == {"permanent": 1}


# --- model selection: --models is required, disabled entries stay out --------------------------

def registry(monkeypatch, *specs):
    monkeypatch.setattr(harness, "MODELS", list(specs))


def test_models_is_required_and_a_bare_run_lists_the_labels(monkeypatch):
    registry(monkeypatch, fake_spec("alpha"), fake_spec("beta"))
    refuse_to_load(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--dataset", "nls"])

    message = str(excinfo.value)
    assert "--models is required" in message
    assert "alpha" in message and "beta" in message
    # SystemExit carrying a string is exit status 1, not 0 — a bare run must not look like success.
    assert excinfo.value.code != 0


def test_models_all_runs_every_enabled_entry_and_skips_the_disabled_ones(monkeypatch):
    registry(monkeypatch,
             fake_spec("kept-1"),
             fake_spec("never-run", enabled=False),
             fake_spec("kept-2"))

    args = types.SimpleNamespace(models="all", base_url=None, provider=None, no_thinking=False)
    specs = harness.resolve_run_specs(args, environ={})

    assert [spec["label"] for spec in specs] == ["kept-1", "kept-2"]


def test_an_explicitly_named_disabled_model_is_refused(monkeypatch):
    registry(monkeypatch, fake_spec("kept"), fake_spec("never-run", enabled=False))
    refuse_to_load(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--models", "never-run", "--dataset", "nls"])

    message = str(excinfo.value)
    assert "never-run" in message
    assert "disabled" in message
    assert "models.py" in message      # says where to change it, not just that it refused


def test_the_label_menu_marks_which_entries_are_disabled(monkeypatch):
    registry(monkeypatch, fake_spec("kept"), fake_spec("never-run", enabled=False))
    menu = harness.label_menu()
    assert menu == "kept, never-run (disabled)"


def test_the_registry_carries_no_endpoint_urls():
    """The tripwire for keeping endpoints out of the registry.

    Two dead `https://<job-id>--8000.hf.jobs/v1` URLs sat in this file until a run against them
    failed on every item. An endpoint belongs to a run, not to a model, so the key may not come
    back — including inside a commented-out example, which is where it would come back FIRST.
    """
    import models

    for spec in models.MODELS:
        assert "base_url" not in spec, f"{spec['label']} carries an endpoint in the registry"

    source = (GLAM_BENCH / "models.py").read_text()
    assert "base_url" not in source, "models.py mentions base_url; endpoints come from the CLI/env"


# --- endpoint resolution ------------------------------------------------------------------------

def openai_spec(label, **extra):
    spec = {"label": label, "kind": "openai", "id": "org/model", "params": "1B",
            "cost": "self-host", "api_key_env": "HF_TOKEN"}
    spec.update(extra)
    return spec


def router_spec(label, **extra):
    spec = {"label": label, "kind": "router_vlm", "id": "org/model", "provider": None,
            "params": "1B", "cost": "router"}
    spec.update(extra)
    return spec


def install_adapter_recorder(monkeypatch, kind="openai"):
    """Replace one adapter with a recorder; -> the resolved specs it was called with.

    Nothing here reaches an SDK or a socket: the point is to see exactly which spec the adapter
    would have used, which is where an endpoint, a key env var and the request options all land.
    """
    seen = []

    def recorder(_img, _schema, spec):
        seen.append(spec)
        return harness.endpoint_response(spec, PERFECT, "stop")

    monkeypatch.setitem(harness.ADAPTERS, kind, recorder)
    return seen


def test_an_openai_model_without_an_endpoint_fails_before_the_dataset_loads(monkeypatch):
    registry(monkeypatch, openai_spec("GLM-OCR"))
    refuse_to_load(monkeypatch)
    monkeypatch.delenv("GLAM_BASE_URL_GLM_OCR", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--models", "GLM-OCR", "--dataset", "nls"])

    message = str(excinfo.value)
    assert "GLM-OCR needs an endpoint" in message
    assert "--base-url GLM-OCR=https://.../v1" in message
    assert "$GLAM_BASE_URL_GLM_OCR" in message


def test_the_endpoint_env_var_name_normalises_the_label():
    assert harness.endpoint_env_var("GLM-OCR") == "GLAM_BASE_URL_GLM_OCR"
    assert (harness.endpoint_env_var("LFM2.5-VL-1.6B-Extract")
            == "GLAM_BASE_URL_LFM2_5_VL_1_6B_EXTRACT")
    assert harness.endpoint_env_var("Qwen3.6-35B-A3B") == "GLAM_BASE_URL_QWEN3_6_35B_A3B"


def test_an_endpoint_from_the_environment_alone_is_enough(monkeypatch, tmp_path):
    spec = openai_spec("GLM-OCR")
    seen = install_adapter_recorder(monkeypatch)
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))
    monkeypatch.setenv("GLAM_BASE_URL_GLM_OCR", "https://abc--8000.hf.jobs/v1")

    harness.main(["--models", "GLM-OCR", "--dataset", "nls"])

    assert seen[0]["base_url"] == "https://abc--8000.hf.jobs/v1"
    row = read_result(tmp_path, "GLM-OCR")["items"][0]
    assert row["provenance"]["endpoint"] == "https://abc--8000.hf.jobs/v1"
    assert row["provenance"]["endpoint_attested"] is True


def test_a_cli_endpoint_beats_the_environment(monkeypatch, tmp_path):
    """An exported variable from an earlier serve must not quietly win over the URL just typed."""
    spec = openai_spec("GLM-OCR")
    seen = install_adapter_recorder(monkeypatch)
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))
    monkeypatch.setenv("GLAM_BASE_URL_GLM_OCR", "https://stale--8000.hf.jobs/v1")

    harness.main(["--models", "GLM-OCR", "--dataset", "nls",
                  "--base-url", "GLM-OCR=https://fresh--8000.hf.jobs/v1"])

    assert seen[0]["base_url"] == "https://fresh--8000.hf.jobs/v1"


def test_an_endpoint_url_never_reaches_the_published_model_block(monkeypatch, tmp_path):
    """`model` is what gets published. The endpoint is a run detail and lives per row."""
    spec = openai_spec("GLM-OCR")
    install_adapter_recorder(monkeypatch)
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "GLM-OCR", "--dataset", "nls",
                  "--base-url", "GLM-OCR=https://abc--8000.hf.jobs/v1"])

    out = read_result(tmp_path, "GLM-OCR")
    assert "base_url" not in out["model"]
    assert "api_key_env" not in out["model"]


@pytest.mark.parametrize("url, because", [
    ("http://abc--8000.hf.jobs/v1", "http:// off localhost sends the token in clear"),
    ("https://abc--8000.hf.jobs", "no /v1: every request would 404"),
    ("https://abc--8000.hf.jobs/v1/chat/completions", "a route, not the API root"),
    ("https://bob:hf_secret@abc--8000.hf.jobs/v1", "credentials in the URL"),
    ("https://abc--8000.hf.jobs/v1?api_key=hf_secret", "a query string, credential-shaped"),
    ("https://abc--8000.hf.jobs/v1#tok", "a fragment"),
    ("ftp://abc/v1", "not http or https"),
    ("abc--8000.hf.jobs/v1", "no scheme, so no host either"),
])
def test_a_bad_endpoint_url_is_refused(url, because):
    with pytest.raises(SystemExit) as excinfo:
        harness.validate_base_url(url, "--base-url X")
    assert "--base-url X" in str(excinfo.value), because


@pytest.mark.parametrize("url, expected", [
    ("https://abc--8000.hf.jobs/v1", "https://abc--8000.hf.jobs/v1"),
    ("https://abc--8000.hf.jobs/v1/", "https://abc--8000.hf.jobs/v1"),      # trailing slash trimmed
    ("http://localhost:8000/v1", "http://localhost:8000/v1"),               # loopback may be plain
    ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1"),
    ("https://api.example.test/openai/v1", "https://api.example.test/openai/v1"),
])
def test_a_good_endpoint_url_is_accepted_and_normalised(url, expected):
    assert harness.validate_base_url(url, "--base-url X") == expected


def test_a_space_model_cannot_be_pointed_at_an_openai_endpoint(monkeypatch):
    registry(monkeypatch, {"label": "NuExtract-3", "kind": "nuextract_space",
                           "id": "numind/NuExtract3", "params": "4B", "cost": "self-host"})
    refuse_to_load(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--models", "NuExtract-3", "--dataset", "nls",
                      "--base-url", "NuExtract-3=https://abc--8000.hf.jobs/v1"])
    assert "cannot be pointed at an OpenAI-compatible endpoint" in str(excinfo.value)


def test_an_endpoint_for_a_model_this_run_is_not_running_is_refused(monkeypatch):
    """The typo that would otherwise be silent: the run looks fine and measures the router."""
    registry(monkeypatch, router_spec("Qwen3.6-27B"))
    refuse_to_load(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls",
                      "--base-url", "Qwen3.6-27b=https://abc--8000.hf.jobs/v1"])
    assert "not running" in str(excinfo.value)
    assert "Qwen3.6-27b" in str(excinfo.value)


def test_the_same_label_given_two_endpoints_is_refused():
    with pytest.raises(SystemExit) as excinfo:
        harness.endpoint_overrides_from_cli(["X=https://a/v1", "X=https://b/v1"])
    assert "given twice" in str(excinfo.value)


def test_a_malformed_base_url_switch_is_refused():
    with pytest.raises(SystemExit) as excinfo:
        harness.endpoint_overrides_from_cli(["https://abc/v1"])
    assert "LABEL=VALUE" in str(excinfo.value)


# --- running a routed model against your own serve ----------------------------------------------

def test_a_router_model_with_an_endpoint_runs_through_the_openai_adapter(monkeypatch, tmp_path):
    """The August 2026 rerun shape: a normally-routed model, answered by a serve we pinned.

    The row has to say so. `router:auto` on a row that never touched the router would be a lie,
    and an unattested endpoint on one that was pinned would throw away the only provenance the
    pinned serve buys.
    """
    spec = router_spec("Qwen3.6-27B")
    seen = install_adapter_recorder(monkeypatch, kind="openai")
    router_calls = install_adapter_recorder(monkeypatch, kind="router_vlm")
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls",
                  "--base-url", "Qwen3.6-27B=https://abc--8000.hf.jobs/v1"])

    assert router_calls == []                       # the router was not used at all
    assert seen[0]["base_url"] == "https://abc--8000.hf.jobs/v1"
    # a Jobs serve takes the HF token as its api key; without this the adapter would send "EMPTY"
    assert seen[0]["api_key_env"] == "HF_TOKEN"

    provenance = read_result(tmp_path, "Qwen3.6-27B")["items"][0]["provenance"]
    assert provenance["served_by"] == "openai-compatible"
    assert provenance["endpoint"] == "https://abc--8000.hf.jobs/v1"
    assert provenance["endpoint_attested"] is True

    out = read_result(tmp_path, "Qwen3.6-27B")
    assert out["model"]["kind"] == "router_vlm"      # still the same model, differently served


def test_the_registry_entry_is_not_mutated_by_a_run(monkeypatch, tmp_path):
    """Resolution copies. A spec dict that picked up this run's endpoint would carry it into the
    next run in the same process — which is how the tests, and any future batch script, work."""
    spec = router_spec("Qwen3.6-27B")
    install_adapter_recorder(monkeypatch)
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls", "--no-thinking",
                  "--base-url", "Qwen3.6-27B=https://abc--8000.hf.jobs/v1"])

    assert "base_url" not in spec
    assert "adapter" not in spec
    assert "request_options" not in spec


# --- explicit router providers ------------------------------------------------------------------

def test_an_unpinned_router_provider_stays_unattested(monkeypatch, tmp_path):
    spec = router_spec("Qwen3.6-27B")
    install_adapter_recorder(monkeypatch, kind="router_vlm")
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls"])

    provenance = read_result(tmp_path, "Qwen3.6-27B")["items"][0]["provenance"]
    assert provenance["endpoint"] == "router:auto"
    assert provenance["endpoint_attested"] is False


def test_a_pinned_provider_is_recorded_and_attested(monkeypatch, tmp_path):
    spec = router_spec("Qwen3.6-27B")
    seen = install_adapter_recorder(monkeypatch, kind="router_vlm")
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls",
                  "--provider", "Qwen3.6-27B=nebius"])

    assert seen[0]["provider"] == "nebius"
    out = read_result(tmp_path, "Qwen3.6-27B")
    assert out["model"]["provider"] == "nebius"       # published: which provider these scores are
    provenance = out["items"][0]["provenance"]
    assert provenance["endpoint"] == "router:nebius"
    assert provenance["endpoint_attested"] is True


def test_a_provider_for_a_non_router_model_is_refused(monkeypatch):
    registry(monkeypatch, openai_spec("GLM-OCR"))
    refuse_to_load(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--models", "GLM-OCR", "--dataset", "nls",
                      "--base-url", "GLM-OCR=https://abc--8000.hf.jobs/v1",
                      "--provider", "GLM-OCR=nebius"])
    assert "only a router_vlm model has a provider" in str(excinfo.value)


def test_an_endpoint_and_a_provider_together_are_refused(monkeypatch):
    registry(monkeypatch, router_spec("Qwen3.6-27B"))
    refuse_to_load(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls",
                      "--base-url", "Qwen3.6-27B=https://abc--8000.hf.jobs/v1",
                      "--provider", "Qwen3.6-27B=nebius"])
    assert "no provider to choose" in str(excinfo.value)


# --- request options in provenance ---------------------------------------------------------------

def test_no_thinking_lands_in_the_row_provenance(monkeypatch, tmp_path):
    spec = router_spec("Qwen3.6-27B")
    seen = install_adapter_recorder(monkeypatch, kind="router_vlm")
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls", "--no-thinking"])

    assert seen[0]["request_options"] == {"chat_template_kwargs": {"enable_thinking": False}}
    provenance = read_result(tmp_path, "Qwen3.6-27B")["items"][0]["provenance"]
    assert provenance["request_options"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert provenance["thinking_disabled"] is True


def test_a_transport_error_row_records_the_request_options_too(monkeypatch, tmp_path):
    """A failed call still asked for something. The row that records the failure records that."""
    spec = router_spec("Qwen3.6-27B")
    monkeypatch.setitem(harness.ADAPTERS, "router_vlm",
                        ScriptedAdapter({"card-1": RuntimeError("boom")}))
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls", "--no-thinking"])

    row = read_result(tmp_path, "Qwen3.6-27B")["items"][0]
    assert row["inference_status"] == "transport_error"
    assert row["provenance"]["thinking_disabled"] is True


def test_a_spec_request_option_keeps_its_other_chat_template_kwargs(monkeypatch, tmp_path):
    spec = router_spec("Qwen3.6-27B",
                       request_options={"chat_template_kwargs": {"mode": "markdown"}})
    seen = install_adapter_recorder(monkeypatch, kind="router_vlm")
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls", "--no-thinking"])

    assert seen[0]["request_options"] == {
        "chat_template_kwargs": {"mode": "markdown", "enable_thinking": False}}


def test_no_thinking_is_reported_not_silently_dropped_for_a_space(monkeypatch, tmp_path, capsys):
    """`--models all --no-thinking` includes a Space, whose adapter has no request body. That is
    a note, not a refusal — and the rows say plainly that nothing was asked for."""
    spec = fake_spec("NuExtract-3", kind="nuextract_space", canned=PERFECT)
    monkeypatch.setitem(harness.ADAPTERS, "nuextract_space", ScriptedAdapter({}))
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "NuExtract-3", "--dataset", "nls", "--no-thinking"])

    assert "--no-thinking does not apply to NuExtract-3" in capsys.readouterr().out
    provenance = read_result(tmp_path, "NuExtract-3")["items"][0]["provenance"]
    assert provenance["request_options"] == {}
    assert provenance["thinking_disabled"] is False


def test_request_options_a_spec_cannot_send_are_refused(monkeypatch):
    """The one case that IS a refusal: options written into the spec that would be recorded as
    sent and never sent. A registry authoring bug, caught before the run."""
    registry(monkeypatch, {"label": "NuExtract-3", "kind": "nuextract_space",
                           "id": "numind/NuExtract3", "params": "4B", "cost": "self-host",
                           "request_options": {"chat_template_kwargs": {"enable_thinking": False}}})
    refuse_to_load(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--models", "NuExtract-3", "--dataset", "nls"])
    assert "cannot send request_options" in str(excinfo.value)


@pytest.mark.parametrize("options, expected", [
    ({}, False),
    ({"chat_template_kwargs": {"enable_thinking": False}}, True),
    ({"chat_template_kwargs": {"enable_thinking": True}}, False),
    ({"chat_template_kwargs": {"mode": "markdown"}}, False),
    ({"guided_json": {}}, False),
    ({"chat_template_kwargs": "nonsense"}, False),
])
def test_thinking_disabled_is_read_back_out_of_the_request_options(options, expected):
    assert harness.thinking_disabled(options) is expected


# --- what the openai adapter actually sends -------------------------------------------------------

def stub_image_encoding(monkeypatch):
    """Let the canned `"PIXELS:<id>"` items through `_vlm_messages`.

    `run_openai` base64-encodes the image before it gets anywhere near the request options, and
    only Pillow can encode a real one. Stubbing the one JPEG call keeps the rest of the adapter
    body — the client construction, the message shape, the extra_body — genuinely under test, and
    keeps these tests out of the lean CI env's skip list.
    """
    monkeypatch.setattr(harness, "_png_bytes", lambda _img: b"jpeg-bytes")


def install_fake_openai_sdk(monkeypatch, answer=PERFECT):
    """A stand-in `openai` module; -> (constructor kwargs, request kwargs) it was called with.

    The only tests that go through the real `run_openai` body. They answer the question provenance
    alone cannot: are the request options recorded in the file the same ones put on the wire?

    `answer=None` makes the endpoint return HTTP 200 with an empty `choices` list — a successful
    call carrying nothing to read.
    """
    constructed, requested = [], []

    class Completions:
        def create(self, **kwargs):
            requested.append(kwargs)
            if answer is None:
                return types.SimpleNamespace(choices=[])
            choice = types.SimpleNamespace(message=types.SimpleNamespace(content=answer),
                                           finish_reason="stop")
            return types.SimpleNamespace(choices=[choice])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    return constructed, requested


def test_the_openai_adapter_sends_exactly_the_request_options_it_records(monkeypatch, tmp_path):
    constructed, requested = install_fake_openai_sdk(monkeypatch)
    stub_image_encoding(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    spec = openai_spec("GLM-OCR")
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "GLM-OCR", "--dataset", "nls", "--no-thinking", "--max-tokens", "1600",
                  "--base-url", "GLM-OCR=https://abc--8000.hf.jobs/v1"])

    assert constructed[0]["base_url"] == "https://abc--8000.hf.jobs/v1"
    assert constructed[0]["api_key"] == "hf_test_token"
    assert constructed[0]["max_retries"] == 0          # the harness owns retry, not the SDK

    sent = requested[0]
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert sent["max_tokens"] == 1600
    assert sent["temperature"] == 0

    recorded = read_result(tmp_path, "GLM-OCR")["items"][0]["provenance"]
    assert recorded["request_options"] == sent["extra_body"]
    assert recorded["thinking_disabled"] is True


def test_an_empty_choices_list_is_named_rather_than_an_index_error():
    """One guard, used by both the openai and the router adapter, so they cannot disagree."""
    with pytest.raises(harness.EmptyResponseError, match="no choices"):
        harness.first_choice(types.SimpleNamespace(choices=[]), {"label": "GLM-OCR"})
    with pytest.raises(harness.EmptyResponseError):
        harness.first_choice(types.SimpleNamespace(choices=None), {"label": "GLM-OCR"})
    only = types.SimpleNamespace(message="m")
    assert harness.first_choice(types.SimpleNamespace(choices=[only]), {"label": "X"}) is only


def test_an_answer_with_no_choices_is_unclassified_not_a_transport_failure(monkeypatch, tmp_path):
    """The call succeeded. Reading it did not — and an IndexError here would be recorded as the
    endpoint failing, which is the one thing it demonstrably did not do."""
    _constructed, requested = install_fake_openai_sdk(monkeypatch, answer=None)
    stub_image_encoding(monkeypatch)
    spec = openai_spec("GLM-OCR")
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))
    slept = record_sleeps(monkeypatch)

    harness.main(["--models", "GLM-OCR", "--dataset", "nls",
                  "--base-url", "GLM-OCR=https://abc--8000.hf.jobs/v1"])

    row = read_result(tmp_path, "GLM-OCR")["items"][0]
    assert len(requested) == 1                     # one call, and no retry of it
    assert slept == []
    assert row["inference_status"] == "transport_error"
    assert row["failure_class"] == "unclassified"
    assert row["error"].startswith("EmptyResponseError: GLM-OCR: ")
    assert "no choices" in row["error"]
    assert row["attempts"] == 1


def test_a_plain_openai_call_sends_no_extra_body(monkeypatch, tmp_path):
    """Nothing asked for, nothing sent: an ordinary call is unchanged by this machinery."""
    _constructed, requested = install_fake_openai_sdk(monkeypatch)
    stub_image_encoding(monkeypatch)
    spec = openai_spec("GLM-OCR")
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "GLM-OCR", "--dataset", "nls",
                  "--base-url", "GLM-OCR=https://abc--8000.hf.jobs/v1"])

    assert "extra_body" not in requested[0]
    assert read_result(tmp_path, "GLM-OCR")["items"][0]["provenance"]["request_options"] == {}


def test_guided_decoding_rides_alongside_the_request_options(monkeypatch, tmp_path):
    """`guided_json` is derived per item, so it is sent but deliberately not echoed into
    provenance.request_options — `model.guided` is where the run records that it was on."""
    _constructed, requested = install_fake_openai_sdk(monkeypatch)
    stub_image_encoding(monkeypatch)
    spec = openai_spec("lift-9B", guided=True)
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "lift-9B", "--dataset", "nls", "--no-thinking",
                  "--base-url", "lift-9B=https://abc--8000.hf.jobs/v1"])

    sent = requested[0]["extra_body"]
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert "guided_json" in sent

    out = read_result(tmp_path, "lift-9B")
    assert out["model"]["guided"] is True
    assert out["items"][0]["provenance"]["request_options"] == {
        "chat_template_kwargs": {"enable_thinking": False}}


def test_the_router_client_takes_extra_body_in_the_pinned_huggingface_hub():
    """`--no-thinking` on a router model is only honest if the client can carry it.

    huggingface_hub is pinned to >=1.29,<2 in the PEP 723 header partly for this signature. If a
    future version drops the parameter, the request options would be silently lost — so the pin
    and this assertion travel together.
    """
    import inspect

    huggingface_hub = pytest.importorskip("huggingface_hub")
    parameters = inspect.signature(huggingface_hub.InferenceClient.chat_completion).parameters
    assert "extra_body" in parameters


def test_every_model_missing_an_endpoint_is_named_at_once(monkeypatch):
    """`--models all` selects three self-hosted models. One failed start, not three."""
    registry(monkeypatch, openai_spec("GLM-OCR"), fake_spec("kept"), openai_spec("lift-9B"))
    refuse_to_load(monkeypatch)
    for label in ("GLAM_BASE_URL_GLM_OCR", "GLAM_BASE_URL_LIFT_9B"):
        monkeypatch.delenv(label, raising=False)

    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--models", "all", "--dataset", "nls"])

    message = str(excinfo.value)
    assert "GLM-OCR needs an endpoint" in message
    assert "lift-9B needs an endpoint" in message


def test_a_resume_can_move_failed_rows_onto_a_pinned_serve(monkeypatch, tmp_path):
    """The August 2026 rerun, done properly: the router failed some rows, a Jobs serve answers them.

    Only the transport changed, so `--resume` is allowed (result_io.RESUME_IDENTITY compares the
    model and the generation settings, not where it was served) and each row records which
    endpoint actually answered it.
    """
    items = items_for(["card-1", "card-2"])
    spec = router_spec("Qwen3.6-27B")
    monkeypatch.setitem(harness.ADAPTERS, "router_vlm",
                        ScriptedAdapter({"card-2": RuntimeError("boom")}))
    install_fake_run(monkeypatch, tmp_path, spec, items)
    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls"])

    pinned = install_adapter_recorder(monkeypatch, kind="openai")
    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls", "--resume",
                  "--base-url", "Qwen3.6-27B=https://abc--8000.hf.jobs/v1"])

    assert len(pinned) == 1                       # only the row the router could not answer
    rows = read_result(tmp_path, "Qwen3.6-27B")["items"]
    assert rows[0]["provenance"]["endpoint"] == "router:auto"
    assert rows[1]["provenance"]["endpoint"] == "https://abc--8000.hf.jobs/v1"
    assert rows[1]["provenance"]["endpoint_attested"] is True


def test_a_resume_publishes_the_model_block_the_file_already_had(monkeypatch, tmp_path):
    """`--provider` on a resume must not rewrite what the kept rows were answered by.

    The switch is legal on a resume — pinning a provider for the rows still to run is the same
    move as pinning an endpoint for them. What it may not do is republish `model.provider` as if
    the rows already in the file had come from there. Per-row provenance carries the truth.
    """
    items = items_for(["card-1", "card-2"])
    spec = router_spec("Qwen3.6-27B")
    monkeypatch.setitem(harness.ADAPTERS, "router_vlm",
                        ScriptedAdapter({"card-2": RuntimeError("boom")}))
    install_fake_run(monkeypatch, tmp_path, spec, items)
    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls"])
    assert read_result(tmp_path, "Qwen3.6-27B")["model"]["provider"] is None

    monkeypatch.setitem(harness.ADAPTERS, "router_vlm", ScriptedAdapter({}))
    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls", "--resume",
                  "--provider", "Qwen3.6-27B=nebius"])

    out = read_result(tmp_path, "Qwen3.6-27B")
    assert out["model"]["provider"] is None       # the file's own block, not this run's switch
    assert out["items"][0]["provenance"]["endpoint"] == "router:auto"
    assert out["items"][1]["provenance"]["endpoint"] == "router:nebius"


# --- the model repo moving under a resume --------------------------------------------------------

def install_model_revision(monkeypatch, sha):
    """Make the Hub `main` lookup answer `sha` for every model."""
    monkeypatch.setattr(harness, "resolve_model_revision", lambda _spec: sha)


def test_a_resume_says_when_the_model_repo_moved_under_it(monkeypatch, tmp_path, capsys):
    """A warning, never a refusal: refusing would break the one resume this harness is built for
    — moving the rows that failed on the router onto a serve you pinned."""
    items = items_for(["card-1", "card-2"])
    spec = fake_spec("fake-moved")
    install_scripted_adapter(monkeypatch, {"card-2": RuntimeError("boom")})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    install_model_revision(monkeypatch, "a" * 40)
    harness.main(["--models", spec["label"], "--dataset", "nls"])

    before = read_result(tmp_path, spec["label"])
    assert "model_revision_moved" not in before["run"]

    install_scripted_adapter(monkeypatch, {})
    install_model_revision(monkeypatch, "b" * 40)
    capsys.readouterr()
    harness.main(["--models", spec["label"], "--dataset", "nls", "--resume"])

    assert ("warning: model repo main moved since this run started (aaaaaaaa → bbbbbbbb); new "
            "rows record the new value, kept rows the old — per-row provenance is the truth") \
        in capsys.readouterr().out

    after = read_result(tmp_path, spec["label"])
    assert after["run"]["model_revision_moved"] is True
    assert after["run"]["complete"] is True                    # a warning, so the run finished
    # the rows are the record, and they disagree honestly
    assert after["items"][0]["provenance"]["model_revision_hub_main"] == "a" * 40
    assert after["items"][1]["provenance"]["model_revision_hub_main"] == "b" * 40


def test_a_resume_onto_the_same_revision_says_nothing(monkeypatch, tmp_path, capsys):
    items = items_for(["card-1", "card-2"])
    spec = fake_spec("fake-still")
    install_scripted_adapter(monkeypatch, {"card-2": RuntimeError("boom")})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    install_model_revision(monkeypatch, "a" * 40)
    harness.main(["--models", spec["label"], "--dataset", "nls"])

    install_scripted_adapter(monkeypatch, {})
    capsys.readouterr()
    harness.main(["--models", spec["label"], "--dataset", "nls", "--resume"])

    assert "model repo main moved" not in capsys.readouterr().out
    assert "model_revision_moved" not in read_result(tmp_path, spec["label"])["run"]


@pytest.mark.parametrize("rows, because", [
    ([], "nothing to compare against"),
    ([{"inference_status": "transport_error",
       "provenance": {"model_revision_hub_main": "a" * 40}}],
     "a failed row records the lookup, not a revision any answer came from"),
    ([{"inference_status": "ok", "provenance": {"model_revision_hub_main": "a" * 40}},
      {"inference_status": "ok", "provenance": {"model_revision_hub_main": "b" * 40}}],
     "already mixed: the rows are the record either way"),
    ([{"inference_status": "ok", "provenance": {"model_revision_hub_main": None}}],
     "a Space kind never resolves one"),
    ([{"inference_status": "ok"}], "no provenance block at all"),
])
def test_the_kept_rows_have_to_agree_before_a_move_can_be_claimed(rows, because):
    assert harness.recorded_model_revision(rows) is None, because


def test_a_move_is_not_claimed_when_this_run_could_not_resolve_one(capsys):
    """`resolve_model_revision` returns None for a Space kind, and for any lookup that failed.
    None is not a new revision, and reporting it as one would invent a move."""
    kept = {"card-1": {"inference_status": "ok",
                       "provenance": {"model_revision_hub_main": "a" * 40}}}
    assert harness.warn_if_model_revision_moved(kept, None) is False
    assert capsys.readouterr().out == ""


# --- what scored the rows -----------------------------------------------------------------------

def test_a_file_scored_by_one_scorer_names_it():
    rows = [{"scorer_version": "2026-08-26a"}, {"scorer_version": "2026-08-26a"}]
    assert harness.scoring_block(rows) == {"scorer_version": "2026-08-26a"}


def test_rows_with_nothing_to_attribute_do_not_make_a_file_mixed():
    """A transport failure was never scored; a scorer crash has an error where scores would be."""
    rows = [{"scorer_version": "2026-08-26a", "content_f1": 1.0},
            {"inference_status": "transport_error", "content_f1": 0.0},
            {"scorer_error": "ZeroDivisionError: scorer bug", "content_f1": 0.0}]
    assert harness.scoring_block(rows) == {"scorer_version": "2026-08-26a"}
    # nothing scored at all: this run's scorer, with nothing to be mixed about
    assert harness.scoring_block([{"inference_status": "transport_error"}]) == {
        "scorer_version": harness.SCORER_VERSION}


def test_a_scored_row_that_does_not_say_which_scorer_counts_as_unknown():
    """Skipping it left a resumed file claiming the current scorer alone — the mixed-file-reading-
    as-clean this block exists to prevent. Unknown is not the same claim as agreeing."""
    assert harness.scorer_versions_in([{"content_f1": 1.0, "schema_valid": 1.0}]) == ["unknown"]

    kept_and_new = [{"content_f1": 1.0},                                 # scored, unversioned
                    {"content_f1": 1.0, "scorer_version": harness.SCORER_VERSION}]
    assert harness.scoring_block(kept_and_new) == {
        "scorer_version": "mixed",
        "scorer_versions": ["2026-08-26a", "unknown"]}


@pytest.mark.parametrize("row", [
    {"inference_status": "transport_error", "content_f1": 0.0},
    {"scorer_error": "ZeroDivisionError: scorer bug", "content_f1": 0.0},
    {"prediction": "{}"},                       # no scores of any kind on it
])
def test_a_row_that_was_never_scored_is_not_unknown_either(row):
    """`unknown` means "scored, by something that did not say". A row nothing scored is neither."""
    assert harness.scorer_versions_in([row]) == []


def test_a_resume_that_keeps_older_scored_rows_says_the_file_is_mixed(monkeypatch, tmp_path):
    """Stamping such a file with this run's scorer is how a mixed file gets read as a clean one.

    `rescore.py` is what brings every row onto one version. Until it is run, the file says so.
    """
    items = items_for(["card-1", "card-2"])
    spec = fake_spec("fake-two-scorers")
    install_scripted_adapter(monkeypatch, {"card-2": RuntimeError("boom")})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    harness.main(["--models", spec["label"], "--dataset", "nls"])

    # age the row that will be kept: it was scored by a scorer this harness no longer runs
    path = tmp_path / "nls" / "fake-two-scorers.json"
    document = json.loads(path.read_text())
    document["items"][0]["scorer_version"] = "1999-01-01-stale"
    path.write_text(json.dumps(document))

    install_scripted_adapter(monkeypatch, {})
    harness.main(["--models", spec["label"], "--dataset", "nls", "--resume"])

    after = read_result(tmp_path, spec["label"])
    assert after["scoring"]["scorer_version"] == "mixed"
    assert after["scoring"]["scorer_versions"] == sorted(["1999-01-01-stale",
                                                          harness.SCORER_VERSION])
    assert after["items"][0]["scorer_version"] == "1999-01-01-stale"     # the kept row, untouched
    assert after["items"][1]["scorer_version"] == harness.SCORER_VERSION


def test_a_rescore_unmixes_the_file_and_drops_the_stale_version_list():
    """The list only makes sense beside `mixed`; a rescore is what replaces both."""
    mixed = {"format_version": 2, "model": {"label": "m"},
             "scoring": {"scorer_version": "mixed",
                         "scorer_versions": ["1999-01-01-stale", "2026-08-26a"]},
             "items": []}
    after = rescore.rewritten(mixed, [], gold_revision="d" * 40)
    assert after["scoring"]["scorer_version"] == rescore.SCORER_VERSION
    assert "scorer_versions" not in after["scoring"]


# --- generation settings are part of the resume identity ----------------------------------------

def test_the_envelope_records_what_was_asked_of_the_model(monkeypatch, tmp_path):
    spec = router_spec("Qwen3.6-27B")
    install_adapter_recorder(monkeypatch, kind="router_vlm")
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls", "--no-thinking",
                  "--max-tokens", "1600"])

    out = read_result(tmp_path, "Qwen3.6-27B")
    assert out["generation"] == {
        "max_tokens": 1600, "temperature": 0, "guided": False,
        "request_options": {"chat_template_kwargs": {"enable_thinking": False}}}
    # the same facts the rows carry one at a time, stated once for the run
    assert out["items"][0]["provenance"]["max_tokens"] == 1600
    assert out["items"][0]["provenance"]["request_options"] == out["generation"]["request_options"]


def test_a_resume_with_a_different_token_budget_is_refused(monkeypatch, tmp_path):
    """900 tokens and 1600 tokens are two questions; one file may not answer both."""
    items = items_for(["card-1"])
    spec = fake_spec("fake-budget")
    adapter = install_scripted_adapter(monkeypatch, {})
    install_fake_run(monkeypatch, tmp_path, spec, items)
    harness.main(["--models", spec["label"], "--dataset", "nls"])
    adapter.calls.clear()

    with pytest.raises(SystemExit) as refusal:
        harness.main(["--models", spec["label"], "--dataset", "nls", "--resume",
                      "--select", "all", "--max-tokens", "1600"])

    message = str(refusal.value)
    assert "generation.max_tokens: file has 900, this run has 1600" in message
    assert "same generation settings" in message
    assert adapter.calls == []                    # refused before a single call was made


def test_a_resume_that_turns_thinking_off_is_refused(monkeypatch, tmp_path):
    """The reason `--no-thinking` cannot be added on a resume: the kept rows were reasoned."""
    items = items_for(["card-1", "card-2"])
    spec = router_spec("Qwen3.6-27B")
    monkeypatch.setitem(harness.ADAPTERS, "router_vlm",
                        ScriptedAdapter({"card-2": RuntimeError("boom")}))
    install_fake_run(monkeypatch, tmp_path, spec, items)
    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls"])

    pinned = install_adapter_recorder(monkeypatch, kind="openai")
    with pytest.raises(SystemExit) as refusal:
        harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls", "--resume", "--no-thinking",
                      "--base-url", "Qwen3.6-27B=https://abc--8000.hf.jobs/v1"])

    assert "generation.request_options" in str(refusal.value)
    assert "enable_thinking" in str(refusal.value)
    assert pinned == []


def test_the_run_says_how_each_model_will_be_reached_before_it_starts(monkeypatch, tmp_path,
                                                                      capsys):
    """An endpoint can come from the environment, which the command line does not show.

    So the run prints where each model is going before the first request, rather than leaving it
    to be discovered in the provenance of a finished file.
    """
    spec = router_spec("Qwen3.6-27B")
    install_adapter_recorder(monkeypatch, kind="router_vlm")
    install_fake_run(monkeypatch, tmp_path, spec, items_for(["card-1"]))

    harness.main(["--models", "Qwen3.6-27B", "--dataset", "nls", "--no-thinking"])

    printed = capsys.readouterr().out
    assert "Qwen3.6-27B" in printed
    assert "router:auto" in printed
    assert "unattested" in printed
    assert "thinking off" in printed


# --- the consumer-side publication gate --------------------------------------------------------

def _result_file(label, *, failed=False, **run_overrides):
    """A finished, boardable file. `failed=True` makes it one carrying a transport failure.

    The rows and `run.transport_errors` are built together: a run block claiming a failure the
    rows do not show is a contract problem in its own right, which would refuse the file for the
    wrong reason.
    """
    second = ({"id": "card-2", "prediction": None, "attempts": 3,
               "inference_status": "transport_error", "failure_class": "transient",
               "content_f1": 0.0, "schema_valid": 0.0} if failed
              else _scored_item(id="card-2", inference_status="ok", attempts=1))
    rows_in = [_scored_item(inference_status="ok", attempts=1), second]
    ids = [row["id"] for row in rows_in]
    run = {"started_at": "t0", "finished_at": "t1", "complete": True,
           "transport_errors": 1 if failed else 0}
    run.update(run_overrides)
    return {"format_version": 2,
            "model": {"label": label, "kind": "fake", "id": "fake/model", "params": "0B"},
            "dataset": {"id": NLS_REPO, "inference_revision": "d" * 40, "item_count": len(ids),
                        "item_ids_sha256": result_io.item_ids_sha256(ids)},
            "scoring": {"scorer_version": "v"}, "run": run, "items": rows_in}


def test_the_board_refuses_a_failed_file_and_never_boards_a_smoke_test(monkeypatch, tmp_path,
                                                                      capsys):
    """board.py is one of four consumers; all four discover their input the same way."""
    def two_card_gold():
        schema, gold, sha = _fake_gold()
        return schema, {**gold, "card-2": dict(gold["card-1"])}, sha

    results = tmp_path / "nls"
    results.mkdir()
    (results / "good.json").write_text(json.dumps(_result_file("good")))
    (results / "good.limit2.json").write_text(json.dumps(_result_file("good")))
    (results / "failed.json").write_text(json.dumps(_result_file("failed", failed=True)))

    monkeypatch.setattr(board, "RESULTS", results)
    monkeypatch.setattr(board, "load_gold", two_card_gold)

    with pytest.raises(SystemExit) as refusal:
        board.board_rows(include_baseline=False)
    assert "failed.json: run.transport_errors is 1" in str(refusal.value)

    rows = board.board_rows(include_baseline=False, allow_incomplete=True)
    assert sorted(r["label"] for r in rows) == ["failed", "good"]   # the smoke test is not a row
    printed = capsys.readouterr().out
    assert "skipping smoke-test file good.limit2.json" in printed
    assert "warning: publishing failed.json anyway" in printed


# --- the built page states no count it did not count ---------------------------------------------

SITE_ROW_KEYS = ("identifiers_wrong", "invented_fields", "fields_left_blank", "content_f1",
                 "exact_fields_f1", "fuzzy_fields_f1", "precision", "recall",
                 "verified_f1", "corrected_f1", "schema_valid")


def site_row(label, **overrides):
    """One board row, of the shape site_nls.build() reads. Numbers chosen only to be plottable."""
    row = {key: 0.5 for key in SITE_ROW_KEYS}
    row.update({"label": label, "params": "9B", "n": 3, "n_ident_filled": 3,
                "identifier_clean_cards": 2, "attested": True})
    row.update(overrides)
    return row


def install_three_card_gold(monkeypatch, tmp_path):
    """Point site_nls at a three-card gold set and one canned prediction over it.

    The worked card is built to show both axes, which `build` re-checks: the gold has a heading
    the model left blank, and the model filled in a manuscript number the card does not have.
    """
    worked_id = "card-1"
    gold = {f"card-{n}": {"gold": dict(GOLD), "label_status": "verified",
                          "verdict": "ok", "image_type": "typed"} for n in (1, 2, 3)}
    gold[worked_id]["gold"] = {"heading": "Allan, T.", "ms_no": ""}
    gold["card-3"]["label_status"] = "corrected"        # 2 verified / 1 corrected

    results = tmp_path / "nls"
    results.mkdir()
    (results / "gemma-4-31B.json").write_text(json.dumps(
        {"items": [{"id": worked_id, "prediction": json.dumps({"heading": "", "ms_no": "MS.999"})}]}))

    monkeypatch.setattr(site_nls, "load_gold", lambda: (SCHEMA_STR, gold, "d" * 40))
    monkeypatch.setattr(site_nls, "WORKED", ("gemma-4-31B", worked_id, "worked-card.jpg"))
    monkeypatch.setattr(site_nls, "RESULTS", results)
    monkeypatch.setattr(site_nls, "OUT", tmp_path / "index.html")
    monkeypatch.setattr(site_nls, "board_rows", lambda **_kwargs: [
        site_row("gemma-4-31B"),
        site_row("lift-9B", identifiers_wrong=0.2, fields_left_blank=0.3),
        site_row("constants baseline", baseline=True)])
    return tmp_path / "index.html"


def page_prose(page: str) -> str:
    """The page's copy: stylesheet and plotted SVG dropped, whitespace collapsed.

    Both dropped parts are full of numbers nobody wrote as prose — a `max-width:980px` copied out
    of index.html, a scatter coordinate that lands on 98.4 — and a count-in-the-copy test that
    read them would fail on a layout tweak instead of on a stale sentence. Whitespace is
    collapsed because a sentence is free to wrap anywhere in the source and still be one
    sentence on the page.
    """
    without_style = re.sub(r"<style>.*?</style>", "", page, flags=re.S)
    without_svg = re.sub(r"<svg .*?</svg>", "", without_style, flags=re.S)
    return re.sub(r"\s+", " ", without_svg)


def test_the_page_takes_every_card_count_from_the_gold(monkeypatch, tmp_path):
    """The page used to type 98 into six sentences. The gold set is still accepting rows, so a
    typed count is true for one build and silently wrong after the next."""
    out = install_three_card_gold(monkeypatch, tmp_path)

    site_nls.build()

    page = page_prose(out.read_text())
    assert "98" not in page
    assert "3 index cards" in page                       # the header, counted
    assert "over all 3 cards" in page                    # the footer
    assert "across the 3 gold records" in page           # the baseline hover


def test_a_model_that_wrote_no_identifiers_cannot_move_the_better_corner(monkeypatch, tmp_path):
    """A hollow row scores 0% identifiers wrong because it never wrote one, not because it got
    them right. Counting it toward the median would drag the "better" corner of the plot down and
    let real models into a box they had not earned."""
    real = [site_row("a", identifiers_wrong=0.30), site_row("b", identifiers_wrong=0.40),
            site_row("c", identifiers_wrong=0.50)]
    hollow = site_row("hollow", identifiers_wrong=0.0, n_ident_filled=0)
    base = site_row("constants baseline", baseline=True)

    def better_corner(board, name):
        workdir = tmp_path / name
        workdir.mkdir()
        out = install_three_card_gold(monkeypatch, workdir)
        monkeypatch.setattr(site_nls, "board_rows", lambda **_kwargs: board)
        site_nls.build()
        return re.search(r'<rect class="best"[^>]*\sy="([\d.]+)"', out.read_text()).group(1)

    assert better_corner(real + [base], "without") == better_corner(real + [hollow, base], "with")


def test_the_page_takes_the_verified_corrected_split_from_the_gold(monkeypatch, tmp_path):
    """This split moves for two reasons: the gold gains rows, and a reviewer editing one label
    moves a card from one side to the other without changing the total."""
    out = install_three_card_gold(monkeypatch, tmp_path)     # 2 verified, 1 corrected

    site_nls.build()

    page = page_prose(out.read_text())
    assert "66" not in page and "(32)" not in page
    assert "verified (2)" in page and "corrected (1)" in page          # the breakdown header
    assert "<b>verified</b> (2 cards)" in page                         # the footnote
    assert "<b>corrected</b> (1)" in page


def test_the_page_says_who_drafted_the_labels_before_the_table(monkeypatch, tmp_path):
    """The drafting model used to be disclosed only by a marker on its own row. It is not on the
    board, so the marker never rendered and the page read as "human-checked" alone. The sentence
    now lives above the table and takes its counts from the gold, like every other count."""
    out = install_three_card_gold(monkeypatch, tmp_path)     # 2 verified, 1 corrected

    site_nls.build()

    page = page_prose(out.read_text())
    disclosure = page.index("drafted by Qwen3.6-35B-A3B")
    assert disclosure < page.index("<table")
    assert "2 accepted as drafted, 1 corrected" in page
    assert "2 models and the constants baseline" in out.read_text()   # the chart's aria-label


def test_the_interval_on_the_page_is_computed_from_the_result_files(tmp_path):
    """The noise sentence used to be typed. The half-width comes from resampling the cards in the
    result files, is deterministic, and is zero when there is nothing to resample."""
    results = tmp_path / "nls"
    results.mkdir()
    (results / "steady.json").write_text(json.dumps(
        {"items": [{"id": f"c{n}", "content_f1": 0.8} for n in range(20)]}))
    assert board.f1_halfwidth_points(results) == 0.0

    (results / "swingy.json").write_text(json.dumps(
        {"items": [{"id": f"c{n}", "content_f1": n % 2} for n in range(20)]}))
    width = board.f1_halfwidth_points(results)
    assert 15 < width < 30                    # ~±22 points for a 0/1 coin over 20 cards
    assert width == board.f1_halfwidth_points(results)


def test_the_truncation_count_comes_from_the_result_file(tmp_path, monkeypatch):
    """A row note used to type "two of the 98 outputs hit the token limit". The count now comes
    from finish_reason in the file, like every other number on the page."""
    results = tmp_path / "nls"
    results.mkdir()
    monkeypatch.setattr(site_nls, "RESULTS", results)
    assert site_nls.truncated(None) == (0, 0)
    assert site_nls.truncated(results / "absent.json") == (0, 0)
    # Named unlike its label on purpose: the gate accepts such files, so the page must read the
    # file the board found, not one it guessed from the label.
    (results / "submitted-as-something-else.json").write_text(json.dumps({"items": [
        {"id": "a", "provenance": {"finish_reason": "length"}},
        {"id": "b", "provenance": {"finish_reason": "stop"}},
        {"id": "c"}]}))
    assert site_nls.truncated(results / "submitted-as-something-else.json") == (1, 3)
