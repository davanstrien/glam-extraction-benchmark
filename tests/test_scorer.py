"""Small example tests for the scorer, one per decision that moved a published number.

Run: uv run --with pytest -m pytest tests
"""
import unicodedata

from scorer import _norm, score

STRING = {"type": "string"}
EXACT = {"type": "string", "x-match": "exact"}


def test_norm_keeps_accented_letters_whole():
    # NFKD split "ü" into "u" + combining mark, and the punctuation strip made that a space.
    assert _norm("Müller") == "müller"
    assert _norm("Sébastien") == "sébastien"


def test_norm_composed_and_decomposed_input_match():
    composed = "Sá e Melo"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed  # different codepoints in
    assert _norm(composed) == _norm(decomposed)  # same string out


def test_norm_folds_compatibility_forms_and_punctuation():
    assert _norm("ﬁne") == "fine"
    assert _norm("f. 102") == "f 102"


def test_norm_keeps_accents_meaningful_for_exact_match():
    # An NLS reviewer corrected "Sa" to "Sá"; the scorer must be able to see that difference.
    schema = {"type": "object", "properties": {"heading": EXACT}}
    assert score({"heading": "Sá"}, {"heading": "Sa"}, schema)["content_f1"] == 0.0
    assert score({"heading": "Sá"}, {"heading": "Sá"}, schema)["content_f1"] == 1.0


def test_explicit_null_takes_the_boolean_default_like_an_absent_key():
    schema = {"type": "object", "properties": {"has_corrections": {"type": "boolean", "default": False}}}
    gold = {"has_corrections": False}
    absent = score(gold, {}, schema)
    null = score(gold, {"has_corrections": None}, schema)
    assert absent["n_filled"] == null["n_filled"] == 1
    assert absent["content_f1"] == null["content_f1"] == 1.0


def test_skipped_entry_does_not_steal_its_neighbours_prediction():
    entry = {"type": "object", "properties": {"ms_no": EXACT, "description": STRING}}
    schema = {"type": "object", "properties": {"entries": {"type": "array", "items": entry}}}
    gold = {"entries": [{"ms_no": "MS100", "description": "a"},
                        {"ms_no": "MS200", "description": "b"},
                        {"ms_no": "MS300", "description": "c"}]}
    pred = {"entries": [{"ms_no": "MS200", "description": "b"},
                        {"ms_no": "MS300", "description": "c"}]}
    result = score(gold, pred, schema)
    assert result["n_ident_wrong"] == 0  # both identifiers the model wrote are right
    assert result["n_blank"] == 2  # the skipped entry's two fields are blank, not wrong
    assert result["n_invented"] == 0
