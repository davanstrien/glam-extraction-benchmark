"""Schema is model-agnostic. The benchmark's **canonical** format is standard **JSON Schema**
(the lingua franca: OpenAI/Gemini structured outputs, vLLM `guided_json`, Pydantic, outlines).

Each model's *solver* converts FROM JSON Schema to its own dialect:
  - NuExtract        -> NuExtract template      (jsonschema_to_nuextract)
  - vLLM / lift       -> guided_json (native)    (to_guided_json)
  - general VLM       -> JSON Schema in the prompt (use as-is)
  - a user's stack    -> Pydantic, etc.          (json schema -> their format)

KIE annotation: a string field that must be copied *verbatim* (identifiers, call numbers,
registration numbers) is marked `{"type":"string","x-match":"exact"}`. `x-` is a custom keyword,
ignored by JSON-Schema validators and by guided decoding; the scorer reads it to decide exact vs
fuzzy matching. Plain `{"type":"string"}` is scored fuzzily (SequenceMatcher ratio).
"""
from __future__ import annotations

def jsonschema_to_nuextract(s):
    """JSON Schema node -> NuExtract template node (for the NuExtract solver)."""
    if not isinstance(s, dict):
        return "string"
    if "enum" in s:
        return list(s["enum"])
    t = s.get("type")
    if t == "object":
        return {k: jsonschema_to_nuextract(v) for k, v in s.get("properties", {}).items()}
    if t == "array":
        return [jsonschema_to_nuextract(s.get("items", {"type": "string"}))]
    if t == "string":
        if s.get("x-match") == "exact":
            return "verbatim-string"
        if s.get("format") == "date":
            return "date"
        return "string"
    if t in ("integer", "number", "boolean"):
        return t
    return "string"


def to_guided_json(s):
    """Strip KIE/custom annotations (x-*) so the schema is clean for vLLM guided_json / lift."""
    if isinstance(s, dict):
        return {k: to_guided_json(v) for k, v in s.items() if not k.startswith("x-")}
    if isinstance(s, list):
        return [to_guided_json(v) for v in s]
    return s


def field_notes(s, path="") -> list[str]:
    """Collect "field: description" lines from a JSON Schema. NuExtract's template dialect is
    pure structure (descriptions are lost in conversion), so its solver must carry these in the
    instruction text — otherwise guidance like "For cross_reference cards: ..." never reaches
    the model, and the comparison with schema-in-prompt models isn't fair."""
    notes = []
    if not isinstance(s, dict):
        return notes
    if s.get("description") and path:
        notes.append(f"{path}: {s['description']}")
    for k, v in s.get("properties", {}).items():
        notes.extend(field_notes(v, f"{path}.{k}" if path else k))
    if "items" in s:
        notes.extend(field_notes(s["items"], path + "[]" if path else "[]"))
    return notes


def field_match_type(node) -> str:
    """How the scorer should match a leaf field, read from its JSON Schema node."""
    if not isinstance(node, dict):
        return "fuzzy"
    if "enum" in node or node.get("x-match") == "exact" or node.get("format") == "date":
        return "exact"
    if node.get("type") in ("integer", "number", "boolean"):
        return "exact"
    return "fuzzy"  # plain string


def field_kind(node) -> str:
    """What *sort* of thing a leaf field holds — finer than field_match_type, and used only for
    reporting (never for matching). The exact-match lane is not one thing: a wrong manuscript
    number and a wrong `has_corrections` flag are both "exact, and wrong", but only the first is
    an identifier a catalogue record is judged on.

      identifier -> copied verbatim, right or wrong: x-match:exact, format:date
      category   -> a closed vocabulary (enum)
      flag       -> boolean / numeric
      text       -> free text, scored on character closeness
    """
    if not isinstance(node, dict):
        return "text"
    if node.get("x-match") == "exact" or node.get("format") == "date":
        return "identifier"
    if "enum" in node:
        return "category"
    if node.get("type") in ("boolean", "integer", "number"):
        return "flag"
    return "text"
