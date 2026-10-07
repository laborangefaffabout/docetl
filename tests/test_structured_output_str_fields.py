"""Structured output keeps a field declared ``str`` as a string, even when
its value is JSON text; undeclared or non-string fields are still decoded."""
import json
from types import SimpleNamespace

from docetl.operations.utils.api import APIWrapper, _declares_string
from docetl.operations.utils.validation import validate_output_types


def _response(payload: dict):
    message = SimpleNamespace(content=json.dumps(payload))
    return SimpleNamespace(model="anthropic/claude-haiku-5.5",
                           choices=[SimpleNamespace(message=message)])


def _parse(payload: dict, schema: dict) -> dict:
    api = APIWrapper.__new__(APIWrapper)
    (out,) = api._parse_llm_response_helper(
        _response(payload), schema=schema, use_structured_output=True,
    )
    return out


def test_json_text_in_a_str_field_stays_a_string():
    schema = {"extraction_json": "str"}
    out = _parse({"extraction_json": '{"questions": [1, 2]}'}, schema)
    assert out["extraction_json"] == '{"questions": [1, 2]}'
    assert validate_output_types(out, schema)[0]


def test_json_text_in_a_list_field_is_still_decoded():
    out = _parse({"items": "[1, 2]"}, {"items": "list[int]"})
    assert out["items"] == [1, 2]


def test_declares_string_spellings():
    for t in ("str", "string", "text", "varchar", " String "):
        assert _declares_string({"k": t}, "k")
    assert not _declares_string({"k": "list[str]"}, "k")
    assert not _declares_string({}, "k")
    assert not _declares_string(None, "k")
