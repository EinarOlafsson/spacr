"""A release changes the bio.tools version and carries everything else back.

bio.tools answers 405 to PATCH, so a single field cannot be written on its
own -- the whole entry goes back by PUT. That makes the fetch the safety
mechanism: the entry is read from bio.tools immediately before the write, so
an ELIXIR curator's edits ride back out with it. Sending a copy stored in
this repository would revert their work on every release.

Two shapes the API emits are refused by the same API on input, and both
appear here because a release failed on them once: server-managed fields,
and the nulls and empty lists GET returns for unset optional fields.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib

import pytest

_SOURCE = (pathlib.Path(__file__).resolve().parents[1]
           / "tools" / "biotools_set_version.py")
_SPEC = importlib.util.spec_from_file_location("biotools_set_version", _SOURCE)
biotools_set_version = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(biotools_set_version)

with_version = biotools_set_version.with_version
strip_empty = biotools_set_version.strip_empty
SERVER_MANAGED = biotools_set_version.SERVER_MANAGED


@pytest.fixture
def fetched_entry():
    """An entry shaped the way bio.tools actually returns one."""
    return {
        "name": "spaCR",
        "biotoolsID": "spacr",
        "version": ["1.5.0.9"],
        "description": "A curator rewrote this sentence.",
        "topic": [{"term": "Bioimaging", "uri": "http://edamontology.org/topic_3383"}],
        "publication": [{"doi": "10.1016/j.celrep.2025.115694",
                         "pmid": "40349346", "type": ["Usage"],
                         "version": None, "metadata": None}],
        "function": [{"operation": [{"term": "Image analysis"}], "cmd": None}],
        "collectionID": [],
        "elixirNode": [],
        "additionDate": "2026-09-23T21:02:15Z",
        "lastUpdate": "2026-09-24T10:00:00Z",
        "owner": "einar_birnir",
        "validated": 1,
    }


def test_the_version_is_replaced(fetched_entry):
    body = with_version(fetched_entry, "1.5.1.0")

    assert body["version"] == ["1.5.1.0"]


def test_a_curators_edits_are_carried_back_out(fetched_entry):
    """Whatever came back from bio.tools goes back to bio.tools."""
    body = with_version(fetched_entry, "1.5.1.0")

    assert body["description"] == "A curator rewrote this sentence."
    assert body["topic"][0]["term"] == "Bioimaging"
    assert body["publication"][0]["doi"] == "10.1016/j.celrep.2025.115694"


def test_server_managed_fields_are_removed(fetched_entry):
    """PUT refuses the fields its own GET emits."""
    body = with_version(fetched_entry, "1.5.1.0")

    for field in SERVER_MANAGED:
        assert field not in body, f"{field} is not writable"


def test_nulls_and_empty_lists_are_removed(fetched_entry):
    """"This field may not be null" is what a release failed on."""
    blob = json.dumps(with_version(fetched_entry, "1.5.1.0"))

    assert ": null" not in blob
    assert ": []" not in blob


def test_the_fetched_entry_is_not_mutated(fetched_entry):
    """A failed write must leave the caller's copy alone."""
    before = copy.deepcopy(fetched_entry)

    with_version(fetched_entry, "1.5.1.0")

    assert fetched_entry == before


def test_a_falsy_value_is_not_an_absent_one():
    """Zero, False and "" are real values; only None and [] are absences."""
    assert strip_empty({"a": 0, "b": False, "c": "", "d": None, "e": []}) == {
        "a": 0, "b": False, "c": ""}
