"""A release updates P348 on Wikidata by replacing the claim, not rebuilding it.

Wikidata statements are edited by volunteers. A release that removed the
version statement and created a new one would silently discard whatever a
curator had added to it -- its references, its rank, and its identity in the
item's history. The release therefore edits the claim it finds.

These are the parts that can be checked without touching the network: the
claim the job would submit, and the decision not to submit at all.
"""
from __future__ import annotations

import copy
import datetime
import importlib.util
import pathlib

import pytest

_SOURCE = (pathlib.Path(__file__).resolve().parents[1]
           / "tools" / "wikidata_set_version.py")
_SPEC = importlib.util.spec_from_file_location("wikidata_set_version", _SOURCE)
wikidata_set_version = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wikidata_set_version)

build_claim = wikidata_set_version.build_claim
claim_is_current = wikidata_set_version.claim_is_current

RELEASED = datetime.date(2026, 9, 23)


@pytest.fixture
def curated_claim():
    """A P348 statement as it looks after someone has worked on it."""
    return {
        "id": "Q123456$0d4f-1111-2222",
        "type": "statement",
        "rank": "preferred",
        "mainsnak": {
            "snaktype": "value",
            "property": "P348",
            "datatype": "string",
            "datavalue": {"value": "1.5.0.9", "type": "string"},
        },
        "references": [{"hash": "abc", "snaks": {}}],
        "qualifiers": {"P577": [{"snaktype": "value", "property": "P577"}]},
        "qualifiers-order": ["P577"],
    }


def test_the_statement_keeps_its_identity_rank_and_references(curated_claim):
    """Everything that is not the version or the date survives the edit."""
    updated = build_claim("1.5.1.0", RELEASED, curated_claim)

    assert updated["id"] == curated_claim["id"]
    assert updated["rank"] == "preferred"
    assert updated["references"] == curated_claim["references"]


def test_the_version_and_the_date_are_the_only_things_that_move(curated_claim):
    updated = build_claim("1.5.1.0", RELEASED, curated_claim)

    assert updated["mainsnak"]["datavalue"]["value"] == "1.5.1.0"
    stamp = updated["qualifiers"]["P577"][0]["datavalue"]["value"]
    assert stamp["time"] == "+2026-09-23T00:00:00Z"
    assert stamp["precision"] == 11, "a release is dated to the day"


def test_the_claim_on_the_item_is_never_mutated(curated_claim):
    """The caller's claim is left alone, so a failed edit changes nothing."""
    before = copy.deepcopy(curated_claim)

    build_claim("1.5.1.0", RELEASED, curated_claim)

    assert curated_claim == before


def test_an_item_with_no_version_statement_gets_one():
    """The first release on a fresh item creates the statement."""
    created = build_claim("1.5.1.0", RELEASED)

    assert "id" not in created, "a new statement is created without an id"
    assert created["mainsnak"]["property"] == "P348"
    assert created["qualifiers-order"] == ["P577"]


def test_a_release_that_changes_nothing_makes_no_edit(curated_claim):
    """A no-op edit still shows in the history and on every watchlist."""
    assert claim_is_current(curated_claim, "1.5.0.9") is True
    assert claim_is_current(curated_claim, "1.5.1.0") is False


def test_a_malformed_claim_is_not_mistaken_for_a_current_one():
    """An unreadable claim must be replaced, never assumed to be right."""
    assert claim_is_current({}, "1.5.1.0") is False
    assert claim_is_current({"mainsnak": {}}, "1.5.1.0") is False
