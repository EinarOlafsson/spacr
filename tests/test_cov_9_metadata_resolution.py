"""Every way the metadata resolver refuses to guess an identity.

A plate, a row and a column are what every downstream number is grouped by.
The resolver's whole purpose is that it never invents one: when it cannot
derive an identity it says so, and when the identity it would derive
disagrees with one already in the frame it stops rather than remapping the
plate underneath the user.
"""
from __future__ import annotations

from dataclasses import fields

import pandas as pd
import pytest

from spacr import schema
from spacr.metadata_resolution import (
    MetadataDecision,
    MetadataResolutionRequired,
    _derive_well_columns,
    _pseudo_wells,
    clear_run_metadata_decisions,
    resolve_metadata_columns,
)


@pytest.fixture(autouse=True)
def _no_remembered_decisions():
    """Remembered prompt answers are process-wide; start and end each test clean."""
    clear_run_metadata_decisions()
    yield
    clear_run_metadata_decisions()


# ---------------------------------------------------------------------------
# The column map
# ---------------------------------------------------------------------------

def test_a_column_map_naming_a_column_that_is_not_there_is_refused():
    """Mapping from a missing source cannot be honoured, silently or otherwise.

    The map is ``{canonical_target: actual_source}``; if the source is absent
    the target stays missing, so the caller has to be told what is available
    instead of receiving a frame that quietly lacks the column it asked for.
    """
    frame = pd.DataFrame({"barcode": ["p1"], "value": [1.0]})

    with pytest.raises(MetadataResolutionRequired) as excinfo:
        resolve_metadata_columns(frame, [schema.PLATE_KEY],
                                 column_map={schema.PLATE_KEY: "not_a_column"})
    assert excinfo.value.missing == (schema.PLATE_KEY,)
    assert excinfo.value.available == ("barcode", "value")


def test_a_map_that_would_collide_only_by_case_is_refused():
    """SQLite and pandas disagree about case, so a case-only clash is a clash.

    Renaming ``Sample`` to ``sample`` beside an existing ``SAMPLE`` produces
    two columns that a database round trip cannot tell apart. One of them
    would be lost on write, so the rename is refused rather than planned.
    """
    frame = pd.DataFrame({"Sample": [1], "SAMPLE": [2]})

    with pytest.raises(ValueError) as excinfo:
        resolve_metadata_columns(frame, ["sample"],
                                 column_map={"sample": "Sample"})
    assert "case-insensitive column collision" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Deriving row and column from a well
# ---------------------------------------------------------------------------

def test_a_well_column_that_does_not_exist_derives_nothing():
    """Naming a missing well column must not look like a successful derivation.

    Reporting it as derived would leave the run believing it had rowID and
    columnID when the frame has neither, and the failure would surface later
    as an empty grouping rather than a refusal here.
    """
    frame = pd.DataFrame({"condition": ["a", "b"]})

    with pytest.raises(MetadataResolutionRequired) as excinfo:
        resolve_metadata_columns(frame, [schema.ROW_KEY, schema.COLUMN_KEY],
                                 well_column="well")
    assert set(excinfo.value.missing) == {schema.ROW_KEY, schema.COLUMN_KEY}


def test_a_well_column_that_cannot_be_parsed_derives_nothing():
    """One unparseable well invalidates the whole derivation, not just its row.

    Deriving row and column for the parseable wells only would produce a
    frame where some objects carry a plate position and others do not, and
    every per-well summary computed from it would silently omit them.
    """
    frame = pd.DataFrame({"well": ["A01", "sample-2"]})

    with pytest.raises(MetadataResolutionRequired):
        resolve_metadata_columns(frame, [schema.ROW_KEY, schema.COLUMN_KEY],
                                 well_column="well")


def test_a_column_that_is_already_present_is_not_overwritten_by_the_well():
    """Only the identity columns actually missing are filled in from the well.

    A columnID the user supplied is authoritative. Recomputing it from the
    well string would overwrite a deliberate value with a derived one and
    change which wells the analysis groups together.
    """
    frame = pd.DataFrame({"well": ["A01", "B02"], schema.COLUMN_KEY: ["c9", "c9"]})

    result = resolve_metadata_columns(
        frame, [schema.ROW_KEY, schema.COLUMN_KEY], well_column="well")

    assert result.derived_from_well == "well"
    assert list(result.frame[schema.ROW_KEY]) == ["r1", "r2"]
    assert list(result.frame[schema.COLUMN_KEY]) == ["c9", "c9"]


def test_a_well_that_disagrees_with_an_existing_column_stops_the_run():
    """A derived identity that contradicts a stored one is a plate remap.

    Preferring either value silently would move objects to wells they were
    never measured in, so the disagreement is reported with the first row it
    was seen at instead of being resolved by precedence.
    """
    frame = pd.DataFrame({"well": ["A01", "B02"],
                          schema.ROW_KEY: ["r1", "r5"]})

    with pytest.raises(ValueError) as excinfo:
        _derive_well_columns(frame, "well", [schema.ROW_KEY])
    message = str(excinfo.value)
    assert schema.ROW_KEY in message
    assert "refusing a silent plate remap" in message


# ---------------------------------------------------------------------------
# Pseudo wells
# ---------------------------------------------------------------------------

def test_a_pseudo_source_that_is_not_a_column_is_refused():
    """Pseudo wells need a real column to be distinct about.

    Without one there is nothing to assign positions from, and inventing a
    position per row would give every object its own well -- an audit trail
    that describes no experiment.
    """
    frame = pd.DataFrame({"condition": ["a", "b"]})

    with pytest.raises(MetadataResolutionRequired) as excinfo:
        resolve_metadata_columns(frame, [schema.ROW_KEY, schema.COLUMN_KEY],
                                 pseudo_source="treatment", allow_pseudo=True)
    assert set(excinfo.value.missing) == {schema.ROW_KEY, schema.COLUMN_KEY}


def test_two_conditions_never_share_one_pseudo_well(monkeypatch):
    """The pseudo-well assignment refuses itself if it stops being one-to-one.

    Two distinct source values landing in the same pseudo well would merge two
    conditions into one group, and every number computed per well afterwards
    would describe a mixture. The check is driven here by replacing the
    enumeration the positions are derived from with one that repeats an index,
    which is the only way the formula could ever collide.
    """
    import spacr.metadata_resolution as mr

    def _stuck(iterable, start=0):
        return [(start, item) for item in iterable]

    monkeypatch.setattr(mr, "enumerate", _stuck, raising=False)
    frame = pd.DataFrame({"treatment": ["drug", "vehicle"]})

    with pytest.raises(RuntimeError) as excinfo:
        _pseudo_wells(frame, "treatment", [schema.ROW_KEY, schema.COLUMN_KEY])
    assert "not injective" in str(excinfo.value)


# ---------------------------------------------------------------------------
# What the resolver accepts as input and as an answer
# ---------------------------------------------------------------------------

def test_the_resolver_refuses_anything_that_is_not_a_dataframe():
    """A dict of columns is not a frame, and treating it as one fails obscurely.

    Naming the type received turns a downstream ``AttributeError`` about
    ``.columns`` into a message the caller can act on at the call site.
    """
    with pytest.raises(TypeError) as excinfo:
        resolve_metadata_columns({"plateID": ["p1"]}, [schema.PLATE_KEY])
    assert "dict" in str(excinfo.value)


def test_a_prompt_that_answers_with_something_else_is_refused():
    """The prompt contract is one decision object, not a bare mapping.

    A dialog that returns a dict of choices would otherwise be re-entered as
    keyword arguments it does not match, and the failure would look like a
    resolver bug rather than a UI one.
    """
    frame = pd.DataFrame({"condition": ["a"]})

    with pytest.raises(TypeError) as excinfo:
        resolve_metadata_columns(frame, [schema.PLATE_KEY],
                                 prompt=lambda request: {"plateID": "condition"})
    assert "MetadataDecision" in str(excinfo.value)


def test_a_prompt_that_answers_properly_still_resolves_the_frame():
    """The refusal above must not have broken the answer it guards."""
    frame = pd.DataFrame({"condition": ["a", "b"]})

    for field in fields(MetadataDecision):
        assert f":param {field.name}:" in (MetadataDecision.__doc__ or "")
    result = resolve_metadata_columns(
        frame, [schema.PLATE_KEY],
        prompt=lambda request: MetadataDecision(
            column_map={schema.PLATE_KEY: "condition"}, remember=False))

    assert list(result.frame[schema.PLATE_KEY]) == ["a", "b"]
