"""Every way the metadata resolver refuses to guess an identity.

A plate, a row and a column are what every downstream number is grouped by.
The resolver's whole purpose is that it never invents one: when it cannot
derive an identity it says so, and when the identity it would derive
disagrees with one already in the frame it stops rather than remapping the
plate underneath the user.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr import schema
from spacr.metadata_resolution import (
    _derive_well_columns,
    _pseudo_wells,
    clear_run_metadata_decisions,
)


@pytest.fixture(autouse=True)
def _no_remembered_decisions():
    """Remembered prompt answers are process-wide; start and end each test clean."""
    clear_run_metadata_decisions()
    yield
    clear_run_metadata_decisions()


# ---------------------------------------------------------------------------
# Deriving row and column from a well
# ---------------------------------------------------------------------------


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
