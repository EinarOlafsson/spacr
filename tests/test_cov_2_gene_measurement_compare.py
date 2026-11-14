"""What the gene/measurement comparison does when part of its input is unusable.

The panel behind this module resolves wells and groups and attaches
statistics to a comparison. Every path here is one where something is
missing: a group that matches no row, a count table with no guide column, an
empty comparison, a statistics engine that refuses. None of them may raise
-- the panel has no console -- and none of them may be silent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import gene_measurement_compare as gmc
from spacr.gene_measurement_compare import Comparison, build


@pytest.fixture()
def objects():
    """Twelve cells over four wells of one plate, with two measurements."""
    n = 12
    return pd.DataFrame({
        "pathogen_area": np.arange(n, dtype=float) + 1.0,
        "cell_area": np.arange(n, dtype=float) + 20.0,
        "plateID": ["p1"] * n,
        "rowID": ["r1"] * 6 + ["r2"] * 6,
        "columnID": ["c1", "c2", "c3"] * 4,
    })


# ---------------------------------------------------------------------------
# resolving wells and groups
# ---------------------------------------------------------------------------

def test_a_group_that_matches_no_object_contributes_no_wells(objects):
    """A gene with no annotated cells has no wells, and is left out entirely.

    Naming it with an empty well tuple would put an empty group on the panel;
    naming it with every well would be worse. The group simply is not there.
    """
    picked = gmc.wells_of(objects, {"present": objects.index[:3],
                                    "absent": ["no-such-row"]})

    assert set(picked) == {"present"}
    assert picked["present"] == ("p1_r1_c1", "p1_r1_c2", "p1_r1_c3")


def test_controls_cannot_be_resolved_without_a_guide_column(objects):
    """Count data with no guide column names no controls.

    Returning every well instead would make "against the controls" a
    comparison against the whole plate, which is a different experiment
    wearing the same label.
    """
    counts = pd.DataFrame({"plateID": ["p1"], "rowID": ["r1"],
                           "columnID": ["c1"]})

    assert gmc.control_wells(counts, ["nc"], guide_column="grna") == ()


def test_no_operator_returns_the_first_measurement_untouched(objects):
    """An empty operator is "just this measurement", not an error.

    The panel's operator box starts empty, so this is the ordinary case, and
    it must not consult the second column -- which is also empty.
    """
    values, name, dropped = gmc.combine(objects, "pathogen_area", "", None)

    assert name == "pathogen_area"
    assert dropped == 0
    pd.testing.assert_series_equal(values, objects["pathogen_area"],
                                   check_names=False)


def test_an_empty_comparison_has_no_groups_and_no_counts():
    """A comparison that could not be built still answers its own questions.

    The panel asks for the groups to draw the legend before it asks for the
    values, so an empty comparison has to answer that without raising.
    """
    empty = Comparison(measurement="m", level="cell",
                       frame=pd.DataFrame(columns=["group", "value"]))

    assert empty.groups == ()
    assert empty.counts() == {}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_a_statistics_engine_that_refuses_is_reported_as_not_testable(
        objects, monkeypatch):
    """A failure in the test engine must not take the comparison with it.

    The figure and the numbers are still worth having. What must not happen
    is a comparison that quietly carries no statistics and reads as one that
    was never asked for a test -- so the refusal is written into the record
    where the test would have been.
    """
    from spacr import sp_stats

    def _refuse(*_args, **_kwargs):
        raise RuntimeError("scipy is not speaking to us")

    monkeypatch.setattr(sp_stats, "perform_statistical_tests", _refuse)

    comparison = build(objects, "pathogen_area", level="cell",
                       groups={"g": objects.index[:6]})
    out = gmc.with_statistics(comparison)

    assert out is comparison
    assert len(out.statistics) == 1
    assert out.statistics[0]["Test Name"] == "not testable"
    assert "scipy is not speaking to us" in out.statistics[0]["Why This Test"]


def test_an_assumption_check_that_was_not_run_says_so():
    """``None`` from a check is "not computed", not "passed".

    The line goes on the figure caption and into the saved record. A blank
    would read as an assumption that was tested and held.
    """
    assert gmc._summarise(None) == "not computed"


def test_an_assumption_check_returned_as_a_frame_is_read_like_the_rest():
    """The engine answers with a DataFrame for some checks and dicts for others.

    Both have to reduce to the same one-line verdict, or the caption would
    say "not computed" for a check that ran.
    """
    frame = pd.DataFrame([{"Verdict": "normal", "Test Name": "shapiro",
                           "p-value": 0.4}])

    line = gmc._summarise(frame)

    assert "normal" in line and "shapiro" in line and "0.4" in line
