"""What the gene/measurement comparison does when part of its input is unusable.

The panel behind this module joins measurement tables onto a montage, draws
one group against the rest, and saves the folder a reader is meant to check
the figure from. Every path here is one where something is missing: a database
that will not open, a group that matches no row, a measurement that is NaN on
every cell, a crop that will not encode. None of them may raise -- the panel
has no console -- and none of them may be silent, because a comparison drawn
on a third of the cells and a comparison drawn on all of them look identical
once they are a picture.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import gene_measurement_compare as gmc
from spacr.gene_measurement_compare import REST, Comparison, build


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


def _comparison(values_by_group, *, level="cell"):
    """A comparison built by hand, so a frame no ``build`` produces can be drawn."""
    rows = []
    for group, values in values_by_group.items():
        for value in values:
            rows.append({"group": group, "value": value, "unit": group})
    return Comparison(measurement="pathogen_area", level=level,
                      frame=pd.DataFrame(rows, columns=["group", "value",
                                                        "unit"]))


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


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def test_a_comparison_with_no_finite_values_draws_nothing(tmp_path):
    """A measurement that is NaN on every cell has no figure to draw.

    Drawing empty axes with the group names under them would look like a
    result -- two groups, no difference -- for data that does not exist.
    """
    comparison = _comparison({"g": [np.nan, np.inf], REST: [np.nan]})
    assert len(comparison.frame) and comparison.groups

    assert gmc.plot(comparison) is None
    assert gmc.render_comparison(comparison) == (None, None)


@pytest.mark.parametrize("kind", ["jitter", "jitter_box"])
def test_a_group_with_no_finite_values_is_skipped_not_drawn_at_zero(kind):
    """One empty group among several must not become a point cloud at zero.

    ``np.full(0, i)`` is harmless, but the surrounding arithmetic is not: an
    empty group has to be stepped over so the remaining groups keep their
    positions and their tick labels.
    """
    comparison = _comparison({"g": [np.nan, np.nan],
                              REST: [1.0, 2.0, 3.0, 4.0]})

    figure = gmc.plot(comparison, kind=kind)
    assert figure is not None
    labels = [t.get_text() for t in figure.axes[0].get_xticklabels()]
    assert any(label.startswith("g\n(n=0)") for label in labels)
    assert any(label.startswith(f"{REST}\n(n=4)") for label in labels)

    style = gmc.ComparisonStyle(kind=kind)
    rendered, axes = gmc.render_comparison(comparison, style)
    assert rendered is not None and axes is not None


def test_a_path_writes_the_figure_through_the_one_export_writer(objects,
                                                                tmp_path):
    """Saving from the panel goes through ``save_figure``, like every figure.

    A direct ``savefig`` here would miss the DPI rule and the repaint for
    paper, so a comparison kept from this panel would not match the rest of
    the figures in the same folder.
    """
    comparison = build(objects, "pathogen_area", level="cell",
                       groups={"g": objects.index[:6]})
    target = tmp_path / "comparison.pdf"

    figure = gmc.plot(comparison, str(target))

    assert figure is not None
    assert target.exists() and target.stat().st_size > 0


# ---------------------------------------------------------------------------
# saving the folder
# ---------------------------------------------------------------------------

def test_a_figure_format_that_will_not_write_costs_only_that_format(
        objects, tmp_path, monkeypatch):
    """The data and the record are the deliverable; a format is not.

    A PDF backend that fails must not lose the CSV and the settings beside
    it -- those are what a reader checks the figure against.
    """
    from spacr import plot as plot_module

    def _refuse(*_args, **_kwargs):
        raise OSError("no writable font cache")

    monkeypatch.setattr(plot_module, "save_figure", _refuse)

    comparison = build(objects, "pathogen_area", level="cell",
                       groups={"g": objects.index[:6]})
    written = gmc.save(comparison, str(tmp_path / "out"))

    assert "pdf" not in written and "png" not in written
    assert os.path.isfile(written["data"])
    assert os.path.isfile(written["settings"])


def test_the_saved_settings_hold_values_json_cannot_take(objects, tmp_path):
    """Every setting is recorded, whatever type it arrived as.

    A settings block that dropped the list-valued and object-valued entries
    would record a run that cannot be repeated. Anything JSON will not hold
    is written as its own text rather than omitted.
    """
    comparison = build(objects, "pathogen_area", level="cell",
                       groups={"g": objects.index[:6]})

    written = gmc.save(comparison, str(tmp_path / "out"), settings={
        "channels": [0, 1, 2],
        "shape": (4, 4),
        "nested": {"alpha": np.int64(3)},
        "backend": np.dtype("float32"),
    })

    with open(written["settings"], encoding="utf-8") as handle:
        record = json.load(handle)
    saved = record["regression_settings"]
    assert saved["channels"] == [0, 1, 2]
    assert saved["shape"] == [4, 4], "a tuple is a list once it is JSON"
    assert saved["nested"] == {"alpha": "3"}
    assert saved["backend"] == "float32"


def test_a_well_with_no_crops_makes_no_folder(objects, tmp_path):
    """An empty image list is not a well with pictures in it.

    Creating ``cells/A01/`` and leaving it empty tells a reader the crops
    were saved and are missing, rather than that there were none.
    """
    comparison = build(objects, "pathogen_area", level="cell",
                       groups={"g": objects.index[:6]})
    folder = tmp_path / "out"

    written = gmc.save(comparison, str(folder), images={"p1_r1_c1": []})

    assert "cells" not in written
    assert not (folder / "cells").exists()


def test_a_crop_that_will_not_encode_costs_one_image_not_the_save(objects,
                                                                  tmp_path):
    """One bad crop among several must not lose the good ones or the folder.

    Crops come from wherever the montage got them, and one of the wrong shape
    is not a reason to abandon a save whose point is the figure and the
    numbers.
    """
    comparison = build(objects, "pathogen_area", level="cell",
                       groups={"g": objects.index[:6]})
    folder = tmp_path / "out"

    grey = np.zeros((4, 4), dtype=np.uint8)      # 2-D: widened to three planes
    grey[1, 1] = 200
    unusable = np.zeros((4,), dtype=np.uint8)    # 1-D: nothing to encode

    written = gmc.save(comparison, str(folder),
                       images={"p1_r1_c1": [grey, unusable]})

    saved = sorted(os.listdir(os.path.join(written["cells"], "p1_r1_c1")))
    assert saved == ["0000.png"], "the good crop is there and the bad one is not"
    assert os.path.isfile(written["data"])


# ---------------------------------------------------------------------------
# joining measurement tables
# ---------------------------------------------------------------------------

@pytest.fixture()
def montage_rows():
    """Three montage object rows carrying a ``prcfo`` identity."""
    return pd.DataFrame({
        "prcfo": ["p1_r1_c1_f1_1", "p1_r1_c1_f1_2", "p1_r1_c1_f2_1"],
        "count": [1, 2, 3],
    })


@pytest.fixture()
def wide(montage_rows):
    """What a readable measurement database joins to."""
    return pd.DataFrame({"prcfo": montage_rows["prcfo"],
                         "cell_area": [10.0, 20.0, 30.0]})


def _reader(monkeypatch, fn):
    from spacr import io as spacr_io
    monkeypatch.setattr(spacr_io, "_read_and_join_tables", fn)


def test_a_database_that_joins_to_nothing_is_stepped_over(montage_rows, wide,
                                                          monkeypatch):
    """An empty measurement table is not a failure worth reporting.

    A plate with no measurements yet is an ordinary state during a run. What
    matters is that the other database still joins, and that the note stays
    clean -- a warning here would cry wolf on every partially-measured screen.
    """
    _reader(monkeypatch, lambda path, **kw:
            pd.DataFrame() if "empty" in str(path) else wide)

    out, note = gmc.join_measurements(montage_rows, ["/empty.db", "/good.db"])

    assert note == ""
    assert "cell_area" in out.columns
    assert list(out["cell_area"]) == [10.0, 20.0, 30.0]


def test_a_table_whose_rows_have_no_identity_is_named_in_the_note(
        montage_rows, wide, monkeypatch):
    """Rows that cannot be keyed cannot be joined, and the user needs to know.

    Silently dropping the file would leave the user looking for measurements
    that are in a database they can see, with nothing on screen saying why
    they did not arrive.
    """
    _reader(monkeypatch, lambda path, **kw:
            pd.DataFrame({"x": [1, 2, 3]}) if "anon" in str(path) else wide)

    out, note = gmc.join_measurements(montage_rows, ["/anon.db", "/good.db"])

    assert "/anon.db" in note
    assert "name no object" in note
    assert "cell_area" in out.columns, "the readable database still joined"


def test_tables_that_add_nothing_new_leave_the_rows_alone(montage_rows,
                                                          monkeypatch):
    """A join that would replace columns the montage already has is refused.

    The existing column is the value the montage selected its cells on.
    Overwriting it moves the cells under the user's feet, so the rows come
    back untouched with the reason.
    """
    _reader(monkeypatch, lambda path, **kw:
            pd.DataFrame({"prcfo": montage_rows["prcfo"], "count": [9, 9, 9]}))

    out, note = gmc.join_measurements(montage_rows, ["/same.db"])

    assert out is montage_rows
    assert "add no column" in note
    assert list(out["count"]) == [1, 2, 3]


def test_a_database_that_could_not_be_opened_is_reported_beside_a_good_join(
        montage_rows, wide, monkeypatch):
    """A successful join still has to carry the news about the file that failed.

    This is the case that would otherwise disappear: the measurements are
    there, the panel draws, and the user never learns that one of the two
    plates they selected contributed nothing.
    """
    def _read(path, **_kwargs):
        if "bad" in str(path):
            raise RuntimeError("database is locked")
        return wide

    _reader(monkeypatch, _read)

    out, note = gmc.join_measurements(montage_rows, ["/bad.db", "/good.db"])

    assert "/bad.db" in note
    assert "database is locked" in note
    assert "cell_area" in out.columns
    assert list(out["cell_area"]) == [10.0, 20.0, 30.0]
