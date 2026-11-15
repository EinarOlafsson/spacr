"""Loading a table into the Outliers screen, and the ways that goes wrong.

Everything here is about the screen's edges rather than its statistics: the
file picker, the table picker inside a measurement database, a scan asked for
with nothing ticked, and an export to a place that cannot be written. Each one
has to end in a sentence in the status line -- the screen has no modal, because
a dialog nobody can dismiss is how a headless run hangs.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.outliers import OutliersScreen, _cell
from spacr.qt.widgets.outlier_model import OBJECT_COLUMNS

pytestmark = pytest.mark.qt


def planted_plate(seed: int = 3, bad_well: int = 3, shift: float = 1.4,
                  n_wells: int = 8, per_well: int = 25) -> pd.DataFrame:
    """Eight wells, one of them shifted; small enough to draw entirely."""
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(n_wells):
        factor = shift if well == bad_well else 1.0
        area = factor * rng.lognormal(0.0, 0.2, per_well)
        perimeter = rng.lognormal(0.0, 0.2, per_well)
        for i in range(per_well):
            rows.append(("p1", "r1", f"c{well + 1}", "f1", i,
                         area[i], perimeter[i]))
    return pd.DataFrame(rows, columns=[
        "plateID", "rowID", "columnID", "fieldID", "object_label",
        "cell_area", "cell_perimeter"])


@pytest.fixture
def screen(qtbot):
    """The screen with its jobs inline, so a scan is finished when it returns."""
    widget = OutliersScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_choosing_a_table_from_the_file_dialog_loads_it(screen, tmp_path,
                                                        monkeypatch):
    """The Open button is the same path as ``load_path``, with a dialog."""
    csv = tmp_path / "objects.csv"
    planted_plate().to_csv(csv, index=False)
    monkeypatch.setattr(
        "spacr.qt.screens.outliers.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(csv), "")))

    screen.choose_table()

    assert screen.result is not None
    assert screen.frame is not None and len(screen.frame) == 200


def test_a_cancelled_file_dialog_loads_nothing(screen, monkeypatch):
    """Dismissing the picker leaves the screen exactly as it was."""
    monkeypatch.setattr(
        "spacr.qt.screens.outliers.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: ("", "")))
    before = screen._source.text()

    screen.choose_table()

    assert screen.frame is None
    assert screen.result is None
    assert screen._source.text() == before


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def test_the_flagged_table_is_offered_with_its_flag_columns(screen):
    """``objects_frame`` is the whole table plus the engine's columns.

    It is what the export writes, so a host embedding this screen can hand the
    same frame on without re-running anything.
    """
    frame = planted_plate()
    assert screen.objects_frame() is None

    screen.set_frame(frame)
    objects = screen.objects_frame()

    assert len(objects) == len(frame)
    for name in OBJECT_COLUMNS.values():
        assert name in objects.columns
    assert screen.active_jobs() == 0, "an inline scan leaves no worker behind"


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def test_a_missing_number_is_an_empty_cell():
    """NaN is drawn as nothing at all.

    A well with too few objects has no score, and a table full of the string
    "nan" reads as a computed value rather than as a gap.
    """
    assert _cell(float("nan")) == ""
    assert _cell(np.nan) == ""
    assert _cell(1.5) == "1.5"
    assert _cell(True) == "yes"
    assert _cell(None) == ""
