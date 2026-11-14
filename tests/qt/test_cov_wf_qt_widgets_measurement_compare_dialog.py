"""What the compare panel does when the data no longer matches the controls.

Three seams, each one a thing a user sees rather than a thing the code does.
A re-run can drop the measurement the chooser was pointing at; the figure slot
can hold something that is not a figure; and a statistical test can come back
with a name and none of the numbers a full test reports. Each has a
fall-through path that only runs on the awkward input, and each one leaves a
stale figure, a wrong measurement, or a made-up number on screen if it breaks.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.measurement_compare_dialog import MeasurementComparePanel

pytestmark = pytest.mark.qt

#: The two selected object rows out of forty, so every comparison has the
#: annotated side and "the rest" to hold it against.
PICKED = {"picked": list(range(0, 20))}


def _objects(measurements=("cell_area", "nucleus_area"), n=40, seed=0):
    """Object rows carrying ``measurements`` and the well identifiers."""
    rng = np.random.default_rng(seed)
    columns = {name: rng.normal(10.0, 2.0, n) for name in measurements}
    columns.update({
        "plateID": np.repeat(["p1", "p2"], n // 2),
        "rowID": rng.choice(["r1", "r2"], n),
        "columnID": rng.choice(["c1", "c2"], n),
        "fieldID": rng.choice(["f1", "f2"], n),
        "object_label": np.arange(n),
    })
    return pd.DataFrame(columns)


@pytest.fixture
def objects():
    """Forty objects over two plates with two measurements on them."""
    return _objects()


FULL_ROW = {
    "Test Name": "Student t-test",
    "Normality": "Shapiro p=0.30",
    "Equal variance": "Levene p=0.60",
    "p-value": "0.012",
    "Effect Size": "0.80",
    "Why This Test": "both sides pass the checks",
}


def test_a_measurement_the_re_run_dropped_does_not_leave_the_box_empty(
        qtbot, objects):
    """A re-run may not carry the column the chooser was pointing at.

    ``set_data`` restores the user's measurement so a re-run does not throw
    away what they were looking at. When the new montage does not HAVE that
    column there is nothing to restore, and the box must land on a
    measurement the new data actually has -- an index left pointing at the
    old name would offer a measurement that cannot be built, and the panel
    would compare nothing while looking like it was comparing something.
    """
    panel = MeasurementComparePanel(objects, PICKED)
    qtbot.addWidget(panel)
    panel.measurement.setCurrentIndex(1)
    assert panel.measurement.currentData() == "nucleus_area"

    # The column is still there: the choice survives the re-run.
    kept = panel.set_data(_objects(seed=1), PICKED)
    assert panel.measurement.currentData() == "nucleus_area"
    assert kept.measurement == "nucleus_area"

    # The column is gone: the box falls back to what the new data offers.
    lost = panel.set_data(_objects(("cytoplasm_perimeter",), seed=2), PICKED)
    assert [panel.measurement.itemData(i)
            for i in range(panel.measurement.count())] == [
        "cytoplasm_perimeter"]
    assert panel.measurement.currentData() == "cytoplasm_perimeter"
    assert lost.measurement == "cytoplasm_perimeter"


