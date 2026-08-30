"""What the compare panel does when the data no longer matches the controls.

Three seams, each one a thing a user sees rather than a thing the code does.
A re-run can drop the measurement the chooser was pointing at; the figure slot
can hold something that is not a figure; and a statistical test can come back
with a name and none of the numbers a full test reports. Each has a
fall-through path that only runs on the awkward input, and each one leaves a
stale figure, a wrong measurement, or a made-up number on screen if it breaks.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

import spacr.qt.widgets.measurement_compare_dialog as mcd
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


@pytest.fixture
def statistics(monkeypatch):
    """Put chosen test records into every comparison the panel builds.

    The report is the only place the statistics reach the user, and the rows
    that expose its awkward paths -- a test that reported no p-value, no
    effect size and no reason -- are the ones a healthy engine never emits.
    Feeding them in through ``with_statistics`` drives the real
    ``refresh -> _report`` path rather than writing the report by hand.
    """
    rows: list = []

    def hand_them_over(comparison):
        return replace(comparison, statistics=list(rows))

    monkeypatch.setattr(mcd, "with_statistics", hand_them_over)
    return rows


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


def test_a_spacer_in_the_figure_slot_does_not_stop_the_redraw(
        qtbot, objects):
    """The figure slot is emptied of everything, not only of figures.

    A layout item is not always a widget -- a stretch or a nested layout has
    no widget behind it -- and ``takeAt`` hands those over too. If the
    emptying loop assumed a widget it would raise halfway through, leaving
    the previous figure on screen under a panel that believes it redrew:
    the user would be reading last run's plot with this run's caption.
    """
    panel = MeasurementComparePanel(objects, PICKED)
    qtbot.addWidget(panel)
    first_figure = panel._canvas
    assert panel._figure_holder.itemAt(0).widget() is first_figure

    # A widget item AND an item with no widget behind it, in one slot.
    panel._figure_holder.addStretch(1)
    assert panel._figure_holder.count() == 2

    panel.refresh()

    assert panel._figure_holder.count() == 1
    assert panel._figure_holder.itemAt(0).widget() is panel._canvas
    assert panel._canvas is not first_figure
    # The figure that was there was taken out of the layout, not left in it.
    assert first_figure.parent() is None


def test_the_report_names_every_check_and_number_the_test_returned(
        qtbot, objects, statistics):
    """The report is the whole of what the user is told about the test.

    The order is the argument: the assumption checks come first, then the
    test they chose, then why. A p-value printed above the checks reads as a
    decision already made, and a missing effect size turns "significant" into
    a claim with no size attached -- so every field the test returned has to
    appear, in that order.
    """
    statistics[:] = [dict(FULL_ROW)]
    panel = MeasurementComparePanel(objects, PICKED)
    qtbot.addWidget(panel)

    lines = panel.report.toPlainText().splitlines()

    assert "normality: Shapiro p=0.30" in lines
    assert "equal variance: Levene p=0.60" in lines
    assert "test: Student t-test · p = 0.012 · effect size = 0.80" in lines
    assert "why: both sides pass the checks" in lines
    assert lines.index("normality: Shapiro p=0.30") < lines.index(
        "test: Student t-test · p = 0.012 · effect size = 0.80")


def test_a_test_that_returned_no_numbers_has_none_invented_for_it(
        qtbot, objects, statistics):
    """A refused test must not be dressed up as a completed one.

    ``not testable`` comes back with a name and nothing else: no normality
    check, no p-value, no effect size, no reason. Printing "p = None" or an
    empty "effect size =" beside it would read as a result, and a reader
    scanning the caption would take a test that never ran for one that did.
    The same panel is driven with the complete row first, so the strings this
    then asserts are absent are strings it has just been seen to produce.
    """
    statistics[:] = [dict(FULL_ROW)]
    panel = MeasurementComparePanel(objects, PICKED)
    qtbot.addWidget(panel)
    complete = panel.report.toPlainText()
    assert " · p = 0.012" in complete
    assert " · effect size = 0.80" in complete
    assert "normality: " in complete
    assert "why: " in complete

    statistics[:] = [{"Test Name": "not testable"}]
    panel.refresh()
    refused = panel.report.toPlainText()

    assert "test: not testable" in refused.splitlines()
    assert " · p = " not in refused
    assert "effect size" not in refused
    assert "normality" not in refused
    assert "equal variance" not in refused
    assert "why: " not in refused
    # The counts still come through: only the test's own fields dropped out.
    assert "n: picked = 4, the rest = 4" in refused.splitlines()


def test_the_report_still_names_the_test_when_only_some_fields_arrive(
        qtbot, objects, statistics):
    """Half a record is the common case, not the exotic one.

    A Mann-Whitney reports a p-value and an effect size but no equal-variance
    check; a permutation test reports neither p-value nor effect size. The
    report has to print exactly the fields that came back for each row of a
    multi-row comparison, because a row silently dropped for missing one
    field would hide a whole test from the reader.
    """
    statistics[:] = [
        {"Test Name": "Mann-Whitney", "Normality": "Shapiro p=0.01",
         "p-value": "0.04", "Why This Test": "one side is not normal"},
        {"Test Name": "permutation", "Equal variance": "Levene p=0.20",
         "Effect Size": "0.35"},
    ]
    panel = MeasurementComparePanel(objects, PICKED)
    qtbot.addWidget(panel)

    lines = panel.report.toPlainText().splitlines()

    assert "normality: Shapiro p=0.01" in lines
    assert "test: Mann-Whitney · p = 0.04" in lines
    assert "why: one side is not normal" in lines
    assert "equal variance: Levene p=0.20" in lines
    assert "test: permutation · effect size = 0.35" in lines
    # Neither row's missing fields were filled in from the other row's.
    assert "test: Mann-Whitney · p = 0.04 · effect size = 0.35" not in lines
