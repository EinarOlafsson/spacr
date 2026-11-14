"""The measurement compare panel: its report, its figure, the join and saving.

:class:`spacr.qt.widgets.measurement_compare_dialog.MeasurementComparePanel`
makes one claim about one measurement.  Every path here ends in a sentence in
the panel (the report box or the join note) rather than a traceback: a test
that returned only some fields, a figure slot holding a spacer, a join that
raises or matches nothing, and a save that is cancelled or refused.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QFileDialog                           # noqa: E402

from spacr.qt.widgets import measurement_compare_dialog as mcd      # noqa: E402
from spacr.qt.widgets.measurement_compare_dialog import (           # noqa: E402
    MeasurementComparePanel)

pytestmark = pytest.mark.qt

NOT_ATTACHED = ("the dependent variable was asked for and no table carrying "
                "it is attached")


def _objects(n=200, seed=0):
    """``n`` cells over two plates with two measurements and the identity
    columns; the first quarter is shifted so the comparison has an effect."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "cell_area": rng.normal(10.0, 2.0, n),
        "nucleus_area": rng.normal(5.0, 1.0, n),
        "plateID": np.repeat(["p1", "p2"], n // 2),
        "rowID": rng.choice(["r1", "r2"], n),
        "columnID": rng.choice(["c1", "c2", "c3"], n),
        "fieldID": rng.choice(["f1", "f2"], n),
        "object_label": np.arange(n),
    })
    frame.loc[frame.index[: n // 4], "cell_area"] += 3.0
    return frame


def _forty(seed=0):
    """Forty unshifted objects over two plates and four wells."""
    rng = np.random.default_rng(seed)
    n = 40
    columns = {name: rng.normal(10.0, 2.0, n)
               for name in ("cell_area", "nucleus_area")}
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
    """Two hundred cells."""
    return _objects()


@pytest.fixture
def panel(qtbot, objects):
    """A panel comparing the first fifty cells against the rest."""
    widget = MeasurementComparePanel(objects, {"TGGT1_220950": objects.index[:50]})
    qtbot.addWidget(widget)
    return widget


def test_the_report_prints_exactly_the_fields_each_test_returned(
        qtbot, monkeypatch, subtests):
    """``_report``: checks, then the test with its p and effect size, then
    why -- only the fields a statistics row carried, never invented ones."""
    rows: list = []
    monkeypatch.setattr(mcd, "with_statistics",
                        lambda comparison: replace(comparison,
                                                   statistics=list(rows)))
    objects = _forty()
    full = {"Test Name": "Student t-test", "Normality": "Shapiro p=0.30",
            "Equal variance": "Levene p=0.60", "p-value": "0.012",
            "Effect Size": "0.80", "Why This Test": "both sides pass the checks"}
    cases = [
        ("a complete row", [full], [
            "normality: Shapiro p=0.30",
            "equal variance: Levene p=0.60",
            "test: Student t-test · p = 0.012 · effect size = 0.80",
            "why: both sides pass the checks"]),
        ("a test that returned only its name", [{"Test Name": "not testable"}],
         ["test: not testable"]),
        ("two half rows keep their own fields", [
            {"Test Name": "Mann-Whitney", "Normality": "Shapiro p=0.01",
             "p-value": "0.04", "Why This Test": "one side is not normal"},
            {"Test Name": "permutation", "Equal variance": "Levene p=0.20",
             "Effect Size": "0.35"}], [
            "normality: Shapiro p=0.01",
            "test: Mann-Whitney · p = 0.04",
            "why: one side is not normal",
            "equal variance: Levene p=0.20",
            "test: permutation · effect size = 0.35"]),
    ]
    panel = MeasurementComparePanel(objects, {"picked": list(range(20))})
    qtbot.addWidget(panel)
    for case, statistics, expected in cases:
        with subtests.test(msg=case):
            rows[:] = statistics
            panel.refresh()
            lines = panel.report.toPlainText().splitlines()
            n_at = lines.index("n: picked = 4, the rest = 4")
            assert lines[n_at + 1:] == expected


def test_the_figure_slot_is_redrawn_whatever_it_holds(qtbot, objects,
                                                      monkeypatch, subtests):
    """``refresh`` / ``_draw``: a spacer in the slot, a pyqtgraph failure
    and a table with no measurement column at all."""
    with subtests.test(msg="a spacer in the figure slot"):
        panel = MeasurementComparePanel(_forty(), {"picked": list(range(20))})
        qtbot.addWidget(panel)
        first = panel._canvas
        assert panel._figure_holder.itemAt(0).widget() is first
        panel._figure_holder.addStretch(1)
        assert panel._figure_holder.count() == 2
        panel.refresh()
        assert panel._figure_holder.count() == 1
        assert panel._figure_holder.itemAt(0).widget() is panel._canvas
        assert panel._canvas is not first
        assert first.parent() is None

    with subtests.test(msg="pyqtgraph refuses: a matplotlib canvas instead"):
        import spacr.qt.widgets.grouped_plot as gp
        from spacr.qt.widgets.graph_builder import _canvas_class

        class Refuses:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("pyqtgraph is not available")

        panel = MeasurementComparePanel(objects, {"g": objects.index[:50]})
        qtbot.addWidget(panel)
        monkeypatch.setattr(gp, "GroupedPlot", Refuses)
        panel.refresh()
        holder = panel._figure_holder
        assert holder.count() == 1
        assert isinstance(holder.itemAt(0).widget(), _canvas_class())
        monkeypatch.undo()

    with subtests.test(msg="no measurement column"):
        frame = pd.DataFrame({"plateID": ["p1", "p1"], "rowID": ["r1", "r1"],
                              "columnID": ["c1", "c2"],
                              "fieldID": ["f1", "f1"], "object_label": [1, 2]})
        widget = MeasurementComparePanel(frame, {})
        qtbot.addWidget(widget)
        assert widget.refresh() is None
        assert widget.report.toPlainText() == (
            "These objects carry no measurement column to compare.")

    with subtests.test(msg="the heading filter cannot install"):
        from spacr.qt.screens import settings_model

        class NoFilter:
            def __init__(self, *_args):
                raise RuntimeError("no event filter here")

        monkeypatch.setattr(settings_model, "_ApiTooltipFilter", NoFilter)
        from spacr.gene_measurement_compare import HEADING_HELP
        widget = MeasurementComparePanel(objects, {})
        qtbot.addWidget(widget)
        assert widget.headings["level"].toolTip() == HEADING_HELP["level"]
        assert widget.headings["level"].toolTip() != ""
        assert getattr(widget, "_heading_filter", None) is None


def test_the_join_ends_in_a_note_never_a_traceback(panel, objects, tmp_path,
                                                   monkeypatch, subtests):
    """``join_the_tables``, ``_finish_join``, the dependent-variable join and
    closing with a wedged runner."""
    from spacr import dependent_join

    with subtests.test(msg="no database attached"):
        assert panel.join_the_tables() == "no measurements database is attached"
        assert panel.join_button.isVisible() is False

    panel._databases = [str(tmp_path / "measurements.db")]

    with subtests.test(msg="a second press while joining"):
        panel._joining = True
        label = panel.join_button.text()
        assert panel.join_the_tables() == ""
        assert panel.join_button.text() == label
        panel._joining = False

    with subtests.test(msg="a join that raises"):
        seen = {}

        def explode(_objects, _databases, png_list=False):
            raise RuntimeError("that file is not a measurements database")

        monkeypatch.setattr(mcd, "join_measurements", explode)
        monkeypatch.setattr(panel._jobs, "submit",
                            lambda work, done: seen.update(outcome=work()) or True)
        assert panel.join_the_tables() == ""
        assert seen["outcome"] == {
            "error": "that file is not a measurements database"}
        assert panel._finish_join(seen["outcome"]) == (
            "that file is not a measurements database")
        assert panel.join_note.text() == (
            "Could not join: that file is not a measurements database")
        assert panel.join_button.isEnabled() is True
        assert panel.join_button.text() == panel.JOIN_LABEL

    with subtests.test(msg="a join replaces the rows and keeps the groups"):
        wider = objects.copy()
        wider["pathogen_area"] = 1.0
        monkeypatch.setattr(mcd, "join_measurements",
                            lambda o, d, png_list=False: (wider, ""))
        monkeypatch.setattr(panel._jobs, "submit",
                            lambda work, done: done(work()) or True)
        panel.join_the_tables()
        assert panel._joined_once is True
        names = [panel.measurement.itemData(i)
                 for i in range(panel.measurement.count())]
        assert "pathogen_area" in names
        assert list(panel._groups) == ["TGGT1_220950"]

    panel.join_dependent.setChecked(True)
    for case, frame, join, expected_note, expect_pred in [
            ("no dependent table", None, None, NOT_ATTACHED, False),
            ("an empty dependent table", pd.DataFrame({"pred": []}), None,
             NOT_ATTACHED, False),
            ("a dependent join that matches nothing",
             pd.DataFrame({"pred": [0.5]}), "raise",
             "the dependent variable did not join: no identifier column in common",
             False),
            ("a dependent join that works", pd.DataFrame({"pred": [0.5]}),
             "ok", "matched on prcfo", True),
    ]:
        with subtests.test(msg=case):
            panel.set_dependent_frame(frame)
            if join == "raise":
                def refuse(_wide, _frame):
                    raise ValueError("no identifier column in common")
                monkeypatch.setattr(dependent_join, "join", refuse)
            elif join == "ok":
                joined = objects.assign(pred=0.5)
                monkeypatch.setattr(dependent_join, "join",
                                    lambda wide, f: (joined, {"route": "prcfo"}))
                monkeypatch.setattr(dependent_join, "describe",
                                    lambda report: "matched on prcfo")
            wide, note = panel._join_the_dependent_variable(objects)
            assert note == expected_note
            assert ("pred" in wide.columns) is expect_pred
            if not expect_pred:
                assert wide is objects

    with subtests.test(msg="the dependent note follows the join note"):
        panel.set_dependent_frame(None)
        trouble = panel._finish_join(
            {"wide": objects, "trouble": "one plate had no rows"})
        assert trouble == f"one plate had no rows {NOT_ATTACHED}"
        assert panel.join_note.text().endswith(trouble)

    with subtests.test(msg="closing with a runner that will not shut down"):
        panel._joining = True

        def wedged():
            raise RuntimeError("the join runner is wedged")

        monkeypatch.setattr(panel._jobs, "shutdown", wedged)
        panel.close()
        assert panel._joining is False
        assert panel.isVisible() is False


def test_saving_writes_into_the_chosen_folder_or_says_why_not(
        qtbot, objects, panel, tmp_path, monkeypatch, subtests):
    """``save_everything``: no comparison, a cancelled dialog, a refused
    write, an explicit folder and the folder the dialog returns."""
    with subtests.test(msg="no comparison yet"):
        widget = MeasurementComparePanel(objects, {})
        qtbot.addWidget(widget)
        widget._comparison = None
        assert widget.save_everything(str(tmp_path)) == {}
        assert list(tmp_path.iterdir()) == []

    with subtests.test(msg="a cancelled folder dialog"):
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            staticmethod(lambda *a, **k: ""))
        assert panel.save_everything() == {}

    with subtests.test(msg="a save that refuses"):
        before = panel.report.toPlainText()

        def refuse(*_args, **_kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(mcd, "save", refuse)
        assert panel.save_everything("/nowhere") == {}
        assert panel.report.toPlainText() == (
            f"{before}\n\nCould not save: read-only file system")
        monkeypatch.undo()

    for case, folder, via_dialog in [("an explicit folder", "given", False),
                                     ("the dialog's folder", "picked", True)]:
        with subtests.test(msg=case):
            target = tmp_path / folder
            target.mkdir()
            if via_dialog:
                monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                                    staticmethod(lambda *a, **k: str(target)))
            before = panel.report.toPlainText()
            written = panel.save_everything("" if via_dialog else str(target))
            assert len(written) >= 1
            assert sorted(p.name for p in target.iterdir()) == sorted(
                __import__("os").path.basename(str(p))
                for p in written.values())
            assert panel.report.toPlainText() == (
                f"{before}\n\nSaved {len(written)} item(s) to {target}")
