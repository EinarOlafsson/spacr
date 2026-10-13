"""Figure menu actions keep working when data or optional services disappear."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QEvent, QPoint  # noqa: E402
from PySide6.QtWidgets import QFileDialog  # noqa: E402

from spacr.figures import bundle, style  # noqa: E402
from spacr.qt.widgets import figure_settings as fs  # noqa: E402

pytestmark = pytest.mark.qt


def test_a_failed_redraw_notification_does_not_break_a_figure_action(
        monkeypatch):
    figure = Figure()
    called = []
    monkeypatch.setattr(figure.canvas, "draw_idle",
                        lambda: called.append("draw"))
    fs._redraw_after(figure, None)
    assert called == ["draw"]

    def refusing_draw():
        raise RuntimeError("canvas already closed")

    monkeypatch.setattr(figure.canvas, "draw_idle", refusing_draw)
    fs._redraw_after(figure, None)

    legacy = []
    fs._redraw_after(figure, lambda: legacy.append("changed"))
    assert legacy == ["changed"]
    fs._redraw_after(figure, lambda: refusing_draw())
    fs._redraw_after(figure, lambda **_kwargs: refusing_draw())


def test_retyping_without_data_or_with_an_unsupported_kind_keeps_the_recipe(
        monkeypatch):
    figure = Figure()
    assert not fs._retype(figure, "violin")

    frame = pd.DataFrame({"group": ["a", "b"], "value": [1.0, 2.0]})
    bundle._register_figure_data(figure, frame, x="group", y="value",
                                 kind="bar")
    original = dict(figure._spacr_spec)
    original_draw = bundle._draw
    monkeypatch.setattr(bundle, "_draw",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            ValueError("the data cannot be drawn")))
    assert not fs._retype(figure, "violin")
    assert figure._spacr_spec == original
    monkeypatch.setattr(bundle, "_draw", original_draw)

    monkeypatch.setattr(style, "_apply_user_style",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            RuntimeError("style unavailable")))
    changed = []
    assert fs._retype(figure, "bar", lambda **kwargs: changed.append(kwargs))
    assert figure._spacr_spec["kind"] == "bar"
    assert changed == [{"preview": False}]


def test_statistics_ignore_unusable_probabilities_but_keep_valid_pairs():
    table = pd.DataFrame([
        {"test_stage": "pairwise", "test_name": "invalid",
         "groups": "a vs b", "correction": "", "p_adjusted": float("nan"),
         "p_value": float("nan")},
        {"test_stage": "pairwise", "test_name": "Welch",
         "groups": "a vs b", "correction": "", "p_adjusted": float("nan"),
         "p_value": 0.001},
        {"test_stage": "pairwise", "test_name": "Welch",
         "groups": "all", "correction": "", "p_adjusted": 0.001,
         "p_value": 0.001},
    ])
    note, brackets = fs._annotations_from(table)
    assert note == "Welch: p = 0.001"
    assert brackets == [{"pair": ["a", "b"], "label": "**"}]


def test_a_replacement_recipe_redraws_the_canvas_without_swapping_it(
        monkeypatch):
    figure = Figure()
    figure._spacr_data = object()
    redrawn = []
    canvas = SimpleNamespace(figure=figure,
                             draw_idle=lambda: redrawn.append("draw"))
    monkeypatch.setattr(fs, "_retype",
                        lambda fig, kind, on_change: redrawn.append(kind))
    replacement = SimpleNamespace(
        _spacr_replot={"graph_type": "jitter_box"})

    fs._canvas_changed(canvas, replacement)

    assert figure._spacr_replot == replacement._spacr_replot
    assert figure._spacr_data is None
    assert redrawn == ["box_strip", "draw"]

    canvas.draw_idle = lambda: (_ for _ in ()).throw(
        RuntimeError("canvas already deleted"))
    fs._canvas_changed(canvas)


def test_local_figure_writer_survives_missing_preferences_and_failed_write(
        tmp_path, monkeypatch):
    figure = Figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])
    monkeypatch.setitem(sys.modules, "spacr.plot", None)
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", None)
    destination = tmp_path / "fallback.svg"

    assert fs.save_figure_as(None, figure, str(destination)) == str(destination)
    assert destination.exists() and destination.stat().st_size > 0

    blocked = tmp_path / "blocker"
    blocked.write_text("a file")
    assert fs.save_figure_as(None, figure, str(blocked / "bad.svg")) == ""


def test_statistics_dialog_keeps_subject_pairs_when_annotation_fails(
        qapp, monkeypatch):
    frame = pd.DataFrame({
        "subject": [1, 2, 3, 1, 2, 3],
        "group": ["control"] * 3 + ["treated"] * 3,
        "value": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
    })
    figure = Figure()
    figure.subplots().boxplot([[1, 2, 3], [2, 3, 4]])
    figure._spacr_replot = {
        "df": frame, "grouping_column": "group", "data_column": "value",
        "graph_type": "box",
    }
    changed = []
    dialog = fs._StatisticsDialog(
        figure, on_change=lambda **kwargs: changed.append(kwargs))
    try:
        assert dialog.pair.findData("subject") >= 0
        dialog.pair.setCurrentIndex(dialog.pair.findData("subject"))

        def unavailable(*_args):
            raise RuntimeError("artist refused annotation")

        monkeypatch.setattr(bundle, "_annotate", unavailable)
        dialog._apply()

        assert figure._spacr_data is frame
        assert figure._spacr_spec["stats"]["pair"] == "subject"
        assert changed == [{"preview": False}]
    finally:
        dialog.deleteLater()


def test_statistics_dialog_offers_a_correction_without_its_catalogue(
        qapp, monkeypatch):
    figure = Figure()
    frame = pd.DataFrame({"group": ["a", "a", "b", "b"],
                          "value": [1.0, 2.0, 3.0, 4.0]})
    bundle._register_figure_data(figure, frame, x="group", y="value",
                                 kind="box")
    monkeypatch.setitem(sys.modules, "spacr.multiple_testing", None)

    dialog = fs._StatisticsDialog(figure)
    try:
        assert [dialog.correction.itemText(index)
                for index in range(dialog.correction.count())] == ["fdr_bh"]
    finally:
        dialog.deleteLater()


def test_statistics_can_be_saved_before_a_figure_has_axes(qapp):
    figure = Figure()
    frame = pd.DataFrame({"group": ["a", "a", "b", "b"],
                          "value": [1.0, 2.0, 3.0, 4.0]})
    bundle._register_figure_data(figure, frame, x="group", y="value",
                                 kind="box")
    dialog = fs._StatisticsDialog(figure)
    try:
        dialog._apply()
        assert figure._spacr_spec["stats"]["show"] is True
        assert not figure.axes
    finally:
        dialog.deleteLater()


def test_zip_chooser_handles_cancel_success_and_write_failure(
        qapp, tmp_path, monkeypatch):
    figure = Figure()
    figure.subplots().plot([0, 1], [1, 2])
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *_args: ("", ""))
    assert fs._save_zip_dialog(None, figure) == ""

    path = tmp_path / "figure.zip"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *_args: (str(path), ""))
    assert fs._save_zip_dialog(None, figure) == str(path)
    assert path.is_file()

    def unavailable(*_args, **_kwargs):
        raise OSError("archive is read only")

    monkeypatch.setattr(bundle, "_save_zip", unavailable)
    assert fs._save_zip_dialog(None, figure) == ""


def test_menu_filter_and_editor_survive_a_closed_canvas(qapp, monkeypatch):
    def unavailable():
        raise RuntimeError("canvas was deleted")

    owner = SimpleNamespace(contextMenuPolicy=unavailable)
    assert fs._FigureMenuFilter().eventFilter(
        owner, QEvent(QEvent.ContextMenu)) is False

    redraws = []

    class Editor:
        def __init__(self, figure, parent, on_change):
            self.on_change = on_change

        def exec(self):
            self.on_change()

    monkeypatch.setattr(fs, "FigureSettingsDialog", Editor)
    figure = Figure()
    fs._open_editor(figure, None,
                    lambda **kwargs: redraws.append(kwargs))
    assert redraws == [{"preview": False}]


def test_attaching_the_menu_twice_installs_one_filter(qapp, monkeypatch):
    monkeypatch.setattr(fs, "_MENU_FILTER", None)
    filters = []
    canvas = SimpleNamespace(
        installEventFilter=lambda event_filter: filters.append(event_filter))
    fs._attach_figure_menu(canvas)
    fs._attach_figure_menu(canvas)
    assert len(filters) == 1

    positions = []
    fs._exec_menu(SimpleNamespace(exec=lambda point: positions.append(point)),
                  QPoint(3, 4))
    assert positions == [QPoint(3, 4)]

    draws = []
    replacement = SimpleNamespace(_spacr_replot={"graph_type": "box"})
    fs._canvas_changed(SimpleNamespace(figure=None,
                                       draw_idle=lambda: draws.append(True)),
                       replacement)
    assert draws == [True]
