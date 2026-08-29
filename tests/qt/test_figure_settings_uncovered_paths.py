"""The figure settings paths that only open when something is missing.

Every branch here is a fallback: a colour matplotlib will not read, a figure
that will not pickle, a preference store that is not there, an artist that
refuses the style it is handed, a sidecar that cannot be written. The dialog
promises to open and to keep working in all of them, so each one is driven
through the real widget and checked by what the user would see -- the value a
control opens at, the colour on a swatch, the file that did or did not appear.
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from matplotlib.lines import Line2D  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QWidget,
)

from spacr.qt.widgets import figure_settings as fs  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _config_sandbox(tmp_path, monkeypatch):
    """Keep every preference read and write inside the test's own directory."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    os.makedirs(tmp_path / "config", exist_ok=True)


@pytest.fixture()
def figure():
    """A small figure with one labelled line."""
    fig = plt.figure(figsize=(4.0, 3.0))
    axis = fig.add_subplot(111)
    axis.plot([0, 1, 2], [1, 2, 3], label="one")
    axis.set_title("only")
    yield fig
    plt.close(fig)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class RefusingLine(Line2D):
    """A line that stops answering questions about its own style.

    Constructed like any other line -- matplotlib sets a colour during
    ``__init__`` -- and then switched into refusing whatever it is asked,
    which is the state the dialog's fallbacks exist for.
    """

    refuse = False

    def get_color(self):
        if self.refuse:
            raise RuntimeError("this line will not report a colour")
        return super().get_color()

    def set_color(self, colour):
        if self.refuse:
            raise RuntimeError("this line will not take a colour")
        return super().set_color(colour)

    def get_linewidth(self):
        if self.refuse:
            raise RuntimeError("this line will not report a width")
        return super().get_linewidth()

    def get_markersize(self):
        if self.refuse:
            raise RuntimeError("this line will not report a marker size")
        return super().get_markersize()

    def get_alpha(self):
        if self.refuse:
            raise RuntimeError("this line will not report an opacity")
        return super().get_alpha()


def _refusing_line(axis, label="awkward"):
    line = RefusingLine([0, 1], [0, 1], label=label)
    axis.add_line(line)
    line.refuse = True
    return line


def _rows(root):
    """``{label text: field widget}`` for every form row under ``root``."""
    found = {}
    for form in root.findChildren(QFormLayout):
        for index in range(form.rowCount()):
            label = form.itemAt(index, QFormLayout.LabelRole)
            field = form.itemAt(index, QFormLayout.FieldRole)
            if label is None or field is None:
                continue
            text = ""
            if label.widget() is not None and isinstance(label.widget(), QLabel):
                text = label.widget().text()
            if text and field.widget() is not None:
                found.setdefault(text, field.widget())
    return found


def _row(root, label):
    rows = _rows(root)
    assert label in rows, f"no {label!r} row; got {sorted(rows)}"
    return rows[label]


def _button_named(root, text):
    for button in root.findChildren(QPushButton):
        if button.text() == text:
            return button
    raise AssertionError(f"no {text!r} button")


def _action_named(menu, text):
    for action in menu.actions():
        if action.text() == text:
            return action
    for action in menu.actions():
        if action.menu() is not None:
            try:
                return _action_named(action.menu(), text)
            except AssertionError:
                continue
    raise AssertionError(f"no {text!r} action")


def _hex(colour):
    return matplotlib.colors.to_hex(colour)


# ---------------------------------------------------------------------------
# _as_hex and the colour button
# ---------------------------------------------------------------------------

def test_a_colour_matplotlib_cannot_read_falls_back_to_the_given_default():
    """An unreadable spec returns the fallback rather than raising."""
    assert fs._as_hex(object()) == "#1f77b4"
    assert fs._as_hex(object(), fallback="#abcdef") == "#abcdef"
    # The specs it CAN read still come back converted, so the fallback is the
    # exception and not the rule.
    assert fs._as_hex(np.array([[1.0, 0.0, 0.0, 1.0]])) == "#ff0000"


def test_cancelling_the_colour_picker_leaves_the_swatch_and_the_caller_alone(
        qapp, monkeypatch):
    """A dismissed picker returns an invalid colour, which changes nothing."""
    picked = []
    monkeypatch.setattr(fs, "pick_colour", lambda *a, **k: QColor())

    button = fs._colour_button("#123456", picked.append)
    button.click()

    assert picked == [], "a cancelled picker must not report a choice"
    assert button.text() == "#123456"


# ---------------------------------------------------------------------------
# opening and cancelling the dialog when something is unavailable
# ---------------------------------------------------------------------------

def test_a_figure_that_will_not_pickle_still_opens_and_still_closes(qapp,
                                                                   figure):
    """No snapshot means Cancel has nothing to put back, and says so quietly."""
    figure._spacr_render_hook = lambda: None      # a closure will not pickle

    dialog = fs.FigureSettingsDialog(figure)
    try:
        assert dialog._snapshot is None
        figure.set_size_inches(9.0, 7.0)
        dialog.reject()
    finally:
        dialog.deleteLater()

    assert tuple(figure.get_size_inches()) == pytest.approx((9.0, 7.0)), (
        "with nothing captured there is nothing to restore")


def test_a_snapshot_that_cannot_be_rebuilt_leaves_the_figure_standing(qapp,
                                                                     figure):
    """Restoration is best-effort: a broken snapshot must not take the figure."""
    dialog = fs.FigureSettingsDialog(figure)
    try:
        dialog._snapshot = b"this is not a pickle"
        figure.set_size_inches(9.0, 7.0)
        dialog.reject()
    finally:
        dialog.deleteLater()

    assert figure.axes, "a failed restore must not leave a cleared figure"
    assert figure.axes[0].get_title() == "only"
    assert tuple(figure.get_size_inches()) == pytest.approx((9.0, 7.0))


def test_the_dialog_opens_when_the_queue_cannot_report_a_text_size(
        qapp, figure, monkeypatch):
    """An older figure_queue with no override reader opens at "no override"."""
    from spacr.qt.widgets import figure_queue

    monkeypatch.delattr(figure_queue, "figure_text_size_override")

    dialog = fs.FigureSettingsDialog(figure)
    try:
        assert dialog._text_size_at_open == 0
        assert isinstance(_row(dialog, "All text size"), QSpinBox)
    finally:
        dialog.deleteLater()


def test_cancelling_survives_a_queue_that_cannot_store_a_text_size(
        qapp, figure, monkeypatch):
    """Cancel still closes when the per-figure size cannot be put back."""
    from spacr.qt.widgets import figure_queue

    dialog = fs.FigureSettingsDialog(figure)
    monkeypatch.delattr(figure_queue, "set_figure_text_size_override")
    try:
        dialog.reject()
    finally:
        dialog.deleteLater()

    assert figure.axes[0].get_title() == "only"


def test_the_umap_tab_is_absent_when_the_umap_module_will_not_import(
        qapp, figure, monkeypatch):
    """A payload with no support module leaves the dialog without the tab."""
    figure._spacr_umap_payload = {"settings": {"n_neighbors": 15}}
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.umap_figure_settings",
                        None)

    dialog = fs.FigureSettingsDialog(figure)
    try:
        titles = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
        assert "Image UMAP" not in titles
        assert dialog.umap_values() == {}
    finally:
        dialog.deleteLater()


def test_a_umap_setting_for_the_next_run_does_not_redraw_this_figure(
        qapp, figure):
    """Only the tiers that change the picture cost a render."""
    pytest.importorskip("spacr.qt.widgets.umap_figure_settings")
    figure._spacr_umap_payload = {"settings": {"metric": "euclidean"}}

    dialog = fs.FigureSettingsDialog(figure)
    try:
        titles = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
        assert "Image UMAP" in titles
        dialog._redraw.stop()

        values = dict(dialog.umap_values())
        values["metric"] = "manhattan"          # read by the next embedding
        dialog._on_umap_changed(values)

        assert not dialog._redraw.isActive(), (
            "a setting the drawn figure cannot show must not re-render it")
        assert dialog._umap_applied["metric"] == "manhattan", (
            "it is still remembered for the run that will use it")
    finally:
        dialog.deleteLater()


# ---------------------------------------------------------------------------
# the Figure tab's ink defaults
# ---------------------------------------------------------------------------

def test_a_figure_with_no_axes_opens_the_ink_controls_on_black(qapp):
    """With nothing to read a colour off, both inks open at black."""
    figure = plt.figure()
    try:
        dialog = fs.FigureSettingsDialog(figure)
        try:
            assert _row(dialog, "Line colour").text() == "#000000"
            assert _row(dialog, "Font colour").text() == "#000000"
        finally:
            dialog.deleteLater()
    finally:
        plt.close(figure)


def test_an_axis_that_will_not_report_its_ink_still_opens_the_dialog(
        qapp, figure, monkeypatch):
    """Both colour reads are guarded, and the line ink falls back to the font."""
    def refuse(*_a, **_k):
        raise RuntimeError("unreadable colour spec")

    axis = figure.axes[0]
    axis.xaxis.label.set_color("#00ff00")
    monkeypatch.setattr(axis.xaxis.label, "get_color", refuse)
    for spine in axis.spines.values():
        monkeypatch.setattr(spine, "get_edgecolor", refuse)

    dialog = fs.FigureSettingsDialog(figure)
    try:
        assert _row(dialog, "Font colour").text() == "#000000"
        assert _row(dialog, "Line colour").text() == "#000000", (
            "the line ink falls back to whatever the font ink resolved to")
    finally:
        dialog.deleteLater()


# ---------------------------------------------------------------------------
# the wheel filter
# ---------------------------------------------------------------------------

def test_an_unfocused_spin_box_does_not_take_the_scroll_wheel(qapp, figure):
    """Scrolling the panel scrolls it instead of editing what is under it."""
    dialog = fs.FigureSettingsDialog(figure)
    try:
        dpi = _row(dialog, "DPI")
        dpi.clearFocus()
        before = dpi.value()
        wheel = QWheelEvent(
            QPointF(5.0, 5.0), QPointF(5.0, 5.0), QPoint(0, 0),
            QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
            Qt.NoScrollPhase, False)

        consumed = QApplication.sendEvent(dpi, wheel)

        assert consumed, "the filter must swallow the event"
        assert dpi.value() == before, "an unfocused spin box must not change"
        assert not wheel.isAccepted()
    finally:
        dialog.deleteLater()


# ---------------------------------------------------------------------------
# the Axes tab
# ---------------------------------------------------------------------------

def test_a_scale_the_dialog_does_not_offer_leaves_the_axis_on_it(qapp):
    """An unlisted scale is shown as the first entry and is not applied back."""
    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot([0.1, 0.5, 0.9], [0.1, 0.5, 0.9])
    axis.set_yscale("logit")
    try:
        dialog = fs.FigureSettingsDialog(figure)
        try:
            combo = _row(dialog, "Y scale")
            assert isinstance(combo, QComboBox)
            assert combo.currentText() == "linear", (
                "a scale outside the offered set cannot be preselected")
            assert axis.get_yscale() == "logit", (
                "building the tab must not rewrite the axis")
        finally:
            dialog.deleteLater()
    finally:
        plt.close(figure)


def test_hiding_top_and_right_is_a_no_op_on_an_axes_without_them(qapp):
    """A polar axes has no top or right spine, and the control must not raise."""
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="polar")
    axis.plot([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    try:
        dialog = fs.FigureSettingsDialog(figure)
        try:
            check = _row(dialog, "Hide top/right")
            assert isinstance(check, QCheckBox)
            assert not check.isChecked()

            before = {name: spine.get_visible()
                      for name, spine in axis.spines.items()}

            check.setChecked(True)

            assert set(axis.spines) == {"polar", "start", "end", "inner"}
            assert {name: spine.get_visible()
                    for name, spine in axis.spines.items()} == before, (
                "there is no top or right spine to hide")
        finally:
            dialog.deleteLater()
    finally:
        plt.close(figure)


def test_moving_the_legend_position_while_it_is_off_creates_no_legend(qapp,
                                                                     figure):
    """The position control must not switch a legend on behind the checkbox."""
    dialog = fs.FigureSettingsDialog(figure)
    try:
        assert figure.axes[0].get_legend() is None
        assert not _row(dialog, "Legend").isChecked()

        _row(dialog, "Legend position").setCurrentText("lower left")

        assert figure.axes[0].get_legend() is None
    finally:
        dialog.deleteLater()


def test_turning_the_legend_on_with_nothing_to_name_leaves_it_absent(qapp):
    """No handles and no legend left: the switch has nothing to build from."""
    figure = plt.figure()
    axis = figure.add_subplot(111)
    line, = axis.plot([0, 1], [0, 1], label="one")
    axis.legend()
    try:
        dialog = fs.FigureSettingsDialog(figure)
        redraws = []
        dialog._on_change = lambda preview=True: redraws.append(preview)
        try:
            # The figure is redrawn beneath the open dialog and comes back
            # with neither a legend nor anything to label.
            axis.get_legend().remove()
            line.set_label("_hidden")

            legend_on = _row(dialog, "Legend")
            assert legend_on.isChecked(), "it opened on a figure with a legend"
            legend_on.setChecked(False)
            legend_on.setChecked(True)

            assert axis.get_legend() is None
            dialog._redraw.stop()
            dialog._redraw_now()
            assert redraws, "the control still asks for a redraw"
        finally:
            dialog.deleteLater()
    finally:
        plt.close(figure)


def test_a_line_that_will_not_report_its_style_gets_the_safe_defaults(
        qapp, monkeypatch):
    """Each per-series control opens at its documented fallback instead."""
    monkeypatch.setattr(fs, "pick_colour", lambda *a, **k: QColor("#ff0000"))
    figure = plt.figure()
    axis = figure.add_subplot(111)
    line = _refusing_line(axis)
    try:
        dialog = fs.FigureSettingsDialog(figure)
        try:
            assert _row(dialog, "  Colour").text() == "#1f77b4"
            assert _row(dialog, "  Line width").value() == pytest.approx(1.0)
            assert _row(dialog, "  Marker size").value() == pytest.approx(6.0)
            assert _row(dialog, "  Opacity").value() == pytest.approx(1.0)

            # And choosing a colour for it is swallowed rather than raised.
            _row(dialog, "  Colour").click()
            line.refuse = False
            assert _hex(line.get_color()) != "#ff0000", (
                "the line refused the colour, so it cannot have taken it")
        finally:
            dialog.deleteLater()
    finally:
        plt.close(figure)


def test_a_palette_skips_the_series_that_will_not_take_a_colour(qapp):
    """Past the detail limit one rule styles them all, minus the refusers."""
    figure = plt.figure()
    axis = figure.add_subplot(111)
    for index in range(fs.FigureSettingsDialog.SERIES_DETAIL_LIMIT):
        axis.plot([0, 1], [index, index + 1], label=f"series {index}")
    refuser = _refusing_line(axis, label="refuser")
    try:
        dialog = fs.FigureSettingsDialog(figure)
        try:
            palette = _row(dialog, "Palette")
            palette.setCurrentText("Set1")

            coloured = [_hex(line.get_color()) for line in axis.lines
                        if line is not refuser]
            assert len(set(coloured)) > 1, "the palette reached the series"
            refuser.refuse = False
            assert _hex(refuser.get_color()) not in ("",), (
                "the refusing line kept whatever colour it had")

            # The size and outline rules reach every artist in the set.
            _row(dialog, "Point size (all)").setValue(64.0)
            _row(dialog, "Opacity (all)").setValue(0.5)
            _row(dialog, "Outline width (all)").setValue(2.0)
            assert axis.lines[0].get_markersize() == pytest.approx(8.0)
            assert axis.lines[0].get_alpha() == pytest.approx(0.5)
            assert axis.lines[0].get_linewidth() == pytest.approx(2.0)
        finally:
            dialog.deleteLater()
    finally:
        plt.close(figure)


# ---------------------------------------------------------------------------
# the Statistics tab
# ---------------------------------------------------------------------------

def test_the_correction_choice_falls_back_when_the_module_is_absent(
        qapp, figure, monkeypatch):
    """Without the corrections table the tab still offers Benjamini-Hochberg."""
    figure._spacr_groups = {"a": [1.0, 2.0, 3.0, 4.0],
                            "b": [2.0, 3.0, 4.0, 5.0]}
    monkeypatch.setitem(sys.modules, "spacr.multiple_testing", None)

    dialog = fs.FigureSettingsDialog(figure)
    try:
        combo = _row(dialog, "Correct across pairs")
        assert [combo.itemText(i) for i in range(combo.count())] == ["fdr_bh"]
        assert dialog._stats_state["correction"] == "fdr_bh"
    finally:
        dialog.deleteLater()


# ---------------------------------------------------------------------------
# apply_line_colour / apply_font_colour
# ---------------------------------------------------------------------------

def test_a_line_that_refuses_the_ink_is_skipped_and_the_rest_are_painted():
    """One awkward artist must not cost the others their colour."""
    figure = plt.figure()
    axis = figure.add_subplot(111)
    good, = axis.plot([0, 1], [0, 1])
    _refusing_line(axis)
    try:
        touched = fs.apply_line_colour(figure, "#ff0000")

        assert _hex(good.get_color()) == "#ff0000"
        spines = len(axis.spines)
        assert touched == 1 + spines, (
            "the refusing line is not counted, the rest are")
    finally:
        plt.close(figure)


def test_ticks_that_refuse_the_ink_do_not_stop_the_line_colour(monkeypatch):
    """The tick pass is separate and its failure is contained."""
    figure = plt.figure()
    axis = figure.add_subplot(111)
    line, = axis.plot([0, 1], [0, 1])
    monkeypatch.setattr(axis, "tick_params", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("these ticks will not take a colour")))
    try:
        touched = fs.apply_line_colour(figure, "#0000ff")

        assert _hex(line.get_color()) == "#0000ff"
        assert touched == 1 + len(axis.spines)
    finally:
        plt.close(figure)


def test_text_that_refuses_the_ink_is_skipped_and_the_rest_are_painted(
        monkeypatch):
    """A single unpaintable text object is dropped from the count, not fatal."""
    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.set_title("a title")
    axis.set_xlabel("x")
    monkeypatch.setattr(axis.title, "set_color", lambda *_a, **_k: (
        _ for _ in ()).throw(RuntimeError("this text will not take a colour")))
    try:
        touched = fs.apply_font_colour(figure, "#00ff00")

        assert _hex(axis.xaxis.label.get_color()) == "#00ff00"
        assert touched == len(fs._every_text(figure)) - 1
    finally:
        plt.close(figure)


def test_ticks_that_refuse_the_ink_do_not_stop_the_font_colour(monkeypatch):
    """Same containment on the font side of the two controls."""
    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.set_xlabel("x")
    monkeypatch.setattr(axis, "tick_params", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("these ticks will not take a colour")))
    try:
        touched = fs.apply_font_colour(figure, "#123456")

        assert _hex(axis.xaxis.label.get_color()) == "#123456"
        assert touched > 0
    finally:
        plt.close(figure)


def test_following_the_theme_without_a_preference_store_gives_black(
        monkeypatch):
    """With no store to read, both inks resolve to black rather than failing."""
    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.set_xlabel("x")
    axis.xaxis.label.set_color("#ff00ff")
    for spine in axis.spines.values():
        spine.set_edgecolor("#ff00ff")
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", None)
    try:
        fs.figure_follows_the_theme(figure)

        assert _hex(axis.xaxis.label.get_color()) == "#000000"
        assert all(_hex(spine.get_edgecolor()) == "#000000"
                   for spine in axis.spines.values())
    finally:
        plt.close(figure)


# ---------------------------------------------------------------------------
# graph-style files
# ---------------------------------------------------------------------------

def test_the_style_dictionary_is_empty_when_the_store_cannot_be_read(
        monkeypatch):
    """No preference module means no overrides, not an exception."""
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", None)

    data = fs.graph_style_as_dict()

    assert data["spacr_style_kind"] == fs.GRAPH_STYLE_FILE_KIND
    assert data["general"] == {}
    assert data["per_graph"] == {}


def test_only_the_half_that_was_not_supplied_is_read_from_the_store(
        monkeypatch):
    """A caller supplying one half keeps it; the other comes from the store."""
    reads = []

    class FakePreferences:
        @staticmethod
        def get_figure_style():
            reads.append("general")
            return {"palette": "deep"}

        @staticmethod
        def get_figure_style_per_graph():
            reads.append("per_graph")
            return {"volcano": {"palette": "muted"}}

    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", FakePreferences)

    supplied_general = fs.graph_style_as_dict(general={"palette": "bright"})
    assert supplied_general["general"] == {"palette": "bright"}
    assert supplied_general["per_graph"] == {"volcano": {"palette": "muted"}}
    assert reads == ["per_graph"]

    reads.clear()
    supplied_per_graph = fs.graph_style_as_dict(per_graph={"qq": {"dpi": 300}})
    assert supplied_per_graph["general"] == {"palette": "deep"}
    assert supplied_per_graph["per_graph"] == {"qq": {"dpi": 300}}
    assert reads == ["general"]


def test_cancelling_the_style_save_dialog_writes_nothing(qapp, tmp_path,
                                                         monkeypatch):
    """An empty path from the chooser is a cancel, and nothing is written."""
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    menu = QMenu()
    try:
        fs.add_graph_style_file_entries(menu)

        _action_named(menu, "Save graph style…").trigger()

        assert list(tmp_path.glob("*.json")) == []
    finally:
        menu.deleteLater()


def test_loading_a_style_without_a_callback_still_applies_it(qapp, tmp_path,
                                                            monkeypatch):
    """The notify step is optional; the style is applied either way."""
    path = tmp_path / "house.json"
    path.write_text(json.dumps({
        "spacr_style_kind": fs.GRAPH_STYLE_FILE_KIND,
        "general": {"palette": "muted"},
        "per_graph": {"volcano": {"dpi": 400}},
    }))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    applied = {}
    monkeypatch.setattr(fs, "apply_graph_style",
                        lambda general, per_graph: applied.update(
                            general=general, per_graph=per_graph))
    menu = QMenu()
    try:
        fs.add_graph_style_file_entries(menu, on_change=None)

        _action_named(menu, "Load graph style…").trigger()

        assert applied["general"] == {"palette": "muted"}
        assert applied["per_graph"] == {"volcano": {"dpi": 400}}
    finally:
        menu.deleteLater()


# ---------------------------------------------------------------------------
# reading a frame back off the artists
# ---------------------------------------------------------------------------

def test_ticks_with_no_text_name_nothing():
    """Before a draw the tick labels are empty, and none becomes a name."""
    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.plot([0, 1, 2], [1, 2, 3])
    axis.set_xticks([0, 1, 2], labels=["", "  ", ""])
    try:
        assert fs._tick_labels(axis) == {}
    finally:
        plt.close(figure)


def test_points_that_are_not_finite_are_left_out_of_the_derived_frame():
    """A NaN in either the scatter or the line is dropped, the rest is kept."""
    figure = plt.figure()
    axis = figure.add_subplot(111)
    axis.scatter([0.0, 1.0, 2.0], [1.0, math.nan, 3.0])
    axis.plot([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, math.nan, 4.0], marker="o")
    try:
        pairs = fs._pairs_from_axes(axis)

        values = [value for _group, value in pairs]
        assert not any(math.isnan(value) for value in values)
        assert sorted(values) == [1.0, 1.0, 2.0, 3.0, 4.0]
    finally:
        plt.close(figure)


# ---------------------------------------------------------------------------
# the right-click menu
# ---------------------------------------------------------------------------

def test_a_menu_built_without_a_parent_keeps_all_of_its_entries(qapp, figure):
    """Nothing but the menu owns the actions, so the menu has to own them.

    ``QMenu.addAction`` does not adopt an action constructed separately, so
    actions parented to a ``None`` caller are collected as soon as the builder
    returns and the menu loses everything except the two submenus.
    """
    menu = fs.build_figure_context_menu(None, figure)
    try:
        texts = [action.text() for action in menu.actions()]

        for expected in ("Legend", "Grid", "Save figure as…",
                         "Save figure with a preview…", "Figure settings…"):
            assert expected in texts, f"{expected!r} was collected away"
        appearance = next(action.menu() for action in menu.actions()
                          if action.menu() is not None
                          and action.menu().title() == "Appearance")
        assert [a.text() for a in appearance.actions()] == [
            "Line colour…", "Font colour…", "Follow the theme (colours)"]
    finally:
        menu.deleteLater()


def test_a_menu_for_a_figure_that_has_gone_still_says_so(qapp):
    """The disabled status entry is the only one, and it must survive too."""
    menu = fs.build_figure_context_menu(None, None)
    try:
        assert [a.text() for a in menu.actions()] == [
            "This figure can no longer be restyled"]
        assert not menu.actions()[0].isEnabled()
    finally:
        menu.deleteLater()


def test_the_settings_entry_opens_the_settings_it_was_given(qapp, figure):
    """``Figure settings…`` is wired only when a caller supplies an opener."""
    opened = []
    menu = fs.build_figure_context_menu(
        None, figure, open_settings=lambda: opened.append(True))
    try:
        _action_named(menu, "Figure settings…").trigger()
        assert opened == [True]
    finally:
        menu.deleteLater()


def test_the_appearance_ink_falls_back_when_the_axis_will_not_be_read(
        qapp, figure, monkeypatch):
    """The menu's colour entries open at black rather than failing to open."""
    axis = figure.axes[0]
    monkeypatch.setattr(axis.xaxis.label, "get_color",
                        lambda *_a: (_ for _ in ()).throw(
                            RuntimeError("unreadable colour spec")))
    offered = []

    def remember(_parent, current, *_rest):
        offered.append(current)
        return QColor("#ff0000")

    monkeypatch.setattr(fs, "pick_colour", remember)
    menu = fs.build_figure_context_menu(None, figure)
    try:
        _action_named(menu, "Line colour…").trigger()

        assert offered == ["#000000"]
        assert all(_hex(spine.get_edgecolor()) == "#ff0000"
                   for spine in axis.spines.values())
    finally:
        menu.deleteLater()


def test_the_ink_entries_work_on_a_figure_with_no_axes(qapp, monkeypatch):
    """With no axes there is no colour to read, and the pick still applies."""
    figure = plt.figure()
    figure.suptitle("a run")
    monkeypatch.setattr(fs, "pick_colour",
                        lambda *a, **k: QColor("#00ff00"))
    menu = fs.build_figure_context_menu(None, figure)
    try:
        _action_named(menu, "Font colour…").trigger()

        assert _hex(figure._suptitle.get_color()) == "#00ff00"
    finally:
        menu.deleteLater()
        plt.close(figure)


def test_cancelling_the_ink_picker_leaves_the_figure_as_it_was(qapp, figure,
                                                               monkeypatch):
    """A dismissed picker applies nothing and notifies nobody."""
    axis = figure.axes[0]
    axis.set_xlabel("x")
    axis.xaxis.label.set_color("#112233")
    notified = []
    monkeypatch.setattr(fs, "pick_colour", lambda *a, **k: QColor())
    menu = fs.build_figure_context_menu(
        None, figure, on_change=lambda preview=True: notified.append(preview))
    try:
        _action_named(menu, "Font colour…").trigger()

        assert notified == []
        assert _hex(axis.xaxis.label.get_color()) == "#112233"
    finally:
        menu.deleteLater()


# ---------------------------------------------------------------------------
# the styled save dialog
# ---------------------------------------------------------------------------

def test_the_styled_save_dialog_is_held_by_the_widget_that_opened_it(qapp,
                                                                    figure):
    """Nothing else refers to it, so the parent has to keep it alive."""
    parent = QWidget()
    try:
        first = fs._open_styled_save(parent, figure)

        assert first is not None
        assert parent._spacr_save_dialogs == [first]
        assert first.isVisible() or first.isVisibleTo(parent)

        second = fs._open_styled_save(parent, figure)
        assert parent._spacr_save_dialogs == [first, second]
    finally:
        for dialog in getattr(parent, "_spacr_save_dialogs", []):
            dialog.close()
        parent.deleteLater()


def test_the_styled_save_opens_without_a_parent_to_be_held_by(qapp, figure):
    """There is nowhere to keep it, so it is handed straight back instead."""
    dialog = fs._open_styled_save(None, figure)
    try:
        assert dialog is not None
        assert dialog.parent() is None
    finally:
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()


def test_the_styled_save_is_declined_when_there_is_no_figure(qapp):
    assert fs._open_styled_save(QWidget(), None) is None


def test_the_styled_save_is_declined_when_the_dialog_cannot_be_built(
        qapp, figure, monkeypatch):
    """A build that raises returns nothing rather than a half-open window."""
    monkeypatch.setitem(sys.modules,
                        "spacr.qt.widgets.save_figure_dialog", None)

    assert fs._open_styled_save(QWidget(), figure) is None


# ---------------------------------------------------------------------------
# save_figure_as
# ---------------------------------------------------------------------------

def test_the_file_chooser_supplies_the_path_when_none_is_given(qapp, figure,
                                                               tmp_path,
                                                               monkeypatch):
    """With no path the dialog is asked, and what it returns is written."""
    target = tmp_path / "chosen.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))

    written = fs.save_figure_as(None, figure)

    assert written == str(target)
    assert target.exists() and target.stat().st_size > 0


def test_there_is_nothing_to_save_without_a_figure(qapp):
    assert fs.save_figure_as(None, None) == ""


def test_cancelling_the_figure_chooser_writes_nothing(qapp, figure, tmp_path,
                                                      monkeypatch):
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    assert fs.save_figure_as(None, figure) == ""
    assert list(tmp_path.glob("*.png")) == []


def test_a_raster_write_that_fails_reports_no_path(qapp, figure, tmp_path):
    """The PNG path goes through spacr.plot.save_figure, and it can fail too."""
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory would have to be")
    target = blocker / "figure.png"

    assert fs.save_figure_as(None, figure, str(target)) == ""
    assert blocker.is_file(), "and the thing in the way is left alone"


def test_a_build_without_the_plot_module_still_writes_the_file(
        qapp, figure, tmp_path, monkeypatch):
    """No spacr.plot means no print rule, but the file is still produced."""
    monkeypatch.setitem(sys.modules, "spacr.plot", None)
    target = tmp_path / "qt_only.png"

    written = fs.save_figure_as(None, figure, str(target))

    assert written == str(target)
    assert target.exists() and target.stat().st_size > 0


def test_a_vector_save_without_the_preference_store_uses_a_clear_ground(
        qapp, figure, tmp_path, monkeypatch):
    """SVG bypasses save_figure, and with no store the ground is transparent."""
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", None)
    target = tmp_path / "vector.svg"

    written = fs.save_figure_as(None, figure, str(target))

    assert written == str(target)
    body = target.read_text()
    assert body.startswith("<?xml"), "an SVG, not a raster renamed"
    # The ground is what the fallback chooses, so assert it rather than the
    # file type: a "white" ground writes an opaque canvas rectangle, and a
    # transparent one writes no white fill at all.
    assert "#ffffff" not in body.lower()


def test_a_write_that_fails_reports_no_path(qapp, figure, tmp_path):
    """A save that cannot happen returns an empty string rather than raising."""
    target = tmp_path / "no-such-directory" / "figure.svg"

    assert fs.save_figure_as(None, figure, str(target)) == ""
    assert not target.exists()


# ---------------------------------------------------------------------------
# export_sidecars
# ---------------------------------------------------------------------------

def _figure_with(**attributes):
    figure = plt.figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])
    for name, value in attributes.items():
        setattr(figure, name, value)
    return figure


def test_one_testable_group_writes_no_statistics_sidecar(tmp_path):
    """A comparison needs two groups; one usable group produces no file."""
    figure = _figure_with(_spacr_groups={"a": [1.0, 2.0, 3.0], "b": [1.0]})
    try:
        written = fs.export_sidecars(figure, str(tmp_path / "figure.png"))

        assert written == []
        assert not (tmp_path / "figure_stats.csv").exists()
    finally:
        plt.close(figure)


def test_a_pair_that_cannot_be_tested_is_dropped_and_the_rest_are_written(
        tmp_path):
    """One refused pair does not cost the pairs that could be compared."""
    figure = _figure_with(_spacr_groups={
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 3.0, 4.0, 5.0],
        # Two entries, so it survives the size filter, but only one of them
        # is a usable number -- every pair it is in is refused.
        "c": [7.0, math.nan],
    })
    try:
        written = fs.export_sidecars(figure, str(tmp_path / "figure.png"))

        stats = tmp_path / "figure_stats.csv"
        assert written == [str(stats)]
        body = stats.read_text()
        assert "a" in body and "b" in body
        assert body.count("\n") == 2, "one comparison and one header row"
    finally:
        plt.close(figure)


def test_no_testable_pair_at_all_writes_no_statistics_sidecar(tmp_path):
    figure = _figure_with(_spacr_groups={"a": [1.0, math.nan],
                                         "b": [2.0, math.nan]})
    try:
        assert fs.export_sidecars(figure, str(tmp_path / "figure.png")) == []
        assert not (tmp_path / "figure_stats.csv").exists()
    finally:
        plt.close(figure)


def test_a_statistics_failure_does_not_cost_the_other_sidecars(tmp_path,
                                                               monkeypatch):
    """The three sidecars are independent; one failing leaves two written."""
    import pandas

    figure = _figure_with(
        _spacr_data=pandas.DataFrame({"x": [1, 2], "y": [3, 4]}),
        _spacr_groups={"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]},
        _spacr_caption="What the figure shows.")
    monkeypatch.setitem(sys.modules, "spacr.figures.stats", None)
    try:
        written = fs.export_sidecars(figure, str(tmp_path / "figure.png"))

        assert written == [str(tmp_path / "figure.csv"),
                           str(tmp_path / "figure_legend.txt")]
        assert not (tmp_path / "figure_stats.csv").exists()
        assert (tmp_path / "figure_legend.txt").read_text() == \
            "What the figure shows.\n"
    finally:
        plt.close(figure)


def test_a_legend_that_cannot_be_written_is_reported_as_not_written(tmp_path):
    """An unwritable destination leaves the caption out of the returned list."""
    figure = _figure_with(_spacr_caption="a caption")
    try:
        target = tmp_path / "missing" / "figure.png"

        assert fs.export_sidecars(figure, str(target)) == []
        assert not (tmp_path / "missing").exists()
    finally:
        plt.close(figure)


# ---------------------------------------------------------------------------
# style choices and the preferences panel
# ---------------------------------------------------------------------------

def test_the_style_choices_fall_back_when_the_style_module_is_absent(
        monkeypatch):
    """The panel stays usable on the local table when the canonical one is gone."""
    monkeypatch.setitem(sys.modules, "spacr.figure_style", None)

    assert fs.style_choices_for("palette") == \
        fs._FALLBACK_CHOICES["palette"]
    assert fs.style_choices_for("not a setting") == ()


@pytest.fixture()
def panel(qapp):
    widget = fs.FigureStylePreferences()
    yield widget
    widget.deleteLater()


def test_the_panel_saves_what_is_on_screen_not_what_is_stored(panel, tmp_path,
                                                              monkeypatch):
    """A panel with unsaved edits writes the controls, not the preference."""
    target = tmp_path / "house.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    panel.apply_values({"palette": "muted"}, {})

    _button_named(panel, "Save style…").click()

    data = json.loads(target.read_text())
    assert data["spacr_style_kind"] == fs.GRAPH_STYLE_FILE_KIND
    assert data["general"]["palette"] == "muted"


def test_cancelling_the_panel_save_writes_no_file(panel, tmp_path, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    _button_named(panel, "Save style…").click()

    assert list(tmp_path.iterdir()) == [tmp_path / "config"]


def test_loading_a_style_fills_the_controls_and_not_the_store(panel, tmp_path,
                                                              monkeypatch):
    """The load is visible and still cancellable; it does not write through."""
    path = tmp_path / "house.json"
    path.write_text(json.dumps({
        "spacr_style_kind": fs.GRAPH_STYLE_FILE_KIND,
        "general": {"palette": "pastel", "grid_width": 2.5},
        "per_graph": {},
    }))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))

    _button_named(panel, "Load style…").click()

    general, _per_graph = panel.values()
    assert general["palette"] == "pastel"
    assert general["grid_width"] == pytest.approx(2.5)


def test_a_file_that_is_not_a_house_style_warns_and_changes_nothing(
        panel, tmp_path, monkeypatch):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"something": "else"}))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *args, **kw: warnings.append(args)))
    panel.apply_values({"palette": "bright"}, {})

    _button_named(panel, "Load style…").click()

    assert warnings, "the user is told why nothing changed"
    assert "not a spaCR graph style" in str(warnings[0][-1])
    assert panel.values()[0]["palette"] == "bright"


def test_cancelling_the_panel_load_changes_nothing(panel, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    panel.apply_values({"palette": "bright"}, {})

    _button_named(panel, "Load style…").click()

    assert panel.values()[0]["palette"] == "bright"


def test_an_explicit_background_unticks_transparent_and_shows_the_colour(
        panel):
    """The ground control's swatch follows the value it is given."""
    row = _row(panel, fs.style_setting_label("background"))
    box = row.findChild(QCheckBox)
    button = row.findChild(QPushButton)
    assert box.isChecked(), "the package default ground is transparent"
    assert not button.isEnabled()

    panel.apply_values({"background": "#204060"}, {})

    assert not box.isChecked()
    assert button.isEnabled()
    assert button.text() == "#204060"
    assert "#204060" in button.styleSheet()
    assert panel.values()[0]["background"] == "#204060"

    panel.apply_values({"background": "wisteria"}, {})
    assert not box.isChecked()
    assert button.text() == "wisteria", (
        "a value Qt cannot read is shown rather than silently corrected")
    assert panel.values()[0]["background"] == "wisteria"

    panel.apply_values({}, {})
    assert box.isChecked(), "the default puts the transparent ground back"


def test_a_choice_the_package_does_not_offer_leaves_the_control_alone(panel):
    """An unknown stored value cannot be selected, and nothing is snapped."""
    panel.apply_values({"palette": "muted"}, {})

    panel.apply_values({"palette": "no such palette"}, {})

    combo = _row(panel, fs.style_setting_label("palette"))
    assert combo.currentData() == "muted"


def test_a_colour_qt_cannot_read_is_still_shown_on_the_button(panel):
    """The value is kept and displayed rather than silently corrected."""
    panel.apply_values({"grid_colour": "not a colour"}, {})

    button = _row(panel, fs.style_setting_label("grid_colour"))
    assert button.text() == "not a colour"
    assert panel.values()[0]["grid_colour"] == "not a colour"


def test_select_kind_moves_to_that_page_and_an_unknown_kind_does_not(panel):
    panel.select_kind("histogram")
    assert panel._kind_box.currentData() == "histogram"

    panel.select_kind("not a graph type")

    assert panel._kind_box.currentData() == "histogram"
