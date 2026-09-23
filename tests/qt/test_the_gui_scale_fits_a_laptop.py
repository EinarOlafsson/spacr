"""471 slice A -- a live whole-GUI scale, Keep or Revert, and preview sliders.

The request: "add scale GUI as a setting from 10% to 200% default 100%
... also for each live preview", then "please make this work with out
restarting. and if either font or GUI scale are modified the user should be
prompted with a Keep or Revert popup".

So the GUI scale is a scaling layer over Qt's own setters
(:mod:`spacr.qt.gui_scale`), and these tests hold it to the four things that
make it usable: a change reaches widgets already built AND widgets built
afterwards, going back to 100 % restores exactly what the code asked for,
the Keep question reverts by itself, and font scale composes with it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel,  # noqa: E402
                               QPushButton, QSlider, QToolButton,
                               QVBoxLayout, QWidget)

from spacr.qt import gui_scale  # noqa: E402
from spacr.qt import preferences as prefs  # noqa: E402
from spacr.qt.widgets import preview_scale as ps  # noqa: E402

pytestmark = pytest.mark.qt

TREE = str(Path(prefs.__file__).resolve().parents[2])


@pytest.fixture(autouse=True)
def _never_the_real_preferences():
    """Refuse to write a GUI scale into the store a person uses."""
    store = QSettings(prefs._ORG, prefs._APP).fileName()
    assert not store.startswith(str(Path.home() / ".config" / "spacr")), store
    yield
    prefs.set_gui_scale(1.0)
    prefs.set_font_scale(1.0)


@pytest.fixture(autouse=True)
def _the_layer_is_in_and_the_scale_goes_back(qapp):
    """Install the scaling layer once, and leave every test at 100 %."""
    gui_scale.install_scaling_layer()
    yield
    gui_scale.set_gui_scale_live(1.0)


def raw(widget, name: str):
    """Call a Qt method as it was before the layer replaced it.

    The layer answers ``minimumWidth`` in 100 % units on purpose, so a test
    that wants the pixels Qt is really using has to ask underneath it.

    :param widget: the widget or layout to ask.
    :param name: the method's name.
    """
    original = gui_scale._original_for(widget, name)
    if original is None:
        from PySide6.QtWidgets import QWidget as _W

        original = gui_scale._ORIGINAL[(_W, name)]
    return original(widget)


# ---------------------------------------------------------------------------
# The preference
# ---------------------------------------------------------------------------

def test_the_range_is_ten_to_two_hundred_and_the_default_is_one_hundred():
    assert (prefs.GUI_SCALE_MIN, prefs.GUI_SCALE_MAX) == (0.10, 2.00)
    assert prefs.DEFAULT_GUI_SCALE == 1.0
    prefs.set_gui_scale(0.01)
    assert prefs.get_gui_scale() == pytest.approx(0.10)
    prefs.set_gui_scale(9)
    assert prefs.get_gui_scale() == pytest.approx(2.0)
    prefs.set_gui_scale(1.0)


# ---------------------------------------------------------------------------
# The live change
# ---------------------------------------------------------------------------

def _panel_with_every_kind_of_size(qtbot):
    """A panel whose sizes come from each setter the layer replaces."""
    panel = QWidget()
    column = QVBoxLayout(panel)
    column.setContentsMargins(8, 8, 8, 8)
    column.setSpacing(6)
    column.addSpacing(20)
    button = QToolButton(panel)
    button.setFixedSize(100, 40)
    button.setIconSize(QSize(24, 24))
    label = QLabel("status", panel)
    label.setStyleSheet(
        "QLabel { font-size: 20px; padding: 4px; border: 1px solid red; }")
    view = QWidget(panel)
    view.setMinimumHeight(160)
    view.setMaximumWidth(400)
    column.addWidget(button)
    column.addWidget(label)
    column.addWidget(view)
    qtbot.addWidget(panel)
    panel.show()
    return panel, column, button, label, view


def test_a_live_change_scales_sizes_margins_icons_and_sheets(qtbot,
                                                            qt_theme_applied):
    """Half the scale: half the pixels, in widgets that already exist."""
    panel, column, button, label, view = _panel_with_every_kind_of_size(qtbot)
    gui_scale.set_gui_scale_live(0.5)
    assert raw(button, "minimumWidth") == 50
    assert raw(button, "minimumHeight") == 20
    assert raw(button, "iconSize") == QSize(12, 12)
    assert raw(view, "minimumHeight") == 80
    assert raw(view, "maximumWidth") == 200
    assert raw(column, "contentsMargins").left() == 4
    assert raw(column, "spacing") == 3
    sheet = raw(label, "styleSheet")
    assert "font-size: 10px" in sheet
    assert "padding: 2px" in sheet
    assert "1px solid red" in sheet, "a hairline must stay a hairline"


def test_a_widget_built_after_the_change_is_scaled_too(qtbot,
                                                       qt_theme_applied):
    gui_scale.set_gui_scale_live(0.5)
    later = QWidget()
    qtbot.addWidget(later)
    later.setFixedWidth(80)
    later.setStyleSheet("QWidget { font-size: 12px; }")
    assert raw(later, "minimumWidth") == 40
    assert "font-size: 6px" in raw(later, "styleSheet")
    assert later.minimumWidth() == 80, "it reads back in 100 % units"


def test_a_round_trip_has_zero_drift_on_every_recorded_size(qtbot,
                                                            qt_theme_applied):
    """100 % to 50 % and back is the same pixel in every size the code set."""
    panel, column, button, label, view = _panel_with_every_kind_of_size(qtbot)
    before = {
        "button_w": raw(button, "minimumWidth"),
        "button_h": raw(button, "minimumHeight"),
        "icon": raw(button, "iconSize"),
        "view_min": raw(view, "minimumHeight"),
        "view_max": raw(view, "maximumWidth"),
        "margins": raw(column, "contentsMargins"),
        "spacing": raw(column, "spacing"),
        "sheet": raw(label, "styleSheet"),
    }
    for scale in (0.5, 1.7, 0.1, 1.0):
        gui_scale.set_gui_scale_live(scale)
    after = {
        "button_w": raw(button, "minimumWidth"),
        "button_h": raw(button, "minimumHeight"),
        "icon": raw(button, "iconSize"),
        "view_min": raw(view, "minimumHeight"),
        "view_max": raw(view, "maximumWidth"),
        "margins": raw(column, "contentsMargins"),
        "spacing": raw(column, "spacing"),
        "sheet": raw(label, "styleSheet"),
    }
    assert after == before


def test_a_widget_that_re_measures_itself_keeps_its_hundred_percent_size(
        qtbot, qt_theme_applied):
    """The close-mark case: a size re-derived from the scaled font.

    Setting a size that is what the layer itself applied must not become a
    new 100 % size, or the round trip leaves the widget shrunken.
    """
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setFixedWidth(40)
    gui_scale.set_gui_scale_live(0.5)
    widget.setFixedWidth(raw(widget, "minimumWidth"))
    gui_scale.set_gui_scale_live(1.0)
    assert raw(widget, "minimumWidth") == 40


def test_the_scale_does_not_touch_the_exempt_window(qtbot, qt_theme_applied):
    """A window marked exempt -- the Keep question -- is drawn at 100 %."""
    dialog = QDialog()
    dialog.setProperty(gui_scale.EXEMPT, True)
    inner = QLabel("x", dialog)
    inner.setFixedWidth(120)
    qtbot.addWidget(dialog)
    gui_scale.set_gui_scale_live(0.25)
    assert raw(inner, "minimumWidth") == 120


def test_font_scale_and_gui_scale_compose(qtbot, qt_theme_applied):
    """50 % GUI at 200 % font: half-size widgets, text its usual size."""
    from spacr.qt.theme import FONT_SIZE, stylesheet

    body = FONT_SIZE["body"]
    sheet = stylesheet(font_scale=2.0, load_widget_registrars=False)
    assert f"font-size: {body * 2}px" in sheet
    gui_scale.set_gui_scale_live(0.5)
    scaled = gui_scale.scale_qss_text(sheet)
    assert f"font-size: {body}px" in scaled, (
        "base x 2 x 0.5 is the size it was at 100 %")


def test_the_style_sheet_rewrite_leaves_everything_but_sizes_alone():
    text = ("QLabel#A { color: red; font-size: 13px; border: 2px solid red;"
            " padding: 0px 10px; background: url(data:image/png;base64,AA); }")
    assert gui_scale.scale_qss_text(text, 1.0) == text
    half = gui_scale.scale_qss_text(text, 0.5)
    assert "font-size: 7px" in half or "font-size: 6px" in half
    assert "border: 2px solid red" in half
    assert "padding: 0px 5px" in half
    assert "color: red" in half
    assert "url(data:image/png;base64,AA)" in half


# ---------------------------------------------------------------------------
# Keep or Revert
# ---------------------------------------------------------------------------

def test_the_countdown_reverts_by_itself(qtbot, qt_theme_applied):
    prefs.set_gui_scale(1.0)
    prefs.set_font_scale(1.0)
    answered = []
    gui_scale.change_scales(None, gui=0.5, font=1.5, seconds=1,
                            require_parent=False,
                            on_done=answered.append)
    assert prefs.get_gui_scale() == pytest.approx(0.5)
    assert gui_scale.current_scale() == pytest.approx(0.5)
    qtbot.waitUntil(lambda: bool(answered), timeout=8000)
    assert answered == [False]
    assert prefs.get_gui_scale() == pytest.approx(1.0)
    assert prefs.get_font_scale() == pytest.approx(1.0)
    assert gui_scale.current_scale() == pytest.approx(1.0)


def test_keep_keeps_and_revert_puts_the_old_values_back(qtbot,
                                                        qt_theme_applied):
    prefs.set_gui_scale(1.0)
    answered = []
    dialog = gui_scale.change_scales(None, gui=0.75, seconds=60,
                                     require_parent=False,
                                     on_done=answered.append)
    assert dialog is not None
    dialog.keep_button.click()
    qtbot.waitUntil(lambda: bool(answered), timeout=3000)
    assert answered == [True]
    assert prefs.get_gui_scale() == pytest.approx(0.75)

    answered.clear()
    dialog = gui_scale.change_scales(None, gui=0.2, seconds=60,
                                     require_parent=False,
                                     on_done=answered.append)
    dialog.revert_button.click()
    qtbot.waitUntil(lambda: bool(answered), timeout=3000)
    assert answered == [False]
    assert prefs.get_gui_scale() == pytest.approx(0.75)
    prefs.set_gui_scale(1.0)
    gui_scale.set_gui_scale_live(1.0)


def test_escape_reverts_as_well(qtbot, qt_theme_applied):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    prefs.set_gui_scale(1.0)
    answered = []
    dialog = gui_scale.change_scales(None, gui=0.5, seconds=60,
                                     require_parent=False,
                                     on_done=answered.append)
    dialog.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape,
                                   Qt.NoModifier))
    qtbot.waitUntil(lambda: bool(answered), timeout=3000)
    assert answered == [False]
    assert prefs.get_gui_scale() == pytest.approx(1.0)


def test_the_question_is_readable_at_ten_percent(qtbot, qt_theme_applied):
    """It is exempt from the scale and states its own text sizes."""
    gui_scale.set_gui_scale_live(0.1)
    dialog = gui_scale.keep_or_revert_dialog(None, seconds=60, what="x")
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.property(gui_scale.EXEMPT)
    assert raw(dialog, "minimumWidth") == 360
    assert "font-size: 17px" in dialog._own_rule
    assert dialog.keep_button.isDefault(), "Enter keeps"
    dialog.reject()


# ---------------------------------------------------------------------------
# Preferences, the reset key and the module's own screens
# ---------------------------------------------------------------------------

def _dialog(qtbot):
    dialog = prefs.PreferencesDialog(None)
    qtbot.addWidget(dialog)
    slider = dialog.findChild(QSlider, "GuiScale")
    font_slider = dialog.findChild(QSlider, "FontScale")
    assert slider is not None and font_slider is not None
    return dialog, slider, font_slider


def test_preferences_offers_it_beside_font_scale(qtbot, qt_theme_applied):
    dialog, slider, font_slider = _dialog(qtbot)
    assert (slider.minimum(), slider.maximum()) == (10, 200)
    assert slider.value() == 100
    assert dialog.findChild(QPushButton, "GuiScaleRestart") is None, (
        "nothing restarts any more")
    tip = prefs.PREFERENCE_TIPS["GUI scale"].lower()
    assert "font scale" in tip and "keep" in tip


def test_moving_the_slider_applies_it_and_asks(qtbot, qt_theme_applied):
    from PySide6.QtCore import QTimer

    dialog, slider, font_slider = _dialog(qtbot)
    dialog.show()
    settle = dialog.findChild(QTimer, "ScaleSettle")
    assert settle is not None
    slider.setValue(50)
    qtbot.waitUntil(lambda: gui_scale.current_scale() == pytest.approx(0.5),
                    timeout=5000)
    question = [w for w in dialog.findChildren(QDialog)
                if w.objectName() == "SpacrKeepOrRevert"]
    if not question:
        from PySide6.QtWidgets import QApplication

        question = [w for w in QApplication.topLevelWidgets()
                    if w.objectName() == "SpacrKeepOrRevert"]
    assert question, "a change of scale asks whether to keep it"
    question[0].revert_button.click()
    qtbot.waitUntil(
        lambda: gui_scale.current_scale() == pytest.approx(1.0), timeout=5000)
    assert slider.value() == 100, "Revert puts the slider back too"


def test_the_reset_key_is_declared_and_clashes_with_nothing():
    from spacr.qt import shortcuts

    declared = [s.keys for s in shortcuts.SHORTCUTS
                + shortcuts.SCREEN_SHORTCUTS]
    assert declared.count("Ctrl+Alt+0") == 1


def test_the_reset_key_puts_every_scale_back_without_a_restart(
        qtbot, qt_theme_applied):
    prefs.set_gui_scale(0.3)
    prefs.set_font_scale(0.4)
    gui_scale.set_gui_scale_live(0.3)
    assert gui_scale.reset_every_scale(None)
    assert prefs.get_gui_scale() == pytest.approx(1.0)
    assert prefs.get_font_scale() == pytest.approx(1.0)
    assert gui_scale.current_scale() == pytest.approx(1.0)


def test_the_figure_dpi_follows_both_scales(qtbot, qt_theme_applied):
    from spacr.qt.widgets.umap_explorer import ImageUmapExplorer

    explorer = ImageUmapExplorer()
    qtbot.addWidget(explorer)
    figure = explorer._canvas.figure
    base = figure.dpi
    gui_scale.set_gui_scale_live(0.5)
    assert figure.dpi == pytest.approx(base * 0.5)
    explorer._scale_control.scaler.set_scale(0.5)
    assert figure.dpi == pytest.approx(base * 0.25), "the two multiply"
    explorer._scale_control.scaler.set_scale(1.0)
    gui_scale.set_gui_scale_live(1.0)
    assert figure.dpi == pytest.approx(base)


# ---------------------------------------------------------------------------
# The per-preview scale
# ---------------------------------------------------------------------------

def _panel(qtbot):
    panel = QWidget()
    root = QVBoxLayout(panel)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(6)
    row = QHBoxLayout()
    label = QLabel("status", panel)
    label.setStyleSheet("color: red; font-size: 12px; border: 1px solid red;")
    thumb = QLabel(panel)
    thumb.setFixedSize(132, 132)
    view = QWidget(panel)
    view.setMinimumHeight(160)
    row.addWidget(label)
    root.addLayout(row)
    root.addWidget(thumb)
    root.addWidget(view)
    control = ps.install_preview_scale(panel, "unit_test", row)
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitUntil(lambda: control.isVisible(), timeout=2000)
    return panel, control, label, thumb, view, root


def test_the_preview_scale_scales_sizes_margins_and_its_own_sheets(qtbot):
    panel, control, label, thumb, view, root = _panel(qtbot)
    control.set_percent(50)
    assert ps.get_preview_scale("unit_test") == pytest.approx(0.5)
    assert (thumb.minimumWidth(), thumb.maximumWidth()) == (66, 66)
    assert view.minimumHeight() == 80
    assert root.contentsMargins().left() == 4 and root.spacing() == 3
    sheet = label.styleSheet()
    assert "font-size: 6px" in sheet
    assert "1px solid" in sheet, "a hairline must stay a hairline"
    assert "red" in sheet


def test_one_hundred_percent_puts_back_exactly_what_the_code_set(qtbot):
    panel, control, label, thumb, view, root = _panel(qtbot)
    original = label.styleSheet()
    control.set_percent(30)
    control.set_percent(100)
    assert label.styleSheet() == original
    assert (thumb.minimumWidth(), thumb.maximumWidth()) == (132, 132)
    assert view.minimumHeight() == 160
    assert root.contentsMargins().left() == 8 and root.spacing() == 6
    assert panel.styleSheet() == ""
    for widget in (panel, label, thumb, view):
        for name in widget.dynamicPropertyNames():
            assert not bytes(name).startswith(b"spacrPs"), bytes(name)


def test_scaling_twice_does_not_compound_and_a_new_size_becomes_the_base(
        qtbot):
    panel, control, label, thumb, view, root = _panel(qtbot)
    control.set_percent(50)
    control.set_percent(50)
    assert view.minimumHeight() == 80
    view.setMinimumHeight(300)
    control.set_percent(200)
    assert view.minimumHeight() == 600


def test_the_slider_itself_is_never_scaled_and_double_click_resets(qtbot):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    panel, control, *_ = _panel(qtbot)
    width = control.slider.width()
    control.set_percent(10)
    assert control.slider.width() == width
    event = QMouseEvent(QEvent.MouseButtonDblClick, QPointF(2, 2),
                        QPointF(2, 2), Qt.LeftButton, Qt.LeftButton,
                        Qt.NoModifier)
    control.eventFilter(control.value, event)
    assert control.scaler.scale() == pytest.approx(1.0)


def test_reset_all_puts_every_preview_back(qtbot):
    panel, control, *_ = _panel(qtbot)
    control.set_percent(40)
    assert ps.reset_all_preview_scales() >= 1
    assert control.scaler.scale() == pytest.approx(1.0)
    assert control.slider.value() == 100
    assert ps.get_preview_scale("unit_test") == pytest.approx(1.0)


def test_a_saved_scale_comes_back_with_the_next_panel(qtbot):
    ps.set_preview_scale("unit_test", 0.5)
    panel, control, label, thumb, view, root = _panel(qtbot)
    qtbot.waitUntil(lambda: view.minimumHeight() == 80, timeout=2000)
    assert control.slider.value() == 50
    ps.set_preview_scale("unit_test", 1.0)


def test_scale_qss_leaves_borders_and_zero_and_can_keep_only_sizes():
    sheet = ("QLabel#A { color: red; font-size: 13px; border: 2px solid red;"
             " padding: 0px 10px; }")
    assert ps.scale_qss(sheet, 1.0) == sheet
    half = ps.scale_qss(sheet, 0.5)
    assert "font-size: 6px" in half or "font-size: 7px" in half
    assert "border: 2px solid red" in half
    assert "padding: 0px 5px" in half
    sizes = ps.scale_qss(sheet, 0.5, sizes_only=True)
    assert "color" not in sizes and "border" not in sizes
    assert ps.scale_qss("QLabel { color: red }", 0.5, sizes_only=True) == ""


def test_every_named_preview_has_its_own_slider(qtbot, qt_theme_applied):
    """Mask, Measure and image UMAP -- the three the request names."""
    from spacr.qt.widgets.live_preview import LivePreviewPanel
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
    from spacr.qt.widgets.umap_explorer import ImageUmapExplorer

    names = {}
    for cls in (LivePreviewPanel, MeasurePreviewPanel, ImageUmapExplorer):
        widget = cls()
        qtbot.addWidget(widget)
        control = widget.findChild(ps.PreviewScaleControl)
        assert control is not None, cls.__name__
        names[cls.__name__] = control.scaler.name
    assert len(set(names.values())) == 3, names


def test_measure_thumbnails_follow_the_preview_scale(qtbot, qt_theme_applied):
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel

    panel = MeasurePreviewPanel()
    qtbot.addWidget(panel)
    panel._scale_control.scaler.set_scale(0.5)
    assert panel._thumb_px == 66
    panel._scale_control.scaler.set_scale(1.0)
    assert panel._thumb_px == 132


def test_the_umap_figure_scales_its_dots_and_labels(qtbot, qt_theme_applied):
    from spacr.qt.widgets.umap_explorer import ImageUmapExplorer

    explorer = ImageUmapExplorer()
    qtbot.addWidget(explorer)
    figure = explorer._canvas.figure
    before = figure.dpi
    explorer._scale_control.scaler.set_scale(0.5)
    assert figure.dpi == pytest.approx(before * 0.5)
    explorer._scale_control.scaler.set_scale(1.0)
    assert figure.dpi == pytest.approx(before)
