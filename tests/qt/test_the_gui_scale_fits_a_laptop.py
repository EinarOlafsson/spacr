"""471 slice A -- a whole-GUI scale, and a scale slider on every preview.

The request: "add scale GUI as a setting from 10% to 200% default 100%
... also for each live preview". The GUI scale is a startup-time Qt scale
factor (see :mod:`spacr.qt.gui_scale` for the audit that decided it), so
the tests that need a scaled application run it in a child process: a Qt
scale factor is fixed for the life of the process that reads it.

The preview scale is live, so its tests run here.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton,  # noqa: E402
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


def _child(code: str, tmp_path, **env) -> list:
    """Run ``code`` in a fresh Python with its own preference store."""
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True, exist_ok=True)
    environ = dict(os.environ, PYTHONPATH=TREE, QT_QPA_PLATFORM="offscreen",
                   HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"),
                   SPACR_NO_BACKDROP="1", SPACR_NO_GL="1", MPLBACKEND="Agg")
    for key in ("QT_SCALE_FACTOR", gui_scale.BASE_ENV,
                gui_scale.OVERRIDE_ENV):
        environ.pop(key, None)
    environ.update({k: str(v) for k, v in env.items()})
    done = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          env=environ, capture_output=True, text=True,
                          timeout=480)
    assert done.returncode == 0, done.stderr[-3000:]
    return [line for line in done.stdout.splitlines() if line.startswith("@")]


# ---------------------------------------------------------------------------
# The preference and the environment it writes
# ---------------------------------------------------------------------------

def test_the_range_is_ten_to_two_hundred_and_the_default_is_one_hundred():
    assert (prefs.GUI_SCALE_MIN, prefs.GUI_SCALE_MAX) == (0.10, 2.00)
    assert prefs.DEFAULT_GUI_SCALE == 1.0
    prefs.set_gui_scale(0.01)
    assert prefs.get_gui_scale() == pytest.approx(0.10)
    prefs.set_gui_scale(9)
    assert prefs.get_gui_scale() == pytest.approx(2.0)
    prefs.set_gui_scale(0.5)
    assert prefs.get_gui_scale() == pytest.approx(0.5)


def test_one_hundred_percent_writes_nothing_so_today_is_unchanged():
    prefs.set_gui_scale(1.0)
    environ = {}
    assert gui_scale.apply_gui_scale_to_environment(environ) == 1.0
    assert "QT_SCALE_FACTOR" not in environ


def test_fifty_percent_is_a_qt_scale_factor_of_one_half():
    prefs.set_gui_scale(0.5)
    environ = {}
    gui_scale.apply_gui_scale_to_environment(environ)
    assert float(environ["QT_SCALE_FACTOR"]) == pytest.approx(0.5)


def test_a_users_own_factor_is_kept_and_a_restart_does_not_compound_it():
    """QT_SCALE_FACTOR=2 set by the user, GUI scale 50 %: 1.0, every launch."""
    prefs.set_gui_scale(0.5)
    environ = {"QT_SCALE_FACTOR": "2"}
    gui_scale.apply_gui_scale_to_environment(environ)
    assert float(environ["QT_SCALE_FACTOR"]) == pytest.approx(1.0)
    restarted = dict(environ)
    gui_scale.apply_gui_scale_to_environment(restarted)
    assert float(restarted["QT_SCALE_FACTOR"]) == pytest.approx(1.0)
    prefs.set_gui_scale(1.0)
    gui_scale.apply_gui_scale_to_environment(restarted)
    assert float(restarted["QT_SCALE_FACTOR"]) == pytest.approx(2.0)


def test_the_environment_override_beats_the_stored_value():
    prefs.set_gui_scale(0.1)
    environ = {gui_scale.OVERRIDE_ENV: "1"}
    assert gui_scale.apply_gui_scale_to_environment(environ) == 1.0
    assert "QT_SCALE_FACTOR" not in environ


def test_the_window_keeps_its_size_on_the_screen():
    """At 50 % a 1200 px window asks for 2400 scaled px, within the screen."""
    from PySide6.QtCore import QRect

    class Screen:
        def __init__(self, w, h):
            self._rect = QRect(0, 0, w, h)

        def availableGeometry(self):
            return self._rect

    class Window:
        def __init__(self, screen):
            self.size = (1200, 800)
            self._screen = screen

        def width(self):
            return self.size[0]

        def height(self):
            return self.size[1]

        def screen(self):
            return self._screen

        def resize(self, w, h):
            self.size = (w, h)

    window = Window(Screen(2732, 1536))
    assert gui_scale.fit_window_to_gui_scale(window, 0.5)
    assert window.size == (2400, 1536), "clamped to the scaled screen"
    window = Window(Screen(2732, 1536))
    assert not gui_scale.fit_window_to_gui_scale(window, 1.0)
    assert window.size == (1200, 800)


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------

def _dialog(qtbot):
    from PySide6.QtWidgets import QSlider

    dialog = prefs.PreferencesDialog(None)
    qtbot.addWidget(dialog)
    slider = dialog.findChild(QSlider, "GuiScale")
    restart = dialog.findChild(QPushButton, "GuiScaleRestart")
    assert slider is not None and restart is not None
    return dialog, slider, restart


def test_preferences_offers_it_beside_font_scale(qtbot, qt_theme_applied):
    dialog, slider, restart = _dialog(qtbot)
    assert (slider.minimum(), slider.maximum()) == (10, 200)
    assert slider.value() == 100
    assert restart.isHidden(), "nothing to restart for at the running scale"
    tip = prefs.PREFERENCE_TIPS["GUI scale"].lower()
    assert "restart" in tip and "font scale" in tip, (
        "the row has to say it takes a restart and how it meets font scale")


def test_moving_it_offers_a_restart_and_save_stores_it(qtbot,
                                                      qt_theme_applied):
    from PySide6.QtWidgets import QDialogButtonBox

    dialog, slider, restart = _dialog(qtbot)
    slider.setValue(50)
    assert not restart.isHidden()
    value = dialog.findChild(QLabel, "GuiScaleValue")
    assert "50" in value.text() and "restart" in value.text().lower()
    buttons = dialog.findChild(QDialogButtonBox)
    buttons.button(QDialogButtonBox.Save).click()
    assert prefs.get_gui_scale() == pytest.approx(0.5)


def test_restart_now_saves_and_restarts_through_the_force_restart_record(
        qtbot, qt_theme_applied, monkeypatch):
    calls = []
    monkeypatch.setattr(gui_scale, "restart_to_apply",
                        lambda owner=None, **kw: calls.append(owner) or True)
    dialog, slider, restart = _dialog(qtbot)
    slider.setValue(60)
    restart.click()
    qtbot.waitUntil(lambda: bool(calls), timeout=3000)
    assert prefs.get_gui_scale() == pytest.approx(0.6)


def test_restart_to_apply_uses_the_screens_force_restart(monkeypatch):
    class Screen:
        def __init__(self):
            self.restarted = False

        def running_modules(self):
            return []

        def force_restart(self, *, launcher=None, exiter=None):
            self.restarted = True
            return True

    class Stack:
        def __init__(self, screen):
            self._screen = screen

        def currentWidget(self):
            return self._screen

    class Window:
        pass

    window = Window()
    window._stack = Stack(Screen())
    monkeypatch.setattr(gui_scale, "_main_window", lambda _w=None: window)
    assert gui_scale.restart_to_apply(None)
    assert window._stack.currentWidget().restarted


# ---------------------------------------------------------------------------
# Composition, and the 10 % floor -- in a child, because Qt reads the factor
# once per process
# ---------------------------------------------------------------------------

def test_fifty_percent_gui_at_two_hundred_percent_font_compose(tmp_path):
    """Half-size widgets with text the usual size on the screen."""
    lines = _child("""
        from spacr.qt import gui_scale, preferences as p
        p.set_gui_scale(0.5)
        p.set_font_scale(2.0)
        gui_scale.apply_gui_scale_to_environment()
        from PySide6.QtGui import QFontInfo
        from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
        app = QApplication([])
        p.apply_preferences_to_app(app)
        from spacr.qt.theme import FONT_SIZE
        page = QWidget(); lay = QVBoxLayout(page)
        body = QLabel("Plate 3"); lay.addWidget(body)
        box = QWidget(); box.setFixedSize(100, 40); lay.addWidget(box)
        page.show()
        for _ in range(30):
            app.processEvents()
        print("@ratio", page.devicePixelRatioF())
        print("@body_logical", QFontInfo(body.font()).pixelSize())
        print("@body_base", FONT_SIZE["body"])
        print("@box_device", box.grab().width())
        """, tmp_path)
    got = dict(line[1:].split(" ", 1) for line in lines)
    assert float(got["ratio"]) == pytest.approx(0.5)
    base = int(got["body_base"])
    logical = int(got["body_logical"])
    assert logical == pytest.approx(base * 2, abs=1), "font scale lost"
    assert logical * 0.5 == pytest.approx(base, abs=1), (
        "on screen the text should be its usual size: 2 x 0.5")
    assert int(got["box_device"]) == 50, "the GUI scale did not reach widgets"


def test_at_ten_percent_the_way_back_needs_no_reading(tmp_path):
    """spaCR opens at 10 %, and Ctrl+Alt+0 then Enter brings it back.

    The window is built for real, at the floor. The shortcut must be bound
    on it, and pressing it must put every scale back to 100 % and ask for a
    restart whose DEFAULT button is Restart now -- the user presses Enter
    without being able to read the dialog. Preferences also still opens,
    with the GUI scale row in it.
    """
    lines = _child("""
        from spacr.qt import gui_scale, preferences as p
        p.set_gui_scale(0.1)
        p.set_font_scale(0.1)
        gui_scale.apply_gui_scale_to_environment()
        from PySide6.QtGui import QKeySequence, QShortcut
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QMessageBox, QSlider
        app = QApplication([])
        p.apply_preferences_to_app(app)
        from spacr.qt.app import MainWindow
        win = MainWindow()
        win.resize(13660, 7680)
        win.show()
        for _ in range(60):
            app.processEvents()
        print("@ratio", win.devicePixelRatioF())
        keys = [s for s in win.findChildren(QShortcut)
                if s.key() == QKeySequence("Ctrl+Alt+0")]
        print("@bound", len(keys))
        dialog = p.PreferencesDialog(win)
        slider = dialog.findChild(QSlider, "GuiScale")
        print("@prefs_slider", slider.value() if slider else None)
        dialog.deleteLater()
        restarted = []
        gui_scale.restart_to_apply = lambda owner=None, **kw: restarted.append(1) or True
        def press_enter(box):
            button = box.defaultButton()
            print("@default", button.text() if button else None)
            button.click()
            return 0
        QMessageBox.exec = press_enter
        keys[0].activated.emit()
        for _ in range(10):
            app.processEvents()
        print("@gui", p.get_gui_scale())
        print("@font", p.get_font_scale())
        print("@restarted", len(restarted))
        """, tmp_path)
    got = dict(line[1:].split(" ", 1) for line in lines)
    assert float(got["ratio"]) == pytest.approx(0.1)
    assert got["bound"] == "1"
    assert got["prefs_slider"] == "10"
    assert got["default"] == "Restart now"
    assert float(got["gui"]) == pytest.approx(1.0)
    assert float(got["font"]) == pytest.approx(1.0)
    assert got["restarted"] == "1"


def test_the_reset_key_is_declared_and_clashes_with_nothing():
    from spacr.qt import shortcuts

    declared = [s.keys for s in shortcuts.SHORTCUTS + shortcuts.SCREEN_SHORTCUTS]
    assert declared.count("Ctrl+Alt+0") == 1
    assert "Meta+Alt+0" not in declared


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
