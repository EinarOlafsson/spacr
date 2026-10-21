"""The runtime panes keep every Actions and System button whole at 75 % and 100 %.

The macOS distribution smoke of nightly 69c954384, on a 1440 x 870 window,
failed its layout check: Run, Stop, the import/export buttons, Submit remote,
Clear and Copy console and the gear were 24-26 px tall with about 20 px of
each visible, and the System card's button lost the same share. The Mac was
drawing spaCR at 75 % GUI scale, which d034b0b873 now makes the default on a
screen under 1600 px wide.

The cause was ``_RuntimeViewport._sync_floor``: it measures each pane's
``minimumSizeHint`` and ``heightForWidth`` from widgets already drawn at the
GUI scale, then handed them to ``setMinimumHeight``, which the scaling layer
scales again. At 75 % the Actions pane needed 196 px and got a floor of 147,
so the splitter squeezed it and its button row was cut. The floors are now
set in the pixels they were measured in.

Content a scroll area has scrolled away (the console's topic row after a
long run) is not a layout fault: the smoke records it in its raw viewport
pass and then scrolls to it, and only an unreachable control fails. The
full controller run below holds that at 75 %.
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QRect  # noqa: E402
from PySide6.QtWidgets import QAbstractButton, QWidget  # noqa: E402

from spacr.qt import gui_scale  # noqa: E402
from spacr.qt import preferences as prefs  # noqa: E402
from spacr.qt.startup_benchmark import \
    _DistributionSmokeController as Controller  # noqa: E402

pytestmark = pytest.mark.qt

WINDOW = (1440, 870)


@pytest.fixture
def scaled(qapp, request):
    """Start the GUI at ``request.param`` GUI and font scale, as a launch does.

    :returns: the scale in force.
    """
    scale = float(request.param)
    prefs.set_gui_scale(scale)
    prefs.set_font_scale(scale)
    gui_scale.apply_saved_gui_scale()
    prefs.apply_preferences_to_app(qapp)
    try:
        yield gui_scale.current_scale()
    finally:
        prefs.set_gui_scale(1.0)
        prefs.set_font_scale(1.0)
        gui_scale.set_gui_scale_live(1.0)
        prefs.apply_preferences_to_app(qapp)


def _window_rect(widget, window):
    """``widget``'s rectangle in ``window`` coordinates."""
    return QRect(widget.mapTo(window, QPoint()), widget.size())


def _visible_rect(widget, window):
    """The part of ``widget`` every ancestor up to ``window`` leaves visible."""
    clip = _window_rect(widget, window)
    ancestor = widget.parentWidget()
    while ancestor is not None and ancestor is not window:
        clip = clip.intersected(_window_rect(ancestor, window))
        ancestor = ancestor.parentWidget()
    return clip.intersected(window.rect())


def _main_window(qtbot):
    """The real main window at the Mac's size."""
    import spacr.qt
    from spacr.qt.app import MainWindow
    from spacr.qt.preferences import set_preload_policy

    spacr.qt.register_self_registering_modules()
    set_preload_policy("on_demand")
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(*WINDOW)
    window.show()
    return window


@pytest.mark.parametrize("scaled", [1.0, 0.9, 0.75], indirect=True)
def test_every_actions_and_system_button_is_whole_on_measure(qtbot, scaled):
    window = _main_window(qtbot)
    try:
        window.open_module("measure")
        qtbot.waitUntil(lambda: window._screens.get("measure") is not None
                        and window._screens["measure"]._btn_run.isEnabled(),
                        timeout=60000)
        qtbot.wait(1500)
        screen = window._screens["measure"]
        assert gui_scale.current_scale() == scaled
        screen._console.append_stdout("".join(
            f"field {line}: segmented, measured and written\n"
            for line in range(1500)))
        qtbot.wait(1500)
        enforced = gui_scale._ORIGINAL.get((QWidget, "minimumHeight"),
                                           QWidget.minimumHeight)
        split = screen._runtime_splitter
        viewport = screen._runtime_viewport
        checked = []
        for index in range(split.count()):
            pane = split.widget(index)
            registered = split._pane_of(pane)
            name = registered.name if registered is not None else pane.objectName()
            if name not in ("Actions", "System") or not pane.isVisible():
                continue
            need = max(pane.minimumSizeHint().height(),
                       split._height_for_width(pane))
            assert enforced(pane) >= need, (name, scaled, enforced(pane), need)
            assert pane.height() >= need, (name, scaled, pane.height(), need)
            for button in pane.findChildren(QAbstractButton):
                if not button.isVisibleTo(pane) or button.height() <= 0:
                    continue
                viewport.ensureWidgetVisible(button, 0, 0)
                qtbot.wait(50)
                visible = _visible_rect(button, window)
                checked.append((name, button.objectName(), button.text()))
                assert visible.height() == button.height(), (
                    name, button.objectName(), button.text(), scaled,
                    visible.height(), button.height())
        assert any(name == "Actions" for name, _o, _t in checked)
        assert any(name == "System" for name, _o, _t in checked)
        assert screen._btn_run.height() > 0
    finally:
        window.close()


@pytest.mark.parametrize("scaled", [0.75], indirect=True)
def test_the_smoke_layout_check_passes_after_a_run_at_three_quarter_scale(
        qapp, qtbot, monkeypatch, tmp_path, scaled):
    """The real controller, Run and layout-reachability pass at 1440 x 870."""
    completed = []

    def source_scope(self):
        """Permit the test interpreter only in this source-level test."""
        self.record["test_scope"] = "source UI integration, not native artifact acceptance"

    def finish(self, code):
        """Retain the test QApplication after the pipeline has finished."""
        self.phase = "finished"
        self.timer.stop()
        self._write()
        completed.append(code)

    monkeypatch.setattr(Controller, "_provenance", source_scope)
    monkeypatch.setattr(Controller, "_finish", finish)
    window = _main_window(qtbot)
    receipt = tmp_path / "smoke.json"
    controller = Controller(qapp, window, str(receipt))
    try:
        qtbot.waitUntil(lambda: bool(completed), timeout=300000)
        record = json.loads(receipt.read_text(encoding="utf-8"))
        layout = record.get("layout", {})
        unreachable = [
            {key: row.get("before", {}).get(key)
             for key in ("pane", "class", "object_name", "rect", "visible_clip")}
            for row in layout.get("reachability", []) if not row.get("reachable")]
        assert completed == [0], json.dumps(
            {"error": record.get("error"), "unreachable": unreachable}, sort_keys=True)
        assert (window.width(), window.height()) == WINDOW
        assert record["visual_layout_status"] == "passed"
        assert not any(control["undersized"] for control in layout["violations"])
        assert all(control["pane"] == "Console" or control["class"] == "QLabel"
                   for control in layout["violations"]), layout["violations"]
        assert layout["scroll_origin_restored"] is True
    finally:
        controller.timer.stop()
        window.close()


@pytest.mark.parametrize("scaled", [1.0, 0.75], indirect=True)
def test_the_floor_is_the_measured_height_at_any_gui_scale(qtbot, scaled):
    """A pane's floor is its measured need, not that need scaled again."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QPushButton, QVBoxLayout

    from spacr.qt.screens.app_screen import _RuntimeViewport
    from spacr.qt.widgets.collapsible_splitter import CollapsibleSplitter

    wrap = QWidget()
    qtbot.addWidget(wrap)
    wrap.resize(600, 160)
    content = QWidget()
    viewport = _RuntimeViewport(content, wrap)
    viewport.setGeometry(0, 0, 600, 160)
    column = QVBoxLayout(content)
    column.setContentsMargins(0, 0, 0, 0)
    split = CollapsibleSplitter(Qt.Vertical)
    column.addWidget(split)
    console = QWidget()
    split.add_pane(console, "Console", extent=300, minimum=0)
    actions = QWidget()
    rows = QVBoxLayout(actions)
    for caption in ("Run", "Stop", "Import settings", "Export settings"):
        rows.addWidget(QPushButton(caption))
    split.add_pane(actions, "Actions", stretch=0)
    viewport._watch_splitter(split)
    wrap.show()
    qtbot.wait(200)
    viewport._sync_floor()
    enforced = gui_scale._ORIGINAL.get((QWidget, "minimumHeight"),
                                       QWidget.minimumHeight)
    need = actions.minimumSizeHint().height()
    assert gui_scale.current_scale() == scaled
    assert enforced(actions) == need, (scaled, enforced(actions), need)
    assert enforced(split) >= need
    assert enforced(content) >= enforced(split)
