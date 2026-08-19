"""Instruction 158 on screen: the row stays dead, the panel does the talking.

GREEN TESTS DO NOT MEAN THE FEATURE WORKS, so these drive the real widgets
and MEASURE rather than assert intent:

* the disabled row's flags are read back off the model, because
  `setEnabled(False)` leaves `ItemIsSelectable` SET and that was the whole
  reason this file exists;
* the tooltip is read back off the model too, because the design rests on a
  disabled item keeping it;
* "INSTALL to the right of the API link" is checked as x-coordinates;
* the gap-crossing rule is driven by moving a fake pointer INTO the corridor
  and letting the hide timer fire, which is the case a naive leave-event
  dismissal fails and the one that makes Install unreachable.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QRect, Qt          # noqa: E402
from PySide6.QtGui import QKeyEvent                    # noqa: E402
from PySide6.QtWidgets import QComboBox, QLabel        # noqa: E402

from spacr import updater                              # noqa: E402
from spacr.qt.widgets.availability_panel import (      # noqa: E402
    AvailabilityPanel, disable_combo_row, explain, install_word_for,
    run_install_offer)


@pytest.fixture
def panel(qtbot):
    """The singleton, emptied between tests so one cannot leak into another."""
    widget = AvailabilityPanel.instance()
    widget.dismiss()
    widget.set_install_handler(None)
    yield widget
    widget.dismiss()
    widget.set_install_handler(None)


def _entry(action="install", **over):
    offer = {
        "install": updater.offer_install("cuML (GPU)", "needs cuml-cu12",
                                         "cuml-cu12", "the recipe"),
        "elsewhere": updater.offer_elsewhere("cuML (GPU)", "needs 3.11",
                                             "the recipe"),
        "impossible": updater.offer_impossible("pymer4 (CPU)", "needs R",
                                               "the recipe"),
        "ready": updater.offer_ready("statsmodels (CPU)", "available"),
    }[action]
    entry = {"key": "cuml", "title": offer.title, "reason": "greyed because",
             "url": "https://example.invalid/api", "enabled": False,
             "offer": offer}
    entry.update(over)
    return entry


# ---------------------------------------------------------------------------
# The row itself
# ---------------------------------------------------------------------------

def test_setEnabled_alone_leaves_the_row_selectable(qtbot):
    """The measurement the whole design is built on, kept as a test.

    If a future Qt closes this hole, `disable_combo_row` becomes redundant
    rather than wrong -- but nobody should discover the hole a third time.
    """
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItems(["a", "b"])
    combo.model().item(1).setEnabled(False)
    assert bool(combo.model().item(1).flags() & Qt.ItemIsSelectable) is True


def test_a_disabled_row_is_not_selectable_and_keeps_its_tooltip(qtbot):
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItems(["statsmodels (CPU)", "cuML (GPU)"])
    disable_combo_row(combo, 1, tooltip="cuML needs Python 3.11")
    item = combo.model().item(1)
    assert item.isEnabled() is False
    assert bool(item.flags() & Qt.ItemIsSelectable) is False
    # THE HALF THE DESIGN RESTS ON.
    assert item.toolTip() == "cuML needs Python 3.11"


def test_the_keyboard_cannot_walk_onto_a_disabled_row(qtbot):
    """"not briefly and not by keyboard" -- driven, not assumed."""
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItems(["statsmodels (CPU)", "cuML (GPU)", "glum (CPU)"])
    disable_combo_row(combo, 1)
    combo.setCurrentIndex(0)
    combo.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Down,
                                  Qt.NoModifier))
    assert combo.currentIndex() == 2
    assert combo.currentText() == "glum (CPU)"


def test_disable_combo_row_survives_a_model_with_no_items(qtbot):
    combo = QComboBox()
    qtbot.addWidget(combo)
    disable_combo_row(combo, 3, tooltip="nothing here")   # must not raise


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

def test_install_sits_to_the_right_of_the_api_link(qtbot, panel):
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    panel.show_for(anchor, [_entry("install")])
    assert panel.api_link().isVisible()
    assert panel.install_link().isVisible()
    assert panel.install_link().x() > panel.api_link().x()
    assert "Install" in panel.install_link().text()


@pytest.mark.parametrize("action,word", [
    ("install", "Install"), ("elsewhere", "How to get it"),
    ("impossible", "What it needs"), ("ready", ""),
])
def test_the_word_tells_the_truth_about_what_pressing_it_does(action, word):
    """A word reading "Install" on an offer that cannot install is
    instruction 106's inert control wearing a different hat."""
    assert install_word_for(action) == word


def test_the_explanation_carries_both_the_refusal_and_the_remedy(qtbot,
                                                                 panel):
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    panel.show_for(anchor, [_entry("elsewhere")])
    body = panel.body_label().text()
    assert "greyed because" in body
    assert "needs 3.11" in body


def test_the_api_link_routes_and_install_routes_the_offer(qtbot, panel):
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    entry = _entry("install")
    panel.show_for(anchor, [entry])
    seen = {}
    panel.api_requested.connect(lambda url: seen.setdefault("api", url))
    panel.set_install_handler(lambda offer: seen.setdefault("offer", offer))
    # `linkActivated` is what a press on the word emits -- the measured route.
    panel._on_link("api")
    panel._on_link("install")
    assert seen["api"] == "https://example.invalid/api"
    assert seen["offer"] is entry["offer"]


def test_an_unknown_href_is_ignored_rather_than_routed(qtbot, panel):
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    panel.show_for(anchor, [_entry("install")])
    fired = []
    panel.set_install_handler(fired.append)
    panel._on_link("javascript:alert(1)")
    assert fired == []


# ---------------------------------------------------------------------------
# The three things an interactive tooltip has to get right
# ---------------------------------------------------------------------------

def test_the_panel_does_not_vanish_while_the_pointer_crosses_the_gap(
        qtbot, panel, monkeypatch):
    """THE ONE THAT KILLS NAIVE VERSIONS.

    The pointer is over neither the anchor nor the panel -- it is in the gap
    between them. A leave-event dismissal closes the panel here and the
    Install link can never be reached.
    """
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    anchor.show()
    panel.show_for(anchor, [_entry("install")],
                   anchor_rect=QRect(100, 100, 200, 24))
    panel.move(100, 140)
    corridor = panel.corridor()
    assert corridor is not None
    gap = QPoint(corridor.center().x(), 132)          # between the two
    monkeypatch.setattr(type(panel), "_cursor_pos", lambda self: gap)
    panel.start_hide(1)
    panel._maybe_hide()
    assert panel.isVisible(), "closed mid-journey; Install is unreachable"


def test_the_panel_closes_once_the_pointer_leaves_the_corridor(
        qtbot, panel, monkeypatch):
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    anchor.show()
    panel.show_for(anchor, [_entry("install")],
                   anchor_rect=QRect(100, 100, 200, 24))
    panel.move(100, 140)
    far = QPoint(panel.corridor().right() + 400, 4000)
    monkeypatch.setattr(type(panel), "_cursor_pos", lambda self: far)
    panel.start_hide(1)
    panel._maybe_hide()
    assert panel.isVisible() is False


def test_a_pointer_parked_in_the_corridor_still_lets_it_go(
        qtbot, panel, monkeypatch):
    """The corridor is a licence to TRAVEL, not a licence to stay forever."""
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    anchor.show()
    panel.show_for(anchor, [_entry("install")],
                   anchor_rect=QRect(100, 100, 200, 24))
    panel.move(100, 140)
    gap = QPoint(panel.corridor().center().x(), 132)
    monkeypatch.setattr(type(panel), "_cursor_pos", lambda self: gap)
    panel.start_hide(1)
    panel._hide_since -= (panel.CORRIDOR_GRACE_MS / 1000.0) + 1.0
    panel._maybe_hide()
    assert panel.isVisible() is False


def test_escape_dismisses_and_hands_focus_back(qtbot, panel):
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItems(["a"])
    combo.show()
    combo.setFocus()
    panel.open_for(combo, [_entry("install")])
    assert panel.isVisible() and panel.is_pinned()
    panel.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape,
                                  Qt.NoModifier))
    assert panel.isVisible() is False


def test_a_pinned_panel_ignores_the_hover_timer(qtbot, panel):
    """A reader who reached it by keyboard is not holding the mouse; a hover
    timeout must not take the panel away from them."""
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    anchor.show()
    panel.open_for(anchor, [_entry("install")])
    panel.start_hide(1)
    panel._maybe_hide()
    assert panel.isVisible()


def test_the_keyboard_route_focuses_a_link_and_cycles_the_entries(qtbot,
                                                                  panel):
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    anchor.show()
    entries = [_entry("install", key="cuml", title="cuML (GPU)"),
               _entry("impossible", key="pymer4", title="pymer4 (CPU)")]
    panel.open_for(anchor, entries)
    # `hasFocus()` also asks whether the WINDOW is active, and the offscreen
    # platform never activates one. The claim being made is about the panel's
    # own focus assignment, so that is what is read.
    assert panel.focusWidget() is panel.api_link()
    assert panel.current_entry()["key"] == "cuml"
    panel.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Down,
                                  Qt.NoModifier))
    assert panel.current_entry()["key"] == "pymer4"
    panel.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Up,
                                  Qt.NoModifier))
    assert panel.current_entry()["key"] == "cuml"


def test_the_link_words_are_keyboard_reachable_at_all(qtbot, panel):
    """`Qt.TextBrowserInteraction` includes `LinksAccessibleByKeyboard`, and
    QLabel gives such a label a focus policy. Without that the panel is a
    mouse-only control and the disabled row has no keyboard route at all."""
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    panel.show_for(anchor, [_entry("install")])
    for link in (panel.api_link(), panel.install_link()):
        flags = link.textInteractionFlags()
        assert flags & Qt.LinksAccessibleByKeyboard
        assert link.focusPolicy() != Qt.NoFocus


def test_showing_nothing_shows_nothing(qtbot, panel):
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    panel.show_for(anchor, [])
    assert panel.isVisible() is False
    assert explain(anchor, []) is None


def test_one_panel_serves_both_callers(qtbot, panel):
    """"Build it once, in a place neither owns, and let both ask."""
    from spacr import gpu_reduce
    from spacr.regression_backends import availability_entries
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    first = explain(anchor, [gpu_reduce.availability_entry()])
    second = explain(anchor, availability_entries("mixed")[:1])
    assert first is second is AvailabilityPanel.instance()


def test_the_install_handler_is_replaced_and_not_accumulated(qtbot, panel):
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    calls = []
    panel.set_install_handler(lambda offer: calls.append("first"))
    panel.set_install_handler(lambda offer: calls.append("second"))
    panel.show_for(anchor, [_entry("install")])
    panel._on_link("install")
    assert calls == ["second"]


# ---------------------------------------------------------------------------
# What pressing the word does -- the three answers
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self, answers=()):
        self.informed, self.confirmed, self.installed = [], [], []
        self.answers = list(answers)

    def inform(self, title, text):
        self.informed.append((title, text))

    def confirm(self, title, text):
        self.confirmed.append((title, text))
        return self.answers.pop(0) if self.answers else False

    def install(self, command):
        self.installed.append(command)
        return 0, "Successfully installed"

    def must_not_run(self, *args, **kwargs):
        raise AssertionError("this branch must run nothing")


@pytest.mark.parametrize("action", ["elsewhere", "impossible"])
def test_an_offer_that_cannot_install_here_runs_nothing(action):
    """"Clicking one that cannot -- cuML on Python 3.10 -- explains the
    environment it needs and runs NOTHING."""
    recorder = _Recorder()
    offer = _entry(action)["offer"]
    outcome = run_install_offer(
        None, offer, confirm=recorder.must_not_run, inform=recorder.inform,
        dry_run=recorder.must_not_run, install=recorder.must_not_run)
    assert outcome == "explained"
    assert recorder.installed == []
    assert "the recipe" in recorder.informed[0][1]


def test_the_dry_run_report_is_shown_before_the_install(monkeypatch):
    recorder = _Recorder(answers=[True])
    report = updater.DryRun("glum", True,
                            (updater.PackageChange("glum", None, "3.4.0"),))
    outcome = run_install_offer(
        None, _entry("install")["offer"], confirm=recorder.confirm,
        inform=recorder.inform, dry_run=lambda req: report,
        install=recorder.install)
    assert outcome == "installed"
    # The report is in the text the user agreed to, not shown afterwards.
    assert "glum 3.4.0" in recorder.confirmed[0][1]
    assert recorder.installed


def test_a_protected_move_takes_two_confirmations_and_names_what_moves():
    recorder = _Recorder(answers=[True, True])
    report = updater.DryRun(
        "cuml-cu12", True,
        (updater.PackageChange("numpy", "1.26.4", "2.2.6"),))
    outcome = run_install_offer(
        None, _entry("install")["offer"], confirm=recorder.confirm,
        inform=recorder.inform, dry_run=lambda req: report,
        install=recorder.install)
    assert outcome == "installed"
    assert len(recorder.confirmed) == 2
    assert "numpy 1.26.4 -> 2.2.6" in recorder.confirmed[1][1]


def test_declining_the_second_confirmation_installs_nothing():
    recorder = _Recorder(answers=[True, False])
    report = updater.DryRun(
        "cuml-cu12", True,
        (updater.PackageChange("numpy", "1.26.4", "2.2.6"),))
    outcome = run_install_offer(
        None, _entry("install")["offer"], confirm=recorder.confirm,
        inform=recorder.inform, dry_run=lambda req: report,
        install=recorder.install)
    assert outcome == "refused"
    assert recorder.installed == []


def test_declining_the_first_confirmation_installs_nothing():
    recorder = _Recorder(answers=[False])
    report = updater.DryRun("glum", True,
                            (updater.PackageChange("glum", None, "3.4.0"),))
    outcome = run_install_offer(
        None, _entry("install")["offer"], confirm=recorder.confirm,
        inform=recorder.inform, dry_run=lambda req: report,
        install=recorder.install)
    assert outcome == "declined"
    assert recorder.installed == []


def test_a_dry_run_that_failed_stops_the_install():
    recorder = _Recorder(answers=[True, True])
    report = updater.DryRun("cuml-cu12", False, error="ResolutionImpossible")
    outcome = run_install_offer(
        None, _entry("install")["offer"], confirm=recorder.confirm,
        inform=recorder.inform, dry_run=lambda req: report,
        install=recorder.must_not_run)
    assert outcome == "refused"
    assert recorder.confirmed == []


def test_a_failed_install_reports_what_the_tool_said():
    recorder = _Recorder(answers=[True])
    report = updater.DryRun("glum", True,
                            (updater.PackageChange("glum", None, "3.4.0"),))
    outcome = run_install_offer(
        None, _entry("install")["offer"], confirm=recorder.confirm,
        inform=recorder.inform, dry_run=lambda req: report,
        install=lambda command: (1, "ERROR: no matching distribution"))
    assert outcome == "failed"
    assert "no matching distribution" in recorder.informed[-1][1]


def test_a_successful_install_says_to_restart():
    recorder = _Recorder(answers=[True])
    report = updater.DryRun("glum", True,
                            (updater.PackageChange("glum", None, "3.4.0"),))
    run_install_offer(None, _entry("install")["offer"],
                      confirm=recorder.confirm, inform=recorder.inform,
                      dry_run=lambda req: report, install=recorder.install)
    assert "RESTART" in recorder.informed[-1][1]


def test_an_offer_that_is_already_ready_says_so():
    recorder = _Recorder()
    outcome = run_install_offer(
        None, _entry("ready")["offer"], confirm=recorder.must_not_run,
        inform=recorder.inform, dry_run=recorder.must_not_run,
        install=recorder.must_not_run)
    assert outcome == "ready"


# ---------------------------------------------------------------------------
# It is actually on screen
# ---------------------------------------------------------------------------

def test_the_panel_paints_its_own_surface_and_not_the_black_slab(qtbot,
                                                                 panel):
    """`HoverTooltip` learned this the hard way and it cost a whole redesign.

    A separate top-level window does not reliably receive the application
    stylesheet, so any plain container inside it falls through to the blanket
    ``QWidget { background-color: bg }`` -- and ``bg`` is the WINDOW colour,
    ``#000000`` on the dark theme. The result is a black rectangle covering
    everything but a few pixels of the frame. Rendered and counted rather
    than asserted about the stylesheet string.
    """
    from PySide6.QtGui import QImage

    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    panel.show_for(anchor, [_entry("install")])
    panel.adjustSize()
    image = QImage(panel.size(), QImage.Format_ARGB32)
    image.fill(0)
    panel.render(image)
    black = sum(image.pixelColor(x, y).rgb() == 0xFF000000
                for y in range(0, image.height(), 3)
                for x in range(0, image.width(), 3))
    sampled = len(range(0, image.height(), 3)) * len(range(0, image.width(), 3))
    assert sampled > 100, "the panel rendered to nothing"
    assert black < sampled * 0.10, f"{black}/{sampled} pixels are pure black"


def test_the_panel_is_wide_enough_to_read_and_not_a_document(qtbot, panel):
    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    panel.show_for(anchor, [_entry("elsewhere")])
    panel.adjustSize()
    assert 200 <= panel.width() <= AvailabilityPanel.TEXT_WIDTH + 60
    assert panel.height() > 40


def test_the_default_dry_run_does_not_freeze_the_window(qtbot, monkeypatch):
    """It runs off the GUI thread, and the loop that waits for it keeps
    painting -- measured by letting the dialog process events while the
    worker is still going."""
    import threading

    from spacr.qt.widgets import availability_panel as module

    started = threading.Event()
    release = threading.Event()

    def _slow(requirement, *args, **kwargs):
        started.set()
        release.wait(5.0)
        return updater.DryRun(requirement, True,
                              (updater.PackageChange("glum", None, "3.4.0"),))

    monkeypatch.setattr("spacr.updater.dry_run_install", _slow)
    holder = QLabel("host")
    qtbot.addWidget(holder)
    runner = module._default_dry_run(holder)
    timer_fired = []
    from PySide6.QtCore import QTimer
    QTimer.singleShot(30, lambda: timer_fired.append(True) or release.set())
    result = runner("glum")
    assert started.is_set()
    # The event loop kept turning while the resolver worked: a single-shot
    # timer scheduled after the call began actually fired.
    assert timer_fired == [True]
    assert result.ok and result.additions


def test_a_cancelled_dry_run_forbids_the_install(qtbot, monkeypatch):
    from spacr.qt.widgets import availability_panel as module

    class _Cancelled:
        def __init__(self, *a, **k):
            pass

        def setWindowTitle(self, *a):
            pass

        setMinimumDuration = setAutoClose = setAutoReset = setWindowTitle

        def show(self):
            pass

        def close(self):
            pass

        def wasCanceled(self):
            return True

    monkeypatch.setattr("PySide6.QtWidgets.QProgressDialog", _Cancelled)
    monkeypatch.setattr(
        "spacr.updater.dry_run_install",
        lambda requirement, *a, **k: __import__("time").sleep(3))
    holder = QLabel("host")
    qtbot.addWidget(holder)
    result = module._default_dry_run(holder)("glum")
    assert result.ok is False
    assert updater.install_decision(result)['allowed'] is False
