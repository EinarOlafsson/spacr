"""Every widget in the window resolves to the theme that is in force.

THE GUARD 380'S LAST LEVER NEEDS BEFORE IT CAN BE WRITTEN. That change --
sheeting the chrome and the visible screen instead of the whole window --
is worth 4.8x on top of the 4x already banked, and its failure mode is a
widget nobody reached. 407 of the window's widgets are chrome, and only TWO
of them are reachable from the central widget, so an implementation that
walked the obvious tree would leave 405 unstyled and LOOK ALMOST RIGHT.

    "Almost right" is the shape a count cannot see and a screenshot of the
    wrong screen does not contain. This asks every widget directly.

WHAT IT HOLDS. A stylesheet colour resolves into a widget's palette, so an
offscreen widget reports the theme it is wearing without anything having to
be painted or looked at -- the same mechanism
`test_a_dialog_never_opens_in_the_previous_theme` uses for windows,
generalised from "the windows" to "everything in them".

WHY A SENTINEL COLOUR RATHER THAN THE REAL PALETTE. A real theme's
foreground appears in many rules and a widget can arrive at it by accident;
a colour no palette contains cannot be arrived at by accident, so a widget
wearing it was reached and a widget not wearing it was not.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtGui import QPalette                            # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QWidget   # noqa: E402

#: Two foregrounds no spaCR palette contains, so a label wearing one was
#: reached by the sheet that carries it and not by a coincidence.
FIRST = "QLabel { color: rgb(11, 22, 33); }"
SECOND = "QLabel { color: rgb(200, 100, 50); }"
FIRST_HEX = "#0b1621"
SECOND_HEX = "#c86432"


def _labels(root):
    """A label under every widget in ``root``, so each can be asked.

    A label is the probe because `color` is what a stylesheet rule can set
    and a palette can report. Parenting one under each widget asks that
    widget's branch of the cascade rather than the window's.
    """
    out = []
    for widget in [root] + root.findChildren(QWidget):
        try:
            if widget.isWindow() and widget is not root:
                continue        # its own window; the dialog guard owns it
            label = QLabel(widget)
            out.append((widget, label))
        except RuntimeError:
            continue
    return out


def _resolved(label) -> str:
    app = QApplication.instance()
    label.ensurePolished()
    app.processEvents()
    return label.palette().color(QPalette.WindowText).name()


@pytest.fixture
def sheeted(qtbot):
    from spacr.qt import theme

    app = QApplication.instance()
    app.setStyleSheet("")
    yield theme.apply_stylesheet_per_window
    theme.apply_stylesheet_per_window(app, "")
    app.setStyleSheet("")


def test_every_widget_in_a_window_wears_the_sheet(sheeted, qtbot):
    """THE ONE THE PER-SCREEN CHANGE HAS TO KEEP.

    Not "the window has the sheet" -- every widget under it, including the
    ones a narrower implementation would forget.
    """
    app = QApplication.instance()
    window = QWidget()
    qtbot.addWidget(window)
    depth_one = QWidget(window)
    depth_two = QWidget(depth_one)
    depth_three = QWidget(depth_two)
    window.show()

    sheeted(app, SECOND)

    probes = _labels(window)
    assert len(probes) >= 4, "the fixture built nothing to check"
    unreached = [type(owner).__name__ for owner, label in probes
                 if _resolved(label) != SECOND_HEX]
    assert not unreached, (
        f"{len(unreached)} widget(s) did not get the sheet: {unreached[:6]}")
    assert depth_three is not None


def test_a_widget_added_after_the_change_wears_it_too(sheeted, qtbot):
    """A branch built later is the half a dirty-mark has to cover."""
    app = QApplication.instance()
    window = QWidget()
    qtbot.addWidget(window)
    window.show()
    sheeted(app, SECOND)

    late = QWidget(window)
    label = QLabel(late)
    assert _resolved(label) == SECOND_HEX, (
        "a widget parented in after the theme change is wearing the "
        "previous theme")


def test_the_guard_can_see_a_widget_that_was_missed(sheeted, qtbot):
    """PROOF IT WORKS, by doing the wrong thing on purpose.

    Without this the two tests above would pass under any implementation
    that sheets the window, and say nothing about one that does not.
    """
    app = QApplication.instance()
    window = QWidget()
    qtbot.addWidget(window)
    reached = QWidget(window)
    window.show()

    # The narrow implementation's mistake: sheet one branch, not the window.
    reached.setStyleSheet(SECOND)
    missed = QWidget(window)

    assert _resolved(QLabel(reached)) == SECOND_HEX
    assert _resolved(QLabel(missed)) != SECOND_HEX, (
        "the guard cannot tell a reached widget from a missed one, so it "
        "would pass for a change that forgot 405 of them")


def test_a_second_change_reaches_everything_again(sheeted, qtbot):
    """Once is not the contract; every time is."""
    app = QApplication.instance()
    window = QWidget()
    qtbot.addWidget(window)
    QWidget(QWidget(window))
    window.show()

    sheeted(app, FIRST)
    assert all(_resolved(label) == FIRST_HEX
               for _owner, label in _labels(window))
    sheeted(app, SECOND)
    stale = [type(owner).__name__ for owner, label in _labels(window)
             if _resolved(label) != SECOND_HEX]
    assert not stale, f"{len(stale)} widget(s) kept the previous theme"


@pytest.mark.slow
def test_the_real_window_wears_one_theme_everywhere(sheeted, qtbot):
    """THE ACTUAL TARGET, on a real MainWindow with modules open.

    The synthetic trees above state the contract; this is the shape the
    per-screen change would be applied to, and the one where "almost
    right" hides. 407 of this window's widgets are chrome and 7,595 are
    inside screens, only one of which is visible -- so an implementation
    that reaches the visible screen and the obvious containers can leave
    hundreds of chrome widgets behind and look fine in a screenshot of the
    screen it did reach.

    PROBES WITH FRESH LABELS, ON A SAMPLE. Reading the colour of the
    labels that are already there does NOT measure what this test needs:
    146 of 480 of them legitimately carry their own colour -- an
    `ElidingLabel`, a `HomeSectionNote` -- so "not wearing the sentinel"
    means "overridden" as often as it means "never reached", and the test
    cannot tell those apart. A NEW label with no stylesheet of its own
    resolves purely from the cascade at the widget it is parented under,
    which is the question. Sampled rather than exhaustive because eight
    thousand probes would cost seconds and change the tree it measures.
    """
    from spacr.qt import register_self_registering_modules
    from spacr.qt.app import MainWindow

    app = QApplication.instance()
    register_self_registering_modules()
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    for _ in range(60):
        app.processEvents()
    for key in ("mask", "measure"):
        try:
            window._on_nav_selected(key)
        except Exception:                                    # noqa: BLE001
            continue
        for _ in range(60):
            app.processEvents()

    sheeted(app, SECOND)

    hosts = [w for w in window.findChildren(QWidget) if not w.isWindow()]
    assert len(hosts) > 500, (
        f"only {len(hosts)} widgets in the window; this test is not "
        f"measuring the tree it thinks it is")
    step = max(1, len(hosts) // 200)
    # A HOST THAT SETS ITS OWN COLOUR IS NOT A MISSED HOST. Its rule is
    # appended after the global sheet and wins for its subtree, so a probe
    # under it reports that colour however the sheet was delivered --
    # `ElidingLabel` and `FlatComboBox` are the two that do it here. The
    # question is whether the cascade REACHED the widget, and those cannot
    # answer it either way.
    sample = [host for host in hosts[::step]
              if "color" not in (host.styleSheet() or "").lower()]
    probes = [(host, QLabel(host)) for host in sample]
    stale = [type(host).__name__ for host, label in probes
             if _resolved(label) != SECOND_HEX]
    assert not stale, (
        f"{len(stale)} of {len(probes)} sampled widgets did not receive "
        f"the sheet: {sorted(set(stale))[:8]}")


@pytest.mark.slow
def test_the_guard_catches_the_mistake_the_per_screen_change_would_make(
        sheeted, qtbot):
    """PROOF, ON THE REAL WINDOW, that this is worth having.

    380's remaining lever sheets an enumerated set instead of the window,
    and its failure mode is a root nobody listed. The measured shape of
    that mistake is specific: the chrome is 407 widgets and only TWO are
    reachable from the central widget, so an implementation that walks the
    central widget's tree leaves 405 behind.

    THIS DOES EXACTLY THAT ON PURPOSE and asserts the probe notices.
    Without it the test above would pass under any implementation that
    happens to sheet the window, and would say nothing about the one that
    does not.
    """
    from spacr.qt import register_self_registering_modules
    from spacr.qt.app import MainWindow

    app = QApplication.instance()
    register_self_registering_modules()
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    for _ in range(60):
        app.processEvents()

    # THE NARROW IMPLEMENTATION'S MISTAKE: the central widget's tree only.
    central = window.centralWidget()
    if central is None:
        pytest.skip("this window has no central widget to be narrow about")
    central.setStyleSheet(SECOND)

    hosts = [w for w in window.findChildren(QWidget)
             if not w.isWindow()
             and "color" not in (w.styleSheet() or "").lower()]
    outside = [w for w in hosts if not central.isAncestorOf(w)]
    assert outside, "every widget is under the central widget here"

    step = max(1, len(outside) // 60)
    missed = [type(host).__name__ for host in outside[::step]
              if _resolved(QLabel(host)) != SECOND_HEX]
    assert missed, (
        "the guard cannot tell a reached widget from a chrome widget the "
        "narrow implementation forgot, so it would pass for a change that "
        "left 405 of them unstyled")
