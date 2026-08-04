"""The background-activity indicator: visible while busy, gone when idle.

Two properties are worth a test and one is worth a benchmark.

1. It reflects **real** state. It is driven by the process-wide run registry,
   not by a flag each caller has to remember to set, so a job nobody told it
   about still turns it.
2. Idle costs **zero**, not "little": the animation timer is stopped, so an
   idle window posts no timer events and schedules no repaints. A spinner
   that keeps turning behind a hidden widget is the bug this asserts against.

The benchmark reports the way :mod:`spacr.qt.widgets.dna_rain` reports its
own (0.53 ms a frame, 3.2 % of one core at 60 fps), so the two numbers can be
compared without re-deriving either.
"""
from __future__ import annotations

import time

import pytest

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from spacr.qt.widgets.activity_spinner import (
    ActivitySpinner, attach_activity_spinner)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def test_a_fresh_spinner_is_hidden_and_still(qtbot):
    """Construction alone must not start an animation."""
    spinner = ActivitySpinner(auto=False)
    qtbot.addWidget(spinner)
    assert not spinner.is_busy()
    assert not spinner.is_spinning()
    assert not spinner.isVisible()


def test_it_appears_while_work_runs_and_goes_away_when_it_stops(qtbot):
    spinner = ActivitySpinner(auto=False)
    qtbot.addWidget(spinner)
    spinner.show()
    qtbot.waitExposed(spinner)

    spinner.set_busy(True)
    assert spinner.isVisible()
    assert spinner.is_spinning()

    spinner.set_busy(False)
    assert not spinner.isVisible()
    assert not spinner.is_spinning()


def test_an_idle_spinner_paints_nothing_at_all(qtbot):
    """The "zero idle cost" claim, asserted rather than assumed.

    A stopped ``QTimer`` posts no events, so no frame is painted. Counting
    frames over a real second is the only way to tell that apart from a
    timer that is running but drawing something invisible.
    """
    spinner = ActivitySpinner(auto=False)
    qtbot.addWidget(spinner)
    spinner.show()
    qtbot.waitExposed(spinner)
    qtbot.wait(50)

    before = spinner.frames_painted
    qtbot.wait(600)
    assert spinner.frames_painted == before, (
        "an idle spinner painted "
        f"{spinner.frames_painted - before} frames in 600 ms")
    assert not spinner.is_spinning()

    # ...and it does animate when there is something to animate, or the
    # assertion above would pass on a spinner that is simply broken.
    spinner.set_busy(True)
    qtbot.wait(300)
    assert spinner.frames_painted > before


def test_hiding_the_screen_stops_the_animation(qtbot):
    """Switching module or minimising must not leave it spinning unseen."""
    spinner = ActivitySpinner(auto=False)
    qtbot.addWidget(spinner)
    spinner.show()
    qtbot.waitExposed(spinner)
    spinner.set_busy(True)
    assert spinner.is_spinning()

    spinner.hide()
    assert not spinner.is_spinning()

    spinner.show()
    assert spinner.is_spinning()          # still busy, so it resumes


# ---------------------------------------------------------------------------
# Driven by the real registry
# ---------------------------------------------------------------------------

def test_it_follows_the_run_registry_rather_than_a_flag(qtbot, tmp_path):
    """A job started through ``make_thread`` turns it on, with no wiring.

    This is the difference between an indicator that is right and one that
    is right when every caller remembers. Nothing here tells the spinner
    anything -- it reads the same registry ``active_jobs()`` is counting.
    """
    from spacr.qt.bridge import (make_thread, registry,
                                 thread_has_stopped)

    spinner = ActivitySpinner()
    qtbot.addWidget(spinner)
    spinner.show()
    qtbot.waitExposed(spinner)
    assert not spinner.is_busy()

    gate = {"go": False}

    def slow(_settings):
        while not gate["go"]:
            time.sleep(0.005)

    thread, worker = make_thread(slow, {}, app_key="a slow job",
                                 journal=False)
    thread.start()
    try:
        qtbot.waitUntil(lambda: spinner.is_busy(), timeout=5000)
        assert spinner.isVisible()
        assert spinner.is_spinning()
        # The tooltip says what is running, by name.
        assert "a slow job" in spinner.toolTip()
    finally:
        gate["go"] = True
    # `thread_has_stopped`, not `isRunning()`: `make_thread` chains
    # `deleteLater` off `finished`, so by the time this runs the C++
    # QThread may already be gone and asking it anything raises.
    qtbot.waitUntil(lambda: thread_has_stopped(thread), timeout=10000)
    qtbot.waitUntil(lambda: not registry().active(), timeout=10000)
    qtbot.waitUntil(lambda: not spinner.is_busy(), timeout=5000)
    assert not spinner.isVisible()
    assert not spinner.is_spinning()


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def _clear_console_host(parent=None):
    """A stand-in for an AppScreen's actions row: a button in a box layout
    that its owner exposes as ``_btn_clear``."""
    host = QWidget(parent)
    outer = QHBoxLayout(host)
    row = QWidget(host)
    layout = QHBoxLayout(row)
    layout.addWidget(QPushButton("Run", row))
    button = QPushButton("Clear console", row)
    layout.addWidget(button)
    layout.addWidget(QPushButton("File as issue", row))
    outer.addWidget(row)
    host._btn_clear = button
    return host, row, button


def test_it_installs_immediately_to_the_right_of_clear_console(qtbot):
    host, row, button = _clear_console_host()
    qtbot.addWidget(host)

    spinner = attach_activity_spinner(host)
    assert spinner is not None

    layout = row.layout()
    assert layout.indexOf(spinner) == layout.indexOf(button) + 1


def test_installing_twice_does_not_add_a_second_spinner(qtbot):
    """It is called from an event filter that fires on every Show/Polish."""
    host, row, _button = _clear_console_host()
    qtbot.addWidget(host)

    first = attach_activity_spinner(host)
    second = attach_activity_spinner(host)
    assert first is second
    assert len(host.findChildren(ActivitySpinner)) == 1


def test_a_screen_without_the_button_is_not_an_error(qtbot):
    """Annotate, the Database Browser and every non-AppScreen surface."""
    plain = QWidget()
    qtbot.addWidget(plain)
    assert attach_activity_spinner(plain) is None


def test_it_is_found_from_a_descendant_not_only_the_screen(qtbot):
    """The button-role filter hands it the *button*, not the screen."""
    host, _row, button = _clear_console_host()
    qtbot.addWidget(host)
    assert attach_activity_spinner(button) is not None


def test_the_button_filter_installs_it_without_the_screen_asking(qtbot):
    """End to end through the real application event filter.

    ``spacr/qt/screens/app_screen.py`` builds the actions row and is owned
    elsewhere, so the spinner attaches itself through the same central
    filter that already tags every button in the application.
    """
    from PySide6.QtWidgets import QApplication
    from spacr.qt.button_roles import install_button_roles

    host, row, button = _clear_console_host()
    qtbot.addWidget(host)
    install_button_roles(QApplication.instance())
    host.show()
    qtbot.waitExposed(host)
    qtbot.waitUntil(lambda: bool(row.findChildren(ActivitySpinner)),
                    timeout=3000)
    spinner = row.findChildren(ActivitySpinner)[0]
    assert row.layout().indexOf(spinner) == row.layout().indexOf(button) + 1


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def test_a_row_built_parentless_and_reparented_still_gets_one(qtbot):
    """The exact shape ``AppScreen`` builds, and the bug it caused.

    ``AppScreen._build_runtime_panel`` creates its actions row as a
    *parentless* ``QWidget`` and reparents it when it is added to a layout.
    The button's first Polish therefore arrives while the walk up to the
    screen cannot reach it. The filter used to latch on that first attempt
    and never look again, so the real application got no spinner at all
    while every test with a pre-parented row passed.
    """
    from PySide6.QtWidgets import QApplication, QVBoxLayout
    from spacr.qt.button_roles import install_button_roles

    install_button_roles(QApplication.instance())

    row = QWidget()                       # no parent, exactly like AppScreen
    layout = QHBoxLayout(row)
    button = QPushButton("Clear console", row)
    layout.addWidget(button)
    button.ensurePolished()               # the premature Polish

    screen = QWidget()                    # the screen it will belong to
    qtbot.addWidget(screen)
    QVBoxLayout(screen).addWidget(row)    # reparents the row
    screen._btn_clear = button

    screen.show()
    qtbot.waitExposed(screen)
    qtbot.waitUntil(lambda: bool(row.findChildren(ActivitySpinner)),
                    timeout=3000)
    spinner = row.findChildren(ActivitySpinner)[0]
    assert layout.indexOf(spinner) == layout.indexOf(button) + 1


def test_the_spinner_frame_costs_microseconds(qtbot):
    """Report the per-frame cost the way the DNA rain reports its own.

    Painted into an offscreen ``QImage`` so the number is the widget's
    drawing, not the platform's compositing. Prints the figures; asserts
    only a ceiling loose enough to survive CI, because the point of the
    number is the comparison, not the gate.
    """
    spinner = ActivitySpinner(auto=False)
    qtbot.addWidget(spinner)
    spinner.set_busy(True)

    image = QImage(QSize(16, 16), QImage.Format_ARGB32_Premultiplied)
    frames = 400
    # Warm up: the first paint resolves the palette and builds the pens.
    for _ in range(20):
        spinner.render(image)

    start = time.perf_counter()
    for _ in range(frames):
        spinner._advance()
        spinner.render(image)
    per_frame = (time.perf_counter() - start) / frames

    at_20fps = per_frame * 20 * 100
    print(f"\nActivitySpinner: {per_frame * 1000:.3f} ms a frame, "
          f"{at_20fps:.2f} % of one core at 16 px and 20 fps, "
          f"0.00 % while idle")
    assert per_frame < 0.005, (
        f"{per_frame * 1000:.2f} ms a frame is not a cheap indicator")


def test_the_gif_it_replaces_is_measurably_the_wrong_asset():
    """The measurement behind not reusing ``loading_spinner.gif``.

    Kept as a test so the numbers in the module docstring cannot rot into a
    claim nobody can check. If someone re-encodes the asset to something a
    16 px indicator can actually use, this fails and the decision gets
    revisited on purpose.
    """
    import os

    import spacr

    gif = os.path.join(os.path.dirname(spacr.__file__), "resources",
                       "icons", "loading_spinner.gif")
    if not os.path.exists(gif):
        pytest.skip("the spinner GIF is not in this checkout")

    from PIL import Image

    with Image.open(gif) as movie:
        width, height = movie.size
        frames = 0
        try:
            while True:
                movie.seek(movie.tell() + 1)
                frames += 1
        except EOFError:
            frames += 1
        movie.seek(0)
        first = movie.convert("RGBA")

    decoded_mb = width * height * 4 * frames / 1024 / 1024
    assert (width, height) == (800, 600)
    assert frames == 144
    assert decoded_mb > 200, decoded_mb        # ~264 MB of pixels

    # And it has no transparency at all: it is a video OF a spinner, not a
    # sprite. Dropped beside a button it paints an opaque black square.
    alphas = {px[3] for px in list(first.getdata())[::997]}
    assert alphas == {255}, alphas
    assert first.getpixel((0, 0))[:3] == (0, 0, 0)
