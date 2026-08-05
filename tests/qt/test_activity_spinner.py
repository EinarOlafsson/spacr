"""The background-activity indicator: visible while busy, gone when idle.

Two properties are worth a test and one is worth a benchmark.

1. It reflects **real** state. It is driven by the process-wide run registry,
   not by a flag each caller has to remember to set, so a job nobody told it
   about still turns it.
2. Idle costs **zero**, not "little": the animation timer is stopped, so an
   idle window posts no timer events and schedules no repaints. A spinner
   that keeps turning behind a hidden widget is the bug this asserts against.
3. It waits before it appears, and the wait is a *delay* rather than a guess
   at how long the work will take. The tests for that run real jobs against
   a real wall clock (see the ``delay`` section at the bottom): a one-second
   job must never put anything on screen, a three-second one must, and the
   threshold has to come from the preference rather than from a constant in
   the widget.

Everything above group 3 passes ``delay_ms=0``, because those tests are about
what the spinner does once it is up and would otherwise all be re-testing the
delay.

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
    spinner = ActivitySpinner(auto=False, delay_ms=0)
    qtbot.addWidget(spinner)
    assert not spinner.is_busy()
    assert not spinner.is_spinning()
    assert not spinner.isVisible()


def test_it_appears_while_work_runs_and_goes_away_when_it_stops(qtbot):
    spinner = ActivitySpinner(auto=False, delay_ms=0)
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
    spinner = ActivitySpinner(auto=False, delay_ms=0)
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
    spinner = ActivitySpinner(auto=False, delay_ms=0)
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

    spinner = ActivitySpinner(delay_ms=0)
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
    spinner = ActivitySpinner(auto=False, delay_ms=0)
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


# ---------------------------------------------------------------------------
# The delay before it appears
# ---------------------------------------------------------------------------
# Real jobs, real threads, a real wall clock. A test that drove the delay by
# calling the timeout handler directly would prove the handler works and
# nothing about whether the timer is armed at the right moment, which is the
# only part of this that can go wrong.
#
# The spinner is put inside a shown host, never shown directly, because that
# is how it lives: ``attach_activity_spinner`` inserts it into the button row
# and it is the widget's own decision whether to be visible. A test that
# called ``spinner.show()`` would have made that decision for it and then
# asserted on the answer.

@pytest.fixture
def prefs(monkeypatch, tmp_path):
    """Route the preference store into ``tmp_path`` and prove it landed there.

    ``QSettings("spacr", "qt")`` -- the (organization, application) form the
    module uses -- resolves to the NATIVE location whatever ``setPath`` says,
    so redirecting the class is not isolation. Replacing the accessor is.
    """
    from PySide6.QtCore import QSettings

    from spacr.qt import preferences as module

    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(module, "_settings", lambda: store)
    assert str(tmp_path) in module._settings().fileName()
    return module


@pytest.fixture
def hosted(qtbot):
    """A spinner in a shown parent, the way a screen holds one."""
    # Held for the life of the test: a QWidget whose last Python reference
    # goes out of scope takes its C++ half with it, and the spinner would
    # then be a live QTimer calling into a deleted parent.
    hosts = []

    def build(**kwargs):
        host = QWidget()
        hosts.append(host)
        layout = QHBoxLayout(host)
        spinner = ActivitySpinner(host, **kwargs)
        layout.addWidget(spinner)
        qtbot.addWidget(host)
        host.show()
        qtbot.waitExposed(host)
        assert not spinner.isVisible(), "it came up before any work started"
        return spinner
    return build


def run_for(seconds):
    """A job that occupies a worker thread for ``seconds`` and no longer."""
    def job(_settings):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(0.01)
    return job


def drive(qtbot, spinner, seconds):
    """Run a job of ``seconds`` through the real registry.

    Returns whether the spinner was ever seen on screen, sampled every 25 ms
    for the life of the job -- sampling rather than checking once at the end,
    because a spinner that appeared and vanished mid-job is exactly the flash
    this feature exists to prevent and a single check would miss it.
    """
    from spacr.qt.bridge import make_thread, registry, thread_has_stopped

    seen = False
    thread, _worker = make_thread(run_for(seconds), {}, app_key="job",
                                  journal=False)
    thread.start()
    deadline = time.monotonic() + seconds + 5.0
    while time.monotonic() < deadline:
        qtbot.wait(25)
        seen = seen or spinner.isVisible()
        if thread_has_stopped(thread):
            break
    qtbot.waitUntil(lambda: thread_has_stopped(thread), timeout=10000)
    qtbot.waitUntil(lambda: not registry().active(), timeout=10000)
    qtbot.wait(60)
    return seen


def test_a_short_job_never_flashes_the_spinner(qtbot, prefs, hosted):
    """One second of work against a two-second threshold: nothing appears.

    Sampled throughout, not checked at the end, because "it was gone by the
    time we looked" is the failure this is guarding against.
    """
    prefs.set_spinner_delay(2.0)
    spinner = hosted()
    assert spinner.delay_ms() == 2000, "the preference did not reach it"

    assert not drive(qtbot, spinner, 1.0), "a 1 s job put it on screen"
    assert not spinner.is_spinning()
    assert not spinner.isVisible()


def test_a_long_job_shows_it_and_it_goes_when_the_work_does(qtbot, prefs,
                                                            hosted):
    prefs.set_spinner_delay(2.0)
    spinner = hosted()

    assert drive(qtbot, spinner, 3.0), "a 3 s job never showed the spinner"
    # ...and the moment the work stops, so does it. No lingering.
    qtbot.waitUntil(lambda: not spinner.is_busy(), timeout=5000)
    assert not spinner.isVisible()
    assert not spinner.is_spinning()


def test_the_threshold_comes_from_the_preference(qtbot, prefs, hosted):
    """The same job, either side of a threshold the test moved."""
    prefs.set_spinner_delay(4.0)
    patient = hosted()
    assert patient.delay_ms() == 4000
    assert not drive(qtbot, patient, 1.5), "4 s threshold, 1.5 s job"

    prefs.set_spinner_delay(0.5)
    eager = hosted()
    assert eager.delay_ms() == 500
    assert drive(qtbot, eager, 1.5), "0.5 s threshold, 1.5 s job"


def test_while_it_waits_it_costs_nothing(qtbot, prefs, hosted):
    """The delay is not a hidden animation with the paint suppressed."""
    prefs.set_spinner_delay(2.0)
    spinner = hosted(auto=False)

    spinner.set_busy(True)
    assert spinner.is_waiting()
    assert not spinner.is_spinning()
    assert not spinner.isVisible()
    before = spinner.frames_painted
    qtbot.wait(600)
    assert spinner.frames_painted == before, "it painted while waiting"

    # ...and it does come up, so the assertion above is not passing on a
    # spinner that is simply broken.
    qtbot.waitUntil(lambda: spinner.isVisible(), timeout=4000)
    assert spinner.is_spinning()
    assert not spinner.is_waiting()


def test_a_second_job_does_not_push_the_deadline_back(qtbot, prefs, hosted):
    """The delay measures how long *work* has been going, not how long the
    longest single job has.

    Three hundred milliseconds of work, four times over with no gap, is more
    than a second during which spaCR was continuously busy -- and the reader
    is owed the indicator for it. Restarting the timer on every new job
    (which is what a bare ``QTimer.start()`` in the busy branch does) would
    mean a queue of short jobs never showed anything at all.
    """
    prefs.set_spinner_delay(1.0)
    spinner = hosted(auto=False)

    spinner.set_busy(True)
    for _ in range(4):
        qtbot.wait(300)
        spinner.set_busy(True)          # "another job started"
        assert spinner.is_busy()
    assert spinner.isVisible(), "continuous work never earned the spinner"
    assert spinner.is_spinning()


def test_going_idle_disarms_the_wait(qtbot, prefs, hosted):
    """Work that stops before the delay fires must not show a spinner a
    second later, on a screen that is now doing nothing."""
    prefs.set_spinner_delay(1.0)
    spinner = hosted(auto=False)

    spinner.set_busy(True)
    assert spinner.is_waiting()
    spinner.set_busy(False)
    assert not spinner.is_waiting()
    qtbot.wait(1400)
    assert not spinner.isVisible()
    assert not spinner.is_spinning()


def test_zero_means_show_it_at_once(qtbot, prefs, hosted):
    """The debugging setting, and the one the rest of this file relies on."""
    prefs.set_spinner_delay(0.0)
    spinner = hosted(auto=False)
    assert spinner.delay_ms() == 0
    spinner.set_busy(True)
    assert spinner.isVisible()
    assert spinner.is_spinning()
    assert not spinner.is_waiting()


def test_the_delay_preference_defaults_to_two_seconds(prefs):
    assert prefs.get_spinner_delay() == 2.0
    assert prefs.DEFAULT_SPINNER_DELAY == 2.0


def test_the_delay_preference_round_trips_and_clamps(prefs):
    prefs.set_spinner_delay(3.5)
    assert prefs.get_spinner_delay() == pytest.approx(3.5)
    prefs.set_spinner_delay(-4)
    assert prefs.get_spinner_delay() == prefs.SPINNER_DELAY_MIN
    prefs.set_spinner_delay(9999)
    assert prefs.get_spinner_delay() == prefs.SPINNER_DELAY_MAX
    prefs._settings().setValue("prefs/spinner_delay", "not a number")
    assert prefs.get_spinner_delay() == prefs.DEFAULT_SPINNER_DELAY


def test_the_dialog_offers_the_delay_and_saves_it(qtbot, prefs,
                                                  qt_theme_applied):
    from PySide6.QtWidgets import QDialogButtonBox, QSlider

    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    slider = dialog.findChild(QSlider, "SpinnerDelay")
    assert slider is not None, "no delay control in Preferences"
    assert slider.value() == 20, "does not open on the two-second default"

    slider.setValue(35)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert prefs.get_spinner_delay() == pytest.approx(3.5)
