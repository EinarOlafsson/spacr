"""The backdrop while a run is going: what lagged, and what stopped lagging.

The complaint (instruction 126) was "when something is running after hitting
run, the theme starts lagging". It reproduces, and the cause is not the one
the instruction listed first. Measured on a real X server (Xvfb 1920x1080x24),
a real ``QMainWindow`` over a real ``ConsolePanel`` with the real spaCR
stylesheet, a real Qt event loop, ``blobs`` at the shipped 24 fps cap, best of
five interleaved rounds because the box is shared:

===============================================  =========  ===========
condition                                        delivered   GUI paint
===============================================  =========  ===========
idle                                              25.0 fps      2.04 ms
a numpy thread (1024² matmul + FFT), flat out     24.5 fps      2.10 ms
200 console lines a second, nothing else          24.9 fps      1.27 ms
**a pure-Python thread**                          24.5 fps   **17.21 ms**
a worker doing Python work *and* printing         17.3 fps     17.16 ms
===============================================  =========  ===========

So of the three causes the instruction named, **CPU saturation is innocent**
(numpy releases the interpreter lock; a core burning flat out costs the
backdrop nothing) and **the signal flood is innocent on its own** (200 lines a
second cost nothing measurable; it only bites above a few thousand, where the
console's own per-line work saturates the GUI thread and no change to this
module can help). What is guilty is **the interpreter lock**: identical
drawing work, eleven times slower, because the shading pass is Python and
numpy and a Python worker is holding the lock.

There is a fourth thing the instruction does not name and this file records
because it doubles the bill: **the frame-rate cap is not a cap.** The console
sits on a translucent surface over the backdrop, so every line it prints
exposes the widget and Qt asks it for a whole frame — measured at 0.99 ambient
repaints per console line *with the animation timer stopped*. Each of those
used to be a full shading pass.

The fix is :class:`~spacr.qt.widgets.ambient._FrameProducer`: the shading pass
moves to a thread and the GUI thread does one ``drawImage``. It does not make
the shading faster — same lock, same Python — it changes who waits. Before and
after, same process, interleaved, HEAD's module exec'd from git beside the
working tree's:

=============================  ===================  ===================
condition (``cells``, 1080p)   before               after
=============================  ===================  ===================
idle                           24.9 fps / 2.53 ms   25.0 fps / 1.52 ms
one Python thread              19.9 fps / 33.23 ms  24.9 fps / 1.76 ms
worker + 20 lines a second     15.0 fps / 18.80 ms  25.7 fps / 1.49 ms
=============================  ===================  ===================

``blobs``, the default, goes 17.3 fps to 24.7 on the same worker. What is
*not* fixed is a genuinely chatty run: at 200 lines a second both land at
4 fps, because by then the GUI thread is inside ``ConsolePanel`` and not
inside this widget at all. Coalescing ``line_ready`` is a separate change in
files this one does not own.

Every test here drives real widgets. The two that need wall-clock time are
marked ``heavy`` so ``-m "not heavy"`` deselects them.
"""
from __future__ import annotations

import statistics
import threading
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor                             # noqa: E402
from PySide6.QtWidgets import QWidget                        # noqa: E402

from spacr.qt.widgets.ambient import (AMBIENT_THEMES,        # noqa: E402
                                      AmbientWidget,
                                      default_palette_for, make_engine)

pytestmark = pytest.mark.qt

#: The page colour every documented figure above was taken on.
DARK = "#0f1115"

#: The size those figures were taken at. Worth keeping: the blit is bound by
#: destination pixels, so a 200x200 test would measure a different thing.
W, H = 1920, 1080

#: The five themes that shade into a buffer, i.e. the five that have a frame
#: to hand to another thread. ``drift`` has none and keeps the synchronous
#: path — see :meth:`AmbientWidget._start_producer` for the numbers behind
#: that exclusion.
BUFFERED_THEMES = tuple(t for t in AMBIENT_THEMES if t != "drift")

#: What a repeated frame is allowed to cost the GUI thread, in ms.
#:
#: A repeat is ``fillRect`` plus one ``drawImage``, measured at 1.4-1.8 ms at
#: 1080p under every load tried. The number this is really guarding against is
#: not 2 ms versus 5: it is that an implementation which *waited* for the
#: shading thread would sit here for as long as the lock is held, which the
#: test below holds for ten seconds.
REPEAT_CEILING_MS = 50.0

#: What a frame is allowed to cost the GUI thread while a Python worker runs.
#:
#: Measured 1.49 ms for ``cells`` at 1080p with the producer, against 18.80 ms
#: without it and 33.23 ms with an unbroken Python thread. Sits at four times
#: the measurement and a third of the failure.
LOADED_PAINT_CEILING_MS = 6.0


def _python_worker(stop: threading.Event) -> None:
    """Pure-Python work in another thread, which is the load that hurts.

    Deliberately not numpy: numpy releases the interpreter lock and measured
    zero effect on the backdrop. spaCR's pipeline code is Python around numpy,
    and it is the Python half that starves the GUI thread.
    """
    total = 0
    while not stop.is_set():
        for i in range(20000):
            total += i * i % 7
    return total


class _TimedBackdrop(AmbientWidget):
    """A real backdrop that records what each frame cost the GUI thread."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.costs = []

    def paintEvent(self, event):        # noqa: N802 (Qt override)
        started = time.perf_counter()
        super().paintEvent(event)
        self.costs.append((time.perf_counter() - started) * 1000.0)


def _shading_threads() -> int:
    """How many ambient shading threads are alive in this process right now.

    Counted by name rather than by asking a widget, because the failure this
    guards against is a thread nobody has a handle on any more.
    """
    return sum(1 for t in threading.enumerate()
               if t.name == "spacr-ambient-shade" and t.is_alive())


def _shown(qtbot, theme: str = "blobs", cls=AmbientWidget):
    """A real, visible backdrop at the size the figures were measured at."""
    widget = cls(theme=theme, palette=default_palette_for(theme),
                 background=DARK, seed=4242)
    qtbot.addWidget(widget)
    widget.resize(W, H)
    widget.show()
    qtbot.waitExposed(widget)
    # The platform defers the very first paint until it has actually exposed
    # the window, and a ``repaint()`` issued before that is swallowed rather
    # than served — which reads in a failure as "the widget never painted"
    # and sends the reader looking in the wrong module. Wait for the first
    # frame, so every test below starts from a backdrop that is really up.
    qtbot.waitUntil(lambda: widget.frames_painted > 0, timeout=10000)
    return widget


# ---------------------------------------------------------------------------
# The claim the whole change rests on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theme", BUFFERED_THEMES)
def test_the_shading_thread_draws_exactly_what_the_gui_thread_would(theme):
    """Moving the shading off the GUI thread must not move a single pixel.

    This is the test that lets the change claim "no engine moved". It is also
    the one that fails the day somebody puts a ``QPixmap``, a ``QFontMetrics``
    or anything else with GUI-thread affinity inside an engine — which would
    be undefined behaviour on the shading thread rather than merely a
    different picture, so it is worth catching here rather than in a crash
    report from somebody with a slow machine.
    """
    engine = make_engine(theme, default_palette_for(theme), DARK, seed=99)
    engine.set_time(12.5)
    here = engine.shade(W, H)

    box = {}

    def shade_over_there():
        box["image"] = engine.shade(W, H)

    thread = threading.Thread(target=shade_over_there)
    thread.start()
    thread.join(timeout=30.0)
    assert not thread.is_alive(), "the shading thread never finished"
    there = box["image"]

    assert (there.width(), there.height()) == (here.width(), here.height())
    assert bytes(there.constBits()) == bytes(here.constBits()), \
        f"{theme} shades differently off the GUI thread"

    # The comparison has to be capable of failing, or it asserts nothing: the
    # same engine at a different clock is a different picture.
    engine.set_time(40.0)
    assert bytes(engine.shade(W, H).constBits()) != bytes(here.constBits())


def test_the_two_halves_of_a_frame_compose_back_into_the_whole_frame(qtbot):
    """``paint`` is ``shade`` then ``blit``, and the split proves it.

    The engines were split at a seam, not rewritten, so the synchronous path
    every existing test measures has to still produce what it produced. Drawn
    both ways into two images and compared byte for byte.
    """
    from PySide6.QtGui import QImage, QPainter

    for theme in BUFFERED_THEMES:
        engine = make_engine(theme, default_palette_for(theme), DARK, seed=11)
        engine.set_time(6.25)

        whole = QImage(640, 400, QImage.Format_RGB32)
        whole.fill(QColor(DARK))
        painter = QPainter(whole)
        engine.paint(painter, 640, 400)
        painter.end()

        halves = QImage(640, 400, QImage.Format_RGB32)
        halves.fill(QColor(DARK))
        painter = QPainter(halves)
        engine.blit(painter, engine.shade(640, 400), 640, 400)
        painter.end()

        assert bytes(halves.constBits()) == bytes(whole.constBits()), \
            f"{theme}: shade+blit is not what paint draws"


# ---------------------------------------------------------------------------
# Degrade, never stutter
# ---------------------------------------------------------------------------

def test_a_frame_that_is_not_ready_is_repeated_and_never_waited_for(qtbot):
    """The GUI thread must not block on a frame somebody else is shading.

    Instruction 126's third requirement, asserted as the thing it actually
    means. The engine lock is taken and *held* from another thread, so the
    shading thread cannot publish anything at all for as long as the test
    likes. Every paint in that window still has to complete, still has to put
    a picture up, and still has to be counted as a repeat — an implementation
    that waited for a fresh frame would sit in the first ``repaint()`` until
    the hold is released, which is exactly the frozen interface being
    reported.
    """
    widget = _shown(qtbot, theme="cells")
    qtbot.waitUntil(lambda: widget.frames_shaded() > 0, timeout=10000)

    holding, release = threading.Event(), threading.Event()

    def hold_the_engine():
        with widget._engine_lock:
            holding.set()
            release.wait(timeout=10.0)

    hog = threading.Thread(target=hold_the_engine, daemon=True)
    hog.start()
    assert holding.wait(timeout=10.0), "could not take the engine lock"
    try:
        # A frame the thread finished shading just before the lock was taken
        # is published a moment after it releases it, so one genuinely fresh
        # frame can still arrive here. Let it, then start counting from a
        # state where nothing more can.
        qtbot.wait(120)
        before_painted = widget.frames_painted
        before_repeated = widget.repeated_frames
        before_shaded = widget.frames_shaded()
        first = widget.grab().toImage()
        worst = 0.0
        for _ in range(10):
            started = time.perf_counter()
            widget.repaint()
            worst = max(worst, (time.perf_counter() - started) * 1000.0)
        last = widget.grab().toImage()

        painted = widget.frames_painted - before_painted
        repeated = widget.repeated_frames - before_repeated
    finally:
        release.set()
        hog.join(timeout=10.0)

    assert painted >= 10, "the GUI thread stopped painting"
    assert repeated == painted, \
        "a paint with nothing new to show was not counted as a repeat"
    assert widget.frames_shaded() == before_shaded, \
        "something shaded a frame while the engine was locked away"
    assert bytes(last.constBits()) == bytes(first.constBits()), \
        "a repeated frame is supposed to be the same frame"
    assert worst < REPEAT_CEILING_MS, (
        f"a paint took {worst:.1f} ms with the engine locked away; the GUI "
        f"thread is waiting for the shading thread")


def test_a_slot_being_written_is_skipped_rather_than_waited_on(qtbot):
    """The frame slot is the last lock a paint could still have blocked on.

    The engine lock is never waited on, but reading the published frame takes
    a lock of its own, and the shading thread holds it while it swaps a frame
    in. A blocking read there would be the same bug in miniature — rare,
    brief, and impossible to reproduce from a bug report. So the read refuses
    the lock instead, answers "nothing new", and the paint repeats.
    """
    widget = _shown(qtbot)
    qtbot.waitUntil(lambda: widget.frames_shaded() > 0, timeout=10000)
    producer = widget._producer_box[0]

    holding, release = threading.Event(), threading.Event()

    def hold_the_slot():
        with producer._frame_lock:
            holding.set()
            release.wait(timeout=10.0)

    hog = threading.Thread(target=hold_the_slot, daemon=True)
    hog.start()
    assert holding.wait(timeout=10.0), "could not take the frame slot"
    try:
        started = time.perf_counter()
        answer = producer.latest()
        asking_ms = (time.perf_counter() - started) * 1000.0

        before_repeated = widget.repeated_frames
        widget.repaint()
        repeated = widget.repeated_frames - before_repeated
    finally:
        release.set()
        hog.join(timeout=10.0)

    assert answer is None, \
        "a slot that is being written answered with a frame anyway"
    assert asking_ms < REPEAT_CEILING_MS, \
        f"asking for the newest frame took {asking_ms:.1f} ms; it waited"
    assert repeated == 1, \
        "a paint that could not read the slot was not counted as a repeat"


def test_a_backdrop_shown_before_it_has_a_size_still_paints_its_first_frame(
        qtbot):
    """A screen is often built and shown before its layout has given it one.

    The shading thread cannot shade a zero-by-zero canvas, so there is nothing
    published when the size finally arrives, and the obvious implementation
    puts up a flat rectangle for a frame — a visible flash on every screen
    that is laid out after it is shown. Instead the paint shades that one
    frame itself, and only if the thread is not mid-pass, so the fallback can
    never become the wait it exists to avoid.
    """
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(0, 0)
    host.show()
    widget = AmbientWidget(host, background=DARK, seed=8)
    widget.setGeometry(0, 0, 0, 0)
    widget.show()
    qtbot.waitUntil(lambda: widget.shading_thread_alive(), timeout=10000)
    assert widget.frames_shaded() == 0, "something shaded a zero-size canvas"

    host.resize(800, 600)
    widget.setGeometry(0, 0, 800, 600)
    image = widget.grab().toImage()

    assert widget.frames_painted >= 1
    assert any(QColor(image.pixel(x, y)).lightness() > 8
               for x in range(0, 800, 37) for y in range(0, 600, 41)), \
        "the first frame after the layout arrived was a flat rectangle"


def test_a_burst_of_repaints_costs_a_blit_each_and_not_a_shading_pass(qtbot):
    """Exposures are not capped by the frame rate, so they must be cheap.

    The console sits on a translucent surface over the backdrop; measured on
    a real X server with the real stylesheet, printing a line costs 0.99
    ambient repaints *with the animation timer stopped*. A chatty run
    therefore asks this widget for far more frames than its own cap allows,
    and before this change every one of them was a full shading pass.

    Now the shading is on its own beat and a repaint is a blit, so the work
    is bounded by the frame rate whatever the console does.
    """
    widget = _shown(qtbot, theme="aurora")
    qtbot.waitUntil(lambda: widget.frames_shaded() > 0, timeout=10000)

    shaded = widget.frames_shaded()
    painted = widget.frames_painted
    started = time.perf_counter()
    for _ in range(30):
        widget.repaint()
    elapsed = time.perf_counter() - started

    assert widget.frames_painted == painted + 30
    # Whatever the shading thread managed in that window is its own business;
    # what matters is that it is bounded by the clock and not by the 30.
    allowed = int(elapsed * widget.fps()) + 2
    assert widget.frames_shaded() - shaded <= allowed, (
        f"{widget.frames_shaded() - shaded} shading passes for 30 repaints "
        f"in {elapsed * 1000:.0f} ms: repaints are shading again")


# ---------------------------------------------------------------------------
# The regression this change exists for
# ---------------------------------------------------------------------------

@pytest.mark.heavy
def test_a_python_worker_no_longer_shades_the_backdrop_on_the_gui_thread(
        qtbot):
    """The reported lag, as a number that fails if it comes back.

    A ``PipelineWorker``-shaped Python thread runs while a real backdrop is
    driven through a real event loop. What is asserted is the GUI thread's
    own per-frame cost, because that is the thing the change moved: 18.80 ms
    for ``cells`` at 1080p before, 1.49 ms after.

    The control in the same test is what makes the number mean anything. A
    machine that simply failed to produce any load would also report a cheap
    frame, so the test shades one frame *synchronously* under the same load
    and asserts it is dear. Cheap frame plus dear shade is the split working;
    cheap frame plus cheap shade is a test that measured nothing.
    """
    widget = _shown(qtbot, theme="cells", cls=_TimedBackdrop)
    qtbot.waitUntil(lambda: widget.frames_shaded() > 0, timeout=10000)

    stop = threading.Event()
    worker = threading.Thread(target=_python_worker, args=(stop,), daemon=True)
    worker.start()
    try:
        qtbot.wait(300)              # let the load settle
        widget.costs.clear()
        qtbot.wait(2000)
        costs = list(widget.costs)

        # The control: the same work, on the GUI thread, under the same load.
        engine = make_engine("cells", default_palette_for("cells"), DARK,
                             seed=4242)
        engine.shade(W, H)                     # allocate the buffer first
        shades = []
        for _ in range(21):
            started = time.perf_counter()
            engine.shade(W, H)
            shades.append((time.perf_counter() - started) * 1000.0)
        # The median, not the best: a lucky pass that fell in a gap between
        # the worker's turns with the interpreter lock measures the machine
        # without the load, which is the opposite of what this controls for.
        loaded_shade = statistics.median(shades)
    finally:
        stop.set()
        worker.join(timeout=10.0)

    assert len(costs) >= 20, f"only {len(costs)} frames; nothing was measured"
    median = statistics.median(costs)
    assert loaded_shade > 2.0 * median, (
        f"shading cost {loaded_shade:.2f} ms against a {median:.2f} ms "
        f"frame — the load never materialised, so this test proved nothing")
    assert median < LOADED_PAINT_CEILING_MS, (
        f"the GUI thread spent {median:.2f} ms a frame on the backdrop while "
        f"a Python worker ran; the shading is back on the GUI thread")


# ---------------------------------------------------------------------------
# What it costs when nobody is looking
# ---------------------------------------------------------------------------

def test_a_hidden_backdrop_has_no_shading_thread(qtbot):
    """Off screen is 0 %, and that is now a claim about a thread too.

    The existing suite asserts a hidden backdrop paints no frames. That is no
    longer the whole cost: an unjoined shading thread would keep a core warm
    behind a screen nobody is looking at, and would go on doing it for every
    module screen the user ever opened. So the thread is started and stopped
    with the timer, and this asserts both halves of that.
    """
    widget = _shown(qtbot)
    assert widget.is_running()
    assert widget.shading_thread_alive()
    producer = widget._producer_box[0]

    widget.hide()

    assert not widget.is_running()
    assert not widget.shading_thread_alive()
    assert widget._producer_box[0] is None
    assert not producer.is_alive(), "the shading thread outlived the timer"

    # Not merely detached: actually stopped, so it is not still shading into
    # a slot nobody reads.
    shaded = producer.frames_shaded
    qtbot.wait(300)
    assert producer.frames_shaded == shaded

    widget.show()
    qtbot.waitExposed(widget)
    assert widget.shading_thread_alive(), \
        "the backdrop came back on screen without its shading thread"


def test_a_backdrop_started_twice_still_has_exactly_one_shading_thread(qtbot):
    """Several gates start the animation and more than one of them fires.

    ``showEvent``, the window's own Show event and the Preferences toggle all
    reach ``start()`` on the same widget, sometimes in the same tick. Two
    threads shading one engine into one slot would race for the buffer, so
    both the widget and the producer refuse a second start — independently,
    because either one alone would be a guard nobody re-checks after an edit.
    """
    baseline = _shading_threads()
    widget = _shown(qtbot)
    assert _shading_threads() == baseline + 1
    producer = widget._producer_box[0]

    widget.start()
    widget.start()
    assert widget._producer_box[0] is producer
    assert _shading_threads() == baseline + 1

    producer.start()
    assert _shading_threads() == baseline + 1


def test_lowering_the_frame_cap_slows_the_shading_thread_too(qtbot):
    """A lower cap has to reduce the work, not just how much of it is shown.

    The cap is what somebody whose machine cannot afford the backdrop reaches
    for. If it throttled only the timer, the shading thread would go on
    shading twenty-four frames a second behind it and the setting would buy
    nothing at all — which is exactly the shape of bug this whole change could
    have introduced, since the thread has its own clock now.
    """
    widget = _shown(qtbot, theme="blobs")
    qtbot.waitUntil(lambda: widget.frames_shaded() > 0, timeout=10000)

    widget.set_fps(4)
    assert widget.fps() == 4
    qtbot.wait(300)                  # let the frame in flight finish
    before = widget.frames_shaded()
    qtbot.wait(1500)
    shaded = widget.frames_shaded() - before

    assert shaded <= 10, \
        f"{shaded} frames shaded in 1.5 s at a 4 fps cap; the cap is not"\
        f" reaching the shading thread"
    assert shaded >= 1, "the shading thread stopped altogether"


def test_an_unbuffered_theme_keeps_the_synchronous_path(qtbot):
    """``drift`` is excluded on purpose, and the exclusion is a behaviour.

    It is the one engine with no buffer, it degrades the least of the six
    under a Python worker (0.528 -> 1.084 ms, 2.1x, against 48.7x for
    ``cells``), and a producer for it would have to publish a full-resolution
    frame — 7.91 MiB a slot at 1080p against 126.6 KiB for ``blobs``. So it
    paints where it always painted, and still paints.
    """
    widget = _shown(qtbot, theme="drift")
    assert widget.is_running()
    assert not widget.shading_thread_alive()
    assert widget.frames_shaded() == 0

    painted = widget.frames_painted
    widget.repaint()
    assert widget.frames_painted == painted + 1
    image = widget.grab().toImage()
    assert any(QColor(image.pixel(x, y)).lightness() > 8
               for x in range(0, W, 37) for y in range(0, H, 41)), \
        "the starfield painted nothing"


def test_a_backdrop_that_was_never_shown_never_starts_a_thread(qtbot):
    """Every existing test drives frames by hand on a widget it never shows.

    They stay deterministic because the thread and the timer start together,
    so an unshown widget has neither — its frames come from ``advance_frame``
    and from nowhere else. If that ever stops being true, several hundred
    tests start racing a background thread for the same clock.
    """
    widget = AmbientWidget(background=DARK, seed=7)
    qtbot.addWidget(widget)
    widget.resize(640, 400)

    assert not widget.is_running()
    assert not widget.shading_thread_alive()

    widget.set_time(3.0)
    for _ in range(30):
        widget.advance_frame(1.0 / 60.0)
    assert widget.time() == pytest.approx(3.0 + 30 / 60.0)
    assert not widget.shading_thread_alive()


# ---------------------------------------------------------------------------
# Live settings, which is the one moment a running widget gets mutated
# ---------------------------------------------------------------------------

def test_a_setting_changed_while_it_runs_shows_up_in_the_very_next_frame(
        qtbot):
    """``apply_ambient_preferences`` mutates backdrops that are already
    running, and the next paint has to show the change.

    It walks ``app.allWidgets()`` and calls eight setters on every live
    backdrop, which is the one moment a widget is changed *while* a thread is
    shading it. A published frame is by definition older than the change, so
    every setter re-shades before it returns; without that, moving a slider
    would leave the old picture up until the thread happened to finish
    another one.
    """
    widget = _shown(qtbot, theme="blobs")
    qtbot.waitUntil(lambda: widget.frames_shaded() > 0, timeout=10000)

    before = widget.grab().toImage()
    widget.set_density(3.0)
    after = widget.grab().toImage()
    assert bytes(after.constBits()) != bytes(before.constBits()), \
        "density changed and the next frame was the old one"

    # The mode flip is the one that would be visible as a flash rather than
    # as one stale frame: dark composites additively, light multiplies.
    widget.set_background_color("#f6f7f9")
    light = widget.grab().toImage()
    assert not widget.engine.dark
    assert bytes(light.constBits()) != bytes(after.constBits())


def test_switching_theme_while_it_runs_retires_the_old_shading_thread(qtbot):
    """A theme switch replaces the engine, and the thread holds the old one.

    Flipping through the animation menu must not leave a thread behind per
    theme, each still shading an engine nobody paints — the same claim the
    widget already makes about timers and engines, now about threads.
    """
    widget = _shown(qtbot, theme="blobs")
    qtbot.waitUntil(lambda: widget.frames_shaded() > 0, timeout=10000)
    first = widget._producer_box[0]

    widget.set_theme("ripple")

    assert not first.is_alive(), "the old shading thread was left running"
    assert widget.shading_thread_alive()
    assert widget._producer_box[0] is not first
    assert widget.theme() == "ripple"

    qtbot.waitUntil(lambda: widget.frames_shaded() > 0, timeout=10000)
    widget.repaint()
    assert widget.frames_painted > 0
