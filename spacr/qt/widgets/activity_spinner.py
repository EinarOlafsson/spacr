"""The background-activity indicator: a braided ring that turns while spaCR
is busy, and does not exist on the CPU when it is not.

Why this is not the GIF
=======================
``spacr/resources/icons/loading_spinner.gif`` is the asset this widget was
asked to reuse, and reuse was rejected on three measurements, not on taste:

* **Size.** 800x600, 144 frames, 682 KB on disk. ``QMovie`` decodes a frame
  to a 32-bit ``QImage``, so one frame is 800 x 600 x 4 = 1.92 MB and the
  whole loop is 276 MB. ``QMovie.CacheAll`` really does hold all of it;
  ``CacheNone`` (the default) trades that for re-decoding 1.92 MB, and
  rescaling it to 16 px, 33 times a second, forever.
* **Transparency.** Every pixel of every frame has alpha 255 and the
  background is ``#000000``. The GIF is not a sprite, it is a *video of* a
  spinner. Dropped beside "Clear console" it paints an opaque black square,
  which is invisible in the dark theme and a hole in the light one.
* **Legibility.** The motif is a 1-2 px white stroke on an 800 px canvas.
  Scaled to 16 px that stroke is 0.02 px wide: it does not survive, with or
  without smooth transformation.

Pre-scaling the frames once into ``QPixmap``s fixes the per-frame cost but
none of the other two, and it still costs a 276 MB decode pass at start-up
for a 16 px dot. So the motif is redrawn instead: the GIF's *character* is a
thin ring with a double-helix braid travelling around it (see the frames --
the braid sweeps, the ring stays), and that is exactly what
:meth:`ActivitySpinner.paintEvent` draws with two ``QPolygonF``s. It costs
microseconds, it takes its colours from the live palette so it themes in
both light and dark, and it is sharp at any size.

Measured cost
=============
Reported the way :mod:`spacr.qt.widgets.dna_rain` reports its own
(0.53 ms a frame, 3.2 % of one core at 60 fps) -- see
``tests/qt/test_activity_spinner.py::test_spinner_frame_cost_is_negligible``
for the harness that produces the number, and the module-level docstring
there for the figures.

Idle cost is **zero**, not "small": :meth:`_sync` stops the ``QTimer``
outright when the registry goes quiet, so an idle spinner posts no timer
events, schedules no repaints and is hidden as well. There is no invisible
animation running behind an idle window.

What drives it
==============
:func:`spacr.qt.bridge.registry` -- the process-wide ``RunRegistry`` that
``make_thread`` adds every job to. Not an ad-hoc "busy" flag each caller has
to remember to set and, more importantly, remember to clear: the registry is
the same state every screen's ``active_jobs()`` is counting, and it is
maintained by ``make_thread`` itself, so a job that forgets to tell anyone it
started still turns the spinner on.

When it appears
===============
Not immediately. Most of what goes through ``make_thread`` -- reading a
measurement table, listing a plate, loading a settings file -- is finished
inside a second, and an indicator that appears and vanishes in that time is
not information. It is a flicker at the edge of vision, and it teaches the
reader to stop looking at the one place the app says it is busy.

So the widget waits :func:`spacr.qt.preferences.get_spinner_delay` seconds
(default 2) before showing. The mechanism is a **delay, not a prediction**:
:meth:`ActivitySpinner._sync` starts a single-shot timer the moment work
begins and the spinner becomes visible only if :meth:`ActivitySpinner.is_busy`
is *still* true when that timer fires. A job that finishes at 1.9 s cancels
the timer on its way out and never puts anything on screen -- there is no
estimate of duration anywhere in this file, and therefore nothing to be wrong
about.

The clock runs on the *work*, not on the job. A second job starting while
the timer is pending does not restart it: the timer is armed on the
idle-to-busy edge only, so two seconds of continuous background activity
shows the spinner even if no single job lasted that long. Going idle
disarms it, and the next burst of work starts a fresh two seconds.

Hiding is not delayed. The moment the registry goes quiet the widget hides
and its animation timer stops -- a spinner that lingered after the work
finished would be saying something untrue.
"""
from __future__ import annotations

import math
from typing import List, Optional

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QPushButton, QWidget

from ..theme import active_palette

__all__ = ["ActivitySpinner", "attach_activity_spinner"]

#: Degrees the braid advances per frame. 20 fps x 9 deg = one turn every
#: 2.0 s, close to the GIF's own 144 x 30 ms = 4.3 s but half as leisurely,
#: which reads better at 16 px where the whole ring is in one glance.
STEP_DEGREES = 9.0

#: Frame interval in ms. 20 fps is smooth for a 16 px rotation and costs a
#: third of what the GIF's 33 fps would.
INTERVAL_MS = 50

#: Arc the braid covers, in degrees.
BRAID_SPAN = 130.0

#: Points along the braid. 18 is where a 16 px strand stops looking faceted;
#: more points cost more and change nothing on screen.
BRAID_POINTS = 18

#: Twists of the helix across the span.
BRAID_TWISTS = 2.5


def _preferred_delay_ms() -> int:
    """The appearance delay from Preferences, in milliseconds.

    Defended like every other preference read from a widget: a spinner that
    cannot find the setting is a spinner on the shipped default, never a
    screen that refuses to open.
    """
    try:
        from ..preferences import get_spinner_delay
        return max(0, int(round(get_spinner_delay() * 1000)))
    except Exception:
        return 2000


class ActivitySpinner(QWidget):
    """A small braided ring, visible only while background work is running.

    It watches :func:`spacr.qt.bridge.registry` and needs no cooperation
    from the code that starts the work::

        spinner = ActivitySpinner(parent)      # hidden, no timer running

    :param parent: the usual Qt parent.
    :param diameter: side length in pixels. 16 is the size that sits level
        with a push button's text.
    :param auto: watch the run registry. ``False`` leaves the widget under
        manual :meth:`set_busy` control, which is what the tests use to
        drive it without spawning threads.
    :param delay_ms: how long work has to run before this appears. ``None``
        -- the default -- reads
        :func:`spacr.qt.preferences.get_spinner_delay` at construction, which
        is the only place that preference is consulted: the widget is built
        per screen, so a change to it reaches the next screen opened without
        this file having to watch a settings key.
    """

    def __init__(self, parent: Optional[QWidget] = None, diameter: int = 16,
                 auto: bool = True, delay_ms: Optional[int] = None) -> None:
        super().__init__(parent)
        self._diameter = max(8, int(diameter))
        self.setFixedSize(self._diameter, self._diameter)
        self.setObjectName("ActivitySpinner")
        # Nothing here reacts to the mouse, and a transparent-for-mouse
        # widget cannot swallow a click meant for the button beside it.
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._angle = 0.0
        self._manual_busy = False
        self._auto = bool(auto)
        self._timer = QTimer(self)
        self._timer.setInterval(INTERVAL_MS)
        self._timer.timeout.connect(self._advance)
        self._delay_ms = (_preferred_delay_ms() if delay_ms is None
                          else max(0, int(delay_ms)))
        #: Armed on the idle-to-busy edge, and only then. Not restarted by a
        #: second job, so the delay measures how long *work* has been going
        #: rather than how long the longest single job has.
        self._delay = QTimer(self)
        self._delay.setSingleShot(True)
        self._delay.timeout.connect(self._on_delay_elapsed)
        #: Has the current stretch of work earned the spinner yet? Kept as
        #: its own flag rather than read back off ``isVisible()``, because a
        #: widget can be visible for reasons that have nothing to do with
        #: this decision -- a caller that called ``show()`` on it, a layout
        #: that adopted it -- and reading visibility would let any of those
        #: quietly cancel the delay.
        self._due = False
        #: Frames painted since construction. Read by the CPU-cost test.
        self.frames_painted = 0
        self.hide()
        if self._auto:
            self._connect_registry()
        self._sync()

    # -- registry ---------------------------------------------------------

    def _connect_registry(self) -> None:
        """Follow the process-wide run registry.

        A BOUND METHOD is connected, never a closure: ``RunRegistry`` is a
        long-lived GUI-thread singleton, so a closure connected to it would
        keep this widget alive for the life of the process and would go on
        calling into a deleted C++ half after the screen closed. Binding to
        ``self`` lets Qt drop the connection with the widget.
        """
        try:
            from ..bridge import registry
            registry().changed.connect(self._sync)
        except Exception:
            # A spinner that cannot find the registry is a spinner that
            # never turns. It must never be a spinner that stops the
            # screen it lives on from opening.
            self._auto = False

    def _running_handles(self) -> List:
        if not self._auto:
            return []
        try:
            from ..bridge import registry
            return [h for h in registry().active() if h is not None]
        except Exception:
            return []

    # -- state ------------------------------------------------------------

    def is_busy(self) -> bool:
        """Whether the spinner considers spaCR to be working."""
        if self._manual_busy:
            return True
        return bool(self._running_handles())

    def is_spinning(self) -> bool:
        """Whether the animation timer is actually running.

        The assertion behind "idle costs zero": when this is ``False`` the
        widget posts no events at all.
        """
        return self._timer.isActive()

    def set_busy(self, busy: bool) -> None:
        """Force the spinner on or off, on top of whatever the registry says.

        For work that does not go through ``make_thread`` -- a bare
        ``QThread`` subclass, a ``QRunnable`` -- and for tests.
        """
        self._manual_busy = bool(busy)
        self._sync()

    def delay_ms(self) -> int:
        """How long work must run before this widget appears, in ms."""
        return self._delay_ms

    def set_delay_ms(self, value: int) -> None:
        """Change the appearance delay. Applies from the next idle-to-busy
        edge; it never yanks a spinner that is already up off the screen."""
        self._delay_ms = max(0, int(value))

    def is_waiting(self) -> bool:
        """True while work is running but the delay has not elapsed.

        The state that makes this a delay rather than a guess: busy, not
        shown, not spinning, costing nothing but one pending timer.
        """
        return self._delay.isActive()

    def _sync(self) -> None:
        """Match the widget to the current busy state. Idempotent.

        Three states, not two. *Idle*: hidden, both timers stopped. *Waiting*:
        work is running, the single-shot delay is armed, nothing on screen.
        *Showing*: the delay elapsed with work still running.

        Arming happens on the idle-to-busy edge only — ``isActive()`` is
        checked before ``start()``, and ``start()`` on a running QTimer
        restarts it. Without that guard a stream of short jobs would push the
        deadline forward for ever and the spinner would never appear during a
        genuinely long stretch of work.
        """
        if not self.is_busy():
            # Hiding is immediate and unconditional. Anything else would
            # leave the widget saying something that is not true.
            self._delay.stop()
            self._due = False
            if self._timer.isActive():
                self._timer.stop()
                self._angle = 0.0
            if self.isVisible():
                self.setVisible(False)
            self.setToolTip(self._describe())
            return
        if not self._due:
            if self._delay_ms <= 0:
                self._due = True
            elif not self._delay.isActive():
                self._delay.start(self._delay_ms)
        self.setToolTip(self._describe())
        if self._due:
            self._show_now()

    def _on_delay_elapsed(self) -> None:
        """The delay fired. Show only if there is still work to show.

        This is the whole of the "not a prediction" claim: the question is
        asked *after* the wait, about the present, so a job that finished at
        1.9 s is simply not busy here and nothing appears.
        """
        if self.is_busy():
            self._due = True
            self._show_now()

    def _show_now(self) -> None:
        self.setVisible(True)
        # ``isVisible`` is False while an ancestor is hidden, and the whole
        # idle-costs-zero claim rests on never running the animation timer
        # for pixels nobody can see. ``showEvent`` starts it if and when the
        # screen this lives on comes back.
        if self.isVisible() and not self._timer.isActive():
            self._timer.start()
        self.setToolTip(self._describe())
        self.update()

    def _describe(self) -> str:
        """A tooltip naming what is running, for the times it matters."""
        handles = self._running_handles()
        if not handles:
            return "" if not self._manual_busy else "Working in the background…"
        names = []
        for handle in handles:
            key = str(getattr(handle, "app_key", "") or "job")
            line = str(getattr(handle, "last_line", "") or "")
            names.append(f"{key} — {line}" if line else key)
        head = ("Running in the background:" if len(names) > 1
                else "Running in the background:")
        return head + "\n• " + "\n• ".join(names)

    # -- animation --------------------------------------------------------

    def _advance(self) -> None:
        self._angle = (self._angle + STEP_DEGREES) % 360.0
        self.update()

    def hideEvent(self, event):      # noqa: N802 - Qt override
        """Stop the timer whenever the widget leaves the screen.

        Hiding covers the cases the registry cannot see: the module was
        switched away from, the window was minimised, the screen closed.
        A timer left running there is exactly the invisible spin this
        widget exists to avoid.

        The pending appearance delay goes with it, for the same reason: a
        screen the user has left should not schedule itself back on.
        """
        self._timer.stop()
        self._delay.stop()
        super().hideEvent(event)

    def showEvent(self, event):      # noqa: N802 - Qt override
        """Resume only if there is still something to report — and only if
        the work had already earned the spinner before the screen went away.

        Without the :attr:`_due` half of that condition, coming back to a
        screen would restart the animation for work that started three
        milliseconds ago, which is precisely the flicker the delay exists to
        prevent.
        """
        super().showEvent(event)
        if self._due and self.is_busy() and not self._timer.isActive():
            self._timer.start()

    # -- painting ---------------------------------------------------------

    def paintEvent(self, event):     # noqa: N802 - Qt override
        """Draw the ring and the braid travelling around it.

        Two ``QPolygonF``s of :data:`BRAID_POINTS` points each plus one
        ellipse. No pixmap, no cache to invalidate on a theme change, and
        nothing to scale.
        """
        self.frames_painted += 1
        palette = active_palette()
        ring = QColor(palette.get("fg_dim", "#6b6f76"))
        ring.setAlpha(110)
        accent = QColor(palette.get("accent", "#4A9EFF"))

        size = min(self.width(), self.height())
        radius = size / 2.0 - 2.0
        centre = QPointF(self.width() / 2.0, self.height() / 2.0)
        stroke = max(1.0, size / 12.0)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(ring, stroke * 0.75))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, radius, radius)

        amplitude = max(1.0, size / 9.0)
        start = math.radians(self._angle)
        span = math.radians(BRAID_SPAN)
        strands = (QPolygonF(), QPolygonF())
        for index in range(BRAID_POINTS):
            t = index / float(BRAID_POINTS - 1)
            theta = start + t * span
            twist = t * BRAID_TWISTS * 2.0 * math.pi
            # The two strands are the same wave half a turn apart -- which
            # is what makes it read as a double helix rather than as a
            # wobbling line.
            for strand, phase in zip(strands, (0.0, math.pi)):
                r = radius + amplitude * math.sin(twist + phase)
                strand.append(QPointF(
                    centre.x() + r * math.cos(theta),
                    centre.y() + r * math.sin(theta)))

        pen = QPen(accent, stroke)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        for strand in strands:
            painter.drawPolyline(strand)
        painter.end()


def attach_activity_spinner(screen: QWidget) -> Optional[ActivitySpinner]:
    """Put an :class:`ActivitySpinner` immediately right of *Clear console*.

    Idempotent: calling it twice on the same screen returns the spinner that
    is already installed rather than adding a second one, so it is safe from
    a ``showEvent``.

    The button is found by attribute (``screen._btn_clear``) and only then by
    text, because the text is translated -- ``retranslate_widget_tree`` runs
    over every screen as it opens, and by the time a user in a non-English
    locale sees the row the string "Clear console" is not in the tree.

    :param screen: any widget in the tree that owns the button -- the
        ``AppScreen`` itself, or a descendant of it.
    :returns: the spinner, or ``None`` when this tree has no such button
        (Annotate, the Database Browser and every other non-``AppScreen``
        surface), which is not an error.
    """
    host = screen
    button = None
    while host is not None:
        candidate = getattr(host, "_btn_clear", None)
        if isinstance(candidate, QPushButton):
            button = candidate
            break
        existing = getattr(host, "_activity_spinner", None)
        if isinstance(existing, ActivitySpinner):
            return existing
        host = host.parentWidget()
    if button is None:
        return None

    existing = getattr(host, "_activity_spinner", None)
    if isinstance(existing, ActivitySpinner):
        try:
            existing.objectName()
        except RuntimeError:
            # Its C++ half is gone (the screen was rebuilt); fall through
            # and install a fresh one.
            pass
        else:
            return existing

    row = button.parentWidget()
    layout = row.layout() if row is not None else None
    if layout is None:
        return None
    index = layout.indexOf(button)
    if index < 0:
        return None
    spinner = ActivitySpinner(row)
    try:
        layout.insertWidget(index + 1, spinner)
    except (AttributeError, TypeError):
        # Not a box layout. Nothing sensible to insert into; the caller
        # gets None and the screen opens exactly as before.
        spinner.setParent(None)
        return None
    host._activity_spinner = spinner
    return spinner
