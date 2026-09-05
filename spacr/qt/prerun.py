"""What spaCR already knows about your data, on screen before you press Run.

Two things the pipeline computes, has always computed, and nobody has ever
seen — because both of them print to a terminal that has since scrolled, or
live in a module nothing calls:

* **The segmentation verdict.**  :mod:`spacr.seg_qc` scores every mask the
  moment it is written and files
  ``<plate>/qc/segmentation_qc_<object>.csv``.  Measure then spends hours
  cropping and measuring those masks without anyone having read it.  The
  banner this module puts on the Measure screen reads that card back and says
  what it says, naming the plate, the wells and the likely cause — a verdict
  a user can act on, not a count of failures.

* **The diameter.**  :mod:`spacr.diameter` measures characteristic object size
  from a handful of the user's own fields, without loading Cellpose or torch.
  ``diameter`` is the single most consequential Cellpose 4 setting spaCR
  exposes — ``CellposeModel.eval(diameter=...)`` rescales every image by
  ``30/diameter`` so objects land near the size ``cpsam`` works at — and it is
  the one users guess at.  The panel this module puts on the Mask screen turns
  the guess into a measurement, per object type, and shows how many objects it
  measured so the number can be disbelieved.

**Neither one blocks anything.**  The banner is advisory by construction: it
never touches the Run button, never disables it, never intercepts the click.
A plate that failed QC is still a plate its owner may have every reason to
measure, and a quality report that stops people is a quality report they
switch off.  :data:`BLOCKS_RUN` is False, ``tests/qt/test_prerun.py`` asserts
it against the real screen, and there is no code path here that could change
it.

Cost
----
The banner **reads** a verdict; it does not compute one.  Opening a plate's
masks costs seconds to minutes, and a screen that pays that on every visit is
a screen nobody keeps.  So :func:`spacr.seg_qc.read_digest` parses the CSVs
the mask run already wrote, dates each one against its mask stack, and reports
a card older than its masks as OUT OF DATE rather than believing it.  Only the
*Score the masks now* button scores anything, only when pressed, and it does
it on a worker thread and writes the card so the next open is cheap again.

Neither half of that runs on the GUI thread any more.  The *read* moved onto a
worker on 2026-09-04, when a single ``os.path.exists`` under a sleeping
``autofs`` mount was measured not returning for twenty seconds while
``install_qc_banner`` was doing it inside ``MainWindow._build_screen`` --
cheap is not the same as free, and only free may run where the frames are
painted.  See :meth:`SegQCBanner.refresh`.

Installation goes through the seams that already exist rather than through the
shared screen: :data:`spacr.qt.app.APP_FACTORIES`, consulted by
``MainWindow._build_screen``, and :func:`spacr.qt.theme.register_widget_qss`
for the colours.  ``AppScreen`` is untouched.  A factory already registered
for one of these keys — :mod:`spacr.qt.chaining` registers one for every
module that declares ports — is kept and delegated to, so installing this
never costs a screen the strip it already had, in either registration order.
"""
from __future__ import annotations

import html
import inspect
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

LOG = logging.getLogger("spacr.qt.prerun")

__all__ = [
    "BLOCKS_RUN",
    "DIAMETER_APP",
    "DIAMETER_OBJECT_NAME",
    "DiameterPanel",
    "QC_APP",
    "QC_OBJECT_NAME",
    "QSS_NAME",
    "SegQCBanner",
    "diameter_panel",
    "install_diameter_panel",
    "install_qc_banner",
    "qc_banner",
    "register",
    "unregister",
]

#: The module whose screen carries the segmentation-QC banner.
QC_APP = "measure"

#: The module whose screen carries the diameter estimator.
DIAMETER_APP = "mask"

#: Object names, for the stylesheet and for tests that look widgets up.
QC_OBJECT_NAME = "MeasureQCBanner"
DIAMETER_OBJECT_NAME = "DiameterPanel"

#: Key the shared stylesheet block is registered under. Both widgets are
#: styled by one block, so one registration covers them.
QSS_NAME = "SpacrPrerun"

#: Nothing in this module gates a run. Read by the tests, and by anyone who
#: wonders whether a failing verdict can stop them: it cannot.
BLOCKS_RUN = False

#: Debounce for anything triggered by typing a path.
REFRESH_DELAY_MS = 450

#: Findings shown before the user asks for the rest.
_FINDINGS_COLLAPSED = 2

#: Field-name links shown inline before one compact "browse all" link.  A
#: 1536-field positional finding must not turn the Measure screen into a
#: thousand-line hyperlink list; every field remains reachable with one click
#: and then Left/Right inside the browser.
_FIELD_LINKS_SHOWN = 8

#: Object types the diameter estimator offers, in report order.
_DIAMETER_OBJECTS: Tuple[str, ...] = ("cell", "nucleus", "pathogen")

#: Values a settings dict uses to mean "no source set yet".
_PLACEHOLDERS = frozenset({"", "path", "/path", "/path/to/src"})


# ---------------------------------------------------------------------------
# Styling
#
# The block is registered at IMPORT time, at the bottom of this section, and
# `spacr.qt.prerun` is listed in `theme.WIDGET_QSS_MODULES`. Both halves are
# required and neither is optional -- see INVARIANTS 1.
#
# It used to be registered only from `register()`, which runs after app.py
# has imported. The application stylesheet is built and applied before that,
# so `QFrame#MeasureQCBanner` was not in the sheet when the sheet was made:
# the panel fell through to the blanket `QWidget { background-color: bg }`
# and `bg` is #000000 on the dark theme. The verdict text sat on a solid
# black slab while every container around it was translucent -- which is
# exactly the symptom INVARIANTS 1 describes, arrived at by a different
# route.
#
# Measured on a fresh interpreter: 'MeasureQCBanner' in theme.stylesheet()
# was False at launch and True only after
# register_self_registering_modules().
# ---------------------------------------------------------------------------

def _qss(palette: Dict[str, Any], opacity: Any) -> str:
    """Both widgets' stylesheet for one palette.

    Registered through :func:`spacr.qt.theme.register_widget_qss`, so the
    colours follow the user's theme without a line in ``theme.py``.

    :param palette: the theme palette, surfaces already rendered through the
        page opacity.
    :param opacity: the user's page-opacity preference, passed through.
    """
    # Straight off the palette, which is what a REGISTERED block is handed:
    # `register_widget_qss` documents that `surface`, `surface_alt` and
    # `surface_hi` arrive already rendered through the user's page opacity,
    # so this is the value the built-in rules interpolate and the panel
    # matches the app by construction.
    #
    # It used to call `pane_surface("surface_alt", palette.get("theme"),
    # opacity)`. Two things were wrong with that and neither was visible
    # while this block was missing from the sheet:
    #
    #   * the palette carries no "theme" key, so that argument was always
    #     None, and `opacity` is None for a registered block -- so
    #     pane_surface fell through to reading the LIVE preference. A
    #     stylesheet that reads live preferences is the thing
    #     `test_the_sheet_does_not_read_the_live_page_opacity` exists to
    #     forbid;
    #   * it emitted rgba() on the opaque themes, which have no scrim, so
    #     the panel carried a translucency the theme never authorised
    #     (`test_opaque_themes_still_emit_plain_hex`).
    #
    # Both tests were green only because the block was not reaching the
    # sheet they inspect.
    surface = palette["surface_alt"]
    return f"""
    QFrame#{QC_OBJECT_NAME}, QFrame#{DIAMETER_OBJECT_NAME} {{
        background: {surface};
        border: 1px solid {palette['border_soft']};
        border-radius: 8px;
        padding: 8px;
    }}
    QLabel#PrerunTitle {{
        color: {palette['fg']};
        font-weight: 600;
        background: transparent;
    }}
    QLabel#PrerunHeadline {{
        color: {palette['fg']};
        background: transparent;
    }}
    QLabel#PrerunSub, QLabel#PrerunNote, QLabel#QCFieldLinks {{
        color: {palette['fg_muted']};
        background: transparent;
    }}
    QLabel#PrerunAdvisory {{
        color: {palette['fg_dim']};
        font-style: italic;
        background: transparent;
    }}
    QLabel#PrerunFail {{
        color: {palette['error']};
        background: transparent;
    }}
    QLabel#PrerunWarn {{
        color: {palette['warning']};
        background: transparent;
    }}
    QLabel#PrerunOk {{
        color: {palette['success']};
        background: transparent;
    }}
    QLabel#PrerunValue {{
        color: {palette['fg']};
        font-weight: 600;
        background: transparent;
    }}
    """


try:
    from .theme import register_widget_qss as _register_widget_qss
    _register_widget_qss(QSS_NAME, _qss, replace=True)
except Exception:
    # INVARIANTS 10: a stylesheet that cannot be registered costs this panel
    # its background, not the Measure module its run.
    LOG.exception("could not register the pre-run stylesheet at import")


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

class _ShowFilter(QObject):
    """Follow a watched widget's visible lifetime; consume nothing.

    A module screen is built once and kept, so ``__init__`` fires exactly
    once — while *returning* to the screen, which is when a mask run on
    another tab may have replaced everything this widget is about, fires
    ``Show``.
    """

    def __init__(self, on_show, parent=None, *, on_hide=None) -> None:
        """Call back when the watched widget is shown or hidden.

        :param on_show: called each time the widget is shown.
        :param parent: parent object.
        :param on_hide: called on hide and on close, or ``None`` to ignore
            them.

        NEITHER CALLBACK CAN SWALLOW THE EVENT. :meth:`eventFilter` forwards
        and returns False whatever they do, and an exception in one is logged
        rather than raised -- a refresh that fails must not stop the widget
        it was watching from appearing.
        """
        super().__init__(parent)
        self._on_show = on_show
        self._on_hide = on_hide

    def eventFilter(self, obj, event) -> bool:      # noqa: N802 - Qt override
        """Forward Show/Hide/Close events and never consume them."""
        if event.type() == QEvent.Show:
            try:
                self._on_show()
            except Exception:
                LOG.exception("pre-run refresh failed on show")
        elif event.type() in (QEvent.Hide, QEvent.Close):
            try:
                if self._on_hide is not None:
                    self._on_hide()
            except Exception:
                LOG.exception("pre-run cleanup failed on hide")
        return False


def _widgets(screen) -> Dict[str, QWidget]:
    """The screen's settings widgets, keyed by settings key."""
    model = getattr(screen, "_settings_model", None)
    return dict(getattr(model, "_widgets", {}) or {})


def _widget_value(widget) -> Any:
    """What a settings widget currently holds, or None.

    Prefers the ``get_value`` contract every list and scalar editor in
    :mod:`spacr.qt.screens.settings_model` implements, and falls back to
    ``text()`` for a plain ``QLineEdit``.
    """
    getter = getattr(widget, "get_value", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    text = getattr(widget, "text", None)
    if callable(text):
        try:
            return text()
        except Exception:
            return None
    return None


def _src_of(screen) -> Any:
    """The source the screen's ``src`` field currently names, or ``''``.

    List-valued sources (several plates in one run) are passed through
    whole: :func:`spacr.seg_qc.qc_roots` and
    :func:`spacr.diameter.estimate_diameters` both take a list.
    """
    value = _widget_value(_widgets(screen).get("src"))
    if isinstance(value, (list, tuple)):
        kept = [str(v).strip() for v in value
                if str(v).strip() not in _PLACEHOLDERS]
        return kept
    text = "" if value is None else str(value).strip()
    return "" if text in _PLACEHOLDERS else text


def _has_src(src: Any) -> bool:
    """True when ``src`` names anything at all."""
    return bool(src) if not isinstance(src, (list, tuple)) else bool(list(src))


def _label(text: str, name: str, *, wrap: bool = True) -> QLabel:
    """A themed, word-wrapped, selectable label.

    Word-wrapped labels need ``(Preferred, Minimum)``: with Qt's default
    ``Preferred`` height a parent is free to hand the label less than its
    heightForWidth, and the last line of a long fix is silently clipped —
    which is the one line that says what to do about the problem.
    """
    lbl = QLabel(text)
    lbl.setObjectName(name)
    lbl.setWordWrap(wrap)
    if wrap:
        lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lbl


def _transparent(widget):
    """Stop ``widget`` painting a background, and return it.

    Wrapped rather than calling ``theme.make_transparent`` at each site so a
    theme that cannot be imported costs these panels their transparency and
    nothing else (INVARIANTS 10) -- and so the reason is written down once.

    :param widget: any QWidget used purely as a layout container.
    :returns: the same widget, for use inline.
    """
    try:
        from .theme import make_transparent
        make_transparent(widget)
    except Exception:        # pragma: no cover - decoration is not load-bearing
        LOG.debug("could not make %r transparent", widget, exc_info=True)
    return widget


def _first_sentence(text: str, limit: int = 170) -> str:
    """The actionable head of a long fix, for the collapsed view.

    The full text is always one click away — and always in the copied report
    — so nothing is lost; what is gained is a banner that is a banner rather
    than a page.
    """
    body = str(text or "").strip()
    if len(body) <= limit:
        return body
    cut = body.find(". ")
    if 0 < cut <= limit:
        return body[:cut + 1] + " …"
    return body[:limit].rsplit(" ", 1)[0] + " …"


_SEVERITY_NAME = {"fail": "PrerunFail", "warn": "PrerunWarn", "ok": "PrerunOk"}


class _JobMixin:
    """One background job at a time, owned properly.

    PySide6 will not keep a worker alive through the ``started -> run``
    connection alone, and a ``QThread`` garbage-collected while still running
    takes the process down with it, so both are held until the thread's own
    ``finished``. Retirement hangs off a **bound method** rather than a
    lambda: ``QThread.finished`` crosses a thread boundary, and a closure
    would run with the emitting thread's affinity — which is exactly how a
    handler that touches widgets ends up off the GUI thread.

    ONE SLOT, AND MORE THAN ONE KIND OF WORK WANTS IT. Since the banner's
    read moved onto a worker, its housekeeping and its *Score the masks now*
    button compete for this single slot, and ``busy`` refuses the loser.
    Refusing is only safe where something remembers the refusal:
    :meth:`SegQCBanner._pending_work` is that something. A host that starts
    work from more than one place and does not keep such a record has turned
    a button into a silent no-op.
    """

    def _init_jobs(self) -> None:
        """Start with no jobs and not busy.

        Called from the host's own constructor rather than by inheritance, so a
        screen that forgets it has no job list at all -- which is why every
        reader below uses ``getattr`` with a default.
        """
        self._jobs: List[Tuple[Any, Any]] = []
        self._busy = False

    @property
    def busy(self) -> bool:
        """True while a background job is in flight."""
        return bool(getattr(self, "_busy", False))

    def _start_job(self, fn, box: Dict[str, Any], on_done, app_key: str, *,
                   user_visible: bool = True,
                   capture_figures: bool = True) -> bool:
        """Run ``fn(box)`` on a worker thread; call ``on_done(box)`` after.

        :param user_visible: False for work the user did not ask for. Such a
            job still turns the activity spinner but never claims a run
            banner on Home -- a banner reading "measure - running" every time
            somebody opens the Measure screen is a lie about what is running.
        :param capture_figures: False for a read that cannot emit a figure.
            It is not an optimisation for its own sake: ``make_thread``
            imports ``matplotlib.pyplot`` on the CALLING thread before the
            first capturing job, and the caller here is the GUI thread.
        """
        if self.busy:
            return False
        try:
            from .bridge import make_thread
        except Exception:
            LOG.exception("no worker thread available")
            return False
        try:
            # journal=False: this is read-only UI housekeeping, not an
            # analysis run, and a reproducibility manifest per button press
            # would bury the runs that are.
            thread, worker = make_thread(fn, box, app_key=app_key,
                                         journal=False,
                                         user_visible=user_visible,
                                         capture_figures=capture_figures)
        except Exception:
            LOG.exception("could not build the worker thread")
            return False
        self._jobs.append((thread, worker))
        self._on_done = on_done
        self._box = box
        worker.finished.connect(self._job_settled)
        thread.finished.connect(self._retire_finished_job)
        self._busy = True
        thread.start()
        return True

    def _job_settled(self, ok: bool) -> None:
        """Always on the GUI thread — see the class docstring."""
        self._busy = False
        done = getattr(self, "_on_done", None)
        box = getattr(self, "_box", {}) or {}
        self._on_done = None
        if callable(done):
            try:
                done(box if ok else {"error": box.get("error") or "failed"})
            except Exception:
                LOG.exception("pre-run job completion failed")

    def _retire_finished_job(self) -> None:
        """Release this job's references once its own event loop has exited."""
        thread = self.sender()
        self._jobs = [(t, w) for (t, w) in self._jobs if t is not thread]


# ---------------------------------------------------------------------------
# The segmentation-QC banner
# ---------------------------------------------------------------------------

class SegQCBanner(_JobMixin, QFrame):
    """The segmentation verdict, on the Measure screen, before Measure runs.

    Reads ``<plate>/qc/segmentation_qc_<object>.csv`` — the card
    :mod:`spacr.seg_qc` wrote when the masks were made — and shows what it
    says: the plate, the wells, the likely cause and what to do. It scores
    nothing on its own; the *Score the masks now* button is the only path in
    this class that opens a mask, and it exists because "no card" and "a card
    older than the masks" are both answers a user should be able to fix from
    here.

    It has no opinion about whether Measure should run. See
    :data:`BLOCKS_RUN`.

    :param screen: the ``AppScreen`` it belongs to.
    :param reader: what to call to read a digest, for tests. Defaults to
        :func:`spacr.seg_qc.read_digest`.
    :param threaded: ``False`` runs the read inline instead of on a worker,
        emitting the same signals in the same order, so a test can drive the
        banner synchronously without the behaviour diverging. The interface
        never builds one that way -- see :meth:`refresh`.
    :param parent: parent widget; ownership only.
    """

    #: Emitted after every refresh, with the verdict. Tests wait on it; the
    #: screen ignores it.
    refreshed = Signal(str)

    def __init__(self, screen: QWidget, *, reader=None, threaded: bool = True,
                 parent=None) -> None:
        """Build the segmentation-QC banner for one screen.

        :param screen: the screen this banner reports on.
        :param reader: how to read the QC scores.
        :param threaded: whether the read runs on a worker.
        :param parent: parent widget.
        """
        super().__init__(parent or screen)
        self.setObjectName(QC_OBJECT_NAME)
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self._init_jobs()

        self._screen = screen
        self._reader = reader
        self._threaded = bool(threaded)
        self._digest = None
        self._expanded = False
        self._cache_key: Optional[Tuple] = None
        self._field_browser = None
        self._field_targets: Tuple[Any, ...] = ()
        #: Bumped by every request. The read in flight carries the value it
        #: was issued under in ``_reading_gen``, so an answer to a question
        #: nobody is asking any more -- the src field cleared, or retyped --
        #: is dropped instead of painted over the new source.
        self._refresh_gen = 0
        self._reading_gen = -1
        #: True while the READ, rather than the scoring pass, holds the one
        #: job slot. The two are not interchangeable: a click that arrives
        #: during a read is worth waiting for, a click during a scoring pass
        #: is the pass already running.
        self._reading = False
        #: Exactly one catch-up each, in the shape of
        #: ``spacr.qt.chaining.ChainingBar._refresh``'s ``_resolve_again`` --
        #: see :meth:`_pending_work`.
        self._refresh_again = False
        self._score_again = False

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        self._title = _label("Segmentation QC", "PrerunTitle", wrap=False)
        head.addWidget(self._title)
        head.addStretch(1)
        self._btn_more = QPushButton("Show all findings")
        self._btn_more.setObjectName("GhostButton")
        self._btn_more.setCursor(Qt.PointingHandCursor)
        self._btn_more.clicked.connect(self._on_toggle_findings)
        self._btn_more.hide()
        head.addWidget(self._btn_more)
        column.addLayout(head)

        self._headline = _label("", "PrerunHeadline")
        column.addWidget(self._headline)
        self._sub = _label("", "PrerunSub")
        column.addWidget(self._sub)

        # Scaffolding, so it must paint nothing (INVARIANTS 3). A plain
        # QWidget used as a layout container inherits the blanket
        # `QWidget { background-color: bg }` rule, and `bg` is the WINDOW
        # colour -- #000000 on the dark theme. The findings text sat on a
        # solid black rectangle inside a panel that was otherwise a
        # translucent surface, which is exactly what it looked like: a black
        # box behind the text.
        #
        # The panel's own background already follows the page opacity
        # (`pane_surface` in `_qss`); this is what lets it show through.
        self._findings_box = _transparent(QWidget(self))
        self._findings_layout = QVBoxLayout(self._findings_box)
        self._findings_layout.setContentsMargins(0, 2, 0, 2)
        self._findings_layout.setSpacing(6)
        column.addWidget(self._findings_box)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(8)
        self._btn_score = QPushButton("Score the masks now")
        self._btn_score.setObjectName("GhostButton")
        self._btn_score.setCursor(Qt.PointingHandCursor)
        self._btn_score.clicked.connect(self._on_score_clicked)
        actions.addWidget(self._btn_score)
        self._btn_copy = QPushButton("Copy report")
        self._btn_copy.setObjectName("GhostButton")
        self._btn_copy.setCursor(Qt.PointingHandCursor)
        self._btn_copy.clicked.connect(self._on_copy_clicked)
        actions.addWidget(self._btn_copy)
        actions.addStretch(1)
        column.addLayout(actions)

        self._advisory = _label(
            "Advisory only — this never stops Measure from running.",
            "PrerunAdvisory")
        column.addWidget(self._advisory)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REFRESH_DELAY_MS)
        self._timer.timeout.connect(self.refresh)

        self._show_filter = _ShowFilter(
            self._on_screen_shown, self,
            on_hide=self._close_field_browser)
        try:
            screen.installEventFilter(self._show_filter)
        except Exception:
            LOG.exception("could not watch the Measure screen for show events")
        # HIDDEN UNTIL THERE IS SOMETHING TO SAY. `install_qc_banner` now
        # SCHEDULES the first read instead of doing it inline -- that is the
        # freeze fix -- so for the 450 ms of the debounce the banner would
        # otherwise sit in the layout visible and empty: a title, two buttons
        # and no verdict, on every Measure screen build, appearing and
        # vanishing. Measured against HEAD: with no src, HEAD had it hidden
        # from the start and the working tree showed it for 1.2 s.
        #
        # Every path that has something to draw calls `show()` itself, so
        # this only removes the flash.
        self.hide()
        self._wire_src()

    # -- wiring -----------------------------------------------------------

    def _wire_src(self) -> None:
        """Re-read the verdict a beat after the source folder changes."""
        widget = _widgets(self._screen).get("src")
        signal = getattr(widget, "textChanged", None) or getattr(
            widget, "valueChanged", None)
        if signal is not None:
            try:
                signal.connect(lambda *_a: self._timer.start())
            except Exception:
                LOG.exception("could not follow the src field")

    def _on_screen_shown(self) -> None:
        self._timer.start()

    def schedule_refresh(self) -> None:
        """Ask for a refresh a beat from now, on the same debounce as typing.

        What screen construction calls instead of :meth:`refresh`. Even the
        asynchronous refresh builds a ``QThread``, and there is no reason to
        pay for one inside ``MainWindow._build_screen``: the banner is
        advisory, and 450 ms later is soon enough for advice.
        """
        self._timer.start()

    # -- reading ----------------------------------------------------------

    def _read(self, src: Any):
        """Read the digest for ``src``. Injected in tests."""
        reader = self._reader
        if reader is None:
            from ..seg_qc import read_digest
            reader = read_digest
        return reader(src)

    def _fingerprint(self, src: Any) -> Optional[Tuple]:
        """A cheap key that changes exactly when the cards do.

        WORKER THREAD ONLY. It used to be described here as "one listdir plus
        one stat per card — microseconds", and that was a local-disk
        assumption: ``find_scorecards`` first calls ``qc_roots``, which
        ``isdir``s the user's root, lists its plate children and ``isdir``s a
        ``qc`` folder inside each, all before the ``os.stat`` below. On a
        sleeping ``autofs`` mount the first of those had not returned after
        twenty seconds. It is still cheap — it is just not free, and nothing
        that is not free may run on the GUI thread.

        It exists so that returning to the screen ten times does not re-parse
        ten times, while a re-mask that rewrites a card is still picked up on
        the next visit. Returns None when the fingerprint cannot be taken,
        which forces a read rather than trusting a stale cache.
        """
        try:
            from ..seg_qc import find_scorecards
            paths = find_scorecards(src)
        except Exception:
            return None
        out = []
        for path in paths:
            try:
                info = os.stat(path)
            except OSError:
                return None
            out.append((path, info.st_mtime_ns, info.st_size))
        return tuple(out)

    def refresh(self) -> None:
        """Ask for the verdict and redraw when it lands. Never scores a mask.

        SPLIT IN TWO, and the split is the fix for a frozen application.
        What stays here is widget state — the ``src`` field, and whether it
        names anything — and it is free. What moved into :meth:`_refresh_job`
        is every filesystem call: ``find_scorecards`` walking the user's
        plate folders, one ``os.stat`` per card, and ``read_digest`` opening
        and parsing the CSVs. All of it is I/O on a path the USER supplied.

        Measured on the maintainer's machine 2026-09-04: ``os.path.exists``
        on a path under ``/nas_mnt`` — an ``autofs`` mount whose share was
        asleep — had not returned after TWENTY SECONDS, because the stat is
        what triggers the automount. This method used to do that work inline,
        and ``install_qc_banner`` used to call it inside
        ``MainWindow._build_screen``, so it was the whole interface frozen on
        every Measure open. It left no traceback, because a stalled event
        loop is not a crash; it was reported as "opening map barcodes
        crashes spacr", plus hover flicker and glimpses of other screens.

        The banner keeps whatever it last drew while a read is in flight.
        There is deliberately no :mod:`spacr.qt.path_probe` gate in front of
        the job: the probe answers ``isdir`` optimistically-cheap but the
        scan has to happen off the GUI thread regardless, so a gate would
        only add a first-visit blank for no protection this does not give.

        EVERY CALL BUMPS ``_refresh_gen``, the one that finds the field empty
        included. That is what makes a read cancellable without being
        interruptible: nothing can stop the worker mid-``stat``, but its
        answer is checked against the question before it is painted and
        dropped when the source has moved on. Without it, clearing src hid
        the banner and the read still in flight put the old plate's verdict
        straight back on screen under no name at all.
        """
        src = _src_of(self._screen)
        self._refresh_gen += 1
        gen = self._refresh_gen
        if not _has_src(src):
            self._digest = None
            self._cache_key = None
            self.hide()
            self.refreshed.emit("")
            return
        # Both cache fields are read HERE and compared on the worker. The job
        # body may not read the banner's state: by the time it runs, the GUI
        # thread may have changed it.
        box: Dict[str, Any] = {
            "src": src,
            "cache_key": self._cache_key,
            "cached": self._digest is not None,
        }
        if not self._threaded:
            self._reading_gen = gen
            try:
                self._refresh_job(box)
            except Exception:
                LOG.exception("could not read the segmentation verdict")
                box = {}
            self._on_refreshed(box)
            return
        if self.busy:
            # COALESCED -- neither dropped nor queued. Twenty keystrokes are
            # twenty requests for one answer: a job each would ask the same
            # question twenty times, and dropping them loses the last one,
            # which is the only one that matters. `_pending_work` runs
            # exactly one catch-up when the slot frees. This is
            # `ChainingBar._refresh`'s `_resolve_again`, and it replaces a
            # re-armed debounce that re-asked every 450 ms for as long as a
            # sleeping mount took to answer.
            self._refresh_again = True
            return
        self._reading = True
        self._reading_gen = gen
        # user_visible=False: nobody asked for this, and a job that claims a
        # run banner would put "measure - running" on Home for a CSV read.
        if not self._start_job(self._refresh_job, box, self._on_refreshed,
                               QC_APP, user_visible=False,
                               capture_figures=False):
            self._reading = False
            LOG.debug("no worker available for the segmentation-QC refresh")

    def _refresh_job(self, box: Dict[str, Any]) -> None:
        """The filesystem half of a refresh. OFF THE GUI THREAD.

        Writes nothing but ``box``, and reads nothing but ``box``, the disk
        and ``self._reader`` -- which is set once in ``__init__`` and never
        again. Everything that can change while this runs was copied into
        ``box`` on the GUI thread before it started, because by the time it
        runs the GUI thread may have changed any of it.
        """
        src = box["src"]
        key = (repr(src), self._fingerprint(src))
        box["key"] = key
        if not (box["cached"] and key == box["cache_key"] and key[1]):
            box["digest"] = self._read(src)

    def _on_refreshed(self, box: Dict[str, Any]) -> None:
        """Draw what the read found. Always on the GUI thread.

        Runs on failure as well as on success -- ``_JobMixin._job_settled``
        calls it either way, with ``{"error": ...}`` when the worker raised
        -- so nothing this leaves behind can outlive a read that went wrong.
        """
        self._reading = False
        try:
            if self._reading_gen != self._refresh_gen:
                # THE ANSWER TO A QUESTION NOBODY IS ASKING. The src field
                # has moved on since this read was issued, so painting it
                # would put the previous source's verdict on screen under
                # the current source's name -- and after a CLEARED field
                # would un-hide a banner `refresh` had just hidden. The
                # catch-up below asks what is actually outstanding.
                #
                # `_job_settled` is not generation-guarded and cannot be: it
                # is shared with the diameter panel. The guard belongs here,
                # where what was asked is known.
                return
            key = box.get("key")
            if key is None:
                if box.get("error"):
                    LOG.error("could not read the segmentation verdict: %s",
                              box["error"])
                self._digest = None
                self.hide()
                self.refreshed.emit("")
                return
            if "digest" not in box:
                # The cards on disk are the ones already parsed. This is the
                # whole point of the fingerprint: ten visits, one parse.
                digest = self._digest
                if digest is None:
                    self.refreshed.emit("")
                    return
                self.show()
                self.refreshed.emit(digest.verdict)
                return
            self._digest = box["digest"]
            self._cache_key = key
            self._draw()
            self.show()
            self.refreshed.emit(self._digest.verdict)
        finally:
            self._pending_work()

    def _pending_work(self) -> None:
        """Run whatever was asked for while the last job held the slot.

        Exactly one catch-up each, which is the shape
        ``spacr.qt.chaining.ChainingBar._refresh`` uses: dropping a request
        loses it, and starting a job per request asks one question hundreds
        of times.

        A QUEUED CLICK GOES FIRST. It is what a person is sitting there
        waiting for, and scoring rewrites the very cards a read would have
        parsed -- so the re-read that follows the scoring pass is the one
        worth doing, and it is not lost: ``_refresh_again`` stays set and
        :meth:`_on_scored` drains it. A click that turns out to have nothing
        to score falls through to the re-read here instead of swallowing it.
        """
        if self._score_again:
            self._score_again = False
            self._on_score_clicked()
            if self.busy:
                return
        if self._refresh_again:
            self._refresh_again = False
            self.refresh()

    # -- drawing ----------------------------------------------------------

    @property
    def digest(self):
        """The last digest read, or None."""
        return self._digest

    def _draw(self) -> None:
        digest = self._digest
        if digest is None:
            return
        # A refreshed scorecard can point at a different set of fields.  Do
        # not leave an already-open browser navigating the previous digest.
        self._close_field_browser()
        verdict = digest.verdict
        title = {
            "ok": "Segmentation QC — passed",
            "warn": "Segmentation QC — look at this first",
            "fail": "Segmentation QC — failed",
            "missing": "Segmentation QC — not run",
            "error": "Segmentation QC — unreadable",
        }.get(verdict, "Segmentation QC")
        if digest.stale:
            title += " (out of date)"
        self._title.setText(title)
        self._title.setObjectName(_SEVERITY_NAME.get(verdict, "PrerunTitle"))
        self._restyle(self._title)

        self._headline.setText(digest.headline)
        sub = digest.subhead
        if digest.stale:
            names = ", ".join(
                card.object_type for card in digest.scorecards if card.stale)
            sub += (
                f" These masks have been written again since the {names} card "
                f"was scored, so what follows describes the previous masks. "
                f"Score them again to be sure."
            )
        self._sub.setText(sub)
        self._sub.setVisible(bool(sub))

        self._draw_findings(digest)
        self._btn_score.setText(
            "Score the masks now"
            if verdict == "missing" or digest.stale else "Score again")
        self._btn_copy.setEnabled(verdict not in ("missing",))

    def _restyle(self, widget: QWidget) -> None:
        """Make a changed objectName take effect without a full re-polish."""
        style = widget.style()
        if style is not None:
            style.unpolish(widget)
            style.polish(widget)

    def _clear_findings(self) -> None:
        while self._findings_layout.count():
            item = self._findings_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _draw_findings(self, digest) -> None:
        self._clear_findings()
        findings = list(digest.findings)
        try:
            from .widgets.qc_field_browser import targets_from_digest

            self._field_targets = targets_from_digest(digest)
        except Exception:
            LOG.exception("could not resolve segmentation-QC browser targets")
            self._field_targets = ()
        if not findings:
            self._btn_more.hide()
            self._findings_box.setVisible(False)
            return
        self._findings_box.setVisible(True)
        shown = findings if self._expanded else findings[:_FINDINGS_COLLAPSED]
        for finding in shown:
            # One per finding, and each is its own anonymous QWidget, so
            # each needs tagging: making only the parent transparent left a
            # black rectangle behind every finding's text -- which is what
            # the first attempt at this fixed and what the user still saw.
            block = _transparent(QWidget(self._findings_box))
            layout = QVBoxLayout(block)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            head = _label(
                f"• {finding.headline}",
                _SEVERITY_NAME.get(finding.severity, "PrerunHeadline"))
            layout.addWidget(head)
            self._add_field_links(layout, finding)
            if self._expanded and finding.detail:
                layout.addWidget(_label(finding.detail, "PrerunSub"))
            if finding.fix:
                fix = (finding.fix if self._expanded
                       else _first_sentence(finding.fix))
                layout.addWidget(_label(f"→ {fix}", "PrerunNote"))
            self._findings_layout.addWidget(block)
        extra = len(findings) - len(shown)
        self._btn_more.setVisible(bool(extra) or self._expanded)
        self._btn_more.setText(
            "Show less" if self._expanded else f"Show all {len(findings)} findings")

    def _add_field_links(self, layout: QVBoxLayout, finding: Any) -> None:
        """Render field stems as links into the plate-aware browser."""
        digest = self._digest
        if digest is None:
            return
        try:
            from .widgets.qc_field_browser import finding_targets

            targets = finding_targets(
                digest, finding, getattr(self, "_field_targets", ()))
        except Exception:
            LOG.exception("could not resolve segmentation-QC field links")
            return
        if not targets:
            return
        from .i18n import tr

        link_targets: Dict[str, Any] = {}
        anchors: List[str] = []
        for index, target in enumerate(targets[:_FIELD_LINKS_SHOWN]):
            href = str(index)
            link_targets[href] = target
            anchors.append(
                f'<a href="{href}">{html.escape(target.field)}</a>')
        if len(targets) > _FIELD_LINKS_SHOWN:
            href = "all"
            link_targets[href] = targets[0]
            browse_all = html.escape(tr(
                "Browse all {count} implicated fields…", count=len(targets)))
            anchors.append(
                f'<a href="{href}">{browse_all}</a>')
        label = QLabel(html.escape(tr("Inspect fields:")) + " "
                       + " · ".join(anchors),
                       self._findings_box)
        label.setObjectName("QCFieldLinks")
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        label.setOpenExternalLinks(False)
        label.setToolTip(tr(
            "Open the merged image, mask overlays, and this field's QC flags."))
        label.linkActivated.connect(
            lambda href, mapping=link_targets: self._on_field_link(
                mapping.get(str(href))))
        layout.addWidget(label)

    def _measure_run_active(self) -> bool:
        """Whether this screen's pipeline worker is currently in flight."""
        thread = getattr(self._screen, "_thread", None)
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except (AttributeError, RuntimeError):
            return False

    def _on_field_link(self, target: Any) -> None:
        """Open the browser at exactly the field whose link was activated."""
        if target is None or self._digest is None:
            return
        browser = self._field_browser
        if browser is not None:
            try:
                if browser.open_at(target.field, target.plate_root):
                    browser.show()
                    browser.raise_()
                    browser.activateWindow()
                    return
            except RuntimeError:
                self._field_browser = None
        try:
            from .widgets.qc_field_browser import (
                QCFieldBrowser,
                targets_from_digest,
            )

            factory = getattr(self, "_field_browser_factory", None)
            if factory is None:
                factory = QCFieldBrowser
            targets = getattr(self, "_field_targets", ())
            if not targets:
                targets = targets_from_digest(self._digest)
            browser = factory(
                targets,
                initial_field=target.field,
                initial_plate_root=target.plate_root,
                run_active=self._measure_run_active,
                parent=self,
            )
            browser.destroyed.connect(self._on_field_browser_destroyed)
            self._field_browser = browser
            browser.show()
            browser.raise_()
            browser.activateWindow()
        except Exception:
            LOG.exception("could not open the segmentation-QC field browser")

    def _on_field_browser_destroyed(self, *_args) -> None:
        self._field_browser = None

    def _close_field_browser(self) -> None:
        browser = getattr(self, "_field_browser", None)
        if browser is None:
            return
        self._field_browser = None
        try:
            browser.close()
        except RuntimeError:
            pass

    def _on_toggle_findings(self) -> None:
        self._expanded = not self._expanded
        if self._digest is not None:
            self._draw_findings(self._digest)

    # -- the one expensive path, and only on request -----------------------

    def _on_copy_clicked(self) -> None:
        """Put the whole report on the clipboard, for a lab notebook or an issue."""
        if self._digest is None:
            return
        try:
            from ..seg_qc import format_digest
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(format_digest(self._digest))
        except Exception:
            LOG.exception("could not copy the segmentation report")

    def _on_score_clicked(self) -> None:
        """Score the masks under ``src`` on a worker thread, then redraw.

        The only place in this class that opens a mask. It writes the cards
        it produces, so the next time this screen is opened the cheap path
        finds a fresh card and this button is not needed again.

        THE CLICK IS NEVER SWALLOWED. Until the read moved onto a worker,
        ``busy`` here could only mean "a scoring pass is already running",
        and refusing was the whole answer. Now the advisory read holds the
        same single slot -- on every screen open, on every return to the
        screen, and 450 ms after every keystroke in src -- so a plain
        refusal turned this button into a silent no-op for exactly as long
        as the filesystem took to answer, which on the mount that started
        all this was twenty seconds. The click is remembered instead, and
        :meth:`_pending_work` runs it the instant the read lets go.

        Waiting is not a compromise: the scoring pass would have had to walk
        the same sleeping mount, so nothing arrives sooner, and serialising
        the two keeps a write off the cards the read is parsing.
        """
        src = _src_of(self._screen)
        if not _has_src(src):
            # Includes a queued click whose source has since been cleared:
            # put the button back rather than leave it disabled forever.
            self._score_again = False
            self._btn_score.setEnabled(True)
            return
        if self.busy:
            if not self._reading:
                return          # a scoring pass is already running
            self._score_again = True
            self._btn_score.setEnabled(False)
            # The caption the pass itself uses. Scoring is what happens next
            # and it needs no further input, so a second wording for the
            # same state would only be a second thing to read.
            self._title.setText("Segmentation QC — scoring the masks…")
            return
        self._score_again = False
        settings: Dict[str, Any] = {}
        model = getattr(self._screen, "_settings_model", None)
        if model is not None:
            try:
                settings = dict(model.collect() or {})
            except Exception:
                settings = {}

        def _job(box: Dict[str, Any]) -> None:
            """Score the segmentation QC. Off the GUI thread."""
            from ..seg_qc import score_digest, thresholds_from_settings
            box["digest"] = score_digest(
                box["src"], thresholds=thresholds_from_settings(box["settings"]))

        box = {"src": src, "settings": settings}
        self._btn_score.setEnabled(False)
        self._title.setText("Segmentation QC — scoring the masks…")
        if not self._start_job(_job, box, self._on_scored, "seg_qc"):
            self._btn_score.setEnabled(True)
            self._draw()

    def _on_scored(self, box: Dict[str, Any]) -> None:
        """Show what the scoring pass found. Always on the GUI thread.

        Runs on failure too -- ``_JobMixin._job_settled`` calls it with
        ``{"error": ...}`` -- so the button comes back and the "scoring the
        masks…" caption is replaced whatever happened.
        """
        self._btn_score.setEnabled(True)
        try:
            digest = box.get("digest")
            if digest is None:
                self._title.setText(
                    "Segmentation QC — could not score these masks")
                return
            self._digest = digest
            self._cache_key = None      # the cards on disk have just changed
            self._draw()
            self.show()
            self.refreshed.emit(digest.verdict)
        finally:
            # A read asked for while this pass held the slot. It is worth
            # running even now: `src` may have changed under the scoring
            # pass, and this is what puts the current source back on screen.
            self._pending_work()


# ---------------------------------------------------------------------------
# The diameter estimator
# ---------------------------------------------------------------------------

class DiameterPanel(_JobMixin, QFrame):
    """A measured Cellpose ``diameter``, per object type, from the user's own fields.

    ``CellposeModel.eval(diameter=...)`` under Cellpose 4 rescales the input
    by ``30/diameter`` so objects land near the size ``cpsam`` works at, which
    makes a two-fold error in this one number a two-fold error in every mask,
    count and measurement downstream. :mod:`spacr.diameter` measures it from a
    handful of sampled fields — no Cellpose, no torch — and this panel is
    where that measurement reaches the settings form.

    Every row carries its evidence: the value, the 10th-90th percentile range,
    how many objects it was pooled from, how many fields contributed, how it
    was measured and how much to trust it. A proposal without those is just a
    different guess.

    :param screen: the ``AppScreen`` it belongs to.
    :param estimator: what to call to estimate, for tests. Defaults to
        :func:`spacr.diameter.estimate_diameters`.
    :param parent: parent widget; ownership only.
    """

    #: Emitted after every estimate, with the object types that produced one.
    estimated = Signal(list)

    def __init__(self, screen: QWidget, *, estimator=None, parent=None) -> None:
        super().__init__(parent or screen)
        self.setObjectName(DIAMETER_OBJECT_NAME)
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self._init_jobs()

        self._screen = screen
        self._estimator = estimator
        self._estimates: Dict[str, Any] = {}

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        head.addWidget(_label("Diameter — measure it, do not guess it",
                              "PrerunTitle", wrap=False))
        head.addStretch(1)
        self._btn_measure = QPushButton("Measure from my images")
        self._btn_measure.setObjectName("GhostButton")
        self._btn_measure.setCursor(Qt.PointingHandCursor)
        self._btn_measure.clicked.connect(self._on_measure_clicked)
        head.addWidget(self._btn_measure)
        self._btn_use_all = QPushButton("Use all")
        self._btn_use_all.setObjectName("GhostButton")
        self._btn_use_all.setCursor(Qt.PointingHandCursor)
        self._btn_use_all.clicked.connect(self._on_use_all_clicked)
        self._btn_use_all.hide()
        head.addWidget(self._btn_use_all)
        column.addLayout(head)

        self._sub = _label(
            "Cellpose 4 rescales every image by 30/diameter before it "
            "segments, so a diameter two-fold off moves your objects out of "
            "the size cpsam works at. This reads a few fields of your own "
            "data and measures it — Cellpose is not loaded.",
            "PrerunSub")
        column.addWidget(self._sub)

        # Same as the QC panel's findings box: scaffolding paints nothing.
        self._rows_box = _transparent(QWidget(self))
        self._rows_layout = QVBoxLayout(self._rows_box)
        self._rows_layout.setContentsMargins(0, 2, 0, 2)
        self._rows_layout.setSpacing(6)
        self._rows_box.setVisible(False)
        column.addWidget(self._rows_box)

        self._status = _label("", "PrerunNote")
        self._status.hide()
        column.addWidget(self._status)

        self._advisory = _label(
            "Advisory only — nothing here changes a setting until you press Use.",
            "PrerunAdvisory")
        column.addWidget(self._advisory)

    # -- inputs -----------------------------------------------------------

    def _settings(self) -> Dict[str, Any]:
        model = getattr(self._screen, "_settings_model", None)
        if model is None:
            return {}
        try:
            return dict(model.collect() or {})
        except Exception:
            LOG.exception("could not read the mask settings")
            return {}

    def _channels(self, settings: Dict[str, Any]) -> Dict[str, int]:
        """``{object_type: channel}`` for the object types this screen offers."""
        from ..diameter import channels_from_settings
        found = channels_from_settings(settings)
        widgets = _widgets(self._screen)
        return {
            obj: channel for obj, channel in found.items()
            if obj in _DIAMETER_OBJECTS and f"{obj}_diameter" in widgets
        }

    # -- measuring --------------------------------------------------------

    def _on_measure_clicked(self) -> None:
        src = _src_of(self._screen)
        if not _has_src(src):
            self._say("Point src at a plate folder first — there is nothing "
                      "to measure yet.")
            return
        if self.busy:
            return
        settings = self._settings()
        channels = self._channels(settings)
        if not channels:
            self._say(
                "No object channel is set. Fill in cell_channel, "
                "nucleus_channel or pathogen_channel — they are 0-based "
                "indices into the sorted channel IDs — and measure again.")
            return

        estimator = self._estimator

        def _job(box: Dict[str, Any]) -> None:
            """Run the estimator. Off the GUI thread."""
            fn = box["estimator"]
            if fn is None:
                from ..diameter import estimate_diameters as fn  # noqa: N806
            box["estimates"] = fn(
                box["src"],
                box["channels"],
                n_fields=box["n_fields"],
                metadata_type=box["metadata_type"],
                custom_regex=box["custom_regex"],
            )

        try:
            n_fields = int(settings.get("diameter_estimate_n_fields") or 5)
        except (TypeError, ValueError):
            n_fields = 5
        box = {
            "src": src,
            "channels": channels,
            "n_fields": max(1, n_fields),
            "metadata_type": settings.get("metadata_type", "cellvoyager"),
            "custom_regex": settings.get("custom_regex"),
            "estimator": estimator,
        }
        self._btn_measure.setEnabled(False)
        self._say(
            f"Measuring {', '.join(sorted(channels))} across "
            f"{box['n_fields']} field(s)…")
        if not self._start_job(_job, box, self._on_estimated, "diameter"):
            self._btn_measure.setEnabled(True)
            self._say("Could not start the measurement.")

    def _on_estimated(self, box: Dict[str, Any]) -> None:
        """Draw the proposals. Always on the GUI thread."""
        self._btn_measure.setEnabled(True)
        estimates = box.get("estimates")
        if not isinstance(estimates, dict):
            self._say("Could not measure a diameter from these images: "
                      f"{box.get('error') or 'the estimator failed'}.")
            self.estimated.emit([])
            return
        self._estimates = estimates
        self._draw_rows()
        usable = [obj for obj, est in estimates.items() if est.usable]
        self._btn_use_all.setVisible(bool(usable))
        self._status.hide()
        self.estimated.emit(usable)

    def _say(self, text: str) -> None:
        self._status.setText(text)
        self._status.setVisible(bool(text))

    # -- drawing ----------------------------------------------------------

    @property
    def estimates(self) -> Dict[str, Any]:
        """The last set of proposals, keyed by object type."""
        return dict(self._estimates)

    def _clear_rows(self) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _draw_rows(self) -> None:
        self._clear_rows()
        order = [o for o in _DIAMETER_OBJECTS if o in self._estimates]
        order += [o for o in self._estimates if o not in order]
        for obj in order:
            est = self._estimates[obj]
            row = _transparent(QWidget(self._rows_box))
            layout = QVBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            line = QHBoxLayout()
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(8)
            if est.usable:
                line.addWidget(_label(
                    f"{obj}: {est.diameter:.1f} px", "PrerunValue", wrap=False))
                # The evidence, not just the number. A proposal whose object
                # count is 3 is a different claim from one whose count is 800,
                # and the user cannot tell them apart from the value alone.
                line.addWidget(_label(
                    f"10th-90th percentile {est.low:.1f}-{est.high:.1f} px · "
                    f"measured on {est.n_objects} object(s) across "
                    f"{est.n_fields} field(s) · {est.confidence} confidence · "
                    f"{est.method}",
                    "PrerunSub", wrap=False))
                line.addStretch(1)
                button = QPushButton(f"Use {est.diameter:.0f}")
                button.setObjectName("GhostButton")
                button.setCursor(Qt.PointingHandCursor)
                button.clicked.connect(
                    lambda _checked=False, o=obj: self.apply(o))
                line.addWidget(button)
            else:
                line.addWidget(_label(
                    f"{obj}: no estimate", "PrerunWarn", wrap=False))
                line.addStretch(1)
            layout.addLayout(line)
            layout.addWidget(_label(est.note, "PrerunNote"))
            self._rows_layout.addWidget(row)
        self._rows_box.setVisible(bool(order))

    # -- applying ---------------------------------------------------------

    def apply(self, object_type: str) -> bool:
        """Write one proposal into its ``<object>_diameter`` field.

        Only ever called from the row's own button — nothing here writes a
        setting on its own, and an unusable estimate (NaN, by construction in
        :class:`spacr.diameter.DiameterEstimate`) is never written at all.

        :param object_type: ``'cell'``, ``'nucleus'`` or ``'pathogen'``.
        :returns: True when the value reached the widget.
        """
        est = self._estimates.get(object_type)
        if est is None or not est.usable:
            return False
        model = getattr(self._screen, "_settings_model", None)
        key = f"{object_type}_diameter"
        # An int, because `spacr.settings.expected_types` declares these keys
        # int and `collect()` hands a float straight back as the *string*
        # "24.0" — which then reaches check_settings as a string. Sub-pixel
        # precision is meaningless here anyway: the value's only effect is a
        # 30/diameter rescale. The panel still SHOWS the measured value to a
        # decimal, so nothing about the measurement is hidden.
        value = int(round(float(est.diameter)))
        setter = getattr(model, "set_value_for_key", None)
        if callable(setter):
            try:
                if setter(key, value):
                    self._say(f"{key} set to {value:g} px.")
                    return True
            except Exception:
                LOG.exception("could not write %s", key)
        widget = _widgets(self._screen).get(key)
        applier = getattr(self._screen, "_apply_value", None)
        if widget is not None and callable(applier):
            try:
                applier(widget, value)
                self._say(f"{key} set to {value:g} px.")
                return True
            except Exception:
                LOG.exception("could not write %s", key)
        return False

    def _on_use_all_clicked(self) -> None:
        applied = [obj for obj in self._estimates if self.apply(obj)]
        if applied:
            self._say("Set " + ", ".join(
                f"{obj}_diameter" for obj in sorted(applied)) + ".")


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def qc_banner(screen) -> Optional[SegQCBanner]:
    """The banner installed on ``screen``, or None."""
    found = getattr(screen, "_seg_qc_banner", None)
    return found if isinstance(found, SegQCBanner) else None


def diameter_panel(screen) -> Optional[DiameterPanel]:
    """The diameter panel installed on ``screen``, or None."""
    found = getattr(screen, "_diameter_panel", None)
    return found if isinstance(found, DiameterPanel) else None


def _insert_above_actions(screen, widget) -> bool:
    """Put ``widget`` in the runtime panel just above the Run row.

    Both anchors (``_runtime_wrap`` and ``_actions_row``) are attributes
    ``AppScreen`` keeps for exactly this kind of reach, so nothing here
    depends on that panel's internal layout order. Above the actions row is
    the last thing the eye crosses on its way to Run, which is the whole
    point: a panel the user would have to go and open is a panel nobody opens.
    """
    wrap = getattr(screen, "_runtime_wrap", None)
    actions = getattr(screen, "_actions_row", None)
    if wrap is None or actions is None:
        return False
    layout = wrap.layout()
    if layout is None:
        return False
    index = layout.indexOf(actions)
    layout.insertWidget(index if index >= 0 else layout.count(), widget)
    return True


def install_qc_banner(screen, *, reader=None,
                      threaded: bool = True) -> Optional[SegQCBanner]:
    """Put a :class:`SegQCBanner` above ``screen``'s Run row.

    :param screen: an ``AppScreen``.
    :param reader: digest reader, for tests.
    :param threaded: ``False`` to read inline, for tests.
    :returns: the banner, or None when this screen cannot carry one. Never
        raises: a screen that opens without the banner is the old behaviour,
        and that is always better than a screen that does not open.
    """
    try:
        existing = qc_banner(screen)
        if existing is not None:
            return existing
        banner = SegQCBanner(screen, reader=reader, threaded=threaded)
        if not _insert_above_actions(screen, banner):
            banner.setParent(None)
            banner.deleteLater()
            return None
        screen._seg_qc_banner = banner
        # SCHEDULED, NOT CALLED. This runs inside `MainWindow._build_screen`,
        # which does not yield -- so anything done here is done before the
        # screen can be painted, and until 2026-09-04 that included statting
        # the user's src folder. See `SegQCBanner.refresh`.
        banner.schedule_refresh()
        return banner
    except Exception:
        LOG.exception("could not install the segmentation-QC banner on %s",
                      getattr(screen, "app_key", "?"))
        return None


def install_diameter_panel(screen, *, estimator=None) -> Optional[DiameterPanel]:
    """Put a :class:`DiameterPanel` above ``screen``'s Run row.

    :param screen: an ``AppScreen``.
    :param estimator: diameter estimator, for tests.
    :returns: the panel, or None when this screen cannot carry one.
    """
    try:
        existing = diameter_panel(screen)
        if existing is not None:
            return existing
        if not any(f"{obj}_diameter" in _widgets(screen)
                   for obj in _DIAMETER_OBJECTS):
            # A screen with no diameter to set has no use for an estimate.
            return None
        panel = DiameterPanel(screen, estimator=estimator)
        if not _insert_above_actions(screen, panel):
            panel.setParent(None)
            panel.deleteLater()
            return None
        screen._diameter_panel = panel
        return panel
    except Exception:
        LOG.exception("could not install the diameter panel on %s",
                      getattr(screen, "app_key", "?"))
        return None


def install(screen) -> None:
    """Install whichever of the two belongs on ``screen``."""
    key = str(getattr(screen, "app_key", ""))
    if key == QC_APP:
        install_qc_banner(screen)
    if key == DIAMETER_APP:
        install_diameter_panel(screen)


#: app key -> the factory this module displaced, so it can delegate to it.
_INNER: Dict[str, Any] = {}


def _call(factory, app_key: str, host):
    """Invoke a screen factory with the arguments it declares.

    The same contract ``spacr.qt.app._call_screen_factory`` implements, by
    inspection rather than by calling and retrying on ``TypeError`` — a retry
    cannot tell a wrong call from a ``TypeError`` raised inside a factory that
    was called correctly, and would then build the screen twice.
    """
    kwargs: Dict[str, Any] = {}
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        params = {}
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD
                    for p in params.values())
    for wanted, value in (("app_key", app_key), ("host", host)):
        if takes_any or wanted in params:
            kwargs[wanted] = value
    return factory(**kwargs)


def _base_screen(app_key: str, host=None):
    """Build the screen this module then decorates.

    Delegates to whatever factory was registered for this key before us —
    :mod:`spacr.qt.chaining` registers one for every module that declares
    ports — so installing this never costs a screen the strip it already had.
    When there was none, the generic ``AppScreen`` is built here and given the
    same host connections and the same chaining strip ``_build_screen`` and
    ``chaining`` would have given it, which is what makes the two
    registrations order-independent.
    """
    inner = _INNER.get(app_key)
    if inner is not None:
        return _call(inner, app_key, host)

    from .screens.app_screen import AppScreen
    screen = AppScreen(app_key=app_key)
    try:
        from .chaining import HOST_CONNECTIONS, install_chaining
        if host is not None:
            for signal_name, slot_name in HOST_CONNECTIONS.items():
                signal = getattr(screen, signal_name, None)
                slot = getattr(host, slot_name, None)
                if signal is not None and callable(slot):
                    signal.connect(slot)
        install_chaining(screen)
    except Exception:
        LOG.exception("could not wire %s the way _build_screen does", app_key)
    return screen


def _prerun_screen(app_key: str, host=None):
    """The registered factory: build the screen, then decorate it."""
    screen = _base_screen(app_key, host)
    install(screen)
    return screen


def register() -> bool:
    """Install the banner and the panel on their screens. Idempotent.

    Called by :func:`spacr.qt.register_self_registering_modules` after
    ``app.py`` has finished importing and before the first window is built.

    :returns: True when anything was registered.
    """
    from .app import APP_FACTORIES

    try:
        # Already registered at import (see the Styling section). Repeated
        # here because `teardown()` unregisters it, so a register/teardown/
        # register cycle -- which the tests do -- has to put it back.
        # `replace=True` makes the ordinary case a no-op.
        from .theme import register_widget_qss
        register_widget_qss(QSS_NAME, _qss, replace=True)
    except Exception:
        LOG.exception("could not register the pre-run stylesheet")

    installed = False
    for key in (QC_APP, DIAMETER_APP):
        current = APP_FACTORIES.get(key)
        if current is _prerun_screen:
            continue
        if current is not None:
            _INNER[key] = current
        APP_FACTORIES[key] = _prerun_screen
        installed = True
    return installed


def unregister() -> int:
    """Undo :func:`register`, restoring whatever factory was displaced.

    :returns: how many keys were handed back.
    """
    from .app import APP_FACTORIES

    restored = 0
    for key in list(APP_FACTORIES):
        if APP_FACTORIES[key] is not _prerun_screen:
            continue
        inner = _INNER.pop(key, None)
        if inner is None:
            APP_FACTORIES.pop(key, None)
        else:
            APP_FACTORIES[key] = inner
        restored += 1
    try:
        from .theme import unregister_widget_qss
        unregister_widget_qss(QSS_NAME)
    except Exception:
        LOG.exception("could not remove the pre-run stylesheet")
    return restored
