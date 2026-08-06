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

Installation goes through the seams that already exist rather than through the
shared screen: :data:`spacr.qt.app.APP_FACTORIES`, consulted by
``MainWindow._build_screen``, and :func:`spacr.qt.theme.register_widget_qss`
for the colours.  ``AppScreen`` is untouched.  A factory already registered
for one of these keys — :mod:`spacr.qt.chaining` registers one for every
module that declares ports — is kept and delegated to, so installing this
never costs a screen the strip it already had, in either registration order.
"""
from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

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
    QLabel#PrerunSub, QLabel#PrerunNote {{
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
except Exception:            # pragma: no cover - decoration is not load-bearing
    # INVARIANTS 10: a stylesheet that cannot be registered costs this panel
    # its background, not the Measure module its run.
    LOG.exception("could not register the pre-run stylesheet at import")


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

class _ShowFilter(QObject):
    """Calls back when the watched widget is shown.

    A module screen is built once and kept, so ``__init__`` fires exactly
    once — while *returning* to the screen, which is when a mask run on
    another tab may have replaced everything this widget is about, fires
    ``Show``.
    """

    def __init__(self, on_show, parent=None) -> None:
        super().__init__(parent)
        self._on_show = on_show

    def eventFilter(self, obj, event) -> bool:      # noqa: N802 - Qt override
        """Forward a Show event and never consume it."""
        if event.type() == QEvent.Show:
            try:
                self._on_show()
            except Exception:
                LOG.exception("pre-run refresh failed on show")
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
    """

    def _init_jobs(self) -> None:
        self._jobs: List[Tuple[Any, Any]] = []
        self._busy = False

    @property
    def busy(self) -> bool:
        """True while a background job is in flight."""
        return bool(getattr(self, "_busy", False))

    def _start_job(self, fn, box: Dict[str, Any], on_done, app_key: str) -> bool:
        """Run ``fn(box)`` on a worker thread; call ``on_done(box)`` after."""
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
            thread, worker = make_thread(fn, box, app_key=app_key, journal=False)
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
    """

    #: Emitted after every refresh, with the verdict. Tests wait on it; the
    #: screen ignores it.
    refreshed = Signal(str)

    def __init__(self, screen: QWidget, *, reader=None, parent=None) -> None:
        super().__init__(parent or screen)
        self.setObjectName(QC_OBJECT_NAME)
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self._init_jobs()

        self._screen = screen
        self._reader = reader
        self._digest = None
        self._expanded = False
        self._cache_key: Optional[Tuple] = None

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

        self._show_filter = _ShowFilter(self._on_screen_shown, self)
        try:
            screen.installEventFilter(self._show_filter)
        except Exception:
            LOG.exception("could not watch the Measure screen for show events")
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

        One ``listdir`` plus one ``stat`` per card — microseconds — so that
        returning to the screen ten times does not re-parse ten times, while a
        re-mask that rewrites a card is still picked up on the next visit.
        Returns None when the fingerprint cannot be taken, which forces a
        read rather than trusting a stale cache.
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
        """Re-read the verdict and redraw. Cheap; never scores a mask."""
        src = _src_of(self._screen)
        if not _has_src(src):
            self._digest = None
            self._cache_key = None
            self.hide()
            self.refreshed.emit("")
            return
        key = (repr(src), self._fingerprint(src))
        if self._digest is not None and key == self._cache_key and key[1]:
            self.show()
            self.refreshed.emit(self._digest.verdict)
            return
        try:
            digest = self._read(src)
        except Exception:
            LOG.exception("could not read the segmentation verdict")
            self._digest = None
            self.hide()
            self.refreshed.emit("")
            return
        self._digest = digest
        self._cache_key = key
        self._draw()
        self.show()
        self.refreshed.emit(digest.verdict)

    # -- drawing ----------------------------------------------------------

    @property
    def digest(self):
        """The last digest read, or None."""
        return self._digest

    def _draw(self) -> None:
        digest = self._digest
        if digest is None:
            return
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
        if not findings:
            self._btn_more.hide()
            self._findings_box.setVisible(False)
            return
        self._findings_box.setVisible(True)
        shown = findings if self._expanded else findings[:_FINDINGS_COLLAPSED]
        for finding in shown:
            block = QWidget(self._findings_box)
            layout = QVBoxLayout(block)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            head = _label(
                f"• {finding.headline}",
                _SEVERITY_NAME.get(finding.severity, "PrerunHeadline"))
            layout.addWidget(head)
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
        """
        src = _src_of(self._screen)
        if not _has_src(src) or self.busy:
            return
        settings: Dict[str, Any] = {}
        model = getattr(self._screen, "_settings_model", None)
        if model is not None:
            try:
                settings = dict(model.collect() or {})
            except Exception:
                settings = {}

        def _job(box: Dict[str, Any]) -> None:
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
        """Show what the scoring pass found. Always on the GUI thread."""
        self._btn_score.setEnabled(True)
        digest = box.get("digest")
        if digest is None:
            self._title.setText("Segmentation QC — could not score these masks")
            return
        self._digest = digest
        self._cache_key = None          # the cards on disk have just changed
        self._draw()
        self.show()
        self.refreshed.emit(digest.verdict)


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
            fn = box["estimator"]
            if fn is None:
                from ..diameter import estimate_diameters as fn      # noqa: N806
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
            row = QWidget(self._rows_box)
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


def install_qc_banner(screen, *, reader=None) -> Optional[SegQCBanner]:
    """Put a :class:`SegQCBanner` above ``screen``'s Run row.

    :param screen: an ``AppScreen``.
    :param reader: digest reader, for tests.
    :returns: the banner, or None when this screen cannot carry one. Never
        raises: a screen that opens without the banner is the old behaviour,
        and that is always better than a screen that does not open.
    """
    try:
        existing = qc_banner(screen)
        if existing is not None:
            return existing
        banner = SegQCBanner(screen, reader=reader)
        if not _insert_above_actions(screen, banner):
            banner.setParent(None)
            banner.deleteLater()
            return None
        screen._seg_qc_banner = banner
        banner.refresh()
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
