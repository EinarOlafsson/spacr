"""One screen that answers "is this run usable?".

The screen half of :mod:`spacr.qt.widgets.qc_summary`. It shows the
segmentation, units, leakage, plate-effect and agreement verdicts side by
side, with the one-line summary they add up to.

It **reads**; it does not score. The rule is :mod:`spacr.qt.prerun`'s, and
that module says why: opening a plate's masks costs seconds to minutes, and a
screen that pays that on every visit is a screen nobody keeps. So the reads go
through a fingerprint cache -- one listdir and one stat per artifact -- and
the parse only happens when something on disk has actually changed.

Nothing here disables anything. ``Dashboard.blocks_run`` is a constant False,
and this screen has no Run button to gate. A QC verdict that stops work gets
switched off; one that informs it gets read.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from ..job_runner import JobRunner
from ..theme import SPACING, register_widget_qss
from .app_screen import ModuleHeader
from ..widgets.qc_summary import (
    Dashboard, format_dashboard, read_dashboard,
)

__all__ = [
    "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO", "APP_CLI_NOTE",
    "APP_TRANSLATIONS", "QCDashboardScreen", "make_qc_dashboard_screen",
    "register",
]

#: Stable app id. Chosen once; saved user state and the registry key off it.
APP_KEY = "qc_dashboard"

APP_NAME = "QC Dashboard"
APP_DESCRIPTION = (
    "Segmentation, units, leakage, plate effects and annotator agreement in "
    "one place, with the verdict they add up to."
)
APP_INTRO = (
    "Every verdict here was written by the run that produced it -- this "
    "screen reads them, it does not score anything, so opening it costs a "
    "directory listing rather than minutes of mask loading. A card whose "
    "inputs are newer than it is says OUT OF DATE rather than pretending to "
    "describe them. A card that says 'missing' means the check has not been "
    "run, which is not the same as clean."
)
APP_CLI_NOTE = (
    "The QC Dashboard is a GUI screen: it aggregates verdicts other runs "
    "wrote so they can be read together. Headless, call "
    "spacr.qt.widgets.qc_summary.read_dashboard(src) and "
    "format_dashboard() instead -- that is the same code this screen runs."
)
#: sv, de, es, zh_CN, pt, hi, ko, is, fr
APP_TRANSLATIONS: Tuple[str, ...] = (
    "QC-panel", "QC-Übersicht", "Panel de control de QC", "质控面板",
    "Painel de QC", "QC डैशबोर्ड", "QC 대시보드", "Gæðayfirlit",
    "Tableau de bord QC",
)

LOG = logging.getLogger(__name__)

VERDICT_OBJECT = "spacrQCVerdict"
CARDS_OBJECT = "spacrQCCards"
STATUS_OBJECT = "spacrQCStatus"


def _dashboard_qss(palette: dict, opacity: Optional[float] = None) -> str:
    """QSS for this screen, rebuilt on every theme change.

    ``palette`` arrives with its surface roles already rendered through the
    page-opacity preference, so every rule below that names one follows the
    slider. The card panel needs a rule at all for that to matter: a named
    ``QWidget`` with no rule of its own falls back to the blanket
    ``QWidget {{ background-color: bg }}``, which is the WINDOW colour and
    not a surface, and no setting can reach it.
    """
    from ..theme import block_surface
    cards_bg = block_surface("surface_alt", palette.get("theme"), opacity)
    return f"""
#{CARDS_OBJECT} {{
    background: {cards_bg};
    border: 1px solid {palette['border_soft']};
    border-radius: 6px;
}}
#{VERDICT_OBJECT} {{
    color: {palette['fg']};
    background: {palette['surface_alt']};
    border: 1px solid {palette['border']};
    border-radius: 6px;
    padding: {SPACING['sm']}px;
}}
/* The screen's own plain labels -- the intro paragraph and the "Folder:"
   caption. They sit on the page rather than on a panel, and `page` is not
   `bg` (INVARIANTS 2), so a label painting the window colour shows as a
   black rectangle there too. */
#QCDashboardScreen > QLabel {{
    background: transparent;
}}

/* Every label on the cards panel, before the colour rules below.
   A QLabel is a QWidget, so a label with no background of its own is
   matched by the blanket `QWidget {{ background-color: bg }}` and paints
   the WINDOW colour -- #000000 on dark -- as a solid rectangle behind its
   own text, on top of a panel that DOES have a background. That is the
   black box behind the segmentation-QC text.

   Transparent, not a colour: the panel's background already carries the
   user's page opacity through `block_surface`, and repeating a colour here
   would freeze one opacity into the labels while the panel behind them
   kept following the preference. */
#{CARDS_OBJECT} QLabel {{
    background: transparent;
}}
#{STATUS_OBJECT} {{
    background: transparent;
}}
#{CARDS_OBJECT} QLabel[spacrQCVerdictLevel="ok"] {{
    color: {palette['success']};
}}
#{CARDS_OBJECT} QLabel[spacrQCVerdictLevel="warn"] {{
    color: {palette['warning']};
}}
#{CARDS_OBJECT} QLabel[spacrQCVerdictLevel="fail"] {{
    color: {palette['error']};
}}
#{CARDS_OBJECT} QLabel[spacrQCVerdictLevel="error"] {{
    color: {palette['error']};
}}
#{CARDS_OBJECT} QLabel[spacrQCVerdictLevel="missing"] {{
    color: {palette['fg_muted']};
}}
#{CARDS_OBJECT} QLabel[spacrQCStale="true"] {{
    border-left: 3px solid {palette['warning']};
    padding-left: {SPACING['sm']}px;
}}
#{CARDS_OBJECT} QLabel[spacrQCRole="detail"] {{
    color: {palette['fg_muted']};
}}
#{STATUS_OBJECT} {{ color: {palette['fg_muted']}; }}
#{STATUS_OBJECT}[spacrError="true"] {{ color: {palette['error']}; }}
"""


# `replace=True`: reachable both through the screens package and by direct
# import, and a second import must refresh the block rather than raise.
register_widget_qss("QCDashboard", _dashboard_qss, replace=True)


class QCDashboardScreen(QWidget):
    """Read every QC verdict for a project and show them together.

    :param src: project folder to read; may be set later.
    :param threaded: ``False`` reads inline, emitting the same signals in
        the same order, so a test can drive the screen synchronously.
    :param reader: substitute for
        :func:`spacr.qt.widgets.qc_summary.read_dashboard`, for tests.
    """

    def __init__(self, parent: Optional[QWidget] = None, *,
                 src: Any = "", threaded: bool = True, reader=None) -> None:
        super().__init__(parent)
        self.setObjectName("QCDashboardScreen")
        self._jobs = JobRunner(self, threaded=threaded, app_key=APP_KEY)
        self._jobs.job_failed.connect(self._on_job_failed)
        self._reader = reader
        self._dashboard: Optional[Dashboard] = None
        self._cache_key: Any = None
        self._card_labels: List[QLabel] = []
        self._build()
        if src:
            self.set_source(src)
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "qc_dashboard")

    # -- construction -----------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"],
                                 SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["md"])

        header = ModuleHeader(
            APP_NAME,
            description=APP_DESCRIPTION,
            instruction="Point it at a project or plate folder, then "
                        "refresh.",
        )
        self._header = header
        outer.addWidget(header)

        intro = QLabel(APP_INTRO)
        intro.setWordWrap(True)
        outer.addWidget(intro)

        row = QHBoxLayout()
        row.setSpacing(SPACING["sm"])
        self._src_edit = QLineEdit("")
        self._src_edit.setPlaceholderText("project or plate folder")
        self._src_edit.returnPressed.connect(self.refresh)
        row.addWidget(QLabel("Folder:"))
        row.addWidget(self._src_edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._on_browse)
        row.addWidget(browse)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        outer.addLayout(row)

        self._verdict = QLabel("No folder set.")
        self._verdict.setObjectName(VERDICT_OBJECT)
        self._verdict.setWordWrap(True)
        outer.addWidget(self._verdict)

        self._cards_panel = QWidget()
        self._cards_panel.setObjectName(CARDS_OBJECT)
        self._cards_layout = QVBoxLayout(self._cards_panel)
        # Room for the panel's own border: the cards sit ON a surface now
        # rather than directly on the window, and zero margins would put the
        # first heading through the hairline.
        self._cards_layout.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                              SPACING["sm"], SPACING["sm"])
        self._cards_layout.setSpacing(SPACING["sm"])
        self._cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidget(self._cards_panel)
        # setWidget() enables autoFillBackground on its child. This panel
        # already owns a QSS surface, so leaving that flag on paints the same
        # 30% fill twice (0.7 * 0.7 = 0.49 transmission).
        self._cards_panel.setAutoFillBackground(False)
        scroll.setWidgetResizable(True)
        # A QScrollArea's viewport auto-fills by default, and what it fills
        # with is the WINDOW colour -- not a surface -- so no page-opacity
        # setting can reach it and the card column reads as an opaque slab
        # over the animated backdrop. Same call the settings column and the
        # sidebar make.
        scroll.viewport().setAutoFillBackground(False)
        # ...and tag it, because autoFillBackground(False) does NOT stop a
        # STYLESHEET background: QSS paints through QStyle regardless of that
        # flag, so the blanket `QWidget { background-color: bg }` still
        # reaches the viewport. `make_transparent` tags a scroll area's
        # viewport along with it, which is the whole reason it takes one.
        try:
            from ..theme import make_transparent
            make_transparent(scroll)
        except Exception:      # pragma: no cover - decoration is not load-bearing
            LOG.debug("could not make the QC scroll area transparent",
                      exc_info=True)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        outer.addWidget(scroll, 1)

        self._status = QLabel("")
        self._status.setObjectName(STATUS_OBJECT)
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

    # -- reading ----------------------------------------------------------

    def set_source(self, src: Any) -> None:
        """Point the screen at a project folder and read it."""
        self._src_edit.setText(str(src))
        self.refresh()

    def source(self) -> str:
        """The folder currently shown."""
        return self._src_edit.text().strip()

    def _fingerprint(self, src: str) -> Optional[Tuple]:
        """A cheap key that changes exactly when the artifacts do.

        One stat per artifact, so returning to the screen ten times does not
        re-parse ten times, while a re-mask that rewrites a scorecard is
        picked up on the next visit. ``None`` forces a read rather than
        trusting a cache that could not be verified.
        """
        try:
            from ...seg_qc import find_scorecards
            paths = list(find_scorecards(src))
        except Exception:
            return None
        for extra in ("measurements/measurements.db", "measurements.db"):
            candidate = os.path.join(src, extra)
            if os.path.isfile(candidate):
                paths.append(candidate)
        out = []
        for path in paths:
            try:
                info = os.stat(path)
            except OSError:
                return None
            out.append((path, info.st_mtime_ns, info.st_size))
        return tuple(out)

    def refresh(self, *, force: bool = False) -> bool:
        """Re-read the verdicts. Off the GUI thread.

        :param force: read even when the fingerprint says nothing changed.
        :returns: whether a read was started.
        """
        src = self.source()
        if not src:
            self._verdict.setText("No folder set.")
            self._set_status("Pick a project folder to read its verdicts.")
            return False
        if not os.path.isdir(src):
            self._verdict.setText("That folder does not exist.")
            self._set_status(f"{src} is not a folder.", is_error=True)
            return False

        key = (src, self._fingerprint(src))
        if not force and self._dashboard is not None and key == self._cache_key \
                and key[1]:
            self._draw(self._dashboard)
            self._set_status("Nothing on disk has changed since the last read.")
            return False
        self._cache_key = key

        reader = self._reader or read_dashboard
        self._jobs.cancel()
        self._set_status("Reading...")
        return self._jobs.submit(lambda s=src, r=reader: r(s), self._on_read)

    def _on_read(self, dashboard) -> None:
        if dashboard is None:
            return
        self._dashboard = dashboard
        self._draw(dashboard)
        self._set_status(
            "Read from disk; nothing was recomputed."
            + (" One or more cards are out of date -- their inputs have been "
               "written again since they were scored."
               if dashboard.stale else ""))

    def dashboard(self) -> Optional[Dashboard]:
        """The most recent :class:`~spacr.qt.widgets.qc_summary.Dashboard`."""
        return self._dashboard

    # -- drawing ----------------------------------------------------------

    def _draw(self, dashboard: Dashboard) -> None:
        self._verdict.setText(
            f"{dashboard.verdict.upper()} — {dashboard.headline}")
        self._verdict.setProperty("spacrQCVerdictLevel", dashboard.verdict)

        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self._card_labels = []

        for card in dashboard.cards:
            heading = QLabel(
                f"[{card.display_verdict}]  {card.title} — {card.headline}")
            heading.setWordWrap(True)
            heading.setProperty("spacrQCVerdictLevel", card.verdict)
            heading.setProperty("spacrQCStale", "true" if card.stale
                                else "false")
            heading.setProperty("cardKey", card.key)
            if card.source:
                heading.setToolTip(card.source)
            self._cards_layout.addWidget(heading)
            self._card_labels.append(heading)
            for line in card.detail:
                detail = QLabel(line)
                detail.setWordWrap(True)
                detail.setProperty("spacrQCRole", "detail")
                detail.setIndent(SPACING["md"])
                self._cards_layout.addWidget(detail)
                self._card_labels.append(detail)
            if card.verdict == "missing" and card.how_to_produce:
                todo = QLabel(f"-> {card.how_to_produce}")
                todo.setWordWrap(True)
                todo.setProperty("spacrQCRole", "detail")
                todo.setIndent(SPACING["md"])
                self._cards_layout.addWidget(todo)
                self._card_labels.append(todo)

    def visible_text(self) -> str:
        """Every card line currently on screen, joined. For tests."""
        return "\n".join(label.text() for label in self._card_labels)

    def as_text(self) -> str:
        """The dashboard as plain text, for a log or a paste."""
        if self._dashboard is None:
            return "No folder read yet."
        return format_dashboard(self._dashboard)

    def _set_status(self, text: str, *, is_error: bool = False) -> None:
        self._status.setText(text)
        self._status.setProperty("spacrError", "true" if is_error else "false")
        style = self._status.style()
        if style is not None:
            style.unpolish(self._status)
            style.polish(self._status)

    def status_text(self) -> str:
        """The status line. For tests."""
        return self._status.text()

    # -- events -----------------------------------------------------------

    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Project folder")
        if folder:
            self.set_source(folder)

    def _on_job_failed(self, message: str) -> None:
        self._set_status(f"Could not read the verdicts: {message}",
                         is_error=True)

    # -- lifecycle --------------------------------------------------------

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return self._jobs.active_jobs()

    def is_busy(self) -> bool:
        """True while a read has not delivered its result."""
        return self._jobs.is_busy()

    def closeEvent(self, event):  # noqa: N802 - Qt name
        self._jobs.shutdown()
        super().closeEvent(event)


def make_qc_dashboard_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return QCDashboardScreen()


def register() -> bool:
    """Add the QC Dashboard to the app registry. Idempotent."""
    from ..app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app

    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(
        APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_EXPLORE,
        factory=make_qc_dashboard_screen, stage=STAGE_ALPHA,
        title=APP_NAME, intro=APP_INTRO, cli_note=APP_CLI_NOTE,
        api_module="qt/screens/qc_dashboard",
        translations=APP_TRANSLATIONS)
    return True


register()
