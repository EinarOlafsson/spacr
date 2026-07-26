"""Report — the Tools module that turns a run folder into one shareable file.

The screen is deliberately thin. Everything it knows about a run folder it
learns from :mod:`spacr.report`, which is headless, read-only and testable
without Qt. This file is the part that has to be a GUI: pick a folder, say
what was found and — just as loudly — what was **not**, choose a format,
and write the file off the GUI thread.

Two decisions are worth stating, because both are visible to the user:

* **Missing sections are shown, greyed, with the reason.** The section list
  is not "here is what you will get"; it is "here is what exists and here is
  what does not". A run with no segmentation QC shows *Segmentation QC —
  not available* before you generate anything, so you find out before your
  collaborator does.
* **No modal dialogs, ever.** Every failure — a folder that is not a folder,
  an unwritable output path, a crash inside collection — lands in the inline
  status label and in :attr:`ReportScreen.last_error`. A ``QMessageBox``
  hangs a headless run (it did, in ``MakeMasksScreen``), and this screen is
  exercised headlessly.

Collection walks the folder and base64-encodes figures, which is slow enough
on a full plate to freeze the window, so both scanning and generating go
through :func:`spacr.qt.bridge.make_thread` like every other spaCR job.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ... import report as rep
from ..bridge import make_thread
from ..theme import PALETTE, SPACING
from ..widgets import Divider

__all__ = ["ReportScreen", "FORMATS", "FIGURE_CAP_RANGE"]


#: Output formats offered, mapping the label to ``build_report``'s ``fmt``.
FORMATS: Tuple[Tuple[str, str], ...] = (
    ("HTML — one self-contained file", "html"),
    ("PDF — matplotlib transcription", "pdf"),
    ("Both", "both"),
)

#: (min, max) the figure-cap spin box allows.
FIGURE_CAP_RANGE = (0, 200)

#: Colour of the overall-status line, per :attr:`spacr.report.Report.status`.
_STATUS_COLOURS = {
    "complete": "success",
    "partial": "error",
    "failed": "error",
    "unknown": "warning",
    "empty": "warning",
}


class ReportScreen(QWidget):
    """Build a shareable HTML/PDF report from a finished run folder.

    :param parent: Qt parent.
    :param threaded: run scanning and generation on a worker thread (the
        default). Tests pass ``False`` for deterministic, synchronous
        behaviour.
    :ivar last_error: text of the most recent failure, ``""`` when the last
        operation succeeded. Errors are only ever reported here and in the
        inline status label — never in a modal dialog.
    """

    #: emitted with the folder path whenever a scan completes
    folder_scanned = Signal(str)
    #: emitted with the list of written paths after a successful generate
    report_written = Signal(list)
    #: emitted after every job settles (ok or not)
    job_finished = Signal(bool)
    #: private. Re-emitted from ``PipelineWorker.finished`` purely to hop
    #: back onto the GUI thread — see :meth:`_run_job`.
    _job_settled = Signal(bool)

    def __init__(self, parent=None, threaded: bool = True):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._src: str = ""
        self._report: Optional[rep.Report] = None
        self._written: List[str] = []
        self._busy = False
        # Ownership list for in-flight (QThread, worker) pairs — a QThread
        # collected while still running takes the process down with it.
        self._jobs: List[tuple] = []
        self._pending: List[Tuple[Dict[str, Any], Callable[[Any], None]]] = []
        self._thread = None
        self._worker = None
        self.last_error: str = ""

        self._job_settled.connect(self._on_job_settled)
        self._build_ui()
        self._set_status(
            "Choose a run folder — the plate folder holding measurements/, "
            "qc/ and results/ — then Scan.")
        self._update_controls()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel("Report")
        title.setObjectName("DisplayHeading")
        outer.addWidget(title)

        subtitle = QLabel(
            "One file a collaborator can open without spaCR: what ran and "
            "when, whether it finished, the QC verdict, the figures, the "
            "statistics and the exact settings. The HTML is fully "
            "self-contained — images are embedded and nothing loads from the "
            "network. Read-only: this never writes into the run folder.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        outer.addWidget(Divider())

        # ── Source row ────────────────────────────────────────────────
        src_row = QHBoxLayout()
        src_row.setSpacing(SPACING["sm"])
        self._path_edit = QLineEdit(self)
        self._path_edit.setPlaceholderText("…/plate1  — the run folder")
        self._path_edit.setClearButtonEnabled(True)
        self._path_edit.returnPressed.connect(self.scan)
        self._btn_pick_src = QPushButton("Choose run folder…", self)
        self._btn_pick_src.clicked.connect(self._pick_run_folder)
        self._btn_scan = QPushButton("Scan", self)
        self._btn_scan.clicked.connect(self.scan)
        src_row.addWidget(self._path_edit, 1)
        src_row.addWidget(self._btn_pick_src)
        src_row.addWidget(self._btn_scan)
        outer.addLayout(src_row)

        # ── Overall verdict ───────────────────────────────────────────
        self._verdict = QLabel("", self)
        self._verdict.setWordWrap(True)
        outer.addWidget(self._verdict)

        # ── Section list ──────────────────────────────────────────────
        outer.addWidget(QLabel("Sections found in this folder:", self))
        self._sections = QListWidget(self)
        self._sections.setAlternatingRowColors(True)
        self._sections.setSelectionMode(QListWidget.NoSelection)
        outer.addWidget(self._sections, 1)

        # ── Options ───────────────────────────────────────────────────
        opts = QHBoxLayout()
        opts.setSpacing(SPACING["sm"])
        opts.addWidget(QLabel("Format", self))
        self._format = QComboBox(self)
        for label, key in FORMATS:
            self._format.addItem(label, key)
        opts.addWidget(self._format)
        opts.addWidget(QLabel("Embed at most", self))
        self._figure_cap = QSpinBox(self)
        self._figure_cap.setRange(*FIGURE_CAP_RANGE)
        self._figure_cap.setValue(rep.DEFAULT_MAX_FIGURES)
        self._figure_cap.setSuffix(" figures")
        self._figure_cap.valueChanged.connect(lambda _v: self._update_controls())
        opts.addWidget(self._figure_cap)
        opts.addStretch(1)
        outer.addLayout(opts)

        # ── Output row ────────────────────────────────────────────────
        out_row = QHBoxLayout()
        out_row.setSpacing(SPACING["sm"])
        self._out_edit = QLineEdit(self)
        self._out_edit.setPlaceholderText(
            "…/plate1_report.html  — or a folder to write into")
        self._out_edit.setClearButtonEnabled(True)
        self._btn_pick_out = QPushButton("Choose output…", self)
        self._btn_pick_out.clicked.connect(self._pick_output)
        self._btn_generate = QPushButton("Generate report", self)
        self._btn_generate.setObjectName("PrimaryButton")
        self._btn_generate.clicked.connect(self.generate)
        self._btn_open = QPushButton("Open", self)
        self._btn_open.clicked.connect(self.open_output)
        out_row.addWidget(self._out_edit, 1)
        out_row.addWidget(self._btn_pick_out)
        out_row.addWidget(self._btn_generate)
        out_row.addWidget(self._btn_open)
        outer.addLayout(out_row)

        self._status = QLabel("", self)
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

    # -- state -------------------------------------------------------------

    @property
    def report(self) -> Optional[rep.Report]:
        """The most recently collected :class:`spacr.report.Report`."""
        return self._report

    @property
    def written(self) -> List[str]:
        """Paths written by the last successful generate."""
        return list(self._written)

    def found_sections(self) -> List[str]:
        """Keys of the sections the last scan found."""
        return list(self._report.found_sections) if self._report else []

    def missing_sections(self) -> List[str]:
        """Keys of the sections the last scan did not find."""
        return list(self._report.missing_sections) if self._report else []

    def figure_cap(self) -> int:
        """The figure cap currently selected."""
        return int(self._figure_cap.value())

    def output_format(self) -> str:
        """``"html"``, ``"pdf"`` or ``"both"``."""
        return str(self._format.currentData() or "html")

    def set_source(self, path: str) -> None:
        """Put ``path`` in the source box without scanning."""
        self._path_edit.setText(str(path or ""))
        self._update_controls()

    def set_output(self, path: str) -> None:
        """Put ``path`` in the output box."""
        self._out_edit.setText(str(path or ""))
        self._update_controls()

    def set_format(self, fmt: str) -> None:
        """Select an output format by its ``build_report`` key."""
        index = self._format.findData(str(fmt))
        if index >= 0:
            self._format.setCurrentIndex(index)
        self._update_controls()

    # -- pickers -----------------------------------------------------------

    def _pick_run_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose a run folder", self._path_edit.text().strip()
            or os.path.expanduser("~"))
        if path:
            self._path_edit.setText(path)
            self.scan()

    def _pick_output(self) -> None:
        suggested = self._suggested_output()
        path, _ = QFileDialog.getSaveFileName(
            self, "Write the report to", suggested,
            "HTML (*.html);;PDF (*.pdf);;All files (*)")
        if path:
            self._out_edit.setText(path)
            self._update_controls()

    def _suggested_output(self) -> str:
        """A default output path next to the user's home, never inside ``src``.

        Reports are written where the user chooses; defaulting *into* the run
        folder would quietly add a file to a dataset somebody else may be
        treating as immutable.
        """
        name = os.path.basename(os.path.normpath(self._src or "spacr"))
        suffix = ".pdf" if self.output_format() == "pdf" else ".html"
        return os.path.join(os.path.expanduser("~"), f"{name}_report{suffix}")

    # -- scanning ----------------------------------------------------------

    def scan(self) -> bool:
        """Collect the report for the folder in the source box.

        Nothing is written; this only discovers what a report would contain,
        so the section list can be shown before the user commits.

        :returns: True when the job was started (or, unthreaded, ran).
        """
        raw = self._path_edit.text().strip()
        if not raw:
            self._set_status("No run folder given — choose one first.",
                             error=True)
            return False
        path = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isdir(path):
            self._set_status(f"Not a folder: {path}", error=True)
            self._report = None
            self._sections.clear()
            self._verdict.setText("")
            self._update_controls()
            return False
        self._src = path
        cap = self.figure_cap()
        self._set_status(f"Scanning {path} …")
        return self._run_job(
            lambda: rep.collect_report(path, max_figures=cap),
            self._on_scanned)

    def _on_scanned(self, report: Any) -> None:
        self._report = report if isinstance(report, rep.Report) else None
        self._render_sections()
        if self._report is None:
            self._set_status("Scan produced nothing.", error=True)
            return
        if not self._out_edit.text().strip():
            self._out_edit.setText(self._suggested_output())
        missing = self._report.missing_sections
        message = (f"Scanned {self._report.src}: "
                   f"{len(self._report.found_sections)} section(s) found")
        if missing:
            message += f", {len(missing)} not available"
        message += (f"; {self._report.n_figures_embedded} of "
                    f"{self._report.n_figures_found} figure(s) would be embedded.")
        self._set_status(message)
        self.folder_scanned.emit(str(self._report.src))

    def _render_sections(self) -> None:
        """Fill the section list — found in normal text, missing greyed."""
        self._sections.clear()
        self._verdict.setText("")
        report = self._report
        if report is None:
            return
        colour = PALETTE.get(_STATUS_COLOURS.get(report.status, "warning"),
                             PALETTE["fg_muted"])
        self._verdict.setStyleSheet(f"color: {colour}; font-weight: 600;")
        self._verdict.setText(report.status_detail)
        for section in report.sections:
            if section.status == rep.STATUS_MISSING:
                text = f"{section.title} — not available"
            elif section.status == rep.STATUS_PROBLEM:
                text = f"{section.title} — needs attention"
            else:
                text = section.title
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, section.key)
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            if section.status == rep.STATUS_MISSING:
                item.setForeground(_brush(PALETTE["fg_dim"]))
                item.setToolTip(
                    "This section will still appear in the report, saying "
                    "what was looked for and not found.")
            elif section.status == rep.STATUS_PROBLEM:
                item.setForeground(_brush(PALETTE["error"]))
            self._sections.addItem(item)

    # -- generating --------------------------------------------------------

    def generate(self) -> bool:
        """Write the report to the path in the output box.

        Re-collects rather than reusing the scan, so the file reflects the
        folder as it is now and the figure cap as it is now.

        :returns: True when the job was started (or, unthreaded, ran).
        """
        raw = self._path_edit.text().strip()
        if not raw:
            self._set_status("No run folder given — choose one first.",
                             error=True)
            return False
        src = os.path.abspath(os.path.expanduser(raw))
        if not os.path.isdir(src):
            self._set_status(f"Not a folder: {src}", error=True)
            return False
        out = self._out_edit.text().strip() or self._suggested_output()
        out = os.path.abspath(os.path.expanduser(out))
        fmt = self.output_format()
        cap = self.figure_cap()
        self._set_status(f"Writing the report to {out} …")
        return self._run_job(
            lambda: rep.build_report(src, out, fmt=fmt, max_figures=cap),
            self._on_generated)

    def _on_generated(self, paths: Any) -> None:
        written = [str(p) for p in (paths or [])]
        self._written = written
        if not written:
            self._set_status("Nothing was written.", error=True)
            return
        self._set_status("Wrote " + ", ".join(os.path.basename(p) for p in written)
                         + f" to {os.path.dirname(written[0])}.")
        self.report_written.emit(written)

    def open_output(self) -> bool:
        """Hand the newest written report to the desktop's default opener.

        :returns: True when there was something to open.
        """
        if not self._written:
            self._set_status("Nothing has been generated yet.", error=True)
            return False
        target = self._written[0]
        if not os.path.isfile(target):
            self._set_status(f"{target} is no longer there.", error=True)
            return False
        QDesktopServices.openUrl(QUrl.fromLocalFile(target))
        self._set_status(f"Opened {os.path.basename(target)}.")
        return True

    # -- job plumbing ------------------------------------------------------

    def _run_job(self, fn: Callable[[], Any],
                 on_done: Callable[[Any], None]) -> bool:
        """Run ``fn`` off the GUI thread and hand its result to ``on_done``.

        Mirrors ``PlateViewScreen._run_job`` — one threading idiom for the
        whole Qt layer. ``PipelineWorker.finished`` is emitted *in the
        worker thread*, so it is chained through :attr:`_job_settled`, a
        signal on this widget, which gives Qt a GUI-thread receiver to queue
        the completion onto.

        With ``threaded=False`` the call runs inline and the same signals
        fire, so both paths behave identically from outside.
        """
        if self._busy:
            self._set_status("Still working on the previous request.",
                             error=True)
            return False
        if not self._threaded:
            ok = True
            try:
                on_done(fn())
            except Exception as e:
                self._on_job_error(e)
                ok = False
            self._update_controls()
            self.job_finished.emit(ok)
            return ok

        box: Dict[str, Any] = {}

        def _job(payload: Dict[str, Any]) -> None:
            payload["result"] = fn()

        thread, worker = make_thread(_job, box)
        # Strong references: PySide6 will not keep the worker alive through
        # the started→run connection alone, and a QThread garbage-collected
        # while still running takes the process down with it.
        self._jobs.append((thread, worker))
        self._thread, self._worker = thread, worker
        self._pending.append((box, on_done))
        worker.error.connect(self._on_worker_error_text)
        worker.finished.connect(self._job_settled)
        thread.finished.connect(lambda t=thread: self._retire_job(t))
        self._busy = True
        self._update_controls()
        thread.start()
        return True

    def _on_job_settled(self, ok: bool) -> None:
        """Finish the oldest in-flight job. Always on the GUI thread."""
        self._busy = False
        box, on_done = self._pending.pop(0) if self._pending else ({}, None)
        ok = bool(ok)
        if ok and on_done is not None:
            try:
                on_done(box.get("result"))
            except Exception as e:
                self._on_job_error(e)
                ok = False
        self._update_controls()
        self.job_finished.emit(ok)

    def _retire_job(self, thread) -> None:
        """Release *this* job's refs once its own event loop has exited."""
        self._jobs = [(t, w) for (t, w) in self._jobs if t is not thread]
        if self._thread is thread:
            self._thread = None
            self._worker = None

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return len(self._jobs)

    def is_busy(self) -> bool:
        """True while a scan or a generate is in flight."""
        return self._busy

    def _on_job_error(self, exc: Exception) -> None:
        self._busy = False
        self._set_status(str(exc) or exc.__class__.__name__, error=True)

    def _on_worker_error_text(self, text: str) -> None:
        line = (text or "").strip().splitlines()[-1] if text else "unknown error"
        self._busy = False
        self._set_status(f"Report failed: {line}", error=True)

    # -- chrome ------------------------------------------------------------

    def _set_status(self, text: str, error: bool = False) -> None:
        """Report inline. Deliberately never a QMessageBox — a modal dialog
        would hang a headless run (and did, in MakeMasksScreen)."""
        self.last_error = text if error else ""
        colour = PALETTE["error"] if error else PALETTE["fg_muted"]
        self._status.setStyleSheet(f"color: {colour};")
        self._status.setText(text)

    def status_text(self) -> str:
        """The inline status line, for tests and for the tutorial engine."""
        return self._status.text()

    def _update_controls(self) -> None:
        idle = not self._busy
        has_src = bool(self._path_edit.text().strip())
        self._btn_scan.setEnabled(idle and has_src)
        self._btn_pick_src.setEnabled(idle)
        self._btn_pick_out.setEnabled(idle)
        self._btn_generate.setEnabled(idle and has_src)
        self._btn_open.setEnabled(idle and bool(self._written))
        self._format.setEnabled(idle)
        self._figure_cap.setEnabled(idle)


def _brush(colour: str):
    """A QBrush for a hex colour, imported lazily to keep the header short."""
    from PySide6.QtGui import QBrush, QColor
    return QBrush(QColor(colour))
