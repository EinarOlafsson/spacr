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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
from ..i18n import tr
from ..theme import SPACING, active_palette
from ..widgets import Divider
from ..widgets.collapsible_splitter import FoldSection
from ..widgets.measurements_example import install_test_data_button

__all__ = ["ReportScreen", "FORMATS", "FIGURE_CAP_RANGE"]


def _has_stopped(thread) -> bool:
    """True when ``thread`` is finished, never started, or already deleted.

    ``None`` counts as stopped. So does a QThread whose C++ half PySide6 has
    already taken away: asking it anything raises ``RuntimeError``, and an
    object that no longer exists is certainly not still running.
    """
    if thread is None:
        return True
    try:
        return not thread.isRunning()
    except RuntimeError:
        return True


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
        """Build the screen and arm its drop zone.

        :param parent: parent widget, or ``None``.
        :param threaded: scan and generate on a worker thread. Set ``False`` in
            tests so ``scan`` finishes before it returns.
        """
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._src: str = ""
        self._report: Optional[rep.Report] = None
        self._written: List[str] = []
        self._archive_problems: List[str] = []
        self._zenodo_record: Optional[Dict[str, Any]] = None
        self._busy = False
        self._jobs: List[tuple] = []
        self._pending: List[Tuple[Dict[str, Any], Callable[[Any], None]]] = []
        self._thread = None
        self._worker = None
        self.last_error: str = ""

        self._job_settled.connect(self._on_job_settled)
        self._build_ui()
        from ..dnd import install_dropzone
        from ..dnd_handlers import get_handler
        install_dropzone(self, get_handler("report"), self)
        self._set_status(
            "Choose a run folder — the plate folder holding measurements/, "
            "qc/ and results/ — then Scan.")
        self._update_controls()


    def _build_ui(self) -> None:
        """Lay out the source row, the section list, the output row and the actions.

        The section list folds by its heading (item 471); folded, the
        heading sits at the bottom of the room the list had, directly above
        the output row.
        """
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
        install_test_data_button(
            self, src_row, self._open_the_example,
            say=lambda message: self._set_status(message, error=True))
        outer.addLayout(src_row)

        self._verdict = QLabel("", self)
        self._verdict.setWordWrap(True)
        outer.addWidget(self._verdict)

        self._sections = QListWidget(self)
        self._sections.setAlternatingRowColors(True)
        self._sections.setSelectionMode(QListWidget.NoSelection)
        self._sections_section = FoldSection(
            self._sections, "Sections found in this folder:",
            persist_key="report/Sections")
        outer.addWidget(self._sections_section, 1)

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
        out_row.addWidget(self._build_archive_button())
        out_row.addWidget(self._build_zenodo_button())
        outer.addLayout(out_row)

        self._status = QLabel("", self)
        self._status.setWordWrap(True)
        outer.addWidget(self._status)


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
        """Put ``path`` in the source box without scanning.

        :param path: run folder shown in the source box; None or an empty value
            clears it.
        """
        self._path_edit.setText(str(path or ""))
        self._update_controls()

    def _open_the_example(self, folder, _database) -> None:
        """Put the example plate folder in the source box and scan it.

        :param folder: the example plate folder, which is the run folder a
            report is built from.
        :param _database: its measurements database; the scan finds it.
        """
        self.set_source(str(folder))
        self.scan()

    def set_output(self, path: str) -> None:
        """Put ``path`` in the output box.

        :param path: output location shown in the output box; None or an empty
            value clears it.
        """
        self._out_edit.setText(str(path or ""))
        self._update_controls()

    def set_format(self, fmt: str) -> None:
        """Select an output format by its ``build_report`` key.

        :param fmt: a key from :data:`FORMATS` (``"html"``, ``"pdf"`` or
            ``"both"``); an unknown key leaves the current choice unchanged.
        """
        index = self._format.findData(str(fmt))
        if index >= 0:
            self._format.setCurrentIndex(index)
        self._update_controls()


    def _pick_run_folder(self) -> None:
        """Ask for a run folder and scan it straight away."""
        path = QFileDialog.getExistingDirectory(
            self, "Choose a run folder", self._path_edit.text().strip()
            or os.path.expanduser("~"))
        if path:
            self._path_edit.setText(path)
            self.scan()

    def _pick_output(self) -> None:
        """Ask where the report should be written.

        The suggestion is derived from the scanned folder, so the usual answer
        is already in the box.
        """
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
        """Show what a finished scan found, and suggest where to write it.

        Sections that are not available are counted rather than hidden: a report
        missing half its sections is a fact about the run, and the number is how
        the user finds out before generating.

        :param report: the scan result, or anything else -- which is treated as
            a scan that produced nothing.
        """
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
        palette = active_palette()
        colour = palette.get(_STATUS_COLOURS.get(report.status, "warning"),
                             palette["fg_muted"])
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
                item.setForeground(_brush(active_palette()["fg_dim"]))
                item.setToolTip(
                    "This section will still appear in the report, saying "
                    "what was looked for and not found.")
            elif section.status == rep.STATUS_PROBLEM:
                item.setForeground(_brush(active_palette()["error"]))
            self._sections.addItem(item)


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
        """Report which files were written.

        :param paths: the written files; an empty list is reported as a failure
            rather than as a silent success.
        """
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


    def _build_archive_button(self) -> QPushButton:
        """The Archive package button: metadata and files for IDR or BioImage Archive.

        Opens :meth:`_archive_dialog`. An alpha feature, registered as
        ``ReportArchivePackage`` in :data:`spacr.settings.ALPHA_FEATURES`.

        :returns: the button.
        """
        from ..preferences import _apply_alpha_widgets

        button = QPushButton(tr("Archive package…"), self)
        button.setObjectName("ReportArchivePackage")
        button.setToolTip(tr(
            "Assemble a submission package for the Image Data Resource or "
            "the BioImage Archive: MIHCSME and REMBI metadata taken from "
            "the settings spaCR saved, the images and a plate map, plus a "
            "short form for what spaCR cannot know, with the IDR study and "
            "library files, a BioStudies study and file list, and MD5 "
            "checksums. Nothing is uploaded and the run folder is not "
            "written to. Default not made."))
        button.clicked.connect(self._on_archive_package)
        self._btn_archive = button
        _apply_alpha_widgets(button)
        return button

    def _archive_labels(self) -> Dict[str, str]:
        """The caption of each archive form field, keyed as the form is."""
        return {
            "title": tr("Title"),
            "description": tr("Description"),
            "authors": tr("Authors (Last First; …)"),
            "email": tr("Contact email"),
            "affiliation": tr("Affiliation"),
            "organism": tr("Organism (; between several)"),
            "cell_line": tr("Cell line"),
            "technology": tr("Screen technology"),
            "screen_type": tr("Screen type"),
            "imaging_method": tr("Imaging method"),
            "microscope": tr("Microscope"),
            "growth_protocol": tr("Growth protocol"),
            "treatment_protocol": tr("Treatment protocol"),
            "sample_preparation": tr("Sample preparation"),
            "keywords": tr("Keywords (; between several)"),
            "license": tr("License"),
            "release_date": tr("Public release date"),
            "plate_map": tr("Plate map (optional)"),
        }

    def _archive_form_rows(self, dialog, form, src: str) -> Dict[str, QLineEdit]:
        """Add one line per archive form field to ``form``, filled from ``src``.

        :returns: the line edits, keyed as the form is.
        """
        defaults = rep._archive_form_defaults(src)
        labels = self._archive_labels()
        fields: Dict[str, QLineEdit] = {}
        for key, required in rep._ARCHIVE_FORM_FIELDS:
            edit = QLineEdit(defaults.get(key, ""), dialog)
            edit.setObjectName(f"ArchiveField_{key}")
            caption = labels[key] + (" *" if required else "")
            form.addRow(caption, edit)
            fields[key] = edit
        return fields

    def _archive_dialog(self):
        """Build the archive form, filled from the run folder's settings.

        :returns: the dialog, not yet shown, or ``None`` without a folder.
        """
        from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                                       QFormLayout, QPlainTextEdit)

        raw = self._path_edit.text().strip()
        src = os.path.abspath(os.path.expanduser(raw)) if raw else ""
        if not src or not os.path.isdir(src):
            self._set_status(tr("Choose a run folder first."), error=True)
            return None
        dialog = QDialog(self)
        dialog.setObjectName("ReportArchiveDialog")
        dialog.setWindowTitle(tr("Archive package"))
        form = QFormLayout(dialog)
        fields = self._archive_form_rows(dialog, form, src)
        out = QLineEdit(os.path.dirname(src.rstrip(os.sep)), dialog)
        out.setObjectName("ArchiveOutput")
        form.addRow(tr("Write the package into"), out)
        copy = QCheckBox(tr("Copy the images into the package"), dialog)
        copy.setObjectName("ArchiveCopyImages")
        form.addRow("", copy)
        more = QPlainTextEdit(dialog)
        more.setObjectName("ArchiveScreenSources")
        more.setPlaceholderText(tr(
            "Optional: further run folders, one per line, each becoming "
            "another screen of the same study."))
        more.setFixedHeight(72)
        form.addRow(tr("Further screens"), more)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        dialog.accepted.connect(lambda: self._write_archive(
            src, out.text().strip(),
            {k: e.text() for k, e in fields.items()}, copy.isChecked(),
            self._archive_screen_sources(more.toPlainText())))
        return dialog

    @staticmethod
    def _archive_screen_sources(text: str) -> List[str]:
        """The further run folders typed in the form, one per line, expanded."""
        return [os.path.abspath(os.path.expanduser(line.strip()))
                for line in text.splitlines() if line.strip()]

    def _on_archive_package(self) -> None:
        """Show the archive form for the folder in the source box."""
        dialog = self._archive_dialog()
        if dialog is not None:
            dialog.setAttribute(Qt.WA_DeleteOnClose, True)
            dialog.open()

    def _write_archive(self, src: str, out: str, form: Dict[str, str],
                      copy_images: bool = False,
                      screens: Sequence[str] = ()) -> bool:
        """Write and validate an archive package off the GUI thread.

        With further run folders in ``screens`` the package is one study
        with a screen per run, ``src`` being screen A.

        :param src: the run folder.
        :param out: the folder the package folder is made in.
        :param form: the form values.
        :param copy_images: also copy the images into the package.
        :param screens: further run folders, each another screen.
        :returns: True when the job was started (or, unthreaded, ran).
        """
        target = out or os.path.dirname(src.rstrip(os.sep))
        self._set_status(tr("Writing the archive package…"))

        def _job():
            """Write the package, then check it against the templates."""
            if screens:
                pkg = rep._write_archive_study([src, *screens], target, form,
                                               copy_images=copy_images)
                return pkg, rep._validate_archive_study(pkg)
            pkg = rep._write_archive_package(src, target, form,
                                             copy_images=copy_images)
            return pkg, rep._validate_archive_package(pkg)

        return self._run_job(_job, self._on_archive_written)

    def _on_archive_written(self, result: Any) -> None:
        """Say where the package went and whether it passed its checks."""
        pkg, problems = result
        self._archive_problems = list(problems)
        if problems:
            self._set_status(
                tr("Wrote {path}, but it does not pass: {problems}").format(
                    path=pkg, problems="; ".join(problems[:4])), error=True)
            return
        self._set_status(tr(
            "Wrote {path}. It passes the IDR, BioStudies and MIHCSME "
            "checks; nothing was uploaded.").format(path=pkg))

    def _build_zenodo_button(self) -> QPushButton:
        """The Deposit on Zenodo button: the run as a Zenodo deposit with a DOI.

        Opens :meth:`_zenodo_dialog`. An alpha feature, registered as
        ``ReportZenodoDeposit`` in :data:`spacr.settings.ALPHA_FEATURES`.

        :returns: the button.
        """
        from ..preferences import _apply_alpha_widgets

        button = QPushButton(tr("Deposit on Zenodo…"), self)
        button.setObjectName("ReportZenodoDeposit")
        button.setToolTip(tr(
            "Deposit this run on Zenodo so the analysis gets a citable DOI: "
            "the archive package, settings, run journal, report, result "
            "tables and, if asked, the masks, with the archive form as its "
            "metadata. Uses your own Zenodo token, kept in the system "
            "keyring or a private file. The sandbox, for trying it out, is "
            "on until you turn it off; a draft is left to publish on Zenodo "
            "unless you publish here. Default not deposited."))
        button.clicked.connect(self._on_zenodo_deposit)
        self._btn_zenodo = button
        _apply_alpha_widgets(button)
        return button

    def _zenodo_dialog(self):
        """Build the Zenodo form: the archive fields, the token and options.

        :returns: the dialog, not yet shown, or ``None`` without a folder.
        """
        from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                                       QFormLayout)

        raw = self._path_edit.text().strip()
        src = os.path.abspath(os.path.expanduser(raw)) if raw else ""
        if not src or not os.path.isdir(src):
            self._set_status(tr("Choose a run folder first."), error=True)
            return None
        dialog = QDialog(self)
        dialog.setObjectName("ReportZenodoDialog")
        dialog.setWindowTitle(tr("Deposit on Zenodo"))
        form = QFormLayout(dialog)
        fields = self._archive_form_rows(dialog, form, src)
        out = QLineEdit(os.path.dirname(src.rstrip(os.sep)), dialog)
        out.setObjectName("ZenodoOutput")
        form.addRow(tr("Stage the files in"), out)
        sandbox = QCheckBox(
            tr("Use the Zenodo sandbox (a test deposit, no real DOI)"), dialog)
        sandbox.setObjectName("ZenodoSandbox")
        sandbox.setChecked(True)
        form.addRow("", sandbox)
        token = QLineEdit(dialog)
        token.setObjectName("ZenodoToken")
        token.setEchoMode(QLineEdit.Password)

        def _placeholder() -> None:
            """Say whether a token is already kept for this Zenodo."""
            kept = bool(rep._load_zenodo_token(sandbox.isChecked()))
            token.setPlaceholderText(
                tr("A token is kept; type one to replace it") if kept else
                tr("Personal access token with deposit:write"))

        _placeholder()
        sandbox.toggled.connect(lambda _on: _placeholder())
        form.addRow(tr("Zenodo token"), token)
        remember = QCheckBox(tr("Remember the token"), dialog)
        remember.setObjectName("ZenodoRememberToken")
        remember.setChecked(True)
        form.addRow("", remember)
        masks = QCheckBox(tr("Include the masks"), dialog)
        masks.setObjectName("ZenodoIncludeMasks")
        form.addRow("", masks)
        publish = QCheckBox(
            tr("Publish now (the DOI becomes permanent)"), dialog)
        publish.setObjectName("ZenodoPublish")
        form.addRow("", publish)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        dialog.accepted.connect(lambda: self._deposit_zenodo(
            src, out.text().strip(),
            {k: e.text() for k, e in fields.items()}, token=token.text(),
            sandbox=sandbox.isChecked(), remember=remember.isChecked(),
            include_masks=masks.isChecked(), publish=publish.isChecked()))
        return dialog

    def _on_zenodo_deposit(self) -> None:
        """Show the Zenodo form for the folder in the source box."""
        dialog = self._zenodo_dialog()
        if dialog is not None:
            dialog.setAttribute(Qt.WA_DeleteOnClose, True)
            dialog.open()

    def _deposit_zenodo(self, src: str, out: str, form: Dict[str, str], *,
                        token: str = "", sandbox: bool = True,
                        remember: bool = True, include_masks: bool = False,
                        publish: bool = False) -> bool:
        """Deposit the run on Zenodo off the GUI thread.

        A typed token is remembered first when ``remember`` is set; without
        one the kept token is used.

        :param src: the run folder.
        :param out: the folder the staged files go in.
        :param form: the archive form values.
        :param token: the typed token, or empty for the kept one.
        :param sandbox: deposit on the Zenodo sandbox.
        :param remember: keep the typed token for next time.
        :param include_masks: also deposit the masks.
        :param publish: publish, making the DOI permanent.
        :returns: True when the job was started (or, unthreaded, ran).
        """
        token = str(token or "").strip()
        if token and remember:
            rep._store_zenodo_token(token, sandbox)
        token = token or rep._load_zenodo_token(sandbox)
        if not token:
            self._set_status(tr("A Zenodo token is needed."), error=True)
            return False
        target = out or os.path.dirname(src.rstrip(os.sep))
        self._set_status(tr("Depositing on Zenodo…"))

        def _job():
            """Stage and upload the deposit; a failure comes back as text."""
            try:
                return rep._zenodo_archive_run(
                    src, target, form, token=token, sandbox=sandbox,
                    publish=publish, include_masks=include_masks)
            except (ValueError, RuntimeError, OSError) as exc:
                return str(exc)

        return self._run_job(_job, self._on_zenodo_done)

    def _on_zenodo_done(self, result: Any) -> None:
        """Say where the deposit is and its DOI, or why it failed."""
        self._zenodo_record = result if isinstance(result, dict) else None
        if not isinstance(result, dict):
            self._set_status(tr("The Zenodo deposit failed: {error}").format(
                error=result), error=True)
            return
        text = (tr("Published {count} files on Zenodo: DOI {doi}, {url}")
                if result["published"] else
                tr("Deposited {count} files as a Zenodo draft at {url}; "
                   "its reserved DOI is {doi}. Publish it there."))
        self._set_status(text.format(count=len(result["files"]),
                                     doi=result["doi"], url=result["url"]))

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
            """Call the wrapped function, stashing its result in the payload."""
            payload["result"] = fn()

        thread, worker = make_thread(_job, box)
        self._jobs.append((thread, worker))
        self._thread, self._worker = thread, worker
        self._pending.append((box, on_done))
        worker.error.connect(self._on_worker_error_text)
        worker.finished.connect(self._job_settled)
        thread.finished.connect(self._on_thread_finished)
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

    def _on_thread_finished(self) -> None:
        """Release the refs of every job whose event loop has exited.

        A sweep rather than "retire the thread that sent this", because the
        sender is exactly what may already be gone: ``make_thread`` queues
        ``thread.deleteLater`` off the same signal, and by the time this runs
        on the GUI thread the QThread's C++ half can be destroyed. Asking a
        destroyed wrapper anything raises ``RuntimeError``, so a pair that
        raises is treated as finished — its C++ object is gone, which is the
        strongest possible evidence that holding a reference to it is no
        longer keeping anything alive.
        """
        self._jobs = [(t, w) for (t, w) in self._jobs if not _has_stopped(t)]
        if _has_stopped(self._thread):
            self._thread = None
            self._worker = None

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return len(self._jobs)

    def is_busy(self) -> bool:
        """True while a scan or a generate is in flight."""
        return self._busy

    def _on_job_error(self, exc: Exception) -> None:
        """Clear the busy flag and report a failed job.

        :param exc: the exception raised by the worker; its class name is used
            when it carries no message.
        """
        self._busy = False
        self._set_status(str(exc) or exc.__class__.__name__, error=True)

    def _on_worker_error_text(self, text: str) -> None:
        """Clear the busy flag and report a worker failure given as text.

        :param text: the worker's error output; only its last line is shown,
            which for a traceback is the exception itself.
        """
        line = (text or "").strip().splitlines()[-1] if text else "unknown error"
        self._busy = False
        self._set_status(f"Report failed: {line}", error=True)


    def _set_status(self, text: str, error: bool = False) -> None:
        """Report inline. Deliberately never a QMessageBox — a modal dialog
        would hang a headless run (and did, in MakeMasksScreen)."""
        self.last_error = text if error else ""
        palette = active_palette()
        colour = palette["error"] if error else palette["fg_muted"]
        self._status.setStyleSheet(f"color: {colour};")
        self._status.setText(text)

    def status_text(self) -> str:
        """The inline status line, for tests and for the tutorial engine."""
        return self._status.text()

    def _update_controls(self) -> None:
        """Enable the actions to match the source and the run state.

        Open is the exception: it needs a written report rather than a source,
        since it opens what was produced rather than what would be.
        """
        idle = not self._busy
        has_src = bool(self._path_edit.text().strip())
        self._btn_scan.setEnabled(idle and has_src)
        self._btn_pick_src.setEnabled(idle)
        self._btn_pick_out.setEnabled(idle)
        self._btn_generate.setEnabled(idle and has_src)
        self._btn_open.setEnabled(idle and bool(self._written))
        self._btn_archive.setEnabled(idle and has_src)
        self._btn_zenodo.setEnabled(idle and has_src)
        self._format.setEnabled(idle)
        self._figure_cap.setEnabled(idle)


def _brush(colour: str):
    """A QBrush for a hex colour, imported lazily to keep the header short."""
    from PySide6.QtGui import QBrush, QColor
    return QBrush(QColor(colour))
