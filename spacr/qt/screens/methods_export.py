"""Methods & Results — the two paragraphs, written from the run.

The end of an analysis is a paper, and the part of the paper that is pure
transcription — what software, what version, what parameters, what n, what
came out — is the part everyone writes from memory three months later and
gets subtly wrong. spaCR knows all of it: the run journal has the versions
and the timings, the emitted macro has the modules and the parameters the
user actually chose, the artifact registry has the provenance, the QC modules
have the verdicts, and the hit list has the statistics.

This screen assembles that into a **run digest** and turns it into two
sections: **Methods** and **Results**.

**The model never sees the data.** It sees the digest — numbers already
computed — and writes prose around them. Then every number it wrote is
checked back against the digest, and a draft carrying a figure that is not
there is refused: the panel says which numbers were invented, keeps the
model's text where you can read it, and shows the sections spaCR wrote from
the digest instead. The check is visible in the strip under the tabs, because
"an AI wrote my methods section" is only acceptable if the reader can see the
provenance of every figure in it.

**With no AI configured it still works.** The deterministic renderer writes
both sections from the same digest with no model at all. The panel says which
one produced what you are looking at, and what to install if you want the
other.

**The caveats are not optional.** Which QC flags fired, whether illumination
correction ran, the seed, the ``on_error`` policy and what it dropped, the
classifier's held-out metrics — the digest collects them and the Methods
section has to state every one. A draft that drops one is refused for that
alone.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...methods_export import (build_digest, render_methods, render_prompt,
                               render_results)
from ..job_runner import JobRunner
from ..theme import SPACING, block_surface, register_widget_qss
from .app_screen import ModuleHeader

__all__ = ["APP_KEY", "MethodsExportScreen", "make_methods_export_screen",
           "register"]

#: The app key this screen is registered under.
APP_KEY = "methods_export"

#: Sidebar / tile name.
APP_NAME = "Methods & Results"

#: One-line summary; the tooltip and status tip.
APP_DESCRIPTION = (
    "Draft the methods and results sections from the run, with every number "
    "traced")

#: The paragraph under this app's header, handed to the seam as ``intro``.
APP_INTRO = (
    "Assemble everything the run knows — versions, parameters, counts, QC "
    "verdicts, held-out metrics, statistics — into one run digest, then draft "
    "the Methods and Results sections of a paper from it. An AI writes the "
    "prose if one is configured, but it never sees the data: it sees the "
    "digest, and every number it writes is checked back against it. A draft "
    "with an invented figure, or one that drops a caveat the run recorded, is "
    "refused and spaCR's own version is shown instead.")

#: Why there is no ``spacr-run methods_export``; reaches
#: ``cli.INTERACTIVE_ONLY``.
APP_CLI_NOTE = (
    "Methods & Results is the interactive drafting panel; headless, call "
    "spacr.methods_export.build_digest(...) and then render_methods(digest) "
    "and render_results(digest), which need no AI provider at all.")

#: "Methods & Results" in the nine non-English UI languages, in
#: :data:`spacr.qt.i18n.LANGUAGES` order after English — sv, de, es, zh_CN,
#: pt, hi, ko, is, fr.
APP_TRANSLATIONS = (
    "Metod och resultat",
    "Methoden und Ergebnisse",
    "Métodos y resultados",
    "方法与结果",
    "Métodos e resultados",
    "विधियाँ और परिणाम",
    "방법 및 결과",
    "Aðferðir og niðurstöður",
    "Méthodes et résultats",
)


def _methods_qss(palette: dict, opacity) -> str:
    """QSS for the source panel and the provenance strip."""
    surface = block_surface("surface_alt", palette["theme"], opacity)
    return f"""
QFrame#MethodsExportSources {{
    background: {surface};
    border: 1px solid {palette["border_soft"]};
    border-radius: 8px;
}}
QLabel#MethodsExportProvenance {{
    font-weight: 600;
}}
QLabel#MethodsExportProvenance[problem="true"] {{
    color: {palette["warning"]};
}}
"""


# ``replace=True`` because this module owns the name: a reimport must
# re-register the same block rather than raise and leave the screen unstyled.
register_widget_qss("MethodsExportSources", _methods_qss, replace=True)


class MethodsExportScreen(QWidget):
    """Build the run digest; draft the two sections; show the number check.

    :param parent: Qt parent.
    :param project: pre-fill the project field.
    :param run_dir: pre-fill the run-journal folder field.
    :param results_folder: pre-fill the regression results field.
    :param model_path: pre-fill the classifier checkpoint field.
    :param threaded: ``False`` runs everything inline, so a test drives the
        screen synchronously without the behaviour diverging.
    :ivar last_error: text of the most recent failure, ``""`` when the last
        operation worked.
    """

    #: Emitted with the digest after every build.
    digest_built = Signal(object)
    #: Emitted with the :class:`~spacr.qt.ai.manuscript.ManuscriptDraft`.
    draft_ready = Signal(object)

    #: The four inputs, as ``(attribute, label, tooltip, is_folder)``.
    SOURCES = (
        ("project", "Project",
         "The spaCR project root. Supplies the provenance summary from the "
         "artifact registry and the recorded segmentation QC.", True),
        ("run_dir", "Run folder",
         "A run-journal folder under ~/.spacr/runs. Supplies the package "
         "versions, the timings, the seed and the emitted macro's modules "
         "and parameters.", True),
        ("results", "Regression results",
         "A regression results folder. Supplies the hit statistics.", True),
        ("model", "Classifier",
         "A trained checkpoint. Its model card supplies the held-out "
         "metrics the methods section has to state.", False),
    )

    def __init__(self, parent=None, project: str = "", run_dir: str = "",
                 results_folder: str = "", model_path: str = "",
                 threaded: bool = True):
        super().__init__(parent)
        self._digest: Optional[Dict[str, Any]] = None
        self._draft: Any = None
        self._fields: Dict[str, QLineEdit] = {}
        self._jobs = JobRunner(self, threaded=threaded, app_key=APP_KEY)
        self._jobs.job_failed.connect(self._on_job_failed)
        self.last_error: str = ""

        self._build_ui()
        self._fields["project"].setText(project)
        self._fields["run_dir"].setText(run_dir)
        self._fields["results"].setText(results_folder)
        self._fields["model"].setText(model_path)
        self._set_provenance(
            "Name at least one source, then build the digest.", problem=False)
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "methods_export")

    # -- construction -----------------------------------------------------

    def _build_ui(self) -> None:
        """Source panel, action row, provenance strip, then the tabs."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        header = ModuleHeader(
            APP_NAME,
            description="The two paragraphs of the paper that are pure "
                        "transcription, written from the run rather than"
                        "from memory.",
            instruction="Choose a run folder, then draft and export.",
        )
        self._header = header
        outer.addWidget(header)

        sources = QFrame()
        sources.setObjectName("MethodsExportSources")
        grid = QGridLayout(sources)
        grid.setContentsMargins(SPACING["md"], SPACING["sm"],
                                SPACING["md"], SPACING["sm"])
        grid.setHorizontalSpacing(SPACING["sm"])
        grid.setVerticalSpacing(SPACING["xs"])
        for row, (key, label, tip, is_folder) in enumerate(self.SOURCES):
            caption = QLabel(label)
            caption.setToolTip(tip)
            grid.addWidget(caption, row, 0)
            edit = QLineEdit()
            edit.setToolTip(tip)
            edit.setPlaceholderText("optional")
            grid.addWidget(edit, row, 1)
            button = QPushButton("Browse…")
            button.clicked.connect(
                lambda _checked=False, k=key, folder=is_folder:
                self._on_browse(k, folder))
            grid.addWidget(button, row, 2)
            self._fields[key] = edit
        grid.setColumnStretch(1, 1)
        outer.addWidget(sources)

        actions = QHBoxLayout()
        actions.setSpacing(SPACING["sm"])
        self._build_button = QPushButton("Build digest")
        self._build_button.setToolTip(
            "Read every named source and assemble the structured record the "
            "sections are written from. Nothing is written to disk.")
        self._build_button.clicked.connect(self.build)
        actions.addWidget(self._build_button)

        self._generate_button = QPushButton("Draft with AI")
        self._generate_button.setToolTip(
            "Hand the digest — and only the digest — to the configured AI "
            "provider, then check every number it writes back against it.")
        self._generate_button.setEnabled(False)
        self._generate_button.clicked.connect(self.generate)
        actions.addWidget(self._generate_button)

        actions.addStretch(1)
        self._copy_button = QPushButton("Copy sections")
        self._copy_button.clicked.connect(self._on_copy)
        actions.addWidget(self._copy_button)
        self._export_button = QPushButton("Export Markdown…")
        self._export_button.clicked.connect(self._on_export)
        actions.addWidget(self._export_button)
        outer.addLayout(actions)

        self._provenance = QLabel("")
        self._provenance.setObjectName("MethodsExportProvenance")
        self._provenance.setWordWrap(True)
        outer.addWidget(self._provenance)

        self._tabs = QTabWidget()
        self._methods_view = _reader()
        self._results_view = _reader()
        self._caveats_view = _reader()
        self._digest_view = _reader()
        self._rejected_view = _reader()
        self._tabs.addTab(self._methods_view, "Methods")
        self._tabs.addTab(self._results_view, "Results")
        self._tabs.addTab(self._caveats_view, "Caveats")
        self._tabs.addTab(self._digest_view, "Run digest")
        self._tabs.addTab(self._rejected_view, "Rejected draft")
        self._tabs.setTabToolTip(
            2, "What the run knows about its own limitations. Every one of "
               "these has to appear in the Methods section.")
        self._tabs.setTabToolTip(
            3, "The structured record the sections are written from — the "
               "model's only input.")
        self._tabs.setTabToolTip(
            4, "A draft that failed the number check, kept so you can see "
               "what the model claimed.")
        self._tabs.setTabVisible(4, False)
        outer.addWidget(self._tabs, 1)

    # -- the digest -------------------------------------------------------

    def sources(self) -> Dict[str, str]:
        """The four source paths as the fields currently stand."""
        return {key: edit.text().strip()
                for key, edit in self._fields.items()}

    def build(self) -> None:
        """Assemble the digest from whichever sources were named."""
        chosen = self.sources()
        self.last_error = ""
        self._set_provenance("Reading the run…", problem=False)
        self._jobs.cancel()
        self._jobs.submit(lambda paths=chosen: build_digest(
            project=paths["project"] or None,
            run_dir=paths["run_dir"] or None,
            results_folder=paths["results"] or None,
            model_path=paths["model"] or None,
            title=os.path.basename(paths["project"].rstrip(os.sep)) or ""),
            self._on_digest_ready)

    def digest(self) -> Optional[Dict[str, Any]]:
        """The digest currently built, or ``None``."""
        return self._digest

    def draft(self) -> Any:
        """The draft currently shown, or ``None``."""
        return self._draft

    def _on_digest_ready(self, digest: Optional[Dict[str, Any]]) -> None:
        """Show a freshly built digest and spaCR's own sections."""
        self._digest = digest
        if not digest:                                # pragma: no cover
            self._set_provenance("The digest could not be built.",
                                 problem=True)
            return
        self._digest_view.setPlainText(
            json.dumps(digest, indent=2, default=str))
        self._caveats_view.setPlainText(
            "\n".join(f"• {caveat}"
                      for caveat in digest.get("caveats", ()))
            or "The run recorded no caveats.")
        self._methods_view.setPlainText(render_methods(digest))
        self._results_view.setPlainText(render_results(digest))
        self._generate_button.setEnabled(True)
        self._tabs.setTabVisible(4, False)
        self._draft = None
        notes = digest.get("notes") or []
        message = (
            f"Digest built: {len(digest.get('modules') or ())} module(s), "
            f"{len(digest.get('caveats') or ())} caveat(s). The sections "
            f"below were written by spaCR from it, so every number in them "
            f"is from the run.")
        if notes:
            message += f" {len(notes)} source(s) could not be read."
        self._set_provenance(message, problem=bool(notes))
        self.digest_built.emit(digest)

    # -- the draft --------------------------------------------------------

    def generate(self) -> None:
        """Ask the configured AI provider for the two sections."""
        if not self._digest:
            self._set_provenance("Build the digest first.", problem=True)
            return
        self.last_error = ""
        self._set_provenance("Drafting…", problem=False)
        digest = self._digest
        self._jobs.cancel()
        self._jobs.submit(lambda payload=digest: _generate(payload),
                          self._on_draft_ready)

    def _on_draft_ready(self, draft: Any) -> None:
        """Show a draft and, above all, the verdict on its numbers."""
        if draft is None:                             # pragma: no cover
            self._set_provenance("The draft could not be produced.",
                                 problem=True)
            return
        self._draft = draft
        self._methods_view.setPlainText(draft.methods)
        self._results_view.setPlainText(draft.results)
        self._rejected_view.setPlainText(draft.rejected or "")
        self._tabs.setTabVisible(4, bool(draft.rejected))
        self._set_provenance(self.provenance_message(draft),
                             problem=not draft.ok)
        self.draft_ready.emit(draft)

    def provenance_message(self, draft: Any) -> str:
        """The sentence under the tabs: where the prose came from, and why.

        Split out from the widget so a test can assert on the wording — this
        line is the entire user-facing guarantee, and a vague one would be
        worse than none.
        """
        if draft.ok:
            checked = ((draft.methods_check.checked if draft.methods_check
                        else 0) +
                       (draft.results_check.checked if draft.results_check
                        else 0))
            return (
                f"Drafted by {draft.provider or 'the AI provider'}. "
                f"{checked} number(s) in the text were checked against the "
                f"run digest and every one of them came from it.")
        return "\n".join(str(problem) for problem in draft.problems)

    # -- output -----------------------------------------------------------

    def text(self) -> str:
        """Both sections as they currently stand, ready to paste."""
        methods = self._methods_view.toPlainText().strip()
        results = self._results_view.toPlainText().strip()
        return "\n\n".join(part for part in (methods, results) if part)

    def export(self, path: str) -> str:
        """Write both sections plus the digest to a Markdown file.

        The digest travels WITH the prose, as an appendix. A methods section
        whose provenance is a separate file is a methods section whose
        provenance is lost.

        :param path: where to write.
        :returns: the path written, or ``""`` when there was nothing to write.
        """
        if not self.text():
            self._set_provenance("There is nothing to export yet.",
                                 problem=True)
            return ""
        target = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        parts = [self.text()]
        if self._digest:
            parts.append("")
            parts.append("## Appendix: run digest")
            parts.append("")
            parts.append("```json")
            parts.append(json.dumps(self._digest, indent=2, default=str))
            parts.append("```")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("\n".join(parts) + "\n")
        self._set_provenance(
            f"Written to {os.path.basename(target)}, with the run digest as "
            f"an appendix.", problem=False)
        return target

    def prompt(self) -> str:
        """The exact user message the model would be sent, or ``""``."""
        if not self._digest:
            return ""
        return render_prompt(self._digest)[1]

    # -- slots ------------------------------------------------------------

    def _on_browse(self, key: str,                   # pragma: no cover - modal
                   is_folder: bool) -> None:
        """Ask for a path for one source field."""
        if is_folder:
            chosen = QFileDialog.getExistingDirectory(self, "Choose a folder")
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, "Choose a file")
        if chosen:
            self._fields[key].setText(chosen)

    def _on_copy(self) -> None:
        """Put both sections on the clipboard."""
        from PySide6.QtWidgets import QApplication

        body = self.text()
        if not body:
            self._set_provenance("There is nothing to copy yet.", problem=True)
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(body)
            self._set_provenance("Both sections copied.", problem=False)

    def _on_export(self) -> None:                    # pragma: no cover - modal
        """Ask where to write and export."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export methods and results", "", "Markdown (*.md)")
        if path:
            self.export(path)

    def _on_job_failed(self, message: str) -> None:
        """Report a background failure inline; never a modal."""
        self.last_error = message
        self._set_provenance(f"Could not read the run: {message}",
                             problem=True)

    def _set_provenance(self, text: str, *, problem: bool) -> None:
        """Write the provenance strip and repolish it."""
        self._provenance.setText(text)
        self._provenance.setProperty("problem", "true" if problem else "false")
        style = self._provenance.style()
        if style is not None:
            style.unpolish(self._provenance)
            style.polish(self._provenance)

    # -- lifecycle --------------------------------------------------------

    def is_busy(self) -> bool:
        """True while a digest or a draft is still being produced."""
        return self._jobs.is_busy()

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return self._jobs.active_jobs()

    def closeEvent(self, event) -> None:             # noqa: N802 - Qt override
        """Drain the worker before the widget goes."""
        self._jobs.shutdown()
        super().closeEvent(event)


def _generate(digest: Dict[str, Any]) -> Any:
    """Call the AI half, importing it only when it is actually used.

    Deferred so that opening this screen does not import the provider
    plumbing (and its ``QSettings`` read) for a user who only ever wants the
    deterministic sections.
    """
    from ..ai.manuscript import generate_sections

    return generate_sections(digest)


def _reader() -> QPlainTextEdit:
    """A read-only monospace pane."""
    view = QPlainTextEdit()
    view.setReadOnly(True)
    view.setLineWrapMode(QPlainTextEdit.WidgetWidth)
    return view


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def make_methods_export_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory the registry calls to build this screen."""
    return MethodsExportScreen()


def register() -> bool:
    """Add Methods & Results to the app registry. Idempotent."""
    from ..app import APPS, SECTION_RESULTS, STAGE_ALPHA, register_app

    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(
        APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_RESULTS,
        factory=make_methods_export_screen, stage=STAGE_ALPHA,
        title="Methods & Results", intro=APP_INTRO, cli_note=APP_CLI_NOTE,
        api_module="qt/screens/methods_export",
        translations=APP_TRANSLATIONS)
    return True


register()
