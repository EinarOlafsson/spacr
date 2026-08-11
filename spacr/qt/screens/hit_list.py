"""Hit List — the deliverable at the end of a screen.

A regression run leaves a folder of plots and four CSVs, and none of them is
the thing the experiment was for. ``results_significant.csv`` is
``p <= 0.05`` with no multiple-testing correction, no gene names, and no
indication of whether a gene's own guides agree with each other. What the
user wants is one table they can sort, narrow and send to a collaborator.

This screen is that table. Point it at a results folder and it builds the
list through :mod:`spacr.hits`, which does the work that makes a hit list
interpretable rather than merely present:

* the **effect size** with its standard error and 95% interval;
* a **q-value** across the genes actually tested, so a 0.05 on two thousand
  genes stops meaning what it does not mean;
* **gRNA agreement** — how many of the gene's own guides push the same way.
  A gene called by one guide of six is the commonest way a pooled screen
  produces a confident artefact, and it was invisible in every table spaCR
  wrote before this one;
* the **metadata join**, collapsed to one row per gene before it is joined.

The filters across the top are the ones a user actually applies: FDR,
minimum effect, minimum guide agreement, minimum guide count, direction,
controls in or out, and a free-text search over the annotation. They compose,
they are recorded on the list, and they travel into the export — so the CSV a
collaborator receives says which filters produced it rather than being an
anonymous subset.

Three exports, because the three uses are different: **CSV** to re-analyse,
**Markdown** to paste into an email or an issue, **HTML** to open on a
machine that has never heard of spaCR. The HTML is self-contained — no
stylesheet, no script, no network — which is what makes it safe to send.

Building the list reads several CSVs and joins them, so it runs through
:class:`spacr.qt.job_runner.JobRunner`, off the GUI thread.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...hits import FLAG_MEANING, HitList, build_hit_list
from ..job_runner import JobRunner
from ..theme import (SPACING, block_surface, mark_surface,
                     register_widget_qss)
from .app_screen import ModuleHeader

__all__ = ["APP_KEY", "HitListScreen", "make_hit_list_screen", "register"]

#: The app key this screen is registered under. Load-bearing: saved user
#: state, the command palette and the sidebar all key off it.
APP_KEY = "hit_list"

#: Sidebar / tile name.
APP_NAME = "Hit List"

#: One-line summary; the tooltip and status tip.
APP_DESCRIPTION = (
    "Ranked, annotated, filterable hits with effect size, FDR and gRNA "
    "agreement")

#: The paragraph under this app's header, handed to the seam as ``intro``.
APP_INTRO = (
    "The deliverable at the end of a screen: one row per gene, ranked, with "
    "the effect size and its 95% interval, a Benjamini-Hochberg q-value over "
    "the genes actually tested, how many of the gene's own guides agree in "
    "sign, and the curated annotation joined on. Filter by FDR, effect, guide "
    "agreement, direction or free text, then export the exact list you are "
    "looking at as CSV, Markdown or a self-contained HTML page you can send "
    "to a collaborator.")

#: Why there is no ``spacr-run hit_list``; reaches
#: ``spacr.cli.INTERACTIVE_ONLY``, which prints it instead of "unknown
#: module".
APP_CLI_NOTE = (
    "Hit List is the interactive view of a regression's ranked hits; "
    "headless, call spacr.hits.build_hit_list(results_folder, "
    "metadata_files=[...]) and then .filter(...).write_csv(path) for exactly "
    "the same table.")

#: "Hit List" in the nine non-English UI languages, in
#: :data:`spacr.qt.i18n.LANGUAGES` order after English — sv, de, es, zh_CN,
#: pt, hi, ko, is, fr.
APP_TRANSLATIONS = (
    "Träfflista",
    "Trefferliste",
    "Lista de aciertos",
    "命中列表",
    "Lista de acertos",
    "हिट सूची",
    "히트 목록",
    "Niðurstöðulisti",
    "Liste des résultats",
)
from ..widgets.toggle import Toggle

#: The table's columns, so the drawing code and the tests cannot disagree.
COLUMNS = ("#", "Gene", "Name", "Effect", "95% CI", "p", "q", "Guides",
           "Agree", "Cond.", "Flags")


def _hit_list_qss(palette: dict, opacity) -> str:
    """QSS for the filter bar and the summary strip.

    The table itself is deliberately unstyled: the shipped
    ``QHeaderView::section`` chips and ``::item:hover`` accent already carry
    the page opacity, and a screen that restyles them is a screen that stops
    following the theme.
    """
    surface = block_surface("surface_alt", palette["theme"], opacity)
    return f"""
QFrame#HitListFilters {{
    background: {surface};
    border: 1px solid {palette["border_soft"]};
    border-radius: 8px;
}}
QLabel#HitListSummary {{
    font-weight: 600;
}}
QLabel#HitListSummary[problem="true"] {{
    color: {palette["warning"]};
}}
"""


# ``replace=True`` because this module owns the name: a reimport (a test that
# reloads it, a plugin that pulls it in twice) must re-register the same block
# rather than raise on the duplicate and leave the screen unstyled.
register_widget_qss("HitListFilters", _hit_list_qss, replace=True)


class HitListScreen(QWidget):
    """Build, filter and export the hit list of one regression run.

    :param parent: Qt parent.
    :param folder: open straight onto this results folder.
    :param metadata_files: annotation CSVs to join.
    :param regression_type: the backend, when the caller knows it. Only
        changes how the list is ranked — the penalised backends have no
        p-value and rank by bootstrap selection frequency instead.
    :param threaded: ``False`` builds the list inline, so a test drives the
        screen synchronously without the behaviour diverging.
    :ivar last_error: text of the most recent failure, ``""`` when the last
        operation worked. Failures land here and in the summary strip —
        never in a modal dialog, which hangs a headless run.
    """

    #: Emitted with the full :class:`~spacr.hits.HitList` after every build.
    hits_loaded = Signal(object)
    #: Emitted with the filtered list every time the filters change.
    hits_filtered = Signal(object)

    def __init__(self, parent=None, folder: str = "",
                 metadata_files: Sequence[str] = (),
                 regression_type: str = "", threaded: bool = True):
        super().__init__(parent)
        self._all: Optional[HitList] = None
        self._shown: Optional[HitList] = None
        self._metadata_files: List[str] = [str(p) for p in metadata_files]
        self._regression_type = str(regression_type)
        self._jobs = JobRunner(self, threaded=threaded, app_key=APP_KEY)
        self._jobs.job_failed.connect(self._on_job_failed)
        self.last_error: str = ""

        self._build_ui()
        if folder:
            self.load_folder(folder)
        else:
            self._set_summary(
                "Choose a regression results folder — the one holding "
                "results_gene.csv.", problem=False)
        # Drop anywhere on this screen: the path is resolved through spaCR's
        # project layout, so the plate folder finds what this screen reads.
        from ..dnd import install_for
        install_for(self, "hit_list")

    # -- construction -----------------------------------------------------

    def _build_ui(self) -> None:
        """Picker, filter bar, summary strip, then the table."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        header = ModuleHeader(
            APP_NAME,
            description="One row per gene: effect size, FDR, how many of its "
                        "guides agree, and the annotation.",
            instruction="Point it at a regression results folder, then "
                        "filter and annotate.",
        )
        self._header = header
        outer.addWidget(header)

        picker = QHBoxLayout()
        picker.setSpacing(SPACING["sm"])
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText(
            "Regression results folder (results/<score>/<type>)")
        self._folder_edit.returnPressed.connect(self._on_folder_entered)
        picker.addWidget(QLabel("Results"))
        picker.addWidget(self._folder_edit, 1)
        self._browse_button = QPushButton("Browse…")
        self._browse_button.clicked.connect(self._on_browse)
        picker.addWidget(self._browse_button)
        self._metadata_button = QPushButton("Metadata…")
        self._metadata_button.setToolTip(
            "Curated gene annotation CSVs. Each is collapsed to one row per "
            "gene before it is joined, so a file with one row per transcript "
            "cannot multiply a hit.")
        self._metadata_button.clicked.connect(self._on_pick_metadata)
        picker.addWidget(self._metadata_button)
        outer.addLayout(picker)

        self._filters = QFrame()
        self._filters.setObjectName("HitListFilters")
        row = QHBoxLayout(self._filters)
        row.setContentsMargins(SPACING["md"], SPACING["sm"],
                               SPACING["md"], SPACING["sm"])
        row.setSpacing(SPACING["md"])

        self._q_spin = QDoubleSpinBox()
        self._q_spin.setRange(0.0, 1.0)
        self._q_spin.setDecimals(3)
        self._q_spin.setSingleStep(0.01)
        # Opens at 1.0 — the whole ranked list — rather than at the FDR. The
        # ranking is the deliverable; the cut is the user's decision, and a
        # screen that silently opens pre-filtered hides both the controls
        # (whose position IS the QC) and the near-misses a reader wants to
        # see. The summary strip still reports how many clear the FDR.
        self._q_spin.setValue(1.0)
        self._q_spin.setToolTip(
            "Benjamini-Hochberg FDR ceiling. 1.0, the default, shows every "
            "gene in rank order; lower it to cut the list.")
        row.addWidget(QLabel("Max q"))
        row.addWidget(self._q_spin)

        self._effect_spin = QDoubleSpinBox()
        self._effect_spin.setRange(0.0, 1e6)
        self._effect_spin.setDecimals(3)
        self._effect_spin.setSingleStep(0.1)
        self._effect_spin.setToolTip(
            "Minimum absolute coefficient — the effect size, in the units of "
            "the dependent variable.")
        row.addWidget(QLabel("Min |effect|"))
        row.addWidget(self._effect_spin)

        self._agreement_spin = QDoubleSpinBox()
        self._agreement_spin.setRange(0.0, 1.0)
        self._agreement_spin.setDecimals(2)
        self._agreement_spin.setSingleStep(0.1)
        self._agreement_spin.setToolTip(
            "Fraction of the gene's guides that push the same way as the "
            "gene. A gene called by one guide of six agrees 0.17.")
        row.addWidget(QLabel("Min agreement"))
        row.addWidget(self._agreement_spin)

        self._guides_spin = QSpinBox()
        self._guides_spin.setRange(0, 100)
        self._guides_spin.setToolTip(
            "Minimum number of guides the per-gRNA table holds for the gene.")
        row.addWidget(QLabel("Min guides"))
        row.addWidget(self._guides_spin)

        self._direction = QComboBox()
        self._direction.addItems(["any", "up", "down"])
        self._direction.setToolTip("Sign of the effect.")
        row.addWidget(QLabel("Direction"))
        row.addWidget(self._direction)

        self._drop_controls = Toggle("Hide controls")
        self._drop_controls.setToolTip(
            "Controls are listed by default: a screen whose positive control "
            "is not near the top has a problem, and that is only visible if "
            "it is in the list.")
        row.addWidget(self._drop_controls)

        self._query = QLineEdit()
        self._query.setPlaceholderText("Search gene, name or annotation")
        self._query.setClearButtonEnabled(True)
        row.addWidget(self._query, 1)
        outer.addWidget(self._filters)

        for widget, signal in (
                (self._q_spin, "valueChanged"),
                (self._effect_spin, "valueChanged"),
                (self._agreement_spin, "valueChanged"),
                (self._guides_spin, "valueChanged"),
                (self._direction, "currentIndexChanged"),
                (self._drop_controls, "toggled"),
                (self._query, "textChanged")):
            getattr(widget, signal).connect(self._on_filters_changed)

        strip = QHBoxLayout()
        strip.setSpacing(SPACING["sm"])
        self._summary = QLabel("")
        self._summary.setObjectName("HitListSummary")
        self._summary.setWordWrap(True)
        strip.addWidget(self._summary, 1)
        for label, tip, slot in (
                ("Export CSV…", "The exact table above, as CSV.",
                 "_on_export_csv"),
                ("Export Markdown…",
                 "The top rows as a Markdown table with the flag legend — "
                 "for an email or an issue.", "_on_export_markdown"),
                ("Export HTML…",
                 "A self-contained page with no stylesheet, script or "
                 "network access, safe to send to a collaborator.",
                 "_on_export_html")):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(getattr(self, slot))
            strip.addWidget(button)
        outer.addLayout(strip)

        self._table = QTreeWidget()
        self._table.setColumnCount(len(COLUMNS))
        self._table.setHeaderLabels(list(COLUMNS))
        self._table.setRootIsDecorated(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(False)
        header = self._table.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        # The hit list IS the page below the filter bar. The bar is the
        # only panel on this screen and the tree does not sit on it.
        mark_surface(self._table)
        outer.addWidget(self._table, 1)

        self._legend = QLabel("")
        self._legend.setObjectName("Muted")
        self._legend.setWordWrap(True)
        outer.addWidget(self._legend)

    # -- loading ----------------------------------------------------------

    def load_folder(self, folder: str) -> None:
        """Build the hit list for ``folder``, off the GUI thread."""
        folder = str(folder or "").strip()
        self.last_error = ""
        self._folder_edit.setText(folder)
        if not folder:
            self._set_summary("Choose a regression results folder.",
                              problem=False)
            return
        self._set_summary(f"Reading {os.path.basename(folder) or folder}…",
                          problem=False)
        metadata = list(self._metadata_files)
        backend = self._regression_type
        self._jobs.cancel()
        self._jobs.submit(
            lambda root=folder: build_hit_list(
                root, metadata_files=metadata, regression_type=backend),
            self._on_hits_ready)

    def set_metadata_files(self, paths: Sequence[str]) -> None:
        """Replace the annotation files and rebuild if a folder is loaded."""
        self._metadata_files = [str(p) for p in paths]
        if self._folder_edit.text().strip():
            self.load_folder(self._folder_edit.text())

    def metadata_files(self) -> List[str]:
        """The annotation files currently joined."""
        return list(self._metadata_files)

    def hits(self) -> Optional[HitList]:
        """The unfiltered list, or ``None`` before anything is loaded."""
        return self._all

    def filtered(self) -> Optional[HitList]:
        """The list as the filters currently narrow it."""
        return self._shown

    def _on_hits_ready(self, hit_list: Optional[HitList]) -> None:
        """Take a freshly built list. Runs on the GUI thread."""
        self._all = hit_list
        if hit_list is None:                          # pragma: no cover
            self._set_summary("The hit list could not be built.", problem=True)
            return
        self.hits_loaded.emit(hit_list)
        self._apply_filters()

    # -- filtering --------------------------------------------------------

    def current_filters(self) -> Dict[str, Any]:
        """The filter arguments the controls currently spell out.

        A control at its neutral value contributes nothing rather than a
        no-op criterion, so the recorded filters on an exported list name
        only what the user actually asked for.
        """
        arguments: Dict[str, Any] = {}
        if self._q_spin.value() < 1.0 and self._all is not None and \
                self._all.ranking == "q-value":
            arguments["max_q"] = float(self._q_spin.value())
        if self._q_spin.value() < 1.0 and self._all is not None and \
                self._all.ranking == "selection-frequency":
            # The same dial means the opposite thing for a backend that ranks
            # by selection frequency: there is no q-value to be under, and a
            # threshold of 0.6 is a floor on how often the guide was chosen.
            arguments["min_selection"] = float(self._q_spin.value())
        if self._effect_spin.value() > 0.0:
            arguments["min_effect"] = float(self._effect_spin.value())
        if self._agreement_spin.value() > 0.0:
            arguments["min_agreement"] = float(self._agreement_spin.value())
        if self._guides_spin.value() > 0:
            arguments["min_guides"] = int(self._guides_spin.value())
        if self._direction.currentText() != "any":
            arguments["direction"] = self._direction.currentText()
        if self._drop_controls.isChecked():
            arguments["exclude_controls"] = True
        if self._query.text().strip():
            arguments["query"] = self._query.text().strip()
        return arguments

    def _on_filters_changed(self, *_args) -> None:
        """Re-narrow the list whenever a control moves."""
        self._apply_filters()

    def _apply_filters(self) -> None:
        """Narrow, redraw and report."""
        if self._all is None:
            return
        self._shown = self._all.filter(**self.current_filters())
        self._fill_table(self._shown)
        summary = self._shown.summary()
        parts = [f"{len(self._shown)} of {len(self._all)} genes shown",
                 f"{summary['n_up']} up, {summary['n_down']} down",
                 f"{summary['n_corroborated']} corroborated by two or more "
                 f"agreeing guides"]
        message = "; ".join(parts) + "."
        if self._all.notes:
            message = f"{message} {self._all.notes[0]}"
        self._set_summary(message, problem=len(self._shown) == 0)
        used = self._shown.flag_counts()
        self._legend.setText(
            "  ".join(f"{flag}: {FLAG_MEANING.get(flag, flag)}"
                      for flag in used) if used else "")
        self.hits_filtered.emit(self._shown)

    def _fill_table(self, hit_list: HitList) -> None:
        """Redraw the table from a list."""
        self._table.clear()
        for hit in hit_list:
            interval = ("—" if math.isnan(hit.ci_low)
                        else f"{hit.ci_low:.3g} … {hit.ci_high:.3g}")
            agreement = ("—" if math.isnan(hit.agreement)
                         else f"{hit.agreement:.0%}")
            item = QTreeWidgetItem([
                str(hit.rank), hit.gene, hit.name, _number(hit.effect),
                interval, _number(hit.p_value), _number(hit.q_value),
                f"{hit.n_agree}/{hit.n_guides}", agreement,
                hit.condition or "—", ", ".join(hit.flags) or "",
            ])
            item.setData(0, Qt.UserRole, hit.gene)
            if hit.flags:
                item.setToolTip(len(COLUMNS) - 1, "\n".join(
                    f"{flag}: {FLAG_MEANING.get(flag, flag)}"
                    for flag in hit.flags))
            self._table.addTopLevelItem(item)

    # -- export -----------------------------------------------------------

    def export(self, path: str, fmt: str = "csv") -> str:
        """Write the list as the filters currently stand.

        :param path: where to write.
        :param fmt: ``"csv"``, ``"markdown"`` or ``"html"``.
        :returns: the path written, or ``""`` when there was nothing to write.
        :raises ValueError: on an unknown format.
        """
        if self._shown is None:
            self._set_summary("There is no hit list to export yet.",
                              problem=True)
            return ""
        if fmt == "csv":
            written = self._shown.write_csv(path)
        elif fmt == "html":
            written = self._shown.write_html(path)
        elif fmt == "markdown":
            target = os.path.abspath(os.path.expanduser(path))
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(self._shown.to_markdown(limit=len(self._shown)))
            written = target
        else:
            raise ValueError(
                f"unknown export format {fmt!r}; use csv, markdown or html")
        self._set_summary(
            f"{len(self._shown)} row(s) written to {os.path.basename(written)}.",
            problem=False)
        return written

    def _ask_and_export(self, fmt: str, caption: str, filters: str) -> None:
        """Common half of the three export buttons."""
        if self._shown is None:
            self._set_summary("There is no hit list to export yet.",
                              problem=True)
            return
        path, _ = QFileDialog.getSaveFileName(self, caption, "", filters)
        if path:
            self.export(path, fmt)

    def _on_export_csv(self) -> None:                # pragma: no cover - modal
        self._ask_and_export("csv", "Export hit list", "CSV (*.csv)")

    def _on_export_markdown(self) -> None:           # pragma: no cover - modal
        self._ask_and_export("markdown", "Export hit list",
                             "Markdown (*.md)")

    def _on_export_html(self) -> None:               # pragma: no cover - modal
        self._ask_and_export("html", "Export hit list", "HTML (*.html)")

    # -- slots ------------------------------------------------------------

    def _on_folder_entered(self) -> None:
        """Load whatever was typed into the folder box."""
        self.load_folder(self._folder_edit.text())

    def _on_browse(self) -> None:                    # pragma: no cover - modal
        """Ask for a results folder and load it."""
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose a regression results folder")
        if chosen:
            self.load_folder(chosen)

    def _on_pick_metadata(self) -> None:             # pragma: no cover - modal
        """Ask for annotation CSVs and rebuild with them."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose gene metadata CSVs", "", "CSV (*.csv)")
        if paths:
            self.set_metadata_files(paths)

    def _on_job_failed(self, message: str) -> None:
        """Report a background failure inline; never a modal."""
        self.last_error = message
        self._set_summary(f"Could not build the hit list: {message}",
                          problem=True)

    def _set_summary(self, text: str, *, problem: bool) -> None:
        """Write the summary strip and repolish it for the problem colour."""
        self._summary.setText(text)
        self._summary.setProperty("problem", "true" if problem else "false")
        style = self._summary.style()
        if style is not None:
            style.unpolish(self._summary)
            style.polish(self._summary)

    # -- lifecycle --------------------------------------------------------

    def is_busy(self) -> bool:
        """True while a hit list is still being built."""
        return self._jobs.is_busy()

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return self._jobs.active_jobs()

    def closeEvent(self, event) -> None:             # noqa: N802 - Qt override
        """Drain the worker before the widget goes."""
        self._jobs.shutdown()
        super().closeEvent(event)


def _number(value: Any) -> str:
    """Format a number for a table cell; an em dash for a missing one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    if number and (abs(number) < 1e-3 or abs(number) >= 1e5):
        return f"{number:.3g}"
    return f"{number:.4g}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def make_hit_list_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory the registry calls to build this screen."""
    return HitListScreen()


def register() -> bool:
    """Add Hit List to the app registry. Idempotent.

    :returns: True when this call added the row, False when it was already
        there — which is what a second import, or a plugin that pulls the
        module in again, must not treat as an error.
    """
    from ..app import APPS, SECTION_RESULTS, STAGE_ALPHA, register_app

    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(
        APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_RESULTS,
        factory=make_hit_list_screen, stage=STAGE_ALPHA,
        title="Hit List", intro=APP_INTRO, cli_note=APP_CLI_NOTE,
        api_module="qt/screens/hit_list",
        translations=APP_TRANSLATIONS)
    return True


register()
