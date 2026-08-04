"""Design the plate before it is acquired, and export it once.

The screen half of :mod:`spacr.qt.widgets.plate_layout`. It draws the plate,
lists what is wrong with the layout, and writes the three files the rest of
spaCR reads -- so the plate map is typed once rather than once for the plate
handler and again into ``treatment_plate_metadata`` when the data comes back.

The warnings are the point rather than the drawing. Every finding it shows is
about something that cannot be repaired after acquisition: controls sitting
only on the plate edge, a control confined to one column, a condition with a
single replicate. All of them are free to fix the day before and impossible to
fix the day after.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..job_runner import JobRunner
from ..theme import SPACING, register_widget_qss
from .app_screen import ModuleHeader
from ...schema import letters_from_row_index
from ..widgets.plate_layout import (
    EDGE_LEAVE_EMPTY, EDGE_USE, LAYOUTS, PLATE_FORMATS, ROLES,
    ROLE_NEGATIVE, ROLE_POSITIVE, ROLE_TREATMENT, Condition, PlateDesign,
    assign_wells, check_design, plate_shape, to_settings_fragment,
    write_design,
)

__all__ = [
    "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO", "APP_CLI_NOTE",
    "APP_TRANSLATIONS", "ExperimentDesignScreen",
    "make_experiment_design_screen", "register",
]

#: Stable app id. Chosen once; saved user state and the registry key off it.
APP_KEY = "experiment_design"

APP_NAME = "Experiment Design"
APP_DESCRIPTION = (
    "Lay out conditions, controls and replicates on a plate, check the "
    "layout, and export it for the pipeline to read later."
)
APP_INTRO = (
    "Everything on this screen is a decision that cannot be undone after "
    "acquisition. Where the controls sit, whether a condition is confounded "
    "with a row, whether the plate edge is used at all -- no analysis "
    "repairs any of them, and all of them are free to change today. Export "
    "writes a plate_map.csv keyed the way spaCR keys measurements, so the "
    "layout is typed once instead of twice."
)
APP_CLI_NOTE = (
    "Experiment Design is a GUI screen: it exists to draw a plate and warn "
    "about its layout before acquisition. For a headless design, build a "
    "spacr.qt.widgets.plate_layout.PlateDesign and call write_design() "
    "instead -- that is the same code this screen runs."
)
#: sv, de, es, zh_CN, pt, hi, ko, is, fr
APP_TRANSLATIONS: Tuple[str, ...] = (
    "Experimentdesign", "Experimentdesign", "Diseño de experimento",
    "实验设计", "Desenho do experimento", "प्रयोग डिज़ाइन", "실험 설계",
    "Tilraunahönnun", "Conception d'expérience",
)

PLATE_OBJECT = "spacrPlateMap"
FINDINGS_OBJECT = "spacrDesignFindings"
STATUS_OBJECT = "spacrDesignStatus"

def _design_qss(palette: dict, opacity: Optional[float] = None) -> str:
    """QSS for this screen, rebuilt on every theme change.

    Registered rather than set inline: an inline stylesheet is baked at
    construction and survives a theme switch, so a well painted under the
    dark theme keeps its dark-theme colour on a light background.

    ``palette`` arrives with its surface roles already rendered through the
    page-opacity preference, so a rule naming one follows the slider. The
    findings panel needs a rule at all for that to matter: a named
    ``QWidget`` with no rule of its own takes the blanket
    ``QWidget {{ background-color: bg }}``, which is the WINDOW colour and
    not a surface, so no setting could ever reach it.
    """
    from ..theme import block_surface
    findings_bg = block_surface("surface_alt", palette.get("theme"), opacity)
    return f"""
#{FINDINGS_OBJECT} {{
    background: {findings_bg};
    border: 1px solid {palette['border_soft']};
    border-radius: 6px;
}}
#{PLATE_OBJECT} {{
    background: {palette['surface_alt']};
    border: 1px solid {palette['border']};
    border-radius: 6px;
}}
#{PLATE_OBJECT} QLabel[spacrWellRole="negative_control"] {{
    background: {palette['info']};
    color: {palette['bg']};
    border-radius: 3px;
}}
#{PLATE_OBJECT} QLabel[spacrWellRole="positive_control"] {{
    background: {palette['warning']};
    color: {palette['bg']};
    border-radius: 3px;
}}
#{PLATE_OBJECT} QLabel[spacrWellRole="treatment"] {{
    background: {palette['accent']};
    color: {palette['bg']};
    border-radius: 3px;
}}
#{PLATE_OBJECT} QLabel[spacrWellRole="blank"] {{
    background: {palette['surface']};
    color: {palette['fg_muted']};
    border: 1px dashed {palette['border']};
    border-radius: 3px;
}}
#{PLATE_OBJECT} QLabel[spacrWellRole="empty"] {{
    background: transparent;
    color: {palette['fg_muted']};
    border: 1px solid {palette['border']};
    border-radius: 3px;
}}
#{PLATE_OBJECT} QLabel[spacrWellEdge="true"] {{
    border: 2px solid {palette['warning']};
}}
#{FINDINGS_OBJECT} QLabel[spacrFindingSeverity="error"] {{
    color: {palette['error']};
}}
#{FINDINGS_OBJECT} QLabel[spacrFindingSeverity="warn"] {{
    color: {palette['warning']};
}}
#{FINDINGS_OBJECT} QLabel[spacrFindingSeverity="note"] {{
    color: {palette['fg_muted']};
}}
#{STATUS_OBJECT} {{ color: {palette['fg_muted']}; }}
#{STATUS_OBJECT}[spacrError="true"] {{ color: {palette['error']}; }}
"""


# `replace=True`: this module is reachable both through the screens package
# and by direct import, and a second import must refresh the block rather
# than raise. Same posture as the power screen.
register_widget_qss("ExperimentDesign", _design_qss, replace=True)


class ExperimentDesignScreen(QWidget):
    """Plate designer: conditions in, plate map and warnings out.

    :param threaded: ``False`` runs the export inline, emitting the same
        signals in the same order, so a test can drive the screen
        synchronously without the behaviour diverging.
    """

    def __init__(self, parent: Optional[QWidget] = None, *,
                 threaded: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("ExperimentDesignScreen")
        self._jobs = JobRunner(self, threaded=threaded, app_key=APP_KEY)
        self._jobs.job_failed.connect(self._on_job_failed)
        self._well_labels: List[QLabel] = []
        self._findings_labels: List[QLabel] = []
        self._build()
        self.refresh()

    # -- construction -----------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"],
                                 SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["md"])

        header = ModuleHeader(
            APP_NAME,
            description=APP_DESCRIPTION,
            instruction="Name the plate, choose the format, lay the "
                        "conditions out, then export the map.",
        )
        self._header = header
        outer.addWidget(header)

        intro = QLabel(APP_INTRO)
        intro.setWordWrap(True)
        outer.addWidget(intro)

        form = QHBoxLayout()
        form.setSpacing(SPACING["sm"])
        self._plate_id = QLineEdit("plate1")
        self._plate_id.setToolTip(
            "Must match the plate name the image file names will carry, or "
            "the exported map will not join to the measurements.")
        self._plate_id.textChanged.connect(self._on_changed)
        form.addWidget(QLabel("Plate:"))
        form.addWidget(self._plate_id)

        self._format = QComboBox()
        for well_count in sorted(PLATE_FORMATS):
            rows, columns = PLATE_FORMATS[well_count]
            self._format.addItem(f"{well_count} ({rows}x{columns})",
                                 well_count)
        self._format.setCurrentIndex(
            self._format.findData(96))
        self._format.currentIndexChanged.connect(self._on_changed)
        form.addWidget(QLabel("Format:"))
        form.addWidget(self._format)

        self._layout_box = QComboBox()
        self._layout_box.addItems(LAYOUTS)
        self._layout_box.setToolTip(
            "random is the only layout that cannot be confounded with a "
            "position gradient. block is the easiest to pipette and is "
            "guaranteed to confound condition with position.")
        self._layout_box.currentIndexChanged.connect(self._on_changed)
        form.addWidget(QLabel("Layout:"))
        form.addWidget(self._layout_box)

        self._edge = QComboBox()
        self._edge.addItem("use edge wells", EDGE_USE)
        self._edge.addItem("leave edge empty", EDGE_LEAVE_EMPTY)
        self._edge.currentIndexChanged.connect(self._on_changed)
        form.addWidget(QLabel("Edge:"))
        form.addWidget(self._edge)

        self._seed = QSpinBox()
        self._seed.setRange(0, 999999)
        self._seed.setToolTip(
            "A random layout nobody can regenerate is a layout nobody can "
            "check.")
        self._seed.valueChanged.connect(self._on_changed)
        form.addWidget(QLabel("Seed:"))
        form.addWidget(self._seed)
        form.addStretch(1)
        outer.addLayout(form)

        # -- conditions ---------------------------------------------------
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Condition", "Replicates",
                                               "Role"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._table.setMaximumHeight(220)
        self._table.itemChanged.connect(self._on_changed)
        outer.addWidget(self._table)

        buttons = QHBoxLayout()
        add = QPushButton("Add condition")
        add.clicked.connect(self._add_row)
        buttons.addWidget(add)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_row)
        buttons.addWidget(remove)
        buttons.addStretch(1)
        self._export = QPushButton("Export plate map...")
        self._export.clicked.connect(self._on_export)
        buttons.addWidget(self._export)
        outer.addLayout(buttons)

        # -- plate --------------------------------------------------------
        self._plate_panel = QWidget()
        self._plate_panel.setObjectName(PLATE_OBJECT)
        self._plate_grid = QGridLayout(self._plate_panel)
        self._plate_grid.setSpacing(2)
        self._plate_grid.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                            SPACING["sm"], SPACING["sm"])
        scroll = QScrollArea()
        scroll.setWidget(self._plate_panel)
        scroll.setWidgetResizable(True)
        # The viewport auto-fills with the WINDOW colour, which no page
        # opacity can reach. The plate map covers most of it, but the strip
        # beside a short plate is the same slab the settings column was.
        scroll.viewport().setAutoFillBackground(False)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        outer.addWidget(scroll, 1)

        # -- findings -----------------------------------------------------
        self._findings_panel = QWidget()
        self._findings_panel.setObjectName(FINDINGS_OBJECT)
        self._findings_layout = QVBoxLayout(self._findings_panel)
        # Room for the panel's own border, now that the findings sit on a
        # surface rather than straight on the window.
        self._findings_layout.setContentsMargins(SPACING["sm"], SPACING["xs"],
                                                 SPACING["sm"], SPACING["xs"])
        self._findings_layout.setSpacing(2)
        outer.addWidget(self._findings_panel)

        self._status = QLabel("")
        self._status.setObjectName(STATUS_OBJECT)
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        # A default worth starting from rather than an empty table: both
        # controls present, enough replicates to mean something.
        self._set_conditions([
            Condition("negative", 6, ROLE_NEGATIVE),
            Condition("positive", 6, ROLE_POSITIVE),
            Condition("treatment_a", 12, ROLE_TREATMENT),
        ])

    # -- the design -------------------------------------------------------

    def _set_conditions(self, conditions) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for condition in conditions:
            self._append_row(condition)
        self._table.blockSignals(False)

    def _append_row(self, condition: Condition) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(condition.name))
        self._table.setItem(
            row, 1, QTableWidgetItem(str(int(condition.replicates))))
        box = QComboBox()
        box.addItems(ROLES)
        box.setCurrentText(condition.role)
        box.currentIndexChanged.connect(self._on_changed)
        self._table.setCellWidget(row, 2, box)

    def _add_row(self) -> None:
        self._append_row(Condition(f"condition_{self._table.rowCount() + 1}",
                                   3, ROLE_TREATMENT))
        self.refresh()

    def _remove_row(self) -> None:
        rows = sorted({index.row() for index in
                       self._table.selectedIndexes()}, reverse=True)
        for row in rows:
            self._table.removeRow(row)
        self.refresh()

    def conditions(self) -> Tuple[Condition, ...]:
        """The conditions currently in the table, skipping unusable rows.

        A half-typed row is not an error to shout about -- the user is in the
        middle of typing it -- so it is dropped and the plate redrawn from
        what is complete.
        """
        out: List[Condition] = []
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 0)
            count_item = self._table.item(row, 1)
            box = self._table.cellWidget(row, 2)
            name = (name_item.text() if name_item else "").strip()
            if not name:
                continue
            try:
                replicates = int((count_item.text() if count_item else "1"))
            except (TypeError, ValueError):
                continue
            if replicates < 1:
                continue
            role = box.currentText() if isinstance(box, QComboBox) else \
                ROLE_TREATMENT
            out.append(Condition(name, replicates, role))
        return tuple(out)

    def design(self) -> PlateDesign:
        """The design the form currently describes."""
        return PlateDesign(
            plate_id=self._plate_id.text().strip() or "plate1",
            plate_format=int(self._format.currentData()),
            conditions=self.conditions(),
            layout=self._layout_box.currentText(),
            edge_policy=self._edge.currentData(),
            seed=int(self._seed.value()),
        )

    # -- drawing ----------------------------------------------------------

    def _on_changed(self, *_args) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Redraw the plate and the findings from the current form."""
        design = self.design()
        try:
            table = assign_wells(design)
            error = ""
        except ValueError as exc:
            table = None
            error = str(exc)
        self._draw_plate(design, table)
        self._draw_findings(check_design(design, table))
        if error:
            self._set_status(error, is_error=True)
        elif table is not None:
            fragment = to_settings_fragment(design, table)
            if fragment["expressible"]:
                self._set_status(
                    f"{len(table)} of {design.wells_available} usable wells "
                    "assigned. This layout can also be exported as "
                    "treatment_plate_metadata.")
            else:
                self._set_status(
                    f"{len(table)} of {design.wells_available} usable wells "
                    "assigned. " + fragment["reason"])

    def _draw_plate(self, design: PlateDesign, table) -> None:
        while self._plate_grid.count():
            item = self._plate_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self._well_labels = []
        rows, columns = plate_shape(design.plate_format)
        assigned = {}
        if table is not None and len(table):
            for record in table.to_dict("records"):
                assigned[(record["row_index"], record["column_index"])] = record

        for column in range(1, columns + 1):
            header = QLabel(str(column))
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._plate_grid.addWidget(header, 0, column)
        for row in range(1, rows + 1):
            header = QLabel(letters_from_row_index(row))
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._plate_grid.addWidget(header, row, 0)
            for column in range(1, columns + 1):
                record = assigned.get((row, column))
                label = QLabel("")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumSize(22, 18)
                if record is None:
                    label.setProperty("spacrWellRole", "empty")
                    label.setToolTip("unassigned")
                else:
                    label.setProperty("spacrWellRole", record["role"])
                    label.setText(str(record["condition"])[:2])
                    label.setToolTip(
                        f"{record['well']} - {record['condition']} "
                        f"({record['role']}, replicate {record['replicate']})"
                        + (" - EDGE WELL" if record["is_edge"] else ""))
                    label.setProperty("wellName", record["well"])
                    label.setProperty("wellCondition", record["condition"])
                label.setProperty(
                    "spacrWellEdge",
                    "true" if (record is not None and record["is_edge"])
                    else "false")
                self._plate_grid.addWidget(label, row, column)
                self._well_labels.append(label)

    def _draw_findings(self, findings) -> None:
        while self._findings_layout.count():
            item = self._findings_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self._findings_labels = []
        marks = {"error": "STOP", "warn": "!", "note": "-"}
        for finding in findings:
            label = QLabel(f"{marks[finding.severity]} {finding.message}")
            label.setWordWrap(True)
            label.setProperty("spacrFindingSeverity", finding.severity)
            label.setProperty("findingKey", finding.key)
            self._findings_layout.addWidget(label)
            self._findings_labels.append(label)

    def findings_text(self) -> str:
        """Every finding line currently on screen, joined. For tests."""
        return "\n".join(label.text() for label in self._findings_labels)

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

    # -- export -----------------------------------------------------------

    def _on_export(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Write the plate map into")
        if folder:
            self.export_to(folder)

    def export_to(self, folder) -> bool:
        """Write the plate map. The file writing happens off the GUI thread.

        :param folder: destination directory.
        :returns: whether the job was started.
        """
        design = self.design()
        if not design.conditions:
            self._set_status("Nothing to export: the plate has no conditions.",
                             is_error=True)
            return False
        return self._jobs.submit(
            lambda d=design, f=folder: write_design(d, f),
            self._on_exported)

    def _on_exported(self, paths) -> None:
        if not paths:
            return
        self._set_status(
            "Wrote " + ", ".join(sorted(p.name for p in paths.values()))
            + f" to {paths['plate_map'].parent}. plate_map.csv joins to a "
              "measurements table on (plateID, rowID, columnID).")

    def _on_job_failed(self, message: str) -> None:
        self._set_status(f"Export failed: {message}", is_error=True)

    # -- lifecycle --------------------------------------------------------

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return self._jobs.active_jobs()

    def is_busy(self) -> bool:
        """True while an export has not finished."""
        return self._jobs.is_busy()

    def closeEvent(self, event):  # noqa: N802 - Qt name
        self._jobs.shutdown()
        super().closeEvent(event)


def make_experiment_design_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return ExperimentDesignScreen()


def register() -> bool:
    """Add Experiment Design to the app registry. Idempotent."""
    from ..app import APPS, SECTION_DESIGN, STAGE_ALPHA, register_app

    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(
        APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_DESIGN,
        factory=make_experiment_design_screen, stage=STAGE_ALPHA,
        title=APP_NAME, intro=APP_INTRO, cli_note=APP_CLI_NOTE,
        api_module="qt/screens/experiment_design",
        translations=APP_TRANSLATIONS)
    return True


register()
