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

ONE PLATE WIDGET WHERE THERE CAN BE ONE. The square, the constant it is
pitched at and the row/column headers come from
:mod:`spacr.qt.widgets.plate_map_picker` rather than being declared a second
time here; the two plates had already drifted apart once. What cannot be
shared is the well itself, and the difference is real rather than
historical: the picker's well is a checkable ``QPushButton`` whose checked
state IS the selection and which paints itself from a fixed pair of colours,
while this one is a ``QLabel`` carrying a role, an edge mark, a name and a
condition, painted by the theme through the properties below -- a plate map
that says what each well holds rather than only whether it is chosen.
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
# THE OTHER PLATE'S GEOMETRY, IMPORTED RATHER THAN REPEATED. `WELL_SIDE` is
# the side both plates are pitched at, `_locked_square` states that side in
# the one language that survives being polished under the application
# stylesheet, and `_Header` is the row letter and the column number locked to
# a cell of it. All three were written for the picker and are not specific to
# it; a second copy here is what let the two plates drift.
from ..widgets.plate_map_picker import WELL_SIDE, _Header, _locked_square
from ..widgets.sortable_table import install_sorting, table_item
from ..app_catalog import declared_app, register_declared

__all__ = [
    "APP_KEY", "APP_NAME", "APP_DESCRIPTION", "APP_INTRO", "APP_CLI_NOTE",
    "APP_TRANSLATIONS", "ExperimentDesignScreen",
    "make_experiment_design_screen", "register",
]

#: Stable app id. Chosen once; saved user state and the registry key off it.
APP_KEY = "experiment_design"

# The row this screen puts in the registry is declared in
# `spacr.qt.app_catalog`, which is what lets the app be registered without
# importing this module -- the launch reads the table, not the screen. These
# read the same row back rather than restating it, so the name, the blurb and
# the nine translations have one spelling and no second copy to drift from.
_ROW = declared_app(APP_KEY)
APP_NAME = _ROW.name
APP_DESCRIPTION = _ROW.desc
APP_INTRO = _ROW.intro
APP_CLI_NOTE = _ROW.cli_note
#: sv, de, es, zh_CN, pt, hi, ko, is, fr
APP_TRANSLATIONS: Tuple[str, ...] = (
    "Experimentdesign", "Experimentdesign", "Diseño de experimento",
    "实验设计", "Desenho do experimento", "प्रयोग डिज़ाइन", "실험 설계",
    "Tilraunahönnun", "Conception d'expérience",
)

#: How thick a rim a well draws, in pixels per side, in each of the two
#: states that draw one: an outline for a well that holds nothing, and a
#: mark for one on the plate edge or under the selection. A well filled with
#: its role draws no rim at all.
#:
#: ONE TABLE READ TWICE. :func:`_design_qss` draws the border from it and
#: :func:`_well_sheet` subtracts it from :data:`WELL_SIDE` to reach the
#: content box the square is stated in -- Qt adds the border back on. Read
#: apart, the two disagree by the width of a rim and the well changes size
#: the moment it is selected, shoving every well to the right of it along.
OUTLINE_RIM = 1
MARK_RIM = 2

#: The roles drawn as an outline rather than as a fill.
OUTLINED_ROLES = ("blank", "empty")


def _well_rim(role, edge: bool, chosen: bool) -> int:
    """The border width the sheet ends up drawing on one well.

    The last matching rule in :func:`_design_qss` wins, and the order there is
    role, then edge, then selection -- so a chosen edge well is marked once,
    at :data:`MARK_RIM`.
    """
    if chosen or edge:
        return MARK_RIM
    return OUTLINE_RIM if role in OUTLINED_ROLES else 0


def _well_sheet(rim: int) -> str:
    """The sheet ONE well wears: its square, stated on the well itself.

    A widget's own sheet outranks the application's, and that is the whole
    point of this rather than :meth:`~PySide6.QtWidgets.QWidget.setFixedSize`.
    ``QStyleSheetStyle`` does not merely draw a geometry rule -- it polishes
    it into a real ``setMinimumHeight`` on the widget, which OVERWRITES the
    minimum ``setFixedSize`` wrote while leaving the maximum in place, and Qt
    resolves a minimum taller than its maximum in favour of the minimum. One
    blanket ``QLabel { min-height: ... ; padding: ... }`` anywhere in the
    application sheet is enough to turn every well of this plate into an
    oblong, silently. The neighbouring plate lost its square to exactly that,
    through ``QPushButton``; nothing but the absence of such a rule for
    ``QLabel`` was keeping this one.

    ``border-width`` is stated here for the same reason, and only the width:
    the style and the colour still come from :func:`_design_qss`, so the
    outline the map is read by is unchanged, while the space it takes is the
    plate's own arithmetic rather than whatever a blanket rule asks for.
    """
    rim = int(rim)
    return "QLabel { border-width: %dpx; %s }" % (rim, _locked_square(rim))


def _plate_header(text: str, parent) -> _Header:
    """One header cell: the picker's ``_Header``, with the rim pinned too.

    The class comes from the other plate rather than being written again --
    the row letters and the column numbers are cells of the grid, pitched to
    the same square as a well, and both plates want exactly that. What it
    does not pin is the border width, so a blanket ``QLabel { border: ... }``
    still inflates a header by twice that border while the wells beside it
    hold their square. This plate's headers draw no rim of their own, so
    saying so costs nothing and closes the gap for them; the durable fix is
    for ``_locked_square`` to state the width it is already subtracting.
    """
    cell = _Header(text, parent)
    cell.setStyleSheet(_well_sheet(0))
    return cell


class _Well(QLabel):
    """One well of the plate map, which knows where it is and reports clicks.

    A `QLabel` still -- the map is a picture, not a form -- with the two
    facts a selection needs: its coordinate, and a press that reaches the
    screen. Drawing 1,536 checkable buttons to get a click is a heavier
    answer to a lighter question.

    :param row: the well's row index, counting from zero.
    :param column: its column index.
    :param parent: parent widget; ownership only.

    The pair is the well's IDENTITY, not merely its position -- the screen
    addresses wells by it, and it is what a press reports back.
    """

    def __init__(self, row: int, column: int, parent=None):
        """Build one well, centred and locked to a square."""
        super().__init__("", parent)
        self.row, self.column = int(row), int(column)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # The hint. The floor and the ceiling are in the sheet below, which
        # is the half of this that survives being polished.
        self.setFixedSize(WELL_SIDE, WELL_SIDE)
        self._rim = None
        self.lock_square()

    def lock_square(self) -> None:
        """Re-state the square for the state the well is now in.

        Called whenever a property the sheet reads changes, because both
        halves of the drawing hang off it. THE STYLE IS RE-POLISHED, not
        merely restated: the well's colour comes from the sheet's
        ``spacrWellRole``, and Qt does not re-evaluate a selector when a
        property changes unless it is asked to. And the square is re-stated
        with it, because the rim a state draws is part of the widget's
        height: a chosen well marked 2 px where it was outlined at 1 would
        grow by two pixels if the content box were left where it was.
        """
        rim = _well_rim(self.property("spacrWellRole"),
                        self.property("spacrWellEdge") == "true",
                        self.property("spacrWellChosen") == "true")
        if rim != self._rim:
            self._rim = rim
            # Setting a sheet repolishes the widget on its own.
            self.setStyleSheet(_well_sheet(rim))
            return
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)

    def _screen(self):
        """Walk up to the screen that handles well drags, or ``None``.

        Found by CAPABILITY rather than by type -- the first ancestor that has
        ``begin_well_drag`` -- so the well works under any host that offers the
        same handle, including a test double.
        """
        widget = self.parent()
        while widget is not None and not hasattr(widget, "begin_well_drag"):
            widget = widget.parent()
        return widget

    def mousePressEvent(self, event):                # noqa: N802 - Qt
        screen = self._screen()
        if screen is not None:
            screen.begin_well_drag(self.row, self.column, event.modifiers())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):                 # noqa: N802 - Qt
        # THE PRESSED WIDGET KEEPS THE GRAB, so the well under the pointer
        # has to be found by asking rather than by waiting to be entered.
        screen = self._screen()
        if screen is not None:
            screen.drag_wells_to(
                self.mapToGlobal(event.position().toPoint()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):              # noqa: N802 - Qt
        screen = self._screen()
        if screen is not None:
            screen.finish_well_drag()
        super().mouseReleaseEvent(event)

#: The grid row and column that absorb everything the window has spare. Far
#: past any real plate -- 1536 is 32 x 48 -- so it is never a well's own.
_SLACK = 200

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
    # THE RIM WIDTHS BELOW COME FROM THE TABLE, not from a number typed
    # here: a well states its own square in content-box pixels, and Qt adds
    # the border back on, so a border this sheet draws wider than the well
    # allowed for is a well two pixels bigger than its neighbours.
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
    border: {OUTLINE_RIM}px dashed {palette['border']};
    border-radius: 3px;
}}
#{PLATE_OBJECT} QLabel[spacrWellRole="empty"] {{
    background: transparent;
    color: {palette['fg_muted']};
    border: {OUTLINE_RIM}px solid {palette['border']};
    border-radius: 3px;
}}
#{PLATE_OBJECT} QLabel[spacrWellEdge="true"] {{
    border: {MARK_RIM}px solid {palette['warning']};
}}
/* THE SELECTION IS AN OUTLINE, NOT A FILL (194 A). The fill already says
   what the well IS -- control, treatment, blank -- and replacing it would
   trade the information the map exists for against the one thing the user
   can see for themselves, which is where they just dragged. LAST in the
   sheet so it wins over the role and the edge rules above. */
#{PLATE_OBJECT} QLabel[spacrWellChosen="true"] {{
    border: {MARK_RIM}px solid {palette['accent']};
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
    :param parent: parent widget; ownership only.
    """

    def __init__(self, parent: Optional[QWidget] = None, *,
                 threaded: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("ExperimentDesignScreen")
        self._jobs = JobRunner(self, threaded=threaded, app_key=APP_KEY)
        self._jobs.job_failed.connect(self._on_job_failed)
        self._well_labels: List["_Well"] = []
        # 194 A: the drag's state. `None` anchor means no gesture is live.
        self._well_anchor = None
        self._well_last = None
        self._wells_before: set = set()
        self._well_adding = False
        self._findings_labels: List[QLabel] = []
        self._build()
        self.refresh()
        # Hover help belongs on a setting's NAME, not on the field the user
        # is about to type into (instruction 113). One post-pass rather than
        # a convention every hand-built row has to remember.
        from .settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

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
        install_sorting(self._table)
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
        # THE SLACK GOES OUTSIDE THE PLATE, NOT BETWEEN ITS WELLS. The grid
        # is inside a resizable scroll area, so the panel grows with the
        # window; without somewhere for that extra space to go, a
        # QGridLayout hands it to the cells and the map spreads. One
        # trailing row and column take all of it, which keeps every well
        # the same distance from its neighbour at every window size.
        self._plate_grid.setRowStretch(_SLACK, 1)
        self._plate_grid.setColumnStretch(_SLACK, 1)
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
        self._table.setItem(row, 0, table_item(condition.name))
        self._table.setItem(
            row, 1, table_item(str(int(condition.replicates))))
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

    # ------------------------------------------------------------- 194 A
    #
    # "the plate map should be made up of squares that are clickable nad the
    # user should be able to drag and select."
    #
    # PRESS ANCHORS, MOVE PREVIEWS, RELEASE COMMITS -- and the preview is
    # visible while dragging, because a selection you cannot see until you
    # let go is one you have to undo to correct. Ctrl adds a second
    # rectangle; a plain drag replaces, which is what every other grid in
    # this application does.

    def begin_well_drag(self, row: int, column: int, modifiers=None) -> None:
        """Anchor a selection on one well."""
        self._well_anchor = (int(row), int(column))
        self._well_adding = bool(
            modifiers is not None and (modifiers & Qt.ControlModifier))
        # REDRAWN FROM THE STATE AT THE PRESS, not from the last frame, so
        # growing and then shrinking the rectangle leaves nothing behind.
        self._wells_before = set(self.selected_wells())
        self._select_wells(self._rectangle(self._well_anchor,
                                           self._well_anchor))

    def well_at(self, position):
        """The ``(row, column)`` under a GLOBAL point, or ``None``."""
        for label in self._well_labels:
            if label.rect().contains(label.mapFromGlobal(position)):
                return (label.row, label.column)
        return None

    def drag_wells_to(self, position) -> None:
        """Preview the rectangle from the anchor to the well under a point."""
        anchor = getattr(self, "_well_anchor", None)
        if anchor is None:
            return
        cell = self.well_at(position)
        if cell is None or cell == getattr(self, "_well_last", None):
            return
        self._well_last = cell
        self._select_wells(self._rectangle(anchor, cell))

    def finish_well_drag(self) -> None:
        """End the gesture. The selection is already what the preview showed."""
        self._well_anchor = None
        self._well_last = None

    def selected_wells(self) -> set:
        """Every ``(row, column)`` currently chosen on the map."""
        return {(label.row, label.column) for label in self._well_labels
                if label.property("spacrWellChosen") == "true"}

    def selected_well_names(self) -> list:
        """The chosen wells as names, in reading order."""
        return [str(label.property("wellName") or "")
                for label in self._well_labels
                if label.property("spacrWellChosen") == "true"
                and label.property("wellName")]

    def _rectangle(self, start, end) -> set:
        """Every cell in the block between two coordinates, inclusive."""
        top, bottom = sorted((int(start[0]), int(end[0])))
        left, right = sorted((int(start[1]), int(end[1])))
        block = {(r, c) for r in range(top, bottom + 1)
                 for c in range(left, right + 1)}
        if getattr(self, "_well_adding", False):
            block |= set(getattr(self, "_wells_before", set()))
        return block

    def _select_wells(self, cells) -> None:
        """Mark exactly ``cells`` as chosen and restyle what changed."""
        wanted = {(int(r), int(c)) for r, c in cells}
        for label in self._well_labels:
            chosen = "true" if (label.row, label.column) in wanted else "false"
            if label.property("spacrWellChosen") != chosen:
                label.setProperty("spacrWellChosen", chosen)
                # Repaints AND re-squares: the selection is drawn as a rim,
                # and a rim is part of the widget's size. See
                # `_Well.lock_square`.
                label.lock_square()

    def _draw_plate(self, design: PlateDesign, table) -> None:
        # WHAT THE USER CHOSE OUTLIVES THE REDRAW. Every well on this plate
        # is destroyed and rebuilt on every `refresh`, and `refresh` runs on
        # ONE KEYSTROKE in the plate name, on a nudge of the seed spinner and
        # on every edit of the condition table -- so a selection read off the
        # widgets alone was wiped by typing, not by anything the user did to
        # the selection. Carried across as coordinates, which is the one form
        # of it that survives the widgets being thrown away.
        chosen = self.selected_wells()
        while self._plate_grid.count():
            item = self._plate_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self._well_labels = []
        rows, columns = plate_shape(design.plate_format)
        # A SMALLER PLATE DROPS WHAT IS NO LONGER ON IT rather than keeping a
        # coordinate that names nothing: 384 down to 96 has to forget H13,
        # or a later switch back would resurrect a well the user cannot see.
        chosen = {(row, column) for row, column in chosen
                  if 1 <= row <= rows and 1 <= column <= columns}
        assigned = {}
        if table is not None and len(table):
            for record in table.to_dict("records"):
                assigned[(record["row_index"], record["column_index"])] = record

        # THE HEADERS ARE CELLS OF THE PLATE, not captions beside it, and
        # `_Header` is the same one the picker's plate uses: locked to the
        # well's square, so the header row is exactly one cell tall and the
        # header column exactly one cell wide however tall the theme's font
        # makes a label. Left to size themselves they answer to a blanket
        # QLabel rule like anything else.
        for column in range(1, columns + 1):
            self._plate_grid.addWidget(
                _plate_header(str(column), self._plate_panel), 0, column)
        for row in range(1, rows + 1):
            self._plate_grid.addWidget(
                _plate_header(letters_from_row_index(row),
                              self._plate_panel), row, 0)
            for column in range(1, columns + 1):
                record = assigned.get((row, column))
                # SQUARE, AND FIXED TO ITS NEIGHBOURS (194). This was a
                # `QLabel` with `setMinimumSize(22, 18)` and no maximum, so
                # every well stretched with the window -- measured at 900,
                # 1500 and 1900 px wide, one went 65 -> 111 -> 141 px across
                # while staying 18 tall. A plate map is a picture of a
                # physical object, and the point of the picture is that its
                # proportions are the object's: at 141 x 18 it is not a
                # plate any more, and "the elements drift appart" is what a
                # reader sees.
                label = _Well(row, column, self._plate_panel)
                # EVERY WELL KNOWS ITS NAME, assigned or not. A name is a
                # COORDINATE -- `letters_from_row_index(row)` and the column
                # -- and it was being set only on the assigned branch, so a
                # user who selected an empty block got a selection that
                # could not say which wells it held.
                label.setProperty("wellName",
                                  f"{letters_from_row_index(row)}"
                                  f"{column:02d}")
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
                # SET BEFORE THE SQUARE IS LOCKED, not restored afterwards:
                # the selection is drawn as a rim, a rim is part of the
                # widget's size, and stating it here settles the well at its
                # final size in one pass rather than resizing a plate's worth
                # of wells the moment the selection is put back on them.
                label.setProperty(
                    "spacrWellChosen",
                    "true" if (row, column) in chosen else "false")
                # The role, the edge mark and the selection are what decide
                # the rim, so the square is settled once they are on the
                # widget.
                label.lock_square()
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
    return register_declared(__name__) is not None


register()
