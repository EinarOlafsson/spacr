"""The repeated per-object settings, drawn as one row per question.

78 of Mask's 201 settings are the SAME twenty-odd questions asked once per
object type -- ``cell_diameter``, ``nucleus_diameter``,
``pathogen_diameter``, ``organelle_diameter`` -- and a form that lists them
flat asks 203 questions before anything is segmented. A table was chosen over
tabs and over leaving the names flat.

:mod:`spacr.object_settings_table` is the model and draws nothing; this is
the view over it. The split matters more than it looks: the stored keys never
change, so no settings file, notebook, tutorial or ``spacr-run`` invocation
migrates. What was wrong was the presentation, so only the presentation
changes.

WHY THIS SHAPE IS WHAT LETS AN ARBITRARY ORGANELLE COUNT LAND. The number of
organelles a run may declare is not fixed. In a flat vocabulary each new organelle is twenty new
settings that every tooltip table and translation catalog has to learn; here
it is one COLUMN, and the number of questions does not move.
:meth:`ObjectSettingsGrid.add_object` is that operation, and it starts a new
organelle from the first one's answers rather than from a global default
nobody chose.

TWO THINGS THIS VIEW IS CAREFUL ABOUT, both of which would corrupt a settings
file rather than merely look wrong:

* **A value keeps its type.** ``cell_diameter`` is an int, ``organelle_
  cellprob_threshold`` is a float, and a cell edited in a table arrives as a
  string. Writing ``"12"`` where ``12`` was is a settings file that has
  quietly changed meaning, and the pipeline reading it either coerces
  silently or fails a long way from here.
* **A question an object does not ask stays absent.** ``cytoplasm`` has no
  channel, no diameter and no detection method -- it is DERIVED, cell minus
  the rest, not found in a channel. Those cells are blank and not editable,
  because writing a value there invents a key nothing reads.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

from PySide6.QtCore import (QAbstractTableModel, QEvent, QModelIndex, QSize,
                            Qt, QTimer, Signal)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)

LOG = logging.getLogger(__name__)

from ...object_roles import setting_label
from ...object_settings_table import (OBJECT_ORDER, _FILTER_PREFIX,
                                      _SINGLE_OBJECT_QUESTIONS, _filter_text,
                                      _parse_filter_text, _settings_key,
                                      column_label, from_table, to_table,
                                      widen)
from ...organelle_types import MAX_ORGANELLES, organelle_role
from ..theme import SPACING
from .sortable_table import install_sorting

__all__ = ["AUTO_TEXT", "OFF_TEXT", "ObjectSettingsGrid",
           "ObjectSettingsModel"]

#: What an unset value reads as. ``None`` means "work it out" for most of
#: these -- a diameter of None is Cellpose estimating it -- and an empty cell
#: would read as "nobody has filled this in yet", which is a different claim.
AUTO_TEXT = "auto"

#: What an unset CHANNEL reads as, and it is not "auto".
#:
#: ``cell_channel = None`` does not mean spaCR picks a channel. It means no
#: cell masks, no cell table and no cell crops are produced -- the object is
#: not segmented at all. Drawn as "auto" it read as a promise to work
#: something out, which is the opposite of what it does.
OFF_TEXT = "off"

#: Questions whose ``None`` means OFF rather than AUTO.
_OFF_QUESTIONS = frozenset({"channel"})

#: The question whose cells get a model-zoo button, one per object column.
#:
#: A MODEL IS PER OBJECT. Cells and pathogens are not segmented by the same
#: checkpoint, so a single button for the row would be a button that has to
#: ask which column it meant. The button sits in the cell and already knows.
MODEL_QUESTION = "model_name"


def _unset_text(question: str) -> str:
    """How an unset value reads for ``question``."""
    if question in _OFF_QUESTIONS or _is_filter(question):
        return OFF_TEXT
    return AUTO_TEXT


def _is_filter(question: str) -> bool:
    """Whether ``question`` is an object-filter row (``filter:area``)."""
    return str(question).startswith(_FILTER_PREFIX)


def _cell_key(obj: str, question: str) -> str:
    """The settings key a cell edits; ``object_filters`` for a filter row."""
    if _is_filter(question):
        return "object_filters"
    return _settings_key(obj, question)


#: The questions that say whether an object is in the run at all.
#:
#: 2026-09-29 (item 592): not drawn as a table row. A column is hidden while
#: its object's channel is unset, and a hidden column cannot hold the cell
#: that would bring it back, so the channel stays on the ordinary form.
_SWITCH_QUESTIONS = frozenset({"channel", "mask_dim"})


def _names_a_plane(value) -> bool:
    """Whether a channel value names a plane of the stack.

    The form's rule (:func:`spacr.qt.screens.settings_model._names_a_plane`),
    restated so the grid does not import the settings panel: ``None``,
    ``False``, blank and ``"none"`` name no plane; ``0`` is the first plane.

    :param value: the channel setting's value.
    """
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    text = str(value).strip()
    if not text or text.lower() == "none":
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def _object_is_in_the_run(obj: str, settings: Mapping[str, Any]) -> bool:
    """Whether ``obj`` gets a column: cell always, others once a channel is set.

    2026-09-29 (item 592, "hide unset objects"). Cell is the reference
    object and is never gated by its channel, the same rule the flat form
    keeps. An object with no switch in ``settings`` at all (cytoplasm, which
    is derived) cannot be switched off and keeps its column.

    :param obj: an object column, e.g. ``"nucleus"`` or ``"organelleb"``.
    :param settings: the flat settings the table was read from.
    """
    if obj == "cell":
        return True
    switches = [f"{obj}_{question}" for question in sorted(_SWITCH_QUESTIONS)
                if f"{obj}_{question}" in settings]
    if not switches:
        return True
    return any(_names_a_plane(settings.get(key)) for key in switches)


def _filter_objects():
    """The objects an object filter may be set for, in table order.

    Cytoplasm is left out: it is derived from the other masks at measure
    time, and a Mask run filters only the objects it segments.
    """
    return tuple(obj for obj in OBJECT_ORDER if obj != "cytoplasm")


def _question_help(question: str, obj: str) -> str:
    """The settings description behind one cell, or "" when there is none.

    READ FROM :data:`spacr.settings.tooltips`, THE ONE THE FLAT FORM USES.
    A table that wrote its own sentences would be a second set of
    explanations to keep in step with the first, and the two would disagree
    first where nobody was looking. ``descriptions`` is a different dict and
    does not carry these keys -- reading it returned nothing for every row,
    which is a tooltip that silently says the key back.

    Falls back to another object's answer to the SAME question, because the
    row is one question and the flat vocabulary spells it once per object:
    ``cell_channel`` is written up and ``organellec_channel`` is not.
    """
    try:
        from ...settings import tooltips
    except Exception:                                        # noqa: BLE001
        return ""
    return str(tooltips.get(_cell_key(obj, question), "") or "")


def _checked(value) -> bool:
    """Whether a check-state value from a view means checked.

    :param value: a ``Qt.CheckState``, its integer, or a bool.
    """
    if isinstance(value, bool):
        return value
    try:
        return int(getattr(value, "value", value)) == int(
            Qt.CheckState.Checked.value)
    except (TypeError, ValueError):
        return str(value).strip().lower() in ("1", "true", "yes", "on")


def _coerce(text: str, like: Any) -> Any:
    """Read ``text`` back as the type ``like`` already had.

    THE WHOLE REASON THIS FUNCTION EXISTS is that a table hands back strings.
    ``cell_diameter`` is an int and ``organelle_cellprob_threshold`` is a
    float; storing either as ``"12"`` is a settings file that has changed
    meaning without anyone saying so.

    :param text: what the user typed.
    :param like: the value being replaced, whose type is the target. When it
        is ``None`` there is no type to copy -- the caller passes a sibling
        answer from the same row instead, because the same question about a
        different object is the best evidence available about what this one
        is.
    """
    raw = str(text).strip()
    if raw == "" or raw.lower() in (AUTO_TEXT, OFF_TEXT):
        return None
    if isinstance(like, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    for kind in ((int, float) if isinstance(like, (int, float))
                 and not isinstance(like, bool) else ()):
        try:
            return kind(raw)
        except (TypeError, ValueError):
            continue
    if like is None or isinstance(like, str):
        for kind in (int, float):
            try:
                return kind(raw)
            except (TypeError, ValueError):
                continue
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
    return raw


class ObjectSettingsModel(QAbstractTableModel):
    """One row per question, one column per object type.

    A model rather than a widget full of cells because the table is 55 rows
    by as many objects as the run has, and every one of those cells would
    otherwise be a widget the form has to build, lay out and translate.

    :param parent: parent widget.
    """

    #: Emitted when a cell's value actually changed.
    edited = Signal()

    def __init__(self, parent=None):
        """Create the empty per-object settings table model.

        :param parent: parent object, or ``None``.
        """
        super().__init__(parent)
        self._table: Dict[str, Dict[str, Any]] = {}
        self._questions: Tuple[str, ...] = ()
        self._objects: Tuple[str, ...] = ()


    def set_table(self, table: Mapping[str, Mapping[str, Any]]) -> None:
        """Show ``table``, as :func:`spacr.object_settings_table.to_table`
        returns it.

        :param table: ``{question: {object: value}}`` mapping, or ``None`` for
            an empty table. Rows keep its order; columns follow the canonical
            object order.
        """
        self.beginResetModel()
        self._table = {q: dict(row) for q, row in (table or {}).items()}
        self._questions = tuple(self._table)
        order = {name: index for index, name in enumerate(OBJECT_ORDER)}
        present = {obj for row in self._table.values() for obj in row}
        self._objects = tuple(sorted(
            present, key=lambda o: order.get(o, len(order))))
        self.endResetModel()

    def table(self) -> Dict[str, Dict[str, Any]]:
        """The table as it now stands, including every edit."""
        return {q: dict(row) for q, row in self._table.items()}

    def objects(self) -> Tuple[str, ...]:
        """The object columns, in the order they are drawn."""
        return self._objects

    def question_at(self, row: int) -> str:
        """The settings question one row asks, or ``''``.

        :param row: zero-based table row; out of range gives ``''``.
        """
        return self._questions[row] if 0 <= row < len(self._questions) else ""

    def value_at(self, question: str, obj: str) -> Any:
        """One cell's stored value. ``KeyError``-free: absent is ``None``.

        :param question: settings question, i.e. the key suffix shared by every
            object (``"min_area"`` for ``cell_min_area``).
        :param obj: object name, the key prefix (``"cell"``, ``"nucleus"``, ...).
        """
        return self._table.get(question, {}).get(obj)

    def asks(self, question: str, obj: str) -> bool:
        """Whether ``obj`` asks ``question`` at all.

        Absence is a fact about the object, not a value it has yet to be
        given: cytoplasm is derived and has no channel to be found in.

        :param question: settings question, i.e. the key suffix shared by every
            object (``"min_area"`` for ``cell_min_area``).
        :param obj: object name, the key prefix (``"cell"``, ``"nucleus"``, ...).
        """
        return obj in self._table.get(question, {})


    def rowCount(self, parent=QModelIndex()) -> int:
        """How many questions the table asks.

        :param parent: unused; the model is flat.
        :returns: the row count.
        """
        return 0 if parent.isValid() else len(self._questions)

    def columnCount(self, parent=QModelIndex()) -> int:
        """How many objects the table has a column for.

        :param parent: unused; the model is flat.
        :returns: the column count.
        """
        return 0 if parent.isValid() else len(self._objects)

    def flags(self, index):
        """Which cells are editable.

        ONLY THE CELLS AN OBJECT ACTUALLY ASKS. A blank cell means that
        object does not ask that question, and making it editable would
        invite an answer to a question nobody posed.

        :param index: the cell.
        :returns: the Qt item flags.
        """
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if not index.isValid():
            return base
        question = self.question_at(index.row())
        obj = self._objects[index.column()]
        if not self.asks(question, obj):
            return Qt.ItemIsSelectable
        if isinstance(self.value_at(question, obj), bool):
            return base | Qt.ItemIsEditable | Qt.ItemIsUserCheckable
        return base | Qt.ItemIsEditable

    def data(self, index, role=Qt.DisplayRole):
        """One cell of the table.

        :param index: the cell.
        :param role: the Qt display role.
        :returns: the cell's value for that role, or None.
        """
        if not index.isValid():
            return None
        question = self.question_at(index.row())
        obj = self._objects[index.column()]
        if not self.asks(question, obj):
            if role == Qt.ToolTipRole:
                return (f"{column_label(obj)} does not ask this. It is not a "
                        f"value waiting to be filled in.")
            return None
        value = self.value_at(question, obj)
        unset = _unset_text(question)
        if role == Qt.CheckStateRole and isinstance(value, bool):
            return Qt.Checked if value else Qt.Unchecked
        if role == Qt.DisplayRole:
            return unset if value is None else str(value)
        if role == Qt.EditRole:
            return "" if value is None else str(value)
        if role == Qt.ToolTipRole:
            head = (f"{_cell_key(obj, question)}  =  "
                    f"{unset if value is None else value!r}")
            if _is_filter(question):
                head += (f"\n\nKeep only {column_label(obj)} objects whose "
                         f"{question[len(_FILTER_PREFIX):]} is within "
                         f"min \u2013 max. Type one number for a minimum "
                         f"alone, \u2013 then a number for a maximum alone, "
                         f"or clear the cell to switch the filter off.")
            if question == MODEL_QUESTION:
                head += (f"\n\nClick this cell to choose {column_label(obj)}'s "
                         f"model from the zoo. Double-click to type a path.")
            help_text = _question_help(question, obj)
            if question in _OFF_QUESTIONS and value is None:
                head += (f"\n\n{column_label(obj)} is NOT SEGMENTED. "
                         f"Give it a channel number to turn it on.")
            return f"{head}\n\n{help_text}" if help_text else head
        return None

    def setData(self, index, value, role=Qt.EditRole) -> bool:
        """Write one cell back into the settings.

        :param index: the cell.
        :param value: what the user typed.
        :param role: the Qt edit role.
        :returns: True when the value was taken.
        """
        if not index.isValid() or role not in (Qt.EditRole,
                                               Qt.CheckStateRole):
            return False
        question = self.question_at(index.row())
        obj = self._objects[index.column()]
        if not self.asks(question, obj):
            return False
        row = self._table[question]
        current = row.get(obj)
        if _is_filter(question):
            try:
                low, high = _parse_filter_text(value)
            except ValueError:
                return False
            new = _filter_text({"min": low, "max": high})
            if new == current:
                return False
            row[obj] = new
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            self.edited.emit()
            return True
        if role == Qt.CheckStateRole:
            value = "true" if _checked(value) else "false"
        like = current
        if like is None:
            like = next((v for o, v in row.items()
                         if o != obj and v is not None), None)
        new = _coerce(value, like)
        if new == current and type(new) is type(current):
            return False
        row[obj] = new
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
        self.edited.emit()
        return True

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """One header label: a question name, or an object name.

        :param section: the row or column number.
        :param orientation: which header.
        :param role: the Qt display role.
        :returns: the label, or None.
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return column_label(self._objects[section])
            question = self._questions[section]
            if _is_filter(question):
                return (f"Filter: {question[len(_FILTER_PREFIX):]} "
                        f"(min \u2013 max)")
            return setting_label(question)
        if role == Qt.ToolTipRole:
            if orientation == Qt.Vertical:
                return None
            obj = self._objects[section]
            return (f"{column_label(obj)}. Every row below asks this object "
                    f"the question on the left; a blank cell is a question "
                    f"it does not ask.")
        return None


class _GridHeightGrip(QFrame):
    """Thin drag handle along the per-object table's lower edge.

    The table is one row of a scrolling settings form, so without this it
    gets whatever height the form gives it and puts twenty-odd questions
    behind an inner scrollbar inside an outer one. Dragging this sets the
    height; double-clicking gives it back to the content.

    :param grid: the :class:`ObjectSettingsGrid` this resizes. ALSO ITS
        QWIDGET PARENT, so the grip is laid out under the table it drags and
        cannot outlive it.
    """

    HEIGHT = 7

    def __init__(self, grid: "ObjectSettingsGrid"):
        """Build the handle and give it a vertical-resize cursor."""
        super().__init__(grid)
        self._grid = grid
        self._press_y: Optional[float] = None
        self._start_height = 0
        self.setObjectName("ConsoleSectionResizeHandle")
        self.setCursor(Qt.SizeVerCursor)
        self.setFixedHeight(self.HEIGHT)
        source = ("Drag to make the table taller or shorter. "
                  "Double-click to fit its rows.")
        self.setProperty("_spacr_i18n_tooltip", source)
        self.setToolTip(source)

    def sizeHint(self) -> QSize:
        """Wide and thin -- the handle is an edge, not a bar."""
        return QSize(80, self.HEIGHT)

    def mousePressEvent(self, event) -> None:            # noqa: N802
        """Remember where the drag started, and from what height."""
        if event.button() == Qt.LeftButton:
            self._press_y = event.globalPosition().y()
            self._start_height = self._grid._table.height()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:             # noqa: N802
        """Resize the table by how far the pointer has moved since the press.

        Measured from the PRESS rather than the last move, so a drag that
        outruns the redraw lands where the pointer is instead of accumulating
        rounding.
        """
        if self._press_y is not None and event.buttons() & Qt.LeftButton:
            delta = event.globalPosition().y() - self._press_y
            self._grid.set_user_height(self._start_height + int(delta))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:          # noqa: N802
        """End the drag."""
        self._press_y = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:      # noqa: N802
        """Give the height back to the content."""
        if event.button() == Qt.LeftButton:
            self._grid.reset_user_height()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


def _kind_of(obj: str) -> str:
    """The KIND of object ``obj`` is, collapsing every organelle into one.

    `organelle_role_of` answers which SLOT a name is -- ``'organelleb'`` --
    which is what the settings keys need and the wrong grain for asking
    whether a question is per-object. Two organelles are one kind of thing
    asked twice.
    """
    from ...organelle_types import organelle_role_of
    return "organelle" if organelle_role_of(obj) else obj


def _parsed_filters(raw) -> Dict[str, Any]:
    """``object_filters`` as a mapping, or empty when it cannot be read.

    :param raw: the setting's value, a mapping or its JSON/literal text.
    """
    try:
        from ..mask_engine import parse_object_filters

        return parse_object_filters(raw)
    except Exception:                                        # noqa: BLE001
        return {}


class ObjectSettingsGrid(QWidget):
    """The per-object settings table, and the button that widens it.

    :param parent: parent widget.
    """

    #: Emitted when any cell changed, or a column was added.
    settings_changed = Signal()

    def __init__(self, parent=None):
        """Build the per-object settings grid.

        Sorting is installed after the model is set, as the contract requires --
        the view is wrapped in a proxy, so the selection model has to be taken
        afterwards. Sorting the questions on screen reorders nothing on disk,
        because the stored answers are read from the model rather than the view.

        The table opens tall enough to show its rows and can be dragged from the
        grip: inside a settings panel it is one row of a scrolling form, and a
        plain ``QTableView`` default put twenty-odd questions behind an inner
        scrollbar inside an outer one.

        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self._base: Dict[str, Any] = {}
        self._model = ObjectSettingsModel(self)
        self._model.edited.connect(self.settings_changed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["sm"])

        from .hover_tooltip import (ANIMATION_MARK, API_MARK, PURPLE, TEAL,
                                    _AnimationView, _LinkWord)

        self._help_band = QWidget(self)
        band = QHBoxLayout(self._help_band)
        band.setContentsMargins(0, 0, 0, 0)
        band.setSpacing(SPACING["sm"])

        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        self._help = QLabel("", self._help_band)
        self._help.setObjectName("SubtitleSmall")
        self._help.setStyleSheet("background: transparent;")
        self._help.setWordWrap(True)
        self._help.setTextFormat(Qt.TextFormat.RichText)
        self._help.setAlignment(Qt.AlignmentFlag.AlignLeft
                                | Qt.AlignmentFlag.AlignTop)
        self._help.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Fixed)
        column.addWidget(self._help)

        self._help_links = QWidget(self._help_band)
        links = QHBoxLayout(self._help_links)
        links.setContentsMargins(0, 0, 0, 0)
        links.setSpacing(SPACING["sm"])
        self._help_api = _LinkWord(API_MARK, "HoverTooltipApiLink",
                                   self._help_links)
        self._help_api.setAccessibleName("API")
        self._help_api.setAccessibleDescription(
            "Open spaCR API documentation for this setting.")
        self._help_api.clicked.connect(self._open_help_api)
        self._help_anim = _LinkWord(ANIMATION_MARK,
                                    "HoverTooltipAnimationLink",
                                    self._help_links)
        self._help_anim.setAccessibleName("Animation")
        self._help_anim.setAccessibleDescription(
            "Show or hide this setting's animation.")
        self._help_anim.clicked.connect(self._toggle_help_animation)
        links.addWidget(self._help_api)
        links.addWidget(self._help_anim)
        links.addStretch(1)
        self._help_links.setStyleSheet(
            f"QLabel#HoverTooltipApiLink {{ color: {TEAL};"
            f" text-decoration: none; }}"
            f"QLabel#HoverTooltipAnimationLink {{ color: {PURPLE};"
            f" text-decoration: none; }}")
        column.addWidget(self._help_links)
        column.addStretch(1)
        band.addLayout(column, 1)

        self._help_animation = _AnimationView(self.HELP_ANIMATION_PX,
                                              self._help_band)
        self._help_animation.hide()
        band.addWidget(self._help_animation, 0,
                       Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self._help_band)

        #: The animation offered for the hovered setting, and whether the
        #: reader has asked to see it. Local to the band: pressing
        #: **Animation** names ONE setting, exactly as the popup's does.
        self._help_offered_animation = None
        self._help_animation_shown = False
        self._help_api_url = ""

        self._table = QTableView(self)
        self._table.setModel(self._model)
        install_sorting(self._table)
        self._table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Fixed)
        self._table.clicked.connect(self._cell_clicked)
        #: Which module's API the tooltips link to. Set by the screen that
        #: mounts the grid -- the table itself has no way to know, and a
        #: guess would send the reader to another module's page.
        self._app_key = ""
        self._hovered_key = ""
        self._table.setMouseTracking(True)
        self._table.viewport().setMouseTracking(True)
        self._table.viewport().installEventFilter(self)
        self._user_height: Optional[int] = None
        outer.addWidget(self._table)

        self._grip = _GridHeightGrip(self)
        outer.addWidget(self._grip)

        row = QHBoxLayout()
        row.setSpacing(SPACING["sm"])
        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        self._add = QPushButton("Add an organelle", self)
        self._add.setToolTip(
            "One more organelle is one more COLUMN. In the flat settings "
            "vocabulary it was twenty new settings, which is why the count "
            "could not be arbitrary before this table existed.")
        self._add.clicked.connect(self.add_organelle)
        self._add_filter = QPushButton("Add a filter", self)
        self._add_filter.setObjectName("ObjectGridAddFilter")
        self._add_filter.setToolTip(
            "Add a row that keeps only the objects whose measurement -- "
            "area, mean intensity, solidity or any other scalar region "
            "property -- falls between a minimum and a maximum. Fill in the "
            "column of every object it should apply to; a blank cell leaves "
            "that object unfiltered. Default no filters.")
        self._add_filter.clicked.connect(self._offer_filters)
        row.addWidget(self._status, 1)
        row.addWidget(self._add_filter)
        row.addWidget(self._add)
        outer.addLayout(row)
        #: Filter rows the user added and has not filled in yet, which the
        #: settings alone cannot show because an empty filter is no entry.
        self._added_filters: list = []
        #: The table with EVERY object's column, before the objects whose
        #: channel is unset are hidden; what the grid claims from the form.
        self._claimed: Dict[str, Dict[str, Any]] = {}

        #: Fires once the pointer has rested on a cell long enough.
        from ..tooltip_policy import HoverDelay
        self._help_hover_delay = HoverDelay(self)
        self._help_show_timer = self._help_hover_delay._timer
        self._help_pending = ""
        #: Fires after the pointer has left, unless it came back.
        self._help_hide_timer = QTimer(self)
        self._help_hide_timer.setSingleShot(True)
        self._help_hide_timer.timeout.connect(lambda: self._write_help(""))
        self._help_band.installEventFilter(self)

        self._sync_help_height()
        self._write_help("")
        self.installEventFilter(self)


    #: Lines reserved above the table for a setting's help.
    #:
    #: FIVE, and fixed. Three was the first value and it was not enough:
    #: a setting's help is a paragraph, and the longer ones were being cut
    #: off. The band has to be tall enough for the longest help a cell can
    #: show WITHOUT the label growing when it arrives -- growing would push
    #: the table down under the pointer mid-hover and move the cell out
    #: from under it, which is the failure the fixed band exists to
    #: prevent in the first place. So the cost of being too short is text
    #: nobody can read, and the cost of being too tall is a little space:
    #: the second is the mistake worth making.
    HELP_LINES = 5

    #: Side of the animation square in the band, in pixels.
    #:
    #: The popup uses 220. This is space reserved ABOVE the table for the
    #: life of the panel, and the band may not grow when an animation
    #: arrives -- growing would push the table down under the pointer that
    #: asked for it. So the square is sized to what the band can afford
    #: rather than the band to the square.
    HELP_ANIMATION_PX = 132

    #: How long the last help stays after the pointer leaves, in ms.
    #:
    #: The band carries an API link and an Animation word, and a reader
    #: has to be able to reach them. Clearing on `Leave` put the words
    #: under a pointer that was travelling towards them and then took them
    #: away. The popup solves the same problem the same way; this is its
    #: HIDE_DELAY_MS.
    HELP_HIDE_DELAY_MS = 700

    def _sync_help_height(self) -> None:
        """Reserve :data:`HELP_LINES` using the font Qt is actually painting.

        Measured from the polished widget rather than from the theme's
        point size, because a stylesheet or the platform can change what is
        painted and a height computed from the wrong font reserves the
        wrong number of lines.
        """
        self._help.ensurePolished()
        lines = self._help.fontMetrics().lineSpacing() * self.HELP_LINES
        self._help.setFixedHeight(lines)
        self._help_band.setFixedHeight(max(lines, self.HELP_ANIMATION_PX))

    def set_app_key(self, app_key: str) -> None:
        """Say which module's API documentation the tooltips should link to.

        :param app_key: registry key of the module, e.g. ``"mask"``; ``None``
            or empty clears it.
        """
        self._app_key = str(app_key or "")

    def _key_under(self, pos) -> str:
        """The settings key of the cell at ``pos``, or ``""``."""
        index = self._table.indexAt(pos)
        if not index.isValid():
            return ""
        mapper = getattr(self._table.model(), "mapToSource", None)
        source = mapper(index) if mapper is not None else index
        if not source.isValid():
            return ""
        question = self._model.question_at(source.row())
        objects = self._model.objects()
        if not question or source.column() >= len(objects):
            return ""
        obj = objects[source.column()]
        return (_cell_key(obj, question) if self._model.asks(question, obj)
                else "")

    def eventFilter(self, watched, event):               # noqa: N802
        """Show the same sticky, linked tooltip the form shows.

        THE SAME POPUP, NOT A SECOND ONE. The flat form puts rich help on a
        widget and lets `HoverTooltip` draw it: a typed body, an API link,
        and the setting's animation when it has one. A table has no widget
        per cell, so the anchor is the view and the cell under the pointer
        decides which setting it is speaking for.

        The native tooltip is swallowed for the same reason the form
        swallows it -- it disappears the moment the pointer moves toward the
        API link, and that link is the point.

        :param watched: the object the filter is installed on: this widget
            (font changes), the help band (enter and leave) or the table's
            viewport (tooltip, mouse move and leave).
        :param event: the event; a tooltip event on the viewport is swallowed
            and every other event is passed on to the base class.
        """
        try:
            kind = event.type()
            if watched is self:
                if kind == QEvent.Type.FontChange:
                    self._sync_help_height()
                return super().eventFilter(watched, event)
            if watched is self._help_band:
                if kind == QEvent.Type.Enter:
                    self._help_hide_timer.stop()
                elif kind == QEvent.Type.Leave:
                    self._help_hide_timer.start(self.HELP_HIDE_DELAY_MS)
                return super().eventFilter(watched, event)
            if kind == QEvent.Type.ToolTip:
                return True
            if kind == QEvent.Type.MouseMove:
                self._offer_tooltip(event.position().toPoint())
            elif kind == QEvent.Type.Leave:
                self._hovered_key = ""
                self._help_hover_delay.cancel()
                self._help_hide_timer.start(self.HELP_HIDE_DELAY_MS)
        except Exception:                                    # noqa: BLE001
            LOG.debug("the table could not offer its tooltip", exc_info=True)
        return super().eventFilter(watched, event)

    def _offer_tooltip(self, pos) -> None:
        """Show help for the cell under ``pos``, if it is a new cell.

        RE-SHOWN ONLY ON A NEW CELL. `show_for` re-anchors and restarts the
        popup, so calling it on every mouse-move would rebuild the tooltip
        dozens of times a second and make its links unclickable.
        """
        key = self._key_under(pos)
        if key == self._hovered_key:
            return
        self._hovered_key = key
        self._help_hide_timer.stop()
        if not key:
            self._help_hover_delay.cancel()
            self._help_hide_timer.start(self.HELP_HIDE_DELAY_MS)
            return
        self._help_pending = key
        self._help_hover_delay.schedule(
            self._table.viewport(), self._show_pending_help)

    def _show_pending_help(self) -> None:
        """Write the help for the cell the pointer settled on."""
        key = self._help_pending
        if not key:
            return
        try:
            from ..screens.settings_model import format_tooltip, get_tooltips
            body = str(get_tooltips().get(key) or "")
            html = format_tooltip(body, self._app_key, key)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not build the tooltip for %s", key, exc_info=True)
            return
        self._write_help(html, key=key)

    def _open_help_api(self) -> None:
        """Open the documentation page the band's **API** word points at."""
        if not self._help_api_url:
            return
        try:
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl(self._help_api_url))
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not open %s", self._help_api_url, exc_info=True)

    def _toggle_help_animation(self) -> None:
        """Show or fold away the square beside the text.

        Names ONE setting, like the popup's word: moving to another cell
        falls back to the preference rather than carrying the reveal.
        """
        self._help_animation_shown = not self._help_animation_shown
        self._apply_help_animation()

    def _apply_help_animation(self) -> None:
        """Draw, pause or drop the square for the offered animation."""
        animation = self._help_offered_animation
        showing = False
        if animation is not None and self._help_animation_shown:
            showing = bool(self._help_animation.load(animation))
        else:
            self._help_animation.clear_animation()
        self._help_animation.setVisible(showing)
        self._help_anim.setVisible(
            animation is not None and (showing or not self._help_animation_shown))
        self._help_links.setVisible(
            self._help_api.isVisibleTo(self._help_links)
            or self._help_anim.isVisibleTo(self._help_links))

    def _write_help(self, html: str, key: str = "") -> None:
        """Put ``html`` in the band above the table, or the resting prompt.

        The API link is kept. `format_tooltip` ends the body with an anchor
        to the setting's documentation page, and the popup used to lift that
        out into its own **API** word; here the band is a rich-text label
        with external links enabled, so the anchor works where it already
        is and there is nothing to lift.

        The band does not go blank between cells. Moving the pointer across
        a row would otherwise flicker it empty and back, and an empty band
        is also what a reader sees before touching anything -- so the rest
        state is a sentence saying what the band is for. That sentence is
        the one the bottom-of-window strip already uses, reused rather than
        written again: a new one would be a new user-facing string in ten
        languages.
        """
        from ..i18n import tr
        from .hover_tooltip import split_api_link

        text = str(html or "").strip()
        if not text:
            text = tr("Hover any setting for details and a link to its "
                      "documentation.")
            body, url = text, ""
        else:
            body, url = split_api_link(text)
        self._help.setText(body)
        self._help_api_url = url
        self._help_api.setVisible(bool(url))

        animation = None
        if key:
            try:
                from ...setting_animations import animation_for_setting
                animation = animation_for_setting(key)
            except Exception:                                # noqa: BLE001
                LOG.debug("no animation lookup for %s", key, exc_info=True)
        if animation is not self._help_offered_animation:
            self._help_animation_shown = False
        self._help_offered_animation = animation
        self._apply_help_animation()


    def _cell_clicked(self, index) -> None:
        """Open the model zoo when a model-name cell is clicked.

        NO BUTTON, AND NOTHING DRAWN. A button small enough to fit in a table
        cell has no room for a word, so the whole cell is the control: click
        anywhere in the model row, under the object you mean, and the picker
        opens for that object.

        The cell is still editable by double-click, which is how a path that
        is not in the zoo gets typed.
        """
        try:
            source = index
            mapper = getattr(self._table.model(), "mapToSource", None)
            if mapper is not None:
                source = mapper(index)
            if not source.isValid():
                return
            if self._model.question_at(source.row()) != MODEL_QUESTION:
                return
            obj = self._model.objects()[source.column()]
            if self._model.asks(MODEL_QUESTION, obj):
                self.choose_model_for(obj)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not open the model zoo", exc_info=True)

    #: Kinds the per-object Model cell offers.
    #:
    #: ``cellpose3`` is in the list because this cell writes
    #: ``cell_model_name`` / ``nucleus_model_name`` / ``pathogen_model_name``,
    #: and those are exactly the settings
    #: :func:`spacr.settings._get_object_settings` reads when
    #: ``segmentation_backend`` is ``'cellpose3'`` -- so cyto3, cyto2, cyto,
    #: nuclei and any bioimage.io Cellpose 3 checkpoint belong here. Without
    #: it the four stock Cellpose 3 models were listed nowhere a button opens
    #: and had to be typed by hand. The Cellpose 4 preview boxes keep
    #: ``("cellpose",)``: they load the checkpoint in spaCR's own process.
    #: A Cellpose-DINO checkpoint runs in its own backend (item 525), which
    #: Mask generation reaches, so ``cellpose_dino`` belongs here too, and
    #: so do StarDist's, InstanSeg's and Omnipose's models, which the button
    #: adds from :data:`spacr.model_zoo.PREFIXED_KINDS` (items 551-553).
    MODEL_KINDS = ("cellpose", "cellpose3", "cellpose_dino")

    def choose_model_for(self, obj: str) -> bool:
        """Open the model zoo for one object and store what it returns.

        :param obj: object name whose ``model_name`` cell receives the chosen
            path, e.g. ``"cell"``.
        :returns: True when a model was chosen. Cancelling leaves the cell
            alone rather than clearing it -- a cancelled dialog is not an
            instruction to forget the model already set.
        """
        from ... import model_zoo
        from .model_zoo_picker import choose_model

        path = choose_model(self, kinds=self.MODEL_KINDS + tuple(
            kind for kind in model_zoo.PREFIXED_KINDS
            if kind not in self.MODEL_KINDS))
        if not path:
            return False
        return self.set_value(MODEL_QUESTION, obj, path)


    #: Never shorter than this, however few rows there are: a table that
    #: collapses to its header is one the user cannot grab to make bigger.
    MIN_TABLE_H = 90
    #: How tall it opens at most. Past this the form is one long table and
    #: the settings above and below it stop being findable -- the grip is
    #: there for anyone who wants more.
    AUTO_TABLE_H = 420

    def content_height(self) -> int:
        """The height that would show every row without an inner scrollbar."""
        header = self._table.horizontalHeader().height()
        rows = sum(self._table.rowHeight(r)
                   for r in range(self._model.rowCount()))
        return header + rows + 2 * self._table.frameWidth()

    def set_user_height(self, height: int) -> None:
        """Fix the table at ``height`` px, clamped to at least MIN_TABLE_H.

        :param height: wanted table height in pixels.
        """
        self._user_height = max(self.MIN_TABLE_H, int(height))
        self._apply_height()

    def reset_user_height(self) -> None:
        """Forget a dragged height and go back to fitting the rows."""
        self._user_height = None
        self._apply_height()

    def _apply_height(self) -> None:
        """Put the chosen height on the table.

        ``setFixedHeight`` rather than a minimum, because the table sits in a
        form that would otherwise stretch it: the point of the grip is that
        the height is the USER's answer, and a layout free to grow it is a
        layout that overrules them.
        """
        if self._user_height is not None:
            self._table.setFixedHeight(self._user_height)
            return
        fit = self.content_height()
        self._table.setFixedHeight(
            max(self.MIN_TABLE_H, min(self.AUTO_TABLE_H, fit)))


    def set_settings(self, settings: Mapping[str, Any]) -> None:
        """Show the per-object half of a flat settings dict.

        The rest is KEPT, not dropped: :meth:`settings` returns it unchanged
        beside the table's own keys, so this widget can edit a corner of a
        settings file without holding the whole of it hostage.

        :param settings: flat settings dict, or ``None``; its
            ``<object>_<question>`` keys become the table.
        """
        self._base = dict(settings or {})
        self._model.set_table(self._visible_table())
        self._announce()

    def _visible_table(self) -> Dict[str, Dict[str, Any]]:
        """The table as drawn: the claimed table less the objects the run lacks.

        2026-09-29 (item 592, the decision "hide unset
        objects"): a column is drawn only for an object whose channel names
        a plane, and cell always (see :func:`_object_is_in_the_run`). The
        channel row itself is not drawn -- a hidden object has no column to
        hold it -- so every object's channel stays a row of the ordinary
        form, where it can always be set. HIDDEN, NEVER DELETED: the hidden
        columns' answers stay in ``self._base``, which :meth:`settings`
        writes back unchanged, so setting the channel again brings the
        column back with them.
        """
        full = self._every_column_table()
        self._claimed = full
        shown = {obj for row in full.values() for obj in row
                 if _object_is_in_the_run(obj, self._base)}
        table: Dict[str, Dict[str, Any]] = {}
        for question, row in full.items():
            if question in _SWITCH_QUESTIONS:
                continue
            kept = {obj: value for obj, value in row.items() if obj in shown}
            if kept:
                table[question] = kept
        table.update(self._filter_rows(shown))
        return table

    def _claimed_objects(self) -> Tuple[str, ...]:
        """Every object column the table holds, drawn or hidden, in order."""
        order = {name: index for index, name in enumerate(OBJECT_ORDER)}
        present = {obj for row in self._claimed.values() for obj in row}
        return tuple(sorted(present, key=lambda o: order.get(o, len(order))))

    def _claimed_table(self) -> Dict[str, Dict[str, Any]]:
        """Every object's answers the table holds, drawn or hidden.

        The columns of objects whose channel is unset are in here although
        they are not on screen, and so is the channel row, which is never
        drawn. The binding claims its keys from this, so a hidden object's
        settings stay off the flat form too.
        """
        return {q: dict(row) for q, row in self._claimed.items()}

    def _shows_the_same_objects_as(self, settings: Mapping[str, Any]) -> bool:
        """Whether ``settings`` would draw the columns drawn now.

        Cheap enough to ask on every change of a channel field: only the
        channel values are read, and the table is rebuilt only when this
        says no.

        :param settings: the flat settings as the form now holds them.
        """
        objects = {obj for row in self._claimed.values() for obj in row}
        wanted = {obj for obj in objects
                  if _object_is_in_the_run(obj, settings)}
        return wanted == set(self._model.objects())

    def _keep_the_switches(self, settings: Mapping[str, Any]) -> None:
        """Hold the form's channel values without redrawing anything.

        The channels are not cells, so a channel moved between two planes
        changes no column; it is kept so that :meth:`settings` hands back
        what the form holds rather than the channel of the last redraw.

        :param settings: the flat settings as the form now holds them.
        """
        for question in _SWITCH_QUESTIONS:
            for obj in self._claimed.get(question, {}):
                key = _settings_key(obj, question)
                if key in settings:
                    self._base[key] = settings[key]

    def _every_column_table(self) -> Dict[str, Dict[str, Any]]:
        """The table with the organelle slots the count does not ask for cut.

        `number_of_organelles` IS THE SOURCE OF TRUTH FOR THE COLUMNS. The
        settings dict keeps a typed placeholder for every slot up to the
        maximum -- that is what makes lowering the count reversible -- so a
        table built straight off the keys shows an Organelle 1 column at a
        count of zero, which is a column for something the run will not
        segment.

        CUT FROM THE VIEW, NOT FROM THE SETTINGS. `settings()` writes the
        table back over ``self._base``, so a hidden slot's keys are carried
        through untouched and raising the count again brings its answers
        back rather than a row of defaults.
        """
        from ...organelle_types import active_organelle_roles

        live = tuple(active_organelle_roles(self._base))
        keep = set(live)
        table = {
            question: {
                obj: value for obj, value in row.items()
                if not obj.startswith("organelle") or obj in keep
            }
            for question, row in to_table(self._base).items()
        }
        for index, role in enumerate(live):
            if any(role in row for row in table.values()):
                continue
            table = widen(table, role,
                          like=live[index - 1] if index else None)
        return self._only_the_shared_questions(table)

    def _filter_rows(self, objects) -> Dict[str, Dict[str, Any]]:
        """``object_filters`` as table rows, one per filtered property.

        Rows appear for every property any shown object filters on, and for
        each property added with :meth:`add_filter` and not yet filled in.
        Every segmented object on screen has a cell in each row.

        :param objects: the object columns the table shows.
        """
        if "object_filters" not in self._base:
            return {}
        filters = _parsed_filters(self._base.get("object_filters"))
        columns = [obj for obj in _filter_objects() if obj in objects]
        order = []
        for obj in columns:
            for entry in filters.get(obj) or ():
                name = str(entry.get("property", ""))
                if name and name not in order:
                    order.append(name)
        for name in self._added_filters:
            if name not in order:
                order.append(name)
        rows: Dict[str, Dict[str, Any]] = {}
        for name in order:
            row = {}
            for obj in columns:
                cell = None
                for entry in filters.get(obj) or ():
                    if str(entry.get("property", "")) == name:
                        cell = _filter_text(entry)
                        break
                row[obj] = cell
            rows[f"{_FILTER_PREFIX}{name}"] = row
        return rows

    @staticmethod
    def _only_the_shared_questions(
            table: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Drop the questions only ONE object asks.

        A TABLE IS A CLAIM THAT ITS ROWS AND COLUMNS ARE INDEPENDENT, and a
        question a single object asks is not a row -- it is one setting, and
        it belongs in the form beside that object's others. The organelle's
        own detection settings are 37 of these: ridge filters, hysteresis,
        LoG sigmas, which no cell, nucleus or pathogen has.

        MORE THAN ONE, NOT ALL. This required EVERY object to ask, and that
        was wrong in a way only a real settings dict shows: Measure asks
        `mask_dim` of cell, nucleus, pathogen and organelle but not
        cytoplasm, and `max_size` of three of its five objects -- so
        requiring all five dropped EVERY per-object setting Measure has and
        left the table empty. A blank cell already means "this object does
        not ask", and the model draws it as not-editable, so a question
        three objects share is a perfectly good row.

        ROLES, NOT COLUMNS. Every organelle counts once between them. The
        obvious reading -- more than one COLUMN asks -- makes the row set
        depend on how many organelles there are, because a second organelle
        is seeded from the first and so doubles every question only an
        organelle asked. Adding a column would then silently add 34 rows.
        A question is per-object because DIFFERENT KINDS of object ask it,
        and two organelles are the same kind asked twice.

        THEY ARE NOT LOST. The grid claims only the keys it shows, so
        everything dropped here stays in the ordinary form under the
        category it belongs to -- which is where a setting only one object
        has belongs, next to the others that object has.
        """
        return {
            question: row for question, row in table.items()
            if len({_kind_of(obj) for obj in row}) > 1
            or question in _SINGLE_OBJECT_QUESTIONS
        }

    def settings(self) -> Dict[str, Any]:
        """The whole settings dict, with the table's answers written back.

        The filter rows are written back into ``object_filters``: each shown
        object's list is rebuilt from its cells, in row order, and an object
        the table does not show keeps the list it had.
        """
        table = self._model.table()
        plain = {q: row for q, row in table.items() if not _is_filter(q)}
        out = from_table(plain, self._base)
        if "object_filters" not in self._base:
            return out
        filters = {obj: [dict(entry) for entry in entries or ()]
                   for obj, entries in _parsed_filters(
                       self._base.get("object_filters")).items()}
        shown = [obj for obj in _filter_objects()
                 if obj in set(self._model.objects())]
        for obj in shown:
            rows = []
            for question, row in table.items():
                if not _is_filter(question):
                    continue
                low, high = _parse_filter_text(row.get(obj))
                if low is None and high is None:
                    continue
                rows.append({"property": question[len(_FILTER_PREFIX):],
                             "min": low, "max": high})
            if rows:
                filters[obj] = rows
            else:
                filters.pop(obj, None)
        out["object_filters"] = filters
        return out

    def filter_properties(self) -> Tuple[str, ...]:
        """The properties the table has a filter row for, in order."""
        return tuple(q[len(_FILTER_PREFIX):] for q in self._model.table()
                     if _is_filter(q))

    def add_filter(self, name: str) -> bool:
        """Add a filter row for regionprop ``name``, every cell empty.

        :param name: a scalar scikit-image regionprop, e.g. ``"area"`` or
            ``"intensity_mean"``; legacy spellings are accepted.
        :returns: False when the name is not a filterable property or the row
            is already there.
        """
        try:
            from ..mask_engine import canonical_property

            name = canonical_property(name)
        except Exception:                                    # noqa: BLE001
            self._status.setText(
                f"'{name}' is not a region property an object filter can "
                f"use.")
            return False
        if name in self.filter_properties():
            return False
        self._base = self.settings()
        self._added_filters.append(name)
        self._model.set_table(self._visible_table())
        self._announce()
        return True

    def set_filters(self, value) -> None:
        """Show ``object_filters`` as it now stands, e.g. after a file load.

        :param value: the setting's value, a mapping or its text.
        """
        self._base = self.settings()
        self._base["object_filters"] = value
        self._model.set_table(self._visible_table())
        self._announce()

    def _offer_filters(self) -> None:
        """Open the list of properties a new filter row can measure."""
        from PySide6.QtWidgets import QMenu

        try:
            from ..mask_engine import filter_properties
            names = filter_properties(intensity=True)
        except Exception:                                    # noqa: BLE001
            LOG.debug("no filter catalogue", exc_info=True)
            names = ("area", "intensity_mean")
        menu = QMenu(self)
        menu.setObjectName("ObjectGridFilterMenu")
        first = [n for n in ("area", "intensity_mean") if n in names]
        for name in first + [n for n in names if n not in first]:
            action = menu.addAction(name)
            action.triggered.connect(
                lambda _checked=False, n=name: self._added(n))
        menu.popup(self._add_filter.mapToGlobal(
            self._add_filter.rect().bottomLeft()))

    def _added(self, name: str) -> None:
        """Add the chosen filter row and tell the form."""
        if self.add_filter(name):
            self.settings_changed.emit()

    def table(self) -> Dict[str, Dict[str, Any]]:
        """The table itself, for a caller that wants the shape."""
        return self._model.table()

    def objects(self) -> Tuple[str, ...]:
        """Which object columns are on screen."""
        return self._model.objects()

    def questions(self) -> Tuple[str, ...]:
        """Which questions are on screen, in order."""
        return tuple(self._model.table())


    def next_organelle(self) -> str:
        """The role the next organelle column would take, or ``''``.

        Empty at the ceiling. Slot names are lettered -- an object type is
        embedded in an underscore-separated object key, so a digit would be
        ambiguous against the object LABEL -- and the lettering CARRIES past
        ``z``, so the ceiling is where two letters run out rather than one.

        COUNTS UP FROM THE SLOTS IN USE rather than from one. Walking every
        slot from the start was fine while there were 26; there are now 702,
        and the caller that presses Add repeatedly turned an O(slots) scan
        into an O(slots squared) one.

        Hidden slots count as in use too.
        """
        used = {obj for row in self._claimed.values() for obj in row
                if obj.startswith("organelle")}
        for number in range(len(used) + 1, MAX_ORGANELLES + 1):
            role = organelle_role(number)
            if role not in used:
                return role
        return ""

    def add_organelle(self) -> bool:
        """Add the next organelle column, seeded from the first one.

        :returns: False when there is no slot left, with the reason on screen
            rather than as an exception into a GUI slot.

        A slot whose channel is unset is hidden.
        """
        from ...organelle_types import NUMBER_OF_ORGANELLES, organelle_count

        role = self.next_organelle()
        if not role:
            self._status.setText(
                f"{MAX_ORGANELLES} organelles is the ceiling: the slots are "
                f"lettered and carry past 'z', so that is where two "
                f"letters run out.")
            return False
        self._base = self.settings()
        self._base[NUMBER_OF_ORGANELLES] = organelle_count(self._base) + 1
        full = self._every_column_table()
        if not any(role in row for row in full.values()):
            previous = [o for o in self._claimed_objects()
                        if o.startswith("organelle")]
            full = widen(full, role, like=previous[-1] if previous else None)
            self._base = from_table(full, self._base)
        self._model.set_table(self._visible_table())
        self._announce()
        if role not in self.objects():
            self._status.setText(
                f"{column_label(role)} added. Give it a channel to show its "
                f"column.")
        self.settings_changed.emit()
        return True

    def _announce(self) -> None:
        """Say what the table is holding, and re-fit its height.

        Called after every content change, which is why the height is
        refreshed here rather than in each of the callers: adding an
        organelle adds a column and a re-read replaces every row, and a
        height computed before either is the height of the old table.
        A height the user dragged is LEFT ALONE -- see
        :meth:`_apply_height`.
        """
        self._apply_height()
        questions = len(self.questions())
        objects = len(self.objects())
        self._status.setText(
            f"{questions} question(s) x {objects} object(s) = "
            f"{sum(len(row) for row in self._model.table().values())} "
            f"settings, asked once each.")

    def status_text(self) -> str:
        """What the line under the table says."""
        return self._status.text()

    def set_value(self, question: str, obj: str, text: str) -> bool:
        """Type into one cell the way the editor would.

        The screen's own edit path, exposed so a test drives the same code an
        item delegate does rather than reaching into the model.

        :param question: settings question, i.e. the key suffix shared by every
            object (``"min_area"`` for ``cell_min_area``).
        :param obj: object name, the key prefix (``"cell"``, ``"nucleus"``, ...).
        :param text: what the user would type; converted to the type of the
            cell's current value (or a sibling's). Returns ``False`` for an
            unknown row or column, a cell the object does not ask, or no
            change.
        """
        objects = self.objects()
        rows = self.questions()
        if question not in rows or obj not in objects:
            return False
        index = self._model.index(rows.index(question), objects.index(obj))
        return self._model.setData(index, text, Qt.EditRole)
