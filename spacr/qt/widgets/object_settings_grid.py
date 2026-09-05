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
it is one COLUMN, and the number of questions does not move. :meth:`
ObjectSettingsGrid.add_object` is that operation, and it starts a new
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

from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QSize, Qt, Signal
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
from ...object_settings_table import OBJECT_ORDER, column_label, from_table, to_table, widen
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
    return OFF_TEXT if question in _OFF_QUESTIONS else AUTO_TEXT


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
    return str(tooltips.get(f"{obj}_{question}", "") or "")


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
    # BOTH WORDS CLEAR THE CELL. The table draws an unset value as "auto" for
    # most questions and "off" for a channel, and whichever word the user is
    # looking at is the one they will type back.
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
        # NO TYPE TO COPY. Read a number as a number so a diameter typed
        # into an empty cell is not stored as text, and leave anything else
        # as the string it is -- a model name is a string and always was.
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

    # -- content -----------------------------------------------------------

    def set_table(self, table: Mapping[str, Mapping[str, Any]]) -> None:
        """Show ``table``, as :func:`spacr.object_settings_table.to_table`
        returns it."""
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
        """The settings question one row asks, or ``''``."""
        return self._questions[row] if 0 <= row < len(self._questions) else ""

    def value_at(self, question: str, obj: str) -> Any:
        """One cell's stored value. ``KeyError``-free: absent is ``None``."""
        return self._table.get(question, {}).get(obj)

    def asks(self, question: str, obj: str) -> bool:
        """Whether ``obj`` asks ``question`` at all.

        Absence is a fact about the object, not a value it has yet to be
        given: cytoplasm is derived and has no channel to be found in.
        """
        return obj in self._table.get(question, {})

    # -- QAbstractTableModel ----------------------------------------------

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
            # NOT EDITABLE AND NOT ENABLED: a cell that can be typed into
            # invents a settings key nothing reads.
            return Qt.ItemIsSelectable
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
        if role == Qt.DisplayRole:
            return unset if value is None else str(value)
        if role == Qt.EditRole:
            return "" if value is None else str(value)
        if role == Qt.ToolTipRole:
            head = f"{obj}_{question}  =  {unset if value is None else value!r}"
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
        if not index.isValid() or role != Qt.EditRole:
            return False
        question = self.question_at(index.row())
        obj = self._objects[index.column()]
        if not self.asks(question, obj):
            return False
        row = self._table[question]
        current = row.get(obj)
        like = current
        if like is None:
            # THE SAME QUESTION ABOUT ANOTHER OBJECT is the best evidence
            # available about what this one is: a diameter is a diameter
            # whether it is a cell's or a nucleus's.
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
            return setting_label(self._questions[section])
        if role == Qt.ToolTipRole:
            if orientation == Qt.Vertical:
                # NO TOOLTIP ON THE ROW HEADER. Every cell in the row now
                # carries the full help -- the typed body, the API link and
                # the setting's animation -- and a second, plainer tooltip on
                # the name beside them is the same explanation twice, in the
                # place the pointer crosses on its way to the cell.
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
        # The console's handle is styled by this name; the two are the same
        # affordance and should not look like two.
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

        self._table = QTableView(self)
        self._table.setModel(self._model)
        # AFTER setModel, as the contract requires: a QTableView is wrapped in
        # a proxy, so the selection model has to be taken afterwards. The
        # stored answers are unaffected -- `table()` reads the model, not the
        # view, so sorting the questions on screen reorders nothing on disk.
        install_sorting(self._table)
        self._table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        # THE TABLE OWNS ITS HEIGHT, and the grip below changes it. Inside a
        # settings panel the table is one row of a scrolling form, so it gets
        # whatever height the form hands it -- which was a QTableView's
        # default and put twenty-odd questions behind an inner scrollbar
        # inside an outer one. It now opens tall enough to show its rows and
        # can be dragged taller or shorter from the grip.
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
        row.addWidget(self._status, 1)
        row.addWidget(self._add)
        outer.addLayout(row)

    # -- the rich tooltips -------------------------------------------------

    def set_app_key(self, app_key: str) -> None:
        """Say which module's API documentation the tooltips should link to."""
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
        return f"{obj}_{question}" if self._model.asks(question, obj) else ""

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
        """
        try:
            kind = event.type()
            if kind == QEvent.Type.ToolTip:
                return True
            if kind == QEvent.Type.MouseMove:
                self._offer_tooltip(event.position().toPoint())
            elif kind == QEvent.Type.Leave:
                self._hovered_key = ""
                from .hover_tooltip import HoverTooltip
                HoverTooltip.instance().start_hide()
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
        from .hover_tooltip import HoverTooltip
        if not key:
            HoverTooltip.instance().start_hide()
            return
        try:
            from ..screens.settings_model import format_tooltip, get_tooltips
            body = str(get_tooltips().get(key) or "")
            html = format_tooltip(body, self._app_key, key)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not build the tooltip for %s", key, exc_info=True)
            return
        # The ANCHOR carries the setting, because that is what the popup
        # looks up its animation by -- see `_anchor_setting_key`.
        self._table.setProperty("settingKey", key)
        self._table.setProperty("settingsAppKey", self._app_key)
        HoverTooltip.instance().show_for(self._table, html)

    # -- the model-zoo buttons ---------------------------------------------

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

    def choose_model_for(self, obj: str) -> bool:
        """Open the model zoo for one object and store what it returns.

        :returns: True when a model was chosen. Cancelling leaves the cell
            alone rather than clearing it -- a cancelled dialog is not an
            instruction to forget the model already set.
        """
        from .model_zoo_picker import choose_model

        path = choose_model(self, kinds=("cellpose",))
        if not path:
            return False
        return self.set_value(MODEL_QUESTION, obj, path)

    # -- height ------------------------------------------------------------

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
        """Fix the table at ``height`` px, clamped to at least MIN_TABLE_H."""
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

    # -- content -----------------------------------------------------------

    def set_settings(self, settings: Mapping[str, Any]) -> None:
        """Show the per-object half of a flat settings dict.

        The rest is KEPT, not dropped: :meth:`settings` returns it unchanged
        beside the table's own keys, so this widget can edit a corner of a
        settings file without holding the whole of it hostage.
        """
        self._base = dict(settings or {})
        self._model.set_table(self._visible_table())
        self._announce()

    def _visible_table(self) -> Dict[str, Dict[str, Any]]:
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
        # AND THE COUNT CAN ASK FOR MORE THAN THE FILE HOLDS. A settings dict
        # carries keys for the slots it has been given, which is usually one;
        # a count of three is then three columns, and two of them have to be
        # made. Seeded from the slot before, so a second organelle starts
        # where the first one is.
        for index, role in enumerate(live):
            if any(role in row for row in table.values()):
                continue
            table = widen(table, role,
                          like=live[index - 1] if index else None)
        return self._only_the_shared_questions(table)

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
        }

    def settings(self) -> Dict[str, Any]:
        """The whole settings dict, with the table's answers written back."""
        return from_table(self._model.table(), self._base)

    def table(self) -> Dict[str, Dict[str, Any]]:
        """The table itself, for a caller that wants the shape."""
        return self._model.table()

    def objects(self) -> Tuple[str, ...]:
        """Which object columns are on screen."""
        return self._model.objects()

    def questions(self) -> Tuple[str, ...]:
        """Which questions are on screen, in order."""
        return tuple(self._model.table())

    # -- widening ----------------------------------------------------------

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
        """
        used = {obj for obj in self.objects() if obj.startswith("organelle")}
        for number in range(len(used) + 1, MAX_ORGANELLES + 1):
            role = organelle_role(number)
            if role not in used:
                return role
        return ""

    def add_organelle(self) -> bool:
        """Add the next organelle column, seeded from the first one.

        :returns: False when there is no slot left, with the reason on screen
            rather than as an exception into a GUI slot.
        """
        from ...organelle_types import NUMBER_OF_ORGANELLES, organelle_count

        role = self.next_organelle()
        if not role:
            self._status.setText(
                f"{MAX_ORGANELLES} organelles is the ceiling: the slots are "
                f"lettered and carry past 'z', so that is where two "
                f"letters run out.")
            return False
        # THE COUNT IS RAISED, NOT JUST THE TABLE. `number_of_organelles` is
        # what every other reader of these settings goes by -- the flat form,
        # the pipeline, a saved settings file -- so a column added here
        # without it would be a column the rest of the application does not
        # believe in, and would vanish the next time the table was rebuilt
        # from the count.
        # THE EDITS FIRST. What is on screen may differ from `_base` -- every
        # cell the user has typed into lives in the model until something
        # folds it back -- and the rebuild below reads `_base`. Without this
        # line, adding an organelle silently reverts every unsaved edit and
        # seeds the new column from the values on disk.
        self._base = from_table(self._model.table(), self._base)
        self._base[NUMBER_OF_ORGANELLES] = organelle_count(self._base) + 1
        # REBUILT FROM THE COUNT, not widened from what is on screen. The
        # settings dict already carries this slot's keys -- that is why
        # lowering the count is reversible -- so raising it brings back the
        # answers the slot had rather than a copy of its neighbour's.
        table = self._visible_table()
        if not any(role in row for row in table.values()):
            # A settings dict that never held this slot at all. Seed it from
            # the organelle before it, so a second mitochondrion starts where
            # the first one is rather than at a default nobody chose.
            previous = [o for o in self._model.objects()
                        if o.startswith("organelle")]
            table = widen(table, role, like=previous[-1] if previous else None)
            self._base = from_table(table, self._base)
        self._model.set_table(table)
        self._announce()
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
        """
        objects = self.objects()
        rows = self.questions()
        if question not in rows or obj not in objects:
            return False
        index = self._model.index(rows.index(question), objects.index(obj))
        return self._model.setData(index, text, Qt.EditRole)
