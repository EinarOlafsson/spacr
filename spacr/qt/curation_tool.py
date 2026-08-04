"""``B12`` ``C7`` — the brush and the track surgery, as widgets.

:mod:`spacr.curation` is the session: the undo history, the ledger, and the
rules a join or a split has to obey, all in plain numpy and pandas so they can
be tested without a display. This is the mouse and the two panels.

The brush is a :class:`~spacr.qt.layer_viewer.CanvasTool`, like the ROI pen
and the counter, so it borrows the canvas's mouse without the canvas knowing
what it is for — and the world coordinates it is handed are what make a stroke
painted at 8× zoom land where the same stroke at 1× does.

One drag is one undo
--------------------

A stroke is dozens of ``move`` events and exactly one thing the user did.
:meth:`BrushTool.press` opens a stroke, :meth:`BrushTool.release` closes it,
and undo takes back the stroke — not the last few pixels of it. That is the
whole reason ``CanvasTool`` grew a ``release`` hook.

Nothing is edited off the record
--------------------------------

Both panels write their ledger after every action rather than on a Save
button. A correction that is only in memory when the application is killed is
a correction that happened to the data and not to the record of it, and the
two disagreeing is worse than neither existing.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox,
                               QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QSpinBox, QVBoxLayout, QWidget)

from ..curation import CurationError, CurationLog, MaskCuration, TrackCuration
from ..layers import LabelsLayer, LayerError, LayerEvent
from .layer_viewer import CanvasTool, LayerCanvas
from .theme import register_widget_qss
from .widgets.preview_controls import FlatButton, FlatComboBox

LOG = logging.getLogger(__name__)

__all__ = [
    "BrushTool",
    "BrushPanel",
    "TrackCurationPanel",
]


def _curation_qss(palette: Dict[str, Any], opacity) -> str:
    """These panels' QSS block, appended to every generated stylesheet."""
    return f"""
QWidget#BrushPanel, QWidget#TrackCurationPanel {{
    background: transparent;
}}
QListWidget#TrackList, QListWidget#CurationLedger {{
    background: {palette["surface_alt"]};
    border: 1px solid {palette["border_soft"]};
    border-radius: 10px;
    padding: 4px;
    color: {palette["fg"]};
}}
QListWidget#TrackList::item, QListWidget#CurationLedger::item {{
    padding: 4px 6px;
    border-radius: 6px;
}}
QListWidget#TrackList::item:selected {{
    background: {palette["accent_soft"]};
    color: {palette["fg"]};
}}
QLabel#CurationStatus {{
    color: {palette["fg_muted"]};
}}
QLabel#CuratedBadge {{
    color: {palette["warning"]};
}}
"""


register_widget_qss("BrushPanel", _curation_qss, replace=True)


# ---------------------------------------------------------------------------
# The brush
# ---------------------------------------------------------------------------

class BrushTool(CanvasTool):
    """Turns a drag on a :class:`~spacr.qt.layer_viewer.LayerCanvas` into paint.

    Left drag paints the active label; right drag erases (paints 0), because
    "take that bit off the mask" is the other half of the same gesture and
    having to switch to an eraser mode for it doubles the interactions in a
    job that is already mostly correction. ``[`` and ``]`` resize the brush
    and Backspace undoes, so a whole correction pass is one hand on the mouse.

    :param session: the :class:`spacr.curation.MaskCuration` to paint into.
    """

    cursor = Qt.CrossCursor

    def __init__(self, session: MaskCuration):
        if not isinstance(session, MaskCuration):
            raise LayerError(
                f"a brush paints into a MaskCuration, got {session!r}")
        self.session = session
        self._erasing = False
        self._painting = False

    # -- the mouse ----------------------------------------------------------
    def press(self, view: LayerCanvas, world: Dict[str, float],
              event: Any) -> bool:
        """Open a stroke and lay the first dab."""
        button = event.button() if hasattr(event, "button") else Qt.LeftButton
        if button not in (Qt.LeftButton, Qt.RightButton):
            return False
        self._erasing = button == Qt.RightButton
        self._painting = True
        self.session.begin_stroke()
        self._dab(world)
        return True

    def move(self, view: LayerCanvas, world: Dict[str, float],
             event: Any) -> bool:
        """Continue the stroke, but only while a button is actually down.

        The canvas sends ``move`` for every mouse motion, drag or not. Without
        this guard the brush would paint wherever the cursor happened to
        travel after the button came up — which is the sort of bug that
        destroys a mask in the time it takes to reach for the undo button.
        """
        if not self._painting:
            return False
        buttons = event.buttons() if hasattr(event, "buttons") else Qt.NoButton
        if not (buttons & (Qt.LeftButton | Qt.RightButton)):
            return False
        self._dab(world)
        return True

    def release(self, view: LayerCanvas, world: Dict[str, float],
                event: Any) -> bool:
        """Close the stroke, so undo takes back all of it."""
        if not self._painting:
            return False
        self._painting = False
        self.session.end_stroke()
        return True

    def key(self, view: LayerCanvas, event: Any) -> bool:
        """``[`` / ``]`` resize the brush; Backspace undoes a stroke."""
        if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            self.session.undo()
            return True
        text = event.text()
        if text == "[":
            self.session.radius = max(0.5, self.session.radius / 1.5)
            return True
        if text == "]":
            self.session.radius = self.session.radius * 1.5
            return True
        return False

    def detach(self) -> None:
        """Close any stroke still open, so it is recorded rather than lost."""
        if self._painting:
            self._painting = False
            self.session.end_stroke()

    def _dab(self, world: Dict[str, float]) -> int:
        try:
            if self._erasing:
                return self.session.erase(world)
            return self.session.paint(world)
        except LayerError:
            LOG.exception("Could not paint")
            return 0


# ---------------------------------------------------------------------------
# The brush panel
# ---------------------------------------------------------------------------

class BrushPanel(QWidget):
    """The brush controls, the undo button, and the ledger beside the image.

    :param canvas: the canvas to paint on.
    :param layer: the labels layer to edit. Defaults to the first one in the
        canvas's stack, so the ordinary case needs no argument.
    :param artifact: the mask's path, so the ledger is written beside it.
    """

    #: The mask changed. Carries how many elements moved.
    painted = Signal(int)
    #: The ledger was written. Carries the path.
    logged = Signal(str)

    def __init__(self, canvas: LayerCanvas, parent=None, *,
                 layer: Optional[LabelsLayer] = None, artifact: str = "",
                 session: Optional[MaskCuration] = None):
        super().__init__(parent)
        self.setObjectName("BrushPanel")
        self._canvas = canvas
        self._artifact = str(artifact or "")
        layer = layer if layer is not None else self._first_labels()
        if session is None:
            if layer is None:
                raise LayerError(
                    "a brush needs a labels layer to paint into; this stack "
                    "has none. Add the mask before switching the brush on.")
            session = MaskCuration(layer, artifact=self._artifact or layer.name)
        self._session = session
        self._tool: Optional[BrushTool] = None
        self._build()
        self._canvas.stack.subscribe(self._on_layers_changed)
        # Two subscriptions, because they fire at different moments. The
        # stack fires per dab (mid-stroke, before anything is recorded); the
        # session fires when a ledger entry lands, which is what this panel
        # is showing. Listening only to the first left the ledger and the
        # undo button one stroke behind for ever.
        self._session.subscribe(self._on_edit_recorded)
        self.refresh()

    def _first_labels(self) -> Optional[LabelsLayer]:
        for layer in self._canvas.stack:
            if isinstance(layer, LabelsLayer):
                return layer
        return None

    # -- construction --------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.paint_button = FlatButton(
            "Brush", self,
            tooltip="Drag to paint the active label, right-drag to erase, "
                    "[ and ] resize, Backspace undoes the last stroke")
        self.paint_button.setCheckable(True)
        self.paint_button.toggled.connect(self._on_paint_toggled)
        outer.addWidget(self.paint_button)

        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label", self))
        self.label_spin = QSpinBox(self)
        self.label_spin.setRange(0, 1_000_000)
        self.label_spin.setValue(int(self._session.label))
        self.label_spin.setToolTip(
            "The label the brush paints. 0 is background — painting 0 is how "
            "you take a piece off the mask.")
        self.label_spin.valueChanged.connect(self._on_label_changed)
        label_row.addWidget(self.label_spin, 1)
        self.next_label_button = FlatButton(
            "New", self, tooltip="Paint an object that does not exist yet")
        self.next_label_button.clicked.connect(self.use_next_label)
        label_row.addWidget(self.next_label_button)
        outer.addLayout(label_row)

        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("Radius", self))
        self.radius_spin = QDoubleSpinBox(self)
        self.radius_spin.setRange(0.5, 500.0)
        self.radius_spin.setDecimals(2)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setValue(float(self._session.radius))
        self.radius_spin.valueChanged.connect(self._on_radius_changed)
        radius_row.addWidget(self.radius_spin, 1)
        self.units = QLabel("", self)
        self.units.setObjectName("CurationStatus")
        radius_row.addWidget(self.units)
        outer.addLayout(radius_row)

        actions = QHBoxLayout()
        self.undo_button = FlatButton("Undo stroke", self,
                                      tooltip="Take back the last stroke")
        self.undo_button.clicked.connect(self.undo)
        actions.addWidget(self.undo_button)
        self.save_button = FlatButton(
            "Save log", self,
            tooltip="Write the correction ledger beside the mask")
        self.save_button.clicked.connect(self.save_log)
        actions.addWidget(self.save_button)
        outer.addLayout(actions)

        self.badge = QLabel("", self)
        self.badge.setObjectName("CuratedBadge")
        self.badge.setWordWrap(True)
        outer.addWidget(self.badge)

        self.ledger = QListWidget(self)
        self.ledger.setObjectName("CurationLedger")
        self.ledger.setSelectionMode(QAbstractItemView.NoSelection)
        self.ledger.setToolTip(
            "Every correction made in this session, in order. Written beside "
            "the mask so a curated dataset can be told from a raw one.")
        outer.addWidget(self.ledger, 1)

    # -- model ---------------------------------------------------------------
    @property
    def session(self) -> MaskCuration:
        """The curation session this panel drives."""
        return self._session

    @property
    def tool(self) -> Optional[BrushTool]:
        """The brush while it is switched on, else ``None``."""
        return self._tool

    def start_painting(self) -> BrushTool:
        """Attach the brush to the canvas and return it."""
        self._tool = BrushTool(self._session)
        self._canvas.set_tool(self._tool)
        return self._tool

    def stop_painting(self) -> None:
        """Give the canvas its mouse back, closing any open stroke."""
        if self._canvas.tool is self._tool and self._tool is not None:
            self._canvas.set_tool(None)
        self._tool = None

    def _on_paint_toggled(self, checked: bool) -> None:
        if checked:
            self.start_painting()
        else:
            self.stop_painting()

    def _on_layers_changed(self, event: LayerEvent) -> None:
        # Derived, not stored: the ledger and the badge follow the model, so
        # a paint made through the tool and one made from a script look the
        # same here.
        if event.kind == "data":
            self.refresh()

    def _on_edit_recorded(self, _edit) -> None:
        """A stroke closed (or was undone): the ledger has a new line."""
        self.refresh()

    # -- actions -------------------------------------------------------------
    def _on_label_changed(self, value: int) -> None:
        self._session.label = int(value)

    def _on_radius_changed(self, value: float) -> None:
        self._session.radius = float(value)

    def use_next_label(self) -> int:
        """Point the brush at a label the mask is not using yet."""
        labels = self._session.layer.labels()
        nxt = int(labels.max()) + 1 if len(labels) else 1
        self.label_spin.setValue(nxt)
        return nxt

    def undo(self) -> bool:
        """Take back the last stroke. ``True`` if there was one."""
        edit = self._session.undo()
        self.refresh()
        return edit is not None

    def save_log(self, path: Optional[str] = None) -> Optional[str]:
        """Write the ledger beside the mask. Returns the path."""
        try:
            written = self._session.save_log(path or self._artifact or None)
        except OSError as exc:
            LOG.info("could not write the curation ledger", exc_info=True)
            self.badge.setText(f"Could not write the ledger: {exc}")
            return None
        self.logged.emit(written)
        self.refresh()
        return written

    # -- the ledger ----------------------------------------------------------
    def refresh(self) -> None:
        """Redraw the ledger and the badge from the session."""
        self.undo_button.setEnabled(self._session.can_undo)
        spacing = getattr(self._session.layer, "spacing", None)
        self.units.setText(getattr(spacing, "units", "") or "")
        self.ledger.clear()
        for edit in self._session.log.edits:
            item = QListWidgetItem(edit.describe())
            item.setToolTip(str(dict(edit.detail)))
            self.ledger.addItem(item)
        self.badge.setText(self._session.log.describe())
        self.painted.emit(sum(edit.n_changed
                              for edit in self._session.log.edits))

    def closeEvent(self, event) -> None:
        """Stop painting and let go of the model."""
        self.stop_painting()
        self._canvas.stack.unsubscribe(self._on_layers_changed)
        self._session.unsubscribe(self._on_edit_recorded)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Track curation
# ---------------------------------------------------------------------------

class TrackCurationPanel(QWidget):
    """Join, split and delete tracks, with the ledger beside them.

    :param tracks: a track table, or ``None`` to open one later.
    :param artifact: the tracks CSV the table came from.

    The three operations are three buttons and no modes. A join takes the two
    selected tracks; a split takes one track and the frame in the spinner;
    a delete takes whatever is selected. Every one of them is refused with a
    sentence rather than silently declined when it would break the table —
    a button that sometimes does nothing is indistinguishable from a bug.
    """

    #: The table changed. Carries the number of tracks now in it.
    tracks_changed = Signal(int)
    #: The curated table was written. Carries the path.
    saved = Signal(str)

    def __init__(self, parent=None, *, tracks: Optional[pd.DataFrame] = None,
                 artifact: str = "",
                 session: Optional[TrackCuration] = None):
        super().__init__(parent)
        self.setObjectName("TrackCurationPanel")
        self._artifact = str(artifact or "")
        self._session: Optional[TrackCuration] = session
        if self._session is None and tracks is not None:
            self._session = TrackCuration(tracks, artifact=self._artifact)
        self._build()
        self.refresh()

    # -- construction --------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        heading = QLabel("Tracks", self)
        outer.addWidget(heading)

        self.track_list = QListWidget(self)
        self.track_list.setObjectName("TrackList")
        self.track_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.track_list.setToolTip(
            "Select two tracks to join them, or one to split or delete it.")
        self.track_list.itemSelectionChanged.connect(self._refresh_buttons)
        outer.addWidget(self.track_list, 1)

        join_row = QHBoxLayout()
        self.join_button = FlatButton(
            "Join", self,
            tooltip="Make the second selected track a continuation of the "
                    "first. Refused when they overlap in time — two tracks in "
                    "one frame are two objects.")
        self.join_button.clicked.connect(self.join_selected)
        join_row.addWidget(self.join_button)
        self.delete_button = FlatButton(
            "Delete", self, tooltip="Remove the selected track entirely")
        self.delete_button.clicked.connect(self.delete_selected)
        join_row.addWidget(self.delete_button)
        outer.addLayout(join_row)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Split at frame", self))
        self.frame_spin = QSpinBox(self)
        self.frame_spin.setRange(0, 1_000_000)
        self.frame_spin.setToolTip(
            "The first frame of the NEW track. Everything from here on gets a "
            "new id.")
        split_row.addWidget(self.frame_spin, 1)
        self.split_button = FlatButton(
            "Split", self,
            tooltip="Break the selected track in two at that frame")
        self.split_button.clicked.connect(self.split_selected)
        split_row.addWidget(self.split_button)
        outer.addLayout(split_row)

        save_row = QHBoxLayout()
        self.save_button = FlatButton(
            "Save tracks…", self,
            tooltip="Write the curated table and its ledger together")
        self.save_button.clicked.connect(self.save)
        save_row.addWidget(self.save_button)
        outer.addLayout(save_row)

        self.status = QLabel("", self)
        self.status.setObjectName("CurationStatus")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        self.ledger = QListWidget(self)
        self.ledger.setObjectName("CurationLedger")
        self.ledger.setSelectionMode(QAbstractItemView.NoSelection)
        outer.addWidget(self.ledger, 1)

    # -- model ---------------------------------------------------------------
    @property
    def session(self) -> Optional[TrackCuration]:
        """The curation session, or ``None`` before a table is open."""
        return self._session

    def set_tracks(self, tracks: pd.DataFrame, *, artifact: str = "") -> None:
        """Open a track table. The seam a screen (or a test) goes through."""
        self._artifact = str(artifact or self._artifact)
        self._session = TrackCuration(tracks, artifact=self._artifact)
        self.refresh()

    def load(self, path: str) -> Optional[TrackCuration]:
        """Read a tracks CSV and open it.

        Any ledger already beside it is read back too, so a second curation
        session continues the first one's history rather than starting a
        fresh one that makes the earlier edits invisible.
        """
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            self.status.setText(f"Could not read {path}: {exc}")
            return None
        try:
            self._session = TrackCuration(
                frame, artifact=path, log=CurationLog.read_beside(path))
        except CurationError as exc:
            self._session = None
            self.refresh()
            # After refresh for the same reason as above: refresh() resets
            # this label to "No tracks open", which is true and useless
            # next to the sentence saying WHY nothing opened.
            self.status.setText(str(exc))
            return None
        self._session.log.artifact = path
        self._artifact = path
        self.refresh()
        return self._session

    def selected_tracks(self) -> List[Any]:
        """The track ids selected, in the list's order."""
        return [item.data(Qt.UserRole) for item in self.track_list.selectedItems()]

    # -- operations ----------------------------------------------------------
    def join_selected(self) -> bool:
        """Join the two selected tracks. ``True`` when it happened."""
        return self._do(lambda s, ids: s.join(ids[0], ids[1]), needs=2)

    def split_selected(self) -> bool:
        """Split the selected track at the frame in the spinner."""
        return self._do(
            lambda s, ids: s.split(ids[0], self.frame_spin.value()), needs=1)

    def delete_selected(self) -> bool:
        """Delete the selected track."""
        return self._do(lambda s, ids: s.delete(ids[0]), needs=1)

    def _do(self, action, *, needs: int) -> bool:
        session = self._session
        if session is None:
            self.status.setText("Open a tracks table first.")
            return False
        ids = self.selected_tracks()
        if len(ids) < needs:
            self.status.setText(
                f"Select {needs} track(s) first — {len(ids)} selected.")
            return False
        try:
            edit = action(session, ids)
        except CurationError as exc:
            # Said, not swallowed: a button that silently declines is
            # indistinguishable from a broken one.
            self.status.setText(str(exc))
            return False
        self.refresh()
        # After refresh, not before: refresh() writes the session summary
        # into this same label, and setting the line first meant every
        # successful action was immediately overwritten by the summary. Both
        # facts matter, so both are shown.
        self.status.setText(f"{edit.describe()}\n{session.describe()}")
        return True

    def save(self, path: Optional[str] = None) -> Optional[str]:
        """Write the curated table and its ledger. Returns the CSV path."""
        if self._session is None:
            return None
        target = path or self._artifact
        if not target:
            target, _ = QFileDialog.getSaveFileName(
                self, "Save the curated tracks",
                os.path.join(os.getcwd(), "tracks_curated.csv"), "CSV (*.csv)")
            if not target:
                return None
        try:
            written = self._session.save(target)
        except OSError as exc:
            self.status.setText(f"Could not write the tracks: {exc}")
            return None
        self.saved.emit(written)
        self.refresh()
        return written

    # -- rendering -----------------------------------------------------------
    def refresh(self) -> None:
        """Redraw the track list, the ledger and the consistency line."""
        self.track_list.clear()
        self.ledger.clear()
        session = self._session
        if session is None:
            self.status.setText("No tracks open.")
            self._refresh_buttons()
            self.tracks_changed.emit(0)
            return
        for track_id in session.track_ids:
            span = session.span(track_id)
            frames = session.frames_of(track_id)
            item = QListWidgetItem(
                f"track {track_id}   {len(frames)} frame(s)   "
                f"{span[0]}–{span[1]}" if span else f"track {track_id}")
            item.setData(Qt.UserRole, track_id)
            self.track_list.addItem(item)
        for edit in session.log.edits:
            entry = QListWidgetItem(edit.describe())
            entry.setToolTip(str(dict(edit.detail)))
            self.ledger.addItem(entry)
        self.status.setText(session.describe())
        self._refresh_buttons()
        self.tracks_changed.emit(len(session.track_ids))

    def _refresh_buttons(self) -> None:
        selected = len(self.selected_tracks())
        self.join_button.setEnabled(selected >= 2)
        self.split_button.setEnabled(selected == 1)
        self.delete_button.setEnabled(selected == 1)
        self.save_button.setEnabled(self._session is not None)
