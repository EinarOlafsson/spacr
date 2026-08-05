"""``B16`` — N panels of the same field, panned and zoomed together.

Four channels of one field. The same well at four timepoints. The same field
under four conditions. The comparison is only worth anything if the panels are
looking at the same place at the same magnification, and doing that by hand —
zoom each one, pan each one, hope — is how two panels end up half a cell out
and a difference in framing is read as a difference in biology.

:class:`spacr.layers.CanvasLink` is the model: N canvases sharing one world
window, each keeping its own pixel size because they are different widgets.
This module is the grid of widgets over it, plus the two things that only
exist once there is more than one panel:

* **Selection reaches across.** Clicking an object in one panel publishes its
  key through :mod:`spacr.qt.linked_selection`, so the same cell lights up in
  the other panels — and in the UMAP, the plate view and the annotation grid,
  which were already listening.
* **One panel can be let go.** "Look closely at this one without losing the
  others' place" is the ordinary next request, and it is a checkbox rather
  than a mode.
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel,
                               QVBoxLayout, QWidget)

from ..layers import (Canvas, CanvasLink, LabelsLayer, LayerError, LayerStack)
from .layer_viewer import LayerCanvas
from .linked_selection import LinkedView
from .theme import font_px, register_widget_qss
from .widgets.preview_controls import FlatButton
from .widgets.toggle import Toggle

LOG = logging.getLogger(__name__)

__all__ = [
    "ComparisonPanel",
    "ComparisonGrid",
    "GRID_LINK_SOURCE",
]

#: What this view calls itself on the shared selection, so it can ignore the
#: echo of its own clicks.
GRID_LINK_SOURCE = "comparison_grid"


def _grid_qss(palette: Dict[str, Any], opacity) -> str:
    """This view's QSS block, appended to every generated stylesheet."""
    return f"""
QWidget#ComparisonGrid {{
    background: transparent;
}}
QLabel#ComparisonPanelName {{
    color: {palette["fg_dim"]};
    font-size: {font_px(10)}px;
    letter-spacing: 1px;
}}
QLabel#ComparisonStatus {{
    color: {palette["fg_muted"]};
}}
"""


register_widget_qss("ComparisonGrid", _grid_qss, replace=True)


# ---------------------------------------------------------------------------
# One cell of the grid
# ---------------------------------------------------------------------------

class _LinkedCanvas(LayerCanvas):
    """A canvas whose widget resize keeps the MAGNIFICATION, not the view.

    :meth:`spacr.qt.layer_viewer.LayerCanvas._ensure_canvas` holds the world
    span when the widget changes size, which is right for a lone viewer: the
    user still sees the same field of view. In a grid it is exactly wrong — the
    cells are not all the same size, so holding the span puts two panels at two
    magnifications and a cell that is 4 px narrower shows the same sample 2%
    bigger. Nothing about the picture says so.

    Here the step is held and the shape follows the widget, which is
    :class:`spacr.layers.CanvasLink`'s own contract.
    """

    def _ensure_canvas(self) -> Optional[Canvas]:
        height = max(1, self.height() - 2)
        width = max(1, self.width() - 2)
        if self._canvas is not None and self._canvas.shape != (height, width):
            self._canvas = replace(self._canvas, shape=(height, width))
            return self._canvas
        return super()._ensure_canvas()


class ComparisonPanel(QWidget):
    """One cell: a caption, a canvas and the checkbox that frees it.

    :param key: the panel's name in the :class:`~spacr.layers.CanvasLink`.
    :param stack: what this panel shows. Each panel has its OWN stack — that
        is the whole point, since they hold different channels, timepoints or
        conditions.
    """

    #: This panel's view moved. Carries ``(key, canvas)``.
    view_changed = Signal(str, object)
    #: Something was picked here. Carries ``(key, layer, world, value)``.
    picked = Signal(str, object, object, object)
    #: The lock checkbox was toggled. Carries ``(key, locked)``.
    lock_changed = Signal(str, bool)

    def __init__(self, key: str, stack: LayerStack, parent=None, *,
                 title: str = ""):
        super().__init__(parent)
        self._key = str(key)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setSpacing(4)
        self.caption = QLabel(title or self._key, self)
        self.caption.setObjectName("ComparisonPanelName")
        header.addWidget(self.caption, 1)
        self.lock_box = Toggle("linked", self)
        self.lock_box.setChecked(True)
        self.lock_box.setToolTip(
            "Uncheck to pan and zoom this panel on its own without losing "
            "where the others are looking.")
        self.lock_box.toggled.connect(self._on_lock_toggled)
        header.addWidget(self.lock_box)
        layout.addLayout(header)

        self.canvas = _LinkedCanvas(stack, self)
        self.canvas.view_changed.connect(self._on_view_changed)
        self.canvas.picked.connect(self._on_picked)
        layout.addWidget(self.canvas, 1)

    @property
    def key(self) -> str:
        """This panel's name in the link."""
        return self._key

    @property
    def stack(self) -> LayerStack:
        """What this panel is showing."""
        return self.canvas.stack

    def _on_view_changed(self) -> None:
        self.view_changed.emit(self._key, self.canvas.canvas)

    def _on_picked(self, layer, world, value) -> None:
        self.picked.emit(self._key, layer, world, value)

    def _on_lock_toggled(self, checked: bool) -> None:
        self.lock_changed.emit(self._key, bool(checked))

    def detach(self) -> None:
        """Let go of the model. Call from the grid's ``closeEvent``."""
        self.canvas.detach()


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

class ComparisonGrid(LinkedView, QWidget):
    """N panels of the same field, locked together.

    :param panels: ``[(key, stack), …]`` or ``{key: stack}`` — what each panel
        shows. Order is the order they are laid out in.
    :param columns: how many panels per row; the default is the squarest grid
        that fits them.
    :param titles: per-panel captions, defaulting to the keys.
    """

    #: The shared world window moved. Carries the driving panel's key.
    view_changed = Signal(str)
    #: An object was picked in one of the panels. Carries its key.
    object_picked = Signal(str)

    def __init__(self, panels: Any = None, parent=None, *,
                 columns: Optional[int] = None,
                 titles: Optional[Dict[str, str]] = None):
        super().__init__(parent)
        self.setObjectName("ComparisonGrid")
        # NOT `_link`: that name belongs to LinkedView, which keeps the
        # process-wide selection bus in it. Shadowing it here would leave this
        # grid publishing selections into its own canvas link and hearing
        # nothing from the rest of the app.
        self._canvas_link = CanvasLink()
        self._panels: Dict[str, ComparisonPanel] = {}
        self._titles = dict(titles or {})
        self._columns = columns
        self._syncing = False
        self._build()
        for key, stack in self._as_pairs(panels):
            self.add_panel(key, stack)
        self._refresh_status()
        self.link_selection(GRID_LINK_SOURCE)

    @staticmethod
    def _as_pairs(panels: Any) -> List[Tuple[str, LayerStack]]:
        if panels is None:
            return []
        items = (list(panels.items()) if isinstance(panels, dict)
                 else [tuple(entry) for entry in panels])
        return [(str(key), stack) for key, stack in items]

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)
        self.grid = QGridLayout()
        self.grid.setSpacing(8)
        outer.addLayout(self.grid, 1)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.fit_button = FlatButton("Fit", self,
                                     tooltip="Fit every linked panel again")
        self.fit_button.clicked.connect(self.reset_view)
        row.addWidget(self.fit_button)
        self.lock_all_button = FlatButton(
            "Link all", self, tooltip="Bring every panel back to this view")
        self.lock_all_button.clicked.connect(self.lock_all)
        row.addWidget(self.lock_all_button)
        row.addStretch(1)
        outer.addLayout(row)

        self.status = QLabel("", self)
        self.status.setObjectName("ComparisonStatus")
        outer.addWidget(self.status)

    # -- panels ------------------------------------------------------------
    @property
    def canvas_link(self) -> CanvasLink:
        """The shared world window every locked panel is on.

        Not called ``link``: that is
        :attr:`spacr.qt.linked_selection.LinkedView.link`, the process-wide
        selection bus this grid also joins. Two different links, and confusing
        them is how a view ends up publishing selections to itself.
        """
        return self._canvas_link

    @property
    def panels(self) -> Dict[str, ComparisonPanel]:
        """``{key: panel}``, in layout order."""
        return dict(self._panels)

    def add_panel(self, key: str, stack: LayerStack, *,
                  title: str = "") -> ComparisonPanel:
        """Add one panel; returns it.

        A panel added to a grid the user has already zoomed into starts where
        the others are, not fitted to its own extent — otherwise adding a
        fifth channel throws away the view.
        """
        key = str(key)
        if key in self._panels:
            raise LayerError(f"panel {key!r} is already in this grid")
        panel = ComparisonPanel(key, stack, self,
                                title=title or self._titles.get(key, key))
        panel.view_changed.connect(self._on_panel_view_changed)
        panel.picked.connect(self._on_panel_picked)
        panel.lock_changed.connect(self._on_lock_changed)
        self._panels[key] = panel
        self._relayout()
        canvas = panel.canvas._ensure_canvas()
        if canvas is not None:
            self._canvas_link.add(key, canvas)
            panel.canvas._canvas = self._canvas_link[key]
            panel.canvas.update()
        self._refresh_status()
        return panel

    def remove_panel(self, key: str) -> ComparisonPanel:
        """Take a panel out of the grid and return it."""
        key = str(key)
        if key not in self._panels:
            raise LayerError(
                f"no panel {key!r}; the panels are {list(self._panels)}")
        panel = self._panels.pop(key)
        if key in self._canvas_link:
            self._canvas_link.remove(key)
        panel.detach()
        panel.setParent(None)
        self._relayout()
        self._refresh_status()
        return panel

    def _relayout(self) -> None:
        while self.grid.count():
            self.grid.takeAt(0)
        n = len(self._panels)
        columns = self._columns or max(1, int(math.ceil(math.sqrt(n))))
        for index, panel in enumerate(self._panels.values()):
            self.grid.addWidget(panel, index // columns, index % columns)
            panel.show()

    # -- keeping the panels together ---------------------------------------
    def _on_panel_view_changed(self, key: str, canvas: Optional[Canvas]
                               ) -> None:
        """One panel moved: put every locked panel on the same window."""
        if self._syncing or canvas is None or key not in self._canvas_link:
            return
        self._syncing = True
        try:
            self._canvas_link.set(key, canvas)
            self._push_to_panels(skip=key)
        finally:
            self._syncing = False
        self.view_changed.emit(key)
        self._refresh_status()

    def _push_to_panels(self, *, skip: str = "") -> None:
        """Put every following panel's widget on its canvas from the link.

        Each panel's SHAPE comes from its widget and its WINDOW comes from the
        link, which is the split that lets a grid of unequal cells share one
        magnification. ``LayerCanvas`` would otherwise refit on its next paint
        and hold the world span instead, quietly leaving two cells at two
        magnifications.
        """
        for other, panel in self._panels.items():
            if other == skip or other not in self._canvas_link:
                continue
            widget = panel.canvas
            height = max(1, widget.height() - 2)
            width = max(1, widget.width() - 2)
            if self._canvas_link[other].shape != (height, width):
                self._canvas_link.resize(other, height, width)
            widget._canvas = self._canvas_link[other]
            widget.update()

    def resizeEvent(self, event) -> None:
        """A layout change resizes the cells; re-share the window afterwards."""
        super().resizeEvent(event)
        self._push_to_panels()

    def _on_lock_changed(self, key: str, locked: bool) -> None:
        if key not in self._canvas_link:
            return
        if locked:
            self._canvas_link.lock(key)
            self._push_to_panels()
        else:
            self._canvas_link.unlock(key)
        self._refresh_status()

    def lock_all(self) -> None:
        """Bring every panel back onto the shared window."""
        for key, panel in self._panels.items():
            panel.lock_box.setChecked(True)
            if key in self._canvas_link:
                self._canvas_link.lock(key)
        self._push_to_panels()
        self._refresh_status()

    def reset_view(self) -> None:
        """Fit every linked panel to its own stack again."""
        for key, panel in self._panels.items():
            if key not in self._canvas_link or not self._canvas_link.is_locked(key):
                continue
            panel.canvas.reset_view()
            fitted = panel.canvas._ensure_canvas()
            if fitted is not None:
                self._syncing = True
                try:
                    self._canvas_link.set(key, fitted)
                    self._push_to_panels(skip=key)
                finally:
                    self._syncing = False
                break
        self._refresh_status()

    # -- selection ---------------------------------------------------------
    def _on_panel_picked(self, key: str, layer, world, value) -> None:
        """A click in one panel reaches every other view in the app."""
        if not isinstance(layer, LabelsLayer) or not value:
            self._refresh_status()
            return
        object_key = layer.object_key_at_world(world)
        if object_key is None:
            return
        layer.selected_label = int(value)
        self.highlight(object_key)
        self.object_picked.emit(object_key)
        self.publish_selection([object_key])

    def highlight(self, object_key: str) -> List[str]:
        """Select ``object_key`` in every panel that holds it; returns which.

        The other half of the comparison: a cell picked in the DAPI panel is
        the same cell in the phalloidin panel, and saying so is what makes the
        four pictures one observation rather than four.
        """
        found: List[str] = []
        for key, panel in self._panels.items():
            for layer in panel.stack:
                if not isinstance(layer, LabelsLayer) or layer.field is None:
                    continue
                for label in layer.labels():
                    if layer.field.object_key(int(label)) != object_key:
                        continue
                    layer.selected_label = int(label)
                    found.append(key)
                    break
                else:
                    continue
                break
        self.status.setText(
            f"{object_key} · in {len(found)} of {len(self._panels)} panel(s)")
        return found

    def on_linked_selection_changed(self, selection) -> None:
        """Another view selected something: show it in every panel."""
        if selection.keys is None or len(selection.keys) != 1:
            return
        self.highlight(str(selection.keys[0]))

    # -- status ------------------------------------------------------------
    def _refresh_status(self) -> None:
        if not self._panels:
            self.status.setText("No panels")
            return
        free = [key for key in self._panels
                if key in self._canvas_link
                and not self._canvas_link.is_locked(key)]
        note = (f" · {len(free)} free ({', '.join(free)})" if free
                else " · all linked")
        self.status.setText(f"{len(self._panels)} panel(s){note}")

    def closeEvent(self, event) -> None:
        """Leave the shared selection and let go of every panel's model."""
        self.unlink_selection()
        for panel in self._panels.values():
            panel.detach()
        super().closeEvent(event)
