"""Process-wide linked selection and filter, so the views stop being islands.

:mod:`spacr.selection` holds the logic and knows nothing about Qt. This module
is the thin part that makes it *shared*: one object per process that every open
view subscribes to, so lassoing a cluster in the UMAP highlights the same cells
on the plate heatmap, in the measurement table and in the crop grid.

Why a singleton rather than passing a model around
--------------------------------------------------

The views are constructed independently by ``AppScreen`` as the user opens
tabs; none of them owns another, and there is no common parent below the main
window to hang a shared model off. A process-wide accessor is the same shape
:func:`spacr.qt.bridge.registry` already uses for run state, for the same
reason, and it means a view added later joins the conversation by importing one
function.

The subscription rule
---------------------

**Views must disconnect in ``closeEvent``.** This object outlives every screen,
and holds plain references to whatever connected to it. A lambda would keep a
destroyed page alive as a receiver — the exact leak
:class:`spacr.qt.widgets.home.HomePage` documents for the run registry — so
connect bound methods and drop them on close.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtCore import QObject, Signal

from ..selection import DataFilter, Selection

__all__ = ["LinkedSelection", "linked_selection"]


class LinkedSelection(QObject):
    """The shared filter and selection every linked view reads.

    Signals:
        filter_changed()     — the population narrowed or widened
        selection_changed()  — the highlighted subset moved

    The two are separate signals because they cost different amounts to
    honour. A filter change means a view has to re-query and re-lay-out; a
    selection change usually means it only has to repaint. Collapsing them into
    one ``changed`` would make every lasso trigger a full reload of a
    million-row table.
    """

    filter_changed = Signal()
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filter = DataFilter()
        self._selection = Selection.none()

    # -- filter --------------------------------------------------------
    @property
    def filter(self) -> DataFilter:
        """The active filter. Mutate through :meth:`set_filter`, not in place.

        Returned rather than copied because a copy per read would be wasteful
        on a hot path, but mutating it directly will not emit — which is the
        one way to get views showing different populations.
        """
        return self._filter

    def set_filter(self, data_filter: DataFilter) -> None:
        """Replace the filter and tell every view, even if it looks the same.

        No equality short-circuit on purpose. ``DataFilter`` holds a list of
        dataclasses, and a caller that mutated one in place then handed the
        same object back would compare equal to itself and emit nothing,
        leaving the views showing a population that no longer matches the
        controls.
        """
        self._filter = data_filter
        self.filter_changed.emit()

    def clear_filter(self) -> None:
        self.set_filter(DataFilter())

    # -- selection -----------------------------------------------------
    @property
    def selection(self) -> Selection:
        return self._selection

    def set_selection(self, selection: Selection) -> None:
        """Publish a new highlighted subset.

        ``selection.source`` names the view that made it, so a view can ignore
        the echo of its own selection rather than re-applying what it just
        drew — which otherwise costs a repaint per view per lasso, and can
        loop if a view normalises what it publishes.
        """
        self._selection = selection
        self.selection_changed.emit()

    def clear_selection(self) -> None:
        """Return to the resting state.

        Distinct from selecting nothing: :class:`spacr.selection.Selection`
        keeps "no selection" and "an empty selection" apart so views can draw
        the resting state differently from a lasso that caught nothing.
        """
        self.set_selection(Selection.none())

    def select_frame(self, frame: pd.DataFrame, source: str = "",
                     *, timelapse: bool = False) -> None:
        """Convenience: publish the rows of ``frame`` as the selection."""
        self.set_selection(
            Selection.from_frame(frame, source=source, timelapse=timelapse))

    # -- convenience for views -----------------------------------------
    def visible(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` narrowed by the active filter.

        The one call a view needs to honour the filter. Selection is deliberately
        NOT applied — a selection highlights, it does not hide, and a view that
        dropped unselected rows would make the lasso destructive.
        """
        return self._filter.apply(frame)


_LINKED: Optional[LinkedSelection] = None


def linked_selection() -> LinkedSelection:
    """The process-wide :class:`LinkedSelection` (created on first use)."""
    global _LINKED
    if _LINKED is None:
        _LINKED = LinkedSelection()
    return _LINKED
