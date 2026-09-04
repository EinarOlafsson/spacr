"""Keeping the per-object grid and the flat settings form saying the same thing.

The grid in :mod:`spacr.qt.widgets.object_settings_grid` edits VALUES; the
settings panel is a dictionary of WIDGETS, and the pipeline reads the widgets.
Mounting the grid without a binding would give a screen two answers to the
same question -- the table showing what the user typed and ``collect()``
still returning what the widget holds -- and the run would silently use the
second one. That is a settings file that means something other than what it
looks like, which is worse than the flat form the grid replaces.

So the grid never becomes the source of truth. It is a VIEW that writes
through: every edit lands in the widget behind it, and the widget stays what
the pipeline reads. Nothing downstream of ``collect()`` learns that the grid
exists, so no settings file, notebook or ``spacr-run`` invocation changes.

WHY THE PANEL IS DUCK-TYPED. This asks for two methods -- ``collect`` and
``set_value_for_key`` -- and not for a class. That keeps the binding testable
against a dictionary rather than against a built screen, and it is the whole
of what the binding needs.
"""
from __future__ import annotations

import logging

from typing import Any, Dict, FrozenSet, Mapping, Optional

from PySide6.QtCore import QObject

from spacr.object_settings_table import to_table


LOG = logging.getLogger(__name__)


class ObjectGridBinding(QObject):
    """Bind a per-object grid to the settings panel behind it.

    :param grid: the :class:`~spacr.qt.widgets.object_settings_grid.
        ObjectSettingsGrid` to drive.
    :param panel: anything offering ``collect()`` and
        ``set_value_for_key(key, value)``.
    :param parent: parent object.
    """

    def __init__(self, grid, panel, parent=None):
        super().__init__(parent)
        self._grid = grid
        self._panel = panel
        # REENTRANCY. Writing a value into a widget makes that widget emit,
        # and a screen that reseeds the grid on every widget change would
        # then rebuild the table under the cursor that is still in a cell.
        # NO VALUE depends on this: `write_through` reads the grid once,
        # before the first widget moves, so a reseed halfway cannot drop an
        # edit. What the guard buys is that the table does not visibly
        # rebuild, and the cell being typed into keeps its focus.
        self._busy = False
        self._grid.settings_changed.connect(self._write_back)

    # -- what the grid speaks for -----------------------------------------

    def owned_keys(self) -> FrozenSet[str]:
        """The settings keys the grid answers, as the panel now stands.

        Read from the panel every time rather than remembered, because a
        panel hides the settings of an object whose channel names no plane
        and the grid must not claim a key that is no longer there.
        """
        # READ FROM THE GRID, NOT FROM THE SETTINGS. The grid draws only the
        # questions every object asks, and only the organelle slots the count
        # asks for. Claiming a key it does not show would hide that setting
        # from the form as well, and it would then be reachable from nowhere
        # at all -- which is worse than either place on its own.
        owned = set()
        for question, row in self._grid.table().items():
            for obj in row:
                owned.add(f"{obj}_{question}")
        return frozenset(owned)

    # -- the two directions ------------------------------------------------

    def seed(self) -> None:
        """Show the panel's current answers in the grid.

        Idempotent, and safe to call whenever something else has written the
        panel -- a settings file being loaded, a preset applied, the Live
        Preview propagating what it tuned.
        """
        if self._busy:
            return
        self._busy = True
        try:
            self._grid.set_settings(self._panel.collect())
        finally:
            self._busy = False

    def _write_back(self) -> None:
        """Push the grid's edits into the widgets behind them."""
        if self._busy:
            return
        self._busy = True
        try:
            self.write_through()
        finally:
            self._busy = False

    def write_through(self) -> Dict[str, Any]:
        """Write every changed cell into its widget; return what changed.

        ONLY WHAT DIFFERS. Setting a widget to the value it already holds
        still makes it emit, and a panel that re-validates on every emit
        would do the whole form's work on each keystroke in the table.
        """
        # READ BOTH SIDES FIRST. Writing a widget makes it emit, which a
        # screen may answer by reseeding the grid; taking the grid's answers
        # as a snapshot here means the rest of the write proceeds from what
        # the user actually typed rather than from a table reloaded halfway.
        before = self._panel.collect()
        after = self._grid.settings()
        changed: Dict[str, Any] = {}
        for key in self.owned_keys():
            if key not in after:
                continue
            new = after[key]
            if key in before and _same(before[key], new):
                continue
            if self._panel.set_value_for_key(key, new):
                changed[key] = new
        if changed:
            self._reconsider_which_objects_the_run_has(changed)
        return changed

    def _reconsider_which_objects_the_run_has(self, changed) -> None:
        """Re-gate the form when a CHANNEL changed, and only then.

        A channel is the switch that says whether the run has an object at
        all, so typing one into the table has to turn that object's other
        categories on the way typing it into the form does. Nothing else in
        the table gates anything, and this is the whole reason for the
        filter below.

        ONLY ON A CHANNEL, and only once per write rather than per keystroke.
        The panel's own note is explicit that re-deciding this on every
        keystroke is what made the Mask module hang; `write_through` runs on
        a committed cell, so this is one call after an edit is finished
        rather than one per character.
        """
        if not any(str(key).endswith("_channel") for key in changed):
            return
        refresh = getattr(self._panel, "refresh_object_visibility", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:                                # noqa: BLE001
                LOG.debug("could not re-gate the form", exc_info=True)


def _same(a: Any, b: Any) -> bool:
    """Whether two settings values mean the same thing.

    ``1`` and ``1.0`` are the same answer asked of a spin box, and treating
    them as different would write on every pass.
    """
    if a is b:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b
