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

#: The change signals a settings widget might carry, most specific first.
#:
#: A `Toggle` is a QCheckBox and has NONE of the combo or text signals, so a
#: list that stopped at `textChanged` connected nothing at all to the two
#: per-object switches and they never reached the table.
_CHANGE_SIGNALS = ("currentTextChanged", "currentIndexChanged", "toggled",
                   "stateChanged", "textChanged", "value_changed")


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
        """Bind a per-object grid to the settings panel beside it.

        The re-entrancy guard is what stops the table visibly rebuilding under a
        cursor that is still in a cell: writing a value into a widget makes that
        widget emit, and a screen that reseeds the grid on every widget change
        would rebuild it mid-edit. No value depends on the guard -- the write
        reads the grid once, before the first widget moves -- so a reseed
        halfway cannot drop an edit either way.

        :param grid: the object settings grid.
        :param panel: the settings panel it mirrors.
        :param parent: parent object, or ``None``.
        """
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
        #: Widgets already connected, so `follow_the_form` can be called
        #: again after the table widens without doubling every connection.
        self._followed: set = set()
        self._grid.settings_changed.connect(self._write_back)
        self._grid.settings_changed.connect(self.follow_the_form)

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
        # AFTER THE TABLE EXISTS, not before. `follow_the_form` connects the
        # widgets behind the cells the grid is SHOWING, and before the first
        # seed it is showing none.
        self.follow_the_form()

    def follow_the_form(self) -> int:
        """Make the form's own fields write INTO the table.

        THE OTHER HALF OF A TWO-WAY BINDING, and the half that was missing:
        the grid wrote through to the widgets, but nothing told the grid when
        a widget moved, so a value changed anywhere else -- the flat row for
        an object the table does not claim, a preset, the Live Preview, a
        settings file poured in -- left the table showing the old answer.

        WHY NOT JUST RESEED. `seed` rebuilds the whole table, which is 30 ms
        on Mask and 100 ms once there are ten organelles. On a `textChanged`
        that is per KEYSTROKE, and it would also take the cursor out of
        whatever cell was being typed into. One cell is written instead.

        :returns: how many widgets were newly connected.
        """
        widgets = getattr(self._panel, "_widgets", None) or {}
        connected = 0
        for key in self.owned_keys():
            widget = widgets.get(key)
            if widget is None or id(widget) in self._followed:
                continue
            for name in _CHANGE_SIGNALS:
                signal = getattr(widget, name, None)
                if signal is None:
                    continue
                try:
                    signal.connect(self._form_moved)
                except Exception:                            # noqa: BLE001
                    LOG.debug("could not follow %s", key, exc_info=True)
                    break
                self._followed.add(id(widget))
                connected += 1
                break
        return connected

    def _form_moved(self, *_args) -> None:
        """Copy what the form now holds into the cells that show it.

        Reads the panel once and compares, so a widget that emitted without
        changing -- which most of them do on focus -- costs a dictionary
        lookup rather than a table write.
        """
        if self._busy:
            return
        self._busy = True
        try:
            self._show_what_the_panel_holds()
        finally:
            self._busy = False

    def _show_what_the_panel_holds(self) -> int:
        """Write the panel's answers into the cells, one cell at a time.

        :returns: how many cells changed.
        """
        try:
            current = self._panel.collect()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not read the panel", exc_info=True)
            return 0
        shown = self._grid.settings()
        moved = 0
        for key in self.owned_keys():
            if key not in current:
                continue
            if key in shown and _same(shown[key], current[key]):
                continue
            # SPLIT BY THE TABLE'S OWN RULE. `cell_mask_dim` divides into
            # `cell` and `mask_dim`, but `organelleb_min_area` has to divide
            # at the longest object prefix rather than the first underscore.
            # Asking `to_table` for a one-key dict gets exactly the split the
            # grid itself was built with, instead of a second rule beside it
            # that could disagree.
            value = current[key]
            for question, row in to_table({key: value}).items():
                for obj in row:
                    if self._grid.set_value(
                            question, obj,
                            "" if value is None else str(value)):
                        moved += 1
        return moved

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
