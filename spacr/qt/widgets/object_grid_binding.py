"""Keep the per-object grid and the flat settings form in agreement.

The grid in :mod:`spacr.qt.widgets.object_settings_grid` edits values; the
settings panel is a dictionary of widgets, and the pipeline reads the widgets.
Mounting the grid without a binding would give a screen two answers to the
same question -- the table showing what the user typed and ``collect()``
still returning what the widget holds -- and the run would silently use the
second one, so the saved settings would differ from what the grid shows.

So the grid never becomes the source of truth. It is a view that writes
through: every edit lands in the widget behind it, and the widget stays what
the pipeline reads. Nothing downstream of ``collect()`` learns that the grid
exists, so no settings file, notebook or ``spacr-run`` invocation changes.

The panel is duck-typed: the binding asks for two methods -- ``collect`` and
``set_value_for_key`` -- and not for a class. That keeps the binding testable
against a dictionary rather than against a built screen.
"""
from __future__ import annotations

import logging

from typing import Any, Dict, FrozenSet, Mapping, Optional

from PySide6.QtCore import QObject

from spacr.object_settings_table import (_CALCULATE_DIAMETERS,
                                         _DIAMETER_OBJECTS, _FILTER_PREFIX,
                                         _blank_diameters_held_still,
                                         _calculates_diameters,
                                         _settings_key, to_table)
from spacr.qt.widgets.object_settings_grid import _SWITCH_QUESTIONS

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
        self._busy = False
        #: Widgets already connected, so `follow_the_form` can be called
        #: again after the table widens without doubling every connection.
        self._followed: set = set()
        self._grid.settings_changed.connect(self._write_back)
        self._grid.settings_changed.connect(self.follow_the_form)


    def owned_keys(self) -> FrozenSet[str]:
        """The settings keys the grid answers, as the panel now stands.

        Read from the panel every time rather than remembered, because a
        panel hides the settings of an object whose channel names no plane
        and the grid must not claim a key that is no longer there.

        Columns hidden because their channel is unset are still claimed, so
        their rows stay off the flat form; their channels are not claimed,
        because a hidden object's channel on the form is how it is brought
        back.
        """
        owned = set()
        claimed = getattr(self._grid, "_claimed_table", None)
        table = claimed() if callable(claimed) else self._grid.table()
        for question, row in table.items():
            if (question.startswith(_FILTER_PREFIX)
                    or question in _SWITCH_QUESTIONS):
                continue
            for obj in row:
                owned.add(_settings_key(obj, question))
        if "object_filters" in getattr(self._grid, "_base", {}):
            owned.add("object_filters")
        return frozenset(owned)


    def _switch_keys(self) -> FrozenSet[str]:
        """The channel keys that decide which columns the grid draws.

        Followed like the claimed keys, so a channel set on the form shows or
        hides its object's column (2026-09-29, item 592).
        """
        claimed = getattr(self._grid, "_claimed_table", None)
        if not callable(claimed):
            return frozenset()
        return frozenset(
            _settings_key(obj, question)
            for question, row in claimed().items()
            if question in _SWITCH_QUESTIONS for obj in row)

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
        self.follow_the_form()

    def follow_the_form(self) -> int:
        """Make the form's own fields write into the table.

        This is the form-to-grid half of the two-way binding. Without it, a
        value changed outside the grid -- the flat row for an object the
        table does not claim, a preset, the Live Preview, a loaded settings
        file -- would leave the table showing the old value.

        The table is not reseeded: `seed` rebuilds the whole table, which
        takes 30 ms on Mask and 100 ms with ten organelles. On a
        `textChanged` that would run per keystroke and move the cursor out
        of the cell being edited. One cell is written instead.

        The magnification field is followed as well: a blank object
        diameter shows the magnification default, which moves with it
        (item 696).

        :returns: how many widgets were newly connected.
        """
        widgets = getattr(self._panel, "_widgets", None) or {}
        connected = 0
        switch = widgets.get(_CALCULATE_DIAMETERS)
        clicked = getattr(switch, "clicked", None)
        if clicked is not None and id(switch) not in self._followed:
            try:
                clicked.connect(self._calculate_clicked)
            except Exception:                                # noqa: BLE001
                LOG.debug("could not follow %s", _CALCULATE_DIAMETERS,
                          exc_info=True)
        for key in (self.owned_keys() | self._switch_keys()
                    | {"magnification", _CALCULATE_DIAMETERS}):
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

        When a channel on the form switches an object on or off, the columns
        are redrawn from the panel, which also shows every value.
        """
        try:
            current = self._panel.collect()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not read the panel", exc_info=True)
            return 0
        same = getattr(self._grid, "_shows_the_same_objects_as", None)
        if callable(same) and not same(current):
            self._grid.set_settings(current)
            return 1
        keep = getattr(self._grid, "_keep_the_switches", None)
        if callable(keep):
            keep(current)
        calculation = getattr(self._grid, "_follow_calculation", None)
        if callable(calculation) and _CALCULATE_DIAMETERS in current:
            calculation(current[_CALCULATE_DIAMETERS])
        shown_magnification = getattr(getattr(self._grid, "_model", None),
                                      "_magnification", None)
        if self._hold_the_diameters_still(current, shown_magnification):
            current = self._panel.collect()
        follow = getattr(self._grid, "_follow_magnification", None)
        if callable(follow) and "magnification" in current:
            follow(current["magnification"])
        shown = self._grid.settings()
        moved = 0
        for key in self.owned_keys():
            if key not in current:
                continue
            if key in shown and _same(shown[key], current[key]):
                continue
            if key == "object_filters":
                self._grid.set_filters(current[key])
                moved += 1
                continue
            value = current[key]
            for question, row in to_table({key: value}).items():
                for obj in row:
                    if self._grid.set_value(
                            question, obj,
                            "" if value is None else str(value)):
                        moved += 1
        return moved

    def _write_back(self) -> None:
        """Push the grid's edits into the widgets behind them.

        With "Calculate diameters" off, a diameter cleared in the table is
        refilled with its value at the current magnification rather than
        left blank to follow the next one.
        """
        if self._busy:
            return
        self._busy = True
        try:
            self.write_through()
            if self._hold_the_diameters_still(self._panel.collect()):
                self._show_what_the_panel_holds()
        finally:
            self._busy = False

    def _hold_the_diameters_still(self, current, magnification=None) -> int:
        """With calculating off, write numbers into the blank diameters.

        A blank diameter is the magnification default and would move with
        the next magnification change, which "Calculate diameters" off
        promises not to do. Each blank one is written as the number it was
        showing, at ``magnification`` -- the magnification the table drew
        it at -- so the run segments at what the user saw.

        Skipped while a settings file or preset is being applied: its values
        arrive one widget at a time, and freezing a diameter halfway would
        pin it at a magnification the file has not finished setting.

        :param current: the panel's settings now.
        :param magnification: the magnification the blanks were drawn at;
            the panel's own when ``None``.
        :returns: how many diameters were written.
        """
        if getattr(self._panel, "_applying_settings", False):
            return 0
        held = _blank_diameters_held_still(current, magnification)
        written = 0
        for key, number in held.items():
            if self._panel.set_value_for_key(key, number):
                written += 1
        return written

    def _calculate_clicked(self, *_args) -> None:
        """The user turned "Calculate diameters"; on, recalculate all three.

        Connected to the switch's ``clicked``, which only a user's click or
        drag sends -- never a settings file or preset being applied.

        Turning it on is the reset for pinned diameters: the cell, nucleus
        and pathogen diameters go back to blank, which is their
        magnification default, and follow magnification from here on.
        Turning it off is handled where every form change is, by
        :meth:`_hold_the_diameters_still`.
        """
        if self._busy:
            return
        try:
            current = self._panel.collect()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not read the panel", exc_info=True)
            return
        if not _calculates_diameters(current):
            return
        self._busy = True
        try:
            for obj in _DIAMETER_OBJECTS:
                key = f"{obj}_diameter"
                if key in current and current[key] is not None:
                    self._panel.set_value_for_key(key, None)
            self._show_what_the_panel_holds()
        finally:
            self._busy = False

    def write_through(self) -> Dict[str, Any]:
        """Write every changed cell into its widget; return what changed.

        Only changed values are written. Setting a widget to the value it already holds
        still makes it emit, and a panel that re-validates on every emit
        would do the whole form's work on each keystroke in the table.
        """
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
    if isinstance(a, (dict, str)) and isinstance(b, (dict, str)) and (
            isinstance(a, dict) or isinstance(b, dict)):
        return _filters_of(a) == _filters_of(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def _filters_of(value: Any) -> Any:
    """``object_filters`` in one canonical shape, for comparing two spellings.

    The form may hold the mapping as text and the table as a dict; both are
    read into ``{object: [{property, min, max}, ...]}`` with empty lists
    dropped. A value that is not an ``object_filters`` mapping comes back
    unchanged.
    """
    try:
        from spacr.qt.mask_engine import (normalise_filters,
                                          parse_object_filters)

        parsed = parse_object_filters(value)
        return {str(obj): normalise_filters(rows, strict=False)
                for obj, rows in sorted(parsed.items()) if rows}
    except Exception:                                        # noqa: BLE001
        return value
