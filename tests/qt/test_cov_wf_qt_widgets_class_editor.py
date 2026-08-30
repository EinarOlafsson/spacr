"""The class editor's guards against a class list that no longer lines up.

The ``classes`` setting decides what a classifier is trained on, and the
editor holds it in three places at once: ``self._rules``, a strip of chips
whose close marks carry a numeric index, and a hidden ``QTreeWidget`` whose
row order is meant to mirror the rules. Anything that lets those three drift
-- a chip left over from a previous redraw, a table row nothing backs, a
stretch item wedged into the chip strip, a settings CSV that stored a class
name as bare text -- must not delete the wrong class or leave the strip half
drawn. A class removed by accident here is a class missing from every figure
and every results table afterwards, with nothing on screen saying it went.

Offscreen Qt, no database, no dialogs.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.qt.widgets.class_editor import ClassChip, ClassEditorWidget  # noqa: E402
from spacr.qt.widgets.sortable_table import tree_item  # noqa: E402


@pytest.fixture
def editor(qtbot):
    """An editor over a two-value table, the shape the settings panel builds."""
    widget = ClassEditorWidget(
        frame=pd.DataFrame({"condition": ["nc", "pc", "nc"]}))
    qtbot.addWidget(widget)
    return widget


def _chips(widget):
    """The ClassChip widgets currently in the chip strip, in strip order."""
    layout = widget._chips_layout
    found = []
    for i in range(layout.count()):
        item = layout.itemAt(i)
        child = item.widget()
        if isinstance(child, ClassChip):
            found.append(child)
    return found


def _two_classes(widget):
    widget.set_value({"nc": {"column": "condition", "value": "nc"},
                      "pc": {"column": "condition", "value": "pc"}})


def test_a_class_name_stored_as_bare_text_is_not_run_through_the_parser(editor):
    """A settings CSV stores ``repr(value)``, so ``classes`` comes back as
    text. Text that opens a bracket is a Python literal and has to be parsed
    back into class names; text that does not is a user's own word, and
    handing it to ``ast.literal_eval`` would raise on every ordinary name.
    Both spellings have to leave the editor usable, and the bracketed one has
    to come back with its names intact -- that is the round trip that decides
    what a run trains on."""
    editor.set_value("['nc', 'pc']")
    assert list(editor.value()) == ["nc", "pc"]
    assert [r.column for r in editor.rules()] == ["?", "?"]

    editor.set_value("nc")
    # Bare text is neither a mapping nor a list, so nothing is claimed from
    # it -- and, crucially, no exception escapes set_value.
    assert editor.value() == {}
    assert editor.rules() == []
    assert editor.table.topLevelItemCount() == 0


def test_a_stale_chip_index_cannot_delete_a_class_it_does_not_name(editor):
    """Every chip carries the index it had when the strip was last drawn, and
    the close mark emits that number. After a redraw the numbers move, so a
    press that arrives late -- a queued click, a chip kept alive by a pending
    deleteLater -- names a row that is no longer there. Without the range
    guard a stale index of -1 would delete the LAST class instead of none,
    quietly changing what the run trains on. The in-range press must still
    remove exactly the class the user pointed at."""
    _two_classes(editor)
    assert [r.name for r in editor.rules()] == ["nc", "pc"]

    editor.remove_at(7)            # past the end
    editor.remove_at(-1)           # the Python index that would wrap around
    assert [r.name for r in editor.rules()] == ["nc", "pc"]
    assert [c.name_pill.text() for c in _chips(editor)] == ["nc", "pc"]

    # The same entry point, driven the way a user drives it, does remove one.
    _chips(editor)[0]._close.click()
    assert [r.name for r in editor.rules()] == ["pc"]
    assert [c.name_pill.text() for c in _chips(editor)] == ["pc"]


def test_the_remove_button_ignores_a_row_no_class_stands_behind(editor):
    """The hidden table is the accessible surface, and Remove works off the
    row index in it. A row that no rule backs -- one left over from a redraw
    that raced, or added by anything else holding the tree -- would index
    past the rules list, and ``del`` on that index raises IndexError, which
    from a button press is a traceback in the middle of the settings panel.
    The guard has to keep the classes exactly as they were, and Remove has to
    still work on a real row in the same editor."""
    editor.set_value({"nc": {"column": "condition", "value": "nc"}})
    ghost = tree_item(["ghost", "x", "condition"])
    editor.table.addTopLevelItem(ghost)
    assert editor.table.topLevelItemCount() == 2

    editor.table.setCurrentItem(ghost)
    editor.remove_selected()
    assert [r.name for r in editor.rules()] == ["nc"]
    assert editor.table.topLevelItemCount() == 2   # nothing was taken out

    # A row a rule does stand behind is removed, and the redraw takes the
    # unbacked row with it.
    editor.table.setCurrentItem(editor.table.topLevelItem(0))
    editor.remove_selected()
    assert editor.rules() == []
    assert editor.table.topLevelItemCount() == 0
    assert _chips(editor) == []


def test_the_chip_strip_is_cleared_even_where_it_holds_no_widget(editor):
    """The strip is torn down and rebuilt on every change rather than diffed,
    and the teardown takes items out one at a time. A layout item is not
    always a widget -- a stretch or a spacer is an item with no widget behind
    it -- and treating one as a widget while clearing would abort the redraw
    part way, leaving the chips of the previous edit on screen beside the new
    ones. The user would then be looking at classes that are no longer in the
    setting."""
    editor.set_value({"nc": {"column": "condition", "value": "nc"}})
    editor._chips_layout.addStretch(1)
    assert editor._chips_layout.count() == 2

    editor.add_random_complement()

    names = [c.name_pill.text() for c in _chips(editor)]
    assert names == ["nc", "rest"]
    assert editor._chips_layout.count() == 2       # the spacer is gone
    assert [c.value_pill.text() for c in _chips(editor)] == [
        "nc", "the rest, at random"]
    assert editor.value()["rest"]["random_complement"] is True


def test_a_second_everything_else_class_is_refused_with_a_reason(editor):
    """Two classes that both mean "the objects nothing else claimed" have no
    boundary between them, so the model is asked to separate one population
    from itself. The editor has to say why the press did nothing rather than
    adding a second one, and the first one has to survive the refusal."""
    editor.set_value({"nc": {"column": "condition", "value": "nc"}})
    editor.add_random_complement()
    assert list(editor.value()) == ["nc", "rest"]

    editor.add_random_complement()
    assert list(editor.value()) == ["nc", "rest"]
    assert "already a random-rest class" in editor._hint.text()


def test_adding_a_column_twice_adds_its_values_once(editor):
    """"Add values" is how a class per value gets made, and columns are meant
    to be additive -- a second column joins the first. That makes pressing
    the same column twice easy to do by accident, and a duplicated class
    would train two identically named groups on the same objects. The second
    press has to report that it added nothing rather than growing the list."""
    editor.column.setCurrentText("condition")
    editor.populate_from_column()
    assert list(editor.value()) == ["condition=nc", "condition=pc"]
    assert "added 2 value(s) from condition" in editor._hint.text()

    editor.populate_from_column()
    assert list(editor.value()) == ["condition=nc", "condition=pc"]
    assert editor._hint.text() == "condition adds nothing new"


def test_a_hand_typed_class_list_without_brackets_empties_the_table(editor):
    """``classes`` is stored as ``repr(value)``, so anyone editing the settings
    file by hand types what looks right -- ``nc,pc`` -- rather than a Python
    literal. Text with no opening bracket is not a literal and must never be
    handed to the parser, which would raise out of ``set_value`` and take the
    whole settings panel down while it is being populated. It must instead
    land as "nothing was claimed", clearing the classes that were there so the
    table cannot keep showing a list the setting no longer holds, and the
    editor must still accept a well-formed value straight afterwards."""
    _two_classes(editor)
    assert list(editor.value()) == ["nc", "pc"]

    editor.set_value("  nc,pc  ")
    assert editor.value() == {}
    assert editor.table.topLevelItemCount() == 0
    assert _chips(editor) == []

    editor.set_value({"nc": {"column": "condition", "value": "nc"}})
    assert [c.name_pill.text() for c in _chips(editor)] == ["nc"]


def test_a_close_press_left_over_from_a_longer_list_removes_nothing(editor):
    """Chips carry the index they had when the strip was drawn, and the strip
    is redrawn whenever the setting is reloaded -- switching to another
    settings file, or a measurement run writing its own classes back. A press
    that was already in flight then names a position past the end of the
    shorter list. ``del`` on it raises IndexError out of a signal handler,
    which is a traceback with no dialog behind it; and the index that the
    guard is really there for is the last one, which would otherwise silently
    delete a class the user never pointed at."""
    _two_classes(editor)
    editor.set_value({"nc": {"column": "condition", "value": "nc"}})
    assert [r.name for r in editor.rules()] == ["nc"]

    editor.remove_at(1)            # the position "pc" used to sit at
    assert [r.name for r in editor.rules()] == ["nc"]
    assert [c.name_pill.text() for c in _chips(editor)] == ["nc"]

    # The close mark on a chip that is actually drawn still removes its class.
    _chips(editor)[0]._close.click()
    assert editor.rules() == []
    assert _chips(editor) == []


def test_remove_ignores_a_selection_that_is_not_a_class_row(editor):
    """Remove reads the selected row's position among the top-level rows, and
    ``indexOfTopLevelItem`` answers -1 for anything that is not one -- a child
    row, or a row belonging to another tree. A negative index is a perfectly
    good Python index, so without the lower bound the Remove button would
    delete the LAST class whenever the selection was not a class row at all.
    That is a class gone from the training set with nothing on screen saying
    so, which is the failure this editor exists to prevent."""
    from PySide6.QtWidgets import QTreeWidgetItem

    _two_classes(editor)
    parent_row = editor.table.topLevelItem(0)
    parent_row.addChild(QTreeWidgetItem(["not a class"]))
    editor.table.setCurrentItem(parent_row.child(0))

    editor.remove_selected()
    assert [r.name for r in editor.rules()] == ["nc", "pc"]
    assert [c.name_pill.text() for c in _chips(editor)] == ["nc", "pc"]

    # A real class row under the same button does come out, last one first.
    editor.table.setCurrentItem(editor.table.topLevelItem(1))
    editor.remove_selected()
    assert [r.name for r in editor.rules()] == ["nc"]


def test_spacing_at_either_end_of_the_chip_strip_is_cleared_too(editor):
    """The strip is emptied item by item before it is redrawn, and layouts
    hold spacing and stretch alongside their widgets -- an item with nothing
    behind ``item.widget()``. The very first item taken can be one, so the
    guard has to hold on the opening pass and not only somewhere in the
    middle. Treating spacing as a widget would abort the teardown, and the
    chips of the previous edit would stay on screen next to the new ones,
    showing classes the setting no longer contains."""
    _two_classes(editor)
    editor._chips_layout.insertSpacing(0, 8)
    editor._chips_layout.addStretch(1)
    assert editor._chips_layout.count() == 4

    editor.add_random_complement()

    assert [c.name_pill.text() for c in _chips(editor)] == ["nc", "pc", "rest"]
    assert editor._chips_layout.count() == 3   # the spacing and stretch went
    assert list(editor.value()) == ["nc", "pc", "rest"]
