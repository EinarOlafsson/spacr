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

from spacr.qt.widgets.class_editor import (              # noqa: E402
    ClassChip, ClassEditorWidget)
from spacr.qt.widgets.sortable_table import tree_item    # noqa: E402


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
