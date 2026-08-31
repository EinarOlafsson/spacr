"""The merged Classify panel's group order, and the row-visibility walk.

Dict order IS the panel order here: the family choice first, then the
shared groups, then each family's own settings under a heading that names
the family. Four of the five loops that rebuild it had never gone round
on a name the panel does not have -- and they cannot, which is the point.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget

from spacr.qt.screens import settings_model as SM

pytestmark = pytest.mark.qt


class TestTheMergedClassifyGroupOrder:

    def _panel_groups(self):
        """The section TITLES the merged panel builds.

        ``SettingsWidgets`` is not a QWidget -- it owns them -- and the
        groups reach the screen as sections, so that is where the order
        is read back from. A family group's title carries its family
        prefix, which is stripped here to compare against the tuples.
        """
        widgets = SM.SettingsWidgets("classify_merged")
        sections = widgets.build_sections()
        titles = [str(getattr(section, "title", "")) for section in sections]
        bare = set()
        for title in titles:
            for separator in (" — ", " -- ", " - "):
                if separator in title:
                    title = title.split(separator, 1)[1]
                    break
            bare.add(title)
        return widgets, bare

    def test_the_family_choice_comes_first(self):
        widgets = SM.SettingsWidgets("classify_merged")
        sections = widgets.build_sections()

        assert str(getattr(sections[0], "title", "")) == "Classifier", (
            "the panel asks which model to train after asking how to "
            "train it")

        source = inspect.getsource(SM)
        assert 'rebuilt = {"Classifier": ["classifier_family"]}' in source, (
            "the family choice is no longer the first group, so the panel "
            "asks which model to train after asking how to train it")

    def test_every_named_group_exists_in_the_panel(self):
        """THE PIN, for four ``if name in ordered`` tests.

        The five tuples name the groups by hand, and every name in them
        is a key the builder above has already put into ``ordered`` --
        which is why the catch-all that copied unnamed groups was
        removed as unreachable, and the same reasoning applies to the
        membership tests themselves.

        Keeping them is right: a group renamed in the builder and not in
        the tuple would otherwise be a KeyError while the panel is being
        laid out, and the panel is what the user is looking at. But a
        rename would ALSO silently drop that group from the order, which
        is the quieter failure -- so this compares the two lists rather
        than trusting either.
        """
        source = inspect.getsource(SM)
        block = source[source.index('if app_key == "classify_merged":'):]
        block = block[:block.index("rebuilt[name] = ordered[name]",
                                   block.index("shared_last")) + 40]

        named = set()
        for tuple_name in ("cv_groups", "ml_groups", "shared_first",
                           "shared_last"):
            start = block.index(f"{tuple_name} = (")
            literal = block[start + len(f"{tuple_name} = "):]
            literal = literal[:literal.index(")") + 1]
            named.update(
                part.strip().strip('",\'')
                for part in literal.strip("()").split('",')
                if part.strip().strip('",\''))

        assert named, "the group tuples are gone"
        _widgets, groups = self._panel_groups()
        assert groups, "the merged panel built no sections at all"

        missing = sorted(name for name in named if name not in groups)
        assert not missing, (
            f"{missing} are named in the order tuples but are not groups the "
            f"panel builds, so they are silently dropped from the order")

    def test_the_shared_groups_are_not_prefixed_with_a_family(self):
        """"Labels & Classes" applies to both families, and prefixing it
        would imply it belonged to one."""
        source = inspect.getsource(SM)
        block = source[source.index("shared_first = ("):]
        block = block[:block.index("shared_last")]
        assert "_family_heading" not in block, (
            "a shared group is now prefixed with a family name")


class TestWalkingUpToARowToHideIt:
    """``SettingsWidgets._set_row_visible``, driven through the real thing.

    THE ROW, NOT THE FIELD: the screen keeps the label side inside a
    wrapper it does not hand back, so hiding the field alone strands its
    name on an empty row.
    """

    def _widgets(self):
        """A built panel: the fields exist only after build_sections."""
        widgets = SM.SettingsWidgets("mask")
        widgets.build_sections()
        assert widgets._widgets, "the panel built no fields"
        return widgets

    def _nest(self, qtbot, widget, depth):
        """Put ``widget`` ``depth`` layouts below a real QFormLayout."""
        root = QWidget()
        qtbot.addWidget(root)
        form = QFormLayout(root)
        holder = QWidget()
        form.addRow("a label", holder)

        node = holder
        for _ in range(depth):
            child = QWidget(node)
            QVBoxLayout(node).addWidget(child)
            node = child
        if node is not holder:
            QVBoxLayout(node).addWidget(widget)
        else:
            QVBoxLayout(holder).addWidget(widget)
        return root, form, holder

    def test_a_field_one_step_below_the_form_is_hidden_as_a_row(self, qtbot):
        """The ROW goes, not just the field.

        The screen keeps the label side in a wrapper it does not hand
        back, so hiding the field alone strands its name on an empty
        row. ``setRowVisible`` reaches both halves, and what this
        asserts is that the form was asked at all -- the holder is the
        widget the form knows, and it is the one the walk has to find.
        """
        widgets = self._widgets()
        key = next(iter(widgets._widgets))
        field = widgets._widgets[key]
        _root, form, holder = self._nest(qtbot, field, depth=0)

        row, _role = form.getWidgetPosition(holder)
        assert row >= 0, "the holder is not the widget the form knows"

        widgets._set_row_visible(key, False)
        assert form.isRowVisible(row) is False, (
            "the field was hidden without its label, which strands the name "
            "on an empty row")

        widgets._set_row_visible(key, True)
        assert form.isRowVisible(row) is True

    def test_a_field_nested_deeper_than_three_falls_through_the_walk(
            self, qtbot):
        """THE UNCOVERED ARC: the walk runs out of steps.

        Three is the deepest the panel nests a field -- field, the
        button holder, the section body that owns the form -- and the
        bound is what stops a widget reparented into an unexpected tree
        walking to the top of the application on every visibility
        change. Past it the walk gives up and the field is hidden on its
        own, which is a frame late rather than wrong: the scheduled pass
        catches up.
        """
        widgets = self._widgets()
        key = next(iter(widgets._widgets))
        field = widgets._widgets[key]
        # The root is held, not discarded: it owns every widget below it,
        # and letting it go takes the field's C++ side with it.
        root, _form, _holder = self._nest(qtbot, field, depth=5)

        widgets._set_row_visible(key, False)     # must not raise

        assert root is not None
        assert field.parentWidget() is not None

    def test_a_parentless_field_is_left_alone(self, qtbot):
        """The line the walk falls through to.

        ``SettingsWidgets`` is built with no parent by everything that
        wants the VALUES rather than a form, and showing a parentless
        widget would paint a window of its own on the next turn of the
        event loop -- mid-construction, long after the panel that made
        it was finished.
        """
        widgets = self._widgets()
        key = next(iter(widgets._widgets))
        field = widgets._widgets[key]
        assert field.parentWidget() is None, (
            "a bare SettingsWidgets now parents its fields")

        # A widget that was never shown reports isHidden() either way, so
        # what is asserted is that NOTHING CHANGED -- setVisible was not
        # called at all.
        calls = []
        field.setVisible = lambda shown: calls.append(shown)

        widgets._set_row_visible(key, False)

        assert calls == [], (
            "a parentless field was told to change visibility, which opens "
            "a window of its own on the next turn of the event loop")

    def test_a_key_the_panel_does_not_have_is_ignored(self):
        widgets = self._widgets()

        widgets._set_row_visible("not_a_setting", False)   # must not raise
