"""Greying a setting must not delete its documentation.

Two passes grey settings out — the training basis and the classifier
family — and both replaced the setting's API-linked help with a plain
sentence. The field kept its documentation; the label, which composes its
help from the control's tooltip, ended up with the note alone.
"""

import pytest

from spacr.qt.screens.settings_model import (
    _apply_greyed_note, _basis_note, _clear_greyed_note, _family_note,
)

from PySide6.QtWidgets import QLineEdit

HELP = '<b>Annotation column</b><br>Name of the column. <a href="x">API</a>'


@pytest.fixture
def control(qt_theme_applied, qtbot):
    widget = QLineEdit()
    qtbot.addWidget(widget)
    widget.setProperty("apiTooltipHtml", HELP)
    widget.setToolTip(HELP)
    return widget


class TestTheNoteIsAdditive:

    def test_the_api_link_survives_the_note(self, control):
        """The regression. Before this, the tooltip WAS the note."""
        _apply_greyed_note(control, _basis_note("metadata"))
        assert "href=" in control.toolTip()

    def test_the_note_is_actually_shown(self, control):
        """Preserving the help must not lose the reason it is greyed."""
        _apply_greyed_note(control, _basis_note("metadata"))
        assert "Not used when the training basis is 'metadata'" in (
            control.toolTip())

    def test_the_documented_help_properties_are_left_alone(self, control):
        """Downstream composes labels from these. If the note leaks into
        them it becomes the setting's description for good."""
        _apply_greyed_note(control, _basis_note("metadata"))
        assert control.property("apiTooltipHtml") == HELP

    def test_re_greying_does_not_stack_notes(self, control):
        for basis in ("metadata", "metadata", "images"):
            _apply_greyed_note(control, _basis_note(basis))
        assert control.toolTip().count("The value is kept") == 1

    def test_a_changed_reason_replaces_the_old_one(self, control):
        _apply_greyed_note(control, _basis_note("metadata"))
        _apply_greyed_note(control, _basis_note("images"))
        assert "'images'" in control.toolTip()
        assert "'metadata'" not in control.toolTip()

    def test_becoming_applicable_again_restores_the_help_exactly(self,
                                                                 control):
        _apply_greyed_note(control, _basis_note("metadata"))
        _clear_greyed_note(control)
        assert control.toolTip() == HELP

    def test_clearing_a_control_that_was_never_greyed_is_a_no_op(self,
                                                                 control):
        _clear_greyed_note(control)
        assert control.toolTip() == HELP

    def test_a_control_with_no_help_still_gets_the_note(self,
                                                        qt_theme_applied,
                                                        qtbot):
        """Not every setting has documentation. The note is all there is."""
        bare = QLineEdit()
        qtbot.addWidget(bare)
        _apply_greyed_note(bare, _family_note("cv"))
        assert bare.toolTip() == _family_note("cv")

    def test_both_greying_reasons_read_as_sentences(self):
        assert "training basis is 'metadata'" in _basis_note("metadata")
        assert "'cv' classifier" in _family_note("cv")
