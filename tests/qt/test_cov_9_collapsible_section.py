"""An open section's floor moves with the panel inside it.

The splitter reads the section's minimum height to decide how little room it
may be dragged to. A panel that declares a taller floor while it is already
open has to have that floor applied immediately; applied only on the next
fold, the splitter would still be free to crush the open panel to the height
the previous content asked for.
"""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QLabel

from spacr.qt.widgets.collapsible_section import CollapsibleSection


@pytest.mark.qt
def test_an_open_sections_floor_changes_the_moment_it_is_set(qapp):
    """Setting the open minimum while open raises the widget's floor at once."""
    section = CollapsibleSection("Attached databases", QLabel("body"),
                                 expanded=True)

    section.set_open_minimum(180)

    assert section.is_expanded()
    assert section.minimumHeight() == 180 + CollapsibleSection.FOLDED_HEIGHT


@pytest.mark.qt
def test_a_folded_sections_floor_waits_until_it_opens(qapp):
    """While folded the section stays exactly the header's height.

    Applying the open floor to a folded section is what makes a fold look
    like it did nothing: the splitter hands the folded section back the space
    it just gave up.
    """
    section = CollapsibleSection("Attached databases", QLabel("body"),
                                 expanded=False)

    section.set_open_minimum(180)
    assert section.minimumHeight() == CollapsibleSection.FOLDED_HEIGHT

    section.set_expanded(True)
    assert section.minimumHeight() == 180 + CollapsibleSection.FOLDED_HEIGHT
