"""A section can open on construction, and a string label still right-aligns.

Both are constructor-time choices a settings screen makes once and never
revisits, so a regression in either shows up as a whole category rendering
wrong from the moment it is built -- collapsed when the screen asked for it
open, or a wall of left-hugging labels with the page showing through.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt.widgets.section import Section  # noqa: E402

pytestmark = pytest.mark.qt


def test_a_section_asked_to_start_open_is_open_before_anyone_clicks(qapp):
    """``expanded=True`` must show the body and point the chevron down.

    A screen that opens one category by default relies on the constructor
    doing the whole toggle, not just remembering a flag: the arrow, the body's
    visibility and the header's checked state all have to agree, or the first
    user click closes an already-closed-looking section.
    """
    holder = QWidget()
    section = Section("Plate Layout & Controls", holder, expanded=True)

    assert section.is_expanded() is True
    assert section.header().isChecked() is True
    assert section._body.isVisibleTo(section) is True

    collapsed = Section("Plate Layout & Controls", holder)
    assert collapsed.is_expanded() is False
    assert collapsed._body.isVisibleTo(collapsed) is False


