"""A section can open on construction, and a string label still right-aligns.

Both are constructor-time choices a settings screen makes once and never
revisits, so a regression in either shows up as a whole category rendering
wrong from the moment it is built -- collapsed when the screen asked for it
open, or a wall of left-hugging labels with the page showing through.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QCheckBox, QFormLayout, QLabel, QWidget  # noqa: E402

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


def test_a_string_label_gets_a_real_qlabel_inside_the_alignment_host(qapp):
    """Wrapping a plain string must produce a QLabel carrying that string.

    ``wrap_label`` exists to build the right-aligning host even with no info
    dot. If the string branch failed to add a label the row would be built,
    the form would report a row, and the caption would simply be missing --
    a blank cell rather than an exception.
    """
    holder = QWidget()
    section = Section("Embedding & Clustering", holder)
    field = QCheckBox("on", holder)

    section.add_row("Cell diameter", field, wrap_label=True)

    host = section._form.itemAt(0, QFormLayout.LabelRole).widget()
    assert host.objectName() == "SettingLabelWithInfo"
    captions = [w.text() for w in host.findChildren(QLabel)]
    assert captions == ["Cell diameter"]
    # The caller's own label, not the host, is what rows read back as.
    assert section._row_widgets == [("Cell diameter", field)]


def test_a_widget_label_is_hosted_as_itself_rather_than_copied(qapp):
    """A caller that passes a live QLabel must get that same object back.

    Screens attach tooltips and API help to the label they created; rebuilding
    it from its text would silently drop all of that.
    """
    holder = QWidget()
    section = Section("Embedding & Clustering", holder)
    caption = QLabel("Cell diameter", holder)
    field = QCheckBox("on", holder)

    section.add_row(caption, field, wrap_label=True)

    host = section._form.itemAt(0, QFormLayout.LabelRole).widget()
    assert caption in host.findChildren(QLabel)
    assert section._row_widgets == [(caption, field)]
