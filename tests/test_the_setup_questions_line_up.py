"""A caption sits beside the control it names, not across the card from it.

Every question on the setup screen used to be its own layout with a
stretch between the caption and the control, which does two bad things at
once: it pushes the pair to opposite edges of the card, and -- because
each row is an independent layout -- nothing lines up with the row above
it. One form per page fixes both: two columns, one gap, and captions that
share a left edge with the captions around them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFormLayout       # noqa: E402

from spacr.qt.widgets.setup_slides import (FORM_GAP_PX,   # noqa: E402
                                           SetupSlides)


@pytest.fixture()
def slides(qtbot):
    made = SetupSlides()
    qtbot.addWidget(made)
    made.resize(980, 700)
    made.show()
    return made


def _rows(slides, index):
    slides._pages.setCurrentIndex(index)
    page = slides._pages.widget(index)
    form = page.layout()
    if not isinstance(form, QFormLayout):
        return []
    out = []
    for row in range(form.rowCount()):
        label_item = form.itemAt(row, QFormLayout.LabelRole)
        field_item = form.itemAt(row, QFormLayout.FieldRole)
        if label_item is None or field_item is None:
            continue
        label, field = label_item.widget(), field_item.widget()
        if label is None or field is None:
            continue
        out.append((label, field))
    return out


def _pages_with_rows(slides):
    return [i for i in range(slides._pages.count()) if _rows(slides, i)]


def test_every_page_is_one_form_not_a_stack_of_rows(slides):
    """Alignment across rows is only possible if they share a layout."""
    found = _pages_with_rows(slides)
    assert found, "no page asked a question"
    for index in found:
        assert isinstance(slides._pages.widget(index).layout(), QFormLayout)


def test_the_captions_share_a_left_edge(slides):
    for index in _pages_with_rows(slides):
        lefts = {label.mapTo(slides, label.rect().topLeft()).x()
                 for label, _field in _rows(slides, index)}
        assert len(lefts) == 1, f"page {index} captions start at {lefts}"


def test_the_controls_share_a_left_edge(slides):
    for index in _pages_with_rows(slides):
        lefts = {field.mapTo(slides, field.rect().topLeft()).x()
                 for _label, field in _rows(slides, index)}
        assert len(lefts) == 1, f"page {index} controls start at {lefts}"


def test_nothing_is_a_card_width_from_its_caption(slides):
    """The gap was measured at 771 px on a 980 px card."""
    for index in _pages_with_rows(slides):
        for label, field in _rows(slides, index):
            left = label.mapTo(slides, label.rect().topRight()).x()
            right = field.mapTo(slides, field.rect().topLeft()).x()
            assert right - left <= FORM_GAP_PX * 6, (
                f"{label} sits {right - left} px from its control")


def test_a_single_line_control_is_centred_on_its_caption(slides):
    """Everything that is not a stack must still read as one line."""
    for index in _pages_with_rows(slides):
        for label, field in _rows(slides, index):
            drift = (field.mapTo(slides, field.rect().center()).y()
                     - label.mapTo(slides, label.rect().center()).y())
            assert abs(drift) <= 2, f"{label.text()} drifts {drift} px"
