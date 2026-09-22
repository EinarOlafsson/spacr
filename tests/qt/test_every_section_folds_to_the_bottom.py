"""Item 471 slice C: a section inside a screen folds, and folded, sits at the bottom.

The pieces every screen uses -- :class:`FoldSection`, ``add_section`` and
:func:`fold_card` in :mod:`spacr.qt.widgets.collapsible_splitter` -- tested on
their own: the heading folds the body, a folded section's heading is at the
bottom of its room, a user's fold is remembered and an automatic one is not,
and folding never shows or builds a body its owner keeps hidden.
"""
from __future__ import annotations

import uuid

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt              # noqa: E402
from PySide6.QtGui import QMouseEvent                       # noqa: E402
from PySide6.QtWidgets import (QApplication, QLabel,        # noqa: E402
                               QPushButton, QVBoxLayout, QWidget)

from spacr.qt.widgets import collapsible_splitter as cs     # noqa: E402
from spacr.qt.widgets.card import Card, _CardBuiltWhenShown  # noqa: E402


def _pump(n: int = 8) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _click(label) -> None:
    at = QPointF(3, 3)
    QApplication.sendEvent(label, QMouseEvent(
        QEvent.MouseButtonRelease, at, at, Qt.LeftButton, Qt.LeftButton,
        Qt.NoModifier))


def _key() -> str:
    return f"test471c/{uuid.uuid4().hex[:8]}"


def _body(height=200):
    body = QWidget()
    body.setMinimumHeight(height)
    return body


@pytest.mark.qt
def test_a_heading_click_folds_the_section_and_it_sits_at_the_bottom(qapp):
    host = QWidget()
    column = QVBoxLayout(host)
    column.setContentsMargins(0, 0, 0, 0)
    section = cs.FoldSection(_body(), "Result table")
    column.addWidget(section, 1)
    host.resize(300, 500)
    host.show()
    _pump()
    assert section.heading.text().endswith("Result table")
    assert section.heading.objectName() == "FoldHeading"

    _click(section.heading)
    _pump()

    assert section.shut
    assert not section.body.isVisible()
    assert section.heading.geometry().bottom() >= section.height() - 4
    section.set_folded(False)
    _pump()
    assert not section.shut and section.body.isVisible()
    host.close()


@pytest.mark.qt
def test_a_user_fold_is_remembered_and_an_automatic_one_is_not(qapp):
    from spacr.qt.preferences import get_folded_panels, set_folded_panel

    key, other = _key(), _key()
    first = cs.FoldSection(_body(), "Gate table", persist_key=key)
    first.set_folded(True, by_user=True)
    automatic = cs.FoldSection(_body(), "Filter", persist_key=other)
    automatic.set_folded(True, by_user=False)

    assert get_folded_panels().get(key) is True
    assert not get_folded_panels().get(other)
    again = cs.FoldSection(_body(), "Gate table", persist_key=key)
    assert again.shut
    set_folded_panel(key, False)


@pytest.mark.qt
def test_the_section_follows_its_body_and_never_shows_it(qapp):
    host = QWidget()
    column = QVBoxLayout(host)
    body = _body()
    body.setVisible(False)
    section = cs.FoldSection(body, "Filter")
    column.addWidget(section)
    host.show()
    _pump()
    assert section.isHidden() and body.isHidden()

    section.set_folded(True)
    section.set_folded(False)
    assert body.isHidden(), "unfolding showed a body its owner hid"

    body.setVisible(True)
    _pump()
    assert not section.isHidden()
    body.setVisible(False)
    _pump()
    assert section.isHidden()

    stays = cs.FoldSection(_body(), "Always", follow_body=False)
    column.addWidget(stays)
    _pump()
    stays.body.setVisible(False)
    assert not stays.isHidden()
    stays.eventFilter(stays.body, QEvent(QEvent.HideToParent))
    assert not stays.isHidden()
    host.close()


@pytest.mark.qt
def test_a_lazy_card_in_a_section_stays_unbuilt(qapp):
    built = []
    card = _CardBuiltWhenShown(title="Hits")
    card.build_body_when_first_shown(lambda: built.append(1))
    card.hide()
    section = cs.FoldSection(card, "Hits")
    section.set_folded(True)
    section.set_folded(False)
    assert built == [] and not card.body_is_built()

    folder = cs.fold_card(card, persist_key="")
    folder.set_shut(True)
    folder.set_shut(False)
    assert built == []


@pytest.mark.qt
def test_heading_actions_stay_in_the_heading_row(qapp):
    refresh = QPushButton("Refresh")
    section = cs.FoldSection(_body(), "Preview", actions=[refresh, None])
    section.show()
    _pump()
    section.set_folded(True)
    _pump()
    assert refresh.isVisible()
    section.close()


@pytest.mark.qt
def test_a_section_in_a_splitter_drags_folds_and_keeps_its_size(qapp):
    key = _key()
    split = cs.CollapsibleSplitter(Qt.Vertical, persist_key=key)
    table = split.add_section(_body(80), "Table", stretch=0, extent=150,
                              persist_key="")
    image = split.add_section(_body(80), "Image")
    split.resize(300, 700)
    split.show()
    _pump()
    assert split.pane("Table").mode == cs.HEADER
    assert isinstance(table, cs.FoldSection)

    split.moveSplitter(260, 1)
    _pump()
    assert cs.get_pane_extents(key).get("Table", 0) > 200

    image.set_folded(True)
    _pump()
    assert not image.body.isVisible()
    assert split.sizes()[0] > 200, "the open table lost its dragged size"
    assert image.heading.mapTo(split, image.heading.rect().bottomLeft()).y() \
        >= split.height() - 8
    split.close()


@pytest.mark.qt
def test_lock_folded_to_bottom_twice_adds_one_stretch(qapp):
    section = cs.FoldSection(_body(), "Once")
    before = section.layout().count()
    cs.lock_folded_to_bottom(section, section.folder, Qt.Horizontal)
    section.set_folded(True)
    assert section.layout().count() == before + 1


@pytest.mark.qt
def test_lock_folded_to_bottom_survives_a_container_that_takes_no_attribute(
        qapp):
    class Frozen(QWidget):
        """A container whose lock record cannot be written."""

        _spacr_bottom_locks = property(lambda self: None)

    heading, body = QLabel("x"), QWidget()
    from spacr.qt.widgets.foldable import make_foldable

    folder = make_foldable(heading, body, name="x")
    frozen = Frozen()
    QVBoxLayout(frozen).addWidget(heading)
    cs.lock_folded_to_bottom(frozen, folder)
    folder.set_shut(True)
    assert frozen.layout().count() == 2


@pytest.mark.qt
def test_fold_card_folds_a_titled_card_and_leaves_an_untitled_one(qapp):
    card = Card(title="Dose-response curve")
    folder = cs.fold_card(card)
    assert folder is card.folder
    assert cs.fold_card(card) is folder
    folder.set_shut(True)
    assert not card.body.isVisibleTo(card)
    assert cs.fold_card(Card()) is None
    already = Card(title="QC", foldable=True)
    assert cs.fold_card(already) is already.folder


@pytest.mark.qt
def test_an_edge_pane_hint_says_what_its_drag_does(qapp):
    split = cs.CollapsibleSplitter(Qt.Horizontal)
    split.add_pane(QWidget(), "Chart")
    split.add_pane(QWidget(), "Sidebar", mode=cs.EDGE,
                   hint="The points do not move.")
    handle = split.handle(1)
    handle.retranslate_dynamic_content()
    assert handle.toolTip() == "Click to hide Sidebar. The points do not move."
    split.set_collapsed("Sidebar", True)
    handle.retranslate_dynamic_content()
    assert handle.toolTip().startswith("Click to show Sidebar again.")


@pytest.mark.qt
def test_a_section_paints_no_background_of_its_own(qapp):
    """The page's backdrop shows through a section, as through a card.

    The blanket ``QWidget`` rule fills every plain widget with the window
    colour, and a section is two plain widgets over whatever the screen
    draws. Without this the fold's own wrapper would paint a grey slab over
    the page.
    """
    from spacr.qt.theme import stylesheet

    sheet = stylesheet()
    assert "QWidget#FoldSection, QWidget#FoldSectionBody" in sheet
    section = cs.FoldSection(_body(), "See through me")
    assert section.objectName() == "FoldSection"
    assert section.folder.body.objectName() == "FoldSectionBody"
