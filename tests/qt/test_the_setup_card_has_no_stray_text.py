"""The GPU note belongs in its band, not in the corner of the card.

The note is shown on the first slide, and the only thing that laid it out
gave up on a note that was not visible YET -- while running before the
slide made it visible. So the label was shown having never been placed,
keeping the geometry a QLabel is born with: the top left corner, where it
read as a stray sentence above the language list and went away only when
the first Next hid it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.setup_slides import (                    # noqa: E402
    GPU_NOTE_BAND, GPU_REQUIREMENT, SetupSlides,
)

CARD_MARGIN = 28


@pytest.fixture
def slides(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    window = SetupSlides()
    qtbot.addWidget(window)
    window.resize(900, 640)
    window.show()
    qtbot.waitExposed(window)
    return window


def test_the_gpu_note_is_not_in_the_corner(slides):
    """The corner is where an unplaced QLabel sits, and it is the defect."""
    note = slides._gpu_note
    assert note.isVisible(), "the note is shown on the first slide"
    assert note.geometry().topLeft() != note.geometry().topLeft().__class__(0, 0)


def test_the_gpu_note_sits_in_its_own_band(slides):
    """At the band, or lifted off the buttons -- never below either.

    THE BAND IS A CEILING, NOT A COORDINATE, and this used to assert
    equality. `_place_the_gpu_note` clamps: the note grew a capability
    table on 2026-08-31, and on a short card the band plus the wrapped
    height put it straight over Back and the step counter -- "how a label
    that is only decoration ends up eating a button". So it LIFTS rather
    than overflowing, and at 900x640 it sits at 487 where the bare band
    would be 499.

    Asserting the equality again would mean choosing a stray label over a
    reachable button. What is asserted instead is the property the clamp
    exists for: at or above the band, and clear of the nav row.
    """
    note, card = slides._gpu_note, slides.card
    band = int(card.height() * GPU_NOTE_BAND)
    assert note.geometry().y() <= band, (
        "the note must never sit BELOW its band -- that is the stray-label "
        "defect this file is about")
    assert note.geometry().x() == CARD_MARGIN
    assert note.width() == card.width() - 2 * CARD_MARGIN

    back = getattr(slides, "_back", None)
    if back is not None and back.parent() is not None:
        top_of_buttons = back.mapTo(card, back.rect().topLeft()).y()
        assert note.geometry().bottom() <= top_of_buttons, (
            "the note is over the Back button, which is the defect the "
            "clamp was added for")


def test_showing_the_note_places_it(slides):
    """The defect was an ordering one, and this is the ordering.

    The placement used to bail out on a note that was not visible YET, and
    it ran before the slide made the note visible -- so the label was shown
    having never been laid out. Whatever the order, becoming visible has to
    be enough to put it in its band.
    """
    note, card = slides._gpu_note, slides.card
    note.setVisible(False)
    note.move(0, 0)
    slides._place_the_greeting()          # the pass that used to give up
    note.setVisible(True)
    slides._place_the_gpu_note()
    # At or above the band, for the reason
    # `test_the_gpu_note_sits_in_its_own_band` sets out: the band is a
    # ceiling and the clamp lifts the note off the buttons.
    assert 0 < note.geometry().y() <= int(card.height() * GPU_NOTE_BAND)


def test_the_note_stays_in_its_band_when_the_card_is_resized(slides, qtbot):
    """A resize re-places it rather than stranding it at the old height.

    Asserted as "at or below the band, and moved with the card", not as an
    equality. The band is a CEILING: the clamp lifts the note when the
    wrapped height would put it over the nav row, and since the note grew
    its capability table that happens at any card shorter than about 1000
    px -- including the 820 this resizes to. Demanding the exact band here
    tested the clamp and called it a stranding.
    """
    slides.resize(1100, 620)
    qtbot.wait(30)
    slides._place_the_gpu_note()
    before = slides._gpu_note.geometry().y()

    slides.resize(1100, 820)
    qtbot.wait(30)
    slides._place_the_gpu_note()
    note, card = slides._gpu_note, slides.card
    assert note.geometry().y() != before, "the note did not follow the resize"
    assert 0 < note.geometry().y() <= int(card.height() * GPU_NOTE_BAND)
    assert note.geometry().bottom() <= card.height(), "it hangs off the card"


def test_the_note_carries_the_requirement_sentence(slides):
    """What is drawn there is the GPU sentence, not something else."""
    assert GPU_REQUIREMENT.split(".")[0] in slides._gpu_note.text()


def test_the_note_leaves_with_the_first_slide(slides):
    """It is a first-slide note; the later slides must not carry it."""
    slides._show_slide(1) if hasattr(slides, "_show_slide") else None
    if hasattr(slides, "_show_slide"):
        assert not slides._gpu_note.isVisible()


def test_gpu_note_measures_its_actual_scaled_width(qtbot):
    from types import SimpleNamespace
    from PySide6.QtCore import QSize, Qt
    from PySide6.QtGui import QFont, QTextDocument
    from PySide6.QtWidgets import QLabel, QWidget
    from spacr.qt import gui_scale

    previous = gui_scale.current_scale()
    gui_scale.install_scaling_layer()
    gui_scale.set_gui_scale_live(.9)
    card = QWidget()
    qtbot.addWidget(card)
    gui_scale._resize_actual(card, QSize(648, 504))
    note = QLabel(card)
    note.setWordWrap(True)
    note.setTextFormat(Qt.RichText)
    font = QFont('Helvetica Neue')
    font.setPixelSize(23)
    note.setFont(font)
    note.setText('<div>' + GPU_REQUIREMENT + '</div><div>'
                 + ('GPU acceleration and CPU fallback retain every task. ' * 5)
                 + '</div>')
    card.show()
    note.show()
    try:
        SetupSlides._place_the_gpu_note(SimpleNamespace(card=card, _gpu_note=note, _back=None))
        document = QTextDocument()
        document.setDefaultFont(note.font())
        document.setDocumentMargin(0)
        document.setHtml(note.text())
        document.setTextWidth(note.contentsRect().width())
        assert document.size().height() <= note.contentsRect().height() + 1
        assert note.width() == card.width() - 2 * CARD_MARGIN
        assert note.geometry().bottom() < card.height()
    finally:
        card.close()
        gui_scale.set_gui_scale_live(previous)


def test_joint_maximum_setup_opens_on_screen_with_scrollable_card(qtbot, qapp, monkeypatch):
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QAbstractButton, QAbstractScrollArea, QDialog, QLabel
    from spacr.qt import gui_scale, theme

    old_scale, old_sheet = gui_scale.current_scale(), qapp.styleSheet()
    gui_scale.install_scaling_layer()
    gui_scale.set_gui_scale_live(2.0)
    qapp.setStyleSheet(theme.stylesheet('dark', font_scale=2.0))
    monkeypatch.setattr(SetupSlides, '_install_backdrop', lambda self: None)
    from spacr.qt.widgets.setup_slides import GPU_MAC_CPU_HINT
    note_text = ('<div>' + GPU_REQUIREMENT + '</div><div>GPU: Intel Iris Plus Graphics: Metal</div><div>'
                 + GPU_MAC_CPU_HINT + '</div><table><tr><td>Cellpose 4</td><td>CPU</td><td>Segmentation</td></tr>'
                 '<tr><td>Torch models</td><td>CPU</td><td>Classification</td></tr>'
                 '<tr><td>Live backdrop</td><td>GPU</td><td>Visualization</td></tr>'
                 '<tr><td>UMAP / t-SNE / cluster</td><td>CPU</td><td>Machine learning</td></tr></table>')
    monkeypatch.setattr(SetupSlides, '_say_what_the_gpu_is', lambda self: self._gpu_note.setText(note_text))
    dialog = SetupSlides()
    qtbot.addWidget(dialog)
    try:
        dialog.show()
        qtbot.waitExposed(dialog)
        qapp.processEvents()
        handle = dialog.windowHandle()
        assert handle.screen().availableGeometry().contains(handle.frameGeometry())
        ancestor = dialog.card.parentWidget()
        while ancestor is not None and not isinstance(ancestor, QAbstractScrollArea):
            ancestor = ancestor.parentWidget()
        assert ancestor is not None, 'the actual floating card must be the scrolled content'
        assert ancestor.horizontalScrollBar().maximum() > 0 or ancestor.verticalScrollBar().maximum() > 0
        viewport = ancestor.viewport()
        mapped = dialog._next.rect().translated(dialog._next.mapTo(viewport, QPoint()))
        horizontal, vertical = ancestor.horizontalScrollBar(), ancestor.verticalScrollBar()
        assert mapped.right() + horizontal.value() <= viewport.rect().right() + horizontal.maximum()
        assert mapped.bottom() + vertical.value() <= viewport.rect().bottom() + vertical.maximum()
        note = dialog._gpu_note
        for control in dialog.findChildren(QLabel) + dialog.findChildren(QAbstractButton):
            if control is note or not control.isVisibleTo(dialog) or not control.text():
                continue
            mapped = control.rect().translated(control.mapTo(dialog.card, QPoint()))
            assert not note.geometry().intersects(mapped), (control.text(), note.geometry(), mapped)
    finally:
        QDialog.reject(dialog)
        qapp.setStyleSheet(old_sheet)
        gui_scale.set_gui_scale_live(old_scale)


def test_small_gui_large_font_setup_keeps_all_pages_whole(qtbot, qapp, monkeypatch):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QRegion, QTextDocument, QTextOption
    from PySide6.QtWidgets import QAbstractButton, QAbstractScrollArea, QComboBox, QDialog, QLabel
    from spacr.qt import gui_scale, theme
    from spacr.qt.widgets.setup_slides import SLIDES, GPU_MAC_CPU_HINT

    old_scale, old_sheet = gui_scale.current_scale(), qapp.styleSheet()
    gui_scale.install_scaling_layer()
    gui_scale.set_gui_scale_live(.1)
    qapp.setStyleSheet(theme.stylesheet('dark', font_scale=2.0))
    monkeypatch.setattr(SetupSlides, '_install_backdrop', lambda self: None)
    text = ('<div>' + GPU_REQUIREMENT + '</div><div>GPU: Intel Iris Plus Graphics: Metal</div><div>'
            + GPU_MAC_CPU_HINT + '</div><table><tr><td>Cellpose 4</td><td>CPU</td><td>Segmentation</td></tr>'
            '<tr><td>Live backdrop</td><td>GPU</td><td>Visualization</td></tr></table>')
    monkeypatch.setattr(SetupSlides, '_say_what_the_gpu_is', lambda self: self._gpu_note.setText(text))
    dialog = SetupSlides()
    qtbot.addWidget(dialog)
    try:
        dialog.show()
        qtbot.waitExposed(dialog)
        qapp.processEvents()
        assert dialog._editors['language'].font().pixelSize() == 3
        assert dialog.height() >= dialog.card.minimumSizeHint().height()
        for index in range(len(SLIDES)):
            dialog._show_slide(index, fade=False)
            qapp.processEvents()
            for control in dialog.findChildren(QLabel) + dialog.findChildren(QAbstractButton) + dialog.findChildren(QComboBox):
                if not control.isVisibleTo(dialog):
                    continue
                assert not control.rect().isEmpty()
                whole = QRegion(control.rect()).subtracted(control.visibleRegion()).isEmpty()
                ancestor = control.parentWidget()
                while ancestor is not None and ancestor is not dialog and not isinstance(ancestor, QAbstractScrollArea):
                    ancestor = ancestor.parentWidget()
                reachable = False
                if isinstance(ancestor, QAbstractScrollArea):
                    viewport = ancestor.viewport()
                    mapped = control.rect().translated(control.mapTo(viewport, QPoint()))
                    area = viewport.rect()
                    horizontal, vertical = ancestor.horizontalScrollBar(), ancestor.verticalScrollBar()
                    reachable = (mapped.left() + horizontal.value() >= area.left() + horizontal.minimum()
                                 and mapped.right() + horizontal.value() <= area.right() + horizontal.maximum()
                                 and mapped.top() + vertical.value() >= area.top() + vertical.minimum()
                                 and mapped.bottom() + vertical.value() <= area.bottom() + vertical.maximum()
                                 and not area.contains(mapped))
                assert whole or reachable, (index, type(control).__name__, control.rect())
                if isinstance(control, QLabel) and control.text():
                    document = QTextDocument()
                    document.setDocumentMargin(0)
                    document.setDefaultFont(control.font())
                    option = document.defaultTextOption()
                    option.setWrapMode(QTextOption.WordWrap if control.wordWrap() else QTextOption.NoWrap)
                    document.setDefaultTextOption(option)
                    document.setHtml(control.text())
                    document.setTextWidth(control.contentsRect().width() if control.wordWrap() else -1)
                    assert document.size().height() <= control.contentsRect().height() + 1
            if index == 0:
                note = dialog._gpu_note
                for control in [dialog._title, dialog._blurb, dialog._next, dialog._back,
                                dialog._editors['language']]:
                    mapped = control.rect().translated(control.mapTo(dialog.card, QPoint()))
                    assert not note.geometry().intersects(mapped)
                dialog._show_the_greeting()
                if dialog._hello is not None:
                    dialog._hello.setCurrentTime(dialog._hello.duration())
                qapp.processEvents()
                greeting = dialog._greeting
                assert QRegion(greeting.rect()).subtracted(greeting.visibleRegion()).isEmpty()
                assert not greeting.geometry().intersects(note.geometry())
                for control in [dialog._title, dialog._blurb, dialog._editors['language']]:
                    mapped = control.rect().translated(control.mapTo(dialog.card, QPoint()))
                    assert not greeting.geometry().intersects(mapped)
    finally:
        QDialog.reject(dialog)
        qapp.setStyleSheet(old_sheet)
        gui_scale.set_gui_scale_live(old_scale)


@pytest.mark.parametrize('font_scale', [.9, 2.0])
def test_fitting_setup_opening_retains_overlay_and_nominal_size(qtbot, qapp, monkeypatch, font_scale):
    from PySide6.QtCore import QSize
    from spacr.qt import gui_scale, theme
    from PySide6.QtWidgets import QDialog

    old_scale, old_sheet = gui_scale.current_scale(), qapp.styleSheet()
    gui_scale.install_scaling_layer()
    gui_scale.set_gui_scale_live(.9)
    qapp.setStyleSheet(theme.stylesheet('dark', font_scale=font_scale))
    monkeypatch.setattr(SetupSlides, '_install_backdrop', lambda self: None)
    monkeypatch.setattr(SetupSlides, '_say_what_the_gpu_is', lambda self: self._gpu_note.setText(GPU_REQUIREMENT))
    dialog = SetupSlides()
    qtbot.addWidget(dialog)
    try:
        assert not getattr(dialog, '_gpu_note_in_layout', False)
        assert not getattr(dialog, '_card_scrolls', False)
        assert dialog.size() == QSize(648, 504)
        dialog.show()
        qtbot.waitExposed(dialog)
        assert not getattr(dialog, '_gpu_note_in_layout', False)
        assert dialog.size() == QSize(648, 504)
        if font_scale == 2.0:
            from PySide6.QtWidgets import QAbstractScrollArea

            dialog._show_slide(2)
            qapp.processEvents()
            assert dialog._pages.currentIndex() == 2
            assert dialog._card_scrolls
            scroll = dialog.card.parentWidget()
            while scroll is not None and not isinstance(scroll, QAbstractScrollArea):
                scroll = scroll.parentWidget()
            assert scroll is not None
            assert dialog.windowHandle().screen().availableGeometry().contains(
                dialog.windowHandle().frameGeometry())
            card_parent = dialog.card.parentWidget()
            dialog._show_slide(1)
            dialog._show_slide(2)
            assert dialog.card.parentWidget() is card_parent
    finally:
        QDialog.reject(dialog)
        qapp.setStyleSheet(old_sheet)
        gui_scale.set_gui_scale_live(old_scale)
