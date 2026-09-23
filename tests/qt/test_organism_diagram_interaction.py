"""Upright compartment geometry, persistent selections and linked hover legends."""
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from spacr.qt.preferences import scaled_px
from spacr.qt.widgets.organism_diagram import OrganismDiagram, _COMPARTMENT_COLOURS
from spacr.qt.screens.organism_screen import OrganismScreen


@pytest.fixture(params=['toxoplasma', 'plasmodium'])
def diagram(request, qtbot, qt_theme_applied):
    root = Path(__file__).resolve().parents[2]
    made = OrganismDiagram(request.param, root/'spacr/resources/images/organism_apicomplexa.svg')
    qtbot.addWidget(made)
    made.resize(450, 850)
    made.show()
    qtbot.wait(20)
    return made


def row(diagram, code):
    return next(diagram.selector.item(i) for i in range(diagram.selector.count())
                if diagram.selector.item(i).data(Qt.UserRole) == code)


def rgb(artwork):
    image = artwork.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    pixels = np.frombuffer(image.bits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return pixels[:, :image.width()*4].reshape(image.height(), image.width(), 4)[:,:,:3].copy()


def test_apical_rhoptries_are_above_the_nucleus_in_actual_rendered_geometry(diagram):
    masks = diagram.artwork.masks()
    rhoptry_y, _ = np.where(masks['SL0233'][1])
    nucleus_y, _ = np.where(masks['SL0191'][1])
    assert len(rhoptry_y) and len(nucleus_y)
    assert rhoptry_y.mean() < nucleus_y.mean()
    size = diagram.artwork.renderer.viewBoxF().size()
    assert size.height() > size.width()


def test_several_checked_compartments_stay_filled_and_clear_independently(diagram):
    before = rgb(diagram.artwork)
    assert not np.any(before[:,:,0] != before[:,:,1])
    row(diagram, 'SL0233').setCheckState(Qt.Checked)
    row(diagram, 'SL0191').setCheckState(Qt.Checked)
    assert {'SL0233', 'SL0191'} <= diagram.artwork.selected
    colored = rgb(diagram.artwork)
    assert np.any(colored[:,:,0] != colored[:,:,1])
    row(diagram, 'SL0233').setCheckState(Qt.Unchecked)
    assert 'SL0191' in diagram.artwork.selected
    assert 'SL0233' not in diagram.artwork.selected
    assert row(diagram, 'SL0191').foreground().color() == QColor(_COMPARTMENT_COLOURS['SL0191'])
    diagram.clear_button.click()
    assert diagram.artwork.selected == set()
    assert not np.any(rgb(diagram.artwork)[:,:,0] != rgb(diagram.artwork)[:,:,1])


def test_hover_hits_the_shape_and_colors_the_legend_without_changing_selection(diagram, qtbot):
    art = diagram.artwork
    ys, xs = np.where(art.masks()['SL0233'][1])
    point = next(QPoint(int(x), int(y)) for x,y in zip(xs[::10], ys[::10])
                 if art.organelle_at(x,y) == 'SL0233')
    qtbot.mouseMove(art, QPoint(1, 1))
    qtbot.mouseMove(art, point)
    qtbot.waitUntil(lambda: art.hover_location == 'SL0233')
    assert art.selected == set()
    assert row(diagram, 'SL0233').foreground().color() == QColor(_COMPARTMENT_COLOURS['SL0233'])
    assert diagram.descriptions['SL0233'] in diagram.caption.text()
    qtbot.mouseClick(art, Qt.LeftButton, pos=point)
    assert 'SL0233' in art.selected
    QApplication.sendEvent(art, QEvent(QEvent.Leave))
    assert art.hover_location == ''
    assert 'SL0233' in art.selected


def test_thin_divider_and_packed_tile_columns(qtbot, qt_theme_applied):
    screen = OrganismScreen('toxoplasma')
    qtbot.addWidget(screen)
    screen.resize(1280, 850)
    screen.show()
    qtbot.wait(30)
    assert screen._splitter.handleWidth() == scaled_px(1)
    tiles = screen._tiles[:screen._columns]
    assert len(tiles) >= 3
    for left, right in zip(tiles, tiles[1:]):
        gap = right.geometry().left() - left.geometry().right() - 1
        assert gap <= scaled_px(12)
