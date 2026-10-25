"""Upright compartment geometry, persistent selections and linked hover legends."""
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QLabel

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
    qtbot.waitUntil(lambda: screen._columns > 0)
    assert screen._splitter.handleWidth() == scaled_px(1)
    tiles = screen._tiles[:screen._columns]
    assert len(tiles) >= 3
    for left, right in zip(tiles, tiles[1:]):
        gap = right.geometry().left() - left.geometry().right() - 1
        assert gap <= scaled_px(12)


@pytest.mark.parametrize('key', ['toxoplasma', 'plasmodium', 'candida'])
def test_hover_and_selection_keep_model_text_and_following_sections_fixed(key, qtbot, qt_theme_applied):
    screen = OrganismScreen(key)
    qtbot.addWidget(screen)
    screen.resize(1366, 768)
    screen.show()
    qtbot.wait(30)
    diagram = screen._diagram
    following = screen.findChildren(QLabel, 'OrganismSectionTitle')[0]

    def geometry():
        return [(widget.mapTo(screen._intro, QPoint()), widget.size()) for widget in
                (diagram.artwork, diagram.selector, diagram.caption, following)]

    for width in (360, 640, 900):
        screen._splitter.setSizes([width, 1366-width])
        qtbot.waitUntil(lambda:
            diagram.caption.geometry().top() > diagram._model_row.geometry().bottom()
            and diagram.height() > diagram.caption.geometry().bottom()
            and following.mapTo(screen._intro, QPoint()).y()
                > diagram.mapTo(screen._intro, diagram.rect().bottomLeft()).y())
        baseline = geometry()
        legend_right = diagram.selector.mapTo(diagram, diagram.selector.rect().topRight()).x()
        assert legend_right < diagram.artwork.mapTo(diagram, QPoint()).x()
        for code in ('', *dict.fromkeys(diagram.labels.values()), ''):
            diagram.artwork.hovered.emit(code)
            qtbot.wait(1)
            assert geometry() == baseline
            assert diagram.caption.heightForWidth(diagram.caption.width()) <= diagram.caption.height()
        diagram.selector.item(0).setCheckState(Qt.Checked)
        qtbot.wait(1)
        assert geometry() == baseline
        diagram.clear_button.click()
        qtbot.wait(1)
        assert geometry() == baseline


def test_model_panel_uses_home_tile_transparency(diagram):
    from spacr.qt.preferences import get_pane_opacity, resolve_effective_theme
    from spacr.qt.theme import panel_alpha

    color = diagram.artwork.panel_color()
    assert color.getRgb()[:3] == (0, 0, 0)
    assert color.alpha() == round(255 * panel_alpha(resolve_effective_theme(), 'tile', get_pane_opacity()))


@pytest.mark.parametrize("gui_scale", [0.75, 0.9, 1.25])
def test_caption_measurement_is_applied_once_at_each_gui_scale(
        qtbot, qt_theme_applied, gui_scale):
    from spacr.qt import gui_scale as scaling, preferences

    settings = preferences._settings()
    keys = ("prefs/font_scale", "prefs/gui_scale")
    previous = {key: settings.value(key) for key in keys
                if settings.contains(key)}
    scaling.install_scaling_layer()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scaling, "_SCALE", gui_scale)
        try:
            preferences.set_font_scale(1.0)
            preferences.set_gui_scale(gui_scale)
            root = Path(__file__).resolve().parents[2]
            diagram = OrganismDiagram(
                "toxoplasma", root / "spacr/resources/images/organism_apicomplexa.svg")
            qtbot.addWidget(diagram)
            diagram.resize(360, 850)
            diagram.show()
            qtbot.waitUntil(lambda: diagram._layout_width == diagram.width())
            measured_row, measured = diagram._reserved_heights(diagram.width())
            assert diagram._model_row.height() == measured_row
            assert diagram.caption.height() == measured
            for code in ("", *dict.fromkeys(diagram.labels.values()), ""):
                diagram.artwork.hovered.emit(code)
                assert diagram.caption.height() == measured
                assert diagram.caption.heightForWidth(diagram.caption.width()) <= measured
        finally:
            for key in keys:
                if key in previous:
                    settings.setValue(key, previous[key])
                else:
                    settings.remove(key)


def test_a_width_seen_before_is_not_measured_again(diagram, monkeypatch):
    """Settling back and forth between widths measures each width once.

    Showing an organism screen took its splitter through widths 136, 367
    and 372 about thirty times; with only the last width cached, every
    alternation re-measured every compartment description, about 0.75 s of
    one open on an Intel Mac. Counted, not timed.
    """
    measured = []
    real = QLabel.heightForWidth

    def counted(label, width):
        measured.append(width)
        return real(label, width)

    diagram._reserved_cache = None
    first = {width: diagram._measure_reserved_heights(width) for width in (136, 367, 372)}
    monkeypatch.setattr(QLabel, "heightForWidth", counted)
    for _ in range(10):
        for width in (136, 367, 136, 372):
            assert diagram._measure_reserved_heights(width) == first[width]
    assert measured == []

    QApplication.sendEvent(diagram, QEvent(QEvent.FontChange))
    measured.clear()
    diagram._measure_reserved_heights(136)
    assert measured, "a font change must re-measure"
