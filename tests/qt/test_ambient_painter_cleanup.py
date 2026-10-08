"""Actual Qt painters finish even when a backdrop paint step raises."""
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QImage, QPainter, QPaintEvent

from spacr.qt.widgets import ambient


@pytest.mark.parametrize('stage', ['base', 'paint', 'latest', 'blit', 'render_size'])
def test_failed_backdrop_paint_releases_its_device(qtbot, monkeypatch, stage):
    widget = ambient.AmbientWidget(theme='blobs', seed=7)
    qtbot.addWidget(widget)
    widget.resize(80, 60)
    widget.stop()
    device = QImage(80, 60, QImage.Format_ARGB32_Premultiplied)
    painters = []

    def painter(_widget):
        value = QPainter(device)
        assert value.isActive()
        painters.append(value)
        return value

    def fail(*_args):
        raise RuntimeError('controlled paint failure')

    monkeypatch.setattr(ambient, 'QPainter', painter)
    if stage == 'base':
        monkeypatch.setattr(widget, '_paint_base', fail)
    elif stage == 'paint':
        monkeypatch.setattr(widget.engine, 'paint', fail)
    else:
        widget._producer_box[0] = SimpleNamespace(
            size=None, latest=(fail if stage == 'latest' else lambda: device),
            stop=lambda: None)
        if stage == 'blit':
            monkeypatch.setattr(widget.engine, 'blit', fail)
        elif stage == 'render_size':
            monkeypatch.setattr(widget, '_art_render_size', fail)
    try:
        with pytest.raises(RuntimeError, match='controlled paint failure'):
            widget._paint_ambient(QPaintEvent(widget.rect()))
        assert painters and all(not value.isActive() for value in painters)
    finally:
        for value in painters:
            if value.isActive():
                value.end()
        widget._producer_box[0] = None


def test_failed_spinn_tile_paint_releases_its_image(monkeypatch, qapp):
    engine = ambient.make_engine('data_art_tissue_facets', 'spacr', '#101010', seed=7)
    device = QImage(80, 60, QImage.Format_ARGB32_Premultiplied)
    outer = QPainter(device)
    painters = []
    tiles = []

    class FailingPainter(QPainter):
        def __init__(self, tile):
            tiles.append(tile)
            super().__init__(tile)
            assert self.isActive()
            painters.append(self)

        def drawPolygon(self, *_args):
            raise RuntimeError('controlled tile failure')

    monkeypatch.setattr(ambient, 'QPainter', FailingPainter)
    try:
        with pytest.raises(RuntimeError, match='controlled tile failure'):
            engine._paint_tissue_facets(outer, 80, 60)
        assert painters and all(not value.isActive() for value in painters)
    finally:
        for value in painters:
            if value.isActive():
                value.end()
        outer.end()
