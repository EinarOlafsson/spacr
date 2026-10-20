"""Application-wide event filters never hand a foreign wrapper to the base class.

An application-wide filter sees every event in the process. PySide can pass
such a filter a wrapper of the wrong type (a ``QSpacerItem``) for an object
whose C++ side is being torn down. The ambient backdrop used to forward that
object to ``QObject.eventFilter``, which raised ``TypeError`` and was followed
by a segmentation fault. Both application-wide filters now answer ``False``
for anything they do not consume.
"""

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QSettings
from PySide6.QtWidgets import QSpacerItem, QWidget

from spacr.qt import preferences as prefs
from spacr.qt.screens import make_masks as mm
from spacr.qt.widgets import ambient


@pytest.fixture(autouse=True)
def quiet_backdrop(tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / "filters.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    monkeypatch.setattr(ambient.AmbientWidget, "_start_producer", lambda self: None)


def test_backdrop_filter_passes_a_spacer_item_through(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(200, 120)
    field = ambient.AmbientWidget(
        host, theme="data_art_impulse_lens", gravity_radius=0,
        popup_wave_frequency=0, blink_percent=0, density=1, seed=3)
    field._timer.stop()
    spacer = QSpacerItem(10, 10)
    for kind in (QEvent.Paint, QEvent.Show, QEvent.Hide, QEvent.Move):
        assert field.eventFilter(spacer, QEvent(kind)) is False


def test_magnifier_filter_passes_a_spacer_item_through(qtbot):
    canvas = mm._MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    canvas.set_image_and_mask(np.zeros((20, 20), dtype=np.uint16),
                              np.zeros((20, 20), dtype=np.uint16))
    magnifier = mm._LiveMagnifier(canvas, canvas)
    try:
        spacer = QSpacerItem(10, 10)
        assert magnifier.eventFilter(spacer, QEvent(QEvent.Paint)) is False
    finally:
        magnifier.close()
        canvas.close_enhancer()
