"""Item 691: MitoNet in Make Masks' Method box, behind the alpha gate.

No network and no weights: the segmenter is checked with
``spacr.model_zoo.segment_mitonet`` replaced.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QSettings

from spacr import model_zoo as mz
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import MakeMasksScreen


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences through a throwaway INI file, alpha features off."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


@pytest.fixture
def screen(qtbot, qt_theme_applied, prefs):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    return widget


def _row_hidden(screen) -> bool:
    index = screen._mag_mode.findData("mitonet")
    assert index >= 0
    return screen._mag_mode.view().isRowHidden(index)


def test_mitonet_is_registered_as_alpha():
    from spacr.settings import ALPHA_FEATURES

    entry = ALPHA_FEATURES[691]
    assert mz.MITONET_KEY in entry["models"]
    assert "MakeMasksMitoNetSettings" in entry["widgets"]
    assert "ModelZooSourceThirdParty" in entry["widgets"]


def test_the_method_row_follows_the_alpha_switch(screen, prefs):
    assert prefs._get_show_alpha_features() is False
    assert _row_hidden(screen)
    card = screen.findChild(mm._MethodGroup, "MakeMasksMitoNetSettings")
    assert card is not None
    prefs._set_show_alpha_features(True)
    screen._refresh_alpha_visibility()
    assert not _row_hidden(screen)
    index = screen._mag_mode.findData("mitonet")
    assert screen._mag_mode.itemText(index) == "MitoNet (TEM mitochondria)"
    screen._mag_mode.setCurrentIndex(index)
    assert screen._mode_family("mitonet") == "mitonet"
    assert not card.isHidden()
    assert "Zenodo" in screen._method_note.text()
    prefs._set_show_alpha_features(False)
    screen._refresh_alpha_visibility()
    assert _row_hidden(screen)


def test_the_zoo_heading_is_hidden_with_the_gate_shut(qtbot, prefs):
    from spacr.qt.widgets.model_zoo_picker import SourceStrip

    strip = SourceStrip()
    qtbot.addWidget(strip)
    assert strip.heading("Third-party").isHidden()
    prefs._set_show_alpha_features(True)
    shown = SourceStrip()
    qtbot.addWidget(shown)
    assert not shown.heading("Third-party").isHidden()
    assert shown.heading("Third-party").objectName() == \
        "ModelZooSourceThirdParty"


def test_the_segmenter_hands_the_raw_crop_and_minimum_area_on(monkeypatch):
    seen = {}

    def fake(image, **kwargs):
        seen["image"] = np.asarray(image)
        seen.update(kwargs)
        kwargs["on_tile"](1, 1)
        return np.ones(np.asarray(image).shape, np.int32)

    monkeypatch.setattr(mz, "segment_mitonet", fake)
    crop = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    ticket = mm._RunTicket()
    request = mm._MagnifierRequest(
        key=(), crop=crop, box=(0, 0, 8, 8), shape=(8, 8), mode="mitonet",
        sensitivity=1.0, bright=True, min_area=25, model_name="cpsam",
        diameter=30, colour=(0, 0, 0), invert=True, ticket=ticket)
    labels, used, note = mm._segment_region(request)
    assert used == "mitonet" and note == ""
    assert labels.shape == (8, 8)
    np.testing.assert_allclose(seen["image"], 1.0 - crop)
    assert seen["min_size"] == 25
    assert ticket.total == 1 and ticket.done == 1
    assert mm._magnifier_mode_label("mitonet") == "MitoNet (TEM mitochondria)"
