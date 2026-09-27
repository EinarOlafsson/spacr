"""Make Masks maps segmentation uncertainty and ranks fields by it.

Driven offscreen with a threshold segmenter in place of Cellpose. The button
is an ALPHA feature, registered as ``MakeMasksUncertaintyButton`` in
``spacr.settings.ALPHA_FEATURES``: shown only while Preferences -> "Show
alpha features" is on, and what it saves still reaches a terminal session
while it is hidden.
"""
from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
import pytest
from scipy import ndimage as ndi

pytest.importorskip("PySide6")


def _blind_corner(image):
    labels = ndi.label(np.asarray(image) > 1000)[0]
    labels[:16, :16] = 0
    return labels


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.make_masks import MakeMasksScreen

    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: False)
    folder = tmp_path / "field"
    (folder / "masks").mkdir(parents=True)
    for name, corner in (("calm", False), ("doubtful", True)):
        image = np.zeros((48, 48), np.uint16)
        image[30:40, 30:40] = 5000
        if corner:
            image[4:12, 4:12] = 5000
        imageio.imwrite(folder / f"{name}.tif", image)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget._open_folder(str(folder))
    return widget


def test_the_button_is_hidden_unless_alpha_features_are_shown(
        qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.make_masks import MakeMasksScreen

    for shown in (False, True):
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: s)
        widget = MakeMasksScreen()
        qtbot.addWidget(widget)
        button = widget._btn_uncertainty
        assert button.objectName() == "MakeMasksUncertaintyButton"
        assert button.isHidden() is not shown
        assert set(widget._uncertainty_actions) == {"map", "rank"}
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: not s)
        preferences._apply_alpha_widgets(widget)
        assert button.isHidden() is shown


def test_ranking_while_hidden_reorders_and_reaches_the_terminal_queue(screen):
    from spacr.curation_queue import build_queue

    assert screen._btn_uncertainty.isHidden()
    assert screen._image_files == ["calm.tif", "doubtful.tif"]
    assert screen._on_rank_uncertainty(segment=_blind_corner,
                                       threaded=False)
    assert screen._image_files == ["doubtful.tif", "calm.tif"]
    assert screen._current_index == 0
    queue = build_queue(screen._folder, order="uncertain",
                        announce=lambda _text: None)
    assert [item.stem for item in queue.items] == ["doubtful", "calm"]
    assert queue.effective_order == "uncertain"


def test_the_map_opens_on_its_own_tab(screen):
    screen._current_index = 1
    screen._load_current()
    tabs = screen._view_tabs.count()
    assert screen._on_map_uncertainty(segment=_blind_corner, threaded=False)
    assert screen._view_tabs.count() == tabs + 1
    assert screen._view_tabs.currentWidget() is screen._uncertainty_pane
    assert screen._uncertainty_pane.has_image()
    assert "uncertainty" in screen._status_label.text().lower()
    screen._on_map_uncertainty(segment=_blind_corner, threaded=False)
    assert screen._view_tabs.count() == tabs + 1


def test_ranking_is_refused_while_blind(screen):
    screen._blind = {"codes": {}}
    assert not screen._on_rank_uncertainty(segment=_blind_corner,
                                           threaded=False)
    assert "Unblind" in screen._status_label.text()
