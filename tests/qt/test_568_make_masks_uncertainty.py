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


def test_cellpose_passes_add_near_misses_and_flow_errors(screen, monkeypatch):
    from spacr.qt.screens import make_masks as mm

    seen = []

    def fake_detect(image, model, *, diameter=0, normalize=True,
                    flow_threshold=0.4, cellprob_threshold=0.0, min_size=0,
                    channel_axis=None):
        seen.append(True)
        labels = ndi.label(np.asarray(image) > 1000)[0].astype(np.int32)
        logit = np.where(labels > 0, 5.0, -8.0).astype(np.float32)
        logit[(np.asarray(image) > 100) & (labels == 0)] = -1.0
        vectors = np.zeros((2,) + labels.shape, np.float32)
        return labels, logit, None, vectors

    monkeypatch.setattr(mm, "_cellpose_detect_with_vectors", fake_detect)
    image = np.zeros((48, 48), np.uint16)
    image[20:30, 20:30] = 5000
    image[40:46, 2:10] = 500
    request = dict(detect=screen._uncertainty_detect_request(), kind="field",
                   image=image)
    model = request["detect"]["model"]
    result = mm._uncertainty_snapshot(request, {model: object()})
    assert seen == [True] * 4
    assert result["missed"] > 0.0 and result["area"] == 0.0
    assert result["field"] == result["missed"]
    assert result["map"][42, 5] > 0.5
    assert set(result["flow_errors"]) == {1}
    assert result["objects"][1] > 0.5


def test_the_object_operations_setting_is_the_same_uncertainty(
        qtbot, qt_theme_applied, monkeypatch):
    """Uncertainty also sits in Object operations, right after Swap.

    It opens the same Map / Rank menu, is disabled with the toolbar button
    while a run is under way, and is hidden by the same alpha gate.
    """
    from PySide6.QtWidgets import QPushButton

    from spacr.qt import preferences
    from spacr.qt.screens.make_masks import MakeMasksScreen

    for shown in (False, True):
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: s)
        widget = MakeMasksScreen()
        qtbot.addWidget(widget)
        setting = widget._btn_uncertainty_setting
        assert setting.objectName() == "MakeMasksUncertaintySetting"
        assert setting.isHidden() is not shown
        assert setting.menu() is widget._btn_uncertainty.menu()
        column = setting.parentWidget().layout()
        texts = [column.itemAt(i).widget().text()
                 for i in range(column.count())
                 if isinstance(column.itemAt(i).widget(), QPushButton)]
        index = texts.index("Swap object and background")
        assert texts[index + 1] == "Uncertainty…"
        widget._set_uncertainty_enabled(False)
        assert not setting.isEnabled()
        assert not widget._btn_uncertainty.isEnabled()
        widget._set_uncertainty_enabled(True)
        assert setting.isEnabled() and widget._btn_uncertainty.isEnabled()
