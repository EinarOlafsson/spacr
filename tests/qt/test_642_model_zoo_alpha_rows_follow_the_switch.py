"""Model Zoo's alpha rows follow Show alpha features, wherever the page lives.

Item 642: with Show alpha features on, the alpha models (StarDist, InstanSeg,
Omnipose, micro-SAM, the colony detector, ...) were missing from the Model
Zoo table. The table decides its alpha rows when it is filled; the Model Zoo
folded into Make Masks is not a screen the app walks when Preferences
closes, so it kept the answer it had when it was built.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


_MUST_SHOW = {"stardist_v1", "instanseg_v1", "omnipose_v1", "microsam_v1",
              "colony_yolo11n_makrai_v1"}


def _alpha_keys():
    from spacr import model_zoo as zoo
    from spacr.settings import _is_alpha

    keys = {e.key for e in zoo.catalogue(remote=True, block=False)
            if _is_alpha("models", e.key)}
    assert _MUST_SHOW <= keys
    return keys


def _visible_alpha_rows(screen, alpha):
    from spacr.qt.screens.model_zoo import _stem_version

    shown = set()
    for group, (stem, pairs) in enumerate(screen._groups):
        row = screen._row_of_group(group)
        if row is None or screen._table.isRowHidden(row):
            continue
        for _label, entry in pairs:
            if entry.key in alpha or _stem_version(entry)[0] in alpha:
                shown.add(entry.key)
    return shown


def test_the_table_lists_every_alpha_model_only_with_the_switch_on(
        qtbot, prefs):
    from spacr import model_zoo as zoo
    from spacr.qt.screens.model_zoo import ModelZooScreen

    alpha = _alpha_keys()
    screen = ModelZooScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.scan(include_catalogue=True)
    assert _visible_alpha_rows(screen, alpha) == set()

    prefs._set_show_alpha_features(True)
    screen._refresh_alpha_visibility()
    listed = {e.key for e in screen.entries()
              if e.key in alpha
              and zoo.source_of(e) in screen.sources.enabled()}
    assert _MUST_SHOW <= listed
    assert _visible_alpha_rows(screen, alpha) == listed

    prefs._set_show_alpha_features(False)
    screen._refresh_alpha_visibility()
    assert _visible_alpha_rows(screen, alpha) == set()


def test_make_masks_passes_the_switch_to_its_folded_model_zoo(qtbot, prefs):
    from spacr.qt.screens.make_masks import MakeMasksScreen

    alpha = _alpha_keys()
    masks = MakeMasksScreen()
    qtbot.addWidget(masks)
    try:
        zoo_page = masks.folded_screen("model_zoo")
        zoo_page._threaded = False
        zoo_page.scan(include_catalogue=True)
        assert _visible_alpha_rows(zoo_page, alpha) == set()

        prefs._set_show_alpha_features(True)
        masks._refresh_alpha_visibility()
        assert _MUST_SHOW <= _visible_alpha_rows(zoo_page, alpha)

        prefs._set_show_alpha_features(False)
        masks._refresh_alpha_visibility()
        assert _visible_alpha_rows(zoo_page, alpha) == set()
    finally:
        masks.close()
