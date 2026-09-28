"""Cloud sources on Make Masks and Measure are an alpha feature.

Everything built from the future-features list stays hidden until
Preferences -> Show alpha features is on. For cloud sources that is the
eight cloud settings on the Make Masks form, the five on the Measure form,
and the cloud-storage button in each src field. A value set while they are
hidden still reaches the run. The browser dialog lists a store on a worker
thread; here the store is fsspec's in-memory filesystem, never a network.
"""
from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
fsspec = pytest.importorskip("fsspec")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, QSettings                     # noqa: E402

MASK_SETTINGS = ("cloud_anonymous", "cloud_profile", "cloud_endpoint",
                 "cloud_cache", "cloud_wells", "cloud_fields", "cloud_level",
                 "cloud_results")
MEASURE_SETTINGS = ("cloud_anonymous", "cloud_profile", "cloud_endpoint",
                    "cloud_cache", "cloud_results")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_every_part_of_cloud_access_is_registered_under_its_item():
    from spacr.settings import ALPHA_FEATURES, _is_alpha

    entry = ALPHA_FEATURES[550]
    assert set(entry["settings"]) == set(MASK_SETTINGS)
    assert entry["widgets"] == ("CloudSourceBrowse",)
    for key in MASK_SETTINGS:
        assert _is_alpha("settings", key)
    assert _is_alpha("widgets", "CloudSourceBrowse")


def test_mask_cloud_controls_have_one_dedicated_category():
    """Cloud controls have their own section without moving the image source."""
    from spacr.settings import categories
    from spacr.qt.screens.settings_model import categories_for_app

    sections = categories_for_app("mask", categories)
    assert tuple(sections["Cloud"]) == MASK_SETTINGS
    assert "src" in sections["Input & Metadata"]
    for key in MASK_SETTINGS:
        assert [title for title, keys in sections.items() if key in keys] == [
            "Cloud"]
    assert "Cloud" not in categories_for_app("measure", categories)


@pytest.mark.parametrize("language", ["sv", "de", "es", "zh_CN", "pt",
                                     "hi", "ko", "is", "fr"])
def test_cloud_heading_and_help_are_translated(language):
    """Render a translated heading and curated help for each shipped locale.

    :param language: supported non-English interface language.
    """
    from spacr.qt.i18n import tr
    from spacr.qt.screens.settings_model import (
        category_tooltip, category_tooltip_is_curated,
    )

    assert tr("Cloud", language=language) != "Cloud"
    assert category_tooltip_is_curated("mask", "Cloud")
    assert category_tooltip("mask", "Cloud", language) != category_tooltip(
        "mask", "Cloud", "en")


def _browse_action(screen):
    """The cloud-storage button on the screen's src field."""
    found = screen.findChildren(QObject, "CloudSourceBrowse")
    assert len(found) == 1
    return found[0]


@pytest.mark.parametrize("app_key, keys", [("mask", MASK_SETTINGS),
                                           ("measure", MEASURE_SETTINGS)])
def test_cloud_settings_and_button_follow_the_switch(qtbot, prefs, app_key,
                                                     keys):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen(app_key)
    try:
        for key in keys:
            screen._open_the_heading_of(key)
        screen._open_the_heading_of("src")
        screen._refresh_alpha_visibility()

        assert not any(screen.setting_row_is_visible(key) for key in keys)
        assert not _browse_action(screen).isVisible()
        model = screen._settings_model
        values = {
            "cloud_anonymous": True,
            "cloud_profile": "lab-readonly",
            "cloud_endpoint": "https://minio.lab",
            "cloud_cache": "/tmp/cloud-cache",
            "cloud_wells": "A1, B03",
            "cloud_fields": 2,
            "cloud_level": 1,
            "cloud_results": "s3://lab-results/run",
        }
        expected = {key: values[key] for key in keys}
        for key, value in expected.items():
            assert model.set_value_for_key(key, value)
        collected = model.collect()
        assert {key: collected[key] for key in keys} == expected

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(key) for key in keys)
        assert _browse_action(screen).isVisible()

        prefs._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key) for key in keys)
        assert not _browse_action(screen).isVisible()
        collected = model.collect()
        assert {key: collected[key] for key in keys} == expected
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()


def test_the_browser_lists_a_store_off_the_gui_thread(qtbot):
    from spacr.qt.screens.settings_model import _CloudBrowserDialog

    fs = fsspec.filesystem("memory")
    bucket = f"/bucket-{uuid.uuid4().hex[:8]}"
    fs.pipe(f"{bucket}/screens/a.csv", b"x\n1\n")
    fs.pipe(f"{bucket}/screens/plate/.keep", b"")
    try:
        dialog = _CloudBrowserDialog(
            f"memory://{bucket.lstrip('/')}/screens", anonymous=True,
            endpoint="https://minio.lab")
        qtbot.addWidget(dialog)
        qtbot.waitUntil(lambda: dialog.entries.count() == 2, timeout=5000)
        labels = [dialog.entries.item(i).text()
                  for i in range(dialog.entries.count())]
        assert labels[0] == "plate/"
        assert labels[1].startswith("a.csv")
        assert dialog.options() == {"cloud_anonymous": True,
                                    "cloud_endpoint": "https://minio.lab"}

        dialog._enter(dialog.entries.item(0))
        assert dialog.location().endswith("/screens/plate")
        dialog.go_up()
        assert dialog.location().endswith("/screens")

        dialog.address.setText("gs-nothing://bucket")
        dialog.open_address()
        qtbot.waitUntil(lambda: "ValueError" in dialog.details.toPlainText(),
                        timeout=5000)
    finally:
        fs.rm(bucket, recursive=True)
