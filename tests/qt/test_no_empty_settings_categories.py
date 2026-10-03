"""No module shows a settings heading with nothing under it.

The maintainer, 2026-10-03: "In mask generation there are some setting
categories related to cell that now have no settings in them, delete any
empty settings classes, this goes for all settings categories, check all
modules".

Every module is built with Preferences -> Show alpha features off and on,
every closed heading is opened, and each heading still on the form must
hold at least one visible row or a visible sub-heading. Mask's "Cell
Segmentation" was the case found: the per-object grid holds all its rows.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402
from PySide6.QtWidgets import QFormLayout                         # noqa: E402

from spacr.qt.app import APPS                                     # noqa: E402

APP_KEYS = tuple(dict.fromkeys(row[0] for row in APPS))


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def _visible_rows(section) -> int:
    form = getattr(section, "_form", None)
    if not isinstance(form, QFormLayout):
        return 0
    count = 0
    for row in range(form.rowCount()):
        item = form.itemAt(row, QFormLayout.FieldRole)
        widget = item.widget() if item is not None else None
        if widget is not None and form.isRowVisible(row) \
                and not widget.isHidden():
            count += 1
    return count


def _empty_headings(screen) -> list:
    from spacr.qt.widgets.section import Section, _sections_below

    screen._open_every_waiting_heading()
    empty = []
    for section in screen.rendered_settings_sections():
        if not section.isVisibleTo(screen):
            continue
        below = [child for child in _sections_below(section)
                 if isinstance(child, Section) and child.isVisibleTo(screen)]
        if not below and _visible_rows(section) == 0:
            empty.append(str(section.title()))
    return empty


def _screen(qtbot, app_key):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    return screen


@pytest.mark.parametrize("alpha", [False, True], ids=["alpha_off", "alpha_on"])
@pytest.mark.parametrize("app_key", APP_KEYS)
def test_no_heading_renders_empty(qtbot, prefs, app_key, alpha):
    prefs.set_show_alpha(alpha)
    screen = _screen(qtbot, app_key)
    assert _empty_headings(screen) == []


@pytest.mark.parametrize("alpha", [False, True], ids=["alpha_off", "alpha_on"])
def test_mask_object_headings_stay_off_when_their_channels_are_set(
        qtbot, prefs, alpha):
    prefs.set_show_alpha(alpha)
    screen = _screen(qtbot, "mask")
    model = screen._settings_model
    for key in ("nucleus_channel", "pathogen_channel"):
        model.set_value_for_key(key, 1)
    model.refresh_object_visibility()
    titles = [str(s.title()).upper() for s in
              screen.rendered_settings_sections() if s.isVisibleTo(screen)]
    assert "CELL SEGMENTATION" not in titles
    assert _empty_headings(screen) == []


def test_measure_offers_no_image_enhancement_heading():
    from spacr import settings as S
    from spacr.qt.screens.settings_model import (_APP_CATEGORY_SPECS,
                                                 _category_parents)

    assert "Image Enhancement" not in {
        title for title, _tokens in _APP_CATEGORY_SPECS["measure"]}
    assert "Image Enhancement" not in _category_parents("measure")
    assert "Image Enhancement" in S.categories
