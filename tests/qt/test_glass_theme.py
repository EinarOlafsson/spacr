"""The Glass preference and its translucent module material."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_qsettings(qt_theme_applied, tmp_path):
    """Redirect QSettings into ``tmp_path`` before clearing it.

    NativeFormat, not just Ini: ``QSettings("spacr", "qt")`` is a NativeFormat
    object no matter what ``setDefaultFormat`` says, so an Ini-only redirect
    left the ``.clear()`` calls below deleting the real user preferences.
    """
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.IniFormat)
    for fmt in (QSettings.NativeFormat, QSettings.IniFormat):
        QSettings.setPath(fmt, QSettings.UserScope, str(tmp_path))
    QSettings("spacr", "qt").clear()
    yield
    QSettings("spacr", "qt").clear()


def _rule(stylesheet, selector):
    start = stylesheet.index(selector)
    end = stylesheet.index("}", start)
    return stylesheet[start:end]


def test_glass_is_a_persisted_theme():
    from spacr.qt import preferences
    from spacr.qt import theme

    preferences.set_theme("glass")
    assert preferences.get_theme() == "glass"
    assert preferences.resolve_effective_theme() == "glass"
    assert "glass" in theme.THEMES
    assert "glass" in theme.IMAGE_THEMES


def test_preferences_dialog_offers_glass(qtbot):
    from PySide6.QtWidgets import QComboBox
    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    combos = dialog.findChildren(QComboBox)
    values = {
        combo.itemData(index)
        for combo in combos
        for index in range(combo.count())
    }
    assert "glass" in values


def test_glass_scrims_are_translucent_and_accessible():
    from spacr.qt import theme

    assert theme.scrim_under("glass") == theme.GLASS_BACKDROP_UNDER
    assert theme.scrim_failures("glass") == []
    assert theme.contrast_failures("glass") == []
    for role in theme.SCRIM_ROLES:
        assert 0.0 < theme.scrim_alpha("glass", role) < 0.88


def test_glass_material_is_neutral_not_a_blue_overlay():
    from spacr.qt import theme

    palette = theme.palette_for("glass")
    for role in ("surface", "surface_alt", "surface_hi"):
        channels = theme._channels(palette[role])
        assert max(channels) - min(channels) <= 12
    # Colour is selective: the action accent remains visibly blue.
    accent = theme._channels(palette["accent"])
    assert accent[2] - accent[0] > 50


def test_full_page_opacity_keeps_glass_translucent_by_design():
    from spacr.qt import theme

    for role in theme.SCRIM_ROLES:
        designed = theme.scrim_alpha("glass", role)
        assert theme.panel_alpha("glass", role, 1.0) == designed
        assert theme.panel_alpha("glass", role, 0.5) <= designed
    assert theme.pane_alpha("glass", 1.0) == \
        theme.scrim_alpha("glass", "surface")


def test_glass_material_has_highlight_body_and_depth_layers():
    from spacr.qt import theme

    material = theme.glass_material("#303238", 0.28)
    assert material.startswith("qlineargradient(")
    assert material.count("stop:") == 4
    assert material.count("rgba(") == 4


def test_glass_styles_every_module_box_with_rgba_material():
    from spacr.qt import theme

    qss = theme.stylesheet("glass")
    assert "qlineargradient" in qss
    for selector in (
        "QFrame#Card {",
        "QFrame#ConsoleBox {",
        "QFrame#SectionCard {",
        "QLineEdit, QSpinBox",
    ):
        rule = _rule(qss, selector)
        assert "rgba(" in rule, f"{selector} remained opaque"


def test_glass_adds_neutral_light_field_specular_rims_and_rounding():
    from spacr.qt import theme

    qss = theme.stylesheet("glass", surface_opacity=1.0)
    window = _rule(qss, "QMainWindow, QDialog {")
    assert "qradialgradient" in window
    assert "#454950" in window

    start = qss.index("QFrame#Card, QFrame#ConsoleBox {")
    material = qss[start:qss.index("}", start)]
    assert "qlineargradient" in material
    assert "rgba(255, 255, 255, 0.270)" in material
    assert "border-radius: 14px" in material


def test_glass_home_pane_paints_no_box_behind_the_tiles():
    """The container behind the tiles is gone, not dialled.

    This settled after three passes: a surface at the effective alpha, then
    transparent, then briefly painted at the preference again on the reading
    that opacity should "apply to the containers the tiles are in". The final
    instruction is the clearest — remove the black boxes behind the tiles and
    make the TILES subject to opacity instead — so the container paints
    nothing and the dialling lives on the tile fill, where it can be seen.

    The rim stays: it is what the selected tab joins onto, and without it the
    tab strip floats with nothing under it.
    """
    from spacr.qt import theme
    from spacr.qt.widgets.home import _tab_qss

    palette = theme.palette_for("glass")
    for alpha in (0.0, theme.pane_alpha("glass", 1.0), 1.0):
        pane = _rule(_tab_qss(palette, alpha, glass=True),
                     "QTabWidget#HomeTabs::pane {")
        assert "background: transparent" in pane, \
            "the pane painted a fill at alpha %r" % (alpha,)
        assert "qlineargradient" not in pane
        assert "rgba(255, 255, 255, 0.270)" in pane
        assert "border-radius: 14px" in pane


def test_glass_preference_explains_material_strength(qtbot):
    from PySide6.QtWidgets import QLabel
    from spacr.qt import preferences

    preferences.set_theme("glass")
    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    texts = [label.text() for label in dialog.findChildren(QLabel)]
    assert any("material strength" in text for text in texts)
    assert any("stays translucent" in text for text in texts)


def test_glass_popups_remain_opaque():
    from spacr.qt import theme

    assert theme.scrim_alpha("glass", "elevated") == 1.0
    popup = _rule(theme.stylesheet("glass"), "QMenu {")
    assert "rgba(" not in popup
