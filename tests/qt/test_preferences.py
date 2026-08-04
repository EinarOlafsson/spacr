"""Tests for spacr.qt.preferences — theme + font scale + CB mode."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, qt_theme_applied, tmp_path):
    """Route QSettings into a temp store so tests don't touch real prefs.

    NativeFormat is the one that has to move: ``QSettings("spacr", "qt")`` —
    what ``preferences._settings()`` builds — is a NativeFormat object and
    ignores ``setDefaultFormat``/``setPath(IniFormat, ...)`` entirely. Setting
    only the Ini path left ``.clear()`` below pointed at the developer's real
    ``~/.config/spacr/qt.conf``.

    Every one of the four calls below is **process-global and permanent**, and
    for a long time none of them was put back. The organisation name, the
    application name and the default format therefore kept this module's
    values for the rest of the session, and
    ``tests/qt/test_all_module_smoke.py`` — which measures real pixels through
    a themed ``QApplication`` — failed its setting-row background assertion on
    every module whenever this file happened to run first. A shard ordering
    decided whether the suite was green.

    ``setPath`` is the one leak the root ``conftest`` already repairs (its
    ``_isolated_qsettings_store`` re-points every format/scope pair on
    teardown, and it is torn down after this one). The other three are
    snapshotted and restored here.

    The **QApplication's own appearance** is restored for the same reason and
    it is the half that actually moved the pixels:
    ``test_apply_preferences_to_app_takes_the_theme_and_the_font_scale`` puts
    the app into the *light* palette at 125 % and leaves it there. The smoke
    test downstream re-applies a dark stylesheet before it measures — but a
    stylesheet is not a ``QPalette``, so the light palette survived and one
    corner of the setting-label wrapper came back ``#161719`` instead of the
    card's ``#0d0e10``.
    """
    from PySide6.QtCore import QCoreApplication, QSettings
    from spacr.qt.first_run import mark_tour_seen

    was_org = QCoreApplication.organizationName()
    was_app = QCoreApplication.applicationName()
    was_format = QSettings.defaultFormat()
    was_palette = qt_theme_applied.palette()
    was_stylesheet = qt_theme_applied.styleSheet()
    was_font = qt_theme_applied.font()

    QCoreApplication.setOrganizationName("spacr-test")
    QCoreApplication.setApplicationName("qt-prefs-test")
    QSettings.setDefaultFormat(QSettings.IniFormat)
    for fmt in (QSettings.NativeFormat, QSettings.IniFormat):
        QSettings.setPath(fmt, QSettings.UserScope, str(tmp_path))
    QSettings("spacr", "qt").clear()
    # Re-mark first-launch tour seen after the QSettings clear so the
    # autouse conftest fixture keeps its promise. Imported at the top of the
    # fixture rather than guarded: a spacr.qt that cannot supply
    # mark_tour_seen is a product failure, and swallowing it here only moved
    # the report to whichever unrelated test then met the tour dialog.
    mark_tour_seen()
    try:
        yield
    finally:
        QCoreApplication.setOrganizationName(was_org)
        QCoreApplication.setApplicationName(was_app)
        QSettings.setDefaultFormat(was_format)
        qt_theme_applied.setPalette(was_palette)
        qt_theme_applied.setStyleSheet(was_stylesheet)
        qt_theme_applied.setFont(was_font)


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

def test_language_default_is_english(qt_theme_applied):
    from spacr.qt.preferences import get_language

    assert get_language() == "en"


def test_language_roundtrip_and_validation(qt_theme_applied):
    from spacr.qt.i18n import VALID_LANGUAGE_CODES
    from spacr.qt.preferences import get_language, set_language

    for code in VALID_LANGUAGE_CODES:
        set_language(code)
        assert get_language() == code
    with pytest.raises(ValueError, match="unknown language"):
        set_language("klingon")


def test_corrupt_language_falls_back_to_english(qt_theme_applied):
    from PySide6.QtCore import QSettings
    from spacr.qt.preferences import get_language

    QSettings("spacr", "qt").setValue("prefs/language", "garbage")
    assert get_language() == "en"


def test_preferences_dialog_offers_and_saves_every_language(
    qtbot, qt_theme_applied,
):
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox
    from spacr.qt.i18n import LANGUAGES
    from spacr.qt.preferences import PreferencesDialog, get_language

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    combo = dlg.findChild(QComboBox, "LanguagePreference")
    assert combo is not None
    assert combo.count() == len(LANGUAGES)
    codes = [combo.itemData(index) for index in range(combo.count())]
    assert codes == [language.code for language in LANGUAGES]

    combo.setCurrentIndex(codes.index("zh_CN"))
    buttons = dlg.findChild(QDialogButtonBox)
    buttons.button(QDialogButtonBox.Save).click()
    assert get_language() == "zh_CN"


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def test_theme_default_follows_the_system(qt_theme_applied):
    """A desktop app that ignores the OS colour scheme looks broken on a
    light desktop, so the shipped default follows it."""
    from spacr.qt.preferences import get_theme_choice
    assert get_theme_choice() == "system"


def test_theme_roundtrip(qt_theme_applied):
    from spacr.qt.preferences import get_theme, set_theme
    set_theme("light")
    assert get_theme() == "light"
    set_theme("system")
    assert get_theme() == "system"


def test_theme_invalid_raises(qt_theme_applied):
    from spacr.qt.preferences import set_theme
    with pytest.raises(ValueError):
        set_theme("purple")


def test_theme_recovers_from_corrupt_value(qt_theme_applied):
    from spacr.qt.preferences import get_theme
    from PySide6.QtCore import QSettings
    QSettings("spacr", "qt").setValue("prefs/theme", "garbage")
    assert get_theme() == "system"


def test_figure_png_dpi_roundtrip_and_validation(qt_theme_applied):
    from spacr.qt.preferences import get_figure_png_dpi, set_figure_png_dpi

    set_figure_png_dpi(600)
    assert get_figure_png_dpi() == 600
    with pytest.raises(ValueError, match="unknown PNG resolution"):
        set_figure_png_dpi(72)


def test_resolve_effective_theme_dark_and_light(qt_theme_applied):
    from spacr.qt.preferences import (
        resolve_effective_theme, set_theme,
    )
    set_theme("dark")
    assert resolve_effective_theme() == "dark"
    set_theme("light")
    assert resolve_effective_theme() == "light"


def test_resolve_system_returns_valid_choice(qt_theme_applied):
    from spacr.qt.preferences import (
        resolve_effective_theme, set_theme,
    )
    set_theme("system")
    assert resolve_effective_theme() in ("dark", "light")


# ---------------------------------------------------------------------------
# Font scale
# ---------------------------------------------------------------------------

def test_font_scale_default_is_150pct(qt_theme_applied):
    from spacr.qt.preferences import get_font_scale
    assert get_font_scale() == 1.5


def test_font_scale_roundtrip(qt_theme_applied):
    from spacr.qt.preferences import get_font_scale, set_font_scale
    set_font_scale(1.25)
    assert get_font_scale() == pytest.approx(1.25)
    set_font_scale(1.5)
    assert get_font_scale() == pytest.approx(1.5)


def test_font_scale_clamps_out_of_range(qt_theme_applied):
    from spacr.qt.preferences import (
        get_font_scale, set_font_scale,
        FONT_SCALE_MIN, FONT_SCALE_MAX,
    )
    set_font_scale(10.0)
    assert get_font_scale() == FONT_SCALE_MAX
    set_font_scale(0.01)
    assert get_font_scale() == FONT_SCALE_MIN


def test_font_scale_recovers_from_corrupt_value(qt_theme_applied):
    from spacr.qt.preferences import get_font_scale
    from PySide6.QtCore import QSettings
    QSettings("spacr", "qt").setValue("prefs/font_scale", "garbage")
    assert get_font_scale() == 1.5


# ---------------------------------------------------------------------------
# Colour-blind mode
# ---------------------------------------------------------------------------

def test_cb_mode_default_is_off(qt_theme_applied):
    from spacr.qt.preferences import get_color_blind_mode
    assert get_color_blind_mode() == "off"


def test_cb_mode_roundtrip(qt_theme_applied):
    from spacr.qt.preferences import (
        get_color_blind_mode, set_color_blind_mode,
    )
    set_color_blind_mode("deuteranopia")
    assert get_color_blind_mode() == "deuteranopia"


def test_cb_mode_invalid_raises(qt_theme_applied):
    from spacr.qt.preferences import set_color_blind_mode
    with pytest.raises(ValueError):
        set_color_blind_mode("technicolor")


def test_categorical_palette_switches_with_cb_mode(qt_theme_applied):
    from spacr.qt.preferences import (
        color_blind_categorical_palette, set_color_blind_mode,
    )
    set_color_blind_mode("off")
    off = color_blind_categorical_palette()
    set_color_blind_mode("deuteranopia")
    on = color_blind_categorical_palette()
    assert off != on
    # Okabe-Ito starts with blue #0072B2
    assert on[0] == "#0072B2"


def test_continuous_cmap_switches_with_cb_mode(qt_theme_applied):
    from spacr.qt.preferences import (
        color_blind_continuous_cmap, set_color_blind_mode,
    )
    set_color_blind_mode("off")
    assert color_blind_continuous_cmap() == "viridis"
    set_color_blind_mode("protanopia")
    assert color_blind_continuous_cmap() == "cividis"


# ---------------------------------------------------------------------------
# Theme + font-scale integration
# ---------------------------------------------------------------------------

def test_stylesheet_accepts_theme_and_font_scale(qt_theme_applied):
    from spacr.qt.theme import stylesheet
    dark = stylesheet(theme="dark", font_scale=1.0)
    light = stylesheet(theme="light", font_scale=1.0)
    scaled = stylesheet(theme="dark", font_scale=1.5)
    assert dark != light            # palette-driven
    assert dark != scaled           # font-size driven
    assert "font-size" in dark


def test_palette_for_returns_light_for_light(qt_theme_applied):
    """After the theme-invariant CONSTANT_ROLES addition, palette_for
    returns a NEW dict merging the base palette with the constant
    roles, so identity-compare no longer holds. Value-compare the
    theme-specific colours instead."""
    from spacr.qt.theme import palette_for, DARK_PALETTE, LIGHT_PALETTE
    for key in ("bg", "surface", "fg", "accent"):
        assert palette_for("light")[key] == LIGHT_PALETTE[key]
        assert palette_for("dark")[key] == DARK_PALETTE[key]
        assert palette_for("bogus")[key] == DARK_PALETTE[key]     # fallback


def _qss_font_sizes(qss: str):
    """Every ``font-size: Npx`` in a stylesheet, in source order."""
    import re
    return [int(px) for px in re.findall(r"font-size:\s*(\d+)px", qss)]


def test_apply_preferences_to_app_takes_the_theme_and_the_font_scale(
        qt_theme_applied):
    """The QApplication has to actually *wear* the saved preferences.

    Neither half is asserted against a literal. The palette is compared
    to ``palette_for("light")`` and, in the same breath, to the dark
    colour it must no longer be; the font sizes are compared to the
    stylesheet this very call would have produced at 100 %, so an
    ``apply`` that dropped the scale on the floor emits exactly the
    baseline and fails.
    """
    from PySide6.QtGui import QPalette
    from spacr.qt.preferences import (
        apply_preferences_to_app, get_language, get_pane_opacity,
        set_theme, set_font_scale,
    )
    from spacr.qt.theme import apply_qpalette, palette_for, stylesheet

    app = qt_theme_applied
    # Start from a known dark/100 % application, so nothing below can be
    # satisfied by state an earlier test happened to leave behind. Only
    # the palette and the QSS are seeded — `apply_preferences_to_app` is
    # called exactly once, because it also walks every live widget.
    apply_qpalette(app, theme="dark")
    app.setStyleSheet(stylesheet(theme="dark", font_scale=1.0))
    seeded_sizes = _qss_font_sizes(app.styleSheet())

    set_theme("light"); set_font_scale(1.25)
    apply_preferences_to_app()

    light, dark = palette_for("light"), palette_for("dark")
    assert light["bg"] != dark["bg"]        # the swap is observable at all
    assert app.palette().color(QPalette.Window).name() == light["bg"]
    assert app.palette().color(QPalette.WindowText).name() == light["fg"]
    assert app.palette().color(QPalette.Base).name() == light["surface"]
    assert app.palette().color(QPalette.Highlight).name() == light["accent"]

    applied = _qss_font_sizes(app.styleSheet())
    unscaled = _qss_font_sizes(stylesheet(theme="light", font_scale=1.0,
                                          surface_opacity=get_pane_opacity()))
    assert applied and len(applied) == len(unscaled) == len(seeded_sizes)
    assert applied != seeded_sizes          # the QSS was replaced at all
    assert applied != unscaled              # ... and the scale reached it
    assert all(big >= small for small, big in zip(unscaled, applied))
    assert max(applied) > max(unscaled)
    # ... and it is the light stylesheet, not the dark QSS at 125 %.
    assert light["bg"] in app.styleSheet()

    assert app.property("spacrLanguage") == get_language()


def test_preferences_dialog_shows_the_saved_values_and_cancel_saves_nothing(
        qtbot, qt_theme_applied):
    """Cancel is the whole test: every control is moved off the stored
    value first, so a ``reject()`` that wrote anything through would be
    read back below."""
    from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QSlider
    from spacr.qt.preferences import (
        PreferencesDialog, get_color_blind_mode, get_font_scale,
        get_theme_choice, set_color_blind_mode, set_font_scale,
        set_theme_choice, theme_choices,
    )

    set_theme_choice("light")
    set_font_scale(1.5)
    set_color_blind_mode("tritanopia")

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)

    def _combo(*wanted):
        """The one combo offering exactly these keys — none are named."""
        found = [c for c in dlg.findChildren(QComboBox)
                 if [c.itemData(i) for i in range(c.count())] == list(wanted)]
        assert len(found) == 1, f"{len(found)} combos offer {wanted}"
        return found[0]

    theme_combo = _combo(*[token for _label, token in theme_choices()])
    cb_combo = _combo("off", "deuteranopia", "protanopia", "tritanopia")
    language_combo = dlg.findChild(QComboBox, "LanguagePreference")
    assert language_combo is not None
    scale_slider = next(s for s in dlg.findChildren(QSlider)
                        if (s.minimum(), s.maximum()) == (75, 200))

    # It opened on what is stored.
    assert theme_combo.currentData() == "light"
    assert cb_combo.currentData() == "tritanopia"
    assert scale_slider.value() == 150

    # Move every one of them somewhere else, then cancel.
    theme_combo.setCurrentIndex(
        [theme_combo.itemData(i) for i in range(theme_combo.count())]
        .index("dark"))
    cb_combo.setCurrentIndex(0)
    scale_slider.setValue(100)
    assert theme_combo.currentData() != get_theme_choice()
    assert cb_combo.currentData() != get_color_blind_mode()
    assert scale_slider.value() / 100.0 != get_font_scale()

    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Cancel).click()

    assert dlg.result() == QDialog.Rejected
    assert not dlg.isVisible()
    assert get_theme_choice() == "light"
    assert get_color_blind_mode() == "tritanopia"
    assert get_font_scale() == 1.5


# ---------------------------------------------------------------------------
# Animated background
#
# Every test below runs under the module-level ``_isolated_qsettings``
# autouse fixture, which repoints QSettings at a per-test tmp .ini. None of
# this touches the developer's real preferences.
# ---------------------------------------------------------------------------

def _ambient():
    """The real ambient module — the source of truth for valid values."""
    from spacr.qt.widgets import ambient
    return ambient


@pytest.fixture
def fake_ambient(monkeypatch):
    """A controlled stand-in for :mod:`spacr.qt.widgets.ambient`.

    The real module is free to give every theme the same palettes; these
    tests need a theme whose palettes are *disjoint* from another's to
    prove that a theme change repairs a stranded palette, and a theme
    that does not offer ``DEFAULT_PALETTE`` at all to prove the
    first-palette fallback. It also supplies a recording AmbientWidget so
    :func:`apply_ambient_preferences` can be observed.
    """
    import sys
    import types
    from PySide6.QtWidgets import QWidget

    module = types.ModuleType("spacr.qt.widgets.ambient")
    # "bare" is a theme with no palettes at all — the shape a new
    # animation has while its colours are still being written.
    module.AMBIENT_THEMES = ("blobs", "mesh", "bare")
    module.DEFAULT_THEME = "blobs"
    module.DEFAULT_PALETTE = "spacr"
    palettes = {"blobs": ("spacr", "ember"), "mesh": ("steel", "rust"),
                "bare": ()}
    module.palettes_for = lambda theme: palettes.get(theme, ())
    module.theme_label = lambda name: {"blobs": "Diffuse blobs",
                                       "mesh": "Mesh",
                                       "bare": "Bare"}[name]
    module.palette_label = lambda theme, palette: (
        "spaCR" if palette == "spacr" else palette.title())

    class _RecordingAmbient(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.themes = []
            self.palettes = []
            self.animating = None
            self.motion = {}

        def set_theme(self, name):
            self.themes.append(name)

        def set_palette(self, name):
            self.palettes.append(name)

        def set_background_color(self, color):
            pass

        def set_animating(self, on):
            self.animating = bool(on)

        def set_blur(self, value):
            self.motion["blur"] = value

        def set_speed(self, value):
            self.motion["speed"] = value

        def set_size_scale(self, value):
            self.motion["size"] = value

        def set_resolution(self, value):
            self.motion["resolution"] = value

        def set_density(self, value):
            self.motion["density"] = value

        def set_direction(self, name):
            self.motion["direction"] = name

    module.AmbientWidget = _RecordingAmbient
    # Both bindings, so code reaching the module either way sees the same
    # object. It deliberately has no ``theme_note``, which also exercises
    # the dialog's tolerance of a module without one.
    import spacr.qt.widgets as widgets_pkg
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.ambient", module)
    monkeypatch.setattr(widgets_pkg, "ambient", module, raising=False)
    return module


def test_ambient_defaults(qt_theme_applied):
    """Out of the box: on, blobs, spaCR's own colours."""
    from spacr.qt.preferences import (
        get_ambient_enabled, get_ambient_palette, get_ambient_theme,
    )
    ambient = _ambient()
    assert ambient.DEFAULT_THEME == "blobs"
    assert ambient.DEFAULT_PALETTE == "spacr"
    assert get_ambient_enabled() is True
    assert get_ambient_theme() == "blobs"
    assert get_ambient_palette() == "spacr"


def test_ambient_enabled_roundtrip(qt_theme_applied):
    from spacr.qt.preferences import get_ambient_enabled, set_ambient_enabled

    set_ambient_enabled(False)
    assert get_ambient_enabled() is False
    set_ambient_enabled(True)
    assert get_ambient_enabled() is True


def test_ambient_enabled_survives_the_ini_backend(qt_theme_applied):
    """The INI backend hands bools back as the strings "true"/"false"."""
    from PySide6.QtCore import QSettings
    from spacr.qt.preferences import get_ambient_enabled

    QSettings("spacr", "qt").setValue("prefs/ambient_enabled", "false")
    assert get_ambient_enabled() is False
    QSettings("spacr", "qt").setValue("prefs/ambient_enabled", "true")
    assert get_ambient_enabled() is True


def test_ambient_enabled_recovers_from_corrupt_value(qt_theme_applied):
    from PySide6.QtCore import QSettings
    from spacr.qt.preferences import get_ambient_enabled

    QSettings("spacr", "qt").setValue("prefs/ambient_enabled", "garbage")
    assert get_ambient_enabled() is True


def test_ambient_theme_roundtrip(qt_theme_applied):
    from spacr.qt.preferences import get_ambient_theme, set_ambient_theme

    themes = _ambient().AMBIENT_THEMES
    assert themes, "the ambient module must offer at least one theme"
    for theme in themes:
        set_ambient_theme(theme)
        assert get_ambient_theme() == theme


def test_ambient_palette_roundtrip_for_every_theme(qt_theme_applied):
    from spacr.qt.preferences import (
        get_ambient_palette, set_ambient_palette, set_ambient_theme,
    )
    ambient = _ambient()
    for theme in ambient.AMBIENT_THEMES:
        set_ambient_theme(theme)
        assert ambient.palettes_for(theme), f"{theme} offers no palette"
        for palette in ambient.palettes_for(theme):
            set_ambient_palette(palette)
            assert get_ambient_palette() == palette


def test_ambient_blobs_offers_the_spacr_palette(qt_theme_applied):
    """The user asked for a spaCR-coloured blob palette by name."""
    assert "spacr" in _ambient().palettes_for("blobs")


def test_ambient_theme_invalid_raises(qt_theme_applied):
    from spacr.qt.preferences import set_ambient_theme

    with pytest.raises(ValueError, match="unknown ambient theme"):
        set_ambient_theme("kaleidoscope")


def test_ambient_palette_invalid_raises(qt_theme_applied):
    from spacr.qt.preferences import set_ambient_palette

    with pytest.raises(ValueError, match="unknown ambient palette"):
        set_ambient_palette("neon-hotdog")


def test_stored_unknown_ambient_theme_falls_back(qt_theme_applied):
    """A settings file from a newer spaCR must not break an older one."""
    from PySide6.QtCore import QSettings
    from spacr.qt.preferences import get_ambient_theme

    QSettings("spacr", "qt").setValue("prefs/ambient_theme", "hyperspace")
    assert get_ambient_theme() == _ambient().DEFAULT_THEME


def test_stored_unknown_ambient_palette_falls_back(qt_theme_applied):
    from PySide6.QtCore import QSettings
    from spacr.qt.preferences import (
        ambient_default_palette, get_ambient_palette,
    )

    settings = QSettings("spacr", "qt")
    for theme in _ambient().AMBIENT_THEMES:
        settings.setValue("prefs/ambient_theme", theme)
        settings.setValue("prefs/ambient_palette", "chartreuse-nightmare")
        assert get_ambient_palette() == ambient_default_palette(theme)


def test_ambient_palette_never_escapes_its_theme(qt_theme_applied):
    """Whatever is on disk, the getter only ever names a paintable palette."""
    from PySide6.QtCore import QSettings
    from spacr.qt.preferences import get_ambient_palette

    ambient = _ambient()
    settings = QSettings("spacr", "qt")
    for theme in ambient.AMBIENT_THEMES:
        settings.setValue("prefs/ambient_theme", theme)
        for stored in ("", "chartreuse-nightmare", "12345"):
            settings.setValue("prefs/ambient_palette", stored)
            assert get_ambient_palette() in ambient.palettes_for(theme)


def test_setting_the_theme_repairs_a_stranded_palette(qt_theme_applied):
    """Switching themes rewrites a palette the new theme cannot draw."""
    from PySide6.QtCore import QSettings
    from spacr.qt.preferences import set_ambient_theme

    ambient = _ambient()
    settings = QSettings("spacr", "qt")
    for theme in ambient.AMBIENT_THEMES:
        settings.setValue("prefs/ambient_palette", "chartreuse-nightmare")
        set_ambient_theme(theme)
        stored = str(QSettings("spacr", "qt").value("prefs/ambient_palette"))
        assert stored in ambient.palettes_for(theme)


def test_theme_change_carries_a_still_valid_palette_across(qt_theme_applied):
    """A repair must not clobber a choice the new theme also offers."""
    from spacr.qt.preferences import (
        get_ambient_palette, set_ambient_palette, set_ambient_theme,
    )
    ambient = _ambient()
    for source in ambient.AMBIENT_THEMES:
        for target in ambient.AMBIENT_THEMES:
            shared = [p for p in ambient.palettes_for(source)
                      if p in ambient.palettes_for(target)]
            if not shared:
                continue
            set_ambient_theme(source)
            set_ambient_palette(shared[-1])
            set_ambient_theme(target)
            assert get_ambient_palette() == shared[-1]


def test_theme_change_replaces_a_palette_the_new_theme_lacks(
    qt_theme_applied,
):
    """The real modules disagree about palettes — e.g. only some themes
    offer "pastel". Moving to a theme without it must land on that
    theme's default rather than keep a palette it cannot draw."""
    from spacr.qt.preferences import (
        get_ambient_palette, ambient_default_palette, set_ambient_palette,
        set_ambient_theme,
    )
    ambient = _ambient()
    pairs = [
        (source, target, palette)
        for source in ambient.AMBIENT_THEMES
        for target in ambient.AMBIENT_THEMES
        for palette in ambient.palettes_for(source)
        if palette not in ambient.palettes_for(target)
    ]
    assert pairs, "no theme pair with divergent palettes to exercise"
    for source, target, palette in pairs:
        set_ambient_theme(source)
        set_ambient_palette(palette)
        set_ambient_theme(target)
        assert get_ambient_palette() == ambient_default_palette(target)


# --- the same behaviour, against a controlled two-theme module ------------

def test_fake_theme_change_replaces_an_impossible_palette(
    fake_ambient, qt_theme_applied,
):
    from spacr.qt.preferences import (
        get_ambient_palette, set_ambient_palette, set_ambient_theme,
    )
    set_ambient_theme("blobs")
    set_ambient_palette("ember")
    assert get_ambient_palette() == "ember"

    # "mesh" has no "ember" — the palette is repaired, not left dangling
    # and not raised over.
    set_ambient_theme("mesh")
    assert get_ambient_palette() == "steel"


def test_fake_default_palette_falls_to_first_when_spacr_absent(
    fake_ambient, qt_theme_applied,
):
    from spacr.qt.preferences import ambient_default_palette

    assert ambient_default_palette("blobs") == "spacr"
    assert ambient_default_palette("mesh") == "steel"
    # An unknown theme reports the global default rather than exploding.
    assert ambient_default_palette("nope") == "spacr"


def test_fake_palette_valid_for_one_theme_is_rejected_for_another(
    fake_ambient, qt_theme_applied,
):
    from spacr.qt.preferences import set_ambient_palette, set_ambient_theme

    set_ambient_theme("mesh")
    with pytest.raises(ValueError, match="unknown ambient palette"):
        set_ambient_palette("ember")      # belongs to "blobs"


# ---------------------------------------------------------------------------
# Live application of the preference
# ---------------------------------------------------------------------------

def test_apply_ambient_preferences_drives_live_widgets(
    fake_ambient, qtbot, qt_theme_applied,
):
    from spacr.qt.preferences import (
        apply_ambient_preferences, set_ambient_enabled, set_ambient_palette,
        set_ambient_theme,
    )
    widget = fake_ambient.AmbientWidget()
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitExposed(widget)

    set_ambient_enabled(True)
    set_ambient_theme("mesh")
    set_ambient_palette("rust")
    apply_ambient_preferences()
    assert widget.themes[-1] == "mesh"
    assert widget.palettes[-1] == "rust"
    assert widget.animating is True

    # Turning it off must stop the frames, not merely repaint them.
    set_ambient_enabled(False)
    apply_ambient_preferences()
    assert widget.animating is False
    assert widget.isVisible() is False


def test_apply_ambient_preferences_drives_a_real_ambient_widget(
    qtbot, qt_theme_applied,
):
    """The recording double proves the walk; this proves the contract.

    A real :class:`AmbientWidget` built through ``install_ambient`` must
    accept every call the preferences path makes to it, and the toggle
    must actually take the widget off screen and put it back.
    """
    from PySide6.QtWidgets import QVBoxLayout, QWidget
    from spacr.qt.preferences import (
        apply_ambient_preferences, set_ambient_enabled, set_ambient_palette,
        set_ambient_theme,
    )
    from spacr.qt.widgets.ambient import AmbientWidget, install_ambient

    ambient = _ambient()
    host = QWidget()
    qtbot.addWidget(host)
    QVBoxLayout(host)
    widget = install_ambient(host, theme="blobs", palette="spacr")
    assert isinstance(widget, AmbientWidget)
    host.resize(320, 240)
    host.show()
    qtbot.waitExposed(host)

    other = [t for t in ambient.AMBIENT_THEMES if t != "blobs"][0]
    set_ambient_enabled(True)
    set_ambient_theme(other)
    set_ambient_palette(ambient.palettes_for(other)[-1])
    apply_ambient_preferences()
    assert widget.isVisible() is True

    set_ambient_enabled(False)
    apply_ambient_preferences()
    assert widget.isVisible() is False

    set_ambient_enabled(True)
    apply_ambient_preferences()
    assert widget.isVisible() is True
    widget.set_animating(False)     # leave no timer running past the test


def test_apply_ambient_preferences_does_not_mute_offscreen_widgets(
    fake_ambient, qtbot, qt_theme_applied,
):
    """Saving Preferences must not permanently silence background tabs.

    The original concern is real and still holds: every module screen keeps
    its ambient widget alive while the user is on another tab, and Save must
    not set fifty of them ticking at once.

    But this test used to enforce that by asserting the widget's PAUSE FLAG
    stayed False, and `apply_ambient_preferences` was written to satisfy it
    with `set_animating(widget.isVisible())`. Every background tab's backdrop
    is invisible, so saving Preferences latched `_animating = False` onto all
    of them — and `showEvent` honours that flag, which is the entire point of
    a pause. Those screens then never animated again for the rest of the
    session: turning the feature ON switched it off everywhere the user was
    not currently looking.

    The pause flag was never the invariant. "Costs nothing while hidden" is,
    and `AmbientWidget._should_run` already guarantees that by refusing to
    tick while hidden, whatever the flag says. So the flag is now set
    unconditionally and the assertion moved to the property that matters.
    """
    from spacr.qt.preferences import (
        apply_ambient_preferences, set_ambient_enabled,
    )
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    page = QWidget()
    qtbot.addWidget(page)
    layout = QVBoxLayout(page)
    hidden = fake_ambient.AmbientWidget()
    layout.addWidget(hidden)
    # `page` is never shown — exactly the state of a module screen sitting
    # behind another tab.
    assert hidden.isVisible() is False

    set_ambient_enabled(True)
    apply_ambient_preferences()
    assert hidden.themes, "the widget was skipped entirely"
    # Not left latched off. Were this False, the tab would stay dead when the
    # user came back to it.
    assert hidden.animating is True


def test_apply_ambient_preferences_survives_a_dead_widget(
    fake_ambient, qtbot, qt_theme_applied,
):
    """A widget that raises must not fail the preferences save."""
    from spacr.qt.preferences import apply_ambient_preferences

    class _Broken(fake_ambient.AmbientWidget):
        def set_theme(self, name):
            raise RuntimeError("C++ object already deleted")

    broken = _Broken()
    healthy = fake_ambient.AmbientWidget()
    qtbot.addWidget(broken)
    qtbot.addWidget(healthy)

    apply_ambient_preferences()          # must not raise
    assert healthy.themes, "a broken sibling stopped the walk"


def test_apply_ambient_preferences_without_the_module(
    fake_ambient, qtbot, monkeypatch, qt_theme_applied,
):
    """No ambient module (stripped build) → a quiet no-op, never a crash."""
    import builtins
    from spacr.qt.preferences import apply_ambient_preferences

    widget = fake_ambient.AmbientWidget()
    qtbot.addWidget(widget)
    blocked = []
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.endswith("widgets.ambient"):
            blocked.append(name)
            raise ImportError("no ambient module in this build")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    apply_ambient_preferences()
    # Non-vacuous: the import really was blocked, and the walk really did
    # give up rather than half-apply.
    assert blocked, "the ambient import was never attempted"
    assert widget.themes == []
    assert widget.animating is None


def test_apply_ambient_preferences_without_an_application(
    fake_ambient, monkeypatch, qtbot, qt_theme_applied,
):
    """Headless callers (a settings migration, a script) get a no-op.

    A no-op, not merely a non-throw: a live ambient widget is put in front of
    it and must come out with nothing recorded. "Must not raise" on its own
    passed just as happily if the function had walked the tree anyway.
    """
    from PySide6.QtWidgets import QApplication
    from spacr.qt.preferences import apply_ambient_preferences

    widget = fake_ambient.AmbientWidget()
    qtbot.addWidget(widget)
    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
    apply_ambient_preferences()
    assert widget.themes == [] and widget.palettes == []
    assert widget.animating is None


def test_apply_ambient_preferences_when_the_widget_list_fails(
    fake_ambient, qtbot, qt_theme_applied,
):
    """A Qt teardown can make allWidgets() itself fail; do not propagate.

    Asserts the failure was actually reached (``allWidgets`` was called) and
    that the walk stopped there, leaving a live widget untouched — neither of
    which "must not raise" could tell apart from the function returning early
    for some entirely different reason.
    """
    from spacr.qt.preferences import apply_ambient_preferences

    widget = fake_ambient.AmbientWidget()
    qtbot.addWidget(widget)

    class _DyingApp:
        def __init__(self):
            self.asked = 0

        def allWidgets(self):
            self.asked += 1
            raise RuntimeError("application is being destroyed")

    app = _DyingApp()
    apply_ambient_preferences(app)
    assert app.asked == 1, "the widget list was never asked for"
    assert widget.themes == [] and widget.palettes == []
    assert widget.animating is None


def test_apply_preferences_to_app_applies_the_ambient_prefs(
    fake_ambient, qtbot, qt_theme_applied,
):
    """The startup/save path is what makes 'no restart' true."""
    from spacr.qt.preferences import (
        apply_preferences_to_app, set_ambient_blur, set_ambient_density,
        set_ambient_drift_direction, set_ambient_enabled,
        set_ambient_resolution, set_ambient_size, set_ambient_speed,
        set_ambient_theme,
    )
    widget = fake_ambient.AmbientWidget()
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitExposed(widget)
    set_ambient_enabled(True)
    set_ambient_theme("mesh")
    set_ambient_blur(1.5)
    set_ambient_speed(0.5)
    set_ambient_size(2.0)
    set_ambient_resolution(1.75)
    set_ambient_density(2.5)
    set_ambient_drift_direction("random")

    apply_preferences_to_app()
    assert widget.themes[-1] == "mesh"
    assert widget.animating is True
    # Every control rides the same path as the theme, or a screen that was
    # open when Preferences was saved keeps the old motion until it is
    # rebuilt — which for a module screen means until the app restarts.
    # All of them, not a subset: this loop is inside one try/except, so a
    # setter added to Preferences and forgotten here would be swallowed and
    # nobody would find out until a user reported that a slider does
    # nothing until the app is restarted.
    assert widget.motion == {"blur": 1.5, "speed": 0.5, "size": 2.0,
                             "resolution": 1.75, "density": 2.5,
                             "direction": "random"}


# ---------------------------------------------------------------------------
# Preferences UI
# ---------------------------------------------------------------------------

def test_dialog_offers_the_ambient_controls(qtbot, qt_theme_applied):
    from PySide6.QtWidgets import QComboBox
    from spacr.qt.i18n import tr
    from spacr.qt.preferences import PreferencesDialog
    from spacr.qt.widgets.toggle import Toggle

    ambient = _ambient()
    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)

    check = dlg.findChild(Toggle, "AmbientEnabled")
    theme_combo = dlg.findChild(QComboBox, "AmbientTheme")
    palette_combo = dlg.findChild(QComboBox, "AmbientPalette")
    assert check is not None and theme_combo is not None
    assert palette_combo is not None

    assert check.isChecked() is True
    keys = [theme_combo.itemData(i) for i in range(theme_combo.count())]
    assert keys == list(ambient.AMBIENT_THEMES)
    # Human labels, not raw keys.
    labels = [theme_combo.itemText(i) for i in range(theme_combo.count())]
    assert labels == [tr(ambient.theme_label(k)) for k in keys]

    theme = theme_combo.currentData()
    palette_keys = [palette_combo.itemData(i)
                    for i in range(palette_combo.count())]
    assert palette_keys == list(ambient.palettes_for(theme))
    palette_labels = [palette_combo.itemText(i)
                      for i in range(palette_combo.count())]
    assert palette_labels == [tr(ambient.palette_label(theme, p))
                              for p in palette_keys]


def test_dialog_palette_list_follows_the_selected_theme(qtbot,
                                                        qt_theme_applied):
    from PySide6.QtWidgets import QComboBox
    from spacr.qt.preferences import PreferencesDialog

    ambient = _ambient()
    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    theme_combo = dlg.findChild(QComboBox, "AmbientTheme")
    palette_combo = dlg.findChild(QComboBox, "AmbientPalette")

    for index in range(theme_combo.count()):
        theme_combo.setCurrentIndex(index)
        theme = theme_combo.itemData(index)
        offered = [palette_combo.itemData(i)
                   for i in range(palette_combo.count())]
        assert offered == list(ambient.palettes_for(theme))
        assert palette_combo.currentData() in ambient.palettes_for(theme)
        # The picker also says what the animation looks like.
        from spacr.qt.i18n import tr
        assert theme_combo.toolTip() == tr(ambient.theme_note(theme))
        assert theme_combo.toolTip()


def test_dialog_disables_the_pickers_when_the_animation_is_off(
    qtbot, qt_theme_applied,
):
    from PySide6.QtWidgets import QComboBox
    from spacr.qt.preferences import PreferencesDialog
    from spacr.qt.widgets.toggle import Toggle

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    check = dlg.findChild(Toggle, "AmbientEnabled")
    theme_combo = dlg.findChild(QComboBox, "AmbientTheme")
    palette_combo = dlg.findChild(QComboBox, "AmbientPalette")

    assert theme_combo.isEnabled() and palette_combo.isEnabled()
    check.setChecked(False)
    assert not theme_combo.isEnabled()
    assert not palette_combo.isEnabled()
    check.setChecked(True)
    assert theme_combo.isEnabled() and palette_combo.isEnabled()


def test_dialog_saves_the_ambient_choices(qtbot, qt_theme_applied):
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox
    from spacr.qt.preferences import (
        PreferencesDialog, get_ambient_enabled, get_ambient_palette,
        get_ambient_theme,
    )
    from spacr.qt.widgets.toggle import Toggle

    ambient = _ambient()
    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    theme_combo = dlg.findChild(QComboBox, "AmbientTheme")
    palette_combo = dlg.findChild(QComboBox, "AmbientPalette")

    theme_combo.setCurrentIndex(theme_combo.count() - 1)
    wanted_theme = theme_combo.currentData()
    palette_combo.setCurrentIndex(palette_combo.count() - 1)
    wanted_palette = palette_combo.currentData()
    assert wanted_palette in ambient.palettes_for(wanted_theme)

    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert get_ambient_theme() == wanted_theme
    assert get_ambient_palette() == wanted_palette
    assert get_ambient_enabled() is True

    # ...and the off switch, which is the user's explicit ask.
    dlg2 = PreferencesDialog()
    qtbot.addWidget(dlg2)
    dlg2.findChild(Toggle, "AmbientEnabled").setChecked(False)
    dlg2.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert get_ambient_enabled() is False

    # Reopening reflects what was saved rather than the defaults.
    dlg3 = PreferencesDialog()
    qtbot.addWidget(dlg3)
    assert dlg3.findChild(Toggle, "AmbientEnabled").isChecked() is False
    assert (dlg3.findChild(QComboBox, "AmbientTheme").currentData()
            == wanted_theme)
    assert (dlg3.findChild(QComboBox, "AmbientPalette").currentData()
            == wanted_palette)


def test_dialog_save_never_writes_an_impossible_pair(fake_ambient, qtbot,
                                                     qt_theme_applied):
    """Switching theme in the dialog cannot save a palette it cannot draw."""
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox
    from spacr.qt.preferences import (
        PreferencesDialog, get_ambient_palette, get_ambient_theme,
        set_ambient_palette, set_ambient_theme,
    )
    set_ambient_theme("blobs")
    set_ambient_palette("ember")

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    theme_combo = dlg.findChild(QComboBox, "AmbientTheme")
    keys = [theme_combo.itemData(i) for i in range(theme_combo.count())]
    theme_combo.setCurrentIndex(keys.index("mesh"))
    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()

    assert get_ambient_theme() == "mesh"
    assert get_ambient_palette() in fake_ambient.palettes_for("mesh")


def test_dialog_save_survives_a_theme_with_no_palettes(fake_ambient, qtbot,
                                                       qt_theme_applied):
    """A decorative background must never block the Save button.

    An animation whose palettes are not defined leaves the palette combo
    empty; saving must still store the theme and close the dialog.
    """
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox
    from spacr.qt.preferences import (
        PreferencesDialog, get_ambient_palette, get_ambient_theme,
    )

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    theme_combo = dlg.findChild(QComboBox, "AmbientTheme")
    palette_combo = dlg.findChild(QComboBox, "AmbientPalette")
    keys = [theme_combo.itemData(i) for i in range(theme_combo.count())]
    theme_combo.setCurrentIndex(keys.index("bare"))
    assert palette_combo.count() == 0
    assert theme_combo.toolTip() == ""      # this fake has no theme_note

    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    from PySide6.QtWidgets import QDialog
    assert dlg.result() == QDialog.Accepted
    assert get_ambient_theme() == "bare"
    # And the getter still names something rather than blowing up.
    assert get_ambient_palette() == "spacr"


# ---------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------

def test_preferences_imports_without_touching_the_ambient_widget():
    """``preferences`` must stay importable without constructing widgets.

    Run in a clean interpreter with HOME/XDG redirected, so this also
    proves the shipped default on a machine that has never run spaCR —
    and proves it without reading the developer's real settings.
    """
    import os
    import subprocess
    import sys
    import tempfile

    import spacr

    root = os.path.dirname(os.path.dirname(os.path.abspath(spacr.__file__)))
    code = (
        "import sys\n"
        "import spacr.qt.preferences as prefs\n"
        "assert 'spacr.qt.widgets.ambient' not in sys.modules, 'eager ambient'\n"
        "assert 'PySide6.QtWidgets' not in sys.modules, 'eager QtWidgets'\n"
        "assert prefs.get_ambient_enabled() is True\n"
    )
    with tempfile.TemporaryDirectory() as home:
        env = dict(os.environ)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        env["HOME"] = home
        env["XDG_CONFIG_HOME"] = os.path.join(home, "config")
        env["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
