"""spaCR draws in the font it ships, not in the platform's.

The bundled Open Sans was REGISTERED and never APPLIED: nothing set the
application font, so every widget without an explicit `font-family` rule used
the platform default. That is what Qt was naming in "OpenType support missing
for Ubuntu Sans" -- the application was drawing in the system font.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QFont, QFontDatabase  # noqa: E402

from spacr.qt import app as qt_app  # noqa: E402
from spacr.qt import preferences as prefs  # noqa: E402


@pytest.fixture
def application(qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    instance = QApplication.instance()
    qt_app._load_bundled_fonts()
    return instance


def test_the_bundled_family_is_registered(application):
    assert "Open Sans" in set(QFontDatabase.families())


def test_every_weight_we_offer_is_bundled():
    import os

    here = os.path.dirname(os.path.dirname(qt_app.__file__))
    static = os.path.join(here, "resources", "font", "open_sans", "static")
    shipped = set(os.listdir(static))
    assert "OpenSans-Regular.ttf" in shipped
    assert "OpenSans-Light.ttf" in shipped


def test_it_becomes_the_application_font(application):
    """Registered is not applied. This is the assertion that was missing."""
    assert qt_app._use_open_sans(application) == "Open Sans"
    assert application.font().family() == "Open Sans"


def test_regular_is_weight_400(application):
    qt_app._use_open_sans(application, "regular")
    assert application.font().weight() == QFont.Weight.Normal


def test_light_is_weight_300(application):
    qt_app._use_open_sans(application, "light")
    assert application.font().weight() == QFont.Weight.Light


def test_the_point_size_the_platform_chose_is_kept(application):
    """The font-scale preference is applied on top; overriding the size
    here would silently undo it."""
    before = application.font().pointSizeF()
    qt_app._use_open_sans(application)
    if before > 0:
        assert application.font().pointSizeF() == pytest.approx(before)


def test_a_missing_family_leaves_the_platform_font_alone(application,
                                                         monkeypatch):
    """A font that will not load is not a reason to refuse to draw."""
    monkeypatch.setattr(QFontDatabase, "families", staticmethod(lambda: []))
    assert qt_app._use_open_sans(application) == ""


def test_the_weight_round_trips(application):
    for weight in prefs.INTERFACE_FONT_WEIGHTS:
        prefs.set_interface_font_weight(weight)
        assert prefs.get_interface_font_weight() == weight


def test_an_unknown_weight_is_refused(application):
    with pytest.raises(ValueError, match="unknown interface font weight"):
        prefs.set_interface_font_weight("ultrablack")


def test_an_unreadable_stored_weight_reads_as_regular(application):
    prefs._settings().setValue(prefs._KEY_FONT_WEIGHT, "sideways")
    assert prefs.get_interface_font_weight() == "regular"


def test_setting_it_applies_it_to_the_running_application(application):
    prefs.set_interface_font_weight("light")
    assert application.font().weight() == QFont.Weight.Light
    prefs.set_interface_font_weight("regular")
    assert application.font().weight() == QFont.Weight.Normal


def test_launch_actually_calls_it():
    """Wired, not merely defined -- the bug being fixed is exactly a font
    that was loaded and never used."""
    import inspect

    assert "_use_open_sans(app)" in inspect.getsource(qt_app.launch)


def test_the_row_has_a_caption():
    assert "Interface font" in prefs.PREFERENCE_TIPS
