"""Style sheets name only font families the platform has.

On an Intel Mac (2026-10-18) every launch logged "Populating font family
aliases took 120-780 ms. Replace uses of missing font family "Segoe UI"":
naming a family macOS lacks makes Qt scan every installed font name. Segoe
UI is a Windows family, so only Windows style sheets name it.
"""
import sys

from spacr.qt import theme


def test_segoe_ui_is_named_only_on_windows():
    assert ("Segoe UI" in theme.QSS_FONT_FAMILY) == sys.platform.startswith("win")
    assert theme.QSS_FONT_FAMILY.startswith('"Open Sans"')


def test_the_application_sheet_uses_the_platform_family():
    sheet = theme.stylesheet("dark")
    assert f"font-family: {theme.QSS_FONT_FAMILY};" in sheet
    if not sys.platform.startswith("win"):
        assert "Segoe UI" not in sheet
