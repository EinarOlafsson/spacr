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


def test_the_ai_usage_row_is_transparent_like_the_usage_bar_row():
    """2026-10-20 macOS: the console's "N% of your share left" row (object
    name AiUsageRow) painted pure black behind its bar and text; the
    transparency rule named only UsageBarRow."""
    sheet = theme.stylesheet("dark")
    rule = sheet[sheet.index("QWidget#CardBody, QWidget#UsageBarRow"):]
    rule = rule[:rule.index("}")]
    assert "QWidget#AiUsageRow" in rule and "QLabel#AiUsageText" in rule
    assert "background: transparent" in rule
