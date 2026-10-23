"""Style sheets name only font families the platform has.

On an Intel Mac (2026-10-18) every launch logged "Populating font family
aliases took 120-780 ms. Replace uses of missing font family "Segoe UI"":
naming a family macOS lacks makes Qt scan every installed font name. Segoe
UI is a Windows family, so only Windows style sheets name it.
"""
import sys
import re

import pytest

from spacr.qt import theme


def _assert_platform_families(sheets):
    declarations = [match for sheet in sheets for match in
                    re.findall(r'font-family\s*:\s*([^;{}]+)', sheet)]
    assert declarations
    expected = [family.strip().strip('\"\'')
                for family in theme.QSS_FONT_FAMILY.split(',')]
    for declaration in declarations:
        families = [family.strip().strip('\"\'') for family in declaration.split(',')]
        assert families == expected, declaration
        if sys.platform == 'darwin':
            assert not {'sans-serif', 'serif', 'monospace', 'segoe ui'}.intersection(
                family.casefold() for family in families)
    return len(declarations)


@pytest.mark.parametrize('palette', ['dark', 'light'])
def test_every_application_font_declaration_uses_platform_families(palette):
    sheet = theme.stylesheet(palette)
    rule = _assert_shortcut_font_rule(sheet)
    remaining = sheet[:rule.start()] + sheet[rule.end():]
    console = list(re.finditer(r'QTextEdit#GateConsoleLog\s*\{([^}]+)\}', remaining))
    assert len(console) == 1
    _assert_monospace_family(console[0].group(1))
    _assert_platform_families([remaining[:console[0].start()] + remaining[console[0].end():]])


def _assert_monospace_family(rule):
    declarations = re.findall(r'font-family\s*:\s*([^;{}]+)', rule)
    expected = 'Menlo' if sys.platform == 'darwin' else 'monospace'
    assert [family.strip().strip('\"\'') for family in declarations] == [expected]


def _assert_shortcut_font_rule(sheet):
    rules = list(re.finditer(r'QLabel#ShortcutOverlayKeys\s*\{([^}]+)\}', sheet))
    assert len(rules) == 1
    _assert_monospace_family(rules[0].group(1))
    return rules[0]


def test_shortcut_keys_keep_an_available_platform_monospace_family(qapp):
    from PySide6.QtGui import QFontDatabase

    _assert_shortcut_font_rule(theme.stylesheet('dark'))
    if sys.platform == 'darwin':
        assert 'Menlo' in QFontDatabase.families()


@pytest.mark.parametrize('generator', ['_ghost_btn_qss', '_primary_btn_qss'])
def test_actual_tour_button_fonts_use_platform_families(generator):
    from spacr.qt import first_run

    assert _assert_platform_families([getattr(first_run, generator)()]) == 1


def test_every_generated_home_font_declaration_uses_platform_families(qtbot):
    from PySide6.QtWidgets import QWidget
    from shiboken6 import delete
    from spacr.qt.widgets.home import HomePage, Panel

    page = HomePage.__new__(HomePage)
    QWidget.__init__(page)
    page._P = theme.palette_for('dark')
    page._section_notes = {}
    page._grids = []
    panel = Panel('Test panel')
    qtbot.addWidget(panel)
    panel.add_action('Refresh')
    try:
        roots = [panel, page._build_hero(), page._band_header('Core', 0),
                 page._build_category_tab('Core', [])]
    finally:
        delete(page)
    for root in roots[1:]:
        qtbot.addWidget(root)
    sheets = [widget.styleSheet() for root in roots
              for widget in [root, *root.findChildren(QWidget)]]
    assert _assert_platform_families(sheets) == 5


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
    assert "QWidget#ActionStatusRow" in rule, "the black row under Run and Stop"
    assert "background: transparent" in rule


def test_macos_sheets_do_not_ask_for_the_generic_sans_serif():
    """2026-10-20 (Intel Mac, installed 1.5.1.5): with the generic family last
    in the list, sheeting the startup window still logged "Populating font
    family aliases ... missing font family "Sans-serif"". Open Sans is
    bundled and Helvetica Neue ships with every macOS, so nothing is lost."""
    if sys.platform == "darwin":
        assert theme.QSS_FONT_FAMILY == '"Open Sans", "Helvetica Neue"'
    else:
        assert theme.QSS_FONT_FAMILY.endswith("sans-serif")
