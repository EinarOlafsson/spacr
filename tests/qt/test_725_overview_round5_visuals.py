"""Round-5 Overview visual fixes: the framed filename regex and the light panel colour.

Item 725, round 5, approved by the maintainer: the metadata step's "Filename
regex" field showed no frame, and the plate, well and image panels looked
mid-grey in the light theme.
"""
from spacr.qt.theme import palette_for, stylesheet, widget_qss_names
from spacr.qt.widgets import qc_field_browser as browser


def _rule(sheet, selector):
    start = sheet.index(selector + " {")
    return sheet[start:sheet.index("}", start)]


def test_the_panel_colour_is_part_of_every_stylesheet_built_for_a_theme():
    assert "FieldOverviewPanel" in widget_qss_names()
    for theme in ("light", "dark"):
        rule = _rule(stylesheet(theme), "QFrame#FieldOverviewPanel")
        assert f"background: {palette_for(theme)['surface_alt']};" in rule
    assert "#161719" not in _rule(stylesheet("light"), "QFrame#FieldOverviewPanel")


def test_the_regex_field_keeps_a_full_frame_in_every_theme(qtbot):
    wizard = browser._OverviewMetadataWizard("")
    qtbot.addWidget(wizard)
    assert wizard.regex.objectName() == "OverviewWizardRegex"
    assert wizard.regex.property("spacrNoFieldFade") is True
    from spacr.qt.widgets.field_fade import fades

    assert not fades(wizard.regex)
    for theme in ("light", "dark"):
        palette = palette_for(theme)
        sheet = stylesheet(theme)
        rule = _rule(sheet, "QLineEdit#OverviewWizardRegex")
        assert f"border: 1px solid {palette['border']};" in rule
        assert f"background-color: {palette['surface_alt']};" in rule
        assert palette["accent"] in _rule(sheet, "QLineEdit#OverviewWizardRegex:focus")
        assert "QLineEdit#OverviewWizardRegex" in browser._overview_module_qss(palette, None)
