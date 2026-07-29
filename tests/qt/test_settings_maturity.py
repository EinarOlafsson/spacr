"""Maturity colours propagate from Home's registry into module settings."""
from __future__ import annotations

from PySide6.QtWidgets import QLabel, QLineEdit


def test_section_maturity_uses_the_least_mature_app_or_category_stage():
    from spacr.qt.screens.app_screen import settings_section_maturity

    assert settings_section_maturity("mask", "Paths") == "stable"
    assert settings_section_maturity("mask", "3D Settings (Beta)") == "beta"
    assert settings_section_maturity("mask", "4D Settings (Beta)") == "beta"
    assert settings_section_maturity("mask", "Motility (beta)") == "beta"
    assert settings_section_maturity("timelapse", "Paths") == "beta"
    assert settings_section_maturity("invasion", "Paths") == "alpha"
    assert settings_section_maturity(
        "invasion", "3D Settings (Beta)") == "alpha"


def test_section_marks_its_card_header_and_every_setting(qtbot):
    from spacr.qt.widgets.section import Section

    section = Section("Measurements")
    qtbot.addWidget(section)
    label = QLabel("3D measurement")
    field = QLineEdit()
    section.add_row(label, field)
    section.set_maturity("beta")
    section.set_hint("Measurement controls.")

    assert section.maturity() == "beta"
    assert section.property("maturity") == "beta"
    assert section._header.property("maturity") == "beta"
    assert section._body.property("maturity") == "beta"
    assert label.property("settingMaturity") == "beta"
    assert field.property("settingMaturity") == "beta"
    assert "BETA" in section._header.text()
    assert "Measurement controls." in section._header.toolTip()
    assert "not signed off" in section._header.toolTip()
    assert "Beta maturity" in section.accessibleDescription()


def test_full_width_section_widgets_receive_the_same_maturity(qtbot):
    from spacr.qt.widgets.section import Section

    section = Section("Alpha")
    qtbot.addWidget(section)
    field = QLineEdit()
    section.set_maturity("alpha")
    section.add_widget(field)

    assert field.property("settingMaturity") == "alpha"
    assert "ALPHA" in section._header.text()


def test_maturity_badge_is_not_duplicated_in_section_header(qtbot):
    from spacr.qt.widgets.section import Section

    beta = Section("3D Settings (Beta)")
    qtbot.addWidget(beta)
    beta.set_maturity("beta")
    assert beta.title() == "3D SETTINGS (BETA)"
    assert beta._header.text() == "3D SETTINGS   ·   BETA"
    assert beta._header.text().count("BETA") == 1

    stage_only = Section("Beta")
    qtbot.addWidget(stage_only)
    stage_only.set_maturity("beta")
    assert stage_only._header.text() == "·   BETA"


def test_invalid_maturity_falls_back_to_stable(qtbot):
    from spacr.qt.widgets.section import Section

    section = Section("General")
    qtbot.addWidget(section)
    section.set_maturity("experimental-ish")

    assert section.maturity() == "stable"
    assert section.property("maturity") == "stable"
    assert "STABLE" not in section._header.text()


def test_stylesheet_uses_home_stage_hues_for_sections_and_rows():
    from spacr.qt import theme

    qss = theme.stylesheet("dark")
    for stage, hue in theme.STAGE_HOVER.items():
        assert f'QFrame#SectionCard[maturity="{stage}"]' in qss
        assert f'QLabel[settingMaturity="{stage}"]' in qss
        assert f"border-left: 4px solid {hue}" in qss
        assert f"border-left: 2px solid {hue}" in qss


def test_beta_module_colours_every_settings_section(
    qtbot,
    qt_theme_applied,
):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.section import Section

    screen = AppScreen("timelapse")
    qtbot.addWidget(screen)
    sections = screen.findChildren(Section)

    assert sections
    assert {section.maturity() for section in sections} == {"beta"}
    assert all(
        widget.property("settingMaturity") == "beta"
        for section in sections
        for _label, widget in section._row_widgets
    )


def test_alpha_module_colours_every_settings_section(
    qtbot,
    qt_theme_applied,
):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.section import Section

    screen = AppScreen("invasion")
    qtbot.addWidget(screen)
    sections = screen.findChildren(Section)

    assert sections
    assert {section.maturity() for section in sections} == {"alpha"}


def test_stable_mask_module_keeps_only_experimental_section_beta(
    qtbot,
    qt_theme_applied,
):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.section import Section

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    by_title = {section.title(): section for section in screen.findChildren(Section)}

    assert by_title["3D SETTINGS (BETA)"].maturity() == "beta"
    assert by_title["4D SETTINGS (BETA)"].maturity() == "beta"
    three_d_labels = {
        label.text() for label, _widget
        in by_title["3D SETTINGS (BETA)"]._row_widgets
    }
    four_d_labels = {
        label.text() for label, _widget
        in by_title["4D SETTINGS (BETA)"]._row_widgets
    }
    assert "Z stack" in three_d_labels
    assert "T stack" in four_d_labels
    assert by_title["PATHS"].maturity() == "stable"
    assert by_title["GENERAL"].maturity() == "stable"
