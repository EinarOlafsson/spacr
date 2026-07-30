"""Offscreen construction and visual-contract smoke tests for every module.

Most screen tests instantiate one screen class directly.  That leaves two
important integration gaps:

* a new entry can be added to :data:`spacr.qt.app.APPS` but fail only through
  the real ``MainWindow`` factory; and
* a QWidget method name can accidentally override a Qt virtual.  The former
  ``TrainCompareScreen.metric()`` did exactly that: every ordinary unit test
  passed, while showing the screen made Qt call it with a paint-device
  argument and could terminate the process.

This module deliberately shows and grabs every registered screen using the
offscreen platform.  For the shared settings screen it also checks the two
visual/help contracts users see on every row: linked help lives on the label,
and the label/API-dot wrapper paints the section's dark-gray surface rather
than an opaque black rectangle.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, QPoint
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from spacr.qt.app import APPS, MainWindow
from spacr.qt.button_roles import action_role, install_button_roles
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.theme import DARK_PALETTE
from spacr.qt.widgets.info_link import InfoLink


class _FactoryHost(QObject):
    """Small receiver exposing the callbacks ``_build_screen`` connects."""

    def _on_train_requested(self, *_args):
        pass

    def _snapshot_current_screen_settings(self):
        return "mask", {}

    def _on_zoo_compare_requested(self, *_args):
        pass

    def _on_explain_error(self, *_args):
        pass


def _setting_row_contract(screen: AppScreen, qapp) -> None:
    """Check label help, API links and the rendered section background."""
    rows = [
        (section, label, field)
        for section in screen._settings_sections
        for label, field in section._row_widgets
    ]
    assert rows, f"{screen.app_key}: the settings screen has no rows"

    for _section, label, field in rows:
        key = field.property("settingKey")
        assert key, f"{screen.app_key}: a setting field has no settingKey"
        assert isinstance(label, QLabel), (
            f"{screen.app_key}.{key}: setting label is not a QLabel")
        # AppScreen uses the clickable HoverTooltip rather than Qt's native
        # tooltip (the native popup vanishes before its API link can be
        # clicked). The linked HTML therefore lives in label metadata/map.
        html = (
            label.property("apiTooltipHtml")
            or screen._html_tip_map.get(label, "")
        )
        assert html, (
            f"{screen.app_key}.{key}: help is missing from the label")
        assert "href=" in html, (
            f"{screen.app_key}.{key}: label help has no API link")
        wrapper = label.parentWidget()
        assert wrapper is not None
        assert wrapper.objectName() == "SettingLabelWithInfo", (
            f"{screen.app_key}.{key}: label/API wrapper is missing")
        links = wrapper.findChildren(InfoLink)
        assert len(links) == 1, (
            f"{screen.app_key}.{key}: expected one API dot, got {len(links)}")
        assert links[0].url().startswith(("https://", "http://"))

    # Render one representative row per module.  Structural QSS assertions
    # alone missed the original black rectangle: only the composed screenshot
    # proves a transparent wrapper actually shows its SectionCard through.
    section, label, _field = rows[0]
    section.show()
    section.set_expanded(True)
    qapp.processEvents()
    wrapper = label.parentWidget()
    image = screen.grab().toImage()
    origin = wrapper.mapTo(screen, QPoint(0, 0))
    points = (
        (1, 1),
        (max(1, wrapper.width() - 2), 1),
        (1, max(1, wrapper.height() - 2)),
        (max(1, wrapper.width() - 2), max(1, wrapper.height() - 2)),
    )
    expected = QColor(DARK_PALETTE["surface"]).name()
    observed = [
        QColor(image.pixel(origin.x() + x, origin.y() + y)).name()
        for x, y in points
    ]
    assert observed == [expected] * len(points), (
        f"{screen.app_key}: setting-label background is {observed}, "
        f"expected its container color {expected}")


@pytest.mark.parametrize(
    "app_key",
    [key for key, _name, _description, _section in APPS],
)
def test_every_registered_module_constructs_shows_and_renders(
        qtbot, qt_theme_applied, app_key):
    """Exercise the real screen factory and one composed dark-theme frame."""
    install_button_roles(qt_theme_applied)
    host = _FactoryHost()
    screen = MainWindow._build_screen(host, app_key)
    qtbot.addWidget(screen)
    screen.resize(1200, 720)
    screen.show()
    qt_theme_applied.processEvents()

    build_errors = [
        label.text() for label in screen.findChildren(QLabel)
        if label.text().startswith("Failed to build settings")
    ]
    assert not build_errors, f"{app_key}: {build_errors}"
    frame = screen.grab()
    assert not frame.isNull(), f"{app_key}: screen did not render"
    # A few dense tools declare a minimum width larger than 1200. Qt correctly
    # honours that contract when resize(1200, 720) is requested.
    assert frame.width() >= 1200 and frame.height() >= 720

    # Every requested semantic action is tagged after its Show/Polish event.
    for button in screen.findChildren(QPushButton):
        expected_role = action_role(button.text())
        if expected_role is not None:
            assert button.property("buttonActionRole") == expected_role, (
                f"{app_key}: {button.text()!r} is not styled as "
                f"{expected_role}")

    if isinstance(screen, AppScreen):
        _setting_row_contract(screen, qt_theme_applied)


def test_training_runs_does_not_override_qwidget_metric():
    """Qt must retain its ``QPaintDevice.metric(enum)`` virtual method."""
    from spacr.qt.screens.train_compare import TrainCompareScreen

    assert "metric" not in TrainCompareScreen.__dict__
