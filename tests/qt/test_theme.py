"""Theme + palette invariants."""
from __future__ import annotations

import pytest

from spacr.qt import theme


def test_palette_has_required_keys():
    required = {
        "bg", "surface", "surface_alt", "border", "border_soft",
        "fg", "fg_muted", "fg_dim",
        "accent", "accent_hi", "accent_lo",
        "success", "warning", "error", "info",
    }
    assert required.issubset(theme.DARK_PALETTE.keys())


def test_palette_uses_hex_colors():
    for name, value in theme.DARK_PALETTE.items():
        assert value.startswith("#") and len(value) == 7, \
            f"{name} = {value!r} is not a #rrggbb hex color"


def test_spacing_scale():
    assert set(theme.SPACING) == {"xs", "sm", "md", "lg", "xl", "xxl"}
    values = [theme.SPACING[k] for k in ("xs", "sm", "md", "lg", "xl", "xxl")]
    assert values == sorted(values), "SPACING must be monotonically increasing"


def test_stylesheet_is_non_empty_and_references_palette():
    qss = theme.stylesheet()
    assert isinstance(qss, str) and len(qss) > 1000
    # Must include the accent color so at least one selector is styled.
    assert theme.DARK_PALETTE["accent"] in qss
    # Common named selectors we rely on.
    for name in ("#Sidebar", "#SidebarItem", "#Card", "#Tile",
                 "#PrimaryButton", "#DangerButton", "#Console",
                 "#UsageBar", "QMainWindow"):
        assert name in qss, f"stylesheet is missing {name}"
    assert '[buttonActionRole="positive"]' in qss
    assert '[buttonActionRole="negative"]' in qss
    assert '[buttonActionBusy="true"]' in qss
    # API-help label wrappers are layout-only. They must not paint the black
    # window canvas over a dark-gray SectionCard.
    assert "QWidget#SettingLabelWithInfo" in qss
    assert "QWidget#SettingControlWithInfo" in qss
    assert "background: transparent" in qss
    # SQL column-picker tool buttons need their own theme rule because they
    # are QToolButtons rather than QPushButtons.
    assert "QToolButton#ColumnPickerButton" in qss
    assert "QToolButton#ColumnPickerButton:hover" in qss
    assert "QWidget#ColumnPickerRow" in qss


def test_apply_qpalette_sets_expected_colors(qapp):
    from PySide6.QtGui import QPalette, QColor
    theme.apply_qpalette(qapp)
    pal = qapp.palette()
    assert pal.color(QPalette.Window) == QColor(theme.DARK_PALETTE["bg"])
    assert pal.color(QPalette.WindowText) == QColor(theme.DARK_PALETTE["fg"])
    assert pal.color(QPalette.Highlight) == QColor(theme.DARK_PALETTE["accent"])
