"""No information dot is drawn, and every setting still links its API page.

"in the help menue remove the blue dots with an i."

They were the API-link dots ``install_api_tooltips`` put beside a setting
label. Three forms had already switched them off one at a time -- 68 of them
down the Mask live preview, twenty-six down the Annotate settings dialog,
three in the figure dialog -- each recording the same complaint: a column of
dots reads as texture rather than as one affordance per setting.

Removing them costs nothing because the API link was never in the dot alone.
What these tests hold:

* NO DOT IS DRAWN, on a hand-built form or on a real module screen.
* THE PARAMETER IS GONE, not merely defaulted off. A flag with one value is
  a flag nobody reads, and nothing passes it any more.
* EVERY SETTING THAT CARRIED A DOT STILL OFFERS ITS API LINK ON HOVER --
  which is the whole reason the dot is safe to drop, so it is asserted
  rather than assumed.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (
    QApplication, QFormLayout, QLabel, QSpinBox, QWidget,
)

from spacr.qt import theme
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.settings_model import install_api_tooltips

QT_PACKAGE = Path(theme.__file__).resolve().parent


def _api_links(root) -> list:
    """Every widget below ``root`` drawn as a clickable API-link dot."""
    return [child for child in root.findChildren(QWidget)
            if child.property("apiTooltipDisplayRole") == "api-link"
            or child.objectName() in ("SettingInfoLink", "InfoLink")]


def _help_labels(root) -> list:
    """Every setting label the decoration pass put hover help on."""
    return [child for child in root.findChildren(QWidget)
            if child.property("settingHelpLabel")]


def _labels_with_help(root) -> list:
    """Every label carrying setting help, however it was built.

    A module screen writes the same properties itself when it lays its form
    out, so the dot's removal has to be checked against that path too --
    it is the one a user opens first.
    """
    return [child for child in root.findChildren(QWidget)
            if child.property("apiTooltipDisplayRole") == "tooltip"]


def _form(qtbot, keys=("cell_diameter", "cell_channel", "cell_min_size")):
    """A hand-built settings form of the shape the decorator sweeps."""
    owner = QWidget()
    qtbot.addWidget(owner)
    layout = QFormLayout(owner)
    for key in keys:
        field = QSpinBox()
        field.setProperty("settingKey", key)
        layout.addRow(QLabel(key.replace("_", " ").capitalize()), field)
    return owner


# ---------------------------------------------------------------------------
# The dots are gone
# ---------------------------------------------------------------------------

def test_a_decorated_form_draws_no_dot(qtbot, qt_theme_applied):
    """The plain case: three settings, no dots."""
    owner = _form(qtbot)

    install_api_tooltips(owner, "mask")

    assert _api_links(owner) == []


def test_a_form_decorated_again_still_draws_no_dot(qtbot, qt_theme_applied):
    """The live-preview form is re-decorated whenever it is re-gated."""
    owner = _form(qtbot)

    for _ in range(4):
        install_api_tooltips(owner, "mask")

    assert _api_links(owner) == []


def test_a_self_labelling_control_draws_no_dot(qtbot, qt_theme_applied):
    """A Toggle/QCheckBox is its own row label and got a dot of its own."""
    from PySide6.QtWidgets import QCheckBox, QVBoxLayout

    owner = QWidget()
    qtbot.addWidget(owner)
    column = QVBoxLayout(owner)
    box = QCheckBox("Save PNG stacks")
    box.setProperty("settingKey", "save_png")
    column.addWidget(box)

    install_api_tooltips(owner, "mask")

    assert _api_links(owner) == []


@pytest.mark.parametrize("app_key", ["mask", "measure", "umap"])
def test_a_real_module_screen_draws_no_dot(app_key, qtbot, qt_theme_applied):
    """The settings form a user actually opens, not a stand-in."""
    screen = AppScreen(app_key)
    qtbot.addWidget(screen)

    dots = [d for d in _api_links(screen)
            if d.objectName() == "SettingInfoLink"]

    assert dots == [], f"{app_key} still draws {len(dots)} setting dots"


# ---------------------------------------------------------------------------
# The link survives, on hover, for every setting that carried a dot
# ---------------------------------------------------------------------------

def test_every_decorated_setting_still_links_its_api_page(
        qtbot, qt_theme_applied):
    """The whole reason the dot is safe to drop."""
    owner = _form(qtbot)

    install_api_tooltips(owner, "mask")
    labels = _help_labels(owner)

    assert len(labels) == 3
    for label in labels:
        html = label.property("apiTooltipHtml")
        assert "href=" in str(html), f"{label.text()} lost its API link"
        assert label.toolTip() == html


@pytest.mark.parametrize("app_key", ["mask", "measure", "umap"])
def test_a_real_module_keeps_the_link_on_every_setting_label(
        app_key, qtbot, qt_theme_applied):
    """Measured on the screen, not on the helper that builds it."""
    screen = AppScreen(app_key)
    qtbot.addWidget(screen)

    labels = _labels_with_help(screen)

    assert labels, f"{app_key} decorated no setting label at all"
    without = [label for label in labels
               if "href=" not in str(label.property("apiTooltipHtml") or "")]
    assert without == [], (
        f"{app_key}: {len(without)} setting labels lost their API link")


def test_hovering_a_setting_label_still_delivers_its_help(
        qtbot, qt_theme_applied):
    """Driven with the Enter event Qt sends, through the installed filter."""
    from spacr.qt.widgets import hover_tooltip as ht

    owner = _form(qtbot, keys=("cell_diameter",))
    install_api_tooltips(owner, "mask")
    label = _help_labels(owner)[0]

    shown = []
    original = ht.HoverTooltip.show_for
    ht.HoverTooltip.show_for = (
        lambda self, anchor, html: shown.append((anchor, html)))
    try:
        QApplication.sendEvent(label, QEvent(QEvent.Type.Enter))
        QApplication.processEvents()
    finally:
        ht.HoverTooltip.show_for = original

    assert len(shown) == 1
    anchor, html = shown[0]
    assert anchor is label
    assert "href=" in str(html)


# ---------------------------------------------------------------------------
# The parameter is gone, not merely defaulted off
# ---------------------------------------------------------------------------

def test_install_api_tooltips_has_no_dot_switch():
    """A flag with one value is a flag nobody reads."""
    parameters = inspect.signature(install_api_tooltips).parameters

    assert "api_dots" not in parameters
    assert list(parameters) == ["owner", "app_key", "widget_keys"]


def test_nothing_asks_for_dots_any_more():
    """The sweep that would catch the parameter creeping back."""
    offenders = []
    for path in sorted(QT_PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bapi_dots\b", text):
            offenders.append(str(path.relative_to(QT_PACKAGE)))
    assert offenders == [], (
        "these still name the removed dot switch: " + ", ".join(offenders))


def test_the_dot_builders_are_gone():
    """Nothing is left that could draw one."""
    from spacr.qt.screens import settings_model

    for name in ("build_setting_link_widget", "_add_api_dot_to_label",
                 "_add_api_dot_to_combined_control"):
        assert not hasattr(settings_model, name), (
            f"{name} still exists and can still draw a setting dot")
