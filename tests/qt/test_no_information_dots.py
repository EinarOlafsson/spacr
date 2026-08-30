"""No information dot is drawn, and every setting still links its API page.

"in the help menue remove the blue dots with an i."

They were the API-link dots ``install_api_tooltips`` put beside a setting
label. Three forms had already switched them off one at a time -- 68 of them
down the Mask live preview, twenty-six down the Annotate settings dialog,
three in the figure dialog -- each recording the same complaint: a column of
dots reads as texture rather than as one affordance per setting.

Removing them costs nothing because the API link was never in the dot alone.
What these tests hold:

* NO DOT IS DRAWN, on a hand-built form, on a real module screen, on any
  module's masthead or on any row of the profile dialog -- and nothing in
  the package constructs one.
* THE PARAMETER IS GONE, not merely defaulted off. A flag with one value is
  a flag nobody reads, and nothing passes it any more.
* EVERY SETTING THAT CARRIED A DOT STILL OFFERS ITS API LINK ON HOVER --
  which is the whole reason the dot is safe to drop, so it is asserted
  rather than assumed. The masthead's link is held to the same standard,
  including the two things that keep its destination current: the tab of a
  workbench serving two modules, and the language the documentation is
  read in.
"""
from __future__ import annotations

import inspect
import re
from html import escape
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
    """Every label carrying linked API setting help.

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


def test_generic_hover_help_stays_sticky_without_claiming_an_api_role(
        qtbot, qt_theme_applied):
    """Authored panel help uses the shared popup but is not API metadata."""
    from spacr.qt.widgets import hover_tooltip as ht

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    label = next(child for child in screen.findChildren(QLabel)
                 if child.text() == "View:")

    assert label.property("settingHelpLabel")
    assert label.property("apiTooltipDisplayRole") == "hover-help"
    assert not label.property("settingKey")
    assert "href=" not in str(label.property("apiTooltipHtml") or "")

    shown = []
    original = ht.HoverTooltip.show_for
    ht.HoverTooltip.show_for = (
        lambda self, anchor, html: shown.append((anchor, html)))
    try:
        QApplication.sendEvent(label, QEvent(QEvent.Type.Enter))
        QApplication.processEvents()
    finally:
        ht.HoverTooltip.show_for = original

    assert shown and shown[-1][0] is label


def test_measure_path_status_is_not_the_sample_limit_setting_label(
        qtbot, qt_theme_applied):
    """A stretching path placeholder cannot name the editor after it."""
    from spacr.qt.widgets.preview_controls import MAX_SETS_TOOLTIP

    screen = AppScreen("measure")
    qtbot.addWidget(screen)
    panel = screen._measure_preview

    assert panel._path_label.property("settingHelpLabel") is None
    assert panel._path_label.toolTip() == ""
    assert panel._path_label.property("apiTooltipHtml") is None
    assert panel._max_sets_box.toolTip() == MAX_SETS_TOOLTIP


def test_umap_objective_settings_gain_links_and_generic_help_does_not(
        qtbot, qt_theme_applied):
    """The three keyed controls and the surrounding authored help differ."""
    expected = {
        "umap_neighborhood_weight",
        "umap_stability_weight",
        "umap_cluster_structure_weight",
    }
    screen = AppScreen("umap")
    qtbot.addWidget(screen)
    labels = [child for child in screen.findChildren(QWidget)
              if child.property("settingHelpLabel")]
    objectives = {str(label.property("settingKey")): label
                  for label in labels
                  if str(label.property("settingKey") or "") in expected}

    assert set(objectives) == expected
    for label in objectives.values():
        assert label.property("settingsAppKey") == "umap"
        assert label.property("apiTooltipDisplayRole") == "tooltip"
        assert "href=" in str(label.property("apiTooltipHtml") or "")

    generic = [label for label in labels
               if label.property("apiTooltipDisplayRole") == "hover-help"]
    assert generic
    assert all(not (label.property("settingsAppKey")
                    and label.property("settingKey"))
               for label in generic)
    assert all("href=" not in str(label.property("apiTooltipHtml") or "")
               for label in generic)


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


# ---------------------------------------------------------------------------
# The last two dots: the module masthead and the profile row
# ---------------------------------------------------------------------------
# "i like simplisity". Two dots outlived the settings sweep above -- one on
# every module's masthead, beside its one-line description, and one beside
# every row of the distributed-execution profile dialog. They go the same
# way and for the same reason, and the link goes the same place: into the
# hover help, ending in the ``Open spaCR API documentation`` line
# `format_tooltip` already puts on every setting.
#
# The masthead dot was not decoration. Two callers kept its destination
# current -- the Cellpose workbench, whose one masthead serves two modules,
# and the language pass, because the API pages are per language -- so what
# carries the help has to be findable and repointable, not a label with a
# frozen string. Both are measured below.


def _mastheads():
    """One masthead per registry row, plus its key and description."""
    from spacr.qt.app import APPS
    from spacr.qt.screens.app_screen import ModuleHeader

    for app_key, title, description, *_rest in APPS:
        yield app_key, description, ModuleHeader(
            title, description, "Configure settings, then press Run.",
            app_key=app_key)


def test_no_module_masthead_draws_a_dot(qtbot, qt_theme_applied):
    """Every module, not the three that happen to have a screen test."""
    offenders = []
    for app_key, _description, header in _mastheads():
        qtbot.addWidget(header)
        if _api_links(header):
            offenders.append(app_key)
    assert offenders == [], (
        f"{len(offenders)} mastheads still draw a dot: "
        + ", ".join(offenders))


def test_every_masthead_still_reaches_its_own_api_page(
        qtbot, qt_theme_applied):
    """The reason the dot is safe to drop, measured module by module."""
    from spacr.qt.app import APPS
    from spacr.qt.screens.settings_model import api_docs_url

    checked = 0
    for app_key, description, header in _mastheads():
        qtbot.addWidget(header)
        help_label = header.api_help
        assert help_label is not None, f"{app_key}: masthead has no API help"
        html = help_label.help_html()
        # Escaped, because the help is rich text: an apostrophe in a
        # description reaches the reader as `&#x27;` and renders correctly.
        assert escape(description) in html, (
            f"{app_key}: its description is not in its hover help")
        assert help_label.url() == api_docs_url(app_key), (
            f"{app_key}: its help links to {help_label.url()}")
        checked += 1
    assert checked == len(APPS) > 20, (
        f"{checked} of {len(APPS)} module mastheads were checked")


def test_hovering_a_masthead_delivers_the_description_and_the_link(
        qtbot, qt_theme_applied):
    """Driven with the Enter event Qt sends, read out of the popup shown.

    The property being right is not the claim. The claim is that a reader
    who hovers the sentence gets the sentence and a way to the API page.
    """
    from spacr.qt.screens.app_screen import APP_INTROS, ModuleHeader
    from spacr.qt.screens.settings_model import api_docs_url
    from spacr.qt.widgets.hover_tooltip import HoverTooltip

    header = ModuleHeader("Mask Generation", APP_INTROS["mask"],
                          app_key="mask")
    qtbot.addWidget(header)
    header.show()
    qtbot.waitExposed(header)
    tooltip = HoverTooltip.instance()

    QApplication.sendEvent(header.api_help, QEvent(QEvent.Type.Enter))
    try:
        assert tooltip.isVisible()
        assert APP_INTROS["mask"] in tooltip._label.text()
        assert tooltip.api_url() == api_docs_url("mask")
    finally:
        QApplication.sendEvent(header.api_help, QEvent(QEvent.Type.Leave))


def test_a_masthead_hover_delivers_exactly_one_popup(qtbot, qt_theme_applied):
    """One filter per label. Qt keeps a LIST of them, and a second install
    would show two popups for one hover -- invisible in the text, and the
    bug that had the settings forms popping help twice."""
    from spacr.qt.screens.app_screen import APP_INTROS, ModuleHeader
    from spacr.qt.widgets import hover_tooltip as ht

    header = ModuleHeader("Mask Generation", APP_INTROS["mask"],
                          app_key="mask")
    qtbot.addWidget(header)
    # Repointed twice on the way in, as a workbench tab change does.
    header.api_help.set_api_app_key("measure")
    header.api_help.set_api_app_key("mask")

    shown = []
    original = ht.HoverTooltip.show_for
    ht.HoverTooltip.show_for = (
        lambda self, anchor, html: shown.append((anchor, html)))
    try:
        QApplication.sendEvent(header.api_help, QEvent(QEvent.Type.Enter))
        QApplication.processEvents()
    finally:
        ht.HoverTooltip.show_for = original

    assert len(shown) == 1, f"{len(shown)} popups for one hover"
    anchor, html = shown[0]
    assert anchor is header.api_help
    assert "href=" in str(html)


def test_a_masthead_with_no_app_key_offers_help_without_an_empty_link(
        qtbot, qt_theme_applied):
    """A module page to link to is what makes a link worth showing.

    The formatter falls back to the documentation index when it is given no
    app key, which is a link that answers no question the reader asked.
    """
    from spacr.qt.screens.app_screen import ModuleHeader

    header = ModuleHeader("Somewhere New", "A line about a module.")
    qtbot.addWidget(header)

    assert header.api_help is None
    blurb = header.description_label
    assert blurb is not None
    assert "A line about a module." in blurb.help_html()
    assert "<a " not in blurb.help_html(), (
        f"a masthead with no app key still offers {blurb.url()!r}")
    assert blurb.url() == ""


def test_a_language_change_repoints_a_masthead_link(qtbot, qt_theme_applied):
    """The API pages are per language; the help has to follow the choice.

    The language pass finds the help by the ``moduleApiAppKey`` property the
    dot used to carry, and repoints it with ``set_url`` -- so the property
    and the method are both part of the contract, not implementation.
    """
    from spacr.qt.i18n import retranslate_widget_tree
    from spacr.qt.screens.app_screen import APP_INTROS, ModuleHeader
    from spacr.qt.screens.settings_model import api_docs_url

    header = ModuleHeader("Mask Generation", APP_INTROS["mask"],
                          app_key="mask")
    qtbot.addWidget(header)
    help_label = header.api_help
    assert str(help_label.property("moduleApiAppKey")) == "mask"
    assert callable(getattr(help_label, "set_url", None))

    try:
        retranslate_widget_tree(header, "sv")
        assert help_label.url() == api_docs_url("mask", language="sv")
        assert "?lang=sv" in help_label.help_html()
    finally:
        retranslate_widget_tree(header, "en")
    assert help_label.url() == api_docs_url("mask")


def test_the_masthead_help_can_be_sent_to_another_url(qtbot,
                                                     qt_theme_applied):
    """``set_url`` is the contract the language pass calls, so it is pinned.

    Rewriting the destination must not cost the prose: the help is one
    string holding both, and a repointing that rebuilt only the link would
    hand the reader a bare URL.
    """
    from spacr.qt.screens.app_screen import APP_INTROS, ModuleHeader

    header = ModuleHeader("Mask Generation", APP_INTROS["mask"],
                          app_key="mask")
    qtbot.addWidget(header)

    header.api_help.set_url("https://example.test/elsewhere/index.html")

    assert header.api_help.url() == "https://example.test/elsewhere/index.html"
    html = header.api_help.help_html()
    assert "https://example.test/elsewhere/index.html" in html
    assert escape(APP_INTROS["mask"]) in html


def test_no_profile_row_draws_a_dot(qtbot, qt_theme_applied):
    """The distributed-execution profile dialog, row by row."""
    from PySide6.QtWidgets import QLabel

    from spacr.qt.screens.distributed_jobs import ExecutionProfileDialog

    dialog = ExecutionProfileDialog()
    qtbot.addWidget(dialog)

    assert _api_links(dialog) == []
    labels = dialog.findChildren(QLabel, "SettingsLabel")
    assert labels, "the profile dialog built no setting labels at all"
    without = [label.text() for label in labels
               if "href=" not in str(label.property("apiTooltipHtml") or "")]
    assert without == [], (
        "these profile rows lost their API link: " + ", ".join(without))


def test_nothing_builds_an_information_dot_any_more():
    """The sweep that would catch one being drawn again.

    ``InfoLink`` itself is left in place -- it is what the tests above look
    for when they check that no dot exists -- but nothing constructs one.
    """
    offenders = []
    for path in sorted(QT_PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.name in ("info_link.py", "dot_link.py", "__init__.py"):
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bInfoLink\s*\(", text):
            offenders.append(str(path.relative_to(QT_PACKAGE)))
    assert offenders == [], (
        "these draw an information dot: " + ", ".join(offenders))
