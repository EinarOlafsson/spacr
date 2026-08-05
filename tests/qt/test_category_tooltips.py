"""Every settings CATEGORY explains itself, under the Run / Stop row.

A category is a collapsible header in a module's settings panel. There are
124 of them across the eighteen modules that render a settings form, drawn
from 81 distinct titles, and between them they group 563 settings. A user who
opens Mask for the first time meets thirteen headings before a single control,
so the headings have to carry their own weight.

Three things are pinned here:

* **Coverage.** Every (module, category) pair has a written blurb.
  Parametrised, so adding a category without one fails by name rather than
  quietly falling back to "Settings that control <title>."
* **Placement.** The blurb is rendered in a strip UNDER the actions row, not
  as a popup over the form it describes: hovering a header fills the strip,
  expanding one pins it there.
* **No duplication.** Re-gating the form -- the Mask live preview switching
  Primary object from cell to nucleus -- must not double anything. That
  regression had three causes (help dots carrying ``settingKey`` and being
  re-swept, ``_unwrap_setting_label`` handing back the wrapper host, and a
  missing ``removeEventFilter``), and it came back once already, so the exact
  user-facing scenario is tested rather than a synthetic panel.
"""
from __future__ import annotations

from html import unescape

import pytest

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QLabel

import spacr.settings as S
from spacr.qt.screens.app_screen import AppScreen, CATEGORY_STRIP_LINES
from spacr.qt.screens.settings_model import (
    CATEGORY_TOOLTIPS,
    CATEGORY_TOOLTIPS_BY_APP,
    _APP_HIDDEN_CATEGORIES,
    categories_for_app,
    category_tooltip,
    category_tooltip_is_curated,
    resolve_default_settings,
)


# Modules that render their own bespoke screen rather than the shared
# settings form. They have no categories, so nothing here applies to them.
#
# Power / Design and Run Compare are here for that reason and no other:
# both register a screen factory, so the shared form is never built for
# them. Power is the one that would otherwise show up, because it DOES
# register defaults -- its form is its settings, and the keys are recorded
# in spacr.settings under one "Power analysis" heading so that the macro
# recorder, the settings diff and the per-app inventory can see them. That
# heading is never drawn as a settings-panel section, so writing a section
# blurb for it would be writing help for a screen nobody opens.
#
# The rule this set encodes: an app belongs here when it has a factory. An
# app with a factory AND registered defaults, like Power, is the only kind
# that can reach the parametrisation below by accident.
CUSTOM_SCREENS = frozenset({
    "annotate", "make_masks", "queue", "db_browser", "agreement",
    "plate_view", "model_compare", "align", "convert", "foreign", "batch",
    "distributed_jobs", "model_zoo", "report", "train_compare",
    "classifier_evaluation", "run_history", "power", "run_compare",
})


def _settings_modules() -> list[str]:
    """Every app key whose screen is the shared settings form."""
    from spacr.qt.app import APPS

    keys = [row[0] for row in APPS if row[0] not in CUSTOM_SCREENS]
    # Reachable from the Tk GUI and the CLI, absent from the Qt home grid.
    keys.append("cellpose_all")
    return keys


def _rendered_categories(app_key: str) -> list[str]:
    """The category titles one module's settings panel actually renders.

    Mirrors ``SettingsWidgets.build_sections``: hidden categories are skipped,
    a key is claimed by the first category that lists it, and a category with
    no surviving key is not drawn at all.
    """
    defaults = resolve_default_settings(app_key)
    hidden = _APP_HIDDEN_CATEGORIES.get(app_key, set())
    used: set[str] = set()
    titles: list[str] = []
    for name, keys in categories_for_app(app_key, S.categories).items():
        if name in hidden:
            continue
        rows = [k for k in keys if k in defaults and k not in used]
        used.update(rows)
        if rows:
            titles.append(name)
    return titles


def _every_pair() -> list[tuple[str, str]]:
    return [(app_key, title)
            for app_key in _settings_modules()
            for title in _rendered_categories(app_key)]


PAIRS = _every_pair()


def _make_screen(qtbot, app_key: str) -> AppScreen:
    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    return screen


# ---------------------------------------------------------------------------
# 1. Coverage — every category of every module has written help
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key,title", PAIRS,
                         ids=[f"{a}:{t}" for a, t in PAIRS])
def test_every_category_of_every_module_has_a_tooltip(app_key, title):
    """The parametrised guard: a new category without a blurb fails by name."""
    text = category_tooltip(app_key, title)
    assert text, f"{app_key}/{title} renders with no category help at all"
    assert category_tooltip_is_curated(app_key, title), (
        f"{app_key}/{title} falls back to the generic sentence -- add an "
        f"entry to CATEGORY_TOOLTIPS[{title.upper().strip()!r}], or to "
        f"CATEGORY_TOOLTIPS_BY_APP[{app_key!r}] when it means something "
        f"different in this module"
    )


@pytest.mark.parametrize("app_key,title", PAIRS,
                         ids=[f"{a}:{t}" for a, t in PAIRS])
def test_category_help_says_more_than_the_heading(app_key, title):
    """A blurb that restates the title tells the reader nothing new."""
    text = category_tooltip(app_key, title)
    words = [w for w in title.replace("&", " ").split() if len(w) > 2]
    assert len(text.split()) >= 12, (
        f"{app_key}/{title}: {text!r} is too short to say what the group "
        f"decides and when you would change it")
    assert text.lower().strip() != f"settings that control {title.lower()}."
    # The heading may of course appear inside a real sentence; what it may not
    # be is the whole of it.
    stripped = text.lower()
    for word in words:
        stripped = stripped.replace(word.lower(), "")
    assert len(stripped.split()) >= 8, (
        f"{app_key}/{title}: {text!r} is mostly the heading again")


def test_no_module_renders_an_uncategorised_other_bucket():
    """"Other" is the absence of a heading, so it can have no blurb.

    It appears when a key falls out of the category map entirely. Catching it
    here keeps the two tests above honest -- otherwise a dropped key would
    show up as a missing tooltip and get "fixed" by writing one.
    """
    stray = {app_key: _rendered_categories(app_key)
             for app_key in _settings_modules()}
    offenders = {k: v for k, v in stray.items() if "Other" in v}
    assert not offenders, f"keys fell out of the category map: {offenders}"


def test_per_app_overrides_all_name_a_category_that_module_renders():
    """A dead override is help nobody will ever see."""
    dead = {}
    for app_key, overrides in CATEGORY_TOOLTIPS_BY_APP.items():
        rendered = {t.upper().strip() for t in _rendered_categories(app_key)}
        missing = sorted(set(overrides) - rendered)
        if missing:
            dead[app_key] = missing
    assert not dead, f"per-module category help that never renders: {dead}"


def test_overrides_actually_differ_from_the_shared_table():
    """An override that repeats the shared text is a maintenance trap."""
    same = [
        (app_key, title)
        for app_key, overrides in CATEGORY_TOOLTIPS_BY_APP.items()
        for title, text in overrides.items()
        if CATEGORY_TOOLTIPS.get(title) == text
    ]
    assert not same, f"overrides identical to the shared blurb: {same}"


def test_resolution_order_is_module_then_shared_then_generic():
    # Timelapse overrides "Runtime & Reliability"; Mask does not.
    assert (category_tooltip("timelapse", "Runtime & Reliability")
            != category_tooltip("mask", "Runtime & Reliability"))
    assert (category_tooltip("mask", "Runtime & Reliability")
            == CATEGORY_TOOLTIPS["RUNTIME & RELIABILITY"])
    # Unknown category: visible fallback, never an empty string.
    assert (category_tooltip("mask", "Nonexistent Group")
            == "Settings that control nonexistent group.")
    assert not category_tooltip_is_curated("mask", "Nonexistent Group")
    assert category_tooltip("mask", "") == ""
    # Case and padding are not part of the key.
    assert (category_tooltip("mask", "  cell segmentation  ")
            == CATEGORY_TOOLTIPS["CELL SEGMENTATION"])


# ---------------------------------------------------------------------------
# 2. Placement — the strip lives under the actions row
# ---------------------------------------------------------------------------

def _strip(screen) -> QLabel:
    return screen._category_hint


def _strip_text(screen) -> str:
    """The strip's text as the user reads it.

    The label is rich text -- the heading is bold -- so "Input & Metadata"
    is stored as ``Input &amp; Metadata``. Unescape before asserting, or half
    the category titles in the app never match.
    """
    return unescape(_strip(screen).text())


def test_the_strip_sits_under_the_actions_row(qtbot):
    screen = _make_screen(qtbot, "mask")
    layout = screen._actions_row.parentWidget().layout()
    positions = {layout.indexOf(w): w for w in
                 (screen._actions_row, _strip(screen))}
    assert -1 not in positions, "both belong to the runtime panel's layout"
    assert layout.indexOf(_strip(screen)) > layout.indexOf(screen._actions_row)
    assert _strip(screen).objectName() == "CategoryHintStrip"


def test_the_strip_reserves_a_fixed_height_so_run_stop_never_moves(qtbot):
    screen = _make_screen(qtbot, "mask")
    strip = _strip(screen)
    strip.ensurePolished()
    expected = strip.fontMetrics().lineSpacing() * CATEGORY_STRIP_LINES
    assert strip.height() == expected
    before = strip.height()
    screen.show_category_hint("Organelle Segmentation")   # the longest blurb
    assert strip.height() == before


def test_hovering_a_category_header_fills_the_strip(qtbot):
    screen = _make_screen(qtbot, "mask")
    strip = _strip(screen)
    default = strip.text()
    section = next(s for s in screen._settings_sections
                   if s.title() == "IMAGE PREPROCESSING")

    QApplication.sendEvent(section.header(), QEvent(QEvent.Type.Enter))
    QApplication.processEvents()
    shown = _strip_text(screen)
    assert shown != default
    assert "IMAGE PREPROCESSING" in shown
    assert "normalisation" in shown

    QApplication.sendEvent(section.header(), QEvent(QEvent.Type.Leave))
    QApplication.processEvents()
    assert strip.text() == default


def test_every_rendered_category_reaches_the_strip(qtbot):
    """Not just the one we happened to pick: all thirteen of Mask's."""
    screen = _make_screen(qtbot, "mask")
    strip = _strip(screen)
    for section in screen._settings_sections:
        QApplication.sendEvent(section.header(), QEvent(QEvent.Type.Enter))
        QApplication.processEvents()
        assert section.title() in _strip_text(screen)
        assert category_tooltip(screen.app_key, section.title()) in (
            _strip_text(screen))
        QApplication.sendEvent(section.header(), QEvent(QEvent.Type.Leave))


def test_expanding_a_category_pins_its_blurb(qtbot):
    """"Selected" outlives the pointer: the open category stays described."""
    screen = _make_screen(qtbot, "mask")
    strip = _strip(screen)
    section = next(s for s in screen._settings_sections
                   if s.title() == "QUALITY CONTROL")

    section.set_expanded(True)
    assert "QUALITY CONTROL" in _strip_text(screen)

    # Wander over another header and back off it -- the open one is restored,
    # not the placeholder.
    other = screen._settings_sections[0]
    QApplication.sendEvent(other.header(), QEvent(QEvent.Type.Enter))
    QApplication.processEvents()
    assert other.title() in _strip_text(screen)
    QApplication.sendEvent(other.header(), QEvent(QEvent.Type.Leave))
    QApplication.processEvents()
    assert "QUALITY CONTROL" in _strip_text(screen)

    section.set_expanded(False)
    assert strip.text() == screen._default_category_hint()


def test_the_category_strip_is_not_the_per_setting_strip(qtbot):
    """Two regions, two jobs: crossing a header must not blank the other."""
    screen = _make_screen(qtbot, "mask")
    labels = [w for w in screen._settings_content.findChildren(QLabel)
              if w.property("settingKey")]
    setting_label = labels[0]
    QApplication.sendEvent(setting_label, QEvent(QEvent.Type.Enter))
    QApplication.processEvents()
    setting_text = screen._hint_strip.text()
    assert setting_text != screen._default_hint()

    header = screen._settings_sections[0].header()
    QApplication.sendEvent(header, QEvent(QEvent.Type.Enter))
    QApplication.processEvents()
    assert screen._hint_strip.text() == setting_text, (
        "hovering a category header wiped the per-setting hint")
    assert screen._settings_sections[0].title() in _strip_text(screen)


@pytest.mark.parametrize("app_key", ["measure", "umap", "regression"])
def test_other_modules_get_the_same_region(qtbot, app_key):
    screen = _make_screen(qtbot, app_key)
    section = screen._settings_sections[0]
    QApplication.sendEvent(section.header(), QEvent(QEvent.Type.Enter))
    QApplication.processEvents()
    assert section.title() in _strip_text(screen)


# ---------------------------------------------------------------------------
# 3. No duplication — the regression that came back once already
# ---------------------------------------------------------------------------

def test_switching_primary_object_does_not_duplicate_setting_help(qtbot):
    """Cell -> nucleus in the Mask live preview, the reported scenario.

    ``LiveSettingsDialog.refresh_visibility`` re-runs the whole decoration
    pass, and it is wired to ``_object_box.currentTextChanged``. Before the
    fix every setting then carried two API dots and emitted two tooltips per
    hover.

    This used to count ``DotLink``, the base of every dot. The dialog now
    passes ``api_dots=False`` -- 68 dots down one form read as texture
    rather than as an affordance -- so the count is taken on the decorated
    labels themselves, which is what actually duplicated. ``DotLink`` is
    still counted, pinned at zero: that keeps the guard against the purple
    animation dot the user asked to have removed ever coming back, and
    against the API dots quietly returning to this dialog.
    """
    from PySide6.QtWidgets import QLabel

    from spacr.qt.widgets.dot_link import DotLink
    from spacr.qt.widgets.live_preview import LivePreviewPanel, LiveSettingsDialog

    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    dialog = LiveSettingsDialog(panel)
    qtbot.addWidget(dialog)

    def decorated():
        return len([w for w in dialog.findChildren(QLabel)
                    if w.property("settingHelpLabel") and w.toolTip()])

    before = decorated()
    assert before >= 1, "the popup should have been decorated at all"
    assert not dialog.findChildren(DotLink), (
        "this dialog draws no dots -- neither the teal API dot nor the "
        "purple animation dot")

    panel._object_box.setCurrentText("nucleus")
    QApplication.processEvents()
    assert decorated() == before

    # And it is not a one-shot guard: users flip this repeatedly.
    for choice in ("cell", "pathogen", "cell + nucleus", "nucleus", "cell"):
        panel._object_box.setCurrentText(choice)
        QApplication.processEvents()
    assert decorated() == before
    assert not dialog.findChildren(DotLink)


def test_switching_primary_object_emits_one_tooltip_per_hover(qtbot):
    """The user-visible half: one hover, one popup, after any number of
    re-gates.

    Counts what the filter DOES rather than how often Qt calls it, because Qt
    dispatches to the C++ side and a Python wrapper around ``eventFilter`` is
    never consulted.

    HONEST LIMITATION, measured rather than assumed: reverting the
    ``removeEventFilter`` line alone leaves this green, because Qt's own
    ``installEventFilter`` moves an already-installed filter to the front
    instead of appending a second copy, and every pass reuses the one filter
    cached on the owner. Reverting the *dot sweep* guard turns it red. The
    line is still worth keeping -- it states the intent and survives an owner
    whose filter is recreated -- but the tests that actually catch the
    reported regression are this file's dot-count one and
    ``test_only_one_help_filter_is_ever_created``.
    """
    from spacr.qt.widgets import hover_tooltip as ht
    from spacr.qt.widgets.live_preview import LivePreviewPanel, LiveSettingsDialog

    panel = LivePreviewPanel()
    qtbot.addWidget(panel)
    dialog = LiveSettingsDialog(panel)
    qtbot.addWidget(dialog)

    target = next(
        w for w in dialog.findChildren(QLabel)
        if w.property("settingHelpLabel") and w.property("settingKey"))

    def _hovers() -> int:
        shown = []
        original = ht.HoverTooltip.show_for
        ht.HoverTooltip.show_for = lambda self, anchor, html: shown.append(anchor)
        try:
            QApplication.sendEvent(target, QEvent(QEvent.Type.Enter))
            QApplication.processEvents()
        finally:
            ht.HoverTooltip.show_for = original
        return len(shown)

    baseline = _hovers()
    for choice in ("nucleus", "cell", "nucleus"):
        panel._object_box.setCurrentText(choice)
        QApplication.processEvents()
    assert _hovers() == baseline, (
        "switching Primary object doubled the setting tooltips again")


def test_help_dots_are_never_swept_as_settings(qtbot):
    """The first of the three causes, stated directly.

    The dots this pass creates carry ``settingKey`` themselves. Without the
    display-role guard the next sweep treats each dot as a setting and gives
    it its own pair of dots -- which is why the count grew by more than one
    per pass.
    """
    from spacr.qt.screens.settings_model import install_api_tooltips
    from spacr.qt.widgets.info_link import InfoLink

    from PySide6.QtWidgets import QFormLayout, QSpinBox, QWidget

    owner = QWidget()
    qtbot.addWidget(owner)
    form = QFormLayout(owner)
    field = QSpinBox()
    field.setProperty("settingKey", "cell_diameter")
    form.addRow(QLabel("Cell diameter"), field)

    install_api_tooltips(owner, "mask")
    first = len(owner.findChildren(InfoLink))
    for _ in range(4):
        install_api_tooltips(owner, "mask")
    assert len(owner.findChildren(InfoLink)) == first
    for dot in owner.findChildren(InfoLink):
        assert dot.property("apiTooltipDisplayRole") == "api-link"


def test_the_label_host_is_unwrapped_on_a_second_pass(qtbot):
    """The second cause, which the label cache hides in the live dialog.

    Pass one replaces the form's label with a ``SettingLabelWithInfo`` host
    holding ``[stretch][label][dot]``. ``QFormLayout.labelForField`` then
    hands back the HOST -- a fresh widget with none of the label's guard
    properties -- so pass two decorated it again and the row grew a second
    dot and a second tooltip.

    Asserted on ``_unwrap_setting_label`` directly because
    ``_setting_label_for_field`` remembers the label on the field and never
    reaches ``labelForField`` a second time in the live-preview flow. The
    guard is what keeps that cache from being the only thing standing between
    the user and the duplicate.
    """
    from PySide6.QtWidgets import QHBoxLayout, QWidget

    from spacr.qt.screens.settings_model import _unwrap_setting_label

    host = QWidget()
    qtbot.addWidget(host)
    host.setObjectName("SettingLabelWithInfo")
    row = QHBoxLayout(host)
    inner = QLabel("Cell diameter", host)
    inner.setProperty("settingHelpLabel", True)
    row.addWidget(inner)

    assert _unwrap_setting_label(host) is inner
    # Anything that is not a host is returned untouched, and None survives.
    plain = QLabel("plain")
    qtbot.addWidget(plain)
    assert _unwrap_setting_label(plain) is plain
    assert _unwrap_setting_label(None) is None


def test_only_one_help_filter_is_ever_created(qtbot):
    """A per-pass filter object would double every delivery for real.

    Qt collapses a repeated ``installEventFilter`` of the *same* filter, so
    the duplication only becomes observable once a second filter instance
    exists. The owner caches exactly one; this pins that.
    """
    from PySide6.QtWidgets import QFormLayout, QSpinBox, QWidget

    from spacr.qt.screens.settings_model import (
        _ApiTooltipFilter, install_api_tooltips,
    )

    owner = QWidget()
    qtbot.addWidget(owner)
    form = QFormLayout(owner)
    field = QSpinBox()
    field.setProperty("settingKey", "cell_diameter")
    form.addRow(QLabel("Cell diameter"), field)

    for _ in range(5):
        install_api_tooltips(owner, "mask")
    filters = [c for c in owner.children()
               if isinstance(c, _ApiTooltipFilter)]
    assert len(filters) == 1, f"{len(filters)} help filters on one panel"


def test_rewiring_category_hints_stays_at_one_delivery(qtbot):
    """The new path is held to the same rule as the old one.

    ``_wire_category_hints`` installs an event filter on every header. Qt
    keeps a LIST of filters, so a second install would write the strip twice
    per hover -- invisible in the text, but the same latent bug.
    """
    screen = _make_screen(qtbot, "mask")
    section = screen._settings_sections[0]

    calls = []
    original = AppScreen.show_category_hint
    AppScreen.show_category_hint = lambda self, title: calls.append(title)
    try:
        for _ in range(3):
            screen._wire_category_hints()
        QApplication.sendEvent(section.header(), QEvent(QEvent.Type.Enter))
        QApplication.processEvents()
    finally:
        AppScreen.show_category_hint = original

    assert len(calls) == 1, f"one hover wrote the strip {len(calls)} times"


def test_rewiring_does_not_multiply_the_toggled_connection(qtbot):
    """The other half of the same guard: `toggled` is connected once.

    Counted through ``show_category_hint`` rather than the handler itself --
    the handler is bound into a ``partial`` at connect time, so replacing the
    class attribute afterwards would never be consulted, and the test would
    pass against a doubly-connected signal.
    """
    screen = _make_screen(qtbot, "mask")
    section = screen._settings_sections[0]
    for _ in range(3):
        screen._wire_category_hints()

    calls = []
    original = AppScreen.show_category_hint
    AppScreen.show_category_hint = lambda self, title: calls.append(title)
    try:
        section.set_expanded(True)
        QApplication.processEvents()
    finally:
        AppScreen.show_category_hint = original
    assert len(calls) == 1, f"one expand fired {len(calls)} times"
