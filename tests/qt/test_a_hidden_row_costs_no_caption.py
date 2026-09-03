"""The Mask panel captioning every row it has to show a fraction of them.

The panel builds a control for every organelle slot that CAN be named,
because a control that was never built cannot be revealed however the count
is driven. The object rule then hides the great majority of them before the
panel is painted -- a run segments a handful of objects, not every slot that
has a name.

Each of those hidden rows was still given a caption and the host widget that
right-aligns it against the field: two widgets and two style repolishes for a
caption nobody can read, on a panel where a style recalculation walks every
widget alive.

WHAT DID NOT CHANGE IS THE ROW. It is still on the form, still hideable,
still findable by everything that walks the form -- the settings search
indexes it, `setting_row_is_visible` answers for it, the object rule reaches
it exactly as before. Only the CAPTION waits, and the row is given one, in
place, the moment the rule says the run has that object after all.

Every assertion below is made by BUILDING THE REAL SCREEN and reading the
real form, because the whole risk of deferring anything is that something
stops checking without saying so.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _screen(qtbot, app_key: str = "mask", current=None):
    from spacr.qt.screens.app_screen import AppScreen

    before = AppScreen.values_the_next_screen_is_built_for
    AppScreen.values_the_next_screen_is_built_for = current
    try:
        screen = AppScreen(app_key)
    finally:
        AppScreen.values_the_next_screen_is_built_for = before
    qtbot.addWidget(screen)
    qtbot.wait(1)
    return screen, screen._settings_model


def _form_rows(screen) -> dict:
    """``id(field) -> (section, form, row index, label widget or None)``."""
    from PySide6.QtWidgets import QFormLayout

    found = {}
    for section in screen._settings_sections:
        form = getattr(section, "_form", None)
        if not isinstance(form, QFormLayout):
            continue
        for index in range(form.rowCount()):
            item = form.itemAt(index, QFormLayout.FieldRole)
            field = item.widget() if item is not None else None
            if field is None:
                continue
            label = form.itemAt(index, QFormLayout.LabelRole)
            found[id(field)] = (section, form, index,
                                label.widget() if label is not None else None)
    return found


def _captioned(screen) -> set:
    """The setting keys whose row has a caption beside it.

    WALKS UP FROM THE FIELD. The widget the FORM knows is not always the
    widget the model holds: a setting that takes a Cellpose checkpoint sits
    in a little holder beside its "Model zoo…" button, and the form's row is
    the holder. Matching on the field alone reported `cell_model_name` as
    having no caption when it has one, which is the same confusion the
    panel's own `_set_row_visible` walks up to avoid.
    """
    rows = _form_rows(screen)
    found = set()
    for key, field in screen._settings_model._widgets.items():
        node = field
        while node is not None:
            row = rows.get(id(node))
            if row is not None:
                if row[3] is not None:
                    found.add(key)
                break
            node = node.parentWidget()
    return found


# ---------------------------------------------------------------------------
# What is saved
# ---------------------------------------------------------------------------

def test_only_the_rows_the_run_has_an_object_for_are_captioned(qtbot):
    """One caption per row the user can read, not one per row that exists."""
    screen, model = _screen(qtbot)
    hidden = set(model.keys_whose_object_the_run_lacks())
    assert hidden, "Mask hides no object rows at all; this panel changed"

    captioned = _captioned(screen)
    assert not (captioned & hidden), sorted(captioned & hidden)[:5]
    assert captioned == set(model._widgets) - hidden


def test_the_panel_builds_one_caption_per_row_it_can_show(qtbot):
    """The caption and the host that right-aligns it are what is not built.

    Counted off the panel rather than off a number that would rot: every
    caption sits in a ``SettingLabelWithInfo`` host of its own, so the hosts
    ARE the captions, and there is one per row the run has an object for.
    """
    from PySide6.QtWidgets import QWidget

    screen, model = _screen(qtbot)
    hidden = model.keys_whose_object_the_run_lacks()
    shown = set(model._widgets) - set(hidden)
    hosts = [widget for widget in screen.findChildren(QWidget)
             if widget.objectName() == "SettingLabelWithInfo"]

    assert len(hosts) == len(shown), (len(hosts), len(shown))
    # Which is the saving: two widgets and two style repolishes each for the
    # rows the run cannot show, on a panel where a style recalculation walks
    # every widget alive.
    assert len(hosts) < len(model._widgets)


# ---------------------------------------------------------------------------
# What did not change
# ---------------------------------------------------------------------------

def test_a_waiting_row_is_still_a_row_on_the_form(qtbot):
    """Hidden, not absent -- which is what the rest of the panel goes on."""
    screen, model = _screen(qtbot)
    hidden = sorted(model.keys_whose_object_the_run_lacks())
    rows = _form_rows(screen)

    for key in hidden[:40]:
        field = model._widgets[key]
        assert id(field) in rows, f"{key} is on no form at all"
        section, form, index, label = rows[id(field)]
        assert not form.isRowVisible(index), key
        assert label is None, f"{key} was captioned while it is hidden"
        assert screen.setting_row_is_visible(key) is False


def test_the_search_strip_still_indexes_every_setting(qtbot):
    """The strip indexes the RENDERED FORM, so an unrendered row is one it
    cannot show. Every row is rendered; only the captions wait."""
    from spacr.qt.settings_search import SettingsSearchBar

    screen, model = _screen(qtbot)
    bar = SettingsSearchBar(screen)
    qtbot.addWidget(bar)
    assert set(bar.indexed_keys()) == set(model._widgets)


def test_reading_a_headings_rows_back_captions_them_all(qtbot):
    """Several checks walk ``Section._row_widgets``; each must get every row.

    And asking must not put a setting for an absent object on screen.
    """
    from spacr.qt.widgets.section import Section

    screen, model = _screen(qtbot)
    hidden = set(model.keys_whose_object_the_run_lacks())
    heading = max(screen.findChildren(Section),
                  key=lambda s: len(getattr(s, "_spacr_declared_rows", ())))
    declared = [key for key, _label, _widget
                in heading._spacr_declared_rows if key]
    assert len(declared) > 1

    from PySide6.QtWidgets import QLabel

    rows = list(heading._row_widgets)  # noqa: F841 - read back deliberately
    assert len(rows) == len(heading._spacr_declared_rows)
    for label, field in rows:
        assert isinstance(label, QLabel)
        assert label.property("settingKey")
    assert not any(screen.setting_row_is_visible(key)
                   for key in declared if key in hidden)


# ---------------------------------------------------------------------------
# What the rule reveals
# ---------------------------------------------------------------------------

def test_a_shape_built_for_seven_captions_the_seven_slots(qtbot):
    """A committed count rebuilds the optimized panel at its new shape."""
    from spacr.organelle_types import ALL_ORGANELLE_ROLES

    screen, _model = _screen(
        qtbot, current={"number_of_organelles": 7})
    shown = [role for role in ALL_ORGANELLE_ROLES
             if screen.setting_row_is_visible(f"{role}_channel")]
    assert len(shown) == 7, shown
    for role in shown:
        key = f"{role}_channel"
        assert key in _captioned(screen), f"{key} was shown with no caption"


def test_a_revealed_row_keeps_the_place_it_was_declared_in(qtbot):
    """``Section.add_row`` appends; a revealed setting belongs where the
    module wrote it, not underneath the sub-headings."""
    screen, model = _screen(qtbot)
    combo = model._widgets["number_of_organelles"]
    combo.setCurrentIndex(combo.findData(7))
    qtbot.wait(1)

    rows = _form_rows(screen)
    for section in screen._settings_sections:
        declared = [key for key, _label, widget
                    in getattr(section, "_spacr_declared_rows", ())
                    if key and id(widget) in rows
                    and rows[id(widget)][0] is section]
        places = [rows[id(model._widgets[key])][2] for key in declared]
        assert places == sorted(places), (section.title(), declared[:6])


def test_a_revealed_row_carries_the_help_its_caption_holds(qtbot):
    """The caption is the hover target for a setting's documentation, so a
    row that got one late has to get the whole of it."""
    screen, model = _screen(qtbot)
    key = "remove_background_nucleus"
    assert key not in _captioned(screen)
    model._widgets["nucleus_channel"].setText("1")
    model.refresh_object_visibility()
    qtbot.wait(1)

    from PySide6.QtWidgets import QLabel

    section, _form, _index, host = _form_rows(screen)[id(model._widgets[key])]
    assert host is not None
    assert not section.property("settingsSectionDiscarded")
    assert section in screen.rendered_settings_sections()
    # The label sits in the host that right-aligns it against the field.
    caption = host if isinstance(host, QLabel) else host.findChild(QLabel)
    assert caption is not None
    assert caption.property("settingKey") == key
    assert caption.property("settingsAppKey") == "mask"
    assert caption.property("apiTooltipHtml")
    assert screen._hint_map.get(caption)


# ---------------------------------------------------------------------------
# A caption that arrives after the panel does
# ---------------------------------------------------------------------------

def test_a_late_caption_is_not_left_in_english(qtbot, monkeypatch):
    """The language pass runs once, when the panel is built.

    A caption written after it would sit in English inside a translated
    window -- and worse, the pass reads a caption it did not render as that
    widget's English source and opts it out of every later pass.
    """
    from PySide6.QtWidgets import QLabel

    from spacr.qt.i18n import retranslate_widget_tree, tr

    monkeypatch.setenv("SPACR_LANGUAGE", "sv")
    screen, model = _screen(qtbot)
    retranslate_widget_tree(screen)

    key = "remove_background_nucleus"
    model._widgets["nucleus_channel"].setText("1")
    model.refresh_object_visibility()
    qtbot.wait(1)

    host = _form_rows(screen)[id(model._widgets[key])][3]
    caption = host if isinstance(host, QLabel) else host.findChild(QLabel)
    assert caption is not None
    # Setting captions have a direct key-level catalog entry; translating a
    # prettified fallback word by word is weaker and can produce a different
    # but still partly translated phrase.
    translated = tr(key, "sv")
    if translated != key:
        assert caption.text() == translated, caption.text()


def test_walking_the_caption_index_hands_over_every_caption(qtbot):
    """``_hint_map`` is what the panel's captions are checked through."""
    screen, model = _screen(qtbot)
    named = {label.property("settingKey") for label in screen._hint_map}
    assert set(model._widgets) <= named
    assert len(screen._hint_map) == len(model._widgets)


def test_looking_a_hint_up_builds_nothing(qtbot):
    """A pointer crossing the panel asks this several times a second."""
    screen, model = _screen(qtbot)
    waiting = dict(screen._rows_awaiting_layout)
    assert waiting
    assert screen._hint_map.get(screen) is None
    assert screen._hint_map.get(model._widgets["src"]) is None
    assert dict(screen._rows_awaiting_layout) == waiting
