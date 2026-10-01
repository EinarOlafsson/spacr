"""`number_of_organelles` decides both the settings and their categories."""
from __future__ import annotations

import pytest

import spacr.qt.app as app_module
from spacr.qt.widgets.section import Section, _sections_below
from tests.qt.per_object_table import (grid_section, table_value,
                                       the_table_answers)


# Every nucleus row Mask builds at "All settings", nucleus_channel included.
# Measured 2026-09-26 (item 43/288): 13 before item 511 (ed845f885^), 9 after.
# 511 retired nucleus_{min,max}_{area,intensity} into object_filters rows,
# which is exactly the four that went; the same 9 are there after one and
# after two rebuilds, so no rebuild drops a row. This was "> 10" while those
# four were still on the form.
NUCLEUS_ROWS = 9


@pytest.fixture(scope="module")
def mask(qapp):
    win = app_module.MainWindow()
    win.resize(1200, 800)
    win.show()
    win._on_nav_selected("mask")
    qapp.processEvents()
    yield win._screens["mask"]
    win.close()
    win.deleteLater()


def _categories(screen):
    """Every category heading, including those inside a body that waits
    off the page until its category is opened."""
    return [str(s.property("settingsCategorySource"))
            for s in _sections_below(screen) if isinstance(s, Section)]


def test_the_count_defaults_to_none(mask):
    """A run has the organelles it says it has."""
    assert (mask._settings_model.collect() or {}).get(
        "number_of_organelles") == 0


def test_no_organelle_settings_at_zero(mask):
    """Four "Organelle N — Channel" rows were there whatever the count."""
    keys = [k for k in mask._settings_model._widgets
            if "organelle" in k.lower() and k != "number_of_organelles"]
    assert keys == [], f"{len(keys)} organelle settings on a form saying none"


def test_no_organelle_categories_at_zero(mask):
    """A category is not its contents: an empty heading reads as a section
    that failed to load rather than one that does not apply."""
    left = [name for name in _categories(mask) if "rganelle" in name]
    assert left == [], f"empty organelle categories survived: {left}"


def test_the_control_itself_survives(mask):
    """Or a run with none would have no way to ask for one."""
    assert "number_of_organelles" in mask._settings_model._widgets


def test_no_category_is_empty(mask):
    """Asked for 2026-08-28, for every category and not only organelles."""
    empty = []
    # 2026-09-29 (item 592): the "Per-object settings" heading holds the
    # per-object table as a prose row -- deliberately not a setting row, see
    # AppScreen._mount_the_object_grid -- so it is judged by the table.
    table = grid_section(mask)
    for section in _sections_below(mask):
        if not isinstance(section, Section):
            continue
        if section is table:
            assert mask._object_grid.objects(), "a table with no column"
            continue
        rows = getattr(section, "_row_widgets", None) or ()
        if any(w is not None for _label, w in rows):
            continue
        if any(any(w is not None for _l, w in
                   (getattr(child, "_row_widgets", None) or ()))
               for child in _sections_below(section)):
            continue
        empty.append(str(section.property("settingsCategorySource")))
    assert empty == [], f"headings over nothing: {empty}"


def test_raising_the_count_builds_the_slots(mask):
    """A control that was never built cannot be revealed, so it must grow.

    RELATIVE TO WHAT IS ALREADY BUILT, not the literal 3 this asked for
    until 2026-09-08. `mask` is module-scoped and growing never shrinks,
    so `test_growing_never_shrinks` -- which takes the same panel to five
    -- left this one asserting that a request for three yields three when
    five slots already existed. pytest-randomly orders the file, so the
    suite passed or failed on the seed: green on the seeds where this ran
    first, red on 2 and 7 among others, with no change to the package
    between them.

    The property is unchanged and is the one the docstring states: ask
    for more than is built and the panel builds up to it.
    """
    model = mask._settings_model
    target = max(3, model._slots_built_for + 1)
    assert model.grow_to_fit_the_organelle_count(target) == target
    assert model._slots_built_for == target


def test_growing_never_shrinks(mask):
    """A slot built once keeps whatever has since been put in it."""
    model = mask._settings_model
    model.grow_to_fit_the_organelle_count(5)
    before = model._slots_built_for
    model.grow_to_fit_the_organelle_count(1)
    assert model._slots_built_for == before


def test_an_old_settings_file_still_means_what_it_meant():
    """The default is none; a file carrying slots is not claiming none."""
    from spacr.organelle_types import organelle_count

    assert organelle_count({}) == 0
    assert organelle_count({"cell_channel": 1}) == 0
    # Written before the count existed, carrying four slots.
    old = {"organelle_channel": 1, "organelleb_channel": 2,
           "organellec_channel": 3, "organelled_channel": 0}
    assert organelle_count(old) == 4
    # An explicit count still wins over the inference.
    assert organelle_count({**old, "number_of_organelles": 2}) == 2
    # A key present but blank is a placeholder, not a slot in use.
    assert organelle_count({"organelle_channel": ""}) == 0


def test_an_object_with_no_channel_brings_no_settings(mask):
    """"do the same for the other object classes, except cell".

    THE PREMISE IS ESTABLISHED RATHER THAN ASSUMED, and it has to be.
    `test_the_rule_is_decided_once_and_not_while_typing` types into
    `nucleus_channel` six times and its last write is `5 % 3` -- it leaves
    a 2 there -- and the panel remembers a typed value across windows, so
    the NEXT test's freshly built screen opens with a nucleus channel and
    the rule correctly shows the nucleus settings.

    Measured: this file alone at `--randomly-seed=288` fails, and so does
    the three-test sequence
    `test_the_rule_is_decided_once_and_not_while_typing`,
    `test_no_category_is_empty`, this one -- 3.5 seconds. The panel's
    channel goes [None x6, '2' x8] as it settles. Nothing was wrong with
    the visibility rule, which is where four earlier hypotheses went.

    A test whose subject is "an object with NO channel" must make sure the
    object has none. Assuming it made this test a report on whatever the
    previous one typed.
    """
    model = mask._settings_model
    for role in ("nucleus", "pathogen"):
        widget = model._widgets.get(f"{role}_channel")
        if widget is None:
            continue
        if hasattr(widget, "setText"):
            widget.setText("")
        elif hasattr(widget, "setCurrentText"):
            widget.setCurrentText("")
    model.refresh_object_visibility()

    # HIDDEN, NOT ABSENT. This read `_widgets` and required the rows to be
    # missing, which was true while an unset object's keys were dropped from
    # the build. That is exactly what made 356's reveal impossible, so the
    # rows are built and the object rule hides them. Absence is the wrong
    # measure of "brings no settings"; not being on screen is the right one,
    # and it is what the user experiences either way.
    widgets = mask._settings_model._widgets
    for role in ("nucleus", "pathogen"):
        shown = [k for k in widgets
                 if k.startswith(f"{role}_") and k != f"{role}_channel"
                 and not widgets[k].isHidden()]
        assert shown == [], f"{role} brought {len(shown)} settings unasked"


def test_the_channel_itself_always_stays(mask):
    """Or there is no way to say the run has this object after all."""
    for role in ("cell", "nucleus", "pathogen"):
        assert f"{role}_channel" in mask._settings_model._widgets


def test_cell_is_never_gated(mask):
    """It is the object every other one is measured against."""
    cell = [k for k in mask._settings_model._widgets
            if k.startswith("cell_")]
    assert len(cell) > 5, (
        "cell settings were hidden; the form a user just opened is empty")


def test_the_rule_is_decided_once_and_not_while_typing(mask, qapp):
    """Re-running it per keystroke is what made the module hang."""
    import time

    model = mask._settings_model
    widget = model._widgets.get("nucleus_channel")
    if widget is None:
        pytest.skip("nucleus_channel is not on this panel")

    worst = 0.0
    for index in range(6):
        started = time.perf_counter()
        if hasattr(widget, "setText"):
            widget.setText(str(index % 3))
        else:
            widget.setValue(index % 3)
        qapp.processEvents()
        worst = max(worst, time.perf_counter() - started)
    assert worst < 0.20, f"{worst * 1000:.0f} ms a keystroke"


def test_a_committed_channel_brings_its_settings_back(qapp):
    """Hiding them was right; they have to come back when asked for.

    2026-09-29 (item 592): the per-object table is Mask generation's only
    layout of an object's settings, so "coming back" is the nucleus COLUMN
    of the table showing the committed channel, and no flat nucleus row is
    ever put back beside it -- the flat form this test used to count
    (``DEFAULT_OBJECT_GRID`` False) was removed on request. What it still
    holds from 356: a committed channel is answered WITHOUT reloading the
    module, and clearing it again is answered too.

    AT "ALL SETTINGS", the level that used to show every gated row, and with
    every category opened first: since 2026-09-22 a closed category's rows
    are not built until it is, and a row that does not exist is neither
    shown nor hidden.
    """
    from spacr.qt.settings_search import forget_disclosure, remember_disclosure

    remember_disclosure("mask", "all")
    win = app_module.MainWindow()
    win.show()
    win._on_nav_selected("mask")
    qapp.processEvents()
    try:
        screen = win._screens["mask"]
        screen._open_every_waiting_heading()
        qapp.processEvents()
        widgets = screen._settings_model._widgets
        nucleus = [k for k in widgets if k.startswith("nucleus_")]
        assert len(nucleus) >= NUCLEUS_ROWS, (
            f"only {len(nucleus)} nucleus rows built")
        # 2026-09-29 (item 592, "hide unset objects"): the nucleus channel
        # is the one nucleus row on the form -- it is what draws the
        # nucleus column -- and the column is not drawn until it is set.
        assert not widgets["nucleus_channel"].isHidden()
        nucleus = [k for k in nucleus if k != "nucleus_channel"]
        shown = [k for k in nucleus if not widgets[k].isHidden()]
        assert shown == [], shown
        assert "nucleus" not in screen._object_grid.objects()

        field = widgets["nucleus_channel"]
        field.setText("1")
        field.editingFinished.emit()
        qapp.processEvents()

        # NOT RELOADED. 356 measured the old behaviour at 455 ms and a
        # DIFFERENT screen object in the window's stack, taking every
        # uncommitted value, scroll position and expanded fold with it.
        assert win._screens["mask"] is screen, "the commit reloaded the module"

        assert "nucleus" in screen._object_grid.objects()
        assert all(the_table_answers(screen, k) for k in nucleus), [
            k for k in nucleus if not the_table_answers(screen, k)]
        shown = [k for k in nucleus if not widgets[k].isHidden()]
        assert shown == [], f"flat nucleus rows beside the table: {shown}"
        assert str((screen._settings_model.collect() or {}).get(
            "nucleus_channel")) == "1"

        # AND CLEARING IT IS ANSWERED TOO, or the toggle is one-way.
        field.setText("")
        field.editingFinished.emit()
        qapp.processEvents()
        assert "nucleus" not in screen._object_grid.objects()
        shown = [k for k in nucleus if not widgets[k].isHidden()]
        assert shown == [], shown
    finally:
        win.close()
        win.deleteLater()
        forget_disclosure("mask")


def test_a_raised_count_brings_the_organelle_rows_and_categories(qapp):
    win = app_module.MainWindow()
    win.show()
    win._on_nav_selected("mask")
    qapp.processEvents()
    try:
        screen = win._screens["mask"]
        assert _categories(screen).count("Organelle Segmentation") == 0

        count = screen._settings_model._widgets["number_of_organelles"]
        if hasattr(count, "setCurrentText"):
            count.setCurrentText("2")
        else:
            count.setValue(2)
        qapp.processEvents()

        screen = win._screens["mask"]
        categories = _categories(screen)
        assert "Organelle Segmentation" in categories
        assert "Organelle Segmentation (advanced)" in categories
        # One channel row per slot the count asked for, and no more.
        channels = [k for k in screen._settings_model._widgets
                    if k.endswith("_channel") and "organelle" in k]
        assert len(channels) == 2, channels
    finally:
        win.close()
        win.deleteLater()


def test_two_rebuilds_keep_what_the_first_one_set(qapp):
    """A second rebuild collected a nucleus channel of None and took the
    nucleus settings away again."""
    win = app_module.MainWindow()
    win.show()
    win._on_nav_selected("mask")
    qapp.processEvents()
    try:
        screen = win._screens["mask"]
        field = screen._settings_model._widgets["nucleus_channel"]
        field.setText("1")
        field.editingFinished.emit()
        qapp.processEvents()

        screen = win._screens["mask"]
        count = screen._settings_model._widgets["number_of_organelles"]
        if hasattr(count, "setCurrentText"):
            count.setCurrentText("2")
        else:
            count.setValue(2)
        qapp.processEvents()

        screen = win._screens["mask"]
        values = screen._settings_model.collect() or {}
        assert str(values.get("nucleus_channel")) == "1"
        assert len([k for k in screen._settings_model._widgets
                    if k.startswith("nucleus_")]) >= NUCLEUS_ROWS
    finally:
        win.close()
        win.deleteLater()


def test_the_rebuild_never_shows_the_home_screen(qapp):
    """Typing a channel value sent the user back to Home and returned them.

    Removing the old screen from the stack first drops the window to
    whatever is left showing; the replacement has to exist before the stack
    changes at all.
    """
    win = app_module.MainWindow()
    win.show()
    win._on_nav_selected("mask")
    qapp.processEvents()
    try:
        seen = []
        stack = win._stack
        stack.currentChanged.connect(
            lambda i: seen.append(type(stack.widget(i)).__name__))

        field = win._screens["mask"]._settings_model._widgets[
            "nucleus_channel"]
        field.setText("1")
        field.editingFinished.emit()
        qapp.processEvents()

        assert "StartupScreen" not in seen, seen
        assert all(name == "AppScreen" for name in seen), seen
        assert type(stack.currentWidget()).__name__ == "AppScreen"
    finally:
        win.close()
        win.deleteLater()


def test_the_rebuild_reports_no_error(qapp, caplog):
    """`SettingsWidgets` has no apply/set method; the values are carried by
    seeding the defaults the widgets are built from."""
    import logging

    win = app_module.MainWindow()
    win.show()
    win._on_nav_selected("mask")
    qapp.processEvents()
    try:
        with caplog.at_level(logging.ERROR):
            field = win._screens["mask"]._settings_model._widgets[
                "nucleus_channel"]
            field.setText("1")
            field.editingFinished.emit()
            qapp.processEvents()
        bad = [r for r in caplog.records if "mask screen" in r.getMessage()]
        assert bad == [], [r.getMessage() for r in bad]
    finally:
        win.close()
        win.deleteLater()
