"""`number_of_organelles` decides both the settings and their categories."""
from __future__ import annotations

import pytest

import spacr.qt.app as app_module
from spacr.qt.widgets.section import Section


@pytest.fixture(scope="module")
def mask(qapp):
    win = app_module.MainWindow()
    win.resize(1200, 800)
    win.show()
    win._on_nav_selected("mask")
    qapp.processEvents()
    yield win._screens["mask"]
    win.close()


def _categories(screen):
    return [str(s.property("settingsCategorySource"))
            for s in screen.findChildren(Section)]


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
    for section in mask.findChildren(Section):
        rows = getattr(section, "_row_widgets", None) or ()
        if any(w is not None for _label, w in rows):
            continue
        if any(any(w is not None for _l, w in
                   (getattr(child, "_row_widgets", None) or ()))
               for child in section.findChildren(Section)):
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

    THE FLAT FORM, deliberately: `DEFAULT_OBJECT_GRID` is False, so these
    rows are the interface most users get, and this is the case that was
    broken. This test was xfailed on 2026-09-08 because two of the
    maintainer's instructions cancelled each other -- 356 asks a committed
    channel to reveal its object's settings WITHOUT reloading the module,
    and the fix for "i saw the object settings eaven when object channels
    were all none" kept those rows out of the build, so there was nothing to
    reveal.

    Both hold now. The rows are built and hidden; the search strip re-asks
    the object rule through `rehide_the_rows_the_run_has_no_object_for`, so
    "All settings" cannot put an absent object back; and the first pass runs
    synchronously at the end of the panel build rather than on a zero-delay
    timer, which is what left a freshly built panel showing every gated row
    to anyone who looked before the event loop turned.
    """
    win = app_module.MainWindow()
    win.show()
    win._on_nav_selected("mask")
    qapp.processEvents()
    try:
        screen = win._screens["mask"]
        widgets = screen._settings_model._widgets
        nucleus = [k for k in widgets if k.startswith("nucleus_")]
        assert len(nucleus) > 10, f"only {len(nucleus)} nucleus rows built"
        shown = [k for k in nucleus if not widgets[k].isHidden()]
        assert shown == ["nucleus_channel"], shown

        field = widgets["nucleus_channel"]
        field.setText("1")
        field.editingFinished.emit()
        qapp.processEvents()

        # NOT RELOADED. 356 measured the old behaviour at 455 ms and a
        # DIFFERENT screen object in the window's stack, taking every
        # uncommitted value, scroll position and expanded fold with it.
        assert win._screens["mask"] is screen, "the commit reloaded the module"

        shown = [k for k in nucleus if not widgets[k].isHidden()]
        assert len(shown) > 10, f"only {len(shown)} nucleus settings came back"
        categories = _categories(screen)
        assert any("Nucleus" in c for c in categories), categories
        assert str((screen._settings_model.collect() or {}).get(
            "nucleus_channel")) == "1"

        # AND CLEARING IT PUTS THEM BACK AWAY, or the toggle is one-way and a
        # mistyped channel leaves the form permanently wider.
        field.setText("")
        field.editingFinished.emit()
        qapp.processEvents()
        shown = [k for k in nucleus if not widgets[k].isHidden()]
        assert shown == ["nucleus_channel"], shown
    finally:
        win.close()


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
                    if k.startswith("nucleus_")]) > 10
    finally:
        win.close()


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
