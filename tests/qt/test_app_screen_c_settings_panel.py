"""The settings panel's headings: waiting rows, empty categories, rebuilds.

A heading that holds nothing is parked rather than shown, a parked heading is
put back at the position it was declared in, a row the object rule hides is
built only when something reads it, and the form is rebuilt in place (or the
rebuild deferred while a run owns the screen). A pruned heading and a heading
that failed to build look identical from outside, so every prune case also
asserts the rows are still reachable.

These cases rebuild panels and register rows, so each builds its own screen
through ``make_screen``.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QFormLayout, QLineEdit             # noqa: E402

from spacr.qt.screens import app_screen as aps                   # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widgets.section import Section                     # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def make_screen(qtbot):
    """Build a fresh regression screen per call; all are retired at the end."""
    built = []

    def make():
        widget = AppScreen("regression")
        built.append(widget)
        return widget

    yield make
    for widget in built:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


def _a_heading_with_waiting_rows(screen, plain=False):
    """A heading whose first row the object rule has decided to hide.

    Built through ``_build_settings_section``, the only thing that makes a
    waiting row, so ``_rows_awaiting_layout`` and the declared-row record are
    left exactly as a panel build leaves them. ``plain`` adds a visible row
    declared after it.
    """
    ghost = QLineEdit()
    screen._settings_model._widgets["ghost_setting"] = ghost
    rows = [("Ghost setting", ghost)]
    if plain:
        other = QLineEdit()
        screen._settings_model._widgets["plain_setting"] = other
        rows.append(("Plain setting", other))
    screen._widget_key_stamp = None
    screen._run_has_no_object_for = {"ghost_setting"}
    section = screen._build_settings_section(("Ghost", rows))
    return section, ghost, (rows[1][1] if plain else None)


def _hollow(section):
    """Strip a built heading of everything ``_section_holds_anything`` reads."""
    for member in (section, *section.findChildren(Section)):
        member.__dict__["_row_widgets"] = []
        member._spacr_declared_rows = ()


def _rebuild_with_hollow_headings(screen, hollow_titles):
    """Lay the settings panel out again with ``hollow_titles`` left empty."""
    build = screen._build_settings_section

    def build_but_empty_some(spec, depth: int = 0):
        section = build(spec, depth)
        if depth == 0 and section.title() in hollow_titles:
            _hollow(section)
        return section

    screen._build_settings_section = build_but_empty_some
    try:
        screen._lay_out_the_settings_panel().setParent(screen)
    finally:
        del screen._build_settings_section


def test_a_waiting_row_is_laid_out_whole_when_something_needs_it(
        make_screen, monkeypatch, subtests):
    """A hidden row is built lazily, and always as a complete labelled row.

    Invariant: indexing a heading's rows builds the waiting row (label and
    field) and clears it from ``_rows_awaiting_layout``; captioning every
    waiting row moves the rows (judging them) once, and a second pass with
    nothing waiting leaves the panel alone; a parked heading comes back with
    its rows even with no hint strip to rewire; a late caption goes back in
    its declared place with its label; the late-caption language pass
    retranslates the heading and forgets it, even when the heading was
    destroyed first.
    """
    from spacr.qt import i18n

    with subtests.test(case="indexing builds the waiting row"):
        screen = make_screen()
        section, field, _ = _a_heading_with_waiting_rows(screen)
        rows = section.__dict__["_row_widgets"]
        assert isinstance(rows, aps._RowsBuiltWhenTheyAreAskedFor)
        assert screen._rows_awaiting_layout == {"ghost_setting": section}
        first = rows[0]
        assert first[1] is field
        assert first[0].text() == "Ghost setting"
        assert screen._rows_awaiting_layout == {}

    with subtests.test(case="captioning every waiting row, once"):
        screen = make_screen()
        moved = []
        screen._the_rows_moved = lambda **kwargs: moved.append(kwargs)
        _section, field, _ = _a_heading_with_waiting_rows(screen)
        screen._caption_every_waiting_row()
        assert moved == [{"judge_them": True}]
        assert field._spacr_setting_label.text() == "Ghost setting"
        screen._caption_every_waiting_row()
        assert moved == [{"judge_them": True}]

    with subtests.test(case="restored heading with no hint strip"):
        screen = make_screen()
        section, field, _ = _a_heading_with_waiting_rows(screen)
        screen._discard_settings_section(section)
        assert section.property("settingsSectionDiscarded") is True
        screen._category_hint = None
        screen._lay_out_the_rows_that_are_back(())
        assert section.property("settingsSectionDiscarded") is False
        assert screen._rows_awaiting_layout == {}
        assert field._spacr_setting_label.text() == "Ghost setting"

    with subtests.test(case="late caption keeps its place and label"):
        screen = make_screen()
        section, ghost, plain = _a_heading_with_waiting_rows(screen,
                                                             plain=True)
        form = section._form
        assert form.getWidgetPosition(ghost)[0] == 0
        assert form.getWidgetPosition(plain)[0] == 1
        screen._lay_out_one_waiting_row("ghost_setting")
        assert form.getWidgetPosition(ghost)[0] == 0
        assert form.getWidgetPosition(plain)[0] == 1
        label_item = form.itemAt(0, QFormLayout.LabelRole)
        assert label_item is not None
        assert ghost._spacr_setting_label.text() == "Ghost setting"
        assert ghost._spacr_setting_label.parent() is label_item.widget()

    with subtests.test(case="late caption language pass"), \
            monkeypatch.context() as m:
        screen = make_screen()
        section, _field, _ = _a_heading_with_waiting_rows(screen)
        seen = []
        m.setattr(i18n, "retranslate_widget_tree", seen.append)
        screen._captioned_late = {section}
        screen._the_rows_moved(judge_them=False)
        assert seen == [section]
        assert screen._captioned_late == set()

        def gone(_widget):
            raise RuntimeError("Internal C++ object already deleted.")

        m.setattr(i18n, "retranslate_widget_tree", gone)
        screen._captioned_late = {section}
        screen._the_rows_moved(judge_them=False)
        assert screen._captioned_late == set()


def test_an_empty_category_is_parked_and_put_back_in_order(
        make_screen, subtests):
    """A heading holding nothing leaves the panel but not the form.

    Invariant: a hollow heading is not rendered, the others keep their
    order, and the parked heading stays registered, owned by the discard
    host, hidden and marked; putting it back answers True, clears the mark
    and places it by declaration order -- above the categories declared
    after it, above the stretch when it is the last, at the end when the
    column has no stretch; a heading that was never parked is not moved.
    """
    def parked_screen(which):
        screen = make_screen()
        titles = [s.title() for s in screen.rendered_settings_sections()]
        assert len(titles) > 2
        emptied = titles[which]
        _rebuild_with_hollow_headings(screen, {emptied})
        parked = next(s for s in screen._settings_sections
                      if s.title() == emptied)
        return screen, titles, emptied, parked, \
            screen._settings_content.layout()

    with subtests.test(case="parked, not shown"):
        screen, titles, emptied, parked, layout = parked_screen(0)
        rendered = [s.title() for s in screen.rendered_settings_sections()]
        assert rendered == [t for t in titles if t != emptied]
        assert [s for s in screen._settings_sections
                if s.title() == emptied] == [parked]
        assert parked in screen._discarded_settings_sections
        assert parked.parent() is screen._discarded_settings_host
        assert parked.isHidden()
        assert parked.property("settingsSectionDiscarded") is True
        assert layout.indexOf(parked) == -1

    with subtests.test(case="back above the categories declared after it"):
        screen, titles, emptied, parked, layout = parked_screen(0)
        assert screen._restore_settings_section(parked) is True
        assert parked.property("settingsSectionDiscarded") is False
        assert parked not in screen._discarded_settings_sections
        assert emptied in [s.title()
                           for s in screen.rendered_settings_sections()]
        following = next(s for s in screen.rendered_settings_sections()
                         if s.title() == titles[1])
        assert layout.indexOf(parked) < layout.indexOf(following)

    with subtests.test(case="the last goes back above the stretch"):
        screen, titles, emptied, parked, layout = parked_screen(-1)
        assert layout.itemAt(layout.count() - 1).spacerItem() is not None
        assert screen._restore_settings_section(parked) is True
        assert layout.indexOf(parked) == layout.count() - 2
        assert layout.itemAt(layout.count() - 1).spacerItem() is not None

    with subtests.test(case="no stretch: back at the end"):
        screen, titles, emptied, parked, layout = parked_screen(-1)
        layout.takeAt(layout.count() - 1)
        assert layout.itemAt(layout.count() - 1).spacerItem() is None
        assert screen._restore_settings_section(parked) is True
        assert layout.indexOf(parked) == layout.count() - 1

    with subtests.test(case="never parked, not moved"):
        screen = make_screen()
        live = screen.rendered_settings_sections()[0]
        layout = screen._settings_content.layout()
        before = layout.indexOf(live)
        assert screen._restore_settings_section(live) is False
        assert layout.indexOf(live) == before


def test_a_heading_holds_something_only_through_rows_or_asked_for_slots(
        make_screen, subtests):
    """What keeps a heading on the form is a row, or a slot the run asked for.

    Invariant: an umbrella whose only child declares an organelle row (the
    count owns the category before a channel is picked) holds something;
    one whose child declares rows for a switch-controlled object (nucleus)
    does not; every heading the prune reads owns a row list (a fresh
    ``Section`` an empty list, a waiting heading the lazy list), which is
    why the prune's ``rows is None`` arm is unreachable.
    """
    for name, title, child_title, declared, holds in [
            ("organelle slot asked for", "Organelle segmentation",
             "Organelle 1", ("organelle_channel", "Organelle channel", None),
             True),
            ("nucleus rows are prunable", "Object segmentation", "Nucleus",
             ("nucleus_channel", "Nucleus channel", None), False)]:
        with subtests.test(case=name):
            umbrella = Section(title)
            child = Section(child_title, parent=umbrella)
            umbrella.add_prose(child)
            assert AppScreen._section_holds_anything(umbrella) is False
            child._spacr_declared_rows = (declared,)
            assert AppScreen._section_holds_anything(umbrella) is holds

    with subtests.test(case="every heading owns its row list"):
        assert Section("Fresh").__dict__["_row_widgets"] == []
        screen = make_screen()
        waiting, _field, _ = _a_heading_with_waiting_rows(screen)
        assert isinstance(waiting.__dict__["_row_widgets"],
                          aps._RowsBuiltWhenTheyAreAskedFor)
        seen = 0
        for section in screen._settings_sections:
            for member in (section, *section.findChildren(Section)):
                assert getattr(member, "_row_widgets", None) is not None
                seen += 1
        assert seen > 5


def test_the_guards_no_caller_can_trip_still_hold(make_screen, subtests):
    """Each defensive arm below is pinned by the guarantee that makes it dead.

    Invariant: every screen's figure queue exists and declares
    ``figure_clicked``; the five modules offered a preview toggle are exactly
    the five whose runtime branch fills that card, and each card builder
    returns a card; every slot role has an organelle number, and
    ``object_of_setting`` answers only ``None``, a channelled object or a
    numbered slot role.
    """
    import inspect
    import re

    from PySide6.QtCore import Signal
    from spacr.organelle_types import _ROLE_MATCH, organelle_number
    from spacr.qt.screens.settings_model import (CHANNELLED_OBJECTS,
                                                  object_of_setting)
    from spacr.qt.widgets.figure_queue import FigureQueue
    from spacr.qt.widgets.motility_preview import build_motility_preview_card
    from spacr.qt.widgets.plaque_preview import build_plaque_preview_card
    from spacr.qt.widgets.timelapse_preview import (
        build_timelapse_preview_card)

    screen = make_screen()
    with subtests.test(case="the figure queue can be clicked"):
        assert isinstance(FigureQueue.__dict__["figure_clicked"], Signal)
        assert screen._figure_queue is not None
        assert hasattr(screen._figure_queue, "figure_clicked")

    with subtests.test(case="every preview toggle has a card"):
        source = inspect.getsource(AppScreen._build_runtime_panel)
        offered = set(re.findall(r'"(\w+)": \(\n\s+"(_\w+_card)"', source))
        assert offered == {("mask", "_live_preview_card"),
                           ("analyze_plaques", "_live_preview_card"),
                           ("timelapse", "_timelapse_preview_card"),
                           ("motility", "_motility_preview_card"),
                           ("measure", "_measure_preview_card")}
        for key, attr in offered:
            condition = ('self.app_key in ("mask", "analyze_plaques"):'
                         if key in {"mask", "analyze_plaques"}
                         else f'self.app_key == "{key}":')
            branch = source.split(condition)[1].split("elif self.app_key")[0]
            assert (f"self.{attr} = (" in branch
                    or f"_, self.{attr} = " in branch), (
                f"{key} never fills {attr}")
        for builder in (aps._build_live_preview_card,
                        aps._build_measure_preview_card,
                        build_timelapse_preview_card,
                        build_motility_preview_card,
                        build_plaque_preview_card):
            _panel, card = builder(screen)
            assert card is not None

    with subtests.test(case="every slot role has a number"):
        for role in _ROLE_MATCH:
            assert organelle_number(role) >= 1
        keys = ["src", "cell_channel", "nucleus_channel", "pathogen_channel",
                "organelle_channel", "organellez_mask_dim",
                "number_of_organelles", "summarize_organelles_by"]
        answered = {object_of_setting(key) for key in keys}
        assert answered - {None} - set(CHANNELLED_OBJECTS) == {"organelle",
                                                               "organellez"}
        for role in answered:
            assert role is None or role in CHANNELLED_OBJECTS or (
                organelle_number(role) >= 1)


def test_the_form_is_rebuilt_in_place_or_deferred_while_a_run_owns_it(
        make_screen, subtests):
    """A shaping edit hides rows in place; during a run it is remembered.

    Invariant: a worker thread that cannot be asked, or has no
    ``isRunning``, counts as stopped; a shaping field is wired to
    ``_show_the_objects_the_run_has`` (never a screen rebuild), and a key
    with no widget is skipped; during a run the rebuild is deferred, an
    unreadable model keeps the earlier deferred values and a readable one
    merges what is on screen; after the run an unreadable model keeps the
    snapshot (and clears the deferral) while a readable one refreshes it.
    """
    class Deleted:
        @staticmethod
        def isRunning():
            raise RuntimeError("Internal C++ object already deleted.")

    screen = make_screen()
    for name, thread, running in [
            ("deleted thread", Deleted(), False),
            ("running thread", types.SimpleNamespace(isRunning=lambda: True),
             True),
            ("no isRunning", object(), False)]:
        with subtests.test(case=name):
            screen._thread = thread
            assert screen._worker_thread_is_running() is running
    screen._thread = None

    with subtests.test(case="a shaping key with no widget is skipped"):
        class Signal:
            def __init__(self):
                self.connected = []

            def connect(self, slot):
                self.connected.append(slot)

        done = Signal()
        screen._settings_model = types.SimpleNamespace(_widgets={
            "nucleus_channel": types.SimpleNamespace(editingFinished=done),
            "pathogen_channel": None})
        screen._watch_the_settings_that_decide_the_form()
        assert done.connected == [screen._show_the_objects_the_run_has]

    def model(answers):
        answers = list(answers)

        def collect():
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        return types.SimpleNamespace(collect=collect, _widgets={})

    with subtests.test(case="during a run"):
        screen = make_screen()
        screen._thread = types.SimpleNamespace(isRunning=lambda: True)
        screen._deferred_form_values = {"imported": "1"}
        screen._settings_model = types.SimpleNamespace(
            collect=lambda: (_ for _ in ()).throw(
                RuntimeError("the model is mid-rebuild")), _widgets={})
        screen._rebuild_the_form()
        assert screen._deferred_form_values == {"imported": "1"}
        assert screen._form_rebuild_deferred is True
        screen._settings_model = types.SimpleNamespace(
            collect=lambda: {"typed": "2"}, _widgets={})
        screen._rebuild_the_form()
        assert screen._deferred_form_values == {"imported": "1",
                                                "typed": "2"}

    # After the run: the first read refreshes the snapshot, `_form_shape`
    # makes a second one, which must answer for there to be a shape.
    for name, first, deferred in [
            ("after the run, unreadable model", RuntimeError("mid-rebuild"),
             {"imported": "1"}),
            ("after the run, readable model", {"typed": "2"},
             {"imported": "1", "typed": "2"})]:
        with subtests.test(case=name):
            screen = make_screen()
            screen._thread = None
            screen._deferred_form_values = {"imported": "1"}
            screen._settings_model = model([first, {}])
            screen._rebuild_the_form()
            assert screen._deferred_form_values == deferred
            assert screen._form_rebuild_deferred is False
