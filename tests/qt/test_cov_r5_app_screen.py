"""Empty categories, the headings that come back, and a bulk apply with no model.

``test_cov_r4_app_screen.py`` covers the screen's refusals and its lazy rows.
What is left, and what this file drives, is the CATEGORY-level bookkeeping
around them: a heading that holds nothing is parked rather than shown, a parked
heading is put back at the position it was declared in, and everything the
settings panel does when the collaborator it reads -- the settings model, the
worker thread, the window that owns the screen -- is missing or raising.

The shared risk is that a pruned heading and a heading that failed to build
look identical from outside, so every prune test also asserts the form rows are
still reachable, and every "nothing happened" test drives the input that makes
something happen in the same test.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QLineEdit                          # noqa: E402

from spacr.qt.screens import app_screen as aps                   # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widgets.section import Section                     # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    """The same regression screen the round-4 file uses."""
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


def _a_heading_with_one_waiting_row(screen):
    """A heading holding one row the object rule has decided to hide.

    Built through ``_build_settings_section``, which is the only thing that
    makes a waiting row, so ``_rows_awaiting_layout`` and the declared-row
    record are left exactly as a panel build leaves them.
    """
    field = QLineEdit()
    screen._settings_model._widgets["ghost_setting"] = field
    screen._widget_key_stamp = None
    screen._run_has_no_object_for = {"ghost_setting"}
    section = screen._build_settings_section(
        ("Ghost", [("Ghost setting", field)]))
    return section, field


def _hollow(section):
    """Strip a built heading of everything ``_section_holds_anything`` reads."""
    for member in (section, *section.findChildren(Section)):
        member.__dict__["_row_widgets"] = []
        member._spacr_declared_rows = ()


def _rebuilt_panel_with_hollow_headings(screen, hollow_titles):
    """Lay the settings panel out again with ``hollow_titles`` left empty.

    The prune is decided over the FINISHED section, so the only way to drive
    it from outside is to hand the panel a heading that holds nothing --
    exactly what a run with no organelles does to "Organelle segmentation".
    """
    build = screen._build_settings_section

    def build_but_empty_some(spec, depth: int = 0):
        section = build(spec, depth)
        if depth == 0 and section.title() in hollow_titles:
            _hollow(section)
        return section

    screen._build_settings_section = build_but_empty_some
    try:
        screen._lay_out_the_settings_panel()
    finally:
        del screen._build_settings_section


# ---------------------------------------------------------------------------
# The lazy row list
# ---------------------------------------------------------------------------

class TestReadingOneRowByIndex:

    def test_indexing_a_heading_builds_the_rows_that_were_waiting(self,
                                                                   screen):
        """``rows[0]`` is a read, and a read has to hand over a whole row."""
        section, field = _a_heading_with_one_waiting_row(screen)
        rows = section.__dict__["_row_widgets"]
        assert isinstance(rows, aps._RowsBuiltWhenTheyAreAskedFor)
        assert screen._rows_awaiting_layout == {"ghost_setting": section}

        first = rows[0]

        assert first[1] is field
        assert first[0].text() == "Ghost setting"
        assert screen._rows_awaiting_layout == {}


# ---------------------------------------------------------------------------
# A category with nothing in it is not shown
# ---------------------------------------------------------------------------

class TestAnEmptyCategoryIsParkedNotShown:

    def test_a_heading_that_holds_nothing_leaves_the_panel_but_not_the_form(
            self, screen):
        """It must keep collecting: its widgets are the model's value store."""
        titles = [section.title()
                  for section in screen.rendered_settings_sections()]
        assert len(titles) > 2
        emptied = titles[0]

        _rebuilt_panel_with_hollow_headings(screen, {emptied})

        rendered = [s.title() for s in screen.rendered_settings_sections()]
        assert emptied not in rendered
        assert rendered == [t for t in titles if t != emptied]
        dormant = [s for s in screen._settings_sections
                   if s.title() == emptied]
        assert len(dormant) == 1
        parked = dormant[0]
        # Still registered, still owned, and explicitly out of sight rather
        # than a stray top-level window.
        assert parked in screen._discarded_settings_sections
        assert parked.parent() is screen._discarded_settings_host
        assert parked.isHidden()
        assert parked.property("settingsSectionDiscarded") is True
        assert screen._settings_content.layout().indexOf(parked) == -1


class TestPuttingADormantHeadingBack:

    def test_it_goes_back_above_the_categories_declared_after_it(self,
                                                                  screen):
        """Declaration order, not "wherever the rule got round to it"."""
        titles = [s.title() for s in screen.rendered_settings_sections()]
        emptied = titles[0]
        _rebuilt_panel_with_hollow_headings(screen, {emptied})
        parked = next(s for s in screen._settings_sections
                      if s.title() == emptied)
        layout = screen._settings_content.layout()

        assert screen._restore_settings_section(parked) is True

        assert parked.property("settingsSectionDiscarded") is False
        assert parked not in screen._discarded_settings_sections
        after = [s.title() for s in screen.rendered_settings_sections()]
        assert after[:1] == [emptied] or emptied in after
        following = next(s for s in screen.rendered_settings_sections()
                         if s.title() == titles[1])
        assert layout.indexOf(parked) < layout.indexOf(following)

    def test_the_last_category_goes_back_above_the_stretch_not_below_it(
            self, screen):
        """The stretch is what keeps the cards at the top of the column."""
        titles = [s.title() for s in screen.rendered_settings_sections()]
        emptied = titles[-1]
        _rebuilt_panel_with_hollow_headings(screen, {emptied})
        parked = next(s for s in screen._settings_sections
                      if s.title() == emptied)
        layout = screen._settings_content.layout()
        assert layout.itemAt(layout.count() - 1).spacerItem() is not None

        assert screen._restore_settings_section(parked) is True

        at = layout.indexOf(parked)
        assert at == layout.count() - 2
        assert layout.itemAt(layout.count() - 1).spacerItem() is not None

    def test_with_no_stretch_below_them_it_goes_back_at_the_end(self, screen):
        """A column that never got its stretch still gets the card."""
        titles = [s.title() for s in screen.rendered_settings_sections()]
        emptied = titles[-1]
        _rebuilt_panel_with_hollow_headings(screen, {emptied})
        parked = next(s for s in screen._settings_sections
                      if s.title() == emptied)
        layout = screen._settings_content.layout()
        layout.takeAt(layout.count() - 1)        # the stretch
        assert layout.itemAt(layout.count() - 1).spacerItem() is None

        assert screen._restore_settings_section(parked) is True

        assert layout.indexOf(parked) == layout.count() - 1

    def test_a_heading_that_was_never_parked_is_not_moved(self, screen):
        live = screen.rendered_settings_sections()[0]
        layout = screen._settings_content.layout()
        before = layout.indexOf(live)

        assert screen._restore_settings_section(live) is False
        assert layout.indexOf(live) == before


# ---------------------------------------------------------------------------
# What counts as "holding something"
# ---------------------------------------------------------------------------

class TestWhetherAHeadingHoldsAnything:

    def test_a_sub_heading_whose_slot_the_run_asked_for_keeps_its_umbrella(
            self):
        """The organelle COUNT owns the category before a channel is picked.

        The umbrella itself is empty and the child's one row is still waiting
        for its caption -- what keeps both on the form is the child's
        DECLARED organelle row.
        """
        umbrella = Section("Organelle segmentation")
        child = Section("Organelle 1", parent=umbrella)
        umbrella.add_prose(child)
        assert AppScreen._section_holds_anything(umbrella) is False

        child._spacr_declared_rows = (
            ("organelle_channel", "Organelle channel", None),)

        assert AppScreen._section_holds_anything(umbrella) is True

    def test_a_sub_heading_declaring_rows_for_no_slot_does_not(self):
        """nucleus/pathogen rows are prunable while their switch is empty."""
        umbrella = Section("Object segmentation")
        child = Section("Nucleus", parent=umbrella)
        umbrella.add_prose(child)
        child._spacr_declared_rows = (
            ("nucleus_channel", "Nucleus channel", None),)

        assert AppScreen._section_holds_anything(umbrella) is False


# ---------------------------------------------------------------------------
# Watching the values that shape the form
# ---------------------------------------------------------------------------

class _RecordingSignal:
    def __init__(self):
        self.connected = []

    def connect(self, slot):
        self.connected.append(slot)


class TestWatchingTheShapingSettings:

    def test_a_shaping_key_with_no_widget_is_skipped_not_guessed(self,
                                                                 screen):
        """``_form_shaping_keys`` reads the inventory; the model may lag it."""
        done = _RecordingSignal()
        present = types.SimpleNamespace(editingFinished=done)
        screen._settings_model = types.SimpleNamespace(
            _widgets={"nucleus_channel": present, "pathogen_channel": None})

        screen._watch_the_settings_that_decide_the_form()

        # IT HIDES THE ROWS, it does not rebuild the screen. Committing a
        # plane used to call `rebuild_app_screen` -- 455 ms and a different
        # screen object in the stack -- to change which rows are visible,
        # taking every uncommitted edit, scroll position and open fold with
        # it. `_show_the_objects_the_run_has` does the same job in place, and
        # is what a shaping field is wired to now.
        assert done.connected == [screen._show_the_objects_the_run_has]


# ---------------------------------------------------------------------------
# Is a run still holding this screen?
# ---------------------------------------------------------------------------

class TestWhetherTheWorkerIsRunning:

    def test_a_thread_that_cannot_be_asked_counts_as_stopped(self, screen):
        """A deleted QThread wrapper must not make the screen unrebuildable."""
        class _Deleted:
            @staticmethod
            def isRunning():
                raise RuntimeError("Internal C++ object already deleted.")

        screen._thread = _Deleted()
        assert screen._worker_thread_is_running() is False

        screen._thread = types.SimpleNamespace(isRunning=lambda: True)
        assert screen._worker_thread_is_running() is True

    def test_a_thread_object_with_no_isrunning_counts_as_stopped(self, screen):
        screen._thread = object()
        assert screen._worker_thread_is_running() is False


# ---------------------------------------------------------------------------
# Rebuilding the form
# ---------------------------------------------------------------------------

class TestRebuildingWhileARunOwnsTheScreen:

    def test_values_the_form_cannot_collect_are_kept_not_dropped(self,
                                                                  screen):
        """A shaping edit during a run is remembered; a model that cannot be
        read must not take the earlier import down with it."""
        screen._thread = types.SimpleNamespace(isRunning=lambda: True)
        screen._deferred_form_values = {"imported": "1"}

        def _boom():
            raise RuntimeError("the model is mid-rebuild")

        screen._settings_model = types.SimpleNamespace(collect=_boom,
                                                        _widgets={})
        screen._rebuild_the_form()

        assert screen._deferred_form_values == {"imported": "1"}
        assert screen._form_rebuild_deferred is True

        # The same call with a readable model merges what is on screen, so
        # the untouched dict above is the failure and not a dead branch.
        screen._settings_model = types.SimpleNamespace(
            collect=lambda: {"typed": "2"}, _widgets={})
        screen._rebuild_the_form()

        assert screen._deferred_form_values == {"imported": "1",
                                                "typed": "2"}


class TestRefreshingADeferredSnapshotAfterTheRun:

    def _model(self, screen, answers):
        answers = list(answers)

        def collect():
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        model = types.SimpleNamespace(collect=collect, _widgets={})
        screen._settings_model = model
        return model

    def test_a_snapshot_survives_a_model_that_cannot_be_read(self, screen):
        """The deferred values are minutes old and are all there is left."""
        screen._thread = None
        screen._deferred_form_values = {"imported": "1"}
        # The first read fails; `_form_shape` makes a second one, which must
        # still answer or there would be nothing to compare shapes with.
        self._model(screen, [RuntimeError("mid-rebuild"), {}])

        screen._rebuild_the_form()

        assert screen._deferred_form_values == {"imported": "1"}
        assert screen._form_rebuild_deferred is False

    def test_a_readable_model_refreshes_the_snapshot_before_it_is_used(
            self, screen):
        screen._thread = None
        screen._deferred_form_values = {"imported": "1"}
        self._model(screen, [{"typed": "2"}, {}])

        screen._rebuild_the_form()

        assert screen._deferred_form_values == {"imported": "1",
                                                "typed": "2"}


# ---------------------------------------------------------------------------
# Captioning
# ---------------------------------------------------------------------------

class TestCaptioningEveryWaitingRow:

    def test_with_nothing_waiting_the_panel_is_left_alone(self, screen):
        moved = []
        screen._the_rows_moved = lambda **kwargs: moved.append(kwargs)
        _section, field = _a_heading_with_one_waiting_row(screen)

        screen._caption_every_waiting_row()
        assert moved == [{"judge_them": True}]
        assert field._spacr_setting_label.text() == "Ghost setting"

        screen._caption_every_waiting_row()

        assert moved == [{"judge_them": True}]


class TestARestoredHeadingWithNoHintStrip:

    def test_the_rows_come_back_without_a_category_hint_to_rewire(self,
                                                                  screen):
        """A screen whose runtime panel is not built has no hint strip."""
        section, field = _a_heading_with_one_waiting_row(screen)
        screen._discard_settings_section(section)
        assert section.property("settingsSectionDiscarded") is True
        screen._category_hint = None

        screen._lay_out_the_rows_that_are_back(())

        assert section.property("settingsSectionDiscarded") is False
        assert screen._rows_awaiting_layout == {}
        assert field._spacr_setting_label.text() == "Ghost setting"


class TestTranslatingACaptionThatArrivedLate:

    def test_a_heading_destroyed_before_the_language_pass_is_not_an_error(
            self, screen, monkeypatch):
        from spacr.qt import i18n

        section, _field = _a_heading_with_one_waiting_row(screen)
        seen = []
        monkeypatch.setattr(i18n, "retranslate_widget_tree", seen.append)
        screen._captioned_late = {section}

        screen._the_rows_moved(judge_them=False)

        assert seen == [section]
        assert screen._captioned_late == set()

        def _gone(_widget):
            raise RuntimeError("Internal C++ object already deleted.")

        monkeypatch.setattr(i18n, "retranslate_widget_tree", _gone)
        screen._captioned_late = {section}

        screen._the_rows_moved(judge_them=False)

        assert screen._captioned_late == set()


# ---------------------------------------------------------------------------
# The publication sheet's import
# ---------------------------------------------------------------------------

class TestTheSheetBuilderIsFoundEitherWay:

    def test_a_screen_whose_package_cannot_be_resolved_still_draws(
            self, screen, monkeypatch):
        """``from ...figures`` needs the module's package; the absolute
        import is the fallback for when it is not there -- a module loaded
        outside its package, which is what the relative form cannot survive.
        """
        import pandas as pd
        from matplotlib.figure import Figure
        from spacr import figures

        screen._results_panel = types.SimpleNamespace(
            results_frame=lambda: pd.DataFrame({"gene": ["A"],
                                                "coefficient": [1.0]}))
        figure = Figure()
        sheet = types.SimpleNamespace(
            figure=figure, skipped=(),
            legend=lambda: "Panels a-c: coefficients.")
        monkeypatch.setattr(figures, "build_sheet",
                            lambda frame, width="double": sheet)
        shown = []
        screen._figure_queue = types.SimpleNamespace(
            add_figure=shown.append, count=lambda: 1,
            show_index=shown.append)
        screen._figures_stack = None

        # A package the relative import cannot resolve. ``__spec__`` is
        # cleared with it so the import machinery reads the name we set
        # rather than warning about the disagreement.
        monkeypatch.setattr(aps, "__package__", "not_a_package.qt.screens")
        monkeypatch.setattr(aps, "__spec__", None)

        screen._show_publication_sheet()

        assert shown == [figure, 0]


# ---------------------------------------------------------------------------
# A bulk apply
# ---------------------------------------------------------------------------

class _Model:
    """The parts of ``SettingsWidgets`` a bulk apply touches."""

    def __init__(self, collect=None, widgets=None):
        self._widgets = dict(widgets or {})
        self._defaults = {}
        self._applying_settings = False
        self._collect = collect
        self.presets = []
        self.dependencies = 0

    def collect(self):
        if self._collect is None:
            return dict(self._widgets)
        return self._collect()

    def set_hidden_value(self, key, value):
        return False

    def _refresh_contextual_widgets(self):
        pass

    def apply_organelle_presets_from_mapping(self, mapping):
        self.presets.append(dict(mapping))

    def _refresh_setting_dependencies(self):
        self.dependencies += 1


class TestABulkApplyWithNothingToReadFrom:

    def test_a_model_that_cannot_be_collected_starts_from_no_values(
            self, screen):
        """The comparison is with what is on screen; an unreadable form
        contributes nothing rather than aborting the import."""
        def _boom():
            raise RuntimeError("half-built")

        screen._settings_model = _Model(collect=_boom)
        seen = []
        screen._bulk_apply_changes_form_shape = (
            lambda settings, current: seen.append(dict(current)) or False)

        assert screen.apply_settings_dict({"src": "/data"}) == 0
        assert seen == [{}]

        screen._settings_model = _Model(collect=lambda: {"src": "/old"})
        assert screen.apply_settings_dict({"src": "/data"}) == 0
        assert seen == [{}, {"src": "/old"}]

    def test_a_screen_with_no_settings_model_still_applies_and_refreshes(
            self, screen):
        """Every step of the apply is guarded, so a screen whose panel never
        built must not raise -- and must still run the after-apply rules."""
        applied = []
        screen._apply_each_setting = lambda settings: (
            applied.append(dict(settings)) or 3)
        screen._bulk_apply_changes_form_shape = lambda settings, current: False
        screen._settings_model = None
        screen._folds_last_switched_on = None

        assert screen.apply_settings_dict({"src": "/data"}) == 3
        assert applied == [{"src": "/data"}]
        assert screen._folds_last_switched_on == ()

        # With a model, the same call brackets the loop in the applying flag
        # and runs the cross-setting rules -- which is what the guards skip.
        model = _Model()
        screen._settings_model = model
        during = []
        screen._apply_each_setting = lambda settings: (
            during.append(model._applying_settings) or 3)

        assert screen.apply_settings_dict({"src": "/data"}) == 3
        assert during == [True]
        assert model._applying_settings is False
        assert model.presets == [{"src": "/data"}]
        assert model.dependencies == 1


class TestAWindowThatHandsBackNoReplacement:

    def _window(self, screen, screens):
        window = types.SimpleNamespace(_screens=dict(screens), rebuilt=[])
        window.rebuild_app_screen = (
            lambda key, values: window.rebuilt.append((key, dict(values))))
        screen.window = lambda: window
        return window

    def test_the_values_are_applied_here_when_no_new_screen_appears(
            self, screen):
        """A rebuild that produced nothing must not lose the settings."""
        screen._settings_model = _Model()
        screen._bulk_apply_changes_form_shape = lambda settings, current: True
        screen._thread = None
        window = self._window(screen, {})
        applied = []
        screen._apply_each_setting = lambda settings: (
            applied.append(dict(settings)) or 1)

        assert screen.apply_settings_dict({"src": "/data"}) == 1

        assert window.rebuilt == [("regression", {"src": "/data"})]
        assert applied == [{"src": "/data"}]

    def test_a_window_that_hands_back_this_same_screen_does_not_recurse(
            self, screen):
        screen._settings_model = _Model()
        screen._bulk_apply_changes_form_shape = lambda settings, current: True
        screen._thread = None
        self._window(screen, {"regression": screen})
        applied = []
        screen._apply_each_setting = lambda settings: (
            applied.append(dict(settings)) or 1)

        assert screen.apply_settings_dict({"src": "/data"}) == 1
        assert applied == [{"src": "/data"}]

    def test_a_replacement_screen_is_handed_the_settings_instead(self,
                                                                  screen):
        screen._settings_model = _Model()
        screen._bulk_apply_changes_form_shape = lambda settings, current: True
        screen._thread = None
        handed = []
        fresh = types.SimpleNamespace(
            apply_settings_dict=lambda settings: (
                handed.append(dict(settings)) or 7))
        self._window(screen, {"regression": fresh})
        screen._apply_each_setting = lambda settings: 1

        assert screen.apply_settings_dict({"src": "/data"}) == 7
        assert handed == [{"src": "/data"}]


class TestWhetherABulkApplyChangesTheFormShape:

    def test_a_slot_setting_on_a_form_with_no_organelle_count_changes_nothing(
            self, screen):
        """A module without the count has no slots to create or destroy."""
        from spacr.organelle_types import NUMBER_OF_ORGANELLES

        assert screen._bulk_apply_changes_form_shape(
            {"organelle_channel": 2}, {}) is False

        # The same key against a form that HAS the count does reshape it.
        assert screen._bulk_apply_changes_form_shape(
            {"organelle_channel": 2},
            {NUMBER_OF_ORGANELLES: 1, "organelle_channel": None}) is True

    def test_a_slot_beyond_the_requested_count_changes_nothing(self, screen):
        from spacr.organelle_types import NUMBER_OF_ORGANELLES

        current = {NUMBER_OF_ORGANELLES: 1, "organelleb_channel": None}
        assert screen._bulk_apply_changes_form_shape(
            {"organelleb_channel": 2}, current) is False


# ---------------------------------------------------------------------------
# Guards no caller can trip
#
# Each of the four below is a defensive arm that this round could not drive.
# Rather than contort a test into reaching one, each test pins the guarantee
# that makes it unreachable, so the day the guarantee stops holding the test
# is what says so.
# ---------------------------------------------------------------------------

class TestTheGuardsNothingCanTrip:

    def test_every_heading_the_prune_reads_already_owns_its_row_list(
            self, screen):
        """``already_built_rows``' ``rows is None`` arm cannot be reached.

        Both callers of ``_section_holds_anything`` hand it a heading built by
        ``_build_settings_section``, and every heading -- top level, nested,
        or one whose rows are still waiting -- is a ``Section``, whose
        ``__init__`` sets ``_row_widgets`` to a list. Nothing in spaCR ever
        assigns it ``None``.
        """
        assert Section("Fresh").__dict__["_row_widgets"] == []
        waiting, _field = _a_heading_with_one_waiting_row(screen)
        assert isinstance(waiting.__dict__["_row_widgets"],
                          aps._RowsBuiltWhenTheyAreAskedFor)

        seen = 0
        for section in screen._settings_sections:
            for member in (section, *section.findChildren(Section)):
                assert getattr(member, "_row_widgets", None) is not None
                seen += 1
        assert seen > 5

    def test_a_late_caption_goes_back_as_a_complete_labelled_row(self,
                                                                  screen):
        """The taken row always has both halves, which is why the two
        ``item is None`` arms in ``_lay_out_one_waiting_row`` are dead.

        ``_lay_out_setting_row`` finishes with ``Section.add_row``, which is
        ``QFormLayout.addRow(label, field)`` -- two items -- and ``last`` names
        exactly the row it just appended.
        """
        from PySide6.QtWidgets import QFormLayout

        ghost, plain = QLineEdit(), QLineEdit()
        screen._settings_model._widgets["ghost_setting"] = ghost
        screen._settings_model._widgets["plain_setting"] = plain
        screen._widget_key_stamp = None
        screen._run_has_no_object_for = {"ghost_setting"}
        section = screen._build_settings_section(
            ("Ghost", [("Ghost setting", ghost), ("Plain setting", plain)]))
        form = section._form
        assert form.getWidgetPosition(ghost)[0] == 0
        assert form.getWidgetPosition(plain)[0] == 1

        screen._lay_out_one_waiting_row("ghost_setting")

        # Back in its own place, above the row that was declared after it...
        assert form.getWidgetPosition(ghost)[0] == 0
        assert form.getWidgetPosition(plain)[0] == 1
        # ...and with the label the move had to carry with it.
        label_item = form.itemAt(0, QFormLayout.LabelRole)
        assert label_item is not None
        assert ghost._spacr_setting_label.text() == "Ghost setting"
        assert ghost._spacr_setting_label.parent() is label_item.widget()

    def test_the_figure_queue_every_screen_builds_can_be_clicked(self,
                                                                  screen):
        """Why the ``queue is not None and hasattr(...)`` guard is dead.

        ``_build_runtime_panel`` constructs the queue unconditionally, before
        anything reads it, and ``figure_clicked`` is declared on the class --
        so every instance has it.
        """
        from spacr.qt.widgets.figure_queue import FigureQueue
        from PySide6.QtCore import Signal

        assert isinstance(FigureQueue.__dict__["figure_clicked"], Signal)
        assert screen._figure_queue is not None
        assert hasattr(screen._figure_queue, "figure_clicked")

    def test_every_module_with_a_preview_toggle_has_a_card_to_toggle(self,
                                                                     screen):
        """Why the ``card_attr is not None`` guard in the actions row is dead.

        The four keys the toggle is offered for are exactly the four the
        runtime panel builds a preview card for, and none of the four builders
        can return without one.
        """
        import inspect
        import re

        source = inspect.getsource(AppScreen._build_runtime_panel)
        offered = set(re.findall(r'"(\w+)": \(\n\s+"(_\w+_card)"', source))
        assert offered == {("mask", "_live_preview_card"),
                           ("timelapse", "_timelapse_preview_card"),
                           ("motility", "_motility_preview_card"),
                           ("measure", "_measure_preview_card")}
        for key, attr in offered:
            # ...and each of those four keys is a branch that fills the very
            # attribute the toggle then reads.
            branch = source.split(f'self.app_key == "{key}":')[1]
            assert f"self.{attr} = (" in branch.split("elif self.app_key")[0]

        from spacr.qt.widgets.motility_preview import (
            build_motility_preview_card)
        from spacr.qt.widgets.timelapse_preview import (
            build_timelapse_preview_card)

        for builder in (aps._build_live_preview_card,
                        aps._build_measure_preview_card,
                        build_timelapse_preview_card,
                        build_motility_preview_card):
            _panel, card = builder(screen)
            assert card is not None

    def test_every_object_a_setting_can_belong_to_has_a_slot_number(self):
        """Why the ``except ValueError`` in the bulk-shape check is dead.

        ``object_of_setting`` answers with one of the three channelled objects
        or with a slot role from ``organelle_types``; the first three never
        reach the ``organelle_number`` call, and every slot role has a number
        by construction.
        """
        from spacr.organelle_types import _ROLE_MATCH, organelle_number
        from spacr.qt.screens.settings_model import (CHANNELLED_OBJECTS,
                                                      object_of_setting)

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
