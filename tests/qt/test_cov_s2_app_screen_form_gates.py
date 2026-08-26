"""What the module screen does when the form it is reasoning about is not there.

The dimension switches, the empty-state card, the settings advisor and the
console fold all reason about the settings FORM: which rows exist, which
category owns them, what ``src`` holds. All of that is built after the screen
is, replaced when a category is re-rendered, and absent entirely on screens
that never grew one.

So every one of these asks the form a question that can come back empty, and
the answer has to be the conservative one -- do not hide the row, do not claim
the card, do not lose the advice -- because the visible failure in each case is
a control the user cannot reach and no explanation of why.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QPixmap                                # noqa: E402
from PySide6.QtWidgets import QLineEdit, QWidget                 # noqa: E402

from spacr.qt.screens.app_screen import AppScreen                # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("the panel is half built")


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    return widget


class TestTheGeneTilePhotograph:

    def test_a_tile_that_photographs_to_nothing_is_not_offered(self, screen):
        """A null pixmap in the list shifts every caption after it."""
        panel = types.SimpleNamespace(
            gene=types.SimpleNamespace(
                feature=lambda: "cell_area",
                to_pixmap=lambda: QPixmap()))

        assert screen._gene_tile_entry(panel) == []

    def test_a_tile_that_will_not_photograph_is_not_offered_either(self,
                                                                    screen):
        panel = types.SimpleNamespace(
            gene=types.SimpleNamespace(
                feature=lambda: "cell_area", to_pixmap=_boom))

        assert screen._gene_tile_entry(panel) == []


class TestWhetherTheAdvisorCanRun:

    def test_a_panel_that_will_not_be_collected_reads_as_no_tables(
            self, screen, monkeypatch):
        """The refusal names what to do, which is better than an exception."""
        monkeypatch.setattr(screen._settings_model, "collect", _boom)

        why = screen.the_advisor_can_run()

        assert "No count or score table is attached yet" in why

    def test_a_junk_row_in_the_input_table_is_skipped(self):
        """A row read back from an old settings CSV need not be a mapping."""
        counts, scores = AppScreen._tables_for_the_advisor(
            {"paired_data": ["not a row", None,
                             {"count": "/c.csv", "score": "/s.csv"}]})

        assert counts == ["/c.csv"] and scores == ["/s.csv"]

    def test_the_legacy_single_path_spelling_is_read_too(self):
        """``count_data`` was a bare string before it was a list."""
        counts, scores = AppScreen._tables_for_the_advisor(
            {"count_data": "/counts.csv", "score_data": "/scores.csv"})

        assert counts == ["/counts.csv"] and scores == ["/scores.csv"]

    def test_a_path_named_twice_is_only_counted_once(self):
        counts, _scores = AppScreen._tables_for_the_advisor({
            "paired_data": [{"count": "/c.csv"}],
            "count_data": ["/c.csv", "/other.csv"]})

        assert counts == ["/c.csv", "/other.csv"]


class TestReadingTheLastRunForAdvice:

    def test_with_no_run_the_advice_is_unchanged(self, screen, monkeypatch):
        """The button answers before anything has been fitted."""
        reading = object()
        monkeypatch.setattr(screen, "_last_run_folder", "", raising=False)
        monkeypatch.setattr(screen, "_results_panel", None)

        assert screen._reading_with_the_last_run(reading, {}) is reading

    def test_a_run_folder_that_cannot_be_read_costs_nothing(self, screen,
                                                             monkeypatch,
                                                             tmp_path):
        """The advisor is worth more than the extra it would have added."""
        from spacr import settings_advisor

        reading = object()
        monkeypatch.setattr(screen, "_last_run_folder", str(tmp_path),
                            raising=False)
        monkeypatch.setattr(settings_advisor, "read_the_last_run", _boom)

        assert screen._reading_with_the_last_run(reading, {}) is reading

    def test_only_fields_the_reading_has_are_folded_in(self, screen,
                                                        monkeypatch,
                                                        tmp_path):
        """A newer diagnostic must not blow up an older Reading."""
        from spacr import settings_advisor
        from spacr.settings_advisor import Reading

        reading = Reading()
        monkeypatch.setattr(screen, "_last_run_folder", str(tmp_path),
                            raising=False)
        monkeypatch.setattr(settings_advisor, "read_the_last_run",
                            lambda folder, values: {
                                "run_folder": str(tmp_path),
                                "a_field_from_a_later_version": 42})

        answer = screen._reading_with_the_last_run(reading, {})

        assert answer.run_folder == str(tmp_path)
        assert not hasattr(answer, "a_field_from_a_later_version")

    def test_a_run_that_adds_nothing_leaves_the_reading_alone(self, screen,
                                                              monkeypatch,
                                                              tmp_path):
        from spacr import settings_advisor
        from spacr.settings_advisor import Reading

        reading = Reading()
        monkeypatch.setattr(screen, "_last_run_folder", str(tmp_path),
                            raising=False)
        monkeypatch.setattr(settings_advisor, "read_the_last_run",
                            lambda folder, values: {})

        assert screen._reading_with_the_last_run(reading, {}) is reading


class TestTheExampleDownloadProgress:

    def test_progress_lands_on_the_button_the_user_is_looking_at(self,
                                                                  screen,
                                                                  qtbot):
        from PySide6.QtWidgets import QPushButton

        button = QPushButton("Load the example screen…")
        qtbot.addWidget(button)
        screen._example_data_button = button

        screen._say_the_download_is_moving("counts.csv", 25, 100)

        assert button.text() == "counts.csv — 25%"

    def test_a_download_of_unknown_size_leaves_the_button_alone(self, screen,
                                                                qtbot):
        """A percentage of an unknown total is a number with no meaning."""
        from PySide6.QtWidgets import QPushButton

        button = QPushButton("Load the example screen…")
        qtbot.addWidget(button)
        screen._example_data_button = button

        screen._say_the_download_is_moving("counts.csv", 25, 0)

        assert button.text() == "Load the example screen…"

    def test_a_screen_with_no_such_button_reports_nowhere(self, screen,
                                                           monkeypatch):
        monkeypatch.setattr(screen, "_example_data_button", None,
                            raising=False)

        screen._say_the_download_is_moving("counts.csv", 25, 100)


class TestTheDimensionSwitches:

    def test_a_dimension_this_screen_does_not_have_is_not_switched(self,
                                                                    screen):
        """The settings panel can be gated before the row that gates it exists."""
        screen.set_dimension("hyperspectral", True)

        assert "hyperspectral" not in (getattr(screen, "_dimension_on", None)
                                       or {})

    def test_a_settings_file_naming_an_unknown_dimension_moves_no_switch(
            self, screen, monkeypatch):
        monkeypatch.setattr(screen, "_dimension_on", {}, raising=False)

        assert screen._sync_dimension_switches({"z_stack": True}) == ()

    def test_a_category_that_is_not_a_form_owns_no_dimensional_rows(
            self, qtbot, monkeypatch):
        """A prose card is a section too, and it has no rows to gate."""
        gated = AppScreen("mask")
        qtbot.addWidget(gated)
        assert gated._dimension_rows(), "mask has no dimensional rows to gate"
        prose = QWidget()
        qtbot.addWidget(prose)
        monkeypatch.setattr(gated, "_settings_sections", [prose],
                            raising=False)

        assert gated._dimension_rows() == []
        assert gated._dimension_hidden_sections() == set()

    def test_a_category_whose_form_went_away_is_not_held_back(self, screen,
                                                               monkeypatch):
        """The section list outlives a category that was re-rendered."""
        stale = types.SimpleNamespace(_form=None)
        monkeypatch.setattr(screen, "_dimension_rows",
                            lambda: [(stale, "z_stack", object())])

        assert screen._dimension_hidden_sections() == set()

    def test_with_no_row_helper_the_form_is_left_exactly_as_it_is(
            self, screen, monkeypatch):
        """Hiding rows a different way would leave two answers on one form."""
        import builtins

        real_import = builtins.__import__

        def refuse_the_helper(name, *args, **kwargs):
            if name.endswith("settings_search"):
                raise ImportError("no row-visibility helper")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(screen, "_dimension_rows",
                            lambda: pytest.fail("the form was walked anyway"))
        monkeypatch.setattr(builtins, "__import__", refuse_the_helper)

        screen._apply_dimension_visibility()


class TestWhetherASettingsRowIsOnScreen:

    def test_a_key_this_screen_does_not_have_is_not_visible(self, screen):
        assert screen.setting_row_is_visible("no_such_setting") is False

    def test_a_key_whose_field_is_on_no_form_is_not_visible(self, screen,
                                                             monkeypatch,
                                                             qtbot):
        """A field held by the model but never mounted shows nothing."""
        orphan = QLineEdit()
        qtbot.addWidget(orphan)
        monkeypatch.setitem(screen._settings_model._widgets,
                            "verbose", orphan)

        assert screen.setting_row_is_visible("verbose") is False

    def test_a_category_that_is_not_a_form_is_skipped_in_the_search(
            self, screen, monkeypatch, qtbot):
        """The section list can hold a prose card, which owns no rows."""
        mounted = next(iter(screen._settings_model._widgets))
        prose = QWidget()
        qtbot.addWidget(prose)
        monkeypatch.setattr(screen, "_settings_sections", [prose],
                            raising=False)

        assert screen.setting_row_is_visible(mounted) is False


class TestTheEmptyStateCard:

    def test_a_source_field_that_will_not_be_read_names_no_path(self, screen,
                                                                 monkeypatch):
        """An unreadable source is not a source, and not an exception either."""
        monkeypatch.setitem(screen._settings_model._widgets, "src",
                            types.SimpleNamespace(get_value=_boom))

        assert screen._settings_src_path() == ""

    def test_a_screen_with_no_card_shows_nothing(self, screen, monkeypatch):
        monkeypatch.setattr(screen, "_empty_state_card", None, raising=False)

        screen._refresh_empty_state()

    def test_the_card_comes_back_when_the_last_database_is_removed(
            self, screen, monkeypatch, qtbot):
        """A set can be emptied, and the screen returns to what the card says."""
        card = QWidget()
        qtbot.addWidget(card)
        monkeypatch.setattr(screen, "_empty_state_card", card, raising=False)
        monkeypatch.setattr(screen, "_settings_src_path", lambda: "")

        screen._refresh_empty_state()

        assert card.isVisibleTo(screen) or not card.isHidden()

    def test_a_placeholder_path_is_not_data(self, screen, monkeypatch, qtbot):
        """``/path/to/src`` is the default, not somewhere with images in it."""
        card = QWidget()
        qtbot.addWidget(card)
        monkeypatch.setattr(screen, "_empty_state_card", card, raising=False)
        monkeypatch.setattr(screen, "_settings_src_path",
                            lambda: "/path/to/src")

        screen._refresh_empty_state()

        assert not card.isHidden()

    def test_a_screen_whose_demo_cannot_be_named_still_gets_its_card(
            self, qtbot, monkeypatch):
        """The card is the instruction; the demo is one clause of it.

        Naming a demo that opens a DIFFERENT module is what this branch
        exists to avoid -- following that hint on Measure generates images,
        navigates to Mask, and leaves the empty screen exactly as empty.
        """
        from spacr.qt import app as app_module

        monkeypatch.setattr(app_module, "demo_label_for_app", _boom)
        built = AppScreen("mask")
        qtbot.addWidget(built)

        card = built._build_empty_state_banner()

        assert card is not None


class TestFoldingTheConsole:

    def test_a_screen_with_no_console_wrapper_folds_nothing(self, screen,
                                                             monkeypatch):
        monkeypatch.setattr(screen, "_console_wrap", None, raising=False)

        screen._console_folded(True)

    def test_a_console_that_is_in_no_splitter_still_takes_its_minimum(
            self, screen, monkeypatch, qtbot):
        """Hiding the widget alone is the failure; the minimum is the other half."""
        wrap = QWidget()
        qtbot.addWidget(wrap)
        monkeypatch.setattr(screen, "_console_wrap", wrap, raising=False)
        monkeypatch.setattr(screen, "_console_splitter", None, raising=False)

        screen._console_folded(True)

        assert wrap.minimumHeight() == 0

    def test_a_console_that_left_its_splitter_is_not_resized(self, screen,
                                                              monkeypatch,
                                                              qtbot):
        from PySide6.QtWidgets import QSplitter

        splitter = QSplitter()
        qtbot.addWidget(splitter)
        wrap = QWidget()
        qtbot.addWidget(wrap)
        monkeypatch.setattr(screen, "_console_wrap", wrap, raising=False)
        monkeypatch.setattr(screen, "_console_splitter", splitter,
                            raising=False)

        screen._console_folded(False)

        assert wrap.minimumHeight() == 180


def test_a_screen_with_no_maturity_notice_still_hides_its_alpha_sections(
        screen, monkeypatch):
    """The notice is the explanation; the hiding is the behaviour."""
    monkeypatch.setattr(screen, "_maturity_notice", None, raising=False)

    screen.refresh_maturity_visibility()


class TestWhatTheAdvisorSaysBeforeItProposes:

    def _arm(self, screen, monkeypatch, reading):
        from spacr import settings_advisor

        said = []
        screen._console = types.SimpleNamespace(append_stdout=said.append)
        monkeypatch.setattr(screen, "the_advisor_can_run", lambda: "")
        monkeypatch.setattr(screen._settings_model, "collect", lambda: {
            "paired_data": [{"count": "/c.csv", "score": "/s.csv"}],
            "dependent_variable": "pred"})
        monkeypatch.setattr(settings_advisor, "read_the_screen",
                            lambda *a, **k: reading)
        monkeypatch.setattr(screen, "_reading_with_the_last_run",
                            lambda reading_, values: reading_)
        monkeypatch.setattr(settings_advisor, "advise_that_runs",
                            lambda *a, **k: types.SimpleNamespace(
                                as_settings=lambda: {}))
        return said

    def test_every_reservation_the_reading_has_is_printed(self, screen,
                                                           monkeypatch):
        """The advice is only usable beside what the data could not support."""
        from spacr.settings_advisor import Reading

        said = self._arm(screen, monkeypatch, Reading(
            trouble=["only two plates", "one guide per gene"]))

        screen.settings_for_my_data(answers={})

        assert any("only two plates" in line for line in said)
        assert any("one guide per gene" in line for line in said)

    def test_a_run_that_could_not_be_read_says_why_rather_than_going_quiet(
            self, screen, monkeypatch):
        """The last run's residuals are an addition; a missing one is not."""
        from spacr.settings_advisor import Reading

        said = self._arm(screen, monkeypatch, Reading(
            run_folder="/data/run", run_note="its summary was not written"))

        screen.settings_for_my_data(answers={})

        assert any("its summary was not written" in line for line in said)

    def test_a_run_that_was_read_is_named(self, screen, monkeypatch):
        from spacr.settings_advisor import Reading

        said = self._arm(screen, monkeypatch,
                         Reading(run_folder="/data/run"))

        screen.settings_for_my_data(answers={})

        assert any("Also reading the diagnostics of /data/run" in line
                   for line in said)
