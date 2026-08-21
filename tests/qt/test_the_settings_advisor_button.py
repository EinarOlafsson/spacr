"""192 at the panel: the button, the questions, and the proposal.

The arithmetic is tested headless in
`tests/test_a_button_picks_the_settings_for_your_data.py`. What is tested here
is that a user can reach it, that the button is where the request put it, and
-- the part that matters most -- that NOTHING is written until the proposal
has been seen and accepted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

pytestmark = pytest.mark.qt

from spacr.settings_advisor import Reading


def test_every_advisor_question_and_caption_is_in_the_runtime_catalog_source():
    """Dynamic question records must not bypass supported UI languages."""
    from tools import build_i18n_catalogs as builder
    from spacr.qt.widgets.settings_advisor_dialog import (
        _SETTINGS_ADVISOR_UI_SOURCES,
    )

    sources = set(builder.canonical_sources()["ui"])
    assert set(_SETTINGS_ADVISOR_UI_SOURCES) <= sources


ANSWERS = {"hits_per_thousand": 20, "direction": "either",
           "controls": "000000", "cost": "balanced"}


def _counts(path):
    out = []
    for plate in range(1, 5):
        for row in range(1, 5):
            for column in range(1, 7):
                for gene in range(20):
                    for guide in range(3):
                        out.append({"plate": f"plate{plate}",
                                    "row_name": f"r{row}",
                                    "column_name": f"c{column}",
                                    "grna_name": f"TGGT1_{gene:06d}_{guide+1}",
                                    "count": 100 + gene})
    pd.DataFrame(out).to_csv(path, index=False)
    return str(path)


def _scores(path):
    rng = np.random.default_rng(3)
    out = []
    for plate in range(1, 5):
        for row in range(1, 5):
            for column in range(1, 7):
                for _ in range(20):
                    out.append({"prc": f"plate{plate}_r{row}_c{column}",
                                "pred": float(np.clip(rng.beta(2, 5), 1e-4,
                                                      1 - 1e-4))})
    pd.DataFrame(out).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def screen(qtbot, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    widget._tables = (_counts(tmp_path / "counts.csv"),
                      _scores(tmp_path / "scores.csv"))
    return widget


@pytest.fixture
def loaded(screen, monkeypatch):
    """A screen whose Input Tables name the two files."""
    counts, scores = screen._tables
    original = screen._settings_model.collect
    monkeypatch.setattr(
        screen._settings_model, "collect",
        lambda: {**(original() or {}),
                 "paired_data": [{"count": counts, "score": scores}],
                 "dependent_variable": "pred"})
    return screen


class TestTheButtonIsWhereTheRequestPutIt:

    def test_the_regression_screen_has_one(self, screen):
        assert getattr(screen, "_advisor_button", None) is not None

    def test_it_is_to_the_LEFT_of_inference(self, screen):
        """"a button to the left of inference alligned with the text box".
        The plate map (185) sits on the RIGHT of the field it fills, and the
        two must not be mistaken for each other."""
        field = screen._settings_model._widgets.get("inference")
        holder = field.parent()
        layout = holder.layout()
        order = [layout.itemAt(i).widget() for i in range(layout.count())]

        assert order.index(screen._advisor_button) < order.index(field)

    def test_the_field_is_still_what_the_panel_collects(self, screen):
        """A wrapper that made the value unreadable would be worse than no
        button."""
        assert "inference" in (screen._settings_model.collect() or {})

    def test_another_module_does_not_get_one(self, qtbot):
        from spacr.qt.screens.app_screen import AppScreen

        other = AppScreen("mask")
        qtbot.addWidget(other)

        assert getattr(other, "_advisor_button", None) is None


class TestItSaysWhyItCannotRun:

    def test_with_no_tables_it_names_what_is_missing(self, screen):
        why = screen.the_advisor_can_run()

        assert "Input Tables" in why

    def test_and_pressing_it_writes_nothing(self, screen):
        before = dict(screen._settings_model.collect() or {})

        assert screen.settings_for_my_data(answers=ANSWERS) == {}
        assert dict(screen._settings_model.collect() or {}) == before

    def test_with_tables_attached_it_can(self, loaded):
        assert loaded.the_advisor_can_run() == ""

    def test_the_legacy_input_spelling_is_read_too(self, screen, monkeypatch):
        """A panel filled from an old settings CSV has count_data/score_data
        rather than paired_data."""
        counts, scores = screen._tables
        original = screen._settings_model.collect
        monkeypatch.setattr(
            screen._settings_model, "collect",
            lambda: {**(original() or {}), "count_data": [counts],
                     "score_data": [scores]})

        assert screen.the_advisor_can_run() == ""


class TestNothingIsWrittenUntilItIsAccepted:

    def test_the_headless_route_writes_the_proposal(self, loaded):
        chosen = loaded.settings_for_my_data(answers=ANSWERS)

        assert chosen["regression_type"] == "beta"
        assert loaded._settings_model.collect()["regression_type"] == "beta"

    def test_a_cancelled_dialog_writes_nothing(self, loaded, monkeypatch):
        from PySide6.QtWidgets import QDialog

        before = dict(loaded._settings_model.collect() or {})
        monkeypatch.setattr(
            "spacr.qt.widgets.settings_advisor_dialog.SettingsAdvisorDialog"
            ".exec", lambda self: QDialog.Rejected)

        assert loaded.settings_for_my_data() == {}
        assert dict(loaded._settings_model.collect() or {}) == before

    def test_accepting_without_seeing_the_proposal_writes_nothing(
            self, loaded, monkeypatch):
        """The Apply button only exists on the second page; this guards the
        path where a caller accepts the dialog without ever getting there."""
        from PySide6.QtWidgets import QDialog

        monkeypatch.setattr(
            "spacr.qt.widgets.settings_advisor_dialog.SettingsAdvisorDialog"
            ".exec", lambda self: QDialog.Accepted)

        assert loaded.settings_for_my_data() == {}


class TestTheDialogShowsTheArgument:

    @pytest.fixture
    def dialog(self, qtbot):
        from spacr.qt.widgets.settings_advisor_dialog import \
            SettingsAdvisorDialog

        reading = Reading(plates=4, wells=384, guides=60, genes=20,
                          guides_per_gene=3.0, rows=4, columns=6,
                          n_response=1000, low=0.02, high=0.97,
                          inside_unit=True, on_unit=True, normal_p=1e-30,
                          skew=1.4, response="pred")
        widget = SettingsAdvisorDialog(reading, {"regression_type": "mixed",
                                                 "inference": "nonparametric"})
        qtbot.addWidget(widget)
        return widget

    def test_it_opens_on_the_questions(self, dialog):
        assert dialog.pages.currentWidget() is dialog.questions

    def test_apply_is_not_offered_before_the_proposal(self, dialog):
        assert not dialog.apply.isVisible()

    def test_and_nothing_can_be_written_from_that_page(self, dialog):
        assert dialog.accepted_settings() == {}

    def test_seeing_the_proposal_offers_apply(self, dialog):
        dialog.show_the_proposal()

        assert dialog.pages.currentWidget() is dialog.proposal
        assert dialog.accepted_settings()

    def test_the_table_shows_the_current_value_beside_the_new_one(self,
                                                                  dialog):
        dialog.show_the_proposal()
        table = dialog.proposal.table

        headings = [table.horizontalHeaderItem(i).text()
                    for i in range(table.columnCount())]
        assert headings == ["setting", "now", "proposed", "because"]
        row = next(r for r in range(table.rowCount())
                   if table.item(r, 0).text() == "regression_type")
        assert table.item(row, 1).text() == "mixed"
        assert table.item(row, 2).text().startswith("beta")

    def test_an_unchanged_setting_is_shown_rather_than_hidden(self, dialog):
        """A proposal listing only differences reads as 'everything else is
        wrong', when most of a tuned panel is usually already right."""
        dialog.show_the_proposal()
        table = dialog.proposal.table

        row = next(r for r in range(table.rowCount())
                   if table.item(r, 0).text() == "inference")
        assert "unchanged" in table.item(row, 2).text()

    def test_every_row_carries_its_reason(self, dialog):
        dialog.show_the_proposal()
        table = dialog.proposal.table

        for row in range(table.rowCount()):
            assert len(table.item(row, 3).text()) > 30

    def test_the_undecided_are_listed_with_their_reasons(self, qtbot):
        from spacr.qt.widgets.settings_advisor_dialog import \
            SettingsAdvisorDialog

        widget = SettingsAdvisorDialog(Reading(plates=2, wells=96), {})
        qtbot.addWidget(widget)
        widget.show_the_proposal()

        said = widget.proposal.undecided.toPlainText()
        assert "not decided" in said.lower()
        assert "regression_type" in said

    def test_the_summary_says_what_was_read(self, dialog):
        dialog.show_the_proposal()

        said = dialog.proposal.summary.text()
        assert "4 plate(s)" in said
        assert "384 well(s)" in said

    def test_a_capped_read_says_so_on_the_proposal(self, qtbot):
        from spacr.qt.widgets.settings_advisor_dialog import \
            SettingsAdvisorDialog

        widget = SettingsAdvisorDialog(
            Reading(plates=1, wells=96, n_response=400_000, low=0.1,
                    high=0.9, inside_unit=True, on_unit=True, capped=True), {})
        qtbot.addWidget(widget)
        widget.show_the_proposal()

        assert "sample" in widget.proposal.summary.text()


class TestTheQuestionsPage:

    @pytest.fixture
    def page(self, qtbot):
        from spacr.qt.widgets.settings_advisor_dialog import QuestionsPage

        widget = QuestionsPage(Reading(n_response=100, wells=96))
        qtbot.addWidget(widget)
        return widget

    def test_the_maintainers_question_has_a_field(self, page):
        assert "hits_per_thousand" in page.answers()

    def test_the_answers_come_back_typed(self, page):
        got = page.answers()

        assert isinstance(got["hits_per_thousand"], int)
        assert isinstance(got["controls"], str)

    def test_a_question_the_data_answers_is_not_on_the_page(self, qtbot):
        from spacr.qt.widgets.settings_advisor_dialog import QuestionsPage

        widget = QuestionsPage(Reading(n_response=100, binary=True))
        qtbot.addWidget(widget)

        assert "direction" not in widget.answers()
