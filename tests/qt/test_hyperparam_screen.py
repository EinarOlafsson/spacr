"""Tests for the Hyperparameter search panel.

Runs offscreen with matplotlib on Agg. The search backend is always injected,
so nothing here fits a UMAP or trains a model — the panel is what is under
test: does the table fill in as trials land, does Stop actually stop and mark
the result partial, does Apply write the chosen configuration back into the
settings panel, and does bad input land as a sentence under the controls rather
than in a modal dialog that would hang a headless run forever.

The ``_no_modal_dialogs`` fixture is the same one ``tests/qt/test_db_browser.py``
uses, for the same reason.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from PySide6.QtCore import Qt

from spacr.hyperparam import SearchResult, SearchSpace, Trial
from spacr.qt.screens.hyperparam import (
    APP_PARAMS,
    HyperparamPanel,
    MAX_PANELS,
    SearchRequest,
    TOGGLE_TEXT,
    build_hyperparam_card,
    build_panel_figure,
    figure_to_pixmap,
    format_params,
    parse_values,
    searchable,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    Copied from ``tests/qt/test_db_browser.py``: a QMessageBox in a headless
    run hangs the suite forever, so every error this panel can produce must
    land on its inline status label instead.
    """
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("exec", "exec_", "open", "show"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


@pytest.fixture
def panel(qtbot, qt_theme_applied):
    """A UMAP panel, built offscreen."""
    p = HyperparamPanel("umap")
    qtbot.addWidget(p)
    return p


def make_result(*, partial=False, n=3, metric="trustworthiness",
                fold_std=None, embeddings=False):
    """A hand-built :class:`SearchResult` — no search was run to make it."""
    rng = np.random.default_rng(0)
    trials = []
    for i in range(n):
        extra = {}
        if fold_std is not None:
            extra["fold_std"] = fold_std
        if embeddings:
            extra["embedding"] = rng.normal(size=(12, 2))
        trials.append(Trial(params={"n_neighbors": 5 * (i + 1)},
                            score=0.9 - 0.01 * i, extra_metrics=extra,
                            duration=0.01, index=i))
    space = SearchSpace({"n_neighbors": [5 * (i + 1) for i in range(n)]})
    result = SearchResult(trials=trials, best=trials[0], space=space,
                          metric=metric, notes=["a note"], partial=partial)
    return result


def scripted_search(scores, *, stop_check_hook=None, fail_at=None):
    """Build an injectable search function that emits ``scores`` as trials.

    :param scores: one score per configuration; ``None`` makes that trial fail.
    :param stop_check_hook: called with the trial index before each trial.
    :param fail_at: raise from the whole search at this index.
    """
    def _search(request: SearchRequest, on_trial, should_stop):
        combos = request.space.grid()
        result = SearchResult(space=request.space, metric=request.criterion,
                              notes=["scripted"])
        for i, params in enumerate(combos):
            if stop_check_hook is not None:
                stop_check_hook(i)
            if should_stop():
                result.partial = True
                result.notes.append(
                    f"Search stopped early after {i} of {len(combos)} "
                    f"configurations.")
                break
            if fail_at is not None and i == fail_at:
                raise RuntimeError("backend exploded")
            score = scores[i % len(scores)]
            trial = Trial(params=dict(params), index=i,
                          score=score,
                          error=None if score is not None else "boom",
                          extra_metrics={"fold_std": 0.001})
            result.trials.append(trial)
            on_trial(trial, i + 1, len(combos))
        ok = [t for t in result.trials if t.ok]
        result.best = max(ok, key=lambda t: t.score) if ok else None
        return result
    return _search


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestPureHelpers:
    def test_parse_values_reads_a_comma_list(self):
        assert parse_values("5, 15, 50", "int", "n_neighbors") == [5, 15, 50]
        assert parse_values("0.0,0.1", "float", "min_dist") == [0.0, 0.1]
        assert parse_values("euclidean, cosine", "str", "metric") == [
            "euclidean", "cosine"]

    def test_parse_values_ignores_blank_entries(self):
        assert parse_values(" 5 , , 15 ,", "int", "k") == [5, 15]
        assert parse_values("", "int", "k") == []
        assert parse_values(None, "int", "k") == []

    def test_parse_values_explains_a_bad_int(self):
        with pytest.raises(ValueError) as e:
            parse_values("5, abc", "int", "n_neighbors")
        msg = str(e.value)
        assert "'n_neighbors' takes whole numbers" in msg
        assert "'abc'" in msg
        assert "5, 15, 50" in msg              # shows the shape of a good value

    def test_parse_values_explains_a_bad_float(self):
        with pytest.raises(ValueError) as e:
            parse_values("0.1, wide", "float", "min_dist")
        assert "'min_dist' takes numbers" in str(e.value)

    def test_parse_values_rejects_a_float_for_an_int_param(self):
        with pytest.raises(ValueError):
            parse_values("1.5", "int", "epochs")

    def test_format_params_sorts_keys(self):
        assert format_params({"b": 2, "a": 1}) == "a=1, b=2"

    def test_searchable_matches_the_parameter_table(self):
        assert searchable("umap")
        assert searchable("classify")
        assert searchable("ml_analyze")
        assert not searchable("mask")
        assert set(APP_PARAMS) == {"umap", "classify", "ml_analyze"}

    def test_toggle_text_is_what_the_user_asked_for(self):
        assert TOGGLE_TEXT == "Hyperparameter search"


class TestPanelFigure:
    def test_embeddings_become_small_multiples(self):
        fig = build_panel_figure(make_result(n=3, embeddings=True))
        assert fig is not None
        assert len(fig.axes) >= 3
        assert "look at them" in fig._suptitle.get_text()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_panels_are_capped(self):
        fig = build_panel_figure(make_result(n=30, embeddings=True))
        titled = [ax for ax in fig.axes if ax.get_title()]
        assert len(titled) == MAX_PANELS
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_without_embeddings_it_plots_score_versus_rank(self):
        fig = build_panel_figure(make_result(n=5, fold_std=0.05))
        assert fig is not None
        ax = fig.axes[0]
        assert ax.get_xlabel() == "rank"
        assert "within noise" in ax.get_legend().get_texts()[0].get_text()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_nothing_to_draw_returns_none(self):
        empty = SearchResult(space=SearchSpace({"a": [1]}), metric="acc")
        assert build_panel_figure(empty) is None

    def test_figure_to_pixmap_produces_an_image(self, qapp):
        fig = build_panel_figure(make_result(n=2, embeddings=True))
        pm = figure_to_pixmap(fig)
        assert not pm.isNull()
        assert pm.width() > 0 and pm.height() > 0
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_builds_offscreen_for_every_searchable_app(self, qtbot,
                                                       qt_theme_applied):
        for app_key in APP_PARAMS:
            p = HyperparamPanel(app_key)
            qtbot.addWidget(p)
            assert p.app_key == app_key
            assert p._table.columnCount() == len(HyperparamPanel.COLUMNS)
            assert p._table.rowCount() == 0
            assert p.result is None

    def test_an_unsearchable_app_is_refused_with_the_list(self, qtbot,
                                                          qt_theme_applied):
        with pytest.raises(ValueError) as e:
            HyperparamPanel("mask")
        assert "No hyperparameter search is defined" in str(e.value)
        assert "umap" in str(e.value)

    def test_default_space_is_prefilled(self, panel):
        assert panel._value_edits["n_neighbors"].text() == "5, 15, 50, 100"
        assert panel._value_edits["min_dist"].text() == "0.0, 0.1, 0.5"

    def test_criteria_combo_offers_the_apps_criteria(self, panel):
        items = [panel._criterion.itemText(i)
                 for i in range(panel._criterion.count())]
        assert items == ["trustworthiness", "continuity", "silhouette"]

    def test_folds_control_is_hidden_for_umap(self, panel, qtbot,
                                              qt_theme_applied):
        assert not panel._n_folds.isVisible()
        ml = HyperparamPanel("ml_analyze")
        qtbot.addWidget(ml)
        assert ml._n_folds.value() == 5

    def test_stop_and_apply_start_disabled(self, panel):
        assert not panel._stop_btn.isEnabled()
        assert not panel._apply_btn.isEnabled()
        assert panel._run_btn.isEnabled()

    def test_card_factory_returns_a_titled_card(self, qtbot, qt_theme_applied):
        class _Host:
            app_key = "ml_analyze"

        p, card = build_hyperparam_card(_Host())
        qtbot.addWidget(card)
        assert isinstance(p, HyperparamPanel)
        assert p.app_key == "ml_analyze"
        assert card.minimumHeight() >= 320

    def test_apply_settings_seeds_empty_fields_only(self, panel):
        panel._value_edits["metric"].setText("")
        panel.apply_settings({"metric": "cosine", "n_neighbors": 999,
                              "src": "/data"})
        assert panel._value_edits["metric"].text() == "cosine"
        # n_neighbors already had the default grid; it is not overwritten.
        assert panel._value_edits["n_neighbors"].text() == "5, 15, 50, 100"
        assert panel._settings["src"] == "/data"


# ---------------------------------------------------------------------------
# Space validation — inline, never modal
# ---------------------------------------------------------------------------

class TestInlineErrors:
    def test_bad_value_is_reported_under_the_controls(self, panel):
        panel._value_edits["n_neighbors"].setText("5, abc")
        assert panel.run_search() is False
        assert "takes whole numbers" in panel._status.text()
        assert "'abc'" in panel._status.text()
        assert panel._table.rowCount() == 0
        assert panel._run_btn.isEnabled()          # still usable

    def test_empty_space_is_reported_inline(self, panel):
        for edit in panel._value_edits.values():
            edit.setText("")
        assert panel.run_search() is False
        assert "Nothing to search" in panel._status.text()
        assert "at least one parameter" in panel._status.text()

    def test_current_space_drops_empty_fields(self, panel):
        panel._value_edits["min_dist"].setText("")
        panel._value_edits["metric"].setText("")
        space = panel.current_space()
        assert set(space.params) == {"n_neighbors"}

    def test_apply_without_a_selection_is_reported_inline(self, panel):
        assert panel.apply_selected() is False
        assert "no configuration to apply" in panel._status.text()

    def test_stop_with_nothing_running_is_reported_inline(self, panel):
        panel.stop_search()
        assert panel._status.text() == "No search is running."

    def test_apply_without_a_callback_is_reported_inline(self, panel, qtbot):
        panel._result = make_result()
        panel._rebuild_table(panel._result)
        assert panel.apply_selected() is False
        assert "not attached to a settings panel" in panel._status.text()

    def test_a_raising_apply_callback_is_reported_inline(self, panel):
        panel._result = make_result()
        panel._rebuild_table(panel._result)
        panel.set_apply_callback(
            lambda cfg: (_ for _ in ()).throw(RuntimeError("no such key")))
        assert panel.apply_selected() is False
        assert "Could not apply the configuration" in panel._status.text()
        assert "no such key" in panel._status.text()

    def test_a_backend_crash_is_reported_inline(self, panel, qtbot):
        """The whole search blowing up lands on the status label, not in a
        dialog and not as an unhandled exception in the worker thread."""
        panel._value_edits["n_neighbors"].setText("5, 15")
        panel._value_edits["min_dist"].setText("")
        panel._value_edits["metric"].setText("")
        panel.set_search_fn(scripted_search([0.9], fail_at=0))
        assert panel.run_search() is True
        qtbot.waitUntil(
            lambda: panel._status.text().startswith("Search failed"),
            timeout=5000)
        assert "backend exploded" in panel._status.text()
        assert panel._run_btn.isEnabled()
        assert not panel._stop_btn.isEnabled()
        assert panel.result is None


# ---------------------------------------------------------------------------
# Running a (mocked) search
# ---------------------------------------------------------------------------

class TestRunningASearch:
    def _prep(self, panel, values="5, 15, 50, 100"):
        panel._value_edits["n_neighbors"].setText(values)
        panel._value_edits["min_dist"].setText("")
        panel._value_edits["metric"].setText("")

    def test_a_mocked_search_fills_the_table(self, panel, qtbot):
        self._prep(panel)
        panel.set_search_fn(scripted_search([0.9, 0.8, 0.7, 0.6]))
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            assert panel.run_search() is True
        assert panel._table.rowCount() == 4
        assert panel.result.ok
        assert panel.result.best.params == {"n_neighbors": 5}
        assert panel._table.item(0, 1).text() == "0.9000"
        assert panel._table.item(0, 3).text() == "n_neighbors=5"

    def test_the_table_fills_in_while_the_search_is_still_running(
            self, panel, qtbot):
        """Rows must appear as trials land, not only in the end-of-run rebuild.

        The scripted backend parks on a gate after two trials; while it is
        parked the sweep is demonstrably unfinished, and the table already has
        those two rows.
        """
        gate = threading.Event()
        released = threading.Event()

        def hook(i):
            if i == 2:
                released.set()
                gate.wait(5.0)

        self._prep(panel)
        panel.set_search_fn(
            scripted_search([0.5, 0.6, 0.7, 0.8], stop_check_hook=hook))
        panel.run_search()
        qtbot.waitUntil(released.is_set, timeout=5000)
        qtbot.waitUntil(lambda: panel._table.rowCount() == 2, timeout=5000)

        assert panel.result is None                 # the sweep is not done
        assert panel._stop_btn.isEnabled()
        assert panel._table.item(0, 3).text() == "n_neighbors=5"
        assert panel._table.item(1, 3).text() == "n_neighbors=15"

        gate.set()
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            pass
        assert panel._table.rowCount() == 4

    def test_progress_lands_on_the_status_label(self, panel, qtbot):
        self._prep(panel, "5, 15")
        texts = []
        panel.set_search_fn(scripted_search([0.5, 0.6]))
        panel._status.linkActivated  # touch, no-op
        original = panel._on_trial_ready

        def spy(trial, done, total):
            original(trial, done, total)
            texts.append(panel._status.text())

        panel._on_trial_ready = spy
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        assert texts == ["1 of 2 configurations evaluated",
                         "2 of 2 configurations evaluated"]

    def test_the_table_is_rebuilt_in_ranked_order(self, panel, qtbot):
        self._prep(panel)
        panel.set_search_fn(scripted_search([0.1, 0.9, 0.5, 0.7]))
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        scores = [float(panel._table.item(r, 1).text())
                  for r in range(panel._table.rowCount())]
        assert scores == sorted(scores, reverse=True)
        assert panel._table.item(0, 0).text() == "1"

    def test_failed_trials_land_at_the_bottom_with_their_error(self, panel,
                                                               qtbot):
        self._prep(panel)
        panel.set_search_fn(scripted_search([0.9, None, 0.7, None]))
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        last = panel._table.rowCount() - 1
        assert panel._table.item(last, 4).text() == "boom"
        assert panel._table.item(last, 1).text() == "-"
        assert "2 trials failed" in panel._status.text()

    def test_the_summary_reports_the_spread(self, panel, qtbot):
        self._prep(panel)
        panel.set_search_fn(scripted_search([0.9, 0.5, 0.7, 0.6]))
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        assert "Best trustworthiness=0.9000" in panel._status.text()
        assert "spread over 4 trials" in panel._status.text()

    def test_the_within_noise_flag_reaches_the_status_line(self, panel, qtbot):
        """Four scores inside the fold standard deviation: the panel says the
        winner is arbitrary instead of quietly showing rank 1."""
        self._prep(panel)
        panel.set_search_fn(scripted_search([0.9000, 0.8999, 0.8998, 0.8997]))
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        assert "WITHIN NOISE" in panel._status.text()
        assert "arbitrary" in panel._status.text()

    def test_notes_are_shown(self, panel, qtbot):
        self._prep(panel, "5")
        panel.set_search_fn(scripted_search([0.9]))
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        assert "• scripted" in panel._notes.text()

    def test_a_preview_image_is_drawn(self, panel, qtbot):
        self._prep(panel)
        panel.set_search_fn(scripted_search([0.9, 0.5, 0.7, 0.6]))
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        pm = panel._preview.pixmap()
        assert pm is not None and not pm.isNull()

    def test_a_second_run_clears_the_previous_table(self, panel, qtbot):
        self._prep(panel)
        panel.set_search_fn(scripted_search([0.9, 0.5, 0.7, 0.6]))
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        self._prep(panel, "5, 15")
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        assert panel._table.rowCount() == 2

    def test_the_request_carries_the_controls(self, panel, qtbot):
        self._prep(panel, "5, 15")
        captured = {}

        def _search(request, on_trial, should_stop):
            captured["request"] = request
            return SearchResult(space=request.space, metric=request.criterion)

        panel._criterion.setCurrentText("continuity")
        panel._mode.setCurrentText("random")
        panel._n_trials.setValue(7)
        panel._seed.setValue(11)
        panel.apply_settings({"src": "/data"})
        panel.set_search_fn(_search)
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        req = captured["request"]
        assert req.app_key == "umap"
        assert req.criterion == "continuity"
        assert req.mode == "random"
        assert req.n_trials == 7
        assert req.seed == 11
        assert req.settings["src"] == "/data"
        assert req.space.params["n_neighbors"] == (5, 15)


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------

class TestStopping:
    def test_stop_marks_the_result_partial_and_keeps_finished_trials(
            self, panel, qtbot):
        """Stop is pressed from the GUI thread while the worker is between
        trials; the trials already done survive and the result says partial."""
        gate = threading.Event()
        released = threading.Event()

        def hook(i):
            if i == 2:
                released.set()
                gate.wait(5.0)

        panel._value_edits["n_neighbors"].setText("5, 15, 50, 100")
        panel._value_edits["min_dist"].setText("")
        panel._value_edits["metric"].setText("")
        panel.set_search_fn(
            scripted_search([0.9, 0.8, 0.7, 0.6], stop_check_hook=hook))

        panel.run_search()
        qtbot.waitUntil(released.is_set, timeout=5000)
        panel.stop_search()
        gate.set()
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            pass

        result = panel.result
        assert result.partial is True
        assert len(result.trials) == 2
        assert "PARTIAL" in panel._status.text()
        assert "not a completed sweep" in panel._status.text()
        assert panel._table.rowCount() == 2
        assert panel._stop_btn.isEnabled() is False
        assert panel._run_btn.isEnabled() is True

    def test_stopping_says_so_immediately(self, panel, qtbot):
        gate = threading.Event()
        released = threading.Event()

        def hook(i):
            if i == 1:
                released.set()
                gate.wait(5.0)

        panel._value_edits["n_neighbors"].setText("5, 15, 50")
        panel._value_edits["min_dist"].setText("")
        panel._value_edits["metric"].setText("")
        panel.set_search_fn(scripted_search([0.9, 0.8, 0.7],
                                            stop_check_hook=hook))
        panel.run_search()
        qtbot.waitUntil(released.is_set, timeout=5000)
        panel.stop_search()
        assert "marked partial" in panel._status.text()
        gate.set()
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            pass

    def test_a_second_run_while_one_is_going_is_refused_inline(self, panel,
                                                              qtbot):
        gate = threading.Event()
        released = threading.Event()

        def hook(i):
            if i == 1:
                released.set()
                gate.wait(5.0)

        panel._value_edits["n_neighbors"].setText("5, 15, 50")
        panel._value_edits["min_dist"].setText("")
        panel._value_edits["metric"].setText("")
        panel.set_search_fn(scripted_search([0.9, 0.8, 0.7],
                                            stop_check_hook=hook))
        panel.run_search()
        qtbot.waitUntil(released.is_set, timeout=5000)
        assert panel.run_search() is False
        assert "already running" in panel._status.text()
        gate.set()
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            pass

    def test_closing_mid_search_does_not_abort_the_process(self, panel, qtbot):
        panel._value_edits["n_neighbors"].setText("5, 15")
        panel._value_edits["min_dist"].setText("")
        panel._value_edits["metric"].setText("")
        panel.set_search_fn(scripted_search([0.9, 0.8]))
        panel.run_search()
        panel.close()
        assert True          # reaching here means the thread was joined


# ---------------------------------------------------------------------------
# Applying a configuration back into the settings panel
# ---------------------------------------------------------------------------

class TestApply:
    def _run(self, panel, qtbot, scores=(0.5, 0.9, 0.7, 0.6)):
        panel._value_edits["n_neighbors"].setText("5, 15, 50, 100")
        panel._value_edits["min_dist"].setText("")
        panel._value_edits["metric"].setText("")
        panel.set_search_fn(scripted_search(list(scores)))
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()

    def test_apply_writes_the_selected_configuration_back(self, panel, qtbot):
        written = {}
        panel.set_apply_callback(written.update)
        self._run(panel, qtbot)
        assert panel._apply_btn.isEnabled()
        assert panel.apply_selected() is True
        # Row 0 is the winner (0.9 -> the second configuration, n_neighbors=15).
        assert written == {"n_neighbors": 15}
        assert "Applied n_neighbors=15" in panel._status.text()

    def test_selecting_a_different_row_applies_that_row(self, panel, qtbot):
        written = {}
        panel.set_apply_callback(written.update)
        self._run(panel, qtbot)
        panel._table.selectRow(2)
        applied = panel.selected_params()
        assert panel.apply_selected() is True
        assert written == applied
        assert written != {"n_neighbors": 15}

    def test_applying_a_partial_sweep_warns(self, panel, qtbot):
        written = {}
        panel.set_apply_callback(written.update)
        gate = threading.Event()
        released = threading.Event()

        def hook(i):
            if i == 2:
                released.set()
                gate.wait(5.0)

        panel._value_edits["n_neighbors"].setText("5, 15, 50, 100")
        panel._value_edits["min_dist"].setText("")
        panel._value_edits["metric"].setText("")
        panel.set_search_fn(
            scripted_search([0.5, 0.9, 0.7, 0.6], stop_check_hook=hook))
        panel.run_search()
        qtbot.waitUntil(released.is_set, timeout=5000)
        panel.stop_search()
        gate.set()
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            pass

        assert panel.apply_selected() is True
        assert written
        assert "partial sweep" in panel._status.text()
        assert "may be better" in panel._status.text()

    def test_the_apply_callback_matches_the_settings_panel_contract(
            self, panel, qtbot, qt_theme_applied):
        """The callback the AppScreen registers is
        ``SettingsWidgets.set_value_for_key``; the parameter names the panel
        emits must be real settings keys for that app."""
        from spacr.qt.screens.settings_model import SettingsWidgets
        widgets = SettingsWidgets("umap")
        widgets.build_sections()
        for key, _label, _kind in APP_PARAMS["umap"]:
            assert key in widgets._widgets, key

        written = {}
        panel.set_apply_callback(
            lambda cfg: [written.__setitem__(k, v) for k, v in cfg.items()
                         if widgets.set_value_for_key(k, v)])
        self._run(panel, qtbot)
        assert panel.apply_selected() is True
        assert written == {"n_neighbors": 15}
        assert widgets.collect()["n_neighbors"] == 15

    def test_classify_and_ml_parameters_are_real_settings_keys(self):
        from spacr.qt.screens.settings_model import resolve_default_settings
        for app_key in ("classify", "ml_analyze"):
            defaults = resolve_default_settings(app_key)
            for key, _label, _kind in APP_PARAMS[app_key]:
                assert key in defaults, f"{app_key}.{key}"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class TestWorker:
    def test_request_stop_flips_the_flag(self, qtbot):
        from spacr.qt.screens.hyperparam import _SearchWorker
        w = _SearchWorker(SearchRequest(space=SearchSpace({"a": [1]})))
        assert w.stopped is False
        w.request_stop()
        assert w.stopped is True

    def test_selected_params_falls_back_to_the_best(self, panel, qtbot):
        panel._result = make_result()
        assert panel.selected_params() == {"n_neighbors": 5}

    def test_selected_params_is_none_without_a_result(self, panel):
        assert panel.selected_params() is None
