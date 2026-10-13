"""The hyperparameter search card on the Activation screen.

Attribution has no ground truth, so the point of this panel is not to name a
winner: it is to show four criteria that measure different things, to say
plainly when they pick different configurations, and to put the maps in front of
the user. These tests pin that behaviour.

Nothing here trains, downloads or attributes anything real — the search backend
is injected, exactly as ``tests/qt/test_hyperparam_screen.py`` injects it, and
the attribution maps in the mocked results are small numpy arrays. The
``_no_modal_dialogs`` fixture is the one ``tests/qt/test_db_browser.py`` uses,
for the same reason: a QMessageBox in a headless run hangs the suite forever.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from PySide6.QtCore import Qt

from spacr.hyperparam import (
    ACTIVATION_CRITERIA,
    ACTIVATION_NO_GROUND_TRUTH,
    APP_CRITERIA,
    DEFAULT_SPACES,
    LOWER_IS_BETTER,
    SearchResult,
    SearchSpace,
    Trial,
)
from spacr.qt.screens.hyperparam import (
    APP_PARAMS,
    HyperparamPanel,
    SearchRequest,
    build_hyperparam_card,
    build_panel_figure,
    criteria_disagree,
    figure_to_pixmap,
    format_scores,
    searchable,
)


IMG = 8

#: The four numbers every Activation trial must carry.
SCORE_KEYS = ("deletion_auc", "insertion_auc", "pointing_game", "sanity_gap")


def _col(panel, name: str) -> int:
    """Column index by NAME.

    Hard-coded indices broke on 2026-08-09 (13a7b335, "The Walk was climbing;
    the table was showing the wrong thing"), which inserted a "best so far"
    column between "score" and "fold sd" and pushed "parameters" from 3 to 4.
    The names are stable; the positions never were. Same helper, same reason,
    as ``tests/qt/test_hyperparam_screen.py``.
    """
    return panel.COLUMNS.index(name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    Copied from ``tests/qt/test_db_browser.py``: a QMessageBox in a headless run
    hangs the suite forever, so every error this panel can produce must land on
    its inline status label instead.
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
    """An Activation search panel, built offscreen."""
    p = HyperparamPanel("activation")
    qtbot.addWidget(p)
    return p


class FakeAttribution:
    """The shape :func:`build_panel_figure` needs from an attribution."""

    def __init__(self, seed=0):
        """Build a small deterministic map."""
        self.map = np.random.default_rng(seed).random((IMG, IMG))
        self.method = "gradcam"


#: Scores per method, chosen so the criteria pick DIFFERENT winners: gradcam
#: wins on deletion (lowest), saliency on insertion, occlusion on the pointing
#: game and the sanity check. That disagreement is the thing under test.
SCRIPTED_SCORES = {
    "gradcam": {"deletion_auc": 0.10, "insertion_auc": 0.55,
                "pointing_game": 0.50, "sanity_gap": 0.30},
    "saliency": {"deletion_auc": 0.30, "insertion_auc": 0.90,
                 "pointing_game": 0.60, "sanity_gap": 0.05},
    "occlusion": {"deletion_auc": 0.25, "insertion_auc": 0.70,
                  "pointing_game": 1.00, "sanity_gap": 0.95},
}


def scripted_search(scores=None, *, keep_maps=True, fail_at=None):
    """An injectable search function that scores by ``cam_type`` from a table.

    :param scores: ``{cam_type: {criterion: value}}``; defaults to
        :data:`SCRIPTED_SCORES`.
    :param keep_maps: attach a fake attribution map to every trial.
    :param fail_at: raise from the whole search at this trial index.
    """
    table = dict(scores or SCRIPTED_SCORES)

    def _search(request: SearchRequest, on_trial, should_stop):
        """Emit one trial per configuration, carrying every criterion."""
        combos = request.space.grid()
        result = SearchResult(
            space=request.space, metric=request.criterion,
            higher_is_better=request.criterion not in LOWER_IS_BETTER,
            notes=[ACTIVATION_NO_GROUND_TRUTH,
                   f"Criterion '{request.criterion}': "
                   f"{ACTIVATION_CRITERIA[request.criterion]}"])
        for i, params in enumerate(combos):
            if should_stop():
                result.partial = True
                break
            if fail_at is not None and i == fail_at:
                raise RuntimeError("attribution backend exploded")
            extra = dict(table.get(str(params.get("cam_type")),
                                   SCRIPTED_SCORES["gradcam"]))
            extra["sanity_passed"] = extra["sanity_gap"] > 0.5
            extra["sanity_verdict"] = (
                f"{params.get('cam_type')} "
                f"{'PASSES' if extra['sanity_passed'] else 'FAILS'} the "
                f"randomisation sanity check")
            extra["n_images"] = 4
            if keep_maps:
                extra["attribution"] = FakeAttribution(seed=i)
            trial = Trial(params=dict(params), index=i,
                          score=extra[request.criterion], extra_metrics=extra)
            result.trials.append(trial)
            on_trial(trial, i + 1, len(combos))
        ok = [t for t in result.trials if t.ok]
        if ok:
            result.best = (max(ok, key=lambda t: t.score)
                           if result.higher_is_better
                           else min(ok, key=lambda t: t.score))
        return result
    return _search


def make_result(criterion="deletion_auc", methods=("gradcam", "saliency",
                                                   "occlusion"),
                keep_maps=True):
    """A hand-built Activation :class:`SearchResult` — no search was run."""
    trials = []
    for i, method in enumerate(methods):
        extra = dict(SCRIPTED_SCORES[method])
        extra["sanity_verdict"] = f"{method} verdict"
        if keep_maps:
            extra["attribution"] = FakeAttribution(seed=i)
        trials.append(Trial(params={"cam_type": method}, index=i,
                            score=extra[criterion], extra_metrics=extra,
                            duration=0.01))
    higher = criterion not in LOWER_IS_BETTER
    best = (max(trials, key=lambda t: t.score) if higher
            else min(trials, key=lambda t: t.score))
    return SearchResult(trials=trials, best=best,
                        space=SearchSpace({"cam_type": list(methods)}),
                        metric=criterion, higher_is_better=higher,
                        notes=[ACTIVATION_NO_GROUND_TRUTH])


def _prep(panel, methods="gradcam, saliency, occlusion"):
    """Fill only the method field, leaving the rest of the space empty."""
    for key, _label, _kind in APP_PARAMS["activation"]:
        panel._value_edits[key].setText("")
    panel._value_edits["cam_type"].setText(methods)


# ---------------------------------------------------------------------------
# Registration — the card only appears because 'activation' is in APP_PARAMS
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_activation_is_searchable(self):
        assert searchable("activation") is True
        assert "activation" in APP_PARAMS
        assert "activation" in APP_CRITERIA
        assert "activation" in DEFAULT_SPACES

    def test_the_app_screen_gate_turns_the_card_on(self):
        """``AppScreen`` asks exactly one question, and it is this one.

        Nothing in ``app_screen.py`` needed changing: adding the app to
        ``APP_PARAMS`` is what makes ``_hyperparam_searchable`` say yes.
        """
        from spacr.qt.screens.app_screen import _hyperparam_searchable
        assert _hyperparam_searchable("activation") is True
        assert _hyperparam_searchable("mask") is False

    def test_the_swept_parameters_are_the_ones_that_change_the_map(self):
        keys = [k for k, _l, _kind in APP_PARAMS["activation"]]
        assert keys == ["cam_type", "target_layer", "smoothgrad_samples",
                        "smoothgrad_sigma", "occlusion_window",
                        "occlusion_stride", "ig_steps", "ig_baseline"]

    def test_every_criterion_is_documented_and_pointed_the_right_way(self):
        assert APP_CRITERIA["activation"] == list(SCORE_KEYS)
        for name in SCORE_KEYS:
            assert ACTIVATION_CRITERIA[name]
        assert "deletion_auc" in LOWER_IS_BETTER
        assert "insertion_auc" not in LOWER_IS_BETTER
        assert "pointing_game" not in LOWER_IS_BETTER
        assert "sanity_gap" not in LOWER_IS_BETTER

    def test_the_default_grid_spans_more_than_one_family(self):
        from spacr.attribution import ATTRIBUTION_METHODS
        methods = DEFAULT_SPACES["activation"]["cam_type"]
        families = {ATTRIBUTION_METHODS[m].family for m in methods}
        assert len(families) >= 3, families

    def test_the_card_builds_for_the_activation_screen(self, qtbot,
                                                       qt_theme_applied):
        class _Host:
            """Minimal stand-in for the AppScreen the card asks for app_key."""

            app_key = "activation"

        p, card = build_hyperparam_card(_Host())
        qtbot.addWidget(card)
        assert p.app_key == "activation"
        assert card.title() if hasattr(card, "title") else True

    def test_the_panel_offers_every_criterion_in_order(self, panel):
        items = [panel._criterion.itemText(i)
                 for i in range(panel._criterion.count())]
        assert items == APP_CRITERIA["activation"]

    def test_the_fold_control_is_hidden_because_nothing_is_cross_validated(
            self, panel):
        assert panel._n_folds.isVisible() is False


# ---------------------------------------------------------------------------
# A mocked sweep
# ---------------------------------------------------------------------------

class TestMockedSweep:
    def test_a_mocked_sweep_fills_the_table(self, panel, qtbot):
        """Every column of the winning row, looked up by name.

        The row is the whole point of the table, so this covers all five
        visible cells rather than two of them. When "best so far" was inserted
        on 2026-08-09 (13a7b335) the old ``item(0, 3)`` silently started
        reading "fold sd" — an assertion that moved to a different column
        without saying so is the failure mode this shape prevents.
        """
        _prep(panel)
        panel.set_search_fn(scripted_search())
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            assert panel.run_search() is True
        assert panel._table.rowCount() == 3
        assert panel.result.ok

        def cell(row, name):
            return panel._table.item(row, _col(panel, name)).text()

        assert cell(0, "#") == "1"
        assert cell(0, "parameters") == "cam_type=gradcam"
        assert cell(0, "score") == "0.1000"
        assert cell(0, "status") == "ok"
        # "best so far" is a LIVE column: it shows the running best while
        # trials arrive out of order, and the ranked rebuild that follows the
        # sweep blanks it, because rows sorted best-first already say it.
        assert cell(0, "best so far") == "-"
        # Attribution scores one map per configuration, so there are no folds
        # to take a standard deviation over and "fold sd" is structurally
        # empty — the same reason the fold control is hidden for this app.
        assert cell(0, "fold sd") == "-"
        # Ranked best-first: gradcam wins deletion_auc (lowest), then
        # occlusion 0.25, then saliency 0.30.
        assert [cell(row, "parameters") for row in range(3)] == [
            "cam_type=gradcam", "cam_type=occlusion", "cam_type=saliency",
        ]

    def test_deletion_auc_is_minimised_not_maximised(self, panel, qtbot):
        """The one criterion here where a small number wins.

        Ranking it the wrong way puts the least faithful map on top, which is
        worse than not ranking at all.
        """
        _prep(panel)
        panel._criterion.setCurrentText("deletion_auc")
        panel.set_search_fn(scripted_search())
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        assert panel.result.higher_is_better is False
        assert panel.result.best.params == {"cam_type": "gradcam"}

    def test_switching_criterion_switches_the_winner(self, panel, qtbot):
        _prep(panel)
        panel._criterion.setCurrentText("pointing_game")
        panel.set_search_fn(scripted_search())
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        assert panel.result.higher_is_better is True
        assert panel.result.best.params == {"cam_type": "occlusion"}

    def test_every_trial_carries_every_score(self, panel, qtbot):
        """A row that shows only the ranked criterion hides the finding."""
        _prep(panel)
        panel.set_search_fn(scripted_search())
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        for trial in panel.result.trials:
            for key in SCORE_KEYS:
                assert key in trial.extra_metrics, key
                assert isinstance(trial.extra_metrics[key], float)
            assert "sanity_verdict" in trial.extra_metrics

    def test_the_scores_reach_the_row_tooltip_and_the_stashed_metrics(
            self, panel, qtbot):
        _prep(panel)
        panel.set_search_fn(scripted_search())
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        item = panel._table.item(0, 0)
        tip = item.toolTip()
        for key in SCORE_KEYS:
            assert key in tip, (key, tip)
        assert "randomisation sanity check" in tip
        stashed = item.data(Qt.UserRole + 1)
        assert set(SCORE_KEYS).issubset(stashed)

    def test_the_status_says_the_criteria_disagree(self, panel, qtbot):
        _prep(panel)
        panel.set_search_fn(scripted_search())
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        status = panel._status.text()
        assert "THE CRITERIA DISAGREE" in status
        assert "Look at the maps" in status

    def test_the_no_ground_truth_caveat_is_shown(self, panel, qtbot):
        _prep(panel)
        panel.set_search_fn(scripted_search())
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        assert "no ground truth" in panel._notes.text()

    def test_applying_a_row_writes_the_method_back(self, panel, qtbot):
        _prep(panel)
        written = {}
        panel.set_apply_callback(written.update)
        panel.set_search_fn(scripted_search())
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        assert panel.apply_selected() is True
        assert written == {"cam_type": "gradcam"}

    def test_a_backend_failure_lands_inline(self, panel, qtbot):
        _prep(panel)
        panel.set_search_fn(scripted_search(fail_at=0))
        assert panel.run_search() is True
        qtbot.waitUntil(
            lambda: panel._status.text().startswith("Search failed"),
            timeout=5000)
        assert "attribution backend exploded" in panel._status.text()
        assert panel._run_btn.isEnabled()
        assert panel.result is None

    def test_an_empty_space_lands_inline_and_starts_nothing(self, panel):
        for key, _label, _kind in APP_PARAMS["activation"]:
            panel._value_edits[key].setText("")
        assert panel.run_search() is False
        assert "Nothing to search" in panel._status.text()

    def test_a_typo_in_a_numeric_field_lands_inline(self, panel):
        _prep(panel)
        panel._value_edits["ig_steps"].setText("fifty")
        assert panel.run_search() is False
        assert "whole numbers" in panel._status.text()

    def test_the_worker_is_not_scheduled_for_deletion(self, panel, qtbot):
        """``worker.deleteLater`` here segfaulted; the relay is a bound method.

        See ``spacr.qt.bridge.make_thread`` for the measured account. This pins
        the connection so it cannot be reintroduced by accident.
        """
        _prep(panel, "gradcam")
        panel.set_search_fn(scripted_search())
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        import inspect
        code = [line.split("#", 1)[0]
                for line in inspect.getsource(
                    HyperparamPanel.run_search).splitlines()]
        assert not any("deleteLater" in line for line in code)
        assert any("worker.finished.connect(self._on_worker_finished)" in line
                   for line in code)
        assert panel._run_btn.isEnabled()
        assert not panel._stop_btn.isEnabled()


# ---------------------------------------------------------------------------
# The panel of maps — the actual deliverable
# ---------------------------------------------------------------------------

class TestSmallMultiples:
    def test_the_maps_become_small_multiples(self):
        fig = build_panel_figure(make_result())
        assert fig is not None
        assert len(fig.axes) >= 3
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert any("deletion_auc" in t for t in titles)
        assert any("insertion_auc" in t for t in titles)
        assert any("pointing_game" in t for t in titles)
        assert any("sanity_gap" in t for t in titles)

    def test_the_figure_says_which_way_each_criterion_points(self):
        fig = build_panel_figure(make_result())
        assert "LOW" in fig._suptitle.get_text()

    def test_the_figure_renders_to_a_pixmap(self, qt_theme_applied):
        fig = build_panel_figure(make_result())
        pixmap = figure_to_pixmap(fig)
        assert not pixmap.isNull()

    def test_a_sweep_without_maps_falls_back_to_the_score_plot(self):
        fig = build_panel_figure(make_result(keep_maps=False))
        assert fig is not None
        assert fig.axes[0].get_ylabel() == "deletion_auc"

    def test_the_preview_is_drawn_after_a_sweep(self, panel, qtbot):
        _prep(panel)
        panel.set_search_fn(scripted_search())
        with qtbot.waitSignal(panel.search_finished, timeout=5000):
            panel.run_search()
        assert panel._preview.pixmap() is not None
        assert not panel._preview.pixmap().isNull()


# ---------------------------------------------------------------------------
# The pure helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_format_scores_leads_with_the_named_criteria(self):
        trial = make_result().trials[0]
        text = format_scores(trial, APP_CRITERIA["activation"])
        lines = text.splitlines()
        assert lines[0].startswith("deletion_auc")
        assert lines[1].startswith("insertion_auc")
        assert lines[-1] == "gradcam verdict"

    def test_format_scores_survives_a_trial_with_no_metrics(self):
        assert format_scores(Trial(params={}, score=1.0)) == ""

    def test_criteria_disagree_reports_each_criterion_s_winner(self):
        message = criteria_disagree(make_result(), APP_CRITERIA["activation"])
        assert message is not None
        assert "deletion_auc -> cam_type=gradcam" in message
        assert "insertion_auc -> cam_type=saliency" in message
        assert "pointing_game -> cam_type=occlusion" in message

    def test_agreement_between_criteria_is_not_announced_as_disagreement(self):
        agreeing = {m: {"deletion_auc": 0.1 * i, "insertion_auc": 1.0 - 0.1 * i,
                        "pointing_game": 1.0 - 0.1 * i,
                        "sanity_gap": 1.0 - 0.1 * i}
                    for i, m in enumerate(("gradcam", "saliency", "occlusion"))}
        trials = [Trial(params={"cam_type": m}, index=i,
                        score=agreeing[m]["deletion_auc"],
                        extra_metrics=agreeing[m])
                  for i, m in enumerate(agreeing)]
        result = SearchResult(trials=trials, best=trials[0],
                              metric="deletion_auc", higher_is_better=False)
        assert criteria_disagree(result, APP_CRITERIA["activation"]) is None

    def test_criteria_disagree_needs_two_scored_trials(self):
        result = make_result(methods=("gradcam",))
        assert criteria_disagree(result, APP_CRITERIA["activation"]) is None


# ---------------------------------------------------------------------------
# The backend the panel calls, with the attribution stubbed out
# ---------------------------------------------------------------------------

class TestSearchBackend:
    def test_run_search_for_app_routes_activation_to_the_attribution_sweep(
            self, monkeypatch):
        """No model is loaded and nothing is attributed — both are injected."""
        import torch
        import torch.nn as nn

        from spacr.hyperparam import ActivationSearchData, run_search_for_app

        class Corner(nn.Module):
            """A model whose logit depends only on the top-left corner."""

            def __init__(self):
                """Wire the weights by hand; nothing is trained."""
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3, padding=1)
                self.head = nn.Linear(1, 1)
                with torch.no_grad():
                    self.conv.weight.zero_()
                    self.conv.weight[0, 0, 1, 1] = 1.0
                    self.conv.bias.zero_()
                    self.head.weight.fill_(8.0)
                    self.head.bias.zero_()

            def forward(self, x):
                """Score the top-left 3x3 corner only."""
                feat = self.conv(x)
                mask = torch.zeros_like(feat)
                mask[:, :, :3, :3] = 1.0
                return self.head((feat * mask).mean(dim=(2, 3)))

        image = torch.zeros(1, IMG, IMG)
        image[0, :3, :3] = 2.0
        mask = np.zeros((IMG, IMG), dtype=bool)
        mask[:3, :3] = True
        data = ActivationSearchData(model=Corner(), images=[image],
                                    masks=[mask], filenames=["synthetic"],
                                    model_type="corner")

        result = run_search_for_app(
            "activation", {}, SearchSpace({"cam_type": ["saliency",
                                                        "gradcam"]}),
            criterion="deletion_auc", data=data)
        assert result.ok
        assert result.higher_is_better is False
        if importlib.util.find_spec("torchcam") is None:
            assert len(result.successful) == 1
            assert len(result.failed) == 1
            assert result.failed[0].params["cam_type"] == "gradcam"
            assert "spacr[attribution]" in result.failed[0].error
        else:
            assert len(result.successful) == 2
        for trial in result.successful:
            for key in ("deletion_auc", "insertion_auc", "pointing_game",
                        "sanity_gap"):
                assert key in trial.extra_metrics, key
            assert "attribution" in trial.extra_metrics
        assert any("no ground truth" in n for n in result.notes)

    def test_the_spacr_setting_names_map_onto_the_method_arguments(self):
        from spacr.hyperparam import _activation_params

        method, kw, n_samples, sigma = _activation_params({
            "cam_type": "occlusion", "target_layer": "features.4",
            "occlusion_window": 12, "occlusion_stride": 6, "ig_steps": 40,
            "ig_baseline": "blur", "smoothgrad_samples": 8,
            "smoothgrad_sigma": 0.2})
        assert method == "occlusion"
        assert kw == {"layer": "features.4", "window": 12, "stride": 6,
                      "n_steps": 40, "baseline": "blur"}
        assert (n_samples, sigma) == (8, 0.2)

    def test_the_legacy_saliency_names_still_sweep(self):
        from spacr.hyperparam import _activation_params

        for legacy in ("saliency_image", "saliency_channel"):
            method, kw, _n, _s = _activation_params({"cam_type": legacy})
            assert method == "saliency"
            assert kw == {}

    def test_an_empty_target_layer_is_not_passed_through_as_a_layer_name(self):
        from spacr.hyperparam import _activation_params

        for empty in (None, "", "None"):
            _m, kw, _n, _s = _activation_params({"cam_type": "gradcam",
                                                 "target_layer": empty})
            assert "layer" not in kw

    def test_an_unavailable_criterion_fails_the_trial_with_the_reason(self):
        """Ranking by the pointing game without masks must say why, not crash."""
        import torch
        import torch.nn as nn

        from spacr.hyperparam import ActivationSearchData, activation_search

        model = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.ReLU(),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                              nn.Linear(2, 2))
        data = ActivationSearchData(model=model,
                                    images=[torch.randn(1, IMG, IMG)],
                                    masks=None)
        result = activation_search(data, SearchSpace({"cam_type": ["saliency"]}),
                                   criterion="pointing_game", n_steps=3,
                                   run_sanity_check=False)
        assert not result.ok
        assert "no object masks" in result.failed[0].error
        assert any("pointing game could not be scored" in n
                   for n in result.notes)


# -------------------------------------------------------------------------
# Loading the images and the answer key
# -------------------------------------------------------------------------

class TestLoadingTheData:
    """``load_activation_data`` — where the images and the answer key come from."""

    @staticmethod
    def _model_file(tmp_path):
        """Save a tiny classifier the loader can torch.load."""
        import torch
        import torch.nn as nn

        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                              nn.Linear(4, 2))
        path = tmp_path / "model.pth"
        torch.save(model, path)
        return str(path)

    def test_merged_arrays_supply_the_images_and_the_object_masks(self,
                                                                 tmp_path):
        """The pointing game's answer key is already on disk.

        spaCR's ``merged/*.npy`` stores the image channels and then one integer
        label plane per object class, so the mask is exactly aligned to the
        image with no re-cropping and no database round trip.
        """
        from spacr.hyperparam import load_activation_data

        merged = tmp_path / "merged"
        merged.mkdir()
        rng = np.random.default_rng(0)
        for i in range(2):
            arr = np.zeros((12, 12, 8), dtype=np.float32)
            arr[..., :4] = rng.random((12, 12, 4))
            arr[3:8, 3:8, 4] = i + 1            # cell label plane
            np.save(merged / f"plate1_A01_f{i}.npy", arr)

        data = load_activation_data(
            {"src": str(tmp_path), "model_path": self._model_file(tmp_path),
             "image_size": 12, "channels": [0, 1, 2], "model_type": "resnet18"},
            n_images=2)
        assert len(data.images) == 2
        assert data.masks is not None and len(data.masks) == 2
        assert data.masks[0].shape == (12, 12)
        assert data.masks[0].any() and not data.masks[0].all()
        assert data.model_type == "resnet18"
        assert any("pointing-game answer key" in n for n in data.notes)

    def test_an_all_background_field_is_skipped_not_scored_as_a_miss(self,
                                                                    tmp_path):
        from spacr.hyperparam import load_activation_data

        merged = tmp_path / "merged"
        merged.mkdir()
        good = np.zeros((12, 12, 8), dtype=np.float32)
        good[..., :4] = 1.0
        good[2:6, 2:6, 4] = 1
        np.save(merged / "a.npy", good)
        np.save(merged / "b.npy", np.zeros((12, 12, 8), dtype=np.float32))

        data = load_activation_data(
            {"src": str(tmp_path), "model_path": self._model_file(tmp_path),
             "image_size": 12, "channels": [0, 1, 2]}, n_images=5)
        assert len(data.images) == 1
        assert len(data.masks) == 1

    def test_no_model_is_refused_with_the_reason(self, tmp_path):
        from spacr.hyperparam import load_activation_data

        with pytest.raises(ValueError) as excinfo:
            load_activation_data({"src": str(tmp_path), "model_path": ""})
        assert "No trained model to explain" in str(excinfo.value)

    def test_neither_source_is_refused_with_both_named(self, tmp_path):
        from spacr.hyperparam import load_activation_data

        with pytest.raises(ValueError) as excinfo:
            load_activation_data({"src": str(tmp_path),
                                  "model_path": self._model_file(tmp_path),
                                  "dataset": "/nope.tar"})
        message = str(excinfo.value)
        assert "merged/*.npy" in message and "/nope.tar" in message

    def test_the_crop_tar_fallback_says_the_pointing_game_is_unavailable(
            self, tmp_path):
        """No silent dropping of a criterion the user asked for."""
        import tarfile

        from PIL import Image

        from spacr.hyperparam import load_activation_data

        raw = tmp_path / "raw"
        raw.mkdir()
        rng = np.random.default_rng(1)
        names = []
        for i in range(3):
            name = f"plate1_A01_1_{i}.png"
            Image.fromarray(
                rng.integers(0, 256, (16, 16, 3)).astype(np.uint8)).save(
                    raw / name)
            names.append(name)
        tar_path = tmp_path / "ds.tar"
        with tarfile.open(tar_path, "w") as tar:
            for name in names:
                tar.add(raw / name, arcname=name)

        data = load_activation_data(
            {"src": str(tmp_path), "model_path": self._model_file(tmp_path),
             "dataset": str(tar_path), "image_size": 16,
             "channels": [1, 2, 3], "normalize_input": False}, n_images=2)
        assert data.masks is None
        assert len(data.images) == 2
        assert any("pointing game cannot be scored" in n for n in data.notes)
