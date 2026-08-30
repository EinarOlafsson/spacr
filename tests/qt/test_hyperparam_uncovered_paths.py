"""Hyperparameter-search paths a successful sweep never walks.

Where the per-trial figures go when there is nowhere to put them, what the
worker emits when it cannot draw one, the GPU toggle's three answers, the
cuML install, the deferred UMAP metric list, and the Walk's refusals.

Nothing here fits a UMAP or trains a model: the search function is injected
or the object under test is a pure helper. Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import numpy as np  # noqa: E402

from spacr.hyperparam import SearchResult, SearchSpace, Trial  # noqa: E402
from spacr.qt.screens import hyperparam as hp  # noqa: E402
from spacr.qt.screens.hyperparam import (  # noqa: E402
    HyperparamPanel,
    SearchRequest,
    _search_figure_dir,
    _SearchWorker,
    build_panel_figure,
)

pytestmark = pytest.mark.qt


@pytest.fixture()
def panel(qtbot, qt_theme_applied):
    """A UMAP panel, built offscreen."""
    widget = HyperparamPanel("umap")
    qtbot.addWidget(widget)
    return widget


def make_trial(index: int = 0, score: float = 0.9, **extra) -> Trial:
    """One finished trial with a known index and score."""
    return Trial(params={"n_neighbors": 5 + index}, score=score,
                 extra_metrics=dict(extra), duration=0.01, index=index)


# ---------------------------------------------------------------------------
# Where the trial figures go
# ---------------------------------------------------------------------------

def test_a_settings_object_that_is_not_a_mapping_still_yields_a_directory():
    """The figure directory is never allowed to fail the run."""
    target = _search_figure_dir(object(), "umap")

    assert target is not None
    assert target.is_dir()
    assert target.name.startswith("spacr-search-")


def test_a_directory_that_cannot_be_made_anywhere_is_reported_as_none(
        monkeypatch):
    """With no writable temporary directory the caller gets ``None``."""
    def _read_only(*_a, **_k):
        raise OSError("read-only file system")

    monkeypatch.setattr(hp.tempfile, "mkdtemp", _read_only)

    assert _search_figure_dir({"src": ""}, "umap") is None


def test_a_src_that_cannot_be_written_falls_back_to_a_temporary_directory(
        tmp_path):
    """An unwritable ``src`` costs a location, not the run."""
    blocked = tmp_path / "file_not_a_dir"
    blocked.write_text("this is a file", encoding="utf-8")

    target = _search_figure_dir({"src": str(blocked / "sub")}, "classify")

    assert target is not None
    assert target.is_dir()
    assert target.name.startswith("spacr-search-")


# ---------------------------------------------------------------------------
# What the worker emits when there is no picture
# ---------------------------------------------------------------------------

def test_a_worker_with_nowhere_to_draw_still_announces_the_trial(
        qtbot, monkeypatch):
    """No figure directory means an empty path, not a lost trial."""
    monkeypatch.setattr(hp, "_search_figure_dir", lambda *a, **k: None)
    request = SearchRequest(app_key="classify", settings={},
                            space=SearchSpace({"learning_rate": [0.1]}))
    worker = _SearchWorker(request)
    assert worker._figure_dir is None

    with qtbot.waitSignal(worker.trial_ready, timeout=1000) as caught:
        worker._emit_trial(make_trial(3), 1, 4)

    trial, done, total, path = caught.args
    assert trial.index == 3
    assert (done, total) == (1, 4)
    assert path == ""


def test_a_trial_whose_figure_cannot_be_rendered_announces_an_empty_path(
        qtbot, monkeypatch, tmp_path):
    """A renderer that declines leaves the path empty and the trial intact."""
    monkeypatch.setattr(hp, "_search_figure_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(hp, "render_trial_figure", lambda *a, **k: False)
    request = SearchRequest(app_key="classify", settings={},
                            space=SearchSpace({"learning_rate": [0.1]}))
    worker = _SearchWorker(request)

    with qtbot.waitSignal(worker.trial_ready, timeout=1000) as caught:
        worker._emit_trial(make_trial(7), 2, 4)

    assert caught.args[3] == ""
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# The score plot with no noise to shade
# ---------------------------------------------------------------------------

def test_a_single_trial_score_plot_shades_no_noise_band(qt_theme_applied):
    """One trial gives no estimate of noise, so no band is drawn."""
    trial = make_trial(0, 0.8)
    result = SearchResult(trials=[trial], best=trial,
                          space=SearchSpace({"learning_rate": [0.1]}),
                          metric="accuracy", notes=[])
    assert result.noise_level()[0] is None

    figure = build_panel_figure(result)

    try:
        axes = figure.axes[0]
        assert axes.get_legend() is None, "no band means no band legend"
        assert axes.get_ylabel() == "accuracy"
        assert len(axes.lines) == 1
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


# ---------------------------------------------------------------------------
# The Walk's starting point
# ---------------------------------------------------------------------------

def test_a_walk_axis_with_no_start_falls_back_to_the_panels_own_value(panel):
    """A blank axis start is filled from where the user can already see one."""
    panel._adaptive.setChecked(True)
    panel.set_walk_axes({"spread": {"start": "", "resolution": 3}})

    space = panel.current_adaptive_space()

    assert tuple(space.params["spread"]) == (1.0,), (
        "the UMAP default start is what the panel would show")


def test_a_walk_axis_with_no_start_anywhere_says_where_to_give_it_one(panel):
    """A blank setting leaves nothing to walk from, and it is named.

    The message has to be the one about a missing start. Letting the blank
    reach the parser instead complains that the axis was given the wrong
    *number* of values, which sends the user looking for a list they never
    typed.
    """
    panel._adaptive.setChecked(True)
    panel.apply_settings({"spread": ""})
    panel.set_walk_axes({"spread": {"start": "", "resolution": 3}})

    with pytest.raises(ValueError) as caught:
        panel.current_adaptive_space()

    assert str(caught.value) == (
        "The Walk searches spread but has no value to start from. "
        "Give it one in the Axes dialog.")


def test_a_walk_axis_given_two_starting_values_is_refused(panel):
    """A Walk starts from one point, so a list is an error, not a grid."""
    panel._adaptive.setChecked(True)
    panel.set_walk_axes({"spread": {"start": "0.5, 1.5", "resolution": 3}})

    with pytest.raises(ValueError) as caught:
        panel.current_adaptive_space()

    assert "single value for spread" in str(caught.value)
    assert "not 2" in str(caught.value)


# ---------------------------------------------------------------------------
# The GPU toggle
# ---------------------------------------------------------------------------

def test_turning_the_gpu_off_never_probes_for_cuml(panel, monkeypatch):
    """Switching off is a local decision; nothing is imported or asked."""
    def _refuse():
        raise AssertionError("turning the GPU off must not probe for cuML")

    monkeypatch.setattr("spacr.gpu_reduce.install_plan", _refuse)

    assert panel.request_gpu_enabled(False) is False
    assert panel.gpu_backend() == "cpu"
    assert "CPU reducers will run" in panel._status.text()


def test_an_available_cuml_turns_the_gpu_on_and_says_what_it_found(
        panel, monkeypatch):
    """A ready plan is the only thing that leaves the toggle down."""
    monkeypatch.setattr("spacr.gpu_reduce.install_plan",
                        lambda: {"action": "ready", "message": "cuML 24.02"})

    assert panel.request_gpu_enabled(True) is True
    assert panel.gpu_backend() == "cuml"
    assert "cuML 24.02" in panel._status.text()


def test_an_unavailable_cuml_leaves_the_toggle_off_and_explains(
        panel, monkeypatch):
    """The toggle must not stay down claiming a backend that is not there."""
    calls = []
    monkeypatch.setattr(
        "spacr.gpu_reduce.install_plan",
        lambda: {"action": "no_device", "message": "no CUDA device"})
    monkeypatch.setattr("spacr.gpu_reduce.availability_entry",
                        lambda: {"name": "cuML"})
    monkeypatch.setattr(
        "spacr.qt.widgets.availability_panel.explain",
        lambda anchor, entries, **kw: calls.append((anchor, entries, kw)))

    assert panel.request_gpu_enabled(True) is False
    assert panel.gpu_backend() == "cpu"
    assert "no CUDA device" in panel._status.text()
    assert len(calls) == 1
    anchor, entries, kw = calls[0]
    assert anchor is panel
    assert entries == [{"name": "cuML"}]

    # The panel's Install callback promises a restart rather than a live GPU.
    kw["on_installed"]("an offer")
    assert "Restart spaCR" in panel._status.text()


def test_the_availability_panel_is_anchored_to_the_control_that_asked(
        panel, qtbot, monkeypatch):
    """A caller with a button gets the panel under that button."""
    from PySide6.QtWidgets import QPushButton

    button = QPushButton("GPU")
    qtbot.addWidget(button)
    seen = []
    monkeypatch.setattr(
        "spacr.gpu_reduce.install_plan",
        lambda: {"action": "install", "message": "cuML can be installed"})
    monkeypatch.setattr("spacr.gpu_reduce.availability_entry", lambda: {})
    monkeypatch.setattr(
        "spacr.qt.widgets.availability_panel.explain",
        lambda anchor, entries, **kw: seen.append(anchor))

    panel.request_gpu_enabled(True, anchor=button)

    assert seen == [button]


# ---------------------------------------------------------------------------
# Installing cuML
# ---------------------------------------------------------------------------

def test_a_successful_cuml_install_insists_on_a_restart(panel, monkeypatch):
    """pip may have moved numpy under this process, so nothing is claimed."""
    shown = []
    monkeypatch.setattr("spacr.gpu_reduce.install_command",
                        lambda: ["python", "-m", "pip", "install", "x"])
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **k: shown.append(("ran", a[0])))
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(a[2])))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: pytest.fail(
                            "a successful install must not warn")))

    panel._install_cuml()

    assert shown[0] == ("ran", ["python", "-m", "pip", "install", "x"])
    assert "RESTART spaCR" in shown[1]
    assert panel._status.text() == "cuML installed. Restart spaCR to use it."
    assert panel.gpu_backend() == "cpu", (
        "an install does not make this session's GPU backend usable")


def test_a_failed_cuml_install_shows_the_command_to_run_by_hand(
        panel, monkeypatch):
    """The user is given the exact command rather than a bare failure."""
    warned = []
    monkeypatch.setattr("spacr.gpu_reduce.install_command",
                        lambda: ["python", "-m", "pip", "install", "x"])

    def _explode(*_a, **_k):
        raise RuntimeError("network is down")

    monkeypatch.setattr("subprocess.run", _explode)
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a[2])))
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: pytest.fail(
                            "a failed install must not report success")))

    panel._install_cuml()

    assert len(warned) == 1
    assert "network is down" in warned[0]
    assert "python -m pip install x" in warned[0]
    assert panel._status.text() == "cuML install failed."


# ---------------------------------------------------------------------------
# The deferred UMAP metric list
# ---------------------------------------------------------------------------

class _RecordingCombo(hp.QComboBox):
    """A combo that records ``showPopup`` instead of opening a native popup."""

    def __init__(self):
        super().__init__()
        self.popups = 0

    def showPopup(self):  # noqa: N802 - Qt name
        self.popups += 1


def test_the_metric_list_is_completed_the_first_time_the_combo_opens(
        qtbot, monkeypatch):
    """Opening the list swaps in the installed metrics and keeps the choice."""
    combo = _RecordingCombo()
    qtbot.addWidget(combo)
    combo.addItems(["euclidean", "cosine"])
    combo.setCurrentText("cosine")
    monkeypatch.setattr(hp, "umap_metrics",
                        lambda: ("euclidean", "cosine", "manhattan"))
    hp._complete_metrics_when_opened(combo)

    combo.showPopup()

    assert [combo.itemText(i) for i in range(combo.count())] == [
        "euclidean", "cosine", "manhattan"]
    assert combo.currentText() == "cosine"
    assert combo.popups == 1


def test_the_metric_list_is_not_read_a_second_time(qtbot, monkeypatch):
    """After one completion the combo is an ordinary combo again."""
    combo = _RecordingCombo()
    qtbot.addWidget(combo)
    combo.addItems(["euclidean"])
    reads = []

    def _metrics():
        reads.append(1)
        return ("euclidean", "cosine")

    monkeypatch.setattr(hp, "umap_metrics", _metrics)
    hp._complete_metrics_when_opened(combo)

    combo.showPopup()
    monkeypatch.setattr(hp, "umap_metrics",
                        lambda: ("this", "would", "change", "it"))
    combo.showPopup()

    assert len(reads) == 1
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "euclidean", "cosine"]
    assert combo.popups == 2


def test_a_metric_list_that_cannot_be_read_leaves_the_combo_alone(
        qtbot, monkeypatch):
    """An uninstallable umap costs the completion, not the control."""
    combo = _RecordingCombo()
    qtbot.addWidget(combo)
    combo.addItems(["euclidean", "cosine"])

    def _explode():
        raise ImportError("no module named umap")

    monkeypatch.setattr(hp, "umap_metrics", _explode)
    hp._complete_metrics_when_opened(combo)

    combo.showPopup()
    combo.showPopup()

    assert [combo.itemText(i) for i in range(combo.count())] == [
        "euclidean", "cosine"]
    assert combo.popups == 2, "the popup still opens on both clicks"


# ---------------------------------------------------------------------------
# Rows, trials and the gallery
# ---------------------------------------------------------------------------

def _finished_result(trials):
    """A SearchResult over ``trials``, best first."""
    ranked = sorted(trials, key=lambda t: -t.score)
    return SearchResult(trials=list(trials), best=ranked[0],
                        space=SearchSpace({"n_neighbors": [5, 6]}),
                        metric="trustworthiness", notes=[])


def test_selecting_a_row_for_a_trial_the_table_never_held_changes_nothing(
        panel):
    """A stray trial leaves the current selection where it was."""
    shown = [make_trial(0, 0.9), make_trial(1, 0.8)]
    panel._rebuild_table(_finished_result(shown))
    panel._table.selectRow(1)
    before = [i.row() for i in panel._table.selectionModel().selectedRows()]

    panel._select_trial_row(make_trial(9, 0.5))

    after = [i.row() for i in panel._table.selectionModel().selectedRows()]
    assert after == before == [1]


def test_a_gallery_pick_that_cannot_be_displayed_selects_no_row(panel):
    """A trial with no embedding is refused, so no row is chosen for it."""
    shown = [make_trial(0, 0.9), make_trial(1, 0.8)]
    panel._rebuild_table(_finished_result(shown))
    panel._table.selectRow(1)

    panel._on_gallery_trial(shown[0])

    assert panel._displayed_trial is None
    rows = [i.row() for i in panel._table.selectionModel().selectedRows()]
    assert rows == [1], "the refused trial must not move the selection"


def test_a_second_umap_trial_does_not_steal_the_view_from_the_first(
        panel, monkeypatch):
    """Once a map is displayed, later arrivals fill the table only."""
    rng = np.random.default_rng(0)
    first = make_trial(0, 0.9, embedding=rng.normal(size=(8, 2)))
    second = make_trial(1, 0.95, embedding=rng.normal(size=(8, 2)))
    shown = []
    monkeypatch.setattr(type(panel), "show_trial",
                        lambda self, trial: (shown.append(trial), True)[1])

    panel._on_trial_ready(first, 1, 2, "")
    panel._displayed_trial = first
    panel._on_trial_ready(second, 2, 2, "")

    assert all(trial is not second for trial in shown), (
        "the displayed map is not replaced under the user")
    assert shown and shown[-1] is first
    assert panel._table.rowCount() == 2
    assert panel._grid_btn.isEnabled()
    assert "2 of 2 configurations evaluated" in panel._status.text()


# ---------------------------------------------------------------------------
# The settings window for an app that has no UMAP tabs
# ---------------------------------------------------------------------------

def test_a_non_umap_settings_window_has_only_the_search_tab(qtbot,
                                                            qt_theme_applied):
    """Only a UMAP sweep gets the reducer, clustering and appearance tabs."""
    panel = HyperparamPanel("ml_analyze")
    qtbot.addWidget(panel)

    dialog = hp.UmapSearchSettingsDialog(panel)
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "Hyperparameter search settings"
    assert [dialog._tabs.tabText(i)
            for i in range(dialog._tabs.count())] == ["Search"]
    assert dialog._module_model is None


def test_propagating_from_a_window_with_no_module_tabs_sends_the_fields(
        qtbot, qt_theme_applied):
    """Without module tabs the search fields are the whole payload."""
    panel = HyperparamPanel("ml_analyze")
    qtbot.addWidget(panel)
    panel._value_edits["learning_rate"].setText("0.05")
    panel._value_edits["n_estimators"].setText("200")
    seen = []
    panel.set_apply_callback(seen.append)

    dialog = hp.UmapSearchSettingsDialog(panel)
    qtbot.addWidget(dialog)
    dialog.propagate_settings()

    assert len(seen) == 1
    assert seen[0]["learning_rate"] == 0.05
    assert seen[0]["n_estimators"] == 200


# ---------------------------------------------------------------------------
# The Walk axes dialog
# ---------------------------------------------------------------------------

def test_a_walk_axis_start_that_is_not_one_of_the_choices_is_not_selected(
        panel, qtbot):
    """A stored start outside the choice list leaves the combo on its first."""
    panel.set_walk_axes({"init": {"start": "banana", "resolution": 3}})

    dialog = hp.WalkAxesDialog(panel)
    qtbot.addWidget(dialog)

    _toggle, start, _resolution = dialog._rows["init"]
    assert isinstance(start, hp.QComboBox)
    assert start.currentText() == "spectral"
    assert start.findText("banana") < 0


def test_a_walk_axis_start_that_is_one_of_the_choices_is_selected(
        panel, qtbot):
    """A stored start the control can offer is the one shown."""
    panel.set_walk_axes({"init": {"start": "pca", "resolution": 4}})

    dialog = hp.WalkAxesDialog(panel)
    qtbot.addWidget(dialog)

    _toggle, start, resolution = dialog._rows["init"]
    assert start.currentText() == "pca"
    assert resolution.value() == 4


def test_a_metric_list_that_already_matches_is_left_untouched(
        qtbot, monkeypatch):
    """When the combo already holds the installed metrics, nothing is rebuilt."""
    combo = _RecordingCombo()
    qtbot.addWidget(combo)
    combo.addItems(["euclidean", "cosine"])
    combo.setCurrentText("cosine")
    monkeypatch.setattr(hp, "umap_metrics", lambda: ("euclidean", "cosine"))
    hp._complete_metrics_when_opened(combo)

    combo.showPopup()

    assert [combo.itemText(i) for i in range(combo.count())] == [
        "euclidean", "cosine"]
    assert combo.currentText() == "cosine"
    assert combo.popups == 1


def test_a_choice_the_installed_metrics_no_longer_offer_is_not_restored(
        qtbot, monkeypatch):
    """A metric that has gone away leaves the combo on the first real one."""
    combo = _RecordingCombo()
    qtbot.addWidget(combo)
    combo.addItems(["hand_written_metric"])
    monkeypatch.setattr(hp, "umap_metrics", lambda: ("euclidean", "cosine"))
    hp._complete_metrics_when_opened(combo)

    combo.showPopup()

    assert [combo.itemText(i) for i in range(combo.count())] == [
        "euclidean", "cosine"]
    assert combo.currentText() == "euclidean"


# ---------------------------------------------------------------------------
# reading and writing a search-space control, whichever kind it is
# ---------------------------------------------------------------------------

def test_a_saved_value_the_combo_does_not_offer_is_not_invented(qtbot):
    """"without inventing choices" is the docstring, and this is it.

    A settings file can name a value this build's menu no longer has -- a
    model retired between versions, a metric renamed. Adding it to the combo
    would offer the user a choice the run cannot honour; leaving the combo
    where it is means the form shows what will actually be used.
    """
    from PySide6.QtWidgets import QComboBox

    from spacr.qt.screens.hyperparam import HyperparamPanel

    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItems(["euclidean", "cosine"])
    combo.setCurrentIndex(0)

    HyperparamPanel._set_control_text(combo, "a_metric_that_was_retired")

    assert combo.count() == 2, "a value the build does not offer was added"
    assert combo.currentText() == "euclidean"


def test_a_value_the_combo_does_offer_is_selected(qtbot):
    """Otherwise the refusal above would pass on a no-op setter."""
    from PySide6.QtWidgets import QComboBox

    from spacr.qt.screens.hyperparam import HyperparamPanel

    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItems(["euclidean", "cosine"])

    HyperparamPanel._set_control_text(combo, "cosine")

    assert combo.currentText() == "cosine"


def test_a_free_text_field_takes_the_value_as_written(qtbot):
    """A grid list is free text; there is nothing to match it against."""
    from PySide6.QtWidgets import QLineEdit

    from spacr.qt.screens.hyperparam import HyperparamPanel

    edit = QLineEdit()
    qtbot.addWidget(edit)

    HyperparamPanel._set_control_text(edit, "5, 10, 20")

    assert edit.text() == "5, 10, 20"


def test_a_control_of_neither_kind_is_left_alone(qtbot):
    """The form's widgets are built elsewhere and can be replaced.

    Setting text on something that is neither would raise inside a settings
    load, taking the screen down while it was restoring a saved run.
    """
    from PySide6.QtWidgets import QLabel

    from spacr.qt.screens.hyperparam import HyperparamPanel

    label = QLabel("untouched")
    qtbot.addWidget(label)

    HyperparamPanel._set_control_text(label, "something")

    assert label.text() == "untouched"


def test_reading_a_control_of_neither_kind_is_an_empty_string(qtbot):
    """The counterpart, and ``""`` is what the caller stores as "unset"."""
    from PySide6.QtWidgets import QLabel

    from spacr.qt.screens.hyperparam import HyperparamPanel

    label = QLabel("not a control")
    qtbot.addWidget(label)

    assert HyperparamPanel._control_text(label) == ""
