"""The results side of a module screen, asked to show what may not be there.

Every helper here is called from a signal that can outlive the widgets it
names: the results tabs, the figure stack, the figure grid, the live tiles'
menus, the Runs tab and the action-row cards. Each must do the part it can --
move the page, record the preference, write the figure, say the sentence --
and skip only the part whose collaborator is missing or refuses. Each case
drives both arms where one exists: a collaborator that answers and one that
does not, so the empty answer is a missing widget and not a missing search.

One regression screen per test; each case patches through its own
``monkeypatch.context()``.
"""
from __future__ import annotations

import logging
import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint                                # noqa: E402
from PySide6.QtWidgets import QLineEdit, QWidget                 # noqa: E402

from spacr.qt.screens import app_screen as aps                   # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


def _refusing(*_args, **_kwargs):
    raise RuntimeError("the widget has been destroyed")


def _console_text(console) -> str:
    from spacr.qt.widgets.console_panel import _StdoutBlock

    return "\n".join(block.text()
                     for block in console.findChildren(_StdoutBlock))


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


def _walk(subtests, monkeypatch, cases, **env):
    caplog = env.get("caplog")
    completed = []
    for name, case in cases.items():
        if caplog is not None:
            caplog.clear()
        with subtests.test(case=name), monkeypatch.context() as mp:
            case(types.SimpleNamespace(mp=mp, name=name, **env))
            completed.append(name)
    assert completed == list(cases), "a case did not complete"


def _set(env, **attrs):
    for name, value in attrs.items():
        env.mp.setattr(env.screen, name, value, raising=False)


def _widget(env):
    widget = QWidget()
    env.qtbot.addWidget(widget)
    return widget


# -- which page of the results container is on screen -------------------------

class _Tabs:
    def __init__(self, error=None):
        self.asked = 0
        self._error = error

    def setCurrentWidget(self, _page):          # noqa: N802 - Qt naming
        self.asked += 1
        raise self._error


def _raise_tab_refuses(env, error):
    tabs = _Tabs(error)
    _set(env, _results_tabs=tabs, _results_page=object())
    env.screen._raise_the_results_tab()
    assert tabs.asked == 1, "the tab was never asked to be raised"


def _raise_tab_runtime_error(env):
    _raise_tab_refuses(env, RuntimeError("Internal C++ object already deleted."))


def _raise_tab_type_error(env):
    _raise_tab_refuses(env, TypeError("not a QWidget"))


def _raise_tab_nothing_to_raise(env):
    _set(env, _results_tabs=None, _results_page=None)
    env.screen._raise_the_results_tab()
    assert env.screen._results_tabs is None
    assert env.screen._results_page is None


def _showing_results_follows_the_tab(env):
    page = _widget(env)
    other = _widget(env)
    _set(env, _results_panel=page,
         _results_tabs=types.SimpleNamespace(currentWidget=lambda: page))
    assert env.screen.showing_the_results() is True
    env.screen._results_tabs = types.SimpleNamespace(currentWidget=lambda: other)
    assert env.screen.showing_the_results() is False


def _figure_grid_needs_a_stack(env):
    refreshed, moved = [], []
    env.mp.setattr(env.screen, "_refresh_figure_grid",
                   lambda: refreshed.append(1))
    _set(env, _figures_stack=types.SimpleNamespace(
        setCurrentIndex=moved.append, setCurrentWidget=moved.append))
    env.screen._show_figure_grid()
    assert refreshed == [1] and moved == [0]
    env.screen._figures_stack = None
    env.screen._show_figure_grid()
    assert refreshed == [1] and moved == [0]


def _volcano_page_needs_stack_and_page(env):
    moved = []
    page = _widget(env)
    _set(env, _figures_stack=types.SimpleNamespace(setCurrentWidget=moved.append),
         _volcano_page=page)
    env.screen._show_regression_graph()
    assert moved == [page]
    env.screen._volcano_page = None
    env.screen._show_regression_graph()
    assert moved == [page]


def _gene_tile_opened_once(env):
    env.mp.setattr(env.screen, "_show_regression_graph", lambda: None)
    sized = []
    _set(env, _gene_split=types.SimpleNamespace(
        sizes=lambda: [600, 0], setSizes=sized.append, height=lambda: 600),
        _gene_opened=False)
    env.screen._on_guide_selected("gene_A")
    assert sized == [[360, 240]]
    env.screen._on_guide_selected("gene_A")
    assert sized == [[360, 240]]


def _pressed_tile_opens_the_figure(env):
    moved, opened = [], []
    detail = _widget(env)
    _set(env, _figure_detail=detail,
         _figure_queue=types.SimpleNamespace(show_index=opened.append),
         _figures_stack=types.SimpleNamespace(setCurrentWidget=moved.append))
    env.screen._open_figure_from_grid(2)
    assert opened == [2] and moved == [detail]
    env.screen._figures_stack = None
    env.screen._open_figure_from_grid(3)
    assert opened == [2, 3]
    assert moved == [detail]


class _RecordingQueue:
    def __init__(self):
        self.added = []
        self.shown = 0

    def add_figure(self, fig, prerendered_png=None):
        self.added.append(fig)

    def show(self):
        self.shown += 1


def _figure_mid_run_keeps_interactive_view(env):
    from matplotlib.figure import Figure

    # A screen of its own: the first figure builds the deferred figures card,
    # and it is torn down by qtbot with these stand-ins still in place.
    screen = AppScreen("regression")
    env.qtbot.addWidget(screen)
    env.mp.setattr(screen, "_queue_figure_grid_refresh", lambda: None)
    queue = _RecordingQueue()
    screen._umap_explorer = object()
    screen._umap_payload_ready = True
    screen._figure_queue = queue
    screen._interactive_switch = types.SimpleNamespace(isChecked=lambda: False)
    screen._on_figure_ready(Figure())
    assert queue.shown == 1
    screen._interactive_switch = types.SimpleNamespace(isChecked=lambda: True)
    screen._on_figure_ready(Figure())
    assert len(queue.added) == 2
    assert queue.shown == 1


def test_results_pages_move_only_when_their_widgets_are_there(
        subtests, monkeypatch, qtbot, screen):
    """Raising the results tab survives RuntimeError/TypeError and does nothing
    without tabs or page; the figure grid, volcano page and a pressed tile move
    the stack only when there is one (the queue and refresh still act); a gene
    tile is sized open only once; a figure arriving while the interactive view
    is open is queued without being shown over it."""
    cases = {
        "raise tab: runtime error": _raise_tab_runtime_error,
        "raise tab: type error": _raise_tab_type_error,
        "raise tab: nothing to raise": _raise_tab_nothing_to_raise,
        "showing the results": _showing_results_follows_the_tab,
        "figure grid": _figure_grid_needs_a_stack,
        "volcano page": _volcano_page_needs_stack_and_page,
        "gene tile once": _gene_tile_opened_once,
        "pressed tile": _pressed_tile_opens_the_figure,
        "figure mid-run": _figure_mid_run_keeps_interactive_view,
    }
    _walk(subtests, monkeypatch, cases, screen=screen, qtbot=qtbot)


# -- figures, the publication sheet and a trial's saved figures --------------

class _Sheet:
    def __init__(self, figure, skipped):
        self.figure = figure
        self.skipped = skipped

    def legend(self):
        return "Panels a-c: coefficients, p-values, QC."


def _sheet_ready(env, skipped):
    import pandas as pd
    from matplotlib.figure import Figure
    from spacr import figures

    _set(env, _results_panel=types.SimpleNamespace(
        results_frame=lambda: pd.DataFrame({"gene": ["A"],
                                            "coefficient": [1.0]})))
    figure = Figure()
    env.mp.setattr(figures, "build_sheet",
                   lambda frame, width="double": _Sheet(figure, skipped))
    shown = []
    _set(env, _figure_queue=types.SimpleNamespace(
        add_figure=shown.append, count=lambda: 1,
        show_index=lambda index: shown.append(index)))
    return shown, figure


def _sheet_goes_into_the_figure_queue(env):
    shown, figure = _sheet_ready(env, skipped=())
    _set(env, _figures_stack=None)
    env.screen._show_publication_sheet()
    assert shown == [figure, 0]
    assert "Panels a-c" in _console_text(env.screen._console)


def _sheet_names_a_skipped_panel(env):
    _sheet_ready(env, skipped=(
        types.SimpleNamespace(title="QQ", reason="no residuals"),))
    detail = _widget(env)
    moved = []
    _set(env, _figures_stack=types.SimpleNamespace(setCurrentWidget=moved.append),
         _figure_detail=detail)
    env.screen._show_publication_sheet()
    assert moved == [detail]
    assert "QQ not shown (no residuals)" in _console_text(env.screen._console)


def _write_png(path, colour="red"):
    from PySide6.QtGui import QColor, QPixmap

    pixmap = QPixmap(8, 8)
    pixmap.fill(QColor(colour))
    assert pixmap.save(str(path), "PNG")


def _trial_run_folder(env, with_screen_figure):
    root = env.tmp_path / env.name.replace(" ", "_").replace(":", "")
    run = root / "results" / "run_1"
    run.mkdir(parents=True)
    _write_png(run / "volcano.png")
    if with_screen_figure:
        _write_png(root / "results" / "fraction_threshold.png", "blue")
    laid_out = {}
    _set(env, _figures_stack=None, _figure_grid=types.SimpleNamespace(
        set_figures=lambda pixmaps, titles, sections=(): laid_out.update(
            pixmaps=pixmaps, titles=titles, sections=list(sections))))
    return run, laid_out


def _trial_screen_figures_get_their_heading(env):
    run, laid_out = _trial_run_folder(env, with_screen_figure=True)
    assert env.screen._load_trial_figures(str(run)) == 2
    assert laid_out["titles"] == ["volcano", "fraction_threshold"]
    assert laid_out["sections"][-1] == (
        "the screen's own figures — not this run's", 1, 1)


def _trial_without_screen_figures(env):
    run, laid_out = _trial_run_folder(env, with_screen_figure=False)
    assert env.screen._load_trial_figures(str(run)) == 1
    assert laid_out["titles"] == ["volcano"]
    assert [name for name, _at, _count in laid_out["sections"]] == ["run_1"]


def _tile_width_remembered_without_grid(env):
    from spacr.qt import preferences

    stored, widths = [], []
    env.mp.setattr(preferences, "set_figure_grid_size", stored.append)
    _set(env, _figure_grid=types.SimpleNamespace(
        set_target_cell_width=widths.append))
    env.screen._on_figure_size(320)
    assert widths == [320] and stored == [320]
    env.screen._figure_grid = None
    env.screen._on_figure_size(240)
    assert widths == [320]
    assert stored == [320, 240]


def _trial_loaded_without_undo(env):
    env.mp.setattr(env.screen, "_load_trial_figures", lambda _f: 0)
    _set(env, _pending_trial=("trial 3", str(env.tmp_path)), _sweep_runs=None)
    env.screen._on_trial_loaded(True)
    text = _console_text(env.screen._console)
    assert "trial 3 saved a results table but no figures" in text
    assert "trial 3 is loaded" in text
    assert env.screen._pending_trial is None


def _effects_grid_needs_a_folder(env):
    import pandas as pd
    from spacr import cell_montage

    written = []
    env.mp.setattr(cell_montage, "write_effects_grid",
                   lambda effects, folder: written.append(folder))
    result = types.SimpleNamespace(
        effects=pd.DataFrame({"gene": ["A"], "effect": [1.0]}))
    env.mp.setattr(env.screen, "_results_source_path",
                   lambda: str(env.tmp_path))
    env.screen._keep_the_effects_grid(result)
    assert written == [str(env.tmp_path)]
    env.mp.setattr(env.screen, "_results_source_path", lambda: "")
    env.screen._keep_the_effects_grid(result)
    assert written == [str(env.tmp_path)]


def test_figures_and_trials_land_in_the_ordinary_queue_and_grid(
        subtests, monkeypatch, qtbot, tmp_path, screen, console_info_enabled):
    """The publication sheet goes through the figure queue and names any panel
    it could not draw; a trial's grid separates the screen's own figures under
    their own heading; a tile width is remembered even before the grid exists;
    a loaded trial is announced without an undo-capable Runs tab; the effects
    grid is written only when there is somewhere to write it."""
    cases = {
        "sheet into queue": _sheet_goes_into_the_figure_queue,
        "sheet names skipped panel": _sheet_names_a_skipped_panel,
        "trial: screen figures heading": _trial_screen_figures_get_their_heading,
        "trial: no screen figures": _trial_without_screen_figures,
        "tile width remembered": _tile_width_remembered_without_grid,
        "trial loaded without undo": _trial_loaded_without_undo,
        "effects grid": _effects_grid_needs_a_folder,
    }
    _walk(subtests, monkeypatch, cases, screen=screen, qtbot=qtbot,
          tmp_path=tmp_path)


# -- live-tile menus and the Runs tab's bookkeeping --------------------------

def _panel_with_a_volcano(opened, fail=False):
    def build_style_menu():
        if fail:
            raise RuntimeError("the plot item is gone")
        return types.SimpleNamespace(exec=opened.append)

    return types.SimpleNamespace(
        volcano=types.SimpleNamespace(build_style_menu=build_style_menu))


def _menu_pinned_tile_left_to_its_signal(env):
    from spacr.qt.widgets.figure_grid_view import PINNED_KEY

    env.mp.setattr(env.screen, "_pin_regression_graph", lambda: None)
    opened = []
    panel = _panel_with_a_volcano(opened)
    panel.qq = panel.volcano
    _set(env, _results_panel=panel)
    assert AppScreen._LIVE_TILE_WIDGETS["qq"] == "qq"
    env.screen._live_tile_menu(PINNED_KEY, QPoint(3, 4))
    assert opened == []
    env.screen._live_tile_menu("qq", QPoint(3, 4))
    assert opened == [QPoint(3, 4)]


def _menu_pinned_is_the_graphs_own(env):
    pinned, opened = [], []
    env.mp.setattr(env.screen, "_pin_regression_graph",
                   lambda: pinned.append(1))
    _set(env, _results_panel=_panel_with_a_volcano(opened))
    env.screen._pinned_menu(QPoint(1, 2))
    assert opened == [QPoint(1, 2)]
    assert pinned == [1]


def _menu_that_will_not_build_still_repins(env):
    pinned = []
    env.mp.setattr(env.screen, "_pin_regression_graph",
                   lambda: pinned.append(1))
    _set(env, _results_panel=_panel_with_a_volcano([], fail=True))
    env.screen._pinned_menu(QPoint(1, 2))
    assert "could not open the live tile's menu" in env.caplog.text
    assert pinned == [1]


def _measurements_skip_a_row_without_database(env):
    _set(env, _settings_model=None)
    env.mp.setattr(env.screen, "_attached_database_rows", lambda: [
        {"database": ""},
        {"database": "/proj/plate1/measurements/measurements.db"}])
    assert env.screen._measurements_destination() == os.path.join(
        "/proj/plate1", "measurements")


def _column_fit_without_handle_not_stored(env):
    _set(env, _column_run_handles={})
    env.mp.setattr(env.screen, "_record_run_in_runs_tab",
                   lambda *_a: "handle-7")
    env.screen._on_column_fit_started("response_1", {})
    assert env.screen._column_run_handles == {"response_1": "handle-7"}
    env.mp.setattr(env.screen, "_record_run_in_runs_tab", lambda *_a: None)
    env.screen._on_column_fit_started("response_2", {})
    assert env.screen._column_run_handles == {"response_1": "handle-7"}


def _removed_runs_drop_only_labelled_figures(env):
    forgotten, dropped = [], []
    _set(env, _results_panel=types.SimpleNamespace(
        forget_run=lambda folder: forgotten.append(folder)),
        _compare_panel=None,
        _figure_queue=types.SimpleNamespace(
            forget_run=lambda label: dropped.append(label) or True))
    env.mp.setattr(env.screen, "_queue_figure_grid_refresh", lambda: None)
    env.screen._on_runs_removed([{"folder": "/data/run_1", "run": ""},
                                 {"folder": "/data/run_2", "run": "run 2"},
                                 {"folder": ""}])
    assert forgotten == ["/data/run_1", "/data/run_2"]
    assert dropped == ["run 2"]


def _refused_frame_gets_no_settings(env):
    import pandas as pd

    env.mp.setattr(env.screen, "_update_run_in_runs_tab", lambda **_k: None)
    env.mp.setattr(env.screen, "_say_the_qc_verdict", lambda _p: None)
    taken = []
    _set(env, _last_run_folder="", _results_panel=types.SimpleNamespace(
        set_frame=lambda frame, source="": taken.append(source) and False))
    env.screen._on_pipeline_result({
        "results": pd.DataFrame({"gene": ["A"], "coefficient": [1.0]}),
        "res_folder": "/data/run_9"})
    assert taken == ["/data/run_9"]
    assert env.screen._last_run_folder == "/data/run_9"


def test_tile_menus_and_runs_bookkeeping_skip_only_what_is_missing(
        subtests, monkeypatch, qtbot, caplog, screen):
    """The pinned tile opens the graph's own menu and is retaken even when the
    menu cannot be built; the generic tile path leaves the pinned key alone; a
    database row without a path is skipped; a fit the Runs tab would not record
    stores no handle; removed runs drop figures only by label; a panel that
    refuses a frame is handed no settings after it."""
    caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
    cases = {
        "menu: pinned key via generic path": _menu_pinned_tile_left_to_its_signal,
        "menu: pinned menu": _menu_pinned_is_the_graphs_own,
        "menu: will not build": _menu_that_will_not_build_still_repins,
        "measurements destination": _measurements_skip_a_row_without_database,
        "column fit without handle": _column_fit_without_handle_not_stored,
        "removed runs": _removed_runs_drop_only_labelled_figures,
        "refused frame": _refused_frame_gets_no_settings,
    }
    _walk(subtests, monkeypatch, cases, screen=screen, qtbot=qtbot,
          caplog=caplog)


# -- action-row cards, the run banner and the live status --------------------

def _sweep_card_without_panel(env):
    card = _widget(env)
    seeded = []
    _set(env, _sweep_card=card, _sweep=types.SimpleNamespace(
        apply_settings=lambda values: seeded.append(values)))
    env.screen._on_sweep_switch(True)
    assert len(seeded) == 1
    env.screen._sweep = None
    env.screen._on_sweep_switch(False)
    env.screen._on_sweep_switch(True)
    assert card.isVisible() or card.isVisibleTo(env.screen)
    assert len(seeded) == 1


def _sweep_refused_seed(env):
    _set(env, _sweep_card=_widget(env),
         _sweep=types.SimpleNamespace(apply_settings=_refusing))
    env.screen._on_sweep_switch(True)
    assert "could not seed the parameter sweep" in env.caplog.text


def _permutation_banner(env):
    _set(env, _slow_fit=False)
    env.screen._announce_the_fit({"regression_type": "ols", "level": "gene"})
    assert "Model: ols" in _console_text(env.screen._console)
    assert env.screen._slow_fit is False
    env.screen._announce_the_fit({"inference": "nonparametric",
                                  "guide_permutations": 10000})
    assert "permutation" in _console_text(env.screen._console).lower()
    assert env.screen._slow_fit is False


def _usage_poll_without_timer(env):
    from PySide6.QtGui import QHideEvent, QShowEvent

    timer = env.screen._usage_timer
    assert timer is not None
    timer.stop()
    env.screen.showEvent(QShowEvent())
    assert timer.isActive()
    timer.stop()
    _set(env, _usage_timer=None)
    before = env.screen._usage_generation
    env.screen.showEvent(QShowEvent())
    env.screen.hideEvent(QHideEvent())
    assert env.screen._usage_generation == before + 1


def _per_core_bars_from_one_sample(env):
    painted = []
    _set(env, _per_core_bars=[
        types.SimpleNamespace(set_value=lambda v: painted.append(v))
        for _ in range(3)])
    env.screen._apply_usage({"per_core": [11, 22, 33]})
    assert painted == [11, 22, 33]
    # A sample taken before the screen was hidden again is not painted.
    stale = env.screen._usage_generation - 1
    env.screen._apply_usage({"per_core": [44, 55, 66]}, stale)
    assert painted == [11, 22, 33]


def _help_that_will_not_move(env):
    from spacr.qt.screens import settings_model

    env.mp.setattr(settings_model, "retarget_field_tooltips", lambda _w: 7)
    assert env.screen.the_name_carries_the_help() == 7
    env.mp.setattr(settings_model, "retarget_field_tooltips", _refusing)
    assert env.screen.the_name_carries_the_help() == 0


def _wells_reported_without_settext(env):
    from spacr.qt.widgets import plate_map_picker

    class Picker:
        _layout_size = 96

        def __init__(self, before, parent=None):
            self.before = before

        def exec(self):
            return True

        def value(self):
            return "A1,B2"

    env.mp.setattr(plate_map_picker, "PlateMapPicker", Picker)
    field = QLineEdit()
    env.qtbot.addWidget(field)
    assert env.screen.pick_wells_for(field, "analysis_excluded_wells") == "A1,B2"
    assert field.text() == "A1,B2"
    readonly = types.SimpleNamespace(text=lambda: "")
    assert env.screen.pick_wells_for(readonly,
                                     "analysis_excluded_wells") == "A1,B2"
    assert "A1,B2 chosen from the 96-well map" in _console_text(
        env.screen._console)


def _model_explainer_follows_only_mounted_signals(env):
    from PySide6.QtWidgets import QCheckBox
    from spacr.qt.widgets.section import Section

    rendered = []
    env.mp.setattr(env.screen, "_refresh_model_explainer",
                   lambda *_a: rendered.append(1))
    env.mp.setattr(env.screen, "_model_explainer",
                   getattr(env.screen, "_model_explainer", None), raising=False)
    env.mp.setattr(env.screen, "_section_explainers",
                   dict(getattr(env.screen, "_section_explainers", None) or {}),
                   raising=False)
    signal_less = _widget(env)
    _set(env, _settings_model=types.SimpleNamespace(
        _widgets={"regression_type": signal_less}))
    first = Section("Model & Inference")
    env.qtbot.addWidget(first)
    env.screen._install_section_explainer(first, "Model & Inference")
    assert rendered == [1]
    box = QCheckBox()
    env.qtbot.addWidget(box)
    env.screen._settings_model = types.SimpleNamespace(
        _widgets={"regression_type": box})
    second = Section("Model & Inference")
    env.qtbot.addWidget(second)
    env.screen._install_section_explainer(second, "Model & Inference")
    assert rendered == [1, 1]
    box.setChecked(True)
    assert len(rendered) == 3


def test_action_row_cards_and_live_status_do_what_they_can(
        subtests, monkeypatch, qtbot, caplog, screen, console_info_enabled):
    """The sweep card opens without a panel or with one that refuses its seed;
    a permutation run gets its own banner and is never "slow"; the usage poll
    bumps its generation with or without a timer and paints every core from one
    sample; help that will not move costs nothing; picked wells are reported
    even when the field cannot hold them; the model explainer follows only
    settings that are mounted and have a commit signal."""
    caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
    cases = {
        "sweep card without panel": _sweep_card_without_panel,
        "sweep refused seed": _sweep_refused_seed,
        "permutation banner": _permutation_banner,
        "usage poll without timer": _usage_poll_without_timer,
        "per-core bars": _per_core_bars_from_one_sample,
        "help will not move": _help_that_will_not_move,
        "wells without setText": _wells_reported_without_settext,
        "model explainer followers": _model_explainer_follows_only_mounted_signals,
    }
    _walk(subtests, monkeypatch, cases, screen=screen, qtbot=qtbot,
          caplog=caplog)


class _Switch:
    def __init__(self, checked=False):
        self._checked = checked
        self.blocked = []

    def isChecked(self):
        return self._checked

    def setChecked(self, on):
        self._checked = bool(on)

    def blockSignals(self, on):
        self.blocked.append(bool(on))


def test_the_umap_gpu_switch_follows_the_panel_with_or_without_a_model(
        subtests, qtbot):
    """The hyperparameter panel decides the GPU state; the switch follows it,
    and the settings model records it only when there is a model."""
    screen = AppScreen("umap")
    qtbot.addWidget(screen)
    screen._hyperparam = types.SimpleNamespace(
        request_gpu_enabled=lambda on, anchor=None: True)
    switch = screen._gpu_switch = _Switch(checked=False)
    recorded = []
    with subtests.test(case="with a model"):
        screen._settings_model = types.SimpleNamespace(
            set_hidden_value=lambda key, value: recorded.append((key, value)))
        screen._on_umap_gpu_switch(True)
        assert recorded == [("gpu", True)]
        assert switch.isChecked() is True
    with subtests.test(case="without a model"):
        screen._settings_model = None
        switch.setChecked(False)
        screen._on_umap_gpu_switch(True)
        assert switch.isChecked() is True
        assert recorded == [("gpu", True)]
