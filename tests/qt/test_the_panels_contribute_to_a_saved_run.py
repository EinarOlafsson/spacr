"""Instruction 180, the Qt half — the panels describe themselves.

:mod:`spacr.workspace` holds the document and the copying, and its own tests
cover both. These are about the CONTRIBUTORS: that the regression panel, the
montage, the attached databases and the figure grid each hand over what a
user would notice going missing, and take it back.

The property that matters throughout is that a section is state a panel OWNS.
Nothing here reaches into a widget's private attributes to check a restore
worked — it asks the panel the same question the collector does, which is the
only check that stays true when the panel changes.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr import workspace                               # noqa: E402
from spacr.qt.widgets.cell_montage_view import CellMontageView   # noqa: E402


GENE_KEY = "gene_fraction:gene[GRA14]"


@pytest.fixture(autouse=True)
def _no_leftover_providers():
    workspace.clear_providers()
    yield
    workspace.clear_providers()


# --------------------------------------------------------------------------- #
#  The regression panel
# --------------------------------------------------------------------------- #

def _panel(qtbot):
    pytest.importorskip("pyqtgraph")
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    return panel


def _results(tmp_path, name="ols_1"):
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    csv = folder / "results.csv"
    pd.DataFrame([
        {"feature": GENE_KEY, "coefficient": 1.5, "p_value": 1e-6},
        {"feature": "gene_fraction:gene[ROP18]", "coefficient": -0.8,
         "p_value": 0.4},
    ]).to_csv(csv, index=False)
    return csv


def test_the_regression_panel_offers_every_runs_view_not_only_the_one_on_screen(
        qtbot, tmp_path):
    """The four views a user built and left are the ones that would be lost."""
    panel = _panel(qtbot)
    first, second = _results(tmp_path, "ols_1"), _results(tmp_path, "ols_2")
    panel.load(str(first))
    panel.load(str(second))                 # leaving ols_1 stores its view

    state = panel.workspace_state()
    assert str(second) == state["path"]
    assert {os.path.basename(k) for k in state["runs"]} >= {"ols_1", "ols_2"}


def test_a_restored_workspace_reopens_the_run_and_reads_its_table(qtbot, tmp_path):
    """`_path` without the table behind it is a panel claiming a run it never read."""
    csv = _results(tmp_path)
    saved = _panel(qtbot)
    saved.load(str(csv))
    saved.set_level("gene")
    document = saved.workspace_state()

    fresh = _panel(qtbot)
    assert fresh.apply_workspace_state(document) is True
    assert fresh.results_frame() is not None
    assert len(fresh.results_frame()) == 2


def test_restoring_into_a_session_that_has_runs_open_keeps_them(qtbot, tmp_path):
    """The store is merged: views built since the save are not dropped."""
    first, second = _results(tmp_path, "ols_1"), _results(tmp_path, "ols_2")
    saved = _panel(qtbot)
    saved.load(str(first))
    saved.load(str(second))
    document = saved.workspace_state()

    fresh = _panel(qtbot)
    third = _results(tmp_path, "glm_9")
    fresh.load(str(third))
    fresh.load(str(first))                   # stores glm_9
    fresh.apply_workspace_state(document)

    names = {os.path.basename(k) for k in fresh.workspace_state()["runs"]}
    assert "glm_9" in names and "ols_2" in names


def test_a_run_that_moved_leaves_the_remembered_views_and_says_nothing_opened(
        qtbot, tmp_path):
    saved = _panel(qtbot)
    saved.load(str(_results(tmp_path, "ols_1")))
    document = saved.workspace_state()
    document["path"] = str(tmp_path / "gone" / "results.csv")

    fresh = _panel(qtbot)
    assert fresh.apply_workspace_state(document) is True      # the views arrived
    assert fresh.results_frame() is None                      # the run did not


# --------------------------------------------------------------------------- #
#  The montage
# --------------------------------------------------------------------------- #

def test_the_montage_contributes_the_choice_and_the_settings_not_the_pixels(qtbot):
    view = CellMontageView(threaded=False)
    qtbot.addWidget(view)
    view.set_coefficient(GENE_KEY)

    state = view.workspace_state()
    assert state["coefficient"] == GENE_KEY
    assert state["level"] == "gene"
    assert isinstance(state["picture_settings"], dict)
    assert state["montage_shown"] is False
    # The crops are tens of megabytes and the run's own images regenerate
    # them exactly. Nothing here is an image.
    assert not any(k in state for k in ("images", "crops", "pixmaps"))


def test_the_montages_settings_come_back_and_the_crops_are_not_reloaded(qtbot):
    saved = CellMontageView(threaded=False)
    qtbot.addWidget(saved)
    saved.set_coefficient(GENE_KEY)
    saved._picture_settings = {"normalize_channels": "r,g,b", "img_size": 128}
    document = saved.workspace_state()

    fresh = CellMontageView(threaded=False)
    qtbot.addWidget(fresh)
    assert fresh.apply_workspace_state(document) is True
    assert fresh.workspace_state()["coefficient"] == GENE_KEY
    assert fresh.picture_settings()["normalize_channels"] == "r,g,b"
    # Deliberately: loading the crops is the slow half and a restore that
    # started it would freeze the window the user just opened.
    assert fresh.plans() == ()


def test_the_montage_writes_down_which_cells_the_claim_rested_on(qtbot, tmp_path):
    from tests.qt.test_cells_behind_the_dot_tab import _view

    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)
    view.build()

    picked = view.workspace_state()["picked_groups"]
    assert picked and all(isinstance(v, list) for v in picked.values())


# --------------------------------------------------------------------------- #
#  The attached databases
# --------------------------------------------------------------------------- #

def test_the_attached_set_keeps_its_order_because_a_merge_depends_on_it(qtbot,
                                                                       tmp_path):
    from spacr.qt.widgets.database_set import DatabaseSetWidget

    first, second = tmp_path / "plate1", tmp_path / "plate2"
    for folder in (first, second):
        folder.mkdir()
    widget = DatabaseSetWidget(mode="database")
    qtbot.addWidget(widget)
    widget.set_value([str(second), str(first)])

    state = widget.workspace_state()
    assert state["sources"] == [str(second), str(first)]
    assert len(state["databases"]) == 2


def test_one_moved_plate_does_not_cost_the_user_the_others(qtbot, tmp_path):
    from spacr.qt.widgets.database_set import DatabaseSetWidget

    here, gone = tmp_path / "plate1", tmp_path / "plate2"
    here.mkdir()
    widget = DatabaseSetWidget(mode="database")
    qtbot.addWidget(widget)
    widget.set_value([str(here), str(gone)])
    document = widget.workspace_state()

    fresh = DatabaseSetWidget(mode="database")
    qtbot.addWidget(fresh)
    assert fresh.apply_workspace_state(document) is True
    assert fresh.sources() == [str(here)]


def test_a_set_whose_every_source_is_gone_declines_rather_than_clearing(qtbot,
                                                                       tmp_path):
    from spacr.qt.widgets.database_set import DatabaseSetWidget

    widget = DatabaseSetWidget(mode="database")
    qtbot.addWidget(widget)
    assert widget.apply_workspace_state(
        {"sources": [str(tmp_path / "nowhere")]}) is False


# --------------------------------------------------------------------------- #
#  The figure grid
# --------------------------------------------------------------------------- #

def test_the_grid_contributes_the_arrangement_and_not_seventeen_pngs(qtbot):
    from spacr.qt.widgets.figure_grid_view import FigureGridView

    grid = FigureGridView()
    qtbot.addWidget(grid)
    grid.set_target_cell_width(320)
    grid.set_section_collapsed("ols_1", 0, True)

    state = grid.workspace_state()
    assert state["cell_width"] == 320
    assert state["collapsed"] == [["ols_1", 0]]

    fresh = FigureGridView()
    qtbot.addWidget(fresh)
    assert fresh.apply_workspace_state(state) is True
    assert fresh.is_section_collapsed("ols_1", 0) is True
    assert fresh.workspace_state()["cell_width"] == 320


# --------------------------------------------------------------------------- #
#  End to end, through the registry
# --------------------------------------------------------------------------- #

def test_a_screen_enrols_its_panels_and_withdraws_them_when_it_closes(qtbot,
                                                                     tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    names = set(workspace.providers())
    assert {"regression:settings", "regression:montage",
            "regression:regression", "regression:figures"} <= names

    screen.close()
    assert not any(n.startswith("regression:") for n in workspace.providers())


def test_the_document_a_finished_run_writes_names_the_panels_that_were_open(
        qtbot, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    run = tmp_path / "run"
    assert workspace.save_for_run(run, {}, app_key="regression") is not None

    sections = workspace.load(run)["sections"]
    assert "regression:settings" in sections
    assert isinstance(sections["regression:settings"]["settings"], dict)
    # A panel the screen did not build is simply absent, not a problem: every
    # screen registers the same names and most build none of them.
    assert "problems" not in workspace.load(run) or all(
        not p["section"].startswith("regression:")
        for p in workspace.load(run)["problems"])


# --------------------------------------------------------------------------- #
#  Getting it back from the Runs tab
# --------------------------------------------------------------------------- #

class Panel:
    """A contributor with nothing else about it — the protocol, on its own."""

    def __init__(self, state=None):
        self.state = dict(state or {})
        self.applied = None

    def workspace_state(self):
        return dict(self.state)

    def apply_workspace_state(self, state):
        self.applied = dict(state)
        return True


def _saved_run(tmp_path, panel_state=None):
    """A run folder carrying a workspace bundle."""
    run = tmp_path / "run"
    workspace.register("volcano", lambda: Panel(panel_state or {"level": "gene"}))
    workspace.save_for_run(run, {}, app_key="regression")
    workspace.clear_providers()
    return run


def test_a_run_with_a_bundle_offers_to_restore_it_and_one_without_says_why(
        qtbot, tmp_path):
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    saved, bare = _saved_run(tmp_path), tmp_path / "bare"
    bare.mkdir()

    with_bundle = panel._build_run_menu([{"folder": str(saved)}])
    without = panel._build_run_menu([{"folder": str(bare)}])

    offered = next(a for a in with_bundle.actions() if a.data() == "restore")
    absent = next(a for a in without.actions() if a.data() == "restore")
    assert offered.isEnabled() is True
    # Instruction 106: offered and disabled, SAYING WHY -- an entry that
    # appeared only sometimes is one nobody learns exists.
    assert absent.isEnabled() is False
    assert "saved no workspace" in absent.toolTip()


def test_restoring_is_its_own_gesture_and_not_part_of_loading(qtbot, tmp_path):
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    record = {"folder": str(_saved_run(tmp_path))}
    asked = []
    panel.workspace_restore_requested.connect(asked.append)

    assert panel._apply_run_menu("restore", [record]) is True
    assert asked == [record]


def test_the_restore_report_reaches_the_user_not_the_void(qtbot, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    run = _saved_run(tmp_path, {"level": "gene"})
    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    said = []
    screen._say = said.append

    report = screen.restore_run_workspace({"folder": str(run)})
    # Nothing on this screen owns a section called "volcano", so nothing came
    # back -- and that is exactly what has to be said out loud.
    assert report["restored"] == []
    assert any("volcano" in line for line in said)


def test_a_folder_with_no_bundle_says_so_rather_than_raising(qtbot, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    said = []
    screen._say = said.append
    bare = tmp_path / "bare"
    bare.mkdir()

    assert screen.restore_run_workspace({"folder": str(bare)})["restored"] == []
    assert any("no saved workspace" in line for line in said)
