"""The Cells tab offers the ways to annotate, not only the obvious one.

The strategies themselves live in :mod:`spacr.regression_annotation` and
have their own tests. These hold what the TAB is responsible for:

* the menu is real -- every strategy is on it, each with a sentence saying
  what it is for and what it costs, and the named method points at the
  control for its own trap;
* it is a TAB on the montage, not a window, and pressing the button twice
  raises the one tab rather than opening a second;
* a control that cannot act is greyed AND says why: no cells loaded, a
  strategy with no implementation, a setting the chosen strategy does not
  read;
* the wells the positives come from are the wells the coefficient on
  screen named, and they follow the selection;
* a refusal from the module reaches the report rather than the traceback.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtWidgets import QApplication                     # noqa: E402

from spacr import regression_annotation as ra                  # noqa: E402
from spacr.qt.widgets.annotation_strategy_panel import (       # noqa: E402
    STRATEGY_SETTINGS, AnnotationStrategyPanel, wells_of_plans)
from spacr.qt.widgets.cell_montage_view import (               # noqa: E402
    ANNOTATE_TAB_TOOLTIP, CellMontageView)

HIT_WELLS = ("r1_c1", "r1_c2", "r1_c3", "r1_c4")

# --------------------------------------------------------------------------- #
#  A four-well screen on disk
# --------------------------------------------------------------------------- #
#
# WRITTEN HERE RATHER THAN IMPORTED from the neighbouring Cells-tab test file.
# Importing one collected test module into another gives this suite two routes
# to the same module, and the Qt cleanup that runs at every test boundary then
# segfaults in `QApplication.allWidgets()` — in whichever test happens to be
# starting, with nothing pointing back at the import that caused it.

CELL_DIM, NUC_DIM, PATH_DIM = 4, 5, 6
WELLS = ("r1_c1", "r1_c2", "r1_c3", "r1_c4")
OBJECTS_PER_WELL = 8

#: A gene-level coefficient as the results panel spells one.
GENE_KEY = "gene_fraction:gene[GRA14]"
GUIDE_KEY = "fraction:grna[GRA14_1]"


def _field(labels, h=96, w=112, n_channels=4, seed=0):
    """A merged array: four intensity planes then cell / nucleus / pathogen."""
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 4000, size=(h, w, n_channels + 3)).astype(np.uint16)
    for dim in (CELL_DIM, NUC_DIM, PATH_DIM):
        data[:, :, dim] = 0
    for index, label in enumerate(labels):
        y0 = 4 + (index // 4) * 22
        x0 = 4 + (index % 4) * 26
        data[y0:y0 + 18, x0:x0 + 20, CELL_DIM] = label
        data[y0 + 3:y0 + 15, x0 + 3:x0 + 17, NUC_DIM] = label
        data[y0 + 5:y0 + 8, x0 + 5:x0 + 8, PATH_DIM] = label
    return data


def _scores(well_index, n=OBJECTS_PER_WELL):
    start = 0.05 + 0.02 * well_index
    spread = 0.9 - 0.2 * well_index
    return [round(start + spread * i / (n - 1), 4) for i in range(n)]


def _screen(tmp_path, *, with_png=False, with_merged=True):
    """Write a four-well plate: merged arrays, a database, and the counts.

    :returns: ``(root, db_path, results_csv)``.
    """
    root = tmp_path / "exp"
    (root / "measurements").mkdir(parents=True)
    if with_merged:
        (root / "merged").mkdir(parents=True)
    db_path = str(root / "measurements" / "measurements.db")

    labels = list(range(1, OBJECTS_PER_WELL + 1))
    cell_rows, png_rows = [], []
    for well_index, well in enumerate(WELLS):
        row_id, column_id = well.split("_")
        name = f"plate1_{well}_1"
        npy = str(root / "merged" / f"{name}.npy")
        if with_merged:
            np.save(npy, _field(labels, seed=well_index))
        png_dir = root / "data" / f"plate1_{well}" / "cell_png"
        if with_png:
            png_dir.mkdir(parents=True, exist_ok=True)
        for label, score in zip(labels, _scores(well_index)):
            png_path = str(png_dir / f"{name}_{label}.png")
            if with_png:
                from PIL import Image
                crop = np.zeros((32, 32, 3), dtype=np.uint8)
                crop[:, :, 0] = label * 20
                Image.fromarray(crop).save(png_path)
            cell_rows.append((label, "plate1", row_id, column_id, "f1",
                              npy, f"{name}.npy"))
            # MEASUREMENTS ON THE CROP ROWS. A real screen's `png_list` is
            # joined to the object tables before a strategy sees it, and
            # without one there is nothing to fit on but the score.
            #
            # TWO OF THE FOUR MOVE WITH THE SCORE and two do not, so the
            # leakage control has something to drop AND something to keep --
            # a table where every column tracked the score would only ever
            # produce the leaking fit.
            png_rows.append(("plate1", row_id, column_id, "f1", f"o{label}",
                             png_path,
                             f"plate1_{row_id}_{column_id}_f1_o{label}", score,
                             900.0 + 300.0 * score,
                             1200.0 + 600.0 * score,
                             0.40 + 0.05 * ((label * 7) % 5),
                             250.0 + 30.0 * ((label * 3) % 4)))

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE cell (object_label INTEGER, plateID TEXT, "
                 "rowID TEXT, columnID TEXT, fieldID TEXT, path_name TEXT, "
                 "file_name TEXT)")
    conn.executemany("INSERT INTO cell VALUES (?,?,?,?,?,?,?)", cell_rows)
    conn.execute("CREATE TABLE png_list (plateID TEXT, rowID TEXT, "
                 "columnID TEXT, fieldID TEXT, cell_id TEXT, png_path TEXT, "
                 "prcfo TEXT, pred REAL, cell_area REAL, "
                 "cell_channel_1_mean_intensity REAL, "
                 "cell_eccentricity REAL, nucleus_area REAL)")
    conn.executemany("INSERT INTO png_list VALUES "
                     "(?,?,?,?,?,?,?,?,?,?,?,?)", png_rows)
    conn.commit()
    conn.close()

    results = tmp_path / "results"
    results.mkdir()
    fractions = {
        "r1_c1": {"GRA14_1": 0.25, "GRA14_2": 0.25, "OTHER_1": 0.5},
        "r1_c2": {"GRA14_1": 0.125, "OTHER_1": 0.875},
        "r1_c3": {"GRA14_2": 0.5, "OTHER_1": 0.5},
        "r1_c4": {"OTHER_1": 1.0},
    }
    genes = {"GRA14_1": "GRA14", "GRA14_2": "GRA14", "OTHER_1": "OTHER"}
    rows = []
    for well, guides in fractions.items():
        row_id, column_id = well.split("_")
        for guide, fraction in guides.items():
            rows.append({"prc": f"plate1_{well}", "plateID": "plate1",
                         "rowID": row_id, "columnID": column_id,
                         "grna": guide, "gene": genes[guide],
                         "fraction": fraction,
                         "cell_count": OBJECTS_PER_WELL, "pred": 0.5})
    pd.DataFrame(rows).to_csv(results / "regression_data.csv", index=False)
    coefficients = pd.DataFrame([
        {"feature": GENE_KEY, "coefficient": 0.2, "p_value": 1e-4},
        {"feature": GUIDE_KEY, "coefficient": 0.15, "p_value": 2e-3},
        {"feature": "Intercept", "coefficient": 0.5, "p_value": 0.9},
    ])
    results_csv = str(results / "results.csv")
    coefficients.to_csv(results_csv, index=False)
    return str(root), db_path, results_csv


def _rows(db_path):
    """The input table's own row shape -- ``{"plate", "database"}``."""
    return [{"plate": "plate1", "database": db_path}]


# --------------------------------------------------------------------------- #
#  A screen the strategies can actually be run on
# --------------------------------------------------------------------------- #

def _measured(seed: int = 0, wells: int = 12, per_well: int = 40):
    """Object rows with measurements, which the crop tables do not carry."""
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(wells):
        prevalence = 0.4 if well < 4 else 0.05
        for index in range(per_well):
            hit = rng.random() < prevalence
            rows.append({
                "plateID": "plate1",
                "rowID": f"r{1 + well // 4}",
                "columnID": f"c{1 + well % 4}",
                "fieldID": "f1",
                "prcfo": f"plate1_r{1 + well // 4}_c{1 + well % 4}_f1_o{index}",
                "cell_area": rng.normal(900 + 260 * hit, 90),
                "cell_channel_1_mean_intensity": rng.normal(
                    1200 + 500 * hit, 130),
                "cell_eccentricity": rng.normal(0.55 - 0.2 * hit, 0.07),
                "nucleus_area": rng.normal(300 + 60 * hit, 35),
                "cell_channel_2_mean_intensity": rng.normal(
                    500 + 90 * hit, 55),
            })
    frame = pd.DataFrame(rows)
    z = (0.0042 * (frame["cell_channel_1_mean_intensity"] - 1200)
         + 0.0042 * (frame["cell_area"] - 900)
         + rng.normal(0, 0.2, len(frame)))
    frame["pred"] = 1.0 / (1.0 + np.exp(-z))
    return frame


def _drop(widget) -> None:
    """Shut a widget's worker down and let Qt free it when it is ready.

    Deliberately NOT `deleteLater` plus a hand-delivered DeferredDelete
    sweep: destroying widgets out from under the session's own Qt cleanup
    is how this file learned to segfault in `QApplication.allWidgets()`,
    in whichever test happened to start next.
    """
    try:
        widget.shutdown()
    except Exception:                                        # noqa: BLE001
        pass
    widget.close()


@pytest.fixture(scope="module")
def _one_panel():
    """ONE panel for this whole file, reset between tests.

    A panel is around a hundred widgets, and Qt keeps every one of them for
    the session; twenty of them built and abandoned is a retained tree that
    the suite's own bound then collects mid-test. Building one and putting
    its controls back is the same coverage at a twentieth of the cost.
    """
    frame = _measured()
    widget = AnnotationStrategyPanel(
        objects_provider=lambda: frame,
        wells_provider=lambda: HIT_WELLS,
        score_provider=lambda: "pred",
        folder_provider=lambda: "",
        threaded=False)
    yield widget
    _drop(widget)


@pytest.fixture()
def panel(_one_panel):
    """The shared panel with every control back at its default."""
    _one_panel.set_strategy("top_score_random")
    _one_panel._budget.setValue(30)
    _one_panel._seed.setValue(5)
    _one_panel._split.setCurrentIndex(_one_panel._split.findData("well"))
    _one_panel._leakage.setCurrentIndex(_one_panel._leakage.findData("report"))
    _one_panel._model.setCurrentIndex(_one_panel._model.findData("auto"))
    _one_panel._holdout.setValue(0.25)
    _one_panel._measure.setCurrentIndex(_one_panel._measure.findData("margin"))
    _one_panel._clusters.setValue(0)
    _one_panel._bins.setValue(10)
    _one_panel._confidence.setValue(0.900)
    _one_panel._rounds.setValue(5)
    _one_panel._neighbours.setValue(5)
    _one_panel._distance.setValue(0.10)
    _one_panel._label_column.setText("")
    _one_panel._positive_wells.setText("")
    _one_panel._negative_wells.setText("")
    _one_panel._wells.setText(", ".join(HIT_WELLS))
    _one_panel._result = None
    _one_panel._running = False
    _one_panel._report.setPlainText("")
    _one_panel.refresh()
    return _one_panel


# --------------------------------------------------------------------------- #
#  The menu
# --------------------------------------------------------------------------- #

def test_every_strategy_is_on_the_menu(panel):
    offered = [panel._menu.itemData(i) for i in range(panel._menu.count())]
    assert offered == list(ra.strategy_keys())
    assert len(offered) == 10


def test_choosing_one_says_what_it_is_for_and_what_it_costs(panel):
    for key in ra.strategy_keys():
        assert panel.set_strategy(key)
        about = panel.about_text()
        entry = ra.strategy(key)
        assert entry.purpose in about, key
        assert entry.cost in about, key
        assert "Limitations:" in about, key


def test_the_named_method_points_at_the_control_for_its_own_trap(panel):
    panel.set_strategy("top_score_random")
    about = panel.about_text()
    assert "score reconstruction" in about
    assert "The score's own inputs" in about
    assert "WITHOUT them" in about
    # and that control is on the panel, defaulting to reporting both
    assert panel._leakage.currentData() == "report"


def test_an_unimplemented_strategy_is_labelled_and_cannot_be_run(
        monkeypatch):
    """A menu entry that would select nothing must say so before it is run."""
    parked = ra.Strategy(key="parked", title="Parked", purpose="For later.",
                         cost="Nothing yet.", implemented=False)
    monkeypatch.setattr(ra, "STRATEGIES", ra.STRATEGIES + (parked,))
    frame = _measured()
    widget = AnnotationStrategyPanel(objects_provider=lambda: frame,
                                     threaded=False)
    assert widget.set_strategy("parked")
    assert "not yet implemented" in widget._menu.currentText()
    assert "not yet implemented" in widget.about_text().casefold()
    assert not widget._run_button.isEnabled()
    assert "would select nothing" in widget._run_button.toolTip()
    assert widget.run() is False
    _drop(widget)


# --------------------------------------------------------------------------- #
#  Controls that cannot act
# --------------------------------------------------------------------------- #

def test_with_no_cells_loaded_the_panel_says_so_rather_than_fitting():
    widget = AnnotationStrategyPanel(objects_provider=lambda: None,
                                     threaded=False)
    assert not widget._run_button.isEnabled()
    assert "no cells to choose from" in widget.reason()
    assert widget.run() is False
    assert widget.result() is None
    _drop(widget)


@pytest.mark.parametrize("key", ra.strategy_keys())
def test_a_setting_the_strategy_does_not_read_is_greyed_with_the_reason(
        panel, key):
    panel.set_strategy(key)
    wanted = set(STRATEGY_SETTINGS[key])
    for name, (label, widget) in panel._rows.items():
        if name in wanted:
            assert widget.isEnabled(), f"{key} needs {name}"
        else:
            assert not widget.isEnabled(), f"{key} does not read {name}"
            assert "does not read this setting" in widget.toolTip()


def test_the_control_well_boxes_are_live_only_for_the_anchor_strategy(panel):
    panel.set_strategy("control_anchors")
    assert panel._positive_wells.isEnabled()
    assert panel._negative_wells.isEnabled()
    panel.set_strategy("diversity")
    assert not panel._positive_wells.isEnabled()


# --------------------------------------------------------------------------- #
#  Running one
# --------------------------------------------------------------------------- #

def test_the_named_method_runs_end_to_end_and_reports_both_fits(panel):
    panel.set_strategy("top_score_random")
    assert panel.run() is True
    result = panel.result()
    assert result is not None
    report = panel.report_text()
    assert "Including score inputs" in report
    assert "Excluding score inputs" in report
    assert "survives" in report
    assert "held out" in report
    assert set(result.selection["annotation_role"]) == {"positive", "contrast"}
    assert len(result.predictions) > 0
    assert "measured on the hold-out" in panel._status.text()


def test_the_run_takes_its_wells_from_the_coefficient_on_screen(panel):
    assert panel._wells.text() == ", ".join(HIT_WELLS)
    request = panel.request()
    assert tuple(request.wells) == HIT_WELLS
    panel.set_strategy("top_score_random")
    panel.run()
    chosen = set(panel.result().selection.loc[
        panel.result().selection["annotation_role"] == "positive",
        "annotation_group"])
    assert chosen
    for group in chosen:
        assert "_".join(str(group).split("/")[1:]) in HIT_WELLS


def test_the_hold_out_shares_no_well_with_the_selection(panel):
    panel.set_strategy("top_score_random")
    panel.run()
    result = panel.result()
    assert not (set(result.selection["annotation_group"])
                & set(result.holdout["annotation_group"]))


def test_a_refusal_from_the_strategy_reaches_the_report(panel):
    """A refusal only the run can raise lands in the report, not a traceback.

    The well named here is not on this plate, which nothing short of
    resolving the group ids can know -- so it is the refusal that gets past
    the panel's own pre-flight and has to arrive through the worker.
    """
    panel.set_strategy("top_score_random")
    panel._wells.setText("r9_c9")
    assert panel.run() is True
    assert panel.result() is None
    assert "appear in the object table" in panel.report_text()
    assert "did not run" in panel._status.text()
    assert not panel._save_button.isEnabled()


def test_the_independence_level_reaches_the_request(panel):
    index = panel._split.findData("plate")
    panel._split.setCurrentIndex(index)
    assert panel.request().group_by == "plate"
    assert "optimistic" in panel._split.itemText(panel._split.findData("cell"))


def test_the_strategy_settings_reach_the_request(panel):
    panel.set_strategy("neighbour_propagation")
    panel._neighbours.setValue(9)
    panel._distance.setValue(0.4)
    request = panel.request()
    assert request.neighbours == 9
    assert request.distance_quantile == pytest.approx(0.4)


def test_saving_writes_the_selection_the_holdout_and_the_report(panel,
                                                                tmp_path):
    panel.set_strategy("top_score_random")
    panel.run()
    written = panel.save(str(tmp_path))
    assert set(written) == {"selection", "holdout", "predictions", "report"}
    for path in written.values():
        assert os.path.isfile(path)
    assert "annotation_top_score_random" in written["report"]
    assert "survives" in open(written["report"], encoding="utf-8").read()


def test_saving_before_a_run_writes_nothing(panel, tmp_path):
    assert panel.save(str(tmp_path)) == {}
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------- #
#  The tab on the montage
# --------------------------------------------------------------------------- #
#
# ONE LOADED VIEW FOR THE TESTS THAT NEED A MONTAGE, and a fresh one only for
# the two that are about the state before anything is built. A montage view is
# hundreds of widgets that Qt keeps for the session, and building seven of
# them here is what pushes the suite's retained tree past the bound it
# collects at -- in the middle of some other file's test.


@pytest.fixture()
def fresh_view():
    """A montage view with nothing loaded and its Annotate tab unbuilt."""
    view = CellMontageView(threaded=False)
    yield view
    _drop(view)


@pytest.fixture(scope="module")
def _loaded_view(tmp_path_factory):
    """One montage, built once, for every test that needs cells on screen."""
    root, db_path, results_csv = _screen(tmp_path_factory.mktemp("montage"),
                                         with_png=True)
    frame = pd.read_csv(results_csv)
    view = CellMontageView(frame_provider=lambda: frame,
                           results_provider=lambda: results_csv,
                           database_provider=lambda: _rows(db_path),
                           threaded=False)
    view.set_coefficient(GENE_KEY)
    view.build()
    assert view.plans(), "the montage drew nothing, so these prove nothing"
    yield view
    _drop(view)


@pytest.fixture()
def loaded_view(_loaded_view):
    """The shared montage with the Summary tab in front again."""
    _loaded_view._tabs.setCurrentIndex(0)
    return _loaded_view


def test_the_tab_is_there_before_a_montage_is_and_says_why_it_cannot_run(
        fresh_view):
    """A tab that cannot be filled says why rather than being absent."""
    assert "Annotate" in fresh_view.tab_labels()
    # NAMED FROM THE START, FILLED ON OPENING. The tab is where a user finds
    # the strategies; the forty controls behind it are built when they are
    # asked for.
    assert fresh_view._annotation_panel is None
    panel = fresh_view.annotate_the_cells()
    assert panel is not None
    assert not panel._run_button.isEnabled()
    assert "no cells to choose from" in panel.reason()
    assert "no cells to choose from" in panel._status.text()
    assert panel.run() is False
    assert panel.minimumSizeHint().width() <= 520
    # THE CELLS TAB LIVES IN A SPLITTER THAT FLOORS AT 520 px, and a minimum
    # wider than the panel it sits in forces the whole regression screen
    # wider -- which is why the strategies are a tab rather than a fourth
    # button on a toolbar that already has three.
    assert fresh_view.minimumSizeHint().width() <= 560


def test_opening_the_tab_is_what_builds_it(fresh_view):
    """Selecting the tab builds its content once, and only once."""
    index = fresh_view._tabs.indexOf(fresh_view._annotation_page)
    assert index == 1
    assert fresh_view._annotation_panel is None
    fresh_view._tabs.setCurrentIndex(index)
    # POSTED, NOT BUILT IN THE HANDLER: the content arrives on the next turn
    # of the event loop, so the tab change itself is not carrying a hundred
    # new widgets while Qt is still unwinding it.
    assert fresh_view._annotation_panel is None
    QApplication.processEvents()
    built = fresh_view._annotation_panel
    assert built is not None
    fresh_view._tabs.setCurrentIndex(0)
    fresh_view._tabs.setCurrentIndex(index)
    QApplication.processEvents()
    assert fresh_view._annotation_panel is built


def test_the_annotate_tab_cannot_be_closed_and_names_the_strategies(
        loaded_view):
    """A fixed tab, like Summary, whose hover says what is on it."""
    from PySide6.QtWidgets import QTabBar

    index = loaded_view._tabs.indexOf(loaded_view._annotation_page)
    # SUMMARY, COMPARE, ANNOTATE, then one tab per well: the fixed tabs come
    # first and in the order the montage builds them.
    assert loaded_view.tab_labels()[:3] == ("Summary", "Compare", "Annotate")
    assert index == 2
    bar = loaded_view._tabs.tabBar()
    for side in (QTabBar.LeftSide, QTabBar.RightSide):
        assert bar.tabButton(index, side) is None
    assert loaded_view._tabs.tabToolTip(index) == ANNOTATE_TAB_TOOLTIP
    assert "top-scoring cells against a matched random draw" in \
        ANNOTATE_TAB_TOOLTIP


def test_raising_it_keeps_the_one_tab_rather_than_opening_a_second(
        loaded_view):
    before = loaded_view._tabs.count()
    panel = loaded_view.annotate_the_cells()
    assert panel is loaded_view._annotation_panel
    assert loaded_view._tabs.count() == before
    assert loaded_view._tabs.currentWidget() is loaded_view._annotation_page

    loaded_view._tabs.setCurrentIndex(0)
    again = loaded_view.annotate_the_cells()
    assert again is panel
    assert loaded_view._tabs.count() == before
    assert loaded_view._tabs.currentWidget() is loaded_view._annotation_page


def test_the_tab_is_filled_with_the_wells_the_montage_picked(loaded_view):
    panel = loaded_view.annotate_the_cells()
    named = [w.strip() for w in panel._wells.text().split(",") if w.strip()]
    assert named == list(wells_of_plans(loaded_view.plans()))
    assert named, "the plans named no wells, so the box proves nothing"


def test_the_named_method_runs_through_the_tab_on_a_real_screen(loaded_view):
    """The whole path: a montage on screen, the tab, and both fits.

    The object rows come out of a real ``measurements.db`` through the
    montage's own loader, the wells come from the coefficient's plans, and
    the report is the module's own -- so this is the feature working rather
    than its parts agreeing with each other.
    """
    panel = loaded_view.annotate_the_cells()
    panel.set_strategy("top_score_random")
    panel._budget.setValue(4)
    panel._holdout.setValue(0.25)
    panel._result = None
    assert panel.run() is True
    result = panel.result()
    assert result is not None, panel.report_text()
    report = panel.report_text()
    assert "Including score inputs" in report
    assert "Excluding score inputs" in report
    assert "held out" in report
    assert not (set(result.selection["annotation_group"])
                & set(result.holdout["annotation_group"]))


def test_a_finished_run_reaches_the_montages_status_line(loaded_view):
    """A user who ran a strategy and went back to the pictures is told."""
    panel = loaded_view.annotate_the_cells()
    panel.set_strategy("random_holdout")
    panel._budget.setValue(4)
    panel._result = None
    assert panel.run() is True
    assert panel.result() is not None, panel.report_text()
    assert "Annotate:" in loaded_view.status_text()
    assert "Annotate tab" in loaded_view.status_text()
