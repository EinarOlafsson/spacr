"""Item 471 slice C, screens P-Z (and QC): every figure and table folds.

Pipeline graph, Plate view, Power, Profiler, Project browser, QC dashboard,
Queue, Report, Run compare, Run history, Tabulate (with the pivot builder),
Train compare, Trellis (with its panel) and Volcano (with its explorer).

For each: its figures and tables are :class:`FoldSection` headings that fold
the body away, and a folded heading sits at the bottom of its room; where two
of them share an edge the splitter remembers a drag under the screen's key;
and a panel the screen keeps hidden stays hidden through the folding.

Offscreen, CPU-only, offline. Preferences go to the conftest sandbox.
"""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt              # noqa: E402
from PySide6.QtGui import QMouseEvent                       # noqa: E402
from PySide6.QtWidgets import QApplication                  # noqa: E402

from spacr.qt.widgets.collapsible_splitter import (         # noqa: E402
    CollapsibleSplitter, FoldSection, get_pane_extents)

pytestmark = pytest.mark.qt


def _pump(n: int = 10) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _click(label) -> None:
    at = QPointF(3, 3)
    QApplication.sendEvent(label, QMouseEvent(
        QEvent.MouseButtonRelease, at, at, Qt.LeftButton, Qt.LeftButton,
        Qt.NoModifier))


def _qc_project(tmp_path):
    folder = tmp_path / "measurements"
    folder.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(folder / "measurements.db"))
    try:
        connection.execute(
            "CREATE TABLE cell (object_label INTEGER, cell_area REAL, "
            "measurement_ndim INTEGER, measurement_units TEXT)")
        connection.execute("INSERT INTO cell VALUES (1, 120.0, 2, 'px')")
        connection.commit()
    finally:
        connection.close()
    return str(tmp_path)


def _build(name, tmp_path):
    if name == "qc_dashboard":
        from spacr.qt.screens.qc_dashboard import QCDashboardScreen
        return QCDashboardScreen(threaded=False, src=_qc_project(tmp_path))
    if name == "run_compare":
        from spacr.qt.screens.run_compare import RunCompareScreen
        return RunCompareScreen()
    if name == "power":
        from spacr.qt.screens.power import PowerScreen
        return PowerScreen(threaded=False)
    if name == "tabulate":
        from spacr.qt.screens.tabulate import TabulateScreen
        return TabulateScreen(threaded=False)
    if name == "trellis":
        from spacr.qt.screens.trellis import TrellisScreen
        return TrellisScreen(threaded=False)
    if name == "train_compare":
        from spacr.qt.screens.train_compare import TrainCompareScreen
        return TrainCompareScreen()
    if name == "volcano":
        from spacr.qt.screens.volcano import _make_screen
        return _make_screen()
    if name == "profiler":
        from spacr.qt.screens.profiler import ProfilerScreen
        return ProfilerScreen()
    if name == "run_history":
        from spacr.qt.screens.run_history import RunHistoryScreen
        return RunHistoryScreen()
    if name == "plate_view":
        from spacr.qt.screens.plate_view import PlateViewScreen
        return PlateViewScreen()
    if name == "pipeline_graph":
        from spacr.qt.screens.pipeline_graph import PipelineGraphScreen
        return PipelineGraphScreen(threaded=False)
    if name == "project_browser":
        from spacr.qt.screens.project_browser import ProjectBrowserScreen
        return ProjectBrowserScreen(threaded=False)
    if name == "queue":
        from spacr.qt.screens.queue import QueueScreen
        return QueueScreen()
    if name == "report":
        from spacr.qt.screens.report import ReportScreen
        return ReportScreen()
    raise AssertionError(name)


def _shown(qtbot, name, tmp_path):
    screen = _build(name, tmp_path)
    qtbot.addWidget(screen)
    screen.resize(1366, 768)
    screen.show()
    _pump()
    return screen


#: Every screen, and the sections it must offer.
SECTIONS = {
    "qc_dashboard": {"Segmentation", "Measurement units", "Train/test leakage",
                     "Plate effects", "Annotator agreement"},
    "run_compare": {"Comparison"},
    "power": {"Caveats", "Power vs cells per well", "Power vs wells",
              "Power table"},
    "tabulate": {"Pivot", "Graph", "Filter", "Pivot fields", "Pivot table"},
    "trellis": {"Channels", "Trellis", "Filter and columns"},
    "train_compare": {"Runs", "Curves", "Settings diff"},
    "volcano": {"Volcano plot", "Selected point"},
    "profiler": {"Inputs", "Curve"},
    "run_history": {"Runs", "Run details"},
    "plate_view": {"Plate map", "Edge-effect report"},
    "pipeline_graph": {"Pipeline graph", "Details"},
    "project_browser": {"Projects", "Project details"},
    "queue": {"Queue"},
    "report": {"Sections found in this folder:"},
}


def _sections(screen):
    return {s.folder.name: s for s in screen.findChildren(FoldSection)}


@pytest.mark.parametrize("name", sorted(SECTIONS))
def test_every_figure_and_table_folds_to_its_heading_at_the_bottom(
        qtbot, tmp_path, name):
    screen = _shown(qtbot, name, tmp_path)
    found = _sections(screen)
    assert SECTIONS[name] <= set(found), sorted(found)
    for section_name in sorted(SECTIONS[name]):
        section = found[section_name]
        assert section.heading.objectName() == "FoldHeading"
        _click(section.heading)
        _pump()
        assert section.shut, section_name
        assert not section.body.isVisible(), section_name
        assert (section.heading.geometry().bottom()
                >= section.height() - 4), section_name
        section.set_folded(False)
        _pump()
        assert not section.shut
        assert section.body.isVisible(), section_name


def _splitter(screen, name):
    explorer = getattr(screen, "explorer", None)
    return {
        "power": lambda: screen._results_splitter,
        "tabulate": lambda: screen._stack_splitter,
        "pivot": lambda: screen.pivot._splitter,
        "trellis": lambda: screen._body_splitter,
        "trellis_panel": lambda: screen.panel._splitter,
        "train_compare": lambda: screen._curves_splitter,
        "volcano": lambda: explorer._plot_splitter,
        "profiler": lambda: screen._curve_splitter,
        "run_history": lambda: screen._body_splitter,
        "plate_view": lambda: screen._body_splitter,
        "pipeline_graph": lambda: screen._body_splitter,
        "project_browser": lambda: screen._body_splitter,
    }[name]()


#: (screen, which splitter, its key, the pane that is dragged larger).
DRAGS = [
    ("power", "power", "power::results", "Power table"),
    ("tabulate", "tabulate", "tabulate::stack", "Graph"),
    ("tabulate", "pivot", "pivot_builder::panel", "Pivot table"),
    ("trellis", "trellis", "trellis::body", "Plot"),
    ("trellis", "trellis_panel", "trellis_view::panel", "Trellis"),
    ("train_compare", "train_compare", "train_compare::curves",
     "Settings diff"),
    ("volcano", "volcano", "volcano_explorer::plot", "Selected point"),
    ("profiler", "profiler", "profiler::curve", "Held values"),
    ("run_history", "run_history", "run_history::body", "Run details"),
    ("plate_view", "plate_view", "plate_view::body", "Edge-effect report"),
    ("pipeline_graph", "pipeline_graph", "pipeline_graph::body", "Details"),
    ("project_browser", "project_browser", "project_browser::body",
     "Project details"),
]


@pytest.mark.parametrize("name,which,key,pane", DRAGS,
                         ids=[d[2] for d in DRAGS])
def test_a_dragged_edge_is_remembered_under_the_screens_key(
        qtbot, tmp_path, name, which, key, pane):
    screen = _shown(qtbot, name, tmp_path)
    split = _splitter(screen, which)
    assert isinstance(split, CollapsibleSplitter)
    index = split.indexOf(split.pane(pane).widget)
    sizes = split.sizes()
    total = sum(sizes)
    moved = list(sizes)
    grow = max(40, total // 10)
    donor = 0 if index != 0 else 1
    moved[index] += grow
    moved[donor] = max(1, moved[donor] - grow)
    split.setSizes(moved)
    split.splitterMoved.emit(split.sizes()[0], 1)
    stored = get_pane_extents(key)
    assert stored.get(pane, 0) == split.sizes()[index] > 0


def test_a_rebuilt_screen_reopens_at_the_dragged_size(qtbot, tmp_path):
    first = _shown(qtbot, "run_history", tmp_path)
    split = first._body_splitter
    total = sum(split.sizes())
    split.setSizes([total // 4, total - total // 4])
    split.splitterMoved.emit(total // 4, 1)
    wanted = split.sizes()[1]
    first.close()

    second = _shown(qtbot, "run_history", tmp_path)
    _pump()
    assert abs(second._body_splitter.sizes()[1] - wanted) <= 16


def test_a_folded_table_hands_its_room_to_the_one_beside_it(qtbot, tmp_path):
    screen = _shown(qtbot, "run_history", tmp_path)
    split = screen._body_splitter
    before = split.sizes()[1]
    split.set_collapsed("Runs", True, by_user=True)
    _pump()
    assert split.sizes()[1] > before
    runs = split.pane("Runs").widget
    assert runs.height() <= runs.heading.sizeHint().height() + 12
    split.set_collapsed("Runs", False, by_user=True)


def test_hidden_panels_stay_hidden_through_the_folds(qtbot, tmp_path):
    power = _shown(qtbot, "power", tmp_path)
    caveats = _sections(power)["Caveats"]
    rest = power._caveats._rest
    assert rest.isHidden()
    caveats.set_folded(True)
    caveats.set_folded(False)
    _pump()
    assert rest.isHidden(), "unfolding the caveats showed the held ones"

    volcano = _shown(qtbot, "volcano", tmp_path)
    plot = _sections(volcano)["Volcano plot"]
    problems = volcano.explorer._problem_line
    assert problems.isHidden()
    plot.set_folded(True)
    plot.set_folded(False)
    _pump()
    assert problems.isHidden()

    tabulate = _shown(qtbot, "tabulate", tmp_path)
    for section in _sections(tabulate).values():
        section.set_folded(True)
        section.set_folded(False)
    _pump()
    assert tabulate._table_picker.isHidden()


def test_a_qc_card_keeps_its_verdict_on_the_folded_heading(qtbot, tmp_path):
    screen = _shown(qtbot, "qc_dashboard", tmp_path)
    before = screen.visible_text()
    card = _sections(screen)["Measurement units"]
    card.set_folded(True)
    _pump()
    chips = [w for w in card.findChildren(type(card.heading))
             if w.text().startswith("[") and w.isVisible()]
    assert chips, "a folded card must still say its verdict"
    assert screen.visible_text() == before
    card.set_folded(False)
