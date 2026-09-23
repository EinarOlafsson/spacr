"""Item 471 slice C, screens A to F: every figure and table folds, and drags.

Agreement, Align, Annotate, Batch, Classifier Evaluation, Control Chart,
Convert, Curate, Data Manager, DB Browser, Distributed Jobs, Dose-Response,
Embeddings, Experiment Design, Feature Explorer and Foreign import each put
their figures, tables and side panels under the one heading control
(:class:`~spacr.qt.widgets.collapsible_splitter.FoldSection`), and where two
of them share an edge, in a
:class:`~spacr.qt.widgets.collapsible_splitter.CollapsibleSplitter`.

For every screen these tests check that

* each named section is there, folds by a click on its heading, and folded,
  its heading sits at the bottom of the room it is given -- and when every
  section of a column is folded, the last heading is at the column's bottom;
* a size the user drags is stored under the screen's key and a rebuilt
  screen opens at it;
* folding and unfolding everything never shows a widget the screen keeps
  hidden (Annotate's Console + AI pane, the raw-SQL field, a table picker),
  which is the item 284/380 rule for lazy and switched panels.

The QSettings store is sandboxed per test by the suite's conftest, so the
real keys written here never reach a user's preferences.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt              # noqa: E402
from PySide6.QtGui import QMouseEvent                       # noqa: E402
from PySide6.QtWidgets import (QApplication, QStackedWidget,  # noqa: E402
                               QWidget)

from spacr.qt.widgets import collapsible_splitter as cs     # noqa: E402


def _pump(n: int = 10) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _click(label) -> None:
    at = QPointF(3, 3)
    QApplication.sendEvent(label, QMouseEvent(
        QEvent.MouseButtonRelease, at, at, Qt.LeftButton, Qt.LeftButton,
        Qt.NoModifier))


def _agreement():
    from spacr.qt.screens.agreement import AgreementScreen
    return AgreementScreen(threaded=False)


def _align():
    from spacr.qt.screens.align import AlignScreen
    return AlignScreen(threaded=False)


def _annotate():
    from spacr.qt.screens.annotate import AnnotateScreen
    return AnnotateScreen()


def _batch():
    from spacr.qt.screens.batch import BatchScreen
    return BatchScreen()


def _classifier_evaluation():
    from spacr.qt.screens.classifier_evaluation import (
        ClassifierEvaluationScreen)
    return ClassifierEvaluationScreen(threaded=False)


def _control_chart():
    from spacr.qt.screens.control_chart import ControlChartScreen
    return ControlChartScreen(threaded=False)


def _convert():
    from spacr.qt.screens.convert import ConvertScreen
    return ConvertScreen(threaded=False)


def _curate():
    from spacr.qt.screens.curate import CurateScreen
    return CurateScreen()


def _data_manager():
    from spacr.qt.screens.data_manager import DataManagerScreen
    return DataManagerScreen(threaded=False)


def _db_browser():
    from spacr.qt.screens.db_browser import DbBrowserScreen
    return DbBrowserScreen(threaded=False)


def _distributed_jobs(tmp_path):
    from spacr.qt.screens.distributed_jobs import DistributedJobsScreen
    from spacr.remote_execution import (JobStore, ProfileStore,
                                        RemoteJobManager)
    manager = RemoteJobManager(ProfileStore(tmp_path / "profiles.json"),
                               JobStore(tmp_path / "jobs.json"),
                               lambda *a, **k: None)
    return DistributedJobsScreen(manager=manager, threaded=False,
                                 auto_poll=False)


def _dose_response():
    from spacr.qt.screens.dose_response import DoseResponseScreen
    return DoseResponseScreen(threaded=False)


def _embeddings():
    from spacr.qt.screens.embeddings import EmbeddingsScreen
    return EmbeddingsScreen(threaded=False)


def _experiment_design():
    from spacr.qt.screens.experiment_design import ExperimentDesignScreen
    return ExperimentDesignScreen(threaded=False)


def _feature_explorer():
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.screens.feature_explorer import FeatureExplorerScreen
    return FeatureExplorerScreen(link=LinkedSelection(), threaded=False)


def _foreign():
    from spacr.qt.screens.foreign import ForeignScreen
    return ForeignScreen(threaded=False)


#: (id, factory, section names, [(splitter path, every pane a section)])
#: The splitter path is the attribute chain from the screen; "all sections"
#: marks a column whose every pane is a section, where folding them all must
#: stack the headings at the column's bottom.
SCREENS = [
    ("agreement", _agreement,
     ["Annotation columns", "Pairwise agreement", "Confusion matrix",
      "Disagreeing rows", "Crop preview"],
     [("_body_splitter", False)]),
    ("align", _align, ["Tile layout", "Plan report"],
     [("_body_splitter", False)]),
    ("annotate", _annotate, ["Crops", "Console + AI"],
     [("_runtime_splitter", False)]),
    ("batch", _batch, ["Job queue", "Validation problems", "Job log"],
     [("_body_splitter", True)]),
    ("classifier_evaluation", _classifier_evaluation,
     ["Evaluation results", "Confusion table", "Error inspector"],
     [("_confusion_splitter", True)]),
    ("control_chart", _control_chart,
     ["Control chart", "Report", "Rule violations"],
     [("_body_splitter", False)]),
    ("convert", _convert, ["Conversion plan", "Summary"],
     [("_body_splitter", True)]),
    ("curate", _curate, ["Image", "Curation tools"],
     [("_body_splitter", False)]),
    ("data_manager", _data_manager,
     ["Project data", "Can be deleted", "Kept, and why"],
     [("_prune_splitter", True)]),
    ("db_browser", _db_browser, ["Tables", "Table preview"],
     [("_body_splitter", False)]),
    ("distributed_jobs", _distributed_jobs, ["Jobs", "Job details and log"],
     [("_body_splitter", True)]),
    ("dose_response", _dose_response,
     ["Dose-response curves", "Fit results", "Report"],
     [("_body_splitter", False)]),
    ("embeddings", _embeddings, ["Crop preview"], []),
    ("experiment_design", _experiment_design,
     ["Conditions", "Plate map", "Findings"],
     [("_body_splitter", True)]),
    ("feature_explorer", _feature_explorer,
     ["Feature ranking", "Distributions", "Filter and columns"],
     [("_body_splitter", False), ("explorer._body_splitter", False)]),
    ("foreign", _foreign, ["Column mapping", "Import report"],
     [("_body_splitter", True)]),
]

IDS = [case[0] for case in SCREENS]


def _build(qtbot, factory, tmp_path):
    screen = (factory(tmp_path) if factory is _distributed_jobs
              else factory())
    qtbot.addWidget(screen)
    screen.resize(1500, 1000)
    screen.show()
    _pump()
    return screen


def _sections(screen) -> dict:
    return {s.folder.name: s for s in screen.findChildren(cs.FoldSection)}


def _resolve(screen, path):
    target = screen
    for part in path.split("."):
        target = getattr(target, part)
    return target


def _into_view(widget) -> None:
    """Turn every page stack above ``widget`` to the page holding it."""
    child, parent = widget, widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QStackedWidget) and parent.indexOf(child) >= 0:
            parent.setCurrentWidget(child)
        child, parent = parent, parent.parentWidget()
    _pump()


def _explicitly_hidden(screen) -> list:
    return [w for w in screen.findChildren(QWidget)
            if w.testAttribute(Qt.WA_WState_ExplicitShowHide)
            and w.isHidden()]


@pytest.mark.qt
@pytest.mark.parametrize("key,factory,names,splitters", SCREENS, ids=IDS)
def test_each_section_folds_by_its_heading_and_sits_at_the_bottom(
        qtbot, tmp_path, key, factory, names, splitters):
    screen = _build(qtbot, factory, tmp_path)
    found = _sections(screen)
    missing = [n for n in names if n not in found]
    assert not missing, f"{key}: no section called {missing}"
    for name in names:
        section = found[name]
        assert section.heading.objectName() == "FoldHeading"
        if section.isHidden():
            continue
        _into_view(section)
        _click(section.heading)
        _pump()
        assert section.shut, f"{key}/{name}: the heading click did not fold"
        assert not section.body.isVisible()
        assert section.heading.geometry().bottom() >= section.height() - 8, (
            f"{key}/{name}: the folded heading is not at the bottom")
        section.set_folded(False)
        _pump()
        assert not section.shut
        assert section.body.isVisible(), f"{key}/{name}: did not reopen"


@pytest.mark.qt
@pytest.mark.parametrize(
    "key,factory,path",
    [(c[0], c[1], p) for c in SCREENS for p, every in c[3] if every],
    ids=[f"{c[0]}:{p}" for c in SCREENS for p, every in c[3] if every])
def test_folding_every_section_of_a_column_stacks_them_at_its_bottom(
        qtbot, tmp_path, key, factory, path):
    screen = _build(qtbot, factory, tmp_path)
    split = _resolve(screen, path)
    assert isinstance(split, cs.CollapsibleSplitter)
    _into_view(split)
    sections = [p.widget for p in split.panes()
                if isinstance(p.widget, cs.FoldSection)
                and not p.widget.isHidden()]
    assert len(sections) == len(split.panes()) >= 2
    for section in sections:
        section.set_folded(True, by_user=False)
    _pump()
    last = sections[-1]
    bottom = last.heading.mapTo(split, last.heading.rect().bottomLeft()).y()
    assert bottom >= split.height() - 10, (
        f"{key}: the folded headings float at {bottom} of {split.height()}")
    for section in sections:
        section.set_folded(False, by_user=False)
    _pump()
    assert all(s.body.isVisible() for s in sections)


@pytest.mark.qt
@pytest.mark.parametrize(
    "key,factory,path",
    [(c[0], c[1], p) for c in SCREENS for p, _every in c[3]],
    ids=[f"{c[0]}:{p}" for c in SCREENS for p, _every in c[3]])
def test_a_dragged_size_is_stored_and_a_rebuilt_screen_opens_at_it(
        qtbot, tmp_path, key, factory, path):
    screen = _build(qtbot, factory, tmp_path)
    split = _resolve(screen, path)
    assert isinstance(split, cs.CollapsibleSplitter)
    _into_view(split)
    if key == "annotate":
        screen._console_switch.setChecked(True)
        _pump()
    persist = split._persist_key
    assert persist.startswith(key.split("_")[0]), persist
    visible = [i for i in range(split.count())
               if not split.widget(i).isHidden()]
    first = visible[0]
    sizes = split.sizes()
    along = split.width() if split.orientation() == Qt.Horizontal \
        else split.height()
    target = max(120, min(along - 200, sizes[first] + 70))
    split.moveSplitter(target, visible[1])
    _pump()
    stored = cs.get_pane_extents(persist)
    name = split._pane_of(split.widget(first)).name
    assert stored.get(name, 0) > 0, f"{key}: the drag was not stored"
    screen.close()

    again = _build(qtbot, factory, tmp_path)
    reopened = _resolve(again, path)
    assert reopened.pane(name).extent == stored[name]


@pytest.mark.qt
@pytest.mark.parametrize("key,factory,names,splitters", SCREENS, ids=IDS)
def test_folding_everything_never_shows_what_the_screen_keeps_hidden(
        qtbot, tmp_path, key, factory, names, splitters):
    screen = _build(qtbot, factory, tmp_path)
    hidden = _explicitly_hidden(screen)
    for section in _sections(screen).values():
        section.set_folded(True)
        section.set_folded(False)
        section.folder.toggle()
        section.folder.toggle()
    _pump()
    shown = [w for w in hidden if not w.isHidden()]
    assert not shown, f"{key}: folding showed {shown}"


@pytest.mark.qt
def test_annotates_console_pane_stays_hidden_until_its_switch(qtbot,
                                                            tmp_path):
    screen = _build(qtbot, _annotate, tmp_path)
    split = screen._runtime_splitter
    console = screen._console_wrap
    assert isinstance(console, cs.FoldSection)
    assert split.pane("Console + AI").mode == cs.HEADER
    assert console.isHidden()

    split.set_collapsed("Console + AI", True, by_user=False)
    split.set_collapsed("Console + AI", False, by_user=False)
    screen._runtime_splitter.pane("Crops").folder.toggle()
    screen._runtime_splitter.pane("Crops").folder.toggle()
    _pump()
    assert console.isHidden(), "folding showed the switched-off console"
    assert not screen._btn_file_issue.isVisible()

    screen._console_switch.setChecked(True)
    _pump()
    assert not console.isHidden() and console.body.isVisible()
    assert screen._btn_copy_console.isVisible()
    console.set_folded(True)
    _pump()
    assert screen._btn_copy_console.isVisible(), "an action left the heading"
    bottom = console.heading.mapTo(
        split, console.heading.rect().bottomLeft()).y()
    assert bottom >= split.height() - 10
    screen._console_switch.setChecked(False)
    assert console.isHidden()


@pytest.mark.qt
def test_annotates_console_reopens_at_the_height_it_was_dragged_to(
        qtbot, tmp_path):
    screen = _build(qtbot, _annotate, tmp_path)
    split = screen._runtime_splitter
    screen._console_switch.setChecked(True)
    _pump()
    split.moveSplitter(split.height() // 2, 1)
    _pump()
    dragged = split.pane("Console + AI").extent
    assert dragged > 0
    screen._console_switch.setChecked(False)
    _pump()
    screen._console_switch.setChecked(True)
    _pump()
    assert abs(split.sizes()[1] - dragged) <= 12
