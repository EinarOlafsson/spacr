"""The last unexercised corners of six screens, each pinned by behaviour.

Every case here is a branch a healthy screen almost never takes, and each
one is paired -- in the same test -- with the input that makes the same
code do the opposite thing, so "nothing happened" cannot pass for "the
guard worked".

* **Project Browser** -- a double-click that lands on no row opens
  nothing; a stale result whose cause codes explain nothing gets no
  explanation line under it; and a project the registry *does* know about
  is never told that nothing can be checked.
* **Model Explanation** -- an unavailable surrogate family still carries
  its reason as a tooltip even when the dropdown's model hands back no
  item to grey out; and ``configure_hit`` only touches the phenotype
  dropdown when it is given a phenotype, and never adds one twice.
* **Data Manager** -- a job that settles as a failure does not run its
  completion handler but still announces itself; the note strip still
  says what it has to say when Qt hands back no style to repolish it
  with; and a project with nothing unregistered says so by omission.
* **Plate View** -- a well with no value is blank rather than zero; a
  selection off the end of the grid draws no highlight; and a threaded
  job that settles as a failure leaves the screen as it was.
* **Regression** -- a panel whose ``results_frame`` is not callable falls
  back to the run folder; and the Hits tab goes on the end when there is
  no "Guide support" tab to sit beside.
* **Mask Generation** -- the invariant behind ``install_folds``'s
  unreachable ``build_strip() is None`` arm (see the PROOF below).

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Qt, Signal                 # noqa: E402
from PySide6.QtGui import QPixmap, QStandardItemModel          # noqa: E402
from PySide6.QtWidgets import (QComboBox, QLabel, QTabWidget,  # noqa: E402
                               QVBoxLayout, QWidget)

from spacr import data_manager as dm                           # noqa: E402
from spacr.projects import ProjectSummary, StaleArtifact       # noqa: E402
from spacr.qt.screens import data_manager as data_manager_mod  # noqa: E402
from spacr.qt.screens import mask as mask_mod                  # noqa: E402
from spacr.qt.screens import model_explanation as explain_mod  # noqa: E402
from spacr.qt.screens import plate_view as plate_view_mod      # noqa: E402
from spacr.qt.screens import project_browser as browser_mod    # noqa: E402
from spacr.qt.screens import regression as regression_mod      # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# A worker thread that settles when the test says so
# ---------------------------------------------------------------------------

class _StubWorker(QObject):
    """Stands in for ``PipelineWorker``: the two signals a screen connects."""

    error = Signal(str)
    finished = Signal(bool)


class _HandThread(QObject):
    """A ``QThread`` stand-in that runs the job body on the calling thread.

    ``start`` does what the worker thread does -- runs the callable and
    fills the box -- but it does NOT announce the outcome. The test emits
    ``worker.finished`` itself, so both settlements, the success and the
    failure, can be driven through the screen's real signal path.
    """

    finished = Signal()

    def __init__(self, fn, settings, worker):
        super().__init__()
        self._fn = fn
        self._settings = settings
        self.worker = worker
        self.quit_calls = 0
        self.waited = []

    def start(self):
        self._fn(self._settings)

    def quit(self):
        self.quit_calls += 1

    def wait(self, msecs=0):
        self.waited.append(msecs)
        return True


def _hand_threads(monkeypatch, target, attribute="make_thread"):
    """Replace ``target.<attribute>`` with a hand-settled thread factory."""
    made = []

    def _make_thread(fn, settings, app_key="", **_kwargs):
        worker = _StubWorker()
        thread = _HandThread(fn, settings, worker)
        made.append(thread)
        return thread, worker

    monkeypatch.setattr(target, attribute, _make_thread)
    return made


# ===========================================================================
# Project Browser
# ===========================================================================

def _browser(qtbot, tmp_path):
    """A browser searching ``tmp_path``, scanning inline on this thread."""
    browser = browser_mod.ProjectBrowserScreen(threaded=False,
                                               roots=(str(tmp_path),))
    qtbot.addWidget(browser)
    return browser


def test_a_double_click_on_no_row_opens_nothing(qtbot, tmp_path, monkeypatch):
    """Opening a project needs a project; an empty selection is not one.

    ``itemDoubleClicked`` is the only way into the handler -- the screen
    offers no "open the selected row" call -- so the table's own signal is
    what is emitted here, exactly as Qt emits it.
    """
    root = os.path.abspath(str(tmp_path / "plate1"))
    monkeypatch.setattr(
        browser_mod, "_browse",
        lambda roots, depth: (ProjectSummary(root=root, name="plate1",
                                             known=True),))
    browser = _browser(qtbot, tmp_path)
    browser.rescan()
    chosen = []
    browser.project_chosen.connect(chosen.append)
    item = browser._table.item(0, 0)
    assert item is not None, "the scan produced no row to double-click"

    browser._table.clearSelection()
    browser._table.itemDoubleClicked.emit(item)
    assert chosen == [], "a double-click with nothing selected opened a project"

    browser._table.selectRow(0)
    browser._table.itemDoubleClicked.emit(item)
    assert chosen == [root]


def test_a_stale_result_with_nothing_to_explain_gets_no_explanation_line(
        qtbot, tmp_path, monkeypatch):
    """The indented "why" line is only drawn when there is a why.

    ``StaleArtifact.explain`` renders the machine cause codes, and a
    registry row can carry a human reason with no cause code behind it. A
    blank line indented under the entry would read as a missing sentence.
    """
    root = os.path.abspath(str(tmp_path / "plate1"))
    uncaused = StaleArtifact(artifact_id="a1", kind="masks", module="mask",
                             role="masks", path=os.path.join(root, "masks"),
                             reasons=("the operator said so",), causes=())
    caused = StaleArtifact(artifact_id="a2", kind="measurements",
                           module="measure", role="db",
                           path=os.path.join(root, "measurements.db"),
                           reasons=(), causes=("upstream-newer",))
    monkeypatch.setattr(
        browser_mod, "_browse",
        lambda roots, depth: (ProjectSummary(root=root, name="plate1",
                                             known=True,
                                             stale=(uncaused, caused)),))
    browser = _browser(qtbot, tmp_path)
    browser.rescan()

    lines = browser.show_detail(root).splitlines()
    uncaused_at = lines.index(f"  {uncaused.describe()}")
    caused_at = lines.index(f"  {caused.describe()}")

    # The one with a cause is explained on the line below it...
    assert lines[caused_at + 1] == f"    {caused.explain()}"
    assert caused.explain(), "the cause code rendered no sentence"
    # ...and the one without goes straight on to the next entry.
    assert uncaused.explain() == ""
    assert lines[uncaused_at + 1] == f"  {caused.describe()}"


def test_a_registered_project_is_never_told_nothing_can_be_checked(
        qtbot, tmp_path, monkeypatch):
    """"No run record" is a claim about the registry, not about staleness.

    A project the registry knows and finds nothing wrong with is current,
    and saying its staleness is unknowable would be the browser lying by
    omission. The unexamined project still has to be told.
    """
    known_root = os.path.abspath(str(tmp_path / "recorded"))
    fresh_root = os.path.abspath(str(tmp_path / "copied_in"))
    monkeypatch.setattr(
        browser_mod, "_browse",
        lambda roots, depth: (
            ProjectSummary(root=known_root, name="recorded", known=True,
                           next_steps=(("measure", ""),)),
            ProjectSummary(root=fresh_root, name="copied_in", known=False),
        ))
    browser = _browser(qtbot, tmp_path)
    browser.rescan()

    recorded = browser.show_detail(known_root)
    assert "Nothing here has a run record" not in recorded
    assert "What could run next" in recorded and "measure" in recorded

    copied_in = browser.show_detail(fresh_root)
    assert "Nothing here has a run record" in copied_in


# ===========================================================================
# Model Explanation
# ===========================================================================

class _ItemlessCombo(QComboBox):
    """A dropdown whose ``model()`` holds no item for any row.

    ``QComboBox.model()`` is not a C++ virtual, so Qt's own machinery goes
    on using the real model; only the panel's Python-side ``item(index)``
    lookup sees this one, and an empty ``QStandardItemModel`` answers it
    with ``None`` -- the case the panel guards against before greying a
    backend out.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._itemless = QStandardItemModel(self)

    def model(self):
        return self._itemless


def _unavailable_backends():
    """Every surrogate family present, with xgboost unusable."""
    return {key: {"available": key != "xgboost", "version": "1.0",
                  "reason": "" if key != "xgboost"
                            else "install xgboost to enable this backend"}
            for key in explain_mod.MODEL_FAMILIES}


def _backend_index(panel, key):
    return next(i for i in range(panel.backend.count())
                if panel.backend.itemData(i) == key)


def test_an_unavailable_backend_keeps_its_reason_even_with_no_item_to_grey(
        qtbot, monkeypatch):
    """The reason is what the user reads; the grey is only how it looks.

    Disabling the row needs the dropdown's model to hand back an item for
    it, and the tooltip that says *why* the family cannot run must not
    depend on that lookup succeeding.
    """
    monkeypatch.setattr(explain_mod, "available_backends",
                        _unavailable_backends)
    reason = "install xgboost to enable this backend"

    ordinary = explain_mod.ExplainCvPanel()
    qtbot.addWidget(ordinary)
    index = _backend_index(ordinary, "xgboost")
    assert ordinary.backend.model().item(index) is not None
    assert ordinary.backend.model().item(index).isEnabled() is False
    assert ordinary.backend.itemData(index, Qt.ToolTipRole) == reason

    monkeypatch.setattr(explain_mod, "QComboBox", _ItemlessCombo)
    itemless = explain_mod.ExplainCvPanel()
    qtbot.addWidget(itemless)
    index = _backend_index(itemless, "xgboost")

    assert itemless.backend.model().item(index) is None, (
        "the stand-in still produced an item to grey out")
    assert itemless.backend.itemData(index, Qt.ToolTipRole) == reason
    # And the families that CAN run still carry no excuse.
    assert not itemless.backend.itemData(
        _backend_index(itemless, "random_forest"), Qt.ToolTipRole)


def test_a_hit_only_moves_the_phenotype_dropdown_when_it_names_one(qtbot):
    """A hit with no phenotype leaves the score column the user chose.

    ``configure_hit`` seeds the screen from a row on the hit list, and a
    row that names no phenotype must not silently re-point the column the
    explanation will be computed over.
    """
    panel = explain_mod.InvestigateHitPanel()
    qtbot.addWidget(panel)
    panel.score.setCurrentText("class")
    before = [panel.score.itemText(i) for i in range(panel.score.count())]

    panel.configure_hit(gene="g1", phenotype="")
    assert panel.score.currentText() == "class"
    assert [panel.score.itemText(i)
            for i in range(panel.score.count())] == before

    # A phenotype the dropdown already lists is selected, not added twice.
    panel.configure_hit(gene="g1", phenotype="pred")
    assert panel.score.currentText() == "pred"
    assert [panel.score.itemText(i)
            for i in range(panel.score.count())] == before

    # One it does not know is added, then selected.
    panel.configure_hit(gene="g1", phenotype="parasite_count")
    assert panel.score.currentText() == "parasite_count"
    assert [panel.score.itemText(i)
            for i in range(panel.score.count())] == before + [
                "parasite_count"]


# ===========================================================================
# Data Manager
# ===========================================================================

class _RecordingStyle:
    """A style that writes down the polish calls it is asked for."""

    def __init__(self):
        self.calls = []

    def unpolish(self, widget):
        self.calls.append("unpolish")

    def polish(self, widget):
        self.calls.append("polish")


class _StyledLabel(QLabel):
    """A label whose style records what it was asked to repolish."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = _RecordingStyle()

    def style(self):
        return self.recorder


class _StylelessLabel(QLabel):
    """A label caught after the style that would polish it has gone."""

    def style(self):
        return None


def _usage(root, **fields):
    """A ``ProjectUsage`` with only the fields a test cares about."""
    return dm.ProjectUsage(root=str(root), total_bytes=2048, total_files=4,
                           **fields)


def test_a_job_that_settles_as_a_failure_runs_no_handler_but_still_reports(
        qtbot, tmp_path, monkeypatch):
    """A worker that failed has no result, so the handler must not run.

    ``job_finished`` still has to carry False: a caller waiting on it
    would otherwise wait for a job that is already over.
    """
    root = str(tmp_path)
    monkeypatch.setattr(dm, "scan_project",
                        lambda path, **kw: _usage(path, symlinks=("link",)))
    from spacr.qt import bridge
    threads = _hand_threads(monkeypatch, bridge)

    screen = data_manager_mod.DataManagerScreen(threaded=True)
    qtbot.addWidget(screen)
    settled = []
    screen.job_finished.connect(settled.append)

    screen.set_project(root)
    assert threads, "the threaded path was not taken"
    threads[-1].worker.finished.emit(False)

    assert settled == [False]
    assert screen.usage is None, "a failed job's handler drew a result anyway"
    assert screen.rescan_button.isEnabled(), "the screen was left marked busy"

    # The same job, settled as a success, does reach the handler.
    screen.set_project(root)
    threads[-1].worker.finished.emit(True)

    assert settled == [False, True]
    assert screen.usage is not None
    assert screen.usage.total_bytes == 2048


def test_the_note_strip_says_its_piece_with_or_without_a_style(
        qtbot, tmp_path, monkeypatch):
    """The warning is the text and the property; the repolish is cosmetic.

    A dynamic property only changes the colours after an unpolish/polish
    pair, and a widget whose style has gone cannot have one -- but the
    sentence the user has to read must land either way.
    """
    monkeypatch.setattr(data_manager_mod, "QLabel", _StyledLabel)
    styled = data_manager_mod.DataManagerScreen(threaded=False)
    qtbot.addWidget(styled)

    assert styled.set_project(str(tmp_path / "not_a_folder")) is None
    assert styled.note_label.text() == "Choose a project folder first."
    assert styled.note_label.property("warn") == "true"
    assert styled.note_label.recorder.calls[-2:] == ["unpolish", "polish"]

    monkeypatch.setattr(data_manager_mod, "QLabel", _StylelessLabel)
    styleless = data_manager_mod.DataManagerScreen(threaded=False)
    qtbot.addWidget(styleless)

    styleless.set_project(str(tmp_path / "not_a_folder"))
    assert styleless.note_label.style() is None
    assert styleless.note_label.text() == "Choose a project folder first."
    assert styleless.note_label.property("warn") == "true"


def test_a_project_with_nothing_unregistered_does_not_mention_it(
        qtbot, tmp_path, monkeypatch):
    """The note lists the things worth acting on and nothing else.

    "0 bytes have no registry record" is noise on a project where every
    byte is accounted for, and it would drown the notes that matter.
    """
    root = str(tmp_path)
    monkeypatch.setattr(
        dm, "scan_project",
        lambda path, **kw: _usage(path, unregistered_bytes=0,
                                  unregistered_files=0,
                                  symlinks=("data -> /elsewhere",)))
    screen = data_manager_mod.DataManagerScreen(threaded=False)
    qtbot.addWidget(screen)

    screen.set_project(root)
    assert "no registry record" not in screen.note_label.text()
    assert "1 symlink(s), not followed" in screen.note_label.text()

    monkeypatch.setattr(
        dm, "scan_project",
        lambda path, **kw: _usage(path, unregistered_bytes=4096,
                                  unregistered_files=3,
                                  symlinks=("data -> /elsewhere",)))
    screen.set_project(root)
    assert "no registry" in screen.note_label.text()
    assert "3 files" in screen.note_label.text()
    assert "1 symlink(s), not followed" in screen.note_label.text()


# ===========================================================================
# Plate View
# ===========================================================================

def _layout_frame():
    """Two wells on a 1x2 plate; the first one has no value at all."""
    frame = pd.DataFrame({
        "row_index": [1, 1],
        "column_index": [1, 2],
        "n": [5, 7],
        "value": [float("nan"), 2.5],
    })
    frame.attrs["n_rows"] = 1
    frame.attrs["n_cols"] = 2
    return frame


def _image(grid):
    pixmap = QPixmap(grid.size())
    grid.render(pixmap)
    return pixmap.toImage()


def test_a_well_with_no_value_keeps_its_count_and_stays_blank(qtbot):
    """A well that survived filtering with no number is blank, not zero.

    Drawing it as zero would put it at the bottom of the colour scale --
    a well with objects in it reading as the strongest possible negative.
    """
    grid = plate_view_mod.PlateGridWidget()
    qtbot.addWidget(grid)

    grid.set_plate(_layout_frame(), vmin=0.0, vmax=5.0)

    assert grid.well_value(1, 1) is None
    assert grid.well_count(1, 1) == 5, "the count was dropped with the value"
    assert grid.well_value(1, 2) == 2.5
    assert grid.grid_size() == (1, 2)


def test_a_selection_off_the_end_of_the_grid_draws_no_highlight(qtbot):
    """The highlight is bounded by the plate, not by the selection.

    ``select`` takes any pair of numbers -- a linked view can hand it a
    well from a 384 plate while a 96 is on screen -- and a rectangle drawn
    for a well that is not there would sit on top of one that is.
    """
    grid = plate_view_mod.PlateGridWidget()
    qtbot.addWidget(grid)
    grid.resize(400, 300)
    grid.set_plate(_layout_frame(), vmin=0.0, vmax=5.0)
    unselected = _image(grid)

    grid.select(9, 9)
    assert grid.selected_well() == (9, 9), "the selection was silently dropped"
    assert _image(grid) == unselected, "an off-grid well was outlined anyway"

    grid.select(1, 2)
    assert _image(grid) != unselected, "an on-grid well was not outlined"


def _measurements_db(tmp_path):
    """A one-table measurements database with a single numeric column."""
    path = str(tmp_path / "measurements.db")
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE cell (plate TEXT, row_name TEXT, "
                    "column_name TEXT, value REAL)")
        con.execute("INSERT INTO cell VALUES ('p1', 'A', 'c1', 1.5)")
        con.commit()
    finally:
        con.close()
    return path


def test_a_threaded_job_that_failed_leaves_the_columns_alone(
        qtbot, tmp_path, monkeypatch):
    """A failed read has no columns, so the dropdown must not be rebuilt.

    Clearing it on the way in and never refilling it would leave the user
    looking at an empty measurement list with no idea a read had failed.
    """
    db = _measurements_db(tmp_path)
    threads = _hand_threads(monkeypatch, plate_view_mod)

    screen = plate_view_mod.PlateViewScreen(threaded=True)
    qtbot.addWidget(screen)
    settled = []
    screen.job_finished.connect(settled.append)

    assert screen.open_database(db) is True
    assert threads, "opening the database started no job"
    threads[-1].worker.finished.emit(False)

    assert settled == [False]
    assert screen.current_value_column() == ""
    assert screen.is_busy() is False

    screen.set_table("cell")
    threads[-1].worker.finished.emit(True)

    assert settled == [False, True]
    assert screen.current_value_column() == "value"


# ===========================================================================
# Regression
# ===========================================================================

def _coefficients(feature_a, feature_b):
    """A two-family guide permutation table over two named guides."""
    return pd.DataFrame({
        "feature": [feature_a, feature_b, feature_a, feature_b],
        "grna": [feature_a, feature_b, feature_a, feature_b],
        "coefficient": [1.5, -1.2, 1.5, -1.2],
        "p_value": [0.001, 0.04, 0.001, 0.04],
        "adjusted_p_value": [0.002, 0.04, 0.004, 0.08],
        "minimum_wells_threshold": [1, 1, 2, 2],
    })


def test_a_panel_whose_frame_is_not_a_reader_falls_back_to_the_folder(
        qtbot, qt_theme_applied, tmp_path):
    """``results_frame`` is a method on the panel -- but only on a panel.

    The builder is handed whatever the caller has, including a plain
    object that carries the attribute without being able to answer it,
    and the run folder is still there to publish.
    """
    folder = tmp_path / "ols_1"
    folder.mkdir()
    _coefficients("on_disk_1", "on_disk_2").to_csv(folder / "results.csv",
                                                   index=False)

    class _NotAReader:
        results_frame = None

        def run_folder(self):
            return str(folder)

    window = regression_mod.build_publication_figure(_NotAReader())
    qtbot.addWidget(window)
    assert sorted(window.explorer.results()["feature"]) == ["on_disk_1",
                                                            "on_disk_2"]

    class _Reader(_NotAReader):
        def results_frame(self):
            return _coefficients("on_screen_1", "on_screen_2")

    window = regression_mod.build_publication_figure(_Reader())
    qtbot.addWidget(window)
    assert sorted(window.explorer.results()["feature"]) == ["on_screen_1",
                                                            "on_screen_2"]


class _TabbedPanel(QWidget):
    """A results panel that is nothing but its tab bar."""

    def __init__(self, titles):
        super().__init__()
        self.tabs = QTabWidget(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        for title in titles:
            self.tabs.addTab(QWidget(), title)


def _titles(panel):
    return [panel.tabs.tabText(i) for i in range(panel.tabs.count())]


def test_the_hits_tab_goes_last_when_there_is_no_guide_support_tab(
        qtbot, qt_theme_applied, tmp_path):
    """The Hits tab sits beside Guide support, or at the end.

    A panel built without the guide table -- an OLS run has none -- must
    still get the tab, and putting it at a position guessed from a title
    that is not there would drop it in front of Coefficients.
    """
    without = _TabbedPanel(["Coefficients", "Diagnostics"])
    qtbot.addWidget(without)

    hits = regression_mod.install_hits_tab(without)

    assert hits is not None and without.hits is hits
    assert _titles(without) == ["Coefficients", "Diagnostics",
                                regression_mod.HITS_TAB_TITLE]

    with_support = _TabbedPanel(["Coefficients",
                                 regression_mod.HITS_TAB_AFTER,
                                 "Diagnostics"])
    qtbot.addWidget(with_support)

    regression_mod.install_hits_tab(with_support)

    assert _titles(with_support) == ["Coefficients",
                                     regression_mod.HITS_TAB_AFTER,
                                     regression_mod.HITS_TAB_TITLE,
                                     "Diagnostics"]


# ===========================================================================
# Mask Generation
# ===========================================================================
#
# PROOF that ``install_folds``'s ``if strip is None: return None`` (mask.py
# lines 359-360) cannot run, so it is not driven by a test here.
#
# ``install_folds`` builds the concrete
# ``spacr.qt.screens.map_barcodes.CategoryFoldSet`` itself (mask.py:351) --
# there is no injection seam and no subclass anywhere in spacr -- and it
# calls, in this order:
#
#     if not folds.mount():        # mask.py:356-357 -> returns None
#         return None
#     strip = folds.build_strip(header)
#     if strip is None:            # mask.py:359-360 -- DEAD
#         return None
#
# ``CategoryFoldSet.mount`` (map_barcodes.py:946-961) ends with
# ``self.order = tuple(mounted); return self.order``, and ``build_strip``
# (map_barcodes.py:963-974) returns None on exactly one condition,
# ``if not self.order``, and otherwise always returns a ``FoldStrip``.
# Nothing between the two calls touches ``self.order``. So
# ``build_strip()`` returns None only when ``mount()`` returned an empty
# tuple -- and that case has already returned at line 357.
#
# The test below pins the invariant the proof rests on: an order that
# mounted something always yields a strip, and only an empty one yields
# None.

def test_a_fold_set_that_mounted_something_always_builds_a_strip(
        qtbot, qt_theme_applied):
    """``build_strip`` answers None on one condition ``mount`` rules out.

    This is the guarantee that makes ``install_folds``'s second None check
    unreachable, so it is asserted here rather than faked there.
    """
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.map_barcodes import CategoryFoldSet

    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)
    header = screen._header

    folds = CategoryFoldSet(
        screen,
        {key: mask_mod.FOLD_GATES[key] for key in mask_mod.FOLDED_APPS},
        implies=mask_mod.FOLD_IMPLIES,
    )
    assert folds.mount(), "Mask Generation mounted no folded settings"
    strip = folds.build_strip(header)
    assert strip is not None
    assert list(strip.keys()) == list(mask_mod.FOLDED_APPS)

    # The only way to a None strip, and install_folds has already returned
    # by then.
    empty = CategoryFoldSet(screen, {})
    assert empty.mount() == ()
    assert empty.build_strip(header) is None
