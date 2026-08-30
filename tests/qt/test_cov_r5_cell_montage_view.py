"""The montage tab's quiet halves: a refusal, a closed tab, a state ignored.

Every branch here is one a user meets when the panel is handed less than it
expects, and in every one of them the right behaviour is to stay standing:

* a coefficient table that names the feature but carries no fitted effect;
* a well grid emptied of something that is not a thumbnail;
* the coefficient already on screen selected again -- the montage must NOT
  be thrown away and reloaded;
* the Annotate tab, which follows a montage that lands, survives a strategy
  panel that cannot be built at all, and still answers after its own tab has
  been closed off the strip;
* the Compare tab answering after the same;
* a saved workspace naming none of the settings, or naming a picture mode
  this build's control never offered;
* both close handlers meeting a tab that has already gone.

Each branch is driven BOTH ways in the same test. An assertion that nothing
happened passes just as well against a widget nothing ever reached, so the
half that does nothing is always pinned beside the half that does something.
"""
from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QSpacerItem                       # noqa: E402

from spacr.crops import LOAD_IMAGES, STREAM_IMAGES              # noqa: E402
from spacr.qt.widgets import annotation_strategy_panel as asp   # noqa: E402
from spacr.qt.widgets import cell_montage_view as cmv           # noqa: E402

pytestmark = pytest.mark.qt

GENE_KEY = "gene_fraction:gene[GRA14]"
GUIDE_KEY = "fraction:grna[GRA14_1]"


# ---------------------------------------------------------------------------
# Fixtures and stand-ins
# ---------------------------------------------------------------------------

@pytest.fixture()
def make_view(qtbot):
    """Build montage views that stop their loader when the test ends."""
    made = []

    def build(**kwargs):
        widget = cmv.CellMontageView(threaded=False, **kwargs)
        qtbot.addWidget(widget)
        made.append(widget)
        return widget

    yield build
    for widget in made:
        widget.shutdown()


@pytest.fixture()
def view(make_view):
    return make_view()


def _crop():
    """One crop, as a source hands them over: a flat RGB square."""
    return np.full((8, 8, 3), 7, dtype=np.uint8)


def _plan(name="GRA14", *, objects=None, wells=()):
    """The shape :meth:`CellMontageView._fill` reads a plan through.

    A real :class:`spacr.cell_montage.MontagePlan` needs a screen on disk to
    produce; what the completion handler touches is this much of it.
    """
    rows = pd.DataFrame({"object_id": [1, 2]}) if objects is None else objects
    return SimpleNamespace(
        objects=rows,
        guides=(),
        wells=wells,
        n_objects=len(rows),
        coefficient=SimpleNamespace(name=name, level="gene",
                                    describe=lambda: f"{name} (gene)"),
        caption=lambda: f"cells behind {name}",
        summary=lambda: f"{name}: {len(rows)} cells",
    )


def _well(name):
    """A well the plan names but which contributed no object of its own."""
    return SimpleNamespace(well=name, contributed=False,
                           describe=lambda: f"{name}: 0 of 8")


def _landed(plan, crops=None):
    """What the loader hands the GUI thread when a montage comes back."""
    return cmv.MontageLoad(plans=(plan,),
                           images=(tuple(crops or ()),),
                           objects=plan.objects)


# ---------------------------------------------------------------------------
# The coefficient table
# ---------------------------------------------------------------------------

def test_a_table_that_names_the_feature_but_no_effect_reports_no_effect():
    """The effect is what the score window is centred on, so a table with a
    p-value and no coefficient has to come back as ``None`` rather than as a
    number the fit never produced -- the panel then refuses by name."""
    named = pd.DataFrame({"feature": [GENE_KEY], "p_value": [1e-3]})
    fitted = pd.DataFrame({"feature": [GENE_KEY], "coefficient": [0.25]})

    assert cmv.coefficient_from_frame(GENE_KEY, named) == (
        "GRA14", "gene", None)
    assert cmv.coefficient_from_frame(GENE_KEY, fitted) == (
        "GRA14", "gene", 0.25)


# ---------------------------------------------------------------------------
# The well grid
# ---------------------------------------------------------------------------

def test_clearing_a_well_grid_takes_out_what_is_not_a_thumbnail_too(qapp):
    """A grid holds layout items, and only some of them are widgets. One
    left behind is a cell of the previous montage still occupying a
    position in the next one."""
    tab = cmv._WellTab(("plate1", "r1", "c1"), "r1/c1")
    try:
        tab.set_content(pd.DataFrame({"object_id": [1, 2]}),
                        [_crop(), _crop()], "two cells", 2)
        assert len(tab.thumbs()) == 2, "nothing was drawn to clear"
        # The grid is reached directly because nothing in the tab's own API
        # puts a spacer in it -- the guard is for the item kinds Qt itself
        # can hand back from `takeAt`.
        tab._grid.addItem(QSpacerItem(8, 8))
        assert tab._grid.count() == 3

        tab.clear()

        assert tab._grid.count() == 0
        assert tab.thumbs() == ()
    finally:
        tab.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# Selecting the coefficient that is already on screen
# ---------------------------------------------------------------------------

def test_reselecting_the_coefficient_on_screen_keeps_its_montage(view):
    """Clicking the same point twice must not throw the cells away: the grid
    is only dropped because it would otherwise describe a gene the selection
    has left, and this selection has not moved."""
    plan = _plan()
    view.set_coefficient(GENE_KEY)
    view._on_loaded(_landed(plan, [_crop(), _crop()]))
    assert view.plans() == (plan,)

    view.set_coefficient(GENE_KEY)

    assert view.plans() == (plan,), "the montage was dropped for its own key"
    assert "GRA14" in view.caption_text()

    view.set_coefficient(GUIDE_KEY)

    assert view.plans() == (), "a different coefficient must drop the grid"
    assert view.caption_text() == ""


# ---------------------------------------------------------------------------
# What the picker chose
# ---------------------------------------------------------------------------

def test_a_plan_that_marked_no_candidate_contributes_no_group(view):
    """The window admitted nothing for this guide. A group of zero cells in
    the comparison would read as a gene whose cells were all alike."""
    view._plans = (
        _plan("GRA14", objects=pd.DataFrame(
            {"montage_candidate": [False, False]})),
        _plan("ROP18", objects=pd.DataFrame(
            {"montage_candidate": [True, False]})),
    )

    assert view.picked_groups() == {"ROP18": [0]}


# ---------------------------------------------------------------------------
# Writing the run's scores into the databases
# ---------------------------------------------------------------------------

def _database(tmp_path):
    import sqlite3

    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE png_list (file_name, plateID, rowID, "
                   "columnID, fieldID, object_label, png_path)")
        for i in range(4):
            db.execute("INSERT INTO png_list VALUES (?,?,?,?,?,?,?)",
                       (f"img{i}.png", "plate1", "r1", "c1", "1", i,
                        str(tmp_path / f"img{i}.png")))
    return str(path)


def _score_file(tmp_path):
    path = tmp_path / "plate1_dv.csv"
    pd.DataFrame({"path": [f"img{i}.png" for i in range(4)],
                  "pred": [i / 4 for i in range(4)],
                  "cv_predictions": [0, 1, 0, 1],
                  "prc": ["plate1_r1_c1"] * 4,
                  "object": list(range(4))}).to_csv(path, index=False)
    return str(path)


def test_when_no_database_takes_the_scores_none_is_reported_as_merged(
        make_view, tmp_path):
    """The count sentence claims rows were written. Printing it after every
    database refused would tell a user their databases now carry scores that
    are not in them."""
    broken = str(tmp_path / "not-a-database.db")
    open(broken, "w").write("this is not sqlite")
    scores = _score_file(tmp_path)
    table = [{"plate": "plate1", "database": broken, "score": scores}]
    view = make_view(database_provider=lambda: table)

    assert view.write_scores_into_the_databases(confirm=lambda *_a: True) == {}
    refusal = view.status_text()
    assert "not-a-database.db" in refusal
    assert "matched" not in refusal

    table[:] = [{"plate": "plate1", "database": _database(tmp_path),
                 "score": scores}]

    written = view.write_scores_into_the_databases(confirm=lambda *_a: True)

    assert written, "nothing merged, so the empty answer proves nothing"
    assert "rows matched" in view.status_text()


# ---------------------------------------------------------------------------
# A saved run's montage settings
# ---------------------------------------------------------------------------

def test_a_saved_state_naming_none_of_the_settings_applies_nothing(view):
    """A run saved by another build carries keys this one does not read. The
    panel has to open on it rather than refuse the whole run."""
    assert view.apply_workspace_state(
        {"montage_shown": True, "results_path": "/gone/results.csv"}) is False
    assert view._channels.text() == ""

    assert view.apply_workspace_state(
        {"widgets": {"channels": "r,g,b"}}) is True
    assert view._channels.text() == "r,g,b"


def test_a_picture_mode_this_build_never_offered_is_not_applied(view):
    """Setting an index of -1 empties the combo box, so a mode from a build
    with different sources would leave the montage with no source at all."""
    assert view.picture_mode() == LOAD_IMAGES

    assert view.apply_workspace_state({"picture_mode": "sideways"}) is False
    assert view.picture_mode() == LOAD_IMAGES

    assert view.apply_workspace_state(
        {"picture_mode": STREAM_IMAGES}) is True
    assert view.picture_mode() == STREAM_IMAGES


def test_a_crop_source_the_menu_never_offered_leaves_the_control_alone(view):
    """The settings window writes back into the hidden widgets. A value it
    cannot find must leave the widget where it was and let the rest land."""
    view._write_back({"crop_source": "sideways", "cap": 12})

    assert view.picture_mode() == LOAD_IMAGES
    assert view._cap.value() == 12

    view._write_back({"crop_source": STREAM_IMAGES})

    assert view.picture_mode() == STREAM_IMAGES


# ---------------------------------------------------------------------------
# The Annotate tab
# ---------------------------------------------------------------------------

def _boom(*_args, **_kwargs):
    raise RuntimeError("the strategy panel could not be built")


def test_a_montage_that_lands_points_the_annotate_tab_at_its_wells(view):
    """The tab is built before the montage and must follow it. A strategy
    fitted on the previous coefficient's wells would take its positives from
    wells the grid on screen has nothing to do with."""
    panel = view.annotate_the_cells()
    assert panel is not None
    assert panel._wells.text().strip() == "", "there is no montage yet"

    view.set_coefficient(GENE_KEY)
    view._on_loaded(_landed(_plan(wells=(_well("r1_c1"), _well("r1_c2"))),
                            [_crop(), _crop()]))

    assert panel._wells.text() == "r1_c1, r1_c2"


def test_a_strategy_panel_that_cannot_be_built_leaves_the_tab_standing(
        view, monkeypatch, caplog):
    """Forty controls and a fitting runner is a lot to build. Failing must
    cost the Annotate tab its content and nothing else -- the montage, its
    caption and the tab itself stay on screen."""
    monkeypatch.setattr(asp, "AnnotationStrategyPanel", _boom)

    with caplog.at_level(logging.ERROR, logger=cmv.LOG.name):
        assert view.annotate_the_cells() is None

    assert "Could not build the annotation strategies" in caplog.text
    assert view._annotation_panel is None
    assert "Annotate" in view.tab_labels()
    assert view._annotation_placeholder is not None

    monkeypatch.undo()

    assert view.annotate_the_cells() is not None


def test_opening_the_annotate_tab_when_it_cannot_be_built_keeps_the_notice(
        view, qtbot, monkeypatch):
    """Opening the tab posts the build rather than doing it inside Qt's own
    tab change. When that build fails the page must keep the sentence saying
    what the tab is for, not go blank."""
    monkeypatch.setattr(asp, "AnnotationStrategyPanel", _boom)

    view._tabs.setCurrentWidget(view._annotation_page)
    qtbot.wait(30)

    assert view._annotation_panel is None
    assert view._annotation_placeholder.isVisibleTo(view._annotation_page)

    monkeypatch.undo()
    view._tabs.setCurrentIndex(0)
    view._tabs.setCurrentWidget(view._annotation_page)
    qtbot.wait(30)

    assert view._annotation_panel is not None
    assert view._annotation_placeholder is None


def test_a_build_that_failed_after_the_notice_went_still_installs_the_panel(
        view, monkeypatch):
    """The placeholder is hidden and forgotten before the panel is put in the
    page. A build that got that far and then failed leaves the tab with
    neither, and the next one must add the panel rather than skip the page
    it thinks it has already filled."""
    monkeypatch.setattr(asp, "AnnotationStrategyPanel",
                        lambda *a, **k: object())

    with pytest.raises(TypeError):
        view.annotate_the_cells()
    assert view._annotation_placeholder is None
    assert view._annotation_panel is None

    monkeypatch.undo()
    panel = view.annotate_the_cells()

    assert panel is not None
    assert view._annotation_page.layout().indexOf(panel) >= 0


def test_the_annotate_tab_still_answers_after_its_tab_has_been_closed(view):
    """Closing the page does not destroy the panel, and the button that opens
    it must not hunt for a tab index that is no longer there."""
    panel = view.annotate_the_cells()
    assert view._tabs.currentWidget() is view._annotation_page

    view._tabs.tabCloseRequested.emit(
        view._tabs.indexOf(view._annotation_page))
    assert view._tabs.indexOf(view._annotation_page) < 0
    view._tabs.setCurrentIndex(0)

    assert view.annotate_the_cells() is panel
    assert view._tabs.indexOf(view._annotation_page) < 0
    assert view._tabs.currentIndex() == 0


def test_a_strategy_with_no_result_says_nothing_in_the_status_line(view):
    """The run signal arrives whether the fit produced a selection or not.
    Announcing a count for a strategy that selected nothing would put a
    number on screen no result stands behind."""
    panel = view.annotate_the_cells()
    view.set_coefficient(GENE_KEY)
    before = view.status_text()

    panel.finished.emit("top_score_random")

    assert view.status_text() == before

    panel._result = SimpleNamespace(
        title="Top score, random control",
        role_counts=lambda: {"positive": 3, "negative": 4, "holdout": 2})
    panel.finished.emit("top_score_random")

    assert "Top score, random control chose 7 cell(s)" in view.status_text()


def test_a_saved_annotation_goes_beside_the_results_file_not_inside_it(
        make_view, tmp_path):
    """The folder the strategy panel offers to save into. A results CSV is a
    file, and writing a selection into a path that names one would fail at
    the end of a fit rather than before it."""
    results = tmp_path / "results.csv"
    results.write_text("feature,coefficient\n", encoding="utf-8")
    where = {"path": ""}
    view = make_view(results_provider=lambda: where["path"])

    # `_annotation_folder` is the `folder_provider` the strategy panel is
    # built with; the panel keeps it private, so it is asked here directly.
    assert view._annotation_folder() == ""

    where["path"] = str(results)
    assert view._annotation_folder() == str(tmp_path)

    where["path"] = str(tmp_path)
    assert view._annotation_folder() == str(tmp_path)


# ---------------------------------------------------------------------------
# The Compare tab
# ---------------------------------------------------------------------------

def _measured(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "plateID": ["plate1"] * n,
        "rowID": [f"r{1 + i % 4}" for i in range(n)],
        "columnID": [f"c{1 + i % 5}" for i in range(n)],
        "cell_area": rng.uniform(100.0, 900.0, n),
        "pathogen_area": rng.uniform(10.0, 90.0, n),
        "montage_candidate": [i < n // 2 for i in range(n)],
    })


def test_the_compare_tab_still_answers_after_its_tab_has_been_closed(view):
    """The panel outlives its page. Reopening it must return the panel that
    holds the comparison rather than raising an index that has moved."""
    view._plans = (_plan(objects=_measured()),)

    panel = view.compare_a_measurement()

    assert panel is not None
    assert view._tabs.currentWidget() is panel

    view._tabs.tabCloseRequested.emit(view._tabs.indexOf(panel))
    assert view._tabs.indexOf(panel) < 0

    assert view.compare_a_measurement() is panel
    assert view._tabs.indexOf(panel) < 0, "the closed tab came back"


# ---------------------------------------------------------------------------
# Closing a well tab
# ---------------------------------------------------------------------------

def test_a_close_mark_whose_tab_has_gone_closes_nothing_else(view):
    """The mark captures its widget rather than its index precisely so that
    a second click cannot close whatever tab took that index -- and the
    second click has to close nothing at all."""
    tab = view._open_well_tab(("plate1", "r1", "c1"), "r1/c1 — GRA14",
                              "one well")
    assert "r1/c1 — GRA14" in view.tab_labels()

    view._close_widget(tab)

    assert "r1/c1 — GRA14" not in view.tab_labels()
    remaining = view.tab_labels()

    view._close_widget(tab)

    assert view.tab_labels() == remaining
