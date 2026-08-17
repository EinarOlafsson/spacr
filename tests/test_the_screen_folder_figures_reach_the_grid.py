"""The second half of "there are graphs that are saved but never shown".

Reported 2026-08-17 (instruction 128 P, item 1): "for some reason now i dont
see the grna threshold graph".

The first half was fixed the same day: `_load_trial_figures` listed the run
folder FLAT and almost everything a run writes is in a subfolder, so it now
walks the run folder recursively (`test_every_saved_figure_reaches_the_grid`).

Three figures are still missed by that walk, and not because it is not deep
enough -- because they are ABOVE it. MEASURED, because the brief for this
work had them all in one folder and they are not:

    <src>/plate_heatmap_unique_counts.pdf    plot_plates(dst=<src>)
    <src>/results/fraction_threshold.pdf     spacr.sequencing
    <src>/results/cell_min_threshold.pdf     spacr.ml
    <src>/results/ols_12/...                 <- the run folder

`plot_plates` documents itself as saving to ``<dst>/plate_heatmap_<variable>
.pdf`` and `graph_sequencing_stats` hands it the COUNT-DATA folder, so the
heatmap lands one level higher than the other two. `spacr.ml
._screen_figure_folders` lists exactly this pair (``base`` and
``base/results``) from the writing end.

All three are per-screen preprocessing: drawn once from the count data,
before any model is fitted, and shared by every run under that folder. A walk
that starts at the run folder and goes DOWN cannot reach any of them.

So they get their own section, labelled as the screen's. Not mixed into the
run's, because captioning `fraction_threshold` as ols_12's output is a false
claim about what ols_12 produced -- it is the same picture for ols_11 and for
every trial the sweep ran. And those folders are listed FLAT, because
``<src>/results/``'s subfolders are the SIBLING RUNS: recursing it would pull
every other run's figures onto this run's grid, which is wrong rather than
merely incomplete.
"""
from __future__ import annotations

import os
import pathlib

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

pytestmark = pytest.mark.qt

#: The three the maintainer is missing, by the names the pipeline writes and
#: at the depth it writes them: two in ``<src>/results/`` and the heatmap in
#: ``<src>/``. Getting that split wrong is what made the third one look like
#: it was already covered.
SCREEN_FIGURES = {
    "results/fraction_threshold.pdf": "fraction_threshold",
    "results/cell_min_threshold.pdf": "cell_min_threshold",
    "plate_heatmap_unique_counts.pdf": "plate_heatmap_unique_counts",
}


def _figure(path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [1, 0])
    figure.savefig(path)
    plt.close(figure)


@pytest.fixture()
def screen_folder(tmp_path):
    """A screen shaped the way `perform_regression` leaves one.

    ``<src>`` with the plate heatmap in it, ``<src>/results/`` with the two
    threshold graphs, and two runs under that. Returns ``<src>/results`` --
    the runs live in it, so it is what the fixtures below hang paths off.
    """
    src = tmp_path / "screen"
    root = src / "results"
    root.mkdir(parents=True)
    for name in SCREEN_FIGURES:
        _figure(src / name)
    for run in ("ols_11", "ols_12"):
        _figure(root / run / "regression_figure.pdf")
        _figure(root / run / "regression_qc" / "panel_00.pdf")
    return root


@pytest.fixture()
def screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    assert widget._figure_grid is not None
    return widget


def _handed_to_the_grid(screen, folder):
    """``(titles, sections)`` the loader passes to the grid.

    Captured at the SEAM rather than read back off a cell: a cell keeps no
    title attribute, and earlier versions of these tests guessed one and
    failed for no reason. What the loader hands over is the observable thing
    and the thing the complaint is about.
    """
    captured = {}
    original = screen._figure_grid.set_figures

    def spy(pixmaps, titles=None, sections=None):
        captured["titles"] = list(titles or [])
        captured["sections"] = list(sections or [])
        return original(pixmaps, titles, sections)

    screen._figure_grid.set_figures = spy
    try:
        screen._load_trial_figures(str(folder))
    finally:
        screen._figure_grid.set_figures = original
    return captured.get("titles", []), captured.get("sections", [])


# --------------------------------------------------------------------------- #
#  They are found at all
# --------------------------------------------------------------------------- #

def test_the_grna_threshold_graph_is_on_the_grid(screen, screen_folder):
    """The one the maintainer named: "i dont see the grna threshold graph"."""
    titles, _sections = _handed_to_the_grid(screen, screen_folder / "ols_12")

    assert "fraction_threshold" in titles, titles


def test_all_three_screen_figures_are_found(screen, screen_folder):
    """Including the plate heatmap, which is a folder further up than the
    other two -- the detail the brief for this work had wrong and the reason
    the climb is a list of folders rather than a parent."""
    titles, _sections = _handed_to_the_grid(screen, screen_folder / "ols_12")

    for written, caption in SCREEN_FIGURES.items():
        assert caption in titles, (written, titles)


def test_the_runs_own_figures_are_still_there(screen, screen_folder):
    """Reaching one level up must not cost the recursive walk downwards."""
    titles, _sections = _handed_to_the_grid(screen, screen_folder / "ols_12")

    assert "regression_figure" in titles, titles
    assert any("panel_00" in title for title in titles), titles


def test_the_count_is_the_run_plus_the_screen(screen, screen_folder):
    """2 of this run's own + 3 shared by the screen. The other run's 2 are
    NOT in it -- see the next test for why that matters."""
    assert screen._load_trial_figures(str(screen_folder / "ols_12")) == 5


# --------------------------------------------------------------------------- #
#  And they are labelled as the screen's, not as this run's
# --------------------------------------------------------------------------- #

def test_they_are_in_their_own_section(screen, screen_folder):
    """A section, because the grid letters its cells per section and a panel
    letter belongs to a figure -- and because these are not this run's
    output."""
    _titles, sections = _handed_to_the_grid(screen, screen_folder / "ols_12")

    assert len(sections) == 2, sections
    assert sections[0][0] == "ols_12"
    assert sections[0][1:] == (0, 2)
    assert sections[1][1:] == (2, 3)


def test_the_section_says_the_figures_are_not_this_runs(screen,
                                                        screen_folder):
    """"clearly labelled as belonging to the screen rather than to this run"
    -- a heading reading `results` would be read as a folder name and tell a
    reader nothing about whose figures these are."""
    _titles, sections = _handed_to_the_grid(screen, screen_folder / "ols_12")

    heading = sections[1][0]
    assert "not this run" in heading.lower(), heading


def test_both_screen_folders_land_in_the_one_section(screen, screen_folder):
    """The heatmap comes from `<src>/` and the two thresholds from
    `<src>/results/`, and a reader is being told one thing about all three:
    they are not this run's. Which directory above the run each happens to
    sit in is not a distinction anyone can act on, so it is not a heading."""
    _titles, sections = _handed_to_the_grid(screen, screen_folder / "ols_12")

    assert len(sections) == 2, sections
    assert sections[1][2] == 3


def test_the_sibling_runs_figures_are_not_dragged_in(screen, screen_folder):
    """The reason the screen folder is listed FLAT. Its subfolders are the
    other runs, and a recursive sweep of it would put ols_11's figures on
    ols_12's grid under ols_12's heading -- a worse answer than missing
    three, because it is wrong rather than incomplete."""
    titles, _sections = _handed_to_the_grid(screen, screen_folder / "ols_12")

    assert len(titles) == 5, titles
    assert not any("ols_11" in title for title in titles), titles


# --------------------------------------------------------------------------- #
#  What it must not break
# --------------------------------------------------------------------------- #

def test_a_figure_the_run_already_carries_is_not_shown_twice(screen,
                                                             screen_folder):
    """`spacr.ml._keep_figures_with_the_run`, landed the same day, makes a
    run COPY these into its own folder under the same basename.

    So from the next run onwards the recursive walk finds them where they
    belong and the screen sweep must not add a second copy beside it. The
    sweep still earns its place: it is what rescues the runs that predate the
    copy, which is every run already on the maintainer's screen.
    """
    _figure(screen_folder / "ols_12" / "fraction_threshold.pdf")

    titles, sections = _handed_to_the_grid(screen, screen_folder / "ols_12")

    assert titles.count("fraction_threshold") == 1, titles
    # It is the RUN's copy that is kept -- the run folder is the record of
    # the run, and the section it sits in is the claim being made about it.
    assert sections[0][1:] == (0, 3)
    assert sections[1][1:] == (3, 2)


def test_a_run_whose_screen_folder_has_no_figures_is_unchanged(screen,
                                                               tmp_path):
    """The ordinary case before this change, and it has to stay ordinary:
    one section, the run's own."""
    _figure(tmp_path / "run" / "a.pdf")
    _figure(tmp_path / "run" / "b.pdf")

    titles, sections = _handed_to_the_grid(screen, tmp_path / "run")

    assert len(titles) == 2
    assert len(sections) == 1
    assert sections[0] == ("run", 0, 2)


def test_a_run_with_no_figures_empties_the_grid(screen, screen_folder):
    """128 J makes the results follow the selected run. A grid still holding
    the last run's pictures beside this run's coefficient table is the exact
    disagreement that binding exists to remove, so "this run drew nothing"
    has to show as nothing rather than as somebody else's figures."""
    empty = screen_folder.parent / "empty_screen" / "ols_99"
    empty.mkdir(parents=True)
    assert screen._load_trial_figures(str(screen_folder / "ols_12")) == 5

    assert screen._load_trial_figures(str(empty)) == 0
    assert screen._figure_grid._cells == []
