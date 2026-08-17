"""The gRNA threshold graph is in the run folder, where the grid looks.

Reported 2026-08-17, instruction 128 P item 1: "for some reason now i dont
see the grna threshold graph".

ESTABLISHED FIRST, because the fix differs. The figure IS drawn and IS
streamed: `find_and_visualize_fraction_threshold` calls `plt.show()` with the
figure still open, and `spacr.qt.bridge._capture_show` emits every open figure
to the queue, so the live all-figures grid still receives it. Nothing stopped
that. `test_the_sweep_graph_is_open_at_plt_show` and
`test_the_cell_count_graph_is_open_at_plt_show` pin that half so a future
change that closes the figure before showing it fails here rather than
silently emptying the grid.

WHAT IS MISSING IS EVERY VIEW BUILT FROM DISK. Both figures are written
beside the COUNT DATA -- `<screen>/results/fraction_threshold.pdf`,
`<screen>/results/cell_min_threshold.pdf` and `<screen>/
plate_heatmap_unique_counts.pdf` -- and the run folder is `<screen>/results/
ols_12/`. `AppScreen._load_trial_figures` walks the RUN folder, so a run
reopened from the Runs tab shows none of them, and neither does anyone who
opens the run folder themselves.

So `perform_regression` copies whatever those two helpers drew into the run's
own folder. Copied, not moved: scripts and every past run still expect the
screen folder, and a figure is small.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from spacr.ml import (_figure_stamps, _keep_figures_with_the_run,
                      _screen_figure_folders, minimum_cell_simulation)


# --------------------------------------------------------------------------- #
#  A screen on disk
# --------------------------------------------------------------------------- #

def _screen_on_disk(tmp_path, wells=4, cells=60, guides=8):
    """A score CSV and a count CSV in one folder, as a real screen has."""
    folder = tmp_path / "screen"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    scores, counts = [], []
    for row in range(1, wells + 1):
        for column in range(1, wells + 1):
            for _ in range(cells):
                scores.append({"plateID": "plate1", "rowID": f"r{row}",
                               "columnID": f"c{column}",
                               "pred": float(rng.normal(0.5, 0.1))})
            for guide in range(guides):
                counts.append({"plateID": "plate1", "rowID": f"r{row}",
                               "columnID": f"c{column}", "grna": f"g{guide}",
                               "count": int(rng.integers(5, 400))})
    score_path = folder / "scores.csv"
    count_path = folder / "counts.csv"
    pd.DataFrame(scores).to_csv(score_path, index=False)
    pd.DataFrame(counts).to_csv(count_path, index=False)
    return {
        "score_data": [str(score_path)], "count_data": [str(count_path)],
        "score_column": "pred", "tolerance": 0.02, "min_cell_count": None,
        "target_unique_count": 5, "filter_column": "columnID",
        "control_wells": [], "log_x": False, "log_y": False,
    }, folder


class _ShowRecorder:
    """What `spacr.qt.bridge._capture_show` would have emitted.

    The bridge iterates `plt.get_fignums()` inside its `plt.show` replacement
    and emits every figure it finds, so "was it streamed?" is exactly "was a
    figure open when show() was called?".
    """

    def __init__(self):
        self.calls = []

    def __call__(self, *_args, **_kwargs):
        self.calls.append([_title(plt.figure(num))
                           for num in plt.get_fignums()])

    @property
    def newest(self):
        return self.calls[-1] if self.calls else []


def _title(figure):
    axes = figure.get_axes()
    return axes[0].get_title() if axes else ""


# --------------------------------------------------------------------------- #
#  The premise: they were streamed, and still are
# --------------------------------------------------------------------------- #

def test_the_cell_count_graph_is_open_at_plt_show(tmp_path, monkeypatch):
    """It reaches the LIVE figure queue, so the report is not "it stopped
    being drawn" -- it is "it is not in the run folder"."""
    settings, _folder = _screen_on_disk(tmp_path)
    recorder = _ShowRecorder()
    monkeypatch.setattr(plt, "show", recorder)

    minimum_cell_simulation(settings, num_repeats=3, increment=20)

    assert recorder.calls, "minimum_cell_simulation drew nothing"
    assert "Mean Absolute Difference vs. Sample Size" in " ".join(
        recorder.newest)
    plt.close("all")


def test_the_sweep_graph_is_open_at_plt_show(tmp_path, monkeypatch):
    """The gRNA threshold graph itself: drawn, open, and streamed."""
    from spacr.sequencing import graph_sequencing_stats

    settings, _folder = _screen_on_disk(tmp_path)
    recorder = _ShowRecorder()
    monkeypatch.setattr(plt, "show", recorder)

    graph_sequencing_stats(dict(settings))

    titles = " ".join(title for call in recorder.calls for title in call)
    assert "unique_count vs fraction_threshold" in titles
    plt.close("all")


def test_they_are_written_beside_the_count_data_not_beside_the_run(tmp_path):
    """The place the report names: the screen folder, one level up from where
    a run writes and shared by every run of that screen."""
    from spacr.sequencing import graph_sequencing_stats

    settings, folder = _screen_on_disk(tmp_path)
    minimum_cell_simulation(settings, num_repeats=3, increment=20)
    graph_sequencing_stats(dict(settings))
    plt.close("all")

    assert (folder / "results" / "cell_min_threshold.pdf").is_file()
    assert (folder / "results" / "fraction_threshold.pdf").is_file()
    # The plate heatmap lands one level higher still, directly in the screen
    # folder rather than in results/.
    heatmaps = [name for name in os.listdir(folder)
                if name.startswith("plate_heatmap_unique_counts")]
    assert heatmaps, sorted(os.listdir(folder))


# --------------------------------------------------------------------------- #
#  Where the run looks for them
# --------------------------------------------------------------------------- #

def test_the_screen_folders_are_the_two_the_helpers_write_to(tmp_path):
    """Both, and in this order: `graph_sequencing_stats` puts the sweep in
    `results/` and the plate heatmap in the folder above it."""
    settings, folder = _screen_on_disk(tmp_path)
    assert _screen_figure_folders(settings) == [
        str(folder), os.path.join(str(folder), "results")]


def test_a_missing_folder_is_not_an_error(tmp_path):
    """`results/` does not exist until the first run, and the snapshot is
    taken before it."""
    assert _figure_stamps([str(tmp_path / "nope")]) == {}


def test_only_figures_are_collected(tmp_path):
    """The screen folder holds the count and score CSVs. Copying those into
    every run folder would double the inputs on disk."""
    tmp_path.joinpath("counts.csv").write_text("a,b\n1,2\n")
    tmp_path.joinpath("figure.pdf").write_bytes(b"%PDF-1.4\n")
    tmp_path.joinpath("figure.PNG").write_bytes(b"\x89PNG\r\n")
    stamps = _figure_stamps([str(tmp_path)])
    assert sorted(os.path.basename(path) for path in stamps) == [
        "figure.PNG", "figure.pdf"]


def test_a_figure_left_by_an_earlier_run_is_not_claimed_by_this_one(tmp_path):
    """Every run of a screen writes the same file NAMES into the same shared
    folder, so identity has to include the stamp. A name-only comparison would
    copy last week's picture into today's run folder and label it today's."""
    screen = tmp_path / "screen"
    screen.mkdir()
    stale = screen / "fraction_threshold.pdf"
    stale.write_bytes(b"%PDF-old\n")

    before = _figure_stamps([str(screen)])
    run = tmp_path / "results" / "ols"
    assert _keep_figures_with_the_run(before, [str(screen)], str(run)) == []
    assert not run.exists()

    # Rewritten with different content: now it belongs to this run.
    stale.write_bytes(b"%PDF-new-and-longer\n")
    kept = _keep_figures_with_the_run(before, [str(screen)], str(run))
    assert kept == [str(run / "fraction_threshold.pdf")]
    assert (run / "fraction_threshold.pdf").read_bytes() == b"%PDF-new-and-longer\n"


def test_the_original_is_left_where_it_was(tmp_path):
    """Copied, not moved. Scripts and every past run still expect the screen
    folder, and a reader who knows where it used to be must still find it."""
    screen = tmp_path / "screen"
    screen.mkdir()
    run = tmp_path / "run"
    before = _figure_stamps([str(screen)])
    (screen / "fraction_threshold.pdf").write_bytes(b"%PDF-1.4\n")

    _keep_figures_with_the_run(before, [str(screen)], str(run))
    assert (screen / "fraction_threshold.pdf").is_file()
    assert (run / "fraction_threshold.pdf").is_file()


def test_copying_into_the_folder_a_figure_is_already_in_is_a_no_op(tmp_path):
    """A caller that names the screen folder as the run folder must not have
    `shutil.copy2` asked to copy a file onto itself."""
    screen = tmp_path / "screen"
    screen.mkdir()
    before = _figure_stamps([str(screen)])
    (screen / "fraction_threshold.pdf").write_bytes(b"%PDF-1.4\n")
    assert _keep_figures_with_the_run(before, [str(screen)], str(screen)) == []
    assert (screen / "fraction_threshold.pdf").is_file()


def test_a_copy_that_fails_costs_a_message_and_not_the_run(tmp_path, capsys,
                                                           monkeypatch):
    """The threshold these figures describe has already been computed. A
    read-only disk must not throw it away."""
    import shutil

    screen = tmp_path / "screen"
    screen.mkdir()
    before = _figure_stamps([str(screen)])
    (screen / "fraction_threshold.pdf").write_bytes(b"%PDF-1.4\n")

    def refuse(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(shutil, "copy2", refuse)
    assert _keep_figures_with_the_run(before, [str(screen)],
                                      str(tmp_path / "run")) == []
    assert "Could not keep fraction_threshold.pdf with the run" in \
        capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  End to end, through perform_regression
# --------------------------------------------------------------------------- #

def _regression_screen(tmp_path):
    """A screen `perform_regression` can actually fit, on disk."""
    genes = ("000000", "233460", "239740", "111111")
    rows = ("r1", "r2", "r3")
    columns = ("c1", "c2", "c3", "c4", "c5", "c6")
    guides = [f"TGGT1_{gene}_{index}" for gene in genes
              for index in range(1, 4)]
    rng = np.random.default_rng(1)

    folder = tmp_path / "screen"
    folder.mkdir(parents=True, exist_ok=True)
    scores, counts = [], []
    for row in rows:
        for column in columns:
            base = float(rng.uniform(0.2, 0.8))
            for _ in range(6):
                scores.append({
                    "plateID": "plate1", "rowID": row, "columnID": column,
                    "fieldID": "f1",
                    "pred": float(np.clip(base + rng.normal(0, 0.1),
                                          0.02, 0.98))})
            for guide in guides:
                counts.append({"plateID": "plate1", "rowID": row,
                               "columnID": column, "grna": guide,
                               "count": int(rng.integers(20, 400))})
    score_path = folder / "xgb_scores.csv"
    count_path = folder / "counts.csv"
    pd.DataFrame(scores).to_csv(score_path, index=False)
    pd.DataFrame(counts).to_csv(count_path, index=False)
    return str(score_path), str(count_path), folder


def test_a_run_leaves_both_threshold_graphs_in_its_own_folder(tmp_path):
    """The whole point: `AppScreen._load_trial_figures` walks the run folder,
    so the figures have to be IN it.

    Driven through `perform_regression` with nothing stubbed, and with
    `fraction_threshold=None` and `min_cell_count=None` so both helpers are
    the ones that choose -- which is the only condition under which either
    figure is drawn at all.
    """
    from spacr.ml import perform_regression
    from spacr.settings import get_perform_regression_default_settings

    score, count, folder = _regression_screen(tmp_path)
    settings = get_perform_regression_default_settings({
        "score_data": [score], "count_data": [count],
        "dependent_variable": "pred", "regression_type": "ols",
        "min_cell_count": None, "fraction_threshold": None,
        "metadata_files": [], "toxo": False, "controls": None,
        "outlier_detection": False, "alpha": 1.0, "regression_qc": False,
    })
    perform_regression(settings)
    plt.close("all")

    run_folder = os.path.join(str(folder), "results", "ols")
    figures = sorted(name for name in os.listdir(run_folder)
                     if name.lower().endswith((".pdf", ".png")))
    assert any(name.startswith("cell_min_threshold") for name in figures), figures
    assert any(name.startswith("fraction_threshold") for name in figures), figures
    assert any(name.startswith("plate_heatmap_unique_counts")
               for name in figures), figures
    # And the screen folder still has its own copies, so nothing moved.
    assert os.path.isfile(
        os.path.join(str(folder), "results", "fraction_threshold.pdf"))


def test_a_set_fraction_threshold_says_why_there_is_no_sweep_graph(tmp_path,
                                                                   capsys):
    """The graph is a by-product of CHOOSING the threshold, so setting one --
    including loading a settings CSV carrying a value an earlier run derived
    -- means it is never drawn. Silence there is indistinguishable from the
    figure having gone missing, which is how it was reported."""
    import inspect

    from spacr.ml import perform_regression

    source = inspect.getsource(perform_regression)
    assert "the gRNA fraction-threshold sweep graph is not drawn" in source
    body = source.split("if settings['fraction_threshold'] is None:", 1)[1]
    assert "_keep_figures_with_the_run" in body
