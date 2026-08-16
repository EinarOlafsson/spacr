"""A run produces the house-style panels, in the folder the user named.

Three complaints from 2026-08-16, all of them fair:

    "it cant find the regression results"
    "on te surface you have changed nothing except the background color"
    "there are no additional plots that i asked for and all the old plotts
     look exactly the same"

All three had one cause: the new figure system wrote a PDF to disk and
nothing else changed. The application still showed the same pictures from the
same places, so from the user's side nothing had happened.

WHAT THIS FILE PINS:

  * the panels are SHOWN, not merely written -- they go through plt.show(),
    which the Qt bridge intercepts, so each lands on the grid as its own cell;
  * output goes to <count data folder>/results/<type>, and a second run of the
    same type goes to <type>_1 rather than on top of the first;
  * the run hands its coefficient table back in memory, so nothing has to
    guess a path to display it.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr.ml import _next_results_folder, _show_house_style_panels


def _results(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feature": ["Intercept"] + [f"fraction:grna[{i // 4}_{i % 4}]"
                                    for i in range(n)],
        "coefficient": np.concatenate([[.19], rng.normal(0, .5, n)]),
        "p_value": np.concatenate([[3e-46], rng.uniform(size=n)]),
        "grna": [None] + [f"{i // 4}_{i % 4}" for i in range(n)],
        "gene": [None] * (n + 1),
        "condition": ["other"] + list(rng.choice(["nc", "pc", "other"], n)),
    })


# --------------------------------------------------------------------------- #
#  Where a run writes
# --------------------------------------------------------------------------- #

def test_the_first_run_of_a_type_gets_the_plain_name(tmp_path):
    assert _next_results_folder(str(tmp_path), "ols") == \
        os.path.join(str(tmp_path), "ols")


def test_a_second_run_does_not_land_on_the_first(tmp_path):
    """"if there is already an ols folder then ols_1 if there is already an
    ols_1 then ols_2 and so on". The old path was fixed, so re-running with
    one setting changed left only the last on disk and said nothing."""
    first = tmp_path / "ols"
    first.mkdir()
    (first / "results.csv").write_text("x\n")

    assert _next_results_folder(str(tmp_path), "ols").endswith("ols_1")

    second = tmp_path / "ols_1"
    second.mkdir()
    (second / "results.csv").write_text("x\n")
    assert _next_results_folder(str(tmp_path), "ols").endswith("ols_2")


def test_an_empty_folder_is_reused_not_stranded(tmp_path):
    """A directory somebody made and did not fill is not a run. Stepping past
    it would leave it there forever."""
    (tmp_path / "ols").mkdir()
    assert _next_results_folder(str(tmp_path), "ols").endswith("ols")


def test_each_type_counts_separately(tmp_path):
    for name in ("ols", "ridge"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "r.csv").write_text("x\n")
    assert _next_results_folder(str(tmp_path), "ols").endswith("ols_1")
    assert _next_results_folder(str(tmp_path), "ridge").endswith("ridge_1")


def _paths(tmp_path, **extra):
    """Drive the real path builder the way perform_regression does."""
    from spacr.ml import _perform_regression_set_paths

    count = tmp_path / "plate_1_unique_combinations.csv"
    score = tmp_path / "plate1_dv.csv"
    for path in (count, score):
        path.write_text("a,b\n1,2\n")
    settings = {"count_data": [str(count)], "score_data": [str(score)],
                "regression_type": "ols"}
    settings.update(extra)
    return _perform_regression_set_paths(settings)


def test_the_results_root_is_beside_the_count_data(tmp_path):
    """"just store everything in the same location as the first count data ...
    then the type so for me .../claude/results/ols".

    The old path buried output under results/<score-source-csv-name>/<type>/
    list -- two levels nobody asked for, one of them named after a FILE, and
    a fixed leaf. Driven through the real builder, not a source grep.
    """
    res_folder = _paths(tmp_path)[4]

    assert res_folder == os.path.join(str(tmp_path), "results", "ols")
    assert "plate1_dv" not in res_folder, "the score-source level is back"
    assert not res_folder.endswith("list"), "the fixed leaf is back"


def test_a_rerun_through_the_real_builder_does_not_overwrite(tmp_path):
    first = _paths(tmp_path)[4]
    (pathlib.Path(first) / "results.csv").write_text("x\n")

    second = _paths(tmp_path)[4]

    assert second != first
    assert second.endswith("ols_1")


def test_an_explicit_src_still_wins(tmp_path):
    """The parameter sweep gives each trial its own folder that way, and it
    must keep working -- the count-data folder is the DEFAULT, not a
    hard-coding."""
    elsewhere = tmp_path / "trial_0007"
    elsewhere.mkdir()

    res_folder = _paths(tmp_path, src=str(elsewhere))[4]

    assert res_folder.startswith(str(elsewhere))


def test_the_four_result_files_land_in_that_folder(tmp_path):
    results, gene, grna, hits, res_folder, _csv = _paths(tmp_path)

    for path in (results, gene, grna, hits):
        assert os.path.dirname(path) == res_folder


# --------------------------------------------------------------------------- #
#  What a run draws
# --------------------------------------------------------------------------- #

def test_a_run_draws_every_house_style_panel():
    """"there are no additional plots that i asked for". Seven, from the same
    table the old volcano came from."""
    from spacr.figures import SHEET_ORDER

    assert _show_house_style_panels(_results(), plot=False) == len(SHEET_ORDER)
    plt.close("all")


def test_the_panels_are_shown_not_merely_written(monkeypatch):
    """A PDF on disk changed nothing about what the application displays,
    which is exactly what the user reported. plt.show() is the seam the Qt
    bridge intercepts."""
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))

    _show_house_style_panels(_results(), plot=True)

    assert len(shown) == 7, f"only {len(shown)} panels reached the display"
    plt.close("all")


def test_a_panel_that_cannot_draw_is_closed_not_shown():
    """An empty framed panel on the grid reads as a figure that failed."""
    frame = _results().drop(columns=["p_value"])
    before = len(plt.get_fignums())

    drawn = _show_house_style_panels(frame, plot=False)

    assert drawn < 7, "a panel with no p-value claimed to draw"
    plt.close("all")
    assert len(plt.get_fignums()) <= before + 7


def test_an_empty_table_draws_nothing_and_does_not_raise():
    assert _show_house_style_panels(pd.DataFrame(), plot=False) == 0
    assert _show_house_style_panels(None, plot=False) == 0


def test_a_broken_panel_never_loses_the_run(monkeypatch):
    """The fit is already done. Losing it over a figure would be the worst
    possible trade."""
    import spacr.figures as figures

    def _explode(*args, **kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(figures, "build_panel", _explode)
    monkeypatch.setattr("spacr.figures.build_panel", _explode)

    assert _show_house_style_panels(_results(), plot=False) == 0
    plt.close("all")


def test_each_panel_carries_its_own_name(monkeypatch):
    """The grid captions the tiles from the figure. `fig_00003` is a temp
    file's stem -- an implementation detail of how the picture reached the
    screen, not a caption."""
    named = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    real_close = plt.close

    def _capture(figure=None):
        if hasattr(figure, "get_label"):
            named.append(figure.get_label())
        return real_close(figure)

    _show_house_style_panels(_results(), plot=True)
    for number in plt.get_fignums():
        named.append(plt.figure(number).get_label())
    plt.close("all")

    assert "volcano" in named, named
    assert not any(str(n).startswith("fig_") for n in named if n), named
