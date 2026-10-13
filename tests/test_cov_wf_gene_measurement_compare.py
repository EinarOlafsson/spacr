"""What the comparison panel draws, and saves, when the data is too thin to plot.

Three situations this module has to survive without a console to complain in:
a violin asked of groups that hold a single observation each, a folder saved
for a comparison whose measurement is NaN on every row, and a folder saved for
a comparison that has no rows at all. In each case the panel still owes the
reader something honest -- an axis with the real counts on it, the numbers the
figure could not be drawn from, and a settings record naming what was asked --
because a saved folder is what a reader checks a claim against later, and a
folder that quietly contains nothing looks exactly like a folder for a
comparison that simply had no effect.
"""
from __future__ import annotations

import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from spacr import gene_measurement_compare as gmc
from spacr.gene_measurement_compare import REST, Comparison


def _comparison(values_by_group, *, level="cell", statistics=None):
    """A comparison built by hand, so frames ``build`` would never make can be drawn."""
    rows = []
    for group, values in values_by_group.items():
        for index, value in enumerate(values):
            rows.append({"group": group, "value": value,
                         "unit": f"{group}-{index}"})
    frame = pd.DataFrame(rows, columns=["group", "value", "unit"])
    return Comparison(measurement="pathogen_area", level=level, frame=frame,
                      statistics=list(statistics or []))


def _empty_comparison(*, level="well"):
    """A comparison whose frame has the right columns and no rows at all."""
    frame = pd.DataFrame(columns=["group", "value", "unit"])
    return Comparison(measurement="pathogen_area", level=level, frame=frame)


# ---------------------------------------------------------------------------
# a violin needs a spread, and one point per group is not one
# ---------------------------------------------------------------------------

def test_a_violin_of_one_point_per_group_draws_no_shape_but_keeps_its_counts():
    """A single observation has no distribution, so no violin may be invented for it.

    ``violinplot`` estimates a density; handed one value it either fails or
    draws a shape whose width is an artefact of the smoother, and a reader
    would read that width as spread that was measured. The figure still has to
    appear with its axis and its ``(n=1)`` labels, because the alternative --
    no figure -- hides that the comparison ran on one cell per group.
    """
    thin = _comparison({"g": [3.0], REST: [5.0]})
    thick = _comparison({"g": [3.0, 4.0, 9.0], REST: [5.0, 6.0, 11.0]})

    thin_figure = gmc.plot(thin, kind="violin")
    thick_figure = gmc.plot(thick, kind="violin")

    # The same call on data that DOES have a spread draws two bodies, so the
    # empty collection below is this input's doing and not a dead code path.
    assert len(thick_figure.axes[0].collections) == 2
    assert len(thin_figure.axes[0].collections) == 0

    labels = [t.get_text() for t in thin_figure.axes[0].get_xticklabels()]
    assert labels == ["g\n(n=1)", f"{REST}\n(n=1)"]
    assert thin_figure.axes[0].get_ylabel() == "pathogen_area"


def test_a_violin_skips_only_the_thin_group_and_leaves_the_others_in_place():
    """One single-observation group must not cost the groups that do have data.

    The bodies are drawn at explicit positions, so dropping a group from the
    drawing has to keep every remaining group on the x position its tick label
    claims. If the thin group shifted its neighbours left, every point in the
    figure would be labelled with the wrong gene.
    """
    comparison = _comparison({"g": [1.0], "h": [2.0, 3.0, 4.0],
                              REST: [5.0, 6.0, 7.0]})

    figure = gmc.plot(comparison, kind="violin")
    axes = figure.axes[0]
    labels = [t.get_text() for t in axes.get_xticklabels()]

    assert labels == ["g\n(n=1)", "h\n(n=3)", f"{REST}\n(n=3)"]
    assert list(axes.get_xticks()) == [0, 1, 2]
    # Two bodies for three groups: the thin one is skipped, not drawn flat.
    assert len(axes.collections) == 2
    centres = sorted(float(np.mean(body.get_paths()[0].vertices[:, 0]))
                     for body in axes.collections)
    assert centres[0] > 0.5 and centres[1] > 1.5


# ---------------------------------------------------------------------------
# saving a folder for a comparison that has no figure in it
# ---------------------------------------------------------------------------

def test_a_comparison_that_cannot_be_drawn_still_leaves_its_numbers_behind(
        tmp_path):
    """A measurement that is NaN everywhere still owes the reader its rows.

    ``plot`` returns nothing for data with no finite value, and the save must
    carry on from there: the CSV and the statistics are what tell a reader the
    comparison was attempted and on what. Stopping at the missing figure would
    leave a folder that cannot be told apart from one that was never written.
    """
    unplottable = _comparison({"g": [np.nan, np.inf], REST: [np.nan]},
                              statistics=[{"test": "t-test", "p": 0.5}])
    drawable = _comparison({"g": [1.0, 2.0, 3.0], REST: [4.0, 5.0, 6.0]})

    written = gmc.save(unplottable, str(tmp_path / "nan"))
    drawn = gmc.save(drawable, str(tmp_path / "fine"))

    # The same call on drawable data DOES produce the two figure formats, so
    # their absence above is this comparison's doing.
    assert os.path.isfile(drawn["pdf"]) and os.path.isfile(drawn["png"])
    assert "pdf" not in written and "png" not in written

    assert sorted(written) == ["data", "settings", "statistics"]
    assert len(pd.read_csv(written["data"])) == 3
    assert pd.read_csv(written["statistics"])["test"].tolist() == ["t-test"]
    assert sorted(os.listdir(tmp_path / "nan")) == [
        "data.csv", "settings.json", "statistics.csv"]


def test_an_empty_comparison_writes_the_record_and_no_data_file(tmp_path):
    """A comparison with no rows must still say, in the folder, that it had none.

    ``settings.json`` is the only thing that reports what was asked for. When
    a group selection matches nothing, the empty ``groups`` list and empty
    ``n_per_group`` in that record are the evidence; an empty ``data.csv``
    beside it would suggest rows were plotted and lost instead.
    """
    empty = _empty_comparison(level="well")
    filled = _comparison({"g": [1.0, 2.0], REST: [3.0, 4.0]}, level="well",
                         statistics=[{"test": "t-test", "p": 0.1}])

    written = gmc.save(empty, str(tmp_path / "empty"), kind="bar")
    populated = gmc.save(filled, str(tmp_path / "filled"), kind="bar")

    # The same call on a frame with rows writes both, so their absence in the
    # empty folder is the empty frame's doing.
    assert os.path.isfile(populated["data"])
    assert os.path.isfile(populated["statistics"])
    assert list(written) == ["settings"]
    assert os.listdir(tmp_path / "empty") == ["settings.json"]

    record = json.loads((tmp_path / "empty" / "settings.json").read_text())
    assert record["groups"] == []
    assert record["n_per_group"] == {}
    assert record["measurement"] == "pathogen_area"
    assert record["plot"] == "bar"
    assert record["level"] == "well"
    assert record["level_means"].startswith("one row per well")


def test_an_empty_comparison_folder_is_created_and_named_in_the_return(
        tmp_path):
    """The destination folder is made by the save, not assumed to exist.

    The panel hands over a path a user typed. If the save only wrote into a
    folder that already existed, an empty comparison would raise out of the
    panel instead of leaving the record that explains why it is empty.
    """
    target = tmp_path / "nested" / "run-01"
    assert not target.exists()

    written = gmc.save(_empty_comparison(), str(target),
                       settings={"threshold": np.float64(0.25),
                                 "genes": ("a", "b")})

    assert target.is_dir()
    assert written["settings"] == str(target / "settings.json")
    record = json.loads((target / "settings.json").read_text())
    assert record["regression_settings"] == {"threshold": 0.25,
                                             "genes": ["a", "b"]}


# ---------------------------------------------------------------------------
# the styled renderer's own violin guard
# ---------------------------------------------------------------------------

def test_the_styled_violin_draws_every_group_that_has_a_single_point():
    """The styled renderer keeps a one-point group that the quick plot drops.

    ``render_comparison`` is the live-canvas path: a user restyling a figure
    watches groups appear and disappear as they switch plot kind. It admits
    any group with at least one value, which is exactly what the renderer's
    own "nothing finite to draw" guard has already established, so a violin
    here is never silently empty. Tightening it to match :func:`plot` would
    make a group vanish from the canvas without explanation.
    """
    comparison = _comparison({"g": [3.0], REST: [5.0]})

    figure, axes = gmc.render_comparison(
        comparison, gmc.ComparisonStyle(kind="violin"))

    assert len(axes.collections) == 2
    labels = [t.get_text() for t in axes.get_xticklabels()]
    assert labels == ["g\n(n=1)", f"{REST}\n(n=1)"]
    assert figure.axes[0] is axes


def test_the_styled_renderer_refuses_a_frame_with_no_finite_value():
    """Nothing finite means no canvas, and that guard is what makes the violin safe.

    Every drawing branch below it assumes at least one group has values. A
    comparison of NaNs has to stop here and return no figure, so the panel can
    say so, rather than reaching a violin or a bar that would be drawn on an
    empty array.
    """
    nothing = _comparison({"g": [np.nan], REST: [np.inf]})
    something = _comparison({"g": [np.nan, 2.0], REST: [np.inf, 7.0]})

    refused = gmc.render_comparison(nothing, gmc.ComparisonStyle(kind="violin"))
    drawn, axes = gmc.render_comparison(something,
                                        gmc.ComparisonStyle(kind="violin"))

    # One finite value per group is enough to get a canvas, so the refusal
    # above is the all-NaN frame's doing.
    assert [t.get_text() for t in axes.get_xticklabels()] == [
        "g\n(n=1)", f"{REST}\n(n=1)"]
    assert drawn is axes.figure
    assert refused == (None, None)


def test_a_group_filter_that_matches_nothing_stops_before_any_drawing_branch():
    """A "draw only this group" filter naming an absent group must refuse, not draw.

    ``style.only`` is a live-canvas control: a user types or picks a group name
    and the renderer redraws. That filter is applied AFTER the comparison is
    accepted, so it is the one way the data can shrink to nothing on its way to
    the plot kinds below. If it fell through, ``violinplot`` would be handed an
    empty list of positions and the panel would raise instead of redrawing.
    Returning ``(None, None)`` is what lets the panel say "that group is not in
    this comparison" and keep the previous figure on screen.
    """
    comparison = _comparison({"g": [3.0, 4.0, 9.0], REST: [5.0, 6.0, 11.0]})

    absent = gmc.render_comparison(
        comparison, gmc.ComparisonStyle(kind="violin", only="not_a_group"))
    figure, axes = gmc.render_comparison(
        comparison, gmc.ComparisonStyle(kind="violin", only="g"))

    # Naming a group that IS present draws it, so the refusal above is the
    # unmatched name's doing and not a renderer that never draws violins.
    assert [t.get_text() for t in axes.get_xticklabels()] == ["g\n(n=3)"]
    assert len(axes.collections) == 1
    assert figure is axes.figure
    assert absent == (None, None)
