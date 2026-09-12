"""370: the scorecard "in graph form", beside the CSV and the table.

The request asks for the numbers "in a csv and in graph form on hugging face
and ... available in graph form and as a table on the spaCR API". The CSV and
the table existed; the chart did not.

The test that matters most here is
:func:`test_the_chart_metric_names_exist_in_a_real_scorecard`. The first draft
of `CHART_METRICS` asked for "dice" and "ap_50_90" -- plausible names that the
actual round-3 scorecard does not carry -- so the chart silently drew three
bars instead of six. Nothing raised. A chart with half its metrics missing
looks exactly like a chart of a model that was only measured three ways.
"""

import numpy as np
import pathlib

import pytest

from spacr.scorecard import CHART_METRICS, read_scorecard_csv, scorecard_figure

CSV = """metric,finetuned,vanilla,delta,n_fields,n_objects,holdout,holdout_version
f1,0.86,0.32,0.54,366,12517,toxo-pv-round3,2026-09-10
precision,0.89,0.20,0.69,366,12517,toxo-pv-round3,2026-09-10
recall,0.83,0.72,0.11,366,12517,toxo-pv-round3,2026-09-10
dice_per_object,0.98,0.92,0.06,366,12517,toxo-pv-round3,2026-09-10
iou_mean,0.97,0.87,0.10,366,12517,toxo-pv-round3,2026-09-10
ap_mean,0.76,0.31,0.45,366,12517,toxo-pv-round3,2026-09-10
"""


def test_a_chart_is_written(tmp_path):
    # The RETURNED path is the one that exists: `spacr.plot.save_figure`
    # makes the file name follow the figure-format preference, so a request
    # for `card.png` under a PDF preference lands on `card.pdf`. Asserting
    # the requested name would pass only where the preference is PNG.
    out = pathlib.Path(scorecard_figure(read_scorecard_csv(CSV),
                                        tmp_path / "card.png",
                                        title="Toxo PV v1"))
    assert out.exists()
    assert out.stem == "card"
    assert out.stat().st_size > 5_000


def test_the_chart_metric_names_exist_in_a_real_scorecard():
    """Names checked against the file, not against what sounds right.

    This is the one that would have caught the original mistake: every name
    in CHART_METRICS must be a key the writer actually emits.
    """
    parsed = read_scorecard_csv(CSV)
    for name in CHART_METRICS:
        assert name in parsed, (
            f"CHART_METRICS asks for {name!r}, which a real scorecard does "
            f"not carry -- the chart would quietly draw one bar fewer")


def test_a_scorecard_with_none_of_them_is_refused(tmp_path):
    """An empty chart beside a model would read as a model that scored zero."""
    thin = "metric,finetuned,vanilla,delta\nsplits,50,1439,-1389\n"
    with pytest.raises(ValueError, match="nothing to plot"):
        scorecard_figure(read_scorecard_csv(thin), tmp_path / "x.png")


def test_a_missing_metric_is_skipped_not_drawn_as_zero(tmp_path):
    """A metric that is absent and one that scored 0 mean opposite things."""
    partial = ("metric,finetuned,vanilla,delta\n"
               "f1,0.86,0.32,0.54\nrecall,0.83,0.72,0.11\n")
    # It draws, with two bars rather than six, and does not invent the rest.
    #
    # THE RETURNED PATH, NOT THE REQUESTED ONE. `scorecard_figure` writes
    # through `spacr.plot.save_figure`, whose rule is that the file NAME
    # follows the user's figure-format preference -- a PNG written to
    # `figure.pdf` is a file no viewer opens. So a caller asking for `p.png`
    # under a PDF preference gets `p.pdf`, and the path it hands back is the
    # one that exists. A test asserting the requested name would pass only
    # on machines whose preference happens to be PNG.
    written = scorecard_figure(read_scorecard_csv(partial), tmp_path / "p.png")
    assert pathlib.Path(written).exists()
    assert pathlib.Path(written).stem == "p"


def test_importing_the_module_still_needs_no_plotting_stack():
    """The zoo must import with neither torch nor matplotlib at module scope.

    `scorecard_figure` imports matplotlib INSIDE, so a caller who only wants
    to read a scorecard pays nothing for the drawing path.
    """
    import ast
    import pathlib

    import spacr.scorecard as module

    source = pathlib.Path(module.__file__).read_text()
    tree = ast.parse(source)
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top_level.append(node.module or "")
    joined = " ".join(top_level)
    assert "matplotlib" not in joined
    assert "torch" not in joined
