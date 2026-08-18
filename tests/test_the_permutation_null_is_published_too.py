"""The permutation null is published, and its resolution is the user's.

Instruction 139 C left one bare ``fig.savefig`` on the regression path, named
in its own trailing note: ``spacr/guide_permutation.py`` wrote the permutation
volcano with

    fig.savefig(save_path, dpi=600 if save_path.suffix.lower() == ".png"
                else None)

which is two bugs in one line, and they are the same two ``_finish`` had.

  * SAVED AND INVISIBLE. A figure reaches the spaCR gallery through
    ``figure_sink.publish`` or through ``plt.show``, and this used neither, so
    the permutation null was on disk and in no gallery. It is the only picture
    a nonparametric run draws of its own null, which makes it the worst one to
    lose.
  * THE RESOLUTION PREFERENCE REACHED NEITHER FORMAT. 600 is hard-coded, so a
    user asking for 150 or 1200 got 600 on a PNG; and the conditional passes
    ``dpi=None`` for a PDF, whose pages are full of scatter marks that are
    rasterised at the figure's own 100 DPI unless told otherwise. A preference
    that reaches no format is a control that does nothing.

WHAT IS DELIBERATELY *NOT* CHANGED, and it is asserted here so it is not read
as an oversight: ``ml.py`` asks for this picture twice, ``for suffix in
('pdf', 'png')``. Letting the format preference rewrite both extensions would
collapse them onto one destination -- one of the two files a run has always
written would vanish, and 139 C's acceptance is a COUNT of tiles against
files. So an extension that names a real figure format is read as a request
and honoured; anything else lets the preference decide.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr import figure_sink  # noqa: E402
from spacr import guide_permutation as gp  # noqa: E402


@pytest.fixture(autouse=True)
def _no_sink_left_behind():
    """A sink is global; a test that leaks one breaks the next."""
    figure_sink.clear_sink()
    yield
    figure_sink.clear_sink()


def _results():
    """Eight guides, three of them called, in the columns the volcano reads."""
    effects = np.array([-1.9, -1.2, -0.4, -0.1, 0.2, 0.5, 1.4, 2.1])
    adjusted = np.array([0.001, 0.02, 0.4, 0.9, 0.8, 0.3, 0.06, 0.004])
    return pd.DataFrame({
        "guide": [f"g{index}" for index in range(len(effects))],
        "outcome": "score",
        "minimum_wells_threshold": 1,
        "standardized_marginal_effect": effects,
        "adjusted_p_value": adjusted,
        "significant": adjusted < 0.05,
        "alpha": 0.05,
        "multiple_testing_method": "fdr_bh",
    })


def _draw(save_path, **kwargs):
    return gp.plot_guide_permutation_volcano(
        _results(), outcome="score", minimum_wells=1,
        save_path=save_path, **kwargs)


def _opens(path):
    """Magic bytes, not the extension. The extension is what was wrong."""
    with open(path, "rb") as handle:
        head = handle.read(8)
    if str(path).lower().endswith(".pdf"):
        return head.startswith(b"%PDF")
    if str(path).lower().endswith(".png"):
        return head.startswith(b"\x89PNG\r\n\x1a\n")
    return False


def test_the_permutation_volcano_is_announced_not_only_written(tmp_path):
    """The reported bug. With a sink listening, exactly one figure arrives,
    carrying the path that was actually written."""
    announced = []
    figure_sink.set_sink(lambda fig, path: announced.append(path))

    written = _draw(str(tmp_path / "null.pdf"))

    assert os.path.isfile(written)
    assert announced == [str(written)], (
        "the permutation null reached the gallery "
        f"{len(announced)} time(s), not once")


def test_one_picture_one_file_one_tile(tmp_path):
    """139 C's acceptance is a count: tiles against image files."""
    announced = []
    figure_sink.set_sink(lambda fig, path: announced.append(path))

    _draw(str(tmp_path / "null.png"))

    on_disk = sorted(os.listdir(tmp_path))
    assert len(on_disk) == 1 == len(announced)


def test_a_headless_run_still_writes_it(tmp_path):
    """No sink installed is the CLI and the notebook. The file is still
    written and nothing raises."""
    assert figure_sink.sink() is None
    written = _draw(str(tmp_path / "null.png"))
    assert os.path.isfile(written) and _opens(written)


@pytest.mark.parametrize("suffix", ["pdf", "png"])
def test_the_resolution_preference_reaches_both_formats(tmp_path, monkeypatch,
                                                        suffix):
    """The dpi bug, measured on the number matplotlib is handed.

    Before: PNG got a hard-coded 600 whatever the preference said, and PDF got
    ``None``. Both are pinned here because the old line was wrong in two
    different ways at once.
    """
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences",
                        lambda: (suffix, 331))
    seen = {}
    original = matplotlib.figure.Figure.savefig

    def spy(self, *args, **kwargs):
        seen["dpi"] = kwargs.get("dpi")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)

    _draw(str(tmp_path / f"null.{suffix}"))

    assert seen["dpi"] == 331, (
        f"the resolution preference did not reach the {suffix}: {seen['dpi']}")


def test_an_explicit_dpi_still_wins(tmp_path, monkeypatch):
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences", lambda: ("png", 331))
    seen = {}
    original = matplotlib.figure.Figure.savefig

    def spy(self, *args, **kwargs):
        seen["dpi"] = kwargs.get("dpi")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)

    _draw(str(tmp_path / "null.png"), dpi=72)
    assert seen["dpi"] == 72


def test_both_formats_ml_asks_for_still_come_out(tmp_path, monkeypatch):
    """`ml.py` writes this picture as a PDF *and* as a PNG. Two names that
    differ only in extension must stay two files that each open."""
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences", lambda: ("png", 150))
    announced = []
    figure_sink.set_sink(lambda fig, path: announced.append(path))

    pdf = _draw(str(tmp_path / "null.pdf"))
    png = _draw(str(tmp_path / "null.png"))

    assert str(pdf) != str(png)
    assert _opens(pdf) and _opens(png), "an extension named a format it is not"
    assert len(announced) == len(sorted(os.listdir(tmp_path))) == 2


def test_a_path_without_a_figure_extension_takes_the_preference(tmp_path,
                                                                monkeypatch):
    """Nothing was requested, so the preference decides -- which is what will
    happen to the whole call once `ml.py` stops asking for two formats."""
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences", lambda: ("png", 150))

    written = _draw(str(tmp_path / "guide_permutation_min_2.5_wells"))

    assert str(written).endswith(".png")
    assert _opens(written)
    # `_with_extension` only replaces a KNOWN figure extension, so the '.5'
    # is part of the name and not an extension to overwrite.
    assert "min_2.5_wells" in os.path.basename(str(written))


def test_an_explicit_format_beats_both_the_path_and_the_preference(
        tmp_path, monkeypatch):
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences", lambda: ("png", 150))
    written = _draw(str(tmp_path / "null.png"), fmt="pdf")
    assert str(written).endswith(".pdf") and _opens(written)


def test_the_figure_is_closed_and_not_leaked(tmp_path):
    """`publish(close=True)` clears a figure without releasing it: this one
    comes from `plt.subplots`, so pyplot holds it until `plt.close`. A sweep
    over four thresholds and two outcomes calls this eight times."""
    plt.close("all")
    before = set(plt.get_fignums())
    _draw(str(tmp_path / "null.png"))
    assert set(plt.get_fignums()) == before


def test_a_sink_that_raises_does_not_lose_the_file(tmp_path):
    def angry(fig, path):
        raise RuntimeError("the gallery went away mid-run")

    figure_sink.set_sink(angry)
    written = _draw(str(tmp_path / "null.png"))
    assert os.path.isfile(written)


def test_no_bare_savefig_is_left_on_the_regression_path():
    """The grep the instruction asks for, kept as an assertion.

    ``spacr.plot.save_figure`` is the one place a kept figure is written, and
    ``figure_sink.publish`` is the one place it is written AND announced. A
    module on the regression path that calls ``savefig`` itself is a figure
    the format preference, the resolution preference and the gallery all miss
    at once -- which was true of five modules this week.

    PARSED, NOT GREPPED: three of these files DISCUSS ``savefig`` in prose,
    and a text search that counts a docstring is a check nobody can keep
    green. ``spacr/plot.py`` is excluded on purpose -- ``save_figure`` is the
    one call that is supposed to be there.
    """
    import ast
    import pathlib

    root = pathlib.Path(gp.__file__).parent
    offenders = []
    for name in ("guide_permutation.py", "regression_diagnostics.py",
                 "regression_qc.py", "ml.py", "toxo.py", "figure_sink.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "savefig"):
                offenders.append(f"{name}:{node.lineno}")
    assert not offenders, (
        "a bare savefig is back on the regression path: " + ", ".join(offenders))
