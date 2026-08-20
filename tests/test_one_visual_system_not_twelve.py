"""Instruction 136 — the old matplotlib figures get the house style.

    "also overhaul the old matplotlib figures to look better in accordance
     with the figure making skill"

Two checks this instruction asks for BY NAME, and one it implies.

THE COUNT HAS FLIPPED. When 136 was filed it was 133 raw `plt.subplots(` /
`plt.figure(` against 4 uses of `figure_style`. It is the other way round now,
and this holds the direction — the same argument as 145's reader ratchet: a
partial conversion silently un-converts itself as new code is written, and a
figure drawn outside the house style does not FAIL, it just looks like a
different application.

NO MODULE WRITES rcParams GLOBALLY. `plt.rcParams` is process-wide, so a
module that themes one figure through it themes every LATER figure in the
process — including one being saved for paper. `spacr.figure_style` is the one
place allowed to, because applying the house style globally is its job.
"""
from __future__ import annotations

import collections
import pathlib
import re

import pytest

RAW_FIGURE = re.compile(r"plt\.subplots\(|plt\.figure\(")
STYLE_USE = re.compile(r"figure_style|from \.figures\.style|rc_params\(")

#: Writes that reach process-wide state. `rc_context` is the scoped form and
#: is not one.
GLOBAL_RC = re.compile(r"(?:plt|mpl|matplotlib)\.rcParams\s*(?:\.update\(|\[)")

#: The ceiling on raw figure creation, measured 2026-08-20. LOWER IT as call
#: sites are converted; never raise it without saying why the figure cannot
#: go through the house style.
CEILING = 145

#: Allowed to write rcParams globally, and why.
MAY_SET_RCPARAMS = {
    # Applying the house style process-wide IS this module's job -- it is
    # what `figure_style.apply()` means.
    "figure_style.py",
}


def _spacr() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "spacr"


def _files():
    for path in sorted(_spacr().rglob("*.py")):
        if "i18n_catalogs" in str(path):
            continue
        yield path, path.read_text(encoding="utf-8")


def test_the_checker_is_reading_the_tree_it_thinks_it_is():
    assert (_spacr() / "figure_style.py").is_file()
    assert sum(len(RAW_FIGURE.findall(text)) for _p, text in _files()) > 0


def test_raw_figure_creation_does_not_go_up():
    counts = collections.Counter()
    for path, text in _files():
        found = len(RAW_FIGURE.findall(text))
        if found:
            counts[path.name] = found
    total = sum(counts.values())
    assert total <= CEILING, (
        f"{total} raw figure creations, up from {CEILING}. A figure drawn "
        f"outside the house style does not fail -- it just looks like a "
        f"different application. Route it through spacr.figure_style, or "
        f"lower this ceiling deliberately.\n"
        + "\n".join(f"  {n:3}  {name}" for name, n in counts.most_common(8)))


def test_the_house_style_is_used_more_than_it_is_bypassed():
    """133 to 4 when this was filed. The direction is the whole point."""
    raw = sum(len(RAW_FIGURE.findall(text)) for _p, text in _files())
    styled = sum(len(STYLE_USE.findall(text)) for _p, text in _files())
    assert styled > raw, f"{styled} style uses against {raw} raw figures"


def test_no_module_writes_rcParams_globally():
    offenders = {}
    for path, text in _files():
        if path.name in MAY_SET_RCPARAMS:
            continue
        for line in text.splitlines():
            if GLOBAL_RC.search(line) and "rcParams[" in line and "=" not in line.split("rcParams[")[1][:40]:
                continue        # a READ, e.g. plt.rcParams["axes.prop_cycle"]
            if GLOBAL_RC.search(line) and (".update(" in line or "] =" in line):
                offenders.setdefault(path.name, []).append(line.strip()[:70])
    assert not offenders, (
        "these write matplotlib's PROCESS-WIDE state, so every later figure "
        "in the process inherits it -- including one saved for paper. Use "
        f"`matplotlib.rc_context` instead:\n{offenders}")


@pytest.mark.parametrize("black", [True, False])
def test_setup_plot_leaves_no_colour_behind_it(black):
    """The one that was leaking, held directly."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spacr.utils import setup_plot

    watched = ("figure.facecolor", "axes.facecolor", "text.color",
               "xtick.color", "ytick.color", "axes.labelcolor",
               "axes.edgecolor")
    before = {key: plt.rcParams[key] for key in watched}

    figure, axes = setup_plot(4, black_background=black)
    try:
        assert {key: plt.rcParams[key] for key in watched} == before
        # And the figure itself still carries the theme, which is the point
        # of the function.
        assert axes.get_facecolor() is not None
    finally:
        plt.close(figure)
