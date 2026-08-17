"""The old volcano is off by default -- ALL of it, not one of three.

Asked for on 2026-08-16: "your new volcano plot is much much faster than my
old one so hide my old version behid a boolean that defaults to off". Fixed
once, and reported still happening on 2026-08-17: "by default i still see my
old volcano plot, i thought we said its generation would be hidden behind a
boolean which defaults is off".

THE FIRST FIX GATED ONE CALL SITE OF THREE, and the one it gated is the one a
Toxoplasma screen never reaches. `settings['toxo']` defaults to TRUE, so
`toxo.custom_volcano_plot` is the picture the maintainer was still being
shown after being told it was hidden.

The count is asserted here so a fourth path cannot be added ungated -- that
is exactly how this was half-fixed, by nobody having counted them.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

import numpy as np
import pandas as pd
import pytest


def _ml_source() -> str:
    import spacr.ml

    return pathlib.Path(spacr.ml.__file__).read_text()


# --------------------------------------------------------------------------- #
#  Every path is gated, and the count is pinned
# --------------------------------------------------------------------------- #

def test_every_old_volcano_call_is_gated():
    """Counts the call sites so a fourth cannot appear ungated.

    The three are: `plot.volcano_plot` under `plot and legacy_volcano`, the
    three `custom_volcano_plot` calls under the toxo block, and the plain
    fallback for a toxo=False run.
    """
    source = _ml_source()

    drawing = [line for line in source.splitlines()
               if "custom_volcano_plot(" in line and "def " not in line
               and "import" not in line]
    assert len(drawing) == 3, (
        f"expected three custom_volcano_plot call sites, found "
        f"{len(drawing)}: {drawing}")

    # Each is handed the gate rather than deciding for itself.
    assert source.count("draw=draw_legacy_volcano") == 3, (
        "a custom_volcano_plot call does not take the legacy gate")


def test_the_gate_is_resolved_once():
    """Three branches reading `settings.get('legacy_volcano')` separately is
    three chances to disagree."""
    source = _ml_source()

    assert source.count("draw_legacy_volcano = bool(") == 1
    assert "legacy_volcano" in source


def test_the_plain_fallback_is_gated_too():
    source = _ml_source()

    assert "not settings.get('toxo') and draw_legacy_volcano" in source


def test_the_default_is_off():
    """`.get(..., False)` -- a run that says nothing draws nothing old."""
    source = _ml_source()

    assert "settings.get('legacy_volcano', False)" in source


# --------------------------------------------------------------------------- #
#  Gating the DRAWING, not the block
# --------------------------------------------------------------------------- #

def test_the_toxo_block_still_runs():
    """It also produces the gene list the GT1 phenotype plot and the ME49
    transcription heatmap are built from. Gating the CALL would have removed
    two reports nobody asked to lose."""
    source = _ml_source()

    assert "if settings['toxo']:" in source, (
        "the toxo block is gated on the legacy volcano, which takes the "
        "phenotype and heatmap reports with it")


def test_not_drawing_returns_EXACTLY_what_drawing_returns():
    """The whole reason the toxo block can keep running.

    Compared against the DRAWN path rather than against a list written here:
    the claim being made is "same rule", and a hand-written expectation only
    tests that I can restate the rule, not that the two agree. A first
    version of this test asserted the `variable` values it put IN, and the
    merge rewrites that column to the gene number -- so it failed while both
    paths were already correct.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spacr.toxo import custom_volcano_plot

    data = pd.DataFrame({
        "feature": [f"gene_fraction:gene[{i}]"
                    for i in (292020, 306060, 253750, 244480)],
        "variable": ["a", "b", "c", "d"],
        "coefficient": [0.9, 0.01, -0.8, 0.6],
        "p_value": [0.001, 0.9, 0.002, 0.4],
    })
    meta = pd.DataFrame({
        "gene_nr": [292020, 306060, 253750, 244480],
        "tagm_location": ["micronemes", "cytosol", "ER 1", "apicoplast"]})

    quiet = custom_volcano_plot(data, meta, threshold=0.5, draw=False)
    drawn = custom_volcano_plot(data, meta, threshold=0.5, draw=True)
    plt.close("all")

    assert quiet == drawn, (quiet, drawn)
    # And it is a real selection, not "everything" or "nothing".
    assert 0 < len(quiet) < 4, quiet


def test_not_drawing_opens_no_figure():
    """"much much faster than my old one" -- skipping the picture has to
    actually skip it, not build one and throw it away."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spacr.toxo import custom_volcano_plot

    data = pd.DataFrame({
        "feature": [f"gene_fraction:gene[{i}]" for i in (292020, 306060)],
        "variable": ["a", "b"],
        "coefficient": [0.9, 0.01],
        "p_value": [0.001, 0.9]})
    meta = pd.DataFrame({"gene_nr": [292020, 306060],
                         "tagm_location": ["micronemes", "cytosol"]})

    plt.close("all")
    before = len(plt.get_fignums())
    custom_volcano_plot(data, meta, threshold=0.5, draw=False)

    assert len(plt.get_fignums()) == before


def test_the_hit_rule_is_the_one_the_picture_uses():
    """A hit list that disagreed with the volcano would be the worst of both:
    the phenotype plot reporting genes the volcano does not mark."""
    source = pathlib.Path(
        __import__("spacr.toxo", fromlist=["toxo"]).__file__).read_text()

    # The scatter loop's rule, and the vectorised one, are the same predicate.
    assert "(row['p_value'] <= 0.05) and (abs(row['coefficient']) >= abs(threshold))" in source
    assert "merged_data['p_value'] <= 0.05" in source
    assert "merged_data['coefficient'].abs() >= abs(threshold)" in source


def test_nothing_is_claimed_when_nothing_was_drawn():
    """A stale file from an EARLIER run sits at that exact path, so an
    existence check would announce a figure this run did not make."""
    source = _ml_source()

    assert "if not draw_legacy_volcano:" in source
