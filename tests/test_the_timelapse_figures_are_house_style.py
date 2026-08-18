"""``spacr.timelapse`` draws its eleven figures in the house style.

Every panel in this module compares the same two things -- infected cells and
uninfected cells -- and every one of them drew that comparison as ``"red"``
against ``"green"``, at equal weight, with a thick black threshold line over
the top and a rounded white box holding the numbers.

Three separate failures in one pair of colours:

1. **Both conditions were the claim.** The house style spends colour on the
   argument and greys everything else; a control is the ground, not a second
   headline. Infected takes the highlight blue; uninfected takes the dark grey
   every control in the published figures takes.
2. **Red against green is the one pair a colour-blind reader cannot split.**
3. **A 2 pt solid black line is heavier than the distributions it separates.**
   A threshold is a reference, and a reference is thin, dashed and grey.

These tests DRAW each figure on a small synthetic input and read the artists
back. The velocities, the histogram counts and the per-well file names are
asserted too: this is a restyle, so if a number moved the change is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from spacr import timelapse as TL  # noqa: E402
from spacr.figures.style import (ROLES, TYPE_SCALE, Palette,  # noqa: E402
                                 resolve_ink, theme_target)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


#: What every one of these panels used to draw the comparison with.
OLD_CONDITION_HUES = ("red", "green")


@pytest.fixture
def saved_figures(monkeypatch):
    """Every figure this module hands to ``save_figure_to_path``, kept open.

    These functions close their figures as soon as they are written -- which
    is right, because a tracking run writes one per well and pyplot would
    otherwise hold all of them. So the figure is grabbed on its way to disk
    rather than picked off ``plt.get_fignums()`` afterwards.
    """
    kept = []
    real = TL.save_figure_to_path

    def spy(fig, path, *args, **kwargs):
        kept.append(fig)
        return real(fig, path, *args, **kwargs)

    monkeypatch.setattr(TL, "save_figure_to_path", spy)
    return kept

def _track(plate, well, field, cell, infected, xs, ys):
    return {
        "plateID": plate, "wellID": well, "fieldID": field, "cellID": cell,
        "infected": bool(infected),
        "x_px": np.asarray(xs, dtype=float),
        "y_px": np.asarray(ys, dtype=float),
        "v_px_per_frame": 1.0, "straightness": 0.5,
    }


def _motility_inputs():
    """Two wells: A01 mixed infected/uninfected, A02 infected-only.

    The same shape ``tests/test_cov_timelapse_corr_plots.py`` uses, so the
    velocities below are the ones that file already pins.
    """
    per_well = {
        ("p1", "A01"): [
            _track("p1", "A01", "f1", 1, True, [0.0, 4.0, 8.0], [0.0, 4.0, 12.0]),
            _track("p1", "A01", "f1", 2, False, [20.0, 24.0], [20.0, 28.0]),
        ],
        ("p1", "A02"): [
            _track("p1", "A02", "f1", 3, True, [1.0, 5.0], [1.0, 9.0]),
            _track("p1", "A02", "f1", 4, True, [2.0, 10.0], [2.0, 6.0]),
        ],
    }
    track_df = pd.DataFrame({
        "plateID": ["p1"] * 4,
        "wellID": ["A01", "A01", "A02", "A02"],
        "fieldID": ["f1"] * 4,
        "cellID": [1, 2, 3, 4],
        "infected": [True, False, True, True],
        "velocity": [1.0, 3.0, 5.0, 7.0],
    })
    well_summary_df = pd.DataFrame({
        "plateID": ["p1", "p1"],
        "wellID": ["A01", "A02"],
        "mean_velocity_infected": [1.0, 6.0],
        "mean_velocity_uninfected": [3.0, np.nan],
    })
    return track_df, per_well, well_summary_df


# --------------------------------------------------------------------------- #
#  The condition vocabulary itself.
# --------------------------------------------------------------------------- #

def test_the_two_conditions_are_a_highlight_and_a_control_grey():
    """Assign once and never re-map: one pair, used by all eleven figures."""
    assert TL.INFECTED_COLOUR == ROLES["highlight"]
    assert TL.UNINFECTED_COLOUR == Palette.GREY_DARK
    # The control must not be a second headline.
    assert TL.UNINFECTED_COLOUR != ROLES["highlight"]
    for gone in OLD_CONDITION_HUES:
        assert to_hex(TL.INFECTED_COLOUR) != to_hex(gone)
        assert to_hex(TL.UNINFECTED_COLOUR) != to_hex(gone)


# --------------------------------------------------------------------------- #
#  The motility figures.
# --------------------------------------------------------------------------- #

def test_the_motility_tracks_are_blue_infected_against_grey(tmp_path, saved_figures):
    """Four tracks, three infected: the highlight is the minority it should be."""
    track_df, per_well, well_summary_df = _motility_inputs()
    TL._make_motility_plots(track_df, per_well, well_summary_df,
                            str(tmp_path / "motility"),
                            pixels_per_um=4.0, seconds_per_frame=30.0,
                            vel_unit="µm/min", settings={})

    combined = saved_figures[0].axes[0]
    colours = sorted(to_hex(line.get_color()) for line in combined.lines)
    assert colours == sorted([to_hex(TL.INFECTED_COLOUR)] * 3
                             + [to_hex(TL.UNINFECTED_COLOUR)])
    for gone in OLD_CONDITION_HUES:
        assert to_hex(gone) not in colours


def test_the_motility_note_lost_its_white_box(tmp_path, saved_figures):
    """A rounded white panel with a black edge is furniture, and on the dark
    theme it is a white rectangle laid over the tracks.

    The patch itself stays, invisible: the four text lines are positioned
    against its corner, so deleting it would move the note.
    """
    track_df, per_well, well_summary_df = _motility_inputs()
    TL._make_motility_plots(track_df, per_well, well_summary_df,
                            str(tmp_path / "motility"),
                            pixels_per_um=4.0, seconds_per_frame=30.0,
                            vel_unit="µm/min", settings={})

    combined = saved_figures[0].axes[0]
    assert len(combined.patches) == 1, "the spacer patch is gone"
    box = combined.patches[0]
    assert box.get_facecolor()[3] == 0.0, "the white panel is back"
    assert box.get_edgecolor()[3] == 0.0, "the black edge is back"


def test_the_motility_note_takes_the_condition_colours_and_the_theme_ink(
        tmp_path, saved_figures):
    """The two condition lines are coloured; the unit lines are ink."""
    track_df, per_well, well_summary_df = _motility_inputs()
    TL._make_motility_plots(track_df, per_well, well_summary_df,
                            str(tmp_path / "motility"),
                            pixels_per_um=4.0, seconds_per_frame=30.0,
                            vel_unit="µm/min", settings={})

    combined = saved_figures[0].axes[0]
    by_text = {t.get_text(): to_hex(t.get_color()) for t in combined.texts}
    infected = next(k for k in by_text if k.startswith("Infected"))
    uninfected = next(k for k in by_text if k.startswith("Uninfected"))
    assert by_text[infected] == to_hex(TL.INFECTED_COLOUR)
    assert by_text[uninfected] == to_hex(TL.UNINFECTED_COLOUR)

    ink = to_hex(resolve_ink(theme_target()))
    units = [v for k, v in by_text.items() if k not in (infected, uninfected)]
    assert units and all(colour == ink for colour in units)


def test_the_motility_velocities_did_not_move(tmp_path, saved_figures):
    """The restyle touched the ink, not the arithmetic.

    Infected mean over the whole plate is (1 + 5 + 7) / 3; uninfected is 3.
    """
    track_df, per_well, well_summary_df = _motility_inputs()
    out = tmp_path / "motility"
    TL._make_motility_plots(track_df, per_well, well_summary_df, str(out),
                            pixels_per_um=4.0, seconds_per_frame=30.0,
                            vel_unit="µm/min", settings={})

    combined = saved_figures[0].axes[0]
    texts = [t.get_text() for t in combined.texts]
    assert any("4.33" in text for text in texts), texts
    assert any(text.startswith("Uninfected (3.00") for text in texts), texts
    # The extension follows the figure-format preference, not the .png the
    # caller proposes -- every kept figure goes through `save_figure`.
    assert sorted(p.stem for p in out.iterdir()) == [
        "motility_all_tracks",
        "motility_p1_A01_all_tracks",
        "motility_p1_A01_infected_origin",
        "motility_p1_A01_uninfected_origin",
        "motility_p1_A02_all_tracks",
        "motility_p1_A02_infected_origin",
    ]
    assert len({p.suffix for p in out.iterdir()}) == 1


# --------------------------------------------------------------------------- #
#  The intensity QC histogram.
# --------------------------------------------------------------------------- #

def _qc_frame(n_cells=30, frames=2, chan=2, seed=0):
    """A per-frame per-cell table with a clean infected/uninfected split.

    The shape ``_infection_qc_histogram`` parses: uninfected cells around
    100-200, infected around 5000-9000, so the threshold search has an
    obvious answer and the two histograms do not overlap.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_cells):
        infected = index >= n_cells // 2
        base = (5000.0 + 4000.0 * rng.random() if infected
                else 100.0 + 80.0 * rng.random())
        for frame in range(frames):
            rows.append({
                "plateID": "plate1", "wellID": "A01", "fieldID": 1,
                "cellID": index + 1, "frame": frame,
                "infected": bool(infected),
                f"cell_p95_intensity_ch{chan}": float(base + rng.random()),
            })
    return pd.DataFrame(rows)


def test_the_qc_histogram_greys_the_control_and_dashes_the_threshold(
        tmp_path, saved_figures):
    """Two hues at equal weight and a 2 pt black line became one claim."""
    TL._infection_qc_histogram(_qc_frame(), {}, "infected", 2, str(tmp_path))

    axis = saved_figures[-1].axes[0]
    fills = {to_hex(patch.get_facecolor()) for patch in axis.patches}
    assert to_hex(TL.INFECTED_COLOUR) in fills
    assert to_hex(TL.UNINFECTED_COLOUR) in fills
    for gone in OLD_CONDITION_HUES:
        assert to_hex(gone) not in fills

    threshold = [line for line in axis.lines if line.get_linestyle() != "-"]
    assert threshold, "the threshold did not draw"
    assert to_hex(threshold[0].get_color()) == to_hex(ROLES["reference"])
    assert threshold[0].get_linewidth() <= 1.0


def test_the_qc_histogram_still_counts_every_cell(tmp_path, saved_figures):
    """A restyle must not drop a count: 30 cells, one measurement each."""
    TL._infection_qc_histogram(_qc_frame(n_cells=30), {}, "infected", 2,
                               str(tmp_path))
    axis = saved_figures[-1].axes[0]
    assert sum(patch.get_height() for patch in axis.patches) == pytest.approx(30)


# --------------------------------------------------------------------------- #
#  The three-panel QC figure layout.
# --------------------------------------------------------------------------- #

def test_the_results_figure_axes_are_built_inside_the_style():
    """rcParams reach an artist at construction, so the AXES have to be
    created inside the context and not merely the Figure."""
    figure, ax_pca, ax_xgb, ax_hist = TL.create_results_figure()
    ink = to_hex(resolve_ink(theme_target()))
    for axis in (ax_pca, ax_xgb, ax_hist):
        assert to_hex(axis.spines["left"].get_edgecolor()) == ink
        assert to_hex(axis.xaxis.label.get_color()) == ink
        assert not axis.spines["top"].get_visible()
        assert not axis.spines["right"].get_visible()
        assert not axis.xaxis.get_gridlines()[0].get_visible()
        assert axis.xaxis.label.get_fontsize() == pytest.approx(
            TYPE_SCALE["label"])


# --------------------------------------------------------------------------- #
#  Rule 2, across everything reachable here.
# --------------------------------------------------------------------------- #

def test_no_timelapse_figure_leaves_its_style_on_the_globals(tmp_path, saved_figures):
    """One global write here restyles every later figure in the session."""
    before = dict(matplotlib.rcParams)

    TL.create_results_figure()
    track_df, per_well, well_summary_df = _motility_inputs()
    TL._make_motility_plots(track_df, per_well, well_summary_df,
                            str(tmp_path / "motility"),
                            pixels_per_um=4.0, seconds_per_frame=30.0,
                            vel_unit="µm/min", settings={})
    TL._infection_qc_histogram(_qc_frame(), {}, "infected", 2,
                               str(tmp_path / "qc"))

    after = dict(matplotlib.rcParams)
    changed = [key for key in before
               if repr(before[key]) != repr(after.get(key))]
    assert not changed, f"spacr.timelapse leaked rcParams: {sorted(changed)}"
