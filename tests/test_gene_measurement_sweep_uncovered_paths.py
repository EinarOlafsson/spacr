"""What the sweep does when something is missing, empty, or refuses to move.

Every branch here is one the module walks when the screen, the theme store or
the export palette is not what the happy path assumes: a well that belongs to
no plate, a filter that excludes nobody, a volcano with no survivor on it, a
figure written while the export appearance cannot be read. Each of them still
ends in a table or a picture somebody will believe, so each is asserted on
what came out -- a value, a colour, a pixel, a message that was or was not
printed -- rather than on what was called.
"""
from __future__ import annotations

import io
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from spacr.gene_measurement_sweep import (SweepResult, _readable,
                                          _residualise, _write,
                                          plot_grid_volcano,
                                          plot_guide_concordance, sweep)


def _table(**columns) -> pd.DataFrame:
    """A sweep table with every column the pictures read, defaults filled."""
    n = len(next(iter(columns.values())))
    base = {"level": ["guide"] * n, "guide": [f"g{i}" for i in range(n)],
            "measurement": [f"cell_area_{i}" for i in range(n)],
            "effect": [0.5] * n, "p": [0.001] * n, "q": [0.001] * n,
            "circularity": [0.05] * n, "n_wells": [40] * n,
            "effective_wells": [35.0] * n, "share": [0.2] * n,
            "ubiquitous": [False] * n, "control": [False] * n}
    base.update(columns)
    return pd.DataFrame(base)


@pytest.fixture()
def screen():
    """24 wells on one plate, two sparse guides and one real effect."""
    rng = np.random.default_rng(11)
    n = 24
    index = [f"plate1_r{i}_c1" for i in range(n)]
    driver = np.zeros(n)
    driver[rng.choice(n, 10, replace=False)] = rng.uniform(0.3, 0.6, 10)
    quiet = np.zeros(n)
    quiet[rng.choice(n, 8, replace=False)] = rng.uniform(0.05, 0.1, 8)
    wells = pd.DataFrame({
        "cell_area": driver * 7.0 + rng.normal(0, 0.2, n),
        "pathogen_area": rng.normal(0, 1, n),
    }, index=index)
    fractions = pd.DataFrame({"TGGT1_111_1": driver, "TGGT1_222_1": quiet},
                             index=index)
    return wells, fractions


# --------------------------------------------------------------------------- #
#  The table
# --------------------------------------------------------------------------- #

def test_a_circularity_bar_removes_the_pairs_the_score_already_tracks():
    """`survivors` filters on circularity once the score really joined.

    The bar is only allowed to be applied when the column holds numbers, and
    when it is, a surviving pair whose measurement the classifier already
    tracks must not be handed back as independent corroboration.
    """
    table = _table(guide=["clean", "circular", "weak"],
                   q=[0.001, 0.001, 0.40],
                   circularity=[0.02, 0.91, 0.01])
    result = SweepResult(table=table, effects=pd.DataFrame(), n_wells=40,
                         n_blocks=2, circularity_known=True)

    assert list(result.survivors(alpha=0.05)["guide"]) == ["clean", "circular"]
    kept = result.survivors(alpha=0.05, max_circularity=0.15)
    assert list(kept["guide"]) == ["clean"]


def test_a_well_that_belongs_to_no_plate_keeps_its_own_value():
    """A block label that matches no row is skipped, silently and safely.

    `np.unique` keeps every NaN it is given and NaN equals nothing, so such a
    label selects no row at all. Centring on the mean of an empty selection
    would warn about an empty slice and mean nothing; the wells carrying the
    NaN block must simply come back as they went in, while the wells that do
    have a plate are still centred on it.
    """
    blocks = np.array([1.0, 1.0, np.nan, np.nan])
    matrix = np.array([[10.0], [20.0], [7.0], [9.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = _residualise(matrix, blocks)

    assert list(out.ravel()) == [-5.0, 5.0, 7.0, 9.0]


def test_a_both_level_sweep_of_unnamed_guides_reports_only_guide_rows():
    """Guides that name no gene produce no gene rows at ``level='both'``.

    `gene_fractions` leaves out a guide it cannot read a gene from rather than
    pooling it into an "unknown" gene, so a library of blank guide names has
    no gene side at all. The sweep must then hand back the guide rows under
    their own names instead of an empty table or a phantom gene.
    """
    rng = np.random.default_rng(5)
    n = 20
    index = [f"plate1_r{i}_c1" for i in range(n)]
    driver = np.zeros(n)
    driver[rng.choice(n, 9, replace=False)] = rng.uniform(0.3, 0.6, 9)
    wells = pd.DataFrame({
        "cell_area": driver * 6.0 + rng.normal(0, 0.2, n),
        "pathogen_area": rng.normal(0, 1, n),
    }, index=index)
    fractions = pd.DataFrame({"": driver, "  ": np.roll(driver, 3)},
                             index=index)

    result = sweep(wells, fractions, level="both")

    assert set(result.table["level"]) == {"guide"}
    assert sorted(result.effects.index) == ["", "  "]
    assert not any("(gene)" in str(g) for g in result.table["guide"])
    assert set(result.table["measurement"]) == {"cell_area", "pathogen_area"}


@pytest.mark.parametrize("kwargs, why", [
    ({"drop_guides": ["TGGT1_999_1", "999"]}, "named"),
    ({"max_wells_fraction": 1.0}, "in too many wells"),
    ({"max_share": 1.0}, "too large a share"),
])
def test_a_guide_filter_that_excludes_nobody_keeps_everyone_and_says_nothing(
        screen, capsys, kwargs, why):
    """A filter matching no guide leaves the screen and the log untouched.

    Every exclusion is announced, because a sweep that quietly dropped the
    gene somebody was looking for sends them hunting for a row that was never
    computed. The other half of that promise is that a filter which excluded
    nobody must not announce an exclusion of nobody either.
    """
    wells, fractions = screen

    result = sweep(wells, fractions, **kwargs)

    assert set(result.table["guide"]) == {"TGGT1_111_1", "TGGT1_222_1"}
    assert why not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  Saving a figure when the export palette will not answer
# --------------------------------------------------------------------------- #

def _swatch(figure) -> tuple:
    """Write ``figure`` through `_write` and return (bytes, corner pixel)."""
    from PIL import Image

    buffer = io.BytesIO()
    _write(figure, buffer)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    return buffer.getvalue(), image.getpixel((0, 0))


def test_a_figure_still_saves_when_the_export_appearance_cannot_be_read(
        tmp_path, monkeypatch):
    """An unreadable export palette costs the figure its repaint, not its file.

    `saved_figure_appearance` reaches for the preference store, which is not
    there in a headless render. The write must fall through to the figure's
    own colours and still produce a file, because the alternative is a run
    that computed everything and saved nothing.
    """
    import spacr.figure_style as figure_style

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("no preference store")

    monkeypatch.setattr(figure_style, "saved_figure_appearance", unavailable)

    figure, axes = plt.subplots()
    figure.patch.set_facecolor("#204060")
    figure.patch.set_alpha(1.0)
    axes.set_title("kept")
    axes.title.set_color("#FFFFFF")
    path = tmp_path / "unreadable.png"

    _write(figure, str(path))

    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert axes.title.get_color() == "#FFFFFF"
    assert to_hex(figure.get_facecolor()) == "#204060"
    plt.close(figure)


def test_saving_in_screen_mode_writes_the_figures_own_ground(monkeypatch):
    """Screen mode repaints nothing, so the file carries the screen colours.

    ``screen`` is the mode for a figure that stays in the application, and it
    is the one appearance that must leave the ground alone: the pixels that
    land in the file are the ones the user was already looking at.
    """
    monkeypatch.setenv("SPACR_FIGURE_SAVE_MODE", "screen")

    figure, axes = plt.subplots()
    figure.patch.set_facecolor("#204060")
    figure.patch.set_alpha(1.0)
    axes.set_title("screen")
    axes.title.set_color("#FFFFFF")

    data, corner = _swatch(figure)

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert corner == (32, 64, 96)
    assert axes.title.get_color() == "#FFFFFF"
    plt.close(figure)


def test_a_transparent_save_recolours_the_chrome_but_not_the_ground(
        monkeypatch):
    """Transparent mode has no ground to paint, so only the chrome moves.

    ``transparent`` means the page is whatever the figure is pasted onto, so
    the figure's own ground must survive into the file untouched. The chrome
    still has to be made legible for that unknown page -- and put back
    afterwards, because the caller is still holding the figure to display.
    """
    from spacr.figure_style import saved_figure_appearance

    monkeypatch.setenv("SPACR_FIGURE_SAVE_MODE", "transparent")
    export_ink = saved_figure_appearance().ink
    assert export_ink and export_ink != "#FFFFFF"

    figure, axes = plt.subplots()
    figure.patch.set_facecolor("#204060")
    figure.patch.set_alpha(1.0)
    axes.set_title("transparent")
    axes.title.set_color("#FFFFFF")

    class Watcher(io.BytesIO):
        """Records the title colour at the moment the file is written."""

        during = None

        def write(self, chunk):
            if Watcher.during is None:
                Watcher.during = axes.title.get_color()
            return super().write(chunk)

    buffer = Watcher()
    _write(figure, buffer)

    from PIL import Image

    buffer.seek(0)
    assert Image.open(buffer).convert("RGB").getpixel((0, 0)) == (32, 64, 96)
    assert Watcher.during == export_ink
    assert axes.title.get_color() == "#FFFFFF"
    plt.close(figure)


# --------------------------------------------------------------------------- #
#  Styling axes the theme cannot reach
# --------------------------------------------------------------------------- #

def test_the_reference_ink_is_used_when_the_theme_will_not_resolve():
    """A theme that cannot be resolved falls back to the reference grey.

    The house reference colour is legible on both of spaCR's grounds, which is
    why it is the fallback: ink that is slightly wrong beats ink that is
    invisible, and beats a traceback out of a plotting call entirely.
    """
    from spacr.figures.style import ROLES

    figure, axes = plt.subplots()
    axes.set_title("unresolvable")

    import spacr.figures.style as style

    original = style.resolve_ink

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("no theme store")

    style.resolve_ink = unavailable
    try:
        ink = _readable(figure, axes)
    finally:
        style.resolve_ink = original

    assert ink == ROLES["reference"]
    assert axes.title.get_color() == ROLES["reference"]
    assert axes.spines["top"].get_visible() is False
    plt.close(figure)


def test_a_figure_ground_that_refuses_to_change_does_not_stop_the_axes():
    """A figure patch that will not go transparent still leaves styled axes.

    The transparent ground is a preference; the readable axes are the point.
    One must not cost the other, so the axes are styled even when the figure
    patch rejects the alpha it is handed.
    """
    figure, axes = plt.subplots()
    axes.set_title("styled anyway")

    def refuse(*_args, **_kwargs):
        raise RuntimeError("this patch is locked")

    figure.patch.set_alpha = refuse

    ink = _readable(figure, axes)

    assert axes.title.get_color() == ink
    assert axes.patch.get_alpha() == 0.0
    plt.close(figure)


def test_one_unstylable_axes_does_not_cost_the_next_one_its_ink():
    """A failure on one axes stops that axes only.

    A panel figure that lost every label because its first axes objected would
    be worse than one unstyled panel, so the loop moves on to the next axes
    rather than abandoning the figure.
    """
    figure, (first, second) = plt.subplots(2)
    first.set_title("refuses")
    first.title.set_color("#112233")
    second.set_title("styled")

    def refuse(*_args, **_kwargs):
        raise RuntimeError("this patch is locked")

    first.patch.set_alpha = refuse

    ink = _readable(figure, first, second)

    assert first.title.get_color() == "#112233"
    assert second.title.get_color() == ink
    assert second.patch.get_alpha() == 0.0
    plt.close(figure)


# --------------------------------------------------------------------------- #
#  The pictures, drawn on tables that hold less than the usual
# --------------------------------------------------------------------------- #

def test_guide_concordance_reads_a_table_that_carries_no_level_column():
    """A table from before the level column still draws its concordance.

    The picture asks whether a gene's own guides agree, which needs guide rows
    and nothing else. A table without a ``level`` column has only guide rows
    by construction, so it must be drawn rather than refused.
    """
    table = pd.DataFrame({
        "guide": ["TGGT1_111_1", "TGGT1_111_2"] * 2,
        "measurement": ["cell_area"] * 2 + ["nucleus_area"] * 2,
        "effect": [0.5, 0.6, -0.4, -0.3],
        "q": [0.001] * 4,
    })
    result = SweepResult(table=table, effects=pd.DataFrame(), n_wells=40,
                         n_blocks=1)

    figure = plot_guide_concordance(result)

    assert figure is not None
    axes = figure.axes[0]
    assert [t.get_text() for t in axes.get_yticklabels()] == ["111  (2)"]
    # Both measurements agreed on their sign, so the gene sits at 1.0.
    assert axes.collections[0].get_offsets()[:, 0].tolist() == [1.0, 1.0]
    plt.close(figure)


def test_the_volcano_rings_nothing_when_no_survivor_is_circular():
    """Known circularity with nothing above the bar draws no rings.

    The ring says "the score already tracks this measurement". Drawing it on a
    screen where nothing is tracked would say the opposite of what the data
    holds, so the extra layer is only added when there is something to mark --
    while the key still names the ring, because circularity WAS computed.
    """
    table = _table(q=[0.001, 0.002, 0.003], p=[1e-6, 1e-5, 1e-4],
                   effect=[0.6, -0.5, 0.4], circularity=[0.02, 0.03, 0.01])
    result = SweepResult(table=table, effects=pd.DataFrame(), n_wells=40,
                         n_blocks=1, circularity_known=True)

    figure = plot_grid_volcano(result)

    axes = figure.axes[0]
    assert len(axes.collections) == 3
    notes = [t.get_text() for t in axes.texts]
    assert "ringed: the score already tracks it" in notes
    assert "raises it (n=2)" in notes and "lowers it (n=1)" in notes
    plt.close(figure)


def test_the_volcano_draws_no_threshold_and_no_labels_when_nothing_passes():
    """A screen with no survivor gets no correction line and no gene names.

    The dotted line is drawn where the correction actually landed, and there
    is no such place when nothing cleared it; a line at the nominal alpha
    would claim a threshold the data never reached. The labels go the same
    way -- naming the six best q values of a screen with no hit in it reads as
    a hit list.
    """
    table = _table(q=[0.6, 0.7, 0.8], p=[0.3, 0.4, 0.5],
                   effect=[0.10, -0.08, 0.05])
    result = SweepResult(table=table, effects=pd.DataFrame(), n_wells=40,
                         n_blocks=1)

    figure = plot_grid_volcano(result, alpha=0.05)

    axes = figure.axes[0]
    assert len(axes.lines) == 0
    notes = [t.get_text() for t in axes.texts]
    assert notes == ["not significant", "raises it (n=0)", "lowers it (n=0)",
                     "circularity NOT computed"]
    assert axes.get_title() == "3 gene x measurement pair(s)"
    plt.close(figure)
