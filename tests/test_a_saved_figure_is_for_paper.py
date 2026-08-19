"""A figure saved from a dark session is dark ink on a light page.

Instruction 150, reported 2026-08-18: "when a graph is saved and the user is in
dark mode white elements are changed to black for saving (text lines, etc)".

THE MECHANISM. `spacr.qt.preferences.get_figure_colors()` is the one source of
figure colour and both renderers read it, which is why a theme switch moves
both -- and why in dark mode the axes, ticks, tick labels, axis labels, title,
legend and annotation lines are all WHITE. Nothing inverted them at export
time, so the file was a blank rectangle with some coloured dots in it. A PNG
saved with a transparent ground even looked right in a dark file manager and
disappeared when it was pasted into a manuscript, which means the user found
out at the point of writing the paper.

THE HALF THAT IS EASY TO GET WRONG IS THE HALF THAT MUST NOT MOVE. A blanket
white-to-black would turn a white data point black, and on a volcano black is
the colour of "not a hit" -- the flip would change what the figure SAYS. So
every assertion below about the chrome has a partner asserting the data came
through untouched, measured on the written pixels rather than on the artists.

WHY rcParams ALONE CANNOT DO IT, since the instruction points at the
`rc_context` that already scopes `pdf.fonttype`: rcParams are read when an
artist is CREATED, and by the time `save_figure` runs the figure is drawn.
Only `savefig.facecolor`, `savefig.edgecolor` and `savefig.transparent` are
read at write time. The chrome is repainted artist by artist and put back --
hence `test_the_figure_on_screen_is_unchanged_by_the_save`, which is the
acceptance line "the plot on screen is byte-identical before and after".
"""
from __future__ import annotations

import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr import figure_style as FS  # noqa: E402
from spacr import plot as P  # noqa: E402

DARK_GROUND, DARK_INK = "#1E1E1E", "#FFFFFF"
UP, DOWN, GREY = "#D55E00", "#0072B2", "#B8BDC5"


@pytest.fixture(autouse=True)
def _no_mode_from_the_environment(monkeypatch):
    monkeypatch.delenv("SPACR_FIGURE_SAVE_MODE", raising=False)
    yield
    plt.close("all")


def _dark_volcano():
    """A volcano as a dark-themed session draws one: white furniture, and
    three data colours that carry the claim."""
    fig, ax = plt.subplots(figsize=(4, 3), facecolor=DARK_GROUND)
    ax.set_facecolor(DARK_GROUND)
    rng = np.random.default_rng(0)
    ax.scatter(rng.normal(size=200), rng.random(200) * 5, s=18, color=GREY)
    ax.scatter([2.4, -2.6], [4.6, 4.4], s=40, color=[UP, DOWN])
    ax.axhline(1.3, color=DARK_INK, linestyle="--")   # the significance line
    ax.axvline(0, color=DARK_INK, linewidth=0.7)      # the zero line
    ax.set_title("volcano", color=DARK_INK)
    ax.set_xlabel("effect", color=DARK_INK)
    ax.set_ylabel("-log10 q", color=DARK_INK)
    ax.tick_params(colors=DARK_INK)
    for spine in ax.spines.values():
        spine.set_edgecolor(DARK_INK)
    ax.annotate("GRA14", (2.4, 4.6), color=DARK_INK, xytext=(1.0, 3.0),
                arrowprops=dict(arrowstyle="->", color=DARK_INK))
    ax.legend(["ns", "hit"], facecolor=DARK_GROUND, edgecolor=DARK_INK,
              labelcolor=DARK_INK)
    ax.grid(True, color="#3A3A3A")
    return fig, ax


def _fingerprint(fig):
    """Every colour on the figure's furniture, as a comparable tuple."""
    out = []
    for kind, artist, getter, _setter in P._chrome(fig):
        try:
            out.append((kind, type(artist).__name__,
                        tuple(np.ravel(getter()))))
        except Exception:                                        # noqa: BLE001
            pass
    return out


def _pixels(path):
    import imageio.v2 as imageio

    return imageio.imread(path)


def _carries(image, hexcolour, tolerance=6):
    rgb = np.array([int(hexcolour[i:i + 2], 16) for i in (1, 3, 5)])
    flat = image[..., :3].reshape(-1, 3).astype(int)
    return bool((np.abs(flat - rgb).sum(axis=1) <= tolerance).any())


def _save(fig, tmp_path, name="v.png", **kwargs):
    return P.save_figure(fig, os.path.join(str(tmp_path), name),
                         fmt="png", dpi=100, **kwargs)


# ---------------------------------------------------------------- the decision


def test_the_decision_is_one_shared_function():
    """Both renderers have to reach the same answer, so the answer is not in
    either of them. `spacr.figure_style` imports neither matplotlib nor Qt at
    module level -- its own docstring promises that, and the pyqtgraph
    exporter is the caller who will need it to stay true."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(FS))
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level.append(node.module or "")
    heavy = [name for name in top_level
             if name.split(".")[0] in {"matplotlib", "PySide6", "PyQt5",
                                       "pyqtgraph", "seaborn"}]
    assert not heavy, f"figure_style imports {heavy} at module level"
    for mode in FS.SAVE_MODES:
        assert FS.saved_figure_appearance(mode).mode == mode


def test_the_three_states_are_three_different_jobs():
    printed = FS.saved_figure_appearance("print")
    screen = FS.saved_figure_appearance("screen")
    clear = FS.saved_figure_appearance("transparent")

    assert (printed.ground, printed.transparent, printed.flip) == (
        FS.PRINT_GROUND, False, True)
    # 'as on screen' is a no-op BY CONSTRUCTION, not by a branch somewhere.
    assert (screen.ground, screen.ink, screen.flip) == (None, None, False)
    # THE INK FOLLOWS THE THEME HERE, changed 2026-08-19 on the maintainer's
    # second request: "the lines should be white in dark mode and black in
    # light mode", then reported as a fault when they were not. Transparent
    # MEANS the ground is whatever the figure is pasted onto, and the only
    # thing that knows what that is, is the user -- who says so by their
    # theme. `print` is unchanged, so a manuscript figure is untouched.
    assert clear.transparent is True
    assert clear.ink in (FS.PRINT_INK, FS.DARK_INK)
    assert clear.ink == FS.theme_ink()[0]


def test_an_unreadable_preference_does_not_lose_the_figure():
    assert FS.saved_figure_appearance("chartreuse").mode == "print"


def test_the_environment_can_ask_for_a_mode_without_a_qt_store(monkeypatch):
    monkeypatch.setenv("SPACR_FIGURE_SAVE_MODE", "screen")
    assert FS.figure_save_mode() == "screen"
    monkeypatch.setenv("SPACR_FIGURE_SAVE_MODE", "nonsense")
    assert FS.figure_save_mode() == "print"


def test_contrast_says_nothing_rather_than_guessing():
    """None is not 1.0. A caller that read it as 'no contrast' would repaint
    every transparent artist in the figure."""
    assert FS.contrast_ratio("#FFFFFF", "#FFFFFF") == pytest.approx(1.0)
    assert FS.contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert FS.contrast_ratio("none", "#FFFFFF") is None
    assert FS.contrast_ratio((0.0, 0.0, 0.0, 0.0), "#FFFFFF") is None
    assert FS.is_legible_on("none", "#FFFFFF") is True


def test_the_data_floor_is_under_the_house_style_not_over_it():
    """A warning that fires on every figure is a warning nobody reads. The
    palest colour the house style deliberately puts on the page must stay
    quiet."""
    from spacr.figures.style import ROLES

    palest = min(FS.contrast_ratio(colour, FS.PRINT_GROUND)
                 for colour in ROLES.values()
                 if FS.contrast_ratio(colour, FS.PRINT_GROUND) is not None)
    assert FS.DATA_CONTRAST_FLOOR < palest


# ------------------------------------------------------------------- the save


def test_the_page_is_light_and_the_ink_is_dark(tmp_path):
    fig, _ax = _dark_volcano()
    image = _pixels(_save(fig, tmp_path, save_mode="print"))

    corner = image[2, 2]
    assert tuple(int(v) for v in corner[:3]) == (255, 255, 255), (
        "the saved page is not white")
    assert int(corner[3]) == 255, "the print page is opaque, not transparent"

    band = image[:int(image.shape[0] * 0.12), :, :3].reshape(-1, 3)
    darkest = band[band.sum(axis=1).argmin()]
    assert darkest.max() < 80, (
        f"the title band has no dark ink in it: {tuple(darkest)}")


def test_the_data_colours_come_through_untouched(tmp_path):
    """The half that must not move. A white data point turned black is, on a
    volcano, the colour of 'not a hit'."""
    fig, _ax = _dark_volcano()
    image = _pixels(_save(fig, tmp_path, save_mode="print"))

    for colour in (UP, DOWN, GREY):
        assert _carries(image, colour), f"{colour} did not survive the save"


def test_the_figure_on_screen_is_unchanged_by_the_save(tmp_path):
    """"The plot on screen is byte-identical before and after the save." A
    user watching a plot while it saves must not see it flash."""
    fig, _ax = _dark_volcano()
    before = _fingerprint(fig)
    _save(fig, tmp_path, save_mode="print")
    assert _fingerprint(fig) == before


def test_as_on_screen_still_produces_the_old_behaviour(tmp_path):
    fig, _ax = _dark_volcano()
    image = _pixels(_save(fig, tmp_path, save_mode="screen"))
    assert tuple(int(v) for v in image[2, 2][:3]) == (30, 30, 30)
    for colour in (UP, DOWN, GREY):
        assert _carries(image, colour)


def test_transparent_drops_the_ground_and_inks_for_the_theme(tmp_path,
                                                             monkeypatch):
    """The ground always goes; the ink is whichever theme is in force."""
    import spacr.figure_style as _fs

    for theme, readable in (("light", lambda v: v.max() < 80),
                            ("dark", lambda v: v.min() > 150)):
        monkeypatch.setattr(
            "spacr.qt.preferences.resolve_effective_theme", lambda t=theme: t)
        fig, _ax = _dark_volcano()
        image = _pixels(_save(fig, tmp_path, save_mode="transparent"))

        assert int(image[2, 2][3]) == 0, "the transparent save has a ground"
        band = image[:int(image.shape[0] * 0.12), :, :3].reshape(-1, 3)
        # In light mode the chrome is the darkest thing in the band; in dark
        # mode it is the lightest. Either way it is the mark being checked.
        mark = (band[band.sum(axis=1).argmin()] if theme == "light"
                else band[band.sum(axis=1).argmax()])
        assert readable(mark), (
            f"{theme} mode chrome came out {tuple(int(v) for v in mark)}")


def test_a_light_mode_save_changes_nothing_at_all(tmp_path):
    """The property that makes 'print' safe as the default: chrome that was
    already legible on the page is not touched, so nobody's existing figures
    move."""
    fig, ax = plt.subplots(facecolor="white")
    ax.plot([0, 1], [0, 1])
    ax.set_title("light")
    before = _fingerprint(fig)

    with P.print_ready(fig, mode="print", announce=False):
        during = _fingerprint(fig)

    assert during == before


def test_the_chrome_really_is_repainted_inside_the_block():
    """The partner of the test above: in a DARK figure the block does move
    things, so 'nothing changed' in light mode is evidence and not a no-op."""
    fig, ax = _dark_volcano()
    before = _fingerprint(fig)
    with P.print_ready(fig, mode="print", announce=False):
        during = _fingerprint(fig)
    assert during != before
    assert _fingerprint(fig) == before


def test_a_dark_axes_patch_is_repainted_and_a_tinted_light_one_is_not():
    """`savefig.facecolor` reaches the FIGURE patch only. The axes patch is a
    separate artist, and a dark one left behind is a dark rectangle inside a
    white page."""
    fig, ax = plt.subplots(facecolor=DARK_GROUND)
    ax.set_facecolor(DARK_GROUND)
    with P.print_ready(fig, mode="print", announce=False):
        assert ax.patch.get_facecolor()[:3] == (1.0, 1.0, 1.0)

    fig2, ax2 = plt.subplots()
    ax2.set_facecolor("#FFF6E5")            # somebody's deliberate warm tint
    with P.print_ready(fig2, mode="print", announce=False):
        assert ax2.patch.get_facecolor()[:3] != (1.0, 1.0, 1.0)


def test_a_reference_line_flips_and_a_data_line_does_not():
    """The one hard case in `_chrome`. A significance line, a zero line and a
    plotted series are all Line2D; they are told apart by their TRANSFORM,
    which is a property of what the line means rather than of its colour."""
    fig, ax = plt.subplots(facecolor=DARK_GROUND)
    ax.set_facecolor(DARK_GROUND)
    series, = ax.plot([0, 1], [0, 1], color="white")
    threshold = ax.axhline(0.5, color="white")
    zero = ax.axvline(0.5, color="white")

    with P.print_ready(fig, mode="print", announce=False):
        assert threshold.get_color() == FS.PRINT_INK
        assert zero.get_color() == FS.PRINT_INK
        assert series.get_color() == "white", (
            "a plotted series was repainted; that is the data")


def test_a_grid_becomes_faint_rather_than_ink():
    """A grid repainted in the ink is a cage over the data."""
    fig, ax = plt.subplots(facecolor=DARK_GROUND)
    ax.grid(True, color="white")
    with P.print_ready(fig, mode="print", announce=False):
        gridline = ax.xaxis.get_major_ticks()[0].gridline
        assert gridline.get_color() == FS.PRINT_GRID
        assert gridline.get_color() != FS.PRINT_INK


def test_an_illegible_data_colour_is_named_and_not_changed(tmp_path, capsys):
    """150 D. The data deliberately does not flip, so a palette chosen against
    near-black can be illegible on paper -- and a substitution the user did not
    ask for changes the picture."""
    fig, ax = plt.subplots(facecolor=DARK_GROUND)
    ax.set_facecolor(DARK_GROUND)
    ax.scatter([0, 1], [0, 1], color="#FFFDE7")   # reads on near-black only

    written = _save(fig, tmp_path, name="pale.png")
    captured = capsys.readouterr().out
    assert "#FFFDE7" in captured, "the stranded colour was not named"
    assert _carries(_pixels(written), "#FFFDE7"), (
        "it was named AND substituted; naming is the whole point")


def test_a_legible_palette_says_nothing(tmp_path, capsys):
    fig, ax = plt.subplots(facecolor=DARK_GROUND)
    ax.set_facecolor(DARK_GROUND)
    ax.scatter([0, 1], [0, 1], color=UP)
    _save(fig, tmp_path, name="fine.png")
    assert "no contrast" not in capsys.readouterr().out


def test_an_explicit_facecolor_still_wins(tmp_path):
    fig, ax = _dark_volcano()
    image = _pixels(_save(fig, tmp_path, name="pink.png", facecolor="#FFCCCC"))
    assert tuple(int(v) for v in image[2, 2][:3]) == (255, 204, 204)


def test_the_pdf_route_gets_the_same_treatment(tmp_path):
    """Two formats, one rule -- and the PDF is the one that goes into the
    manuscript."""
    fig, _ax = _dark_volcano()
    written = P.save_figure(fig, os.path.join(str(tmp_path), "v.pdf"),
                            fmt="pdf", save_mode="print")
    with open(written, "rb") as handle:
        head = handle.read(4)
    assert head == b"%PDF"
    # The chrome that was white is not white any more, checked on the artists
    # because a PDF has no pixels to read.
    with P.print_ready(fig, mode="print", announce=False):
        assert fig.axes[0].title.get_color() == FS.PRINT_INK


def test_a_wash_is_not_a_mark(tmp_path, capsys):
    """`figures.plates` lays the "never measured" colour down at 9% opacity.
    Judging its base hue would name a colour nobody is being asked to find, on
    every plate figure -- and a warning that fires on every figure is a warning
    nobody reads."""
    fig, ax = plt.subplots(facecolor=DARK_GROUND)
    ax.set_facecolor(DARK_GROUND)
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=(1.0, 1.0, 1.0, 0.09),
                           edgecolor="none"))
    ax.scatter([0, 1], [0, 1], color=UP)
    _save(fig, tmp_path, name="wash.png")
    assert "no contrast" not in capsys.readouterr().out


def test_an_opaque_white_mark_is_still_named(tmp_path, capsys):
    """The partner: the alpha rule must not become a way for a real mark to
    escape the check."""
    fig, ax = plt.subplots(facecolor=DARK_GROUND)
    ax.set_facecolor(DARK_GROUND)
    ax.scatter([0, 1], [0, 1], color=(1.0, 1.0, 1.0, 1.0))
    _save(fig, tmp_path, name="opaque.png")
    assert "#FFFFFF" in capsys.readouterr().out
