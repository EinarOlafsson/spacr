"""The declining answers of the figure style: the paths that change nothing.

Most of :mod:`spacr.figure_style` is about producing a value -- a colour, an
rcParam, an export mode. The cases here are the other half of every one of
those decisions: the style that asks for no automatic layout, the palette that
names nothing, the colour library that will not answer, the settings store
that raises. Each of them has to leave the figure exactly as it found it, and
"exactly" is measurable: an rcParam that is absent rather than set to a
default, a colour cycle that still holds the colours it held before, a mode
string that is still ``'print'``.

The colour reader is here for the opposite reason. ``to_rgb`` parses hex and
numeric colours itself and hands everything else to Matplotlib, which is the
path that lets a figure be styled in ``'red'`` or ``'tab:blue'`` at all.
"""
from __future__ import annotations

import sys

import pytest

from spacr.figure_style import (
    GENERAL_DEFAULTS,
    SAVE_MODES,
    _apply_palette,
    apply,
    figure_save_mode,
    palette_colours,
    rc_params,
    saved_figure_appearance,
    to_rgb,
)


@pytest.fixture()
def restored_rc_params():
    """Hand back the global rcParams afterwards; the store is process-wide."""
    import matplotlib as mpl

    before = dict(mpl.rcParams)
    yield mpl
    mpl.rcParams.update(before)


@pytest.fixture()
def sentinel_cycle(restored_rc_params):
    """Put a recognisable colour cycle in place and return its colours."""
    from cycler import cycler

    colours = ["#101010", "#202020"]
    restored_rc_params.rcParams["axes.prop_cycle"] = cycler(color=colours)
    return colours


def _cycle_of(mpl):
    """Read the colour cycle back out of the global rcParams."""
    return list(mpl.rcParams["axes.prop_cycle"].by_key()["color"])


# ---------------------------------------------------------------------------
# rcParams a style does not ask for are absent, not defaulted
# ---------------------------------------------------------------------------

def test_a_style_that_declines_tight_layout_emits_no_autolayout_param():
    """Absent is not the same as ``False`` here.

    `spacr.figures.style.rc` builds its overrides by diffing two `rc_params`
    dicts, so emitting ``figure.autolayout: False`` for a style that simply
    never asked about layout would push a decision into every figure. Only a
    style that asks for tight layout may say anything about it.
    """
    assert "figure.autolayout" not in rc_params({"tight_layout": False})
    assert "figure.autolayout" not in rc_params({})
    assert rc_params({"tight_layout": True})["figure.autolayout"] is True


def test_a_style_with_no_palette_emits_no_colour_cycle():
    """A nameless palette leaves the cycle to whatever is already set."""
    assert "axes.prop_cycle" not in rc_params({"palette": ""})
    assert "axes.prop_cycle" not in rc_params({"palette": None})

    named = rc_params({"palette": "colorblind"})
    assert list(named["axes.prop_cycle"].by_key()["color"])[:3] == [
        "#0173b2", "#de8f05", "#029e73"]


# ---------------------------------------------------------------------------
# Applying a style
# ---------------------------------------------------------------------------

def test_applying_a_style_with_no_palette_never_reaches_the_palette_applier(
        monkeypatch, sentinel_cycle, restored_rc_params):
    """An empty palette name is a decision not to touch the colour cycle.

    The applier is replaced by one that installs an unmistakable cycle, so
    the question "did the empty name get through the guard" is answered by
    the colours the figure would actually be drawn in.
    """
    import spacr.figure_style as figure_style
    from cycler import cycler

    def install_marker(name):
        restored_rc_params.rcParams["axes.prop_cycle"] = cycler(
            color=["#FF00FF"])

    monkeypatch.setattr(figure_style, "_apply_palette", install_marker)

    style = apply(general={"palette": ""})

    assert style["palette"] == ""
    assert _cycle_of(restored_rc_params) == sentinel_cycle

    apply(general={"palette": "deep"})
    assert _cycle_of(restored_rc_params) == ["#FF00FF"]


def test_a_style_matplotlib_cannot_read_does_not_stop_the_run(
        restored_rc_params):
    """A broken style setting loses the styling, not the figure.

    ``font_size`` arrives from a settings store and from notebooks, so a
    value that is not a number is reachable. `apply` still has to return the
    resolved style -- the caller reads per-graph settings such as
    ``label_top_n`` out of it that have no rcParam at all -- and it must
    leave the global rcParams untouched rather than half-written.
    """
    before = float(restored_rc_params.rcParams["font.size"])

    style = apply(general={"font_size": "eleven"})

    assert style["font_size"] == "eleven"
    assert style["title_size"] == GENERAL_DEFAULTS["title_size"]
    assert float(restored_rc_params.rcParams["font.size"]) == before


# ---------------------------------------------------------------------------
# Resolving a palette by name
# ---------------------------------------------------------------------------

def test_a_palette_with_no_name_resolves_to_no_colours():
    """No name is a question that cannot be answered, so nothing comes back.

    An empty list is what lets `rc_params` and `_apply_palette` leave the
    existing cycle alone; a fallback palette here would silently overwrite
    the colours a caller had chosen for itself.
    """
    assert palette_colours(None) == []
    assert palette_colours("") == []


def test_a_palette_neither_library_can_supply_resolves_to_no_colours(
        monkeypatch):
    """With seaborn refusing and spaCR's own colours unimportable, ``[]``.

    Figures are drawn from headless workers where the Qt widget package may
    not import at all. The last resort has to be "keep the cycle you have",
    because raising here would take the plot down over its colours.
    """
    import seaborn as sns

    def refuse(*args, **kwargs):
        raise ValueError("no palette by that name")

    monkeypatch.setattr(sns, "color_palette", refuse)
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.fast_plots", None)

    assert palette_colours("colorblind") == []


def test_an_empty_palette_leaves_the_existing_colour_cycle_in_place(
        sentinel_cycle, restored_rc_params):
    """`_apply_palette` with nothing to apply is a no-op, not an empty cycle.

    Writing an empty cycler would leave Matplotlib with no colours to hand
    out and every subsequent series drawn in the same colour.
    """
    _apply_palette("")

    assert _cycle_of(restored_rc_params) == sentinel_cycle

    _apply_palette("colorblind")
    assert _cycle_of(restored_rc_params)[0] == "#0173b2"


# ---------------------------------------------------------------------------
# Colours Matplotlib reads and spaCR's parser does not
# ---------------------------------------------------------------------------

def test_a_named_matplotlib_colour_is_read_through_matplotlib():
    """``'red'`` and ``'tab:blue'`` are colours a user can put in a style.

    The hand-written branch of `to_rgb` only knows hex and the transparent
    names, so every other colour spelling Matplotlib accepts has to reach
    Matplotlib or the contrast check would treat it as unreadable and refuse
    to recolour it on export.
    """
    assert to_rgb("red") == (1.0, 0.0, 0.0)
    assert to_rgb("tab:blue") == pytest.approx(
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765))
    assert to_rgb("0.5") == pytest.approx((0.5, 0.5, 0.5))


def test_a_colour_matplotlib_reads_as_fully_transparent_is_not_a_colour():
    """``(name, alpha)`` is a Matplotlib colour spec, and alpha 0 draws nothing.

    Reporting the underlying red would let the export contrast check pass a
    figure element that is not on the page at all.
    """
    assert to_rgb(("red", 0.0)) is None
    assert to_rgb(("red", 1.0)) == (1.0, 0.0, 0.0)


@pytest.mark.parametrize("spec", [(0.5, 0.5), "chartreusey", (), "rgb(1,2,3)"])
def test_a_specification_matplotlib_refuses_is_refused_here_too(spec):
    """Anything neither parser can read comes back as None, never a guess.

    A guessed colour would produce a contrast number about a colour nobody
    chose, and the export path would then repaint an element on the strength
    of it.
    """
    assert to_rgb(spec) is None


# ---------------------------------------------------------------------------
# Where the export mode comes from
# ---------------------------------------------------------------------------

def test_a_stored_save_mode_is_used_when_the_environment_names_none(
        monkeypatch):
    """The settings store answers once the environment variable is out of it.

    The environment variable exists for command-line and notebook runs; the
    GUI's answer is the stored preference, and it has to reach both
    renderers through this one function.
    """
    from spacr.qt import preferences

    monkeypatch.delenv("SPACR_FIGURE_SAVE_MODE", raising=False)
    monkeypatch.setattr(preferences, "get_figure_save_mode",
                        lambda: "  Transparent  ", raising=False)

    assert figure_save_mode() == "transparent"
    assert saved_figure_appearance().mode == "transparent"


def test_the_environment_outranks_the_stored_save_mode(monkeypatch):
    """A named mode in the environment wins over whatever is stored."""
    from spacr.qt import preferences

    monkeypatch.setenv("SPACR_FIGURE_SAVE_MODE", "screen")
    monkeypatch.setattr(preferences, "get_figure_save_mode",
                        lambda: "transparent", raising=False)

    assert figure_save_mode() == "screen"


def test_a_stored_save_mode_that_is_not_a_mode_falls_back_to_print(
        monkeypatch):
    """An unrecognised stored value is ignored rather than passed through.

    ``print`` is the safe answer: a figure exported for paper is legible
    whatever the application theme is.
    """
    from spacr.qt import preferences

    monkeypatch.delenv("SPACR_FIGURE_SAVE_MODE", raising=False)
    monkeypatch.setattr(preferences, "get_figure_save_mode",
                        lambda: "chartreuse", raising=False)

    assert figure_save_mode() == "print"


def test_a_settings_store_that_raises_still_names_a_save_mode(monkeypatch):
    """A broken preference read leaves an export mode, not a traceback."""
    from spacr.qt import preferences

    def refuse():
        raise RuntimeError("the settings store is unreadable")

    monkeypatch.delenv("SPACR_FIGURE_SAVE_MODE", raising=False)
    monkeypatch.setattr(preferences, "get_figure_save_mode", refuse,
                        raising=False)

    assert figure_save_mode() == "print"


def test_no_environment_and_no_stored_getter_means_print(monkeypatch):
    """With neither source present the mode is still one of the three."""
    from spacr.qt import preferences

    monkeypatch.delenv("SPACR_FIGURE_SAVE_MODE", raising=False)
    monkeypatch.delattr(preferences, "get_figure_save_mode", raising=False)

    mode = figure_save_mode()

    assert mode == "print"
    assert mode in SAVE_MODES
