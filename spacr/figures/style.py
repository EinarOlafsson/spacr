"""Shared publication and on-screen styling for spaCR figures.

The module provides a fixed data palette, theme-aware foreground colors, and
scoped Matplotlib style contexts. Data colors retain the same meaning across
panels, while text and axes adapt to screen or print backgrounds. Styles are
applied without mutating process-wide ``rcParams``.

Text and lines are two colours, not one. Titles, axis labels, tick labels,
annotations and legend entries are TEXT; axis spines and tick marks are
LINES. Each follows the matching user preference when one has been chosen and
the measured house ink otherwise, so an untouched settings store draws the
published look unchanged.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import sys
from typing import Iterable, Optional, Sequence

#: Print ink: the near-black the published figures use for text and axes.
INK_PRINT = "#231F20"
#: Screen ink: what the dark GUI needs instead. Not pure white — pure white
#: on a dark ground reads as glare, and the published figures never use a
#: maximal ink either.
INK_SCREEN = "#E8EDEE"

#: Transparent, spelled the way matplotlib understands it.
TRANSPARENT = "none"


class Palette:
    """Fixed data colors shared by all house-style figure panels."""

    GREY = "#B4B4B4"
    GREY_DARK = "#7F7F7F"
    BLUE = "#2E77BC"
    BLUE_LIGHT = "#7FB3E0"
    GREEN = "#2E7D4F"
    RUST = "#C4441C"
    CORAL = "#E8A88C"
    GOLD = "#E8C33A"
    OCHRE = "#C87A28"
    PURPLE = "#8B4A82"
    NAVY = "#1F3F6E"

    #: Single-hue ramp for a p-value or a score. Diverging maps are for
    #: genuinely signed quantities only.
    SEQUENTIAL = "Blues"

    #: The reference-paper palette (spacr-graphs skill, 2026-11-18), measured
    #: from the figures of the Lourido-lab reference papers: a mid grey for
    #: the control or reference condition, a light grey for background
    #: points, one saturated blue accent and four further series. Never
    #: brown.
    CONTROL = "#8C8C8C"
    BACKGROUND = "#C8C8C8"
    ACCENT = "#0A77B8"
    TEAL = "#0A9878"
    NAVY_DARK = "#283868"
    ACCENT_LIGHT = "#78AED2"
    MAGENTA = "#C878A8"
    AMBER = "#F0A830"


#: What each role means in a spaCR regression figure, fixed once so a colour
#: cannot drift between panels. This is the whole of the colour vocabulary.
ROLES = {
    "data": Palette.GREY,
    "up": Palette.GREEN,
    "down": Palette.RUST,
    "highlight": Palette.BLUE,
    "control_negative": Palette.GREY_DARK,
    "control_positive": Palette.PURPLE,
    "fill": Palette.CORAL,
    "reference": Palette.GREY_DARK,
}

#: Absolute type sizes that reproduce the published hierarchy at 300 dpi.
#: Centralising them prevents individual panels from drifting apart.
TYPE_SCALE = {
    "tick": 8.0,
    "label": 8.0,
    "annotation": 8.0,
    "panel_letter": 10.0,
    "legend": 8.0,
}

#: Line weights, likewise measured rather than chosen.
WEIGHTS = {"spine": 0.6, "data": 1.0, "reference": 0.6, "median": 0.9}

#: The rcParams that are TEXT: everything a reader reads. Tick LABELS are in
#: here and tick MARKS are not.
TEXT_KEYS = ("text.color", "axes.labelcolor",
             "xtick.labelcolor", "ytick.labelcolor")
#: The rcParams that are LINES: the axis spines and the tick marks. The
#: figure's chrome, not its data — :func:`resolve_line_ink` says why the
#: data's own series are left alone.
LINE_KEYS = ("axes.edgecolor", "xtick.color", "ytick.color")



#: True inside :func:`pipeline_figures`: the figures a pipeline writes ignore
#: the user's figure preferences.
_PIPELINE_FIGURES = contextvars.ContextVar("spacr_pipeline_figures",
                                           default=False)

#: Set by :func:`pipeline_figures` for child processes of a pipeline run.
_PIPELINE_FIGURES_ENV = "SPACR_PIPELINE_FIGURES"


def _gui_process() -> bool:
    """Whether this process is the spaCR GUI (a Qt application is running)."""
    widgets = sys.modules.get("PySide6.QtWidgets")
    try:
        return widgets is not None and widgets.QApplication.instance() is not None
    except Exception:                                          # noqa: BLE001
        return False


def _pipeline_figures_active() -> bool:
    """Whether figures saved now ignore the user's figure preferences.

    True inside :func:`pipeline_figures`, and in a process started from one
    (the environment marker), unless this process is the GUI: figures the
    GUI saves, including those of pipelines it runs, keep the preferences.
    """
    if _gui_process():
        return False
    return bool(_PIPELINE_FIGURES.get()) or (
        os.environ.get(_PIPELINE_FIGURES_ENV) == "1")


def _refuse_preferences_in_pipeline() -> None:
    """Raise inside a pipeline run, so a preference reader falls back to the
    house style exactly as it does when there is no preference store."""
    if _pipeline_figures_active():
        raise LookupError("pipeline figures ignore the figure preferences")


@contextlib.contextmanager
def pipeline_figures():
    """Save figures in the plain house style, ignoring the figure preferences.

    Runs outside the GUI (``spacr-run``, and :func:`spacr.ops_engine.run_ops`
    called from a script) save their figures inside this, so a settings-file
    rerun gives byte-identical files whatever the user chose in Preferences.
    In the GUI process it changes nothing: figures the GUI saves, including
    those of the pipelines it runs, keep the preferences.
    """
    token = _PIPELINE_FIGURES.set(True)
    previous = os.environ.get(_PIPELINE_FIGURES_ENV)
    if not _gui_process():
        os.environ[_PIPELINE_FIGURES_ENV] = "1"
    try:
        yield
    finally:
        _PIPELINE_FIGURES.reset(token)
        if previous is None:
            os.environ.pop(_PIPELINE_FIGURES_ENV, None)
        else:
            os.environ[_PIPELINE_FIGURES_ENV] = previous

def chosen_ink() -> Optional[str]:
    """The TEXT colour the user picked, or ``None`` while it follows the theme.

    Reads the stored TOKEN, never the resolved pair. A resolved pair has
    already lost the one bit that matters here — whether the user chose the
    colour or the theme produced it — so seeding the house style from
    ``get_figure_colors()`` would replace the measured publication ink with
    whatever the current theme happens to answer, for every user who has
    never opened the dialog.

    ``None`` when there is no settings store at all: a headless render or a
    bare unit run, where the house style is the only answer there is.
    """
    try:
        _refuse_preferences_in_pipeline()
        from ..qt.preferences import (figure_color_is_auto,
                                      get_figure_color_tokens)

        _ground, text = get_figure_color_tokens()
    except Exception:                                          # noqa: BLE001
        return None
    if figure_color_is_auto(text):
        return None
    return str(text).strip() or None


def chosen_line_ink() -> Optional[str]:
    """The LINE colour the user picked, or ``None`` while it follows the text.

    The user's second colour control: "line color which should change the
    color of all lines including axis lines and ticks". Stored as a token
    like the text half, and read as a token for the same reason —
    :func:`chosen_ink` says which.
    """
    try:
        _refuse_preferences_in_pipeline()
        from ..qt.preferences import (figure_color_is_auto,
                                      get_figure_line_token)

        token = get_figure_line_token()
    except Exception:                                          # noqa: BLE001
        return None
    if figure_color_is_auto(token):
        return None
    return str(token).strip() or None


def resolve_ink(target: str = "screen", ink: Optional[str] = None) -> str:
    """Return the text colour for the figure's output target.

    Titles, axis labels, tick LABELS, annotations and legend entries. The
    axis spines and the tick MARKS are lines, and they have their own
    resolver — see :func:`resolve_line_ink`.

    :param target: ``'screen'`` for the GUI, ``'print'`` for a file that will
        be looked at on paper or in a white-page viewer.
    :param ink: an explicit override, which always wins.
    """
    if ink:
        return ink
    return chosen_ink() or (INK_PRINT if target == "print" else INK_SCREEN)


def resolve_line_ink(target: str = "screen", ink: Optional[str] = None,
                     line: Optional[str] = None) -> str:
    """Resolve the colour used for axis spines and tick marks.

    Textual elements, including tick labels, use :func:`resolve_ink`; the
    separate resolvers allow line and text colours to be configured
    independently. If no line colour is configured, the resolved text colour
    preserves the earlier single-ink behavior. Data-series colours remain
    governed by :data:`ROLES` and per-figure styling.

    :param target: Output target passed to :func:`resolve_ink` when a fallback
        is required.
    :param ink: Explicit text colour used as the fallback for line work.
    :param line: Explicit line colour, which takes precedence over all other
        values.
    :returns: Resolved line colour.
    """
    if line:
        return line
    return chosen_line_ink() or resolve_ink(target, ink)


#: The page a LABEL sits on, per target. Not the figure's ground -- that is
#: transparent by design (118) -- but what goes behind a text box that has to
#: stay readable over the data underneath it.
#:
#: THIS EXISTS BECAUSE A BOX WAS HARD-CODED WHITE while its text followed the
#: theme, so on the dark theme it was white ink on a white box: a label that
#: was there, was drawn, and could not be read. `resolve_ink`'s opposite
#: number, and used the same way.
LABEL_GROUND_PRINT = "#FFFFFF"
LABEL_GROUND_SCREEN = "#1B1E20"


def resolve_label_ground(target: str = "screen",
                         ground: Optional[str] = None) -> str:
    """Return the background colour for a text label on the output target.

    :param target: ``'screen'`` for the GUI, ``'print'`` for a file.
    :param ground: an explicit override, which always wins.
    """
    if ground:
        return ground
    return LABEL_GROUND_PRINT if target == "print" else LABEL_GROUND_SCREEN


def user_overrides(kind: Optional[str] = None) -> dict:
    """Return explicit preference changes that override the house style.

    Only settings that differ from spaCR's figure defaults are returned. This
    preserves the house style for untouched preferences while allowing general
    and graph-specific choices to take precedence.

    Parameters
    ----------
    kind : str or None, default=None
        Graph kind from :data:`spacr.figure_style.GRAPH_KINDS`. ``None`` uses
        only the general preference layer.

    Returns
    -------
    dict
        Matplotlib ``rcParams`` overrides. An empty dictionary is returned
        when no preference differs or preferences cannot be read safely.
    """
    try:
        _refuse_preferences_in_pipeline()
        from ..qt.preferences import (get_figure_style,
                                      get_figure_style_per_graph)

        general = get_figure_style()
        per_graph = get_figure_style_per_graph()
    except Exception:                                          # noqa: BLE001
        return {}
    if not general and not per_graph:
        return {}
    try:
        from ..figure_style import rc_params, resolve

        chosen = resolve(kind, general, per_graph)
        untouched = resolve(kind)
        after, before = rc_params(chosen), rc_params(untouched)
        return {key: value for key, value in after.items()
                if before.get(key) != value}
    except Exception:                                          # noqa: BLE001
        return {}


def _preference_deltas(kind: Optional[str] = None) -> dict:
    """The figure settings the user changed in Preferences, by style name.

    :param kind: graph kind whose per-graph changes are laid on top.
    :returns: changed settings only; empty when nothing changed or the
        preference store cannot be read, so a headless run draws as before.
    """
    try:
        _refuse_preferences_in_pipeline()
        from ..qt.preferences import (get_figure_style,
                                      get_figure_style_per_graph)

        general = get_figure_style()
        per_graph = get_figure_style_per_graph()
    except Exception:                                          # noqa: BLE001
        return {}
    try:
        from ..figure_style import _user_deltas

        return _user_deltas(kind, general, per_graph)
    except Exception:                                          # noqa: BLE001
        return {}


#: Palettes whose colours a figure may have been drawn in without asking the
#: preference: matplotlib's default cycle and seaborn's named palettes.
_SOURCE_PALETTES = ("tab10", "deep", "colorblind", "muted", "pastel",
                    "bright", "dark")

#: Attribute set on a figure once the user's settings have been applied.
_STYLED_FLAG = "_spacr_user_style_applied"


def _palette_map(name: str) -> dict:
    """Map every known default palette colour to the chosen palette's colour.

    A colour at position ``i`` of a source palette becomes position ``i`` of
    the chosen palette, so groups keep their order and their distinctness.
    """
    from matplotlib.colors import to_hex

    from ..figure_style import palette_colours

    target = palette_colours(name)
    if not target:
        return {}
    mapping = {}
    for source in _SOURCE_PALETTES:
        try:
            if source == "tab10":
                from matplotlib import colormaps
                colours = [to_hex(c) for c in colormaps["tab10"].colors]
            else:
                colours = palette_colours(source)
        except Exception:                                      # noqa: BLE001
            continue
        for index, colour in enumerate(colours):
            mapping.setdefault(str(colour).lower(),
                               target[index % len(target)])
    return mapping


def _recolour(artist, mapping: dict) -> None:
    """Swap an artist's default-palette colours for the chosen palette."""
    from matplotlib.colors import to_hex, to_rgba

    def swap(colour):
        """Map a default colour to the selected palette, preserving opacity."""
        try:
            rgba = to_rgba(colour)
        except (TypeError, ValueError):
            return None
        new = mapping.get(to_hex(rgba, keep_alpha=False).lower())
        if new is None:
            return None
        return to_rgba(new, alpha=rgba[3])

    for getter, setter in (("get_color", "set_color"),
                           ("get_markerfacecolor", "set_markerfacecolor"),
                           ("get_markeredgecolor", "set_markeredgecolor")):
        if hasattr(artist, getter) and hasattr(artist, "get_xdata"):
            new = swap(getattr(artist, getter)())
            if new is not None:
                getattr(artist, setter)(new)
    for getter, setter in (("get_facecolor", "set_facecolor"),
                           ("get_edgecolor", "set_edgecolor")):
        if not hasattr(artist, getter) or hasattr(artist, "get_xdata"):
            continue
        current = getattr(artist, getter)()
        try:
            import numpy as np

            values = np.atleast_2d(np.asarray(current, dtype=float))
        except (TypeError, ValueError):
            continue
        if values.size == 0 or values.shape[-1] != 4:
            continue
        swapped = [swap(tuple(row)) for row in values]
        if any(new is not None for new in swapped):
            rows = [new if new is not None else tuple(row)
                    for new, row in zip(swapped, values)]
            getattr(artist, setter)(rows if len(rows) > 1 else rows[0])


def _apply_user_style(figure, kind: Optional[str] = None, *,
                      force: bool = False) -> dict:
    """Apply the user's changed figure settings to an already drawn figure.

    This is the one place a finished figure takes the Preferences figure
    settings, whichever module drew it: :func:`spacr.plot.save_figure` calls
    it before writing, and the figure panel and the embedded canvases call it
    before showing. Figures drawn inside :func:`figure_style` already carry
    the settings through ``rcParams``; applying them again changes nothing.

    Only settings the user changed are applied: font family and the title,
    axis-label, tick and legend sizes, line width, marker size, palette,
    colormap, background, grid, spines and despine offset; the resolution
    and file formats are applied by the writer. A figure is styled once; a later right-click restyle is never undone by a save.

    :param figure: the matplotlib figure, changed in place.
    :param kind: graph kind whose per-graph settings apply; when ``None`` the
        figure's ``_spacr_graph_kind`` attribute is used if it has one.
    :param force: apply again even if the figure was already styled.
    :returns: the settings that were applied, by style name.
    """
    if figure is None or (getattr(figure, _STYLED_FLAG, False) and not force):
        return {}
    kind = kind or getattr(figure, "_spacr_graph_kind", None)
    changed = _preference_deltas(kind)
    try:
        setattr(figure, _STYLED_FLAG, True)
    except Exception:                                          # noqa: BLE001
        pass
    if not changed:
        return {}
    try:
        _restyle(figure, changed)
    except Exception:                                          # noqa: BLE001
        return {}
    return changed


_STYLED_CONTENT = "_spacr_styled_content"


def _content_signature(figure) -> tuple:
    """What a figure has drawn on it, as far as restyling cares.

    One entry per axes: the axes itself and how many lines, collections,
    patches and images it holds. Statistics annotations (gid
    ``spacr-stats``) and texts are left out, so a test drawn from the figure
    menu or a retitled axis does not count as new content.
    """
    signature = []
    for axes in figure.axes:
        count = 0
        for group in (axes.lines, axes.collections, axes.patches,
                      axes.images):
            count += sum(1 for artist in group
                         if artist.get_gid() != "spacr-stats")
        signature.append((id(axes), count))
    return tuple(signature)


def _style_new_content(figure, kind: Optional[str] = None) -> dict:
    """Apply the user's figure settings to whatever was drawn since last time.

    For a canvas that a screen draws into again and again (control charts,
    dose response, PCA, the Feature Explorer and every other screen on the
    shared menu canvas): the canvas calls this before each draw, and the
    settings are applied only when the figure holds new content, so a
    right-click edit is not undone by the next repaint.

    :param figure: the matplotlib figure about to be drawn.
    :param kind: graph kind whose per-graph settings apply.
    :returns: the settings that were applied, empty when nothing was new.
    """
    if figure is None:
        return {}
    try:
        signature = _content_signature(figure)
    except Exception:
        return {}
    if not signature or getattr(figure, _STYLED_CONTENT, None) == signature:
        return {}
    applied = _apply_user_style(figure, kind, force=True)
    try:
        setattr(figure, _STYLED_CONTENT, _content_signature(figure))
    except Exception:
        pass
    return applied


def _restyle(figure, changed: dict) -> None:
    """Write each changed setting onto the figure's artists."""
    from ..figure_style import SPINE_PRESETS

    def texts(ax):
        """Title, axis-label, tick-label and legend texts of one axes."""
        titles = [ax.title, getattr(ax, "_left_title", None),
                  getattr(ax, "_right_title", None)]
        labels = [ax.xaxis.label, ax.yaxis.label]
        ticks = list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        legend = ax.get_legend()
        entries = []
        if legend is not None:
            entries = list(legend.get_texts()) + [legend.get_title()]
        return [t for t in titles if t is not None], labels, ticks, entries

    sizes = {"title_size": 0, "label_size": 1, "tick_size": 2,
             "legend_size": 3}
    family = changed.get("font_family")
    if "background" in changed:
        ground = changed["background"]
        figure.patch.set_facecolor(ground)
    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is not None and "title_size" in changed:
        suptitle.set_fontsize(float(changed["title_size"]))
    if family:
        for text in figure.findobj(lambda a: hasattr(a, "set_fontfamily")):
            text.set_fontfamily(family)
    if "tick_size" in changed:
        for ax in figure.get_axes():
            ax.tick_params(labelsize=float(changed["tick_size"]))
    mapping = (_palette_map(changed["palette"])
               if changed.get("palette") else {})
    preset = SPINE_PRESETS.get(str(changed.get("spines", "")))
    for ax in figure.get_axes():
        groups = texts(ax)
        for key, position in sizes.items():
            if key in changed:
                for text in groups[position]:
                    text.set_fontsize(float(changed[key]))
        if "background" in changed:
            ax.set_facecolor(changed["background"])
        if "grid" in changed:
            ax.grid(bool(changed["grid"]))
        if preset is not None:
            for name, shown in zip(("top", "right", "bottom", "left"),
                                   preset):
                ax.spines[name].set_visible(shown)
        if "spine_width" in changed:
            for spine in ax.spines.values():
                spine.set_linewidth(float(changed["spine_width"]))
        if float(changed.get("despine_offset", 0) or 0) > 0:
            for spine in ax.spines.values():
                spine.set_position(("outward",
                                    float(changed["despine_offset"])))
        for line in ax.get_lines():
            if "line_width" in changed:
                line.set_linewidth(float(changed["line_width"]))
            if "marker_size" in changed and line.get_marker() not in (
                    None, "", "None", " "):
                line.set_markersize(float(changed["marker_size"]) ** 0.5)
        for collection in ax.collections:
            if "marker_size" in changed and hasattr(collection, "get_sizes"):
                current = collection.get_sizes()
                if len(current) and len(set(current.tolist())) == 1:
                    collection.set_sizes([float(changed["marker_size"])])
        if "colormap" in changed:
            for mappable in list(ax.images) + list(ax.collections):
                cmap = getattr(mappable, "get_cmap", lambda: None)()
                if (cmap is not None and mappable.get_array() is not None
                        and getattr(cmap, "name", "") in ("viridis",)):
                    mappable.set_cmap(str(changed["colormap"]))
        if mapping:
            for artist in (list(ax.lines) + list(ax.collections)
                           + list(ax.patches)):
                _recolour(artist, mapping)


def _group_colours(count: int, palette: Sequence = (),
                  kind: Optional[str] = "jitter_bar") -> Optional[list]:
    """The colours ``count`` groups take under the user's ``mark_colouring``.

    `mark_colouring` is a rule about which colour each mark takes, not an
    rcParam, so a grouped renderer that names its own colours has to ask for
    it. This is that question, answered once for every such renderer.

    * ``group`` (the default) returns ``None``: the renderer keeps its own
      house rule -- grey unless the colour is what separates the series.
    * ``uniform`` returns the data ink for every group.
    * ``random`` returns ``palette`` reordered by the fixed seed in
      :func:`spacr.figure_style._marks_coloured_by`, cycled to ``count``, so
      neighbouring groups are told apart by eye and a redraw gives the same
      colours.

    :param count: how many groups will be drawn.
    :param palette: the renderer's categorical palette; the house default
        palette is used when it is empty.
    :param kind: the graph kind whose per-graph preference applies.
    :returns: ``count`` colour specs, or ``None`` to keep the house rule.
    """
    from ..figure_style import (GENERAL_DEFAULTS, STYLE_CHOICES,
                                _marks_coloured_by, palette_colours, resolve)

    try:
        _refuse_preferences_in_pipeline()
        from ..qt.preferences import (get_figure_style,
                                      get_figure_style_per_graph)

        chosen = resolve(kind, get_figure_style(),
                         get_figure_style_per_graph()).get("mark_colouring")
    except Exception:
        return None
    rule = str(chosen or "group").strip().lower()
    if rule not in STYLE_CHOICES["mark_colouring"] or rule == "group":
        return None
    count = max(1, int(count))
    if rule == "uniform":
        return [ROLES["data"]] * count
    colours = list(palette or ()) or palette_colours(
        GENERAL_DEFAULTS["palette"])
    colours = _marks_coloured_by(colours, "random")
    if not colours:
        return None
    return [colours[index % len(colours)] for index in range(count)]


def rc(target: str = "screen", *, frame: str = "box",
       ink: Optional[str] = None,
       line: Optional[str] = None,
       ground: Optional[str] = None,
       kind: Optional[str] = None) -> dict:
    """The rcParams for the house style, as a plain dict.

    Returned rather than applied, so a caller can hand it to
    :func:`figure_style` or to ``plt.rc_context`` directly. Nothing in this
    module ever mutates the global rcParams.

    :param frame: ``'box'`` (the default, spacr-graphs) draws all four
        spines (Nature Microbiology); ``'L'`` draws the left and bottom
        spines only (the Cell figures). Pick one
        per figure and hold it — box reads better when panels are small and
        dense, L when they are sparse.
    :param ink: the TEXT colour — titles, labels, tick labels, annotations.
    :param line: the LINE colour — the axis spines and the tick marks. Falls
        back to the resolved text colour when neither this argument nor the
        user's line-colour preference is set.
    :param ground: the figure and axes background. Defaults to transparent,
        which lets the GUI theme
        show through.
    :param kind: which graph kind this is, so the user's PER-GRAPH preference
        for it can be applied on top. See :func:`user_overrides`.
    """
    box = frame == "box"
    picked_ink = ink or chosen_ink()
    picked_line = line or chosen_line_ink()
    colour = resolve_ink(target, picked_ink)
    from ..figure_font import FAMILY as _FIGURE_FAMILY
    from ..figure_font import use_open_sans_for_figures
    use_open_sans_for_figures()
    line_colour = resolve_line_ink(target, ink=picked_ink, line=picked_line)
    ground = TRANSPARENT if ground is None else ground
    params = {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": [_FIGURE_FAMILY, "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": TYPE_SCALE["tick"],
        "axes.labelsize": TYPE_SCALE["label"],
        "axes.titlesize": TYPE_SCALE["label"],
        "axes.titleweight": "regular",
        "axes.titlelocation": "center",
        "axes.edgecolor": line_colour,
        "axes.labelcolor": colour,
        "axes.linewidth": WEIGHTS["spine"],
        "axes.grid": False,
        "axes.spines.top": box,
        "axes.spines.right": box,
        "xtick.color": line_colour, "ytick.color": line_colour,
        "xtick.labelcolor": colour, "ytick.labelcolor": colour,
        "text.color": colour,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.labelpad": 2.0, "axes.titlepad": 2.0,
        "errorbar.capsize": 0,
        "xtick.major.width": WEIGHTS["spine"],
        "ytick.major.width": WEIGHTS["spine"],
        "xtick.labelsize": TYPE_SCALE["tick"],
        "ytick.labelsize": TYPE_SCALE["tick"],
        "legend.frameon": False,
        "legend.fontsize": TYPE_SCALE["legend"],
        "legend.handlelength": 0.9,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        "figure.facecolor": ground,
        "axes.facecolor": ground,
        "savefig.facecolor": ground,
        "savefig.transparent": ground == TRANSPARENT,
        "lines.linewidth": WEIGHTS["data"],
        "patch.linewidth": WEIGHTS["spine"],
    }
    params.update(user_overrides(kind))
    if picked_ink:
        params.update(dict.fromkeys(TEXT_KEYS, colour))
    if picked_ink or picked_line:
        params.update(dict.fromkeys(LINE_KEYS, line_colour))
    return params


@contextlib.contextmanager
def figure_style(target: str = "screen", **kwargs):
    """Context manager that applies the house style and restores rcParams on exit.

    This is the supported way to apply the style.

    ::

        with figure_style("print"):
            figure = build_volcano(results)
        figure.savefig(path)          # already styled; the globals are back

    A plain ``rcParams.update`` would leak: spaCR draws figures from a
    long-lived GUI, so a style applied once applies to every figure drawn
    afterwards, in every other module, until the process exits.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(rc(target, **kwargs)):
        yield


@contextlib.contextmanager
def _figure_axes(*args, **kwargs):
    """Create subplot axes and draw their artists in the selected house style.

    The style remains active for the entire drawing block and is restored
    even if drawing raises. Arguments pass through to matplotlib's subplots.

    :returns: context manager yielding the figure and its axes.
    """
    import matplotlib.pyplot as plt

    with figure_style(theme_target()):
        yield plt.subplots(*args, **kwargs)


def theme_target() -> str:
    """``'screen'`` or ``'print'``, from the user's own figure preferences.

    The GROUND decides this, and only the ground: it is the question "what is
    this figure going to sit on", which is what picks between the two
    measured house inks. The colours the user may have CHOSEN are a different
    question and are read where they are used —
    :func:`chosen_ink` for the text, :func:`chosen_line_ink` for the lines —
    because a chosen colour outranks whichever house ink this returns.

    Falls back to ``'screen'`` when there is no settings store — a headless
    render or a bare unit test — because spaCR's themes are dark and ink that
    is slightly wrong is better than ink that is invisible.
    """
    try:
        _refuse_preferences_in_pipeline()
        from ..qt.preferences import get_figure_colors

        background, _foreground = get_figure_colors()
    except Exception:
        return "screen"
    text = str(background).strip().lower()
    return "print" if text in ("white", "#ffffff", "#fff") else "screen"



def panel_letter(ax, letter: str, dx: float = -0.16, dy: float = 1.06) -> None:
    """Draw a bold upper-case panel letter, without a period, at the top left.

    Sized from the measured 1.9-2.2x of the axis-label tier.

    :param ax: Matplotlib axes that receives the panel label.
    :param letter: panel identifier, converted to upper case.
    """
    ax.text(dx, dy, letter.upper(), transform=ax.transAxes,
            fontsize=TYPE_SCALE["panel_letter"], fontweight="bold",
            va="bottom", ha="left")


def descriptor(ax, text: str) -> None:
    """Set a short descriptor (two to four lower-case words) as the axes title.

    Not a sentence title. The axis labels carry the content; a descriptor
    only names the condition the panel shows.

    :param ax: Matplotlib axes whose title is set.
    :param text: short descriptor to place above the axes.
    """
    ax.set_title(text, fontsize=TYPE_SCALE["label"], pad=3.0)


def reference_line(ax, *, x=None, y=None, label: str = "",
                   colour: Optional[str] = None) -> None:
    """Draw a vertical or horizontal reference line, such as a threshold.

    Thin, dashed and grey by default (``ROLES["reference"]``), so the
    reference is visually distinct from the data.

    :param ax: Matplotlib axes on which to draw the reference line.
    """
    colour = colour or ROLES["reference"]
    drawn = (ax.axvline(x, color=colour, lw=WEIGHTS["reference"], ls=(0, (4, 3)),
                        zorder=0) if x is not None else
             ax.axhline(y, color=colour, lw=WEIGHTS["reference"], ls=(0, (4, 3)),
                        zorder=0))
    if label:
        if x is not None:
            ax.annotate(label, (x, 0.98), xycoords=("data", "axes fraction"),
                        fontsize=TYPE_SCALE["annotation"], color=colour,
                        ha="left", va="top", rotation=90,
                        xytext=(2, -2), textcoords="offset points")
        else:
            ax.annotate(label, (0.99, y), xycoords=("axes fraction", "data"),
                        fontsize=TYPE_SCALE["annotation"], color=colour,
                        ha="right", va="bottom",
                        xytext=(0, 2), textcoords="offset points")
    return drawn


def text_legend(ax, entries: Sequence, x: float = 0.02, y: float = 0.97,
                dy: float = 0.075) -> None:
    """Draw a legend as coloured text labels, without markers or a frame.

    What the published figures do. A framed legend with sample markers costs
    a corner of the axes and adds a box the style has no other boxes to
    match.

    :param ax: matplotlib axes that receives the labels in axes-relative
        coordinates.
    :param entries: ``[(label, colour), ...]``.
    """
    for index, (label, colour) in enumerate(entries):
        ax.text(x, y - index * dy, label, transform=ax.transAxes,
                fontsize=TYPE_SCALE["annotation"], color=colour,
                ha="left", va="top")


def rotate_ticks(ax, degrees: int = 45) -> None:
    """Rotate the x tick labels by ``degrees`` (45 by default), right-aligned.

    :param ax: Matplotlib axes whose x tick labels are rotated.
    """
    for label in ax.get_xticklabels():
        label.set_rotation(degrees)
        label.set_ha("right")
        label.set_rotation_mode("anchor")


def annotate(ax, text: str, *, x: float = 0.02, y: float = 0.97,
             colour: Optional[str] = None, ha: str = "left",
             va: str = "top") -> None:
    """Draw an in-panel text note, such as an n or a correlation coefficient.

    The note is drawn without a frame or box, as in the published figures.

    :param ax: Matplotlib axes that receives the note.
    :param text: annotation text to draw inside the axes.
    """
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=TYPE_SCALE["annotation"],
            color=colour or ax.xaxis.label.get_color(), ha=ha, va=va)


def hide_unused(axes: Iterable) -> None:
    """Turn off axes a grid allocated and no panel filled.

    An empty framed box in a figure sheet reads as a panel that failed to
    draw, which is worse than a gap.

    :param axes: iterable of unused Matplotlib axes to turn off.
    """
    for ax in axes:
        ax.set_axis_off()


#: One condition slot on a condition axis, in inches: every graph with the
#: same number of conditions is the same width (spacr-graphs skill).
SLOT_IN = 0.22
#: Bar, box and violin width as a fraction of a slot; the gap is the rest.
BAR_FRAC = 0.6
#: Above this many points in a group a box (or violin) replaces the bar.
AUTO_N = 20
#: Opacity of every data element: bars, boxes, violins, points, data lines
#: and fills. Text, axes and ticks stay opaque.
DATA_ALPHA = 0.6
#: The series colours after the control grey, in order.
SERIES = (Palette.ACCENT, Palette.TEAL, Palette.NAVY_DARK, Palette.MAGENTA,
          Palette.ACCENT_LIGHT, Palette.AMBER)


def condition_colours(count: int, control=None, colours=None) -> list:
    """One colour per condition, the control (or controls) grey.

    :param count: how many conditions.
    :param control: the index, or indexes, of the control or reference.
    :param colours: colours to use instead of the house series, cycled.
    :returns: ``count`` colour specs.
    """
    pool = list(colours) if colours else list(SERIES)
    out = [pool[index % len(pool)] for index in range(max(0, int(count)))]
    for index in ([control] if isinstance(control, int) else (control or [])):
        if 0 <= int(index) < len(out):
            out[int(index)] = Palette.CONTROL
    return out


def condition_kind(values, many: str = "box") -> str:
    """``'bar'`` for at most :data:`AUTO_N` values, else ``many``.

    :param values: one condition's values.
    :param many: ``'box'`` or ``'violin'``.
    :returns: ``'bar'`` or ``many``.
    """
    import numpy as np

    count = int(np.isfinite(np.asarray(values, float)).sum())
    return many if count > AUTO_N else "bar"


def condition_plot(ax, data, colours=None, control=None, kind: str = "auto",
                   many: str = "box", points: bool = True, seed: int = 0,
                   positions=None) -> list:
    """Draw one element per condition slot in the house style.

    A condition with at most :data:`AUTO_N` values is a bar at its mean with
    every value as a jittered point; one with more is a box (median,
    quartiles, 5-95 % whiskers) or a violin with small points. Every element
    is :data:`BAR_FRAC` of a slot wide and drawn at :data:`DATA_ALPHA`;
    medians, whiskers and error lines are the figure's ink (black in print);
    the control is grey.

    :param ax: the axes.
    :param data: one sequence of values per condition.
    :param colours: colours instead of the house series.
    :param control: the control's index, or indexes.
    :param kind: ``'auto'``, ``'bar'``, ``'box'`` or ``'violin'``.
    :param many: what ``'auto'`` draws above :data:`AUTO_N` values.
    :param points: overlay the values as jittered points.
    :param seed: the jitter's random seed.
    :param positions: x position of each condition; ``0 .. n-1`` by default.
    :returns: the kind drawn for each condition.
    """
    import numpy as np

    import matplotlib

    ink = matplotlib.rcParams["text.color"]
    rng = np.random.default_rng(seed)
    cols = condition_colours(len(data), control, colours)
    places = list(range(len(data))) if positions is None else list(positions)
    width = BAR_FRAC
    used = []
    for slot, values, colour in zip(places, data, cols):
        arr = np.asarray(values, float)
        arr = arr[np.isfinite(arr)]
        chosen = condition_kind(arr, many) if kind == "auto" else kind
        if chosen == "bar":
            ax.bar(slot, arr.mean() if len(arr) else np.nan, width,
                   color=colour, alpha=DATA_ALPHA, linewidth=0, zorder=1)
        elif chosen == "box" and len(arr):
            ax.boxplot([arr], positions=[slot], widths=width,
                       patch_artist=True, showfliers=False, whis=(5, 95),
                       medianprops=dict(color=ink,
                                        linewidth=WEIGHTS["median"]),
                       boxprops=dict(facecolor=colour, edgecolor=colour,
                                     linewidth=WEIGHTS["spine"],
                                     alpha=DATA_ALPHA),
                       whiskerprops=dict(color=ink, linewidth=0.8),
                       capprops=dict(color=ink, linewidth=0.8),
                       manage_ticks=False, zorder=2)
        elif chosen == "violin" and len(arr) > 1:
            drawn = ax.violinplot([arr], positions=[slot], widths=width,
                                  showextrema=False)
            for body in drawn["bodies"]:
                body.set_facecolor(colour)
                body.set_edgecolor("none")
                body.set_alpha(DATA_ALPHA)
            ax.plot([slot - width / 2, slot + width / 2], [np.median(arr)] * 2,
                    color=ink, linewidth=WEIGHTS["median"], zorder=4)
        if points and len(arr) > (1 if chosen == "bar" else 0):
            spread = width * 0.8
            ax.scatter(slot + rng.uniform(-spread / 2, spread / 2, len(arr)),
                       arr, s=4.0 if len(arr) <= AUTO_N else 1.2,
                       color=colour, alpha=DATA_ALPHA, linewidths=0, zorder=3)
        used.append(chosen)
    if places:
        ax.set_xlim(min(places) - 0.5, max(places) + 0.5)
    return used


def size_condition_axes(ax, count: int, *, height: float = 1.6,
                        left: float = 0.75, right: float = 0.25,
                        bottom: float = 0.95, top: float = 0.35) -> None:
    """Size ``ax`` so its plot area is ``count`` slots wide, in inches.

    The figure is resized to hold the axes and the margins, so every graph
    with the same number of conditions has the same width.

    :param ax: the axes, the only one on its figure.
    :param count: how many condition slots.
    :param height: the plot area's height in inches.
    :param left: the left margin in inches.
    :param right: the right margin in inches.
    :param bottom: the bottom margin in inches, which holds the tick labels.
    :param top: the top margin in inches, which holds the descriptor. A
        descriptor wider than the figure is wrapped onto more lines, at
        about 0.55 em per character.
    """
    width = max(1, int(count)) * SLOT_IN
    figure = ax.figure
    total_w, total_h = left + width + right, bottom + height + top
    figure.set_size_inches(total_w, total_h)
    ax.set_position([left / total_w, bottom / total_h, width / total_w,
                     height / total_h])
    ax.set_xlim(-0.5, max(1, int(count)) - 0.5)
    title = ax.get_title()
    if title:
        import textwrap

        chars = int(total_w * 72 / (0.55 * ax.title.get_fontsize()))
        ax.set_title(textwrap.fill(title, max(8, chars)),
                     fontsize=ax.title.get_fontsize(), pad=3.0)


def square_axes(ax):
    """Make a continuous-axis plot (histogram, scatter, line) square.

    :param ax: the axes.
    :returns: ``ax``.
    """
    ax.set_box_aspect(1)
    return ax


def diverging_cmap(low: str = Palette.MAGENTA, high: str = Palette.ACCENT):
    """Pale ``low`` to white to ``high``, for deviation from an expected value.

    Use with :func:`centred_norm` so the expected value is white.

    :param low: the colour below the expected value.
    :param high: the colour above it.
    :returns: a matplotlib colormap.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("spacr_diverging",
                                             [low, (1.0, 1.0, 1.0), high])


def centred_norm(vmin: float, vcenter: float, vmax: float):
    """A colour normalisation with ``vcenter`` at the middle of the map.

    :param vmin: the lowest value.
    :param vcenter: the expected value, drawn white.
    :param vmax: the highest value.
    :returns: a matplotlib ``TwoSlopeNorm``.
    """
    from matplotlib.colors import TwoSlopeNorm

    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


def bracket(ax, x1: float, x2: float, y: float, text: str) -> None:
    """A significance bracket from ``x1`` to ``x2`` with ``text`` above it.

    :param ax: the axes.
    :param x1: the left condition's x position.
    :param x2: the right condition's x position.
    :param y: the bracket's height in data units.
    :param text: the exact p value, for example ``'p = 0.011'``.
    """
    import matplotlib

    low, high = ax.get_ylim()
    drop = 0.02 * (high - low)
    ax.plot([x1, x1, x2, x2], [y - drop, y, y, y - drop],
            color=matplotlib.rcParams["text.color"],
            linewidth=WEIGHTS["reference"])
    ax.text((x1 + x2) / 2, y + drop * 0.3, text, ha="center", va="bottom",
            fontsize=TYPE_SCALE["annotation"])


def figure_problems(figure) -> list:
    """What breaks the house rules in ``figure``, one sentence each.

    Text under 8 pt, text cut off at the canvas edge, overlapping text,
    error bars that are not black (or the screen theme's neutral ink, which
    is the same rule on a dark page), and square-marked axes that are not
    square. Tick labels outside the view limits are not drawn and are not
    checked. An empty list means none were found.

    :param figure: the matplotlib figure; it is drawn to measure the text.
    :returns: one sentence per problem found.
    """
    import matplotlib
    import numpy as np
    from matplotlib.colors import to_rgb

    if not hasattr(figure.canvas, "get_renderer"):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    problems = []
    undrawn = set()
    for ax in figure.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            span = (high - low) * 1e-6
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low - span <= tick.get_loc() <= high + span:
                    undrawn.update((id(tick.label1), id(tick.label2)))
    boxes = []
    for text in figure.findobj(matplotlib.text.Text):
        if (not text.get_visible() or not text.get_text().strip()
                or id(text) in undrawn):
            continue
        if text.get_fontsize() < 8.0 - 1e-6:
            problems.append(f"text {text.get_text()[:30]!r} is "
                            f"{text.get_fontsize():.1f} pt, under 8 pt")
        try:
            box = text.get_window_extent(renderer)
        except Exception:                                      # noqa: BLE001
            continue
        if box.width == 0:
            continue
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            problems.append(f"text {text.get_text()[:30]!r} is cut off")
        boxes.append((text, box))
    for index, (first, a) in enumerate(boxes):
        for second, b in boxes[index + 1:]:
            if (min(a.x1, b.x1) - max(a.x0, b.x0) > 1
                    and min(a.y1, b.y1) - max(a.y0, b.y0) > 1
                    and not (first.get_rotation() % 90
                             and second.get_rotation() % 90)):
                problems.append(f"texts overlap: {first.get_text()[:20]!r} "
                                f"and {second.get_text()[:20]!r}")
    for ax in figure.axes:
        for container in ax.containers:
            if isinstance(container, matplotlib.container.ErrorbarContainer):
                for line in container.lines[2]:
                    for colour in line.get_colors():
                        rgb = np.array(to_rgb(colour))
                        if rgb.max() - rgb.min() > 0.03:
                            problems.append("an error bar is not black")
        aspect = ax.get_box_aspect()
        if aspect is not None and abs(float(aspect) - 1.0) < 1e-9:
            box = ax.get_window_extent(renderer)
            if box.height and abs(box.width / box.height - 1) > 0.02:
                problems.append("a square axes is not square")
    return problems


__all__ = [
    "AUTO_N", "BAR_FRAC", "DATA_ALPHA", "SERIES", "SLOT_IN", "bracket",
    "centred_norm", "condition_colours", "condition_kind", "condition_plot",
    "diverging_cmap", "figure_problems", "size_condition_axes",
    "square_axes",
    "INK_PRINT", "INK_SCREEN", "LINE_KEYS", "TEXT_KEYS", "TRANSPARENT",
    "Palette", "ROLES", "TYPE_SCALE", "WEIGHTS", "annotate", "chosen_ink",
    "chosen_line_ink", "descriptor", "figure_style", "hide_unused",
    "panel_letter", "pipeline_figures", "rc", "reference_line", "resolve_ink",
    "resolve_line_ink", "rotate_ticks", "text_legend", "theme_target",
    "user_overrides",
]
