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
from dataclasses import dataclass
import os
import sys
from typing import Iterable, Optional, Sequence

#: Print ink: the near-black the published figures use for text and axes.
INK_PRINT = "#000000"
#: Screen ink: what the dark GUI needs instead. Not pure white — pure white
#: on a dark ground reads as glare, and the published figures never use a
#: maximal ink either.
INK_SCREEN = "#000000"

#: Transparent, spelled the way matplotlib understands it.
TRANSPARENT = "none"
TICK_ROTATION = 45


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
_LEGACY_ROLES = {
    "data": Palette.GREY,
    "up": Palette.GREEN,
    "down": Palette.RUST,
    "highlight": Palette.BLUE,
    "control_negative": Palette.GREY_DARK,
    "control_positive": Palette.PURPLE,
    "fill": Palette.CORAL,
    "reference": Palette.GREY_DARK,
}


ROLES = {
    "data": Palette.BACKGROUND, "up": Palette.TEAL, "down": Palette.MAGENTA,
    "highlight": Palette.ACCENT, "control_negative": Palette.CONTROL,
    "control_positive": Palette.CONTROL, "fill": Palette.ACCENT_LIGHT,
    "reference": Palette.CONTROL, "primary": Palette.ACCENT,
    "secondary": Palette.TEAL, "tertiary": Palette.NAVY_DARK,
    "control": Palette.CONTROL, "background": Palette.BACKGROUND,
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
    return chosen_ink() or GENERAL_DEFAULTS["foreground"]


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
_SOURCE_PALETTES = ("reference", "tab10", "deep", "colorblind", "muted", "pastel",
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


def _restyle(figure, changed: dict, *, axes=None) -> None:
    """Apply explicit choices to all axes, or only the supplied producer panels.

    Panel-specific choices preserve sibling panels and the shared page ground.
    """
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

    targets = list(figure.get_axes()) if axes is None else list(axes)
    sizes = {"title_size": 0, "label_size": 1, "tick_size": 2,
             "legend_size": 3}
    family = changed.get("font_family")
    if "background" in changed and (axes is None or len(figure.axes) == 1):
        ground = changed["background"]
        figure.patch.set_facecolor(ground)
    suptitle = getattr(figure, "_suptitle", None)
    if axes is None and suptitle is not None and "title_size" in changed:
        suptitle.set_fontsize(float(changed["title_size"]))
    if family:
        parents = [figure] if axes is None else targets
        for parent in parents:
            for text in parent.findobj(lambda artist: hasattr(artist, "set_fontfamily")):
                text.set_fontfamily(family)
    if "tick_size" in changed:
        for ax in targets:
            ax.tick_params(labelsize=float(changed["tick_size"]))
    mapping = (_palette_map(changed["palette"])
               if changed.get("palette") else {})
    preset = SPINE_PRESETS.get(str(changed.get("spines", "")))
    for ax in targets:
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
    from ..figure_font import _default_font_params, use_open_sans_for_figures
    open_sans = use_open_sans_for_figures()
    font_parameters = _default_font_params()
    line_colour = resolve_line_ink(target, ink=picked_ink, line=picked_line)
    ground = GENERAL_DEFAULTS["background"] if ground is None else ground
    params = {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": font_parameters.get("font.family", ["DejaVu Sans"]),
        "font.sans-serif": ([_FIGURE_FAMILY] if open_sans else [])
                           + ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": TYPE_SCALE["tick"],
        "font.weight": "light",
        "axes.labelweight": "light",
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



def panel_letter(ax, letter: str | int, dx: float = -0.16, dy: float = 1.06) -> None:
    """Draw a bold upper-case panel letter, without a period, at the top left.

    Sized from the measured 1.9-2.2x of the axis-label tier.

    :param ax: Matplotlib axes that receives the panel label.
    :param letter: panel identifier, converted to upper case, or a zero-based
        ordinal that continues A through Z, AA through AZ, and so on.
    :raises ValueError: for a negative panel ordinal.
    """
    if isinstance(letter, str):
        label = letter.upper()
    else:
        from operator import index

        ordinal = index(letter) + 1
        if ordinal < 1:
            raise ValueError("Panel ordinal cannot be negative")
        label = ""
        while ordinal:
            ordinal, remainder = divmod(ordinal - 1, 26)
            label = chr(65 + remainder) + label
    ax.text(dx, dy, label, transform=ax.transAxes,
            fontsize=TYPE_SCALE["panel_letter"], fontweight="bold",
            fontfamily="Open Sans", color="#000000",
            va="bottom", ha="left", gid="spacr-panel-letter")


def descriptor(ax, text: str) -> None:
    """Set a short descriptor (two to four lower-case words) as the axes title.

    Not a sentence title. The axis labels carry the content; a descriptor
    only names the condition the panel shows.

    :param ax: Matplotlib axes whose title is set.
    :param text: short descriptor to place above the axes.
    """
    ax.set_title(text, fontsize=TYPE_SCALE["label"], pad=3.0)


def _legend_note(ax, text, colour, kind="note") -> None:
    """Retain a complete annotation for the figure legend without adding floating plot text."""
    entries = tuple(getattr(ax, "_spacr_legend_entries", ()))
    entry = {"label": str(text), "colour": str(colour), "kind": str(kind)}
    if entry not in entries:
        ax._spacr_legend_entries = (*entries, entry)


def reference_line(ax, *, x=None, y=None, label: str = "",
                   colour: Optional[str] = None):
    """Draw a sixty-percent reference line and retain its label in the figure legend.

    :param ax: Matplotlib axes receiving the reference.
    :param x: vertical reference coordinate, or None.
    :param y: horizontal reference coordinate, or None.
    :param label: complete legend label; never drawn over observations.
    :param colour: explicit identity colour, or black reference ink by default.
    :returns: the reference-line artist.
    """
    colour = colour or "#000000"
    options = dict(color=colour, lw=WEIGHTS["reference"], ls=(0, (4, 3)),
                   alpha=DATA_ALPHA, zorder=0, gid="spacr-reference")
    drawn = ax.axvline(x, **options) if x is not None else ax.axhline(y, **options)
    if label:
        _legend_note(ax, label, colour, "reference")
        drawn.set_label(label)
    return drawn


def text_legend(ax, entries: Sequence, x: float = 0.02, y: float = 0.97,
                dy: float = 0.075) -> None:
    """Retain complete colour-key entries in the figure legend, clear of the plotting area.

    :param ax: Matplotlib axes carrying the legend metadata.
    :param entries: ordered label and colour pairs.
    :param x: historical placement argument retained for recipe compatibility.
    :param y: historical placement argument retained for recipe compatibility.
    :param dy: historical spacing argument retained for recipe compatibility.
    """
    for label, colour in entries:
        _legend_note(ax, label, colour, "identity")


def rotate_ticks(ax, degrees: int = TICK_ROTATION) -> None:
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
    """Retain statistical notes in the figure legend without text over data.

    :param ax: Matplotlib axes carrying the note metadata.
    :param text: complete note retained without changing its numerical values.
    :param x: historical position argument retained for recipe compatibility.
    :param y: historical position argument retained for recipe compatibility.
    :param colour: legend-key colour, or the existing text colour.
    :param ha: historical alignment retained for recipe compatibility.
    :param va: historical alignment retained for recipe compatibility.
    """
    _legend_note(ax, text, colour or ax.xaxis.label.get_color())


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


GENERAL_DEFAULTS = {'font_family': 'Open Sans',
 'font_size': 8.0,
 'title_size': 8.0,
 'label_size': 8.0,
 'tick_size': 8.0,
 'palette': 'reference',
 'background': '#FFFFFF',
 'foreground': '#000000',
 'grid': False,
 'grid_colour': '#DDDDDD',
 'grid_width': 0.6,
 'grid_style': '-',
 'spines': 'all',
 'spine_width': 0.6,
 'marker_size': 4.0,
 'line_width': 1.0,
 'dpi': 300,
 'format': 'pdf',
 'tight_layout': True,
 'chrome_colour': '',
 'mark_colouring': 'group',
 'marker_style': 'o',
 'page_shape': 'landscape',
 'legend_size': 8.0,
 'colormap': 'Blues',
 'figure_width': 6.4,
 'figure_height': 4.8,
 'despine_offset': 0.0,
 'also_save': 'none',
 'vector_text': True,
 'error_bars': 'sem',
 'ci_level': 95.0,
 'error_capsize': 1.5,
 'jitter_width': 0.4,
 'point_alpha': 0.6,
 'point_overlay': True}

GRAPH_DEFAULTS = {'volcano': {'threshold_style': '--',
             'threshold_colour': '#8C8C8C',
             'threshold_width': 0.6,
             'label_top_n': 0,
             'annotate': False,
             'split_axis': False,
             'legend': False},
 'plate_heatmap': {'centred': False, 'annotate_cells': False, 'per_row': 2, 'aspect': 'equal'},
 'histogram': {'bins': 'auto',
               'fill_colour': '#0A77B8',
               'edge_colour': 'none',
               'edge_width': 0.0,
               'log_y': False},
 'scatter': {'trend_line': True, 'trend_colour': '#0A77B8'},
 'residuals': {'reference_style': '--', 'reference_colour': '#8C8C8C', 'trend_line': True},
 'qq': {'reference_style': '--', 'reference_colour': '#8C8C8C'},
 'jitter_bar': {}}

_FINALIZING = contextvars.ContextVar("spacr_figure_finalizing", default=())

POLICY_VERSION = "2026-11-20"
DENSE_N = 1000


@dataclass(frozen=True)
class StyleSpec:
    """Immutable appearance and declared semantics of one statistical panel.

    Source observations and calculated statistics are not stored or changed.
    Metadata remains explicit; labels never determine control identity.

    :param policy_version: the recorded presentation-policy version.
    :param target: screen, print, or transparent output.
    :param kind: categorical, continuous, heatmap, image, schematic, axis_overlay, or unspecified.
    :param conditions: number of categorical slots, when known.
    :param condition_labels: explicitly supplied condition identities.
    :param sample_counts: finite observations per experimental unit and condition.
    :param controls: explicit reference-condition indexes.
    :param kinds: automatic representation per condition.
    :param colours: one resolved colour per categorical condition.
    :param roles: immutable semantic-role and colour pairs.
    :param identities: declared identity metadata, without inferred labels.
    :param comparison: declared comparison and image-display metadata.
    :param legend: declared caption, quantities, units, and abbreviation expansions.
    :param slot_inches: physical width allocated to each condition.
    :param alpha: required opacity of data marks.
    :param font_family: required bundled font family.
    :param minimum_font_points: minimum readable text size.
    :param font_weights: immutable body, title, emphasis and panel-letter face weights.
    """

    policy_version: str
    target: str
    kind: str
    conditions: Optional[int]
    condition_labels: tuple
    sample_counts: tuple
    controls: tuple
    kinds: tuple
    colours: tuple
    roles: tuple
    identities: tuple
    comparison: tuple
    legend: tuple
    slot_inches: float = SLOT_IN
    alpha: float = DATA_ALPHA
    font_family: str = "Open Sans"
    minimum_font_points: float = 8.0
    font_weights: tuple = (("body", "light"), ("title", "regular"),
                           ("emphasis", "semibold"), ("panel_letter", "bold"))


    def __post_init__(self) -> None:
        """Freeze direct-constructor sequences and nested metadata independently of caller-owned containers."""
        for name in ("condition_labels", "sample_counts", "controls", "kinds"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("colours", "roles", "identities", "comparison", "legend", "font_weights"):
            object.__setattr__(self, name, _style_metadata(getattr(self, name)))


@dataclass(frozen=True)
class AxesGeometry:
    """Physical plot-area dimensions, independent of any numerical source data.

    :param width_inches: required plot-area width, or None for content-led axes.
    :param height_inches: required plot-area height, or None for content-led axes.
    :param element_width: categorical element width in slot coordinates.
    :param gap: categorical gap in slot coordinates.
    :param aspect: categorical, square, square_cells, or content geometry.
    :param cell_inches: physical heatmap cell size, or None.
    """

    width_inches: Optional[float]
    height_inches: Optional[float]
    element_width: float
    gap: float
    aspect: str
    cell_inches: Optional[float] = None


@dataclass(frozen=True)
class FigureDiagnostics:
    """Applicable checks and actionable findings from the shared finalizer.

    :param policy_version: recorded presentation-policy version, if registered.
    :param problems: findings; an unregistered panel cannot be certified.
    :param checks: checks that actually ran, rather than inferred compliance.
    """

    policy_version: str
    problems: tuple
    checks: tuple

    @property
    def compliant(self) -> bool:
        """Whether every applicable registered-panel check completed without findings."""
        return not self.problems and "completed presentation checks" in self.checks


def _style_metadata(value) -> tuple:
    """Freeze explicit semantic metadata without copying source arrays into a style record."""
    from collections.abc import Mapping

    if value is None:
        return ()
    if isinstance(value, str):
        return (("caption", value),)
    if isinstance(value, Mapping):
        return tuple((str(key), _style_metadata_value(item)) for key, item in value.items())
    return tuple(_style_metadata_value(item) for item in value)


def _style_metadata_value(value):
    """Freeze nested scalar metadata while refusing arbitrary numerical-array payloads."""
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return _style_metadata(value)
    if isinstance(value, (list, tuple)):
        return tuple(_style_metadata_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Figure metadata requires scalar values, mappings, or sequences")


def resolve_style(target="screen", kind=None, *, conditions=None,
                  sample_counts=None, control=None, identities=None,
                  comparison=None, legend=None, policy_version=None) -> StyleSpec:
    """Resolve the one versioned presentation policy without changing scientific values.

    :param target: screen, print, or transparent output target.
    :param kind: declared graph kind; None remains explicitly unspecified.
    :param conditions: slot count or ordered condition labels.
    :param sample_counts: finite observation counts after the existing unit aggregation.
    :param control: explicit reference index, label, or sequence of those.
    :param identities: identity-to-role metadata; names are never guessed.
    :param comparison: comparison provenance and display-only settings.
    :param legend: caption and objective quantity/unit metadata.
    :param policy_version: current policy or legacy appearance for archived recipes.
    :returns: immutable style and semantic specification.
    :raises ValueError: for unsupported versions, targets, or inconsistent counts.
    """
    from operator import index

    version = str(policy_version or POLICY_VERSION)
    if version not in (POLICY_VERSION, "legacy"):
        raise ValueError("Unsupported figure presentation policy")
    target = str(target)
    if target not in ("screen", "print", "transparent"):
        raise ValueError("Figure target must be screen, print, or transparent")
    labels = ()
    if conditions is None:
        count = None
    else:
        try:
            count = index(conditions)
        except TypeError:
            labels = tuple(str(value) for value in conditions)
            count = len(labels)
    counts = () if sample_counts is None else tuple(index(value) for value in sample_counts)
    if count is None and counts:
        count = len(counts)
    if (count is not None and count < 0) or any(value < 0 for value in counts):
        raise ValueError("Figure condition and observation counts cannot be negative")
    if counts and len(counts) != count:
        raise ValueError("Figure observation counts must match its conditions")
    if control is None:
        selected = ()
    elif isinstance(control, str):
        selected = (control,)
    else:
        try:
            selected = (index(control),)
        except TypeError:
            selected = tuple(control)
    controls = []
    for value in selected:
        if isinstance(value, str):
            if value not in labels:
                raise ValueError("Figure reference must name an explicit condition")
            value = labels.index(value)
        value = index(value)
        if count is None or not 0 <= value < count:
            raise ValueError("Figure reference index is outside its conditions")
        if value not in controls:
            controls.append(value)
    categorical = {"auto", "categorical", "bar", "bar_jitter", "jitter_bar", "box",
                   "box_jitter", "jitter_box", "violin", "jitter", "controls", "effect_rank"}
    continuous = {"continuous", "line", "line_std", "scatter", "volcano", "qq",
                  "histogram", "p_histogram", "effect_distribution", "response",
                  "guide_fraction", "agreement", "curve", "sweep", "density"}
    declared = str(kind or "unspecified")
    resolved = ("categorical" if declared in categorical else
                "continuous" if declared in continuous else
                declared if declared in ("heatmap", "image", "schematic", "axis_overlay") else "unspecified")
    roles = dict(_LEGACY_ROLES if version == "legacy" else ROLES)
    colours = list(condition_colours(count or 0, controls, [Palette.ACCENT]))
    bindings = dict(_style_metadata(identities))
    for position, label in enumerate(labels):
        role = bindings.get(label)
        if role is not None:
            if role not in roles:
                raise ValueError("Figure identity must name a supported semantic colour role")
            colours[position] = roles[role]
    for position in controls:
        colours[position] = roles["control_negative"]
    colours = tuple(colours)
    kinds = tuple(_condition_kind_count(value) for value in counts)
    return StyleSpec(version, target, resolved, count, labels, counts, tuple(controls),
                     kinds, colours, tuple(roles.items()), _style_metadata(identities),
                     _style_metadata(comparison), _style_metadata(legend))


def axes_geometry(spec: StyleSpec, *, cell_count=None, panel_height=None) -> AxesGeometry:
    """Return physical geometry for the declared chart without touching its observations.

    Split-axis specifications retain the declared fractional panel heights
    while their combined plot-area height equals the common panel height.

    :param spec: resolved style specification.
    :param cell_count: heatmap (row count, column count), when applicable.
    :param panel_height: plot-area height in inches; defaults to 40 mm.
    :returns: fixed slot, square, square-cell, or content-led geometry.
    :raises ValueError: for nonpositive panel height or invalid heatmap dimensions.
    """
    height = 40.0 / 25.4 if panel_height is None else float(panel_height)
    if height <= 0:
        raise ValueError("Figure plot-area height must be positive")
    if spec.kind == "categorical":
        width = None if spec.conditions is None else spec.conditions * spec.slot_inches
        if dict(spec.comparison).get("orientation") == "horizontal":
            return AxesGeometry(height, width, BAR_FRAC, 1.0 - BAR_FRAC, "categorical")
        return AxesGeometry(width, height, BAR_FRAC, 1.0 - BAR_FRAC, "categorical")
    if spec.kind == "heatmap" and cell_count is not None:
        rows, columns = (int(value) for value in cell_count)
        if rows <= 0 or columns <= 0:
            raise ValueError("Figure heatmap dimensions must be positive")
        cell = height / rows
        return AxesGeometry(columns * cell, height, BAR_FRAC, 1.0 - BAR_FRAC,
                            "square_cells", cell)
    if spec.kind == "continuous":
        fraction = dict(spec.comparison).get("split_fraction")
        if fraction is not None:
            if not 0. < float(fraction) < 1.:
                raise ValueError("Split-axis panel fractions must lie strictly between zero and one")
            return AxesGeometry(height, height * float(fraction), BAR_FRAC, 1.0 - BAR_FRAC, "split")
        return AxesGeometry(height, height, BAR_FRAC, 1.0 - BAR_FRAC, "square")
    return AxesGeometry(None, None, BAR_FRAC, 1.0 - BAR_FRAC, "content")



def _physical_panel_problems(spec, width, height, *, pixel):
    """Observe the common panel dimension without repairing manual geometry.

    The required statistical plot-area height is 40 mm. Horizontal
    categories rotate this dimension; split windows share a 40-mm width
    and retain their independently checked declared height fractions.
    The caller supplies one device pixel in inches, matching the existing
    slot and aspect checks rather than changing their measured margin.
    """
    geometry = axes_geometry(spec)
    if geometry.aspect not in ("categorical", "square", "split"):
        return ()
    across = geometry.aspect == "split" or dict(spec.comparison).get("orientation") == "horizontal"
    measured = width if across else height
    expected = geometry.width_inches if across else geometry.height_inches
    if abs(measured - expected) > pixel:
        return (_replay_message("Use the shared 40 mm panel dimension; preserve explicit manual sizes with a diagnostic"),)
    return ()


def _data_artists(ax) -> tuple:
    """Include visible annotation arrow patches in the actual statistical marks.

    :param ax: axes containing ordinary marks and optional arrow annotations.
    :returns: original artist objects without modifying their values or appearance.
    """
    arrows = tuple(text.arrow_patch for text in ax.texts
                   if text.get_visible() and getattr(text, "arrow_patch", None) is not None)
    return (*ax.lines, *ax.collections, *ax.patches, *ax.images, *arrows)


def _artist_alphas(artist) -> tuple:
    """Read effective Matplotlib alpha; artist alpha overrides stored face RGBA alpha."""
    import numpy as np
    from matplotlib.colors import to_rgba

    alpha = artist.get_alpha()
    if alpha is not None:
        return tuple(np.asarray(alpha, dtype=float).ravel())
    for getter in ("get_facecolors", "get_colors", "get_facecolor", "get_color"):
        if not hasattr(artist, getter):
            continue
        try:
            colour = getattr(artist, getter)()
            values = np.asarray(colour, dtype=float)
            if values.ndim == 2 and values.shape[-1] == 4:
                if not len(values):
                    continue
                return tuple(values[:, 3])
            return (float(to_rgba(colour)[3]),)
        except (TypeError, ValueError):
            continue
    return (1.0,)



def _axis_overlay_parent(ax, spec=None):
    """Resolve an explicitly declared and genuinely shared primary statistical axes.

    :param ax: proposed overlay axes.
    :param spec: overlay specification, or its attached specification.
    :returns: registered primary axes, or None for an invalid or independent overlay.
    """
    item = spec or getattr(ax, "_spacr_style_spec", None)
    if not isinstance(item, StyleSpec) or item.kind != "axis_overlay":
        return None
    comparison = dict(item.comparison)
    index = comparison.get("overlay_of")
    shared = comparison.get("shared_axis")
    if not isinstance(index, int) or not 0 <= index < len(ax.figure.axes) or shared not in ("x", "y"):
        return None
    primary = ax.figure.axes[index]
    declared = getattr(primary, "_spacr_style_spec", None)
    if primary is ax or not isinstance(declared, StyleSpec) or declared.kind not in ("categorical", "continuous"):
        return None
    siblings = ax.get_shared_x_axes() if shared == "x" else ax.get_shared_y_axes()
    return primary if siblings.joined(ax, primary) else None


def _axis_overlays(ax) -> tuple:
    """Find visible overlays actually linked to one registered primary axes.

    :param ax: primary statistical axes.
    :returns: overlay axes whose labels belong to the same physical chart envelope.
    """
    return tuple(candidate for candidate in ax.figure.axes
                 if candidate.get_visible() and _axis_overlay_parent(candidate) is ax)


def _axis_overlay_problems(ax, spec) -> tuple:
    """Reject independent axes disguised as overlays and observe their actual alignment.

    :param ax: proposed linked overlay.
    :param spec: explicit primary index and shared-axis declaration.
    :returns: diagnostics without moving axes or modifying values and limits.
    """
    import numpy as np
    from ..qt.i18n import tr

    primary = _axis_overlay_parent(ax, spec)
    if primary is None:
        return (tr("Link the overlay to a declared primary statistical axis"),)
    dimensions = np.tile(ax.figure.get_size_inches(), 2)
    first = np.asarray(ax.get_position().bounds) * dimensions
    second = np.asarray(primary.get_position().bounds) * dimensions
    if not np.allclose(first, second, rtol=0.0, atol=0.5 / ax.figure.dpi):
        return (tr("Align the linked axes on the same physical plotting area"),)
    return ()


def _axes_layout_bbox(ax, renderer):
    """Measure full visible axis titles as well as the ordinary tight envelope.

    Matplotlib may collapse the long dimension of an axis title for layout.
    Physical packing must include that actual ink, especially for short
    broken-axis windows, without altering any source data or view limits.
    """
    from matplotlib.transforms import Bbox

    boxes = [ax.get_tightbbox(renderer)]
    if ax.axison:
        for text in (ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title):
            if text.get_visible() and text.get_text().strip():
                boxes.append(text.get_window_extent(renderer))
    boxes = [box for box in boxes if box is not None]
    return Bbox.union(boxes) if boxes else None


def _tight_export_artists(figure):
    """Retain default legends and full axis-title ink in tight crops of declared figures.

    Matplotlib always collapses the long dimension of axis labels in its
    axes tight box. Include those actual text artists explicitly while
    keeping every normal extra artist, including external legends.
    Legacy figures without a declaration keep their original crop.
    """
    if not any(isinstance(getattr(ax, "_spacr_style_spec", None), StyleSpec)
               for ax in figure.axes):
        return None
    artists = list(figure.get_default_bbox_extra_artists())
    for ax in figure.axes:
        if ax.get_visible() and ax.axison:
            artists.extend(text for text in (ax.xaxis.label, ax.yaxis.label)
                           if text.get_visible() and text.get_text().strip())
    return list(dict.fromkeys(artists))


def fit_axes_geometry(ax, spec: StyleSpec, *, panel_height=None) -> None:
    """Fit one panel around measured text while preserving its physical plot area.

    Only a single-panel figure is resized here. Multi-panel layout owners use
    axes_geometry to pack their panels together. Data and limits are retained.

    :param ax: Matplotlib axes owned by a producer.
    :param spec: resolved presentation and semantic specification.
    :param panel_height: desired physical axes height in inches.
    """
    figure = ax.figure
    active = [item for item in figure.axes if item.axison and
              getattr(item, "_colorbar", None) is None]
    if active != [ax]:
        return
    cells = ax.images[0].get_array().shape[:2] if spec.kind == "heatmap" and ax.images else None
    geometry = axes_geometry(spec, cell_count=cells, panel_height=panel_height)
    if geometry.width_inches is None or geometry.height_inches is None:
        return
    width, height = geometry.width_inches, geometry.height_inches
    if width <= 0:
        return
    left, right, bottom, top = 0.65, 0.15, 0.65, 0.15
    figure.set_size_inches(left + width + right, bottom + height + top)
    ax.set_position([left / figure.get_size_inches()[0], bottom / figure.get_size_inches()[1],
                     width / figure.get_size_inches()[0], height / figure.get_size_inches()[1]])
    original_canvas = figure.canvas
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(figure)
    try:
        for attempt in range(3):
            renderer = canvas.get_renderer()
            _position_colourbars(ax, renderer)
            from matplotlib.transforms import Bbox

            boxes = [_axes_layout_bbox(ax, renderer)] + [_axes_layout_bbox(item, renderer) for item in _axis_colourbars(ax)]
            tight = Bbox.union([box for box in boxes if box is not None])
            area = ax.get_window_extent(renderer)
            if tight is None:
                break
            padding = 0.04
            dpi = float(figure.dpi)
            measured = (max(padding, (area.x0 - tight.x0) / dpi + padding),
                        max(padding, (tight.x1 - area.x1) / dpi + padding),
                        max(padding, (area.y0 - tight.y0) / dpi + padding),
                        max(padding, (tight.y1 - area.y1) / dpi + padding))
            change = max(abs(first - second) for first, second in
                         zip(measured, (left, right, bottom, top)))
            left, right, bottom, top = measured
            total_width, total_height = left + width + right, bottom + height + top
            figure.set_size_inches(total_width, total_height)
            ax.set_position([left / total_width, bottom / total_height,
                             width / total_width, height / total_height])
            if change * dpi <= 0.5:
                break
        _position_colourbars(ax, canvas.get_renderer())
    finally:
        figure.set_canvas(original_canvas)


def pack_axes_geometry(axes, *, maximum_width=7.0, gap=0.25, padding=0.10,
                       legend_space=False) -> None:
    """Pack declared panels with physical slot widths and aligned plot-area edges.

    The requested width is a packing target. A categorical panel that needs
    more room retains its physical slots and expands the page. All source
    values, statistical calculations and numerical axis limits are retained.
    Linked overlays contribute their actual text bounds to the primary panel
    rather than being packed as a separate chart. Shared horizontal views
    align their left edges across rows, and a visible shared vertical title
    receives its own measured margin.

    :param axes: ordered data axes with attached StyleSpec records.
    :param maximum_width: preferred maximum page width in inches.
    :param gap: spacing between neighboring panel envelopes in inches.
    :param padding: outer page and measured-text padding in inches.
    :param legend_space: reserve an external right column for figure legends.
    """
    axes = list(axes)
    if not axes:
        return
    figure = axes[0].figure
    if any(ax.figure is not figure or not isinstance(getattr(ax, "_spacr_style_spec", None), StyleSpec) for ax in axes):
        raise ValueError("Declare each panel before packing its physical geometry")
    if maximum_width <= 0 or gap < 0 or padding <= 0:
        raise ValueError("Panel packing requires positive page width and padding, and nonnegative spacing")
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    original_canvas = figure.canvas
    canvas = FigureCanvasAgg(figure)
    try:
        for attempt in range(3):
            renderer = canvas.get_renderer()
            panels = []
            for ax in axes:
                spec = ax._spacr_style_spec
                cells = ax.images[0].get_array().shape[:2] if spec.kind == "heatmap" and ax.images else None
                geometry = axes_geometry(spec, cell_count=cells)
                area = ax.get_window_extent(renderer)
                from matplotlib.transforms import Bbox

                _position_colourbars(ax, renderer)
                boxes = ([_axes_layout_bbox(ax, renderer)] + [_axes_layout_bbox(overlay, renderer)
                                                        for overlay in _axis_overlays(ax)]
                         + [_axes_layout_bbox(item, renderer) for item in _axis_colourbars(ax)])
                tight = Bbox.union([box for box in boxes if box is not None])
                width = geometry.width_inches or area.width / figure.dpi
                height = geometry.height_inches or area.height / figure.dpi
                left = max(padding, (area.x0 - tight.x0) / figure.dpi + padding)
                right = max(padding, (tight.x1 - area.x1) / figure.dpi + padding)
                bottom = max(padding, (area.y0 - tight.y0) / figure.dpi + padding)
                top = max(padding, (tight.y1 - area.y1) / figure.dpi + padding)
                panels.append((ax, width, height, left, right, bottom, top))
            aligned = []
            for panel in panels:
                ax, width, height, left, right, bottom, top = panel
                shared = [item[3] for item in panels if ax.get_shared_x_axes().joined(ax, item[0])]
                aligned.append((ax, width, height, max([left] + shared), right, bottom, top))
            panels = aligned
            shared_y = getattr(figure, "_supylabel", None)
            shared_box = (shared_y.get_window_extent(renderer) if shared_y is not None
                          and shared_y.get_visible() and shared_y.get_text().strip() else None)
            shared_space = shared_box.width / figure.dpi + padding if shared_box is not None else 0.
            limit = max(maximum_width, max(width + left + right + 2 * padding for _, width, _, left, right, _, _ in panels))
            rows = [[]]
            used = 0.0
            for panel in panels:
                _, width, _, left, right, _, _ = panel
                span = width + left + right
                if rows[-1] and used + gap + span > limit - 2 * padding:
                    rows.append([])
                    used = 0.0
                used += span + (gap if rows[-1] else 0.0)
                rows[-1].append(panel)
            row_widths = [sum(panel[1] + panel[3] + panel[4] for panel in row) + gap * (len(row) - 1) for row in rows]
            row_heights = [max(panel[2] for panel in row) + max(panel[5] for panel in row) + max(panel[6] for panel in row) for row in rows]
            legend_boxes = [legend.get_window_extent(renderer) for legend in figure.legends] if legend_space else []
            legend_width = max((box.width / figure.dpi for box in legend_boxes), default=0.)
            legend_height = max((box.height / figure.dpi for box in legend_boxes), default=0.)
            total_width = max(row_widths) + 2 * padding + shared_space + (legend_width + gap if legend_boxes else 0.)
            total_height = max(sum(row_heights) + gap * (len(rows) - 1) + 2 * padding,
                               legend_height + 2 * padding,
                               shared_box.height / figure.dpi + 2 * padding if shared_box is not None else 0.)
            before = tuple(tuple(ax.get_position().bounds) for ax in axes)
            figure.set_size_inches(total_width, total_height)
            upper = total_height - padding
            for row, row_height in zip(rows, row_heights):
                top = max(panel[6] for panel in row)
                left_edge = padding + shared_space
                for ax, width, height, left, right, bottom, own_top in row:
                    ax.set_position([(left_edge + left) / total_width,
                                     (upper - top - height) / total_height,
                                     width / total_width, height / total_height])
                    for text in ax.texts:
                        if text.get_gid() == "spacr-panel-letter":
                            text.set_position(((padding - left) / width, 1.06))
                    left_edge += left + width + right + gap
                upper -= row_height + gap
            if shared_box is not None:
                shared_y.set_position(((padding + shared_box.width / figure.dpi / 2.) / total_width, .5))
            for ax in axes:
                _position_colourbars(ax, canvas.get_renderer())
            after = tuple(tuple(ax.get_position().bounds) for ax in axes)
            if before == after:
                break
    finally:
        figure.set_canvas(original_canvas)


def _retain_live_axes_geometry(figure, viewport_size) -> None:
    """Retain a packed physical layout while restoring the interactive viewport.

    :param figure: figure already packed by the shared geometry owner.
    :param viewport_size: original interactive width and height in inches.
    """
    import numpy as np

    packed = np.asarray(figure.get_size_inches(), dtype=float)
    axes = [tuple(np.asarray(ax.get_position().bounds) * np.tile(packed, 2)) for ax in figure.axes]
    legends = [dict(bounds=tuple(np.asarray(legend.get_bbox_to_anchor().bounds) / figure.dpi),
                    location=legend._loc) for legend in figure.legends]
    figure._spacr_live_geometry = dict(size=tuple(packed), axes=axes, legends=legends)
    figure.set_size_inches(*viewport_size, forward=False)
    _apply_live_axes_geometry(figure)


def _apply_live_axes_geometry(figure) -> bool:
    """Place cached physical panels in the current viewport without painting marks.

    Explicit later panel placement releases automatic layout. Manual legend
    placement is retained independently. Data, scales and artist values are
    never modified by a viewport resize.

    :param figure: figure with a retained packed layout, when available.
    :returns: whether the automatic physical layout was applied.
    """
    import numpy as np

    saved = getattr(figure, "_spacr_live_geometry", None)
    if not saved or len(saved["axes"]) != len(figure.axes):
        return False
    previous = saved.get("positions")
    if previous is not None and any(tuple(ax.get_position(original=True).bounds) != position
                                    for ax, position in zip(figure.axes, previous)):
        figure._spacr_live_geometry = None
        return False
    viewport = np.asarray(figure.get_size_inches(), dtype=float)
    if np.any(viewport <= 0):
        return False
    offset = np.maximum((viewport - saved["size"]) / 2., 0.)
    for ax, rect in zip(figure.axes, saved["axes"]):
        coordinates = np.asarray(rect, dtype=float)
        coordinates[:2] += offset
        ax.set_position(coordinates / np.tile(viewport, 2))
    saved["positions"] = [tuple(ax.get_position(original=True).bounds) for ax in figure.axes]
    for legend, record in zip(figure.legends, saved["legends"]):
        current = tuple(np.asarray(legend.get_bbox_to_anchor().bounds) / np.tile(figure.bbox.size, 2))
        if legend._loc != record["location"] or record.get("position") is not None and not np.allclose(current, record["position"], rtol=0, atol=1e-12):
            continue
        coordinates = np.asarray(record["bounds"], dtype=float)
        coordinates[:2] += offset
        record["position"] = tuple(coordinates / np.tile(viewport, 2))
        legend.set_bbox_to_anchor(record["position"], transform=figure.transFigure)
    return True



@contextlib.contextmanager
def _file_axes_geometry(figure):
    """Temporarily contain automatically packed live panels in the output page.

    Interactive viewports can be smaller than the retained physical layout.
    Expand only those automatic pages, preserving larger viewport dimensions.
    Explicit panel placement or page dimensions release the retained layout.
    Restore the viewport, panel and legend geometry, layout metadata and live
    validation findings even when writing fails. Source values and axis limits are never changed.

    :param figure: figure whose live viewport may contain a packed layout.
    :yields: whether the file page was expanded.
    """
    import copy
    import numpy as np

    if not _apply_live_axes_geometry(figure):
        yield False
        return
    saved = figure._spacr_live_geometry
    viewport = np.asarray(figure.get_size_inches(), dtype=float)
    page = np.maximum(viewport, saved["size"])
    if np.array_equal(page, viewport):
        yield False
        return
    layout = copy.deepcopy(saved)
    missing = object()
    diagnostics = getattr(figure, "_spacr_style_diagnostics", missing)
    positions = [(ax.get_position(original=True).frozen(), ax.get_position().frozen())
                 for ax in figure.axes]
    legends = [(legend, legend.get_bbox_to_anchor().transformed(
                    figure.transFigure.inverted()).frozen()) for legend in figure.legends]
    try:
        figure.set_size_inches(*page, forward=False)
        _apply_live_axes_geometry(figure)
        yield True
    finally:
        figure.set_size_inches(*viewport, forward=False)
        for ax, (original, active) in zip(figure.axes, positions):
            ax.set_position(original, which="original")
            ax.set_position(active, which="active")
        for legend, bounds in legends:
            legend.set_bbox_to_anchor(bounds, transform=figure.transFigure)
        figure._spacr_live_geometry = layout
        if diagnostics is missing:
            figure.__dict__.pop("_spacr_style_diagnostics", None)
        else:
            figure._spacr_style_diagnostics = diagnostics


def _figure_text_role(text, ax=None) -> str:
    """Resolve declared body, heading, emphasis and panel-letter font roles.

    :param text: actual Matplotlib text artist.
    :param ax: owning data axes, if the text belongs to a panel.
    :returns: one of the shared specification font-weight roles.
    """
    if text.get_gid() == "spacr-panel-letter":
        return "panel_letter"
    if text.get_gid() == "spacr-emphasis":
        return "emphasis"
    figure = text.figure
    titles = [ax.title, ax._left_title, ax._right_title] if ax is not None else []
    legends = list(figure.legends) if figure is not None else []
    if ax is not None and ax.get_legend() is not None:
        legends.append(ax.get_legend())
    titles.extend(legend.get_title() for legend in legends)
    titles.append(getattr(figure, "_suptitle", None))
    return "title" if text in titles else "body"


def _line_origin_limits(limits, data_bounds) -> tuple:
    """Include zero in a linear line view without clipping negative values.

    :param limits: existing padded lower and upper display bounds.
    :param data_bounds: observed lower and upper data extents, including intervals.
    :returns: an honest origin with existing data-side padding retained.
    """
    import math

    low, high = (float(value) for value in limits)
    bottom, top = (float(value) for value in data_bounds)
    if not all(math.isfinite(value) for value in (low, high, bottom, top)):
        return low, high
    return (0. if bottom >= 0 else min(0., low), max(0., high) if top < 0 else high)


def _marker_display_bounds(ax):
    """Yield actual display bounds of visible point marks, including their edge stroke.

    This read-only observation uses marker paths and point sizes, retaining
    all source coordinates, collection masks and manually selected styles.
    """
    import numpy as np
    from matplotlib.collections import PathCollection
    from matplotlib.markers import MarkerStyle
    from matplotlib.transforms import Bbox

    scale = ax.figure.dpi / 72.
    for artist in ax.collections:
        if not isinstance(artist, PathCollection) or not artist.get_visible():
            continue
        offsets = np.ma.asarray(artist.get_offsets())
        centres = artist.get_offset_transform().transform(offsets.filled(np.nan))
        paths = artist.get_paths()
        sizes = artist.get_sizes()
        widths = artist.get_linewidths()
        if not len(centres) or not paths or not len(sizes):
            continue
        for index, centre in enumerate(centres):
            if not np.isfinite(centre).all():
                continue
            path = paths[index % len(paths)].get_extents()
            diameter = np.sqrt(max(0., sizes[index % len(sizes)])) * scale
            stroke = widths[index % len(widths)] * scale / 2. if len(widths) else 0.
            yield Bbox.from_extents(centre[0] + path.x0 * diameter - stroke,
                centre[1] + path.y0 * diameter - stroke,
                centre[0] + path.x1 * diameter + stroke,
                centre[1] + path.y1 * diameter + stroke)
    for artist in ax.lines:
        if not artist.get_visible() or artist.get_marker() in ('None', None, '', ' '):
            continue
        marker = MarkerStyle(artist.get_marker())
        path = marker.get_path().transformed(marker.get_transform()).get_extents()
        centres = artist.get_transform().transform(artist.get_xydata())
        diameter = artist.get_markersize() * scale
        stroke = artist.get_markeredgewidth() * scale / 2.
        for centre in centres:
            if np.isfinite(centre).all():
                yield Bbox.from_extents(centre[0] + path.x0 * diameter - stroke,
                    centre[1] + path.y0 * diameter - stroke,
                    centre[0] + path.x1 * diameter + stroke,
                    centre[1] + path.y1 * diameter + stroke)


def _pad_marker_limits(ax, *, clearance=.5):
    """Expand default views until physical point marks clear the frame without changing data.

    :param ax: axes whose final physical geometry has already been packed.
    :param clearance: empty distance in points beyond the outer marker stroke.
    :returns: whether either default view was expanded.
    """
    from matplotlib.transforms import Bbox

    changed = False
    gap = float(clearance) * ax.figure.dpi / 72.
    for unused in range(4):
        boxes = tuple(_marker_display_bounds(ax))
        if not boxes:
            break
        marks = Bbox.union(boxes)
        panel = ax.bbox
        bounds = (min(panel.x0, marks.x0-gap), min(panel.y0, marks.y0-gap),
                  max(panel.x1, marks.x1+gap), max(panel.y1, marks.y1+gap))
        if bounds == tuple(panel.extents):
            break
        lower, upper = ax.transData.inverted().transform((bounds[:2], bounds[2:]))
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(lower[0] if bounds[0] < panel.x0 else xlim[0],
                    upper[0] if bounds[2] > panel.x1 else xlim[1])
        ax.set_ylim(lower[1] if bounds[1] < panel.y0 else ylim[0],
                    upper[1] if bounds[3] > panel.y1 else ylim[1])
        changed = True
    return changed


def _apply_line_origin(ax, spec) -> None:
    """Apply the shared linear line origin before later explicit manual edits.

    :param ax: Matplotlib axes containing the unchanged data artists.
    :param spec: declared comparison, including optional explicit view metadata.
    """
    comparison = dict(spec.comparison)
    if (comparison.get("graph_type") not in ("line", "line_std", "sweep")
            or comparison.get("cut_axis_reason") or comparison.get("manual_limits") is not None
            or ax.get_yscale() != "linear" or ax.yaxis_inverted()):
        return
    current = ax.get_ylim()
    limits = _line_origin_limits(current, (ax.dataLim.y0, ax.dataLim.y1))
    if limits != current and limits[0] != limits[1]:
        ax.set_ylim(*limits)


def apply_axes_style(ax, spec: StyleSpec, *, layout=True) -> None:
    """Apply producer defaults once, before subsequent manual figure edits.

    This is separate from finalization, which validates without restyling.
    Image data, normalization, statistical results and numerical limits stay
    unchanged. Explicit control identities are carried in the specification.
    Stored manual choices apply after producer defaults and remain subject
    to diagnostics rather than being silently reset.

    :param ax: newly drawn Matplotlib axes.
    :param spec: immutable resolved presentation specification.
    :param layout: fit single-panel physical geometry when true.
    """
    from matplotlib.container import ErrorbarContainer
    from matplotlib.text import Text
    from ..figure_font import use_open_sans_for_figures

    use_open_sans_for_figures()
    ax._spacr_style_spec = spec
    if spec.kind not in ("image", "schematic"):
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            size = max([spec.minimum_font_points] +
                       [tick.label1.get_fontsize() for tick in axis.get_major_ticks()])
            ax.tick_params(axis=name, which="both", labelsize=size,
                           direction="out", length=2.5, width=WEIGHTS["spine"])
        for artist in _data_artists(ax):
            artist.set_alpha(spec.alpha)
        for container in ax.containers:
            if isinstance(container, ErrorbarContainer):
                for cap in container.lines[1]:
                    cap.set_color("#000000")
                    cap.set_markeredgecolor("#000000")
                    cap.set_markerfacecolor("#000000")
                for collection in container.lines[2]:
                    collection.set_color("#000000")
        for spine in ax.spines.values():
            spine.set_visible(spec.kind != "heatmap")
            spine.set_linewidth(WEIGHTS["spine"])
        ax.grid(False)
    if spec.kind == "categorical":
        horizontal = dict(spec.comparison).get("orientation") == "horizontal"
        automatic = ax.get_autoscalex_on() if horizontal else ax.get_autoscaley_on()
        if automatic:
            ax.use_sticky_edges = False
            axis = "x" if horizontal else "y"
            ax.margins(**{axis: max(0.08, ax.margins()[0 if horizontal else 1])})
    font_weights = dict(spec.font_weights)
    for text in ax.findobj(Text):
        text.set_fontfamily(spec.font_family)
        text.set_fontsize(max(spec.minimum_font_points, text.get_fontsize()))
        if text.get_gid() == "spacr-panel-letter":
            text.set_fontsize(TYPE_SCALE["panel_letter"])
            text.set_fontweight(font_weights["panel_letter"])
        else:
            text.set_fontweight(font_weights[_figure_text_role(text, ax)])
    if spec.kind == "continuous":
        ax.set_box_aspect(1)
    elif spec.kind == "heatmap":
        ax.set_aspect("equal", adjustable="box")
    _apply_line_origin(ax, spec)
    graph_type = dict(spec.comparison).get("graph_type") or getattr(ax.figure, "_spacr_graph_kind", None)
    changes = _preference_deltas(graph_type)
    if changes:
        _restyle(ax.figure, changes, axes=[ax])
    if layout:
        fit_axes_geometry(ax, spec)


def _status_panel(ax, message, *, target=None):
    """Draw a readable refusal or empty-state sentence without inventing chart semantics.

    :param ax: newly allocated empty axes.
    :param message: status sentence supplied by the caller.
    :param target: explicit figure target, or the current resolved target.
    :returns: the attached shared status specification.
    """
    target = target or theme_target()
    ax.figure.set_facecolor(GENERAL_DEFAULTS["background"])
    ax.figure.patch.set_alpha(1.0)
    ax.set_axis_off()
    ax.set_facecolor(GENERAL_DEFAULTS["background"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(.5, .5, str(message), transform=ax.transAxes, ha="center", va="center", wrap=True,
            fontsize=TYPE_SCALE["label"], color=resolve_ink(target), gid="spacr-status")
    specification = resolve_style(target, "schematic", comparison={"status": True, "graph_type": "status"},
                                  legend={"caption": str(message)})
    apply_axes_style(ax, specification, layout=False)
    _legend_note(ax, message, resolve_ink(target), "caption")
    ax.figure._spacr_caption = str(message)
    return specification


def _text_appearance(text) -> dict:
    """Record actual text properties without storing a machine-specific font path."""
    from matplotlib.colors import to_rgba

    return dict(text=text.get_text(), family=list(text.get_fontfamily()),
                size=text.get_fontsize(), weight=text.get_fontweight(),
                style=text.get_fontstyle(), stretch=text.get_fontproperties().get_stretch(),
                colour=list(to_rgba(text.get_color())), alpha=text.get_alpha(),
                rotation=text.get_rotation(), ha=text.get_ha(), va=text.get_va(),
                visible=text.get_visible())


def _appearance_value(value):
    """Normalize nested numerical style values without changing general recipe serialization."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return _appearance_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_appearance_value(item) for item in value]
    return value


def _colour_norm_appearance(norm) -> dict:
    """Record exact constructor parameters for supported standard Matplotlib normalizations."""
    from matplotlib import colors

    kind = type(norm).__name__
    supported = ("Normalize", "TwoSlopeNorm", "CenteredNorm", "PowerNorm", "LogNorm",
                 "SymLogNorm", "BoundaryNorm", "NoNorm")
    if kind not in supported or type(norm) is not getattr(colors, kind):
        return dict(kind=kind, portable=False)
    parameters = dict(vmin=None if norm.vmin is None else float(norm.vmin),
                      vmax=None if norm.vmax is None else float(norm.vmax), clip=bool(norm.clip))
    if kind == "TwoSlopeNorm":
        parameters["vcenter"] = float(norm.vcenter)
    elif kind == "CenteredNorm":
        parameters = dict(vcenter=float(norm.vcenter), halfrange=None if norm.halfrange is None else float(norm.halfrange), clip=bool(norm.clip))
    elif kind == "PowerNorm":
        parameters["gamma"] = float(norm.gamma)
    elif kind == "SymLogNorm":
        parameters.update(linthresh=float(norm.linthresh), linscale=float(norm._scale.linscale), base=float(norm._scale.base))
    elif kind == "BoundaryNorm":
        parameters = dict(boundaries=norm.boundaries.tolist(), ncolors=int(norm.Ncmap), clip=bool(norm.clip), extend=norm.extend)
    return dict(kind=kind, portable=True, parameters=parameters)


def _restore_colour_norm(saved):
    """Recreate a supported standard normalization without importing or guessing custom transform code."""
    from matplotlib import colors

    supported = ("Normalize", "TwoSlopeNorm", "CenteredNorm", "PowerNorm", "LogNorm",
                 "SymLogNorm", "BoundaryNorm", "NoNorm")
    if not saved.get("portable") or saved.get("kind") not in supported:
        return None
    parameters = dict(saved["parameters"])
    if saved["kind"] == "TwoSlopeNorm":
        clip = parameters.pop("clip")
        result = colors.TwoSlopeNorm(**parameters)
        result.clip = clip
        return result
    return getattr(colors, saved["kind"])(**parameters)


def _artist_appearance(artist) -> dict:
    """Capture presentation properties without copying data or artist geometry."""
    import numpy as np
    from matplotlib.colors import to_rgba

    result = dict(type=type(artist).__name__, alpha=_appearance_value(artist.get_alpha()),
                  visible=artist.get_visible(), zorder=artist.get_zorder())
    if getattr(artist, "_spacr_numeric_cells", False):
        result["numeric_cells"] = True
    properties = ("sizes", "rasterized", "interpolation", "gid", "linewidth", "linewidths", "linestyle", "linestyles", "hatch",
                  "marker", "markersize", "markeredgewidth", "fill")
    colours = ("color", "facecolor", "edgecolor", "markerfacecolor", "markeredgecolor")
    for key in properties + colours:
        getter = getattr(artist, "get_" + key, None)
        if getter is None:
            continue
        value = getter()
        if key in colours and not isinstance(value, np.ndarray):
            if not isinstance(value, str) or not value.startswith("#"):
                value = list(to_rgba(value))
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = _appearance_value(value)
    if hasattr(artist, "_us_linestyles"):
        result["linestyle"] = _appearance_value(artist._us_linestyles)
        result.pop("linestyles", None)
    elif getattr(artist, "is_dashed", lambda: False)():
        result["linestyle"] = _appearance_value(artist._unscaled_dash_pattern)
    from matplotlib.collections import PathCollection

    if isinstance(artist, PathCollection):
        result["marker_paths"] = [dict(vertices=path.vertices.tolist(),
            codes=None if path.codes is None else path.codes.tolist()) for path in artist.get_paths()]
    if hasattr(artist, "get_cmap") and artist.get_cmap() is not None:
        ramp = artist.get_cmap()
        result.update(cmap={"name": ramp.name,
                            "colours": ramp(np.linspace(0.0, 1.0, ramp.N)).tolist(),
                            "bad": ramp.get_bad().tolist(), "under": ramp.get_under().tolist(),
                            "over": ramp.get_over().tolist()}, clim=list(artist.get_clim()),
                      norm=_colour_norm_appearance(artist.norm))
    return result


def _legend_appearance(legend) -> dict:
    """Capture the actual legend keys, text, frame and figure-relative anchor independently of source data."""
    figure = legend.figure
    frame = legend.get_frame()
    box = frame.get_boxstyle()
    attributes = ("borderpad", "borderaxespad", "labelspacing", "handlelength", "handleheight",
                  "handletextpad", "columnspacing", "markerscale", "numpoints", "scatterpoints")
    return dict(labels=[text.get_text() for text in legend.get_texts()],
                texts=[_text_appearance(text) for text in legend.get_texts()],
                title=_text_appearance(legend.get_title()),
                handles=[None if item is None else _artist_appearance(item)
                         for item in legend.legend_handles],
                anchor=list(legend.get_bbox_to_anchor().transformed(figure.transFigure.inverted()).bounds),
                loc=_appearance_value(legend._loc), ncols=legend._ncols, mode=legend._mode,
                alignment=legend.get_alignment(), fontsize=legend._fontsize,
                scatteryoffsets=legend._scatteryoffsets.tolist(),
                options={name: getattr(legend, name) for name in attributes},
                visible=legend.get_visible(), zorder=legend.get_zorder(),
                frameon=legend.get_frame_on(), frame=_artist_appearance(frame),
                boxstyle=dict(name=type(box).__name__.lower(), parameters=_appearance_value(vars(box))),
                shadow=_appearance_value(legend.shadow))


def _replay_message(message):
    """Translate replay diagnostics in spaCR and retain readable text in standalone scripts."""
    if str(globals().get("__package__") or "").startswith("spacr."):
        from ..qt.i18n import tr

        return tr(message)
    return message


def _restore_legend_appearance(owner, saved) -> tuple:
    """Rebuild recorded vector legend keys without adding any artists to the data axes.

    :param owner: the original axes or enclosing figure.
    :param saved: an additive version-one legend record.
    :returns: findings for unsupported custom legend handles.
    """
    from matplotlib.collections import PathCollection
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.path import Path
    from matplotlib.transforms import IdentityTransform

    handles = []
    problems = []
    for record in saved["handles"]:
        kind = record.get("type") if record is not None else None
        if kind == "Line2D":
            handle = Line2D([0., 1.], [0., 0.])
        elif kind == "Rectangle":
            handle = Rectangle((0., 0.), 1., 1.)
        elif kind == "PathCollection":
            paths = [Path(item["vertices"], item["codes"]) for item in record.get("marker_paths", ())]
            handle = PathCollection(paths)
            handle.set_transform(IdentityTransform())
        else:
            handle = Line2D([], [], visible=False)
            problems.append(_replay_message("Captured appearance does not support this custom legend key"))
        if record is not None and kind in ("Line2D", "Rectangle", "PathCollection"):
            problems.extend(_restore_artist_appearance(handle, record))
        handles.append(handle)
    figure = owner.figure if hasattr(owner, "transAxes") else owner
    first = saved["texts"][0] if saved["texts"] else saved["title"]
    options = dict(saved["options"])
    loc = saved["loc"]
    if isinstance(loc, list):
        loc = tuple(loc)
    prop = FontProperties(family=first["family"], size=saved["fontsize"], weight=first["weight"],
                          style=first["style"], stretch=first["stretch"])
    legend = owner.legend(handles, saved["labels"], loc=loc, bbox_to_anchor=saved["anchor"],
                          bbox_transform=figure.transFigure, ncols=saved["ncols"], mode=saved["mode"],
                          alignment=saved["alignment"], prop=prop, title=saved["title"]["text"],
                          scatteryoffsets=saved["scatteryoffsets"], frameon=saved["frameon"],
                          shadow=saved["shadow"], **options)
    for target, record in zip(legend.legend_handles, saved["handles"]):
        if target is not None and record is not None and record["type"] in ("Line2D", "Rectangle", "PathCollection"):
            problems.extend(_restore_artist_appearance(target, record))
    for text, record in zip(legend.get_texts(), saved["texts"]):
        _restore_text_appearance(text, record)
    _restore_text_appearance(legend.get_title(), saved["title"])
    problems.extend(_restore_artist_appearance(legend.get_frame(), saved["frame"]))
    legend.get_frame().set_boxstyle(saved["boxstyle"]["name"], **saved["boxstyle"]["parameters"])
    legend.set_visible(saved["visible"])
    legend.set_zorder(saved["zorder"])
    return tuple(problems)


def _capture_appearance(figure) -> dict:
    """Freeze actual manual appearance separately from the immutable chart policy.

    Artist styles are guarded by their ordered type signatures during replay.
    Numerical coordinates, source observations and statistics remain in their
    existing recipe and table owners.
    """
    from matplotlib.colors import to_rgba

    panels = []
    for ax in figure.axes:
        panel = dict(rect=list(ax.get_position().bounds), facecolor=list(to_rgba(ax.get_facecolor())),
                     visible=ax.get_visible(), axison=ax.axison,
                     aspect=ax.get_aspect(), box_aspect=ax.get_box_aspect(), adjustable=ax.get_adjustable(), anchor=ax.get_anchor(),
                     title=_text_appearance(ax.title), xlabel=_text_appearance(ax.xaxis.label),
                     ylabel=_text_appearance(ax.yaxis.label),
                     xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                     xscale=ax.get_xscale(), yscale=ax.get_yscale(),
                     x_autoscale=ax.get_autoscalex_on(), y_autoscale=ax.get_autoscaley_on(),
                     title_position=list(ax.title.get_position()), title_auto=bool(ax._autotitlepos),
                     side_titles={side: dict(_text_appearance(text), position=list(text.get_position()))
                                  for side, text in (("left", ax._left_title), ("right", ax._right_title))},
                     texts=[_text_appearance(text) for text in ax.texts],
                     spines={name: dict(colour=list(spine.get_edgecolor()),
                              width=spine.get_linewidth(), visible=spine.get_visible())
                             for name, spine in ax.spines.items()})
        for name in ("lines", "collections", "patches", "images"):
            panel[name] = [_artist_appearance(artist) for artist in getattr(ax, name)]
        panel["legend"] = _legend_appearance(ax.get_legend()) if ax.get_legend() is not None else None
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            ticks = []
            for tick in axis.get_major_ticks():
                ticks.append(dict(label1=_text_appearance(tick.label1), label2=_text_appearance(tick.label2),
                                  tick1=_artist_appearance(tick.tick1line), tick2=_artist_appearance(tick.tick2line)))
            panel[name + "ticks"] = ticks
            panel[name + "offset"] = _text_appearance(axis.offsetText)
        panels.append(panel)
    try:
        size = figure.get_size_inches().tolist()
    except (OSError, ValueError, TypeError):
        size = None
    return dict(schema=1, policy_version=POLICY_VERSION, facecolor=list(to_rgba(figure.get_facecolor())),
                size=size, dpi=float(figure.dpi), axes=panels,
                figure_legends=[_legend_appearance(legend) for legend in figure.legends],
                figure_texts=[dict(appearance=_text_appearance(text),
                    position=figure.transFigure.inverted().transform(text.get_transform().transform(text.get_position())).tolist(),
                    role=next((name for name in ("_suptitle", "_supxlabel", "_supylabel")
                               if getattr(figure, name, None) is text), None)) for text in figure.texts])


def _restore_text_appearance(text, saved, *, require_text=False) -> bool:
    """Restore a captured font and ink while keeping locator-generated labels intact."""
    if require_text and text.get_text() != saved["text"]:
        return False
    text.set_fontfamily(saved["family"])
    text.set_fontsize(saved["size"])
    text.set_fontweight(saved["weight"])
    text.set_fontstyle(saved["style"])
    text.set_fontstretch(saved["stretch"])
    text.set_color(saved["colour"])
    text.set_alpha(saved["alpha"])
    text.set_rotation(saved["rotation"])
    text.set_horizontalalignment(saved["ha"])
    text.set_verticalalignment(saved["va"])
    text.set_visible(saved["visible"])
    return True


def _restore_artist_appearance(artist, saved) -> tuple:
    """Restore only supported style setters, preserving every data coordinate and no-ink colour."""
    import numpy as np

    problems = []
    for key, value in saved.items():
        if key in ("color", "facecolor", "edgecolor", "markerfacecolor", "markeredgecolor") and isinstance(value, (list, tuple)):
            rgba = np.asarray(value)
            if key == "facecolor" and saved.get("fill") is False and not rgba.size:
                value = "none"
            if rgba.ndim and rgba.shape[-1] == 4 and rgba.size and np.issubdtype(rgba.dtype, np.number) and np.all(rgba == 0.):
                value = "none"
        if key == "type":
            continue
        if key == "norm":
            norm = _restore_colour_norm(value)
            if norm is None:
                problems.append(_replay_message("Captured appearance does not support this colour normalization"))
            else:
                artist.set_norm(norm)
            continue
        if key == "marker_paths":
            from matplotlib.path import Path

            artist.set_paths([Path(record["vertices"], record["codes"]) for record in value])
            continue
        if key == "numeric_cells":
            artist._spacr_numeric_cells = bool(value)
            continue
        if key == "linestyle" and isinstance(value, list):
            if value and isinstance(value[0], list):
                value = [(offset, None if pattern is None else tuple(pattern)) for offset, pattern in value]
            else:
                value = (value[0], None if value[1] is None else tuple(value[1]))
        if key == "cmap" and isinstance(value, dict):
            from matplotlib.colors import ListedColormap

            ramp = ListedColormap(value["colours"], name=value["name"])
            ramp.set_bad(value["bad"])
            ramp.set_under(value["under"])
            ramp.set_over(value["over"])
            artist.set_cmap(ramp)
            continue
        setter = getattr(artist, "set_" + key, None)
        if setter is not None:
            setter(value)
    return tuple(problems)


def _restore_appearance(figure, saved) -> tuple:
    """Replay a versioned captured appearance without applying current defaults.

    Missing or incompatible records remain visible in the returned findings.
    Legacy recipes without this record keep their historical rendering path.
    """
    if not saved:
        return ()
    problems = []
    if saved.get("schema") != 1:
        problems.append("Unsupported captured appearance schema")
    elif len(figure.axes) != len(saved["axes"]):
        problems.append("Captured appearance does not match the recreated panel count")
    else:
        figure.set_facecolor(saved["facecolor"])
        if saved["size"] is not None:
            figure.set_size_inches(*saved["size"])
        else:
            problems.append("Physical figure size was unavailable during capture")
        figure.set_dpi(saved["dpi"])
        for ax, panel in zip(figure.axes, saved["axes"]):
            ax.set_position(panel["rect"])
            ax.set_facecolor(panel["facecolor"])
            ax.set_visible(panel["visible"])
            ax.set_axis_on() if panel["axison"] else ax.set_axis_off()
            if "box_aspect" in panel:
                ax.set_box_aspect(panel["box_aspect"])
            ax.set_aspect(panel["aspect"], adjustable=panel["adjustable"], anchor=panel["anchor"])
            for name in ("x", "y"):
                scale = panel.get(name + "scale")
                if scale is not None and getattr(ax, "get_" + name + "scale")() != scale:
                    getattr(ax, "set_" + name + "scale")(scale)
                if name + "lim" in panel:
                    getattr(ax, "set_" + name + "lim")(panel[name + "lim"], auto=panel.get(name + "_autoscale", False))
            for side, record in panel.get("side_titles", {}).items():
                text = ax.set_title(record["text"], loc=side)
                _restore_text_appearance(text, record)
                text.set_position(record["position"])
            if "title_position" in panel:
                ax.title.set_position(panel["title_position"])
                ax._autotitlepos = panel["title_auto"]
            for name, text in (("title", ax.title), ("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)):
                text.set_text(panel[name]["text"])
                _restore_text_appearance(text, panel[name])
            if len(ax.texts) != len(panel["texts"]):
                problems.append("Captured appearance does not match the recreated annotations")
            else:
                for text, record in zip(ax.texts, panel["texts"]):
                    if not _restore_text_appearance(text, record, require_text=True):
                        problems.append("Captured annotation identity changed during replay")
            for name, record in panel["spines"].items():
                ax.spines[name].set_edgecolor(record["colour"])
                ax.spines[name].set_linewidth(record["width"])
                ax.spines[name].set_visible(record["visible"])
            for name in ("lines", "collections", "patches", "images"):
                artists = list(getattr(ax, name))
                records = panel[name]
                if [type(artist).__name__ for artist in artists] != [record["type"] for record in records]:
                    problems.append("Captured appearance does not match recreated " + name)
                    continue
                for artist, record in zip(artists, records):
                    problems.extend(_restore_artist_appearance(artist, record))
            for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
                records = panel[name + "ticks"]
                ticks = axis.get_major_ticks(max(len(records), 1))
                for index, tick in enumerate(ticks):
                    if not records:
                        break
                    record = records[min(index, len(records) - 1)]
                    for side in ("label1", "label2"):
                        _restore_text_appearance(getattr(tick, side), record[side])
                    for side in ("tick1", "tick2"):
                        problems.extend(_restore_artist_appearance(getattr(tick, side + "line"), record[side]))
                _restore_text_appearance(axis.offsetText, panel[name + "offset"])
            if "legend" in panel:
                previous = ax.get_legend()
                if previous is not None:
                    previous.remove()
                if panel["legend"] is not None:
                    problems.extend(_restore_legend_appearance(ax, panel["legend"]))
        if "figure_legends" in saved:
            for legend in list(figure.legends):
                legend.remove()
            for record in saved["figure_legends"]:
                problems.extend(_restore_legend_appearance(figure, record))
        if "figure_texts" in saved:
            for text in list(figure.texts):
                text.remove()
            for name in ("_suptitle", "_supxlabel", "_supylabel"):
                setattr(figure, name, None)
            for record in saved["figure_texts"]:
                text = figure.text(*record["position"], record["appearance"]["text"])
                _restore_text_appearance(text, record["appearance"])
                if record["role"] is not None:
                    setattr(figure, record["role"], text)
    figure._spacr_replay_problems = tuple(dict.fromkeys(problems))
    return figure._spacr_replay_problems


def _register_replay_fonts(directory, records) -> None:
    """Register bundled replay fonts after checking their names and recorded hashes."""
    import hashlib
    from pathlib import Path
    from matplotlib import font_manager

    for record in records:
        name = record["name"]
        if Path(name).name != name or not name.endswith((".ttf", ".otf")):
            raise ValueError("Invalid captured font filename")
        path = Path(directory) / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Captured font does not match its recorded hash")
        font_manager.fontManager.addfont(str(path))


def _colour_contrast(first, second) -> float:
    """Return WCAG luminance contrast for two already composited opaque colours."""
    import numpy as np
    from matplotlib.colors import to_rgb

    channels = np.asarray([to_rgb(first), to_rgb(second)])
    linear = np.where(channels <= 0.04045, channels / 12.92,
                      ((channels + 0.055) / 1.055) ** 2.4)
    levels = linear @ np.asarray([0.2126, 0.7152, 0.0722])
    return float((levels.max() + 0.05) / (levels.min() + 0.05))


def _errorbar_colours(ax):
    """Yield every actual error-bar cap and segment colour for exact black-ink validation."""
    from matplotlib.container import ErrorbarContainer

    for container in ax.containers:
        if isinstance(container, ErrorbarContainer):
            for cap in container.lines[1]:
                yield cap.get_markeredgecolor()
                if cap.get_marker() not in ("_", "|"):
                    yield cap.get_markerfacecolor()
            for collection in container.lines[2]:
                yield from collection.get_colors()


def _heatmap_cell_sizes(ax):
    """Yield rendered cell width and height without changing image values or normalization."""
    import numpy as np
    from matplotlib.collections import QuadMesh

    for image in ax.images:
        shape = image.get_array().shape
        if len(shape) < 2 or not shape[0] or not shape[1]:
            continue
        left, right, bottom, top = image.get_extent()
        corners = ax.transData.transform([[left, bottom], [right, top]])
        yield abs(corners[1, 0] - corners[0, 0]) / shape[1], abs(corners[1, 1] - corners[0, 1]) / shape[0]
    for artist in ax.collections:
        if isinstance(artist, QuadMesh):
            coordinates = np.asarray(artist.get_coordinates())
            points = ax.transData.transform(coordinates.reshape(-1, 2)).reshape(coordinates.shape)
            widths = np.linalg.norm(np.diff(points, axis=1), axis=-1)
            heights = np.linalg.norm(np.diff(points, axis=0), axis=-1)
            if widths.size and heights.size:
                yield float(np.median(widths)), float(np.median(heights))


def _visible_artist_colours(artist) -> tuple:
    """Read visible artist RGB values without confusing a scalar map with a series colour."""
    import numpy as np
    from matplotlib.colors import to_rgba

    if not artist.get_visible():
        return ()
    colours = []
    for name in ("get_facecolor", "get_edgecolor", "get_color", "get_colors"):
        getter = getattr(artist, name, None)
        if getter is None:
            continue
        value = getter()
        if isinstance(value, np.ndarray):
            if value.size == 0:
                continue
            values = np.asarray(value).reshape(-1, value.shape[-1])
        else:
            values = [to_rgba(value)]
        for colour in values:
            if to_rgba(colour)[3] == 0:
                continue
            rgba = to_rgba(colour, artist.get_alpha())
            if rgba[3] > 0:
                colours.append(tuple(rgba[:3]))
    return tuple(colours)


def _condition_record_problems(spec, records, *, colour_tolerance=1e-8) -> tuple:
    """Check backend-independent observations of condition forms, points, widths and explicit identities.

    Matplotlib uses direct float RGB values; the native adapter allows one
    sixteen-bit QColor channel step through its explicit tolerance argument.
    """
    import numpy as np
    from matplotlib.colors import to_rgb

    if not records:
        return ("Register per-condition artist roles before checking identities and observation counts",)
    problems = []
    identities, roles = dict(spec.identities), dict(spec.roles)
    estimates = dict(spec.comparison).get("representation") == "estimates"
    if not spec.sample_counts and not estimates:
        problems.append("Declare finite observation counts or an explicit estimate representation")
    for index in range(spec.conditions or 0):
        artists = [item for item in records if item["condition"] == index and item["visible"]]
        count = spec.sample_counts[index] if spec.sample_counts else None
        expected = ("bar" if estimates else _condition_kind_count(count)) if count is not None or estimates else None
        bodies = [item for item in artists if item["role"] in ("bar", "box", "violin")]
        allowed = {"box", "violin"} if expected == "box" else {expected}
        if expected is not None and (count != 0 or estimates) and not allowed.intersection(item["role"] for item in bodies):
            problems.append("Use the count-dependent bar, box or violin representation for every condition")
        points = sum(item["point_count"] for item in artists if item["role"] == "points")
        if count is not None and points != (count if count < DENSE_N else 0):
            problems.append("Show every finite observation below one thousand and no individual points at or above one thousand")
        expected_colour = Palette.CONTROL if index in spec.controls else None
        if expected_colour is None and index < len(spec.condition_labels):
            role = identities.get(spec.condition_labels[index])
            expected_colour = roles.get(role) if role is not None else None
        if expected_colour is not None:
            target = to_rgb(expected_colour)
            coloured = [item for item in artists if item["role"] in ("bar", "box", "violin", "points")]
            if not coloured or any(not np.allclose(colour, target, rtol=0, atol=colour_tolerance)
                                   for item in coloured for colour in item["colours"]):
                problems.append("Draw each explicit control in grey and preserve declared identity colours")
        if any(item["width"] is None or abs(item["width"] - BAR_FRAC) > 1e-8 for item in bodies):
            problems.append("Use the shared six-tenths slot width for every bar, box and violin")
    return tuple(dict.fromkeys(problems))


def _condition_presentation_problems(ax, spec) -> tuple:
    """Observe real Matplotlib artists before passing them to the shared condition validator."""
    import numpy as np
    from matplotlib.collections import PathCollection

    records = []
    problems = []
    horizontal = dict(spec.comparison).get("orientation") == "horizontal"
    coordinate = 1 if horizontal else 0
    limits = ax.get_ylim() if horizontal else ax.get_xlim()
    if spec.conditions and abs(abs(limits[1] - limits[0]) - spec.conditions) > 1e-8:
        problems.append("Use one complete slot per condition; retain a diagnostic for an explicit manual viewport")
    for artist in (*ax.patches, *ax.collections, *ax.lines):
        tag = artist.get_gid() or ""
        if not tag.startswith("spacr-condition:"):
            continue
        try:
            _, index, role = tag.split(":", 2)
            index = int(index)
        except (ValueError, TypeError):
            problems.append("Use a valid explicit condition index in the artist registration")
            continue
        width = None
        if role == "bar":
            width = artist.get_height() if horizontal else artist.get_width()
        elif role == "box":
            width = float(np.ptp(artist.get_path().vertices[:, coordinate]))
        elif role == "violin":
            width = max((float(np.ptp(path.vertices[:, coordinate])) for path in artist.get_paths()), default=0.)
        records.append(dict(condition=index, role=role, visible=artist.get_visible(), width=width,
            point_count=len(artist.get_offsets()) if isinstance(artist, PathCollection) else 0,
            colours=_visible_artist_colours(artist)))
    return tuple(dict.fromkeys((*problems, *_condition_record_problems(spec, records))))

def _scalar_rgba(mappable, values, alpha):
    """Read a resolved scalar mapping and effective uniform opacity without autoscaling it."""
    import numpy as np

    rgba = np.asarray(mappable.get_cmap()(mappable.norm(values))).copy()
    if alpha is not None:
        rgba[..., 3] = alpha
    return rgba


def _colourbar_problems(ax, spec) -> tuple:
    """Check actual scalar-key mappings, rendered cells and text without repairing explicit edits.

    :param ax: declared data axes associated with zero or more scalar keys.
    :param spec: immutable font and quantity policy for the data panel.
    :returns: diagnostics for stale keys or manual presentation departures.
    """
    import numpy as np
    from matplotlib.colors import to_rgba
    from matplotlib.font_manager import FontProperties, findfont, weight_dict
    from matplotlib.text import Text
    from ..qt.i18n import tr

    problems = []
    mismatch = tr("Match the colour key to the plotted scalar values and opacity")
    outer = np.asarray(to_rgba(ax.figure.get_facecolor()))
    for cax in ax.figure.axes:
        bar = getattr(cax, "_colorbar", None)
        if bar is None or not cax.get_visible():
            continue
        source = getattr(cax, "_spacr_colourbar_source", bar.mappable)
        if getattr(source, "axes", None) is None:
            problems.append(tr("Register the scalar source associated with each colour key"))
            continue
        if source.axes is not ax:
            continue
        colour_quantity = dict(spec.legend).get("colour_quantity")
        colour_unit = dict(spec.legend).get("colour_unit")
        if colour_quantity:
            if not colour_unit:
                problems.append(tr("Declare the quantity and unit represented by the colour key"))
            else:
                label = cax.xaxis.label if bar.orientation == "horizontal" else cax.yaxis.label
                expected_label = str(colour_quantity) + " (" + str(colour_unit) + ")"
                if label.get_text() != expected_label:
                    problems.append(tr("Match the colour-key label to its declared quantity and unit"))
        limits = source.get_clim()
        source_alpha = source.get_alpha()
        solids = bar.solids
        key_alpha = solids.get_alpha() if solids is not None else None
        if (solids is None or not solids.get_visible()
                or any(value is None or not np.isfinite(value) for value in (*limits, *bar.mappable.get_clim()))
                or source_alpha is not None and not np.isscalar(source_alpha)
                or key_alpha is not None and not np.isscalar(key_alpha)):
            problems.append(mismatch)
        else:
            try:
                values = np.linspace(*limits, 257)
                expected = _scalar_rgba(source, values, source_alpha)
                constant = (getattr(cax, "_spacr_colourbar_source_limits", None) == (limits[0], limits[0])
                            and limits[0] == limits[1] and bar.cmap.N == 1
                            and tuple(bar.mappable.get_clim()) == (0., 1.))
                observed = _scalar_rgba(bar.mappable, np.full(257, .5) if constant else values, key_alpha)
                correct = np.allclose(observed, expected, rtol=0., atol=1e-8)
                cells = solids.get_array()
                faces = solids.get_facecolors()
                if cells is None or not len(faces):
                    correct = False
                else:
                    rendered = _scalar_rgba(bar.mappable, np.ma.asarray(cells).reshape(-1), key_alpha)
                    correct = correct and faces.shape == rendered.shape and np.allclose(faces, rendered, rtol=0., atol=1e-8)
                if not correct:
                    problems.append(mismatch)
            except (ValueError, TypeError, OverflowError, AttributeError):
                problems.append(mismatch)
        face = np.asarray(to_rgba(cax.get_facecolor()))
        ground = face[:3] * face[3] + outer[:3] * (1.-face[3])
        for text in cax.findobj(Text):
            if not text.get_visible() or not text.get_text().strip():
                continue
            family = FontProperties(fname=findfont(text.get_fontproperties())).get_name()
            if family != spec.font_family:
                problems.append(tr("Use bundled Open Sans for every colour-key text element"))
            if text.get_fontsize() < spec.minimum_font_points:
                problems.append(tr("Use at least eight points for every colour-key text element"))
            weight = text.get_fontweight()
            if weight_dict.get(str(weight), weight) != weight_dict[dict(spec.font_weights)["body"]]:
                problems.append(tr("Use the declared body font weight for colour-key text"))
            rgba = np.asarray(to_rgba(text.get_color(), text.get_alpha()))
            if not np.isclose(rgba[3], 1.):
                problems.append(tr("Keep colour-key text fully opaque"))
            ink = rgba[:3] * rgba[3] + ground * (1.-rgba[3])
            if _colour_contrast(ink, ground) < 4.5:
                problems.append(tr("Use readable text contrast against the colour-key background"))
    return tuple(dict.fromkeys(problems))


def _quantity_unit_problems(spec) -> tuple:
    """Check units for each explicitly declared numerical axis or matrix value.

    :param spec: immutable chart semantics; category identity axes need no physical unit.
    :returns: actionable missing-unit diagnostics without guessing from column names.
    """
    if spec.kind in ("image", "schematic"):
        return ()
    legend = dict(spec.legend)
    expected_units = dict(dict(spec.comparison).get("expected_units") or ())
    declared_units = legend.get("units") or ()
    common_unit = legend.get("unit") or (declared_units if isinstance(declared_units, str) else None)
    units_by_quantity = {} if isinstance(declared_units, str) else dict(declared_units)
    quantities = dict(legend.get("quantities") or ())
    for quantity, required in expected_units.items():
        actual = units_by_quantity.get(quantity) or common_unit
        if actual is None:
            actual = next((units_by_quantity[axis] for axis, label in quantities.items()
                           if label == quantity and axis in units_by_quantity), None)
        if actual is not None and actual != required:
            from ..qt.i18n import tr

            return (tr("State units consistent with the declared mathematical transformation"),)
    if legend.get("unit"):
        return ()
    raw_units = legend.get("units") or ()
    if isinstance(raw_units, str):
        return ()
    units = dict(raw_units)
    quantities = dict(legend.get("quantities") or ())
    axes = ("x", "y")
    if spec.kind == "categorical":
        axes = ("x",) if dict(spec.comparison).get("orientation") == "horizontal" else ("y",)
    elif spec.kind == "heatmap":
        axes = ("value",)
    categorical_axes = dict(spec.comparison).get("categorical_axes") or ()
    for axis in axes:
        if axis in categorical_axes:
            continue
        quantity = quantities.get(axis)
        labels = quantity if isinstance(quantity, (tuple, list)) else (quantity,)
        if not units.get(axis) and any(value is None or not units.get(str(value)) for value in labels):
            return ("Declare units for every numerical quantity on the displayed axes or heatmap scale",)
    return ()



def _heatmap_artist_problems(ax) -> tuple:
    """Inspect actual heatmap lookup colours and visible missing-cell crosses.

    :param ax: axes containing scalar heatmap images or colour meshes.
    :returns: diagnostics; source arrays and manual artist choices remain untouched.
    """
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.collections import QuadMesh
    from matplotlib.markers import MarkerStyle
    from ..qt.i18n import tr

    problems = []
    positions = np.linspace(0.0, 1.0, 256)
    allowed = [colormaps[Palette.SEQUENTIAL](positions), diverging_cmap()(positions)]
    images = [image for image in ax.images if image.get_visible()]
    meshes = [item for item in ax.collections if isinstance(item, QuadMesh) and item.get_visible()]
    if not images and not meshes:
        problems.append(tr("Draw scalar heatmap cells before validating their appearance"))
    for artist in (*images, *meshes):
        colours = artist.get_cmap()(positions)
        if not any(np.allclose(colours, expected, rtol=0.0, atol=1e-8) for expected in allowed):
            problems.append(tr("Use the shared heatmap colour map; preserve manual colours with a diagnostic"))
    if any(spine.get_visible() for spine in ax.spines.values()):
        problems.append(tr("Remove the frame from heatmap cells"))
    marker = MarkerStyle("x")
    expected_path = marker.get_path().transformed(marker.get_transform())
    crosses = [item for item in ax.collections
               if item.get_gid() == "spacr-missing" and item.get_visible()]
    for image in (*images, *meshes):
        values = np.ma.masked_invalid(np.ma.asarray(image.get_array()))
        if isinstance(image, QuadMesh) and values.ndim == 1:
            shape = np.asarray(image.get_coordinates()).shape
            if values.size == (shape[0]-1)*(shape[1]-1):
                values = values.reshape(shape[0]-1, shape[1]-1)
        if values.ndim != 2:
            problems.append(tr("Declare scalar heatmap values for missing-cell validation"))
            continue
        rows, columns = values.shape
        missing_rows, missing_columns = np.nonzero(np.ma.getmaskarray(values))
        if isinstance(image, QuadMesh):
            coordinates = np.asarray(image.get_coordinates())
            centres = (coordinates[:-1, :-1] + coordinates[1:, :-1]
                       + coordinates[:-1, 1:] + coordinates[1:, 1:]) / 4.0
            x = centres[missing_rows, missing_columns, 0]
            y = centres[missing_rows, missing_columns, 1]
        else:
            left, right, bottom, top = image.get_extent()
            x = left + (missing_columns + 0.5) * (right-left) / columns
            y = (top + (missing_rows + 0.5) * (bottom-top) / rows if image.origin == "upper"
                 else bottom + (missing_rows + 0.5) * (top-bottom) / rows)
        expected = sorted(zip(x.tolist(), y.tolist()))
        actual = sorted(tuple(point) for item in crosses for point in item.get_offsets().tolist())
        if actual != expected:
            problems.append(tr("Mark every missing cell with a thin black ×"))
        for item in crosses:
            paths = item.get_paths()
            widths = np.asarray(item.get_linewidths())
            colours = np.asarray(item.get_edgecolors())
            if (not paths or not all(np.array_equal(path.vertices, expected_path.vertices)
                                     and np.array_equal(path.codes, expected_path.codes) for path in paths)
                    or not len(widths) or np.any(widths <= 0) or np.any(widths > WEIGHTS["spine"])
                    or not len(colours) or not np.all(colours[:, :3] == 0.0)):
                problems.append(tr("Mark every missing cell with a thin black ×"))
        if expected:
            ramp = image.get_cmap()
            if not np.isclose(ramp(positions), ramp.get_bad(), rtol=0.0, atol=1e-8).all(axis=1).any():
                problems.append(tr("Use a colour from the heatmap scale beneath missing-cell crosses"))
    return tuple(dict.fromkeys(problems))


def _semantic_presentation_problems(ax, spec) -> tuple:
    """Validate explicit quantities, visible palette, notes and applicable image metadata."""
    import numpy as np
    from matplotlib.colors import to_rgb
    from matplotlib.collections import QuadMesh

    problems = []
    legend = dict(spec.legend)
    comparison = dict(spec.comparison)
    if spec.kind == "axis_overlay":
        problems.extend(_axis_overlay_problems(ax, spec))
    if spec.kind == "heatmap":
        problems.extend(_heatmap_artist_problems(ax))
    if spec.kind == "heatmap" and comparison.get("centre") is not None:
        from ..qt.i18n import tr

        if any(getattr(image.norm, "vcenter", None) != comparison["centre"]
               for image in ax.images):
            problems.append(tr("The declared heatmap reference is not inside the displayed colour range"))
    if spec.kind == "image":
        if not all(comparison.get(key) is not None for key in ("display_range", "gamma", "processing")):
            problems.append("Declare image display range, gamma and processing in comparison metadata")
        return tuple(problems)
    if spec.kind == "schematic":
        if comparison.get("status"):
            if any(artist.get_visible() for artist in _data_artists(ax)):
                from ..qt.i18n import tr

                problems.append(tr("A status panel must not contain plotted observations"))
            return tuple(problems)
        if sum(len(text.get_text().split()) for text in ax.texts if text.get_visible()) > 10:
            problems.append("Keep schematic labels to at most ten words")
        if comparison.get("graph_type") == "venn":
            from ..qt.i18n import tr

            colours = dict(comparison.get("region_colours") or {})
            for patch in ax.patches:
                if not patch.get_visible():
                    continue
                region = str(patch.get_gid() or "").removeprefix("spacr-venn-")
                expected = colours.get(region)
                if expected is None or not np.allclose(to_rgb(patch.get_facecolor()), to_rgb(expected), rtol=0., atol=1e-8):
                    problems.append(tr("Match Venn region colours to the declared input and overlap identities"))
                if any(abs(value - spec.alpha) > 1e-6 for value in _artist_alphas(patch)):
                    problems.append(tr("Draw every data mark at sixty percent opacity"))
        return tuple(problems)
    if not legend.get("quantities"):
        problems.append("Declare the objective quantities shown on both axes")
    for text in ax.texts:
        if (text.get_visible() and text.get_text().strip() and
                text.get_gid() not in ("spacr-panel-letter", "spacr-status", "spacr-missing")):
            problems.append("Move ordinary in-panel text, values and counts into the figure legend")
    allowed = [to_rgb(value) for value in (*SERIES, Palette.CONTROL, Palette.BACKGROUND, "#000000", "#FFFFFF")]
    if spec.kind != "heatmap":
        for artist in _data_artists(ax):
            if isinstance(artist, QuadMesh):
                continue
            if (getattr(artist, "get_array", lambda: None)() is not None
                    and comparison.get("colour_kind") != "categorical"):
                if artist.get_cmap().name not in ("Blues", "spacr_diverging"):
                    problems.append("Use the declared shared sequential or centred diverging colour map")
                continue
            if any(not any(np.allclose(colour, target, rtol=0, atol=1e-8) for target in allowed)
                   for colour in _visible_artist_colours(artist)):
                problems.append("Use only the shared data palette and black summary marks")
    for entry in getattr(ax, "_spacr_legend_entries", ()):
        if entry.get("kind") != "reference":
            continue
        matching = [artist for artist in (*ax.lines, *ax.patches)
                    if artist.get_label() == entry["label"]]
        correct = bool(matching)
        for artist in matching:
            colours = _visible_artist_colours(artist)
            correct = correct and bool(colours) and all(
                np.allclose(colour, to_rgb(entry["colour"]), rtol=0, atol=1e-8)
                for colour in colours)
        if not correct:
            problems.append("Match reference legend colours to their actual plotted lines")
    if spec.kind == "categorical":
        problems.extend(_condition_presentation_problems(ax, spec))
    if comparison.get("forbid_inside_title") and ax.get_title():
        from ..qt.i18n import tr

        problems.append(tr("Move the plot title into the figure legend"))
    for artist in (*ax.lines, *ax.patches):
        if artist.get_gid() != "spacr-reference":
            continue
        colours = _visible_artist_colours(artist)
        if not colours or any(not np.allclose(colour, (0.,0.,0.), rtol=0., atol=1e-8) for colour in colours):
            from ..qt.i18n import tr

            problems.append(tr("Draw reference lines in black; preserve explicit colours with a diagnostic"))
    if comparison.get("graph_type") in ("line", "line_std", "sweep"):
        if ax.get_ylim()[0] > 0 and not comparison.get("cut_axis_reason"):
            problems.append("Start line and sweep axes at zero or declare an authorized cut-axis reason")
    return tuple(dict.fromkeys(problems))



def _native_colour_table(name, steps):
    """Read the existing native colour lookup once through the shared colour owner."""
    import pyqtgraph as pg

    table = pg.colormap.get(name, source="matplotlib") if name == Palette.SEQUENTIAL else pg.colormap.get(name)
    return table.getLookupTable(nPts=int(steps), alpha=True)


def _native_scalar_steps(values, low, high, steps):
    """Retain the original nearest-table-step mapping without modifying scalar values."""
    import numpy as np

    return np.clip(np.round((values - low) / (high - low) * (steps - 1)), 0, steps - 1)


def _native_scalar_cells(plot, edges, colours):
    """Paint supplied scalar boundaries and RGBA cells once for all native colour keys.

    Adjacent vector cells disable antialias seams while keeping their exact
    boundaries and colours. Source adapters retain normalization and chrome.
    """
    import numpy as np
    import pyqtgraph as pg
    from PySide6.QtGui import QColor, QPainter

    class ScalarCells(pg.BarGraphItem):
        """Paint adjacent key cells without antialias seams while retaining every supplied boundary and RGBA value."""

        def paint(self, painter, *arguments):
            """Keep shared cell edges solid and restore the surrounding painter state even on failure."""
            painter.save()
            try:
                painter.setRenderHint(QPainter.Antialiasing, False)
                super().paint(painter, *arguments)
            finally:
                painter.restore()

    edges = np.asarray(edges, dtype=float)
    rgba = np.asarray(colours, dtype=float)
    if len(edges) != len(rgba) + 1 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError(_replay_message("Numeric colour keys need increasing boundaries aligned with their colours"))
    plot.clear()
    cells = ScalarCells(x0=np.zeros(len(rgba)), x1=np.ones(len(rgba)),
        y0=edges[:-1], y1=edges[1:], pen=None,
        brushes=[pg.mkBrush(QColor.fromRgbF(*row)) for row in rgba])
    cells._spacr_condition_role = "colour_key"
    plot.addItem(cells)
    return cells


def _paint_native_scalar_key(plot, edges, colours, *, label, ink, font, dpi):
    """Draw the declared native key around the same vector-cell painter used by scene translation."""
    import pyqtgraph as pg

    cells = _native_scalar_cells(plot, edges, colours)
    for name in ("left", "bottom", "top"):
        plot.showAxis(name, False)
    plot.showAxis("right", True)
    axis = plot.getAxis("right")
    pen = pg.mkPen(ink, width=WEIGHTS["spine"] * float(dpi) / 72.)
    axis.setPen(pen)
    axis.setTickPen(pen)
    axis.setTextPen(pg.mkPen(ink))
    axis.setStyle(tickFont=font, tickLength=-max(1, int(round(2.5 * float(dpi) / 72.))))
    axis.setLabel(label, color=ink, **{"font-family": font.family(),
        "font-size": (f"{font.pointSizeF():g}pt" if font.pointSizeF() > 0 else f"{font.pixelSize()}px"), "font-weight": str(int(font.weight()))})
    low, high = float(edges[0]), float(edges[-1])
    axis.setTicks([[(value, f"{value:.3g}") for value in (low, (low + high) / 2., high)]])
    view = plot.getViewBox()
    view.setRange(xRange=(0., 1.), yRange=(low, high), padding=0.)
    view.disableAutoRange()
    plot.setMouseEnabled(x=False, y=False)
    plot.setMenuEnabled(False)
    plot.layout.setContentsMargins(0., 0., 0., 0.)
    plot.layout.setHorizontalSpacing(0.)
    plot.layout.setVerticalSpacing(0.)
    plot.layout.setColumnFixedWidth(1, .10 * float(dpi))
    plot._spacr_scalar_cells = cells
    return cells


def _remove_native_scalar_key(figure):
    """Remove only the owned external scalar key when its original native scene is reset."""
    key = getattr(figure, "_spacr_scalar_key", None)
    if key is not None:
        scene = key.scene()
        layout = figure.plot.plotItem.layout
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item is key or item.graphicsItem() is key:
                layout.removeAt(index)
                break
        if scene is not None:
            scene.removeItem(key)
    figure._spacr_scalar_key = None
    figure._spacr_scalar_channel = None


def _native_scalar_label(comparison):
    """Format the declared scalar quantity and unit as short, escaped native rich-text lines."""
    import html
    import textwrap

    quantity = str(comparison.get("colour_quantity") or "")
    unit = str(comparison.get("colour_unit") or "")
    lines = textwrap.wrap(quantity, width=24) + (["(" + unit + ")"] if unit else [])
    return "<br>".join(html.escape(line) for line in lines)


def _bind_native_scalar_column(figure, channel):
    """Bind a chosen column to explicit source metadata without carrying units from a different quantity."""
    from dataclasses import replace

    specification = getattr(figure, "_spacr_style_spec", None)
    if channel is None or not isinstance(specification, StyleSpec):
        return
    comparison = dict(specification.comparison)
    if "colour_quantity" not in comparison:
        return
    column = channel["column"]
    frame = figure.frame()
    same_quantity = comparison.get("colour_column") == column
    comparison.update(colour_column=column,
        colour_quantity=(frame.attrs.get("quantities") or {}).get(column,
            comparison.get("colour_quantity") if same_quantity else str(column)),
        colour_unit=(frame.attrs.get("units") or {}).get(column,
            comparison.get("colour_unit") if same_quantity else None))
    figure._spacr_style_spec = replace(specification, comparison=comparison)


def _attach_native_scalar_key(figure):
    """Attach a key to the actual recorded scalar operation, outside the source data viewport."""
    import copy
    import numpy as np
    import pyqtgraph as pg
    from PySide6.QtGui import QFont, QPen
    from ..figure_font import use_open_sans_for_qt

    channel = getattr(figure, "_spacr_scalar_channel", None)
    _bind_native_scalar_column(figure, channel)
    specification = getattr(figure, "_spacr_style_spec", None)
    comparison = dict(specification.comparison) if isinstance(specification, StyleSpec) else {}
    if channel is None or comparison.get("colour_column") != channel["column"]:
        return None
    use_open_sans_for_qt()
    key = getattr(figure, "_spacr_scalar_key", None)
    retained = None
    if key is not None:
        axis = key.getAxis("right")
        retained = dict(font=QFont(axis.style["tickFont"]), label_style=dict(axis.labelStyle),
            label=axis.labelText, pens=(QPen(axis.pen()), QPen(axis.tickPen()), QPen(axis.textPen())),
            ticks=copy.deepcopy(axis._tickLevels), ranges=copy.deepcopy(key.getViewBox().viewRange()),
            same_column=getattr(key, "_spacr_scalar_column", None) == channel["column"],
            default_label=getattr(key, "_spacr_scalar_label", axis.labelText),
            default_ticks=copy.deepcopy(getattr(key, "_spacr_scalar_ticks", axis._tickLevels)),
            default_ranges=copy.deepcopy(getattr(key, "_spacr_scalar_ranges", key.getViewBox().viewRange())))
    if key is None:
        key = pg.PlotItem()
        figure.plot.plotItem.layout.addItem(key, 2, 3)
        figure._spacr_scalar_key = key
    label = _native_scalar_label(comparison)
    font = figure.plot.getAxis("bottom").style.get("tickFont")
    if font is None:
        font = QFont(specification.font_family)
        font.setPointSizeF(specification.minimum_font_points)
        font.setWeight(QFont.Light)
    font = QFont(font)
    rgba = np.asarray(channel["lookup"], dtype=float) / 255.
    rgba[:, 3] = specification.alpha
    steps = len(rgba)
    fractions = np.r_[0., (np.arange(steps - 1) + .5) / (steps - 1), 1.]
    edges = channel["low"] + fractions * (channel["high"] - channel["low"])
    _paint_native_scalar_key(key, edges, rgba, label=label,
        ink=figure._font_colour or figure._foreground, font=font, dpi=figure.logicalDpiX())
    axis = key.getAxis("right")
    key._spacr_scalar_label = label
    key._spacr_scalar_ticks = copy.deepcopy(axis._tickLevels)
    key._spacr_scalar_ranges = copy.deepcopy(key.getViewBox().viewRange())
    key._spacr_scalar_edges = edges.copy()
    key._spacr_scalar_column = channel["column"]
    if retained is not None:
        manual_label = retained["same_column"] and retained["label"] != retained["default_label"]
        axis.setStyle(tickFont=retained["font"])
        axis.labelStyle = dict(retained["label_style"])
        axis.setLabel(retained["label"] if manual_label else label, **retained["label_style"])
        axis.setPen(retained["pens"][0])
        axis.setTickPen(retained["pens"][1])
        axis.setTextPen(retained["pens"][2])
        if retained["same_column"] and retained["ticks"] != retained["default_ticks"]:
            axis.setTicks(retained["ticks"])
        if retained["same_column"] and retained["ranges"] != retained["default_ranges"]:
            key.getViewBox().setRange(xRange=retained["ranges"][0], yRange=retained["ranges"][1], padding=0.)
    return key


def _native_scalar_key_problems(figure, specification):
    """Check actual source-to-key colours, numeric boundaries and quantity labels without repairing them.

    A numeric palette exemption is returned only for a fully verified shared
    ramp. Manual maps and tampered keys remain visible diagnostics.
    """
    import numpy as np
    import pyqtgraph as pg
    from PySide6.QtGui import QColor

    channel = getattr(figure, "_spacr_scalar_channel", None)
    comparison = dict(specification.comparison)
    if not comparison.get("colour_column"):
        return (), ()
    problem = _replay_message("Match the native scalar key to its source values, colours, opacity, quantity and unit")
    key = getattr(figure, "_spacr_scalar_key", None)
    if channel is None or key is None or not key.isVisible() or channel["column"] != comparison["colour_column"]:
        return (problem,), ()
    cells = key._spacr_scalar_cells
    lookup = np.asarray(channel["lookup"])
    rgba = np.asarray([pg.mkBrush(brush).color().getRgbF() for brush in cells.opts["brushes"]])
    expected = lookup.astype(float) / 255.
    expected[:, 3] = specification.alpha
    findings = []
    if (rgba.shape != expected.shape or not np.allclose(rgba, expected, rtol=0., atol=1/65535)
            or cells.opacity() != 1. or not cells.isVisible()):
        findings.append(problem)
    edges = np.r_[cells.opts["y0"], cells.opts["y1"][-1]]
    steps = len(lookup)
    expected_edges = channel["low"] + np.r_[0., (np.arange(steps - 1) + .5) / (steps - 1), 1.] * (channel["high"] - channel["low"])
    if (not np.array_equal(edges, expected_edges) or not np.array_equal(cells.opts["y1"], expected_edges[1:])
            or not np.array_equal(key.getViewBox().viewRange()[1], [channel["low"], channel["high"]])):
        findings.append(problem)
    quantity = str(comparison.get("colour_quantity") or "")
    unit = str(comparison.get("colour_unit") or "")
    expected_label = _native_scalar_label(comparison)
    if not quantity or not unit or key.getAxis("right").labelText != expected_label:
        findings.append(problem)
    tick_levels = key.getAxis("right")._tickLevels
    if not tick_levels:
        findings.append(problem)
    else:
        for level in tick_levels:
            for value, label in level:
                try:
                    printed = float(label)
                    tolerance = abs(float(f"{value:.3g}") - value) + 4 * abs(np.spacing(float(value)))
                    if not channel["low"] <= value <= channel["high"] or abs(printed - value) > tolerance:
                        findings.append(problem)
                except (ValueError, TypeError):
                    findings.append(problem)
    frame = figure.frame()
    values = np.asarray(frame[channel["column"]], dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite) or (float(finite.min()), float(finite.max())) != (channel["low"], channel["high"]):
        findings.append(problem)
    for artist in figure._scatter_items():
        rows = figure._rows_of(artist)
        if rows is None:
            findings.append(problem)
            continue
        picked = values[rows]
        indices = _native_scalar_steps(picked, channel["low"], channel["high"], steps)
        for point, value, index in zip(artist.points(), picked, indices):
            expected_colour = (QColor(channel["missing_colour"]) if not np.isfinite(value)
                               else QColor(*(int(c) for c in lookup[int(index)])))
            expected_colour.setAlphaF(specification.alpha)
            if point.brush().color() != expected_colour or artist.opacity() != 1.:
                findings.append(problem)
    approved = channel["colormap"] == Palette.SEQUENTIAL
    if not approved or not np.array_equal(lookup, _native_colour_table(Palette.SEQUENTIAL, steps)):
        findings.append(_replay_message("Use the shared sequential ramp for a numeric channel; keep explicit manual maps with a diagnostic"))
        approved = False
    allowed = tuple(tuple(row[:3]) for row in expected) if approved and not findings else ()
    return tuple(dict.fromkeys(findings)), allowed


def _apply_native_marks(figure, spec, artists, *, preserve_opacity=False) -> None:
    """Apply producer defaults to explicitly identified native marks before manual edits.

    :param figure: native adapter supplying existing manual line-colour preferences.
    :param spec: shared resolved presentation specification.
    :param artists: triples of original artist, explicit semantic role and optional colour.
        A missing colour retains the existing series identity.
        Reference lines default to shared black; an explicit line-colour
        preference is retained and remains subject to read-only validation.
    :param preserve_opacity: retain an explicit per-point opacity encoding instead of producer defaults.
    """
    import pyqtgraph as pg
    from PySide6.QtGui import QColor

    manual_line = getattr(figure, "_line_colour", None)
    for artist, role, colour in artists:
        if role == "reference" and manual_line is None:
            colour = "#000000"
        artist._spacr_condition_role = role
        if isinstance(artist, pg.ScatterPlotItem):
            if preserve_opacity:
                continue
            brushes = []
            cache = {}
            fallback = artist.opts.get("brush")
            for brush in artist.data["brush"]:
                existing = pg.mkBrush(fallback if brush is None else brush)
                key = colour if colour is not None else existing.color().rgba()
                if key not in cache:
                    ink = QColor(colour) if colour is not None else QColor(existing.color())
                    ink.setAlphaF(spec.alpha)
                    cache[key] = pg.mkBrush(ink)
                brushes.append(cache[key])
            artist.setBrush(brushes)
        elif isinstance(artist, pg.BarGraphItem):
            ink = QColor(colour) if colour is not None else QColor(pg.mkBrush(artist.opts["brush"]).color())
            ink.setAlphaF(spec.alpha)
            artist.setOpts(brush=pg.mkBrush(ink))
        elif isinstance(artist, pg.FillBetweenItem):
            ink = QColor(colour or Palette.ACCENT)
            ink.setAlphaF(spec.alpha)
            artist.setBrush(pg.mkBrush(ink))
        else:
            old = figure._pen_of(artist)
            if old is None:
                continue
            pen = pg.mkPen(old)
            ink = QColor(manual_line or colour) if manual_line or colour else QColor(pen.color())
            if manual_line is None:
                ink.setAlphaF(spec.alpha)
            pen.setColor(ink)
            pen.setWidthF(WEIGHTS["reference" if role == "reference" else "data"]
                          *float(figure.logicalDpiX())/72.)
            artist.setPen(pen)
        artist.setOpacity(1.0)


@contextlib.contextmanager
def _native_file_strokes(item, *, dpi):
    """Keep native cosmetic pen widths physical for one export, then restore them.

    Qt cosmetic pens stay in device pixels when a scene is scaled. Their
    coordinate widths need the file-to-source DPI ratio. Translated scenes
    declare their source resolution; native widgets use display logical DPI.
    Noncosmetic pens already follow the scene transform and remain intact.
    View ranges and automatic-range choices are retained during the change.

    :param item: exported native graphics item or scene; unsupported adapters are unchanged.
    :param dpi: actual output-device resolution in dots per inch.
    :yields: whether a real native device scale was available.
    """
    if item is None or dpi is None:
        yield False
        return
    if not callable(getattr(item, "views", None)) and not callable(getattr(item, "scene", None)):
        yield False
        return
    scene = item if callable(getattr(item, "views", None)) else item.scene()
    displays = scene.views() if callable(getattr(scene, "views", None)) else []
    if not displays:
        yield False
        return
    coordinate_dpi = getattr(item, "_spacr_coordinate_dpi", None)
    if coordinate_dpi is None:
        coordinate_dpi = getattr(displays[0], "_spacr_scene_dpi", displays[0].logicalDpiX())
    scale = float(dpi)/float(coordinate_dpi)
    if abs(scale-1.0) < 1e-12:
        yield True
        return
    import pyqtgraph as pg
    from functools import partial
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPen

    stack = list(item.items()) if item is scene else [item]
    artists = []
    seen = set()
    while stack:
        artist = stack.pop()
        if id(artist) in seen:
            continue
        seen.add(id(artist))
        artists.append(artist)
        stack.extend(artist.childItems())
    views = [(artist, [list(pair) for pair in artist.viewRange()], list(artist.state["autoRange"]))
             for artist in artists if isinstance(artist, pg.ViewBox)]
    restore = []

    def adjusted(pen):
        """Copy a cosmetic pen at the output DPI without changing its colour or dash style."""
        original = pg.mkPen(pen)
        if original.style() == Qt.PenStyle.NoPen or not original.isCosmetic() or original.widthF() == 0:
            return original
        changed = QPen(original)
        changed.setWidthF(original.widthF()*scale)
        return changed

    def change(setter, pen):
        """Retain one original pen before applying an actual device-width change."""
        original = pg.mkPen(pen)
        changed = adjusted(original)
        if changed != original:
            restore.append((setter, original))
            setter(changed)

    def option(artist, key, value):
        """Set one existing bar option through its native invalidation path."""
        artist.setOpts(**{key: value})

    def scatter_pens(artist, pens):
        """Restore original point-pen slots, including inheritance from a default pen."""
        artist.data["pen"] = pens
        artist.data["sourceRect"] = 0
        artist.updateSpots(artist.data)

    try:
        for view, _ranges, _automatic in views:
            view.disableAutoRange()
        for artist in artists:
            if isinstance(artist, pg.AxisItem):
                change(artist.setPen, artist.pen())
                change(artist.setTickPen, artist.tickPen())
            elif isinstance(artist, pg.PlotCurveItem):
                change(artist.setPen, artist.opts.get("pen"))
                if artist.opts.get("shadowPen") is not None:
                    change(artist.setShadowPen, artist.opts["shadowPen"])
            elif isinstance(artist, pg.InfiniteLine):
                change(artist.setPen, artist.pen)
                change(artist.setHoverPen, artist.hoverPen)
            elif isinstance(artist, pg.BarGraphItem):
                change(partial(option, artist, "pen"), artist.opts.get("pen"))
            elif isinstance(artist, pg.ScatterPlotItem):
                previous = list(artist.data["pen"])
                if all(pen is None for pen in previous):
                    change(artist.setPen, artist.opts.get("pen"))
                    continue
                cache = {}
                changed = []
                for pen in previous:
                    key = id(pen)
                    if key not in cache:
                        cache[key] = adjusted(artist.opts.get("pen") if pen is None else pen)
                    changed.append(cache[key])
                if any(after != pg.mkPen(before) for before, after in zip(previous, changed)):
                    restore.append((partial(scatter_pens, artist), previous))
                    artist.setPen(changed)
        yield True
    finally:
        for setter, previous in reversed(restore):
            setter(previous)
        for view, ranges, automatic in views:
            view.enableAutoRange(x=automatic[0], y=automatic[1])
            view.setRange(xRange=ranges[0], yRange=ranges[1], padding=0., disableAutoRange=False)


def _write_native_raster(exporter, path, *, dpi) -> bool:
    """Write native raster pixels with an honest physical resolution tag.

    The measured pyqtgraph PNG used 3780 dots/m (96.012 dpi), despite
    dimensions calculated at 300 dpi. Qt stores integer dots/m: 300 dpi
    becomes 11811 dots/m (299.9994 dpi), with at most half a dots/m step
    of rounding.
    Unsupported exporters retain their file path and report false.
    """
    with _native_file_strokes(getattr(exporter, "item", None), dpi=dpi):
        try:
            rendered = exporter.export(toBytes=True)
        except TypeError:
            exporter.export(str(path))
            return False
        if rendered is None or not callable(getattr(rendered, "setDotsPerMeterX", None)):
            exporter.export(str(path))
            return False
        resolution = int(round(float(dpi) / .0254))
        rendered.setDotsPerMeterX(resolution)
        rendered.setDotsPerMeterY(resolution)
        if not rendered.save(str(path)):
            raise OSError("Native raster exporter could not write the figure")
        return True


def _queue_native_finalization(figure):
    """Validate the latest native scene once the current burst of redraws finishes.

    Full live observation is coalesced on the Qt event loop rather than
    repeated inside each slider redraw. Pending observations never retain
    an earlier compliance result. Explicit validation and file exports
    still inspect the current scene synchronously and stop redundant work.
    """
    import weakref
    from PySide6.QtCore import QTimer

    timer = getattr(figure, "_spacr_style_validation_timer", None)
    if timer is None:
        timer = QTimer(figure)
        timer.setSingleShot(True)
        reference = weakref.ref(figure)

        def validate():
            """Observe the latest still-owned widget after the redraw burst, never a captured old scene."""
            current = reference()
            if current is not None:
                finalize_figure(current)

        timer.timeout.connect(validate)
        figure._spacr_style_validation_timer = timer
    figure._spacr_style_diagnostics = FigureDiagnostics(
        getattr(getattr(figure, "_spacr_style_spec", None), "policy_version", ""),
        (_replay_message("Validate the updated native plot before certifying its appearance"),), ())
    timer.start(0)


def _finalize_native_figure(figure, spec=None, *, diagnostics=True, record=True) -> FigureDiagnostics:
    """Validate native observations with shared policy values, retaining explicit incomplete checks.

    Actual QColor channels use sixteen-bit storage; colour comparisons allow
    one channel step. Native text extents must be observed before certification.
    """
    import numpy as np
    from matplotlib.colors import to_rgb

    timer = getattr(figure, "_spacr_style_validation_timer", None)
    if record and timer is not None:
        timer.stop()
    observed = figure._spacr_style_observation()
    declared = observed.get("specification")
    item = spec or (StyleSpec(**declared) if declared else None)
    problems = []
    checks = []
    incomplete = tuple(observed.get("incomplete_checks", ()))
    problems.extend(incomplete)
    if not isinstance(item, StyleSpec):
        problems.append("Register the chart kind, reference, quantities, and legend before validation")
    else:
        checks.append("registered semantics")
        legend = dict(item.legend)
        comparison = dict(item.comparison)
        if item.kind == "axis_overlay":
            from ..qt.i18n import tr

            problems.append(tr("Axis overlays require a declared shared Matplotlib axis"))
        if item.policy_version != POLICY_VERSION:
            problems.append("Captured native policy differs from the current policy; keep its recorded appearance")
        if not item.controls and not legend.get("reference") and not comparison.get("reference"):
            problems.append("Name the reference or control in the figure legend")
        if not legend.get("caption") or not legend.get("quantities") or not (legend.get("unit") or legend.get("units")):
            problems.append("Declare the caption, plotted quantities and units before validation")
        problems.extend(_quantity_unit_problems(item))
        if observed.get("title") and item.kind not in ("image", "schematic"):
            problems.append(_replay_message("Move the plot title into the figure legend"))
        checks.append("declared reference and quantities")
        geometry = axes_geometry(item)
        pixel = 1. / observed["logical_dpi"]
        width, height = observed["width_inches"], observed["height_inches"]
        problems.extend(_physical_panel_problems(item, width, height, pixel=pixel))
        if geometry.aspect == "categorical":
            horizontal = comparison.get("orientation") == "horizontal"
            measured, expected = (height, geometry.height_inches) if horizontal else (width, geometry.width_inches)
            if expected is None or abs(measured - expected) > pixel:
                problems.append("Use the fixed physical height for horizontal conditions" if horizontal else "Use the fixed physical width for the declared number of conditions")
        if geometry.aspect == "square" and abs(width - height) > pixel:
            problems.append("Use a square plot area for continuous data")
        checks.append("actual native plot area")
        if comparison.get("graph_type") in ("line", "line_std", "sweep"):
            if observed["ranges"][1][0] > 0 and not comparison.get("cut_axis_reason"):
                problems.append("Start line and sweep axes at zero or declare an authorized cut-axis reason")
            checks.append("line and sweep view origin")
        if item.kind not in ("heatmap", "image", "schematic") and not all(edge["visible"] for edge in observed["frame"]):
            problems.append("Use all four frame spines for statistical plots")
        checks.append("statistical frame")
        ground = np.asarray(observed["background"])
        ground_rgb = ground[:3] * ground[3] + np.ones(3) * (1. - ground[3])
        for font in observed["fonts"]:
            if font["family"] != item.font_family or font["size"] < item.minimum_font_points:
                problems.append("Use bundled Open Sans at eight points or larger for every figure text element")
            required = dict(item.font_weights)["title" if font["role"] == "title" else "body"]
            qt_weights = {"light": 300, "regular": 400, "semibold": 600, "bold": 700}
            if font["weight"] != qt_weights[required]:
                problems.append("Use the declared Light body and Regular title font weights")
            ink = np.asarray(font["colour"])
            composited = ink[:3] * ink[3] + ground_rgb * (1. - ink[3])
            if _colour_contrast(composited, ground_rgb) < 4.5:
                problems.append("Use readable text contrast against the actual plot background")
        checks.extend(("actual native font family and size", "composited text contrast"))
        scalar_problems, scalar_colours = _native_scalar_key_problems(figure, item)
        problems.extend(scalar_problems)
        allowed = [to_rgb(value) for value in (*SERIES, Palette.CONTROL, Palette.BACKGROUND, "#000000", "#FFFFFF")]
        allowed.extend(scalar_colours)
        for mark in observed["marks"]:
            for colour in mark["colours"]:
                if colour[3] == 0:
                    continue
                if item.kind not in ("image", "schematic") and abs(colour[3] * mark["opacity"] - item.alpha) > 1e-6:
                    problems.append("Draw every data mark at sixty percent opacity")
                if item.kind not in ("image", "heatmap", "schematic") and not any(np.allclose(colour[:3], target, rtol=0, atol=1/65535) for target in allowed):
                    problems.append("Use only the shared data palette and black summary marks")
            if mark["role"] == "unregistered":
                problems.append("Register native artist roles before checking identities, forms and error bars")
            if mark["role"] in ("error", "summary") and any(not np.allclose(pen["colour"][:3], (0.,0.,0.), rtol=0, atol=1/65535) for pen in mark["pens"]):
                problems.append("Draw every error bar and summary line in black")
            if (mark["role"] == "reference"
                    and any(not np.allclose(pen["colour"][:3], (0.,0.,0.), rtol=0, atol=1/65535) for pen in mark["pens"])):
                problems.append(_replay_message("Draw reference lines in black; preserve explicit colours with a diagnostic"))
        checks.extend(("effective data opacity", "actual native palette", "registered native artist roles"))
        if item.kind == "categorical":
            records = []
            for mark in observed["marks"]:
                indices = mark.get("conditions") or ([mark["condition"]] if mark["condition"] is not None else [])
                for index in indices:
                    records.append(dict(condition=index, role=mark["role"], visible=True,
                        width=mark.get("width"), point_count=mark["point_count"],
                        colours=[tuple(colour[:3]) for colour in mark["colours"] if colour[3] > 0]))
            problems.extend(_condition_record_problems(item, records, colour_tolerance=1/65535))
            checks.append("actual condition forms, counts and identities")
        text_geometry = observed.get("text_geometry", {})
        problems.extend(text_geometry.get("findings", ()))
        if not diagnostics or not observed.get("text_bounds_checked"):
            problems.append("Validate actual native text bounds and overlaps before certification")
        else:
            if text_geometry.get("outside"):
                problems.append("Keep every native text element inside the exported canvas")
            if text_geometry.get("overlaps"):
                problems.append("Separate native labels so their actual text polygons do not overlap")
            checks.extend(("text bounds", "text overlap"))
        limit_checks = True
        for mark in observed["marks"]:
            bounds = mark.get("bounds", {})
            if not bounds.get("checked"):
                limit_checks = False
                problems.append("Observe every native data transform before checking limits")
                continue
            for axis, name in enumerate(("x", "y")):
                values = bounds.get(name)
                if values is None:
                    continue
                if name == "x" and comparison.get("visible_conditions") is not None:
                    continue
                low, high = sorted(observed["ranges"][axis])
                if min(values) < low or max(values) > high:
                    declared = dict(comparison.get("display_ranges") or ()).get(name)
                    try:
                        honest = bool(comparison.get("display_range_reason")) and np.array_equal(
                            np.asarray(declared, dtype=float), np.asarray(observed["ranges"][axis], dtype=float))
                    except (TypeError, ValueError):
                        honest = False
                    if not honest:
                        problems.append("Keep native data marks within the viewport or declare the displayed subset")
        checks.append("actual native data limits")
        if comparison.get("graph_type") in ("line", "line_std", "sweep"):
            if min(observed["ranges"][1]) > 0 and not comparison.get("cut_axis_reason"):
                problems.append("Start line and sweep axes at zero or declare an authorized cut-axis reason")
        if item.kind not in ("categorical", "continuous"):
            problems.append("Complete native image, matrix or schematic observations before certification")
        elif diagnostics and observed.get("text_bounds_checked") and limit_checks and not incomplete:
            checks.append("completed presentation checks")
    result = FigureDiagnostics(item.policy_version if isinstance(item, StyleSpec) else "",
                               tuple(dict.fromkeys(problems)), tuple(dict.fromkeys(checks)))
    if record:
        figure._spacr_style_diagnostics = result
    return result



def finalize_figure(figure, spec=None, *, diagnostics=True, record=True, renderer=None) -> FigureDiagnostics:
    """Check declared semantics and rendered appearance without changing data or manual edits.

    Unregistered figures receive an actionable diagnostic without an expensive
    render. Geometry, opacity, references, fonts, contrast, error-bar ink and
    rendered layout must all be checked before compliance can be reported.
    Declared image and schematic panels are checked even with hidden axes.

    :param figure: drawn Matplotlib figure to validate.
    :param spec: panel specification, or None to use attached specifications.
    :param diagnostics: render layout checks; false never certifies compliance.
    :param record: attach the result to the figure; false supports read-only scene translation.
    :param renderer: already drawn Matplotlib renderer; avoids another live-canvas draw.
    :returns: immutable applicable-check and problem record attached to the figure.
    """
    import numpy as np
    from matplotlib.colors import to_rgb, to_rgba
    from matplotlib.text import Text

    if callable(getattr(figure, "_spacr_style_observation", None)):
        return _finalize_native_figure(figure, spec, diagnostics=diagnostics, record=record)
    if id(figure) in _FINALIZING.get():
        return getattr(figure, "_spacr_style_diagnostics",
                       FigureDiagnostics("", ("Figure validation is already running",), ()))
    token = _FINALIZING.set((*_FINALIZING.get(), id(figure)))
    try:
        specs = [(ax, getattr(ax, "_spacr_style_spec", None) or spec or
                  getattr(figure, "_spacr_style_spec", None))
                 for ax in getattr(figure, "axes", ())
                 if getattr(ax, "_colorbar", None) is None and ax.get_visible()
                 and (ax.axison or getattr(getattr(ax, "_spacr_style_spec", None) or spec or
                     getattr(figure, "_spacr_style_spec", None), "kind", None) in ("image", "schematic"))]
        problems = []
        checks = []
        registered = bool(specs) and all(isinstance(item, StyleSpec) for _, item in specs)
        if not registered:
            problems.append("Register the chart kind, reference, quantities, and legend before validation")
        else:
            checks.append("registered semantics")
        if diagnostics and registered:
            problems.extend(figure_problems(figure, renderer=renderer))
            checks.extend(("text size", "text bounds", "text overlap"))
        for ax, item in specs:
            if not isinstance(item, StyleSpec):
                continue
            legend = dict(item.legend)
            comparison = dict(item.comparison)
            if item.policy_version != POLICY_VERSION:
                problems.append("Captured figure policy differs from the current policy; keep its recorded appearance")
            if item.kind == "unspecified":
                problems.append("Declare whether the panel shows categories, continuous data, a heatmap, or an image")
            if item.kind not in ("image", "schematic"):
                if not item.controls and not legend.get("reference") and not comparison.get("reference"):
                    problems.append("Name the reference or control in the figure legend")
                if not legend.get("unit") and not legend.get("units"):
                    problems.append("Declare the plotted quantities and units, including dimensionless quantities")
            if not legend.get("caption"):
                problems.append("Add a figure legend explaining the quantities and their units")
            checks.append("declared reference and quantities")
            problems.extend(_quantity_unit_problems(item))
            problems.extend(_semantic_presentation_problems(ax, item))
            problems.extend(_colourbar_problems(ax, item))
            checks.extend(("actual palette and legend identities", "applicable representation and semantic requirements"))
            box = ax.get_position()
            width, height = box.width * figure.get_size_inches()[0], box.height * figure.get_size_inches()[1]
            geometry = axes_geometry(item)
            pixel = 1.0 / float(figure.dpi)
            problems.extend(_physical_panel_problems(item, width, height, pixel=pixel))
            if geometry.aspect == "categorical":
                if item.conditions is None:
                    problems.append("Declare the number of categorical conditions")
                else:
                    horizontal = comparison.get("orientation") == "horizontal"
                    measured, expected = (height, geometry.height_inches) if horizontal else (width, geometry.width_inches)
                    if abs(measured - expected) > pixel:
                        problems.append("Use the fixed physical height for horizontal conditions" if horizontal else "Use the fixed physical width for the declared number of conditions")
                checks.append("condition slots")
            if geometry.aspect == "square":
                if abs(width - height) > pixel:
                    problems.append("Use a square plot area for continuous data")
                checks.append("continuous aspect")
            if geometry.aspect == "split":
                if abs(height - width * float(comparison["split_fraction"])) > pixel:
                    problems.append("Retain the declared height ratio of the broken-axis windows")
                checks.append("split axis geometry")
            if item.kind == "heatmap":
                if any(abs(cell_width - cell_height) > 1.0 for cell_width, cell_height in _heatmap_cell_sizes(ax)):
                    problems.append("Use square cells for heatmap data")
                checks.append("heatmap cell aspect")
            if item.kind not in ("image", "schematic"):
                marks = _data_artists(ax)
                if any(artist.get_visible() and any(abs(value - item.alpha) > 1e-6 for value in _artist_alphas(artist)) for artist in marks):
                    problems.append("Draw every data mark at sixty percent opacity")
                checks.append("effective data opacity")
                if any(not np.allclose(to_rgb(colour), (0.0, 0.0, 0.0), atol=1e-8) for colour in _errorbar_colours(ax)):
                    problems.append("Draw every error bar in black")
                checks.append("black error bars")
                if item.kind != "heatmap" and not all(spine.get_visible() for spine in ax.spines.values()):
                    problems.append("Use all four frame spines for statistical plots")
                checks.append("statistical frame")
            face = np.asarray(to_rgba(ax.get_facecolor()))
            page = np.asarray(to_rgba(figure.get_facecolor()))
            outer = page[:3] * page[3] + np.ones(3) * (1.0 - page[3])
            ground = face[:3] * face[3] + outer * (1.0 - face[3])
            outer_text = []
            if ax is specs[0][0]:
                outer_text = list(figure.texts) + [text for key in figure.legends for text in key.findobj(Text)]
            for text in (*ax.findobj(Text), *outer_text):
                if not text.get_visible() or not text.get_text().strip():
                    continue
                if item.font_family not in text.get_fontfamily():
                    from matplotlib.font_manager import FontProperties, findfont

                    resolved = FontProperties(fname=findfont(text.get_fontproperties())).get_name()
                    if resolved != item.font_family:
                        problems.append("Use bundled Open Sans for every figure text element")
                from matplotlib.font_manager import weight_dict

                role = _figure_text_role(text, ax)
                required = dict(item.font_weights)[role]
                actual = text.get_fontweight()
                actual_weight = weight_dict.get(str(actual), actual)
                if actual_weight != weight_dict[required]:
                    problems.append("Use the declared body, title and panel-letter font weights")
                if text.get_fontsize() < item.minimum_font_points:
                    problems.append("Use at least eight points for every figure text element")
                text_ground = outer if text in outer_text else ground
                keys = list(figure.legends) + ([ax.get_legend()] if ax.get_legend() is not None else [])
                for key in keys:
                    if key.get_frame_on() and text in key.findobj(Text):
                        key_face = np.asarray(to_rgba(key.get_frame().get_facecolor()))
                        text_ground = key_face[:3] * key_face[3] + text_ground * (1. - key_face[3])
                rgba = np.asarray(to_rgba(text.get_color(), text.get_alpha()))
                ink = rgba[:3] * rgba[3] + text_ground * (1.0 - rgba[3])
                if _colour_contrast(ink, text_ground) < 4.5:
                    problems.append("Use readable text contrast against the actual plot background")
            checks.extend(("font family and size", "composited text contrast"))
        if diagnostics and registered:
            checks.append("completed presentation checks")
        versions = {item.policy_version for _, item in specs if isinstance(item, StyleSpec)}
        result = FigureDiagnostics(next(iter(versions)) if len(versions) == 1 else "",
                                   tuple(dict.fromkeys(problems)), tuple(dict.fromkeys(checks)))
        if record:
            figure._spacr_style_diagnostics = result
        return result
    finally:
        _FINALIZING.reset(token)


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



def _condition_kind_count(count: int, many: str = "box") -> str:
    """Choose the automatic representation from a known finite observation count."""
    return "violin" if count >= DENSE_N else many if count > AUTO_N else "bar"


def condition_kind(values, many: str = "box") -> str:
    """Bars through twenty values, boxes or violins above twenty, and violins at one thousand.

    :param values: one condition's values.
    :param many: ``'box'`` or ``'violin'``.
    :returns: bar, the requested many-value form, or a dense-group violin.
    """
    import numpy as np

    count = int(np.isfinite(np.asarray(values, float)).sum())
    return _condition_kind_count(count, many)


def _distribution_orientation(vertical: bool) -> dict:
    """Spell distribution orientation for the installed Matplotlib API.

    :param vertical: whether values run along the vertical axis.
    :returns: the supported orientation keyword for box and violin artists.
    """
    import matplotlib

    parts = matplotlib.__version__.split(".")
    try:
        modern = (int(parts[0]), int(parts[1])) >= (3, 10)
    except (IndexError, ValueError):
        modern = True
    return {"orientation": "vertical" if vertical else "horizontal"} if modern else {"vert": bool(vertical)}


def categorical_artist_defaults(ax, kind, *, conditions) -> None:
    """Fit existing manual box or violin bodies to shared slots without changing value coordinates.

    :param ax: axes carrying the original producer artists.
    :param kind: explicitly selected box or violin form.
    :param conditions: declared ordered condition count.
    """
    import numpy as np
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    bodies = ([item for item in ax.patches if isinstance(item, PathPatch)] if kind == "box"
              else [item for item in ax.collections if isinstance(item, PolyCollection)])
    for item in bodies:
        paths = [item.get_path()] if isinstance(item, PathPatch) else item.get_paths()
        low = min(path.vertices[:,0].min() for path in paths)
        high = max(path.vertices[:,0].max() for path in paths)
        centre = (low + high) / 2.
        index = int(round(centre))
        if high > low:
            if isinstance(item, PathPatch):
                vertices = item.get_path().vertices.copy()
                vertices[:,0] = centre + (vertices[:,0] - centre) * BAR_FRAC / (high-low)
                item.set_path(Path(vertices, item.get_path().codes))
            else:
                for path in paths:
                    path.vertices[:,0] = centre + (path.vertices[:,0] - centre) * BAR_FRAC / (high-low)
        item.set_alpha(DATA_ALPHA)
        item.set_edgecolor(item.get_facecolor())
        item.set_linewidth(WEIGHTS["spine"])
        if 0 <= index < conditions:
            item.set_gid(f"spacr-condition:{index}:{kind}")
    for line in ax.lines:
        values = np.asarray(line.get_xdata(), dtype=float)
        if not len(values) or np.ptp(values) > 0.8 + 1e-12:
            continue
        centre = (values.min() + values.max()) / 2.
        index = int(round(centre))
        line.set_xdata(centre + (values-centre) * BAR_FRAC / 0.8)
        line.set_color("#000000")
        line.set_markeredgecolor("#000000")
        median = kind == "violin" and line.get_marker() == "_"
        if kind == "violin":
            line.set_markerfacecolor("#000000")
            if line.get_marker() == "_":
                height = float(np.asarray(line.get_ydata())[0])
                line.set_xdata([centre - BAR_FRAC / 2., centre + BAR_FRAC / 2.])
                line.set_ydata([height, height])
                line.set_marker("None")
        elif line.get_marker() not in (None, "", "None", "none", " "):
            line.set_markersize(GENERAL_DEFAULTS["marker_size"] ** 0.5)
        line.set_alpha(DATA_ALPHA)
        line.set_linewidth(WEIGHTS["median"] if median or np.ptp(values) > 0.6 else 0.8)
        if 0 <= index < conditions:
            line.set_gid(f"spacr-condition:{index}:summary")


def line_artist_defaults(ax, *, point_colours=None, marker_area=4.0) -> None:
    """Use shared trend markers and optionally colour the existing group centres by identity.

    :param ax: original trend axes; line and uncertainty coordinates stay unchanged.
    :param point_colours: colours for one line of existing group centres, or None.
    :param marker_area: observation area in square points, including an explicit size preference.
    """
    import numpy as np

    legend = ax.get_legend()
    handles = list(getattr(legend, "legend_handles", ())) if legend is not None else []
    for line in [*ax.lines, *handles]:
        if not hasattr(line, "get_marker"):
            continue
        line.set_alpha(DATA_ALPHA)
        if line.get_marker() not in (None, "", "None", "none", " ", "_", "|"):
            line.set_markersize(float(marker_area) ** 0.5)
            line.set_markeredgecolor(line.get_color())
            line.set_markeredgewidth(0.)
    if point_colours is None:
        return
    line = next((item for item in ax.lines if len(item.get_xdata()) == len(point_colours)
                 and item.get_linestyle() not in ("None", "none", "")), None)
    if line is None:
        return
    line.set_marker("None")
    ax.scatter(np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float),
               s=marker_area, c=point_colours, alpha=DATA_ALPHA, edgecolors="none",
               zorder=line.get_zorder()+0.1, gid="spacr-trend-centres")


def categorical_slot_defaults(ax, kind, *, conditions, colours=None, point_area=4.0) -> None:
    """Fit original categorical marks to consecutive slots without changing any value coordinate.

    :param ax: axes carrying original bars, boxes, violins or observation marks.
    :param kind: explicitly selected chart form.
    :param conditions: ordered number of measurement/condition slots.
    :param colours: optional default colours by slot; None preserves manual palettes.
    :param point_area: default observation area in square points; None preserves a manual size.
    """
    import numpy as np
    from matplotlib.collections import LineCollection, PathCollection, PolyCollection
    from matplotlib.patches import PathPatch, Rectangle

    if kind in ("box", "violin"):
        categorical_artist_defaults(ax, kind, conditions=conditions)
    for artist in list(ax.patches) + list(ax.collections):
        role = None
        if isinstance(artist, Rectangle):
            centre = artist.get_x() + artist.get_width() / 2.
            index = int(round(centre))
            artist.set_width(BAR_FRAC)
            artist.set_x(index - BAR_FRAC / 2.)
            role = "bar"
        elif isinstance(artist, PathPatch):
            index = int(round(np.mean([artist.get_path().vertices[:,0].min(),
                                       artist.get_path().vertices[:,0].max()])))
            role = "box"
        elif isinstance(artist, PolyCollection) and artist.get_paths():
            paths = artist.get_paths()
            index = int(round((min(path.vertices[:,0].min() for path in paths)
                               + max(path.vertices[:,0].max() for path in paths)) / 2.))
            role = "violin"
        elif isinstance(artist, PathCollection) and len(artist.get_offsets()):
            index = int(round(float(np.median(np.asarray(artist.get_offsets()[:,0], dtype=float)))))
            role = "points"
        elif isinstance(artist, LineCollection) and kind == "bar":
            segments = []
            for original in artist.get_segments():
                segment = original.copy()
                if len(segment):
                    centre = float(np.mean(segment[:,0]))
                    segment[:,0] += round(centre) - centre
                segments.append(segment)
            artist.set_segments(segments)
            artist.set_color("#000000")
            artist.set_alpha(DATA_ALPHA)
            continue
        if role is None or not 0 <= index < conditions:
            continue
        if role == "points" and point_area is not None:
            artist.set_sizes([point_area])
        artist.set_gid(f"spacr-condition:{index}:{role}")
        artist.set_alpha(DATA_ALPHA)
        if colours is not None:
            artist.set_facecolor(colours[index])
            artist.set_edgecolor(colours[index])

    if kind == "bar":
        for line in ax.lines:
            values = np.asarray(line.get_xdata(), dtype=float)
            if not len(values):
                continue
            centre = float(np.mean(values))
            line.set_xdata(values + round(centre) - centre)
            line.set_color("#000000")
            line.set_alpha(DATA_ALPHA)


def condition_plot(ax, data, colours=None, control=None, kind: str | Sequence[str] = "auto",
                   many: str = "box", points: bool = True, seed: int = 0,
                   positions=None, labels=None, orientation="vertical") -> list:
    """Draw one element per condition slot in the house style.

    A condition with at most :data:`AUTO_N` values is a bar at its mean with
    every value as a jittered point; one with more is a box (median,
    quartiles, 5-95 % whiskers) or a violin with small points. At one thousand
    observations the automatic form is a violin without individual points.
    Single observations remain visible as points. Every element
    is :data:`BAR_FRAC` of a slot wide and drawn at :data:`DATA_ALPHA`;
    medians, whiskers and error lines are black at the same opacity;
    the control is grey.

    :param ax: the axes.
    :param data: one sequence of values per condition.
    :param colours: colours instead of the house series.
    :param control: the control's index, or indexes.
    :param kind: ``'auto'``, ``'bar'``, ``'box'`` or ``'violin'``, or captured
        per-condition forms for appearance-preserving replay.
    :param many: what ``'auto'`` draws above :data:`AUTO_N` values.
    :param points: overlay the values as jittered points.
    :param seed: the jitter's random seed.
    :param positions: x position of each condition; ``0 .. n-1`` by default.
    :param labels: explicit source-group identities for portable recipes;
        display tick labels may be edited independently.
    :param orientation: vertical or horizontal values, preserving channel binding.
    :returns: the kind drawn for each condition.
    """
    import numpy as np

    from matplotlib.collections import PathCollection
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    if orientation not in ("vertical", "horizontal"):
        raise ValueError("Condition orientation must be vertical or horizontal")
    vertical = orientation == "vertical"
    ink = "#000000"
    rng = np.random.default_rng(seed)
    cols = condition_colours(len(data), control, colours)
    places = list(range(len(data))) if positions is None else list(positions)
    width = BAR_FRAC
    used = []
    counts = []
    for index, (slot, values, colour) in enumerate(zip(places, data, cols)):
        before = (len(ax.lines), len(ax.collections), len(ax.patches))
        arr = np.asarray(values, float)
        arr = arr[np.isfinite(arr)]
        counts.append(len(arr))
        chosen = (condition_kind(arr, many) if kind == "auto" else kind) if isinstance(kind, str) else kind[index]
        if chosen == "bar":
            (ax.bar if vertical else ax.barh)(slot, arr.mean() if len(arr) else np.nan, width,
                   color=colour, alpha=DATA_ALPHA, linewidth=0, zorder=1)
        elif chosen == "box" and len(arr):
            ax.boxplot([arr], positions=[slot], widths=width,
                       patch_artist=True, showfliers=False, whis=(5, 95),
                       medianprops=dict(color=ink,
                                        linewidth=WEIGHTS["median"], alpha=DATA_ALPHA),
                       boxprops=dict(facecolor=colour, edgecolor=colour,
                                     linewidth=WEIGHTS["spine"],
                                     alpha=DATA_ALPHA),
                       whiskerprops=dict(color=ink, linewidth=0.8, alpha=DATA_ALPHA),
                       capprops=dict(color=ink, linewidth=0.8, alpha=DATA_ALPHA),
                       manage_ticks=False, zorder=2, **_distribution_orientation(vertical))
        elif chosen == "violin" and len(arr) > 1:
            drawn = ax.violinplot([arr], positions=[slot], widths=width,
                                  showextrema=False, **_distribution_orientation(vertical))
            for body in drawn["bodies"]:
                body.set_facecolor(colour)
                body.set_edgecolor("none")
                body.set_alpha(DATA_ALPHA)
            offsets, values = [slot - width / 2, slot + width / 2], [np.median(arr)] * 2
            ax.plot(offsets if vertical else values, values if vertical else offsets,
                    color=ink, linewidth=WEIGHTS["median"], alpha=DATA_ALPHA, zorder=4)
        if points and 0 < len(arr) < DENSE_N:
            spread = width * 0.8
            jittered = slot + rng.uniform(-spread / 2, spread / 2, len(arr))
            ax.scatter(jittered if vertical else arr, arr if vertical else jittered,
                       s=4.0 if len(arr) <= AUTO_N else 1.2,
                       color=colour, alpha=DATA_ALPHA, linewidths=0, zorder=3)
        created = (*list(ax.lines)[before[0]:], *list(ax.collections)[before[1]:], *list(ax.patches)[before[2]:])
        for artist in created:
            role = "points" if isinstance(artist, PathCollection) else "summary" if isinstance(artist, Line2D) else chosen
            artist.set_gid(f"spacr-condition:{index}:{role}")
        used.append(chosen)
    if places:
        (ax.set_xlim if vertical else ax.set_ylim)(min(places) - 0.5, max(places) + 0.5)
    ax._spacr_condition_counts = tuple(counts)
    ax._spacr_condition_kinds = tuple(used)
    ax._spacr_condition_recipe = dict(schema=1, policy_version=POLICY_VERSION,
        labels=[str(value) for value in labels] if labels is not None else None,
        kind=list(used), many=many, points=bool(points), seed=int(seed), orientation=orientation,
        positions=[float(value) for value in places], colours=[list(to_rgba(value)) for value in cols])
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





def embedding_colours(labels, *, noise_label=None) -> dict:
    """Resolve supplied cluster identities through the shared categorical palette.

    :param labels: unchanged cluster labels attached to the embedding.
    :param noise_label: explicitly declared noise identity, or None.
    :returns: label-to-colour mapping; noise is background ink, never an inferred control.
    """
    import numpy as np

    unique = np.unique(labels)
    identities = [value for value in unique if value != noise_label]
    result = dict(zip(identities, condition_colours(len(identities))))
    if noise_label is not None and noise_label in unique:
        result[noise_label] = Palette.BACKGROUND
    return result


def embedding_reference() -> str:
    """Explain the supplied partition and expand the embedding method in the external legend."""
    from ..qt.i18n import tr

    return tr("Uniform manifold approximation and projection (UMAP) uses dimensionless embedding coordinates; cluster identities follow the supplied partition.")


def embedding_axes(ax, labels, *, caption, colours, centroid_labels=(), theme_colors=None, layout=True, outline_colours=None) -> None:
    """Apply the shared UMAP presentation without changing coordinates, hulls or explicit opacity.

    :param ax: axes containing the original embedding artists.
    :param labels: unchanged per-point cluster labels.
    :param caption: explanation stored outside the plotting area.
    :param colours: actual label-to-colour bindings used by the producer.
    :param centroid_labels: explicit centroid text artists to move to the external legend.
    :param theme_colors: explicit saved background, foreground and border overrides, or None.
    :param layout: fit the default page only when no explicit size was supplied.
    :param outline_colours: explicit cluster outline bindings when they differ from point colours.
    """
    import numpy as np
    from matplotlib.colors import to_hex
    from ..qt.i18n import tr

    opacity = [(mark, mark.get_alpha()) for mark in ax.collections]
    for text in centroid_labels:
        text.remove()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_title("")
    ground = (theme_colors or {}).get("background", "#FFFFFF")
    ink = (theme_colors or {}).get("foreground", "#000000")
    border = (theme_colors or {}).get("border", ink)
    ax.figure.set_facecolor(ground)
    ax.set_facecolor(ground)
    ax.tick_params(axis="both", colors=ink)
    ax.xaxis.label.set_color(ink)
    ax.yaxis.label.set_color(ink)
    for spine in ax.spines.values():
        spine.set_color(border)
    reference = embedding_reference()
    spec = resolve_style("screen", "scatter", comparison={"graph_type":"scatter", "colour_kind":"categorical"},
        legend={"caption":caption, "reference":reference,
                "quantities":{"x":ax.get_xlabel(), "y":ax.get_ylabel()}, "unit":"dimensionless"})
    apply_axes_style(ax, spec, layout=layout)
    for mark, alpha in opacity:
        mark.set_alpha(alpha)
    ax._spacr_legend_entries = ()
    _legend_note(ax, caption, ink, "caption")
    _legend_note(ax, reference, ink, "reference_tick")
    for label in np.unique(labels):
        count = int(np.count_nonzero(np.asarray(labels) == label))
        _legend_note(ax, tr("Cluster {cluster}: {count} points.").format(cluster=label, count=count),
                     to_hex(colours[label]), "identity")
        if outline_colours and label in outline_colours:
            _legend_note(ax, tr("Cluster {cluster} outline.").format(cluster=label),
                         to_hex(outline_colours[label]), "identity")


def _axis_colourbars(ax) -> tuple:
    """Return actual colorbar axes explicitly attached to this data axes."""
    return tuple(item for item in ax.figure.axes
                 if getattr(item, "_spacr_colourbar_parent", None) is ax
                 and getattr(item, "_colorbar", None) is not None)


def _position_colourbars(ax, renderer) -> None:
    """Keep attached color keys beside the actual data and label bounds in physical units."""
    figure = ax.figure
    area = ax.get_window_extent(renderer)
    text = ax.get_tightbbox(renderer)
    right = max(area.x1, text.x1 if text is not None else area.x1)
    page_width, page_height = figure.get_size_inches() * figure.dpi
    for cax in _axis_colourbars(ax):
        cax.set_position([(right + 0.10 * figure.dpi) / page_width,
                          area.y0 / page_height, 0.08 * figure.dpi / page_width,
                          area.height / page_height])
        bounds = cax.get_tightbbox(renderer)
        right = bounds.x1 if bounds is not None else right + 0.18 * figure.dpi


def attach_colourbar(ax, mappable, *, label: str, unit=None, constant_colour=None):
    """Add a readable color key without mutating source normalization, arrays or limits.

    Constant values receive a one-color key with one exact value label. The
    independent display key cannot expand the original normalization through
    Matplotlib colorbar callbacks. Physical layout remains with the shared
    single-panel and multi-panel owners.

    :param ax: data axes owning the displayed scalar values.
    :param mappable: original scalar image or point collection.
    :param label: objective quantity represented by color.
    :param unit: explicit declared measurement unit, or None.
    :param constant_colour: explicit single colour for a collapsed scalar range;
        None retains the existing source map. Values and limits stay unchanged.
    :returns: the independent Matplotlib colorbar, or None when no finite scalar values exist.
    """
    import numpy as np

    values = mappable.get_array()
    if values is not None and not np.ma.masked_invalid(np.ma.asarray(values)).count():
        return None
    if values is None and any(value is None for value in mappable.get_clim()):
        return None
    limits = mappable.get_clim()
    if constant_colour is not None and limits[0] is not None and limits[0] == limits[1]:
        from matplotlib.colors import ListedColormap

        with mappable.callbacks.blocked():
            mappable.set_cmap(ListedColormap([constant_colour], name=mappable.get_cmap().name))
    cax = ax.figure.add_axes([0.8, 0.2, 0.02, 0.5])
    cax._spacr_colourbar_parent = ax
    return _numeric_colourbar(mappable, cax, orientation="vertical", label=label, unit=unit)


def _detached_colour_norm(norm):
    """Copy normalization state with an independent callback registry."""
    import copy
    from matplotlib import cbook

    copied = copy.copy(norm)
    copied.callbacks = cbook.CallbackRegistry(signals=["changed"])
    return copied


@contextlib.contextmanager
def _vector_numeric_cells(figure, output_format):
    """Temporarily draw declared numerical matrices as vector cells for PDF and SVG.

    Source arrays, masks, normalization, artist identity and view state are
    restored even after a failed writer. Untagged micrographs and explicit
    interpolated or rasterized image choices retain their original renderer.

    :param figure: Matplotlib figure being written.
    :param output_format: requested file format; other formats are untouched.
    :yields: None while the reversible export artists are active.
    """
    import numpy as np

    if str(output_format).lower() not in ("svg", "pdf"):
        yield
        return
    axes_state = []
    images = []
    meshes = []
    try:
        for ax in figure.axes:
            candidates = [image for image in ax.images if image.get_visible()
                and getattr(image, "_spacr_numeric_cells", False)
                and image.get_interpolation() in ("nearest", "none")
                and not image.get_rasterized()
                and (image.get_alpha() is None or np.isscalar(image.get_alpha()))
                and np.ndim(image.get_array()) == 2 and image.get_array().size]
            if not candidates:
                continue
            axes_state.append((ax, ax.get_xlim(), ax.get_ylim(), ax.get_autoscalex_on(),
                ax.get_autoscaley_on(), ax.dataLim.get_points().copy(), ax.dataLim.minpos.copy(),
                ax.ignore_existing_data_limits, dict(ax._stale_viewlims)))
            ax.set_autoscalex_on(False)
            ax.set_autoscaley_on(False)
            for image in candidates:
                values = np.ma.asarray(image.get_array())
                rows, columns = values.shape
                left, right, bottom, top = image.get_extent()
                x = np.linspace(left, right, columns + 1)
                y = np.linspace(top, bottom, rows + 1) if image.origin == "upper" else np.linspace(bottom, top, rows + 1)
                mesh = ax.pcolormesh(x, y, values, shading="flat", edgecolors="none",
                    cmap=image.get_cmap(), norm=_detached_colour_norm(image.norm),
                    alpha=image.get_alpha(), transform=image.get_transform(),
                    zorder=image.get_zorder(), rasterized=False, antialiased=False)
                meshes.append(mesh)
                mesh.set_clip_on(image.get_clip_on())
                mesh.set_clip_box(image.get_clip_box())
                mesh.set_clip_path(image.get_clip_path())
                images.append((image, image.get_visible()))
                image.set_visible(False)
        yield
    finally:
        for mesh in reversed(meshes):
            mesh.remove()
        for image, visible in images:
            image.set_visible(visible)
        for ax, xlim, ylim, auto_x, auto_y, points, minimum, ignore, stale in axes_state:
            ax.dataLim.set_points(points)
            ax.dataLim._minpos = minimum
            ax.ignore_existing_data_limits = ignore
            ax.set_xlim(xlim, emit=False, auto=auto_x)
            ax.set_ylim(ylim, emit=False, auto=auto_y)
            ax._stale_viewlims.update(stale)


def _numeric_colourbar(mappable, cax, *, orientation, label="", unit=None):
    """Create one source-independent scalar key for fixed or attached layouts.

    :param mappable: original scalar image or point collection.
    :param cax: explicitly allocated key axes.
    :param orientation: horizontal or vertical key direction.
    :param label: objective color quantity, or an empty label owned by the layout.
    :param unit: explicitly declared measurement unit, or None.
    :returns: the independent Matplotlib colorbar, without source normalization callbacks.
    """
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.text import Text
    from ..figure_font import use_open_sans_for_figures

    use_open_sans_for_figures()
    norm = _detached_colour_norm(mappable.norm)
    limits = tuple(mappable.get_clim())
    norm.autoscale_None(mappable.get_array())
    constant = norm.vmin == norm.vmax
    constant_value = norm.vmin
    ramp = mappable.get_cmap().copy()
    if constant:
        ramp = ListedColormap([ramp(norm(norm.vmin))], name=ramp.name)
        norm = Normalize(0., 1.)
    key = cm.ScalarMappable(norm=norm, cmap=ramp)
    cax._spacr_colourbar_source = mappable
    cax._spacr_colourbar_source_limits = limits
    cax.set_label("<colorbar>")
    bar = cax.figure.colorbar(key, cax=cax, orientation=orientation)
    ticks = [norm.vmin, norm.vmax]
    if getattr(norm, "vcenter", None) is not None:
        ticks.insert(1, norm.vcenter)
    if constant:
        ticks = [0.5]
        bar.set_ticks(ticks, labels=scale_tick_labels([constant_value]))
    else:
        bar.set_ticks(ticks, labels=scale_tick_labels(ticks))
    bar.set_label(str(label) + (" (" + str(unit) + ")" if unit else ""), labelpad=2.)
    cax.set_facecolor(GENERAL_DEFAULTS["background"])
    cax.tick_params(colors=GENERAL_DEFAULTS["foreground"], labelsize=TYPE_SCALE["tick"],
                    length=2.5, width=WEIGHTS["spine"], direction="out", pad=2.)
    cax.xaxis.label.set_color(GENERAL_DEFAULTS["foreground"])
    cax.yaxis.label.set_color(GENERAL_DEFAULTS["foreground"])
    bar.outline.set_edgecolor(GENERAL_DEFAULTS["foreground"])
    bar.outline.set_linewidth(WEIGHTS["spine"])
    alpha = mappable.get_alpha()
    if bar.solids is not None:
        bar.solids.set_rasterized(False)
        bar.solids.set_alpha(float(alpha) if alpha is not None and np.isscalar(alpha) else DATA_ALPHA)
    for item in cax.findobj(Text):
        item.set_fontfamily(GENERAL_DEFAULTS["font_family"])
        item.set_fontsize(TYPE_SCALE["tick"])
        item.set_fontweight("light")
    return bar


def scale_tick_labels(values) -> list:
    """Format scale ticks compactly without collapsing distinct numerical positions.

    :param values: finite colour-scale tick positions.
    :returns: labels using the fewest significant digits that distinguish every tick.
    """
    numbers = [float(value) for value in values]
    for precision in range(3, 18):
        labels = [f"{value:.{precision}g}" for value in numbers]
        if len(set(labels)) == len(set(numbers)):
            return labels
    return labels


def heatmap_cmap(*, signed=False):
    """Return a detached shared heatmap map with its missing colour on the same scale.

    :param signed: explicitly declared deviation from zero; never inferred from names.
    :returns: sequential blue or the shared centred diverging map.
    """
    from matplotlib import colormaps

    ramp = diverging_cmap() if signed else colormaps[Palette.SEQUENTIAL].copy()
    ramp.set_bad(ramp(0.0))
    if signed:
        ramp._spacr_reference_value = 0.0
    return ramp


def heatmap_image(ax, matrix, *, cmap, vmin, vmax, extent=None, centre=None, origin="upper", aspect="equal"):
    """Draw unchanged heatmap values and mark each missing cell with a thin black cross.

    The colour normalization is display-only. Its endpoints stay unchanged;
    an explicit centre is used only when it lies strictly inside them.
    Missing cells use a colour from the same scale and are distinguished by
    the cross, rather than being presented as an extra numerical value.

    :param ax: destination axes.
    :param matrix: numerical matrix; masked or nonfinite cells remain missing.
    :param cmap: colormap object or registered name, copied before changing its missing colour.
    :param vmin: unchanged lower scale endpoint, or None for the existing finite minimum.
    :param vmax: unchanged upper scale endpoint, or None for the existing finite maximum.
    :param extent: optional image extent in plotted coordinates.
    :param centre: explicit reference value, or None for ordinary normalization.
    :param origin: unchanged image row origin, either upper or lower.
    :param aspect: equal for categorical cells, or auto for continuous-coordinate density rasters.
    :returns: the heatmap image; cross artists are added in place.
    """
    import numpy as np
    from matplotlib.colors import Normalize

    from matplotlib import colormaps

    values = np.ma.masked_invalid(np.ma.asarray(matrix))
    ramp = colormaps[cmap].copy() if isinstance(cmap, str) else cmap.copy()
    ramp.set_bad(ramp(0.0))
    norm = Normalize(vmin, vmax)
    norm.autoscale_None(values)
    if centre is not None and norm.vmin < centre < norm.vmax:
        norm = centred_norm(norm.vmin, centre, norm.vmax)
    image = ax.imshow(values, cmap=ramp, norm=norm, origin=origin, extent=extent,
                      interpolation="nearest", aspect=aspect, alpha=DATA_ALPHA, zorder=1)
    image._spacr_numeric_cells = True
    rows, columns = values.shape
    missing_rows, missing_columns = np.nonzero(np.ma.getmaskarray(values))
    if len(missing_rows):
        left, right, bottom, top = image.get_extent()
        x = left + (missing_columns + 0.5) * (right - left) / columns
        y = (top + (missing_rows + 0.5) * (bottom - top) / rows if origin == "upper"
             else bottom + (missing_rows + 0.5) * (top - bottom) / rows)
        marker_area = 8.0
        if aspect == "auto":
            geometry = axes_geometry(resolve_style("screen", "scatter"))
            cell_width = geometry.width_inches / columns
            cell_height = geometry.height_inches / rows
            marker_area = min(marker_area, (0.45 * min(cell_width, cell_height) * 72.) ** 2)
        marks = ax.scatter(x, y, marker="x", s=marker_area, linewidths=0.4,
                           color="#000000", alpha=DATA_ALPHA, zorder=2)
        marks.set_gid("spacr-missing")
    return image


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


def figure_problems(figure, *, renderer=None) -> list:
    """What breaks the house rules in ``figure``, one sentence each.

    Text under 8 pt, text cut off at the canvas edge, overlapping text,
    error bars that are not black (or the screen theme's neutral ink, which
    is the same rule on a dark page), and square-marked axes that are not
    square. Tick labels outside the view limits are not drawn and are not
    checked. Hidden axes and disabled axis decorations are excluded, while
    visible image titles and overlay text remain checked. A temporary Agg
    canvas is restored after measurement. An empty list means none were found.

    :param figure: the Matplotlib figure to measure.
    :param renderer: an already drawn renderer; None draws once for measurement.
    :returns: one sentence per problem found.
    """
    import matplotlib
    import numpy as np
    from matplotlib.colors import to_rgb

    original_canvas = figure.canvas
    try:
        if renderer is None:
            if not hasattr(figure.canvas, "get_renderer"):
                from matplotlib.backends.backend_agg import FigureCanvasAgg

                FigureCanvasAgg(figure)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
        width, height = float(renderer.width), float(renderer.height)
        problems = []
        undrawn = set()
        for ax in figure.axes:
            if not ax.get_visible():
                undrawn.update(id(text) for text in ax.findobj(matplotlib.text.Text))
            elif not ax.axison:
                for axis in (ax.xaxis, ax.yaxis):
                    undrawn.update(id(text) for text in axis.findobj(matplotlib.text.Text))
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

    finally:
        if figure.canvas is not original_canvas:
            figure.set_canvas(original_canvas)


__all__ = [
    "TICK_ROTATION", "AUTO_N", "BAR_FRAC", "DATA_ALPHA", "DENSE_N", "SERIES", "SLOT_IN", "bracket",
    "POLICY_VERSION", "StyleSpec", "AxesGeometry", "FigureDiagnostics",
    "resolve_style", "axes_geometry", "finalize_figure", "apply_axes_style", "fit_axes_geometry", "pack_axes_geometry",
    "centred_norm", "condition_colours", "condition_kind", "condition_plot",
    "diverging_cmap", "scale_tick_labels", "heatmap_cmap", "heatmap_image", "figure_problems", "size_condition_axes",
    "square_axes",
    "INK_PRINT", "INK_SCREEN", "LINE_KEYS", "TEXT_KEYS", "TRANSPARENT",
    "Palette", "ROLES", "TYPE_SCALE", "WEIGHTS", "annotate", "chosen_ink",
    "chosen_line_ink", "descriptor", "figure_style", "hide_unused",
    "panel_letter", "pipeline_figures", "rc", "reference_line", "resolve_ink",
    "resolve_line_ink", "rotate_ticks", "text_legend", "theme_target",
    "user_overrides",
]


def resolve_plot_theme(black_background=False, theme_colors=None) -> dict:
    """Resolve explicit saved plot colours through the shared theme boundary.

    Missing or invalid roles use the historical light or dark fallback selected
    by the caller. Valid saved values are retained without consulting live GUI
    preferences, so replay does not substitute the current workstation theme.

    :param black_background: select the black-ground fallback when true.
    :param theme_colors: optional mapping of background, foreground and border.
    :returns: independent mapping containing exactly those three colour roles.
    """
    from collections.abc import Mapping
    from copy import deepcopy
    from matplotlib.colors import to_rgba

    ink = "white" if black_background else "black"
    result = {"background": "black" if black_background else "white",
              "foreground": ink, "border": ink}
    if isinstance(theme_colors, Mapping):
        for role in result:
            value = theme_colors.get(role)
            if value is None:
                continue
            try:
                to_rgba(value)
            except (TypeError, ValueError):
                continue
            result[role] = deepcopy(value)
    return result


def apply_embedding_style(ax, labels, *, caption, colours, method,
                          axis_labels=None, reference=None, unit=None,
                          theme_colors=None, black_background=False,
                          centroid_labels=(), outline_colours=None,
                          preserve_opacity=False, layout=False) -> StyleSpec:
    """Style an existing embedding using its actual method and supplied identities.

    Coordinates, hull paths, source limits, colours, marker sizes and explicit
    outline widths are retained. Producer opacity becomes sixty percent unless
    saved opacity is explicitly requested; zero-alpha marks remain hidden.
    Only the supplied centroid text artists and old axes legend are removed.
    The complete key is retained for :func:`external_legend` to draw later.

    :param ax: axes containing already calculated embedding artists, not images.
    :param labels: original one-dimensional sequence of cluster identities.
    :param caption: already translated explanation of the displayed embedding.
    :param colours: explicit identity-to-colour bindings used by the producer.
    :param method: actual reduction method; generic callers can say embedding.
    :param axis_labels: optional exact x and y labels; otherwise retain both.
    :param reference: explicit reference explanation, or the supplied partition.
    :param unit: coordinate unit; only UMAP and t-SNE default to dimensionless.
    :param theme_colors: saved background, foreground and border overrides.
    :param black_background: fallback theme when saved roles are absent.
    :param centroid_labels: explicit text artists to remove from the data area.
    :param outline_colours: actual outline colours where different from points.
    :param preserve_opacity: retain saved/manual opacity, including alpha arrays.
    :param layout: fit physical geometry; false retains the supplied canvas size.
    :returns: attached immutable style specification, without a compliance claim.
    :raises ValueError: for missing methods, labels, colours or image axes.
    """
    from copy import copy
    import numpy as np
    from matplotlib.colors import to_hex, to_rgba
    from ..qt.i18n import tr

    method = str(method).strip()
    values = np.asarray(labels)
    if not method or values.ndim != 1 or ax.images:
        raise ValueError(tr("Declare the embedding method and one-dimensional labels on non-image axes"))
    quantities = tuple(axis_labels) if axis_labels is not None else (ax.get_xlabel(), ax.get_ylabel())
    if len(quantities) != 2 or any(not str(value).strip() for value in quantities):
        raise ValueError(tr("Declare both embedding coordinate labels"))
    unique = np.unique(values)
    for label in unique:
        to_rgba(colours[label])
        if outline_colours is not None and label in outline_colours:
            to_rgba(outline_colours[label])
    centroid_labels = tuple(centroid_labels)
    if any(text.axes is not ax for text in centroid_labels):
        raise ValueError(tr("Centroid labels must belong to the embedding axes"))
    normalized = method.casefold().replace("-", "").replace("_", "")
    if unit is None and normalized in ("umap", "tsne"):
        unit = "dimensionless"
    if reference is None:
        reference = (embedding_reference() if normalized == "umap" else
                     tr("t-distributed stochastic neighbor embedding (t-SNE) uses dimensionless embedding coordinates; cluster identities follow the supplied partition.")
                     if normalized == "tsne" else tr("Cluster identities follow the supplied partition."))
    method_label = tr("Principal component analysis (PCA)") if normalized == "pca" else method
    method_note = tr("Embedding method: {method}.").format(method=method_label)
    legend_metadata = {"caption": caption, "reference": reference,
                       "quantities": {"x": quantities[0], "y": quantities[1]}}
    if unit is not None:
        legend_metadata["unit"] = unit
    spec = resolve_style("screen", "scatter",
                         comparison={"graph_type": "scatter", "colour_kind": "categorical",
                                     "embedding_method": method}, legend=legend_metadata)
    saved = []
    for mark in _data_artists(ax):
        properties = {}
        for name in ("alpha", "color", "facecolor", "edgecolor", "hatchcolor", "markerfacecolor",
                     "markeredgecolor", "linewidth", "linewidths", "markersize", "sizes", "cmap"):
            getter = getattr(mark, "get_" + name, None)
            if getter is not None:
                properties[name] = copy(getter())
        saved.append((mark, properties, _artist_alphas(mark)))
    limits = (ax.get_xlim(), ax.get_ylim(), ax.get_autoscalex_on(), ax.get_autoscaley_on())
    for text in centroid_labels:
        text.remove()
    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()
    ax.set_title("")
    ax.set_xlabel(quantities[0])
    ax.set_ylabel(quantities[1])
    styled = False
    try:
        for mark, properties, alphas in saved:
            alpha = properties.get("alpha")
            if np.ndim(alpha) and np.size(alpha) > 1:
                for role in ("facecolor", "edgecolor", "hatchcolor"):
                    if role in properties:
                        getattr(mark, "set_" + role)("none")
                mark.set_alpha(np.asarray([float(np.ravel(alpha)[0])]))
                mark.set_alpha(None)
                for role in ("facecolor", "edgecolor", "hatchcolor"):
                    if role in properties:
                        getattr(mark, "set_" + role)(properties[role])
        apply_axes_style(ax, spec, layout=layout)
        styled = True
    finally:
        for mark, properties, alphas in saved:
            for name, value in properties.items():
                getattr(mark, "set_" + name)(value)
            if styled and not preserve_opacity:
                opacity = np.where(np.asarray(alphas) == 0., 0., spec.alpha)
                mark.set_alpha(float(opacity[0]) if opacity.size == 1 else opacity)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_autoscalex_on(limits[2])
    ax.set_autoscaley_on(limits[3])
    theme = resolve_plot_theme(black_background, theme_colors)
    ax.figure.set_facecolor(theme["background"])
    ax.set_facecolor(theme["background"])
    ax.tick_params(axis="both", color=theme["border"], labelcolor=theme["foreground"])
    ax.xaxis.label.set_color(theme["foreground"])
    ax.yaxis.label.set_color(theme["foreground"])
    for spine in ax.spines.values():
        spine.set_color(theme["border"])
    previous_entries = getattr(ax, "_spacr_embedding_legend_entries", ())
    ax._spacr_legend_entries = tuple(entry for entry in getattr(ax, "_spacr_legend_entries", ())
                                     if entry not in previous_entries)
    retained_entries = ax._spacr_legend_entries
    note_ink = to_hex(theme["foreground"])
    _legend_note(ax, caption, note_ink, "caption")
    _legend_note(ax, method_note, note_ink, "note")
    _legend_note(ax, reference, note_ink, "reference_tick")
    for label in unique:
        count = int(np.count_nonzero(values == label))
        _legend_note(ax, tr("Cluster {cluster}: {count} points.").format(cluster=label, count=count),
                     to_hex(colours[label]), "identity")
        if outline_colours is not None and label in outline_colours:
            _legend_note(ax, tr("Cluster {cluster} outline.").format(cluster=label),
                         to_hex(outline_colours[label]), "identity")
    ax._spacr_embedding_legend_entries = tuple(entry for entry in ax._spacr_legend_entries
                                              if entry not in retained_entries)
    identities = list(colours.values())
    if outline_colours is not None:
        identities.extend(outline_colours.values())
    ensure_identity_contrast(ax, artists=[mark for mark, _, _ in saved], colours=identities)
    return spec


def _external_legend_label(text, renderer, font, width):
    """Wrap complete legend text to a measured pixel width without losing words.

    :param text: complete original entry, including any intentional line breaks.
    :param renderer: current renderer used only to measure text.
    :param font: exact legend font properties.
    :param width: preferred column text width in pixels.
    :returns: wrapped text; indivisible long words remain intact.
    """
    lines = []
    for paragraph in str(text).split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = current + " " + word if current else word
            measured = renderer.get_text_width_height_descent(candidate, font, ismath=False)[0]
            if current and measured > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return "\n".join(lines)


def external_legend(figure, *, axes=None, columns=1, theme_colors=None,
                    fontsize=8., column_width=2.5, maximum_width=7.,
                    padding=.10, gap=.25, pack=True, placement="right"):
    """Materialize retained captions and colour keys outside measured panel bounds.

    Text-only notes have no invented colour swatch. Identity and reference
    keys use their declared colours at full opacity, with a contrasting backing
    stroke when the key would disappear on the saved ground. The helper replaces only
    its own prior legend, preserves other figure content and does not alter
    source data or numerical limits. Call after all panel metadata is ready.
    A later layout operation requires calling this helper again.

    :param figure: Matplotlib figure receiving one visible external legend.
    :param axes: contributing axes; defaults to visible non-colourbar axes.
    :param columns: positive number of legend columns.
    :param theme_colors: explicit saved background, foreground and border roles.
    :param fontsize: legend size in points, clamped to the eight-point minimum.
    :param column_width: preferred text width per column in inches.
    :param maximum_width: packing target, or strict final canvas width for below keys.
    :param padding: positive outer margin in inches.
    :param gap: nonnegative gap between panels and legend in inches.
    :param pack: pack declared axes with legend space; false retains panel inches
        and adds outer canvas padding where existing labels extend beyond it.
    :param placement: right, or below with pack=False to preserve panel inches
        within the explicit width bound, trimming only exterior whitespace.
    :returns: visible figure Legend, or None when no metadata entries exist.
    :raises ValueError: for invalid layout arguments or axes from another figure.
    """
    from math import isfinite
    from operator import index
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.colors import to_rgba
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.patheffects import withStroke
    from matplotlib.transforms import Bbox
    from ..figure_font import use_open_sans_for_figures
    from ..qt.i18n import tr

    columns = index(columns)
    numeric = tuple(float(value) for value in (fontsize, column_width, maximum_width, padding, gap))
    fontsize, column_width, maximum_width, padding, gap = numeric
    if (columns < 1 or not all(isfinite(value) for value in numeric) or
            column_width <= 0 or maximum_width <= 0 or padding <= 0 or gap < 0):
        raise ValueError(tr("Legend layout requires positive widths, columns and padding"))
    if placement not in ("right", "below") or (placement == "below" and pack):
        raise ValueError(tr("Below-panel legends require pack=False; placement must be right or below"))
    fontsize = max(8., fontsize)
    selected = list(axes) if axes is not None else [ax for ax in figure.axes
                        if ax.get_visible() and getattr(ax, "_colorbar", None) is None]
    if any(ax.figure is not figure for ax in selected):
        raise ValueError(tr("Legend axes must belong to the supplied figure"))
    if pack and any(not isinstance(getattr(ax, "_spacr_style_spec", None), StyleSpec) for ax in selected):
        raise ValueError(tr("Declare each panel before packing its physical geometry"))
    entries = []
    for ax in selected:
        for entry in getattr(ax, "_spacr_legend_entries", ()):
            kind = str(entry.get("kind", "note"))
            row = (str(entry["label"]), to_rgba(entry["colour"], alpha=1.), kind,
                   entry.get("linestyle", (0, (4, 3))) if kind == "reference" else None,
                   entry.get("reference_role") if kind == "reference" else None)
            if row[0].strip() and row not in entries:
                entries.append(row)
    previous = getattr(figure, "_spacr_external_legend", None)
    if previous is not None and previous in figure.legends:
        previous.remove()
    figure._spacr_external_legend = None
    figure._spacr_external_legend_layout = None
    if not entries:
        return None
    fallback = {"background": figure.get_facecolor()}
    if selected:
        fallback.update(foreground=selected[0].xaxis.label.get_color(),
                        border=selected[0].spines["left"].get_edgecolor())
    if theme_colors is not None:
        fallback.update(theme_colors)
    theme = resolve_plot_theme(False, fallback)
    use_open_sans_for_figures()
    font = FontProperties(family="Open Sans", weight="light", size=fontsize)
    handles = []
    for label, colour, kind, linestyle, role in entries:
        if kind == "identity":
            handle = Line2D([], [], linestyle="none", marker="s", markersize=5.,
                            markerfacecolor=colour, markeredgecolor=colour, alpha=1.)
        elif kind == "reference":
            handle = Line2D([], [], color=colour, linewidth=WEIGHTS["reference"],
                            linestyle=linestyle, alpha=1.)
        else:
            handle = Line2D([], [], linestyle="none", marker="none", alpha=1.)
        if kind in ("identity", "reference") and _colour_contrast(colour, theme["background"]) < 3.:
            backing = ("white" if _colour_contrast("white", theme["background"]) >=
                       _colour_contrast("black", theme["background"]) else "black")
            handle.set_path_effects([withStroke(linewidth=3., foreground=backing)])
        handles.append(handle)
    original_canvas = figure.canvas
    renderer = RendererAgg(8, 8, figure.dpi)
    try:
        labels = [_external_legend_label(label, renderer, font, column_width * figure.dpi)
                  for label, colour, kind, linestyle, role in entries]
        legend = figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0., 1.),
                               bbox_transform=figure.transFigure, borderaxespad=0.,
                               ncol=columns, prop=font, frameon=True, framealpha=1.,
                               facecolor=theme["background"], edgecolor=theme["border"])
        legend.set_gid("spacr-external-legend")
        for text in legend.get_texts():
            text.set_color(theme["foreground"])
            text.set_alpha(1.)
        legend.get_frame().set_alpha(1.)
        if placement == "below":
            available_width = maximum_width - 2. * padding
            wrap_width = min(column_width, available_width / columns)
            for attempt in range(8):
                if wrap_width <= fontsize / 72.:
                    raise ValueError(tr("Legend text cannot fit the declared canvas width at its required font size"))
                for text, entry in zip(legend.get_texts(), entries):
                    text.set_text(_external_legend_label(entry[0], renderer, font, wrap_width * figure.dpi))
                excess = legend.get_window_extent(renderer).width / figure.dpi - available_width
                if excess <= 1e-8:
                    break
                wrap_width -= excess / columns + .02
            else:
                raise ValueError(tr("Legend text cannot fit the declared canvas width at its required font size"))
        if pack and selected:
            pack_axes_geometry(selected, maximum_width=maximum_width, gap=gap,
                               padding=padding, legend_space=True)
        envelopes = [ax.get_tightbbox(renderer) for ax in figure.axes if ax.get_visible()]
        envelopes = [box for box in envelopes if box is not None]
        if not pack:
            envelopes.extend(text.get_window_extent(renderer) for text in figure.texts
                             if text.get_visible() and text.get_text().strip()
                             and text.get_transform() is figure.transFigure)
        shift_x = (max(0., padding - min(box.x0 / figure.dpi for box in envelopes))
                   if not pack and envelopes else 0.)
        shift_y = (max(0., padding - min(box.y0 / figure.dpi for box in envelopes))
                   if not pack and envelopes else 0.)
        panel_right = max((box.x1 / figure.dpi for box in envelopes), default=padding)
        panel_right += shift_x
        box = legend.get_window_extent(renderer)
        old_width, old_height = figure.get_size_inches()
        if placement == "below":
            content_bottom = min((item.y0 / figure.dpi for item in envelopes), default=padding)
            shift_y = max(shift_y, padding + box.height / figure.dpi + gap - content_bottom)
            width = max(min(old_width + shift_x, maximum_width), panel_right + padding,
                        box.width / figure.dpi + 2. * padding)
            if width > maximum_width + 1e-8:
                raise ValueError(tr("Panel content cannot fit the declared canvas width without changing its physical size"))
        else:
            width = max(old_width + shift_x, panel_right + gap + box.width / figure.dpi + padding)
        height = max(old_height + shift_y, box.height / figure.dpi + 2 * padding)
        if not pack and envelopes:
            height = max(height, max(box.y1 / figure.dpi for box in envelopes) + shift_y + padding)
        positions = [(ax, ax.get_position(original=True).frozen()) for ax in figure.axes]
        outer_text = [(text, text.get_position()) for text in figure.texts
                      if text.get_transform() is figure.transFigure]
        figure.set_size_inches(width, height)
        for ax, position in positions:
            ax.set_position([(position.x0 * old_width + shift_x) / width,
                             (position.y0 * old_height + shift_y) / height,
                             position.width * old_width / width, position.height * old_height / height])
        for text, position in outer_text:
            text.set_position(((position[0] * old_width + shift_x) / width,
                               (position[1] * old_height + shift_y) / height))
        anchor = ((padding / width, (padding + box.height / figure.dpi) / height)
                  if placement == "below" else ((panel_right + gap) / width, (height - padding) / height))
        legend.set_bbox_to_anchor(anchor, transform=figure.transFigure)
        for ax in figure.axes:
            locator = ax.get_axes_locator()
            ax.apply_aspect(locator(ax, renderer) if locator is not None else None)
        bounds = legend.get_window_extent(renderer)
        figure._spacr_external_legend = legend
        figure._spacr_external_legend_layout = {
            "legend_bounds_inches": tuple(value / figure.dpi for value in bounds.bounds),
            "panel_bounds_inches": tuple(tuple(value / figure.dpi for value in ax.get_window_extent(renderer).bounds)
                                         for ax in selected),
            "canvas_inches": tuple(float(value) for value in figure.get_size_inches()),
            "content_shift_inches": (shift_x, shift_y),
            "placement": placement,
            "maximum_width_inches": maximum_width if placement == "below" else None,
            "entries": tuple({"label": label, "colour": colour, "kind": kind,
                              "linestyle": linestyle, "reference_role": role}
                             for label, colour, kind, linestyle, role in entries),
            "outside_panels": all(not bounds.overlaps(ax.get_window_extent(renderer)) for ax in selected),
            "inside_canvas": Bbox.from_bounds(0., 0., figure.bbox.width, figure.bbox.height).fully_contains(bounds.x0, bounds.y0)
                             and Bbox.from_bounds(0., 0., figure.bbox.width, figure.bbox.height).fully_contains(bounds.x1, bounds.y1),
        }
        return legend
    finally:
        figure.set_canvas(original_canvas)


def ensure_reference_contrast(ax, *, theme_colors=None, legends=True, artists=()) -> tuple:
    """Back declared reference strokes when their core ink disappears on the ground.

    Existing ``spacr-reference`` lines, matching reference legend keys and
    explicitly supplied summary/error artists are considered. Colour, opacity,
    coordinates, dashes and widths remain
    unchanged. A contrasting backing is a display effect, not a new data line.

    :param ax: axes carrying explicitly registered reference lines.
    :param theme_colors: optional saved background role; otherwise use actual ground.
    :param legends: also back matching keys in existing axes and figure legends.
    :param artists: explicit additional summary ticks, CI caps or line collections.
    :returns: artists that currently require a contrasting backing.
    """
    import numpy as np
    from matplotlib.colors import to_rgba
    from matplotlib.patheffects import Normal, Stroke

    page = np.asarray(to_rgba(ax.figure.get_facecolor()))
    page_rgb = page[:3] * page[3] + (1. - page[3])
    ground = np.asarray(to_rgba(ax.get_facecolor()))
    ground_rgb = ground[:3] * ground[3] + page_rgb * (1. - ground[3])
    if theme_colors is not None:
        ground = np.asarray(to_rgba(resolve_plot_theme(False, theme_colors)["background"]))
        ground_rgb = ground[:3] * ground[3] + page_rgb * (1. - ground[3])
    references = [line for line in ax.lines if line.get_gid() == "spacr-reference"]
    candidates = [(line, ground_rgb) for line in references]
    candidates.extend((artist, ground_rgb) for artist in artists if artist not in references)
    if legends:
        keys = ([ax.get_legend()] if ax.get_legend() is not None else []) + list(ax.figure.legends)
        for key in keys:
            key_ground = ground_rgb if key is ax.get_legend() else page_rgb
            if key.get_frame_on():
                face = np.asarray(to_rgba(key.get_frame().get_facecolor()))
                key_ground = face[:3] * face[3] + key_ground * (1. - face[3])
            handles = getattr(key, "legend_handles", None)
            if handles is None:
                handles = getattr(key, "legendHandles", ())
            for handle, label in zip(handles, key.get_texts()):
                if not callable(getattr(handle, "get_color", None)):
                    continue
                if any(label.get_text() == line.get_label() and
                       to_rgba(handle.get_color())[:3] == to_rgba(line.get_color())[:3]
                       for line in references):
                    candidates.append((handle, key_ground))
    backed = []
    for artist, background in candidates:
        if not hasattr(artist, "_spacr_reference_original_effects"):
            artist._spacr_reference_original_effects = tuple(artist.get_path_effects() or ())
        original = artist._spacr_reference_original_effects
        colours = _visible_artist_colours(artist)
        if colours and any(_colour_contrast(colour, background) < 3. for colour in colours):
            backing = ("white" if _colour_contrast("white", background) >=
                       _colour_contrast("black", background) else "black")
            widths = (artist.get_linewidths() if hasattr(artist, "get_linewidths") else
                      (artist.get_linewidth(),))
            artist.set_path_effects([Stroke(linewidth=float(np.max(widths)) + 2., foreground=backing),
                                     *(original or (Normal(),))])
            backed.append(artist)
        else:
            artist.set_path_effects(list(original))
    return tuple(backed)


def image_report_css(target="screen", theme_colors=None) -> str:
    """Return shared typography and page chrome for an existing image report.

    The rules do not change image opacity, colours, normalization or pixels.
    Existing responsive cards and bounded image layout are retained. The
    browser must have Open Sans available; CSS alone does not certify its font.

    :param target: normal shared screen, print or transparent style target.
    :param theme_colors: explicit saved background, foreground and border roles.
    :returns: CSS for body, header, main, article and image elements.
    """
    from matplotlib.colors import to_hex

    spec = resolve_style(target, "image")
    theme = resolve_plot_theme(target == "screen", theme_colors)
    colours = {key: to_hex(value, keep_alpha=True) for key, value in theme.items()}
    size = max(spec.minimum_font_points, 12.)
    weight = {"light": 300, "regular": 400, "semibold": 600, "bold": 700}[dict(spec.font_weights)["body"]]
    return (f"body{{background:{colours['background']};color:{colours['foreground']};"
            f"font:{weight} {size:g}pt '{spec.font_family}',sans-serif;margin:24px}}"
            "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}"
            f"article{{border:1px solid {colours['border']};border-radius:12px;padding:12px;"
            f"background:{colours['background']}}}"
            "img{max-width:100%;display:block;margin:0 auto 8px;border-radius:8px}"
            "header{margin-bottom:18px;line-height:1.5}")


def native_plot_font(spec=None, *, role="body"):
    """Return the bundled native figure font with the shared size and weight.

    Qt is imported only when a native creator requests its font. This helper
    changes no application-wide font or user interface scaling preference.
    """
    from PySide6.QtGui import QFont
    from ..figure_font import use_open_sans_for_qt

    use_open_sans_for_qt()
    item = spec or resolve_style("screen", "image")
    font = QFont(item.font_family)
    font.setPointSizeF(item.minimum_font_points)
    weights = {"light": QFont.Light, "regular": QFont.Normal,
               "semibold": QFont.DemiBold, "bold": QFont.Bold}
    font.setWeight(weights[dict(item.font_weights).get(role, "light")])
    return font


def native_histogram_geometry(widget, *, series=(), thresholds=(), xlabel="",
                              ylabel="", caption="", reference=""):
    """Lay out a square histogram and measured external explanatory text.

    ``series`` contains label/value mappings; ``thresholds`` contains explicit
    label/value mappings. Labels never determine reference identity. The
    returned rectangle also defines the inverse mapping for draggable levels.
    Scientific values and widget size are never changed by this calculation.
    """
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QFontMetricsF

    spec = resolve_style("screen", "histogram")
    font = native_plot_font(spec)
    metrics = QFontMetricsF(font, widget)
    line = metrics.height() + 4.0
    width = max(1.0, float(widget.width()) - 16.0)
    texts = [str(caption)] if caption else []
    entries = []
    for index, item in enumerate(series):
        entries.append((str(item["label"]), "data", index))
    for item in thresholds:
        entries.append((f'{item["label"]}: {float(item["value"]):.4g}', "reference", None))
    texts.extend(item[0] for item in entries)
    if reference:
        texts.append(str(reference))
    heights = [max(line, metrics.boundingRect(
        QRectF(0.0, 0.0, width - 20.0, 10000.0),
        Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, value).height() + 4.0)
        for value in texts]
    left = 54.0 + (line if ylabel else 0.0)
    bottom = 2.0 * line + sum(heights) + 12.0
    top = line / 2.0 + 6.0
    available = min(max(1.0, widget.width() - left - 12.0),
                    max(1.0, widget.height() - bottom - top))
    # The shared 40 mm panel (:func:`_physical_panel_problems`); a widget too
    # small for it keeps the side that fits, and validation reports it.
    dpi = widget.logicalDpiY()
    standard = axes_geometry(spec).height_inches * dpi
    side = min(standard, available)
    plot = QRectF(left + max(0.0, (widget.width() - left - 12.0 - side) / 2.0),
                  top, side, side)
    y = plot.bottom() + 2.0 * line + 8.0
    rects = []
    for height in heights:
        rects.append(QRectF(8.0, y, width, height))
        y += height
    return dict(plot_rect=plot, font=font, line_height=line,
                legend_rects=rects, legend_texts=texts, entries=entries,
                minimum_height=max(160.0, top + standard) + bottom)


def paint_native_histogram(widget, painter, *, series, edges, xlabel, ylabel,
                           caption, reference, thresholds=(),
                           display_transform="linear", theme_colors=None,
                           xunit="intensity"):
    """Paint supplied histogram values and observe its actual native artists.

    This adapter never bins observations or computes threshold values.
    ``sqrt`` retains the existing QC display of sqrt(count / maximum).
    Returned observations describe the actual paint geometry, font, colours,
    opacity and text bounds; callers retain them for ``finalize_figure``.
    """
    from dataclasses import asdict, replace
    import math
    import numpy as np
    from matplotlib.colors import to_rgba
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QBrush, QColor, QFontInfo, QFontMetricsF, QPainter, QPen, QTransform
    from ..qt.i18n import tr

    if display_transform not in ("linear", "sqrt"):
        raise ValueError(tr("Native histogram transform must be linear or sqrt"))
    edges = np.asarray(edges, dtype=float)
    values = [np.asarray(item["values"], dtype=float) for item in series]
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or not (np.all(np.diff(edges) > 0) or np.all(np.diff(edges) == 0))
            or any(value.shape != (len(edges) - 1,) or not np.isfinite(value).all()
                   or np.any(value < 0) for value in values)):
        raise ValueError(tr("Native histogram requires finite ordered edges and nonnegative bin counts"))
    levels = [float(item["value"]) for item in thresholds]
    if not all(math.isfinite(level) for level in levels):
        raise ValueError(tr("Native histogram thresholds must be finite"))
    spec = resolve_style("screen", "histogram",
        conditions=[str(item["label"]) for item in series],
        comparison={"graph_type": "histogram", "display_transform": display_transform},
        legend={"caption": caption, "reference": reference,
                "quantities": {"x": xlabel, "y": ylabel},
                "units": {"x": str(xunit), "y": "count"}})
    spec = replace(spec, colours=tuple(condition_colours(len(series))))
    widget._spacr_style_spec = spec
    geometry = native_histogram_geometry(widget, series=series, thresholds=thresholds,
        xlabel=xlabel, ylabel=ylabel, caption=caption, reference=reference)
    plot, font = geometry["plot_rect"], geometry["font"]
    supplied = dict(theme_colors or {})
    requested = {name: supplied.get(name, supplied.get(short))
                 for name, short in (("background", "bg"), ("foreground", "fg"), ("border", "fg"))}
    validated = resolve_plot_theme(False, requested)
    dark = QColor.fromRgbF(*to_rgba(validated["background"])).lightnessF() < 0.5
    theme = resolve_plot_theme(dark, requested)
    background = QColor.fromRgbF(*to_rgba(theme["background"]))
    ink = QColor.fromRgbF(*to_rgba(theme["foreground"]))
    frame_colour = QColor.fromRgbF(*to_rgba(theme["border"]))
    background.setAlphaF(1.0)
    ink.setAlphaF(1.0)
    frame_colour.setAlphaF(1.0)
    painter.save()
    try:
        painter.setOpacity(1.0)
        painter.fillRect(widget.rect(), background)
        painter.setFont(font)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        low, high = min([float(edges[0]), *levels]), max([float(edges[-1]), *levels])
        if high <= low:
            high = low + 1.0
        peak = max((float(value.max()) for value in values), default=0.0)
        ceiling = peak or 1.0
        marks, rectangles = [], []
        patterns = (Qt.SolidPattern, Qt.BDiagPattern, Qt.FDiagPattern,
                    Qt.CrossPattern, Qt.DiagCrossPattern)
        brush_styles = [patterns[(index // len(SERIES)) % len(patterns)]
                        for index in range(len(values))]
        for index, counts in enumerate(values):
            colour = QColor(spec.colours[index])
            colour.setAlphaF(spec.alpha)
            bars = []
            heights = []
            for position, count in enumerate(counts):
                fraction = float(count) / ceiling
                height = math.sqrt(fraction) if display_transform == "sqrt" else fraction
                heights.append(height)
                if count <= 0:
                    continue
                start = plot.left() + (edges[position] - low) / (high - low) * plot.width()
                end = plot.left() + (edges[position + 1] - low) / (high - low) * plot.width()
                slot = max(1.0 if edges[position] == edges[position + 1] else 0.0,
                           (end - start) / max(1, len(values)))
                rect = QRectF(start + index * slot, plot.bottom() - height * plot.height(),
                              slot, height * plot.height())
                painter.fillRect(rect, QBrush(colour, brush_styles[index]))
                bars.append(list(rect.getRect()))
            rectangles.append(bars)
            marks.append(dict(role="data", opacity=1.0, colours=[list(colour.getRgbF())],
                pens=[], condition=index, point_count=int(np.count_nonzero(counts)),
                bounds=dict(checked=True,
                            x=([min(bar[0] for bar in bars), max(bar[0] + bar[2] for bar in bars)] if bars else None),
                            y=([min(bar[1] for bar in bars), max(bar[1] + bar[3] for bar in bars)] if bars else None)),
                bin_counts=counts.tolist(), displayed_heights=heights, rectangles=bars,
                brush_style=int(brush_styles[index].value)))
        threshold_colour = QColor("#000000")
        threshold_colour.setAlphaF(spec.alpha)
        threshold_pen = QPen(threshold_colour, 1.5, Qt.DashLine)
        painter.setPen(threshold_pen)
        threshold_positions = []
        for level in levels:
            x = plot.left() + (level - low) / (high - low) * plot.width()
            if _colour_contrast((0.0, 0.0, 0.0), background.getRgbF()[:3]) < 4.5:
                painter.setPen(QPen(QColor("#FFFFFF"), 3.5, Qt.DashLine))
                painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(threshold_pen)
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            threshold_positions.append(x)
            marks.append(dict(role="reference", opacity=1.0,
                colours=[list(threshold_colour.getRgbF())],
                pens=[dict(width=threshold_pen.widthF(), colour=list(threshold_colour.getRgbF()))],
                condition=None, point_count=1,
                bounds=dict(checked=True, x=[x, x], y=[plot.top(), plot.bottom()])))
        frame_pen = QPen(frame_colour, 1.0)
        painter.setPen(frame_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(plot)
        metrics = QFontMetricsF(painter.font(), widget)
        text_bounds = []

        def text(rect, flags, value):
            """Draw text and retain the actual bounds from the font engine."""
            previous = painter.pen()
            painter.setPen(ink)
            painter.drawText(rect, flags, str(value))
            painter.setPen(previous)
            text_bounds.append(metrics.boundingRect(rect, flags, str(value)))

        line = geometry["line_height"]
        for fraction in (0.0, 0.5, 1.0):
            x = plot.left() + fraction * plot.width()
            painter.drawLine(QPointF(x, plot.bottom()), QPointF(x, plot.bottom() + 4.0))
            label = f"{low + fraction * (high - low):.3g}"
            width = metrics.horizontalAdvance(label) + 4.0
            text(QRectF(x - width / 2, plot.bottom() + 5.0, width, line),
                 Qt.AlignHCenter | Qt.AlignTop, label)
            y = plot.bottom() - fraction * plot.height()
            painter.drawLine(QPointF(plot.left() - 4.0, y), QPointF(plot.left(), y))
            count = ceiling * (fraction ** 2 if display_transform == "sqrt" else fraction)
            label_top = y - (line if fraction == 0.0 else line / 2.0)
            text(QRectF(plot.left() - 48.0, label_top, 40.0, line),
                 Qt.AlignRight | Qt.AlignVCenter, f"{count:.3g}")
        text(QRectF(plot.left(), plot.bottom() + line + 5.0, plot.width(), line),
             Qt.AlignHCenter | Qt.AlignTop, xlabel)
        if ylabel:
            painter.save()
            try:
                transform = QTransform().translate(6.0, plot.bottom()).rotate(-90.0)
                painter.setTransform(transform, True)
                painter.setPen(ink)
                rect = QRectF(0.0, 0.0, plot.height(), line)
                painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, ylabel)
                text_bounds.append(transform.mapRect(metrics.boundingRect(
                    rect, Qt.AlignHCenter | Qt.AlignTop, ylabel)))
            finally:
                painter.restore()
        offset = 1 if caption else 0
        legend_swatches = []
        for index, (rect, value) in enumerate(zip(geometry["legend_rects"], geometry["legend_texts"])):
            entry = index - offset
            content = QRectF(rect)
            if 0 <= entry < len(geometry["entries"]):
                _, role, condition = geometry["entries"][entry]
                colour = QColor(spec.colours[condition]) if role == "data" else QColor(threshold_colour)
                colour.setAlphaF(1.0)
                swatch = QRectF(rect.left(), rect.top() + 3.0, 12.0, metrics.height() - 2.0)
                if role == "reference":
                    painter.fillRect(swatch.adjusted(-1.0, -1.0, 1.0, 1.0), QColor("#FFFFFF"))
                pattern = brush_styles[condition] if role == "data" else Qt.SolidPattern
                painter.fillRect(swatch, QBrush(colour, pattern))
                legend_swatches.append(dict(role=role, colour=list(colour.getRgbF()),
                    rectangle=list(swatch.getRect()), brush_style=int(pattern.value)))
                content.adjust(20.0, 0.0, 0.0, 0.0)
            painter.setPen(ink)
            text(content, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, value)
        actual = QFontInfo(painter.font())
        font_record = dict(family=actual.family(), size=actual.pointSizeF(),
            weight=int(actual.weight()), role="label", colour=list(ink.getRgbF()))
        canvas = QRectF(widget.rect())
        outside = [list(rect.getRect()) for rect in text_bounds if not canvas.contains(rect)]
        overlaps = [(a, b) for a, first in enumerate(text_bounds)
                    for b, second in enumerate(text_bounds) if a < b and first.intersects(second)]
        dpi = float(widget.logicalDpiY())
        observation = dict(backend="QPainter", specification=asdict(spec), logical_dpi=dpi,
            width_inches=plot.width() / dpi, height_inches=plot.height() / dpi,
            plot_rect=list(plot.getRect()), ranges=[[plot.left(), plot.right()], [plot.top(), plot.bottom()]],
            data_ranges=[[low, high], [0.0, ceiling]], bin_edges=edges.tolist(),
            threshold_values=levels, threshold_positions=threshold_positions,
            background=list(background.getRgbF()), fonts=[dict(font_record) for _ in text_bounds],
            frame=[dict(edge=edge, visible=True, pen=dict(width=frame_pen.widthF(), colour=list(frame_colour.getRgbF())))
                   for edge in ("left", "right", "top", "bottom")], marks=marks,
            legend_swatches=legend_swatches,
            text_bounds_checked=True,
            text_geometry=dict(outside=outside, overlaps=overlaps,
                               rectangles=[list(rect.getRect()) for rect in text_bounds]),
            incomplete_checks=[])
    finally:
        painter.restore()
    return observation


def apply_colourbar_theme(ax, *, theme_colors=None) -> tuple:
    """Match attached scalar-key chrome to the actual styled parent axes.

    Call after creating the key and applying the parent panel style, before
    final layout or writing. Only key backgrounds, text, ticks and outlines
    change. Source arrays, mapped colours, normalization, limits, scalar-key
    cells and their opacity remain untouched. Explicit saved role overrides
    use the same validation as other G2 presentation adapters.

    :param ax: styled source axes with explicitly attached shared scalar keys.
    :param theme_colors: optional saved background, foreground and border roles.
    :returns: the existing colorbar objects whose chrome was updated.
    """
    from collections.abc import Mapping
    from matplotlib.text import Text

    requested = {"background": ax.get_facecolor(),
                 "foreground": ax.xaxis.label.get_color(),
                 "border": ax.spines["left"].get_edgecolor()}
    if isinstance(theme_colors, Mapping):
        requested.update(theme_colors)
    theme = resolve_plot_theme(False, requested)
    bars = []
    for cax in _axis_colourbars(ax):
        bar = cax._colorbar
        cax.set_facecolor(theme["background"])
        cax.patch.set_alpha(1.)
        cax.tick_params(which="both", color=theme["border"],
                        labelcolor=theme["foreground"])
        for text in cax.findobj(Text):
            text.set_color(theme["foreground"])
            text.set_alpha(1.)
        for spine in cax.spines.values():
            spine.set_edgecolor(theme["border"])
        if bar.outline is not None:
            bar.outline.set_edgecolor(theme["border"])
        bars.append(bar)
    return tuple(bars)


def _identity_contrast_effect(ax, colours, width):
    """Build a hollow identity backing that respects the alpha of each rendered path."""
    import numpy as np
    from matplotlib.colors import to_rgba
    from matplotlib.patheffects import AbstractPathEffect

    class IdentityBacking(AbstractPathEffect):
        """Draw only a thin border behind visible, declared low-contrast identities."""

        def __reduce__(self):
            """Reconstruct through the module factory with the transported axes graph."""
            return (_identity_contrast_effect, (ax, colours, width))

        def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
            """Inspect actual per-path colors without repainting the marker face."""
            page = np.asarray(to_rgba(ax.figure.get_facecolor()))
            ground = np.asarray(to_rgba(ax.get_facecolor()))
            background = ground[:3] * ground[3] + (
                page[:3] * page[3] + 1. - page[3]) * (1. - ground[3])
            present = [rgbFace] if rgbFace is not None else []
            if gc.get_linewidth() > 0:
                present.append(gc.get_rgb())
            opacity = []
            for value in present:
                rgba = np.asarray(to_rgba(value))
                alpha = gc.get_alpha() if gc.get_forced_alpha() else rgba[3]
                if alpha <= 0 or not any(np.allclose(rgba[:3], colour[:3], rtol=0., atol=np.finfo(np.float32).eps)
                                         for colour in colours):
                    continue
                visible = rgba[:3] * alpha + background * (1. - alpha)
                if _colour_contrast(visible, background) < 3.:
                    opacity.append(alpha)
            if not opacity:
                return
            backing = ("white" if _colour_contrast("white", background) >=
                       _colour_contrast("black", background) else "black")
            border = renderer.new_gc()
            border.copy_properties(gc)
            try:
                border.set_alpha(max(opacity))
                border.set_foreground(backing)
                border.set_linewidth(gc.get_linewidth() + width)
                renderer.draw_path(border, tpath, affine, None)
            finally:
                border.restore()

    return IdentityBacking()


def ensure_identity_contrast(ax, *, artists, colours, width=.8) -> tuple:
    """Back explicit embedding identities without changing their core appearance.

    Only supplied Line2D hulls and PathCollection points are eligible, and only
    colors in the supplied identity bindings can receive a border. Each draw
    checks its actual composited contrast and alpha, including mixed arrays;
    invisible paths stay invisible. The hollow backing never fills a marker
    face twice. Artist colors, alpha, sizes, widths, paths and offsets remain
    exact. Existing manual path effects are left untouched. The thin additional
    border can affect boundary antialiasing and requires actual visual review.

    :param ax: embedding axes owning the explicitly supplied artists.
    :param artists: existing point collections and hull lines, never images.
    :param colours: explicit point and outline color bindings, without role inference.
    :param width: extra total border width in points, preserving core line width.
    :returns: artists with this additive drawing effect installed, not a validation result.
    """
    from math import isfinite
    import numpy as np
    from matplotlib.collections import PathCollection
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patheffects import Normal
    from ..qt.i18n import tr

    width = float(width)
    if not isfinite(width) or width <= 0:
        raise ValueError(tr("Identity contrast border width must be positive and finite"))
    bindings = tuple(to_rgba(colour) for colour in colours)
    page = np.asarray(to_rgba(ax.figure.get_facecolor()))
    ground = np.asarray(to_rgba(ax.get_facecolor()))
    background = ground[:3] * ground[3] + (
        page[:3] * page[3] + 1. - page[3]) * (1. - ground[3])
    installed = []
    for artist in artists:
        if (artist.axes is not ax or not isinstance(artist, (Line2D, PathCollection))
                or artist.get_gid() == "spacr-reference"):
            continue
        current = tuple(artist.get_path_effects() or ())
        previous = getattr(artist, "_spacr_identity_contrast_effects", None)
        if current and current != previous:
            continue
        if previous is not None and current != previous:
            continue
        actual = (list(artist.get_facecolors()) + list(artist.get_edgecolors())
                  if isinstance(artist, PathCollection)
                  else [to_rgba(artist.get_color(), artist.get_alpha())])
        eligible = artist.get_visible() and any(
            rgba[3] > 0 and any(np.allclose(rgba[:3], colour[:3], rtol=0.,
                                          atol=np.finfo(np.float32).eps) for colour in bindings)
            and _colour_contrast(np.asarray(rgba[:3]) * rgba[3] + background * (1. - rgba[3]), background) < 3.
            for rgba in actual)
        if not eligible:
            if previous is not None and current == previous:
                artist.set_path_effects([])
                artist._spacr_identity_contrast_effects = None
            continue
        effects = (_identity_contrast_effect(ax, bindings, width), Normal())
        artist.set_path_effects(list(effects))
        artist._spacr_identity_contrast_effects = effects
        installed.append(artist)
    return tuple(installed)


def styled_reference_line(ax, *, x=None, y=None, label="", linestyle="--", role="reference"):
    """Draw an explicit black reference with the same dash pattern in its visible key.

    The existing numerical reference-line owner supplies coordinate, opacity,
    width and black core. Only the explicit dash and role from the caller are added;
    coincident-reference suppression remains with the scientific caller.
    Actual line/key artists use the ordinary appearance capture and replay.

    :param ax: receiving axes.
    :param x: vertical coordinate, mutually exclusive with y.
    :param y: horizontal coordinate, mutually exclusive with x.
    :param label: complete translated key label.
    :param linestyle: explicit Matplotlib line style, shared by line and key.
    :param role: caller-declared identity such as used or reference.
    :returns: the original reference-line artist.
    """
    from matplotlib.lines import Line2D
    from ..qt.i18n import tr

    if (x is None) == (y is None) or not str(role).strip():
        raise ValueError(tr("Declare one reference coordinate and its explicit role"))
    Line2D([], [], linestyle=linestyle)
    drawn = reference_line(ax, x=x, y=y, label=label)
    drawn.set_linestyle(linestyle)
    if label:
        entries = list(ax._spacr_legend_entries)
        for index in range(len(entries) - 1, -1, -1):
            entry = entries[index]
            if entry["label"] == str(label) and entry["kind"] == "reference":
                entries[index] = dict(entry, linestyle=linestyle, reference_role=str(role))
                break
        ax._spacr_legend_entries = tuple(entries)
    return drawn
