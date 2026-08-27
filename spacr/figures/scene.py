"""Translate completed matplotlib figures into pyqtgraph scenes.

This module preserves the geometry and statistics already computed by a
matplotlib panel while changing the renderer used for the saved file. Artist
types listed in :data:`CARRIED` are translated; an unsupported artist marks
the translation incomplete and causes the caller to retain the original
matplotlib output.

Scene exports support headless rendering when Qt is available and otherwise
fall back to matplotlib. Output names follow :func:`spacr.plot.figure_path`,
saved files are announced through :func:`spacr.figure_sink.publish_file`, and
print colours are resolved by :func:`spacr.figure_style.export_colour` from
each artist's role.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: Matplotlib artist types this module can carry into a pyqtgraph scene.
#:
#: A census of the whole QC suite produced nothing outside this set. It is a
#: WHITELIST rather than a blacklist deliberately: an artist nobody thought
#: about is a piece of the picture that would go missing, and a figure that is
#: quietly missing its reference line is worse than one drawn in the old
#: library.
CARRIED = (
    "Line2D", "Text", "Annotation", "Rectangle", "PathCollection",
    "LineCollection", "PolyCollection", "AxesImage", "Legend",
    # Chrome, translated as axis CONFIGURATION rather than as items.
    "Spine", "XAxis", "YAxis", "XTick", "YTick", "AxesSubplot", "Axes",
)

#: Artists that are part of the axes rather than of the plot, and are skipped
#: without that counting as an incomplete translation.
IGNORED = ("Spine", "XAxis", "YAxis", "XTick", "YTick")

#: Renderer names, shared with :mod:`spacr.figures.fast_render`.
RENDERERS = ("auto", "pyqtgraph", "matplotlib")

#: Points-per-inch, for turning a matplotlib font size into a pixel size.
_POINTS_PER_INCH = 72.0

#: How wide a colour bar column is, in pixels. A key, not a panel.
COLORBAR_PX = 96

#: A QApplication this module had to start, kept alive. Qt destroys the
#: application the moment its last Python reference goes, and an export that
#: happens after that is a segfault rather than an exception.
_APPLICATION = None


@dataclass
class SceneReport:
    """What the translation could and could not carry.

    ``complete`` is the only field a caller has to read, and it is the whole
    contract: a translation that dropped something is not a picture of the
    panel, so the caller writes the matplotlib page instead. The rest is for
    the message, because "it fell back" without naming the artist is a report
    nobody can act on.
    """

    axes: int = 0
    items: int = 0
    missing: List[str] = field(default_factory=list)
    data_colours: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing

    def reason(self) -> str:
        """One sentence naming what stopped the translation."""
        if self.complete:
            return ""
        counts: Dict[str, int] = {}
        for name in self.missing:
            counts[name] = counts.get(name, 0) + 1
        named = ", ".join(f"{name} x{count}" for name, count
                          in sorted(counts.items()))
        return f"pyqtgraph cannot yet carry {named}"


# --------------------------------------------------------------------------- #
#  Choosing a renderer
# --------------------------------------------------------------------------- #

def requested_renderer() -> str:
    """The renderer ``SPACR_FIGURE_RENDERER`` asks for, or ``'auto'``.

    An unrecognised value is ``'auto'`` rather than an error, for the reason
    :func:`spacr.figure_style.figure_save_mode` gives: a run must not lose its
    figures over a misspelt environment variable.
    """
    asked = os.environ.get("SPACR_FIGURE_RENDERER", "").strip().lower()
    return asked if asked in RENDERERS else "auto"


def scene_renderer(force: Optional[str] = None) -> Tuple[str, str]:
    """``(renderer, reason)`` for a generated figure with NO interactive twin.

    THIS IS A DIFFERENT QUESTION FROM THE ONE
    :func:`spacr.figures.fast_render.renderer_for` ANSWERS, and the difference
    is why there are two functions rather than one with a flag. There, the
    question is "is there a live widget to render, so the file can BE the
    tab", and the answer must never be guessed -- two attempts to detect a
    live GUI were built and both were wrong (a QApplication exists because
    matplotlib's QtAgg backend made one; a module is imported because a test
    imported it). Here there is no widget to find and nothing to disagree
    with: the only question is whether this machine can paint with pyqtgraph
    at all, and that is answerable by trying it.

    So ``auto`` means "pyqtgraph if it is available here". A machine without
    Qt writes the matplotlib page it always wrote, and says so.

    :param force: one of :data:`RENDERERS`, overruling the environment.
    :returns: ``('pyqtgraph', '')`` or ``('matplotlib', why)``. The reason is
        never empty for matplotlib, because "why does this figure not look
        like the others" is the question a user asks of it.
    """
    choice = str(force).strip().lower() if force else requested_renderer()
    if choice not in RENDERERS:
        choice = "auto"
    if choice == "matplotlib":
        return "matplotlib", "matplotlib was asked for"
    if choice == "auto":
        blocked = _the_gallery_could_not_show_it()
        if blocked:
            return "matplotlib", blocked
    available, why = pyqtgraph_ready()
    if not available:
        return "matplotlib", why
    return "pyqtgraph", ""


def _the_gallery_could_not_show_it() -> str:
    """Why a rendered file would be invisible in this process, or ``''``.

    Matplotlib figures and completed scene files use separate publication
    sinks. If a figure sink is active but no file sink is installed, choose
    matplotlib so the result remains visible in the gallery. Headless runs,
    which install neither sink, may continue to use the scene renderer.
    """
    from ..figure_sink import file_sink, sink

    if sink() is not None and file_sink() is None:
        return ("a figure sink is attached and no file sink is, so a rendered "
                "file would not reach the gallery (spacr/qt/bridge.py installs "
                "only set_sink)")
    return ""


def pyqtgraph_ready() -> Tuple[bool, str]:
    """``(ok, reason)``: can a scene be built and exported here and now?

    Starting a ``QApplication`` is licensed by this being the renderer that
    was chosen, not by a guess about a GUI. The offscreen platform is set only
    when there is no display, and it is MEASURED to work: with it,
    ``ImageExporter`` writes a PNG and a ``QPdfWriter`` writes a real vector
    PDF on a machine with no display at all.
    """
    global _APPLICATION

    # THE IMPORT ORDER IS LOAD-BEARING AND THE FAILURE IS BRUTAL. pyqtgraph
    # picks its Qt binding on first import by trying PyQt5, PyQt6, PySide2,
    # PySide6 in that order -- and PyQt6 is installed in this environment, so
    # a bare `import pyqtgraph` binds to PyQt6 and loads ITS libQt6Core.
    # PySide6 6.11 then cannot load at all:
    #
    #   libpyside6.abi3.so.6.11: undefined symbol:
    #   _ZN9QtPrivate9sizedFreeEPvm, version Qt_6
    #
    # Measured 2026-08-18, and it is not hypothetical -- it took the whole QC
    # suite back to matplotlib on the first run through this path. Importing
    # PySide6 FIRST leaves it in `sys.modules` where pyqtgraph finds it, and
    # PYQTGRAPH_QT_LIB says so out loud for anything that imports pyqtgraph
    # before this function is ever called.
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    try:
        from PySide6.QtWidgets import QApplication
        import pyqtgraph  # noqa: F401
    except Exception as error:                                 # noqa: BLE001
        return False, f"pyqtgraph is unavailable here: {error}"
    try:
        if QApplication.instance() is None:
            if not os.environ.get("DISPLAY") and not os.environ.get(
                    "WAYLAND_DISPLAY"):
                os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            _APPLICATION = QApplication.instance() or QApplication([])
    except Exception as error:                                 # noqa: BLE001
        return False, f"no QApplication could be started: {error}"

    # NOT ON A WORKER THREAD, WHEN THERE IS A GUI TO BE A WORKER OF.
    #
    # `build_scene` makes a `pg.GraphicsLayoutWidget`, and a QWidget must be
    # constructed on the GUI thread. Built on a worker it LIVES there, and
    # every later touch -- including Qt destroying it -- is undefined. Qt
    # reports the one case it can detect ("QBasicTimer::start: Timers cannot
    # be started from another thread") and says nothing about the rest.
    #
    # Traced 2026-08-19 from a crash dump, after the process had segfaulted
    # twice in places that had nothing to do with it: once inside an
    # application-wide event filter, once inside pandas' CSV parser. The
    # construction guard named the real one:
    #
    #   WidgetGroup was CONSTRUCTED on 'Dummy-2'
    #     bridge.py run  ->  perform_regression
    #     ->  _run_guide_permutation_analysis  ->  write_diagnostic_suite
    #     ->  plot_inference_diagnostics  ->  write_figure  ->  render_figure
    #     ->  build_scene
    #
    # i.e. the QC suite of every regression, on the run's own worker thread.
    #
    # Answered HERE because this function already exists to say whether a
    # scene can be built "here and now", and its callers already know what to
    # do with a no: `render_figure` returns None and the caller writes the
    # matplotlib page instead. The figure is still produced; only the renderer
    # changes, which is the trade this module already makes for a missing
    # pyqtgraph.
    #
    # A HEADLESS RUN IS NOT AFFECTED. With no GUI, the run IS the main thread
    # and this is true; the check only fires for a worker under a live
    # application, which is exactly the dangerous case.
    if threading.current_thread() is not threading.main_thread():
        return False, (
            "a pyqtgraph scene is made of Qt widgets and this is not the GUI "
            f"thread (it is {threading.current_thread().name!r}); a widget "
            "built here would live on a thread that is about to end. The "
            "matplotlib page is written instead.")
    return True, ""


# --------------------------------------------------------------------------- #
#  Colour
# --------------------------------------------------------------------------- #

def _hex(colour) -> Optional[str]:
    """A matplotlib colour as ``'#RRGGBB'``, or None when it is not paint.

    ``'none'``, a fully transparent RGBA and anything unreadable all answer
    None, which every caller treats as "there is nothing to draw here". That
    is the safe direction: inventing a colour for an artist that had none puts
    ink on the page that the panel deliberately left off it.
    """
    if colour is None:
        return None
    try:
        import matplotlib.colors as mcolors

        rgba = mcolors.to_rgba(colour)
    except Exception:                                          # noqa: BLE001
        return None
    if rgba[3] <= 0.0:
        return None
    return "#%02X%02X%02X" % tuple(int(round(channel * 255))
                                   for channel in rgba[:3])


def _alpha(colour, artist_alpha=None) -> int:
    """The 0-255 alpha of ``colour``, combined with an artist-level alpha."""
    try:
        import matplotlib.colors as mcolors

        value = float(mcolors.to_rgba(colour)[3])
    except Exception:                                          # noqa: BLE001
        value = 1.0
    if artist_alpha is not None:
        try:
            value *= float(artist_alpha)
        except (TypeError, ValueError):
            pass
    return max(0, min(255, int(round(value * 255))))


class _Look:
    """Apply one saved-figure appearance consistently to translated artists.

    Delegate colour decisions to :func:`spacr.figure_style.export_colour`.
    Each matplotlib artist's type determines whether its colour represents
    chrome, data, a reference, or the figure ground.
    """

    def __init__(self, mode=None, dpi: float = 100.0):
        from ..figure_style import saved_figure_appearance

        self.look = saved_figure_appearance(mode)
        self.data_colours: List[str] = []
        # POINTS ARE NOT PIXELS, AND THE FIGURE'S OWN DPI IS THE EXCHANGE
        # RATE. Every size matplotlib carries -- a line width, a marker
        # diameter, a font size -- is in POINTS; a pyqtgraph scene is in the
        # pixels the widget is that many inches wide at. Passing the number
        # through unchanged draws a 3 pt marker 3 px across, which at this
        # suite's 140 dpi is 1.9 times too small, and a panel of dust where
        # the panel had points.
        self.scale = max(float(dpi), 1.0) / _POINTS_PER_INCH

    def px(self, points, minimum: float = 0.1) -> float:
        """A matplotlib size in points, as scene pixels."""
        try:
            return max(float(points) * self.scale, minimum)
        except (TypeError, ValueError):
            return minimum

    @property
    def ground(self) -> Optional[str]:
        return self.look.ground

    def paint(self, colour: Optional[str], kind: str,
              record: bool = True) -> Optional[str]:
        """The colour to paint one artist, having asked the shared rule.

        :param record: whether a data colour joins the legibility check.
            ``False`` for a SEPARATOR -- the white edge a histogram draws
            between its bars is not a mark anybody has to find, and counting
            it fired the "this colour will not read on paper" warning on every
            figure in the suite. A warning that fires on everything is a
            warning nobody reads, which is the rule the floor itself was
            chosen by.
        """
        if colour is None:
            return None
        if kind == "data":
            # Recorded rather than changed: a palette chosen against a dark
            # ground can be illegible on paper, and the honest answer is to
            # NAME it rather than substitute a colour the user did not choose.
            if record:
                self.data_colours.append(colour)
            return colour
        from ..figure_style import export_colour

        return export_colour(colour, kind, self.look) or colour


# --------------------------------------------------------------------------- #
#  Translating one artist
# --------------------------------------------------------------------------- #

def _dash(style) -> Optional[Sequence[float]]:
    """A pyqtgraph dash pattern for a matplotlib linestyle, or None for solid."""
    patterns = {"--": [4, 3], "-.": [5, 2, 1, 2], ":": [1, 2]}
    if isinstance(style, str):
        return patterns.get(style)
    # matplotlib also carries (offset, (on, off, ...)) tuples.
    try:
        _, sequence = style
        return [max(float(v), 0.1) for v in sequence] if sequence else None
    except Exception:                                          # noqa: BLE001
        return None


def _pen(colour, width, style, alpha, look, kind):
    """A QPen, or None when the artist has nothing to draw with."""
    import pyqtgraph as pg

    painted = look.paint(_hex(colour), kind)
    if painted is None:
        return None
    pen = pg.mkPen(color=pg.mkColor(painted), width=look.px(width or 1.0))
    colour_object = pen.color()
    colour_object.setAlpha(_alpha(colour, alpha))
    pen.setColor(colour_object)
    dash = _dash(style)
    if dash:
        pen.setDashPattern([max(v, 0.1) for v in dash])
    return pen


def _reference_line(artist, axes) -> Optional[str]:
    """``'h'``, ``'v'`` or None: is this Line2D a reference line?

    THE TEST IS THE TRANSFORM, NOT THE COLOUR, and it is the same one
    :func:`spacr.plot._chrome` uses on the matplotlib side. ``axhline`` and
    ``axvline`` draw in a blended axes/data transform; a plotted series draws
    in ``ax.transData`` itself. That is a property of what the line MEANS, so
    it survives a user restyle -- and it is the only way to tell a zero line
    from a fitted curve, both of which are ``Line2D``.
    """
    try:
        if artist.get_transform() is axes.transData:
            return None
    except Exception:                                          # noqa: BLE001
        return None
    data = artist.get_xydata()
    if data is None or len(data) < 2:
        return None
    xs = {round(float(point[0]), 12) for point in data}
    ys = {round(float(point[1]), 12) for point in data}
    if len(ys) == 1 and len(xs) > 1:
        return "h"
    if len(xs) == 1 and len(ys) > 1:
        return "v"
    return None


def _add_line(plot, artist, axes, look) -> int:
    """Translate one ``Line2D``. Returns the number of items added."""
    import numpy as np
    import pyqtgraph as pg

    reference = _reference_line(artist, axes)
    alpha = artist.get_alpha()
    if reference:
        data = artist.get_xydata()
        pen = _pen(artist.get_color(), artist.get_linewidth(),
                   artist.get_linestyle(), alpha, look, "chrome")
        if pen is None:
            return 0
        position = (float(data[0][1]) if reference == "h"
                    else float(data[0][0]))
        line = pg.InfiniteLine(pos=position,
                               angle=0 if reference == "h" else 90, pen=pen)
        line.setZValue(float(artist.get_zorder() or 0))
        plot.addItem(line)
        return 1

    data = np.asarray(artist.get_xydata(), dtype=float)
    if data.size == 0:
        return 0
    xs, ys = data[:, 0], data[:, 1]
    marker = artist.get_marker()
    style = artist.get_linestyle()
    added = 0
    if style not in ("None", " ", "", None):
        pen = _pen(artist.get_color(), artist.get_linewidth(), style, alpha,
                   look, "data")
        if pen is not None:
            label = artist.get_label()
            curve = pg.PlotDataItem(
                x=xs, y=ys, pen=pen,
                name=(label if label and not str(label).startswith("_")
                      else None))
            curve.setZValue(float(artist.get_zorder() or 0))
            plot.addItem(curve)
            added += 1
    if marker not in ("None", " ", "", None):
        brush_colour = look.paint(_hex(artist.get_markerfacecolor()), "data")
        size = look.px(artist.get_markersize() or 4.0, minimum=1.0)
        points = pg.ScatterPlotItem(
            x=xs, y=ys, size=size, pen=None,
            brush=pg.mkBrush(brush_colour) if brush_colour else None)
        points.setZValue(float(artist.get_zorder() or 0))
        plot.addItem(points)
        added += 1
    return added


def _add_path_collection(plot, artist, look) -> int:
    """Translate a ``scatter``.

    matplotlib's ``s`` is an AREA in points squared and pyqtgraph's ``size``
    is a DIAMETER, so the conversion is a square root rather than a scale
    factor. Getting that wrong is not subtle -- an s=18 scatter drawn at
    diameter 18 is a panel of overlapping blobs.
    """
    import numpy as np
    import pyqtgraph as pg

    offsets = np.asarray(artist.get_offsets(), dtype=float)
    if offsets.size == 0:
        return 0
    sizes = np.asarray(artist.get_sizes(), dtype=float)
    if sizes.size == 0:
        sizes = np.array([20.0])
    diameters = np.sqrt(np.maximum(sizes, 0.0)) * look.scale
    if diameters.size == 1:
        diameters = np.repeat(diameters, len(offsets))
    elif diameters.size < len(offsets):
        diameters = np.resize(diameters, len(offsets))

    faces = artist.get_facecolor()
    alpha = artist.get_alpha()
    brushes = []
    for index in range(len(offsets)):
        face = faces[index % len(faces)] if len(faces) else None
        painted = look.paint(_hex(face), "data")
        if painted is None:
            brushes.append(None)
            continue
        brush = pg.mkBrush(painted)
        colour = brush.color()
        colour.setAlpha(_alpha(face, alpha))
        brush.setColor(colour)
        brushes.append(brush)

    edges = artist.get_edgecolor()
    pen = None
    if len(edges):
        edge = look.paint(_hex(edges[0]), "data", record=False)
        widths = artist.get_linewidth()
        width = look.px(np.ravel(widths)[0] if np.size(widths) else 1.0)
        pen = pg.mkPen(edge, width=width) if edge else None

    points = pg.ScatterPlotItem(x=offsets[:, 0], y=offsets[:, 1],
                                size=diameters, brush=brushes, pen=pen)
    points.setZValue(float(artist.get_zorder() or 0))
    plot.addItem(points)
    return 1


def _add_line_collection(plot, artist, look) -> int:
    """Translate ``vlines`` / ``hlines``.

    ONE ITEM, NOT ONE PER SEGMENT. A Cook's-distance stem plot on a real
    screen is one segment per well; adding four hundred ``PlotDataItem``s
    makes an export that takes minutes and a PDF nobody can open.
    ``connect='pairs'`` draws them as a single disconnected curve.
    """
    import numpy as np
    import pyqtgraph as pg

    segments = artist.get_segments()
    if not len(segments):
        return 0
    xs: List[float] = []
    ys: List[float] = []
    for segment in segments:
        points = np.asarray(segment, dtype=float)
        if len(points) < 2:
            continue
        xs.extend([float(points[0][0]), float(points[-1][0])])
        ys.extend([float(points[0][1]), float(points[-1][1])])
    if not xs:
        return 0
    colours = artist.get_colors()
    widths = np.ravel(artist.get_linewidth())
    pen = _pen(colours[0] if len(colours) else None,
               float(widths[0]) if widths.size else 1.0,
               "-", artist.get_alpha(), look, "data")
    if pen is None:
        return 0
    curve = pg.PlotDataItem(x=np.asarray(xs), y=np.asarray(ys), pen=pen,
                            connect="pairs")
    curve.setZValue(float(artist.get_zorder() or 0))
    plot.addItem(curve)
    return 1


def _add_poly_collection(plot, artist, look) -> int:
    """Translate ``fill_between`` as a filled polygon."""
    import numpy as np
    import pyqtgraph as pg
    from PySide6.QtGui import QPolygonF
    from PySide6.QtWidgets import QGraphicsPolygonItem
    from PySide6.QtCore import QPointF

    added = 0
    faces = artist.get_facecolor()
    alpha = artist.get_alpha()
    for index, path in enumerate(artist.get_paths()):
        vertices = np.asarray(path.vertices, dtype=float)
        if len(vertices) < 3:
            continue
        face = faces[index % len(faces)] if len(faces) else None
        painted = look.paint(_hex(face), "data")
        if painted is None:
            continue
        polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in vertices])
        item = QGraphicsPolygonItem(polygon)
        brush = pg.mkBrush(painted)
        colour = brush.color()
        colour.setAlpha(_alpha(face, alpha))
        brush.setColor(colour)
        item.setBrush(brush)
        item.setPen(pg.mkPen(None))
        item.setZValue(float(artist.get_zorder() or 0))
        plot.addItem(item)
        added += 1
    return added


def _add_rectangles(plot, rectangles, axes, look) -> int:
    """Translate the bars of a histogram or a bar chart.

    One :class:`pyqtgraph.BarGraphItem` for the whole set rather than one
    graphics item per bar: a twenty-bin p-value histogram is cheap either way,
    but the response distribution of a real screen is not, and the difference
    is visible in the export time.
    """
    import numpy as np
    import pyqtgraph as pg

    x0, x1, y0, y1, brushes, pens = [], [], [], [], [], []
    for rectangle in rectangles:
        width = float(rectangle.get_width())
        height = float(rectangle.get_height())
        if width == 0.0 and height == 0.0:
            continue
        left, bottom = float(rectangle.get_x()), float(rectangle.get_y())
        # A PATCH'S `get_transform` IS NOT THE DATA TRANSFORM, and testing it
        # against `ax.transData` silently dropped EVERY bar in the suite --
        # the p-value histogram, the VIF bars, the response distribution and
        # the design conditioning all came out as empty axes with correct
        # ranges, which is the most convincing kind of wrong figure there is.
        # `Patch.get_transform` is `get_patch_transform() + the artist's`, so
        # it is a composite and is never that object; `get_data_transform` is
        # the artist's, and is.
        if rectangle.get_data_transform() is not axes.transData:
            # A backdrop drawn in axes fraction -- `_skip_box` draws one
            # behind the reason a panel is missing. Converted, because a
            # skipped tile with no backdrop stops looking skipped.
            corner = _in_data_coordinates(_FractionPoint(left, bottom), axes)
            far = _in_data_coordinates(
                _FractionPoint(left + width, bottom + height), axes)
            if corner is None or far is None:
                continue
            left, bottom = corner
            width, height = far[0] - corner[0], far[1] - corner[1]
        # CLAMPED TO THE VIEW. A ViewBox does not clip its children, and
        # matplotlib does clip a patch, so a `barh` whose bars start at x = 0
        # on an axis that starts above zero drew them straight out through the
        # left spine and across the page -- seen on `vif`. Clamping the
        # GEOMETRY rather than clipping the ITEM is deliberate: an annotation
        # deliberately placed below the axes (`predictor_correlation` puts its
        # caption at y = -0.34) must still be drawn, and a clip on the ViewBox
        # would take that with it.
        left, right = _clamp(left, left + width, *sorted(axes.get_xlim()))
        bottom, top = _clamp(bottom, bottom + height, *sorted(axes.get_ylim()))
        if right <= left or top < bottom:
            continue
        x0.append(left)
        x1.append(right)
        y0.append(bottom)
        y1.append(top)
        face = look.paint(_hex(rectangle.get_facecolor()), "data")
        brush = pg.mkBrush(face) if face else pg.mkBrush(None)
        if face:
            colour = brush.color()
            colour.setAlpha(_alpha(rectangle.get_facecolor(),
                                   rectangle.get_alpha()))
            brush.setColor(colour)
        brushes.append(brush)
        edge = look.paint(_hex(rectangle.get_edgecolor()), "data", record=False)
        pens.append(pg.mkPen(edge, width=look.px(rectangle.get_linewidth() or 0.5))
                    if edge else pg.mkPen(None))
    if not x0:
        return 0
    bars = pg.BarGraphItem(x0=np.asarray(x0), x1=np.asarray(x1),
                           y0=np.asarray(y0), y1=np.asarray(y1),
                           brushes=brushes, pens=pens)
    plot.addItem(bars)
    return 1


def _clamp(low, high, floor, ceiling) -> Tuple[float, float]:
    """``(low, high)`` held inside ``[floor, ceiling]``."""
    return max(min(low, ceiling), floor), min(max(high, floor), ceiling)


def _anchor(text) -> Tuple[float, float]:
    """A pyqtgraph anchor for a matplotlib alignment pair."""
    horizontal = {"left": 0.0, "center": 0.5, "right": 1.0}
    vertical = {"top": 0.0, "center": 0.5, "center_baseline": 0.5,
                "baseline": 1.0, "bottom": 1.0}
    return (horizontal.get(text.get_horizontalalignment(), 0.0),
            vertical.get(text.get_verticalalignment(), 1.0))


class _FractionPoint:
    """A point in axes fraction, shaped like the bit of ``Text``
    :func:`_in_data_coordinates` reads.

    A three-line adapter rather than a second copy of the conversion: the log
    handling in there is the part that is easy to get subtly wrong, and one
    figure with a rectangle placed by one rule and its label by another is the
    kind of disagreement this whole module exists to remove.
    """

    def __init__(self, x, y):
        self._position = (x, y)

    def get_position(self):
        return self._position

    def get_transform(self):
        return None


def _in_data_coordinates(text, axes) -> Optional[Tuple[float, float]]:
    """Where a Text sits, in DATA coordinates, or None if it cannot be placed.

    A panel's annotation is positioned in axes fraction (``transform=
    ax.transAxes``) so it stays in the corner whatever the data does.
    pyqtgraph has no axes-fraction transform for an item inside a ViewBox, so
    the fraction is converted against the range the panel itself settled on --
    which is exact, because the range is set from the same limits and then
    frozen.
    """
    import numpy as np

    x, y = (float(value) for value in text.get_position())
    try:
        if text.get_transform() is axes.transData:
            return x, y
    except Exception:                                          # noqa: BLE001
        return x, y
    (left, right), (bottom, top) = axes.get_xlim(), axes.get_ylim()
    if axes.get_xscale() == "log":
        left, right = math.log10(max(left, 1e-300)), math.log10(max(right, 1e-300))
        x = 10 ** (left + x * (right - left))
    else:
        x = left + x * (right - left)
    if axes.get_yscale() == "log":
        bottom, top = (math.log10(max(bottom, 1e-300)),
                       math.log10(max(top, 1e-300)))
        y = 10 ** (bottom + y * (top - bottom))
    else:
        y = bottom + y * (top - bottom)
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y


#: The mathtext this translator understands, as plain Unicode. Deliberately
#: SHORT: a panel writes its labels for a reader, and the whole QC suite uses
#: exactly one mathtext string. Anything outside this table makes the
#: translation incomplete and the panel falls back to matplotlib -- a label
#: that comes out as raw TeX is a figure with a bug printed on its axis, which
#: is worse than a figure drawn in the old library.
_MATHTEXT = {
    r"\mathrm": "", r"\mathbf": "", r"\mathit": "", r"\text": "",
    r"\rm": "", r"\,": " ", r"\;": " ", r"\ ": " ",
    r"\times": "\u00d7", r"\pm": "\u00b1", r"\leq": "\u2264",
    r"\geq": "\u2265", r"\neq": "\u2260", r"\approx": "\u2248",
    r"\infty": "\u221e", r"\cdot": "\u00b7",
    r"\alpha": "\u03b1", r"\beta": "\u03b2", r"\gamma": "\u03b3",
    r"\delta": "\u03b4", r"\Delta": "\u0394", r"\lambda": "\u03bb",
    r"\mu": "\u03bc", r"\pi": "\u03c0", r"\rho": "\u03c1",
    r"\sigma": "\u03c3", r"\Sigma": "\u03a3", r"\chi": "\u03c7",
    r"\tau": "\u03c4", r"\phi": "\u03c6", r"\theta": "\u03b8",
    r"\log": "log", r"\ln": "ln", r"\exp": "exp", r"\max": "max",
    r"\min": "min", r"\mathdefault": "", r"\times10": "\u00d710",
}

#: Digits and signs that have a Unicode superscript, for ``^`` runs.
_SUPERSCRIPT = str.maketrans("0123456789+-=()n",
                             "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
                             "\u207a\u207b\u207c\u207d\u207e\u207f")

#: The same for ``_`` runs.
_SUBSCRIPT = str.maketrans("0123456789+-=()",
                           "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
                           "\u208a\u208b\u208c\u208d\u208e")


def _plain_text(raw) -> Tuple[str, bool]:
    """``(text, understood)`` for a label that may be matplotlib mathtext.

    pyqtgraph draws Qt rich text and has no mathtext at all, so an untouched
    ``$\\sqrt{|\\mathrm{standardised\\ residual}|}$`` reaches the file as exactly
    those characters -- measured on `scale_location`, which is the variance
    homogeneity panel this suite was asked for by name.

    The vocabulary is :data:`_MATHTEXT`, ``\\sqrt`` and the sub/superscript
    runs, and NOTHING ELSE IS GUESSED: a construct outside it comes back
    ``understood=False``, which makes the whole translation incomplete and
    sends the panel back to matplotlib. Half a formula silently rewritten is a
    figure that says something the panel did not.

    Example:
        >>> _plain_text("residual")
        ('residual', True)
        >>> _plain_text(r"$x^2$")
        ('x\u00b2', True)
        >>> _plain_text(r"$\\frac{a}{b}$")[1]
        False
    """
    import re

    text = str(raw)
    if "$" not in text:
        return text, True
    body = text.replace("$", "")
    for token, replacement in _MATHTEXT.items():
        body = body.replace(token, replacement)

    # THE BRACES THAT ARE STILL DOING WORK ARE THE ONES AFTER \sqrt, ^ AND _.
    # Everything else is a group left behind by a removed \mathrm, and it has
    # to go before \sqrt can be read -- its argument is not brace-free until
    # then. Stripping the lot in one pass instead ate the sqrt's own braces
    # and, separately, let a greedy subscript run swallow the "(p)" after
    # `_{10}`. Both measured.
    for _ in range(6):
        stripped = re.sub(r"(?<![A-Za-z^_])\{([^{}]*)\}", r"\1", body)
        if stripped == body:
            break
        body = stripped

    body = re.sub(r"\\sqrt\s*\{([^{}]*)\}", "\u221a(\\1)", body)

    def _script(match, table):
        run = match.group(1)
        converted = run.translate(table)
        # A run with one character outside the table is left ALONE and the
        # leftover ^ or _ then fails the check below, rather than coming out
        # half-raised.
        return converted if all(
            character in table for character in map(ord, run)) else match.group(0)

    body = re.sub(r"\^\{([^{}]*)\}", lambda m: _script(m, _SUPERSCRIPT), body)
    body = re.sub(r"_\{([^{}]*)\}", lambda m: _script(m, _SUBSCRIPT), body)
    body = re.sub(r"\^([0-9A-Za-z+\-])", lambda m: _script(m, _SUPERSCRIPT), body)
    body = re.sub(r"_([0-9A-Za-z+\-])", lambda m: _script(m, _SUBSCRIPT), body)
    body = re.sub(r"\{([^{}]*)\}", r"\1", body)
    # `_` IS ON THIS LIST AND `\lambda_{GC}` IS WHY. A subscript whose
    # characters have no Unicode form comes back with its `_` intact, and
    # "lambda underscore GC" printed on an axis is a label the panel did not
    # write. A subscript that DID convert leaves no `_` behind, so the honest
    # ones cost nothing.
    understood = not any(character in body for character in "\\{}^_")
    return body, understood


def _font(points, look, *, bold: bool = False, italic: bool = False):
    """A QFont whose height is the matplotlib point size, IN PIXELS.

    ``setPointSizeF`` would be the obvious call and it is the wrong one: Qt
    resolves a point size against the SCREEN's logical DPI (96 here), while
    matplotlib resolves it against the FIGURE's (140 in this suite). The same
    number then means two different sizes, and the export comes out with type
    a third too small for the picture it is on. A pixel size has no such
    ambiguity, because the scene is measured in pixels.
    """
    from PySide6.QtGui import QFont

    font = QFont()
    font.setPixelSize(max(1, int(round(look.px(points or 8.0, minimum=1.0)))))
    font.setWeight(QFont.Weight(75 if bold else 50))
    font.setItalic(bool(italic))
    return font


def _add_text(plot, artist, axes, look, report) -> int:
    """Translate one ``Text`` or ``Annotation``.

    Titles, axis labels, tick labels, legend text, and annotations are treated
    as figure chrome. Their colours therefore pass through the print-colour
    resolver instead of being copied as data colours.
    """
    import pyqtgraph as pg

    body, understood = _plain_text(artist.get_text())
    if not body or not body.strip():
        return 0
    if not understood:
        report.missing.append("mathtext this translator does not know")
        return 0
    position = _in_data_coordinates(artist, axes)
    if position is None:
        return 0
    painted = look.paint(_hex(artist.get_color()), "chrome")
    if painted is None:
        return 0
    fill = None
    border = None
    box = getattr(artist, "get_bbox_patch", lambda: None)()
    if box is not None:
        face = look.paint(_hex(box.get_facecolor()), "ground")
        if face:
            fill = pg.mkBrush(face)
        edge = look.paint(_hex(box.get_edgecolor()), "chrome")
        if edge:
            border = pg.mkPen(edge, width=look.px(box.get_linewidth() or 0.6))
    item = pg.TextItem(text=body, color=painted, anchor=_anchor(artist),
                       fill=fill, border=border,
                       angle=-float(artist.get_rotation() or 0.0))
    try:
        item.setFont(_font(artist.get_fontsize(), look,
                           bold=str(artist.get_fontweight()) in ("bold",
                                                                 "heavy"),
                           italic=str(artist.get_style()) == "italic"))
    except Exception:                                          # noqa: BLE001
        pass
    item.setPos(*position)
    item.setZValue(float(artist.get_zorder() or 5))
    plot.addItem(item, ignoreBounds=True)
    return 1


def _add_image(plot, artist, look) -> int:
    """Translate an ``imshow``, colour map and all.

    Preserve the colour map because it encodes data values and must not be
    inverted by print styling.
    """
    import numpy as np
    import pyqtgraph as pg
    from PySide6.QtCore import QRectF

    array = np.asarray(artist.get_array(), dtype=float)
    if array.size == 0:
        return 0
    low, high = artist.get_clim()
    colour_map = artist.get_cmap()
    lut = (np.asarray(colour_map(np.linspace(0.0, 1.0, 256))) * 255).astype(
        np.uint8)
    # pyqtgraph indexes an image as [x, y]; matplotlib's array is
    # [row, column]. Transposing is the whole conversion -- the way UP is
    # already carried, because `imshow` states it in its extent (a default
    # `origin='upper'` returns bottom > top) and `_configure_axes` inverts the
    # view for it. Flipping the array as WELL, which is the obvious-looking
    # thing to do, turns a correlation matrix's diagonal into its
    # anti-diagonal -- measured on `predictor_correlation`, where the identity
    # cells came out at (0,2), (1,1), (2,0).
    image = pg.ImageItem(array.T)
    image.setLookupTable(lut)
    image.setLevels((float(low), float(high)))
    left, right, bottom, top = artist.get_extent()
    image.setRect(QRectF(float(min(left, right)), float(min(bottom, top)),
                         abs(float(right - left)), abs(float(top - bottom))))
    image.setZValue(float(artist.get_zorder() or 0))
    plot.addItem(image)
    # THE COLOUR MAP IS NOT A LIST OF MARKS, so it is deliberately NOT fed to
    # the legibility check. A diverging map's midpoint is pale BY DESIGN --
    # RdBu_r at r = 0 is near white -- and reporting it would fire the warning
    # on every correlation panel ever written, which is the "a warning that
    # fires on every figure is a warning nobody reads" failure the floor was
    # chosen to avoid. A ramp is read against its own bar, not against the page.
    return 1


# --------------------------------------------------------------------------- #
#  Translating one axes
# --------------------------------------------------------------------------- #

def _axis_pen(look):
    import pyqtgraph as pg

    ink = look.paint("#FFFFFF", "chrome") or "#222222"
    return pg.mkPen(ink, width=look.px(0.8))


def _css_size(points, look) -> str:
    """A matplotlib point size as the CSS pyqtgraph's label markup wants.

    In ``px``, for the reason :func:`_font` gives: a point in Qt's rich text
    is a point at the screen's DPI, and this scene is measured at the
    figure's.
    """
    return f"{max(1, int(round(look.px(points or 8.0, minimum=1.0))))}px"


def _configure_axes(plot, axes, look, report) -> None:
    """Titles, labels, ranges, ticks and scales: the chrome, as configuration.

    The axis ink is asked of the same shared rule as everything else, with the
    kind fixed at ``chrome``, so a suite saved from a dark session comes out
    with dark axes on a light page and a suite saved from a light one is
    untouched.
    """
    ink = look.paint("#FFFFFF", "chrome") or "#222222"
    for name, matplotlib_axis in (("bottom", axes.xaxis), ("left", axes.yaxis)):
        axis = plot.getAxis(name)
        axis.setPen(_axis_pen(look))
        axis.setTextPen(_axis_pen(look))
        # THE TYPE SIZE IS THE PANEL'S, NOT A DEFAULT. The house style pins
        # ticks at 6 pt and labels at 7 pt precisely so a page of twenty
        # panels reads as one figure; a renderer that substitutes pyqtgraph's
        # own defaults undoes that silently and the two libraries stop
        # agreeing about the thing they most obviously should.
        try:
            ticks = matplotlib_axis.get_ticklabels()
            axis.setStyle(tickFont=_font(
                ticks[0].get_fontsize() if ticks else 6.0, look))
        except Exception:                                      # noqa: BLE001
            pass
    plot.showAxis("top", False)
    plot.showAxis("right", False)
    plot.showGrid(x=False, y=False)
    plot.setMenuEnabled(False)
    for name, raw, size in (
            ("title", axes.get_title(), axes.title.get_fontsize()),
            ("bottom", axes.get_xlabel(), axes.xaxis.label.get_fontsize()),
            ("left", axes.get_ylabel(), axes.yaxis.label.get_fontsize())):
        if not raw:
            continue
        body, understood = _plain_text(raw)
        if not understood:
            report.missing.append("mathtext this translator does not know")
            continue
        if name == "title":
            plot.setTitle(body, color=ink, size=_css_size(size, look))
        else:
            plot.setLabel(name, body, color=ink,
                          **{"font-size": _css_size(size, look)})
    if not axes.axison:
        plot.hideAxis("bottom")
        plot.hideAxis("left")

    log_x = axes.get_xscale() == "log"
    log_y = axes.get_yscale() == "log"
    if log_x or log_y:
        plot.setLogMode(x=log_x, y=log_y)
    (left, right), (bottom, top) = axes.get_xlim(), axes.get_ylim()
    if top < bottom:
        plot.getViewBox().invertY(True)
        bottom, top = top, bottom
    if log_x:
        left, right = (math.log10(max(left, 1e-300)),
                       math.log10(max(right, 1e-300)))
    if log_y:
        bottom, top = (math.log10(max(bottom, 1e-300)),
                       math.log10(max(top, 1e-300)))
    plot.setXRange(left, right, padding=0)
    plot.setYRange(bottom, top, padding=0)
    plot.getViewBox().disableAutoRange()
    _carry_ticks(plot, axes)


def _carry_ticks(plot, axes) -> None:
    """Copy explicit tick labels across.

    A CATEGORICAL AXIS IS NOT A NUMERIC ONE WITH NICE LABELS. The screen-level
    panels put plate and row names on the axis with ``set_xticklabels``, and a
    renderer that re-derives ticks from the range draws 0, 1, 2 where the panel
    wrote plate1, plate2, plate3 -- a different figure, silently.
    """
    for name, matplotlib_axis in (("bottom", axes.xaxis), ("left", axes.yaxis)):
        try:
            locations = list(matplotlib_axis.get_ticklocs())
            labels = [tick.get_text()
                      for tick in matplotlib_axis.get_ticklabels()]
        except Exception:                                      # noqa: BLE001
            continue
        if not labels or len(labels) != len(locations):
            continue
        if not any(label and not _looks_numeric(label) for label in labels):
            continue
        converted = [_plain_text(label) for label in labels]
        if not all(understood for _, understood in converted):
            # A LOG AXIS WRITES ITS TICKS IN MATHTEXT, and carrying those
            # across raw put `$\mathdefault{10^{-1}}$` down the side of the
            # design-spectrum panel. pyqtgraph's own log axis writes correct
            # labels, so the honest move is to leave them to it rather than to
            # print a formatter's source code on the figure.
            continue
        plot.getAxis(name).setTicks(
            [[(float(position), body)
              for position, (body, _) in zip(locations, converted) if body]])


def _looks_numeric(label: str) -> bool:
    try:
        float(str(label).replace("−", "-"))
        return True
    except ValueError:
        return False


def _is_chrome_artist(artist) -> bool:
    """True for a spine, an axis or a tick: configuration, not content.

    Checked by TYPE rather than by class NAME, because matplotlib subclasses
    its own chrome -- a colour bar's frame is a ``_ColorbarSpine``, which is a
    Spine and does not answer to the name. A name test dropped exactly one
    panel of the suite for that reason, measured, which is how this function
    came to exist.
    """
    from matplotlib.axis import Axis, Tick
    from matplotlib.spines import Spine

    return isinstance(artist, (Spine, Axis, Tick))


def _colorbar_of(axes):
    """The ``Colorbar`` this axes IS, or None. matplotlib stamps it on the axes."""
    return getattr(axes, "_colorbar", None)


def _translate_colorbar(plot, axes, look, report: SceneReport) -> None:
    """Translate a colour bar as the ramp it is, rather than as its mesh.

    A matplotlib colour bar is a ``QuadMesh`` of 256 quadrilaterals plus a
    ``_ColorbarSpine``, and translating it quad by quad would be 256 graphics
    items drawing a gradient that one image draws exactly. So the RAMP is read
    off the mappable's own colour map -- the same object the panel handed to
    ``imshow``, so the bar and the image cannot come out keyed differently,
    which is the one failure a colour bar has.
    """
    import numpy as np
    import pyqtgraph as pg
    from PySide6.QtCore import QRectF

    bar = _colorbar_of(axes)
    ink = look.paint("#FFFFFF", "chrome") or "#222222"
    low, high = (float(value) for value in bar.mappable.get_clim())
    ramp = np.linspace(low, high, 256).reshape(1, 256)
    lut = (np.asarray(bar.mappable.get_cmap()(np.linspace(0, 1, 256))) * 255
           ).astype(np.uint8)
    image = pg.ImageItem(ramp)
    image.setLookupTable(lut)
    image.setLevels((low, high))
    image.setRect(QRectF(0.0, low, 1.0, high - low))
    plot.addItem(image)
    report.items += 1
    plot.showAxis("left", False)
    plot.showAxis("bottom", False)
    plot.showAxis("top", False)
    plot.showAxis("right", True)
    right = plot.getAxis("right")
    right.setPen(_axis_pen(look))
    right.setTextPen(_axis_pen(look))
    plot.setXRange(0, 1, padding=0)
    plot.setYRange(low, high, padding=0)
    plot.getViewBox().disableAutoRange()
    plot.setMenuEnabled(False)
    label = bar.ax.get_ylabel() or bar.ax.get_xlabel()
    if label:
        plot.setLabel("right", label, color=ink,
                      **{"font-size": _css_size(
                          bar.ax.yaxis.label.get_fontsize(), look)})
    try:
        ticks = bar.ax.yaxis.get_ticklabels()
        right.setStyle(tickFont=_font(
            ticks[0].get_fontsize() if ticks else 6.0, look))
    except Exception:                                          # noqa: BLE001
        pass


def _translate_axes(plot, axes, look, report: SceneReport) -> None:
    """Put everything on one matplotlib ``Axes`` into one pyqtgraph plot."""
    if _colorbar_of(axes) is not None:
        _translate_colorbar(plot, axes, look, report)
        return
    _configure_axes(plot, axes, look, report)
    rectangles = []
    for artist in axes.get_children():
        name = type(artist).__name__
        if artist is axes.patch:
            continue
        if name in IGNORED or _is_chrome_artist(artist):
            continue
        if name == "Line2D":
            report.items += _add_line(plot, artist, axes, look)
        elif name in ("Text", "Annotation"):
            if getattr(artist, "arrow_patch", None) is not None:
                report.notes.append("an annotation's arrow was not carried")
            report.items += _add_text(plot, artist, axes, look, report)
        elif name == "Rectangle":
            rectangles.append(artist)
        elif name == "PathCollection":
            report.items += _add_path_collection(plot, artist, look)
        elif name == "LineCollection":
            report.items += _add_line_collection(plot, artist, look)
        elif name == "PolyCollection":
            report.items += _add_poly_collection(plot, artist, look)
        elif name == "AxesImage":
            report.items += _add_image(plot, artist, look)
        elif name == "Legend":
            report.items += _add_legend(plot, artist, look)
        elif name not in CARRIED:
            report.missing.append(name)
    if rectangles:
        report.items += _add_rectangles(plot, rectangles, axes, look)


def _add_legend(plot, legend, look) -> int:
    """Translate a legend as its text, beside a swatch of each entry's colour.

    pyqtgraph's own ``LegendItem`` is built by naming curves as they are
    added, which cannot express a legend matplotlib assembled from handles the
    panel passed in by hand -- and three of these panels do exactly that. So
    the entries are read off the finished legend, which is the same rule the
    rest of this module follows: translate what was drawn, not what was meant.
    """
    import pyqtgraph as pg

    entries = []
    for text in legend.get_texts():
        body = text.get_text()
        if body:
            entries.append(str(body))
    if not entries:
        return 0
    ink = look.paint(_hex(legend.get_texts()[0].get_color()), "chrome")
    item = pg.LegendItem(offset=(30, 20), labelTextColor=ink or "#222222",
                         labelTextSize="7pt")
    item.setParentItem(plot.getViewBox())
    for entry, handle in zip(entries, legend.legend_handles):
        colour = None
        for getter in ("get_color", "get_facecolor", "get_edgecolor"):
            try:
                colour = _hex(getattr(handle, getter)())
            except Exception:                                  # noqa: BLE001
                colour = None
            if colour:
                break
        sample = pg.PlotDataItem(
            [0, 1], [0, 0],
            pen=pg.mkPen(look.paint(colour, "data") or "#B4B4B4", width=2))
        item.addItem(sample, entry)
    return 1


# --------------------------------------------------------------------------- #
#  Translating a figure
# --------------------------------------------------------------------------- #

def _grid_position(axes, index: int) -> Tuple[int, int]:
    """``(row, column)`` for one axes on the page.

    Read off the subplot spec where there is one, which is what keeps a 2x3
    diagnostic sheet a 2x3 sheet. A figure whose axes were placed by hand falls
    back to one row, which is honest: this module does not attempt to
    reconstruct a hand-placed layout, and a sheet in the wrong order would be a
    worse answer than a sheet in a line.

    THE SPEC HAS TO BE THE TOPMOST ONE. ``Figure.colorbar`` replaces its
    mappable's subplot spec with one from a NEW gridspec nested inside the old
    cell, so a panel that was cell 10 of a 6x4 page answers ``(1, 2, 0, 0)``
    afterwards -- cell 0. On the combined QC page that put the correlation
    panel on top of the first panel, and Qt said so:
    ``QGridLayoutEngine::addItem: Can't add ... at cell (0, 0) because it's
    already taken``. ``get_topmost_subplotspec`` answers in the page's own
    grid, which is the grid the reader sees.
    """
    try:
        spec = axes.get_subplotspec().get_topmost_subplotspec()
        _, columns, start, _ = spec.get_geometry()
        return int(start // columns), int(start % columns)
    except Exception:                                          # noqa: BLE001
        return 0, index


def _positions(figure) -> List[Tuple[int, int, int]]:
    """``(row, column, colspan)`` per axes, on a grid of HALF columns.

    EVERY PANEL SPANS TWO SUB-COLUMNS AND A COLOUR BAR TAKES ONE OF THEM. A
    colour bar has no cell of its own on the page it came from -- matplotlib
    takes its room out of the panel it keys -- so giving it a full column of
    its own either overlaps the next panel or doubles the page's width. The
    half-column grid gives it exactly the room it had, in the place it had it,
    and costs a page with no colour bars nothing at all.

    A colour bar is also placed beside WHAT IT KEYS rather than where its own
    spec says, for the reason :func:`_grid_position` records: the spec belongs
    to a gridspec nested inside the panel's cell and answers in its
    coordinates.
    """
    plain = [_grid_position(axes, index)
             for index, axes in enumerate(figure.axes)]
    positions = [(row, column * 2, 2) for row, column in plain]
    for index, axes in enumerate(figure.axes):
        bar = _colorbar_of(axes)
        if bar is None:
            continue
        try:
            parent = list(figure.axes).index(bar.mappable.axes)
        except (ValueError, AttributeError):                   # noqa: BLE001
            continue
        row, column, _ = positions[parent]
        positions[parent] = (row, column, 1)
        positions[index] = (row, column + 1, 1)
    return positions


def _columns(figure) -> int:
    """How many sub-columns the figure's axes occupy."""
    return max([column + span for _, column, span in _positions(figure)] + [0])


def build_scene(figure, *, mode=None, dpi=None):
    """Translate a matplotlib ``Figure`` into a pyqtgraph scene.

    :param figure: a drawn matplotlib Figure. It is NOT modified: everything
        here is a read.
    :param mode: a :data:`spacr.figure_style.SAVE_MODES` value, or None to ask
        the preference.
    :param dpi: output resolution used to size the scene. By default, use the
        figure's current resolution.
    :returns: ``(widget, report)``. The widget is a
        ``pyqtgraph.GraphicsLayoutWidget`` holding one plot per axes; the
        :class:`SceneReport` says whether the translation was COMPLETE, and a
        caller must not write an incomplete one.
    """
    import pyqtgraph as pg

    width, height = (float(value) for value in figure.get_size_inches())
    dpi = float(dpi if dpi is not None else (figure.get_dpi() or 100.0))
    look = _Look(mode, dpi=dpi)
    widget = pg.GraphicsLayoutWidget(size=(int(width * dpi), int(height * dpi)))
    widget.resize(int(width * dpi), int(height * dpi))
    if look.ground:
        widget.setBackground(look.ground)
    else:
        widget.setBackground(None)

    report = SceneReport()
    positions = _positions(figure)
    title = getattr(figure, "_suptitle", None)
    # THE SUPTITLE TAKES ROW 0 AND EVERYTHING ELSE MOVES DOWN. pyqtgraph's
    # `addLabel(row=-1)` is not "the row above"; it is an invalid row, and Qt
    # says `QGraphicsGridLayout::addItem: invalid row/column: -1` and drops the
    # label -- so the three multi-panel diagnostic sheets lost their titles
    # silently, which for a page called "Screen design diagnostics" is the one
    # line a reader needs.
    offset = 1 if title is not None and title.get_text() else 0
    for index, axes in enumerate(figure.axes):
        row, column, span = positions[index]
        plot = widget.addPlot(row=row + offset, col=column, colspan=span)
        _translate_axes(plot, axes, look, report)
        if _colorbar_of(axes) is not None:
            # A KEY IS NOT A PANEL. A GraphicsLayout gives every column the
            # same width, so a colour bar added as a second column took HALF
            # the page -- measured on `predictor_correlation`, where a 3x3
            # matrix came out smaller than its own legend. The width is fixed
            # here, on the layout, rather than on the item: a maximum width on
            # the PlotItem shrinks the item inside a cell that stays half the
            # page and leaves it floating in the middle of the white.
            widget.ci.layout.setColumnFixedWidth(column, COLORBAR_PX)
        report.axes += 1
    if offset:
        ink = look.paint(_hex(title.get_color()), "chrome") or "#222222"
        widget.addLabel(str(title.get_text()), row=0, col=0,
                        colspan=max(1, _columns(figure)), color=ink,
                        size=_css_size(title.get_fontsize(), look),
                        bold=str(title.get_fontweight()) in ("bold", "heavy"))
    _lay_out(widget)
    report.data_colours = list(dict.fromkeys(look.data_colours))
    return widget, report


def _lay_out(widget) -> None:
    """Lay the scene out AND paint it once, before anything measures it.

    THE PAINT IS NOT OPTIONAL AND THAT IS THE SURPRISE. A widget that was
    never shown has never been laid out, and -- worse -- a pyqtgraph
    ``AxisItem`` learns how wide it needs to be while it PAINTS: ``textWidth``
    is measured in ``generateDrawSpecs``, and until then the axis reserves the
    default. Measured on the QC suite with only a layout pass: the x-axis
    label sat on top of the tick labels and a categorical y axis showed its
    tick MARKS with no room for a single character of text. Grabbing the
    widget once forces the paint, and the second ``activate`` spends the
    measurement the paint just made.

    This is the same trap ``FastPlot.snapshot`` records for a plot on a page
    nobody has raised, and it has the same shape: ask for the work rather than
    assume it happened.
    """
    from PySide6.QtWidgets import QApplication

    try:
        application = QApplication.instance()
        widget.ci.layout.activate()
        widget.grab()
        if application is not None:
            application.processEvents()
        widget.ci.layout.activate()
    except Exception:                                          # noqa: BLE001
        pass


def export_scene(widget, path) -> Optional[str]:
    """Write a built scene to ``path``. PDF, SVG and PNG, by the name.

    THE EXPORTER IS THE ONE THE TABS USE. ``FastPlot._export_pdf`` and
    ``_export_svg`` are classmethods over a plot item, and both go through
    ``_paint_scene``, which turns pyqtgraph's export mode on for every mark
    before painting -- without it a ScatterPlotItem copies its cached marker
    PIXMAPS into the vector file, and a PDF full of little bitmaps of a dot is
    a PDF that claims to be vector and is not. Writing a second exporter here
    would recreate the same exporter duplication one level down.
    """
    from ..qt.widgets.fast_plots import FastPlot

    item = widget.ci
    name = str(path).lower()
    folder = os.path.dirname(os.path.abspath(str(path)))
    if folder:
        os.makedirs(folder, exist_ok=True)
    if name.endswith(".pdf"):
        FastPlot._export_pdf(item, path)
    elif name.endswith(".svg"):
        FastPlot._export_svg(item, path)
    else:
        from pyqtgraph import exporters

        exporter = exporters.ImageExporter(item)
        exporter.export(str(path))
    return str(path) if os.path.exists(str(path)) else None


def render_figure(figure, path, *, fmt=None, mode=None, dpi=None,
                  announce=True, title=None):
    """Write ``figure`` as a pyqtgraph render. ``None`` when it could not be.

    :param figure: the drawn matplotlib figure, used as the GEOMETRY. It is
        not written and not modified.
    :param path: destination; the extension is settled by
        :func:`spacr.plot.figure_path` BEFORE the exporter sees it, because
        pyqtgraph decides what it writes from the file name.
    :param dpi: output resolution used to size the scene. By default, use the
        figure's current resolution.
    :param announce: put the file in the gallery as well as on disk.
    :returns: ``(written_path, report)``, or ``(None, report)``. A caller that
        gets None writes the matplotlib page and says why.

    NOTHING HERE RAISES. A figure is the last thing a run produces and the
    least important thing it produces; losing an hour's fit to a renderer is
    the worst trade in this module, which is the same rule
    :func:`spacr.figure_sink.publish` follows.
    """
    from ..plot import figure_path

    report = SceneReport()
    ready, why = pyqtgraph_ready()
    if not ready:
        report.missing.append("pyqtgraph")
        report.notes.append(why)
        return None, report
    widget = None
    try:
        scene_options = {"mode": mode}
        if dpi is not None:
            scene_options["dpi"] = dpi
        widget, report = build_scene(figure, **scene_options)
        if not report.complete:
            return None, report
        written = export_scene(widget, figure_path(path, fmt))
    except Exception as error:                                 # noqa: BLE001
        report.missing.append(type(error).__name__)
        report.notes.append(f"{type(error).__name__}: {error}")
        return None, report
    finally:
        if widget is not None:
            try:
                widget.deleteLater()
            except Exception:                                  # noqa: BLE001
                pass
    if written is None:
        report.missing.append("the exporter wrote nothing")
        return None, report
    _warn_about_data_colours(report)
    if announce:
        from ..figure_sink import publish_file

        publish_file(written, title=title or os.path.basename(written))
    return written, report


def write_figure(figure, path, *, fmt=None, dpi=None, renderer=None,
                 announce=True, title=None, **savefig):
    """Write a generated figure with the screen's renderer where that is possible.

    THE ONE CALL A GENERATED-FIGURE MODULE MAKES. It replaces a
    :func:`spacr.figure_sink.publish` and behaves exactly like one when
    pyqtgraph is not the renderer, so a module adopting it cannot lose the
    gallery tile, the format preference or the resolution preference on the
    way past.

    :param renderer: force one of :data:`RENDERERS`; None asks
        :func:`scene_renderer`.
    :param dpi: force an output resolution; otherwise use the configured
        figure preference.
    :param savefig: forwarded to matplotlib on the fallback path (``bbox_inches``
        and friends). pyqtgraph has no use for them.
    :returns: ``(path, renderer, reason)``. ``reason`` is why it is not the
        screen's renderer, and it is empty only when it is.

    THE FALLBACK IS NOT A FAILURE. A panel whose translation is incomplete --
    an artist nobody has taught this module, a formula it will not guess at --
    is written by matplotlib exactly as it always was. The picture is never the
    thing that is lost.
    """
    chosen, why = scene_renderer(renderer)
    if chosen == "pyqtgraph":
        from ..plot import deliverable_dpi, figure_output_preferences

        requested_dpi = dpi
        if requested_dpi is None:
            _preferred_format, requested_dpi = figure_output_preferences()
        scene_dpi = deliverable_dpi(figure, requested_dpi, path=path)
        written, report = render_figure(
            figure, path, fmt=fmt, dpi=scene_dpi, announce=announce,
            title=title)
        if written:
            return written, "pyqtgraph", ""
        why = report.reason() or (report.notes[-1] if report.notes
                                  else "the scene could not be built")
    from ..figure_sink import publish

    if announce:
        written = publish(figure, path, fmt=fmt, dpi=dpi, **savefig)
    else:
        from ..plot import save_figure

        written = save_figure(figure, path, fmt=fmt, dpi=dpi, **savefig)
    return written, "matplotlib", why


def _warn_about_data_colours(report: SceneReport) -> None:
    """Warn about data colours with poor contrast on the saved page.

    Name offending colours without replacing them. Use the figure-style
    contrast floor, which is tuned to flag colours less legible than the
    house-style reference greys without warning on every figure.
    """
    from ..figure_style import (illegible_colour_warning, illegible_colours,
                                saved_figure_appearance)

    look = saved_figure_appearance()
    if not look.flip or not report.data_colours:
        return
    offenders = illegible_colours(report.data_colours,
                                  look.ground or "#FFFFFF")
    if offenders:
        print(illegible_colour_warning(offenders))


__all__ = ["CARRIED", "COLORBAR_PX", "RENDERERS", "SceneReport",
           "build_scene", "export_scene", "pyqtgraph_ready", "render_figure",
           "requested_renderer", "scene_renderer", "write_figure"]
