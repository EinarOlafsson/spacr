"""The plate heatmaps: every plate of a screen as ONE panel, wells square.

Asked for on 2026-08-16, and it is the loudest of the figure complaints:

    "the lpates look super small on the collected figure please"

A plate is 16 rows by 24 columns, so ONE plate is a wide, short picture.
:func:`spacr.plot.plot_plates` drew the four plates of a screen side by side
on a 40 x 5 inch figure -- an 8:1 strip. Letterboxed into a square tile of
the figure grid that strip is one eighth of the tile high and uses about
12% of its area, which is exactly what "super small" describes. The wells
were not square either: 0.275 x 0.241 inches, a 1.14:1 rectangle.

THE CHOICE MADE HERE, of the three instruction 124 offers: **all the plates
as a SMALL MULTIPLE inside one panel** (option b).

* Four plates of one measurement are ONE figure, not four. They answer a
  single question -- does this measurement depend on where a well sits --
  and that question is only answerable by comparing plates.
* Stacked 2 x 2 the composite is 1.3:1 instead of 8:1, so it fills a tile
  instead of lying along the top of one. Same tile, same wells, ~6x the
  area and ~2x the linear well size.
* One panel means ONE COLOUR SCALE. Drawn per plate, as it was, the same
  blue meant 0.24 on plate 1 and 0.28 on plate 3 -- a 16% difference,
  adjacent, with four separate colour bars -- and comparing plates by eye
  was not merely hard but wrong.

THE WELLS STAY SQUARE. Every plate is drawn with ``aspect='equal'`` and the
figure is SIZED FROM THE GRID rather than the grid squeezed into a figure,
so square is what the layout produces rather than what it survives. A plate
heatmap with rectangular wells is not a heatmap of a plate: positional
artefacts, the whole reason to look at one, stop being visible. That is what
instruction 117 exists for.

TWO THINGS THAT WERE WRONG WITH THE PICTURE, NOT JUST ITS SIZE
---------------------------------------------------------------

1. **A well that was never measured was drawn as a measurement of zero.**
   ``generate_plate_heatmap`` ends in ``.fillna(0)``, and on the tsg101
   screen 155 of a plate's 384 wells carry data -- so 54% of every plate
   panel was a solid block of "the lowest value there is", in the darkest
   ink, indistinguishable from a real well that scored badly.

2. **Those zeros then set the colour scale.** ``min_max='allq'`` takes the
   2nd and 98th percentiles OF THE MATRIX, and a matrix that is 54% zeros
   has a 2nd percentile of 0. Measured on the real screen the drawn range
   was 0.000-0.243 where the range of the wells that exist is 0.060-0.273,
   and on plate 3 the top of the scale was 0.281 against a real 98th
   percentile of 0.408: the whole upper third of that plate's dynamic range
   was saturated flat by wells that do not exist.

   Here an absent well is absent -- painted as a neutral wash so a hole
   reads as a hole -- and the colour scale is computed over the wells that
   were measured.

The visual system is ``.claude/skills/apicomplexan-figures``; the style is
applied with :func:`spacr.figures.style.figure_style`, a context manager,
and NEVER by writing rcParams globally. spaCR draws from a long-lived GUI,
where a global style change restyles every later figure in the session.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .panels import Panel
from .style import (TYPE_SCALE, WEIGHTS, Palette, figure_style, resolve_ink,
                    theme_target)

#: The composite the small multiple aims at. Slightly wider than square
#: because the colour bar takes a strip along the bottom; a tile in the
#: figure grid is square-ish, so the closer the composite is to 1 the more
#: of the tile the plates get.
TARGET_ASPECT = 1.2

#: Figure width in inches. The double-column measure the sheet uses, so a
#: plate panel and a regression sheet are the same width on a page.
WIDTH = 7.0

#: Margins, in inches, measured from what the type actually needs: the row
#: letters on the left, the column numbers and the colour bar below, the
#: plate name above.
#: The gaps are small because the labels are SHARED: only the left column
#: carries row letters and only the bottom row carries column numbers, so
#: nothing has to fit between two plates except the lower one's name. A
#: small multiple should read as one block, not as four pictures that
#: happen to be near each other.
MARGIN = {"left": 0.28, "right": 0.10, "top": 0.20, "bottom": 0.58,
          "wspace": 0.18, "hspace": 0.22}

#: Where the shared colour bar sits, in inches from the bottom of the
#: figure, and how thick it is. Below it go its own tick labels and the
#: measurement's name, which is why it is not at the very bottom.
BAR = {"bottom": 0.24, "height": 0.075, "width": 2.4}

#: How dark the "no measurement here" wash is, as a fraction of the ink.
#: Neutral, so an empty well differs from a measured one in HUE and not
#: only in lightness -- a difference that survives a bad monitor and a
#: greyscale print, which a lightness-only difference does not.
EMPTY_WASH_ALPHA = 0.09


# --------------------------------------------------------------------------- #
#  The colour ramp
# --------------------------------------------------------------------------- #

def plate_ramp(target: str = "screen"):
    """The single-hue ramp a plate is painted with, light to dark.

    The skill is explicit: "Sequential encodings (a p-value, a score) use a
    single-hue blue ramp, light→dark." Every internal call site passed
    ``cmap='viridis'``, which is not in the palette and was never chosen --
    it is the literal the first version was written with.

    ONLY THE DARK END RESOLVES AGAINST THE GROUND, exactly as the ink does
    in :func:`spacr.figures.style.resolve_ink`. On paper the ramp runs to
    NAVY; on spaCR's dark theme NAVY is within a hair of the background, so
    a well at the top of the scale would be confusable with a well that has
    no measurement at all. The hues are the palette's and do not change --
    the ramp simply stops one stop earlier where the page is dark.

    :param target: ``'screen'`` or ``'print'``.
    :returns: a matplotlib ``Colormap`` whose "bad" colour is transparent,
        so masked (unmeasured) wells show the wash beneath them.
    """
    from matplotlib.colors import LinearSegmentedColormap

    if target == "print":
        stops = ["#F7FAFD", Palette.BLUE_LIGHT, Palette.BLUE, Palette.NAVY]
    else:
        stops = ["#E8EDEE", Palette.BLUE_LIGHT, Palette.BLUE]
    ramp = LinearSegmentedColormap.from_list(f"spacr_plate_{target}", stops)
    ramp.set_bad("none")
    return ramp


# --------------------------------------------------------------------------- #
#  The wells
# --------------------------------------------------------------------------- #

def plate_names(frame) -> List[str]:
    """The plates in this frame, in the order their identifiers give.

    The plate is the LEADING token of ``prc``, which is what
    :func:`spacr.plot.plot_plates` has always split a screen on.

    WHICH MEANS A 4-TOKEN IDENTIFIER NAMES ITS EXPERIMENT, NOT ITS PLATE.
    ``exp_plate1_r1_c1`` and ``exp_plate2_r1_c1`` come back as the single
    name ``exp``, and :func:`spacr.plot.generate_plate_heatmap` -- which
    reads the position from the LAST two tokens and takes the plate it was
    asked for as the authority on the rest -- then draws every row of both
    plates as one grid, averaging the two plates well by well. That is the
    behaviour of the code this replaced, unchanged here so that the picture
    does not move under a screen that already has one; a screen whose plates
    must stay apart puts the plate first.
    """
    if "prc" not in getattr(frame, "columns", ()):
        return []
    tokens = frame["prc"].astype(str).str.split("_")
    heads = [parts[0] for parts in tokens if parts]
    seen, order = set(), []
    for name in heads:
        if name not in seen:
            seen.add(name)
            order.append(name)
    return order


def full_plate_grid(rows: Sequence[int], columns: Sequence[int]) -> Tuple[int, int]:
    """The plate the measured wells sit on, not the box that bounds them.

    THE EDGE HAS TO BE THE EDGE. Pivoting only the wells that carry data
    drops an entirely unused column, so a screen that never used columns 1-3
    -- which the tsg101 screen does not -- drew its first measured column
    hard against the left spine. Every edge effect then reads one plate
    position out, and an edge effect is the artefact a plate heatmap exists
    to show.

    :returns: ``(n_rows, n_columns)`` of the smallest standard format that
        contains every measured well, or the bounding box when the wells fit
        no standard plate (a partial or non-standard layout).
    """
    from .. import schema as _schema

    if not len(rows) or not len(columns):
        return 0, 0
    top, right = int(max(rows)), int(max(columns))
    fmt = _schema.plate_format_for(top, right)
    if fmt is None:
        return top, right
    n_rows, n_columns = _schema.PLATE_FORMATS[fmt]
    return int(n_rows), int(n_columns)


def well_matrices(frame, variable: str, *, grouping: str = "mean",
                  min_count=0, plates: Optional[Sequence[str]] = None):
    """One matrix per plate, on a shared grid, with absent wells as ``nan``.

    Wraps :func:`spacr.plot.generate_plate_heatmap` -- which is where the
    prc parsing, the letter walk past row P and the min_count filter live --
    and undoes the one thing it does that a picture must not inherit: its
    ``.fillna(0)``. A well with no rows and a well that measured zero are
    the same cell afterwards, so the count map is fetched alongside the
    value map and a well with a count of zero is masked back out.

    A well with rows but nothing numeric in any of them is masked too. The
    count map counts ROWS, and the aggregation coerces the variable with
    ``errors='coerce'``, so such a well aggregates to NaN and is filled with
    the same invented zero one step further in.

    :param frame: long-format frame with a ``prc`` column.
    :param variable: the measurement column to aggregate.
    :param grouping: ``'mean'``, ``'sum'`` or ``'count'``.
    :param min_count: wells with fewer rows than this are dropped, and then
        read as absent rather than as zero.
    :param plates: draw only these plates, in this order.
    :returns: ``(names, matrices, (n_rows, n_columns))``.
    """
    import pandas as pd

    from ..plot import generate_plate_heatmap
    from .. import schema as _schema

    names = list(plates) if plates is not None else plate_names(frame)
    if not names:
        return [], [], (0, 0)

    # Only the columns the aggregation needs. generate_plate_heatmap writes
    # plateID/rowID/columnID onto the frame it is given, and that must not
    # land on the caller's table; a two-column projection is also a great
    # deal cheaper than copying a measurement frame.
    wanted = ["prc"] + ([variable] if variable in frame.columns else [])
    work = frame.loc[:, wanted].copy()

    # ONE PLATE'S ROWS, ONCE. generate_plate_heatmap re-parses every prc in
    # the frame it is handed, on every call, and this asks it for two maps
    # per plate rather than one -- so on a million-row measurement frame the
    # naive loop costs 2n per plate. Handing it only the rows of the plate
    # being drawn makes that 2n across ALL the plates, which is less work
    # than the single-map version was doing.
    #
    # Only for the plain 3-token identifier. A LONGER prc carries an
    # experiment prefix, and generate_plate_heatmap then treats every row as
    # belonging to the plate it was asked for -- so splitting on the leading
    # token would silently change which wells are drawn.
    text = work["prc"].astype(str)
    head = None
    if text.str.count(_schema.KEY_SEPARATOR).eq(2).all():
        head = text.str.split(_schema.KEY_SEPARATOR, n=1).str[0].to_numpy()

    # A ROW IS NOT A MEASUREMENT, and the count map counts rows.
    # generate_plate_heatmap coerces the variable with ``errors='coerce'``,
    # so a well whose every row holds nothing numeric -- an empty cell, an
    # 'n/a', a merge that did not find a match -- aggregates to NaN, is
    # filled with 0, and, having a row count above zero, survives the mask
    # below as a measurement of zero. That is the same defect one step
    # further in, and it sets the bottom of the shared scale in the same
    # way. Which rows carry a number is therefore worked out ONCE, and only
    # when some row does not: on a clean frame this costs one pass and no
    # extra heatmap at all.
    readable = None
    if grouping != "count" and variable in work.columns:
        numeric = pd.to_numeric(work[variable], errors="coerce").notna().to_numpy()
        if not numeric.all():
            readable = numeric

    maps, counts = [], []
    for name in names:
        # .copy(), because generate_plate_heatmap assigns columns onto the
        # frame it is given and a boolean-mask slice is a view: pandas warns
        # (SettingWithCopyWarning) and, under copy-on-write, the assignment
        # would land somewhere the next call cannot see.
        on_plate = None if head is None else head == str(name)
        subset = work if on_plate is None else work[on_plate].copy()
        values, _limits = generate_plate_heatmap(
            subset, name, variable, grouping, "all", min_count)
        # The count map IS the value map when the caller asked for counts,
        # so that case does not pay for a second pass over the frame.
        if grouping == "count":
            present = values
        else:
            present = generate_plate_heatmap(
                subset, name, variable, "count", "all", min_count)[0]
            if readable is not None:
                present = _drop_unreadable_wells(
                    generate_plate_heatmap, present, subset,
                    readable if on_plate is None else readable[on_plate],
                    name, variable)
        maps.append(values)
        counts.append(present)

    rows = sorted({index for m in maps for index in
                   (_schema.row_index(label) for label in m.index)
                   if index is not None})
    columns = sorted({index for m in maps for index in
                      (_schema.column_index(label) for label in m.columns)
                      if index is not None})
    n_rows, n_columns = full_plate_grid(rows, columns)
    if not n_rows or not n_columns:
        return names, [], (0, 0)

    row_ids = [_schema.row_id(i) for i in range(1, n_rows + 1)]
    column_ids = [_schema.column_id(i) for i in range(1, n_columns + 1)]

    matrices = []
    for values, present in zip(maps, counts):
        grid = values.reindex(index=row_ids, columns=column_ids)
        seen = present.reindex(index=row_ids, columns=column_ids)
        block = grid.to_numpy(dtype="float64")
        # A count of zero -- or a well the reindex invented -- is a well that
        # was not measured. It is not a measurement of zero and must not be
        # painted as one, nor counted when the colour scale is chosen.
        empty = ~(seen.to_numpy(dtype="float64") > 0)
        block[empty] = np.nan
        matrices.append(block)
    return names, matrices, (n_rows, n_columns)


def _drop_unreadable_wells(heatmap, present, subset, readable, name, variable):
    """Zero the presence of a well that has rows but no number in any of them.

    ``min_count`` KEEPS ITS MEANING -- it counts rows, as it always has, so
    the readable-row map is taken at ``min_count`` 0 and used only to knock
    out the wells that have nothing numeric at all. A well that is half
    unreadable is still the mean of the half that is readable, which is what
    ``generate_plate_heatmap`` computes.
    """
    measured = subset[readable].copy()
    if not len(measured):
        return present * 0
    numbers = heatmap(measured, name, variable, "count", "all", 0)[0]
    numbers = numbers.reindex(index=present.index, columns=present.columns,
                              fill_value=0)
    return present.where(numbers.to_numpy() > 0, 0)


def shared_limits(matrices: Sequence[np.ndarray], min_max="allq"
                  ) -> Tuple[float, float]:
    """One colour scale for every plate, over the wells that exist.

    SHARED IS THE POINT. A plate heatmap is read by comparing plates; four
    independent scales make the same colour mean four different numbers and
    turn a batch effect into an invisible one.

    The ``min_max`` spec is :func:`spacr.plot.generate_plate_heatmap`'s --
    ``'all'``, ``'allq'`` or a two-element range, floats being quantiles --
    so a caller's existing setting keeps its meaning.
    """
    pool = np.concatenate([m.ravel() for m in matrices]) if matrices \
        else np.array([], dtype="float64")
    pool = pool[np.isfinite(pool)]
    if not pool.size:
        return 0.0, 1.0

    if isinstance(min_max, (list, tuple)) and len(min_max) == 2:
        if all(isinstance(value, float) for value in min_max):
            low, high = np.quantile(pool, [min_max[0], min_max[1]])
        else:
            low, high = float(min_max[0]), float(min_max[1])
    elif min_max == "allq":
        low, high = np.quantile(pool, [0.02, 0.98])
    else:
        low, high = float(np.nanmin(pool)), float(np.nanmax(pool))
    low, high = float(low), float(high)
    if low == high:
        high = low + 1e-6
    return low, high


# --------------------------------------------------------------------------- #
#  The layout
# --------------------------------------------------------------------------- #

def small_multiple_layout(count: int, plate_aspect: float,
                          target: float = TARGET_ASPECT) -> Tuple[int, int]:
    """Rows and columns of plates that put the composite nearest square.

    Four plates in a row is 4 x 1.5 = a 6:1 composite; four in a 2 x 2 is
    1.5:1. Both hold the same picture; only one of them fills a tile.

    :param plate_aspect: one plate's width over its height, in wells.
    :param target: the composite width-over-height to aim at.
    :returns: ``(rows, columns)``.
    """
    if count <= 0:
        return 0, 0
    best, choice = None, (1, count)
    for columns in range(1, count + 1):
        rows = -(-count // columns)
        aspect = (columns * plate_aspect) / rows
        # Compared in log space, so "twice too wide" and "twice too tall"
        # cost the same. Ties go to the wider arrangement, which is what a
        # screen is.
        penalty = abs(math.log(aspect / target))
        if best is None or penalty < best - 1e-12:
            best, choice = penalty, (rows, columns)
    return choice


def plate_figure_name(variable: str, prefix: str = "plate_heatmap",
                      suffix: str = ".pdf") -> str:
    """The file a plate panel is written to: named for what it draws.

    NOT :func:`spacr.schema.escape_filename_component`, which escapes the
    key separator -- ``log_pred`` would be written as ``log%5Fpred``. This
    is a file name and not a key: an underscore is exactly what belongs in
    it, and only characters a path cannot hold are replaced.
    """
    text = "".join(character if character.isalnum() or character in "-_."
                   else "_" for character in str(variable).strip())
    return f"{prefix}_{text or 'value'}{suffix}"


def _tick_step(n: int) -> int:
    """Label every well, every other one, or every fourth.

    Twenty-four column numbers under a plate that is two inches wide at
    6.2 pt is a solid line of digits. The step is chosen from the count
    rather than the width because the width is derived from the count.
    """
    if n <= 12:
        return 1
    if n <= 26:
        return 2
    return 4


# --------------------------------------------------------------------------- #
#  The panel
# --------------------------------------------------------------------------- #

def draw_plate(ax, matrix: np.ndarray, *, vmin: float, vmax: float, cmap,
               ink: str, name: str = "", row_labels=None,
               column_labels=None) -> None:
    """One plate into one axes, with square wells and no gridlines.

    :param matrix: ``(n_rows, n_columns)``, ``nan`` where no well was
        measured.
    :param row_labels: ``True`` to draw the row letters, ``False`` to leave
        the axis bare (an inner plate of the small multiple shares the
        outer one's).
    """
    from matplotlib.patches import Rectangle

    n_rows, n_columns = matrix.shape
    # The wash goes down as a real artist rather than as the axes facecolor:
    # savefig(transparent=True) -- which the house style asks for -- forces
    # every axes patch to 'none', so a facecolor wash is on screen and gone
    # in the file.
    ax.add_patch(Rectangle((0, 0), n_columns, n_rows,
                           facecolor=_wash(ink), edgecolor="none", zorder=0))
    ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=vmin, vmax=vmax,
              origin="upper", extent=(0, n_columns, n_rows, 0),
              interpolation="nearest", aspect="equal", zorder=1)

    from .. import schema as _schema

    row_step, column_step = _tick_step(n_rows), _tick_step(n_columns)
    ax.set_yticks([i + 0.5 for i in range(0, n_rows, row_step)])
    ax.set_xticks([i + 0.5 for i in range(0, n_columns, column_step)])
    ax.set_yticklabels(
        [_schema.letters_from_row_index(i + 1)
         for i in range(0, n_rows, row_step)] if row_labels else [])
    ax.set_xticklabels(
        [str(i + 1) for i in range(0, n_columns, column_step)]
        if column_labels else [])
    # Colour and size named here rather than left to the rcParams: the
    # figure is built inside the style context but DRAWN outside it, and a
    # tick that resolves its properties at draw time would resolve them
    # against whatever the process happens to hold then.
    ax.tick_params(length=1.6, width=WEIGHTS["spine"], pad=1.4, colors=ink,
                   labelsize=TYPE_SCALE["tick"])
    for spine in ax.spines.values():
        spine.set_linewidth(WEIGHTS["spine"])
        spine.set_color(ink)
    if name:
        # A descriptor, not a sentence title: the plate's own name.
        ax.set_title(name, fontsize=TYPE_SCALE["annotation"], pad=2.0,
                     color=ink)


def _wash(ink: str) -> tuple:
    """The colour of a well that was never measured."""
    from matplotlib.colors import to_rgba

    return to_rgba(ink, EMPTY_WASH_ALPHA)


def build_plates(frame, variable: str, *, grouping: str = "mean",
                 min_max="allq", min_count=0, cmap=None,
                 target: Optional[str] = None, width: float = WIDTH,
                 plates: Optional[Sequence[str]] = None,
                 limits: Optional[Tuple[float, float]] = None):
    """Every plate of a screen as one figure, on one colour scale.

    :param frame: long-format frame with a ``prc`` column and ``variable``.
    :param variable: the measurement to aggregate per well.
    :param grouping: ``'mean'``, ``'sum'`` or ``'count'``.
    :param min_max: colour-scale spec, as
        :func:`spacr.plot.generate_plate_heatmap` defines it -- but applied
        ONCE, over every plate at the same time.
    :param cmap: a colormap to override the house ramp. ``None`` uses
        :func:`plate_ramp`, which is what the style asks for.
    :param target: ``'screen'`` or ``'print'``; defaults to the user's own
        figure preference.
    :param width: figure width in inches. The HEIGHT is derived from it, so
        that the wells come out square.
    :param plates: draw only these plates, in this order.
    :param limits: an explicit ``(vmin, vmax)``, overriding ``min_max``.
        THIS IS WHAT MAKES ONE-PLATE-PER-FIGURE SAFE: a caller that wants a
        plate to a tile computes :func:`shared_limits` over every plate once
        and passes the same pair to each figure, so splitting the small
        multiple up does not silently give each plate its own scale again.
    :returns: ``(figure, Panel)``. The panel carries the legend sentence,
        generated from what was actually drawn.
    """
    import matplotlib.pyplot as plt

    target = target or theme_target()
    ink = resolve_ink(target)

    with figure_style(target, frame="box"):
        names, matrices, (n_rows, n_columns) = well_matrices(
            frame, variable, grouping=grouping, min_count=min_count,
            plates=plates)
        if not matrices:
            figure = plt.figure(figsize=(width, width / TARGET_ASPECT))
            return figure, Panel(
                "plates", "plate heatmaps", drawn=False,
                reason=(f"no plate in this table carries a well grid for "
                        f"{variable!r}"),
                needs=("prc", variable))

        measured = [int(np.isfinite(m).sum()) for m in matrices]
        vmin, vmax = (float(limits[0]), float(limits[1])) if limits \
            else shared_limits(matrices, min_max)
        rows, columns = small_multiple_layout(
            len(matrices), n_columns / max(n_rows, 1))

        # SIZED FROM THE GRID. The cell is whatever is left after the
        # margins, and the figure is made tall enough for cells of exactly
        # the plate's proportions -- so square wells are what the layout
        # produces, not what it survives.
        cell_w = (width - MARGIN["left"] - MARGIN["right"]
                  - (columns - 1) * MARGIN["wspace"]) / columns
        cell_h = cell_w * n_rows / n_columns
        height = (MARGIN["top"] + MARGIN["bottom"] + rows * cell_h
                  + (rows - 1) * MARGIN["hspace"])

        figure = plt.figure(figsize=(width, height))
        ramp = plate_ramp(target) if cmap is None else _named(cmap)

        image = None
        for index, (name, matrix) in enumerate(zip(names, matrices)):
            row, column = divmod(index, columns)
            left = (MARGIN["left"] + column * (cell_w + MARGIN["wspace"])) / width
            bottom = (MARGIN["bottom"] + (rows - row - 1)
                      * (cell_h + MARGIN["hspace"])) / height
            ax = figure.add_axes([left, bottom, cell_w / width,
                                  cell_h / height])
            draw_plate(ax, matrix, vmin=vmin, vmax=vmax, cmap=ramp, ink=ink,
                       name=str(name), row_labels=column == 0,
                       column_labels=row == rows - 1
                       or index + columns >= len(matrices))
            if image is None:
                image = ax.images[0]

        _colour_bar(figure, image, variable, ink, width, height)
        blank = sum(m.size for m in matrices) - sum(measured)
        # WHAT WAS ACTUALLY DONE TO THE NUMBERS. A legend that says
        # "averaged" under a panel drawn with grouping='count' is a legend
        # that misreports its own figure.
        subject = ("objects per well, counted" if grouping == "count" else
                   f"{variable} per well, "
                   + ("summed" if grouping == "sum" else "averaged")
                   + " over the objects in it")
        plural = "" if len(matrices) == 1 else "s"
        return figure, Panel(
            "plates", "plate heatmaps",
            caption=(
                f"{subject}, for {len(matrices)} plate{plural} of "
                f"{n_rows}x{n_columns} wells. All plates share one colour "
                f"scale ({vmin:.3g} to {vmax:.3g}) so the same colour is the "
                f"same number on every plate. {sum(measured)} wells were "
                f"measured; the {blank} that were not are left as a neutral "
                f"wash and are excluded from the scale."),
            needs=("prc", variable))


def _named(cmap):
    """A colormap from whatever the caller passed."""
    if isinstance(cmap, str):
        from matplotlib import colormaps

        cmap = colormaps[cmap]
    cmap = cmap.copy()
    cmap.set_bad("none")
    return cmap


def _colour_bar(figure, image, variable: str, ink: str, width: float,
                height: float) -> None:
    """One thin horizontal bar under the whole small multiple.

    One, because there is one scale. Horizontal and low, because the
    composite is wider than it is tall and a bar down the right side would
    steal a plate's width.
    """
    bar_w = min(BAR["width"], width * 0.42)
    cax = figure.add_axes([(width - bar_w) / 2 / width, BAR["bottom"] / height,
                           bar_w / width, BAR["height"] / height])
    bar = figure.colorbar(image, cax=cax, orientation="horizontal")
    bar.outline.set_linewidth(WEIGHTS["spine"])
    bar.outline.set_edgecolor(ink)
    cax.tick_params(length=1.6, width=WEIGHTS["spine"], pad=1.2,
                    labelsize=TYPE_SCALE["annotation"], colors=ink)
    bar.set_ticks([image.norm.vmin, image.norm.vmax])
    cax.set_xticklabels([f"{image.norm.vmin:.3g}", f"{image.norm.vmax:.3g}"])
    # The name goes on the same line as the two numbers, centred, and is
    # placed in FIGURE INCHES rather than left to `set_xlabel`: a colour bar
    # 0.075 inches tall gives matplotlib almost nothing to measure a label
    # offset from, and where it lands then depends on whether the figure has
    # been drawn yet. Lower case, spelled out -- the axis-label rule.
    figure.text(0.5, (BAR["bottom"] - 0.022) / height,
                str(variable).replace("_", " ").lower(),
                ha="center", va="top", color=ink,
                fontsize=TYPE_SCALE["annotation"])


__all__ = ["BAR", "EMPTY_WASH_ALPHA", "MARGIN", "TARGET_ASPECT", "WIDTH",
           "build_plates", "draw_plate", "full_plate_grid",
           "plate_figure_name", "plate_names", "plate_ramp", "shared_limits",
           "small_multiple_layout", "well_matrices"]
