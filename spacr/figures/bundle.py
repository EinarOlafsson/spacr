"""Export a graph with its data, statistics, and generating settings.

Each export creates a uniquely named directory containing PDF and PNG
renderings, the plotted data, a statistical summary, and a JSON settings
record. Statistical summaries use :func:`spacr.figures.stats.compare`, the
same comparison engine used by interactive figures.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .style import ROLES

LOG = logging.getLogger("spacr.figures.bundle")

#: The files a bundle always has. ALWAYS, including the ones with nothing to
#: say -- an absent file reads as a bug, and "there was nothing to compare"
#: is a result.
FILES = ("data.csv", "statistics.csv", "settings.json")

#: What `statistics.csv` says when a graph has no groups. A scatter or a
#: one-column histogram has no test to run, which is not the same as a test
#: that failed.
NOTHING_TO_COMPARE = (
    "This graph has no groups to compare, so no test was run. A scatter or "
    "a histogram of one column is a description rather than a comparison; "
    "the data it was drawn from is in data.csv.")


def _unique(folder: str) -> str:
    """A folder that does not exist yet, by suffixing.

    OVERWRITING A FOLDER IS A BIGGER ACT THAN OVERWRITING A FILE. A file
    replaced loses one thing the user can see; a folder replaced can take
    the data and the statistics of an earlier save with it, silently. So a
    second save of the same graph sits beside the first rather than on it.
    """
    if not os.path.exists(folder):
        return folder
    for index in range(2, 1000):
        candidate = f"{folder}-{index}"
        if not os.path.exists(candidate):
            return candidate
    raise FileExistsError(f"{folder} and 998 suffixes of it all exist")


def statistics_rows(comparison) -> list:
    """Convert a statistical comparison into labeled CSV rows.

    Parameters
    ----------
    comparison : spacr.figures.stats.Comparison
        Comparison containing group sizes, assumption checks, test results,
        and effect estimates.

    Returns
    -------
    list of tuple
        ``(item, value, note)`` rows suitable for ``statistics.csv``.
    """
    rows = [("unit", comparison.unit,
             "what ONE observation is; a test across cells when the "
             "replicate is the well is pseudoreplication")]
    for label, count in zip(comparison.groups, comparison.n):
        rows.append((f"n [{label}]", int(count), "usable observations"))
    for assumption in comparison.assumptions:
        state = "holds" if assumption.passed else "does not hold"
        if not assumption.informative:
            state = "could not tell"
        rows.append((assumption.name, state, assumption.verdict))
        rows.append((f"{assumption.name} p", float(assumption.p_value),
                     "the number behind the verdict above"))
    rows.append(("test", comparison.test, comparison.reason))
    rows.append(("statistic", float(comparison.statistic), ""))
    rows.append(("p_value", float(comparison.p_value), ""))
    if np.isfinite(comparison.p_adjusted):
        rows.append(("p_adjusted", float(comparison.p_adjusted),
                     comparison.correction))
    if np.isfinite(comparison.effect_size):
        rows.append(("effect_size", float(comparison.effect_size),
                     comparison.effect_name))
    if comparison.ci is not None:
        rows.append(("effect_ci_low", float(comparison.ci[0]), "95%"))
        rows.append(("effect_ci_high", float(comparison.ci[1]), "95%"))
    rows.append(("sentence", comparison.sentence(),
                 "the legend line, as it would be reported"))
    return rows


def statistics_frame(groups: Optional[Mapping[str, Sequence]] = None, *,
                     unit: str = "observation",
                     paired: bool = False) -> pd.DataFrame:
    """Build the statistical table stored with an exported graph.

    Parameters
    ----------
    groups : mapping of str to sequence, optional
        Values grouped by comparison label. Fewer than two groups produce an
        explanatory row rather than a statistical test.
    unit : str, default="observation"
        Experimental unit represented by one value.
    paired : bool, default=False
        Whether observations are paired across groups.

    Returns
    -------
    pandas.DataFrame
        Columns ``item``, ``value``, and ``note``. Comparison errors are
        recorded in the table instead of being raised.
    """
    columns = ["item", "value", "note"]
    if not groups or len(groups) < 2:
        return pd.DataFrame(
            [("comparison", "none", NOTHING_TO_COMPARE)], columns=columns)
    try:
        from .stats import compare

        comparison = compare(groups, unit=unit, paired=paired)
    except Exception as error:                                   # noqa: BLE001
        return pd.DataFrame(
            [("comparison", "refused",
              f"{type(error).__name__}: {error}")], columns=columns)
    return pd.DataFrame(statistics_rows(comparison), columns=columns)


def save(folder: str, name: str, *,
         render: Callable[[str], Any],
         data: Optional[pd.DataFrame] = None,
         groups: Optional[Mapping[str, Sequence]] = None,
         unit: str = "observation",
         paired: bool = False,
         settings: Optional[Mapping[str, Any]] = None) -> str:
    """Write a reproducible figure-export bundle.

    Parameters
    ----------
    folder : str
        Parent directory for the export.
    name : str
        Graph name used for the directory and image files.
    render : callable
        Function called once with the PDF path and once with the PNG path.
    data : pandas.DataFrame, optional
        Filtered rows represented by the graph.
    groups : mapping of str to sequence, optional
        Values used to generate ``statistics.csv``.
    unit : str, default="observation"
        Experimental unit represented by one value.
    paired : bool, default=False
        Whether group observations are paired.
    settings : mapping, optional
        Settings and filters used to generate the graph.

    Returns
    -------
    str
        Path to the newly created bundle directory. A numeric suffix is added
        when the requested directory already exists.
    """
    safe = "".join(c if c.isalnum() or c in "-_. " else "_"
                   for c in str(name or "graph")).strip() or "graph"
    out = _unique(os.path.join(str(folder), safe))
    os.makedirs(out, exist_ok=True)

    written = []
    for extension in ("pdf", "png"):
        path = os.path.join(out, f"{safe}.{extension}")
        try:
            render(path)
            written.append(path)
        except Exception:                                        # noqa: BLE001
            LOG.debug("could not write %s", path, exc_info=True)

    from ..tabular import write_table

    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame()
    write_table(frame, os.path.join(out, "data.csv"))
    write_table(
        statistics_frame(groups, unit=unit, paired=paired),
        os.path.join(out, "statistics.csv"))

    payload = {str(k): _plain(v) for k, v in dict(settings or {}).items()}
    payload.setdefault("graph", safe)
    payload.setdefault("rows", int(len(frame)))
    with open(os.path.join(out, "settings.json"), "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return out


def _plain(value):
    """Whatever JSON can hold; a string for everything else."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)


#: Every plot kind a figure can be redrawn as: ``(kind, caption)``.
_PLOT_KINDS = (
    ("box", "Box"), ("violin", "Violin"), ("strip", "Strip"),
    ("swarm", "Swarm"), ("bar", "Bar"), ("point", "Point"),
    ("boxen", "Boxen"), ("box_strip", "Box with points"),
    ("bar_strip", "Bar with points"), ("scatter", "Scatter"),
    ("line", "Line"), ("hex", "Hexbin"), ("kde", "Density (KDE)"),
    ("reg", "Regression"), ("hist", "Histogram"), ("ecdf", "ECDF"),
    ("count", "Count"), ("heatmap", "Heatmap"),
    ("clustermap", "Clustered heatmap"),
)

#: Which kinds fit which data structure.
_FAMILY_KINDS = {
    "groups": ("box", "violin", "strip", "swarm", "bar", "point", "boxen",
               "box_strip", "bar_strip", "hist", "kde", "ecdf"),
    "numeric": ("scatter", "line", "hex", "kde", "reg"),
    "distribution": ("hist", "kde", "ecdf", "box", "violin", "strip",
                     "boxen"),
    "counts": ("count", "heatmap"),
    "matrix": ("heatmap", "clustermap"),
}

#: The spaCR house graph types, in the vocabulary :func:`_draw` speaks.
_HOUSE_KINDS = {
    "jitter_bar": "bar_strip", "bar_jitter": "bar_strip",
    "jitter_box": "box_strip", "box_jitter": "box_strip",
    "jitter": "strip", "line": "point", "bar": "bar", "box": "box",
    "violin": "violin", "histogram": "hist", "scatter": "scatter",
    "heatmap": "heatmap",
}


def _plot_family(frame, spec) -> str:
    """The data structure a figure's frame and spec describe.

    :returns: a key of :data:`_FAMILY_KINDS`, or ``""`` without data.
    """
    if frame is None or not len(getattr(frame, "columns", ())):
        return ""
    if spec.get("matrix"):
        return "matrix"
    x, y = str(spec.get("x") or ""), str(spec.get("y") or "")
    columns = frame.columns

    def numeric(name):
        """Whether a present column is numeric."""
        return (name in columns and pd.api.types.is_numeric_dtype(frame[name])
                and not pd.api.types.is_bool_dtype(frame[name]))

    if x in columns and y in columns:
        if numeric(x) and numeric(y):
            return "numeric"
        if numeric(y):
            return "groups"
        if not numeric(x):
            return "counts"
        return "groups"
    if numeric(y) or numeric(x):
        return "distribution"
    return ""


def _kinds_for(frame, spec) -> tuple:
    """``(kind, caption)`` pairs the figure's data can be drawn as."""
    allowed = _FAMILY_KINDS.get(_plot_family(frame, spec), ())
    return tuple((kind, caption) for kind, caption in _PLOT_KINDS
                 if kind in allowed)


#: Kinds that mark a figure as a picture rather than a plot of a table.
_IMAGE_KINDS = ("image", "mask", "montage", "overlay")


def _as_figure(target):
    """The Matplotlib figure behind ``target``: a figure, an axes or a grid."""
    if target is None or hasattr(target, "savefig"):
        return target
    for name in ("get_figure", "figure", "fig"):
        found = getattr(target, name, None)
        found = found() if callable(found) and name == "get_figure" else found
        if found is not None and hasattr(found, "savefig"):
            return found
    flat = getattr(target, "flat", None)
    if flat is not None:
        for axis in flat:
            return _as_figure(axis)
    return target


def _as_frame(data, x: str, y: str):
    """A tidy frame from a frame, a series, a mapping of groups or a vector."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame(name=str(data.name or y or "value"))
    if isinstance(data, Mapping):
        columns = {str(k): np.asarray(v).ravel() for k, v in data.items()}
        sizes = {len(v) for v in columns.values()}
        if len(sizes) <= 1:
            return pd.DataFrame(columns)
        return pd.DataFrame({
            x or "group": np.concatenate(
                [[k] * len(v) for k, v in columns.items()]),
            y or "value": np.concatenate(list(columns.values()))})
    values = np.asarray(data)
    if values.ndim == 2:
        return pd.DataFrame(values)
    return pd.DataFrame({y or "value": values.ravel()})


def _register_figure_data(figure, data, *, x: str = "", y: str = "",
                          hue: str = "", kind: str = "", **spec) -> None:
    """Attach what a figure was drawn from: its tidy data and plot spec.

    The one call every figure-producing function makes. A plot of a table
    can then be redrawn as another kind, tested, and saved with its data,
    statistics and a script that re-creates it. A picture (``kind`` of
    ``"image"``, ``"mask"``, ``"montage"`` or ``"overlay"``, or arrays of
    two or more dimensions) keeps its arrays and metadata instead; its menu
    offers editing and a zip of the image with its metadata, never a graph
    type or a test. Errors never reach the caller.

    :param figure: the Matplotlib figure, or one of its axes.
    :param data: the tidy frame (one row per observation), a series, a
        mapping of group to values, or for a picture the array or arrays
        shown (``None`` when only the rendered image is kept). A callable
        returning any of these is called here, so building the frame can
        never break the figure.
    :param x: the column on the horizontal axis, or the grouping column.
    :param y: the column on the vertical axis, or the measurement.
    :param hue: an optional colour-grouping column.
    :param kind: the kind drawn, from :data:`_PLOT_KINDS`, a spaCR graph
        type, or a picture kind.
    :param spec: further keys, such as ``order``, ``pair`` (the subject
        column of repeated measures), ``matrix`` or ``title``. Multi-panel
        plots provide ``panels`` (one column recipe per populated axes) and
        ``grid`` (rows, columns); panel recipes inherit shared options.
    """
    try:
        figure = _as_figure(figure)
        if callable(data):
            data = data()
        record = {k: v for k, v in dict(spec).items() if v is not None}
        kind = str(kind or "")
        arrays = None
        if kind in _IMAGE_KINDS:
            arrays = data
        elif (not isinstance(data, (pd.DataFrame, pd.Series, Mapping))
              and data is not None and not record.get("matrix")):
            if isinstance(data, (list, tuple)) and data and all(
                    np.ndim(item) >= 2 for item in data):
                arrays = data
            elif np.ndim(data) >= 3 or (np.ndim(data) == 2
                                        and kind not in ("heatmap",
                                                         "clustermap")):
                arrays = data
        if arrays is not None or (data is None and kind in _IMAGE_KINDS):
            if arrays is None:
                arrays = []
            elif not isinstance(arrays, (list, tuple)):
                arrays = [arrays]
            record.update(kind=kind or "image")
            figure._spacr_image = [np.asarray(a) for a in arrays]
            figure._spacr_data = None
            figure._spacr_drawn_data = None
            figure._spacr_spec = record
            return
        if data is None:
            return
        frame = _as_frame(data, str(x or ""), str(y or ""))
        if kind in ("heatmap", "clustermap") and not (x or y):
            record.setdefault("matrix", True)
        record.update(x=str(x or ""), y=str(y or ""), hue=str(hue or ""),
                      kind=_HOUSE_KINDS.get(kind, kind))
        figure._spacr_image = None
        figure._spacr_data = frame
        figure._spacr_drawn_data = None
        figure._spacr_spec = record
    except Exception:
        LOG.debug("could not attach data to the figure", exc_info=True)


def _is_image_figure(figure) -> bool:
    """Whether ``figure`` shows a picture rather than a plot of a table."""
    return getattr(figure, "_spacr_image", None) is not None


def _figure_record(figure):
    """``(frame, spec)`` for a figure, or ``(None, {})`` when it has none.

    Registered data comes first; a figure drawn by
    :func:`spacr.plot.create_grouped_plot` carries a redraw recipe that is
    read as the same thing.
    """
    frame = getattr(figure, "_spacr_data", None)
    spec = getattr(figure, "_spacr_spec", None)
    if isinstance(frame, pd.DataFrame) and isinstance(spec, dict):
        return frame, spec
    recipe = getattr(figure, "_spacr_replot", None)
    if isinstance(recipe, dict) and isinstance(recipe.get("df"),
                                               pd.DataFrame):
        spec = {"x": str(recipe.get("grouping_column") or ""),
                "y": str(recipe.get("data_column") or ""), "hue": "",
                "kind": _HOUSE_KINDS.get(str(recipe.get("graph_type") or ""),
                                         "box_strip")}
        order = recipe.get("order")
        if order:
            spec["order"] = [str(v) for v in order]
        return recipe["df"], spec
    return None, {}


def _capture_view(figure, spec) -> dict:
    """The spec with the figure's current titles, labels, scales and size.

    So an edit made on screen is part of the recipe a saved figure is
    re-created from.
    """
    out = dict(spec)
    axes = [a for a in getattr(figure, "axes", ()) if a.get_label()
            != "<colorbar>"]
    if axes:
        ax = axes[0]
        out.update(title=ax.get_title(), xlabel=ax.get_xlabel(),
                   ylabel=ax.get_ylabel(), xscale=ax.get_xscale(),
                   yscale=ax.get_yscale())
        if out.get("keep_limits"):
            limits = tuple(ax.get_ylim())
            previous = getattr(ax, "_spacr_stats_ylim", None)
            if previous is not None and limits == previous[1]:
                limits = previous[0]
            out.update(xlim=list(ax.get_xlim()), ylim=list(limits))
    if spec.get("panels"):
        out["panels"] = []
        for index, panel in enumerate(spec["panels"]):
            ax = axes[panel.get("slot", index)]
            record = dict(panel)
            record.update(title=ax.get_title(), xlabel=ax.get_xlabel(),
                          ylabel=ax.get_ylabel(), xscale=ax.get_xscale(),
                          yscale=ax.get_yscale())
            if out.get("keep_limits"):
                limits = tuple(ax.get_ylim())
                previous = getattr(ax, "_spacr_stats_ylim", None)
                if previous is not None and limits == previous[1]:
                    limits = previous[0]
                record.update(xlim=list(ax.get_xlim()), ylim=list(limits))
            out["panels"].append(record)
    try:
        out["size"] = [float(v) for v in figure.get_size_inches()]
        out["dpi"] = float(figure.get_dpi())
    except Exception:
        pass
    return out


def _annotate(ax, spec) -> None:
    """Draw the statistics a spec carries onto ``ax``.

    Brackets with stars join each significant pair of categories, and the
    test line sits in the top-left corner. Earlier annotations are removed
    first, so applying twice draws once.
    """
    previous = getattr(ax, "_spacr_stats_ylim", None)
    if previous is not None and tuple(ax.get_ylim()) == previous[1]:
        ax.set_ylim(*previous[0])
    ax._spacr_stats_ylim = None
    for artist in list(ax.texts) + list(ax.lines):
        if artist.get_gid() == "spacr-stats":
            artist.remove()
    note = str(spec.get("stats_note") or "")
    if note:
        ax.text(0.01, 0.99, note, transform=ax.transAxes, va="top",
                ha="left", fontsize=7, gid="spacr-stats")
    marks = list(spec.get("annotations") or [])
    if not marks:
        return
    positions = {}
    for location, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        text = label.get_text().strip()
        if text:
            positions[text] = float(location)
    low, high = ax.get_ylim()
    step = (high - low) * 0.06
    level = high
    for mark in marks:
        left, right = (str(v) for v in mark.get("pair", ("", "")))
        if left not in positions or right not in positions:
            continue
        a, b = positions[left], positions[right]
        ax.plot([a, a, b, b], [level, level + step / 2, level + step / 2,
                               level], color=ROLES["reference"], lw=0.8,
                gid="spacr-stats", clip_on=False)
        ax.text((a + b) / 2, level + step / 2, str(mark.get("label", "")),
                ha="center", va="bottom", fontsize=8, gid="spacr-stats")
        level += step * 1.4
    if level != high:
        ax.set_ylim(low, level + step)
        ax._spacr_stats_ylim = ((low, high), tuple(ax.get_ylim()))


def _recipe_frame(frame, spec):
    """The rows and optional wide-to-long values displayed by one panel.

    ``where`` selects recorded column values. ``melt`` selects measurement
    columns and names their category/value fields; its optional labels map
    original measurement names to their displayed names. The registered
    source frame is never changed or narrowed by this rendering transform.
    Selected non-measurement hue, pairing and count metadata stays available
    in melted rows.
    """
    data = frame.copy()
    for column, values in (spec.get("where") or {}).items():
        data = data.loc[data[column].isin(values)]
    plate = spec.get("plate")
    if plate:
        coordinates = plate["coordinates"]
        data = data.loc[data["prc"].astype(str).isin(coordinates)].copy()
        if str(spec.get("kind") or "") == "plate_heatmap":
            import numpy as np

            positions = [coordinates[key] for key in data["prc"].astype(str)]
            data["__row__"] = [position[0] for position in positions]
            data["__column__"] = [position[1] for position in positions]
            grouped = data.groupby(["__row__", "__column__"])
            counts = grouped.size()
            method = plate["grouping"]
            if method == "count":
                values = counts.astype(float)
            elif method in ("mean", "sum"):
                data["__value__"] = pd.to_numeric(data[plate["variable"]], errors="coerce")
                measured = data.groupby(["__row__", "__column__"])["__value__"]
                values = getattr(measured, method)().where(measured.count() > 0)
            else:
                raise ValueError("Invalid plate aggregation")
            values = values.where(counts >= max(1, plate.get("min_count", 0)))
            rows, columns = plate["shape"]
            matrix = pd.DataFrame(np.nan, index=range(1, rows + 1),
                                  columns=range(1, columns + 1))
            for (row, column), value in values.items():
                matrix.loc[row, column] = value
            return matrix
    melted = spec.get("melt")
    if melted:
        category = melted["var_name"]
        identifiers = list(melted.get("id_vars", []))
        selected = spec.get("stats") or {}
        for column in (spec.get("hue"), spec.get("pair"), selected.get("pair"),
                       spec.get("count")):
            if (column and column in data.columns and column not in identifiers
                    and column not in melted["columns"]):
                identifiers.append(column)
        data = data.melt(
            id_vars=identifiers, value_vars=melted["columns"],
            var_name=category, value_name=melted["value_name"])
        if melted.get("labels"):
            data[category] = data[category].map(melted["labels"]).fillna(data[category])
    return data


def _draw(figure, frame, spec, *, axes=None):
    """Draw ``frame`` onto ``figure`` as ``spec`` describes, and return the axes.

    The figure is cleared and redrawn with seaborn and Matplotlib only, so
    the same function, copied into a saved figure's script, re-creates it
    without spaCR installed.

    :param axes: an existing panel axes for recursive multi-panel rendering;
        ``None`` clears the figure and creates the recorded layout.

    A panel's ``where`` mapping restricts each named column to its recorded
    values before plotting. The source data remains complete in the export.
    ``melt`` describes wide-to-long rendering; ``slot`` places a panel in the
    recorded grid when blank cells separate condition rows.
    """
    import numpy as np
    import pandas as pd
    import seaborn as sns

    np.random.seed(int(spec.get("seed", 0)))
    kind = str(spec.get("kind") or "box")
    x = spec.get("x") or None
    y = spec.get("y") or None
    hue = spec.get("hue") or None
    if axes is None:
        figure._spacr_drawn_data = None
        panels = spec.get("panels")
        if panels:
            rows, columns = spec["grid"]
            if (not isinstance(rows, int) or not isinstance(columns, int)
                    or rows < 1 or columns < 1
                    or rows * columns < len(panels)
                    or rows * columns > 2 * len(panels)):
                raise ValueError("Invalid multi-panel figure layout")
            slots = [panel.get("slot", index) for index, panel in enumerate(panels)]
            if (any(not isinstance(slot, int) or not 0 <= slot < rows * columns
                    for slot in slots) or len(set(slots)) != len(slots)):
                raise ValueError("Invalid multi-panel figure layout")
        figure.clear()
        if spec.get("size"):
            figure.set_size_inches(*spec["size"])
        if panels:
            targets = np.asarray(
                figure.subplots(rows, columns, squeeze=False)).ravel()
            for slot, panel in zip(slots, panels):
                target = targets[slot]
                merged = dict(spec, **panel)
                merged.pop("panels", None)
                merged.pop("grid", None)
                _draw(figure, frame, merged, axes=target)
                if panel.get("rect"):
                    target.set_position(panel["rect"])
            bar = spec.get("plate_colorbar")
            if kind == "plate_heatmap" and bar:
                cax = figure.add_axes(bar["rect"], label="<colorbar>")
                image = next(target.images[0] for target in targets if target.images)
                colorbar = figure.colorbar(image, cax=cax, orientation="horizontal")
                colorbar.set_ticks(bar["ticks"])
                cax.set_xticklabels([f"{value:.3g}" for value in bar["ticks"]])
                colorbar.outline.set_linewidth(bar["linewidth"])
                colorbar.outline.set_edgecolor(bar["ink"])
                cax.tick_params(length=1.6, width=bar["linewidth"], pad=1.2,
                                labelsize=bar["fontsize"], colors=bar["ink"])
                figure.text(*bar["text_position"], bar["text"], ha="center", va="top",
                            color=bar["ink"], fontsize=bar["fontsize"])
            for index, target in enumerate(targets):
                if index not in slots:
                    target.set_visible(False)
            return targets
        ax = figure.add_subplot(111)
    else:
        ax = axes
    data = _recipe_frame(frame, spec)
    groups_kinds = ("box", "violin", "strip", "swarm", "bar", "point",
                    "boxen", "box_strip", "bar_strip", "count")
    order = None
    if kind in groups_kinds and x and x in data.columns:
        horizontal = (y in data.columns
                      and pd.api.types.is_numeric_dtype(data[x])
                      and not pd.api.types.is_bool_dtype(data[x])
                      and (not pd.api.types.is_numeric_dtype(data[y])
                           or pd.api.types.is_bool_dtype(data[y])))
        category = y if horizontal else x
        data[category] = data[category].astype(str)
        order = [str(v) for v in (spec.get("order")
                                  or pd.unique(data[category]))]
    common = {"data": data, "x": x, "y": y, "ax": ax}
    if hue and hue in data.columns:
        common["hue"] = hue
    points = {"color": ROLES["reference"], "size": 3, "alpha": 0.7}
    if kind == "box":
        sns.boxplot(order=order, **common)
    elif kind == "violin":
        sns.violinplot(order=order, **common)
    elif kind == "strip":
        sns.stripplot(order=order, **common)
    elif kind == "swarm":
        sns.swarmplot(order=order, size=3, **common)
    elif kind == "bar":
        sns.barplot(order=order, errorbar="sd", **common)
    elif kind == "point":
        sns.pointplot(order=order, errorbar="sd", **common)
    elif kind == "boxen":
        sns.boxenplot(order=order, **common)
    elif kind == "box_strip":
        sns.boxplot(order=order, showfliers=False, **common)
        sns.stripplot(order=order, **dict(common, hue=None), **points)
    elif kind == "bar_strip":
        sns.barplot(order=order, errorbar="sd", alpha=0.6, **common)
        sns.stripplot(order=order, **dict(common, hue=None), **points)
    elif kind == "count":
        sns.countplot(data=data, x=x, hue=y, order=order, ax=ax)
    elif kind == "scatter":
        if spec.get("scatter"):
            ax.scatter(data[x], data[y], **spec["scatter"])
        else:
            sns.scatterplot(**common)
    elif kind == "line":
        sns.lineplot(**common)
    elif kind == "hex":
        ax.hexbin(data[x], data[y], gridsize=int(spec.get("gridsize", 30)),
                  cmap=spec.get("cmap") or "viridis", mincnt=1)
    elif kind == "reg":
        sns.regplot(data=data, x=x, y=y, ax=ax)
    elif kind == "hist" and spec.get("histogram"):
        ax.hist(data[y or x], **spec["histogram"])
    elif kind == "qq":
        import statsmodels.api as sm

        recipe = spec["qq"]
        sm.qqplot(data[y].to_numpy(), fit=recipe["fit"], line=recipe["line"], ax=ax)
        for line in ax.lines:
            if line.get_linestyle() == "None":
                line.set_color(recipe["data_color"])
                line.set_markerfacecolor(recipe["data_color"])
                line.set_markeredgecolor("none")
            else:
                line.set_color(recipe["reference_color"])
                line.set_linewidth(recipe["reference_width"])
                line.set_linestyle((0, (4, 3)))
    elif kind == "lorenz":
        helpers = globals()
        if "lorenz_curve" not in helpers:
            helpers = {}
            exec(_lorenz_script(), helpers)
        recipe = spec["lorenz"]
        combined = []
        entries = []
        for curve in recipe["curves"]:
            selected = frame.loc[
                (frame[recipe["input_column"]] == curve["input"])
                & frame[recipe["row_column"]].isin(curve["rows"]), y].to_numpy()
            combined.extend(selected)
            shares = helpers["lorenz_curve"](selected)
            gini = helpers["gini_coefficient"](selected)
            label = f"{curve['label']} (Gini: {gini:.4f})"
            ax.plot(np.linspace(0, 1, len(shares)), shares, label=label,
                    color=curve["color"], linestyle=curve["linestyle"])
            entries.append((label, curve["color"]))
        shares = helpers["lorenz_curve"](np.asarray(combined))
        gini = helpers["gini_coefficient"](np.asarray(combined))
        label = f"Combined (Gini: {gini:.4f})"
        ax.plot(np.linspace(0, 1, len(shares)), shares, label=label,
                color=recipe["combined_color"], linestyle="--")
        entries.append((label, recipe["combined_color"]))
        helpers["text_legend"](ax, entries)
    elif kind == "venn":
        from matplotlib_venn import venn2

        recipe = spec["venn"]
        sets = []
        for part in recipe["inputs"]:
            selected = data.loc[data[recipe["input_column"]] == part["input"]]
            threshold = recipe["filter_coeff"]
            if threshold is not None:
                selected = selected.loc[
                    selected["coefficient"] > threshold if threshold >= 0
                    else selected["coefficient"] < threshold]
            sets.append(set(selected[recipe["gene_column"]].dropna()))
        diagram = venn2(sets, recipe["labels"], ax=ax)
        for region, color in recipe["colors"].items():
            patch = diagram.get_patch_by_id(region)
            if patch is not None:
                patch.set_color(color)
                patch.set_alpha(1.0)
                patch.set_edgecolor("none")
        for label in list(diagram.set_labels or []) + list(diagram.subset_labels or []):
            if label is not None:
                label.set_fontsize(recipe["fontsize"])
                label.set_color(recipe["ink"])
    elif kind == "regression_panel":
        renderers = globals().get("_REGRESSION_PANELS")
        letter = globals().get("panel_letter")
        if renderers is None:
            from .panels import REGISTRY as renderers
            from .style import panel_letter as letter

        globals()["_REGRESSION_LOCALISATIONS"] = dict(spec.get("localisations") or {})
        panel = renderers[spec["regression"]](
            ax, frame, **dict(spec.get("regression_options") or {}))
        if axes is None:
            figure._spacr_drawn_data = panel.data
        if spec.get("letter"):
            letter(ax, spec["letter"])
    elif kind == "plate_heatmap":
        from matplotlib.colors import ListedColormap, to_rgba
        from matplotlib.patches import Rectangle

        plate = spec["plate"]
        rows, columns = data.shape
        ink = plate["ink"]
        cmap = ListedColormap(plate["colors"])
        cmap.set_bad(plate["bad_color"])
        cmap.set_under(plate["under_color"])
        cmap.set_over(plate["over_color"])
        ax.add_patch(Rectangle((0, 0), columns, rows,
                               facecolor=to_rgba(ink, plate["wash_alpha"]),
                               edgecolor="none", zorder=0))
        ax.imshow(np.ma.masked_invalid(data.to_numpy(dtype=float)), cmap=cmap,
                  vmin=plate["limits"][0], vmax=plate["limits"][1], origin="upper",
                  extent=(0, columns, rows, 0), interpolation="nearest",
                  aspect="equal", zorder=1)
        ax.set_xticks(plate["xticks"])
        ax.set_yticks(plate["yticks"])
        ax.set_xticklabels(plate["xticklabels"])
        ax.set_yticklabels(plate["yticklabels"])
        ax.tick_params(length=1.6, width=plate["linewidth"], pad=1.4,
                       colors=ink, labelsize=plate["tick_fontsize"])
        for spine in ax.spines.values():
            spine.set_linewidth(plate["linewidth"])
            spine.set_color(ink)
        ax.set_title(plate["name"], fontsize=plate["title_fontsize"], pad=2.0, color=ink)
        if plate.get("outline"):
            selected = dict(plate, variable=plate["outline"], grouping="mean")
            marks = _recipe_frame(frame, dict(spec, plate=selected))
            for row, column in zip(*np.nonzero(np.nan_to_num(marks.to_numpy()) > 0)):
                ax.add_patch(Rectangle((column + 0.1, row + 0.1), 0.8, 0.8,
                                       fill=False, edgecolor=ink,
                                       linewidth=plate["outline_linewidth"], zorder=3))
    elif kind in ("heatmap", "clustermap"):
        if spec.get("matrix"):
            matrix = data.set_index(spec["index"]) if spec.get("index") \
                in data.columns else data
            matrix = matrix.select_dtypes("number")
        else:
            matrix = pd.crosstab(data[x].astype(str), data[y].astype(str))
        if kind == "clustermap" and min(matrix.shape) > 1:
            from scipy.cluster.hierarchy import leaves_list, linkage

            filled = matrix.fillna(0).to_numpy(dtype=float)
            rows = leaves_list(linkage(filled, "average"))
            cols = leaves_list(linkage(filled.T, "average"))
            matrix = matrix.iloc[rows, cols]
        sns.heatmap(matrix, ax=ax, cmap=spec.get("cmap") or "viridis")
    else:
        value, group = y, x
        if x and y and pd.api.types.is_numeric_dtype(data[x]) and \
                pd.api.types.is_numeric_dtype(data[y]) and kind == "kde":
            sns.kdeplot(data=data, x=x, y=y, ax=ax, fill=True)
        else:
            if not value or value not in data.columns:
                value, group = x, None
            if group and group in data.columns:
                data[group] = data[group].astype(str)
            render_args = {"data": data, "x": value, "ax": ax}
            if group and group in data.columns:
                render_args["hue"] = group
            {"hist": sns.histplot, "kde": sns.kdeplot,
             "ecdf": sns.ecdfplot}.get(kind, sns.histplot)(**render_args)
    for reference in spec.get("references", []):
        options = {key: value for key, value in reference.items()
                   if key not in ("axis", "value", "dashes")}
        if reference.get("dashes"):
            options["linestyle"] = (0, tuple(reference["dashes"]))
        {"x": ax.axvline, "y": ax.axhline}[reference["axis"]](reference["value"], **options)
    for key, setter in (("title", ax.set_title), ("xlabel", ax.set_xlabel),
                        ("ylabel", ax.set_ylabel),
                        ("xscale", ax.set_xscale),
                        ("yscale", ax.set_yscale)):
        if spec.get(key):
            if key.endswith("scale") and getattr(ax, f"get_{key}")() == spec[key]:
                continue
            setter(spec[key])
    if spec.get("xlim"):
        ax.set_xlim(*spec["xlim"])
    if spec.get("ylim"):
        ax.set_ylim(*spec["ylim"])
    if spec.get("rect"):
        ax.set_position(spec["rect"])
    _annotate(ax, spec)
    return ax


_SCRIPT = '''"""Re-create the figure in this folder from data.csv and spec.json.

Needs pandas, numpy, scipy, matplotlib and seaborn (also statsmodels for QQ
plots and matplotlib-venn for Venn diagrams). Run it beside the two
files; it writes recreated.png (and any format named on the command line,
for example ``python recreate_figure.py pdf svg``).
"""
import json
import os
import sys

import pandas as pd
from matplotlib.figure import Figure

ROLES = {{"reference": {reference}}}


{annotate}

{recipe_frame}

{regression_helpers}

{draw}

def main():
    """Read the data and the recipe, draw, and save."""
    here = os.path.dirname(os.path.abspath(__file__))
    frame = pd.read_csv(os.path.join(here, "data.csv"))
    with open(os.path.join(here, "spec.json"), encoding="utf-8") as handle:
        spec = json.load(handle)
    figure = Figure()
    _draw(figure, frame, spec)
    for fmt in (sys.argv[1:] or ["png"]):
        figure.savefig(os.path.join(here, "recreated." + fmt),
                       dpi=spec.get("dpi", 100))


if __name__ == "__main__":
    main()
'''


def _regression_script() -> str:
    """Copy the normal sheet renderers and their numerical helpers into a script.

    Relative imports for hit labels and baselines bind to the same copied
    functions. The exported sheet needs only the scientific Python libraries
    already required by the normal standalone figure script.
    """
    import ast
    import inspect
    import textwrap

    from .. import baseline, hits, localisation
    from . import panels, style

    class LocalImports(ast.NodeTransformer):
        """Bind copied hit-label and baseline functions without package imports."""

        def visit_ImportFrom(self, node):
            """Replace relative helper imports with their copied local bindings.

            :param node: import-from syntax node from an existing renderer.
            :returns: replacement assignments, or the unchanged import node.
            """
            if node.level and node.module in ("hits", "baseline", "localisation"):
                return [ast.Assign(targets=[ast.Name(id=name.asname, ctx=ast.Store())],
                                   value=ast.Subscript(value=ast.Call(func=ast.Name(id="globals", ctx=ast.Load()), args=[], keywords=[]),
                                                       slice=ast.Constant(value=name.name), ctx=ast.Load()))
                        for name in node.names if name.asname and name.asname != name.name]
            return node

    pieces = ["import math\nimport re\nimport numpy as np\n"
              "from dataclasses import dataclass, field\n"
              "from typing import Any, Callable, Dict, Iterable, Optional, Sequence\n"]
    constants = dict(ROLES=style.ROLES, TYPE_SCALE=style.TYPE_SCALE, WEIGHTS=style.WEIGHTS,
        CONTROL_CONDITIONS=panels.CONTROL_CONDITIONS, MIN_CONTROLS=panels.MIN_CONTROLS,
        ZERO=baseline.ZERO, CONTROLS=baseline.CONTROLS, NAMED=baseline.NAMED,
        VALUE=baseline.VALUE, CONTROL_LABELS=baseline.CONTROL_LABELS,
        _GENE_ID_PREFIXES=hits._GENE_ID_PREFIXES)
    pieces.extend(f"{name} = {value!r}" for name, value in constants.items())
    for name in ("_BRACKET", "NUISANCE_TERMS"):
        pattern = getattr(hits, name)
        pieces.append(f"{name} = re.compile({pattern.pattern!r}, {int(pattern.flags)})")
    objects = [style.Palette, panels.Panel, baseline.Baseline,
        baseline._control_rows, baseline.describe_intercept, baseline.resolve, baseline.apply,
        hits.tested_family, hits._gene_id_of, hits.gene_of,
        localisation.of, localisation.mask,
        style.panel_letter, style.reference_line, style.annotate, style.text_legend,
        panels._column, panels.effect_column, panels.p_column, panels.q_column,
        panels.tested, panels._finite, panels.label_series, panels.control_threshold,
        *panels.REGISTRY.values()]
    for obj in objects:
        tree = LocalImports().visit(ast.parse(textwrap.dedent(inspect.getsource(obj))))
        pieces.append(ast.unparse(ast.fix_missing_locations(tree)))
    pieces.append("def table():\n    return _REGRESSION_LOCALISATIONS")
    pieces.append("_REGRESSION_PANELS = {" + ", ".join(
        f"{key!r}: {value.__name__}" for key, value in panels.REGISTRY.items()) + "}")
    return "\n\n".join(pieces)


def _lorenz_script() -> str:
    """Copy the original Lorenz and Gini calculations into a standalone script.

    The numerical functions come from the normal producer's nested helpers;
    the saved curves use their complete input rows and recorded selections.

    :returns: Python source for the calculations and the normal text legend.
    """
    import ast
    import inspect
    import textwrap

    from ..plot import plot_lorenz_curves
    from .style import TYPE_SCALE, text_legend

    tree = ast.parse(textwrap.dedent(inspect.getsource(plot_lorenz_curves)))
    calculations = [node for node in tree.body[0].body
                    if isinstance(node, ast.FunctionDef)
                    and node.name in ("lorenz_curve", "gini_coefficient")]
    return "\n\n".join([
        "import numpy as np\nfrom typing import Sequence",
        f"TYPE_SCALE = {TYPE_SCALE!r}",
        *(ast.unparse(node) for node in calculations),
        textwrap.dedent(inspect.getsource(text_legend))])


def _recreate_script(spec=None) -> str:
    """The Python script saved beside a figure that re-creates it."""
    import inspect
    import textwrap

    return _SCRIPT.format(
        reference=repr(ROLES["reference"]),
        annotate=textwrap.dedent(inspect.getsource(_annotate)),
        recipe_frame=textwrap.dedent(inspect.getsource(_recipe_frame)),
        regression_helpers=(_regression_script() if (spec or {}).get("kind") == "regression_panel"
                            else _lorenz_script() if (spec or {}).get("kind") == "lorenz" else ""),
        draw=textwrap.dedent(inspect.getsource(_draw)))


def _default_formats() -> list:
    """The image formats a saved figure is written in.

    The Preferences figure format, and always a PNG beside it.
    """
    formats = []
    try:
        from ..plot import figure_output_preferences

        formats.append(str(figure_output_preferences()[0] or "pdf").lower())
    except Exception:
        formats.append("pdf")
    if "png" not in formats:
        formats.append("png")
    return formats


def _panel_statistics(frame, spec, *, choices=None):
    """Tests and a readable report for each recorded panel's plotted rows.

    A dialog may supply shared choices; otherwise each panel inherits or
    overrides the saved choices. Pair identifiers are retained by the same
    transform used for rendering before each test runs.
    """
    from .stats import _auto_statistics, _statistics_text

    tables = []
    summaries = []
    for index, panel in enumerate(spec["panels"]):
        options = dict(spec, **panel)
        if choices is not None:
            options["stats"] = dict(choices)
        selected = dict(options.get("stats") or {})
        measurement = str(options.get("measurement")
                          or options.get("y") or options.get("x") or "")
        panel_data = _recipe_frame(frame, options)
        part = _auto_statistics(
            panel_data, str(options.get("x") or ""),
            str(options.get("y") or ""),
            test=selected.get("test") or None,
            paired=selected.get("paired"),
            pair=str(selected.get("pair") or options.get("pair") or ""),
            correction=str(selected.get("correction") or "fdr_bh"),
            order=options.get("order"), count=str(options.get("count") or ""))
        summaries.append(measurement + "\n" + _statistics_text(part))
        part.insert(0, "measurement", measurement)
        part.insert(0, "panel", index)
        part["correction_scope"] = "within measurement"
        tables.append(part)
    return pd.concat(tables, ignore_index=True), "\n\n".join(summaries)


def _save_zip(figure, path: str, *, formats=None, name: str = "") -> str:
    """Write ONE zip holding a figure, its data, statistics and recipe.

    Inside: the image in every default format, ``data.csv`` (the source rows
    needed to recreate the figure), optional ``drawn_data.csv`` (the plotted
    rows of a single regression panel),
    ``statistics.csv`` (one table: normality, equal variance, omnibus and
    pairwise tests, each with whether it was chosen automatically or by the
    user) and ``statistics.txt``, ``spec.json`` (the full plotting recipe)
    and ``recreate_figure.py``, which draws the figure again from the CSV
    and the JSON.

    Multi-panel recipes identify the panel and measurement in each statistics
    row. Automatic choices and user overrides run per measurement; pairwise
    multiple-testing correction is scoped within that measurement.

    :param figure: the Matplotlib figure.
    :param path: the zip to write; ``.zip`` is added when missing.
    :param formats: image formats, or ``None`` for :func:`_default_formats`.
    :param name: base name of the image files.
    :returns: the path written.
    """
    import tempfile
    import zipfile

    from ..plot import save_figure
    from ..tabular import write_table
    from .stats import _auto_statistics, _statistics_text

    path = str(path)
    if not path.lower().endswith(".zip"):
        path += ".zip"
    if _is_image_figure(figure):
        return _save_image_zip(figure, path, formats=formats, name=name)
    frame, spec = _figure_record(figure)
    spec = _capture_view(figure, spec)
    base = "".join(c if c.isalnum() or c in "-_." else "_"
                   for c in str(name or spec.get("title") or "figure"))
    base = base.strip("._") or "figure"
    stats_spec = dict(spec.get("stats") or {})
    tested, sx, sy = frame, str(spec.get("x") or ""), str(spec.get("y") or "")
    groups = getattr(figure, "_spacr_groups", None)
    if not (sx or sy) and isinstance(groups, Mapping) and groups:
        sx, sy = "group", "value"
        tested = pd.DataFrame(
            [(str(label), value) for label, values in groups.items()
             for value in np.asarray(values).ravel()], columns=(sx, sy))
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            frame = tested
            spec.update(x=sx, y=sy)
            if not spec.get("kind"):
                spec["kind"] = "box"
    with tempfile.TemporaryDirectory(prefix="spacr_fig_") as folder:
        for fmt in (formats or _default_formats()):
            try:
                save_figure(figure, os.path.join(folder, f"{base}.{fmt}"),
                            fmt=fmt, bbox_inches="tight", close=False)
            except Exception:
                LOG.debug("could not write %s", fmt, exc_info=True)
        data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        write_table(data, os.path.join(folder, "data.csv"))
        drawn = getattr(figure, "_spacr_drawn_data", None)
        if isinstance(drawn, pd.DataFrame):
            write_table(drawn, os.path.join(folder, "drawn_data.csv"))
        if spec.get("panels"):
            table, statistics_text = _panel_statistics(frame, spec)
        else:
            table = _auto_statistics(
                tested, sx, sy,
                test=stats_spec.get("test") or None,
                paired=stats_spec.get("paired"),
                pair=str(stats_spec.get("pair") or spec.get("pair") or ""),
                correction=str(stats_spec.get("correction") or "fdr_bh"),
                order=spec.get("order"), count=str(spec.get("count") or ""))
            statistics_text = _statistics_text(table)
        write_table(table, os.path.join(folder, "statistics.csv"))
        with open(os.path.join(folder, "statistics.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write(statistics_text)
        with open(os.path.join(folder, "spec.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(_plain(spec), handle, indent=2)
        with open(os.path.join(folder, "recreate_figure.py"), "w",
                  encoding="utf-8") as handle:
            handle.write(_recreate_script(spec))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for entry in sorted(os.listdir(folder)):
                archive.write(os.path.join(folder, entry), entry)
    return path


def _save_image_zip(figure, path: str, *, formats=None, name: str = "") -> str:
    """Write ONE zip holding a picture figure, its arrays and metadata.

    Inside: the figure in every default format, each array it shows as
    ``image_<n>.tif`` (``.npy`` when tifffile is missing) and
    ``metadata.json`` with the spec, titles, and every array's shape and
    type. A picture has no tidy data and no statistics, so neither file is
    written.
    """
    import tempfile
    import zipfile

    from ..plot import save_figure

    spec = _capture_view(figure, dict(getattr(figure, "_spacr_spec", None)
                                      or {}))
    base = "".join(c if c.isalnum() or c in "-_." else "_"
                   for c in str(name or spec.get("title") or "figure"))
    base = base.strip("._") or "figure"
    arrays = list(getattr(figure, "_spacr_image", None) or [])
    shapes = []
    with tempfile.TemporaryDirectory(prefix="spacr_img_") as folder:
        for fmt in (formats or _default_formats()):
            try:
                save_figure(figure, os.path.join(folder, f"{base}.{fmt}"),
                            fmt=fmt, bbox_inches="tight", close=False)
            except Exception:
                LOG.debug("could not write %s", fmt, exc_info=True)
        for index, array in enumerate(arrays):
            array = np.asarray(array)
            stem = os.path.join(folder, f"image_{index}")
            try:
                from ..tiff_io import write_tiff

                write_tiff(stem + ".tif", array)
                written = f"image_{index}.tif"
            except Exception:
                np.save(stem + ".npy", array)
                written = f"image_{index}.npy"
            shapes.append({"file": written, "shape": list(array.shape),
                           "dtype": str(array.dtype)})
        metadata = dict(spec)
        metadata["arrays"] = shapes
        with open(os.path.join(folder, "metadata.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(_plain(metadata), handle, indent=2)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for entry in sorted(os.listdir(folder)):
                archive.write(os.path.join(folder, entry), entry)
    return path
