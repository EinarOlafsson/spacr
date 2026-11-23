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
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .style import ROLES, _capture_appearance, _replay_message, _restore_appearance, condition_plot, heatmap_image

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
    appearance = payload.get("appearance")
    if isinstance(appearance, Mapping) and appearance.get("backend") == "pyqtgraph":
        owner = getattr(render, "__self__", None)
        latest = getattr(owner, "export_settings", None)
        if callable(latest):
            payload.update({str(key): _plain(value) for key, value in latest().items()})
        payload["fonts"] = _write_replay_fonts(out)
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
                      kind=(kind if kind == "line" and not record.get("house_graph_type")
                            else _HOUSE_KINDS.get(kind, kind)))
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
    from .style import _file_axes_geometry

    out = dict(spec)
    with _file_axes_geometry(figure):
        out["appearance"] = _capture_appearance(figure)
    from dataclasses import asdict
    from .style import StyleSpec

    presentation = []
    for axes_index, panel_axes in enumerate(getattr(figure, "axes", ())):
        declared = getattr(panel_axes, "_spacr_style_spec", None)
        if isinstance(declared, StyleSpec):
            presentation.append({"axes": axes_index, "specification": asdict(declared)})
    if presentation:
        out["presentation"] = presentation
    notes = [{"axes": index, "entries": list(getattr(panel_axes, "_spacr_legend_entries", ()))}
             for index, panel_axes in enumerate(getattr(figure, "axes", ()))
             if getattr(panel_axes, "_spacr_legend_entries", ())]
    if notes:
        out["legend_entries"] = notes
    axes = [a for a in getattr(figure, "axes", ()) if a.get_label()
            != "<colorbar>"]
    if axes:
        ax = axes[0]
        condition_recipe = getattr(ax, "_spacr_condition_recipe", None)
        if len(axes) == 1 and not spec.get("panels") and spec.get("kind") in (
                "box", "violin", "strip", "bar", "box_strip", "bar_strip") and condition_recipe:
            condition_recipe = dict(condition_recipe)
            declared = getattr(ax, "_spacr_style_spec", None)
            labels = condition_recipe.get("labels") or (
                list(declared.condition_labels) if isinstance(declared, StyleSpec) else None)
            if labels:
                condition_recipe["labels"] = list(labels)
                condition_recipe["ticklabels"] = [label.get_text() for label in (
                    ax.get_yticklabels() if condition_recipe.get("orientation") == "horizontal" else ax.get_xticklabels())]
                out["condition_recipe"] = condition_recipe
        if spec.get('training_curves'):
            recipe = dict(spec['training_curves'])
            recipe['curves'] = []
            for curve in spec['training_curves']['curves']:
                curve = dict(curve)
                for line in ax.lines:
                    if line.get_label() == curve['label']:
                        curve.update(color=line.get_color(),
                                     linestyle=line.get_linestyle(),
                                     linewidth=line.get_linewidth(),
                                     alpha=line.get_alpha(),
                                     marker=line.get_marker(),
                                     markersize=line.get_markersize())
                    elif line.get_gid() == 'spacr-training-best:' + curve['label']:
                        curve['best_style'] = dict(color=line.get_color(),
                            linestyle=line.get_linestyle(),
                            linewidth=line.get_linewidth(), alpha=line.get_alpha(),
                            marker=line.get_marker(), markersize=line.get_markersize())
                for area in ax.collections:
                    if area.get_gid() == 'spacr-training-band:' + curve['label']:
                        curve['band_style'] = dict(color=area.get_facecolor()[0].tolist(),
                            alpha=area.get_alpha(), linewidth=float(area.get_linewidth()[0]))
                recipe['curves'].append(curve)
            recipe['notes'] = [dict(text=text.get_text(), x=text.get_position()[0],
                                   y=text.get_position()[1], fontsize=text.get_fontsize(),
                                   alpha=text.get_alpha(), ha=text.get_ha(), va=text.get_va())
                               for text in ax.texts]
            out['training_curves'] = recipe
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
        out["size"] = out["appearance"]["size"] or [float(v) for v in figure.get_size_inches()]
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
    grouped_line = spec.get("grouped_line")
    if grouped_line:
        group = grouped_line["grouping_column"]
        columns = grouped_line["columns"]
        data = data.dropna(subset=[group] + columns).copy()
        representation = grouped_line["representation"]
        if representation == "well":
            grouped = ["prc", group]
        elif representation == "plate":
            if "plateID" not in data.columns:
                data[["plateID", "rowID", "columnID"]] = data["prc"].str.split("_", expand=True)
            grouped = ["plateID"] if group == "plateID" else ["plateID", group]
        elif representation == "object":
            grouped = None
        else:
            raise ValueError("Invalid grouped line representation")
        if grouped:
            data = data.groupby(grouped, observed=False)[columns].agg(
                grouped_line["summary_func"]).reset_index()
        data[group] = pd.Categorical(data[group], categories=grouped_line["order"], ordered=True)
        if grouped_line.get("remove_outliers"):
            removed = pd.Index([])
            for value in data[group].unique():
                group_mask = data[group] == value
                for column in columns:
                    values = data.loc[group_mask, column]
                    lower, upper = values.quantile([0.25, 0.75])
                    spread = upper - lower
                    outside = group_mask & ((data[column] < lower - 1.5 * spread)
                                            | (data[column] > upper + 1.5 * spread))
                    removed = removed.union(data.index[outside])
            data = data.drop(removed)
        if grouped_line["mode"] != "groups":
            import numpy as np

            for index, name in enumerate(("log_x", "log_y")):
                if grouped_line[name]:
                    data[columns[index]] = np.log10(data[columns[index]])
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


class _CapturedControlChart(SimpleNamespace):
    """Original control-chart display arrays, independent of the QC engine or GUI."""

    def __len__(self):
        """Return the number of original plate-level observations."""
        return len(self.values)


def _control_chart_view(frame, recipe):
    """Bind frozen limits and rule positions only to their original plate identities and order."""
    from types import SimpleNamespace

    if recipe.get("schema") != 1:
        raise ValueError(_replay_message("Unsupported captured control-chart recipe"))
    recorded = recipe["result"]
    plates = frame["plate"].astype(str).tolist()
    if plates != recorded["plates"]:
        raise ValueError(_replay_message("Captured control limits require the original plate identities and order"))
    values = pd.to_numeric(frame[recorded["value_column"]], errors="coerce").to_numpy(float)
    attributes = dict(recorded, plates=tuple(plates), values=values)
    for name in ("sigma_at", "lower", "upper"):
        attributes[name] = np.asarray(recorded[name], dtype=float)
        if len(attributes[name]) != len(values):
            raise ValueError(_replay_message("Captured control arrays do not match the source plate count"))
    attributes["baseline"] = np.asarray(recorded["baseline"], dtype=int)
    attributes["violations"] = tuple(SimpleNamespace(**item) for item in recorded["violations"])
    return _CapturedControlChart(**attributes)



def _feature_explorer_view(recipe, frame, levels_of):
    """Bind captured ranking descriptions while the original engine recomputes source histogram counts.

    :param recipe: additive feature-histogram recipe from the actual widget.
    :param frame: source table whose original class identities must match.
    :param levels_of: original class-level helper, or its unchanged portable copy.
    :returns: original feature order and frozen ranking descriptions, without a new ranking run.
    :raises ValueError: for an unsupported recipe version.
    """
    from types import SimpleNamespace

    if recipe.get("schema") != 1:
        raise ValueError(_replay_message("Unsupported captured feature-distribution recipe"))
    _, classes = levels_of(frame, recipe["label"])
    if tuple(recipe["classes"]) != classes:
        raise ValueError(_replay_message("Captured feature class identities do not match the source table; retain the original class labels"))
    return SimpleNamespace(label=recipe["label"],
        scores=tuple(SimpleNamespace(**record) for record in recipe["features"]))


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
    An explorer volcano recipe delegates to its original renderer and retains
    the complete source table, transformations and captured identity lookup.
    """
    import numpy as np
    import pandas as pd
    import seaborn as sns

    np.random.seed(int(spec.get("seed", 0)))
    control_recipe = spec.get("control_chart_recipe")
    if control_recipe:
        if axes is not None:
            raise ValueError(_replay_message("Captured control charts require their original panel"))
        result = _control_chart_view(frame, control_recipe)
        renderer = globals().get("_CONTROL_CHART_RENDERER")
        if renderer is None:
            from ..qt.screens.control_chart import _draw_control_chart

            renderer = _draw_control_chart
        figure.clear()
        appearance = spec.get("appearance") or {}
        if appearance.get("size"):
            figure.set_size_inches(*appearance["size"])
        ax = figure.add_subplot(111)
        ink = ((appearance.get("axes") or [{}])[0].get("xlabel") or {}).get("colour", "#000000")
        renderer(ax, result, unit=control_recipe.get("unit"), quantity=control_recipe.get("quantity"), ink=ink)
        _restore_appearance(figure, appearance)
        return ax
    volcano_recipe = spec.get("volcano_style_recipe")
    if volcano_recipe:
        if volcano_recipe.get("schema") != 1 or axes is not None:
            raise ValueError(_replay_message("Unsupported captured explorer volcano recipe or nested panel"))
        from types import SimpleNamespace

        renderer = globals().get("_VOLCANO_STYLE_RENDERER")
        if renderer is None:
            from ..volcano_style import VolcanoStyle, render_volcano

            appearance = VolcanoStyle.from_dict(volcano_recipe["style"])
            renderer = render_volcano
        else:
            appearance = SimpleNamespace(**volcano_recipe["style"])
        if "localizations" in volcano_recipe:
            appearance._spacr_localization_recipe = volcano_recipe["localizations"]
        data = frame.copy()
        data.attrs["units"] = dict(volcano_recipe.get("source_units") or {})
        size = (spec.get("appearance") or {}).get("size") or spec.get("size")
        if size:
            figure.set_size_inches(*size)
        restored, panels = renderer(data, appearance, figure=figure, screen=volcano_recipe.get("screen", False))
        _restore_appearance(restored, spec.get("appearance"))
        return panels if len(panels) > 1 else panels[0]
    feature_recipe = spec.get("feature_explorer_recipe")
    if feature_recipe and str(spec.get("kind")) in ("hist", "histogram"):
        if axes is not None:
            raise ValueError(_replay_message("Captured feature distributions require their own figure"))
        renderer = globals().get("_render_feature_histograms")
        presentation = not bool(spec.get("appearance"))
        if renderer is None:
            from ..qt.widgets.feature_explorer import _render_feature_histograms as renderer
        elif presentation:
            raise ValueError(_replay_message("Standalone feature recreation requires captured appearance"))
        levels_of = globals().get("_class_levels")
        if levels_of is None:
            from ..qt.widgets.feature_rank import _class_levels as levels_of

        result = _feature_explorer_view(feature_recipe, frame, levels_of)
        panels = renderer(figure, frame, result, feature_recipe["bins"],
            colours=feature_recipe["colours"], alpha=feature_recipe["alpha"],
            font_points=feature_recipe["font_points"], presentation=presentation)
        figure._spacr_drawn_data = None
        _restore_appearance(figure, spec.get("appearance"))
        return panels[0] if len(panels) == 1 else panels
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
                if "alpha" in bar:
                    colorbar.solids.set_alpha(bar["alpha"])
                cax.set_xticklabels(bar.get("ticklabels", [f"{value:.3g}" for value in bar["ticks"]]))
                colorbar.outline.set_linewidth(bar["linewidth"])
                colorbar.outline.set_edgecolor(bar["ink"])
                cax.tick_params(length=1.6, width=bar["linewidth"], pad=1.2,
                                labelsize=bar["fontsize"], colors=bar["ink"])
                figure.text(*bar["text_position"], bar["text"], ha="center", va="top",
                            color=bar["ink"], fontsize=bar["fontsize"])
            for index, target in enumerate(targets):
                if index not in slots:
                    target.set_visible(False)
            _restore_appearance(figure, spec.get("appearance"))
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
    condition_recipe = spec.get("condition_recipe")
    if condition_recipe:
        if condition_recipe.get("schema") != 1:
            raise ValueError("Unsupported captured condition recipe schema")
        labels = condition_recipe["labels"]
        horizontal = condition_recipe.get("orientation") == "horizontal"
        category, value = (y, x) if horizontal else (x, y)
        sources = condition_recipe.get("sources")
        if sources:
            if len(sources) != len(labels):
                raise ValueError("Captured condition sources must match the declared slots")
            values = [pd.to_numeric(data.loc[
                data[source["group_column"]].astype(str) == str(source["group"]),
                source["value_column"]], errors="coerce").to_numpy(float) for source in sources]
        else:
            values = [pd.to_numeric(data.loc[data[category].astype(str) == str(label), value], errors="coerce").to_numpy(float)
                      for label in labels]
        condition_plot(ax, values, colours=condition_recipe["colours"],
            kind=condition_recipe["kind"], many=condition_recipe["many"],
            points=condition_recipe["points"], seed=condition_recipe["seed"],
            positions=condition_recipe["positions"], labels=labels,
            orientation=condition_recipe.get("orientation", "vertical"))
        (ax.set_yticks if horizontal else ax.set_xticks)(condition_recipe["positions"], condition_recipe.get("ticklabels", labels))
    elif kind == "box":
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
        grouped_line = spec.get("grouped_line")
        training = spec.get('training_curves')
        if training:
            import numpy as np

            for curve in training['curves']:
                selected = data.loc[data[hue] == curve['label']]
                epochs = pd.to_numeric(selected[x], errors='coerce').to_numpy()
                values = pd.to_numeric(selected[y], errors='coerce').to_numpy()
                options = {key: curve[key] for key in (
                    'color', 'linestyle', 'linewidth', 'alpha', 'marker', 'markersize')
                    if key in curve}
                ax.plot(epochs, values, label=curve['label'], **options)
                sd_column = y + '__sd'
                if training['band'] and curve['kind'] == 'mean' and sd_column in selected:
                    sd = pd.to_numeric(selected[sd_column], errors='coerce').to_numpy()
                    if np.isfinite(sd).any():
                        ax.fill_between(epochs, values - sd, values + sd,
                                        **curve.get('band_style', dict(
                                            color=curve['color'], alpha=0.15, linewidth=0)))
                if training['mark_best'] and training['direction'] is not None and np.isfinite(values).any():
                    finite = np.flatnonzero(np.isfinite(values))
                    pick = np.argmin(values[finite]) if training['direction'] == 'min' else np.argmax(values[finite])
                    index = finite[pick]
                    ax.plot([epochs[index]], [values[index]],
                            **curve.get('best_style', dict(color=curve['color'],
                                marker='o', linestyle='None', markersize=5,
                                alpha=curve['alpha'])))
            for note in training.get('notes', []):
                options = {key: value for key, value in note.items()
                           if key not in ('x', 'y', 'text')}
                ax.text(note['x'], note['y'], note['text'],
                        transform=ax.transAxes, **options)
            if training['curves']:
                ax.grid(True, alpha=0.15)
                ax.legend(loc='best', fontsize=8, framealpha=0.6)
        elif grouped_line and grouped_line["mode"] == "groups":
            summary = (data.groupby(x, observed=False)
                       .agg(centre=(y, grouped_line["summary_func"]), spread=(y, grouped_line.get("spread_func", "std")))
                       .reindex(grouped_line["order"]).reset_index())
            ax.errorbar(range(len(summary)), summary["centre"],
                        yerr=summary["spread"].fillna(0.0), marker="o", linewidth=1,
                        markersize=6, capsize=3, color=grouped_line["palette"][0])
            ax.set_xticks(range(len(summary)))
            ax.set_xticklabels([str(value) for value in summary[x]])
            if grouped_line.get("point_colours") is not None:
                helpers = globals()
                if "line_artist_defaults" not in helpers:
                    from .style import line_artist_defaults as draw_centres
                else:
                    draw_centres = helpers["line_artist_defaults"]
                draw_centres(ax, point_colours=grouped_line["point_colours"])
        elif grouped_line and grouped_line["mode"] == "std_area":
            summary = data.pivot_table(index=x, values=y, aggfunc=["mean", "std"]).reset_index()
            summary.columns = [x, "centre", "spread"]
            color = grouped_line["band_color"]
            sns.lineplot(data=summary, x=x, y="centre", ax=ax, marker="o",
                         linewidth=grouped_line["band_width"], markersize=0,
                         color=color, label="mean_" + y)
            ax.fill_between(summary[x], summary["centre"] - summary["spread"],
                            summary["centre"] + summary["spread"], color=color, alpha=0.25)
        else:
            options = dict(estimator=None, errorbar=None, sort=False)
            if grouped_line:
                options.update(estimator="mean", errorbar=("ci", 95), sort=True,
                               palette=grouped_line["palette"], marker="o",
                               linewidth=1, markersize=6, seed=0)
            options.update(spec.get("line_options") or {})
            sns.lineplot(**common, **options)
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
                patch.set_alpha(recipe.get("alpha", 1.0))
                patch.set_gid("spacr-venn-" + region)
                patch.set_edgecolor("none")
        for label in list(diagram.set_labels or []) + list(diagram.subset_labels or []):
            if label is not None:
                label.set_fontsize(recipe["fontsize"])
                label.set_color(recipe["ink"])
        if recipe.get("set_label_positions") is not None:
            for label, position in zip(diagram.set_labels, recipe["set_label_positions"]):
                label.set_transform(ax.transAxes)
                label.set_position(position)
                label.set_horizontalalignment("center")
                label.set_verticalalignment("top")
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
        if plate.get("presentation") == 1:
            heatmap_image(ax, data.to_numpy(dtype=float), cmap=cmap,
                          vmin=plate["limits"][0], vmax=plate["limits"][1],
                          extent=(0, columns, rows, 0), centre=plate.get("centre"))
        else:
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
    if axes is None:
        _restore_appearance(figure, spec.get("appearance"))
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

{appearance_helpers}

def main():
    """Read the data and the recipe, draw, and save."""
    here = os.path.dirname(os.path.abspath(__file__))
    frame = pd.read_csv(os.path.join(here, "data.csv"), float_precision="round_trip")
    with open(os.path.join(here, "spec.json"), encoding="utf-8") as handle:
        spec = json.load(handle)
    _register_replay_fonts(here, spec.get("fonts", ()))
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
    constants = dict(ROLES=style.ROLES, TYPE_SCALE=style.TYPE_SCALE, WEIGHTS=style.WEIGHTS, TICK_ROTATION=style.TICK_ROTATION, DATA_ALPHA=style.DATA_ALPHA, AUTO_N=style.AUTO_N, DENSE_N=style.DENSE_N, BAR_FRAC=style.BAR_FRAC, SERIES=style.SERIES, POLICY_VERSION=style.POLICY_VERSION,
        CONTROL_CONDITIONS=panels.CONTROL_CONDITIONS, MIN_CONTROLS=panels.MIN_CONTROLS,
        ZERO=baseline.ZERO, CONTROLS=baseline.CONTROLS, NAMED=baseline.NAMED,
        VALUE=baseline.VALUE, CONTROL_LABELS=baseline.CONTROL_LABELS,
        _GENE_ID_PREFIXES=hits._GENE_ID_PREFIXES)
    pieces.extend(f"{name} = {value!r}" for name, value in constants.items())
    pieces.append('def tr(message):\n    """Return captured English labels without a spaCR dependency."""\n    return message')
    for name in ("_BRACKET", "NUISANCE_TERMS"):
        pattern = getattr(hits, name)
        pieces.append(f"{name} = re.compile({pattern.pattern!r}, {int(pattern.flags)})")
    objects = [style.Palette, panels.Panel, baseline.Baseline,
        baseline._control_rows, baseline.describe_intercept, baseline.resolve, baseline.apply,
        hits.tested_family, hits._gene_id_of, hits.gene_of,
        localisation.of, localisation.mask,
        style._distribution_orientation, style._condition_kind_count, style.condition_kind, style.condition_colours, style.condition_plot, style.panel_letter, style._legend_note, style.reference_line, style.annotate, style.text_legend, style.rotate_ticks,
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


def _condition_script() -> str:
    """Copy captured categorical and heatmap rendering into standalone archives.

    Old archives without a condition recipe keep their existing drawing path.

    :returns: source for the shared renderer and its exact policy constants.
    """
    import inspect
    import textwrap
    from . import style

    constants = dict(AUTO_N=style.AUTO_N, DENSE_N=style.DENSE_N,
        BAR_FRAC=style.BAR_FRAC, DATA_ALPHA=style.DATA_ALPHA, SERIES=style.SERIES,
        WEIGHTS=style.WEIGHTS, POLICY_VERSION=style.POLICY_VERSION)
    pieces = ["from typing import Sequence", *(f"{name} = {value!r}" for name, value in constants.items())]
    pieces.extend(textwrap.dedent(inspect.getsource(function)) for function in (
        style.Palette, style._distribution_orientation, style._condition_kind_count, style.condition_kind,
        style.condition_colours, style.condition_plot, style.line_artist_defaults, style.centred_norm,
        style.heatmap_image))
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
    from .style import TYPE_SCALE, _legend_note, text_legend

    tree = ast.parse(textwrap.dedent(inspect.getsource(plot_lorenz_curves)))
    calculations = [node for node in tree.body[0].body
                    if isinstance(node, ast.FunctionDef)
                    and node.name in ("lorenz_curve", "gini_coefficient")]
    return "\n\n".join([
        "import numpy as np\nfrom typing import Sequence",
        f"TYPE_SCALE = {TYPE_SCALE!r}",
        *(ast.unparse(node) for node in calculations),
        textwrap.dedent(inspect.getsource(_legend_note)),
        textwrap.dedent(inspect.getsource(text_legend))])


def _portable_renderer_source(objects, *, renamed=None, omit_calls=()):
    """Copy normal renderer source with one shared application-binding adapter.

    :param objects: original renderer functions or policy classes to copy.
    :param renamed: explicit helper names isolated from the general script.
    :param omit_calls: application registration, presentation and delivery adapters.
    :returns: standalone source fragments retaining the original calculations.
    """
    import ast
    import inspect
    import textwrap
    from ..qt.i18n import tr

    renamed = dict(renamed or {})
    omit_calls = set(omit_calls)

    class ReplayBindings(ast.NodeTransformer):
        """Keep original calculations while adapting application-only bindings for a saved script."""

        def visit_FunctionDef(self, node):
            """Rename copied producer helpers and erase dependency-only annotations."""
            node.name = renamed.get(node.name, node.name)
            node.returns = None
            for argument in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                argument.annotation = None
            for argument in (node.args.vararg, node.args.kwarg):
                if argument is not None:
                    argument.annotation = None
            return self.generic_visit(node)

        def visit_Name(self, node):
            """Bind original cross-calls to their copied source helpers."""
            node.id = renamed.get(node.id, node.id)
            return node

        def visit_ImportFrom(self, node):
            """Omit only known application adapters whose shared implementation is copied."""
            if node.level:
                if node.module == "qt.i18n":
                    return ast.Assign(targets=[ast.Name(id="tr", ctx=ast.Store())],
                                      value=ast.Name(id="_replay_message", ctx=ast.Load()))
                if node.module not in ("figures.bundle", "figures.style", "figure_font", "plot"):
                    raise ValueError(tr("Portable figure replay found an uncaptured application dependency"))
                return None
            return node

        def visit_Expr(self, node):
            """Remove application registration and defaults after retaining the original scientific drawing."""
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in omit_calls:
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            """Use the supplied replay figure and leave file delivery to the archive main function."""
            condition = ast.unparse(node.test)
            if condition == "figure is None":
                return [self.visit(child) for child in node.orelse]
            if condition == "save_path is not None":
                return None
            return self.generic_visit(node)

    pieces = []
    for function in objects:
        original = ast.parse(textwrap.dedent(inspect.getsource(function)))
        copied = ReplayBindings().visit(original)
        pieces.append(ast.unparse(ast.fix_missing_locations(copied)))
    return pieces


def _volcano_style_script() -> str:
    """Copy the original explorer renderer and shared scalar-key owner for portable replay.

    Only application registration, final presentation and file delivery are
    omitted: captured appearance supplies those operations. Transformations,
    threshold estimation, masks, shapes, annotations and scalar-key drawing
    come directly from their normal implementation source.
    """
    from .. import style_base, volcano_style
    from . import style

    functions = (volcano_style._resolve_effect_threshold, volcano_style._prepare,
        volcano_style._colour_values, volcano_style._localization_column,
        volcano_style._captured_localizations, volcano_style._draw_by_localization,
        volcano_style._draw_points, volcano_style._scatter_by_shape,
        volcano_style._draw_reference_lines, volcano_style._annotate,
        volcano_style.page_ground, volcano_style._paint_ink, volcano_style._finish_axes,
        volcano_style.render_volcano)
    renamed = {function.__name__: "_volcano" + function.__name__ if function.__name__.startswith("_")
               else "_volcano_" + function.__name__ for function in functions}

    constants = dict(SERIES=style.SERIES, TYPE_SCALE=style.TYPE_SCALE, WEIGHTS=style.WEIGHTS,
        DATA_ALPHA=style.DATA_ALPHA, GENERAL_DEFAULTS=style.GENERAL_DEFAULTS,
        MARKER_SHAPES=volcano_style.MARKER_SHAPES,
        _MIN_CONTROLS_FOR_THRESHOLD=volcano_style._MIN_CONTROLS_FOR_THRESHOLD,
        FONT_FAMILY=style_base.FONT_FAMILY, GENERIC_FAMILIES=style_base.GENERIC_FAMILIES)
    pieces = ["import numpy as np\nfrom typing import Dict, List, Optional, Sequence",
              *(f"{name} = {value!r}" for name, value in constants.items())]
    objects = (style.Palette, style._legend_note, style.reference_line,
        style._detached_colour_norm, style._numeric_colourbar, style.attach_colourbar,
        style.scale_tick_labels, style_base.font_names, style_base.font_rc, *functions)
    pieces.extend(_portable_renderer_source(objects, renamed=renamed,
        omit_calls=("_register_figure_data", "_present_volcano", "use_open_sans_for_figures")))
    pieces.append("_VOLCANO_STYLE_RENDERER = " + renamed["render_volcano"])
    return "\n\n".join(pieces)


def _control_chart_script() -> str:
    """Copy the original control renderer and captured-view binding without importing Qt in replay."""
    from ..qt.screens import control_chart
    from ..qt.widgets.control_chart import RULE_NAMES
    from . import style

    constants = dict(DATA_ALPHA=style.DATA_ALPHA, GENERAL_DEFAULTS=style.GENERAL_DEFAULTS,
                     SERIES=style.SERIES, TYPE_SCALE=style.TYPE_SCALE, WEIGHTS=style.WEIGHTS,
                     RULE_NAMES=RULE_NAMES)
    pieces = ["import numpy as np\nfrom typing import Dict, List\nfrom types import SimpleNamespace",
              *(f"{name} = {value!r}" for name, value in constants.items())]
    pieces.append('def tr(message):\n    """Use the existing portable message adapter."""\n    return _replay_message(message)')
    pieces.extend(_portable_renderer_source((style.Palette, _CapturedControlChart, _control_chart_view,
        control_chart._draw_control_chart), omit_calls=("_present_control_chart",)))
    pieces.append("_CONTROL_CHART_RENDERER = _draw_control_chart")
    return "\n\n".join(pieces)



def _feature_explorer_script() -> str:
    """Copy original class-bin calculations and the normal painter into feature archives.

    :returns: package-independent source using the same distribution engine and captured appearance.
    """
    from ..qt.widgets import feature_explorer, feature_rank
    from . import style

    constants = dict(MAX_DRAWN=feature_explorer.MAX_DRAWN, MAX_CLASSES=feature_rank.MAX_CLASSES,
                     TYPE_SCALE=style.TYPE_SCALE)
    pieces = ["import numpy as np\nimport pandas as pd",
              "_UNLABELLED = object()",
              *(str(name)+" = "+repr(value) for name,value in constants.items()),
              'def tr(message):\n    """Retain the shared standalone translation binding."""\n    return _replay_message(message)']
    pieces.extend(_portable_renderer_source((feature_rank.ExplorerError, feature_rank._class_levels,
        feature_rank.distributions, style.panel_letter, _feature_explorer_view,
        feature_explorer._render_feature_histograms)))
    return "\n\n".join(pieces)


def _appearance_script() -> str:
    """Copy the shared appearance replay into new scripts without a spaCR dependency."""
    import inspect
    import textwrap
    from . import style

    return "\n\n".join(textwrap.dedent(inspect.getsource(function)) for function in (
        style._restore_text_appearance, style._restore_artist_appearance,
        style._replay_message, style._restore_colour_norm, style._restore_legend_appearance,
        style._restore_appearance, style._register_replay_fonts))


def _write_replay_fonts(folder) -> list:
    """Package the licensed bundled faces and record their content hashes for replay."""
    import hashlib
    import shutil
    from pathlib import Path
    from ..figure_font import bundled_faces, font_dir

    records = []
    for filename in bundled_faces():
        source = Path(filename)
        shutil.copyfile(source, Path(folder) / source.name)
        records.append(dict(name=source.name, sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    license_path = Path(font_dir()).parent / "OFL.txt"
    shutil.copyfile(license_path, Path(folder) / "OpenSans-OFL.txt")
    return records


def _recreate_script(spec=None) -> str:
    """The Python script saved beside a figure that re-creates it."""
    import inspect
    import textwrap

    return _SCRIPT.format(
        reference=repr(ROLES["reference"]),
        appearance_helpers=_appearance_script(),
        annotate=textwrap.dedent(inspect.getsource(_annotate)),
        recipe_frame=textwrap.dedent(inspect.getsource(_recipe_frame)),
        regression_helpers=(_feature_explorer_script() if (spec or {}).get("feature_explorer_recipe")
                            else _control_chart_script() if (spec or {}).get("control_chart_recipe")
                            else _volcano_style_script() if (spec or {}).get("volcano_style_recipe")
                            else _regression_script() if (spec or {}).get("kind") == "regression_panel"
                            else _lorenz_script() if (spec or {}).get("kind") == "lorenz"
                            else _condition_script() if ((spec or {}).get("condition_recipe")
                                or (spec or {}).get("kind") == "plate_heatmap"
                                or ((spec or {}).get("grouped_line") or {}).get("point_colours") is not None) else ""),
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
    if spec.get("grouped_line"):
        statistics_recipe = dict(spec["grouped_line"], remove_outliers=False,
                                 log_x=False, log_y=False)
        tested = _recipe_frame(frame, dict(spec, grouped_line=statistics_recipe))
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
        spec["fonts"] = _write_replay_fonts(folder)
        for fmt in (formats or _default_formats()):
            try:
                save_figure(figure, os.path.join(folder, f"{base}.{fmt}"),
                            fmt=fmt, bbox_inches="tight", close=False)
            except Exception:
                LOG.debug("could not write %s", fmt, exc_info=True)
        data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        write_table(data, os.path.join(folder, "data.csv"), canonicalise=False)
        drawn = getattr(figure, "_spacr_drawn_data", None)
        if isinstance(drawn, pd.DataFrame):
            write_table(drawn, os.path.join(folder, "drawn_data.csv"),
                        canonicalise=False)
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
