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
        column of repeated measures), ``matrix`` or ``title``.
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
            out.update(xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()))
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


def _draw(figure, frame, spec):
    """Draw ``frame`` onto ``figure`` as ``spec`` describes, and return the axes.

    The figure is cleared and redrawn with seaborn and Matplotlib only, so
    the same function, copied into a saved figure's script, re-creates it
    without spaCR installed.
    """
    import numpy as np
    import pandas as pd
    import seaborn as sns

    np.random.seed(int(spec.get("seed", 0)))
    kind = str(spec.get("kind") or "box")
    x = spec.get("x") or None
    y = spec.get("y") or None
    hue = spec.get("hue") or None
    figure.clear()
    if spec.get("size"):
        figure.set_size_inches(*spec["size"])
    ax = figure.add_subplot(111)
    data = frame.copy()
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
        sns.scatterplot(**common)
    elif kind == "line":
        sns.lineplot(**common)
    elif kind == "hex":
        ax.hexbin(data[x], data[y], gridsize=int(spec.get("gridsize", 30)),
                  cmap=spec.get("cmap") or "viridis", mincnt=1)
    elif kind == "reg":
        sns.regplot(data=data, x=x, y=y, ax=ax)
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
    for key, setter in (("title", ax.set_title), ("xlabel", ax.set_xlabel),
                        ("ylabel", ax.set_ylabel),
                        ("xscale", ax.set_xscale),
                        ("yscale", ax.set_yscale)):
        if spec.get(key):
            setter(spec[key])
    if spec.get("xlim"):
        ax.set_xlim(*spec["xlim"])
    if spec.get("ylim"):
        ax.set_ylim(*spec["ylim"])
    _annotate(ax, spec)
    return ax


_SCRIPT = '''"""Re-create the figure in this folder from data.csv and spec.json.

Needs pandas, numpy, scipy, matplotlib and seaborn. Run it beside the two
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


def _recreate_script() -> str:
    """The Python script saved beside a figure that re-creates it."""
    import inspect
    import textwrap

    return _SCRIPT.format(
        reference=repr(ROLES["reference"]),
        annotate=textwrap.dedent(inspect.getsource(_annotate)),
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


def _save_zip(figure, path: str, *, formats=None, name: str = "") -> str:
    """Write ONE zip holding a figure, its data, statistics and recipe.

    Inside: the image in every default format, ``data.csv`` (the tidy rows),
    ``statistics.csv`` (one table: normality, equal variance, omnibus and
    pairwise tests, each with whether it was chosen automatically or by the
    user) and ``statistics.txt``, ``spec.json`` (the full plotting recipe)
    and ``recreate_figure.py``, which draws the figure again from the CSV
    and the JSON.

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
        tested, sx, sy = _as_frame(groups, "group", "value"), "group", "value"
        if sx not in tested.columns:
            tested = tested.melt(var_name=sx, value_name=sy)
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
        table = _auto_statistics(
            tested, sx, sy,
            test=stats_spec.get("test") or None,
            paired=stats_spec.get("paired"),
            pair=str(stats_spec.get("pair") or spec.get("pair") or ""),
            correction=str(stats_spec.get("correction") or "fdr_bh"),
            order=spec.get("order"), count=str(spec.get("count") or ""))
        write_table(table, os.path.join(folder, "statistics.csv"))
        with open(os.path.join(folder, "statistics.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write(_statistics_text(table))
        with open(os.path.join(folder, "spec.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(_plain(spec), handle, indent=2)
        with open(os.path.join(folder, "recreate_figure.py"), "w",
                  encoding="utf-8") as handle:
            handle.write(_recreate_script())
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
