"""Publication panel packages for regression and permutation results.

One call writes exactly four files for one manuscript panel: a vector PDF
with its narrative legend, a plot-only PNG, a point-level data CSV, and a
key-value statistics CSV.  The numerical result table remains the source of
truth; this module only applies a declared call rule and renders it.
"""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import font_manager
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.annotations import Link
from pypdf.generic import (
    NameObject,
    TextStringObject,
)

from .figures.style import INK_PRINT, LABEL_GROUND_PRINT

OPEN_SANS_DIRECTORY = Path(__file__).resolve().parent / "resources/font/open_sans/static"
OPEN_SANS_REGULAR = OPEN_SANS_DIRECTORY / "OpenSans-Regular.ttf"
OPEN_SANS_BOLD = OPEN_SANS_DIRECTORY / "OpenSans-Bold.ttf"
for _font_path in (OPEN_SANS_REGULAR, OPEN_SANS_BOLD):
    if _font_path.exists():
        font_manager.fontManager.addfont(_font_path)
OPEN_SANS_FAMILY = (
    font_manager.FontProperties(fname=OPEN_SANS_REGULAR).get_name()
    if OPEN_SANS_REGULAR.exists()
    else "Open Sans"
)


LOPIT_ORDER = (
    "dense granules",
    "rhoptries 1",
    "rhoptries 2",
    "micronemes",
    "apical 2",
    "PM - integral",
    "IMC",
    "ER 1",
    "Golgi",
    "mitochondrion - soluble",
    "mitochondrion - membranes",
    "nucleus - chromatin",
    "nucleus - non-chromatin",
    "cytosol",
    "Unknown",
    "No metadata",
    "Non-targeting",
)

LOPIT_COLOURS = {
    "dense granules": "#1B9E77",
    "rhoptries 1": "#6A3D9A",
    "rhoptries 2": "#B159C0",
    "micronemes": "#E7298A",
    "apical 2": "#A6A600",
    "PM - integral": "#E6AB02",
    "IMC": "#66A61E",
    "ER 1": "#CC79A7",
    "Golgi": "#A6761D",
    "mitochondrion - soluble": "#D95F02",
    "mitochondrion - membranes": "#E6550D",
    "nucleus - chromatin": "#1F78B4",
    "nucleus - non-chromatin": "#80B1D3",
    "cytosol": "#00A6D6",
    "Unknown": "#969696",
    "No metadata": "#D0D0D0",
    "Non-targeting": "#222222",
}


@dataclass(frozen=True)
class PanelNarrative:
    """Text placed below a panel in its PDF deliverable."""

    legend: str
    purpose: str
    shows: str
    implications: str


@dataclass(frozen=True)
class PanelStyle:
    """Visual settings shared by all manuscript panels."""

    point_size: float = 104.0
    point_alpha: float = 0.60
    line_width: float = 0.50
    line_color: str = INK_PRINT
    figure_width: float = 7.2
    plot_height: float = 5.443902439
    pdf_height: float = 8.8
    png_dpi: int = 400
    axes_left: float = 0.10
    axes_width: float = 0.62
    pdf_axes_bottom: float = 0.48
    pdf_axes_height: float = 0.507272727


DEFAULT_PANEL_STYLE = PanelStyle()


def _normalise_guide(value: object) -> str:
    text = str(value).strip()
    for prefix in ("TGGT1_", "TGME49_"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def guide_control_threshold(
    guide_results: pd.DataFrame,
    *,
    effect_column: str,
    guide_column: str = "grna",
    multiplier: float = 3.0,
) -> tuple[float, dict[str, float | int]]:
    """Return arithmetic mean + ``multiplier`` sample SD of NT gRNAs."""
    if guide_column not in guide_results:
        raise ValueError(f"Guide table lacks {guide_column!r}")
    guides = guide_results[guide_column].map(_normalise_guide)
    control = guides.str.startswith("000000_")
    values = pd.to_numeric(
        guide_results.loc[control, effect_column], errors="raise"
    )
    if len(values) < 2:
        raise ValueError("At least two 000000_* gRNAs are required")
    centre = float(values.mean())
    sample_sd = float(values.std(ddof=1))
    threshold = centre + float(multiplier) * sample_sd
    return threshold, {
        "control_grnas": int(len(values)),
        "control_mean": centre,
        "control_sample_sd": sample_sd,
        "effect_multiplier": float(multiplier),
        "effect_threshold": threshold,
    }


def apply_primary_call(
    results: pd.DataFrame,
    *,
    effect_column: str,
    bh_column: str,
    effect_threshold: float,
) -> pd.DataFrame:
    """Attach the common BH-plus-positive-gRNA-cut call fields."""
    frame = results.copy()
    effect = pd.to_numeric(frame[effect_column], errors="raise")
    bh = frame[bh_column].astype(bool)
    frame["plot_effect_threshold"] = float(effect_threshold)
    frame["passes_effect_threshold"] = effect.gt(float(effect_threshold))
    frame["primary_call"] = bh & frame["passes_effect_threshold"]
    return frame


def shared_limits(
    frames: Sequence[pd.DataFrame],
    *,
    x_column: str,
    y_column: str,
    x_padding: float = 0.05,
    y_padding: float = 0.06,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return finite padded limits shared by a set of matched panels."""
    if not frames:
        raise ValueError("At least one frame is required")
    x = pd.concat(
        [pd.to_numeric(frame[x_column], errors="raise") for frame in frames],
        ignore_index=True,
    ).to_numpy(float)
    y = pd.concat(
        [pd.to_numeric(frame[y_column], errors="raise") for frame in frames],
        ignore_index=True,
    ).to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Shared-limit columns must be finite")
    x_span = max(float(x.max() - x.min()), 1e-9)
    y_span = max(float(y.max() - y.min()), 1e-9)
    x_limits = (
        min(0.0, float(x.min()) - x_padding * x_span),
        float(x.max()) + x_padding * x_span,
    )
    y_limits = (0.0, float(y.max()) + y_padding * y_span)
    return x_limits, y_limits


def _draw_panel(
    axis,
    data: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    lopit_column: str,
    x_label: str,
    y_label: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    horizontal_threshold: float,
    effect_threshold: float,
    style: PanelStyle,
    palette: Mapping[str, str],
    point_label_column: str,
) -> None:
    for category in LOPIT_ORDER:
        mask = data[lopit_column].astype(str).eq(category)
        if not mask.any():
            continue
        axis.scatter(
            data.loc[mask, x_column],
            data.loc[mask, y_column],
            s=style.point_size,
            color=palette[category],
            alpha=style.point_alpha,
            edgecolors="none",
            linewidths=0,
            rasterized=False,
            label=category,
        )
    axis.axhline(
        float(horizontal_threshold), color=style.line_color, linestyle="--",
        linewidth=style.line_width,
    )
    axis.axvline(
        float(effect_threshold), color=style.line_color, linestyle=":",
        linewidth=style.line_width,
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_box_aspect(1)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_color(style.line_color)
        axis.spines[side].set_linewidth(style.line_width)
    axis.tick_params(
        axis="both", which="both", color=style.line_color,
        labelcolor=INK_PRINT, width=style.line_width,
    )
    axis.xaxis.label.set_color(INK_PRINT)
    axis.yaxis.label.set_color(INK_PRINT)
    axis.legend(
        title="LOPIT/TAGM",
        loc="upper left",
        bbox_to_anchor=(1.015, 1.01),
        borderaxespad=0,
        frameon=False,
        fontsize=6.3,
        title_fontsize=6.6,
        handletextpad=0.35,
        labelspacing=0.30,
        markerscale=0.65,
        scatterpoints=1,
    )
    label_mask = data["label_above_threshold"].astype(bool)
    texts = []
    for _, row in data.loc[label_mask].iterrows():
        texts.append(
            axis.text(
                float(row[x_column]),
                float(row[y_column]),
                str(row[point_label_column]),
                color=INK_PRINT,
                fontsize=5.8,
                ha="left",
                va="bottom",
                clip_on=True,
                zorder=5,
            )
        )
    if texts:
        adjust_text(
            texts,
            ax=axis,
            ensure_inside_axes=True,
            prevent_crossings=True,
            expand=(1.22, 1.30),
            force_text=(0.80, 1.00),
            force_static=(0.20, 0.30),
            force_explode=(0.80, 1.00),
            explode_radius=55,
            max_move=(24, 24),
            min_arrow_len=2,
            arrowprops={
                "arrowstyle": "-",
                "color": style.line_color,
                "linewidth": style.line_width,
            },
        )


def _add_pdf_point_links(
    path: Path,
    data: pd.DataFrame,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    x_column: str,
    y_column: str,
    gene_label_column: str,
    gene_url_column: str,
    style: PanelStyle,
) -> int:
    """Add one borderless link annotation per plotted point."""
    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise ValueError("Panel link annotation expects a one-page PDF")
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    figure_width_points = style.figure_width * 72.0
    figure_height_points = style.pdf_height * 72.0
    axes_left = style.axes_left
    axes_bottom = style.pdf_axes_bottom
    axes_width = style.axes_width
    axes_height = style.pdf_axes_height
    xmin, xmax = map(float, x_limits)
    ymin, ymax = map(float, y_limits)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("PDF point links require increasing axis limits")
    radius = max(2.8, float(np.sqrt(style.point_size)) * 0.60)
    linked = 0
    for _, row in data.iterrows():
        url = str(row[gene_url_column]).strip()
        label = str(row[gene_label_column]).strip()
        if not url or not label:
            raise ValueError("Every plotted point must have a gene label and URL")
        x_fraction = axes_left + axes_width * (
            (float(row[x_column]) - xmin) / (xmax - xmin)
        )
        y_fraction = axes_bottom + axes_height * (
            (float(row[y_column]) - ymin) / (ymax - ymin)
        )
        x_point = x_fraction * figure_width_points
        y_point = y_fraction * figure_height_points
        annotation = Link(
            rect=(
                x_point - radius,
                y_point - radius,
                x_point + radius,
                y_point + radius,
            ),
            url=url,
        )
        annotation[NameObject("/Contents")] = TextStringObject(label)
        annotation[NameObject("/T")] = TextStringObject(label)
        writer.add_annotation(0, annotation)
        linked += 1
    temporary = path.with_name(f".{path.stem}.linked.pdf")
    with temporary.open("wb") as handle:
        writer.write(handle)
    temporary.replace(path)
    return linked


def _wrapped_block(title: str, text: str, width: int = 112) -> str:
    return f"{title}\n" + textwrap.fill(str(text).strip(), width=width)


def write_panel_package(
    results: pd.DataFrame,
    destination: str | Path,
    *,
    panel_id: str,
    x_column: str,
    y_column: str,
    lopit_column: str,
    x_label: str,
    y_label: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    horizontal_threshold: float,
    horizontal_threshold_label: str,
    effect_threshold: float,
    effect_threshold_label: str,
    narrative: PanelNarrative,
    gene_label_column: str | None = None,
    gene_url_column: str | None = None,
    point_label_column: str | None = None,
    statistics: Mapping[str, object] | None = None,
    style: PanelStyle = DEFAULT_PANEL_STYLE,
    palette: Mapping[str, str] = LOPIT_COLOURS,
) -> dict[str, Path]:
    """Write the PDF, PNG, stats CSV, and plotted-data CSV for one panel."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(results[lopit_column].astype(str)) - set(palette))
    if unknown:
        raise ValueError("LOPIT categories lack colours: " + ", ".join(unknown))
    if not (0 < style.point_alpha <= 1):
        raise ValueError("point_alpha must be in (0, 1]")
    if style.point_size <= 0:
        raise ValueError("point_size must be positive")

    data = results.copy()
    data["plot_x"] = pd.to_numeric(data[x_column], errors="raise")
    data["plot_y"] = pd.to_numeric(data[y_column], errors="raise")
    data["lopit_color"] = data[lopit_column].astype(str).map(palette)
    data["horizontal_threshold"] = float(horizontal_threshold)
    data["effect_threshold"] = float(effect_threshold)
    data["point_size"] = float(style.point_size)
    data["point_alpha"] = float(style.point_alpha)
    data["marker_face_color"] = data["lopit_color"]
    data["marker_edge_color"] = "none"
    data["line_color"] = style.line_color
    data["line_width_points"] = float(style.line_width)
    resolved_point_label = point_label_column or gene_label_column
    data["point_label"] = (
        data[resolved_point_label].astype(str)
        if resolved_point_label is not None
        else ""
    )
    data["label_above_threshold"] = (
        data["plot_x"].gt(float(effect_threshold))
        & data["plot_y"].gt(float(horizontal_threshold))
        & data["point_label"].str.strip().ne("")
    )

    stem = destination / panel_id
    paths = {
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "stats": destination / f"{panel_id}_stats.csv",
        "data": destination / f"{panel_id}_data.csv",
    }

    rc = {
        "font.family": OPEN_SANS_FAMILY,
        "font.size": 9.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
    with plt.rc_context(rc):
        from .plot import save_figure

        png_figure = plt.figure(
            figsize=(style.figure_width, style.plot_height),
            facecolor=LABEL_GROUND_PRINT,
        )
        png_axis = png_figure.add_axes(
            [style.axes_left, 0.16, style.axes_width, 0.82]
        )
        _draw_panel(
            png_axis, data, x_column="plot_x", y_column="plot_y",
            lopit_column=lopit_column, x_label=x_label, y_label=y_label,
            x_limits=x_limits, y_limits=y_limits,
            horizontal_threshold=horizontal_threshold,
            effect_threshold=effect_threshold, style=style, palette=palette,
            point_label_column="point_label",
        )
        save_figure(
            png_figure,
            paths["png"],
            fmt="png",
            dpi=style.png_dpi,
            close=True,
            save_mode="print",
            announce_colours=False,
            bbox_inches="tight",
            facecolor=LABEL_GROUND_PRINT,
        )

        pdf_figure = plt.figure(
            figsize=(style.figure_width, style.pdf_height),
            facecolor=LABEL_GROUND_PRINT,
        )
        pdf_axis = pdf_figure.add_axes(
            [
                style.axes_left,
                style.pdf_axes_bottom,
                style.axes_width,
                style.pdf_axes_height,
            ]
        )
        _draw_panel(
            pdf_axis, data, x_column="plot_x", y_column="plot_y",
            lopit_column=lopit_column, x_label=x_label, y_label=y_label,
            x_limits=x_limits, y_limits=y_limits,
            horizontal_threshold=horizontal_threshold,
            effect_threshold=effect_threshold, style=style, palette=palette,
            point_label_column="point_label",
        )
        legend = (
            f"{narrative.legend} Points are colored by LOPIT/TAGM "
            "compartment using the shared palette; the exact category and "
            "hex color for every point are recorded in the accompanying data "
            f"CSV. The dashed horizontal line is {horizontal_threshold_label}; "
            f"the dotted vertical line is {effect_threshold_label}."
        )
        blocks = (
            _wrapped_block("Panel legend", legend),
            _wrapped_block("The purpose of the panel", narrative.purpose),
            _wrapped_block("What the panel shows", narrative.shows),
            _wrapped_block("The implications of the panel's data", narrative.implications),
        )
        y = 0.445
        for block in blocks:
            pdf_figure.text(
                0.06, y, block, ha="left", va="top", fontsize=7.3,
                linespacing=1.15,
            )
            line_count = block.count("\n") + 1
            line_height = (7.3 / 72.0) / style.pdf_height * 1.20
            y -= line_count * line_height + 0.014
        save_figure(
            pdf_figure,
            paths["pdf"],
            fmt="pdf",
            close=True,
            save_mode="print",
            announce_colours=False,
            facecolor=LABEL_GROUND_PRINT,
        )

    linked_points = 0
    if gene_label_column is not None or gene_url_column is not None:
        if not gene_label_column or not gene_url_column:
            raise ValueError("gene label and URL columns must be supplied together")
        linked_points = _add_pdf_point_links(
            paths["pdf"], data,
            x_limits=x_limits, y_limits=y_limits,
            x_column="plot_x", y_column="plot_y",
            gene_label_column=gene_label_column,
            gene_url_column=gene_url_column,
            style=style,
        )

    stats = {
        "panel_id": panel_id,
        "plotted_points": int(len(data)),
        "point_size": float(style.point_size),
        "point_alpha": float(style.point_alpha),
        "marker_edge_color": "none",
        "line_color": style.line_color,
        "line_width_points": float(style.line_width),
        "font_family": OPEN_SANS_FAMILY,
        "linked_points": int(linked_points),
        "legend_title": "LOPIT/TAGM",
        "legend_categories": "|".join(
            category
            for category in LOPIT_ORDER
            if data[lopit_column].astype(str).eq(category).any()
        ),
        "labeled_points": int(data["label_above_threshold"].sum()),
        "point_label_rule": "effect > vertical cutoff and plotted y > horizontal cutoff",
        "plot_region": "square",
        "x_min": float(x_limits[0]),
        "x_max": float(x_limits[1]),
        "y_min": float(y_limits[0]),
        "y_max": float(y_limits[1]),
        "horizontal_threshold": float(horizontal_threshold),
        "horizontal_threshold_label": horizontal_threshold_label,
        "effect_threshold": float(effect_threshold),
        "effect_threshold_label": effect_threshold_label,
        **dict(statistics or {}),
    }
    from .tabular import write_table

    write_table(
        pd.DataFrame(
            [{"metric": key, "value": value} for key, value in stats.items()]
        ),
        paths["stats"],
    )
    write_table(data, paths["data"])

    actual = {path.name for path in destination.iterdir() if path.is_file()}
    expected = {path.name for path in paths.values()}
    if actual != expected:
        raise RuntimeError(
            f"Panel folder must contain exactly four files; expected "
            f"{sorted(expected)}, found {sorted(actual)}"
        )
    return paths


def _load_panel_manifest(
    manifest: Mapping[str, object] | str | Path,
) -> dict[str, object]:
    """Load one explicit figure manifest without inferring from filenames."""
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not load panel manifest {path}: {error}") from error
    else:
        loaded = dict(manifest)
    if not isinstance(loaded, dict):
        raise ValueError("Panel manifest must be a JSON object")
    panels = loaded.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("Panel manifest needs a non-empty 'panels' list")
    figure_id = str(loaded.get("figure_id") or "").strip()
    if not figure_id or Path(figure_id).name != figure_id:
        raise ValueError("Panel manifest needs a filename-safe 'figure_id'")
    columns = loaded.get("columns", 2)
    if not isinstance(columns, int) or isinstance(columns, bool) or columns < 1:
        raise ValueError("Panel manifest 'columns' must be a positive integer")
    return {**loaded, "figure_id": figure_id, "columns": columns}


def _resolve_run_artifacts(
    artifacts: Mapping[str, Mapping[str, object]],
    sources: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Resolve caller-named artifacts and retain their declared identity."""
    resolved: dict[str, dict[str, object]] = {}
    for source in dict.fromkeys(sources):
        if source not in artifacts:
            raise ValueError(f"Manifest source {source!r} is absent from run artifacts")
        specification = artifacts[source]
        if not isinstance(specification, Mapping):
            raise ValueError(f"Run artifact {source!r} must be a mapping")
        level = str(specification.get("level") or "").strip().lower()
        phenotype = str(specification.get("phenotype") or "").strip()
        if level not in {"grna", "gene"}:
            raise ValueError(
                f"Run artifact {source!r} declares invalid level {level!r}"
            )
        if not phenotype:
            raise ValueError(f"Run artifact {source!r} needs a phenotype")
        has_data = "data" in specification
        has_path = "path" in specification
        if has_data == has_path:
            raise ValueError(
                f"Run artifact {source!r} must supply exactly one of data or path"
            )
        if has_data:
            data = specification["data"]
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Run artifact {source!r} data must be a DataFrame")
            frame = data.copy()
        else:
            path = Path(str(specification["path"]))
            if not path.is_file():
                raise ValueError(f"Run artifact {source!r} path does not exist: {path}")
            try:
                frame = pd.read_csv(path)
            except Exception as error:  # noqa: BLE001 - report the exact source
                raise ValueError(
                    f"Could not read run artifact {source!r} from {path}: {error}"
                ) from error
        resolved[source] = {
            "level": level,
            "phenotype": phenotype,
            "data": frame,
        }
    return resolved


def _manifest_narrative(panel: Mapping[str, object]) -> PanelNarrative:
    raw = panel.get("narrative")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Panel {panel.get('panel_id')!r} needs a narrative")
    values = {}
    for name in ("legend", "purpose", "shows", "implications"):
        value = str(raw.get(name) or "").strip()
        if not value:
            raise ValueError(
                f"Panel {panel.get('panel_id')!r} narrative lacks {name!r}"
            )
        values[name] = value
    return PanelNarrative(**values)


def _required_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    panel_id: str,
) -> None:
    missing = sorted({name for name in columns if name not in frame.columns})
    if missing:
        raise ValueError(f"Panel {panel_id!r} source lacks columns {missing}")


def _finite_column(frame: pd.DataFrame, column: str, *, panel_id: str) -> None:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"Panel {panel_id!r} column {column!r} must be finite")


def _panel_file_paths(destination: Path, panel_id: str) -> dict[str, Path]:
    stem = destination / panel_id
    return {
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "stats": destination / f"{panel_id}_stats.csv",
        "data": destination / f"{panel_id}_data.csv",
    }


def _draw_box_jitter(
    axis,
    data: pd.DataFrame,
    *,
    categories: Sequence[str],
    x_label: str,
    y_label: str,
    style: PanelStyle,
) -> None:
    grouped = [
        data.loc[data["plot_category"].eq(category), "plot_y"].to_numpy(float)
        for category in categories
    ]
    box = axis.boxplot(
        grouped,
        positions=np.arange(len(categories), dtype=float),
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": style.line_color, "linewidth": style.line_width},
        whiskerprops={"color": style.line_color, "linewidth": style.line_width},
        capprops={"color": style.line_color, "linewidth": style.line_width},
        boxprops={"color": style.line_color, "linewidth": style.line_width},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#56B4E9")
        patch.set_alpha(style.point_alpha)
    axis.scatter(
        data["plot_x"], data["plot_y"], s=style.point_size,
        color="#0072B2", alpha=style.point_alpha, edgecolors="none",
        linewidths=0, rasterized=False, zorder=3,
    )
    axis.set_xlim(float(data["plot_x_min"].iloc[0]),
                  float(data["plot_x_max"].iloc[0]))
    axis.set_ylim(float(data["plot_y_min"].iloc[0]),
                  float(data["plot_y_max"].iloc[0]))
    axis.set_xticks(np.arange(len(categories), dtype=float), categories)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_color(style.line_color)
        axis.spines[side].set_linewidth(style.line_width)
    axis.tick_params(
        axis="both", which="both", color=style.line_color,
        labelcolor=INK_PRINT, width=style.line_width,
    )


def write_box_jitter_package(
    results: pd.DataFrame,
    destination: str | Path,
    *,
    panel_id: str,
    category_column: str,
    value_column: str,
    category_order: Sequence[str],
    x_label: str,
    y_label: str,
    narrative: PanelNarrative,
    gene_label_column: str | None = None,
    gene_url_column: str | None = None,
    statistics: Mapping[str, object] | None = None,
    style: PanelStyle = DEFAULT_PANEL_STYLE,
) -> dict[str, Path]:
    """Write one deterministic box-and-jitter four-file panel package."""
    if not (0 < style.point_alpha <= 1):
        raise ValueError("point_alpha must be in (0, 1]")
    categories = [str(value) for value in category_order]
    if not categories or len(categories) != len(set(categories)):
        raise ValueError("category_order must contain distinct categories")
    observed = set(results[category_column].astype(str))
    if observed != set(categories):
        raise ValueError(
            "category_order must name every observed category exactly; "
            f"observed {sorted(observed)}, declared {categories}"
        )
    values = pd.to_numeric(results[value_column], errors="raise")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("Box-and-jitter values must be finite")
    if gene_label_column is not None or gene_url_column is not None:
        if not gene_label_column or not gene_url_column:
            raise ValueError("gene label and URL columns must be supplied together")

    data = results.copy()
    data["plot_category"] = data[category_column].astype(str)
    data["plot_y"] = values
    data["plot_x"] = np.nan
    for position, category in enumerate(categories):
        indexes = data.index[data["plot_category"].eq(category)]
        offsets = (
            np.array([0.0])
            if len(indexes) == 1
            else np.linspace(-0.18, 0.18, len(indexes))
        )
        data.loc[indexes, "plot_x"] = float(position) + offsets
    y_min = float(values.min())
    y_max = float(values.max())
    span = max(y_max - y_min, 1e-9)
    y_limits = (y_min - 0.08 * span, y_max + 0.08 * span)
    x_limits = (-0.5, float(len(categories)) - 0.5)
    data["plot_x_min"] = x_limits[0]
    data["plot_x_max"] = x_limits[1]
    data["plot_y_min"] = y_limits[0]
    data["plot_y_max"] = y_limits[1]
    data["box_alpha"] = float(style.point_alpha)
    data["point_alpha"] = float(style.point_alpha)
    data["point_size"] = float(style.point_size)
    data["marker_edge_color"] = "none"

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = _panel_file_paths(destination, panel_id)
    rc = {
        "font.family": OPEN_SANS_FAMILY,
        "font.size": 9.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
    with plt.rc_context(rc):
        from .plot import save_figure

        png_figure = plt.figure(
            figsize=(style.figure_width, style.plot_height),
            facecolor=LABEL_GROUND_PRINT,
        )
        png_axis = png_figure.add_axes([style.axes_left, 0.16, 0.78, 0.82])
        _draw_box_jitter(
            png_axis, data, categories=categories, x_label=x_label,
            y_label=y_label, style=style,
        )
        save_figure(
            png_figure, paths["png"], fmt="png", dpi=style.png_dpi,
            close=True, save_mode="print", announce_colours=False,
            bbox_inches="tight", facecolor=LABEL_GROUND_PRINT,
        )

        pdf_figure = plt.figure(
            figsize=(style.figure_width, style.pdf_height),
            facecolor=LABEL_GROUND_PRINT,
        )
        pdf_axis = pdf_figure.add_axes(
            [style.axes_left, style.pdf_axes_bottom, style.axes_width,
             style.pdf_axes_height]
        )
        _draw_box_jitter(
            pdf_axis, data, categories=categories, x_label=x_label,
            y_label=y_label, style=style,
        )
        blocks = (
            _wrapped_block("Panel legend", narrative.legend),
            _wrapped_block("The purpose of the panel", narrative.purpose),
            _wrapped_block("What the panel shows", narrative.shows),
            _wrapped_block(
                "The implications of the panel's data", narrative.implications
            ),
        )
        y = 0.445
        for block in blocks:
            pdf_figure.text(0.06, y, block, ha="left", va="top", fontsize=7.3,
                            linespacing=1.15)
            line_count = block.count("\n") + 1
            line_height = (7.3 / 72.0) / style.pdf_height * 1.20
            y -= line_count * line_height + 0.014
        save_figure(
            pdf_figure, paths["pdf"], fmt="pdf", close=True,
            save_mode="print", announce_colours=False,
            facecolor=LABEL_GROUND_PRINT,
        )

    linked_points = 0
    if gene_label_column and gene_url_column:
        linked_points = _add_pdf_point_links(
            paths["pdf"], data, x_limits=x_limits, y_limits=y_limits,
            x_column="plot_x", y_column="plot_y",
            gene_label_column=gene_label_column,
            gene_url_column=gene_url_column, style=style,
        )
    stats = {
        "panel_id": panel_id,
        "plot_kind": "box_jitter",
        "plotted_points": int(len(data)),
        "box_alpha": float(style.point_alpha),
        "point_alpha": float(style.point_alpha),
        "point_size": float(style.point_size),
        "marker_edge_color": "none",
        "linked_points": int(linked_points),
        "category_order": "|".join(categories),
        "x_min": x_limits[0], "x_max": x_limits[1],
        "y_min": y_limits[0], "y_max": y_limits[1],
        **dict(statistics or {}),
    }
    from .tabular import write_table

    write_table(
        pd.DataFrame([{"metric": key, "value": value}
                      for key, value in stats.items()]),
        paths["stats"],
    )
    write_table(data, paths["data"])
    actual = {path.name for path in destination.iterdir() if path.is_file()}
    expected = {path.name for path in paths.values()}
    if actual != expected:
        raise RuntimeError(
            f"Panel folder must contain exactly four files; expected "
            f"{sorted(expected)}, found {sorted(actual)}"
        )
    return paths


def _copy_transformed_links(
    writer: PdfWriter,
    annotations: Sequence[object],
    *,
    scale: float,
    translate_x: float,
    translate_y: float,
) -> int:
    copied = 0
    for reference in annotations:
        annotation = reference.get_object()
        action = annotation.get("/A")
        uri = action.get("/URI") if action is not None else None
        rect = annotation.get("/Rect")
        if annotation.get("/Subtype") != "/Link" or not uri or rect is None:
            continue
        left, bottom, right, top = map(float, rect)
        transformed = (
            translate_x + scale * left,
            translate_y + scale * bottom,
            translate_x + scale * right,
            translate_y + scale * top,
        )
        link = Link(rect=transformed, url=str(uri))
        for key in ("/Contents", "/T"):
            if annotation.get(key) is not None:
                link[NameObject(key)] = TextStringObject(str(annotation[key]))
        writer.add_annotation(0, link)
        copied += 1
    return copied


def compose_vector_figure(
    panel_pdfs: Sequence[str | Path],
    destination: str | Path,
    *,
    columns: int = 2,
) -> Path:
    """Compose one vector page and transform every URI link rectangle."""
    if not panel_pdfs:
        raise ValueError("At least one panel PDF is required")
    if columns < 1:
        raise ValueError("columns must be positive")
    readers = [PdfReader(Path(path)) for path in panel_pdfs]
    pages = []
    for path, reader in zip(panel_pdfs, readers):
        if len(reader.pages) != 1:
            raise ValueError(f"Panel PDF must have one page: {path}")
        pages.append(reader.pages[0])
    widths = [float(page.mediabox.width) for page in pages]
    heights = [float(page.mediabox.height) for page in pages]
    tile_width = max(widths)
    tile_height = max(heights)
    rows = int(math.ceil(len(pages) / columns))
    writer = PdfWriter()
    canvas = writer.add_blank_page(
        width=tile_width * columns,
        height=tile_height * rows,
    )
    for index, page in enumerate(pages):
        width = widths[index]
        height = heights[index]
        scale = min(tile_width / width, tile_height / height)
        column = index % columns
        row = index // columns
        translate_x = column * tile_width + (tile_width - width * scale) / 2
        translate_y = (
            (rows - row - 1) * tile_height + (tile_height - height * scale) / 2
        )
        annotations = list(page.get("/Annots", []))
        page.pop(NameObject("/Annots"), None)
        transform = Transformation().scale(scale).translate(
            translate_x, translate_y
        )
        canvas.merge_transformed_page(page, transform, over=True, expand=False)
        _copy_transformed_links(
            writer, annotations, scale=scale,
            translate_x=translate_x, translate_y=translate_y,
        )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as handle:
        writer.write(handle)
    temporary.replace(destination)
    return destination


def build_manifest_packages(
    manifest: Mapping[str, object] | str | Path,
    artifacts: Mapping[str, Mapping[str, object]],
    destination: str | Path,
    *,
    style: PanelStyle = DEFAULT_PANEL_STYLE,
) -> dict[str, Any]:
    """Validate a declared run manifest, then write every panel and figure."""
    loaded = _load_panel_manifest(manifest)
    raw_panels = loaded["panels"]
    panels: list[dict[str, object]] = []
    panel_ids: set[str] = set()
    sources: list[str] = []
    for index, raw in enumerate(raw_panels):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Panel manifest entry {index} must be a mapping")
        panel = dict(raw)
        panel_id = str(panel.get("panel_id") or "").strip()
        if (not panel_id or Path(panel_id).name != panel_id
                or panel_id in panel_ids):
            raise ValueError(f"Panel entry {index} has an invalid/duplicate panel_id")
        panel_ids.add(panel_id)
        source = str(panel.get("source") or "").strip()
        phenotype = str(panel.get("phenotype") or "").strip()
        level = str(panel.get("level") or "").strip().lower()
        kind = str(panel.get("kind") or "").strip().lower()
        if not source or not phenotype or level not in {"grna", "gene"}:
            raise ValueError(
                f"Panel {panel_id!r} needs exact source, phenotype and level"
            )
        if kind not in {"scatter", "box_jitter"}:
            raise ValueError(f"Panel {panel_id!r} has unsupported kind {kind!r}")
        panel.update({"panel_id": panel_id, "source": source,
                      "phenotype": phenotype, "level": level, "kind": kind,
                      "narrative_object": _manifest_narrative(panel)})
        panels.append(panel)
        sources.append(source)

    resolved = _resolve_run_artifacts(artifacts, sources)
    for panel in panels:
        source = str(panel["source"])
        artifact = resolved[source]
        if panel["level"] != artifact["level"]:
            raise ValueError(
                f"Panel {panel['panel_id']!r} declares level {panel['level']!r} "
                f"but source {source!r} declares level {artifact['level']!r}"
            )
        if panel["phenotype"] != artifact["phenotype"]:
            raise ValueError(
                f"Panel {panel['panel_id']!r} declares phenotype "
                f"{panel['phenotype']!r} but source {source!r} declares "
                f"phenotype {artifact['phenotype']!r}"
            )
        frame = artifact["data"].copy()
        common = [str(panel.get(name) or "") for name in (
            "effect_column", "bh_column", "gene_label_column", "gene_url_column"
        )]
        _required_columns(frame, common, panel_id=str(panel["panel_id"]))
        _finite_column(frame, common[0], panel_id=str(panel["panel_id"]))
        if frame[common[2]].astype(str).str.strip().eq("").any() \
                or frame[common[3]].astype(str).str.strip().eq("").any():
            raise ValueError(
                f"Panel {panel['panel_id']!r} needs a label and URL for every row"
            )
        if panel["kind"] == "scatter":
            scatter = [str(panel.get(name) or "") for name in (
                "y_column", "lopit_column", "x_label", "y_label",
                "horizontal_threshold_label", "limit_group"
            )]
            if not all(scatter):
                raise ValueError(
                    f"Scatter panel {panel['panel_id']!r} lacks plot/limit metadata"
                )
            _required_columns(frame, scatter[:2], panel_id=str(panel["panel_id"]))
            _finite_column(frame, scatter[0], panel_id=str(panel["panel_id"]))
            unknown = sorted(set(frame[scatter[1]].astype(str)) - set(LOPIT_COLOURS))
            if unknown:
                raise ValueError(
                    f"Panel {panel['panel_id']!r} LOPIT categories lack colours: "
                    + ", ".join(unknown)
                )
            try:
                float(panel["horizontal_threshold"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Scatter panel {panel['panel_id']!r} needs a numeric "
                    "horizontal_threshold"
                ) from error
        else:
            box = [str(panel.get(name) or "") for name in (
                "category_column", "value_column", "x_label", "y_label"
            )]
            order = panel.get("category_order")
            if not all(box) or not isinstance(order, list) or not order:
                raise ValueError(
                    f"Box panel {panel['panel_id']!r} lacks category/value metadata"
                )
            _required_columns(frame, box[:2], panel_id=str(panel["panel_id"]))
            _finite_column(frame, box[1], panel_id=str(panel["panel_id"]))
        panel["frame"] = frame

    thresholds: dict[str, tuple[float, dict[str, float | int]]] = {}
    for phenotype in sorted({str(panel["phenotype"]) for panel in panels}):
        guides = [panel for panel in panels
                  if panel["phenotype"] == phenotype and panel["level"] == "grna"]
        owners = {(str(panel["source"]), str(panel["effect_column"]))
                  for panel in guides}
        if len(owners) != 1:
            raise ValueError(
                f"Phenotype {phenotype!r} needs exactly one declared gRNA "
                "source/effect column for its control threshold"
            )
        source, effect_column = next(iter(owners))
        thresholds[phenotype] = guide_control_threshold(
            resolved[source]["data"], effect_column=effect_column,
        )

    for panel in panels:
        threshold, audit = thresholds[str(panel["phenotype"])]
        panel["threshold"] = threshold
        panel["threshold_audit"] = audit
        panel["frame"] = apply_primary_call(
            panel["frame"], effect_column=str(panel["effect_column"]),
            bh_column=str(panel["bh_column"]), effect_threshold=threshold,
        )

    limits: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for group in sorted({str(panel["limit_group"]) for panel in panels
                         if panel["kind"] == "scatter"}):
        members = [panel for panel in panels
                   if panel.get("limit_group") == group]
        normalised = []
        for panel in members:
            frame = panel["frame"].copy()
            frame["manifest_plot_x"] = pd.to_numeric(
                frame[str(panel["effect_column"])], errors="raise"
            )
            frame["manifest_plot_y"] = pd.to_numeric(
                frame[str(panel["y_column"])], errors="raise"
            )
            normalised.append(frame)
        limits[group] = shared_limits(
            normalised, x_column="manifest_plot_x", y_column="manifest_plot_y"
        )

    destination = Path(destination)
    for panel in panels:
        folder = destination / str(panel["panel_id"])
        expected = {path.name for path in _panel_file_paths(
            folder, str(panel["panel_id"])).values()}
        if folder.exists():
            unexpected = {path.name for path in folder.iterdir()} - expected
            if unexpected:
                raise ValueError(
                    f"Panel folder {folder} contains unmanifested entries: "
                    f"{sorted(unexpected)}"
                )

    written: dict[str, dict[str, Path]] = {}
    for panel in panels:
        panel_id = str(panel["panel_id"])
        frame = panel["frame"]
        threshold = float(panel["threshold"])
        statistics = {
            **dict(panel["threshold_audit"]),
            "source": panel["source"],
            "phenotype": panel["phenotype"],
            "level": panel["level"],
            "primary_calls": int(frame["primary_call"].sum()),
        }
        folder = destination / panel_id
        if panel["kind"] == "scatter":
            x_limits, y_limits = limits[str(panel["limit_group"])]
            written[panel_id] = write_panel_package(
                frame, folder, panel_id=panel_id,
                x_column=str(panel["effect_column"]),
                y_column=str(panel["y_column"]),
                lopit_column=str(panel["lopit_column"]),
                x_label=str(panel["x_label"]), y_label=str(panel["y_label"]),
                x_limits=x_limits, y_limits=y_limits,
                horizontal_threshold=float(panel["horizontal_threshold"]),
                horizontal_threshold_label=str(
                    panel["horizontal_threshold_label"]),
                effect_threshold=threshold,
                effect_threshold_label="gRNA NT mean + 3 sample SD",
                narrative=panel["narrative_object"],
                gene_label_column=str(panel["gene_label_column"]),
                gene_url_column=str(panel["gene_url_column"]),
                point_label_column=str(panel["gene_label_column"]),
                statistics=statistics, style=style,
            )
        else:
            written[panel_id] = write_box_jitter_package(
                frame, folder, panel_id=panel_id,
                category_column=str(panel["category_column"]),
                value_column=str(panel["value_column"]),
                category_order=panel["category_order"],
                x_label=str(panel["x_label"]), y_label=str(panel["y_label"]),
                narrative=panel["narrative_object"],
                gene_label_column=str(panel["gene_label_column"]),
                gene_url_column=str(panel["gene_url_column"]),
                statistics={**statistics, "effect_threshold": threshold},
                style=style,
            )
    figure_pdf = compose_vector_figure(
        [written[str(panel["panel_id"])]["pdf"] for panel in panels],
        destination / f"{loaded['figure_id']}.pdf",
        columns=int(loaded["columns"]),
    )
    return {"panels": written, "figure_pdf": figure_pdf}


__all__ = [
    "LOPIT_COLOURS",
    "LOPIT_ORDER",
    "OPEN_SANS_BOLD",
    "OPEN_SANS_FAMILY",
    "OPEN_SANS_REGULAR",
    "PanelNarrative",
    "PanelStyle",
    "apply_primary_call",
    "build_manifest_packages",
    "compose_vector_figure",
    "guide_control_threshold",
    "shared_limits",
    "write_box_jitter_package",
    "write_panel_package",
]
