"""Publication panel packages for regression and permutation results.

One call writes exactly four files for one manuscript panel: a vector PDF
with its narrative legend, a plot-only PNG, a point-level data CSV, and a
key-value statistics CSV.  The numerical result table remains the source of
truth; this module only applies a declared call rule and renders it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Link
from adjustText import adjust_text
from pypdf.generic import (
    NameObject,
    TextStringObject,
)

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
    line_color: str = "#000000"
    figure_width: float = 7.2
    plot_height: float = 5.443902439
    pdf_height: float = 8.8
    png_dpi: int = 400
    axes_left: float = 0.10
    axes_width: float = 0.62
    pdf_axes_bottom: float = 0.48
    pdf_axes_height: float = 0.507272727


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
        labelcolor="#000000", width=style.line_width,
    )
    axis.xaxis.label.set_color("#000000")
    axis.yaxis.label.set_color("#000000")
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
                color="#000000",
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
    style: PanelStyle = PanelStyle(),
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
        png_figure = plt.figure(
            figsize=(style.figure_width, style.plot_height), facecolor="white"
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
        png_figure.savefig(
            paths["png"], dpi=style.png_dpi, bbox_inches="tight",
            facecolor="white",
        )
        plt.close(png_figure)

        pdf_figure = plt.figure(
            figsize=(style.figure_width, style.pdf_height), facecolor="white"
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
        pdf_figure.savefig(paths["pdf"], format="pdf", facecolor="white")
        plt.close(pdf_figure)

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


__all__ = [
    "LOPIT_COLOURS",
    "LOPIT_ORDER",
    "OPEN_SANS_BOLD",
    "OPEN_SANS_FAMILY",
    "OPEN_SANS_REGULAR",
    "PanelNarrative",
    "PanelStyle",
    "apply_primary_call",
    "guide_control_threshold",
    "shared_limits",
    "write_panel_package",
]
