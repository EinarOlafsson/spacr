#!/usr/bin/env python3
"""Refresh the alpha-feature slide from the current registry and reviewed labels.

Run ``python tools/refresh_alpha_deck.py`` after rebuilding the presentation.
Only the alpha page, its JPEG and its thumbnail are replaced. The PDF keeps
vector text; all other pages and the viewer's numbering are preserved.
Requires PySide6, Pillow, pypdf and pdftoppm (the normal deck render tool).
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = "Alpha features: switch them on in Preferences"
# Each registry entry appears once. The two columns balance complete categories.
GROUPS = (
    ("Segmentation", (
        (551, "StarDist segmentation backend"),
        (552, "InstanSeg segmentation backend"),
        (553, "Omnipose segmentation backend"),
        (405, "SAMCell segmentation backend"),
        (554, "Spotiflow spot detection"),
        (475, "SpotNet spot detection"),
        (555, "Click- and box-prompted segmentation (micro-SAM)"),
        (568, "Segmentation uncertainty maps"),
        (578, "Parameter-robustness report"),
        (546, "Run CellProfiler .cppipe pipelines"),
        (545, "Masks and ROIs in and out: QuPath, ImageJ, COCO"),
        (558, "Virtual staining from brightfield"),
        (493, "Mask batches across several GPUs"),
        (508, "Use Make Masks settings in Mask Generation"),
    )),
    ("Image preprocessing", (
        (538, "Spectral unmixing (bleed-through correction)"),
        (539, "Photobleaching correction"),
        (543, "Vendor flat-field profiles (Harmony, ZEN)"),
        (580, "Intensity calibration across sessions"),
        (559, "Deep-learning image QC: focus, debris, empty fields"),
        (557, "Self-supervised denoising (Noise2Void)"),
        (591, "Illumination, enhancement and PSF settings"),
    )),
    ("Measurements and\nassays", (
        (535, "Cell-cycle phase classification"),
        (540, "Live/dead viability and cytotoxicity index"),
        (541, "Confluency for any channel"),
        (536, "Scratch / wound-closure assay"),
        (542, "Colony / CFU counting"),
        (547, "Image-based profiling: normalise, select, mAP"),
        (566, "GPU measurement (cuCIM, CuPy)"),
        (501, "Estimate plaque scale and growth time"),
    )),
    ("Time-lapse", (
        (556, "SAM2 propagation: segment once, follow each object"),
        (426, "Timeflows: a temporal Cellpose"),
        (537, "Lineage trees from tracks"),
        (567, "Event detection: mitosis, egress, invasion, death"),
        (571, "Time-to-event: Kaplan-Meier and Cox"),
    )),
    ("Cells and machine\nlearning", (
        (560, "Foundation-model embeddings (SubCell, OpenPhenom, …)"),
        (562, "Well labels to responding cells (multiple-instance)"),
        (563, "Anomaly detection against negative controls"),
        (565, "Find cells like this one (similarity search)"),
        (544, "Blind scoring"),
        (564, "Counterfactuals: morph a control cell toward a hit"),
        (561, "Self-supervised DINO pretraining"),
        (470, "Judge every label from cross-channel ground truth"),
    )),
    ("Screens and statistics", (
        (570, "Arrayed-screen hits: SSMD, robust z, B-score"),
        (584, "Chemistry-aware hits: SMILES, clusters, SAR"),
        (585, "Sample-size planner for arrayed assays"),
        (534, "Spatial transcriptomics: Visium and Xenium"),
        (583, "LIMS / plate-barcode linkage"),
    )),
    ("Automation and data", (
        (548, "Watch a folder, analyse images as they arrive"),
        (549, "Smart microscopy: send hits back for re-imaging"),
        (550, "Cloud storage (S3 and others)"),
        (575, "Export a run as a Nextflow / Snakemake workflow"),
        (577, "Run-finished notifications (email, Slack, ntfy)"),
        (581, "Export for R (SingleCellExperiment), tidy Parquet"),
        (582, "Plugin and recipe catalogue"),
        (576, "Postgres or DuckDB / Parquet measurement store"),
        (633, "Test data on seven more modules"),
    )),
    ("Reproducibility", (
        (572, "Figure-integrity check"),
        (573, "Pre-registered analysis lock"),
        (574, "Metadata and archive packages (REMBI, IDR)"),
        (579, "One-click Zenodo archive with a DOI"),
    )),
    ("Organisms", (
        (634, "Trypanosoma, Leishmania, Giardia, virus, mammalian pages"),
    )),
)
# Filed future alpha items that are listed before they enter the registry.
PLANNED = frozenset({634})


def _registry_ids(source):
    # Read the registry without importing scientific libraries or evaluating it.
    # ALPHA_SPECIES (the organism pages behind Show alpha species) counts too.
    found = {}
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("ALPHA_FEATURES", "ALPHA_SPECIES"):
                    found[target.id] = {ast.literal_eval(key) for key in node.value.keys}
    if "ALPHA_FEATURES" not in found:
        raise ValueError("ALPHA_FEATURES registry not found")
    return found["ALPHA_FEATURES"] | found.get("ALPHA_SPECIES", set())


def _check_registry(source):
    registered = _registry_ids(source)
    listed = [number for _, rows in GROUPS for number, _ in rows
              if number not in PLANNED or number in registered]
    if len(listed) != len(set(listed)) or set(listed) != registered:
        raise ValueError(
            f"Update alpha slide labels: missing {sorted(registered - set(listed))}; "
            f"retired {sorted(set(listed) - registered)}; duplicates are forbidden"
        )
    return len(listed)


def _draw_pdf(target, count, page_size):
    # Qt emits vector text with embedded fonts. No application window is opened.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QMarginsF, QRectF, QSizeF, Qt
    from PySide6.QtGui import (
        QColor,
        QFont,
        QFontDatabase,
        QFontMetricsF,
        QGuiApplication,
        QPageSize,
        QPainter,
        QPdfWriter,
        QPen,
    )

    app = QGuiApplication.instance() or QGuiApplication([])
    font_dir = ROOT / "spacr/resources/font/open_sans/static"
    for name in ("Regular", "SemiBold", "LightItalic"):
        QFontDatabase.addApplicationFont(str(font_dir / f"OpenSans-{name}.ttf"))
    pdf = QPdfWriter(str(target))
    pdf.setResolution(240)
    pdf.setPageSize(QPageSize(QSizeF(*page_size), QPageSize.Unit.Point))
    pdf.setPageMargins(QMarginsF(0, 0, 0, 0))
    painter = QPainter(pdf)
    painter.setWindow(0, 0, 3200, 1800)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.fillRect(0, 0, 3200, 1800, QColor("#0e1116"))

    def label(text, x, y, w, h, size=29, color="#e3e8ed", bold=False):
        font = QFont("Open Sans")
        font.setPixelSize(size)
        font.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
        # Refuse to publish truncated labels when a registry label changes.
        metrics = QFontMetricsF(font, pdf)
        if any(metrics.horizontalAdvance(line) > w for line in text.splitlines()):
            raise ValueError(f"Alpha slide label does not fit: {text}")
        painter.setFont(font)
        painter.setPen(QColor(color))
        painter.drawText(QRectF(x, y, w, h), Qt.AlignmentFlag.AlignVCenter, text)

    try:
        label("I N T R O D U C T I O N", 120, 82, 1100, 80, 36, "#29bfb3")
        label(TITLE, 120, 160, 2650, 130, 94)
        painter.setPen(QPen(QColor("#d63fa1"), 3))
        painter.setBrush(QColor("#222932"))
        painter.drawRoundedRect(QRectF(2850, 118, 230, 76), 38, 38)
        label("ALPHA", 2910, 123, 160, 64, 32, "#d63fa1", True)
        row_height = 43
        for x, groups in ((120, GROUPS[:3]), (1640, GROUPS[3:])):
            painter.fillRect(QRectF(x, 310, 1440, 65), QColor("#222932"))
            label("Category", x + 14, 310, 345, 65, 34, bold=True)
            label("Alpha feature", x + 388, 310, 1035, 65, 34, bold=True)
            y = 375
            for category, rows in groups:
                painter.fillRect(QRectF(x, y, 371, len(rows) * row_height - 3),
                                 QColor("#191f27"))
                label(category, x + 14, y, 345, len(rows) * row_height - 3)
                for _, caption in rows:
                    painter.fillRect(QRectF(x + 374, y, 1066, row_height - 3),
                                     QColor("#191f27"))
                    label(caption, x + 388, y, 1038, row_height - 3)
                    y += row_height
        planned = len(PLANNED - set(_registry_ids(ROOT / "spacr/settings.py")))
        label(f"{count} alpha features{f' and {planned} planned' if planned else ''}, hidden until "
              "Preferences → Modules → Show alpha features is on. Early versions: expect changes.", 120, 1700, 2960, 62, 30, "#acb5be")
    finally:
        painter.end()
    # Keep the application alive until the paint device is finalized.
    assert app is not None


def refresh(folder, settings_source=ROOT / "spacr/settings.py"):
    from PIL import Image
    from pypdf import PdfReader, PdfWriter

    count = _check_registry(settings_source)
    manifest = json.loads((folder / "slides.json").read_text(encoding="utf-8"))
    matches = [i for i, slide in enumerate(manifest["slides"])
               if slide["title"] == TITLE]
    if len(matches) != 1:
        raise ValueError("The deck must have exactly one alpha-feature slide")
    index = matches[0]
    pdf_path = folder / "spacr_deck.pdf"
    reader = PdfReader(io.BytesIO(pdf_path.read_bytes()))
    if len(reader.pages) != manifest["count"] or len(manifest["slides"]) != manifest["count"]:
        raise ValueError("PDF and slide manifest page counts disagree")
    slide = manifest["slides"][index]
    image_path, thumb_path = (folder / slide[key] for key in ("image", "thumb"))
    for path in (image_path, thumb_path):
        if not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Slide assets must exist inside the deck folder")
    size = tuple(float(value) for value in (
        reader.pages[index].mediabox.width, reader.pages[index].mediabox.height))
    with tempfile.TemporaryDirectory(prefix=".alpha-deck-", dir=folder) as tmp:
        stage = Path(tmp)
        page_pdf = stage / "alpha.pdf"
        _draw_pdf(page_pdf, count, size)
        subprocess.run(["pdftoppm", "-singlefile", "-png", "-scale-to-x", "3200",
                        "-scale-to-y", "1800", str(page_pdf), str(stage / "alpha")],
                       check=True, capture_output=True, timeout=120)
        with Image.open(stage / "alpha.png") as picture:
            picture.convert("RGB").save(stage / "slide.jpg", quality=86,
                                         optimize=True, subsampling=0, progressive=True)
            picture.resize((320, 1800 * 320 // 3200), Image.Resampling.LANCZOS).convert(
                "RGB").save(stage / "thumb.jpg", quality=75, optimize=True)
        writer = PdfWriter()
        replacement = PdfReader(page_pdf)
        for number, page in enumerate(reader.pages):
            writer.add_page(replacement.pages[0] if number == index else page)
        if reader.metadata:
            writer.add_metadata({str(k): str(v) for k, v in reader.metadata.items()})
        writer.write(stage / "deck.pdf")
        # All validation/rendering finishes before replacing a published artifact.
        for source, target in (("deck.pdf", pdf_path), ("slide.jpg", image_path),
                               ("thumb.jpg", thumb_path)):
            os.replace(stage / source, target)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/source/_static/deck")
    args = parser.parse_args()
    print(f"Updated alpha slide: {refresh(args.out)} registered features")
