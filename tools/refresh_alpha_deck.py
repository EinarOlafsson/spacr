#!/usr/bin/env python3
"""Refresh the alpha-feature slide from the current registry and grouped lines.

Run ``python tools/refresh_alpha_deck.py`` after rebuilding the presentation.
Only the alpha page, its JPEG and its thumbnail are replaced, and the slide is
retitled ``PUBLISHED_TITLE``. The PDF keeps vector text; all other pages and
the viewer's numbering are preserved. A fresh build from ``tools/readme_deck``
names the slide ``TITLE``.
Requires PySide6, Pillow, pypdf and pdftoppm (the normal deck render tool).

The slide lists lines, not single registry items (2026-11-15: 145 rows were
too small to read). ``GROUPS`` holds ``(category, lines)`` and each line is
``(text, item numbers)``. Every ALPHA_FEATURES item number is in exactly one
line. To add a new alpha item, make one edit: append its number to the line it
extends, or add a ``(text, (number,))`` line to its category. Keep the text
truthful to every item on the line and short enough for one row; a line that
does not fit stops the refresh. The ALPHA_SPECIES item ``SPECIES_ITEM`` is the
species strip, not a line.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The alpha slide of a fresh build (tools/readme_deck/scripts/build.js).
TITLE = "Alpha features: switch them on in Preferences"
# The title of the published alpha slide.
PUBLISHED_TITLE = "Alpha features"
GROUPS = (
    ("Segmentation", (
        ("Segmentation backends: StarDist, InstanSeg, Omnipose, DINOCell, SAMCell",
         (551, 552, 553, 404, 405)),
        ("Spot detection: Spotiflow, SpotNet", (554, 475)),
        ("Click- and box-prompted segmentation (micro-SAM)", (555,)),
        ("Segmentation uncertainty maps and parameter-robustness report", (568, 578)),
        ("CellProfiler .cppipe pipelines, masks and ROIs via QuPath, ImageJ, COCO",
         (546, 545)),
        ("Virtual staining from brightfield", (558,)),
        ("MitoNet: mitochondria in TEM images", (691,)),
        ("GPU use: batches across GPUs, workers sharing a GPU, memory budget",
         (493, 801, 803, 1011)),
        ("Use Make Masks settings in Mask Generation", (508,)),
        ("Large cropped FOVs: Gallery reader, tiled masks, RAM/VRAM caps, splits/merges",
         (919, 921, 945, 956, 965)),
    )),
    ("Image preprocessing", (
        ("Spectral unmixing, photobleaching correction, intensity calibration",
         (538, 539, 580)),
        ("Flat field: vendor profiles, nuclear median, reuse in masks-only runs",
         (543, 933, 802)),
        ("Deep-learning image QC and Noise2Void denoising", (559, 557)),
        ("Illumination, enhancement and PSF settings", (591,)),
    )),
    ("Measurements and\nassays", (
        ("Cell-cycle phase classification", (535,)),
        ("Viability, confluency, wound closure and colony / CFU counts",
         (540, 541, 536, 542)),
        ("Profiling: normalise, select, mAP, feature QC, TVN batch correction",
         (547, 791, 770)),
        ("Size filters from pixel size, label images from other tools", (800, 790)),
        ("Per-cell export (cell, cytoplasm, nucleus) and GPU measurement", (769, 566)),
        ("OPS features in Measure, slow families off by default", (934, 941, 1008)),
        ("Plaques: scale, growth time, sizes and effects vs reports", (501, 911, 900)),
        ("Infection by titration, effector translocation by parasite load", (898, 910)),
    )),
    ("Time-lapse", (
        ("Tracking: SAM2 propagation, Timeflows, lineage trees", (556, 426, 537)),
        ("Division analysis: lead-up and onset, minutes and µm, effect sizes",
         (771, 901, 864, 867)),
        ("Division prediction: will it divide soon, shape-and-motion arm", (866, 909)),
        ("Budding-yeast size control", (865,)),
        ("Event detection and time-to-event (Kaplan-Meier, Cox)", (567, 571)),
    )),
    ("Cells and machine\nlearning", (
        ("Embeddings: foundation models, DINO pretraining, similarity search",
         (560, 561, 565)),
        ("Well labels to responding cells, anomalies vs controls", (562, 563)),
        ("Blind scoring and point counts, label checks from cross-channel ground truth",
         (544, 957, 470)),
        ("Counterfactuals: morph a control cell toward a hit", (564,)),
        ("Classifiers: auto pick, class thresholds, crop scaling, shuffled labels, held-out wells",
         (772, 899, 963)),
        ("Channel-adaptive CNN, ViT and well transformer", (897,)),
    )),
    ("Screens", (
        ("Arrayed-screen hits (SSMD, robust z, B-score), sample-size planner",
         (570, 585)),
        ("Chemistry-aware hits: SMILES, clusters, SAR", (584,)),
        ("Spatial transcriptomics: Visium, Xenium", (534,)),
        ("LIMS / plate-barcode linkage", (583,)),
    )),
    ("OPS runs", (
        ("OPS in Mask Generation, phenotype path to per-gene profiles", (372, 752, 971)),
        ("Stitching: Align & Stitch core, seeded placement, overlap match rates",
         (936, 940, 915)),
        ("Phenotype store and crops, two acquisitions, yield funnel", (871, 947, 875)),
        ("Runs resume, run chosen sites, skip missing fields, add controls, count cells once",
         (872, 874, 951, 958, 977)),
        ("DNA flat-field map from nuclei, held-out check", (912,)),
        ("Merge routes: tile pattern, well mosaic, automatic fallback", (962,)),
        ("Pathogen and organelle objects per cell, carried into gene profiles", (970,)),
        ("spaCR store: Parquet tables, Zarr crops, one DuckDB catalog", (974,)),
        ("OME-Zarr images and ragged crops, any-channel Annotate roles", (983, 1007)),
    )),
    ("OPS genotyping", (
        ("OPS spacer-prefix decode with quality classes",
         (920,)),
        ("False calls, calibrated confidence, matching to other pipelines",
         (876, 877, 878, 922)),
        ("Detection, read mode and cell calls tuned together, prefix decoy FDR",
         (942, 952, 949, 955, 967, 960, 988, 982, 961)),
    )),
    ("OPS gene profiles", (
        ("Profiles over all cells with NT rows, re-run from stored tables", (883, 879)),
        ("Guide noise: contrastive weighting, efficacy blend, reliability shrinkage",
         (907, 887, 902)),
        ("Cell-cycle-resolved and cell-state profiles, mitotic index test", (890, 914)),
        ("Penetrance, escapers, responder fraction, distribution distances", (891, 905)),
        ("Fitness and generic-axis removal, embedding and fusion profiles", (903, 906)),
        ("Feature information screen: what new columns add about genes", (964,)),
    )),
    ("OPS benchmarks", (
        ("Gene-pair mAP vs null, guide-level P values, versioned references",
         (880, 904, 881)),
        ("Specificity vs decoys, CORUM, STRING, MitoCarta, zero-TPM hit calls",
         (882, 923, 908)),
        ("copairs and EFAAR scores, per-gene E-test and power", (913, 917, 968)),
    )),
    ("OPS genes and\nclusters", (
        ("Clusters: Leiden sweep, bootstrap and gene-pairs consensus, test, atlas",
         (916, 953, 959, 928, 948, 929, 888)),
        ("Gene map: profile clusters, labels, medoid crops", (918,)),
        ("Gene graph, protein knowledge, fusion hypotheses, gene-class training",
         (884, 885, 886, 889)),
        ("Voyager3: cluster roles from UniProt facts through your own AI, or by enrichment", (973, 980)),
    )),
    ("Automation and data", (
        ("Watch a folder as images arrive, send hits back for re-imaging", (548, 549)),
        ("Storage: cloud (S3 and others), Postgres or DuckDB / Parquet", (550, 576)),
        ("Public datasets: fetch, check vs archive, compare per-gene tables",
         (750, 751)),
        ("MoA benchmarks: leave one compound out, CellProfiler Analyst", (795, 894)),
        ("Screen plates from one settings file, aggregation, crop shards, UMAP",
         (893, 937, 896)),
        ("Exports: Nextflow / Snakemake, R / Parquet, masks for vvvv", (575, 581, 685)),
        ("Run-finished notifications (email, Slack, ntfy)", (577,)),
        ("Plugin and recipe catalogue", (582,)),
        ("Test data on seven more modules", (633,)),
    )),
    ("Reproducibility", (
        ("Figure-integrity check and pre-registered analysis lock", (572, 573)),
        ("Archives: REMBI / IDR metadata, one-click Zenodo DOI", (574, 579)),
    )),
)


def _columns(groups=GROUPS):
    """Split whole categories into two columns whose lengths differ least."""
    sizes = [len(lines) for _, lines in groups]
    total = sum(sizes)
    cut = min(range(len(groups) + 1),
              key=lambda i: abs(total - 2 * sum(sizes[:i])))
    return groups[:cut], groups[cut:]


# Alpha organism pages (ALPHA_SPECIES), drawn as a strip with their Home icons.
SPECIES_ITEM = 634
SPECIES = (
    ("plasmodium", "Plasmodium"),
    ("candida", "Candida"),
    ("trypanosoma", "Trypanosoma"),
    ("leishmania", "Leishmania"),
    ("giardia", "Giardia"),
    ("virus", "Virus"),
    ("mammalian", "Mammalian"),
)


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


def _species_icons():
    # The Home tile icon of each alpha species, read from iconset's table.
    tree = ast.parse((ROOT / "spacr/qt/iconset.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "SHARED_ICON_ASSETS"):
            table = ast.literal_eval(node.value)
            return [ROOT / "spacr/resources/icons" / table[key] for key, _ in SPECIES]
    raise ValueError("SHARED_ICON_ASSETS not found in spacr/qt/iconset.py")


def _listed_items():
    """Every item number on a slide line, in slide order."""
    return [number for _, lines in GROUPS for _, numbers in lines for number in numbers]


def _check_registry(source):
    """Return the number of alpha features listed; the species strip is extra."""
    registered = _registry_ids(source)
    listed = _listed_items()
    if SPECIES_ITEM in registered:
        listed.append(SPECIES_ITEM)
    if len(listed) != len(set(listed)) or set(listed) != registered:
        duplicates = sorted({n for n in listed if listed.count(n) > 1})
        raise ValueError(
            f"Update alpha slide lines: missing {sorted(registered - set(listed))}; "
            f"retired {sorted(set(listed) - registered)}; duplicated {duplicates}"
        )
    return len(listed) - (SPECIES_ITEM in registered)


def _footer(count):
    lines = sum(len(lines) for _, lines in GROUPS)
    return (f"{count} alpha features in {lines} groups and {len(SPECIES)} alpha species, "
            "hidden until Preferences → Modules → Show alpha features and Show alpha "
            "species are on. Early versions: expect changes.")


def _draw_pdf(target, count, page_size):
    """Draw the alpha page into a one-page PDF."""
    # Qt emits vector text with embedded fonts. No application window is opened.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QMarginsF, QRectF, QSizeF, Qt
    from PySide6.QtSvg import QSvgRenderer
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

    def label(text, x, y, w, h, size=27, color="#e3e8ed", bold=False):
        font = QFont("Open Sans")
        font.setPixelSize(size)
        font.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
        # Refuse to publish truncated labels when a line changes.
        metrics = QFontMetricsF(font, pdf)
        if any(metrics.horizontalAdvance(line) > w for line in text.splitlines()):
            raise ValueError(f"Alpha slide label does not fit: {text}")
        painter.setFont(font)
        painter.setPen(QColor(color))
        painter.drawText(QRectF(x, y, w, h), Qt.AlignmentFlag.AlignVCenter, text)

    columns = _columns()
    # Rows shrink from 36 px so the longest column ends above the species strip.
    longest = max(sum(len(lines) for _, lines in column) for column in columns)
    row_height = min(36, int((1545 - 375) // longest))
    # The captions shrink with the rows, so a line never spills into the next.
    caption_size = min(27, row_height - 4)
    icons = _species_icons()
    try:
        painter.fillRect(0, 0, 3200, 1800, QColor("#0e1116"))
        label("I N T R O D U C T I O N", 120, 82, 1100, 80, 36, "#29bfb3")
        label(PUBLISHED_TITLE, 120, 160, 2650, 130, 94)
        painter.setPen(QPen(QColor("#d63fa1"), 3))
        painter.setBrush(QColor("#222932"))
        painter.drawRoundedRect(QRectF(2850, 118, 230, 76), 38, 38)
        label("ALPHA", 2910, 123, 160, 64, 32, "#d63fa1", True)
        for x, groups in zip((120, 1640), columns):
            painter.fillRect(QRectF(x, 310, 1440, 65), QColor("#222932"))
            label("Category", x + 14, 310, 345, 65, 34, bold=True)
            label("Alpha features", x + 388, 310, 1035, 65, 34, bold=True)
            y = 375
            for category, lines in groups:
                painter.fillRect(QRectF(x, y, 371, len(lines) * row_height - 3),
                                 QColor("#191f27"))
                label(category, x + 14, y, 345, len(lines) * row_height - 3)
                for text, _ in lines:
                    painter.fillRect(QRectF(x + 374, y, 1066, row_height - 3),
                                     QColor("#191f27"))
                    label(text, x + 388, y, 1038, row_height - 3, caption_size)
                    y += row_height
        top, cell_h = 1560, 110
        painter.fillRect(QRectF(120, top, 371, cell_h), QColor("#222932"))
        label("Alpha species", 134, top, 345, cell_h, 34, bold=True)
        cell_w = (2960 - 374) / len(SPECIES)
        for i, (path, (_, name)) in enumerate(zip(icons, SPECIES)):
            x = 494 + i * cell_w
            painter.fillRect(QRectF(x, top, cell_w - 3, cell_h), QColor("#191f27"))
            renderer = QSvgRenderer(str(path))
            if not renderer.isValid():
                raise ValueError(f"Species icon does not load: {path}")
            renderer.render(painter, QRectF(x + 18, top + 23, 64, 64))
            label(name, x + 96, top, cell_w - 108, cell_h, 30)
        label(_footer(count), 120, 1700, 2960, 62, 30, "#acb5be")
    finally:
        painter.end()
    # Keep the application alive until the paint device is finalized.
    assert app is not None


def _viewer_pages(folder, manifest):
    # The GitHub pages and the viewer carry the slide titles.
    spec = importlib.util.spec_from_file_location(
        "build_readme_deck", ROOT / "tools/build_readme_deck.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    if (folder / "pages").is_dir():
        builder.github_pages(folder, [slide["title"] for slide in manifest["slides"]])
    if (folder / "index.html").is_file():
        (folder / "index.html").write_text(builder.VIEWER.read_text(encoding="utf-8").replace(
            "__SLIDES__", json.dumps(manifest, ensure_ascii=False)), encoding="utf-8")


def refresh(folder, settings_source=ROOT / "spacr/settings.py"):
    from PIL import Image
    from pypdf import PdfReader, PdfWriter

    count = _check_registry(settings_source)
    manifest_path = folder / "slides.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [i for i, slide in enumerate(manifest["slides"])
               if slide["title"] in (TITLE, PUBLISHED_TITLE)]
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
        replacement = PdfReader(page_pdf)
        if len(replacement.pages) != 1:
            raise ValueError("The alpha list must render as one page")
        subprocess.run(["pdftoppm", "-singlefile", "-png", "-scale-to-x", "3200",
                        "-scale-to-y", "1800", str(page_pdf), str(stage / "alpha")],
                       check=True, capture_output=True, timeout=120)
        with Image.open(stage / "alpha.png") as picture:
            picture.convert("RGB").save(stage / "slide.jpg", quality=86,
                                         optimize=True, subsampling=0, progressive=True)
            picture.resize((320, 1800 * 320 // 3200), Image.Resampling.LANCZOS).convert(
                "RGB").save(stage / "thumb.jpg", quality=75, optimize=True)
        writer = PdfWriter()
        for number, page in enumerate(reader.pages):
            writer.add_page(replacement.pages[0] if number == index else page)
        if reader.metadata:
            writer.add_metadata({str(k): str(v) for k, v in reader.metadata.items()})
        writer.write(stage / "deck.pdf")
        # All validation/rendering finishes before replacing a published artifact.
        for source, target in (("deck.pdf", pdf_path), ("slide.jpg", image_path),
                               ("thumb.jpg", thumb_path)):
            os.replace(stage / source, target)
    slide["title"] = PUBLISHED_TITLE
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    _viewer_pages(folder, manifest)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/source/_static/deck")
    args = parser.parse_args()
    published = args.out.resolve() == (ROOT / "docs/source/_static/deck").resolve()
    if published:
        _spec = importlib.util.spec_from_file_location("fetch_docs_media", ROOT / "tools/fetch_docs_media.py")
        media = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(media)
        media.fetch("deck", ROOT)
    print(f"Updated alpha slide: {refresh(args.out)} registered features")
    if published:
        revision = media.publish("deck", ROOT, "deck: alpha feature slide follows the registry")
        print(f"deck media pinned at {revision}" if revision else "deck media unchanged")
