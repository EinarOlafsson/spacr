#!/usr/bin/env python3
"""Build separate translated README and API-docstring catalogs.

English docstrings remain beside their Python symbols.  Translations are
stored below ``docs/i18n`` and copied into Sphinx's static tree, keyed by the
fully-qualified symbol plus a hash of the canonical English text.  The hash
makes a changed English docstring an explicit stale-catalog failure rather
than silently displaying an obsolete translation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import inspect
import json
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Callable, Iterable, Mapping

from build_i18n_catalogs import (
    MODEL_SPECS,
    SECONDARY_LICENSE,
    SECONDARY_MODEL,
    _COMPUTE_RUN_SOURCE,
    _COMPUTE_THREAD_SOURCE,
    _CONTEXT_HARD_PROTECT_RE,
    _DATA_GATE_SOURCE,
    _DICTIONARY_SOURCE,
    _IMAGE_CROP_SOURCE,
    _IMAGING_CHANNEL_SOURCE,
    _IMAGING_FIELD_SOURCE,
    _HUMAN_READABLE_SOURCE,
    _MAPPING_KEY_SOURCE,
    _PIPELINE_SOURCE,
    _PLANE_SOURCE,
    _PROTECT_RE,
    _RST_ROLE_PATTERN,
    _SCIENTIFIC_PLATE_SOURCE,
    _SCIENTIFIC_SCREEN_SOURCE,
    _SOFTWARE_CLASSIFIER_SOURCE,
    _SOFTWARE_QUEUE_SOURCE,
    _TOKEN_RE,
    _context_prose,
    _contextualize,
    _english_well_sense_counts,
    _gui_screen_source,
    _has_traditional_chinese_prose,
    _looks_degenerate,
    _raise_sense_counts,
    _semantic_false_friends,
    _statistical_power_source,
    _syntax_preserved,
    _translate_batches,
    _translation_chunks,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs" / "i18n"
README_DIR = SOURCE_DIR / "readme"
STATIC_API_DIR = ROOT / "docs" / "source" / "_static" / "i18n" / "api"
API_DIR = STATIC_API_DIR
REVIEWED_API_DIR = ROOT / "docs" / "i18n" / "reviewed" / "api"
README_SOURCE = ROOT / "README.rst"

# A cache entry is meaningful only for the exact English text sent to the
# model.  v7 adds structural RST detachment and target-neutral sense context;
# v6 checkpoints can contain English fallbacks or translations generated from
# an ambiguous word and must never be promoted under the new source hash.
API_BLOCK_CACHE_NAMESPACE = "api-block-v7"

LANGUAGE_PICKER_LABELS = {
    "sv": "Språk",
    "de": "Sprachen",
    "es": "Idiomas",
    "zh_CN": "语言",
    "pt": "Idiomas",
    "hi": "भाषाएँ",
    "ko": "언어",
    "is": "Tungumál",
    "fr": "Langues",
}

# These records are code/identifier examples, not English prose.  Exact text
# is intentional: translating it would corrupt the API contract being shown.
API_EXACT_TEXT_ALLOWLIST = frozenset({
    "spacr.align.CanvasSpec.shape",
    "spacr.hits.HitList.flag_counts",
    "spacr.macro.MacroStep.entry",
    "spacr.qt.widgets.plate_layout.PlateDesign.shape",
    "spacr.resources.home.versions._generators.common.app_map",
    "spacr.run_compare.HitList.by_key",
    "spacr.schema.field_index",
})

# AutoAPI renders the value of these documented public string constants behind
# a ``Show Value`` control.  The control label and the Python quotes are HTML
# chrome, not documentation.  Extract the literal's value from the AST so the
# catalog contains only the prose a reader sees after expanding the control.
API_VALUE_DOC_ASSIGNMENTS = frozenset({
    "spacr.anndata_export.ANNDATA_MISSING_MESSAGE",
    "spacr.napari_bridge.NAPARI_MISSING_MESSAGE",
    "spacr.ome_zarr.CODEC_MISSING_MESSAGE",
    "spacr.ome_zarr.ZARR_MISSING_MESSAGE",
    "spacr.omero.OMERO_MISSING_MESSAGE",
    "spacr.qt.widgets.home.PAUSE_UNAVAILABLE",
    "spacr.qt.widgets.preview_controls.MAX_SETS_TOOLTIP",
})

# AutoAPI visibly repeats inherited methods under the concrete subclass id.
# These aliases were reviewed against the current generated API pages and the
# canonical source docstrings.  Keeping exact ids avoids the ambiguous suffix
# matching that previously made imported or stdlib members unsafe.  In
# particular, ``spacr.logging_util.LevelSetFilter.filter`` is intentionally
# absent: its documentation comes from external ``logging.Filter.filter``.
API_DOC_ALIASES: Mapping[str, str] = {
    "spacr.layers.ImageLayer.world_extent":
        "spacr.layers.Layer.world_extent",
    "spacr.layers.ImageLayer.ndim":
        "spacr.layers.Layer.ndim",
    "spacr.layers.ImageLayer.shape":
        "spacr.layers.Layer.shape",
    "spacr.layers.LabelsLayer.world_extent":
        "spacr.layers.Layer.world_extent",
    "spacr.layers.LabelsLayer.ndim":
        "spacr.layers.Layer.ndim",
    "spacr.layers.LabelsLayer.shape":
        "spacr.layers.Layer.shape",
    "spacr.layers.PointsLayer.world_extent":
        "spacr.layers.Layer.world_extent",
    "spacr.layers.PointsLayer.ndim":
        "spacr.layers.Layer.ndim",
    "spacr.layers.ShapesLayer.world_extent":
        "spacr.layers.Layer.world_extent",
    "spacr.layers.ShapesLayer.ndim":
        "spacr.layers.Layer.ndim",
    "spacr.qt.dnd_handlers.AlignDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.AlignDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.AlignDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.AnnotateDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.AnnotateDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.AnnotateDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.BatchDropHandler.accepts_multiple":
        "spacr.qt.dnd.DropHandler.accepts_multiple",
    "spacr.qt.dnd_handlers.BatchDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.BatchDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.BatchDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.ClassifyDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.ClassifyDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.ClassifyDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.CoefficientsDropHandler.deliver":
        "spacr.qt.dnd_handlers.LayoutDropHandler.deliver",
    "spacr.qt.dnd_handlers.ConvertDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.ConvertDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.ConvertDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.DataManagerDropHandler.deliver":
        "spacr.qt.dnd_handlers.LayoutDropHandler.deliver",
    "spacr.qt.dnd_handlers.DatabaseDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.DatabaseDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.DatabaseDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.EvaluationBundleDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.EvaluationBundleDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.EvaluationBundleDropHandler.deliver":
        "spacr.qt.dnd_handlers.LayoutDropHandler.deliver",
    "spacr.qt.dnd_handlers.ExternalMasksDropHandler.accepts_multiple":
        "spacr.qt.dnd.DropHandler.accepts_multiple",
    "spacr.qt.dnd_handlers.ExternalMasksDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.ExternalMasksDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.ExternalMasksDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.ForeignProjectDropHandler.accepts_multiple":
        "spacr.qt.dnd.DropHandler.accepts_multiple",
    "spacr.qt.dnd_handlers.ForeignProjectDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.ForeignProjectDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.ForeignProjectDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.ImageFieldsDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.ImageFieldsDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.ImageFieldsDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.LabelMaskDropHandler.deliver":
        "spacr.qt.dnd_handlers.LayoutDropHandler.deliver",
    "spacr.qt.dnd_handlers.LayerStackDropHandler.accepts_multiple":
        "spacr.qt.dnd.DropHandler.accepts_multiple",
    "spacr.qt.dnd_handlers.LayerStackDropHandler.deliver":
        "spacr.qt.dnd_handlers.LayoutDropHandler.deliver",
    "spacr.qt.dnd_handlers.LayoutDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.LayoutDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.LayoutDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.LayoutDropHandler.suggest_alternatives":
        "spacr.qt.dnd.DropHandler.suggest_alternatives",
    "spacr.qt.dnd_handlers.LineageDropHandler.deliver":
        "spacr.qt.dnd_handlers.LayoutDropHandler.deliver",
    "spacr.qt.dnd_handlers.MakeMasksDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.MakeMasksDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.MakeMasksDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.MakeMasksDropHandler.suggest_alternatives":
        "spacr.qt.dnd.DropHandler.suggest_alternatives",
    "spacr.qt.dnd_handlers.MapBarcodesDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.MapBarcodesDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.MapBarcodesDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.MaskDropHandler.accepts_multiple":
        "spacr.qt.dnd.DropHandler.accepts_multiple",
    "spacr.qt.dnd_handlers.MaskDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.MaskDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.MaskDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.MaskDropHandler.suggest_alternatives":
        "spacr.qt.dnd.DropHandler.suggest_alternatives",
    "spacr.qt.dnd_handlers.MeasureDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.MeasureDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.MeasureDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.MeasureDropHandler.suggest_alternatives":
        "spacr.qt.dnd.DropHandler.suggest_alternatives",
    "spacr.qt.dnd_handlers.MeasurementsDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.MeasurementsDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.MeasurementsDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.MethodsSourcesDropHandler.accepts_multiple":
        "spacr.qt.dnd.DropHandler.accepts_multiple",
    "spacr.qt.dnd_handlers.MethodsSourcesDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.MethodsSourcesDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.ModelZooDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.ModelZooDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.ModelZooDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.PlateQueueDropHandler.accepts_multiple":
        "spacr.qt.dnd.DropHandler.accepts_multiple",
    "spacr.qt.dnd_handlers.PlateQueueDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.PlateQueueDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.PlateQueueDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.ProjectFolderDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.ProjectFolderDropHandler.deliver":
        "spacr.qt.dnd_handlers.LayoutDropHandler.deliver",
    "spacr.qt.dnd_handlers.ProjectRootsDropHandler.accepts_multiple":
        "spacr.qt.dnd.DropHandler.accepts_multiple",
    "spacr.qt.dnd_handlers.ProjectRootsDropHandler.deliver":
        "spacr.qt.dnd_handlers.LayoutDropHandler.deliver",
    "spacr.qt.dnd_handlers.ReportDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.ReportDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.ReportDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.ResultsDatabaseDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.ResultsFolderDropHandler.deliver":
        "spacr.qt.dnd_handlers.LayoutDropHandler.deliver",
    "spacr.qt.dnd_handlers.RunHistoryDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.RunHistoryDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.ScatterTableDropHandler.deliver":
        "spacr.qt.dnd_handlers.LayoutDropHandler.deliver",
    "spacr.qt.dnd_handlers.SourceDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.SourceDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.SourceDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.SubmissionSettingsDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.SubmissionSettingsDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.SubmissionSettingsDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.dnd_handlers.TableDropHandler.deliver":
        "spacr.qt.dnd_handlers.LayoutDropHandler.deliver",
    "spacr.qt.dnd_handlers.TrainingRunsDropHandler.apply":
        "spacr.qt.dnd.DropHandler.apply",
    "spacr.qt.dnd_handlers.TrainingRunsDropHandler.can_accept":
        "spacr.qt.dnd.DropHandler.can_accept",
    "spacr.qt.dnd_handlers.TrainingRunsDropHandler.error_message":
        "spacr.qt.dnd.DropHandler.error_message",
    "spacr.qt.screens.db_browser.DbBrowserScreen.on_linked_filter_changed":
        "spacr.qt.linked_selection.LinkedView.on_linked_filter_changed",
    "spacr.qt.widgets.gate_spec.PolygonGate.scaled":
        "spacr.qt.widgets.gate_spec.Gate.scaled",
    "spacr.qt.widgets.gate_spec.PolygonGate.translated":
        "spacr.qt.widgets.gate_spec.Gate.translated",
    "spacr.qt.widgets.gate_spec.PolygonGate.with_handle":
        "spacr.qt.widgets.gate_spec.Gate.with_handle",
    "spacr.qt.widgets.gate_spec.RectGate.centre":
        "spacr.qt.widgets.gate_spec.Gate.centre",
    "spacr.qt.widgets.gate_spec.RectGate.range_filters":
        "spacr.qt.widgets.gate_spec.Gate.range_filters",
    "spacr.qt.widgets.gate_spec.RectGate.scaled":
        "spacr.qt.widgets.gate_spec.Gate.scaled",
    "spacr.qt.widgets.gate_spec.RectGate.translated":
        "spacr.qt.widgets.gate_spec.Gate.translated",
    "spacr.qt.widgets.gate_spec.RectGate.with_handle":
        "spacr.qt.widgets.gate_spec.Gate.with_handle",
    "spacr.qt.widgets.gate_spec.ThresholdGate.centre":
        "spacr.qt.widgets.gate_spec.Gate.centre",
    "spacr.qt.widgets.gate_spec.ThresholdGate.range_filters":
        "spacr.qt.widgets.gate_spec.Gate.range_filters",
    "spacr.qt.widgets.gate_spec.ThresholdGate.scaled":
        "spacr.qt.widgets.gate_spec.Gate.scaled",
    "spacr.qt.widgets.gate_spec.ThresholdGate.with_handle":
        "spacr.qt.widgets.gate_spec.Gate.with_handle",
    "spacr.qt.widgets.pca_view.PCAScoresCanvas.render_now":
        "spacr.qt.widgets.graph_builder.GraphCanvas.render_now",
    "spacr.qt.widgets.umap_explorer.ImageUmapExplorer.on_linked_filter_changed":
        "spacr.qt.linked_selection.LinkedView.on_linked_filter_changed",
}

# Hashes, rather than symbol names, make the exception as narrow as possible:
# moving an unrelated English paragraph into one of these symbols cannot make
# it pass. Each allowed block is an API literal/example with no translatable
# prose. New exceptions require reviewing and hashing the exact source block.
API_EXACT_BLOCK_SHA256_ALLOWLIST = frozenset({
    "fc1a080cac9c8a69250235e646f7d54c59232e73c664a392b6813e5abf088937",
    "a8356398b7efb4933388f52627f8426e5131037bde864f92cf70702d8febf0d8",
    "52e945ac179782227292ff5d8bd6ee111fee7c9d4af2507fbb4df88274f0530a",
    "88ea1fff9abccfac5e734cd0a4432d94fa3d040e9c9ef61fe8917754c5e60f60",
    "2ce9dc6784fba8f7e3145c6b68fdf1bf3c65c64d36fbd35f8bd25ada10ce3a78",
    "aa92ef42ce3a0eb9152971bc75aaf0ec6a8e91924c42fec7233b2646f47f29b4",
    "429f6345c1f6b0d30acf6ff8aaed026ebbbf5df506280a11b662699d1b2e8f33",
    "dad752c61526a8407fc3793497731f5c989b6cfb1eb05d4177c317ee7fdfbfca",
    "a1b0b118e72f9ab34aedf8a8f00d1da31f02e6843e02de8f5ff7568f6c62ea2c",
})

# Some terse API fragments are ambiguous noun labels, and general translation
# checkpoints legitimately treat words such as "layout", "backend" and
# "benchmark" as English loanwords. These target-neutral English expansions
# give the model enough context to produce useful prose; target-language text
# is still generated and validated, never hand-authored here.
API_TRANSLATION_CONTEXT = {
    (
        "Matplotlib is forced to ``Agg`` when there is no display, and "
        "``plt.show`` is replaced by a close-the-figure shim for the duration "
        "of the run — the same thing :func:`spacr.gui_utils.spacrFigShow` does "
        "inside the GUI — so a pipeline that calls ``plt.show()`` neither "
        "blocks nor leaks figures."
    ): (
        "When no display is available, force Matplotlib to use ``Agg``. During "
        "the run, replace ``plt.show`` with a helper that closes the figure. "
        ":func:`spacr.gui_utils.spacrFigShow` uses the same behavior inside the "
        "GUI. Therefore a workflow that calls ``plt.show()`` does not block "
        "and does not leak figures."
    ),
    (
        "`close_polygon` ALREADY emits `gate_drawn`. Emitting again here made "
        "one drawn polygon prompt for a name twice and create two identical "
        "gates -- which is exactly what was reported. This wrapper exists only "
        "so the click-the-first-vertex path and the Close button share a name."
    ): (
        "`close_polygon` already emits `gate_drawn`. Emitting it a second time "
        "would ask for the polygon name twice and create two identical gates. "
        "This wrapper only gives the first-vertex click path and the Close "
        "button one shared name."
    ),
    (
        ":func:`sibling_sources` lists **every** comparable file in the folder, "
        "and the panels fed that straight into their field-of-view dropdown. "
        "Measured on a 384-well plate at 16 fields and 4 channels (24 576 "
        "files) and on one four times larger (98 304 files):"
    ): (
        ":func:`sibling_sources` returns **every** comparable file in the "
        "folder. The panels previously sent that complete list directly to "
        "their field-of-view selector. We measured this behavior on a 384-well "
        "plate with 16 fields and 4 channels (24 576 files), and on another "
        "plate four times larger (98 304 files):"
    ),
    (
        "stamp this id instead of the ambient one. Used by the per-run log so "
        "a nested run's records are not misattributed."
    ): (
        "write this identifier instead of the ambient identifier. The log for "
        "each run uses it to attribute records from a nested run correctly."
    ),
    "worker threads for the scoring pass. Default ``8``.":
        "Worker threads used by the scoring pass. The default is ``8``.",
    "Parse plate/well/field/object identities from a crop filename.":
        "Read the plate, well, field, and object identifiers from a crop filename.",
    "Layouts:": "Screen layouts:",
    "Layout:": "Screen layout:",
    "Layout": "Screen layout",
    "1-based z-slice id.": "Identifier of the z slice, counted from one.",
    "``(H, W, 3)`` uint8 RGB array.":
        "A three-channel uint8 RGB image array with shape ``(H, W, 3)``.",
    "display-sized ``(size, size, 3)`` uint8 RGB frames.":
        "Display-ready uint8 RGB image frames with shape ``(size, size, 3)``.",
    "bytes.": "Size measured in bytes.",
    "label text.": "Text displayed on the label.",
    "benchmarks.": "Performance benchmark results.",
    "Backends": "Available computation backends",
    "QApplication bootstrap + MainWindow.":
        "Initialize QApplication and MainWindow.",
    "Design:": "Software design:",
    "App": "Application",
    "ditto": "Same as the row above.",
    "ditto + ``scores.csv``": "Same as the row above, plus ``scores.csv``.",
    "In-app fallback via ``QSystemTrayIcon``.":
        "Fallback inside the application using ``QSystemTrayIcon``.",
    "Threading": "Thread management",
    "pixels...": "Number of pixels...",
    "Model": "Machine-learning model",
    "``source``'s surviving rows with ``PC1…PCk`` columns added.":
        "Rows from ``source`` that survived filtering, with ``PC1…PCk`` columns added.",
    "Fitted statsmodels results object or sklearn estimator.":
        "A fitted statsmodels results object or a fitted sklearn estimator.",
    "Standalone": "Standalone use",
    "Args": "Function arguments",
    "Mosaic": "Image mosaic",
    "Assembles:": "Builds the following components:",
    "API  Animation": "Animation for the API",
    "Register ``hook(channel_arrays, context) -> np.ndarray``.":
        "Register the preprocessing callback ``hook(channel_arrays, context) -> np.ndarray``.",
    "Register ``hook(context) -> np.ndarray[bool]``.":
        "Register the region-filter callback ``hook(context) -> np.ndarray[bool]``.",
    "1. MAD — :data:`METHOD_MAD`":
        "Outlier-detection method 1: MAD — :data:`METHOD_MAD`",
    "2. IQR / Tukey — :data:`METHOD_IQR`":
        "Outlier-detection method 2: IQR / Tukey — :data:`METHOD_IQR`",
    'what made the edits ("spacr-qt curation").':
        'A description of what made the edits ("spacr-qt curation").',
    'Populate default settings for the "graph importance" plot utility.':
        'Fill in the default settings for the "graph importance" plot utility.',
    (
        "the well name, e.g. ``'A01'`` — zero padded to two column digits, "
        "which is what spaCR's strict Yokogawa regex requires."
    ): (
        "The well name, for example ``'A01'``. It is zero-padded to two "
        "column digits, as required by spaCR's strict Yokogawa filename pattern."
    ),
    (
        "Design constraint: this module must be usable *before* committing to a "
        "segmentation run, which means it may not cost what a segmentation run "
        "costs. It therefore imports **no torch and no cellpose** — that is a "
        "tested property, not an aspiration (see ``tests/test_diameter_estimator.py``). "
        "It reuses :mod:`spacr.validate` for filename-metadata parsing, which is "
        "the other deliberately dependency-light module in the package "
        "(``spacr.utils``, where ``_get_regex`` and "
        "``_extract_filename_metadata`` live, imports torch and cellpose at module "
        "scope and so cannot be touched from here). The regexes in "
        "``spacr.validate.METADATA_REGEXES`` mirror ``spacr.utils._get_regex``, "
        "and the channel ordering used below mirrors "
        "``spacr.io._rename_and_organize_image_files``, which concatenates one "
        "plane per distinct ``chanID`` in sorted order — so channel index *i* is "
        "the *i*-th sorted ``chanID``, exactly as ``cell_channel`` and friends mean it."
    ): (
        "*Early availability:* the module works before segmentation. "
        "**Lightweight dependencies:** it avoids torch and cellpose. This is "
        "tested by ``tests/test_diameter_estimator.py``. :mod:`spacr.validate` "
        "parses filename metadata without heavy imports. The alternative "
        "``spacr.utils`` imports torch and cellpose because it defines "
        "``_get_regex`` and ``_extract_filename_metadata``. Equivalent rules "
        "appear in ``spacr.validate.METADATA_REGEXES`` and "
        "``spacr.utils._get_regex``. "
        "``spacr.io._rename_and_organize_image_files`` sorts planes by ``chanID``. "
        "Position *i* selects one entry. At position *i*, the sorted ``chanID`` "
        "corresponds to ``cell_channel``."
    ),
    "original x.": "Original horizontal coordinate.",
    "original y.": "Original vertical coordinate.",
    "bool.": "Boolean value.",
    "dispersion": "dispersion parameter",
    "distribution": "probability distribution",
    "``(key, name, description, section)`` per app.":
        "One ``(key, name, description, section)`` tuple for each application.",
    "Post (bool)": "Post-processing option (Boolean value)",
    "Pre  (bool)": "Pre-processing option (Boolean value)",
    "Instantiated ``cellpose.models.CellposeModel``.":
        "An initialized instance of ``cellpose.models.CellposeModel``.",
    "Copyright © 2025 olafsson lab":
        "Copyright © 2025, olafsson lab.",
    "Return up to ``n`` image paths from ``path`` — used by the filename-regex preview in the mask handler.":
        "Return at most ``n`` paths to image files found under ``path``. These names support the mask handler's pattern preview.",
    "stale rows deleted by the delete-before-insert.":
        "Obsolete rows removed prior to inserting new rows.",
    "rows already deleted by the caller's delete-before-insert, recorded for the report.":
        "Rows already removed prior to insertion by the caller, as recorded in the report.",
    "filename stem (no extension).":
        "Base name of the file, excluding its extension.",
    (
        "write ``layers['missing']``. Defaults to True for the imputing policies "
        "and False otherwise."
    ): (
        "Whether to write ``layers['missing']``. The default is True for imputation "
        "policies and False for other policies."
    ),
    "dict of defaults; empty when the pipeline has no helper.":
        "Dictionary containing default settings; empty if the pipeline has no helper.",
    (
        "The one exception is a folder finished with ``on_error='skip'``: its marker "
        "names the files that could not be rewritten, those stay legacy (and are read "
        "as legacy) inside an otherwise format-2 folder, and a later run retries "
        "**only** them."
    ): (
        "Usually all files are migrated. A folder completed with "
        "``on_error='skip'`` is different: its marker lists files that were not "
        "rewritten. Those files remain in the legacy format while the other files "
        "use format 2. A later run retries **just those files**."
    ),
    (
        'An empty string is the *only* value that means "delete this". Every failure '
        "mode — including one this function did not anticipate — produces a sentence, "
        "which is the direction a delete predicate must fail in."
    ): (
        'An empty string is the *sole* value meaning "delete this". Any failure, '
        "including an unexpected one, returns an explanatory sentence. A deletion "
        "predicate must fail safely in this direction."
    ),
    (
        "kinds to consider. Defaults to :data:`DEFAULT_PRUNABLE_KINDS`. Naming a "
        ":data:`PROTECTED_KINDS` member opts it in — it still has to pass every "
        "safety rule. Naming an :data:`ORIGINAL_KINDS` member does nothing: there is "
        "no path through this module that deletes an original."
    ): (
        "Object kinds to inspect. The default is :data:`DEFAULT_PRUNABLE_KINDS`. "
        "Explicitly naming a :data:`PROTECTED_KINDS` member permits that kind, but "
        "all safety rules still apply. Naming an :data:`ORIGINAL_KINDS` member has "
        "no effect; this module has no route that deletes an original."
    ),
    (
        "The **only** place this module imports napari. See the module docstring for "
        "why that matters more here than in most optional-dependency code."
    ): (
        "This function is the **single** place where the module imports napari. The "
        "module docstring explains why this matters for an optional dependency."
    ),
    (
        "callable taking the settings and returning an unconnected gateway. Defaults "
        "to building a real ``BlitzGateway``, which is the only line in this module "
        "that needs the extra."
    ): (
        "A function that receives the settings and returns a disconnected gateway. "
        "The default constructs a real ``BlitzGateway``. This is the single line in "
        "the module that requires the optional dependency."
    ),
    "which modules to judge, in the order to report them. Defaults to :func:`pipeline_order`.":
        "Modules to evaluate, in report order. When omitted, use :func:`pipeline_order`.",
    (
        "checkpoint JSON path. Defaults to "
        "``dst/.spacr_conversion.checkpoint.json``. A checkpoint is written after "
        "every complete field even when ``resume`` is False, so a later invocation "
        "can opt in after a crash."
    ): (
        "Path to the checkpoint JSON file. The default is "
        "``dst/.spacr_conversion.checkpoint.json``. A checkpoint is written after "
        "each completed field even when ``resume`` is False. This lets a later "
        "invocation enable recovery after a crash."
    ),
    (
        "header shown at the top of the app's own screen. Defaults to ``name``; give "
        'it only when the screen wants the longer form ("Illumination Correction" '
        'over a tile that reads "Illumination"). Reaches ``app_screen.APP_TITLES``.'
    ): (
        "Header displayed at the top of the application's screen. If omitted, it "
        'uses ``name``. Supply a custom header when the screen needs longer wording '
        '("Illumination Correction" above a tile labeled "Illumination"). The value '
        "is stored in ``app_screen.APP_TITLES``."
    ),
    (
        "1. :data:`PROMOTIONS` — the apps somebody assessed and moved. Idempotent, "
        "and it never *demotes*: a module some other code has already promoted "
        "further than this table says stays where it is. That matters because the "
        "table is a snapshot of one assessment, and the next assessment should not "
        "be silently undone by re-importing this module. 2. Every registered app "
        "that is in none of the assessment tables and has no line of its own is "
        "written in as :data:`UNASSESSED_STAGE`. This is what stops a new module "
        "inheriting ``stable`` from the absence of an entry — see the module "
        "docstring. It only ever writes where there is nothing, so it cannot "
        "overrule an author, a plugin, or phase 1."
    ): (
        "1. :data:`PROMOTIONS` lists applications that were assessed and moved. "
        "Applying it repeatedly is safe, and it never *moves an application to a "
        "lower stage*. If another component promoted an application further, it "
        "stays there. This table represents one assessment; loading it later must "
        "not undo a newer assessment. 2. Each registered application absent from "
        "all assessment tables and without its own line receives "
        ":data:`UNASSESSED_STAGE`. This prevents a new module from inheriting "
        "``stable`` merely because no entry exists. The function writes only empty "
        "entries, so it cannot override an author, a plugin, or step 1."
    ),
    (
        "The restart-free path for the Preferences toggle: turning it off deletes "
        "the widget outright rather than hiding it, turning it on builds one on a "
        "screen that has been open all along, and a new theme/palette is pushed at "
        "the existing one without rebuilding it. Idempotent, and cheap enough to "
        "call on every show."
    ): (
        "Preferences can change this with no restart. Disabling the option removes "
        "the widget; enabling it creates one even if the screen is already open. A "
        "new theme or palette is applied to the existing widget. Repeated calls are "
        "safe and inexpensive whenever the screen is shown."
    ),
    (
        "So the sweep cannot simply be narrowed to exact types. Doing that flips "
        "**every** view in the application at once, and the ones already sitting on "
        "a pane would then stack two translucent greys and read about 0.49 — a shade "
        "no position of the page-opacity slider can produce. Nor can a type test "
        "make the distinction: Hit List's ``QTreeWidget`` is the page and Control "
        "Chart's ``QListWidget`` is a passenger, and the pair after them is the other "
        "way round. Only the screen that built the layout knows which it is, so the "
        "screen is asked, once, per view."
    ): (
        "A type-only sweep cannot work. It would change **all** views at once. Views "
        "already inside a pane would stack two translucent greys and appear near "
        "0.49, a shade the page-opacity slider cannot create. Type checks also "
        "cannot distinguish roles: Hit List's ``QTreeWidget`` is a page while "
        "Control Chart's ``QListWidget`` is a passenger, and another pair reverses "
        "the roles. The screen that built the layout knows the role, so each screen "
        "marks its views once."
    ),
    (
        "optional ordered ``(title, [app key])`` — one entry per tab after Home. "
        "Defaults to grouping ``apps`` by their section in first-appearance order, "
        "which is what every test that builds a HomePage out of a handful of tuples "
        "wants."
    ): (
        "Optional ordered ``(title, [app key])`` with one entry for each tab after "
        "Home. When omitted, group ``apps`` by section in first-appearance order. "
        "This matches tests that construct a HomePage from a small set of tuples."
    ),
    (
        "run the decomposition on a worker thread. **Defaults to False**, and the "
        "default is the interesting part -- see :meth:`recompute`. ``PCAScreen`` "
        "passes its own ``threaded`` through, so the application gets the threaded "
        "panel and a panel built directly keeps returning its result from the call."
    ): (
        "Whether decomposition runs on a worker thread. **The default is False**; "
        "see :meth:`recompute` for the reason. ``PCAScreen`` forwards its own "
        "``threaded`` value. Thus the application uses a threaded panel, while a "
        "directly constructed panel returns its result synchronously."
    ),
    (
        "keep only :data:`MEASURE_OWNED_TABLES`. Pass False to see every per-field "
        "table in the database *whoever wrote it* — for inspection only, and "
        ":func:`clear_field_rows` refuses the extras by name if that list is handed "
        "to it."
    ): (
        "Restrict results to :data:`MEASURE_OWNED_TABLES`. Pass False to list all "
        "per-field tables, *regardless of which module wrote them*; this is for "
        "inspection. :func:`clear_field_rows` rejects extra table names if that list "
        "is supplied."
    ),
    (
        "**Measure appends; therefore a resume must delete first.** "
        "``_merge_and_save_to_database`` and ``filepaths_to_database`` both use "
        "``to_sql(..., if_exists='append')``. Re-measuring a field that already "
        "wrote some rows does not overwrite them, it *adds* to them, and every "
        "per-well aggregate downstream is then computed over inflated counts with "
        "nothing anywhere to indicate it. :func:`clear_field_rows` is the "
        "delete-before-insert that makes re-running a field idempotent, and it is "
        "the reason this module exists at all. It runs in one transaction across "
        "every table the field touched, so a failure part-way leaves the database "
        "exactly as it was."
    ): (
        "**The measurement workflow appends rows, so existing rows must be removed "
        "prior to continuing an interrupted run.** ``_merge_and_save_to_database`` "
        "and ``filepaths_to_database`` both call "
        "``to_sql(..., if_exists='append')``. Measuring a field again does not "
        "replace existing rows; it *inserts additional rows*. This makes downstream "
        "per-well aggregates too large without a warning. :func:`clear_field_rows` "
        "removes old rows so repeated measurement produces one consistent result. "
        "One transaction covers every affected table, so an error leaves the "
        "database unchanged."
    ),
    (
        "**This is the delete-before-insert.** Measure appends "
        "(``to_sql(if_exists='append')``), so re-running a field that already wrote "
        "rows adds a second copy of every object rather than replacing the first. "
        "Downstream, ``count_cell`` doubles, per-well means are computed over the "
        "doubled population, and nothing in the artifact says so. Calling this "
        "immediately before re-measuring a field is what makes a resume idempotent."
    ): (
        "**Existing rows are removed prior to inserting replacements.** The "
        "measurement workflow calls ``to_sql(if_exists='append')`` to append rows. "
        "Measuring a field again would create a duplicate of each object rather than "
        "replace the old rows. This makes ``count_cell`` twice as large and biases "
        "the mean for each well. Removing old rows prior to another measurement "
        "makes the operation safe to repeat."
    ),
    (
        '**"The python menu\'s Preferences opens the module recipes window."** '
        '``recipes.MENU_ACTION_TEXT`` is ``"Settings recipes…"``. It contains '
        '*settings*, so Qt gave it ``PreferencesRole`` too — and with two '
        "actions claiming one slot, the wrong one won."
    ): (
        "**The Preferences item in the python menu opens the module recipes "
        "window.** ``recipes.MENU_ACTION_TEXT`` is ``\"Settings recipes…\"``. "
        "It contains *settings*, so Qt also assigned ``PreferencesRole`` to "
        "it; two actions then claimed one slot, and the wrong one won."
    ),
    (
        'Every "N apps" the variants draw or write goes through here. The '
        "count used to be typed into two dozen strings as ``29``; the registry "
        "then grew Distributed Jobs, Classifier Evaluation and Run History and "
        "every one of those strings became a lie that no test could see, "
        "because a literal cannot disagree with itself."
    ): (
        'Every "N apps" count displayed or written by a variant goes '
        "through this function. The count used to be copied into two dozen "
        "strings as ``29``. After Distributed Jobs, Classifier Evaluation, and "
        "Run History were added, every copied count became wrong without a "
        "test noticing, because a literal cannot disagree with itself."
    ),
    (
        "Point-in-polygon is the even–odd ray-casting rule, vectorised. "
        "A row whose x or y is missing is **outside every gate** — not "
        '"unknown", not silently kept. An object with no measurement is '
        "not an object inside the region, and letting it through would put "
        "objects with no value into a population the user defined by value; "
        "the same rule :class:`~spacr.selection.RangeFilter` applies to NaN."
    ): (
        "Point-in-polygon uses the vectorised even–odd ray-casting rule. "
        "A row whose x or y is missing is **outside every gate**. Its state "
        'is not "unknown", and it is not silently kept. An object with no '
        "measurement is not inside the region. Letting it through would put "
        "objects with no value into a population the user defined by value; "
        "the same rule :class:`~spacr.selection.RangeFilter` applies to NaN."
    ),
    (
        "fraction in ``[0, 1]``. ``0.5`` aborts when more than half the items "
        "failed; a rate exactly equal to the threshold does *not* abort."
    ): (
        "fraction in ``[0, 1]``. ``0.5`` aborts when more than half the items "
        "failed; a rate exactly equal to the threshold remains *acceptable*."
    ),
    (
        "``worker.finished -> thread.quit`` is a **DirectConnection**. The "
        "QThread object is created here, on the GUI thread, so it is GUI-affine "
        "— a queued ``quit()`` is posted to the *GUI* thread's event queue, not "
        "to the worker's. Measured: with a queued connection, a GUI thread that "
        "goes straight into ``thread.wait()`` (which is exactly what "
        "``ConsolePanel.shutdown`` and every \"drain before closing\" path does) "
        "waits out its whole timeout on a worker that has already finished, "
        "because the event that would stop the thread is sitting behind the "
        "wait. ``QThread::quit`` is explicitly thread-safe, so calling it inline "
        "from the worker thread is correct."
    ): (
        "``worker.finished -> thread.quit`` uses a **DirectConnection**. The "
        "QThread object is created on the GUI thread. That gives it GUI thread affinity. "
        "A queued ``quit()`` goes to the *GUI* event queue rather than "
        "the worker queue. The GUI thread immediately enters ``thread.wait()`` "
        "during the \"drain before closing\" behavior in "
        "``ConsolePanel.shutdown``. This blocks the event that would stop the worker and waits "
        "for the full timeout even though the worker has finished. "
        "``QThread::quit`` is thread-safe, so the worker thread may call it "
        "directly."
    ),
    (
        "That much was already known. What was *wrong* was the remedy: "
        "\"default it for measure the way the Mask app does\" does not fix "
        "this, and cannot, for two reasons that have to be fixed together."
    ): (
        "That much was already known. The proposed remedy copied measurement "
        "defaults from the Mask app. That remedy was *incorrect*: it cannot "
        "solve the problem, for two reasons that must be fixed together."
    ),
    (
        "1. ``set_default_settings_preprocess_generate_masks`` defaults it to "
        "the **string** ``'cell'``, and ``measure.py`` tests it with "
        "``\"organelle\" in settings['summarize_organelles_by']`` — a "
        "*substring* test when the value is a str. Running this demo with "
        "``summarize_organelles_by='cell'`` gives ``cell_organelle_summary`` "
        "(16 rows/field) and still **no ``organelle`` table**. Only a value "
        "containing ``'organelle'`` writes the per-organelle table "
        "(``['cell', 'organelle']`` → organelle 64 rows/field, verified). 2. A "
        "list cannot be shipped today: ``spacr.settings.expected_types`` "
        "declares ``'summarize_organelles_by': str``, so "
        "``spacr.validate.validate_settings`` rejects "
        "``['cell', 'organelle']`` with \"is a list, but str is expected\" — "
        "a hard pre-flight **error** on a demo that must load clean. The tooltip "
        "and ``spacr.gui_utils`` both describe it as a list, and "
        "``spacr.external_masks`` builds one; only the type table disagrees."
    ): (
        "1. **The default is a text value.** "
        "``set_default_settings_preprocess_generate_masks`` sets it to ``'cell'``. "
        "``measure.py`` evaluates "
        "``\"organelle\" in settings['summarize_organelles_by']`` for a str. "
        "*This is a partial-string match.* Running with "
        "``summarize_organelles_by='cell'`` produces "
        "``cell_organelle_summary`` (16 rows/field). **The table is absent.** The "
        "``organelle`` table appears only for a value containing ``'organelle'`` "
        "(``['cell', 'organelle']`` → organelle 64 rows/field, verified). 2. A "
        "list cannot be supplied currently. ``spacr.settings.expected_types`` "
        "declares ``'summarize_organelles_by': str``, so "
        "``spacr.validate.validate_settings`` rejects "
        "``['cell', 'organelle']`` with \"is a list, but str is expected\". "
        "**This is a validation error.** The demo must load cleanly. The tooltip "
        "and ``spacr.gui_utils`` describe a list, and ``spacr.external_masks`` "
        "builds one; the type table is the conflicting part."
    ),
    (
        "every metric on both sides, unchanged ones included — a count that did "
        "*not* move is evidence, unlike a setting that did not move."
    ): (
        "every metric on both sides, including unchanged metrics — an "
        "*unchanged* count is evidence, unlike an unchanged setting."
    ),
    (
        "the honest part. Every place this seed does **not** buy determinism, in "
        "plain sentences, so a caller can quote them rather than assume a "
        "guarantee that does not exist."
    ): (
        "the honest part: **known limits of deterministic behavior** for this seed, "
        "stated in plain sentences, so a caller can quote them instead of "
        "assuming a guarantee that does not exist."
    ),
    (
        "whether ``gain`` is larger than one standard error of the held-out "
        "accuracy itself. When it is *not*, \"flat\" and \"we cannot tell\" look "
        "identical from the numbers, and this says which you have."
    ): (
        "whether ``gain`` exceeds one standard error of validation accuracy. When "
        "the evidence is *inconclusive*, \"flat\" and "
        "\"we cannot tell\" look identical from the numbers, and this says which "
        "you have."
    ),
    (
        "A row is a disagreement when at least two annotators committed to a "
        "label and those labels are not all the same. A row where one annotator "
        "abstained is **not** a disagreement — by default it is simply scored on "
        "whoever did label it, and dropped entirely if that leaves fewer than two "
        "labels."
    ): (
        "A row is a disagreement when at least two annotators committed to a "
        "label and those labels are not all the same. A row where one annotator "
        "abstained is **different from a disagreement**. By default it is scored "
        "using the annotators who did label it, and dropped if fewer than two "
        "labels remain."
    ),
    "Every tile that did *not* register, worst confidence first.":
        "Images whose registration *failed*, ordered from lowest confidence to highest.",
    "A settings key auto-chaining did **not** touch, because the user owns it.":
        "A settings key **reserved by the user**, so auto-chaining leaves it unchanged.",
    (
        "Auto-chaining is only welcome while it is filling in a blank.  The "
        "moment a user types a path of their own, that path is theirs: it survives "
        "a reopen, a restart, and every subsequent upstream run.  This is the "
        "record that makes that true, and it is deliberately *not* the settings "
        "dict — a settings dict cannot distinguish \"the user chose this\" from "
        "\"we put it there\"."
    ): (
        "Auto-chaining fills only blank values. Once a user enters a path, that "
        "path belongs to the user: it survives reopening, restarting, and later "
        "upstream runs. This record preserves that ownership and is deliberately "
        "*kept outside* the settings dict. A settings dict cannot distinguish "
        "\"the user chose this\" from \"we put it there\"."
    ),
    (
        "a leftover ``<name>.spacr_v2`` staging file means the file next to it "
        "has **not** been converted yet -- it is still legacy;"
    ): (
        "a leftover ``<name>.spacr_v2`` staging file means the adjacent file "
        "**still awaits conversion** -- it remains legacy;"
    ),
    (
        "an existing database file.  It is made absolute but **not** "
        "tilde-expanded, so ``~/x.db`` is resolved under the working directory "
        "and raises ``FileNotFoundError`` even when the home-relative file exists.  "
        "A missing file raises the same and no database is created. The returned "
        "report carries the absolute path, not the string given."
    ): (
        "an existing database file. It is made absolute **without tilde "
        "expansion**, so ``~/x.db`` is resolved under the working directory and "
        "raises ``FileNotFoundError`` even when the file exists under the user's "
        "home folder. A missing file raises the same error and no database is created. The "
        "returned report carries the absolute path, not the supplied string."
    ),
    (
        "An artifact whose status *cannot be read* reads as **not** complete. "
        "That is the one case where the two answers differ in consequence: an "
        "unstamped file is silent, whereas a locked or truncated one is positive "
        "evidence that something was interrupted, and answering \"complete\" "
        "there is how a killed run passed for a finished one."
    ): (
        "An artifact whose status *cannot be read* is reported as **incomplete**. "
        "This is the one case where the two answers have different consequences: "
        "an unstamped file is silent, while a locked or truncated file is evidence "
        "of interruption. Reporting \"complete\" there would make a terminated "
        "run appear finished."
    ),
    (
        "the warnings that do *not* come from the column mapping (unpaired masks, "
        "the join, z handling). Kept apart so :meth:`with_column_maps` can rebuild "
        "the mapping's own warnings without losing them or duplicating them."
    ): (
        "the warnings *outside* the column mapping (unpaired masks, the join, z "
        "handling). They remain separate so :meth:`with_column_maps` can rebuild "
        "the mapping warnings without losing or duplicating them."
    ),
    (
        "``(settings, defaulted_keys, defaults_source)`` where "
        "``defaulted_keys`` are the keys the caller did **not** set and the script "
        "is therefore pinning on their behalf."
    ): (
        "``(settings, defaulted_keys, defaults_source)`` where "
        "``defaulted_keys`` contains the keys the caller **omitted** and the "
        "script therefore supplies for the caller."
    ),
    "Does **not** start an event loop: see :func:`run_event_loop` for why that is a separate decision.":
        "**Event-loop startup remains a caller decision**: details are in :func:`run_event_loop`.",
    (
        "Only Extra Performance does this, and it is deliberately **not** the "
        "same cleanup as at launch:"
    ): (
        "Only Extra Performance performs this, and it is deliberately "
        "**different from** cleanup at launch:"
    ),
    (
        "Deliberately does **not** chain to ``super()`` when it fills. The base "
        "implementation is what draws the stylesheet background, and the "
        "stylesheet background is the ``bg`` slab being replaced — calling it "
        "afterwards would paint black straight back over this."
    ): (
        "The call to ``super()`` is deliberately **omitted** when it fills. The base "
        "implementation draws the stylesheet background, which is the ``bg`` slab "
        "being replaced; calling it afterwards would paint black over this fill."
    ),
    (
        "The one call a host needs. It deliberately does **not** fit: which "
        "column is the dose is not guessable from a measurement table, and a curve "
        "through the wrong pair of columns is worse than an empty axis."
    ): (
        "The one call a host needs. It deliberately **leaves fitting to the "
        "caller**: the dose column cannot be inferred safely from a measurement "
        "table, and fitting the wrong pair of columns is worse than an empty axis."
    ),
    (
        "The screen is deliberately thin. Everything it knows about a run folder "
        "it learns from :mod:`spacr.report`, which is headless, read-only and "
        "testable without Qt. This file is the part that has to be a GUI: pick a "
        "folder, say what was found and — just as loudly — what was **not**, choose "
        "a format, and write the file off the GUI thread."
    ): (
        "The screen intentionally contains little logic. It learns about a run "
        "folder from :mod:`spacr.report`, which is headless, read-only, and "
        "testable without Qt. The GUI must choose a folder, "
        "clearly report **both present and missing results**, choose a format, and "
        "write the file outside the GUI thread."
    ),
    (
        "**Density is linear**, as it should be: 300 % is 2.7-3.0 times the "
        "shading on every theme. What it is *not* is brighter — see "
        ":meth:`AmbientEngine.alpha_scale` for why three times the elements at a "
        "third of the alpha is the only reading of the control that leaves the "
        "backdrop legible at both ends of its range."
    ): (
        "**Density is linear**, as it should be: 300 % produces 2.7-3.0 times the "
        "shading on every theme. The control changes *density rather than "
        "brightness* — see :meth:`AmbientEngine.alpha_scale` for why three times "
        "the elements at one third the alpha keeps the backdrop legible across "
        "the range."
    ),
    (
        "The controls are **not** placed on the screen. They live in a popover "
        "behind a ``DNA`` toggle built from the same class as the ``AI`` toggle "
        "beside it; a decorative backdrop does not get to keep a permanent strip "
        "of a screen whose job is a settings form."
    ): (
        "The controls are **kept off the main screen**. They appear in a popover "
        "opened by a ``DNA`` toggle built from the same class as the adjacent "
        "``AI`` toggle. A decorative backdrop should not permanently occupy a "
        "strip of a screen used for a settings form."
    ),
    (
        "**Not** ``column_kinds() == CONTINUOUS``, and the difference matters. "
        ":func:`~spacr.qt.widgets.data_filter_panel.classify_columns` calls a "
        "numeric column with twelve or fewer distinct values a *category*, which "
        "is the right rule for deciding whether to offer a slider or a tick list "
        "— and the wrong one here, because ``pathogen_count`` runs 0–8 and is "
        "exactly the kind of feature a ranking exists to surface. A separation "
        "statistic is perfectly happy on a discrete count."
    ): (
        "**This differs from** ``column_kinds() == CONTINUOUS``, and the "
        "distinction matters. "
        ":func:`~spacr.qt.widgets.data_filter_panel.classify_columns` calls a "
        "numeric column with twelve or fewer distinct values a *category*. That "
        "rule correctly chooses between a slider and a tick list, but it is wrong "
        "here: ``pathogen_count`` runs 0–8 and is exactly the feature a ranking "
        "should surface. A separation statistic works with a discrete count."
    ),
    (
        "So the well pass is emphatically **not** \"the well contains many flagged "
        "objects\". That statistic exists — it is reported as ``flagged_share`` — "
        "and it answers a different question: it finds a well containing a few "
        "*catastrophic* objects (a segmentation blow-up, a piece of dust measured "
        "as a cell) while being blind to a uniform shift. The well-level robust "
        "score finds the uniform shift while being blind to the isolated "
        "catastrophe. Both are on the well frame because neither subsumes the other."
    ): (
        "The well pass tests something emphatically **different from** \"the well "
        "contains many flagged objects\". That statistic is reported as "
        "``flagged_share`` and answers another question: it finds a well "
        "containing a few *catastrophic* objects (a segmentation failure or dust "
        "measured as a cell) but misses a uniform shift. The well-level robust "
        "score finds the uniform shift but misses the isolated catastrophe. Both "
        "belong on the well frame because neither replaces the other."
    ),
    (
        "The table lists the fields that are *not* clean — on a good plate that "
        "is no rows at all, and on a bad one it is the list you want. Clean fields "
        "are counted, not printed: a 1536-field plate must not scroll a terminal."
    ): (
        "The table lists fields *with problems* — no rows on a good plate, and the "
        "useful problem list on a bad plate. Clean fields are counted rather than "
        "printed: a 1536-field plate must not scroll a terminal."
    ),
    (
        "A folder that is missing its curves, missing its settings or holding a "
        "zero-epoch log does **not** raise: the run comes back with whatever could "
        "be read and a :attr:`TrainingRun.notes` entry per problem, so one bad "
        "folder in a scan cannot stop the rest being compared."
    ): (
        "A folder missing its curves or settings, or containing a zero-epoch log, "
        "**returns partial data instead of raising**. The run contains whatever "
        "could be read plus one :attr:`TrainingRun.notes` entry per problem, so one "
        "bad folder cannot stop comparison of the others."
    ),
    (
        "opt in to collapsing z before linking, so that linking happens on the "
        "projection rather than on the volume. Off by default and never implied. "
        "It does **not** unlock the backends spaCR cannot drive volumetrically -- "
        "see :func:`track_4d`."
    ): (
        "opt in to collapsing z before linking, so linking occurs on the "
        "projection rather than the volume. Off by default and never implied. "
        "spaCR's **volumetric backend support remains unchanged**. Details: "
        ":func:`track_4d`."
    ),
    (
        "**The run happens off the GUI thread, and its completion handler comes "
        "back onto it.** ``PipelineWorker.finished`` is emitted *in the worker "
        "thread*; every widget-mutating receiver therefore uses an explicit queued "
        "connection to a bound method of this GUI-thread widget."
    ): (
        "The run uses a worker. **Execution model:** completion returns to the GUI "
        "thread. ``PipelineWorker.finished`` is emitted *inside the worker thread*; "
        "therefore each receiver that changes a widget uses an explicit queued "
        "connection to a bound method of this GUI-thread widget."
    ),
    (
        "**Off the GUI thread.** Scanning a plate's worth of ND2 headers takes "
        "seconds; converting takes minutes. Both go through "
        ":func:`spacr.qt.bridge.make_thread`, and the completion handler is reached "
        "through a *bound method* (:attr:`ConvertScreen._job_settled`) rather than "
        "a closure, because ``PipelineWorker.finished`` is emitted in the worker "
        "thread and a closure connected to it would build widget children there. "
        "Tests pass ``threaded=False``."
    ): (
        "**Execution model:** scanning ND2 headers takes seconds and conversion "
        "takes minutes, so both stay off the GUI thread. Both operations use "
        ":func:`spacr.qt.bridge.make_thread`. Completion reaches a *bound method*, "
        ":attr:`ConvertScreen._job_settled`, rather than a closure. "
        "``PipelineWorker.finished`` is emitted in the worker thread; a connected "
        "closure would create widget children there. Tests pass ``threaded=False``."
    ),
    (
        "**Off the GUI thread, and the thread actually retires.** A full sweep is "
        "minutes. It goes through :func:`spacr.qt.bridge.make_thread`, and every "
        "``thread.finished`` slot is a BOUND METHOD — see "
        ":meth:`PowerScreen._retire_finished_jobs` for what a closure does here and "
        "why it is not a style preference."
    ): (
        "**Execution model:** a full sweep takes minutes, so it runs outside the GUI "
        "thread and the thread must retire. It uses "
        ":func:`spacr.qt.bridge.make_thread`, and each ``thread.finished`` slot is a "
        "BOUND METHOD. :meth:`PowerScreen._retire_finished_jobs` explains why a "
        "closure is incorrect here."
    ),
}

# Longer blocks can be grammatically complete yet still cause the compact
# OPUS checkpoints to echo an English loanword (especially ``default``,
# ``thread`` and ``string``).  These target-neutral rewrites keep the exact API
# literals while expressing the same idea with less ambiguous source prose.
API_TRANSLATION_CONTEXT.update({
    (
        "The midpoint between chance and certainty is the honest default: 0.75 "
        "for two classes, 0.55 for ten. It is a *default*, not a law — every "
        "function here takes an explicit ``threshold``, and the screen exposes it, "
        "because where \"sure\" starts is a property of the assay and not of "
        "arithmetic."
    ): (
        "Use a balanced starting threshold: 0.75 for two classes and 0.55 for ten. "
        "It is a *starting point*, not a law. Each function accepts an explicit "
        "``threshold``, and the screen exposes it because the assay determines "
        "where \"sure\" begins."
    ),
    "optional root frame to color-match.":
        "optional root frame for coordinated colors.",
    (
        "So :func:`apply` has a second phase. Every registered app that appears in "
        "none of the three assessment tables below and carries no explicit stage "
        "is written into ``APP_STAGE`` as :data:`UNASSESSED_STAGE` — alpha — which "
        "is what \"nobody has checked this one yet\" means. It is a *default*, not "
        "a demotion: a module that declares beta keeps beta, and an assessment "
        "recorded here always wins over both."
    ): (
        "So :func:`apply` has a second phase. Each registered app absent from all "
        "three assessment tables and carrying no explicit stage is written to "
        "*``APP_STAGE``* as :data:`UNASSESSED_STAGE` — alpha, meaning \"nobody "
        "has checked this one yet\". This is a fallback-stage rule, not a "
        "demotion. A module declaring beta keeps beta, and a recorded assessment "
        "takes precedence."
    ),
    (
        "``db_browser_editable``: bool, default ``False``. Permits the Database "
        "Browser to open a read-write connection at all; see "
        ":func:`get_db_browser_editable`."
    ): (
        "``db_browser_editable``: bool, initial value ``False``. Permits the "
        "Database Browser to open a read-write connection. Details: "
        ":func:`get_db_browser_editable`."
    ),
    (
        "``pane_opacity``: int percent, default ``60``. How solid shared surfaces "
        "are, or the relative material strength in Glass. Clamped up to "
        ":func:`spacr.qt.theme.pane_alpha_floor` at paint time — the preference is "
        "a request, legibility is not negotiable."
    ): (
        "``pane_opacity``: int percent, initial value ``60``. Controls the solidity "
        "of shared surfaces, or material strength in Glass. At paint time "
        ":func:`spacr.qt.theme.pane_alpha_floor` sets a lower bound: the preference "
        "is a request, while legibility is mandatory."
    ),
    (
        "``ambient_enabled``: bool, default ``True``. Whether module screens paint "
        "the animated background at all. Turning it off is a first-class choice — "
        "see :func:`get_ambient_enabled`. The user-facing control is the ``None`` "
        "entry in the Animation list rather than a second switch: one row, one "
        "meaning. Choosing an animation turns it back on."
    ): (
        "``ambient_enabled``: bool, initial value ``True``. Controls whether module "
        "screens paint the animated background. Disabling it is a first-class "
        "choice; details: :func:`get_ambient_enabled`. The user-facing control is "
        "the ``None`` entry in the Animation list rather than another switch: one "
        "row, one meaning. Choosing an animation enables it again."
    ),
    (
        "**Off the GUI thread.** Every chunk, count and export goes through "
        ":func:`spacr.qt.bridge.make_thread`, the same helper the pipeline screens "
        "use, and each worker opens (and closes) its **own** sqlite connection — "
        "``sqlite3`` objects are not shareable across threads. Jobs queue and run "
        "**one at a time**; see :meth:`DbBrowserScreen._run_job` for why two "
        "``PipelineWorker``\\ s must not overlap."
    ): (
        "**Background execution.** Each chunk, count, and export runs outside the "
        "GUI thread through :func:`spacr.qt.bridge.make_thread`, also used by "
        "workflow screens. Each worker opens and closes **a separate** sqlite "
        "connection; ``sqlite3`` objects cannot be shared between execution "
        "threads. Jobs run **sequentially**. :meth:`DbBrowserScreen._run_job` "
        "explains why two ``PipelineWorker``\\ s cannot overlap."
    ),
    (
        "GREYED, never removed. INVARIANTS 6: a key ABSENT from the settings dict "
        "makes the pipeline fall back to its own default, which can differ from the "
        "value the module needs and says nothing when it does. A disabled widget "
        "keeps its value and still collects; it just stops being editable."
    ): (
        "GREYED, never removed. INVARIANT 6: when a key is missing from the settings "
        "dict, the workflow uses its own initial value. That value may differ from "
        "the module requirement without a warning. A disabled widget retains its "
        "value and remains collectable; only editing stops."
    ),
    (
        "Deliberately conservative. A key qualifies only when its *default* is "
        "already a list or tuple, or is ``None`` and the declared type admits "
        "nothing but a list. That keeps three groups of keys on their old widgets:"
    ): (
        "Deliberately conservative. A key qualifies only when its *initial value* "
        "is already a list or tuple, or is ``None`` and the declared type permits "
        "only a list. This keeps three groups of keys on their previous widgets:"
    ),
    (
        "``count_data`` / ``score_data``, declared ``list`` but shipped with the "
        "placeholder *string* ``'list of paths'``;"
    ): (
        "``count_data`` / ``score_data``, declared ``list`` but shipped with the "
        "*text marker* ``'list of paths'``;"
    ),
    (
        "**Off the GUI thread.** Cheap is not free, and opening the picker is four "
        "sequential sqlite round trips — open, list tables, read one table's "
        "columns, estimate its rows — every one of which used to happen inside "
        "``__init__``, before the modal appeared. Measured cold on a 383 MB "
        "measurements.db that is 45 ms, and on a 1 500-table schema 87 ms, entirely "
        "between the click and any window. The button now builds the dialog with "
        "``threaded=True`` and the reads arrive from a "
        ":class:`~spacr.qt.job_runner.JobRunner`. The default is still the "
        "synchronous mode, deliberately; :class:`ColumnPickerDialog` says why."
    ): (
        "**Background loading.** Opening the picker away from the GUI event loop "
        "performs "
        "four sequential sqlite queries: open, list tables, read one table's "
        "columns, and estimate its rows. These used to run in ``__init__`` while no "
        "dialog was visible. Cold measurements were 45 ms for a 383 MB "
        "measurements.db and 87 ms for a 1 500-table schema. The button now "
        "constructs the dialog with ``threaded=True`` and reads through "
        ":class:`~spacr.qt.job_runner.JobRunner`. Synchronous mode remains the "
        "initial choice; :class:`ColumnPickerDialog` explains why."
    ),
    (
        "``threaded=True``: this is the user-facing path, and the dialog it builds "
        "is run modally by :meth:`open_picker`, so every millisecond the constructor "
        "spends in sqlite is a millisecond between the click and any window at all. "
        "See :class:`ColumnPickerDialog` for the two modes and why the *default* is "
        "the other one."
    ): (
        "``threaded=True`` selects the user-facing path. :meth:`open_picker` runs "
        "the resulting dialog modally, so constructor time directly delays the "
        "first visible window. :class:`ColumnPickerDialog` describes both modes "
        "and explains why the *initial choice* is synchronous."
    ),
    (
        "the columns to test. Empty means :func:`candidate_features` of whatever "
        "frame it runs against — a *default*, not a promise; the result records "
        "what it actually used."
    ): (
        "the columns to test. Empty uses :func:`candidate_features` for the current "
        "frame — a *starting selection*, not a promise. The result records the "
        "columns actually used."
    ),
    (
        "the columns to decompose. Empty means :func:`candidate_features` of "
        "whatever frame it is run against — which is a *default*, not a promise: "
        "the result records the features it actually used."
    ): (
        "the columns to decompose. Empty uses :func:`candidate_features` for the "
        "current frame — a *starting selection*, not a promise. The result records "
        "the features actually used."
    ),
    "Toggle — QCheckBox styled as an iOS-style switch.":
        "Switch control — QCheckBox styled like an iOS switch.",
    (
        "State is thread-local because the batch runner and database browser can "
        "execute independent workers concurrently."
    ): (
        "Each execution worker has isolated state because the batch runner and "
        "database browser can operate independent workers concurrently."
    ),
    (
        "only ever prints for STRING cluster labels; silent for the integer labels "
        "DBSCAN and KMeans produce."
    ): (
        "prints only for text cluster labels; silent for integer labels produced by "
        "DBSCAN and KMeans."
    ),
    (
        "prints the label and its index for STRING cluster labels only; integer "
        "labels never print anything."
    ): (
        "prints the label and its index only for text cluster labels; integer labels "
        "never print anything."
    ),
    (
        "opt in to collapsing z before linking, so that linking happens on the "
        "projection rather than on the volume. Off by default and never implied. "
        "It does **not** unlock the backends spaCR cannot drive volumetrically -- "
        "see :func:`track_4d`."
    ): (
        "Collapse z first so linking uses the projection instead of the volume. "
        "Disabled initially and never implied. spaCR's **volumetric backend "
        "compatibility stays the same**. Details: :func:`track_4d`."
    ),
})


def _rewrite_unprotected_prose(
    text: str, rewrite,
) -> str:
    """Apply ``rewrite`` only outside API/RST literals, losslessly."""
    source = str(text)
    pieces: list[str] = []
    cursor = 0
    for match in _CONTEXT_HARD_PROTECT_RE.finditer(source):
        pieces.append(rewrite(source[cursor:match.start()]))
        pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(rewrite(source[cursor:]))
    return "".join(pieces)


def _replace_words(text: str, pattern: str, replacement: str) -> str:
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


def _initial_case(match: re.Match[str], replacement: str) -> str:
    """Give an English sense expansion the source token's initial case."""
    if match.group(0)[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _first_unprotected_ascii_letter(text: str) -> int | None:
    """Return the first English prose-letter offset outside hard literals."""
    source = str(text)
    cursor = 0
    for protected in _CONTEXT_HARD_PROTECT_RE.finditer(source):
        found = re.search(r"[A-Za-z]", source[cursor:protected.start()])
        if found is not None:
            return cursor + found.start()
        cursor = protected.end()
    found = re.search(r"[A-Za-z]", source[cursor:])
    return None if found is None else cursor + found.start()


def _preserve_initial_prose_case(source: str, rendered: str) -> str:
    """Keep a sentence-leading capital after a sense-neutral paraphrase."""
    source_offset = _first_unprotected_ascii_letter(source)
    rendered_offset = _first_unprotected_ascii_letter(rendered)
    if source_offset is None or rendered_offset is None:
        return rendered
    if source[source_offset].isupper() and rendered[rendered_offset].islower():
        return (
            rendered[:rendered_offset]
            + rendered[rendered_offset].upper()
            + rendered[rendered_offset + 1:]
        )
    return rendered


def _replace_alternatives_once(
    text: str, alternatives: Iterable[tuple[str, object]],
) -> str:
    """Apply ordered alternatives to the original text without cascading.

    Sense-expanding replacements deliberately contain ordinary English words
    that may also be covered by a later alternative (for example ``FOVs`` ->
    ``FOVs (microscope fields of view)`` while the same family also rewrites
    bare ``fields``).  Sequential ``re.sub`` calls would rewrite the newly
    inserted explanation a second time.  This scanner selects the earliest
    source match (then the declared order for ties) and never scans rendered
    replacement text.
    """
    source = str(text)
    compiled = [
        (re.compile(pattern, re.IGNORECASE), replacement)
        for pattern, replacement in alternatives
    ]
    rendered: list[str] = []
    cursor = 0
    while cursor < len(source):
        choices: list[tuple[int, int, re.Match[str], object]] = []
        for order, (pattern, replacement) in enumerate(compiled):
            match = pattern.search(source, cursor)
            if match is not None:
                choices.append((match.start(), order, match, replacement))
        if not choices:
            rendered.append(source[cursor:])
            break
        _start, _order, match, replacement = min(
            choices, key=lambda item: (item[0], item[1])
        )
        rendered.append(source[cursor:match.start()])
        rendered.append(
            str(replacement(match)) if callable(replacement)
            else match.expand(str(replacement))
        )
        cursor = match.end()
    return "".join(rendered)


# Empirically hard blocks from the strict Portuguese repair pass.  These are
# target-neutral English paraphrases, not target-language answers: they reduce
# ambiguity and length while retaining every source-side API/RST literal.  The
# resulting model output still has to pass both contextual and canonical gates.
API_TRANSLATION_CONTEXT.update({
    (
        "``src``/``merged/*.npy`` — spaCR's own merged arrays, which carry the "
        "image channels *and* the object label planes in one file. Preferred "
        "because the object mask comes free and exactly aligned, which is what "
        "makes the pointing game possible at all."
    ): (
        "``src``/``merged/*.npy`` contains spaCR's merged arrays. One file "
        "contains both image channels *and* object-label planes. Prefer it "
        "because the included object mask is exactly aligned, which enables "
        "the pointing task."
    ),
    (
        'detector : {"ORB","SIFT"} Feature detector for keypoint matching. '
        "nfeatures : int Feature budget for detector. max_keypoints : "
        "Optional[int] Hard cap on kept keypoints after detection (by "
        "detector’s internal ranking). downsample : float in (0,1] Downsample "
        "factor for feature/score pass. ransac_thresh_px : float Reprojection "
        "threshold (pixels) for affine estimation (downsampled space). "
        "allow_scale : bool If False, constrain to rotation+translation (or "
        "translation only if allow_rotation=False). allow_rotation : bool If "
        "False, constrain to translation only. outdir : str Output directory "
        "for images/csv. opencv_threads : int Limit OpenCV internal threading "
        "(avoid oversubscription)."
    ): (
        'detector : {"ORB","SIFT"} Keypoint detector. nfeatures : int Detector '
        "feature budget. max_keypoints : Optional[int] Hard cap on keypoints "
        "kept after detection, using the detector’s ranking. downsample : float "
        "in (0,1] Factor for feature detection and scoring. "
        "ransac_thresh_px : float Pixel reprojection threshold for affine "
        "estimation in downsampled space. allow_scale : bool If False, permit "
        "rotation and translation but no scaling; if allow_rotation=False, "
        "permit translation only. allow_rotation : bool If False, permit "
        "translation only. outdir : str Directory for image and csv output. "
        "opencv_threads : int OpenCV thread limit used to prevent "
        "oversubscription."
    ),
    (
        "**Crop names match.** A real crop is "
        "``<file_name>_<cell_id>.png`` where ``file_name`` is the merged "
        "stack's ``<plate>_<well>_<field>_<time>`` "
        "(:func:`spacr.utils._generate_names`) — e.g. "
        "``plate1_A01_1_1_1.png``. That is exactly what this writes, and "
        "exactly what ``spacr.utils._map_wells_png`` parses "
        "plate/row/column/field back out of."
    ): (
        "**Image-crop names use the production format.** Each crop is "
        "``<file_name>_<cell_id>.png``. Its ``file_name`` contains "
        "``<plate>_<well>_<field>_<time>`` "
        "(:func:`spacr.utils._generate_names`), for example "
        "``plate1_A01_1_1_1.png``. This function writes that format, and "
        "``spacr.utils._map_wells_png`` reads back its four parts: plate, row, "
        "column, and field."
    ),
    (
        "A spaCR run carries around two hundred keys, so an ungrouped diff of "
        "two runs that differ in one Cellpose knob and one plate-map column "
        "reads as an undifferentiated list. Grouping under the same headings "
        "the settings panel uses makes it answerable at a glance: *the change "
        "was in Cellpose*."
    ): (
        "A spaCR execution has about two hundred setting names. An ungrouped "
        "comparison of two executions can mix one Cellpose setting with one "
        "laboratory plate-map column. Group the differences under the "
        "settings-panel headings so the user can see at once: *the change was "
        "in Cellpose*."
    ),
    "the caught exception, chained onto the raise.": (
        "the caught exception, attached as the cause of the newly raised "
        "exception."
    ),
    (
        "1. **A replicate whose fit failed or did not converge counts as a "
        "non-detection**, not as a missing value. Dropping it would raise the "
        "reported power by removing exactly the runs where the design was too "
        "thin to fit — which is the failure mode the analysis is supposed to "
        "find. 2. **The mean AUROC is reported beside the power**, over the "
        "replicates that did converge, with the count of those that did not. A "
        "power of 0/5 with five non-converged fits and a power of 0/5 with five "
        "converged fits at AUROC 0.52 are different findings."
    ): (
        "1. **Count failed or non-converged fits as non-detections.** Do not "
        "drop them: that would inflate statistical power by removing "
        "executions whose design was too weak to fit. 2. **Report mean AUROC "
        "beside statistical power.** Calculate the mean over converged fits "
        "and also report how many fits failed or did not converge. Thus power "
        "0/5 with five non-converged fits differs from power 0/5 with five "
        "converged fits at AUROC 0.52."
    ),
    (
        "1. :func:`spacr.qt.preferences.get_db_browser_editable` must be on — "
        "it is off by default and lives in Preferences, not on this screen. "
        "2. The database must have been chosen explicitly in this session "
        "(``set_database(..., explicit=True)``). 3. The user must tick \"Edit "
        "mode\" *and* confirm; ticking alone does nothing. 4. The row must be "
        "addressable by ``rowid`` or a primary key. Without one, the edit is "
        "refused — an UPDATE matching on column values can hit many rows, "
        "which on a measurements table is silent mass corruption. The write "
        "also probes ``COUNT(*)`` for the row address first and rolls back "
        "unless ``rowcount == 1``. 5. The typed text must be coercible to the "
        "column's declared type. SQLite will cheerfully store ``'abc'`` in an "
        "INTEGER column; :func:`coerce_for_column` refuses instead."
    ): (
        "Editing is allowed only when all five checks pass. 1. "
        ":func:`spacr.qt.preferences.get_db_browser_editable` is enabled in "
        "Preferences; it is off by default and is not on this screen. 2. This "
        "session explicitly selected the database with "
        "``set_database(..., explicit=True)``. 3. The user selected \"Edit "
        "mode\" *and* confirmed it; selection alone does nothing. 4. The row "
        "has ``rowid`` or a primary-key identifier. Otherwise refuse the edit: "
        "matching column values could silently update many measurement rows. "
        "Before writing, probe ``COUNT(*)`` and roll back unless "
        "``rowcount == 1``. 5. The text converts to the declared column type. "
        "SQLite accepts ``'abc'`` in an INTEGER column, but "
        ":func:`coerce_for_column` must refuse it."
    ),
    (
        "1. **Flatten.** Denoise with a 1 px Gaussian, then subtract a heavily "
        "smoothed copy (sigma = max(32, min(H, W) / 4)) to remove illumination "
        "gradients. The sigma is deliberately far larger than any plausible "
        "object so that flattening removes vignetting without eating the "
        "objects — the opposite trade-off (a tight rolling ball) shrinks what "
        "it is trying to measure. 2. **Reject noise.** Compare the structural "
        "amplitude of the flattened plane (p99 - p30) against the *pixel-level* "
        "noise scale, measured as ``1.4826 * MAD(img - gaussian(img, 1))`` on "
        "the raw plane. A pure-noise plane scores below 1; a plane with real "
        "objects scores in the tens. Below ``min_snr`` the field is discarded "
        "rather than thresholded, because Otsu will happily bisect pure noise "
        "and hand back a confident-looking number. 3. **Threshold and label.** "
        "Otsu, fill holes, label, drop components that touch the image border "
        "(truncated, so their size is a lie) and components that are absurd "
        "(equivalent diameter below ``min_object_diameter``, or area above "
        "``max_object_fraction`` of the field). Characteristic size is the "
        "median equivalent diameter, ``2 * sqrt(area / pi)``. 4. **Cross-check "
        "by distance transform.** Step 3 has one dominant failure mode: a "
        "confluent monolayer fuses into a single component, that component "
        "touches the border and is dropped, and the estimate is then computed "
        "from whatever debris survived — biased **low**, and silently. So the "
        "Euclidean distance transform of the (unfilled) foreground is computed "
        "as well, its local maxima are taken as one seed per object (two "
        "passes: a coarse pass sets the suppression radius for the refined "
        "pass), and a watershed on ``-EDT`` splits the fused foreground back "
        "into objects whose equivalent diameters are measured the same way."
    ): (
        "1. **Flatten.** Apply a 1 px Gaussian filter, then subtract a smoother "
        "copy with sigma = max(32, min(H, W) / 4). This sigma is larger than a "
        "plausible object, so it removes vignetting without the object "
        "shrinkage caused by a tight rolling ball. 2. **Reject noise.** Compare "
        "the flattened plane’s structural amplitude (p99 - p30) with the "
        "*pixel-level* raw-plane noise estimate "
        "``1.4826 * MAD(img - gaussian(img, 1))``. Pure noise scores below 1; "
        "real objects score in the tens. Discard a field below ``min_snr`` "
        "instead of applying Otsu to pure noise. 3. **Threshold and label.** "
        "Apply Otsu, fill holes, and label components. Remove border-touching "
        "components because they are truncated; also remove equivalent "
        "diameters below ``min_object_diameter`` and areas above "
        "``max_object_fraction``. Use the median equivalent diameter, "
        "``2 * sqrt(area / pi)``. 4. **Cross-check by distance transform.** A "
        "confluent layer can fuse into one border-touching component, which is "
        "discarded and leaves debris that biases the estimate silently "
        "**low**. Compute the Euclidean distance transform of the unfilled "
        "foreground. Use local maxima as object seeds in two passes: a coarse "
        "pass sets the suppression radius for the refined pass. Watershed on "
        "``-EDT`` separates the fused foreground before measuring equivalent "
        "diameters the same way."
    ),
})


def _api_translation_source(block: str) -> str:
    """Return the deterministic, target-neutral English model input.

    The translation model sees a complete sentence/block, but ambiguous spaCR
    domain terms are expressed with their intended English sense.  No target
    language text is authored or substituted here; output still has to pass
    every canonical semantic gate.  Protected code/RST literals are exact.
    """
    source = str(block)
    reviewed = API_TRANSLATION_CONTEXT.get(source)
    if reviewed is not None:
        return reviewed

    prose = _context_prose(source)
    # GUI evidence can itself be a protected dotted path. Inspect the full
    # source before the protection view blanks it; otherwise a paragraph that
    # names ``spacr.qt.widgets`` can be misrouted as a scientific screen.
    has_gui_screen = _gui_screen_source(source)
    has_scientific_screen = bool(
        re.search(_SCIENTIFIC_SCREEN_SOURCE, prose, re.IGNORECASE)
    )
    total_wells, scientific_wells = _english_well_sense_counts(prose)
    exception_raises, _quantitative_raises, _window_raises = (
        _raise_sense_counts(prose)
    )
    transforms: list[tuple[str, str]] = []
    if re.search(_PLANE_SOURCE, prose, re.IGNORECASE):
        transforms.extend((
            # Match the combined crop phrase at the same source offset as
            # ``plane``.  The non-cascading scanner otherwise consumes the
            # earlier noun first and leaves an awkward "image layer to crop
            # by" fragment for the later crop family.
            (r"\b(?:mask\s+)?plane\s+to\s+crop\s+by\b", "mask image layer used to extract the image region"),
            (r"\ba\s+one[- ]plane\b", "a single-image-layer"),
            (r"\bone[- ]plane(?=\s+(?:list|image|stack|array|mode|case)\b)", "single-image-layer"),
            (r"\bone[- ]plane\b", "one image layer"),
            (r"\b(\d+)[- ]plane\b", r"\1-image-layer"),
            (r"\bper-plane\b", lambda m: _initial_case(m, "per-image-layer")),
            (r"\bper plane\b", lambda m: _initial_case(m, "for each image layer")),
            (r"\bz[- ]plane\b", "z image layer"),
            (r"\bmask[- ]plane\b", "mask image layer"),
            (r"\ba\s+plane\b", "an image layer"),
            (r"\bimage\s+planes\b", "image layers"),
            (r"\bimage\s+plane\b", "image layer"),
            (r"\bplanes\b", lambda m: _initial_case(m, "image layers")),
            (r"\bplane\b", lambda m: _initial_case(m, "image layer")),
        ))
    if re.search(_COMPUTE_RUN_SOURCE, prose, re.IGNORECASE):
        transforms.extend((
            (r"\blong[- ]running\b", "long-running"),
            (r"\b(a|an)\s+running\b", lambda m: _initial_case(m, "an executing")),
            (r"\b(has|have|had)\s+re[- ]run\s+([^,.;:]+?)(?=\s+(?:and|but|so)\b|[,.;:]|$)", r"\1 executed \2 again"),
            (r"\b(had)\s+([^,.;:]+?)\s+re[- ]run\s+([^,.;:]+?)(?=\s+(?:and|but|so)\b|[,.;:]|$)", r"\1 \2 executed again \3"),
            (r"\b(do|does|did)\s+not\s+run\b", r"\1 not execute"),
            (r"\b(has|have|had)\s+to\s+be\s+re[- ]run\b", r"\1 to be executed again"),
            (r"\b(can|cannot|could|will|would|should|must|may|might)\s+(simply\s+)?be\s+re[- ]run\b", lambda m: m.group(1) + " " + (m.group(2) or "") + "be executed again"),
            (r"\b(is|are|was|were|be|been|being)\s+re[- ]run\b", r"\1 executed again"),
            (r"\b(is|are|was|were)\s+run[- ]journal\s+runs\b", r"\1 processing-session journal folders"),
            (r"\b(has|have|had)\s+re[- ]run\b", r"\1 executed again"),
            (r"\bre[- ]run\s+(it|them)\b", r"execute \1 again"),
            (r"\bre[- ]run\s+(the|this|that|a|an)\s+([^,.;:]+)", r"execute \1 \2 again"),
            (r"\bre[- ]run\s+jobs?\s+that\s+failed\b", "repeat failed jobs"),
            (r"\bre[- ]run\s+(jobs?|modules?|pipelines?|workflows?|masks?)\b", r"execute \1 again"),
            (r"\bre[- ]run(?=\s+(?:button|control|action|command|path|workflow)\b)", "repeat-execution"),
            (r"\bre[- ]run(?=\s+(?:produces?|creates?|writes?|returns?)\b)", "repeat execution"),
            (r"\b(a|an|the|this|that)\s+re[- ]run\b", r"\1 repeat execution"),
            (r"\bre[- ]run\s+again\b", "executed again"),
            (r"\b(partly|partially)\s+re[- ]run\b", r"\1 executed again"),
            (r"\bbeen\s+re[- ]run\b", "been executed again"),
            (r"\bwas\s+re[- ]run\b", "was executed again"),
            (r"\bwere\s+re[- ]run\b", "were executed again"),
            (r"\bis\s+re[- ]run\b", "is executed again"),
            (r"\bare\s+re[- ]run\b", "are executed again"),
            (r"\bre[- ]run\s+it\b", "execute it again"),
            (r"\bre[- ]run\s+them\b", "execute them again"),
            (r"\bre[- ]run\b", "execute again"),
            (r"\b(?:is|are|was|were|be|been|being)\s+run\b", lambda m: m.group(0)[:-3] + "executed"),
            (r"\b((?:used\s+)?to|can(?:not|'t)?|could|will|would|should|must|may|might|do|does|did)\s+run\b", r"\1 execute"),
            (r"\b(?:deliberately\s+)?not\s+run\b", lambda m: m.group(0)[:-3] + "executed"),
            (r"\bnever\s+run\b", "never executed"),
            (r"\b(?-i:Run)(?=\s+(?:the|this|that|a|an|one|each|every|before|after|on|in|with|without|preprocessing|analysis|inference|training|classification|segmentation|jobs?|modules?|pipelines?|workflows?|commands?)\b)", "Execute"),
            (r"\bpipeline\s+runs\b", "workflow processing sessions"),
            (r"\bpipeline\s+run\b", "workflow processing session"),
            (r"\b(jobs?|modules?|functions?|callbacks?|code|pipelines?|workflows?|queues?|applications?|workers?|everything|nothing|it|this|that)\s+runs\b", r"\1 executes"),
            (r"\b(pipelines?|workflows?)\s+(that\s+call\s+this)\s+run\b", r"\1 \2 execute"),
            (r"\b(pipelines?|workflows?)\s+(?:that\s+)?run\b", r"\1 execute"),
            (r"\b(\d+)\s+runs\b", r"\1 executions"),
            (r"\bruns\s+(?=(?:again|inline|sequentially|successfully|unchanged|on|inside|under|through|against|before|after|when|if|unless|without)\b)", lambda m: _initial_case(m, "executes ")),
            (r"\brunning\b", lambda m: _initial_case(m, "executing")),
            (r"\bper[- ]run\b", "per processing session"),
            (r"\bcross[- ]run\b", "cross-session"),
            (r"\bmulti[- ]run\b", "multi-session"),
            (r"\brun(?:'s|’s)\b", "processing session's"),
            (r"\b(a|an|the|this|that|each|every|one|another|previous|current|failed|finished|completed|new|old|single|same|second|training|regression|GUI|overnight|ten-hour|twenty-minute)\s+run[- ](id|ids|identifier|identifiers|settings|folder|folders|path|paths|root|roots|manifest|manifests|status|statuses|ledger|ledgers|journal|journals|history|histories|result|results|output|outputs|artifact|artifacts|order|digest|digests|record|records|comparison|comparisons)\b", r"\1 processing-session \2"),
            (r"\b(?:run|runs)\s+(?=(?:id|ids|identifier|identifiers|settings|folder|folders|path|paths|root|roots|manifest|manifests|status|statuses|ledger|ledgers|journal|journals|history|histories|result|results|output|outputs|artifact|artifacts|order|digest|digests|record|records|comparison|comparisons)\b)", lambda m: _initial_case(m, "processing-session ")),
            (r"\b(a|an|the|this|that|each|every|one|another|previous|current|failed|finished|completed|new|old|single|same|second|training|regression|GUI|overnight|ten-hour|twenty-minute)\s+run\b", r"\1 processing session"),
            (r"\b(two|three|several|many|all|these|those|both|different|broken|completed|failed|finished|previous|current|training)\s+runs\b", r"\1 processing sessions"),
            (r"\blong[- ]run\s+path\b", "long-running-operation path"),
            (r"\bruns?\s+root(?:s)?\b", "processing-session root"),
            (r"\brun[- ]journal\b", lambda m: _initial_case(m, "processing-session journal")),
        ))
    if re.search(_COMPUTE_THREAD_SOURCE, prose, re.IGNORECASE):
        transforms.extend((
            (r"\b(?-i:A)\s+(?i:worker)\s+(?i:thread)\b", "A background execution unit"),
            (r"\b(?-i:a)\s+(?i:worker)\s+(?i:thread)\b", "a background execution unit"),
            (r"\bworker[- ]thread\s+counts\b", lambda m: _initial_case(m, "background-worker counts")),
            (r"\bworker[- ]thread\s+count\b", lambda m: _initial_case(m, "background-worker count")),
            (r"\b(a|an|the|this|that|each|every)\s+thread\s+pools\b", r"\1 worker pools"),
            (r"\b(a|an|the|this|that|each|every)\s+thread\s+pool\b", r"\1 worker pool"),
            (r"\bthread\s+pools\b", lambda m: _initial_case(m, "worker pools")),
            (r"\bthread\s+pool\b", lambda m: _initial_case(m, "worker pool")),
            (r"\bthread\s+counts\b", lambda m: _initial_case(m, "worker-count limits")),
            (r"\bthread\s+count\b", lambda m: _initial_case(m, "worker-count limit")),
            (r"\bthreaded\s+path\b", "background execution path"),
            (r"\bthreaded\s+JobRunner\b", "background JobRunner"),
            (r"\ba\s+threaded\b", "a background"),
            (r"\bthreaded,\s+not\s+repeated\b", "linked together, not repeated"),
            (r"\b(?-i:Threaded),", "With worker execution,"),
            (r"\b(?-i:threaded),", "with worker execution,"),
            (r"\bthread[- ]safety\b", lambda m: _initial_case(m, "safe use across execution paths")),
            (r"\bthread[- ]safe\b", lambda m: _initial_case(m, "safe across execution paths")),
            (r"\bGUI[- ]thread[- ]only\b", "allowed only in the main GUI execution path"),
            (r"\bworker[- ]thread[- ]safe\b", lambda m: _initial_case(m, "safe in a background execution unit")),
            (r"\bthread[- ]agnostic\b", "independent of the execution path"),
            (r"\bcross[- ]thread\b", "cross-execution-path"),
            (r"\boff[- ]thread\b", lambda m: _initial_case(m, "off-execution-path")),
            (r"\bthread[- ]local\b", "execution-path-local"),
            (r"\bthread\s+affinity\b", "execution-path affinity"),
            (r"\bthreading\b", "execution concurrency"),
            (r"\bthreaded\b", "executed in a worker"),
            (r"\bGUI[- ]worker[- ]threads\b", "main GUI execution paths"),
            (r"\bGUI[- ]worker[- ]thread\b", "main GUI execution path"),
            (r"\bGUI[- ]threads\b", "main GUI execution paths"),
            (r"\bGUI[- ]thread\b", "main GUI execution path"),
            (r"\bworker[- ]threads\b", lambda m: _initial_case(m, "background execution units")),
            (r"\bworker[- ]thread\b", lambda m: _initial_case(m, "background execution unit")),
            (r"\bbackground\s+threads\b", lambda m: _initial_case(m, "background execution units")),
            (r"\bbackground\s+thread\b", lambda m: _initial_case(m, "background execution unit")),
            (r"\bmain\s+threads\b", lambda m: _initial_case(m, "main execution paths")),
            (r"\bmain\s+thread\b", lambda m: _initial_case(m, "main execution path")),
            (r"\ba\s+thread\b", "an independent execution path"),
            (r"\bthreads\b", lambda m: _initial_case(m, "independent execution paths")),
            (r"\bthread\b", lambda m: _initial_case(m, "independent execution path")),
        ))
    if re.search(_IMAGE_CROP_SOURCE, prose, re.IGNORECASE):
        transforms.extend((
            (r"\bkeys\s+a\s+crop\b", "keys an extracted image region"),
            (r"\bResolve\s+object\s+keys\s+to\s+crop\s+rows\b", "Resolve object identifiers to rows of extracted image regions"),
            (r"\bIndex\s+of\s+the\s+crop\s+the\s+keyboard\b", "Index of the extracted image region that the keyboard"),
            (r"\bRe[- ]crop\s+the\s+loaded\s+array\b", "Extract image regions from the loaded array again"),
            (r"\bfrom\s+its\s+re[- ]crop\b", "after extracting image regions again"),
            (r"\ba\s+re[- ]crop\b", "extracting image regions again"),
            (r"\b(?-i:Crop)\s+every\s+object\s+out\s+of\b", "Extract an image region around every object in"),
            (r"\b(?-i:Crop)\s+away\b", "Remove"),
            (r"\b(?-i:Crop)\s+and\s+rescale\b", "Extract and rescale"),
            (r"\b(it|this\s+function|the\s+function)\s+crops\b", r"\1 extracts image regions from"),
            (r"\bit\s+cropped\b", "it extracted an image region around"),
            (r"\balso\s+crops\s+per[- ]object\b", "also extracts per-object image regions as"),
            (r"\bannotation[- ]crop\b", "annotation image region"),
            (r"\bper[- ]crop\b", "per extracted image region"),
            (r"\bobject[- ]crop\b", "object-image-region"),
            (r"\bmeasure[- ]and[- ]crop\b", "measurement-and-image-region-extraction"),
            (r"\b(?-i:a)\s+crop[- ]format\b", "an image-region-format"),
            (r"\b(?-i:A)\s+crop[- ]format\b", "An image-region-format"),
            (r"\b(?-i:a)\s+crop[- ]PNG\b", "an extracted-image-region PNG"),
            (r"\b(?-i:A)\s+crop[- ]PNG\b", "An extracted-image-region PNG"),
            (r"\b(?-i:a)\s+crop[- ]and[- ]measure\b", "an image-region extraction-and-measurement operation"),
            (r"\b(?-i:A)\s+crop[- ]and[- ]measure\b", "An image-region extraction-and-measurement operation"),
            (r"\bcrop[- ]and[- ]measure\b", "image-region extraction and measurement"),
            (r"\bcrop[- ]format\b", "image-region format"),
            (r"\bcrop[- ]PNG\b", "extracted-image-region PNG"),
            (r"\bre[- ]cropping\b", lambda m: _initial_case(m, "extracting image regions again")),
            (r"\bre[- ]cropped\b", lambda m: _initial_case(m, "extracted image regions again")),
            (r"\bre[- ]crops\b", lambda m: _initial_case(m, "extracts image regions again")),
            (r"\bre[- ]crop\b", lambda m: _initial_case(m, "extract image regions again")),
            (r"\bcropping\b", lambda m: _initial_case(m, "extracting image regions")),
            (r"\bcropped\b", lambda m: _initial_case(m, "extracted image regions")),
            (r"\brows\s+to\s+crop\b", "rows whose extracted image regions are requested"),
            (r"\bslice\s+index\s+of\s+the\s+object[- ]class\s+mask\s+to\s+crop\s+by\b", "mask image layer for the object class used to extract the image region"),
            (r"\bobject\s+table\s*\+\s*mask\s+slice\s+to\s+crop\b", "object table and mask image layer used to extract image regions"),
            (r"\b(?:plane|slice)(?:\s+index)?\s+to\s+crop(?:\s+by)?\b", "mask image layer used to extract the image region"),
            (r"\bcannot\s+crop\b", "cannot extract an image region from"),
            (r"\bto\s+crop\b", "to extract image regions from"),
            (r"\b(?-i:Crop)\s+(?=(?:the|these|those|each|all|objects?)\b)", "Extract image regions around "),
            (r"\bcrop\s+(?=(?:the|these|those|each|all|objects?)\b)", "extract image regions around "),
            (r"\b(?-i:an)\s+image\s+crop\b", "an extracted image region"),
            (r"\b(?-i:An)\s+image\s+crop\b", "An extracted image region"),
            (r"\b(?-i:a)\s+crop\b", "an extracted image region"),
            (r"\b(?-i:A)\s+crop\b", "An extracted image region"),
            (r"\bcrop(?:'s|’s)\b", "extracted image region's"),
            (r"\bcrop[- ](?=(?:PNG|path|file|table|row|key|column|grid|folder|manifest|mode|settings?|source|writer|pass|knob|preview|configuration|metadata|shaping|generation)\b)", "extracted-image-region "),
            (r"\bimage\s+crops\b", lambda m: _initial_case(m, "extracted image regions")),
            (r"\bimage\s+crop\b", lambda m: _initial_case(m, "extracted image region")),
            (r"\bcrops\b", lambda m: _initial_case(m, "extracted image regions")),
            (r"\bcrop\b", lambda m: _initial_case(m, "extracted image region")),
        ))
    if exception_raises:
        transforms.extend((
            (r"\b(?-i:Raises)\s+([A-Za-z_]\w*(?:Error|Exception|Cancelled))\b", r"Throws \1"),
            (r"\b(?-i:raises)\s+([A-Za-z_]\w*(?:Error|Exception|Cancelled))\b", r"throws \1"),
            (r"\b(?-i:Raise)\s+([A-Za-z_]\w*(?:Error|Exception|Cancelled))\b", r"Throw \1"),
            (r"\b(?-i:raise)\s+([A-Za-z_]\w*(?:Error|Exception|Cancelled))\b", r"throw \1"),
            (r"\b(?-i:Raised)\s+when\b", "Exception used when"),
            (r"\b(?-i:raised)\s+when\b", "reported as an exception when"),
            (r"\bexception\s+raised\b", "exception thrown"),
            (r"\bre[- ]raise\s+it\b", "throw the same exception again"),
            (r"\bre[- ]raises?\s+this\b", "throws this same exception again"),
            (r"\bre[- ]raises?\s+the\s+same\s+exception\b", "throws the same exception again"),
            (r"\b(?-i:Re[- ]raise)\s+([^.;]+?)\s+instead\b", r"Throw \1 again instead"),
            (r"\bre[- ]raises?\b", "throws the same exception again"),
            (r"\bnever\s+raises\s+the\s+run\s+down\b", "never stops the processing session with an error"),
            (r"\braise\s+on\s+failure\b", "throw an exception on failure"),
            (r"\braise\s+(if|when|on)\b", r"throw an exception \1"),
            (r"\b(?:does\s+not|do\s+not|cannot|never)\s+raise\b", lambda m: m.group(0)[:-5] + "throw"),
            (r"\b(without|instead\s+of|rather\s+than)\s+raising\b", r"\1 throwing"),
            (r"\breported\s+on\s+stderr\s+rather\s+than\s+raised\b", "reported on stderr without producing an error"),
            (r"\bdetected\s+rather\s+than\s+raised\b", "detected without producing an error"),
            (r"\bskipped\s+rather\s+than\s+raised\b", "skipped without producing an error"),
            (r"\bcached\s+as\s+``None``\s+rather\s+than\s+raised\b", "cached as ``None`` without producing an error"),
            (r"\brather\s+than\s+raised\b", "without producing an error"),
            (r"\b(?-i:raises)\s+on\s+click\b", "produces an error when clicked"),
            (r"\b(?-i:raises)\s+instead\b", "produces an error instead"),
            (r"\b(?-i:Raises)\s+(?=:[A-Za-z])", "Throws "),
            (r"\b(?-i:raises)\s+(?=:[A-Za-z])", "throws "),
            (r"\b(?-i:Raise)\s+(?=:[A-Za-z])", "Throw "),
            (r"\b(?-i:raise)\s+(?=:[A-Za-z])", "throw "),
            # RST field-list names are structural chrome, not exception prose.
            # ``:raises SpecError:`` must remain byte-identical so the parser
            # continues to recognize the standard field.
            (r"(?<!:)\braises(?=\s+[^:\n]+:\s)", "raises"),
            (r"\b(?-i:Never\s+raises)\b", "Never throws"),
            (r"\b(?-i:never\s+raises)\b", "never throws"),
            (r"\b(?-i:never\s+raised)\b", "never reported as an error"),
            (r"\b(?-i:anything\s+else\s+raises)\b", "anything else produces an error"),
            (r"\b(?-i:raises)\s+(?=(?:for|with|from|on)\b)", "produces an error "),
            (r"\b(?-i:raise)\s+(?=(?:for|with|from|on)\b)", "produce an error "),
            (r"\b(?-i:Raises)(?=\s*$)", "Throws"),
            (r"\b(?-i:raises)(?=\s*$)", "throws"),
            (r"\b(?-i:Raise)(?=\s*$)", "Throw"),
            (r"\b(?-i:raise)(?=\s*$)", "throw"),
        ))
    if re.search(_SCIENTIFIC_PLATE_SOURCE, prose, re.IGNORECASE):
        transforms.extend((
            (r"\b(?-i:A)\s+(?i:plate)\b", "A laboratory microplate"),
            (r"\b(?-i:a)\s+(?i:plate)\b", "a laboratory microplate"),
            (r"\b(\d+)\s*[- ]well\s+plates\b", r"\1-position laboratory microplates"),
            (r"\b(\d+)\s*[- ]well\s+plate\b", r"\1-position laboratory microplate"),
            (r"\bplates?\s+wells\b", lambda m: _initial_case(m, "microplate sample positions")),
            (r"\bplates?\s+well\b", lambda m: _initial_case(m, "microplate sample position")),
            (r"\bplates\b", lambda m: _initial_case(m, "laboratory microplates")),
            (r"\bplate\b", lambda m: _initial_case(m, "laboratory microplate")),
        ))
    if scientific_wells:
        alternatives = [
            (r"\ba\s+wells[- ]by[- ]genes\b", lambda m: _initial_case(
                m, "a microplate-sample-position-by-gene",
            )),
            (r"\bwells\b", lambda m: _initial_case(m, "microplate sample positions")),
        ]
        if total_wells == scientific_wells:
            alternatives.append((r"\bwell\b", lambda m: _initial_case(m, "microplate sample position")))
        transforms.extend(alternatives)
    if re.search(_MAPPING_KEY_SOURCE, prose, re.IGNORECASE):
        transforms.extend((
            (r"\bA\s+one-column-per-key-column\s+frame\s+for\s+", "A frame with one output column for each identifier column in "),
            (r",\s+in\s+their\s+order\.", ", preserving their order."),
            (r"\bkey/query/value\b", "attention-key/query/value"),
            (r"\bkey[- ]value\b", "key-value"),
            (r"\bquery/key/value\b", "query/identifier/value"),
            (r"\bkey[- ]to[- ]key\b", "identifier-to-identifier"),
            (r"\bkey\s+name\b", "field name"),
            (r"\bkey\s+entries\b", lambda m: _initial_case(m, "structured-data entries")),
            (r"\ba\s+key\s+column\b", "an identifier column"),
            (r"\bsettings?\s+keys\b", "setting names"),
            (r"\bsettings?\s+key\b", "setting name"),
            (r"\bto\s+key\s+on\b", "to use as an index"),
            (r"\bkeys\s+its\s+([^.;,]+?)\s+off\s+this\b", r"indexes its \1 using this"),
            (r"\bcache\s+keys\s+off\s+it\b", "cache is indexed by it"),
            (r"\bspaCR\s+keys\s+objects\s+by\b", "spaCR identifies objects by"),
            (r"\btable\s+keys\s+each\b", "table identifies each"),
            (r"\bdoes\s+not\s+key\s+objects\s+by\b", "does not identify objects by"),
            (r"\bmeasurement\s+tables\s+key\s+on\b", "measurement tables are indexed by"),
            (r"\b(features?|models?|results?)\s+key\s+on\b", lambda m: m.group(1) + (" are indexed by" if m.group(1).lower().endswith("s") else " is indexed by")),
            (r"\bkeyed\s+on\b", "indexed by"),
            (r"\bkeys\s+on\b", "is indexed by"),
            (r"\bkey/value\b", "name/value"),
            (r"\bevery\s+key\s+of\b", "every field name in"),
            (r"\bkeys\s+the\b", "structured-data names that the"),
            (r"\bkeys\s+a\b", "structured-data names that a"),
            (r"\bwithout\s+keys\b", lambda m: _initial_case(m, "without identifiers")),
            (r"\bobject\s+keys\b", "object identifiers"),
            (r"\bobject\s+key\b", "object identifier"),
            (r"\brow\s+keys\b", "row identifiers"),
            (r"\brow\s+key\b", "row identifier"),
            (r"\bkey\s+columns\b", "identifier columns"),
            (r"\bkey\s+column\b", "identifier column"),
            (r"\bstate[- ]dict\s+keys\b", "state-dictionary entry names"),
            (r"\bstate[- ]dict\s+key\b", "state-dictionary entry name"),
            (r"\b(?:settings?\s+)?dict\s+keys\b", "dictionary entry names"),
            (r"\b(?:settings?\s+)?dict\s+key\b", "dictionary entry name"),
            (r"\bimage[- ]key\s+values\b", "image-identifier values"),
            (r"\bimage[- ]key\s+value\b", "image-identifier value"),
            (r"\bmapping\s+keys\b", lambda m: _initial_case(m, "structured-data names")),
            (r"\bmapping\s+key\b", "structured-data name"),
            (r"\bconfiguration\s+keys\b", "configuration field names"),
            (r"\bconfiguration\s+key\b", "configuration field name"),
            (r"\bkeys\b", lambda m: _initial_case(m, "structured-data names")),
            (r"\bkey\b", lambda m: _initial_case(m, "structured-data name")),
        ))
    # Mixed GUI/scientific-screen paragraphs need review rather than a blind
    # inverse rewrite because both English occurrences are intentional.
    if has_gui_screen and not has_scientific_screen:
        transforms.extend((
            (r"\ba\s+screens\b", "application views"),
            (r"\ba\s+screen\b", lambda m: _initial_case(m, "an application view")),
            (r"\bscreens\b", lambda m: _initial_case(m, "application views")),
            (r"\bscreen\b", lambda m: _initial_case(m, "application view")),
        ))
    elif has_scientific_screen and not has_gui_screen:
        transforms.extend((
            (r"\bscreens\b", lambda m: _initial_case(m, "screening experiments")),
            (r"\bscreen\b", lambda m: _initial_case(m, "screening experiment")),
        ))
    if re.search(_DICTIONARY_SOURCE, prose, re.IGNORECASE):
        transforms.extend((
            (r"\bdictionary\s+mapping\b", "key-value association"),
            (r"\bdict\s+keys\b", "dictionary entry names"),
            (r"\bdict\s+key\b", "dictionary entry name"),
            (r"\bdictionaries\b", "key-value mappings"),
            (r"\bdictionary\b", "key-value mapping"),
            (r"\bdicts\b", "key-value mappings"),
            (r"\bdict\b", "key-value mapping"),
        ))
    if re.search(_PIPELINE_SOURCE, prose, re.IGNORECASE):
        transforms.extend((
            (r"\bpipelines\b", lambda m: _initial_case(m, "workflows")),
            (r"\bpipeline\b", lambda m: _initial_case(m, "workflow")),
        ))
    if re.search(_DATA_GATE_SOURCE, prose, re.IGNORECASE):
        transforms.extend((
            (r"\b(users?|callers?|filters?)\s+gates\s+on\b", r"\1 selects rows using"),
            (r"\bto\s+gate\b", "to filter data"),
            (r"\bgating\b", lambda m: _initial_case(m, "data-selection filtering")),
            (r"\bgates\b", lambda m: _initial_case(m, "data-selection boundaries")),
            (r"\bgate\b", lambda m: _initial_case(m, "data-selection boundary")),
        ))
    if _statistical_power_source(prose):
        transforms.extend((
            (r"\bstatistical\s+power\b", "statistical detection sensitivity"),
            (r"\bpower\b", "statistical detection sensitivity"),
        ))
    if re.search(_SOFTWARE_QUEUE_SOURCE, prose, re.IGNORECASE):
        queue_is_annotation_list = bool(re.search(
            r"(?i)\b(?:active[- ]learning|annotation|annotator|uncertainty|"
            r"diversif(?:y|ied|ication))\b",
            prose,
        ))
        queue_is_batch_buffer = bool(re.search(
            r"(?i)\b(?:DataLoader|pre[- ]?fetch|batches?\s+ahead|sentinel|"
            r"coalesced\s+transactions)\b",
            prose,
        ))
        queue_is_job_list = bool(re.search(
            r"(?i)\b(?:jobs?|tasks?|plates?|batches?|enqueue|dequeue|"
            r"scheduler|runner)\b",
            prose,
        ))
        if queue_is_batch_buffer:
            queue_singular, queue_plural = "batch-data buffer", "batch-data buffers"
        elif queue_is_annotation_list:
            queue_singular = "annotation work list"
            queue_plural = "annotation work lists"
        elif queue_is_job_list:
            queue_singular, queue_plural = "software job list", "software job lists"
        else:
            queue_singular, queue_plural = "work list", "work lists"
        transforms.extend((
            (r"\ba\s+queue\b(?!\s*[- ]+\s*(?:based|backed)\b)", lambda m: _initial_case(m, (
                "an annotation work list" if queue_is_annotation_list
                else "a " + queue_singular
            ))),
            (r"\b(?-i:Queue)\s+plate\s+folders\b", "Add plate folders to the software job list"),
            (r"\bQt\s+then\s+queues\b", "Qt then schedules"),
            (r"\bbatches\s+from\s+a\s+Queue\b", "batches from a batch-data buffer"),
            (r"\bA\s+private\s+Queue\b", "A private error-message list"),
            (r"\bmust\s+never\s+queue\b", "must never schedule"),
            (r"\b(?:can|cannot|will|should|does|did|then)\s+queue\b", lambda m: m.group(0)[:-5] + "schedule"),
            (r"\bqueues\s+(?=(?:the|a|an|plates?|jobs?|tasks?|them)\b)", lambda m: _initial_case(m, "schedules ")),
            (r"\bqueue\s+(?=(?:the|a|an|plates?|jobs?|tasks?|them)\b)", "schedule "),
            (r"\btwelve[- ]job\s+queue\b", "software job list with twelve jobs"),
            (r"\bjob\s+queue\b", "software job list"),
            (r"\b(?-i:Plate\s+Queue)\b", "Plate-processing list"),
            (r"\b(?-i:plate\s+queue)\b", "plate-processing list"),
            (r"\bqueue[- ]+\s*based\b", "work-list-based"),
            (r"\bqueue[- ]backed\b", "work-list-backed"),
            (r"\bmid[- ]queue\b", "while processing the software job list"),
            (r"\bqueue[- ]level\b", "software-job-list level"),
            (r"\breview\s+queue\b", lambda m: _initial_case(m, "review list")),
            (r"\bactive[- ]learning\s+queue\b", lambda m: _initial_case(m, "annotation work list")),
            (r"\bannotation\s+queue\b", lambda m: _initial_case(m, "annotation work list")),
            (r"\buncertainty\s+queue\b", lambda m: _initial_case(m, "uncertainty-ranked work list")),
            (r"\bwork\s+queue\b", lambda m: _initial_case(m, "work list")),
            (r"\bfigure\s+queue\b", lambda m: _initial_case(m, "figure list")),
            (r"\bmessage\s+queue\b", lambda m: _initial_case(m, "message list")),
            (r"\bevent\s+queue\b", lambda m: _initial_case(m, "event-delivery list")),
            (r"\bworker\s+queue\b", lambda m: _initial_case(m, "worker-event list")),
            (r"\boptional\s+queue\s+used\s+to\s+surface\s+error\s+strings\b", "optional error-message list used to surface error strings"),
            (r"\blog(?:/error)?\s+queue\b", lambda m: _initial_case(m, "log-message list")),
            (r"\bqueues\b", lambda m: _initial_case(m, queue_plural)),
            (r"\bqueue\b", lambda m: _initial_case(m, queue_singular)),
        ))
    if re.search(_IMAGING_FIELD_SOURCE, prose, re.IGNORECASE):
        transforms.extend((
            # Retain the protected spaCR acronym while spelling out its sense
            # for the model.  The non-cascading rewrite below prevents the
            # inserted ``fields`` from matching the next alternative.
            (r"\bFOVs\b(?!-)", "FOVs (microscope fields of view)"),
            (r"\bFOV\b(?!-)", "FOV (microscope field of view)"),
            (r"\bimage\s+fields\b", "microscope image fields of view"),
            (r"\bimage\s+field\b", "microscope image field of view"),
            (r"\bfields\s+of\s+view\b", "microscope fields of view"),
            (r"\bfield\s+of\s+view\b", "microscope field of view"),
            (r"\bfields\b", "microscope fields of view"),
            (r"\bfield\b", "microscope field of view"),
        ))
    if re.search(_IMAGING_CHANNEL_SOURCE, prose, re.IGNORECASE):
        transforms.extend((
            (r"\bintensity\s+channels\s+first\b", "intensity image data channels first"),
            (r"\bis\s+often\s+channel[- ]first\b", "often stores image data channels first"),
            (r"\bis\s+often\s+channel[- ]last\b", "often stores image data channels last"),
            (r"\bis\s+channel[- ]first\b", "stores image data channels first"),
            (r"\bis\s+channel[- ]last\b", "stores image data channels last"),
            (r"\ba\s+(\d+)[- ]channel\s+image\b", r"an image with \1 data channels"),
            (r"\b(\d+)[- ]channel\s+image\b", r"image with \1 data channels"),
            (r"\bimage\s+channels\b", "image data channels"),
            (r"\bimage\s+channel\b", "image data channel"),
            (r"\ba\s+channels?[- ]last\b", "an image-channel-last"),
            (r"\ba\s+channels?[- ]first\b", "an image-channel-first"),
            (r"\bchannels?[- ]last\b", "image-channel-last"),
            (r"\bchannels?[- ]first\b", "image-channel-first"),
            (r"\ba\s+channel\b", "an image data channel"),
            (r"\bchannels\b", "image data channels"),
            (r"\bchannel\b", "image data channel"),
        ))
    if re.search(_SOFTWARE_CLASSIFIER_SOURCE, prose, re.IGNORECASE):
        transforms.extend((
            (r"\bclassifiers\b", "machine-learning classification models"),
            (r"\bclassifier\b", "machine-learning classification model"),
        ))
    if re.search(_HUMAN_READABLE_SOURCE, prose, re.IGNORECASE):
        transforms.extend((
            (r"\ba\s+human[- ]readable\b", lambda m: _initial_case(m, "an easy-to-read")),
            (r"\bhuman[- ]readable\b", lambda m: _initial_case(m, "easy-to-read")),
            (r"\bhuman\s+description\b", "user-facing description"),
            (r"\bhuman\s+summary\b", "user-facing summary"),
            (r"\bhuman\s+text\b", "user-facing text"),
            (r"\bhuman\s+label\b", "user-facing label"),
            (r"\bhuman\s+reference\b", "user-facing reference"),
        ))

    if not transforms:
        return source

    def rewrite(fragment: str) -> str:
        return _replace_alternatives_once(fragment, transforms)

    contextual = _rewrite_unprotected_prose(source, rewrite)
    contextual = _preserve_initial_prose_case(source, contextual)
    if not _syntax_preserved(source, contextual):
        raise ValueError(
            "API sense context changed a protected literal: "
            f"{source!r} -> {contextual!r}"
        )
    return contextual

# Shorter English model inputs for prose that OPUS repeatedly decoded only in
# part. These retain the complete semantic contract and every protected API
# literal; they do not alter the English Python docstrings.
API_TRANSLATION_CONTEXT.update({
    "enable SQLite foreign-key enforcement on this connection. SQLite defaults it off per connection.":
        "Turn on SQLite foreign-key checks for this connection. SQLite starts "
        "with these checks disabled for each connection.",
    "Settings dict; canonicalized via :func:`spacr.settings.deep_spacr_defaults`. Key flags/inputs:":
        "Settings dictionary normalized by "
        ":func:`spacr.settings.deep_spacr_defaults`. Important options:",
    (
        "``cross_validation_enabled`` / ``cross_validation_folds`` — train with "
        "k-fold cross-validation over ``train/`` instead of a single validation "
        "split (enabling it with fewer than 2 folds uses 5, and 1 fold falls back "
        "to the single split); ``cv_group_by`` names the grouping level used for "
        "the folds and the leakage audits."
    ): (
        "``cross_validation_enabled`` and ``cross_validation_folds`` select "
        "k-fold cross-validation over ``train/`` instead of one validation "
        "split. Fewer than 2 folds selects 5; 1 fold selects the single split. "
        "``cv_group_by`` chooses the grouping level for folds and leakage audits."
    ),
    (
        "1. **Flatten.** Denoise with a 1 px Gaussian, then subtract a heavily "
        "smoothed copy (sigma = max(32, min(H, W) / 4)) to remove illumination "
        "gradients. The sigma is deliberately far larger than any plausible "
        "object so that flattening removes vignetting without eating the objects "
        "— the opposite trade-off (a tight rolling ball) shrinks what it is "
        "trying to measure. 2. **Reject noise.** Compare the structural amplitude "
        "of the flattened plane (p99 - p30) against the *pixel-level* noise "
        "scale, measured as ``1.4826 * MAD(img - gaussian(img, 1))`` on the raw "
        "plane. A pure-noise plane scores below 1; a plane with real objects "
        "scores in the tens. Below ``min_snr`` the field is discarded rather "
        "than thresholded, because Otsu will happily bisect pure noise and hand "
        "back a confident-looking number. 3. **Threshold and label.** Otsu, fill "
        "holes, label, drop components that touch the image border (truncated, "
        "so their size is a lie) and components that are absurd (equivalent "
        "diameter below ``min_object_diameter``, or area above "
        "``max_object_fraction`` of the field). Characteristic size is the median "
        "equivalent diameter, ``2 * sqrt(area / pi)``. 4. **Cross-check by "
        "distance transform.** Step 3 has one dominant failure mode: a confluent "
        "monolayer fuses into a single component, that component touches the "
        "border and is dropped, and the estimate is then computed from whatever "
        "debris survived — biased **low**, and silently. So the Euclidean distance "
        "transform of the (unfilled) foreground is computed as well, its local "
        "maxima are taken as one seed per object (two passes: a coarse pass sets "
        "the suppression radius for the refined pass), and a watershed on "
        "``-EDT`` splits the fused foreground back into objects whose equivalent "
        "diameters are measured the same way."
    ): (
        "1. **Flatten.** Apply a 1 px Gaussian filter, then subtract a smoother "
        "copy with sigma = max(32, min(H, W) / 4). This sigma is larger than a "
        "plausible object, so it removes vignetting without the object shrinkage "
        "caused by a tight rolling ball. 2. **Reject noise.** Compare the "
        "flattened plane’s structural amplitude (p99 - p30) with the *pixel-level* "
        "raw-plane noise estimate ``1.4826 * MAD(img - gaussian(img, 1))``. Pure "
        "noise scores below 1; real objects score in the tens. Discard a field "
        "below ``min_snr`` instead of applying Otsu to pure noise. 3. **Threshold "
        "and label.** Apply Otsu, fill holes, and label components. Remove "
        "border-touching components because they are truncated; also remove "
        "equivalent diameters below ``min_object_diameter`` and areas above "
        "``max_object_fraction``. Use the median equivalent diameter, "
        "``2 * sqrt(area / pi)``. 4. **Cross-check by distance transform.** A "
        "confluent layer can fuse into one border-touching component, which is "
        "discarded and leaves debris that biases the estimate silently **low**. "
        "Compute the Euclidean distance transform of the unfilled foreground. "
        "Use local maxima as object seeds in two passes: a coarse pass sets the "
        "suppression radius for the refined pass. Watershed on ``-EDT`` separates "
        "the fused foreground before measuring equivalent diameters the same way."
    ),
    "…which Mask, Measure, Annotate, Classify, the Plate Viewer and the Database Browser all read without knowing it was imported.":
        "Mask, Measure, Annotate, Classify, the Plate Viewer, and the Database "
        "Browser can all read the imported result transparently.",
    "Build the run/abort/download/settings button row and progress bar.":
        "Create one row containing the run, abort, download, and settings "
        "buttons, plus the progress bar.",
    "Build the RAM/VRAM/GPU/per-core CPU usage bars beside the button strip.":
        "Create RAM, VRAM, GPU, and per-core CPU usage bars next to the buttons.",
    "Create the toolbar with draw/wand/erase/brush/divider mode buttons.":
        "Create a toolbar containing the draw, wand, erase, brush, and divider "
        "mode buttons.",
    "Return the gene id a model term names, or ``None``.":
        "Return the gene identifier named by a model term, or ``None``.",
    "Plot training vs. validation curves from a saved model's per-epoch CSVs.":
        "Plot training and validation curves from the per-epoch CSVs of a saved "
        "model.",
    (
        "ordered class names; index i names label i. Defaults to "
        "``['class_0', ...]`` sized to the largest label seen."
    ): (
        "Class names in order: position i names label i. The default is "
        "``['class_0', ...]``, sized for the largest observed label."
    ),
    (
        "dict mapping object type → its mask slice index. Defaults to spaCR's "
        "layout ``{cell:4, nucleus:5, pathogen:6, organelle:7}`` (four image "
        "channels). Override if your arrays have a different channel count."
    ): (
        "Dictionary mapping each object type → its mask-slice index. The spaCR "
        "default is ``{cell:4, nucleus:5, pathogen:6, organelle:7}`` for four "
        "image channels. Supply another mapping for arrays with a different "
        "channel count."
    ),
    "1-based row (spaCR's convention). OMERO's row + 1.":
        "Row counted from 1, following spaCR conventions; equal to the OMERO row "
        "plus 1.",
    "1-based field (imaging site) id.":
        "Imaging-site identifier counted from 1.",
    (
        "a list of sources means several plates, and the first names the first "
        "project (:func:`spacr.core.preprocess_generate_masks` loops over "
        "``settings['src']``);"
    ): (
        "A source list represents multiple plates. Its first source names the "
        "first project; :func:`spacr.core.preprocess_generate_masks` iterates "
        "over ``settings['src']``."
    ),
    'The answer to "Measure needs merged arrays — who makes those?".':
        "Identifies the upstream module that creates the merged arrays needed by "
        "Measure.",
    (
        "The AI Console now shells out to the vendor coding-agent CLIs (claude / "
        "codex / gemini), each of which authenticates against the user's chat "
        "subscription instead of a metered API key. Nothing in the Qt GUI uses "
        "API keys anymore."
    ): (
        "The AI Console launches the vendor coding-agent CLIs (claude / codex / "
        "gemini). Each CLI authenticates with the user's chat subscription, not "
        "a metered API key. The Qt GUI no longer uses API keys."
    ),
    "shell one-liner suggested for installation.":
        "One-line shell command suggested for installation.",
    (
        "Every navigable app, every preference toggle, every menu action gets "
        "registered as a :class:`Command` and the palette lets users jump to it "
        "by typing a few characters. Modelled on VS Code / Slack / Linear."
    ): (
        "Register each app, preference switch, and menu action as a "
        ":class:`Command`. Users can open an item by typing a few characters in "
        "the palette. The design follows VS Code / Slack / Linear."
    ),
    "Dropdown to pick a built-in as a starting point.":
        "Dropdown for selecting an included pattern as the initial value.",
    (
        "**Keyset paging, not OFFSET.** Each chunk asks for "
        "``rowid > <last rowid seen>`` and ``ORDER BY rowid``. "
        "``LIMIT ? OFFSET ?`` makes SQLite walk (and throw away) every skipped "
        "row, so chunk 500 of a 400 k-row table would cost 500× chunk 1 — the "
        '"fast" version would get slower the further you scrolled. Only tables '
        "with no usable key (views, composite-primary-key ``WITHOUT ROWID`` "
        "tables) fall back to ``OFFSET``, and those are small by construction."
    ): (
        "**Use keyset paging instead of OFFSET.** Fetch each chunk with "
        "``rowid > <last rowid seen>`` and ``ORDER BY rowid``. "
        "``LIMIT ? OFFSET ?`` makes SQLite scan every skipped row, so chunk "
        '500 costs about 500× as much as chunk 1 and the "fast" version gets '
        "slower. Small views and "
        "composite-key ``WITHOUT ROWID`` tables have no usable row key and use "
        "``OFFSET`` as a fallback."
    ),
    (
        "The flow-cytometry gesture, on spaCR measurement tables. Drag a "
        "threshold across a histogram or a polygon round the cloud on a "
        "two-parameter scatter, name it, and the shape becomes a "
        ":class:`spacr.selection.DataFilter` clause that every open view honours "
        "— the UMAP, the plate map, the crop grid, the Graph Builder, Small "
        "Multiples."
    ): (
        "Apply flow-cytometry gestures to spaCR measurement tables. Drag a "
        "threshold on a histogram or draw a polygon on a two-variable scatter. "
        "After naming it, the shape becomes a "
        ":class:`spacr.selection.DataFilter` used by every open view: UMAP, plate "
        "map, crop grid, Graph Builder, and Small Multiples."
    ),
    (
        "Sibling screens (align / batch / convert / report / plate_view / …) "
        "solve this by never opening a modal at all — see their ``_set_status`` "
        "docstrings, which cite this screen as the case that actually hung. That "
        'is not sufficient here because "Clear mask" genuinely needs a yes/no '
        "answer, so this screen keeps the modal when — and only when — there is "
        "somebody able to answer it."
    ): (
        "The align / batch / convert / report / plate_view screens never open a "
        "modal dialog; their ``_set_status`` documentation refers to the earlier "
        "hang in this screen. Here, however, "
        '"Clear mask" requires a yes-or-no response. Therefore this screen opens '
        "the dialog only when a person can answer it."
    ),
    "Search box, Modified filter, Essentials/All switch, and a count line.":
        "Search field, Modified filter, Essentials/All switch, and result count.",
    (
        "Uses the Piper CLI (already installed via pip install piper-tts). Voice "
        "model defaults to ~/.spacr/piper/en_US-lessac-medium.onnx but any Piper "
        ".onnx can be passed via `voice_model=`."
    ): (
        "Uses the Piper CLI installed with pip install piper-tts. The default "
        "voice is ~/.spacr/piper/en_US-lessac-medium.onnx; `voice_model=` accepts "
        "any Piper .onnx voice."
    ),
    (
        "Returns True once the PNG — the raster the GUI actually displays — is on "
        "disk. A failed *sibling PDF* does not turn that into False, and the "
        "asymmetry is deliberate rather than sloppy: the callers turn False into "
        '"no pixmap, no thumbnail", so reporting a missing export that way would '
        "delete the figure from the gallery over a file nothing has asked for "
        "yet. It is logged at WARNING instead, and "
        ":meth:`FigureQueue._request_pdf_refinement` notices the absent page, "
        "says so, and stops waiting for a render that will never arrive."
    ): (
        "Return True after writing the PNG displayed by the GUI. Failure of the "
        "*related PDF* does not return False. Callers interpret False as "
        '"no pixmap, no thumbnail", which would incorrectly remove the figure. '
        "Instead, log a warning. "
        ":meth:`FigureQueue._request_pdf_refinement` detects the missing page "
        "and stops waiting for it."
    ),
    (
        "Each entry is a dict with keys ``dir`` (Path), ``app_key`` (str), "
        "``status`` (str), ``start_utc`` (ISO str), ``elapsed_s`` (float), and "
        "the raw ``manifest`` (dict, best-effort)."
    ): (
        "Each dictionary contains ``dir`` (Path), ``app_key`` (str), ``status`` "
        "(str), ``start_utc`` (ISO str), ``elapsed_s`` (float), and the raw "
        "``manifest`` dictionary when available."
    ),
    (
        "-- and that was the whole remaining cost of the first module open, "
        "measured with the event-loop watchdog in "
        "``tests/qt/test_gui_responsiveness.py`` *after* ``spacr`` and "
        "``spacr.settings`` were already imported. The function being fetched "
        "is a hundred lines of dictionary lookups. Everything else in the 770 "
        "ms belongs to the module it happened to live in: ``spacr.gui_utils`` "
        "imports ``spacr.gui_elements`` (IPython 154 ms, matplotlib.pyplot 145 "
        "ms), ``cv2`` (79 ms), ``tkinter``, ``huggingface_hub``, ``requests``, "
        "``PIL`` and ``screeninfo`` -- the *Tk* interface's dependencies, none "
        "of which the Qt interface has any use for."
    ): (
        "This was the remaining cost of the first module opening, measured by "
        "``tests/qt/test_gui_responsiveness.py`` *after* importing ``spacr`` and "
        "``spacr.settings``. The requested function performs dictionary lookups. "
        "The other 770 ms came from its old module: ``spacr.gui_utils`` imports "
        "``spacr.gui_elements`` (IPython 154 ms and matplotlib.pyplot 145 ms), "
        "``cv2`` (79 ms), ``tkinter``, ``huggingface_hub``, ``requests``, "
        "``PIL``, and ``screeninfo``. These are dependencies of the *Tk* "
        "interface and are unnecessary for the Qt interface."
    ),
    "Return precision/recall/F1/PR-AUC arrays for a DataFrame of ``is_active``/``score`` rows.":
        "Return precision, recall, F1, and PR-AUC arrays for DataFrame rows "
        "containing ``is_active`` and ``score``.",
    "Grid-plot PR-AUC vs ``variable`` for every unique combination of the other sweep dimensions.":
        "Plot a grid of PR-AUC against ``variable`` for each unique combination "
        "of the remaining sweep dimensions.",
    "Picks T-test vs Mann-Whitney U for two groups (based on a normality check) and ANOVA vs Kruskal-Wallis for three or more.":
        "For two groups, choose between a T-test and Mann-Whitney U after a "
        "normality check. For three or more groups, choose between ANOVA and "
        "Kruskal-Wallis.",
    (
        'detector : {"ORB","SIFT"} Feature detector for keypoint matching. '
        "nfeatures : int Feature budget for detector. max_keypoints : "
        "Optional[int] Hard cap on kept keypoints after detection (by detector’s "
        "internal ranking). downsample : float in (0,1] Downsample factor for "
        "feature/score pass. ransac_thresh_px : float Reprojection threshold "
        "(pixels) for affine estimation (downsampled space). allow_scale : bool "
        "If False, constrain to rotation+translation (or translation only if "
        "allow_rotation=False). allow_rotation : bool If False, constrain to "
        "translation only. outdir : str Output directory for images/csv. "
        "opencv_threads : int Limit OpenCV internal threading (avoid "
        "oversubscription)."
    ): (
        'detector : {"ORB","SIFT"} Keypoint detector. nfeatures : int Detector '
        "feature budget. max_keypoints : Optional[int] Hard cap on keypoints "
        "kept after detection, using the detector’s ranking. downsample : float "
        "in (0,1] Factor for feature detection and scoring. "
        "ransac_thresh_px : float Pixel reprojection threshold for affine "
        "estimation in downsampled space. allow_scale : bool If False, permit "
        "rotation and translation but no scaling; if allow_rotation=False, "
        "permit translation only. allow_rotation : bool If False, permit "
        "translation only. outdir : str Directory for image and csv output. "
        "opencv_threads : int OpenCV thread limit used to prevent "
        "oversubscription."
    ),
    "Auto-updater — compare local ``spacr`` to PyPI + the nightly branch.":
        "Automatic updater that compares local ``spacr`` with PyPI and the "
        "nightly branch.",
    (
        "Console decoration must never be able to end a run. No Windows codepage "
        "encodes spaCR's own output set -- ``▸`` (U+25B8) is absent from cp1252, "
        "cp437, cp850, cp932 *and* cp936, and the box-drawing frame is absent "
        "from cp1252 -- and neither does any of them encode the domain vocabulary "
        "that ends up in settings values, such as the parental strain ``Δku80`` "
        "or a ``µm`` voxel size. Printing either to a non-UTF-8 stream raises "
        "``UnicodeEncodeError``, and on Windows that is the normal case the moment "
        "stdout is redirected: a batch-queue job, ``spacr-run``, a legacy console."
    ): (
        "Console decoration must never terminate a run. Windows code pages do "
        "not encode every spaCR output character. For example, ``▸`` is missing "
        "from cp1252, cp437, cp850, cp932 *and* cp936, and cp1252 lacks the frame "
        "characters. Settings may also contain ``Δku80`` or ``µm``. Writing these "
        "values to a non-UTF-8 stream raises ``UnicodeEncodeError``. On Windows, "
        "this commonly occurs when stdout is redirected by a batch job, "
        "``spacr-run``, or a legacy console."
    ),
    "directory containing mask .tif/.tiff/.npy files.":
        "Directory containing mask files with .tif, .tiff, or .npy extensions.",
})

# A second, deliberately target-neutral review pass for prose where the local
# models chose a literal English cognate or a UI false friend.  These are still
# English model inputs (never hand-written target text), so every locale is
# regenerated from the same semantic contract and Python docstrings remain the
# canonical English source.
API_TRANSLATION_CONTEXT.update({
    (
        "Auto-chaining is only welcome while it is filling in a blank.  The "
        "moment a user types a path of their own, that path is theirs: it "
        "survives a reopen, a restart, and every subsequent upstream run.  This "
        "is the record that makes that true, and it is deliberately *not* the "
        "settings dict — a settings dict cannot distinguish \"the user chose "
        "this\" from \"we put it there\"."
    ): (
        "Auto-chaining fills only blank values. Once a user enters a path, that "
        "path belongs to the user and survives reopening, restarting, and later "
        "upstream runs. This *separate record* stores that choice. The settings "
        "collection cannot distinguish \"the user chose this\" from \"we put it "
        "there\"."
    ),
    (
        "``(settings, defaulted_keys, defaults_source)`` where "
        "``defaulted_keys`` are the keys the caller did **not** set and the "
        "script is therefore pinning on their behalf."
    ): (
        "``(settings, defaulted_keys, defaults_source)`` where "
        "``defaulted_keys`` contains keys the caller **did not provide**. The "
        "script supplies those values for the caller."
    ),
    (
        "Does **not** start an event loop: see :func:`run_event_loop` for why "
        "that is a separate decision."
    ): (
        "**The caller decides whether to start the event loop.** Details: "
        ":func:`run_event_loop`."
    ),
    (
        "Deliberately does **not** chain to ``super()`` when it fills. The base "
        "implementation is what draws the stylesheet background, and the "
        "stylesheet background is the ``bg`` slab being replaced — calling it "
        "afterwards would paint black straight back over this."
    ): (
        "The implementation deliberately **does not call** ``super()`` when it "
        "fills. The base implementation draws the stylesheet background, the "
        "``bg`` slab being replaced. Calling it afterwards would paint black "
        "over this fill."
    ),
    (
        "**Off the GUI thread.** Scanning a plate's worth of ND2 headers takes "
        "seconds; converting takes minutes. Both go through "
        ":func:`spacr.qt.bridge.make_thread`, and the completion handler is "
        "reached through a *bound method* "
        "(:attr:`ConvertScreen._job_settled`) rather than a closure, because "
        "``PipelineWorker.finished`` is emitted in the worker thread and a "
        "closure connected to it would build widget children there. Tests pass "
        "``threaded=False``."
    ): (
        "**Background execution.** Scanning ND2 headers takes seconds and "
        "conversion takes minutes. Both operations use "
        ":func:`spacr.qt.bridge.make_thread`. Completion calls "
        "*:attr:`ConvertScreen._job_settled`* on the GUI object instead of a "
        "closure. ``PipelineWorker.finished`` is emitted during worker "
        "execution; a connected closure would construct widgets there. Tests "
        "pass ``threaded=False``."
    ),
    (
        "1. ``set_default_settings_preprocess_generate_masks`` defaults it to "
        "the **string** ``'cell'``, and ``measure.py`` tests it with "
        "``\"organelle\" in settings['summarize_organelles_by']`` — a "
        "*substring* test when the value is a str. Running this demo with "
        "``summarize_organelles_by='cell'`` gives ``cell_organelle_summary`` "
        "(16 rows/field) and still **no ``organelle`` table**. Only a value "
        "containing ``'organelle'`` writes the per-organelle table "
        "(``['cell', 'organelle']`` → organelle 64 rows/field, verified). 2. A "
        "list cannot be shipped today: "
        "``spacr.settings.expected_types`` declares "
        "``'summarize_organelles_by': str``, so "
        "``spacr.validate.validate_settings`` rejects "
        "``['cell', 'organelle']`` with \"is a list, but str is expected\" — a "
        "hard pre-flight **error** on a demo that must load clean. The tooltip "
        "and ``spacr.gui_utils`` both describe it as a list, and "
        "``spacr.external_masks`` builds one; only the type table disagrees."
    ): (
        "1. **The starting value is text.** "
        "``set_default_settings_preprocess_generate_masks`` sets it to "
        "``'cell'``. ``measure.py`` evaluates "
        "``\"organelle\" in settings['summarize_organelles_by']`` for a str. "
        "*This membership test accepts substrings.* Running with "
        "``summarize_organelles_by='cell'`` produces "
        "``cell_organelle_summary`` (16 rows/field). **No output table is "
        "produced.** The ``organelle`` table appears only for a value containing "
        "``'organelle'`` (``['cell', 'organelle']`` → 64 rows/field in that "
        "table, verified). 2. Lists cannot currently be supplied. "
        "``spacr.settings.expected_types`` declares "
        "``'summarize_organelles_by': str``, so "
        "``spacr.validate.validate_settings`` rejects "
        "``['cell', 'organelle']`` with \"is a list, but str is expected\". "
        "**This is a validation conflict.** The demo must load cleanly. The "
        "tooltip and ``spacr.gui_utils`` describe a list, and "
        "``spacr.external_masks`` constructs one; the type table conflicts with "
        "them."
    ),
    (
        "The controls are **not** placed on the screen. They live in a popover "
        "behind a ``DNA`` toggle built from the same class as the ``AI`` toggle "
        "beside it; a decorative backdrop does not get to keep a permanent strip "
        "of a screen whose job is a settings form."
    ): (
        "The controls remain **outside the main screen**. A popover opens from "
        "the ``DNA`` switch, built from the same class as the adjacent ``AI`` "
        "switch. A decorative backdrop should not permanently occupy space on a "
        "screen used for a settings form."
    ),
    (
        "**Not** ``column_kinds() == CONTINUOUS``, and the difference matters. "
        ":func:`~spacr.qt.widgets.data_filter_panel.classify_columns` calls a "
        "numeric column with twelve or fewer distinct values a *category*, which "
        "is the right rule for deciding whether to offer a slider or a tick list "
        "— and the wrong one here, because ``pathogen_count`` runs 0–8 and is "
        "exactly the kind of feature a ranking exists to surface. A separation "
        "statistic is perfectly happy on a discrete count."
    ): (
        "**This uses a different rule from** "
        "``column_kinds() == CONTINUOUS``. "
        ":func:`~spacr.qt.widgets.data_filter_panel.classify_columns` treats a "
        "numeric column with twelve or fewer distinct values as a *category*. "
        "That rule is useful for choosing a slider or a list of selectable "
        "values. Feature ranking has a different goal: discrete values remain "
        "useful. Counts from 0 to 8 remain valid ranking features. The feature "
        "named ``pathogen_count`` is one example. A separation statistic works "
        "with a discrete count."
    ),
    (
        "So the well pass is emphatically **not** \"the well contains many flagged "
        "objects\". That statistic exists — it is reported as ``flagged_share`` — "
        "and it answers a different question: it finds a well containing a few "
        "*catastrophic* objects (a segmentation blow-up, a piece of dust measured "
        "as a cell) while being blind to a uniform shift. The well-level robust "
        "score finds the uniform shift while being blind to the isolated "
        "catastrophe. Both are on the well frame because neither subsumes the "
        "other."
    ): (
        "The well-level test is **different from counting flagged objects**. "
        "``flagged_share`` provides that count and answers another question: it "
        "finds a few *catastrophic* objects, such as a segmentation failure or "
        "dust measured as a cell, but misses a uniform shift. The well-level "
        "robust score finds the uniform shift but misses an isolated catastrophe. "
        "Both belong on the well frame because neither replaces the other."
    ),
    (
        "opt in to collapsing z before linking, so that linking happens on the "
        "projection rather than on the volume. Off by default and never implied. "
        "It does **not** unlock the backends spaCR cannot drive volumetrically -- "
        "see :func:`track_4d`."
    ): (
        "**Collapse z before linking.** Linking then uses the projection instead "
        "of the volume. This option starts disabled and is never implied. "
        "spaCR still cannot process volume data with those processing engines. Details: "
        ":func:`track_4d`."
    ),
    (
        "So :func:`apply` has a second phase. Every registered app that appears in "
        "none of the three assessment tables below and carries no explicit stage "
        "is written into ``APP_STAGE`` as :data:`UNASSESSED_STAGE` — alpha — which "
        "is what \"nobody has checked this one yet\" means. It is a *default*, not "
        "a demotion: a module that declares beta keeps beta, and an assessment "
        "recorded here always wins over both."
    ): (
        "So :func:`apply` has a second phase. Each registered app absent from all "
        "three assessment tables and carrying no explicit stage is written to "
        "*``APP_STAGE``* as :data:`UNASSESSED_STAGE` — alpha, meaning \"nobody "
        "has checked this one yet\". This assigns a starting label and does not "
        "demote anything. A module declaring beta keeps beta, and a recorded "
        "assessment takes precedence."
    ),
    (
        "**Off the GUI thread.** Cheap is not free, and opening the picker is four "
        "sequential sqlite round trips — open, list tables, read one table's "
        "columns, estimate its rows — every one of which used to happen inside "
        "``__init__``, before the modal appeared. Measured cold on a 383 MB "
        "measurements.db that is 45 ms, and on a 1 500-table schema 87 ms, entirely "
        "between the click and any window. The button now builds the dialog with "
        "``threaded=True`` and the reads arrive from a "
        ":class:`~spacr.qt.job_runner.JobRunner`. The default is still the "
        "synchronous mode, deliberately; :class:`ColumnPickerDialog` says why."
    ): (
        "**Background loading.** The picker does not freeze the GUI. It performs "
        "four sequential sqlite queries: open, list tables, read one "
        "table's columns, and estimate its rows. Previously, before a dialog "
        "appeared, all four queries ran in ``__init__``. Cold measurements were 45 ms "
        "for measurements.db, a 383 MB database, and 87 ms for a schema with "
        "1 500 tables. The dialog now selects background mode with "
        "``threaded=True``. All reads are performed by "
        ":class:`~spacr.qt.job_runner.JobRunner`. Synchronous mode remains the "
        "initial choice; :class:`ColumnPickerDialog` explains why."
    ),
    (
        "The half of the routing contract a *caller* uses. A scatter plot::"
    ): (
        "This half of the routing contract is used by *code that opens the "
        "view*. A scatter plot::"
    ),
    (
        "optional dict, populated in place with ``{field: reason}`` for every "
        "candidate that was **rejected**. This is how the caller can report "
        "\"3 fields rejected as truncated\" rather than silently doing more "
        "work."
    ): (
        "optional dict populated in place with ``{field: reason}`` for each "
        "candidate. **Rejected candidates** are recorded there. Reporting code "
        "can then say \"3 fields rejected as truncated\" instead of doing extra "
        "work without explanation."
    ),
    (
        "Parameters: df (pd.DataFrame): The input DataFrame. group_col (str): "
        "Column name to group by, or a list of column names. Grouping passes "
        "``observed=False``, so unused categories of a Categorical are kept. Rows "
        "whose group key is missing are discarded, because pandas drops ``NaN`` "
        "group keys and the per-row bound then comes back ``NaN``. value_col "
        "(str): Column containing values to check for outliers. A name that is "
        "not in the frame raises ``KeyError``. method (str): 'iqr' or 'zscore'. "
        "Anything else raises ``ValueError``. The two now agree on tiny groups: "
        "a one-row group has an undefined standard deviation, and since one row "
        "cannot be an outlier within its own group it is KEPT under both. It used "
        "to be dropped by 'zscore' and kept by 'iqr'. threshold (float): "
        "Multiplier on the IQR (default 1.5), or the z-score cutoff. Must be >= 0; "
        "a negative value inverts the keep-band and is refused, because under "
        "'iqr' it silently emptied every group with a nonzero IQR. Note ``0`` "
        "under 'zscore' still keeps only rows sitting exactly on the group mean, "
        "which is what a zero cutoff means. Under 'zscore' an outlier inflates "
        "its own group's standard deviation, so the usual cutoffs keep far more "
        "than 'iqr' does on the same data -- that is the statistic, not a defect."
    ): (
        "Parameters: df (pd.DataFrame): Input DataFrame. group_col (str): One "
        "column name or a list of names. Grouping uses ``observed=False``, so "
        "unused Categorical categories remain. Rows with missing group keys are "
        "discarded because pandas drops ``NaN`` keys and their row bounds become "
        "``NaN``. value_col (str): Column containing the values. An absent column "
        "raises ``KeyError``. method (str): 'iqr' or 'zscore'; other values raise "
        "``ValueError``. Both methods retain a one-row group: its undefined "
        "standard deviation is kept under 'zscore', matching 'iqr'. threshold "
        "(float): IQR multiplier (1.5 initially) or z-score cutoff. It must be "
        ">= 0. A negative value would empty nonzero-IQR groups under 'iqr'. With "
        "``0``, 'zscore' keeps only values equal to the group mean. Under "
        "'zscore', an outlier increases its group's standard deviation, so usual "
        "cutoffs retain more rows than 'iqr'; this follows from the statistic."
    ),
    (
        "Nodes are keyed by :func:`spacr.selection.object_keys`, the same string "
        "the UMAP, the plate view and the crop grid use, so selecting a node in "
        "the tree publishes something every other view already understands. A "
        "child's parent is resolved *within its own field*: ``cell_id`` is a "
        "label, not a key, and label 7 exists in every field on the plate."
    ): (
        "Tree nodes use :func:`spacr.selection.object_keys`. The same identifier "
        "appears in three views: UMAP, the plate view, and the crop grid. Selecting "
        "a tree node therefore publishes an identifier understood by all three. "
        "Resolve each child's parent *only inside its own field*. ``cell_id`` is "
        "a label rather than a unique key; label 7 can appear in each field."
    ),
    (
        "**Chunked.** The array is stored as a grid of independently compressed "
        "blocks, so a 100 GB plate is readable a tile at a time. Nothing here ever "
        "reads an array to answer a question about it: :func:`read_ome_zarr` opens "
        "a handful of small JSON files and returns the levels, their shapes, their "
        "voxel sizes and their chunk grids without touching a single chunk, and "
        ":meth:`OmeZarrImage.read` with a ``region=`` decodes only the chunks that "
        "region intersects. Every byte of chunk data in the pure-Python path "
        "passes through one function, :func:`_read_chunk_bytes`, precisely so "
        "that \"it is lazy\" is a countable claim rather than a sentence in a "
        "docstring — ``tests/test_ome_zarr.py`` counts its calls."
    ): (
        "**Chunked storage.** Independently compressed blocks make a 100 GB plate "
        "readable one tile at a time. Metadata queries do not load array data. "
        ":func:`read_ome_zarr` reads small JSON files and reports levels, shapes, "
        "voxel sizes, and chunk grids without reading a chunk. "
        ":meth:`OmeZarrImage.read` with ``region=`` decodes only intersecting "
        "chunks. In the pure-Python path, all chunk bytes pass through "
        ":func:`_read_chunk_bytes`. Therefore ``tests/test_ome_zarr.py`` can count "
        "calls and verify lazy loading directly."
    ),
    (
        "``beta_g`` is the evidence that gene *g* is a hit: higher means the wells "
        "carrying more of gene *g* had more positive cells than their cell count "
        "alone explains.  There are far more genes than wells (``p >> n``), so "
        "``beta`` is given a **regularized horseshoe** prior (Piironen & Vehtari, "
        "*Electron. J. Statist.* 11(2), 2017) — heavy tails so genuine hits escape "
        "shrinkage, a sharp spike at zero so the hundreds of non-hits collapse "
        "onto it, and a Student-t slab so the tails stay proper."
    ): (
        "Evidence score: *``beta_g``*. Higher values mean that wells carrying "
        "more of a gene have more positive cells than cell count alone explains. "
        "The relation ``p >> n`` means that genes greatly outnumber wells. "
        "Therefore *``beta``* uses a **regularized horseshoe prior** (Piironen & "
        "Vehtari, *Electron. J. Statist.* 11(2), 2017). Heavy tails preserve "
        "genuine hits, a sharp zero spike shrinks non-hits, and a Student-t slab "
        "keeps the tails proper."
    ),
    (
        "The imaging and sequencing plates are both simulated from the *same* "
        "spot plate — which genotypes are in a well is one physical fact, "
        "observed twice. Each stage draws from its own spawned child stream, so "
        "changing the number of imaging cells does not shift the sequencing "
        "draws; that independence is what makes a parameter sweep interpretable, "
        "since otherwise every point on the sweep would differ by an unrelated "
        "re-randomisation as well as by the parameter."
    ): (
        "Both imaging and sequencing plates originate from *one shared source*: "
        "the spot plate. Thus the genotypes in a well are one physical fact "
        "observed twice. Each "
        "stage uses an independent random stream. Changing the imaging-cell count "
        "therefore leaves sequencing draws unchanged. This separation allows "
        "meaningful comparison across a sweep of parameter values. Otherwise each "
        "point would also contain unrelated randomization."
    ),
    (
        "**Geometric units are not fixed any more.** A 2-D run measures in "
        "pixels, but a 3-D run measures a volume, and with ``voxel_size_z_um`` / "
        "``voxel_size_xy_um`` set it measures in micrometres — under the *same* "
        "column names, because :mod:`spacr.measure` deliberately does not rename "
        "``<object>_area`` (renaming would break every downstream selector). "
        "Which one a row is in is recorded on the row itself, in "
        "``measurement_units``. So the unit of a geometric column is a "
        ":class:`ConditionalUnit`: :func:`describe_database` reads "
        "``measurement_units`` out of the database it is documenting and "
        "resolves it, and a caller who has no database says so and gets the "
        "condition spelled out instead of a confident guess. See "
        ":data:`MEASUREMENT_UNITS`."
    ): (
        "**Geometric units vary.** A 2-D run records pixels, while a 3-D run "
        "records volume. Micrometre units are used when both size settings are "
        "provided: ``voxel_size_z_um`` / ``voxel_size_xy_um``. Column names "
        "remain *unchanged* because "
        ":mod:`spacr.measure` cannot rename ``<object>_area`` without breaking "
        "downstream selectors. Each row records its unit in "
        "``measurement_units``. A geometric column therefore uses a "
        ":class:`ConditionalUnit`. :func:`describe_database` reads "
        "``measurement_units`` from the database and resolves the condition. "
        "Without a database it reports the condition instead of guessing. "
        "Details: :data:`MEASUREMENT_UNITS`."
    ),
    (
        "Both modes run the *same* reads in the same order through the same "
        ":func:`read_schema`; the runner is simply constructed unthreaded in the "
        "first, which makes it call its job inline."
    ): (
        "*Identical operations in both modes.* Each mode calls "
        ":func:`read_schema` in the same order. The first mode executes the job "
        "immediately; the second mode executes it in the background."
    ),
    (
        "**Why not accumulate.** Walking a row and adding each measured shift to "
        "the previous position gives ``p_k = sum(d_0..d_k)``, so every measurement "
        "error is carried forward: ten tiles with 0.3 px errors put the last one "
        "3 px out, and nothing in the output says so. Least squares over the "
        "*same* measurements distributes the error instead — and because a real "
        "grid has redundant edges (the tile below as well as the tile to the "
        "right, or the ``i``/``i+2`` pairs of a heavily overlapped row), "
        "disagreements cancel rather than compound. The surviving disagreement "
        "is the per-tile residual, which is returned."
    ): (
        "**Why accumulation fails.** Adding every measured shift to the previous "
        "position gives ``p_k = sum(d_0..d_k)`` and carries each measurement error "
        "forward. Ten tiles with 0.3 px errors place the last tile 3 px away from "
        "its correct position. Least squares distributes errors across "
        "*identical input measurements*. A real grid has redundant edges, "
        "including the tile below, the tile to the right, and the ``i``/``i+2`` "
        "pairs in a heavily overlapped row. Their disagreements can cancel "
        "instead of accumulating. The returned per-tile residual records the "
        "remaining disagreement."
    ),
    (
        "the gate's display name. It is stripped, every run of characters outside "
        "``[0-9A-Za-z_]`` collapses to one underscore, leading and trailing "
        "underscores are dropped, and a result starting with a digit is prefixed "
        "``g_`` -- legal in quoted SQLite but not in the tools that read the table "
        "afterwards. Two gates whose names differ only in punctuation therefore "
        "land on the SAME column, and :func:`export_gate` replaces rather than "
        "suffixes."
    ): (
        "the visible gate name. Remove surrounding whitespace, replace each run "
        "of characters outside ``[0-9A-Za-z_]`` with one underscore, and remove "
        "underscores at either "
        "end. Prefix ``g_`` when the result starts with a digit. This is legal in "
        "SQLite and compatible with downstream tools. Gate names differing only "
        "in punctuation produce one identical output column. "
        ":func:`export_gate` replaces that column instead of adding a suffix."
    ),
    (
        "The binarisation cut used by the hinge fit; the bootstrap below must "
        "reproduce the SAME two classes the fit saw."
    ): (
        "Binarization cutoff used by the hinge model. The bootstrap must reproduce "
        "exactly the two classes used during fitting."
    ),
    "One model over one field set. Only comparable to results on the *same* set.": (
        "One model evaluated on one field set. Compare it only with results from "
        "*that identical field set*."
    ),
    (
        "The probe is pulled off the *same* lazy generator the planner then "
        "drains, so a folder whose layout is not recognisable costs 30 files, not "
        "a traversal."
    ): (
        "The probe consumes 30 entries from *the generator later consumed by the "
        "planner*. An unrecognized folder therefore costs only 30 files rather "
        "than a complete traversal."
    ),
    (
        "Timelapse needs multi-T frames per (well, field) so tracking has something "
        "to lock onto. Same cellvoyager naming, just with T01..T<N>, and every "
        "frame holds the *same* cells drifting a couple of pixels rather than a "
        "fresh random field."
    ): (
        "Timelapse needs multiple time frames per (well, field) so tracking can "
        "link objects. It uses cellvoyager naming with T01..T<N>. Every frame "
        "contains *one unchanged cell population* shifted by a few pixels, not a "
        "new random field."
    ),
    (
        "The locked dock moves the sidebar into the window's own layout; switching "
        "back to the reveal has to move the *same* object here again — building a "
        "second Sidebar would leave the tutorial, the command palette and every "
        "test pointing at the dead one."
    ): (
        "Locked mode moves the sidebar into the window layout. Returning to reveal "
        "mode must move *that existing object* back. Building a second Sidebar "
        "would leave the tutorial, command palette, and tests referencing the "
        "obsolete first object."
    ),
    (
        "Re-hovering the SAME setting keeps its reveal, so moving the pointer "
        "between a label and the popup below it does not fight the reader. The "
        "state is one key and one bool — see "
        ":meth:`HoverTooltip.animations_shown`."
    ): (
        "Returning to an unchanged setting preserves its visible help bubble. "
        "Moving the pointer between a label and that bubble therefore does not "
        "restart the animation. The state stores one key and one Boolean value. Details: "
        ":meth:`HoverTooltip.animations_shown`."
    ),
    (
        "The Timelapse module is mask generation over a time series followed by "
        "frame-to-frame linking. Those two halves cost wildly different amounts: "
        "segmenting twelve frames with Cellpose is tens of seconds, re-linking the "
        "*same* twelve label images with a new ``timelapse_displacement`` is "
        "milliseconds. A preview that re-segments on every slider move is "
        "unusable, so this panel splits them:"
    ): (
        "Timelapse first generates masks for a time series and then links adjacent "
        "frames. The costs differ greatly: segmenting twelve frames with Cellpose "
        "takes tens of seconds, while relinking *the twelve existing label images* "
        "with a new ``timelapse_displacement`` takes milliseconds. The preview "
        "separates these stages so a slider change does not rerun segmentation:"
    ),
    (
        "A 2-D field measures areas in px^2; a 3-D field measures volumes, in "
        "voxels or um^3, and writes them into the *same* ``<object>_area`` column, "
        "because that column is read by name by every downstream selector, model "
        "and threshold ever written against a spaCR database and renaming it "
        "would break all of them silently. Appending both into one table would "
        "therefore leave a numeric column that mixes two incompatible quantities "
        "with nothing in the row to tell them apart, which no amount of downstream "
        "care could recover from. So it is refused here instead."
    ): (
        "A 2-D field measures area in px^2; a 3-D field measures volume in voxels "
        "or um^3. Both write to *one shared* ``<object>_area`` column because spaCR "
        "selectors, models, and thresholds read that name. Renaming it would "
        "silently break them. Combining both field types in one table would mix "
        "incompatible quantities without a row-level unit marker. Downstream code "
        "could not recover the distinction, so this operation is rejected."
    ),
    (
        "``QThread.finished`` must be connected to a **bound method of a GUI-thread "
        "QObject**, never a closure. PySide6 makes the QThread itself the receiver "
        "for a closure, and :func:`spacr.qt.bridge.make_thread` connects "
        "``thread.finished -> thread.deleteLater`` first. Slots run in connection "
        "order, so the DeferredDelete is posted ahead of the closure's metacall "
        "and Qt discards queued events for a destroyed receiver: the job is never "
        "retired and ``active_jobs()`` never returns to zero."
    ): (
        "**QObject receiver requirement.** Connect ``QThread.finished`` to a "
        "method on a QObject that belongs to the GUI thread. A closure is unsafe. "
        "PySide6 assigns the QThread itself as a closure's receiver. "
        ":func:`spacr.qt.bridge.make_thread` first connects "
        "``thread.finished -> thread.deleteLater``. Slots execute in connection "
        "order. DeferredDelete is therefore posted before the closure callback. "
        "Qt discards the queued callback after destroying its receiver. The job "
        "then remains active and ``active_jobs()`` never reaches zero."
    ),
    (
        ":class:`_ConsoleRelay` is the fix: a QObject pinned to the GUI thread "
        "whose ``line`` signal is connected to its own **bound method**, so Qt "
        "queues the delivery whenever the emitting thread is not the GUI thread "
        "and the panel only ever touches widgets on the thread that owns them."
    ): (
        "**GUI-thread relay.** :class:`_ConsoleRelay` is a QObject that belongs "
        "to the GUI thread. Its ``line`` signal calls one method on that relay. "
        "Qt queues a signal emitted from another thread. Therefore the panel "
        "accesses widgets only from their owning thread."
    ),
    (
        "**Lazy, and that is the whole point.** The drop handlers need two things "
        "from a dropped folder — a small probe to guess the layout from, and "
        "(only if the guess succeeds) the full file list to plan an extraction. "
        "Pulling both off one generator means the tree is traversed once; "
        "abandoning the generator after the probe means a folder with no layout "
        "to detect is never fully walked at all. The previous shape returned "
        "lists, and the same tree was walked three times per drop."
    ): (
        "**File iteration is deferred until needed.** Drag-and-drop handlers need "
        "a small layout probe from a dropped folder and, only after a successful "
        "probe, the complete file list for extraction planning. Both come from "
        "one generator, so the folder tree is traversed once. Abandoning the "
        "generator after a failed probe avoids a complete traversal. The previous "
        "list-based design traversed the same tree three times for each drop "
        "operation."
    ),
    (
        "``blobs``   (default) Big and small colour blobs drifting over the page, "
        "each pulsing in size on its own period. They overlap and blend, so the "
        "result reads as soft colour *fields* rather than as a bag of circles. "
        "This is the one the feature was asked for. ``aurora`` Three overlapping "
        "curtains of vertical rays, folding along their own length. The folds are "
        "travelling waves — several superposed frequencies running lengthwise "
        "along the arc — with brightness surges on a separate schedule, a sharp "
        "lower edge, a diffuse top, and the real thing's vertical colour order: "
        "green through the body, red high up, a violet fringe underneath. "
        "``ripple`` Concentric rings expanding out of three fixed sources and "
        "fading as they grow, like rain on water. Soft-edged, so it never reads as "
        "line work. ``drift`` A slow starfield in three parallax layers: small, "
        "dim, slow ones behind; bigger, brighter, faster ones in front. The one "
        "crisp theme. It travels up, down, or every which way — see "
        ":data:`DRIFT_DIRECTIONS`. ``bokeh`` Out-of-focus points of light, the way "
        "a fluorescence field looks off the focal plane: an aperture image is a "
        "*disc with a bright rim*, not a Gaussian smudge, and the ones further out "
        "of focus are larger and flatter. ``cells`` Cells drifting through the "
        "field, turning as they go — a soft body, a slightly brighter membrane "
        "where the edge is seen nearly edge-on, and a distinctly brighter nucleus "
        "set off centre."
    ): (
        "``blobs`` (default): Large and small colour blobs drift and pulse over "
        "the page. Their overlap forms soft colour *fields* instead of separate "
        "circles; this is the requested mode. ``aurora``: Three curtains of "
        "vertical rays fold as travelling waves, with independent brightness "
        "surges, a sharp lower edge, and a diffuse top. Colour runs from green "
        "through the body to red above and violet below. ``ripple``: Soft "
        "concentric rings expand from three fixed sources and fade like rain on "
        "water. ``drift``: A starfield moves in three parallax layers, with dim, "
        "slow stars behind and brighter, faster stars in front. Direction is "
        "controlled by :data:`DRIFT_DIRECTIONS`. ``bokeh``: Out-of-focus light "
        "resembles fluorescence away from the focal plane. Each aperture is a "
        "*disc with a bright rim*; greater defocus makes it larger and flatter. "
        "``cells``: Cells drift and turn, with a soft body, a brighter edge-on "
        "membrane, and a brighter off-centre nucleus."
    ),
})

# Compact, target-neutral expansions for four Portuguese-resistant blocks.
# The original Python docstrings remain canonical; these sentences are model
# input only, and preserve every code/product literal byte for byte.
API_TRANSLATION_CONTEXT.update({
    (
        "**The R-to-numpy trap this function exists to close:** R's ``rgamma`` "
        "takes a *rate*, numpy's takes a *scale*, and "
        "``scale = 1 / rate = var / mean``. A port that passes the rate to "
        "numpy gets a distribution wrong by a factor of "
        "``var**2 / mean**2`` in the variance while still looking entirely "
        "plausible."
    ): (
        "**This function resolves a conversion trap between R and numpy.** "
        "R's ``rgamma`` accepts a *rate*. numpy accepts a *scale*. Use "
        "``scale = 1 / rate = var / mean``. Passing the rate to numpy makes "
        "the variance wrong by ``var**2 / mean**2`` while the distribution "
        "can still look plausible."
    ),
    "True iff no item is in QUEUED or RUNNING.":
        "Return True only if no item has status QUEUED or RUNNING.",
    (
        "Each item's status transitions QUEUED → RUNNING → SUCCESS/FAILED. "
        "If ``stop_on_error`` is True, the first failure halts the loop with "
        "remaining items left as QUEUED."
    ): (
        "Each item's status changes QUEUED → RUNNING → SUCCESS/FAILED. If "
        "``stop_on_error`` is True, the first failure stops the "
        "loop; all remaining items stay QUEUED."
    ),
    "Fold-to-fold sd for a ``'mean'`` series, else ``None``.":
        "Standard deviation across folds for a ``'mean'`` series; otherwise "
        "``None``.",
})

# The project summary is the first prose a reader sees on GitHub and contains
# several domain-sensitive senses (pooled screen, plate, guide and hit).  Keep
# these two blocks human-reviewed instead of trusting a generic model to pick
# the scientific meaning from a short sentence.
_SUMMARY_SOURCE = (
    "spaCR segments and measures single cells in high-content microscopy "
    "images, links each cell to the gRNA it received, and reports which genes "
    "changed the phenotype. Plate images and FASTQ reads go in; per-object "
    "measurements, trained classifiers, per-guide and per-gene effect sizes, "
    "and a ranked hit list come out."
)
_SCOPE_SOURCE = (
    "If you run image-based pooled CRISPR screens, that is the whole path. If "
    "you have high-content microscopy and no screen, the segmentation, "
    "measurement, annotation and classification half runs on its own."
)
_TAGLINE_SOURCE = "**Spatial phenotype analysis of CRISPR screens.**"
_ATTRIBUTION_SOURCE = "Translation model attribution"
_STORAGE_SOURCE = (
    "Images, masks, crops, measurements, annotations, predictions, barcodes "
    "and well identifiers live in one SQLite project, so a number in a "
    "result can be traced back to the object it came from."
)
_EXECUTION_SOURCE = (
    "Run spaCR as a desktop application or headlessly on a workstation, "
    "server or cluster. Both drive the same modules, and CUDA is used "
    "automatically where a module supports it."
)
_WORKFLOW_SOURCE = (
    "Microscopy images (TIFF, OME-TIFF, LIF, CZI, ND2) and sequencing reads "
    "(FASTQ) enter complementary image-analysis and barcode-mapping "
    "pipelines. Object tables, crops, annotations, predictions, guide "
    "identities, QC results and well-level summaries are then analyzed "
    "together."
)
REVIEWED_README_BLOCKS = {
    _SUMMARY_SOURCE: {
        "sv": "spaCR segmenterar och mäter enskilda celler i mikroskopibilder med högt innehåll, kopplar varje cell till den gRNA den fick och rapporterar vilka gener som förändrade fenotypen. Plattbilder och FASTQ-läsningar matas in; ut kommer mätningar per objekt, tränade klassificerare, effektstorlekar per guide och gen samt en rangordnad träfflista.",
        "de": "spaCR segmentiert und vermisst einzelne Zellen in High-Content-Mikroskopiebildern, verknüpft jede Zelle mit der erhaltenen gRNA und berichtet, welche Gene den Phänotyp verändert haben. Plattenbilder und FASTQ-Reads dienen als Eingabe; ausgegeben werden Messungen pro Objekt, trainierte Klassifikatoren, Effektgrößen pro Guide und Gen sowie eine Rangliste der Treffer.",
        "es": "spaCR segmenta y mide células individuales en imágenes de microscopía de alto contenido, vincula cada célula con el gRNA que recibió e indica qué genes modificaron el fenotipo. Las imágenes de placas y las lecturas FASTQ son la entrada; las mediciones por objeto, los clasificadores entrenados, los tamaños del efecto por guía y por gen y una lista ordenada de resultados son la salida.",
        "zh_CN": "spaCR 对高内涵显微镜图像中的单细胞进行分割和测量，将每个细胞与其获得的 gRNA 关联，并报告哪些基因改变了表型。输入为孔板图像和 FASTQ 读段；输出包括逐对象测量、训练后的分类器、逐向导 RNA 和逐基因效应量，以及按优先级排序的候选结果列表。",
        "pt": "O spaCR segmenta e mede células individuais em imagens de microscopia de alto conteúdo, associa cada célula ao gRNA que ela recebeu e informa quais genes alteraram o fenótipo. As entradas são imagens de placas e leituras FASTQ; as saídas incluem medições por objeto, classificadores treinados, tamanhos de efeito por guia e por gene e uma lista classificada de resultados.",
        "hi": "spaCR उच्च-सामग्री माइक्रोस्कोपी छवियों में एकल कोशिकाओं का विभाजन और मापन करता है, प्रत्येक कोशिका को मिले gRNA से जोड़ता है और बताता है कि किन जीनों ने फीनोटाइप बदला। इनपुट के रूप में प्लेट छवियाँ और FASTQ रीड आती हैं; आउटपुट में प्रति-वस्तु मापन, प्रशिक्षित वर्गीकारक, प्रति-गाइड और प्रति-जीन प्रभाव आकार तथा प्राथमिकता के अनुसार परिणामों की सूची मिलती है।",
        "ko": "spaCR는 고함량 현미경 영상에서 단일 세포를 분할하고 측정하며, 각 세포를 전달받은 gRNA와 연결하고 어떤 유전자가 표현형을 바꾸었는지 보고합니다. 플레이트 영상과 FASTQ 리드를 입력하면 객체별 측정값, 학습된 분류기, 가이드별·유전자별 효과 크기와 우선순위가 지정된 후보 목록이 출력됩니다.",
        "is": "spaCR aðgreinir og mælir stakar frumur í afkastamiklum smásjármyndum, tengir hverja frumu við gRNA-ið sem hún fékk og greinir frá því hvaða gen breyttu svipgerðinni. Plötumyndir og FASTQ-raðir eru inntak; mælingar fyrir hvert viðfang, þjálfaðir flokkarar, áhrifastærðir fyrir hverja leiðarsameind og hvert gen og forgangsraðaður niðurstöðulisti eru úttak.",
        "fr": "spaCR segmente et mesure les cellules individuelles dans des images de microscopie à haut contenu, associe chaque cellule au gRNA qu’elle a reçu et indique quels gènes ont modifié le phénotype. Les images de plaques et les lectures FASTQ constituent les entrées ; les mesures par objet, les classificateurs entraînés, les tailles d’effet par guide et par gène et une liste de résultats classés constituent les sorties.",
    },
    _SCOPE_SOURCE: {
        "sv": "För bildbaserade poolade CRISPR-screeningar täcker detta hela arbetsflödet. Om du har mikroskopi med högt innehåll men ingen screening kan delarna för segmentering, mätning, annotering och klassificering köras fristående.",
        "de": "Für bildbasierte gepoolte CRISPR-Screens deckt dies den gesamten Arbeitsablauf ab. Bei High-Content-Mikroskopie ohne Screen können Segmentierung, Messung, Annotation und Klassifizierung eigenständig ausgeführt werden.",
        "es": "Para los cribados CRISPR agrupados y basados en imágenes, este es el flujo de trabajo completo. Si dispone de microscopía de alto contenido sin cribado, las etapas de segmentación, medición, anotación y clasificación pueden ejecutarse por separado.",
        "zh_CN": "对于基于图像的混合 CRISPR 筛选，这涵盖了完整工作流程。如果只有高内涵显微镜数据而没有筛选实验，也可以单独运行分割、测量、标注和分类部分。",
        "pt": "Para triagens CRISPR agrupadas e baseadas em imagens, esse é o fluxo de trabalho completo. Se você tiver microscopia de alto conteúdo sem uma triagem, as etapas de segmentação, medição, anotação e classificação poderão ser executadas de forma independente.",
        "hi": "छवि-आधारित पूल्ड CRISPR स्क्रीनिंग के लिए यह पूरा कार्यप्रवाह है। यदि आपके पास उच्च-सामग्री माइक्रोस्कोपी है लेकिन कोई स्क्रीनिंग नहीं है, तो विभाजन, मापन, एनोटेशन और वर्गीकरण वाले भाग स्वतंत्र रूप से चलाए जा सकते हैं।",
        "ko": "영상 기반 풀드 CRISPR 스크리닝에서는 이것이 전체 작업 흐름입니다. 고함량 현미경 데이터만 있고 스크리닝 실험은 없는 경우에도 분할, 측정, 주석 및 분류 단계를 독립적으로 실행할 수 있습니다.",
        "is": "Fyrir myndgreindar samsettar CRISPR-skimanir nær þetta yfir allt verkflæðið. Ef þú ert með afkastamiklar smásjármyndir en enga skimun er hægt að keyra aðgreiningu, mælingar, merkingar og flokkun sjálfstætt.",
        "fr": "Pour les criblages CRISPR groupés fondés sur l’imagerie, ce flux couvre l’ensemble du parcours. Avec des images de microscopie à haut contenu mais sans criblage, les étapes de segmentation, de mesure, d’annotation et de classification peuvent être exécutées indépendamment.",
    },
    _TAGLINE_SOURCE: {
        "sv": "**Rumslig fenotypanalys av CRISPR-screeningar.**",
        "de": "**Räumliche Phänotypanalyse von CRISPR-Screens.**",
        "es": "**Análisis espacial del fenotipo en cribados CRISPR.**",
        "zh_CN": "**CRISPR 筛选的空间表型分析。**",
        "pt": "**Análise espacial de fenótipos em triagens CRISPR.**",
        "hi": "**CRISPR स्क्रीनिंग का स्थानिक फीनोटाइप विश्लेषण।**",
        "ko": "**CRISPR 스크리닝의 공간 표현형 분석.**",
        "is": "**Rýmisbundin svipgerðargreining á CRISPR-skimunum.**",
        "fr": "**Analyse spatiale des phénotypes de criblages CRISPR.**",
    },
    _ATTRIBUTION_SOURCE: {
        "sv": "Information om översättningsmodellerna",
        "de": "Angaben zu den Übersetzungsmodellen",
        "es": "Información sobre los modelos de traducción",
        "zh_CN": "翻译模型说明",
        "pt": "Informações sobre os modelos de tradução",
        "hi": "अनुवाद मॉडल की जानकारी",
        "ko": "번역 모델 정보",
        "is": "Upplýsingar um þýðingarlíkön",
        "fr": "Informations sur les modèles de traduction",
    },
    _STORAGE_SOURCE: {
        "sv": "Bilder, masker, bildutsnitt, mätningar, annoteringar, prediktioner, streckkoder och brunnsidentifierare lagras i ett enda SQLite-projekt, så ett värde i ett resultat kan spåras tillbaka till objektet det kom från.",
        "de": "Bilder, Masken, Bildausschnitte, Messungen, Annotationen, Vorhersagen, Barcodes und Well-Kennungen liegen in einem einzigen SQLite-Projekt. Dadurch lässt sich ein Ergebniswert bis zu seinem Ursprungsobjekt zurückverfolgen.",
        "es": "Las imágenes, máscaras, recortes, mediciones, anotaciones, predicciones, códigos de barras e identificadores de pocillo se guardan en un único proyecto SQLite, por lo que cualquier valor de un resultado puede rastrearse hasta su objeto de origen.",
        "zh_CN": "图像、掩膜、图像裁剪、测量值、标注、预测、条形码和孔位标识符都存储在同一个 SQLite 项目中，因此结果中的数值可以追溯到其来源对象。",
        "pt": "Imagens, máscaras, recortes, medições, anotações, previsões, códigos de barras e identificadores de poço ficam em um único projeto SQLite, permitindo rastrear qualquer valor de resultado até o objeto de origem.",
        "hi": "छवियाँ, मास्क, इमेज क्रॉप, मापन, एनोटेशन, पूर्वानुमान, बारकोड और वेल पहचानकर्ता एक ही SQLite प्रोजेक्ट में रहते हैं, इसलिए किसी परिणाम के मान को उसके स्रोत ऑब्जेक्ट तक वापस खोजा जा सकता है।",
        "ko": "영상, 마스크, 이미지 크롭, 측정값, 주석, 예측, 바코드 및 웰 식별자는 하나의 SQLite 프로젝트에 저장되므로 결과의 값을 그 출처 객체까지 추적할 수 있습니다.",
        "is": "Myndir, grímur, myndúrklippur, mælingar, merkingar, spár, strikamerki og brunnaauðkenni eru geymd í einu SQLite-verkefni, þannig að rekja má niðurstöðugildi aftur til viðfangsins sem það kom frá.",
        "fr": "Les images, masques, recadrages, mesures, annotations, prédictions, codes-barres et identifiants de puits sont conservés dans un même projet SQLite, ce qui permet de relier chaque valeur d’un résultat à son objet d’origine.",
    },
    _EXECUTION_SOURCE: {
        "sv": "Kör spaCR som skrivbordsprogram eller utan grafiskt gränssnitt på en arbetsstation, server eller beräkningskluster. Båda sätten använder samma moduler, och CUDA används automatiskt när modulen stöder det.",
        "de": "Führen Sie spaCR als Desktopanwendung oder ohne grafische Oberfläche auf einer Workstation, einem Server oder Cluster aus. Beide Varianten verwenden dieselben Module; CUDA wird automatisch genutzt, wenn das jeweilige Modul es unterstützt.",
        "es": "Ejecute spaCR como aplicación de escritorio o sin interfaz gráfica en una estación de trabajo, servidor o clúster. Ambos modos usan los mismos módulos y CUDA se utiliza automáticamente cuando el módulo lo admite.",
        "zh_CN": "spaCR 可作为桌面应用程序运行，也可在工作站、服务器或集群上以无图形界面模式运行。两种方式使用相同的模块；模块支持 CUDA 时会自动启用。",
        "pt": "Execute o spaCR como aplicativo para desktop ou sem interface gráfica em uma estação de trabalho, servidor ou cluster. Os dois modos usam os mesmos módulos, e o CUDA é ativado automaticamente quando houver suporte no módulo.",
        "hi": "spaCR को डेस्कटॉप एप्लिकेशन के रूप में या वर्कस्टेशन, सर्वर अथवा क्लस्टर पर बिना ग्राफ़िकल इंटरफ़ेस के चलाएँ। दोनों तरीके समान मॉड्यूल चलाते हैं और समर्थित मॉड्यूल में CUDA अपने आप उपयोग होता है।",
        "ko": "spaCR를 데스크톱 애플리케이션으로 실행하거나 워크스테이션, 서버 또는 클러스터에서 그래픽 인터페이스 없이 실행할 수 있습니다. 두 방식 모두 동일한 모듈을 사용하며, 모듈이 지원하면 CUDA가 자동으로 활성화됩니다.",
        "is": "Keyrðu spaCR sem skjáborðsforrit eða án grafísks viðmóts á vinnustöð, þjóni eða reikniklasa. Báðar leiðir nota sömu einingar og CUDA er virkjað sjálfkrafa þegar einingin styður það.",
        "fr": "Exécutez spaCR comme application de bureau ou sans interface graphique sur une station de travail, un serveur ou un cluster. Les deux modes utilisent les mêmes modules et CUDA est activé automatiquement lorsqu’un module le prend en charge.",
    },
    _WORKFLOW_SOURCE: {
        "sv": "Mikroskopibilder (TIFF, OME-TIFF, LIF, CZI, ND2) och sekvenseringsläsningar (FASTQ) matas in i kompletterande arbetsflöden för bildanalys och streckkodsmappning. Objekttabeller, bildutsnitt, annoteringar, prediktioner, guideidentiteter, QC-resultat och sammanfattningar per brunn analyseras sedan tillsammans.",
        "de": "Mikroskopiebilder (TIFF, OME-TIFF, LIF, CZI, ND2) und Sequenzierungs-Reads (FASTQ) durchlaufen einander ergänzende Pipelines für Bildanalyse und Barcode-Zuordnung. Objekttabellen, Bildausschnitte, Annotationen, Vorhersagen, Guide-Identitäten, QC-Ergebnisse und Zusammenfassungen auf Well-Ebene werden anschließend gemeinsam analysiert.",
        "es": "Las imágenes de microscopía (TIFF, OME-TIFF, LIF, CZI, ND2) y las lecturas de secuenciación (FASTQ) pasan por flujos complementarios de análisis de imágenes y asignación de códigos de barras. Después se analizan conjuntamente las tablas de objetos, los recortes, las anotaciones, las predicciones, las identidades de guía, los resultados de QC y los resúmenes por pocillo.",
        "zh_CN": "显微镜图像（TIFF、OME-TIFF、LIF、CZI、ND2）和测序读段（FASTQ）分别进入互补的图像分析与条形码映射流程。随后对对象表、图像裁剪、标注、预测、向导 RNA 身份、QC 结果和孔位级汇总进行联合分析。",
        "pt": "Imagens de microscopia (TIFF, OME-TIFF, LIF, CZI, ND2) e leituras de sequenciamento (FASTQ) entram em fluxos complementares de análise de imagens e mapeamento de códigos de barras. Em seguida, tabelas de objetos, recortes, anotações, previsões, identidades de guia, resultados de QC e resumos por poço são analisados em conjunto.",
        "hi": "माइक्रोस्कोपी छवियाँ (TIFF, OME-TIFF, LIF, CZI, ND2) और सीक्वेंसिंग रीड (FASTQ) पूरक इमेज-विश्लेषण तथा बारकोड-मैपिंग कार्यप्रवाह में जाती हैं। इसके बाद ऑब्जेक्ट तालिकाएँ, इमेज क्रॉप, एनोटेशन, पूर्वानुमान, गाइड पहचान, QC परिणाम और प्रति-वेल सारांश एक साथ विश्लेषित किए जाते हैं।",
        "ko": "현미경 영상(TIFF, OME-TIFF, LIF, CZI, ND2)과 시퀀싱 리드(FASTQ)는 서로 보완적인 영상 분석 및 바코드 매핑 작업 흐름으로 들어갑니다. 그런 다음 객체 테이블, 이미지 크롭, 주석, 예측, 가이드 식별 정보, QC 결과 및 웰 단위 요약을 함께 분석합니다.",
        "is": "Smásjármyndir (TIFF, OME-TIFF, LIF, CZI, ND2) og raðgreiningarlestur (FASTQ) fara í samverkandi ferli fyrir myndgreiningu og strikamerkjavörpun. Síðan eru viðfangstöflur, myndúrklippur, merkingar, spár, auðkenni leiðarsameinda, QC-niðurstöður og samantektir fyrir hvern brunn greind saman.",
        "fr": "Les images de microscopie (TIFF, OME-TIFF, LIF, CZI, ND2) et les lectures de séquençage (FASTQ) alimentent des flux complémentaires d’analyse d’images et d’association des codes-barres. Les tables d’objets, recadrages, annotations, prédictions, identités des guides, résultats de QC et résumés par puits sont ensuite analysés ensemble.",
    },
}

REVIEWED_README_HEADINGS = {
    "Workflow at a glance": {
        "sv": "Arbetsflödet i korthet", "de": "Workflow auf einen Blick",
        "es": "Flujo de trabajo de un vistazo", "zh_CN": "工作流程概览",
        "pt": "Visão geral do fluxo de trabalho", "hi": "कार्यप्रवाह का अवलोकन",
        "ko": "작업 흐름 개요", "is": "Yfirlit yfir verkflæðið",
        "fr": "Vue d’ensemble du flux de travail",
    },
    "Quick start": {
        "sv": "Snabbstart", "de": "Schnellstart", "es": "Inicio rápido",
        "zh_CN": "快速开始", "pt": "Início rápido", "hi": "त्वरित शुरुआत",
        "ko": "빠른 시작", "is": "Flýtiræsing", "fr": "Démarrage rapide",
    },
    "Installation details": {
        "sv": "Installationsinformation", "de": "Installationsdetails",
        "es": "Detalles de instalación", "zh_CN": "安装详情",
        "pt": "Detalhes da instalação", "hi": "स्थापना विवरण",
        "ko": "설치 세부 정보", "is": "Upplýsingar um uppsetningu",
        "fr": "Détails de l’installation",
    },
    "Lightweight installers — no conda or existing Python required": {
        "sv": "Lätta installationsprogram — varken conda eller befintlig Python krävs",
        "de": "Leichte Installationsprogramme — weder conda noch vorhandenes Python erforderlich",
        "es": "Instaladores ligeros — no requieren conda ni una instalación de Python",
        "zh_CN": "轻量级安装程序 — 无需 conda 或现有 Python 环境",
        "pt": "Instaladores leves — não exigem conda nem uma instalação existente do Python",
        "hi": "हल्के इंस्टॉलर — conda या पहले से स्थापित Python की आवश्यकता नहीं",
        "ko": "경량 설치 프로그램 — conda 또는 기존 Python 환경 불필요",
        "is": "Létt uppsetningarforrit — hvorki conda né uppsett Python nauðsynlegt",
        "fr": "Programmes d’installation légers — ni conda ni installation Python existante requis",
    },
    "Desktop application from PyPI": {
        "sv": "Skrivbordsprogram från PyPI", "de": "Desktopanwendung von PyPI",
        "es": "Aplicación de escritorio desde PyPI", "zh_CN": "通过 PyPI 安装桌面应用程序",
        "pt": "Aplicativo para desktop pelo PyPI", "hi": "PyPI से डेस्कटॉप एप्लिकेशन",
        "ko": "PyPI에서 데스크톱 애플리케이션 설치", "is": "Skjáborðsforrit frá PyPI",
        "fr": "Application de bureau depuis PyPI",
    },
    "Headless or server installation": {
        "sv": "Installation utan grafiskt gränssnitt eller på server",
        "de": "Installation ohne grafische Oberfläche oder auf einem Server",
        "es": "Instalación sin interfaz gráfica o en servidor",
        "zh_CN": "无图形界面或服务器安装", "pt": "Instalação sem interface gráfica ou em servidor",
        "hi": "बिना ग्राफ़िकल इंटरफ़ेस या सर्वर पर स्थापना",
        "ko": "그래픽 인터페이스 없이 또는 서버에 설치",
        "is": "Uppsetning án grafísks viðmóts eða á þjóni",
        "fr": "Installation sans interface graphique ou sur serveur",
    },
    "Latest development branch": {
        "sv": "Senaste utvecklingsgrenen", "de": "Neuester Entwicklungszweig",
        "es": "Rama de desarrollo más reciente", "zh_CN": "最新开发分支",
        "pt": "Ramificação de desenvolvimento mais recente", "hi": "नवीनतम विकास शाखा",
        "ko": "최신 개발 브랜치", "is": "Nýjasta þróunargrein",
        "fr": "Branche de développement la plus récente",
    },
    "Conda environments": {
        "sv": "Conda-miljöer", "de": "Conda-Umgebungen", "es": "Entornos conda",
        "zh_CN": "Conda 环境", "pt": "Ambientes conda", "hi": "Conda वातावरण",
        "ko": "Conda 환경", "is": "Conda-umhverfi", "fr": "Environnements conda",
    },
    "Optional capabilities": {
        "sv": "Valfria funktioner", "de": "Optionale Funktionen",
        "es": "Funciones opcionales", "zh_CN": "可选功能",
        "pt": "Recursos opcionais", "hi": "वैकल्पिक सुविधाएँ",
        "ko": "선택 기능", "is": "Valfrjálsir eiginleikar",
        "fr": "Fonctionnalités facultatives",
    },
    "Command-line entry points": {
        "sv": "Kommandoradskommandon", "de": "Befehle für die Kommandozeile",
        "es": "Comandos de línea de comandos", "zh_CN": "命令行入口",
        "pt": "Comandos de linha de comando", "hi": "कमांड-लाइन प्रवेश बिंदु",
        "ko": "명령줄 진입점", "is": "Skipanalínuskipanir",
        "fr": "Points d’entrée en ligne de commande",
    },
    "Features": {
        "sv": "Funktioner", "de": "Funktionen", "es": "Funciones",
        "zh_CN": "功能", "pt": "Recursos", "hi": "विशेषताएँ",
        "ko": "기능", "is": "Eiginleikar", "fr": "Fonctionnalités",
    },
    "The six modules most screens use": {
        "sv": "De sex moduler som används i de flesta screeningar",
        "de": "Die sechs Module, die in den meisten Screens verwendet werden",
        "es": "Los seis módulos más usados en los cribados",
        "zh_CN": "大多数筛选实验使用的六个模块",
        "pt": "Os seis módulos mais usados nas triagens",
        "hi": "अधिकांश स्क्रीनिंग में उपयोग होने वाले छह मॉड्यूल",
        "ko": "대부분의 스크리닝에서 사용하는 6개 모듈",
        "is": "Einingarnar sex sem flestar skimanir nota",
        "fr": "Les six modules les plus utilisés dans les criblages",
    },
    "New in 1.5.0.0": {
        "sv": "Nytt i 1.5.0.0", "de": "Neu in 1.5.0.0", "es": "Novedades de 1.5.0.0",
        "zh_CN": "1.5.0.0 新增功能", "pt": "Novidades na 1.5.0.0",
        "hi": "1.5.0.0 में नया", "ko": "1.5.0.0의 새로운 기능",
        "is": "Nýtt í 1.5.0.0", "fr": "Nouveautés de la version 1.5.0.0",
    },
    "Internationalized desktop interface": {
        "sv": "Flerspråkigt skrivbordsgränssnitt", "de": "Mehrsprachige Desktopoberfläche",
        "es": "Interfaz de escritorio multilingüe", "zh_CN": "多语言桌面界面",
        "pt": "Interface multilíngue para desktop", "hi": "बहुभाषी डेस्कटॉप इंटरफ़ेस",
        "ko": "다국어 데스크톱 인터페이스", "is": "Fjöltyngt skjáborðsviðmót",
        "fr": "Interface de bureau multilingue",
    },
    "Animated setting guidance": {
        "sv": "Animerad hjälp för inställningar", "de": "Animierte Einstellungshilfe",
        "es": "Guía animada de ajustes", "zh_CN": "动画设置指南",
        "pt": "Guia animado de configurações", "hi": "एनिमेटेड सेटिंग मार्गदर्शन",
        "ko": "애니메이션 설정 안내", "is": "Hreyfimyndaleiðbeiningar fyrir stillingar",
        "fr": "Guide animé des paramètres",
    },
    "Module reference": {
        "sv": "Modulreferens", "de": "Modulreferenz", "es": "Referencia de módulos",
        "zh_CN": "模块参考", "pt": "Referência dos módulos", "hi": "मॉड्यूल संदर्भ",
        "ko": "모듈 참조", "is": "Tilvísun eininga", "fr": "Référence des modules",
    },
    "Data": {
        "sv": "Data", "de": "Daten", "es": "Datos", "zh_CN": "数据",
        "pt": "Dados", "hi": "डेटा", "ko": "데이터", "is": "Gögn", "fr": "Données",
    },
    "Reference datasets": {
        "sv": "Referensdatauppsättningar", "de": "Referenzdatensätze",
        "es": "Conjuntos de datos de referencia", "zh_CN": "参考数据集",
        "pt": "Conjuntos de dados de referência", "hi": "संदर्भ डेटासेट",
        "ko": "참조 데이터세트", "is": "Viðmiðunargagnasöfn",
        "fr": "Jeux de données de référence",
    },
    "Contributing and support": {
        "sv": "Bidrag och support", "de": "Beiträge und Support",
        "es": "Contribuciones y soporte", "zh_CN": "贡献与支持",
        "pt": "Contribuições e suporte", "hi": "योगदान और सहायता",
        "ko": "기여 및 지원", "is": "Framlög og aðstoð",
        "fr": "Contributions et assistance",
    },
    "Licensing": {
        "sv": "Licens", "de": "Lizenz", "es": "Licencia", "zh_CN": "许可",
        "pt": "Licença", "hi": "लाइसेंस", "ko": "라이선스", "is": "Leyfi",
        "fr": "Licence",
    },
    "Tutorials": {
        "sv": "Handledningar", "de": "Tutorials", "es": "Tutoriales",
        "zh_CN": "教程", "pt": "Tutoriais", "hi": "ट्यूटोरियल",
        "ko": "튜토리얼", "is": "Kennsluefni", "fr": "Tutoriels",
    },
    "Citing spaCR": {
        "sv": "Citera spaCR", "de": "spaCR zitieren", "es": "Citar spaCR",
        "zh_CN": "引用 spaCR", "pt": "Como citar o spaCR", "hi": "spaCR का संदर्भ",
        "ko": "spaCR 인용", "is": "Tilvísun í spaCR", "fr": "Citer spaCR",
    },
}

_DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+")
_TRANSLATABLE_DIRECTIVE_RE = re.compile(
    r"^(?P<head>\s*\.\.\s+(?:note|warning|important|tip|caution|attention|"
    r"admonition)\s*::)(?P<spacing>\s*)(?P<title>.*)$"
)
_DIRECTIVE_OPTION_RE = re.compile(
    r"^\s*:[A-Za-z][\w-]*:(?:\s+.*)?$"
)
_CODE_DEFINITION_RE = re.compile(
    r"^(\s*(?:``[^`]+``|:[A-Za-z][\w:-]*:`[^`]+`)\s{2,})(.+)$"
)
_ALIGNED_LITERAL_DEFINITION_RE = re.compile(
    r"^(?P<prefix>(?:"
    r"[012]|"
    r"(?:Ctrl\+(?:[A-Za-z0-9]|1\.\.9|[,/]))|"
    r"F\d+\s+/\s+\?|Esc|"
    r"!?pathogen|NOT\s+pathogen|"
    r"cell\s+(?:AND\s+(?:NOT\s+)?pathogen|AND\s+nucleus|OR\s+nucleus)"
    r")\s{2,})(?P<prose>.+)$"
)
_UNDERLINE_RE = re.compile(r"^\s*[=~^`'\-:#*+]{3,}\s*$")
_GRID_BORDER_RE = re.compile(r"^(\s*)\+(?:[-=]+\+)+\s*$")
_SIMPLE_TABLE_BORDER_RE = re.compile(
    r"^(\s*)(?:[=~-]{3,}\s{2,})+[=~-]{3,}\s*$"
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_FIELD_RE = re.compile(
    r"^(:(?:param|parameter|arg|argument|keyword|kwarg|type|returns?|"
    r"rtype|raises?|yields?|seealso|ivar|vartype|cvar|var)\b[^:]*:)\s*(.*)$"
)


def _is_doctest_line(line: str) -> bool:
    """Match a Python doctest prompt, not ordinary prose beginning ``...``."""
    return bool(re.match(r"^\s*(?:>>>|\.\.\.)(?:\s|$)", str(line)))


def _indented_block_is_literal(
    previous_line: str, lines: Iterable[str],
) -> bool:
    """Classify an indented block as code/diagram rather than prose.

    RST literal blocks are normally introduced by ``::``.  The additional
    shape checks cover legacy docstrings containing unlabelled Python or ASCII
    diagrams.  Everything else is explanatory prose and must participate in
    translation/source hashes even though indentation is used for visual
    grouping or as a continuation after a blank field line.
    """
    values = [str(line) for line in lines if str(line).strip()]
    if not values:
        return True
    if str(previous_line).rstrip().endswith("::"):
        return True
    if any(_is_doctest_line(line) for line in values):
        return True
    stripped = [line.strip() for line in values]
    dedented = inspect.cleandoc("\n".join(values))
    # Parse only blocks that first look executable.  ``ast.parse('name')`` is
    # valid Python too, so parsing without this positive signal would hide
    # short explanatory prose.  Control statements require their Python colon:
    # prose tables such as ``if True  → use affine`` must stay translatable.
    python_signal = bool(re.search(
        r"(?m)^\s*(?:#|@|(?:async\s+)?def\b|class\b|import\b|"
        r"from\s+\S+\s+import\b|return\b|yield\b|raise\b|try\s*:|"
        r"except\b[^\n]*:|finally\s*:|else\s*:|"
        r"(?:if|elif|for|while|with)\b[^\n]*:\s*(?:#.*)?$|"
        r"[A-Za-z_]\w*\s*=|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\s*\(|"
        r"[\[{])",
        dedented,
    ))
    if python_signal:
        try:
            ast.parse(dedented)
        except SyntaxError:
            pass
        else:
            return True
    # A few API docstrings contain literal formats that are deliberately not
    # Python.  Keep these narrow and shape-based so ordinary indented prose is
    # still visible to the translation/source-hash audit.
    if (
        stripped[0].startswith("Permission is hereby granted")
        and any("THE SOFTWARE IS PROVIDED" in line for line in stripped)
    ):
        return True
    if any(re.match(r"^[A-Za-z_]\w*\s*[\u2248≃∝]\s*", line) for line in stripped):
        return True
    if (
        len(stripped) >= 2
        and sum(bool(re.match(r"^[→←↔]", line)) for line in stripped)
        >= len(stripped) - 1
    ):
        return True
    if all(re.match(
        r"^(?:python\s+-m\s+(?:spacr(?:\.\w+)*|pip\s+install\b)|"
        r"spacr(?:-qt)?)\b", line,
    ) for line in stripped):
        return True
    if all(re.match(
        r"^\[[^\]]+\]\s+(?:[A-Za-z_]\w*=|->\s+)", line,
    ) for line in stripped):
        return True
    if all(re.match(r"^\d+(?:\.\d+)?\s+m?s\s{2,}\S+", line) for line in stripped):
        return True
    if all(re.match(
        r"^\S+\.(?:json|csv|tsv|txt|npy|npz|tiff?)\s{2,}\S+", line,
    ) for line in stripped):
        return True
    diagram = sum(bool(re.search(
        r"(?:-->|==>|<--|<==|\+[-=]{2,}\+|\|[-=]{2,}\||[┌┐└┘├┤┬┴┼─│])",
        line,
    )) for line in stripped)
    threshold = max(1 if len(stripped) == 1 else 2, (len(stripped) + 1) // 2)
    return diagram >= threshold


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _clean_doc(node: ast.AST) -> str:
    value = ast.get_docstring(node, clean=False) or ""
    return inspect.cleandoc(value).strip()


def _is_visible_function_name(name: str, *, module_is_package: bool) -> bool:
    """Return whether AutoAPI emits this documented function/member name."""
    if not name.startswith("_"):
        return True
    # ``special-members`` exposes documented protocol methods, but not private
    # helpers or ``__init__``. Package ``__init__.py`` forwarding hooks are not
    # emitted as module members by the generated pages, unlike PEP-562 hooks in
    # ordinary modules such as ``spacr.version`` and ``spacr.qt.theme``.
    return (
        not module_is_package
        and name != "__init__"
        and name.startswith("__")
        and name.endswith("__")
    )


def _assignment_names(node: ast.AST) -> tuple[str, ...]:
    """Return simple names documented by a module/class assignment."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return ()
    return tuple(
        target.id for target in targets
        if isinstance(target, ast.Name) and not target.id.startswith("_")
    )


def _assigned_string(node: ast.AST) -> str:
    """Return a statically-known string assignment without importing code."""
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return ""
    try:
        value = ast.literal_eval(node.value)
    except (TypeError, ValueError):
        return ""
    if not isinstance(value, str):
        return ""
    return inspect.cleandoc(value).strip()


def _additional_assignment_docs(
    body: list[ast.stmt], owner: str,
) -> dict[str, str]:
    """Extract PEP-258 assignment docs and reviewed AutoAPI value prose."""
    docs: dict[str, str] = {}
    for index, node in enumerate(body):
        names = _assignment_names(node)
        if not names:
            continue
        following = body[index + 1] if index + 1 < len(body) else None
        additional = ""
        if (
            isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        ):
            additional = inspect.cleandoc(following.value.value).strip()
        for name in names:
            key = f"{owner}.{name}"
            text = additional
            if not text and key in API_VALUE_DOC_ASSIGNMENTS:
                text = _assigned_string(node)
            if text:
                docs[key] = text
    return docs


def public_docstrings() -> dict[str, str]:
    """Extract every canonical or exact-alias body visible in AutoAPI."""
    docs: dict[str, str] = {}
    for path in sorted((ROOT / "spacr").rglob("*.py")):
        if any(
            part in {"tests", "__pycache__", "backup_icons"}
            for part in path.parts
        ):
            continue
        # These are generated translation payloads, not Python API.  Including
        # their module headers makes every locale regeneration stale every API
        # locale and needlessly exposes generator metadata in the API picker.
        if "i18n_catalogs" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        module = _module_name(path)
        module_doc = _clean_doc(tree)
        if module_doc:
            docs[module] = module_doc
        docs.update(_additional_assignment_docs(tree.body, module))
        for node in tree.body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visible = _is_visible_function_name(
                    node.name, module_is_package=path.name == "__init__.py",
                )
            else:
                visible = not node.name.startswith("_")
            if not visible:
                continue
            key = f"{module}.{node.name}"
            doc = _clean_doc(node)
            if doc:
                docs[key] = doc
            if isinstance(node, ast.ClassDef):
                docs.update(_additional_assignment_docs(node.body, key))
                for child in node.body:
                    if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if not _is_visible_function_name(
                        child.name, module_is_package=False,
                    ):
                        continue
                    child_doc = _clean_doc(child)
                    if child_doc:
                        docs[f"{key}.{child.name}"] = child_doc
    for alias, canonical in API_DOC_ALIASES.items():
        if alias in docs:
            raise ValueError(
                f"API doc alias now has its own canonical source: {alias}"
            )
        if canonical in API_DOC_ALIASES:
            raise ValueError(
                f"API doc alias chains are not allowed: {alias} -> {canonical}"
            )
        if canonical not in docs:
            raise ValueError(
                f"API doc alias target is missing: {alias} -> {canonical}"
            )
        docs[alias] = docs[canonical]
    return dict(sorted(docs.items()))


def _split_long(text: str, limit: int = 1000) -> list[str]:
    """Split prose below the OPUS models' 480-token generation ceiling.

    A 1,000-character ceiling leaves room for German/Portuguese expansion and
    for protected RST markers.  Silent tokenizer truncation is unacceptable
    here because it can produce a fluent-looking translation with the end of
    a docstring missing.
    """
    if len(text) <= limit:
        return [text]
    sentences = _SENTENCE_RE.split(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    bounded: list[str] = []
    for chunk in chunks:
        while len(chunk) > limit:
            split_at = chunk.rfind(" ", 0, limit)
            if split_at < limit // 2:
                split_at = limit
            bounded.append(chunk[:split_at].strip())
            chunk = chunk[split_at:].strip()
        if chunk:
            bounded.append(chunk)
    return bounded


def translatable_blocks(
    text: str, *, preserve_directive_options: bool = False,
) -> tuple[list[str], list[tuple[str, object]]]:
    """Split reStructuredText into prose blocks and lossless layout tokens.

    A trailing ``::`` introduces a literal block.  It is RST chrome rather
    than prose: exposing it to a translation model lets ``::`` collapse to
    ``:``, after which the following indented code is parsed as visible prose.
    The final ``block_suffixes`` token therefore owns and reconstructs every
    such marker while model-facing blocks contain only their prose.
    """
    lines = text.splitlines()
    blocks: list[str] = []
    layout: list[tuple[str, object]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            layout.append(("raw", ""))
            index += 1
            continue
        # A standalone literal-block introducer is pure RST chrome.  Treating
        # it as a paragraph and then detaching its suffix creates an empty
        # model-facing block, which can later be reported as a degenerate
        # translation even though there is no prose to translate.
        if line.strip() == "::":
            layout.append(("raw", line))
            index += 1
            continue
        # The README language picker is navigation markup, not prose.  Passing
        # all ten RST links through a translation model can reorder or damage
        # their delimiters even when the visible language names are unchanged.
        # Keep it byte-for-byte here and localize only its leading label after
        # the translated document has been rebuilt.
        if line.startswith("Languages:") and "README" in line:
            literal = [line]
            index += 1
            while index < len(lines) and lines[index].strip():
                literal.append(lines[index])
                index += 1
            layout.append(("raw_lines", literal))
            continue
        simple_match = _SIMPLE_TABLE_BORDER_RE.match(line)
        if simple_match:
            indent = simple_match.group(1)
            border_parts = re.split(
                r"\s{2,}", line[len(indent):].strip()
            )
            column_count = len(border_parts)
            source_widths = [len(part) for part in border_parts]
            column_starts: list[int] = []
            cursor = 0
            for width in source_widths:
                column_starts.append(cursor)
                cursor += width + 2
            table_lines: list[str] = []
            end = index
            while end < len(lines):
                candidate = lines[end]
                border = _SIMPLE_TABLE_BORDER_RE.match(candidate)
                if not border and not candidate.strip():
                    break
                table_lines.append(candidate)
                end += 1
                if (
                    border
                    and len(table_lines) > 1
                    and (end == len(lines) or not lines[end].strip())
                ):
                    break
            if (
                len(table_lines) >= 3
                and _SIMPLE_TABLE_BORDER_RE.match(table_lines[-1])
            ):
                entries: list[tuple[str, object]] = []
                row_number = 0
                for table_line in table_lines:
                    stripped = table_line[len(indent):].strip()
                    if _SIMPLE_TABLE_BORDER_RE.match(table_line):
                        entries.append(("border", stripped[0]))
                        continue
                    raw_row = table_line[len(indent):]
                    separated = re.split(r"\s{2,}", raw_row.strip())
                    if len(separated) == column_count:
                        values = [value.strip() for value in separated]
                    else:
                        values = [
                            raw_row[
                                start:start + source_widths[column]
                            ].strip()
                            for column, start in enumerate(column_starts)
                        ]
                    nonempty = [
                        column for column, value in enumerate(values) if value
                    ]
                    if (
                        row_number > 0
                        and nonempty
                        and 0 not in nonempty
                        and entries
                        and entries[-1][0] == "row"
                    ):
                        # A blank first cell marks a wrapped continuation of
                        # the preceding simple-table row. Merge every populated
                        # column; wide comparison tables often wrap several
                        # cells on the same physical line.
                        previous_cells = entries[-1][1]
                        for column in nonempty:
                            position, raw = previous_cells[column]
                            if position is None:
                                previous_cells[column] = (
                                    None, f"{raw} {values[column]}".strip()
                                )
                            else:
                                blocks[position] = (
                                    f"{blocks[position]} {values[column]}".strip()
                                )
                        continue
                    cells: list[tuple[int | None, str]] = []
                    for column, value in enumerate(values):
                        value = value.strip()
                        code_key = (
                            (
                                row_number > 0
                                and bool(
                                    (
                                        column == 0
                                        and re.search(
                                            r"[`/<>*]|\bspacr\b", value
                                        )
                                    )
                                    or re.fullmatch(
                                        r"(?:[a-z]+[A-Z][A-Za-z0-9]*|"
                                        r"[A-Z][A-Za-z0-9]*[a-z][A-Z]"
                                        r"[A-Za-z0-9]*)",
                                        value,
                                    )
                                    or (
                                        column == 0
                                        and re.fullmatch(
                                            r"[a-z][a-z0-9_-]*", value
                                        )
                                    )
                                )
                            )
                            or (
                                row_number == 0
                                and f"``{value}``" in text
                            )
                            or (
                                row_number == 0
                                and column > 0
                                and column_count > 2
                                and bool(re.fullmatch(
                                    r"[a-z][a-z0-9_-]*", value
                                ))
                            )
                        )
                        if not value or code_key:
                            cells.append((None, value))
                        else:
                            cells.append((len(blocks), ""))
                            blocks.append(value)
                    entries.append(("row", cells))
                    row_number += 1
                layout.append((
                    "simple_table",
                    {"indent": indent, "columns": column_count,
                     "entries": entries},
                ))
                index = end
                continue
        grid_match = _GRID_BORDER_RE.match(line)
        if grid_match and index + 1 < len(lines):
            indent = grid_match.group(1)
            first_parts = line[len(indent):].strip()[1:-1].split("+")
            column_count = len(first_parts)
            table_lines: list[str] = []
            end = index
            while end < len(lines):
                candidate = lines[end]
                stripped = candidate[len(indent):] if candidate.startswith(
                    indent
                ) else candidate
                border = _GRID_BORDER_RE.match(candidate)
                row_ok = (
                    stripped.startswith("|")
                    and stripped.rstrip().endswith("|")
                    and len(stripped.rstrip()[1:-1].split("|"))
                    == column_count
                )
                border_ok = bool(
                    border
                    and len(
                        candidate[len(indent):].strip()[1:-1].split("+")
                    ) == column_count
                )
                if not (row_ok or border_ok):
                    break
                table_lines.append(candidate)
                end += 1
            if (
                len(table_lines) >= 3
                and any(
                    item[len(indent):].lstrip().startswith("|")
                    for item in table_lines
                )
                and _GRID_BORDER_RE.match(table_lines[-1])
            ):
                entries: list[tuple[str, object]] = []
                row_number = 0
                for table_line in table_lines:
                    stripped = table_line[len(indent):].rstrip()
                    if stripped.startswith("+"):
                        entries.append((
                            "border",
                            "=" if "=" in stripped else "-",
                        ))
                        continue
                    cells: list[tuple[int | None, str]] = []
                    for column, cell in enumerate(
                        stripped[1:-1].split("|")
                    ):
                        value = cell.strip()
                        # The first column of an API grid commonly contains
                        # command/module identifiers. Keep lowercase code-like
                        # keys exact while translating its human header and
                        # prose catch-all rows such as "other modules".
                        code_key = (
                            column == 0
                            and row_number > 0
                            and bool(re.fullmatch(r"[a-z][a-z0-9_]*", value))
                        )
                        if not value or code_key:
                            cells.append((None, value))
                        else:
                            cells.append((len(blocks), ""))
                            blocks.append(value)
                    entries.append(("row", cells))
                    row_number += 1
                layout.append((
                    "grid_table",
                    {"indent": indent, "columns": column_count,
                     "entries": entries},
                ))
                index = end
                continue
        field_match = _FIELD_RE.match(line)
        if field_match:
            prefix, first = field_match.groups()
            # Type declarations are API syntax, not prose. Keep both the field
            # marker and its value byte-for-byte instead of asking a language
            # model to translate bare Python/type identifiers.
            if re.match(r":(?:type|rtype|vartype|cvar)\b", prefix):
                literal = [line]
                index += 1
                while (index < len(lines) and lines[index].strip()
                       and lines[index].startswith((" ", "\t"))):
                    literal.append(lines[index])
                    index += 1
                layout.append(("raw_lines", literal))
                continue
            field = [first] if first else []
            index += 1
            while (index < len(lines) and lines[index].strip()
                   and lines[index].startswith((" ", "\t"))):
                field.append(lines[index].strip())
                index += 1
            if not field:
                layout.append(("raw", line))
                continue
            prose = " ".join(field)
            positions = [len(blocks)]
            blocks.append(prose)
            layout.append(("translated_prefixed", (prefix, positions)))
            continue
        # Directive options are RST configuration, not visible prose. This
        # branch also runs recursively inside translated admonitions, where a
        # dedented ``:class: warning`` line previously went to the model.
        if preserve_directive_options and _DIRECTIVE_OPTION_RE.match(line):
            literal = [line]
            index += 1
            while (
                index < len(lines)
                and lines[index].strip()
                and lines[index].startswith((" ", "\t"))
            ):
                literal.append(lines[index])
                index += 1
            layout.append(("raw_lines", literal))
            continue
        # Definition-list descriptions are visible prose only at the current
        # structural level.  An indented match can belong to a literal block
        # introduced by ``::`` and must first pass the block classifier below.
        code_definition = (
            None if line.startswith((" ", "\t"))
            else _CODE_DEFINITION_RE.match(line)
        )
        if code_definition:
            prefix, prose = code_definition.groups()
            position = len(blocks)
            blocks.append(prose.strip())
            layout.append(("translated_prefixed", (prefix, [position])))
            index += 1
            continue
        aligned_definition = (
            None if line.startswith((" ", "\t"))
            else _ALIGNED_LITERAL_DEFINITION_RE.match(line)
        )
        if aligned_definition:
            position = len(blocks)
            blocks.append(aligned_definition.group("prose").strip())
            layout.append((
                "translated_prefixed",
                (aligned_definition.group("prefix"), [position]),
            ))
            index += 1
            continue
        bullet_match = re.match(r"^(\s*(?:[*-]|#\.)\s+)(.*)$", line)
        if bullet_match:
            prefix, first = bullet_match.groups()
            base_indent = len(prefix) - len(prefix.lstrip())
            item = [first]
            index += 1
            while index < len(lines) and lines[index].strip():
                following = lines[index]
                following_bullet = re.match(
                    r"^(\s*)(?:[*-]|#\.)\s+", following
                )
                if following_bullet and len(following_bullet.group(1)) <= base_indent:
                    break
                following_indent = len(following) - len(following.lstrip())
                if following_indent <= base_indent:
                    break
                item.append(lines[index].strip())
                index += 1
            prose = " ".join(item)
            positions = [len(blocks)]
            blocks.append(prose)
            layout.append(("translated_prefixed", (prefix, positions)))
            continue
        # Doctest prompts, their continuation lines and expected output are
        # executable documentation.  Translating any token in that block can
        # turn valid Python into convincing-looking broken code, so retain the
        # complete example through the next blank line byte-for-byte.
        if _is_doctest_line(line):
            literal = [line]
            index += 1
            while index < len(lines) and lines[index].strip():
                literal.append(lines[index])
                index += 1
            layout.append(("raw_lines", literal))
            continue
        if line.startswith((" ", "\t")):
            literal = [line]
            block_start = index
            index += 1
            while index < len(lines) and (
                not lines[index].strip() or lines[index].startswith((" ", "\t"))
            ):
                literal.append(lines[index])
                index += 1
            trailing_blanks = 0
            while (
                trailing_blanks < len(literal)
                and not literal[len(literal) - trailing_blanks - 1].strip()
            ):
                trailing_blanks += 1
            core_end = len(literal) - trailing_blanks if trailing_blanks else len(literal)
            core = literal[:core_end]
            previous = ""
            previous_index = block_start - 1
            while previous_index >= 0:
                if lines[previous_index].strip():
                    previous = lines[previous_index]
                    break
                previous_index -= 1
            if _indented_block_is_literal(previous, core):
                layout.append(("raw_lines", literal))
                continue
            nonblank = [item for item in core if item.strip()]
            indentation = min(
                len(item) - len(item.lstrip()) for item in nonblank
            )
            indent = nonblank[0][:indentation]
            dedented = [
                item[indentation:] if item.strip() else "" for item in core
            ]
            nested_blocks, nested_layout = translatable_blocks(
                "\n".join(dedented)
            )
            if not nested_blocks:
                layout.append(("raw_lines", literal))
                continue
            start = len(blocks)
            blocks.extend(nested_blocks)
            layout.append((
                "nested_indented",
                {
                    "indent": indent,
                    "trailing_blanks": trailing_blanks,
                    "layout": nested_layout,
                    "start": start,
                    "count": len(nested_blocks),
                },
            ))
            continue
        if _DIRECTIVE_RE.match(line):
            translatable_directive = _TRANSLATABLE_DIRECTIVE_RE.match(line)
            literal = [line]
            index += 1
            while index < len(lines) and (
                not lines[index].strip() or lines[index].startswith((" ", "\t"))
            ):
                literal.append(lines[index])
                index += 1
            if translatable_directive:
                title = translatable_directive.group("title").strip()
                title_position: int | None = None
                header_prefix = line
                if title:
                    title_position = len(blocks)
                    blocks.append(title)
                    header_prefix = (
                        translatable_directive.group("head")
                        + translatable_directive.group("spacing")
                    )
                body = literal[1:]
                leading_blanks = 0
                while leading_blanks < len(body) and not body[leading_blanks].strip():
                    leading_blanks += 1
                trailing_blanks = 0
                while (
                    trailing_blanks < len(body) - leading_blanks
                    and not body[len(body) - trailing_blanks - 1].strip()
                ):
                    trailing_blanks += 1
                core_end = len(body) - trailing_blanks if trailing_blanks else len(body)
                core = body[leading_blanks:core_end]
                nonblank = [item for item in core if item.strip()]
                if nonblank:
                    indentation = min(
                        len(item) - len(item.lstrip()) for item in nonblank
                    )
                    indent = nonblank[0][:indentation]
                    dedented = [
                        item[indentation:] if item.strip() else ""
                        for item in core
                    ]
                    nested_blocks, nested_layout = translatable_blocks(
                        "\n".join(dedented), preserve_directive_options=True,
                    )
                    start = len(blocks)
                    blocks.extend(nested_blocks)
                    layout.append((
                        "nested_directive",
                        {
                            "header": header_prefix,
                            "title_position": title_position,
                            "indent": indent,
                            "leading_blanks": leading_blanks,
                            "trailing_blanks": trailing_blanks,
                            "layout": nested_layout,
                            "start": start,
                            "count": len(nested_blocks),
                        },
                    ))
                elif title_position is not None:
                    layout.append((
                        "nested_directive",
                        {
                            "header": header_prefix,
                            "title_position": title_position,
                            "indent": "",
                            "leading_blanks": leading_blanks,
                            "trailing_blanks": trailing_blanks,
                            "layout": [],
                            "start": len(blocks),
                            "count": 0,
                        },
                    ))
                else:
                    layout.append(("raw_lines", literal))
            else:
                layout.append(("raw_lines", literal))
            continue
        if _UNDERLINE_RE.match(line):
            layout.append(("raw", line))
            index += 1
            continue
        paragraph = [line.strip()]
        index += 1
        while index < len(lines):
            following = lines[index]
            if not following.strip() or _DIRECTIVE_RE.match(following):
                break
            if _is_doctest_line(following):
                break
            if _UNDERLINE_RE.match(following):
                break
            if following.lstrip().startswith(("* ", "- ", "#. ")):
                break
            if _FIELD_RE.match(following):
                break
            if following.startswith((" ", "\t")):
                # Break only when the complete indented run is literal.  Many
                # docstrings use indentation merely to align explanatory prose;
                # splitting every such continuation loses its sentence context
                # and needlessly changes hundreds of stable block identities.
                indented_run: list[str] = []
                lookahead = index
                while (
                    lookahead < len(lines)
                    and lines[lookahead].strip()
                    and lines[lookahead].startswith((" ", "\t"))
                ):
                    indented_run.append(lines[lookahead])
                    lookahead += 1
                if _indented_block_is_literal(lines[index - 1], indented_run):
                    break
            paragraph.append(following.strip())
            index += 1
        prose = " ".join(paragraph)
        positions = [len(blocks)]
        blocks.append(prose)
        layout.append(("translated", positions))
    literal_intro_suffixes: dict[int, str] = {}
    for position, block in enumerate(blocks):
        stripped = block.rstrip()
        if stripped.endswith("::"):
            blocks[position] = stripped[:-2].rstrip()
            literal_intro_suffixes[position] = "::"
    if literal_intro_suffixes:
        layout.append(("block_suffixes", literal_intro_suffixes))
    return blocks, layout


def rebuild_document(layout: Iterable[tuple[str, object]], translated: list[str]) -> str:
    layout = list(layout)
    block_suffixes: dict[int, str] = {}
    for kind, payload in layout:
        if kind == "block_suffixes":
            block_suffixes.update({
                int(position): str(suffix)
                for position, suffix in payload.items()
            })

    def rendered(position: int) -> str:
        value = translated[position]
        suffix = block_suffixes.get(position, "")
        # Marian commonly translates a heading-like ``Usage`` block as
        # ``Uso:``. The canonical layout already owns the literal introducer
        # ``::``; appending it without removing that model-added prose colon
        # creates invalid ``Uso:::`` and makes the code block ordinary text.
        if suffix == "::":
            value = value.rstrip().rstrip(":").rstrip()
        return value + suffix

    def paragraph_rendered(position: int) -> str:
        value = rendered(position)
        # A model may insert two spaces after a leading code/role literal.
        # At the top RST level that spelling is a definition list, so a later
        # parse would detach the literal even though the candidate passed the
        # literal counter. Canonical definition lists already use the separate
        # ``translated_prefixed`` layout; ordinary paragraphs normalize this
        # ambiguous boundary to one space.
        return re.sub(
            rf"^((?:``[^`]+``|{_RST_ROLE_PATTERN}))\s{{2,}}(?=\S)",
            r"\1 ",
            value,
        )

    lines: list[str] = []
    for kind, payload in layout:
        if kind == "block_suffixes":
            continue
        if kind == "raw":
            lines.append(str(payload))
        elif kind == "raw_lines":
            lines.extend(str(line) for line in payload)
        elif kind == "translated_prefixed":
            prefix, positions = payload
            separator = (
                "" if not positions or str(prefix).endswith((" ", "\t"))
                else " "
            )
            lines.append(
                str(prefix) + separator
                + " ".join(rendered(index) for index in positions)
            )
        elif kind == "nested_directive":
            title_position = payload.get("title_position")
            header = str(payload["header"])
            if title_position is not None:
                header += rendered(int(title_position))
            lines.append(header)
            lines.extend("" for _ in range(int(payload["leading_blanks"])))
            start = int(payload["start"])
            count = int(payload["count"])
            nested = rebuild_document(
                payload["layout"], translated[start:start + count]
            )
            indent = str(payload["indent"])
            lines.extend(
                indent + nested_line if nested_line else ""
                for nested_line in nested.splitlines()
            )
            lines.extend("" for _ in range(int(payload["trailing_blanks"])))
        elif kind == "nested_indented":
            start = int(payload["start"])
            count = int(payload["count"])
            nested = rebuild_document(
                payload["layout"], translated[start:start + count]
            )
            indent = str(payload["indent"])
            lines.extend(
                indent + nested_line if nested_line else ""
                for nested_line in nested.splitlines()
            )
            lines.extend("" for _ in range(int(payload["trailing_blanks"])))
        elif kind in {"grid_table", "simple_table"}:
            indent = str(payload["indent"])
            column_count = int(payload["columns"])
            rendered_entries: list[tuple[str, object]] = []
            widths = [1] * column_count
            for entry_kind, entry_payload in payload["entries"]:
                if entry_kind == "border":
                    rendered_entries.append((entry_kind, entry_payload))
                    continue
                rendered_cells: list[str] = []
                for column, (position, raw) in enumerate(entry_payload):
                    value = raw if position is None else rendered(position)
                    value = str(value).replace("|", r"\|").strip()
                    rendered_cells.append(value)
                    widths[column] = max(widths[column], len(value))
                rendered_entries.append(("row", rendered_cells))
            for entry_kind, entry_payload in rendered_entries:
                if entry_kind == "border":
                    character = str(entry_payload)
                    if kind == "grid_table":
                        lines.append(
                            indent + "+" + "+".join(
                                character * (width + 2) for width in widths
                            ) + "+"
                        )
                    else:
                        lines.append(
                            indent + "  ".join(
                                character * width for width in widths
                            )
                        )
                else:
                    cells = entry_payload
                    if kind == "grid_table":
                        lines.append(
                            indent + "| " + " | ".join(
                                cell.ljust(widths[column])
                                for column, cell in enumerate(cells)
                            ) + " |"
                        )
                    else:
                        lines.append(
                            indent + "  ".join(
                                cell.ljust(widths[column])
                                for column, cell in enumerate(cells)
                            ).rstrip()
                        )
        else:
            lines.append(
                " ".join(paragraph_rendered(index) for index in payload)
            )
    for index in range(1, len(lines)):
        underline = lines[index].strip()
        if (
            _UNDERLINE_RE.match(underline)
            and not _GRID_BORDER_RE.match(lines[index])
            and lines[index - 1].strip()
        ):
            character = underline[0]
            lines[index] = character * max(
                len(underline), len(lines[index - 1].strip())
            )
    return "\n".join(lines).strip()


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_block_hashes(text: str) -> list[str]:
    blocks, _layout = translatable_blocks(text)
    return [_source_hash(block) for block in blocks]


def _translation_source_block_hashes(text: str) -> list[str]:
    """Hash the exact English model input selected for every API block.

    Most blocks use their canonical source verbatim.  Ambiguous terse blocks
    use a target-neutral English expansion from ``API_TRANSLATION_CONTEXT``.
    Recording both contracts prevents a revised expansion from silently
    reusing a translation generated from older wording.
    """
    blocks, _layout = translatable_blocks(text)
    return [
        _source_hash(_api_translation_source(block))
        for block in blocks
    ]


def _normalized_block(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _api_block_requires_translation(source: str) -> bool:
    """Identify prose after removing every protected code/API literal."""
    if _source_hash(source) in API_EXACT_BLOCK_SHA256_ALLOWLIST:
        return False
    residual = _PROTECT_RE.sub(" ", str(source))
    # A single lexical word can be a section heading (``Examples``) or a
    # complete short description (``Deprecated.``), both of which are visible
    # prose. Raw doctests, directives, indented code and type fields never
    # enter this function because ``translatable_blocks`` keeps them literal.
    return bool(
        re.search(r"[A-Za-z]{3,}", residual)
        # The preview contract exposes bare yes/no cells. ``no`` is a complete
        # visible answer even though it is only two letters; quoted/code forms
        # have already been removed by the protection pass above.
        or re.search(r"(?<![A-Za-z])no(?![A-Za-z])", residual, re.IGNORECASE)
    )


_TARGET_SCRIPT_PATTERN = {
    "zh_CN": re.compile(r"[\u3400-\u9fff]"),
    "hi": re.compile(r"[\u0900-\u097f]"),
    "ko": re.compile(r"[\uac00-\ud7af]"),
}

# These ordinary English words are not scientific identifiers or accepted UI
# loanwords.  If one survives verbatim from the English model input into a
# Latin-script target block, the decode is partial even when the rest of the
# paragraph is translated.  The check is source-conditioned, so a coincidental
# target-language word cannot fail unless the English input used that word too.
_ENGLISH_RESIDUE_WORDS = frozenset({
    "absent", "adds", "appends", "available", "before", "both", "bound",
    "calls", "default", "defaults", "density", "does", "doubles", "either",
    "ever", "every", "failed", "filename", "idempotent", "initialized",
    "keep", "leaves", "left", "loads", "match", "matches", "neither", "not",
    "only", "otherwise", "prior", "puts", "rather", "returns", "rule", "see",
    "selected", "skips", "sorted", "string", "these", "those", "thread",
    "threading", "through", "toggle", "unclean", "unset", "verifies", "where",
    "whether", "which", "while", "without", "works", "caller", "decides",
    "event", "kept", "lazy", "loop", "omitted", "outside", "same", "trim",
    "startup", "the",
})
_LATIN_TARGET_LANGUAGES = frozenset({"sv", "de", "es", "pt", "is", "fr"})
_ENGLISH_RESIDUE_BY_LANGUAGE = {
    # These function words cannot be ordinary Portuguese words.  OPUS may
    # translate most of a technical sentence while echoing one clause (for
    # example ``when ... is a folder`` or ``shared with one worker``); the
    # general vocabulary above caught only a subset of those partial decodes.
    # Keep this locale-specific because words such as ``in`` are legitimate
    # Swedish/German vocabulary even though they are English residue in PT.
    "pt": frozenset({
        "across", "after", "again", "all", "already", "also", "among",
        "and", "another", "any", "are", "around", "at", "before", "been",
        "being", "between", "by", "can", "cannot", "could", "did", "done",
        "during", "each", "else", "few", "first", "from", "had",
        "has", "have", "here", "how", "if", "in", "inside", "into", "it",
        "its", "keep", "kept", "last", "least", "left", "less", "many",
        "may", "might", "more", "most", "much", "must", "new", "now",
        "of", "off", "old", "on", "one", "onto", "other", "out",
        "outside", "over", "return", "returned", "right", "run", "running",
        "runs", "should", "so", "still", "such", "than", "that", "their",
        "them", "then", "there", "these", "they", "this", "those",
        "through", "to", "two", "under", "while", "who", "whose", "why",
        "will", "with", "within", "would", "write", "yes",
    }),
}
_ENGLISH_RESIDUE_ALLOWLIST = {
    # ``thread`` is the standard technical term in Portuguese software prose;
    # forcing a literal expansion produced broken agreement and less useful
    # API documentation.  This exception applies only to the residue detector,
    # never to protected Python identifiers or exact-English completeness.
    "pt": frozenset({"thread", "threads"}),
}

# A single shared token is not always evidence of an untranslated fragment.
# In Portuguese, for example, ``for`` is a form of *ser/ir* (``seja qual
# for``). Still reject unambiguously English uses of such shared words by
# looking for a complete source phrase in the target prose. These patterns
# deliberately require at least two lexical words and run only when the same
# phrase occurs in the English model input.
_ENGLISH_RESIDUE_PHRASES_BY_LANGUAGE = {
    "pt": (
        r"\bfor\s+(?:all|an|any|each|every|more|one|that|the|this|two)\b",
    ),
}

# Copied runs are a model failure even when the target also contains enough of
# its own script to satisfy the script gate.  This matters especially for
# mixed Hindi, Korean and Chinese decodes, so the check deliberately applies
# to every target language.  A small reviewed allowlist covers scientific
# names, file-format lists, matrix-axis notation and API code alphabets that
# must remain literal.  Runs of four words always fail; shorter runs fail only
# when they contain an English grammar/content word, which catches fragments
# such as ``what format`` and ``panel gets`` without treating a shared
# scientific name as prose.
API_SHARED_PHRASE_ALLOWLIST = frozenset({
    "score cam xgrad cam layer cam eigen cam",
    "sv de es pt hi ko is fr",
    "nd czi lif multi page tiff npz",
    "c matthew o meara maom orcid",
    "matthew o meara maom orcid",
    "spacrpower copyright c matthew o meara",
    "ported from copyright c matthew o meara maom orcid",
    "anthropic claude openai google gemini",
    "zlib gzip bz lzma",
    "claude openai google gemini",
    "pro chatgpt plus pro team",
    "benjamini hochberg fdr q values",
    "piironen vehtari electron j statist",
    "t test mann whitney anova kruskal",
    "csv tsv tab xls xlsx",
    "dml insert update delete replace",
    "r l u d",
    "h w m m m m m m",
    "t f l a z c",
    "h w c uint",
    "o n log n",
    "rmse mae durbin watson",
    "r hat timings",
    # Reviewed API/type notation and standard scientific names. These blocks
    # intentionally retain the source spelling; the surrounding prose still
    # has to pass the ordinary completeness gates.
    "id or",
    "list or",
    "list or str",
    "plate or",
    "gb or",
    "axis interpretation or",
    "label array or",
    "k means",
    "k means discovery",
})

_COPIED_ENGLISH_GRAMMAR_WORDS = frozenset({
    "an", "the", "is", "are", "was", "were", "been", "being", "this",
    "that", "these", "those", "what", "which", "who", "whose", "when",
    "where", "why", "how", "from", "with", "without", "into", "onto",
    "of", "to", "and", "or", "but", "if", "then", "than", "its",
    "their", "them", "they", "it", "not", "only", "each", "every",
    "either", "neither", "another", "any", "all", "one", "two", "more",
    "most", "much", "many", "few", "does", "did", "done", "gets",
    "needs", "means", "has", "have", "had", "can", "cannot", "could",
    "should", "would", "will", "must", "may", "might", "also", "still",
    "already", "again", "over", "under", "across", "around", "between",
    "before", "after", "during", "inside", "outside", "return",
    "returned", "run", "running", "runs", "write", "writes", "written",
    "keep", "kept", "left", "less", "same", "such", "through",
    "whether", "while", "available", "absent", "adds", "appends", "calls",
    "failed", "leaves", "loads", "match", "matches", "puts", "rather",
    "returns", "see", "selected", "skips", "sorted", "string", "works",
    "caller", "decides", "event", "lazy", "loop", "omitted", "startup",
})

# High-confidence content bigrams observed in otherwise Portuguese model
# output. They carry ordinary explanatory meaning, not API nomenclature. Keep
# this reviewed inventory narrow rather than rejecting every shared bigram.
_PT_COPIED_CONTENT_BIGRAMS = frozenset({
    "trailing newline",
    "scratch database",
    "staging file",
    "escape hatch",
    "dtype name",
    "config dict",
    "empty bin",
    "wishful thinking",
})


def _copied_english_phrases(
    source: str, value: str, language: str, *, minimum_words: int = 2,
) -> tuple[str, ...]:
    """Return non-literal English lexical runs copied into a target block."""
    if language not in MODEL_SPECS:
        return ()

    def lexical_words(text: str) -> list[str]:
        return re.findall(
            r"[^\W\d_]+",
            _PROTECT_RE.sub(" ", str(text)).casefold(),
        )

    source_words = lexical_words(source)
    value_words = lexical_words(value)
    copied: list[str] = []
    for match in SequenceMatcher(
        None, source_words, value_words, autojunk=False,
    ).get_matching_blocks():
        if match.size < minimum_words:
            continue
        phrase = " ".join(source_words[match.a:match.a + match.size])
        if phrase in API_SHARED_PHRASE_ALLOWLIST:
            continue
        if (
            match.size < 4
            and not _COPIED_ENGLISH_GRAMMAR_WORDS.intersection(
                source_words[match.a:match.a + match.size]
            )
        ):
            continue
        copied.append(phrase)
    if language == "pt":
        source_bigrams = {
            " ".join(source_words[index:index + 2])
            for index in range(len(source_words) - 1)
        }
        value_bigrams = {
            " ".join(value_words[index:index + 2])
            for index in range(len(value_words) - 1)
        }
        for phrase in sorted(
            _PT_COPIED_CONTENT_BIGRAMS & source_bigrams & value_bigrams
        ):
            if phrase not in copied:
                copied.append(phrase)
    return tuple(copied)


def _has_english_residue(source: str, value: str, language: str) -> bool:
    if language not in _LATIN_TARGET_LANGUAGES:
        return False
    source_prose = _PROTECT_RE.sub(" ", str(source)).casefold()
    value_prose = _PROTECT_RE.sub(" ", str(value)).casefold()
    # Match complete Unicode words.  ASCII-only tokenization split Portuguese
    # ``notícia`` at ``í`` and falsely reported the prefix ``not`` as retained
    # English prose (with analogous risks in every accented Latin language).
    source_words = set(re.findall(r"[^\W\d_]+", source_prose))
    value_words = set(re.findall(r"[^\W\d_]+", value_prose))
    residue_words = (
        _ENGLISH_RESIDUE_WORDS
        | _ENGLISH_RESIDUE_BY_LANGUAGE.get(language, frozenset())
    )
    residue = residue_words & source_words & value_words
    residue -= _ENGLISH_RESIDUE_ALLOWLIST.get(language, frozenset())
    if residue:
        return True
    return any(
        re.search(pattern, source_prose)
        and re.search(pattern, value_prose)
        for pattern in _ENGLISH_RESIDUE_PHRASES_BY_LANGUAGE.get(language, ())
    )


def _api_block_valid(source: str, value: str, language: str) -> bool:
    if not str(value).strip():
        return False
    if not _syntax_preserved(source, value):
        return False
    if _looks_degenerate(source, value, language):
        return False
    if _semantic_false_friends(source, value, language):
        return False
    if language == "zh_CN" and _has_traditional_chinese_prose(value):
        return False
    if not _api_block_requires_translation(source):
        return _normalized_block(source) == _normalized_block(value)
    normalized_value = _normalized_block(value)
    if _normalized_block(source) == normalized_value:
        return False
    if _has_english_residue(source, value, language):
        return False
    if _copied_english_phrases(source, value, language):
        return False
    # Long semantic paragraphs are decoded in bounded pieces. Re-parsing a
    # translation by character count is unstable because languages expand at
    # different rates, so detect an echoed source piece directly inside the
    # corresponding semantic target block instead.
    pieces = _split_long(source)
    for piece in pieces if len(pieces) > 1 else ():
        normalized_piece = _normalized_block(piece)
        if (
            _api_block_requires_translation(piece)
            and normalized_piece
            and normalized_piece in normalized_value
        ):
            return False
    pattern = _TARGET_SCRIPT_PATTERN.get(language)
    return pattern is None or bool(pattern.search(str(value)))


def _english_manifest(docs: Mapping[str, str]) -> dict[str, object]:
    symbols = {
        key: {
            "source_sha256": _source_hash(value),
            "source_blocks_sha256": _source_block_hashes(value),
            "text": value,
        }
        for key, value in docs.items()
        if key not in API_DOC_ALIASES
    }
    _materialize_alias_records(docs, symbols)
    return {
        "schema": 2,
        "language": "en",
        "symbols": symbols,
    }


def _materialize_alias_records(
    docs: Mapping[str, str], symbols: dict[str, dict[str, object]],
) -> None:
    """Clone canonical records under the exact ids AutoAPI visibly repeats."""
    for alias, canonical in API_DOC_ALIASES.items():
        if alias not in docs:
            continue
        if canonical not in docs or canonical not in symbols:
            raise ValueError(
                f"cannot materialize API alias without target: "
                f"{alias} -> {canonical}"
            )
        if docs[alias] != docs[canonical]:
            raise ValueError(
                f"API alias text differs from target: {alias} -> {canonical}"
            )
        symbols[alias] = {
            **symbols[canonical],
            "alias_of": canonical,
        }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        destination_mode = 0o664
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n"
            )
        temporary.chmod(destination_mode)
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _translate_blocks(
    blocks: Iterable[str],
    language: str,
    model_root: Path,
    args,
    *,
    force: bool = False,
    repair_protected: bool = False,
    cache_namespace: str = "",
    candidate_validator: Callable[[str, str, str], bool] | None = None,
) -> dict[str, str]:
    """Translate semantic RST blocks through bounded model-sized pieces."""
    ordered_blocks = list(dict.fromkeys(map(str, blocks)))
    pieces_by_block = {
        block: _split_model_safe(block) for block in ordered_blocks
    }
    pieces = sorted({
        piece
        for block_pieces in pieces_by_block.values()
        for piece in block_pieces
    })
    translated_pieces = _translate_batches(
        pieces,
        language,
        model_root,
        device=args.device,
        batch_size=args.batch_size,
        beams=args.beams,
        threads=args.threads,
        force_sources=pieces if force else (),
        repair_protected=repair_protected,
        cache_namespace=cache_namespace,
        candidate_validator=candidate_validator,
    )
    return {
        block: " ".join(
            translated_pieces.get(piece, piece).strip()
            for piece in block_pieces
        ).strip()
        for block, block_pieces in pieces_by_block.items()
    }


def _split_model_safe(text: str, limit: int = 760) -> list[str]:
    """Pre-split long model inputs without separating protected literals.

    The shared decoder performs a tokenizer-level no-truncation assertion.
    This conservative character bound keeps every current M2M/OPUS piece well
    below 480 tokens; punctuation-aware splitting retains complete sentence
    context and recursively handles unusually long sentences.
    """
    source = str(text).strip()
    if len(source) <= limit:
        return [source]
    chunks: list[str] = []
    current = ""
    for sentence in _translation_chunks(source):
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
        while len(current) > limit:
            split_at = current.rfind(" ", 0, limit)
            if split_at < limit // 2:
                split_at = limit
            chunks.append(current[:split_at].strip())
            current = current[split_at:].strip()
    if current:
        chunks.append(current)
    return chunks or [source]


def _translate_documents(
    documents: Mapping[str, str], language: str, model_root: Path, args,
) -> dict[str, str]:
    block_map: dict[str, tuple[list[str], list[tuple[str, object]]]] = {}
    unique: set[str] = set()
    for key, value in documents.items():
        blocks, layout = translatable_blocks(value)
        block_map[key] = (blocks, layout)
        unique.update(blocks)
    translations = _translate_blocks(
        sorted(unique), language, model_root, args,
    )
    reviewed_blocks = {**REVIEWED_README_BLOCKS, **REVIEWED_README_HEADINGS}
    for source, reviewed in reviewed_blocks.items():
        if source in translations:
            translations[source] = reviewed[language]
    result: dict[str, str] = {}
    for key, (blocks, layout) in block_map.items():
        result[key] = rebuild_document(
            layout, [translations[block] for block in blocks],
        )
    return result


def _translate_api_documents(
    documents: Mapping[str, str], language: str, model_root: Path, args,
) -> dict[str, str]:
    """Translate API documents through their current reviewed source context.

    ``translation_source_blocks_sha256`` is a freshness contract, not merely
    metadata.  Every model/cache lookup must use the exact contextual block
    whose hash is later written; otherwise an old translation of the raw
    canonical sentence could be relabelled as current after context changes.
    """
    block_map: dict[str, tuple[list[str], list[tuple[str, object]]]] = {}
    translation_inputs: dict[str, str] = {}
    for key, value in documents.items():
        blocks, layout = translatable_blocks(value)
        block_map[key] = (blocks, layout)
        for block in blocks:
            contextual = _api_translation_source(block)
            if not _syntax_preserved(block, contextual):
                raise ValueError(
                    "API translation context changed a protected literal: "
                    f"{block!r} -> {contextual!r}"
                )
            translation_inputs[block] = contextual
    translated_context = _translate_blocks(
        sorted(set(translation_inputs.values())),
        language,
        model_root,
        args,
        cache_namespace=API_BLOCK_CACHE_NAMESPACE,
        candidate_validator=lambda contextual, value, target: (
            _api_block_valid(contextual, value, target)
            and all(
                _api_block_valid(canonical, value, target)
                for canonical, current_context in translation_inputs.items()
                if current_context == contextual
            )
        ),
    )
    translations = {
        block: translated_context.get(contextual, block)
        for block, contextual in translation_inputs.items()
    }
    return {
        key: rebuild_document(
            layout, [translations[block] for block in blocks],
        )
        for key, (blocks, layout) in block_map.items()
    }


def write_language(
    docs: Mapping[str, str], language: str, translations: Mapping[str, str],
) -> None:
    if language == "zh_CN":
        non_normalized = [
            key for key, value in translations.items()
            if _has_traditional_chinese_prose(value)
        ]
        if non_normalized:
            raise ValueError(
                "zh_CN API output is not OpenCC t2s-normalized: "
                + ", ".join(non_normalized[:5])
            )
    model, _folder, license_name, _prefix = MODEL_SPECS[language]
    symbols = {
        key: {
            "source_sha256": _source_hash(source),
            "source_blocks_sha256": _source_block_hashes(source),
            "translation_source_blocks_sha256":
                _translation_source_block_hashes(source),
            "text": translations[key],
        }
        for key, source in docs.items()
        if key not in API_DOC_ALIASES
    }
    _materialize_alias_records(docs, symbols)
    payload = {
        "schema": 2,
        "language": language,
        "generator": model,
        "license": license_name,
        "secondary_generator": SECONDARY_MODEL,
        "secondary_license": SECONDARY_LICENSE,
        **({"normalizer": "OpenCC 1.1+ t2s"}
           if language == "zh_CN" else {}),
        "symbols": symbols,
    }
    path = API_DIR / f"{language}.json"
    _write_json(path, payload)


def reusable_api_translations(
    docs: Mapping[str, str], language: str,
) -> dict[str, str]:
    """Return entries whose canonical and contextual sources are unchanged."""
    path = API_DIR / f"{language}.json"
    try:
        symbols = json.loads(path.read_text(encoding="utf-8")).get(
            "symbols", {}
        )
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return {}
    reusable: dict[str, str] = {}
    for key, source in docs.items():
        record = symbols.get(key, {})
        text = str(record.get("text", "")).strip()
        if (
            record.get("source_sha256") == _source_hash(source)
            and record.get("source_blocks_sha256")
                == _source_block_hashes(source)
            and record.get("translation_source_blocks_sha256")
                == _translation_source_block_hashes(source)
            and text
        ):
            reusable[key] = text
    # A reviewed alias has exactly the canonical source and translation. This
    # lets an older catalog gain all alias records without decoding duplicate
    # prose, while ``write_language`` still materializes and marks each record.
    for alias, canonical in API_DOC_ALIASES.items():
        if alias in docs and canonical in reusable:
            reusable[alias] = reusable[canonical]
    return reusable


def reviewed_api_block_translations(
    docs: Mapping[str, str], language: str,
) -> dict[str, str]:
    """Load reviewed API blocks only after revalidating exact evidence.

    Review files are evidence, not catalogs. Every record remains bound to an
    exact symbol/block, canonical hash, and current reviewed model context;
    any drift is a hard error rather than a silently promoted translation.
    """
    directory = REVIEWED_API_DIR / language
    if not directory.is_dir():
        return {}
    reviewed: dict[str, str] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"invalid reviewed API evidence {path}") from exc
        if payload.get("schema") != 1 or payload.get("language") != language:
            raise ValueError(f"invalid reviewed API evidence header: {path}")
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"invalid reviewed API record list: {path}")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"invalid reviewed API record: {path}")
            label = str(record.get("label", ""))
            symbol, separator, raw_index = label.rpartition("#")
            if not separator:
                raise ValueError(f"invalid reviewed API label {label!r}: {path}")
            # Callers may repair an intentional documentation subset. Evidence
            # for symbols outside that subset is neither admitted nor stale.
            if symbol not in docs:
                continue
            try:
                index = int(raw_index)
            except ValueError as exc:
                raise ValueError(
                    f"invalid reviewed API block index {label!r}: {path}"
                ) from exc
            blocks, _layout = translatable_blocks(docs[symbol])
            if not 0 <= index < len(blocks):
                raise ValueError(f"stale reviewed API index {label!r}: {path}")
            source = str(record.get("source", ""))
            context = str(record.get("context", ""))
            target = str(record.get("translation", ""))
            if blocks[index] != source:
                raise ValueError(f"stale reviewed API source {label!r}: {path}")
            if record.get("source_sha256") != _source_hash(source):
                raise ValueError(f"stale reviewed API hash {label!r}: {path}")
            if _api_translation_source(source) != context:
                raise ValueError(f"stale reviewed API context {label!r}: {path}")
            if not (
                _syntax_preserved(source, target)
                and _api_block_valid(source, target, language)
                and _api_block_valid(context, target, language)
            ):
                raise ValueError(f"rejected reviewed API target {label!r}: {path}")
            previous = reviewed.setdefault(source, target)
            if previous != target:
                raise ValueError(
                    f"conflicting reviewed API targets for {source!r}"
                )
    return reviewed


def repair_api_translations(
    docs: Mapping[str, str], language: str, model_root: Path, args,
) -> dict[str, str]:
    """Repair stale/untranslated API blocks while retaining valid blocks.

    Reuse is positional only when the translated document has exactly the
    canonical block count. A layout mismatch causes that symbol's prose to be
    regenerated from its English blocks; executable/type/code-only blocks are
    always copied from the canonical source.
    """
    try:
        payload = json.loads(
            (API_DIR / f"{language}.json").read_text(encoding="utf-8")
        )
        current_symbols = payload.get("symbols", {})
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        current_symbols = {}

    plans: dict[
        str,
        tuple[list[str], list[tuple[str, object]], list[str | None], list[int]],
    ] = {}
    pending_sources: set[str] = set()
    reused_blocks = 0
    relaid_symbols = 0

    for key, source in docs.items():
        source_blocks, source_layout = translatable_blocks(source)
        record = current_symbols.get(
            API_DOC_ALIASES.get(key, key), {}
        )
        current_text = str(record.get("text", ""))
        current_ok = record.get("source_sha256") == _source_hash(source)
        current_blocks, _current_layout = translatable_blocks(current_text)
        positional = current_ok and len(current_blocks) == len(source_blocks)
        recorded_translation_hashes = record.get(
            "translation_source_blocks_sha256",
        )
        if not (
            isinstance(recorded_translation_hashes, list)
            and len(recorded_translation_hashes) == len(source_blocks)
        ):
            # Schema-2 catalogs written before contextual-source hashes used
            # the canonical source for every block.  This compatibility value
            # reuses ordinary translations while invalidating only blocks that
            # now have a reviewed English context expansion.
            recorded_translation_hashes = [
                _source_hash(block) for block in source_blocks
            ]
        if not positional:
            relaid_symbols += 1

        selected: list[str | None] = []
        pending_indexes: list[int] = []
        for index, source_block in enumerate(source_blocks):
            if not _api_block_requires_translation(source_block):
                selected.append(source_block)
                continue
            contextual_source = _api_translation_source(source_block)
            context_is_current = (
                recorded_translation_hashes[index]
                == _source_hash(contextual_source)
            )
            candidate = (
                current_blocks[index]
                if positional and context_is_current
                else ""
            )
            candidate = _contextualize(candidate, language, source_block)
            if (
                _api_block_valid(source_block, candidate, language)
                and _api_block_valid(
                    contextual_source, candidate, language,
                )
            ):
                selected.append(candidate)
                reused_blocks += 1
            else:
                selected.append(None)
                pending_indexes.append(index)
                pending_sources.add(source_block)
        plans[key] = (
            source_blocks, source_layout, selected, pending_indexes,
        )

    generated: dict[str, str] = {}
    translation_input: dict[str, str] = {}
    recovered_cache = 0
    recovered_review = 0
    if pending_sources:
        reviewed_blocks = reviewed_api_block_translations(docs, language)
        # A stricter audit can reject a catalog entry after its model output
        # was already checkpointed. If a later review narrows that audit (for
        # example by recognizing the Python type name ``dict`` as code), reuse
        # the namespaced checkpoint only after applying every current API
        # block validator. This avoids decoding hundreds of already-valid
        # paragraphs while never bypassing source freshness or syntax gates.
        try:
            api_cache = json.loads(
                (
                    model_root / ".spacr_translation_cache" /
                    f"{language}.json"
                ).read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError, AttributeError):
            api_cache = {}
        for source in pending_sources:
            contextual_source = _api_translation_source(source)
            reviewed_target = reviewed_blocks.get(source, "")
            if reviewed_target:
                generated[source] = reviewed_target
                recovered_review += 1
                continue
            candidate = ""
            # Before API contexts were namespaced, the shared decoder stored
            # accepted output under its model input itself. Reuse that exact
            # key only when the current contextual model input is unchanged;
            # a dual-valid value from a different prompt is still not honest
            # provenance for ``translation_source_blocks_sha256``.
            raw_candidates = [api_cache.get(
                f"{API_BLOCK_CACHE_NAMESPACE}\0{contextual_source}", "",
            )]
            if source == contextual_source:
                raw_candidates.append(api_cache.get(source, ""))
            for raw_candidate in raw_candidates:
                reviewed_candidate = _contextualize(
                    str(raw_candidate), language, source,
                )
                if (
                    _api_block_valid(
                        contextual_source, reviewed_candidate, language,
                    )
                    and _api_block_valid(source, reviewed_candidate, language)
                ):
                    candidate = reviewed_candidate
                    break
            if not candidate:
                translation_input[source] = contextual_source
                continue
            generated[source] = candidate
            recovered_cache += 1
        for source, contextual_source in translation_input.items():
            if not _syntax_preserved(source, contextual_source):
                raise ValueError(
                    "API translation context changed a protected literal: "
                    f"{source!r} -> {contextual_source!r}"
                )
        if translation_input:
            canonical_by_context: dict[str, list[str]] = {}
            for canonical, contextual in translation_input.items():
                canonical_by_context.setdefault(contextual, []).append(canonical)

            def api_repair_candidate_valid(
                contextual: str, value: str, target: str,
            ) -> bool:
                return bool(
                    _api_block_valid(contextual, value, target)
                    and all(
                        _api_block_valid(canonical, value, target)
                        for canonical in canonical_by_context.get(
                            contextual, (contextual,)
                        )
                    )
                )

            contextual_generated = _translate_blocks(
                sorted(set(translation_input.values())),
                language,
                model_root,
                args,
                force=True,
                repair_protected=True,
                # v7 keeps structural RST outside the model input and retains
                # emphasis in its grammatical context during hard-literal
                # fallback. Never promote an older ambiguous checkpoint.
                cache_namespace=API_BLOCK_CACHE_NAMESPACE,
                candidate_validator=api_repair_candidate_valid,
            )
            generated.update({
                source: contextual_generated.get(contextual_source, source)
                for source, contextual_source in translation_input.items()
            })

    repaired: dict[str, str] = {}
    unresolved = 0
    for key, (source_blocks, layout, selected, pending_indexes) in plans.items():
        for index in pending_indexes:
            source_block = source_blocks[index]
            contextual_source = _api_translation_source(source_block)
            candidate = _contextualize(
                generated.get(source_block, source_block),
                language,
                source_block,
            )
            # A failed contextual decode must not become an apparently valid
            # translation merely because its English fallback differs from
            # the original wording.  Validate both semantic contracts: the
            # target must translate the rewritten model input and preserve
            # every literal required by the canonical source.
            if (
                not _api_block_valid(
                    contextual_source, candidate, language,
                )
                or not _api_block_valid(source_block, candidate, language)
            ):
                candidate = source_block
                unresolved += 1
            selected[index] = candidate
        repaired[key] = rebuild_document(
            layout,
            [
                source_blocks[index] if value is None else str(value)
                for index, value in enumerate(selected)
            ],
        )

    print(
        f"{language}: API blocks reused={reused_blocks} "
        f"generated={len(pending_sources)} unresolved={unresolved} "
        f"review_recovered={recovered_review} "
        f"cache_recovered={recovered_cache} "
        f"decoded={len(translation_input)} relaid_symbols={relaid_symbols}",
        flush=True,
    )
    return repaired


def audit(docs: Mapping[str, str], languages: Iterable[str]) -> int:
    languages = tuple(languages)
    if "zh_CN" in languages:
        # Fail before inspecting catalog contents when the exact normalizer
        # used by generation is unavailable. A character heuristic is not an
        # equivalent zh_CN release check.
        _has_traditional_chinese_prose("")
    failures: list[str] = []
    expected = set(docs)
    field_pattern = re.compile(
        r"(?m)^(:(?:param|parameter|arg|argument|keyword|kwarg|type|"
        r"returns?|rtype|raises?|yields?|seealso|ivar|vartype|cvar|var)"
        r"\b[^:]*:)"
    )
    rst_link_pattern = re.compile(r"`[^`\n]+\s+<([^>\n]+)>`_")
    readme_protected_pattern = re.compile(
        r"``[^`]+``|" + _RST_ROLE_PATTERN + r"|"
        r"https?://[^\s)>}\]]+"
    )

    def protected_values(
        text: str, pattern: re.Pattern[str] = _PROTECT_RE,
    ) -> list[str]:
        values: list[str] = []
        for match in pattern.finditer(text):
            # Paragraph reflow may collapse a source line break inside an
            # inline literal or an RST role; its referenced value is still
            # unchanged and the one-line rendering is valid RST.
            values.append(re.sub(r"\s+", " ", match.group(0)))
        return sorted(values)

    def syntax_contract(
        source: str,
        translated: str,
        label: str,
        protected_pattern: re.Pattern[str] = _PROTECT_RE,
    ) -> None:
        source_protected = Counter(protected_values(source, protected_pattern))
        translated_protected = Counter(
            protected_values(translated, protected_pattern)
        )
        if protected_pattern is _PROTECT_RE:
            # Whole API documents also contain raw code/table blocks whose
            # multiplication stars can pair across paragraph boundaries.
            # Emphasis is validated on each parsed prose block above; retain
            # the document-level code/link/literal check without re-parsing
            # cross-block star pairs here.
            literals_ok = _syntax_preserved(
                source, translated, check_emphasis=False,
            )
        else:
            literals_ok = translated_protected == source_protected
        if not literals_ok:
            failures.append(f"{label}: code/link/RST roles changed")
        if sorted(field_pattern.findall(source)) != sorted(
            field_pattern.findall(translated)
        ):
            failures.append(f"{label}: RST fields changed")
        if sorted(rst_link_pattern.findall(source)) != sorted(
            rst_link_pattern.findall(translated)
        ):
            failures.append(f"{label}: RST link targets changed")
        source_doctest = [
            line for line in source.splitlines()
            if _is_doctest_line(line)
        ]
        if any(line not in translated.splitlines() for line in source_doctest):
            failures.append(f"{label}: doctest code changed")
        if _TOKEN_RE.search(translated) or re.search(
            r"Z\s*X\s*Q\s*\d", translated
        ):
            failures.append(f"{label}: leaked protection token")

    readme_source = README_SOURCE.read_text(encoding="utf-8")
    english_path = API_DIR / "en.json"
    try:
        english_payload = json.loads(english_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        failures.append("en: API source manifest is missing or invalid")
        english_symbols: Mapping[str, object] = {}
    else:
        english_symbols = english_payload.get("symbols", {})
        if english_payload.get("schema") != 2:
            failures.append("en: API manifest schema is not 2")
        if set(english_symbols) != expected:
            failures.append("en: API source manifest keys are stale")
        for key, source in docs.items():
            record = english_symbols.get(key, {})
            canonical = API_DOC_ALIASES.get(key)
            if record.get("alias_of") != canonical:
                failures.append(f"en/{key}: incorrect API alias metadata")
            if canonical:
                canonical_record = english_symbols.get(canonical, {})
                for field in (
                    "source_sha256", "source_blocks_sha256", "text",
                ):
                    if record.get(field) != canonical_record.get(field):
                        failures.append(
                            f"en/{key}: alias {field} differs from {canonical}"
                        )
            if record.get("source_sha256") != _source_hash(source):
                failures.append(f"en/{key}: stale source hash")
            if record.get("source_blocks_sha256") != _source_block_hashes(source):
                failures.append(f"en/{key}: stale source-block hashes")
            if record.get("text") != source:
                failures.append(f"en/{key}: canonical text differs")

    for language in languages:
        path = API_DIR / f"{language}.json"
        if not path.is_file():
            failures.append(f"{language}: API catalog is missing")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != 2:
            failures.append(f"{language}: API manifest schema is not 2")
        if payload.get("secondary_generator") != SECONDARY_MODEL:
            failures.append(
                f"{language}: API secondary generator provenance is stale"
            )
        if payload.get("secondary_license") != SECONDARY_LICENSE:
            failures.append(
                f"{language}: API secondary license provenance is stale"
            )
        if language == "zh_CN" and payload.get("normalizer") != (
            "OpenCC 1.1+ t2s"
        ):
            failures.append(
                "zh_CN: API manifest lacks OpenCC 1.1+ t2s provenance"
            )
        symbols = payload.get("symbols", {})
        missing = expected - set(symbols)
        stale = set(symbols) - expected
        if missing:
            failures.append(f"{language}: {len(missing)} API symbols missing")
        if stale:
            failures.append(f"{language}: {len(stale)} stale API symbols")
        unexpected_unchanged: list[str] = []
        missing_target_script: list[str] = []
        block_layout_errors: list[str] = []
        protected_block_errors: list[str] = []
        contextual_errors: list[str] = []
        contextual_translation_errors: list[str] = []
        english_residue_errors: list[str] = []
        copied_english_errors: list[str] = []
        semantic_false_friend_errors: list[str] = []
        for key, source in docs.items():
            record = symbols.get(key, {})
            canonical = API_DOC_ALIASES.get(key)
            if record.get("alias_of") != canonical:
                failures.append(
                    f"{language}/{key}: incorrect API alias metadata"
                )
            if canonical:
                canonical_record = symbols.get(canonical, {})
                for field in (
                    "source_sha256",
                    "source_blocks_sha256",
                    "translation_source_blocks_sha256",
                    "text",
                ):
                    if record.get(field) != canonical_record.get(field):
                        failures.append(
                            f"{language}/{key}: alias {field} differs from "
                            f"{canonical}"
                        )
            if record.get("source_sha256") != _source_hash(source):
                failures.append(f"{language}: stale source hash for {key}")
            source_blocks, _source_layout = translatable_blocks(source)
            expected_block_hashes = [
                _source_hash(block) for block in source_blocks
            ]
            if record.get("source_blocks_sha256") != expected_block_hashes:
                failures.append(
                    f"{language}: stale source-block hashes for {key}"
                )
            if (
                record.get("translation_source_blocks_sha256")
                != _translation_source_block_hashes(source)
            ):
                failures.append(
                    f"{language}: stale translation-source block hashes for {key}"
                )
            if not str(record.get("text", "")).strip():
                failures.append(f"{language}: blank translation for {key}")
            else:
                translated_text = str(record.get("text", ""))
                translated_blocks, _translated_layout = translatable_blocks(
                    translated_text
                )
                if len(translated_blocks) != len(source_blocks):
                    block_layout_errors.append(key)
                else:
                    for index, (source_block, translated_block) in enumerate(
                        zip(source_blocks, translated_blocks)
                    ):
                        label = f"{key}#{index}"
                        if not _syntax_preserved(source_block, translated_block):
                            protected_block_errors.append(label)
                        contextualized = _contextualize(
                            translated_block, language, source_block
                        )
                        if contextualized != translated_block:
                            contextual_errors.append(label)
                        semantic_failures = _semantic_false_friends(
                            source_block, translated_block, language,
                        )
                        # The contextual-error list already proves that the
                        # stored catalog still needs deterministic repair.
                        # Keep the semantic list focused on families for which
                        # the current repair table is insufficient, rather
                        # than double-counting every repairable old output.
                        if (
                            semantic_failures
                            and _semantic_false_friends(
                                source_block, contextualized, language,
                            )
                        ):
                            semantic_false_friend_errors.append(label)
                        if _api_block_requires_translation(source_block):
                            if (_normalized_block(source_block)
                                    == _normalized_block(translated_block)):
                                unexpected_unchanged.append(label)
                            contextual_source = _api_translation_source(
                                source_block
                            )
                            if (
                                contextual_source != source_block
                                and not _api_block_valid(
                                    contextual_source,
                                    translated_block,
                                    language,
                                )
                            ):
                                contextual_translation_errors.append(label)
                            if _has_english_residue(
                                source_block, translated_block, language,
                            ):
                                english_residue_errors.append(label)
                            if _copied_english_phrases(
                                source_block, translated_block, language,
                            ):
                                copied_english_errors.append(label)
                            pattern = _TARGET_SCRIPT_PATTERN.get(language)
                            if (pattern is not None
                                    and not pattern.search(translated_block)):
                                missing_target_script.append(label)
                        elif (_normalized_block(source_block)
                              != _normalized_block(translated_block)):
                            protected_block_errors.append(label)
                        if _looks_degenerate(
                            source_block, translated_block, language
                        ):
                            failures.append(
                                f"{language}/{label}: degenerate block"
                            )
                if _looks_degenerate(source, translated_text, language):
                    failures.append(f"{language}/{key}: degenerate translation")
                # Compare whole-document literals in the canonical parser
                # layout. Wrapped simple-table prose can be one short quoted
                # fragment in the raw docstring and several complete quoted
                # cells after rebuilding. Every prose block already passed
                # the strict literal gate above; canonical layout prevents the
                # document check from mistaking those translations for newly
                # invented source literals.
                syntax_contract(
                    rebuild_document(_source_layout, source_blocks),
                    translated_text,
                    f"{language}/{key}",
                )
        if block_layout_errors:
            failures.append(
                f"{language}: {len(block_layout_errors)} API entries have "
                "translated/source block-count mismatches "
                f"({', '.join(block_layout_errors[:5])})"
            )
        if unexpected_unchanged:
            failures.append(
                f"{language}: {len(unexpected_unchanged)} API prose blocks "
                "remain exact English outside the hash allowlist "
                f"({', '.join(unexpected_unchanged[:5])})"
            )
        if missing_target_script:
            failures.append(
                f"{language}: {len(missing_target_script)} API prose blocks "
                "lack target script "
                f"({', '.join(missing_target_script[:5])})"
            )
        if protected_block_errors:
            failures.append(
                f"{language}: {len(protected_block_errors)} API blocks changed "
                "protected code/literals "
                f"({', '.join(protected_block_errors[:5])})"
            )
        if contextual_errors:
            failures.append(
                f"{language}: {len(contextual_errors)} API blocks retain "
                "reviewed contextual false friends "
                f"({', '.join(contextual_errors[:5])})"
            )
        if contextual_translation_errors:
            failures.append(
                f"{language}: {len(contextual_translation_errors)} API blocks "
                "failed the contextual-source translation gate "
                f"({', '.join(contextual_translation_errors[:5])})"
            )
        if english_residue_errors:
            failures.append(
                f"{language}: {len(english_residue_errors)} API blocks retain "
                "ordinary English prose words "
                f"({', '.join(english_residue_errors[:5])})"
            )
        if copied_english_errors:
            failures.append(
                f"{language}: {len(copied_english_errors)} API blocks retain "
                "copied English prose sequences "
                f"({', '.join(copied_english_errors[:5])})"
            )
        if semantic_false_friend_errors:
            failures.append(
                f"{language}: {len(semantic_false_friend_errors)} API blocks "
                "retain reviewed semantic false friends "
                f"({', '.join(semantic_false_friend_errors[:5])})"
            )
        readme_path = README_DIR / f"README.{language}.rst"
        if not readme_path.is_file():
            failures.append(f"{language}: translated README is missing")
        else:
            readme = readme_path.read_text(encoding="utf-8")
            if len(readme) < 10_000:
                failures.append(f"{language}: translated README is too short")
            contract_readme = readme.replace(
                "<../../../README.rst>", "<README.rst>"
            ).replace(
                "<../TRANSLATION_MODELS.md>",
                "<docs/i18n/TRANSLATION_MODELS.md>",
            )
            contract_readme = re.sub(
                r"<README\.([A-Za-z_]+)\.rst>",
                r"<docs/i18n/readme/README.\1.rst>",
                contract_readme,
            )
            syntax_contract(
                readme_source,
                contract_readme,
                f"{language}/README",
                readme_protected_pattern,
            )
            if "../../../README.rst" not in readme:
                failures.append(f"{language}: English README link is broken")
    if failures:
        print("\n".join(failures[:200]), file=sys.stderr)
        return 1
    print(f"verified API catalogs: languages={len(tuple(languages))} symbols={len(docs)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument(
        "--model-root", type=Path,
        default=Path("/mnt/firecuda2/Claude/toxoplasma_projects/tutorials/project/translation_models/opus"),
    )
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument(
        "--repair-api-blocks",
        action="store_true",
        help=(
            "repair only stale, exact-English, wrong-script or "
            "literal-damaged API prose blocks; do not regenerate READMEs"
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="retranslate current API entries and README instead of reusing them",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    docs = public_docstrings()
    if args.audit:
        return audit(docs, args.languages)

    _write_json(API_DIR / "en.json", _english_manifest(docs))
    print(f"wrote English API manifest: symbols={len(docs)}")
    if args.sources_only:
        return 0
    if args.repair_api_blocks:
        for language in args.languages:
            translated = repair_api_translations(
                docs, language, args.model_root, args
            )
            write_language(docs, language, translated)
        return audit(docs, args.languages)

    readme = README_SOURCE.read_text(encoding="utf-8")
    readme_links: list[tuple[str, str, str]] = []
    for index, match in enumerate(re.finditer(
        r"`([^`<>]+?)\s+<([^>]+)>`_", readme
    )):
        label, target = match.group(1), match.group(2)
        # The language picker deliberately shows every language in its own
        # spelling. All other prose labels (not their destinations) belong to
        # the translated GitHub page.
        if target == "README.rst" or target.startswith(
            "docs/i18n/readme/README."
        ):
            continue
        key = f"__readme_link_{index}__"
        readme_links.append((key, label, target))
    for language in args.languages:
        reusable = {} if args.force else reusable_api_translations(
            docs, language,
        )
        pending = {key: source for key, source in docs.items() if key not in reusable}
        translated = dict(reusable)
        if pending:
            translated.update(
                _translate_api_documents(
                    pending, language, args.model_root, args,
                )
            )
        write_language(docs, language, translated)

        readme_path = README_DIR / f"README.{language}.rst"
        rebuild_readme = args.force or not readme_path.is_file()
        if rebuild_readme:
            documents = {"__readme__": readme}
            documents.update({key: label for key, label, _target in readme_links})
            readme_translation = _translate_documents(
                documents, language, args.model_root, args,
            )
            README_DIR.mkdir(parents=True, exist_ok=True)
            localized_readme = readme_translation["__readme__"]
            localized_readme = localized_readme.replace(
                "Languages:", f"{LANGUAGE_PICKER_LABELS[language]}:", 1
            )
            for key, label, target in readme_links:
                localized_readme = localized_readme.replace(
                    f"`{label} <{target}>`_",
                    f"`{readme_translation[key]} <{target}>`_",
                )
            localized_readme = localized_readme.replace(
                "docs/i18n/readme/README.", "README."
            ).replace(
                "docs/i18n/TRANSLATION_MODELS.md", "../TRANSLATION_MODELS.md"
            ).replace(
                "<README.rst>", "<../../../README.rst>"
            )
            readme_path.write_text(localized_readme + "\n", encoding="utf-8")
        print(
            f"wrote {language}: API={len(docs)} "
            f"translated={len(pending)} README={int(rebuild_readme)}"
        )
    return audit(docs, args.languages)


if __name__ == "__main__":
    raise SystemExit(main())
