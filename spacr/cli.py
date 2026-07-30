"""``spacr-run`` — run any spaCR module from a settings file, with no GUI.

Every spaCR pipeline has, until now, been reachable only through a GUI: the
PySide6 app (``spacr`` / ``spacr-qt``) or the classic Tk app (``spacr-tk``).
That is fine on a workstation and impossible on a cluster — importing either
entry point pulls Qt or Tk, and a compute node has no display to give them.

This module is the headless path. It is deliberately, testably light: importing
``spacr.cli`` must not pull Qt, Tk, torch, cellpose, numpy or pandas. Everything
heavy is imported inside the command that needs it, so ``spacr-run --help`` and
``spacr-run --list`` answer instantly on a login node, and ``--dry-run``
validates a 40-plate settings file without touching a GPU.

Usage::

    spacr-run <module> --settings settings.csv [--set key=value ...] [--dry-run]
    spacr-run --list                  # every module that can run headless
    spacr-run --describe measure      # what it does, what it needs, what it writes
    spacr-run validate --settings f --module mask     # pre-flight only

The settings file is the one the GUI writes. Both spaCR CSV layouts are read —
``Key,Value`` (what :func:`spacr.utils.save_settings` emits next to every run)
and ``setting_key,setting_value`` (the documented default of
:func:`spacr.utils.load_settings`) — plus the ``settings.json`` written into
each run-journal folder. So the round trip is: click through the GUI once on a
laptop, copy ``<src>/settings/gen_mask_settings.csv`` to the cluster, and
``sbatch`` it unchanged.

Exit codes (a cluster job that exits 0 after failing is the classic headless
footgun, so these are exact):

  0  the module ran to completion, or the dry run / validation found no errors
  1  the module raised
  2  bad arguments, unreadable settings, or pre-flight found errors

Matplotlib is forced to ``Agg`` when there is no display, and ``plt.show`` is
replaced by a close-the-figure shim for the duration of the run — the same
thing :func:`spacr.gui_utils.spacrFigShow` does inside the GUI — so a pipeline
that calls ``plt.show()`` neither blocks nor leaks figures.
"""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import importlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "EXIT_OK",
    "EXIT_RUNTIME",
    "EXIT_USAGE",
    "Module",
    "MODULES",
    "ALIASES",
    "SettingsError",
    "resolve_module",
    "module_defaults",
    "load_settings_file",
    "coerce_value",
    "apply_overrides",
    "resolve_settings",
    "build_parser",
    "main",
]

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2

LOG = logging.getLogger("spacr.cli")


class SettingsError(Exception):
    """A settings file, an override or a module name the user got wrong.

    Always maps to exit code 2: the run never started, so it is an argument
    problem rather than a runtime failure.
    """


# ---------------------------------------------------------------------------
# module registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Module:
    """One headless-runnable spaCR pipeline.

    :param key: name the user types, matching the GUI's ``app_key``.
    :param summary: one line for ``--list``.
    :param entry: ``"module:function"`` of the callable that does the work.
    :param defaults: name of the ``spacr.settings`` helper that fills the
        defaults, or ``None`` when the pipeline has none there.
    :param validate_key: app key understood by :mod:`spacr.validate`; empty
        when that module has no specific rules and only generic checks apply.
    :param requires: settings that must be supplied, phrased for a human.
    :param writes: what lands on disk.
    :param call_style: ``"settings"`` for ``fn(settings_dict)``; ``"folder"``
        for the one entry point that takes a bare path.
    :param note: caveat worth printing in ``--describe``.
    :param defaults_entry: ``"module:function"`` of a defaults helper that does
        **not** live in :mod:`spacr.settings`. Two pipelines keep their own
        (``spacr.foreign.default_settings``, ``spacr.convert.default_settings``)
        because their keys are theirs alone; without this the CLI would resolve
        an empty defaults dict for them, and ``--set`` would then reject every
        one of their keys as a setting that does not exist.
    """

    key: str
    summary: str
    entry: str
    defaults: Optional[str]
    validate_key: str
    requires: Tuple[str, ...] = ()
    writes: Tuple[str, ...] = ()
    call_style: str = "settings"
    note: str = ""
    defaults_entry: str = ""

    @property
    def module_name(self) -> str:
        """Import path of the module holding :attr:`entry`."""
        return self.entry.split(":", 1)[0]

    @property
    def func_name(self) -> str:
        """Attribute name of the callable inside :attr:`module_name`."""
        return self.entry.split(":", 1)[1]

    @property
    def defaults_label(self) -> str:
        """How ``--describe`` names this module's defaults helper."""
        if self.defaults_entry:
            return self.defaults_entry.replace(":", ".") + "()"
        if self.defaults:
            return f"spacr.settings.{self.defaults}()"
        return ""


# The mapping below is not invented: every entry is a callable that some
# existing dispatcher already runs. The sources, in order of authority:
#
#   * spacr/qt/bridge.py :: resolve_pipeline_entry  — the PySide6 GUI
#   * spacr/gui_utils.py :: run_function_gui        — the Tk GUI
#   * spacr/validate.py  :: APP_FUNCTIONS           — the pre-flight registry
#
# and the defaults helper for each is the one the pipeline itself calls to
# canonicalize its settings (grep "from .settings import" in the target
# module), not the one a GUI screen happens to show.
#
# Every app in spacr.qt.app.APPS is either here or in INTERACTIVE_ONLY below,
# and tests/test_app_registry_parity.py fails when one is in neither. Three
# were in neither until that test was written: `invasion` and `replication`
# (both Toxo assays with a Qt button, a settings panel and a submodules entry
# point, but no `spacr-run`) and `foreign`, which even had a validate entry.
# An app that ships with a GUI button and no headless path is an app nobody
# can run on a cluster, and nothing said so.
_MODULE_LIST: Tuple[Module, ...] = (
    Module(
        key="mask",
        summary="Segment cells / nuclei / pathogens with Cellpose and write merged stacks.",
        entry="spacr.core:preprocess_generate_masks",
        defaults="set_default_settings_preprocess_generate_masks",
        validate_key="mask",
        requires=("src — folder of raw acquisition images",
                  "at least one of cell_channel / nucleus_channel / "
                  "pathogen_channel / organelle_channel"),
        writes=("<src>/masks/", "<src>/merged/*.npy",
                "<src>/settings/gen_mask_settings.csv"),
    ),
    Module(
        key="timelapse",
        summary="Mask pipeline with object tracking across the frames of a time series.",
        entry="spacr.core:preprocess_generate_masks_timelapse",
        defaults="get_timelapse_settings",
        validate_key="mask",
        requires=("src — folder of raw time-series images",
                  "at least one segmentation channel",
                  "timelapse_mode — trackpy / btrack / iou / trackastra / ultrack"),
        writes=("<src>/masks/", "<src>/merged/*.npy", "tracked object tables"),
    ),
    Module(
        key="motility",
        summary="Automated motility assay: per-track velocity plus infection QC.",
        entry="spacr.timelapse:automated_motility_assay",
        defaults="get_automated_motility_assay_default_settings",
        validate_key="",
        requires=("src — plate folder already processed by the timelapse module",),
        writes=("motility results tables and QC figures next to src",),
    ),
    Module(
        key="measure",
        summary="Measure per-object morphology / intensity and crop single-object PNGs.",
        entry="spacr.measure:measure_crop",
        defaults="get_measure_crop_settings",
        validate_key="measure",
        requires=("src — plate folder holding merged/*.npy written by the mask module",
                  "the *_mask_dim of every object named in crop_mode",
                  "normalize — a [lower, upper] percentile pair, or False"),
        writes=("<src>/measurements/measurements.db", "<src>/data/**/<mode>_png/"),
    ),
    Module(
        key="align",
        summary="Register and stitch an arbitrary number of tiles into one canvas.",
        entry="spacr.align:align_folder",
        defaults=None,
        validate_key="align",
        requires=("src — folder of .npy/.tif tiles",),
        writes=("<dst>/<plate>_<well>_stitched.npy — the stitched canvas",
                "align_coordinates in measurements.db when db_path is set"),
    ),
    Module(
        key="foreign",
        summary="Import someone else's images, masks and measurement table as a spaCR project.",
        entry="spacr.foreign:import_project",
        defaults=None,
        defaults_entry="spacr.foreign:default_settings",
        validate_key="foreign",
        requires=("images — their folder of images",
                  "masks — their mask folder, or a list of them",
                  "measurements — their measurement table (CSV / TSV / sqlite)",
                  "column_map — a reviewed map file from a preview_only run"),
        writes=("<dst>/ (default <images>_spacr) — a spaCR project: renamed images, "
                "masks, and measurements.db with their columns mapped onto spaCR's",),
        note=("Takes 'images', not 'src' — there is no spaCR project yet. Run "
              "it once with --set preview_only=True: that prints the column "
              "mapping and the join counts, writes nothing, and is the only way "
              "to see what a column_map would have to fix."),
    ),
    Module(
        key="external_masks",
        summary="Measure images using label masks generated outside spaCR.",
        entry="spacr.external_masks:prepare_external_masks",
        defaults=None,
        defaults_entry="spacr.external_masks:default_settings",
        validate_key="external_masks",
        requires=(
            "inputs — image/mask paths or reviewed input-group mappings",
            "each mask group assigned to cell, nucleus, pathogen or organelle",
            "dst — a new output project folder",
        ),
        writes=(
            "<dst>/merged/*.npy and masks/*_mask_stack/*.npy",
            "<dst>/measurements/measurements.db",
            "<dst>/data/**/<object>_png/ for annotation",
        ),
    ),
    Module(
        key="classify",
        summary="Full DL pipeline: build dataset, train, apply the model, merge predictions.",
        entry="spacr.deep_spacr:deep_spacr",
        defaults="deep_spacr_defaults",
        validate_key="classify",
        requires=("src — plate folder with per-object PNGs from the measure module",
                  "classes — the class names",
                  "model_path when train=False and apply_model_to_dataset=True"),
        writes=("<src>/datasets/", "<src>/model/*.pth",
                "predictions merged into measurements.db"),
        note=("Same function the Classify (CV) button runs in both GUIs, so a "
              "settings.csv saved there behaves identically here. Use the "
              "'train_only' module for the training stage alone."),
    ),
    Module(
        key="train_only",
        summary="The training stage alone: train / evaluate on an existing dataset folder.",
        entry="spacr.deep_spacr:train_test_model",
        defaults="get_train_test_model_settings",
        validate_key="classify",
        requires=("src — dataset folder laid out as train/<class>/*.png and test/<class>/*.png",
                  "classes — the class folder names",
                  "train and/or test"),
        writes=("<src>/model/*.pth", "training + evaluation metrics CSVs"),
        note=("Ignores generate_training_dataset and apply_model_to_dataset — "
              "the dataset must already exist. Use 'classify' for the full "
              "pipeline, which is what both GUIs run."),
    ),
    Module(
        key="activation",
        summary="Generate class-activation maps for a trained classifier.",
        entry="spacr.deep_spacr:generate_activation_map",
        defaults="get_default_generate_activation_map_settings",
        validate_key="",
        requires=("dataset — tar of single-object PNGs", "model_path — trained checkpoint"),
        writes=("activation-map PNGs and correlation CSVs next to the dataset",),
    ),
    Module(
        key="umap",
        summary="Embed single-object images with UMAP and plot them as image glyphs.",
        entry="spacr.core:generate_image_umap",
        defaults="set_default_umap_image_settings",
        validate_key="umap",
        requires=("src — plate folder holding measurements/measurements.db",),
        writes=("UMAP embedding CSV and figure next to src",),
    ),
    Module(
        key="ml_analyze",
        summary="Classical ML (XGBoost / RF / logistic) on per-object screen features.",
        entry="spacr.ml:generate_ml_scores",
        defaults="set_default_analyze_screen",
        validate_key="ml_analyze",
        requires=("src — plate folder holding measurements/measurements.db",
                  "positive/negative control wells, or an annotation_column"),
        writes=("<src>/results/ — per-object scores, feature importance, plate heatmap",),
    ),
    Module(
        key="regression",
        summary="Regress per-well scores against sgRNA counts to call screen hits.",
        entry="spacr.ml:perform_regression",
        defaults="get_perform_regression_default_settings",
        validate_key="regression",
        requires=("score_data — CSV(s) of per-well scores",
                  "count_data — CSV(s) of per-well sgRNA counts",
                  "dependent_variable — the score column to regress"),
        writes=("volcano plots, plate heatmaps, gene phenotype plots, GO reports",),
    ),
    Module(
        key="map_barcodes",
        summary="Map row / column / gRNA barcodes out of sequencing reads onto wells.",
        entry="spacr.sequencing:generate_barecode_mapping",
        defaults="set_default_generate_barecode_mapping",
        validate_key="map_barcodes",
        requires=("src — folder of FASTQ reads",
                  "grna_csv, row_csv, column_csv — barcode tables with name/sequence columns",
                  "regex — must name the columnID / grna / rowID groups"),
        writes=("<src>/*.h5 read table", "unique-combination and QC CSVs"),
    ),
    Module(
        key="recruitment",
        summary="Analyze recruitment of a channel of interest to pathogen compartments.",
        entry="spacr.submodules:analyze_recruitment",
        defaults="get_analyze_recruitment_default_settings",
        validate_key="recruitment",
        requires=("src — plate folder holding measurements/measurements.db",
                  "channel_of_interest", "cell_plate_metadata / pathogen_plate_metadata"),
        writes=("recruitment figures and per-condition CSVs next to src",),
    ),
    Module(
        key="invasion",
        summary="Red/green invasion assay: score every parasite attached or invaded, per well.",
        entry="spacr.submodules:analyze_invasion",
        defaults="set_analyze_invasion_defaults",
        validate_key="invasion",
        requires=("src — plate folder holding measurements/measurements.db",
                  "outside_channel / total_channel — the pre- and "
                  "post-permeabilisation stain channels",
                  "pathogen_types + pathogen_plate_metadata — which wells are "
                  "which condition",
                  "control_wells — wells whose parasites carry no outside stain, "
                  "if you have them"),
        writes=("<src>/results/analyze_invasion/parasite_calls.csv, "
                "field_thresholds.csv, well_invasion.csv, condition_summary.csv, "
                "condition_comparisons.csv, chi_squared_results.csv",
                "<src>/results/analyze_invasion/invasion_per_well.pdf and "
                "invasion_by_condition.pdf",
                "<src>/settings/analyze_invasion.csv"),
        note=("'Invaded' is defined by the ABSENCE of outside stain, so every "
              "staining or focus failure inflates invasion efficiency and "
              "nothing pushes it the other way. Headless, set control_wells: "
              "the threshold is then a quantile of a real negative "
              "distribution rather than an Otsu cut on whatever the field "
              "happened to contain, and threshold_source says which you got."),
    ),
    Module(
        key="replication",
        summary="Count parasites per vacuole and compare replication distributions.",
        entry="spacr.submodules:analyze_replication",
        defaults="set_analyze_replication_defaults",
        validate_key="replication",
        requires=("src — plate folder holding measurements/measurements.db",
                  "one row per segmented parasite with centroids or a "
                  "vacuole-ID column",
                  "cell_types / pathogen_types / treatments and their "
                  "*_plate_metadata well maps, which define group_column"),
        writes=("<src>/results/analyze_replication/vacuole_counts.csv, "
                "well_distribution.csv, condition_summary.csv and tests",
                "<src>/results/analyze_replication/"
                "parasites_per_vacuole_*.pdf",
                "<src>/settings/analyze_replication.csv"),
        note=("The counting unit is a vacuole, not a host cell. Check the "
              "reported vacuole_key and non-power-of-two QC fraction before "
              "quoting the result."),
    ),
    Module(
        key="endodyogeny",
        summary="Legacy size proxy: bin pathogen area-derived volume by doublings.",
        entry="spacr.submodules:analyze_endodyogeny",
        defaults="set_analyze_endodyogeny_defaults",
        validate_key="endodyogeny",
        requires=("src — plate folder holding measurements/measurements.db",
                  "um_per_px — pixel calibration used by the size bins"),
        writes=("<src>/results/analyze_endodyogeny/ — proxy tables and plots",),
        note=("This is not a parasite count: pathogen areas are collapsed onto "
              "host cells. Use `spacr-run replication` when individual "
              "parasites are resolvable."),
    ),
    Module(
        key="analyze_plaques",
        summary="Segment and quantify plaques in a plaque assay.",
        entry="spacr.submodules:analyze_plaques",
        defaults="get_analyze_plaque_settings",
        validate_key="analyze_plaques",
        requires=("src — folder of plaque assay images",),
        writes=("plaque masks and a per-image plaque count / area CSV",),
    ),
    Module(
        key="train_cellpose",
        summary="Train or fine-tune a Cellpose model on your own labelled images.",
        entry="spacr.submodules:train_cellpose",
        defaults="get_train_cellpose_default_settings",
        validate_key="train_cellpose",
        requires=("src — folder of images plus matching label masks",
                  "model_name — where the trained model is saved"),
        writes=("<src>/models/<model_name>",),
    ),
    Module(
        key="cellpose_masks",
        summary="Run one Cellpose model over a folder and save the masks.",
        entry="spacr.spacr_cellpose:identify_masks_finetune",
        defaults="get_identify_masks_finetune_default_settings",
        validate_key="cellpose_masks",
        requires=("src — folder of images", "model_name or custom_model"),
        writes=("<dst>/ — one mask .tif per input image",),
    ),
    Module(
        key="cellpose_all",
        summary="Compare every available Cellpose model on the same images.",
        entry="spacr.spacr_cellpose:check_cellpose_models",
        defaults="get_check_cellpose_models_default_settings",
        validate_key="cellpose_all",
        requires=("src — folder of images",),
        writes=("<src>/cellpose_test/ — masks and a comparison figure per model",),
    ),
    Module(
        key="convert",
        summary="Convert vendor images into mapped, collision-safe Yokogawa TIFFs.",
        entry="spacr.convert:convert_folder",
        defaults=None,
        defaults_entry="spacr.convert:default_settings",
        validate_key="convert",
        requires=("src — folder of images to convert",),
        writes=("<dst>/ — Yokogawa TIFFs, conversion_map.csv, and a run ledger",),
        note=("The default keeps every Z plane. Set z_handling='max' or "
              "'first' only when lossy projection is intentional."),
    ),
    Module(
        key="simulation",
        summary="Sweep the pooled-screen simulator across a grid of parameters.",
        entry="spacr.sim:run_multiple_simulations",
        defaults=None,
        validate_key="simulation",
        requires=("max_workers — process-pool size (None means cpu_count - 4)",
                  "the sweep grid keys read by spacr.sim.generate_paramiters"),
        writes=("one results CSV per simulation under the configured output folder",),
        note=("No set_default_* helper exists for the simulator, so every key must "
              "come from the settings file."),
    ),
)

MODULES: Dict[str, Module] = {m.key: m for m in _MODULE_LIST}

# Friendly spellings. Seeded from spacr.validate.APP_ALIASES so a name that
# works there works here, plus the function names themselves.
ALIASES: Dict[str, str] = {
    "sequencing": "map_barcodes",
    "barcodes": "map_barcodes",
    "barcode_mapping": "map_barcodes",
    "generate_barecode_mapping": "map_barcodes",
    "preprocess_generate_masks": "mask",
    "generate_masks": "mask",
    "masks": "mask",
    "measure_crop": "measure",
    "train_test_model": "train_only",
    "train": "train_only",
    "deep_spacr": "classify",
    "classify_dl": "classify",
    "classify_ml": "ml_analyze",
    "generate_ml_scores": "ml_analyze",
    "generate_image_umap": "umap",
    "embedding": "umap",
    "perform_regression": "regression",
    "analyze_recruitment": "recruitment",
    "analyze_invasion": "invasion",
    "invasion_assay": "invasion",
    "analyze_replication": "replication",
    "analyze_endodyogeny": "endodyogeny",
    "replication_assay": "replication",
    "import_project": "foreign",
    "foreign_import": "foreign",
    "prepare_external_masks": "external_masks",
    "import_external_masks": "external_masks",
    "plaques": "analyze_plaques",
    "plaque": "analyze_plaques",
    "motility_assay": "motility",
    "sim": "simulation",
    "activation_map": "activation",
}

# Apps the GUI offers that have NO headless-runnable callable. Naming them in
# the error message is kinder than "unknown module": the user did not typo, the
# thing simply cannot run without a person looking at a screen.
INTERACTIVE_ONLY: Dict[str, str] = {
    "annotate": "Annotate paints labels onto a grid of single-object images by hand; "
                "run it in the GUI (spacr-qt) — there is no batch equivalent.",
    "make_masks": "Make Masks is a manual mask editor; run it in the GUI (spacr-qt).",
    "queue": "Plate Queue is a GUI convenience that chains plates through another "
             "module. Headless, loop over plates in your batch script and call "
             "spacr-run once per plate.",
    "db_browser": "Database Browser is an interactive sqlite viewer; use the sqlite3 "
                  "CLI or pandas on measurements.db instead.",
    "agreement": "Annotator Agreement is an interactive review of the crops two "
                 "annotators disagreed on; headless, call "
                 "spacr.agreement.agreement_report + format_agreement from Python.",
    "plate_view": "Plate Viewer is an interactive heatmap; headless, call "
                  "spacr.plate_qc.detect_edge_effect + format_edge_report.",
    "model_compare": "Model Compare runs two Cellpose models side by side for you to "
                     "look at; headless, call spacr.model_compare.compare_models.",
    "batch": "Batch Runner is the GUI for building a queue file. Headless, run the "
             "queue itself: from spacr.batch import load_queue, run_queue; "
             "run_queue(load_queue('night.queue.json'), path='night.queue.json') "
             "-- each job in it is a spacr-run invocation.",
    "model_zoo": "Model Zoo is an interactive browser; headless, call "
                 "spacr.model_zoo.discover_local + format_zoo, and "
                 "benchmark(entry, source=...) to test one on three fields.",
    "report": "Report is a one-click document builder; headless, call "
              "spacr.report.build_report(src, out, fmt='html').",
    "train_compare": "Training Runs is an interactive curve/settings comparison; "
                     "headless, use spacr.train_compare.find_runs + "
                     "format_comparison from Python.",
    "run_history": "Run History is an interactive searchable dashboard; headless, "
                   "call spacr.run_journal.search_runs() instead.",
}


def resolve_module(name: Any) -> Optional[Module]:
    """Return the :class:`Module` for a user-typed name, or None.

    :param name: module key, alias, or the bare name of the pipeline function.
    :returns: the matching :class:`Module`, or None when nothing matches.
    """
    if not isinstance(name, str):
        return None
    key = name.strip().lower().replace("-", "_")
    key = ALIASES.get(key, key)
    return MODULES.get(key)


def _unknown_module_message(name: str) -> str:
    """Explain an unrecognised module name, with a suggestion when there is one."""
    key = str(name).strip().lower().replace("-", "_")
    if key in INTERACTIVE_ONLY:
        return (f"'{name}' is a GUI-only module and cannot run headless.\n"
                f"  {INTERACTIVE_ONLY[key]}")
    pool = sorted(set(MODULES) | set(ALIASES))
    close = difflib.get_close_matches(key, pool, n=1, cutoff=0.6)
    hint = f" Did you mean '{resolve_module(close[0]).key}'?" if close else ""
    return (f"unknown module '{name}'.{hint}\n"
            f"  Run 'spacr-run --list' to see every module that can run headless.")


# ---------------------------------------------------------------------------
# settings: defaults, file, overrides
# ---------------------------------------------------------------------------


def module_defaults(module: Module) -> Dict[str, Any]:
    """Return a fresh defaults dict for ``module``.

    Calls the same helper the pipeline itself uses to canonicalize its
    settings, so the resolved dict the CLI prints is the one the pipeline will
    see. That is usually a :mod:`spacr.settings` function
    (:attr:`Module.defaults`); for the pipelines that keep their own it is
    :attr:`Module.defaults_entry`, imported here rather than at module load so
    ``--list`` stays instant.

    :param module: the module whose defaults are wanted.
    :returns: dict of defaults; empty when the pipeline has no helper.
    """
    fn = None
    if module.defaults_entry:
        target, _, name = module.defaults_entry.partition(":")
        try:
            fn = getattr(importlib.import_module(target), name, None)
        except Exception:
            # A missing optional dependency must not break --describe; the run
            # itself will fail loudly in import_entry with a real message.
            return {}
    elif module.defaults:
        from . import settings as _settings

        fn = getattr(_settings, module.defaults, None)
    if fn is None:
        return {}
    try:
        produced = fn({})
    except TypeError:
        produced = fn()
    return dict(produced) if isinstance(produced, dict) else {}


# Column-name pairs a spaCR settings CSV can use. ('Key', 'Value') is what
# spacr.utils.save_settings writes next to every run and what the GUI's
# "Export settings" button produces; ('setting_key', 'setting_value') is the
# documented default of spacr.utils.load_settings.
_CSV_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("Key", "Value"),
    ("setting_key", "setting_value"),
    ("key", "value"),
    ("Setting", "Value"),
    ("name", "value"),
)


def _parse_csv_value(value: Any) -> Any:
    """Turn one CSV cell back into its original Python type.

    A faithful port of the ``parse_value`` closure inside
    :func:`spacr.utils.load_settings`, reproduced here rather than imported
    because ``spacr.utils`` pulls torch and cellpose — twenty seconds and a
    CUDA context to read a two-column CSV. :mod:`spacr.validate` reproduces
    ``_get_regex`` for the same reason. Any behaviour change there must be
    mirrored here; the round trip is covered by ``tests/test_cli.py``.

    :param value: raw cell text.
    :returns: bool, int, float, None, list, tuple, dict or str.
    """
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    if value.startswith(("(", "[", "{")):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
        if isinstance(parsed, dict):
            return {k: _parse_csv_value(v) for k, v in parsed.items()}
        return parsed
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    return value


def _load_settings_csv(path: str) -> Dict[str, Any]:
    """Read a two-column spaCR settings CSV into a dict.

    :param path: path to the CSV.
    :returns: parsed settings.
    :raises SettingsError: when no recognised key/value column pair is present.
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [f for f in (reader.fieldnames or []) if f is not None]
        pair = None
        for key_col, value_col in _CSV_COLUMNS:
            if key_col in fieldnames and value_col in fieldnames:
                pair = (key_col, value_col)
                break
        if pair is None:
            expected = " or ".join(f"'{k},{v}'" for k, v in _CSV_COLUMNS[:2])
            raise SettingsError(
                f"{path} is not a spaCR settings CSV: its columns are "
                f"{fieldnames or ['<none>']}, but {expected} is expected.\n"
                f"  Export the settings again from the GUI, or use the "
                f"settings.csv written into <src>/settings/ by any run.")
        key_col, value_col = pair
        out: Dict[str, Any] = {}
        for row in reader:
            key = row.get(key_col)
            if key is None or not str(key).strip():
                continue
            raw = row.get(value_col)
            overflow = row.get(None)
            if overflow and value_col == fieldnames[-1]:
                # A hand-edited CSV with an unquoted list — `channels,[0, 1, 2]`
                # — splits across columns. Rejoin rather than silently storing
                # the fragment '[0'.
                raw = ",".join([str(raw)] + [str(x) for x in overflow])
            out[str(key).strip()] = _parse_csv_value(raw)
    return out


def _load_settings_json(path: str) -> Dict[str, Any]:
    """Read a ``settings.json`` (as written into every run-journal folder).

    :param path: path to the JSON file.
    :returns: parsed settings.
    :raises SettingsError: when the file is not a JSON object.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SettingsError(
            f"{path} holds a {type(data).__name__}, not a settings object.")
    return data


def load_settings_file(path: Any) -> Dict[str, Any]:
    """Load a settings file written by the GUI, a pipeline run or a run journal.

    :param path: path to a ``.csv`` or ``.json`` settings file.
    :returns: the settings dict.
    :raises SettingsError: when the path is missing, unreadable or malformed.
        Never a traceback — a cluster job should fail with a sentence.
    """
    if not isinstance(path, str) or not path.strip():
        raise SettingsError("no settings file given; pass --settings <file>.")
    if not os.path.exists(path):
        raise SettingsError(
            f"settings file not found: {path}\n"
            f"  Check the path, and that the share holding it is mounted on this node.")
    if os.path.isdir(path):
        raise SettingsError(
            f"--settings expects a file, but {path} is a folder.\n"
            f"  A run writes its settings to <src>/settings/*.csv — point at one of those.")
    try:
        if path.lower().endswith(".json"):
            return _load_settings_json(path)
        return _load_settings_csv(path)
    except SettingsError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise SettingsError(f"could not read {path}: {exc}") from exc
    except (json.JSONDecodeError, csv.Error) as exc:
        raise SettingsError(f"could not parse {path}: {exc}") from exc


# expected_types declares a few keys more narrowly than the code that reads
# them. Mirrors _EXPECTED_TYPE_OVERRIDES in spacr.validate — kept in step with
# it deliberately, so a value the validator accepts is a value --set can write.
_TYPE_OVERRIDES: Dict[str, Tuple[type, ...]] = {
    "src": (str, list),
    "normalize": (bool, list),
    "save": (bool, list),
}

# Per-module narrowings, for keys whose name two pipelines share. Mirrors
# _APP_TYPE_OVERRIDES in spacr.validate for the same reason as above, and
# tests/test_app_registry_parity.py asserts the two are equal so the mirror
# cannot rot: `masks` is declared bool in expected_types (the mask pipeline's
# save switch), but spacr.foreign.import_project takes it as their mask folder,
# so `--set masks=/their/masks` was rejected as "cannot be read as bool".
_APP_TYPE_OVERRIDES: Dict[str, Dict[str, Tuple[type, ...]]] = {
    "foreign": {"masks": (str, list)},
}

_TRUE_WORDS = frozenset({"true", "t", "yes", "y", "on", "1"})
_FALSE_WORDS = frozenset({"false", "f", "no", "n", "off", "0"})
_NONE_WORDS = frozenset({"none", "null", "nil", ""})


def _allowed_types(key: str, current: Any, expected_types: Mapping[str, Any],
                   app: str = "") -> Tuple[type, ...]:
    """Types ``key`` may take, from ``expected_types`` or the current value.

    :param key: settings key being overridden.
    :param current: the value the key holds before the override, used to infer
        a type for keys that ``expected_types`` does not declare.
    :param expected_types: :data:`spacr.settings.expected_types`.
    :param app: module key, for the per-module narrowings above.
    :returns: tuple of types; empty means "anything, parse it literally".
    """
    per_app = _APP_TYPE_OVERRIDES.get(app, {})
    if key in per_app:
        return per_app[key]
    if key in _TYPE_OVERRIDES:
        return _TYPE_OVERRIDES[key]
    if key in expected_types:
        declared = expected_types[key]
        raw = declared if isinstance(declared, tuple) else (declared,)
        # expected_types spells NoneType two ways: type(None) for most keys and
        # a bare None for 'sample' / 'x_lim'. Normalize both.
        out = tuple(type(None) if t is None else t for t in raw if isinstance(t, type) or t is None)
        if out:
            return out
    if isinstance(current, bool):
        return (bool,)
    if isinstance(current, int):
        return (int,)
    if isinstance(current, float):
        return (float, int)
    if isinstance(current, (list, tuple)):
        return (list,)
    if isinstance(current, dict):
        return (dict,)
    if isinstance(current, str):
        return (str,)
    return ()


def _literal_scalar(text: str) -> Any:
    """Parse one bare token of a comma-separated list into a Python scalar."""
    token = text.strip()
    if token.lower() in _NONE_WORDS:
        return None
    if token.lower() in _TRUE_WORDS - {"1"}:
        return True
    if token.lower() in _FALSE_WORDS - {"0"}:
        return False
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token


def _type_label(types: Sequence[type]) -> str:
    """Render a tuple of types as readable prose for an error message."""
    if not types:
        return "any value"
    return " or ".join("None" if t is type(None) else getattr(t, "__name__", str(t))
                       for t in types)


def coerce_value(key: str, text: str, current: Any,
                 expected_types: Mapping[str, Any], app: str = "") -> Any:
    """Coerce a ``--set key=value`` string into the type the setting expects.

    The type comes from :data:`spacr.settings.expected_types` when the key is
    declared there, otherwise from the type of the value the key already holds.
    A value that cannot be coerced is an error rather than a silently-stored
    string: ``cell_mask_dim='4'`` is exactly the bug the settings CSV round trip
    keeps producing, and measure_crop only notices it an hour in.

    :param key: settings key.
    :param text: the raw text after the first ``=``.
    :param current: the value ``key`` holds before the override.
    :param expected_types: :data:`spacr.settings.expected_types`.
    :param app: module key, so a key two pipelines share is read as the module
        being run means it (see :data:`_APP_TYPE_OVERRIDES`).
    :returns: the coerced value.
    :raises SettingsError: when ``text`` is not a legal value for ``key``.
    """
    types = _allowed_types(key, current, expected_types, app)
    stripped = text.strip()
    lowered = stripped.lower()
    allow = (lambda t: True) if not types else (lambda t: t in types)

    if lowered in _NONE_WORDS and (not types or type(None) in types):
        return None

    if stripped.startswith("{") and allow(dict):
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError) as exc:
            raise SettingsError(f"--set {key}={text!r} is not a valid dict: {exc}") from exc
        if isinstance(parsed, dict):
            return parsed

    if stripped.startswith(("[", "(")) and (allow(list) or allow(tuple)):
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError) as exc:
            raise SettingsError(f"--set {key}={text!r} is not a valid list: {exc}") from exc
        if isinstance(parsed, (list, tuple)):
            return tuple(parsed) if (allow(tuple) and not allow(list)) else list(parsed)

    if allow(bool):
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False

    if allow(int):
        try:
            return int(stripped)
        except ValueError:
            # '4.0' for an int setting is a float that happens to be whole —
            # accept it rather than making the user retype it.
            try:
                as_float = float(stripped)
            except ValueError:
                as_float = None
            if as_float is not None and as_float.is_integer():
                return int(as_float)

    if allow(float):
        try:
            return float(stripped)
        except ValueError:
            pass

    if allow(str):
        return text

    if allow(list) or allow(tuple):
        items = [_literal_scalar(part) for part in stripped.split(",")] if stripped else []
        return tuple(items) if (allow(tuple) and not allow(list)) else items

    # Only reachable with a non-empty ``types``: when nothing is declared every
    # branch above is allowed and the ``str`` one always returns.
    raise SettingsError(
        f"--set {key}={text!r} cannot be read as {_type_label(types)}.\n"
        f"  {key} expects {_type_label(types)}; the current value is {current!r}.")


def _split_override(item: str) -> Tuple[str, str]:
    """Split a ``key=value`` override, raising a useful error when it has no ``=``."""
    if "=" not in item:
        raise SettingsError(
            f"--set {item!r} is not a key=value pair.\n"
            f"  Write it as --set {item}=<value> (quote values containing spaces).")
    key, _, value = item.partition("=")
    key = key.strip()
    if not key:
        raise SettingsError(f"--set {item!r} has an empty key.")
    return key, value


def apply_overrides(settings: Dict[str, Any], overrides: Sequence[str],
                    module: Optional[Module] = None) -> Dict[str, Any]:
    """Apply ``--set key=value`` overrides on top of a settings dict.

    An override naming a key spaCR does not know is an error, not a no-op: a
    typo'd override that quietly does nothing costs a whole run to discover,
    and the run looks like it succeeded.

    :param settings: settings resolved from defaults plus file; mutated in place.
    :param overrides: raw ``key=value`` strings from the command line.
    :param module: the module being run, used only for the error message.
    :returns: ``settings``.
    :raises SettingsError: on an unknown key or an uncoercible value.
    """
    if not overrides:
        return settings
    from .settings import expected_types

    known = set(settings) | set(expected_types)
    for item in overrides:
        key, text = _split_override(item)
        if key not in known:
            close = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.6)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            where = f" for module '{module.key}'" if module is not None else ""
            raise SettingsError(
                f"--set {key}=... names a setting that does not exist{where}: "
                f"'{key}'.{hint}\n"
                f"  Run 'spacr-run --describe {module.key if module else '<module>'}' "
                f"to list the settings this module accepts.")
        settings[key] = coerce_value(key, text, settings.get(key), expected_types,
                                     module.key if module is not None else "")
    return settings


def resolve_settings(module: Module, settings_path: Optional[str],
                     overrides: Sequence[str] = ()) -> Dict[str, Any]:
    """Build the settings dict the pipeline will actually receive.

    Layered lowest-to-highest: the module's own defaults, the settings file,
    then the ``--set`` overrides.

    :param module: module being run.
    :param settings_path: path to the settings file, or None for defaults only.
    :param overrides: ``key=value`` strings.
    :returns: the fully-resolved settings dict.
    :raises SettingsError: on any unreadable file, unknown key or bad value.
    """
    resolved = module_defaults(module)
    if settings_path:
        resolved.update(load_settings_file(settings_path))
    apply_overrides(resolved, overrides, module)
    return resolved


# ---------------------------------------------------------------------------
# headless environment
# ---------------------------------------------------------------------------


def _has_display() -> bool:
    """True when a windowing system is available for matplotlib to draw on."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def use_agg_if_headless() -> bool:
    """Force matplotlib's Agg backend when there is no display.

    Called before the first spaCR import that could pull pyplot. An explicit
    ``MPLBACKEND`` in the environment always wins, and interactive local use
    (a display is present) is left alone, so this only bites on a compute node.

    :returns: True when Agg was forced.
    """
    if os.environ.get("MPLBACKEND"):
        return False
    if _has_display():
        return False
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:  # matplotlib is optional for --list / --describe
        return False
    return True


class _NoShow:
    """Context manager that neutralises ``plt.show`` for the length of a run.

    A pipeline that calls ``plt.show()`` under Agg emits a UserWarning per
    figure and leaks every one of them — forty plates' worth of open figures is
    a real memory problem on a shared node. Closing the current figure instead
    is exactly what :func:`spacr.gui_utils.spacrFigShow` does inside the GUI,
    so the pipelines are already built for it.
    """

    def __init__(self) -> None:
        self._plt = None
        self._original = None

    def __enter__(self) -> "_NoShow":
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return self
        self._plt = plt
        self._original = plt.show

        def _close_instead(*args: Any, **kwargs: Any) -> None:
            try:
                plt.close(plt.gcf())
            except Exception:
                pass

        plt.show = _close_instead
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        if self._plt is not None and self._original is not None:
            try:
                self._plt.show = self._original
                self._plt.close("all")
            except Exception:
                pass
        return False


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure timestamped logging to stdout for a batch run.

    :param verbose: raise the level from INFO to DEBUG.
    :returns: the ``spacr.cli`` logger.
    """
    level = logging.DEBUG if verbose else logging.INFO
    LOG.setLevel(level)
    LOG.propagate = False
    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    LOG.addHandler(handler)
    return LOG


def _quiet_progress_bars() -> bool:
    """Disable tty-only progress rendering when stdout is a file or a pipe.

    :returns: True when stdout is not a tty and the bars were disabled.
    """
    try:
        is_tty = bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        is_tty = False
    if is_tty:
        return False
    # tqdm reads TQDM_DISABLE; spaCR's own print_progress already emits whole
    # lines, but the handful of `print(..., end='\r')` sites in io / utils /
    # sim do not, so a redirected log gets one long line from those.
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("SPACR_NO_PROGRESS", "1")
    return True


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    """Compact single-line rendering of a settings value."""
    text = repr(value) if isinstance(value, str) else str(value)
    return text if len(text) <= 70 else text[:67] + "..."


def render_settings(settings: Mapping[str, Any]) -> str:
    """Render a resolved settings dict as an aligned, sorted table.

    :param settings: the resolved settings.
    :returns: the table as one string, no trailing newline.
    """
    if not settings:
        return "  (no settings)"
    width = min(38, max(len(str(k)) for k in settings))
    return "\n".join(f"  {str(k).ljust(width)}  {_format_value(v)}"
                     for k, v in sorted(settings.items(), key=lambda kv: str(kv[0])))


def render_module_list() -> str:
    """Render the ``--list`` table of headless-runnable modules.

    :returns: the table as one string, no trailing newline.
    """
    width = max(len(m.key) for m in _MODULE_LIST)
    lines = ["spaCR modules that can run headless:", ""]
    for module in _MODULE_LIST:
        lines.append(f"  {module.key.ljust(width)}  {module.summary}")
        lines.append(f"  {' ' * width}  -> {module.module_name}.{module.func_name}()")
    lines.append("")
    lines.append("GUI-only (no headless equivalent):")
    for key in sorted(INTERACTIVE_ONLY):
        lines.append(f"  {key.ljust(width)}  {INTERACTIVE_ONLY[key]}")
    lines.append("")
    lines.append("Run 'spacr-run --describe <module>' for required settings and outputs.")
    return "\n".join(lines)


def render_module_description(module: Module) -> str:
    """Render the ``--describe`` block for one module.

    :param module: module to describe.
    :returns: the description as one string, no trailing newline.
    """
    lines = [f"{module.key} — {module.summary}", "=" * max(len(module.key) + 3, 60), ""]
    lines.append(f"  runs        {module.module_name}.{module.func_name}(settings)"
                 if module.call_style == "settings" else
                 f"  runs        {module.module_name}.{module.func_name}(settings['src'])")
    lines.append(f"  defaults    {module.defaults_label}"
                 if module.defaults_label else
                 "  defaults    none — every setting must come from the settings file")
    lines.append(f"  pre-flight  spacr.validate rules for '{module.validate_key}'"
                 if module.validate_key else
                 "  pre-flight  generic checks only (no module-specific rules)")

    try:
        defaults = module_defaults(module)
    except Exception:  # a broken settings helper must not break --describe
        defaults = {}
    if defaults:
        lines.append(f"  settings    {len(defaults)} keys, all optional unless listed below")

    if module.requires:
        lines.append("")
        lines.append("Required settings:")
        for item in module.requires:
            lines.append(f"  - {item}")
    if module.writes:
        lines.append("")
        lines.append("Writes:")
        for item in module.writes:
            lines.append(f"  - {item}")
    if module.note:
        lines.append("")
        lines.append(f"Note: {module.note}")
    lines.append("")
    lines.append(f"  spacr-run {module.key} --settings settings.csv --dry-run")
    return "\n".join(lines)


def _preflight(settings: Mapping[str, Any], validate_key: str,
               printer: Callable[[str], None] = print) -> List[Any]:
    """Validate settings against the data they point at and print the report.

    Delegates every rule to :mod:`spacr.validate` — :func:`validate_settings`
    for the checks, :func:`format_report` for the errors and warnings,
    :func:`describe_plan` for the "here is what would happen" summary. Only the
    trailer differs from :func:`spacr.validate.run_preflight`, which is worded
    for the in-pipeline ``dry_run=True`` setting rather than for ``--dry-run``.

    :param settings: the resolved settings.
    :param validate_key: app key understood by :mod:`spacr.validate`.
    :param printer: where the text goes.
    :returns: the list of ``spacr.validate.Problem`` found.
    """
    from .validate import describe_plan, format_report, validate_settings

    problems = validate_settings(dict(settings), validate_key)
    printer(format_report(problems, dict(settings), validate_key))
    printer("")
    printer(describe_plan(dict(settings), validate_key))
    return problems


def _error_count(problems: Sequence[Any]) -> int:
    """Number of problems that would break or corrupt the run."""
    return sum(1 for p in problems if getattr(p, "is_error", False))


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def import_entry(module: Module) -> Callable[..., Any]:
    """Import and return the pipeline callable for ``module``.

    Deliberately late: this is where torch, cellpose and the rest of the heavy
    stack finally load, long after ``--help`` and ``--list`` have answered.

    :param module: module whose entry point is wanted.
    :returns: the callable.
    :raises SettingsError: when the module or attribute cannot be imported.
    """
    try:
        imported = importlib.import_module(module.module_name)
    except Exception as exc:
        raise SettingsError(
            f"could not import {module.module_name} for module '{module.key}': "
            f"{type(exc).__name__}: {exc}\n"
            f"  Check that spaCR's dependencies are installed in this environment.") from exc
    func = getattr(imported, module.func_name, None)
    if func is None or not callable(func):
        raise SettingsError(
            f"{module.module_name} has no callable '{module.func_name}' — "
            f"module '{module.key}' cannot run against this spaCR version.")
    return func


def _call_entry(module: Module, func: Callable[..., Any],
                settings: Dict[str, Any]) -> Any:
    """Invoke a pipeline entry point with the calling convention it expects."""
    if module.call_style == "folder":
        src = settings.get("src")
        if not isinstance(src, str) or not src.strip():
            raise SettingsError(
                f"module '{module.key}' needs a single folder in src, "
                f"but src is {src!r}.")
        return func(src)
    return func(settings)


def cmd_list(_args: argparse.Namespace) -> int:
    """``--list`` — print every module that can run headless."""
    print(render_module_list())
    return EXIT_OK


def cmd_describe(name: str) -> int:
    """``--describe <module>`` — print one module's contract."""
    module = resolve_module(name)
    if module is None:
        print(_unknown_module_message(name), file=sys.stderr)
        return EXIT_USAGE
    print(render_module_description(module))
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    """``validate --settings f`` — pre-flight only, nothing is executed."""
    module = resolve_module(args.module) if args.module else None
    if args.module and module is None:
        print(_unknown_module_message(args.module), file=sys.stderr)
        return EXIT_USAGE
    if not args.settings:
        print("error: validate needs a settings file: "
              "spacr-run validate --settings <file> [--module <module>]",
              file=sys.stderr)
        return EXIT_USAGE

    try:
        if module is not None:
            settings = resolve_settings(module, args.settings, args.set or [])
        else:
            settings = load_settings_file(args.settings)
            apply_overrides(settings, args.set or [], None)
    except SettingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    validate_key = module.validate_key if module is not None else ""
    problems = _preflight(settings, validate_key)
    errors = _error_count(problems)
    print("")
    if errors:
        print(f"validate: {errors} error{'' if errors == 1 else 's'} — "
              f"these settings would not run.")
        return EXIT_USAGE
    print("validate: settings are runnable.")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    """``<module> --settings f`` — the real thing, or ``--dry-run`` for the plan."""
    module = resolve_module(args.module)
    if module is None:
        print(_unknown_module_message(args.module), file=sys.stderr)
        return EXIT_USAGE
    if not args.settings:
        print(f"error: no settings file given.\n"
              f"  spacr-run {module.key} --settings <file>   "
              f"(see 'spacr-run --describe {module.key}')", file=sys.stderr)
        return EXIT_USAGE

    log = setup_logging(args.verbose)
    try:
        settings = resolve_settings(module, args.settings, args.set or [])
    except SettingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    log.info("module   %s -> %s.%s()", module.key, module.module_name, module.func_name)
    log.info("settings %s (%d keys resolved)", args.settings, len(settings))
    if args.set:
        log.info("overrides %s", ", ".join(args.set))

    if args.dry_run:
        print("")
        print("Resolved settings:")
        print(render_settings(settings))
        print("")
        problems = _preflight(settings, module.validate_key)
        errors = _error_count(problems)
        print("")
        print(f"--dry-run: nothing was executed. "
              f"{module.module_name}.{module.func_name}() was not called.")
        if errors:
            log.error("dry run found %d error%s in the settings",
                      errors, "" if errors == 1 else "s")
            return EXIT_USAGE
        log.info("dry run clean — drop --dry-run to execute")
        return EXIT_OK

    if args.verbose:
        log.debug("resolved settings:\n%s", render_settings(settings))

    if not args.no_preflight:
        problems = _preflight(settings, module.validate_key)
        errors = _error_count(problems)
        if errors and not args.force:
            log.error("pre-flight found %d error%s; refusing to start.",
                      errors, "" if errors == 1 else "s")
            log.error("Fix them, or pass --force to run anyway "
                      "(or --no-preflight to skip the check).")
            return EXIT_USAGE
        if errors:
            log.warning("pre-flight found %d error%s — running anyway (--force).",
                        errors, "" if errors == 1 else "s")

    if use_agg_if_headless():
        log.info("no display detected — matplotlib backend forced to Agg")
    if _quiet_progress_bars():
        log.debug("stdout is not a tty — tty-only progress output disabled")

    try:
        func = import_entry(module)
    except SettingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    started = time.time()
    log.info("starting %s", module.key)
    try:
        from .run_journal import open_run
        with _NoShow():
            log.info("recording reproducibility input hashes")
            with open_run(module.key, settings) as run:
                log.info("reproducibility manifest %s", run.dir)
                _call_entry(module, func, settings)
    except SettingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        log.error("interrupted after %.1fs", time.time() - started)
        return EXIT_RUNTIME
    except BaseException as exc:  # noqa: BLE001 - a batch job must report, not crash
        if isinstance(exc, SystemExit):
            code = exc.code if isinstance(exc.code, int) else EXIT_RUNTIME
            log.info("%s exited with code %s after %.1fs",
                     module.key, code, time.time() - started)
            return code
        import traceback
        log.error("%s failed after %.1fs: %s: %s",
                  module.key, time.time() - started, type(exc).__name__, exc)
        traceback.print_exc()
        return EXIT_RUNTIME

    log.info("%s finished in %.1fs", module.key, time.time() - started)
    return EXIT_OK


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors exit 2 through the same path as ours."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def build_parser() -> argparse.ArgumentParser:
    """Return the ``spacr-run`` argument parser.

    Building the parser imports nothing beyond the standard library, so
    ``spacr-run --help`` is instant even on a node with a cold NFS cache.

    :returns: the parser.
    """
    parser = _Parser(
        prog="spacr-run",
        description="Run a spaCR module from a settings file, with no GUI and "
                    "no display.",
        epilog="Exit codes: 0 success, 1 the module raised, 2 bad arguments or "
               "settings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "module", nargs="?",
        help="Module to run, or 'validate' for a pre-flight check. "
             "Use --list to see them all.")
    parser.add_argument(
        "--settings", "-s", metavar="FILE",
        help="Settings CSV or JSON, as written by the GUI or by any spaCR run "
             "into <src>/settings/.")
    parser.add_argument(
        "--set", action="append", metavar="KEY=VALUE", default=[],
        help="Override one setting after the file is loaded. Repeatable. "
             "Unknown keys are an error.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved settings and the plan, run the pre-flight "
             "checks, and stop without executing anything.")
    parser.add_argument(
        "--module", "-m", dest="module_opt", metavar="MODULE",
        help="Module the settings belong to; only needed with the 'validate' "
             "subcommand.")
    parser.add_argument(
        "--force", action="store_true",
        help="Run even when the pre-flight check reports errors.")
    parser.add_argument(
        "--no-preflight", action="store_true",
        help="Skip the pre-flight check entirely.")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Log at DEBUG, including the fully-resolved settings.")
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List every module that can run headless, and exit.")
    parser.add_argument(
        "--describe", metavar="MODULE",
        help="Describe one module: what it runs, what it needs, what it "
             "writes; then exit.")
    parser.add_argument(
        "--version", action="store_true",
        help="Print the spaCR version and exit.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``spacr-run`` entry point.

    :param argv: argument list; ``sys.argv[1:]`` when None.
    :returns: process exit code — 0 success, 1 the module raised, 2 bad
        arguments or settings.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE

    if args.version:
        from .version import __version__
        print(__version__)
        return EXIT_OK

    if args.list:
        return cmd_list(args)

    if args.describe:
        return cmd_describe(args.describe)

    if not args.module:
        parser.print_usage(sys.stderr)
        print("error: no module given. Use --list to see what is available.",
              file=sys.stderr)
        return EXIT_USAGE

    if args.module.lower() == "validate":
        args.module = args.module_opt
        return cmd_validate(args)

    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
