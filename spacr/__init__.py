"""spaCR public package and version metadata."""

from __future__ import annotations

import warnings as _warnings
from importlib import import_module
from typing import Final

from .version import __version__

# Third-party FutureWarnings that fire at import — noise the user
# can't act on from inside spaCR. Silenced before the modules that trigger
# them import. The Statsmodels warning formerly listed here was fixed at its
# source by switching from the deprecated ``logit`` alias to ``Logit``.
# (Users can re-enable with `warnings.filterwarnings("default")` in
# their own code.)
_warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\..*",
    category=FutureWarning,
)
_warnings.filterwarnings(
    "ignore",
    message=r"You are using a Python version.*google\.api_core.*",
    category=FutureWarning,
)
_warnings.filterwarnings(
    "ignore",
    message=r"You are using a Python version.*",
    category=FutureWarning,
    module=r"google\..*",
)

# Cellpose 4 builds a sparse COO tensor in `dynamics.py` and torch notes that
# invariant checking is off. It fires on the first mask of every run, names a
# torch internal, and there is nothing a spaCR user can do about it.
#
# Both patterns are written the way `spacr.qt._LIBRARY_NOISE` explains, and
# for the same two reasons. The message is not anchored: `filterwarnings`
# matches it with `re.match`, so the anchored version this replaced would
# have missed the same notice from any build of torch that prefixes it. The
# module IS given, and it is the raising frame's dotted `__name__` -- NOT
# the `cellpose/dynamics.py` path a traceback shows, which is the natural
# thing to write here and matches nothing. Scoping it to cellpose means the
# sentence is only ignored where it is noise.
_warnings.filterwarnings(
    "ignore",
    message=r".*[Ss]parse invariant checks are implicitly disabled",
    category=UserWarning,
    module=r"cellpose(\.|$)",
)

_SUBMODULES: Final[tuple[str, ...]] = (
    "core",
    "schema",
    "database_schema",
    "database_concurrency",
    "io",
    "utils",
    "errors",
    "settings",
    "setting_animations",
    # Which widget a setting gets, decided without importing a GUI: the Tk
    # and Qt front ends read the same spec rather than each keeping their
    # own opinion about what a given key looks like.
    "settings_spec",
    "plot",
    "measure",
    # Opt-in preprocessing / region-filter extension points for the measure
    # path. Separate from `measure` so registering a hook does not import
    # matplotlib, skimage and cv2.
    "measure_hooks",
    # A drawn region of interest, honoured by Measure. Pure geometry plus
    # the mask it resolves to, so a headless run can apply an ROI drawn in
    # the GUI without importing one.
    "roi",
    # Illumination / flat-field correction. Estimates the microscope's uneven
    # illumination from the plate's own fields and applies it through the
    # preprocessing hook above, so measure.py needs no second path.
    "illumination",
    "measurement_schema",
    "sequencing",
    # QC over what `sequencing` produced — reads per well, starved wells,
    # barcode collisions, unmapped reads, library coverage — plus the
    # gRNAs-per-well target that derives the abundance threshold. Separate
    # from `sequencing` so the multiprocessing read workers do not import
    # the plotting and statistics only the post-run analysis needs.
    "sequencing_qc",
    # cell → nucleus → pathogen, read as the tree the `cell_id` links in
    # measurements.db already describe. Query-only; it adds no column.
    "lineage",
    "timelapse",
    "tiff_io",
    "deep_spacr",
    "diameter",
    "feature_dict",
    "image_colors",
    "crops",
    "align",
    "convert",
    "foreign",
    "external_masks",
    "resume",
    "checkpoint",
    "normalization",
    "openmp_guard",
    # Plate-wide intensity rescaling provenance and the desktop installer's
    # hardware/consent hand-off are both public, dependency-light modules.
    "intensity_rescale",
    "install_profile",
    "umap_search",
    "cancellation",
    "zstack",
    "report",
    "train_compare",
    "hyperparam",
    "attribution",
    "agreement",
    "active_learning",
    # Correcting a mask and a track by hand, on the record: every edit is
    # journalled so a curated result still says where it came from.
    "curation",
    "plate_qc",
    "seg_qc",
    "model_compare",
    "model_zoo",
    "batch",
    "batch_correction",
    "classifier_evaluation",
    # The confusion matrix as a set of live queries rather than a picture —
    # "which objects are in this cell" is answerable, so a misclassified
    # object can be opened instead of counted.
    "confusion",
    "gui_utils",
    "gui_elements",
    "gui_core",
    "gui",
    "app_annotate",
    "app_make_masks",
    "app_mask",
    "app_measure",
    "app_classify",
    "app_sequencing",
    "app_umap",
    "submodules",
    "ml",
    "predictions",
    "toxo",
    "spacr_cellpose",
    "spacrops",
    "sp_stats",
    "sim",
    "object",
    # The one registry of what object kinds exist. Eleven modules used to
    # spell the vocabulary out independently and now derive from this, so it
    # is imported by nearly everything that touches a mask.
    "object_roles",
    # The organelle presets: one cell-biology choice that fills in the
    # fifty-three organelle settings a user would otherwise have to reason
    # about. `settings`, `settings_spec` and `measure` all import it, so it
    # is part of the surface whether or not it is listed here — and
    # `test_smoke.py::test_lazy_loader_matches_files` is what says so. It
    # was added without this line, which turned every cell of
    # `compat-matrix` red on the same assertion.
    "organelle_types",
    # Image I/O against the two standards a lab is most likely to already be
    # keeping plates in. Both sit behind optional extras, so importing either
    # without its dependency names the `pip install "spacr[...]"` that fixes
    # it rather than raising six frames deep.
    "ome_zarr",
    "omero",
    "cli",
    "cli_database",
    # Whole-installation diagnosis behind `spacr-doctor`.
    "doctor",
    # `spacr-crashreport`: everything a maintainer needs about a failed run
    # in one attachable file, so a bug report is a file rather than a
    # remembered traceback.
    "crashreport",
    "cli_leakage",
    "cli_plugins",
    "cli_remote",
    "cli_repro",
    "_v1_v2_bridge",
    "logger",
    "logging_util",
    "mask_io",
    # A napari-style layer model — images, labels, points and shapes in one
    # world — and manual counting on top of a points layer. Both are plain
    # data models with no Qt in them, so the viewer is a renderer of a state
    # that tests (and notebooks) can build directly.
    "layers",
    "counting",
    # The napari bridge: a field's image and mask out to napari, the
    # corrected labels back, written the way spaCR writes masks and recorded
    # in the same append-only curation ledger the brush uses. napari is an
    # optional extra and is never imported at module scope.
    "napari_bridge",
    # The shared filter/selection model the linked views are built on. Pure
    # pandas, no Qt, so it is usable headless and from a notebook too.
    "selection",
    # Diagnostic figures for a fitted regression.
    "regression_qc",
    # Where each gene's protein lives, for colouring ONE compartment against
    # grey. Pure pandas: the join belongs to the screen, not to the picture.
    "localisation",
    # Everything spaCR knows about a Toxoplasma gene, joined onto an export
    # by gene NUMBER. Next to `localisation` because it is the same join
    # widened from one compartment to the whole annotation, and separate
    # from `toxo` because that module draws figures and this one only reads
    # the five bundled CSVs.
    "annotation",
    # What an effect size is measured FROM, and the sentence that says so.
    # Separate from `figures` because the answer belongs to the fit, not to
    # the picture: the console summary and the exported stats table state the
    # same baseline the panel does.
    "baseline",
    # The cells behind a coefficient: which objects a dot on the volcano is
    # most consistent with. Pure pandas -- the montage is a Qt tab, but WHICH
    # objects to show is a question about the screen, not about a widget.
    "cell_montage",
    # The headless half of the per-plate measurements merge: {plate: db} plus
    # the chosen tables in, one merged frame out, by CALLING multi_database
    # and merge_tables rather than aggregating anything itself.
    "plate_measurements",
    # How wide a coefficient has to be before it counts as a hit. Seven
    # ways of measuring the control spread, in one place so the run and the
    # plot's right-click menu cannot offer different ones.
    "thresholds",
    # Which regression backends exist and which settings each one reads.
    # Pure data; imports NOTHING, which is why it is not part of ml.
    "regression_spec",
    # Building the settings for a second run of the same screen through a
    # different model. No Qt: the GUI offers the gesture, but what a re-fit
    # is allowed to change is a question about the fit, not about a menu.
    "refit",
    # The spaCRPower port: `power_simulate` generates a synthetic pooled
    # screen, `power_model` fits the horseshoe-Poisson hit model to it. They
    # are separate modules because the simulator is cheap and dependency-free
    # while the model pulls in torch, and a parameter sweep re-runs the first
    # far more often than the second.
    "power_simulate",
    "power_model",
    # The ranked, annotated, filterable deliverable of a screen, and the
    # interrogation of the model that ranked it: move one input, watch the
    # prediction move. `hits` is what a collaborator receives; `profiler`
    # is how you decide whether to believe it.
    "hits",
    "profiler",
    "pipeline_v2",
    "plugins",
    "remote_execution",
    # One id, one seed, one error policy, for a whole run. Everything that
    # records anything about a run reads it from here rather than growing a
    # second opinion about which run it is in.
    "runctx",
    "run_journal",
    # Two runs of the same project side by side: what changed in the
    # settings, how many fewer objects, which hits moved.
    "run_compare",
    # The macro recorder: every run also emits the Python script that
    # repeats it — real imports, a real settings dict, a real call — with
    # the run id, the settings hash and a machine-readable record of what
    # was chosen versus defaulted. `run_journal` hooks it; nothing else
    # needs to know it exists.
    "macro",
    "notebook_export",
    # Methods and Results sections written from a run digest, so the prose
    # cannot drift from the settings that produced the numbers.
    "methods_export",
    "custom_features",
    "umap_annotations",
    "row_exclusions",
    "torch_artifacts",
    # The pipeline contract. `ports` declares what each module consumes and
    # produces and answers "can this module run here?" before a run starts;
    # `artifacts` records what produced every file, so "is this result still
    # current?" has an answer. Both are dependency-light on purpose.
    "ports",
    "artifacts",
    # Built directly on that record: `pipeline_graph` is the DAG of what
    # produced what with staleness marked, and `chaining` is the same graph
    # read forwards — a module's inputs default to where the last run
    # *actually* wrote rather than to a path retyped by hand.
    "pipeline_graph",
    "chaining",
    # Disk accounting built on those two: what a project costs per artifact
    # kind, what of it is regenerable and may therefore be pruned, and
    # archiving that leaves the registry knowing where the data went.
    "data_manager",
    # Every project on disk in one list — stage reached, size, last run and
    # what is stale — assembled from `ports`, `artifacts`, `data_manager` and
    # `chaining` rather than re-derived. A project the registry has never
    # seen is listed too, and is reported as unexamined rather than clean.
    "projects",
    "validate",
    "updater",
    "version",
    # The Classify overhaul and the Gate Editor added these and the lazy
    # loader was not told. `test_lazy_loader_matches_files` caught it on
    # every CI platform: a module that exists as a file but is not listed
    # cannot be reached as `spacr.<name>`, so `import spacr; spacr.filters`
    # raised AttributeError while `from spacr import filters` worked --
    # which is the kind of split that gets diagnosed as "sometimes the
    # import fails".
    "classify",
    "classify_classes",
    "crop_source",
    "benchmark",
    "column_groups",
    "filters",
    "gate_library",
    "gpu_reduce",
    "merge_tables",
    "model_check",
    "openmp_guard",
    "surrogate",
    "guide_permutation",
    "hit_attribution",
    "hit_investigation",
    "training_basis",
    # The regression surface, added over 2026-08-15/16. A module missing from
    # this tuple is not reachable as `spacr.<name>` at all -- the lazy loader
    # is the only path -- so leaving one out ships a module nobody outside the
    # package can import, and `test_lazy_loader_matches_files` exists to catch
    # exactly that.
    "multiple_testing",       # every FDR / FWER correction, in one place
    "volcano_style",          # the volcano's thresholds and their rules
    "guide_concordance",      # do a gene's own guides agree in direction
    "regression_diagnostics", # design, residual and inference panels
    "regression_search",      # the dependent-variable search
    "metadata_resolution",    # which metadata column is which
    "multi_database",         # read and merge several measurement databases
    "measurement_scan",       # which measurement has genes with an effect
    "gene_tile",              # everything spaCR knows about one gene
    "parameter_sweep",        # the settings sweep and its containment
    "sweep_child",            # one contained trial, exec'd in its own cgroup
    "trial_metrics",          # what makes a sweep row judgeable
    "figure_style",           # the older per-figure style store
)

__all__ = ["__version__", "download_models", *_SUBMODULES]


def __getattr__(name: str):
    """Lazily import declared submodules and the ``download_models`` helper on first access.

    :param name: Attribute name requested on the ``spacr`` package.
    :returns: Imported submodule or the ``download_models`` callable.
    :raises AttributeError: If ``name`` is neither a known submodule nor ``download_models``.
    """
    if name == "download_models":
        from .utils import download_models
        return download_models

    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy submodule names in ``dir(spacr)`` for tab-completion."""
    return sorted(set(globals()) | {"download_models"} | set(_SUBMODULES))


def _silence_glyph_logging() -> None:
    """Pin fontTools at WARNING as soon as spaCR is imported.

    ``fontTools.subset`` emits about forty INFO lines for every figure saved
    -- each glyph name and glyph ID, twice, for MATH then GSUB then glyf, then
    one line per font table. A regression run saves a dozen figures, so
    thousands of lines of glyph inventory bury the run's own output, and the
    line the user is actually looking for scrolls past unread.

    ``logging_util.QUIET_LOGGERS`` lists it too, but that only applies when
    ``setup_logging()`` runs, it short-circuits on ``_INITIALISED`` if
    something configured logging first, and a script or notebook that never
    calls it gets no protection at all. Doing it at import means importing
    spaCR is sufficient, whatever the startup order.

    This sets a floor, not a lock: anyone who genuinely wants glyph traces can
    lower the level again after importing.
    """
    import logging

    for name in ("fontTools", "fontTools.subset", "fontTools.ttLib"):
        logging.getLogger(name).setLevel(logging.WARNING)


_silence_glyph_logging()
