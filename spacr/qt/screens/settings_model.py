"""
Bridge between spacr's plain-python default settings and Qt form widgets.

The existing spacr GUI expresses settings as `{name: (widget_type, options,
default)}` triples via `spacr.gui_utils.convert_settings_dict_for_gui`.
Here we consume the same conversion output and materialize each entry as
a real Qt widget grouped into logical Section boxes based on
`spacr.settings.categories`.
"""
from __future__ import annotations

import ast
import csv
from contextlib import contextmanager
from functools import partial
from html import escape
import logging
import os
import sys
import textwrap
import weakref
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from PySide6.QtCore import (QEvent, QObject, QPoint, QRect, QSize, Qt,
                            QTimer, Signal)
from PySide6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QLabel,
)

from ..widgets.availability_panel import (AvailabilityPanel,
                                         disable_combo_row,
                                         run_install_offer)
from ..widgets.barcode_regex import BarcodeRegexWidget
from ..widgets.channel_mapping import ChannelMappingWidget
from ..widgets.class_editor import ClassEditorWidget
from ..widgets.database_set import DatabaseSetWidget
from ..widgets.external_mask_inputs import ExternalMaskInputWidget
from ..widgets.file_list import FilePathListWidget, PairedFileTableWidget
from ..widgets.row_exclusion import RowExclusionEditor
from ..widgets.toggle import Toggle
from ...object_roles import ORGANELLE_ROLES, setting_label
# EVERY SLOT A FILE MAY CARRY, not the four the schema segments today. A
# layout that lists all of them costs nothing -- build_sections drops any
# key the module's settings dict does not hold -- and a layout that lists
# four puts slot five in the "Additional Settings" bucket nobody chose.
from ...organelle_types import (ALL_ORGANELLE_ROLES,
                                MAX_ORGANELLES as _MAX_ORGANELLES,
                                organelle_number, organelle_slot_label)
# Pure data, and it imports nothing -- that is the whole point of the module
# (see its docstring). The explainer box below reads it so that a backend
# joining the no-p-value set changes what the box says about correction
# without a second edit here.
from ...regression_spec import NO_P_VALUE_TYPES
# The one separator a spaCR key is built from, so the design scan below
# splits gRNA names the way the pipeline does rather than on a literal '_'.
from ...schema import KEY_SEPARATOR


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings resolvers per app_key
# ---------------------------------------------------------------------------

def timelapse_and_motility_keys() -> set:
    """Every setting key owned by the Timelapse / Motility Assay modules.

    Derived from the category lists in :mod:`spacr.settings` so the two never
    drift apart. Used to strip those keys out of the Mask module's editable
    settings — they still exist in the *pipeline* defaults (spacr.object reads
    ``timelapse`` on every run and ``motility_analysis`` inside the timelapse
    branch), the Mask GUI just no longer offers them.
    """
    from spacr.settings import (
        motility_advanced_settings, motility_settings, timelapse_settings,
    )
    return (set(timelapse_settings) | {"timelapse"}
            | set(motility_settings) | set(motility_advanced_settings))


def _registered_app_metadata(app_key: str) -> Dict[str, Any]:
    """One app's :data:`spacr.qt.app.APP_META` entry, or ``{}``.

    Read out of :data:`sys.modules`, never imported: ``spacr.qt.app``
    builds the screens that build this model, so importing it from here
    would be a cycle, and a process that has not loaded the registry
    simply has no registered apps to ask about.
    """
    app = sys.modules.get("spacr.qt.app")
    return (getattr(app, "APP_META", {}).get(app_key) or {}) if app else {}


#: Folded modules' defaults modules — app key → the module that calls
#: :func:`spacr.settings.register_defaults` for it.
#:
#: A module with a registry row names this through ``register_app(...,
#: defaults_module=...)``, and that is still the seam a new module should
#: use. A module that has been FOLDED into another one has no row left to
#: name it from, and nothing in a fresh window imports it — so
#: :func:`resolve_default_settings` would find no registered defaults and
#: fall through to the bare ``{"src": "path"}`` placeholder, i.e. the
#: folded page would open on an empty form with a Run button that has
#: nothing to run.
#:
#: Consulted only when the registry has no answer, so a module that still
#: has a row is served by its own registration exactly as before.
_FOLDED_DEFAULTS_MODULES: Dict[str, str] = {
    "barcode_qc": "spacr.sequencing_qc",
    "explain_cv": "spacr.surrogate",
    "anndata_export": "spacr.anndata_export",
}


def _import_registered_defaults_module(app_key: str) -> None:
    """Import the module that registers ``app_key``'s settings defaults.

    Named by ``register_app(..., defaults_module=...)`` while the app has
    a row, and by :data:`_FOLDED_DEFAULTS_MODULES` once it has been folded
    into another module and the row is gone. Failure is logged and
    swallowed: an unimportable optional dependency should cost that app
    its settings panel, not stop the window opening.
    """
    module = (_registered_app_metadata(app_key).get("defaults_module")
              or _FOLDED_DEFAULTS_MODULES.get(app_key))
    if not module or module in sys.modules:
        return
    import importlib
    try:
        importlib.import_module(module)
    except Exception:
        LOGGER.warning("Could not import %s, which owns the %r settings",
                       module, app_key, exc_info=True)


def resolve_default_settings(app_key: str) -> Dict[str, Any]:
    """Return a fresh defaults dict for an app key, mirroring the Tk GUI
    dispatch in gui_core.setup_settings_panel."""
    try:
        from spacr.plugins import get_app, load_object
        plugin_app = get_app(app_key)
    except Exception:
        plugin_app = None
    if plugin_app is not None:
        defaults = load_object(plugin_app.defaults)
        if not callable(defaults):
            raise TypeError(f"Plugin defaults {plugin_app.defaults!r} are not callable")
        try:
            result = defaults({})
        except TypeError:
            result = defaults()
        if not isinstance(result, dict):
            raise TypeError(
                f"Plugin defaults {plugin_app.defaults!r} returned "
                f"{type(result).__name__}, expected dict"
            )
        return dict(result)
    # Modules that shipped their own defaults through the `register_defaults`
    # seam. Consulted after plugins and before the built-in dispatch below, so
    # a registered module is served without editing this function -- which is
    # the whole point of the seam, and without this line every
    # `register_defaults` call in the codebase is inert.
    #
    # Import first, ask second. `register_defaults` runs at the module's own
    # import, so the seam only answers for a module something has already
    # imported -- and a pipeline module has no reason to be imported by the
    # process that is merely drawing its settings panel. `register_app(...,
    # defaults_module=...)` names it; this is what makes the panel appear
    # instead of an empty form.
    _import_registered_defaults_module(app_key)
    from spacr.settings import defaults_for, has_registered_defaults
    if has_registered_defaults(app_key):
        return defaults_for(app_key, {})
    from spacr.settings import (
        get_identify_masks_finetune_default_settings,
        set_default_analyze_screen,
        set_default_settings_preprocess_generate_masks,
        get_automated_motility_assay_default_settings,
        get_measure_crop_settings,
        deep_spacr_defaults,
        set_default_generate_barecode_mapping,
        set_default_umap_image_settings,
        get_analyze_recruitment_default_settings,
        get_check_cellpose_models_default_settings,
        get_analyze_plaque_settings,
        set_analyze_invasion_defaults,
        get_perform_regression_default_settings,
        get_train_cellpose_default_settings,
        get_default_generate_activation_map_settings,
        get_timelapse_settings,
        set_analyze_replication_defaults,
    )
    if app_key == "mask":
        # Timelapse tracking and the automated motility assay are first-class
        # modules of their own now (app keys 'timelapse' / 'motility'), so the
        # Mask module edits neither set of knobs. The keys are dropped from the
        # *editable* dict only — preprocess_generate_masks re-applies
        # set_default_settings_preprocess_generate_masks internally, so a Mask
        # run still gets timelapse=False / motility_analysis=False, and a CSV
        # driven straight through the API keeps working unchanged.
        s = set_default_settings_preprocess_generate_masks(settings={})
        for key in timelapse_and_motility_keys():
            s.pop(key, None)
        return s
    if app_key == "timelapse":
        s = get_timelapse_settings(settings={})
        # The Timelapse module tracks objects; running the assay is what the
        # Motility Assay module is for, so its inline gate isn't offered here.
        s.pop("motility_analysis", None)
        # `timelapse` stays in the dict and stays True. It is not rendered
        # as a control -- see the layout -- because this module is the
        # timelapse one and a user turning it off here would be left with a
        # screen of controls about a time dimension it was told to ignore.
        # Forced rather than merely defaulted, so a settings CSV saved by
        # an older build with `timelapse: False` cannot silently turn this
        # module into a slower Mask Generation.
        s["timelapse"] = True
        return s
    if app_key == "motility":
        s = get_automated_motility_assay_default_settings(settings={})
        # `motility_analysis` is the Mask-pipeline gate for the inline assay
        # (spacr.object), not a knob of the assay itself — opening the
        # Motility module *is* asking for the assay.
        s.pop("motility_analysis", None)
        return s
    if app_key == "measure":
        return get_measure_crop_settings(settings={})
    if app_key == "external_masks":
        from spacr.external_masks import default_settings
        return default_settings({})
    if app_key == "classify_merged":
        from spacr.settings import set_default_classify
        settings = set_default_classify(settings={})
        settings["src"] = []
        return settings
    if app_key == "classify":
        settings = deep_spacr_defaults(settings={})
        settings["src"] = []
        return settings
    if app_key == "umap":
        settings = set_default_umap_image_settings(settings={})
        # The original controls describe one lab's c1/c2/c3 plate convention.
        # Keep them as API-compatible backend defaults, but do not expose them
        # in the general UMAP UI. ``exclude_rows`` replaces them with rules
        # based on the columns and values in the user's own database.
        for key in (
            "col_to_compare", "pos", "neg", "mix",
            "embedding_by_controls", "exclude_conditions",
        ):
            settings.pop(key, None)
        return settings
    if app_key == "train_cellpose":
        return get_train_cellpose_default_settings(settings={})
    if app_key == "ml_analyze":
        return set_default_analyze_screen(settings={})
    if app_key == "cellpose_masks":
        return get_identify_masks_finetune_default_settings(settings={})
    if app_key == "cellpose_all":
        return get_check_cellpose_models_default_settings(settings={})
    if app_key == "map_barcodes":
        return set_default_generate_barecode_mapping(settings={})
    if app_key == "regression":
        return get_perform_regression_default_settings(settings={})
    if app_key == "recruitment":
        return get_analyze_recruitment_default_settings(settings={})
    if app_key == "activation":
        return get_default_generate_activation_map_settings(settings={})
    if app_key == "invasion":
        return set_analyze_invasion_defaults(settings={})
    if app_key == "replication":
        return set_analyze_replication_defaults(settings={})
    if app_key == "analyze_plaques":
        return get_analyze_plaque_settings(settings={})
    if app_key in ("annotate", "make_masks"):
        # These are interactive apps; return minimal placeholder.
        return {"src": "path to images"}
    return {"src": "path"}


# Per-app category suppression. Keys not in a shown category fall into the
# trailing "Other" section, so the setting stays reachable — only the tab goes.
#: Settings a module keeps but never shows.
#:
#: Not the same as dropping the key. A dropped key is absent from the run's
#: settings, which means the pipeline falls back to ITS default and the two
#: can disagree. These stay in the dict, at the value the module needs, and
#: are simply not rendered.
#:
#: Removing a key from an app's layout is not enough to hide it: anything a
#: layout does not place lands in "Additional Settings", which is the bucket
#: the layouts exist to keep empty. This is the mechanism that actually
#: hides one.
_APP_HIDDEN_KEYS: Dict[str, set] = {
    # This module IS the timelapse one. A user who turned this off would be
    # left looking at a screen whose every remaining control is about a time
    # dimension it had just been told to ignore -- and Mask Generation is
    # right there for that. `resolve_default_settings` forces it True.
    "timelapse": {"timelapse"},
    # `png_type` was one of two names for a path filter, and the one that
    # pretended to name a file type. The Classify overhaul replaced it with
    # `path_string` (the substring) and `file_type` (an actual extension).
    #
    # It stays in the settings dict because `spacr.crop_source` still reads
    # it as a fallback, so a settings CSV written before the split keeps
    # working. It is not OFFERED, because offering both halves of a
    # superseded pair is how a user sets one and wonders why the other
    # wins.
    # AND THE SAME RULE FOR THE 230 SUPERSESSIONS. `crop_source`,
    # `file_metadata` and `file_type` are what `image_source` and
    # `load_path_regex` replaced, and `coordinate_columns` is DERIVED from
    # `object_array` -- so none of the four is a control any more.
    #
    # They stay in the settings dict because the old readers still consult
    # them as a fallback, which is what keeps a settings CSV written before
    # the rename working. They are not OFFERED, for the reason `png_type`
    # is not: offering both halves of a superseded pair is how a user sets
    # one and wonders why the other wins.
    "classify": {
        "png_type", "crop_source", "file_metadata", "file_type",
        "path_string", "extract_channels", "coordinate_columns",
        "class_metadata", "annotation_column",
        # AND THE FOLDER NAMES (instruction 229, reported again 2026-08-21:
        # "i asked you to remove class folder names and just use the classes
        # given in the classes setting"). The first pass only made the class
        # field OUTRANK it, which left a control on screen that could
        # disagree with the classes above it and lose -- a control the user
        # can change that changes nothing.
        #
        # It stays in the settings dict because dataset generation WRITES
        # it: it records what actually went to disk, which is a different
        # fact from what the user asked for.
        "class_folder_names",
    },
    "classify_merged": {
        "png_type", "crop_source", "file_metadata", "file_type",
        "path_string", "extract_channels", "coordinate_columns",
        "class_metadata", "annotation_column",
        # AND THE FOLDER NAMES (instruction 229, reported again 2026-08-21:
        # "i asked you to remove class folder names and just use the classes
        # given in the classes setting"). The first pass only made the class
        # field OUTRANK it, which left a control on screen that could
        # disagree with the classes above it and lose -- a control the user
        # can change that changes nothing.
        #
        # It stays in the settings dict because dataset generation WRITES
        # it: it records what actually went to disk, which is a different
        # fact from what the user asked for.
        "class_folder_names",
    },
    # One action-strip GPU toggle drives both the main reducer and the search.
    # The setting remains in _defaults and therefore in collect(); only the
    # duplicate form control is hidden.
    # `crop_source` reaches UMAP through the shared picture settings and is
    # superseded there by `image_source` for the same reason as above.
    "umap": {"gpu", "crop_source"},
    # WHAT "REGRESSION PLOTS" AND "RUNTIME & RELIABILITY" HELD, per
    # instruction 135. The sections are deleted from the layout above; these
    # keys keep their values and reach the run exactly as before, they are
    # simply not asked about.
    #
    # Hidden and not dropped, deliberately, and each for its own reason:
    #
    #   regression_qc          `parameter_sweep` sets it False so a
    #                          hundred-trial sweep does not pay ~5.8 s and
    #                          ~19 figures per trial. Drop the key and the
    #                          sweep has nothing to set. One analysis still
    #                          gets the suite, because the default is True.
    #   guide_permutation_plot hard True: the permutation run's only picture.
    #   log_x, log_y           hard False.
    #   x_lim, y_lims,         set ON the plot, where the axes being changed
    #   split_axis_lims        are visible; a number typed before the figure
    #                          exists is a guess.
    #   strict_errors,         how the APPLICATION behaves on a failure, not
    #   max_failure_rate,      how this regression is fitted. The same answer
    #   verbose, random_seed,  on every module, so it is one answer in
    #   on_error*              Preferences rather than eleven in the modules.
    #
    # Keys this module does not declare (`on_error`, `random_seed`, ...) are
    # named anyway: hiding a key that is not there costs nothing, and the day
    # a shared runtime default reaches this module it must not appear on the
    # panel the instruction just cleared.
    "regression": {
        "regression_qc", "guide_permutation_plot",
        "log_x", "log_y", "x_lim", "y_lims", "split_axis_lims",
        "strict_errors", "max_failure_rate", "on_error",
        "on_error_attempts", "on_error_backoff", "random_seed", "verbose",
        # Regression derives this aggregate from the positive, negative, and
        # mixed control-well settings, so it is not an independent GUI choice.
        # The invasion assay retains its separate control because there it
        # identifies wells without pre-permeabilisation stain.
        "control_wells",
        # SUPERSEDED BY `annotation_source`, and hidden here rather than in
        # a second "regression" entry further up this dict -- which is where
        # it was, and which a later key of the same name silently replaced.
        # A dict literal keeps the last value, so `Toxoplasma` was declared
        # hidden and then offered anyway, ungrouped, in the bucket the
        # layouts exist to keep empty.
        "Toxoplasma",
        # THE FOUR OBJECT-OUTLIER FILTERS. Each excludes objects by a
        # robust z-score over a measurement, and this module joins the
        # measurements to the scores AFTER the fit -- so at the moment they
        # would run there is no column to take a deviation over. They are
        # still read from a settings file; they are not offered.
        "cell_area_outlier_mads", "nucleus_area_outlier_mads",
        "cell_intensity_outlier_mads", "nucleus_intensity_outlier_mads",
    },
}

_APP_HIDDEN_CATEGORIES: Dict[str, set] = {
    "classify": {"Cellpose"},
    # Mask no longer owns tracking or the motility assay — those are the
    # 'timelapse' and 'motility' modules. resolve_default_settings already
    # drops the keys so nothing spills into "Other"; this entry is the
    # declaration of intent and keeps the tabs gone even if a future default
    # re-introduces one of the keys.
    "mask": {"Timelapse", "Motility (beta)", "Motility Advanced (beta)"},
    # The Timelapse module tracks objects; the motility assay is its own
    # module and its ~50 knobs would swamp the tracking settings.
    "timelapse": {"Motility (beta)", "Motility Advanced (beta)"},
}

# ---------------------------------------------------------------------------
# A setting is visible when its object is in the run
# ---------------------------------------------------------------------------
#
# WHY THIS IS A THIRD MECHANISM AND NOT ONE OF THE TWO ABOVE.
#
#   * ``spacr.settings.setting_dependencies`` GREYS a control and writes the
#     reason beside it. That is right for a setting the run is about to
#     decide for itself -- one row among a handful, where the note is the
#     point. It is wrong here: an object a run does not segment takes forty
#     rows with it, and forty greyed rows are not an explanation, they are
#     the wall this exists to remove.
#   * ``_APP_HIDDEN_KEYS`` builds no widget at all. It is decided once, per
#     MODULE, before any value exists, and it is not reversible: with no
#     widget the value falls back to ``_defaults``, so everything the user
#     typed into a row is gone the moment the row is hidden. "Changing a
#     channel back must bring the old answers back with it" is exactly the
#     thing that mechanism cannot do.
#
# So the widget is built and kept in ``_widgets``, and its ROW is hidden.
# HIDDEN, NOT DELETED: ``collect()`` walks ``_widgets``, so a hidden setting
# is still read from its own widget, still carries what the user last typed
# into it, and is still written to the settings file. A settings CSV cannot
# lose a key because the panel was not showing it when Save was pressed.

#: The keys that say whether an object is in the run at all.
#:
#: Both spellings, because which one a module offers depends on what the
#: module does: a module that SEGMENTS asks for a channel to segment it in
#: (``cell_channel``), and one that reads masks somebody else made asks which
#: plane holds them (``cell_mask_dim`` -- Measure offers no ``cell_channel``
#: at all, so a rule that knew only about channels would gate nothing there).
#: ``spacr.settings.category_integer_dependencies`` already declares exactly
#: this pair for cell, nucleus and pathogen; this is the same switch read per
#: SETTING rather than per category, because an organelle slot is not a
#: category -- four of them share two.
OBJECT_SWITCH_SUFFIXES: Tuple[str, ...] = ("channel", "mask_dim")

#: The objects that have a channel and are not organelle slots.
#:
#: ``cytoplasm`` is deliberately absent: it is DERIVED from the cell mask
#: minus everything found inside it, so it has no channel, no diameter and no
#: detection method, and there is nothing to switch it with. See
#: ``spacr.object_roles``.
CHANNELLED_OBJECTS: Tuple[str, ...] = ("cell", "nucleus", "pathogen")

#: Which of a slot's detection settings each ``organelle_morphology`` reads.
#:
#: Read off ``spacr.object``, which is the authority: ``_segment_spots``,
#: ``_segment_network``, ``_segment_irregular`` and ``_segment_ring``, plus
#: the methods ``_validate_organelle_settings`` accepts for each morphology.
#: An entry is the union over that morphology's LEGAL METHODS rather than
#: over the one method currently chosen: the method is a separate choice, and
#: a spots slot that will be switched to ``log`` tomorrow needs its sigmas on
#: screen today.
#:
#: A suffix in NO entry is never hidden by a morphology, and that is most of
#: them. ``adaptive_block_size`` is one -- ``'adaptive'`` is legal under all
#: four morphologies, so a block size applies whatever the slot is -- and so
#: is everything cellpose reads, for the same reason. ``morph_radius`` is in
#: TWO entries, because it is irregular's closing radius and also the closing
#: radius of network's otsu/adaptive path, which is why this is a membership
#: table and not a partition.
_MORPHOLOGY_SETTINGS: Dict[str, frozenset] = {
    "spots": frozenset({
        "tophat_radius", "watershed_spots",
        "log_min_sigma", "log_max_sigma", "log_num_sigma", "log_threshold",
        "dog_sigma_low", "dog_sigma_high",
    }),
    "network": frozenset({
        "ridge_filter", "ridge_sigmas", "network_threshold",
        "hysteresis_low", "hysteresis_high", "skeletonize",
        "morph_radius", "unet_model_path", "unet_threshold",
    }),
    "irregular": frozenset({"morph_radius", "fill_holes"}),
    "ring": frozenset({
        "ring_sigma_inner", "ring_sigma_outer", "ring_min_prominence",
        "ring_fill_method",
        # Ring accepts 'log' and reads the LoG sigmas when it is chosen. It
        # does NOT read the DoG pair: its 'dog' path band-passes with
        # `ring_sigma_inner`/`_outer` instead. See `_segment_ring`.
        "log_min_sigma", "log_max_sigma", "log_num_sigma", "log_threshold",
    }),
}

#: Every suffix some morphology claims. A slot setting outside this set is
#: shown whenever its slot is, whatever the slot is typed as.
_MORPHOLOGY_OWNED: frozenset = frozenset().union(
    *_MORPHOLOGY_SETTINGS.values())


#: The signals a settings widget announces a change on, most specific first.
#: ONE of them is connected, not all: a QComboBox emits both
#: `currentIndexChanged` and `currentTextChanged` for a single choice, so
#: connecting every signal a widget has would run the handler twice per edit.
_VALUE_CHANGED_SIGNALS: Tuple[str, ...] = (
    'value_changed', 'currentTextChanged', 'currentIndexChanged',
    'textChanged', 'valueChanged', 'toggled', 'stateChanged',
)


def _connect_value_changed(widget, handler) -> bool:
    """Connect ``handler`` to the first change signal ``widget`` has.

    :returns: whether a signal was found. A widget with none of them cannot
        announce an edit, and a rule that follows it will only be re-read
        when something else on the panel moves.
    """
    for name in _VALUE_CHANGED_SIGNALS:
        signal = getattr(widget, name, None)
        if signal is not None:
            signal.connect(handler)
            return True
    return False


def _names_a_plane(value: Any) -> bool:
    """True when a channel or mask-dim setting names a plane of the stack.

    ``False`` is not a plane. A boolean reaches here only from a settings
    file that put one in a channel, and ``int(False)`` would read it as plane
    zero -- which would switch an object on because someone wrote "no".
    """
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    text = str(value).strip()
    if not text or text.lower() == "none":
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


#: How many organelle slots a panel builds controls for.
#:
#: Every slot that can be named, because a count the panel cannot render is
#: a count that does nothing -- which is the defect this number exists to
#: close. What it costs is measurable and worth writing down: the Mask panel
#: renders 54 settings per slot, so at twenty-six it builds about 1,500
#: controls instead of 350 and takes a few seconds to open the first time in
#: a session rather than well under one. Measure pays almost nothing, because
#: a slot is three settings there.
#:
#: Lowering this is the one-line trade: the panel opens faster and a count
#: above it becomes inert again for the slots it cannot draw. It is a
#: PANEL number and nothing else reads it -- the run, the settings file and
#: the registries are all bounded by
#: :data:`spacr.organelle_types.MAX_ORGANELLES`, which is where the slot
#: names actually run out.
PANEL_ORGANELLE_SLOTS: int = _MAX_ORGANELLES


#: The key endings that name ONE PLANE of the stack: the raw acquisition
#: channel an object is imaged in, the channel paired with its mask when an
#: overlay is drawn, and the plane its label mask sits on in the merged array.
PLANE_SUFFIXES: Tuple[str, ...] = ("_channel", "_mask_dim", "_chann_dim")


def _is_clearable_plane_setting(key: str) -> bool:
    """True when ``key`` names a plane and is declared to accept None.

    BOTH HALVES MATTER. The suffix says the value is a plane index, and the
    declaration in ``spacr.settings.expected_types`` says whether the object
    it belongs to may be absent. ``outside_channel`` ends in ``_channel`` and
    is declared ``int`` alone -- the invasion assay thresholds on it and has
    no reading without it -- so it keeps its spin box.
    """
    if not str(key).endswith(PLANE_SUFFIXES):
        return False
    try:
        from ... import settings as _settings

        declared = _settings.expected_types.get(str(key))
    except Exception:                                        # noqa: BLE001
        return False
    if declared is None:
        return False
    allowed = declared if isinstance(declared, tuple) else (declared,)
    return type(None) in allowed


def object_switch_keys(role: str) -> Tuple[str, ...]:
    """The keys that decide whether ``role`` is in the run."""
    return tuple(f"{role}_{suffix}" for suffix in OBJECT_SWITCH_SUFFIXES)


def object_of_setting(key: str) -> Optional[str]:
    """Which object a setting belongs to, or None for the great majority.

    Organelle slots are resolved by :mod:`spacr.organelle_types`, which owns
    the slot naming: the prefixes are lettered -- ``organelle``,
    ``organelleb``, ... -- and ``organelle`` is a prefix of every other one,
    so the match has to be longest-first and belongs where the names are
    generated rather than being written out a second time here.

    Both spellings of the other three are understood, ``cell_min_size`` and
    ``remove_background_cell``, the way
    ``spacr.settings.advanced_object_of`` understands them: spaCR is not
    consistent about which end of a key the object name goes on, and a rule
    that knew only one end would leave half a family on screen.
    """
    from ...organelle_types import organelle_role_of

    text = str(key)
    role = organelle_role_of(text)
    if role is not None:
        return role
    for obj in CHANNELLED_OBJECTS:
        if text.startswith(f"{obj}_") or text.endswith(f"_{obj}"):
            return obj
    return None


def organelle_morphology_now(role: str,
                             settings: Dict[str, Any]) -> Optional[str]:
    """The morphology a slot is in, as far as the panel can tell.

    THE TYPE DECIDES, and the type alone is not always enough: ``vesicular``
    and ``spherical`` split on SIZE -- a 200 nm transport vesicle is a dot
    and a 2 um vacuole is a ring, and both are Vesicular -- so the slot's
    diameter is read through the same ``morphology_for`` the preset table
    exposes. :mod:`spacr.organelle_types` explains at length why the mapping
    is ``(type, size) -> morphology`` rather than ``type -> morphology``.

    ``custom`` recommends nothing by design, so there the slot's own
    ``<role>_morphology`` is the answer -- which is also what a settings file
    written before the types existed carries.

    :returns: one of the four morphologies, or None when neither the type nor
        the morphology names one, which narrows nothing rather than guessing.
    """
    from ...organelle_types import resolve_type

    try:
        preset = resolve_type(settings.get(f"{role}_type"))
    except ValueError:
        preset = None
    if preset is not None:
        diameter = settings.get(f"{role}_diameter")
        try:
            diameter = None if diameter is None else float(diameter)
        except (TypeError, ValueError):
            diameter = None
        morphology = preset.morphology_for(diameter)
        if morphology in _MORPHOLOGY_SETTINGS:
            return morphology
    own = settings.get(f"{role}_morphology")
    return own if own in _MORPHOLOGY_SETTINGS else None


def keys_hidden_by_their_object(keys, settings: Dict[str, Any]) -> set:
    """Which of ``keys`` must not be on the form, because they do not apply.

    Three reasons, in the order they are decided:

      * the slot is beyond ``number_of_organelles`` -- and that takes the
        slot's channel with it, because a slot the run does not have is not a
        slot with its channel left showing;
      * the object's channel (or its mask plane) names no plane, so the run
        does not have that object at all;
      * the slot's type puts it in one morphology and the setting belongs to
        a different one -- a punctate organelle has no ridge filter.

    :param keys: every setting this panel has a control for. WHAT THE PANEL
        HOLDS IS WHAT DECIDES WHAT MAY BE HIDDEN: a role is gated only when
        its switch is on the panel too, and a slot is gated by the count only
        when the count is. Hiding a row whose switch lives on another screen
        would leave the user a control they cannot bring back --
        ``_rules_for_this_panel`` refuses to grey one for the same reason.
    :param settings: the panel's current values. Only the switches, the
        count and the slots' type, diameter and morphology are read.
    :returns: the keys whose rows are to be hidden.
    """
    from ...organelle_types import (NUMBER_OF_ORGANELLES,
                                    active_organelle_roles)

    on_panel = {str(key) for key in keys}
    counted = NUMBER_OF_ORGANELLES in on_panel
    active = active_organelle_roles(settings) if counted else ()
    hidden = set()
    for key in on_panel:
        role = object_of_setting(key)
        if role is None:
            continue
        is_slot = role not in CHANNELLED_OBJECTS
        if counted and is_slot and role not in active:
            hidden.add(key)
            continue
        switches = [k for k in object_switch_keys(role) if k in on_panel]
        if not switches or key in switches:
            continue
        if not any(_names_a_plane(settings.get(k)) for k in switches):
            hidden.add(key)
            continue
        if not is_slot:
            continue
        morphology = organelle_morphology_now(role, settings)
        if morphology is None:
            continue
        suffix = key[len(role) + 1:]
        if (suffix in _MORPHOLOGY_OWNED
                and suffix not in _MORPHOLOGY_SETTINGS[morphology]):
            hidden.add(key)
    return hidden



def section_shows_anything(section) -> bool:
    """Whether a settings heading still has something under it.

    A HEADING WITH EVERY ROW HIDDEN IS A SMALLER WALL, BUT IT IS STILL A
    WALL. A default Mask panel leaves twenty-three of them -- "Organelle 3",
    "Pathogen segmentation" -- and a heading that opens onto nothing tells
    the user only that they have found the wrong screen.

    Public because the SCREEN has to be the one to act on it: a section's
    visibility is decided in exactly one place (``AppScreen.
    refresh_maturity_visibility``), and two things calling ``setVisible`` on
    the same card is how a card comes back the next time Preferences is
    saved. This answers the question; it does not hide anything.

    :param section: a :class:`spacr.qt.widgets.section.Section`.
    :returns: True unless the heading owns setting rows or nested headings
        and every one of them is hidden. A heading that owns neither -- a
        prose panel, an explainer -- is never judged empty, because it was
        never carrying rows to lose.
    """
    from ..widgets.section import Section

    form = getattr(section, "_form", None)
    if not isinstance(form, QFormLayout):
        return True
    own_rows = 0
    for index in range(form.rowCount()):
        item = form.itemAt(index, QFormLayout.FieldRole)
        if item is None or item.widget() is None:
            continue
        own_rows += 1
        if form.isRowVisible(index):
            return True
    children = [child for child in section.findChildren(Section)
                if child is not section]
    if any(section_shows_anything(child) for child in children):
        return True
    return not own_rows and not children


#: The batch-correction alphabet, offered identically by every screen that
#: shows the setting.
#:
#: It is one named tuple rather than a literal repeated per app because the
#: fourth copy was the one that never got written: Classify (merged) resolves
#: its defaults through ``set_default_classify``, which sets all eight
#: ``batch_*`` keys, but ``_APP_COMBO_OPTIONS['classify_merged']`` listed
#: neither this nor ``batch_missing_control``. Both were free-text boxes on
#: that screen alone, and a typo in one reached
#: ``batch_correction.correct_batch_effects`` as
#: ``ValueError: Unknown batch_correction='zcore'`` at run time, after the
#: user had walked away — the same failure the ``classifier_family`` alphabet
#: right below exists to prevent.
#:
#: ``combat`` is last because it is the only one that needs an answer from
#: the user first: without ``batch_covariate_column`` it refuses to run
#: rather than deleting the contrast the screen is measuring. See
#: ``spacr.batch_correction._combat``.
_BATCH_CORRECTION_OPTIONS = [
    "none", "control_center", "robust_zscore", "center", "zscore", "combat",
]

#: What ``control_center`` does on a plate with too few reference controls.
_BATCH_MISSING_CONTROL_OPTIONS = ["error", "skip"]

#: Crop-source choices shown by the settings panel. ``Load images`` is first
#: because it is the default and reads existing crops from ``data/``.
#:
#: The stored values stay 'png' and 'merged' -- `spacr.crops` reads those,
#: and no settings file written before this changes meaning.
_CROP_SOURCE_OPTIONS = [
    ("png", "load images — crops already in data/"),
    ("merged", "stream images — cut from merged/"),
]

# Options that are enumerations for one module but not necessarily for every
# setting with the same generic key.  Keeping these app-scoped avoids turning
# unrelated ``mode`` fields into sequencing controls.
_APP_COMBO_OPTIONS: Dict[str, Dict[str, List[Any]]] = {
    "umap": {
        "reduction_method": ["umap", "tsne", "pca", "isomap", "spectral"],
        # Replaced with the installed UMAP metric inventory in _widget_for.
        "metric": ["euclidean"],
        "pca_svd_solver": [
            "auto", "full", "covariance_eigh", "arpack", "randomized",
        ],
        "isomap_path_method": ["auto", "FW", "D"],
        "spectral_affinity": ["nearest_neighbors", "rbf"],
        "clustering": ["dbscan", "kmeans"],
        # 'auto' is retired FROM THE PANEL and not from the code
        # (instruction 171): it answers "what is available here", which is not
        # an answer to somebody asked which mode they want.
        "crop_source": _CROP_SOURCE_OPTIONS,
        "batch_correction": _BATCH_CORRECTION_OPTIONS,
        "batch_missing_control": _BATCH_MISSING_CONTROL_OPTIONS,
    },
    "annotate": {
        # The choice the annotation app has always had a SETTING for and
        # never offered -- it shipped 'auto' and took the PNG folder whenever
        # one existed. "in the annotation app how do i choose to stream images
        # from database or dataset" (2026-08-19).
        "crop_source": _CROP_SOURCE_OPTIONS,
    },
    "ml_analyze": {
        "batch_correction": _BATCH_CORRECTION_OPTIONS,
        "batch_missing_control": _BATCH_MISSING_CONTROL_OPTIONS,
    },
    "regression": {
        "batch_correction": _BATCH_CORRECTION_OPTIONS,
        "batch_missing_control": _BATCH_MISSING_CONTROL_OPTIONS,
        # Filled from the modules that own each inventory in _widget_for, so
        # a family added to spacr.ml or a correction added to
        # spacr.multiple_testing appears here without a second edit.
        "regression_type": ["ols"],
        "multiple_testing_method": ["fdr_bh"],
        "inference": ["auto", "parametric", "nonparametric"],
        # Instruction 134, asked for on 2026-08-17: "analasys mode should be
        # a dropdown". Two valid values and it was a FREE-TEXT box, so a typo
        # in it survived until the run had read the whole database.
        # `_resolve_regression_analysis_choices` is what maps `inference` onto
        # this, and it accepts exactly these two.
        #
        # (value, label) PAIRS: the key is called 'guide_permutation' and the
        # dropdown says what that IS, the same way 132's model box explains
        # what it fits. The stored values are unchanged, so every settings
        # file already written goes on meaning what it meant.
        "analysis_mode": [
            ("regression",
             "regression — fit every guide at once in the chosen model"),
            ("guide_permutation",
             "guide permutation — test each guide on its own, plate-blocked"),
        ],
        "analysis_unit": ["well", "cell"],
        # Exactly the branches process_scores implements; anything else
        # reaches the pipeline and is silently ignored rather than applied.
        "agg_type": ["mean", "median", "quantile", None],
        "transform": [None, "log", "sqrt", "square", "beta"],
        # A link-like transform plus a non-identity family link would transform
        # the response twice. The first two choices select one scale; the
        # legacy choice exists only to reproduce an earlier run.
        "glm_transform_conflict": [
            ("untransformed",
             "fit the response as measured — let the family's link transform it"),
            ("transformed",
             "keep my transform — fit Gaussian with an identity link"),
            ("warn",
             "legacy behavior — reproduce an earlier fit and show a warning"),
        ],
        "cov_type": [None, "HC0", "HC1", "HC2", "HC3"],
        "threshold_method": ["std", "var"],
        # WHICH P THE SIGNIFICANCE LINE IS DRAWN ON. Two values and no third
        # reading, so it is a closed alphabet rather than a box a user can
        # type "adj" into. "adjusted" leads because it is the only one of the
        # two that is evidence with hundreds of guides in the family.
        "p_threshold_kind": ["adjusted", "raw"],
    },
    "classify": {
        "evaluation_calibration": ["temperature", "none"],
    },
    "classify_merged": {
        "evaluation_calibration": ["temperature", "none"],
        # A closed alphabet: there are two families and a typo in a free-text
        # box would raise ClassifierFamilyError at run time, after the user
        # had walked away.
        "classifier_family": ["cv", "ml"],
        # set_default_classify gives this screen all eight batch_* keys, so
        # it corrects batches exactly like the other three — but it was the
        # one app that listed no alphabet for them.
        "batch_correction": _BATCH_CORRECTION_OPTIONS,
        "batch_missing_control": _BATCH_MISSING_CONTROL_OPTIONS,
    },
    "external_masks": {
        "layout": ["auto", "flat", "well", "plate_well"],
        "z_handling": ["max", "first"],
        "plate_naming": ["index", "name"],
    },
    "map_barcodes": {
        "mode": ["paired", "single"],
        "single_direction": ["R1", "R2"],
        "comp_type": ["zlib", "lzo", "bzip2", "blosc"],
    },
    "explain_cv": {
        "surrogate_model": [
            "random_forest", "hist_gradient_boosting", "xgboost",
        ],
        "surrogate_split_by": ["well", "plate"],
    },
    "investigate_hit": {
        "hit_direction": ["positive", "negative"],
        "hit_split_by": ["auto", "plate", "well"],
    },
}


class _CsvColumnSource(NamedTuple):
    """Where a column-name setting's candidate names come from."""

    #: Which side of the paired input table holds the CSVs to read --
    #: ``score``, ``count``, or both. `dependent_variable` is a column of the
    #: score CSV and of nothing else; `filter_column` is applied to BOTH
    #: (`ml.clean_controls` on the scores, `ml.process_reads` on the counts),
    #: so offering only one side would hide half the answer.
    roles: Tuple[str, ...]
    #: What kind of column, for the message. It reads "no response column
    #: dependent_variable='pred' in ..." rather than "no column ...".
    what: str


#: Settings whose value NAMES A COLUMN OF AN INPUT CSV, per module.
#:
#: Regression column pickers read the score and count CSV headers rather than
#: a ``measurements.db`` file, because those CSVs are the inputs against which
#: the selected names are validated.
#:
#: The reading is `spacr.columns`, which takes the HEADER ROW ONLY
#: (`nrows=0`). This runs on the GUI thread against score CSVs that are
#: hundreds of megabytes, and there is no second reader here for that reason.
CSV_COLUMN_SOURCES: Dict[str, Dict[str, _CsvColumnSource]] = {
    "regression": {
        "dependent_variable": _CsvColumnSource(("score",), "response column"),
        "filter_column": _CsvColumnSource(("score", "count"),
                                          "filter column"),
        # The count table's own header names, which were HARD-CODED and had
        # no setting at all until instruction 135. They fail the same way
        # `dependent_variable` did -- inside the merge, naming a column the
        # file has not got -- and they earn the same button.
        "count_grna_column": _CsvColumnSource(("count",), "count column"),
        "count_value_column": _CsvColumnSource(("count",), "count column"),
    },
}


def has_csv_column_picker(app_key: str, key: str) -> bool:
    """True when this module gives ``key`` a CSV picker of its own.

    Read by the screen so it does not ALSO hang the measurements.db "SQL"
    button off the same field: two buttons that disagree about which file the
    column comes from is worse than the one wrong button this replaces.
    """
    return str(key or "") in CSV_COLUMN_SOURCES.get(str(app_key or ""), {})


# Settings read by exactly one Image UMAP reducer.  The controls remain in the
# form so switching methods preserves their values; only the inactive families
# are greyed.  Shared controls (random_seed and, where applicable, metric) are
# handled separately in _refresh_umap_reducer_enablement.
_UMAP_REDUCER_SETTINGS: Dict[str, set] = {
    "umap": {"n_neighbors", "min_dist"},
    "tsne": {
        "tsne_perplexity", "tsne_learning_rate",
        "tsne_early_exaggeration", "tsne_max_iter",
    },
    "pca": {"pca_whiten", "pca_svd_solver"},
    "isomap": {"isomap_n_neighbors", "isomap_path_method"},
    "spectral": {"spectral_affinity", "spectral_n_neighbors"},
}

_UMAP_TOOLTIP_OVERRIDES = {
    "reduction_method": (
        "Dimensionality reducer run before clustering and plotting. UMAP "
        "balances local and global structure; t-SNE emphasizes local "
        "neighborhoods; PCA is a fast linear baseline; Isomap preserves "
        "geodesic distances; Spectral Embedding follows a neighborhood "
        "graph. Inactive reducer controls stay visible but greyed."
    ),
    "metric": (
        "Distance metric used by UMAP, t-SNE, Isomap and DBSCAN. The "
        "dropdown contains every metric accepted by the installed UMAP "
        "implementation; PCA and Spectral Embedding ignore it."
    ),
    "n_neighbors": (
        "UMAP neighborhood size. Small values sharpen local structure; "
        "large values give a smoother global embedding. Used only by UMAP."
    ),
}

#: Every setting owned by SOME training basis. Re-enabling is restricted to
#: these, so `refresh_training_basis_enablement` cannot switch a control back
#: on that something else disabled for its own reasons.
try:
    from spacr.training_basis import BASIS_SETTINGS as _BASIS_SETTINGS
    _ALL_BASIS_SETTINGS = {k for keys in _BASIS_SETTINGS.values() for k in keys}
except Exception:      # pragma: no cover - keeps the GUI importable
    _ALL_BASIS_SETTINGS = set()


# App-specific category layouts. ``@Name`` expands the corresponding legacy
# category; plain entries are individual setting keys. The backend settings
# dictionaries remain unchanged — this controls only the order and grouping in
# Qt, just like the Classify (CV) regroup below.
_APP_CATEGORY_SPECS: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    "explain_cv": (
        ("Source & provenance", (
            "db_path", "predictions_file", "path_column",
            "prediction_column",
        )),
        ("Surrogate & validation", (
            "surrogate_model", "surrogate_split_by", "surrogate_test_size",
            "surrogate_n_estimators", "surrogate_random_seed",
            "surrogate_min_fidelity_improvement",
        )),
        ("Importance & diagnostics", (
            "surrogate_n_repeats", "surrogate_shap_max_samples",
            "surrogate_exclude", "surrogate_correlation_threshold",
        )),
        ("Output & runtime", ("dst", "verbose")),
    ),
    "investigate_hit": (
        ("Source & provenance", (
            "db_path", "predictions_file", "guide_fractions_file",
            "results_folder", "path_column", "score_column",
        )),
        ("Selected hit", (
            "target_gene", "target_guides", "hit_phenotype",
            "hit_effect", "hit_fdr", "hit_guide_agreement",
            "hit_n_guides", "hit_well_support", "hit_direction",
        )),
        ("Attribution model", (
            "hit_feature_columns", "hit_include_original_score",
            "hit_probability_threshold", "hit_split_by",
            "hit_random_seed",
        )),
        ("Evidence & output", (
            "hit_bootstrap", "hit_permutations",
            "hit_pipeline_permutations",
            "hit_gallery_per_stratum", "hit_store_database", "dst",
            "verbose",
        )),
    ),
    "umap": (
        ("Input Data", (
            "src", "tables", "crop_source", "filter_by", "row_limit",
            "exclude", "exclude_rows", "remove_highly_correlated",
            "log_data", "resnet_features", "visualize",
        )),
        ("Dimensionality Reduction", (
            "reduction_method", "random_seed", "metric",
        )),
        ("UMAP", ("n_neighbors", "min_dist")),
        ("t-SNE", (
            "tsne_perplexity", "tsne_learning_rate",
            "tsne_early_exaggeration", "tsne_max_iter",
        )),
        ("PCA", ("pca_whiten", "pca_svd_solver")),
        ("Isomap", ("isomap_n_neighbors", "isomap_path_method")),
        ("Spectral Embedding", (
            "spectral_affinity", "spectral_n_neighbors",
        )),
        ("Clustering", (
            "clustering", "eps", "min_samples", "remove_cluster_noise",
            "analyze_clusters", "color_by",
        )),
        ("Plate & Batch Correction", (
            "batch_correction", "batch_column", "batch_control_column",
            "batch_control_values", "batch_covariate_column",
            "batch_combat_mean_only", "batch_min_samples",
            "batch_missing_control",
        )),
        ("Points & Images", (
            "dot_size", "point_color", "point_alpha", "outline_width",
            "img_zoom", "image_nr", "plot_images", "remove_image_canvas",
            "plot_points", "plot_outlines", "smooth_lines",
            "plot_by_cluster", "plot_cluster_grids",
        )),
        ("Canvas & Output", (
            "figuresize", "umap_canvas_width", "umap_sidebar_width",
            "black_background", "save_figure",
        )),
        ("Runtime", ("n_jobs", "verbose")),
    ),
    "ml_analyze": (
        # Category names shared with Classify (CV) wherever the two do the
        # same job -- "Labels & Classes", "Classifier & Validation",
        # "Runtime & Reliability". The CV layout is built inline further
        # down; the names are what has to match, not the mechanism.
        ("Labels & Classes", (
            "src", "dataset_mode",
            # metadata basis
            "location_column", "positive_control", "negative_control",
            # annotation basis
            "annotation_column",
            # measurement basis
        )),
        ("Feature Preparation", (
            "channel_of_interest", "exclude", "nuclei_limit",
            "pathogen_limit", "remove_highly_correlated_features",
            "remove_low_variance_features", "min_cell_count",
        )),
        ("Plate & Batch Correction", (
            "batch_correction", "batch_column", "batch_control_column",
            "batch_control_values", "batch_covariate_column",
            "batch_combat_mean_only", "batch_min_samples",
            "batch_missing_control",
        )),
        ("Classifier & Validation", (
            "model_type_ml", "n_estimators", "learning_rate", "test_size",
            "cross_validation", "reg_alpha", "reg_lambda",
        )),
        ("Feature Selection & Importance", (
            "prune_features", "top_features", "n_repeats",
        )),
        ("Output & Database", ("save_to_db",)),
        ("Plots & Heatmaps", (
            "cmap", "heatmap_feature", "grouping", "min_max",
        )),
        ("Runtime & Reliability", ("verbose", "n_jobs")),
    ),
    "mask": (
        ("Input & Metadata", (
            "src", "cell_channel", "nucleus_channel", "pathogen_channel",
            "organelle_channel",
            *(f"{role}_channel" for role in ALL_ORGANELLE_ROLES[1:]),
            "channels", "magnification",
            "metadata_type", "custom_regex",
        )),
        ("Workflow & Test Run", (
            "preprocess", "masks", "test_mode", "test_images", "resume",
            "dry_run",
        )),
        ("Image Preprocessing", (
            "normalize", "lower_percentile", "randomize", "batch_fields",
            "consolidate",
            "denoise",
        )),
        ("Cell Segmentation", ("@Cell",)),
        ("Nucleus Segmentation", ("@Nucleus",)),
        ("Pathogen Segmentation", ("@Pathogen",)),
        ("Organelle Segmentation", ("@Organelle",)),
        # The advanced half is its own heading rather than being folded into
        # the one above, so the six settings a biologist recognises are not
        # buried under forty-eight detection parameters. Instruction 72.
        ("Organelle Segmentation (advanced)", ("@Organelle advanced",)),
        # Instruction 73: the families that are one decision applied to
        # several objects, grouped by what they do rather than by which
        # object they do it to. All three nest under "Advanced settings",
        # which is derived from the group they reference rather than
        # restated here -- see `_shared_category_parents`.
        #
        # "(per object)", NOT "Image Preprocessing": the heading above is the
        # whole-image one, and the category-help table is keyed on the
        # heading text, so two headings spelled alike would share one blurb.
        ("Image Preprocessing (per object)",
         ("@Image preprocessing (per object)",)),
        ("Object Filtration (all objects)", ("@Object filtration",)),
        ("Intensity Handling (all objects)", ("@Intensity handling",)),
        ("Quality Control", ("@Segmentation QC",)),
        ("Volumetric Processing (Beta)", ("@3D Settings (Beta)",)),
        ("Time Axes & Tracking (Beta)", ("@4D Settings (Beta)",)),
        ("Visualization & Diagnostics", (
            "plot", "cmap", "figuresize", "normalize_plots",
            "examples_to_plot",
        )),
        ("Output & Storage", (
            "save", "delete_intermediate", "keep_intermediate",
            "keep_original_images", "save_original_images", "keep_npz",
            "filter", "merge_pathogens",
        )),
        ("Runtime & Reliability", (
            "strict_errors", "max_failure_rate", "on_error",
            "on_error_attempts", "on_error_backoff", "random_seed", "verbose", "n_jobs",
            "batch_size", "pipeline_style", "diameter_estimate_n_fields",
        )),
    ),
    "measure": (
        ("Input & Experiment", ("src", "experiment")),
        ("Mask & Channel Mapping", (
            "channels", "cell_mask_dim", "nucleus_mask_dim",
            "pathogen_mask_dim",
            # HOW MANY SLOTS THERE ARE, before the slots themselves. It
            # belongs to no slot, so nothing hides it, and unclaimed it
            # landed in the bucket the layouts exist to keep empty.
            "number_of_organelles", "organelle_mask_dim",
            *(f"{role}_mask_dim" for role in ALL_ORGANELLE_ROLES[1:]),
            # WHAT KIND OF ORGANELLE each slot holds. Measure needs it for
            # the same reason mask does, and for one more: it decides
            # whether "how many, and how spread out" is the phenotype or a
            # segmentation artefact, which is what a measure run says out
            # loud about its own organelle numbers. Beside the mask
            # dimension because the two answer one question -- which plane,
            # and what is on it.
            "organelle_type",
            *(f"{role}_type" for role in ALL_ORGANELLE_ROLES[1:]),
            "cytoplasm",
            "timelapse", "timelapse_objects",
        )),
        # Illumination correction sits between the mapping and the features
        # because that is where it runs: it rewrites the pixels every
        # intensity feature below is then computed from. The Illumination
        # screen spreads these across four tabs -- correction model, field
        # sampling, QC, failure handling -- which is the right shape when
        # estimating a field is the whole job. Inside Measure it is one
        # decision with its details attached, so it is one section.
        #
        # `src` and `channels`, the other two keys the estimate reads, are
        # not repeated here: Measure already offers them above, and the
        # estimate deliberately reads the same fields the run measures.
        ("Illumination Correction", (
            "illumination_correction", "illumination_model",
            "illumination_estimator", "illumination_degree",
            "illumination_dark",
            "illumination_per_plate", "illumination_max_fields",
            "illumination_qc", "illumination_on_missing",
        )),
        ("Measurement Features", (
            "save_measurements", "calculate_correlation",
            # Instruction 71's two opt-in measurements. They were added to
            # the measure defaults and to the shared "Measurements" category
            # but NOT to this literal list, so the measure panel dropped
            # them into the trailing "Additional Settings" bucket -- which is
            # not a heading anyone chose, it is the absence of one. They
            # extend calculate_correlation, so they sit beside it.
            "corrected_manders", "spatial_measurements",
            # The radius the neighbourhood is counted in, immediately after
            # the switch that turns it on: it is baked into the column name,
            # so a screen has to pick one value and keep it.
            "spatial_neighbor_radius",
            "manders_thresholds", "homogeneity", "homogeneity_distances",
            "radial_dist", "distance_gaussian_sigma",
            # The spatial-distance block: how far every object is from
            # every other, and from the intensity maxima inside it. Filed
            # beside `radial_dist` because they answer the same kind of
            # question, one object pair at a time instead of one radius.
            "object_distances", "object_distance_maxima",
            "object_distance_intensity",
            # Not a segmentation control -- it decides which organelle summary
            # TABLES a measure run writes, so it belongs with the other
            # what-gets-measured settings rather than under the mask
            # pipeline's Organelle Segmentation heading.
            "summarize_organelles_by",
        )),
        ("Object Filtering", (
            "uninfected", "cell_min_size", "cell_max_size",
            "cytoplasm_min_size",
            "nucleus_min_size", "nucleus_max_size",
            "pathogen_min_size", "pathogen_max_size", "organelle_min_size",
            *(f"{role}_min_size" for role in ALL_ORGANELLE_ROLES[1:]),
            "merge_edge_pathogen_cells",
        )),
        ("Crop Output", (
            "save_png", "save_arrays", "crop_mode", "png_size",
            "png_channel_mapping",
            "dialate_pngs", "dialate_png_ratios", "use_bounding_box",
            "normalize", "normalize_by",
        )),
        ("Preview & Diagnostics", ("plot", "test_mode", "test_nr")),
        ("3D Calibration (Beta)", (
            "anisotropy", "voxel_size_z_um", "voxel_size_xy_um",
        )),
        ("Runtime & Reliability", (
            "resume", "strict_errors", "max_failure_rate", "on_error",
            "on_error_attempts", "on_error_backoff", "random_seed", "dry_run",
            "verbose", "n_jobs",
        )),
    ),
    "timelapse": (
        ("Input & Metadata", (
            "src", "cell_channel", "nucleus_channel", "pathogen_channel",
            "organelle_channel",
            *(f"{role}_channel" for role in ALL_ORGANELLE_ROLES[1:]),
            "channels", "magnification",
            "metadata_type", "custom_regex",
        )),
        # `timelapse` is not offered here. This module IS the timelapse
        # one -- turning it off would leave a screen whose every remaining
        # control is about a time dimension it had just been told to
        # ignore, and there is no reason a user would want that rather
        # than opening Mask Generation. It stays in the settings dict at
        # True (see `_ALWAYS_ON`), so a run gets what it expects and a
        # mask-settings CSV from before the split still round-trips.
        ("Acquisition & Axes", (
            "t_stack", "t_axis_order", "t_axis",
            "frame_interval_s", "z_stack", "z_segmentation_mode", "z_axis",
            "z_projection", "anisotropy", "voxel_size_z_um",
            "voxel_size_xy_um", "stitch_threshold",
        )),
        ("Image Preprocessing", (
            "normalize", "lower_percentile", "randomize", "batch_fields",
            "consolidate",
            "denoise",
        )),
        ("Cell Segmentation", ("@Cell",)),
        ("Nucleus Segmentation", ("@Nucleus",)),
        ("Pathogen Segmentation", ("@Pathogen",)),
        ("Organelle Segmentation", ("@Organelle",)),
        # The advanced half is its own heading rather than being folded into
        # the one above, so the six settings a biologist recognises are not
        # buried under forty-eight detection parameters. Instruction 72.
        ("Organelle Segmentation (advanced)", ("@Organelle advanced",)),
        # Instruction 73: the families that are one decision applied to
        # several objects, grouped by what they do rather than by which
        # object they do it to. All three nest under "Advanced settings",
        # which is derived from the group they reference rather than
        # restated here -- see `_shared_category_parents`.
        #
        # "(per object)", NOT "Image Preprocessing": the heading above is the
        # whole-image one, and the category-help table is keyed on the
        # heading text, so two headings spelled alike would share one blurb.
        ("Image Preprocessing (per object)",
         ("@Image preprocessing (per object)",)),
        ("Object Filtration (all objects)", ("@Object filtration",)),
        ("Intensity Handling (all objects)", ("@Intensity handling",)),
        ("Quality Control", ("@Segmentation QC",)),
        ("Tracking Setup", (
            "timelapse_objects", "timelapse_frame_limits",
            "timelapse_remove_transient", "fps",
        )),
        ("Tracking Backends", (
            "timelapse_mode", "trackastra_model", "trackastra_linking",
            "ultrack_max_distance", "ultrack_division_weight",
            "ultrack_contour_sigma", "ultrack_n_workers",
            "timelapse_displacement", "timelapse_memory",
            "t_track_backend", "t_link_threshold",
            "t_max_displacement_px", "t_max_displacement_um",
            "t_project_for_tracking",
        )),
        ("Visualization & Diagnostics", (
            "plot", "cmap", "figuresize", "normalize_plots",
            "examples_to_plot",
        )),
        ("Output & Storage", (
            "save", "delete_intermediate", "keep_intermediate",
            "keep_original_images", "save_original_images", "keep_npz",
            "filter", "merge_pathogens",
        )),
        ("Runtime & Reliability", (
            "preprocess", "masks", "test_mode", "test_images", "resume",
            "strict_errors", "max_failure_rate", "on_error",
            "on_error_attempts", "on_error_backoff", "random_seed", "dry_run", "verbose",
            "n_jobs", "batch_size", "pipeline_style",
            "diameter_estimate_n_fields",
        )),
    ),
    "motility": (
        ("Objects & Channels", (
            "src", "tracked_object", "cell_channel", "nucleus_channel",
            "pathogen_channel", "channels",
        )),
        ("Spatial & Temporal Calibration", (
            "seconds_per_frame", "pixels_per_um",
        )),
        ("Motion Filtering", (
            "max_displacement", "straightness_threshold",
            "straightness_filter", "zscore_thresh",
        )),
        ("Infection Classification", (
            "infection_intensity_strategy", "infection_intensity_qc_scope",
            "infection_intensity_mode", "infection_intensity_n_bins",
            "db_table_name", "reuse_existing_measurements",
            "infection_xgb_proba_column", "infection_xgb_drop_ambiguous",
            "infection_xgb_ambiguous_low", "infection_xgb_ambiguous_high",
        )),
        ("XGBoost Infection Model", (
            "infection_xgb_min_cells_per_class",
            "infection_xgb_n_estimators", "infection_xgb_max_depth",
            "infection_xgb_learning_rate", "infection_xgb_subsample",
            "infection_xgb_colsample_bytree", "infection_xgb_reg_lambda",
            "infection_xgb_random_state", "infection_xgb_n_jobs",
            "infection_xgb_proba_threshold", "infection_xgb_margin",
            "infection_xgb_top_features",
        )),
        ("Infection Clustering", (
            "infection_pca_n_clusters", "infection_pca_random_state",
            "infection_pca_pathogen_weight", "infection_pca_log_intensity",
            "infection_pca_min_silhouette",
            "infection_pca_min_gt_separation", "infection_pca_max_cells",
        )),
        ("Embedding Search", (
            "infection_pca_umap_search",
            "infection_pca_umap_n_neighbors_grid",
            "infection_pca_umap_min_dist_grid",
            "infection_pca_umap_n_neighbors",
            "infection_pca_umap_min_dist", "infection_pca_tsne_search",
            "infection_pca_tsne_perplexity_grid",
            "infection_pca_tsne_learning_rate_grid",
            "infection_pca_tsne_perplexity",
        )),
        ("Motility Plots & QC", (
            "motility_ylim", "motility_xlim",
            "infection_intensity_qc_graphs",
        )),
        ("Runtime & Reliability", ("n_jobs",)),
    ),
    "regression": (
        # `count_grna_column` and `count_value_column` are the count CSV's
        # own header names, which were HARD-CODED until instruction 135.
        # They belong beside the table they name: a user who has to say what
        # their count file calls its guide column is looking at the count
        # file, not at the model.
        ("Input Tables", ("paired_data", "metadata_files",
                          "count_grna_column", "count_value_column")),
        # CONTROLS AND FILTERS ARE ONE QUESTION: which rows reach the model.
        # Asked for on 2026-08-17 -- "merge quality & filters in here. change
        # the settings categoty to Controlls & Filters". They were two
        # sections with the response, the estimator and the hit-calling rules
        # between them, so it was not obvious that seven separate settings
        # each drop data.
        # THE THREE CONTROL BLOCKS AND THE EXCLUSION LIVE HERE, not in the
        # trailing "additional settings" they fell into for want of being
        # named (2026-08-21). A settings key that no panel section claims
        # lands in the catch-all, which is where a reader looks last.
        #
        # ORDER IS THE ASK: they follow `negative_control`, because they are
        # about the same thing -- which wells and which guides are controls
        # -- and the eye should not have to travel to collect them.
        #
        # `control_wells` IS GONE FROM THIS PANEL. It said "these wells are
        # controls" without saying WHICH control, and the three settings
        # below say that. Still read by the invasion-assay panel, which has
        # its own meaning for it.
        ("Controls & Filters", (
            "positive_control", "negative_control",
            "positive_control_wells", "negative_control_wells",
            "mixed_control_wells", "exclude_grnas", "controls",
            "filter_column", "filter_value",
            "min_cell_count", "min_n", "fraction_threshold",
            # DIRECTLY UNDER THE NUMBER IT REPLACES. It says "measure this
            # from the control wells instead", so it is only readable
            # beside the number it is an alternative to.
            "calibrate_fraction_threshold",
            "normalise_fraction",
            "target_unique_count", "tolerance", "outlier_detection",
        )),
        ("Plate & Batch Correction", (
            "batch_correction", "batch_column", "batch_control_column",
            "batch_control_values", "batch_covariate_column",
            "batch_combat_mean_only", "batch_min_samples",
            "batch_missing_control",
        )),
        # WHAT IS BEING MODELLED, before HOW. The response was previously
        # interleaved with the estimator settings under "Model & Covariates",
        # so the two questions a user actually asks in order -- what am I
        # measuring, and how should it be tested -- were answered in one
        # twelve-row block.
        ("Response", (
            # `score_column` retired with instruction 135 A: it named the
            # same measurement as `dependent_variable` and only offered a
            # way to disagree with it.
            "dependent_variable", "invert_dependent_variable",
            "analysis_unit", "agg_type", "transform",
            # This decides the response scale when `transform` is itself a
            # link, so it belongs immediately beside `transform`.
            "glm_transform_conflict",
        )),
        # `inference` leads because it decides whether "Estimator Tuning" or
        # "Permutation Test" below is the section that does anything.
        ("Model & Inference", (
            # `level` was in no section at all, so it fell into "Additional
            # Settings" -- the bucket this layout exists to keep empty.
            # Asked for on 2026-08-17: "level should be in model and
            # inference not additional settings". It is not a plate-layout
            # setting: it decides WHICH FITS RUN, and it is greyed out under
            # regression_type='mixed', which nests guides in genes and
            # therefore fits both levels at once. A control whose enabled
            # state is decided by `regression_type` belongs beside it, not
            # three sections away.
            # WHICH MODEL, THEN WHO FITS IT, THEN AT WHICH LEVELS. Asked for
            # on 2026-08-18: "regression backend should be in Model and
            # inference right after regression type". `regression_type` says
            # WHAT is fitted and `regression_backend` says WHO fits it -- the
            # same mixed model through statsmodels or through torch on the
            # GPU should give the same answer and not the same runtime -- so
            # the two belong adjacent, and `level` follows them.
            "inference", "analysis_mode", "regression_type",
            "regression_backend", "level",
            # WHERE THE FITTED LINE IS ANCHORED. Still part of WHAT is
            # fitted rather than which terms are in it, so it reads with
            # the four above: `intercept` chooses fitted, zero, control or
            # value, and `intercept_value` is the number the last of those
            # pins it at -- greyed for the other three.
            "intercept", "intercept_value",
            # `model_plate_position` decides whether rowID and columnID are in
            # the model at all; `random_row_column_effects` then decides fixed
            # vs random for terms that ARE in. Adjacent because setting one
            # without seeing the other is how they end up contradicting.
            "model_plate_position", "random_row_column_effects",
            # SIGNIFICANCE MERGED IN, asked for on 2026-08-17: "significance
            # nad hit calling is good but merge all of these settings into
            # Model and inference". They are not a separate question -- which
            # correction, at what level, above which effect size IS how the
            # model's output is turned into a claim, and a user reading the
            # model section had to scroll past three others to find out.
            # `p_threshold_alpha` and `p_threshold_kind` are the line the
            # plot draws significance at. The plot already had a raw/adjusted
            # choice on its right-click menu and the RUN had no say in it, so
            # the exported hit list and the picture could disagree about what
            # "significant" meant. They sit with `fdr_alpha` because the three
            # of them are one question: what counts as a hit.
            "multiple_testing_method", "fdr_alpha", "p_threshold_alpha",
            "p_threshold_kind", "threshold_method",
            "threshold_multiplier",
            # THE FIELD, NOT THE BOOLEAN. `annotation_source` supersedes
            # `Toxoplasma`: empty or 'toxoplasma' is the bundled tables
            # exactly as the True case was, and any organism name, taxon id
            # or accession is a UniProt lookup. The boolean stays in the
            # settings dict, because every CSV in existence carries it and
            # `_annotation_source` still reads it, but offering both halves
            # of a superseded pair is how a user sets one and wonders why
            # the other wins.
            "annotation_source",
        )),
        # The estimator-specific knobs, added by the robust and regularised
        # fits after this layout was first written. They landed in
        # "Additional Settings" -- the bucket a layout exists to keep empty --
        # because only the shared estimator settings above were named.
        ("Estimator Tuning", (
            # `cov_type` moved here from Model & Inference on 2026-08-17 --
            # "mooveCov type here". It is estimator-specific in exactly the
            # way everything else in this section is: the penalised, robust
            # and quantile fits have no such estimator and REFUSE it rather
            # than quietly reporting ordinary errors under a robust label.
            "cov_type",
            "alpha", "l1_ratio", "quantile", "huber_t",
            "hinge_threshold", "hinge_n_boot", "lasso_n_boot",
            "lasso_selection_threshold",
            # One knob per family, filed with the rest of them:
            # `group_lasso_lambda` is the group lasso's penalty, and
            # `rra_alpha`/`rra_permutations` are robust rank aggregation's
            # cutoff and null size.
            "group_lasso_lambda", "rra_alpha", "rra_permutations",
        )),
        # The permutation test's own settings, previously split across three
        # sections: its block and nuisance columns sat under the model, its
        # permutation count and seed under estimator tuning, and its support
        # thresholds under hit calling. Nothing here is read unless inference
        # resolves to the nonparametric test.
        ("Permutation Test", (
            # FIRST: it says WHAT is measured, and everything after it says
            # how the null is built and who is eligible -- answers to a
            # question this setting asks. Absent from this list it fell to
            # the trailing "Additional Settings", which is where a reader
            # looks last.
            "grna_statistic",
            "guide_min_wells", "guide_primary_min_wells",
            "guide_permutations", "guide_permutation_seed",
            "guide_permutation_block", "guide_nuisance_columns",
            "guide_presence_threshold", "guide_permutation_batch_size",
        )),
        # "REGRESSION PLOTS" AND "RUNTIME & RELIABILITY" ARE GONE, asked for
        # on 2026-08-17: "Regression plot can be removed" and "Runtime and
        # reliability should be removed and go to prefgerences/general".
        #
        # Neither was a question about the regression. The plot section asked
        # a user to decide the axis scaling of a figure they had not seen yet
        # -- `x_lim` and `y_lims` are set on the plot now -- and the runtime
        # section asked how the whole application handles a failure, which is
        # the same answer for every module and belongs in Preferences.
        #
        # The keys they held are not dropped, they are HIDDEN
        # (`_APP_HIDDEN_KEYS`): dropping `regression_qc` would take
        # `parameter_sweep`'s `settings.setdefault("regression_qc", False)`
        # with it, and a hundred-trial sweep would pay ~5.8 s and ~19 figures
        # per trial for diagnostics nobody opens.
    ),
    "activation": (
        ("Model & Data", (
            "dataset", "model_path", "model_type", "image_size",
            "object_type", "channels",
        )),
        ("Attribution Method", (
            "cam_type", "target_layer", "smoothgrad_samples",
            "smoothgrad_sigma", "occlusion_window", "occlusion_stride",
            "ig_steps", "ig_baseline",
        )),
        ("Attribution Validation", (
            "attribution_steps", "attribution_baseline", "sanity_check",
        )),
        ("Map Display", (
            "normalize", "normalize_input", "overlay", "plot",
        )),
        ("Map Quantification", ("correlation", "manders_thresholds")),
        ("Output & Runtime", (
            "save", "shuffle", "batch_size", "n_jobs",
        )),
    ),
    "recruitment": (
        ("Data source", ("src",)),
        ("Mask & Channel Mapping", (
            "cell_mask_dim", "cell_chann_dim", "nucleus_mask_dim",
            "nucleus_chann_dim", "pathogen_mask_dim", "pathogen_chann_dim",
            "channel_dims", "channel_of_interest",
        )),
        ("Object Filtering", (
            "cell_size_range", "cell_intensity_range", "nucleus_size_range",
            "nucleus_intensity_range", "pathogen_size_range",
            "pathogen_intensity_range", "cells_per_well",
            "target_intensity_min", "nuclei_limit", "pathogen_limit",
        )),
        ("Plate Layout & Controls", ("@Plate Layout & Controls",)),
        ("Plots & Diagnostics", (
            "plot", "figuresize", "plot_control", "plot_nr",
        )),
    ),
    "invasion": (
        ("Assay Inputs", ("src", "parasite_table", "compartment")),
        ("Channels & Intensity", (
            "outside_channel", "total_channel", "intensity_statistic",
            "background_correction", "min_total_intensity",
        )),
        ("Thresholding", (
            "outside_threshold_method", "outside_threshold",
            "threshold_agreement_tolerance", "threshold_sensitivity",
            "bimodality_cutoff", "extracellular_class",
        )),
        ("Controls & Minimum Counts", (
            "control_wells", "control_quantile", "min_control_objects",
            "min_objects_for_threshold", "min_objects_for_bimodality",
            "min_parasites_per_well", "inflation_warn",
        )),
        ("Object Filtering", ("min_parasite_area", "max_parasite_area")),
        ("Condition Metadata", ("@Plate Layout & Controls",)),
        ("Assay Output", (
            "cmap", "qc_plot_max_panels", "seed_wells_from_cells", "save",
        )),
        ("Runtime & Reliability", ("verbose",)),
    ),
    # -- the three Cellpose-facing modules -------------------------------
    #
    # All three used to render the shared "Cellpose" category as one drop of
    # ten to thirteen knobs. They are not one decision: the model you run,
    # the thresholds that decide how much it finds, the geometry it sees and
    # the background correction applied before it are four separate
    # questions, asked at four different times. The groups below are the same
    # four in all three modules so that moving between them is not a
    # relearning exercise.
    "cellpose_masks": (
        ("Input & Channels", (
            "src", "channels", "grayscale", "invert", "normalize",
            "percentiles",
        )),
        ("Model", ("model_name", "custom_model", "diameter")),
        ("Detection Thresholds", (
            "CP_prob", "flow_threshold", "rescale", "resample", "fill_in",
        )),
        ("Image Geometry", ("resize", "target_height", "target_width")),
        ("Background & Denoising", (
            "remove_background", "background", "Signal_to_noise",
        )),
        ("Output & Runtime", ("save", "batch_size", "verbose")),
    ),
    "cellpose_all": (
        ("Input & Channels", (
            "channels", "grayscale", "invert", "normalize", "percentiles",
        )),
        ("Model", ("diameter",)),
        ("Detection Thresholds", ("CP_prob", "flow_threshold")),
        ("Image Geometry", ("resize", "target_height", "target_width")),
        ("Background & Denoising", (
            "remove_background", "background", "Signal_to_noise",
        )),
        ("Output & Runtime", ("plot", "save", "batch_size", "verbose")),
    ),
    "train_cellpose": (
        ("Starting Point", ("model_type", "from_scratch", "model_name")),
        ("Training Schedule", (
            "n_epochs", "learning_rate", "weight_decay", "batch_size",
            "augment",
        )),
        ("Image Geometry", (
            "width_height", "target_size", "diameter", "resize",
        )),
        ("Background & Denoising", (
            "remove_background", "background", "Signal_to_noise",
        )),
        ("Output & Runtime", ("verbose",)),
    ),
    "analyze_plaques": (
        ("Input & Channels", ("src", "masks")),
        ("Model", ("diameter",)),
        ("Detection Thresholds", (
            "CP_prob", "flow_threshold", "rescale", "resample", "fill_in",
        )),
        ("Image Geometry", ("resize", "target_height", "target_width")),
        ("Background & Denoising", (
            "remove_background", "background", "Signal_to_noise",
        )),
        ("Output & Runtime", ("save", "batch_size", "verbose")),
    ),
    "map_barcodes": (
        ("Sequencing Input", ("src", "mode", "single_direction")),
        ("Barcode References", ("grna_csv", "row_csv", "column_csv")),
        ("Read Parsing", (
            "target_sequence", "regex", "offset_start", "expected_end",
            # How far a read may be from a listed barcode and still be
            # called as it -- a parsing tolerance, filed with the rest of
            # the parse. Left out of this layout it fell into "Additional
            # Settings", which is not a heading anyone chose; it is the
            # absence of one.
            "barcode_mismatches",
        )),
        ("Output & Storage", (
            "save_h5", "comp_type", "comp_level", "fill_na",
        )),
        ("Runtime & Reliability", ("chunk_size", "n_jobs", "test")),
    ),
    "barcode_qc": (
        ("Reference & Count Tables", (
            "grna_csv", "row_csv", "column_csv", "count_data", "qc_data",
        )),
        ("Well Expectations", (
            "target_grnas_per_well", "target_statistic", "min_reads_per_well",
        )),
        ("Starvation & Exclusion", (
            "starved_read_fraction", "exclude_starved_wells",
        )),
        ("Position & Collision Checks", (
            "position_effect_ratio", "collision_max_distance",
        )),
        ("Threshold Sweep", ("sweep_span", "sweep_points")),
        ("QC Output", ("dst", "plot", "save")),
        ("Runtime & Reliability", ("verbose",)),
    ),
    "illumination": (
        ("Input & Channels", ("src", "channels")),
        ("Correction Model", (
            "illumination_correction", "illumination_model",
            "illumination_estimator", "illumination_degree",
            "illumination_dark",
        )),
        ("Field Sampling", (
            "illumination_per_plate", "illumination_max_fields",
        )),
        ("QC & Failure Handling", (
            "illumination_qc", "illumination_on_missing",
        )),
    ),
    # Power / Design draws its own screen, so these groups are never a
    # settings form. They are still the layout of record: the settings diff,
    # the run journal and `utils.pretty_print_settings` all group by
    # category, and fifteen keys under one "Power analysis" heading make a
    # design change unreadable in all three.
    "power": (
        ("Library Design", (
            "power_n_genes", "power_n_grnas_per_gene",
            "power_constructs_per_well",
        )),
        ("Plate Layout", (
            "power_wells_per_plate", "power_n_plates", "power_n_replicates",
            "power_cells_per_well",
        )),
        ("Effect & Prevalence", (
            "power_effect_fold", "power_hit_rate",
            "power_background_positive_rate", "power_detection_auroc",
        )),
        ("Sequencing Depth", ("power_reads_per_well",)),
        ("Simulation", ("power_score_per", "power_backend", "power_seed")),
    ),
    "anndata_export": (
        ("Input Tables", ("src", "anndata_tables")),
        ("Output File", (
            "anndata_out", "anndata_single_table", "anndata_compression",
            "anndata_dtype",
        )),
        ("Rows & Missing Values", (
            "anndata_row_limit", "anndata_nan_policy",
        )),
        ("Post-processing", (
            "anndata_compute_umap", "anndata_register_artifact",
        )),
    ),
    "replication": (
        ("Assay Inputs", ("src", "parasite_table", "compartment")),
        ("Vacuole Assignment", (
            "vacuole_key", "vacuole_link_distance", "vacuole_link_factor",
            "parasite_count_column", "require_host_cell",
        )),
        ("Condition Metadata", (
            "cell_types", "cell_plate_metadata", "pathogen_types",
            "pathogen_plate_metadata", "treatments",
            "treatment_plate_metadata", "group_column", "level",
            "change_plate",
        )),
        ("Object Filtering", (
            "min_parasite_area", "max_parasite_area",
        )),
        ("Replication Scoring", (
            "max_parasites_per_vacuole", "non_power_of_two_warn",
            "seed_wells_from_cells",
        )),
        ("Assay Output", ("cmap", "save")),
        ("Runtime & Reliability", ("verbose",)),
    ),
}


#: Settings a first-time user of a module has to touch beyond its first
#: group, in the same ``@Section``-or-key language as
#: :data:`_APP_CATEGORY_SPECS`.
#:
#: The first group of a curated layout is by construction the "what you must
#: set" group — every layout in this module opens with the inputs — so it is
#: taken as essential automatically and never restated here. This table only
#: adds the second thing: Measure's mask-to-channel mapping, Regression's
#: model choice, Train Cellpose's schedule. Anything naming a key or a group
#: that no longer exists is dropped silently, the same way a spec token is,
#: so a stale entry costs a row of disclosure and never an exception.
_APP_ESSENTIAL_EXTRAS: Dict[str, Tuple[str, ...]] = {
    "mask": ("preprocess", "masks", "test_mode", "test_images", "plot",
             "save"),
    "timelapse": ("timelapse", "t_stack", "frame_interval_s",
                  "timelapse_objects", "test_mode", "save"),
    "measure": ("@Mask & Channel Mapping", "test_mode"),
    "motility": ("@Spatial & Temporal Calibration",),
    "ml_analyze": ("channel_of_interest", "model_type_ml"),
    "regression": ("@Controls & Filters", "regression_type",
                   "dependent_variable"),
    "activation": ("cam_type", "target_layer"),
    "replication": ("@Vacuole Assignment",),
    "recruitment": ("@Mask & Channel Mapping",),
    "invasion": ("@Channels & Intensity",),
    "cellpose_masks": ("@Model",),
    "cellpose_all": ("@Model",),
    "analyze_plaques": ("@Model",),
    "train_cellpose": ("n_epochs", "learning_rate"),
    "map_barcodes": ("@Barcode References",),
    "barcode_qc": ("@Well Expectations",),
    "illumination": ("illumination_correction", "illumination_model"),
    "anndata_export": ("anndata_out",),
    "classify": ("@Labels & Classes", "model_type", "train_channels"),
    "umap": ("tables", "reduction_method", "color_by"),
    "external_masks": ("channels", "experiment"),
}


def _expand_layout_tokens(
    source: Dict[str, List[str]],
    tokens: Tuple[str, ...],
) -> List[str]:
    """Resolve ``@Section``-or-key tokens against a category map, in order.

    The same token language :data:`_APP_CATEGORY_SPECS` uses, so a layout and
    the essentials drawn from it can never disagree about what ``@Cell``
    means. (:func:`_categories_from_spec` keeps its own copy of the loop
    because it additionally has to remember which keys earlier *sections*
    already claimed; this one resolves a single flat list.)

    Unknown tokens and keys the module does not actually have are dropped,
    and a key named twice is kept once, at its first position.
    """
    available = {key for keys in source.values() for key in keys}
    out: List[str] = []
    for token in tokens:
        candidates = (
            source.get(token[1:], []) if token.startswith("@") else [token]
        )
        for key in candidates:
            if key in available:
                out.append(key)
    return list(dict.fromkeys(out))


def essential_keys(
    app_key: str,
    categories: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """The settings a first-time user of ``app_key`` should meet first.

    Progressive disclosure needs a defensible answer to "which of these 190
    matter?", and a hand-written list per module would rot the first time a
    layout changed. So it is *derived*: the first group of the module's
    curated layout, which is always its inputs, plus whatever
    :data:`_APP_ESSENTIAL_EXTRAS` adds for that module.

    A module with no curated layout gets the first shared category, which is
    "Paths" — still the right answer, just a thinner one.

    :param app_key: the module's app key.
    :param categories: optional pre-computed :func:`categories_for_app`
        output, to save recomputing it.
    :returns: setting keys in display order, without duplicates.
    """
    cats = (categories if categories is not None
            else categories_for_app(app_key, get_categories()))
    ordered = list(cats.items())
    keys: List[str] = list(ordered[0][1]) if ordered else []
    keys.extend(
        _expand_layout_tokens(
            cats, _APP_ESSENTIAL_EXTRAS.get(str(app_key or ""), ())
        )
    )
    return list(dict.fromkeys(keys))


def _categories_from_spec(
    source: Dict[str, List[str]],
    spec: Tuple[Tuple[str, Tuple[str, ...]], ...],
) -> Dict[str, List[str]]:
    """Expand one app layout and retain future settings under a named bucket."""
    ordered: Dict[str, List[str]] = {}
    assigned = set()
    available = {key for keys in source.values() for key in keys}
    for title, tokens in spec:
        keys: List[str] = []
        for token in tokens:
            if token.startswith("@"):
                # A group reference can only mean what the shared map says
                # it means, so it is filtered by what is actually in there.
                candidates = [key for key in source.get(token[1:], [])
                              if key in available]
            else:
                # A literal key is the spec ASSERTING where that setting
                # belongs, and it outranks the shared category map — which
                # for Barcode QC and Illumination has never heard of their
                # keys at all. Filtering literals by `available` sent all
                # eleven of Barcode QC's checks to the trailing "Other"
                # bucket, which is the exact thing the layout exists to
                # prevent. Whether the key exists is decided at render time,
                # where `build_sections` already drops any key that produced
                # no widget.
                candidates = [token]
            for key in candidates:
                if key not in assigned:
                    assigned.add(key)
                    keys.append(key)
        ordered[title] = keys

    remaining = []
    for keys in source.values():
        for key in keys:
            if key not in assigned:
                assigned.add(key)
                remaining.append(key)
    if remaining:
        ordered["Additional Settings"] = remaining
    return ordered


def _drop_hidden_keys(app_key: str,
                      categories: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Remove :data:`_APP_HIDDEN_KEYS` from a rendered layout.

    Applied after the layout rather than before, and after the fallback
    bucket is filled, because the bucket is exactly where a key goes when
    no layout claims it -- hiding one by leaving it out of the spec moves
    it to "Additional Settings" instead of hiding it.

    A category emptied by this disappears with it, so a module does not
    grow a heading with nothing under it.
    """
    hidden = _APP_HIDDEN_KEYS.get(app_key)
    if not hidden:
        return categories
    out: Dict[str, List[str]] = {}
    for title, keys in categories.items():
        kept = [key for key in keys if key not in hidden]
        if kept:
            out[title] = kept
    return out


def get_categories() -> Dict[str, List[str]]:
    """Return the {category_name: [setting keys]} mapping."""
    from spacr.settings import categories
    return categories


# ---------------------------------------------------------------------------
# THE SETTINGS TREE. Instruction 73.
# ---------------------------------------------------------------------------
#
# The panel used to group by OBJECT and nothing else, so `cell_min_size` and
# `nucleus_min_size` -- one decision applied to two objects -- read as two
# unrelated knobs filed under two headings. The request is a second axis:
# group the advanced settings by WHAT THEY DO, then by which object they do
# it to, under one "advanced settings" umbrella.
#
# That needs three levels, and the panel had one. `build_sections` returned
# List[Tuple[str, List[Tuple[str, QWidget]]]] -- a header and its rows, no
# third element and no recursion -- so a sub-sub-section could not be
# expressed at all. Widening that return type is a contract change for every
# module in the tool, which is why the section below is a TUPLE SUBCLASS: it
# still IS the pair it always was, so nothing that unpacks or `dict()`s the
# result has to change, and the tree hangs off attributes beside it.


class SettingsSection(tuple):
    """One heading of a settings panel, and whatever nests under it.

    IT IS STILL A ``(title, rows)`` PAIR, and that is the point. Every caller
    that writes ``for title, rows in build_sections()`` or
    ``dict(build_sections())`` keeps working untouched, and ``rows`` holds
    every row in the whole subtree -- so a panel that cannot draw a tree
    still draws each control exactly once, under the outermost heading it
    belongs to, instead of losing the ones a sub-heading owns.

    THE TREE IS THE ADDITION. :attr:`own_rows` are the rows of this heading
    itself, :attr:`children` the headings below it, and :attr:`path` the
    breadcrumb down to it -- ``("Advanced settings", "Object filtration",
    "Cell")`` -- which is what tells a sub-heading titled "Cell" apart from
    the top-level "Cell" segmentation category when help is looked up.
    """

    # No `__slots__`: a variable-length tuple subclass cannot have one, and
    # the four attributes below are what carries the tree.

    def __new__(cls, title, own_rows=(), children=()):
        children = tuple(children)
        own = list(own_rows)
        rows = list(own)
        for child in children:
            rows.extend(child.rows)
        section = super().__new__(cls, (str(title), rows))
        section.title = str(title)
        section.own_rows = own
        section.children = children
        section.path = (section.title,)
        for child in children:
            child._reparent(section.path)
        return section

    def _reparent(self, parent_path) -> None:
        """Record this section's place under a parent that now exists.

        A child is built before the parent that will hold it, so its path is
        completed from above rather than passed down.
        """
        self.path = tuple(parent_path) + (self.title,)
        for child in self.children:
            child._reparent(self.path)

    @property
    def rows(self) -> List[Tuple[str, QWidget]]:
        """Every row in this heading and in everything nested under it."""
        return self[1]

    def walk(self):
        """This section and every section below it, outermost first."""
        yield self
        for child in self.children:
            yield from child.walk()


def _shared_category_parents() -> Dict[str, str]:
    """Which heading each category nests under, including renamed ones.

    `spacr.settings.CATEGORY_PARENTS` is keyed on the SHARED category name,
    and a module layout may draw the same group under its own spelling --
    mask calls "Object filtration" "Object Filtration (all objects)". A
    layout entry built out of nothing but ``@Family`` references is that
    family, so its place in the tree is DERIVED rather than restated. The
    alternative is a second table listing every rename, and this project has
    already shipped three defects from a module being registered in one such
    table and not the other.
    """
    from spacr.settings import CATEGORY_PARENTS

    parents = dict(CATEGORY_PARENTS)
    for spec in _APP_CATEGORY_SPECS.values():
        for title, tokens in spec:
            groups = [t[1:] for t in tokens if str(t).startswith("@")]
            if len(groups) != len(tokens) or not groups:
                continue
            inherited = {CATEGORY_PARENTS[g] for g in groups
                         if g in CATEGORY_PARENTS}
            if len(inherited) == 1 and len(groups) == 1:
                parents[title] = inherited.pop()
    return parents


def _object_subheading(obj: str) -> str:
    """The heading one object's rows are drawn under.

    Organelle slots are numbered rather than spelled `organelleb`, which is
    an internal name chosen so object keys can round-trip through `prcfo`
    and was never meant to be read.

    THE SLOT IS RECOGNISED BY ITS NAME, not by the schema's list of the slots
    that carry a mask plane today. That list stops at four, so a run with
    seven organelles grouped its fifth slot's rows correctly and then drew
    them under "Organellee" -- the internal spelling, leaked by the one
    function whose job is to keep it out of sight. Asking
    :mod:`spacr.organelle_types`, which owns the naming, covers every slot
    the alphabet allows.
    """
    from spacr.object_roles import organelle_label
    from ...organelle_types import organelle_role_of

    if organelle_role_of(obj) == str(obj):
        return organelle_label(obj)
    return str(obj).replace("_", " ").capitalize()


def _split_rows_by_object(rows, keys):
    """Split one family's rows into a sub-section per object.

    :param rows: ``(label, widget)`` in the order the family lists them.
    :param keys: the setting key behind each row, positionally aligned.
    :returns: ``(own_rows, children)`` -- a row whose key names no object
        stays with the family itself rather than being dropped, because a
        control that reaches no sub-heading is a control the user cannot
        reach.
    """
    from spacr.settings import ADVANCED_OBJECT_ORDER, advanced_object_of

    grouped: Dict[str, List[Tuple[str, QWidget]]] = {}
    own: List[Tuple[str, QWidget]] = []
    for row, key in zip(rows, keys):
        obj = advanced_object_of(key)
        if obj is None:
            own.append(row)
        else:
            grouped.setdefault(obj, []).append(row)
    children = tuple(
        SettingsSection(_object_subheading(obj), grouped[obj])
        for obj in ADVANCED_OBJECT_ORDER if grouped.get(obj)
    )
    return own, children


def _nest_sections(flat) -> List[SettingsSection]:
    """Hang each flat section under the parent its category declares.

    THE PARENT TAKES THE PLACE OF ITS FIRST CHILD, so the running order of a
    panel is the one its layout wrote. Hoisting the umbrella to the top or
    dropping it to the bottom would move a block of settings the layout
    deliberately put between two others.

    A parent whose children all vanished -- every key hidden, or none
    offered by this module -- is not emitted, the same rule an empty
    category has always followed.
    """
    parents = _shared_category_parents()
    order: List[str] = []
    umbrellas: Dict[str, List[SettingsSection]] = {}
    out: List[object] = []
    for section in flat:
        parent = parents.get(section.title)
        if parent is None:
            out.append(section)
            continue
        if parent not in umbrellas:
            umbrellas[parent] = []
            order.append(parent)
            out.append(parent)          # a placeholder, replaced below
        umbrellas[parent].append(section)
    return [SettingsSection(item, (), umbrellas[item])
            if isinstance(item, str) else item
            for item in out]


#: Below this many settings a module cannot render as an undifferentiated
#: list — six rows fit on one screen and read as one group whatever they are
#: called. Modules at or under it are exempt from :func:`has_curated_layout`;
#: everything above it has to say what its groups are.
CURATION_THRESHOLD = 6

#: Modules whose layout is curated inline in :func:`categories_for_app`
#: rather than declared in :data:`_APP_CATEGORY_SPECS`.
#:
#: Classify is the odd one out on purpose: its ten groups are built as a
#: literal ``ordered`` dict because several of them list keys that are in no
#: shared category at all, which the ``@Name``-expanding spec form cannot
#: express. UMAP and External Masks reshape the shared categories in place —
#: they add groups ("UMAP Display", "Input mapping") rather than replacing
#: the whole layout, and a spec would have to restate every key they leave
#: alone. All four are curated; none of them is a spec.
#:
#: `classify_merged` shares Classify's regroup and then amends it — it is
#: named twice in :func:`categories_for_app`, once with `classify` and once
#: on its own to lift the family switch out of "Model Architecture".
_INLINE_LAYOUT_APPS = frozenset({
    "classify", "classify_merged", "umap", "external_masks",
})


def has_curated_layout(app_key: str) -> bool:
    """Return True when ``app_key``'s settings panel has a layout of its own.

    "Of its own" means somebody decided what this module's groups are — a
    :data:`_APP_CATEGORY_SPECS` entry, an inline regroup in
    :func:`categories_for_app`, or a plugin that shipped ``categories``.

    Falling back to the shared category map is *not* curated. That map is
    keyed by what a setting is (a path, a plot option, "Advanced"), not by
    what the module does with it, so a module that relies on it renders as
    however many buckets its keys happen to fall into — which for Cellpose
    Masks was thirteen knobs under one "Cellpose" heading.

    :param app_key: the module's app key.
    """
    key = str(app_key or "")
    if key in _APP_CATEGORY_SPECS or key in _INLINE_LAYOUT_APPS:
        return True
    try:
        from spacr.plugins import get_app
        plugin_app = get_app(key)
    except Exception:
        return False
    return bool(plugin_app is not None and plugin_app.categories)


def needs_curated_layout(app_key: str) -> bool:
    """Return True when ``app_key`` has enough settings to need grouping.

    Interactive modules whose settings dict is the ``{"src": ...}``
    placeholder render a bespoke screen, not the shared form; they have
    nothing to group. :data:`CURATION_THRESHOLD` draws the line.

    :param app_key: the module's app key.
    """
    try:
        return len(resolve_default_settings(app_key)) > CURATION_THRESHOLD
    except Exception:
        # An app whose defaults will not resolve has no settings panel to
        # judge. Reporting "needs a layout" would fail the invariant test for
        # a reason that has nothing to do with layouts.
        return False


#: The dash between a family prefix and the group name in a merged module's
#: heading. Written once: an em dash swapped for a hyphen in an edit fails
#: silently, because the lookups below simply stop matching.
_FAMILY_HEADING_DASH = "—"


def _family_heading(prefix: str, name: str) -> str:
    """Compose one family-prefixed section heading and catalogue the pair.

    The composed heading is a key as much as a caption: the blurb tables,
    the hidden-category lists and the layout tests are all written against
    the English ``Computer Vision — Images & Cropping``, so the heading is
    returned in English and only its TRANSLATION is composed here. No
    catalog can carry a row for every prefix and group name that meet, and
    asking for the finished pair is what leaves these headings reading half
    English; each half is looked up on its own and the halves joined, so a
    row written for either one reaches the header — which knows only the
    finished pair. A translation that already exists for the whole pair
    wins, so a reviewed caption is never displaced by a composed one.
    """
    heading = f"{prefix} {_FAMILY_HEADING_DASH} {name}"
    try:
        from ..i18n import (VALID_LANGUAGE_CODES, _exact_translation,
                            add_translation, tr)

        add_translation(heading, [
            _exact_translation(heading, code)
            or f"{tr(prefix, code)} {_FAMILY_HEADING_DASH} {tr(name, code)}"
            for code in VALID_LANGUAGE_CODES[1:]
        ])
    except (ImportError, AttributeError, ValueError):
        # A catalog that will not take the row leaves the heading reading
        # exactly as it does today. A panel that is otherwise ready to build
        # must not fail over a caption.
        pass
    return heading


def categories_for_app(
    app_key: str,
    categories: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Return category keys after applying module-specific relocations.

    Map Barcodes previously showed an ``Advanced`` tab containing only
    ``n_jobs`` and a ``Model Training`` tab containing only ``test``.  Both
    controls belong to the sequencing run, but changing the global category
    table would also move training controls in unrelated modules.
    """
    try:
        from spacr.plugins import get_app
        plugin_app = get_app(app_key)
    except Exception:
        plugin_app = None
    if plugin_app is not None and plugin_app.categories:
        return {
            str(name): list(keys)
            for name, keys in plugin_app.categories.items()
        }
    result = {name: list(keys) for name, keys in categories.items()}
    if app_key == "external_masks":
        input_keys = (
            "inputs", "dst", "recursive", "layout", "z_handling",
            "plate_naming", "overwrite", "preview_only",
        )
        for keys in result.values():
            for key in input_keys:
                while key in keys:
                    keys.remove(key)
        result = {"Input mapping": list(input_keys), **result}
    # Map Barcodes used to relocate `n_jobs` and `test` into "Sequencing"
    # here, so the module would stop rendering an "Advanced" tab holding one
    # setting and a "Model Training" tab holding another. That left thirteen
    # unrelated keys in one "Sequencing" drop; `_APP_CATEGORY_SPECS` now
    # names all five groups the module actually has, which places those two
    # keys — and every other one — explicitly. The relocation is not deleted
    # behaviour, it is superseded behaviour.
    if app_key == "umap":
        batch_correction = (
            "batch_correction", "batch_column", "batch_control_column",
            "batch_control_values", "batch_covariate_column",
            "batch_combat_mean_only", "batch_min_samples",
            "batch_missing_control",
        )
        display = (
            "figuresize", "dot_size", "point_color", "point_alpha",
            "outline_width", "umap_canvas_width", "umap_sidebar_width",
            "img_zoom", "image_nr", "plot_images", "remove_image_canvas",
            "plot_points", "plot_outlines", "smooth_lines",
            "plot_by_cluster", "plot_cluster_grids", "black_background",
            "save_figure",
        )
        for keys in result.values():
            for key in (*display, *batch_correction):
                while key in keys:
                    keys.remove(key)
        result["Plate & Batch Correction"] = list(batch_correction)
        result["UMAP Display"] = list(display)
    if app_key in _APP_CATEGORY_SPECS:
        result = _categories_from_spec(result, _APP_CATEGORY_SPECS[app_key])
    if app_key in ("classify", "classify_merged"):
        # NINE groups became SIX, named for what they hold rather than for
        # the stage of a workflow. "Validation", "Evaluation Workbench" and
        # "Monitoring & Runtime" were three headings for one question -- how
        # do I know whether this worked -- and nobody looking for a
        # cross-validation setting knew which of the three to open.
        ordered = {
            "Plate Sources & Workflow": [
                "src", "experiment", "generate_training_dataset", "train",
                "test", "generate_full_dataset", "apply_model_to_dataset",
                "dataset", "model_path", "tar_path"],

            # `classes` is the heart of this module: it is the only setting
            # that says which objects the model is being taught to tell
            # apart. `dataset_mode` sits above it because it decides which
            # columns the Classes editor offers.
            #
            # `location_column`, `positive_control` and `negative_control` are
            # NOT here: a control well is a class defined by a metadata
            # column, which is exactly a row of the Classes dict, so three
            # settings saying it a second way were three ways to disagree.
            "Labels & Classes": [
                # `classes` is what each class MEANS; `class_folder_names`
                # is where its crops are written. One key used to be both.
                # `metadata_type_by` and `measurement_rules` are GONE, not
                # hidden. The first named the column a class is defined by,
                # which is the Classes editor's own column field; the second
                # was a second vocabulary for "a class is a rule about a
                # column", written as hand-edited JSON because it had no
                # editor. Both were answers to a question `classes` now asks
                # once.
                # `annotation_column` and `class_metadata` are GONE from
                # the panel (instruction 229): the Classes editor names the
                # column each class is defined by and the value that
                # defines it, so a box for either was a second place to say
                # the same thing. Both are still WRITTEN, derived from
                # `classes`, so every consumer downstream is unchanged.
                "dataset_mode", "classes", "class_folder_names",
                "metadata_item_1_name", "metadata_item_1_value",
                "metadata_item_2_name", "metadata_item_2_value",
                "balance_to_smallest", "test_split",
                "val_split", "sample"],

            "Images & Cropping": [
                "image_source", "load_path_regex", "tables",
                "channel_of_interest", "stream_method", "object_array",
                "mask_array", "channel_arrays", "bounding_box",
                "crop_shape", "train_channels", "image_size", "augment"],

            "Model & Regularization": [
                "classifier_family",
                "model_type", "custom_model", "custom_model_path",
                "resume_checkpoint", "init_weights",
                "normalize", "normalization", "normalization_scope",
                "dropout_rate", "weight_decay", "use_checkpoint"],

            "Training & Loss": [
                "epochs", "optimizer_type", "learning_rate", "schedule",
                "amsgrad", "loss_type", "class_balance", "label_smoothing",
                "focal_gamma", "focal_alpha", "logit_adjust_tau",
                "batch_size", "mixed_precision", "gradient_accumulation",
                "gradient_accumulation_steps", "early_stopping_patience"],

            "Evaluation & Results": [
                "cross_validation_enabled", "cross_validation_folds",
                # The plate held back from fitting, beside the folds it is
                # the alternative to. Left out of this layout it fell into
                # "Additional Settings", which is not a heading anyone chose;
                # it is the absence of one.
                "cv_group_by", "holdout_plate", "nested_cv_inner_folds",
                "score_threshold",
                "classifier_evaluation", "evaluation_calibration",
                "evaluation_bins", "evaluation_fail_on_leakage",
                "leakage_audit_train_test", "leakage_hash_content",
                "leakage_require_identity", "n_top_examples", "save_to_db",
                "plot", "tensorboard", "intermedeate_save", "pin_memory",
                "random_seed", "n_jobs", "verbose", "strict_errors",
                "max_failure_rate"],
        }
        if app_key == "classify_merged":
            # The family switch is the TOP-LEVEL choice, not one setting
            # among ninety, so it gets its own group at the top rather than
            # sitting inside "Model Architecture". Greying tells the user a
            # control is inactive; it does not tell them what they are
            # DOING, and that is what the merged module has to make obvious.
            ordered["Model & Regularization"] = [
                k for k in ordered["Model & Regularization"]
                if k != "classifier_family"]

            # The control wells are NOT added back. ML used to express the
            # metadata basis through location_column plus two control values,
            # and Classify (CV) through class_metadata; the Classes dict now
            # says both, as rows naming a value of a metadata column. Three
            # settings restating one thing were three ways for it to disagree
            # with itself.

            # The ML-only groups, appended to the CV ordering rather than
            # duplicated: every CV key is already placed above, so this is
            # exactly the difference between the two modules. Group names
            # match Classify (ML)'s own, so a user moving between the three
            # screens sees the same headings.
            ordered.update({

                "Plate & Batch Correction": [
                    "batch_correction", "batch_column",
                    "batch_control_column", "batch_control_values",
                    "batch_covariate_column", "batch_combat_mean_only",
                    "batch_min_samples", "batch_missing_control"],
                # Preparing features, choosing a model and ranking features
                # were three headings asking one question: which features the
                # model uses. One heading, read top to bottom.
                "Model & Features": [
                    "model_type_ml", "n_estimators", "test_size",
                    "cross_validation", "reg_alpha", "reg_lambda",
                    "exclude", "nuclei_limit", "pathogen_limit",
                    "remove_highly_correlated_features",
                    "remove_low_variance_features", "min_cell_count",
                    "prune_features", "top_features", "n_repeats"],
            })
            # The heatmap is not a machine-learning setting: it is how a
            # result is shown, and the CV family wants it just as much. So it
            # joins the shared evaluation group rather than being prefixed
            # onto one family.
            ordered["Evaluation & Results"] = (
                ordered["Evaluation & Results"]
                + ["cmap", "heatmap_feature", "grouping", "min_max"])

        if app_key == "classify_merged":
            # Rebuilt in order, because dict order IS the panel order: the
            # family choice first, then the shared groups, then each
            # family's own settings under a heading that names the family.
            #
            # The shared groups are deliberately NOT prefixed. "Labels &
            # Classes" applies to both families, and prefixing it would
            # imply it belonged to one.
            cv_family = "Computer Vision"
            ml_family = "Machine Learning"
            cv_groups = ("Images & Cropping", "Model & Regularization",
                         "Training & Loss")
            # Feature preparation and feature importance were two headings
            # asking one question -- which features the model uses -- and
            # "Output & Database" was one setting under a heading of its own.
            ml_groups = ("Model & Features", "Plate & Batch Correction")
            shared_first = ("Plate Sources & Workflow", "Labels & Classes")
            shared_last = ("Evaluation & Results",)

            rebuilt = {"Classifier": ["classifier_family"]}
            for name in shared_first:
                if name in ordered:
                    rebuilt[name] = ordered[name]
            for name in cv_groups:
                if name in ordered:
                    rebuilt[_family_heading(cv_family, name)] = ordered[name]
            for name in ml_groups:
                # A rename of `label` to "Classifier & Validation" stood here,
                # guarded on `name.startswith("ML Classifier")` so the heading
                # would not read "Machine Learning - ML Classifier". No entry
                # of `ml_groups` has been called that since the groups above
                # were consolidated, so it could not run and was removed.
                if name in ordered:
                    rebuilt[_family_heading(ml_family, name)] = ordered[name]
            # Shared groups that come LAST -- evaluation applies to both
            # families, so prefixing it onto one would be a lie about who it
            # belongs to, and putting it first would bury the settings that
            # decide what is being trained.
            for name in shared_last:
                if name in ordered:
                    rebuilt[name] = ordered[name]
            # A catch-all that copied any group the five tuples above do not
            # name stood here, and it could not run either: `ordered` is the
            # literal a hundred lines up plus this branch's own two additions,
            # and the tuples enumerate every one of them. It was a net against
            # that literal growing a group nobody added to a tuple -- so the
            # invariant it was catching is asserted directly instead, by
            # `test_the_merged_classifier_panel_loses_no_setting_to_the_rebuild`.
            # A silent net that nobody has ever seen fire is not evidence.
            ordered = rebuilt

        moved = {key for keys in ordered.values() for key in keys}
        leftovers = []
        for keys in result.values():
            leftovers.extend(key for key in keys if key not in moved)
        if leftovers:
            ordered["Additional Settings"] = list(dict.fromkeys(leftovers))
        result = ordered
    if app_key == "external_masks":
        filter_keys = (
            "uninfected", "cell_min_size", "cytoplasm_min_size",
            "nucleus_min_size", "pathogen_min_size", "organelle_min_size",
            "merge_edge_pathogen_cells",
        )
        for keys in result.values():
            for key in filter_keys:
                while key in keys:
                    keys.remove(key)
        reordered: Dict[str, List[str]] = {}
        for name, keys in result.items():
            reordered[name] = keys
            if name == "Measurements":
                reordered["Filter settings"] = list(filter_keys)
        result = reordered
    # Last, so it catches a key wherever it ended up -- including the
    # "Additional Settings" bucket, which is where a key goes precisely
    # when no layout claimed it.
    return _drop_hidden_keys(app_key, result)


# ---------------------------------------------------------------------------
# Category help — one blurb per settings CATEGORY
# ---------------------------------------------------------------------------
#
# A category is a collapsible header in a module's settings panel; the map
# above decides which keys land under which header. These are the blurbs the
# panel shows for the header itself, keyed by the title uppercased and
# stripped, because that is what a rendered ``Section`` has in hand.
#
# They are deliberately NOT restatements of the heading. Someone reading
# "Image Preprocessing" already knows the words; what they cannot tell is what
# the group decides and whether today's problem lives inside it. Each entry
# therefore says what the settings determine and when you would open them.
#
# ``CATEGORY_TOOLTIPS_BY_APP`` overrides this table for the handful of
# headings that genuinely mean different things per module: "Cellpose" is a
# training schedule under Train Cellpose and a set of inference thresholds
# under Cellpose Masks, and "Runtime & Reliability" carries Timelapse's stage
# toggles but only ``n_jobs`` under Motility.
#
# ``app_screen`` re-exports this as ``SECTION_HINTS`` for the tests and
# integrations that already read it by that name.
CATEGORY_TOOLTIPS: Dict[str, str] = {
    # -- shared headings from spacr.settings.categories --------------------
    "PATHS":
        "Where the module reads its images or tables from, plus any lookup "
        "file it needs alongside them. Set these when you point the module "
        "at a new plate or experiment; every other group assumes they are "
        "right.",
    "GENERAL":
        "The few decisions the rest of the run depends on: which channel is "
        "which, whether intensities are normalised, and whether preview "
        "figures are drawn. Worth a look on any dataset you have not run "
        "before.",
    "CELL":
        "How the cell mask is found — model, expected diameter, probability "
        "and flow thresholds. What is done to the channel first, and which "
        "of the masks it produces are kept, are under Advanced settings, "
        "where the same choices for the other objects sit beside them. Open "
        "it when cells are missed, merged into their neighbours, or split "
        "in two.",
    "NUCLEUS":
        "How the nucleus mask is found — model, expected diameter, "
        "probability and flow thresholds; the channel preprocessing and the "
        "size filters are under Advanced settings. "
        "Nuclei are the easiest object to get right, so they are a good "
        "place to check the channel assignment.",
    "PATHOGEN":
        "How the pathogen mask is found — model, expected diameter, "
        "probability and flow thresholds; the channel preprocessing and the "
        "size filters are under Advanced settings. "
        "Tightly packed parasites fusing into one object are the usual "
        "reason to come here.",
    "ORGANELLE":
        "The six choices you need to segment an organelle: which channel it "
        "is in, what KIND of organelle it is, how big it is, and the size "
        "and border filters. Setting the type fills in the detection "
        "parameters for you and says on the console what it picked — the "
        "rest are under Organelle advanced, still editable, if you want to "
        "change any of them.",
    "ADVANCED SETTINGS":
        "The umbrella over the settings that are one decision applied to "
        "several objects — what is done to the pixels before segmentation, "
        "which detected objects are kept, and how intensity decides "
        "splitting and merging. Each group inside it is broken down per "
        "object, so the same choice for cells and for nuclei sits side by "
        "side instead of under two unrelated headings. Nothing here needs "
        "touching on a first run.",
    # PER OBJECT, and the parenthetical is load-bearing: "IMAGE
    # PREPROCESSING" is already this table's key for the whole-image step
    # mask and timelapse render, and a second heading spelled the same way
    # would silently serve that blurb instead of this one.
    "IMAGE PREPROCESSING (PER OBJECT)":
        "What is done to each object's own channel before anything is "
        "segmented — the background floor below which pixels are zeroed, "
        "the signal-to-noise ratio that sets where the contrast stretch "
        "tops out, and, for organelles, rolling-ball flattening and CLAHE. "
        "The objects do not all offer the same steps, and each sub-heading "
        "shows exactly the ones its object has.",
    # The workflow-ordered layouts render these under a longer title, and
    # the tooltip table is keyed on the heading's EXACT text -- which is the
    # trap the "Computer Vision — " prefix fell into and why instruction 73
    # says to write the blurbs in the same change.
    "OBJECT FILTRATION (ALL OBJECTS)":
        "Which detected objects are kept, for every object class in one "
        "place. `cell_min_size` and `nucleus_min_size` do the same thing to "
        "different objects, so they are one decision applied once per "
        "object rather than a row of unrelated knobs — the settings are "
        "ordered by "
        "object, so each group reads together. Raise the minimum size to "
        "drop debris, set a maximum to drop merged clumps, and use the "
        "border filters when objects cut off by the image edge would bias "
        "your measurements.",
    "INTENSITY HANDLING (ALL OBJECTS)":
        "How object intensity decides splitting, merging and inclusion, for "
        "every object class in one place. The percentiles set the window "
        "that intensities are read against; merge and split use intensity "
        "to join objects the segmentation cut apart or separate ones it ran "
        "together. Open this when the masks look right but the objects are "
        "systematically over- or under-segmented.",
    "OBJECT FILTRATION":
        "Which detected objects are kept, for every object class in one "
        "place. `cell_min_size` and `nucleus_min_size` do the same thing to "
        "different objects, so they are one decision applied once per "
        "object rather than a row of unrelated knobs — the settings are "
        "ordered by "
        "object, so each group reads together. Raise the minimum size to "
        "drop debris, set a maximum to drop merged clumps, and use the "
        "border filters when objects cut off by the image edge would bias "
        "your measurements.",
    "INTENSITY HANDLING":
        "How object intensity decides splitting, merging and inclusion, for "
        "every object class in one place. The percentiles set the window "
        "that intensities are read against; merge and split use intensity "
        "to join objects the segmentation cut apart or separate ones it ran "
        "together. Open this when the masks look right but the objects are "
        "systematically over- or under-segmented.",
    "ORGANELLE ADVANCED":
        "The forty-eight detection parameters behind the organelle type: "
        "shape family and method, the background and contrast correction "
        "applied first, the knobs belonging to the method chosen, and the "
        "intensity filters applied to what was found. Choosing an organelle "
        "type sets the ones that matter for it; anything you change here "
        "wins and is never overwritten. Punctate, tubular and ring-shaped "
        "organelles each want a different method, which is what the type is "
        "choosing for you.",
    "ORGANELLE SEGMENTATION (ADVANCED)":
        "The forty-eight detection parameters behind the organelle type. "
        "Choosing a type sets the ones that matter for it; anything you "
        "change here wins and is never overwritten.",
    "CELLPOSE":
        "How Cellpose itself is run: expected object diameter, probability "
        "and flow thresholds, rescaling and inversion. Reach for these when "
        "masks are systematically too many, too few or the wrong size; "
        "which model runs is chosen under Model Training.",
    "SEGMENTATION QC":
        "Automatic pass/fail checks on the finished masks — object counts, "
        "size and split ratios, border and foreground fractions, and how "
        "much of a plate may fail before the run is called off. Tighten "
        "them once you know what a good field looks like; loosen them when "
        "a legitimately unusual plate keeps being rejected.",
    "MEASUREMENTS":
        "Which objects are measured and which features are computed for "
        "them — intensity, morphology, texture, radial distribution and "
        "colocalisation. Switch families off to keep the table narrow and "
        "the run short; switch them on when an analysis needs a column that "
        "is not there.",
    "FILTER SETTINGS":
        "Which segmented objects survive into the measurement table: the "
        "minimum size per compartment, whether uninfected cells are kept, "
        "and whether a pathogen straddling two cells merges them. Change "
        "them when debris is being measured, or when real cells vanish.",
    "OBJECT CROPS":
        "The per-object images written next to the measurements — crop mode "
        "and size, which mask each crop is centred on, how far it is "
        "dilated, and which channels are baked in. Annotate and the CV "
        "classifier read these later, so set them before generating a "
        "training set.",
    "PLATE LAYOUT & CONTROLS":
        "The plate map: which wells hold which cell line, strain and "
        "treatment, which are the positive and negative controls, and how "
        "wells are grouped for reporting. Filled in once per plate design; "
        "everything downstream labels its results from it.",
    "TRAINING CLASSES":
        "What makes an object a member of a class: the basis (plate metadata, "
        "an annotation column, or both) and the Classes dict naming which "
        "value of which column each class is. Open it first — everything "
        "downstream is a model of whatever this says.",
    "COMPUTER VISION DATA SOURCE":
        "Where the training images come from and how they are cut. Loading "
        "reads crops that were already exported, selected by one path "
        "pattern; streaming cuts them from the merged arrays as training "
        "runs, which needs the channel arrays and either an object table or "
        "a mask array to cut around. The settings that do not apply to the "
        "chosen source and stream method are greyed rather than hidden.",
    "COMPUTER VISION MODEL":
        "Which architecture, and how its input is scaled. A custom model path "
        "that loads supersedes the model type. Normalisation matters more "
        "than it looks: a pretrained backbone expects the statistics it was "
        "trained with.",
    "COMPUTER VISION TRAINING":
        "How the model is fitted — epochs, learning rate, schedule, and which "
        "loss. Open it when training is unstable, stalls, or ignores the "
        "smaller class.",
    "COMPUTER VISION OPTIMIZATION AND REGULARIZATION":
        "What keeps the model from memorising the training set: dropout, "
        "weight decay, gradient checkpointing. Reach for these when training "
        "accuracy climbs and validation accuracy does not.",
    "MODEL EVALUATION":
        "Cross-validation design used to estimate generalization performance: "
        "whether validation runs, the number of folds, and the grouping unit. "
        "Grouped folds keep related observations in the same partition and "
        "reduce information leakage. These settings apply to both classifier "
        "families.",
    "EVALUATION REPORTS":
        "Outputs produced after model evaluation: the metric bundle, "
        "calibration curve and bins, output score column, and decision "
        "threshold. Calibration compares predicted probabilities with "
        "observed frequencies. These settings apply to both classifier "
        "families.",
    "LEAKAGE AUDIT":
        "Checks whether training and test partitions share objects or "
        "identical content, with configurable detection and failure behavior. "
        "Partition overlap invalidates held-out performance estimates because "
        "the evaluation then includes data seen during training.",
    "MACHINE LEARNING MODEL AND FEATURES":
        "The feature-based classifier: which model, and which measured "
        "features it is allowed to see. Feature preparation and feature "
        "importance are one heading because they answer one question.",
    "IMAGES & CROPPING":
        "Where the training images come from and how they are cut — the crop "
        "source, the path and format filters for crops already on disk, and "
        "the channels and object to cut around for crops made on demand.",
    "MODEL & REGULARIZATION":
        "Which architecture, how its input is normalised, and what keeps it "
        "from memorising the training set. A custom model path that loads "
        "supersedes the model type.",
    "TRAINING & LOSS":
        "How the model is fitted: epochs, learning rate, schedule, and which "
        "loss. Open it when training is unstable, stalls, or ignores the "
        "smaller class.",
    # RESTORED. These two were deleted on 2026-08-12 as unreachable, on the
    # strength of an app list that did not include `classify_merged` -- which
    # renders BOTH of them. The list came from
    # `test_every_qt_section_hint_names_a_real_category`, whose own comment
    # says it has to be exhaustive rather than representative, and it was
    # neither. The test now includes classify_merged, so deleting a live
    # blurb on that evidence again fails instead of shipping.
    "CLASSIFIER":
        "Which family of classifier runs — a computer-vision network trained "
        "on the object images, or a tabular model trained on the measurements "
        "already in the database. This is the top-level choice: it decides "
        "which of the groups below apply.",
    "MACHINE LEARNING — MODEL & FEATURES":
        "The tabular model and the feature table it learns from — which "
        "estimator, how much of the data is held back, and the pruning that "
        "decides which measured features survive. Open it when the model "
        "overfits, or when thousands of correlated features are drowning the "
        "few that matter.",
    "EVALUATION & RESULTS":
        "How the fitted model is judged and how the result is shown — "
        "cross-validation, calibration, the leakage audit, the heatmap, and "
        "where the scores are written. Shared by both classifier families.",
    "EMBEDDING & CLUSTERING":
        "How the feature table is reduced to two dimensions and clustered "
        "on top of that — neighbourhood size, distance metric, and the "
        "DBSCAN/KMeans parameters with their noise handling. Change these "
        "when the embedding is one undifferentiated blob, or shatters into "
        "dozens of tiny clusters.",
    "DIMENSIONALITY REDUCTION":
        "Choose the reducer and the shared random seed and distance metric. "
        "The method-specific groups below grey themselves automatically.",
    "UMAP":
        "UMAP-only neighbourhood and minimum-distance controls. These values "
        "are retained but greyed whenever another reducer is selected.",
    "T-SNE":
        "t-SNE-only neighbourhood scale and optimisation controls.",
    "PCA":
        "PCA-only whitening and decomposition-solver controls.",
    "ISOMAP":
        "Isomap-only graph-neighbourhood and shortest-path controls.",
    "SPECTRAL EMBEDDING":
        "Spectral-only affinity graph and neighbourhood controls.",
    "POINTS & IMAGES":
        "How points, outlines and image thumbnails are rendered. These "
        "presentation controls never refit or move the embedding.",
    "CANVAS & OUTPUT":
        "Canvas dimensions, background and figure-saving controls.",
    "ACTIVATION MAPS":
        "Attribution settings for a trained image model — which method, "
        "which layer is hooked, how the map is overlaid, and the "
        "normalisation applied at inference. Open it when you want to know "
        "what the classifier is actually looking at.",
    "PLOT":
        "What is drawn from the results and how it looks — figure size, "
        "colour map, which control is shown alongside, and how many panels "
        "are produced. Cosmetic: it changes the figures, never the numbers.",
    "TIMELAPSE":
        "Linking masks of the same object across frames when the data has a "
        "time axis. Only relevant to a time series; a single-timepoint "
        "plate ignores it.",
    "ADVANCED":
        "Run-level knobs that rarely need touching — verbosity, worker and "
        "batch sizing, background handling, and whether results are written "
        "at all. Come here to make a run quieter or lighter on the machine, "
        "or to keep a scratch run from saving anything.",
    "3D SETTINGS (BETA)":
        "Experimental volumetric handling: how the z-axis is read, whether "
        "planes are projected or stitched, and the physical voxel size used "
        "for calibration. Needed only for z-stacks — and the voxel size is "
        "what makes a 3-D measurement physically meaningful.",
    "4D SETTINGS (BETA)":
        "Experimental time-plus-volume handling: how the time axis is laid "
        "out, the interval between frames, which backend links objects, and "
        "how far one may move between frames. For data that is both a "
        "z-stack and a time series.",
    "MOTILITY (BETA)":
        "The beta motility assay run inline with the mask pipeline: whether "
        "it runs at all, and the per-object tracking parameters it uses. "
        "The standalone Motility Assay module is the fuller version of the "
        "same analysis.",
    "MOTILITY ADVANCED (BETA)":
        "Fine-grained control over the beta motility pipeline — which "
        "features are selected and the filter windows applied to tracks. "
        "Only worth opening once the basic assay runs and the tracks look "
        "wrong in a specific way.",
    # The Qt regression layout's own section names. The shared
    # spacr.settings.categories names for the same six groups are the
    # "REGRESSION: ..." entries below; both maps are rendered, so both need a
    # curated blurb or the section shows the generic fallback.
    "RESPONSE":
        "What is being modelled: which score column (or columns — name "
        "several and each is fitted and corrected as its own family), "
        "whether one row is a well or a single cell, and how the values are "
        "collapsed and transformed before the model sees them.",
    "PERMUTATION TEST":
        "Read only when inference resolves to the nonparametric test. These "
        "control the permutation itself: how many, what is held fixed "
        "(normally the plate), the random seed, and how many wells a guide "
        "must appear in before it is testable.",
    "REGRESSION: RESPONSE":
        "What is being modelled: which score column (or columns — name "
        "several and each is fitted and corrected as its own family), "
        "whether one row is a well or a single cell, and how the values are "
        "collapsed and transformed before the model sees them.",
    "MODEL & INFERENCE":
        "How the effect is estimated AND what counts as a hit. 'Inference' "
        "is the top-level choice, 'Regression type' selects the family, and "
        "'Level' says whether the guide model, the gene model or both are "
        "fitted. Below them: the multiple-testing correction applied across "
        "the tested family, the level it targets, and the control-based "
        "effect-size threshold. With hundreds of guides an uncorrected P "
        "value is not evidence, so this is the section to get right.",
    "REGRESSION: MODEL":
        "How the effect is estimated. 'Inference' is the top-level choice: a "
        "parametric model fits every guide simultaneously, a nonparametric "
        "one tests each guide by plate-blocked permutation, and 'auto' picks "
        "whichever the design can actually support — a simultaneous fit needs "
        "more wells than guides. 'Regression type' then selects the family.",
    "REGRESSION: MODEL TUNING":
        "Per-family knobs. Each applies to only some regression types, and a "
        "family refuses a setting it cannot read rather than ignoring it, so "
        "nothing here changes a fit silently. Leave them alone unless the "
        "chosen family documents the one you are changing.",
    "REGRESSION: PERMUTATION TEST":
        "Read only when inference resolves to the nonparametric test. These "
        "control the permutation itself: how many, what is held fixed "
        "(normally the plate), the random seed, and how many wells a guide "
        "must appear in before it is testable.",
    "REGRESSION: SIGNIFICANCE":
        "What counts as a hit: the multiple-testing correction applied across "
        "the tested family, the level it targets, and the control-based "
        "effect-size threshold. With hundreds of guides an uncorrected P "
        "value is not evidence, so this is the section to get right.",
    "REGRESSION: QUALITY FILTERS":
        "Everything that decides which rows reach the model — minimum cells "
        "per well, minimum observations per guide, the read-fraction cutoff "
        "and outlier removal. Each one silently shrinks the dataset, so "
        "check the diagnostics after changing any of them.",
    "REGRESSION: DIAGNOSTICS":
        "Not what was significant, but whether the fit deserves to be "
        "believed: variance homogeneity, residuals, the design that actually "
        "reached the model, influence and calibration. Written per fit as "
        "figures, a combined PDF and a text report. On by default because a "
        "single analysis should have them; a parameter sweep turns them off "
        "on its own, since a hundred trials is two thousand files.",
    "INVASION ASSAY":
        "The two-colour invasion readout: which channels carry the outside "
        "and total stains, how the outside signal is measured, how its "
        "threshold is chosen and sanity-checked, and which objects count as "
        "parasites at all. The table the parasites are read from is under "
        "Measurements.",
    "SEQUENCING":
        "How reads become barcode counts — read mode and direction, the "
        "target sequence and regex, where the barcode starts and ends, "
        "chunk size, and how the output is compressed. Match these to how "
        "the library was built and how it was sequenced.",
    "REPLICATION ASSAY":
        "How parasites are assigned to vacuoles and counted into "
        "replication states, including the warning raised when a vacuole "
        "holds a biologically implausible, non-power-of-two number of "
        "parasites.",
    "ENDODYOGENY SIZE PROXY (LEGACY)":
        "The older area-bin approximation of replication state, kept so "
        "historical analyses still reproduce. New runs should use the "
        "direct parasite-per-vacuole counts instead.",
    # -- Mask / Timelapse --------------------------------------------------
    "INPUT & METADATA":
        "The image folder, which channel holds which object, and how spaCR "
        "reads plate, well and field out of the file names. Nothing "
        "segments correctly until the channel assignment and the naming "
        "convention here are right.",
    "WORKFLOW & TEST RUN":
        "Which stages actually execute, whether this is a small test pass "
        "over a few fields, and whether an interrupted run picks up where "
        "it stopped. Start every new dataset here with a test run before "
        "committing to the full plate.",
    "IMAGE PREPROCESSING":
        "What happens to the pixels before any mask is made — intensity "
        "normalisation, projection, upscaling, denoising, and how fields "
        "are batched. Reach for it when the images are dim, noisy, or at a "
        "different scale from the one the model expects.",
    "CELL SEGMENTATION":
        "Everything that produces the cell mask: model and expected "
        "diameter, probability and flow thresholds, background removal, and "
        "the size, intensity and border filters applied afterwards. The "
        "group to open when cells are missed, merged or split.",
    "NUCLEUS SEGMENTATION":
        "Everything that produces the nucleus mask: model and expected "
        "diameter, thresholds, background removal, and the size, intensity "
        "and border filters applied afterwards. Usually the easiest object "
        "to get right, so a good sanity check on the channel assignment.",
    "PATHOGEN SEGMENTATION":
        "Everything that produces the pathogen mask: model and expected "
        "diameter, thresholds, background removal, and the size, intensity "
        "and border filters applied afterwards. Parasites packed into one "
        "vacuole fusing into a single object is the usual reason to come "
        "here.",
    "ORGANELLE SEGMENTATION":
        "Everything the organelle mask needs, in the order you set it up: "
        "shape family and detection method, the background and contrast "
        "correction applied first, the knobs belonging to the method you "
        "chose (adaptive, spot, ridge, ring, irregular, Cellpose or U-Net), "
        "the size, intensity and border filters applied to what was found, "
        "and which parent compartment the results are summarised into. The "
        "largest group in the module, because punctate, tubular and "
        "ring-shaped organelles each want a different method.",
    "QUALITY CONTROL":
        "Automatic pass/fail checks on the finished masks — object counts, "
        "size and split ratios, border and foreground fractions, and how "
        "much of a plate may fail before the run is called off. Tighten "
        "them once you know what a good field looks like; loosen them when "
        "an unusual but legitimate plate keeps being rejected.",
    "VOLUMETRIC PROCESSING (BETA)":
        "How a z-stack is turned into something segmentable — whether "
        "planes are projected or stitched, which axis is z, and the "
        "physical voxel size. Ignore it entirely for single-plane data.",
    "TIME AXES & TRACKING (BETA)":
        "How the time axis is read and, experimentally, how objects are "
        "linked between frames. The full tracking workflow is the Timelapse "
        "module; this is the inline version.",
    "VISUALIZATION & DIAGNOSTICS":
        "The diagnostic figures a run draws as it goes — how many example "
        "fields, at what size, with which colour map and normalisation. "
        "Useful while tuning, and the first thing to switch off for a long "
        "unattended run.",
    "OUTPUT & STORAGE":
        "What survives the run: which masks and images are written, which "
        "intermediates are kept, how arrays are compressed, and whether "
        "objects are filtered or merged on the way out. Disk usage is "
        "decided here.",
    "RUNTIME & RELIABILITY":
        "How hard the run pushes the machine and what it does when a field "
        "fails — worker count, batch size, the tolerated failure rate, and "
        "how much it prints. Turn strict errors on while debugging; raise "
        "the failure tolerance for a plate with known-bad fields.",
    "ACQUISITION & AXES":
        "How the file's dimensions map onto time and z, the interval "
        "between frames, and the physical voxel size. Getting the axis "
        "order right is the prerequisite for any tracking, and everything "
        "downstream inherits it.",
    "TRACKING SETUP":
        "Which objects are tracked, over which range of frames, whether "
        "short-lived tracks are discarded, and the frame rate of the movies "
        "that come out. Start here, then pick a linker under Tracking "
        "Backends.",
    "TRACKING BACKENDS":
        "Which algorithm links objects between frames — Trackastra, Ultrack "
        "or a plain distance/overlap linker — and the parameters belonging "
        "to whichever you pick. Switch backends when cells swap identities "
        "or tracks break at division.",
    # -- Measure -----------------------------------------------------------
    "INPUT & EXPERIMENT":
        "The folder holding the masked images and the experiment name the "
        "measurements are filed under. Set once at the start of a "
        "measurement run.",
    "MASK & CHANNEL MAPPING":
        "Which plane of the stack holds each mask and each intensity "
        "channel, whether a cytoplasm compartment is derived, and whether "
        "the data is a time series. A wrong index here quietly measures the "
        "wrong object, so it is worth checking twice.",
    "MEASUREMENT FEATURES":
        "Which families of measurement are computed for every object — "
        "intensity, morphology, texture, radial distribution and "
        "colocalisation, with their parameters. More features means a wider "
        "table and a longer run, so enable what the analysis needs.",
    "OBJECT FILTERING":
        "Which objects are large enough, infected enough or clean enough to "
        "be measured at all. Raise the minimum sizes when debris is being "
        "counted; lower them when small but real objects disappear.",
    "CROP OUTPUT":
        "The per-object PNGs and arrays written alongside the measurements "
        "— crop mode and size, which channels and masks are included, "
        "dilation, and how they are normalised. These are the images "
        "Annotate and the CV classifier read later.",
    "PREVIEW & DIAGNOSTICS":
        "The small test run and the plots used to check a configuration "
        "before committing to a whole plate. The fastest way to find out "
        "that a channel index is wrong.",
    "3D CALIBRATION (BETA)":
        "The physical size of a voxel and the anisotropy between z and xy. "
        "Only these turn volumetric measurements from pixel counts into "
        "real units.",
    # -- Motility ----------------------------------------------------------
    "OBJECTS & CHANNELS":
        "The measurement source, which tracked object the assay is about, "
        "and which channels carry the cell, nucleus and pathogen signal. "
        "The rest of the assay is only as good as this mapping.",
    "SPATIAL & TEMPORAL CALIBRATION":
        "Pixel size and seconds per frame — the two numbers that convert "
        "movement in pixels into micrometres per second. Wrong here means "
        "every speed in the report is wrong by a constant factor.",
    "MOTION FILTERING":
        "The rules that keep implausible tracks out of the result — the "
        "largest jump allowed between frames, how straight a path has to "
        "be, and the outlier cutoff. Tighten them when tracking errors show "
        "up as impossibly fast cells.",
    "INFECTION CLASSIFICATION":
        "How a tracked cell is called infected, uninfected or ambiguous — "
        "which strategy is used, which table it reads, and where the "
        "probability cutoffs sit. The strategy chosen here decides which of "
        "the groups below actually apply.",
    "XGBOOST INFECTION MODEL":
        "Training and tree parameters for the supervised infection "
        "classifier, plus the probability threshold and margin that turn "
        "its output into a call. In play only when the strategy above is "
        "the XGBoost one.",
    "INFECTION CLUSTERING":
        "The unsupervised alternative: how many clusters, how the pathogen "
        "channel is weighted, and the minimum separation and silhouette a "
        "split has to reach before it is trusted. Use it when there are no "
        "labels to train on.",
    "EMBEDDING SEARCH":
        "The UMAP and t-SNE parameter ranges searched while trying to "
        "separate infected from uninfected phenotypes. Widen the grids when "
        "nothing separates the groups; fix single values to make a result "
        "reproducible.",
    "MOTILITY PLOTS & QC":
        "Axis limits and the diagnostic graphs used to review track quality "
        "and the infection call. Look here first when the summary numbers "
        "are surprising.",
    # -- Classify (CV) -----------------------------------------------------
    "PLATE SOURCES & WORKFLOW":
        "Which plates the classifier is built from, the experiment it is "
        "filed under, and which stages run — build the training set, train, "
        "test. Uncheck the stages you have already done to re-run only the "
        "part you are iterating on.",
    "LABELS & CLASSES":
        "Where the labels come from and what they mean — an annotation "
        "column or well metadata, the class names, and the measurement that "
        "defines them. Everything the model learns rests on this being the "
        "label you think it is.",
    # -- Classify (ML) -----------------------------------------------------
    # Named to match Classify (CV)'s group of the same purpose. The two
    # modules did the same job under different words, which is what made a
    # settings CSV non-portable between them.
    "LABELS & CLASSES":
        "What defines a class. Pick the training basis first — metadata "
        "(the wells named as positive and negative control), annotation (a "
        "column the Annotate module wrote), or measurement (thresholds on "
        "measured features). The controls the other two bases use are "
        "greyed out, not hidden: they keep their values. Get this wrong and "
        "every number downstream is meaningless, so check it first.",
    "FEATURE PREPARATION":
        "Which measurement columns are allowed into the model, and the "
        "variance, correlation, object-count and compartment filters "
        "applied before fitting. Prune here when the feature table is wide, "
        "redundant, or contains a column that leaks the answer.",
    "PLATE & BATCH CORRECTION":
        "Whether per-plate offsets are removed before analysis, which "
        "column identifies the batch, and which wells anchor the "
        "correction. Use it when plates were run on different days or "
        "instruments and plate identity shows up as a larger effect than "
        "the biology.",
    "CLASSIFIER & VALIDATION":
        "The estimator itself and how honestly it is scored — algorithm, "
        "learning rate and regularisation, held-out fraction and "
        "cross-validation. Change these when the model overfits, or when "
        "the reported accuracy looks too good to be true.",
    "FEATURE SELECTION & IMPORTANCE":
        "Whether features are pruned before the final fit, and how repeated "
        "permutation importance is computed afterwards. This is the part "
        "that answers which measurements the decision is actually based on.",
    "OUTPUT & DATABASE":
        "Whether model scores are written back into the measurements "
        "database so later modules can read them. Leave it off for "
        "exploratory fits you would rather not record.",
    "PLOTS & HEATMAPS":
        "Which feature the heatmap shows, how wells are grouped, and the "
        "colour map and value range used to draw it. Presentation of the "
        "classifier's output; it does not change the fit.",
    # -- Regression --------------------------------------------------------
    "INPUT TABLES":
        "The metadata, score and count tables the regression runs on. All "
        "three have to agree on well and gRNA naming — disagreement there "
        "is the usual cause of an empty result.",
    # "CONTROLS & PLATE DESIGN", "QUALITY FILTERS" and "SIGNIFICANCE & HIT
    # CALLING" were retired on 2026-08-17 with instruction 135: the first two
    # merged into "CONTROLS & FILTERS" and the third into "MODEL &
    # INFERENCE". Their hints merged with them rather than being dropped.
    "CONTROLS & FILTERS":
        "Which rows reach the model, and what they are measured against. The "
        "plate identifier, the positive and negative control wells, any row "
        "filter — and every cutoff that drops data: minimum cells per well, "
        "minimum observations per guide, the read-fraction cutoff and "
        "outlier removal. The controls set the scale the effect sizes are "
        "reported on; each cutoff silently shrinks the dataset, so check the "
        "diagnostics after changing one.",
    # "MODEL & COVARIATES", "HIT CALLING & OUTLIERS" and the flat "REGRESSION"
    # heading were retired when the regression layout was split into Response
    # / Model & Inference / Estimator Tuning / Permutation Test / Significance
    # & Hit Calling / Quality Filters. Their replacements are above.
    "ADDITIONAL SETTINGS":
        "The remaining knobs belonging to individual regression families "
        "and plots — bootstrap counts, quantile and hinge parameters, "
        "solver tolerance and axis limits. Only the ones for the model you "
        "chose above have any effect.",
    # -- Activation --------------------------------------------------------
    "MODEL & DATA":
        "The trained model, the dataset it is applied to, and the input "
        "channels, object type and image size it expects. These have to "
        "match how the model was trained or the maps mean nothing.",
    "ATTRIBUTION METHOD":
        "Which algorithm explains the prediction — Grad-CAM, SmoothGrad, "
        "occlusion or integrated gradients — which layer it hooks, and the "
        "parameters of whichever you pick. Methods disagree; comparing two "
        "is often more informative than tuning one.",
    "ATTRIBUTION VALIDATION":
        "The checks that separate a real explanation from a pretty picture "
        "— insertion and deletion steps, the baseline they are measured "
        "against, and the model-weight sanity check. Worth running before "
        "an attribution map goes into a figure.",
    "MAP DISPLAY":
        "How the finished map is rendered — input and map normalisation, "
        "overlay on the source image, and whether it is plotted at all. "
        "Presentation only.",
    "MAP QUANTIFICATION":
        "Turning a map into numbers: channel correlation and the Manders "
        "thresholds used to ask how much of the attribution sits on a given "
        "structure.",
    "OUTPUT & RUNTIME":
        "Whether maps are saved, whether the input order is shuffled, and "
        "the batch size and worker count used to generate them.",
    # -- Replication -------------------------------------------------------
    "ASSAY INPUTS":
        "The measurements database, the parasite table inside it, and the "
        "compartment the parasites were measured in. The assay scores "
        "existing measurements — it does not segment anything itself.",
    "VACUOLE ASSIGNMENT":
        "How individual parasites are grouped into vacuoles — an existing "
        "vacuole identifier, or a spatial link whose distance scales with "
        "parasite size — and whether a host cell is required. The whole "
        "replication readout rests on this grouping.",
    "CONDITION METADATA":
        "Which wells hold which cell line, strain and treatment, and the "
        "column and level the conditions are grouped and reported at.",
    "REPLICATION SCORING":
        "How grouped parasites become a replication state: the largest "
        "vacuole accepted, the warning for biologically implausible counts, "
        "and whether wells with cells but no parasites are seeded as zeros. "
        "Leaving those wells out silently inflates the mean.",
    "ASSAY OUTPUT":
        "Whether the assay's results and figures are written, and the "
        "colour map used to draw them.",
    # -- External Masks ----------------------------------------------------
    "INPUT MAPPING":
        "How externally generated images and label masks are found and "
        "paired — the input list, the project folder written to, recursion, "
        "plate and well layout, z handling and naming. Preview the mapping "
        "before writing anything; this is where a mismatched pairing is "
        "caught.",
    # -- shared by the three Cellpose-facing modules -----------------------
    #
    # Mask, Cellpose Masks, Cellpose All and Train Cellpose ask the same four
    # questions about a segmentation run in the same order. Naming the groups
    # identically is the point: someone who learned them once should not have
    # to relearn them in the next module.
    "INPUT & CHANNELS":
        "Where the images come from and which planes of each one the module "
        "actually looks at, plus whether they are normalised or inverted "
        "first. A run that finds nothing at all is usually a channel index "
        "pointing at an empty plane.",
    "MODEL":
        "Which weights do the segmenting — a packaged model, or a checkpoint "
        "of your own — and the object size they should expect. Nothing is "
        "trained here; this is the picker, and the expected size matters "
        "more than the choice of weights.",
    "DETECTION THRESHOLDS":
        "How much the model is allowed to find: the probability floor below "
        "which a candidate is discarded, how strictly flow has to agree, and "
        "whether holes are filled. Come here when there are too many objects, "
        "too few, or one blob where two cells belong.",
    "IMAGE GEOMETRY":
        "The pixel dimensions the images are resampled to before anything "
        "else happens. Getting this wrong rescales every object and quietly "
        "changes what the expected size means, so set it once per "
        "acquisition and leave it.",
    "BACKGROUND & DENOISING":
        "Correction applied before segmentation: the intensity floor treated "
        "as empty and the signal-to-noise gate a field has to clear. Raise "
        "the floor when autofluorescence is being segmented as objects; "
        "lower it when genuinely dim cells disappear.",
    # -- Train Cellpose ----------------------------------------------------
    "STARTING POINT":
        "What the training run begins from — a pretrained model fine-tuned "
        "on your data, or randomly initialised weights — and the name the "
        "result is saved under. Fine-tuning needs far fewer labelled images "
        "than starting from scratch.",
    "TRAINING SCHEDULE":
        "How long the fit runs and how fast it moves: epochs, learning rate, "
        "weight decay, batch size and augmentation. Reach for these when the "
        "loss stops falling early, or when the model memorises the training "
        "images instead of generalising.",
    # -- Map Barcodes ------------------------------------------------------
    "SEQUENCING INPUT":
        "The read files and whether they are treated as a pair or a single "
        "direction. Everything downstream assumes this is right, and a "
        "single-end run pointed at paired reads finds nothing without "
        "reporting an error.",
    "BARCODE REFERENCES":
        "The three lookup CSVs a read is matched against — gRNA, row and "
        "column. A mapping run that returns no counts at all is almost "
        "always one of these three pointing at the wrong file, or at a file "
        "written with different column names.",
    "READ PARSING":
        "How a barcode is located inside each read: the anchoring sequence, "
        "the regular expression around it, and where the match is expected "
        "to begin and end. Change these when the library was built with a "
        "different adapter layout.",
    # -- Barcode QC --------------------------------------------------------
    "REFERENCE & COUNT TABLES":
        "The barcode references and the counts produced by a mapping run, "
        "which the checks below are computed from. Point them at the outputs "
        "of the run you want to judge, not at a newer plate.",
    "WELL EXPECTATIONS":
        "What a healthy well should look like — how many distinct guides it "
        "ought to carry, which statistic that is judged by, and the read "
        "floor below which a well is not worth trusting. These set the bar "
        "that everything else is measured against.",
    "STARVATION & EXCLUSION":
        "How wells that received too few reads are detected and whether they "
        "are dropped before the rest of the analysis. Leaving them in drags "
        "every plate-level summary toward noise, so exclude them once you "
        "trust the read floor above.",
    "POSITION & COLLISION CHECKS":
        "Two systematic artefacts worth ruling out before believing a hit: "
        "counts that track a well's position on the plate, and barcodes "
        "close enough in sequence to be confused for one another. Both look "
        "like biology until they are checked.",
    "THRESHOLD SWEEP":
        "The range and resolution of the scan used to show how the results "
        "would change under a different cut-off. Widen the span when the "
        "chosen threshold sits near the edge of the scanned range.",
    "QC OUTPUT":
        "Where the report is written and whether figures are drawn and kept. "
        "Leave saving off while you are still deciding which checks matter "
        "for this library.",
    # Measure's single illumination heading. The Illumination screen's four
    # tabs keep their own blurbs below; this one covers all of them, because
    # under Measure they are one section.
    "ILLUMINATION CORRECTION":
        "Whether the microscope's uneven lighting is estimated from these "
        "fields and divided out before any intensity is measured, and how "
        "that estimate is made and checked. Turn it on when the same cell "
        "measures differently depending on where in the field it sat; "
        "leaving it off keeps that bias in every intensity feature.",
    # -- Illumination ------------------------------------------------------
    "CORRECTION MODEL":
        "How the uneven lighting field is estimated and removed — the family "
        "of surface fitted, the estimator behind it, its flexibility, and "
        "the dark reference subtracted first. Too flexible a surface absorbs "
        "real biological signal along with the shading.",
    "FIELD SAMPLING":
        "How many fields the correction is estimated from and whether each "
        "plate gets its own estimate. More fields make a steadier surface "
        "and a slower run; per-plate estimates matter when plates were "
        "acquired in separate sessions.",
    "QC & FAILURE HANDLING":
        "Whether the fitted surface is checked before being applied, and "
        "what happens when a plate has no usable estimate — skip it, or stop "
        "the run. Stopping is the safer choice the first time you correct an "
        "unfamiliar dataset.",
    # -- AnnData Export ----------------------------------------------------
    "OUTPUT FILE":
        "Where the exported object is written and how it is shaped: one "
        "matrix or one per table, the numeric precision kept, and the "
        "compression applied. Precision and compression trade file size "
        "against how faithfully the measurements survive the round trip.",
    "ROWS & MISSING VALUES":
        "How many rows are exported and what happens to gaps in them — kept "
        "as missing, dropped, or filled. Downstream tools differ sharply in "
        "what they tolerate, so this usually follows from whatever reads the "
        "file next.",
    "POST-PROCESSING":
        "Optional work done after the matrix is written: computing an "
        "embedding inside the exported object, and recording it as a run "
        "artifact so later steps can find it. Both are off by default "
        "because both cost time.",
    # -- Recruitment / Invasion -------------------------------------------
    "DATA SOURCE":
        "The measurements this module reads. One setting, and every group "
        "below assumes it is right — point it at the project folder a "
        "measure run wrote, not at the raw images.",
    "PLOTS & DIAGNOSTICS":
        "Whether preview figures are drawn, how large they are, and how many "
        "examples are produced. Worth turning on for the first plate of an "
        "experiment and off again once the numbers are trusted.",
    "CHANNELS & INTENSITY":
        "Which channels carry the signals the assay compares, the statistic "
        "each object is summarised by, and whether background is subtracted "
        "first. Swapping two channels here inverts the result without "
        "producing an error.",
    "THRESHOLDING":
        "How the cut-off separating the two populations is chosen, and how "
        "much disagreement between methods is tolerated before the run says "
        "so. This is the single most consequential group in the assay.",
    "CONTROLS & MINIMUM COUNTS":
        "Which wells anchor the threshold, and how many objects a well or a "
        "plate must contribute before its number is believed. Raise the "
        "minimums when sparse wells produce implausibly extreme rates.",
    # -- Regression --------------------------------------------------------
    "ESTIMATOR TUNING":
        "The knobs that belong to one estimator rather than to all of them — "
        "the elastic-net mixing ratio, the quantile being fitted, Huber's "
        "cut-off, the convergence tolerance, and the bootstrap counts behind "
        "the hinge and lasso selection thresholds. Only the ones matching "
        "the model chosen above have any effect.",
    "SOURCE & PROVENANCE":
        "The exact database, prediction, sequencing and result artifacts used "
        "by this run. Preserve these paths and hashes so the explanation can "
        "be reproduced instead of silently following the newest file.",
    "SURROGATE & VALIDATION":
        "The interpretable estimator and grouped held-out test used to decide "
        "whether it reproduces the CV model well enough to explain it.",
    "IMPORTANCE & DIAGNOSTICS":
        "Permutation, SHAP and correlation controls. These are only "
        "interpretable after the held-out fidelity gate passes.",
    "SELECTED HIT":
        "The gene, guides, direction and regression evidence carried forward "
        "from the exact selected result.",
    "ATTRIBUTION MODEL":
        "Cross-fit grouping, independent morphology features and probability "
        "threshold for hit-like candidates; these are not genotype calls.",
    "EVIDENCE & OUTPUT":
        "Well-level bootstrap/permutation evidence, blinded gallery sampling, "
        "versioned database storage and exported artifacts.",
    # -- Power / Design ----------------------------------------------------
    #
    # "Power analysis" is the single heading `spacr/qt/screens/power.py`
    # registers all fifteen of its keys under, which is what the settings
    # diff and the run journal group them by when the module's own screen is
    # not involved. The five headings below are what the layout splits it
    # into; this entry covers the undivided one.
    "POWER ANALYSIS":
        "Everything a screening design has to commit to before a plate is "
        "poured: library size and redundancy, how it is spread over plates "
        "and replicates, the effect worth detecting and how rare it is, "
        "sequencing depth, and how the estimate itself is simulated.",
    "LIBRARY DESIGN":
        "The size and redundancy of the screening library: how many genes "
        "are targeted, how many guides each one gets, and how many "
        "constructs land in a well. Guides per gene is usually the cheapest "
        "lever on detection power.",
    "PLATE LAYOUT":
        "How the library is spread over physical plates — wells per plate, "
        "plate count, replicates, and cells sampled per well. This is where "
        "a design becomes a number of plates somebody has to actually run.",
    "EFFECT & PREVALENCE":
        "What the screen is looking for and how rare it is: the effect size "
        "worth detecting, the fraction of genes expected to show it, the "
        "background rate underneath, and how well the readout separates a "
        "hit from a miss. Optimism here is the usual reason a real screen "
        "underperforms its power curve.",
    "SEQUENCING DEPTH":
        "How many reads each well is allotted. Too few and guide counts "
        "become noise before any biology is involved, which no amount of "
        "extra replicates recovers.",
    "SIMULATION":
        "How the estimate itself is produced — the level the score is "
        "computed at, the backend that runs it, and the random seed. Fix "
        "the seed when you want two designs compared rather than two draws.",
}


#: Per-module overrides for headings that mean different things per module.
#: Missing entries fall through to :data:`CATEGORY_TOOLTIPS`.
CATEGORY_TOOLTIPS_BY_APP: Dict[str, Dict[str, str]] = {
    "train_cellpose": {
        # Train Cellpose fits weights; the other three run them. "Model"
        # therefore names the thing being produced rather than the thing
        # being picked, which is a different sentence.
        "OUTPUT & RUNTIME":
            "How much the training run prints as it goes. Turn it up when a "
            "fit is diverging and the loss curve alone does not say which "
            "epoch it went wrong at.",
    },
    "cellpose_masks": {
        "OUTPUT & RUNTIME":
            "Whether the masks are written, how many images are handed to "
            "the GPU at once, and how much the run prints. Reduce the batch "
            "size when the GPU runs out of memory.",
    },
    "cellpose_all": {
        "MODEL":
            "The object size every candidate model is told to expect. The "
            "point of this module is that the models differ, so this is the "
            "one thing held constant while they are compared.",
        "OUTPUT & RUNTIME":
            "Whether the comparison figures and masks are written, the GPU "
            "batch size, and how much each candidate run prints on its way "
            "through.",
    },
    "analyze_plaques": {
        "MODEL":
            "The expected plaque diameter, and whether previously written "
            "masks are reused instead of segmenting again. Plaques are far "
            "larger than cells, so the default cell-sized expectation is "
            "almost never right here.",
        "OUTPUT & RUNTIME":
            "Whether masks and results are written, the GPU batch size, and "
            "how much the run prints. Leave saving off for the first pass "
            "over a new plate.",
    },
    "umap": {
        "INPUT DATA":
            "Choose the measurements database, tables and feature columns "
            "that enter the map, then exclude unwanted rows or redundant "
            "measurements before fitting anything.",
        "DIMENSIONALITY REDUCTION":
            "Choose the reducer and the shared random seed and distance "
            "metric. The method-specific groups below grey themselves "
            "automatically when another reducer is selected.",
        "UMAP":
            "Tune UMAP's neighbourhood size and minimum distance to trade "
            "fine local structure against a smoother view of global "
            "relationships.",
        "T-SNE":
            "Tune t-SNE's perplexity, learning rate, exaggeration and "
            "iteration budget when its neighbourhoods collapse or fail to "
            "separate.",
        "PCA":
            "Choose PCA whitening and its decomposition solver. Change these "
            "when component scales or the dimensions of a large table make "
            "the default solver unsuitable.",
        "ISOMAP":
            "Set Isomap's graph neighbourhood and shortest-path method. "
            "Change them when the manifold disconnects or bends across "
            "biologically separate populations.",
        "SPECTRAL EMBEDDING":
            "Choose how Spectral Embedding builds its affinity graph and how "
            "many neighbours connect it. Sparse or fragmented data usually "
            "needs this group.",
        "CLUSTERING":
            "Choose the clustering algorithm and its density or cluster-size "
            "controls, then decide whether noise is retained and which "
            "metadata colours the result.",
        "PLATE & BATCH CORRECTION":
            "Describe plate, control and covariate columns used to remove "
            "technical batch structure without treating real biological "
            "differences as nuisance variation.",
        "POINTS & IMAGES":
            "Control point, outline and crop-thumbnail rendering after the "
            "embedding is fitted. These presentation choices never move or "
            "refit a sample.",
        "CANVAS & OUTPUT":
            "Set canvas and sidebar dimensions, background colour and figure "
            "saving. Use these controls to prepare an export without changing "
            "the analysis.",
        "RUNTIME":
            "Set worker parallelism and diagnostic verbosity. Reduce workers "
            "when memory is constrained, or increase logging while tracing a "
            "failed run.",
    },
    "recruitment": {
        "MASK & CHANNEL MAPPING":
            "Which array plane holds each mask and each intensity channel, "
            "and which one the recruitment is measured on. A wrong index "
            "here measures the wrong compartment without complaining.",
        "OBJECT FILTERING":
            "The size and intensity windows an object has to fall inside to "
            "count, plus the per-well cell limits. These gates decide which "
            "cells the recruitment ratio is averaged over.",
        "PLATE LAYOUT & CONTROLS":
            "Which wells hold which cell line, strain and treatment, and "
            "which channel the recruitment is measured on. Filled in once "
            "per plate design.",
    },
    "invasion": {
        "ASSAY INPUTS":
            "Which measurement table the parasites are read from and which "
            "compartment they were measured in. The assay scores existing "
            "measurements rather than segmenting again.",
        "CONDITION METADATA":
            "Which wells hold which cell line, strain and treatment, and "
            "the column and level the invasion rates are grouped and "
            "reported at.",
        "ASSAY OUTPUT":
            "The colour map the assay's figures are drawn with, how many QC "
            "panels are produced, and whether wells with cells but no scored "
            "parasites are seeded as zeros. Leaving those wells out "
            "silently inflates the invasion rate.",
        "RUNTIME & RELIABILITY":
            "How much the assay prints as it runs. Turn it up while you are "
            "still deciding on a threshold and need to see which wells the "
            "controls were drawn from.",
    },
    "external_masks": {
        "GENERAL":
            "The experiment name, channel list, normalisation and whether a "
            "cytoplasm compartment is derived — the frame the imported "
            "masks are measured in. Check the channel list matches the "
            "images you are importing.",
        "TIMELAPSE":
            "Which objects are linked across frames when the imported data "
            "is a time series. Leave it alone for single-timepoint plates.",
        "MEASUREMENTS":
            "Which feature families are computed for the imported masks — "
            "intensity, texture, radial distribution and colocalisation. "
            "The expensive ones are off by default.",
        "ADVANCED":
            "Resume, failure tolerance, dry runs, worker count and "
            "verbosity for the import. Turn strict errors on the first time "
            "you import someone else's data.",
    },
    "timelapse": {
        "RUNTIME & RELIABILITY":
            "Which stages run, whether this is a small test pass, and how "
            "the run behaves under load and failure — workers, batch size, "
            "tolerated failure rate and verbosity. Track a few fields in "
            "test mode before committing to a whole plate.",
    },
    "motility": {
        "RUNTIME & RELIABILITY":
            "How many worker processes the assay uses. Lower it when the "
            "machine has other work to do.",
    },
    "ml_analyze": {
        "RUNTIME & RELIABILITY":
            "How many cores the fit is spread over, and how much it prints "
            "on the way. Lower the worker count when the machine has other "
            "work to do; raise the verbosity when a fit is failing and you "
            "cannot see where.",
    },
    "replication": {
        "OBJECT FILTERING":
            "The area window a segmented object has to fall inside to count "
            "as a parasite. Debris below it and clumps above it are "
            "excluded.",
        "RUNTIME & RELIABILITY":
            "How much the assay prints as it runs. Turn it up when a well "
            "comes out empty and you need to see which step discarded its "
            "parasites.",
    },
}


#: Family prefixes :func:`categories_for_app` puts in front of a merged
#: module's group titles, e.g. "Computer Vision — Training & Loss". The
#: tooltip tables are keyed on the UNPREFIXED name, so a lookup has to try
#: both: commit c41a75b6 added these prefixes and orphaned every blurb the
#: plain Classify module was already using, leaving six of Classify
#: (merged)'s nine headings describing themselves.
_CATEGORY_FAMILY_PREFIXES = ("COMPUTER VISION", "MACHINE LEARNING")

#: Dashes seen between a family prefix and the group name. Written out
#: because the em dash in the source is easy to lose in an edit and the
#: failure is silent — the lookup just misses.
_CATEGORY_PREFIX_DASHES = ("—", "–", "-")


def _category_blurb(app_key: str, title: str) -> str:
    """The written blurb for a category title, or ``""`` if there is none.

    Tries the module's own override then the shared table, first for the
    title as rendered and then for the title with a family prefix removed.
    """
    key = str(title or "").upper().strip()
    if not key:
        return ""
    candidates = [key]
    for prefix in _CATEGORY_FAMILY_PREFIXES:
        for dash in _CATEGORY_PREFIX_DASHES:
            marker = f"{prefix} {dash} "
            if key.startswith(marker):
                candidates.append(key[len(marker):].strip())
    overrides = CATEGORY_TOOLTIPS_BY_APP.get(str(app_key or ""), {})
    for candidate in candidates:
        text = overrides.get(candidate) or CATEGORY_TOOLTIPS.get(candidate, "")
        if text:
            return text
    return ""


def category_tooltip(
    app_key: str,
    title: str,
    language: Optional[str] = None,
) -> str:
    """Return the plain-language blurb for one settings category.

    Resolution order: the module's own override, then the shared table, then
    a generic sentence built from the title. The generic one is a *visible*
    fallback rather than an empty string so a brand-new category is never
    silently blank — ``tests/qt/test_category_tooltips.py`` fails on it.

    :param app_key: module the category is being rendered for.
    :param title: category title as shown on the header (any case).
    :param language: optional language override; defaults to the UI language.
    """
    if not str(title or "").strip():
        # An empty title has no fallback: "Settings that control ." is worse
        # than nothing, and a caller passing "" wants silence.
        return ""
    text = _category_blurb(app_key, title)
    if not text:
        text = f"Settings that control {str(title).lower().strip()}."
    return _translated_body(text, language, category=True)


#: Help for the per-object SUB-HEADINGS inside an advanced family.
#:
#: KEYED ON THE OBJECT, NOT ON THE FAMILY, and written to read under any of
#: them: the family heading above already says what the group decides, so the
#: sub-heading only has to say which object it decides it for, and what is
#: different about that object.
#:
#: A SEPARATE TABLE BECAUSE THE TITLES COLLIDE. A sub-heading titled "Cell"
#: under "Object filtration" is not the top-level "Cell" segmentation
#: category, and the shared table is keyed on the heading text alone -- so a
#: title-only lookup hands a filtration sub-heading the blurb about Cellpose
#: models and expected diameters. :func:`section_tooltip` tells them apart by
#: the section's PATH, which is the only thing that differs.
OBJECT_SUBHEADING_TOOLTIPS: Dict[str, str] = {
    "CELL": (
        "This group's decision as it applies to the cell mask -- the outer "
        "boundary every other object is assigned to. Changing it moves the "
        "denominator of every per-cell measurement, so it is the one to be "
        "most careful with."),
    "NUCLEUS": (
        "This group's decision as it applies to the nucleus mask. Nuclei are "
        "the roundest and best separated objects in a typical screen, so "
        "values that are far from the ones the other objects need usually "
        "mean the channel assignment is wrong rather than the filter."),
    "PATHOGEN": (
        "This group's decision as it applies to the pathogen mask. Parasites "
        "sit inside a host cell and often touch each other, so this is where "
        "a clump segmented as one object, or a vacuole counted as several, "
        "is dealt with."),
    "CYTOPLASM": (
        "This group's decision as it applies to the cytoplasm, which is not "
        "segmented at all -- it is the cell with the nucleus and the "
        "pathogens subtracted. A filter here therefore acts on what is left "
        "over, and follows whatever the other three were set to."),
}


def _organelle_subheading_tooltip(number: int) -> str:
    """Help for one organelle slot's sub-heading, written from its number.

    GENERATED, BECAUSE THE SLOTS ARE. Four of these were written out by hand,
    which was the whole complaint: the fifth slot a run may declare had no
    help at all and fell back to "Settings that control organelle 5", and the
    fourth one's text told the user it was "the last one spaCR offers" --
    true while the slots were fixed at four and a lie the moment the count
    became a setting.
    """
    if number == 1:
        return ("This group's decision as it applies to the first organelle "
                "slot. Organelles are the most varied objects spaCR handles, "
                "from diffraction-limited dots to a network filling the whole "
                "cell, so the useful values here depend on which kind was "
                "chosen.")
    if number == 2:
        return ("The same decision for the second organelle slot, which is "
                "an independent object with its own channel and its own "
                "type. A screen staining two organelles keeps their settings "
                "apart here rather than sharing one set of values between "
                "them.")
    return (f"The same decision for organelle slot {number}, an independent "
            "object with its own channel and its own type. It is defaulted "
            "from the first slot, so a screen using fewer organelles can "
            "ignore this heading without leaving anything unset, and it is "
            "only worth opening when this slot's channel is actually being "
            "segmented.")


#: Every slot gets one, for the reason the registries are generated for every
#: slot too: lowering `number_of_organelles` HIDES a slot rather than deleting
#: it, so a heading that can come back has to have help waiting when it does.
OBJECT_SUBHEADING_TOOLTIPS.update({
    organelle_slot_label(role).upper(): _organelle_subheading_tooltip(
        organelle_number(role))
    for role in ALL_ORGANELLE_ROLES
})


def section_tooltip(app_key: str, section, language: Optional[str] = None) -> str:
    """Return the blurb for one heading of the settings TREE.

    A nested heading is resolved by its :attr:`SettingsSection.path`, not by
    its title: "Cell" under "Object filtration" and the top-level "Cell"
    segmentation category are the same word for two different groups, and a
    title-only lookup would give the first one the second one's help.

    :param app_key: module the section is being rendered for.
    :param section: a :class:`SettingsSection`, or any ``(title, rows)``
        pair -- an un-nested pair resolves exactly as before.
    :param language: optional language override; defaults to the UI language.
    """
    path = tuple(getattr(section, "path", ()) or ())
    title = getattr(section, "title", None)
    if title is None:
        title = section[0] if isinstance(section, tuple) else str(section)
    if len(path) > 1:
        text = OBJECT_SUBHEADING_TOOLTIPS.get(str(title).upper().strip(), "")
        if text:
            return _translated_body(text, language, category=True)
    return category_tooltip(app_key, title, language)


def section_tooltip_is_curated(app_key: str, section) -> bool:
    """True when a tree heading has written help rather than the fallback."""
    path = tuple(getattr(section, "path", ()) or ())
    title = getattr(section, "title", None)
    if title is None:
        title = section[0] if isinstance(section, tuple) else str(section)
    if len(path) > 1 and str(title).upper().strip() in OBJECT_SUBHEADING_TOOLTIPS:
        return True
    return category_tooltip_is_curated(app_key, title)


def category_tooltip_is_curated(app_key: str, title: str) -> bool:
    """True when a category has a written blurb rather than the fallback.

    Shares :func:`_category_blurb` with :func:`category_tooltip` rather than
    repeating the lookup: the two used to hold separate copies, so a lookup
    rule added to one would silently not apply to the other.
    """
    return bool(_category_blurb(app_key, title))


def get_tooltips() -> Dict[str, str]:
    """Return per-key tooltip text (spacr.settings.descriptions and .tooltips)."""
    tips: Dict[str, str] = {}
    try:
        from spacr.settings import descriptions, tooltips
    except Exception:
        return tips
    tips.update({k: v for k, v in descriptions.items() if isinstance(v, str)})
    tips.update({k: v for k, v in tooltips.items() if isinstance(v, str)})
    return tips


# ---------------------------------------------------------------------------
# API doc link per app
# ---------------------------------------------------------------------------

DOCS_API_BASE = "https://einarolafsson.github.io/spacr/api"

_APP_API_MODULE = {
    # Registered without a mapping, so their help had no API page to link to.
    # The Volcano Explorer redraws a finished regression's coefficient table
    # and the Parameter Sweep reads the trials of a search that already ran;
    # each points at the module that produced what it is showing.
    # The Cells tab -- which objects a dot on the volcano is most consistent
    # with. Instruction 131; the answer is pure pandas in `cell_montage` and
    # the tab only loads what it names.
    "cell_montage": "cell_montage",
    "feature_dict": "feature_dict",
    # THE FOLDED MODULES. These three reached this table through
    # ``register_app(..., api_module=...)`` -- the push half of the seam
    # absorbed below -- so folding them into a host screen and dropping
    # the row would take the API link out of the hover help on every one
    # of their settings, and the folded page's help would point at the
    # generated API index instead of at the module that does the work.
    "barcode_qc": "sequencing_qc",
    "explain_cv": "surrogate",
    "anndata_export": "anndata_export",
    # Illumination reached this table from its own row too, and folded into
    # Measure. Its module is the one that estimates the flat field, so the
    # settings on the folded page point there rather than at the index.
    "illumination": "illumination",
    "volcano_explorer": "volcano_style",
    # Image Scatter and PCA reached this table the same way, from their own
    # rows. Both are folded onto Image UMAP now, and `unregister_app` takes a
    # pushed entry back out with the row it came from -- so without these two
    # lines the help on either screen falls back to the generated API index
    # instead of the page that documents it.
    "image_scatter": "qt/screens/image_scatter",
    "pca": "qt/screens/pca",
    # Curate the same way, from its own row into Make Masks. Its page is the
    # brush rather than a settings form, so nothing asks for its link
    # today; the line is here because the alternative is that the answer
    # silently became the generated API index the first time anything did.
    "curate": "qt/screens/curate",
    "parameter_sweep": "parameter_sweep",
    "align": "align",
    "convert": "convert",
    "foreign": "foreign",
    "queue": "qt/plate_queue",
    "batch": "batch",
    "db_browser": "qt/screens/db_browser",
    "mask": "core",
    "measure": "measure",
    "external_masks": "external_masks",
    "annotate": "qt/screens/annotate",
    "classify": "deep_spacr",
    "classify_merged": "classify",
    "map_barcodes": "sequencing",
    "umap": "core",
    "timelapse": "core",
    "motility": "timelapse",
    "ml_analyze": "ml",
    "regression": "ml",
    "activation": "deep_spacr",
    "make_masks": "qt/screens/make_masks",
    "train_cellpose": "submodules",
    "cellpose_masks": "spacr_cellpose",
    "cellpose_all": "spacr_cellpose",
    "model_compare": "model_compare",
    "model_zoo": "model_zoo",
    "plate_view": "plate_qc",
    "agreement": "agreement",
    "train_compare": "train_compare",
    "classifier_evaluation": "classifier_evaluation",
    "run_history": "run_journal",
    "report": "report",
    "distributed_jobs": "remote_execution",
    "recruitment": "submodules",
    "analyze_plaques": "submodules",
    "invasion": "submodules",
    "replication": "submodules",
    "figure": "plot",
    "ai": "qt/ai",
}


def _absorb_registered_api_modules() -> None:
    """Take the API-doc module of every registered app into the table above.

    The PULL half of the app-registration seam;
    :func:`spacr.qt.app.register_app` PUSHES into this table when this
    module is already imported, and this picks up whatever registered
    before it was, so the order of the two imports stops mattering.
    Without it a module that registers itself sends its API link to the
    generated API index rather than to its own page.
    """
    app = sys.modules.get("spacr.qt.app")
    # `getattr(..., None)`: `spacr.qt.app` may be half-built when this
    # runs, in which case nothing has registered yet and the push half of
    # the seam delivers every row later.
    pull = getattr(app, "registered_metadata", None) if app else None
    if pull is None:
        return
    for key, module in pull("api_module").items():
        _APP_API_MODULE.setdefault(key, module)


_absorb_registered_api_modules()


#: Settings whose documentation lives on the evaluation module's page rather
#: than on the page of whichever app happens to display them. A constant, not
#: a literal inside :func:`api_docs_url`: that function is called once per
#: setting per tooltip, so building these two sets there rebuilt them
#: thousands of times per panel.
_EVALUATION_DOC_KEYS = frozenset({
    "classifier_evaluation",
    "nested_cv_inner_folds",
    "evaluation_calibration",
    "evaluation_bins",
    "evaluation_fail_on_leakage",
    "leakage_audit_train_test",
    "leakage_hash_content",
    "leakage_require_identity",
})

#: UMAP settings documented on the hyperparameter-search page.
_UMAP_SEARCH_DOC_KEYS = frozenset({
    "criterion", "search_mode", "adaptive", "n_trials", "n_folds",
    "random_seed", "resume_search", "n_neighbors_step",
    "min_dist_step", "min_improvement", "max_panels",
    "umap_stability_repeats", "umap_neighborhood_weight",
    "umap_stability_weight", "umap_cluster_structure_weight",
})


def api_docs_url(
    app_key: str,
    key: str = "",
    language: Optional[str] = None,
) -> str:
    """Return the spaCR API URL for an app or shared setting.

    Known app keys land on their module page. New or UI-only modules fall
    back to the generated API index rather than the documentation homepage.
    Shared batch-correction settings always land on their implementation,
    rather than whichever consumer app happens to display them.
    """
    try:
        from spacr.plugins import get_app
        plugin_app = get_app(app_key)
    except Exception:
        plugin_app = None
    if plugin_app is not None and plugin_app.docs_url:
        return plugin_app.docs_url
    if key.startswith("batch_"):
        module = "batch_correction"
    elif key in _EVALUATION_DOC_KEYS:
        module = "classifier_evaluation"
    elif app_key == "umap" and key in _UMAP_SEARCH_DOC_KEYS:
        module = "hyperparam"
    else:
        module = _APP_API_MODULE.get(app_key)
    if module:
        url = f"{DOCS_API_BASE}/spacr/{module}/index.html"
    else:
        url = f"{DOCS_API_BASE}/index.html"
    code = _language_code(language)
    return f"{url}?lang={code}" if code != "en" else url



#: Phrases a setting's own description uses to state a 0-to-1 domain.
_UNIT_INTERVAL_PHRASES = (
    "between 0 and 1",
    "strictly inside 0 and 1",
    "0 and 1",
)


#: Settings whose value is EITHER a positive number OR the word "auto".
#: Built as a QDoubleSpinBox whose minimum reads "auto"
#: (:meth:`QDoubleSpinBox.setSpecialValueText`) -- one control that expresses
#: both, with no new widget class and no second field to keep in step.
#:
#: `alpha` is here because it was UNSETTABLE. Its shipped default is the
#: integer 1, so the panel inferred an integer and built a QSpinBox: the
#: documented 'auto' could not be typed, and neither could any value below 1.
#: Every value the control could reach shrinks a fraction-scale design to
#: nothing -- measured on the reference screen, alpha=1 sent all 790
#: coefficients to exactly zero -- so the penalised families could not be run
#: from the GUI at all.
AUTO_OR_NUMBER_SETTINGS = ("alpha",)

#: What the minimum of such a spin box means, and what it shows.
AUTO_TEXT = "auto"


def _auto_or_number_box(default):
    """A spin box for a setting that takes a positive number or "auto"."""
    box = QDoubleSpinBox()
    box.setDecimals(6)
    box.setRange(0.0, 1e6)
    box.setSingleStep(0.001)
    # Qt shows the SPECIAL TEXT in place of the minimum, so 0.0 is the
    # spelling of "auto" and the user reaches it by winding the box down --
    # which is also where somebody hunting for a smaller penalty is heading.
    box.setSpecialValueText(AUTO_TEXT)
    _set_auto_or_number(box, default)
    return box


def _set_auto_or_number(box, value) -> None:
    """Put ``value`` -- a number, ``None``, or "auto" -- into such a box."""
    if value is None or str(value).strip().lower() == AUTO_TEXT:
        box.setValue(box.minimum())
        return
    try:
        box.setValue(float(value))
    except (TypeError, ValueError):
        box.setValue(box.minimum())


def _read_auto_or_number(box):
    """"auto" when the box is at its minimum, otherwise the float."""
    return AUTO_TEXT if box.value() <= box.minimum() else float(box.value())


def _float_domain(key: str, default: float):
    """Return the minimum, maximum, and step for a float editor.

    The range follows the setting's documented numeric domain. The step
    follows the magnitude of the default so a single wheel event cannot move
    fractional settings by a whole unit.
    """
    magnitude = abs(float(default))
    if magnitude and magnitude < 1:
        step = 0.01
    elif magnitude < 10:
        step = 0.1
    else:
        step = 1.0

    text = ""
    try:
        from ... import settings as _settings

        # `tooltips`, not `descriptions`: `descriptions` is keyed by APP, and
        # `tooltips` is the per-setting text that states the domain.
        text = str(_settings.tooltips.get(key, "") or "").lower()
    except Exception:                                    # noqa: BLE001
        text = ""
    if any(phrase in text for phrase in _UNIT_INTERVAL_PHRASES):
        # The setting says it lives in the unit interval. Hold the box to it:
        # a probability the user cannot type is better than a run that dies
        # forty seconds in having already written half a results folder.
        # The floor is the smallest value the box can express, not 0: a
        # setting whose own text says "between 0 and 1" is refused at 0 by
        # the code that reads it, so a box that clamps a bad saved value to
        # 0.0 has only moved the failure. With decimals=6 that floor is 1e-6.
        return 1e-6, 1.0, min(step, 0.01)
    return -1e12, 1e12, step

_TYPE_NAMES = {int: "integer", float: "float", bool: "boolean",
               str: "string", list: "list", tuple: "tuple",
               dict: "dictionary"}


def _type_hint(key: str) -> str:
    """Human-readable type of a setting, from spacr.settings.expected_types.

    e.g. ``'integer'``, ``'float'``, ``'boolean'``, ``'list'``, or
    ``'integer or float'`` / ``'string (optional)'`` for unions/None."""
    if not key:
        return ""
    try:
        from spacr.settings import expected_types
    except Exception:
        return ""
    t = expected_types.get(key)
    if t is None:
        return ""
    if isinstance(t, tuple):
        parts, optional = [], False
        for x in t:
            if x is type(None):
                optional = True
                continue
            parts.append(_TYPE_NAMES.get(x, getattr(x, "__name__", str(x))))
        s = " or ".join(dict.fromkeys(parts))   # dedupe, keep order
        if optional and s:
            s += " (optional)"
        return s
    return _TYPE_NAMES.get(t, getattr(t, "__name__", str(t)))


def _humanize(key: str) -> str:
    return setting_label(key) if key else ""


def _strip_type_prefix(text: str) -> str:
    """Drop a leading ``(int) - `` / ``(bool) `` style prefix — the type is
    rendered separately + authoritatively from expected_types."""
    import re
    return re.sub(r"^\s*\([^)]*\)\s*[-–:]?\s*", "", text or "").strip()


#: ``argument -> resolved code`` while a :func:`language_resolved_once`
#: scope is open, and ``None`` when none is. See that function for why the
#: cache is scoped rather than permanent.
_LANGUAGE_SCOPE: Optional[Dict[Any, str]] = None

#: How many nested :func:`language_resolved_once` scopes are open. Nesting is
#: the normal case, not an edge one: a screen wraps its whole panel build and
#: ``build_sections`` wraps itself, so the inner scope must not drop the cache
#: the outer one is still using.
_LANGUAGE_SCOPE_DEPTH = 0

#: Translated fragments already resolved inside the open scope, or ``None``.
#: Every setting is rendered TWICE while a panel is built -- once as the
#: HTML tooltip on the widget and once as the plain hint under the form --
#: and the two share their name, their type hint and their prose. Scoped for
#: the same reason the language is: a catalog upgrade or a renamed organelle
#: slot must reach the next panel, and inside one synchronous build neither
#: can happen.
_TRANSLATION_MEMO: Optional[Dict[Any, Any]] = None


@contextmanager
def language_resolved_once():
    """Cache language and translation lookups during one synchronous build.

    Nested scopes share the outermost cache. The cache is discarded when the
    outermost scope exits so subsequent builds observe language or catalog
    changes.
    """
    global _LANGUAGE_SCOPE, _LANGUAGE_SCOPE_DEPTH, _TRANSLATION_MEMO
    if _LANGUAGE_SCOPE is None:
        _LANGUAGE_SCOPE = {}
        _TRANSLATION_MEMO = {}
    _LANGUAGE_SCOPE_DEPTH += 1
    try:
        yield
    finally:
        _LANGUAGE_SCOPE_DEPTH -= 1
        if _LANGUAGE_SCOPE_DEPTH <= 0:
            _LANGUAGE_SCOPE_DEPTH = 0
            _LANGUAGE_SCOPE = None
            _TRANSLATION_MEMO = None


def _language_code(language: Optional[str] = None) -> str:
    """Resolve ``language`` without making settings metadata depend on Qt."""
    scope = _LANGUAGE_SCOPE
    if scope is not None:
        try:
            return scope[language]
        except KeyError:
            pass
        except TypeError:
            # An unhashable argument cannot be cached; resolve it directly
            # rather than refuse to answer.
            scope = None

    from ..i18n import current_language, normalize_language

    code = normalize_language(language or current_language())
    if scope is not None:
        scope[language] = code
    return code


def _translated_ui_text(
    source: str,
    language: Optional[str] = None,
    **values: object,
) -> str:
    """Translate one complete explainer template, or retain its English.

    Scientific guidance must use an exact catalog record. Falling through to
    :func:`spacr.qt.i18n.tr`'s short-label term substitution could otherwise
    produce a partly translated sentence while catalogs are being upgraded.
    Format values are applied after translation so a locale may reorder them.
    """
    from ..i18n import _exact_translation, tr

    code = _language_code(language)
    if code == "en" or _exact_translation(str(source), code) is not None:
        return tr(source, code, **values)
    return tr(source, "en", **values)


def _translated_body(
    text: str,
    language: Optional[str] = None,
    *,
    setting_key: str = "",
    category: bool = False,
) -> str:
    """Translate setting prose only when a complete translation exists.

    The general UI translator deliberately supports conservative word-level
    translation for short labels.  Applying that behavior to a scientific
    paragraph produces a misleading half-English paragraph, however.  Tooltip
    bodies therefore accept exact catalog/plugin translations only and
    otherwise retain the canonical English source byte-for-byte.
    """
    source = " ".join(_strip_type_prefix(text).split())
    if not source:
        return ""
    code = _language_code(language)
    if code == "en":
        return source
    memo = _TRANSLATION_MEMO
    memo_key = ("body", source, code, setting_key, category)
    if memo is not None and memo_key in memo:
        return memo[memo_key]
    from ..i18n import _exact_translation, tr

    try:
        from ..i18n_catalogs import category_help, setting_tooltip
        translated = (
            setting_tooltip(setting_key, source, code)
            if setting_key
            else category_help(source, code) if category else None
        )
        if translated is not None:
            if memo is not None:
                memo[memo_key] = translated
            return translated
    except (ImportError, AttributeError):
        pass

    resolved = (
        tr(source, code)
        if _exact_translation(source, code) is not None
        else source
    )
    if memo is not None:
        memo[memo_key] = resolved
    return resolved


def _translated_type_hint(key: str, language: Optional[str] = None) -> str:
    """Return a localized type signature while preserving English defaults."""
    source = _type_hint(key)
    code = _language_code(language)
    if not source or code == "en":
        return source

    memo = _TRANSLATION_MEMO
    memo_key = ("type_hint", source, code)
    if memo is not None and memo_key in memo:
        return memo[memo_key]

    from ..i18n import tr

    optional = source.endswith(" (optional)")
    core = source[:-11] if optional else source
    # A slash is a language-neutral union separator.  Translating each atomic
    # type avoids asking the catalog to enumerate every possible union.
    translated = " / ".join(tr(part, code) for part in core.split(" or "))
    if optional:
        translated = f"{translated} ({tr('optional', code)})"
    if memo is not None:
        memo[memo_key] = translated
    return translated


def _translated_setting_name(
    key: str,
    language: Optional[str] = None,
    app_key: str = "",
) -> str:
    """Translate a short humanized setting label using the UI term catalog."""
    code = _language_code(language)
    memo = _TRANSLATION_MEMO
    memo_key = ("setting_name", key, code, app_key)
    if memo is not None and memo_key in memo:
        return memo[memo_key]

    from ..i18n import _ROWS, _TERM_ROWS, tr

    source = _humanize(key)
    # The compact catalog is the hand-reviewed authority for exact terms.
    # External generated labels extend it, but never override a correction.
    if source in _ROWS or source in _TERM_ROWS:
        resolved = tr(source, code)
    else:
        resolved = None
        try:
            from ..i18n_catalogs import setting_label
            resolved = setting_label(key, source, code, app_key)
        except (ImportError, AttributeError):
            resolved = None
        if resolved is None:
            resolved = tr(source, code)
    if memo is not None:
        memo[memo_key] = resolved
    return resolved


def _api_reference_tooltip(
    key: str,
    language: Optional[str] = None,
    app_key: str = "",
) -> str:
    """Localized accessible caption for a setting's teal API dot."""
    from ..i18n import tr

    code = _language_code(language)
    return tr(
        "Open API reference for {name}",
        code,
        name=_translated_setting_name(key, code, app_key),
    )


def format_tooltip(
    text: str,
    app_key: str,
    key: str = "",
    language: Optional[str] = None,
) -> str:
    """Return localized typed HTML with an unchanged API-document URL."""
    from ..i18n import tr

    code = _language_code(language)
    body_source = _translated_body(text, code, setting_key=key)
    body = escape(body_source)
    header = escape(_translated_setting_name(key, code, app_key))
    th = escape(_translated_type_hint(key, code))
    if header and th:
        header = f"<b>{header}</b> <i>({th})</i>"
    elif header:
        header = f"<b>{header}</b>"
    if not body:
        if code == "en" and key:
            body = f"Controls {escape(_humanize(key).lower())}."
        else:
            body = escape(tr("Controls this setting.", code))
    url = escape(api_docs_url(app_key, key, code), quote=True)
    link = (
        f'<a href="{url}">'
        f'{escape(tr("Open spaCR API documentation", code))}</a>'
    )
    parts = [p for p in (header, body, link) if p]
    return "<br>".join(parts)


def plain_tooltip(
    text: str,
    app_key: str,
    key: str = "",
    language: Optional[str] = None,
) -> str:
    """Same content as `format_tooltip` but plain text — used by the
    hover-follows footer at the bottom of each AppScreen."""
    from ..i18n import tr

    code = _language_code(language)
    body = _translated_body(text, code, setting_key=key)
    if not body:
        body = (f"Controls {_humanize(key).lower()}."
                if code == "en" and key
                else tr("Controls this setting.", code))
    th = _translated_type_hint(key, code)
    name = _translated_setting_name(key, code, app_key)
    head = f"{name} ({th})" if (name and th) else name
    parts = [p for p in (head, body) if p]
    summary = " — ".join(parts)
    url = api_docs_url(app_key, key, code)
    api = tr("API: {url}", code, url=url)
    return f"{summary} — {api}" if summary else api


def _is_self_labelling(widget) -> bool:
    """Does this control carry its own visible label?

    A `QCheckBox` does: its text sits beside the box and there is no separate
    label to hang the help on. A composite field does NOT -- it is a
    container, its text belongs to a child, and Qt delivers `Enter` to it
    whenever the pointer crosses into any of those children, so decorating
    it puts the help on the field.
    """
    from PySide6.QtWidgets import QAbstractButton, QLabel

    if isinstance(widget, QLabel):
        return True
    if isinstance(widget, QAbstractButton):
        try:
            return bool(widget.text())
        except (AttributeError, RuntimeError):
            return False
    return False


class _ApiTooltipFilter(QObject):
    """Show rich setting help in the clickable sticky tooltip."""

    def eventFilter(self, watched, event):  # noqa: N802 (Qt naming)
        # Re-render on entry so a Preferences language change cannot leave a
        # sticky popup displaying an earlier language.
        if event.type() == QEvent.Enter:
            refresh_api_tooltips(watched)
        html = watched.property("apiTooltipHtml")
        if not html:
            return False
        if event.type() == QEvent.Enter:
            from ..widgets.hover_tooltip import HoverTooltip
            HoverTooltip.instance().show_for(watched, str(html))
        elif event.type() == QEvent.Leave:
            from ..widgets.hover_tooltip import HoverTooltip
            HoverTooltip.instance().start_hide()
        elif event.type() == QEvent.ToolTip:
            # Suppress the native tooltip: it disappears when the pointer moves
            # toward its link, whereas HoverTooltip is intentionally clickable.
            return True
        return False



#: Marks a tooltip carrying a "not used here" note, so the note is appended
#: once and removed cleanly rather than accumulating.
_BASIS_NOTE_PROPERTY = "_spacr_basis_note"

#: Where a label's own help is kept while a greyed-out reason is appended to
#: it. Restored verbatim rather than stripped back off, because the note is
#: rendered into HTML and un-rendering it is guesswork.
_NOTE_BACKUP_PROPERTY = "_spacr_help_before_note"

#: The reason a control is currently greyed, held on the CONTROL so it can be
#: put on a label that does not exist yet, and removed from one that acquired
#: it before there was anywhere to keep the original.
_PENDING_NOTE_PROPERTY = "_spacr_greyed_reason"


# ---------------------------------------------------------------------------
# The Model & Inference explainer box (instruction 132)
# ---------------------------------------------------------------------------
#
# "it is important for the user to know all of this."  A read-only box in the
# Model & Inference section that states, for the CURRENT selection, the formula
# that will be fitted and what it models.
#
# It is prose, not a tooltip, because the thing it has to say does not fit in
# one: the default changed to `mixed`, and a mixed fit answers the gene
# question WELL while giving up something the previous default appeared to
# give -- a guide-level hit list. A user who takes the default and later goes
# looking for their guide p-values is exactly who this box is for, so the cost
# is a named section of it rather than a clause someone might not hover.
#
# THE TEXT IS BUILT BY A PURE FUNCTION so it can be asserted without a
# QApplication, and so the formulas have one spelling in this file rather than
# one per branch of a widget callback.

#: The `level` choices offered when the backend is a fixed-effects one.
#: `both` is the default: it fits the guide model and the gene model
#: SEPARATELY and writes both tables.
REGRESSION_LEVELS = ("both", "grna", "gene")

#: The model part of each formula, without the plate terms -- those are
#: decided by the settings and added by :func:`formula_for`.
GRNA_TERM = "fraction:grna"
GENE_TERM = "gene_fraction:gene"
MIXED_TERM = "gene_fraction:gene + (1 | gene/grna)"

#: One coefficient per guide. The guide is the unit the screen measures.
GRNA_FORMULA = "y ~ fraction:grna"

#: One coefficient per gene, from the summed guide fraction.
GENE_FORMULA = "y ~ gene_fraction:gene"

#: The mixed model: gene fixed, guide random and nested inside its gene.
MIXED_FORMULA = "y ~ gene_fraction:gene + (1 | gene/grna)"


def formula_for(term: str, *, plate_position: bool = False,
                random_row_column: bool = False) -> str:
    """The formula actually fitted, for one model term and the plate settings.

    The box must show the formula the run fits. Previously,
    `regression_model_explainer` took only `(regression_type, level)`, so the
    three constants above were printed whatever the two plate settings said,
    and a user who turned plate position OFF still read `+ rowID + columnID`.

    That is the same class of failure as an axis that relabels itself without
    moving its dots: the display asserts something the code does not do, and
    nothing on screen says which to believe.

    The three states produced by :func:`spacr.ml.prepare_formula` are no
    position terms when ``plate_position=False``; fixed ``rowID`` and
    ``columnID`` effects when ``plate_position=True``; and row/column variance
    components when ``random_row_column=True``.

    ``random_row_column`` implies the terms are present, so it wins over
    ``plate_position=False``; that combination is refused upstream
    (`_reconcile_random_row_column_effects`) and this renders what the refusal
    would be about rather than inventing a fourth state.

    :param term: the model part, e.g. ``"fraction:grna"`` or
        ``"gene_fraction:gene + (1 | gene/grna)"``.
    """
    if random_row_column:
        position = " + (1 | rowID) + (1 | columnID)"
    elif plate_position:
        position = " + rowID + columnID"
    else:
        position = ""
    return f"y ~ {term}{position}"

#: Deprecated formula retained so the explainer can show why it is refused.
#: `gene_fraction` is the
#: SUM of the gene's gRNA fractions (`spacr.ml.check_and_clean_data`), so every
#: gene column here is an exact linear combination of that gene's own guide
#: columns and the combined design is rank deficient.
COLLINEAR_FORMULA = (
    "y ~ fraction:grna + gene_fraction:gene + rowID + columnID")

#: Final explainer line linking to the detailed formula-change rationale.
#:
#: The full explanation lives in :func:`regression_model_explainer`; keeping a
#: short pointer in the panel avoids repeating a long retired-design history.
_HISTORY_POINTER_SYMBOL = "regression_model_explainer.__doc__"
_HISTORY_POINTER_SOURCE = "WHY THE FORMULA CHANGED -> {symbol}"
_HISTORY_POINTER = _HISTORY_POINTER_SOURCE.format(
    symbol=_HISTORY_POINTER_SYMBOL,
)

#: The column the prose is wrapped to.
#:
#: Set against the width the settings pane ACTUALLY grants the box, measured
#: rather than assumed: the pane opens at ~400px and the splitter stretches it
#: to ~490 for this box, which is about 57 monospace characters. Wrapping
#: prose wider than that put every sentence behind a horizontal scrollbar.
#: The indented mixed formula is 63 characters and deliberately exceeds this
#: -- it is one line and it must not be broken, so :func:`explainer_width`
#: hands that length to the box as its minimum rather than this column.
_EXPLAINER_WIDTH = 54

#: The short name shown in the box header beside the key the user selected.
_MODE_TITLES = {
    "auto": "chosen from the response",
    "ols": "ordinary least squares",
    "wls": "weighted least squares",
    "rlm": "robust M-estimation",
    "huber": "robust M-estimation (Huber)",
    "glm": "generalised linear model",
    "poisson": "Poisson GLM",
    "quasi_binomial": "quasi-binomial GLM",
    "beta": "beta regression",
    "logit": "binomial GLM, logit link",
    "probit": "binomial GLM, probit link",
    "quantile": "quantile regression",
    "mixed": "mixed effects, guides nested in genes",
    "lasso": "penalised least squares, L1",
    "ridge": "penalised least squares, L2",
    "elasticnet": "penalised least squares, L1 + L2",
    "hinge": "linear SVM on a binarised response",
    "horseshoe": "sparse Poisson GLM, horseshoe",
    # Kept short on purpose: the header renders as "MODEL: <key> -- <title>"
    # on ONE unwrapped line, and the box does not soft-wrap, so a title long
    # enough to pass 54 characters puts the model's own name behind a
    # horizontal scrollbar.
    "group_lasso": "guides grouped by gene",
    "rra": "MAGeCK alpha rank aggregation",
}

#: Backends suited to pooled CRISPR screens, with the reason each is
#: recommended. The explanations focus on sparse, high-dimensional designs
#: and correlated guides rather than on a method's general popularity.
RECOMMENDED_FOR_SCREENS = {
    "mixed": "treats guides as repeated perturbations nested within genes",
    "horseshoe": "uses a sparse prior when most guides have small effects "
                 "and a few have large effects",
    "elasticnet": "combines L1 and L2 to retain correlated guides from one "
                  "gene",
    "lasso": "builds a sparse model and ranks bootstrap stability",
    "group_lasso": "selects or drops each gene's guides as a group",
    "rra": "aggregates guide ranks by gene without fitting every guide "
           "jointly",
}

#: Information-limit caveat shown beside every recommended backend.
INFORMATION_LIMIT_NOTE = (
    "Fewer wells than guides puts a joint guide fit below the information "
    "limit. Penalties, priors and groups do not create information. The "
    "permutation test is the exception because it tests one guide at a time.")


#: One- or two-sentence descriptions of what each regression mode fits, based
#: on :func:`spacr.ml.regression_model` rather than the general
#: reputation of the method. Where a backend reads a setting from this panel it
#: is named, so the box and the Estimator Tuning section below it agree.
_MODE_NOTES = {
    "auto": (
        "spaCR reads the response and picks the model itself "
        "(check_distribution): 0/1 data gets logit, a fraction strictly "
        "inside (0, 1) gets beta -- or quasi_binomial when values sit within "
        "1e-6 of a boundary -- a fraction including exact 0 or 1 gets "
        "quasi_binomial, and anything that passes a normality test gets ols. "
        "The run prints the model it chose, so read the console before "
        "naming a model in a methods section."
    ),
    "ols": (
        "Least squares: minimises the summed squared residual and assumes "
        "the well residuals are roughly normal around one common variance. "
        "It is the baseline the others are worth comparing against."
    ),
    "wls": (
        "Least squares weighted by the well's cell count, so a well of 400 "
        "cells outweighs one of 30. Worth choosing when wells differ widely "
        "in how many cells their score was averaged over, which ols ignores."
    ),
    "rlm": (
        "Robust M-estimation with a Huber loss, tuned by huber_t (default "
        "1.345, which is 95% efficient under normality). Wells far from the "
        "fit are down-weighted instead of dragging it, and no R-squared is "
        "reported."
    ),
    "hinge": (
        "A linear support-vector fit (hinge loss) on the response BINARISED "
        "at hinge_threshold, so it asks which guides SEPARATE high wells "
        "from low wells rather than how far they move the score. It has no "
        "likelihood: the p-values are bootstrap Wald values over "
        "hinge_n_boot resamples, not a likelihood-ratio test."
    ),
    "glm": (
        "A generalised linear model whose FAMILY AND LINK are picked from "
        "the response by pick_glm_family_and_link rather than assumed, and "
        "the run prints the pair it chose. Where the family comes out "
        "Poisson, log(cell_count) enters as an offset, so the coefficients "
        "are effects on a per-cell rate."
    ),
    "poisson": (
        "Poisson GLM with a log link and offset(log(cell_count)), for "
        "per-well counts. The offset is what makes the coefficients effects "
        "on the per-cell RATE rather than on the well's headcount -- without "
        "it, a well simply holding more cells reads as a hit."
    ),
    "quasi_binomial": (
        "Binomial GLM whose dispersion is estimated from the Pearson "
        "chi-square instead of being fixed at 1, for a fraction that varies "
        "more than binomial sampling allows; the cell count enters as "
        "var_weights. Choose it over logit when the residual deviance says "
        "the response is overdispersed."
    ),
    "beta": (
        "Beta regression, for a response that is a fraction strictly inside "
        "(0, 1): it models the mean of a bounded variable as bounded, rather "
        "than fitting a proportion as if it could exceed 1, which is what "
        "ols on a fraction does. Exact 0 or 1 values must be handled before "
        "it can fit."
    ),
    "quantile": (
        "Quantile regression fits a CONDITIONAL QUANTILE of the response -- "
        "the `quantile` setting, where 0.5 is the median -- NOT the mean, so "
        "a perturbation that moves the tail without shifting the centre "
        "appears here and in no mean model. It is the one backend whose "
        "answer changes meaning with a setting, so name the quantile "
        "alongside the result."
    ),
    "lasso": (
        "L1-penalised least squares sets coefficients to zero; alpha='auto' "
        "selects the penalty by 5-fold cross-validation. It reports "
        "bootstrap selection frequency across lasso_n_boot resamples, not "
        "p-values, and applies lasso_selection_threshold."
    ),
    "elasticnet": (
        "Elastic net combines L1 and L2 through l1_ratio (1 is lasso; 0 is "
        "ridge), with alpha='auto' chosen by 5-fold cross-validation. It "
        "reports bootstrap selection frequency across lasso_n_boot resamples, "
        "not p-values, and applies lasso_selection_threshold."
    ),
    "ridge": (
        "Penalised least squares with an L2 penalty, which never sets a "
        "coefficient to exactly zero -- so there is no selection frequency "
        "to report, every feature would score 1.0, and it falls back to an "
        "approximate p-value. That test is mis-specified, in the safe "
        "direction: the standard error is unpenalised while the coefficient "
        "it divides has been shrunk, so the statistic is too small and ridge "
        "under-detects rather than manufacturing hits."
    ),
    "horseshoe": (
        "A sparse Poisson GLM with a horseshoe prior -- spaCRPower's "
        "power-analysis model -- with offset(log(cell_count)). The prior "
        "shrinks the bulk of the guides hard toward zero while leaving a "
        "genuinely large effect close to untouched, which suits a screen "
        "where most guides are expected to do nothing."
    ),
    "mixed": (
        "The gene is a FIXED effect; each guide is a RANDOM effect nested "
        "inside its gene, treating guides as repeated perturbations with "
        "different efficiencies and off-target effects. Guide disagreement "
        "widens the gene interval, and a gene supported by one noisy guide "
        "shrinks toward zero."
    ),
    "group_lasso": (
        "Group-penalised least squares treats A GENE'S GUIDES AS ONE BLOCK, "
        "retaining or zeroing the whole block. group_lasso_lambda sets the "
        "penalty relative to group_lasso.max_lambda; hits use bootstrap "
        "selection frequency over lasso_n_boot resamples and "
        "lasso_selection_threshold."
    ),
    "rra": (
        "MAGeCK alpha-RRA ranks guides across the screen, then scores each "
        "gene from its strongest ranks within rra_alpha. It builds an "
        "empirical null with rra_permutations for each guide count and "
        "reports depletion and enrichment separately."
    ),
}
_MODE_NOTES["huber"] = _MODE_NOTES["rlm"]
_MODE_NOTES["logit"] = (
    "Binomial GLM with a logit link on a fraction, weighted by the well's "
    "cell count as var_weights -- which is what tells the variance function "
    "that a fraction measured from 400 cells is firmer evidence than the "
    "same fraction measured from 30. Coefficients are log-odds."
)
_MODE_NOTES["probit"] = (
    "Binomial GLM with a probit link on a fraction, weighted by the well's "
    "cell count as var_weights. It differs from logit only in the link: the "
    "fitted probabilities are near-identical, and the coefficients are on a "
    "different scale and are not log-odds."
)


# ---------------------------------------------------------------------------
# What the default costs (instruction 140)
# ---------------------------------------------------------------------------
#
# Reported 2026-08-18: "im running the mixed model now and it is taking much
# longer than before is that normal?" ... "it is still going, cpu at 100
# percent". NOTHING WAS WRONG. MixedLM is an iterative REML optimisation and
# it is single-threaded, so one core at 100% for an hour is exactly what a
# healthy fit looks like -- and an hour of silence at 100% CPU is
# indistinguishable from a hang. 132 made `mixed` the DEFAULT, which means
# everybody pays this, so it belongs where the model is chosen.

#: The measurement, as ``(genes, wells, ols seconds, mixed seconds)``.
#:
#: Measured by calling :func:`spacr.ml.regression_model` directly on a
#: well-conditioned gene-level design without guide random effects. The full
#: nested model is more expensive, so these values are lower bounds.
MIXED_COST_ANCHORS = (
    (40, 400, 0.03, 1.62),
    (80, 600, 0.16, 10.66),
)

#: Reference design for the "tens of minutes to hours" expectation, expressed
#: as ``(genes, guides, wells)``.
MIXED_COST_SCREEN = (823, 389, 610)


_MIXED_COST_NOTE_TEMPLATE = (
    "MEASURED 2026-08-18: {small_genes} genes/{small_wells} wells took "
    "{small_ols:g}s as ols and {small_mixed:g}s as mixed ({small_ratio:g}x); "
    "{big_genes}/{big_wells}, {big_ols:g}s against {big_mixed:g}s "
    "({big_ratio:g}x) -- and both were gene level only. This fit adds a "
    "random effect per guide too, so {guides} guides over {wells} wells is "
    "tens of minutes to hours. Single-threaded REML: one core at 100% is a "
    "healthy fit, not a hang. For an answer now, use ols at level='both'."
)


def mixed_cost_note(language: Optional[str] = None) -> str:
    """What ``mixed`` costs, as one paragraph, built from the measurement.

    :param language: UI language code. ``None`` uses the active language.
    :returns: Exact localized guidance when its catalog record is current;
        otherwise the canonical English paragraph.

    ONE SOURCE FOR TWO PLACES. The model box states it before the user
    chooses, and the run states it again before it blocks; two hand-written
    copies of a measurement are two numbers that drift apart, and the second
    one to be edited is the one nobody believes afterwards.

    A MEASURED RANGE, NOT "THIS MAY BE SLOW" -- the digits are what make it
    actionable, and "may be slow" is what the console said by saying nothing.
    """
    (small_genes, small_wells, small_ols, small_mixed), \
        (big_genes, big_wells, big_ols, big_mixed) = MIXED_COST_ANCHORS
    guides, _genes, wells = MIXED_COST_SCREEN
    # EVERY NUMBER KEPT, half the words. Instruction 143 B: "do not shorten
    # by deleting the numbers -- they are what makes the claim checkable.
    # Shorten by removing what does not need re-reading."
    return _translated_ui_text(
        _MIXED_COST_NOTE_TEMPLATE,
        language,
        small_genes=small_genes,
        small_wells=small_wells,
        small_ols=small_ols,
        small_mixed=small_mixed,
        small_ratio=round(small_mixed / small_ols),
        big_genes=big_genes,
        big_wells=big_wells,
        big_ols=big_ols,
        big_mixed=big_mixed,
        big_ratio=round(big_mixed / big_ols),
        guides=guides,
        wells=wells,
    )


#: The models worth warning about before they block, and the reason for each.
#: `mixed` is the one that is MEASURED (:func:`mixed_cost_note`) and the one
#: that is the default. The other two are named because their cost is set by
#: a control on this panel -- `rra_permutations` for the permuted null, and
#: the sampler behind `horseshoe` -- rather than by the size of the screen,
#: so a user who is waiting has something to change.
SLOW_MODELS = ("mixed", "rra", "horseshoe")


def _count_files_of(settings) -> list:
    """The sgRNA count CSVs this run was given, in order.

    ``paired_data`` is the current shape (one row per score/count pair) and
    ``count_data`` is the legacy list :func:`spacr.ml.perform_regression`
    still migrates; both are read here because a settings CSV saved before
    the migration is exactly the kind of run somebody re-opens.
    """
    paths = []
    pairs = (settings or {}).get("paired_data") or []
    if isinstance(pairs, (list, tuple)):
        for pair in pairs:
            value = pair.get("count") if isinstance(pair, dict) else None
            if isinstance(value, str) and value.strip():
                paths.append(value.strip())
    if not paths:
        legacy = (settings or {}).get("count_data")
        if isinstance(legacy, str):
            legacy = [legacy]
        for value in legacy or []:
            if isinstance(value, str) and value.strip():
                paths.append(value.strip())
    return paths


#: Where the scan records which plate a count file stands for, when the file
#: itself does not say. A frame attribute rather than a column, so nothing
#: downstream sees an invented plate in the data.
_FILE_PLATE = "spacr_scan_plate"


def _well_keys(frame):
    """One identifier per well in a count frame, or ``None``.

    Mirrors :func:`spacr.ml.process_reads`: a well is plate + row + column,
    ``plate_row`` is ``<plate>_<row>`` split on the LAST separator (the plate
    is the half that may itself contain one), and ``prc`` is that answer
    already composed. Returns ``None`` when the frame carries none of them,
    rather than guessing -- a well count off by the number of plates is
    worse than no well count.
    """
    if "prc" in frame.columns:
        return frame["prc"].astype(str)
    columns = set(frame.columns)
    if "plate_row" in columns and "columnID" in columns:
        return (frame["plate_row"].astype(str) + KEY_SEPARATOR
                + frame["columnID"].astype(str))
    if {"rowID", "columnID"} <= columns:
        if "plateID" in columns:
            plate = frame["plateID"].astype(str)
        else:
            # NO PLATE COLUMN, SO THE WELLS OF THIS FILE ARE THIS FILE'S.
            # It used to substitute the literal "plate1", which is the guess
            # its own docstring forbids -- and it cost exactly what that
            # sentence predicts: the example screen's four count files each
            # name the same 384 row/column pairs, so the union across them
            # was 384 for a 1,536-well screen. Off by the number of plates,
            # stated confidently, on the first line of the console.
            #
            # `_FILE_PLATE` is filled by the caller with the file's position,
            # which is the same rule `load_regression_input_pairs` uses when
            # neither side declares a plate: the pair-row order.
            plate = str(frame.attrs.get(_FILE_PLATE, "plate1"))
        return (plate + KEY_SEPARATOR + frame["rowID"].astype(str)
                + KEY_SEPARATOR + frame["columnID"].astype(str))
    return None


def _split_guide_names(names):
    """``(genes, guides)`` for a set of gRNA names, or ``(None, guides)``.

    THE SAME POSITIONAL RULE THE PIPELINE USES, and it is positional:
    :func:`spacr.ml.process_reads` splits ``<org>_<gene>_<guide>`` and
    requires EVERY name to have the same three components, because
    ``str.split(expand=True)`` pads a short name with ``None`` instead of
    raising -- which silently deleted those reads from the screen. Names of
    another shape get no gene count here for the same reason: a gene total
    taken from a rule the run will not apply is a number that disagrees with
    the fit.
    """
    guides = {str(name) for name in names}
    widths = {len(name.split(KEY_SEPARATOR)) for name in guides}
    if widths == {3}:
        genes = {name.split(KEY_SEPARATOR)[1] for name in guides}
        return genes, {KEY_SEPARATOR.join(name.split(KEY_SEPARATOR)[1:])
                       for name in guides}
    if widths == {2}:
        return {name.split(KEY_SEPARATOR)[0] for name in guides}, guides
    return None, guides


def regression_design_scan(settings) -> dict:
    """How big the fit is about to be, read off the count files it was given.

    The design: "The useful line names the design -- 'fitting 389
    genes and 823 guide random effects over 610 wells' -- because that is
    also the line that tells a user their filters did something unexpected."

    WHAT THIS IS AND IS NOT. It reads the sgRNA count CSVs and nothing else,
    so it is the design AS THE INPUT FILES HOLD IT: before the merge with
    the score data, before ``fraction_threshold`` and before the well
    filters. That is deliberate -- it is the number to compare the run's own
    post-cleaning counts against, and comparing them is how a filter that
    did something unexpected becomes visible. Every caller says which it is.

    NEVER RAISES. It runs to put a sentence in the console beside a fit that
    is already starting; a scan that threw would take the run's own message
    with it. What it could not work out comes back as ``None`` with a
    ``note`` saying why.

    :returns: ``{'genes', 'guides', 'wells', 'rows', 'files', 'note'}``.
    """
    out = {"genes": None, "guides": None, "wells": None, "rows": 0,
           "files": 0, "note": ""}
    paths = _count_files_of(settings)
    if not paths:
        out["note"] = "no count files in the settings"
        return out

    # THE ONE READER (145), and this line is why. Reading raw, the count
    # tables of the example screen -- which spell their keys `row_name` and
    # `column_name` -- carried no column this scan recognises, so it reported
    # "no 'prc', 'plate_row' or 'rowID'/'columnID' column, so wells were not
    # counted" over 642,551 rows that name 1,536 wells perfectly well.
    #
    # A count of NOTHING, printed confidently, on a table that has the
    # answer: exactly the failure instruction 145 exists to stop, and the
    # first line of the console a user reads before a run.
    #
    # `report=None`, because a column-collision note belongs to the run and
    # not to a sizing scan the user did not ask for.
    from ...tabular import read_table

    names, wells, unread = set(), set(), []
    no_wells = False
    for path in paths:
        try:
            frame = read_table(path, report=None)
        except Exception as error:                              # noqa: BLE001
            unread.append(f"{path} ({type(error).__name__})")
            continue
        out["files"] += 1
        out["rows"] += int(len(frame))
        # ONE FILE IS ONE PLATE when the file does not say otherwise, which
        # is `load_regression_input_pairs`' rule for the same question.
        frame.attrs[_FILE_PLATE] = f"plate{out['files']}"
        column = ("grna" if "grna" in frame.columns else
                  "grna_name" if "grna_name" in frame.columns else None)
        if column is not None:
            names |= set(frame[column].astype(str).unique().tolist())
        keys = _well_keys(frame)
        if keys is None:
            no_wells = True
        else:
            wells |= set(keys.unique().tolist())

    notes = []
    if unread:
        notes.append("could not read " + ", ".join(unread))
    if names:
        genes, guides = _split_guide_names(names)
        out["guides"] = len(guides)
        if genes is None:
            notes.append("the gRNA names are not "
                         "'<org>_<gene>_<guide>', so genes were not counted")
        else:
            out["genes"] = len(genes)
    else:
        notes.append("no 'grna' column")
    if no_wells:
        notes.append("no 'prc', 'plate_row' or 'rowID'/'columnID' column, "
                     "so wells were not counted")
    elif wells:
        out["wells"] = len(wells)
    out["note"] = "; ".join(notes)
    return out


# ---------------------------------------------------------------------------
# The box is TYPESET, not dumped (instruction 144)
# ---------------------------------------------------------------------------
#
# 2026-08-18: "actually my main problem was it dosnt look great. i want you to
# use markdown and colors for negative (CANNOT) and positive (MODEL, LEVEL,
# ETC.) text. make the formula look better (write the math symbol version then
# the code version if possible) short discriptions that contain the vital
# information for the user and links to APIs for the different methods".
#
# 143 read the first report as "too long" and cut 2,438 characters to 892. The
# content is settled; what was left is that nothing was EMPHASISED, so a
# formula read exactly like a caveat.
#
# ONE SOURCE, TWO LAYOUTS. Everything below composes the SAME pieces the plain
# renderer does -- `_MODE_TITLES`, `_MODE_NOTES`, `formula_for`,
# `mixed_cost_note` -- so the two cannot say different things. Only the layout
# is written twice, and the plain one stays because it is what a test can
# assert on and what a headless caller can print.

#: Mathematical notation shown for each model term.
#:
#: REAL UNICODE, NOT LATEX SOURCE. The box is a widget, not a renderer, and
#: ``\beta`` on screen is worse than no symbol at all.
_MATHS_RESPONSE = {
    "grna": "yᵢ = μ + Σ_g β_g·f_gi",
    "gene": "yᵢ = μ + Σ_G β_G·F_Gi",
    "mixed": "yᵢ = μ + Σ_G β_G·F_Gi + u_G + u_G:g",
}


def maths_for(kind: str, *, plate_position: bool = False,
              random_row_column: bool = False) -> List[str]:
    """The statistical statement, as lines, for one model term.

    THE MATHS AND THE CODE MUST AGREE, and this is the half that keeps them
    agreeing: it takes the same two plate arguments :func:`formula_for` does
    and reads them the same way. If the code line says ``+ rowID +
    columnID``, ρ and γ are here; if the plate-position toggle
    turns them off, BOTH lose them; if they are random effects, both say
    random. A box whose two formulas disagree is worse than a box with one.

    :param kind: ``'grna'``, ``'gene'`` or ``'mixed'``.
    :returns: the response line first, then the distribution line(s).
    """
    response = _MATHS_RESPONSE[str(kind)]
    distributions = []
    if str(kind) == "mixed":
        distributions.append("u_G ~ N(0, σ²_gene)   u_G:g ~ N(0, σ²_guide)")
    if random_row_column:
        response += " + u_r(i) + u_c(i)"
        distributions.append("u_r(i) ~ N(0, σ²_row)   u_c(i) ~ N(0, σ²_col)")
    elif plate_position:
        response += " + ρ_r(i) + γ_c(i)"
    response += " + εᵢ"
    distributions.append("εᵢ ~ N(0, σ²)")
    return [response] + distributions


_STATSMODELS = "https://www.statsmodels.org/stable/generated/"
_SKLEARN = "https://scikit-learn.org/stable/modules/generated/"

#: Where each backend's API lives, as ``(what to call it, where it is)``.
#:
#: External backends use direct documentation URLs; spaCR backends use paths
#: resolved against :data:`DOCS_API_BASE`.
MODEL_API_LINKS = {
    "auto": ("spacr.ml.check_distribution", "ml"),
    "ols": ("statsmodels OLS",
            _STATSMODELS + "statsmodels.regression.linear_model.OLS.html"),
    "wls": ("statsmodels WLS",
            _STATSMODELS + "statsmodels.regression.linear_model.WLS.html"),
    "rlm": ("statsmodels RLM",
            _STATSMODELS + "statsmodels.robust.robust_linear_model.RLM.html"),
    "huber": ("statsmodels RLM",
              _STATSMODELS
              + "statsmodels.robust.robust_linear_model.RLM.html"),
    "glm": ("statsmodels GLM",
            _STATSMODELS
            + "statsmodels.genmod.generalized_linear_model.GLM.html"),
    "poisson": ("statsmodels GLM",
                _STATSMODELS
                + "statsmodels.genmod.generalized_linear_model.GLM.html"),
    "logit": ("statsmodels GLM",
              _STATSMODELS
              + "statsmodels.genmod.generalized_linear_model.GLM.html"),
    "probit": ("statsmodels GLM",
               _STATSMODELS
               + "statsmodels.genmod.generalized_linear_model.GLM.html"),
    "quasi_binomial": ("statsmodels GLM",
                       _STATSMODELS
                       + "statsmodels.genmod.generalized_linear_model.GLM"
                         ".html"),
    "beta": ("statsmodels BetaModel",
             _STATSMODELS + "statsmodels.othermod.betareg.BetaModel.html"),
    "quantile": ("statsmodels QuantReg",
                 _STATSMODELS
                 + "statsmodels.regression.quantile_regression.QuantReg"
                   ".html"),
    "mixed": ("statsmodels MixedLM",
              _STATSMODELS
              + "statsmodels.regression.mixed_linear_model.MixedLM.html"),
    "ridge": ("scikit-learn Ridge",
              _SKLEARN + "sklearn.linear_model.Ridge.html"),
    "lasso": ("scikit-learn Lasso",
              _SKLEARN + "sklearn.linear_model.Lasso.html"),
    "elasticnet": ("scikit-learn ElasticNet",
                   _SKLEARN + "sklearn.linear_model.ElasticNet.html"),
    "hinge": ("scikit-learn LinearSVC",
              _SKLEARN + "sklearn.svm.LinearSVC.html"),
    "group_lasso": ("spacr.group_lasso", "group_lasso"),
    "rra": ("spacr.rra (MAGeCK alpha-RRA)", "rra"),
    "horseshoe": ("spacr.power_model", "power_model"),
}


def model_api_link(
    regression_type: Any,
    language: Optional[str] = None,
) -> Tuple[str, str]:
    """``(name, url)`` for one backend's API, or ``("", "")``.

    A spaCR backend is named by its MODULE and resolved against the published
    API documentation, so `group_lasso` and `rra` get the same kind of link
    statsmodels does rather than a module path a user has to go and find.

    :param regression_type: Backend key shown in the regression selector.
    :param language: UI language code appended to spaCR documentation links.
        Third-party links are returned unchanged.
    :returns: Link label and absolute documentation URL.
    """
    key = str(regression_type or "").strip().lower()
    entry = MODEL_API_LINKS.get(key)
    if entry is None:
        return "", ""
    name, target = entry
    if target.startswith("http"):
        return name, target
    url = f"{DOCS_API_BASE}/spacr/{target}/index.html"
    code = _language_code(language)
    return name, f"{url}?lang={code}" if code != "en" else url


#: Phrases the box emphasises, and the palette token each takes.
#:
#: AN EXPLICIT, SHORT TABLE rather than a rule over the prose. "Everything is
#: plain except what the sentence is about" (`spacr/figures/style.py`) applies
#: to text as much as to a figure; a regex that coloured every capitalised
#: phrase would over-emphasise unrelated text.
_EMPHASIS = (
    ("NO GUIDE-LEVEL HIT LIST", "error"),
    ("REPORTS NO P-VALUE", "error"),
    ("NOTHING TO BH-CORRECT", "error"),
    ("TWO MODELS, TWO TABLES", "success"),
)

#: The one heading that is a refusal rather than a description, so it takes
#: `error` where every other heading takes `accent`.
_REFUSAL_HEADING = "WHAT YOU DO NOT GET"

#: The two mixed-model decisions repeated by the plain and rich renderers.
#: Keeping one source prevents the user-visible settings box and the text API
#: from drifting apart as either is edited for clarity.
_MIXED_GUIDE_OUTPUT_NOTE = (
    "Guide results are BLUPs -- shrunken PREDICTIONS of departure from the "
    "gene -- NOT coefficients with standard errors and p-values. This model "
    "has NO GUIDE-LEVEL HIT LIST or guide-level BH correction. For a ranked, "
    "tested guide list, choose another model with level='grna'.")
_MIXED_MULTIPLE_TESTING_NOTE = (
    "Gene coefficients form one BH family; there is no second family because "
    "the guide effects are not tested.")
_UNKNOWN_MODEL_NOTE = (
    "spaCR has no description for this model, which means it is not one of "
    "the backends spacr.ml can fit. The run will refuse it and name the "
    "models it accepts."
)
_NO_P_VALUE_BOTH_NOTE = (
    "Each fit ranks features by bootstrap selection frequency and REPORTS "
    "NO P-VALUE, so there is NOTHING TO BH-CORRECT. A selection frequency "
    "is not a false-discovery rate and should not be quoted as one."
)
_NO_P_VALUE_SINGLE_NOTE = (
    "The fit ranks features by bootstrap selection frequency and REPORTS "
    "NO P-VALUE, so there is NOTHING TO BH-CORRECT. A selection frequency "
    "is not a false-discovery rate and should not be quoted as one."
)

#: What the box falls back to when no palette is handed in -- which is what a
#: test that is not about colour wants. Named tokens, not hexes, so a reader
#: of the rendered HTML can see which token a colour came from.
_TOKEN_FALLBACK = {name: name for name in
                   ("fg", "fg_muted", "accent", "error", "success",
                    "chip_value")}


def _colours(palette: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Return the six theme tokens used by the model explainer."""
    if not palette:
        return dict(_TOKEN_FALLBACK)
    return {name: str(palette.get(name) or _TOKEN_FALLBACK[name])
            for name in _TOKEN_FALLBACK}


def _ink(text: str, colour: str, *, bold: bool = False) -> str:
    """One coloured run of already-escaped text."""
    weight = " font-weight:600;" if bold else ""
    return f'<span style="color:{colour};{weight}">{text}</span>'


def _prose_html(text: str, ink: Dict[str, str]) -> str:
    """One paragraph, escaped, with the emphasis table applied."""
    out = escape(str(text))
    for phrase, token in _EMPHASIS:
        if phrase in out:
            out = out.replace(phrase, _ink(phrase, ink[token], bold=True))
    return f'<p style="margin:2px 0 8px 0;">{out}</p>'


def _heading_html(text: str, ink: Dict[str, str], *,
                  writes: str = "", refusal: bool = False) -> str:
    """A section heading: accent, or `error` when the section is a refusal.

    :param writes: the output file this section's formula produces. It is on
        the HEADING rather than under the formula because "which file does
        this end up in" is the question asked while scanning, and a name in
        `chip_value` is what makes it findable without reading the prose.
    """
    token = "error" if refusal or text == _REFUSAL_HEADING else "accent"
    tail = (f' → {_ink(escape(writes), ink["chip_value"])}' if writes else "")
    return (f'<p style="margin:10px 0 2px 0;">'
            f'{_ink(escape(text), ink[token], bold=True)}{tail}</p>')


def _formula_html(maths: List[str], code: str, ink: Dict[str, str]) -> str:
    """Render copyable, unwrapped formula and code blocks as HTML."""
    lines = "\n".join(escape(line) for line in maths)
    return (f'<pre style="margin:2px 0 2px 12px; color:{ink["fg"]};">'
            f'{lines}</pre>'
            f'<pre style="margin:2px 0 8px 12px;">'
            f'{_ink(escape(code), ink["accent"])}</pre>')


def _api_html(regression_type: Any, ink: Dict[str, str],
              language: Optional[str] = None) -> str:
    """The backend's API link, or "" when there is none to give."""
    name, url = model_api_link(regression_type, language)
    if not url:
        return ""
    return (f'<p style="margin:10px 0 2px 0;">'
            f'{_ink("API", ink["accent"], bold=True)} '
            f'<a href="{escape(url)}" style="color:{ink["accent"]};">'
            f'{escape(name)}</a></p>')


#: Brief guidance shown when nonparametric inference bypasses model fitting.
#: The separate Permutation Test section contains the full method description.
NONPARAMETRIC_NOTE = (
    "No model is fitted. Each guide is tested independently as a marginal "
    "association, with P values from plate-blocked permutations -- so there "
    "is no formula, no family, and no coefficient for a guide conditional "
    "on the others.",
    "The regression settings are greyed because this path never reads them. "
    "Their values are kept, so switching back restores the model you chose. "
    "The Permutation Test section explains the test itself.",
)


def _nonparametric_selected(inference: Any, analysis_mode: Any = "") -> bool:
    """Return whether the current settings select permutation inference.

    ``'auto'`` returns ``False`` because resolving it requires the data-dependent
    guide and well counts used by :func:`spacr.ml.resolve_auto_inference`.
    """
    from spacr.settings import INFERENCE_MODES

    name = str(inference or "auto").strip().lower()
    selected = INFERENCE_MODES.get(name)
    if selected is not None:
        return selected == "guide_permutation"
    return (name != "auto"
            and str(analysis_mode or "").strip().lower() == "guide_permutation")


def regression_model_explainer_html(regression_type: Any,
                                    level: Any = "both",
                                    plate_position: Any = False,
                                    random_row_column: Any = False,
                                    palette: Optional[Dict[str, Any]] = None,
                                    language: Optional[str] = None,
                                    inference: Any = "auto",
                                    analysis_mode: Any = "",
                                    ) -> str:
    """Render localized model or inference guidance as HTML.

    Parameters
    ----------
    regression_type : Any
        Selected regression backend.
    level : Any, default='both'
        Coefficient level: guide, gene, or both.
    plate_position : Any, default=False
        Include fixed plate-position terms when true.
    random_row_column : Any, default=False
        Use row and column variance components when true.
    palette : dict, optional
        Resolved theme palette. The active semantic color names are used when
        omitted.
    language : str, optional
        UI language. Missing or stale translations fall back by whole sentence.
    inference : Any, default='auto'
        Selected inference mode.
    analysis_mode : Any, default=''
        Compatibility value used to identify permutation inference.

    Returns
    -------
    str
        Rich text describing the run that the current settings will execute.
    """
    position = {"plate_position": bool(plate_position),
                "random_row_column": bool(random_row_column)}
    ink = _colours(palette)
    key = str(regression_type or "auto").strip().lower() or "auto"
    parts = [f'<div style="color:{ink["fg"]};">']

    def tx(source: str, **values: object) -> str:
        return _translated_ui_text(source, language, **values)

    if _nonparametric_selected(inference, analysis_mode):
        parts.append(
            f'<p>{_ink(escape(tx("INFERENCE:")), ink["accent"], bold=True)} '
            f'{escape(tx("nonparametric — guide permutation"))}</p>')
        for line in NONPARAMETRIC_NOTE:
            parts.append(_prose_html(tx(line), ink))
        parts.append("</div>")
        return "".join(parts)

    if key not in _MODE_NOTES:
        parts.append(f'<p>{_ink(escape(tx("MODEL:")), ink["accent"], bold=True)} '
                     f'{escape(key)}</p>')
        parts.append(_prose_html(tx(_UNKNOWN_MODEL_NOTE), ink))
        parts.append("</div>")
        return "".join(parts)

    title = tx(_MODE_TITLES.get(key, key))
    if key == "mixed":
        parts.append(
            f'<p style="margin:0 0 2px 0;">'
            f'{_ink(escape(tx("MODEL:")), ink["accent"], bold=True)} '
            f'mixed — '
            f'{escape(title)}<br/>'
            f'{_ink(escape(tx("LEVEL:")), ink["accent"], bold=True)} '
            f'{escape(tx("not applicable — one model carries both levels"))}'
            f'</p>')
        parts.append(_heading_html(tx("FORMULA"), ink))
        parts.append(_formula_html(maths_for("mixed", **position),
                                   formula_for(MIXED_TERM, **position), ink))
        parts.append(_heading_html(tx("WHAT IS MODELLED"), ink))
        parts.append(_prose_html(tx(_MODE_NOTES["mixed"]), ink))
        parts.append(_heading_html(
            tx(_REFUSAL_HEADING), ink, refusal=True,
        ))
        parts.append(_prose_html(tx(_MIXED_GUIDE_OUTPUT_NOTE), ink))
        # 133 A. `mixed` takes its own branch above, so the flag every other
        # backend gets from the shared path has to be added here too -- and
        # missing it on the DEFAULT would have been the one place it mattered
        # most.
        recommended_label = _ink(
            escape(tx("Recommended for CRISPR screens")),
            ink["success"],
            bold=True,
        )
        parts.append(
            f'<p style="margin:6px 0 2px 0;">{recommended_label}'
            f' — {escape(tx(RECOMMENDED_FOR_SCREENS["mixed"]))}</p>')
        parts.append(
            f'<p style="margin:2px 0 8px 0; color:{ink["fg_muted"]};">'
            f'{escape(tx(INFORMATION_LIMIT_NOTE))}</p>')
        parts.append(_heading_html(tx("WHAT IT COSTS"), ink))
        parts.append(_prose_html(mixed_cost_note(language), ink))
        parts.append(_heading_html(tx("MULTIPLE TESTING"), ink))
        parts.append(_prose_html(tx(_MIXED_MULTIPLE_TESTING_NOTE), ink))
    else:
        chosen = normalise_regression_level(level)
        level_line = {
            "both": "both — the two fits below, run SEPARATELY",
            "grna": "grna — the guide fit only",
            "gene": "gene — the gene fit only",
        }[chosen]
        parts.append(
            f'<p style="margin:0 0 2px 0;">'
            f'{_ink(escape(tx("MODEL:")), ink["accent"], bold=True)} '
            f'{escape(key)} — '
            f'{escape(title)}<br/>'
            f'{_ink(escape(tx("LEVEL:")), ink["accent"], bold=True)} '
            f'{escape(tx(level_line))}</p>')
        fixed_effects_note = escape(tx(
            "Fixed effects only — no nesting of guides inside genes."
        ))
        parts.append(f'<p style="margin:2px 0 6px 0; '
                     f'color:{ink["fg_muted"]};">'
                     f'{fixed_effects_note}</p>')
        if chosen in ("both", "grna"):
            parts.append(_heading_html(tx("FORMULA (guide fit)"), ink,
                                       writes="results_grna.csv"))
            parts.append(_formula_html(maths_for("grna", **position),
                                       formula_for(GRNA_TERM, **position),
                                       ink))
            parts.append(_prose_html(
                tx("One coefficient per guide, the unit the screen measures."),
                ink))
        if chosen in ("both", "gene"):
            parts.append(_heading_html(tx("FORMULA (gene fit)"), ink,
                                       writes="results_gene.csv"))
            parts.append(_formula_html(maths_for("gene", **position),
                                       formula_for(GENE_TERM, **position),
                                       ink))
            parts.append(_prose_html(
                tx("One coefficient per gene, from the summed guide fraction."),
                ink))
        if chosen == "both":
            parts.append(_prose_html(
                tx("TWO MODELS, TWO TABLES — fitted separately, NOT one "
                   "design containing both."), ink))
        parts.append(_heading_html(
            tx("WHAT {model} DOES", model=key.upper()), ink,
        ))
        parts.append(_prose_html(tx(_MODE_NOTES[key]), ink))
        # 133 A: say WHICH backends answer this question well, and WHY each.
        # In `success`, the same colour "TWO MODELS, TWO TABLES" uses, because
        # both are affirmations about the model rather than caveats about it.
        if key in RECOMMENDED_FOR_SCREENS:
            recommended_label = _ink(
                escape(tx("Recommended for CRISPR screens")),
                ink["success"],
                bold=True,
            )
            parts.append(
                f'<p style="margin:6px 0 2px 0;">{recommended_label}'
                f' — {escape(tx(RECOMMENDED_FOR_SCREENS[key]))}</p>')
            # AND THE CAVEAT, because a badge without it reads as a promise.
            parts.append(
                f'<p style="margin:2px 0 8px 0; color:{ink["fg_muted"]};">'
                f'{escape(tx(INFORMATION_LIMIT_NOTE))}</p>')
        parts.append(_heading_html(tx("MULTIPLE TESTING"), ink))
        if key in NO_P_VALUE_TYPES:
            source = (_NO_P_VALUE_BOTH_NOTE if chosen == "both"
                      else _NO_P_VALUE_SINGLE_NOTE)
            parts.append(_prose_html(tx(source), ink))
        elif chosen == "both":
            parts.append(_prose_html(
                tx("Each fit is its OWN multiple-testing family and is "
                   "BH-corrected within itself."), ink))
        else:
            parts.append(_prose_html(
                tx("The single fit is BH-corrected as one family."), ink))

    history_pointer = escape(tx(
        _HISTORY_POINTER_SOURCE,
        symbol=_HISTORY_POINTER_SYMBOL,
    ))
    parts.append(_api_html(key, ink, language))
    parts.append(f'<p style="margin:12px 0 0 0; color:{ink["fg_muted"]};">'
                 f'{history_pointer}</p>')
    parts.append("</div>")
    return "".join(parts)


def permutation_test_explainer_html(
        palette: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None) -> str:
    """Render localized permutation-test guidance as HTML.

    Parameters
    ----------
    palette : dict, optional
        Resolved theme palette.
    language : str, optional
        UI language. ``None`` uses the active language.

    Returns
    -------
    str
        Rich text with translated prose and unchanged formulas.
    """
    ink = _colours(palette)
    return (f'<div style="color:{ink["fg"]};">'
            + _heading_html(
                _translated_ui_text("WHAT THIS TEST DOES", language), ink)
            + _prose_html(
                _translated_ui_text(_PERMUTATION_NOTE, language), ink)
            + '</div>')


def section_explainer_html(app_key: str, title: str,
                           settings: Optional[Dict[str, Any]] = None,
                           palette: Optional[Dict[str, Any]] = None,
                           language: Optional[str] = None) -> str:
    """Return localized HTML guidance for a settings section.

    Parameters
    ----------
    app_key : str
        Application whose settings section is rendered.
    title : str
        Canonical English section title.
    settings : dict, optional
        Current values used to render formulas and selected inference.
    palette : dict, optional
        Resolved theme palette.
    language : str, optional
        UI language. ``None`` uses the active language.

    Returns
    -------
    str
        Rich text, or ``""`` when the section has no explainer.
    """
    if not has_section_explainer(app_key, title):
        return ""
    values = settings or {}
    if title == "Model & Inference":
        return regression_model_explainer_html(
            values.get("regression_type", "auto"),
            values.get("level", "both"),
            plate_position=values.get("model_plate_position", False),
            random_row_column=values.get("random_row_column_effects", False),
            palette=palette,
            language=language)
    return permutation_test_explainer_html(palette, language)


def explainer_width() -> int:
    """Return the minimum explainer width in monospace characters.

    The width is derived from the longest unbreakable formula. Prose remains
    free to wrap to the available panel width.
    """
    longest = _EXPLAINER_WIDTH
    for text in _every_explainer_line():
        if text.strip().startswith(("y ~", "rho =", "minimise")):
            longest = max(longest, len(text))
    return longest


def _every_explainer_line():
    """Return every line that an explainer may render.

    The result supplies representative content to :func:`explainer_width`.
    """
    from spacr.regression_spec import REGRESSION_TYPES

    lines = []
    positions = (
        {"plate_position": False, "random_row_column": False},
        {"plate_position": True, "random_row_column": False},
        {"plate_position": True, "random_row_column": True},
    )
    for family in REGRESSION_TYPES:
        for level in REGRESSION_LEVELS:
            for position in positions:
                try:
                    lines.extend(regression_model_explainer(
                        family, level, **position).splitlines())
                except Exception:                              # noqa: BLE001
                    continue
    return lines


def _wrap_block(text: str, indent: str = "    ") -> str:
    """Indent a paragraph while leaving line wrapping to the widget.

    Formula lines bypass this helper so they remain copyable as complete
    expressions. Prose stays on one logical line and adapts to the current
    width of the explainer pane.
    """
    out = []
    for paragraph in str(text).split("\n"):
        if not paragraph.strip():
            out.append("")
            continue
        out.append(indent + " ".join(paragraph.split()))
    return "\n".join(out)


def normalise_regression_level(level: Any) -> str:
    """Return a supported regression level, defaulting to ``'both'``.

    Missing or unrecognized values can occur in settings saved by older
    versions and are handled without interrupting panel rendering.
    """
    text = str(level or "").strip().lower()
    return text if text in REGRESSION_LEVELS else "both"


def regression_model_explainer(regression_type: Any,
                               level: Any = "both",
                               plate_position: Any = False,
                               random_row_column: Any = False,
                               language: Optional[str] = None,
                               inference: Any = "auto",
                               analysis_mode: Any = "") -> str:
    """Describe the regression formula selected in the settings panel.

    Parameters
    ----------
    regression_type : Any
        Requested regression backend, such as ``"ols"`` or ``"mixed"``.
    level : Any, default="both"
        Coefficient level to describe: ``"grna"``, ``"gene"``, or ``"both"``.
    plate_position : Any, default=False
        Whether the formula includes row and column position terms.
    random_row_column : Any, default=False
        Whether row and column terms are variance components instead of fixed
        effects.
    language : str or None, default=None
        UI language code. ``None`` uses the active language. Only exact,
        source-current paragraph translations are used.
    inference : Any, default='auto'
        Selected inference mode.
    analysis_mode : Any, default=''
        Compatibility value used to identify permutation inference.

    Returns
    -------
    str
        Plain text containing the selected model, fitted formula, output, and
        interpretation notes. Unknown backends receive an explicit warning.

    Notes
    -----
    Guide and gene effects are described as separate fits. The retired design,
    ``y ~ fraction:grna + gene_fraction:gene + rowID + columnID``, contains
    both guide fractions and their gene-level sums. It is rank deficient, so
    its individual coefficients are not uniquely interpretable.
    :data:`COLLINEAR_FORMULA` stores the formula used by the compatibility
    checks.
    """
    position = {"plate_position": bool(plate_position),
                "random_row_column": bool(random_row_column)}
    key = str(regression_type or "auto").strip().lower() or "auto"

    def tx(source: str, **values: object) -> str:
        return _translated_ui_text(source, language, **values)

    if _nonparametric_selected(inference, analysis_mode):
        # THE SAME WORDS AS THE TYPESET BOX, from the one constant, so the
        # plain renderer and the HTML one cannot describe different runs.
        return "\n\n".join(
            [tx("INFERENCE: nonparametric — guide permutation")]
            + [tx(line) for line in NONPARAMETRIC_NOTE])

    if key not in _MODE_NOTES:
        # An unknown name is the pipeline's error to raise, with its own list
        # of what it accepts. The box says it cannot describe the choice
        # rather than inventing a formula for it.
        return (f"{tx('MODEL:')} {key}\n\n"
                + _wrap_block(tx(_UNKNOWN_MODEL_NOTE)))

    title = tx(_MODE_TITLES.get(key, key))
    lines: List[str] = []

    if key == "mixed":
        lines.append(f"{tx('MODEL:')} mixed -- {title}")
        lines.append(
            f"{tx('LEVEL:')} "
            f"{tx('not applicable -- one model carries both levels')}"
        )
        lines.append("")
        lines.append(tx("FORMULA"))
        lines.append(f"    {formula_for(MIXED_TERM, **position)}")
        lines.append("")
        lines.append(tx("WHAT IS MODELLED"))
        lines.append(_wrap_block(tx(_MODE_NOTES["mixed"])))
        lines.append("")
        # THE COST OF THE DEFAULT, in its own named section. This is the
        # paragraph the box exists for, and the one section instruction 143
        # left at full length: a user who takes the default and then goes
        # looking for guide p-values reads it exactly once, but they cannot
        # be told to go elsewhere for it.
        lines.append(tx(_REFUSAL_HEADING))
        lines.append(_wrap_block(tx(_MIXED_GUIDE_OUTPUT_NOTE)))
        lines.append("")
        # WHAT IT COSTS, beside "what you do not get" and for the same
        # reason: both are things a user can only find out by having already
        # spent the afternoon. Instruction 140.
        lines.append(tx("Recommended for CRISPR screens").upper())
        lines.append(_wrap_block(tx(RECOMMENDED_FOR_SCREENS["mixed"])))
        lines.append(_wrap_block(tx(INFORMATION_LIMIT_NOTE)))
        lines.append("")
        lines.append(tx("WHAT IT COSTS"))
        lines.append(_wrap_block(mixed_cost_note(language)))
        lines.append("")
        lines.append(tx("MULTIPLE TESTING"))
        lines.append(_wrap_block(tx(_MIXED_MULTIPLE_TESTING_NOTE)))
    else:
        chosen = normalise_regression_level(level)
        level_line = {
            "both": "both -- the two fits below, run SEPARATELY",
            "grna": "grna -- the guide fit only",
            "gene": "gene -- the gene fit only",
        }[chosen]
        lines.append(f"{tx('MODEL:')} {key} -- {title}")
        lines.append(f"{tx('LEVEL:')} {tx(level_line)}")
        lines.append("")
        lines.append(_wrap_block(
            tx("Fixed effects only -- no nesting of guides inside genes."),
            ""))
        lines.append("")

        # ONE SENTENCE UNDER EACH FORMULA, and no blank line between them:
        # the sentence says what a coefficient IS, which is the one thing a
        # reader needs it for, and it belongs to the formula above it.
        if chosen in ("both", "grna"):
            lines.append(
                f"{tx('FORMULA (guide fit)')}  ->  results_grna.csv"
            )
            lines.append(f"    {formula_for(GRNA_TERM, **position)}")
            lines.append(_wrap_block(
                tx("One coefficient per guide, the unit the screen "
                   "measures.")))
            lines.append("")
        if chosen in ("both", "gene"):
            lines.append(
                f"{tx('FORMULA (gene fit)')}   ->  results_gene.csv"
            )
            lines.append(f"    {formula_for(GENE_TERM, **position)}")
            lines.append(_wrap_block(
                tx("One coefficient per gene, from the summed guide "
                   "fraction.")))
            lines.append("")
        if chosen == "both":
            lines.append(_wrap_block(
                tx("TWO MODELS, TWO TABLES -- fitted separately, NOT one "
                   "design containing both."), ""))
            lines.append("")

        if key in RECOMMENDED_FOR_SCREENS:
            lines.append(tx("Recommended for CRISPR screens").upper())
            lines.append(_wrap_block(tx(RECOMMENDED_FOR_SCREENS[key])))
            lines.append(_wrap_block(tx(INFORMATION_LIMIT_NOTE)))
            lines.append("")
        lines.append(tx("WHAT {model} DOES", model=key.upper()))
        lines.append(_wrap_block(tx(_MODE_NOTES[key])))
        lines.append("")
        lines.append(tx("MULTIPLE TESTING"))
        if key in NO_P_VALUE_TYPES:
            # Saying "BH-corrected" under a backend that reports no p-value
            # would contradict this box's own WHAT ... DOES paragraph two
            # lines above it.
            source = (_NO_P_VALUE_BOTH_NOTE if chosen == "both"
                      else _NO_P_VALUE_SINGLE_NOTE)
            lines.append(_wrap_block(tx(source)))
        elif chosen == "both":
            # ITS FIRST SENTENCE ONLY, per instruction 143. The four that
            # followed said why pooling would be wrong and warned that a gene
            # called by both fits is two tests of one hypothesis -- both true,
            # both read once, and the second belongs beside the hit list where
            # somebody is making the claim.
            lines.append(_wrap_block(
                tx("Each fit is its OWN multiple-testing family and is "
                   "BH-corrected within itself.")))
        else:
            lines.append(_wrap_block(
                tx("The single fit is BH-corrected as one family.")))

    lines.append("")
    lines.append(tx(
        _HISTORY_POINTER_SOURCE,
        symbol=_HISTORY_POINTER_SYMBOL,
    ))
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# The Permutation Test explainer box (instruction 135)
# ---------------------------------------------------------------------------
#
# "Permutation test is good it just needs a text box at the top briefly
# explaining what it does."  ONE PARAGRAPH, and shorter than the model box
# above: the eight controls under it are already named for what they do, so
# what is missing is only the sentence that says what the test IS.
#
# Written from `spacr.guide_permutation` rather than from the general
# reputation of permutation tests, which is why it says "marginal" out loud.
# The module's own docstring is explicit that it "does not claim to estimate a
# simultaneous conditional coefficient for every guide", and a user who reads
# "permutation test" as "the same fit, only distribution-free" would take a
# marginal association for a conditional one.

#: What the nonparametric branch actually runs, in one paragraph.
#:
#: Every clause is a line of `guide_freedman_lane_test`: the block-wise
#: reshuffle is its `for indexes in block_indexes` loop, the two-sided
#: comparison is `np.abs(permutation_effects) >= np.abs(observed)`, the floor
#: on the P value is `(exceedances + 1) / (n_permutations + 1)`, and the
#: per-threshold family is the `for threshold in thresholds` loop that
#: corrects each support level on its own.
_PERMUTATION_NOTE = (
    # BOTH THE PLAIN WORDS AND THE TERM. The longhand -- "one coefficient in
    # a design holding every guide at once" -- is what a reader who does not
    # know the vocabulary needs; "conditional coefficients" is what a reader
    # who does will look for, and it is the phrase
    # `guide_freedman_lane_test`'s own docstring uses when it says the test
    # "does not claim to estimate a simultaneous conditional coefficient".
    # Dropping the term left the distinction true but unsearchable.
    "Each guide is tested independently, as a marginal association rather "
    "than as one coefficient in a design holding every guide at once -- so "
    "these are marginal associations, not conditional coefficients. Its "
    "read fraction and the well phenotype are first residualised against "
    "the block (normally plateID) and any nuisance columns; the P value is "
    "then the share of Freedman-Lane permutations -- the phenotype residual "
    "reshuffled WITHIN each block -- whose statistic reaches the observed "
    "one, so it is empirical and two-sided and can never be smaller than "
    "1/(permutations + 1). A guide becomes testable once it appears in "
    "guide_min_wells wells above guide_presence_threshold, and each of "
    "those thresholds is corrected as its own family. This avoids the rank "
    "requirement of a simultaneous guide model and can be used when guides "
    "outnumber wells; interpretation still depends on valid blocking, "
    "exchangeability, and adequate guide support."
)


#: The head of the regression menu. 'auto' is NOT a family -- it is the
#: readable spelling of the historical ``None``, which ``ml.regression`` turns
#: into ``check_distribution(response)`` -- so it carries no group title and
#: no assumption, and it must not be labelled as though it were one.
_REGRESSION_AUTO_CHOICE = (
    "auto",
    "auto — chosen from the response by check_distribution",
)


def _regression_type_menu():
    """Every entry of the ``regression_type`` dropdown, as (value, caption).

    ONE TABLE FOR BOTH ROUTES. The families and their captions come from
    :func:`spacr.regression_families.regression_family_choices`, which
    :func:`spacr.settings_spec._regression_type_choices` also asks -- so the
    Qt panel and the settings spec cannot disagree about what a family is
    called or which of the three kinds it is in. This panel used to build its
    own flat list out of the bare inventory, and the two routes did disagree:
    one showed nineteen unlabelled names, the other showed them explained.

    ONE LIST FOR THE MENU AND FOR THE CATALOG. ``_SETTINGS_MODEL_UI_SOURCES``
    is built from this, so a caption that reaches the dropdown reaches the
    translators with it and cannot be left behind as the only English row in
    a Swedish panel.

    Asked of ``spacr.regression_families`` rather than ``spacr.ml``: both
    re-export the function, but ``spacr.ml`` imports ``spacr.plot`` and
    therefore torch, which is 2.2 seconds and 900 MB on the GUI thread to
    read a tuple of strings.

    :returns: ``[('auto', caption), (family, caption), ...]`` -- 'auto'
        first, then parametric, robust/semiparametric and rank-based, which
        is the order a reader meets the three kinds in.
    """
    from spacr.regression_families import regression_family_choices

    return [_REGRESSION_AUTO_CHOICE, *regression_family_choices()]


#: Every caption the regression menu shows, for the catalog builder.
#:
#: A CAPTION SHIPS WITH ITS ROWS. These are assembled at runtime from
#: :mod:`spacr.regression_families`, so the literal-string extractor in
#: ``tools/build_i18n_catalogs.py`` cannot see them at the ``addItem`` call
#: site the way it sees a quoted label. Declaring them here is how a
#: dynamically composed caption still reaches the translators.
#:
#: SEPARATE FROM ``_SETTINGS_MODEL_UI_SOURCES`` ON PURPOSE. That set is
#: pinned by ``tests/qt/test_external_i18n_catalogs.py`` to exactly the
#: templates the model explainers render, so it is that surface's inventory
#: and not this module's. Folding a menu caption into it would make the
#: explainer check fail on a string no explainer has ever rendered.
_REGRESSION_MENU_UI_SOURCES = frozenset(
    caption for _value, caption in _regression_type_menu())


_SETTINGS_MODEL_UI_SOURCES = frozenset({
    *_MODE_TITLES.values(),
    *RECOMMENDED_FOR_SCREENS.values(),
    *_MODE_NOTES.values(),
    INFORMATION_LIMIT_NOTE,
    _HISTORY_POINTER_SOURCE,
    _MIXED_COST_NOTE_TEMPLATE,
    _REFUSAL_HEADING,
    _MIXED_GUIDE_OUTPUT_NOTE,
    _MIXED_MULTIPLE_TESTING_NOTE,
    _UNKNOWN_MODEL_NOTE,
    _NO_P_VALUE_BOTH_NOTE,
    _NO_P_VALUE_SINGLE_NOTE,
    _PERMUTATION_NOTE,
    "MODEL:",
    "LEVEL:",
    "not applicable — one model carries both levels",
    "not applicable -- one model carries both levels",
    "FORMULA",
    "WHAT IS MODELLED",
    "Recommended for CRISPR screens",
    "WHAT IT COSTS",
    "MULTIPLE TESTING",
    "both — the two fits below, run SEPARATELY",
    "grna — the guide fit only",
    "gene — the gene fit only",
    "both -- the two fits below, run SEPARATELY",
    "grna -- the guide fit only",
    "gene -- the gene fit only",
    "Fixed effects only — no nesting of guides inside genes.",
    "Fixed effects only -- no nesting of guides inside genes.",
    "FORMULA (guide fit)",
    "FORMULA (gene fit)",
    "One coefficient per guide, the unit the screen measures.",
    "One coefficient per gene, from the summed guide fraction.",
    "TWO MODELS, TWO TABLES — fitted separately, NOT one design containing "
    "both.",
    "TWO MODELS, TWO TABLES -- fitted separately, NOT one design containing "
    "both.",
    "WHAT {model} DOES",
    "Each fit is its OWN multiple-testing family and is BH-corrected within "
    "itself.",
    "The single fit is BH-corrected as one family.",
    "WHAT THIS TEST DOES",
})


def permutation_test_explainer(
    language: Optional[str] = None,
) -> str:
    """Return localized plain-text permutation-test guidance.

    Parameters
    ----------
    language : str, optional
        UI language. ``None`` uses the active language.

    Returns
    -------
    str
        Wrapped guidance, using canonical English when a complete translation
        is unavailable.
    """
    return (_translated_ui_text("WHAT THIS TEST DOES", language) + "\n"
            + _wrap_block(
                _translated_ui_text(_PERMUTATION_NOTE, language)) + "\n")


#: Sections that open with a read-only prose box instead of a control, per
#: module.
#:
#: A table keeps placement and coverage in one place and makes additional
#: explainer sections explicit.
SECTION_EXPLAINERS: Dict[str, Tuple[str, ...]] = {
    "regression": ("Model & Inference", "Permutation Test"),
}


def has_section_explainer(app_key: str, title: str) -> bool:
    """Return whether a settings section begins with explanatory prose."""
    return str(title or "") in SECTION_EXPLAINERS.get(str(app_key or ""), ())


def section_explainer(app_key: str, title: str,
                      settings: Optional[Dict[str, Any]] = None,
                      language: Optional[str] = None) -> str:
    """Return localized plain-text guidance for a settings section.

    Parameters
    ----------
    app_key : str
        Application whose section is rendered.
    title : str
        Section heading.
    settings : dict, optional
        Current values used to render formulas and selected inference.
    language : str, optional
        UI language. ``None`` uses the active language.

    Returns
    -------
    str
        Guidance text, or ``""`` when the section has no explainer.
    """
    if not has_section_explainer(app_key, title):
        return ""
    values = settings or {}
    if title == "Model & Inference":
        return regression_model_explainer(
            values.get("regression_type", "auto"),
            values.get("level", "both"),
            plate_position=values.get("model_plate_position", False),
            random_row_column=values.get("random_row_column_effects", False),
            language=language)
    return permutation_test_explainer(language)


def _basis_note(basis: str) -> str:
    """The sentence shown on a setting the current training basis ignores."""
    return (f"Not used when the training basis is '{basis}'. "
            f"The value is kept and still saved.")


def _family_note(family: str) -> str:
    """The sentence shown on a setting the chosen classifier ignores."""
    return (f"Not used by the '{family}' classifier. The value is "
            f"kept and still saved.")


def _apply_greyed_note(control, note: str) -> None:
    """Append a disabled-state note without replacing the setting help.

    Existing tooltip text and API-link properties remain intact so labels
    and fields expose the same documentation while the control is disabled.
    """
    _clear_greyed_note(control)     # the reason may have changed; it is named
    base = control.property("apiTooltipHtml") or control.toolTip()
    control.setProperty(_BASIS_NOTE_PROPERTY, True)
    control.setToolTip(f"{base}<br><i>{note}</i>" if base else note)
    # REMEMBERED ON THE CONTROL, because the label may not exist yet. The
    # first greying pass runs while the panel is being built and the labels
    # are decorated afterwards, so a note written only where a label would be
    # is a note that never appears. Held here, it can be put on the label the
    # moment there is one.
    control.setProperty(_PENDING_NOTE_PROPERTY, note)
    label = getattr(control, "_spacr_setting_label", None)
    if label is not None:
        label.setEnabled(False)
        # ON THE LABEL, WHICH IS WHERE THE HELP ACTUALLY SHOWS. The editor is
        # deliberately SILENT on hover -- decoration sets its display role to
        # "metadata" and clears its tooltip so the panel does not show two
        # tooltips for one setting -- so the note above went to a string
        # nothing reads. EVERY greyed setting in spaCR was disabled WITHOUT
        # saying why, which is the one thing instruction 106 asks of a greyed
        # control, and it was invisible precisely because the reason was
        # written where it could not be seen.
        _note_on_label(label, note)


def _note_on_label(label, note: str) -> None:
    """Append the greyed-out reason to the help the LABEL shows on hover.

    The original help is kept under its own property so
    :func:`_clear_greyed_note` restores it exactly rather than trying to
    strip the note back off a rendered string.
    """
    if label.property(_NOTE_BACKUP_PROPERTY) is None:
        base = str(label.property("apiTooltipHtml") or label.toolTip() or "")
        # A note may already be BAKED IN: the label's help was composed from
        # the control's tooltip at decoration time, and that tooltip carried
        # the note from the build-time greying pass. Stripped by its exact
        # text, which is known, rather than by a pattern -- guessing where
        # help ends and a note begins is how a restore loses a sentence.
        base = _without_note(base, note)
        label.setProperty(_NOTE_BACKUP_PROPERTY, base)
    base = str(label.property(_NOTE_BACKUP_PROPERTY) or "")
    text = f"{base}<br><i>{note}</i>" if base else note
    label.setProperty("apiTooltipHtml", text)
    label.setToolTip(text)


def _without_note(text: str, note: str) -> str:
    """``text`` with a trailing greyed-out ``note`` removed, if it has one."""
    for suffix in (f"<br><i>{note}</i>", note):
        if suffix and text.endswith(suffix):
            return text[:-len(suffix)].rstrip()
    return text


def _clear_greyed_note(control) -> None:
    """Put the setting's own help back when it applies again."""
    if not control.property(_BASIS_NOTE_PROPERTY):
        return
    control.setProperty(_BASIS_NOTE_PROPERTY, False)
    restored = control.property("apiTooltipHtml")
    if restored:
        control.setToolTip(restored)
    pending = str(control.property(_PENDING_NOTE_PROPERTY) or "")
    control.setProperty(_PENDING_NOTE_PROPERTY, None)
    label = getattr(control, "_spacr_setting_label", None)
    if label is not None:
        label.setEnabled(control.isEnabled())
        backup = label.property(_NOTE_BACKUP_PROPERTY)
        if backup is not None:
            label.setProperty("apiTooltipHtml", backup)
            label.setToolTip(str(backup))
            label.setProperty(_NOTE_BACKUP_PROPERTY, None)
        elif pending:
            # No backup because the note was applied before this label
            # existed and was baked into its help by decoration. Removed by
            # its own text.
            cleaned = _without_note(
                str(label.property("apiTooltipHtml") or label.toolTip() or ""),
                pending)
            label.setProperty("apiTooltipHtml", cleaned)
            label.setToolTip(cleaned)


def attach_api_tooltip(
    widget: QWidget,
    app_key: str,
    key: str,
    description: str = "",
    _descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Attach typed, linked API help metadata to one setting widget."""
    descriptions = _descriptions if _descriptions is not None else get_tooltips()
    existing_tooltip = "" if widget.property("apiTooltipHtml") else widget.toolTip()
    body = (descriptions.get(key) or description
            or widget.property("apiTooltipDescriptionSource")
            or widget.property("apiTooltipDescription")
            or existing_tooltip)
    # Keep an absent body absent: format_tooltip owns the localized generic
    # fallback.  Synthesizing an English sentence here bypasses it.
    body = str(body or "")
    html = format_tooltip(body, app_key, key)
    widget.setProperty("settingsAppKey", app_key)
    widget.setProperty("settingKey", key)
    widget.setProperty("apiTooltipDescriptionSource", body)
    # Retain the old property as canonical English for integrations that read
    # it, rather than replacing it with rendered/localized HTML.
    widget.setProperty("apiTooltipDescription", body)
    widget.setProperty("apiTooltipHtml", html)
    if widget.property("apiTooltipDisplayRole") is None:
        widget.setProperty("apiTooltipDisplayRole", "tooltip")
    widget.setToolTip(html)
    widget.setToolTipDuration(-1)
    return html


def refresh_api_tooltips(
    root: QWidget,
    language: Optional[str] = None,
) -> None:
    """Refresh semantic setting help beneath ``root`` in ``language``.

    Canonical English prose is retained in ``apiTooltipDescriptionSource``;
    only the presentation HTML/plain accessibility chrome is regenerated.
    Field widgets marked ``metadata`` stay quiet because their visible label
    owns hover help. API-dot destinations carry the selected documentation
    language while retaining the same module page.
    """
    if root is None:
        return
    from ..i18n import tr

    code = _language_code(language)
    widgets = [root]
    try:
        widgets.extend(root.findChildren(QWidget))
    except (AttributeError, RuntimeError):
        return

    descriptions: Optional[Dict[str, str]] = None
    for widget in widgets:
        try:
            app_key = widget.property("settingsAppKey")
            key = widget.property("settingKey")
        except RuntimeError:
            continue
        if not app_key or not key:
            continue
        source = (widget.property("apiTooltipDescriptionSource")
                  or widget.property("apiTooltipDescription"))
        if not source:
            if descriptions is None:
                descriptions = get_tooltips()
            source = descriptions.get(str(key), "")
        source = str(source or "")
        html = format_tooltip(source, str(app_key), str(key), code)
        widget.setProperty("apiTooltipDescriptionSource", source)
        widget.setProperty("apiTooltipDescription", source)
        widget.setProperty("apiTooltipHtml", html)

        role = str(widget.property("apiTooltipDisplayRole") or "tooltip")
        if role == "metadata":
            widget.setToolTip("")
        elif role == "api-link":
            caption = _api_reference_tooltip(str(key), code, str(app_key))
            set_url = getattr(widget, "set_url", None)
            if callable(set_url):
                set_url(api_docs_url(str(app_key), str(key), code))
            widget.setToolTip(caption)
            widget.setAccessibleName(caption)
            widget.setAccessibleDescription(
                tr("Open spaCR API documentation", code))
        else:
            widget.setToolTip(html)
            widget.setToolTipDuration(-1)


def install_api_tooltips(
    owner: QWidget,
    app_key: str,
    widget_keys: Optional[Dict[QWidget, str]] = None,
) -> None:
    """Give every mapped/generated popup setting label consistent API help.

    ``SettingsWidgets`` controls are discovered through their ``settingKey``
    property. Hand-built Live/Crop/Search controls are supplied in
    ``widget_keys``. Descriptive help belongs to the label, not the editable
    field, and the whole of it -- description and API link both -- is in the
    label's hover text.

    NOTHING IS DRAWN BESIDE THE LABEL. A teal link dot used to be, and three
    forms had already switched it off one at a time: 68 of them down the Mask
    live preview, twenty-six down the Annotate settings dialog, three in the
    figure dialog. A column of dots reads as texture rather than as one
    affordance per setting, and the API link was never in the dot alone --
    it is in the hover text, which is where it was being read from.
    """
    event_filter = getattr(owner, "_api_tooltip_filter", None)
    if event_filter is None:
        event_filter = _ApiTooltipFilter(owner)
        owner._api_tooltip_filter = event_filter

    mapped = dict(widget_keys or {})
    for widget in owner.findChildren(QWidget):
        if widget.property("settingHelpLabel"):
            continue
        # The dot this pass CREATES carries `settingKey` itself, so it is
        # found by the sweep the next time it runs and decorated as though it
        # were a setting — each one growing its own dot. That is what made the
        # live-preview panel sprout duplicates every time the form was re-gated
        # (switching Primary object from cell to nucleus).
        # It is help, not a setting; skip it.
        if widget.property("apiTooltipDisplayRole") == "api-link":
            continue
        key = widget.property("settingKey")
        if key and widget not in mapped:
            mapped[widget] = str(key)
    descriptions = get_tooltips()
    for widget, key in mapped.items():
        # Explicitly hidden controls are not settings in this popup. Decorating
        # one would create a visible wrapper/dot with a hidden field at (0, 0),
        # recreating the very kind of orphan overlay this helper should avoid.
        if widget.isHidden():
            continue
        html = attach_api_tooltip(
            widget, app_key, key, _descriptions=descriptions)
        label = _setting_label_for_field(owner, widget)
        if label is None and not _is_self_labelling(widget):
            # A COMPOSITE FIELD IS NOT A SELF-LABELLING CONTROL, and treating
            # it as one is what put the tooltip on the field.
            #
            # Reported repeatedly, and measured on the regression panel:
            # THIRTY-THREE editors sit inside a composite -- a `_ScalarEdit`
            # inside a `_CsvColumnField`, a line edit inside a chip field --
            # and the composite was landing in the branch below, which
            # installs the hover filter on the widget itself. Qt delivers
            # `Enter` to a parent when the pointer crosses into any of its
            # children, so hovering the FIELD fired the help.
            #
            # The branch below is right for a `QCheckBox`, which carries its
            # own visible text and IS its own label. It is wrong for a
            # container, which has no text and whose label is elsewhere or
            # missing. Where there is no label to put the help on, the help
            # goes nowhere -- a field that stays quiet is the requested
            # behaviour, and a tooltip on the field is not a lesser version
            # of it.
            widget.setProperty("apiTooltipHtml", "")
            widget.setProperty("apiTooltipDisplayRole", "metadata")
            widget.setToolTip("")
            widget.removeEventFilter(event_filter)
            continue
        if label is None:
            # A one-widget form row (usually a Toggle/QCheckBox) carries its
            # own visible label, so the hover help goes on its own text.
            # Remove before installing. Qt keeps a LIST of filters and calls
            # each installation separately, so decorating the same widget
            # twice makes one hover emit two tooltips.
            # `removeEventFilter` is a no-op when the filter is not
            # installed, which makes this idempotent for free.
            widget.removeEventFilter(event_filter)
            widget.installEventFilter(event_filter)
            continue

        body_source = str(widget.property("apiTooltipDescriptionSource") or "")
        label.setCursor(Qt.WhatsThisCursor)
        label.setProperty("settingHelpLabel", True)
        label.setProperty("settingsAppKey", app_key)
        label.setProperty("settingKey", key)
        label.setProperty("apiTooltipDescriptionSource", body_source)
        label.setProperty("apiTooltipDescription", body_source)
        label.setProperty("apiTooltipHtml", html)
        label.setProperty("apiTooltipDisplayRole", "tooltip")
        label.setToolTip(html)
        label.setToolTipDuration(-1)
        # Idempotent: this decoration pass runs again whenever the
        # live-preview form is re-gated -- changing the primary object from
        # cell to nucleus, for instance -- and a second installation on the
        # same label duplicated every tooltip on the panel.
        label.removeEventFilter(event_filter)
        label.installEventFilter(event_filter)

        # The editor itself remains quiet on hover. Keep its metadata so tests,
        # integrations and a later re-parenting pass can still identify it.
        widget.setProperty("apiTooltipDisplayRole", "metadata")
        widget.setToolTip("")
        widget.removeEventFilter(event_filter)


def _unwrap_setting_label(candidate: Optional[QWidget]) -> Optional[QWidget]:
    """Return the real label inside a `SettingLabelWithInfo` host.

    A section builds that host to right-align a label against its field, so
    ``QFormLayout.labelForField`` hands back the HOST rather than the label —
    a widget with none of the label's guard properties, which the decoration
    pass then decorated again, giving the panel a second tooltip per setting.
    That is what switching Primary object from cell to nucleus did in the
    Mask live preview.

    Unwrapping restores the invariant the guards rely on: the same label
    object is found every time.
    """
    if candidate is None:
        return None
    if candidate.objectName() != "SettingLabelWithInfo":
        return candidate
    for child in candidate.findChildren(QWidget):
        if child.property("settingHelpLabel"):
            return child
    return candidate


def _setting_label_for_field(owner: QWidget, field: QWidget) -> Optional[QWidget]:
    """Find the visual label immediately to the left of a popup field."""
    remembered = getattr(field, "_spacr_setting_label", None)
    if isinstance(remembered, QWidget):
        try:
            remembered.objectName()
            if remembered.window() is owner.window():
                return _unwrap_setting_label(remembered)
        except RuntimeError:
            pass

    for form in owner.findChildren(QFormLayout):
        # A form field is often a wrapper QWidget containing an editor and a
        # Browse button (or two numeric editors). QFormLayout only knows the
        # wrapper, so walk the editor's parent chain before concluding that it
        # is a label-less combined control. Otherwise its hover help ends up
        # on the editor instead of on the form label.
        candidate: Optional[QWidget] = field
        while isinstance(candidate, QWidget):
            label = _unwrap_setting_label(form.labelForField(candidate))
            if isinstance(label, QWidget):
                field._spacr_setting_label = label
                return label
            if candidate is owner:
                break
            candidate = candidate.parentWidget()

    # Hand-built search panels use compact grids rather than QFormLayout.
    # Select the nearest widget to the field's left on the same row.
    for grid in owner.findChildren(QGridLayout):
        index = grid.indexOf(field)
        if index < 0:
            continue
        row, column, _row_span, _column_span = grid.getItemPosition(index)
        for candidate_column in range(column - 1, -1, -1):
            item = grid.itemAtPosition(row, candidate_column)
            candidate = item.widget() if item is not None else None
            if isinstance(candidate, QLabel):
                field._spacr_setting_label = candidate
                return candidate
    return None


# ---------------------------------------------------------------------------
# Widget factory
# ---------------------------------------------------------------------------

class _ListEdit(QLineEdit):
    """A QLineEdit that round-trips a Python list via repr()."""
    def get_value(self) -> Any:
        """Return the field parsed as a Python literal (or raw text on failure)."""
        text = self.text().strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except Exception:
            return text

    def set_value(self, v: Any) -> None:
        """Render ``v`` into the field via ``repr``; ``None`` clears the field."""
        self.setText(repr(v) if v is not None else "")


class _ValueCombo(QComboBox):
    """A dropdown settable by the value it stores, not only by its caption.

    Every entry is added as ``addItem(caption, userData=value)``, and for most
    settings the two are the same string. They are not the same for a menu
    that explains itself: ``regression_type`` stores ``'quantile'`` and shows
    ``'quantile -- robust/semiparametric: ...'`` so the user can tell the
    nineteen families apart.

    Qt's ``setCurrentText`` matches the CAPTION and, on a non-editable combo,
    silently does nothing when there is no match. So the ordinary way to say
    "choose ols" -- ``combo.setCurrentText('ols')`` -- becomes a no-op the
    moment a caption stops being its own value, and the control is left on
    whatever it was showing while the caller believes it was set. Nothing
    raises and nothing is logged; the run simply fits a different model.

    Matching the caption FIRST keeps Qt's own behaviour exactly, and falling
    back to the stored value adds the case that used to vanish.
    """

    def setCurrentText(self, text: Any) -> None:                # noqa: N802
        """Select the entry whose caption -- or, failing that, whose stored
        value -- is ``text``.

        :param text: a caption or a stored value. Anything matching neither
            leaves the selection alone on a non-editable combo, which is what
            Qt does.
        """
        wanted = "" if text is None else str(text)
        index = self.findText(wanted)
        if index < 0:
            index = self.findData(wanted)
        if index >= 0:
            self.setCurrentIndex(index)
            return
        super().setCurrentText(text)


class _HiddenRowWatcher(QObject):
    """Tells a :class:`SettingsWidgets` that one of its hidden rows is back.

    An event filter rather than a signal, because the thing that put the row
    back does not know the rule exists -- the settings-search strip shows
    every row it indexed when nothing is narrowing, and a recipe or a fold
    reaches the panel by a different door again. ``ShowToParent`` is the one
    event every route has in common: Qt delivers it on ``setVisible(True)``
    even when the widget's ancestors are hidden, which is the case that
    matters here because a settings section is usually collapsed.

    A WEAK REFERENCE TO THE MODEL, AND A QT PARENT. The model owns this and
    this is installed on the model's own widgets, so a strong reference back
    would make a cycle with a QObject in it -- and a QObject destroyed by
    Python's cyclic collector, while it is still an event filter on several
    hundred live widgets, is the shape of crash that is impossible to read
    afterwards. Parented to the panel instead, so Qt decides when it dies and
    unhooks it from everything it watches on the way out.
    """

    def __init__(self, model: "SettingsWidgets",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = weakref.ref(model)

    def eventFilter(self, obj, event) -> bool:            # noqa: N802
        if event.type() == QEvent.ShowToParent:
            model = self._model()
            if model is None:
                return False
            try:
                model._shown_against_the_rule(obj)
            except Exception:                                # noqa: BLE001
                LOGGER.debug("could not re-assert object visibility",
                             exc_info=True)
        return False


class _ScalarEdit(QLineEdit):
    """A plain QLineEdit that returns None for empty text."""
    def get_value(self) -> Optional[str]:
        """Return the current text, or ``None`` when the field is empty."""
        return self.text() or None

    def set_value(self, v: Any) -> None:
        """Set the field text; ``None`` clears the field."""
        self.setText("" if v is None else str(v))


class _CsvColumnField(QWidget):
    """A column-name box with a CSV button that offers the columns that exist.

    The box is the setting; the button answers the question the box asks. A
    misnamed `dependent_variable` used to survive every early check and die
    inside the merge -- after the whole score table had been read -- with a
    message naming a column the file does not have and saying nothing about
    what it does have.

    THREE RULES, and each is a failure this is built not to repeat:

    * THE HEADER ROW ONLY. Every read goes through :mod:`spacr.columns`,
      which uses ``nrows=0``. This runs on the GUI thread and a score CSV is
      hundreds of megabytes; a picker that has to load the file to populate
      itself is a picker nobody waits for.

    * NO CSV IS NOT AN EMPTY LIST. With nothing loaded the button SAYS SO --
      `columns.describe` writes the sentence -- rather than opening a chooser
      with nothing in it. An empty list of choices presented as though it
      were the answer teaches a user that the file has no columns.

    * THE CHOOSER AND THE REPORTER ARE INJECTABLE (:meth:`set_chooser`,
      :meth:`set_reporter`), so a headless test drives the whole path without
      ever entering a modal event loop.
    """

    #: Emitted when the name changes, typed or picked. Named `value_changed`
    #: because that is the first signal `_connect_setting_dependency_signals`
    #: looks for, so a rule gated on this setting re-evaluates on a pick and
    #: not only on a keystroke.
    value_changed = Signal()

    def __init__(self, key: str = "", default: Any = None,
                 paths: Any = None, what: str = "column",
                 parent: Optional[QWidget] = None):
        """
        :param key: the settings key, named in the not-found message.
        :param default: the column name to start with.
        :param paths: callable returning the CSVs to read, or a fixed
            sequence of them. A CALLABLE by default: the user picks their
            input files after the panel is built, so a list captured at
            construction is always the empty one.
        :param what: what kind of column, for the message.
        """
        super().__init__(parent)
        self._key = str(key or "")
        self._what = str(what or "column")
        self._paths = paths
        self._chooser: Optional[Callable[[List[str], Any], Any]] = None
        self._reporter: Optional[Callable[[str], None]] = None

        self.edit = _ScalarEdit()
        self.edit.set_value(default)
        self.edit.textChanged.connect(self._on_edited)
        self.button = QPushButton("CSV", self)
        self.button.setObjectName("CsvColumnPicker")
        self.button.setCursor(Qt.PointingHandCursor)
        self.button.setToolTip(
            "Read the header row of the input CSVs and choose a column.")
        self.button.clicked.connect(self.pick)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(self.edit, 1)
        row.addWidget(self.button, 0)
        # Typing goes to the box, not to the button, when the row is tabbed
        # into or given focus programmatically.
        self.setFocusProxy(self.edit)

    # -- the settings-widget contract ---------------------------------------

    def get_value(self) -> Optional[str]:
        """The column name currently typed or picked, or None if empty."""
        return self.edit.get_value()

    def set_value(self, value: Any) -> None:
        """Write a column name into the box; ``None`` clears it."""
        self.edit.set_value(value)

    def text(self) -> str:
        """The raw text -- the QLineEdit contract callers may still use."""
        return self.edit.text()

    def setText(self, value: str) -> None:  # noqa: N802 - QLineEdit contract
        """Set the raw text -- the QLineEdit contract callers may still use."""
        self.edit.setText(value)

    # -- the picker ---------------------------------------------------------

    def set_chooser(self, chooser) -> None:
        """Replace the modal chooser with ``chooser(choices, current)``."""
        self._chooser = chooser

    def set_reporter(self, reporter) -> None:
        """Replace the modal message box with ``reporter(message)``."""
        self._reporter = reporter

    def input_paths(self) -> List[str]:
        """The CSVs this field's columns are read from, right now."""
        paths = self._paths() if callable(self._paths) else self._paths
        return [path for path in (paths or []) if path]

    def pick(self) -> Optional[str]:
        """Offer the columns the input CSVs have; return the one chosen.

        :returns: the chosen name, or None when there was nothing to offer or
            the user cancelled.
        """
        from spacr import columns as columns_module

        paths = self.input_paths()
        # ONE read, and it is the only one. `columns.describe` and
        # `columns.resolve` would each re-read the headers to build the same
        # list; the list is already here, so the near-miss below is computed
        # from it rather than by asking the files a second time.
        choices = columns_module.available(paths)
        if not choices:
            self.report(columns_module.describe(
                self.get_value(), paths, what=self._what, setting=self._key))
            return None
        current = self.get_value()
        chosen = self.choose(choices, current,
                             self._prompt(columns_module, choices, current))
        if chosen:
            self.set_value(chosen)
        return chosen or None

    def _prompt(self, columns_module, choices: List[str],
                current: Any) -> str:
        """The line above the chooser: how many, and the likely typo."""
        if current is not None and current not in choices:
            close = columns_module.suggest(current, choices)
            if close:
                return (f"No {self._what} {current!r} in the input CSVs. "
                        f"Did you mean {close[0]!r}?")
            return f"No {self._what} {current!r} in the input CSVs."
        return f"{len(choices)} column(s) in the input CSVs:"

    def choose(self, choices: List[str], current: Any,
               prompt: str = "") -> Optional[str]:
        """Ask the user which column. Overridden by :meth:`set_chooser`."""
        if self._chooser is not None:
            return self._chooser(choices, current)
        from PySide6.QtWidgets import QInputDialog

        index = choices.index(current) if current in choices else 0
        name, ok = QInputDialog.getItem(
            self, f"Choose a {self._what}",
            prompt or f"{len(choices)} column(s) available:",
            choices, index, False)
        return name if ok else None

    def report(self, message: str) -> None:
        """Say why there is nothing to choose from. See :meth:`set_reporter`."""
        if self._reporter is not None:
            self._reporter(message)
            return
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.information(self, "No columns to offer", message)

    def _on_edited(self, *_args) -> None:
        self.value_changed.emit()


# ---------------------------------------------------------------------------
# List / list-of-list editor
# ---------------------------------------------------------------------------
#
# A list setting used to be a text box holding a Python literal:
#
#     class_metadata   [['c1'], ['c2']]
#     train_channels   ['r', 'g', 'b']
#
# which is both ugly and unforgiving -- a dropped bracket is a parse
# failure with no diagnosis, and `_ListEdit.get_value` silently handed the
# unparseable text through as a plain string. Worse, `_ListEdit` was never
# reached: `gui_utils.convert_settings_dict_for_gui` stringifies every list
# default before this module sees it (`('entry', None, str(value))`), so
# `isinstance(default, list)` in `_widget_for` was always False and every
# list setting got a `_ScalarEdit`. `collect()` then returned the raw text,
# because `_coerce_to_expected_type` only ever handled bool/int/float. That
# is how `class_metadata` reached `io.generate_training_dataset` as the
# *string* "[['c1'], ['c2']]" and got iterated character by character.
#
# The widgets below replace the literal with removable chips -- one chip per
# value, one row per inner list -- and hand `collect()` a real Python list.
# The stored value is unchanged, so every settings CSV on disk still loads
# and every consumer reads what it always did.

#: Keys whose value may be a list of lists even when it is currently flat.
#: Taken from the same list ``spacr.settings.check_settings`` parses with
#: ``ast.literal_eval`` for the Tk GUI, so the two front ends agree on which
#: fields can hold groups.
NESTED_CAPABLE_KEYS = frozenset({
    "cell_plate_metadata", "class_metadata", "crop_mode", "dialate_png_ratios",
    "pathogen_plate_metadata", "png_dims", "png_size", "timelapse_frame_limits",
    "timelapse_objects", "treatment_plate_metadata",
    # declared ``(list, list)`` in expected_types, the in-tree marker for
    # "this can be a list of lists"
    "cell_loc", "pathogen_loc", "treatment_loc", "barcode_coordinates",
})

# Channel selections that contain more than one channel use the same
# add/remove-chip editor as ``manders_thresholds``.  The legacy GUI converter
# still labels the first three as curated combos, so keep this declaration
# close to the list editor and let the real per-module default decide whether
# the setting is actually a list.  Scalar selectors such as ``cell_channel``
# and ``channel_of_interest`` are intentionally absent.
CHANNEL_LIST_KEYS = frozenset({
    "channels", "channel_dims", "train_channels", "normalize_channels",
    "overlay_chans",
    # png_dims is deliberately absent: it is superseded by
    # png_channel_mapping, which has its own three-field R/G/B editor
    # (widgets/channel_mapping.py). Leaving it here would have offered a
    # chip list for a setting nothing renders.
})


class _RegressionBackendField(QWidget):
    """Backend selector with availability and compatibility guidance.

    Every registered backend remains visible. Unavailable or incompatible
    entries are disabled and show the reason in the menu, tooltip, and
    description pane. Changing the regression family refreshes availability
    without silently replacing the selected backend.

    Descriptions come from
    :func:`spacr.regression_backends.describe_backends` in compact form.
    """

    #: Emitted when the chosen backend changes. Named `value_changed` because
    #: that is the first signal `_connect_setting_dependency_signals` looks
    #: for, so a rule gated on this setting re-evaluates on a pick.
    value_changed = Signal()

    #: How tall the description may get before it scrolls, in pixels.
    #: The ceiling keeps the settings panel the same length for every backend;
    #: longer descriptions scroll inside the box.
    BOX_HEIGHT = 168

    def __init__(self, default: Any = None, regression_type: Any = None,
                 parent: Optional[QWidget] = None):
        """
        :param default: the stored value -- a label, a short name or None.
        :param regression_type: what the panel currently asks to fit, used to
            decide which entries are choosable. ``'auto'``/``None`` mean the
            family is chosen from the response after the data is read.
        """
        super().__init__(parent)
        from spacr.regression_backends import backend_choices

        self._regression_type = self._normalise_type(regression_type)

        self.combo = QComboBox(self)
        self.combo.setObjectName("RegressionBackendCombo")
        self.combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(12)
        for label in backend_choices():
            # The LABEL is the stored value (spacr.settings.
            # _resolve_regression_backend says why), and it is kept in
            # userData rather than read back off the text: the text carries
            # the refusal for a disabled entry and is therefore not the value.
            self.combo.addItem(label, userData=label)

        self.description = QTextBrowser(self)
        self.description.setObjectName("RegressionBackendBox")
        self.description.setReadOnly(True)
        # CLICKABLE, which is the ask -- "linkt the the API for each". A
        # QTextBrowser without this swallows the click and tries to navigate
        # itself to a URL it cannot render.
        self.description.setOpenExternalLinks(True)
        self.description.setMaximumHeight(self.BOX_HEIGHT)
        # Seven lines at the pane's default width, measured on the real
        # screen: enough that the selected backend's paragraph and the first
        # of the other seven are both on screen before anyone scrolls.
        self.description.setMinimumHeight(132)
        self.description.setSizePolicy(QSizePolicy.Preferred,
                                       QSizePolicy.Preferred)

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addWidget(self.combo, 0)
        column.addWidget(self.description, 1)
        self.setFocusProxy(self.combo)

        self.set_value(default)
        self.combo.currentIndexChanged.connect(self._on_choice_changed)
        self._install_availability_hooks()
        self.refresh()

    # -- the settings-widget contract ---------------------------------------

    def get_value(self) -> Optional[str]:
        """The chosen backend, as the label the settings CSV stores."""
        index = self.combo.currentIndex()
        if index < 0:
            return None
        return self.combo.itemData(index)

    def set_value(self, value: Any) -> None:
        """Select whatever ``value`` names -- label, short name or alias.

        An unknown name is LEFT ALONE rather than raising or silently
        selecting the default: this is called while a settings CSV is being
        loaded, and a typo there is answered by
        :func:`spacr.regression_backends.resolve_backend_name` at run time
        with a message naming every valid choice.
        """
        from spacr.regression_backends import backend_label

        try:
            label = backend_label(value)
        except (ValueError, KeyError):
            return
        index = self.combo.findData(label)
        if index >= 0:
            self.combo.setCurrentIndex(index)

    def text(self) -> str:
        """The chosen label -- the QComboBox contract callers may still use."""
        return str(self.get_value() or "")

    def setText(self, value: str) -> None:  # noqa: N802 - Qt contract
        """Select by label -- the QComboBox contract callers may still use."""
        self.set_value(value)

    # -- what is choosable, and what the box says ---------------------------

    @staticmethod
    def _normalise_type(value: Any) -> Optional[str]:
        """`'auto'`, `''` and `None` all mean "chosen from the response".

        The regression-type combo offers ``'auto'`` as the readable spelling
        of the historical ``None``, and
        :func:`spacr.settings.get_perform_regression_default_settings`
        normalises it back. `backend_status` is asked the same question in
        the same spelling, so the panel and the run agree about which
        backends can promise to fit a family nobody has chosen yet.
        """
        text = str(value if value is not None else "").strip().lower()
        return None if text in ("", "auto", "none") else text

    def regression_type(self) -> Optional[str]:
        """The family the entries are currently judged against."""
        return self._regression_type

    def set_regression_type(self, value: Any) -> None:
        """Re-judge every entry against a new ``regression_type``."""
        normalised = self._normalise_type(value)
        if normalised == self._regression_type:
            return
        self._regression_type = normalised
        self.refresh()

    def refresh(self) -> None:
        """Re-grey the entries and re-render the box."""
        from spacr.regression_backends import (backend_menu,
                                               describe_backends)

        statuses = backend_menu(self._regression_type)
        model = self.combo.model()
        blocked = self.combo.blockSignals(True)
        try:
            for index, status in enumerate(statuses):
                if index >= self.combo.count():
                    break
                label = str(status['label'])
                # THE REFUSAL IS IN THE ENTRY'S OWN TEXT. A disabled row in a
                # dropdown is grey and silent; Qt's item tooltip is shown only
                # while the popup is open and only under the cursor, so on its
                # own it is a reason a user can walk straight past.
                self.combo.setItemText(
                    index,
                    label if status['enabled']
                    else f"{label}  --  {status['short_reason']}")
                self.combo.setItemData(index, label)
                self.combo.setItemData(index, status['reason'] or
                                       f"{label}: {status['summary']}",
                                       Qt.ToolTipRole)
                if status['enabled']:
                    item = (model.item(index) if hasattr(model, "item")
                            else None)
                    if item is not None:
                        item.setEnabled(True)
                        item.setFlags(item.flags() | Qt.ItemIsSelectable)
                else:
                    # NOT `setEnabled(False)` ALONE. Measured 2026-08-18: it
                    # leaves `ItemIsSelectable` set, so Qt refuses to activate
                    # the row from the popup but a model-level selection can
                    # still land on it. `disable_combo_row` clears the flag
                    # too, and keeps the tooltip -- which is what the hover
                    # panel is hung off.
                    disable_combo_row(self.combo, index,
                                      tooltip=str(status['reason'] or ''))
        finally:
            self.combo.blockSignals(blocked)

        current = self.get_value()
        html = describe_backends(self._regression_type, html=True,
                                 selected=current, compact=True)
        chosen = next((status for status in statuses
                       if status['label'] == current), None)
        if chosen is not None and not chosen['enabled']:
            # THE SELECTION IS KEPT AND THE REFUSAL IS SHOWN. Re-pointing the
            # setting at statsmodels here would be exactly the silent
            # fallback instruction 141 C forbids -- and the sentence below is
            # the one `spacr.ml._require_backend` will use if the run starts
            # anyway, so the panel and the run say the same thing.
            html = ("<p><b>This run will be refused.</b><br>"
                    + escape(str(chosen['reason'])) + "</p>") + html
        self.description.setHtml(html)

    def api_links(self) -> List[str]:
        """Return rendered anchor URLs in document order.

        URLs are read from the laid-out document so the result contains only
        anchors that Qt parsed as clickable links.
        """
        from PySide6.QtGui import QTextCursor

        found: List[str] = []
        cursor = QTextCursor(self.description.document())
        while not cursor.atEnd():
            cursor.movePosition(QTextCursor.NextCharacter,
                                QTextCursor.KeepAnchor)
            href = cursor.charFormat().anchorHref()
            if href and href not in found:
                found.append(href)
            cursor.clearSelection()
        return found

    def _on_choice_changed(self, *_args) -> None:
        """A new backend: re-render the box for it, then tell the panel."""
        self.refresh()
        self.value_changed.emit()

    # -- the unavailable entries explain themselves (instruction 158) -------
    #
    # THE ROW STAYS DEAD and everything interactive lives in the hover panel.
    # Three routes reach it and they are all here rather than in the panel,
    # because the panel is shared with the Image UMAP and must not know what a
    # regression backend is:
    #
    #   * hovering a greyed row in the OPEN popup, anchored on that row;
    #   * hovering the CLOSED combo while the value it holds has gone
    #     unavailable -- 141 C keeps a stale selection rather than silently
    #     re-pointing it, so this is a state a user can sit in;
    #   * Shift+F1 on the combo, which is the keyboard route. It has to be
    #     explicit: the rows are disabled, so nothing about them is tabbable
    #     and no help can be inherited from them.
    #
    # THE POPUP IS CLOSED THE MOMENT THE POINTER LEAVES IT. A QComboBox popup
    # is a `Qt.Popup` with an active mouse grab, so with it still open the
    # first click on the panel would be eaten by the grab -- the Install link
    # would need two presses and the first would look like it did nothing.

    def availability_entries(self) -> List[dict]:
        """Every backend as the shared panel wants it, in panel order."""
        from spacr.regression_backends import availability_entries
        return availability_entries(self._regression_type)

    def unavailable_entries(self) -> List[dict]:
        """Just the greyed ones -- what the panel cycles through."""
        return [entry for entry in self.availability_entries()
                if not entry['enabled']]

    def _install_availability_hooks(self) -> None:
        """Watch the combo and its popup for the three routes above."""
        self.combo.installEventFilter(self)
        view = self.combo.view()
        if view is not None:
            view.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt contract
        """Route hover and Shift+F1 to the shared availability panel."""
        combo = getattr(self, "combo", None)
        if combo is None:
            return super().eventFilter(obj, event)
        try:
            view = combo.view()
        except RuntimeError:                             # pragma: no cover
            return super().eventFilter(obj, event)
        viewport = view.viewport() if view is not None else None
        kind = event.type()
        if obj is viewport:
            if kind == QEvent.MouseMove:
                self._hover_popup_row(view, event)
            elif kind == QEvent.Leave:
                # Leaving the popup is how the pointer travels to the panel.
                self._release_popup()
        elif obj is combo:
            if kind == QEvent.KeyPress and self._is_help_key(event):
                self.open_availability_panel()
                return True
            if kind == QEvent.Enter:
                self._hover_closed_combo()
            elif kind == QEvent.Leave:
                panel = AvailabilityPanel.instance()
                if panel.isVisible():
                    panel.start_hide()
        return super().eventFilter(obj, event)

    @staticmethod
    def _is_help_key(event) -> bool:
        """Shift+F1 -- Qt's own "explain this control" chord."""
        return (event.key() == Qt.Key_F1
                and bool(event.modifiers() & Qt.ShiftModifier))

    def _hover_popup_row(self, view, event) -> None:
        """A greyed row under the pointer opens the panel beside it."""
        try:
            position = event.position().toPoint()
        except AttributeError:                           # pragma: no cover
            position = event.pos()
        index = view.indexAt(position)
        if not index.isValid():
            return
        statuses = self.availability_entries()
        if index.row() >= len(statuses):
            return
        entry = statuses[index.row()]
        if entry['enabled']:
            panel = AvailabilityPanel.instance()
            if panel.isVisible():
                panel.start_hide()
            return
        rect = view.visualRect(index)
        top_left = view.viewport().mapToGlobal(rect.topLeft())
        self.show_availability_panel(
            entry['key'], anchor=view.viewport(),
            anchor_rect=QRect(top_left, rect.size()))

    def _hover_closed_combo(self) -> None:
        """Hovering the combo explains a selection that has gone stale."""
        current = self.get_value()
        entry = next((e for e in self.availability_entries()
                      if e['title'] == current), None)
        if entry is None or entry['enabled']:
            return
        self.show_availability_panel(entry['key'], anchor=self.combo)

    def _release_popup(self) -> None:
        """Close the dropdown so its mouse grab stops owning the pointer."""
        panel = AvailabilityPanel.instance()
        if panel.isVisible():
            self.combo.hidePopup()

    def show_availability_panel(self, key, *, anchor=None,
                                anchor_rect=None, pinned: bool = False):
        """Open the shared panel on the unavailable backend named ``key``.

        :param key: a backend name. Ignored when it is not unavailable.
        :returns: the panel, or ``None`` when there was nothing to explain.
        """
        entries = self.unavailable_entries()
        if not entries:
            return None
        index = next((i for i, entry in enumerate(entries)
                      if entry['key'] == key), 0)
        panel = AvailabilityPanel.instance()
        self._connect_panel(panel)
        if pinned:
            panel.open_for(anchor or self.combo, entries, index,
                           anchor_rect=anchor_rect)
        else:
            panel.show_for(anchor or self.combo, entries, index,
                           anchor_rect=anchor_rect)
        return panel

    def open_availability_panel(self):
        """The keyboard route: Shift+F1 pins the panel and focuses it."""
        current = self.get_value()
        entries = self.unavailable_entries()
        if not entries:
            return None
        key = next((e['key'] for e in entries if e['title'] == current),
                   entries[0]['key'])
        return self.show_availability_panel(key, anchor=self.combo,
                                            pinned=True)

    def _connect_panel(self, panel) -> None:
        """Take ownership of the shared panel's Install signal.

        The panel is a process-wide singleton with two callers, so the
        connection is remade on every show rather than once in ``__init__`` --
        otherwise the Image UMAP's copy and this one would both answer.
        """
        panel.set_install_handler(self._run_install_offer)

    def _run_install_offer(self, offer) -> None:
        """Press Install: the dry run first, then the install, or neither."""
        outcome = run_install_offer(self, offer)
        if outcome == "installed":
            self.refresh()


# The chip strip's wrapping row now lives beside the other widgets, because
# the regression results header needs the same thing. The private names stay
# so nothing that imported them from here has to move.
from ..widgets.flow import FlowHost as _FlowHost, FlowLayout as _FlowLayout


class _Chip(QFrame):
    """One value, rendered as a removable pill."""

    removed = Signal(object)

    def __init__(self, text: str, colours: dict, parent=None):
        super().__init__(parent)
        from ..i18n import tr
        from ..theme import apply_close_mark, font_px
        self.setObjectName("SettingChip")
        self._text = text
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 1, 3, 1)
        row.setSpacing(4)
        label = QLabel(text, self)
        label.setObjectName("SettingChipText")
        row.addWidget(label)
        close = QToolButton(self)
        close.setObjectName("SettingChipClose")
        # THE APPLICATION'S CLOSE MARK -- see `theme.apply_close_mark`.
        #
        # THE VALUE IS A VALUE. Splicing it in first asks the catalog for
        # "Remove Cell", "Remove cytoplasm" and one key per chip anyone ever
        # types; the caption is looked up as a template and the value put in
        # after, so the verb translates whatever the chip holds.
        apply_close_mark(close, tooltip=tr("Remove {value}", value=text))
        close.setFocusPolicy(Qt.NoFocus)
        close.clicked.connect(lambda: self.removed.emit(self))
        row.addWidget(close)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet(
            f"""
            QFrame#SettingChip {{
                background: {colours['accent_soft']};
                border: 1px solid {colours['border']};
                border-radius: 9px;
            }}
            QLabel#SettingChipText {{
                color: {colours['fg']};
                background: transparent;
                font-size: {font_px(12)}px;
            }}
            """
        )

    def text(self) -> str:
        """The value this chip carries, as typed."""
        return self._text


class _ChipStrip(QWidget):
    """A wrapping strip of chips plus the field that adds another one."""

    changed = Signal()
    emptied = Signal(object)

    def __init__(self, placeholder: str = "add value…",
                 removable: bool = False, parent=None):
        super().__init__(parent)
        from ..theme import active_palette, apply_close_mark, font_px
        self._colours = active_palette()
        self._chips: List[_Chip] = []

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._host = _FlowHost(self)
        self._flow = _FlowLayout(self._host, spacing=4)
        outer.addWidget(self._host, 1)

        self._entry = QLineEdit(self)
        self._entry.setObjectName("SettingChipEntry")
        self._entry.setPlaceholderText(placeholder)
        self._entry.setMinimumWidth(96)
        self._entry.returnPressed.connect(self._commit_entry)
        self._entry.editingFinished.connect(self._commit_entry)
        self._entry.textEdited.connect(self._on_typed)
        self._flow.addWidget(self._entry)

        self._drop = None
        if removable:
            self._drop = QToolButton(self)
            # THE APPLICATION'S CLOSE MARK -- see `theme.apply_close_mark`.
            apply_close_mark(self._drop, tooltip="Remove this group")
            self._drop.setFocusPolicy(Qt.NoFocus)
            self._drop.clicked.connect(lambda: self.emptied.emit(self))
            outer.addWidget(self._drop, 0, Qt.AlignTop)

    # -- value -----------------------------------------------------------
    def values(self) -> List[str]:
        """The chip texts, in order, plus anything still uncommitted."""
        out = [chip.text() for chip in self._chips]
        pending = self._entry.text().strip()
        if pending:
            out.append(pending)
        return out

    def set_values(self, values) -> None:
        """Replace every chip with ``values``."""
        for chip in list(self._chips):
            self._remove_chip(chip, notify=False)
        self._entry.clear()
        for value in values or []:
            self._add_chip(str(value), notify=False)
        self.changed.emit()

    # -- internals -------------------------------------------------------
    def _on_typed(self, text: str) -> None:
        """Commit on a comma so a pasted 'c1,c2,c3' becomes three chips."""
        if "," not in text:
            return
        head, _, tail = text.partition(",")
        self._entry.setText(tail.lstrip())
        head = head.strip()
        if head:
            self._add_chip(head)

    def _commit_entry(self) -> None:
        text = self._entry.text().strip()
        if not text:
            return
        self._entry.clear()
        self._add_chip(text)

    def _add_chip(self, text: str, notify: bool = True) -> None:
        chip = _Chip(text, self._colours, self._host)
        chip.removed.connect(self._remove_chip)
        # Keep the entry field last so it always trails the chips.
        self._flow.removeWidget(self._entry)
        self._flow.addWidget(chip)
        self._flow.addWidget(self._entry)
        self._chips.append(chip)
        self._host.updateGeometry()
        self.updateGeometry()
        if notify:
            self.changed.emit()

    def _remove_chip(self, chip, notify: bool = True) -> None:
        if chip in self._chips:
            self._chips.remove(chip)
        self._flow.removeWidget(chip)
        chip.setParent(None)
        chip.deleteLater()
        self._host.updateGeometry()
        self.updateGeometry()
        if notify:
            self.changed.emit()


#: Column-name settings that hold any number of names, rendered as a chip
#: strip rather than a text box.
#:
#: They are declared ``(str, None)`` in :mod:`spacr.settings`, which is what
#: sent them to a single-value field: one column per run, and a SQL button
#: that replaced whatever was already typed. The declaration stays as it is
#: -- every consumer accepts a bare string and always has -- so old settings
#: CSVs keep loading and the CLI keeps working; only the control widens.
EXCLUDE_LIST_KEYS: Tuple[str, ...] = ("exclude",)


#: Settings that name one or more input FILES, mapped to the kind of file each
#: one wants. They get :class:`FilePathListWidget`: a real file dialog that can
#: be pressed repeatedly to gather sources from several folders, plus
#: drag-and-drop.
#:
#: These previously rendered as the free-text chip strip, which meant a
#: four-plate screen was configured by typing four absolute paths by hand --
#: and ``score_data``/``count_data`` shipped the literal default string
#: ``'list of paths'``, so the first thing every user had to do was delete a
#: placeholder that looked like a value. A mistyped path was not detected
#: until the run had already read the other CSVs and died.
#:
#: The value stays a plain ``list[str]``, so settings CSVs written by the Tk
#: panel or by hand still load, and the CLI is unaffected.
PATH_LIST_KEYS: Dict[str, str] = {
    "score_data": "table",
    "count_data": "table",
    "metadata_files": "table",
    "grna_csv": "csv",
    "row_csv": "csv",
    "column_csv": "csv",
    "barcodes": "csv",
    "grna": "csv",
}


#: Human-readable dialog titles, so the file chooser says what it is for
#: instead of "Choose input files" four times in one panel.
PATH_LIST_TITLES: Dict[str, str] = {
    "score_data": "Choose per-object score CSVs",
    "count_data": "Choose gRNA count CSVs (one per plate)",
    "metadata_files": "Choose metadata CSVs",
    "grna_csv": "Choose the gRNA barcode CSV",
    "row_csv": "Choose the row barcode CSV",
    "column_csv": "Choose the column barcode CSV",
    "barcodes": "Choose the barcode CSV",
    "grna": "Choose the gRNA CSV",
}


#: The subset of :data:`PATH_LIST_KEYS` that names exactly ONE file.
#:
#: Every one of these is declared ``str`` in :mod:`spacr.settings` and is
#: handed to ``pd.read_csv`` unchanged -- ``sequencing.map_sequences_to_names``
#: for the three barcode references, the legacy helpers for the other two.
#: Giving them the multi-file control made the panel COLLECT a one-element
#: list, so merely opening the module and saving rewrote
#: ``column_csv=/…/barcodes_column.csv`` to ``['/…/barcodes_column.csv']`` in
#: the user's settings file -- and `validate` then refused every run from it
#: with "column_csv=[...] is a list, but str is expected", about a value the
#: user had never typed. The dialog and the drop target stay; the shape of the
#: value goes back to what its consumer reads.
PATH_LIST_SINGLE_KEYS: Tuple[str, ...] = (
    "grna_csv", "row_csv", "column_csv", "barcodes", "grna",
)


#: Settings whose legal values are a short, closed, ordered set.
#:
#: ``train_channels`` is the reason this table exists. It is declared a plain
#: ``list``, so it rendered as a free-text chip strip that accepted ``x``,
#: ``red``, ``4`` and ``rgb`` without complaint — and
#: :func:`spacr.io._resolve_channel_indices` maps letters to planes with
#: three ``if 'r' in channels`` tests, so an off-alphabet value is dropped
#: silently and the model trains on fewer planes than the user asked for.
#: :func:`spacr.deep_spacr.train_test_model` then joins the same list into a
#: directory name, so the typo reaches the filesystem too.
#:
#: Order is part of the alphabet, not part of the user's input: ``['b','r']``
#: and ``['r','b']`` select the same two planes but write two different model
#: directories. A control that can only emit canonical order removes that
#: whole class of confusion, which a text field cannot.
FIXED_ALPHABETS: Dict[str, Tuple[Tuple[Any, str], ...]] = {
    "train_channels": (("r", "Red"), ("g", "Green"), ("b", "Blue")),
    # WHAT THE MODEL IS ALLOWED TO LOOK AT (236 A2), asked for as "the user
    # can train on channel_1 measurements only or morphological
    # measurements or channel combinations, localization ... This should be
    # straight forward and easy."
    #
    # `utils.filter_dataframe_features` has always taken a list,
    # 'morphology' and a free-text fragment. The setting declared `int`, so
    # a spin box was all the panel could draw and three of the four
    # documented ways of choosing a feature space were unreachable. A
    # multi-select says the whole question in one row: light one chip for
    # one channel, two for the combination, Shape for morphology, none for
    # every feature.
    #
    # LOCALISATION NEEDS NO CHIP. A colocalisation column names the two
    # channels it measures and survives a request for either, so asking for
    # channel 1 already brings channel 1's relationships with it.
    "channel_of_interest": ((0, "Ch 0"), (1, "Ch 1"), (2, "Ch 2"),
                            (3, "Ch 3"), ("morphology", "Shape")),
}


def _alphabet_qss(palette: dict, opacity) -> str:
    """QSS for the fixed-alphabet toggles, registered through the theme seam.

    Selected and unselected have to differ at a glance without colour alone
    carrying the meaning — the text is the value either way, and the border
    does the work, so the control still reads on a monochrome display and for
    a red-green colour-blind reader choosing red and green channels.
    """
    from ..theme import block_surface
    surface = block_surface("surface_alt", palette["theme"], opacity)
    return f"""
QToolButton#SettingAlphabetChip {{
    background: {surface};
    color: {palette["fg_dim"]};
    border: 1px solid {palette["border_soft"]};
    border-radius: 10px;
    padding: 2px 12px;
}}
QToolButton#SettingAlphabetChip:hover {{
    border-color: {palette["accent"]};
}}
QToolButton#SettingAlphabetChip:checked {{
    color: {palette["fg"]};
    border: 1px solid {palette["accent"]};
    font-weight: 600;
}}
"""


try:  # pragma: no cover - the theme seam is present in every real launch
    from ..theme import register_widget_qss as _register_widget_qss
    _register_widget_qss("SettingAlphabetChip", _alphabet_qss, replace=True)
except Exception:  # pragma: no cover
    LOGGER.debug("Could not register the alphabet-chip QSS", exc_info=True)


class _AlphabetSelect(QWidget):
    """Multi-select over a fixed, ordered alphabet of values.

    One checkable pill per legal value, always shown, always in the
    alphabet's own order. Nothing else can be entered and nothing can be
    entered twice, so the two failure modes of the free-text strip it
    replaces — an unrecognised letter that is silently dropped downstream,
    and a permutation that changes the output path without changing the
    result — are both unrepresentable.

    ``get_value`` / ``set_value`` mirror :class:`_ListEditor`'s contract so
    the settings-CSV import path, the Live Preview propagation path and
    :meth:`SettingsWidgets.collect` need no special case beyond the class.
    """

    changed = Signal()

    def __init__(self, key: str = "", default: Any = None,
                 choices: Tuple[Tuple[Any, str], ...] = (), parent=None):
        super().__init__(parent)
        self._key = key
        self._choices = tuple(choices)
        self._buttons: List[Tuple[Any, QToolButton]] = []

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for value, label in self._choices:
            button = QToolButton(self)
            button.setObjectName("SettingAlphabetChip")
            button.setText(str(label))
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setFocusPolicy(Qt.StrongFocus)
            # The accessible name is the value, not the label: a screen
            # reader user is choosing 'r', and "Red" is only the gloss.
            button.setAccessibleName(str(value))
            button.setProperty("alphabetValue", value)
            button.toggled.connect(self._on_toggled)
            row.addWidget(button)
            self._buttons.append((value, button))
        row.addStretch(1)

        self.set_value(default)

    # -- public contract -------------------------------------------------
    def get_value(self) -> List[Any]:
        """The checked values, always in alphabet order."""
        return [value for value, button in self._buttons if button.isChecked()]

    def set_value(self, value: Any) -> None:
        """Check exactly the members of ``value``; ignore anything else.

        Strings are parsed as Python literals first, because settings CSVs
        and the Live Preview both hand back ``"['r', 'g']"`` rather than a
        list. A value outside the alphabet is dropped rather than shown,
        which is the whole point of the control — but it is dropped
        *visibly*, because the corresponding pill is not lit.
        """
        wanted = self._as_members(value)
        for member, button in self._buttons:
            blocked = button.blockSignals(True)
            button.setChecked(member in wanted)
            button.blockSignals(blocked)
        self.changed.emit()

    def text(self) -> str:
        """Line-edit-compatible rendering, for callers that expect one."""
        return repr(self.get_value())

    def setText(self, value: str) -> None:  # noqa: N802 - QLineEdit contract
        """Accept a textual value, for callers that expect a QLineEdit."""
        self.set_value(value)

    def choices(self) -> Tuple[Any, ...]:
        """The legal values, in order. Public so tests need no internals."""
        return tuple(value for value, _label in self._choices)

    # -- internals -------------------------------------------------------
    def _on_toggled(self, _checked: bool) -> None:
        self.changed.emit()

    @staticmethod
    def _as_members(value: Any) -> set:
        if value is None:
            return set()
        if isinstance(value, str):
            text = value.strip()
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                # A bare "r,g" or "r g" from a hand-edited CSV.
                parsed = [part for part in text.replace(",", " ").split()
                          if part]
            value = parsed
        if isinstance(value, (list, tuple, set, frozenset)):
            return set(value)
        return {value}


class _ListEditor(QWidget):
    """The widget behind every list-valued setting.

    Flat lists are one strip of chips. Lists of lists are one strip per
    inner list, stacked, each with its own remove button and a footer that
    adds another group. A key that *can* hold groups but currently does not
    gets a "Use groups" button instead, so nothing that was editable as a
    literal becomes uneditable here.

    ``get_value`` / ``set_value`` mirror ``_ListEdit``'s contract, so the
    Live Preview propagation path and the settings-CSV import path need no
    special case beyond knowing the class.
    """

    def __init__(self, key: str = "", default: Any = None,
                 nested_capable: bool = False, allow_none: bool = False,
                 element_type: Any = None, container: Any = list, parent=None):
        super().__init__(parent)
        # font_px is used further down this method. Importing only
        # active_palette here raised NameError out of build_sections(), and
        # AppScreen turns that into "Failed to build settings for '<app>'" --
        # so sixteen shipped modules, mask and measure and classify among
        # them, opened with no settings form at all.
        from ..theme import active_palette, font_px
        self._colours = active_palette()
        self._key = key
        self._nested_capable = bool(nested_capable)
        self._allow_none = bool(allow_none)
        self._element_type = element_type
        self._container = container if container in (list, tuple) else list
        self._nested = False
        self._strips: List[_ChipStrip] = []

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(4)

        self._rows = QVBoxLayout()
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(4)
        self._outer.addLayout(self._rows)

        self._footer = QToolButton(self)
        self._footer.setObjectName("SettingListFooter")
        self._footer.setCursor(Qt.PointingHandCursor)
        self._footer.setFocusPolicy(Qt.NoFocus)
        self._footer.clicked.connect(self._on_footer)
        self._footer.setStyleSheet(
            f"QToolButton#SettingListFooter {{ color: {self._colours['accent']};"
            f" background: transparent; border: none; font-size: {font_px(12)}px;"
            f" padding: 0px; text-align: left; }}"
        )
        self._outer.addWidget(self._footer, 0, Qt.AlignLeft)

        self.set_value(default)

    # -- public contract -------------------------------------------------
    def get_value(self) -> Any:
        """Return a real ``list`` (or list of lists); ``None`` when empty
        and the setting declares ``None`` as legal."""
        make = self._container
        if self._nested:
            groups = [make(self._cast(v) for v in strip.values())
                      for strip in self._strips]
            groups = [g for g in groups if g]
            if not groups:
                return None if self._allow_none else make()
            return make(groups)
        values = [self._cast(v) for v in self._strips[0].values()] \
            if self._strips else []
        if not values:
            return None if self._allow_none else make()
        return make(values)

    def set_value(self, value: Any) -> None:
        """Render ``value``; strings are parsed as Python literals first.

        Settings CSVs and the Live Preview both hand back text, so a
        ``"[['c1'], ['c2']]"`` has to land as two groups rather than as
        seventeen chips full of punctuation.
        """
        value = self._as_sequence(value)
        nested = bool(value) and all(
            isinstance(item, (list, tuple)) for item in value)
        self._rebuild(nested, value)

    def text(self) -> str:
        """Return a line-edit-compatible textual representation.

        A single path is returned without list punctuation for compatibility
        with callers that treated ``src`` as a ``QLineEdit`` before it became
        a multi-plate setting. Multiple values use their unambiguous Python
        representation.
        """
        value = self.get_value()
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return str(value[0])
        return "" if value is None else str(value)

    def setText(self, value: str) -> None:  # noqa: N802 - QLineEdit contract
        """Accept the legacy ``QLineEdit.setText`` API."""
        self.set_value(value)

    # -- shape -----------------------------------------------------------
    def _rebuild(self, nested: bool, value) -> None:
        for strip in list(self._strips):
            # editingFinished fires while a focused QLineEdit is being torn
            # down, which would call _commit_entry on a half-deleted strip.
            strip._entry.blockSignals(True)
            self._rows.removeWidget(strip)
            strip.setParent(None)
            strip.deleteLater()
        self._strips = []
        self._nested = bool(nested)
        if nested:
            for group in value:
                self._add_strip(list(group))
            if not self._strips:
                self._add_strip([])
        else:
            self._add_strip(list(value))
        self._refresh_footer()

    def _add_strip(self, values) -> _ChipStrip:
        strip = _ChipStrip(placeholder=self._placeholder(),
                           removable=self._nested, parent=self)
        strip.emptied.connect(self._drop_strip)
        self._rows.addWidget(strip)
        self._strips.append(strip)
        strip.set_values(values)
        return strip

    def _drop_strip(self, strip) -> None:
        if len(self._strips) <= 1:
            # Removing the only group is how you go back to a flat list.
            self._rebuild(False, [])
            return
        self._strips.remove(strip)
        strip._entry.blockSignals(True)
        self._rows.removeWidget(strip)
        strip.setParent(None)
        strip.deleteLater()
        self._refresh_footer()

    def _on_footer(self) -> None:
        if self._nested:
            self._add_strip([])
            return
        # Flat -> grouped: the values already typed become the first group.
        current = list(self._strips[0].values()) if self._strips else []
        self._rebuild(True, [current] if current else [[]])

    def _refresh_footer(self) -> None:
        if self._nested:
            self._footer.setText("＋  Add group")
            self._footer.setToolTip(
                "Add another group. Each group is one inner list — one "
                "class, one condition, one crop mode.")
            self._footer.setVisible(True)
        elif self._nested_capable:
            self._footer.setText("⌗  Use groups")
            self._footer.setToolTip(
                "This setting also accepts a list of lists. Grouping turns "
                "the values above into the first group.")
            self._footer.setVisible(True)
        else:
            self._footer.setVisible(False)

    # -- element handling ------------------------------------------------
    def _placeholder(self) -> str:
        # Short enough to survive the narrow settings column without
        # eliding -- the point of the placeholder is to say what KIND of
        # value belongs here, and an elided "add a whole numb…" says less
        # than "add number".
        if self._element_type is int:
            return "add number"
        if self._element_type is float:
            return "add number"
        if self._element_type is str:
            return "add text"
        return "add value"

    def _cast(self, text: str) -> Any:
        """Turn typed text back into the element type the list holds.

        Inferred from the default value rather than guessed per keystroke,
        so ``classes = ['1', '2']`` stays strings and ``png_dims = [0, 1, 2]``
        stays ints.
        """
        text = str(text).strip()
        if self._element_type is str:
            return text
        if self._element_type in (int, float):
            try:
                return self._element_type(text)
            except (TypeError, ValueError):
                return text
        if text.lower() == "none":
            return None
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text

    @staticmethod
    def _as_sequence(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            text = value.strip()
            if not text or text == "None":
                return []
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                # Not a literal: treat it as a comma-separated list, which is
                # what a user hand-editing a settings CSV most often means.
                return [part.strip() for part in text.split(",") if part.strip()]
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            return [parsed]
        return [value]


def list_shape_for(key: str, default: Any) -> Optional[Tuple[bool, bool, Any, Any]]:
    """Decide whether ``key`` is a list setting, and of what shape.

    Deliberately conservative. A key qualifies only when its *default* is
    already a list or tuple, or is ``None`` and the declared type admits
    nothing but a list. That keeps three groups of keys on their old
    widgets:

    * ``src`` and ``file_metadata``, declared ``(str, list)`` -- they are
      normally one path / one substring, and ``src`` in particular has to
      stay a ``QLineEdit`` for drag-and-drop, the empty-state banner and
      the column picker's ``_settings_src_path``;
    * ``count_data`` / ``score_data``, declared ``list`` but shipped with
      the placeholder *string* ``'list of paths'``;
    * ``sample``, whose declared "type" is the value ``None``.

    :returns: ``(nested_capable, allow_none, element_type, container)`` when
        the key holds a list, or ``None`` when it should keep its ordinary
        widget.
    """
    declared = None
    try:
        from spacr.settings import expected_types
        declared = expected_types.get(key)
    except Exception:
        declared = None
    allowed = declared if isinstance(declared, tuple) else (declared,)
    declares_list = any(t in (list, tuple) for t in allowed)
    declares_scalar = any(t in (str, int, float, bool, dict) for t in allowed)

    if isinstance(default, (list, tuple)):
        pass
    elif default is None and declares_list and not declares_scalar:
        pass
    else:
        return None

    container = tuple if (declares_list and list not in allowed) else list
    if isinstance(default, tuple) and not declares_list:
        container = tuple
    allow_none = (type(None) in allowed) or default is None
    items = list(default) if isinstance(default, (list, tuple)) else []
    flat = []
    nested_now = bool(items) and all(isinstance(i, (list, tuple)) for i in items)
    for item in items:
        flat.extend(item if isinstance(item, (list, tuple)) else [item])
    element_type = None
    # bool first: bool is a subclass of int, and a list of flags is not a
    # list of numbers.
    if flat and all(isinstance(v, str) for v in flat):
        element_type = str
    elif flat and all(isinstance(v, bool) for v in flat):
        element_type = None
    elif flat and all(isinstance(v, int) for v in flat):
        element_type = int
    elif flat and all(isinstance(v, (int, float)) for v in flat):
        element_type = float

    nested_capable = nested_now or key in NESTED_CAPABLE_KEYS or (
        isinstance(declared, tuple) and list(declared).count(list) > 1)
    return nested_capable, allow_none, element_type, container


class SettingsWidgets:
    """Container for the Qt widgets bound to a settings dict.

    Instantiate with an `app_key`; call `.build_sections()` to get a list
    of (section_title, list_of_(label, widget)) tuples to feed into the
    Section widgets on a screen. `.collect()` returns the current settings
    dict after user edits."""

    def __init__(self, app_key: str, parent: Optional[QWidget] = None):
        """Load the app's default settings dict and prepare an empty widget map.

        :param app_key: id of the app whose settings are being edited.
        :param parent: optional Qt parent for created widgets.
        """
        self.app_key = app_key
        self._parent = parent
        # EVERY SLOT THAT CAN BE NAMED, not the four the module ships. A
        # control that was never built cannot be revealed, so a panel whose
        # defaults stop at `number_of_organelles` slots can only ever render
        # that many however the count is driven -- which is exactly why
        # raising the count to seven went on drawing the same four. The extra
        # keys arrive with the values they would have had and their rows are
        # hidden by `refresh_object_visibility`, so what the count changes is
        # which of them is ON SCREEN. `number_of_organelles` itself is left
        # at the module's own number: this widens what can be shown, not what
        # the panel opens showing.
        from spacr.settings import organelle_slots_beyond_the_count

        shipped = resolve_default_settings(app_key)
        self._defaults = organelle_slots_beyond_the_count(
            shipped, PANEL_ORGANELLE_SLOTS)
        # WHICH KEYS THE PANEL INVENTED, and what it gave them. A settings
        # file is not a panel: writing every slot that can be named into
        # every CSV would bury the four a run uses. `collect` leaves these
        # out again while they are above the count AND still hold exactly
        # what was put here -- so a value that came from anywhere else, a
        # user or a loaded file, is written out whatever the count is.
        self._slots_the_panel_added = {
            key: value for key, value in self._defaults.items()
            if key not in shipped}
        self._widgets: Dict[str, QWidget] = {}
        # What the object rule decided last, and the rows watching to see
        # that it sticks. See `_guard_hidden_rows`.
        self._hidden_by_the_run: set = set()
        self._guarded_rows: Dict[int, str] = {}
        #: ``id(section) -> section`` for the slot headings this hid, so it
        #: can put back exactly what it took and nothing else.
        self._headings_of_absent_slots: Dict[int, Any] = {}
        self._object_row_guard = _HiddenRowWatcher(self, parent)
        self._object_rule_pass_queued = False
        #: Called with the keys this pass is hiding, just before row
        #: visibility is decided, so the screen can lay out any row it left
        #: unbuilt that is about to be shown. The model decides WHETHER a row
        #: is on the form; only the screen can BUILD one. Left ``None`` on a
        #: model built for its values rather than for a screen.
        self.rows_are_laid_out_by = None
        self._tooltips = get_tooltips()
        self._data_context: Dict[str, Any] = {'plate_count': None}
        if app_key == "umap":
            # These app-scoped descriptions supersede legacy shared strings
            # without invalidating source-bound reviewed translation evidence.
            self._tooltips.update(_UMAP_TOOLTIP_OVERRIDES)
        try:
            from spacr.plugins import get_app
            plugin_app = get_app(app_key)
            if plugin_app is not None:
                self._tooltips.update(plugin_app.tooltips)
        except Exception:
            pass

    def build_sections(self) -> List["SettingsSection"]:
        """Build the section tree with the UI language resolved once.

        The scope is the whole reason this wrapper exists; see
        :func:`language_resolved_once`. Every tooltip, type hint, label and
        documentation URL below asks what language the interface is in, and
        without the scope each of those asks reads ``QSettings`` again.

        :returns: what :meth:`_build_sections` returns, unchanged.
        """
        with language_resolved_once():
            return self._build_sections()

    def _build_sections(self) -> List["SettingsSection"]:
        """Group the settings and return the panel's section TREE.

        Each entry is a :class:`SettingsSection`, which still IS the
        ``(title, rows)`` pair this returned before nesting existed -- a
        caller that unpacks or ``dict()``s the result needs no change, and
        ``rows`` holds every row in the subtree, so a panel that draws only
        the outer level still draws every control exactly once.

        Three levels are expressible: the "Advanced settings" umbrella, the
        family headings that declare it as their parent
        (``spacr.settings.CATEGORY_PARENTS``), and one sub-heading per object
        inside each family, derived from the setting keys. A category that
        declares no parent and splits into no objects is a single flat
        section exactly as before.

        Anything in no category at all lands in a trailing "Other".
        """
        # `spacr.settings_spec`, NOT `spacr.gui_utils`. The function is the
        # same one (gui_utils re-exports it); the module it now lives in
        # imports nothing. Reaching it through gui_utils cost 770 ms of Tk
        # dependencies -- IPython, matplotlib.pyplot, cv2, tkinter,
        # huggingface_hub -- on the GUI thread, and it was the whole remaining
        # cost of opening the first module. See spacr/settings_spec.py.
        from spacr.settings_spec import convert_settings_dict_for_gui
        variables = convert_settings_dict_for_gui(self._defaults)

        # Materialize a widget per key; attach a rich HTML tooltip that ends
        # with a compact information-icon link to the spaCR documentation.
        # A hidden key gets no widget at all, which is what actually hides
        # it: the trailing "Other" section below is built from
        # `self._widgets`, so a key left out of every category still
        # renders as long as a widget exists for it. The value stays in
        # `self._defaults` and reaches the run unchanged.
        hidden_keys = _APP_HIDDEN_KEYS.get(self.app_key, frozenset())
        for key, meta in variables.items():
            if key in hidden_keys:
                continue
            kind, options, default = meta
            widget = self._widget_for(kind, options, default, key)
            if widget is not None:
                attach_api_tooltip(
                    widget,
                    self.app_key,
                    key,
                    _descriptions=self._tooltips,
                )
                self._widgets[key] = widget

        src_widget = self._widgets.get("src")
        if isinstance(src_widget, QLineEdit):
            src_widget.editingFinished.connect(
                self._refresh_contextual_widgets)
        elif isinstance(src_widget, DatabaseSetWidget):
            # The same obligation through a different control: adding a plate
            # changes which columns and which rows the dependent fields can
            # offer, so the panel follows the SET as it is edited rather than
            # only when the screen is built.
            src_widget.value_changed.connect(self._refresh_contextual_widgets)

        # The training basis changes which controls matter, so the panel has
        # to follow it as it is changed rather than only when the screen is
        # built. A bound method, not a lambda: see INVARIANTS 4 for what a
        # closure connected to a Qt signal costs.
        family_widget = self._widgets.get("classifier_family")
        if family_widget is not None:
            for signal_name in ("currentTextChanged", "currentIndexChanged",
                                "textChanged"):
                signal = getattr(family_widget, signal_name, None)
                if signal is not None:
                    signal.connect(self._on_classifier_family_changed)
                    break

        basis_widget = self._widgets.get("dataset_mode")
        if basis_widget is not None:
            for signal_name in ("currentTextChanged", "currentIndexChanged",
                                "textChanged"):
                signal = getattr(basis_widget, signal_name, None)
                if signal is not None:
                    signal.connect(self._on_training_basis_changed)
                    break

        # An entry greyed for one family is choosable for another, so the
        # backend control has to follow `regression_type` rather than be
        # judged once when the panel is built. Bound method, not a lambda:
        # INVARIANTS 4.
        type_widget = self._widgets.get("regression_type")
        if (isinstance(self._widgets.get("regression_backend"),
                       _RegressionBackendField) and type_widget is not None):
            for signal_name in ("currentTextChanged", "currentIndexChanged",
                                "textChanged"):
                signal = getattr(type_widget, signal_name, None)
                if signal is not None:
                    signal.connect(self._on_regression_type_changed)
                    break

        reducer_widget = self._widgets.get("reduction_method")
        if self.app_key == "umap" and isinstance(reducer_widget, QComboBox):
            reducer_widget.currentTextChanged.connect(
                self._on_umap_reducer_changed)
        affinity_widget = self._widgets.get("spectral_affinity")
        if self.app_key == "umap" and isinstance(affinity_widget, QComboBox):
            affinity_widget.currentTextChanged.connect(
                self._on_umap_reducer_changed)

        # Every panel, not only the regression one it was built against.
        # _rules_for_this_panel decides what applies here, and a panel with
        # no gated setting connects nothing. See its docstring.
        self._connect_setting_dependency_signals()

        # THE OBJECTS THIS RUN HAS. A channel that gains a number reveals
        # its object's settings and losing it hides them again, and the type
        # a slot is given decides which of that slot's detection settings are
        # on screen at all. Bound method, not a lambda: INVARIANTS 4.
        self._connect_object_visibility_signals()

        self._refresh_contextual_widgets()
        self._refresh_umap_reducer_enablement()
        self._refresh_analysis_unit_lock()
        self._refresh_regression_backend()

        # Bucket into sections.
        cats = categories_for_app(self.app_key, get_categories())
        used_keys = set()
        # Categories that don't apply to a given app (e.g. the classify app
        # trains a Torch model, not Cellpose — so it gets no Cellpose tab).
        hidden = _APP_HIDDEN_CATEGORIES.get(self.app_key, set())
        split_by_object = set(_shared_category_parents())
        sections: List[SettingsSection] = []
        for cat_name, keys in cats.items():
            if cat_name in hidden:
                continue
            rows: List[Tuple[str, QWidget]] = []
            # THE KEYS, ALONGSIDE THE ROWS. A row is `(label, widget)` and a
            # label is a sentence for a human, so the object a row belongs to
            # can only be read off the KEY -- the same confusion that put the
            # plate map on nothing at all when a label was matched instead.
            row_keys: List[str] = []
            for k in keys:
                if k in self._widgets and k not in used_keys:
                    rows.append((self._label_for(k), self._widgets[k]))
                    row_keys.append(k)
                    used_keys.add(k)
            if not rows:
                continue
            if cat_name in split_by_object:
                own, children = _split_rows_by_object(rows, row_keys)
                sections.append(SettingsSection(cat_name, own, children))
            else:
                sections.append(SettingsSection(cat_name, rows))

        # Trailing 'Other' for anything not in a category.
        remaining = [(self._label_for(k), self._widgets[k])
                     for k in self._widgets if k not in used_keys]
        if remaining:
            sections.append(SettingsSection("Other", remaining))

        # ONCE THE SCREEN HAS LAID THE ROWS OUT. This hands the rows back and
        # the screen builds each label and puts the pair into a QFormLayout
        # afterwards -- so there is no ROW to hide yet, and hiding the field
        # here and nothing else would leave its name behind on an empty row.
        # Zero delay, so it lands on the next turn of the event loop, before
        # the panel has been painted.
        #
        # BOUND TO THE PANEL'S OWN WIDGET, which is the three-argument form's
        # whole point: the connection is dropped when that widget is
        # destroyed, so a screen closed inside the same turn is not reached
        # into afterwards. Nothing is scheduled at all without one -- a
        # `SettingsWidgets` built with no parent is being used for its values
        # and has no rows to lay out.
        if self._parent is not None:
            QTimer.singleShot(0, self._parent, self.refresh_object_visibility)

        return _nest_sections(sections)

    def tooltip_for(self, key: str) -> str:
        """Return the HTML-formatted tooltip for a given setting key."""
        return format_tooltip(self._tooltips.get(key, ""), self.app_key, key)

    def plain_tooltip_for(self, key: str) -> str:
        """Return the plain-text hint (description + docs URL) for a setting."""
        return plain_tooltip(self._tooltips.get(key, ""), self.app_key, key)

    # ------------------------------------------------------------------
    # Finding a setting among the many
    # ------------------------------------------------------------------
    #
    # Mask alone renders 190 settings under thirteen collapsed headings.
    # Someone who knows the knob exists still has to guess which heading
    # somebody else filed it under, and someone who only knows what they want
    # to change ("stop merging touching cells") has no entry point at all.
    #
    # So the haystack is deliberately wider than the key: the description is
    # the only part of a setting written in the language a user thinks in.
    # Searching "gpu" has to find `n_jobs`, and "touching" has to find
    # `merge_edge_pathogen_cells`, and neither word is in either name.

    def search_text_for(self, key: str) -> str:
        """The lower-cased haystack one setting is matched against.

        Three fields, in the order a reader would scan them: the key as the
        API spells it, the label as the form spells it, and the description
        as the tooltip explains it.

        :param key: the setting key.
        """
        return " ".join((
            str(key),
            self._label_for(key),
            self.plain_tooltip_for(key),
        )).lower()

    def keys_matching(self, query: str) -> List[str]:
        """Setting keys matching every whitespace-separated term in ``query``.

        Terms are ANDed and matched as substrings, which is what makes
        "cell diameter" narrow rather than widen — the alternative, OR, turns
        a second word into a way of getting *more* results, which is the
        opposite of what typing more means.

        An empty or whitespace-only query matches everything, so the caller
        can wire this straight to ``textChanged`` without special-casing the
        moment the box is cleared.

        :param query: raw text from the search box.
        :returns: matching keys, in the order the widgets were built.
        """
        terms = str(query or "").lower().split()
        if not terms:
            return list(self._widgets)
        out: List[str] = []
        for key in self._widgets:
            haystack = self.search_text_for(key)
            if all(term in haystack for term in terms):
                out.append(key)
        return out

    def modified_keys(self) -> List[str]:
        """Setting keys whose widget no longer holds the module's default.

        Compared with the same normaliser the run journal and the settings
        diff use, so "differs from default" means one thing across the app.
        Without that, a value round-tripped through CSV — ``channels`` read
        back as the string ``"[0, 1, 2]"`` — reads as an edit here and as
        unchanged there.

        :returns: keys in the order the widgets were built.
        """
        from ..settings_diff import _values_equal

        out: List[str] = []
        for key, widget in self._widgets.items():
            if key not in self._defaults:
                # Rendered but not defaulted: there is nothing to differ
                # from, so calling it modified would be an assertion the
                # module never made.
                continue
            try:
                current = self._coerce_to_expected_type(
                    key, self._read_widget(widget))
            except Exception:
                continue
            if not _values_equal(current, self._defaults[key]):
                out.append(key)
        return out

    def essential_keys(self) -> List[str]:
        """The rendered subset of :func:`essential_keys` for this module.

        Filtered to keys that actually produced a widget, so a key named in
        a layout but skipped by ``convert_settings_dict_for_gui`` cannot make
        the disclosure control promise a row that is not there.
        """
        return [key for key in essential_keys(self.app_key)
                if key in self._widgets]

    def _label_for(self, key: str) -> str:
        try:
            from spacr.plugins import get_app
            plugin_app = get_app(self.app_key)
            if plugin_app is not None and key in plugin_app.labels:
                return plugin_app.labels[key]
        except Exception:
            pass
        if self.app_key in ("measure", "external_masks"):
            measure_labels = {
                "uninfected": "Keep uninfected cells",
                "cytoplasm": "Measure cytoplasm",
                "merge_edge_pathogen_cells": "Merge edge-pathogen cells",
            }
            if key in measure_labels:
                return measure_labels[key]
        if self.app_key == "umap":
            if key == "exclude_rows":
                return "Exclude"
            if key == "exclude":
                return "Exclude features"
        return setting_label(key)

    def _widget_for(self, kind: str, options: Any, default: Any,
                    key: str) -> Optional[QWidget]:
        parent = self._parent
        # MORE THAN ONE DATABASE (instruction 109). A screen acquired as three
        # plates is three project folders, and `generate_image_umap` has
        # always taken a list of them -- the panel was the half that could
        # only express one, so the comparison the user actually wants could
        # not be asked for from the application at all. The control adds,
        # removes, and SAYS WHAT THE MERGE WOULD COST before anything runs.
        #
        # `on_colour_by` writes into this panel's own `color_by` field, looked
        # up when the box is ticked rather than captured now: the fields are
        # built in one pass and `color_by` does not exist yet at this point.
        if self.app_key == "umap" and key == "src":
            return DatabaseSetWidget(
                value=self._defaults.get(key, default),
                mode="folder",
                table="cell",
                title="Choose one or more spaCR project folders",
                on_colour_by=partial(self.set_value_for_key, "color_by"),
                parent=parent,
            )
        if self.app_key == "umap" and key == "exclude_rows":
            return RowExclusionEditor(
                value=self._defaults.get(key, default),
                parent=parent,
            )
        if self.app_key == "external_masks" and key == "inputs":
            return ExternalMaskInputWidget(
                value=self._defaults.get(key, default),
                parent=parent,
            )
        # Not scoped to one app: every module that crops object PNGs offers
        # this key, and all of them mean the same thing by it.
        # `classes` is a dict of name -> {column, value}, so it gets the
        # editor that can populate it from a column rather than a text box the
        # user has to type JSON into.
        if key == "classes":
            widget = ClassEditorWidget(
                value=self._defaults.get(key, default),
                parent=parent,
            )
            frame = getattr(self, "_preview_frame", None)
            if frame is not None:
                widget.set_frame(frame)
            return widget
        if key == "png_channel_mapping":
            return ChannelMappingWidget(
                value=self._defaults.get(key, default),
                parent=parent,
            )
        # A setting that names input files gets a file dialog and a drop
        # target, not a box to type absolute paths into. Checked before the
        # chip-editor and combo paths below, because several of these keys are
        # declared ``list`` and would otherwise take the free-text route.
        if key in PATH_LIST_KEYS:
            return FilePathListWidget(
                value=self._defaults.get(key, default),
                kind=PATH_LIST_KEYS[key],
                title=PATH_LIST_TITLES.get(key, "Choose input files"),
                single=key in PATH_LIST_SINGLE_KEYS,
                parent=parent,
            )
        if key == "paired_data":
            return PairedFileTableWidget(
                value=self._defaults.get(key, default), parent=parent)
        # A setting whose value NAMES A COLUMN of an input CSV gets the box
        # plus a button that reads those CSVs' header row. Checked before the
        # combo and chip paths below because these keys are declared `str`
        # and would otherwise take the plain text box that made a typo
        # indistinguishable from a name.
        #
        # The paths are read WHEN THE BUTTON IS PRESSED, not here: the user
        # chooses their input files after this panel is built, so a list read
        # at construction is always the empty one.
        source = CSV_COLUMN_SOURCES.get(self.app_key, {}).get(key)
        if source is not None:
            return _CsvColumnField(
                key=key,
                default=self._defaults.get(key, default),
                # `partial`, not a lambda: the callable outlives this call
                # and is read on a button press minutes later, so what it
                # captures should be visible rather than implied.
                paths=partial(self._input_csv_paths, source.roles),
                what=source.what,
                parent=parent,
            )
        # WHO fits the model gets a control that can say why an option is
        # not choosable and what each one is. A plain combo could only offer
        # eight labels and be silent about all of it -- which is what it did
        # until 2026-08-18. See _RegressionBackendField.
        if key == "regression_backend":
            return _RegressionBackendField(
                default=self._defaults.get(key, default),
                regression_type=self._defaults.get("regression_type"),
                parent=parent,
            )
        app_options = _APP_COMBO_OPTIONS.get(self.app_key, {})
        if key in app_options:
            kind = "combo"
            options = app_options[key]
        # Two inventories are owned by the modules that implement them, so the
        # dropdown cannot list a model spaCR cannot fit or omit a correction it
        # can apply. Both imports are cheap: regression_families reads only
        # regression_spec, which imports nothing, and multiple_testing imports
        # only numpy at module scope.
        if key == "regression_type":
            # THE SAME TABLE THE OTHER ROUTE READS -- see
            # _regression_type_menu, which settings_spec's
            # _regression_type_choices shares the family half of. Building a
            # second list here out of the bare inventory is what let this
            # panel show nineteen unlabelled names while the other route
            # showed them grouped and explained.
            #
            # 'auto' is the readable spelling of the historical None, which
            # ml.regression turns into check_distribution(response). It is
            # normalised back to None in
            # settings.get_perform_regression_default_settings, so the fit
            # path is unchanged and old settings CSVs holding None still work.
            #
            # A bare string and a (value, label) pair may share this list --
            # the combo builder below takes either -- and every entry here is
            # a pair, so the stored value is what a settings CSV gets while
            # the caption says which kind of fit it is and what it assumes.
            kind = "combo"
            options = _regression_type_menu()
        elif key == "multiple_testing_method":
            from spacr.multiple_testing import method_choices
            kind = "combo"
            options = method_choices()
        if self.app_key == "umap" and key == "metric":
            # One closed alphabet rather than a text field that accepts a
            # typo and fails after the reducer starts.  Importing the constant
            # does not import umap-learn (and therefore does not put a model
            # load on the GUI thread); the runtime validator still consults
            # the installed package.
            from spacr.hyperparam import UMAP_METRICS
            kind = "combo"
            options = list(UMAP_METRICS)
        if self.app_key == "map_barcodes" and key == "regex":
            return BarcodeRegexWidget(
                value=self._defaults.get(key, default),
                parent=parent,
            )
        # A closed alphabet gets a control that cannot express anything
        # outside it. Checked BEFORE the chip-editor override below, because
        # `train_channels` is in CHANNEL_LIST_KEYS and would otherwise take
        # the free-text path that let 'x' through.
        if key in FIXED_ALPHABETS:
            return _AlphabetSelect(
                key=key,
                default=self._defaults.get(key, default),
                choices=FIXED_ALPHABETS[key],
                parent=parent,
            )
        # 'Exclude' names measurement columns to drop from the feature set,
        # and there is never a reason it should be exactly one -- but
        # spacr.settings declares it (str, None), so list_shape_for
        # (deliberately conservative, and reading only what is declared) sent
        # it to a plain text box. One column per run, and the SQL button
        # overwrote whatever was already there. It gets the same chip strip
        # as Classify (CV)'s `classes`: type a name and it becomes a chip to
        # the right, remove them one at a time, and the SQL button beside it
        # (COLUMN_TABLES) hands back however many columns were selected.
        # Consumers already take either shape -- utils.filter_dataframe_
        # features and preprocess_data both wrap a bare str in a list -- so a
        # settings CSV written before this still loads, and one written now
        # still runs on the CLI.
        if key in EXCLUDE_LIST_KEYS:
            return _ListEditor(
                key=key,
                default=self._defaults.get(key, default),
                nested_capable=False,
                allow_none=True,
                element_type=str,
                container=list,
                parent=parent,
            )
        # Unlike enumerated strings, a list remains a list in every module.
        # The legacy converter presents channel lists and timelapse objects as
        # dropdowns of Python literals. Render them with the same chip editor
        # as manders_thresholds so users can add/remove arbitrary values.
        actual_default = self._defaults.get(key, default)
        if key == "timelapse_objects" or (
            key in CHANNEL_LIST_KEYS
            and list_shape_for(key, actual_default) is not None
        ):
            kind = "entry"
        if kind == "check":
            w = Toggle()
            w.setChecked(bool(default))
            return w
        if kind == "combo":
            # _ValueCombo, not QComboBox: some of these lists are
            # (value, label) pairs, and on a plain combo `setCurrentText`
            # takes the caption only -- so "choose ols" silently does nothing
            # as soon as the caption stops being the value.
            w = _ValueCombo()
            # Long inventories (notably UMAP's complete metric list) must not
            # become the minimum width of the whole settings sidebar. The
            # popup still shows every option; the closed control elides.
            w.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(12)
            for opt in (options or []):
                # A (value, label) pair shows the LABEL and stores the VALUE.
                # Instruction 171 wants "load images" and "stream images" in
                # those words in every panel that offers the choice, while
                # 'png' and 'merged' go on meaning what they meant to every
                # settings file already written.
                if isinstance(opt, tuple) and len(opt) == 2:
                    stored, shown = opt
                else:
                    stored = opt
                    shown = "None" if opt is None else str(opt)
                w.addItem(str(shown), userData=stored)
            # Pre-select the value THIS module declares, not the one
            # hard-coded in gui_utils.convert_settings_dict_for_gui's
            # special_cases table. That table is one row per key for the whole
            # app, so it shipped 'resnet50' as the model_type default to
            # Classify (which sets 'maxvit_t') and to Activation Maps (which
            # sets 'maxvit'), and '[0,1,2,3]' as the channels default to
            # Cellpose Masks (which sets [0, 0]).
            if key in self._defaults:
                default = self._defaults[key]
            for i in range(w.count()):
                if w.itemData(i) == default or w.itemText(i) == str(default):
                    w.setCurrentIndex(i)
                    break
            else:
                # The default is not one of the curated options. Silently
                # leaving index 0 selected substitutes a value the module
                # never asked for -- the activation-map app defaults
                # channels to [1, 2, 3] and the channel combo only lists
                # '[0,1,2,3]', so every run started with a different channel
                # set than the defaults declare. Offer the real default too.
                if default is not None and str(default) != "":
                    w.insertItem(0, str(default), userData=default)
                    w.setCurrentIndex(0)
            return w
        if kind == "entry":
            # A list setting gets the chip editor, not a text box holding a
            # Python literal. The shape is decided from expected_types plus
            # the REAL default (self._defaults), because
            # convert_settings_dict_for_gui has already str()'d the value
            # that arrives here as ``default``.
            shape = list_shape_for(key, self._defaults.get(key, default))
            if shape is not None:
                nested_capable, allow_none, element_type, container = shape
                return _ListEditor(key=key,
                                   default=self._defaults.get(key, default),
                                   nested_capable=nested_capable,
                                   allow_none=allow_none,
                                   element_type=element_type,
                                   container=container)
            # BY NAME, BEFORE THE TYPE SNIFF. These settings take a number
            # or the word "auto", and the shipped default happens to be a
            # number -- so inferring from it built a control that could not
            # express half of what the setting accepts.
            if key in AUTO_OR_NUMBER_SETTINGS:
                return _auto_or_number_box(self._defaults.get(key, default))
            # A PLANE THIS RUN MAY NOT HAVE, for the same reason and one step
            # further: `cell_mask_dim` names a plane of the merged stack, and
            # a screen with no nucleus has no nucleus plane. The control is
            # otherwise chosen from the SHIPPED DEFAULT, so the three that
            # ship a number -- cell 4, nucleus 5, pathogen 6 -- got a spin
            # box, and a spin box has no empty state: the value could be
            # changed but never CLEARED, and being made to name a plane for
            # an object that is not in the run is being made to lie about it.
            # The organelle slots ship None and have always had the box
            # below; this is what makes the family agree.
            if _is_clearable_plane_setting(key):
                w = _ScalarEdit()
                w.set_value(self._defaults.get(key, default))
                return w
            # Choose widget by inferred type from the DEFAULT value
            if isinstance(default, bool):
                w = Toggle()
                w.setChecked(default)
                return w
            if isinstance(default, int):
                w = QSpinBox()
                # Wide enough for the defaults the modules actually ship:
                # the replication assay's max_area is 1e9, and a +/-1e6 range
                # silently clamped it to 1e6 -- a thousand-fold change to the
                # largest vacuole the assay will score, applied before the
                # user touched anything.
                w.setRange(-2_147_483_648, 2_147_483_647)
                w.setValue(default)
                return w
            if isinstance(default, float):
                w = QDoubleSpinBox()
                low, high, step = _float_domain(key, default)
                w.setRange(low, high)
                w.setSingleStep(step)
                w.setDecimals(6)
                w.setValue(default)
                return w
            if isinstance(default, list):
                w = _ListEdit()
                w.set_value(default)
                return w
            # Fallback — string or None
            w = _ScalarEdit()
            w.set_value(default)
            return w
        return None

    @staticmethod
    def _coerce_to_expected_type(key: str, value: Any) -> Any:
        """Parse a raw widget string into the type ``settings`` declares.

        A setting whose DEFAULT is None gets a free-text widget, so it comes
        back as a raw string even when ``spacr.settings.expected_types`` says
        it is an int -- and cellpose received ``diameter='37'``. The Tk GUI
        never had this problem because it runs
        ``settings.check_settings(vars_dict, expected_types)`` before
        dispatch; the Qt path had no equivalent step. check_settings itself
        cannot be reused here: it takes the Tk widget map
        ``key -> (label, widget, var, frame)``, not a plain dict.

        Anything not declared, or not parseable, is returned untouched -- this
        coerces, it does not validate, and it must never turn a real value
        into None behind the user's back.
        """
        if not isinstance(value, str):
            return value
        try:
            from ... import settings as _settings
            declared = _settings.expected_types.get(key)
        except Exception:
            return value
        if declared is None:
            return value
        allowed = declared if isinstance(declared, tuple) else (declared,)
        text = value.strip()
        if text == "" or text == "None":
            return None if type(None) in allowed else value
        for typ in allowed:
            if typ is bool:
                if text.lower() in ("true", "false"):
                    return text.lower() == "true"
                continue
            if typ in (int, float):
                try:
                    return typ(text)
                except ValueError:
                    continue
            if typ in (list, tuple):
                # The curated combos ('channels', 'crop_mode',
                # 'train_channels', 'timelapse_objects', ...) offer their
                # options as TEXT -- "['r','g','b']" -- so a list setting
                # picked from a dropdown reached the pipeline as a string and
                # got iterated character by character. The chip editor already
                # returns a real list; this is the same repair for the combos.
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(parsed, (list, tuple)):
                    return typ(parsed)
                continue
        return value

    #: Settings whose value has ONE canonical form, whatever shape the
    #: widget hands back. `channel_of_interest` is drawn as a multi-select,
    #: so one channel comes back as `[3]` where the default is `3` -- the
    #: same feature space, but a panel that rewrites a default makes every
    #: settings file differ from it and breaks "has this been changed?".
    CANONICAL_READERS = {
        "channel_of_interest": "spacr.utils:feature_selection",
    }

    def _canonical(self, key: str, value: Any) -> Any:
        """Put a widget's answer into the one form the setting is stored in."""
        where = self.CANONICAL_READERS.get(key)
        if where is None:
            return value
        module_name, function_name = where.split(":")
        try:
            import importlib

            reader = getattr(importlib.import_module(module_name),
                             function_name)
            return reader(value)
        except Exception:                                    # noqa: BLE001
            LOGGER.debug("could not canonicalise %s", key, exc_info=True)
            return value

    def collect(self) -> Dict[str, Any]:
        """Read all widgets and return the current settings dict."""
        out: Dict[str, Any] = {}
        for key, w in self._widgets.items():
            out[key] = self._canonical(
                key, self._coerce_to_expected_type(key, self._read_widget(w)))
        # Also carry over any defaults we didn't render (e.g. things not
        # in the categories map that convert_settings_dict_for_gui also
        # skipped).
        for k, v in self._defaults.items():
            out.setdefault(k, v)
        return self._organelle_slots_worth_keeping(out)

    def _organelle_slots_worth_keeping(self,
                                       settings: Dict[str, Any]
                                       ) -> Dict[str, Any]:
        """Drop the slots this run neither has nor has anything to say about.

        THE PANEL AND THE FILE ARE NOT THE SAME QUESTION. The panel builds a
        control for every slot that can be named, because the count has to
        have something to reveal; a settings file written that way would bury
        the four slots a run uses under twelve hundred keys nobody set.

        WHAT SURVIVES: every slot ``number_of_organelles`` reaches, every slot
        the MODULE itself declared, and every slot above the count holding
        something other than the value the panel invented for it. That last
        part is the whole of "a file written at seven opens at two and still
        carries seven" -- the five hidden slots hold what the file said, not
        what the panel put there, so they are written back out and raising
        the number again brings their answers with them.

        CONTIGUOUS, because a slot's number is its position: keeping the
        seventh without the fifth and sixth would leave a settings dict that
        ``number_of_organelles`` cannot describe.
        """
        from ..settings_diff import _values_equal
        from ...organelle_types import (organelle_count, organelle_number,
                                        organelle_role_of, organelle_roles)

        invented = getattr(self, "_slots_the_panel_added", None)
        if not invented:
            return settings
        roles = {key: organelle_role_of(key) for key in settings}
        highest = organelle_count(settings)
        for key, role in roles.items():
            if role is None:
                continue
            number = organelle_number(role)
            if number <= highest:
                continue
            if key in invented:
                try:
                    if _values_equal(settings[key], invented[key]):
                        continue
                except Exception:                            # noqa: BLE001
                    pass
            highest = number
        kept = set(organelle_roles(highest))
        return {key: value for key, value in settings.items()
                if roles[key] is None or roles[key] in kept}

    def set_value_for_key(self, key: str, value: Any) -> bool:
        """Write ``value`` into the widget bound to ``key`` (if present).

        Used by the Live Preview's "Propagate settings" toggle to push
        interactively-tuned values back into the main settings panel.
        Returns True if the key existed and was set.
        """
        w = self._widgets.get(key)
        if w is None:
            return False
        try:
            if isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, QSpinBox):
                w.setValue(int(value))
            elif isinstance(w, QDoubleSpinBox):
                if str(w.specialValueText() or "") == AUTO_TEXT:
                    _set_auto_or_number(w, value)
                else:
                    w.setValue(float(value))
            elif isinstance(w, QComboBox):
                idx = w.findData(value)
                if idx < 0:
                    idx = w.findText(str(value))
                if idx >= 0:
                    w.setCurrentIndex(idx)
                else:
                    w.setEditText(str(value))
            elif isinstance(
                w,
                (
                    _AlphabetSelect, _ListEditor, _ListEdit, _ScalarEdit,
                    BarcodeRegexWidget, RowExclusionEditor,
                    ExternalMaskInputWidget, ChannelMappingWidget,
                    ClassEditorWidget, DatabaseSetWidget,
                    FilePathListWidget,
                    PairedFileTableWidget, _CsvColumnField,
                    _RegressionBackendField,
                ),
            ):
                w.set_value(value)
            elif isinstance(w, QLineEdit):
                w.setText("" if value is None else str(value))
            else:
                return False
        except Exception:
            return False
        if key in {"src", "tables"}:
            self._refresh_contextual_widgets()
        elif self.app_key == "regression":
            self._refresh_setting_dependencies()
        if key in {"reduction_method", "spectral_affinity"}:
            self._refresh_umap_reducer_enablement()
        if key == "analysis_unit":
            self._refresh_analysis_unit_lock()
        return True

    def set_hidden_value(self, key: str, value: Any) -> bool:
        """Update a deliberately hidden run setting.

        Some values have a dedicated control outside the form.  Image UMAP's
        action-strip GPU toggle is one: duplicating it as a form checkbox would
        create two sources of truth.  Hidden does not mean absent (invariant
        6), so the value lives in ``_defaults`` and still reaches collect().
        """
        if key not in self._defaults or key not in _APP_HIDDEN_KEYS.get(
                self.app_key, set()):
            return False
        self._defaults[key] = self._coerce_to_expected_type(key, value)
        return True

    def _on_regression_type_changed(self, *_args) -> None:
        """Re-judge the backends against the family now being fitted."""
        self._refresh_regression_backend()

    def _refresh_regression_backend(self) -> None:
        """Point the backend control at the panel's current regression type.

        Reads the WIDGET rather than the defaults, so the greying and the
        description follow what is on screen. With no `regression_type`
        widget -- another module, or a layout that hides it -- the declared
        default is used, which is what the run would fit anyway.
        """
        backend = self._widgets.get("regression_backend")
        if not isinstance(backend, _RegressionBackendField):
            return
        widget = self._widgets.get("regression_type")
        if widget is None:
            value = self._defaults.get("regression_type")
        else:
            try:
                value = self._read_widget(widget)
            except Exception:                                  # noqa: BLE001
                value = self._defaults.get("regression_type")
        backend.set_regression_type(value)

    def _on_umap_reducer_changed(self, *_args) -> None:
        """Re-grey method-specific Image UMAP controls immediately."""
        self._refresh_umap_reducer_enablement()

    def _refresh_analysis_unit_lock(self) -> None:
        """Apply and display settings constrained by ``analysis_unit``.

        Constraints come from :mod:`spacr.settings_advisor`, which also
        validates imported settings that did not pass through this panel.
        Keeping one constraint registry ensures the interface and preflight
        checks use the same requirements.
        """
        control = self._widgets.get("analysis_unit")
        if control is None:
            return
        try:
            from ...settings_advisor import requirements_for_unit
        except Exception:                                    # noqa: BLE001
            return
        unit = str(self._read_widget(control) or "well").strip().lower()
        required = requirements_for_unit(unit)
        # EVERY SETTING ANY UNIT CONSTRAINS, so switching back to `well`
        # releases what `cell` locked. Refreshing only the current unit's
        # keys would leave a control greyed after the reason for it was
        # withdrawn.
        try:
            from ...settings_advisor import UNIT_REQUIREMENTS

            owned = set().union(*(set(v) for v in UNIT_REQUIREMENTS.values()))
        except Exception:                                    # noqa: BLE001
            owned = set(required)
        # WHAT THIS RULE ITSELF LOCKED LAST TIME. Only these are released,
        # so a control greyed by another rule stays greyed.
        released = set(getattr(self, "_unit_locked", set()))
        note = (f"Fixed by analysis_unit={unit!r}: the run reads this value "
                f"and no other, so it is shown rather than left editable. "
                f"Choose analysis_unit='well' to set it yourself.")
        for key in sorted(owned):
            widget = self._widgets.get(key)
            if widget is None:
                continue
            if key in required:
                # SET IT, THEN GREY IT. A greyed control still showing the
                # old value tells the user the run will use that value, and
                # it will not -- which is worse than an editable control
                # that disagrees, because it looks settled.
                # `set_value_for_key`, which is the one writer -- a second
                # way of putting a value into a widget is a second set of
                # type rules to keep in step. It re-enters this method only
                # for `analysis_unit` itself, which is never a required key.
                self.set_value_for_key(key, required[key])
                widget.setEnabled(False)
                _apply_greyed_note(widget, note)
            elif key in released:
                # RELEASE ONLY WHAT THIS RULE GREYED. `analysis_mode` is also
                # greyed by the inference rule -- it is set for you by
                # inference='parametric' -- and a blanket setEnabled(True)
                # here undid that, so the combo came back editable while
                # something else was still deciding its value. Enabling a
                # control another rule disabled is worse than leaving one
                # greyed: the user changes it and the run ignores them.
                widget.setEnabled(True)
                _clear_greyed_note(widget)
        self._unit_locked = {k for k in required if k in self._widgets}
        # WHATEVER ELSE HAD A SAY, AFTER. The other refreshers re-assert
        # their own greying over anything this one just released.
        if hasattr(self, "_refresh_setting_dependencies"):
            try:
                self._refresh_setting_dependencies()
            except Exception:                                # noqa: BLE001
                LOGGER.debug("could not re-run the dependency rules",
                             exc_info=True)

    def _refresh_umap_reducer_enablement(self) -> None:
        """Enable only the settings the selected reducer actually reads."""
        if self.app_key != "umap":
            return
        selector = self._widgets.get("reduction_method")
        if selector is None:
            return
        method = str(self._read_widget(selector) or "umap").strip().lower()
        if method not in _UMAP_REDUCER_SETTINGS:
            return
        owned = set().union(*_UMAP_REDUCER_SETTINGS.values())
        active = _UMAP_REDUCER_SETTINGS[method]
        note = f"Used only when dimensionality reduction is {method}."
        for key in owned:
            control = self._widgets.get(key)
            if control is None:
                continue
            enabled = key in active
            if key == "spectral_n_neighbors" and method == "spectral":
                affinity = self._widgets.get("spectral_affinity")
                enabled = str(
                    self._read_widget(affinity) if affinity is not None
                    else "nearest_neighbors"
                ) == "nearest_neighbors"
            control.setEnabled(enabled)
            if enabled:
                _clear_greyed_note(control)
            else:
                _apply_greyed_note(control, note)

        metric = self._widgets.get("metric")
        if metric is not None:
            # The projection may ignore this setting, but DBSCAN always reads
            # it. Keep the shared metric editable instead of greying a control
            # that can still change the result.
            metric.setEnabled(True)
            _clear_greyed_note(metric)

    def _refresh_classifier_family_enablement(self) -> None:
        """Grey the settings the OTHER classifier family reads.

        Only the merged Classify module has this control; the two original
        modules are one family each and have nothing to grey.

        Same rule as the training basis: greyed, never removed
        (INVARIANTS 6), and the list lives in spacr.classify so the panel and
        the pipeline cannot drift apart about which settings matter.
        """
        widget = self._widgets.get("classifier_family")
        if widget is None:
            return
        try:
            from spacr.classify import (
                FAMILY_SETTINGS, inapplicable_settings, resolve_family,
            )
            family = resolve_family(
                {"classifier_family": self._read_widget(widget)})
            greyed = set(inapplicable_settings(family))
            owned = {k for keys in FAMILY_SETTINGS.values() for k in keys}
        except Exception:
            # An unknown family is the pipeline's error to raise, loudly, at
            # run time. Greying on a guess would hide the control the user
            # needs to fix it.
            return

        for key, control in self._widgets.items():
            if key in greyed:
                control.setEnabled(False)
                _apply_greyed_note(control, _family_note(family))
            elif key in owned:
                control.setEnabled(True)
                _clear_greyed_note(control)

    def _on_classifier_family_changed(self, *_args) -> None:
        """Re-grey the panel when the classifier family changes."""
        self._refresh_classifier_family_enablement()

    def _on_training_basis_changed(self, *_args) -> None:
        """Re-grey the panel when the training basis changes.

        A named method rather than a lambda: INVARIANTS 4 is about
        QThread.finished specifically, but the same lifetime reasoning
        applies to any signal connection that has to outlive the call that
        made it.
        """
        self.refresh_training_basis_enablement()

    def refresh_training_basis_enablement(self) -> None:
        """Disable settings that the selected training basis does not use.

        Controls remain present so their values are still collected and the
        pipeline does not substitute defaults for missing keys. Applicability
        is read from :mod:`spacr.training_basis`, which is shared with the
        pipeline.
        """
        self._refresh_classifier_family_enablement()
        widget = self._widgets.get("dataset_mode")
        if widget is None:
            return
        try:
            from spacr.training_basis import (
                inapplicable_settings, resolve_basis,
            )
            basis = resolve_basis({"dataset_mode": self._read_widget(widget)})
            greyed = set(inapplicable_settings(basis))
        except Exception:
            # An unrecognised basis is the pipeline's error to raise, loudly,
            # at run time. Greying nothing is the safe response here --
            # disabling controls on a guess would hide the one the user needs.
            return

        for key, control in self._widgets.items():
            if key in greyed:
                control.setEnabled(False)
                _apply_greyed_note(control, _basis_note(basis))
            elif key in _ALL_BASIS_SETTINGS:
                control.setEnabled(True)
                _clear_greyed_note(control)

    def _refresh_contextual_widgets(self) -> None:
        """Refresh widgets whose choices come from the selected data source."""
        self.refresh_training_basis_enablement()
        self._refresh_setting_dependencies()
        editor = self._widgets.get("exclude_rows")
        if not isinstance(editor, RowExclusionEditor):
            return
        src_widget = self._widgets.get("src")
        tables_widget = self._widgets.get("tables")
        source = self._read_widget(src_widget) if src_widget is not None else None
        tables = (
            self._read_widget(tables_widget)
            if tables_widget is not None
            else self._defaults.get("tables")
        )
        editor.set_source(source, tables)

    def _rules_for_this_panel(self) -> Dict[str, Any]:
        """The applicability rules THIS panel can honestly evaluate.

        ``settings.setting_dependencies`` is keyed by setting NAME and says
        nothing about which screen a setting appears on --
        ``batch_correction='none'`` kills ``batch_column`` wherever the two
        are shown together, which is four screens, not one. Both entry points
        below nevertheless opened with ``if self.app_key != 'regression'``,
        so on Image UMAP, Classify (merged) and ML Analyze all seven
        ``batch_*`` controls stayed live and editable under the default
        ``batch_correction='none'``. The table was module-agnostic; the
        wiring was not.

        The guard is not widened into an allow-list of app keys, because an
        allow-list is the same bug with a longer line in it -- the next
        module to gain a gated setting silently would not gate. What the
        guard was actually protecting against is stated directly instead:

          * the setting must be ON THIS PANEL, or there is nothing to grey;
          * at least one of the rule's SOURCES must be on this panel too.

        The second is the one that matters. A predicate reads other settings,
        and on a panel that shows the ruled setting but none of the settings
        it depends on, the predicate would be evaluated against a default the
        user can neither see nor change -- a control greyed by an invisible
        value, which nobody can ever re-enable. Such a rule must not fire at
        all. ``any`` rather than ``all`` because a rule combined from two
        independent reasons carries the union of both reasons' sources, and
        a panel is entitled to have only one of them.
        """
        if not self._widgets:
            return {}
        try:
            from spacr.settings import get_setting_dependencies
            dependencies = get_setting_dependencies()
        except Exception:
            return {}
        return {
            key: rule for key, rule in dependencies.items()
            if key in self._widgets
            and any(source in self._widgets
                    for source in rule.get('sources', ()))
        }

    def _connect_setting_dependency_signals(self) -> None:
        """Re-evaluate applicability whenever one of its source keys moves."""
        dependencies = self._rules_for_this_panel()
        sources = {source for rule in dependencies.values()
                   for source in rule.get('sources', ())}
        for key in sources:
            widget = self._widgets.get(key)
            if widget is not None:
                _connect_value_changed(widget,
                                       self._on_dependency_source_changed)

    def _on_dependency_source_changed(self, *_args) -> None:
        self._refresh_setting_dependencies()

    def _current_dependency_settings(self) -> Dict[str, Any]:
        current = dict(self._defaults)
        for key, widget in self._widgets.items():
            try:
                current[key] = self._coerce_to_expected_type(
                    key, self._read_widget(widget))
            except Exception:
                pass
        return current

    def _loaded_table_paths(self, current: Dict[str, Any]):
        """Return index-tagged score and count CSVs loaded by the user.

        Paired inputs share one logical index so a score file and its count
        file represent one plate when neither contains a plate column. Legacy
        flat input keys remain supported after the paired table.
        """
        return [(index, path)
                for index, _role, path in self._input_tables(current)]

    @staticmethod
    def _input_tables(current: Dict[str, Any],
                      roles: Tuple[str, ...] = ('score', 'count')):
        """``[(index, role, path)]`` for the loaded regression input CSVs.

        THE ONE PLACE THE PAIRED TABLE IS UNPACKED, and it carries the role
        because its two callers need different projections of the same read:
        the plate-count scan wants every path with its logical index, and the
        CSV column picker wants one side only -- `dependent_variable` names a
        column of the SCORE file and offering it `grna` from the count file
        would offer a name the run cannot use. A second unpacker would be a
        second thing to update the day the table gains a third side.
        """
        pairs = current.get('paired_data')
        if isinstance(pairs, (list, tuple)) and pairs:
            found = []
            for index, row in enumerate(pairs):
                if not isinstance(row, dict):
                    continue
                for role in roles:
                    path = row.get(role)
                    if path:
                        found.append((index, role, path))
            if found:
                return found
        found = []
        for role in roles:
            paths = current.get(f'{role}_data') or []
            if isinstance(paths, (str, os.PathLike)):
                paths = [paths]
            found.extend((index, role, path)
                         for index, path in enumerate(paths))
        return found

    def _input_csv_paths(self, roles: Tuple[str, ...]) -> List[str]:
        """The input CSVs a column picker for ``roles`` should read.

        Deduplicated in order: one score CSV shared by two plate rows is one
        file to read, and `spacr.columns.available` would merge its columns
        anyway.

        Only the three input keys are read, not the whole panel: this runs on
        a button press and `_current_dependency_settings` walks every widget
        on the screen to answer a question about three of them.
        """
        current = {}
        for key in ('paired_data', 'score_data', 'count_data'):
            widget = self._widgets.get(key)
            current[key] = (self._read_widget(widget) if widget is not None
                            else self._defaults.get(key))
        seen: List[str] = []
        for _index, _role, path in self._input_tables(current, tuple(roles)):
            text = os.fspath(path)
            if text not in seen:
                seen.append(text)
        return seen

    @staticmethod
    def _plate_context(paths) -> Dict[str, Any]:
        """Inspect only CSV headers/plate columns; never load feature data."""
        sources = []
        for fallback_index, item in enumerate(paths or []):
            logical_index, path = (item if isinstance(item, tuple)
                                   else (fallback_index, item))
            if path and os.path.isfile(os.fspath(path)):
                sources.append((logical_index, os.fspath(path)))
        if not sources:
            return {'plate_count': None, 'has_plate_id': False}
        # A very large single-plate file should not stall the GUI merely to
        # grey one field. Leave it unknown; the run still validates it.
        if sum(os.path.getsize(path) for _, path in sources) > 5_000_000:
            return {'plate_count': None, 'has_plate_id': None}
        plates = set()
        has_plate = False
        for logical_index, path in sources:
            with open(path, newline='', encoding='utf-8-sig') as handle:
                sample = handle.read(4096)
                handle.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=',\t;')
                except csv.Error:
                    dialect = csv.excel
                reader = csv.DictReader(handle, dialect=dialect)
                names = reader.fieldnames or []
                plate_key = next((name for name in names
                                  if str(name).casefold() in {
                                      'plateid', 'plate', 'plate_name'}), None)
                if plate_key is None:
                    # score_data[i] and count_data[i] describe the same plate;
                    # their absent IDs therefore share one fallback identity.
                    plates.add(('source', logical_index))
                    continue
                has_plate = True
                for row in reader:
                    value = str(row.get(plate_key, '')).strip()
                    if value:
                        plates.add(('value', value))
                    if len(plates) > 1:
                        break
            if len(plates) > 1:
                break
        return {'plate_count': len(plates) or None,
                'has_plate_id': has_plate}

    def _refresh_setting_dependencies(self) -> None:
        # THE ROWS FIRST, THEN WHICH OF THE ONES LEFT ON SCREEN ARE GREYED.
        # This is the hook `apply_settings_dict` calls when it has finished
        # pouring a settings file in, and a file that sets `cell_channel` has
        # to bring the cell settings back on screen with it. A reason written
        # beside a control on a hidden row is a reason nobody can read.
        self.refresh_object_visibility()
        dependencies = self._rules_for_this_panel()
        if not dependencies:
            return
        current = self._current_dependency_settings()
        # Only scanned when a rule on this panel can actually read it. It
        # opens the loaded CSVs, and doing that on every combo change of a
        # panel with no data-dependent rule is a stall for nothing.
        if any('paired_data' in rule.get('sources', ())
               or 'score_data' in rule.get('sources', ())
               or 'count_data' in rule.get('sources', ())
               for rule in dependencies.values()):
            self._data_context = self._plate_context(
                self._loaded_table_paths(current))
        for key, rule in dependencies.items():
            control = self._widgets[key]
            try:
                enabled = bool(rule['predicate'](current, self._data_context))
            except Exception:
                enabled = True
            control.setEnabled(enabled)
            if enabled:
                _clear_greyed_note(control)
            else:
                reason = str(rule['reason'](current, self._data_context))
                _apply_greyed_note(control, reason)
                self._show_the_value_it_will_have(key, current)

    #: Settings whose value another setting DECIDES, and the translator that
    #: decides it. A greyed control here shows the value the run will use.
    _DECIDED_BY_ANOTHER = ("analysis_mode", "agg_type", "regression_type")

    def _show_the_value_it_will_have(self, key, current) -> None:
        """Put the resolved value into a control the run overrides anyway.

        A GREYED CONTROL SHOWING THE WRONG VALUE IS WORSE THAN A GREYED ONE.
        Asked 2026-08-20: "if nonparametric is chosen should guide permutation
        be in analysis mode". It should, and it was not:
        `_resolve_regression_analysis_choices` rewrites `analysis_mode` from
        `inference` AT RUN TIME, so the panel showed 'regression' while the
        run used 'guide_permutation' -- and the greyed note beside it said as
        much in words. Words next to a contradicting value is the worst of
        the three states.

        Only the settings another setting genuinely decides, and only through
        the SAME translator the run uses, so the panel cannot come to a
        different answer than the fit.
        """
        if key not in self._DECIDED_BY_ANOTHER:
            return
        # NOT WHILE A SETTINGS FILE IS BEING POURED IN. `apply_settings_dict`
        # sets one widget at a time, so `inference` may still hold the old
        # value when `analysis_mode` arrives -- and forcing then would
        # overwrite the file's value from an inference that is about to
        # change. Caught by loading a file carrying inference='auto' and
        # analysis_mode='guide_permutation': the mode was clobbered to
        # 'regression' before 'auto' had landed. The refresh that runs once
        # the whole dict is applied does the right thing.
        if getattr(self, "_applying_settings", False):
            return
        try:
            from spacr.settings import _resolve_regression_analysis_choices

            resolved = dict(current)
            _resolve_regression_analysis_choices(resolved)
        except Exception:                                    # noqa: BLE001
            return
        value = resolved.get(key)
        if value is None or value == current.get(key):
            return
        setter = getattr(self, "set_value_for_key", None)
        if callable(setter):
            setter(key, value)

    # ------------------------------------------------------------------
    # A setting is visible when its object is in the run
    # ------------------------------------------------------------------

    def _object_visibility_keys(self) -> set:
        """The few settings the visibility rule reads.

        NOT ``_current_dependency_settings``, which walks and coerces EVERY
        widget on the screen: this runs on each keystroke in a channel box,
        and Mask has three hundred and fifty settings of which the rule reads
        about thirty. The same reason ``_input_csv_paths`` reads three keys
        rather than the panel.
        """
        from ...organelle_types import NUMBER_OF_ORGANELLES

        wanted = {NUMBER_OF_ORGANELLES}
        for key in self._widgets:
            role = object_of_setting(key)
            if role is None:
                continue
            wanted.update(object_switch_keys(role))
            # The type narrows a slot, the diameter decides which way a
            # size-split type narrows it, and the morphology is the answer
            # for a slot left on 'custom'.
            wanted.update(f"{role}_{name}"
                          for name in ("type", "diameter", "morphology"))
        return wanted

    def _object_visibility_settings(self) -> Dict[str, Any]:
        """Current values of the settings the visibility rule reads.

        A key with no control on this panel is read from ``_defaults``, which
        is where its value lives and where the run will read it from too.
        """
        current: Dict[str, Any] = {}
        for key in self._object_visibility_keys():
            widget = self._widgets.get(key)
            if widget is None:
                current[key] = self._defaults.get(key)
                continue
            try:
                current[key] = self._coerce_to_expected_type(
                    key, self._read_widget(widget))
            except Exception:                                # noqa: BLE001
                current[key] = self._defaults.get(key)
        return current

    def keys_whose_object_the_run_lacks(self) -> set:
        """Return setting keys excluded by the current object configuration.

        :returns: Hidden setting keys, or an empty set if visibility cannot
            be determined.
        """
        try:
            return set(keys_hidden_by_their_object(
                self._widgets, self._object_visibility_settings()))
        except Exception:                                    # noqa: BLE001
            LOGGER.debug("could not decide which objects are in the run",
                         exc_info=True)
            return set()

    def remember_section_rows(self, section, keys, has_children: bool) -> None:
        """Record the settings and nesting state associated with a section.

        :param section: Section-heading widget.
        :param keys: Settings declared directly in the section, in order.
        :param has_children: Whether the section contains nested headings.
        """
        declared = getattr(self, "_section_rows", None)
        if declared is None:
            declared = self._section_rows = {}
        declared[id(section)] = (section, tuple(keys), bool(has_children))
        self._slot_heading_cache = None

    def refresh_object_visibility(self) -> None:
        """Show only the rows whose object this run actually has.

        Idempotent, and it decides EVERY gated row every time rather than
        toggling the ones that changed -- so a row put back on screen by
        something else answering a different question (the settings search
        releasing its filter shows every row it indexed) is hidden again on
        the next call instead of drifting.

        Public because the screen has to be able to ask for it: it is the
        screen that lays the rows out, and the screen that hands row
        visibility back after a filter.
        """
        # NOT WHILE A SETTINGS FILE IS BEING POURED IN. `apply_settings_dict`
        # sets one widget at a time, so a channel may already hold its new
        # value while the type beside it still holds the old one; hiding rows
        # against that half-applied panel would show a slot narrowed to the
        # wrong morphology and then narrow it again. The bulk apply calls
        # `_refresh_setting_dependencies` when it is finished, which is where
        # this runs instead.
        if getattr(self, "_applying_settings", False):
            return
        try:
            current = self._object_visibility_settings()
            hidden = keys_hidden_by_their_object(self._widgets, current)
            # BEFORE THE ROWS MOVE, so the guard installed below judges each
            # row against the answer this pass is applying rather than the
            # last one -- otherwise showing a row whose channel was just
            # typed would look, to the guard, like something else putting a
            # hidden row back.
            self._hidden_by_the_run = set(hidden)
            # BEFORE THE ROWS MOVE, for the other reason too: a row the screen
            # left unbuilt because this rule hid it has to exist before the
            # rule can show it, or `_set_row_visible` would put a bare field
            # on screen in no layout at all.
            lay_out = getattr(self, "rows_are_laid_out_by", None)
            if lay_out is not None:
                try:
                    lay_out(hidden)
                except Exception:                            # noqa: BLE001
                    LOGGER.debug("could not lay out the rows that are back",
                                 exc_info=True)
            for key in list(self._widgets):
                self._set_row_visible(key, key not in hidden)
            self._guard_hidden_rows(hidden)
            self._hide_the_headings_of_slots_the_run_lacks(current)
        except Exception:                                    # noqa: BLE001
            LOGGER.debug("could not decide which objects are in the run",
                         exc_info=True)

    def keys_hidden_by_the_run(self) -> List[str]:
        """Return settings hidden by the latest object-visibility pass.

        :returns: Hidden keys in no guaranteed order. The result is empty
            before visibility is evaluated or when the model has no rows.
        """
        return list(getattr(self, "_hidden_by_the_run", ()) or ())

    def _slot_headings(self) -> Dict[int, Tuple[Any, Tuple[str, ...]]]:
        """Each leaf heading on the panel and the settings it owns.

        Computed once: which settings a heading holds is decided when the
        panel is built and does not change afterwards, and this runs on every
        keystroke in a channel box.

        LEAF HEADINGS ONLY -- one with sub-headings inside it is answered by
        them. ``id(section) -> (section, keys)``, because a ``Section`` is
        not hashable in a way that survives Qt taking it apart.
        """
        # AN EMPTY ANSWER IS NOT CACHED. The first pass is scheduled from
        # `build_sections`, and on a model built for its values rather than
        # for a screen there are no sections to find at all -- caching that
        # would answer "no headings" for the life of the panel.
        cached = getattr(self, "_slot_heading_cache", None)
        if cached:
            return cached
        cache: Dict[int, Tuple[Any, Tuple[str, ...]]] = {}
        if self._parent is None:
            return cache
        # WHAT THE PANEL DECLARED, when it declared anything. The walk below
        # recovers the same two facts from the rendered form, at the cost of a
        # `findChildren` per heading; a panel that said what it was building
        # has already answered. See :meth:`remember_section_rows`.
        declared = getattr(self, "_section_rows", None)
        if declared:
            for ident, (section, keys, has_children) in declared.items():
                if has_children or not keys:
                    continue
                cache[ident] = (section, tuple(keys))
            if cache:
                self._slot_heading_cache = cache
            return cache
        try:
            from ..widgets.section import Section

            by_widget = {id(widget): key
                         for key, widget in self._widgets.items()}
            for section in self._parent.findChildren(Section):
                if [child for child in section.findChildren(Section)
                        if child is not section]:
                    continue
                form = getattr(section, "_form", None)
                if not isinstance(form, QFormLayout):
                    continue
                keys = []
                for index in range(form.rowCount()):
                    item = form.itemAt(index, QFormLayout.FieldRole)
                    field = item.widget() if item is not None else None
                    key = by_widget.get(id(field)) if field is not None \
                        else None
                    if key is not None:
                        keys.append(key)
                if keys:
                    cache[id(section)] = (section, tuple(keys))
        except Exception:                                    # noqa: BLE001
            LOGGER.debug("could not map the panel's headings", exc_info=True)
            return {}
        if cache:
            self._slot_heading_cache = cache
        return cache

    def _hide_the_headings_of_slots_the_run_lacks(
            self, settings: Dict[str, Any]) -> None:
        """A slot the count does not reach has no heading either.

        A HEADING WITH EVERY ROW HIDDEN IS A SMALLER WALL, BUT IT IS STILL A
        WALL, and the panel now builds a heading for every slot that can be
        named: without this, opening Mask meant scrolling past ORGANELLE 5
        through ORGANELLE 26 three times over to reach anything.

        ONLY THE SLOT HEADINGS, and only the ones this method hid. A heading
        is left alone unless every setting under it belongs to an organelle
        slot the run does not have -- so nothing here has an opinion about a
        heading hidden for its maturity, by a dimension switch, or by the
        settings search, and a heading this did not hide is never shown by
        it. That is what keeps one card from being decided in two places.

        :param settings: the values the object rule just read, so the count
            is not walked out of the panel a second time on every keystroke.
        """
        from ..preferences import maturity_is_visible
        from ...organelle_types import active_organelle_roles

        headings = self._slot_headings()
        if not headings:
            return
        active = set(active_organelle_roles(settings))
        emptied = self._headings_of_absent_slots
        for ident, (section, keys) in headings.items():
            roles = {object_of_setting(key) for key in keys}
            gone = bool(roles) and all(
                role is not None and role not in CHANNELLED_OBJECTS
                and role not in active for role in roles)
            try:
                if gone:
                    # EVERY PASS, not only the first: the settings search
                    # puts a heading back whenever its filter is released,
                    # and a method that only hid one it had not hidden
                    # before would hide it once and never again.
                    if not section.isHidden():
                        emptied[ident] = section
                        section.setVisible(False)
                        section.installEventFilter(self._object_row_guard)
                elif ident in emptied:
                    del emptied[ident]
                    # ONLY WHAT MATURITY WOULD ALSO SHOW. A heading this hid
                    # may since have been hidden again as Alpha or Beta, and
                    # putting a slot back must not overrule Preferences.
                    if maturity_is_visible(section.maturity()):
                        section.setVisible(True)
            except RuntimeError:
                # The section went away with the screen that owned it.
                emptied.pop(ident, None)

    def _guard_hidden_rows(self, hidden) -> None:
        """Keep settings for inactive object roles hidden after UI updates.

        Search filters, recipes, and section expansion can make a previously
        hidden row visible. Each affected row therefore watches
        ``ShowToParent`` events and schedules another visibility pass whenever
        an external update reveals it, including within collapsed sections.
        """
        guard = getattr(self, "_object_row_guard", None)
        if guard is None or self._parent is None:
            return
        guarded = self._guarded_rows
        for key in hidden:
            widget = self._widgets.get(key)
            if widget is None or id(widget) in guarded:
                continue
            guarded[id(widget)] = key
            widget.installEventFilter(guard)

    def _shown_against_the_rule(self, widget: QWidget) -> None:
        """Something outside put a hidden row or heading back; ask for a pass.

        DEFERRED, not undone here: this runs while Qt is delivering the show
        event, and hiding the widget again inside its own event would leave
        whatever is walking a form mid-walk. One pass is queued however many
        rows were shown, because the pass decides every gated row anyway.
        """
        key = self._guarded_rows.get(id(widget))
        contested = (
            (key is not None and key in getattr(self, "_hidden_by_the_run", ()))
            or id(widget) in getattr(self, "_headings_of_absent_slots", {}))
        if not contested:
            return
        if self._object_rule_pass_queued or self._parent is None:
            return
        self._object_rule_pass_queued = True
        QTimer.singleShot(0, self._parent, self._reassert_object_visibility)

    def _reassert_object_visibility(self) -> None:
        self._object_rule_pass_queued = False
        self.refresh_object_visibility()

    def _set_row_visible(self, key: str, visible: bool) -> None:
        """Show or hide the whole ROW a setting sits on.

        THE ROW, NOT THE FIELD. The screen builds the label and puts the pair
        into a ``QFormLayout`` after ``build_sections`` has handed the rows
        back, and it keeps the label side inside a wrapper it does not hand
        back -- so hiding the field alone strands its name on an empty row.
        ``QFormLayout.setRowVisible`` reaches both halves, and it is reached
        through the same helper the settings search and the 3D/Time switches
        hide rows with, so a row is hidden one way whatever the reason for
        hiding it.

        The widget the FORM knows is not always the field: a handful of
        settings sit in a little holder with a button beside them, and it is
        the holder that is in the row. The walk goes up until a form
        recognises the node it is being handed.
        """
        widget = self._widgets.get(key)
        if widget is None:
            return
        from ..settings_search import _set_row_visible as set_row

        node = widget
        # Three steps is the deepest the panel nests a field: field, the
        # button holder, the section body that owns the form.
        for _ in range(3):
            parent = node.parentWidget()
            if parent is None:
                break
            layout = parent.layout()
            if isinstance(layout, QFormLayout):
                row, _role = layout.getWidgetPosition(node)
                if row >= 0:
                    set_row(parent, node, visible)
                    return
            node = parent
        # NOT UNTIL THE SCREEN HAS TAKEN THE WIDGET. `SettingsWidgets` is
        # built with no parent by everything that wants the values rather
        # than a form, and a parentless widget shown here would not be a row
        # coming back -- it would be a window of its own, opened and painted
        # on the next turn of the event loop, mid-construction and long after
        # the panel that made it was finished with.
        if widget.parentWidget() is None:
            return
        # There is a widget but no row yet: the screen builds the label and
        # the form after `build_sections` hands the rows back. Hide the field
        # so the panel is not a frame late; the scheduled pass takes the
        # label once the row has one.
        widget.setVisible(visible)
        label = getattr(widget, "_spacr_setting_label", None)
        if label is not None:
            label.setVisible(visible)

    def _connect_object_visibility_signals(self) -> None:
        """Follow the switches, the count and the types as they are changed.

        Bound method, not a lambda: see INVARIANTS 4 for what a closure
        connected to a Qt signal costs.
        """
        for key in sorted(self._object_visibility_keys()):
            widget = self._widgets.get(key)
            if widget is not None:
                _connect_value_changed(widget, self._on_object_switch_changed)

    def _on_object_switch_changed(self, *_args) -> None:
        self.refresh_object_visibility()

    def _read_widget(self, w: QWidget) -> Any:
        if isinstance(w, QCheckBox):
            return bool(w.isChecked())
        if isinstance(w, QSpinBox):
            return int(w.value())
        if isinstance(w, QDoubleSpinBox):
            if str(w.specialValueText() or "") == AUTO_TEXT:
                return _read_auto_or_number(w)
            return float(w.value())
        if isinstance(w, QComboBox):
            idx = w.currentIndex()
            # EVERY item is added with userData=opt, including the Python None
            # option (`addItem("None" if opt is None else str(opt),
            # userData=opt)`). So currentData() returning None means the chosen
            # option IS None -- not that the item carries no data. The old
            # fallback to currentText() therefore handed back the STRING
            # 'None', which is how every Qt run shipped strict_errors='None'
            # and turned strict error handling silently ON, since
            # errors.strict_errors() saw a non-None value and took
            # bool('None') == True. cov_type and 'transform' reached
            # statsmodels the same way.
            #
            # currentText() is still right for an EDITABLE combo showing
            # something the user typed that is not in the list -- detected by
            # the displayed text not matching the current item's text.
            if idx >= 0 and w.itemText(idx) == w.currentText():
                return w.itemData(idx)
            return w.currentText()
        if isinstance(
            w,
            (
                _AlphabetSelect, _ListEditor, _ListEdit, BarcodeRegexWidget,
                RowExclusionEditor, ExternalMaskInputWidget,
                ChannelMappingWidget, ClassEditorWidget, DatabaseSetWidget,
                FilePathListWidget,
                PairedFileTableWidget, _CsvColumnField,
                _RegressionBackendField,
            ),
        ):
            return w.get_value()
        if isinstance(w, _ScalarEdit):
            return w.get_value()
        if isinstance(w, QLineEdit):
            return w.text() or None
        return None


#: Widget types that are an EDITOR for a setting rather than its name.
#: A QCheckBox is deliberately absent: it carries its own text, so it is its
#: own label and hovering it is hovering the name.
_EDITOR_TYPES = (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
                 QPlainTextEdit, QTextEdit)

#: Marks a field tooltip that is a DISABLED-REASON rather than help.
#: "This control does nothing because ..." explains that control, so the
#: control is the right place for it and :func:`retarget_field_tooltips`
#: leaves it alone. Set it where such a note is written.
DISABLED_REASON_TOOLTIP = "spacrDisabledReasonTooltip"


def _owning_layout(root: QLayout, field: QWidget):
    """The innermost layout that holds ``field`` directly, and its index."""
    stack = [root]
    while stack:
        layout = stack.pop()
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item is None:
                continue
            if item.widget() is field:
                return layout, index
            child = item.layout()
            if child is not None:
                stack.append(child)
    return None, -1


def _sibling_label_for(field: QWidget) -> Optional[QWidget]:
    """The QLabel a LAYOUT says names this field.

    Asked of the layout rather than of the geometry, because the screens run
    :func:`retarget_field_tooltips` at the end of ``__init__`` -- before the
    widget has ever been shown, laid out or resized. Every child is still at
    (0, 0) there, so a matcher that compares x and y answers "the first label
    in this parent" for EVERY field, and the pass then moves one setting's
    help onto that label and DELETES the rest. It measurably did: 80 settings
    across the Qt screens had no help left anywhere.

    A layout knows the pairing with no geometry at all. Three shapes cover
    what the hand-built screens use, and each is the same claim -- the name
    sits to the LEFT of the editor:

    * ``QFormLayout`` -- ``labelForField`` is the pairing, exactly;
    * ``QGridLayout`` -- the nearest label in a lower column of the same row;
    * a horizontal box -- the nearest label before it in the row.

    Anything else returns None, which leaves the field's tooltip alone. A
    setting whose help is on the field is a smaller defect than a setting
    with no help at all.
    """
    parent = field.parentWidget()
    root = parent.layout() if parent is not None else None
    if root is None:
        return None
    layout, index = _owning_layout(root, field)
    if layout is None:
        return None

    def _named(widget) -> Optional[QWidget]:
        if not (isinstance(widget, QLabel) and widget.text().strip()):
            return None
        # A label with a pointing hand is this repository's convention for
        # "this text is clickable" -- AiToggleLabel, _ClearFiguresLabel, the
        # console's copy glyph. Such a label is a CONTROL sharing the row,
        # not the name of the editor beside it, and its own tooltip explains
        # itself rather than its neighbour.
        if widget.cursor().shape() == Qt.PointingHandCursor:
            return None
        return widget

    if isinstance(layout, QFormLayout):
        return _named(layout.labelForField(field))

    if isinstance(layout, QGridLayout):
        row, column, _rows, _cols = layout.getItemPosition(index)
        for candidate in range(column - 1, -1, -1):
            item = layout.itemAtPosition(row, candidate)
            found = _named(item.widget()) if item is not None else None
            if found is not None:
                return found
        return None

    if (isinstance(layout, QBoxLayout)
            and layout.direction() == QBoxLayout.LeftToRight
            and index > 0):
        # The item IMMEDIATELY before it, and nothing further back. A row of
        # several controls has labels belonging to each of them, and scanning
        # backwards past an intervening control pairs an editor with the
        # previous setting's name -- or, in the preview panels, with the
        # "drop a folder here" placeholder that happens to sit first in the
        # row.
        item = layout.itemAt(index - 1)
        return _named(item.widget()) if item is not None else None
    return None


def retarget_field_tooltips(root: QWidget) -> int:
    """Move editor tooltips to the labels that identify their settings.

    Parameters
    ----------
    root : QWidget
        Constructed screen or dialog to inspect recursively.

    Returns
    -------
    int
        Number of tooltips moved.

    Notes
    -----
    Tooltips stay on editors that have no sibling label, whose label already
    has different help, or that carry :data:`DISABLED_REASON_TOOLTIP`.
    """
    moved = 0
    for field in root.findChildren(QWidget):
        if not isinstance(field, _EDITOR_TYPES):
            continue
        tip = field.toolTip()
        if not tip:
            continue
        if field.property(DISABLED_REASON_TOOLTIP):
            continue
        label = _sibling_label_for(field)
        if label is None:
            continue
        existing = label.toolTip()
        if existing and existing != tip:
            # Two settings cannot share one name, so this pairing is wrong.
            # Leave the help where it is: clearing it here is how 80 settings
            # ended up with no help anywhere, which is a worse defect than
            # the one this pass exists to fix.
            continue
        if not existing:
            label.setToolTip(tip)
            label.setToolTipDuration(-1)
            label.setCursor(Qt.WhatsThisCursor)
        field.setToolTip("")
        moved += 1
    return moved
