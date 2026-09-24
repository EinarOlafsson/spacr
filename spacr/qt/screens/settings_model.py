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
from collections.abc import MutableMapping
import csv
from contextlib import contextmanager
from functools import partial
from html import escape
import logging
import os
import sys
import textwrap
import weakref
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Sequence,
                    Tuple)

from PySide6.QtCore import (QEvent, QObject, QPoint, QRect, QSize, Qt,
                            QThread, QTimer, Signal)
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

from .. import timing as _timing
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
from ...organelle_types import (ALL_ORGANELLE_ROLES,
                                MAX_ORGANELLES as _MAX_ORGANELLES,
                                NUMBER_OF_ORGANELLES,
                                organelle_number, organelle_slot_label)
from ...regression_spec import NO_P_VALUE_TYPES
from ...schema import KEY_SEPARATOR


LOGGER = logging.getLogger(__name__)



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
        s = set_default_settings_preprocess_generate_masks(settings={})
        for key in timelapse_and_motility_keys():
            s.pop(key, None)
        return s
    if app_key == "timelapse":
        s = get_timelapse_settings(settings={})
        s.pop("motility_analysis", None)
        s["timelapse"] = True
        return s
    if app_key == "motility":
        s = get_automated_motility_assay_default_settings(settings={})
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
    if app_key == 'host_pathogen':
        from spacr.host_pathogen import default_settings
        return default_settings()
    if app_key == "analyze_plaques":
        return get_analyze_plaque_settings(settings={})
    if app_key in ("annotate", "make_masks"):
        return {"src": "path to images"}
    return {"src": "path"}


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
    "train_cellpose": {"model_type", "from_scratch", "Signal_to_noise", "background",
                       "remove_background", "diameter", "resize", "width_height",
                       "target_size", "augment", "verbose"},
    "mask": {"pathogen_model"},
    "timelapse": {"timelapse"},
    "classify": {
        "png_type", "crop_source", "file_metadata", "file_type",
        "path_string", "extract_channels", "coordinate_columns",
        "class_metadata", "annotation_column",
        "class_folder_names",
    },
    "classify_merged": {
        "png_type", "crop_source", "file_metadata", "file_type",
        "path_string", "extract_channels", "coordinate_columns",
        "class_metadata", "annotation_column",
        "class_folder_names",
    },
    "umap": {"gpu", "crop_source"},
    "regression": {
        "regression_panel_manifest",
        "regression_qc", "guide_permutation_plot",
        "log_x", "log_y", "x_lim", "y_lims", "split_axis_lims",
        "strict_errors", "max_failure_rate", "on_error",
        "on_error_attempts", "on_error_backoff", "random_seed", "verbose",
        "analysis_excluded_wells",
        "cell_area_outlier_mads", "nucleus_area_outlier_mads",
        "cell_intensity_outlier_mads", "nucleus_intensity_outlier_mads",
    },
}

_APP_HIDDEN_CATEGORIES: Dict[str, set] = {
    "classify": {"Cellpose"},
    "mask": {"Timelapse", "Motility (beta)", "Motility Advanced (beta)"},
    "timelapse": {"Motility (beta)", "Motility Advanced (beta)"},
}


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
    tail = organelle_role_of(text.rpartition("_")[2])
    if tail is not None and text.startswith("remove_background_"):
        return tail
    for obj in CHANNELLED_OBJECTS:
        if text.startswith(f"{obj}_") or text.endswith(f"_{obj}"):
            return obj
    return None


def organelle_morphology_now(role: str,
                             settings: Dict[str, Any]) -> Optional[str]:
    """Resolve the morphology currently applicable to an organelle slot.

    The collected ``<role>_morphology`` is authoritative. Selecting a type
    writes its recommendation into that control, while a later explicit
    advanced choice must win over the preset and over its display inference.
    Only a sparse mapping with no morphology falls back to resolving
    ``<role>_type`` and ``<role>_diameter`` directly.

    :param role: Prefix for the organelle-slot settings, such as ``organelle``
        or ``organelleb``.
    :param settings: Current values keyed by setting name.
    :returns: ``"spots"``, ``"network"``, ``"irregular"``, ``"ring"``, or
        ``None`` when neither resolution path supplies a supported morphology.
    """
    own = settings.get(f"{role}_morphology")
    if own in _MORPHOLOGY_SETTINGS:
        return own

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
    return None


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
        if role == "cell":
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
    """Report whether a settings section contains visible content.

    A section whose setting rows and nested sections are all hidden should not
    leave an empty heading in the panel. This predicate reports whether
    content remains after row-level visibility rules have been applied; it
    does not change widget visibility itself.

    :param section: A :class:`spacr.qt.widgets.section.Section`.
    :returns: ``False`` only when a section owns rows or nested sections and
        all of them are hidden. Sections without setting rows remain visible.
    """
    from ..widgets.section import Section, _sections_below

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
    children = [child for child in _sections_below(section)
                if child is not section and isinstance(child, Section)]
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

#: The SAME choice as :data:`_CROP_SOURCE_OPTIONS`, in the spelling training
#: stores it under. LOAD IMAGES and STREAM IMAGES are the only two names for
#: these two things anywhere in spaCR, and the sentences here are word-for-word
#: the ones above so that a user reading the annotation panel and the training
#: panel can see they are being asked one question.
#:
#: It had to exist as a SEPARATE table because the stored values differ and
#: must not change: the viewers persist 'png'/'merged' and training persists
#: 'load_images'/'stream_images' (`spacr.settings._canonical_image_source`
#: rewrites every choice into that pair, and `settings.py` then copies
#: `image_source` onto `crop_source`). Offering 'png' here would write a
#: value that normaliser does not produce; offering 'load_images' in the
#: viewer table would move the meaning of settings files already on disk.
#: Both spellings resolve through `crop_source.CROP_SOURCE_ALIASES`.
#:
#: THE TRAINING PANEL HAD NO CONTROL FOR THIS AT ALL. `image_source` is not
#: in `_APP_HIDDEN_KEYS`, so the Classify screens laid out a row for it -- as
#: a FREE-TEXT box, because a key absent from this table gets whatever widget
#: its default's type implies. That is the one panel the whole migration onto
#: these two names was for, and it was the one panel where a user could type
#: a fourth spelling. A typo, or a remembered 'pre_generated', reached
#: `crop_source.CROP_SOURCE_ALIASES` and refused the run at the door.
#:
#: 'generate' is deliberately NOT offered. It is an ACTION -- it WRITES a
#: crop set -- rather than one of the two sources, which is why it is not one
#: of the two names; `crop_source.CROP_SOURCE_OPTIONS` still carries it for
#: readers that need the third entry, and the training panel's own
#: `generate_training_dataset` switch is where a user asks for the write.
#:
#: It is still ACCEPTED, and it SELECTS rather than being shown. A settings
#: file carrying it opens on LOAD IMAGES, because that is what
#: `settings._canonical_image_source` -- and therefore
#: `settings.deep_spacr_defaults`, which rewrites `image_source` through it
#: before any run -- turns it into. The panel showing the word the file
#: holds while the run reads a different source is the disagreement
#: `_image_source_the_panel_offers` exists to prevent, so 'generate' is
#: treated here exactly as every other retired spelling is.
_IMAGE_SOURCE_OPTIONS = [
    ("load_images", "load images — crops already in data/"),
    ("stream_images", "stream images — cut from merged/"),
]


def _image_source_the_panel_offers(value) -> str:
    """Which of the two offered modes a stored ``image_source`` selects.

    A combo whose stored value matches no item keeps that value as a new
    first item (see ``_widget_for``), which is right for a free alphabet and
    wrong for this one: a settings CSV carrying ``'on_demand'`` put a THIRD
    entry, spelled in a retired vocabulary, in front of a user who is being
    asked a two-way question. That is the two-name rule's failure in the one
    panel the whole migration was for.

    RESOLVED, NOT REFUSED, and resolved through the table the pipeline reads
    -- `spacr.settings._canonical_image_source` -- so the panel and the run
    cannot disagree about what an old file means. Every retired spelling
    still loads; it just selects the mode it has always meant.

    ``'auto'`` comes back from that function as itself, because
    `crops.resolve_crop_source` still computes "what is available here". It
    is NOT an answer a user is offered -- it is an action, not a source -- so here it
    selects LOAD IMAGES -- rule A, the default is a named mode -- and rule
    B's fallback is what keeps a project with no ``data/`` drawing: LOAD
    IMAGES with nothing to load streams instead and says so. Anything else
    unrecognised lands on LOAD IMAGES for the same reason the normaliser
    does.
    """
    offered = [stored for stored, _label in _IMAGE_SOURCE_OPTIONS]
    try:
        from spacr.settings import _canonical_image_source
        resolved = _canonical_image_source(value)
    except Exception:                                            # noqa: BLE001
        resolved = str(value or "").strip().lower()
    return resolved if resolved in offered else offered[0]


_APP_COMBO_OPTIONS: Dict[str, Dict[str, List[Any]]] = {
    "umap": {
        "reduction_method": ["umap", "tsne", "pca", "isomap", "spectral"],
        "metric": ["euclidean"],
        "pca_svd_solver": [
            "auto", "full", "covariance_eigh", "arpack", "randomized",
        ],
        "isomap_path_method": ["auto", "FW", "D"],
        "spectral_affinity": ["nearest_neighbors", "rbf"],
        "clustering": ["dbscan", "kmeans"],
        "crop_source": _CROP_SOURCE_OPTIONS,
        "batch_correction": _BATCH_CORRECTION_OPTIONS,
        "batch_missing_control": _BATCH_MISSING_CONTROL_OPTIONS,
    },
    "annotate": {
        "crop_source": _CROP_SOURCE_OPTIONS,
    },
    "ml_analyze": {
        "batch_correction": _BATCH_CORRECTION_OPTIONS,
        "batch_missing_control": _BATCH_MISSING_CONTROL_OPTIONS,
    },
    "regression": {
        "batch_correction": _BATCH_CORRECTION_OPTIONS,
        "batch_missing_control": _BATCH_MISSING_CONTROL_OPTIONS,
        "independent_variable_layout": ["auto", "long", "wide"],
        "model_data_layout": ["long", "wide"],
        "regression_type": ["ols"],
        "multiple_testing_method": ["fdr_bh"],
        "inference": [
            ("auto",
             "auto — take the simultaneous fit only if the design supports it"),
            ("parametric",
             "parametric — fit every term at once; needs more wells than terms"),
            ("nonparametric",
             "nonparametric — test each term on its own by permutation; "
             "valid at any width"),
        ],
        "analysis_mode": [
            ("regression",
             "regression — fit every guide at once in the chosen model"),
            ("guide_permutation",
             "guide permutation — test each guide on its own, wells "
             "reshuffled within each plate"),
        ],
        "analysis_unit": ["well", "cell"],
        "agg_type": ["mean", "median", "quantile", None],
        "transform": [None, "log", "sqrt", "square", "beta"],
        "cov_type": [None, "HC0", "HC1", "HC2", "HC3"],
        "threshold_method": ["std", "var"],
        "p_threshold_kind": ["adjusted", "raw"],
    },
    "classify": {
        "evaluation_calibration": ["temperature", "none"],
        "image_source": _IMAGE_SOURCE_OPTIONS,
    },
    "classify_merged": {
        "evaluation_calibration": ["temperature", "none"],
        "image_source": _IMAGE_SOURCE_OPTIONS,
        "classifier_family": [
            ("cv", "Computer Vision (Torch)"),
            ("ml", "Tabular Machine Learning"),
        ],
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

_REGRESSION_TOOLTIP_OVERRIDES = {
    "src": (
        "Output root for regression results. Leave blank to use the directory "
        "containing the first count table. An existing directory is used "
        "directly; if only its final component is missing, spaCR creates that "
        "directory. If the configured path is a file, its parent is missing, "
        "or the directory cannot be created, spaCR reports the problem and "
        "uses the automatic location. Home-directory shortcuts and relative "
        "path components are resolved before validation. Each run is stored "
        "below this root in results/<analysis> or, when that directory is "
        "occupied, results/<analysis>_<n>. Default: blank (automatic)."
    ),
}

_TRAIN_CELLPOSE_TOOLTIPS = {
    "src": "Folder containing training microscopy images. Masks default to the masks subfolder. Legacy project/train/images and project/train/masks layouts also remain supported.",
    "mask_src": "Optional separate folder of integer object-label masks (background 0). Leave blank to use masks inside the image folder. Match image basenames, optionally with a _masks suffix. Missing or ambiguous pairs stop training.",
    "test_src": "Optional validation image folder, separate from training images. Leave blank to train without validation losses. Split by well or experiment to avoid leakage between related fields.",
    "test_mask_src": "Optional validation label-mask folder. Leave blank to use the masks subfolder inside the validation image folder.",
    "save_path": "Checkpoint output folder. Cellpose writes weights into its models subfolder. Leave blank for <image source>/models/cellpose_model. Use trained model reads this location too.",
    "model_name": "Name for the newly trained checkpoint. The epoch count is appended. Use a distinct name for each experiment to preserve earlier runs.",
    "learning_rate": "AdamW learning rate for Cellpose 4 fine-tuning. Default 0.00001 (1e-5), the Cellpose-SAM recommendation. Reduce it if loss becomes unstable.",
    "weight_decay": "AdamW weight decay. Default 0.1, as recommended for Cellpose-SAM fine-tuning.",
    "batch_size": "Training minibatch size, not a dataset limit. Default 1 reduces GPU memory use for Cellpose-SAM. Raise it only when memory permits.",
    "n_epochs": "Training epochs. Default 100 follows Cellpose-SAM fine-tuning guidance. Monitor training and validation losses before extending a run.",
    "channels": "Zero-based image channels to train on (one to three), or leave blank to preserve all channels in images with at most three. Channels are never averaged. Larger images require an explicit selection.",
    "channel_axis": "Image channel axis: 0 for channel-first, -1 for channel-last, or blank to infer it from the mask dimensions. Ambiguous images require an explicit axis. Training supports 2-D fields with optional channels, not Z stacks.",
    "normalize": "Apply Cellpose's per-channel percentile normalization once during training. Disable only for intentionally pre-normalized data. Default True.",
    "percentiles": "Lower and upper intensity percentiles used when normalization is enabled. Default [1, 99], matching Cellpose. Native image geometry and separate channels are preserved.",
    "min_train_masks": "Minimum labeled objects required per training image. Cellpose excludes fields below this count. Default 5. Lower it for deliberately sparse training fields.",
    "max_train_images": "Optional limit on paired training images loaded into RAM, in filename order. Blank or a nonpositive value uses every pair. This does not change the minibatch size.",
    "nimg_per_epoch": "Optional number of images sampled per training epoch. Blank uses every training image. This changes sampling, not the number of files loaded into RAM.",
    "nimg_test_per_epoch": "Optional number of validation images sampled per evaluation epoch. Blank uses all validation images. Requires a validation image source.",
    "scale_range": "Range of Cellpose's random training scale augmentation, from 0 to 2. Default 0.5. Cellpose also applies its native rotation, flip and crop augmentation; no eight-fold duplicate dataset is created.",
    "save_every": "Checkpoint interval in epochs. Default 100. Cellpose always saves the final model even when the run is shorter than this interval.",
    "save_each": "Keep separate epoch checkpoints instead of replacing the periodic checkpoint. Default False. Enable to compare intermediate models; it consumes additional disk space.",
}

_APP_TOOLTIP_OVERRIDES = {
    "measure": {
        "psf_source": "gaussian constructs an explicitly sampled Gaussian approximation from the supplied FWHM and image sampling. measured captures a calibrated TIFF or NPY kernel. The same kernel applies independently to every selected measurement intensity channel, so its calibration must suit all selected channels. Neither choice estimates microscope optics.",
        "psf_operation": "With processed measurement intensities selected, convolve adds calibrated blur and deconvolve performs Richardson–Lucy restoration. Select original to keep normal Measure intensities without PSF processing. Stored images and exported crops are unchanged; the database records the intensity source and full PSF provenance. Configure a kernel appropriate to every selected intensity channel.",
        "psf_path": "Measured PSF TIFF or NPY kernel, with odd spatial dimensions and finite nonnegative values. Use YX for a 2D field or ZYX for a volume. The center pixel is the optical origin; sampling must match the image. Captured once per run and sent to each worker, with exact kernel/file identity recorded.",
        "psf_image_sampling_um": "Explicit image sampling in micrometers: [Y, X] for 2D or [Z, Y, X] for a volume. All values must be positive and finite. A volume's sampling must agree with Measure's voxel calibration or anisotropy. No physical sampling is guessed.",
        "psf_kernel_sampling_um": "Measured kernel sampling in micrometers, in the same YX or ZYX order as the image. Must match image sampling exactly within numerical tolerance; no implicit resampling. Unused for a Gaussian approximation.",
        "psf_fwhm_um": "Gaussian full width at half maximum in micrometers: [Y, X] or [Z, Y, X]. This is an explicit approximation, not an inferred microscope PSF. All widths must be positive and finite.",
    },
    "train_cellpose": _TRAIN_CELLPOSE_TOOLTIPS,
    "regression": _REGRESSION_TOOLTIP_OVERRIDES,
    "umap": _UMAP_TOOLTIP_OVERRIDES,
}


#: Every setting owned by SOME training basis. Re-enabling is restricted to
#: these, so `refresh_training_basis_enablement` cannot switch a control back
#: on that something else disabled for its own reasons.
try:
    from spacr.training_basis import BASIS_SETTINGS as _BASIS_SETTINGS
    _ALL_BASIS_SETTINGS = {k for keys in _BASIS_SETTINGS.values() for k in keys}
except Exception:      # pragma: no cover - keeps the GUI importable
    _ALL_BASIS_SETTINGS = set()


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
            "log_data", "resnet_features",
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
        ("Labels & Classes", (
            "src", "dataset_mode",
            "location_column", "positive_control_id", "negative_control_id",
            "annotation_column",
        )),
        ("Feature Preparation", (
            "channel_of_interest", "exclude", "nuclei_limit",
            "pathogen_limit", "remove_highly_correlated_features",
            "remove_low_variance_features", "min_cells_per_well",
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
        ("Plots & Heatmaps", (
            "cmap", "heatmap_feature", "grouping", "min_max",
        )),
        ("Runtime & Reliability", ("verbose", "n_jobs")),
    ),
    "mask": (
        ("Input & Metadata", (
            "src", "cell_channel", "nucleus_channel", "pathogen_channel",
            NUMBER_OF_ORGANELLES,
            "organelle_channel",
            *(f"{role}_channel" for role in ALL_ORGANELLE_ROLES[1:]),
            # 404/405: which model segments every object channel above.
            "segmentation_backend",
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
        )),
        ("Image Quality", ("@Image Quality",)),
        ("Illumination Correction", (
            "illumination_correction", "illumination_model",
            "illumination_estimator", "illumination_degree",
            "illumination_dark", "illumination_per_plate",
            "illumination_max_fields", "illumination_qc",
            "illumination_on_missing",
        )),
        ("Point Spread Function", ("@Point Spread Function",)),
        ("Cell Segmentation", ("@Cell",)),
        ("Nucleus Segmentation", ("@Nucleus",)),
        ("Pathogen Segmentation", ("@Pathogen",)),
        ("Organelle Segmentation", ("@Organelle",)),
        ("Organelle Segmentation (advanced)", ("@Organelle advanced",)),
        ("Image Preprocessing (per object)",
         ("@Image preprocessing (per object)",)),
        ("Object Filtration (all objects)", ("@Object filtration",)),
        ("Quality Control", ("@Segmentation QC",)),
        ("Volumetric Processing (Beta)", ("@3D Settings (Beta)",)),
        ("Time Axes & Tracking (Beta)", ("@4D Settings (Beta)",)),
        ("Visualization & Diagnostics", (
            "plot", "cmap", "figuresize", "examples_to_plot",
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
            "number_of_organelles", "organelle_mask_dim",
            *(f"{role}_mask_dim" for role in ALL_ORGANELLE_ROLES[1:]),
            "organelle_type",
            *(f"{role}_type" for role in ALL_ORGANELLE_ROLES[1:]),
            "cytoplasm",
            "timelapse", "timelapse_objects",
        )),
        ("Illumination Correction", (
            "illumination_correction", "illumination_model",
            "illumination_estimator", "illumination_degree",
            "illumination_dark",
            "illumination_per_plate", "illumination_max_fields",
            "illumination_qc", "illumination_on_missing",
        )),
        ("Point Spread Function", ("@Point Spread Function",)),
        ("Measurement Features", (
            "save_measurements", "calculate_correlation",
            "spatial_measurements",
            "spatial_neighbor_radius",
            "bystander_measurements", "bystander_reach_in_diameters",
            "manders_thresholds", "homogeneity", "homogeneity_distances",
            "radial_dist", "distance_gaussian_sigma",
            "object_distances", "object_distance_maxima",
            "object_distance_intensity",
            "summarize_organelles_by",
        )),
        ("Object Filtering", (
            "uninfected", "cell_min_size", "cell_max_size",
            "cytoplasm_min_size",
            "nucleus_min_size", "nucleus_max_size",
            "pathogen_min_size", "pathogen_max_size", "organelle_min_area",
            *(f"{role}_min_area" for role in ALL_ORGANELLE_ROLES[1:]),
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
            NUMBER_OF_ORGANELLES,
            "organelle_channel",
            *(f"{role}_channel" for role in ALL_ORGANELLE_ROLES[1:]),
            # 404/405: which model segments every object channel above.
            "segmentation_backend",
            "channels", "magnification",
            "metadata_type", "custom_regex",
        )),
        ("Acquisition & Axes", (
            "t_stack", "t_axis_order", "t_axis",
            "frame_interval_s", "z_stack", "z_segmentation_mode", "z_axis",
            "z_projection", "anisotropy", "voxel_size_z_um",
            "voxel_size_xy_um", "stitch_threshold",
        )),
        ("Image Preprocessing", (
            "normalize", "lower_percentile", "randomize", "batch_fields",
            "consolidate",
        )),
        ('Image Quality', ('@Image Quality',)),
        ("Illumination Correction", (
            "illumination_correction", "illumination_model",
            "illumination_estimator", "illumination_degree",
            "illumination_dark", "illumination_per_plate",
            "illumination_max_fields", "illumination_qc",
            "illumination_on_missing",
        )),
        ("Point Spread Function", ("@Point Spread Function",)),
        ("Cell Segmentation", ("@Cell",)),
        ("Nucleus Segmentation", ("@Nucleus",)),
        ("Pathogen Segmentation", ("@Pathogen",)),
        ("Organelle Segmentation", ("@Organelle",)),
        ("Organelle Segmentation (advanced)", ("@Organelle advanced",)),
        ("Image Preprocessing (per object)",
         ("@Image preprocessing (per object)",)),
        ("Object Filtration (all objects)", ("@Object filtration",)),
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
            "plot", "cmap", "figuresize", "examples_to_plot",
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
            "drop_straight_tracks", "track_outlier_zscore",
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
            "infection_pca_random_state",
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
        ("Input Tables", ("paired_data", "metadata_files",
                          "count_grna_column", "count_value_column",
                          "independent_variable_layout",
                          "wide_predictor_columns", "src")),
        ("Controls & Filters", (
            "positive_control_id", "negative_control_id",
            "positive_control_wells", "negative_control_wells",
            "mixed_control_wells", "exclude_grnas", "nontargeting_control_grnas",
            "filter_column", "filter_value",
            "min_cells_per_well", "min_observations_per_hit", "fraction_threshold",
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
        ("Response", (
            "dependent_variable", "invert_dependent_variable",
            "analysis_unit", "agg_type", "transform",
        )),
        ("Model & Inference", (
            "inference", "analysis_mode", "regression_type",
            "regression_backend", "level", "model_data_layout",
            "intercept", "intercept_value",
            "model_plate_position", "random_row_column_effects",
            "multiple_testing_method", "fdr_alpha", "p_threshold_alpha",
            "p_threshold_kind", "threshold_method",
            "threshold_multiplier",
            "annotation_source",
        )),
        ("Estimator Tuning", (
            "cov_type",
            "alpha", "l1_ratio", "quantile", "huber_t",
            "spline_knots", "spline_degree",
            "hinge_threshold", "hinge_n_boot", "lasso_n_boot",
            "lasso_selection_threshold",
            "group_lasso_lambda", "rra_alpha", "rra_permutations",
        )),
        ("Permutation Test", (
            "grna_statistic",
            "guide_min_wells", "guide_primary_min_wells",
            "guide_permutations", "guide_permutation_seed",
            "guide_permutation_block", "guide_nuisance_columns",
            "guide_presence_threshold", "guide_permutation_batch_size",
        )),
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
        ("Channel Mapping", (
            "cell_chann_dim", "nucleus_chann_dim", "pathogen_chann_dim",
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
            "stain_baseline_wells", "control_quantile", "min_control_objects",
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
        ("Training Data", ("src", "mask_src", "test_src", "test_mask_src")),
        ("Starting Point", ("base_model", "model_name")),
        ("Training Schedule", ("n_epochs", "learning_rate", "weight_decay", "batch_size")),
        ("Input & Channels", ("channels", "channel_axis", "normalize", "percentiles")),
        ("Sampling & Augmentation", ("min_train_masks", "max_train_images", "nimg_per_epoch",
                                     "nimg_test_per_epoch", "scale_range")),
        ("Checkpoints", ("save_path", "save_every", "save_each")),
    ),
    "analyze_plaques": (
        ("Input & Channels", ("src", "masks")),
        ("Scale & Time", ("plate_format", "well_diameter_mm", "plaque_pixels_per_um", "plaque_formation_hours")),
        ("Experimental Growth Estimates", ("plaque_estimate_growth", "plaque_growth_reference_um", "plaque_growth_reference_hours")),
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
            "target_sequence", "regex", "offset_start", "window_length",
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
        ('Replication Method', ('replication_method',)),
        ("Assay Inputs", ("src", "parasite_table", "compartment")),
        ('Size Proxy (Legacy)', ('tables', 'min_area_bin', 'max_area', 'max_bins',
                                 'um_per_px', 'pathogen_limit', 'nuclei_limit',
                                 'group_by_class', 'class_column')),
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
    'host_pathogen': (
        ('Assay Inputs', ('src', 'hp_vacuole_table', 'hp_vacuole_prefix')),
        ('Marker Recruitment', ('hp_reference_table', 'hp_reference_prefix',
                                'hp_marker_channels', 'hp_marker_thresholds')),
        ('Parasite Counts', ('hp_parasite_table', 'hp_parasite_parent', 'hp_count_column')),
        ('Assay Output', ('save',)),
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
    "recruitment": ("@Channel Mapping",),
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

_APP_ESSENTIALS_THAT_FOLLOW_THEIR_OBJECT: Dict[str, Tuple[str, ...]] = {
    "mask": ("@Cell Segmentation", "@Nucleus Segmentation",
             "@Pathogen Segmentation", "@Organelle Segmentation"),
    "timelapse": ("@Cell Segmentation", "@Nucleus Segmentation",
                  "@Pathogen Segmentation", "@Organelle Segmentation"),
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
                candidates = [key for key in source.get(token[1:], [])
                              if key in available]
            else:
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




class SettingsSection(tuple):
    """Represent one settings-panel heading and its nested content.

    The class remains a ``(title, rows)`` tuple for compatibility with callers
    that unpack section pairs or pass them to ``dict``. ``rows`` contains all
    controls in the subtree, allowing clients without nested-section support
    to render every control exactly once.

    The hierarchy is exposed through :attr:`own_rows`, :attr:`children`, and
    :attr:`path`. The path contains section titles from the root to the current
    node, for example ``("Advanced settings", "Object filtration", "Cell")``,
    so sections with identical titles remain distinguishable.
    """


    def __new__(cls, title, own_rows=(), children=()):
        """Build a section, flattening its children's rows into its own.

        The tuple half is ``(title, rows)`` where ``rows`` is this section's own
        rows followed by every descendant's, so a consumer that only knows the
        tuple still sees the whole subtree.

        :param title: the section's caption.
        :param own_rows: rows belonging to this section itself.
        :param children: nested sections; each is re-parented onto this
            section's path.
        :returns: the new section.
        """
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
            out.append(parent)
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
        return False


#: The dash between a family prefix and the group name in a merged module's
#: heading. Written once: an em dash swapped for a hyphen in an edit fails
#: silently, because the lookups simply stop matching.
#:
#: It has to be the em dash ``spacr.qt.i18n._COMPOSITE_SEPARATOR`` splits on,
#: and that is now the whole of what makes these headings translate: nothing
#: catalogues the finished pair any more, so a heading joined with any other
#: character is one the composite pass cannot take apart, and it would read
#: English in every language while the settings around it did not.
_FAMILY_HEADING_DASH = "—"


def _family_heading(prefix: str, name: str) -> str:
    """Compose one family-prefixed section heading.

    The composed heading is a key as much as a caption: the blurb tables,
    the hidden-category lists and the layout tests are all written against
    the English ``Computer Vision — Images & Cropping``, so the heading is
    returned in English and translated where it is drawn. No catalog can
    carry a row for every prefix and group name that meet, so the pair is
    resolved by looking each half up on its own and joining the halves,
    which is what stops these headings reading half English; a translation
    that already exists for the whole pair still wins, so a reviewed
    caption is never displaced by a composed one.

    NOTHING IS CATALOGUED HERE, AND THAT IS WHAT KEEPS CLASSIFY CHEAP TO
    OPEN. This used to compose the pair for all nine translated languages
    and hand the finished row to ``add_translation`` while the settings
    panel was being built. Composing a pair in a language means asking for
    that language, and the first question asked of a language imports its
    whole catalog module — ten thousand lines each. So opening Classify
    imported nine of them, every one of which was then guaranteed to miss:
    the pair is built at run time and no generated catalog can hold a
    string that does not exist in the source. Measured under the branch
    tracer the coverage lane runs, that cost 22.3 seconds of a 20 second
    ceiling, against 1.9 for Mask Generation and 2.4 for Regression, which
    ask for nothing but the language on screen.

    ``spacr.qt.i18n.tr`` already composes exactly this shape on demand:
    ``_composite_translation`` splits an ``A — B`` label on the same dash,
    resolves each side exactly and then by term, and joins them — the same
    two lookups in the same order and the same authority this function was
    doing ahead of time. So the row is not lost, it is computed when the
    caption is drawn, for the one language being drawn. Checked against the
    values the eager pass produced: byte-identical for all five headings in
    all nine languages.
    """
    return f"{prefix} {_FAMILY_HEADING_DASH} {name}"


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
        ordered = {
            "Plate Sources & Workflow": [
                "src", "experiment", "generate_training_dataset", "train",
                "test", "generate_full_dataset", "apply_model_to_dataset",
                "dataset", "model_path", "tar_path"],

            "Labels & Classes": [
                "dataset_mode", "classes", "class_folder_names",
                "metadata_item_1_name", "metadata_item_1_value",
                "metadata_item_2_name", "metadata_item_2_value",
                "balance_to_smallest", "test_split",
                "val_split", "sample"],

            "Images & Cropping": [
                "image_source", "tables",
                "channel_of_interest", "stream_method", "object_array",
                "channel_arrays", "bounding_box",
                "crop_shape", "train_channels", "image_size", "augment"],

            "Model & Regularization": [
                "classifier_family",
                "model_type", "custom_model_path",
                "resume_checkpoint", "init_weights",
                "normalize", "dropout_rate", "weight_decay",
                "use_checkpoint"],

            "Training & Loss": [
                "epochs", "optimizer_type", "learning_rate", "schedule",
                "amsgrad", "loss_type", "class_balance", "label_smoothing",
                "focal_gamma", "focal_alpha", "logit_adjust_tau",
                "batch_size", "mixed_precision",
                "gradient_accumulation_steps", "early_stopping_patience"],

            "Test-time augmentation": ['tta_enabled', 'tta_rotations', 'tta_horizontal_flip',
                                       'tta_vertical_flip', 'tta_aggregation', 'tta_min_agreement', 'tta_max_std'],

            "Evaluation & Results": [
                "cross_validation_enabled", "cross_validation_folds",
                "cv_group_by", "holdout_plate", "nested_cv_inner_folds",
                "score_threshold",
                "classifier_evaluation", "evaluation_calibration",
                "evaluation_bins", "evaluation_fail_on_leakage",
                "leakage_audit_train_test", "leakage_hash_content",
                "leakage_require_identity", "n_top_examples",
                "plot", "tensorboard", "intermedeate_save", "pin_memory",
                "random_seed", "n_jobs", "verbose", "strict_errors",
                "max_failure_rate"],
        }
        if app_key == "classify_merged":
            ordered["Model & Regularization"] = [
                k for k in ordered["Model & Regularization"]
                if k != "classifier_family"]


            ordered.update({

                "Plate & Batch Correction": [
                    "batch_correction", "batch_column",
                    "batch_control_column", "batch_control_values",
                    "batch_covariate_column", "batch_combat_mean_only",
                    "batch_min_samples", "batch_missing_control"],
                "Model & Features": [
                    "model_type_ml", "n_estimators", "test_size",
                    "cross_validation", "reg_alpha", "reg_lambda",
                    "exclude", "nuclei_limit", "pathogen_limit",
                    "remove_highly_correlated_features",
                    "remove_low_variance_features", "min_cells_per_well",
                    "prune_features", "top_features", "n_repeats"],
            })
            ordered["Evaluation & Results"] = (
                ordered["Evaluation & Results"]
                + ["cmap", "heatmap_feature", "grouping", "min_max"])

        if app_key == "classify_merged":
            cv_family = "Computer Vision"
            ml_family = "Machine Learning"
            cv_groups = ("Images & Cropping", "Model & Regularization",
                         "Training & Loss", "Test-time augmentation")
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
                if name in ordered:
                    rebuilt[_family_heading(ml_family, name)] = ordered[name]
            for name in shared_last:
                if name in ordered:
                    rebuilt[name] = ordered[name]
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
            "nucleus_min_size", "pathogen_min_size", "organelle_min_area",
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
    return _drop_hidden_keys(app_key, result)


CATEGORY_TOOLTIPS: Dict[str, str] = {
    "OPS INPUT":
        "Where the tiles are read from and where the results are written: "
        "the sequencing folder searched, subfolders included, for tiles "
        "named by magnification, cycle, well, channels and site, the "
        "phenotype folder placed on top of it, the guide library the calls "
        "are matched against, and the folder that receives measurements.db "
        "and a report for each well. Set these first; every other OPS group "
        "assumes they are right.",
    "OPS ALIGNMENT":
        "How the tiles are put together and how the nuclei of each stitched "
        "well are segmented: the overlap the microscope left between "
        "neighbouring tiles, the overlap the segmentation windows share, "
        "and the model. Each read is attributed to the nucleus it falls on "
        "or just beside, so the model and the diameter here decide which "
        "objects can receive a barcode.",
    "OPS DECODING":
        "How a spot becomes a base: which channel carries which letter, how "
        "much brighter than its surroundings a spot must be to count as a "
        "read at all, and how far from a nucleus a read may lie and still "
        "be that nucleus's. These are measured properties of the "
        "acquisition rather than preferences, and the run reports the "
        "library match rate that says whether they are right.",
    "OPS PERFORMANCE":
        "How much of the machine the run may use: whether the graphics card "
        "is used, and how many fields are decoded at once. Nothing here "
        "changes the result, only what it costs.",

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
        "which detected objects are kept by area, mean intensity in their "
        "own channel and border filters, and whether touching labels are "
        "merged by their shared perimeter. Each group is broken down per "
        "object, so the same choice for cells and for nuclei sits side by "
        "side instead of under two unrelated headings. Nothing here needs "
        "touching on a first run.",
    "IMAGE PREPROCESSING (PER OBJECT)":
        "What is done to each object's own channel before anything is "
        "segmented — the background floor below which pixels are zeroed, "
        "the signal-to-noise ratio that sets where the contrast stretch "
        "tops out, and, for organelles, rolling-ball flattening and CLAHE. "
        "The objects do not all offer the same steps, and each sub-heading "
        "shows exactly the ones its object has.",
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
    "TEST-TIME AUGMENTATION":
        "Optional rotations and reflections during phenotype prediction. "
        "Combine predictions by averaging or voting, retain the original "
        "prediction, and flag disagreement for review. All augmentation "
        "switches are off by default; agreement measures orientation "
        "stability, not calibrated confidence or biological accuracy.",
    "IMAGE QUALITY":
        "Screen raw fields before segmentation using channel-specific focus, "
        "saturation and nonfinite-pixel criteria. Choose report-only review "
        "or explicit saved exclusions; calibrate thresholds for the acquisition. "
        "Screening is off by default and never excludes images for low object counts.",
    "HOST–PATHOGEN ANALYSIS":
        "Relate whole vacuoles to their host cells and compare marker "
        "intensities against an explicit host reference compartment. Optional "
        "linked parasite counts describe replication; absent counts and invalid "
        "reference intensities remain unknown. Calibrate marker thresholds "
        "using assay controls.",
    "MARKER RECRUITMENT":
        "Compare each vacuole's marker intensity with its host reference "
        "compartment and classify joint marker states using explicit ratio "
        "thresholds. Calibrate thresholds with assay controls; missing or "
        "invalid references remain unknown.",
    "PARASITE COUNTS":
        "Choose an individual-parasite table with explicit parent-vacuole "
        "links, or a measured count column on each vacuole. Do not supply "
        "both. Without count inputs, replication remains unmeasured rather "
        "than being inferred from recruitment or host identity.",
    "REPLICATION METHOD":
        "Choose direct parasite counts or the legacy host-aggregated area "
        "proxy. The whole-vacuole deep-learning classifier is coming soon "
        "and cannot run until a trained model is available.",
    "SIZE PROXY (LEGACY)":
        "Configure area bins and scale for the legacy replication estimate. "
        "It combines pathogen area within a host and is neither a direct "
        "parasite count nor a measured three-dimensional volume.",
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
        "Attribution settings for a trained image model: method, target "
        "layer, overlay rendering and inference normalization. These settings "
        "determine which image regions are reported as contributing to a "
        "classification.",
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
        "Two questions, asked in order. WHAT IS ESTIMATED is 'Level': "
        "'gRNA' gives one estimate per guide, 'gene' one estimate per gene "
        "with its guides pooled, 'both' gives each separately, corrected as "
        "its own family. HOW THE P VALUE IS REACHED is 'Inference': "
        "'parametric' fits every term at once and needs more wells than "
        "terms; 'nonparametric' tests each term on its own by permutation "
        "and has no such limit; 'auto' counts them and picks. The two are "
        "independent — every level is available under either.\n"
        "A THIRD ANALYSIS exists on the fitted side only: 'Regression type' "
        "= 'mixed' estimates gene effects while modelling the guide-to-guide "
        "spread inside each gene, which is why it answers both levels at "
        "once and greys 'Level'. There is no permutation equivalent — a "
        "variance component is something a model estimates — so under "
        "nonparametric inference the family is not read at all and says so.\n"
        "Below them: the multiple-testing correction applied across the "
        "tested family, the level it targets, and the control-based "
        "effect-size threshold. With hundreds of guides an uncorrected P "
        "value is not evidence. Together, these settings define what counts "
        "as a hit.",
    "REGRESSION: MODEL":
        "Select the estimation level and inference method independently. "
        "'Level' requests one effect per guide, one per gene, or separate "
        "results for both. 'Parametric' inference fits all terms "
        "simultaneously and therefore requires more wells than terms; "
        "'nonparametric' inference tests terms by plate-blocked permutation; "
        "'auto' selects the method supported by the design. 'Regression "
        "type' selects the fitted model family. The mixed model nests guides "
        "within genes and reports both levels.",
    "REGRESSION: MODEL TUNING":
        "Per-family knobs. Each applies to only some regression types, and a "
        "family refuses a setting it cannot read rather than ignoring it, so "
        "nothing here changes a fit silently. Leave them alone unless the "
        "chosen family documents the one you are changing.",
    "REGRESSION: PERMUTATION TEST":
        "Read only when inference resolves to the nonparametric test. These "
        "control the permutation itself: what is measured, how many "
        "reshuffles, what is held fixed (normally the plate), the random "
        "seed, and how many wells a guide must appear in before it is "
        "testable. They apply to the gene pass as well as the guide pass: a "
        "gene is tested as a SET, its regressor the sum of its guides' "
        "fractions, permuted with the same scheme and the same seed and "
        "corrected as its own family — never by combining its guides' P "
        "values, which would assume an independence guides scored in the "
        "same wells do not have.",
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
        "Assess fit validity using variance homogeneity, residuals, the model "
        "design matrix, influence and calibration. Outputs are written per "
        "fit as figures, a combined PDF and a text report. Diagnostics are "
        "enabled for individual analyses and disabled during parameter sweeps "
        "to avoid generating large numbers of intermediate files.",
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
    "INPUT & METADATA":
        "The image folder, which channel holds which object, and how spaCR "
        "reads plate, well and field out of the file names. Nothing "
        "segments correctly until the channel assignment and the naming "
        "convention here are right.",
    "WORKFLOW & TEST RUN":
        "Select the stages to execute, enable a small test run over a subset "
        "of fields, and configure resumption after interruption. Validate a "
        "new dataset with a test run before processing the complete plate.",
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
        "probability cutoffs are placed. The selected strategy determines "
        "which settings groups below are applicable.",
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
        "permutation importance is computed afterwards. These settings "
        "identify the measurements contributing to model decisions.",
    "PLOTS & HEATMAPS":
        "Which feature the heatmap shows, how wells are grouped, and the "
        "colour map and value range used to draw it. Presentation of the "
        "classifier's output; it does not change the fit.",
    "INPUT TABLES":
        "The metadata, score and count tables the regression runs on. All "
        "three have to agree on well and gRNA naming — disagreement there "
        "is the usual cause of an empty result.",
    "CONTROLS & FILTERS":
        "Which rows reach the model, and what they are measured against. The "
        "plate identifier, the positive and negative control wells, any row "
        "filter — and every cutoff that drops data: minimum cells per well, "
        "minimum observations per guide, the read-fraction cutoff and "
        "outlier removal. The controls set the scale the effect sizes are "
        "reported on; each cutoff silently shrinks the dataset, so check the "
        "diagnostics after changing one.",
    "ADDITIONAL SETTINGS":
        "The remaining knobs belonging to individual regression families "
        "and plots — bootstrap counts, quantile and hinge parameters, "
        "solver tolerance and axis limits. Only the ones for the model you "
        "chose above have any effect.",
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
    "INPUT MAPPING":
        "How externally generated images and label masks are found and "
        "paired — the input list, the project folder written to, recursion, "
        "plate and well layout, z handling and naming. Preview the mapping "
        "before writing anything; this is where a mismatched pairing is "
        "caught.",
    "INPUT & CHANNELS":
        "Image source, planes read by the module, and normalization or "
        "inversion applied before analysis. An empty result can indicate a "
        "channel index assigned to an empty plane.",
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
    "POINT SPREAD FUNCTION":
        "Apply a calibrated measured PSF or an explicit Gaussian approximation "
        "to segmentation channels before normalization. Convolution adds blur; "
        "Richardson–Lucy attempts deconvolution and can amplify noise. Raw "
        "images and measurement intensities remain unchanged. Leave this off "
        "unless the same kernel and pixel calibration fit every selected channel.",
    "ILLUMINATION CORRECTION":
        "Whether the microscope's uneven lighting is estimated from these "
        "fields and divided out before any intensity is measured, and how "
        "that estimate is made and checked. Turn it on when the same cell "
        "measures differently depending on where in the field it sat; "
        "leaving it off keeps that bias in every intensity feature.",
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
        "plate count, replicates and cells sampled per well. These values "
        "determine the physical plate and acquisition requirements.",
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
    "measure": {
        "POINT SPREAD FUNCTION": "Choose normal Measure intensities or calibrated PSF-processed intensities for quantitative features. PSF processing follows standard rescaling and registered preprocessing hooks. Source files and exported crops retain their existing pixels; database provenance records the choice and exact kernel. A changed kernel cannot be mixed with existing measurements.",
    },
    "train_cellpose": {
        "TRAINING DATA": "Pair microscopy images with integer object-label masks, and optionally supply a separate validation set.",
        "STARTING POINT": "Fine-tune stock Cellpose-SAM or an existing checkpoint; name the new trained model separately.",
        "TRAINING SCHEDULE": "Cellpose 4 fine-tuning uses AdamW. Start with 100 epochs, learning rate 0.00001, weight decay 0.1 and minibatch size 1.",
        "SAMPLING & AUGMENTATION": "Control sparse-field filtering, dataset size and Cellpose's online random scale augmentation without creating duplicate images.",
        "CHECKPOINTS": "Choose where trained weights are saved and whether intermediate epoch checkpoints are retained.",
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
        "SCALE & TIME":
            "Record the image scale in pixels per micrometer and the plaque "
            "formation time in hours. Measured well diameters are saved in "
            "pixels; a known physical well diameter can calibrate the image "
            "scale. These values give physical meaning to plaque sizes.",
        "EXPERIMENTAL GROWTH ESTIMATES":
            "Optionally compare plaque sizes with an experimental reference "
            "growth curve. An independently known scale is needed to estimate "
            "time, or a known time to estimate scale: plaque size alone cannot "
            "determine both. Estimates are approximate and require validation "
            "for the parasite strain, host cells and imaging conditions.",
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
        "CHANNEL MAPPING":
            "Which intensity channel holds each compartment, and which one "
            "the recruitment is measured on. These are indices into the "
            "merged stack, so a wrong one measures the wrong compartment "
            "without complaining.",
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
            "verbosity for the import. Enable strict errors when validating a "
            "new external data source.",
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



DOCS_API_BASE = "https://einarolafsson.github.io/spacr/api"

#: The published docs root. `DOCS_API_BASE` is the AutoAPI subtree of it;
#: the settings-flow page sits beside that subtree, not inside it.
DOCS_SITE_BASE = "https://einarolafsson.github.io/spacr"

#: The anchor prefix `tools/settings_flow.py` writes for each section.
#: Kept as one constant because the page and this link must agree, and
#: they are written by different programs.
FLOW_ANCHOR = "setting-flow-"


def _anchor_inside(key: str, module: str) -> str:
    """The API anchor for ``key`` if its consumer lives in ``module``.

    Returns "" when the setting is read somewhere else, or by a private
    function that AutoAPI publishes no anchor for. Pointing at a symbol
    from a DIFFERENT module would be a fragment the page does not carry,
    which the browser ignores in silence -- item 3's defect.
    """
    try:
        from spacr.qt.screens.setting_api_targets import SETTING_API_TARGETS
    except Exception:                                        # noqa: BLE001
        return ""
    row = SETTING_API_TARGETS.get(key)
    if not row:
        return ""
    where, symbol = row[0], row[1]
    if where != "spacr." + str(module).replace("/", "."):
        return ""
    if not symbol or str(symbol).rsplit(".", 1)[-1].startswith("_"):
        return ""
    return f"{where}.{symbol}"


def _has_a_flow_section(key: str) -> bool:
    """Whether the settings-flow page can answer for this setting.

    Imported lazily and forgivingly: the index is generated, and a
    checkout that has not run the generator should lose the better link
    rather than fail to draw the panel.
    """
    try:
        from spacr.qt.screens.settings_flow_index import (
            SETTINGS_WITH_A_FLOW_SECTION)
    except Exception:                                        # noqa: BLE001
        return False
    return key in SETTINGS_WITH_A_FLOW_SECTION

_APP_API_MODULE = {
    'host_pathogen': 'host_pathogen',
    "cell_montage": "cell_montage",
    "feature_dict": "feature_dict",
    "barcode_qc": "sequencing_qc",
    "explain_cv": "surrogate",
    "anndata_export": "anndata_export",
    "illumination": "illumination",
    "volcano_explorer": "volcano_style",
    "image_scatter": "qt/screens/image_scatter",
    "pca": "qt/screens/pca",
    "curate": "qt/screens/curate",
    "parameter_sweep": "parameter_sweep",
    "align": "align",
    "ops": "ops_engine",
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


def _mapped_api_target(key: str, app_key: str = "") -> tuple[str, str]:
    """``(module_path_segment, anchor)`` for a setting, or ``("", "")``.

    Reads the tables generated by ``tools/build_setting_consumer_map.py``. A
    missing table is not an error -- the caller falls back to the plain
    module link -- so a checkout that has not run the generator still gets
    the old behaviour rather than no link.

    THE APP IS PART OF THE QUESTION, since 2026-09-08. Reported: "i just
    tried sourse in mask and got a 404 error". `src` is shown in 41
    panels and this function was keyed on the SETTING ALONE, so all 41
    were sent to `annotation_dataset.generate_annotation_dataset` -- a
    real consumer, and the right one for at most one of them. No better
    ranking could have fixed that: the function was not told who was
    asking.

    So the per-module table is consulted first, and only for the module
    the asking app's help already points at. A setting that module does
    not read has no row there and falls through to the single answer,
    which is what every setting had before.

    :param key: the setting.
    :param app_key: the app whose panel is drawing it. Omitted, the
        behaviour is exactly the old one.
    """
    if not key:
        return ("", "")
    try:
        from .setting_api_targets import (SETTING_API_TARGETS,
                                          SETTING_API_TARGETS_BY_MODULE)
    except Exception:                                        # noqa: BLE001
        return ("", "")
    target = SETTING_API_TARGETS.get(key)
    own = _APP_API_MODULE.get(app_key) if app_key else None
    if own:
        rows = SETTING_API_TARGETS_BY_MODULE.get(key) or {}
        row = rows.get("spacr." + str(own).replace("/", "."))
        if row is not None:
            symbol, _exact = row
            target = ("spacr." + str(own).replace("/", "."), symbol, _exact)
    if not target:
        return ("", "")
    module, symbol, _exact = target
    if not module.startswith("spacr."):
        return ("", "")
    segment = module[len("spacr."):].replace(".", "/")
    return (segment, f"{module}.{symbol}" if symbol else "")


#: WHERE A TILE LANDS WHEN SIX TILES SHARE THREE PAGES.
#:
#: Measured: `mask` and `umap` both open
#: `spacr.core`, and all four toxoplasma assays -- Analyze Plaques,
#: Recruitment, Invasion, Replication -- open `spacr.submodules`. A reader
#: who clicked "Recruitment" BECAUSE THEY DID NOT KNOW WHAT IT DOES arrived
#: at the same text as someone who clicked "Analyze Plaques", and one page
#: cannot answer for both.
#:
#: It is fixed by where the tile POINTS rather than by new prose, because the
#: prose already exists and is good: each of these six entry points carries
#: between 238 and 684 words about that module specifically -- what the
#: red/green invasion asymmetry means and which direction its error runs,
#: why replication is a distribution and not a mean, which channel ratio
#: recruitment computes. Autoapi gives every function an anchor, so the tile
#: can land on the section that answers for it.
#:
#: ONLY THE MODULE-LEVEL LINK USES THIS -- the one behind the tile and the
#: masthead, where `key` is empty and the question is "what is this module".
#: A SETTING's help is unchanged: it still resolves through the generated
#: consumer map to wherever that value is actually read, which is a
#: different question and usually a different function.
#:
#: An entry whose anchor does not live in the module `_APP_API_MODULE` names
#: for the same key is IGNORED rather than followed, so renaming an entry
#: point degrades to today's plain module link instead of producing a
#: fragment that scrolls nowhere.
_APP_API_ANCHOR = {
    "toxoplasma": "spacr.qt.screens.organism_screen.toxoplasma",
    "plasmodium": "spacr.qt.screens.organism_screen.plasmodium",
    "candida": "spacr.qt.screens.organism_screen.candida",
    "mask": "spacr.core.preprocess_generate_masks",
    "umap": "spacr.core.generate_image_umap",
    "analyze_plaques": "spacr.submodules.analyze_plaques",
    "recruitment": "spacr.submodules.analyze_recruitment",
    "invasion": "spacr.submodules.analyze_invasion",
    "replication": "spacr.submodules.analyze_replication",
}


def _module_level_anchor(app_key: str, module: str) -> str:
    """The anchor for a tile that shares its page, checked against `module`."""
    anchor = _APP_API_ANCHOR.get(app_key, "")
    if not anchor or not module:
        return ""
    expected = f"spacr.{module.replace('/', '.')}."
    if not anchor.startswith(expected):
        return ""
    if app_key in {"toxoplasma", "plasmodium", "candida"}:
        return anchor.replace(".", "-").replace("_", "-")
    return anchor


#: Settings that begin "batch_" and have nothing to do with batch-effect
#: correction.
#:
#: The rule below sends every `batch_*` setting to `spacr.batch_correction`,
#: which is right for the six that module reads and wrong for these two:
#: `batch_fields` is how many fields the mask pipeline processes at once and
#: `batch_size` is a machine-learning batch size. A reader pressing API on
#: either was told to read about removing batch EFFECTS, which is a
#: different subject that happens to share a word.
#:
#: A deny-list rather than an allow-list, deliberately: a new
#: batch-correction setting should be picked up by the prefix without
#: anybody remembering to add it, and a new stranger is the rarer case and
#: the one worth stating.
_BATCH_PREFIX_STRANGERS = frozenset({"batch_fields", "batch_size"})


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
    anchor = ""
    chosen_by_hand = True
    if key == "psf_measurement_source":
        module, anchor = "psf_measurement", "spacr.psf_measurement.prepare_measurement_psf"
    elif key.startswith("psf_"):
        module, anchor = "psf_pipeline", "spacr.psf_pipeline.prepare_psf"
    elif app_key == "make_masks" and key.startswith("make_masks_psf_"):
        module, anchor = "point_spread", "spacr.point_spread.apply_psf"
    elif app_key == "make_masks" and key.startswith("make_masks_"):
        module = "qt/detect_chain" if key.startswith("make_masks_enh_") else "qt/screens/make_masks"
    elif key.startswith("batch_") and key not in _BATCH_PREFIX_STRANGERS:
        module = "batch_correction"
    elif key in _EVALUATION_DOC_KEYS:
        module = "classifier_evaluation"
    elif app_key == "umap" and key in _UMAP_SEARCH_DOC_KEYS:
        module = "hyperparam"
    else:
        chosen_by_hand = False
        module, anchor = _mapped_api_target(key, app_key)
        if not module:
            module = _APP_API_MODULE.get(app_key)
    if not key and not anchor:
        anchor = _module_level_anchor(app_key, module or "")
    if chosen_by_hand and module and not anchor and key:
        anchor = _anchor_inside(key, module)
    if (module and not anchor and key and not chosen_by_hand
            and _has_a_flow_section(key)):
        url = f"{DOCS_SITE_BASE}/settings_flow.html#{FLOW_ANCHOR}{key}"
    elif module:
        url = f"{DOCS_API_BASE}/spacr/{module}/index.html"
        if anchor:
            url = f"{url}#{anchor}"
    else:
        url = f"{DOCS_API_BASE}/index.html"
    code = _language_code(language)
    if code == "en":
        return url
    base, _, frag = url.partition("#")
    return f"{base}?lang={code}" + (f"#{frag}" if frag else "")



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

POSITIVE_INTEGER_SETTINGS = frozenset({"guide_permutations"})

#: What the minimum of such a spin box means, and what it shows.
AUTO_TEXT = "auto"


def _auto_or_number_box(default):
    """A spin box for a setting that takes a positive number or "auto"."""
    box = QDoubleSpinBox()
    box.setDecimals(6)
    box.setRange(0.0, 1e6)
    box.setSingleStep(0.001)
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


def _permits_float(key: str) -> bool:
    """Whether ``spacr.settings`` allows this setting to hold a fraction.

    The widget for a number is otherwise chosen from the DEFAULT VALUE's
    Python type, and a float-valued setting that happens to ship a round
    default ships an ``int``. `cell_flow_threshold` shipped 100 until
    2026-09-19 and is documented "usable range about 0-3" with Cellpose's own
    default at 0.4, so an integer box let the user choose 0, 1, 2 or 3 and
    nothing between.
    `perimeter_fraction` is declared a plain float and a FRACTION, and could
    only be set to 0 or 1.

    :param key: the setting name.
    :returns: True when the declared type admits a float.
    """
    try:
        from spacr.settings import expected_types
    except Exception:                                    # noqa: BLE001
        return False
    declared = expected_types.get(key)
    if declared is None:
        return False
    types = declared if isinstance(declared, tuple) else (declared,)
    return float in types


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

        text = str(_settings.tooltips.get(key, "") or "").lower()
    except Exception:                                    # noqa: BLE001
        text = ""
    if any(phrase in text for phrase in _UNIT_INTERVAL_PHRASES):
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
        s = " or ".join(dict.fromkeys(parts))
        if optional and s:
            s += " (optional)"
        return s
    return _TYPE_NAMES.get(t, getattr(t, "__name__", str(t)))


def _humanize(key: str) -> str:
    """Render a setting key as its human label.

    :param key: the setting name.
    :returns: its label, and ``""`` for an empty key.
    """
    return setting_label(key) if key else ""


def _strip_type_prefix(text: str) -> str:
    """Drop a leading ``(int) -`` / ``(bool)`` style prefix — the type is
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
        from ..i18n import ui_language_resolved_once
        ui_scope = ui_language_resolved_once()
        ui_scope.__enter__()
    except Exception:                                        # noqa: BLE001
        ui_scope = None
    try:
        yield
    finally:
        if ui_scope is not None:
            try:
                ui_scope.__exit__(None, None, None)
            except Exception:                                # noqa: BLE001
                pass
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
    app_key: str = "",
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
    memo_key = ("body", source, code, setting_key, app_key, category)
    if memo is not None and memo_key in memo:
        return memo[memo_key]
    from ..i18n import _exact_translation, tr

    try:
        from ..i18n_catalogs import category_help, setting_tooltip
        translated = (
            setting_tooltip(setting_key, source, code, app_key)
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

    source = _humanize(key.removeprefix("make_masks_") if app_key == "make_masks" else key)
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
    body_source = _translated_body(
        text, code, setting_key=key, app_key=app_key
    )
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
    body = _translated_body(text, code, setting_key=key, app_key=app_key)
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
        """Show the API help popup instead of Qt's own tooltip.

        :param watched: the widget being hovered.
        :param event: the event.
        :returns: ``True`` for the tooltip request it replaces, ``False``
            otherwise.
        """
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

#: The greyed-out reason a LABEL is showing, so the language pass
#: (:func:`refresh_api_tooltips`), which rebuilds the label's help from its
#: description, puts the reason back after it instead of dropping it.
_LABEL_NOTE_PROPERTY = "_spacr_label_note"



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
    "spline": "least squares with spline-adjusted covariates",
    "mixed": "mixed effects, guides nested in genes",
    "lasso": "penalised least squares, L1",
    "ridge": "penalised least squares, L2",
    "elasticnet": "penalised least squares, L1 + L2",
    "hinge": "linear SVM on a binarised response",
    "horseshoe": "sparse Poisson GLM, horseshoe",
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
        "Poisson GLM with a log link and offset(log(cell_count)) for per-well "
        "counts. The offset makes coefficients represent effects on the "
        "per-cell rate rather than total cell count, preventing differences "
        "in cell count from being interpreted as phenotype effects."
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
    "spline": (
        "Ordinary least squares in which each continuous nuisance covariate "
        "may bend through a B-spline basis. spline_knots controls how many "
        "knots each basis receives and spline_degree controls its polynomial "
        "degree; indicators and low-cardinality covariates remain linear. "
        "Guide and gene columns are never expanded, so each perturbation "
        "keeps one coefficient and its usual OLS p-value."
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
    "spline": ("spacr.nonparametric_fits.spline_design",
               "nonparametric_fits"),
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
    "This asks one question of each guide on its own: does its abundance "
    "track the phenotype? It does not fit a model, so nothing here is a "
    "coefficient — there is no formula, no family, and no estimate of what "
    "a guide does with every other guide held fixed.",
    "How the P value is reached: the guide's read fraction and the well "
    "phenotype are both cleaned of plate, row and column effects, and the "
    "cleaned phenotype is then reshuffled between wells of the same plate, "
    "many thousands of times. The P value is the share of those shuffles "
    "that produced an association at least as strong as the real one — so "
    "it is measured from your own data rather than assumed from a "
    "distribution.",
    "USE IT WHEN THE GUIDES OUTNUMBER THE WELLS. A model that fits every "
    "guide at once needs more wells than guides or its coefficients are not "
    "identifiable at all; this has no such limit. The cost is that guides "
    "sharing a well are not told apart, and that no P value can be smaller "
    "than one divided by the number of shuffles plus one.",
    "The regression settings are greyed because this path never reads them. "
    "Their values are kept, so switching back restores the model you chose. "
    "The Permutation Test section sets the number of shuffles and how much "
    "support a guide needs to be tested at all.",
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
        """Translate one source string into the explainer's language."""
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
        if key in RECOMMENDED_FOR_SCREENS:
            recommended_label = _ink(
                escape(tx("Recommended for CRISPR screens")),
                ink["success"],
                bold=True,
            )
            parts.append(
                f'<p style="margin:6px 0 2px 0;">{recommended_label}'
                f' — {escape(tx(RECOMMENDED_FOR_SCREENS[key]))}</p>')
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


#: :func:`explainer_width` per UI language. Rendering every explainer to
#: measure it was 20 ms, paid by every Regression screen that built its
#: model category, and the answer depends only on the language.
_EXPLAINER_WIDTHS: Dict[str, int] = {}


def explainer_width() -> int:
    """Return the minimum explainer width in monospace characters.

    The width is derived from the longest unbreakable formula. Prose remains
    free to wrap to the available panel width. Computed once per UI
    language.
    """
    code = _language_code()
    known = _EXPLAINER_WIDTHS.get(code)
    if known is not None:
        return known
    longest = _EXPLAINER_WIDTH
    for text in _every_explainer_line():
        if text.strip().startswith(("y ~", "rho =", "minimise")):
            longest = max(longest, len(text))
    _EXPLAINER_WIDTHS[code] = longest
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
        """Translate one source string into the explainer's language."""
        return _translated_ui_text(source, language, **values)

    if _nonparametric_selected(inference, analysis_mode):
        return "\n\n".join(
            [tx("INFERENCE: nonparametric — guide permutation")]
            + [tx(line) for line in NONPARAMETRIC_NOTE])

    if key not in _MODE_NOTES:
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
        lines.append(tx(_REFUSAL_HEADING))
        lines.append(_wrap_block(tx(_MIXED_GUIDE_OUTPUT_NOTE)))
        lines.append("")
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
            source = (_NO_P_VALUE_BOTH_NOTE if chosen == "both"
                      else _NO_P_VALUE_SINGLE_NOTE)
            lines.append(_wrap_block(tx(source)))
        elif chosen == "both":
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



#: What the nonparametric branch actually runs, in one paragraph.
#:
#: Every clause is a line of `guide_freedman_lane_test`: the block-wise
#: reshuffle is its `for indexes in block_indexes` loop, the two-sided
#: comparison is `np.abs(permutation_effects) >= np.abs(observed)`, the floor
#: on the P value is `(exceedances + 1) / (n_permutations + 1)`, and the
#: per-threshold family is the `for threshold in thresholds` loop that
#: corrects each support level on its own.
_PERMUTATION_NOTE = (
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


def _help_lives_on_the_label(control) -> bool:
    """True when :func:`retarget_field_tooltips` moved this field's help away.

    THE ONE FACT BOTH NOTE FUNCTIONS HAVE TO RESPECT. That pass ends with

        field.setToolTip("")
        field.setProperty("apiTooltipDisplayRole", "metadata")

    and leaves `apiTooltipHtml` on the field as the SOURCE the label's copy
    was made from -- not as a tooltip the field should be showing. A caller
    that reads `apiTooltipHtml` and calls `setToolTip` with it puts the help
    back on the editor, so the name stops being the hover target for that row.

    Measured on regression and classify_merged before this existed: the
    retarget moved 28 and 27 rows at screen-open, and two event-loop turns
    later `_refresh_setting_dependencies` had put every one of them back, so
    the help was on the field again on both screens.

    :param control: the editor widget the note is about.
    :returns: True when its help belongs to its name label now.
    """
    return str(control.property("apiTooltipDisplayRole") or "") == "metadata"


def _apply_greyed_note(control, note: str) -> None:
    """Append a disabled-state note without replacing the setting help.

    Existing tooltip text and API-link properties remain intact so labels
    and fields expose the same documentation while the control is disabled.
    """
    _clear_greyed_note(control)
    base = control.property("apiTooltipHtml") or control.toolTip()
    control.setProperty(_BASIS_NOTE_PROPERTY, True)
    if not _help_lives_on_the_label(control):
        control.setToolTip(f"{base}<br><i>{note}</i>" if base else note)
    control.setProperty(_PENDING_NOTE_PROPERTY, note)
    label = getattr(control, "_spacr_setting_label", None)
    if label is not None:
        label.setEnabled(False)
        _note_on_label(label, note)


def _note_on_label(label, note: str) -> None:
    """Append the greyed-out reason to the help the LABEL shows on hover.

    The original help is kept under its own property so
    :func:`_clear_greyed_note` restores it exactly rather than trying to
    strip the note back off a rendered string.
    """
    if label.property(_NOTE_BACKUP_PROPERTY) is None:
        base = str(label.property("apiTooltipHtml") or label.toolTip() or "")
        base = _without_note(base, note)
        label.setProperty(_NOTE_BACKUP_PROPERTY, base)
    base = str(label.property(_NOTE_BACKUP_PROPERTY) or "")
    text = f"{base}<br><i>{note}</i>" if base else note
    label.setProperty(_LABEL_NOTE_PROPERTY, note)
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
    if restored and not _help_lives_on_the_label(control):
        control.setToolTip(restored)
    pending = str(control.property(_PENDING_NOTE_PROPERTY) or "")
    control.setProperty(_PENDING_NOTE_PROPERTY, None)
    label = getattr(control, "_spacr_setting_label", None)
    if label is not None:
        label.setEnabled(control.isEnabled())
        label.setProperty(_LABEL_NOTE_PROPERTY, None)
        backup = label.property(_NOTE_BACKUP_PROPERTY)
        if backup is not None:
            label.setProperty("apiTooltipHtml", backup)
            label.setToolTip(str(backup))
            label.setProperty(_NOTE_BACKUP_PROPERTY, None)
        elif pending:
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
    body = str(body or "")
    if key == "regression_type":
        try:
            body = f"{body} {mixed_cost_note()}".strip()
        except Exception:                                    # noqa: BLE001
            pass
    html = format_tooltip(body, app_key, key)
    widget.setProperty("settingsAppKey", app_key)
    widget.setProperty("settingKey", key)
    widget.setProperty("apiTooltipDescriptionSource", body)
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
    only the presentation HTML/plain accessibility chrome is regenerated. A
    label showing why its setting is greyed keeps that reason after the
    regenerated help (``_LABEL_NOTE_PROPERTY``); before, the language pass
    dropped it, so a greyed row's name never said why.
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
        note = str(widget.property(_LABEL_NOTE_PROPERTY) or "")
        if note:
            widget.setProperty(_NOTE_BACKUP_PROPERTY, html)
            html = f"{html}<br><i>{note}</i>" if html else note
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
        if widget.property("apiTooltipDisplayRole") == "api-link":
            continue
        key = widget.property("settingKey")
        if key and widget not in mapped:
            mapped[widget] = str(key)
    descriptions = get_tooltips()
    for widget, key in mapped.items():
        if widget.isHidden():
            continue
        html = attach_api_tooltip(
            widget, app_key, key, _descriptions=descriptions)
        label = _setting_label_for_field(owner, widget)
        if label is None and not _is_self_labelling(widget):
            widget.setProperty("apiTooltipHtml", "")
            widget.setProperty("apiTooltipDisplayRole", "metadata")
            widget.setToolTip("")
            widget.removeEventFilter(event_filter)
            continue
        if label is None:
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
        label.removeEventFilter(event_filter)
        label.installEventFilter(event_filter)

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
    for child in candidate.findChildren(QLabel):
        if child.text().strip():
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
        candidate: Optional[QWidget] = field
        while isinstance(candidate, QWidget):
            label = _unwrap_setting_label(form.labelForField(candidate))
            if isinstance(label, QWidget):
                field._spacr_setting_label = label
                return label
            if candidate is owner:
                break
            candidate = candidate.parentWidget()

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
    supported families apart.

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

    :param model: the widget set to notify. Held as a WEAK reference for the
        reason above, so every use has to cope with it having gone.
    :param parent: the panel that owns this watcher -- NOT the model, whose
        widgets it is installed on. That split is the whole point: Qt
        decides when the watcher dies, and unhooks it from the widgets on
        the way out.
    """

    def __init__(self, model: "SettingsWidgets",
                 parent: Optional[QWidget] = None) -> None:
        """Hold the model weakly and take the panel as parent."""
        super().__init__(parent)
        self._model = weakref.ref(model)

    def eventFilter(self, obj, event) -> bool:            # noqa: N802
        """Keep a hidden settings row hidden when something tries to show it.

        :param obj: the row's field widget.
        :param event: the event.
        :returns: ``False`` -- the show is observed and corrected, never
            consumed.
        """
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


class _TrainingFolderEdit(_ScalarEdit):
    """An editable directory with a browse action, retaining the standard value contract."""

    def __init__(self, value=None, parent=None):
        """Initialize the folder value and add its trailing directory-picker action."""
        super().__init__(parent)
        from PySide6.QtWidgets import QFileDialog, QStyle
        from ..i18n import tr
        self.set_value(value)
        action = self.addAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), QLineEdit.TrailingPosition)
        action.setToolTip(tr("Choose folder…"))

        def browse():
            """Set a chosen directory; cancelling leaves the existing setting intact."""
            path = QFileDialog.getExistingDirectory(self, tr("Choose folder…"), self.text())
            if path:
                self.setText(path)
                self.editingFinished.emit()

        action.triggered.connect(browse)


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
        :param parent: parent widget; ownership only.
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
        self.setFocusProxy(self.edit)


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
        """Announce that the field changed, whatever changed it."""
        self.value_changed.emit()



#: Keys whose value may be a list of lists even when it is currently flat.
#: Taken from the same list ``spacr.settings.check_settings`` parses with
#: ``ast.literal_eval`` for the Tk GUI, so the two front ends agree on which
#: fields can hold groups.
NESTED_CAPABLE_KEYS = frozenset({
    "cell_plate_metadata", "class_metadata", "crop_mode", "dialate_png_ratios",
    "pathogen_plate_metadata", "png_dims", "png_size", "timelapse_frame_limits",
    "timelapse_objects", "treatment_plate_metadata",
    "cell_loc", "pathogen_loc", "treatment_loc", "barcode_coordinates",
})

CHANNEL_LIST_KEYS = frozenset({
    "channels", "channel_dims", "train_channels", "normalize_channels",
    "overlay_chans",
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
        :param parent: parent widget; ownership only.
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
            self.combo.addItem(label, userData=label)

        self.description = QTextBrowser(self)
        self.description.setObjectName("RegressionBackendBox")
        self.description.setReadOnly(True)
        self.description.setOpenExternalLinks(True)
        self.description.setMaximumHeight(self.BOX_HEIGHT)
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
        except RuntimeError:
            return super().eventFilter(obj, event)
        viewport = view.viewport() if view is not None else None
        kind = event.type()
        if obj is viewport:
            if kind == QEvent.MouseMove:
                self._hover_popup_row(view, event)
            elif kind == QEvent.Leave:
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
        except AttributeError:
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


#: Every caption the microscope-convention row shows. Declared here rather
#: than quoted at each call site because most of them are format templates
#: filled at runtime, which the literal-string extractor in
#: ``tools/build_i18n_catalogs.py`` cannot see from a ``setText`` call --
#: the same reason ``_REGRESSION_MENU_UI_SOURCES`` exists above.
#:
#: NOT FOLDED INTO ``_SETTINGS_MODEL_UI_SOURCES``: that set is pinned to the
#: model explainers' templates and nothing else.
TEST_ON_MY_FOLDER = "Test on my folder"
TEST_ON_MY_FOLDER_HELP = (
    "Count how many image names in the source folder this convention can "
    "read, and show the first one it cannot. Nothing is changed and nothing "
    "is written.")
WHICH_CONVENTION_FITS = "Which convention fits?"
WHICH_CONVENTION_FITS_HELP = (
    "Try every convention over the source folder and rank them by how many "
    "names each one reads. This only reports — the setting is not changed "
    "for you.")
PROVISIONAL_SUFFIX = "   [provisional]"
LOOKS_LIKE = "Looks like:  {example}"
RECONSTRUCTED_NOT_DOCUMENTED = (
    "— reconstructed from real files found in public datasets, not from "
    "vendor documentation. Test it before you run.")
CUSTOM_HAS_NO_EXAMPLE = (
    "Your own expression. It must capture wellID, fieldID and chanID; "
    "plateID is optional and falls back to the folder name.")
NO_FOLDER_TO_TEST = "Choose a source folder first."
READING_THE_FOLDER = "Reading the folder…"
NO_IMAGES_IN_THE_FOLDER = "No image files in that folder."
ALL_FILES_PARSE = "All {total} .{extension} files parse."
SOME_FILES_PARSE = (
    "{matched} of {total} files parse. The first that does not: {first}")
NOTHING_FITS = (
    "None of the built-in conventions reads any of these {total} names. "
    "Write a custom_regex, or run Import, which reads the folder names too.")
RANKING_HEADING = "Of {total} files:"
ONE_RANKING_ROW = "    {matched} of {total} — {label}  ({key})"


#: Every folder scan still running, as ``(thread, worker)``.
#:
#: A QTHREAD GARBAGE-COLLECTED WHILE IT RUNS TAKES THE PROCESS DOWN, and the
#: widget that started this one can be destroyed under it -- switching away
#: from Mask while a 70,000-file plate on a network share is being listed is
#: an ordinary thing to do. Parenting the thread to the widget only moves the
#: crash: Qt would then delete a RUNNING QThread. So the pair is held here,
#: outside any widget's lifetime, and let go on ``finished``. The readout
#: slot is connected to the widget as usual and Qt disconnects it silently
#: if the widget has gone, which is the right outcome -- there is nothing
#: left to draw on.
_LIVE_FOLDER_SCANS: set = set()


def _forget_folder_scan(thread, worker) -> None:
    """Release one finished folder scan.

    :param thread: the QThread that has just emitted ``finished``.
    :param worker: the worker that ran on it.
    """
    _LIVE_FOLDER_SCANS.discard((thread, worker))


class _FolderScanWorker(QObject):
    """List a folder's image names off the GUI thread, and parse them.

    WHY A THREAD FOR A DIRECTORY LISTING. The folder this points at is a raw
    acquisition plate: a 384-well Opera Phenix run with 9 fields, 5 planes
    and 4 channels is 69,120 files, and on a network share ``os.scandir``
    over that takes seconds. Called from the button handler it freezes the
    settings panel, the compositor offers to force-quit spaCR, and the user
    learns nothing about their filenames.

    The worker touches nothing Qt-visual. It emits numbers and two strings;
    the field draws.

    :param folder: the directory to read. Not opened, only listed.
    :param key: the ``metadata_type`` to test, or ``''`` to rank every
        convention instead.
    :param custom_regex: the user's own pattern, for ``'custom'``.
    """

    #: ``(matched, total, first_unparsed, extension)`` for one convention.
    tested = Signal(int, int, str, str)

    #: ``[(key, matched, total)]`` best first, for the autodetect offer.
    ranked = Signal(list, int)

    failed = Signal(str)

    def __init__(self, folder: str, key: str,
                 custom_regex: Optional[str] = None) -> None:
        """Hold what to read and what to test; read nothing yet."""
        super().__init__()
        self._folder = str(folder or "")
        self._key = str(key or "")
        self._custom_regex = custom_regex

    def run(self) -> None:
        """List the folder, then either test one convention or rank them all."""
        from spacr.regex_infer import (_metadata_autodetect,
                                       _metadata_parse_report)

        try:
            names = self._image_names()
        except OSError as exc:
            self.failed.emit(str(exc))
            return
        if not names:
            self.failed.emit("")
            return
        extension = self._commonest_extension(names)
        if self._key:
            matched, total, first = _metadata_parse_report(
                names, self._key, extension, self._custom_regex)
            self.tested.emit(int(matched), int(total), str(first),
                             str(extension))
        else:
            self.ranked.emit(_metadata_autodetect(names, extension),
                             len(names))

    def _image_names(self) -> List[str]:
        """Every image file name in the folder, without its path.

        Reads ``orig/`` when spaCR has already set the originals aside, so
        that testing a convention on a plate that has been run once still
        tests the names the microscope wrote rather than the names spaCR
        wrote over them. Dotted names are skipped for the reason a run skips
        them: the ``._<name>`` sidecars macOS leaves on exFAT and network
        volumes end in ``.tif`` and hold no image.

        SORTED, because the readout names the FIRST name that did not parse
        and ``os.scandir`` returns directory order. An unsorted answer names
        a different file on the same folder on two different machines, which
        makes it useless as something to paste into a bug report.
        """
        from spacr.validate import IMAGE_EXTENSIONS

        folder = self._folder
        originals = os.path.join(folder, "orig")
        if os.path.isdir(originals):
            folder = originals
        names = []
        with os.scandir(folder) as entries:
            for entry in entries:
                name = entry.name
                if name.startswith("."):
                    continue
                if not name.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                if entry.is_file():
                    names.append(name)
        return sorted(names)

    @staticmethod
    def _commonest_extension(names: Sequence[str]) -> str:
        """The extension most of these names carry, without its dot.

        THE COMMONEST RATHER THAN THE FIRST, because one stray ``.png``
        thumbnail in a ``.tif`` plate would otherwise decide the pattern for
        the whole folder and report 0 of 69,120 parsed.
        """
        counts: Dict[str, int] = {}
        for name in names:
            suffix = name.rsplit(".", 1)[-1].lower()
            counts[suffix] = counts.get(suffix, 0) + 1
        if not counts:
            return "tif"
        return max(counts.items(), key=lambda pair: (pair[1], pair[0]))[0]


class _MetadataTypeField(QWidget):
    """The microscope-convention row: a grouped menu that shows its evidence.

    THREE THINGS ON ONE ROW, and each of them is a failure this is built not
    to repeat.

    * THE MENU IS GROUPED BY VENDOR. It used to hold four entries, two of
      which were Yokogawas, so a user with a Zeiss or a Leica met a list
      that did not name their instrument and a note telling them to write a
      regular expression. Vendor headings are in the list and are not
      selectable, so scanning it for 'Leica' finds the Leica rows without
      having to already know they are called ``leica_matrix_screener``.

    * THE EXAMPLE FILENAME IS UNDER THE MENU. It is the one thing a user can
      check in a second: their folder either looks like that or it does not.
      A convention spaCR is guessing about says so on the same line, because
      a provisional pattern that parses 100% of a folder can still be
      reading the field as the channel.

    * "TEST ON MY FOLDER" ANSWERS THE QUESTION BEFORE THE RUN. A wrong
      convention is not loud: :func:`spacr.utils._extract_filename_metadata`
      prints one line per unreadable name and carries on, so half a plate
      goes missing into a scrollback nobody reads. This reports the count
      and the FIRST name that did not parse, which together say whether the
      choice is wrong or the folder is untidy. It also offers a ranking of
      every convention over the same folder -- OFFERED, never applied: the
      setting is the user's, and a value changed without being asked for is
      the same class of defect from the other direction.

    :param default: the stored ``metadata_type``.
    :param source_folder: called with no arguments for the folder to test;
        ``None`` disables the button.
    :param custom_regex: called with no arguments for the user's own
        pattern, used when the convention is ``'custom'``.
    :param parent: parent widget; ownership only.
    """

    #: Emitted when the chosen convention changes. Named `value_changed`
    #: because that is the first signal `_connect_setting_dependency_signals`
    #: looks for, so `custom_regex` greys itself the moment this moves.
    value_changed = Signal()

    def __init__(self, default: Any = None,
                 source_folder: Optional[Callable[[], Any]] = None,
                 custom_regex: Optional[Callable[[], Any]] = None,
                 parent: Optional[QWidget] = None) -> None:
        """Build the menu from the convention table and wire the button."""
        super().__init__(parent)
        self._source_folder = source_folder
        self._custom_regex = custom_regex
        self._thread: Optional[QThread] = None
        self._worker: Optional[_FolderScanWorker] = None
        self._threaded = True

        self.combo = _ValueCombo(self)
        self.combo.setObjectName("MetadataTypeCombo")
        self.combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(12)
        self._fill_the_menu()

        self.example = QLabel(self)
        self.example.setObjectName("MetadataTypeExample")
        self.example.setWordWrap(True)
        self.example.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.report = QLabel(self)
        self.report.setObjectName("MetadataTypeReport")
        self.report.setWordWrap(True)
        self.report.setVisible(False)

        self.test_button = QPushButton(TEST_ON_MY_FOLDER, self)
        self.test_button.setObjectName("MetadataTypeTestButton")
        self.test_button.setToolTip(TEST_ON_MY_FOLDER_HELP)
        self.detect_button = QPushButton(WHICH_CONVENTION_FITS, self)
        self.detect_button.setObjectName("MetadataTypeDetectButton")
        self.detect_button.setToolTip(WHICH_CONVENTION_FITS_HELP)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        buttons.addWidget(self.test_button, 0)
        buttons.addWidget(self.detect_button, 0)
        buttons.addStretch(1)

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addWidget(self.combo, 0)
        column.addWidget(self.example, 0)
        column.addLayout(buttons)
        column.addWidget(self.report, 0)
        self.setFocusProxy(self.combo)

        self.set_value(default)
        self.combo.currentIndexChanged.connect(self._on_choice_changed)
        self.test_button.clicked.connect(self.test_on_the_folder)
        self.detect_button.clicked.connect(self.rank_the_conventions)
        self._refresh_example()

    def _fill_the_menu(self) -> None:
        """One disabled heading per vendor, then that vendor's conventions."""
        from spacr.regex_infer import _metadata_convention_menu

        for vendor, rows in _metadata_convention_menu():
            self.combo.addItem(vendor, userData=None)
            heading = self.combo.model().item(self.combo.count() - 1)
            if heading is not None:
                heading.setFlags(heading.flags() & ~Qt.ItemIsEnabled)
                heading.setFlags(heading.flags() & ~Qt.ItemIsSelectable)
            for key, label, status in rows:
                suffix = "" if status == "confirmed" else PROVISIONAL_SUFFIX
                self.combo.addItem(f"    {label}{suffix}", userData=key)

    def get_value(self) -> Optional[str]:
        """The chosen convention, as the settings CSV stores it."""
        index = self.combo.currentIndex()
        if index < 0:
            return None
        return self.combo.itemData(index)

    def set_value(self, value: Any) -> None:
        """Select whatever ``value`` names.

        An unknown name is LEFT ALONE rather than raising or quietly falling
        back to the default: this runs while a settings CSV is being loaded,
        and :func:`spacr.utils._get_regex` answers a typo at run time with a
        message naming every valid convention. Silently selecting
        'cellvoyager' instead would run the wrong parser under a name the
        user never chose.
        """
        wanted = "" if value is None else str(value)
        index = self.combo.findData(wanted)
        if index >= 0:
            self.combo.setCurrentIndex(index)
        self._refresh_example()

    def text(self) -> str:
        """The chosen key -- the QComboBox contract callers may still use."""
        return str(self.get_value() or "")

    def setText(self, value: str) -> None:  # noqa: N802 - Qt contract
        """Select by key -- the QComboBox contract callers may still use."""
        self.set_value(value)

    def set_threaded(self, threaded: bool) -> None:
        """Run the folder scan inline instead of on a thread.

        For tests. A QThread in a headless test is a second event loop to
        wait on and a crash when it outlives the fixture; the scan itself is
        the same code either way.
        """
        self._threaded = bool(threaded)

    def _on_choice_changed(self, *_args) -> None:
        """A new convention: show its example, drop the stale readout."""
        self._refresh_example()
        self.report.setVisible(False)
        self.report.setText("")
        self.value_changed.emit()

    def _refresh_example(self) -> None:
        """Put the chosen convention's own example filename under the menu."""
        from spacr.regex_infer import (_metadata_convention,
                                       _metadata_convention_example)

        key = self.get_value()
        record = _metadata_convention(key) if key else None
        if record is None:
            self.example.setText("")
            return
        example = _metadata_convention_example(key)
        if not example:
            self.example.setText(CUSTOM_HAS_NO_EXAMPLE)
            return
        line = LOOKS_LIKE.format(example=example)
        if record["status"] != "confirmed":
            line = f"{line}  {RECONSTRUCTED_NOT_DOCUMENTED}"
        self.example.setText(line)

    def test_on_the_folder(self) -> None:
        """Count how many names in the source folder the choice parses."""
        self._start(str(self.get_value() or ""))

    def rank_the_conventions(self) -> None:
        """Rank every convention over the source folder. Applies nothing."""
        self._start("")

    def _start(self, key: str) -> None:
        """Read the folder off the GUI thread and report when it answers."""
        if self._thread is not None:
            return
        folder = ""
        if self._source_folder is not None:
            try:
                folder = str(self._source_folder() or "")
            except Exception:                                 # noqa: BLE001
                folder = ""
        if not folder or not os.path.isdir(folder):
            self._say(NO_FOLDER_TO_TEST)
            return
        custom = None
        if self._custom_regex is not None:
            try:
                custom = self._custom_regex()
            except Exception:                                 # noqa: BLE001
                custom = None
        worker = _FolderScanWorker(folder, key, custom)
        worker.tested.connect(self._on_tested)
        worker.ranked.connect(self._on_ranked)
        worker.failed.connect(self._on_failed)
        self._say(READING_THE_FOLDER)
        if not self._threaded:
            worker.run()
            return
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.tested.connect(thread.quit)
        worker.ranked.connect(thread.quit)
        worker.failed.connect(thread.quit)
        _LIVE_FOLDER_SCANS.add((thread, worker))
        thread.finished.connect(partial(_forget_folder_scan, thread, worker))
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_scan_finished)
        self._thread = thread
        self._worker = worker
        self.test_button.setEnabled(False)
        self.detect_button.setEnabled(False)
        thread.start()

    def _on_scan_finished(self) -> None:
        """Let go of the thread and the worker, and re-enable the buttons."""
        self._thread = None
        self._worker = None
        self.test_button.setEnabled(True)
        self.detect_button.setEnabled(True)

    def _on_tested(self, matched: int, total: int, first: str,
                   extension: str) -> None:
        """Report the count, and the first name that did not parse."""
        if matched == total:
            self._say(ALL_FILES_PARSE.format(total=total,
                                             extension=extension))
            return
        self._say(SOME_FILES_PARSE.format(matched=matched, total=total,
                                          first=first))

    def _on_ranked(self, ranked: list, total: int) -> None:
        """Show which conventions fit, best first. Changes nothing."""
        from spacr.regex_infer import _metadata_convention

        if not ranked:
            self._say(NOTHING_FITS.format(total=total))
            return
        lines = []
        for key, matched, _total in ranked[:5]:
            record = _metadata_convention(key)
            label = record["label"] if record else key
            lines.append(ONE_RANKING_ROW.format(
                matched=matched, total=total, label=label, key=key))
        self._say(RANKING_HEADING.format(total=total) + "\n" +
                  "\n".join(lines))

    def _on_failed(self, message: str) -> None:
        """No files, or the folder could not be read."""
        self._say(message or NO_IMAGES_IN_THE_FOLDER)

    def _say(self, text: str) -> None:
        """Put one readout under the row."""
        self.report.setText(text)
        self.report.setVisible(bool(text))


from ..widgets.flow import FlowHost as _FlowHost, FlowLayout as _FlowLayout


class _Chip(QFrame):
    """One value, rendered as a removable pill.

    :param text: the value shown, and the payload emitted with
        :attr:`removed` -- so it identifies the chip, not just its label.
    :param colours: the active palette, PASSED IN rather than read here so a
        strip of chips is built from one palette lookup instead of one per
        chip.
    :param parent: parent widget; ownership only.
    """

    removed = Signal(object)

    def __init__(self, text: str, colours: dict, parent=None):
        """Build the pill: its text and the mark that removes it."""
        super().__init__(parent)
        from ..i18n import tr
        from ..theme import apply_close_mark
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
            }}
            """
        )

    def text(self) -> str:
        """The value this chip carries, as typed."""
        return self._text


class _ChipStrip(QWidget):
    """A wrapping strip of chips plus the field that adds another one.

    :param placeholder: the prompt in the field that adds a chip. Its only
        instruction -- the strip has no other label.
    :param removable: whether the WHOLE STRIP can be taken away, which is
        separate from the per-chip close marks: a chip is always removable,
        this is for a strip that is one of several and may be dropped
        entirely, and it is what :attr:`emptied` reports against.
    :param parent: parent widget; ownership only.
    """

    changed = Signal()
    emptied = Signal(object)

    def __init__(self, placeholder: str = "add value…",
                 removable: bool = False, parent=None):
        """Build the strip, its entry field and (optionally) its own close mark."""
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
            apply_close_mark(self._drop, tooltip="Remove this group")
            self._drop.setFocusPolicy(Qt.NoFocus)
            self._drop.clicked.connect(lambda: self.emptied.emit(self))
            outer.addWidget(self._drop, 0, Qt.AlignTop)

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
        """Turn what is typed into a chip. Blank input adds nothing."""
        text = self._entry.text().strip()
        if not text:
            return
        self._entry.clear()
        self._add_chip(text)

    def _add_chip(self, text: str, notify: bool = True) -> None:
        """Add one chip, keeping the entry field last.

        The entry TRAILS the chips rather than sitting at a fixed end, so the
        place you type is always after the last value -- which is where the next
        one goes.
        """
        chip = _Chip(text, self._colours, self._host)
        chip.removed.connect(self._remove_chip)
        self._flow.removeWidget(self._entry)
        self._flow.addWidget(chip)
        self._flow.addWidget(self._entry)
        self._chips.append(chip)
        self._host.updateGeometry()
        self.updateGeometry()
        if notify:
            self.changed.emit()

    def _remove_chip(self, chip, notify: bool = True) -> None:
        """Take one chip out and let the strip reflow.

        ``notify=False`` is for a bulk replace, which would otherwise emit once
        per chip removed and make every listener do the work N times for one
        change.
        """
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
}


#: The subset of :data:`PATH_LIST_KEYS` that names exactly ONE file.
#:
#: Every one of these is declared ``str`` in :mod:`spacr.settings` and is
#: handed to ``pd.read_csv`` unchanged -- ``sequencing.map_sequences_to_names``
#: for the three barcode references. The legacy helper's two keys were here
#: too until 364 retired them -- `grna` on 2026-09-14 and `barcodes` on
#: 2026-09-19 -- because both were declared only by
#: ``get_map_barcodes_default_settings``, which nothing calls.
#: Giving them the multi-file control made the panel COLLECT a one-element
#: list, so merely opening the module and saving rewrote
#: ``column_csv=/…/barcodes_column.csv`` to ``['/…/barcodes_column.csv']`` in
#: the user's settings file -- and `validate` then refused every run from it
#: with "column_csv=[...] is a list, but str is expected", about a value the
#: user had never typed. The dialog and the drop target stay; the shape of the
#: value goes back to what its consumer reads.
PATH_LIST_SINGLE_KEYS: Tuple[str, ...] = (
    "grna_csv", "row_csv", "column_csv",
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


try:
    from ..theme import register_widget_qss as _register_widget_qss
    _register_widget_qss("SettingAlphabetChip", _alphabet_qss, replace=True)
except Exception:
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
        """Build a row of exclusive buttons for a small fixed choice.

        :param key: the setting this edits, used for its tooltip and API
            link.
        :param default: the value selected to start with.
        :param choices: the options, as ``(value, caption)`` pairs. The
            VALUE is what the settings file carries and the CAPTION is what
            the user reads, which is why they are a pair rather than one
            string doing both jobs.
        :param parent: parent widget.
        """
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
            button.setAccessibleName(str(value))
            button.setProperty("alphabetValue", value)
            button.toggled.connect(self._on_toggled)
            row.addWidget(button)
            self._buttons.append((value, button))
        row.addStretch(1)

        self.set_value(default)

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

    def _on_toggled(self, _checked: bool) -> None:
        """Announce that the selection changed."""
        self.changed.emit()

    @staticmethod
    def _as_members(value: Any) -> set:
        """Read a stored value as a set of letters, however it was written."""
        if value is None:
            return set()
        if isinstance(value, str):
            text = value.strip()
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
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
        """Build the editor for one list-valued setting.

        :param key: the setting this edits, used for its tooltip and API
            link.
        :param default: the value to start from.
        :param nested_capable: whether the setting accepts a list OF lists.
            Only these offer the nesting control; a flat setting given one
            would produce a value its consumer cannot read.
        :param allow_none: whether "unset" is a legal answer, distinct from
            an empty list -- the same third state a spin box cannot express.
        :param element_type: what each entry is coerced to, or ``None`` to
            keep the typed text.
        :param container: ``list`` or ``tuple``, deciding what
            :meth:`value` returns. ANYTHING ELSE BECOMES ``list`` rather
            than raising, because a settings file naming an odd container is
            still a settings file somebody has.
        :param parent: parent widget.
        """
        super().__init__(parent)
        from ..theme import active_palette
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
            " background: transparent; border: none;"
            f" padding: 0px; text-align: left; }}"
        )
        self._outer.addWidget(self._footer, 0, Qt.AlignLeft)

        self.set_value(default)

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

    def _rebuild(self, nested: bool, value) -> None:
        """Replace every strip, flat or grouped.

        Signals are BLOCKED on each entry before it is torn down:
        ``editingFinished`` fires while a focused QLineEdit is being destroyed,
        and that would call ``_commit_entry`` on a half-deleted strip.
        """
        for strip in list(self._strips):
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
        """Append one chip strip and wire it back to this editor."""
        strip = _ChipStrip(placeholder=self._placeholder(),
                           removable=self._nested, parent=self)
        strip.emptied.connect(self._drop_strip)
        self._rows.addWidget(strip)
        self._strips.append(strip)
        strip.set_values(values)
        return strip

    def _drop_strip(self, strip) -> None:
        """Remove one group, or fall back to a flat list when it was the last.

        Removing the ONLY group is how a user goes back to an ungrouped list, so
        it rebuilds flat rather than leaving an editor with nothing in it.
        """
        if len(self._strips) <= 1:
            self._rebuild(False, [])
            return
        self._strips.remove(strip)
        strip._entry.blockSignals(True)
        self._rows.removeWidget(strip)
        strip.setParent(None)
        strip.deleteLater()
        self._refresh_footer()

    def _on_footer(self) -> None:
        """Add a group when grouped, or a value when flat."""
        if self._nested:
            self._add_strip([])
            return
        current = list(self._strips[0].values()) if self._strips else []
        self._rebuild(True, [current] if current else [[]])

    def _refresh_footer(self) -> None:
        """Label the add button for what it adds -- a group, or a value."""
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

    def _placeholder(self) -> str:
        """A short prompt naming the KIND of value this list takes."""
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
        """Read a stored value as a list, however it was written.

        A list, a Python literal in a string, or a comma-separated line -- the
        last because that is what someone hand-editing a settings CSV most often
        means, and refusing it would reject a file that reads perfectly well. A
        bare ``"None"`` is EMPTY rather than the string "None", which is what a
        CSV round-trip turns an unset value into.
        """
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


#: What ``QLineEdit`` keeps of a longer text: its default ``maxLength``.
_LINE_EDIT_MAX_LENGTH = 32767


def _to_decimals(value: float, decimals: int = 6) -> float:
    """``value`` rounded the way ``QDoubleSpinBox`` rounds what it is given.

    Qt formats the number with ``decimals`` places and reads it back, which
    is what ``'%.*f'`` does here: both round the binary value correctly.
    """
    return float("%.*f" % (decimals, float(value)))


def _value_a_plain_control_holds(plan) -> Any:
    """What the plain control ``plan`` describes reads back, unbuilt.

    The value :meth:`SettingsWidgets._read_widget` returns from the control
    :meth:`SettingsWidgets._build_plain` builds from the same plan -- a spin
    box's rounding and range, a text box's empty-is-``None``, a combo's
    choice among its entries. It is how a setting whose category has not
    been opened is collected without its control being built.
    ``tests/qt/test_a_closed_category_builds_nothing_until_opened.py`` holds
    it to the built control for every setting of every module.

    :param plan: what :meth:`SettingsWidgets._route_control` answered.
    """
    control = plan["control"]
    value = plan["value"]
    if control == "toggle":
        return bool(value)
    if control == "combo":
        for shown, stored in plan["items"]:
            if stored == value or shown == str(value):
                return stored
        if value is not None and str(value) != "":
            return value
        return plan["items"][0][1] if plan["items"] else ""
    if control == "auto":
        if value is None or str(value).strip().lower() == AUTO_TEXT:
            return AUTO_TEXT
        try:
            number = min(max(_to_decimals(value), 0.0), 1e6)
        except (TypeError, ValueError):
            return AUTO_TEXT
        return AUTO_TEXT if number <= 0.0 else number
    if control == "int":
        low, high = plan["range"]
        return int(min(max(int(value), low), high))
    if control == "float":
        low, high = plan["range"]
        number = _to_decimals(value)
        if number != number:
            return _to_decimals(high)
        return float(min(max(number, _to_decimals(low)),
                         _to_decimals(high)))
    if control == "list":
        if value is None:
            return None
        text = repr(value)[:_LINE_EDIT_MAX_LENGTH].strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except Exception:                                    # noqa: BLE001
            return text
    if value is None:
        return None
    return str(value)[:_LINE_EDIT_MAX_LENGTH] or None


class _ControlToCome(NamedTuple):
    """Stands in a section's rows for a control that has not been built.

    What :meth:`SettingsWidgets.build_sections` puts in the widget slot of a
    row whose category waits to be opened (see
    `SettingsWidgets.categories_may_wait`). Only a screen that asked
    for waiting categories is ever handed one, and it swaps each for the
    real control when it builds the category.

    :param key: the setting the control is for.
    """

    key: str


class _ControlsBuiltWhenAskedFor(MutableMapping):
    """``key -> control`` for a settings panel, some controls still to come.

    ``SettingsWidgets._widgets`` has always been a plain dict, and about a
    hundred call sites and five hundred test lines read it as one. This is
    that dict with one difference: a setting can be REGISTERED without its
    control existing yet, and reading it builds the control first. So every
    reader still gets a real control, and everything asking only which
    settings the panel has -- ``in``, ``len``, iterating the keys -- is
    answered without building anything.

    WHY THE CONTROL IS BUILT ON A READ, rather than the value being kept
    somewhere else until the category opens. The control is what puts a
    value into the form the run is given: a spin box turns ``200`` into
    ``200.0``, a free-text box turns ``''`` into ``None``, a folder list
    turns ``'path'`` into ``[]``. Measured over 886 settings on 22 modules,
    48 read back differently from their declared default. A value store
    would have to repeat each control's normalisation, and would drift from
    it; building the one control asked for costs about a millisecond, and
    what waits for the category to be opened -- its rows, captions, layout
    and styling -- is where the time is.

    ``items()`` and ``values()`` build every control still to come in one
    batch, which is what :meth:`SettingsWidgets.collect` needs. Code that
    only wants what exists uses :meth:`built_items` or :meth:`built`.
    """

    def __init__(self, model: "SettingsWidgets") -> None:
        """:param model: the panel that builds a control when it is asked."""
        self._model = model
        self._built: Dict[str, QWidget] = {}
        self._to_come: Dict[str, Any] = {}
        self._order: Dict[str, None] = {}

    def wait_for(self, key: str, meta: Any) -> None:
        """Register ``key`` without building its control.

        :param key: the setting.
        :param meta: the plan :meth:`SettingsWidgets._route_control` gave
            for its plain control.
        """
        self._built.pop(key, None)
        self._to_come[key] = meta
        self._order[key] = None

    def is_built(self, key: str) -> bool:
        """Whether ``key``'s control exists."""
        return key in self._built

    def built(self, key: str) -> Optional[QWidget]:
        """``key``'s control if it exists, never building it."""
        return self._built.get(key)

    def built_items(self) -> List[Tuple[str, QWidget]]:
        """``(key, control)`` for every control that exists, in panel order."""
        return [(key, self._built[key]) for key in self._order
                if key in self._built]

    def keys_to_come(self) -> List[str]:
        """The settings whose control has not been built, in panel order."""
        return [key for key in self._order if key in self._to_come]

    def meta_for(self, key: str) -> Any:
        """The plan a waiting control is built from, or ``None``."""
        return self._to_come.get(key)

    def build(self, keys, *, decide: bool = True) -> None:
        """Build every control in ``keys`` that has not been built yet.

        :param decide: run the passes that grey controls from others after
            the batch; ``False`` for a caller that runs them itself once for
            several batches.
        """
        waiting = [key for key in keys if key in self._to_come]
        if waiting:
            self._model._build_controls(waiting, decide=decide)

    def settle(self, key: str, widget: Optional[QWidget]) -> None:
        """Record the control the model built for a waiting ``key``.

        ``None`` when the kind has no control, in which case the setting
        leaves the panel exactly as an eager build would have left it out.
        """
        self._to_come.pop(key, None)
        if widget is None:
            self._order.pop(key, None)
            return
        self._built[key] = widget

    def __getitem__(self, key):
        """Build a pending setting on first access and return its editor."""
        if key not in self._built and key in self._to_come:
            self.build((key,))
        return self._built[key]

    def __setitem__(self, key, widget) -> None:
        """Replace a pending or built editor while retaining its position in the key order."""
        self._to_come.pop(key, None)
        self._built[key] = widget
        self._order[key] = None

    def __delitem__(self, key) -> None:
        """Remove a known setting from both lazy and built inventories, or raise KeyError."""
        if key not in self._built and key not in self._to_come:
            raise KeyError(key)
        self._built.pop(key, None)
        self._to_come.pop(key, None)
        self._order.pop(key, None)

    def __contains__(self, key) -> bool:
        """Check whether a setting exists without constructing its editor."""
        return key in self._built or key in self._to_come

    def __iter__(self):
        """Iterate a snapshot of setting keys in their declared order without building editors."""
        return iter(list(self._order))

    def __len__(self) -> int:
        """Count pending and constructed settings without triggering construction."""
        return len(self._order)

    def __bool__(self) -> bool:
        """Report whether any setting is registered without constructing a widget."""
        return bool(self._order)

    def items(self):
        """Every ``(key, control)``, building what is still to come first."""
        self.build(self.keys_to_come())
        return [(key, self._built[key]) for key in self._order
                if key in self._built]

    def values(self):
        """Every control, building what is still to come first."""
        return [widget for _key, widget in self.items()]

    def copy(self) -> Dict[str, QWidget]:
        """A plain dict of every control, as ``dict.copy`` would give."""
        return dict(self.items())

    def __repr__(self) -> str:
        """Describe the built and pending editor counts without forcing lazy construction."""
        return (f"<controls: {len(self._built)} built, "
                f"{len(self._to_come)} to come>")


class SettingsWidgets:
    """Container for the Qt widgets bound to a settings dict.

    Instantiate with an `app_key`; call `.build_sections()` to get a list
    of (section_title, list_of_(label, widget)) tuples to feed into the
    Section widgets on a screen. `.collect()` returns the current settings
    dict after user edits."""

    def __init__(self, app_key: str, parent: Optional[QWidget] = None,
                 *, skip_keys=(), current=None):
        """Load the app's defaults and prepare its empty widget map.

        :param app_key: id of the app whose settings are being edited.
        :param parent: optional Qt parent for created widgets.
        :param skip_keys: settings to build NO widget for.

            For a FOLD, which mounts one module's extra settings onto
            another's panel. The timelapse fold on the mask screen built all
            364 of timelapse's settings -- 1,552 widgets, 1,148 ms -- and
            kept the 14 that mask does not already have, discarding the rest
            because the host already owns them. Naming them here skips them
            instead, which is the same result for 4% of the work.
        :param current: optional mapping from the form being rebuilt. Recognized
            settings replace the app defaults; its organelle count, slot, and
            object-channel values determine which controls the replacement
            form builds.
        """
        self.app_key = app_key
        self._parent = parent
        #: Settings to build no widget for. See __init__'s docstring.
        self._skip_keys = frozenset(str(k) for k in (skip_keys or ()))
        from spacr.settings import organelle_slots_beyond_the_count

        shipped = resolve_default_settings(app_key)
        current_values = {str(k): v for k, v in (current or {}).items()}
        deciding = dict(shipped)
        deciding.update(current_values)
        from spacr.organelle_types import (NUMBER_OF_ORGANELLES,
                                           organelle_count,
                                           organelle_role_of)

        current_names_slots = any(
            organelle_role_of(key) is not None for key in current_values)
        if (NUMBER_OF_ORGANELLES in current_values
                or current_names_slots):
            wanted = organelle_count(current_values)
        else:
            wanted = organelle_count(shipped)
        self._slots_built_for = max(0, min(wanted, PANEL_ORGANELLE_SLOTS))
        self._defaults = organelle_slots_beyond_the_count(
            shipped, self._slots_built_for)
        from spacr.settings import expected_types

        for key, value in current_values.items():
            if key in self._defaults or (
                    organelle_role_of(key) is not None
                    and key in expected_types):
                self._defaults[key] = value
        self._skip_keys = frozenset(self._skip_keys) | frozenset(
            self._organelle_keys_beyond(self._slots_built_for,
                                        self._defaults)
        )
        self._slots_the_panel_added = {
            key: value for key, value in self._defaults.items()
            if key not in shipped and key not in current_values}
        self._widgets = _ControlsBuiltWhenAskedFor(self)
        #: ``(title, keys) -> bool`` for a top-level category whose controls
        #: may wait until it is opened, set by a screen before
        #: :meth:`build_sections`. ``None`` builds every control at once,
        #: which is what every model not built for a screen does.
        self.categories_may_wait = None
        #: Nesting depth of :meth:`_build_controls`, so the state passes it
        #: runs afterwards run once for a batch, not once per control.
        self._controls_arriving = 0
        self._hidden_by_the_run: set = set()
        self._guarded_rows: Dict[int, str] = {}
        #: ``id(section) -> section`` for the slot headings this hid, so it
        #: can put back exactly what it took and nothing else.
        self._headings_of_absent_slots: Dict[int, Any] = {}
        self._object_row_guard = _HiddenRowWatcher(self, parent)
        self._object_rule_pass_queued = False
        #: Values the currently selected organelle preset still owns. A user
        #: edit removes its key, so a later diameter change can update a
        #: size-dependent recommendation without overwriting advanced work.
        self._organelle_preset_owned: Dict[str, Dict[str, Any]] = {}
        self._applying_organelle_preset = False
        #: Called with the keys this pass is hiding, just before row
        #: visibility is decided, so the screen can lay out any row it left
        #: unbuilt that is about to be shown. The model decides WHETHER a row
        #: is on the form; only the screen can BUILD one. Left ``None`` on a
        #: model built for its values rather than for a screen.
        self.rows_are_laid_out_by = None
        self.rows_are_filtered_by = None
        #: Called with no arguments during the object pass, after the pass
        #: has decided what it hides: the keys the screen's own filters (the
        #: settings search, the 3D and Time switches) will hide again the
        #: moment the pass ends. Each row is then set ONCE, to where it will
        #: end up, instead of shown by this pass and hidden again by the next;
        #: see :meth:`refresh_object_visibility`. ``None`` on a model built
        #: for its values.
        self.rows_the_screen_hides = None
        self._hidden_by_their_object: set = set()
        self._tooltips = get_tooltips()
        self._data_context: Dict[str, Any] = {'plate_count': None}
        self._tooltips.update(_APP_TOOLTIP_OVERRIDES.get(app_key, {}))
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

        Each result is a :class:`SettingsSection`, a tuple subclass compatible
        with ``(title, rows)`` unpacking and ``dict()`` conversion. Its
        ``rows`` member contains all controls in the section's subtree,
        allowing clients without nested-section support to render every
        control exactly once.

        Three levels are expressible: the "Advanced settings" umbrella, the
        family headings that declare it as their parent
        (``spacr.settings.CATEGORY_PARENTS``), and one sub-heading per object
        inside each family, derived from the setting keys. A category that
        declares no parent and splits into no objects is a single flat
        section exactly as before.

        Anything in no category at all lands in a trailing "Other".
        """
        from spacr.settings_spec import convert_settings_dict_for_gui
        variables = convert_settings_dict_for_gui(self._defaults)

        hidden_keys = set(_APP_HIDDEN_KEYS.get(self.app_key, frozenset()))
        hidden_keys.update(self._skip_keys)
        from PySide6.QtCore import QCoreApplication, QEventLoop

        import time as _time

        cats = categories_for_app(self.app_key, get_categories())
        hidden = _APP_HIDDEN_CATEGORIES.get(self.app_key, set())
        may_wait = self._keys_that_may_wait(cats, hidden, variables,
                                            hidden_keys)
        _BREATH = 0.025
        next_breath = _time.perf_counter() + _BREATH
        with _timing.span("build widgets", f"{len(variables)} settings"):
            for key, meta in variables.items():
                if key in hidden_keys:
                    continue
                kind, options, default = meta
                if key in may_wait:
                    route, what = self._route_control(kind, options, default,
                                                      key)
                    if route == "plain":
                        self._widgets.wait_for(key, what)
                        continue
                if _time.perf_counter() >= next_breath:
                    next_breath = _time.perf_counter() + _BREATH
                    QCoreApplication.processEvents(
                        QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
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
            src_widget.value_changed.connect(self._refresh_contextual_widgets)

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

        self._connect_setting_dependency_signals()

        self._connect_object_visibility_signals()

        self._refresh_contextual_widgets()
        self._refresh_umap_reducer_enablement()
        self._refresh_analysis_unit_lock()
        self._refresh_regression_backend()
        self._state_passes_ready = True

        used_keys = set()
        split_by_object = set(_shared_category_parents())
        sections: List[SettingsSection] = []
        for cat_name, keys in cats.items():
            if cat_name in hidden:
                continue
            rows: List[Tuple[str, QWidget]] = []
            row_keys: List[str] = []
            for k in keys:
                if k in self._widgets and k not in used_keys:
                    rows.append((self._label_for(k), self._row_control(k)))
                    row_keys.append(k)
                    used_keys.add(k)
            if not rows:
                continue
            if cat_name in split_by_object:
                own, children = _split_rows_by_object(rows, row_keys)
                sections.append(SettingsSection(cat_name, own, children))
            else:
                sections.append(SettingsSection(cat_name, rows))

        remaining = [(self._label_for(k), self._row_control(k))
                     for k in self._widgets if k not in used_keys]
        if remaining:
            sections.append(SettingsSection("Other", remaining))

        if self._parent is not None:
            timer = QTimer(self._parent)
            timer.setSingleShot(True)
            timer.timeout.connect(self.refresh_object_visibility)
            timer.timeout.connect(timer.deleteLater)
            timer.start(0)

        return _nest_sections(sections)

    def _keys_that_may_wait(self, cats, hidden, variables,
                            hidden_keys) -> set:
        """The settings whose control can wait for its category to open.

        A setting waits when the top-level category it is laid out under --
        the first category that lists it, as :meth:`_build_sections` places
        it, or the umbrella that category hangs from -- is one
        :attr:`categories_may_wait` says may wait, and its control is one of
        the plain ones (:meth:`_build_plain`), whose value can be read
        without building it. Every other control is built with the panel.

        :returns: the keys that may wait; empty when no screen asked.
        """
        judge = self.categories_may_wait
        if judge is None:
            return set()
        parents = _shared_category_parents()
        owner: Dict[str, str] = {}
        for cat_name, keys in cats.items():
            if cat_name in hidden:
                continue
            for key in keys:
                owner.setdefault(key, parents.get(cat_name, cat_name))
        members: Dict[str, List[str]] = {}
        for key in variables:
            if key not in hidden_keys:
                members.setdefault(owner.get(key, "Other"), []).append(key)
        waiting = set()
        for top, keys in members.items():
            try:
                wait = bool(judge(top, tuple(keys)))
            except Exception:                                # noqa: BLE001
                wait = False
            if wait:
                waiting.update(keys)
        return waiting

    def _row_control(self, key: str):
        """``key``'s control for a section row, or a stand-in if it waits."""
        control = self._built_control(key)
        if control is None and key in self._widgets:
            return _ControlToCome(key)
        return control

    def _build_controls(self, keys, *, decide: bool = True) -> None:
        """Build the waiting controls for ``keys``, then settle their state.

        Each control is built exactly as :meth:`_build_sections` builds one,
        from the same declaration, with its help attached. Then the passes
        that decide a control's STATE rather than its value -- greyed by the
        classifier family, the training basis, a dependency rule, the UMAP
        reducer or the analysis unit -- run once for the batch: they touch
        only controls that exist, so a control that arrives later is decided
        when it arrives, as it would have been had it been there all along.

        :param keys: the settings whose waiting controls to build.
        :param decide: run those passes afterwards; ``False`` for a caller
            that runs them itself once for several batches.
        """
        with language_resolved_once():
            self._controls_arriving += 1
            try:
                for key in keys:
                    plan = self._widgets.meta_for(key)
                    if plan is None:
                        continue
                    widget = self._build_plain(plan)
                    attach_api_tooltip(widget, self.app_key, key,
                                       _descriptions=self._tooltips)
                    self._widgets.settle(key, widget)
            finally:
                self._controls_arriving -= 1
            if (decide and self._controls_arriving == 0
                    and self._state_passes_ready
                    and self._decided_by_a_pass().intersection(keys)):
                self._decide_the_state_of_every_control()

    #: Set once :meth:`_build_sections` has wired the panel; before that a
    #: control that arrives is decided by the passes the build runs itself.
    _state_passes_ready = False

    def _decided_by_a_pass(self) -> set:
        """Every setting whose control's state one of the passes decides.

        What :meth:`_build_controls` checks an arriving batch against, so a
        batch no pass has an opinion about -- most of them -- costs no pass.
        Computed once per panel; the rules and tables it reads are fixed.
        """
        owned = getattr(self, "_keys_decided_by_a_pass", None)
        if owned is not None:
            return owned
        owned = {"exclude_rows", "regression_backend", "metric"}
        owned.update(self._rules_for_this_panel())
        owned.update(_ALL_BASIS_SETTINGS)
        for keys in _UMAP_REDUCER_SETTINGS.values():
            owned.update(keys)
        try:
            from spacr.classify import FAMILY_SETTINGS

            for keys in FAMILY_SETTINGS.values():
                owned.update(keys)
        except Exception:                                    # noqa: BLE001
            pass
        try:
            from ...settings_advisor import UNIT_REQUIREMENTS

            for keys in UNIT_REQUIREMENTS.values():
                owned.update(keys)
        except Exception:                                    # noqa: BLE001
            pass
        self._keys_decided_by_a_pass = owned
        return owned

    def _state_pass_steps(self):
        """:meth:`_decide_the_state_of_every_control`, one pass per step."""
        for decide in (self._refresh_contextual_widgets,
                       self._refresh_umap_reducer_enablement,
                       self._refresh_analysis_unit_lock,
                       self._refresh_regression_backend):
            try:
                decide()
            except Exception:                                # noqa: BLE001
                LOGGER.debug("could not settle the controls that arrived",
                             exc_info=True)
            yield

    def _decide_the_state_of_every_control(self) -> None:
        """Run every pass that greys, locks or fills a control from another.

        The same four calls :meth:`_build_sections` ends with. Each decides
        every control it owns from the values it reads, so running them
        again after controls arrive leaves the panel as an eager build
        would have left it.
        """
        for _step in self._state_pass_steps():
            pass

    @staticmethod
    def _keys_of_objects_the_run_has_no_channel_for(settings,
                                                    deciding=None) -> set:
        """Settings for an object whose switch names no plane.

        :param settings: the defaults the panel is about to build from.
        :returns: the keys not to build.

        A segmenting module switches an object with ``*_channel``; Measure
        has no such settings and switches the same objects with
        ``*_mask_dim``. A run whose applicable switch is empty has no such
        object, so its object-specific settings under multiple headings would
        be settings the run can never use.

        CELL IS ALWAYS THERE. It is the object every other one is measured
        against and the one a run is most likely to want, so hiding it on an
        unset channel would empty the form a user has only just opened.

        DECIDED WHEN THE PANEL IS BUILT, not while typing. Re-running this on
        every keystroke is what made the Mask module hang; a channel typed
        afterwards changes what the run does without rearranging the form
        under the hands typing it.
        """
        gated = ("nucleus", "pathogen")
        absent = set()
        answers = deciding if deciding is not None else settings
        for role in gated:
            switches = tuple(
                key for key in object_switch_keys(role) if key in settings)
            if not switches:
                continue
            named = any(_names_a_plane(answers.get(key)) for key in switches)
            if named:
                continue
            prefix = f"{role}_"
            for key in settings:
                name = str(key)
                if name in switches:
                    continue
                if name.startswith(prefix):
                    absent.add(name)
        return absent

    @staticmethod
    def _organelle_keys_beyond(count: int, settings) -> set:
        """Every organelle key belonging to a slot past ``count``.

        :param count: how many slots the run has.
        :param settings: the defaults the panel is about to build from.
        :returns: the keys not to build.

        BY ROLE PREFIX, because that is what names a slot. `organelle_` is
        slot one, `organelleb_` is slot two, and so on; a key belongs to the
        first role its name starts with, longest first so `organelleb_area`
        is not read as `organelle_` plus a suffix.
        """
        from spacr.organelle_types import ALL_ORGANELLE_ROLES

        from spacr.organelle_types import NUMBER_OF_ORGANELLES

        count = max(0, int(count))
        keep = {f"{role}_" for role in ALL_ORGANELLE_ROLES[:count]}
        every = {f"{role}_" for role in ALL_ORGANELLE_ROLES}
        drop = every - keep
        beyond = set()
        for key in settings:
            name = str(key)
            if name == NUMBER_OF_ORGANELLES:
                continue
            owner = max((p for p in every if name.startswith(p)),
                        key=len, default=None)
            if owner is not None and owner in drop:
                beyond.add(name)
            elif count == 0 and "organelle" in name.lower():
                beyond.add(name)
        return beyond

    def grow_to_fit_the_organelle_count(self, count) -> int:
        """Build the organelle slots a raised count now asks for.

        :param count: the new ``number_of_organelles``.
        :returns: how many slots the panel holds afterwards.

        THE PANEL OPENS WITH WHAT THE RUN HAS. Building every nameable slot
        up front and hiding the surplus is what made the Mask screen 1,551
        widgets; a control that was never built cannot be revealed, so the
        panel has to be able to grow instead.

        ONE CONTROL, DELIBERATELY CHANGED. This is safe to do here and was
        not safe to do per keystroke: the count is a single spinbox somebody
        sets on purpose, where a channel is a field they type digits into.
        Growing never shrinks -- a slot built once keeps whatever the user
        has since put in it, and a count lowered and raised again finds its
        values where it left them.
        """
        try:
            wanted = max(0, int(count or 0))
        except (TypeError, ValueError):
            return getattr(self, "_slots_built_for", 0)
        wanted = min(wanted, PANEL_ORGANELLE_SLOTS)
        if wanted <= getattr(self, "_slots_built_for", 0):
            return self._slots_built_for

        from spacr.settings import organelle_slots_beyond_the_count

        shipped = resolve_default_settings(self.app_key)
        self._defaults = organelle_slots_beyond_the_count(shipped, wanted)
        self._slots_the_panel_added = {
            key: value for key, value in self._defaults.items()
            if key not in shipped}
        self._slots_built_for = wanted
        return wanted

    def tooltip_for(self, key: str) -> str:
        """Return the HTML-formatted tooltip for a given setting key."""
        return format_tooltip(self._tooltips.get(key, ""), self.app_key, key)

    def plain_tooltip_for(self, key: str) -> str:
        """Return the plain-text hint (description + docs URL) for a setting."""
        return plain_tooltip(self._tooltips.get(key, ""), self.app_key, key)


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
        for key in self._widgets:
            if key not in self._defaults:
                continue
            try:
                current = self._coerce_to_expected_type(
                    key, self._read_value(key))
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

        On Mask and Timelapse, each object's segmentation settings are added
        for every object whose channel names a plane, so setting a pathogen
        channel brings the Pathogen Segmentation rows into Essentials as well
        as into All settings. See :meth:`_essentials_that_follow_their_object`.

        The module's layout part is computed once per set of settings: it
        rebuilds the module's whole category layout, 16 ms on Regression,
        and the search strip asks on every pass of the object rule.
        """
        cached = getattr(self, "_essential_keys_cache", None)
        if cached is None or cached[0] != len(self._widgets):
            cached = (len(self._widgets),
                      [key for key in essential_keys(self.app_key)
                       if key in self._widgets])
            self._essential_keys_cache = cached
        keys = list(cached[1])
        keys.extend(self._essentials_that_follow_their_object())
        return list(dict.fromkeys(keys))

    def _essentials_that_follow_their_object(self) -> List[str]:
        """The segmentation settings of every object this run segments.

        Read from ``_APP_ESSENTIALS_THAT_FOLLOW_THEIR_OBJECT``: for Mask and
        Timelapse, the two modules that segment, the four
        ``<Object> Segmentation`` categories. A key joins when its
        object's channel names a plane, read from the widgets now rather
        than at build, so a channel typed after the form opened counts. A key
        the object rule cannot place, such as ``adjust_cells``, goes with the
        rest of its category. Slots beyond ``number_of_organelles`` and rows
        a morphology excludes stay hidden anyway, because the settings search
        takes :meth:`keys_hidden_by_the_run` out before it applies Essentials.

        :returns: keys in layout order; empty for a module with no such
            categories.
        """
        groups = getattr(self, "_essential_object_groups", None)
        if groups is None:
            groups = []
            tokens = _APP_ESSENTIALS_THAT_FOLLOW_THEIR_OBJECT.get(
                str(self.app_key or ""), ())
            if tokens:
                cats = categories_for_app(self.app_key, get_categories())
                for token in tokens:
                    keys = [key for key in _expand_layout_tokens(cats, (token,))
                            if key in self._widgets]
                    if keys:
                        groups.append(tuple(keys))
            self._essential_object_groups = groups
        if not groups:
            return []
        current = self._object_visibility_settings()
        switches: Dict[str, Tuple[str, ...]] = {}
        joined: List[str] = []
        for keys in groups:
            placed: List[str] = []
            unplaced: List[str] = []
            for key in keys:
                role = object_of_setting(key)
                if role is None:
                    unplaced.append(key)
                    continue
                if role not in switches:
                    switches[role] = tuple(
                        k for k in object_switch_keys(role)
                        if k in self._widgets)
                named = switches[role]
                if not named or any(_names_a_plane(current.get(k))
                                    for k in named):
                    placed.append(key)
            if placed:
                joined.extend(placed + unplaced)
        return joined

    def _label_for(self, key: str) -> str:
        """Return the caption a setting is shown under on this screen.

        A plugin's own label wins; then the handful of per-module overrides
        where one key means something narrower than its general name; then the
        shared label table.

        :param key: the setting name.
        :returns: the caption to show.
        """
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
        if self.app_key == "train_cellpose":
            labels = {"src": "Image source folder", "mask_src": "Mask source folder",
                      "test_src": "Validation image folder", "test_mask_src": "Validation mask folder",
                      "save_path": "Checkpoint folder", "channel_axis": "Channel axis"}
            if key in labels:
                return labels[key]
        if self.app_key == "regression" and key == "src":
            return "Output directory"
        return setting_label(key)

    def _widget_for(self, kind: str, options: Any, default: Any,
                    key: str) -> Optional[QWidget]:
        """Build the control one setting gets on this screen.

        See :meth:`_route_control` for how it is chosen.

        :returns: the control, or ``None`` when the kind has none.
        """
        route, what = self._route_control(kind, options, default, key)
        if route == "special":
            return what()
        if route == "plain":
            return self._build_plain(what)
        return None

    @staticmethod
    def _build_plain(plan) -> QWidget:
        """Build the plain control ``plan`` describes, holding its value.

        :param plan: what :meth:`_route_control` answered for the setting.
        """
        control = plan["control"]
        value = plan["value"]
        if control == "toggle":
            w = Toggle()
            w.setChecked(bool(value))
            return w
        if control == "combo":
            w = _ValueCombo()
            w.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(12)
            for shown, stored in plan["items"]:
                w.addItem(shown, userData=stored)
            for i in range(w.count()):
                if w.itemData(i) == value or w.itemText(i) == str(value):
                    w.setCurrentIndex(i)
                    break
            else:
                if value is not None and str(value) != "":
                    w.insertItem(0, str(value), userData=value)
                    w.setCurrentIndex(0)
            return w
        if control == "auto":
            return _auto_or_number_box(value)
        if control == "int":
            w = QSpinBox()
            w.setRange(*plan["range"])
            w.setValue(value)
            return w
        if control == "float":
            w = QDoubleSpinBox()
            w.setRange(*plan["range"])
            w.setSingleStep(plan["step"])
            w.setDecimals(6)
            w.setValue(value)
            return w
        if control == "list":
            w = _ListEdit()
            w.set_value(value)
            return w
        w = _ScalarEdit()
        w.set_value(value)
        return w

    def _route_control(self, kind: str, options: Any, default: Any,
                       key: str):
        """Choose the control one setting gets on this screen.

        The order of the checks is load-bearing. Path-list and column-naming
        keys are matched before the chip-editor and combo routes, because
        several of them are declared ``list`` or ``str`` and would otherwise
        fall through to a free-text box -- which is what made a typo
        indistinguishable from a real column name. Closed alphabets are matched
        before the chip editor for the same reason.

        :param kind: the declared control kind.
        :param options: the declared options, for the kinds that have them.
        :param default: the declared default.
        :param key: the setting name; several controls are chosen from this
            alone, since the setting's meaning is narrower than its type.
        :returns: ``("plain", plan)`` for one of the plain Qt controls,
            which :meth:`_build_plain` builds and
            :func:`_value_a_plain_control_holds` can read without building;
            ``("special", build)`` for every other control, with a callable
            that builds it; ``(None, None)`` when the kind has none.
        """
        parent = self._parent
        if self.app_key == "train_cellpose" and key in {"model_name", "channels"}:
            kind, options = "entry", None
            default = self._defaults.get(key, default)
        if self.app_key == "train_cellpose" and key in {"src", "mask_src", "test_src", "test_mask_src", "save_path"}:
            return "special", lambda: _TrainingFolderEdit(self._defaults.get(key, default), parent)
        if self.app_key == "umap" and key == "src":
            return "special", lambda: DatabaseSetWidget(
                value=self._defaults.get(key, default),
                mode="folder",
                table="cell",
                title="Choose one or more spaCR project folders",
                on_colour_by=partial(self.set_value_for_key, "color_by"),
                parent=parent,
            )
        if self.app_key == "umap" and key == "exclude_rows":
            return "special", lambda: RowExclusionEditor(
                value=self._defaults.get(key, default),
                parent=parent,
            )
        if self.app_key == "external_masks" and key == "inputs":
            return "special", lambda: ExternalMaskInputWidget(
                value=self._defaults.get(key, default),
                parent=parent,
            )
        if key == "classes":
            def class_editor():
                """The class editor, told which frame it is previewing."""
                widget = ClassEditorWidget(
                    value=self._defaults.get(key, default),
                    parent=parent,
                )
                frame = getattr(self, "_preview_frame", None)
                if frame is not None:
                    widget.set_frame(frame)
                return widget
            return "special", class_editor
        if key == "png_channel_mapping":
            return "special", lambda: ChannelMappingWidget(
                value=self._defaults.get(key, default),
                parent=parent,
            )
        if key in PATH_LIST_KEYS:
            return "special", lambda: FilePathListWidget(
                value=self._defaults.get(key, default),
                kind=PATH_LIST_KEYS[key],
                title=PATH_LIST_TITLES.get(key, "Choose input files"),
                single=key in PATH_LIST_SINGLE_KEYS,
                parent=parent,
            )
        if key == "paired_data":
            return "special", lambda: PairedFileTableWidget(
                value=self._defaults.get(key, default), parent=parent)
        source = CSV_COLUMN_SOURCES.get(self.app_key, {}).get(key)
        if source is not None:
            return "special", lambda: _CsvColumnField(
                key=key,
                default=self._defaults.get(key, default),
                paths=partial(self._input_csv_paths, source.roles),
                what=source.what,
                parent=parent,
            )
        if key == "regression_backend":
            return "special", lambda: _RegressionBackendField(
                default=self._defaults.get(key, default),
                regression_type=self._defaults.get("regression_type"),
                parent=parent,
            )
        if key == "segmentation_backend":
            from ..model_install import SegmentationBackendCombo
            return "special", lambda: SegmentationBackendCombo(
                default=self._defaults.get(key, default), parent=parent)
        if key == "metadata_type":
            return "special", lambda: _MetadataTypeField(
                default=self._defaults.get(key, default),
                source_folder=self._current_source_folder,
                custom_regex=partial(self._current_setting, "custom_regex"),
                parent=parent,
            )
        app_options = _APP_COMBO_OPTIONS.get(self.app_key, {})
        if key in app_options:
            kind = "combo"
            options = app_options[key]
        if key == "regression_type":
            kind = "combo"
            options = _regression_type_menu()
        elif key == "multiple_testing_method":
            from spacr.multiple_testing import method_choices
            kind = "combo"
            options = method_choices()
        if self.app_key == "umap" and key == "metric":
            from spacr.hyperparam import UMAP_METRICS
            kind = "combo"
            options = list(UMAP_METRICS)
        if self.app_key == "map_barcodes" and key == "regex":
            return "special", lambda: BarcodeRegexWidget(
                value=self._defaults.get(key, default),
                parent=parent,
            )
        if key in FIXED_ALPHABETS:
            return "special", lambda: _AlphabetSelect(
                key=key,
                default=self._defaults.get(key, default),
                choices=FIXED_ALPHABETS[key],
                parent=parent,
            )
        if key in EXCLUDE_LIST_KEYS:
            return "special", lambda: _ListEditor(
                key=key,
                default=self._defaults.get(key, default),
                nested_capable=False,
                allow_none=True,
                element_type=str,
                container=list,
                parent=parent,
            )
        actual_default = self._defaults.get(key, default)
        if key == "timelapse_objects" or (
            key in CHANNEL_LIST_KEYS
            and list_shape_for(key, actual_default) is not None
        ):
            kind = "entry"
        if kind == "check":
            return "plain", {"control": "toggle", "value": bool(default)}
        if kind == "combo":
            items = []
            for opt in (options or []):
                if isinstance(opt, tuple) and len(opt) == 2:
                    stored, shown = opt
                else:
                    stored = opt
                    shown = "None" if opt is None else str(opt)
                items.append((str(shown), stored))
            if key in self._defaults:
                default = self._defaults[key]
            if key == "image_source":
                default = _image_source_the_panel_offers(default)
            return "plain", {"control": "combo", "items": items,
                             "value": default}
        if kind == "entry":
            shape = list_shape_for(key, self._defaults.get(key, default))
            if shape is not None:
                nested_capable, allow_none, element_type, container = shape
                return "special", lambda: _ListEditor(key=key,
                                   default=self._defaults.get(key, default),
                                   nested_capable=nested_capable,
                                   allow_none=allow_none,
                                   element_type=element_type,
                                   container=container)
            if key in AUTO_OR_NUMBER_SETTINGS:
                return "plain", {"control": "auto",
                                 "value": self._defaults.get(key, default)}
            if _is_clearable_plane_setting(key):
                return "plain", {"control": "text",
                                 "value": self._defaults.get(key, default)}
            if isinstance(default, bool):
                return "plain", {"control": "toggle", "value": default}
            if isinstance(default, int) and _permits_float(key):
                default = float(default)
            if isinstance(default, int):
                minimum = (
                    1 if key in POSITIVE_INTEGER_SETTINGS
                    else -2_147_483_648
                )
                return "plain", {"control": "int", "value": default,
                                 "range": (minimum, 2_147_483_647)}
            if isinstance(default, float):
                low, high, step = _float_domain(key, default)
                return "plain", {"control": "float", "value": default,
                                 "range": (low, high), "step": step}
            if isinstance(default, list):
                return "plain", {"control": "list", "value": default}
            return "plain", {"control": "text", "value": default}
        return None, None

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
            if typ in (list, tuple, dict):
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    continue
                if typ is dict and isinstance(parsed, dict):
                    return parsed
                if typ in (list, tuple) and isinstance(parsed, (list, tuple)):
                    return typ(parsed)
                continue
        return value

    #: Settings whose value has ONE canonical form, whatever shape the
    #: widget hands back. `channel_of_interest` is drawn as a multi-select,
    #: so one channel comes back as `[3]` where the default is `3` -- the
    #: same feature space, but a panel that rewrites a default makes every
    #: settings file differ from it and breaks "has this been changed?".
    CANONICAL_READERS = {
        "channel_of_interest": "spacr.settings:canonical_feature_selection",
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

    def _built_control(self, key: str) -> Optional[QWidget]:
        """``key``'s control if it has been built, never building it.

        Tolerates a plain dict in ``_widgets``, which is what several tests
        and older callers put there.
        """
        built = getattr(self._widgets, "built", None)
        return built(key) if callable(built) else self._widgets.get(key)

    def _built_controls(self) -> List[Tuple[str, QWidget]]:
        """``(key, control)`` for every control built so far, in panel order."""
        built = getattr(self._widgets, "built_items", None)
        return built() if callable(built) else list(self._widgets.items())

    def _plan_of(self, key: str) -> Any:
        """The plan a control still waiting is built from, or ``None``."""
        meta_for = getattr(self._widgets, "meta_for", None)
        return meta_for(key) if callable(meta_for) else None

    def _read_value(self, key: str) -> Any:
        """``key``'s value as its control reads, built or not.

        A built control is read. One still waiting for its category is read
        from its plan by :func:`_value_a_plain_control_holds`, which is what
        the control would read back; only plain controls wait. A setting
        without a control reads as ``None``, as ``_read_widget(None)`` did.
        """
        widget = self._built_control(key)
        if widget is not None:
            return self._read_widget(widget)
        plan = self._plan_of(key)
        if plan is not None:
            return _value_a_plain_control_holds(plan)
        return None

    def collect(self) -> Dict[str, Any]:
        """Read all widgets and return the current settings dict.

        A control still waiting for its category to be opened is read
        without being built; see :meth:`_read_value`.
        """
        out: Dict[str, Any] = {}
        for key in self._widgets:
            out[key] = self._canonical(
                key, self._coerce_to_expected_type(key, self._read_value(key)))
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
                    _RegressionBackendField, _MetadataTypeField,
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
        """Update a known run setting whose widget is not on this form.

        This includes dedicated controls outside the form and object rows
        omitted by the current shape.

        Hidden does not mean absent: imported
        values live in ``_defaults`` and still reach ``collect()``.

        A slot
        above the current count is accepted only when this app owns the count
        and the key is a declared setting; foreign-app keys remain rejected.
        """
        if key not in self._defaults:
            from ...organelle_types import (NUMBER_OF_ORGANELLES,
                                            organelle_role_of)
            from ...settings import expected_types

            if (NUMBER_OF_ORGANELLES not in self._defaults
                    or organelle_role_of(key) is None
                    or key not in expected_types):
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
        """Re-grey method-specific Image UMAP controls immediately.

        Then the dependency rules, for the reason
        :meth:`_on_classifier_family_changed` gives.
        """
        self._refresh_umap_reducer_enablement()
        self._refresh_setting_dependencies()

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
        try:
            from ...settings_advisor import UNIT_REQUIREMENTS

            owned = set().union(*(set(v) for v in UNIT_REQUIREMENTS.values()))
        except Exception:                                    # noqa: BLE001
            owned = set(required)
        released = set(getattr(self, "_unit_locked", set()))
        note = (f"Fixed by analysis_unit={unit!r}: the run reads this value "
                f"and no other, so it is shown rather than left editable. "
                f"Choose analysis_unit='well' to set it yourself.")
        for key in sorted(owned):
            widget = self._built_control(key)
            if widget is None:
                continue
            if key in required:
                self.set_value_for_key(key, required[key])
                widget.setEnabled(False)
                _apply_greyed_note(widget, note)
            elif key in released:
                widget.setEnabled(True)
                _clear_greyed_note(widget)
        self._unit_locked = {k for k in required if k in self._widgets}
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
            control = self._built_control(key)
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

        metric = self._built_control("metric")
        if metric is not None:
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
            return

        for key, control in self._built_controls():
            if key in greyed:
                control.setEnabled(False)
                _apply_greyed_note(control, _family_note(family))
            elif key in owned:
                control.setEnabled(True)
                _clear_greyed_note(control)

    def _on_classifier_family_changed(self, *_args) -> None:
        """Re-grey the panel when the classifier family changes.

        The dependency rules run after it, as they do when the panel is
        built: the family pass re-enables every setting the family owns, and
        a setting a rule greys -- ``batch_column`` under
        ``batch_correction='none'`` -- has to stay greyed whichever family is
        chosen. Without this the order of the last two changes decided it,
        and a control built after the change (in a category opened later)
        would disagree with one built before.
        """
        self._refresh_classifier_family_enablement()
        self._refresh_setting_dependencies()

    def _on_training_basis_changed(self, *_args) -> None:
        """Re-grey the panel when the training basis changes.

        A named method rather than a lambda: INVARIANTS 4 is about
        QThread.finished specifically, but the same lifetime reasoning
        applies to any signal connection that has to outlive the call that
        made it. The dependency rules run after it, for the reason
        :meth:`_on_classifier_family_changed` gives.
        """
        self.refresh_training_basis_enablement()
        self._refresh_setting_dependencies()

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
            return

        for key, control in self._built_controls():
            if key in greyed:
                control.setEnabled(False)
                _apply_greyed_note(control, _basis_note(basis))
            elif key in _ALL_BASIS_SETTINGS:
                control.setEnabled(True)
                _clear_greyed_note(control)

    def _current_setting(self, key: str) -> Any:
        """Whatever the panel currently holds for ``key``, or its default.

        Reads the WIDGET when there is one, so a convention tested against a
        folder is tested against the path on screen rather than the one the
        settings file was loaded with.
        """
        if key not in self._widgets:
            return self._defaults.get(key)
        try:
            return self._read_value(key)
        except Exception:                                      # noqa: BLE001
            return self._defaults.get(key)

    def _current_source_folder(self) -> str:
        """The folder ``src`` names right now, as a string.

        ``src`` is a LIST on the screens that take several inputs, so the
        first entry is taken there: testing a filename convention needs one
        folder of raw images and any of them will answer the question.
        """
        value = self._current_setting("src")
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        return "" if value is None else str(value)

    def _refresh_contextual_widgets(self) -> None:
        """Refresh widgets whose choices come from the selected data source."""
        self.refresh_training_basis_enablement()
        self._refresh_setting_dependencies()
        editor = self._built_control("exclude_rows")
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
        """Re-evaluate the dependency rules after a source setting changed.

        :param _args: whatever the emitting widget passes; ignored, since every
            control is re-read either way.
        """
        self._refresh_setting_dependencies()

    def _current_dependency_settings(self) -> Dict[str, Any]:
        """Read every control into a settings dict for the dependency rules.

        :returns: the declared defaults overlaid with whatever each control now
            holds; a control that cannot be read or coerced leaves its default
            in place rather than dropping the key. A control not built yet is
            read without being built (:meth:`_read_value`).
        """
        current = dict(self._defaults)
        for key in self._widgets:
            try:
                current[key] = self._coerce_to_expected_type(
                    key, self._read_value(key))
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
        """Re-apply the row visibility and then grey the rows that stay.

        Visibility goes first: this is the hook ``apply_settings_dict`` calls
        once a settings file has been poured in, and a file that sets
        ``cell_channel`` has to bring the cell rows back on screen with it -- a
        reason written beside a control on a hidden row is a reason nobody can
        read.

        The loaded tables are only scanned when a rule on this panel can
        actually read them; doing it on every combo change of a panel with no
        data-dependent rule is a stall for nothing. A rule that raises leaves
        its control enabled, since refusing a setting because the check broke
        is worse than allowing one that will be rejected later.
        """
        self.refresh_object_visibility()
        dependencies = self._rules_for_this_panel()
        if not dependencies:
            return
        current = self._current_dependency_settings()
        if any('paired_data' in rule.get('sources', ())
               or 'score_data' in rule.get('sources', ())
               or 'count_data' in rule.get('sources', ())
               for rule in dependencies.values()):
            self._data_context = self._plate_context(
                self._loaded_table_paths(current))
        for key, rule in dependencies.items():
            control = self._built_control(key)
            if control is None:
                continue
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
            if key not in self._widgets:
                current[key] = self._defaults.get(key)
                continue
            try:
                current[key] = self._coerce_to_expected_type(
                    key, self._read_value(key))
            except Exception:                                # noqa: BLE001
                current[key] = self._defaults.get(key)
        return current

    def keys_whose_object_the_run_lacks(self) -> set:
        """Return setting keys excluded by the current object configuration.

        The screen calls this before constructing captions and tooltips so
        settings for unavailable object types remain unbuilt. It uses the
        same visibility rule as :meth:`refresh_object_visibility`.

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

    def hide_the_rows_the_grid_speaks_for(self, keys) -> None:
        """Take ``keys`` off the form because a grid now shows them.

        The widgets STAY -- they are what `collect()` reads and what the grid
        writes through to -- so this hides rows rather than dropping them.
        The settings search still indexes them and every check that walks the
        form still finds them holding their values.

        :param keys: the setting keys the grid answers.
        """
        self._hidden_by_the_grid = set(keys or ())
        self.refresh_object_visibility()

    def hide_the_rows_the_mode_leaves_out(self, keys) -> None:
        """Take ``keys`` off the form because the module's mode does not read them.

        Plaque Assay's Plaque and Figure modes read different settings. The
        widgets stay, holding their values, exactly as for
        :meth:`hide_the_rows_the_grid_speaks_for`; only the rows go.

        :param keys: the setting keys the current mode does not read.
        """
        self._hidden_by_the_mode = set(keys or ())
        self.refresh_object_visibility()

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

        EACH ROW IS SET ONCE. The rows the screen's filters hide anyway --
        everything under the Essentials view that is not essential, the rows
        of a switched-off dimension -- are left hidden here instead of being
        shown and then hidden again by the filter that runs next
        (``rows_the_screen_hides``). The rows end where they always ended;
        what goes is the round trip. Measured on Regression under
        Essentials, showing and re-hiding 145 rows was 68 ms of a 90 ms
        pass, run twice every time a category was built.

        The pass ends by calling ``rows_are_filtered_by`` when the screen has
        set it. This pass shows every row its objects allow, so the settings
        search, which also decides rows, has to be applied after it; without
        that, a channel committed in Essentials put non-essential rows back on
        the form and left a newly relevant heading off it.
        """
        if getattr(self, "_applying_settings", False):
            return
        try:
            current = self._object_visibility_settings()
            lacking = set(keys_hidden_by_their_object(self._widgets, current))
            self._hidden_by_their_object = set(lacking)
            hidden = lacking | set(
                getattr(self, "_hidden_by_the_grid", ()) or ()) | set(
                getattr(self, "_hidden_by_the_mode", ()) or ())
            self._hidden_by_the_run = set(hidden)
            lay_out = getattr(self, "rows_are_laid_out_by", None)
            if lay_out is not None:
                try:
                    lay_out(hidden)
                except Exception:                            # noqa: BLE001
                    LOGGER.debug("could not lay out the rows that are back",
                                 exc_info=True)
            also = set()
            ask = getattr(self, "rows_the_screen_hides", None)
            if ask is not None:
                try:
                    also = set(ask() or ())
                except Exception:                            # noqa: BLE001
                    LOGGER.debug("could not ask what the screen hides",
                                 exc_info=True)
                    also = set()
            for key in list(self._widgets):
                self._set_row_visible(
                    key, key not in hidden and key not in also)
            if also:
                self._lay_the_forms_out_again()
            self._guard_hidden_rows(hidden)
            self._hide_the_headings_of_slots_the_run_lacks(current)
        except Exception:                                    # noqa: BLE001
            LOGGER.debug("could not decide which objects are in the run",
                         exc_info=True)
            return
        refilter = getattr(self, "rows_are_filtered_by", None)
        if refilter is not None:
            try:
                refilter()
            except Exception:                                # noqa: BLE001
                LOGGER.debug("could not re-apply the settings filter",
                             exc_info=True)

    def _lay_the_forms_out_again(self) -> None:
        """Ask every settings form to lay itself out again.

        What showing and re-hiding every row used to do as a side effect,
        without the round trip. A field whose height changed while its
        own row stayed put -- Mask's filename-convention box grows a line
        when its example is filled in -- was re-measured only because some
        row on the panel was shown and hidden again; with every row set
        once, nothing asked its form, and it kept the height of the line
        before. Invalidating a form costs one layout pass, not an event per
        widget.
        """
        for section, _keys, _nested in list(
                (getattr(self, "_section_rows", None) or {}).values()):
            try:
                form = getattr(section, "_form", None)
                if isinstance(form, QFormLayout):
                    form.invalidate()
            except RuntimeError:
                continue

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
        cached = getattr(self, "_slot_heading_cache", None)
        if cached:
            return cached
        cache: Dict[int, Tuple[Any, Tuple[str, ...]]] = {}
        if self._parent is None:
            return cache
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
            from ..widgets.section import Section, _sections_below

            by_widget = {id(widget): key
                         for key, widget in self._built_controls()}
            for section in _sections_below(self._parent):
                if (not isinstance(section, Section)
                        or _sections_below(section)):
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

        A heading whose rows all belong to a nucleus or a pathogen whose
        channel names no plane is hidden the same way. Without that, clearing
        the pathogen channel left "Pathogen Segmentation" on the form as a
        heading over no rows. Cell is never gated, so its headings stay.
        """
        from ..preferences import maturity_is_visible
        from ...organelle_types import active_organelle_roles

        headings = self._slot_headings()
        if not headings:
            return
        active = set(active_organelle_roles(settings))
        emptied = self._headings_of_absent_slots
        switched_off: Dict[str, bool] = {}

        def absent(role) -> bool:
            """Whether the run has no ``role`` at all."""
            if role is None or role == "cell":
                return False
            if role not in CHANNELLED_OBJECTS:
                return role not in active
            if role not in switched_off:
                switches = [key for key in object_switch_keys(role)
                            if key in self._widgets]
                switched_off[role] = bool(switches) and not any(
                    _names_a_plane(settings.get(key)) for key in switches)
            return switched_off[role]

        for ident, (section, keys) in headings.items():
            roles = {object_of_setting(key) for key in keys}
            gone = bool(roles) and all(absent(role) for role in roles)
            try:
                if gone:
                    if not section.isHidden():
                        emptied[ident] = section
                        section.setVisible(False)
                        section.installEventFilter(self._object_row_guard)
                elif ident in emptied:
                    del emptied[ident]
                    if maturity_is_visible(section.maturity()):
                        section.setVisible(True)
            except RuntimeError:
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
            widget = self._built_control(key)
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
        return

    def _reassert_object_visibility(self) -> None:
        """Run the queued object-visibility pass and clear the queue flag."""
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
        widget = self._built_control(key)
        if widget is None:
            return
        from ..settings_search import _set_row_visible as set_row

        node = widget
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
        if widget.parentWidget() is None:
            return
        widget.setVisible(visible)
        label = getattr(widget, "_spacr_setting_label", None)
        if label is not None:
            label.setVisible(visible)

    def _connect_object_visibility_signals(self) -> None:
        """Follow the three committed values that narrow an organelle slot.

        Object channels and slot counts are deliberately NOT connected here.
        A channel is typed character by character, and the former connection
        ran a whole 1,551-row visibility pass per keystroke. ``AppScreen``
        watches their committed values and rebuilds the optimized form once.

        Type and morphology are closed choices, however, and diameter is
        relevant when a size-dependent preset is selected. Those values can
        change which already-owned detection rows apply, so one user choice
        refreshes them in place. Diameter waits for ``editingFinished`` where
        available; it never rearranges the panel while a number is typed.
        """
        if getattr(self, "_object_visibility_signals_connected", False):
            return
        self._object_visibility_signals_connected = True
        from ...organelle_types import ORGANELLE_TYPES, slot_setting

        primary_targets = {"organelle_morphology", "organelle_method"}
        for preset in ORGANELLE_TYPES.values():
            primary_targets.update(preset.params)

        roles = {object_of_setting(key) for key in self._widgets}
        roles = {role for role in roles
                 if role is not None and role not in CHANNELLED_OBJECTS}
        for role in roles:
            recommended = self._organelle_recommendations(role)
            owned = self._organelle_preset_owned.setdefault(role, {})
            for key, value in recommended.items():
                if self._setting_value_equals(key, value):
                    owned[key] = value

            type_widget = self._widgets.get(f"{role}_type")
            if type_widget is not None:
                _connect_value_changed(
                    type_widget,
                    partial(self._on_organelle_type_changed, role))

            diameter = self._widgets.get(f"{role}_diameter")
            if diameter is not None:
                changed = partial(self._on_organelle_diameter_changed, role)
                committed = getattr(diameter, "editingFinished", None)
                if committed is not None:
                    try:
                        committed.connect(changed)
                    except Exception:                        # noqa: BLE001
                        _connect_value_changed(diameter, changed)
                else:
                    _connect_value_changed(diameter, changed)

            for primary in primary_targets:
                key = slot_setting(primary, role)
                widget = self._widgets.get(key)
                if widget is None:
                    continue
                _connect_value_changed(
                    widget,
                    partial(self._on_organelle_preset_target_changed,
                            role, key))

    def _on_object_switch_changed(self, *_args) -> None:
        """Refresh rows after one slot-narrowing value is committed."""
        self.refresh_object_visibility()

    def _setting_value(self, key: str) -> Any:
        """Return one widget value with the same coercion as ``collect``."""
        if key not in self._widgets:
            return self._defaults.get(key)
        try:
            return self._coerce_to_expected_type(key, self._read_value(key))
        except Exception:                                    # noqa: BLE001
            return self._defaults.get(key)

    def _setting_value_equals(self, key: str, expected: Any) -> bool:
        """Compare a setting's current value with an expected one.

        :param key: the setting to read.
        :param expected: what to compare against.
        :returns: ``True`` when they match under the settings-diff comparison,
            falling back to ``==`` and finally to ``False`` -- an unreadable
            setting is not equal to anything.
        """
        try:
            from ..settings_diff import _values_equal

            return bool(_values_equal(self._setting_value(key), expected))
        except Exception:                                    # noqa: BLE001
            try:
                return bool(self._setting_value(key) == expected)
            except Exception:                                # noqa: BLE001
                return False

    def _organelle_recommendations(self, role: str) -> Dict[str, Any]:
        """Return the selected type's recommendations in this slot's keys."""
        from ...organelle_types import preset_for, slot_setting

        try:
            recommended = preset_for(
                self._setting_value(f"{role}_type"),
                self._setting_value(f"{role}_diameter"),
            )
        except (TypeError, ValueError):
            return {}
        return {slot_setting(key, role): value
                for key, value in recommended.items()}

    def _apply_organelle_recommendations(
            self, role: str, *, overwrite: bool) -> None:
        """Write one preset into widgets while preserving diameter overrides."""
        recommended = self._organelle_recommendations(role)
        previous = dict(self._organelle_preset_owned.get(role, {}))
        now_owned: Dict[str, Any] = {}
        self._applying_organelle_preset = True
        try:
            for key, value in recommended.items():
                may_write = overwrite or (
                    key in previous
                    and self._setting_value_equals(key, previous[key])
                )
                if not may_write or not self.set_value_for_key(key, value):
                    continue
                now_owned[key] = value
        finally:
            self._applying_organelle_preset = False
        self._organelle_preset_owned[role] = now_owned

    def _on_organelle_type_changed(self, role: str, *_args) -> None:
        """A deliberate type choice populates its actual execution values."""
        if getattr(self, "_applying_settings", False):
            return
        self._apply_organelle_recommendations(role, overwrite=True)
        self.refresh_object_visibility()

    def _on_organelle_diameter_changed(self, role: str, *_args) -> None:
        """Update only size-dependent values the preset still owns."""
        if getattr(self, "_applying_settings", False):
            return
        self._apply_organelle_recommendations(role, overwrite=False)
        self.refresh_object_visibility()

    def _on_organelle_preset_target_changed(
            self, role: str, key: str, *_args) -> None:
        """Mark an advanced edit as the user's, then refresh morphology rows."""
        if (getattr(self, "_applying_settings", False)
                or self._applying_organelle_preset):
            return
        self._organelle_preset_owned.setdefault(role, {}).pop(key, None)
        if key == f"{role}_morphology":
            self.refresh_object_visibility()

    def apply_organelle_presets_from_mapping(
            self, settings: Dict[str, Any]) -> None:
        """Apply sparse imported presets without replacing explicit values.

        A settings file that supplies morphology/method/thresholds owns those
        values. A file that supplies only a type asks the picker to populate
        its missing recommendations just as a direct user choice does.
        """
        from ...organelle_types import organelle_role_of

        supplied = {str(key) for key in settings}
        roles = {organelle_role_of(key) for key in supplied}
        roles.discard(None)
        for role in roles:
            owned = self._organelle_preset_owned.setdefault(role, {})
            for key in supplied:
                if organelle_role_of(key) == role:
                    owned.pop(key, None)
            if f"{role}_type" not in supplied:
                if f"{role}_diameter" in supplied:
                    self._apply_organelle_recommendations(
                        role, overwrite=False)
                continue
            recommended = self._organelle_recommendations(role)
            self._applying_organelle_preset = True
            try:
                for key, value in recommended.items():
                    if key in supplied and settings.get(key) is not None:
                        continue
                    if (self.set_value_for_key(key, value)
                            or self.set_hidden_value(key, value)):
                        owned[key] = value
            finally:
                self._applying_organelle_preset = False

    def _read_widget(self, w: QWidget) -> Any:
        """Read one control's value in the form the settings dict expects.

        A combo's ``userData`` is authoritative, not its caption: every item is
        added with its option as data, including the Python ``None`` option, so
        ``currentData()`` returning ``None`` means the chosen option *is*
        ``None``. Falling back to the caption is what shipped
        ``strict_errors='None'`` -- a non-empty string, and therefore truthy --
        turning strict error handling silently on. The caption is still right
        for an editable combo showing text the user typed, which is detected by
        the displayed text differing from the current item's.

        :param w: the control to read.
        :returns: its value, or ``None`` for a control kind this does not know.
        """
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
                _RegressionBackendField, _MetadataTypeField,
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



def _widget_is_alive(widget) -> bool:
    """Whether ``widget``'s C++ half still exists.

    The same check `spacr.qt.live_zoom._alive` makes, and for the same
    reason one layer down: a Python wrapper outlives the object it wraps,
    and reading through it is undefined rather than an exception.

    :param widget: any Qt object, or None.
    :returns: True when it is safe to touch.
    """
    if widget is None:
        return False
    try:
        from shiboken6 import isValid
        return bool(isValid(widget))
    except Exception:                                        # noqa: BLE001
        try:
            widget.objectName()
            return True
        except RuntimeError:
            return False

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
    if field.property(DISABLED_REASON_TOOLTIP):
        return None
    parent = field.parentWidget()
    root = parent.layout() if parent is not None else None
    if root is None:
        return None
    layout, index = _owning_layout(root, field)
    if layout is None:
        return None

    def _named(widget) -> Optional[QWidget]:
        """The real label inside a row, unwrapping a host if there is one.

        `Section.add_row` wraps the caption in a `SettingLabelWithInfo` whenever
        the row is right-aligned against its field -- the form's normal shape --
        so the layout hands back a plain QWidget and a bare isinstance rejects
        it. Measured on Mask: 1,541 of 1,657 rows kept their help on the FIELD
        for this reason alone, and only 13 labels had it.
        """
        if widget is None or not _widget_is_alive(widget):
            return None
        widget = _unwrap_setting_label(widget)
        if widget is None or not _widget_is_alive(widget):
            return None
        try:
            if not (isinstance(widget, QLabel) and widget.text().strip()):
                return None
        except RuntimeError:
            return None
        try:
            if widget.cursor().shape() == Qt.PointingHandCursor:
                return None
        except RuntimeError:
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
        candidate_index = index - 1
        if layout.stretch(candidate_index) > 0:
            return None
        item = layout.itemAt(candidate_index)
        return _named(item.widget()) if item is not None else None
    return None


def _is_a_settings_field(widget: QWidget) -> bool:
    """Whether ``widget`` is a setting's editor, whose help belongs on a name.

    :param widget: any widget found under the panel.
    :returns: ``True`` when its tooltip should move to its row's label.

    A TYPE LIST ALONE WAS TOO NARROW. It named the six Qt editors, and the
    settings form is largely spaCR's own controls: measured on the Mask
    screen, 27 ``Toggle`` rows and 3 ``_ListEditor`` rows each had a real
    name beside them in the form and kept their help on the control anyway,
    because neither type is a ``QLineEdit``. Carrying a ``settingKey`` is
    the definitive mark of "this widget is a setting's field", whatever it
    was built from.

    A CONTROL THAT IS ITS OWN LABEL KEEPS ITS HELP. A checkbox or button
    with visible text of its own has no separate name to move the help to --
    hovering its text IS hovering its name -- and taking the tooltip off it
    would leave that setting with no help anywhere.
    """
    from PySide6.QtWidgets import QAbstractButton

    if isinstance(widget, _EDITOR_TYPES):
        return True
    if not widget.property("settingKey"):
        return False
    if isinstance(widget, QAbstractButton) and (widget.text() or "").strip():
        return False
    return True


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
    event_filter = getattr(root, "_api_tooltip_filter", None)
    if event_filter is None:
        event_filter = _ApiTooltipFilter(root)
        root._api_tooltip_filter = event_filter

    moved = 0
    for field in root.findChildren(QWidget):
        if field.property("settingHelpLabel"):
            continue
        if not _is_a_settings_field(field):
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
            continue
        key = str(field.property("settingKey") or "")
        app_key = str(field.property("settingsAppKey") or "")
        if key and not app_key:
            app_key = str(getattr(root, "app_key", "") or "")

        display_tip = tip
        if app_key and key:
            source = str(
                field.property("apiTooltipDescriptionSource")
                or field.property("apiTooltipDescription")
                or tip
            )
            html = str(field.property("apiTooltipHtml") or "")
            display_tip = (html if "href=" in html
                           else format_tooltip(source, app_key, key))
            field.setProperty("settingsAppKey", app_key)
            field.setProperty("apiTooltipDescriptionSource", source)
            field.setProperty("apiTooltipDescription", source)
            field.setProperty("apiTooltipHtml", display_tip)

        if not existing:
            label.setToolTipDuration(-1)
            label.setCursor(Qt.WhatsThisCursor)
        label.setToolTip(display_tip)
        label.setProperty("apiTooltipHtml", display_tip)
        label.setProperty(
            "apiTooltipDisplayRole",
            "tooltip" if app_key and key else "hover-help",
        )
        label.setProperty("settingHelpLabel", True)
        for prop in ("settingsAppKey", "settingKey", "settingAnimationKey",
                     "apiTooltipDescriptionSource", "apiTooltipDescription"):
            carried = field.property(prop)
            if carried:
                label.setProperty(prop, carried)
        label.removeEventFilter(event_filter)
        label.installEventFilter(event_filter)
        field.setToolTip("")
        field.setProperty("apiTooltipDisplayRole", "metadata")
        field.removeEventFilter(event_filter)
        moved += 1
    return moved
