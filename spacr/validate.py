"""Pre-flight validation of a spaCR settings dict.

Every crash this module exists to prevent was, in practice, a twenty-line
check away: an ``organelle_channel`` one past the end of a three-channel
plate, a ``src`` that points at the plate folder instead of ``plate/merged``,
a ``cell_mask_dim`` beyond the last plane of the merged array, an integer
that came back from a settings CSV as the string ``"4"``. Each of those costs
a full GPU run to discover.

The module is deliberately dependency-light: it imports nothing from spaCR
except :mod:`spacr.settings` (which itself imports only ``os`` and ``ast``),
and it touches ``numpy`` only lazily, to read the *header* of a single
``.npy`` file. No torch, no cellpose, no image decoding. Importing and
running it costs well under a second, which is the whole point — it is what
``dry_run`` uses to answer "would this run work?" before anything is
allocated, loaded or written.

Public API
----------
``Problem``
    One thing that is wrong, with the fix.
``validate_settings(settings, app_key)``
    Returns a list of :class:`Problem`, errors and warnings mixed.
``format_report(problems, settings, app_key)``
    Human-readable report; errors first, each with its fix line.
``describe_plan(settings, app_key)``
    "Here is what would actually happen" summary.

Rules are derived from the code that consumes the settings, not invented:
each check names its source in a comment.
"""
from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Problem",
    "validate_settings",
    "format_report",
    "describe_plan",
    "ERROR",
    "WARNING",
    "APP_FUNCTIONS",
]

ERROR = "error"
WARNING = "warning"


@dataclass
class Problem:
    """One thing wrong with a settings dict.

    :param severity: ``"error"`` (the run would fail or silently produce
        wrong output) or ``"warning"`` (suspicious, but runnable).
    :param setting: the settings key at fault, or ``""`` when the problem is
        about the dataset rather than a single key.
    :param message: what is wrong, phrased in the user's terms.
    :param fix: what to actually do about it.
    """

    severity: str
    setting: str
    message: str
    fix: str

    @property
    def is_error(self) -> bool:
        """True when this problem would break or corrupt the run."""
        return self.severity == ERROR

    def __str__(self) -> str:
        head = f"[{self.setting}] {self.message}" if self.setting else self.message
        return f"{head}\n    fix: {self.fix}"


# ---------------------------------------------------------------------------
# app registry
# ---------------------------------------------------------------------------

# The names are the ``settings_type`` strings dispatched by
# spacr.gui_utils.run_function_gui, so a caller can pass the same key the GUI
# uses. Values are the function that would run.
APP_FUNCTIONS: Dict[str, str] = {
    "mask": "spacr.core.preprocess_generate_masks",
    "measure": "spacr.measure.measure_crop",
    "classify": "spacr.deep_spacr.deep_spacr",
    "umap": "spacr.core.generate_image_umap",
    "train_cellpose": "spacr.submodules.train_cellpose",
    "ml_analyze": "spacr.ml.generate_ml_scores",
    "cellpose_masks": "spacr.spacr_cellpose.identify_masks_finetune",
    "cellpose_all": "spacr.spacr_cellpose.check_cellpose_models",
    "map_barcodes": "spacr.sequencing.generate_barecode_mapping",
    "regression": "spacr.ml.perform_regression",
    "recruitment": "spacr.submodules.analyze_recruitment",
    "analyze_plaques": "spacr.submodules.analyze_plaques",
    "convert": "spacr.io.process_non_tif_non_2D_images",
    "simulation": "spacr.sim.run_multiple_simulations",
}

# Friendly spellings a caller (or a notebook) might reasonably use.
APP_ALIASES: Dict[str, str] = {
    "sequencing": "map_barcodes",
    "barcodes": "map_barcodes",
    "barcode_mapping": "map_barcodes",
    "preprocess_generate_masks": "mask",
    "generate_masks": "mask",
    "measure_crop": "measure",
    "deep_spacr": "classify",
    "train": "classify",
    "generate_image_umap": "umap",
    "embedding": "umap",
}

# Apps whose ``src`` is a plate folder that must already contain
# measurements/measurements.db — see spacr.ml.perform_regression
# (``src + '/measurements/measurements.db'``), spacr.submodules
# .analyze_recruitment and spacr.io._read_and_join_tables.
DB_APPS = frozenset({"umap", "ml_analyze", "regression", "recruitment", "activation", "classify"})

# Apps that read the merged/*.npy stacks produced by the mask pipeline.
MERGED_APPS = frozenset({"measure"})

CHANNEL_KEYS: Tuple[str, ...] = (
    "cell_channel",
    "nucleus_channel",
    "pathogen_channel",
    "organelle_channel",
)

MASK_DIM_KEYS: Tuple[str, ...] = (
    "cell_mask_dim",
    "nucleus_mask_dim",
    "pathogen_mask_dim",
    "organelle_mask_dim",
)

OBJECT_NAMES = ("cell", "nucleus", "pathogen", "organelle")

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

# Mirrors spacr.utils._get_regex. Reproduced here (rather than imported)
# because spacr.utils pulls in torch, which would defeat the point of a
# one-second pre-flight check. The trailing extension group replaces the
# ``.{img_format}`` suffix that _get_regex interpolates.
_EXT_SUFFIX = r"\.(?:tif|tiff|png|jpg|jpeg|bmp)$"
METADATA_REGEXES: Dict[str, str] = {
    "cellvoyager": (
        r"(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)"
        r"L(?P<laserID>..)A(?P<AID>..)Z(?P<sliceID>.*)C(?P<chanID>.*)" + _EXT_SUFFIX
    ),
    "cq1": (
        r"W(?P<wellID>.*)F(?P<fieldID>.*)T(?P<timeID>.*)Z(?P<sliceID>.*)"
        r"C(?P<chanID>.*)" + _EXT_SUFFIX
    ),
    "auto": (
        r"(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)"
        r"L(?P<laserID>.*)C(?P<chanID>.*)" + _EXT_SUFFIX
    ),
}


def _normalize_app(app_key: Any) -> str:
    """Canonicalize a caller-supplied app key; unknown keys pass through."""
    if not isinstance(app_key, str):
        return ""
    key = app_key.strip().lower()
    return APP_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# known-key universe (for typo detection)
# ---------------------------------------------------------------------------

_KNOWN_KEYS_CACHE: Optional[frozenset] = None


def _known_setting_keys() -> frozenset:
    """Every settings key spaCR knows about.

    Built from ``expected_types``, ``tooltips`` and ``categories``, plus every
    key any ``set_default_*`` / ``get_*_settings`` helper in
    :mod:`spacr.settings` produces. The helpers are pure dict-fillers (the
    whole sweep costs well under a millisecond), so calling them is cheaper
    and far less rot-prone than maintaining a hand-written list.
    """
    global _KNOWN_KEYS_CACHE
    if _KNOWN_KEYS_CACHE is not None:
        return _KNOWN_KEYS_CACHE

    keys = set()
    from . import settings as _settings

    keys.update(getattr(_settings, "expected_types", {}))
    keys.update(getattr(_settings, "tooltips", {}))
    for group in getattr(_settings, "categories", {}).values():
        if isinstance(group, (list, tuple, set)):
            keys.update(k for k in group if isinstance(k, str))

    import contextlib
    import io as _io

    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        for name, fn in list(vars(_settings).items()):
            if not callable(fn):
                continue
            if not (name.startswith("set_") or name.startswith("get_")
                    or name.startswith("default_") or name.startswith("deep_")):
                continue
            try:
                produced = fn({})
            except Exception:
                try:
                    produced = fn()
                except Exception:
                    continue
            if isinstance(produced, dict):
                keys.update(k for k in produced if isinstance(k, str))

    _KNOWN_KEYS_CACHE = frozenset(keys)
    return _KNOWN_KEYS_CACHE


# ---------------------------------------------------------------------------
# dataset inspection
# ---------------------------------------------------------------------------


@dataclass
class _Inventory:
    """What a single ``src`` actually contains, established without loading pixels."""

    src: str = ""
    exists: bool = False
    is_dir: bool = False
    is_db_file: bool = False

    merged_dir: Optional[str] = None
    merged_exists: bool = False
    merged_files: int = 0

    stack_dir: Optional[str] = None
    stack_files: int = 0

    # last-axis length of one merged/ (or stack/) array: image channels plus
    # any appended mask planes. This is what *_mask_dim indexes into.
    array_planes: Optional[int] = None
    array_source: str = ""

    # number of raw acquisition channels. This is what *_channel indexes into.
    raw_channels: Optional[int] = None
    raw_channel_ids: Tuple[str, ...] = ()
    raw_evidence: str = ""

    raw_files: int = 0
    fields: Optional[int] = None
    regex_used: str = ""

    db_path: Optional[str] = None
    db_exists: bool = False


def _listdir(path: Optional[str]) -> List[str]:
    """``os.listdir`` that returns [] instead of raising on a bad path."""
    if not path:
        return []
    try:
        return os.listdir(path)
    except OSError:
        return []


def _resolve_merged_dir(src: str) -> str:
    """Where measure_crop would look for the merged arrays.

    Mirrors spacr.measure.measure_crop, which appends ``merged`` unless
    ``os.path.basename(src)`` already ends with it.
    """
    if os.path.basename(os.path.normpath(src)).endswith("merged"):
        return src
    return os.path.join(src, "merged")


def _peek_planes(directory: str) -> Tuple[Optional[int], str, int]:
    """Read the shape of ONE ``.npy`` in ``directory`` without loading its data.

    :returns: ``(planes, example_filename, npy_count)``. ``planes`` is the
        length of the last axis (1 for a 2-D array), or None when nothing
        could be read.
    """
    npys = sorted(f for f in _listdir(directory) if f.endswith(".npy"))
    if not npys:
        return None, "", 0
    import numpy as np  # local: keeps module import free of numpy's cost

    for name in npys[:3]:
        try:
            arr = np.load(os.path.join(directory, name), mmap_mode="r")
        except (OSError, ValueError, EOFError):
            continue
        shape = tuple(getattr(arr, "shape", ()))
        if not shape:
            continue
        planes = int(shape[-1]) if len(shape) >= 3 else 1
        return planes, name, len(npys)
    return None, "", len(npys)


def _candidate_patterns(settings: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``(label, pattern)`` regexes to try against raw filenames, best guess first."""
    metadata_type = settings.get("metadata_type", "cellvoyager")
    custom = settings.get("custom_regex")
    out: List[Tuple[str, str]] = []
    if isinstance(custom, str) and custom.strip():
        out.append(("custom_regex", custom))
        out.append(("custom_regex+ext", custom.rstrip("$") + _EXT_SUFFIX))
    if isinstance(metadata_type, str) and metadata_type in METADATA_REGEXES:
        out.append((metadata_type, METADATA_REGEXES[metadata_type]))
    for label, pattern in METADATA_REGEXES.items():
        if all(label != existing for existing, _ in out):
            out.append((label, pattern))
    return out


def _scan_raw_images(src: str, settings: Dict[str, Any], inv: _Inventory) -> None:
    """Fill ``inv`` with the channel/field counts implied by raw image filenames.

    Channel identity comes from the ``chanID`` group of the metadata regex,
    exactly as spacr.utils._extract_filename_metadata reads it; the stack
    written by spacr.io._rename_and_organize_image_files concatenates one
    plane per distinct chanID in sorted order, so the number of distinct
    chanIDs *is* the number of valid zero-based channel indices.
    """
    search_dirs = [src, os.path.join(src, "orig"), os.path.join(src, "consolidated")]
    names: List[str] = []
    for directory in search_dirs:
        found = [f for f in _listdir(directory) if f.lower().endswith(IMAGE_EXTENSIONS)]
        if found:
            names = found
            break
    inv.raw_files = len(names)
    if not names:
        return

    best_label = ""
    best_channels: List[str] = []
    best_fields = 0
    best_hits = 0
    for label, pattern in _candidate_patterns(settings):
        try:
            rx = re.compile(pattern)
        except re.error:
            continue
        channels = set()
        fields = set()
        hits = 0
        for name in names:
            match = rx.match(name)
            if not match:
                continue
            groups = match.groupdict()
            if "chanID" not in groups or groups["chanID"] is None:
                continue
            hits += 1
            channels.add(str(groups["chanID"]))
            fields.add((
                str(groups.get("plateID") or ""),
                str(groups.get("wellID") or ""),
                str(groups.get("fieldID") or ""),
                str(groups.get("timeID") or ""),
            ))
        if hits > best_hits:
            best_hits, best_label = hits, label
            best_channels = sorted(channels)
            best_fields = len(fields)

    if best_hits:
        inv.raw_channels = len(best_channels)
        inv.raw_channel_ids = tuple(best_channels)
        inv.fields = best_fields or None
        inv.regex_used = best_label
        inv.raw_evidence = (
            f"{best_hits} raw image files parsed with the '{best_label}' filename pattern"
        )


def _inventory(src: Any, settings: Dict[str, Any], app: str) -> _Inventory:
    """Establish what is on disk for one ``src``, cheaply."""
    inv = _Inventory()
    if not isinstance(src, str):
        return inv
    inv.src = src
    inv.exists = os.path.exists(src)
    inv.is_dir = os.path.isdir(src)
    inv.is_db_file = inv.exists and not inv.is_dir and src.endswith(".db")
    if not inv.is_dir:
        return inv

    inv.merged_dir = _resolve_merged_dir(src)
    inv.merged_exists = os.path.isdir(inv.merged_dir)
    if inv.merged_exists:
        planes, example, count = _peek_planes(inv.merged_dir)
        inv.merged_files = count
        if planes is not None:
            inv.array_planes = planes
            inv.array_source = os.path.join(os.path.basename(inv.merged_dir), example)

    inv.stack_dir = os.path.join(src, "stack")
    if os.path.isdir(inv.stack_dir):
        planes, example, count = _peek_planes(inv.stack_dir)
        inv.stack_files = count
        if planes is not None:
            # stack/ holds image channels only (masks are appended later, in
            # merged/) — see spacr.io._load_and_concatenate_arrays.
            inv.raw_channels = planes
            inv.raw_channel_ids = tuple(str(i) for i in range(planes))
            inv.raw_evidence = f"stack/{example} has {planes} channel planes"
            if inv.array_planes is None:
                inv.array_planes = planes
                inv.array_source = os.path.join("stack", example)

    if inv.raw_channels is None:
        _scan_raw_images(src, settings, inv)

    db_root = os.path.dirname(os.path.normpath(src)) if os.path.basename(
        os.path.normpath(src)).endswith("merged") else src
    inv.db_path = os.path.join(db_root, "measurements", "measurements.db")
    inv.db_exists = os.path.isfile(inv.db_path)
    return inv


def _src_values(settings: Dict[str, Any]) -> List[Any]:
    """``src`` normalized to a list, mirroring spacr.utils.normalize_src_path."""
    src = settings.get("src")
    if isinstance(src, (list, tuple)):
        return list(src)
    if isinstance(src, str):
        stripped = src.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            import ast

            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return [src]
            if isinstance(parsed, list):
                return parsed
        return [src]
    return [src]


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------


def _check_src(settings: Dict[str, Any], app: str, inventories: Sequence[_Inventory]) -> List[Problem]:
    """``src`` exists, is the right kind of thing, and holds what the app needs."""
    problems: List[Problem] = []
    if "src" not in settings:
        # spacr.core.preprocess_generate_masks raises ValueError('src is a
        # required parameter').
        return [Problem(ERROR, "src", "src is missing from the settings.",
                        "Set src to the folder holding the images (or, for measure, the merged folder).")]

    raw = settings.get("src")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return [Problem(ERROR, "src", "src is empty.",
                        "Set src to the folder holding the images (or, for measure, the merged folder).")]

    for value in _src_values(settings):
        if not isinstance(value, str):
            problems.append(Problem(
                ERROR, "src",
                f"src entry {value!r} is a {type(value).__name__}, not a path string.",
                "src must be a path string or a list of path strings."))
            continue
        elif value in ("path", "/path/to/src"):
            problems.append(Problem(
                ERROR, "src",
                f"src is still the placeholder {value!r} that the defaults ship with.",
                "Replace it with the real folder you want to process."))

    for inv in inventories:
        if not isinstance(inv.src, str) or not inv.src:
            continue
        if not inv.exists:
            problems.append(Problem(
                ERROR, "src", f"src does not exist: {inv.src}",
                "Check the path for typos, and that the drive or share is mounted."))
            continue
        if inv.is_db_file:
            continue
        if not inv.is_dir:
            problems.append(Problem(
                ERROR, "src", f"src is a file, not a folder: {inv.src}",
                "Point src at the folder that contains the images."))
            continue

        if app in MERGED_APPS:
            # measure_crop lists settings['src'] for *.npy — an absent or
            # empty merged/ means it silently processes zero files.
            if not inv.merged_exists:
                problems.append(Problem(
                    ERROR, "src",
                    f"no merged folder for measure: {inv.merged_dir} does not exist.",
                    "Run the Mask module on this plate first; measure reads the merged/*.npy it writes."))
            elif inv.merged_files == 0:
                problems.append(Problem(
                    ERROR, "src",
                    f"{inv.merged_dir} exists but contains no .npy arrays.",
                    "Re-run the Mask module: merged/ is written at the end of mask generation and is empty here."))
        elif app == "mask":
            if inv.raw_files == 0 and inv.stack_files == 0 and inv.merged_files == 0:
                problems.append(Problem(
                    ERROR, "src",
                    f"no image files found in {inv.src} (looked for {', '.join(IMAGE_EXTENSIONS)}).",
                    "Point src at the folder that holds the raw acquisition images, or set consolidate=True to gather them from subfolders."))
            elif inv.raw_files and inv.raw_channels is None:
                problems.append(Problem(
                    WARNING, "metadata_type",
                    f"{inv.raw_files} image files found in {inv.src}, but none match the "
                    f"'{settings.get('metadata_type', 'cellvoyager')}' filename pattern.",
                    "Set metadata_type to match your microscope ('cellvoyager', 'cq1', 'auto'), or supply custom_regex with wellID/fieldID/chanID groups."))

        if app in DB_APPS and not inv.db_exists:
            problems.append(Problem(
                ERROR, "src", f"measurements database not found: {inv.db_path}",
                "Run the Measure module on this plate first — the analysis apps read measurements/measurements.db."))

    return problems


def _as_int(value: Any) -> Optional[int]:
    """Return ``value`` as an int when it unambiguously is one, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
    return None


def _check_channels(settings: Dict[str, Any], app: str, inventories: Sequence[_Inventory]) -> List[Problem]:
    """Channel indices are in range, and don't collide.

    ``*_channel`` indexes the raw acquisition channels; ``*_mask_dim`` indexes
    the last axis of the merged array (image channels plus the appended mask
    planes) — see spacr.io._load_and_concatenate_arrays.
    """
    problems: List[Problem] = []

    n_raw = next((inv.raw_channels for inv in inventories if inv.raw_channels is not None), None)
    raw_evidence = next((inv.raw_evidence for inv in inventories if inv.raw_channels is not None), "")
    n_planes = next((inv.array_planes for inv in inventories if inv.array_planes is not None), None)
    plane_evidence = next((inv.array_source for inv in inventories if inv.array_planes is not None), "")

    for key in CHANNEL_KEYS:
        if key not in settings:
            continue
        value = settings[key]
        if value is None:
            continue
        index = _as_int(value)
        if index is None:
            continue  # the type check reports this
        if index < 0:
            problems.append(Problem(
                ERROR, key, f"{key}={value} is negative.",
                f"Channel indices are zero-based; use 0"
                + (f"-{n_raw - 1}" if n_raw else "") + ", or None to skip this object."))
            continue
        if n_raw is not None and index >= n_raw:
            problems.append(Problem(
                ERROR, key,
                f"{key}={index} but the dataset has only {n_raw} channel"
                f"{'' if n_raw == 1 else 's'} ({raw_evidence}); valid indices are "
                f"0-{n_raw - 1}.",
                f"Set {key} to a value between 0 and {n_raw - 1}, or to None if that stain was not acquired."))
        elif n_raw is None and n_planes is not None and index >= n_planes:
            problems.append(Problem(
                ERROR, key,
                f"{key}={index} is past the end of the stored arrays, which have "
                f"{n_planes} planes ({plane_evidence}).",
                f"Set {key} to at most {n_planes - 1}, or to None if that stain was not acquired."))

    for key in MASK_DIM_KEYS:
        if key not in settings:
            continue
        value = settings[key]
        if value is None:
            continue
        index = _as_int(value)
        if index is None:
            continue
        if index < 0:
            problems.append(Problem(
                ERROR, key, f"{key}={value} is negative.",
                "Mask dims are zero-based positions along the last axis of merged/*.npy; use None to skip the object."))
            continue
        if n_planes is not None and index >= n_planes:
            problems.append(Problem(
                ERROR, key,
                f"{key}={index} is past the end of the merged arrays, which have "
                f"{n_planes} planes ({plane_evidence}); valid positions are 0-{n_planes - 1}.",
                f"With {n_planes} planes the masks sit at the top of the stack. "
                f"Set {key} within 0-{n_planes - 1}, or None to skip that object."))

    channels = settings.get("channels")
    if isinstance(channels, (list, tuple)) and n_planes is not None:
        bad = [c for c in channels if (_as_int(c) is not None and _as_int(c) >= n_planes)]
        if bad:
            problems.append(Problem(
                ERROR, "channels",
                f"channels {bad} are past the end of the stored arrays, which have "
                f"{n_planes} planes ({plane_evidence}).",
                f"Reduce channels to indices within 0-{n_planes - 1}."))

    # Collisions. Two objects segmented from the same stain is unusual but
    # occasionally deliberate, so it is a warning; two objects reading the
    # same *mask* plane always produces duplicate labels, so it is an error.
    problems.extend(_collision_problems(settings, CHANNEL_KEYS, WARNING,
                                        "are both assigned channel",
                                        "Give each object its own acquisition channel, unless you really mean to segment both from the same stain."))
    problems.extend(_collision_problems(settings, MASK_DIM_KEYS, ERROR,
                                        "both read mask plane",
                                        "Each object needs its own mask plane; with four image channels the masks land at 4, 5, 6, 7 in the order cell, nucleus, pathogen, organelle."))
    return problems


def _collision_problems(settings: Dict[str, Any], keys: Sequence[str],
                        severity: str, phrase: str, fix: str) -> List[Problem]:
    """Report keys in ``keys`` that share an index value."""
    by_value: Dict[int, List[str]] = {}
    for key in keys:
        index = _as_int(settings.get(key))
        if index is None or index < 0:
            continue
        by_value.setdefault(index, []).append(key)
    out: List[Problem] = []
    for index, sharers in sorted(by_value.items()):
        if len(sharers) > 1:
            out.append(Problem(
                severity, ", ".join(sharers),
                f"{' and '.join(sharers)} {phrase} {index}.", fix))
    return out


def _type_name(expected: Any) -> str:
    """Render an ``expected_types`` entry as readable prose."""
    if isinstance(expected, tuple):
        return " or ".join(
            "None" if t is type(None) else getattr(t, "__name__", str(t))
            for t in expected)
    return getattr(expected, "__name__", str(expected))


# Keys whose expected_types entry is narrower than the code that reads them.
# Enforcing the literal declaration here would reject correct settings.
_EXPECTED_TYPE_OVERRIDES: Dict[str, Any] = {
    # The expected_types literal declares "src" twice; the second entry (str)
    # shadows the first ((str, list)), but core.preprocess_generate_masks and
    # measure.measure_crop both loop over a list of folders.
    "src": (str, list),
    # Declared bool for the mask pipeline, but measure_crop *requires* a
    # [lower, upper] percentile pair and refuses a bare True.
    "normalize": (bool, list),
    # core.preprocess_generate_masks expands a bool into [save]*3 itself, so
    # either form arrives legitimately.
    "save": (bool, list),
}


def _check_types(settings: Dict[str, Any]) -> List[Problem]:
    """Values match ``spacr.settings.expected_types``.

    A settings CSV round-trip is the usual culprit: every value comes back a
    string, so ``cell_mask_dim`` arrives as ``'4'`` and measure_crop bails out
    with "must all be integers".
    """
    from .settings import expected_types

    problems: List[Problem] = []
    for key, value in settings.items():
        if key not in expected_types:
            continue
        expected = _EXPECTED_TYPE_OVERRIDES.get(key, expected_types[key])
        types = expected if isinstance(expected, tuple) else (expected,)
        if value is None:
            # None means "skip this object / leave it unset" nearly everywhere
            # in spaCR, and expected_types declares NoneType for only some of
            # the keys that accept it, so flagging None would be pure noise.
            continue
        if isinstance(value, tuple) and list in types:
            continue
        if isinstance(value, bool) and bool not in types:
            problems.append(Problem(
                WARNING, key,
                f"{key}={value} is a bool but is declared as {_type_name(expected)}.",
                f"Set {key} to a {_type_name(expected)} value."))
            continue
        if isinstance(value, int) and not isinstance(value, bool) and float in types and int not in types:
            continue  # an int is an acceptable float
        if not isinstance(value, types):
            hint = ""
            if isinstance(value, str):
                hint = (" Values read back from a settings CSV are strings; "
                        "re-import through the settings panel or convert it by hand.")
            problems.append(Problem(
                ERROR, key,
                f"{key}={value!r} is a {type(value).__name__}, but "
                f"{_type_name(expected)} is expected.",
                f"Set {key} to a {_type_name(expected)}.{hint}"))
    return problems


def _check_unknown_keys(settings: Dict[str, Any]) -> List[Problem]:
    """Flag keys that look like a typo of a real setting.

    Only keys with a close match are reported: spaCR's newer pipelines
    (stitching, motility, plotting) legitimately carry keys that are not in
    ``expected_types``, and warning about all of them would be noise.
    """
    known = _known_setting_keys()
    problems: List[Problem] = []
    for key in settings:
        if not isinstance(key, str) or key in known:
            continue
        close = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.85)
        if close:
            problems.append(Problem(
                WARNING, key,
                f"'{key}' is not a spaCR setting; did you mean '{close[0]}'?",
                f"Rename '{key}' to '{close[0]}' — as it stands the value is ignored and the default is used."))
    return problems


def _numeric(value: Any) -> Optional[float]:
    """Return ``value`` as a float when it is a real number, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _check_numeric_sanity(settings: Dict[str, Any]) -> List[Problem]:
    """Diameters, percentiles, thresholds and batch sizes are in usable ranges."""
    problems: List[Problem] = []

    for key, value in settings.items():
        if not isinstance(key, str):
            continue
        number = _numeric(value)

        # Cellpose object diameters are divided into and squared by
        # spacr.settings._get_object_settings (min = d**2/4), so zero or
        # negative is meaningless.
        if number is not None and (key.endswith("_diameter") or key == "diameter"):
            if number <= 0:
                problems.append(Problem(
                    ERROR, key, f"{key}={value} must be greater than zero.",
                    "Give the expected object size in pixels, or None to let magnification derive it."))

        if number is not None and key in ("batch_size", "test_images", "test_nr", "nr_imgs",
                                          "epochs", "n_epochs", "image_size", "size",
                                          "chunk_size", "magnification", "examples_to_plot"):
            if number < 1:
                problems.append(Problem(
                    ERROR, key, f"{key}={value} must be at least 1.",
                    f"Set {key} to a positive whole number."))

        # spacr.measure.measure_crop overrides n_jobs with cpu_count()-4, but
        # every other pipeline passes it straight to a Pool / DataLoader.
        if number is not None and key == "n_jobs":
            if number < 1 and number != -1:
                problems.append(Problem(
                    ERROR, key, f"n_jobs={value} is not a usable worker count.",
                    "Use a positive number of CPU workers, -1 for every core, or leave it blank."))

        if number is not None and (key.endswith("_percentile") or key in ("lower_percentile", "upper_percentile")):
            if not 0 <= number <= 100:
                problems.append(Problem(
                    ERROR, key, f"{key}={value} is not a percentile (0-100).",
                    f"Set {key} between 0 and 100."))

        # cellprob_threshold is clamped to about -6..6 by Cellpose itself.
        if number is not None and (key.endswith("_CP_prob") or key in ("CP_prob", "CP_probability")):
            if not -6 <= number <= 6:
                problems.append(Problem(
                    WARNING, key, f"{key}={value} is outside Cellpose's usable -6 to 6 range.",
                    "Lower it toward -6 to grow masks and keep faint objects; raise it toward 6 to shrink them."))

        # flow_threshold: 0 keeps only perfect masks, above ~3 keeps everything.
        if number is not None and (key.endswith("_FT") or key in ("FT", "flow_threshold")):
            if not 0 <= number <= 3:
                problems.append(Problem(
                    WARNING, key, f"{key}={value} is outside the useful 0 to 3 flow-threshold range.",
                    "Cellpose's own default is 0.4; spaCR ships 1.0. Values above 3 disable the filter entirely."))

        if number is not None and key in ("val_split", "test_split", "dropout_rate",
                                          "organelle_unet_threshold", "score_threshold"):
            if not 0 <= number <= 1:
                problems.append(Problem(
                    ERROR, key, f"{key}={value} must be a fraction between 0 and 1.",
                    f"Set {key} between 0 and 1 (0.1 means 10%)."))

        if number is not None and key == "learning_rate" and number <= 0:
            problems.append(Problem(
                ERROR, key, f"learning_rate={value} must be greater than zero.",
                "Typical values are 1e-4 to 1e-2."))

        if isinstance(value, (list, tuple)):
            problems.extend(_check_numeric_list(key, value))

    return problems


def _check_numeric_list(key: str, value: Sequence[Any]) -> List[Problem]:
    """Range-check list-valued percentile settings."""
    problems: List[Problem] = []
    percentile_lists = ("normalization_percentiles", "manders_thresholds",
                        "percentiles", "normalize")
    if key in percentile_lists or key.endswith("_percentiles"):
        numbers = [_numeric(v) for v in value]
        if any(n is None for n in numbers) or not numbers:
            return problems
        out_of_range = [n for n in numbers if not 0 <= n <= 100]
        if out_of_range:
            problems.append(Problem(
                ERROR, key, f"{key}={list(value)} contains values outside 0-100.",
                f"{key} is a list of percentiles; every entry must be between 0 and 100."))
        elif len(numbers) == 2 and numbers[0] >= numbers[1]:
            problems.append(Problem(
                ERROR, key,
                f"{key}={list(value)} has its lower percentile at or above its upper percentile.",
                f"Write {key} as [lower, upper], for example [1, 99]."))
    if key == "png_size":
        flat = value if value and not isinstance(value[0], (list, tuple)) else [
            v for pair in value for v in pair]
        numbers = [_numeric(v) for v in flat]
        if numbers and all(n is not None for n in numbers) and any(n < 1 for n in numbers):
            problems.append(Problem(
                ERROR, key, f"png_size={list(value)} contains a non-positive size.",
                "png_size is [width, height] in pixels, for example [224, 224]."))
    return problems


def _check_required_paths(settings: Dict[str, Any], app: str) -> List[Problem]:
    """App-specific inputs that must already exist on disk."""
    problems: List[Problem] = []

    def _require_file(key: str, purpose: str, fix: str) -> None:
        value = settings.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            problems.append(Problem(
                ERROR, key, f"{key} is not set, but {purpose}.", fix))
            return
        if isinstance(value, str) and not os.path.isfile(value):
            problems.append(Problem(
                ERROR, key, f"{key} points at a file that does not exist: {value}", fix))

    if app == "map_barcodes":
        # spacr.sequencing.generate_barecode_mapping reads all three CSVs to
        # translate the row/column/gRNA barcodes it pulls out of the reads.
        for key, label in (("grna_csv", "the gRNA barcodes"),
                           ("row_csv", "the row barcodes"),
                           ("column_csv", "the column barcodes")):
            _require_file(
                key, f"barcode mapping needs {label}",
                f"Point {key} at a CSV with 'name' and 'sequence' columns.")

    if app == "classify":
        train = settings.get("train", settings.get("generate_training_dataset", False))
        needs_model = bool(settings.get("apply_model_to_dataset", False)) or bool(settings.get("test", False))
        if needs_model and not train:
            _require_file(
                "model_path", "scoring a dataset needs a trained classifier",
                "Point model_path at a saved spaCR model, or set train=True to train one first.")

    custom_model = settings.get("custom_model")
    if isinstance(custom_model, str) and custom_model.strip():
        # spacr.spacr_cellpose prints 'Custom model not found' and returns
        # without segmenting anything when the path is wrong.
        if not os.path.exists(custom_model):
            problems.append(Problem(
                ERROR, "custom_model",
                f"custom_model points at a path that does not exist: {custom_model}",
                "Point custom_model at the saved Cellpose model file, or clear it to use the stock model."))

    unet_path = settings.get("organelle_unet_model_path")
    if settings.get("organelle_method") == "unet":
        if not isinstance(unet_path, str) or not unet_path.strip():
            problems.append(Problem(
                ERROR, "organelle_unet_model_path",
                "organelle_method='unet' but no organelle_unet_model_path is set.",
                "Point organelle_unet_model_path at the serialised U-Net, or pick another organelle_method."))
        elif not os.path.exists(unet_path):
            problems.append(Problem(
                ERROR, "organelle_unet_model_path",
                f"organelle U-Net model not found: {unet_path}",
                "Fix the path, or pick another organelle_method."))

    return problems


def _check_app_specific(settings: Dict[str, Any], app: str) -> List[Problem]:
    """Cross-setting rules the pipeline entry points enforce at runtime."""
    problems: List[Problem] = []

    if app == "mask":
        # core.preprocess_generate_masks prints 'At least one of cell_channel,
        # nucleus_channel, pathogen_channel or organelle_channel must be
        # defined' and returns.
        if all(settings.get(k) is None for k in CHANNEL_KEYS):
            problems.append(Problem(
                ERROR, "cell_channel",
                "no segmentation channel is set: cell, nucleus, pathogen and organelle channels are all None.",
                "Set at least one of cell_channel / nucleus_channel / pathogen_channel / organelle_channel to a channel index."))
        # pathogen_model: the bundled toxo checkpoints were Cellpose-3 and are
        # gone. Anything set here is ignored, so say so rather than validating
        # against a list of models that cannot load.
        if settings.get("pathogen_channel") is not None:
            model = settings.get("pathogen_model")
            if model is not None and model != "cpsam":
                problems.append(Problem(
                    WARNING, "pathogen_model",
                    f"pathogen_model={model!r} is ignored: Cellpose 4 ships only 'cpsam', "
                    f"and the pre-SAM toxo_pv_lumen / toxo_cyto checkpoints have been removed.",
                    "Drop the setting, or set it to 'cpsam' to be explicit."))

    if app == "measure":
        # measure_crop returns early on both of these.
        normalize = settings.get("normalize")
        if isinstance(normalize, bool) and normalize:
            problems.append(Problem(
                ERROR, "normalize",
                "normalize=True is rejected by measure_crop, which needs a percentile pair.",
                "Use a two-element list such as [1, 99], or False to skip normalization."))
        if isinstance(normalize, (list, tuple)) or normalize is True:
            if settings.get("normalize_by") not in ("png", "fov"):
                problems.append(Problem(
                    ERROR, "normalize_by",
                    f"normalize_by={settings.get('normalize_by')!r} is not understood.",
                    "Use 'png' to normalize each crop to its own percentiles, or 'fov' to use the whole field."))
        crop_mode = settings.get("crop_mode")
        if isinstance(crop_mode, (list, tuple)):
            allowed = {"cell", "nucleus", "pathogen", "cytoplasm", "organelle"}
            bad = [m for m in crop_mode if m not in allowed]
            if bad:
                problems.append(Problem(
                    ERROR, "crop_mode", f"crop_mode contains unsupported entries: {bad}.",
                    f"crop_mode entries must come from {sorted(allowed)}."))
            # dialate_png_ratios is indexed per crop mode with no broadcast.
            ratios = settings.get("dialate_png_ratios")
            if settings.get("dialate_pngs") and isinstance(ratios, (list, tuple)):
                needed = len([m for m in crop_mode if m != "cytoplasm"])
                if needed > len(ratios):
                    problems.append(Problem(
                        ERROR, "dialate_png_ratios",
                        f"dialate_png_ratios has {len(ratios)} entr"
                        f"{'y' if len(ratios) == 1 else 'ies'} but {needed} crop modes need one each.",
                        f"Give dialate_png_ratios one value per crop mode, e.g. {[0.2] * needed}."))
            for mode in crop_mode:
                key = f"{mode}_mask_dim"
                if mode in OBJECT_NAMES and settings.get(key) is None and key in settings:
                    problems.append(Problem(
                        ERROR, key,
                        f"crop_mode asks for {mode} crops but {key} is None, so no {mode} mask is read.",
                        f"Set {key} to the plane holding the {mode} mask, or drop '{mode}' from crop_mode."))

    return problems


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def validate_settings(settings: Dict[str, Any], app_key: str) -> List[Problem]:
    """Check a settings dict against the data it points at.

    Nothing is loaded beyond one ``.npy`` header and a directory listing, so
    this is safe to call before committing a GPU to a run.

    :param settings: the settings dict about to be handed to a pipeline.
    :param app_key: which pipeline, e.g. ``'mask'``, ``'measure'``,
        ``'classify'``, ``'umap'``, ``'map_barcodes'``. The
        ``settings_type`` strings used by the GUI all work, as do a few
        aliases (``'sequencing'``, ``'measure_crop'``, ...).
    :returns: list of :class:`Problem`; empty means nothing was found.
    """
    if not isinstance(settings, dict):
        return [Problem(ERROR, "", f"settings must be a dict, got {type(settings).__name__}.",
                        "Pass the settings dictionary you would hand to the pipeline.")]

    app = _normalize_app(app_key)
    problems: List[Problem] = []
    if app and app not in APP_FUNCTIONS:
        problems.append(Problem(
            WARNING, "", f"unknown app '{app_key}'; only the generic checks were run.",
            f"Use one of: {', '.join(sorted(APP_FUNCTIONS))}."))

    inventories = [_inventory(src, settings, app) for src in _src_values(settings)]

    problems.extend(_check_src(settings, app, inventories))
    problems.extend(_check_channels(settings, app, inventories))
    problems.extend(_check_types(settings))
    problems.extend(_check_unknown_keys(settings))
    problems.extend(_check_numeric_sanity(settings))
    problems.extend(_check_required_paths(settings, app))
    problems.extend(_check_app_specific(settings, app))
    return problems


def format_report(problems: Sequence[Problem], settings: Optional[Dict[str, Any]] = None,
                  app_key: str = "") -> str:
    """Render :func:`validate_settings` output for a terminal.

    Errors come first as a group, then warnings, each entry followed by its
    fix line. A clean run says so explicitly.

    :param problems: what :func:`validate_settings` returned.
    :param settings: the settings dict, used only for the header line.
    :param app_key: the app the check was run for, used only for the header.
    :returns: the report as a single string, no trailing newline.
    """
    app = _normalize_app(app_key)
    src = ""
    if isinstance(settings, dict):
        values = _src_values(settings)
        src = str(values[0]) if values and values[0] is not None else ""
        if len(values) > 1:
            src = f"{src} (+{len(values) - 1} more)"

    title = "spaCR pre-flight check"
    if app:
        title += f" — {app}"
    if src:
        title += f" — {src}"
    rule = "=" * max(len(title), 60)
    lines = [rule, title, rule, ""]

    errors = [p for p in problems if p.severity == ERROR]
    warnings = [p for p in problems if p.severity != ERROR]

    if not problems:
        lines.append("No problems found. These settings are runnable against this data.")
        return "\n".join(lines)

    if errors:
        lines.append(f"ERRORS ({len(errors)}) — the run would fail or produce wrong output:")
        lines.append("")
        for problem in errors:
            lines.extend(_render(problem))
        lines.append("")

    if warnings:
        lines.append(f"WARNINGS ({len(warnings)}) — the run would proceed, but check these:")
        lines.append("")
        for problem in warnings:
            lines.extend(_render(problem))
        lines.append("")

    if errors:
        lines.append(f"{len(errors)} error{'' if len(errors) == 1 else 's'} must be fixed before this run will work.")
    else:
        lines.append("No errors. The run would proceed.")
    return "\n".join(lines)


def _render(problem: Problem) -> List[str]:
    """One problem as report lines: the message, then its fix."""
    head = f"  {problem.setting}: {problem.message}" if problem.setting else f"  {problem.message}"
    return [head, f"      fix: {problem.fix}", ""]


def _fmt(value: Any) -> str:
    """Compact display form for a settings value."""
    if value is None:
        return "not set"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(v) for v in value) + "]"
    return str(value)


def describe_plan(settings: Dict[str, Any], app_key: str = "") -> str:
    """Summarise what the run would actually do, without doing it.

    Reports the app and the function behind it, the resolved source folder,
    how many files were found, which objects would be segmented or measured
    with which channels and diameters, where output would land, and roughly
    how many images would be processed.

    :param settings: the settings dict about to be handed to a pipeline.
    :param app_key: which pipeline, as for :func:`validate_settings`.
    :returns: the plan as a single string, no trailing newline.
    """
    if not isinstance(settings, dict):
        return "Plan unavailable: settings is not a dict."

    app = _normalize_app(app_key)
    srcs = _src_values(settings)
    inventories = [_inventory(src, settings, app) for src in srcs]

    rows: List[Tuple[str, str]] = []
    rows.append(("app", f"{app or 'unknown'}"
                        + (f" ({APP_FUNCTIONS[app]})" if app in APP_FUNCTIONS else "")))
    named = [str(s) for s in srcs if s not in (None, "")]
    rows.append(("source", ", ".join(named) if named else "not set"))

    inv = inventories[0] if inventories else _Inventory()
    if app in MERGED_APPS and inv.merged_dir and inv.merged_dir != inv.src:
        # measure_crop silently appends 'merged' when src does not end with
        # it, so say where it would actually look.
        rows.append(("reads", inv.merged_dir))

    rows.append(("inputs found", _describe_inputs(inventories)))

    if inv.raw_channels is not None:
        detail = f"{inv.raw_channels}"
        if inv.raw_channel_ids:
            detail += f" ({', '.join(inv.raw_channel_ids)})"
        rows.append(("channels", detail))
    if inv.array_planes is not None:
        rows.append(("array planes", f"{inv.array_planes} (indices 0-{inv.array_planes - 1})"))

    object_lines = _describe_objects(settings, app)
    if object_lines:
        label = "would segment" if app == "mask" else "would measure"
        rows.append((label, object_lines[0]))
        for extra in object_lines[1:]:
            rows.append(("", extra))

    if app == "measure":
        crop_mode = settings.get("crop_mode")
        if settings.get("save_png") and isinstance(crop_mode, (list, tuple)) and crop_mode:
            size = _fmt(settings.get("png_size"))
            rows.append(("crops", f"{', '.join(str(m) for m in crop_mode)} PNGs at {size}"))
        rows.append(("measures channels", _fmt(settings.get("channels"))))

    for label, path in _describe_outputs(settings, app, inv):
        rows.append((label, path))

    workload = _describe_workload(settings, app, inventories)
    if workload:
        rows.append(("workload", workload))

    if settings.get("test_mode"):
        rows.append(("test mode", "on — only a small subset of the plate would be processed"))

    width = max((len(label) for label, _ in rows if label), default=0)
    lines = ["Plan — what this run would do. Nothing has been written and no model has been loaded:", ""]
    for label, value in rows:
        lines.append(f"  {label.ljust(width)}  {value}" if label else f"  {' ' * width}  {value}")
    return "\n".join(lines)


def _describe_inputs(inventories: Sequence[_Inventory]) -> str:
    """One line describing what was found across every ``src``."""
    parts: List[str] = []
    merged = sum(inv.merged_files for inv in inventories)
    stacks = sum(inv.stack_files for inv in inventories)
    raw = sum(inv.raw_files for inv in inventories)
    missing = [inv.src for inv in inventories if inv.src and not inv.exists]
    if missing:
        return f"nothing — {', '.join(missing)} does not exist"
    if merged:
        parts.append(f"{merged} merged array{'' if merged == 1 else 's'} (.npy)")
    if stacks:
        parts.append(f"{stacks} channel stack{'' if stacks == 1 else 's'} (.npy)")
    if raw:
        parts.append(f"{raw} raw image file{'' if raw == 1 else 's'}")
    return ", ".join(parts) if parts else "no image files found"


def _describe_objects(settings: Dict[str, Any], app: str) -> List[str]:
    """One line per object type the run would work on."""
    lines: List[str] = []
    for name in OBJECT_NAMES:
        if app == "mask":
            channel = settings.get(f"{name}_channel")
            if channel is None:
                continue
            diameter = settings.get(f"{name}_diameter")
            detail = f"{name}: channel {channel}"
            if diameter is not None:
                detail += f", diameter {diameter} px"
            else:
                detail += f", diameter from magnification {settings.get('magnification', 20)}x"
            model = settings.get("pathogen_model") if name == "pathogen" else None
            if model:
                detail += f", model {model}"
            if name == "organelle":
                detail += f", method {settings.get('organelle_method', 'otsu')}"
            lines.append(detail)
        else:
            dim = settings.get(f"{name}_mask_dim")
            if dim is None:
                continue
            detail = f"{name}: mask plane {dim}"
            min_size = settings.get(f"{name}_min_size")
            if min_size is not None:
                detail += f", min size {min_size} px"
            lines.append(detail)
    if app == "measure" and settings.get("cytoplasm"):
        lines.append("cytoplasm: derived from the cell mask")
    return lines


def _describe_outputs(settings: Dict[str, Any], app: str, inv: _Inventory) -> List[Tuple[str, str]]:
    """Where the run would write, as ``(label, path)`` rows."""
    src = inv.src or str(settings.get("src", ""))
    if not src:
        return []
    rows: List[Tuple[str, str]] = []
    if app == "mask":
        rows.append(("would write", os.path.join(src, "masks") + os.sep))
        rows.append(("", os.path.join(src, "merged") + os.sep))
        rows.append(("", os.path.join(src, "settings", "gen_mask_settings.csv")))
    elif app == "measure":
        root = os.path.dirname(os.path.normpath(inv.merged_dir or src))
        rows.append(("would write", os.path.join(root, "measurements", "measurements.db")))
        crop_mode = settings.get("crop_mode")
        if settings.get("save_png") and isinstance(crop_mode, (list, tuple)):
            for mode in crop_mode:
                rows.append(("", os.path.join(root, "data", "...", f"{mode}_png") + os.sep))
    elif app in DB_APPS:
        rows.append(("would read", inv.db_path or os.path.join(src, "measurements", "measurements.db")))
    return rows


def _describe_workload(settings: Dict[str, Any], app: str,
                       inventories: Sequence[_Inventory]) -> str:
    """Rough count of the images the run would touch."""
    parts: List[str] = []
    if app in MERGED_APPS:
        total = sum(inv.merged_files for inv in inventories)
        if total:
            parts.append(f"~{total} field{'' if total == 1 else 's'} to measure")
    elif app == "mask":
        fields = sum(inv.fields or 0 for inv in inventories)
        stacks = sum(inv.stack_files for inv in inventories)
        raw = sum(inv.raw_files for inv in inventories)
        if fields:
            parts.append(f"~{fields} field{'' if fields == 1 else 's'} from {raw} files")
        elif stacks:
            parts.append(f"~{stacks} field stack{'' if stacks == 1 else 's'}")
        elif raw:
            parts.append(f"~{raw} image file{'' if raw == 1 else 's'}")
        objects = len([k for k in CHANNEL_KEYS if settings.get(k) is not None])
        if objects:
            parts.append(f"{objects} segmentation pass{'' if objects == 1 else 'es'} per field")
    n_jobs = settings.get("n_jobs")
    if n_jobs is not None:
        parts.append(f"n_jobs={n_jobs}")
    return ", ".join(parts)


DRY_RUN_TRAILER = "dry_run=True — stopping here. Set dry_run=False to run for real."


def run_preflight(settings: Dict[str, Any], app_key: str, printer=print,
                  trailer: str = DRY_RUN_TRAILER) -> List[Problem]:
    """Validate, print the report and the plan, and hand back the problems.

    This is what the ``dry_run`` branch of every pipeline entry point calls,
    so the wording is identical wherever it is triggered from.

    :param settings: the settings dict that would have been run.
    :param app_key: which pipeline, as for :func:`validate_settings`.
    :param printer: where the text goes; defaults to ``print``.
    :param trailer: the closing line. The default names the in-pipeline
        ``dry_run`` flag, which is the wrong advice for a caller reached some
        other way -- ``spacr-run --dry-run`` passes its own wording. Pass an
        empty string to suppress it entirely.
    :returns: the list of :class:`Problem` found.
    """
    problems = validate_settings(settings, app_key)
    printer(format_report(problems, settings, app_key))
    printer("")
    printer(describe_plan(settings, app_key))
    if trailer:
        printer("")
        printer(trailer)
    return problems
