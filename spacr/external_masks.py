"""Import externally generated masks and finish a spaCR Measure project.

This is the entry point for segmentation performed outside spaCR.  It accepts
one or more mixed folders/files, detects intensity images and label images,
lets callers override every proposed role/object type, builds the canonical
``stack/``, ``masks/`` and ``merged/`` folders, and then delegates all feature
extraction and crop generation to :func:`spacr.measure.measure_crop`.

The output is therefore the same contract Annotate expects from Measure::

    destination/
      data/.../<object>_png/*.png
      images/*.tif
      stack/*.npy
      masks/<object>_mask_stack/*.npy
      merged/*.npy
      measurements/measurements.db

Only object types supplied by the user are measured.  ``cytoplasm`` is
derived by Measure from a cell mask and any supplied interior masks; it is
not an input mask plane.
"""
from __future__ import annotations

from .schema import object_type_summary

import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

class _LazyConvert:
    """``spacr.convert``, imported on first attribute access.

    THIS MODULE IS ON THE STARTUP PATH, three links down: the settings model
    imports the external-mask widget, which imports this, and `convert`
    imports pandas. The packaged smoke test asserts Home crosses no
    operation-only import boundary and was failing on pandas for exactly that
    chain.

    A proxy rather than sixteen function-level imports, so every existing
    ``cv.something`` reads the same. Safe because this module declares
    ``from __future__ import annotations``: the annotations that name
    ``cv.ConversionPlan`` are strings and never evaluated at import.
    """

    __slots__ = ()

    def __getattr__(self, name: str):
        """Forward to :mod:`spacr.convert`, imported on first use."""
        from . import convert

        return getattr(convert, name)


cv = _LazyConvert()
from . import crops
from .errors import ConfigurationError
from .object_roles import ORGANELLE_ROLES


SUPPORTED_SUFFIXES = (
    ".tif", ".tiff", ".ome.tif", ".ome.tiff",
    ".png", ".jpg", ".jpeg", ".bmp",
)
OBJECT_TYPES = tuple(crops.MASK_PLANE_ORDER)
ROLES = ("image", "mask", "ignore")

#: What may sit either side of a name token. Path separators count, so a
#: folder named only ``organelle_2/`` or ``cell_masks/`` (with plain
#: ``fov001.tif`` files inside, or added as a subfolder of a project) names
#: its object type. Before 2026-09-30 (item 76) only ``_-. `` and a space
#: did, and such folders were proposed as nothing, or as the wrong slot.
_B = r"[_\-. /\\]"

_MASK_WORDS = re.compile(
    rf"(?i)(?:^|{_B})(?:mask|masks|label|labels|labelled|labeled|"
    rf"instance|instances|seg|segmentation|outline)(?:$|{_B})"
)
_ORGANELLE_PATTERN = re.compile(
    rf"(?i)(?:^|{_B})"
    r"(?:organelle(?:[_\-. ]?(?P<number>\d+)|(?P<letters>[a-z]+))?"
    r"|organell|mitochondria|mitochondrion)"
    rf"(?={_B}|$)"
)

#: Excitation laser lines. ``organelle_488_masks`` names the 488 nm channel,
#: not Organelle 488; such a token counts as a bare ``organelle`` (item 76,
#: 2026-09-30). Only these numbers are read as wavelengths, so every other
#: slot number from 1 to 702 still names its slot.
_LASER_LINES = frozenset({
    355, 375, 405, 440, 445, 457, 458, 473, 488, 491, 505, 514, 515, 532,
    543, 552, 555, 561, 568, 588, 594, 633, 635, 637, 638, 640, 642, 647,
    660, 685, 730, 750, 785,
})
_ORGANELLE_TAIL = re.compile(
    r"(?i)(?:^|[_\-. ])"
    r"(?:organelle(?:[_\-. ]?\d+|[a-z]+)?|organell|mitochondria|mitochondrion)$"
)
_ORGANELLE_ROLE_SET = frozenset(ORGANELLE_ROLES)
_ORGANELLE_PLURAL = "organelles"
_OBJECT_PATTERNS = (
    ("nucleus", re.compile(rf"(?i)(?:^|{_B})(?:nucleus|nuclei|nuclear|nuc)(?:$|{_B})")),
    ("pathogen", re.compile(rf"(?i)(?:^|{_B})(?:pathogen|parasite|bacteria|bacterial)(?:$|{_B})")),
    ("organelle", _ORGANELLE_PATTERN),
    ("cell", re.compile(rf"(?i)(?:^|{_B})(?:cell|cells|wholecell|cytoplasm)(?:$|{_B})")),
)


@dataclass
class InputGroup:
    """A set of files sharing one proposed role and object type.

    :param key: stable identifier derived from the input root, proposed role,
        and detected file family.
    :param root: absolute directory from which the files were detected and
        later scanned.
    :param paths: absolute paths assigned to this group.
    :param role: reviewed ``"image"``, ``"mask"``, or ``"ignore"`` role.
    :param object_type: proposed spaCR object type for a mask group, or
        ``None`` when no mask type is assigned.
    :param confidence: confidence score for the automatic role proposal;
        detected multi-file groups retain their lowest member score.
    :param reason: pixel or filename evidence supporting the automatic
        proposal.
    """

    key: str
    root: str
    paths: List[str]
    role: str
    object_type: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return every group field as a recursively copied plain mapping."""
        return asdict(self)

    @classmethod
    def from_value(cls, value: Any) -> "InputGroup":
        """Normalize a group instance or serialized mapping.

        :param value: existing :class:`InputGroup` or mapping carrying its
            serialized fields.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ConfigurationError(
                "Each external-mask input must be an InputGroup or mapping.")
        raw_object_type = (
            str(value["object_type"]) if value.get("object_type") else None)
        object_type = (
            raw_object_type if raw_object_type in OBJECT_TYPES else None)
        return cls(
            key=str(value.get("key") or ""),
            root=os.path.abspath(str(value.get("root") or ".")),
            paths=[os.path.abspath(str(path))
                   for path in value.get("paths", [])],
            role=str(value.get("role") or "ignore"),
            object_type=object_type,
            confidence=float(value.get("confidence") or 0.0),
            reason=str(value.get("reason") or ""),
        )


@dataclass(frozen=True)
class MaskMatch:
    """One label-mask file matched to an intensity-image field.

    :param path: absolute path of the matched source label-mask file; the
        importer reads this file and reports it in validation errors.
    :param object_type: valid spaCR object role assigned to the mask group,
        such as ``cell`` or ``nucleus``.
    :param stem: canonical ``<plate>_<well>_<field>`` stem of the intensity
        field to which the mask was paired.
    :param match: pairing rule that succeeded, currently ``"exact"`` or
        ``"normalised"`` after stripping mask/object-role suffixes.
    """

    path: str
    object_type: str
    stem: str
    match: str


@dataclass
class ExternalMaskPlan:
    """Read-only preview of an external-mask import.

    :param groups: normalized image, mask, and ignored input groups reviewed for
        this import.
    :param images: proposed intensity-image conversion plan.
    :param masks: spaCR object types mapped to canonical field stems and their
        matched label-mask files.
    :param destination: root of the spaCR project that would be written.
    :param n_channels: number of intensity-image channels in each generated
        merged field.
    :param mask_dims: merged-array plane index assigned to each supplied mask
        type.
    :param errors: blocking problems that make the preview unrunnable.
    :param warnings: non-blocking ambiguities shown before the import proceeds.
    """

    groups: List[InputGroup]
    images: cv.ConversionPlan
    masks: Dict[str, Dict[str, MaskMatch]]
    destination: str
    n_channels: int
    mask_dims: Dict[str, int]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def stems(self) -> List[str]:
        """Return sorted field stems covered by every supplied mask type."""
        return sorted(set.intersection(
            *(set(per_stem) for per_stem in self.masks.values())
        )) if self.masks else []

    @property
    def object_types(self) -> List[str]:
        """Return supplied mask roles in canonical object-type order."""
        return [name for name in OBJECT_TYPES if name in self.masks]

    @property
    def ok(self) -> bool:
        """Return whether images, shared fields, and importer checks are valid."""
        return bool(self.images.ok and self.stems and not self.errors)

    def summary(self) -> str:
        """Return a multiline read-only preview of mappings and problems."""
        lines = [
            "External masks → Measure project (preview; nothing written)",
            f"  intensity mappings: {len(self.images)}",
            f"  fields ready: {len(self.stems)}",
            f"  intensity channels: {self.n_channels}",
            f"  mask types: {object_type_summary(self.object_types) or 'none'}",
            f"  destination: {self.destination}",
        ]
        for name in self.object_types:
            lines.append(f"  {name}: {len(self.masks[name])} paired mask(s), "
                         f"merged plane {self.mask_dims[name]}")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {message}" for message in self.warnings)
        if self.errors or self.images.errors:
            lines.append("Blocking problems:")
            lines.extend(f"  - {message}"
                         for message in [*self.images.errors, *self.errors])
        return "\n".join(lines)


@dataclass
class ExternalMaskResult:
    """Files and database produced by :func:`prepare_external_masks`.

    :param destination: root of the generated spaCR project.
    :param merged: paths of generated merged image-and-mask arrays.
    :param db_path: path of the generated measurements database.
    :param tables: measurement-table names written to that database.
    :param data_dir: generated annotation-crop directory.
    :param plan: validated read-only import plan used to produce these outputs.
    """

    destination: str
    merged: List[str]
    db_path: str
    tables: List[str]
    data_dir: str
    plan: ExternalMaskPlan

    def summary(self) -> str:
        """Return one line naming materialized fields and output locations."""
        return (
            f"Prepared {len(self.merged)} field(s) in {self.destination}. "
            f"measurements.db tables: {', '.join(self.tables) or 'none'}. "
            f"Annotation crops: {self.data_dir}"
        )


def _all_files(path: Path, recursive: bool) -> List[Path]:
    """Collect supported files from a file or directory input.

    :param path: input file or directory; missing paths produce no files.
    :param recursive: descend through directory children when true.
    """
    if path.is_file():
        return [path] if _supported(path) else []
    if not path.is_dir():
        return []
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(candidate for candidate in iterator
                  if candidate.is_file() and _supported(candidate))


def _supported(path: Path) -> bool:
    """Return whether a path has a supported suffix, case-insensitively."""
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def _suggest_object(name: str) -> Optional[str]:
    """Infer the first matching spaCR object role from path-name tokens.

    :param name: filename or path whose stem and parent tokens are inspected.
    :returns: canonical object type, or ``None`` when no pattern matches.
    """
    stem = cv._split_ext(str(name))[0]
    for object_type, pattern in _OBJECT_PATTERNS:
        if pattern is _ORGANELLE_PATTERN:
            slot = _suggest_organelle_slot(stem)
            if slot is not None:
                return slot
        elif pattern.search(stem):
            return object_type
    return None


def _suggest_organelle_slot(stem: str) -> Optional[str]:
    """Name the organelle slot a filename token spells, for every slot.

    A bare ``organelle`` (or ``organell``, ``mitochondria``,
    ``mitochondrion``) is slot 1. A number after it, with or without one
    separator -- ``organelle_5``, ``organelle 12``, ``organelle5`` -- is the
    slot as the user counts it, so ``organelle_5`` is ``organellee``. A
    letter suffix is the role spelling spaCR itself writes, so a folder such
    as ``organellee_mask_stack`` imports back into its own slot.

    Every slot of :data:`spacr.object_roles.ORGANELLE_ROLES` is reachable,
    rather than the first four a fixed table of patterns listed. A numbered
    token that names no slot -- ``organelle_0``, ``organelle_2024`` -- and the
    English plural ``organelles`` propose nothing, because neither says which
    slot is meant; the object type stays editable in the import table. A
    number that is a laser line (:data:`_LASER_LINES`, e.g. ``organelle_488``)
    names a channel, so it counts as a bare token.

    A token that names a slot outranks a bare one wherever each sits in the
    path, so ``organelle_masks/fov001_organelle_2.tif`` and
    ``organelle_2/fov001_organelle_2_mask.tif`` are slot 2, not slot 1. A
    bare token is slot 1 only when no token in the path names another.

    :param stem: filename stem, optionally with parent-folder tokens.
    :returns: an organelle role, or ``None`` when no token names a slot.
    """
    bare = False
    for match in _ORGANELLE_PATTERN.finditer(stem):
        number = match.group("number")
        letters = match.group("letters")
        if number is not None:
            index = int(number)
            if index in _LASER_LINES:
                bare = True
                continue
            if 1 <= index <= len(ORGANELLE_ROLES):
                return ORGANELLE_ROLES[index - 1]
            continue
        if letters is not None:
            role = f"organelle{letters.lower()}"
            if role in _ORGANELLE_ROLE_SET and role != _ORGANELLE_PLURAL:
                return role
            continue
        bare = True
    return ORGANELLE_ROLES[0] if bare else None


def _label_likelihood(path: Path) -> Tuple[bool, float, str]:
    """Inspect one small sample and decide whether it resembles labels."""
    try:
        from .foreign import _read_mask
        array = np.asarray(_read_mask(str(path)))
    except Exception as exc:
        return False, 0.0, f"could not sample pixels: {exc}"
    if array.ndim != 2 or array.size == 0:
        return False, 0.0, f"shape {array.shape} is not a 2-D label plane"
    if not (np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.bool_)):
        return False, 0.0, f"{array.dtype} is not an integer label dtype"
    stride = max(int(np.sqrt(array.size / 250_000)), 1)
    sampled = array[::stride, ::stride]
    values = np.unique(sampled)
    nonnegative = bool(values.size == 0 or values[0] >= 0)
    compact = len(values) <= max(64, int(sampled.size * 0.002))
    background = bool(np.any(values == 0))
    likely = nonnegative and compact and background
    confidence = 0.88 if likely else 0.85
    return likely, confidence, (
        f"{len(values)} unique integer values in {sampled.size} sampled "
        f"pixels; {'zero background' if background else 'no zero background'}"
    )


def detect_inputs(paths: Sequence[Any], *, recursive: bool = True
                  ) -> List[InputGroup]:
    """Detect image and mask groups without writing anything.

    :param paths: input files or directories to inspect and group.

    Filename evidence wins when a path explicitly says ``mask``/``labels``.
    Otherwise a bounded pixel sample distinguishes compact integer label
    planes from intensity images.  Every result remains editable in the GUI.
    """
    grouped: Dict[Tuple[str, str, str], InputGroup] = {}
    for supplied in paths:
        source = Path(str(supplied)).expanduser().resolve()
        root = source if source.is_dir() else source.parent
        files = _all_files(source, recursive)
        for path in files:
            relative = (
                str(path.relative_to(root)) if path != root else path.name)
            evidence = os.path.join(root.name, relative)
            object_type = _suggest_object(evidence)
            explicit = _MASK_WORDS.search(evidence) is not None
            if explicit:
                role, confidence = "mask", 0.99
                reason = "filename contains a mask/label token"
            else:
                likely, confidence, reason = _label_likelihood(path)
                role = "mask" if likely else "image"
            family = object_type or ("unassigned" if role == "mask" else "intensity")
            key_tuple = (str(root), role, family)
            group = grouped.get(key_tuple)
            if group is None:
                key = f"{root}::{role}::{family}"
                group = grouped[key_tuple] = InputGroup(
                    key=key, root=str(root), paths=[], role=role,
                    object_type=object_type if role == "mask" else None,
                    confidence=confidence, reason=reason)
            group.paths.append(str(path))
            group.confidence = min(group.confidence, confidence)
    return sorted(grouped.values(),
                  key=lambda group: (group.role != "image",
                                     group.object_type or "", group.key))


def _coerce_groups(value: Any, *, recursive: bool) -> List[InputGroup]:
    """Normalize path-like or serialized inputs into reviewed groups.

    :param value: absent input, one path, path sequence, group instances, or
        serialized group mappings.
    :param recursive: directory traversal rule used when detecting paths.
    """
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return detect_inputs([value], recursive=recursive)
    values = list(value)
    if not values:
        return []
    if all(isinstance(item, (str, os.PathLike)) for item in values):
        return detect_inputs(values, recursive=recursive)
    return [InputGroup.from_value(item) for item in values]


def _scan_group(group: InputGroup, *, layout: str = "auto"
                ) -> List[cv.SourceImage]:
    """Scan a group's root once and keep only its selected paths."""
    selected = {os.path.abspath(path) for path in group.paths}
    sources = cv.scan(group.root, layout=layout)
    return [source for source in sources
            if os.path.abspath(source.path) in selected]


def _stem(mapping: cv.Mapping) -> str:
    """Return the canonical ``plate_well_integer-field`` destination stem."""
    return f"{mapping.plate}_{mapping.well}_{int(mapping.field)}"


def _pair_masks(image_plan: cv.ConversionPlan,
                groups: Sequence[InputGroup],
                *,
                layout: str = "auto",
                ) -> Tuple[Dict[str, Dict[str, MaskMatch]], List[str], List[str]]:
    """Pair reviewed mask sources to image fields by canonical identity.

    Exact source identities are preferred before normalized loose-field
    matching. The return tuple contains mappings by object type, blocking
    errors, and non-blocking ambiguity warnings.

    Normalizing drops a trailing mask word and the group's object-type name
    from a mask field. For a group typed as any organelle slot it drops any
    trailing organelle token -- ``organelle``, ``organelle_7``,
    ``organelleb``, ``mitochondria`` -- not only the assigned slot's own
    spelling, so a group re-typed in the import table still pairs, and
    ``fov001_mitochondria_mask`` pairs with ``fov001``.

    :param image_plan: reviewed intensity-image conversion plan.
    :param groups: reviewed inputs, including zero or more mask groups.
    :param layout: filename-layout rule forwarded while scanning each group.
    """
    errors: List[str] = []
    warnings: List[str] = []
    source_to_stem: Dict[Tuple[str, str, str], str] = {}
    loose_fields: Dict[str, set] = {}
    all_stems = set()
    for mapping in image_plan.mappings:
        key = (mapping.source_plate, mapping.source_well,
               mapping.source_field)
        stem = _stem(mapping)
        source_to_stem[key] = stem
        loose_fields.setdefault(mapping.source_field, set()).add(stem)
        all_stems.add(stem)

    by_type: Dict[str, Dict[str, MaskMatch]] = {}
    for group in groups:
        if group.role != "mask":
            continue
        object_type = group.object_type
        if object_type not in OBJECT_TYPES:
            fixed = [role for role in OBJECT_TYPES
                     if role not in _ORGANELLE_ROLE_SET]
            errors.append(
                f"{group.key}: choose whether these masks are "
                f"{', '.join(fixed)} or an organelle slot "
                f"({ORGANELLE_ROLES[0]}, {ORGANELLE_ROLES[1]}, "
                f"{ORGANELLE_ROLES[2]}, …).")
            continue
        matched = by_type.setdefault(object_type, {})
        for source in _scan_group(group, layout=layout):
            field = source.field
            normalised = field
            for token in (
                "mask", "masks", "label", "labels", "labelled", "labeled",
                "instance", "instances", "seg", "segmentation", object_type,
                "nuclei" if object_type == "nucleus" else object_type,
                "parasite" if object_type == "pathogen" else object_type,
                "organell" if object_type == "organelle" else object_type,
            ):
                normalised = re.sub(
                    rf"(?i)(?:^|[_\-. ]){re.escape(token)}$",
                    "", normalised).strip("_-. ")
            if object_type in _ORGANELLE_ROLE_SET:
                normalised = _ORGANELLE_TAIL.sub(
                    "", normalised).strip("_-. ")
            candidates = [
                ((source.plate, source.well, field), "exact"),
                ((source.plate, source.well, normalised), "normalised"),
            ]
            hit: Optional[Tuple[str, str]] = None
            for key, how in candidates:
                if key in source_to_stem:
                    hit = (source_to_stem[key], how)
                    break
                loose = loose_fields.get(key[2], set())
                if len(loose) == 1:
                    hit = (next(iter(loose)), how)
                    break
            if hit is None:
                warnings.append(
                    f"{source.path}: no intensity field has the same "
                    f"plate/well/field name; mask not imported.")
                continue
            stem, how = hit
            if stem in matched:
                errors.append(
                    f"{matched[stem].path} and {source.path} both map to "
                    f"{stem} as {object_type} masks.")
                continue
            matched[stem] = MaskMatch(
                path=source.path, object_type=object_type, stem=stem,
                match=how)

    for object_type, matched in by_type.items():
        missing = sorted(all_stems - set(matched))
        if missing:
            errors.append(
                f"{object_type}: {len(missing)} intensity field(s) have no "
                f"matching mask ({', '.join(missing[:5])}"
                f"{'…' if len(missing) > 5 else ''}).")
    return by_type, errors, warnings


def default_settings(settings: Optional[Mapping[str, Any]] = None
                     ) -> Dict[str, Any]:
    """Return importer settings plus the complete Measure setting contract."""
    from .settings import get_measure_crop_settings

    resolved = get_measure_crop_settings({})
    for key in ("src", *(f"{role}_mask_dim" for role in OBJECT_TYPES),
                *_LABEL_IMPORT_SETTINGS):
        resolved.pop(key, None)
    resolved.update({
        "inputs": [],
        "dst": None,
        "recursive": True,
        "layout": "auto",
        "z_handling": cv.Z_MAX,
        "plate_naming": "index",
        "overwrite": False,
        "preview_only": False,
        "channels": [],
        "png_dims": [],
        "experiment": "external_masks",
        "save_measurements": True,
        "save_png": True,
        "cytoplasm": True,
    })
    resolved.update(dict(settings or {}))
    return resolved


def plan_external_masks(settings: Optional[Mapping[str, Any]] = None
                        ) -> ExternalMaskPlan:
    """Validate and preview an external image/mask import without writing.

    :param settings: Partial settings mapping accepted by
        :func:`default_settings`.
    :returns: Pairing plan containing canonical image mappings, per-object
        mask matches, warnings, and blocking errors.
    """
    resolved = default_settings(settings)
    groups = _coerce_groups(
        resolved.get("inputs"), recursive=bool(resolved.get("recursive", True)))
    destination = os.path.abspath(str(
        resolved.get("dst") or
        (f"{groups[0].root}_spacr" if groups else "external_masks_spacr")))
    errors: List[str] = []
    warnings: List[str] = []

    image_groups = [group for group in groups if group.role == "image"]
    if not image_groups:
        errors.append("No intensity-image group is selected.")
    mask_groups = [group for group in groups if group.role == "mask"]
    if not mask_groups:
        errors.append("No label-mask group is selected.")
    invalid_roles = sorted({group.role for group in groups
                            if group.role not in ROLES})
    if invalid_roles:
        errors.append(f"Unknown input roles: {', '.join(invalid_roles)}.")

    layout = str(resolved.get("layout") or "auto")
    if layout not in cv.LAYOUTS:
        errors.append(
            f"Unknown input layout {layout!r}; choose one of "
            f"{', '.join(cv.LAYOUTS)}.")
        layout = "auto"

    sources: List[cv.SourceImage] = []
    seen = set()
    for group in image_groups:
        for source in _scan_group(group, layout=layout):
            marker = (source.path, source.meta.get("series", 0))
            if marker not in seen:
                sources.append(source)
                seen.add(marker)
    image_plan = cv.plan(
        sources,
        z_handling=str(resolved.get("z_handling") or cv.Z_MAX),
        plate_naming=str(resolved.get("plate_naming") or "index"),
    )
    n_channels = len({mapping.channel for mapping in image_plan.mappings})
    if not n_channels:
        errors.append("No readable intensity channels were detected.")
    mapping_slots = [
        (_stem(mapping), int(mapping.channel))
        for mapping in image_plan.mappings
    ]
    if len(mapping_slots) != len(set(mapping_slots)):
        errors.append(
            "Multiple time or Z planes map to the same field/channel. "
            "Choose z_handling='max' or 'first', or export each 2-D plane "
            "as a separately named field before importing.")
    masks_by_type, pair_errors, pair_warnings = _pair_masks(
        image_plan, mask_groups, layout=layout)
    errors.extend(pair_errors)
    warnings.extend(pair_warnings)
    mask_dims = {
        object_type: n_channels + index
        for index, object_type in enumerate(
            name for name in OBJECT_TYPES if name in masks_by_type)
    }
    if os.path.exists(os.path.join(destination, "measurements",
                                   "measurements.db")):
        errors.append(
            f"{destination} already has measurements/measurements.db. "
            "Choose a new destination; existing measurements are never "
            "silently replaced.")
    return ExternalMaskPlan(
        groups=groups, images=image_plan, masks=masks_by_type,
        destination=destination, n_channels=n_channels,
        mask_dims=mask_dims, errors=errors, warnings=warnings)


def _save_npy(path: str, array: np.ndarray) -> str:
    """Atomically replace one NumPy destination through a hidden partial file.

    An interrupted write cannot look like a finished field to array readers.
    Ordinary save/replace failures remove the scratch file and preserve the
    previous destination; a killed process can leave only a hidden partial.

    :param path: destination ``.npy`` path.
    :param array: array to serialize.
    :returns: ``path`` after replacement succeeds.
    """
    from .io import _save_array_atomic

    return _save_array_atomic(path, array)


def _tables(path: str) -> List[str]:
    """Return sorted SQLite table names, or none for a missing database."""
    if not os.path.isfile(path):
        return []
    with sqlite3.connect(path, timeout=30) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "ORDER BY name").fetchall()
    return [str(row[0]) for row in rows]


def _label_plane_as_uint16(array: Any, path: str, object_type: str,
                           shape: Tuple[int, ...], stem: str, *,
                           integer_only: bool = False) -> np.ndarray:
    """Check one label plane against the Measure array contract.

    :param array: the label image as read.
    :param path: the file it came from, named in every refusal.
    :param object_type: the object the labels are imported as.
    :param shape: the intensity shape of the field it belongs to.
    :param stem: the field stem, for the messages.
    :param integer_only: refuse a boolean or floating-point plane rather than
        casting it. Measure's label-image import (item 790) sets it: a
        probability map or a binary mask cast to integers is measured as one
        object per image, or as nothing, without a word.
    :returns: the labels as ``uint16``.
    :raises ConfigurationError: on a refused dtype, a shape that differs from
        the intensities, a negative ID or an ID above 65535.
    """
    array = np.asarray(array)
    if integer_only and (array.dtype == np.bool_
                         or not np.issubdtype(array.dtype, np.integer)):
        raise ConfigurationError(
            f"{path}: {object_type} labels are {array.dtype}, not an integer "
            "label image. Export one integer ID per object with 0 as "
            "background; a binary mask or a probability map has to be "
            "labelled first.")
    if array.shape != shape:
        raise ConfigurationError(
            f"{path}: {object_type} mask shape {array.shape} "
            f"does not match intensity shape {shape} for {stem}.")
    if np.any(array < 0):
        raise ConfigurationError(
            f"{path}: label masks cannot contain negative IDs.")
    maximum = int(np.max(array, initial=0))
    if maximum > np.iinfo(np.uint16).max:
        raise ConfigurationError(
            f"{path}: label ID {maximum} exceeds the maximum "
            "65535 supported by the Measure array contract.")
    return array.astype(np.uint16, copy=False)


def _intensity_plane_as_uint16(plane: np.ndarray, stem: str) -> np.ndarray:
    """Check one intensity plane against the Measure ``uint16`` contract.

    :param plane: the intensity plane.
    :param stem: the field stem, for the messages.
    :returns: the plane as ``uint16``; integer values that fit are kept
        exactly.
    :raises ConfigurationError: on NaN, infinity, fractional floats or values
        outside 0-65535.
    """
    if np.issubdtype(plane.dtype, np.floating):
        if not np.all(np.isfinite(plane)):
            raise ConfigurationError(
                f"{stem}: intensity data contain NaN or infinity.")
        if not np.all(plane == np.floor(plane)):
            raise ConfigurationError(
                f"{stem}: floating-point intensities would lose "
                "precision in Measure's uint16 arrays. Rescale and "
                "export them as 8- or 16-bit images first.")
    if float(np.min(plane, initial=0)) < 0 or \
            float(np.max(plane, initial=0)) > np.iinfo(np.uint16).max:
        raise ConfigurationError(
            f"{stem}: intensity values must fit the Measure uint16 "
            "contract (0–65535). Rescale the source images first.")
    return plane.astype(np.uint16, copy=False)


def run_external_masks(plan: ExternalMaskPlan,
                       settings: Optional[Mapping[str, Any]] = None
                       ) -> ExternalMaskResult:
    """Materialize ``plan`` and call the standard Measure pipeline.

    :param plan: Validated plan from :func:`plan_external_masks`, whose
        ``ok`` property must be True.
    :param settings: Partial settings mapping accepted by
        :func:`default_settings`; it carries the full Measure contract and
        supplies ``overwrite`` for the intensity conversion and
        ``channels``, ``png_dims``, ``crop_mode`` and ``cytoplasm`` for the
        Measure call.
    :returns: Result describing the written project, its merged arrays,
        measurements database and tables, and the plan used.
    :raises ConfigurationError: If the plan is not ``ok``, if a field's
        channel count, shapes, dtypes or label IDs violate the Measure
        uint16 contract, or if Measure finishes without a required table.
    """
    if not plan.ok:
        raise ConfigurationError(
            "External-mask import refused; nothing was written:\n  "
            + "\n  ".join([*plan.images.errors, *plan.errors]))
    resolved = default_settings(settings)
    dst = plan.destination
    os.makedirs(dst, exist_ok=True)
    images_dir = os.path.join(dst, "images")
    conversion = cv.convert(plan.images, images_dir,
                            overwrite=bool(resolved.get("overwrite", False)))
    available = {mapping.target for mapping in conversion.written}
    available.update(mapping.target for mapping in conversion.existing)

    mappings_by_stem: Dict[str, Dict[int, cv.Mapping]] = {}
    for mapping in plan.images.mappings:
        if _stem(mapping) not in plan.stems:
            continue
        mappings_by_stem.setdefault(_stem(mapping), {})[
            int(mapping.channel)] = mapping

    merged_paths: List[str] = []
    from .foreign import _read_mask
    for stem in plan.stems:
        channels = mappings_by_stem.get(stem, {})
        if len(channels) != plan.n_channels:
            raise ConfigurationError(
                f"{stem}: expected {plan.n_channels} channels, found "
                f"{len(channels)}.")
        image_planes = []
        for channel in sorted(channels):
            mapping = channels[channel]
            if mapping.target not in available:
                raise ConfigurationError(
                    f"{stem}: converted intensity image is missing: "
                    f"{mapping.target}")
            image_planes.append(np.asarray(
                cv._read_source(cv.SourceImage(
                    path=os.path.join(images_dir, mapping.target),
                    plate="", well="", field="",
                    meta={"ext": cv._split_ext(mapping.target)[1]},
                ))
            ).squeeze())
        shape = image_planes[0].shape
        if any(plane.shape != shape for plane in image_planes):
            raise ConfigurationError(
                f"{stem}: intensity channels have inconsistent shapes.")

        mask_planes = []
        for object_type in plan.object_types:
            match = plan.masks[object_type][stem]
            integer = _label_plane_as_uint16(
                _read_mask(match.path), match.path, object_type, shape, stem)
            mask_planes.append(integer)
            _save_npy(os.path.join(
                dst, "masks", f"{object_type}_mask_stack", f"{stem}.npy"),
                integer)

        image_planes = [_intensity_plane_as_uint16(plane, stem)
                        for plane in image_planes]
        stack = np.stack(image_planes, axis=-1)
        _save_npy(os.path.join(dst, "stack", f"{stem}.npy"), stack)
        merged = np.stack(
            [plane.astype(np.uint16, copy=False)
             for plane in [*image_planes, *mask_planes]], axis=-1)
        merged_paths.append(_save_npy(
            os.path.join(dst, "merged", f"{stem}.npy"), merged))

    plane_layout = {
        "version": 1,
        "intensity_channels": list(range(plan.n_channels)),
        "mask_plane_order": list(plan.object_types),
        "mask_dims": dict(plan.mask_dims),
    }
    with open(os.path.join(dst, "merged", crops.MERGED_LAYOUT_SIDECAR),
              "w", encoding="utf-8") as handle:
        json.dump(plane_layout, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest = {
        "module": "external_masks",
        "destination": dst,
        "n_channels": plan.n_channels,
        "mask_dims": plan.mask_dims,
        "groups": [group.to_dict() for group in plan.groups],
        "merged": [os.path.basename(path) for path in merged_paths],
    }
    with open(os.path.join(dst, "external_mask_import.json"), "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    measure_settings = dict(resolved)
    for key in (
        "inputs", "dst", "recursive", "layout", "z_handling",
        "plate_naming", "overwrite", "preview_only",
    ):
        measure_settings.pop(key, None)
    measure_settings["src"] = os.path.join(dst, "merged")
    measure_settings["channels"] = (
        list(range(plan.n_channels))
        if not measure_settings.get("channels")
        else list(measure_settings["channels"])
    )
    measure_settings["png_dims"] = (
        list(range(min(plan.n_channels, 3)))
        if not measure_settings.get("png_dims")
        else list(measure_settings["png_dims"])
    )
    for object_type in OBJECT_TYPES:
        measure_settings[f"{object_type}_mask_dim"] = (
            plan.mask_dims.get(object_type))
    available_crops = list(plan.object_types)
    if "cell" in plan.object_types and measure_settings.get("cytoplasm"):
        available_crops.append("cytoplasm")
    requested = measure_settings.get("crop_mode") or []
    if isinstance(requested, str):
        requested = [requested]
    requested = [name for name in requested if name in available_crops]
    measure_settings["crop_mode"] = (
        requested or available_crops[:1])
    from .measure import measure_crop
    measure_crop(measure_settings)

    db_path = os.path.join(dst, "measurements", "measurements.db")
    tables = _tables(db_path)
    expected = set(plan.object_types)
    if "cell" in expected and measure_settings.get("cytoplasm"):
        expected.add("cytoplasm")
    if measure_settings.get("save_png"):
        expected.add("png_list")
    missing = sorted(expected - set(tables))
    if missing:
        raise ConfigurationError(
            "Measure finished without required output table(s): "
            + ", ".join(missing))
    return ExternalMaskResult(
        destination=dst, merged=merged_paths, db_path=db_path,
        tables=tables, data_dir=os.path.join(dst, "data"), plan=plan)


def prepare_external_masks(settings: Optional[Mapping[str, Any]] = None
                           ) -> Any:
    """Plan an external-mask import, print the preview, and run it.

    The plan summary is always printed to stdout.  Unless ``preview_only``
    is set, the project is written and Measure is run, and the result
    summary is printed too.

    :param settings: Partial settings mapping accepted by
        :func:`default_settings`.
    :returns: The :class:`ExternalMaskPlan` when ``preview_only`` is set,
        otherwise the :class:`ExternalMaskResult` from
        :func:`run_external_masks`.
    """
    resolved = default_settings(settings)
    plan = plan_external_masks(resolved)
    print(plan.summary())
    if resolved.get("preview_only"):
        return plan
    result = run_external_masks(plan, resolved)
    print(result.summary())
    return result


#: Measure's label-image import settings (item 790). The external-mask
#: importer has its own inputs, so its form and its Measure call leave them out.
_LABEL_IMPORT_SETTINGS = (
    "label_import", "label_import_images", "label_import_image_regex",
    "label_import_channels", "cell_label_folder", "nucleus_label_folder",
    "pathogen_label_folder", "cell_label_regex", "nucleus_label_regex",
    "pathogen_label_regex",
)

#: Objects Measure's label-image import reads (item 790, alpha), in the
#: order their planes follow the intensity channels in ``merged/``.
_LABEL_IMPORT_OBJECTS = ("cell", "nucleus", "pathogen")

#: The record a label-image import leaves in the plate folder: every source
#: file of every field, the patterns, the plane layout and what was written.
_LABEL_IMPORT_MANIFEST = "label_import.json"

#: Folders spaCR itself writes into a plate folder. The label search never
#: descends into them, so a label folder that is the plate folder does not
#: read back its own output.
_SPACR_OUTPUT_FOLDERS = frozenset({
    "stack", "merged", "masks", "measurements", "data", "norm_channel_stack",
    "orig", "vvvv",
})

#: Files the import reads: every format the converter opens, plus plain
#: label arrays saved with :func:`numpy.save`.
_LABEL_IMPORT_SUFFIXES = (*SUPPORTED_SUFFIXES, ".npy")

#: Groups a raw-image pattern and a label pattern must name.
_LABEL_IMPORT_IMAGE_GROUPS = ("wellID", "fieldID", "chanID")
_LABEL_IMPORT_LABEL_GROUPS = ("wellID", "fieldID")


def _label_import_files(folder: str, *, recursive: bool) -> List[str]:
    """List the importable files in ``folder``, sorted, skipping dot-files.

    :param folder: the folder to list.
    :param recursive: descend into subfolders, except spaCR's own output
        folders (:data:`_SPACR_OUTPUT_FOLDERS`).
    :returns: absolute paths.
    :raises ConfigurationError: when ``folder`` is not a folder.
    """
    root = os.path.abspath(os.path.expanduser(str(folder)))
    if not os.path.isdir(root):
        raise ConfigurationError(f"{folder} is not a folder.")
    found = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not name.startswith(".")
                         and name not in _SPACR_OUTPUT_FOLDERS)
        for name in sorted(files):
            lower = name.lower()
            if not name.startswith(".") and lower.endswith(
                    _LABEL_IMPORT_SUFFIXES):
                found.append(os.path.join(base, name))
        if not recursive:
            break
    return found


def _label_import_pattern(paths: Sequence[str], explicit: Any,
                          required: Sequence[str], setting: str
                          ) -> Tuple["re.Pattern[str]", str, bool]:
    """The filename pattern for one folder: the user's, or an inferred one.

    A blank setting is inferred with :func:`spacr.regex_infer.propose` from
    the folder's own file names, and accepted only when one proposal names
    every required group and parses every file.

    :param paths: the folder's files.
    :param explicit: the setting's value; blank means infer.
    :param required: named groups the pattern must have.
    :param setting: the setting's name, for the messages.
    :returns: ``(compiled, pattern, inferred)``. The pattern is searched in
        each file name, so it need not match the whole name.
    :raises ConfigurationError: on a pattern that does not compile, lacks a
        required group, or cannot be inferred.
    """
    text = str(explicit or "").strip()
    if text:
        try:
            compiled = re.compile(text)
        except re.error as exc:
            raise ConfigurationError(
                f"{setting}: {text!r} is not a regular expression ({exc}).")
        missing = [name for name in required
                   if name not in compiled.groupindex]
        if missing:
            raise ConfigurationError(
                f"{setting}: the pattern has no "
                f"{', '.join(f'(?P<{name}>...)' for name in missing)} group; "
                f"it needs {', '.join(required)}.")
        return compiled, text, False
    from . import regex_infer

    names = [os.path.basename(path) for path in paths]
    for proposal in regex_infer.propose(names):
        if proposal.coverage == 1.0 and all(
                name in proposal.fields for name in required):
            pattern = f"^(?:{proposal.pattern})$"
            return re.compile(pattern), pattern, True
    raise ConfigurationError(
        f"{setting}: no filename pattern with {', '.join(required)} could be "
        f"inferred from {len(names)} file(s)"
        f"{' such as ' + names[0] if names else ''}. Type a regular "
        "expression with those named groups, for example "
        r"(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+).")


def _label_import_index(token: Any) -> str:
    """A well-independent identity token: digits lose their leading zeros."""
    text = str(token).strip()
    return str(int(text)) if text.isdigit() else text


def _label_import_key(found: "re.Match[str]", plate: Optional[str]
                      ) -> Tuple[Optional[str], str, str, Optional[str]]:
    """``(plate, well, field, time)`` from one filename match.

    :param found: the match.
    :param plate: the plate when the pattern names none; ``None`` leaves it
        open, as for a label file matched to its field by well and field.
    """
    groups = found.groupdict()
    time = groups.get("timeID")
    return (groups.get("plateID") or plate,
            str(groups["wellID"]).strip().upper(),
            _label_import_index(groups["fieldID"]),
            None if time is None else _label_import_index(time))


def _label_import_channel_order(channel: str) -> Tuple[int, Any]:
    """Sort numeric channel ids by value, then the rest by name."""
    return (0, int(channel)) if channel.isdigit() else (1, channel)


def _plan_label_import(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Pair raw channel images with label images for Measure, reading no pixels.

    :param settings: the Measure settings: ``src`` (the plate folder the
        project is written to), ``label_import_images``,
        ``label_import_image_regex``, ``label_import_channels`` and, per
        object, ``<object>_label_folder`` and ``<object>_label_regex``.
    :returns: the plan: ``src``, ``image_folder``, ``image_pattern``,
        ``channels``, ``objects``, ``label_patterns``, ``fields`` (one
        ``(stem, channel paths, {object: label path})`` per complete field),
        ``skipped``, ``errors`` and ``warnings``.
    """
    from .io import _escaped_field_stem

    src = settings.get("src")
    if isinstance(src, (list, tuple)):
        if len(src) != 1:
            raise ConfigurationError(
                "label_import builds one plate folder at a time; src names "
                f"{len(src)}. Measure each plate in its own run.")
        src = src[0]
    if not str(src or "").strip():
        raise ConfigurationError(
            "label_import needs src: the plate folder the merged arrays and "
            "measurements are written to.")
    src = os.path.abspath(os.path.expanduser(str(src)))
    if os.path.basename(src) == "merged":
        src = os.path.dirname(src)
    image_folder = os.path.abspath(os.path.expanduser(str(
        settings.get("label_import_images") or src)))
    errors: List[str] = []
    warnings: List[str] = []

    images = _label_import_files(image_folder, recursive=False)
    image_re, image_pattern, image_inferred = _label_import_pattern(
        images, settings.get("label_import_image_regex"),
        _LABEL_IMPORT_IMAGE_GROUPS, "label_import_image_regex")
    plate = os.path.basename(os.path.normpath(image_folder))
    fields: Dict[Tuple[Any, ...], Dict[str, str]] = {}
    ignored_images = 0
    for path in images:
        found = image_re.search(os.path.basename(path))
        if found is None:
            ignored_images += 1
            continue
        key = _label_import_key(found, plate)
        channel = _label_import_index(found.group("chanID"))
        per_channel = fields.setdefault(key, {})
        if channel in per_channel:
            errors.append(
                f"{per_channel[channel]} and {path} are both channel "
                f"{channel} of {'_'.join(str(part) for part in key if part)}.")
            continue
        per_channel[channel] = path
    if not fields:
        errors.append(
            f"No file in {image_folder} matches label_import_image_regex "
            f"{image_pattern!r}.")
    if ignored_images:
        warnings.append(
            f"{ignored_images} file(s) in {image_folder} do not match the "
            "image pattern and are not imported.")
    found_channels = sorted({channel for per_channel in fields.values()
                             for channel in per_channel},
                            key=_label_import_channel_order)
    wanted = settings.get("label_import_channels")
    channels = ([_label_import_index(channel) for channel in wanted]
                if wanted else found_channels)
    unknown = [channel for channel in channels if channel not in found_channels]
    if unknown:
        errors.append(
            f"label_import_channels names channel(s) {', '.join(unknown)}; "
            f"the images carry {', '.join(found_channels) or 'none'}.")

    label_folders = {
        "cell": settings.get("cell_label_folder"),
        "nucleus": settings.get("nucleus_label_folder"),
        "pathogen": settings.get("pathogen_label_folder"),
    }
    label_regexes = {
        "cell": settings.get("cell_label_regex"),
        "nucleus": settings.get("nucleus_label_regex"),
        "pathogen": settings.get("pathogen_label_regex"),
    }
    objects = [name for name in _LABEL_IMPORT_OBJECTS
               if str(label_folders[name] or "").strip()]
    if not objects:
        errors.append(
            "label_import is on but no label folder is set: fill "
            "cell_label_folder, nucleus_label_folder or "
            "pathogen_label_folder.")
    by_well_field: Dict[Tuple[str, str], List[Tuple[Any, ...]]] = {}
    for key in fields:
        by_well_field.setdefault((key[1], key[2]), []).append(key)
    labels: Dict[str, Dict[Tuple[Any, ...], str]] = {}
    label_patterns: Dict[str, Dict[str, Any]] = {}
    for name in objects:
        folder = os.path.abspath(os.path.expanduser(
            str(label_folders[name])))
        files = _label_import_files(folder, recursive=True)
        compiled, pattern, inferred = _label_import_pattern(
            files, label_regexes[name],
            _LABEL_IMPORT_LABEL_GROUPS, f"{name}_label_regex")
        label_patterns[name] = {"folder": folder, "pattern": pattern,
                                "inferred": inferred}
        matched = labels.setdefault(name, {})
        unpaired = 0
        for path in files:
            found = compiled.search(os.path.basename(path))
            if found is None:
                continue
            want = _label_import_key(found, None)
            targets = [key for key in by_well_field.get((want[1], want[2]), [])
                       if (want[0] is None or key[0] == want[0])
                       and (want[3] is None or key[3] == want[3])]
            if len(targets) != 1:
                if targets:
                    errors.append(
                        f"{path} fits {len(targets)} image fields; add "
                        f"plateID or timeID to {name}_label_regex.")
                else:
                    unpaired += 1
                continue
            if targets[0] in matched:
                errors.append(
                    f"{matched[targets[0]]} and {path} are both {name} "
                    "labels of one field; make "
                    f"{name}_label_regex match only one of them.")
                continue
            matched[targets[0]] = path
        if not matched:
            errors.append(
                f"No file in {folder} matches {name}_label_regex "
                f"{pattern!r} and an image field.")
        if unpaired:
            warnings.append(
                f"{unpaired} {name} label file(s) match the pattern but no "
                "image field; they are not imported.")

    ready, skipped = [], []
    for key in sorted(fields, key=lambda item: tuple(str(part) for part in item)):
        per_channel = fields[key]
        absent = [channel for channel in channels
                  if channel not in per_channel]
        absent += [name for name in objects
                   if key not in labels.get(name, {})]
        stem = _escaped_field_stem(key[0], key[1], key[2], key[3])
        if absent:
            skipped.append(f"{stem}: no {', '.join(absent)}")
            continue
        ready.append((stem, [per_channel[channel] for channel in channels],
                      {name: labels[name][key] for name in objects}))
    if skipped:
        warnings.append(
            f"{len(skipped)} field(s) lack a channel or a label and are not "
            f"imported, e.g. {skipped[0]}.")
    if fields and objects and not ready and not errors:
        errors.append("No field has every channel and every label.")
    return {
        "src": src, "image_folder": image_folder,
        "image_pattern": image_pattern, "image_pattern_inferred": image_inferred,
        "channels": channels, "objects": objects,
        "label_patterns": label_patterns, "fields": ready,
        "skipped": skipped, "errors": errors, "warnings": warnings,
    }


def _label_import_summary(plan: Mapping[str, Any]) -> str:
    """The plan in a few lines, printed before anything is written."""
    lines = [
        f"Label-image import into {plan['src']}",
        f"  images: {plan['image_folder']} "
        f"(pattern {plan['image_pattern']}"
        f"{', inferred' if plan['image_pattern_inferred'] else ''})",
        f"  channels, in stack order: {', '.join(plan['channels'])}",
    ]
    for name, record in plan["label_patterns"].items():
        lines.append(
            f"  {name} labels: {record['folder']} (pattern "
            f"{record['pattern']}{', inferred' if record['inferred'] else ''})")
    lines.append(f"  fields ready: {len(plan['fields'])}")
    lines.extend(f"  warning: {message}" for message in plan["warnings"])
    lines.extend(f"  error: {message}" for message in plan["errors"])
    return "\n".join(lines)


def _label_import_plane(path: str, what: str) -> np.ndarray:
    """Read one single-plane image through spaCR's converter readers.

    :param path: the image or label file.
    :param what: what it is, for the message.
    :raises ConfigurationError: when the file holds more than one plane.
    """
    from .foreign import _source_for

    raw = np.asarray(cv._read_source(_source_for(path)))
    plane = np.squeeze(raw)
    if plane.ndim != 2:
        raise ConfigurationError(
            f"{path}: a {what} file must hold one 2-D plane, got shape "
            f"{raw.shape}. Export one plane per file.")
    return plane


def _label_import_stat(path: str) -> List[Any]:
    """``[path, size, mtime_ns]``: enough to see a source file was replaced."""
    info = os.stat(path)
    return [path, int(info.st_size), int(info.st_mtime_ns)]


def _label_import_signature(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """What a finished import was built from: patterns, channels and files."""
    stat = _label_import_stat
    return {
        "image_pattern": plan["image_pattern"],
        "channels": list(plan["channels"]),
        "objects": list(plan["objects"]),
        "label_patterns": {name: record["pattern"] for name, record
                           in plan["label_patterns"].items()},
        "fields": {stem: {"images": [stat(path) for path in images],
                          "labels": {name: stat(path)
                                     for name, path in found.items()}}
                   for stem, images, found in plan["fields"]},
    }


def _label_import_existing_arrays(src: str) -> List[str]:
    """Arrays already in the folders the import writes, which it must not mix."""
    folders = [os.path.join(src, "stack"), os.path.join(src, "merged")]
    masks = os.path.join(src, "masks")
    if os.path.isdir(masks):
        folders.extend(os.path.join(masks, name)
                       for name in sorted(os.listdir(masks))
                       if name.endswith("_mask_stack"))
    found = []
    for folder in folders:
        if os.path.isdir(folder):
            found.extend(os.path.join(folder, name)
                         for name in sorted(os.listdir(folder))
                         if not name.startswith(".")
                         and name.lower().endswith((".npy", ".tif", ".tiff")))
    return found


def _write_label_import(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Write ``stack/``, ``masks/`` and ``merged/`` for a checked plan.

    The intensities go to ``stack/`` and each object's labels to
    ``masks/<object>_mask_stack/``, both through spaCR's atomic array
    writer; :func:`spacr.io._load_and_concatenate_arrays`, the merge Mask
    runs, then writes ``merged/`` and its plane layout. A plate folder that
    already holds arrays from anything other than an earlier import of the
    same files is refused. An import whose files, patterns and channels are
    unchanged is reused as it is.

    :param plan: from :func:`_plan_label_import`, without errors.
    :returns: the manifest written to ``label_import.json``.
    :raises ConfigurationError: on foreign arrays in the plate folder, or a
        file that fails the dtype, plane or size checks.
    """
    from .io import (_load_and_concatenate_arrays, _replace_atomically,
                     _save_array_atomic)
    from .crops import read_merged_plane_layout
    from . import __version__

    src = plan["src"]
    manifest_path = os.path.join(src, _LABEL_IMPORT_MANIFEST)
    signature = _label_import_signature(plan)
    previous = None
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
    stems = [stem for stem, _images, _labels in plan["fields"]]
    merged_dir = os.path.join(src, "merged")
    if previous and previous.get("signature") == signature and all(
            os.path.isfile(os.path.join(merged_dir, f"{stem}.npy"))
            for stem in stems):
        print(f"Label-image import: {len(stems)} field(s) already built "
              f"from the same files; reusing {merged_dir}.")
        return previous
    if previous:
        for path in previous.get("written", []):
            target = os.path.join(src, path)
            if os.path.isfile(target):
                os.remove(target)
    existing = _label_import_existing_arrays(src)
    if existing:
        raise ConfigurationError(
            f"{src} already holds {len(existing)} array(s) that this import "
            f"did not write, starting with {existing[0]}. The label-image "
            "import writes stack/, masks/ and merged/ itself; choose an "
            "empty plate folder for src, or turn label_import off to measure "
            "the arrays that are there.")

    objects = list(plan["objects"])
    written: List[str] = []
    n_channels = len(plan["channels"])
    for stem, image_paths, label_paths in plan["fields"]:
        planes = [_intensity_plane_as_uint16(
            _label_import_plane(path, "raw image"), stem)
            for path in image_paths]
        shape = planes[0].shape
        if any(plane.shape != shape for plane in planes):
            raise ConfigurationError(
                f"{stem}: intensity channels have inconsistent shapes "
                f"({', '.join(str(plane.shape) for plane in planes)}).")
        _save_array_atomic(os.path.join(src, "stack", f"{stem}.npy"),
                           np.stack(planes, axis=-1))
        written.append(os.path.join("stack", f"{stem}.npy"))
        for name in objects:
            path = label_paths[name]
            labels = _label_plane_as_uint16(
                _label_import_plane(path, f"{name} label"), path, name,
                shape, stem, integer_only=True)
            relative = os.path.join("masks", f"{name}_mask_stack",
                                    f"{stem}.npy")
            _save_array_atomic(os.path.join(src, relative), labels)
            written.append(relative)
    mask_dims = {name: n_channels + index
                 for index, name in enumerate(objects)}
    _load_and_concatenate_arrays(
        src, None, mask_dims.get("cell"), mask_dims.get("nucleus"),
        mask_dims.get("pathogen"), None)
    missing = [stem for stem in stems
               if not os.path.isfile(os.path.join(merged_dir, f"{stem}.npy"))]
    if missing:
        raise ConfigurationError(
            f"The merge wrote no merged array for {len(missing)} field(s), "
            f"starting with {missing[0]}.")
    written.extend(os.path.join("merged", f"{stem}.npy") for stem in stems)
    manifest = {
        "module": "measure label-image import (item 790)",
        "spacr_version": __version__,
        "src": src,
        "image_folder": plan["image_folder"],
        "image_pattern": plan["image_pattern"],
        "image_pattern_inferred": plan["image_pattern_inferred"],
        "channels": list(plan["channels"]),
        "label_patterns": plan["label_patterns"],
        "plane_layout": read_merged_plane_layout(merged_dir),
        "skipped": list(plan["skipped"]),
        "signature": signature,
        "written": written,
    }
    _replace_atomically(
        manifest_path,
        lambda handle: handle.write(json.dumps(
            manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"))
    return manifest


def _label_import_dry_run(settings: Mapping[str, Any]) -> List[Any]:
    """Print the label-image import plan for a Measure ``dry_run``.

    :param settings: the Measure settings.
    :returns: one :class:`spacr.validate.Problem` per blocking problem and
        per warning; nothing is written.
    """
    from .validate import Problem

    plan = _plan_label_import(settings)
    print(_label_import_summary(plan))
    fix = ("Check label_import_images, the label folders and their "
           "patterns.")
    return ([Problem("error", "label_import", message, fix)
             for message in plan["errors"]]
            + [Problem("warning", "label_import", message, fix)
               for message in plan["warnings"]])


def _import_label_images_for_measure(settings: Mapping[str, Any]
                                     ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build a Measure plate folder from raw images and label images (item 790).

    Called by :func:`spacr.measure.measure_crop` when ``label_import`` is on:
    the raw channel images and the label images of another tool
    (CellProfiler, Cellpose, ilastik, ...) are paired by filename pattern,
    checked and written as spaCR's own ``merged/`` stacks, so Measure then
    runs on them exactly as on masks it made.

    :param settings: the Measure settings.
    :returns: ``(settings, manifest)``: the settings with ``src`` set to the
        plate folder, the mask dimensions set from the import's plane layout
        and ``channels`` limited to the imported channels; and the manifest
        written to ``label_import.json``.
    :raises ConfigurationError: when the plan has a blocking problem, or a
        file fails its checks; nothing is measured.
    """
    plan = _plan_label_import(settings)
    print(_label_import_summary(plan))
    if plan["errors"]:
        raise ConfigurationError(
            "Label-image import refused; nothing was written:\n  "
            + "\n  ".join(plan["errors"]))
    manifest = _write_label_import(plan)
    resolved = dict(settings)
    resolved["src"] = plan["src"]
    dims = manifest["plane_layout"]["mask_dims"]
    for name in _LABEL_IMPORT_OBJECTS:
        resolved[f"{name}_mask_dim"] = dims.get(name)
    n_channels = len(plan["channels"])
    requested = resolved.get("channels")
    if isinstance(requested, list) and any(
            isinstance(index, int) and index >= n_channels
            for index in requested):
        kept = [index for index in requested
                if not isinstance(index, int) or index < n_channels]
        resolved["channels"] = kept or list(range(n_channels))
        print(f"Label-image import: {n_channels} channel(s) imported, so "
              f"channels {requested} is measured as {resolved['channels']}.")
    return resolved, manifest


def register_settings(replace: bool = False) -> bool:
    """Publish this module's settings help through the shared registry.

    Registers expected types and tooltips for the seven importer-specific
    controls (``inputs``, ``recursive``, ``layout``, ``z_handling``,
    ``plate_naming``, ``overwrite`` and ``preview_only``) beside the defaults,
    so the inventory, validation types and tooltip prose stay together. A key
    that already has a shared tooltip keeps it, and the Measure settings
    returned by :func:`default_settings` retain their existing shared
    declarations. Nothing is registered when a registration exists and
    ``replace`` is False.

    :param replace: replace this module's existing defaults registration.
    :returns: whether a new registration was made.
    """
    from .settings import has_registered_defaults, register_defaults
    from .settings import tooltips as shared_tooltips

    if has_registered_defaults("external_masks") and not replace:
        return False
    types = {
        "inputs": list,
        "recursive": bool,
        "layout": str,
        "z_handling": str,
        "plate_naming": str,
        "overwrite": bool,
        "preview_only": bool,
    }
    tips = {
        "inputs":
            "(list) - Image and external label-mask files or folders to "
            "import. The preview groups them by source and proposes whether "
            "each group is an intensity image, an object mask, or ignored. "
            "Default [].",
        "recursive":
            "(bool) - Search inside subfolders of every input folder. Turn "
            "this off when only files directly inside each selected folder "
            "belong to the import. Default True.",
        "layout":
            "(str) - Naming-layout hint used to read plate, well, field, "
            "channel, Z, and time identifiers from source files. 'auto' "
            "detects the supported layout from the filenames. Default "
            "'auto'.",
        "z_handling":
            "(str) - How multiple Z planes become a 2-D Measure input. "
            "'max' takes a maximum-intensity projection and 'first' keeps "
            "only the first plane; inputs that still contain separate planes "
            "are rejected. Default 'max'.",
        "plate_naming":
            "(str) - How imported plates are named when the source does not "
            "provide one. 'index' assigns stable plate numbers in discovered "
            "input order. Default 'index'.",
        "overwrite":
            "(bool) - Allow the importer to replace files in an existing "
            "destination project. Leave this off to stop before previously "
            "written images, masks, or measurements can be replaced. Default "
            "False.",
        "preview_only":
            "(bool) - Build and print the complete input-to-mask assignment "
            "plan without writing a project or running Measure. Use this to "
            "review automatic role and object-type detection first. Default "
            "False.",
    }
    tips = {key: value for key, value in tips.items()
            if key not in shared_tooltips}
    register_defaults(
        "external_masks", default_settings, replace=replace,
        expected_types=types, tooltips=tips,
        description=(
            "Import images and externally generated label masks as a "
            "measured spaCR project ready for annotation."),
    )
    return True


register_settings()


__all__ = [
    "InputGroup", "MaskMatch", "ExternalMaskPlan", "ExternalMaskResult",
    "SUPPORTED_SUFFIXES", "OBJECT_TYPES", "ROLES",
    "detect_inputs", "default_settings", "plan_external_masks",
    "run_external_masks", "prepare_external_masks", "register_settings",
]
