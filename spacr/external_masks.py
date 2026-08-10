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

import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import convert as cv
from . import crops
from .errors import ConfigurationError


SUPPORTED_SUFFIXES = (
    ".tif", ".tiff", ".ome.tif", ".ome.tiff",
    ".png", ".jpg", ".jpeg", ".bmp",
)
OBJECT_TYPES = tuple(crops.MASK_PLANE_ORDER)
ROLES = ("image", "mask", "ignore")

_MASK_WORDS = re.compile(
    r"(?i)(?:^|[_\-. ])(?:mask|masks|label|labels|labelled|labeled|"
    r"instance|instances|seg|segmentation|outline)(?:$|[_\-. ])"
)
_OBJECT_PATTERNS = (
    ("nucleus", re.compile(r"(?i)(?:^|[_\-. ])(?:nucleus|nuclei|nuclear|nuc)(?:$|[_\-. ])")),
    ("pathogen", re.compile(r"(?i)(?:^|[_\-. ])(?:pathogen|parasite|bacteria|bacterial)(?:$|[_\-. ])")),
    ("organelle", re.compile(r"(?i)(?:^|[_\-. ])(?:organelle|organell|mitochondria|mitochondrion)(?:$|[_\-. ])")),
    ("cell", re.compile(r"(?i)(?:^|[_\-. ])(?:cell|cells|wholecell|cytoplasm)(?:$|[_\-. ])")),
)


@dataclass
class InputGroup:
    """A set of files sharing one proposed role and object type."""

    key: str
    root: str
    paths: List[str]
    role: str
    object_type: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Any) -> "InputGroup":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ConfigurationError(
                "Each external-mask input must be an InputGroup or mapping.")
        return cls(
            key=str(value.get("key") or ""),
            root=os.path.abspath(str(value.get("root") or ".")),
            paths=[os.path.abspath(str(path))
                   for path in value.get("paths", [])],
            role=str(value.get("role") or "ignore"),
            object_type=(str(value["object_type"])
                         if value.get("object_type") else None),
            confidence=float(value.get("confidence") or 0.0),
            reason=str(value.get("reason") or ""),
        )


@dataclass(frozen=True)
class MaskMatch:
    """One label-mask file matched to an intensity-image field.

    :ivar path: Absolute mask-file path.
    :ivar object_type: spaCR object role such as ``cell`` or ``nucleus``.
    :ivar stem: Canonical field stem shared with the intensity image.
    :ivar match: Description of the matching rule that succeeded.
    """

    path: str
    object_type: str
    stem: str
    match: str


@dataclass
class ExternalMaskPlan:
    """Read-only preview of an external-mask import."""

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
        return sorted(set.intersection(
            *(set(per_stem) for per_stem in self.masks.values())
        )) if self.masks else []

    @property
    def object_types(self) -> List[str]:
        return [name for name in OBJECT_TYPES if name in self.masks]

    @property
    def ok(self) -> bool:
        return bool(self.images.ok and self.stems and not self.errors)

    def summary(self) -> str:
        lines = [
            "External masks → Measure project (preview; nothing written)",
            f"  intensity mappings: {len(self.images)}",
            f"  fields ready: {len(self.stems)}",
            f"  intensity channels: {self.n_channels}",
            f"  mask types: {', '.join(self.object_types) or 'none'}",
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

    :ivar destination: Root of the generated spaCR project.
    :ivar merged: Paths of merged image/mask arrays.
    :ivar db_path: Generated measurements database.
    :ivar tables: Measurement tables written to the database.
    :ivar data_dir: Generated annotation-crop directory.
    :ivar plan: Validated read-only plan used for the import.
    """

    destination: str
    merged: List[str]
    db_path: str
    tables: List[str]
    data_dir: str
    plan: ExternalMaskPlan

    def summary(self) -> str:
        return (
            f"Prepared {len(self.merged)} field(s) in {self.destination}. "
            f"measurements.db tables: {', '.join(self.tables) or 'none'}. "
            f"Annotation crops: {self.data_dir}"
        )


def _all_files(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path] if _supported(path) else []
    if not path.is_dir():
        return []
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(candidate for candidate in iterator
                  if candidate.is_file() and _supported(candidate))


def _supported(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def _suggest_object(name: str) -> Optional[str]:
    # Include parent folders: externally generated masks are commonly named
    # ``cell_masks/fov001.tif`` rather than ``fov001_cell_mask.tif``.
    stem = cv._split_ext(str(name))[0]
    for object_type, pattern in _OBJECT_PATTERNS:
        if pattern.search(stem):
            return object_type
    return None


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
    # Cap the sample without changing its value distribution materially.
    stride = max(int(np.sqrt(array.size / 250_000)), 1)
    sampled = array[::stride, ::stride]
    values = np.unique(sampled)
    nonnegative = bool(values.size == 0 or values[0] >= 0)
    # A normal 8-bit microscopy image often contains all 256 grey values,
    # whereas a label image contains roughly one value per object. Permit
    # large fields with many objects without calling ordinary 8-bit data a
    # mask merely because its value range is bounded.
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
    return f"{mapping.plate}_{mapping.well}_{int(mapping.field)}"


def _pair_masks(image_plan: cv.ConversionPlan,
                groups: Sequence[InputGroup],
                *,
                layout: str = "auto",
                ) -> Tuple[Dict[str, Dict[str, MaskMatch]], List[str], List[str]]:
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
            errors.append(
                f"{group.key}: choose whether these masks are "
                f"{', '.join(OBJECT_TYPES)}.")
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
    for key in (
        "src", "cell_mask_dim", "nucleus_mask_dim",
        "pathogen_mask_dim", "organelle_mask_dim",
    ):
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
        # Empty means all imported intensity channels. This avoids Measure's
        # four-channel default being invalid for a one- or two-channel import.
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}.npy"
    np.save(temporary, array)
    os.replace(temporary, path)
    return path


def _tables(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "ORDER BY name").fetchall()
    return [str(row[0]) for row in rows]


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
            array = np.asarray(_read_mask(match.path))
            if array.shape != shape:
                raise ConfigurationError(
                    f"{match.path}: {object_type} mask shape {array.shape} "
                    f"does not match intensity shape {shape} for {stem}.")
            if np.any(array < 0):
                raise ConfigurationError(
                    f"{match.path}: label masks cannot contain negative IDs.")
            maximum = int(np.max(array, initial=0))
            if maximum > np.iinfo(np.uint16).max:
                raise ConfigurationError(
                    f"{match.path}: label ID {maximum} exceeds the maximum "
                    "65535 supported by the Measure array contract.")
            integer = array.astype(np.uint16, copy=False)
            mask_planes.append(integer)
            _save_npy(os.path.join(
                dst, "masks", f"{object_type}_mask_stack", f"{stem}.npy"),
                integer)

        for plane in image_planes:
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
        stack = np.stack([plane.astype(np.uint16, copy=False)
                          for plane in image_planes], axis=-1)
        _save_npy(os.path.join(dst, "stack", f"{stem}.npy"), stack)
        merged = np.stack(
            [plane.astype(np.uint16, copy=False)
             for plane in [*image_planes, *mask_planes]], axis=-1)
        merged_paths.append(_save_npy(
            os.path.join(dst, "merged", f"{stem}.npy"), merged))

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
    if "organelle" in plan.object_types:
        summaries = list(
            measure_settings.get("summarize_organelles_by") or [])
        if "organelle" not in summaries:
            summaries.append("organelle")
        measure_settings["summarize_organelles_by"] = summaries

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


__all__ = [
    "InputGroup", "MaskMatch", "ExternalMaskPlan", "ExternalMaskResult",
    "SUPPORTED_SUFFIXES", "OBJECT_TYPES", "ROLES",
    "detect_inputs", "default_settings", "plan_external_masks",
    "run_external_masks", "prepare_external_masks",
]
