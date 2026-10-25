"""Pure-Python mask editing and persistence for the Qt Make Masks screen.

This module provides image and mask I/O plus non-brush label operations,
including fill, relabel, size and intensity filtering, Otsu detection, and
magic-wand selection. It has no Qt dependency, so the editing operations can
be tested without a display.

THREE INVERSIONS LIVE HERE AND THEY DO DIFFERENT THINGS. :func:`invert_intensity` is the photographic complement of an
IMAGE -- ``dtype_max - value``, what a viewer's Invert does, exactly
reversible on every integer dtype -- and it is what the Make Masks screen's
"Invert image" draws with. :func:`invert_for_detection` reflects an image
about its OWN range instead, and is what the screen's "Invert for detection"
hands the detectors; its docstring has the measurement that says why a
detector cannot use the complement, and it is NOT a duplicate to be merged
away. :func:`invert_mask` flips a LABEL image's foreground and background;
on an ordinary field that gives one object covering the frame, which is why
it was reported as doing nothing.

:func:`save_mask` passes labels through :func:`canonical_labels`, which
preserves existing nonzero object identifiers rather than renumbering
connected components. This maintains correspondence with measurements,
tracks, and crops keyed by those identifiers.

Saving also writes the artifact's :class:`spacr.curation.CurationLog`
sidecar, consistent with :mod:`spacr.napari_bridge` and
:mod:`spacr.qt.curation_tool`. The sidecar allows
:func:`spacr.curation.is_curated` to distinguish manually edited masks from
pipeline-generated masks.

Where a field's mask lives
--------------------------

The editor opens all three layouts :mod:`spacr.curation_queue` reads, and
edits each one in place rather than converting it:

``nested``
    ``<folder>/<image>`` with the mask at ``<folder>/masks/<stem>.tif``. The
    default, and what every function below does when told nothing else.
``sibling``
    ``<root>/images/<image>`` with the mask at ``<root>/masks/<stem>.tif``.
    The editor opens ``<root>/images`` and passes ``masks_dir=<root>/masks``;
    without it the mask would be read from and written to
    ``<root>/images/masks``, a folder the set does not have.
``seg``
    ``<folder>/<stem>_seg.npy``, a Cellpose bundle holding the image and the
    labels in one pickled dict. The editor's file list names the bundles
    themselves, and a name ending in :data:`SEG_SUFFIX` is read and written
    as a bundle by :func:`load_image_and_mask`, :func:`save_mask`,
    :func:`write_recrop` and :func:`retire_recropped_original`.

In place is sound for both, which is why there is no convert step. A sibling
set differs from a nested one only in where its masks folder is, so passing
that folder is the whole change. A bundle is rewritten with every key it
already had kept, ``masks`` replaced, and the two keys derived from the
masks -- ``outlines`` and ``ismanual`` -- brought up to date with it (see
:func:`save_seg_bundle`); that is what the external curation tool did to the
same files, less its stale outlines. A convert step would have been a second
copy of every field, and the curator would have had to remember which copy
was the truth.
"""
from __future__ import annotations

import csv
import contextlib
import hashlib
import json
import math
import os
import stat
import tempfile
from collections import deque
from numbers import Integral
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from ..curation import LOG_SUFFIX, CurationLog
from ..curation_queue import SEG_SUFFIX
from ..tiff_io import write_tiff

MAX_MASK_LABEL = 2**32 - 1
LINEAGE_SCHEMA = "spacr.lineage/v1"


def validate_lineage(state: Optional[dict] = None) -> dict:
    """Copy and validate a versioned lineage graph without recursive traversal."""
    from uuid import UUID, NAMESPACE_URL, uuid5
    from .i18n import tr

    if state is None:
        return {"schema": LINEAGE_SCHEMA, "cells": {}, "frames": {},
                "frame_revisions": {}, "deleted_cells": {}, "frame_ids": {}}
    try:
        result = json.loads(json.dumps(state, allow_nan=False))
        if result["schema"] != LINEAGE_SCHEMA:
            raise ValueError("schema")
        cells, frames = result["cells"], result["frames"]
        if not isinstance(cells, dict) or not isinstance(frames, dict):
            raise ValueError("graph")
        frame_ids = result.setdefault("frame_ids", {})
        if not isinstance(frame_ids, dict):
            raise ValueError("frame identities")
        paths = set(frames)
        for cell in cells.values():
            paths.update(cell.get(field) for field in ("born_frame", "division_frame")
                         if cell.get(field) is not None)
        for path in paths:
            if not isinstance(path, str) or not path:
                raise ValueError("frame path")
            frame_ids.setdefault(path, str(uuid5(NAMESPACE_URL, "spacr-frame:" + path)))
        for path, identity in frame_ids.items():
            if not isinstance(path, str) or not path or str(UUID(identity)) != identity:
                raise ValueError("frame identity")
        revisions = result.setdefault("frame_revisions", {})
        deleted = result.setdefault("deleted_cells", {})
        if not isinstance(revisions, dict):
            raise ValueError("revisions")
        if not isinstance(deleted, dict):
            raise ValueError("deleted cells")
        for identity, revision in deleted.items():
            if str(UUID(identity)) != identity or not isinstance(revision, int) \
                    or isinstance(revision, bool) or revision < 0:
                raise ValueError("deleted cell revision")
        for key, cell in cells.items():
            if str(UUID(key)) != key or not isinstance(cell, dict):
                raise ValueError("UUID")
            if not isinstance(cell.get("name"), str) or not cell["name"].strip():
                raise ValueError("name")
            parent = cell.get("parent_uuid")
            if parent is not None and (parent not in cells or parent == key):
                raise ValueError("parent")
            if parent is not None and key not in cells[parent].get("children", []):
                raise ValueError("parent division")
            children = cell.get("children", [])
            if not isinstance(children, list) or (children and len(children) != 2) \
                    or len(set(children)) != len(children):
                raise ValueError("children")
            for child in children:
                if child not in cells or cells[child].get("parent_uuid") != key:
                    raise ValueError("daughter")
            if bool(children) != bool(cell.get("division_frame")):
                raise ValueError("division")
            for field in ("born_frame", "division_frame"):
                expected = frame_ids.get(cell.get(field))
                cell.setdefault(field + "_id", expected)
                if cell[field + "_id"] != expected:
                    raise ValueError("ancestry frame identity")
            label = cell.get("label", 1)
            if isinstance(label, bool) or not isinstance(label, int) \
                    or not 1 <= label <= MAX_MASK_LABEL:
                raise ValueError("label")
            revision = cell.get("revision", 0)
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise ValueError("revision")
            if key in deleted and deleted[key] >= revision:
                raise ValueError("deleted cell")
        checked = set()
        for key in cells:
            pending = set()
            cursor = key
            while cursor is not None and cursor not in checked:
                if cursor in pending:
                    raise ValueError("cycle")
                pending.add(cursor)
                cursor = cells[cursor].get("parent_uuid")
            checked.update(pending)
        for frame, mapping in frames.items():
            if not isinstance(frame, str) or not frame or not isinstance(mapping, dict):
                raise ValueError("frame mapping")
            if len(set(mapping.values())) != len(mapping):
                raise ValueError("duplicate identity")
            for label, identity in mapping.items():
                if str(int(label)) != label or not 1 <= int(label) <= MAX_MASK_LABEL \
                        or identity not in cells:
                    raise ValueError("frame label")
            revision = revisions.get(frame, 0)
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise ValueError("frame revision")
        return result
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
        raise ValueError(tr("Invalid or unsupported embedded lineage metadata: {reason}",
                            reason=str(error))) from error


def _next_lineage_revision(state: dict) -> int:
    """Advance beyond persisted edits even if the current machine clock is earlier."""
    from time import time_ns

    revisions = [cell.get("revision", 0) for cell in state["cells"].values()]
    revisions.extend(state.get("deleted_cells", {}).values())
    revisions.extend(state.get("frame_revisions", {}).values())
    return max(time_ns(), max(revisions, default=0) + 1)


def assign_lineage_label(state: dict, frame: str, label: int,
                         cell_uuid: Optional[str] = None,
                         old_label: Optional[int] = None) -> dict:
    """Map an observed frame label to a persistent segment, or remove a mapping.

    :param state: The lineage graph to copy and update.
    :param frame: The source image path for this frame.
    :param label: The local uint32 pixel ID, or zero to remove ``old_label``.
    """
    from .i18n import tr

    result = validate_lineage(state)
    stamp = _next_lineage_revision(result)
    if isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral) \
            or not 0 <= int(label) <= MAX_MASK_LABEL:
        raise ValueError(tr("Labels must be whole numbers from 0 through 4294967295."))
    frame = os.path.abspath(os.fspath(frame))
    mapping = result["frames"].setdefault(frame, {})
    if label:
        if cell_uuid not in result["cells"]:
            raise ValueError(tr("Choose an existing lineage cell before assigning its label."))
        occupied = mapping.get(str(int(label)))
        if occupied is not None and occupied != cell_uuid:
            raise ValueError(tr("This frame label already belongs to another lineage cell."))
    if old_label is not None:
        mapping.pop(str(int(old_label)), None)
    if label:
        mapping = {key: value for key, value in mapping.items() if value != cell_uuid}
        mapping[str(int(label))] = cell_uuid
        result["frames"][frame] = mapping
        result["cells"][cell_uuid]["label"] = int(label)
        result["cells"][cell_uuid]["revision"] = stamp
    result["frame_revisions"][frame] = stamp
    return validate_lineage(result)


def new_lineage_cell(state: Optional[dict], name: str, label: int,
                     frame: str) -> Tuple[dict, str]:
    """Create an unplaced founder without claiming any existing mask pixels.

    :param state: The lineage graph to extend, or ``None`` for an empty graph.
    :param name: The nonempty display name for the new founder.
    :param label: The preferred nonzero uint32 pixel ID, without assigning pixels.
    :param frame: The source image path where the founder is introduced.
    """
    from uuid import uuid4
    from .i18n import tr

    result = validate_lineage(state)
    if not isinstance(name, str) or not name.strip() \
            or isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral) \
            or not 1 <= int(label) <= MAX_MASK_LABEL:
        raise ValueError(tr("A lineage cell needs a name and a nonzero uint32 label."))
    stamp = _next_lineage_revision(result)
    identity = str(uuid4())
    result["cells"][identity] = {
        "name": name.strip(), "parent_uuid": None, "born_frame": os.path.abspath(frame),
        "division_frame": None, "children": [], "label": int(label), "revision": stamp}
    return validate_lineage(result), identity


def assign_tracked_object(mask: np.ndarray, x: int, y: int, state: dict,
                          frame: str, cell_uuid: str,
                          preferred_label: Optional[int] = None):
    """Give the hovered object the selected identity, letting one division branch continue it.

    :param mask: The current two-dimensional integer label mask.
    :param x: The hovered pixel column.
    :param y: The hovered pixel row.
    :param state: The lineage graph to copy and update.
    :param frame: The source image path for this frame.
    :param cell_uuid: The persistent UUID of the segment to assign.
    """
    from .i18n import tr

    result = validate_lineage(state)
    if cell_uuid not in result["cells"]:
        raise ValueError(tr("Choose an existing lineage cell before assigning its label."))
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           for value in (x, y)):
        raise ValueError(tr("Object coordinates must be integer pixels."))
    labels = canonical_labels(mask, preserve_ids=True)
    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
        return labels, result, 0, 0
    original = int(labels[y, x])
    if not original:
        return labels, result, 0, 0
    frame = os.path.abspath(frame)
    cells = result["cells"]
    selected = cells[cell_uuid]
    mapping = result["frames"].get(frame, {})
    assigned = mapping.get(str(original))
    identity, retire = cell_uuid, None
    if assigned is not None and assigned != cell_uuid:
        other = cells[assigned]
        if other.get("parent_uuid") == cell_uuid or selected.get("parent_uuid") == assigned:
            if other.get("parent_uuid") == cell_uuid:
                mother, branch, claimed = selected, assigned, selected["name"]
            else:
                mother, branch = other, cell_uuid
                claimed = mother["name"] if mother["name"] != selected["name"] else None
            if any(child != branch and cells[child]["name"] == claimed
                   for child in mother["children"]):
                raise ValueError(tr(
                    "{name} already continues through its other daughter; undo that "
                    "first or select the other daughter.", name=mother["name"]))
            identity = assigned
        else:
            retire = assigned
    if any(value == identity and int(local) != original and np.any(labels == int(local))
           for local, value in mapping.items()):
        raise ValueError(tr(
            "{name} already labels another object in this frame; one identity can "
            "mark only one object per frame.", name=selected["name"]))
    target = selected["label"] if preferred_label is None else preferred_label
    if isinstance(target, (bool, np.bool_)) or not isinstance(target, Integral) \
            or not 1 <= int(target) <= MAX_MASK_LABEL:
        raise ValueError(tr("A lineage cell needs a nonzero uint32 label."))
    target = int(target)
    if target != original and (np.any(labels == target) or str(target) in mapping):
        target = original
    if target == original and assigned == cell_uuid:
        return labels, result, original, original
    if retire is not None:
        result["frames"][frame] = {
            key: value for key, value in mapping.items() if key != str(original)}
    if identity != cell_uuid:
        stamp = _next_lineage_revision(result)
        cells[identity].update(name=selected["name"], label=int(target), revision=stamp)
    labels, _ = assign_object_label(labels, x, y, target)
    result = assign_lineage_label(result, frame, target, identity, old_label=original)
    if retire is not None:
        cell = result["cells"][retire]
        if not cell.get("parent_uuid") and not cell.get("children") and not any(
                retire in frame_mapping.values() for frame_mapping in result["frames"].values()):
            result["deleted_cells"][retire] = _next_lineage_revision(result)
            del result["cells"][retire]
            result = validate_lineage(result)
    return labels, result, original, target


def _merge_continuation(state: dict, mother: str, daughter: str) -> dict:
    """Make a founder that continues a childless cell part of it, division included.

    Every frame of ``daughter`` maps to ``mother``, its daughters become
    daughters of ``mother`` and ``daughter`` is retired. A pair that shares a
    frame, a daughter with a parent or a mother that already divided is
    refused, since one cell cannot be both.
    """
    from .i18n import tr

    result = validate_lineage(state)
    cells = result["cells"]
    keep, gone = cells[mother], cells[daughter]
    if mother == daughter or gone.get("parent_uuid") is not None or keep.get("children") \
            or any(mother in mapping.values() and daughter in mapping.values()
                   for mapping in result["frames"].values()):
        raise ValueError(tr(
            "{old} cannot become part of {name}: one cell cannot appear twice in a "
            "frame or divide twice.", old=gone["name"], name=keep["name"]))
    stamp = _next_lineage_revision(result)
    for frame, mapping in result["frames"].items():
        if daughter in mapping.values():
            result["frames"][frame] = {
                local: mother if identity == daughter else identity
                for local, identity in mapping.items()}
            result["frame_revisions"][frame] = stamp
    keep.update(children=list(gone["children"]), division_frame=gone.get("division_frame"),
                revision=stamp)
    keep.pop("division_frame_id", None)
    for child in gone["children"]:
        cells[child].update(parent_uuid=mother, revision=stamp)
    del cells[daughter]
    result["deleted_cells"][daughter] = stamp
    return validate_lineage(result)


def rename_lineage_cell(state: dict, cell_uuid: str, name: str) -> dict:
    """Rename a segment while retaining its identity and parent links.

    :param state: The lineage graph to copy and update.
    :param cell_uuid: The persistent UUID of the segment to rename.
    :param name: The new nonempty display name.
    """
    from .i18n import tr

    result = validate_lineage(state)
    if cell_uuid not in result["cells"] or not isinstance(name, str) or not name.strip():
        raise ValueError(tr("Choose a lineage cell and enter a nonempty name."))
    result["cells"][cell_uuid]["name"] = name.strip()
    result["cells"][cell_uuid]["revision"] = _next_lineage_revision(result)
    return result


def divide_lineage_cell(state: dict, parent_uuid: str, frame: str,
                        daughter_names, labels) -> Tuple[dict, Tuple[str, str]]:
    """End a segment and create two distinct daughter segments at one frame.

    :param state: The lineage graph to copy and update.
    :param parent_uuid: The persistent UUID of the undivided parent segment.
    :param frame: The source image path where the division is recorded.
    :param daughter_names: Two nonempty display names in daughter order.
    :param labels: Two distinct nonzero uint32 preferred pixel IDs in daughter order.
    """
    from uuid import uuid4
    from .i18n import tr

    result = validate_lineage(state)
    if parent_uuid not in result["cells"] or result["cells"][parent_uuid].get("children"):
        raise ValueError(tr("Choose a lineage cell that has not already divided."))
    if len(daughter_names) != 2 or len(labels) != 2 or labels[0] == labels[1]:
        raise ValueError(tr("A division needs two daughter names and distinct nonzero labels."))
    if any(not isinstance(name, str) or not name.strip() for name in daughter_names) \
            or any(isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral)
                   or not 1 <= int(label) <= MAX_MASK_LABEL for label in labels):
        raise ValueError(tr("A division needs two daughter names and distinct nonzero labels."))
    frame = os.path.abspath(frame)
    parent = result["cells"][parent_uuid]
    children = (str(uuid4()), str(uuid4()))
    stamp = _next_lineage_revision(result)
    parent.update(division_frame=frame, children=list(children), revision=stamp)
    parent.pop("division_frame_id", None)
    result["frames"][frame] = {
        key: value for key, value in result["frames"].get(frame, {}).items()
        if value != parent_uuid}
    for child, name, label in zip(children, daughter_names, labels):
        result["cells"][child] = {
            "name": name.strip(), "parent_uuid": parent_uuid, "born_frame": frame,
            "division_frame": None, "children": [], "label": int(label), "revision": stamp}
    result["frame_revisions"][frame] = stamp
    return validate_lineage(result), children


def merge_lineage_states(active: Optional[dict], incoming: Optional[dict],
                          frame: Optional[str] = None) -> dict:
    """Merge embedded snapshots, preserving newer edits and immutable ancestry.

    :param active: The active lineage graph, or ``None`` for an empty graph.
    :param incoming: The incoming snapshot, or ``None`` for an empty graph.
    """
    from .i18n import tr

    result, source = validate_lineage(active), validate_lineage(incoming)
    current = os.path.abspath(frame) if frame is not None else None
    known = {identity: path for path, identity in result["frame_ids"].items()}
    active_paths, source_paths = {}, {}
    for path, identity in source["frame_ids"].items():
        if identity in known:
            if path == current:
                active_paths[known[identity]] = path
            else:
                source_paths[path] = known[identity]
        elif path in result["frame_ids"] and result["frame_ids"][path] != identity:
            raise ValueError(tr("Embedded lineage snapshots disagree about a frame's identity."))
    result = _remap_lineage_frames(result, active_paths)
    source = _remap_lineage_frames(source, source_paths)
    result["frame_ids"].update(source["frame_ids"])
    for identity, revision in source["deleted_cells"].items():
        result["deleted_cells"][identity] = max(
            revision, result["deleted_cells"].get(identity, 0))
    for identity, cell in source["cells"].items():
        old = result["cells"].get(identity)
        if old is None:
            result["cells"][identity] = cell
            continue
        if old.get("born_frame_id") != cell.get("born_frame_id") or (
                old.get("revision", 0) == cell.get("revision", 0) and any(
                    old.get(key) != cell.get(key) for key in (
                        "parent_uuid", "children", "division_frame_id"))):
            raise ValueError(tr("Embedded lineage snapshots disagree about a cell's ancestry."))
        newer = cell if cell.get("revision", 0) > old.get("revision", 0) else old
        result["cells"][identity] = dict(newer)
    for identity, revision in result["deleted_cells"].items():
        if identity in result["cells"] and revision >= result["cells"][identity].get("revision", 0):
            del result["cells"][identity]
    for name, mapping in source["frames"].items():
        if name not in result["frames"] or ((current is None or name == current)
                and source["frame_revisions"].get(name, 0)
                > result["frame_revisions"].get(name, 0)):
            result["frames"][name] = mapping
            result["frame_revisions"][name] = source["frame_revisions"].get(name, 0)
    for mapping in result["frames"].values():
        for label, identity in list(mapping.items()):
            if identity not in result["cells"] and identity in result["deleted_cells"]:
                del mapping[label]
    return validate_lineage(result)


def restore_lineage_state(restored: Optional[dict], current: Optional[dict]) -> dict:
    """Make undo or redo an explicit newer edit that survives older TIFF snapshots.

    :param restored: The undo or redo snapshot to restore, or ``None`` for an empty graph.
    :param current: The current graph, or ``None``, used to preserve revision ordering.
    """
    result, previous = validate_lineage(restored), validate_lineage(current)
    stamp = max(_next_lineage_revision(result), _next_lineage_revision(previous))
    result["frame_ids"].update(previous["frame_ids"])
    for identity, revision in previous["deleted_cells"].items():
        result["deleted_cells"][identity] = max(
            revision, result["deleted_cells"].get(identity, 0))
    for identity in previous["cells"].keys() - result["cells"].keys():
        result["deleted_cells"][identity] = stamp
    for identity, cell in result["cells"].items():
        if cell != previous["cells"].get(identity):
            cell["revision"] = stamp
    for frame in result["frames"].keys() | previous["frames"].keys():
        if result["frames"].get(frame) != previous["frames"].get(frame):
            result["frames"].setdefault(frame, {})
            result["frame_revisions"][frame] = stamp
    return validate_lineage(result)


def prune_lineage_frame(state: dict, frame: str, mask: np.ndarray) -> dict:
    """Remove absent pixel mappings as a revisioned edit, retaining cell ancestry.

    :param state: The lineage graph to copy and update.
    :param frame: The source image path for this frame.
    :param mask: The current label mask used to identify surviving pixel IDs.
    """
    result = validate_lineage(state)
    frame = os.path.abspath(frame)
    present = set(map(str, np.unique(np.asarray(mask)[np.asarray(mask) > 0])))
    previous = result["frames"].get(frame)
    mapping = {key: value for key, value in (previous or {}).items() if key in present}
    result["frames"][frame] = mapping
    if mapping != previous:
        result["frame_revisions"][frame] = _next_lineage_revision(result)
    return validate_lineage(result)


def _remap_lineage_frames(state: dict, paths: dict) -> dict:
    """Change descriptive frame paths while retaining embedded frame identities."""
    from .i18n import tr

    result = validate_lineage(state)
    frames, revisions, identities = {}, {}, {}
    for path, identity in result["frame_ids"].items():
        target = paths.get(path, path)
        if target in identities and identities[target] != identity:
            raise ValueError(tr("Moved lineage frames would share an incompatible frame identity."))
        identities[target] = identity
    for frame, mapping in result["frames"].items():
        target = paths.get(frame, frame)
        if target in frames and frames[target] != mapping:
            raise ValueError(tr("Moved lineage frames would share an incompatible frame identity."))
        frames[target] = mapping
        revisions[target] = result["frame_revisions"].get(frame, 0)
    result.update(frames=frames, frame_revisions=revisions, frame_ids=identities)
    for cell in result["cells"].values():
        for field in ("born_frame", "division_frame"):
            cell[field] = paths.get(cell.get(field), cell.get(field))
    return validate_lineage(result)


def _relocate_lineage_frame(state: dict, original: str, current: str) -> dict:
    """Rebind moved Windows or POSIX frame aliases without reading their old paths."""
    import ntpath
    import posixpath

    result = validate_lineage(state)
    original_path = original.replace("\\", "/")
    old_folder = posixpath.dirname(original_path)
    windows = bool(ntpath.splitdrive(original)[0])
    compare = str.casefold if windows else str
    paths = {}
    for path in result["frame_ids"]:
        portable = path.replace("\\", "/")
        if compare(portable) == compare(original_path):
            paths[path] = current
        elif compare(posixpath.dirname(portable)) == compare(old_folder):
            paths[path] = os.path.join(os.path.dirname(current), posixpath.basename(portable))
    return _remap_lineage_frames(result, paths)


def load_mask_metadata(folder: str, filename: str,
                        masks_dir: Optional[str] = None) -> dict:
    """Read identity semantics and lineage from the mask itself, without a sidecar.

    :param folder: The source image folder used to locate the matching mask.
    :param filename: The source image filename whose mask metadata is read.
    """
    import tifffile
    from .i18n import tr

    path = mask_save_path(folder, filename, masks_dir)
    if not os.path.isfile(path) and not is_seg_bundle(filename):
        stem = os.path.splitext(filename)[0]
        candidates = [os.path.join(masks_folder(folder, masks_dir), filename),
                      os.path.join(masks_folder(folder, masks_dir), stem + ".tiff")]
        path = next((candidate for candidate in candidates if os.path.isfile(candidate)), path)
    if not os.path.isfile(path):
        return {}
    if is_seg_bundle(filename):
        metadata = read_seg_bundle(path).get("spacr_mask", {})
    else:
        with open(path, "rb") as source:
            byte_order = source.read(2)
        if not path.lower().endswith((".tif", ".tiff")) and byte_order not in (
                b"II", b"MM", b"EP"):
            return {}
        with tifffile.TiffFile(path) as handle:
            description = handle.pages[0].description or ""
        try:
            metadata = json.loads(description).get("spacr_mask", {})
        except (ValueError, AttributeError) as error:
            if "spacr_mask" in description:
                raise ValueError(tr("The embedded mask metadata cannot be read.")) from error
            return {}
    if not isinstance(metadata, dict) or (metadata and metadata.get("schema") != "spacr.mask/v1"):
        raise ValueError(tr("Invalid or unsupported embedded mask metadata."))
    if metadata.get("lineage") is not None:
        metadata["lineage"] = validate_lineage(metadata["lineage"])
        original = metadata.get("frame")
        current = os.path.abspath(os.path.join(folder, filename))
        if metadata.get("frame_id") is not None and metadata["frame_id"] \
                != metadata["lineage"]["frame_ids"].get(original):
            raise ValueError(tr("Invalid or unsupported embedded mask metadata."))
        if isinstance(original, str) and original != current:
            metadata["lineage"] = _relocate_lineage_frame(metadata["lineage"], original, current)
        metadata["frame"] = current
        metadata["frame_id"] = metadata["lineage"]["frame_ids"].get(current)
    return metadata


def load_mask_lineage(folder: str, filename: str,
                       masks_dir: Optional[str] = None) -> dict:
    """Read an independent lineage graph from a TIFF mask or return an empty graph.

    :param folder: The source image folder used to locate the matching mask.
    :param filename: The source image filename whose mask lineage is read.
    """
    return validate_lineage(load_mask_metadata(folder, filename, masks_dir).get("lineage"))


def _label_boxes(labels: np.ndarray) -> dict:
    """Index observed labels without allocating by the largest numeric identity."""
    if not labels.size:
        return {}
    if int(labels.max()) <= 65535:
        return {index + 1: box for index, box in enumerate(_ndimage().find_objects(labels))
                if box is not None}
    values, inverse = np.unique(labels, return_inverse=True)
    dense = inverse.reshape(labels.shape) + 1
    boxes = _ndimage().find_objects(dense)
    return {int(value): box for value, box in zip(values, boxes) if value and box is not None}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".scn",
              ".dm3", ".dm4", ".mrc", ".rec")
YOLO_ANNOTATIONS_NAME = ".spacr_yolo_annotations.json"
YOLO_CLASSES_NAME = ".classes.json"


_PIXEL_SIZES: Dict[str, Optional[float]] = {}
"""Pixel sizes in nm of the TEM files :func:`read_image` has read, by path,
so saving a mask does not decode the image a second time to find it."""

#: Electron-microscopy formats, read by :func:`spacr.io.read_tem_image`.
TEM_SUFFIXES = (".dm3", ".dm4", ".mrc", ".rec")


def read_image(path: str) -> np.ndarray:
    """Read one image file the way Make Masks shows it.

    A Bio-Rad Image Lab ``.scn`` is read by :func:`spacr.convert.read_scn`
    (``uint16``, inverted so it looks like Image Lab's own exports); every
    other format goes through :func:`imageio.v2.imread`.

    :param path: the image file.
    :returns: the pixels.
    """
    lower = str(path).lower()
    if lower.endswith(".scn"):
        from ..convert import read_scn

        return read_scn(path)[0]
    if lower.endswith(TEM_SUFFIXES):
        from ..io import read_tem_image

        image, meta = read_tem_image(path)
        _PIXEL_SIZES[os.path.abspath(os.fspath(path))] = meta["pixel_size_nm"]
        return image
    return imageio.imread(path)


def pixel_size_nm(path) -> Optional[float]:
    """The physical pixel size of an image file, in nanometres, or None.

    TEM files (``.dm3``, ``.dm4``, ``.mrc``, ``.rec``) are read by
    :func:`spacr.io.read_tem_image` inside :func:`read_image`, which keeps
    their pixel size here so a save does not read the image again. An MRC
    file needs the optional ``mrcfile`` package; without it
    :func:`read_image` raises :class:`spacr.io.MissingTEMReader`.

    :param path: the source image. A TEM file read by :func:`read_image`
        answers from memory; a TIFF from its resolution tags; DM and MRC
        files not yet read are read for their calibration.
    """
    key = os.path.abspath(os.fspath(path))
    if key in _PIXEL_SIZES:
        return _PIXEL_SIZES[key]
    from ..io import image_pixel_size_nm

    size = image_pixel_size_nm(path)
    if str(path).lower().endswith(TEM_SUFFIXES):
        _PIXEL_SIZES[key] = size
    return size


#: Display settings Make Masks' "Contrast preset" box applies in one step.
#:
#: ``tem`` is for transmission electron micrographs: one greyscale channel
#: whose few saturated pixels (holes in the section, stain precipitate) would
#: take the whole range, so the window is the 0.5th to 99.5th percentile, and
#: CLAHE on 128 px tiles evens out the uneven section thickness. Membranes
#: stay dark. ``tem_inverted`` is the same with Invert on, so dark membranes
#: and organelles are drawn bright and Otsu and the magnifier find them.
#: Measured on the ten fields of ``spacr-example-tem``, which EMPIAR ships
#: already stretched to 8 bits: Upper % at 99.5 rather than the default 99.9
#: sets white 2 to 16 grey levels lower, so the brightest cytoplasm is no
#: longer drawn as a flat grey below a few saturated pixels.
CONTRAST_PRESETS: Dict[str, Dict[str, object]] = {
    "tem": {"lower": 0.5, "upper": 99.5, "invert": False, "clahe": True,
            "clahe_tile": 128, "clahe_clip": 0.01},
    "tem_inverted": {"lower": 0.5, "upper": 99.5, "invert": True,
                     "clahe": True, "clahe_tile": 128, "clahe_clip": 0.01},
}


def _resolution_for(pixel_size: Optional[float]) -> dict:
    """``tifffile`` keywords recording ``pixel_size`` nm per pixel, or none.

    Two records. The resolution tags in pixels per centimetre, the one TIFF
    resolution unit that is a length, which other software reads; their
    32-bit rational is chosen with the largest denominator that fits, so
    0.85 nm comes back as 0.85 to nine digits rather than 0.8498. And the
    exact value as ``pixel_size_nm`` in tifffile's JSON description, which
    :func:`spacr.io.tiff_pixel_size_nm` reads first.
    """
    if not pixel_size or pixel_size <= 0:
        return {}
    per_cm = 1e7 / float(pixel_size)
    top = 2 ** 32 - 1
    denominator = max(1, min(top, int(top // per_cm))) if per_cm >= 1 else top
    numerator = min(top, max(1, int(round(per_cm * denominator))))
    rational = (numerator, denominator)
    return {"resolution": (rational, rational), "resolutionunit": "CENTIMETER",
            "metadata": {"pixel_size_nm": float(pixel_size)}}


def _yolo_shape(image_shape) -> tuple:
    """Validate an image shape and return its integer dimensions."""
    try:
        shape = tuple(image_shape)
    except TypeError as exc:
        raise ValueError("image shape must have two or three dimensions") from exc
    if len(shape) not in (2, 3) or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            or value <= 0 for value in shape):
        raise ValueError("image shape must have positive integer dimensions")
    return tuple(int(value) for value in shape)


def _yolo_classes(classes) -> list[str]:
    """Return a nonempty, unambiguous ordered class-name map."""
    if isinstance(classes, (str, bytes)):
        raise ValueError("YOLO classes must be a sequence of names")
    try:
        names = list(classes)
    except TypeError as exc:
        raise ValueError("YOLO classes must be a sequence of names") from exc
    if not names or any(
            not isinstance(name, str) or not name or name != name.strip()
            or name in (".", "..") or "/" in name or "\\" in name
            or not name.isprintable() for name in names):
        raise ValueError("YOLO class names must be nonempty printable names")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("YOLO class names must be unique")
    return names


def _yolo_boxes(boxes, width: int, height: int, n_classes=None) -> list[tuple]:
    """Sort and clip full-image, exclusive-edge boxes without merging overlap."""
    try:
        given = list(boxes)
    except TypeError as exc:
        raise ValueError("YOLO boxes must be a sequence") from exc
    canonical = []
    for box in given:
        try:
            class_id, x0, y0, x1, y1 = box
        except (TypeError, ValueError) as exc:
            raise ValueError("each YOLO box needs class ID and four corners") from exc
        if (isinstance(class_id, bool) or not isinstance(class_id, Integral)
                or class_id < 0 or (n_classes is not None
                                 and class_id >= n_classes)):
            raise ValueError("YOLO box class ID is outside the class map")
        try:
            coordinates = (float(x0), float(y0), float(x1), float(y1))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("YOLO box coordinates must be finite numbers") from exc
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("YOLO box coordinates must be finite numbers")
        left, right = sorted((max(0.0, min(float(width), coordinates[0])),
                              max(0.0, min(float(width), coordinates[2]))))
        top, bottom = sorted((max(0.0, min(float(height), coordinates[1])),
                              max(0.0, min(float(height), coordinates[3]))))
        if left >= right or top >= bottom:
            raise ValueError("YOLO box has no area inside the image")
        canonical.append((int(class_id), left, top, right, bottom))
    return canonical


def yolo_box_lines(boxes, width: int, height: int) -> list[str]:
    """Encode full-image pixel boxes as standard YOLO ``class cx cy w h``.

    Corners may be reversed or outside the image. Every box is clipped and
    checked before any line is returned; overlapping boxes stay distinct.
    Coordinates use exclusive right and bottom edges and are normalised by
    the positive image width and height.

    :param boxes: ``(class_id, x0, y0, x1, y1)`` full-image pixel boxes.
    :param width: positive integer width of the original image.
    :param height: positive integer height of the original image.
    :returns: one normalised YOLO label line per input box, in input order.
    """
    shape = _yolo_shape((height, width))
    lines = []
    for class_id, x0, y0, x1, y1 in _yolo_boxes(boxes, shape[1], shape[0]):
        lines.append(
            f"{class_id} {((x0 + x1) / 2 / width):.17g} "
            f"{((y0 + y1) / 2 / height):.17g} "
            f"{((x1 - x0) / width):.17g} {((y1 - y0) / height):.17g}"
        )
    return lines


def _yolo_relative(folder, filename) -> tuple[str, Path]:
    """Resolve one relative source file without leaving its project folder."""
    raw = str(filename)
    relative = Path(raw)
    if (not raw or "\\" in raw or relative.is_absolute()
            or any(part in (".", "..") for part in relative.parts)):
        raise ValueError("YOLO source filename must stay inside its folder")
    base = Path(folder).resolve(strict=True)
    if not base.is_dir():
        raise ValueError("YOLO source folder is not a directory")
    source = (base / relative).resolve(strict=True)
    try:
        source.relative_to(base)
    except ValueError as exc:
        raise ValueError("YOLO source filename escapes its folder") from exc
    if not source.is_file():
        raise ValueError("YOLO source must be a file")
    return relative.as_posix(), source


def _yolo_source_hash(source: Path) -> str:
    """Hash the original source bytes without decoding or changing them."""
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yolo_unique_object(pairs) -> dict:
    """Refuse duplicate JSON keys rather than silently taking the last one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate YOLO annotation key: {key}")
        result[key] = value
    return result


def _yolo_read_json(path: Path):
    """Read strict JSON from one project or export metadata file."""
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"YOLO metadata is not a regular file: {path}")
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError(f"YOLO metadata is not a regular file: {path}")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_yolo_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid YOLO metadata: {path}") from exc


def _yolo_state(path: Path) -> dict:
    """Validate the complete project ledger before changing or trusting it."""
    if path.is_symlink():
        raise ValueError("YOLO project annotation ledger cannot be a symbolic link")
    if not path.exists():
        return {"version": 1, "classes": ["object"], "images": {}}
    state = _yolo_read_json(path)
    if (not isinstance(state, dict) or set(state) != {"version", "classes", "images"}
            or type(state["version"]) is not int or state["version"] != 1
            or not isinstance(state["images"], dict)):
        raise ValueError("invalid YOLO project annotation ledger")
    classes = _yolo_classes(state["classes"])
    for filename, record in state["images"].items():
        relative = Path(filename)
        if (not filename or "\\" in filename or relative.is_absolute()
                or any(part in (".", "..") for part in relative.parts)
                or relative.as_posix() != filename or not isinstance(record, dict)
                or set(record) != {"shape", "source_sha256", "boxes"}):
            raise ValueError("invalid YOLO project annotation record")
        shape = _yolo_shape(record["shape"])
        digest = record["source_sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or not isinstance(record["boxes"], list)):
            raise ValueError("invalid YOLO project annotation record")
        if [list(box) for box in _yolo_boxes(
                record["boxes"], shape[1], shape[0], len(classes))] != record["boxes"]:
            raise ValueError("invalid YOLO project annotation boxes")
    return state


def _yolo_atomic(path: Path, payload: bytes) -> None:
    """Replace one annotation artifact only after its complete write."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.",
                suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _yolo_json_bytes(value) -> bytes:
    """Encode deterministic UTF-8 JSON with a trailing newline."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n").encode("utf-8")


@contextlib.contextmanager
def _yolo_project_lock(path: Path):
    """Own an exclusive lock for metadata reads, changes and replacements.

    A busy or abandoned lock fails closed. Its owner must resolve an abandoned
    lock explicitly; a second writer never guesses that it may steal one.
    """
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                             0o600)
    except FileExistsError as exc:
        raise ValueError(f"YOLO annotation project is busy: {lock}") from exc
    identity = os.fstat(descriptor)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        try:
            current = lock.lstat()
        except FileNotFoundError:
            current = None
        if current is not None and (current.st_dev, current.st_ino) == (
                identity.st_dev, identity.st_ino):
            lock.unlink()


def save_yolo_boxes(folder, filename, image_shape, boxes, classes,
                    expected_source_sha256=None) -> Path:
    """Atomically save one source-bound box record in the project's ledger.

    ``folder`` holds the untouched source image. ``filename`` is relative to
    that folder. Existing class IDs retain their meanings: the old class map
    must be a prefix of ``classes``. The saved source SHA-256 and full image
    shape must agree when :func:`load_yolo_boxes` is next called. When an
    expected source digest is supplied, an external image change between
    loading and saving is refused before touching the ledger.

    :param folder: project folder holding the original image.
    :param filename: source-image path relative to ``folder``.
    :param image_shape: shape of the displayed source image.
    :param boxes: ``(class_id, x0, y0, x1, y1)`` full-image pixel boxes.
    :param classes: ordered names whose indexes are the class IDs.
    :param expected_source_sha256: digest received from a previous load,
        or None when the caller has no prior image snapshot.
    :returns: the private project-ledger path.
    """
    shape = _yolo_shape(image_shape)
    names = _yolo_classes(classes)
    canonical = _yolo_boxes(boxes, shape[1], shape[0], len(names))
    label, source = _yolo_relative(folder, filename)
    path = Path(folder).resolve() / YOLO_ANNOTATIONS_NAME
    with _yolo_project_lock(path):
        state = _yolo_state(path)
        if path.exists() and names[:len(state["classes"])] != state["classes"]:
            raise ValueError("YOLO class map would reassign an existing class ID")
        digest = _yolo_source_hash(source)
        if expected_source_sha256 is not None and digest != expected_source_sha256:
            raise ValueError("YOLO source image changed since it was loaded")
        existing = state["images"].get(label)
        if existing is not None:
            if existing["source_sha256"] != digest:
                raise ValueError("YOLO annotation source bytes changed")
            if existing["shape"] != list(shape):
                raise ValueError("YOLO annotation image shape changed")
        state["classes"] = names
        state["images"][label] = {
            "shape": list(shape), "source_sha256": digest,
            "boxes": [list(box) for box in canonical],
        }
        _yolo_atomic(path, _yolo_json_bytes(state))
    return path


def load_yolo_boxes(folder, filename, image_shape) -> dict:
    """Load this source's boxes, refusing stale bytes or changed image shape.

    An unannotated source returns the existing project classes and no boxes.
    A project with no ledger starts with class ``object`` and no boxes.

    :param folder: project folder holding the original image.
    :param filename: source-image path relative to ``folder``.
    :param image_shape: shape of the displayed source image.
    :returns: classes, canonical boxes and the current source SHA-256.
    """
    shape = _yolo_shape(image_shape)
    label, source = _yolo_relative(folder, filename)
    state = _yolo_state(Path(folder).resolve() / YOLO_ANNOTATIONS_NAME)
    record = state["images"].get(label)
    digest = _yolo_source_hash(source)
    if record is None:
        return {"classes": list(state["classes"]), "boxes": [],
                "source_sha256": digest}
    if record["shape"] != list(shape) or record["source_sha256"] != digest:
        raise ValueError("YOLO annotation source bytes or image shape changed")
    return {"classes": list(state["classes"]),
            "boxes": [tuple(box) for box in record["boxes"]],
            "source_sha256": digest}


def export_yolo_boxes(path, boxes, image_shape, classes) -> str:
    """Atomically write a YOLO label text file and sibling class metadata.

    The label path must end in ``.txt``. Its sibling ``.classes.json`` keeps
    the ordered class names; an existing map may only be extended. Empty
    ``boxes`` intentionally writes an empty label file for a negative image.
    Source images and masks are never opened or converted by this export.

    :param path: chosen ``.txt`` path in an existing directory.
    :param boxes: ``(class_id, x0, y0, x1, y1)`` full-image pixel boxes.
    :param image_shape: shape of the image the labels describe.
    :param classes: ordered names whose indexes are the class IDs.
    :returns: the written label text path as a string.
    """
    target = Path(path)
    if target.suffix.lower() != ".txt" or not target.parent.is_dir():
        raise ValueError("YOLO export needs a .txt path in an existing folder")
    metadata = target.parent / YOLO_CLASSES_NAME
    shape = _yolo_shape(image_shape)
    names = _yolo_classes(classes)
    canonical = _yolo_boxes(boxes, shape[1], shape[0], len(names))
    lines = yolo_box_lines(canonical, shape[1], shape[0])
    with _yolo_project_lock(metadata):
        if target.is_symlink() or metadata.is_symlink():
            raise ValueError("YOLO export cannot replace symbolic links")
        if target.exists() and not target.is_file():
            raise ValueError("YOLO label destination is not a regular file")
        if metadata.exists():
            old = _yolo_read_json(metadata)
            if not isinstance(old, dict) or set(old) != {"classes"}:
                raise ValueError("invalid YOLO class metadata")
            existing = _yolo_classes(old["classes"])
            if names[:len(existing)] != existing:
                raise ValueError("YOLO class map would reassign an existing class ID")
        _yolo_atomic(metadata, _yolo_json_bytes({"classes": names}))
        _yolo_atomic(target, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
    return str(target)

#: What a ledger created by this screen records as having made the edits.
CURATION_SOURCE = "spacr-qt make_masks"

#: Eight-connectivity: two objects touching only at a corner are one blob to
#: the eye and must be one object to the label image too, or a hand-drawn
#: diagonal stroke arrives on disk as a string of separate cells.
def _ndimage():
    """``scipy.ndimage``, imported on first use rather than at module scope.

    THIS MODULE IS ON THE STARTUP PATH. `app.folded_children()` imports every
    fold host to read its `FOLDED_APPS`, and `make_masks` imports this one --
    so a module-level `from scipy.ndimage import ...` put scipy into the
    process before Home had painted. The packaged smoke test asserts Home
    crosses no operation-only import boundary and named scipy for exactly
    that reason.

    :returns: the ``scipy.ndimage`` module.
    """
    from scipy import ndimage

    return ndimage


_EIGHT = np.ones((3, 3), dtype=np.uint8)


_CENTER_KERNEL_NORMS = {}
"""Seeded white-noise response standard deviations for centre-puncta LoG scales."""


def _center_puncta_kernel_norm(sigma):
    """Return the seeded LoG response to unit white noise at one scale."""
    if sigma not in _CENTER_KERNEL_NORMS:
        white = np.random.default_rng(20261003).standard_normal((512, 512)).astype(np.float32)
        response = -(sigma ** 2) * _ndimage().gaussian_laplace(white, sigma)
        _CENTER_KERNEL_NORMS[sigma] = float(np.std(response))
    return _CENTER_KERNEL_NORMS[sigma]


def _center_puncta_pixels(image, y, x, count):
    """Return the subpixel centroid and stable nearest-pixel coordinates."""
    if y < 4 or x < 4 or y + 5 > image.shape[0] or x + 5 > image.shape[1]:
        return None
    weights = image[y-1:y+2, x-1:x+2].astype(np.float64)
    weights -= weights.min()
    total = weights.sum()
    cy, cx = float(y), float(x)
    if total > 0:
        oy, ox = np.mgrid[-1:2, -1:2]
        cy += float((oy * weights).sum() / total)
        cx += float((ox * weights).sum() / total)
    yy, xx = np.mgrid[y-4:y+5, x-4:x+5]
    order = np.argsort((yy-cy)**2 + (xx-cx)**2, axis=None, kind='stable')[:count]
    return cy, cx, yy.ravel()[order], xx.ravel()[order]


def _center_puncta_candidates(image, parent_labels, params):
    """Measure every noise-standardised LoG centre before intensity inclusion.

    Filters operate on the original float32 channel, never a display stretch.
    The same-parent annulus excludes all nonmaximum-suppressed candidates,
    including centres later rejected by the edge or intensity criteria.
    The centre mean includes the selected pixels even where neighbouring
    centre windows overlap. Output labels resolve that overlap separately.
    """
    import pandas as pd
    from skimage.feature import peak_local_max
    from .i18n import tr

    im = np.asarray(image, dtype=np.float32)
    parents = np.asarray(parent_labels)
    if im.ndim != 2 or parents.shape != im.shape:
        raise ValueError(tr('Puncta detection requires a 2-D image and matching parent mask.'))
    if not np.isfinite(im).all() or not np.isfinite(parents).all():
        raise ValueError(tr('Image and parent mask must contain finite values.'))
    if np.any(parents < 0) or not np.equal(parents, np.floor(parents)).all():
        raise ValueError(tr('Parent masks must contain nonnegative integer labels.'))
    sigmas = tuple(float(s) for s in params.puncta_sigmas)
    scalars = (params.puncta_k, params.puncta_min_corrected, params.puncta_edge_margin,
               params.puncta_min_distance, params.puncta_center_pixels)
    if (not sigmas or not all(np.isfinite(s) and s > 0 for s in sigmas)
            or not all(np.isfinite(s) for s in scalars)
            or params.puncta_k <= 0 or params.puncta_edge_margin < 0
            or not 1 <= params.puncta_center_pixels <= 81
            or int(params.puncta_center_pixels) != params.puncta_center_pixels
            or params.puncta_min_distance < 1
            or int(params.puncta_min_distance) != params.puncta_min_distance):
        raise ValueError(tr('Invalid puncta scale, threshold, centre-pixel or spacing settings.'))
    columns = ['candidate_id', 'object_label', 'parent_id', 'cyst_id', 'y', 'x',
               'z', 'sigma', 'peak', 'center3', 'disc13', 'center10', 'center20', 'center40', 'center_mean',
               'center_pixels', 'center_corrected', 'corr10', 'corrected20', 'corr40', 'cy', 'cx', 'local_bg', 'cyst_bg',
               'corrected', 'corrected_cyst', 'noise_mad', 'included', 'mask_pixels']
    rows = []
    if not np.any(parents > 0):
        return pd.DataFrame(rows, columns=columns)
    ndi = _ndimage()
    high = im - ndi.gaussian_filter(im, 1.0)
    responses = [-(s ** 2) * ndi.gaussian_laplace(im, s) for s in sigmas]
    dy, dx = np.ogrid[-2:3, -2:3]
    disc = dy**2 + dx**2 <= 4
    for parent_id in np.unique(parents[parents > 0]):
        region = parents == parent_id
        distance = ndi.distance_transform_edt(region)
        hv = high[region]
        noise = float(1.4826 * np.median(np.abs(hv - np.median(hv))) / 0.87)
        best = np.full(im.shape, -np.inf, dtype=np.float32)
        best_sigma = np.zeros(im.shape, dtype=np.float32)
        for sigma, response in zip(sigmas, responses):
            z = response / (max(noise, 1e-3) * _center_puncta_kernel_norm(sigma))
            update = z > best
            best[update] = z[update]
            best_sigma[update] = sigma
        best = np.where(region, best, -np.inf)
        peaks = peak_local_max(best, min_distance=int(params.puncta_min_distance),
                               threshold_abs=float(params.puncta_k),
                               exclude_border=False, labels=region.astype(np.int32))
        if not len(peaks):
            continue
        zs = best[peaks[:, 0], peaks[:, 1]]
        order = np.argsort(-zs)
        peaks, zs = peaks[order], zs[order]
        scales = best_sigma[peaks[:, 0], peaks[:, 1]]
        keep = np.ones(len(peaks), dtype=bool)
        for index, (y, x) in enumerate(peaks):
            if keep[index]:
                radius = max(2.0, 1.5 * scales[index])
                gaps = (peaks[index+1:, 0]-y)**2 + (peaks[index+1:, 1]-x)**2
                keep[index+1:] &= gaps > radius * radius
        peaks, zs, scales = peaks[keep], zs[keep], scales[keep]
        excluded = np.zeros(im.shape, dtype=bool)
        for (y, x), sigma in zip(peaks, scales):
            radius = int(np.ceil(2.0 * sigma))
            y0, y1 = max(0, y-radius), min(im.shape[0], y+radius+1)
            x0, x1 = max(0, x-radius), min(im.shape[1], x+radius+1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            excluded[y0:y1, x0:x1] |= (yy-y)**2 + (xx-x)**2 <= radius * radius
        cytoplasm = region & ~excluded
        cyst_bg = (float(np.median(im[cytoplasm])) if cytoplasm.sum() >= 0.2 * region.sum()
                   else float(np.percentile(im[region], 30)))
        for (y, x), z, sigma in zip(peaks, zs, scales):
            if distance[y, x] < params.puncta_edge_margin:
                continue
            selected = _center_puncta_pixels(im, int(y), int(x), int(params.puncta_center_pixels))
            if selected is None:
                continue
            cy, cx, py, px = selected
            center_mean = float(im[py, px].mean())
            twenty = _center_puncta_pixels(im, int(y), int(x), 20)
            center20 = float(im[twenty[2], twenty[3]].mean())
            ten = _center_puncta_pixels(im, int(y), int(x), 10)
            forty = _center_puncta_pixels(im, int(y), int(x), 40)
            center10 = float(im[ten[2], ten[3]].mean())
            center40 = float(im[forty[2], forty[3]].mean())
            center3 = float(im[y-1:y+2, x-1:x+2].mean())
            disc13 = float(im[y-2:y+3, x-2:x+3][disc].mean())
            inner = max(4.0, 2.5 * sigma + 2.0)
            outer = inner + 4.0
            radius = int(np.ceil(outer))
            y0, y1 = max(0, y-radius), min(im.shape[0], y+radius+1)
            x0, x1 = max(0, x-radius), min(im.shape[1], x+radius+1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            squared = (yy-y)**2 + (xx-x)**2
            annulus = ((squared >= inner*inner) & (squared <= outer*outer)
                       & region[y0:y1, x0:x1] & ~excluded[y0:y1, x0:x1])
            if annulus.sum() < 12:
                annulus = ((squared >= inner*inner) & (squared <= outer*outer)
                           & region[y0:y1, x0:x1])
            local_bg = float(np.median(im[y0:y1, x0:x1][annulus])) if annulus.sum() >= 6 else cyst_bg
            included = center_mean - local_bg >= params.puncta_min_corrected
            rows.append(dict(candidate_id=len(rows)+1, object_label=0,
                             parent_id=int(parent_id), cyst_id=int(parent_id),
                             y=int(y), x=int(x), z=float(z), sigma=float(sigma),
                             peak=float(im[y, x]), center3=center3, disc13=disc13,
                             center10=center10, center20=center20, center40=center40, center_mean=center_mean,
                             center_pixels=int(params.puncta_center_pixels),
                             center_corrected=center_mean-local_bg, corr10=center10-local_bg,
                             corrected20=center20-local_bg, corr40=center40-local_bg, cy=cy, cx=cx,
                             local_bg=local_bg, cyst_bg=cyst_bg,
                             corrected=center3-local_bg, corrected_cyst=center3-cyst_bg,
                             noise_mad=noise, included=bool(included), mask_pixels=0))
    return pd.DataFrame(rows, columns=columns)


def _center_puncta_instances(image, parent_labels, params):
    """Return centre labels and exact candidate measurements without growing spots.

    Every accepted integer peak is reserved for its object. Other shared
    centre pixels go to the closest subpixel centre, with stronger-first
    ordering breaking exact ties. Pixels outside the parent are omitted
    from masks. Exact full-N intensity means remain in the candidate table;
    mask areas may therefore be smaller than N and are reported explicitly.
    """
    im = np.asarray(image, dtype=np.float32)
    parents = np.asarray(parent_labels)
    candidates = _center_puncta_candidates(im, parents, params)
    labels = np.zeros(im.shape, dtype=np.int32)
    accepted = candidates.index[candidates['included'].astype(bool)]
    reserved = np.zeros(im.shape, dtype=bool)
    owner_distance = np.full(im.shape, np.inf, dtype=np.float64)
    for label, index in enumerate(accepted, 1):
        candidates.at[index, 'object_label'] = label
        y, x = int(candidates.at[index, 'y']), int(candidates.at[index, 'x'])
        labels[y, x] = label
        reserved[y, x] = True
    for index in accepted:
        row = candidates.loc[index]
        selected = _center_puncta_pixels(im, int(row.y), int(row.x), int(params.puncta_center_pixels))
        cy, cx, yy, xx = selected
        if not np.any((yy == int(row.y)) & (xx == int(row.x))):
            yy[-1], xx[-1] = int(row.y), int(row.x)
        squared = (yy-cy)**2 + (xx-cx)**2
        take = ((parents[yy, xx] == row.parent_id) & ~reserved[yy, xx]
                & (squared < owner_distance[yy, xx]))
        labels[yy[take], xx[take]] = int(row.object_label)
        owner_distance[yy[take], xx[take]] = squared[take]
    counts = np.bincount(labels.ravel())
    for index in accepted:
        candidates.at[index, 'mask_pixels'] = int(counts[int(candidates.at[index, 'object_label'])])
    return labels, candidates


def list_images(folder: str) -> List[str]:
    """Return filenames of image files in `folder`, sorted, or [].

    :param folder: directory to list (not recursively); an empty value or a
        missing directory gives ``[]``. Only names ending in
        :data:`IMAGE_EXTS`, case-insensitively, are kept.
    """
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    )


def is_seg_bundle(filename) -> bool:
    """Whether ``filename`` names a Cellpose ``_seg.npy`` bundle.

    :param filename: a file name or path.
    :returns: ``True`` when it ends in :data:`SEG_SUFFIX`, which is how a
        ``seg`` queue's fields are named in the editor's file list.
    """
    return os.path.basename(str(filename)).endswith(SEG_SUFFIX)


def field_stem(filename) -> str:
    """The stem a field is known by in ``curate_status.csv``.

    ``os.path.splitext`` gives ``well_A1_seg`` for ``well_A1_seg.npy``, and
    the queue calls that field ``well_A1``; a status row written under the
    first name would never take the field out of the queue.

    :param filename: an image file name, or a ``_seg.npy`` bundle name.
    :returns: the file name without its extension, or without
        :data:`SEG_SUFFIX` for a bundle.
    """
    name = os.path.basename(str(filename))
    if name.endswith(SEG_SUFFIX):
        return name[:-len(SEG_SUFFIX)]
    return os.path.splitext(name)[0]


def masks_folder(folder: str, masks_dir: Optional[str] = None) -> str:
    """Where the masks of the images in ``folder`` are kept.

    :param folder: the folder the editor opened.
    :param masks_dir: the masks folder, when it is not beneath ``folder`` --
        the ``sibling`` layout's ``<root>/masks`` beside ``<root>/images``.
    :returns: ``masks_dir`` when given, else ``<folder>/masks``.
    """
    if masks_dir:
        return os.fspath(masks_dir)
    return os.path.join(folder, "masks")


def _as_field_image(image: np.ndarray, image_path: str) -> np.ndarray:
    """Check one decoded image and bring it to the editor's uint16 grey.

    Finite images with a negative minimum have that minimum subtracted in
    float64 before scaling their maximum to 65535. A constant negative field
    becomes zeros. Nonnegative uint16 values are kept; other nonnegative
    images retain their zero-to-maximum display scaling.

    :param image: the decoded pixels.
    :param image_path: where they came from, for the messages.
    :returns: a 2-D uint16 image.
    :raises ValueError: for an unsupported shape or channel count, or
        non-finite intensities.
    """
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = np.squeeze(image, axis=-1)
        elif image.shape[2] == 4:
            image = image[..., :3]
            image = np.dot(
                image, [0.2989, 0.5870, 0.1140]
            )
        elif image.shape[2] == 3:
            image = np.dot(
                image, [0.2989, 0.5870, 0.1140]
            )
        else:
            raise ValueError(
                f"Unsupported channel count {image.shape[2]} in {image_path}; "
                "expected grayscale, RGB, or RGBA."
            )
    if image.ndim != 2:
        raise ValueError(
            f"Unsupported image shape {image.shape} in {image_path}; "
            "Make Masks expects one 2-D field."
        )
    if not np.all(np.isfinite(image)):
        raise ValueError(f"Image contains non-finite values: {image_path}")
    if image.size and float(image.min()) < 0:
        image = image.astype(np.float64) - float(image.min())
    if image.dtype != np.uint16:
        max_val = float(image.max()) if image.size else 1.0
        if max_val <= 0:
            max_val = 1.0
        image = (image / max_val * 65535.0).astype(np.uint16)
    return image


def _as_field_mask(mask: np.ndarray, shape, mask_path: str,
                   filename: str) -> np.ndarray:
    """Check one decoded label image against the image it belongs to.

    :param mask: the decoded labels.
    :param shape: the 2-D shape of the image they label.
    :param mask_path: where they came from, for the messages.
    :param filename: the field's name, for the shape-mismatch message.
    :returns: unsigned labels wide enough for every ID, retaining existing uint32 storage.
    :raises ValueError: for a mask that is not 2-D, does not match the
        image, holds non-integer or negative labels, or needs more than 32
        bits in uint32 storage.
    """
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
    if mask.ndim != 2:
        raise ValueError(
            f"Unsupported mask shape {mask.shape} in {mask_path}; "
            "expected a 2-D label image."
        )
    if mask.shape != tuple(shape):
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape "
            f"{tuple(shape)} for {filename}."
        )
    if not np.issubdtype(mask.dtype, np.integer):
        if not np.all(np.isfinite(mask)):
            raise ValueError(f"Mask contains non-finite values: {mask_path}")
        if np.any(mask < 0) or np.any(mask != np.floor(mask)):
            raise ValueError(
                f"Mask must contain non-negative integer labels: {mask_path}"
            )
    if np.issubdtype(mask.dtype, np.signedinteger) and np.any(mask < 0):
        from .i18n import tr

        raise ValueError(tr(
            "Mask must contain non-negative integer labels: {path}", path=mask_path))
    maximum = int(mask.max()) if mask.size else 0
    if maximum > MAX_MASK_LABEL:
        from .i18n import tr

        raise ValueError(tr("Mask label {label} exceeds uint32 capacity: {path}",
                            label=maximum, path=mask_path))
    dtype = np.uint32 if maximum > 65535 or mask.dtype == np.uint32 else (
        np.uint8 if maximum <= 255 else np.uint16)
    return mask.astype(dtype)


def load_image_and_mask(folder: str, filename: str,
                        masks_dir: Optional[str] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image and its accompanying mask.

    - Multi-channel images are collapsed to grayscale via BT.601 weights.
    - Missing masks are created as zeros of the image shape.
    - Images are returned as uint16; masks preserve uint8/uint16/uint32 label IDs.
    - A mask saved by :func:`save_mask` is found even when the source image
      had a non-TIFF extension. Its canonical TIFF takes precedence over
      an older mask under the source image's extension.
    - A ``filename`` ending in :data:`SEG_SUFFIX` is a Cellpose bundle and is
      read by :func:`load_seg_bundle` instead.

    :param folder: the folder holding the image, or the bundle.
    :param filename: the image, or the ``_seg.npy`` bundle, to load.
    :param masks_dir: where the masks are, when not in ``<folder>/masks``;
        see :func:`masks_folder`.
    :returns: ``(image, mask)``.
    :raises ValueError: for unsupported dimensions or an image/mask shape
        mismatch.
    """
    if is_seg_bundle(filename):
        return load_seg_bundle(os.path.join(folder, filename))
    image_path = os.path.join(folder, filename)
    image = _as_field_image(read_image(image_path), image_path)

    mask_dir = masks_folder(folder, masks_dir)
    stem = os.path.splitext(filename)[0]
    candidates = [
        os.path.join(mask_dir, stem + ".tif"),
        os.path.join(mask_dir, filename),
        os.path.join(mask_dir, stem + ".tiff"),
    ]
    mask_path = next((path for path in candidates if os.path.isfile(path)), "")
    if mask_path:
        mask = _as_field_mask(imageio.imread(mask_path), image.shape,
                              mask_path, filename)
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    return image, mask


def read_seg_bundle(path: str) -> Dict:
    """Read a Cellpose ``_seg.npy`` bundle as the dict it holds.

    A bundle is a pickle, and unpickling runs whatever the file says, so a
    bundle is only ever read from a queue folder the curator chose -- the
    same trust the external curation tool and Cellpose itself extend to it.

    :param path: the bundle.
    :returns: the bundle's dict, with every key it was written with.
    :raises ValueError: when the file does not hold a dict with a ``masks``
        entry.
    """
    loaded = np.load(path, allow_pickle=True)
    try:
        payload = loaded.item()
    except (AttributeError, ValueError):
        payload = None
    if not isinstance(payload, dict) or "masks" not in payload:
        raise ValueError(
            f"{path} is not a Cellpose _seg.npy bundle: expected a dict "
            f"holding 'masks'.")
    return payload


def _bundle_image(path: str, payload: Dict, shape) -> Tuple[np.ndarray, str]:
    """Find the pixels a bundle's labels were drawn on.

    In the order the external curation tool used:

    1. the original the bundle names in ``source_image``, looked for BY
       NAME where that tool looked for it -- ``new_originals/`` and then
       ``training_data/`` beside the queue folder, then beside the bundle
       itself -- and used only when its shape matches the labels. The
       stored path itself is never touched, which is the one step of that
       tool's search left out: it is absolute and from whichever machine
       staged the set, and a stat on another machine's mount can hang the
       thread that asked;
    2. the ``img`` the bundle carries, often an 8-bit display copy of that
       original;
    3. an image of the bundle's own stem beside it, the display copy the
       external tool writes next to each bundle.

    :param path: the bundle.
    :param payload: its dict, from :func:`read_seg_bundle`.
    :param shape: the shape of its labels.
    :returns: ``(pixels, where they came from)``.
    :raises ValueError: when none of the three exists.
    """
    folder = os.path.dirname(path)
    source = payload.get("source_image")
    if source:
        name = os.path.basename(str(source))
        project = os.path.dirname(os.path.abspath(folder))
        for original in (os.path.join(project, "new_originals", name),
                         os.path.join(project, "training_data", name),
                         os.path.join(folder, name)):
            if not os.path.isfile(original):
                continue
            try:
                pixels = np.asarray(read_image(original))
            except Exception:
                continue
            if pixels.shape[:2] == tuple(shape)[:2]:
                return pixels, original
    embedded = payload.get("img")
    if embedded is not None:
        pixels = np.asarray(embedded)
        if (pixels.ndim == 3 and pixels.shape[1:] == tuple(shape)[:2]
                and pixels.shape[:2] != tuple(shape)[:2]):
            pixels = np.moveaxis(pixels, 0, -1)
        return pixels, path
    stem = field_stem(path)
    for ext in IMAGE_EXTS:
        beside = os.path.join(folder, stem + ext)
        if os.path.isfile(beside):
            return np.asarray(read_image(beside)), beside
    raise ValueError(
        f"{path} carries no image ('img') and there is no {stem}.<ext> beside "
        f"it, so there is nothing to draw its labels on.")


def load_seg_bundle(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a Cellpose bundle as the editor's ``(image, mask)`` pair.

    :param path: the ``_seg.npy`` bundle.
    :returns: ``(image, mask)``, checked and converted exactly as
        :func:`load_image_and_mask` converts a TIFF pair.
    :raises ValueError: for a file that is not a bundle, a bundle with no
        image to show, or labels that do not fit the image.
    """
    payload = read_seg_bundle(path)
    labels = np.asarray(payload["masks"])
    pixels, source = _bundle_image(path, payload, labels.shape)
    image = _as_field_image(pixels, source)
    return image, _as_field_mask(labels, image.shape, path,
                                 os.path.basename(path))


def seg_outlines(labels: np.ndarray, like) -> np.ndarray:
    """The ``outlines`` entry of a bundle, redrawn for ``labels``.

    A pixel is on an outline when it belongs to an object and one of its four
    neighbours does not belong to the same one. ``like`` is the entry the
    bundle had: an entry holding only 0 and 1 gets a 0/1 outline back, and
    one holding ids gets each outline pixel's id, which is what Cellpose's
    own ``masks_flows_to_seg`` writes. An entry of all zeros says nothing
    about its form and gets ids, Cellpose's default.

    :param labels: the labels being saved.
    :param like: the bundle's previous ``outlines``, for its form and type.
    :returns: the new outlines, the shape of ``labels``.
    """
    labels = np.asarray(labels)
    old = np.asarray(like)
    padded = np.pad(labels, 1, mode="edge")
    height, width = labels.shape
    edge = np.zeros(labels.shape, dtype=bool)
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        edge |= padded[dy:dy + height, dx:dx + width] != labels
    edge &= labels > 0
    peak = float(np.max(old)) if old.size else 0.0
    binary = old.dtype == bool or 0.0 < peak <= 1.0
    if binary:
        return edge.astype(old.dtype)
    outlined = np.where(edge, labels, 0)
    if np.issubdtype(old.dtype, np.integer) and \
            int(labels.max(initial=0)) <= np.iinfo(old.dtype).max:
        return outlined.astype(old.dtype)
    return outlined.astype(labels.dtype)


def _write_bundle(path: str, payload: Dict) -> None:
    """Write a bundle's dict so that a crash leaves the old file whole.

    Written beside the target under a dot-name the queue does not list,
    then renamed over it. ``np.save`` is given a handle rather than a path
    because, given a path, it appends ``.npy`` to one that does not end in
    it.

    :param path: the bundle to write.
    :param payload: the dict to put in it.
    """
    folder, name = os.path.split(path)
    temporary = os.path.join(folder, f".{name}.tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, payload, allow_pickle=True)
    os.replace(temporary, path)


def save_seg_bundle(path: str, mask: np.ndarray, *, preserve_ids: bool = False) -> str:
    """Write edited labels back into the bundle they came from.

    Every key the bundle already had is kept -- ``img``, ``flows``,
    ``filename``, ``source_image``, ``diameter``, and any other -- and
    ``masks`` is replaced by :func:`canonical_labels` of ``mask``. Two keys
    are DERIVED from the masks and would describe the old ones if left:

    * ``outlines`` is redrawn by :func:`seg_outlines`;
    * ``ismanual``, one flag per object, is resized to the new largest id:
      an id the bundle already flagged keeps its flag, and an id beyond the
      old list was drawn in this editor, so it is flagged manual.

    :param path: the ``_seg.npy`` bundle.
    :param mask: the edited labels.
    :param preserve_ids: keep every supplied label exactly, including a lone
        ID or disconnected pieces with the same ID. Labels must fit uint16.
    :returns: ``path``.
    :raises ValueError: when ``path`` is not a bundle.
    """
    payload = read_seg_bundle(path)
    labels = canonical_labels(mask, preserve_ids=preserve_ids)
    if int(labels.max(initial=0)) > 65535:
        from .i18n import tr

        raise ValueError(tr("Save 32-bit manual identities as TIFF masks, not Cellpose bundles."))
    payload["masks"] = labels
    outlines = payload.get("outlines")
    if outlines is not None and np.shape(outlines) == labels.shape:
        payload["outlines"] = seg_outlines(labels, outlines)
    manual = payload.get("ismanual")
    if manual is not None and np.ndim(manual) == 1:
        old = np.asarray(manual, dtype=bool)
        count = int(labels.max(initial=0))
        flags = np.ones(count, dtype=bool)
        keep = min(count, old.size)
        flags[:keep] = old[:keep]
        payload["ismanual"] = flags
    _write_bundle(path, payload)
    return path


def mask_save_path(folder: str, filename: str,
                   masks_dir: Optional[str] = None) -> str:
    """Where this field's mask is written -- and where its ledger sits.

    :func:`load_image_and_mask` will accept a mask under the image's own
    extension, but everything :func:`save_mask` writes lands on
    ``<masks folder>/<stem>.tif``. The ledger is keyed on the file that was
    actually written, so both have to agree on one name; ask here rather
    than rebuilding it at each call site.

    :param folder: the folder the editor opened.
    :param filename: the field's image, or its ``_seg.npy`` bundle.
    :param masks_dir: where the masks are, when not in ``<folder>/masks``.
    :returns: the mask's path; for a bundle, the bundle itself, which is
        where its labels are written back.
    """
    if is_seg_bundle(filename):
        return os.path.join(folder, os.path.basename(str(filename)))
    stem = os.path.splitext(filename)[0]
    return os.path.join(masks_folder(folder, masks_dir), stem + ".tif")


#: The curation CSV's columns, in a fixed order. They are written on every
#: rewrite, so a reader never has
#: to guess which column is which.
CURATION_COLUMNS: Tuple[str, ...] = (
    "image path", "mask path", "object count", "keep",
)

#: The file itself. One per folder of images rather than one per field: a
#: verdict is only useful beside the others taken in the same sitting.
CURATION_CSV_NAME = "keep_discard.csv"


def curation_folder(folder: str) -> str:
    """Where a folder of images keeps its curation files.

    spaCR's layout: masks live at ``<images>/masks`` and the curation CSV at
    ``<images>/csv``. It is a folder rather than a
    file beside the images so that a folder listing of the fields is still
    a listing of the fields.

    :param folder: the folder the editor opened.
    :returns: ``<folder>/csv``.
    """
    return os.path.join(folder, "csv")


def curation_csv_path(folder: str) -> str:
    """The keep/discard CSV for the images in ``folder``.

    :param folder: the folder the editor opened.
    :returns: ``<folder>/csv/keep_discard.csv``.
    """
    return os.path.join(curation_folder(folder), CURATION_CSV_NAME)


def read_curation(folder: str) -> Dict[str, Dict[str, str]]:
    """Every verdict recorded for ``folder``, keyed by image path.

    A file that is missing, empty or unreadable is NO VERDICTS rather than
    an error: this is a curation aid, and refusing to open a folder because
    its CSV was edited by hand would be the wrong trade.

    :param folder: the folder the editor opened.
    :returns: image path -> the row, as strings.
    """
    path = curation_csv_path(folder)
    rows: Dict[str, Dict[str, str]] = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row.get(CURATION_COLUMNS[0]) or "").strip()
                if key:
                    rows[key] = dict(row)
    except (OSError, csv.Error, UnicodeDecodeError):
        return {}
    return rows


def curation_verdict(folder: str, image_path: str) -> Optional[bool]:
    """Whether this field is marked keep, discard, or not marked at all.

    :param folder: the folder the editor opened.
    :param image_path: the field.
    :returns: True for keep, False for discard, None for no row.
    """
    return _row_verdict(read_curation(folder).get(os.fspath(image_path)))


def _row_verdict(row) -> Optional[bool]:
    """Keep (True), Discard (False) or no verdict (None) from one CSV row."""
    if row is None:
        return None
    value = str(row.get(CURATION_COLUMNS[3], "")).strip().lower()
    if value in ("true", "1", "yes", "keep"):
        return True
    if value in ("false", "0", "no", "discard"):
        return False
    return None


def _first_unreviewed(fields) -> Optional[int]:
    """The queue position of the first field with neither Keep nor Discard.

    Reads the persisted ``keep_discard.csv`` of each field's folder once, so
    verdicts recorded in an earlier session count (item 685).

    :param fields: ``(folder, image path)`` pairs in queue order.
    :returns: the zero-based index, or None when every field has a verdict
        or there are none.
    """
    verdicts = {}
    for index, (folder, image_path) in enumerate(fields):
        if folder not in verdicts:
            verdicts[folder] = read_curation(folder)
        if _row_verdict(verdicts[folder].get(os.fspath(image_path))) is None:
            return index
    return None


VVVV_EXPORT_FORMAT = "spacr-vvvv-export"
VVVV_EXPORT_VERSION = 1


def _vvvv_object_table(labels: np.ndarray, image: Optional[np.ndarray] = None,
                      channel_names=None, classes=None, scores=None):
    """One row per object of ``labels`` for the vvvv export (item 685).

    Coordinates are image pixels with the origin at the top-left pixel's
    corner, x to the right and y down; a centroid is the mean pixel centre
    (``index + 0.5``) and the bounding box is half-open, ``[x0, x1)``.
    Mean intensity is reported per channel of ``image`` as loaded, in its
    own units.

    :param labels: 2-D integer label image; 0 is background.
    :param image: the field, 2-D or ``(height, width, channels)``, or None.
    :param channel_names: one name per channel; ``channel_1`` ... otherwise.
    :param classes: optional ``{label: class name}``.
    :param scores: optional ``{label: score}``.
    :returns: a :class:`pandas.DataFrame`.
    """
    import pandas as pd
    from scipy import ndimage

    labels = np.asarray(labels)
    ids = [int(v) for v in np.unique(labels) if int(v) > 0]
    columns = ["label", "centroid_x", "centroid_y", "area",
               "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]
    channels = []
    if image is not None:
        stack = np.asarray(image, dtype=np.float64)
        if stack.ndim == 2:
            stack = stack[..., None]
        names = list(channel_names or ())
        names += [f"channel_{index + 1}"
                  for index in range(len(names), stack.shape[-1])]
        channels = [(f"mean_intensity_{names[index]}", stack[..., index])
                    for index in range(stack.shape[-1])]
    table = {name: [] for name in columns}
    for name, _plane in channels:
        table[name] = []
    if ids:
        areas = ndimage.sum_labels(np.ones(labels.shape), labels, ids)
        centres = ndimage.center_of_mass(np.ones(labels.shape), labels, ids)
        slices = _label_boxes(labels)
        means = [ndimage.mean(plane, labels, ids) for _name, plane in channels]
        for row, label in enumerate(ids):
            box = slices[label]
            table["label"].append(label)
            table["centroid_x"].append(float(centres[row][1]) + 0.5)
            table["centroid_y"].append(float(centres[row][0]) + 0.5)
            table["area"].append(int(areas[row]))
            table["bbox_x0"].append(int(box[1].start))
            table["bbox_y0"].append(int(box[0].start))
            table["bbox_x1"].append(int(box[1].stop))
            table["bbox_y1"].append(int(box[0].stop))
            for (name, _plane), values in zip(channels, means):
                table[name].append(float(np.atleast_1d(values)[row]))
    frame = pd.DataFrame(table)
    if classes:
        frame["class"] = [classes.get(label, "") for label in ids]
    if scores:
        frame["score"] = [scores.get(label) for label in ids]
    return frame


def _vvvv_outlines(labels: np.ndarray) -> np.ndarray:
    """Opaque white object outlines on a transparent RGBA background."""
    from skimage.segmentation import find_boundaries

    edge = find_boundaries(np.asarray(labels), mode="inner")
    rgba = np.zeros(labels.shape[:2] + (4,), dtype=np.uint8)
    rgba[edge] = (255, 255, 255, 255)
    return rgba


_VVVV_GAMMA_VERSION = "7.4"
_VVVV_LANGUAGE_VERSION = "2025.7.4"
_VL_ID_DIGITS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "abcdefghijklmnopqrstuvwxyz0123456789")


def _vvvv_display_image(image: np.ndarray) -> np.ndarray:
    """The field as 8-bit RGB for the vvvv patch to draw.

    Each channel is stretched between its 1st and 99.9th percentile, as
    Make Masks draws it; one channel is grey, two or three channels fill
    red, green and blue in order and only the first three are used.
    """
    stack = np.asarray(image, dtype=np.float64)
    if stack.ndim == 2:
        stack = stack[..., None]
    planes = []
    for index in range(min(stack.shape[-1], 3)):
        plane = stack[..., index]
        finite = plane[np.isfinite(plane)]
        lo = float(np.percentile(finite, 1.0)) if finite.size else 0.0
        hi = float(np.percentile(finite, 99.9)) if finite.size else 1.0
        if hi <= lo:
            hi = lo + 1.0
        scaled = (np.clip(np.nan_to_num(plane, nan=lo), lo, hi) - lo) / (hi - lo)
        planes.append(np.round(scaled * 255.0).astype(np.uint8))
    if len(planes) == 1:
        planes = planes * 3
    while len(planes) < 3:
        planes.append(np.zeros_like(planes[0]))
    return np.stack(planes, axis=-1)


class _VlWriter:
    """Builds a vvvv gamma ``.vl`` document (item 685, alpha).

    The element and attribute forms copy the documents vvvv gamma 7.4
    ships in its ``packs/*/help`` folders. An id is a version-4 GUID in
    vvvv's encoding: the GUID's 16 bytes as two 64-bit halves, each
    written as 11 base-62 digits over ``A-Z a-z 0-9``. The GUIDs are
    derived from ``seed``, so the same export writes the same file.
    """

    def __init__(self, seed: str):
        """Start a document whose element ids derive from ``seed``.

        :param seed: text that makes the ids of this export reproducible.
        """
        import xml.etree.ElementTree as ElementTree

        self.et = ElementTree
        self.seed = seed
        self.count = 0
        self.links = []

    def new_id(self) -> str:
        """The next reproducible version-4 GUID for an element of this document.

        :returns: the GUID text.
        """
        self.count += 1
        raw = bytearray(hashlib.sha256(
            f"{self.seed}/{self.count}".encode("utf-8")).digest()[:16])
        raw[7] = (raw[7] & 0x0F) | 0x40
        raw[8] = (raw[8] & 0x3F) | 0x80
        text = ""
        for half in (raw[:8], raw[8:]):
            value = int.from_bytes(half, "big")
            digits = []
            for _ in range(11):
                value, digit = divmod(value, 62)
                digits.append(_VL_ID_DIGITS[digit])
            text += "".join(reversed(digits))
        return text

    def sub(self, parent, tag: str, **attrs):
        """Append a child element, leaving out attributes whose value is None.

        :param parent: the parent element.
        :param tag: the child tag.
        :returns: the new element.
        """
        return self.et.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()
                                                if v is not None})

    def type_annotation(self, parent, category: str, dependency: str,
                        name: str):
        """Annotate ``parent`` with the vvvv type it carries.

        :param parent: the element to annotate.
        :param category: the type's category.
        :param dependency: the library the type comes from.
        :param name: the type name.
        :returns: the annotation element.
        """
        note = self.sub(parent, "{property}TypeAnnotation",
                        LastCategoryFullName=category,
                        LastDependency=dependency)
        self.sub(note, "Choice", Kind="TypeFlag", Name=name)
        return note

    def node(self, parent, bounds: str, category: str, dependency: str,
             kind: str, name: str, pins, *, fixed: bool = True,
             position_pin: bool = False):
        """A node and its pins.

        ``pins`` holds ``(name, kind, hidden)``, optionally followed by a
        default value and its ``(category, dependency, type)``.

        :returns: ``{pin name: pin id}``.
        """
        node = self.sub(parent, "Node", Bounds=bounds, Id=self.new_id())
        ref = self.sub(node, "{property}NodeReference",
                       LastCategoryFullName=category,
                       LastDependency=dependency)
        if fixed:
            self.sub(ref, "Choice", Kind="NodeFlag", Name="Node", Fixed="true")
        self.sub(ref, "Choice", Kind=kind, Name=name)
        if position_pin:
            self.sub(ref, "PinReference", Kind="InputPin", Name="Position")
        ids = {}
        for pin_name, pin_kind, hidden, *default in pins:
            ids[pin_name] = self.new_id()
            pin = self.sub(node, "Pin", Id=ids[pin_name], Name=pin_name,
                           Kind=pin_kind,
                           DefaultValue=default[0] if default else None,
                           IsHidden="true" if hidden else None)
            if len(default) > 1:
                self.type_annotation(pin, *default[1])
        return ids

    def pad(self, parent, bounds: str, value: str, category: str,
            dependency: str, type_name: str, comment: Optional[str] = None,
            *, note_size: Optional[int] = None) -> str:
        """An IOBox holding ``value``; a note when ``note_size`` is set."""
        pad_id = self.new_id()
        pad = self.sub(parent, "Pad", Id=pad_id, Comment=comment,
                       Bounds=bounds, ShowValueBox="true", isIOBox="true",
                       Value=value)
        self.type_annotation(pad, category, dependency, type_name)
        if note_size is not None:
            settings = self.sub(pad, "{property}ValueBoxSettings")
            size = self.sub(settings, "{property}fontsize",
                            **{"{property}Type": "Int32"})
            size.text = str(note_size)
            kind = self.sub(settings, "{property}stringtype",
                            **{"{property}Assembly": "VL.Core",
                               "{property}Type": "VL.Core.StringType"})
            kind.text = "Comment"
        return pad_id

    def link(self, source: str, sink: str) -> None:
        """Remember one link from an output pin to an input pin.

        :param source: the output pin id.
        :param sink: the input pin id.
        """
        self.links.append((source, sink))


def _vvvv_project_document(stem: str, width: int, height: int,
                           names: Dict[str, str], columns) -> str:
    """The vvvv gamma patch that shows one exported field (item 685).

    The patch runs as the document's Application: it draws
    ``names["image"]`` with ``names["outlines"]`` on top in a Skia
    Renderer window, marks every row of ``names["objects"]`` with a circle
    at its centroid and its label (and class, when the table has one), and
    reads ``manifest.json`` and the table every frame and reloads the
    images whenever the manifest changes. Paths are relative to the
    document, so the folder can be moved or copied whole.

    :param stem: the field's stem, for the window title.
    :param width: image width in pixels.
    :param height: image height in pixels.
    :param names: the export's file names, keyed as in the manifest.
    :param columns: the objects table's column names, in order.
    :returns: the document as UTF-8 text with a byte-order mark.
    """
    vl = _VlWriter(f"spacr-vvvv/{stem}/{width}x{height}/{','.join(columns)}")
    et = vl.et
    et.register_namespace("p", "property")
    columns = list(columns)
    height = max(int(height), 1)
    width = max(int(width), 1)
    scale = 2.0 / height
    marker = max(3.0, height / 100.0)
    font = max(8.0, height / 45.0)
    stroke = max(1.0, height / 400.0)
    core, basics, skia = "VL.CoreLib.vl", "CoreLibBasics.vl", "VL.Skia.vl"

    def num(value: float) -> str:
        """A number as vvvv writes it.

        :param value: the number.
        :returns: the text.
        """
        return f"{value:.9g}"

    def vec(x: float, y: float) -> str:
        """A 2-D vector as vvvv writes it.

        :param x: horizontal component.
        :param y: vertical component.
        :returns: the text.
        """
        return f"{num(x)}, {num(y)}"

    doc = et.Element("Document", {"xmlns:r": "reflection",
                                  "Id": vl.new_id(),
                                  "LanguageVersion": _VVVV_LANGUAGE_VERSION,
                                  "Version": "0.128"})
    vl.sub(doc, "NugetDependency", Id=vl.new_id(), Location="VL.CoreLib",
           Version=_VVVV_LANGUAGE_VERSION)
    vl.sub(doc, "NugetDependency", Id=vl.new_id(), Location="VL.Skia",
           Version=_VVVV_LANGUAGE_VERSION)
    top = vl.sub(doc, "Patch", Id=vl.new_id())
    vl.sub(top, "Canvas", Id=vl.new_id(), DefaultCategory="Main",
           CanvasType="FullCategory")
    app = vl.sub(top, "Node", Name="Application", Bounds="100,100",
                 Id=vl.new_id())
    ref = vl.sub(app, "{property}NodeReference")
    vl.sub(ref, "Choice", Kind="ContainerDefinition", Name="Process")
    vl.sub(ref, "FullNameCategoryReference", ID="Primitive")
    patch = vl.sub(app, "Patch", Id=vl.new_id())
    canvas = vl.sub(patch, "Canvas", Id=vl.new_id(), CanvasType="Group")

    flag = "ProcessAppFlag"
    call = "OperationCallFlag"
    hidden_context = ("Node Context", "InputPin", True)

    vl.pad(canvas, "40,30,620,60",
           f"spaCR export of {stem}: the image with object outlines and one "
           "marker per row of the objects table. The patch reads "
           "manifest.json and the objects table every frame and reloads the "
           "images when the manifest changes, so a new spaCR export shows "
           "at once.", "Primitive", core, "String", note_size=9)

    manifest_path = vl.pad(canvas, "60,110,160,15", "manifest.json", "IO",
                           basics, "Path", "manifest")
    reader_pins = [hidden_context, ("File Path", "InputPin", False),
                   ("Encoding", "InputPin", False),
                   ("Read", "InputPin", False, "True",
                    ("Primitive", basics, "Boolean")),
                   ("Output", "OutputPin", False)]
    manifest = vl.node(canvas, "60,180,110,19", "IO", core, flag,
                       "FileReader (String)", reader_pins)
    vl.link(manifest_path, manifest["File Path"])
    changed = vl.node(canvas, "60,220,65,19", "Control", core, flag,
                      "Changed", [
                          hidden_context,
                          ("Changed On Create", "InputPin", True),
                          ("Value", "InputPin", False),
                          ("Result", "OutputPin", False),
                          ("Unchanged", "OutputPin", False)])
    vl.link(manifest["Output"], changed["Value"])

    image_pins = [hidden_context, ("Filename", "InputPin", False),
                  ("Load", "InputPin", False), ("Output", "OutputPin", False)]
    draw_pins = [hidden_context, ("Image", "InputPin", False),
                 ("Position", "InputPin", False), ("Size", "InputPin", False),
                 ("Size Mode", "InputPin", False),
                 ("Anchor", "InputPin", False), ("Paint", "InputPin", False),
                 ("Enabled", "InputPin", False), ("Output", "OutputPin", False),
                 ("Actual Bounds", "OutputPin", False)]
    origin = vl.pad(canvas, "300,300,60,22", vec(0, 0), "2D", basics,
                    "Vector2", "Position")
    extent = vl.pad(canvas, "370,300,80,22", vec(width, height), "2D",
                    basics, "Vector2", "Size")
    size_mode = vl.pad(canvas, "460,300,84,15", "Size", "Graphics.Skia",
                       skia, "SizeMode", "Size Mode")
    top_left = vl.pad(canvas, "550,300,85,15", "TopLeft", "2D", basics,
                      "RectangleAnchor", "Anchor")
    layers = []
    for index, key in enumerate(("image", "outlines")):
        x = 60 + 140 * index
        path = vl.pad(canvas, f"{x},260,150,15", names[key], "IO", basics,
                      "Path", key)
        reader = vl.node(canvas, f"{x},290,85,19", "Graphics.Skia.IO", skia,
                         flag, "ImageReader", image_pins)
        vl.link(path, reader["Filename"])
        vl.link(changed["Result"], reader["Load"])
        draw = vl.node(canvas, f"{x},340,125,19", "Graphics.Skia.Layers",
                       skia, flag, "DrawImage", draw_pins, position_pin=True)
        vl.link(reader["Output"], draw["Image"])
        vl.link(origin, draw["Position"])
        vl.link(extent, draw["Size"])
        vl.link(size_mode, draw["Size Mode"])
        vl.link(top_left, draw["Anchor"])
        layers.append(draw["Output"])

    table_path = vl.pad(canvas, "660,110,170,15", names["objects"], "IO",
                        basics, "Path", "objects")
    table = vl.node(canvas, "660,180,110,19", "IO", core, flag,
                    "FileReader (String)", reader_pins)
    vl.link(table_path, table["File Path"])
    lines = vl.node(canvas, "660,220,75,19", "Primitive.String", core, call,
                    "SplitToLines", [("Input", "StateInputPin", False),
                                     ("Remove Empty Lines", "InputPin", False),
                                     ("Output", "StateOutputPin", False)])
    vl.link(table["Output"], lines["Input"])
    no_empty = vl.pad(canvas, "760,200,35,15", "True", "Primitive", core,
                      "Boolean", "Remove Empty Lines")
    vl.link(no_empty, lines["Remove Empty Lines"])
    skip = vl.node(canvas, "660,250,45,19", "Collections.Spread", core, call,
                   "Skip", [("Input", "StateInputPin", False),
                            ("Count", "InputPin", False),
                            ("Output", "StateOutputPin", False)])
    vl.link(lines["Output"], skip["Input"])
    header = vl.pad(canvas, "720,240,35,15", "1", "Primitive", core,
                    "Integer32", "Count")
    vl.link(header, skip["Count"])

    region = vl.sub(canvas, "Node", Bounds="650,290,600,370", Id=vl.new_id())
    ref = vl.sub(region, "{property}NodeReference",
                 LastCategoryFullName="Primitive", LastDependency="Builtin")
    vl.sub(ref, "Choice", Kind="StatefulRegion", Name="Region (Stateful)",
           Fixed="true")
    vl.sub(ref, "Choice", Kind="ApplicationStatefulRegion", Name="ForEach")
    vl.sub(ref, "CategoryReference", Kind="Category", Name="Primitive")
    vl.sub(region, "Pin", Id=vl.new_id(), Name="Break", Kind="OutputPin")
    row_in = vl.new_id()
    vl.sub(region, "ControlPoint", Id=row_in, Bounds="664,296",
           Alignment="Top")
    row_out = vl.new_id()
    vl.sub(region, "ControlPoint", Id=row_out, Bounds="664,654",
           Alignment="Bottom")
    body = vl.sub(region, "Patch", Id=vl.new_id(), ManuallySortedPins="true")
    for part in ("Create", "Update", "Dispose"):
        vl.sub(body, "Patch", Id=vl.new_id(), Name=part,
               ManuallySortedPins="true")
    vl.link(skip["Output"], row_in)

    split = vl.node(body, "662,330,55,19", "Primitive.String", basics, call,
                    "Split (String)", [("Input", "StateInputPin", False),
                                       ("Separator", "InputPin", False),
                                       ("Options", "InputPin", False),
                                       ("Output", "StateOutputPin", False)])
    vl.link(row_in, split["Input"])
    comma = vl.pad(body, "720,310,35,15", ",", "Primitive", core, "String",
                   "Separator")
    vl.link(comma, split["Separator"])
    slice_pins = [("Input", "StateInputPin", False),
                  ("Default Value", "InputPin", False),
                  ("Index", "InputPin", False), ("Result", "OutputPin", False)]

    def cell(column: str, x: int):
        """Patch the node that reads one column of the current row.

        :param column: the column name.
        :param x: the node's horizontal position.
        :returns: the slice node's result pin.
        """
        index = vl.pad(body, f"{x + 40},360,35,15", str(columns.index(column)),
                       "Primitive", core, "Integer32", "Index")
        node = vl.node(body, f"{x},380,55,19", "Collections.Spread", core,
                       call, "GetSlice", slice_pins)
        vl.link(split["Output"], node["Input"])
        vl.link(index, node["Index"])
        return node["Result"]

    def number(column: str, x: int):
        """Patch the parse of one numeric column of the current row.

        :param column: the column name.
        :param x: the node's horizontal position.
        :returns: the parsed value pin.
        """
        parse = vl.node(body, f"{x},410,55,19", "System.Conversion", core,
                        call, "TryParse", [("String", "InputPin", False),
                                           ("Value", "OutputPin", False),
                                           ("Success", "OutputPin", False)])
        vl.link(cell(column, x), parse["String"])
        return parse["Value"]

    label = cell("label", 662)
    centre_x = number("centroid_x", 760)
    centre_y = number("centroid_y", 860)
    join = vl.node(body, "760,450,85,19", "2D.Vector2", core, call,
                   "Vector (Join)", [("X", "InputPin", False),
                                     ("Y", "InputPin", False),
                                     ("Output", "StateOutputPin", False)])
    vl.link(centre_x, join["X"])
    vl.link(centre_y, join["Y"])

    ring = vl.node(body, "960,440,65,19", "Graphics.Skia.Paint", skia, flag,
                   "Stroke", [hidden_context, ("Input", "InputPin", False),
                              ("Color", "InputPin", False),
                              ("Stroke Width", "InputPin", False),
                              ("Join", "InputPin", False),
                              ("Cap", "InputPin", False),
                              ("Miter", "InputPin", False),
                              ("Output", "OutputPin", False)])
    vl.link(vl.pad(body, "990,400,110,15", "1, 0.85, 0, 1", "Color", core,
                   "RGBA", "Color"), ring["Color"])
    vl.link(vl.pad(body, "1110,400,40,15", num(stroke), "Primitive", core,
                   "Float32", "Stroke Width"), ring["Stroke Width"])
    circle = vl.node(body, "760,500,85,19", "Graphics.Skia.Layers", skia, flag,
                     "Circle", [hidden_context,
                                ("Position", "InputPin", False),
                                ("Radius", "InputPin", False),
                                ("Anchor", "InputPin", False),
                                ("Paint", "InputPin", False),
                                ("Enabled", "InputPin", False),
                                ("Output", "OutputPin", False)])
    vl.link(join["Output"], circle["Position"])
    vl.link(vl.pad(body, "860,480,40,15", num(marker), "Primitive", core,
                   "Float32", "Radius"), circle["Radius"])
    vl.link(ring["Output"], circle["Paint"])

    text_pins = [hidden_context,
                 ("Position", "InputPin", False), ("Size", "InputPin", False),
                 ("Anchor", "InputPin", False), ("Text", "InputPin", False),
                 ("Paint", "InputPin", False), ("Enabled", "InputPin", False),
                 ("Output", "OutputPin", False),
                 ("Baseline Position", "OutputPin", False)]
    font_pins = [hidden_context, ("Input", "InputPin", False),
                 ("Family Name", "InputPin", False),
                 ("Style", "InputPin", False), ("Size", "InputPin", False),
                 ("Color", "InputPin", False),
                 ("Line Height in em", "InputPin", False),
                 ("Horizontal Alignment", "InputPin", False),
                 ("Vertical Alignment", "InputPin", False),
                 ("Show Helpers", "InputPin", False),
                 ("Output", "OutputPin", False)]
    enums = "Graphics.Skia.Unwrapped.Enums"
    font_size = vl.pad(body, "960,480,40,15", num(font), "Primitive", core,
                       "Float32", "Size")
    ink = vl.pad(body, "1010,480,110,15", "1, 0.85, 0, 1", "Color", core,
                 "RGBA", "Color")
    left = vl.pad(body, "1130,480,60,15", "Left", enums, skia, "SKTextAlign",
                  "Horizontal Alignment")
    no_box = vl.pad(body, "900,560,60,22", vec(0, 0), "2D", basics,
                    "Vector2", "Size")
    group_inputs = [circle["Output"]]
    for index, (column, vertical) in enumerate((("label", "Bottom"),
                                                ("class", "Top"))):
        if column not in columns:
            continue
        x = 900 + 160 * index
        font_node = vl.node(body, f"{x},530,125,19", "Graphics.Skia.Text",
                            skia, flag, "FontAndParagraph", font_pins)
        vl.link(font_size, font_node["Size"])
        vl.link(ink, font_node["Color"])
        vl.link(left, font_node["Horizontal Alignment"])
        vl.link(vl.pad(body, f"{x + 60},510,62,15", vertical, enums, skia,
                       "VerticalTextAlignment", "Vertical Alignment"),
                font_node["Vertical Alignment"])
        text = vl.node(body, f"{x},580,85,19", "Graphics.Skia.Layers.Text",
                       skia, flag, "Text", text_pins, position_pin=True)
        vl.link(join["Output"], text["Position"])
        vl.link(no_box, text["Size"])
        vl.link(label if column == "label" else cell(column, 1110),
                text["Text"])
        vl.link(font_node["Output"], text["Paint"])
        group_inputs.append(text["Output"])
    group_pins = [hidden_context] + [
        ("Input" if i == 0 else f"Input {i + 1}", "InputPin", False)
        for i in range(len(group_inputs))] + [
        ("Debug", "InputPin", False), ("Enabled", "InputPin", False),
        ("Output", "OutputPin", False)]
    marker_group = vl.node(body, "760,620,85,19", "Graphics.Skia", skia, flag,
                           "Group", group_pins)
    for i, source in enumerate(group_inputs):
        vl.link(source, marker_group["Input" if i == 0 else f"Input {i + 1}"])
    vl.link(marker_group["Output"], row_out)

    markers = vl.node(canvas, "650,680,105,19", "Graphics.Skia", skia, flag,
                      "Group (Spectral)", [hidden_context,
                                           ("Input", "InputPin", False),
                                           ("Debug", "InputPin", False),
                                           ("Enabled", "InputPin", False),
                                           ("Output", "OutputPin", False)])
    vl.link(row_out, markers["Input"])
    layers.append(markers["Output"])
    scene = vl.node(canvas, "60,680,145,19", "Graphics.Skia", skia, flag,
                    "Group", [hidden_context] + [
                        ("Input" if i == 0 else f"Input {i + 1}", "InputPin",
                         False) for i in range(len(layers))] + [
                        ("Debug", "InputPin", False),
                        ("Enabled", "InputPin", False),
                        ("Output", "OutputPin", False)])
    for i, source in enumerate(layers):
        vl.link(source, scene["Input" if i == 0 else f"Input {i + 1}"])
    fit = vl.node(canvas, "60,730,105,19", "Graphics.Skia.Transform", skia,
                  flag, "TransformSRT", [hidden_context,
                                         ("Input", "InputPin", False),
                                         ("Scaling", "InputPin", False),
                                         ("Rotation", "InputPin", False),
                                         ("Translation", "InputPin", False),
                                         ("Output", "OutputPin", False)])
    vl.link(scene["Output"], fit["Input"])
    vl.link(vl.pad(canvas, "120,710,110,22", vec(scale, scale), "2D", basics,
                   "Vector2", "Scaling"), fit["Scaling"])
    vl.link(vl.pad(canvas, "240,710,110,22", vec(-width / height, -1.0), "2D",
                   basics, "Vector2", "Translation"), fit["Translation"])

    window = max(320, min(900, height))
    window_width = max(320, min(1400, int(round(window * width / height))))
    bounds = f"100, 100, {window_width}, {window}"
    renderer = vl.node(canvas, "60,790,185,19", "Graphics.Skia", skia, flag,
                       "Renderer", [
                           hidden_context,
                           ("Bounds", "InputPin", False, bounds,
                            ("System.Drawing", "System.Drawing.dll",
                             "Rectangle")),
                           ("Save Bounds", "InputPin", True),
                           ("Bound to Document", "InputPin", False, "True",
                            ("Primitive", basics, "Boolean")),
                           ("Dialog If Document Changed", "InputPin", True),
                           ("Always On Top", "InputPin", True),
                           ("Extend Into Title Bar", "InputPin", True),
                           ("Input", "InputPin", False),
                           ("Title", "InputPin", False),
                           ("Title Bar Interaction Width", "InputPin", True),
                           ("Color", "InputPin", False),
                           ("Clear", "InputPin", False),
                           ("Space", "InputPin", False),
                           ("Show Cursor", "InputPin", False),
                           ("VSync", "InputPin", False),
                           ("Commands", "InputPin", True),
                           ("Enable Keyboard Shortcuts", "InputPin", True),
                           ("Enabled", "InputPin", False),
                           ("Form Bounds Notifications", "OutputPin", True),
                           ("Form", "OutputPin", False),
                           ("ClientBounds", "OutputPin", False),
                           ("Render Time", "OutputPin", False)])
    vl.link(fit["Output"], renderer["Input"])
    vl.link(vl.pad(canvas, "250,770,160,15", f"spaCR - {stem}", "Primitive",
                   core, "String", "Title"), renderer["Title"])
    vl.link(vl.pad(canvas, "420,770,110,15", "0.1, 0.1, 0.1, 1", "Color",
                   core, "RGBA", "Color"), renderer["Color"])
    vl.link(vl.pad(canvas, "540,770,90,15", "Normalized", "VL.Skia",
                   "VL.Skia.dll", "CommonSpace", "Space"), renderer["Space"])

    create = vl.new_id()
    update = vl.new_id()
    vl.sub(patch, "Patch", Id=create, Name="Create")
    vl.sub(patch, "Patch", Id=update, Name="Update")
    process = vl.sub(patch, "ProcessDefinition", Id=vl.new_id())
    vl.sub(process, "Fragment", Id=vl.new_id(), Patch=create, Enabled="true")
    vl.sub(process, "Fragment", Id=vl.new_id(), Patch=update, Enabled="true")
    for source, sink in vl.links:
        vl.sub(patch, "Link", Id=vl.new_id(), Ids=f"{source},{sink}")
    et.indent(doc, space="  ")
    text = et.tostring(doc, encoding="unicode")
    return "﻿<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" + text + "\n"


def _vvvv_publisher(tabular):
    """``tabular._publish`` that waits out a reader holding the target.

    On Windows a file vvvv is reading cannot be replaced for that moment;
    the rename is retried for about a second before the error is raised.
    """
    import time

    def publish(target, write):
        """Publish ``target``, retrying while another process holds it.

        :param target: the destination path.
        :param write: writes the temporary file that replaces it.
        :returns: what the publish returns.
        """
        for attempt in range(20):
            try:
                return tabular._publish(target, write)
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
        return target

    return publish


def _vvvv_readme(stem: str, project: str, origin: str = "Make Masks") -> str:
    """How to open an export in vvvv gamma, for its README.txt.

    :param stem: the field's stem.
    :param project: the ``.vl`` file name.
    :param origin: the spaCR module that wrote the export.
    """
    extra = ("" if origin == "Make Masks" else
             ",\n                     then the module's own measurements")
    return f"""spaCR export for vvvv gamma (alpha)
====================================

This folder holds one image exported from spaCR {origin} and a vvvv
gamma project that shows it.

Open it
-------
1. Install vvvv gamma {_VVVV_GAMMA_VERSION} or later for Windows from
   https://vvvv.org/download (free for non-commercial use).
2. Start vvvv gamma, choose File > Open and select {project}
   in this folder (or drag the file onto the vvvv window).
3. A window titled "spaCR - {stem}" opens. It shows the image with white
   object outlines on top and, for every object, a yellow circle at its
   centroid with the object's label number next to it (and its class,
   when the table has one).
4. The patch itself opens in the vvvv editor; a note at its top says
   what it does, and every value (colours, marker size) can be edited.

Live updates
------------
With "Export on save" switched on in spaCR, every mask save rewrites this
folder. The project reads manifest.json and the objects table every frame
and reloads the images when the manifest changes, so the window follows
your edits.

Files
-----
{stem}_image.png     the image as 8-bit RGB (display only)
{stem}_labels.png    16-bit labels; 0 is background, ids match the mask
{stem}_outlines.png  white object outlines on a transparent background
{stem}_objects.csv   one row per object: label, centroid_x, centroid_y,
                     area, bounding box, mean intensity per channel{extra}
manifest.json        image size, pixel size, channels and file names;
                     written last, so its change means the set is complete
{project}            the vvvv gamma project

Coordinates are pixels with the origin at the image's top-left corner,
x to the right and y down; centroids are pixel centres.
"""


def _export_vvvv(root: str, image_name: str, labels: np.ndarray,
                image: Optional[np.ndarray] = None, *,
                source_path: str = "", pixel_size=None, unit: str = "µm",
                channel_names=None, classes=None, scores=None,
                table=None, origin: str = "Make Masks") -> str:
    """Write one field for a vvvv patch watching ``root`` (item 685, alpha).

    The folder ``<root>/<stem>/`` holds ``<stem>_labels.png`` (16-bit
    labels), ``<stem>_outlines.png`` (RGBA outlines on transparency),
    ``<stem>_image.png`` (the field as 8-bit RGB, when ``image`` is given;
    :func:`_vvvv_display_image`), ``<stem>_objects.csv``
    (:func:`_vvvv_object_table`), ``<stem>.vl`` (a vvvv gamma project that
    shows them; :func:`_vvvv_project_document`), ``README.txt`` and
    ``manifest.json``. Every file is written to a hidden sibling and
    renamed into place, so a watcher never reads half a file; the manifest
    goes last, so its change means the set is complete. The project and
    README are rewritten only when their text changes, so vvvv does not
    reload an open project after every save. The source image and mask
    are only read.

    :param root: export folder; created when missing.
    :param image_name: the field's file name; its stem names everything.
    :param labels: 2-D label image; ids above 65535 are refused.
    :param image: the field as loaded, for intensities and channel count.
    :param source_path: the field's path, recorded in the manifest.
    :param pixel_size: ``(x, y)`` physical size of a pixel, or None.
    :param unit: the unit of ``pixel_size``.
    :param channel_names: names for the image's channels.
    :param classes: optional ``{label: class name}``.
    :param scores: optional ``{label: score}``.
    :param table: optional per-object frame with a ``label`` column, such
        as Measure's measurements; its other columns are appended to the
        objects table by label, after the columns listed above, and a
        column that repeats one of those is left out.
    :param origin: the spaCR module named in README.txt.
    :returns: the field's export folder.
    :raises ValueError: for labels that are not 2-D or exceed 16 bits.
    """
    import datetime
    import time

    from .. import tabular

    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("vvvv export needs a 2-D label image")
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) > 65535):
        raise ValueError("vvvv export stores labels as 16-bit PNG; "
                         "ids must lie between 0 and 65535")
    stem = field_stem(os.path.basename(image_name))
    folder = os.path.join(os.fspath(root), stem)
    os.makedirs(folder, exist_ok=True)
    frame = _vvvv_object_table(labels, image, channel_names, classes, scores)
    if table is not None and "label" in getattr(table, "columns", ()):
        table = table.loc[:, ~table.columns.duplicated()]
        extra = table.loc[:, [column for column in table.columns
                              if column == "label"
                              or column not in frame.columns]]
        extra = extra.drop_duplicates("label").astype({"label": np.int64})
        frame = frame.astype({"label": np.int64}).merge(
            extra, on="label", how="left")
    names = {
        "labels": f"{stem}_labels.png",
        "outlines": f"{stem}_outlines.png",
        "objects": f"{stem}_objects.csv",
    }
    publish = _vvvv_publisher(tabular)
    publish(os.path.join(folder, names["labels"]),
            lambda pending: imageio.imwrite(
                pending, labels.astype(np.uint16), format="PNG"))
    publish(os.path.join(folder, names["outlines"]),
            lambda pending: imageio.imwrite(
                pending, _vvvv_outlines(labels), format="PNG"))
    image_name_png = f"{stem}_image.png"
    if image is not None:
        publish(os.path.join(folder, image_name_png),
                lambda pending: imageio.imwrite(
                    pending, _vvvv_display_image(image), format="PNG"))
        names["image"] = image_name_png
    publish(os.path.join(folder, names["objects"]),
            lambda pending: tabular.write_table(
                frame, pending, canonicalise=False))
    project = _vvvv_project_document(
        stem, int(labels.shape[1]), int(labels.shape[0]),
        dict(names, image=image_name_png), list(frame.columns))
    texts = {f"{stem}.vl": project,
             "README.txt": _vvvv_readme(stem, f"{stem}.vl", origin)}
    for file_name, text in texts.items():
        target = os.path.join(folder, file_name)
        try:
            unchanged = Path(target).read_text(encoding="utf-8") == text
        except (OSError, UnicodeDecodeError):
            unchanged = False
        if not unchanged:
            publish(target, lambda pending, text=text: Path(pending)
                    .write_text(text, encoding="utf-8", newline="\r\n"))
    names["project"] = f"{stem}.vl"
    names["readme"] = "README.txt"
    channels = [column[len("mean_intensity_"):] for column in frame.columns
                if column.startswith("mean_intensity_")]
    try:
        from .. import __version__ as version
    except ImportError:
        version = ""
    spacing = None
    if pixel_size is not None:
        spacing = {"x": float(pixel_size[0]), "y": float(pixel_size[1]),
                   "unit": str(unit)}
    manifest = {
        "format": VVVV_EXPORT_FORMAT,
        "version": VVVV_EXPORT_VERSION,
        "image": os.path.basename(image_name),
        "source_path": os.fspath(source_path),
        "spacr_version": str(version),
        "exported_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        "width": int(labels.shape[1]),
        "height": int(labels.shape[0]),
        "pixel_size": spacing,
        "channel_names": channels,
        "objects": int(len(frame)),
        "coordinates": ("pixels; origin at the top-left corner of the image, "
                        "x right, y down; centroids are pixel centres, "
                        "bounding boxes are half-open [x0, x1)"),
        "export_id": time.time_ns(),
        "vvvv_gamma": _VVVV_GAMMA_VERSION,
        "files": names,
    }
    publish(os.path.join(folder, "manifest.json"),
            lambda pending: Path(pending).write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8"))
    return folder


def record_curation(folder: str, image_path: str, mask_path: str,
                    object_count: int, keep: bool) -> str:
    """Record one verdict, replacing any the field already had.

    ONE ROW PER FIELD. Keep and then Discard on the same image leaves the
    later verdict and nothing else, because two rows that disagree are
    worse than no file -- whoever reads it downstream would have to guess
    which press came last, and a CSV does not say.

    Written to a dot-name in the same folder and renamed over the target,
    so a reader never sees half a file and a crash leaves the old one
    whole. The same shape as :func:`_write_bundle` above.

    :param folder: the folder the editor opened.
    :param image_path: the field being judged.
    :param mask_path: its mask, from :func:`mask_save_path`.
    :param object_count: how many objects the mask holds right now.
    :param keep: True for Keep, False for Discard.
    :returns: the CSV's path.
    """
    rows = read_curation(folder)
    rows[os.fspath(image_path)] = {
        CURATION_COLUMNS[0]: os.fspath(image_path),
        CURATION_COLUMNS[1]: os.fspath(mask_path),
        CURATION_COLUMNS[2]: str(int(object_count)),
        CURATION_COLUMNS[3]: "true" if keep else "false",
    }
    destination = curation_csv_path(folder)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    directory, name = os.path.split(destination)
    temporary = os.path.join(directory, f".{name}.tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CURATION_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({column: rows[key].get(column, "")
                             for column in CURATION_COLUMNS})
    os.replace(temporary, destination)
    return destination


def canonical_labels(mask: np.ndarray, *, preserve_ids: bool = False) -> np.ndarray:
    """Return validated labels, retaining explicit uint32 identities.

    The old behaviour here was ``label(mask > 0)``, which renumbers the
    connected components 1..N on every save. That throws away the identity
    of every object: erase object 7 of 20 and objects 8..20 each slide down
    by one, so the saved mask no longer keys against the measurements, the
    crops or the tracks derived from the segmentation it was edited from.

    So ids are kept, with two things settled:

    * **A mask with one foreground value carries no ids to keep.** That is
      what a purely brush-painted mask looks like -- every stroke writes the
      same value -- and it is a binary image, not a label image. Its
      components are numbered 1..N, which loses nothing.
    * **A label that names two separated blobs is split.** One id must mean
      one object. The largest piece keeps the id and the rest are given the
      smallest ids not already in use, so painting a second blob with the
      brush over a real segmentation adds an object instead of extending a
      distant one.

    Each id is examined inside its own bounding box
    (:func:`scipy.ndimage.find_objects`) rather than across the whole field,
    which gives the same pieces in the same order and makes the call cheap
    enough to run while the mouse moves: on a 2048 x 2048 field of 400
    objects it went from about 3.5 s to tens of milliseconds.

    :param mask: a label image; any integer or boolean dtype in default mode.
    :param preserve_ids: disable binary interpretation and component splitting.
        Use for primary/secondary relationships: a lone cell 900 remains 900,
        and separated pieces with the same primary ID remain one label.
        Requires a nonempty 2-D nonnegative integer array, not a boolean mask.
    :returns: uint16 labels for ordinary segmentation, or uint32 for explicit
        identities requiring that width. Existing uint32 masks retain their
        exact IDs even without the explicit flag.
    :raises ValueError: when ordinary segmentation exceeds uint16 capacity,
        exact IDs exceed uint32 capacity, or exact-ID mode receives invalid
        labels. Oversized IDs are never truncated.
    """
    m = np.asarray(mask)
    if preserve_ids or m.dtype == np.uint32:
        m = _primary_label_image(m, "Mask")
        if int(m.max()) > MAX_MASK_LABEL:
            from .i18n import tr

            raise ValueError(tr("Exact mask IDs must fit in uint16 or uint32; labels were not renumbered."))
        dtype = np.uint32 if int(m.max()) > 65535 or m.dtype == np.uint32 else np.uint16
        return m.astype(dtype, copy=True)
    boxes = None
    top = int(m.max()) if m.size and np.issubdtype(m.dtype, np.integer) \
        else None
    if top is not None and top <= np.iinfo(np.uint16).max:
        boxes = _ndimage().find_objects(m) if top > 0 else []
        values = [index + 1 for index, box in enumerate(boxes)
                  if box is not None]
    else:
        values = list(np.unique(m[m > 0]))
    if len(values) <= 1:
        labeled, count = _ndimage().label(m > 0, structure=_EIGHT)
        if count > np.iinfo(np.uint16).max:
            raise ValueError(
                f"mask needs {count} object labels, past what a uint16 mask can hold.")
        return labeled.astype(np.uint16)

    largest = int(max(values))
    if largest > np.iinfo(np.uint16).max:
        raise ValueError(
            f"mask carries label {largest}, past what a uint16 mask can hold.")
    whole = tuple(slice(None) for _axis in range(m.ndim))
    out = None
    used = {int(v) for v in values}
    candidate = 1
    for value in values:
        box = boxes[int(value) - 1] if boxes is not None else whole
        pieces, count = _ndimage().label(m[box] == value, structure=_EIGHT)
        if count <= 1:
            continue
        areas = np.bincount(pieces.ravel())
        keep = int(np.argmax(areas[1:])) + 1
        if out is None:
            out = m.astype(np.int64, copy=True)
        region = out[box]
        for piece in range(1, count + 1):
            if piece == keep:
                continue
            while candidate in used:
                candidate += 1
            region[pieces == piece] = candidate
            used.add(candidate)
    if out is None:
        out = m if boxes is not None else m.astype(np.int64)
    top = int(out.max()) if out.size else 0
    if top > np.iinfo(np.uint16).max:
        raise ValueError(
            f"mask carries label {top}, past what a uint16 mask can hold.")
    return out.astype(np.uint16)


_MASK_SAVE_WAIT_SECONDS: Optional[float] = 0.1
"""How long Make Masks waits on the GUI thread for a mask save to land.

A save written in this time is reported, and its follow-up steps run, as
before; a slower one (3.1 s on a busy disk in lesson 14) finishes on the mask
writer thread and reports when it lands. ``None`` waits for every save.
"""


def save_mask(folder: str, filename: str, mask: np.ndarray,
              log: Optional[CurationLog] = None,
              masks_dir: Optional[str] = None, *, preserve_ids: bool = False,
              lineage: Optional[dict] = None) -> str:
    """Write the mask to ``<folder>/masks/<stem>.tif`` and return that path.

    Object ids are preserved -- see :func:`canonical_labels` for what that
    costs and why the alternative is worse. A ``_seg.npy`` bundle is written
    back into itself by :func:`save_seg_bundle`, and ``masks_dir`` moves the
    TIFF to the sibling layout's masks folder.

    :param folder: field directory under which the ``masks`` directory is
        created.
    :param filename: source image name; its extension is discarded and its
        stem becomes the TIFF mask name. A name ending in
        :data:`SEG_SUFFIX` is the bundle to write into.
    :param mask: label image to canonicalise and write. Existing multi-label
        object identifiers are retained where possible.
    :param log: the session's :class:`spacr.curation.CurationLog` for this
        field. Given one holding at least one edit, it is written to
        ``<mask>.curation.json`` beside the mask, so
        :func:`spacr.curation.is_curated` reports the saved mask as
        hand-edited. The log is written whole, so it must have been seeded
        from :meth:`CurationLog.read_beside` to keep earlier sessions'
        entries -- which is what the screen does on load. A log with no
        edits writes no sidecar: a session that opened the editor and
        painted nothing has not curated anything, and a ledger that exists
        for every mask ever opened answers no question.
    :param masks_dir: where the masks are, when not in ``<folder>/masks``.
    :param preserve_ids: retain exact primary/secondary IDs without interpreting
        single-valued masks as binary or splitting disconnected pieces. The
        explicit mode validates nonnegative 2-D integer labels and refuses
        IDs above 4294967295 before writing; it never silently renumbers them.
    :param lineage: versioned UUID graph and per-frame pixel mappings to embed
        in the TIFF. Even an empty graph requests uint32 storage. Existing
        embedded lineage and exact-ID semantics survive ordinary later saves.
    :returns: the path written. The TIFF carries the source image's pixel
        size (:func:`pixel_size_nm`) as its resolution, in pixels per
        centimetre, when the source records one.
    """
    save_path = mask_save_path(folder, filename, masks_dir)
    metadata = load_mask_metadata(folder, filename, masks_dir)
    if lineage is None:
        lineage = metadata.get("lineage")
    preserve_ids = bool(preserve_ids or metadata.get("preserve_ids")
                        or lineage is not None or np.asarray(mask).dtype == np.uint32)
    labels = canonical_labels(mask, preserve_ids=preserve_ids)
    if lineage is not None:
        from .i18n import tr

        if is_seg_bundle(filename):
            raise ValueError(tr("Time labels require TIFF masks with embedded lineage."))
        frame = os.path.abspath(os.path.join(folder, filename))
        lineage = prune_lineage_frame(lineage, frame, labels)
        labels = labels.astype(np.uint32, copy=False)
        metadata["lineage"] = lineage
        metadata["frame"] = frame
        metadata["frame_id"] = lineage["frame_ids"][frame]
    metadata.update(schema="spacr.mask/v1", preserve_ids=preserve_ids)
    if is_seg_bundle(filename):
        save_seg_bundle(save_path, labels, preserve_ids=preserve_ids)
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        size = pixel_size_nm(os.path.join(folder, filename))
        resolution = _resolution_for(size)
        description_metadata = resolution.pop("metadata", {})
        description_metadata.update(shape=labels.shape, spacr_mask=metadata)
        description = json.dumps(description_metadata, ensure_ascii=True, allow_nan=False)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".spacr-mask-", suffix=".tif", dir=os.path.dirname(save_path))
        os.close(descriptor)
        try:
            write_tiff(temporary, labels, metadata=None, description=description,
                       bigtiff=labels.nbytes + len(description) >= 2**32 - 2**25,
                       **resolution)
            with open(temporary, "r+b") as handle:
                os.fsync(handle.fileno())
            if os.path.isfile(save_path):
                os.chmod(temporary, stat.S_IMODE(os.stat(save_path).st_mode))
            os.replace(temporary, save_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if log is not None and len(log):
        if not log.artifact:
            log.artifact = save_path
        log.write_beside(save_path)
    return save_path


def _percentile_bounds(image: np.ndarray, lower_pct: float = 1.0,
                      upper_pct: float = 99.9) -> Tuple[float, float]:
    """The two intensities :func:`normalize_uint16` maps to black and white.

    :param image: a non-empty image.
    :param lower_pct: the percentile mapped to black.
    :param upper_pct: the percentile mapped to white.
    :returns: ``(lo, hi)``, with ``hi`` raised to ``lo + 1`` when the two
        percentiles meet, so a flat field does not divide by zero.
    """
    lo = np.percentile(image, lower_pct)
    hi = np.percentile(image, upper_pct)
    if hi <= lo:
        hi = lo + 1
    return lo, hi


def _stretch_between(image: np.ndarray, lo, hi) -> np.ndarray:
    """``image`` clipped to ``lo..hi`` and rescaled to its dtype's range.

    Pixel by pixel, so stretching a crop gives the same pixels as cropping
    the stretched field: the canvas stretches only what it draws.

    :param image: integer-typed image.
    :param lo: the intensity mapped to 0.
    :param hi: the intensity mapped to the dtype's maximum; above ``lo``.
    An 8- or 16-bit field larger than its value range is stretched through
    a table of every value, built by the same formula, so each pixel is the
    same byte: one look-up instead of four float passes, about a third of
    the time on a 2,000 px field the canvas redraws after every edit.

    :returns: the stretched image, in ``image``'s dtype.
    """
    if (image.dtype in (np.uint8, np.uint16)
            and image.size > 4 * (np.iinfo(image.dtype).max + 1)):
        values = np.arange(np.iinfo(image.dtype).max + 1, dtype=image.dtype)
        return _stretch_between(values, lo, hi)[image]
    out = np.clip(image, lo, hi)
    out = (out - lo) / (hi - lo)
    max_val = float(np.iinfo(image.dtype).max)
    return (out * max_val).astype(image.dtype)


def _display_step(width: int, height: int, target_width: float,
                 target_height: float) -> int:
    """The pixel stride at which a ``width`` x ``height`` view is composed.

    The canvas scales what it composes down to the widget anyway, so a
    region larger than the widget's device pixels is composed from every
    ``step``-th pixel instead: the largest whole stride that still leaves at
    least the widget's resolution on both axes. Measured on a 4,096 px TEM
    field: composing all of it took about 4 s per refresh, and a refresh
    follows every brush point; at the stride a 1,000 px canvas needs (4) it
    is 16 times fewer pixels.

    :param width: the viewed region's width, in image pixels.
    :param height: its height.
    :param target_width: the widget's width, in device pixels.
    :param target_height: its height.
    :returns: 1 or more.
    """
    return max(1, int(min(width / max(1.0, float(target_width)),
                          height / max(1.0, float(target_height)))))


def normalize_uint16(image: np.ndarray,
                     lower_pct: float = 1.0,
                     upper_pct: float = 99.9) -> np.ndarray:
    """Return image clipped + rescaled to its dtype's full range.

    :param image: integer-typed image (any integer dtype, despite the name);
        clipped to its ``lower_pct`` and ``upper_pct`` percentiles. An empty
        array is returned as is.
    """
    if not image.size:
        return image
    lo, hi = _percentile_bounds(image, lower_pct, upper_pct)
    return _stretch_between(image, lo, hi)


def normalize_for_detection(image: np.ndarray, lower_pct: float = 1.0,
                            upper_pct: float = 99.9) -> np.ndarray:
    """``image`` stretched between two percentiles, as Make Masks draws it.

    :func:`normalize_uint16` for an integer field, so a detector reads the
    exact numbers the canvas paints; a float field has no integer range to
    fill and comes back on 0..1 instead.

    :param image: the field.
    :param lower_pct: the percentile mapped to the bottom of the range.
    :param upper_pct: the percentile mapped to the top.
    :returns: the stretched field, the same shape.
    """
    if np.issubdtype(image.dtype, np.integer):
        return normalize_uint16(image, lower_pct, upper_pct)
    if not image.size:
        return image
    lo = float(np.percentile(image, lower_pct))
    hi = float(np.percentile(image, upper_pct))
    if hi <= lo:
        hi = lo + 1.0
    return ((np.clip(image, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def invert_normalized(image: np.ndarray) -> np.ndarray:
    """Normalise ``image`` to 0..1 on its OWN range, take ``1 - v``, fit back.

    THE ONE INVERSION. The field is normalised to 0..1 and every pixel
    becomes ``1 - value``, which gives an image that Otsu and the magnifier
    can work on when the objects are dark. That is the reason for the
    button.

    So the purpose is a DETECTOR reading dark objects, and the picture the
    curator sees has to be the picture the detector reads -- one switch, one
    meaning. That decision replaced the two inversions this module used to
    carry for Make Masks, :func:`invert_intensity` (the dtype complement,
    drawn but never detected on) and :func:`invert_for_detection`
    (detected on but never drawn). Both are kept for
    callers outside Make Masks and neither is what the screen uses now.

    WHY NORMALISING FIRST IS THE POINT AND NOT A DETAIL. The Otsu threshold
    correction is a MULTIPLIER on an absolute level, so what it means depends
    on where the field's intensities sit. A dtype complement moves a 12-bit
    field (216..4095) up into 61440..65535, and a correction of 0.8 then asks
    for a cut below every pixel present while 1.3 asks for one above them all
    -- the dial becomes an on/off switch, which is the measurement recorded in
    :func:`invert_for_detection`. Normalising to the field's own range first
    puts EVERY field on the same 0..1 span before the multiplier is applied,
    so one correction value means the same thing on the next image.

    WHAT IS GIVEN UP, said plainly: this is not exactly reversible on the
    original numbers the way the dtype complement was. Inverting a field
    rescales it to the full range, and the original span cannot be recovered
    from the result. It does not need to be -- the screen keeps the untouched
    array and re-derives this one, so nothing measured, filtered or saved ever
    sees it -- but a caller that inverts an array and keeps only the result
    has lost where it sat.

    THE RETURN DTYPE IS THE INPUT'S, because the display path hands the result
    to :func:`normalize_uint16`, which reads ``np.iinfo`` and raises on a
    float. An integer field comes back spanning that dtype's full range; a
    float field comes back in 0..1, where a float already belongs.

    A FLAT FIELD has no range to normalise onto. ``v - min`` is 0 everywhere,
    so the normalised value is taken as 0 and the inverse as 1: a flat field
    inverts to a flat bright one, which is the literal reading of the formula
    and is what an inversion of "no contrast" should look like.

    :param image: the field, of any shape and any real dtype.
    :returns: a new array, same shape, same dtype, inverted.
    """
    array = np.asarray(image)
    if not array.size:
        return array.copy()
    if array.dtype == np.bool_:
        return ~array
    low = float(array.min())
    high = float(array.max())
    span = high - low
    unit = (np.zeros(array.shape, np.float64) if span <= 0
            else (array.astype(np.float64) - low) / span)
    flipped = 1.0 - unit
    if np.issubdtype(array.dtype, np.floating):
        return flipped.astype(array.dtype)
    info = np.iinfo(array.dtype)
    return (info.min + flipped * (float(info.max) - float(info.min))
            ).astype(array.dtype)


def invert_intensity(image: np.ndarray) -> np.ndarray:
    """Return the photographic complement of ``image``: dark becomes bright.

    Low intensity becomes high intensity and vice versa, fitted to the
    dtype. WHAT IS BUILT IS THE COMPLEMENT, ``dtype_max - value``,
    NOT THE RECIPROCAL, for three reasons that are worth having written
    down because the reciprocal is the obvious first thought:

    * it is what every image viewer means by Invert, so the picture that
      comes back is the one a reader expects from Invert;
    * it is EXACTLY reversible on an integer field -- inverting twice
      returns the identical array, which is what makes it safe to leave
      switched on while curating, and is asserted by comparing arrays;
    * ``1/value`` divides by zero on every background pixel, and it squashes
      the bright end non-linearly, so two objects a thousand counts apart
      come back indistinguishable while the background explodes.

    The reciprocal remains a reasonable SECOND mode for anyone who wants a
    log-like lift of the dim end; it is not this one.

    WHICH RANGE IS COMPLEMENTED depends on the dtype, because fitting to
    the dtype only has a meaning where the dtype has ends:

    ``unsigned integers``
        the dtype's own range, so a ``uint16`` field is ``65535 - value``.
        A 12-bit camera writing into ``uint16`` therefore comes back in the
        top sixteenth of the range; a display that stretches by percentiles
        puts that back where a reader can see it, and the array is still
        exactly invertible, which a data-range complement would not be
        across two fields of different brightness.
    ``signed integers``
        ``iinfo.min + iinfo.max - value``, the same complement on the range
        the dtype actually spans.
    ``bool``
        logical not.
    ``floating point``
        the ARRAY'S OWN range, ``min + max - value``, because a float image
        has no dtype maximum worth speaking of. The complement of a range
        maps its ends onto each other, so a second call computes the same
        two ends and comes back to the original -- but NOT bit for bit:
        ``s - (s - x)`` rounds twice, and the round trip is out by up to one
        unit in the last place of ``s``. Measured on a 200x200 field over
        0..65535: 0.002 in ``float32`` and 4e-12 in ``float64``, against an
        interval of one count. The round trip is EXACT for every integer and
        boolean dtype, which is every dtype a field is read in.

    :param image: any numeric or boolean array. It is not modified.
    :returns: a new array of the same shape and dtype.
    """
    values = np.asarray(image)
    if not values.size:
        return values.copy()
    kind = values.dtype.kind
    if kind == "b":
        return np.logical_not(values)
    if kind in "ui":
        info = np.iinfo(values.dtype)
        span = int(info.min) + int(info.max)
        return (span - values.astype(np.int64)).astype(values.dtype)
    if kind in "fc":
        span = values.min() + values.max()
        return (span - values).astype(values.dtype, copy=False)
    raise TypeError(
        f"invert_intensity needs a numeric or boolean image; got dtype "
        f"{values.dtype!r}.")


def invert_for_detection(image: np.ndarray, *, bounds=None) -> np.ndarray:
    """Reflect an image about its OWN range, for a DETECTOR to read.

    Masks are generated from the inverted image so a threshold written for
    bright objects can take dark ones. It is ``max + min - value`` on the
    field's own extremes.

    WHY THIS IS NOT :func:`invert_intensity`, which is the other inversion
    in this module and is one line away. The difference is not taste and it
    is not a duplicate that wants merging -- the two are read by different
    things and only one of them can afford to move the numbers:

    * :func:`invert_intensity` complements the DTYPE and is what "Invert
      image" draws with. It has to be exactly reversible, because a curator
      leaves it on all day, and nothing downstream reads its result.
    * this one is read by a THRESHOLD, and the Otsu threshold correction
      is a MULTIPLIER on the level Otsu finds, applied to absolute intensity
      in :func:`_otsu_levels`. Multiplying is not invariant to an offset, so
      an inversion that moves the field's span moves what the correction
      means. Measured on a 12-bit field (216..4095) with dark objects,
      inverted and put through :func:`_otsu_instances` on the bright side:

      =============  ==========================  =======================
      correction     dtype complement            reflection about range
      =============  ==========================  =======================
      0.8            1 object, 100% of the       47 objects, 20%
                     field -- everything
      1.0            3 objects, 8%               3 objects, 8%
      1.3            0 objects, 0% -- nothing    3 objects, 8%
      =============  ==========================  =======================

      The complement puts that field into 61440..65535, so a correction of
      0.8 asks for a cut at about 49000, below every pixel there is, and 1.3
      asks for one above all of them. The correction stops being a dial and
      becomes an on/off switch. At exactly 1.0 the two agree, which is why
      this is easy to miss.

    The other two candidates were considered and are worse. The DTYPE's
    maximum is the case above. The CONTRAST-STRETCHED view is a viewing
    choice, and a detector reading it would move when the percentiles moved,
    which is the argument :func:`filter_objects` already makes about
    intensity bounds.

    Reflecting about the image's own extremes keeps the span exactly, maps
    the darkest pixel onto the brightest and back, and is its own inverse on
    an image whose extremes it has not changed.

    Nonfinite pixels take no part in finding the extremes and are returned
    unchanged, since a NaN is not dark and is not bright.

    :param image: any 2-D field, as the canvas holds it.
    :param bounds: ``(lo, hi)`` to reflect about, instead of ``image``'s own
        extremes. WHAT A CROP IS GIVEN: a region inverted about its own
        extremes is inverted differently wherever the box is put, so the
        magnifier hands it the whole field's pair and the box stays a
        preview of what the detect button will do with the same setting.
        (:func:`invert_intensity` needs no such thing, being a function of
        the pixel value alone -- another way the two differ.)
    :returns: a NEW array of the input's dtype. The original is never
        touched: the readout and the filter must keep reporting the raw
        values whatever the detector was shown.
    """
    arr = np.asarray(image)
    if not arr.size:
        return arr.copy()
    if np.issubdtype(arr.dtype, np.integer):
        work = arr.astype(np.int64)
        if bounds is None:
            lo, hi = int(work.min()), int(work.max())
        else:
            lo, hi = int(bounds[0]), int(bounds[1])
        return np.clip(hi + lo - work, lo, hi).astype(arr.dtype)
    work = arr.astype(np.float64)
    finite = np.isfinite(work)
    if not finite.any():
        return arr.copy()
    if bounds is None:
        lo = float(work[finite].min())
        hi = float(work[finite].max())
    else:
        lo, hi = float(bounds[0]), float(bounds[1])
    out = work.copy()
    out[finite] = np.clip(hi + lo - work[finite], lo, hi)
    return out.astype(arr.dtype)


_OVERLAY_BLEND_TABLES: dict = {}


def _overlay_blend_table(alpha) -> np.ndarray:
    """``table[grey, colour]``: the byte :func:`overlay_mask` blends the pair to.

    Built by :func:`overlay_mask`'s own expression over every pair of bytes,
    with the same operand types, so a look-up gives exactly its result.

    :param alpha: the colour's weight.
    :returns: a ``(256, 256)`` uint8 table.
    """
    key = (type(alpha), alpha)
    table = _OVERLAY_BLEND_TABLES.get(key)
    if table is None:
        grey = np.arange(256, dtype=np.uint8)[:, None]
        colour = np.arange(256, dtype=np.uint8)[None, :]
        table = np.clip(grey * (1 - alpha) + colour * alpha, 0, 255).astype(np.uint8)
        if len(_OVERLAY_BLEND_TABLES) > 8:
            _OVERLAY_BLEND_TABLES.clear()
        _OVERLAY_BLEND_TABLES[key] = table
    return table


def _overlay_mask_fast(image: np.ndarray, mask: np.ndarray, alpha):
    """:func:`overlay_mask` for a grey 8/16-bit field, or None for any other.

    A grey 8- or 16-bit image under a nonnegative 8-, 16- or 32-bit integer
    mask is blended only where an object is: at ``alpha`` 0.5 as the
    carry-free byte average of packed pixels (the floor of the mean, which
    is what the float formula gives), otherwise through a table of every
    grey and colour byte pair built by the same formula. Either way the
    result is the same byte for byte. The float blend over all three
    channels of every pixel took 184 ms on a 2,000 px field, 41 ms this way.

    :param image: the field.
    :param mask: its labels.
    :param alpha: the colour's weight.
    :returns: the ``(h, w, 3)`` uint8 picture, or None when the general
        path must run.
    """
    image = np.asarray(image)
    mask = np.asarray(mask)
    if (image.ndim != 2 or image.dtype not in (np.uint8, np.uint16)
            or mask.shape != image.shape
            or mask.dtype not in (np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32)):
        return None
    if mask.dtype.kind == "i" and mask.size and int(mask.min()) < 0:
        return None
    max_label = int(np.max(mask)) if mask.size else 0
    if max_label > 65535:
        return None
    flat = np.ascontiguousarray(mask).reshape(-1)
    inside = np.flatnonzero(flat)
    rng = np.random.default_rng(0)
    colors = rng.integers(30, 255, size=(max_label + 1, 3), dtype=np.uint8)
    if max_label >= 0:
        colors[0] = [0, 0, 0]
    if image.dtype == np.uint16:
        grey = (image >> 8).astype(np.uint8)
    else:
        grey = np.zeros(image.shape, dtype=np.uint8)
    if isinstance(alpha, float) and alpha == 0.5:
        packed = grey.astype(np.uint32) * np.uint32(0x010101)
        pixels = packed.reshape(-1)
        if inside.size:
            paint = ((colors[:, 0].astype(np.uint32) << 16)
                     | (colors[:, 1].astype(np.uint32) << 8) | colors[:, 2])
            under = pixels[inside]
            colour = paint[flat[inside]]
            pixels[inside] = (under & colour) + (
                ((under ^ colour) >> 1) & np.uint32(0x7F7F7F))
        out = np.empty(image.shape + (3,), dtype=np.uint8)
        out[..., 0] = packed >> 16
        out[..., 1] = (packed >> 8) & 0xFF
        out[..., 2] = packed & 0xFF
        return out
    out = np.empty(image.shape + (3,), dtype=np.uint8)
    out[...] = grey[..., None]
    if inside.size:
        pixels = out.reshape(-1, 3)
        pixels[inside] = _overlay_blend_table(alpha)[
            pixels[inside], colors[flat[inside]]]
    return out


def _label_colors(values: np.ndarray) -> np.ndarray:
    """Give each original ID a stable color with a bounded legacy lookup table."""
    values = np.asarray(values, dtype=np.uint64)
    flat = values.reshape(-1)
    hashed = flat[:, None] * np.array([2654435761, 2246822519, 3266489917], dtype=np.uint64)
    colors = (30 + ((hashed ^ (hashed >> 16)) % 225)).astype(np.uint8)
    small = flat <= 65535
    if small.any():
        legacy = np.random.default_rng(0).integers(
            30, 255, size=(int(flat[small].max()) + 1, 3), dtype=np.uint8)
        colors[small] = legacy[flat[small]]
    colors[flat == 0] = 0
    return colors.reshape(values.shape + (3,))


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a colorized label mask onto a grayscale image, uint8 RGB.

    :param image: 2-D grayscale or RGB image in the 16-bit range; it is
        divided by 256 to reach 8 bits.
    :param mask: label mask with the image's height and width; each label
        gets a fixed pseudo-random colour and 0 stays unblended.
    """
    fast = _overlay_mask_fast(image, mask, alpha)
    if fast is not None:
        return fast
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    m = mask.astype(np.int64)
    max_label = int(np.max(m)) if m.size else 0
    if max_label > 65535:
        values, inverse = np.unique(m, return_inverse=True)
        colors = _label_colors(values)
        colored = colors[inverse].reshape(m.shape + (3,))
    else:
        rng = np.random.default_rng(0)
        colors = rng.integers(30, 255, size=(max_label + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        colored = colors[m]
    image_8bit = (image.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
    combined = np.where(
        m[..., None] > 0,
        np.clip(image_8bit * (1 - alpha) + colored * alpha, 0, 255),
        image_8bit,
    ).astype(np.uint8)
    return combined



def paint_disk(mask: np.ndarray, cx: int, cy: int, radius: int,
               value: int = 255) -> None:
    """In-place stamp a filled square (radius half-width) at (cx, cy).

    :param mask: 2-D mask, modified in place; the square is clipped to it.
    :param cx: centre column in pixels.
    :param cy: centre row in pixels.
    :param radius: half-width of the square in pixels; values below 1 are
        treated as 1.
    """
    if radius < 1:
        radius = 1
    h, w = mask.shape[:2]
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = value


def paint_line(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int,
               radius: int, value: int = 255) -> None:
    """In-place stamp a line of disks between two points (Bresenham).

    :param mask: 2-D mask, modified in place.
    :param x0: start column in pixels.
    :param y0: start row in pixels.
    :param x1: end column in pixels.
    :param y1: end row in pixels.
    :param radius: half-width of each stamp; see :func:`paint_disk`.
    """
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        paint_disk(mask, x, y, radius, value)
        if x == x1 and y == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy



#: Width, in image pixels, of the cut a divide draws through an object.
#:
#: Not cosmetic. Everything here calls two pixels that touch only at a
#: corner one object (:data:`_EIGHT`), so a one-pixel-wide cut across a
#: diagonal leaves the two halves corner-to-corner and they are still one
#: blob — the split does not take. Measured on a disk over 60 orientations:
#: a 1.0 px cut failed to separate in 58 of them, 1.5 px separated in all
#: 1800 orientation/offset combinations tried, and anything wider only eats
#: more of the object (1.5 px costs ~90 px of a 2800 px object, 2.0 px ~120).
DIVIDE_CUT_WIDTH = 1.5


def _available_labels(mask: np.ndarray):
    """Yield unused uint32 IDs above the maximum, wrapping only at capacity."""
    from .i18n import tr

    used = set(map(int, np.unique(mask))) - {0}
    largest = max(used, default=0)
    if largest > MAX_MASK_LABEL or min(used, default=0) < 0:
        raise ValueError(tr("Mask IDs must be non-negative uint32 integers."))
    candidate = largest + 1 if largest < MAX_MASK_LABEL else 1
    while len(used) < MAX_MASK_LABEL:
        while candidate in used:
            candidate = candidate + 1 if candidate < MAX_MASK_LABEL else 1
        used.add(candidate)
        yield candidate
        candidate = candidate + 1 if candidate < MAX_MASK_LABEL else 1
    raise ValueError(tr("No unused 32-bit label remains."))


def next_label(mask: np.ndarray) -> int:
    """Prefer the next highest ID, using an unused local ID at uint32 capacity.

    Persistent lineage uses independent UUIDs, so a free local pixel ID can
    be reused without merging cell identities. Allocation visits observed
    labels only and never constructs a table of every possible uint32 ID.

    :param mask: The integer label mask whose observed IDs are already in use.
    """
    largest = int(mask.max()) if mask.size else 0
    return largest + 1 if largest < MAX_MASK_LABEL else next(_available_labels(mask))


def _fit_label_width(out: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Cast a working int64 mask back down, widening only when it must.

    Same rule as :func:`combine_masks`: the width follows the values. Keeping
    uint8 for a mask that has just been given label 256 would wrap it round
    to 0 and the new object would vanish into the background.
    """
    top = int(out.max()) if out.size else 0
    if top > MAX_MASK_LABEL or (out.size and int(out.min()) < 0):
        from .i18n import tr

        raise ValueError(tr("Mask label {label} is outside unsigned uint32 capacity.", label=top))
    if np.issubdtype(like.dtype, np.integer) and top <= np.iinfo(like.dtype).max:
        return out.astype(like.dtype)
    return out.astype(np.uint8 if top <= 255 else (
        np.uint16 if top <= 65535 else np.uint32))


def fill_polygon(mask: np.ndarray, points, label_value: Optional[int] = None):
    """Fill a traced outline as ONE object; return ``(mask, label)``.

    This is the tool a brush is not. A brush stamps disks along the path, so
    tracing a cell's rim with it labels the rim and leaves the middle
    background; ``draw`` closes the path (last point back to the first) and
    fills what it encloses, so one gesture produces one solid object with
    one id. Anything already labelled inside the outline is overwritten,
    which is the point: the outline asserts "all of this is one object".

    :param mask: existing label image to copy and edit; labels outside the
        enclosed pixels are preserved and the dtype widens when the new id
        requires it.
    :param points: the traced path as image-pixel ``(x, y)`` pairs. A path
        that encloses less than one pixel -- two points, or a straight line
        traced back over itself -- is returned unchanged with label 0. It is
        a gesture that enclosed nothing, and the alternative is an object a
        pixel wide that the user then has to find and delete.
    :param label_value: the id to give it; by default :func:`next_label`.
    """
    from skimage.draw import polygon as _polygon

    pts = np.asarray(list(points), dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return mask.copy(), 0
    x, y = pts[:, 0], pts[:, 1]
    if abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) < 1.0:
        return mask.copy(), 0
    rows, cols = _polygon(pts[:, 1], pts[:, 0], shape=mask.shape[:2])
    if not len(rows):
        return mask.copy(), 0
    value = int(next_label(mask) if label_value is None else label_value)
    out = mask.astype(np.int64, copy=True)
    out[rows, cols] = value
    return _fit_label_width(out, mask), value


def _segment_band(shape, p0, p1, width: float) -> np.ndarray:
    """Boolean mask of the pixels within ``width``/2 of the segment p0-p1.

    A distance band rather than a rasterised line: the sampled-line cut the
    standalone curation tool uses (400 points between the ends) both leaves
    gaps on a segment longer than 400 px and is one pixel wide wherever it
    lands, and a one-pixel cut does not separate — see
    :data:`DIVIDE_CUT_WIDTH`.
    """
    height, width_px = shape[:2]
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    half = max(0.5, float(width) / 2.0)
    lo_x = max(0, int(np.floor(min(x0, x1) - half)))
    hi_x = min(width_px, int(np.ceil(max(x0, x1) + half)) + 1)
    lo_y = max(0, int(np.floor(min(y0, y1) - half)))
    hi_y = min(height, int(np.ceil(max(y0, y1) + half)) + 1)
    band = np.zeros((height, width_px), dtype=bool)
    if hi_x <= lo_x or hi_y <= lo_y:
        return band
    yy, xx = np.mgrid[lo_y:hi_y, lo_x:hi_x].astype(np.float64)
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        t = np.zeros_like(xx)
    else:
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / length_sq, 0.0, 1.0)
    distance = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
    band[lo_y:hi_y, lo_x:hi_x] = distance <= half
    return band


def divide_object(mask: np.ndarray, p0, p1,
                  width: float = DIVIDE_CUT_WIDTH):
    """Cut every object the segment crosses in two; return ``(mask, splits)``.

    ``splits`` is a list of ``(id_split, id_created)`` pairs, empty when the
    line separated nothing.

    Three decisions make the result usable:

    * **Only the objects the line actually crosses are touched.** The cut is
      clipped to them, so a line drawn past a neighbour leaves that
      neighbour's every pixel where it was. (The standalone tool relabels
      the whole field after cutting, which renumbers every other object in
      it — the same re-keying :func:`canonical_labels` exists to avoid.)
    * **The larger piece keeps the original id**, and the smaller pieces get
      fresh ones above the mask's top label. That is the rule
      :func:`canonical_labels` already applies when one id names two blobs,
      so dividing and then saving does not renumber anything, and the id
      stays on the piece that carries most of what it used to name.
    * **A line that does not separate an object leaves it alone.** Stopping
      halfway across would otherwise carve a groove into the object and
      call it a division; treating it as a miss means the gesture can just
      be redrawn.

    :param mask: 2-D label mask; it is not modified.
    :param p0: first end of the cut as ``(x, y)`` in pixels.
    :param p1: second end of the cut as ``(x, y)`` in pixels.
    """
    band = _segment_band(mask.shape, p0, p1, width)
    if not band.any():
        return mask.copy(), []
    crossed = [int(v) for v in np.unique(mask[band]) if int(v) > 0]
    if not crossed:
        return mask.copy(), []
    out = mask.astype(np.int64, copy=True)
    free_ids = _available_labels(mask)
    splits = []
    for source in crossed:
        body = out == source
        remainder = body & ~band
        pieces, count = _ndimage().label(remainder, structure=_EIGHT)
        if count < 2:
            continue
        areas = np.bincount(pieces.ravel())
        keeps = int(np.argmax(areas[1:])) + 1
        out[body] = 0
        out[pieces == keeps] = source
        for piece in range(1, count + 1):
            if piece == keeps:
                continue
            free_id = next(free_ids)
            out[pieces == piece] = free_id
            splits.append((source, free_id))
    if not splits:
        return mask.copy(), []
    return _fit_label_width(out, mask), splits


def _divide_or_cut(mask: np.ndarray, p0, p1,
                  width: float = DIVIDE_CUT_WIDTH):
    """Divide what the segment separates and keep the cut through the rest.

    Make Masks' Divide stroke (item 685). An object the line crosses
    without separating keeps a background groove along the line, so the
    next stroke continues the earlier one and several strokes can finish a
    division together. An object the band would remove completely is left
    alone, so a cut never deletes an object. Separated objects follow
    :func:`divide_object`: the larger piece keeps its id.

    :param mask: 2-D label mask; it is not modified.
    :param p0: first end of the cut as ``(x, y)`` in pixels.
    :param p1: second end of the cut as ``(x, y)`` in pixels.
    :returns: ``(mask, splits, cut)``: ``splits`` as for
        :func:`divide_object`, ``cut`` the ids that received only a groove.
        Both empty means the mask is an unchanged copy.

    Each crossed object is cut and numbered inside its own bounding box,
    which gives the same pieces in the same order as the whole field: a
    component lies inside its object's box, and labelling numbers
    components in raster order, which a box keeps. 0.3 s less on the GUI
    thread per stroke on a 2,000 px field.
    """
    if tuple(int(v) for v in p0) == tuple(int(v) for v in p1):
        return mask.copy(), [], []
    band = _segment_band(mask.shape, p0, p1, width)
    if not band.any():
        return mask.copy(), [], []
    crossed = [int(v) for v in np.unique(mask[band]) if int(v) > 0]
    if not crossed:
        return mask.copy(), [], []
    out = mask.astype(np.int64, copy=True)
    free_ids = _available_labels(mask)
    splits, cut = [], []
    boxes = (_label_boxes(mask) if np.issubdtype(mask.dtype, np.integer)
             and int(mask.max()) <= 65535 else None)
    for source in crossed:
        if boxes is None:
            rows, cols = np.nonzero(mask == source)
            box = (slice(int(rows.min()), int(rows.max()) + 1),
                   slice(int(cols.min()), int(cols.max()) + 1))
        else:
            box = boxes[source]
        part, strip = out[box], band[box]
        body = part == source
        remainder = body & ~strip
        pieces, count = _ndimage().label(remainder, structure=_EIGHT)
        if count == 0:
            continue
        part[body & strip] = 0
        if count < 2:
            cut.append(source)
            continue
        areas = np.bincount(pieces.ravel())
        keeps = int(np.argmax(areas[1:])) + 1
        part[body] = 0
        part[pieces == keeps] = source
        for piece in range(1, count + 1):
            if piece == keeps:
                continue
            free_id = next(free_ids)
            part[pieces == piece] = free_id
            splits.append((source, free_id))
    if not splits and not cut:
        return mask.copy(), [], []
    return _fit_label_width(out, mask), splits, cut


def _merge_objects_along_line(mask: np.ndarray, p0, p1):
    """Join crossed labels under the first touched ID without painting pixels.

    :param mask: 2-D integer label image; it is not modified.
    :param p0: first endpoint as image-pixel ``(x, y)``.
    :param p1: last endpoint as image-pixel ``(x, y)``.
    :returns: ``(mask, kept_id, joined_ids)``; fewer than two crossed labels
        returns an unchanged copy, zero and an empty list. Disconnected
        regions deliberately share the retained identity; callers save with
        explicit ID preservation rather than splitting this manual grouping.
    """
    start, end = np.asarray(p0, dtype=int), np.asarray(p1, dtype=int)
    if np.array_equal(start, end):
        return mask.copy(), 0, []
    points = np.rint(np.linspace(start, end,
        int(np.abs(end - start).max()) + 1)).astype(int)
    height, width = mask.shape
    inside = ((points[:, 0] >= 0) & (points[:, 0] < width)
              & (points[:, 1] >= 0) & (points[:, 1] < height))
    points = points[inside]
    joined = list(dict.fromkeys(int(v) for v in mask[points[:, 1], points[:, 0]]
                               if int(v) > 0))
    if len(joined) < 2:
        return mask.copy(), 0, []
    out = mask.copy()
    out[np.isin(out, joined)] = joined[0]
    return out, joined[0], joined


def fill_holes(mask: np.ndarray, *, preserve_ids: bool = False) -> np.ndarray:
    """Fill enclosed background pixels, optionally retaining primary identities.

    Holes are filled per object by :func:`fill_label_holes`, never by
    filling the foreground and labelling it again: that joined every pair of
    touching cells into one object.

    :param mask: label image to fill.
    :param preserve_ids: validate the ids as exact primary IDs first
        (unsigned labels, via :func:`canonical_labels`). Either way every
        id is kept. By default a mask with a single foreground value -- a
        brush-painted binary mask, which carries no ids -- is numbered by
        connectivity first, as :func:`canonical_labels` numbers it.
    """
    if preserve_ids:
        return fill_label_holes(canonical_labels(mask, preserve_ids=True))
    m = np.asarray(mask)
    labels = m
    if m.dtype == bool or not np.issubdtype(m.dtype, np.integer):
        labels = canonical_labels(m)
    elif m.size and m.any():
        top = int(m.max())
        if int(np.min(m, where=m > 0, initial=top)) == top:
            labels = canonical_labels(m)
    return _fit_label_width(fill_label_holes(labels), m)


def assign_object_label(mask: np.ndarray, x: int, y: int,
                        label: int) -> Tuple[np.ndarray, int]:
    """Assign one explicit uint32 identity without joining different objects.

    All pixels bearing the clicked object's existing ID move together,
    including disconnected pieces deliberately sharing that identity. Zero
    removes that object. A nonzero target already belonging to another
    object is refused before any pixel changes.

    :param mask: nonempty 2-D nonnegative integer labels; never modified.
    :param x: integer column; outside the field changes nothing.
    :param y: integer row; outside the field changes nothing.
    :param label: integer in 0 through 4294967295, with zero reserved for background.
    :returns: an unsigned label copy and the original ID under the pointer, or zero
        for background or an outside position. The same ID is a no-op.
    :raises ValueError: for invalid labels, coordinates, or an occupied ID.
    """
    from .i18n import tr

    if isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral) \
            or not 0 <= int(label) <= MAX_MASK_LABEL:
        raise ValueError(tr("Labels must be whole numbers from 0 through 4294967295."))
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           for value in (x, y)):
        raise ValueError(tr("Object coordinates must be integer pixels."))
    labels = canonical_labels(mask, preserve_ids=True)
    if int(label) > 65535:
        labels = labels.astype(np.uint32, copy=False)
    height, width = labels.shape
    if not (0 <= int(y) < height and 0 <= int(x) < width):
        return labels, 0
    original = int(labels[int(y), int(x)])
    target = int(label)
    if not original or original == target:
        return labels, original
    if target and np.any(labels == target):
        raise ValueError(tr(
            "Label {label} already belongs to another object in this frame.",
            label=target))
    labels[labels == original] = target
    return labels, original


def relabel_objects(mask: np.ndarray) -> np.ndarray:
    """Return a mask whose connected components are labeled 1..N.

    :param mask: label or binary mask; every pixel above 0 is foreground and
        the result keeps its dtype.
    """
    labeled, _ = _ndimage().label(mask > 0)
    return labeled.astype(mask.dtype)


def clear_mask(mask: np.ndarray) -> np.ndarray:
    """Return an all-zero array shaped like ``mask``.

    :param mask: label mask whose shape and dtype the result copies.
    """
    return np.zeros_like(mask)


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Swap object and background in a LABEL image, and relabel.

    NOT AN INTENSITY INVERSION -- that is :func:`invert_intensity`, and the
    Make Masks screen's "Invert image" is wired to that one. This flips the
    MASK: every labelled pixel becomes background and every background pixel
    becomes foreground, and what comes out is then labelled afresh. On an
    ordinary field the background is one connected region, so what comes back
    is a SINGLE field-sized object with holes where the objects were -- which
    is why it looks as if it does not invert at all: one flat overlay over
    the whole frame reads as nothing having happened.

    It is kept because it is a real thing to want -- a curator who has
    outlined the space BETWEEN the cells has drawn the complement of what is
    wanted -- but under the name that says what it does.

    :param mask: label mask; every pixel above 0 counts as object. The result
        keeps its dtype.
    """
    out = np.where(mask > 0, 0, 1).astype(mask.dtype)
    labeled, _ = _ndimage().label(out)
    return labeled.astype(mask.dtype)


def remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components with area < min_area (in pixels).

    :param mask: label or binary mask; every pixel above 0 is foreground. The
        survivors are relabelled 1..N in the input's dtype.
    :param min_area: smallest component to keep, in pixels; 0 or less
        returns a copy unchanged.
    """
    if min_area <= 0:
        return mask.copy()
    labeled, n = _ndimage().label(mask > 0)
    if n == 0:
        return mask.copy()
    counts = np.bincount(labeled.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    for i in range(1, len(counts)):
        if counts[i] >= min_area:
            keep[i] = True
    filtered = keep[labeled]
    out = np.where(filtered, mask, 0)
    labeled, _ = _ndimage().label(out > 0)
    return labeled.astype(mask.dtype)


def dilate_objects(mask: np.ndarray, distance: int = 1) -> np.ndarray:
    """Grow every object by ``distance`` pixels, without merging any two.

    A label takes the background pixels within ``distance`` of it; a pixel
    contested by two labels goes to the nearer one, and a pixel that already
    carries a label is never taken. SO THE OBJECT COUNT CANNOT CHANGE, which
    is what makes this safe on a mask that has been curated: ids survive, and
    an object that has been given the right id keeps it, along with every
    measurement, track and crop keyed by it.

    The distance is Euclidean (:func:`skimage.segmentation.expand_labels`),
    so ``1`` adds the four edge neighbours and not the corners -- the same
    metric :func:`shrink_objects` takes away by, which is what makes a shrink
    after a dilate land back where it started on an object with no neighbour
    close enough to have blocked the growth.

    :param mask: label image; 0 is background.
    :param distance: pixels to grow by. 0 or less returns a copy.
    :returns: a mask of the same dtype with the same label values.
    """
    from skimage.segmentation import expand_labels

    out = np.asarray(mask)
    if int(distance) <= 0 or not out.size:
        return out.copy()
    grown = expand_labels(out, distance=float(int(distance)))
    return np.asarray(grown).astype(mask.dtype, copy=False)


def shrink_objects(mask: np.ndarray, distance: int = 1) -> np.ndarray:
    """Erode every object by ``distance`` pixels, each one on its own.

    Each object is eroded against everything that is not itself -- background
    AND the objects touching it -- so two objects sharing a border both pull
    back from it and the seam between them widens. Eroding the foreground as
    one binary would instead leave that seam untouched, which is the opposite
    of what a curator reaching for Shrink wants.

    AN OBJECT THINNER THAN TWICE THE DISTANCE DISAPPEARS. That is what
    erosion means, and the screen says how many went rather than letting them
    go quietly; the edit is one undo step, so the way back is one press.

    The image border counts as background, matching
    :func:`scipy.ndimage.binary_erosion`'s own default, so an object the
    field cut off pulls back from the cut too.

    Each object is eroded inside its own bounding box, so the cost follows
    the area of the objects rather than the area of the field -- the same
    reason :func:`canonical_labels` works in boxes.

    :param mask: label image; 0 is background.
    :param distance: pixels to erode by. 0 or less returns a copy.
    :returns: a mask of the same dtype, holding the ids that survived.
    """
    out = np.array(mask, copy=True)
    steps = int(distance)
    if steps <= 0 or not out.size:
        return out
    ndimage = _ndimage()
    boxes = _label_boxes(out)
    for value, box in boxes.items():
        window = out[box]
        inside = window == value
        core = tuple(slice(steps, steps + n) for n in inside.shape)
        padded = np.zeros(tuple(n + 2 * steps for n in inside.shape),
                          dtype=bool)
        padded[core] = inside
        distances = ndimage.distance_transform_edt(padded)
        kept = distances[core] > float(steps)
        window[inside & ~kept] = 0
    return out


def erase_object_at(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    """Zero out the object under (x, y). No-op if no object there.

    :param mask: 2-D label mask; a modified copy is returned and the input is
        left alone.
    :param x: column in pixels; out of range returns ``mask`` itself.
    :param y: row in pixels; out of range returns ``mask`` itself.
    """
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return mask
    label_to_remove = int(mask[y, x])
    if label_to_remove <= 0:
        return mask
    out = mask.copy()
    out[out == label_to_remove] = 0
    return out


def erase_object_in_place(mask: np.ndarray, x: int, y: int) -> int:
    """Zero the object under (x, y) *in place*; return the id removed, or 0.

    The copy-and-return :func:`erase_object_at` is right for a single click,
    where one edit is one undo step. A right-button sweep is dozens of move
    events and one thing the user did, so it deletes in place against a
    single pre-sweep snapshot: copying a 16-bit field per mouse event would
    both stutter and put every object of the sweep on its own undo step.

    The returned id is what the ledger records as the sweep's targets.

    :param mask: 2-D label mask, modified in place.
    :param x: column in pixels; out of range removes nothing.
    :param y: row in pixels; out of range removes nothing.
    """
    height, width = mask.shape[:2]
    if not (0 <= y < height and 0 <= x < width):
        return 0
    label_to_remove = int(mask[y, x])
    if label_to_remove <= 0:
        return 0
    mask[mask == label_to_remove] = 0
    return label_to_remove


def split_object_at(mask: np.ndarray, x: int, y: int, *,
                    min_area: int = 0) -> Tuple[np.ndarray, List[int]]:
    """Cut the object under (x, y) where its halves meet; ``(mask, new_ids)``.

    What Ctrl + left click does. The cut is a watershed on the
    object's own distance to background, the same recipe
    :func:`_split_touching_objects` runs on a whole field: every local
    maximum of that distance is one half's middle and the ridge between two
    of them is the waist where they meet.

    Three decisions, and the first is the one to read:

    * **AN OBJECT WITH ONE CENTRE IS LEFT ALONE** and reported as such,
      rather than being halved through the click. A single click carries no
      direction, so a forced cut would have to invent one, and the object
      that needs cutting is almost always a pair that merged -- which has
      two centres. The gesture for a cut the user aims themselves already
      exists and is the Divide tool (:func:`divide_object`).
    * **The largest piece keeps the id** and the others are given ids above
      the mask's top label, which is :func:`canonical_labels`' own rule, so
      splitting and then saving renumbers nothing.
    * **NO PIXEL IS LOST.** Every pixel of the object ends up under one of
      the new ids. ``min_area`` sets how far apart two centres must be to
      count as two (:func:`_split_touching_objects`' seed spacing) and is
      NOT applied as a drop here: a hand edit moves pixels between ids, and
      a gesture that quietly erased the smaller half would be a delete
      wearing a split's name.

    :param mask: the label image.
    :param x: column clicked, in image pixels.
    :param y: row clicked, in image pixels.
    :param min_area: the smallest object the screen is willing to keep, in
        pixels; it sets the seed spacing, so an object this size is not
        itself cut in two.
    :returns: ``(mask, new_ids)`` -- a new mask and the ids the split
        created, or a copy and an empty list when the click was on
        background, outside the field, or on an object with one centre.
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    ndimage = _ndimage()
    height, width = mask.shape[:2]
    if not (0 <= int(y) < height and 0 <= int(x) < width):
        return mask.copy(), []
    target = int(mask[int(y), int(x)])
    if target <= 0:
        return mask.copy(), []

    where = np.argwhere(mask == target)
    y0, x0 = (int(v) for v in where.min(axis=0))
    y1, x1 = (int(v) + 1 for v in where.max(axis=0))
    window = mask[y0:y1, x0:x1]
    body = window == target

    distance = ndimage.gaussian_filter(
        ndimage.distance_transform_edt(body), 1.0)
    spacing = max(2, int(np.sqrt(max(int(min_area), 12) / np.pi)))
    peaks = peak_local_max(distance, min_distance=spacing,
                           labels=body.astype(np.int32),
                           exclude_border=False)
    if len(peaks) < 2:
        return mask.copy(), []
    markers = np.zeros(body.shape, dtype=np.int32)
    for index, point in enumerate(peaks, start=1):
        markers[tuple(point)] = index
    pieces = watershed(-distance, markers, mask=body)
    found = [int(v) for v in np.unique(pieces) if int(v) > 0]
    if len(found) < 2:
        return mask.copy(), []

    areas = {piece: int(np.count_nonzero(pieces == piece)) for piece in found}
    keeps = max(found, key=lambda piece: (areas[piece], -piece))
    out = mask.astype(np.int64, copy=True)
    free_ids = _available_labels(mask)
    new_ids: List[int] = []
    for piece in found:
        if piece == keeps:
            continue
        free_id = next(free_ids)
        out[y0:y1, x0:x1][pieces == piece] = free_id
        new_ids.append(free_id)
    return _fit_label_width(out, mask), new_ids


def relative_tolerance(image: np.ndarray, percent: float) -> float:
    """Magic-wand tolerance as ``percent`` of ``image``'s intensity range.

    An absolute tolerance is not a portable setting. The value that grabs
    one nucleus in an 8-bit field (range 0..255) selects nothing at all in
    a 16-bit one (range 0..65535), and the one tuned for 16-bit floods the
    entire 8-bit frame. A percentage of *this* image's own range means one
    number behaves the same on both.

    The floor of 1.0 keeps the wand usable on a flat field: a range of zero
    would otherwise give a tolerance of zero, and a tolerance of zero fills
    only pixels exactly equal to the seed.

    :param image: intensity image whose max minus min sets the range; an
        empty array gives ``1.0``.
    :param percent: share of that range, in percent (``5`` means 5 %).
    """
    values = np.asarray(image, dtype=np.float32)
    if not values.size:
        return 1.0
    span = float(values.max() - values.min())
    return max(1.0, (float(percent) / 100.0) * span)


FILTER_BOUNDS = ("min_area", "max_area", "min_intensity", "max_intensity")
"""The four legacy bounds, named as :func:`filter_report`'s keywords are.

They are no longer a second filter. :func:`legacy_filters` turns them into
entries of the one filter list, so a caller that still passes
``min_area=20`` is judged by the same regionprops pass as a user who added
an ``area`` row, and a :class:`FilterRemoval` still names the legacy bound
an object failed so the older ledgers read the same.
"""

FILTER_KEYS = ("property", "min", "max")
"""The keys of one filter entry, the whole of its serialised form.

A filter list is a plain list of ``{"property": name, "min": number or
None, "max": number or None}`` dicts. It goes into the curation ledger, into
a settings file and through JSON unchanged, which is what lets Make Masks and
Mask generation be handed the same list and give the same answer.
"""

_LEGACY_PROPERTY = {"min_area": ("area", "min"), "max_area": ("area", "max"),
                    "min_intensity": ("intensity_mean", "min"),
                    "max_intensity": ("intensity_mean", "max")}


_FILTER_CATALOGUE: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None


def _filter_catalogue() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Every scalar regionprop, split into shape and intensity properties.

    Enumerated FROM scikit-image rather than typed out here: the property
    table the installed skimage declares is walked and each property is
    computed once on a two-object probe. A property that gives exactly one
    column named after itself is scalar and can carry a minimum and a
    maximum; ``coords``, ``image``, ``bbox``, ``centroid`` and the moment
    matrices give arrays or several columns and are left out. A property
    that fails without an intensity image and works with one is an
    intensity property. A newer skimage that adds a property adds a row
    to the list with no change here.

    :returns: ``(shape, intensity)``, each sorted by name. Cached for the
        process, since the answer depends only on the installed skimage.
    """
    global _FILTER_CATALOGUE
    if _FILTER_CATALOGUE is not None:
        return _FILTER_CATALOGUE
    from skimage.measure import _regionprops, regionprops_table

    probe = np.zeros((12, 12), dtype=np.int32)
    probe[1:5, 1:6] = 1
    probe[7:11, 6:10] = 2
    probe[8, 6] = 0
    grey = np.linspace(0.0, 1.0, probe.size).reshape(probe.shape)
    shape: List[str] = []
    intensity: List[str] = []
    skipped = set(getattr(_regionprops, "OBJECT_COLUMNS", ())) | {"label"}
    for name in sorted(getattr(_regionprops, "COL_DTYPES", {})):
        if name in skipped:
            continue
        target = shape
        try:
            columns = regionprops_table(probe, properties=[name])
        except Exception:
            target = intensity
            try:
                columns = regionprops_table(
                    probe, intensity_image=grey, properties=[name])
            except Exception:
                continue
        if list(columns) == [name]:
            target.append(name)
    _FILTER_CATALOGUE = (tuple(shape), tuple(intensity))
    return _FILTER_CATALOGUE


def filter_properties(*, intensity: bool = False) -> Tuple[str, ...]:
    """The regionprops a filter row may name, for an image at hand.

    :param intensity: True when an intensity image exists for the mask, so
        the intensity statistics are real for it. Without one they are not
        offered at all, rather than offered and refused later.
    :returns: property names, shape properties first, each group sorted.
    """
    shape, measured = _filter_catalogue()
    return shape + measured if intensity else shape


def canonical_property(name) -> str:
    """``name`` as the regionprop it means, or a ValueError naming the choices.

    Old skimage spellings (``mean_intensity``, ``MajorAxisLength``,
    ``convex_area``) are accepted and mapped to the current name through
    skimage's own alias table, so a filter written against an older release
    still names the same measurement.

    :param name: a regionprop name, current or legacy spelling.
    """
    from skimage.measure._regionprops import PROPS

    text = str(name).strip()
    canonical = PROPS.get(text, text)
    shape, measured = _filter_catalogue()
    if canonical not in shape and canonical not in measured:
        raise ValueError(
            f"'{name}' is not a scalar scikit-image regionprop an object "
            f"filter can use. Choose one of: {', '.join(shape + measured)}.")
    return canonical


def property_needs_intensity(name) -> bool:
    """Whether the regionprop ``name`` measures pixel values.

    :param name: a regionprop name, current or legacy spelling.
    """
    return canonical_property(name) in _filter_catalogue()[1]


def _bound(value, side: str, name: str) -> Optional[float]:
    """One side of a filter as a float, with None and blank meaning off."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"The {side} of the filter on {name} must be a number or empty, "
            f"not {value!r}.") from error
    if not np.isfinite(number):
        raise ValueError(
            f"The {side} of the filter on {name} must be finite, not {value!r}.")
    return number


def normalise_filters(filters, *, strict: bool = True) -> List[dict]:
    """``filters`` as the canonical list of ``{property, min, max}`` dicts.

    Accepts the list itself, one dict, ``(property, min, max)`` tuples, or
    the list written as JSON or as a Python literal (a settings file stores
    it as text). Every entry is checked here, once, so a typo in a property
    name fails when the list is read rather than on the hundredth field.

    :param filters: the list, one dict, tuples, or their JSON/literal text;
        ``None`` is no filters.
    :param strict: also refuse a minimum above its maximum. The engine
        itself reads lists with ``strict=False``, because a migrated legacy
        pair such as ``min_area=10, max_area=5`` always meant "remove every
        object" and must go on meaning it.
    :raises ValueError: for an unknown property, an unknown key, a bound
        that is not a finite number, or (strict) a minimum above its maximum.
    """
    if filters is None:
        return []
    if isinstance(filters, str):
        text = filters.strip()
        if not text:
            return []
        try:
            filters = json.loads(text)
        except ValueError:
            import ast
            try:
                filters = ast.literal_eval(text)
            except (ValueError, SyntaxError) as error:
                raise ValueError(
                    f"Object filters must be a list of {{'property', 'min', "
                    f"'max'}} entries, not {text!r}.") from error
    if isinstance(filters, dict):
        filters = [filters]
    out: List[dict] = []
    for entry in filters:
        if isinstance(entry, (list, tuple)):
            entry = dict(zip(FILTER_KEYS, entry))
        if not isinstance(entry, dict) or "property" not in entry:
            raise ValueError(
                f"An object filter needs a 'property' and optional 'min' and "
                f"'max', not {entry!r}.")
        unknown = sorted(set(entry) - set(FILTER_KEYS))
        if unknown:
            raise ValueError(
                f"An object filter takes only {', '.join(FILTER_KEYS)}; "
                f"{', '.join(unknown)} is not one of them.")
        name = canonical_property(entry["property"])
        low = _bound(entry.get("min"), "minimum", name)
        high = _bound(entry.get("max"), "maximum", name)
        if strict and low is not None and high is not None and low > high:
            raise ValueError(
                f"The filter on {name} has a minimum ({low:g}) above its "
                f"maximum ({high:g}), so it would remove every object.")
        out.append({"property": name, "min": low, "max": high})
    return out


def legacy_filters(min_area=0, max_area=0, min_intensity=0.0,
                   max_intensity=0.0) -> List[dict]:
    """The four old hard-coded bounds as entries of the filter list.

    This is the migration: 0 meant off for each old bound, and it becomes a
    missing side here, so an old settings file or call is judged by the one
    filter engine with the answer it always had. Area becomes ``area`` and
    mean intensity becomes ``intensity_mean``; equality was kept before and
    is kept now.
    """
    area_low, area_high = int(min_area or 0), int(max_area or 0)
    int_low, int_high = float(min_intensity or 0.0), float(max_intensity or 0.0)
    out: List[dict] = []
    if area_low > 0 or area_high > 0:
        out.append({"property": "area",
                    "min": float(area_low) if area_low > 0 else None,
                    "max": float(area_high) if area_high > 0 else None})
    if int_low > 0 or int_high > 0:
        out.append({"property": "intensity_mean",
                    "min": int_low if int_low > 0 else None,
                    "max": int_high if int_high > 0 else None})
    return out


def parse_object_filters(raw) -> dict:
    """The ``object_filters`` setting as a dict of object type to rows.

    A settings file stores the mapping as JSON or Python-literal text; the
    form and a script hand over the dict. ``None`` or blank text is an
    empty mapping. The rows are returned as given; :func:`settings_filters`
    checks them.

    :param raw: the setting's value.
    :raises ValueError: when the value is not a mapping.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            import ast
            raw = ast.literal_eval(raw)
    if not isinstance(raw, dict):
        raise ValueError(
            "object_filters maps each object type to its filter list, for "
            "example {'cell': [{'property': 'solidity', 'min': 0.9}]}.")
    return {str(key): value for key, value in raw.items()}


def settings_filters(settings, object_type: str) -> List[dict]:
    """The ``object_filters`` list a Mask run applies to ``object_type``.

    ``object_filters`` maps an object type (``cell``, ``nucleus``,
    ``pathogen``, ``organelle`` or an organelle slot) to its filter list, so
    each object type is filtered on its own properties. A missing or empty
    setting is no filters. The retired ``{object}_min_area`` family of a
    saved settings file is folded into this mapping when the file is
    loaded (:func:`spacr.settings._fold_object_bounds`).

    :param settings: the Mask run's settings; ``None`` is no filters.
    :param object_type: the object type whose list is wanted.
    :raises ValueError: when the setting is not a mapping, names an object
        type spaCR does not segment, or holds an invalid filter.
    """
    raw = parse_object_filters((settings or {}).get("object_filters"))
    if not raw:
        return []
    from ..object_settings_table import OBJECT_ORDER

    unknown = sorted(str(key) for key in raw if str(key) not in OBJECT_ORDER)
    if unknown:
        raise ValueError(
            f"object_filters names {', '.join(unknown)}, which is not an "
            f"object type. Use one of: {', '.join(OBJECT_ORDER)}.")
    return normalise_filters(raw.get(object_type))


def object_filter_area_floor(settings, object_type: str) -> int:
    """The smallest area ``object_filters`` keeps for ``object_type``.

    Segmentation drops masks under this area as it makes them (Cellpose's
    ``min_size``), as the retired ``{object}_min_area`` did; the filter
    pass judges the rest. 0 when no ``area`` row sets a minimum.

    :param settings: the Mask run's settings.
    :param object_type: ``cell``, ``nucleus`` or ``pathogen``.
    """
    floors = [entry["min"] for entry in settings_filters(settings, object_type)
              if entry["property"] == "area" and entry["min"] is not None]
    return int(math.ceil(max(floors))) if floors else 0


def filters_need_intensity(filters) -> bool:
    """Whether any entry of ``filters`` needs an intensity image.

    :param filters: a canonical filter list, as :func:`normalise_filters`
        returns.
    """
    return any(property_needs_intensity(entry["property"])
               for entry in normalise_filters(filters, strict=False))


class FailedBound(NamedTuple):
    """One side of one filter entry that an object fell outside.

    :ivar index: the entry's position in the filter list.
    :ivar property: the regionprop it judged.
    :ivar side: ``"min"`` or ``"max"``.
    :ivar bound: the number the user set.
    :ivar value: the object's measured value.
    """

    index: int
    property: str
    side: str
    bound: float
    value: float


class ObjectRemoval(NamedTuple):
    """One object a filter list removed, with what it was measured as.

    :ivar label: the object's id in the label image that was judged.
    :ivar values: every property the pass measured for it, by name.
    :ivar failed: each bound it fell outside, in list order.
    """

    label: int
    values: Dict[str, float]
    failed: Tuple[FailedBound, ...]


def filter_removals(labels: np.ndarray, filters, intensity=None, *,
                    report=(), require_finite_intensity: bool = False
                    ) -> List[ObjectRemoval]:
    """Judge every object of ``labels`` against ``filters``; return the failures.

    THE ONE FILTER ENGINE. Make Masks' Filter list, :func:`filter_report`,
    and Mask generation's :func:`spacr.utils._filter_objects` all run this,
    so one list gives one answer wherever it is applied. It calls
    :func:`skimage.measure.regionprops_table` ONCE per mask with every
    property the list names plus ``report``, never once per filter.

    An object is kept when ``min <= value <= max`` for every entry; a side
    that is None is off. A NaN measurement fails no bound, since there is
    nothing to compare.

    :param labels: integer label image, 2-D or 3-D.
    :param filters: the filter list, in any form :func:`normalise_filters`
        accepts.
    :param intensity: pixel values the same shape as ``labels``; required
        exactly when an entry names an intensity property.
    :param report: extra properties to measure in the same pass so a caller
        can print them; they judge nothing.
    :param require_finite_intensity: raise when an intensity property is
        not finite for some object, as Mask generation always has.
    :returns: the removals sorted by label; empty when nothing fails.
    :raises ValueError: an intensity property without an intensity image,
        or a property scikit-image cannot compute for this dimensionality.
    """
    rules = normalise_filters(filters, strict=False)
    needs = sorted({entry["property"] for entry in rules
                    if property_needs_intensity(entry["property"])})
    grey = None
    if intensity is not None and np.shape(intensity) == np.shape(labels):
        grey = np.asarray(intensity)
        if not np.issubdtype(grey.dtype, np.floating):
            grey = grey.astype(np.float64)
    if needs and grey is None:
        raise ValueError(
            "An intensity plane with the same shape as the mask is "
            f"required: the filter on {', '.join(needs)} measures pixel "
            "values, so it cannot run on a mask alone.")
    active = [entry for entry in rules
              if entry["min"] is not None or entry["max"] is not None]
    lab = np.asarray(labels)
    if not active or not lab.size or not lab.any():
        return []
    extra = [name for name in (report or ())
             if grey is not None or not property_needs_intensity(name)]
    names = list(dict.fromkeys([entry["property"] for entry in active] + extra))
    from skimage.measure import regionprops_table

    observed = None
    if int(lab.max()) > 65535:
        observed, inverse = np.unique(lab, return_inverse=True)
        compact = inverse.reshape(lab.shape) + 1
        compact[lab == 0] = 0
    else:
        compact = lab if np.issubdtype(lab.dtype, np.integer) else lab.astype(np.int32)
    try:
        table = regionprops_table(
            compact,
            intensity_image=grey, properties=["label"] + names)
    except NotImplementedError as error:
        raise ValueError(
            f"scikit-image cannot measure {', '.join(names)} on a "
            f"{lab.ndim}-D mask: {error}") from error
    if require_finite_intensity:
        for name in needs:
            if not np.all(np.isfinite(np.asarray(table[name], dtype=float))):
                raise ValueError(
                    "Intensity filtering requires finite object mean intensities")
    ids = np.asarray(table["label"], dtype=np.int64)
    if observed is not None:
        ids = observed[ids - 1]
    columns = {name: np.asarray(table[name], dtype=float) for name in names}
    failures: Dict[int, List[FailedBound]] = {}
    for index, entry in enumerate(rules):
        if entry["min"] is None and entry["max"] is None:
            continue
        values = columns[entry["property"]]
        for side, bound, outside in (
                ("min", entry["min"], lambda v, b: v < b),
                ("max", entry["max"], lambda v, b: v > b)):
            if bound is None:
                continue
            for row in np.flatnonzero(outside(values, bound)):
                failures.setdefault(row, []).append(FailedBound(
                    index, entry["property"], side, float(bound),
                    float(values[row])))
    removals = [ObjectRemoval(int(ids[row]),
                              {name: float(columns[name][row]) for name in names},
                              tuple(sorted(failed, key=lambda f: (f.index, f.side != "min"))))
                for row, failed in failures.items()]
    removals.sort(key=lambda removal: removal.label)
    return removals


def apply_filters(mask: np.ndarray, image, filters, *,
                  preserve_ids: bool = False, report=("area",)
                  ) -> Tuple[np.ndarray, List[ObjectRemoval]]:
    """Make Masks' filter list applied to one mask; ``(mask, removals)``.

    Objects are judged under :func:`canonical_labels` ids, the ids the hover
    readout shows, and the failing ones are zeroed in a copy; the other ids
    are left as they were. An intensity property reads ``image``, the raw
    loaded pixels (a 3-D image is averaged over its last axis first), never
    the contrast-stretched display.

    :param mask: the label mask to filter.
    :param image: the raw intensity image, or ``None`` when none is open;
        required by any intensity property in ``filters``.
    :param filters: the filter list, in any form :func:`normalise_filters`
        accepts.
    :param preserve_ids: judge objects by their supplied ids, as
        :func:`canonical_labels` does with it; disconnected pieces sharing an
        id count together.
    :param report: properties measured for the ledger in the same pass;
        ``intensity_mean`` is added whenever an image is given.
    :returns: the original array untouched and an empty list when nothing
        fails.
    """
    if mask is None or not np.asarray(mask).size or not np.asarray(mask).max():
        return mask, []
    grey = None
    if image is not None:
        grey = np.asarray(image, dtype=np.float32)
        if grey.ndim == 3 and np.ndim(mask) == 2:
            grey = grey.mean(axis=2)
    labels = canonical_labels(mask, preserve_ids=preserve_ids)
    extra = tuple(report or ()) + (("intensity_mean",) if grey is not None else ())
    removals = filter_removals(labels, filters, grey, report=extra)
    if not removals:
        return mask, []
    out = mask.copy()
    out[np.isin(labels, [removal.label for removal in removals])] = 0
    return out, removals


class FilterRemoval(NamedTuple):
    """One object the filter dropped, and which bound dropped it.

    The screen reports "object 22 with area x and intensity y was removed
    by minimum intensity", one row per object, so the filter has to say
    more than which ids went.

    :ivar label: the id :func:`canonical_labels` gave the object -- the same
        id the hover readout showed for it.
    :ivar area: its pixel count.
    :ivar mean_intensity: its mean value on the raw image.
    :ivar bounds: the names of every LEGACY bound it failed, a subset of
        :data:`FILTER_BOUNDS` in that order.
    :ivar failed: every bound it failed, legacy or listed, as
        :class:`FailedBound` entries.
    """

    label: int
    area: int
    mean_intensity: float
    bounds: Tuple[str, ...]
    failed: Tuple[FailedBound, ...] = ()


def filter_report(mask: np.ndarray, image: np.ndarray, *,
                  min_area: int = 0, max_area: int = 0,
                  min_intensity: float = 0.0, max_intensity: float = 0.0,
                  filters=None, preserve_ids: bool = False
                  ) -> Tuple[np.ndarray, List[FilterRemoval]]:
    """Filter ``mask`` by the legacy bounds and ``filters``, measuring what went.

    The four keyword bounds are migrated into the filter list by
    :func:`legacy_filters` and judged together with ``filters`` by
    :func:`apply_filters` -- one engine, one regionprops pass.

    :param mask: 2-D label mask; it is not modified.
    :param image: raw intensity image the same height and width as ``mask``;
        a 3-D image is averaged over its last axis first.
    :param filters: further filter entries (see :func:`normalise_filters`).
    :param preserve_ids: measure all pixels bearing an ID as one object,
        including lone or disconnected primary/secondary labels.
    :returns: ``(mask, removals)``, the removals sorted by id. Nothing to do
        returns the original array untouched and an empty list.
    """
    legacy = legacy_filters(min_area, max_area, min_intensity, max_intensity)
    rules = legacy + normalise_filters(filters)
    if not rules:
        return mask, []
    names = {("area", "min"): "min_area", ("area", "max"): "max_area",
             ("intensity_mean", "min"): "min_intensity",
             ("intensity_mean", "max"): "max_intensity"}
    out, removals = apply_filters(mask, image, rules, preserve_ids=preserve_ids)
    reported = []
    for removal in removals:
        legacy_failed = {names[(f.property, f.side)] for f in removal.failed
                         if f.index < len(legacy)}
        reported.append(FilterRemoval(
            removal.label, int(round(removal.values.get("area", 0))),
            float(removal.values.get("intensity_mean", float("nan"))),
            tuple(name for name in FILTER_BOUNDS if name in legacy_failed),
            removal.failed))
    return out, reported


def filter_objects(mask: np.ndarray, image: np.ndarray, *,
                   min_area: int = 0, max_area: int = 0,
                   min_intensity: float = 0.0,
                   max_intensity: float = 0.0, filters=None,
                   preserve_ids: bool = False) -> Tuple[np.ndarray, List[int]]:
    """Drop objects outside the bounds; ``(mask, dropped ids)``.

    :func:`filter_report` without the measurements: the same engine, the
    same list, only the ids kept. Area is the pixel count; intensity is the
    mean on the *raw* image. Each legacy bound is off at 0.

    :param mask: 2-D label mask; it is not modified.
    :param image: raw intensity image the same height and width as ``mask``.
    :param filters: further filter entries (see :func:`normalise_filters`).
    :param preserve_ids: retain primary/secondary identities when measuring
        and removing labels; disconnected pieces sharing an ID count together.
    """
    out, removals = filter_report(
        mask, image, min_area=min_area, max_area=max_area,
        min_intensity=min_intensity, max_intensity=max_intensity,
        filters=filters, preserve_ids=preserve_ids)
    return out, [removal.label for removal in removals]


class PixelReadout(NamedTuple):
    """What the Make Masks readout says about one pixel of the open field.

    :ivar x: column, in image pixels.
    :ivar y: row, in image pixels.
    :ivar intensity: the raw image value at the pixel, before any display
        stretching.
    :ivar label: the id of the object under the pixel, as
        :func:`canonical_labels` numbers it; 0 on background.
    :ivar area: that object's pixel count; 0 on background.
    :ivar mean_intensity: that object's mean raw intensity, or ``None`` on
        background.
    """

    x: int
    y: int
    intensity: float
    label: int = 0
    area: int = 0
    mean_intensity: Optional[float] = None


class ObjectLookup:
    """The objects of one mask, measured as :func:`filter_objects` measures them.

    Built once for a mask state and then asked about one pixel at a time, so
    a readout that follows the mouse costs a bounding box per question rather
    than the whole field. The id is the one :func:`canonical_labels` gives,
    the area is that id's pixel count, and the mean is taken on the raw image
    in ``float32`` over the object's pixels in raster order, which is the
    arithmetic :func:`skimage.measure.regionprops` performs for the filter.
    So an intensity bound set to the mean shown keeps the object, and a bound
    just past it removes it.

    :param mask: the label image.
    :param image: the raw image under it, with the mask's height and width.
    """

    def __init__(self, mask: np.ndarray, image: np.ndarray, *,
                 preserve_ids: bool = False):
        """Number the objects and index their bounding boxes.

        :param mask: the label image.
        :param image: the raw image under it.
        :param preserve_ids: index exact IDs without binary interpretation
            or splitting disconnected pieces; agrees with exact-ID filtering.
        """
        self.labels = canonical_labels(mask, preserve_ids=preserve_ids)
        grey = np.asarray(image, dtype=np.float32)
        if grey.ndim == 3:
            grey = grey.mean(axis=2)
        self._grey = grey
        self._boxes = _label_boxes(self.labels)
        self._measured: dict = {}

    def measure(self, label: int) -> Optional[Tuple[int, float]]:
        """The area and mean intensity of object ``label``.

        :param label: an id in :attr:`labels`.
        :returns: ``(area, mean)``, or ``None`` when no object has that id.
        """
        label = int(label)
        if label in self._measured:
            return self._measured[label]
        if label not in self._boxes:
            return None
        box = self._boxes[label]
        inside = self.labels[box] == label
        found = (int(np.count_nonzero(inside)),
                 float(np.mean(self._grey[box][inside])))
        self._measured[label] = found
        return found

    def at(self, x: int, y: int) -> Optional[PixelReadout]:
        """The readout for image pixel ``(x, y)``.

        :param x: column.
        :param y: row.
        :returns: the readout, or ``None`` for a pixel outside the field.
        """
        height, width = self.labels.shape[:2]
        if not (0 <= int(x) < width and 0 <= int(y) < height):
            return None
        x, y = int(x), int(y)
        intensity = float(self._grey[y, x])
        label = int(self.labels[y, x])
        measured = self.measure(label) if label else None
        if measured is None:
            return PixelReadout(x, y, intensity)
        return PixelReadout(x, y, intensity, label, *measured)


def _full_connectivity(ndim: int) -> np.ndarray:
    """The structuring element joining every face, edge and corner neighbour.

    :param ndim: 2 for a field (8-connected), 3 for a volume (26-connected).
    :returns: :data:`_EIGHT` for a field, ``ones((3,) * ndim)`` otherwise.
    """
    return _EIGHT if int(ndim) == 2 else np.ones((3,) * int(ndim), dtype=np.uint8)


def connected_instances(binary: np.ndarray, min_area: int = 0) -> np.ndarray:
    """Label every separated foreground region as its own object.

    :param binary: array whose truthy pixels are foreground. Regions touching
        diagonally are connected under the editor's eight-neighbour rule; in
        a volume, voxels sharing a face, edge or corner are 26-connected.
    :param min_area: regions smaller than this (pixels, or voxels in a
        volume) are dropped rather than labelled, so a detection does not
        hand back a field of single-pixel speckles to delete by hand.
    """
    foreground = np.asarray(binary, dtype=bool)
    components, _count = _ndimage().label(foreground, structure=_full_connectivity(foreground.ndim))
    sizes = np.bincount(components.ravel(), minlength=1)
    keep = sizes >= int(min_area)
    keep[0] = False
    lookup = np.zeros(sizes.size, dtype=np.int32)
    lookup[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lookup[components]


def otsu_instances(image: np.ndarray, *, bright: bool = True,
                   min_area: int = 0) -> np.ndarray:
    """Threshold ``image`` at Otsu's level and label what is left.

    :param image: numeric intensity image. It is converted to float32 before
        the threshold is estimated and must contain at least one pixel.
    :param bright: objects are brighter than background (fluorescence).
        False takes the dark side instead, for a brightfield or a
        stained-plaque image where the objects absorb.
    :param min_area: passed to :func:`connected_instances`.
    :raises ValueError: on an empty image, which has no threshold to find.
    """
    from skimage.filters import threshold_otsu

    values = np.asarray(image, dtype=np.float32)
    if not values.size:
        raise ValueError("Otsu needs an image; this one is empty.")
    threshold = float(threshold_otsu(values))
    binary = (values > threshold) if bright else (values < threshold)
    return connected_instances(binary, min_area=min_area)


def _otsu_values(image: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    """The float32 array every Otsu level in this module is measured on.

    One reader, so the level the histogram preview marks is measured on the
    same pixels the threshold is taken on rather than on the raw field: a
    preview drawn before the smoothing would mark a level that is not where
    the cut lands.

    :param image: the field, or a crop of it.
    :param smoothing: Gaussian sigma; 0 returns the values unsmoothed.
    :raises ValueError: on an empty image, which has no threshold to find.
    """
    values = np.asarray(image, dtype=np.float32)
    if not values.size:
        raise ValueError("Otsu needs an image; this one is empty.")
    sigma = max(0.0, float(smoothing))
    if sigma > 0.0:
        values = _ndimage().gaussian_filter(values, sigma)
    return values


#: The GLOBAL threshold algorithms this module can cut a field at, as
#: ``name -> the scikit-image function that finds the level``. Each takes
#: one image and returns one intensity, so all of them reach the detection
#: through the very same code: the correction, the smoothing, the
#: bright/dark side, fill holes, the split, the border rule and the minimum
#: area are found once, in :func:`_otsu_instances`, and an algorithm is the
#: one line that differs. ADDING ONE IS A ROW HERE.
#:
#: ``otsu`` is first and is the default everywhere, so a field thresholded
#: by a screen that knows nothing of this dictionary is thresholded exactly
#: as it always was.
GLOBAL_THRESHOLDS: Dict[str, str] = {
    "otsu": "threshold_otsu",
    "li": "threshold_li",
    "yen": "threshold_yen",
    "triangle": "threshold_triangle",
    "isodata": "threshold_isodata",
    "mean": "threshold_mean",
    "minimum": "threshold_minimum",
}

#: The LOCAL threshold algorithms, which return one level PER PIXEL rather
#: than one for the field, as ``name -> the scikit-image function``. They
#: read a window size, and Sauvola and Niblack read ``k`` as well.
#:
#: WHAT IS DELIBERATELY NOT HERE. Local MEAN and local GAUSSIAN
#: (``skimage.filters.threshold_local``) are not listed, because Make Masks
#: already offers them: they are what the Adaptive threshold mode runs,
#: through the organelle engine's own ``adaptive`` branch, with the same
#: block size and offset. Local OTSU is not listed either: it is the Otsu
#: category's "Local threshold (uneven illumination)" switch and is
#: :func:`_local_otsu_binary`. Listing either again would be two controls
#: for one operation.
LOCAL_THRESHOLDS: Dict[str, str] = {
    "sauvola": "threshold_sauvola",
    "niblack": "threshold_niblack",
}


def threshold_algorithms() -> Tuple[str, ...]:
    """Every threshold algorithm name, global then local."""
    return tuple(GLOBAL_THRESHOLDS) + tuple(LOCAL_THRESHOLDS)


def _global_level(values: np.ndarray, algorithm: str) -> float:
    """The one intensity ``algorithm`` cuts ``values`` at.

    :param values: the smoothed float image, from :func:`_otsu_values`.
    :param algorithm: a key of :data:`GLOBAL_THRESHOLDS`.
    :raises ValueError: for an algorithm this module does not know, rather
        than quietly thresholding by Otsu under another name.
    """
    from skimage import filters

    name = GLOBAL_THRESHOLDS.get(str(algorithm))
    if name is None:
        raise ValueError(
            f"{algorithm!r} is not a global threshold algorithm; "
            f"the ones there are: {sorted(GLOBAL_THRESHOLDS)}.")
    return float(getattr(filters, name)(values))


def _local_level_map(values: np.ndarray, algorithm: str, *, window: int,
                     k: float) -> np.ndarray:
    """A per-pixel threshold for ``values`` from a local algorithm.

    Niblack uses ``T = m - k*s`` and Sauvola uses
    ``T = m*(1 + k*(s/R - 1))``, with local mean m and deviation s.
    Values from :func:`_otsu_values` are float32 without range rescaling;
    scikit-image therefore defaults Sauvola's R to 1, not to the observed
    intensity range. Sauvola can consequently behave differently when the
    same image is multiplied by an intensity scale factor.

    :param values: the smoothed float image.
    :param algorithm: a key of :data:`LOCAL_THRESHOLDS`.
    :param window: the odd window size, in pixels.
    :param k: the algorithm's ``k``.
    :raises ValueError: for an algorithm this module does not know.
    """
    from skimage import filters

    name = LOCAL_THRESHOLDS.get(str(algorithm))
    if name is None:
        raise ValueError(
            f"{algorithm!r} is not a local threshold algorithm; "
            f"the ones there are: {sorted(LOCAL_THRESHOLDS)}.")
    return np.asarray(
        getattr(filters, name)(values, window_size=_odd_window(window),
                               k=float(k)),
        dtype=np.float32)


def _otsu_levels(image: np.ndarray, *, bright: bool = True,
                 correction: float = 1.0, smoothing: float = 0.0,
                 classes: int = 2, algorithm: str = "otsu") -> List[float]:
    """The intensity or intensities the field is actually cut at.

    The histogram preview shows the chosen level, and the only way a
    preview can be trusted to show it is for the detector to read the level
    from here too -- so :func:`_otsu_instances` calls this rather than
    finding its own, and a preview cannot drift from the button.

    The numbers are on the SMOOTHED image and already carry ``correction``,
    because that is where the cut is made.

    :param image: the field, or a crop of it.
    :param bright: objects are brighter than background. With two classes
        and a dark-object cut the level returned is the mirrored one --
        ``top - (top - level) * correction`` -- which is the value the
        detector compares against, so the preview marks the cut the user
        gets rather than Otsu's own number. Not read for three classes or
        more, where the class number says which side is meant.
    :param correction: Otsu's level is multiplied by this.
    :param smoothing: Gaussian sigma, applied before the level is found.
    :param classes: 2 for Otsu's own two-class split, 3 or more for
        multi-level Otsu (:func:`skimage.filters.threshold_multiotsu`),
        which returns ``classes - 1`` rising levels.
    :param algorithm: which of :data:`GLOBAL_THRESHOLDS` finds the level.
        Read only for two classes; multi-level Otsu is its own algorithm
        and is asked for by a class count above two.
    :raises ValueError: on an empty image, a correction that is not greater
        than 0, fewer than two classes, or an unknown algorithm.
    """
    from skimage.filters import threshold_multiotsu

    factor = _otsu_correction_factor(correction)
    count = int(classes)
    if count < 2:
        raise ValueError(
            f"Otsu needs at least two classes; got {classes!r}.")
    values = _otsu_values(image, smoothing)
    if count == 2:
        level = _global_level(values, algorithm)
        if bright:
            return [level * factor]
        top = float(values.max())
        return [top - (top - level) * factor]
    return [float(level) * factor
            for level in threshold_multiotsu(values, classes=count)]


def _otsu_histogram(image: np.ndarray, *, smoothing: float = 0.0,
                    bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Counts and bin edges of the values the threshold is measured on.

    The picture behind the histogram preview. Measured on :func:`_otsu_values`
    for the same reason the levels are: a histogram of the raw field under a
    level found on the smoothed one would put the marker in the wrong valley.

    :param image: the field, or a crop of it.
    :param smoothing: Gaussian sigma, matching the detection's.
    :param bins: how many bars.
    :returns: ``(counts, edges)`` as :func:`numpy.histogram` returns them,
        so ``edges`` is one longer than ``counts``.
    :raises ValueError: on an empty image.
    """
    values = _otsu_values(image, smoothing)
    counts, edges = np.histogram(values, bins=max(2, int(bins)))
    return counts, edges


def _otsu_correction_factor(correction: float) -> float:
    """Validate and return the threshold correction multiplier."""
    factor = float(correction)
    if not factor > 0.0:
        raise ValueError(
            f"The Otsu threshold correction must be greater than 0; got "
            f"{correction!r}.")
    return factor


def _odd_window(window: int) -> int:
    """The local-threshold window as an odd number of pixels, validated.

    An even window has no centre pixel, so the level a pixel is judged
    against would be measured off-centre from it.

    :raises ValueError: for a window under 3.
    """
    size = int(window)
    if size < 3:
        raise ValueError(
            f"The local threshold window must be at least 3 px; got "
            f"{window!r}.")
    return size if size % 2 else size + 1


def _square_footprint(size: int) -> np.ndarray:
    """A square footprint of ``size`` px, whatever scikit-image calls it.

    ``footprint_rectangle`` arrived in scikit-image 0.25 and ``square`` is
    deprecated there and gone in 0.27, so both names are tried rather than
    pinning the package on a helper that returns an array of ones.
    """
    try:
        from skimage.morphology import footprint_rectangle

        return footprint_rectangle((size, size))
    except ImportError:
        from skimage.morphology import square

        return square(size)


def _local_otsu_binary(values: np.ndarray, *, window: int, bright: bool,
                       correction: float) -> np.ndarray:
    """Threshold every pixel against Otsu's level in the window around it.

    Adaptive Otsu: one level for the whole field loses
    an object wherever the illumination falls away, because the corner of a
    field can be dimmer than the background at its centre. Here each pixel is
    compared with the level found inside a ``window`` x ``window`` square
    centred on it (:func:`skimage.filters.rank.otsu`), so a dim corner is
    judged against its own corner.

    THE LEVELS ARE FOUND ON A 256-STEP RESCALING of ``values``, which is what
    the rank filters take and what keeps a megapixel field pressable: a
    16-bit rank filter builds a 65,536-bin histogram per pixel. The cut is
    then made on the same rescaling, so nothing is compared across the two.

    :param values: the smoothed float image.
    :param window: odd window size in pixels.
    :param bright: objects are brighter than background.
    :param correction: multiplies the local level, exactly as it multiplies
        the global one.
    :returns: a boolean foreground image.
    """
    from skimage.filters.rank import otsu as rank_otsu

    size = _odd_window(window)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=bool)
    scaled = np.clip(
        np.rint((values - low) * (255.0 / (high - low))), 0.0, 255.0
    ).astype(np.uint8)
    levels = np.asarray(rank_otsu(scaled, _square_footprint(size)),
                        dtype=np.float32)
    here = scaled.astype(np.float32)
    factor = float(correction)
    if bright:
        return here > levels * factor
    return here < 255.0 - (255.0 - levels) * factor


def _otsu_instances(image: np.ndarray, *, bright: bool = True,
                    min_area: int = 0,
                    correction: float = 1.0,
                    smoothing: float = 0.0,
                    fill_holes: bool = False,
                    split_touching: bool = False,
                    exclude_border: bool = False,
                    classes: int = 2,
                    foreground_class: Optional[int] = None,
                    local: bool = False,
                    window: int = 51,
                    algorithm: str = "otsu",
                    local_k: float = 0.2) -> np.ndarray:
    """:func:`otsu_instances` with Otsu's level multiplied by ``correction``.

    OR ANOTHER ALGORITHM'S LEVEL. ``algorithm`` names one of
    :data:`GLOBAL_THRESHOLDS` or :data:`LOCAL_THRESHOLDS`, and it changes
    exactly one thing: where the number the field is cut at comes from.
    The smoothing, the correction, the bright-or-dark side, filling holes,
    the watershed split, the border rule and the minimum area are the same
    code for every one of them, which is the whole reason the algorithms
    are a dictionary and not ten functions.

    The "threshold correction", which is CellProfiler's threshold
    correction factor: the level Otsu finds is multiplied before it is used.
    For positive global thresholds and local Otsu, above 1 is stricter on
    either side: dark-object correction uses the distance below the image
    maximum (255 after rescaling for local Otsu).
    Sauvola and Niblack instead multiply their direct local level maps;
    for positive thresholds, raising the factor keeps fewer bright pixels
    but more dark pixels. Negative thresholds reverse those directions.

    A correction of exactly 1 with every switch below off IS
    :func:`otsu_instances`, looked up by name at call time, so the
    uncorrected path does not move at all.

    THE FOUR SWITCHES ARE THE OTSU MODE'S EXTRA SETTINGS, and they all
    default OFF -- a plain threshold and a connected-components
    labelling, which is what this did before they existed. Three of them are
    the steps the magnifier's Otsu mode has always taken and the detect
    button never did (:func:`_classical_region_labels`), which is why the
    two could disagree about the same field; turning them on is how a user
    makes the button do what the box under the mouse showed.

    :param correction: the factor, greater than 0.
    :param smoothing: Gaussian sigma applied before the level is estimated
        AND before the image is cut at it, so a noisy field is thresholded
        on what a reader sees rather than on its speckle.
    :param fill_holes: close the holes inside the thresholded foreground.
        A nucleus dimmer in the middle than at its rim arrives as a ring
        without this.
    :param split_touching: cut each blob where two objects meet
        (:func:`_split_touching_objects`) instead of labelling it whole.
    :param exclude_border: drop the objects the field's own edge cuts
        through (:func:`_drop_border_objects`).
    :param classes: multi-level Otsu. 2 is Otsu's own two-class
        split and is what this did before. 3 or more splits the histogram
        into that many brightness bands
        (:func:`skimage.filters.threshold_multiotsu`), which is how a field
        holding background, a dim halo and bright nuclei is cut at the
        boundary that matters instead of at the one compromise level between
        all three.
    :param foreground_class: with three classes or more, WHICH band becomes
        the objects, counting 0 for the dimmest. None takes the brightest,
        ``classes - 1``. Exactly that band is taken, so choosing a middle one
        gives the halo without the nuclei inside it -- which is the point of
        asking for more than two classes. ``bright`` is NOT read here: the
        class number already says which side is meant.
    :param local: the adaptive threshold. Each pixel is judged
        against Otsu's level in the ``window`` around it rather than against
        one level for the whole field (:func:`_local_otsu_binary`), which is
        what recovers objects in a corner the illumination has fallen away
        from. Two classes only.
    :param window: the local window, in pixels; rounded up to an odd number
        so it has a centre pixel. Read when ``local`` is on and by the
        algorithms in :data:`LOCAL_THRESHOLDS`.
    :param algorithm: which algorithm finds the level -- one of
        :data:`GLOBAL_THRESHOLDS` or :data:`LOCAL_THRESHOLDS`. ``otsu``,
        the default, is what this function did before there were others.
    :param local_k: dimensionless local contrast weight, default 0.2.
        Niblack uses ``T = m - k*s``; Sauvola uses
        ``T = m*(1 + k*(s/R - 1))``, with local mean m, standard deviation
        s and scikit-image's float-input default R=1. Increasing k lowers
        Niblack's threshold; Sauvola's direction depends on m and s/R.
        Bright foreground is strictly above the corrected level and dark
        foreground strictly below it. No automatic intensity rescaling.
    :raises ValueError: on an empty image, a correction that is not greater
        than 0, fewer than two classes, a foreground class outside them, a
        window under 3 px, or ``local`` asked for together with more than
        two classes -- which have no single meaning together and would
        otherwise silently drop one of the two.
    """
    factor = _otsu_correction_factor(correction)
    sigma = max(0.0, float(smoothing))
    count = int(classes)
    if count < 2:
        raise ValueError(f"Otsu needs at least two classes; got {classes!r}.")
    chosen = count - 1 if foreground_class is None else int(foreground_class)
    if not 0 <= chosen < count:
        raise ValueError(
            f"The foreground class must be one of 0..{count - 1} for "
            f"{count} classes; got {foreground_class!r}.")
    if local and count > 2:
        raise ValueError(
            "A local threshold finds one level per window, so it cannot "
            "also split the field into more than two classes. Turn one of "
            "the two off.")
    if str(algorithm or "otsu") in LOCAL_THRESHOLDS and count > 2:
        raise ValueError(
            f"{algorithm} finds one level per window, so it cannot also "
            f"split the field into {count} classes. Use Multi-Otsu, or "
            f"put the class count back to 2.")
    name = str(algorithm or "otsu")
    plain = (factor == 1.0 and sigma == 0.0 and not fill_holes
             and not split_touching and not exclude_border
             and count == 2 and not local and name == "otsu")
    if plain:
        return otsu_instances(image, bright=bright, min_area=min_area)
    values = _otsu_values(image, sigma)
    if name in LOCAL_THRESHOLDS:
        levels = _local_level_map(values, name, window=window, k=local_k)
        binary = (values > levels * factor if bright
                  else values < levels * factor)
    elif local:
        binary = _local_otsu_binary(values, window=window, bright=bright,
                                    correction=factor)
    elif count == 2:
        level = _otsu_levels(values, bright=bright, correction=factor,
                             smoothing=0.0, classes=2, algorithm=name)[0]
        binary = values > level if bright else values < level
    else:
        levels = _otsu_levels(values, bright=bright, correction=factor,
                              smoothing=0.0, classes=count)
        binary = np.digitize(values, levels) == chosen
    if fill_holes:
        binary = _ndimage().binary_fill_holes(binary)
    if split_touching:
        labels = _split_touching_objects(binary, min_area=min_area)
    else:
        labels = connected_instances(binary, min_area=min_area)
    if exclude_border:
        labels = _drop_border_objects(labels)
    return labels


def combine_masks(old: np.ndarray, new: np.ndarray,
                  mode: str = "replace") -> np.ndarray:
    """Fold a fresh detection into an existing mask.

    :param old: existing label image used as the merge base; ignored when
        ``mode`` is ``"replace"``.
    :param new: newly detected label image. In merge mode its positive labels
        are offset above ``old`` and copied only into background pixels.
    :param mode: ``"replace"`` -- the detection is the mask, and whatever
        was there is gone. ``"merge"`` -- keep every existing object and add
        the detected ones only where nothing is labelled yet, with fresh ids
        above the existing maximum. Merge never overwrites or splits an
        object that was curated by hand, which is the point of offering the
        choice: a detection run halfway through an editing session should
        not be able to silently undo the first half of it.
    :raises ValueError: for an unknown mode, rather than quietly picking
        one and discarding the user's edits.
    """
    if mode not in ("replace", "merge"):
        raise ValueError(
            f"combine mode must be 'replace' or 'merge', not {mode!r}")
    incoming = np.asarray(new).astype(np.int64)
    if mode == "replace":
        out = incoming
    else:
        base = int(old.max()) if old.size else 0
        out = old.astype(np.int64, copy=True)
        if base + int(incoming.max(initial=0)) > MAX_MASK_LABEL:
            added = np.zeros_like(incoming)
            free_ids = _available_labels(old)
            for value in np.unique(incoming[incoming > 0]):
                added[incoming == value] = next(free_ids)
        else:
            added = np.where(incoming > 0, incoming + base, 0)
        free = out == 0
        out[free] = added[free]
    top = int(out.max()) if out.size else 0
    if top > MAX_MASK_LABEL:
        from .i18n import tr

        raise ValueError(tr("Combined mask needs label {label}, past uint32; "
                            "raise the minimum area so the detection makes fewer objects.", label=top))
    return out.astype(np.uint32 if top > 65535 or old.dtype == np.uint32 else (
        np.uint8 if top <= 255 else np.uint16))


#: The rules a live-magnifier commit knows for an object that lands on one
#: already labelled, in the order the editor offers them. ``clip`` keeps
#: only the new object's unlabelled pixels, so no existing object loses a
#: pixel; ``skip`` leaves out any object that touches an existing one;
#: ``replace`` lets the new object take every pixel it covers.
_MAGNIFIER_OVERLAP_RULES = ("clip", "skip", "replace", "merge")

#: How far one unit of magnifier sensitivity moves the classical cut, as a
#: fraction of the region's stretched intensity range. Positive lowers the
#: cut, so a more sensitive magnifier takes in dimmer pixels.
_CLASSICAL_SENSITIVITY_STEP = 0.05

#: Otsu's effectiveness -- between-class over total variance -- at or above
#: which a region is taken to hold two populations and is cut between them.
#: Below it the region is one population, background, with at most a few dim
#: objects in it; and background still HAS an Otsu level -- it splits the
#: noise in half, at an effectiveness near 2/pi for Gaussian noise -- so
#: cutting there hands back a sponge of objects that are not there. Such a
#: region is cut at a noise floor instead, :data:`_CLASSICAL_NOISE_SIGMAS`
#: robust deviations above its median.
_CLASSICAL_MIN_SEPARATION = 0.8

#: How many robust standard deviations (1.4826 x the median absolute
#: deviation) above the median a smoothed pixel must be to count as an
#: object in a one-population region. Each unit of sensitivity takes half a
#: deviation off, down to one.
_CLASSICAL_NOISE_SIGMAS = 4.0

#: Gaussian smoothing, in pixels, before either cut. It averages pixel noise
#: down about three and a half times, which is the difference between a dim
#: object being found and being broken into speckle.
_CLASSICAL_SMOOTHING = 1.0


def _magnifier_box(shape, x: int, y: int,
                   size: int) -> Tuple[int, int, int, int]:
    """The region a magnifier at image pixel ``(x, y)`` covers, clipped.

    :param shape: the image shape, ``(height, width, ...)``.
    :param x: column of the pixel under the cursor.
    :param y: row of the pixel under the cursor.
    :param size: side of the unclipped square, in image pixels. The cursor
        sits at index ``size // 2`` of it, so an even size puts the cursor
        just right of and below the middle.
    :returns: ``(x0, y0, x1, y1)`` with ``x1``/``y1`` exclusive -- the slice
        ``image[y0:y1, x0:x1]``. At the image border the box is cut short on
        that side only and never padded, so a crop is always real pixels and
        its top-left corner is always where its labels go.
    """
    height, width = int(shape[0]), int(shape[1])
    side = max(1, int(size))
    left = int(x) - side // 2
    top = int(y) - side // 2
    return (max(0, left), max(0, top),
            min(width, left + side), min(height, top + side))


def _drop_cut_objects(labels: np.ndarray, box, shape) -> np.ndarray:
    """Zero every object the box's own edge cuts through.

    An object touching an edge of the box that lies INSIDE the image is the
    part of an object the box happened to cover, and its boundary there is
    where the box ends -- the rule :func:`cut_recrop` applies to a recrop.
    An edge that is the image border cuts nothing off, so objects touching
    it are kept.

    :param labels: label image in crop coordinates.
    :param box: the crop's ``(x0, y0, x1, y1)`` in image pixels.
    :param shape: the image shape the box was clipped to.
    :returns: a copy of ``labels`` without the cut objects.
    """
    out = np.array(labels, copy=True)
    if not out.size or not out.max():
        return out
    x0, y0, x1, y1 = (int(v) for v in box[:4])
    height, width = int(shape[0]), int(shape[1])
    edges = []
    if y0 > 0:
        edges.append(out[0, :])
    if y1 < height:
        edges.append(out[-1, :])
    if x0 > 0:
        edges.append(out[:, 0])
    if x1 < width:
        edges.append(out[:, -1])
    if edges:
        cut = np.unique(np.concatenate(edges))
        cut = cut[cut > 0]
        if cut.size:
            out[np.isin(out, cut)] = 0
    return out


def _split_touching_objects(binary: np.ndarray, min_area: int = 0) -> np.ndarray:
    """Label ``binary``, cutting each blob where two objects meet.

    A watershed on the blob's own distance transform: every local maximum of
    the distance to background is one object's middle, and the ridge between
    two of them is the line where they touch. A blob with a single maximum
    comes back whole, so this is not a splitter that cuts everything -- it
    cuts what has two centres.

    Seeds no nearer than the radius of an object of ``min_area``
    (:math:`\\sqrt{A/\\pi}`), so the smallest object the caller is willing to
    keep cannot itself be split in two; a blob the peak finder gave no seed
    at all gets one at its own deepest pixel, or it would be dropped.

    This is the tail :func:`_classical_region_labels` has always ended with,
    which :func:`_otsu_instances` now reaches too -- one recipe, so the Otsu
    mode in the magnifier and the Otsu detect button cut a pair of touching
    cells the same way.

    :param binary: truthy where there is foreground.
    :param min_area: objects smaller than this are dropped, and the seed
        spacing is taken from it.
    :returns: int32 labels 1..N, all zero for an empty ``binary``.
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    ndimage = _ndimage()
    mask = np.asarray(binary, dtype=bool)
    empty = np.zeros(mask.shape, dtype=np.int32)
    if not mask.any():
        return empty
    distance = ndimage.gaussian_filter(
        ndimage.distance_transform_edt(mask), 1.0)
    components, count = ndimage.label(mask, structure=_full_connectivity(mask.ndim))
    spacing = max(2, int(np.sqrt(max(int(min_area), 12) / np.pi)))
    peaks = peak_local_max(distance, min_distance=spacing, labels=components,
                           exclude_border=False)
    markers = np.zeros(mask.shape, dtype=np.int32)
    for index, point in enumerate(peaks, start=1):
        markers[tuple(point)] = index
    seeded = {int(v) for v in np.unique(components[markers > 0])}
    next_marker = len(peaks) + 1
    for component in range(1, count + 1):
        if component in seeded:
            continue
        where = int(np.argmax(np.where(components == component, distance, -1.0)))
        markers.flat[where] = next_marker
        next_marker += 1
    labels = watershed(-distance, markers, mask=mask)
    areas = np.bincount(labels.ravel())
    keep = areas >= max(1, int(min_area))
    keep[0] = False
    lookup = np.zeros(areas.size, dtype=np.int32)
    lookup[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lookup[labels]


def _drop_border_objects(labels: np.ndarray) -> np.ndarray:
    """Drop every object touching the edge of the field, and renumber.

    An object the frame cut through has an area and a mean intensity that
    are properties of where the frame fell, not of the object, so a
    detection meant to be measured is better off without it.

    :param labels: int label image.
    :returns: int32 labels 1..N holding only the objects clear of the edge.
    """
    lab = np.asarray(labels)
    if not lab.size:
        return np.zeros(lab.shape, dtype=np.int32)
    edge = set()
    for axis in range(lab.ndim):
        for index in (0, -1):
            edge.update(int(v) for v in np.unique(np.take(lab, index, axis)))
    edge.discard(0)
    keep = np.ones(int(lab.max()) + 1, dtype=bool)
    keep[0] = False
    for value in edge:
        keep[value] = False
    lookup = np.zeros(keep.size, dtype=np.int32)
    lookup[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lookup[lab]


#: How a propagation decides where to stop growing, as
#: ``name -> what the number beside it means``. See
#: :func:`maxima_propagate_instances`.
PROPAGATE_STOPS: Dict[str, str] = {
    "seed_fraction": "a fraction of THIS seed's own peak value",
    "absolute": "an absolute intensity",
    "percentile": "a percentile of the whole image",
    "threshold": "a global threshold algorithm's level",
}


class PropagateResult(NamedTuple):
    """What one maxima-and-propagate run found.

    :param labels: the objects, one label per seed that survived.
    :param seeds: how many local maxima were found. THE NUMBER THE USER
        TUNES AGAINST: too many and the minimum distance or the seed level
        is too low, too few and an object has no centre to grow from, and
        neither is visible from the objects alone.
    :param level: the intensity the growth stopped at, for the stop rules
        that have ONE -- absolute, percentile and a global threshold. None
        for ``seed_fraction``, which has a different level per object and
        so has no single number to report.
    """

    labels: np.ndarray
    seeds: int
    level: Optional[float]


class PrimarySecondaryReport(NamedTuple):
    """Label relationships between a primary mask and a secondary mask.

    All fields contain sorted tuples of Python integer IDs; 0 is excluded.
    ``matched_ids`` occur in both masks. ``missing_secondary_ids`` occur
    only in the primary mask, and ``orphan_secondary_ids`` only in the
    secondary mask. ``incomplete_primary_ids`` are matched IDs whose
    secondary does not contain every pixel of its primary. Matched IDs
    without any secondary pixels outside their own primary are listed in
    ``unexpanded_primary_ids``; these can indicate a threshold that stopped
    growth immediately. A match alone does not prove correct cell boundaries.

    :ivar primary_ids: nonzero IDs present in the primary mask.
    :ivar secondary_ids: nonzero IDs present in the secondary mask.
    :ivar matched_ids: IDs present in both masks.
    :ivar missing_secondary_ids: primary IDs absent from the secondary mask.
    :ivar orphan_secondary_ids: secondary IDs absent from the primary mask.
    :ivar incomplete_primary_ids: matched IDs whose secondary omits primary pixels.
    :ivar unexpanded_primary_ids: matched IDs with no growth beyond their primary.
    """

    primary_ids: Tuple[int, ...]
    secondary_ids: Tuple[int, ...]
    matched_ids: Tuple[int, ...]
    missing_secondary_ids: Tuple[int, ...]
    orphan_secondary_ids: Tuple[int, ...]
    incomplete_primary_ids: Tuple[int, ...]
    unexpanded_primary_ids: Tuple[int, ...]


class SecondaryResult(NamedTuple):
    """Secondary labels, their primary relationships and the common stop level.

    ``labels`` has the primary mask's dtype, shape and retained object IDs.
    ``relationships`` includes primaries removed by minimum-area filtering.
    ``level`` is None for the per-primary peak-ratio rule or an empty primary
    mask; otherwise it is the common threshold in processed intensity units.

    :ivar labels: secondary label array retaining primary IDs and dtype.
    :ivar relationships: primary/secondary identity report after filtering.
    :ivar level: shared stop threshold, or None when no common threshold applies.
    """

    labels: np.ndarray
    relationships: PrimarySecondaryReport
    level: Optional[float]


def _primary_label_image(labels: np.ndarray, name: str) -> np.ndarray:
    """Validate a nonempty 2-D, nonnegative integer label image without casting."""
    values = np.asarray(labels)
    if values.ndim != 2 or not values.size:
        raise ValueError(f"{name} must be a nonempty 2-D label image.")
    if values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError(f"{name} must contain nonnegative integer labels.")
    return values


def primary_secondary_report(primary: np.ndarray,
                             secondary: np.ndarray) -> PrimarySecondaryReport:
    """Report shared, missing, orphaned and incompletely enclosed object IDs.

    :param primary: nonempty 2-D nonnegative integer primary labels.
    :param secondary: secondary labels of the same shape. Sparse and uint64
        IDs are compared exactly, including values above signed int64.
    :returns: :class:`PrimarySecondaryReport`; no array is modified. An ID
        match means the IDs agree, not that spatial overlap was used to
        infer or repair a parent assignment.
    :raises ValueError: mismatched shapes or invalid label arrays.
    """
    first = _primary_label_image(primary, "Primary mask")
    second = _primary_label_image(secondary, "Secondary mask")
    if first.shape != second.shape:
        raise ValueError("Primary and secondary masks must have the same shape.")
    primary_ids = {int(value) for value in np.unique(first) if value}
    secondary_ids = {int(value) for value in np.unique(second) if value}
    matched = primary_ids & secondary_ids
    at_primary = first > 0
    primary_values = first[at_primary].astype(np.uint64)
    secondary_values = second[at_primary].astype(np.uint64)
    incomplete = {int(value) for value in np.unique(
        primary_values[primary_values != secondary_values])} & matched
    at_secondary = second > 0
    secondary_values = second[at_secondary].astype(np.uint64)
    primary_values = first[at_secondary].astype(np.uint64)
    expanded = {int(value) for value in np.unique(
        secondary_values[primary_values != secondary_values])}
    return PrimarySecondaryReport(
        tuple(sorted(primary_ids)), tuple(sorted(secondary_ids)),
        tuple(sorted(matched)), tuple(sorted(primary_ids - secondary_ids)),
        tuple(sorted(secondary_ids - primary_ids)), tuple(sorted(incomplete)),
        tuple(sorted(matched - expanded)))


def secondary_object_instances(
        image: np.ndarray, primary: np.ndarray, *, sigma: float = 2.0,
        stop: str = "threshold", stop_value: float = 0.4,
        stop_algorithm: str = "otsu", min_area: int = 0,
        fill_holes: bool = True, growth: str = "intensity") -> SecondaryResult:
    """Grow secondary objects from labelled primaries with a seeded watershed.

    Every positive primary label is a marker, including all its pixels.
    Intensity growth follows the negative, optionally Gaussian-smoothed image.
    Distance growth floods a flat surface from the primary pixels. Common
    stop thresholds constrain four-connected paths around excluded pixels;
    seed_fraction trims after growth. This is not unrestricted Euclidean
    nearest-primary assignment.
    Neither mode is CellProfiler's distance/intensity Propagation algorithm.

    The four rules in :data:`PROPAGATE_STOPS` use processed intensities.
    ``absolute``, ``percentile`` and ``threshold`` restrict growth with a
    common foreground mask. ``seed_fraction`` trims each watershed basin
    at a fraction of the brightest processed pixel inside its primary.
    That ratio is not a quantile and depends on background offset. With a
    dark nucleus in a cytoplasmic channel, use a common threshold instead
    of a nucleus-relative peak ratio.

    Primary pixels are always included before minimum-area filtering, even
    below the threshold. Hole filling can also restore below-threshold
    pixels. Filtering can remove a whole secondary together with its seed;
    the missing ID is reported. Remaining labels retain their primary IDs
    exactly, without splitting or renumbering disconnected components.
    Sparse IDs use compact internal markers, never arrays sized by max ID.

    :param image: finite nonempty 2-D intensities, converted to float32.
        The caller supplies normalization, background correction or inversion.
    :param primary: same-shape nonnegative integer primary labels; 0 means
        background. The returned labels retain this dtype and these IDs.
    :param sigma: finite nonnegative Gaussian sigma in pixels; 0 disables
        smoothing. Smoothing affects growth and threshold estimation.
    :param stop: one of :data:`PROPAGATE_STOPS`, default ``"threshold"``.
    :param stop_value: intensity for ``absolute``, percentile in [0,100] for
        ``percentile``, or a fraction in [0,1] for ``seed_fraction``. Ignored
        by ``threshold``. Fractions require nonnegative processed intensities.
    :param stop_algorithm: global threshold algorithm, default ``"otsu"``;
        read only for ``threshold``. The full supplied image determines it.
    :param min_area: minimum secondary area after hole filling; 0 disables
        filtering. The whole primary footprint counts toward the area.
    :param fill_holes: fill enclosed background pixels per label before
        filtering. Does not overwrite another primary's labelled pixels.
    :param growth: ``intensity`` (default) uses negative image intensity;
        ``distance`` floods a flat surface. Common stop rules constrain paths;
        the primary-relative fraction trims after growth.
        Both retain the same stop rules and exact primary IDs. Distance can
        help when bright structures attract an intensity basin across cells.
    :returns: :class:`SecondaryResult`, including ID relationship diagnostics.
        Empty primary masks yield an empty result and no common stop level.
    :raises ValueError: invalid images, labels, shape, sigma, stop rule,
        rule-specific stop value or negative minimum area.
    """
    markers = _primary_label_image(primary, "Primary mask")
    if growth not in ('intensity', 'distance'):
        raise ValueError("Secondary growth must be intensity or distance.")
    values = np.asarray(image, dtype=np.float32)
    if values.shape != markers.shape or not np.isfinite(values).all():
        raise ValueError("Image must be finite and match the primary mask's shape.")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Gaussian sigma must be finite and nonnegative.")
    if str(stop) not in PROPAGATE_STOPS:
        raise ValueError(f"Unknown propagation stop rule: {stop!r}.")
    minimum = float(min_area)
    if not np.isfinite(minimum) or minimum < 0:
        raise ValueError("Minimum area must be finite and nonnegative.")
    if stop != "threshold":
        value = float(stop_value)
        if not np.isfinite(value):
            raise ValueError("Stop value must be finite.")
        if stop == "percentile" and not 0 <= value <= 100:
            raise ValueError("Stop percentile must be between 0 and 100.")
        if stop == "seed_fraction" and (not 0 <= value <= 1 or values.min() < 0):
            raise ValueError("Seed fraction needs a value in [0,1] and nonnegative intensities.")
    ids = np.unique(markers)
    ids = ids[ids > 0]
    dense = np.zeros(markers.shape, dtype=np.int32)
    foreground = markers > 0
    dense[foreground] = np.searchsorted(ids, markers[foreground]) + 1
    blurred = _ndimage().gaussian_filter(values, sigma) if sigma > 0 else values
    grown = _grow_markers(
        blurred, dense, stop=stop, stop_value=stop_value,
        stop_algorithm=stop_algorithm, min_area=int(minimum),
        fill_holes=fill_holes, keep_markers=True, relabel=False, growth=growth)
    lookup = np.concatenate((np.zeros(1, dtype=markers.dtype), ids))
    labels = lookup[grown.labels]
    return SecondaryResult(labels, primary_secondary_report(markers, labels),
                           grown.level)


def maxima_propagate_instances(
        image: np.ndarray, *, sigma: float = 2.0, min_distance: int = 10,
        seed_level: float = 90.0, seed_level_is_percentile: bool = True,
        exclude_border: bool = False, stop: str = "seed_fraction",
        stop_value: float = 0.4, stop_algorithm: str = "otsu",
        min_area: int = 0, fill_holes: bool = True) -> PropagateResult:
    """Segment bright objects with local maxima and an intensity watershed.

    Convert the field to float32, optionally blur it, and find centres with
    :func:`skimage.feature.peak_local_max`. Use those centres as markers for
    :func:`skimage.segmentation.watershed` on the negative blurred image.
    Distinct centres can split touching objects; noise can create extra
    centres, while smoothing or large centre spacing can remove real ones.
    No existing primary-object mask is accepted. This implementation has
    no CellProfiler propagation cost or distance/intensity weighting.

    The four stop rules operate on the blurred values:

    * ``seed_fraction`` first partitions the entire image by watershed,
      then retains pixels at or above ``stop_value`` times their basin's
      seed intensity. This is an intensity ratio, not a quantile. Adding a
      background offset changes the relative cut; equal measurements of
      bright and dim objects are not guaranteed. Trimming can leave
      disconnected pieces with the same label.
    * ``absolute`` restricts the watershed to pixels at or above
      ``stop_value``, in the input's intensity units.
    * ``percentile`` uses that percentile of all blurred input pixels as
      the common threshold. Changing the crop can change this level.
    * ``threshold`` obtains the common level from ``stop_algorithm`` and
      ignores ``stop_value``.

    Fill holes per label if requested, then discard labels smaller than
    ``min_area`` and renumber survivors. Hole filling can restore pixels
    below the selected intensity cut. The seed count is recorded before
    these operations and can exceed the number of surviving objects.

    Input preparation is the caller's responsibility: this function does
    not normalize intensities, subtract background, or invert dark objects.
    Use finite 2-D values; NaNs and infinities are not sanitized. Make Masks
    supplies the processed field or crop after its selected enhancements.
    Absolute levels and peak ratios therefore depend on that preparation.

    :param image: nonempty 2-D intensity array, converted to float32 without
        range rescaling. Output coordinates and shape match this array.
    :param sigma: Gaussian standard deviation in pixels; default 2.0.
        Positive values smooth both seed finding and growth; zero disables
        blur. Increasing it can suppress noise peaks or merge real peaks.
        Make Masks offers 0 to 50; the direct API also skips negative values.
    :param min_distance: centre separation in pixels; default 10. Passed
        to peak finding as ``max(1, int(min_distance))``, using its default
        Chebyshev distance. Increasing it suppresses nearby candidate seeds;
        reducing it can split an object into several detections. Make Masks
        offers 1 to 500.
    :param seed_level: default 90.0. Candidate maxima must exceed this
        intensity, or the intensity at this percentile when
        ``seed_level_is_percentile`` is true. Percentiles must be between
        0 and 100. Increasing the floor excludes dimmer candidate centres;
        it does not directly set the final object boundary.
    :param seed_level_is_percentile: default true. Compute the seed floor
        from all blurred input pixels; false uses an absolute intensity.
        A crop and a whole field can yield different percentile floors.
    :param exclude_border: default false. If true, exclude candidate centres
        within the effective ``min_distance`` of the input edge. This does
        not remove every object whose grown boundary touches the edge.
    :param stop: default ``"seed_fraction"``; one of
        :data:`PROPAGATE_STOPS`, with the behavior described above.
    :param stop_value: default 0.4. For ``seed_fraction``, use a ratio from
        0 to 1 with nonnegative intensities; increasing it removes dimmer
        basin pixels before hole filling. For ``absolute``, use an intensity;
        for ``percentile``, use 0 to 100. Higher common thresholds shrink
        the eligible mask. Ignored for ``threshold``. The API does not clip
        ratios or absolute values; the GUI number box alone does not enforce
        rule-specific limits.
    :param stop_algorithm: default ``"otsu"``; a key of
        :data:`GLOBAL_THRESHOLDS`, read only for ``stop="threshold"``.
        The threshold is estimated from the blurred field or crop.
    :param min_area: default 0, disabling size removal. Labels with fewer
        than ``int(min_area)`` pixels after hole filling are discarded.
        Increasing it removes small labels without merging touching ones.
    :param fill_holes: default true. Fill enclosed background pixels per
        label before size filtering. False preserves those holes.
    :returns: :class:`PropagateResult` containing an int32 label array
        (0 is background, surviving labels are 1 through N), the original
        seed count, and the common stop level. The level is None for
        ``seed_fraction`` or when no seeds were found. A constant image or
        an overly high seed floor can return all-zero labels and zero seeds;
        an empty stop mask or size filtering can remove every seeded object.
    :raises ValueError: for an unknown stop rule, an empty or non-2-D image,
        an out-of-range percentile when evaluated, or an unknown global
        threshold algorithm when that rule is evaluated.

    For example, ``maxima_propagate_instances(image, sigma=2,
    min_distance=10, seed_level=90, stop="seed_fraction", stop_value=0.4,
    min_area=20)`` retains each basin above 40 percent of its seed intensity
    before filling holes and removing labels smaller than 20 pixels.
    """
    from skimage.feature import peak_local_max

    if str(stop) not in PROPAGATE_STOPS:
        raise ValueError(
            f"{stop!r} is not a propagation stop rule; the ones there are: "
            f"{sorted(PROPAGATE_STOPS)}.")
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not values.size:
        raise ValueError("Propagation needs a 2-D image; this one is empty.")
    empty = np.zeros(values.shape, dtype=np.int32)
    blurred = (_ndimage().gaussian_filter(values, float(sigma))
               if float(sigma) > 0.0 else values)

    floor = (float(np.percentile(blurred, float(seed_level)))
             if seed_level_is_percentile else float(seed_level))
    coordinates = peak_local_max(
        blurred, min_distance=max(1, int(min_distance)), threshold_abs=floor,
        exclude_border=max(1, int(min_distance)) if exclude_border else False)
    if not len(coordinates):
        return PropagateResult(empty, 0, None)

    markers = np.zeros(values.shape, dtype=np.int32)
    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)

    return _grow_markers(
        blurred, markers, stop=stop, stop_value=stop_value,
        stop_algorithm=stop_algorithm, min_area=min_area, fill_holes=fill_holes)


def _grow_markers(blurred, markers, *, stop, stop_value, stop_algorithm,
                  min_area, fill_holes, keep_markers=False,
                  relabel=True, growth="intensity") -> PropagateResult:
    """Grow compact markers with shared stop, fill and size-filter semantics."""
    from skimage.segmentation import watershed

    seeds = int(markers.max())
    empty = np.zeros(markers.shape, dtype=np.int32)
    if not seeds:
        return PropagateResult(empty, 0, None)

    level: Optional[float] = None
    surface = np.zeros_like(blurred) if growth == 'distance' else -blurred
    if stop == "seed_fraction":
        grown = watershed(surface, markers)
        peaks = np.zeros(seeds + 1, dtype=np.float32)
        peaks[1:] = _ndimage().maximum(blurred, markers, np.arange(1, seeds + 1))
        keep = blurred >= peaks[grown] * float(stop_value)
        labels = np.where(keep, grown, 0).astype(np.int32)
    else:
        if stop == "absolute":
            level = float(stop_value)
        elif stop == "percentile":
            level = float(np.percentile(blurred, float(stop_value)))
        else:
            level = _global_level(blurred, stop_algorithm)
        mask = blurred >= level
        if keep_markers:
            mask |= markers > 0
        if not mask.any():
            return PropagateResult(empty, seeds, level)
        labels = np.asarray(watershed(surface, markers, mask=mask),
                            dtype=np.int32)

    if keep_markers:
        labels[markers > 0] = markers[markers > 0]
    if fill_holes:
        labels = fill_label_holes(labels)
    return PropagateResult(_drop_small_labels(labels, min_area, relabel=relabel),
                           seeds, level)


def fill_label_holes(labels: np.ndarray) -> np.ndarray:
    """Close the holes inside each object, keeping every id it already had.

    THE ONE HOLE FILLER FOR LABEL IMAGES. :func:`fill_holes`, the
    propagation step above and :func:`spacr.utils.fill_holes_in_mask` (the
    Cellpose ``fill_in`` step) all come here. Filling the binary foreground
    and labelling it afresh with ``ndimage.label`` is what this replaces: it
    makes every pair of TOUCHING objects one object, so a field of 74
    adjacent cells came back as 8. Here nothing is relabelled.

    ``binary_fill_holes`` over the whole foreground would also fill the gap
    BETWEEN objects that happen to ring a piece of background, so each label
    is filled on its own and written back only onto background: a hole never
    overwrites another object's pixels. Each label is filled inside its own
    bounding box (:func:`scipy.ndimage.find_objects`), which is exact --
    background on the edge of the box touches the outside of the box, where
    this label has no pixels, so it can never be one of its holes -- and
    keeps a 2048 x 2048 field of hundreds of objects to a fraction of a
    second. Smaller boxes claim first, so a hole inside a ring that itself
    sits inside another ring goes to the inner one.

    :param labels: a 2-D (or N-D) non-negative integer label image. A
        boolean mask is labelled by connectivity first, since it carries no
        ids to keep.
    :returns: an array of the input's dtype (``int32`` for a boolean input)
        in which every object keeps its id and its holes carry that id.
    """
    ndimage = _ndimage()
    arr = np.asarray(labels)
    if arr.dtype == bool:
        arr, _count = ndimage.label(arr)
        arr = arr.astype(np.int32)
    out = arr.copy()
    if not out.size or not out.any():
        return out
    if not np.issubdtype(out.dtype, np.integer):
        raise ValueError("fill_label_holes needs an integer label image.")
    boxes = _label_boxes(out)
    order = sorted(
        boxes,
        key=lambda index: int(np.prod([s.stop - s.start for s in boxes[index]])))
    for index in order:
        box = boxes[index]
        value = index
        window = out[box]
        own = window == value
        holes = ndimage.binary_fill_holes(own) & ~own & (window == 0)
        if holes.any():
            window[holes] = value
    return out


def _drop_small_labels(labels: np.ndarray, min_area: int, *,
                       relabel: bool = True) -> np.ndarray:
    """Remove objects under ``min_area`` and renumber the rest from 1.

    Set ``relabel=False`` to retain marker IDs after dropping small labels.
    Inputs use compact integer labels; sparse external IDs must be mapped
    before calling this helper. Renumbering by remapping and NOT by
    re-labelling the foreground: two
    objects that touch are two objects, and connected-components would make
    them one again.
    """
    out = np.asarray(labels, dtype=np.int32)
    counts = np.bincount(out.ravel())
    if int(min_area) > 0:
        small = counts < int(min_area)
        small[0] = True
        out = np.where(small[out], 0, out)
    if not relabel:
        return out
    present = np.unique(out)
    present = present[present > 0]
    remap = np.zeros(int(out.max()) + 1, dtype=np.int32)
    remap[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    return remap[out]


def _classical_region_labels(region: np.ndarray, *, sensitivity: float = 0.0,
                             bright: bool = True,
                             min_area: int = 0,
                             correction: float = 1.0,
                             smoothing: float = _CLASSICAL_SMOOTHING,
                             fill_holes: bool = True,
                             split_touching: bool = True,
                             algorithm: str = "otsu",
                             window: int = 51,
                             local_k: float = 0.2) -> np.ndarray:
    """Threshold one magnifier region and split the objects that touch.

    The Otsu magnifier mode -- formerly named ``classical`` --
    and the fallback whenever a model cannot run: it needs nothing beyond
    scikit-image. The region is smoothed (``smoothing``, by default
    :data:`_CLASSICAL_SMOOTHING`) and then cut one of two ways. A region
    that holds two clear populations
    (:data:`_CLASSICAL_MIN_SEPARATION`) is cut at Otsu's level, moved by
    ``sensitivity`` steps of :data:`_CLASSICAL_SENSITIVITY_STEP` of its
    stretched range; any other region is background with at most a few dim
    objects, and is cut :data:`_CLASSICAL_NOISE_SIGMAS` robust deviations
    above its median. The foreground is opened, hole-filled and split with a
    watershed on its distance transform.

    THE THREE SWITCHES DEFAULT TO WHAT THIS DID BEFORE THEY EXISTED, so the
    call the magnifier has always made comes back the mask it has always
    come back. They are here because the Otsu settings panel shows them,
    and the panel drives both this and :func:`_otsu_instances`.

    :param region: 2-D intensity crop.
    :param sensitivity: 0 is the default cut; positive takes in dimmer
        pixels, negative keeps only the brightest.
    :param bright: objects are brighter than background; False takes the
        dark side, for brightfield.
    :param min_area: objects smaller than this are dropped. It also sets how
        far apart two seeds must be, so an object of the smallest allowed
        size is not split in two.
    :param correction: Otsu's level is multiplied by this before
        ``sensitivity`` moves it (the threshold correction; see
        :func:`_otsu_instances`). Above 1 is stricter. A region with no two
        clear populations is cut at its noise floor, which is not Otsu's
        level, and this does not apply to it.
    :param smoothing: Gaussian sigma applied before either cut. 0 cuts the
        raw region, which finds every speckle a noisy field has.
    :param fill_holes: close the holes inside the thresholded foreground
        before it is labelled. Off leaves a dim nucleus as a ring.
    :param split_touching: cut each blob at the ridge between two centres
        (:func:`_split_touching_objects`). Off labels each blob whole, so a
        pair of touching cells arrives as one object.
    :param algorithm: which of :data:`GLOBAL_THRESHOLDS` or
        :data:`LOCAL_THRESHOLDS` finds the level. ``otsu`` is the default
        and is what this did before there were others. A LOCAL ALGORITHM
        SKIPS THE TWO-POPULATION TEST below, because that test is a
        judgement about a whole region's histogram and a local algorithm
        does not take one.
    :param window: the window a local algorithm measures in, in pixels.
    :param local_k: Sauvola's and Niblack's ``k``.
    :returns: int32 labels 1..N shaped like ``region``; all zero for a
        region with nothing above its noise.
    """
    ndimage = _ndimage()
    values = np.asarray(region, dtype=np.float32)
    empty = np.zeros(values.shape, dtype=np.int32)
    if values.ndim != 2 or values.size < 4:
        return empty
    sigma = max(0.0, float(smoothing))
    smooth = (ndimage.gaussian_filter(values, sigma) if sigma > 0.0
              else values)
    if not bright:
        smooth = -smooth
    lo, hi = (float(v) for v in np.percentile(smooth, (1.0, 99.8)))
    if hi <= lo:
        return empty
    stretched = np.clip((smooth - lo) / (hi - lo), 0.0, 1.0)
    name = str(algorithm or "otsu")
    if name in LOCAL_THRESHOLDS:
        levels = _local_level_map(stretched, name, window=window, k=local_k)
        foreground = stretched > levels * float(correction)
        return _finish_region_binary(foreground, ndimage, empty,
                                     fill_holes=fill_holes,
                                     split_touching=split_touching,
                                     min_area=min_area)
    level = _global_level(stretched, name)
    upper = stretched > level
    share = float(upper.mean())
    total = float(stretched.var())
    separation = 0.0
    if 0.0 < share < 1.0 and total > 0.0:
        gap = float(stretched[upper].mean() - stretched[~upper].mean())
        separation = share * (1.0 - share) * gap * gap / total
    if separation >= _CLASSICAL_MIN_SEPARATION:
        cut = (level * float(correction)
               - _CLASSICAL_SENSITIVITY_STEP * float(sensitivity))
        foreground = stretched > cut
    else:
        centre = float(np.median(smooth))
        spread = 1.4826 * float(np.median(np.abs(smooth - centre)))
        if spread <= 0.0:
            return empty
        sigmas = max(1.0, _CLASSICAL_NOISE_SIGMAS - 0.5 * float(sensitivity))
        foreground = smooth > centre + sigmas * spread

    return _finish_region_binary(foreground, ndimage, empty,
                                 fill_holes=fill_holes,
                                 split_touching=split_touching,
                                 min_area=min_area)


def _finish_region_binary(foreground, ndimage, empty, *, fill_holes: bool,
                          split_touching: bool, min_area: int) -> np.ndarray:
    """Open, fill and label a magnifier region's foreground.

    The tail of :func:`_classical_region_labels`, in a function of its own
    because a local algorithm reaches it without passing through the
    two-population test in the middle of that one. Splitting it out is what
    keeps there being ONE description of what happens to a region's
    foreground after it has been decided.
    """
    binary = ndimage.binary_opening(foreground, structure=_EIGHT)
    if fill_holes:
        binary = ndimage.binary_fill_holes(binary)
    if not binary.any():
        return empty
    if not split_touching:
        return connected_instances(binary, min_area=min_area)
    return _split_touching_objects(binary, min_area=min_area)


def _paste_region_objects(mask: np.ndarray, labels: np.ndarray, origin, *,
                          overlap: str = "clip",
                          min_area: int = 0,
                          preserve_ids: bool = False,
                          replace_whole: bool = False) -> Tuple[np.ndarray, List[int]]:
    """Add a region's objects to ``mask`` as new objects.

    What a live-magnifier click commits. The labels arrive in the region's
    own coordinates and ``origin`` is where the region's top-left pixel sits
    in the image, so an object at ``labels[r, c]`` lands on
    ``mask[origin_y + r, origin_x + c]``. Anything that would fall outside
    the image is dropped rather than wrapped or clamped.

    :param mask: the label image to copy and add to. Its objects keep their
        ids; the dtype widens only when a new id needs it.
    :param labels: label image in region coordinates. Each distinct positive
        value is one object.
    :param origin: ``(x, y)`` of ``labels[0, 0]`` in image pixels.
    :param overlap: one of :data:`_MAGNIFIER_OVERLAP_RULES` -- what a new
        object does where the mask is already labelled. Under ``clip`` an
        object that an existing one splits in two keeps only its largest
        piece, because one id must name one object.
    :param min_area: an object left smaller than this once the rule has been
        applied is not added.
    :param preserve_ids: paste the supplied IDs, not newly allocated IDs.
        Same-ID pixels do not conflict under Clip or Skip. Disconnected
        pieces retain their shared identity. Exact-ID masks must fit uint32.
    :param replace_whole: under ``replace``, remove each overlapping old
        object in full before pasting. Same-ID pixels remain when preserving
        IDs. Under ``merge``, fuse overlapping objects into their smallest
        existing ID, retaining old pixels outside the region; a detection on
        background receives a new ID. Returned IDs include fused objects.
    :returns: ``(mask, new_ids)``. New ids start one past the mask's top id
        (:func:`next_label`) and follow the incoming labels' order, so they
        cannot collide with any id the mask holds. Nothing added returns a
        copy of ``mask`` and an empty list.
    :raises ValueError: for an unknown ``overlap`` rule.
    """
    if overlap not in _MAGNIFIER_OVERLAP_RULES:
        raise ValueError(
            f"overlap must be one of {_MAGNIFIER_OVERLAP_RULES}, "
            f"not {overlap!r}")
    incoming = np.asarray(labels)
    if preserve_ids:
        incoming = canonical_labels(incoming, preserve_ids=True)
        mask = canonical_labels(mask, preserve_ids=True)
    height, width = mask.shape[:2]
    ox, oy = int(origin[0]), int(origin[1])
    x0, y0 = max(0, ox), max(0, oy)
    x1 = min(width, ox + incoming.shape[1])
    y1 = min(height, oy + incoming.shape[0])
    if x1 <= x0 or y1 <= y0:
        return mask.copy(), []
    incoming = incoming[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    if not incoming.any():
        return mask.copy(), []
    occupied = np.asarray(mask)[y0:y1, x0:x1] > 0
    if preserve_ids:
        occupied &= np.asarray(mask)[y0:y1, x0:x1] != incoming
    kept = _surviving_region_objects(incoming, occupied, overlap=overlap,
                                    min_area=min_area, preserve_ids=preserve_ids)
    values = [int(v) for v in np.unique(kept) if int(v) > 0]
    if not values:
        return mask.copy(), []
    out = mask.astype(np.int64, copy=True)
    window = out[y0:y1, x0:x1]
    if overlap == "merge":
        added = []
        free_ids = _available_labels(mask)
        for value in values:
            body = kept == value
            touching = np.unique(window[body])
            touching = touching[touching > 0]
            if touching.size:
                target = int(touching[0])
                if touching.size > 1:
                    out[np.isin(out, touching[1:])] = target
            else:
                target = value if preserve_ids else next(free_ids)
            window[body] = target
            added.append(target)
        added = list(dict.fromkeys(int(out[y0:y1, x0:x1][kept == value][0])
                                   for value in values))
        return _fit_label_width(out, mask), added
    if overlap == "replace" and replace_whole:
        body = kept > 0
        conflicts = body & (window != kept) if preserve_ids else body
        replaced = np.unique(window[conflicts])
        replaced = replaced[replaced > 0]
        if replaced.size:
            out[np.isin(out, replaced)] = 0
    if preserve_ids:
        body = kept > 0
        window[body] = kept[body]
        return _fit_label_width(out, mask), values
    free_ids = _available_labels(mask)
    added = [next(free_ids) for value in values]
    body = kept > 0
    window[body] = np.asarray(added, dtype=np.int64)[np.searchsorted(values, kept[body])]
    return _fit_label_width(out, mask), added


def _largest_piece_of_each(labels: np.ndarray) -> np.ndarray:
    """Keep one connected piece of every object: the largest.

    One id must name one object, so an object an existing one has split in
    two cannot go into the mask as two islands under one label.

    A piece is a connected run of ONE id. :func:`skimage.measure.label` is
    what says so and :func:`scipy.ndimage.label` is not: the latter would
    take two different objects that touch as one piece, and the largest
    piece of a pair is not the largest piece of either.

    Ties go to the piece whose topmost-leftmost pixel comes first, which is
    what a per-object ``argmax`` over the areas used to pick.

    :param labels: a label image; 0 is background.
    :returns: the same image with every object's smaller pieces set to 0.
    """
    from skimage.measure import label as label_regions

    pieces = label_regions(labels, connectivity=2, background=0)
    count = int(pieces.max())
    if count <= 1:
        return labels
    areas = np.bincount(pieces.ravel(), minlength=count + 1)
    areas[0] = 0
    flat_pieces, flat_labels = pieces.ravel(), labels.ravel()
    inside = flat_pieces > 0
    owner = np.zeros(count + 1, dtype=np.int64)
    owner[flat_pieces[inside]] = flat_labels[inside]
    order = np.lexsort((-np.arange(count + 1), areas[:count + 1]))
    best = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    best[owner[order]] = order
    chosen = best[1:]
    survives = np.zeros(count + 1, dtype=bool)
    survives[chosen[chosen > 0]] = True
    return np.where(survives[pieces], labels, 0)


def _surviving_region_objects(labels: np.ndarray, occupied: np.ndarray, *,
                             overlap: str = "clip",
                             min_area: int = 0,
                             preserve_ids: bool = False) -> np.ndarray:
    """What is left of a region's objects once the Overlap rule has run.

    The live magnifier's Overlap rule and Min area in one place, so the box
    can draw what a click would add and the click can add exactly that. Both
    read this; nothing applies the rule twice and nothing can drift.

    IT IS COUNTED ONCE OVER THE PIXELS, NOT ONCE PER OBJECT. The loop this
    replaced ran a whole-region comparison and, under ``clip``, a whole
    connected-component pass for EVERY object, which is fine for the ten
    objects in a 128 px box and is not fine for the largest box the
    magnifier allows: on
    a 2048 px region holding 500 objects it took 6.1 s for ``clip`` and
    0.82 s for ``skip``, on the GUI thread, with the user's click waiting on
    it. The same work is 46 ms and 33 ms here.

    :param labels: the objects offered, 0 for background.
    :param occupied: where the mask already has an object, shaped like
        ``labels``.
    :param overlap: ``clip`` keeps only each object's unlabelled pixels (and
        only its largest piece, because one id names one object), ``skip``
        leaves out any object that touches one already there, ``replace``
        keeps everything. ``merge`` also keeps the whole detection, letting
        the paste boundary fuse its existing identities.
    :param min_area: an object left smaller than this by the rule does not
        survive. 0 and 1 both mean "at least one pixel".
    :param preserve_ids: retain disconnected pieces sharing an ID; ``occupied``
        must exclude existing same-ID pixels. Validate labels as uint32 IDs.
    :returns: a copy of ``labels`` with everything the rule takes away set
        to 0. The surviving objects keep the ids they came in with.
    :raises ValueError: for an unknown ``overlap`` rule.
    """
    if overlap not in _MAGNIFIER_OVERLAP_RULES:
        raise ValueError(
            f"overlap must be one of {_MAGNIFIER_OVERLAP_RULES}, "
            f"not {overlap!r}")
    incoming = np.asarray(labels)
    if preserve_ids:
        incoming = canonical_labels(incoming, preserve_ids=True)
    taken = np.asarray(occupied, dtype=bool)
    kept = np.where(incoming > 0, incoming, 0).astype(np.int64)
    if overlap == "skip":
        touching = np.unique(kept[taken])
        touching = touching[touching > 0]
        if touching.size:
            kept[np.isin(kept, touching)] = 0
    elif overlap == "clip":
        kept[taken] = 0
        if not preserve_ids:
            kept = _largest_piece_of_each(kept)
    floor = max(1, int(min_area))
    if floor > 1 and kept.any():
        areas = np.bincount(kept.ravel())
        big = areas >= floor
        big[0] = False
        kept = np.where(big[kept], kept, 0)
    return kept



#: How many pixels the wand may EXAMINE per pixel it is allowed to change.
#:
#: The budget is on WORK, and it has to accommodate a case the change
#: budget deliberately does not bound: re-wanding an object the mask
#: already owns, with a wider tolerance, to grow it. Crossing owned
#: pixels costs nothing against ``max_pixels`` ON PURPOSE -- otherwise
#: the second click would stop at the first owned pixel and do nothing --
#: so the walk is bounded here instead.
VISIT_BUDGET_FACTOR = 4

#: The smallest visit budget, whatever ``max_pixels`` is, and the number
#: that actually matters.
#:
#: It has to be BIGGER THAN ANY OBJECT SOMEBODY RE-WANDS and smaller than
#: a frame. A hundred thousand is a 316x316 region, larger than any single
#: object in a field of cells, and it caps the pathological case -- a
#: mis-click on uniform background -- at about 0.7 s instead of the 4.8 s
#: measured on an 800x800 field or the half-minute at 2048x2048.
#:
#: It cannot be tight. A pixel count cannot tell "usefully growing a large
#: object" from "clicked on the background", because both walk until
#: tolerance stops them and on a uniform field neither does. So this is
#: set to keep every real fill working and to make the wrong click a
#: hitch rather than a hang, which is the honest trade rather than a
#: pretence that the two can be separated.
VISIT_BUDGET_FLOOR = 100_000


def magic_wand(
    image: np.ndarray,
    mask: np.ndarray,
    seed_x: int,
    seed_y: int,
    tolerance: float,
    max_pixels: int = 100_000,
    action: str = "add",
) -> np.ndarray:
    """BFS flood-fill from (seed_x, seed_y) filling pixels whose intensity
    is within `tolerance` (L2 distance) of the seed. Writes 255 (add) or
    0 (erase) into the returned mask copy.

    :param image: intensity image, 2-D or with channels on the last axis;
        distances are taken over the channel values.
    :param mask: mask with the same height and width as ``image``; it is
        copied, not modified.
    :param seed_x: seed column in pixels.
    :param seed_y: seed row in pixels. A seed outside the image returns
        ``mask`` unchanged.
    :param tolerance: largest distance from the seed value that still fills,
        in the image's own intensity units; see :func:`relative_tolerance`.
    """
    if not (0 <= seed_y < image.shape[0] and 0 <= seed_x < image.shape[1]):
        return mask
    out = mask.copy()
    initial = image[seed_y, seed_x].astype(np.float32)
    visited = np.zeros(image.shape[:2], dtype=bool)
    q = deque([(seed_x, seed_y)])
    added = 0
    examined = 0
    visit_budget = max(VISIT_BUDGET_FACTOR * max_pixels,
                       max_pixels + VISIT_BUDGET_FLOOR)
    fill_val = 255 if action == "add" else 0
    while q and added < max_pixels and examined < visit_budget:
        cx, cy = q.popleft()
        if not (0 <= cx < image.shape[1] and 0 <= cy < image.shape[0]):
            continue
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        examined += 1
        cur = image[cy, cx].astype(np.float32)
        if float(np.linalg.norm(cur - initial)) > tolerance:
            continue
        if out[cy, cx] == 0 and action == "add":
            added += 1
        elif out[cy, cx] > 0 and action == "erase":
            added += 1
        out[cy, cx] = fill_val
        if added >= max_pixels:
            break
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and not visited[ny, nx]:
                q.append((nx, ny))
    return out



class VolumeMaskSession:
    """Edit native spatial labels using the same boundary as the 3-D gate view.

    Coordinates are physical X/Y/Z distances from the image's outer origin;
    voxel centres lie at ``(index + 0.5) * spacing``. Arrays retain their
    declared spatial axis order, source dtype and existing object identities.
    At most 16,777,216 voxels and 256 MiB of source intensities are accepted;
    larger fields are refused explicitly. Sparse undo deltas have a combined
    64 MiB budget rather than retaining a volume for every edit.

    :param image: finite real numeric three-dimensional source intensities.
    :param mask: nonnegative integer labels of exactly the source shape.
    :param axes: a permutation of the three spatial axes ``XYZ``.
    :param spacing: positive finite distances in that same array axis order.
    :param unit: the physical distance unit, or ``pixel`` when unspecified.
    :param channels: every channel of a multichannel volume, each shaped
        like ``image`` (which the slice views show); the 3-D view draws each
        as its own layer. Together they share the 256 MiB budget.
    """

    MAX_VOXELS = 16_777_216
    MAX_HISTORY_BYTES = 64 * 1024 * 1024

    def __init__(self, image, mask, *, axes='ZYX', spacing=(1., 1., 1.),
                 unit='pixel', channels=None):
        """Validate one spatial image and own its mutable label array.

        :param image: finite real numeric source intensities.
        :param mask: nonnegative integer object labels matching the image.
        :param axes: spatial storage-axis order, a permutation of XYZ.
        :param spacing: three positive finite distances in storage order.
        :param unit: the common distance unit.
        :param channels: every channel of a multichannel volume, each shaped
            like ``image``; ``None`` means ``image`` is the only channel.
        """
        from .i18n import tr

        image, labels = np.asarray(image), np.asarray(mask)
        axes = str(axes)
        if (image.ndim != 3 or sorted(axes) != ['X', 'Y', 'Z']
                or image.size == 0 or image.size > self.MAX_VOXELS
                or image.nbytes > 256 * 1024 * 1024
                or image.dtype.kind not in 'uif' or not np.isfinite(image).all()):
            raise ValueError(tr('3D masks require a finite numeric spatial TIFF within the volume memory budget.'))
        try:
            distances = tuple(float(value) for value in spacing)
        except (TypeError, ValueError) as exc:
            raise ValueError(tr('Voxel spacing must contain three positive finite distances.')) from exc
        if (len(distances) != 3 or not all(np.isfinite(value) and value > 0 for value in distances)
                or any(isinstance(value, (bool, np.bool_)) for value in spacing)):
            raise ValueError(tr('Voxel spacing must contain three positive finite distances.'))
        if not all(np.isfinite(float(length) * distance) for length, distance in zip(image.shape, distances)):
            raise ValueError(tr('The physical image bounds must be finite.'))
        if (labels.shape != image.shape or labels.dtype.kind not in 'ui'
                or labels.min() < 0 or labels.max() > 65535):
            raise ValueError(tr('The volume mask must contain matching nonnegative integer object labels up to 65535.'))
        stack = (image,) if channels is None else tuple(np.asarray(plane) for plane in channels)
        if (not stack or any(plane.shape != image.shape or plane.dtype.kind not in 'uif'
                             or not np.isfinite(plane).all() for plane in stack)
                or sum(plane.nbytes for plane in stack) > 256 * 1024 * 1024):
            raise ValueError(tr('3D masks require a finite numeric spatial TIFF within the volume memory budget.'))
        self.image = image
        self.channels = stack
        self.mask = np.array(labels, copy=True, order="C")
        self.axes, self.spacing, self.unit = axes, distances, str(unit)
        self.geometries = {}
        self._undo, self._redo = [], []
        self._history_bytes = 0
        self._cancelled = None
        self._editor_controls = {'anchor_density': 2, 'show_anchors': True}
        self.source_path = self.source_sha256 = None
        self.mask_path = None
        self._saved_mask_sha256 = self._saved_geometry_sha256 = None

    @property
    def xyz_spacing(self):
        """Voxel distances in physical X/Y/Z order, independent of storage."""
        return tuple(self.spacing[self.axes.index(axis)] for axis in 'XYZ')

    @property
    def limits(self):
        """The exact physical outer image bounds in X/Y/Z order."""
        return tuple((0., self.image.shape[self.axes.index(axis)] * distance)
                     for axis, distance in zip('XYZ', self.xyz_spacing))

    def next_label(self):
        """Return the first unused positive ID without renumbering any object."""
        from .i18n import tr

        occupied = set(map(int, np.unique(self.mask)))
        maximum = min(65535, np.iinfo(self.mask.dtype).max)
        for value in range(1, maximum + 1):
            if value not in occupied:
                return value
        raise ValueError(tr('No unused object identifier fits this mask dtype.'))

    def geometry_for_label(self, label):
        """Return saved geometry or an exact closed voxel-boundary mesh.

        Corner-touching, disconnected or excessively detailed labels that
        cannot satisfy the shared mesh's bounds/topology are refused rather
        than replaced by a bounding box. Merely previewing this mesh does not
        change voxels, history or saved geometry.

        :param label: an existing positive object identifier.
        :returns: a validated shared editable volume in physical coordinates.
        """
        from .i18n import tr
        from .widgets.gate_spec import EditableVolumeGate
        from ..cancellation import current_token

        token = current_token()
        if token is not None:
            token.checkpoint()

        if (not isinstance(label, Integral) or isinstance(label, (bool, np.bool_))
                or label <= 0):
            raise ValueError(tr('Choose an existing positive object label.'))
        label = int(label)
        if label in self.geometries:
            return self.geometries[label]
        volume = np.transpose(self.mask, tuple(self.axes.index(axis) for axis in 'XYZ'))
        positions = np.argwhere(volume == label)
        if label <= 0 or not len(positions):
            raise ValueError(tr('Choose an existing positive object label.'))
        corners = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
        quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4),
                 (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
        offsets = [(0, 0, -1), (0, 0, 1), (0, -1, 0),
                   (0, 1, 0), (-1, 0, 0), (1, 0, 0)]
        indices, vertices, faces, groups = {}, [], [], []
        for position in positions:
            if token is not None:
                token.checkpoint()
            for quad, offset in zip(quads, offsets):
                neighbour = position + offset
                if (np.all(neighbour >= 0) and np.all(neighbour < volume.shape)
                        and volume[tuple(neighbour)] == label):
                    continue
                shared = []
                for corner in quad:
                    key = tuple(map(int, position + corners[corner]))
                    if key not in indices:
                        indices[key] = len(vertices)
                        vertices.append(tuple(value * distance for value, distance in zip(key, self.xyz_spacing)))
                    shared.append(indices[key])
                a, b, c, d = shared
                groups.extend([len(faces) // 2] * 2)
                faces.extend([(a, b, c), (a, c, d)])
                if len(vertices) > 8192 or len(faces) > 16384:
                    raise ValueError(tr('This label has too much boundary detail for editable volume geometry.'))
        if token is not None:
            token.checkpoint()
        return EditableVolumeGate(name=str(label), x_column='X', y_column='Y',
                                  z_column='Z', vertices=tuple(vertices),
                                  faces=tuple(faces), surface_groups=tuple(groups))

    def _paint_geometry(self, output, label, gate):
        """Voxelize only the physical boundary's bounding region in 4096-point chunks."""
        import pandas as pd
        from .i18n import tr

        vertices = np.asarray(gate.vertices)
        xyz_low, xyz_high = vertices.min(axis=0), vertices.max(axis=0)
        xyz_spacing = np.asarray(self.xyz_spacing)
        xyz_shape = np.asarray([self.mask.shape[self.axes.index(axis)] for axis in 'XYZ'])
        xyz_low = np.maximum(0., xyz_low)
        xyz_high = np.minimum(xyz_shape * xyz_spacing, xyz_high)
        if np.any(xyz_high < xyz_low):
            return
        low = np.maximum(0, np.ceil(xyz_low / xyz_spacing - .5).astype(int))
        high = np.minimum(xyz_shape, np.floor(xyz_high / xyz_spacing - .5).astype(int) + 1)
        if np.any(high <= low):
            return
        order = ['XYZ'.index(axis) for axis in self.axes]
        start, shape = low[order], (high - low)[order]
        for offset in range(0, int(np.prod(shape)), 4096):
            if self._cancelled is not None and self._cancelled():
                raise InterruptedError(tr('Volume editing was cancelled; no labels were changed.'))
            count = min(4096, int(np.prod(shape)) - offset)
            coordinates = np.asarray(np.unravel_index(np.arange(offset, offset + count), tuple(shape))) + start[:, None]
            frame = pd.DataFrame({axis: (coordinates[self.axes.index(axis)] + .5) * self.xyz_spacing[i]
                                  for i, axis in enumerate('XYZ')})
            inside = gate.mask(frame)
            selected = tuple(coordinates[:, inside])
            existing = output[selected]
            if np.any((existing != 0) & (existing != label)):
                raise ValueError(tr('The edited volume overlaps another object; no labels were changed.'))
            output[selected] = label

    def apply_geometry(self, label, gate):
        """Commit one validated shared boundary and its exact mask voxels.

        Other object IDs are protected. Invalid motion, collisions or a delta
        exceeding the history budget leave arrays, geometry and history intact.

        :param label: positive ID representable by the current mask dtype.
        :param gate: shared ``EditableVolumeGate`` with physical X/Y/Z columns.
        :returns: number of changed voxels.
        """
        from .i18n import tr
        from .widgets.gate_spec import EditableVolumeGate

        if (not isinstance(label, Integral) or isinstance(label, (bool, np.bool_))
                or not 0 < label <= min(65535, np.iinfo(self.mask.dtype).max)
                or not isinstance(gate, EditableVolumeGate) or gate.columns != ('X', 'Y', 'Z')
                or gate.parent is not None or gate.name != str(label)):
            raise ValueError(tr('Volume geometry needs one valid object ID and physical X/Y/Z coordinates.'))
        output = self.mask.copy()
        output[output == label] = 0
        self._paint_geometry(output, label, gate)
        if self._cancelled is not None and self._cancelled():
            raise InterruptedError(tr('Volume editing was cancelled; no labels were changed.'))
        after = dict(self.geometries)
        after[int(label)] = gate
        return self._commit(output, after)

    def _commit(self, output, after=None):
        """Record one sparse voxel delta and its geometry as one undo step.

        :param output: the complete edited label array.
        :param after: the geometry after the edit; ``None`` drops the saved
            boundary of every label whose voxels changed.
        :returns: number of changed voxels.
        """
        from .i18n import tr

        changed = np.flatnonzero(output.ravel() != self.mask.ravel()).astype(np.uint32)
        old = self.mask.ravel()[changed].copy()
        new = output.ravel()[changed].copy()
        cost = changed.nbytes + old.nbytes + new.nbytes
        if cost > self.MAX_HISTORY_BYTES:
            raise ValueError(tr('This edit exceeds the volume undo memory budget; no labels were changed.'))
        before = dict(self.geometries)
        if after is None:
            touched = set(map(int, np.unique(np.concatenate((old, new)))))
            after = {key: value for key, value in before.items() if key not in touched}
        if not len(changed) and before == after:
            return 0
        self._redo.clear()
        while self._undo and self._history_bytes + cost > self.MAX_HISTORY_BYTES:
            removed = self._undo.pop(0)
            self._history_bytes -= sum(value.nbytes for value in removed[:3])
        self._undo.append((changed, old, new, before, after))
        self._history_bytes += cost
        if len(self._undo) > 25:
            removed = self._undo.pop(0)
            self._history_bytes -= sum(value.nbytes for value in removed[:3])
        self.mask, self.geometries = output, after
        return len(changed)

    def undo(self):
        """Restore the preceding sparse voxel delta and immutable geometry."""
        if not self._undo:
            return False
        record = self._undo.pop()
        self._history_bytes -= sum(value.nbytes for value in record[:3])
        self.mask.ravel()[record[0]] = record[1]
        self.geometries = dict(record[3])
        self._redo.append(record)
        return True

    def redo(self):
        """Reapply the most recently undone voxel delta and geometry."""
        if not self._redo:
            return False
        record = self._redo.pop()
        self.mask.ravel()[record[0]] = record[2]
        self.geometries = dict(record[4])
        self._undo.append(record)
        self._history_bytes += sum(value.nbytes for value in record[:3])
        return True

    def _section(self, array, axis, index):
        """A writable view of one section: Z gives Y x X, Y gives Z x X, X gives Z x Y."""
        from .i18n import tr

        if axis not in ('X', 'Y', 'Z'):
            raise ValueError(tr('Choose the X, Y or Z axis for a volume section.'))
        position = self.axes.index(axis)
        if (not isinstance(index, Integral) or isinstance(index, (bool, np.bool_))
                or not 0 <= index < array.shape[position]):
            raise ValueError(tr('The section index lies outside the volume.'))
        key = [slice(None)] * 3
        key[position] = int(index)
        section = array[tuple(key)]
        remaining = [name for name in self.axes if name != axis]
        wanted = {'Z': ['Y', 'X'], 'Y': ['Z', 'X'], 'X': ['Z', 'Y']}[axis]
        return section if remaining == wanted else section.T

    def section_count(self, axis):
        """Number of sections along one spatial axis.

        :param axis: ``X``, ``Y`` or ``Z``.
        :returns: the volume's length along that axis.
        """
        return int(self.image.shape[self.axes.index(axis)])

    def plane(self, axis, index):
        """Return one orthogonal section of the image and its labels.

        :param axis: the axis held fixed. ``Z`` gives the XY section (rows Y,
            columns X), ``Y`` the XZ section (rows Z, columns X) and ``X`` the
            YZ section (rows Z, columns Y).
        :param index: the section's index along ``axis``.
        :returns: ``(image, labels)``, copies of that section.
        """
        return (np.array(self._section(self.image, axis, index)),
                np.array(self._section(self.mask, axis, index)))

    def _check_label(self, label):
        """Refuse an object ID the label dtype cannot hold."""
        from .i18n import tr

        if (not isinstance(label, Integral) or isinstance(label, (bool, np.bool_))
                or not 0 < label <= min(65535, np.iinfo(self.mask.dtype).max)):
            raise ValueError(tr('Choose a positive object ID that the mask can store.'))
        return int(label)

    def paint_plane(self, axis, index, label, paint=None, erase=None):
        """Draw or erase one object on one section as a single undo step.

        Painting fills only background and the object's own voxels, so other
        objects are never overwritten. Erasing clears every label under the
        erased pixels. A label whose voxels change loses any saved anchor
        boundary, which no longer describes it.

        :param axis: the axis held fixed, as for :meth:`plane`.
        :param index: the section's index along ``axis``.
        :param label: the object ID to draw.
        :param paint: boolean pixels to assign to ``label``, shaped as the section.
        :param erase: boolean pixels to clear, shaped as the section.
        :returns: number of changed voxels.
        """
        from .i18n import tr

        label = self._check_label(label)
        output = self.mask.copy()
        view = self._section(output, axis, index)
        for pixels in (erase, paint):
            if pixels is not None and np.shape(pixels) != view.shape:
                raise ValueError(tr('The drawn section does not match the volume section.'))
        if erase is not None:
            view[np.asarray(erase, dtype=bool)] = 0
        if paint is not None:
            view[np.asarray(paint, dtype=bool) & ((view == 0) | (view == label))] = label
        return self._commit(output)

    def replace_plane(self, axis, index, labels, keep=()):
        """Write one edited section back as a single undo step.

        This is how the two-dimensional Make Masks tools edit a volume: they
        change a section, and the section is written back whole. An object
        already on this section keeps its ID, and so does any ID in ``keep``
        (the object being drawn). An ID new to this section that belongs to
        another object elsewhere in the volume is given the next unused ID
        instead, so a tool that numbered its own objects never joins them to
        a distant 3-D object.

        :param axis: the axis held fixed, as for :meth:`plane`.
        :param index: the section's index along ``axis``.
        :param labels: the edited section, shaped as :meth:`plane` returns it.
        :param keep: object IDs that may enter the section unchanged.
        :returns: number of changed voxels.
        """
        from .i18n import tr

        output = self.mask.copy()
        view = self._section(output, axis, index)
        edited = np.asarray(labels)
        if edited.shape != view.shape or edited.dtype.kind not in 'uib':
            raise ValueError(tr('The drawn section does not match the volume section.'))
        edited = edited.astype(np.int64)
        if edited.min(initial=0) < 0 or edited.max(initial=0) > min(65535, np.iinfo(self.mask.dtype).max):
            raise ValueError(tr('Choose a positive object ID that the mask can store.'))
        allowed = set(map(int, np.unique(view))) | {int(value) for value in keep} | {0}
        extra = sorted(set(map(int, np.unique(edited))) - allowed)
        if extra:
            occupied = set(map(int, np.flatnonzero(np.bincount(self.mask.ravel()))))
            clashes = [value for value in extra if value in occupied]
            occupied.update(extra)
            maximum = min(65535, np.iinfo(self.mask.dtype).max)
            candidate = 1
            for value in clashes:
                while candidate in occupied:
                    candidate += 1
                if candidate > maximum:
                    raise ValueError(tr('No unused object identifier fits this mask dtype.'))
                edited[edited == value] = candidate
                occupied.add(candidate)
        view[...] = edited
        return self._commit(output)

    def replace_labels(self, labels):
        """Make ``labels`` the whole volume's mask as one undo step.

        :param labels: nonnegative integer labels of exactly the volume's
            shape, such as a whole-volume detection.
        :returns: number of changed voxels.
        :raises ValueError: for another shape, or an identifier the mask's
            dtype cannot hold; the volume is left unchanged.
        """
        from .i18n import tr

        incoming = np.asarray(labels)
        if (incoming.shape != self.mask.shape or incoming.dtype.kind not in 'uib'
                or (incoming.size and int(incoming.min()) < 0)):
            raise ValueError(tr('The detected labels do not match the volume.'))
        top = int(incoming.max()) if incoming.size else 0
        if top > min(65535, np.iinfo(self.mask.dtype).max):
            raise ValueError(tr('No unused object identifier fits this mask dtype.'))
        return self._commit(np.ascontiguousarray(incoming, dtype=self.mask.dtype))

    def delete_label(self, label):
        """Remove one object from every section as a single undo step.

        :param label: the object ID to remove.
        :returns: number of changed voxels.
        """
        label = self._check_label(label)
        output = self.mask.copy()
        output[output == label] = 0
        return self._commit(output)

    def interpolate(self, label, axis='Z'):
        """Fill the empty sections between drawn sections of one object.

        Each gap between two sections that contain ``label`` is filled by
        blending their signed distance maps, measured in physical units, so
        the outline changes shape smoothly from one drawn section to the
        next. Only background voxels are filled.

        :param label: the object to interpolate.
        :param axis: the axis the sections are stacked along.
        :returns: number of changed voxels.
        """
        from scipy import ndimage

        label = self._check_label(label)
        count = self.section_count(axis)
        rows, columns = {'Z': 'YX', 'Y': 'ZX', 'X': 'ZY'}[axis]
        sampling = (self.xyz_spacing['XYZ'.index(rows)], self.xyz_spacing['XYZ'.index(columns)])
        drawn = [index for index in range(count)
                 if np.any(self._section(self.mask, axis, index) == label)]
        output = self.mask.copy()
        for low, high in zip(drawn, drawn[1:]):
            if high - low < 2:
                continue
            first, last = (ndimage.distance_transform_edt(region != label, sampling=sampling)
                           - ndimage.distance_transform_edt(region == label, sampling=sampling)
                           for region in (self._section(self.mask, axis, low),
                                          self._section(self.mask, axis, high)))
            for index in range(low + 1, high):
                weight = (index - low) / (high - low)
                inside = (1 - weight) * first + weight * last < 0
                view = self._section(output, axis, index)
                view[inside & (view == 0)] = label
        return self._commit(output)

    def assign_objects(self):
        """Split every label into its 3-D connected objects.

        The largest connected part of each label keeps its ID; every other
        part, touching it neither by face, edge nor corner, receives the next
        unused ID. Drawing many objects with one ID and then assigning them
        gives each its own ID.

        :returns: ``(changed voxels, objects after assigning)``.
        """
        from scipy import ndimage
        from .i18n import tr

        output = self.mask.copy()
        occupied = set(map(int, np.unique(output)))
        maximum = min(65535, np.iinfo(output.dtype).max)
        candidate = 1
        for label in sorted(occupied - {0}):
            parts, number = ndimage.label(self.mask == label, structure=np.ones((3, 3, 3), dtype=bool))
            if number < 2:
                continue
            sizes = np.bincount(parts.ravel())[1:]
            for part in [index + 1 for index in np.argsort(-sizes, kind='stable')][1:]:
                while candidate in occupied:
                    candidate += 1
                if candidate > maximum:
                    raise ValueError(tr('No unused object identifier fits this mask dtype.'))
                output[parts == part] = candidate
                occupied.add(candidate)
        changed = self._commit(output)
        return changed, int(np.count_nonzero(np.unique(self.mask)))

    def _validate_boundaries(self):
        """Refuse saved geometry that disagrees with its actual labelled voxels."""
        from .i18n import tr
        from .widgets.gate_spec import EditableVolumeGate

        for label, gate in self.geometries.items():
            if (not isinstance(label, Integral) or isinstance(label, (bool, np.bool_))
                    or not 0 < label <= min(65535, np.iinfo(self.mask.dtype).max)
                    or not isinstance(gate, EditableVolumeGate)
                    or gate.name != str(label) or gate.columns != ('X', 'Y', 'Z')
                    or gate.parent is not None):
                raise ValueError(tr('The volume sidecar contains invalid object geometry.'))
            output = self.mask.copy()
            output[output == label] = 0
            self._paint_geometry(output, label, gate)
            if not np.array_equal(output, self.mask):
                raise ValueError(tr('The saved boundary does not match its object voxels.'))


_VOLUME_MRC_SUFFIXES = ('.mrc', '.rec', '.map')


def _spatial_page_axes(axes):
    """Read a plain three-dimensional page stack's page axis as Z."""
    return 'ZYX' if axes in ('QYX', 'IYX') else axes


def _is_volume_file(path):
    """Whether ``path`` holds a 3-D spatial volume, read from its header only.

    TIFFs with three spatial axes (plain page stacks count as Z) and MRC
    files with more than one section are volumes; anything unreadable is not.
    """
    path = Path(path)
    try:
        if path.suffix.lower() in _VOLUME_MRC_SUFFIXES:
            import mrcfile

            with mrcfile.open(str(path), header_only=True, permissive=True) as handle:
                return int(handle.header.nz) > 1
        if path.suffix.lower() not in ('.tif', '.tiff'):
            return False
        import tifffile

        with tifffile.TiffFile(path) as handle:
            series = handle.series[0]
            axes = _spatial_page_axes(series.axes)
            return (len(series.shape) == len(axes) and sorted(axes.replace('C', '', 1)) == ['X', 'Y', 'Z']
                    and len(series.shape) in (3, 4))
    except Exception:
        return False


def _read_tiff_volume(source):
    """Read one spatial TIFF as ``(image, axes, spacing, unit)``.

    A TIFF with one channel axis besides X, Y and Z (``CZYX``, ``ZCYX``)
    gives ``image`` with the channels first.
    """
    import tifffile
    from .i18n import tr

    with tifffile.TiffFile(source) as handle:
        series = handle.series[0]
        stored = _spatial_page_axes(series.axes)
        channel = stored.find('C') if len(series.shape) == 4 and len(stored) == 4 else -1
        axes = stored.replace('C', '', 1) if channel >= 0 else stored
        if channel >= 0 and series.shape[channel] > 16:
            channel, axes = -1, stored
        metadata = (handle.shaped_metadata or ({},))[0] or {}
        spacing = metadata.get('voxel_spacing', (1., 1., 1.))
        unit = metadata.get('unit', 'pixel')
        if 'voxel_spacing' not in metadata and handle.ome_metadata:
            import xml.etree.ElementTree as ET

            pixels = next(node for node in ET.fromstring(handle.ome_metadata).iter()
                          if node.tag.rsplit('}', 1)[-1] == 'Pixels')
            factors = {'nm': .001, 'µm': 1., 'um': 1., 'mm': 1000., 'm': 1_000_000.}
            if any('PhysicalSize' + axis in pixels.attrib for axis in 'XYZ'):
                try:
                    spacing = tuple(float(pixels.attrib['PhysicalSize' + axis])
                                    * factors[pixels.attrib.get('PhysicalSize' + axis + 'Unit', 'µm')]
                                    for axis in axes)
                except (KeyError, ValueError) as exc:
                    raise ValueError(tr('OME volumes need complete supported physical spacing for X, Y and Z.')) from exc
                unit = 'um'
        elif 'voxel_spacing' not in metadata and handle.imagej_metadata:
            metadata = handle.imagej_metadata
            unit = metadata.get('unit', 'pixel')
            distances = {'Z': metadata.get('spacing', 1.)}
            for axis in 'XY':
                value = handle.pages[0].tags[axis + 'Resolution'].value
                if value[0] <= 0 or value[1] <= 0:
                    raise ValueError(tr('Voxel spacing must contain three positive finite distances.'))
                distances[axis] = value[1] / value[0]
            spacing = tuple(distances[axis] for axis in axes)
        if len(series.shape) != 3 + (channel >= 0) or sorted(axes) != ['X', 'Y', 'Z']:
            raise ValueError(tr('Choose a TIFF explicitly declaring three spatial axes; time, channels and RGB are not 3D fields.'))
        spatial = math.prod(series.shape) // (series.shape[channel] if channel >= 0 else 1)
        if spatial > VolumeMaskSession.MAX_VOXELS or math.prod(series.shape) * series.dtype.itemsize > 256 * 1024 * 1024:
            raise ValueError(tr('This TIFF exceeds the native volume memory budget.'))
        image = series.asarray()
    if channel >= 0:
        image = np.moveaxis(image, channel, 0)
        if len(spacing) == 4:
            spacing = tuple(value for index, value in enumerate(spacing) if index != channel)
    return image, axes, spacing, unit


def _read_mrc_volume(source):
    """Read one MRC volume as ZYX with its voxel size in nanometres."""
    from .i18n import tr

    try:
        import mrcfile
    except ImportError as exc:
        raise ValueError(tr('MRC volumes need the optional mrcfile package: pip install "spacr[tem]"')) from exc
    with mrcfile.mmap(str(source), permissive=True, mode='r') as handle:
        data = handle.data
        if data is None or data.ndim != 3 or 1 in data.shape:
            raise ValueError(tr('This MRC file holds one section, not a volume; open it as a 2D image.'))
        if data.size > VolumeMaskSession.MAX_VOXELS or data.nbytes > 256 * 1024 * 1024:
            raise ValueError(tr('This volume exceeds the native volume memory budget.'))
        if data.dtype.kind not in 'uif':
            raise ValueError(tr('3D masks require a finite numeric spatial TIFF within the volume memory budget.'))
        image = np.array(data, dtype=data.dtype.newbyteorder('='))
        size = handle.voxel_size
        angstrom = (float(size.z), float(size.y), float(size.x))
    if all(np.isfinite(value) and value > 0 for value in angstrom):
        return image, 'ZYX', tuple(value / 10. for value in angstrom), 'nm'
    return image, 'ZYX', (1., 1., 1.), 'pixel'


def load_volume_image_and_mask(folder, filename, masks_dir=None):
    """Open a spatial volume, one channel or several, for native 3-D editing.

    TIFF volumes need exactly three spatial axes, in any order. A plain
    three-dimensional page stack (tifffile's ``Q`` or ``I`` page axis over
    ``YX``) is read as Z sections, as z-stacks and FIB-SEM series are saved.
    One channel axis (``CZYX``, ``ZCYX``, at most 16 channels) is kept as
    :attr:`VolumeMaskSession.channels`; the slice views show the first
    channel. Time and colour axes are refused. ``voxel_spacing`` metadata is
    read in storage-axis order. OME physical sizes are converted to micrometres;
    ImageJ spacing and XY resolution are retained. MRC volumes (``.mrc``,
    ``.rec``, ``.map``) are read as ZYX with their voxel size in nanometres;
    they need the optional ``mrcfile`` package. Absent physical metadata
    means pixel coordinates.
    Labels retain dtype and identity. A saved geometry sidecar is bound to
    the exact source file and mask pixels and is refused when stale.

    :param folder: source image directory.
    :param filename: spatial TIFF or MRC volume filename.
    :param masks_dir: optional existing sibling mask directory.
    :returns: a ``VolumeMaskSession`` owning the labels and source metadata.
    """
    import tifffile
    from .i18n import tr
    from .widgets.gate_spec import GateSet, EditableVolumeGate

    source = Path(folder) / filename
    digest = _yolo_source_hash(source)
    if source.suffix.lower() in _VOLUME_MRC_SUFFIXES:
        image, axes, spacing, unit = _read_mrc_volume(source)
    else:
        image, axes, spacing, unit = _read_tiff_volume(source)
    channels = None
    if image.ndim == 4:
        channels, image = tuple(image), image[0]
    mask_path = Path(mask_save_path(folder, filename, masks_dir))
    if mask_path.exists():
        with tifffile.TiffFile(mask_path) as handle:
            mask_series = handle.series[0]
            if _spatial_page_axes(mask_series.axes) != axes:
                raise ValueError(tr('The saved mask axes do not match the spatial source.'))
            if (mask_series.shape != image.shape or mask_series.dtype.kind not in 'ui'
                    or math.prod(mask_series.shape) * mask_series.dtype.itemsize > 256 * 1024 * 1024):
                raise ValueError(tr('The saved spatial mask shape, integer dtype or decoded size does not match this volume.'))
            labels = mask_series.asarray()
    else:
        labels = np.zeros(image.shape, dtype=np.uint16)
    session = VolumeMaskSession(image, labels, axes=axes, spacing=spacing, unit=unit, channels=channels)
    session.source_path, session.source_sha256, session.mask_path = str(source), digest, str(mask_path)
    if _yolo_source_hash(source) != digest:
        raise ValueError(tr('The source volume changed while opening it.'))
    sidecar = mask_path.with_suffix(mask_path.suffix + '.volume.json')
    session._saved_mask_sha256 = _yolo_source_hash(mask_path) if mask_path.exists() else None
    session._saved_geometry_sha256 = _yolo_source_hash(sidecar) if sidecar.exists() else None
    if sidecar.exists():
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        expected = {'source_sha256': digest, 'shape': list(image.shape), 'axes': axes,
                    'spacing': list(session.spacing), 'mask_sha256': hashlib.sha256(session.mask.tobytes(order='C')).hexdigest()}
        if any(record.get(key) != value for key, value in expected.items()) or record.get('version') != 1:
            raise ValueError(tr('The saved volume geometry does not match these source and mask pixels.'))
        geometries = GateSet.from_dict(record.get('geometry', {}))
        for gate in geometries.gates:
            if (not isinstance(gate, EditableVolumeGate) or not gate.name.isdecimal()
                    or not 0 < int(gate.name) <= 65535 or gate.columns != ('X', 'Y', 'Z')
                    or gate.parent is not None):
                raise ValueError(tr('The volume sidecar contains invalid object geometry.'))
            session.geometries[int(gate.name)] = gate
        controls = record.get('editor_controls', session._editor_controls)
        if (not isinstance(controls, dict) or set(controls) != {'anchor_density', 'show_anchors'}
                or type(controls['anchor_density']) is not int or not 1 <= controls['anchor_density'] <= 8
                or type(controls['show_anchors']) is not bool):
            raise ValueError(tr('The saved volume anchor controls are invalid.'))
        session._editor_controls = dict(controls)
        session._validate_boundaries()
    return session


def _volume_file_stamp(path):
    """Return one file's device, inode, length and modification identity."""
    try:
        value = Path(path).stat()
    except FileNotFoundError:
        return None
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def save_volume_mask(session):
    """Write native labels and their source-bound shared boundary sidecar.

    Both artifacts are prepared before replacing either destination. A failed
    second replacement restores the previous pair. Source/output identities
    are rechecked after staging. Two filenames cannot form a fully atomic
    transaction against an uncooperative concurrent writer; interrupted or
    mismatched publications are refused by the source/mask-bound sidecar on
    loading. Source axes, spacing, label dtype and IDs are preserved.

    :param session: a volume session loaded through the existing mask paths.
    :returns: the canonical TIFF mask path.
    """
    import shutil
    from .i18n import tr
    from .widgets.gate_spec import GateSet

    if (not isinstance(session, VolumeMaskSession) or not session.source_path
            or _yolo_source_hash(Path(session.source_path)) != session.source_sha256):
        raise ValueError(tr('The source volume changed; no mask was saved.'))
    session._validate_boundaries()
    target = Path(session.mask_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = target.with_suffix(target.suffix + '.volume.json')
    if target.resolve() == Path(session.source_path).resolve():
        raise ValueError(tr('The mask output must not overwrite the source volume.'))
    record = {'version': 1, 'source_sha256': session.source_sha256,
              'mask_sha256': hashlib.sha256(session.mask.tobytes(order='C')).hexdigest(),
              'shape': list(session.mask.shape), 'axes': session.axes,
              'spacing': list(session.spacing), 'unit': session.unit,
              'geometry': GateSet(list(session.geometries.values())).to_dict(),
              'editor_controls': dict(session._editor_controls)}
    lock = sidecar.with_suffix(sidecar.suffix + '.lock')
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(tr('This volume is already being saved.')) from exc
    try:
        if ((_yolo_source_hash(target) if target.exists() else None) != session._saved_mask_sha256
                or (_yolo_source_hash(sidecar) if sidecar.exists() else None) != session._saved_geometry_sha256):
            raise ValueError(tr('The saved volume changed outside this session; no mask was overwritten.'))
        stamps = {path: _volume_file_stamp(path) for path in (Path(session.source_path), target, sidecar)}
        with tempfile.TemporaryDirectory(dir=target.parent, prefix='.spacr-volume-') as folder:
            folder = Path(folder)
            staged_mask, staged_json = folder / 'mask.tif', folder / 'geometry.json'
            write_tiff(staged_mask, session.mask, metadata={'axes': session.axes,
                        'voxel_spacing': list(session.spacing), 'unit': session.unit},
                       resolution=(1. / session.xyz_spacing[0], 1. / session.xyz_spacing[1]))
            staged_json.write_bytes(_yolo_json_bytes(record))
            backups = {}
            for path in (target, sidecar):
                if path.exists():
                    backup = folder / (path.name + '.backup')
                    shutil.copyfile(path, backup)
                    backups[path] = backup
            if (any(_volume_file_stamp(path) != stamp for path, stamp in stamps.items())
                    or _yolo_source_hash(Path(session.source_path)) != session.source_sha256
                    or (_yolo_source_hash(target) if target.exists() else None) != session._saved_mask_sha256
                    or (_yolo_source_hash(sidecar) if sidecar.exists() else None) != session._saved_geometry_sha256):
                raise ValueError(tr('The volume files changed during saving; no output was replaced.'))
            os.replace(staged_mask, target)
            try:
                os.replace(staged_json, sidecar)
            except BaseException:
                for path in (target, sidecar):
                    if path in backups:
                        os.replace(backups[path], path)
                    else:
                        path.unlink(missing_ok=True)
                raise
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)
    session._saved_mask_sha256 = _yolo_source_hash(target)
    session._saved_geometry_sha256 = _yolo_source_hash(sidecar)
    return str(target)


class MaskHistory:
    """Bounded undo/redo stack of mask arrays. Deep-copies on push so
    callers can mutate in place without corrupting older snapshots."""

    def __init__(self, capacity: int = 20):
        """Prepare an empty history with a bounded snapshot capacity.

        :param capacity: max snapshots kept in the undo (and redo) stack.
        """
        self.capacity = max(1, int(capacity))
        self._undo: deque = deque(maxlen=self.capacity)
        self._redo: deque = deque(maxlen=self.capacity)

    def clear(self) -> None:
        """Discard every snapshot from both the undo and redo stacks."""
        self._undo.clear()
        self._redo.clear()

    def push(self, mask: np.ndarray) -> None:
        """Store a deep-copy of ``mask`` and drop any redo history.

        :param mask: label mask to record as the newest undo state.
        """
        self._undo.append(np.array(mask, copy=True))
        self._redo.clear()

    def head(self) -> Optional[np.ndarray]:
        """The newest snapshot — what an edit in progress started from.

        Returned as held, not copied, so a caller that only wants to diff
        against it does not pay for a copy of a 16-bit field on every mouse
        release. It is already a private copy of whatever was pushed, so
        reading it cannot disturb the history; writing to it would.
        """
        return self._undo[-1] if self._undo else None

    def can_undo(self) -> bool:
        """Return True when at least one prior snapshot is available to undo to."""
        return len(self._undo) >= 2

    def can_redo(self) -> bool:
        """Return True when the redo stack has a snapshot to restore."""
        return bool(self._redo)

    def undo(self) -> Optional[np.ndarray]:
        """Pop the top snapshot, save it to the redo stack, and return the
        previous snapshot (i.e. one step back). None if not possible."""
        if not self.can_undo():
            return None
        current = self._undo.pop()
        self._redo.append(current)
        return np.array(self._undo[-1], copy=True)

    def redo(self) -> Optional[np.ndarray]:
        """Restore the most-recently-undone snapshot, or ``None`` if empty."""
        if not self._redo:
            return None
        snap = self._redo.pop()
        self._undo.append(np.array(snap, copy=True))
        return np.array(snap, copy=True)



#: Smallest side, in image pixels, a recrop box may have. A box smaller than
#: this is a mis-click or the tail of a drag that never really started, and
#: cutting it writes a field too small to hold the object it was aimed at.
RECROP_MIN_SIDE = 32

#: How much a new box may overlap one already cut out of this field, as
#: intersection over union. Above it the box is the SAME region drawn again,
#: not a second object, and it is refused rather than written: without that
#: refusal one object reached disk three times as three near-identical
#: fields, because nothing on screen said the first box had worked.
RECROP_MAX_OVERLAP = 0.5

#: Folder the retired originals are moved into, beside the images they were
#: enumerated with. Not a delete: see the note above.
RECROP_ARCHIVE_DIRNAME = "recropped_originals"

#: The record of every retirement, written inside the archive folder.
RECROP_MANIFEST = "recropped.json"

#: What separates a child crop's name from the field it was cut from:
#: ``<field>__r00``, ``<field>__r01``. Recropping a child re-uses the
#: original field's name rather than nesting, so a name says which field a
#: crop came from however many passes it took.
RECROP_INFIX = "__r"

#: The verb a recrop goes into the curation ledger under.
RECROP_KIND = "recrop"


class RecropRefused(ValueError):
    """Raised when a proposed recrop does not satisfy recropping rules.

    :ivar reason: Stable reason code: ``"no_field"``, ``"too_small"``, or
        ``"redraw"``.

    :param reason: the stable code above. Callers branch on it, so it must
        be one of the three rather than prose.
    :param message: what to say to the user. This is the exception's own
        message, so it is what an unhandled raise would print.
    """

    def __init__(self, reason: str, message: str):
        """Record why a re-crop was refused.

        :param reason: the machine-readable refusal code, kept so a caller can
            branch on it rather than parsing the message.
        :param message: the sentence shown to the user.
        """
        super().__init__(message)
        self.reason = str(reason)


def _ordered_box(shape, p0, p1) -> Tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` for two dragged corners, clipped to ``shape``."""
    height, width = int(shape[0]), int(shape[1])
    x0, x1 = sorted((int(p0[0]), int(p1[0])))
    y0, y1 = sorted((int(p0[1]), int(p1[1])))
    return (max(0, min(width, x0)), max(0, min(height, y0)),
            max(0, min(width, x1)), max(0, min(height, y1)))


def box_overlap(a, b) -> float:
    """Intersection over union of two ``(x0, y0, x1, y1)`` boxes.

    :param a: first box; its first four items are read and truncated to
        ``int``.
    :param b: second box, read the same way.
    :returns: A value from 0 for disjoint boxes to 1 for identical boxes.
    """
    ax0, ay0, ax1, ay1 = (int(v) for v in a[:4])
    bx0, by0, bx1, by1 = (int(v) for v in b[:4])
    inter = (max(0, min(ax1, bx1) - max(ax0, bx0))
             * max(0, min(ay1, by1) - max(ay0, by0)))
    union = ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)
    return (inter / union) if union > 0 else 0.0


def recrop_box(shape, p0, p1, existing=()) -> Tuple[int, int, int, int]:
    """Validate and clip a rectangular recrop selection.

    :param shape: Image shape as ``(height, width)``. Coordinates outside the
        image are clipped to this extent.
    :param p0: First selection corner as ``(x, y)``.
    :param p1: Opposite selection corner as ``(x, y)``.
    :param existing: Previously accepted boxes. A selection whose intersection
        over union exceeds :data:`RECROP_MAX_OVERLAP` is rejected.
    :returns: Validated ``(x0, y0, x1, y1)`` coordinates.
    :raises RecropRefused: If a side is shorter than
        :data:`RECROP_MIN_SIDE` or the selection duplicates an existing box.
    """
    x0, y0, x1, y1 = _ordered_box(shape, p0, p1)
    if (x1 - x0) < RECROP_MIN_SIDE or (y1 - y0) < RECROP_MIN_SIDE:
        raise RecropRefused(
            "too_small",
            f"Recrop box is {x1 - x0}x{y1 - y0} px — at least "
            f"{RECROP_MIN_SIDE} px on each side is needed.")
    for other in existing:
        if box_overlap((x0, y0, x1, y1), other) > RECROP_MAX_OVERLAP:
            name = other[4] if len(other) > 4 else "an earlier box"
            raise RecropRefused(
                "redraw",
                f"That region is already cut out as {name} — it is saved, "
                "draw the next one.")
    return (x0, y0, x1, y1)


def cut_recrop(image: np.ndarray, mask: np.ndarray,
               box) -> Tuple[np.ndarray, np.ndarray]:
    """Extract an image region and its complete labelled objects.

    Objects touching the crop boundary are removed because their masks are
    incomplete. Remaining labels are renumbered consecutively from one.

    :param image: Source microscopy image.
    :param mask: Label image aligned with ``image``.
    :param box: Crop coordinates as ``(x0, y0, x1, y1)``.
    :returns: Cropped image and relabelled mask.
    """
    x0, y0, x1, y1 = (int(v) for v in box[:4])
    sub_image = np.ascontiguousarray(np.asarray(image)[y0:y1, x0:x1])
    sub_mask = np.ascontiguousarray(np.asarray(mask)[y0:y1, x0:x1])
    if sub_mask.size and sub_mask.max():
        edge = np.unique(np.concatenate([
            sub_mask[0, :], sub_mask[-1, :], sub_mask[:, 0], sub_mask[:, -1],
        ]))
        edge = edge[edge > 0]
        if edge.size:
            sub_mask = np.where(np.isin(sub_mask, edge), 0, sub_mask)
    relabelled, _ = _ndimage().label(sub_mask > 0, structure=_EIGHT)
    return sub_image, relabelled.astype(np.uint16)


def _recrop_base(filename: str) -> str:
    """The field name a child crop is named after.

    A recrop of a recrop is named after the ORIGINAL field, not after its
    parent: ``well_A1__r00`` recropped again yields ``well_A1__r03``, never
    ``well_A1__r00__r00``. Nesting would make the name grow with every pass
    while saying nothing more than the manifest already records. A bundle's
    ``_seg`` is not part of its name: ``well_A1_seg.npy`` yields
    ``well_A1__r00_seg.npy``.
    """
    return field_stem(filename).split(RECROP_INFIX)[0]


def recrop_child_name(folder: str, filename: str, ext: str = ".tif",
                      masks_dir: Optional[str] = None) -> str:
    """Return the next unused ``<field>__rNN`` filename.

    Names are checked against the image queue, mask directory, and recrop
    archive to prevent overwriting output from an earlier editing session.

    :param folder: the folder the editor opened.
    :param filename: the field being cut.
    :param ext: the child image's extension. Ignored for a bundle, whose
        child is a bundle too, ``<field>__rNN_seg.npy``, because a TIFF
        written into a folder of bundles would make it two layouts at once.
    :param masks_dir: where the masks are, when not in ``<folder>/masks``.
    :returns: the child's file name.
    """
    base = _recrop_base(filename)
    archive = os.path.join(folder, RECROP_ARCHIVE_DIRNAME)
    masks = masks_folder(folder, masks_dir)
    index = 0
    while True:
        stem = f"{base}{RECROP_INFIX}{index:02d}"
        taken = [os.path.join(masks, stem + ".tif"),
                 os.path.join(archive, "masks", stem + ".tif")]
        taken += [os.path.join(d, stem + e)
                  for d in (folder, archive)
                  for e in IMAGE_EXTS + (SEG_SUFFIX,)]
        if not any(os.path.exists(path) for path in taken):
            return stem + (SEG_SUFFIX if is_seg_bundle(filename) else ext)
        index += 1


class Recrop(NamedTuple):
    """Result of writing a recropped image and mask.

    :ivar name: Filename assigned to the recropped field.
    :ivar image_path: Path of the written image.
    :ivar mask_path: Path of the written label mask.
    :ivar n_objects: Number of complete labelled objects retained.
    :ivar box: Source coordinates as ``(x0, y0, x1, y1)``.
    """

    name: str
    image_path: str
    mask_path: str
    n_objects: int
    box: Tuple[int, int, int, int]


def write_recrop(folder: str, filename: str, image: np.ndarray,
                 mask: np.ndarray, box,
                 masks_dir: Optional[str] = None) -> "Recrop":
    """Write a recropped field, mask, and curation record.

    The image is stored as an unscaled uint16 TIFF beside the source images,
    and the relabelled mask is stored in ``<folder>/masks``. The curation
    record distinguishes deliberately removed boundary objects from missed
    segmentation objects.

    A field cut from a ``_seg.npy`` bundle becomes a bundle of its own,
    holding ``img``, ``masks``, empty ``flows`` and a ``filename`` naming the
    parent and the box -- the form the external curation tool gave its
    recrops, so either tool can open the other's.

    :param folder: Image-queue directory.
    :param filename: Source image filename.
    :param image: Source microscopy image.
    :param mask: Label image aligned with ``image``.
    :param box: Coordinates returned by :func:`recrop_box`.
    :param masks_dir: where the masks are, when not in ``<folder>/masks``.
    :returns: Filename and retained-object count for the new field.
    """
    x0, y0, x1, y1 = (int(v) for v in box[:4])
    sub_image, sub_mask = cut_recrop(image, mask, (x0, y0, x1, y1))
    child = recrop_child_name(folder, filename, masks_dir=masks_dir)
    image_path = os.path.join(folder, child)
    mask_path = mask_save_path(folder, child, masks_dir)
    if is_seg_bundle(child):
        _write_bundle(image_path, {
            "img": np.asarray(sub_image).astype(np.uint16),
            "masks": sub_mask,
            "flows": [None, None, None],
            "filename": (f"recrop of {field_stem(filename)} "
                         f"[{x0}:{x1},{y0}:{y1}]"),
        })
    else:
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        write_tiff(image_path, np.asarray(sub_image).astype(np.uint16))
        write_tiff(mask_path, sub_mask)
    log = CurationLog(mask_path, source=CURATION_SOURCE)
    log.append(RECROP_KIND, child,
               n_changed=int(np.count_nonzero(sub_mask)),
               parent=os.path.basename(str(filename)),
               box=[x0, y0, x1, y1],
               n_objects=int(sub_mask.max()))
    log.write_beside(mask_path)
    return Recrop(child, image_path, mask_path,
                  int(sub_mask.max()), (x0, y0, x1, y1))


def recrop_archive_dir(folder: str) -> str:
    """Return the archive directory for recropped source fields.

    :param folder: image folder; the archive is its
        :data:`RECROP_ARCHIVE_DIRNAME` subfolder. Nothing is created.
    """
    return os.path.join(folder, RECROP_ARCHIVE_DIRNAME)


def retire_recropped_original(folder: str, filename: str, *,
                              children=(), boxes=(),
                              masks_dir: Optional[str] = None) -> dict:
    """Archive a source field after recropped children have been created.

    The source image, mask, and curation ledger are moved to
    ``<folder>/recropped_originals`` and recorded in :data:`RECROP_MANIFEST`.
    This removes the multi-object source from the training queue without
    deleting it. A bundle goes with its ledger and with any display image of
    its stem beside it, as the external curation tool moved its ``.png``.

    :param folder: Image-queue directory.
    :param filename: Source image filename.
    :param children: Filenames created from the source field.
    :param boxes: Crop boxes corresponding to ``children``.
    :param masks_dir: where the masks are, when not in ``<folder>/masks``.
    :returns: Manifest record, including the original and archived paths.
    """
    archive = recrop_archive_dir(folder)
    name = os.path.basename(str(filename))
    if is_seg_bundle(filename):
        bundle = os.path.join(folder, name)
        moves = [(bundle, os.path.join(archive, name)),
                 (bundle + LOG_SUFFIX,
                  os.path.join(archive, name + LOG_SUFFIX))]
        moves += [(os.path.join(folder, field_stem(name) + ext),
                   os.path.join(archive, field_stem(name) + ext))
                  for ext in IMAGE_EXTS]
    else:
        mask_path = mask_save_path(folder, filename, masks_dir)
        moves = [
            (os.path.join(folder, filename),
             os.path.join(archive, name)),
            (mask_path, os.path.join(archive, "masks",
                                     os.path.basename(mask_path))),
            (mask_path + LOG_SUFFIX,
             os.path.join(archive, "masks",
                          os.path.basename(mask_path) + LOG_SUFFIX)),
        ]
    moved = []
    for source, target in moves:
        if not os.path.exists(source):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.replace(source, target)
        moved.append([source, target])
    record = {
        "original": os.path.basename(str(filename)),
        "children": [str(c) for c in children],
        "boxes": [[int(v) for v in box[:4]] for box in boxes],
        "moved": moved,
    }
    _append_recrop_manifest(archive, record)
    return record


def _append_recrop_manifest(archive: str, record: dict) -> str:
    """Append one retirement to the archive's manifest; return its path.

    Read-modify-write of a list rather than a line per record, so the file
    is ordinary JSON that anything can open. A manifest that cannot be read
    is replaced rather than allowed to stop the retirement: the files
    themselves are the recovery, and the manifest is the map to them.
    """
    os.makedirs(archive, exist_ok=True)
    path = os.path.join(archive, RECROP_MANIFEST)
    records = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                records = loaded
        except (OSError, ValueError):
            records = []
    records.append(record)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=1)
    return path


def read_recrop_manifest(folder: str) -> List[dict]:
    """Return recrop-archive records in chronological order.

    :param folder: image folder whose :func:`recrop_archive_dir` holds the
        manifest; a missing, unreadable or non-list manifest gives ``[]``.
    """
    path = os.path.join(recrop_archive_dir(folder), RECROP_MANIFEST)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return []
    return loaded if isinstance(loaded, list) else []


def restore_recropped_original(folder: str, original: str) -> List[str]:
    """Restore an archived source field to the image queue.

    Existing archived files listed for ``original`` are moved back to their
    source locations. Recropped child fields are not modified.

    :param folder: Image-queue directory.
    :param original: Original source filename recorded in the manifest.
    :returns: Paths restored to the queue.
    """
    restored: List[str] = []
    name = os.path.basename(str(original))
    for record in reversed(read_recrop_manifest(folder)):
        if record.get("original") != name:
            continue
        for pair in record.get("moved") or ():
            if len(pair) != 2:
                continue
            source, target = pair
            if not os.path.exists(target) or os.path.exists(source):
                continue
            os.makedirs(os.path.dirname(source), exist_ok=True)
            os.replace(target, source)
            restored.append(source)
        break
    return restored


_LAYER_COLOURS: Tuple[str, ...] = (
    "#00e5ff", "#ffd60a", "#ff4d6d", "#7cff6b", "#c77dff", "#ff9f1c",
    "#4d96ff", "#f72585",
)
"""Default outline colours of mask layers, in the order layers are added."""

_LAYER_SETTINGS_NAME = "curate_layers.json"
"""The file beside the images that keeps a mask-layer session's choices."""

_LAYER_DECISIONS_SUFFIX = ".decisions.json"
"""Suffix of the per-image decisions log written beside a curated mask."""


def _layer_mask_file(layer_dir: str, filename: str) -> Optional[str]:
    """The label file of one layer that belongs to ``filename``.

    Matched by stem the way ``masks/`` is: ``<stem>.tif`` first, then the
    image's own name, ``.tiff``, ``.png`` and ``.npy``.

    :param layer_dir: the layer's folder.
    :param filename: the field's image name.
    :returns: the path, or None when the layer has nothing for the field.
    """
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    for name in (stem + ".tif", os.path.basename(str(filename)), stem + ".tiff",
                 stem + ".png", stem + ".npy"):
        path = os.path.join(str(layer_dir), name)
        if os.path.isfile(path):
            return path
    return None


def _read_layer_mask(layer_dir: str, filename: str,
                     shape) -> Optional[np.ndarray]:
    """One layer's labels for a field, or None.

    :param layer_dir: the layer's folder.
    :param filename: the field's image name.
    :param shape: the field's shape; a file of another shape is refused.
    :returns: the labels, or None when the layer has no file for the field.
    :raises ValueError: when the file does not match the field's shape.
    """
    path = _layer_mask_file(layer_dir, filename)
    if path is None:
        return None
    raw = np.load(path) if path.endswith(".npy") else imageio.imread(path)
    return _as_field_mask(raw, tuple(shape), path, filename)


def _layer_pick(curated: np.ndarray, base: Optional[np.ndarray], layers,
                x: int, y: int, next_id: int):
    """What one click of the Pick tool does to the curated overlay.

    In order: an object of the curated overlay under the click is removed;
    else a base-layer object that is no longer in the overlay there is
    restored (only onto background); else the first of ``layers`` with an
    object under the click adds that object under ``next_id``, again only
    where the overlay is background.

    :param curated: the curated overlay.
    :param base: the base layer's labels, or None when the base is hidden
        or has no file for this field.
    :param layers: ``[(layer index, labels), ...]`` of the other visible
        layers, in the order they are tried.
    :param x: column clicked.
    :param y: row clicked.
    :param next_id: the id an added object takes.
    :returns: ``(new overlay, action)``, or ``(None, None)`` when the click
        changed nothing. ``action`` holds ``kind`` (``remove``, ``restore``
        or ``add``), ``id`` and, for an add, ``layer`` and ``source_id``.
    """
    current = int(curated[y, x])
    if current:
        out = np.array(curated, copy=True)
        out[out == current] = 0
        return out, {"kind": "remove", "id": current}
    if base is not None and int(base[y, x]):
        label = int(base[y, x])
        region = (base == label) & (curated == 0)
        if region.any():
            out = np.array(curated, copy=True)
            out = _widen_labels(out, label)
            out[region] = label
            return out, {"kind": "restore", "id": label}
    for index, labels in layers:
        if labels is None or not int(labels[y, x]):
            continue
        source = int(labels[y, x])
        region = (labels == source) & (curated == 0)
        if not region.any():
            continue
        out = _widen_labels(np.array(curated, copy=True), next_id)
        out[region] = next_id
        return out, {"kind": "add", "id": int(next_id), "layer": int(index),
                     "source_id": source}
    return None, None


def _widen_labels(labels: np.ndarray, value: int) -> np.ndarray:
    """``labels`` in a dtype that can hold ``value``.

    :param labels: a label image.
    :param value: the largest id about to be written.
    :returns: ``labels`` itself when it fits, else a uint16 or uint32 copy.
    """
    if labels.dtype.kind in "ui" and value <= np.iinfo(labels.dtype).max:
        return labels
    return labels.astype(np.uint16 if value <= 65535 else np.uint32)


def _layer_removed_ids(base: Optional[np.ndarray], curated: np.ndarray,
                       previous=()) -> List[int]:
    """Base-layer ids that are not in the curated overlay.

    :param base: the base layer's labels, or None.
    :param curated: the curated overlay.
    :param previous: ids an earlier session recorded as removed; kept while
        they are still absent, so overwriting the base layer loses nothing.
    :returns: the ids, sorted.
    """
    present = set(int(v) for v in np.unique(curated) if v)
    removed = set(int(v) for v in previous) - present
    if base is not None:
        removed |= set(int(v) for v in np.unique(base) if v) - present
    return sorted(removed)


def _layer_overlay(layers, curated: np.ndarray, base_index: Optional[int],
                   edges=None) -> Optional[np.ndarray]:
    """The outlines of every visible layer, as one RGBA picture.

    Each layer is drawn as its objects' inner boundaries in its own colour,
    earlier layers on top. Objects of the base layer that are not in the
    curated overlay are drawn dashed, which is what tells a removed object
    from a kept one.

    :param layers: ``[(labels or None, "#rrggbb", visible), ...]``.
    :param curated: the curated overlay.
    :param base_index: which of ``layers`` is the base, or None.
    :param edges: each layer's boundary pixels, already computed, so an
        edit redraws without tracing every layer again; traced here when
        omitted.
    :returns: an ``(h, w, 4)`` uint8 array, or None when nothing is drawn.
    """
    from skimage.segmentation import find_boundaries

    rgba = None
    for index in reversed(range(len(layers))):
        labels, colour, visible = layers[index]
        if labels is None or not visible:
            continue
        if rgba is None:
            rgba = np.zeros(curated.shape[:2] + (4,), dtype=np.uint8)
        edge = (edges[index] if edges is not None and index < len(edges)
                and edges[index] is not None
                else find_boundaries(labels, mode="inner") & (labels > 0))
        red, green, blue = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
        if index == base_index:
            removed = _layer_removed_ids(labels, curated)
            if removed:
                gone = np.isin(labels, removed)
                rows, cols = np.indices(labels.shape)
                dash = ((rows + cols) // 3) % 2 == 0
                rgba[edge & ~gone] = (red, green, blue, 255)
                rgba[edge & gone & dash] = (red, green, blue, 255)
                continue
        rgba[edge] = (red, green, blue, 255)
    return rgba


def _default_curated_folder(root: str, today: Optional[str] = None) -> str:
    """Where a mask-layer session saves when nobody chose a folder.

    The newest existing ``masks_curated_<date>`` folder in ``root``, so a
    session resumes into the folder it was writing yesterday; otherwise a
    new one named with today's date.

    :param root: the folder the images were opened from.
    :param today: the date as ``YYYYMMDD``; today when omitted.
    :returns: the folder's path, which need not exist yet.
    """
    import datetime

    try:
        existing = sorted(name for name in os.listdir(root)
                          if name.startswith("masks_curated_")
                          and os.path.isdir(os.path.join(root, name)))
    except OSError:
        existing = []
    if existing:
        return os.path.join(root, existing[-1])
    stamp = today or datetime.date.today().strftime("%Y%m%d")
    return os.path.join(root, f"masks_curated_{stamp}")


def _read_layer_settings(root: str) -> Dict:
    """The mask-layer choices kept in ``<root>/curate_layers.json``.

    :param root: the folder the images were opened from.
    :returns: the saved choices, or an empty dict when there are none or the
        file cannot be read.
    """
    path = os.path.join(str(root), _LAYER_SETTINGS_NAME)
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_layer_settings(root: str, data: Dict) -> Optional[str]:
    """Keep the mask-layer choices in ``<root>/curate_layers.json``.

    :param root: the folder the images were opened from.
    :param data: the choices, JSON-serialisable.
    :returns: the path written, or None when it could not be written.
    """
    path = os.path.join(str(root), _LAYER_SETTINGS_NAME)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError:
        return None
    return path


def _layer_decisions_path(target: str, filename: str) -> str:
    """Where the decisions log of one field goes, beside its curated mask.

    :param target: the folder the curated overlay is saved to.
    :param filename: the field's image name.
    :returns: ``<target>/<stem>.decisions.json``.
    """
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    return os.path.join(str(target), stem + _LAYER_DECISIONS_SUFFIX)


def _read_layer_decisions(target: str, filename: str) -> Dict:
    """The decisions an earlier session logged for one field.

    :param target: the save folder.
    :param filename: the field's image name.
    :returns: the log, or an empty dict.
    """
    try:
        with open(_layer_decisions_path(target, filename),
                  encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _layer_decisions(filename: str, base_path: Optional[str],
                     base: Optional[np.ndarray], curated: np.ndarray,
                     added: Dict[int, Dict], previous: Optional[Dict] = None,
                     saved: Optional[str] = None) -> Dict:
    """The decisions log of one field: what was removed and what was added.

    :param filename: the field's image name.
    :param base_path: the base layer's folder.
    :param base: the base layer's labels for the field, or None.
    :param curated: the curated overlay as saved.
    :param added: ``{new id: {"layer": folder, "source_id": id}}`` for every
        object added from another layer; only ids still in ``curated`` are
        logged.
    :param previous: the log an earlier session wrote, whose removals are
        carried forward.
    :param saved: the ISO time of the save; now when omitted.
    :returns: the log, ready for :func:`json.dump`.
    """
    import datetime

    present = set(int(v) for v in np.unique(curated) if v)
    rows = [{"id": int(new), "layer": info.get("layer"),
             "source_id": info.get("source_id")}
            for new, info in sorted(added.items()) if int(new) in present]
    return {
        "schema": 1,
        "image": os.path.basename(str(filename)),
        "base_layer": base_path,
        "removed": _layer_removed_ids(
            base, curated, (previous or {}).get("removed", ())),
        "added": rows,
        "saved": saved or datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _write_layer_decisions(target: str, filename: str, log: Dict) -> str:
    """Write one field's decisions log beside its curated mask.

    :param target: the save folder.
    :param filename: the field's image name.
    :param log: from :func:`_layer_decisions`.
    :returns: the path written.
    """
    path = _layer_decisions_path(target, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2)
        handle.write("\n")
    return path
