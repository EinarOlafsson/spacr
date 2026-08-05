"""``A18`` — hand a field to napari, take the corrected mask back.

spaCR has its own brush now (:mod:`spacr.curation`, :mod:`spacr.qt.layer_viewer`,
:mod:`spacr.qt.curation_tool`): a world-space brush over a labels layer, with
track curation beside it and an append-only ledger under both. This module is
not a replacement for any of that. It exists because a great many people
already have napari muscle memory — the fill tool, the polygon, the
keybindings, their own plugins — and spaCR has no business insisting they
learn a second brush to fix four cells.

So: a mask goes out to napari, the user corrects it there, and it comes back.

Round-trip fidelity is the whole feature
----------------------------------------

A bridge that returns *approximately* the mask is worse than no bridge, so
the rules are stated rather than assumed, and each is a test:

* **Label values survive exactly.** Label 41 comes back as 41 and not as 1,
  not renumbered, not relabelled by connectivity. :func:`to_spacr_mask` casts
  back to spaCR's ``uint16`` mask convention (:func:`spacr.mask_io.save_mask`)
  and **refuses** a label that would not fit rather than letting it wrap —
  ``np.uint16(70000)`` is ``4464``, silently, and a silently renamed cell is
  the worst thing this module could do.
* **Orientation survives exactly.** A napari 2-D layer's array axes are
  ``(row, column)``, which is numpy's order and spaCR's order, so the correct
  amount of transposing is *none*. That sounds too obvious to test until you
  meet a viewer that displays ``(x, y)`` and someone "fixes" it with a ``.T``
  that is invisible on the square test image everybody uses.
* **The shape may not change.** napari's brush cannot change an array's
  shape, but a caller handing back the wrong layer can, and that is a
  mistake worth a refusal rather than a resize.

Corrections are recorded, the same way as spaCR's own
-----------------------------------------------------

Every write-back appends to the artefact's :class:`spacr.curation.CurationLog`
— the same sidecar the brush writes, in the same append-only form. That is the
rule :mod:`spacr.curation` exists for, and it does not stop applying because
the editing happened in another window: a hand-edited mask that looks exactly
like a segmented one is a reproducibility hole no matter which program did the
painting. :func:`spacr.curation.is_curated` answers "was this touched?" for a
napari correction exactly as it does for a brush stroke.

napari is optional, and is never imported at module scope
---------------------------------------------------------

``napari`` is declared in the ``napari`` extra, never in the core
dependencies, and every import of it here is inside the function that needs
it. Two separate reasons, both load-bearing:

* it pulls a second Qt stack, and this module is imported by a settings panel
  inside spaCR's own PySide6 application;
* the missing-dependency path must print
  ``pip install "spacr[napari]"``, not a traceback from six frames inside
  somebody's import machinery. :func:`require_napari` is that path, following
  :data:`spacr.qt._QT_MISSING_MESSAGE` and :func:`spacr.ome_zarr.require_zarr`.

Everything except :func:`require_napari`, :func:`open_in_napari` and
:func:`run_event_loop` works with no napari installed at all, which is also
how the fidelity tests run: :func:`layer_specs` and :func:`labels_from_viewer`
speak to a duck-typed viewer, so the conversion either side of napari is
exercised for real without one.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .curation import CurationEdit, CurationLog, is_curated
from .mask_io import load_mask, save_mask

__all__ = [
    "CorrectionResult",
    "EDIT_KIND",
    "IMAGE_LAYER_NAME",
    "LABELS_LAYER_NAME",
    "LOG_SOURCE",
    "MASK_DTYPE",
    "MAX_LABEL",
    "MaskFidelityError",
    "MaskHandoff",
    "NAPARI_EXTRA",
    "NAPARI_MISSING_MESSAGE",
    "NapariExtraMissing",
    "add_to_viewer",
    "correct_mask",
    "labels_from_viewer",
    "layer_specs",
    "load_handoff",
    "missing_napari_message",
    "napari_available",
    "open_in_napari",
    "read_image",
    "require_napari",
    "run_event_loop",
    "to_spacr_mask",
    "write_back",
]

LOG = logging.getLogger("spacr.napari_bridge")

#: spaCR's mask convention, from :func:`spacr.mask_io.save_mask`. Everything
#: that comes back through this module is cast to it, or refused.
MASK_DTYPE = np.uint16
#: The largest label that fits. A mask with more objects than this is a real
#: thing (a big timelapse), and it has to fail loudly rather than wrap.
MAX_LABEL = int(np.iinfo(MASK_DTYPE).max)

#: The ``setup.py`` extra that provides :mod:`napari`.
NAPARI_EXTRA = "napari"

#: Layer names used in the napari viewer, and looked for on the way back.
IMAGE_LAYER_NAME = "image"
LABELS_LAYER_NAME = "mask"

#: What a correction made through this bridge is called in the ledger, and
#: the ``source`` a ledger this module creates is stamped with. The kind is
#: distinct from the brush's ``"paint"`` on purpose: a reviewer reading the
#: ledger should be able to see *which tool* was used, and a napari edit is
#: one whole-field diff rather than a stroke, so calling it a paint would
#: misdescribe both the granularity and the provenance.
EDIT_KIND = "napari"
LOG_SOURCE = "spacr napari bridge"

#: One sentence of diagnosis and one command, following
#: :data:`spacr.qt._QT_MISSING_MESSAGE`.
NAPARI_MISSING_MESSAGE = """\
Correcting a mask in napari needs the optional `napari` extra, which is not
installed in this environment (missing module: {module}).

Install it with:

    python -m pip install "spacr[napari]"

You do not need it to correct masks: spaCR's own Curate screen has a brush, a
label picker and track curation, and records the same ledger. This bridge is
for people who would rather work in napari.\
"""


class NapariExtraMissing(ImportError):
    """``napari`` is not installed.

    An :class:`ImportError` subclass, so a caller already guarding with
    ``except ImportError`` keeps working and the actionable message — not a
    traceback from inside somebody else's import machinery — is what reaches
    the user.
    """


class MaskFidelityError(ValueError):
    """A mask came back in a shape this module will not silently accept.

    Raised rather than corrected. Every case it covers — a label too large
    for ``uint16``, a negative label, a changed array shape, fractional
    values — is one where "doing something sensible" means quietly writing a
    different mask than the user drew.
    """


def missing_napari_message(module: str = "napari") -> str:
    """The install instruction, naming the module that was actually missing."""
    return NAPARI_MISSING_MESSAGE.format(module=module or "napari")


def require_napari() -> Any:
    """Import and return :mod:`napari`, or raise a message worth reading.

    The **only** place this module imports napari. See the module docstring
    for why that matters more here than in most optional-dependency code.

    :returns: the imported :mod:`napari` module.
    :raises NapariExtraMissing: when the extra is not installed.
    """
    try:
        import napari
    except ImportError as exc:
        module = (getattr(exc, "name", None) or "napari").split(".", 1)[0]
        raise NapariExtraMissing(missing_napari_message(module)) from exc
    return napari


def napari_available() -> bool:
    """Whether napari can be imported. Never raises.

    For a screen that wants to grey a button out rather than let the user
    press it and read a paragraph.
    """
    from importlib.util import find_spec

    try:
        return find_spec("napari") is not None
    except (ImportError, ValueError):         # pragma: no cover - broken meta
        return False


# ---------------------------------------------------------------------------
# Reading the field
# ---------------------------------------------------------------------------

def read_image(path: Any) -> np.ndarray:
    """Read an image file for display beside the mask.

    TIFFs go through ``tifffile`` so a 16-bit field keeps its bit depth,
    ``.npy`` through numpy, everything else through Pillow — the same three
    branches, in the same order, as
    :func:`spacr.qt.widgets.live_preview.load_preview_image`, written here
    because this module must not import anything from :mod:`spacr.qt`.

    :param path: the file.
    :returns: the array exactly as stored. No rescaling and no reordering:
        napari is being handed the data, not a picture of it.
    :raises FileNotFoundError: when there is nothing there.
    """
    target = os.fspath(path)
    if not os.path.isfile(target):
        raise FileNotFoundError(target)
    suffix = os.path.splitext(target)[1].lower()
    if suffix in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(target)
    if suffix == ".npy":
        return np.load(target, allow_pickle=False)
    from PIL import Image
    with Image.open(target) as handle:
        return np.asarray(handle)


@dataclass(frozen=True)
class MaskHandoff:
    """One field on its way to napari, and the mask's way home.

    :param mask: the label array, ``uint16``, exactly as it is on disk.
    :param mask_path: where it came from, and where a correction is written
        back. Also what the curation ledger is named after.
    :param image: the image to show under it, or None.
    :param image_path: where that came from.
    :param name: the labels layer's name in napari, and what is looked for on
        the way back.
    :param scale: per-axis world scale handed to napari, when the field is
        calibrated. Empty means one world unit per pixel.
    """

    mask: np.ndarray
    mask_path: str = ""
    image: Optional[np.ndarray] = None
    image_path: str = ""
    name: str = LABELS_LAYER_NAME
    scale: Tuple[float, ...] = ()

    @property
    def labels(self) -> Tuple[int, ...]:
        """Every non-zero label present, sorted."""
        values = np.unique(np.asarray(self.mask))
        return tuple(int(v) for v in values if v)

    @property
    def curated(self) -> bool:
        """Whether this mask already carries a curation ledger with edits."""
        return bool(self.mask_path) and is_curated(self.mask_path)

    def describe(self) -> str:
        """One line for a status bar."""
        where = os.path.basename(self.mask_path) or "an in-memory mask"
        return (f"{where}: {len(self.labels)} object(s), "
                f"{'x'.join(str(n) for n in self.mask.shape)}"
                + (", already curated" if self.curated else ""))


def load_handoff(mask_path: Any, image_path: Any = "", *,
                 name: str = LABELS_LAYER_NAME,
                 scale: Sequence[float] = ()) -> MaskHandoff:
    """Read a field's mask, and optionally its image, ready for napari.

    The mask is read with :func:`spacr.mask_io.load_mask`, which is spaCR's
    own reader and already probes ``.tif`` / ``.tiff`` / ``.npy`` for a bare
    stem. Reading it any other way here would be the first place the round
    trip could start losing.

    :param mask_path: the label mask.
    :param image_path: the image to show under it. Optional.
    :param name: the labels layer's name.
    :param scale: per-axis world scale.
    :returns: a :class:`MaskHandoff`.
    :raises FileNotFoundError: when the mask is not there.
    """
    mask = load_mask(mask_path)
    image = read_image(image_path) if image_path else None
    return MaskHandoff(mask=mask, mask_path=os.fspath(mask_path),
                       image=image,
                       image_path=os.fspath(image_path) if image_path else "",
                       name=str(name or LABELS_LAYER_NAME),
                       scale=tuple(float(s) for s in scale))


# ---------------------------------------------------------------------------
# Across the bridge
# ---------------------------------------------------------------------------

def layer_specs(handoff: MaskHandoff) -> Tuple[Dict[str, Any], ...]:
    """What napari should be asked to add, as plain dictionaries.

    Separated from the call that adds them so the *contents* of the handoff —
    which array, under which name, at which scale, with no axis reordering —
    can be asserted with no napari installed. Each dict carries a ``"kind"``
    naming the ``add_*`` method it belongs to; everything else is keyword
    arguments for it.

    :param handoff: the field.
    :returns: the image spec (when there is an image) then the labels spec.
    """
    common: Dict[str, Any] = {}
    if handoff.scale:
        common["scale"] = list(handoff.scale)
    specs: list = []
    if handoff.image is not None:
        specs.append({"kind": "image", "data": np.asarray(handoff.image),
                      "name": IMAGE_LAYER_NAME, **common})
    # A COPY, and this is not defensive tidiness. napari's brush edits the
    # array it was handed IN PLACE, so handing over `handoff.mask` itself
    # would mean the "before" spaCR is holding gets painted on too — and the
    # diff in `write_back` would then be uniformly zero, silently, for every
    # correction ever made through this bridge. The dtype is left alone:
    # napari's labels layer takes any integer dtype, so there is nothing to
    # convert on the way out; the conversion that matters is on the way back,
    # in `to_spacr_mask`.
    specs.append({"kind": "labels", "data": np.array(handoff.mask, copy=True),
                  "name": handoff.name, "opacity": 0.6, **common})
    return tuple(specs)


def add_to_viewer(viewer: Any, handoff: MaskHandoff) -> Tuple[Any, ...]:
    """Add the field's layers to ``viewer``. Returns the layers it made.

    ``viewer`` is duck-typed on purpose: anything with ``add_image`` and
    ``add_labels`` will do, which is what lets the round trip be tested for
    real without napari installed.

    :param viewer: a ``napari.Viewer``, or anything shaped like one.
    :param handoff: the field.
    """
    made = []
    for spec in layer_specs(handoff):
        arguments = dict(spec)
        kind = arguments.pop("kind")
        made.append(getattr(viewer, f"add_{kind}")(**arguments))
    return tuple(made)


def open_in_napari(handoff: MaskHandoff, *, viewer: Any = None,
                   title: str = "") -> Any:
    """Open the field in napari and return the viewer.

    Does **not** start an event loop: see :func:`run_event_loop` for why that
    is a separate decision.

    :param handoff: the field.
    :param viewer: an existing viewer to add to, instead of making one.
    :param title: the window title. Defaults to the mask's filename.
    :returns: the viewer.
    :raises NapariExtraMissing: when napari is not installed.
    """
    if viewer is None:
        napari = require_napari()
        viewer = napari.Viewer(
            title=title or os.path.basename(handoff.mask_path) or "spaCR mask")
    add_to_viewer(viewer, handoff)
    return viewer


def run_event_loop() -> None:
    """Block until the napari window is closed. **Headless callers only.**

    Deliberately its own function rather than a flag on
    :func:`open_in_napari`, because whether it may be called depends on where
    the caller is, and getting it wrong is not a small mistake:

    * from a script or a notebook it is what makes ``correct_mask`` mean
      "correct it, then take it back";
    * from **inside spaCR's own Qt application it must never be called** —
      there is already a running ``QApplication`` event loop, and starting a
      second one nests them. The GUI screen therefore opens the viewer and
      lets the user press "Take the mask back" when they are done, which is
      also the friendlier interaction.

    :raises NapariExtraMissing: when napari is not installed.
    """
    require_napari().run()


def labels_from_viewer(viewer: Any, *,
                       name: str = LABELS_LAYER_NAME) -> np.ndarray:
    """Take the corrected labels back out of a viewer.

    :param viewer: a ``napari.Viewer``, or anything whose ``layers`` are
        iterable and carry ``name`` and ``data``.
    :param name: which layer to take. Falls back to the only labels-shaped
        layer when the name is not found, because a user who renamed the
        layer has not thereby thrown their work away.
    :returns: the layer's array, converted with :func:`to_spacr_mask`.
    :raises MaskFidelityError: when there is no such layer, or when what came
        back is not a mask spaCR can write.
    """
    layers = list(getattr(viewer, "layers", ()) or ())
    chosen = next((layer for layer in layers
                   if str(getattr(layer, "name", "")) == name), None)
    if chosen is None:
        candidates = [layer for layer in layers if _is_labels_layer(layer)]
        if len(candidates) == 1:
            chosen = candidates[0]
    if chosen is None:
        available = ", ".join(str(getattr(layer, "name", "?"))
                              for layer in layers) or "none"
        raise MaskFidelityError(
            f"no labels layer called {name!r} in the viewer (layers: "
            f"{available}). Rename the layer back, or pass name= the one you "
            f"edited — spaCR will not guess which of several is the mask.")
    return to_spacr_mask(getattr(chosen, "data", chosen))


def _is_labels_layer(layer: Any) -> bool:
    """Whether a viewer layer holds labels rather than an image.

    Asked of the *layer*, not of its array, because a 16-bit microscope
    image and a label mask are both 2-D integer arrays and no heuristic over
    the values can reliably tell them apart. napari's own layers carry
    ``_type_string`` — ``"image"``, ``"labels"``, ``"points"`` — which is the
    layer's own answer, so it is what is asked first. The class name is the
    fallback, and the array shape is the last resort for a layer object that
    is neither napari's nor named after what it is.
    """
    kind = str(getattr(layer, "_type_string", "") or "").strip().lower()
    if kind:
        return kind == "labels"
    name = type(layer).__name__.lower()
    if "label" in name:
        return True
    if "image" in name:
        return False
    data = getattr(layer, "data", None)
    if data is None:
        return False
    array = np.asarray(data)
    return array.ndim in (2, 3) and np.issubdtype(array.dtype, np.integer)


def to_spacr_mask(data: Any) -> np.ndarray:
    """Convert an array back to spaCR's mask convention, or refuse.

    The fidelity guarantee, in one function. It casts to
    :data:`MASK_DTYPE` and does **nothing else**: no transpose, no flip, no
    relabelling, no renumbering. A napari 2-D layer's axes are
    ``(row, column)``, the same order numpy and spaCR use, so any reordering
    here would be an invented one.

    :param data: whatever came out of the labels layer.
    :returns: a ``uint16`` array with the same shape and the same label
        values.
    :raises MaskFidelityError: for a negative label, a fractional value, a
        label that does not fit in ``uint16``, or an array that is not 2-D or
        3-D. Each is refused rather than repaired, because every repair would
        silently change which pixel belongs to which cell.
    """
    array = np.asarray(data)
    if array.ndim not in (2, 3):
        raise MaskFidelityError(
            f"a label mask is 2-D or 3-D; this is {array.shape}. If you added "
            f"an RGB layer in napari, take the labels layer back instead.")
    if not np.issubdtype(array.dtype, np.integer):
        if array.size and not np.all(np.equal(np.mod(array, 1), 0)):
            raise MaskFidelityError(
                f"a label mask holds whole numbers; this {array.dtype} array "
                f"has fractional values. That is an image layer, not a labels "
                f"layer.")
        array = array.astype(np.int64)
    if array.size:
        smallest = int(array.min())
        largest = int(array.max())
        if smallest < 0:
            raise MaskFidelityError(
                f"label {smallest} is negative; spaCR masks are unsigned and "
                f"0 is background.")
        if largest > MAX_LABEL:
            raise MaskFidelityError(
                f"label {largest} does not fit in {MASK_DTYPE.__name__}, "
                f"which is what spaCR writes masks as — casting it would "
                f"silently rename that object to "
                f"{largest % (MAX_LABEL + 1)}. Relabel the mask below "
                f"{MAX_LABEL} objects, or write it yourself as a wider dtype.")
    return array.astype(MASK_DTYPE, copy=False)


# ---------------------------------------------------------------------------
# Taking it back
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorrectionResult:
    """What one round trip changed, and where it was recorded.

    :param mask_path: the file that was written, or would have been.
    :param mask: the corrected mask, in spaCR's convention.
    :param changed_pixels: how many elements differ from what was there.
    :param added: labels present now that were not before.
    :param removed: labels that were there and are gone.
    :param altered: labels present in both whose pixels moved.
    :param log_path: the curation ledger that was appended to, or ``""``.
    :param edit: the :class:`spacr.curation.CurationEdit` that was appended.
    :param written: whether the mask file was actually rewritten. False for a
        round trip that changed nothing — see :func:`write_back`.
    """

    mask_path: str
    mask: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    changed_pixels: int = 0
    added: Tuple[int, ...] = ()
    removed: Tuple[int, ...] = ()
    altered: Tuple[int, ...] = ()
    log_path: str = ""
    edit: Optional[CurationEdit] = None
    written: bool = False

    def __bool__(self) -> bool:
        """True when the mask came back different."""
        return bool(self.changed_pixels)

    @property
    def touched(self) -> Tuple[int, ...]:
        """Every label the correction affected, sorted."""
        return tuple(sorted(set(self.added) | set(self.removed)
                            | set(self.altered)))

    def describe(self) -> str:
        """One line, for a status bar and for the screen's log."""
        if not self.changed_pixels:
            return "The mask came back unchanged; nothing was written."
        parts = [f"{self.changed_pixels:,} pixel(s)"]
        if self.added:
            parts.append(f"{len(self.added)} object(s) added")
        if self.removed:
            parts.append(f"{len(self.removed)} removed")
        if self.altered:
            parts.append(f"{len(self.altered)} reshaped")
        return (f"{os.path.basename(self.mask_path)}: "
                f"{', '.join(parts)}. Recorded in "
                f"{os.path.basename(self.log_path)}.")


def _label_diff(before: np.ndarray, after: np.ndarray
                ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """``(added, removed, altered)`` between two label arrays."""
    was = {int(v) for v in np.unique(before) if v}
    now = {int(v) for v in np.unique(after) if v}
    added = tuple(sorted(now - was))
    removed = tuple(sorted(was - now))
    moved = before != after
    altered = tuple(sorted(
        {int(v) for v in np.unique(np.concatenate(
            [before[moved].ravel(), after[moved].ravel()])) if v}
        - set(added) - set(removed))) if moved.any() else ()
    return added, removed, altered


def write_back(mask_path: Any, corrected: Any, *,
               original: Optional[np.ndarray] = None,
               source: str = LOG_SOURCE,
               extra: Optional[Mapping[str, Any]] = None,
               write: bool = True) -> CorrectionResult:
    """Write a corrected mask the way spaCR writes masks, and record it.

    Two halves, and both are the point.

    The mask goes through :func:`spacr.mask_io.save_mask`, which is the one
    place spaCR decides what a mask file is — ``uint16``, LZW-compressed TIFF
    by default, ``.npy`` when the path or ``SPACR_MASK_FORMAT`` says so. A
    corrected mask written any other way would be a mask the rest of the
    pipeline reads differently from the one segmentation produced.

    The correction goes into the artefact's
    :class:`spacr.curation.CurationLog`, appended, never rewritten, so
    :func:`spacr.curation.is_curated` tells a curated dataset from a raw one
    whether the editing happened in spaCR's brush or in napari.

    A round trip that changed nothing writes nothing and records nothing.
    That is the same rule :meth:`spacr.curation.MaskCuration.end_stroke`
    applies to a stroke that moved no pixels: a ledger padded with no-op
    entries is one nobody reads, and rewriting the file would move its mtime
    and make every downstream artifact look stale for no reason.

    :param mask_path: where the mask lives. The ledger is written beside it.
    :param corrected: the corrected labels, from napari or anywhere.
    :param original: what it was before. Read from ``mask_path`` when not
        given, so the diff is against what is actually on disk.
    :param source: what to stamp a *new* ledger with. An existing ledger
        keeps its own; the tool that made each edit is recorded on the edit.
    :param extra: anything else to record on the ledger entry.
    :param write: False computes and records nothing, returning the diff
        only — for a preview.
    :returns: a :class:`CorrectionResult`.
    :raises MaskFidelityError: when the mask cannot be written faithfully, or
        when its shape changed.
    """
    target = os.fspath(mask_path)
    mask = to_spacr_mask(corrected)
    before = (to_spacr_mask(original) if original is not None
              else load_mask(target))
    if before.shape != mask.shape:
        raise MaskFidelityError(
            f"the corrected mask is {mask.shape} and the one it replaces is "
            f"{before.shape}. spaCR will not resize a mask to fit: check that "
            f"the layer handed back is the one that was handed over.")

    changed = int(np.count_nonzero(before != mask))
    added, removed, altered = _label_diff(before, mask)
    if not changed or not write:
        return CorrectionResult(mask_path=target, mask=mask,
                                changed_pixels=changed, added=added,
                                removed=removed, altered=altered)

    # The ledger goes beside the file that was actually written, not beside
    # the path that was asked for. `save_mask` resolves a bare stem to
    # `foo.tif`, and `log_path_for` keys on the full name including the
    # extension -- so writing the ledger for `foo` would leave a second,
    # orphaned history next to the one the brush writes for `foo.tif`.
    artifact = str(save_mask(target, mask))
    log = CurationLog.read_beside(artifact)
    if not log.artifact:
        log.artifact = artifact
        log.source = str(source)
    edit = log.append(EDIT_KIND, list(added or altered or removed),
                      n_changed=changed, via="napari",
                      added=list(added), removed=list(removed),
                      altered=list(altered), **dict(extra or {}))
    log_path = log.write_beside(artifact)
    LOG.info("napari correction to %s: %d pixel(s), +%d/-%d objects",
             artifact, changed, len(added), len(removed))
    return CorrectionResult(mask_path=artifact, mask=mask,
                            changed_pixels=changed, added=added,
                            removed=removed, altered=altered,
                            log_path=log_path, edit=edit, written=True)


def correct_mask(mask_path: Any, image_path: Any = "", *,
                 viewer: Any = None, block: bool = True,
                 write: bool = True,
                 name: str = LABELS_LAYER_NAME) -> CorrectionResult:
    """The whole round trip: open in napari, wait, take the mask back.

    For a script or a notebook. **Not** for use from inside spaCR's own Qt
    application with ``block=True`` — see :func:`run_event_loop`; the GUI
    screen calls :func:`open_in_napari` and :func:`labels_from_viewer`
    separately, driven by the user.

    :param mask_path: the mask to correct. The corrected one is written back
        here, and the ledger beside it.
    :param image_path: the image to show under it.
    :param viewer: an existing viewer, instead of opening one.
    :param block: run the napari event loop and return when the window
        closes. False returns as soon as the viewer is open, which is only
        useful when the caller drives the loop itself.
    :param write: passed to :func:`write_back`.
    :param name: the labels layer's name, on the way out and back.
    :returns: a :class:`CorrectionResult`.
    :raises NapariExtraMissing: when napari is not installed.
    """
    handoff = load_handoff(mask_path, image_path, name=name)
    opened = open_in_napari(handoff, viewer=viewer)
    if block:
        run_event_loop()
    return write_back(handoff.mask_path,
                      labels_from_viewer(opened, name=name),
                      original=handoff.mask, write=write)
