"""Say what each object in a mask is, after any segmenter has made it.

Segmentation answers "where is there an object"; this answers
"what is it". The two are kept apart on purpose: the same classification
must be available whether spaCR segmented the field a moment ago or is
handed a mask Cellpose wrote last week.

TWO WAYS IN:

* :func:`segment_and_classify` takes a segmentation backend and its
  options, segments, and classifies what it made.
* :func:`classify_objects` takes an image and a mask that already exists
  and classifies it as it stands. Nothing is re-segmented.

WHAT A HEAD IS. A head is a trained classifier: it is handed a batch of
crops and returns a class and a probability for each. It is a protocol
rather than a base class, so an adapter around a torch module or a
scikit-learn estimator can supply it. Two applications are planned -- parasites per
vacuole, and what each object is -- and the PV one is three heads in
practice, trained on Hoechst, the parasite stain and CellMask separately,
so a plate counts from whichever stain it has.

Geometry reports border contact and suggests pairs that may have been split
by a segmenter. A shared boundary is a heuristic, not proof that two labels
belong to one biological object. Border contact and split candidates are
reported separately; automatic merging excludes either object touching the
field edge. Merge accuracy still requires independently labelled examples.

IDS ARE NEVER RENUMBERED. Everything here keys on the label ids the mask
arrived with, and every mask it returns keeps them. `canonical_labels` in
`mask_engine` says what renumbering costs: the measurements, the crops and
the tracks are all keyed on those ids.
"""
from __future__ import annotations

import logging
import math
import operator
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .crop_source import crop_object

LOG = logging.getLogger(__name__)

#: The classes the object head answers with. The two "cut" classes are
#: separate from their whole counterparts because a cut object is not a
#: measurement anybody should use, and separate from each other because
#: only one of them can be repaired here.
OBJECT_CLASSES: Tuple[str, ...] = (
    "cell", "nucleus", "cut cell", "cut nucleus", "artifact",
)

#: How the parasite count is reported once a head has predicted it. The
#: head predicts a COUNT; these are the bins the count is shown in. A
#: vacuole of three stays a three in the model and is binned only for the
#: report.
PV_BINS: Tuple[int, ...] = (1, 2, 4, 8, 16)


def bin_parasite_count(count: float) -> str:
    """Put a predicted parasite count in the bin a report shows it in.

    THE BIN IS A DISPLAY CHOICE, NOT A TRAINED ONE. The head predicts how
    many parasites it sees; this is only how that number is shown. That
    matters because the rule below can change -- or be replaced by a different one
    per report -- without retraining anything.

    TIES GO DOWN, and 12 is the case: it is four from 8 and four from 16.
    A vacuole of 12 has finished eight and is partway to sixteen, so it is
    reported as the round it has completed rather than the one it has
    not.

    :param count: finite, nonnegative count, including fractional predictions.
    :returns: one of ``"1"``, ``"2"``, ``"4"``, ``"8"``, ``"16"`` or
        ``">16"``.
    :raises ValueError: the count is negative or not finite.
    """
    value = float(count)
    if not math.isfinite(value) or value < 0:
        raise ValueError("count must be finite and nonnegative")
    if value > PV_BINS[-1]:
        return f">{PV_BINS[-1]}"
    nearest = min(PV_BINS, key=lambda edge: (abs(edge - value), edge))
    return str(nearest)


class ParasiteCountHead:
    """Adapt a count regressor to the object-head prediction interface.

    :param regressor: model with ``predict(crops)`` returning one finite,
        nonnegative count per crop, shaped ``(N,)`` or ``(N, 1)``. The
        caller selects this model's stain through ``classify_objects``'s
        ``channels`` argument; separate models can use the same labels.

    Counts retain their regression values. Display classes use
    :func:`bin_parasite_count`; a regressor supplies no class probability.
    """

    def __init__(self, regressor: Any):
        self.regressor = regressor

    def predict(self, crops: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
        """Return unrounded counts and display classes for a crop batch.

        :param crops: channel-selected crops in object order.
        :returns: one count, display class and absent probability per crop.
        :raises ValueError: predictions have the wrong shape or invalid counts.
        """
        values = np.asarray(self.regressor.predict(crops), dtype=float)
        if values.shape == (len(crops), 1):
            values = values[:, 0]
        if values.shape != (len(crops),):
            raise ValueError("count predictions must have one value per crop")
        return [{"count": float(value), "class": bin_parasite_count(value),
                 "probability": None} for value in values]


def _label_mask(mask: np.ndarray) -> np.ndarray:
    """Validate a nonempty 2-D image of nonnegative integer labels."""
    field = np.asarray(mask)
    if field.ndim != 2 or not all(field.shape):
        raise ValueError("mask must be a nonempty 2-D label image")
    if field.dtype.kind not in "iu" or np.any(field < 0):
        raise ValueError("mask labels must be nonnegative integers")
    return field


def _labels_of(mask: np.ndarray) -> np.ndarray:
    """Every object id in ``mask``, background excluded."""
    ids = np.unique(np.asarray(mask))
    return ids[ids != 0]


def touches_border(mask: np.ndarray, label: int) -> bool:
    """Whether object ``label`` runs into the edge of the field.

    :param mask: the label image.
    :param label: the object.
    :returns: True when any of its pixels is in the first or last row or
        column, which is what makes it a cut object nothing here can mend.
    """
    field = np.asarray(mask)
    if field.size == 0:
        return False
    edges = (field[0, :], field[-1, :], field[:, 0], field[:, -1])
    return any(bool(np.any(edge == label)) for edge in edges)


def _perimeter_pixels(field: np.ndarray, label: int) -> int:
    """How many of ``label``'s pixels sit against something that is not it."""
    here = field == label
    if not here.any():
        return 0
    padded = np.pad(here, 1, constant_values=False)
    neighbours = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                  & padded[1:-1, :-2] & padded[1:-1, 2:])
    return int(np.count_nonzero(here & ~neighbours))


def _shared_border(field: np.ndarray, first: int, second: int) -> int:
    """How many pixels of ``first`` are orthogonally against ``second``."""
    here = field == first
    there = field == second
    if not here.any() or not there.any():
        return 0
    padded = np.pad(there, 1, constant_values=False)
    against = (padded[:-2, 1:-1] | padded[2:, 1:-1]
               | padded[1:-1, :-2] | padded[1:-1, 2:])
    return int(np.count_nonzero(here & against))


def split_candidates(mask: np.ndarray,
                     share: float = 0.25) -> List[Tuple[int, int]]:
    """Pairs of labels that look like one object cut in two.

    The rule is a geometry heuristic: two labels are a
    candidate when they touch, and the boundary they share is at least
    ``share`` of the smaller one's perimeter. A cell lying against another
    cell shares a little of its edge; a cell the segmenter cut down the
    middle shares most of it.

    This finds candidates. It does not merge them -- :func:`merge_halves`
    does, and only when asked, because a wrong merge destroys two objects
    to make one.

    :param mask: the label image.
    :param share: fraction of the smaller perimeter that must be shared,
        between zero and one. Shared pixels are counted on the lower-id
        label, with orthogonal adjacency. The default 0.25 has synthetic
        checks but no measured biological merge precision.
    :returns: pairs ``(low id, high id)``, each pair once.
    """
    field = _label_mask(mask)
    if not math.isfinite(share) or not 0 <= share <= 1:
        raise ValueError("share must be finite and between zero and one")
    padded = np.pad(field, 1)
    boundary = ((field != padded[:-2, 1:-1])
                | (field != padded[2:, 1:-1])
                | (field != padded[1:-1, :-2])
                | (field != padded[1:-1, 2:])) & (field != 0)
    ids, counts = np.unique(field[boundary], return_counts=True)
    perimeters = dict(zip(map(int, ids), map(int, counts)))
    contacts = []
    for first, second, axis in ((field[:-1], field[1:], 0),
                                (field[:, :-1], field[:, 1:], 1)):
        rows, cols = np.nonzero((first != second) & (first != 0) & (second != 0))
        if not rows.size:
            continue
        left, right = first[rows, cols], second[rows, cols]
        first_pixel = (rows.astype(np.uint64) * field.shape[1]
                       + cols.astype(np.uint64))
        second_pixel = first_pixel + (field.shape[1] if axis == 0 else 1)
        contacts.append(np.column_stack((
            np.minimum(left, right).astype(np.uint64),
            np.maximum(left, right).astype(np.uint64),
            np.where(left < right, first_pixel, second_pixel))))
    if not contacts:
        return []
    unique_pixels = np.unique(np.concatenate(contacts), axis=0)
    pairs, shared_counts = np.unique(unique_pixels[:, :2], axis=0,
                                     return_counts=True)
    return [(int(first), int(second))
            for (first, second), shared in zip(pairs, shared_counts)
            if shared / min(perimeters[int(first)], perimeters[int(second)])
            >= share]


def merge_halves(mask: np.ndarray,
                 pairs: Iterable[Tuple[int, int]]) -> np.ndarray:
    """Join each pair into one object, keeping the lower id.

    The lower id survives so that a merge is stable however the pairs are
    ordered, and so the object keeps the id that measurements and tracks
    already know it by wherever that is the lower one.

    :param mask: the label image.
    :param pairs: pairs of ids, as :func:`split_candidates` returns them.
    :returns: a NEW mask; the one passed in is not touched.
    :raises ValueError: a pair names background, an absent id, or a noninteger.
    """
    field = _label_mask(mask).copy()
    survivor = {int(label): int(label) for label in _labels_of(field)}

    def root(label):
        """Resolve a merge group and compress the path to its lowest id."""
        while survivor[label] != label:
            survivor[label] = survivor[survivor[label]]
            label = survivor[label]
        return label

    for first, second in pairs:
        try:
            first, second = operator.index(first), operator.index(second)
        except TypeError as exc:
            raise ValueError("merge pairs must contain integer label ids") from exc
        if first not in survivor or second not in survivor:
            raise ValueError("merge pairs must name existing nonzero labels")
        low, high = sorted((root(first), root(second)))
        survivor[high] = low
    for label in survivor:
        kept = root(label)
        if label != kept:
            field[np.asarray(mask) == label] = kept
    return field


def remove_labels(mask: np.ndarray, labels: Iterable[int]) -> np.ndarray:
    """Take objects out of a mask without renumbering what is left.

    :param mask: the label image.
    :param labels: the ids to erase.
    :returns: a NEW mask with those ids set to background.
    """
    field = np.asarray(mask).copy()
    for label in labels:
        field[field == int(label)] = 0
    return field


def describe_objects(mask: np.ndarray) -> List[Dict[str, Any]]:
    """What can be said about every object without a trained head at all.

    :param mask: the label image.
    :returns: one row per object -- ``label``, ``area``, ``touches_border``
        and ``split_with`` (the ids it may be one object with).
    """
    field = _label_mask(mask)
    candidates = split_candidates(field)
    partners: Dict[int, List[int]] = {}
    for first, second in candidates:
        partners.setdefault(first, []).append(second)
        partners.setdefault(second, []).append(first)
    rows = []
    for label in _labels_of(field):
        label = int(label)
        rows.append({
            "label": label,
            "area": int(np.count_nonzero(field == label)),
            "touches_border": touches_border(field, label),
            "split_with": sorted(partners.get(label, [])),
        })
    return rows


def _crops_for(image: np.ndarray, mask: np.ndarray, labels: Sequence[int], *,
               channels: Sequence[int], size: Optional[int],
               shape: str, padding: int) -> List[Optional[np.ndarray]]:
    """Cut every object out, in the order ``labels`` gives them."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    return [crop_object(array, np.asarray(mask), int(label),
                        channels=channels, shape=shape, size=size,
                        padding=padding)
            for label in labels]


def classify_objects(image: np.ndarray, mask: np.ndarray, *,
                     head: Optional[Any] = None,
                     channels: Optional[Sequence[int]] = None,
                     size: Optional[int] = None,
                     shape: str = "bounding_box",
                     padding: int = 0,
                     remove_artifacts: bool = False,
                     merge_split: bool = False,
                     artifact_class: str = "artifact") -> Dict[str, Any]:
    """Classify every object of a mask that already exists.

    :param image: the field, ``(H, W)`` or ``(H, W, planes)``.
    :param mask: its labels. Not modified; any mask returned is a new one.
    :param head: something with ``predict(crops)`` returning a class per
        crop, ``(class, probability)`` pairs, or dictionaries containing
        ``class``, optional ``probability`` and optional regression ``count``.
        None reports geometry and crops without making predictions.
    :param channels: which planes the head sees, in order. Defaults to
        every plane the image has.
    :param size: resize each crop to ``size x size`` for the head.
    :param shape: ``bounding_box`` or ``object``; see
        :func:`spacr.crop_source.crop_object`.
    :param padding: pixels of context around each object.
    :param remove_artifacts: return a mask with whatever the head called
        an artifact erased. OFF by default: a false artifact is a deleted
        object, so this is a decision the caller makes rather than a
        default they inherit.
    :param merge_split: merge pairs :func:`split_candidates` finds, excluding
        any pair with either object touching the field edge. This optional
        geometry rule can mistake adjacent objects for a segmentation split.
    :param artifact_class: which class name means "not a real object", for
        ``remove_artifacts``. A parameter rather than a constant because a
        head trained on somebody else's labels may spell it differently,
        and a removal that silently matches nothing is worse than one that
        cannot be configured.
    :returns: dictionary containing ``objects`` (per-label rows), ``crops``
        (label-keyed arrays), a new ``mask``, ``merged`` pairs, ``removed``
        ids and ``removed_count``. Rows and crops describe objects after
        merging but before removal, retaining evidence for deleted objects.
        ``label_map`` maps every original id to its final surviving id,
        or zero for a removed object. Default mask values and dtype are
        unchanged. Geometry candidates are not validated biological splits.
    :raises ValueError: invalid image/mask geometry, crop settings or head
        output, or artifact removal requested without a head.
    """
    source = _label_mask(mask)
    array = np.asarray(image)
    if (array.ndim not in (2, 3) or array.shape[:2] != source.shape
            or (array.ndim == 3 and array.shape[2] == 0)):
        raise ValueError("image shape must match the 2-D mask, with channels last")
    planes = array.shape[2] if array.ndim == 3 else 1
    try:
        wanted = ([operator.index(channel) for channel in channels]
                  if channels is not None else list(range(planes)))
        padding = operator.index(padding)
        size = operator.index(size) if size is not None else None
    except TypeError as exc:
        raise ValueError("channels, padding and size must be integers") from exc
    if not wanted or any(channel < 0 or channel >= planes for channel in wanted):
        raise ValueError("channels must select existing image planes")
    if padding < 0 or (size is not None and size <= 0):
        raise ValueError("padding must be nonnegative and size must be positive")
    if shape not in ("bounding_box", "object"):
        raise ValueError("crop shape must be bounding_box or object")
    if remove_artifacts and head is None:
        raise ValueError("artifact removal requires a classification head")
    field = source.copy()
    merged: List[Tuple[int, int]] = []
    if merge_split:
        merged = [pair for pair in split_candidates(field)
                  if not (touches_border(field, pair[0])
                          or touches_border(field, pair[1]))]
        field = merge_halves(field, merged)
    rows = describe_objects(field)
    labels = [row["label"] for row in rows]
    crops = _crops_for(array, field, labels, channels=wanted, size=size,
                       shape=shape, padding=padding)
    if head is not None and labels:
        for row, verdict in zip(rows, _predictions(head, crops)):
            row.update(verdict)
    removed: List[int] = []
    if remove_artifacts:
        removed = [row["label"] for row in rows
                   if row.get("class") == artifact_class]
        if removed:
            field = remove_labels(field, removed)
    original_ids, first_pixels = np.unique(source, return_index=True)
    label_map = {int(label): int(field.flat[index])
                 for label, index in zip(original_ids, first_pixels) if label}
    return {"objects": rows, "crops": dict(zip(labels, crops)),
            "mask": field, "merged": merged, "removed": removed,
            "removed_count": len(removed), "label_map": label_map}


def _predictions(head: Any, crops: Sequence[Optional[np.ndarray]]):
    """One ``{"class": ..., "probability": ...}`` per crop.

    An object the cropper could not cut out -- it is not in the mask any
    more, which a merge can do -- is reported with no class rather than
    being dropped, so the rows still line up with the objects.
    """
    usable = [crop for crop in crops if crop is not None]
    answers = list(head.predict(usable)) if usable else []
    if len(answers) != len(usable):
        raise ValueError(
            f"the head answered {len(answers)} of {len(usable)} crops; a "
            f"classification has to line up with the objects it is about")
    answered = iter(answers)
    for crop in crops:
        if crop is None:
            yield {"class": None, "probability": None}
            continue
        answer = next(answered)
        if isinstance(answer, Mapping):
            if "class" not in answer:
                raise ValueError("a head prediction must contain a class")
            verdict = {"class": answer["class"],
                       "probability": answer.get("probability")}
            if "count" in answer:
                count = float(answer["count"])
                verdict["count"] = count
                verdict["class"] = bin_parasite_count(count)
        elif isinstance(answer, tuple) and len(answer) == 2:
            verdict = {"class": answer[0], "probability": answer[1]}
        else:
            verdict = {"class": answer, "probability": None}
        if verdict["probability"] is not None:
            probability = float(verdict["probability"])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("probability must be finite and between 0 and 1")
            verdict["probability"] = probability
        yield verdict


def segment_and_classify(backend: Any, image: np.ndarray, *,
                         eval_kwargs: Optional[Mapping[str, Any]] = None,
                         **classify_kwargs) -> Dict[str, Any]:
    """Segment a field with any backend, then classify what it found.

    The backend is anything with Cellpose's ``eval`` -- Cellpose-SAM,
    Cellpose 3, DINOCell and SAMCell all answer it, which is what items
    404, 405, 423 and 446 built. This function therefore wraps a model
    without knowing which model it has.

    :param backend: the segmenter.
    :param image: one field.
    :param eval_kwargs: passed to the backend's ``eval``.
    :param classify_kwargs: passed to :func:`classify_objects`.
    :returns: what :func:`classify_objects` returns, with the mask the
        backend produced under ``"mask"`` when no option changed it.
    """
    output = backend.eval(x=[np.asarray(image)], **dict(eval_kwargs or {}))
    masks = output[0] if isinstance(output, tuple) else output
    if isinstance(masks, (list, tuple)):
        if len(masks) != 1:
            raise ValueError("the backend must return one mask for one image")
        masks = masks[0]
    mask = np.asarray(masks)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError("the backend must return one mask for one image")
    return classify_objects(image, mask, **classify_kwargs)
