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
rather than a base class, so a torch module, a scikit-learn estimator or a
stub in a test are all equally a head. Two are planned -- parasites per
vacuole, and what each object is -- and the PV one is three heads in
practice, trained on Hoechst, the parasite stain and CellMask separately,
so a plate counts from whichever stain it has.

WHAT THIS MODULE DECIDES WITHOUT A HEAD, because geometry is not a matter
of opinion: whether an object touches the image border, and which pairs of
labels look like one object the segmenter cut in two. Those are the "cut"
classes, and the two kinds are told apart because only a segmenter-split object can be merged where it
stands, because the other half of a border object is in the next field.

IDS ARE NEVER RENUMBERED. Everything here keys on the label ids the mask
arrived with, and every mask it returns keeps them. `canonical_labels` in
`mask_engine` says what renumbering costs: the measurements, the crops and
the tracks are all keyed on those ids.
"""
from __future__ import annotations

import logging
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


def bin_parasite_count(count: int) -> str:
    """Put a predicted parasite count in the bin a report shows it in.

    THE BIN IS A DISPLAY CHOICE, NOT A TRAINED ONE. The head predicts how
    many parasites it sees; this is only how that number is shown. That
    matters because the rule below can change -- or be replaced by a different one
    per report -- without retraining anything.

    TIES GO DOWN, and 12 is the case: it is four from 8 and four from 16.
    A vacuole of 12 has finished eight and is partway to sixteen, so it is
    reported as the round it has completed rather than the one it has
    not.

    :param count: how many parasites the head counted.
    :returns: one of ``"1"``, ``"2"``, ``"4"``, ``"8"``, ``"16"`` or
        ``">16"``.
    """
    value = int(count)
    if value > PV_BINS[-1]:
        return f">{PV_BINS[-1]}"
    nearest = min(PV_BINS, key=lambda edge: (abs(edge - value), edge))
    return str(nearest)


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

    THE RULE, and it is deliberately conservative: two labels are a
    candidate when they touch, and the boundary they share is at least
    ``share`` of the smaller one's perimeter. A cell lying against another
    cell shares a little of its edge; a cell the segmenter cut down the
    middle shares most of it.

    This finds candidates. It does not merge them -- :func:`merge_halves`
    does, and only when asked, because a wrong merge destroys two objects
    to make one.

    :param mask: the label image.
    :param share: the fraction of the smaller object's perimeter that must
        be shared. 0.25 was chosen as the default and is measured in the
        tests rather than argued for here.
    :returns: pairs ``(low id, high id)``, each pair once.
    """
    field = np.asarray(mask)
    ids = _labels_of(field)
    if ids.size < 2:
        return []
    perimeters = {int(one): _perimeter_pixels(field, int(one)) for one in ids}
    pairs: List[Tuple[int, int]] = []
    for index, first in enumerate(ids):
        for second in ids[index + 1:]:
            first_id, second_id = int(first), int(second)
            shared = _shared_border(field, first_id, second_id)
            if not shared:
                continue
            smaller = min(perimeters[first_id], perimeters[second_id])
            if smaller and shared / smaller >= share:
                pairs.append((first_id, second_id))
    return pairs


def merge_halves(mask: np.ndarray,
                 pairs: Iterable[Tuple[int, int]]) -> np.ndarray:
    """Join each pair into one object, keeping the lower id.

    The lower id survives so that a merge is stable however the pairs are
    ordered, and so the object keeps the id that measurements and tracks
    already know it by wherever that is the lower one.

    :param mask: the label image.
    :param pairs: pairs of ids, as :func:`split_candidates` returns them.
    :returns: a NEW mask; the one passed in is not touched.
    """
    field = np.asarray(mask).copy()
    survivor: Dict[int, int] = {}
    for first, second in pairs:
        low, high = (first, second) if first < second else (second, first)
        while low in survivor:
            low = survivor[low]
        survivor[high] = low
    for gone, kept in survivor.items():
        field[field == gone] = kept
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
    field = np.asarray(mask)
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
        crop, or ``(class, probability)`` per crop. None reports geometry
        only, which is what is useful before a head is trained.
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
    :param merge_split: merge pairs :func:`split_candidates` finds. Acts on
        segmenter-split objects only -- an object cut by the field's edge
        has its other half in the next field and is left alone.
    :param artifact_class: which class name means "not a real object", for
        ``remove_artifacts``. A parameter rather than a constant because a
        head trained on somebody else's labels may spell it differently,
        and a removal that silently matches nothing is worse than one that
        cannot be configured.
    :returns: ``{"objects": rows, "mask": mask, "merged": pairs,
        "removed": ids}``. ``mask`` is the mask as it stands after the
        options asked for; with none of them it is the mask passed in.
    """
    field = np.asarray(mask)
    merged: List[Tuple[int, int]] = []
    if merge_split:
        merged = [pair for pair in split_candidates(field)
                  if not (touches_border(field, pair[0])
                          and touches_border(field, pair[1]))]
        field = merge_halves(field, merged)
    rows = describe_objects(field)
    labels = [row["label"] for row in rows]
    if head is not None and labels:
        array = np.asarray(image)
        planes = array.shape[2] if array.ndim == 3 else 1
        wanted = list(channels) if channels is not None else list(range(planes))
        crops = _crops_for(image, field, labels, channels=wanted, size=size,
                           shape=shape, padding=padding)
        for row, verdict in zip(rows, _predictions(head, crops)):
            row.update(verdict)
    removed: List[int] = []
    if remove_artifacts:
        removed = [row["label"] for row in rows
                   if row.get("class") == artifact_class]
        if removed:
            field = remove_labels(field, removed)
    return {"objects": rows, "mask": field, "merged": merged,
            "removed": removed}


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
        if isinstance(answer, tuple) and len(answer) == 2:
            yield {"class": answer[0], "probability": float(answer[1])}
        else:
            yield {"class": answer, "probability": None}


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
    mask = np.asarray(masks[0] if isinstance(masks, (list, tuple)) else masks)
    return classify_objects(image, mask, **classify_kwargs)
