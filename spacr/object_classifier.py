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
import os
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
        """Retain the stain-specific count regressor without loading or fitting it."""
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
    Cellpose 3, DINOCell and SAMCell all answer it. This function therefore wraps a model
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


_REAL_SIDE = 24
_REAL_ROLES: Tuple[str, ...] = ("cell", "pathogen", "nucleus")
_REAL_PADDING: Dict[str, float] = {"pathogen": 1.5, "nucleus": 0.5, "cell": 0.3}
_REAL_TYPES: Tuple[str, ...] = ("cell", "nucleus", "pathogen")


def _real_features(crop: np.ndarray) -> np.ndarray:
    """Describe one crop for the real / not-real classifier.

    Each channel is scaled between its own 1st and 99th percentiles, so an
    8-bit annotation crop and a 16-bit field crop of the same object give
    the same features. The scaled crop is shrunk to a small square and
    joined with the per-channel mean, spread and bright fraction.
    """
    from skimage.transform import resize

    array = np.asarray(crop, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, :, None]
    low, high = np.percentile(array, (1, 99), axis=(0, 1))
    scaled = np.clip((array - low) / np.maximum(high - low, 1e-6), 0, 1)
    small = resize(scaled, (_REAL_SIDE, _REAL_SIDE, scaled.shape[2]),
                   order=1, anti_aliasing=True, preserve_range=True)
    stats = np.concatenate([scaled.mean(axis=(0, 1)), scaled.std(axis=(0, 1)),
                            (scaled > 0.5).mean(axis=(0, 1))])
    return np.concatenate([small.ravel(), stats]).astype(np.float32)


def _read_real_crop(path: str) -> np.ndarray:
    """An annotation crop as ``(H, W, 3)``, CellMask / parasite / Hoechst."""
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    return array


class _RealObjectHead:
    """A trained real / not-real classifier answering the head protocol.

    Each crop gets ``real`` when the probability of being real reaches
    ``threshold`` and ``not real`` otherwise, with the probability of the
    class given.
    """

    def __init__(self, bundle: Mapping[str, Any], threshold: float = 0.5):
        self.estimator = bundle["estimator"]
        self.threshold = float(threshold)
        if not 0 <= self.threshold <= 1:
            raise ValueError("real_object_threshold must be between 0 and 1")

    def real_probability(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Probability that each crop shows a real object."""
        features = np.stack([_real_features(crop) for crop in crops])
        classes = list(self.estimator.classes_)
        return self.estimator.predict_proba(features)[:, classes.index(1)]

    def predict(self, crops: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
        """``{"class", "probability"}`` per crop."""
        if not len(crops):
            return []
        answers = []
        for probability in self.real_probability(crops):
            real = probability >= self.threshold
            answers.append({"class": "real" if real else "not real",
                            "probability": float(probability if real
                                                 else 1 - probability)})
        return answers


def _annotated_real_crops(databases: Sequence[str], column: str = "real"):
    """Annotated crops from Annotate's ``png_list`` tables.

    :param databases: ``measurements.db`` files whose ``png_list`` has the
        annotation column, 1 meaning real and 2 not real.
    :param column: the annotation column.
    :returns: a frame with ``png_path``, ``real`` (True or False) and
        ``group``, the plate and well each crop came from, so a split by
        group never puts one well on both sides.
    :raises ValueError: a database without the column.
    """
    import pandas as pd

    from .tabular import read_table

    frames = []
    for database in databases:
        frame = read_table(database, table="png_list", report=None)
        if column not in frame.columns:
            raise ValueError(f"{database} has no {column!r} column in "
                             f"png_list; annotate it in Annotate first")
        calls = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.loc[calls.isin([1, 2])].copy()
        frame["real"] = calls.loc[frame.index] == 1
        names = frame["png_path"].map(lambda p: os.path.splitext(
            os.path.basename(str(p)))[0].rsplit("_", 3))
        plates = (frame["dataset_plate"].astype(str)
                  if "dataset_plate" in frame.columns
                  else names.map(lambda parts: parts[0]))
        wells = names.map(lambda parts: parts[1] if len(parts) > 1 else "")
        frame["group"] = plates + "|" + wells
        frames.append(frame[["png_path", "real", "group"]])
    if not frames:
        return pd.DataFrame(columns=["png_path", "real", "group"])
    return pd.concat(frames, ignore_index=True)


def _real_scorecard(truth: np.ndarray, probability: np.ndarray,
                    threshold: float) -> Dict[str, Any]:
    """Held-out scores, with "not real" as the class being detected."""
    from sklearn.metrics import (balanced_accuracy_score, precision_score,
                                 recall_score, roc_auc_score)

    truth = np.asarray(truth, dtype=bool)
    called_real = np.asarray(probability) >= threshold
    both = truth.any() and (~truth).any()
    return {
        "n": int(truth.size), "n_not_real": int((~truth).sum()),
        "accuracy": float((called_real == truth).mean()) if truth.size else None,
        "balanced_accuracy": (float(balanced_accuracy_score(truth, called_real))
                              if both else None),
        "not_real_precision": float(precision_score(
            ~truth, ~called_real, zero_division=0)),
        "not_real_recall": float(recall_score(~truth, ~called_real,
                                              zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probability)) if both else None,
        "threshold": float(threshold),
    }


def _train_real_classifier(frame, object_type: str, *,
                           test_fraction: float = 0.2, seed: int = 0,
                           threshold: float = 0.5) -> Dict[str, Any]:
    """Train one real / not-real classifier and score it on held-out wells.

    :param frame: what :func:`_annotated_real_crops` returns.
    :param object_type: ``cell``, ``nucleus`` or ``pathogen``.
    :param test_fraction: share of the wells held out for the score.
    :param seed: the split's seed.
    :param threshold: the probability of real below which an object is
        called not real when scoring.
    :returns: the bundle to save: the estimator refitted on every crop, the
        channel roles its crops carry, the padding used to cut field crops
        and the held-out ``scorecard``.
    :raises ValueError: an unknown object type, one class only, or fewer
        than two wells.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if object_type not in _REAL_TYPES:
        raise ValueError(f"object type must be one of {list(_REAL_TYPES)}")
    truth = np.asarray(frame["real"], dtype=bool)
    groups = np.asarray(frame["group"])
    if truth.all() or not truth.any():
        raise ValueError("training needs crops called real and not real")
    if len(set(groups)) < 2:
        raise ValueError("training needs annotated crops from two wells or more")
    features = np.stack([_real_features(_read_real_crop(path))
                         for path in frame["png_path"]])

    def estimator():
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, max_iter=2000, class_weight="balanced"))

    split = GroupShuffleSplit(n_splits=1, test_size=test_fraction,
                              random_state=seed)
    train, test = next(split.split(features, truth, groups))
    scorecard = {"n_train": int(len(train)),
                 "wells_held_out": sorted(set(groups[test].tolist()))}
    if len(set(truth[train])) == 2:
        held_out = estimator().fit(features[train], truth[train].astype(int))
        probability = held_out.predict_proba(features[test])[
            :, list(held_out.classes_).index(1)]
        scorecard.update(_real_scorecard(truth[test], probability, threshold))
    else:
        scorecard["note"] = "the training wells held one class only; no score"
    final = estimator().fit(features, truth.astype(int))
    return {"version": 1, "estimator": final, "object_type": object_type,
            "roles": _REAL_ROLES, "padding_fraction": _REAL_PADDING[object_type],
            "scorecard": scorecard}


def _real_classifier_bundles(location: str) -> Dict[str, Dict[str, Any]]:
    """Load the classifiers a run names, keyed by object type.

    :param location: one saved classifier, or a folder holding
        ``cell.joblib``, ``nucleus.joblib`` and/or ``pathogen.joblib``.
    :raises ValueError: nothing loadable there.
    """
    import joblib

    location = os.path.expanduser(str(location))
    if os.path.isdir(location):
        paths = [os.path.join(location, f"{kind}.joblib") for kind in _REAL_TYPES]
        paths = [path for path in paths if os.path.isfile(path)]
    elif os.path.isfile(location):
        paths = [location]
    else:
        raise ValueError(f"real_object_classifier not found: {location}")
    bundles = {}
    for path in paths:
        bundle = joblib.load(path)
        if not isinstance(bundle, Mapping) or bundle.get("object_type") not in _REAL_TYPES:
            raise ValueError(f"{path} is not a real / not-real classifier")
        bundles[bundle["object_type"]] = dict(bundle)
    if not bundles:
        raise ValueError(f"no cell, nucleus or pathogen classifier in {location}")
    return bundles


def _drop_unreal_objects(src: str, settings: Mapping[str, Any]) -> Dict[str, int]:
    """Erase the objects a real / not-real classifier rejects from each mask.

    For every object type with a classifier and a mask folder, each field's
    objects are cut from ``stack/<field>.npy`` with the channels the
    classifier was trained on, and those called not real are set to
    background in ``masks/<type>_mask_stack/<field>.npy``. Remaining ids
    are kept. Every verdict goes to ``qc/real_object_filter_<type>.csv``.

    :param src: the plate folder holding ``stack`` and ``masks``.
    :param settings: reads ``real_object_classifier``,
        ``real_object_threshold`` and the ``*_channel`` settings.
    :returns: objects removed per object type.
    """
    import pandas as pd

    from .io import _save_array_atomic
    from .tabular import write_table

    bundles = _real_classifier_bundles(settings["real_object_classifier"])
    threshold = float(settings.get("real_object_threshold", 0.5))
    removed_counts: Dict[str, int] = {}
    for object_type, bundle in bundles.items():
        folder = os.path.join(src, "masks", f"{object_type}_mask_stack")
        planes = [settings.get(f"{role}_channel") for role in bundle["roles"]]
        if not os.path.isdir(folder):
            print(f"No {object_type} masks; its real / not-real classifier is skipped.")
            continue
        if any(plane is None for plane in planes):
            print(f"The {object_type} real / not-real classifier needs the "
                  f"{', '.join(bundle['roles'])} channels; it is skipped.")
            continue
        head = _RealObjectHead(bundle, threshold)
        rows: List[Dict[str, Any]] = []
        for name in sorted(f for f in os.listdir(folder)
                           if f.endswith(".npy") and not f.startswith(".")):
            image_path = os.path.join(src, "stack", name)
            mask_path = os.path.join(folder, name)
            mask = np.load(mask_path)
            if mask.ndim != 2 or not os.path.isfile(image_path):
                continue
            image = np.load(image_path, mmap_mode="r")
            if image.ndim == 2:
                image = image[:, :, None]
            labels, areas = np.unique(mask[mask > 0], return_counts=True)
            crops = [crop_object(np.asarray(image), mask, int(label),
                                 channels=planes, padding=int(round(
                                     bundle["padding_fraction"] * math.sqrt(area))))
                     for label, area in zip(labels, areas)]
            verdicts = list(_predictions(head, crops))
            dropped = [int(label) for label, verdict in zip(labels, verdicts)
                       if verdict["class"] == "not real"]
            if dropped:
                _save_array_atomic(mask_path, remove_labels(mask, dropped))
            rows.extend({"field": os.path.splitext(name)[0],
                         "label": int(label), "class": verdict["class"],
                         "probability": verdict["probability"]}
                        for label, verdict in zip(labels, verdicts))
            del image, mask, crops
        removed_counts[object_type] = sum(row["class"] == "not real" for row in rows)
        write_table(pd.DataFrame(rows, columns=["field", "label", "class", "probability"]),
                    os.path.join(src, "qc", f"real_object_filter_{object_type}.csv"))
        print(f"Real / not-real classifier removed {removed_counts[object_type]} "
              f"of {len(rows)} {object_type} objects.")
    return removed_counts
