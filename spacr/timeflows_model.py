"""Timeflows: a time head on Cellpose-SAM, trained on tracked movies.

Cellpose predicts, per pixel, a flow toward the centre
of the object the pixel belongs to. Timeflows adds the same idea through TIME:
per pixel of frame ``t``, where the centre of that object is in frame ``t+1``,
and whether it has a successor at all (cells die, divide, leave the field).
Linking objects between frames then reads the model instead of guessing from
overlap, which is where the plain IoU stitcher fails: fast cells that move
more than their own size between frames (the MuSC movies move 3-4 diameters
per frame at p95).

WHAT IS HERE, ALL RUNNABLE ON A CPU:

* :func:`track_masks_from_ctc` -- full masks carrying track ids, from a Cell
  Tracking Challenge movie's silver segmentation relabelled by its tracking
  markers (the TRA ground truth marks cells with small markers only).
* :func:`time_targets` -- the training target for one pair: a unit-diameter
  displacement field toward the successor's centre, a successor flag, and the
  pixels both are supervised on.
* :func:`augment_pair` -- one flip/rotation applied identically to both frames
  and both label images; the targets are computed AFTER it from the labels, so
  a flip can never teach a displacement that did not happen.
* :func:`pair_sampling_weights` -- pairs drawn across the displacement
  distribution rather than uniformly over mostly still frames.
* :class:`TimeflowsNet` -- a backbone (Cellpose-SAM's encoder, or any module
  with the same ``features`` contract) shared by both frames, and a head
  that reads their features together.
* :func:`timeflows_loss`, :func:`train_timeflows` -- masked losses, and the
  curriculum: the head alone on a frozen backbone, then all of it at a low
  learning rate.
* :func:`link_by_timeflows`, :func:`scramble_test` -- linking objects from the
  predictions, and the check that a model recovers identities after the next
  frame's labels are shuffled.

Nothing here starts a GPU training run; that is a separate, queued job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "track_masks_from_ctc",
    "object_centroids",
    "time_targets",
    "augment_pair",
    "pair_sampling_weights",
    "TimeflowsNet",
    "CellposeSamFeatures",
    "timeflows_loss",
    "train_timeflows",
    "predict_pair",
    "link_by_timeflows",
    "scramble_test",
    "ctc_pairs",
    "main",
]


def track_masks_from_ctc(segmentation: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Full object masks labelled with their TRACK ids.

    The Cell Tracking Challenge publishes tracking ground truth as MARKERS --
    a small blob inside each cell, labelled with its track id in every frame
    -- and full outlines separately (the silver ``ST/SEG`` masks, labelled per
    frame). Only one-to-one assignments are retained: the object contains
    exactly one marker ID and that ID overlaps no other segmented object.
    Unmarked, merged and split assignments are excluded rather than guessed.

    :param segmentation: one frame's instance labels, any ids.
    :param markers: the same frame's TRA markers, labelled by track id.
    :returns: int64 ``segmentation`` relabelled by track id, 0 elsewhere.
    :raises ValueError: annotations are not matching 2-D non-negative integer
        arrays, or marker IDs cannot be represented in int64.
    """
    return _ctc_track_masks(segmentation, markers)[0]


def _ctc_track_masks(segmentation: np.ndarray, markers: np.ndarray
                     ) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return unambiguous full track masks and auditable exclusion counts.

    :param segmentation: a 2-D non-negative integer instance-label array.
    :param markers: matching 2-D non-negative integer tracking markers.
    :returns: int64 track masks and counts, including excluded marker IDs.
        Multi-marker and duplicate-track object categories can overlap.
    :raises ValueError: shape, label type/range or int64 capacity is invalid.
    """
    segmentation, markers = np.asarray(segmentation), np.asarray(markers)
    if segmentation.ndim != 2 or segmentation.shape != markers.shape:
        raise ValueError("Full masks and tracking markers must share a 2-D shape")
    if not np.issubdtype(segmentation.dtype, np.integer) or not np.issubdtype(markers.dtype, np.integer):
        raise ValueError("Annotation masks must contain integer labels")
    if np.any(segmentation < 0) or np.any(markers < 0):
        raise ValueError("Annotation labels must be non-negative")
    if int(markers.max(initial=0)) > np.iinfo(np.int64).max:
        raise ValueError("Tracking marker IDs exceed int64 capacity")
    object_tracks, track_objects = {}, {}
    unmarked = ambiguous = 0
    for label in np.unique(segmentation):
        if not label:
            continue
        ids = np.unique(markers[segmentation == label])
        ids = [int(track) for track in ids if track]
        object_tracks[int(label)] = ids
        for track in ids:
            track_objects.setdefault(track, set()).add(int(label))
        unmarked += not ids
        ambiguous += len(ids) > 1
    output = np.zeros(segmentation.shape, np.int64)
    duplicate_objects = set()
    for labels in track_objects.values():
        if len(labels) > 1:
            duplicate_objects.update(labels)
    for label, ids in object_tracks.items():
        if len(ids) == 1 and label not in duplicate_objects:
            output[segmentation == label] = ids[0]
    kept = set(np.unique(output)) - {0}
    marker_ids = set(np.unique(markers)) - {0}
    return output, {"retained_tracks": len(kept), "unmarked_objects": unmarked,
                    "multi_marker_objects": ambiguous,
                    "duplicate_track_objects": len(duplicate_objects),
                    "markers_without_retained_full_mask": len(marker_ids - kept),
                    "excluded_track_ids": sorted(int(label) for label in marker_ids - kept)}


def object_centroids(labels: np.ndarray) -> Dict[int, Tuple[float, float, float]]:
    """``label -> (y, x, equivalent diameter)`` for every object.

    :param labels: an instance label image.
    :returns: the centroids and diameters.
    """
    labels = np.asarray(labels)
    ys, xs = np.nonzero(labels)
    out: Dict[int, Tuple[float, float, float]] = {}
    if not ys.size:
        return out
    ids = labels[ys, xs]
    order = np.argsort(ids, kind="stable")
    ids, ys, xs = ids[order], ys[order], xs[order]
    bounds = np.flatnonzero(np.diff(ids)) + 1
    for start, end in zip(np.r_[0, bounds], np.r_[bounds, ids.size]):
        area = end - start
        out[int(ids[start])] = (float(ys[start:end].mean()),
                                float(xs[start:end].mean()),
                                float(2.0 * math.sqrt(area / math.pi)))
    return out


def time_targets(labels_t: np.ndarray, labels_t1: np.ndarray
                 ) -> Dict[str, np.ndarray]:
    """What the time head must predict for one pair of labelled frames.

    Inside every object of frame ``t``:

    * ``vector`` -- ``(dy, dx)`` from the pixel to its object's centre in
      frame ``t+1``, divided by the object's diameter, so one number means
      the same thing for a large cell and a small one;
    * ``successor`` -- 1 where the object's label is present in ``t+1``,
      0 where it is not (it died, divided -- children take new labels in the
      movies checked -- or left the field).

    The vector is supervised only where there IS a successor (``vector_weight``);
    the successor flag everywhere inside an object (``object_weight``).
    Background is supervised by neither.

    :param labels_t: frame ``t``'s labels, track ids.
    :param labels_t1: frame ``t+1``'s labels, the same ids.
    :returns: ``vector`` (2, H, W), ``successor``, ``vector_weight`` and
        ``object_weight`` (H, W), all float32.
    """
    labels_t = np.asarray(labels_t)
    here = object_centroids(labels_t)
    there = object_centroids(labels_t1)
    height, width = labels_t.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    vector = np.zeros((2, height, width), np.float32)
    successor = np.zeros((height, width), np.float32)
    vector_weight = np.zeros((height, width), np.float32)
    object_weight = (labels_t > 0).astype(np.float32)
    for label, (_cy, _cx, diameter) in here.items():
        if label not in there:
            continue
        inside = labels_t == label
        ty, tx, _ = there[label]
        scale = max(diameter, 1.0)
        vector[0][inside] = (ty - yy[inside]) / scale
        vector[1][inside] = (tx - xx[inside]) / scale
        successor[inside] = 1.0
        vector_weight[inside] = 1.0
    return {"vector": vector, "successor": successor,
            "vector_weight": vector_weight, "object_weight": object_weight}


def augment_pair(frames: Sequence[np.ndarray], labels: Sequence[np.ndarray],
                 rng: np.random.Generator) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """One random flip and quarter-turn, the SAME for every frame and label.

    A flip applied to frame ``t`` and not ``t+1`` teaches a displacement that
    never happened, which is the easiest way to get a model that trains
    beautifully and tracks nothing. Targets are computed
    from the augmented labels afterwards, so the vectors follow automatically.

    :param frames: the images, ``(H, W)`` or ``(H, W, C)``.
    :param labels: the label images, ``(H, W)``.
    :param rng: the random generator.
    :returns: the augmented frames and labels.
    """
    turns = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))

    def apply(array):
        """Apply this call's quarter-turns and optional flip to ``array``."""
        out = np.rot90(array, k=turns, axes=(0, 1))
        return np.ascontiguousarray(out[:, ::-1] if flip else out)

    return [apply(f) for f in frames], [apply(l) for l in labels]


#: The side of the square the Cellpose-SAM encoder takes. Its position
#: embedding is fixed at 32 x 32 patches of 8 px; a whole 1,100 px frame fails
#: at the first step, so training reads windows of this
#: size and prediction tiles the frame with them.
TILE = 256


def _pad_to(array: np.ndarray, size: int, *, labels: bool) -> np.ndarray:
    """``array`` padded at the bottom and right to at least ``size`` square.

    :param array: ``(H, W)`` or ``(H, W, C)``.
    :param size: the smallest side wanted.
    :param labels: pad with background (0) rather than by reflection.
    :returns: the padded array.
    """
    pad_y, pad_x = max(0, size - array.shape[0]), max(0, size - array.shape[1])
    if not pad_y and not pad_x:
        return array
    widths = [(0, pad_y), (0, pad_x)] + [(0, 0)] * (array.ndim - 2)
    return np.pad(array, widths, mode="constant" if labels else "reflect")


def random_window(frames: Sequence[np.ndarray], labels: Sequence[np.ndarray],
                  rng: np.random.Generator, size: int = TILE
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """One ``size`` square, the SAME window of every frame and label.

    Centred on a random object of the first label image, jittered by up to a
    quarter window, so windows hold cells rather than empty background. A
    window shared by both frames keeps every displacement true.

    :param frames: the images.
    :param labels: the label images.
    :param rng: the random generator.
    :param size: the window's side.
    :returns: the windowed frames and labels.
    """
    frames = [_pad_to(f, size, labels=False) for f in frames]
    labels = [_pad_to(l, size, labels=True) for l in labels]
    height, width = labels[0].shape[:2]
    ids = np.unique(labels[0])
    ids = ids[ids != 0]
    if ids.size:
        ys, xs = np.nonzero(labels[0] == ids[int(rng.integers(0, ids.size))])
        cy, cx = int(ys.mean()), int(xs.mean())
        jitter = size // 4
        cy += int(rng.integers(-jitter, jitter + 1))
        cx += int(rng.integers(-jitter, jitter + 1))
    else:
        cy, cx = int(rng.integers(0, height)), int(rng.integers(0, width))
    y0 = int(np.clip(cy - size // 2, 0, height - size))
    x0 = int(np.clip(cx - size // 2, 0, width - size))
    window = (slice(y0, y0 + size), slice(x0, x0 + size))
    return ([np.ascontiguousarray(f[window]) for f in frames],
            [np.ascontiguousarray(l[window]) for l in labels])


def pair_sampling_weights(label_stack: np.ndarray, bins: int = 5) -> np.ndarray:
    """Weights that draw pairs evenly across how far their objects move.

    Most consecutive frames are nearly still, so drawing pairs uniformly shows
    the head mostly easy pairs. Each pair's median
    displacement over diameter falls in one of ``bins`` equal-width bins
    across the range seen, and a pair's weight is the inverse of its bin's
    size. Quantile bins were tried first and collapse when most pairs move
    alike, which is exactly the movie this exists for.

    :param label_stack: ``(T, H, W)`` track-labelled frames.
    :param bins: how many displacement bins.
    :returns: one weight per pair ``(t, t+1)``, summing to 1.
    """
    stack = np.asarray(label_stack)
    motion = []
    previous = object_centroids(stack[0]) if len(stack) else {}
    for t in range(1, len(stack)):
        current = object_centroids(stack[t])
        moves = [math.hypot(current[k][0] - v[0], current[k][1] - v[1]) / max(v[2], 1.0)
                 for k, v in previous.items() if k in current]
        motion.append(float(np.median(moves)) if moves else 0.0)
        previous = current
    return _motion_sampling_weights(motion, bins)


def _motion_sampling_weights(motion: Sequence[float], bins: int) -> np.ndarray:
    """Balance observed pair displacements across equal-width motion bins."""
    motion = np.asarray(motion, float)
    if not motion.size:
        return motion
    low, high = float(motion.min()), float(motion.max())
    if high <= low:
        return np.full(motion.size, 1.0 / motion.size)
    inner = np.linspace(low, high, bins + 1)[1:-1]
    which = np.searchsorted(inner, motion, side="right")
    counts = np.bincount(which, minlength=bins).astype(float)
    weights = 1.0 / counts[which]
    return weights / weights.sum()


def _torch():
    """torch, imported when a model is built rather than at module import.

    :returns: the module.
    """
    import torch

    return torch


class CellposeSamFeatures:
    """Cellpose-SAM's encoder as a feature extractor, sharing its weights.

    Replays :meth:`cellpose.vit.CPSAM.forward` up to the neck: 256 feature
    channels at 1/``ps`` of the image size. The segmentation head stays the
    network's own, so what the model already segments is not re-learned.

    :param net: a ``cellpose.vit.CPSAM`` (``CellposeModel(...).net``).
    """

    channels = 256

    def __init__(self, net):
        """Hold the network."""
        self.net = net
        self.ps = int(net.ps)

    def parameters(self):
        """The encoder's parameters, for freezing and the optimiser."""
        return self.net.parameters()

    def __call__(self, x):
        """Features for ``x`` (B, 3, H, W): (B, 256, H/ps, W/ps).

        :param x: a batch of normalised images; cast to the encoder's own
            dtype (Cellpose-SAM runs in bfloat16).
        :returns: the neck features, as float32 for the head.
        """
        import torch.nn.functional as F

        net = self.net
        x = x.to(net.encoder.patch_embed.proj.weight.dtype)
        x = F.conv2d(x, net.encoder.patch_embed.proj.weight[:, :x.shape[1]],
                     bias=net.encoder.patch_embed.proj.bias, stride=net.ps)
        x = x.permute(0, 2, 3, 1)
        if net.encoder.pos_embed is not None:
            x = x + net.encoder.pos_embed
        for block in net.encoder.blocks:
            x = block(x)
        return net.encoder.neck(x.permute(0, 3, 1, 2)).float()


def TimeflowsNet(backbone, channels: Optional[int] = None, ps: Optional[int] = None):
    """The time head on a shared backbone, as a ``torch.nn.Module``.

    Both frames go through the same backbone; their features are joined and
    read by a small head that predicts three channels at full resolution:
    ``dy``, ``dx`` (toward the successor's centre, in diameters) and a
    successor logit.

    :param backbone: callable ``(B, 3, H, W) -> (B, C, H/ps, W/ps)`` with
        ``parameters()``; :class:`CellposeSamFeatures` for the real model.
    :param channels: ``C``; read from ``backbone.channels`` when None.
    :param ps: the backbone's downsampling; read from ``backbone.ps``.
    :returns: the module.
    """
    torch = _torch()
    nn = torch.nn
    channels = int(channels or getattr(backbone, "channels"))
    ps = int(ps or getattr(backbone, "ps"))

    class _Net(nn.Module):
        """Shared backbone, joined features, a time head."""

        def __init__(self):
            """Register the backbone and build the head and upsampler.

            The backbone (or its ``net``) is registered as ``encoder`` when it
            is a module, so its weights train and move with the network.
            """
            super().__init__()
            self.backbone = backbone
            if isinstance(backbone, nn.Module):
                self.add_module("encoder", backbone)
            elif isinstance(getattr(backbone, "net", None), nn.Module):
                self.add_module("encoder", backbone.net)
            self.ps = ps
            self.head = nn.Sequential(
                nn.Conv2d(2 * channels, channels, 1), nn.GELU(),
                nn.Conv2d(channels, channels // 2, 3, padding=1), nn.GELU())
            self.up = nn.ConvTranspose2d(channels // 2, 3, kernel_size=ps,
                                         stride=ps)

        def head_parameters(self):
            """The new layers' parameters: the head and the upsampler."""
            return list(self.head.parameters()) + list(self.up.parameters())

        def forward(self, frame_t, frame_t1):
            """Join both frames' backbone features and predict the time maps.

            :param frame_t: frame ``t``, ``(B, 3, H, W)``.
            :param frame_t1: frame ``t+1``, the same shape.
            :returns: ``(B, 3, H, W)``: the vector ``(dy, dx)`` and the
                successor logit.
            """
            joined = torch.cat([self.backbone(frame_t), self.backbone(frame_t1)], 1)
            return self.up(self.head(joined))

    return _Net()


def timeflows_loss(output, targets: Dict[str, "object"]) -> "object":
    """Masked loss for one batch.

    Mean squared error on the vector where a successor exists, binary
    cross-entropy on the successor flag inside objects, background
    unsupervised.

    :param output: the net's (B, 3, H, W) output.
    :param targets: tensors from :func:`time_targets`, batched.
    :returns: the scalar loss.
    """
    torch = _torch()
    vector_w = targets["vector_weight"].unsqueeze(1)
    object_w = targets["object_weight"]
    vector_err = ((output[:, :2] - targets["vector"]) ** 2 * vector_w).sum()
    vector_loss = vector_err / vector_w.sum().clamp(min=1.0)
    succ = torch.nn.functional.binary_cross_entropy_with_logits(
        output[:, 2], targets["successor"], reduction="none")
    succ_loss = (succ * object_w).sum() / object_w.sum().clamp(min=1.0)
    return vector_loss + succ_loss


@dataclass
class _Pair:
    """One training pair: two normalised frames and their track labels."""

    frame_t: np.ndarray
    frame_t1: np.ndarray
    labels_t: np.ndarray
    labels_t1: np.ndarray


def _training_window(pair: _Pair, rng: np.random.Generator
                     ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Crop a pair and remove source supervision corrupted by the crop.

    A source mask must be complete so its diameter remains correct. If its
    successor exists in the full target frame, that mask must also remain
    complete: a cut centroid is wrong and a missing crop is not a death.
    Genuine full-frame disappearances remain supervised. Images and target
    masks are retained; only unusable source labels are removed from a copy.
    This does not establish the correctness of the supplied full-frame labels.
    """
    frames, labels = random_window([pair.frame_t, pair.frame_t1],
                                   [pair.labels_t, pair.labels_t1], rng)
    source_counts = dict(zip(*np.unique(pair.labels_t, return_counts=True)))
    target_counts = dict(zip(*np.unique(pair.labels_t1, return_counts=True)))
    cropped_target_counts = dict(zip(*np.unique(labels[1], return_counts=True)))
    excluded = []
    for label, count in zip(*np.unique(labels[0], return_counts=True)):
        if label and (count != source_counts[label] or (
                label in target_counts and
                cropped_target_counts.get(label, 0) != target_counts[label])):
            excluded.append(label)
    if excluded:
        labels[0] = labels[0].copy()
        labels[0][np.isin(labels[0], excluded)] = 0
    return frames, labels


def _training_pair_sampling_weights(pairs: Sequence[_Pair], bins: int = 5
                                    ) -> np.ndarray:
    """Measure each pair's own endpoints, including mixed-size movies.

    Pairs may be spaced through a sequence or come from different movies
    whose track IDs overlap. Joining their first frames into a stack would
    measure unrelated motion across those boundaries and omit most second
    frames. Reading each pair independently also avoids a full-stack copy.
    """
    motion = []
    for pair in pairs:
        here = object_centroids(pair.labels_t)
        there = object_centroids(pair.labels_t1)
        moves = [math.hypot(there[k][0] - v[0], there[k][1] - v[1]) / max(v[2], 1.0)
                 for k, v in here.items() if k in there]
        motion.append(float(np.median(moves)) if moves else 0.0)
    return _motion_sampling_weights(motion, bins)


def _to_input(frame: np.ndarray):
    """A frame as a (1, 3, H, W) float tensor, grey replicated to three.

    :param frame: ``(H, W)`` or ``(H, W, C)``, already normalised.
    :returns: the tensor.
    """
    torch = _torch()
    array = np.asarray(frame, np.float32)
    if array.ndim == 2:
        array = np.stack([array] * 3, 0)
    else:
        array = np.moveaxis(array[..., :3], -1, 0)
        if array.shape[0] < 3:
            array = np.concatenate([array] * 3, 0)[:3]
    return torch.from_numpy(np.ascontiguousarray(array))[None]


def train_timeflows(net, pairs: Sequence[_Pair], *, head_steps: int = 100,
                    full_steps: int = 100, lr_head: float = 1e-3,
                    lr_full: float = 1e-5, weights: Optional[np.ndarray] = None,
                    seed: int = 0, device: str = "cpu",
                    log: Optional[Callable[[str], None]] = None,
                    validation_pairs: Optional[Sequence[_Pair]] = None,
                    validation_every: Optional[int] = None,
                    on_validation: Optional[Callable[[dict], None]] = None) -> List[float]:
    """Train the time head, then the whole network, on track-labelled pairs.

    A two-stage curriculum: the backbone's segmentation is already
    paid for, so it is frozen while the new head learns, then everything is
    trained at a low learning rate. Every pair is augmented identically on
    both frames before its targets are computed.

    A training crop keeps supervision only for complete source masks and,
    when present in the full target frame, complete successor masks. A
    successor outside the tile is not a disappearance. Absences in the
    supplied full-frame labels remain supervised; this does not validate
    those annotations. Unusable crops are retried up to 32 times per step.
    Censoring prevents incorrect targets at crop boundaries but removes some
    fast-motion examples; it does not establish full-motion accuracy.

    :param net: from :func:`TimeflowsNet`.
    :param pairs: the training pairs.
    :param head_steps: steps with the backbone frozen.
    :param full_steps: steps with everything trainable.
    :param lr_head: learning rate for the head stage.
    :param lr_full: learning rate for the full stage.
    :param weights: sampling weight per pair (:func:`pair_sampling_weights`);
        uniform when None.
    :param seed: the random seed.
    :param device: ``'cpu'`` or ``'cuda'``.
    :param log: ``fn(line)`` for progress.
    :param validation_pairs: optional held-out full-frame pairs. Exact input
        overlap with training is rejected before any optimizer update.
    :param validation_every: updates between held-out checks; defaults to one
        epoch of ``len(pairs)`` sampled updates. Initial and stage-end checks
        are always included. Must be positive when supplied.
    :param on_validation: callback receiving each stratified validation report,
        including the stage, update count and current training loss. Optional
        validation never changes the returned loss-list contract.
    :returns: the loss at every step.
    :raises ValueError: no usable supervision remains after 32 sampled crops
        for a step; inspect the full masks and motion relative to the tile.
    """
    torch = _torch()
    validation_interval = None
    if validation_pairs is not None:
        import operator

        from .timeflows_validation import check_pair_holdout

        if not pairs:
            raise ValueError("Validation requires nonempty training pairs")
        check_pair_holdout(pairs, validation_pairs)
        try:
            validation_interval = len(pairs) if validation_every is None else operator.index(validation_every)
        except TypeError as exc:
            raise ValueError("validation_every must be a positive integer") from exc
        if isinstance(validation_every, bool) or validation_interval < 1:
            raise ValueError("validation_every must be a positive integer")
    elif validation_every is not None or on_validation is not None:
        raise ValueError("Validation options require validation_pairs")
    rng = np.random.default_rng(seed)
    net = net.to(device)
    losses: List[float] = []
    probs = None if weights is None else np.asarray(weights, float) / np.sum(weights)
    backbone_params = [p for n, p in net.named_parameters()
                       if not (n.startswith("head") or n.startswith("up"))]
    initial_head = ({name: value.detach().clone() for name, value in net.state_dict().items()
                     if name.startswith(("head.", "up."))}
                    if validation_interval is not None else None)

    def report_validation(stage, step):
        """Report one held-out check while preserving the training state."""
        import json

        from .timeflows_validation import validate_timeflows

        report = validate_timeflows(net, validation_pairs, device=device,
                                    seed=seed, initial_head=initial_head)
        report.update(stage=stage, step=step, completed_epochs=step // len(pairs),
                      epoch_size=len(pairs),
                      epoch_definition="len(training_pairs) sampled optimizer updates",
                      training_loss=losses[-1] if losses else None)
        if on_validation is not None:
            on_validation(report)
        if log:
            log("validation " + json.dumps({key: report[key] for key in
                 ("stage", "step", "completed_epochs", "training_loss", "results")}, allow_nan=False))

    if validation_interval is not None:
        report_validation("initial", 0)

    def run(steps, params, lr, frozen):
        """Train ``params`` for ``steps`` steps, the backbone frozen or not."""
        for p in backbone_params:
            p.requires_grad_(not frozen)
        optimiser = torch.optim.AdamW(params, lr=lr)
        net.train()
        for step in range(steps):
            for _attempt in range(32):
                pair = pairs[int(rng.choice(len(pairs), p=probs))]
                frames, labels = _training_window(pair, rng)
                if np.any(labels[0]):
                    break
            else:
                raise ValueError("No usable temporal supervision in 32 sampled windows; "
                                 "check full masks and motion relative to the training tile")
            frames, labels = augment_pair(frames, labels, rng)
            target = time_targets(labels[0], labels[1])
            batch = {k: torch.from_numpy(v)[None].to(device) for k, v in target.items()}
            output = net(_to_input(frames[0]).to(device), _to_input(frames[1]).to(device))
            loss = timeflows_loss(output, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
            if log and (step % 50 == 0 or step == steps - 1):
                log(f"{'head' if frozen else 'full'} step {step}: loss {losses[-1]:.4f}")
            if validation_interval is not None and ((step + 1) % validation_interval == 0 or step == steps - 1):
                report_validation("head" if frozen else "full", step + 1)

    run(head_steps, net.head_parameters(), lr_head, True)
    if full_steps:
        run(full_steps, [p for p in net.parameters()], lr_full, False)
    return losses


def predict_pair(net, frame_t: np.ndarray, frame_t1: np.ndarray,
                 device: str = "cpu") -> Dict[str, np.ndarray]:
    """The time head's prediction for one pair.

    :param net: a trained :func:`TimeflowsNet`.
    :param frame_t: frame ``t``, normalised.
    :param frame_t1: frame ``t+1``, normalised.
    :param device: where to run.
    :returns: ``vector`` (2, H, W) and ``successor`` probability (H, W).
        The frame is read in :data:`TILE`-pixel tiles with a quarter-tile
        overlap and the overlaps averaged; the vectors are in each object's
        own diameters, so a tile needs no context beyond the object.
    """
    torch = _torch()
    net = net.to(device).eval()
    height, width = frame_t.shape[:2]
    a = _pad_to(frame_t, TILE, labels=False)
    b = _pad_to(frame_t1, TILE, labels=False)
    padded_h, padded_w = a.shape[:2]
    stride = TILE * 3 // 4

    def starts(extent):
        """Tile origins covering ``extent`` with a quarter-tile overlap."""
        found = list(range(0, max(1, extent - TILE + 1), stride))
        if found[-1] + TILE < extent:
            found.append(extent - TILE)
        return found

    total = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    count = np.zeros((padded_h, padded_w), dtype=np.float32)
    with torch.no_grad():
        for y0 in starts(padded_h):
            for x0 in starts(padded_w):
                window = (slice(y0, y0 + TILE), slice(x0, x0 + TILE))
                out = net(_to_input(np.ascontiguousarray(a[window])).to(device),
                          _to_input(np.ascontiguousarray(b[window])).to(device))[0]
                total[:, y0:y0 + TILE, x0:x0 + TILE] += out.float().cpu().numpy()
                count[y0:y0 + TILE, x0:x0 + TILE] += 1.0
    out = (total / np.maximum(count, 1.0))[:, :height, :width]
    return {"vector": out[:2], "successor": 1.0 / (1.0 + np.exp(-out[2]))}


def link_by_timeflows(labels_t: np.ndarray, labels_t1: np.ndarray,
                      prediction: Dict[str, np.ndarray], *,
                      min_successor: float = 0.5,
                      max_distance: float = 1.0) -> Dict[int, int]:
    """Which object in ``t+1`` each object in ``t`` becomes, from the model.

    Each object's pixels vote, through the predicted vectors, for where its
    centre will be; the vote's mean is matched to the nearest object centre in
    ``t+1`` with the Hungarian method. An object the model gives no successor,
    or whose best match is further than ``max_distance`` of its own diameter,
    is left unlinked.

    The distance gate is applied before assignment. Leaving an object
    unmatched costs the distance limit (with an inclusive boundary), so an
    impossible edge cannot displace a valid link. The objective minimizes
    distance plus unmatched costs; it does not maximize the number of links.
    Non-finite or out-of-range foreground predictions raise ``ValueError``;
    background predictions do not participate.

    :param labels_t: frame ``t``'s objects, any ids.
    :param labels_t1: frame ``t+1``'s objects, any ids.
    :param prediction: from :func:`predict_pair`.
    :param min_successor: the successor probability an object needs.
    :param max_distance: the furthest link, in the object's diameters.
    :returns: ``{id in t: id in t+1}``.
    """
    from scipy.optimize import linear_sum_assignment

    if not np.isfinite(min_successor) or not 0 <= min_successor <= 1:
        raise ValueError("min_successor must be finite and between 0 and 1")
    if not np.isfinite(max_distance) or max_distance < 0:
        raise ValueError("max_distance must be finite and non-negative")
    labels_t, labels_t1 = np.asarray(labels_t), np.asarray(labels_t1)
    here = object_centroids(labels_t)
    there = object_centroids(labels_t1)
    if not here or not there:
        return {}
    vector = np.asarray(prediction["vector"])
    successor = np.asarray(prediction["successor"])
    if vector.shape != (2,) + labels_t.shape or successor.shape != labels_t.shape:
        raise ValueError("Timeflows prediction shape must match the source frame")
    foreground = labels_t != 0
    probabilities = successor[foreground]
    if (not np.isfinite(vector[:, foreground]).all()
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError("Timeflows foreground prediction must be finite with probabilities in [0, 1]")
    yy, xx = np.indices(labels_t.shape, dtype=np.float32)
    sources, points, scales = [], [], []
    for label, (_cy, _cx, diameter) in here.items():
        inside = labels_t == label
        if float(successor[inside].mean()) < min_successor:
            continue
        scale = max(diameter, 1.0)
        with np.errstate(over="ignore", invalid="ignore"):
            py = float(np.mean(yy[inside] + vector[0][inside] * scale))
            px = float(np.mean(xx[inside] + vector[1][inside] * scale))
        if not math.isfinite(py) or not math.isfinite(px):
            raise ValueError("Timeflows prediction produces a non-finite object centre")
        sources.append(label)
        points.append((py, px))
        scales.append(scale)
    if not sources:
        return {}
    targets = list(there)
    centres = np.array([[there[t][0], there[t][1]] for t in targets])
    delta = np.asarray(points)[:, None, :] - centres[None]
    cost = np.hypot(delta[..., 0], delta[..., 1])
    cost = cost / np.asarray(scales)[:, None]
    transposed = len(sources) > len(targets)
    distance = cost.T if transposed else cost
    nr, nc = distance.shape
    allowed = np.isfinite(distance) & (distance <= max_distance)
    assignment = np.full((nr, nc + nr), np.nextafter(1.0, np.inf))
    assignment[:, :nc] = np.inf
    if max_distance == 0:
        assignment[:, :nc][allowed] = 0
    else:
        np.divide(distance, max_distance, out=assignment[:, :nc], where=allowed)
    rows, cols = linear_sum_assignment(assignment)
    pairs = ((c, r) if transposed else (r, c)
             for r, c in zip(rows, cols) if c < nc)
    return {int(sources[r]): int(targets[c]) for r, c in pairs}


def scramble_test(labels_t: np.ndarray, labels_t1: np.ndarray,
                  predict: Callable[[np.ndarray, np.ndarray], Dict[str, np.ndarray]],
                  frame_t: np.ndarray, frame_t1: np.ndarray, *,
                  seed: int = 0) -> Dict[str, float]:
    """Shuffle frame ``t+1``'s labels and ask whether the model finds them.

    Scrambling removes the numeric-ID shortcut, but object positions and
    overlap remain informative. A good score alone does not prove learned
    motion. The share of objects linked to their true successor is the
    score; the plain IoU stitcher on the same pair is the reference the
    model has to beat.

    :param labels_t: frame ``t``, track ids.
    :param labels_t1: frame ``t+1``, the same track ids.
    :param predict: ``fn(frame_t, frame_t1) -> prediction``.
    :param frame_t: frame ``t``.
    :param frame_t1: frame ``t+1``.
    :param seed: the shuffle's seed.
    :returns: ``{'model': share, 'iou': share, 'objects': n}``.
    """
    from .timeflows_baseline import link_frames

    rng = np.random.default_rng(seed)
    ids = [i for i in np.unique(labels_t1) if i]
    shuffled_ids = rng.permutation(ids)
    mapping = dict(zip(ids, shuffled_ids))
    scrambled = np.zeros_like(labels_t1)
    for old, new in mapping.items():
        scrambled[labels_t1 == old] = new
    truth = {i: mapping[i] for i in np.unique(labels_t) if i and i in mapping}
    if not truth:
        return {"model": float("nan"), "iou": float("nan"), "objects": 0}
    links = link_by_timeflows(labels_t, scrambled, predict(frame_t, frame_t1))
    model_share = sum(links.get(i) == truth[i] for i in truth) / len(truth)
    iou_links = {a: b for a, b, _iou in link_frames(labels_t, scrambled)}
    iou_share = sum(iou_links.get(i) == truth[i] for i in truth) / len(truth)
    return {"model": float(model_share), "iou": float(iou_share),
            "objects": len(truth)}


def _frame_number(name: str) -> Optional[int]:
    """The frame number in a CTC file name (``t012.tif``, ``man_seg012.tif``).

    :param name: the file name.
    :returns: the number, or None.
    """
    import re

    digits = re.findall(r"(\d+)", name)
    return int(digits[-1]) if digits else None


def _normalise(image: np.ndarray) -> np.ndarray:
    """Stretch a frame between its 1st and 99th percentiles, onto 0..1.

    :param image: the frame.
    :returns: float32 in 0..1.
    """
    image = np.asarray(image, np.float32)
    low, high = np.percentile(image, (1, 99))
    return np.clip((image - low) / max(high - low, 1e-6), 0, 1).astype(np.float32)


def ctc_pairs(movie: str, sequence: str = "01",
              max_pairs: Optional[int] = None, *, segmentation: str = "ST") -> List[_Pair]:
    """Consecutive-frame training pairs from one Cell Tracking Challenge movie.

    Frames from ``<movie>/<seq>/t*.tif``, full masks from the silver
    segmentation ``<movie>/<seq>_ST/SEG/man_seg*.tif`` relabelled by the
    tracking markers ``<movie>/<seq>_GT/TRA/man_track*.tif``
    (:func:`track_masks_from_ctc`). A frame missing any of the three is
    skipped, and a pair is only formed from two consecutive frame numbers.
    Slice-mask filenames are ignored and duplicate frame numbers are rejected.
    With ``segmentation='GT'``, full masks are read from ``<seq>_GT/SEG``.
    A source whose next-frame marker lacks an unambiguous full mask is
    censored for that pair, rather than labelled as a disappearance.

    :param movie: the movie folder (e.g. ``.../ctc_dic_hela_timelapse``).
    :param sequence: ``'01'`` or ``'02'``.
    :param max_pairs: at most this many pairs, spaced evenly through the
        movie and chosen before any file is read. Set this limit to bound
        the frames loaded from long movies; loading entire collections can
        require tens of gigabytes of memory.
    :param segmentation: ``'ST'`` for silver masks (the training default), or
        ``'GT'`` for supplied ground-truth full masks during validation.
    :returns: the pairs, in time order.
    :raises ValueError: sequence/limit, duplicate frame identities or annotation
        arrays are invalid.
    """
    import os
    import re

    import tifffile

    if len(sequence) != 2 or not sequence.isascii() or not sequence.isdecimal():
        raise ValueError("CTC sequences must be two-digit directory names")
    if max_pairs is not None and max_pairs < 0:
        raise ValueError("The pair limit must be non-negative")
    if segmentation not in ("ST", "GT"):
        raise ValueError("CTC segmentation must be ST or GT")

    def indexed(folder, prefix):
        """Map frame number to path for the ``prefix*.tif`` files in ``folder``."""
        out = {}
        if not os.path.isdir(folder):
            return out
        pattern = re.compile(re.escape(prefix) + r"(\d+)\.tiff?$", re.IGNORECASE)
        for name in os.listdir(folder):
            match = pattern.fullmatch(name)
            if match:
                number = int(match[1])
                if number in out:
                    raise ValueError(f"Duplicate frame {number} in {folder}")
                out[number] = os.path.join(folder, name)
        return out

    frames = indexed(os.path.join(movie, sequence), "t")
    segs = indexed(os.path.join(movie, f"{sequence}_{segmentation}", "SEG"), "man_seg")
    tracks = indexed(os.path.join(movie, f"{sequence}_GT", "TRA"), "man_track")
    usable = set(frames) & set(segs) & set(tracks)
    starts = sorted(n for n in usable if n + 1 in usable)
    if max_pairs is not None and len(starts) > max_pairs > 0:
        picks = np.linspace(0, len(starts) - 1, max_pairs).round().astype(int)
        starts = [starts[i] for i in sorted(set(picks.tolist()))]
    needed = sorted({n for s in starts for n in (s, s + 1)})
    loaded = {}
    for n in needed:
        labels, counts = _ctc_track_masks(tifffile.imread(segs[n]), tifffile.imread(tracks[n]))
        loaded[n] = (_normalise(tifffile.imread(frames[n])), labels, counts)
    pairs = []
    for n in starts:
        source = loaded[n][1]
        excluded = loaded[n + 1][2]["excluded_track_ids"]
        if excluded:
            source = source.copy()
            source[np.isin(source, excluded)] = 0
        pairs.append(_Pair(loaded[n][0], loaded[n + 1][0], source, loaded[n + 1][1]))
    return pairs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m spacr.timeflows_model --movies DIR [DIR ...] --out FILE``.

    Trains the time head on Cell Tracking Challenge movies (both sequences of
    each), starting from ``--base`` (a Cellpose checkpoint or ``cpsam``), and
    saves the head and, after the full stage, the whole network.

    :param argv: arguments; ``sys.argv[1:]`` when None.
    :returns: the exit status.
    """
    import argparse
    import json
    from contextlib import ExitStack
    from pathlib import Path

    import torch
    from cellpose import models

    parser = argparse.ArgumentParser(prog="python -m spacr.timeflows_model")
    parser.add_argument("--movies", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="where to save the model")
    parser.add_argument("--base", default="cpsam")
    parser.add_argument("--head-steps", type=int, default=2000)
    parser.add_argument("--full-steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-pairs", type=int, default=60,
                        help="pairs per movie sequence, spaced evenly (0 = all)")
    parser.add_argument("--validation-movies", nargs="+",
                        help="held-out CTC movies for checks during training")
    parser.add_argument("--validation-segmentation", choices=("GT", "ST"), default="GT",
                        help="full-mask source for validation; ST is silver annotation")
    parser.add_argument("--validation-max-pairs", type=int, default=3,
                        help="held-out pairs per movie sequence, spaced evenly (0 = all)")
    parser.add_argument("--validation-every", type=int,
                        help="sampled updates per check; default is one training-pair-count epoch")
    args = parser.parse_args(argv)
    if args.validation_every is not None and (not args.validation_movies or args.validation_every < 1):
        parser.error("--validation-every requires --validation-movies and a positive interval")
    if args.validation_movies:
        training_movies = {Path(movie).resolve() for movie in args.movies}
        if any(Path(movie).resolve() in training_movies for movie in args.validation_movies):
            parser.error("Validation movies must be separate from training movies, including aliases")
    pairs: List[_Pair] = []
    for movie in args.movies:
        for sequence in ("01", "02"):
            pairs.extend(ctc_pairs(movie, sequence,
                                   max_pairs=args.max_pairs or None))
            print(f"{movie.rsplit('/', 1)[-1]} {sequence}: {len(pairs)} pairs so far",
                  flush=True)
    if not pairs:
        raise SystemExit("no usable pairs: each movie needs NN/, NN_ST/SEG and NN_GT/TRA")
    validation_pairs = None
    validation_info = {"enabled": False}
    if args.validation_movies:
        import hashlib

        from . import timeflows_validation
        from .timeflows_validation import check_pair_holdout

        validation_pairs = []
        for movie in args.validation_movies:
            for sequence in ("01", "02"):
                validation_pairs.extend(ctc_pairs(movie, sequence,
                    max_pairs=args.validation_max_pairs or None,
                    segmentation=args.validation_segmentation))
        fingerprints = check_pair_holdout(pairs, validation_pairs)
        validation_info = {"enabled": True, "movies": args.validation_movies,
                           "segmentation": args.validation_segmentation,
                           "pairs": len(validation_pairs),
                           "interval_updates": args.validation_every or len(pairs),
                           "epoch_size": len(pairs), "input_fingerprints": fingerprints,
                           "temporal_assignment": timeflows_validation.temporal_assignment_policy(),
                           "model_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "scoring_code_sha256": hashlib.sha256(Path(timeflows_validation.__file__).read_bytes()).hexdigest(),
                           "holdout_check": "resolved movie paths and exact normalized encoder inputs",
                           "log_path": args.out + ".validation.jsonl",
                           "scope": "Linking given supplied full masks, not end-to-end tracking"}
    weights = _training_pair_sampling_weights(pairs)
    with ExitStack() as stack:
        validation_file = (stack.enter_context(open(validation_info["log_path"], "x", encoding="utf-8"))
                           if validation_pairs is not None else None)

        def record_validation(report):
            """Flush each check immediately; completion is recorded only after saving."""
            validation_file.write(json.dumps({"event": "validation", **report}, allow_nan=False) + "\n")
            validation_file.flush()
            validation_info["reports"] = validation_info.get("reports", 0) + 1

        if validation_file is not None:
            validation_file.write(json.dumps({"event": "configuration", **validation_info}) + "\n")
            validation_file.flush()
        base = models.CellposeModel(pretrained_model=args.base,
                                    gpu=args.device.startswith("cuda"))
        net = TimeflowsNet(CellposeSamFeatures(base.net))
        validation_kwargs = ({"validation_pairs": validation_pairs,
                              "validation_every": args.validation_every,
                              "on_validation": record_validation}
                             if validation_pairs is not None else {})
        losses = train_timeflows(net, pairs, head_steps=args.head_steps,
                                 full_steps=args.full_steps, weights=weights,
                                 device=args.device, log=print, **validation_kwargs)
        torch.save(net.state_dict(), args.out)
        with open(args.out + ".json", "w", encoding="utf-8") as handle:
            json.dump({"base": args.base, "movies": args.movies, "pairs": len(pairs),
                   "max_pairs_per_sequence": args.max_pairs,
                   "head_steps": args.head_steps, "full_steps": args.full_steps,
                   "sampling": {"strategy": "inverse_frequency_displacement_bins",
                                "bins": 5, "weights": weights.tolist()},
                   "window_supervision": {
                       "policy": "complete_source_and_present_successor_masks",
                       "tile_size": TILE, "maximum_attempts_per_step": 32,
                       "full_frame_disappearances_supervised": True},
                   "annotation_assignment": {
                       "policy": "one_object_per_track_one_track_per_object",
                       "missing_successor_full_mask": "censor_source_supervision"},
                   "validation": validation_info,
                   "final_loss": losses[-1] if losses else None}, handle, indent=2)
        if validation_file is not None:
            validation_file.write(json.dumps({"event": "training_complete", "updates": len(losses)}) + "\n")
    final_loss = f"{losses[-1]:.4f}" if losses else "n/a"
    print(f"saved {args.out} ({len(pairs)} pairs, final loss {final_loss})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
