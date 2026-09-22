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
    frame). Each segmented object takes the id of the marker it contains; an
    object holding no marker is dropped, and one holding two takes the marker
    it overlaps most.

    :param segmentation: one frame's instance labels, any ids.
    :param markers: the same frame's TRA markers, labelled by track id.
    :returns: ``segmentation`` relabelled by track id, 0 elsewhere.
    """
    segmentation = np.asarray(segmentation)
    markers = np.asarray(markers)
    out = np.zeros(segmentation.shape, dtype=np.int32)
    for obj in np.unique(segmentation):
        if obj == 0:
            continue
        inside = segmentation == obj
        ids, counts = np.unique(markers[inside], return_counts=True)
        keep = ids != 0
        if not keep.any():
            continue
        out[inside] = int(ids[keep][np.argmax(counts[keep])])
    return out


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
#: at the first step (measured 2026-09-22), so training reads windows of this
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
                    log: Optional[Callable[[str], None]] = None) -> List[float]:
    """Train the time head, then the whole network, on track-labelled pairs.

    A two-stage curriculum: the backbone's segmentation is already
    paid for, so it is frozen while the new head learns, then everything is
    trained at a low learning rate. Every pair is augmented identically on
    both frames before its targets are computed.

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
    :returns: the loss at every step.
    """
    torch = _torch()
    rng = np.random.default_rng(seed)
    net = net.to(device)
    losses: List[float] = []
    probs = None if weights is None else np.asarray(weights, float) / np.sum(weights)
    backbone_params = [p for n, p in net.named_parameters()
                       if not (n.startswith("head") or n.startswith("up"))]

    def run(steps, params, lr, frozen):
        """Train ``params`` for ``steps`` steps, the backbone frozen or not."""
        for p in backbone_params:
            p.requires_grad_(not frozen)
        optimiser = torch.optim.AdamW(params, lr=lr)
        net.train()
        for step in range(steps):
            pair = pairs[int(rng.choice(len(pairs), p=probs))]
            frames, labels = random_window([pair.frame_t, pair.frame_t1],
                                           [pair.labels_t, pair.labels_t1], rng)
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

    :param labels_t: frame ``t``'s objects, any ids.
    :param labels_t1: frame ``t+1``'s objects, any ids.
    :param prediction: from :func:`predict_pair`.
    :param min_successor: the successor probability an object needs.
    :param max_distance: the furthest link, in the object's diameters.
    :returns: ``{id in t: id in t+1}``.
    """
    from scipy.optimize import linear_sum_assignment

    here = object_centroids(labels_t)
    there = object_centroids(labels_t1)
    if not here or not there:
        return {}
    yy, xx = np.indices(labels_t.shape, dtype=np.float32)
    sources, points, scales = [], [], []
    for label, (_cy, _cx, diameter) in here.items():
        inside = labels_t == label
        if float(prediction["successor"][inside].mean()) < min_successor:
            continue
        scale = max(diameter, 1.0)
        py = float(np.mean(yy[inside] + prediction["vector"][0][inside] * scale))
        px = float(np.mean(xx[inside] + prediction["vector"][1][inside] * scale))
        sources.append(label)
        points.append((py, px))
        scales.append(scale)
    if not sources:
        return {}
    targets = list(there)
    centres = np.array([[there[t][0], there[t][1]] for t in targets])
    cost = np.linalg.norm(np.asarray(points)[:, None, :] - centres[None], axis=2)
    cost = cost / np.asarray(scales)[:, None]
    rows, cols = linear_sum_assignment(cost)
    return {int(sources[r]): int(targets[c]) for r, c in zip(rows, cols)
            if cost[r, c] <= max_distance}


def scramble_test(labels_t: np.ndarray, labels_t1: np.ndarray,
                  predict: Callable[[np.ndarray, np.ndarray], Dict[str, np.ndarray]],
                  frame_t: np.ndarray, frame_t1: np.ndarray, *,
                  seed: int = 0) -> Dict[str, float]:
    """Shuffle frame ``t+1``'s labels and ask whether the model finds them.

    With the next frame's ids scrambled, the only way to
    recover which object is which is to have learned motion. The share of
    objects linked to their true successor is the score; the plain IoU
    stitcher on the same pair is the reference the model has to beat.

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
              max_pairs: Optional[int] = None) -> List[_Pair]:
    """Consecutive-frame training pairs from one Cell Tracking Challenge movie.

    Frames from ``<movie>/<seq>/t*.tif``, full masks from the silver
    segmentation ``<movie>/<seq>_ST/SEG/man_seg*.tif`` relabelled by the
    tracking markers ``<movie>/<seq>_GT/TRA/man_track*.tif``
    (:func:`track_masks_from_ctc`). A frame missing any of the three is
    skipped, and a pair is only formed from two consecutive frame numbers.

    :param movie: the movie folder (e.g. ``.../ctc_dic_hela_timelapse``).
    :param sequence: ``'01'`` or ``'02'``.
    :param max_pairs: at most this many pairs, spaced evenly through the
        movie and chosen BEFORE any file is read. Measured 2026-09-22: six
        movies read whole do not fit in 48 GB -- the HSC and MuSC movies alone
        are about 3,000 frames -- so a training run takes a sample of each.
    :returns: the pairs, in time order.
    """
    import os

    import tifffile

    def indexed(folder, prefix):
        """Map frame number to path for the ``prefix*.tif`` files in ``folder``."""
        out = {}
        if not os.path.isdir(folder):
            return out
        for name in os.listdir(folder):
            if name.lower().endswith((".tif", ".tiff")) and name.startswith(prefix):
                number = _frame_number(name)
                if number is not None:
                    out[number] = os.path.join(folder, name)
        return out

    frames = indexed(os.path.join(movie, sequence), "t")
    segs = indexed(os.path.join(movie, f"{sequence}_ST", "SEG"), "man_seg")
    tracks = indexed(os.path.join(movie, f"{sequence}_GT", "TRA"), "man_track")
    usable = set(frames) & set(segs) & set(tracks)
    starts = sorted(n for n in usable if n + 1 in usable)
    if max_pairs is not None and len(starts) > max_pairs > 0:
        picks = np.linspace(0, len(starts) - 1, max_pairs).round().astype(int)
        starts = [starts[i] for i in sorted(set(picks.tolist()))]
    needed = sorted({n for s in starts for n in (s, s + 1)})
    loaded = {n: (_normalise(tifffile.imread(frames[n])),
                  track_masks_from_ctc(tifffile.imread(segs[n]),
                                       tifffile.imread(tracks[n])))
              for n in needed}
    return [_Pair(loaded[n][0], loaded[n + 1][0], loaded[n][1], loaded[n + 1][1])
            for n in starts]


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
    args = parser.parse_args(argv)
    pairs: List[_Pair] = []
    for movie in args.movies:
        for sequence in ("01", "02"):
            pairs.extend(ctc_pairs(movie, sequence,
                                   max_pairs=args.max_pairs or None))
            print(f"{movie.rsplit('/', 1)[-1]} {sequence}: {len(pairs)} pairs so far",
                  flush=True)
    if not pairs:
        raise SystemExit("no usable pairs: each movie needs NN/, NN_ST/SEG and NN_GT/TRA")
    weights = pair_sampling_weights(np.stack([p.labels_t for p in pairs]
                                             + [pairs[-1].labels_t1])) \
        if len({p.labels_t.shape for p in pairs}) == 1 else None
    if weights is not None:
        weights = weights[:len(pairs)]
        weights = weights / weights.sum()
    base = models.CellposeModel(pretrained_model=args.base,
                                gpu=args.device.startswith("cuda"))
    net = TimeflowsNet(CellposeSamFeatures(base.net))
    losses = train_timeflows(net, pairs, head_steps=args.head_steps,
                             full_steps=args.full_steps, weights=weights,
                             device=args.device, log=print)
    torch.save(net.state_dict(), args.out)
    with open(args.out + ".json", "w", encoding="utf-8") as handle:
        json.dump({"base": args.base, "movies": args.movies, "pairs": len(pairs),
                   "max_pairs_per_sequence": args.max_pairs,
                   "head_steps": args.head_steps, "full_steps": args.full_steps,
                   "final_loss": losses[-1] if losses else None}, handle, indent=2)
    print(f"saved {args.out} ({len(pairs)} pairs, final loss {losses[-1]:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
