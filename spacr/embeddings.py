"""A label-free embedding for every object, beside the measured features.

WHY THIS EXISTS. spaCR's two existing ways of turning an image into numbers
both require you to already know what you are looking for. The measured panel
in :mod:`spacr.measure` computes a fixed, hand-designed set -- intensity,
shape, texture, distances -- so it finds only what somebody thought to name.
The supervised path in :mod:`spacr.deep_spacr` fine-tunes a backbone against
annotations, so it finds only what somebody thought to annotate. A screen
whose real hit is "the vacuole is subtly mislocalised in a way we have no
column for" returns nothing, and returns it confidently. The cost of that gap
is invisible, which is exactly why it is worth a module.

THE CHANNEL DECISION, SETTLED HERE RATHER THAN IMPLICITLY. Pretrained
encoders expect three channels. spaCR images routinely have four or five,
with no RGB meaning at all -- a DAPI channel is not "red". Two ways out, and
the choice is recorded here rather than inherited from whatever the first
backbone happened to want:

  PER-CHANNEL, THEN CONCATENATE (:data:`CHANNEL_PER_CHANNEL`, the default).
  Each channel is encoded on its own and the vectors are joined. Channel
  identity SURVIVES: dimension 517 belongs to channel 2 and to nothing else,
  so :mod:`spacr.attribution_columns` can say which stain a hit came from and
  a reader can drop a channel without re-embedding. Costs N forward passes.

  PROJECT TO THREE (:data:`CHANNEL_PROJECT`). One pass over a fixed
  channel-to-RGB mapping. Cheaper, and irreversible: once channels are mixed,
  no downstream column can be attributed to a stain, and a screen cannot ask
  "was this the parasite channel or the host one?".

The default is per-channel because THE ATTRIBUTION IS THE POINT. An embedding
that finds an unnamed phenotype and cannot say which stain carried it has
moved the problem rather than solved it. Projection stays available for the
case where forward passes actually bind.

WHAT THIS MODULE DOES NOT DO. It does not crop -- it takes the crops
:mod:`spacr.crops` already produces. It does not write to the database. It
returns a float32 matrix keyed by object id, which is exactly the shape the
downstream already consumes: UMAP, the regression family, hit calling and
the AnnData export do not care where a numeric column came from.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, replace
from typing import (Any, Callable, Dict, List, Mapping, MutableMapping,
                    Optional, Sequence, Tuple)

import numpy as np

__all__ = [
    "EmbeddingError",
    "CHANNEL_PER_CHANNEL", "CHANNEL_PROJECT", "CHANNEL_POLICIES",
    "EMBEDDING_PREFIX", "DEFAULT_BACKBONE", "DEFAULT_POOL",
    "EmbeddingSpec", "EmbeddingResult",
    "embedding_column_names", "embed_array", "channel_of_column",
    "ENCODER_KEY_PREFIX", "encoder_key", "encoder_entry",
]

LOG = logging.getLogger("spacr.embeddings")


class EmbeddingError(ValueError):
    """Raised when embedding is refused, with the reason a caller can show."""


#: Encode each channel separately and concatenate. Keeps channel identity.
CHANNEL_PER_CHANNEL = "per_channel"

#: Map channels onto three and encode once. Mixes channels irreversibly.
CHANNEL_PROJECT = "project"

#: Every policy this module accepts.
CHANNEL_POLICIES: Tuple[str, ...] = (CHANNEL_PER_CHANNEL, CHANNEL_PROJECT)

#: THE COLUMN FAMILY. Every column this module produces starts with it, so an
#: embedding dimension can never collide with a measured feature and
#: `spacr.columns` / `spacr.column_groups` can group the family without
#: knowing how many dimensions a backbone produced.
EMBEDDING_PREFIX = "emb_"

#: Small, ImageNet-pretrained and widely benchmarked. Named rather than
#: hardcoded so a model's scorecard can pin which one produced a given
#: matrix.
DEFAULT_BACKBONE = "resnet18"

#: Global average pooling over the final feature map.
DEFAULT_POOL = "avg"


@dataclass(frozen=True)
class EmbeddingSpec:
    """How to embed, recorded so a matrix can say where it came from.

    :param backbone: encoder name, resolved through timm.
    :param channel_policy: :data:`CHANNEL_PER_CHANNEL` or
        :data:`CHANNEL_PROJECT`.
    :param channels: which channel indices to encode, in order. ``None``
        means every channel the array has.
    :param batch_size: crops per forward pass.
    :param device: torch device string, or ``None`` to choose one.
    :param normalize: divide each channel by a fixed scale before encoding,
        clipped to [0, 1]. Microscopy dynamic range varies by orders of
        magnitude between stains, and an encoder trained on photographs will
        otherwise see one channel as noise.
    :param channel_scale: that scale, one value per encoded channel in
        encoding order -- normally each channel's 99th percentile over a
        random sample of the plate. The same numbers for every crop mean an
        object's embedding does not depend on which crops share its batch.
        ``None`` lets :func:`embed_array` estimate it from the crops it is
        given.
    """

    backbone: str = DEFAULT_BACKBONE
    channel_policy: str = CHANNEL_PER_CHANNEL
    channels: Optional[Tuple[int, ...]] = None
    batch_size: int = 64
    device: Optional[str] = None
    normalize: bool = True
    channel_scale: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        """Reject a specification that could not produce a matrix.

        The check is made where the spec is built rather than where it is
        used, because one spec encodes every crop of a run: an unknown
        channel policy or a batch size below one is a typing mistake, and
        discovering it after the images are loaded wastes the load.

        :raises EmbeddingError: when ``channel_policy`` is not one of
            :data:`CHANNEL_POLICIES`, ``batch_size`` is below one, or
            ``channel_scale`` is set without ``normalize``, holds a negative
            or non-finite value, or does not give one value per channel in
            ``channels``.
        """
        if self.channel_policy not in CHANNEL_POLICIES:
            raise EmbeddingError(
                f"unknown channel policy {self.channel_policy!r}; "
                f"expected one of {', '.join(CHANNEL_POLICIES)}")
        if self.batch_size < 1:
            raise EmbeddingError("batch_size must be at least 1")
        if self.channel_scale is not None:
            _check_channel_scale(self)

    def fingerprint(self) -> str:
        """A short digest of everything that changes the numbers.

        Two matrices with the same fingerprint were produced the same way.
        A scorecard can quote it, and a cached matrix can be invalidated by
        it rather than by a timestamp.
        """
        fields = [
            self.backbone, self.channel_policy,
            "" if self.channels is None else ",".join(map(str, self.channels)),
            str(self.normalize)]
        if self.channel_scale is not None:
            fields.append(",".join(repr(v) for v in self.channel_scale))
        payload = "|".join(fields)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _check_channel_scale(spec: "EmbeddingSpec") -> None:
    """Coerce ``spec.channel_scale`` to a tuple of floats and refuse a bad one.

    Coerced because a scale read back from a run's JSON record is a list, and
    a frozen spec must stay hashable. The scale reaches
    :meth:`EmbeddingSpec.fingerprint` only when set, so an unscaled spec keeps
    the fingerprint it had before 410 and matrices cached under it stay
    named. Zero is allowed: it is what the
    estimate gives a channel that is blank across the plate, and it scales
    that channel to zeros rather than dividing by it.
    """
    try:
        scale = tuple(float(v) for v in spec.channel_scale)
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(
            f"channel_scale must be numbers, one per encoded channel; got "
            f"{spec.channel_scale!r}") from exc
    object.__setattr__(spec, "channel_scale", scale)
    if not spec.normalize:
        raise EmbeddingError(
            "channel_scale is only applied when normalize is True; drop the "
            "scale or turn normalize on")
    if not all(np.isfinite(v) and v >= 0 for v in scale):
        raise EmbeddingError(
            f"channel_scale values must be finite and not negative; got "
            f"{scale}")
    if spec.channels is not None and len(scale) != len(spec.channels):
        raise EmbeddingError(_scale_length_message(len(scale),
                                                   len(spec.channels)))


def _scale_length_message(n_scale: int, n_channels: int) -> str:
    """The one sentence both length checks give."""
    return (f"channel_scale has {n_scale} value(s) for {n_channels} encoded "
            "channel(s); give one per encoded channel, in encoding order")


@dataclass(frozen=True)
class EmbeddingResult:
    """The matrix, its column names, and how it was made.

    :param values: ``(n_objects, n_dims)`` float32.
    :param columns: one name per column, all starting with
        :data:`EMBEDDING_PREFIX`.
    :param spec: the :class:`EmbeddingSpec` that produced it.
    :param n_channels: how many channels were encoded.
    :param per_channel_dims: dimensions contributed by ONE channel.
    """

    values: np.ndarray
    columns: Tuple[str, ...]
    spec: EmbeddingSpec
    n_channels: int
    per_channel_dims: int

    def to_frame(self, object_ids: Sequence[Any]):
        """A DataFrame keyed by object id, ready to join to measurements.

        :param object_ids: one id per row, in the order the crops were given.
        :raises EmbeddingError: when the count does not match the matrix,
            because a silent misalignment here corrupts every downstream
            join and is unrecoverable afterwards.
        """
        import pandas as pd

        if len(object_ids) != self.values.shape[0]:
            raise EmbeddingError(
                f"{len(object_ids)} object ids for "
                f"{self.values.shape[0]} embedded crops")
        frame = pd.DataFrame(self.values, columns=list(self.columns))
        frame.insert(0, "object_id", list(object_ids))
        return frame


def embedding_column_names(per_channel_dims: int, channels: Sequence[int],
                           policy: str = CHANNEL_PER_CHANNEL) -> Tuple[str, ...]:
    """Names for the columns a run produces.

    THE CHANNEL IS IN THE NAME under the per-channel policy -- ``emb_c2_017``
    -- which is what makes attribution possible later without carrying a
    separate map beside the matrix. Under projection there is no channel to
    name, and the name says so rather than implying one.

    :param per_channel_dims: dimensions one forward pass produces.
    :param channels: channel indices, in encoding order.
    :param policy: which channel policy produced them.
    :returns: column names, in matrix order.
    """
    width = max(3, len(str(max(per_channel_dims - 1, 0))))
    if policy == CHANNEL_PROJECT:
        return tuple(f"{EMBEDDING_PREFIX}rgb_{i:0{width}d}"
                     for i in range(per_channel_dims))
    return tuple(f"{EMBEDDING_PREFIX}c{channel}_{i:0{width}d}"
                 for channel in channels
                 for i in range(per_channel_dims))


def channel_of_column(column: str) -> Optional[int]:
    """Which channel a column came from, or ``None`` if it cannot be said.

    ``None`` for a projected column is the honest answer rather than a
    guess: the channels were mixed and no dimension belongs to one stain.

    :param column: an embedding column name.
    """
    if not column.startswith(EMBEDDING_PREFIX):
        return None
    rest = column[len(EMBEDDING_PREFIX):]
    if not rest.startswith("c"):
        return None
    head = rest.split("_", 1)[0][1:]
    return int(head) if head.isdigit() else None


def _prepare(crops: np.ndarray, spec: EmbeddingSpec
             ) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Validate the stack and pick the channels, before any model is loaded."""
    array = np.asarray(crops)
    if array.ndim != 4:
        raise EmbeddingError(
            "crops must be (n, height, width, channels); "
            f"got an array with {array.ndim} dimensions")
    if array.shape[0] == 0:
        raise EmbeddingError("no crops to embed")
    channels = _encoded_channels(array.shape[3], spec)
    return array.astype(np.float32, copy=False), channels


def _encoded_channels(available: int, spec: EmbeddingSpec) -> Tuple[int, ...]:
    """The channel indices ``spec`` encodes from crops with ``available``."""
    channels = (tuple(range(available)) if spec.channels is None
                else tuple(spec.channels))
    for channel in channels:
        if not 0 <= channel < available:
            raise EmbeddingError(
                f"channel {channel} is not in the crops, which have "
                f"{available}")
    adaptive = spec.backbone.startswith(_DINO_PREFIX) or (
        spec.backbone in _FOUNDATION_MODELS
        and _FOUNDATION_MODELS[spec.backbone]["in_channels"] is None)
    if (spec.channel_policy == CHANNEL_PROJECT and len(channels) > 3
            and not adaptive):
        raise EmbeddingError(
            f"{len(channels)} channels cannot be projected onto three "
            "without choosing which to drop; name three in "
            "EmbeddingSpec.channels, or use the per-channel policy")
    if (spec.channel_scale is not None
            and len(spec.channel_scale) != len(channels)):
        raise EmbeddingError(_scale_length_message(len(spec.channel_scale),
                                                   len(channels)))
    return channels


def _scaled(plane: np.ndarray, scale: Optional[float]) -> np.ndarray:
    """One channel divided by a FIXED scale and clipped to [0, 1].

    ``None`` means the spec does not normalise and the plane passes through.
    The scale is never derived from ``plane`` here: ``plane`` is a whole
    batch, and deriving it here is what made a crop's input depend on its
    batch-mates (410).
    """
    if scale is None:
        return plane
    top = float(scale)
    if not np.isfinite(top) or top <= 0:
        return np.zeros_like(plane)
    return np.clip(plane / top, 0.0, 1.0)


#: How a plate's scale is estimated: this percentile of every pixel in a
#: random sample of this many crops, drawn with this seed, read this many
#: crops at a time. 2,048 is what the 395 harness used on plate1.
_SCALE_PERCENTILE = 99.0
_SCALE_SAMPLE_SIZE = 2048
_SCALE_SEED = 0
_SCALE_READ_BATCH = 256


def _estimate_channel_scale(crops: Any,
                            channels: Optional[Sequence[int]] = None, *,
                            sample_size: int = _SCALE_SAMPLE_SIZE,
                            seed: int = _SCALE_SEED,
                            read_batch: int = _SCALE_READ_BATCH
                            ) -> Tuple[float, ...]:
    """One scale per channel for a plate: the 99th percentile of a sample.

    ``crops`` is anything shaped ``(n, height, width, channels)`` that can be
    indexed by a sorted integer array -- an ndarray, a memmap, an HDF5
    dataset -- and it is read ``read_batch`` crops at a time, so a plate that
    does not fit in memory can still be sampled. ``sample_size`` crops are
    drawn without replacement by ``numpy.random.default_rng(seed)``; a plate
    no larger than the sample is used whole, which makes the estimate exactly
    the whole-plate percentile (and, for such a stack, exactly the number
    ``embed_array`` used before 410).

    Memory is the sample's pixels once per channel, float32 -- 2,048 crops of
    96x96 is ~75 MB a channel.

    :returns: one float per channel, in ``channels`` order (every channel
        when ``None``). A channel with no positive, finite percentile -- blank
        across the sample -- gets ``0.0``, which scales it to zeros.
    """
    shape = tuple(getattr(crops, "shape", ()))
    if len(shape) != 4:
        raise EmbeddingError(
            "crops must be (n, height, width, channels) to estimate a scale; "
            f"got shape {shape}")
    n, height, width, available = (int(v) for v in shape)
    if n == 0:
        raise EmbeddingError("no crops to estimate a scale from")
    if sample_size < 1 or read_batch < 1:
        raise EmbeddingError("sample_size and read_batch must be at least 1")
    channels = (tuple(range(available)) if channels is None
                else tuple(int(c) for c in channels))
    for channel in channels:
        if not 0 <= channel < available:
            raise EmbeddingError(
                f"channel {channel} is not in the crops, which have "
                f"{available}")

    size = min(n, sample_size)
    picked = np.sort(np.random.default_rng(seed).choice(n, size=size,
                                                        replace=False))
    per_crop = height * width
    buffers = [np.empty(size * per_crop, dtype=np.float32) for _ in channels]
    filled = 0
    for start in range(0, size, read_batch):
        index = picked[start:start + read_batch]
        chunk = np.asarray(crops[index], dtype=np.float32)
        if chunk.shape != (index.size, height, width, available):
            raise EmbeddingError(
                f"reading {index.size} crops returned shape {chunk.shape}, "
                f"not {(index.size, height, width, available)}")
        stop = filled + index.size * per_crop
        for buffer, channel in zip(buffers, channels):
            buffer[filled:stop] = chunk[..., channel].ravel()
        filled = stop

    scale = []
    for buffer in buffers:
        top = float(np.percentile(buffer, _SCALE_PERCENTILE,
                                  overwrite_input=True))
        scale.append(top if np.isfinite(top) and top > 0 else 0.0)
    return tuple(scale)


def _embed_plate(crops: Any, spec: Optional[EmbeddingSpec] = None, *,
                 encoder: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 record: Optional[MutableMapping[str, Any]] = None,
                 sample_size: int = _SCALE_SAMPLE_SIZE,
                 seed: int = _SCALE_SEED,
                 read_batch: int = _SCALE_READ_BATCH) -> EmbeddingResult:
    """Embed one plate's crops under one fixed scale per channel (410).

    The scale is estimated ONCE per run, from a random sample of the plate and
    for every channel it has, and written into ``record`` -- the run's own
    JSON-serialisable record, which the caller keeps beside the matrix:

        record["channel_scale"]        {"0": 85.0, "1": 79.0, "2": 198.0}
        record["channel_scale_sample"] {"percentile", "size", "seed", "n_crops"}

    A ``record`` that already holds a scale is reused as it stands, so a rerun
    -- another backbone, another batch size, a subset of the channels, the
    crops in another order -- sees the same numbers without estimating again.
    One that lacks a channel this run encodes is refused: it was recorded for
    a different plate, and topping it up from a new sample would mix two.

    A spec that already carries ``channel_scale``, or does not normalise, is
    passed through untouched and ``record`` is not written.
    """
    spec = spec or EmbeddingSpec()
    if spec.normalize and spec.channel_scale is None:
        shape = tuple(getattr(crops, "shape", ()))
        if len(shape) != 4:
            raise EmbeddingError(
                "crops must be (n, height, width, channels); "
                f"got shape {shape}")
        channels = _encoded_channels(int(shape[3]), spec)
        recorded = None if record is None else record.get("channel_scale")
        if recorded is None:
            everything = _estimate_channel_scale(
                crops, None, sample_size=sample_size, seed=seed,
                read_batch=read_batch)
            recorded = {str(c): v for c, v in enumerate(everything)}
            if record is not None:
                record["channel_scale"] = recorded
                record["channel_scale_sample"] = {
                    "percentile": _SCALE_PERCENTILE,
                    "size": int(min(int(shape[0]), sample_size)),
                    "seed": int(seed),
                    "n_crops": int(shape[0]),
                }
        missing = [c for c in channels if str(c) not in recorded]
        if missing:
            raise EmbeddingError(
                f"the run's record holds a channel_scale for channels "
                f"{sorted(recorded, key=int)} but this run also encodes "
                f"{missing}; that record was made for a different plate")
        spec = replace(spec, channel_scale=tuple(
            recorded[str(c)] for c in channels))
    return embed_array(crops, spec, encoder=encoder)


def embed_array(crops: np.ndarray, spec: Optional[EmbeddingSpec] = None, *,
                encoder: Optional[Callable[[np.ndarray], np.ndarray]] = None
                ) -> EmbeddingResult:
    """Embed a stack of crops.

    Every crop is divided by the same per-channel scale,
    ``spec.channel_scale``, so a crop's embedding does not depend on the other
    crops in ``crops``. When the spec carries none, one is estimated from a
    random sample of ``crops`` -- the stack is treated as the whole plate --
    and a warning is logged. To embed one plate in several calls, pass the
    same scale to each: the returned ``spec`` carries it.

    :param crops: ``(n, height, width, channels)``, the shape
        :mod:`spacr.crops` already produces.
    :param spec: how to embed; the default is per-channel resnet18.
    :param encoder: a callable taking ``(n, height, width, 3)`` float32 in
        [0, 1] and returning ``(n, dims)``. An encoder with an
        ``in_channels`` attribute takes that many planes instead, and one
        whose ``in_channels`` is ``None`` takes any number: each channel
        alone under the per-channel policy, every encoded channel in one
        pass under the projection policy. Injected by the tests so the
        wiring can be exercised without downloading a backbone; production
        callers leave it ``None``.
    :returns: an :class:`EmbeddingResult`.
    :raises EmbeddingError: on a shape or channel problem, before any model
        is loaded -- a download is a slow way to find out the array was
        wrong.
    """
    spec = spec or EmbeddingSpec()
    array, channels = _prepare(crops, spec)
    if spec.normalize and spec.channel_scale is None:
        scale = _estimate_channel_scale(array, channels)
        LOG.warning(
            "embed_array was given no channel_scale, so these %d crops were "
            "treated as the whole plate and scaled by %s. Crops embedded in "
            "another call are scaled alike only when that call is given the "
            "same channel_scale -- pass on the returned spec.",
            array.shape[0], scale)
        spec = replace(spec, channel_scale=scale)
    scales: Tuple[Optional[float], ...] = (
        spec.channel_scale if spec.normalize else (None,) * len(channels))
    run = encoder if encoder is not None else _backbone_encoder(spec)
    width = getattr(run, "in_channels", 3)

    if spec.channel_policy == CHANNEL_PROJECT:
        planes = [_scaled(array[..., c], s) for c, s in zip(channels, scales)]
        while width is not None and len(planes) < width:
            planes.append(np.zeros_like(planes[0]))
        stack = np.stack(planes if width is None else planes[:width], axis=-1)
        values = np.asarray(run(stack), dtype=np.float32)
        per_channel = values.shape[1]
    else:
        blocks = []
        for channel, scale in zip(channels, scales):
            plane = _scaled(array[..., channel], scale)
            stack = np.repeat(plane[..., None], width or 1, axis=-1)
            blocks.append(np.asarray(run(stack), dtype=np.float32))
        widths = {block.shape[1] for block in blocks}
        if len(widths) != 1:
            raise EmbeddingError(
                f"the encoder returned different widths per channel: {widths}")
        per_channel = blocks[0].shape[1]
        values = np.concatenate(blocks, axis=1)

    columns = embedding_column_names(per_channel, channels, spec.channel_policy)
    if values.shape[1] != len(columns):
        raise EmbeddingError(
            f"{values.shape[1]} embedding dimensions but {len(columns)} "
            "column names")
    return EmbeddingResult(values=values, columns=columns, spec=spec,
                           n_channels=len(channels),
                           per_channel_dims=per_channel)


def _timm_encoder(spec: EmbeddingSpec) -> Callable[[np.ndarray], np.ndarray]:
    """The real encoder: a pretrained backbone with its head removed."""
    try:
        import timm
        import torch
    except ImportError as exc:                       # pragma: no cover
        raise EmbeddingError(
            "self-supervised embeddings need torch and timm; install the "
            "`spacr[embeddings]` extra") from exc

    device = spec.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(spec.backbone, pretrained=True, num_classes=0)
    model.eval().to(device)

    def run(stack: np.ndarray) -> np.ndarray:
        """Encode one three-channel stack, in batches, under no_grad.

        Batched here rather than by the caller because the batch size belongs
        to the DEVICE, not to the channel policy: the per-channel policy calls
        this once per channel and each call must fit the same GPU.

        :param stack: ``(n, height, width, 3)`` float32 in [0, 1].
        :returns: ``(n, dims)`` float32 features.
        """
        out: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, stack.shape[0], spec.batch_size):
                chunk = stack[start:start + spec.batch_size]
                tensor = torch.from_numpy(
                    np.ascontiguousarray(chunk.transpose(0, 3, 1, 2))
                ).to(device)
                out.append(model(tensor).detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    return run



_FOUNDATION_MODELS: Dict[str, Dict[str, Any]] = {
    "openphenom": {
        "label": "OpenPhenom (Recursion, channel-agnostic MAE ViT-S/16)",
        "repo": "recursionpharma/OpenPhenom",
        "revision": "0f92333685f6e9f031b804c70fe246f9b05ae90d",
        "in_channels": None,
        "size": 256,
        "license": "Recursion non-commercial",
    },
    "chada_vit": {
        "label": "ChAda-ViT (channel-adaptive ViT-T/16, IDRCell100k)",
        "repo": "nicoboou/chadavit16-moyen",
        "revision": "91a41123650ccff12f590364fec95d76af36cc93",
        "in_channels": None,
        "size": 224,
        "license": "see the model card",
    },
    "subcell": {
        "label": "SubCell (CZI / Lundberg lab ViT-B/16, DNA + protein)",
        "url": ("https://czi-subcell-public.s3.amazonaws.com/models/"
                "DNA-Protein_ViT-ProtS-Pool.pth"),
        "in_channels": 2,
        "size": 448,
        "license": "MIT",
    },
    "cell_dino": {
        "label": "Cell-DINO (Meta FAIR DINOv2 on the Human Protein Atlas)",
        "in_channels": None,
        "size": 224,
        "license": "FAIR non-commercial research",
    },
}


def _foundation_names() -> Tuple[str, ...]:
    """The single-cell foundation models :func:`embed_array` can load by name.

    They are offered beside the ``timm`` backbones. OpenPhenom and ChAda-ViT
    are channel-adaptive and take any number of channels; SubCell takes two,
    DNA then the stain of interest; Cell-DINO is listed so a request for it
    gets a reason rather than an unknown-name error.
    """
    return tuple(_FOUNDATION_MODELS)


def _backbone_encoder(spec: EmbeddingSpec) -> Callable[[np.ndarray], np.ndarray]:
    """The encoder ``spec.backbone`` names: a DINO checkpoint, a foundation
    model or a timm backbone."""
    if spec.backbone.startswith(_DINO_PREFIX):
        return _dino_encoder(spec)
    if spec.backbone in _FOUNDATION_MODELS:
        return _foundation_encoder(spec)
    return _timm_encoder(spec)


def _foundation_encoder(spec: EmbeddingSpec) -> Callable[[np.ndarray], np.ndarray]:
    """Load one single-cell foundation model and wrap it as an encoder.

    The returned callable takes ``(n, height, width, k)`` float32 in [0, 1]
    and returns ``(n, dims)``; its ``in_channels`` attribute tells
    :func:`embed_array` how many planes ``k`` it wants, ``None`` meaning any.
    Crops are resized to the size the model was trained at. Weights are
    downloaded once, at a pinned revision, into the Hugging Face or torch
    hub cache.

    :raises EmbeddingError: when torch or transformers is missing, or the
        model's weights are not published.
    """
    try:
        import torch
    except ImportError as exc:
        raise EmbeddingError(
            "foundation-model embeddings need torch; install the "
            "`spacr[embeddings]` extra") from exc
    info = _FOUNDATION_MODELS[spec.backbone]
    device = spec.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if spec.backbone == "cell_dino":
        raise EmbeddingError(
            "Cell-DINO's weights are not published yet (Meta FAIR, "
            "facebookresearch/dinov2 README_CELL_DINO.md). Choose openphenom, "
            "chada_vit or subcell, or a timm DINOv2 backbone such as "
            "vit_small_patch14_dinov2.lvd142m.")
    if spec.backbone == "subcell":
        model, forward = _subcell_model(info, torch)
    else:
        model, forward = _hub_model(spec.backbone, info, torch)
    model.eval().to(device)
    size = int(info["size"])

    def run(stack: np.ndarray) -> np.ndarray:
        """Encode ``(n, h, w, k)`` crops in batches, resized to the model's size.

        :param stack: float32 in [0, 1], channels last.
        :returns: ``(n, dims)`` float32 features.
        """
        out: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, stack.shape[0], spec.batch_size):
                chunk = torch.from_numpy(np.ascontiguousarray(
                    stack[start:start + spec.batch_size].transpose(0, 3, 1, 2)
                )).float().to(device)
                if chunk.shape[-2:] != (size, size):
                    chunk = torch.nn.functional.interpolate(
                        chunk, size=(size, size), mode="bilinear",
                        align_corners=False)
                out.append(forward(model, chunk).detach().float().cpu().numpy())
        return np.concatenate(out, axis=0)

    run.in_channels = info["in_channels"]
    return run


def _hub_model(name: str, info: Mapping[str, Any], torch: Any):
    """A Hugging Face remote-code model and its forward pass.

    :returns: ``(model, forward)``, where ``forward(model, x)`` maps a
        ``(n, k, h, w)`` tensor in [0, 1] to ``(n, dims)``.
    """
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise EmbeddingError(
            f"{name} needs the transformers package; pip install "
            "transformers") from exc
    model = AutoModel.from_pretrained(info["repo"], revision=info["revision"],
                                      trust_remote_code=True)
    if name == "openphenom":
        model.return_channelwise_embeddings = False

        def forward(net, x):
            """OpenPhenom's mean patch token; it rescales 0-255 itself."""
            return net.predict(x * 255.0)

        return model, forward
    model.return_all_tokens = False

    def forward(net, x):
        """ChAda-ViT's class token, each crop's channels as one sequence."""
        n, k, h, w = x.shape
        flat = torch.nn.functional.instance_norm(x).reshape(n * k, 1, h, w)
        return net(flat, index=0, list_num_channels=[[k] * n])

    return model, forward


def _subcell_model(info: Mapping[str, Any], torch: Any):
    """SubCell's DNA + protein ViT encoder with its gated attention pooling.

    The published checkpoint holds a Hugging Face ViT under ``encoder.`` and
    a two-head gated attention pooler under ``pool_model.``; both are loaded
    strictly, so a changed checkpoint is refused rather than half-loaded.
    Each crop channel is min-max scaled, as SubCell's own loader does.
    """
    try:
        from transformers import ViTConfig, ViTModel
    except ImportError as exc:
        raise EmbeddingError(
            "subcell needs the transformers package; pip install "
            "transformers") from exc
    nn = torch.nn
    target = os.path.join(torch.hub.get_dir(), "checkpoints",
                          os.path.basename(info["url"]))
    if not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        torch.hub.download_url_to_file(info["url"], target)
    state = torch.load(target, map_location="cpu", weights_only=False)
    config = ViTConfig(hidden_size=768, num_hidden_layers=12,
                       num_attention_heads=12, intermediate_size=3072,
                       hidden_dropout_prob=0.0,
                       attention_probs_dropout_prob=0.0,
                       layer_norm_eps=1e-12, image_size=448, patch_size=16,
                       num_channels=2, qkv_bias=True)

    class Pooled(nn.Module):
        """The ViT's tokens pooled by two gated attention heads."""

        def __init__(self):
            """Build the ViT and the pooler the checkpoint expects."""
            super().__init__()
            self.encoder = ViTModel(config, add_pooling_layer=False)
            self.attention_v = nn.Sequential(nn.Linear(768, 512), nn.Tanh())
            self.attention_u = nn.Sequential(nn.Linear(768, 512), nn.GELU())
            self.attention = nn.Linear(512, 2)

        def forward(self, x):
            """``(n, 2, h, w)`` to ``(n, 1536)``, the two heads joined."""
            tokens = self.encoder(
                x, interpolate_pos_encoding=True).last_hidden_state
            gate = self.attention(self.attention_v(tokens)
                                  * self.attention_u(tokens))
            weights = torch.softmax(gate.permute(0, 2, 1), dim=-1)
            return torch.bmm(weights, tokens).reshape(x.shape[0], -1)

    model = Pooled()
    model.load_state_dict({
        k.replace("pool_model.", "").replace("_v.1.", "_v.0.")
        .replace("_u.1.", "_u.0."): v for k, v in state.items()})

    def forward(net, x):
        """Min-max scale each crop channel, then encode."""
        low = x.amin(dim=(2, 3), keepdim=True)
        high = x.amax(dim=(2, 3), keepdim=True)
        return net((x - low) / (high - low).clamp_min(1e-6))

    return model, forward


def _retrieval_scorecard(features: Any, labels: Mapping[Any, Any],
                         k: int = 10) -> Dict[str, float]:
    """How well one feature matrix separates known phenotypes.

    Every labelled crop is a query against all the others, compared by
    cosine similarity after the same median-centring, SD-scaling and row
    normalisation "find cells like this" uses. Three numbers come back,
    each against the chance a random ordering would give:

    - ``knn_accuracy``: share of crops whose ``k`` nearest neighbours'
      majority label (ties to the nearer neighbour) is their own.
    - ``map``: mean average precision of retrieving same-label crops over
      the full ranking, averaged over queries.
    - ``precision_at_k``: share of the ``k`` nearest that share the label.

    The whole similarity matrix is held at once, so this is meant for a
    labelled set of up to a few tens of thousands of crops.

    :param features: numeric frame indexed by crop key, such as
        :meth:`EmbeddingResult.to_frame` or the measured features.
    :param labels: crop key to class; blanks are dropped.
    :param k: neighbours per query.
    :returns: the metrics, ``chance_map``, ``chance_precision``, ``n`` and
        ``classes``.
    :raises ValueError: with fewer than two labelled crops or one class.
    """
    from .active_learning import _SimilarityIndex

    index = _SimilarityIndex(features, backend="numpy")
    pairs = [(str(key), str(value)) for key, value in labels.items()
             if value is not None and str(value) != "" and str(key) in index]
    classes = np.asarray([p[1] for p in pairs], dtype=object)
    if len(pairs) < 2 or len(set(classes)) < 2:
        raise ValueError(
            "a scorecard needs at least two labelled crops in two classes")
    rows = np.asarray([index._position[p[0]] for p in pairs], dtype=np.int64)
    sub = index._matrix[rows].astype(np.float64)
    scores = sub @ sub.T
    np.fill_diagonal(scores, -np.inf)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :-1]
    same = classes[order] == classes[:, None]
    k = max(1, min(int(k), len(pairs) - 1))
    correct = 0
    for i in range(len(pairs)):
        near = list(classes[order[i, :k]])
        counts = {c: near.count(c) for c in near}
        best = max(counts.values())
        vote = next(c for c in near if counts[c] == best)
        correct += vote == classes[i]
    ranks = np.arange(1, same.shape[1] + 1)
    positives = same.sum(axis=1)
    precision = np.cumsum(same, axis=1) / ranks
    ap = (precision * same).sum(axis=1) / np.maximum(positives, 1)
    kept = positives > 0
    freq = {c: float(np.mean(classes == c)) for c in set(classes)}
    chance = np.asarray([(freq[c] * len(pairs) - 1) / (len(pairs) - 1)
                         for c in classes])
    return {
        "knn_accuracy": float(correct / len(pairs)),
        "map": float(ap[kept].mean()),
        "precision_at_k": float(same[:, :k].mean()),
        "chance_map": float(chance[kept].mean()),
        "chance_precision": float(chance.mean()),
        "n": float(len(pairs)),
        "classes": float(len(freq)),
    }


_DINO_PREFIX = "dino:"


def _dino_planes(crops: Any, channel_policy: str,
                 channels: Optional[Sequence[int]] = None
                 ) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """The training images DINO sees, laid out as :func:`embed_array` feeds them.

    Each encoded channel is divided by the plate's fixed scale and clipped to
    [0, 1]. Under the per-channel policy every channel of every crop is its
    own one-plane image, so the backbone learns one stain at a time, as it is
    later used; under the projection policy each crop is one image with all
    its encoded channels.

    :returns: ``(images, scale)``: ``(m, k, h, w)`` float32 and the scale
        per encoded channel.
    """
    spec = EmbeddingSpec(backbone=_DINO_PREFIX, channel_policy=channel_policy,
                         channels=None if channels is None else tuple(channels))
    array, chosen = _prepare(crops, spec)
    scale = _estimate_channel_scale(array, chosen)
    planes = [_scaled(array[..., c], s) for c, s in zip(chosen, scale)]
    if channel_policy == CHANNEL_PROJECT:
        images = np.stack(planes, axis=1)
    else:
        images = np.concatenate([p[:, None] for p in planes], axis=0)
    return np.ascontiguousarray(images, dtype=np.float32), tuple(scale)


def _dino_views(batch: Any, size: int, scale: Tuple[float, float],
                generator: Any, torch: Any) -> Any:
    """One randomly augmented view of every image in ``batch``.

    A random crop covering ``scale`` of the image's area, turned by a random
    angle (cells have no up), mirrored half the time, resampled to ``size``
    pixels, then given a random gain and offset per channel and a little
    noise. Done with an affine grid so it runs batched on the training device.

    :param batch: ``(b, k, h, w)`` tensor in [0, 1].
    :returns: ``(b, k, size, size)`` tensor in [0, 1].
    """
    b, k = batch.shape[:2]
    dev = batch.device

    def uniform(low, high, *shape):
        """Uniform draws from the run's generator, on the batch's device."""
        return (torch.rand(*shape, generator=generator) * (high - low)
                + low).to(dev)

    side = torch.sqrt(uniform(scale[0], scale[1], b))
    angle = uniform(0.0, 2 * np.pi, b)
    flip = (torch.rand(b, generator=generator) < 0.5).float().to(dev) * 2 - 1
    shift = (1 - side)[:, None] * uniform(-1.0, 1.0, b, 2)
    cos, sin = torch.cos(angle) * side, torch.sin(angle) * side
    theta = torch.stack([
        torch.stack([cos * flip, -sin, shift[:, 0]], dim=1),
        torch.stack([sin * flip, cos, shift[:, 1]], dim=1)], dim=1)
    grid = torch.nn.functional.affine_grid(theta, (b, k, size, size),
                                           align_corners=False)
    view = torch.nn.functional.grid_sample(batch, grid, mode="bilinear",
                                           padding_mode="zeros",
                                           align_corners=False)
    gain = uniform(0.6, 1.4, b, k, 1, 1)
    offset = uniform(-0.1, 0.1, b, k, 1, 1)
    noise = torch.randn(view.shape, generator=generator).to(dev) * 0.02
    return (view * gain + offset + noise).clamp(0.0, 1.0)


def _dino_network(arch: str, in_chans: int, size: int, pretrained: bool,
                  out_dim: int, torch: Any):
    """A timm backbone and the DINO projection head that sits on it.

    The head is a three-layer MLP to a 128-dimensional bottleneck, L2
    normalised, then a weight-normalised linear layer onto ``out_dim``
    prototypes, as in DINO.

    :returns: ``(backbone, head)`` modules.
    """
    try:
        import timm
    except ImportError as exc:
        raise EmbeddingError(
            "DINO pretraining needs torch and timm; install the "
            "`spacr[embeddings]` extra") from exc
    nn = torch.nn
    kwargs: Dict[str, Any] = {"pretrained": pretrained, "num_classes": 0,
                              "in_chans": in_chans}
    if arch.startswith(("vit", "deit")):
        kwargs.update(img_size=size, dynamic_img_size=True)
    backbone = timm.create_model(arch, **kwargs)
    dim = backbone.num_features

    class Head(nn.Module):
        """MLP, L2 normalisation, then cosine scores against prototypes."""

        def __init__(self):
            """Build the MLP and the prototype layer."""
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(dim, 512), nn.GELU(),
                                     nn.Linear(512, 512), nn.GELU(),
                                     nn.Linear(512, 128))
            self.prototypes = nn.Parameter(torch.randn(out_dim, 128) * 0.02)

        def forward(self, x):
            """``(n, dim)`` features to ``(n, out_dim)`` prototype scores."""
            z = nn.functional.normalize(self.mlp(x), dim=-1)
            return z @ nn.functional.normalize(self.prototypes, dim=-1).T

    return backbone, Head()


def _dino_pretrain(crops: Any, path: str, *, arch: str = "resnet18",
                   channel_policy: str = CHANNEL_PER_CHANNEL,
                   channels: Optional[Sequence[int]] = None, size: int = 64,
                   epochs: int = 10, batch_size: int = 64,
                   lr: float = 5e-4, out_dim: int = 1024, n_local: int = 4,
                   momentum: float = 0.996, teacher_temp: float = 0.04,
                   student_temp: float = 0.1, pretrained: bool = False,
                   device: Optional[str] = None, seed: int = 0,
                   progress: Optional[Callable[[int, int, float], None]] = None
                   ) -> Dict[str, Any]:
    """Pretrain a backbone on unlabelled crops by self-distillation (DINO).

    A student network learns to match, from small local views and large
    global views of a crop, what a teacher -- its own slowly moving average
    -- outputs on the global views. The teacher's outputs are centred and
    sharpened so the two cannot agree by collapsing to one answer. No labels
    are used. The teacher's backbone is what is kept: pass
    ``"dino:" + path`` as :attr:`EmbeddingSpec.backbone` to embed with it.

    Training is resumable: the checkpoint at ``path`` is rewritten after
    every epoch, and a later call with the same ``path`` continues from the
    last finished epoch up to ``epochs`` (a finished run returns at once).
    A checkpoint made with another architecture, input width, crop size or
    channel policy is refused rather than overwritten.

    :param crops: ``(n, height, width, channels)``, channels last.
    :param path: the checkpoint file to write, and to resume from.
    :param arch: a timm architecture, e.g. ``resnet18`` or
        ``vit_tiny_patch16_224``.
    :param channel_policy: :data:`CHANNEL_PER_CHANNEL` trains on one stain
        at a time; :data:`CHANNEL_PROJECT` on all encoded channels together.
    :param channels: the channels to train on; all when ``None``.
    :param size: side of the global views in pixels; local views are half.
    :param pretrained: start from the architecture's ImageNet weights rather
        than from random ones.
    :param progress: called with ``(epoch, epochs, mean_loss)`` after each
        epoch.
    :returns: ``path``, ``epochs`` done, the per-epoch ``loss`` list and the
        seconds this call trained.
    :raises EmbeddingError: when torch or timm is missing, or ``path`` holds
        an incompatible checkpoint.
    """
    try:
        import torch
    except ImportError as exc:
        raise EmbeddingError(
            "DINO pretraining needs torch; install the `spacr[embeddings]` "
            "extra") from exc
    import time

    images, scale = _dino_planes(crops, channel_policy, channels)
    in_chans = int(images.shape[1])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    trained = {"arch": arch, "in_chans": in_chans, "size": int(size),
              "channel_policy": channel_policy, "out_dim": int(out_dim)}
    state = None
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        mismatch = {k: (state["setup"].get(k), v) for k, v in trained.items()
                    if state["setup"].get(k) != v}
        if mismatch:
            raise EmbeddingError(
                f"{path} holds a DINO run with different settings "
                f"({mismatch}); choose another file to start a new run")
        if state["epoch"] >= epochs:
            return {"path": path, "epochs": state["epoch"],
                    "loss": list(state["loss"]), "seconds": 0.0}

    torch.manual_seed(seed)
    student, s_head = _dino_network(arch, in_chans, size, pretrained,
                                    out_dim, torch)
    teacher, t_head = _dino_network(arch, in_chans, size, False, out_dim,
                                    torch)
    teacher.load_state_dict(student.state_dict())
    t_head.load_state_dict(s_head.state_dict())
    for module in (student, s_head, teacher, t_head):
        module.to(device)
    for p in list(teacher.parameters()) + list(t_head.parameters()):
        p.requires_grad_(False)
    params = list(student.parameters()) + list(s_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.04)
    center = torch.zeros(1, out_dim, device=device)
    losses: List[float] = []
    start = 0
    generator = torch.Generator().manual_seed(seed)
    if state is not None:
        student.load_state_dict(state["student"])
        s_head.load_state_dict(state["student_head"])
        teacher.load_state_dict(state["teacher"])
        t_head.load_state_dict(state["teacher_head"])
        optimizer.load_state_dict(state["optimizer"])
        center = state["center"].to(device)
        losses = list(state["loss"])
        start = int(state["epoch"])
        generator.set_state(state["generator"])

    data = torch.from_numpy(images)
    steps = max(1, int(np.ceil(len(data) / batch_size)))
    total = steps * epochs
    began = time.time()
    for epoch in range(start, epochs):
        student.train()
        s_head.train()
        order = torch.randperm(len(data), generator=generator)
        running = 0.0
        for step in range(steps):
            done = epoch * steps + step
            rate = lr * min(1.0, (done + 1) / max(1, steps))
            rate *= 0.5 * (1 + np.cos(np.pi * done / total))
            for group in optimizer.param_groups:
                group["lr"] = rate
            batch = data[order[step * batch_size:(step + 1) * batch_size]]
            batch = batch.to(device)
            globals_ = [_dino_views(batch, size, (0.4, 1.0), generator, torch)
                        for _ in range(2)]
            locals_ = [_dino_views(batch, max(8, size // 2), (0.1, 0.4),
                                   generator, torch) for _ in range(n_local)]
            with torch.no_grad():
                t_out = [t_head(teacher(v)) for v in globals_]
                t_prob = [torch.softmax((t - center) / teacher_temp, dim=-1)
                          for t in t_out]
            s_out = [s_head(student(v)) for v in globals_ + locals_]
            loss, pairs = 0.0, 0
            for ti, tp in enumerate(t_prob):
                for si, so in enumerate(s_out):
                    if si == ti:
                        continue
                    loss = loss + torch.sum(
                        -tp * torch.log_softmax(so / student_temp, dim=-1),
                        dim=-1).mean()
                    pairs += 1
            loss = loss / pairs
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 3.0)
            optimizer.step()
            with torch.no_grad():
                m = 1 - (1 - momentum) * (np.cos(np.pi * done / total) + 1) / 2
                for net_s, net_t in ((student, teacher), (s_head, t_head)):
                    for ps, pt in zip(net_s.parameters(), net_t.parameters()):
                        pt.mul_(m).add_(ps.detach(), alpha=1 - m)
                center = 0.9 * center + 0.1 * torch.cat(t_out).mean(
                    dim=0, keepdim=True)
            running += float(loss.detach())
        losses.append(running / steps)
        payload = {"setup": trained, "epoch": epoch + 1, "loss": losses,
                   "scale": scale, "student": student.state_dict(),
                   "student_head": s_head.state_dict(),
                   "teacher": teacher.state_dict(),
                   "teacher_head": t_head.state_dict(),
                   "optimizer": optimizer.state_dict(),
                   "center": center.cpu(), "generator": generator.get_state()}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(payload, path + ".part")
        os.replace(path + ".part", path)
        if progress is not None:
            progress(epoch + 1, epochs, losses[-1])
    return {"path": path, "epochs": epochs, "loss": losses,
            "seconds": time.time() - began}


def _dino_encoder(spec: EmbeddingSpec) -> Callable[[np.ndarray], np.ndarray]:
    """The teacher backbone of a :func:`_dino_pretrain` checkpoint, as an encoder.

    ``spec.backbone`` is ``"dino:"`` followed by the checkpoint's path. The
    encoder takes as many planes as the backbone was trained on (one under
    the per-channel policy) and resizes crops to the training size.

    :raises EmbeddingError: when the checkpoint is missing or was trained
        under the other channel policy.
    """
    import torch

    path = spec.backbone[len(_DINO_PREFIX):]
    if not os.path.exists(path):
        raise EmbeddingError(f"no DINO checkpoint at {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    trained = state["setup"]
    if trained["channel_policy"] != spec.channel_policy:
        raise EmbeddingError(
            f"{path} was trained under the {trained['channel_policy']} "
            f"policy; embed with that policy")
    size = int(trained["size"])
    model, _head = _dino_network(trained["arch"], trained["in_chans"], size,
                                 False, trained["out_dim"], torch)
    model.load_state_dict(state["teacher"])
    device = spec.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    def run(stack: np.ndarray) -> np.ndarray:
        """Encode ``(n, h, w, k)`` crops in batches at the training size.

        :param stack: float32 in [0, 1], channels last.
        :returns: ``(n, dims)`` float32 features.
        """
        out: List[np.ndarray] = []
        with torch.no_grad():
            for begin in range(0, stack.shape[0], spec.batch_size):
                chunk = torch.from_numpy(np.ascontiguousarray(
                    stack[begin:begin + spec.batch_size].transpose(0, 3, 1, 2)
                )).float().to(device)
                if chunk.shape[-2:] != (size, size):
                    chunk = torch.nn.functional.interpolate(
                        chunk, size=(size, size), mode="bilinear",
                        align_corners=False)
                out.append(model(chunk).detach().float().cpu().numpy())
        return np.concatenate(out, axis=0)

    run.in_channels = int(trained["in_chans"])
    return run


def _backbone_scorecards(crops: Any, labels: Sequence[Any],
                         specs: Mapping[str, EmbeddingSpec], *,
                         k: int = 10) -> Any:
    """Embed one labelled crop stack with several backbones and score each.

    Every backbone sees the same crops scaled by the same per-channel scale,
    and is scored by :func:`_retrieval_scorecard` on the same labels, so the
    rows compare like with like -- e.g. a :func:`_dino_pretrain` checkpoint
    against the ImageNet backbone it started from.

    :param crops: ``(n, height, width, channels)``.
    :param labels: one class per crop, in order.
    :param specs: row name to the spec to embed with.
    :returns: a frame with one row per backbone: ``dims``, the scorecard
        metrics and ``seconds``.
    """
    import time

    import pandas as pd

    array = np.asarray(crops, dtype=np.float32)
    keys = [f"crop{i}" for i in range(array.shape[0])]
    named = dict(zip(keys, labels))
    rows = []
    for name, spec in specs.items():
        began = time.time()
        if spec.normalize and spec.channel_scale is None:
            chosen = _encoded_channels(array.shape[3], spec)
            spec = replace(spec, channel_scale=_estimate_channel_scale(
                array, chosen))
        result = embed_array(array, spec)
        frame = pd.DataFrame(result.values, index=keys,
                             columns=list(result.columns))
        card = _retrieval_scorecard(frame, named, k=k)
        rows.append({"backbone": name, "dims": result.values.shape[1],
                     **card, "seconds": round(time.time() - began, 1)})
    return pd.DataFrame(rows).set_index("backbone")


_MIL_WELL_COLUMN = "wellID"
_MIL_LABEL_COLUMN = "well_label"


def _mil_well_column(frame: Any, well_column: str) -> str:
    """The well column to use.

    The one named, or a plain ``well`` column when the default is asked for
    and only that spelling is present.
    """
    if (well_column not in frame.columns and well_column == _MIL_WELL_COLUMN
            and "well" in frame.columns):
        return "well"
    return well_column


def _mil_bags(frame: Any, *, well_column: str = _MIL_WELL_COLUMN,
              label_column: str = _MIL_LABEL_COLUMN,
              feature_columns: Optional[Sequence[str]] = None,
              positive: Any = None):
    """Group a per-cell table into one bag of cells per well.

    Feature columns default to the embedding columns when the table has any,
    else to every numeric column other than the well and label columns.
    Cells with a missing feature are dropped. Every cell of a well must carry
    the same label.

    :param frame: one row per cell.
    :param well_column: column naming each cell's well.
    :param label_column: column carrying the well's label on every row.
    :param feature_columns: the columns to learn from.
    :param positive: the label value that marks a positive well; defaults to
        1 or True when the labels are 0/1 or boolean.
    :returns: ``(bags, labels, wells, rows, cells)``: one float32 array
        per well, a 0/1 array, the well names, each bag's row positions in
        ``cells``, and the rows that were kept.
    :raises ValueError: for a missing column, a well with two labels, labels
        that are not two classes, or fewer than two wells in each class.
    """
    import pandas as pd

    well_column = _mil_well_column(frame, well_column)
    for column in (well_column, label_column):
        if column not in frame.columns:
            raise ValueError(f"the table has no {column!r} column")
    if feature_columns is None:
        feature_columns = [c for c in frame.columns
                           if str(c).startswith(EMBEDDING_PREFIX)]
        if not feature_columns:
            feature_columns = [
                c for c in frame.select_dtypes("number").columns
                if c not in (well_column, label_column)]
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("the table has no numeric feature columns")
    kept = frame.dropna(
        subset=feature_columns + [well_column, label_column]
    ).reset_index(drop=True)
    values = kept[label_column]
    classes = set(values.unique().tolist())
    if positive is None:
        if classes <= {0, 1} or classes <= {True, False}:
            positive = 1
        else:
            raise ValueError(
                f"{label_column} holds {sorted(map(str, classes))}; give "
                "0/1, or name the positive label")
    if len(classes) != 2 or positive not in classes:
        raise ValueError(
            f"{label_column} must hold two classes, one of them {positive!r}")
    bags, labels, wells, rows = [], [], [], []
    matrix = kept[feature_columns].to_numpy(dtype=np.float32)
    for well, group in kept.groupby(well_column, sort=True):
        seen = group[label_column].unique()
        if len(seen) != 1:
            raise ValueError(f"well {well!r} carries more than one label")
        bags.append(matrix[group.index.to_numpy()])
        labels.append(int(seen[0] == positive))
        wells.append(well)
        rows.append(group.index.to_numpy())
    labels = np.asarray(labels, dtype=np.int64)
    if min(labels.sum(), len(labels) - labels.sum()) < 2:
        raise ValueError("well labels need at least two wells in each class")
    return bags, labels, wells, rows, kept


def _mil_fit(bags: Sequence[np.ndarray], labels: Sequence[int], *,
             hidden: int = 32, epochs: int = 60, batch: int = 8,
             lr: float = 5e-3, weight_decay: float = 1e-3,
             dropout: float = 0.1, seed: int = 0):
    """Train a gated-attention multiple-instance classifier on CPU.

    Each cell passes through one small layer; a gated attention head weighs
    the cells of a well and the weighted mean is classified (Ilse, Tomczak
    and Welling 2018). Features are standardised on the training cells.
    Only the well label is used; which cells carry it is learned.

    :param bags: one ``(cells, features)`` array per well.
    :param labels: 0/1 per well.
    :param hidden: width of the cell layer and the attention head.
    :param epochs: passes over the wells.
    :param batch: wells per step, padded and masked.
    :param lr: Adam learning rate.
    :param weight_decay: Adam weight decay.
    :param dropout: dropout on the cell layer while training.
    :param seed: seeds torch and the well order.
    :returns: a fitted model for :func:`_mil_predict`.
    """
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    stacked = np.concatenate(list(bags), axis=0).astype(np.float64)
    centre = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale[scale < 1e-8] = 1.0
    width = stacked.shape[1]

    class _Attention(nn.Module):
        """Pool cell embeddings into a well score with gated attention."""

        def __init__(self):
            """Build the cell projection, attention gate and binary well head."""
            super().__init__()
            self.cell = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                                      nn.Dropout(dropout))
            self.value = nn.Linear(hidden, hidden)
            self.gate = nn.Linear(hidden, hidden)
            self.weight = nn.Linear(hidden, 1)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x, mask):
            """Return well logits, attention and cell evidence, excluding padding."""
            h = self.cell(x)
            logits = self.weight(torch.tanh(self.value(h))
                                 * torch.sigmoid(self.gate(h))).squeeze(-1)
            logits = logits.masked_fill(~mask, float("-inf"))
            attention = torch.softmax(logits, dim=1)
            pooled = (attention.unsqueeze(-1) * h).sum(dim=1)
            evidence = (self.head(h).squeeze(-1) - self.head.bias)
            return self.head(pooled).squeeze(-1), attention, evidence

    model = _Attention()
    tensors = [torch.from_numpy(((b - centre) / scale).astype(np.float32))
               for b in bags]
    target = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(int(seed))
    model.train()
    for _ in range(int(epochs)):
        order = rng.permutation(len(tensors))
        for start in range(0, len(order), max(1, int(batch))):
            chunk = order[start:start + max(1, int(batch))]
            x, mask = _mil_pad([tensors[i] for i in chunk], torch)
            optimiser.zero_grad()
            logit, _, _ = model(x, mask)
            loss_fn(logit, target[chunk]).backward()
            optimiser.step()
    model.eval()
    return {"model": model, "centre": centre, "scale": scale}


def _mil_pad(tensors, torch):
    """Stack bags of different sizes into one padded batch and its mask."""
    longest = max(int(t.shape[0]) for t in tensors)
    x = torch.zeros(len(tensors), longest, int(tensors[0].shape[1]))
    mask = torch.zeros(len(tensors), longest, dtype=torch.bool)
    for i, t in enumerate(tensors):
        x[i, :t.shape[0]] = t
        mask[i, :t.shape[0]] = True
    return x, mask


def _mil_predict(fitted: Mapping[str, Any], bags: Sequence[np.ndarray]):
    """Well probabilities, per-cell attention and per-cell evidence.

    Attention is returned as each cell's weight times the number of cells
    in its well, so 1 is an equal share and values are comparable between
    wells of different sizes. Because the well classifier is linear, a
    well's score is the attention-weighted sum of each cell's evidence: the
    classifier applied to the cell alone, positive toward the positive
    label. Attention says which cells the model looked at; evidence says
    which way each of them pointed.

    :returns: ``(probabilities, attention, evidence)``: one probability per
        well, and one attention and one evidence array per well.
    """
    import torch

    model = fitted["model"]
    probabilities, attention, evidence = [], [], []
    with torch.no_grad():
        for bag in bags:
            x = torch.from_numpy(((np.asarray(bag, dtype=np.float64)
                                   - fitted["centre"]) / fitted["scale"])
                                 .astype(np.float32)).unsqueeze(0)
            mask = torch.ones(1, x.shape[1], dtype=torch.bool)
            logit, weights, score = model(x, mask)
            probabilities.append(float(torch.sigmoid(logit)[0]))
            attention.append(weights[0].numpy() * x.shape[1])
            evidence.append(score[0].numpy())
    return np.asarray(probabilities), attention, evidence


def _mil_summaries(bags: Sequence[np.ndarray], spread: bool = False):
    """Each well's mean cell, with the per-feature SD appended if asked."""
    rows = []
    for bag in bags:
        bag = np.asarray(bag, dtype=np.float64)
        parts = [bag.mean(axis=0)]
        if spread:
            parts.append(bag.std(axis=0))
        rows.append(np.concatenate(parts))
    return np.asarray(rows)


def _mil_scorecard(bags: Sequence[np.ndarray], labels: Sequence[int], *,
                   responders: Optional[Sequence[np.ndarray]] = None,
                   folds: int = 4, seed: int = 0,
                   **fit: Any) -> Dict[str, float]:
    """Cross-validated well AUROC for attention MIL against mean baselines.

    Wells are split into stratified folds; each fold is predicted by a
    model trained on the others. The baselines are an L2 logistic
    regression on each well's mean cell, and on its mean and SD. When the
    truly responding cells are known (a planted or annotated set), the
    held-out attention is also scored as a ranking of responders above the
    other cells of the positive wells.

    :param bags: one ``(cells, features)`` array per well.
    :param labels: 0/1 per well.
    :param responders: optional boolean array per well marking responders.
    :param folds: cross-validation folds, capped by the smaller class.
    :param seed: fold split and model seed.
    :param fit: passed to :func:`_mil_fit`.
    :returns: ``mil_auroc``, ``mean_auroc``, ``mean_sd_auroc``, ``wells``
        and, with ``responders``, ``attention_auroc`` and
        ``evidence_auroc``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray(labels, dtype=np.int64)
    folds = max(2, min(int(folds), int(labels.sum()),
                       int(len(labels) - labels.sum())))
    split = StratifiedKFold(folds, shuffle=True, random_state=int(seed))
    mil = np.zeros(len(labels))
    base = {False: np.zeros(len(labels)), True: np.zeros(len(labels))}
    attention: List[Optional[np.ndarray]] = [None] * len(labels)
    evidence: List[Optional[np.ndarray]] = [None] * len(labels)
    for train, test in split.split(np.zeros(len(labels)), labels):
        fitted = _mil_fit([bags[i] for i in train], labels[train],
                          seed=seed, **fit)
        probs, weights, scores = _mil_predict(fitted,
                                              [bags[i] for i in test])
        mil[test] = probs
        for i, w, e in zip(test, weights, scores):
            attention[i] = w
            evidence[i] = e
        for spread in (False, True):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000))
            clf.fit(_mil_summaries([bags[i] for i in train], spread),
                    labels[train])
            base[spread][test] = clf.predict_proba(
                _mil_summaries([bags[i] for i in test], spread))[:, 1]
    card = {"mil_auroc": float(roc_auc_score(labels, mil)),
            "mean_auroc": float(roc_auc_score(labels, base[False])),
            "mean_sd_auroc": float(roc_auc_score(labels, base[True])),
            "wells": float(len(labels))}
    if responders is not None:
        truth = np.concatenate([np.asarray(responders[i], dtype=bool)
                                for i in np.flatnonzero(labels)])
        positive = np.flatnonzero(labels)
        if 0 < truth.sum() < len(truth):
            for name, values in (("attention", attention),
                                 ("evidence", evidence)):
                score = np.concatenate([values[i] for i in positive])
                card[f"{name}_auroc"] = float(roc_auc_score(truth, score))
    return card


def _synthetic_mil_bags(wells: int = 24, cells: int = 60,
                        features: int = 16, fraction: float = 0.15,
                        shift: float = 3.0, well_noise: float = 0.5,
                        seed: int = 0):
    """Wells of Gaussian cells with responders planted in the positive half.

    Every cell is standard normal plus a per-well offset of SD
    ``well_noise``; in positive wells a ``fraction`` of cells is moved by
    ``shift`` along one fixed direction. The well mean moves only by
    ``fraction * shift``, so a mean-feature classifier has to find a small
    shift under well-to-well noise while the responders stand out.

    :returns: ``(bags, labels, responders)``.
    """
    rng = np.random.default_rng(int(seed))
    direction = rng.normal(size=features)
    direction /= np.linalg.norm(direction)
    bags, labels, responders = [], [], []
    for well in range(int(wells)):
        positive = well % 2
        bag = (rng.normal(size=(cells, features))
               + rng.normal(scale=well_noise, size=features))
        hit = np.zeros(cells, dtype=bool)
        if positive:
            hit[rng.choice(cells, max(1, int(round(fraction * cells))),
                           replace=False)] = True
            bag[hit] += shift * direction
        bags.append(bag.astype(np.float32))
        labels.append(positive)
        responders.append(hit)
    return bags, np.asarray(labels, dtype=np.int64), responders


def _mil_from_table(frame: Any, *, folds: int = 4, seed: int = 0,
                    well_column: str = _MIL_WELL_COLUMN,
                    label_column: str = _MIL_LABEL_COLUMN,
                    feature_columns: Optional[Sequence[str]] = None,
                    positive: Any = None, **fit: Any):
    """Learn which cells carry a well-level label, from a per-cell table.

    A model trained on every well gives each cell its attention and each
    well its probability; the scorecard comes from cross-validation, so the
    AUROCs are for wells the model did not see.

    :param frame: one row per cell, as :func:`_mil_bags` reads it.
    :returns: ``(cells, wells, scorecard)``: the input rows that were used,
        with ``mil_attention`` and ``mil_evidence`` added (see
        :func:`_mil_predict`); one row per well with its label and
        ``mil_probability``; and :func:`_mil_scorecard`'s numbers.
    """
    import pandas as pd

    well_column = _mil_well_column(frame, well_column)
    bags, labels, wells, rows, kept = _mil_bags(
        frame, well_column=well_column, label_column=label_column,
        feature_columns=feature_columns, positive=positive)
    card = _mil_scorecard(bags, labels, folds=folds, seed=seed, **fit)
    fitted = _mil_fit(bags, labels, seed=seed, **fit)
    probs, attention, evidence = _mil_predict(fitted, bags)
    index = np.concatenate(rows)
    cells = kept.iloc[index].reset_index(drop=True)
    cells["mil_attention"] = np.concatenate(attention)
    cells["mil_evidence"] = np.concatenate(evidence)
    well_frame = pd.DataFrame({well_column: wells, label_column: labels,
                               "mil_probability": probs,
                               "cells": [len(b) for b in bags]})
    return cells, well_frame, card


#: Key prefix for an encoder's model-zoo entry. Distinct from a checkpoint's
#: filename-derived key because an encoder has no file of spaCR's own -- it is
#: named by backbone and policy, which together are what a later run must
#: match for its numbers to be comparable.
ENCODER_KEY_PREFIX = "encoder:"


def encoder_key(spec: "EmbeddingSpec") -> str:
    """The stable id for one encoder configuration.

    BACKBONE AND POLICY TOGETHER, because they are jointly what makes two
    runs' dimensions mean the same thing. The same backbone under
    :data:`CHANNEL_PER_CHANNEL` and :data:`CHANNEL_PROJECT` produces columns
    that are the same width, the same dtype and not remotely the same
    quantity -- one is per-stain, the other is a mixture. A key that named
    only the backbone would let those two be compared silently.

    :param spec: the run's configuration. Only ``backbone`` and
        ``channel_policy`` reach the key: everything else on the spec --
        batch size, device, crop size -- changes how long the run takes and
        not what a dimension means.
    """
    return f"{ENCODER_KEY_PREFIX}{spec.backbone}/{spec.channel_policy}"


def _weights_on_disk(backbone: str) -> Tuple[str, str, int]:
    """Find the cached weights ``timm`` resolved, and hash them.

    :returns: ``(path, sha256, size_bytes)``; the path is ``''`` and the
        digest ``''`` when the weights are not on this machine.

    HASHES WHAT IS ACTUALLY THERE rather than trusting a published digest,
    which is the same rule :class:`spacr.model_zoo.ModelEntry` states for a
    downloaded model: "for a downloaded model this is the digest of the bytes
    that were actually written". A pretrained encoder arrives through the
    HuggingFace cache, so the bytes on this machine are the only thing that
    can be checked here.
    """
    import hashlib

    try:
        import timm
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return "", "", 0

    try:
        config = timm.get_pretrained_cfg(backbone)
        repo = getattr(config, "hf_hub_id", None)
        filename = getattr(config, "hf_hub_filename", None) or "model.safetensors"
        if not repo:
            return "", "", 0
        path = try_to_load_from_cache(repo, filename)
    except Exception:
        return "", "", 0

    if not path or not isinstance(path, str) or not os.path.exists(path):
        return "", "", 0

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return path, digest.hexdigest(), os.path.getsize(path)


def encoder_entry(spec: Optional["EmbeddingSpec"] = None, *,
                  scorecard: Optional[Mapping[str, Any]] = None):
    """This encoder as a :class:`spacr.model_zoo.ModelEntry`.

    AN ENCODER IS A PUBLISHED MODEL like any other, so it belongs in the zoo
    with its checksum and its scorecard. An embedding that ships without one
    is a black box twice over: opaque in what it encodes, and unmeasured in
    how well it does it.

    WHAT AN ENCODER'S PROVENANCE ACTUALLY IS. It has no spaCR checkpoint --
    the weights are ImageNet or a public self-supervised run, resolved by
    ``timm`` and cached by HuggingFace. So the entry records the backbone, the
    channel policy, and the digest of the weights AS THEY SIT ON THIS MACHINE.
    That is the thing a later run has to match, and it is checkable here
    without a network call.

    WHEN THE WEIGHTS ARE NOT CACHED the entry still exists and says so in its
    notes, with an empty digest -- which
    :func:`spacr.model_zoo.fetch` already treats as a refusal rather than a
    pass. An entry that quietly claimed a checksum it had not computed would
    be worse than one that admits it cannot yet.

    :param spec: the configuration to describe; the default spec when omitted.
    :param scorecard: retrieval numbers from 386's "HOW TO KNOW IT WORKED" --
        kNN accuracy on gene identity, embeddings versus the measured panel.
        Attached as :attr:`ModelEntry.metrics`, which is where 370's
        ``scorecard_lines`` reads them from.
    :returns: a ``ModelEntry`` of kind ``'encoder'``.
    """
    from .model_zoo import UNKNOWN, ModelEntry

    spec = spec if spec is not None else EmbeddingSpec()
    path, digest, size = _weights_on_disk(spec.backbone)

    notes = [
        f"Channel policy: {spec.channel_policy}. Dimensions from one policy "
        f"are not comparable with the other's.",
    ]
    if not digest:
        notes.append(
            "No checksum: the pretrained weights are not in this machine's "
            "HuggingFace cache, so there are no bytes to hash yet. Run an "
            "embedding once and re-read this entry.")
    if scorecard is None:
        notes.append(
            "No scorecard. 386: an embedding that ships without one is a "
            "black box twice over. Measure retrieval -- kNN accuracy on gene "
            "identity against a known-phenotype control -- and attach it.")

    return ModelEntry(
        key=encoder_key(spec),
        name=f"{spec.backbone} ({spec.channel_policy})",
        kind="encoder",
        source="local" if path else "remote",
        path=path,
        uri=f"timm:{spec.backbone}",
        version="1",
        sha256=digest,
        size_bytes=size,
        trained_on=UNKNOWN,
        trained_by=f"timm / {spec.backbone} pretrained weights",
        metrics=dict(scorecard or {}),
        notes=tuple(notes),
        verified=False,
    )


def _scored_encoder_entry(spec: Optional["EmbeddingSpec"], features: Any,
                          labels: Mapping[Any, Any], k: int = 10):
    """The encoder's zoo entry with a retrieval scorecard measured here.

    The scorecard is :func:`_retrieval_scorecard` on ``features`` against
    ``labels``. When the labels cannot be scored (fewer than two labelled
    crops or a single class) the entry is returned without one, and its
    notes say so, rather than failing the run.

    :param spec: the configuration to describe; the default spec when omitted.
    :param features: numeric frame indexed by crop key.
    :param labels: crop key to class.
    :param k: neighbours per query.
    :returns: a ``ModelEntry`` of kind ``'encoder'``.
    """
    try:
        scorecard = _retrieval_scorecard(features, labels, k=k)
    except ValueError:
        scorecard = None
    return encoder_entry(spec, scorecard=scorecard)
