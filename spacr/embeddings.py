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
    adaptive = (spec.backbone in _FOUNDATION_MODELS
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
    """The encoder ``spec.backbone`` names: a foundation model or a timm one."""
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
