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
instruction 386 asks for the choice to be recorded rather than inherited from
whatever the first backbone wanted:

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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "EmbeddingError",
    "CHANNEL_PER_CHANNEL", "CHANNEL_PROJECT", "CHANNEL_POLICIES",
    "EMBEDDING_PREFIX", "DEFAULT_BACKBONE", "DEFAULT_POOL",
    "EmbeddingSpec", "EmbeddingResult",
    "embedding_column_names", "embed_array", "channel_of_column",
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
#: hardcoded so a scorecard (instruction 370) can pin which one produced a
#: given matrix.
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
    :param normalize: divide each channel by its own 99th percentile before
        encoding. Microscopy dynamic range varies by orders of magnitude
        between stains, and an encoder trained on photographs will otherwise
        see one channel as noise.
    """

    backbone: str = DEFAULT_BACKBONE
    channel_policy: str = CHANNEL_PER_CHANNEL
    channels: Optional[Tuple[int, ...]] = None
    batch_size: int = 64
    device: Optional[str] = None
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.channel_policy not in CHANNEL_POLICIES:
            raise EmbeddingError(
                f"unknown channel policy {self.channel_policy!r}; "
                f"expected one of {', '.join(CHANNEL_POLICIES)}")
        if self.batch_size < 1:
            raise EmbeddingError("batch_size must be at least 1")

    def fingerprint(self) -> str:
        """A short digest of everything that changes the numbers.

        Two matrices with the same fingerprint were produced the same way.
        A scorecard can quote it, and a cached matrix can be invalidated by
        it rather than by a timestamp.
        """
        payload = "|".join((
            self.backbone, self.channel_policy,
            "" if self.channels is None else ",".join(map(str, self.channels)),
            str(self.normalize)))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
    available = array.shape[3]
    channels = (tuple(range(available)) if spec.channels is None
                else tuple(spec.channels))
    for channel in channels:
        if not 0 <= channel < available:
            raise EmbeddingError(
                f"channel {channel} is not in the crops, which have "
                f"{available}")
    if spec.channel_policy == CHANNEL_PROJECT and len(channels) > 3:
        raise EmbeddingError(
            f"{len(channels)} channels cannot be projected onto three "
            "without choosing which to drop; name three in "
            "EmbeddingSpec.channels, or use the per-channel policy")
    return array.astype(np.float32, copy=False), channels


def _scaled(plane: np.ndarray, normalize: bool) -> np.ndarray:
    """One channel, scaled into roughly [0, 1] by its own 99th percentile."""
    if not normalize:
        return plane
    top = float(np.percentile(plane, 99)) if plane.size else 0.0
    if not np.isfinite(top) or top <= 0:
        return np.zeros_like(plane)
    return np.clip(plane / top, 0.0, 1.0)


def embed_array(crops: np.ndarray, spec: Optional[EmbeddingSpec] = None, *,
                encoder: Optional[Callable[[np.ndarray], np.ndarray]] = None
                ) -> EmbeddingResult:
    """Embed a stack of crops.

    :param crops: ``(n, height, width, channels)``, the shape
        :mod:`spacr.crops` already produces.
    :param spec: how to embed; the default is per-channel resnet18.
    :param encoder: a callable taking ``(n, height, width, 3)`` float32 in
        [0, 1] and returning ``(n, dims)``. Injected by the tests so the
        wiring can be exercised without downloading a backbone; production
        callers leave it ``None``.
    :returns: an :class:`EmbeddingResult`.
    :raises EmbeddingError: on a shape or channel problem, before any model
        is loaded -- a download is a slow way to find out the array was
        wrong.
    """
    spec = spec or EmbeddingSpec()
    array, channels = _prepare(crops, spec)
    run = encoder if encoder is not None else _timm_encoder(spec)

    if spec.channel_policy == CHANNEL_PROJECT:
        planes = [_scaled(array[..., c], spec.normalize) for c in channels]
        while len(planes) < 3:
            planes.append(np.zeros_like(planes[0]))
        stack = np.stack(planes[:3], axis=-1)
        values = np.asarray(run(stack), dtype=np.float32)
        per_channel = values.shape[1]
    else:
        blocks = []
        for channel in channels:
            plane = _scaled(array[..., channel], spec.normalize)
            stack = np.repeat(plane[..., None], 3, axis=-1)
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
            "`spacr[torch]` extra") from exc

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
