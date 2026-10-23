"""Compute label-free image embeddings per object, beside the measured features.

spaCR's other two routes from image to numbers both depend on prior
choices. The measured panel in :mod:`spacr.measure` computes a fixed,
hand-designed set -- intensity, shape, texture, distances -- so it captures
only named properties. The supervised path in :mod:`spacr.deep_spacr`
fine-tunes a backbone against annotations, so it captures only annotated
classes. A phenotype with no corresponding column or annotation (for
example a subtle vacuole mislocalisation) is not detected by either route.

Channel handling is an explicit choice. Pretrained encoders expect three
channels; spaCR images often have four or five, with no RGB meaning (a DAPI
channel is not "red"). Two policies are available, and the choice is recorded
in the spec rather than inherited from a backbone:

  Per-channel, then concatenate (:data:`CHANNEL_PER_CHANNEL`, the default).
  Each channel is encoded on its own and the vectors are joined. Channel
  identity is preserved: dimension 517 belongs to channel 2 only, so
  :mod:`spacr.attribution_columns` can report which stain a hit came from and
  a reader can drop a channel without re-embedding. Costs N forward passes.

  Project to three (:data:`CHANNEL_PROJECT`). One pass over a fixed
  channel-to-RGB mapping. Cheaper, but irreversible: once channels are mixed,
  no downstream column can be attributed to a stain, so a screen cannot tell
  whether a difference came from the parasite or the host channel.

Per-channel is the default because it keeps stain attribution: an embedding
that detects a phenotype without identifying the stain that carries it is of
limited use. Projection remains available when the number of forward passes
is the limiting cost.

Scope: this module does not crop -- it takes the crops :mod:`spacr.crops`
produces -- and does not write to the database. It returns a float32 matrix
keyed by object id, the same shape that UMAP, the regression family, hit
calling and the AnnData export already consume.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import stat
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
    """Embedding configuration, recorded with each matrix as its provenance.

    For ``subcell_rybg``, choose :data:`CHANNEL_PROJECT`, supply four
    explicit, distinct ``channels`` in microtubules (R), ER (Y), DNA (B),
    protein (G) order, and set ``normalize=False``. Crops must be at least
    16 pixels in each spatial dimension. Native dimensions are preserved;
    the encoder applies the authors' whole-crop min-max normalization.

    :param backbone: Name of a timm or registered foundation encoder.
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
    :param cell_dino_factory: pinned official Cell-DINO hub factory.
    :param checkpoint_path: local official Cell-DINO weights file.
    :param checkpoint_sha256: expected SHA-256 of those weights.
    """

    backbone: str = DEFAULT_BACKBONE
    channel_policy: str = CHANNEL_PER_CHANNEL
    channels: Optional[Tuple[int, ...]] = None
    batch_size: int = 64
    device: Optional[str] = None
    normalize: bool = True
    channel_scale: Optional[Tuple[float, ...]] = None
    cell_dino_factory: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None

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
        """Return a 16-character digest of the fields that affect the values.

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
        if self.backbone == "cell_dino":
            fields.extend((self.cell_dino_factory or "",
                           os.path.realpath(os.path.expanduser(
                               self.checkpoint_path or "")),
                           (self.checkpoint_sha256 or "").lower(),
                           _CELL_DINO_REVISION))
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

    Under the per-channel policy the channel is part of the name --
    ``emb_c2_017`` -- so attribution needs no separate map beside the
    matrix. Under projection the names use ``rgb`` instead of a channel
    index, because no column belongs to one channel.

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
    """Return the channel index a column came from, or ``None``.

    A projected column returns ``None`` because the channels were mixed and
    no dimension belongs to one stain. Names without the embedding prefix
    also return ``None``.

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
    if spec.backbone == "subcell_rybg":
        _subcell_rybg_spatial_size(array.shape[1], array.shape[2])
    channels = _encoded_channels(array.shape[3], spec)
    return array.astype(np.float32, copy=False), channels


def _encoded_channels(available: int, spec: EmbeddingSpec) -> Tuple[int, ...]:
    """The channel indices ``spec`` encodes from crops with ``available``."""
    if spec.backbone == "cell_dino":
        _cell_dino_spec(spec)
    channels = (tuple(range(available)) if spec.channels is None
                else tuple(spec.channels))
    if spec.backbone == "subcell_rybg":
        _subcell_rybg_channels(spec)
    for channel in channels:
        if not 0 <= channel < available:
            raise EmbeddingError(
                f"channel {channel} is not in the crops, which have "
                f"{available}")
    multichannel = (
        spec.backbone in ("subcell_rybg", "cell_dino")
        or spec.backbone.startswith(_DINO_PREFIX)
        or (spec.backbone in _FOUNDATION_MODELS
            and _FOUNDATION_MODELS[spec.backbone]["in_channels"] is None))
    if (spec.channel_policy == CHANNEL_PROJECT and len(channels) > 3
            and not multichannel):
        raise EmbeddingError(
            f"{len(channels)} channels cannot be projected onto three "
            "without choosing which to drop; name three in "
            "EmbeddingSpec.channels, or use the per-channel policy")
    if (spec.channel_scale is not None
            and len(spec.channel_scale) != len(channels)):
        raise EmbeddingError(_scale_length_message(len(spec.channel_scale),
                                                   len(channels)))
    return channels


def _subcell_rybg_channels(spec: EmbeddingSpec) -> None:
    """Require four unambiguous R/Y/B/G positions before loading SubCell."""
    channels = spec.channels
    if (spec.channel_policy != CHANNEL_PROJECT or channels is None
            or len(channels) != 4
            or any(isinstance(channel, (bool, np.bool_))
                   or not isinstance(channel, (int, np.integer))
                   or channel < 0 for channel in channels)
            or len(set(channels)) != 4):
        raise EmbeddingError(
            "subcell_rybg needs the projection policy and four explicit, "
            "distinct channel indices in r, y, b, g order")


def _subcell_rybg_spatial_size(height: int, width: int) -> None:
    """Require one full ViT-B/16 patch in each native crop dimension."""
    if min(height, width) < 16:
        raise EmbeddingError(
            "subcell_rybg needs crops at least 16 pixels high "
            "and wide for its ViT-B/16 patch embedding")


_CELL_DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
_CELL_DINO_FACTORIES = {
    "cell_dino_hpa_vitl16": (4, 224),
    "cell_dino_hpa_vitl14": (4, 518),
    "cell_dino_cp_vits8": (5, 128),
    "channel_adaptive_dino_vitl16": (0, 224),
}
_CHANNEL_ADAPTIVE = "channel_adaptive_dino_vitl16"


def _cell_dino_spec(spec: EmbeddingSpec) -> tuple[int, int]:
    """Require an official factory, local digest and ordered source planes.

    Channel-Adaptive DINO takes any number of distinct planes; the other
    factories take exactly their declared width.
    """
    if not spec.checkpoint_path:
        raise EmbeddingError(
            "Choose a local official Cell-DINO checkpoint before embedding; "
            "Meta provides weights through its download request")
    if not isinstance(spec.checkpoint_sha256, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", spec.checkpoint_sha256):
        raise EmbeddingError("Enter the Cell-DINO checkpoint's SHA-256 digest")
    factory = spec.cell_dino_factory
    if not isinstance(factory, str) or factory not in _CELL_DINO_FACTORIES:
        raise EmbeddingError("Choose an official Cell-DINO factory before embedding")
    width, size = _CELL_DINO_FACTORIES[factory]
    channels = spec.channels
    if not width and isinstance(channels, (tuple, list)) and channels:
        width = len(channels)
    if (spec.channel_policy != CHANNEL_PROJECT
            or not isinstance(channels, (tuple, list))
            or len(channels) != width
            or any(isinstance(channel, (bool, np.bool_))
                   or not isinstance(channel, (int, np.integer))
                   or channel < 0 for channel in channels)
            or len(set(channels)) != width):
        raise EmbeddingError(
            f"{factory} needs {width} distinct, explicit crop channel "
            "indices in the declared model-input order")
    return width, size


_OPEN_DINOV2 = "vit_small_patch14_dinov2.lvd142m"
_CELL_DINO_ENV = "SPACR_CELL_DINO_CHECKPOINT"
_CELL_DINO_PATCHES = {(4, 16): "cell_dino_hpa_vitl16",
                      (4, 14): "cell_dino_hpa_vitl14",
                      (5, 8): "cell_dino_cp_vits8",
                      (1, 16): _CHANNEL_ADAPTIVE}
_CELL_DINO_SEEN: Dict[Tuple[str, int, int], Dict[str, str]] = {}


def _cell_dino_folder() -> str:
    """The folder searched for official Cell-DINO checkpoints.

    Every ``*.pth`` or ``*.pt`` file in ``cell_dino`` under spaCR's model
    cache is offered; ``$SPACR_CELL_DINO_CHECKPOINT`` naming a file overrides
    the choice.
    """
    from .model_zoo import _model_cache_root

    return str(_model_cache_root() / "cell_dino")


def _cell_dino_entry(chosen: str, verify: bool = False) -> Dict[str, str]:
    """Describe one checkpoint file without loading its weights.

    The factory is read from the patch-embedding shape of the plain model
    state, which is memory-mapped rather than loaded.

    :param chosen: the file.
    :param verify: also compute its SHA-256; cached per path, size and
        modification time.
    :returns: ``path`` with either ``factory`` (and ``sha256`` when
        verified) or ``problem``.
    """
    path = os.path.realpath(os.path.expanduser(chosen))
    try:
        status = os.stat(path)
    except OSError:
        return {"path": path, "problem": "missing"}
    if not stat.S_ISREG(status.st_mode):
        return {"path": path, "problem": "unreadable"}
    key = (path, status.st_size, status.st_mtime_ns)
    found = _CELL_DINO_SEEN.get(key)
    if found is None:
        found = {"path": path}
        try:
            import torch

            state = torch.load(path, map_location="cpu", weights_only=True,
                               mmap=True)
            shape = tuple(state["patch_embed.proj.weight"].shape)
            factory = _CELL_DINO_PATCHES.get((shape[1], shape[-1]))
        except Exception:
            factory = None
        if factory:
            found["factory"] = factory
        else:
            found["problem"] = "unreadable"
        _CELL_DINO_SEEN[key] = found
    if verify and "factory" in found and "sha256" not in found:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        found["sha256"] = digest.hexdigest()
    return dict(found)


def _cell_dino_checkpoints() -> List[Dict[str, str]]:
    """Every installed Cell-DINO checkpoint, in file-name order.

    ``$SPACR_CELL_DINO_CHECKPOINT`` is listed first when it names a file
    outside the folder.
    """
    folder = _cell_dino_folder()
    try:
        names = sorted(name for name in os.listdir(folder)
                       if name.lower().endswith((".pth", ".pt")))
    except OSError:
        names = []
    paths = [os.path.join(folder, name) for name in names]
    chosen = os.environ.get(_CELL_DINO_ENV, "").strip()
    if chosen and os.path.realpath(os.path.expanduser(chosen)) not in {
            os.path.realpath(path) for path in paths}:
        paths.insert(0, chosen)
    return [_cell_dino_entry(path) for path in paths]


def _find_cell_dino(verify: bool = False, available: int = 0,
                    preferred: str = "") -> Dict[str, str]:
    """Pick the installed Cell-DINO checkpoint to use, without a network.

    ``$SPACR_CELL_DINO_CHECKPOINT`` wins, then ``preferred`` when it is still
    installed and readable. Otherwise the best fit for ``available`` crop
    channels: five takes the Cell Painting ViT-S/8; four takes an HPA model,
    a single-cell (``hpa_sc``) one first; any other count takes
    Channel-Adaptive DINO. Failing those, the first readable file.

    :param verify: also compute the chosen file's SHA-256.
    :param available: how many channels the crops have; ``0`` if unknown.
    :param preferred: the file the user chose last.
    :returns: ``{}`` when no file is installed; else ``path`` with either
        ``factory`` (and ``sha256`` when verified) or ``problem``.
    """
    chosen = os.environ.get(_CELL_DINO_ENV, "").strip()
    if chosen:
        return _cell_dino_entry(chosen, verify)
    entries = _cell_dino_checkpoints()
    if not entries:
        return {}
    readable = [entry for entry in entries if entry.get("factory")]
    wanted = os.path.realpath(os.path.expanduser(preferred)) if preferred else ""
    pick = next((entry for entry in readable if entry["path"] == wanted), None)
    if pick is None:
        if available == 5:
            order = ("cell_dino_cp_vits8",)
        elif available == 4:
            order = ("cell_dino_hpa_vitl16", "cell_dino_hpa_vitl14")
        elif available > 0:
            order = (_CHANNEL_ADAPTIVE,)
        else:
            order = ()
        fits = [entry for factory in order for entry in sorted(
            (entry for entry in readable if entry["factory"] == factory),
            key=lambda entry: "hpa_sc" not in os.path.basename(entry["path"]))]
        pick = (fits or readable or entries)[0]
    return _cell_dino_entry(pick["path"], verify)


def _dino_choice(channels: Optional[Sequence[int]], available: int, *,
                 channel_policy: str = CHANNEL_PROJECT, batch_size: int = 64,
                 device: Optional[str] = None, verify: bool = True,
                 preferred: str = "") -> Tuple[EmbeddingSpec, Dict[str, str]]:
    """Official Cell-DINO when it is installed and mapped, else open DINOv2.

    Channel-Adaptive DINO needs no mapping: without one it encodes every
    crop channel separately, in order.

    :param channels: crop channel per Cell-DINO input plane, in model order.
    :param available: how many channels the crops have.
    :param preferred: the checkpoint file the user chose last.
    :returns: the spec to run and what was decided: ``model`` is
        ``cell_dino`` or ``dinov2``, with ``reason`` (``absent``,
        ``unmapped`` or ``unreadable``) when Cell-DINO is not used.
    """
    found = _find_cell_dino(verify=verify, available=available,
                            preferred=preferred)
    note = dict(found)
    factory = found.get("factory")
    if factory:
        width = _CELL_DINO_FACTORIES[factory][0]
        planes = tuple(channels or ())
        if not width:
            if not planes or any(not 0 <= plane < available for plane in planes):
                planes = tuple(range(available))
            width = len(planes) or 1
        if (len(planes) == width and len(set(planes)) == width and all(
                isinstance(plane, (int, np.integer))
                and not isinstance(plane, (bool, np.bool_))
                and 0 <= plane < available for plane in planes)):
            spec = EmbeddingSpec(
                backbone="cell_dino", channel_policy=CHANNEL_PROJECT,
                channels=tuple(int(plane) for plane in planes),
                batch_size=batch_size, device=device,
                cell_dino_factory=factory, checkpoint_path=found["path"],
                checkpoint_sha256=found.get("sha256"))
            note.update(model="cell_dino", planes=str(width))
            return spec, note
        note.update(reason="unmapped", planes=str(width))
    else:
        note["reason"] = "unreadable" if found.get("problem") else "absent"
    note.update(model="dinov2", backbone=_OPEN_DINOV2,
                folder=_cell_dino_folder())
    return EmbeddingSpec(backbone=_OPEN_DINOV2, channel_policy=channel_policy,
                         batch_size=batch_size, device=device), note


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
        if spec.backbone == "subcell_rybg":
            _subcell_rybg_spatial_size(int(shape[1]), int(shape[2]))
        if spec.backbone == "cell_dino":
            _cell_dino_spec(spec)
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

    When normalization is enabled, every crop is divided by the same
    per-channel scale, ``spec.channel_scale``, so a crop's embedding does not
    depend on the other crops in ``crops``. If normalization is enabled and
    the spec carries no scale, one is estimated from a random sample of
    ``crops`` -- the stack is treated as the whole plate -- and a warning is
    logged. To embed one plate in several calls, pass the same scale to each:
    the returned ``spec`` carries it.

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
        is loaded, so an invalid array does not trigger a model download.
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
    if spec.backbone == "subcell_rybg" and width != 4:
        raise EmbeddingError(
            "subcell_rybg encoder must accept exactly four planes in "
            "r, y, b, g order")
    if spec.backbone == "cell_dino" and width != _cell_dino_spec(spec)[0]:
        raise EmbeddingError(
            "Cell-DINO encoder input width differs from the declared factory")

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
    """Load a headless backbone and honor a strict patch embedding size.

    :param spec: Backbone, device and batch-size choices for the encoder.
    :returns: A callable that encodes float32 image batches.
    """
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
    patch_embed = getattr(model, "patch_embed", None)
    strict_size = (getattr(patch_embed, "img_size", None)
                   if getattr(patch_embed, "strict_img_size", False) else None)

    def run(stack: np.ndarray) -> np.ndarray:
        """Encode one three-channel stack, in batches, under no_grad.

        Batched here rather than by the caller because the batch size belongs
        to the device, not to the channel policy: the per-channel policy calls
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
                if (strict_size is not None
                        and tuple(tensor.shape[-2:]) != tuple(strict_size)):
                    tensor = torch.nn.functional.interpolate(
                        tensor, size=strict_size, mode="bilinear",
                        align_corners=False)
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
    "subcell_rybg": {
        "label": "SubCell (CZI / Lundberg lab ViT-B/16, R/Y/B/G)",
        "url": ("https://czi-subcell-public.s3.amazonaws.com/models/"
                "all_channels_ViT-ProtS-Pool.pth"),
        "in_channels": 4,
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
    are channel-adaptive and take any number of channels. ``subcell`` takes
    DNA and the stain of interest; ``subcell_rybg`` takes four explicitly
    mapped microtubule, ER, DNA and protein planes. Cell-DINO requires a
    declared local official checkpoint and explicit channel mapping.
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

    The returned callable takes ``(n, height, width, k)`` float32 crops
    and returns ``(n, dims)``; its ``in_channels`` attribute tells
    :func:`embed_array` how many planes ``k`` it wants, ``None`` meaning any.
    Other foundation crops are resized to the model's training size. SubCell
    R/Y/B/G keeps its native crop geometry and interpolates position tokens.
    Other weights are downloaded once into the Hugging Face or torch hub
    cache. Cell-DINO uses a declared local checkpoint and the pinned official
    DINOv2 source; no gated weights are fetched.

    :raises EmbeddingError: when dependencies, mapping or verified local
        Cell-DINO weights are unavailable.
    """
    if spec.backbone == "subcell_rybg":
        _subcell_rybg_channels(spec)
    if spec.backbone == "cell_dino":
        _cell_dino_spec(spec)
    try:
        import torch
    except ImportError as exc:
        raise EmbeddingError(
            "foundation-model embeddings need torch; install the "
            "`spacr[embeddings]` extra") from exc
    if spec.backbone == "cell_dino":
        width, size = _cell_dino_spec(spec)
        info = dict(_FOUNDATION_MODELS["cell_dino"],
                    in_channels=width, size=size)
        model, forward = _cell_dino_model(spec, torch)
        device = spec.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)

        def run(stack: np.ndarray) -> np.ndarray:
            """Encode declared Cell-DINO planes at the official crop size.

            Each plane of each crop is centred and scaled to unit standard
            deviation after resizing, as the official evaluation transform
            does.
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
                    chunk = chunk - chunk.mean((-2, -1), keepdim=True)
                    chunk = chunk / (chunk.std((-2, -1), unbiased=False,
                                               keepdim=True) + 1e-7)
                    out.append(forward(model, chunk).detach().float().cpu().numpy())
            return np.concatenate(out, axis=0)

        run.in_channels = info["in_channels"]
        return run
    info = _FOUNDATION_MODELS[spec.backbone]
    device = spec.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if spec.backbone in ("subcell", "subcell_rybg"):
        model, forward = _subcell_model(info, torch)
    else:
        model, forward = _hub_model(spec.backbone, info, torch)
    model.eval().to(device)
    size = int(info["size"])

    def run(stack: np.ndarray) -> np.ndarray:
        """Encode ``(n, h, w, k)`` crops in model-appropriate batches.

        :param stack: float32 crops, channels last. SubCell R/Y/B/G uses
            native intensity values; other foundations expect [0, 1].
        :returns: ``(n, dims)`` float32 features.
        """
        out: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, stack.shape[0], spec.batch_size):
                chunk = torch.from_numpy(np.ascontiguousarray(
                    stack[start:start + spec.batch_size].transpose(0, 3, 1, 2)
                )).float().to(device)
                if spec.backbone == "subcell_rybg":
                    _subcell_rybg_spatial_size(*chunk.shape[-2:])
                if (spec.backbone != "subcell_rybg"
                        and chunk.shape[-2:] != (size, size)):
                    chunk = torch.nn.functional.interpolate(
                        chunk, size=(size, size), mode="bilinear",
                        align_corners=False)
                out.append(forward(model, chunk).detach().float().cpu().numpy())
        return np.concatenate(out, axis=0)

    run.in_channels = info["in_channels"]
    return run


def _cell_dino_model(spec: EmbeddingSpec, torch: Any):
    """Load only a digest-matched local state into the pinned official hub."""
    width, _ = _cell_dino_spec(spec)
    path = os.path.realpath(os.path.expanduser(spec.checkpoint_path or ""))
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            os.close(descriptor)
            raise EmbeddingError("Cell-DINO checkpoint must be a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != spec.checkpoint_sha256.lower():
                raise EmbeddingError("Cell-DINO checkpoint SHA-256 does not match")
            handle.seek(0)
            state = torch.load(handle, map_location="cpu", weights_only=True)
            after = os.fstat(handle.fileno())
            if (before.st_size, before.st_mtime_ns) != (
                    after.st_size, after.st_mtime_ns):
                raise EmbeddingError("Cell-DINO checkpoint changed during loading")
    except OSError as exc:
        if os.path.isdir(path):
            raise EmbeddingError("Cell-DINO checkpoint must be a regular file") from exc
        raise EmbeddingError(f"Cannot read Cell-DINO checkpoint: {exc}") from exc
    if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()):
        raise EmbeddingError("Cell-DINO checkpoint is not a plain model state")
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2:" + _CELL_DINO_REVISION,
            spec.cell_dino_factory, pretrained=False, trust_repo=True)
    except Exception as exc:
        raise EmbeddingError(
            "Cell-DINO needs the pinned official dinov2 source at "
            f"{_CELL_DINO_REVISION}: {exc}") from exc
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise EmbeddingError(
            f"Cell-DINO official {spec.cell_dino_factory} state cannot be "
            f"loaded strictly: {exc}") from exc
    patch_embed = getattr(model, "patch_embed", None)
    projection = getattr(patch_embed, "proj", None)
    adaptive = spec.cell_dino_factory == _CHANNEL_ADAPTIVE
    if getattr(projection, "in_channels", None) != (1 if adaptive else width):
        raise EmbeddingError("Cell-DINO factory has an unexpected input width")

    def forward(net, x):
        """Return the official DINOv2 head output for each declared crop.

        Channel-Adaptive DINO encodes each plane on its own and returns the
        class tokens of every plane side by side, as its official
        bag-of-channels evaluation does.
        """
        if adaptive:
            return net.get_intermediate_layers(
                x, 1, return_class_token=True)[-1][1]
        return net(x)

    return model, forward


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
    """SubCell's fixed-channel ViT encoder with gated attention pooling.

    The published checkpoint holds a Hugging Face ViT under ``encoder.`` and
    a two-head gated attention pooler under ``pool_model.``; both are loaded
    strictly, so a changed checkpoint is refused rather than half-loaded.
    The four-plane R/Y/B/G model scales each whole crop as its own loader
    does; the existing two-plane model retains its per-channel scaling.
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
                       num_channels=int(info["in_channels"]), qkv_bias=True)

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
            """``(n, k, h, w)`` to ``(n, 1536)``, with the two heads joined.

            The checkpoint determines whether ``k`` is two or four channels.
            """
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
        """Min-max scale the expected model planes, then encode."""
        if info["in_channels"] == 4:
            low = x.amin(dim=(1, 2, 3), keepdim=True)
            high = x.amax(dim=(1, 2, 3), keepdim=True)
            return net((x - low) / (high - low + 1e-8))
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

    Queries are scored in bounded blocks against every labelled crop. Full
    rankings and eligible labels are retained without allocating the whole
    pairwise similarity matrix, so tens of thousands of crops remain
    practical.

    ``chance_map`` is the exact expected average precision of a uniformly
    random full ranking, averaged over queries with another same-label crop.
    ``chance_precision`` is the prevalence among candidates, including
    singleton queries; it is not the random average-precision baseline.

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
    k = max(1, min(int(k), len(pairs) - 1))
    ranks = np.arange(1, len(pairs))
    hits = np.empty(len(pairs), dtype=np.float64)
    positives = np.empty(len(pairs), dtype=np.int64)
    ap = np.empty(len(pairs), dtype=np.float64)
    correct = 0
    for start in range(0, len(pairs), 256):
        stop = min(start + 256, len(pairs))
        scores = sub[start:stop] @ sub.T
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        order = np.argsort(-scores, axis=1, kind="stable")[:, :-1]
        same = classes[order] == classes[start:stop, None]
        hits[start:stop] = same[:, :k].mean(axis=1)
        positives[start:stop] = same.sum(axis=1)
        precision = np.cumsum(same, axis=1) / ranks
        ap[start:stop] = ((precision * same).sum(axis=1)
                         / np.maximum(positives[start:stop], 1))
        for label, neighbors in zip(classes[start:stop], classes[order[:, :k]]):
            near = list(neighbors)
            counts = {c: near.count(c) for c in near}
            best = max(counts.values())
            vote = next(c for c in near if counts[c] == best)
            correct += vote == label
    kept = positives > 0
    freq = {c: float(np.mean(classes == c)) for c in set(classes)}
    chance = np.asarray([(freq[c] * len(pairs) - 1) / (len(pairs) - 1)
                         for c in classes])
    candidates = len(pairs) - 1
    harmonic = np.sum(1.0 / ranks)
    chance_ap = positives[kept] / candidates
    if candidates > 1:
        chance_ap = chance_ap + ((candidates - positives[kept])
                                * (harmonic - 1)
                                / (candidates * (candidates - 1)))
    return {
        "knn_accuracy": float(correct / len(pairs)),
        "map": float(ap[kept].mean()),
        "precision_at_k": float(hits.mean()),
        "chance_map": float(chance_ap.mean()),
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


def _dino_classifier_from_config(config: Mapping[str, Any]):
    """Rebuild an own-DINO classifier without reading a checkpoint or network.

    :param config: primitive constructor settings including ordered source
        channels, recorded scaling, resize, policy and encoder freezing.
    :returns: trainable torch module accepting raw ``NCHW`` selected planes.
    :raises EmbeddingError: settings or input violate the recorded contract.
    """
    import copy
    import torch
    import timm
    from .qt.i18n import tr

    def refuse():
        """Refuse an incompatible own-DINO classifier contract."""
        raise EmbeddingError(tr(
            "Own-DINO classifier settings, checkpoint or input are incompatible."))

    config = copy.deepcopy(dict(config))
    try:
        for name in ("in_chans", "size", "num_classes"):
            value = config[name]
            if type(value) is not int or value < 1:
                refuse()
        if type(config["freeze_encoder"]) is not bool:
            refuse()
        if type(config["normalize_features"]) is not bool:
            refuse()
        scales = config["scale"]
        channels = config["source_channels"]
        if (not isinstance(scales, (list, tuple)) or not scales
                or not isinstance(channels, (list, tuple))
                or len(channels) != len(scales)
                or any(type(c) is not int or c < 0 for c in channels)
                or len(set(channels)) != len(channels)
                or any(type(s) not in (int, float) or not np.isfinite(s)
                       or s < 0 or s > np.finfo(np.float32).max for s in scales)):
            refuse()
        policy = config["channel_policy"]
        if policy not in CHANNEL_POLICIES:
            refuse()
        if config["in_chans"] != (len(scales) if policy == CHANNEL_PROJECT else 1):
            refuse()
        arch = config["arch"]
        if not isinstance(arch, str) or arch not in timm.list_models():
            refuse()
        digest = config["source_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            refuse()
    except (KeyError, TypeError, OverflowError) as exc:
        raise EmbeddingError(tr(
            "Own-DINO classifier settings, checkpoint or input are incompatible.")) from exc
    kwargs = {"pretrained": False, "num_classes": 0,
              "in_chans": config["in_chans"]}
    if arch.startswith(("vit", "deit")):
        kwargs.update(img_size=config["size"], dynamic_img_size=True)
    backbone = timm.create_model(arch, **kwargs)
    width = int(backbone.num_features)
    input_channels = len(scales)
    if policy == CHANNEL_PER_CHANNEL:
        width *= input_channels

    class Classifier(torch.nn.Module):
        """Own-DINO encoder with immutable preprocessing and a trainable head."""

        def __init__(self):
            """Attach recorded scaling, teacher encoder and an untrained head."""
            super().__init__()
            self.encoder = backbone
            self.head = torch.nn.Linear(width, config["num_classes"])
            self.model_name = "own_dino"
            self.num_classes = config["num_classes"]
            self.image_size = config["size"]
            self._own_dino_config = copy.deepcopy(config)
            self.register_buffer("channel_scale", torch.tensor(
                scales, dtype=torch.float32).reshape(1, input_channels, 1, 1))
            self.encoder.requires_grad_(not config["freeze_encoder"])
            self.train(True)

        def train(self, mode=True):
            """Keep a frozen teacher in evaluation mode while training its head."""
            super().train(mode)
            if config["freeze_encoder"]:
                self.encoder.eval()
            return self

        def features(self, images):
            """Scale selected raw NCHW planes and return recorded encoder features."""
            if (not isinstance(images, torch.Tensor) or images.ndim != 4
                    or images.shape[1] != input_channels
                    or min(images.shape) < 1 or images.is_complex()
                    or images.dtype == torch.bool):
                refuse()
            images = images.to(dtype=self.channel_scale.dtype)
            if not torch.isfinite(images).all():
                refuse()
            scale = self.channel_scale
            images = torch.where(scale > 0, images / torch.where(
                scale > 0, scale, torch.ones_like(scale)), 0).clamp(0, 1)
            batch = images.shape[0]
            if policy == CHANNEL_PER_CHANNEL:
                images = images.reshape(batch * input_channels, 1,
                                        images.shape[-2], images.shape[-1])
            if images.shape[-2:] != (self.image_size, self.image_size):
                images = torch.nn.functional.interpolate(
                    images, size=(self.image_size, self.image_size),
                    mode="bilinear", align_corners=False)
            features = self.encoder(images)
            if policy == CHANNEL_PER_CHANNEL:
                features = features.reshape(batch, -1)
            if config["normalize_features"]:
                features = torch.nn.functional.normalize(features, dim=1)
            return features

        def forward(self, images):
            """Scale selected raw NCHW planes and return class logits."""
            return self.head(self.features(images))

    return Classifier()


def _dino_classifier_from_checkpoint(path: str, *, num_classes: int,
                                     source_channels: Sequence[int],
                                     freeze_encoder: bool = True,
                                     normalize_features: bool = True):
    """Create a headed classifier from a locally supplied own-DINO teacher.

    :param path: local checkpoint, decoded strictly with ``weights_only=True``.
    :param num_classes: positive output width; the new head is untrained.
    :param source_channels: caller declaration of physical-channel order matching
        the saved scale. Historical DINO files do not prove that declaration.
    :param freeze_encoder: freeze teacher parameters and running statistics.
    :param normalize_features: L2-normalize concatenated features before the head.
    :returns: trainable module; no source file is changed or downloaded.
    :raises EmbeddingError: checkpoint does not describe compatible DINO weights.
    """
    import torch
    from .qt.i18n import tr

    with open(path, "rb", buffering=0) as stream:
        before = os.fstat(stream.fileno())
        before_hash = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            before_hash.update(chunk)
        digest = before_hash.hexdigest()
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
        stream.seek(0)
        after_hash = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            after_hash.update(chunk)
        after = os.fstat(stream.fileno())
        identity = lambda status: (
            status.st_dev, status.st_ino, status.st_size,
            status.st_mtime_ns, status.st_ctime_ns)
        if identity(before) != identity(after) or after_hash.hexdigest() != digest:
            raise EmbeddingError(tr(
                "Own-DINO classifier settings, checkpoint or input are incompatible."))
    if (not isinstance(state, Mapping) or not isinstance(state.get("setup"), Mapping)
            or not isinstance(state.get("teacher"), Mapping)):
        raise EmbeddingError(tr(
            "Own-DINO classifier settings, checkpoint or input are incompatible."))
    setup = state["setup"]
    config = {name: setup.get(name) for name in
              ("arch", "in_chans", "size", "channel_policy")}
    config.update(scale=state.get("scale"), source_channels=list(source_channels),
                  num_classes=num_classes, freeze_encoder=freeze_encoder,
                  normalize_features=normalize_features, source_sha256=digest)
    weights = state["teacher"]
    if any(not isinstance(value, torch.Tensor) or
           (value.is_floating_point() and not torch.isfinite(value).all())
           or value.is_complex() for value in weights.values()):
        raise EmbeddingError(tr(
            "Own-DINO classifier settings, checkpoint or input are incompatible."))
    model = _dino_classifier_from_config(config)
    model.encoder.load_state_dict(weights, strict=True)
    return model


def _dino_finetune_classifier(crops: Any, checkpoint_path: str, output_path: str,
                               *, labels: Sequence[Any], groups: Sequence[Any],
                               source_channels: Sequence[int],
                               freeze_encoder: bool = True,
                               normalize_features: bool = True,
                               epochs: int = 20, batch_size: int = 32,
                               lr: float = 1e-4, holdout: float = 0.2,
                               seed: int = 0, device: str = "cpu",
                               progress: Optional[Callable] = None,
                               cancelled: Optional[Callable] = None) -> Dict[str, Any]:
    """Fine-tune a local own-DINO teacher on explicitly labelled raw crops.

    :param crops: full raw ``NHWC`` stack; inputs are never modified.
    :param checkpoint_path: locally supplied, restricted-load DINO teacher.
    :param output_path: new best-classifier artifact; existing files refuse.
    :param labels: one nonblank class per crop, in crop order.
    :param groups: one known acquisition-well identity per crop; whole groups
        are held out and every class must occur on both sides.
    :param source_channels: ordered physical indices selecting raw planes and
        declaring the order represented by the teacher's saved scaling.
    :param freeze_encoder: train only the new head when true, otherwise both.
    :param normalize_features: L2-normalize features before the classifier head.
    :param epochs: positive supervised epoch count, independent of pretraining.
    :param batch_size: positive number of labelled crops per optimizer step.
    :param lr: finite positive AdamW learning rate.
    :param holdout: fraction passed to the existing grouped split utility.
    :param seed: nonnegative CPU shuffle and initialization seed.
    :param device: explicitly selected torch device; CPU by default.
    :param progress: optional ``(epoch, epochs, train_loss, validation_loss)``.
    :param cancelled: optional predicate checked before each training batch.
    :returns: best artifact path, classes, actual split provenance and losses.
    :raises EmbeddingError: invalid data, settings or teacher configuration.
    :raises InterruptedError: cancellation; no result artifact is published.
    """
    import tempfile
    import torch
    from .qt.i18n import tr
    from .classifier_evaluation import grouped_split
    from .torch_artifacts import atomic_torch_save, make_model_artifact

    def refuse():
        """Refuse an incompatible own-DINO classifier contract."""
        raise EmbeddingError(tr(
            "Own-DINO classifier settings, checkpoint or input are incompatible."))

    for value in (epochs, batch_size):
        if type(value) is not int or value < 1:
            refuse()
    if type(seed) is not int or seed < 0:
        refuse()
    if type(lr) not in (int, float) or not np.isfinite(lr) or lr <= 0:
        refuse()
    channels = tuple(source_channels)
    array = np.asarray(crops)
    if (array.ndim != 4 or min(array.shape) < 1
            or array.dtype.kind not in "uif" or not np.isfinite(array).all()
            or not channels or len(set(channels)) != len(channels)
            or any(type(c) is not int or c < 0 or c >= array.shape[-1]
                   for c in channels)):
        refuse()
    if (len(labels) != len(array) or len(groups) != len(array)
            or any(value is None or str(value).strip().casefold() in
                   ("", "nan", "none", "unknown", "<na>")
                   for value in list(labels) + list(groups))):
        refuse()
    if os.path.exists(output_path) or os.path.realpath(output_path) == os.path.realpath(checkpoint_path):
        refuse()
    classes = sorted({str(value) for value in labels})
    if len(classes) < 2:
        refuse()
    targets = np.asarray([classes.index(str(value)) for value in labels], dtype=np.int64)
    train_indices, validation_indices, split = grouped_split(
        [str(value) for value in groups], targets, holdout, seed, group_by="well")
    selected = np.array(array[..., channels].transpose(0, 3, 1, 2),
                        dtype=np.float32, copy=True)
    if not np.isfinite(selected).all():
        refuse()
    tensor = torch.from_numpy(selected)
    target_tensor = torch.from_numpy(targets)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = _dino_classifier_from_checkpoint(
            checkpoint_path, num_classes=len(classes), source_channels=channels,
            freeze_encoder=freeze_encoder, normalize_features=normalize_features)
    model.to(torch.device(device))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history = []
    best_loss = float("inf")

    def stop_if_requested():
        """Stop between batches when the caller requests cancellation."""
        if cancelled is not None and cancelled():
            raise InterruptedError(tr("Classifier training stopped."))

    output_parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_parent, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".spacr-own-dino-", dir=output_parent) as staging:
        staged_path = os.path.join(staging, "best.pt")
        for epoch in range(1, epochs + 1):
            stop_if_requested()
            model.train()
            total_loss = 0.0
            shuffled = train_indices[torch.randperm(len(train_indices), generator=generator).numpy()]
            for begin in range(0, len(shuffled), batch_size):
                stop_if_requested()
                indices = shuffled[begin:begin + batch_size]
                images = tensor[indices].to(device)
                targets_batch = target_tensor[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(images), targets_batch)
                if not torch.isfinite(loss):
                    refuse()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(indices)
            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for begin in range(0, len(validation_indices), batch_size):
                    stop_if_requested()
                    indices = validation_indices[begin:begin + batch_size]
                    loss = torch.nn.functional.cross_entropy(
                        model(tensor[indices].to(device)), target_tensor[indices].to(device))
                    if not torch.isfinite(loss):
                        refuse()
                    validation_loss += float(loss.cpu()) * len(indices)
            stop_if_requested()
            train_loss = total_loss / len(train_indices)
            validation_loss /= len(validation_indices)
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "validation_loss": validation_loss})
            if validation_loss < best_loss:
                best_loss = validation_loss
                payload = make_model_artifact(
                    model, epoch=epoch, metrics=history[-1], best_metric=best_loss,
                    include_rng=False, classes=classes,
                    channels=[str(c) for c in channels],
                    preprocessing={"raw_channel_last": True,
                                   "scaling_in_model": True,
                                   "source_channel_order_is_caller_declared": True})
                payload["grouped_split"] = split.to_dict()
                payload["train_groups"] = sorted({str(groups[i]) for i in train_indices})
                payload["validation_groups"] = sorted({str(groups[i]) for i in validation_indices})
                stop_if_requested()
                atomic_torch_save(payload, staged_path)
            if progress is not None:
                progress(epoch, epochs, train_loss, validation_loss)
        stop_if_requested()
        os.link(staged_path, output_path)
    return {"path": os.fspath(output_path), "classes": classes,
            "split": split.to_dict(), "loss": history}



def _dino_classifier_apply(crops: Any, path: str, *, batch_size: int = 32,
                           device: str = "cpu") -> Dict[str, Any]:
    """Apply a saved own-DINO classifier to full raw channel-last crops.

    Saved source-channel order selects physical planes once. Scaling, resize
    and feature normalization remain inside the reconstructed classifier;
    this route does not use teacher embedding preprocessing or estimate scales.

    :param crops: nonempty finite numeric raw ``NHWC`` stack, never modified.
    :param path: local classifier artifact, decoded from the same regular-file
        stream that supplies its before/after SHA-256 and identity checks.
    :param batch_size: positive crop count per inference batch.
    :param device: explicit torch device; CPU by default.
    :returns: ``features``, ``logits``, ``probabilities``, ``predictions``, saved
        ``classes`` and ``source_channels``, and the loaded artifact's
        ``checkpoint_sha256``. Arrays preserve crop order; probability columns
        and predicted labels follow saved class order.
    :raises EmbeddingError: incompatible crops, artifact metadata or changed file.
    """
    import torch
    from .qt.i18n import tr
    from .torch_artifacts import load_model_artifact

    def refuse():
        """Refuse an incompatible own-DINO classifier application."""
        raise EmbeddingError(tr(
            "Own-DINO classifier settings, checkpoint or input are incompatible."))

    array = np.asarray(crops)
    if (type(batch_size) is not int or batch_size < 1
            or array.ndim != 4 or min(array.shape) < 1
            or array.dtype.kind not in "uif" or not np.isfinite(array).all()):
        refuse()
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            os.close(descriptor)
            refuse()
        with os.fdopen(descriptor, "rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            handle.seek(0)
            model, payload = load_model_artifact(
                handle, map_location="cpu", strict=True, weights_only=True)
            handle.seek(0)
            after_digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                after_digest.update(chunk)
            after = os.fstat(handle.fileno())
            identity = lambda status: (
                status.st_dev, status.st_ino, status.st_size,
                status.st_mtime_ns, status.st_ctime_ns)
            if identity(before) != identity(after) or digest.digest() != after_digest.digest():
                refuse()
    except OSError as exc:
        raise EmbeddingError(tr(
            "Own-DINO classifier settings, checkpoint or input are incompatible.")) from exc
    config = getattr(model, "_own_dino_config", None)
    classes = payload.get("classes")
    if (not isinstance(config, Mapping) or payload.get("legacy") is not False
            or not isinstance(classes, (list, tuple))
            or len(classes) != config.get("num_classes")
            or any(not isinstance(value, str) or not value.strip() for value in classes)
            or len(set(classes)) != len(classes)):
        refuse()
    channels = config["source_channels"]
    if any(channel >= array.shape[-1] for channel in channels):
        refuse()
    model.eval().to(torch.device(device))
    features = []
    logits = []
    with torch.inference_mode():
        for begin in range(0, len(array), batch_size):
            selected = np.array(array[begin:begin + batch_size, ..., channels].transpose(
                0, 3, 1, 2), dtype=np.float32, copy=True)
            if not np.isfinite(selected).all():
                refuse()
            encoded = model.features(torch.from_numpy(selected).to(device))
            output = model.head(encoded)
            if not torch.isfinite(encoded).all() or not torch.isfinite(output).all():
                refuse()
            features.append(encoded.detach().float().cpu().numpy())
            logits.append(output.detach().float().cpu())
    joined = torch.cat(logits)
    probabilities = torch.softmax(joined, dim=1).numpy()
    return {"features": np.concatenate(features), "logits": joined.numpy(),
            "probabilities": probabilities,
            "predictions": [classes[index] for index in joined.argmax(dim=1).tolist()],
            "classes": list(classes), "source_channels": list(channels),
            "checkpoint_sha256": digest.hexdigest()}


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
    if array.ndim != 4 or len(labels) != len(array):
        from .qt.i18n import tr
        raise EmbeddingError(tr(
            "Provide a four-dimensional crop stack and one label per crop."))
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
    the same label. A well identifier cannot span contradictory nonblank
    plate, row or column metadata. Missing metadata does not establish that
    the identifier names an independent acquisition well.

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
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = kept[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        from .qt.i18n import tr

        raise ValueError(tr("Multiple-instance feature values must remain finite when converted to float32."))
    if any(str(column).startswith(EMBEDDING_PREFIX) for column in feature_columns):
        for column in ("_embedding_fingerprint", "_embedding_weights_sha256"):
            if column not in kept.columns:
                continue
            identities = kept[column].dropna().astype(str).str.strip().drop_duplicates()
            identities = identities[identities.ne("") & identities.str.casefold().ne("unknown")]
            if column == "_embedding_weights_sha256":
                identities = identities[identities.str.fullmatch(r"[0-9a-fA-F]{64}")].str.lower()
            else:
                identities = identities.map(
                    lambda value: value.lower() if re.fullmatch(r"[0-9a-fA-F]{16}", value) else value)
            if identities.nunique() > 1:
                from .qt.i18n import tr

                raise ValueError(tr("Selected embedding features have conflicting known encoder fingerprints or weights."))
    acquisition = [column for column in ('plateID', 'rowID', 'columnID')
                   if column in kept.columns]
    for well, group in kept.groupby(well_column, sort=True):
        for column in acquisition:
            known = group[column].dropna().astype(str).str.strip()
            if known[known.ne('')].nunique() > 1:
                from .qt.i18n import tr

                raise ValueError(tr("Well identifier {well} spans different acquisition wells; use complete, unique well identifiers.", well=well))
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


def _mil_frame_from_result(db_path, keys, result, labels, *, encoder_entry=None,
                           label_column=_MIL_LABEL_COLUMN):
    """Join frozen current-result vectors to their measured object identities.

    :param db_path: The database from which the encoded crops were loaded.
    :param keys: Object keys in the exact order supplied to the encoder.
    :param result: The corresponding embedding result, independent of controls.
    :param labels: Explicit well labels accepted by the existing database join.
    :param encoder_entry: The encoder provenance captured with the result.
    :param label_column: The selected well-label column.
    :returns: Source-ordered vectors, object/well metadata and explicit labels.
    :raises ValueError: When keys are missing, repeated or no longer join crops.
    """
    from dataclasses import asdict
    import json
    from .qt.i18n import tr

    keys = [str(key) for key in keys]
    if not keys or any(not key.strip() for key in keys) or len(set(keys)) != len(keys):
        raise ValueError(tr("The current embedding has no unique source object keys."))
    stored = result.to_frame(keys).rename(columns={"object_id": "prcfo"})
    spec = getattr(result, "spec", None)
    if spec is not None:
        stored["_embedding_fingerprint"] = spec.fingerprint()
        stored["_embedding_spec"] = json.dumps(asdict(spec), sort_keys=True)
    if encoder_entry is not None:
        for column, attribute in (("_embedding_weights_sha256", "sha256"),
                                  ("_embedding_encoder_key", "key"),
                                  ("_embedding_encoder_source", "source")):
            stored[column] = str(getattr(encoder_entry, attribute, "") or "")
    frame = _mil_frame_from_db(db_path, labels, label_column=label_column,
                               stored_embeddings=stored)
    if set(frame["prcfo"]) != set(keys):
        raise ValueError(tr("Some current embedding objects no longer have crop metadata."))
    return frame.set_index("prcfo", drop=False).loc[keys].reset_index(drop=True)


def _mil_frame_from_db(db_path: str, labels: Any = None, *,
                       label_column: str = _MIL_LABEL_COLUMN,
                       stored_embeddings=None):
    """A per-cell table for the well-label model from stored crop embeddings.

    The embeddings saved in ``measurements.db`` by
    :func:`spacr.active_learning._store_crop_embeddings` are joined through
    ``prcfo`` to the crop table, so every row carries its object key
    (``prcfo``), crop path, plate, row, column, field and object, a
    ``wellID`` of ``<plate>_<row>_<column>``, and the ``emb_`` columns.

    :param db_path: path to ``measurements.db``.
    :param labels: optional well labels: a table, or a path to one, with a
        ``wellID`` or ``well`` column and ``label_column``. Its wells may be
        named ``<plate>_<row>_<column>``, ``<row>_<column>`` or
        ``<row><column>``; each key resolves independently. Short keys
        must identify exactly one plate/well, and duplicate/alias labels
        for the same well must agree.
        Cells of an unlabelled well keep an empty label.
    :param label_column: the label column of ``labels``.
    :param stored_embeddings: optional source-bound in-memory object vectors;
        when supplied, the database embedding table is not read or changed.
    :returns: one row per crop with a stored embedding.
    :raises ValueError: when the database has no stored embeddings, none
        joins a crop, ``labels`` has no well or label column, labels
        conflict or a short well key is ambiguous across plates.
    """
    import pandas as pd

    from . import tabular
    from .active_learning import _EMBEDDING_TABLE
    from .agreement import PNG_TABLE

    stored = stored_embeddings
    if stored is None and os.path.exists(str(db_path)):
        try:
            stored = tabular.read_database(
                db_path, _EMBEDDING_TABLE, canonicalise=False, report=None,
                repair_plate_ids=False, migrate=False, read_only=True)[0]
        except ValueError as error:
            if str(error) != f"Table not found in database: {_EMBEDDING_TABLE}":
                raise
    if stored is None or stored.empty:
        raise ValueError(
            f"{db_path} has no stored crop embeddings (no crop_embedding "
            "table with rows)")
    crops = tabular.read_table(db_path, table=PNG_TABLE, canonicalise=False,
                               report=None, repair_plate_ids=False,
                               migrate=False, read_only=True)
    keep = [c for c in ("prcfo", "png_path", "plateID", "rowID", "columnID",
                        "fieldID") if c in crops.columns]
    if "prcfo" not in keep:
        raise ValueError(f"the {PNG_TABLE} table of {db_path} has no prcfo column")
    crops = crops[keep].dropna(subset=["prcfo"]).copy()
    crops["prcfo"] = crops["prcfo"].astype(str)
    stored = stored.copy()
    stored["prcfo"] = stored["prcfo"].astype(str)
    from .qt.i18n import tr

    stored = stored.drop_duplicates()
    repeated = stored["prcfo"].duplicated(keep=False)
    if repeated.any():
        key = stored.loc[repeated, "prcfo"].iloc[0]
        raise ValueError(tr("Stored embeddings disagree for object key {key}.", key=key))
    repeated = crops["prcfo"].duplicated(keep=False)
    for key, group in crops.loc[repeated].groupby("prcfo", sort=False):
        for column in ("plateID", "rowID", "columnID", "fieldID", "png_path"):
            if column not in group.columns:
                continue
            values = group[column].dropna()
            known = values[values.astype(str).str.strip().ne("")]
            if column != "png_path" and known.astype(str).str.strip().nunique() > 1:
                raise ValueError(tr("Crop acquisition metadata disagrees for object key {key}.", key=key))
            if not known.empty:
                crops.at[group.index[0], column] = known.iloc[0]
    frame = crops.drop_duplicates("prcfo").merge(stored, on="prcfo",
                                                 how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError(f"no stored embedding in {db_path} joins a crop")
    from .schema import parse_prcfo

    parsed = {}
    for column in ("plateID", "rowID", "columnID", "fieldID"):
        if column not in frame.columns:
            frame[column] = None
        missing = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
        for index in frame.index[missing]:
            if index not in parsed:
                key = frame.at[index, "prcfo"]
                try:
                    parsed[index] = parse_prcfo(key).field.to_dict()
                except (TypeError, ValueError) as error:
                    raise ValueError(tr("Cannot recover acquisition metadata from object key {key}.", key=key)) from error
            frame.at[index, column] = parsed[index][column]
    frame["object"] = frame["prcfo"].str.rsplit("_", n=1).str[-1]
    frame.insert(1, _MIL_WELL_COLUMN, frame["plateID"].astype(str) + "_"
                 + frame["rowID"].astype(str) + "_"
                 + frame["columnID"].astype(str))
    if labels is None:
        return frame
    if not isinstance(labels, pd.DataFrame):
        labels = tabular.read_table(labels, canonicalise=False, report=None)
    well_column = _mil_well_column(labels, _MIL_WELL_COLUMN)
    for column in (well_column, label_column):
        if column not in labels.columns:
            raise ValueError(f"the well-label table has no {column!r} column")
    named = {}
    for key, label in zip(labels[well_column], labels[label_column]):
        if pd.isna(key) or pd.isna(label):
            continue
        key = str(key)
        if key in named and named[key] != label:
            raise ValueError(tr("The well-label table assigns conflicting labels to {well}.", well=key))
        named[key] = label
    row, col = frame["rowID"].astype(str), frame["columnID"].astype(str)
    full_wells = set(frame[_MIL_WELL_COLUMN])
    aliases = {}
    for full, short, compact in zip(frame[_MIL_WELL_COLUMN], row + "_" + col, row + col):
        for alias in (short, compact):
            aliases.setdefault(alias, set()).add(full)
    resolved = {}
    for key, label in named.items():
        matches = {key} if key in full_wells else aliases.get(key, set())
        if len(matches) > 1:
            raise ValueError(tr("Well label {well} is ambiguous across plates; use the full plate_row_column identifier.", well=key))
        if not matches:
            continue
        full = next(iter(matches))
        if full in resolved and resolved[full] != label:
            raise ValueError(tr("The well-label table assigns conflicting labels to {well}.", well=full))
        resolved[full] = label
    frame[label_column] = frame[_MIL_WELL_COLUMN].map(resolved)
    return frame


#: Key prefix for an encoder's model-zoo entry. Distinct from a checkpoint's
#: filename-derived key because an encoder has no file of spaCR's own -- it is
#: named by backbone and policy, which together are what a later run must
#: match for its numbers to be comparable.
ENCODER_KEY_PREFIX = "encoder:"


def encoder_key(spec: "EmbeddingSpec") -> str:
    """The stable id for one encoder configuration.

    The key combines backbone and channel policy, because both determine
    what a dimension represents. The same backbone under
    :data:`CHANNEL_PER_CHANNEL` and :data:`CHANNEL_PROJECT` produces columns
    of the same width and dtype that measure different quantities -- one is
    per-stain, the other a mixture -- so a key naming only the backbone
    would allow them to be compared.

    :param spec: the run's configuration. Only ``backbone`` and
        ``channel_policy`` reach the key. Other fields that change the
        values, such as ``channels``, ``normalize`` and ``channel_scale``,
        are covered by :meth:`EmbeddingSpec.fingerprint`; batch size and
        device are in neither.
    """
    return f"{ENCODER_KEY_PREFIX}{spec.backbone}/{spec.channel_policy}"


def _weights_on_disk(backbone: str, *, checkpoint_path: Optional[str] = None,
                     checkpoint_sha256: Optional[str] = None) -> Tuple[str, str, int]:
    """Find the encoder's existing checkpoint and hash its actual bytes.

    :returns: ``(path, sha256, size_bytes)``; the path is ``''`` and the
        digest ``''`` when no readable local checkpoint can be resolved.

    Foundation Hugging Face caches use the loader's pinned revision;
    SubCell uses the existing torch hub checkpoint directory; local DINO
    uses its configured path. Other backbones use timm's cache metadata.
    No checkpoint is downloaded or loaded, and torch is not imported.

    Cell-DINO uses ``checkpoint_path`` only when its stable regular-file
    bytes match ``checkpoint_sha256``. Missing, changed or mismatched files
    return the same empty provenance tuple as an unresolved cache.
    """
    import sys

    try:
        if backbone == "cell_dino":
            if (not isinstance(checkpoint_path, str) or not checkpoint_path
                    or not isinstance(checkpoint_sha256, str)
                    or not re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha256)):
                return "", "", 0
            path = os.path.realpath(os.path.expanduser(checkpoint_path))
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                os.close(descriptor)
                return "", "", 0
            with os.fdopen(descriptor, "rb") as handle:
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                after = os.fstat(handle.fileno())
                current = os.stat(path)
                identity = lambda status: (
                    status.st_dev, status.st_ino, status.st_size,
                    status.st_mtime_ns, status.st_ctime_ns)
                if (identity(before) != identity(after)
                        or identity(after) != identity(current)
                        or size != after.st_size
                        or digest.hexdigest() != checkpoint_sha256.lower()):
                    return "", "", 0
                return path, digest.hexdigest(), size
        if backbone.startswith(_DINO_PREFIX):
            path = backbone[len(_DINO_PREFIX):]
        elif backbone in _FOUNDATION_MODELS:
            info = _FOUNDATION_MODELS[backbone]
            if "repo" in info:
                from huggingface_hub import try_to_load_from_cache

                path = None
                for filename in ("model.safetensors", "pytorch_model.bin"):
                    candidate = try_to_load_from_cache(
                        info["repo"], filename, revision=info["revision"])
                    if isinstance(candidate, str) and os.path.isfile(candidate):
                        path = candidate
                        break
            elif "url" in info:
                hub = sys.modules.get("torch.hub")
                cache_home = os.getenv("XDG_CACHE_HOME", "~/.cache")
                torch_home = os.path.expanduser(os.getenv(
                    "TORCH_HOME", os.path.join(cache_home, "torch")))
                directory = (hub.get_dir() if hub is not None else
                             os.path.join(torch_home, "hub"))
                path = os.path.join(directory, "checkpoints",
                                    info["url"].rsplit("/", 1)[-1])
            else:
                return "", "", 0
        else:
            import timm
            from huggingface_hub import try_to_load_from_cache

            config = timm.get_pretrained_cfg(backbone)
            repo = getattr(config, "hf_hub_id", None)
            filename = (getattr(config, "hf_hub_filename", None)
                        or "model.safetensors")
            if not repo:
                return "", "", 0
            path = try_to_load_from_cache(repo, filename)
        if not isinstance(path, str) or not os.path.isfile(path):
            return "", "", 0
        digest = hashlib.sha256()
        size = 0
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except Exception:
        return "", "", 0

    return path, digest.hexdigest(), size


def encoder_entry(spec: Optional["EmbeddingSpec"] = None, *,
                  scorecard: Optional[Mapping[str, Any]] = None):
    """Describe this encoder as a :class:`spacr.model_zoo.ModelEntry`.

    The entry combines encoder configuration, available checkpoint provenance
    and optional retrieval metrics. Compare entries only when their channel
    policies are compatible.

    The entry records the backbone, channel policy and digest of readable
    local checkpoint bytes without downloading or loading a model. Public
    backbones use timm cache metadata; OpenPhenom and ChAda-ViT use pinned
    Hugging Face revisions; SubCell uses its official torch hub checkpoint;
    a local DINO encoder uses its configured checkpoint path. Cell-DINO
    uses the declared local checkpoint only when stable regular-file bytes
    match its expected SHA-256; changed or mismatched files leave no digest.
    Foundation
    entries name the original provider and checkpoint URL. A checksum alone
    does not verify a model or establish its training-data provenance.

    If no readable local checkpoint is found, the entry keeps an empty digest
    and explains this in its notes. :func:`spacr.model_zoo.fetch` refuses
    entries without a digest.

    :param spec: Encoder configuration; uses ``EmbeddingSpec()`` when omitted.
    :param scorecard: Optional retrieval metrics measured on labelled controls,
        stored in :attr:`ModelEntry.metrics` for display by ``scorecard_lines``.
    :returns: A ``ModelEntry`` with kind ``'encoder'``.
    """
    from .model_zoo import UNKNOWN, ModelEntry
    from .qt.i18n import tr

    spec = spec if spec is not None else EmbeddingSpec()
    if spec.backbone == "cell_dino":
        path, digest, size = _weights_on_disk(
            spec.backbone, checkpoint_path=spec.checkpoint_path,
            checkpoint_sha256=spec.checkpoint_sha256)
    else:
        path, digest, size = _weights_on_disk(spec.backbone)

    notes = [
        tr("Channel policy: {policy}. Dimensions from different policies "
           "are not comparable.").format(policy=spec.channel_policy),
    ]
    if not digest:
        notes.append(
            tr("No checksum: no readable local checkpoint could be resolved "
               "in this environment."))
    if scorecard is None:
        notes.append(
            tr("No scorecard. Measure retrieval against labelled phenotype "
               "controls and attach the results."))

    info = _FOUNDATION_MODELS.get(spec.backbone, {})
    uri = f"timm:{spec.backbone}"
    trained_by = f"timm / {spec.backbone} pretrained weights"
    if "repo" in info:
        filename = ("pytorch_model.bin" if path.endswith("pytorch_model.bin")
                    else "model.safetensors")
        uri = (f"https://huggingface.co/{info['repo']}/resolve/"
               f"{info['revision']}/{filename}")
        trained_by = info["repo"]
    elif "url" in info:
        uri = info["url"]
        trained_by = info["label"]
    elif info:
        uri = ""
        trained_by = UNKNOWN
    elif spec.backbone.startswith(_DINO_PREFIX):
        uri = spec.backbone[len(_DINO_PREFIX):]
        trained_by = UNKNOWN

    return ModelEntry(
        key=encoder_key(spec),
        name=f"{spec.backbone} ({spec.channel_policy})",
        kind="encoder",
        source="local" if path else "remote",
        path=path,
        uri=uri,
        version="1",
        sha256=digest,
        size_bytes=size,
        trained_on=UNKNOWN,
        trained_by=trained_by,
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


def _embedding_umap(features: Any, *, n_neighbors: int = 15,
                    min_dist: float = 0.1, seed: int = 0):
    """A two-dimensional UMAP of an embedding matrix.

    Features are median-centred and scaled by their standard deviation
    before the map, so no single dimension dominates. ``n_neighbors`` is
    capped below the number of rows, so a small set still maps.

    :param features: numeric frame, one row per crop.
    :param n_neighbors: UMAP neighbourhood size.
    :param min_dist: UMAP minimum distance between mapped points.
    :param seed: random state, so a rerun draws the same map.
    :returns: a frame with ``umap_1`` and ``umap_2``, indexed like
        ``features``.
    :raises ValueError: with fewer than three rows.
    """
    import pandas as pd

    frame = pd.DataFrame(features)
    values = frame.to_numpy(dtype=np.float64)
    if values.shape[0] < 3:
        raise ValueError("a UMAP needs at least three crops")
    values = values - np.median(values, axis=0)
    spread = values.std(axis=0)
    values = values / np.where(spread > 0, spread, 1.0)
    try:
        from .utils import umap
        _ = umap.UMAP
    except ImportError as exc:
        raise ImportError(
            "The embedding UMAP needs umap-learn: pip install "
            "'umap-learn>=0.5.11,<1.0'") from exc
    reducer = umap.UMAP(n_components=2,
                        n_neighbors=max(2, min(int(n_neighbors),
                                               values.shape[0] - 1)),
                        min_dist=float(min_dist), random_state=int(seed),
                        init="random")
    mapped = reducer.fit_transform(values)
    return pd.DataFrame(mapped, index=frame.index,
                        columns=["umap_1", "umap_2"])


def _embedding_classifier_scorecard(features: Any, labels: Mapping[Any, Any],
                                    *, folds: int = 5,
                                    seed: int = 0) -> Dict[str, float]:
    """Cross-validated accuracy of a linear classifier on embeddings.

    A standardised logistic regression is scored with stratified folds over
    the labelled crops; ``chance`` is the share of the largest class, which
    a classifier that always guesses it would reach.

    :param features: numeric frame indexed by crop key.
    :param labels: crop key to class; blanks and unknown keys are dropped.
    :param folds: folds, capped by the smallest class.
    :param seed: fold shuffling seed.
    :returns: ``accuracy``, ``accuracy_sd``, ``chance``, ``n``, ``classes``
        and ``folds``.
    :raises ValueError: with fewer than two crops per class or one class.
    """
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    frame = pd.DataFrame(features)
    frame.index = frame.index.map(str)
    pairs = [(str(key), str(value)) for key, value in labels.items()
             if value is not None and str(value) != ""
             and str(key) in frame.index]
    classes = np.asarray([p[1] for p in pairs], dtype=object)
    counts = pd.Series(classes).value_counts() if len(pairs) else None
    if counts is None or len(counts) < 2 or int(counts.min()) < 2:
        raise ValueError(
            "a classifier scorecard needs two classes with at least two "
            "labelled crops each")
    values = frame.loc[[p[0] for p in pairs]].to_numpy(dtype=np.float64)
    folds = max(2, min(int(folds), int(counts.min())))
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000))
    scores = cross_val_score(
        model, values, classes,
        cv=StratifiedKFold(n_splits=folds, shuffle=True,
                           random_state=int(seed)))
    return {
        "accuracy": float(np.mean(scores)),
        "accuracy_sd": float(np.std(scores)),
        "chance": float(counts.max() / len(pairs)),
        "n": float(len(pairs)),
        "classes": float(len(counts)),
        "folds": float(folds),
    }
