"""Versioned crop decoding shared by classification and its checkpoints."""

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from PIL import Image, ImageOps


DECLARED_UINT8 = "declared_uint8_v1"
STORED_PIL = "stored_pil_v1"
RAW_FIXED_SCALE = "raw_fixed_scale_v1"

_RAW_FORMAT = 4


def validate_policy(policy):
    """Accept only implemented crop decoding policies; never guess a new one.

    :param policy: ``declared_uint8_v1``, ``stored_pil_v1`` or
        ``raw_fixed_scale_v1``. The last reads raw crops at their stored
        values for a fixed per-plate scale in the model's input transform.
    :returns: the unchanged, validated policy name.
    :raises ValueError: the supplied policy is not implemented.
    """
    if policy not in (DECLARED_UINT8, STORED_PIL, RAW_FIXED_SCALE):
        raise ValueError(f"Unsupported classification crop loading policy: {policy!r}")
    return policy


def checkpoint_policy(metadata, *, announce=False):
    """Read a checkpoint's decoder contract, preserving historical untagged models.

    :param metadata: artifact mapping returned by ``load_model_artifact``.
    :param announce: explain the legacy fallback when the contract is absent.
    :returns: validated policy name. Missing metadata selects stored PIL RGB.
    """
    preprocessing = (metadata.get("preprocessing") or {}) if isinstance(metadata, Mapping) else {}
    policy = preprocessing.get("crop_loading_policy")
    if policy is None:
        policy = STORED_PIL
        if announce:
            print("Model has no crop decoding record: retaining historical stored "
                  "channel order and PIL RGB conversion. High-bit-depth crops "
                  "can clip under this legacy policy; retrain with declared "
                  "uint8 decoding to change it safely.")
    return validate_policy(policy)


def initialization_policy(path):
    """Select decoding before training loaders are built for a resumed model.

    :param path: trusted checkpoint path, or None for a new training run.
    :returns: saved policy, legacy fallback, or declared uint8 for new training.
    """
    if not path:
        return DECLARED_UINT8
    import torch

    metadata = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint_policy(metadata, announce=True)


def read_classification_image(source, policy=DECLARED_UINT8, *, fmt=None,
                              legacy_orient=False):
    """Decode a path or archive stream using the model's recorded policy.

    :param source: image path or seekable binary stream, as accepted by PIL.
    :param policy: declared uint8 decoding or historical stored PIL RGB.
    :param fmt: source format; None resolves the path's per-file crop marker.
        Archive callers must supply a format (1 when unmarked).
    :param legacy_orient: retain the labelled loader's historical EXIF handling.
    :returns: independently owned PIL RGB image with its file handle closed,
        or a float32 ``(H, W, 3)`` array of stored values under
        ``raw_fixed_scale_v1``. A raw crop, format 4, is returned with the old per-crop stretch under
        both policies, so a model sees the pixels it was trained on.
    """
    validate_policy(policy)
    if policy == RAW_FIXED_SCALE:
        return _read_raw_classification_array(source, fmt)
    if policy == STORED_PIL:
        if fmt is None and not hasattr(source, "read"):
            from .crops import crop_format_for_png

            if crop_format_for_png(source) == _RAW_FORMAT:
                fmt = _RAW_FORMAT
        if fmt is not None and int(fmt) == _RAW_FORMAT:
            from .crops import _read_crop_array, _stretched_view

            return Image.fromarray(_stretched_view(_read_crop_array(source)))
        with Image.open(source) as image:
            if legacy_orient:
                image = ImageOps.exif_transpose(image)
            return image.convert("RGB").copy()
    from .crops import crop_format_for_png, decode_crop_image

    if fmt is None:
        fmt = crop_format_for_png(source)
        from .crops import read_crop_folder_marker

        parent = Path(source).absolute().parent
        if (read_crop_folder_marker(parent) is None
                and parent.parent.name in ('train', 'test')):
            marker = read_crop_folder_marker(parent.parent.parent)
            if marker is not None and marker.get('split') == 'train/test':
                fmt = marker['spacr_crop_format']
    if int(fmt) == _RAW_FORMAT:
        from .crops import _read_crop_array, _stretched_view

        return Image.fromarray(_stretched_view(_read_crop_array(source)))
    with Image.open(source) as image:
        return Image.fromarray(decode_crop_image(image, fmt=fmt, orient=True))


def loader_policy(loader):
    """Find the decoding record through loaders, subsets and combined datasets.

    Unknown external tensor datasets return None. Mixed recorded policies are
    refused because a single checkpoint cannot describe both transformations.

    :param loader: loader or dataset exposing ``crop_loading_policy``,
        ``dataset`` or ``datasets``.
    :returns: a validated policy name, or None if no policy is recorded.
    :raises ValueError: a policy is unsupported or combined datasets disagree,
        including a mixture of recorded and unknown policies.
    """
    policy = getattr(loader, "crop_loading_policy", None)
    if policy is not None:
        return validate_policy(policy)
    if hasattr(loader, "dataset"):
        return loader_policy(loader.dataset)
    if hasattr(loader, "datasets"):
        policies = {loader_policy(dataset) for dataset in loader.datasets}
        if len(policies) > 1:
            raise ValueError("Classification datasets use different crop loading policies.")
        return policies.pop() if policies else None
    return None


def training_preprocessing(train_loader, validation_loader, preprocessing=None,
                           checkpoint=None):
    """Record actual loader decoding and reject checkpoint or validation conflicts.

    :param train_loader: training loader or dataset whose decoding is inspected.
    :param validation_loader: validation loader or dataset checked for agreement.
    :param preprocessing: optional preprocessing mapping copied into the result;
        an explicit policy must agree with a recorded training policy.
    :param checkpoint: optional loaded checkpoint metadata checked against the
        selected policy; an untagged checkpoint uses the historical policy.
    :returns: a new preprocessing dictionary, including the policy when known.
        Unknown external datasets are not assigned a decoder automatically.
    :raises ValueError: a supplied policy is unsupported or the recorded
        training, validation, preprocessing or checkpoint policies conflict.
    """
    result = dict(preprocessing or {})
    policy = loader_policy(train_loader)
    requested = result.get("crop_loading_policy")
    if requested is not None:
        validate_policy(requested)
        if policy is not None and requested != policy:
            raise ValueError("Recorded crop loading policy disagrees with training images.")
        policy = requested
    validation = loader_policy(validation_loader)
    if policy is not None and validation is not None and policy != validation:
        raise ValueError("Training and validation crop loading policies must match.")
    if checkpoint is not None and policy is not None and checkpoint_policy(checkpoint) != policy:
        raise ValueError("Checkpoint and training crop loading policies differ. Rebuild "
                         "the loaders with the checkpoint's crop_loading_policy.")
    if policy is not None:
        result["crop_loading_policy"] = policy
    contract = getattr(train_loader, "preprocessing_contract", None)
    if contract:
        result.update(contract)
    return result


_CONTRACT_VERSION = 1

_SCALE_PERCENTILE = 99.0

_SCALE_SAMPLE = 2048


def _read_raw_classification_array(source, fmt=None):
    """Decode a raw crop as float32 ``(H, W, C)`` values, as stored.

    :param source: crop path, or a binary stream with ``fmt`` supplied.
    :param fmt: crop format; None resolves the path's folder marker.
    :returns: float32 array in declared channel order.
    :raises ValueError: the crop is not a raw crop (format 4). A stretched
        crop has lost its intensity scale, so a fixed scale cannot be
        applied to it.
    """
    from .crops import _read_crop_array, crop_format_for_png

    if fmt is None:
        fmt = crop_format_for_png(source)
    if int(fmt) != _RAW_FORMAT:
        raise ValueError(
            f"Fixed per-plate scaling needs raw crops (crop format 4), but "
            f"{source!r} is crop format {fmt}. Re-run Measure with "
            f"crop_intensity='raw', or set crop_scaling to 'legacy'.")
    arr = _read_crop_array(source).astype(np.float32)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def _plate_of(filename):
    """Return the plate token of a crop file name, the text before the first ``_``."""
    return os.path.basename(str(filename)).split('_')[0]


def _estimate_crop_scales(filenames, *, percentile=_SCALE_PERCENTILE,
                          max_crops=_SCALE_SAMPLE, seed=0):
    """Fixed per-plate, per-channel scales from the object pixels of raw crops.

    For each plate a seeded sample of at most ``max_crops`` crops is read,
    and the scale of a channel is the ``percentile`` of its nonzero pixels.
    Pixels outside the object are 0 in a masked crop, so the statistic is
    taken over cell pixels. The scale never depends on a batch or on the
    crop being scaled.

    :param filenames: raw crop paths.
    :param percentile: percentile of the object pixels used as the scale.
    :param max_crops: crops sampled per plate.
    :param seed: sampling seed.
    :returns: ``{plate: [scale per channel]}``. A channel with no signal
        gets 1.0, so it stays 0 after scaling instead of dividing by 0.
    """
    by_plate = {}
    for name in filenames:
        by_plate.setdefault(_plate_of(name), []).append(name)
    rng = np.random.default_rng(int(seed))
    scales = {}
    for plate in sorted(by_plate):
        names = sorted(by_plate[plate])
        if len(names) > max_crops:
            names = [names[i] for i in sorted(rng.choice(len(names), max_crops, replace=False))]
        pixels = None
        for name in names:
            arr = _read_raw_classification_array(name)
            flat = arr.reshape(-1, arr.shape[2])
            keep = flat[np.any(flat > 0, axis=1)]
            pixels = keep if pixels is None else np.concatenate([pixels, keep])
        values = []
        for c in range(pixels.shape[1] if pixels is not None else 3):
            column = pixels[:, c] if pixels is not None else np.zeros(0)
            column = column[column > 0]
            top = float(np.percentile(column, percentile)) if column.size else 0.0
            values.append(top if top > 0 else 1.0)
        scales[plate] = values
    return scales


def _write_crop_scales(path, scales, *, percentile=_SCALE_PERCENTILE, clip=1.5):
    """Write a scales file and return its md5.

    :param path: the JSON file to write.
    :param scales: ``{plate: [scale per channel]}``.
    :param percentile: the percentile the scales were taken at.
    :param clip: the clip applied after scaling.
    :returns: the md5 hex digest of the written file.
    """
    payload = {"version": 1, "percentile": float(percentile),
               "clip": float(clip), "scales": scales}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _read_crop_scales(path):
    """Read a scales file.

    :param path: a file written by :func:`_write_crop_scales`.
    :returns: ``(scales, md5)``.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    payload = json.loads(raw.decode("utf-8"))
    return dict(payload["scales"]), hashlib.md5(raw).hexdigest()


def _weights_input_statistics(model_type):
    """Mean, std and interpolation of a torchvision model's default weights.

    :param model_type: torchvision model name.
    :returns: ``(mean, std, interpolation)``, or None when torchvision has
        no weights for that name.
    """
    try:
        from torchvision import models
    except Exception:
        return None
    wanted = f"{model_type}_weights".lower()
    for attr in dir(models):
        if attr.lower() == wanted:
            enum = getattr(models, attr)
            default = getattr(enum, "DEFAULT", None)
            if default is None:
                return None
            preset = default.transforms()
            interpolation = getattr(preset, "interpolation", None)
            interpolation = getattr(interpolation, "value", "bilinear") or "bilinear"
            return (tuple(float(v) for v in preset.mean),
                    tuple(float(v) for v in preset.std), str(interpolation))
    return None


class _FixedScaleInput:
    """Batch-wise input transform for raw crops, run on the model's device.

    The steps are a fixed per-plate, per-channel division, a clip, optional
    intensity-aware augmentation, a resize to the model input and the
    recorded mean and standard deviation. Every number comes from the
    contract, so training and inference apply the same transform.

    :param contract: the preprocessing mapping recorded with the model.
    """

    def __init__(self, contract):
        """Store the contract values the transform needs."""
        self.contract = dict(contract)
        self.scales = {str(k): [float(x) for x in v]
                       for k, v in (contract.get("scales") or {}).items()}
        self.clip = float(contract.get("clip", 1.5))
        self.image_size = int(contract.get("image_size", 224))
        self.interpolation = str(contract.get("interpolation", "bilinear"))
        self.mean = contract.get("mean")
        self.std = contract.get("std")
        self.channels = list(contract.get("selected_channels") or [1, 2, 3])
        self.augment = bool(contract.get("augment", False))
        self.shift_px = int(contract.get("augment_shift_px", 4))
        self.gain = float(contract.get("augment_gain", 0.1))

    def add_plates(self, filenames):
        """Estimate scales for plates the contract does not know.

        A new plate gets its own fixed scale from its crops, never from a
        batch. The step is printed, because the scale is then not the
        training plate's.

        :param filenames: crop paths about to be scored.
        :returns: the plates that were added.
        """
        missing = sorted({_plate_of(f) for f in filenames} - set(self.scales))
        if missing:
            wanted = [f for f in filenames if _plate_of(f) in missing]
            self.scales.update(_estimate_crop_scales(wanted))
            print(f"Estimated fixed crop scales for {len(missing)} plate(s) the "
                  f"model was not trained on: {', '.join(missing)}.")
        return missing

    def __call__(self, images, filenames, train=False):
        """Transform one batch.

        :param images: ``(B, C, H, W)`` raw float tensor on any device.
        :param filenames: one crop path per image, for the plate lookup.
        :param train: apply the augmentation as well.
        :returns: the model-ready batch on the same device.
        """
        import torch
        import torch.nn.functional as F

        x = images.float()
        rows = []
        for name in filenames:
            plate = _plate_of(name)
            if plate not in self.scales:
                raise ValueError(
                    f"No fixed crop scale for plate {plate!r}. Call "
                    f"add_plates with the crops before scoring them.")
            rows.append(self.scales[plate][:x.shape[1]])
        scale = torch.tensor(rows, dtype=x.dtype, device=x.device)
        x = (x / scale[:, :, None, None]).clamp_(0.0, self.clip)
        for idx in range(x.shape[1]):
            if idx + 1 not in self.channels:
                x[:, idx] = 0
        if train and self.augment:
            x = self._augment(x)
        if tuple(x.shape[-2:]) != (self.image_size, self.image_size):
            mode = self.interpolation if self.interpolation in ("bilinear", "bicubic", "nearest") else "bilinear"
            kwargs = {} if mode == "nearest" else {"align_corners": False}
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode=mode, **kwargs)
        if self.mean is not None and self.std is not None:
            mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)[None, :, None, None]
            std = torch.tensor(self.std, dtype=x.dtype, device=x.device)[None, :, None, None]
            x = (x - mean[:, :x.shape[1]]) / std[:, :x.shape[1]]
        return x

    def _augment(self, x):
        """Dihedral flips and rotations, a small shift and a small per-channel gain.

        Intensity is the phenotype, so the gain stays within
        ``1 +- augment_gain`` and nothing else changes intensity.

        :param x: scaled batch.
        :returns: augmented batch.
        """
        import torch

        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            img = torch.rot90(x[i], int(torch.randint(0, 4, (1,))), dims=(1, 2))
            if bool(torch.randint(0, 2, (1,))):
                img = torch.flip(img, dims=(2,))
            if self.shift_px:
                dy, dx = (int(v) for v in torch.randint(-self.shift_px, self.shift_px + 1, (2,)))
                img = torch.roll(img, shifts=(dy, dx), dims=(1, 2))
            out[i] = img
        if self.gain:
            g = 1.0 + (torch.rand(x.shape[0], x.shape[1], 1, 1, device=x.device) * 2 - 1) * self.gain
            out = out * g
        return out


def _prepare_batch(loader, data, filenames, device, train=False):
    """Move a batch to ``device`` and apply the loader's batch transform, if any.

    :param loader: the loader the batch came from.
    :param data: the batch tensor.
    :param filenames: the batch's file names.
    :param device: the model's device.
    :param train: apply training augmentation.
    :returns: the batch, ready for the model.
    """
    data = data.to(device)
    transform = getattr(loader, "input_transform", None)
    if transform is None:
        return data
    return transform(data, list(filenames), train=train)


def _contract_mismatches(saved, observed):
    """Compare two preprocessing mappings key by key.

    Only keys present in both are compared, so a legacy checkpoint is not
    refused for lacking a key.

    :param saved: the checkpoint's preprocessing.
    :param observed: what the current run would apply.
    :returns: ``[(key, saved_value, observed_value), ...]``.
    """
    keys = ("crop_loading_policy", "image_size", "resize", "interpolation",
            "mean", "std", "selected_channels", "clip", "scales_md5",
            "flat_field_sha256", "mask_mode", "crop_size")
    out = []
    for key in keys:
        a, b = saved.get(key), observed.get(key)
        if a is None or b is None:
            continue
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            same = len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
        else:
            same = _same_value(a, b)
        if not same:
            out.append((key, a, b))
    return out


def _same_value(a, b):
    """Compare two recorded values, numbers to within 1e-9.

    :param a: one value.
    :param b: the other.
    :returns: whether they agree.
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def _check_preprocessing(metadata, observed, *, on_mismatch="refuse"):
    """Refuse or warn when inference would transform crops unlike training.

    :param metadata: the checkpoint metadata from ``load_model_artifact``.
    :param observed: the preprocessing this run would apply.
    :param on_mismatch: ``'refuse'`` raises, ``'warn'`` prints and goes on.
    :returns: the list of mismatches, empty when everything agrees.
    :raises ValueError: a mismatch and ``on_mismatch='refuse'``.
    """
    saved = (metadata.get("preprocessing") or {}) if isinstance(metadata, Mapping) else {}
    if saved.get("contract_version") is None:
        print("This model carries no full preprocessing record. It was saved "
              "before November 2026, so its crops are decoded and normalised "
              "with the settings of this run, as before. Check that image_size, "
              "normalize and train_channels match the ones it was trained with.")
        return []
    mismatches = _contract_mismatches(saved, observed)
    if not mismatches:
        return []
    lines = "; ".join(f"{k}: trained with {a!r}, this run {b!r}" for k, a, b in mismatches)
    message = ("The crops would be transformed differently from training (" + lines
               + "). Rebuild the inputs with the model's recorded preprocessing.")
    if str(on_mismatch).lower() == "warn":
        print("WARNING: " + message)
        return mismatches
    raise ValueError(message + " Set preprocessing_mismatch='warn' to run anyway.")
