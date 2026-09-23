"""Versioned crop decoding shared by classification and its checkpoints."""

from collections.abc import Mapping
from pathlib import Path

from PIL import Image, ImageOps


DECLARED_UINT8 = "declared_uint8_v1"
STORED_PIL = "stored_pil_v1"


def validate_policy(policy):
    """Accept only implemented crop decoding policies; never guess a new one."""
    if policy not in (DECLARED_UINT8, STORED_PIL):
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
    :returns: independently owned PIL RGB image with its file handle closed.
    """
    validate_policy(policy)
    if policy == STORED_PIL:
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
    with Image.open(source) as image:
        return Image.fromarray(decode_crop_image(image, fmt=fmt, orient=True))


def loader_policy(loader):
    """Find the decoding record through loaders, subsets and combined datasets.

    Unknown external tensor datasets return None. Mixed recorded policies are
    refused because a single checkpoint cannot describe both transformations.
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
    """Record actual loader decoding and reject checkpoint or validation conflicts."""
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
    return result
