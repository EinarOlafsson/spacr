"""Optional segmentation backends behind the Cellpose mask path.

NOT AN API PAGE. tools/build_documentation_i18n.py extracts the module
docstring of every file under spacr/, private modules included, and pins it
by hash in nine translated API catalogs -- unless the module is listed in its
AUTOAPI_NON_RENDERED_MODULES, and this one is (2026-09-15). AutoAPI renders
no leading-underscore module, and everything below is underscore-private, so
this docstring carries the design without adding a page that would need
translating.

Items 404 (DINOCell) and 405 (SAMCell), built as ONE seam rather than two.
`generate_cellpose_masks_sam` builds a model and calls
`model.eval(x=batch_list, ...)`; every backend here answers that same call
with the `(masks, flows, styles)` triple Cellpose 4 returns, so the lines
after it -- `parse_cellpose4_output`, merge/split/filter, tracking, the
object-count database, saving and segmentation QC -- run unchanged. The only
dispatch is at model construction, and segmentation_backend='cellpose' (or
the key absent) never reaches this module's loaders at all.

HOW EACH BACKEND BECOMES A MASK

* DINOCell (`pip install "spacr[dinocell]"`) predicts Cellpose-style flows:
  (dx, dy, cell probability). Masks come from
  `cellpose.dynamics.compute_masks`, called with the constants DINOCell's
  own `DINOFlowsSlidingWindowPipeline.run` uses (250 iterations, flow-error
  check off, minimum size 15, maximum size fraction 0.4). Its probability is
  a sigmoid output, so spaCR's `<object>_cellprob_threshold` -- a Cellpose
  logit -- is applied through the logistic function: the default 0 is
  DINOCell's own 0.5.
* SAMCell (`pip install "spacr[samcell]"`) is SAM ViT-B fine-tuned to
  predict a cell distance map. Masks come from SAMCell's own
  `SlidingWindowPipeline.cells_from_dist_map` (contour centroids as seeds,
  watershed inside the fill threshold), with its default thresholds.

Both are single-channel 2-D models. Each reads the object's own channel (the
first in the batch, as `_get_cellpose_channels` orders them), stretched to
8 bits because both packages quantise their input to uint8 before CLAHE.
z-stack and t-stack runs are refused rather than flattened.

Nothing here imports torch, cellpose, transformers or either package at
module scope (item 282); tests/test_perf_guard.py holds the launch path to
that.
"""
from __future__ import annotations

import math
import os

import numpy as np

_CELLPOSE = "cellpose"
_DINOCELL = "dinocell"
_SAMCELL = "samcell"

#: Every value ``segmentation_backend`` accepts, the default first.
_BACKEND_NAMES = (_CELLPOSE, _DINOCELL, _SAMCELL)

#: DINOCell's inference constants, copied from ``dinocell.main.segment`` and
#: ``DINOFlowsSlidingWindowPipeline.run`` (dinocell 0.74).
_DINOCELL_CROP = 512
_DINOCELL_OVERLAP = _DINOCELL_CROP // 4
_DINOCELL_NITER = 250
_DINOCELL_FLOW_THRESHOLD = 0
_DINOCELL_MIN_SIZE = 15
_DINOCELL_MAX_SIZE_FRACTION = 0.4

#: SAMCell's base model, crop and published checkpoints (samcell 1.2.0 README).
_SAMCELL_BASE_MODEL = "facebook/sam-vit-base"
_SAMCELL_CROP = 256
_SAMCELL_RELEASE = (
    "https://github.com/saahilsanganeriya/SAMCell/releases/download/v1/")
_SAMCELL_WEIGHTS = {
    "generalist": "samcell-generalist.pt",   # LIVECell + Cellpose cytoplasm
    "cyto": "samcell-cyto.pt",               # Cellpose cytoplasm only
}


def _backend_name(value):
    """Canonical backend name for a ``segmentation_backend`` value.

    :param value: the setting's value; ``None`` or blank means Cellpose.
    :returns: one of :data:`_BACKEND_NAMES`.
    :raises ValueError: for a name spaCR has no backend for.
    """
    if value is None:
        return _CELLPOSE
    name = str(value).strip().lower()
    if not name:
        return _CELLPOSE
    if name not in _BACKEND_NAMES:
        choices = ", ".join(repr(n) for n in _BACKEND_NAMES)
        raise ValueError(
            f"segmentation_backend={value!r} is not a segmentation backend "
            f"spaCR has. Choose one of {choices}.")
    return name


def _import_dinocell():
    """Import DINOCell's model, pipeline factory and weight resolver.

    :returns: ``(DINOCell, get_pipeline, get_weights_path)``.
    :raises ImportError: naming the extra that installs it.
    """
    try:
        from dinocell.main import get_weights_path
        from dinocell.model import DINOCell
        from dinocell.pipeline import get_pipeline
    except (ImportError, OSError) as exc:
        raise ImportError(
            "DINOCell segmentation requires the optional dinocell package. "
            "Install it with `pip install \"spacr[dinocell]\"`, or segment "
            "with segmentation_backend='cellpose'. NOTE: dinocell pins exact "
            "versions of its own dependencies (torch and cellpose among them), "
            "so pip may want to replace packages spaCR already has; if it "
            "reports a conflict, install it into a separate environment. "
            f"Cellpose segmentation is unaffected. The import failed with: "
            f"{exc}"
        ) from exc
    return DINOCell, get_pipeline, get_weights_path


def _import_samcell():
    """Import SAMCell's model wrapper and sliding-window pipeline.

    :returns: ``(FinetunedSAM, SlidingWindowPipeline)``.
    :raises ImportError: naming the extra that installs it.
    """
    try:
        from samcell.model import FinetunedSAM
        from samcell.pipeline import SlidingWindowPipeline
    except (ImportError, OSError) as exc:
        raise ImportError(
            "SAMCell segmentation requires the optional samcell package. "
            "Install it with `pip install \"spacr[samcell]\"`, or segment "
            "with segmentation_backend='cellpose'. Cellpose segmentation is "
            f"unaffected. The import failed with: {exc}"
        ) from exc
    return FinetunedSAM, SlidingWindowPipeline


def _object_plane(image, channel_axis=-1):
    """The object's own channel of one batch image, as a 2-D array.

    :param image: ``(H, W)`` or ``(H, W, C)`` array.
    :param channel_axis: the axis ``model.eval`` was told holds channels.
    :returns: ``(H, W)`` array.
    :raises ValueError: for anything that is not one 2-D plane.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(
            f"segmentation backends other than Cellpose take 2-D images; got "
            f"an array of shape {arr.shape}")
    axis = -1 if channel_axis is None else channel_axis
    return np.take(arr, 0, axis=axis)


def _to_uint8(plane):
    """Min-max stretch a plane into ``uint8``; a flat plane becomes zeros.

    :param plane: 2-D numeric array; non-finite pixels take the minimum.
    :returns: ``uint8`` array of the same shape.
    """
    arr = np.asarray(plane, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, np.uint8)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8)
    scaled = (np.where(finite, arr, lo) - lo) / (hi - lo)
    return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)


def _tile_starts(length, crop, overlap):
    """Distinct tile origins along one axis, as DINOCell's sliding window.

    ``SlidingWindowHelper.seperate_into_crops_v2`` steps by
    ``crop - 2 * overlap`` and clamps the last tile to the edge, which repeats
    that tile whenever the clamp lands on an earlier origin. Averaging a
    repeat changes nothing, so running it once gives the same prediction
    for less compute.

    :returns: sorted list of origins.
    """
    if length <= crop:
        return [0]
    stride = max(1, crop - 2 * overlap)
    return sorted({min(start, length - crop)
                   for start in range(0, length, stride)})


def _resize_nearest(array, shape):
    """Nearest-neighbour resample of the last two axes to ``shape``.

    :param array: ``(..., h, w)`` array (labels stay labels).
    :param shape: target ``(H, W)``.
    :returns: ``(..., H, W)`` array of the same dtype.
    """
    arr = np.asarray(array)
    h, w = arr.shape[-2:]
    height, width = int(shape[0]), int(shape[1])
    if (h, w) == (height, width):
        return arr
    rows = np.minimum(((np.arange(height) + 0.5) * h / height).astype(np.intp),
                      h - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * w / width).astype(np.intp),
                      w - 1)
    return arr[..., rows[:, None], cols[None, :]]


def _as_label_image(labels):
    """Sequential labels in the dtype Cellpose returns (``uint16``, or
    ``uint32`` past 65,535 objects).

    :param labels: 2-D integer label image, background 0.
    :returns: relabelled array.
    """
    arr = np.asarray(labels)
    if arr.size and arr.max() > 0:
        from skimage.segmentation import relabel_sequential

        arr = relabel_sequential(arr.astype(np.int64, copy=False))[0]
    dtype = np.uint16 if arr.max(initial=0) < 2 ** 16 else np.uint32
    return arr.astype(dtype, copy=False)


def _probability_threshold(cellprob_threshold):
    """A Cellpose cell-probability logit threshold as a probability.

    :param cellprob_threshold: Cellpose's threshold, or ``None`` for 0.
    :returns: ``1 / (1 + exp(-threshold))``; 0 maps to 0.5.
    """
    if cellprob_threshold is None:
        return 0.5
    value = min(50.0, max(-50.0, float(cellprob_threshold)))
    return 1.0 / (1.0 + math.exp(-value))


class _PlaneBackend:
    """A single-channel 2-D segmenter that answers ``CellposeModel.eval``.

    Subclasses implement :meth:`_segment_plane`; this class turns a batch into
    the ``(masks, flows, styles)`` triple ``parse_cellpose4_output`` reads.
    """

    name = "plane"
    #: What the backend does with the Cellpose settings the eval call passes.
    note = ""

    def __init__(self, device=None):
        """Record the device, resolving spaCR's accelerator when None."""
        if device is None:
            from . import accelerator

            device = accelerator.torch_device()
        self.device = device

    def eval(self, x, batch_size=None, channel_axis=-1,
             cellprob_threshold=0.0, flow_threshold=None, progress=None,
             **cellpose_only):
        """Segment each image of a batch.

        :param x: list of ``(H, W, C)`` (or ``(H, W)``) images.
        :param channel_axis: axis holding channels; the first channel is used.
        :param cellprob_threshold: Cellpose logit threshold, used by backends
            that predict a cell probability.
        :param cellpose_only: diameter, min_size, resample and the other
            Cellpose arguments; accepted so the call site is unchanged.
        :returns: ``(masks, flows, None)`` with one entry per image.
        """
        images = ([x] if isinstance(x, np.ndarray) and x.ndim == 2
                  else list(x))
        masks, flows = [], []
        for image in images:
            plane = _object_plane(image, channel_axis)
            labels, flow = self._segment_plane(
                _to_uint8(plane), cellprob_threshold=cellprob_threshold)
            labels = np.asarray(labels)
            if labels.shape != plane.shape:
                raise ValueError(
                    f"the {self.name} backend returned labels of shape "
                    f"{labels.shape} for an image of shape {plane.shape}")
            masks.append(_as_label_image(labels))
            flows.append(flow)
        return masks, flows, None

    def _segment_plane(self, image, cellprob_threshold=None):
        """Label one ``uint8`` plane.

        :returns: ``(labels, flow)`` where ``flow`` is the per-image list
            ``[display image, dP or None, probability or None, None]``.
        """
        raise NotImplementedError


class _DinoCellBackend(_PlaneBackend):
    """DINOCell: DINOv2 ViT-B predicting Cellpose-style flows."""

    name = _DINOCELL
    note = ("cellprob_threshold is applied to DINOCell's cell probability "
            "through the logistic function (0 -> 0.5); diameter, "
            "flow_threshold, min_size and resample are Cellpose settings and "
            "are not used")

    def __init__(self, device=None, weights_path=None):
        """Load DINOCell's checkpoint (from the Hugging Face cache unless
        ``weights_path`` names one) and build its flows pipeline."""
        super().__init__(device)
        DINOCell, get_pipeline, get_weights_path = _import_dinocell()
        import torch

        self.device = torch.device(self.device)
        weights = weights_path or get_weights_path()
        model = DINOCell(
            dino_model=None, decoder_type="upsample", objective_type="flows",
            use_dino_weights=False, patch_size=8, feat_size=64,
            crop_size=_DINOCELL_CROP, drop_rate=0.05, dropout_in_encoder=True,
            finetune_vision=True, finetune_decoder=True,
            finetune_prediction_head=True)
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        model.to(self.device).eval()
        self._pipeline = get_pipeline(
            "flows", model, self.device, crop_size=_DINOCELL_CROP,
            use_advanced_augmentations=False, overlap_size=_DINOCELL_OVERLAP)

    def _predict(self, image):
        """``(dx, dy, cell probability)`` for a plane at least one crop wide.

        :param image: ``uint8`` plane with both sides >= the crop.
        :returns: ``(3, H, W)`` float32 array.
        """
        crop = _DINOCELL_CROP
        height, width = image.shape
        total = np.zeros((3, height, width), np.float32)
        count = np.zeros((height, width), np.float32)
        for y in _tile_starts(height, crop, _DINOCELL_OVERLAP):
            for x in _tile_starts(width, crop, _DINOCELL_OVERLAP):
                tile = np.ascontiguousarray(image[y:y + crop, x:x + crop])
                preds = self._pipeline.get_model_prediction(tile)
                for channel, pred in enumerate(preds[:3]):
                    total[channel, y:y + crop, x:x + crop] += (
                        pred[0].detach().float().cpu().numpy())
                count[y:y + crop, x:x + crop] += 1.0
        return total / count

    def _segment_plane(self, image, cellprob_threshold=None):
        """Predict flows, then label them with Cellpose's dynamics.

        A plane narrower than one tile is upscaled (aspect ratio kept) and the
        labels are resampled back to the plane's own shape.
        """
        import cv2
        from cellpose.dynamics import compute_masks
        from cellpose.plot import dx_to_circ

        height, width = image.shape
        scale = _DINOCELL_CROP / min(height, width)
        if scale > 1:
            # DINOCell's own `_resize` upsamples small images to the crop too,
            # but to a square; this keeps the aspect ratio.
            size = (max(_DINOCELL_CROP, math.ceil(width * scale)),
                    max(_DINOCELL_CROP, math.ceil(height * scale)))
            work = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        else:
            work = image
        dx, dy, probability = self._predict(work)
        d_p = np.stack([dy, dx])
        labels = compute_masks(
            dP=d_p, cellprob=probability, niter=_DINOCELL_NITER,
            cellprob_threshold=_probability_threshold(cellprob_threshold),
            flow_threshold=_DINOCELL_FLOW_THRESHOLD, do_3D=False,
            min_size=_DINOCELL_MIN_SIZE,
            max_size_fraction=_DINOCELL_MAX_SIZE_FRACTION,
            device=self.device)
        shape = (height, width)
        display = np.moveaxis(
            _resize_nearest(np.moveaxis(dx_to_circ(d_p), -1, 0), shape), 0, -1)
        flow = [display, _resize_nearest(d_p, shape),
                _resize_nearest(probability, shape), None]
        return _resize_nearest(labels, shape), flow


def _samcell_weights_path(variant="generalist", download=False):
    """Where a SAMCell checkpoint lives in torch's hub cache.

    :param variant: ``'generalist'`` or ``'cyto'``.
    :param download: fetch the GitHub release asset when it is not cached.
    :returns: the local path (which may not exist when ``download`` is False).
    """
    import torch

    filename = _SAMCELL_WEIGHTS[variant]
    path = os.path.join(torch.hub.get_dir(), "checkpoints", filename)
    if download and not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = _SAMCELL_RELEASE + filename
        print(f"Downloading SAMCell weights {url} -> {path}")
        torch.hub.download_url_to_file(url, path, progress=True)
    return path


class _SamCellBackend(_PlaneBackend):
    """SAMCell: SAM ViT-B fine-tuned to predict a cell distance map."""

    name = _SAMCELL
    note = ("SAMCell's own peak (0.47) and fill (0.09) thresholds apply; "
            "cellprob_threshold, flow_threshold, diameter, min_size and "
            "resample are Cellpose settings and are not used")

    def __init__(self, device=None, weights_path=None, variant="generalist"):
        """Load SAM ViT-B, apply a SAMCell checkpoint (downloaded once into
        torch's hub cache unless ``weights_path`` names one) and build the
        sliding-window pipeline."""
        super().__init__(device)
        FinetunedSAM, SlidingWindowPipeline = _import_samcell()
        import torch

        self.device = torch.device(self.device)
        weights = weights_path or _samcell_weights_path(variant, download=True)
        model = FinetunedSAM(_SAMCELL_BASE_MODEL)
        model.load_weights(weights, map_location=self.device)
        self._pipeline = SlidingWindowPipeline(
            model, self.device, crop_size=_SAMCELL_CROP)

    def _segment_plane(self, image, cellprob_threshold=None):
        """Predict SAMCell's distance map and label it with its own watershed."""
        # `predict_on_full_img` raises on failure. `run` would not: it logs
        # and returns an all-zero label image, which here would be saved as
        # a field with no cells.
        dist_map = self._pipeline.predict_on_full_img(image)
        labels = self._pipeline.cells_from_dist_map(dist_map)
        return labels, [dist_map, None, dist_map, None]


#: Backend name -> class. Tests replace entries with stubs.
_BACKEND_CLASSES = {_DINOCELL: _DinoCellBackend, _SAMCELL: _SamCellBackend}


def _load_backend(name, *, device=None, z_plan=None, t_plan=None, **options):
    """Build the model object ``generate_cellpose_masks_sam`` calls ``eval`` on.

    :param name: a non-Cellpose value of ``segmentation_backend``.
    :param device: torch device; the resolved accelerator when None.
    :param z_plan: the run's z-stack plan; must be None.
    :param t_plan: the run's t-stack plan; must be None.
    :param options: passed to the backend (``weights_path``, ``variant``).
    :returns: a :class:`_PlaneBackend`.
    :raises ValueError: for Cellpose, an unknown name, or a 3-D/4-D run.
    :raises ImportError: when the backend's package is not installed.
    """
    backend = _backend_name(name)
    if backend == _CELLPOSE:
        raise ValueError(
            "segmentation_backend='cellpose' is built by the mask generator "
            "itself, not by _load_backend")
    if z_plan is not None or t_plan is not None:
        raise ValueError(
            f"segmentation_backend={backend!r} segments single 2-D planes, "
            f"and this run has z_stack or t_stack on. Use "
            f"segmentation_backend='cellpose' for 3-D and 4-D runs, or turn "
            f"z_stack and t_stack off.")
    cls = _BACKEND_CLASSES[backend]
    model = cls(device=device, **options)
    note = getattr(model, "note", "")
    print(f"Segmentation backend: {backend}"
          + (f" -- {note}." if note else "."))
    return model
