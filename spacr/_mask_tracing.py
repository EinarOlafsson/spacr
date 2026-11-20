"""Overlap public Cellpose tracing calls within one resident image batch."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from importlib import metadata

import numpy as np


@lru_cache(maxsize=1)
def _public_trace_api():
    """Return the public API whose split-call contract has been verified.

    Cellpose 4.0.9 returns resampled flows and derives the tracing iteration
    count from the diameter. Other releases retain their combined eval call
    until their public return and scaling contracts have been checked.

    :returns: model type, original eval and public tracing function, or None.
    """
    try:
        if metadata.version('cellpose') != '4.0.9':
            return None
        from cellpose import dynamics, models
    except (ImportError, metadata.PackageNotFoundError):
        return None
    return (models.CellposeModel, models.CellposeModel.eval,
            dynamics.resize_and_compute_masks)


def _can_overlap(model, images, keywords):
    """Whether this call has the verified independent two-dimensional contract.

    :param model: the in-process segmentation model.
    :param images: the complete resident batch supplied to eval.
    :param keywords: the unchanged public eval options.
    :returns: True only for supported CUDA models and flat image lists.
    """
    if (getattr(getattr(model, 'device', None), 'type', None) != 'cuda'
            or not isinstance(images, list) or len(images) < 2
            or keywords.get('normalize', True) is not False
            or keywords.get('channel_axis') != -1
            or keywords.get('do_3D', False)
            or keywords.get('stitch_threshold', 0) != 0
            or keywords.get('z_axis') is not None
            or not keywords.get('compute_masks', True)
            or keywords.get('channels') is not None):
        return False
    if any(not isinstance(image, np.ndarray) or image.ndim != 3
           or min(image.shape[:2]) < 2 or image.shape[-1] < 1
           for image in images):
        return False
    diameter = keywords.get('diameter')
    if diameter is not None:
        if (isinstance(diameter, (bool, np.bool_))
                or not np.isscalar(diameter)):
            return False
        try:
            if not np.isfinite(float(diameter)) or float(diameter) <= 0:
                return False
        except (TypeError, ValueError):
            return False
    api = _public_trace_api()
    return (api is not None and type(model) is api[0]
            and getattr(model.eval, '__func__', None) is api[1])


def _trace_mask(flow, shape, keywords, device, stream):
    """Construct one mask with the verified public Cellpose tracing call.

    The network has already returned NumPy flows, so the tracing stream has
    no dependency on model tensors or the network stream. Its result is a
    host mask before the future completes.

    :param flow: public eval flow tuple for one image.
    :param shape: original image height and width.
    :param keywords: unchanged public eval options.
    :param device: the same device used by the combined eval path.
    :param stream: independent CUDA stream used only for tracing.
    :returns: the mask returned by public Cellpose dynamics.
    """
    import torch

    diameter = keywords.get('diameter')
    scale = 1.0 if diameter is None else 30.0 / diameter
    niter = keywords.get('niter')
    if niter is None or niter == 0:
        niter = int(200 / scale)
    resize = None if tuple(flow[2].shape) == tuple(shape) else list(shape)
    with torch.cuda.stream(stream):
        return _public_trace_api()[2](
            flow[1], flow[2], niter=niter,
            cellprob_threshold=keywords.get('cellprob_threshold', 0.0),
            flow_threshold=keywords.get('flow_threshold', 0.4),
            min_size=keywords.get('min_size', 15),
            max_size_fraction=keywords.get('max_size_fraction', 0.4),
            resize=resize, device=device)


def _overlap_images(images, evaluate, trace, checkpoint):
    """Keep one tracing future in flight while evaluating the next image.

    Work remains inside the caller-owned batch: no second batch is loaded.
    Results preserve input order, and the executor drains before any error
    reaches the governor, so a retry cannot overlap abandoned work.

    :param images: images already resident in the current batch.
    :param evaluate: public network-and-flow call for one image.
    :param trace: public tracing call taking flows and image shape.
    :param checkpoint: cancellation check before each network call.
    :returns: masks, per-image flows and styles in public list-eval order.
    """
    masks, flows, styles = [], [], []
    pending = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        for image in images:
            checkpoint()
            _empty, flow, style = evaluate(image)
            if pending is not None:
                masks.append(pending.result())
            flows.append(flow)
            styles.append(style)
            pending = executor.submit(trace, flow, image.shape[:2])
        if pending is not None:
            masks.append(pending.result())
    return masks, flows, styles


def _overlap_eval(model, images, **keywords):
    """Evaluate a supported batch with independent public tracing calls.

    The governor owns the model and retries the complete call. CPU fallback,
    unsupported versions, other backends and volumetric calls keep the
    combined public eval path. Cellpose functions and module attributes are
    never replaced.

    :param model: the original Cellpose model governed by the caller.
    :param images: one complete resident batch.
    :param keywords: unchanged public eval options, including tile batch size.
    :returns: the ordinary public masks, flows and styles tuple.
    """
    if not _can_overlap(model, images, keywords):
        return model.eval(images, **keywords)
    import torch
    from .cancellation import checkpoint

    stream = torch.cuda.Stream(device=model.device)
    network_options = {**keywords, 'compute_masks': False}

    def evaluate(image):
        """Return flows from the original public eval on one image."""
        return model.eval(image, **network_options)

    def trace(flow, shape):
        """Trace one image on its independent stream using public dynamics."""
        return _trace_mask(flow, shape, keywords, model.device, stream)

    return _overlap_images(images, evaluate, trace, checkpoint)
