"""Opt-in deterministic GPU mode for every Cellpose and torch segmentation path (item 1047).

Two runs of the same segmentation on a GPU can differ by one object, with
sub-pixel centroid shifts, because some CUDA kernels add numbers in an order
that is not fixed. The ``gpu_deterministic`` setting (default off, so nothing
changes) makes torch pick the fixed-order kernels.

Every model is built by :func:`spacr.utils.build_cellpose_model` and every
evaluation goes through :func:`spacr.object._mask_eval`; both call
:func:`apply_gpu_deterministic` here, so Make Masks, Measure and OPS share
this one switch. ``CUBLAS_WORKSPACE_CONFIG`` is read when CUDA first creates
a cuBLAS handle, so it is set before the model is built; a process that
already ran CUDA matrix products before the mode was switched on should be
restarted for the setting to be complete.
"""
from __future__ import annotations

import os

CUBLAS_WORKSPACE_CONFIG = ":4096:8"

_TRUE = {"1", "true", "yes", "on"}


class GpuDeterminismError(RuntimeError):
    """An operation has no deterministic implementation while ``gpu_deterministic`` is on."""


def gpu_deterministic_on(settings) -> bool:
    """Whether ``settings`` ask for the deterministic GPU mode.

    :param settings: a mapping or None.
    :returns: True for a true boolean or the text true, yes, on or 1.
    """
    value = (settings or {}).get("gpu_deterministic", False)
    if isinstance(value, str):
        return value.strip().lower() in _TRUE
    return bool(value)


def apply_gpu_deterministic(settings) -> bool:
    """Switch torch to deterministic algorithms when the setting asks for it.

    Off does nothing and never undoes an earlier switch. On sets
    ``CUBLAS_WORKSPACE_CONFIG`` (when the caller has not), then
    ``torch.use_deterministic_algorithms(True, warn_only=False)``,
    ``torch.backends.cudnn.deterministic = True`` and
    ``torch.backends.cudnn.benchmark = False``. Repeated calls are cheap.

    :param settings: read for ``gpu_deterministic``.
    :returns: whether the mode is on.
    """
    if not gpu_deterministic_on(settings):
        return False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    import torch

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return True


def explain_nondeterministic(error: BaseException) -> BaseException:
    """The error to raise for a torch refusal of a nondeterministic operation.

    :param error: what the evaluation raised.
    :returns: a :class:`GpuDeterminismError` naming the setting when torch
        refused an operation for lacking a deterministic implementation,
        otherwise ``error`` itself.
    """
    text = str(error)
    if isinstance(error, RuntimeError) and "deterministic" in text.lower():
        return GpuDeterminismError(
            f"gpu_deterministic is on and an operation has no deterministic "
            f"implementation: {text} Turn gpu_deterministic off to run it, "
            f"accepting that two runs can differ by an object.")
    return error
