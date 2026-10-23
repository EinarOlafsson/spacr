"""Keep Cellpose's CPU-only augmentation/AMP assumptions off the Metal path."""

from contextlib import contextmanager, nullcontext
from functools import lru_cache, wraps
from inspect import signature
from threading import RLock

_LOCK = RLock()


@lru_cache(maxsize=1)
def _radeon_vram_bytes():
    """Dedicated Radeon VRAM, or None when the OS cannot report it."""
    import json
    import re
    import subprocess
    import sys

    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        for card in json.loads(result.stdout).get("SPDisplaysDataType", []):
            if "radeon" not in str(card.get("sppci_model", "")).lower():
                continue
            memory = re.search(r"([\d.]+)\s*(GB|MB)", card.get("spdisplays_vram", ""))
            if memory:
                return int(float(memory[1]) * (1024 ** (3 if memory[2] == "GB" else 2)))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def _fit_training_memory(net):
    """Keep jobs whose weights/AdamW state cannot fit off a discrete Radeon."""
    try:
        parameters = list(net.parameters())
        if not parameters or parameters[0].device.type != "mps":
            return
    except (AttributeError, TypeError):
        return
    budget = _radeon_vram_bytes()
    if not budget:
        return
    weights = sum(p.numel() * max(4, p.element_size()) for p in parameters)
    trainable = sum(p.numel() * max(4, p.element_size()) for p in parameters if p.requires_grad)
    minimum = weights + 3 * trainable  # weights, gradients, two AdamW moments
    if minimum <= budget:
        return
    import torch

    print(
        f"Cellpose training needs at least {minimum / 1024**3:.1f} GiB "
        f"for weights/optimizer, exceeding Radeon VRAM ({budget / 1024**3:.1f} GiB). "
        "Training on CPU in float32; smaller or frozen-backbone models can use Metal."
    )
    net.to("cpu")
    torch.mps.empty_cache()


@contextmanager
def training_context(net=None):
    """Augment on CPU, train in float32 on Metal, and restore upstream globals.

    Cellpose 4.2 builds float64 affine tensors and enters unsupported MPS
    autocast. Only its training-module references are adapted; the torch module
    and other application code are untouched. Network forward/backward stays
    on the selected GPU. Older Cellpose releases lacking this seam are retained.
    """
    from .accelerator import resolve

    if resolve().kind != "mps":
        yield
        return
    from cellpose import train

    with _LOCK:
        if net is not None:
            _fit_training_memory(net)
        original_torch = train.torch
        augment = getattr(train, "random_rotate_and_resize", None)
        dynamics = getattr(train, "dynamics", None)

        class TrainingDynamics:
            def __getattr__(self, name):
                return getattr(dynamics, name)

            def labels_to_flows(self, *args, **kwargs):
                # The same Metal indexing defect affects training-label flow
                # generation as inference's flow-error postprocessing.
                function = dynamics.labels_to_flows
                bound = signature(function).bind(*args, **kwargs)
                bound.arguments["device"] = original_torch.device("cpu")
                return function(*bound.args, **bound.kwargs)

        class TrainingTorch:
            def __getattr__(self, name):
                return getattr(original_torch, name)

            def autocast(self, *args, **kwargs):
                device = kwargs.get("device_type", args[0] if args else "")
                return (
                    nullcontext()
                    if device in ("mps", "cpu")
                    else original_torch.autocast(*args, **kwargs)
                )

        if augment is not None:
            parameters = signature(augment)

            @wraps(augment)
            def cpu_augmentation(*args, **kwargs):
                bound = parameters.bind(*args, **kwargs)
                device = bound.arguments.get("device", original_torch.device("cpu"))
                if original_torch.device(device).type != "mps":
                    return augment(*args, **kwargs)
                bound.arguments["device"] = original_torch.device("cpu")
                result = augment(*bound.args, **bound.kwargs)
                return tuple(
                    value.to(device) if hasattr(value, "to") else value for value in result
                )

            train.random_rotate_and_resize = cpu_augmentation
        train.torch = TrainingTorch()
        if dynamics is not None:
            train.dynamics = TrainingDynamics()
        try:
            yield
        finally:
            train.torch = original_torch
            if dynamics is not None:
                train.dynamics = dynamics
            if augment is not None:
                train.random_rotate_and_resize = augment
