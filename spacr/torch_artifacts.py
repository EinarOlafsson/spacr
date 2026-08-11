"""Versioned, crash-safe PyTorch model artifacts used by :mod:`deep_spacr`.

Older spaCR releases wrote complete ``nn.Module`` objects with ``torch.save``.
Those files remain readable here, but new artifacts store state dictionaries
plus the information needed to reconstruct the model and resume training.
"""

from __future__ import annotations

import os
import platform
import random
import tempfile
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn


ARTIFACT_TYPE = "spacr.torch_model"
ARTIFACT_VERSION = 1


def model_configuration(model: nn.Module) -> dict[str, Any]:
    """Return the constructor information required to rebuild ``model``."""
    return {
        "model_name": getattr(model, "model_name", model.__class__.__name__),
        "num_classes": int(getattr(model, "num_classes", 1)),
        "dropout_rate": getattr(model, "dropout_rate", None),
        "use_checkpoint": bool(getattr(model, "use_checkpoint", False)),
        "image_size": int(getattr(model, "image_size", 224)),
        "multilabel": bool(getattr(model, "multilabel", False)),
    }


def dependency_versions() -> dict[str, str]:
    """Return the runtime versions that materially affect a model artifact."""
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    try:
        import torchvision

        versions["torchvision"] = torchvision.__version__
    except Exception:
        versions["torchvision"] = "unavailable"
    return versions


def capture_rng_state() -> dict[str, Any]:
    """Capture random-generator state needed for deterministic continuation."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore a state returned by :func:`capture_rng_state`."""
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(payload: Any, path: str) -> str:
    """Write ``payload`` beside ``path`` and atomically replace the target."""
    path = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.",
                                     suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def make_model_artifact(
    model: nn.Module,
    *,
    optimizer=None,
    scheduler=None,
    epoch: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    best_metric: float | None = None,
    epochs_without_improvement: int = 0,
    preprocessing: Mapping[str, Any] | None = None,
    classes: list[str] | None = None,
    channels: list[str] | None = None,
    include_rng: bool = True,
    artifact_role: str = "model",
) -> dict[str, Any]:
    """Build the canonical serializable spaCR PyTorch artifact.

    Everything needed to REBUILD the model, not only to load its weights:
    :func:`model_configuration` records the architecture so
    :func:`build_model_from_configuration` can reconstruct it without the
    caller remembering what it was.

    :param model: the module to serialise. Its ``state_dict`` and its
        configuration are both captured.
    :param optimizer: optimiser whose state to store, so training can RESUME
        rather than restart. Silently stored as ``None`` if it has no
        ``state_dict``, which is what makes a plain object safe to pass.
    :param scheduler: learning-rate scheduler, same contract as ``optimizer``.
    :param epoch: epoch this artifact was written at. ``None`` records 0.
    :param metrics: whatever the caller measured, stored verbatim. Not
        interpreted, so the keys are the caller's own.
    :param best_metric: the best value seen so far, for checkpoint selection
        on resume. ``None`` means "no best recorded", which is not the same
        as zero.
    :param epochs_without_improvement: early-stopping counter, carried so a
        resumed run does not forget how close it was to stopping.
    :param preprocessing: the transform the inputs were prepared with.
        Without it a loaded model can be fed differently-normalised images
        and will simply be wrong rather than fail.
    :param classes: class names in OUTPUT-COLUMN order. The order is the
        contract -- a reordered list silently relabels every prediction.
    :param channels: input channel names, in channel order, same contract.
    :param include_rng: capture Python/NumPy/torch RNG state, so a resumed
        run continues the same stream. Turn it off for a smaller artifact
        when exact resumption does not matter.
    :param artifact_role: what this file IS -- ``model`` for a trained
        model, another role for a companion artifact -- recorded so a loader
        can tell them apart.
    :returns: the artifact dict, ready for :func:`atomic_torch_save`.
    """
    optimizer_state = (
        optimizer.state_dict()
        if optimizer is not None and hasattr(optimizer, "state_dict")
        else None
    )
    scheduler_state = (
        scheduler.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict")
        else None
    )
    return {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "artifact_role": str(artifact_role),
        "model_state_dict": model.state_dict(),
        "model_config": model_configuration(model),
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "training_state": {
            "epoch": int(epoch or 0),
            "best_metric": (
                float(best_metric) if best_metric is not None else None
            ),
            "epochs_without_improvement": int(epochs_without_improvement),
        },
        "metrics": dict(metrics or {}),
        "preprocessing": dict(preprocessing or {}),
        "classes": list(classes) if classes is not None else None,
        "channels": list(channels) if channels is not None else None,
        "dependencies": dependency_versions(),
        "rng_state": capture_rng_state() if include_rng else None,
    }


def save_model_artifact(model: nn.Module, path: str, **kwargs) -> str:
    """Build and atomically save a canonical spaCR model artifact."""
    return atomic_torch_save(make_model_artifact(model, **kwargs), path)


def _legacy_configuration(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the pre-versioned ``{'model': state_dict, ...}`` format."""
    return {
        "model_name": payload.get("model_name", "maxvit_t"),
        "num_classes": int(payload.get("num_classes", 2)),
        "dropout_rate": payload.get("dropout_rate"),
        "use_checkpoint": bool(payload.get("use_checkpoint", False)),
        "image_size": int(payload.get("image_size", 224)),
        "multilabel": bool(payload.get("multilabel", False)),
    }


def build_model_from_configuration(config: Mapping[str, Any]) -> nn.Module:
    """Reconstruct a :class:`TorchModel` without downloading pretrained weights."""
    from .utils import TorchModel

    return TorchModel(
        model_name=str(config["model_name"]),
        pretrained=False,
        dropout_rate=config.get("dropout_rate"),
        use_checkpoint=bool(config.get("use_checkpoint", False)),
        num_classes=int(config.get("num_classes", 2)),
        multilabel=bool(config.get("multilabel", False)),
        image_size=int(config.get("image_size", 224)),
    )


def load_model_artifact(
    path: str,
    *,
    map_location: Any = "cpu",
    model: nn.Module | None = None,
    strict: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load current artifacts and legacy full-module/state-dict checkpoints.

    The returned metadata dict always contains ``legacy``. Current artifacts
    retain their optimizer/scheduler/RNG state so callers can resume training.
    """
    raw = torch.load(os.fspath(path), map_location=map_location,
                     weights_only=False)
    if isinstance(raw, nn.Module):
        return raw, {
            "legacy": True,
            "artifact_role": "legacy_full_module",
            "model_config": model_configuration(raw),
            "training_state": {},
        }
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"Unsupported PyTorch artifact at {path!r}: expected an nn.Module "
            "or checkpoint mapping."
        )

    if raw.get("artifact_type") == ARTIFACT_TYPE:
        payload = dict(raw)
        version = int(payload.get("artifact_version", 0))
        if version != ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported spaCR model artifact version {version}; this "
                f"installation supports version {ARTIFACT_VERSION}."
            )
        config = dict(payload.get("model_config") or {})
        state_dict = payload.get("model_state_dict")
        payload["legacy"] = False
    elif "model" in raw and isinstance(raw["model"], Mapping):
        payload = dict(raw)
        config = _legacy_configuration(payload)
        state_dict = payload["model"]
        payload.setdefault("training_state", {})
        payload["model_config"] = config
        payload["legacy"] = True
    elif "state_dict" in raw and isinstance(raw["state_dict"], Mapping):
        payload = dict(raw)
        config = _legacy_configuration(payload)
        state_dict = payload["state_dict"]
        payload.setdefault("training_state", {})
        payload["model_config"] = config
        payload["legacy"] = True
    else:
        raise ValueError(
            f"Unsupported PyTorch checkpoint mapping at {path!r}: no model "
            "state dictionary was found."
        )

    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Model state in {path!r} is not a state dictionary.")
    if model is None:
        if not config.get("model_name"):
            raise ValueError(
                f"Checkpoint {path!r} does not describe its architecture; "
                "provide an initialized model explicitly."
            )
        model = build_model_from_configuration(config)
    model.load_state_dict(state_dict, strict=strict)
    return model, payload


def restore_training_state(
    payload: Mapping[str, Any],
    *,
    optimizer=None,
    scheduler=None,
    restore_random_generators: bool = True,
) -> dict[str, Any]:
    """Restore optimizer/scheduler/RNG state and return training metadata."""
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    if restore_random_generators:
        restore_rng_state(payload.get("rng_state"))
    return dict(payload.get("training_state") or {})
