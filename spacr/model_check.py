"""Is this model compatible with spaCR, and with the classes chosen?

Clicking a model should say whether it will work before an hour of training
finds out. Three questions, in the order they can fail:

1. Can it be LOADED at all? A custom path may hold a checkpoint saved by a
   different framework, a state dict with no architecture, or a corrupt file.
2. Does it take the INPUT this dataset produces -- the right number of image
   channels, at the chosen size?
3. Does it produce the right number of CLASSES?

The third is the one that silently half-works: a two-class head on a
three-class problem trains happily and is wrong about every object of the
third class.

**A custom model that loads supersedes ``model_type``.** There is no boolean
saying which to believe, because a path that holds a working model is a
complete answer on its own and a flag that disagreed with it would just be a
second thing to get wrong.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

LOG = logging.getLogger("spacr.model_check")


@dataclass(frozen=True)
class ModelReport:
    """What was found. ``ok`` first, because it is the answer.

    :ivar ok: whether the chosen model is compatible with the requested run.
    :ivar source: resolved built-in model name or custom model path.
    """

    ok: bool
    source: str
    #: One line per problem, each naming the setting that would fix it.
    problems: Tuple[str, ...] = ()
    #: One line per thing worth knowing that is not a problem.
    notes: Tuple[str, ...] = ()
    channels: Optional[int] = None
    classes: Optional[int] = None

    def summary(self) -> str:
        if self.ok:
            head = f"{self.source} looks compatible"
            return "; ".join([head, *self.notes]) if self.notes else head
        return f"{self.source} will not work: " + "; ".join(self.problems)


def resolve_model_source(settings: Mapping[str, Any]) -> Tuple[str, str]:
    """``(kind, name)`` for the model that will actually be used.

    :param settings: classification settings containing model choices.

    ``kind`` is ``'custom'`` or ``'builtin'``. A custom path that EXISTS wins:
    the old ``custom_model`` boolean could disagree with the path beside it,
    and then which one won depended on which reader you asked.
    """
    path = str(settings.get("custom_model_path") or "").strip()
    if path and os.path.exists(path):
        return "custom", path
    if path:
        # Named rather than ignored: a path that is set and missing is a
        # mistake, not a preference for the built-in model.
        LOG.info("custom_model_path %r does not exist; using model_type", path)
    return "builtin", str(settings.get("model_type") or "").strip()


def expected_channels(settings: Mapping[str, Any]) -> Optional[int]:
    """How many image channels this dataset will hand the model.

    :param settings: classification settings containing channel declarations.
    """
    for key in ("train_channels", "extract_channels", "channels"):
        value = settings.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return len(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def expected_classes(settings: Mapping[str, Any]) -> Optional[int]:
    """How many classes the training set will have.

    :param settings: classification settings containing class definitions.
    """
    from .classify_classes import class_names

    names = class_names(settings)
    return len(names) if names else None


def _load_custom(path: str):
    """Load a saved model, or raise with what is wrong with the file."""
    import torch

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(
            f"this file cannot be read as a PyTorch model: {exc}") from exc

    if isinstance(loaded, dict):
        # A state dict is weights with no architecture. It can be loaded INTO
        # a model but is not one, and the difference is worth saying plainly
        # rather than failing later with a missing-attribute error.
        if any(k in loaded for k in ("state_dict", "model_state_dict")):
            raise ValueError(
                "this is a checkpoint of weights, not a model; set model_type "
                "to the architecture and use resume_checkpoint to load these "
                "weights into it")
        raise ValueError(
            "this file holds a plain dictionary, not a model; if it is a "
            "state dict, set model_type and use resume_checkpoint")
    if not hasattr(loaded, "forward"):
        raise ValueError(
            f"this file holds a {type(loaded).__name__}, which is not "
            f"something that can be run on an image")
    return loaded


def _head_size(model) -> Optional[int]:
    """How many outputs the model's last layer has, if it can be told.

    Walks to the last layer with an ``out_features``. Returns None rather than
    guessing: a model whose head cannot be read is not necessarily wrong, and
    reporting a made-up number is worse than reporting nothing.
    """
    size = None
    try:
        for module in model.modules():
            out = getattr(module, "out_features", None)
            if out is not None:
                size = int(out)
    except Exception:
        LOG.debug("could not read the model head", exc_info=True)
    return size


def _known_builtin_models() -> set[str]:
    """Return the TorchVision factories this installation can build.

    Importing TorchVision is deferred until the user asks for a compatibility
    check.  If that optional stack is unavailable or broken, the lightweight
    settings inventory still lets the checker reject misspellings without
    making the settings screen import torch during startup.
    """
    from .settings_spec import _TORCHVISION_MODELS_CURATED

    known = set(_TORCHVISION_MODELS_CURATED)
    try:
        from torchvision import models as torchvision_models

        known.update(torchvision_models.list_models(
            module=torchvision_models))
    except (ImportError, AttributeError):
        pass
    except Exception:
        LOG.debug("could not read TorchVision's model registry", exc_info=True)
    return known


def check_model(settings: Mapping[str, Any]) -> ModelReport:
    """Whether the chosen model can train on the chosen data and classes.

    :param settings: classification settings to validate against the model.

    Never raises: this runs from a click, and a dialog that crashes the screen
    is a worse answer than one that says what is wrong.
    """
    kind, name = resolve_model_source(settings)
    if not name:
        return ModelReport(
            ok=False, source="no model",
            problems=("no model is chosen: set model_type, or point "
                      "custom_model_path at a saved model",))

    problems: List[str] = []
    notes: List[str] = []
    classes_problem = False
    try:
        wanted_classes = expected_classes(settings)
    except ValueError as exc:
        # ``class_names`` raises ClassDefinitionError (a ValueError) when the
        # Classes editor has written an incomplete rule.  This checker is a
        # click-time diagnostic, so that user error belongs in its report,
        # not on the Qt event loop as an exception.
        wanted_classes = None
        classes_problem = True
        problems.append(f"the classes setting is invalid: {exc}")
    wanted_channels = expected_channels(settings)
    head: Optional[int] = None

    if kind == "custom":
        notes.append("a custom model was loaded, so model_type is not used")
        try:
            model = _load_custom(name)
        except ValueError as exc:
            problems.append(str(exc))
            return ModelReport(ok=False, source=os.path.basename(name),
                               problems=tuple(problems),
                               classes=wanted_classes,
                               channels=wanted_channels)
        except ImportError:
            problems.append(
                "PyTorch is not installed in this environment, so a saved "
                "model cannot be checked")
            return ModelReport(
                ok=False, source=os.path.basename(name),
                problems=tuple(problems),
                classes=wanted_classes,
                channels=wanted_channels)
        head = _head_size(model)
    else:
        known = _known_builtin_models()
        if name not in known:
            problems.append(
                f"{name!r} is not a model spaCR knows; choose one of "
                f"{', '.join(sorted(known)[:8])}…")

    if wanted_classes is None:
        if not classes_problem:
            problems.append(
                "no classes are defined, so the model's output cannot be "
                "checked; set the Classes dict")
    elif wanted_classes < 2:
        problems.append(
            f"only {wanted_classes} class is defined; a classifier needs two")
    elif head is not None and head != wanted_classes:
        # The failure that silently half-works: a two-class head on a
        # three-class problem trains happily and is wrong about every object
        # of the third class.
        problems.append(
            f"the model has {head} output(s) but {wanted_classes} classes are "
            f"defined; its final layer has to be replaced or the classes "
            f"changed")

    if wanted_channels is None:
        notes.append("no channels are chosen yet, so the input was not checked")
    elif wanted_channels not in (1, 3):
        notes.append(
            f"{wanted_channels} channels: a pretrained backbone expects 3, so "
            f"its first layer will be adapted")

    source = os.path.basename(name) if kind == "custom" else name
    return ModelReport(ok=not problems, source=source,
                       problems=tuple(problems), notes=tuple(notes),
                       channels=wanted_channels, classes=wanted_classes)
